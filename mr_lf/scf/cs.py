"""Coherent-state Hartree–Fock energies and reference optimization."""

import numpy as np
from scipy import optimize

from .. import _orbitals
from .._driver import Calculation


def cs_site_density(coeff: np.ndarray) -> np.ndarray:
    """Return ``n[x] = |coeff[x]|**2`` for one normalized occupied orbital.

    Parameters
    ----------
    coeff
        Site-basis occupied-orbital coefficients with shape ``(L,)``.  Real
        and complex coefficients must both be supported.
    """
    return np.abs(coeff) ** 2


def cs_stationary_shift(
    coeff: np.ndarray,
    g: float,
    omega: float,
) -> np.ndarray:
    """Return the stationary coherent displacement ``z[x] = -g*n[x]/omega``."""
    return -g * cs_site_density(coeff) / omega


def cs_fock(tmat: np.ndarray, g: float, shift: np.ndarray) -> np.ndarray:
    """Build the one-electron CS-HF Fock matrix ``t + diag(2*g*shift)``."""
    return tmat + np.diag(2 * g * shift)


def cs_energy(
    tmat: np.ndarray,
    g: float,
    omega: float,
    coeff: np.ndarray,
    shift: np.ndarray,
) -> float:
    """Evaluate ``c† t c + omega*z·z + 2*g*z·n``.

    This function receives ``shift`` explicitly so that its three energy
    contributions can be checked independently.  It must return a real Python
    ``float`` for a Hermitian ``tmat``.
    """
    return (
        np.vdot(coeff, tmat @ coeff).real
        + omega * np.dot(shift, shift)
        + 2 * g * np.dot(shift, cs_site_density(coeff))
    )


def cs_hf_scf(
    tmat: np.ndarray,
    g: float,
    omega: float,
    coeff0: np.ndarray,
    *,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
) -> tuple[float, np.ndarray, np.ndarray, bool, int, float]:
    """Solve the one-electron CS-HF fixed-point equations.

    Parameters
    ----------
    tmat
        Hermitian site-basis hopping matrix with shape ``(L, L)``.
    g
        Electron--phonon coupling strength.
    omega
        Phonon frequency entering ``z[x] = -g*n[x]/omega``.
    coeff0
        Initial occupied-orbital coefficients with shape ``(L,)``.  The
        driver normalizes a copy, so the caller's array is not modified.
    conv_tol
        Convergence threshold for the maximum site-density change.
    max_cycle
        Maximum number of undamped fixed-point iterations.

    Returns
    -------
    energy
        CS-HF total energy, not the occupied Fock eigenvalue.
    coeff
        Final normalized site-basis occupied orbital, shape ``(L,)``.
    shift
        Stationary displacement belonging to ``coeff``, shape ``(L,)``.
    converged
        Whether the density residual fell below ``conv_tol``.
    niter
        Number of Fock diagonalizations performed.
    density_residual
        Maximum site-density change in the last iteration.
    """
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("tmat must be a square matrix")
    if coeff0.ndim != 1 or coeff0.shape[0] != tmat.shape[0]:
        raise ValueError("coeff0 must be a vector of the same length as tmat")
    coeff = coeff0.copy()
    # normalize the initial guess
    norm = np.linalg.norm(coeff)
    if norm == 0:
        raise ValueError("Initial guess coeff0 cannot be the zero vector")
    coeff /= norm
    converged = False
    niter = 0
    density_residual = float("inf")
    for cycle in range(max_cycle):
        old_density = cs_site_density(coeff)
        old_shift = cs_stationary_shift(coeff, g, omega)
        fock = cs_fock(tmat, g, old_shift)
        new_coeff = np.linalg.eigh(fock)[1][:, 0]  # lowest eigenvector
        new_density = cs_site_density(new_coeff)

        density_residual = float(np.max(np.abs(new_density - old_density)))

        coeff = new_coeff
        niter += 1

        if density_residual < conv_tol:
            converged = True
            break

    shift = cs_stationary_shift(coeff, g, omega)
    energy = float(cs_energy(tmat, g, omega, coeff, shift))
    return energy, coeff, shift, converged, niter, density_residual


def cs_multi_state(model, mo_coeff):
    """Stationary coherent shift, CS energy and unrestricted Fock matrices.

    mo_coeff: float64 (2,L,L), occupied columns first. Density matrices
    use [spin,site,site]; spin_density uses [spin,site].
    """
    coeff = _orbitals.initial_orbitals(model, mo_coeff)
    density = _orbitals.densities(coeff, model.nelec)
    spin_density = np.diagonal(density, axis1=1, axis2=2).copy()
    rho = spin_density.sum(axis=0)
    shift = -model.g * rho / model.omega
    energy = (
        np.einsum("pq,sqp->", model.tmat, density)
        + model.U * spin_density[0] @ spin_density[1]
        - model.g**2 / model.omega * (rho @ rho)
    )
    fock = np.stack(
        [
            model.tmat + np.diag(model.U * spin_density[1 - s] + 2 * model.g * shift)
            for s in range(2)
        ]
    )
    return float(energy), shift, spin_density, density, fock


def cs_hf_multi_optimize(model, mo_coeff0=None, *, gtol=1e-8, max_cycle=1000):
    """Optimize the CS energy over real occupied/virtual orbital rotations.

    z is eliminated analytically. Convergence requires independent
    stationarity and energy checks; optimizer_success separately reports
    BFGS termination, including possible finite-difference precision loss.
    """
    _orbitals.validate_options(gtol, max_cycle)
    coeff0 = _orbitals.initial_orbitals(model, mo_coeff0)
    nrot = sum(n * (model.L - n) for n in model.nelec)

    def cost(x):
        return cs_multi_state(model, _orbitals.rotate(coeff0, model.nelec, x))[0]

    if nrot:
        raw = optimize.minimize(
            cost,
            np.zeros(nrot),
            method="BFGS",
            jac="3-point",
            options={"gtol": gtol, "maxiter": max_cycle},
        )
        coeff = _orbitals.rotate(coeff0, model.nelec, raw.x)
    else:
        coeff = coeff0
        raw = optimize.OptimizeResult(
            fun=cost(np.zeros(0)), success=True, nit=0, message="No occupied/virtual rotations"
        )
    _energy, shift, spin_density, density, fock = cs_multi_state(model, coeff)
    orbital_residual = float(np.linalg.norm(fock @ density - density @ fock))
    coeff, mo_energy = _orbitals.canonicalize(coeff, fock, model.nelec)
    check = cs_multi_state(model, coeff)[0]
    residual_ok = orbital_residual <= max(1e-7, 10 * gtol)
    consistent = bool(abs(check - raw.fun) <= 1e-9)
    return {
        "energy": check,
        "mo_coeff": coeff,
        "mo_energy": mo_energy,
        "shift": shift,
        "spin_density": spin_density,
        "orbital_grad_norm": orbital_residual,
        "shift_residual": 0.0,
        "optimizer_success": bool(raw.success),
        "residual_ok": residual_ok,
        "energy_consistent": consistent,
        "converged": bool(residual_ok and consistent),
        "niter": int(raw.nit),
        "message": str(raw.message),
    }


class CSHF(Calculation):
    """CS-HF driver. kernel(mo_coeff0=None) returns total energy.

    One electron uses the original density fixed-point solver. Other
    electron sectors use variational occupied/virtual BFGS rotations.
    """

    def __init__(self, model, *, gtol=1e-8, max_cycle=1000):
        super().__init__(model)
        _orbitals.validate_options(gtol, max_cycle)
        self.gtol, self.max_cycle = gtol, max_cycle

    def kernel(self, mo_coeff0=None):
        self._reset()
        model = self.model
        if sum(model.nelec) == 1:
            coeff0 = _orbitals.initial_orbitals(model, mo_coeff0)
            spin = 0 if model.nelec[0] else 1
            energy, c, z, ok, nit, residual = cs_hf_scf(
                model.tmat,
                model.g,
                model.omega,
                coeff0[spin, :, 0],
                conv_tol=min(self.gtol, 1e-10),
                max_cycle=self.max_cycle,
            )
            coeff0[spin] = np.linalg.eigh(cs_fock(model.tmat, model.g, z))[1]
            _, _, sd, density, fock = cs_multi_state(model, coeff0)
            coeff, eps = _orbitals.canonicalize(coeff0, fock, model.nelec)
            grad = float(np.linalg.norm(fock @ density - density @ fock))
            # Keep the legacy energy/state and verify canonicalization did
            # not change the occupied projector at density convergence.
            consistent = abs(cs_multi_state(model, coeff)[0] - energy) <= 1e-9
            result = {
                "energy": energy,
                "coeff": c,
                "mo_coeff": coeff,
                "mo_energy": eps,
                "shift": z,
                "spin_density": sd,
                "niter": nit,
                "density_residual": residual,
                "orbital_grad_norm": grad,
                "shift_residual": float(np.max(np.abs(z + model.g * sd.sum(axis=0) / model.omega))),
                "energy_consistent": consistent,
                "converged": bool(ok and consistent and grad <= 1e-7),
            }
        else:
            result = cs_hf_multi_optimize(
                model, mo_coeff0, gtol=self.gtol, max_cycle=self.max_cycle
            )
        self.mo_coeff, self.mo_energy = result["mo_coeff"], result["mo_energy"]
        self.shift = result["shift"]
        return self._finish(result, result["energy"], result["converged"])

    def site_density(self):
        from .._driver import require_reference

        require_reference(self)
        return self.result["spin_density"].copy()
