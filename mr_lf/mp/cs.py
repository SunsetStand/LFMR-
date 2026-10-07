"""Coherent-state second-order perturbation theory."""

import numpy as np

from .. import _orbitals
from .._driver import Calculation, require_reference
from ..scf.cs import CSHF, cs_fock, cs_hf_scf


def local_ep_coupling_mo(
    coupling: float,
    mo_coeff: np.ndarray,
) -> np.ndarray:
    """Transform local Holstein couplings from the site basis to the MO basis.

    Parameters
    ----------
    coupling
        Real local electron--phonon coupling strength ``g``.
    mo_coeff
        Canonical MO coefficients with shape ``(nsite, nmo)`` and element
        ``mo_coeff[x, p] = C[x, p]``.  The current one-mode-per-site model
        requires a square matrix, so ``nsite == nmo``.

    Returns
    -------
    numpy.ndarray
        Coupling tensor with shape ``(nmode, nmo, nmo)`` and index order
        ``coupling_mo[x, p, q] = g * C[x, p].conj() * C[x, q]``.  Complex
        MO coefficients produce a complex tensor.
    """
    if mo_coeff.ndim != 2 or mo_coeff.shape[1] != mo_coeff.shape[0]:
        raise ValueError("mo_coeff must be a square matrix")
    norb = mo_coeff.shape[0]
    G = np.zeros((norb, norb, norb), dtype=np.result_type(coupling, mo_coeff))
    for x in range(norb):
        for p in range(norb):
            for q in range(norb):
                G[x, p, q] = coupling * mo_coeff[x, p].conj() * mo_coeff[x, q]
    return G


def cs_mp2_mixed_intermediates(
    mo_energy: np.ndarray,
    coupling_mo: np.ndarray,
    omega: float,
    nocc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mixed electronic-single plus one-phonon CS-MP2 intermediates.

    Parameters
    ----------
    mo_energy
        Canonical MO energies with shape ``(nmo,)``.  Occupied orbitals must
        precede virtual orbitals.
    coupling_mo
        MO-basis electron--phonon couplings with shape
        ``(nmode, nmo, nmo)`` and index order ``[x, p, q]``.
    omega
        Phonon frequency used in every mode of the Fig. 2b model.
    nocc
        Number of occupied MOs, satisfying ``1 <= nocc < nmo``.

    Returns
    -------
    matrix_elements
        Excitation matrix elements ``G[x, a, i]`` arranged with shape
        ``(nmode, nocc, nvir)`` and index order ``[x, i, a]``.  The virtual
        subspace index ``a`` corresponds to global MO index ``nocc + a``.
    denominators
        Real zeroth-order denominators ``epsilon_i - epsilon_a - omega``
        with the same shape and index order as ``matrix_elements``.
    """
    if mo_energy.ndim != 1:
        raise ValueError("mo_energy must be a 1D array")
    nmo = mo_energy.shape[0]
    if coupling_mo.ndim != 3 or coupling_mo.shape[1] != nmo or coupling_mo.shape[2] != nmo:
        raise ValueError("coupling_mo must be a 3D array with shape (nmode, nmo, nmo)")
    if nocc < 1 or nocc >= nmo:
        raise ValueError("nocc must be between 1 and nmo-1")
    matrix_elements = coupling_mo[:, nocc:, :nocc]  # shape (nmode, nvir, nocc)
    matrix_elements = matrix_elements.transpose(0, 2, 1)  # shape (nmode, nocc, nvir)
    denominators = mo_energy[:nocc, None] - mo_energy[None, nocc:] - omega
    denominators = np.broadcast_to(
        denominators, (coupling_mo.shape[0], nocc, nmo - nocc)
    )  # shape (nmode, nocc, nvir)
    return matrix_elements, denominators


def cs_mp2_mixed_energy(
    matrix_elements: np.ndarray,
    denominators: np.ndarray,
) -> float:
    """Return the mixed electron--phonon contribution to the CS-MP2 energy.

    Both inputs use shape ``(nmode, nocc, nvir)`` and index order
    ``[x, i, a]``.  The correction is
    ``sum_xia(abs(matrix_elements[x, i, a])**2 / denominators[x, i, a])``.
    Denominators must be real and strictly negative.

    Returns
    -------
    float
        The one-electron CS-MP2 second-order correction.  For the target
        system this is the complete second-order correction because pure
        phonon excitations vanish at the CS stationary point and electronic
        double excitations do not exist.
    """
    if matrix_elements.ndim != 3 or denominators.ndim != 3:
        raise ValueError("matrix_elements and denominators must be 3D arrays")
    if matrix_elements.shape != denominators.shape:
        raise ValueError("matrix_elements and denominators must have the same shape")
    for denominator in denominators.flat:
        if denominator >= 0 or not np.isreal(denominator):
            raise ValueError("Denominator must be negative for stability")
    energy_contributions = (np.abs(matrix_elements) ** 2) / denominators
    return float(np.sum(energy_contributions))


def cs_mp2_point(
    tmat: np.ndarray,
    coupling: float,
    omega: float,
    coeff0: np.ndarray,
    *,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
) -> dict[str, object]:
    """Solve one complete one-electron CS-HF/CS-MP2 parameter point.

    The CS-HF reference is converged first.  Its final Fock matrix is then
    canonicalized, the local coupling is transformed to the MO basis, and
    the mixed electron--phonon second-order correction is evaluated for one
    occupied MO.

    Parameters
    ----------
    tmat
        Hermitian site-basis hopping matrix with shape ``(L, L)``.
    coupling
        Real local electron--phonon coupling strength ``g``.
    omega
        Positive phonon frequency.
    coeff0
        Initial site-basis occupied orbital with shape ``(L,)``.
    conv_tol
        CS-HF density convergence threshold.
    max_cycle
        Maximum number of CS-HF iterations.

    Returns
    -------
    dict
        Auditable result containing ``hf_energy``, ``mp2_correction``,
        ``total_energy``, ``coeff``, ``shift``, ``mo_energy``, ``mo_coeff``,
        ``niter``, and ``density_residual``.  The total energy equals the
        CS-HF total energy plus the second-order correction.

    Raises
    ------
    RuntimeError
        If the CS-HF reference does not converge.
    """
    energy, coeff, shift, converged, niter, density_residual = cs_hf_scf(
        tmat, coupling, omega, coeff0, conv_tol=conv_tol, max_cycle=max_cycle
    )
    if not converged:
        raise RuntimeError("CS-HF SCF did not converge")
    F = cs_fock(tmat, coupling, shift)
    mo_energy, mo_coeff = np.linalg.eigh(F)
    local_coupling_mo = local_ep_coupling_mo(coupling, mo_coeff)
    nocc = 1
    matrix_elements, denominators = cs_mp2_mixed_intermediates(
        mo_energy, local_coupling_mo, omega, nocc
    )
    mp2_energy = cs_mp2_mixed_energy(matrix_elements, denominators)
    Energy_total = energy + mp2_energy
    return {
        "hf_energy": energy,
        "mp2_correction": mp2_energy,
        "total_energy": Energy_total,
        "coeff": coeff,
        "shift": shift,
        "mo_energy": mo_energy,
        "mo_coeff": mo_coeff,
        "niter": niter,
        "density_residual": density_residual,
    }


def cs_mp2_unrestricted_reference_point(
    model, mo_coeff, mo_energy, shift, *, denominator_tol=1e-10
):
    """CS-MP2 mixed one-phonon singles and Hubbard opposite-spin doubles.

    Canonical mo_coeff=(2,L,L), mo_energy=(2,L), shift=(L,).
    E2 uses positive excitation denominators and includes each double once.
    The reference must satisfy both orbital and displacement stationarity.
    """
    from ..scf.cs import cs_multi_state

    c = _orbitals.initial_orbitals(model, mo_coeff)
    eps = np.asarray(mo_energy)
    z = np.asarray(shift)
    if eps.shape != (2, model.L) or eps.dtype != np.float64 or not np.all(np.isfinite(eps)):
        raise ValueError("mo_energy must be finite float64 (2,L)")
    if z.shape != (model.L,) or z.dtype != np.float64 or not np.all(np.isfinite(z)):
        raise ValueError("shift must be finite float64 (L,)")
    if (
        isinstance(denominator_tol, (bool, np.bool_))
        or not np.isfinite(denominator_tol)
        or denominator_tol <= 0
    ):
        raise ValueError("denominator_tol must be positive and finite")
    energy, stationary, _, density, fock = cs_multi_state(model, c)
    if np.max(np.abs(z - stationary), initial=0.0) > 1e-7:
        raise ValueError("coherent shift is not stationary")
    if np.linalg.norm(fock @ density - density @ fock) > 1e-7:
        raise ValueError("CS orbitals are not stationary")
    gaps, transitions = [], []
    mixed = 0.0
    smallest = model.omega
    for spin, n in enumerate(model.nelec):
        fmo = c[spin].T @ fock[spin] @ c[spin]
        if np.max(np.abs(fmo - np.diag(eps[spin])), initial=0.0) > 1e-7:
            raise ValueError("mo_coeff/mo_energy must diagonalize the CS Fock matrix")
        gap = eps[spin, n:][None, :] - eps[spin, :n][:, None]
        if gap.size:
            smallest = min(smallest, float(np.min(gap)))
            if np.min(gap) <= denominator_tol:
                raise ValueError("nonpositive or near-zero electronic denominator")
        transition = c[spin, :, :n, None] * c[spin, :, None, n:]
        mixed -= float(np.sum(model.g**2 * transition**2 / (gap[None] + model.omega)))
        gaps.append(gap)
        transitions.append(transition)
    amplitude = model.U * np.einsum("xia,xjb->iajb", *transitions)
    denom = gaps[0][:, :, None, None] + gaps[1][None, None, :, :]
    if denom.size:
        smallest = min(smallest, float(np.min(denom)))
        if np.min(denom) <= denominator_tol:
            raise ValueError("nonpositive or near-zero double denominator")
    doubles = -float(np.sum(amplitude**2 / denom))
    correction = mixed + doubles
    return {
        "hf_energy": energy,
        "total_energy": energy + correction,
        "mp2_correction": correction,
        "phonon_single_correction": mixed,
        "zero_double_correction": doubles,
        "phonon_pure_correction": 0.0,
        "zero_single_correction": 0.0,
        "minimum_denominator": smallest,
        "max_total": 1,
        "max_excited_modes": 1,
    }


class CSMP2(Calculation):
    def __init__(self, hf, *, denominator_tol=1e-10):
        if not isinstance(hf, CSHF):
            raise TypeError("CSMP2 requires a CSHF reference")
        super().__init__(hf.model)
        self.hf = hf
        self.denominator_tol = denominator_tol

    def kernel(self):
        self._reset()
        require_reference(self.hf)
        result = cs_mp2_unrestricted_reference_point(
            self.model,
            self.hf.mo_coeff,
            self.hf.mo_energy,
            self.hf.shift,
            denominator_tol=self.denominator_tol,
        )
        return self._finish(result, result["total_energy"], True, result["mp2_correction"])
