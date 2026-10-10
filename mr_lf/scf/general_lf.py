"""General density-diagonal LF vacuum integrals and unrestricted optimization.

The formulas follow Cui et al., Eqs. (27)-(36); POLAR's general-coupling
implementation is a reference, not an imported dependency. Full transformed
two-electron tensors are retained, including their reduced permutation symmetry.
"""

import numpy as np
from pyscf import gto, scf
from scipy import optimize

from .. import _orbitals
from ..model import _real_array


def validate_frame(model, lam, shift=None):
    lam = _real_array(lam, "lam")
    if lam.shape != (model.nmode, model.norb):
        raise ValueError("lam must have shape (nmode,norb)")
    shift = np.zeros(model.nmode) if shift is None else _real_array(shift, "shift")
    if shift.shape != (model.nmode,):
        raise ValueError("shift must have shape (nmode,)")
    return lam, shift


def vacuum_integrals(model, lam, shift=None):
    """Return (constant,h_eff,V_eff) for <0|U†HU|0>, without Fock truncation.

    U=exp(sum_x (z_x-sum_p lam[x,p] n_p)(b_x†-b_x)).
    V_eff uses a_p† a_r† a_s a_q and the Hamiltonian's 1/2 factor.
    """
    lam, z = validate_frame(model, lam, shift)
    n = model.norb
    delta = lam[:, :, None] - lam[:, None, :]
    fc = np.exp(-.5 * np.sum(delta**2, axis=0))
    g = model.coupling
    h = (model.h1e + np.einsum("xpq,xpq->pq", g, 2*z[:, None, None]
         - lam[:, :, None] - lam[:, None, :])) * fc
    p = np.arange(n)
    h[p, p] += np.einsum("x,xp->p", model.omega, lam**2 - 2*z[:, None]*lam)
    exponent = np.zeros((n,) * 4)
    for d in delta:
        exponent += (d[:, :, None, None] + d[None, None, :, :])**2
    eri = model.eri * np.exp(-.5 * exponent)
    induced = 2 * np.einsum("x,xp,xr->pr", model.omega, lam, lam)
    mixed = -2 * np.einsum("xpq,xr->pqr", g * fc, lam)
    for r in range(n):
        eri[r, r, p, p] += induced[r]
        eri[:, :, r, r] += mixed[:, :, r]
        eri[r, r, :, :] += mixed[:, :, r]
    constant = model.ecore + np.dot(model.omega, z**2)
    return float(constant), h, eri


def make_uhf(model, constant, h1e, eri):
    """Use PySCF UHF with full custom integrals and unit overlap."""
    mf = scf.uhf.UHF(getattr(model, "mol", None) or gto.M(verbose=0))
    mf.nelec = model.nelec
    mf.verbose = 0
    mf.get_hcore = lambda *args: h1e
    mf.get_ovlp = lambda *args: np.eye(model.norb)
    mf.energy_nuc = lambda *args: constant
    mf._eri = eri
    mf.direct_scf = False
    return mf


def occupations(model):
    return np.array([np.arange(model.norb) < count for count in model.nelec], dtype=float)


def initial_orbitals(model, supplied=None):
    if supplied is not None:
        c = _real_array(supplied, "mo_coeff0")
        if c.shape != (2, model.norb, model.norb):
            raise ValueError("mo_coeff0 must have shape (2,norb,norb)")
        if not all(np.allclose(x.T @ x, np.eye(model.norb), atol=1e-10, rtol=0) for x in c):
            raise ValueError("mo_coeff0 must be orthogonal")
        return c
    c = np.stack([np.linalg.eigh(model.h1e)[1]] * 2)
    if sum(model.nelec):
        mf = make_uhf(model, model.ecore, model.h1e, model.eri)
        mf.conv_tol = 1e-12
        mf.max_cycle = 200
        mf.kernel(dm0=scf.uhf.make_rdm1(c, occupations(model)))
        if not mf.converged:
            raise RuntimeError("initial PySCF electronic UHF did not converge; supply mo_coeff0")
        c = np.asarray(mf.mo_coeff)
    return c


def fixed_reference(model, mo_coeff, lam, shift=None):
    """Evaluate an LF determinant and its Fock using PySCF UHF/J/K."""
    constant, h1e, eri = vacuum_integrals(model, lam, shift)
    mf = make_uhf(model, constant, h1e, eri)
    dm = scf.uhf.make_rdm1(mo_coeff, occupations(model))
    energy = mf.energy_elec(dm=dm)[0] + constant
    fock = np.asarray(mf.get_fock(dm=dm))
    return float(energy), np.asarray(dm), fock


def optimize_reference(model, mo_coeff0, lam0, *, gtol, max_cycle):
    """Joint occupied/virtual and lambda BFGS in the exact z=0 gauge.

    Numerical gradients are used for this first general-integral extension;
    final physical orbital and lambda stationarity are checked separately.
    """
    c0 = initial_orbitals(model, mo_coeff0)
    if lam0 is None:
        dm = scf.uhf.make_rdm1(c0, occupations(model))
        force = np.einsum("xpq,sqp->x", model.coupling, dm)
        row = force / model.omega / max(1, sum(model.nelec))
        lam0 = np.broadcast_to(row[:, None], (model.nmode, model.norb))
    lam0, _ = validate_frame(model, lam0)
    nrot = sum(count * (model.norb - count) for count in model.nelec)
    x0 = np.r_[np.zeros(nrot), lam0.ravel()]

    def unpack(x):
        return (_orbitals.rotate(c0, model.nelec, x[:nrot]),
                x[nrot:].reshape(model.nmode, model.norb))

    def objective(x):
        c, lam = unpack(x)
        return fixed_reference(model, c, lam)[0]

    if x0.size:
        raw = optimize.minimize(objective, x0, method="BFGS", jac="3-point",
                                options={"gtol": gtol, "maxiter": max_cycle})
    else:
        raw = optimize.OptimizeResult(x=x0, fun=objective(x0), success=True,
                                      nit=0, message="no variational parameters")
    c, lam = unpack(raw.x)
    _, dm, fock = fixed_reference(model, c, lam)
    orbital_residual = float(np.linalg.norm(scf.uhf.get_grad(c, occupations(model), fock)))
    lam_gradient = np.zeros_like(lam)
    for index in np.ndindex(lam.shape):
        step = 1e-5 * max(1., abs(lam[index]))
        plus, minus = lam.copy(), lam.copy()
        plus[index] += step
        minus[index] -= step
        lam_gradient[index] = (fixed_reference(model, c, plus)[0]
                               - fixed_reference(model, c, minus)[0]) / (2*step)
    lam_residual = float(np.max(np.abs(lam_gradient), initial=0.))
    eps, c = scf.uhf.canonicalize(None, c, occupations(model), fock=fock)
    c, eps = np.asarray(c), np.asarray(eps)
    check, dm, fock = fixed_reference(model, c, lam)
    energy_consistent = abs(check - raw.fun) < 1e-9
    residual_ok = max(orbital_residual, lam_residual) <= max(1e-7, 10*gtol)
    fc = np.exp(-.5*np.sum((lam[:, :, None]-lam[:, None, :])**2, axis=0))
    occupied = [c[s, :, :count] for s, count in enumerate(model.nelec)]
    spin_square = scf.uhf.spin_square(occupied)[0] if sum(model.nelec) else 0.
    return {
        "energy": check, "mo_coeff": c, "mo_energy": eps, "mo_occ": occupations(model),
        "lam": lam, "shift": np.zeros(model.nmode), "density": dm,
        "physical_density": dm * fc, "spin_density": np.diagonal(dm, axis1=1, axis2=2).copy(),
        "phonon_mean": -lam @ np.diag(dm.sum(axis=0)), "spin_square": float(spin_square),
        "orbital_grad_norm": orbital_residual, "lam_grad_max": lam_residual,
        "optimizer_success": bool(raw.success), "residual_ok": bool(residual_ok),
        "energy_consistent": bool(energy_consistent), "converged": bool(residual_ok and energy_consistent),
        "nit": int(raw.nit), "message": str(raw.message), "gauge": "shift=0",
    }
