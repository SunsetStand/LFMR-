"""Full-matrix variational Lang–Firsov Hartree–Fock."""

import numpy as np
from scipy import linalg as scipy_linalg
from scipy import optimize as scipy_optimize

from .. import _orbitals
from .._driver import Calculation, require_reference
from ..phonon import lf_franck_condon
from .cs import cs_site_density


def lf_stationary_shift(
    coeff: np.ndarray,
    lam: np.ndarray,
    g: float,
    omega: float,
) -> np.ndarray:
    """Return the stationary coherent displacement for one electron.

    Parameters
    ----------
    coeff
        Normalized site-basis orbital with shape ``(norb,)``.  Real and
        complex coefficients are both supported.
    lam
        Real conditional displacements with shape ``(norb, norb)`` and
        index order ``lam[x, p]``.  This local Holstein specialization
        requires one phonon mode per electronic site.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.

    Returns
    -------
    numpy.ndarray
        Real displacement ``lam @ density - (g / omega) * density`` with
        shape ``(norb,)``.
    """
    if coeff.ndim != 1:
        raise ValueError("Coefficient array must be 1-dimensional.")
    if lam.ndim != 2:
        raise ValueError("Displacement array must be 2-dimensional.")
    if lam.shape != (coeff.size, coeff.size):
        raise ValueError("Displacement array dimensions must match coefficient array length.")
    if omega <= 0:
        raise ValueError("Frequency omega must be positive.")
    density = cs_site_density(coeff)
    conditional_shift = np.dot(lam, density)
    shift = conditional_shift - (g / omega) * density
    return shift


def lf_effective_one_body(
    tmat: np.ndarray,
    g: float,
    omega: float,
    shift: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    """Build the one-electron zero-phonon LF effective Hamiltonian.

    Parameters
    ----------
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    shift
        Real coherent displacement with shape ``(norb,)``.
    lam
        Real conditional displacements with shape ``(norb, norb)`` and
        index order ``lam[x, p]``.

    Returns
    -------
    numpy.ndarray
        Effective one-body matrix ``tmat * overlap + diag(correction)`` with
        shape ``(norb, norb)``.  The global constant ``omega * shift @ shift``
        is not included.
    """
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    if lam.shape != tmat.shape:
        raise ValueError("Displacement array dimensions must match hopping matrix dimensions.")
    if shift.ndim != 1 or shift.size != tmat.shape[0]:
        raise ValueError("Shift array must be 1-dimensional and match hopping matrix size.")
    if omega <= 0:
        raise ValueError("Frequency omega must be positive.")
    norb = tmat.shape[0]
    overlap = lf_franck_condon(lam)
    diagonal_correction = np.zeros(norb)
    for p in range(norb):
        phonon_part = omega * np.sum(lam[:, p] ** 2 - 2 * shift * lam[:, p])
        coupling_part = 2 * g * (shift[p] - lam[p, p])
        diagonal_correction[p] = phonon_part + coupling_part
    heff = tmat * overlap + np.diag(diagonal_correction)
    return heff


def lf_energy(
    tmat: np.ndarray,
    g: float,
    omega: float,
    coeff: np.ndarray,
    shift: np.ndarray,
    lam: np.ndarray,
) -> float:
    """Evaluate the one-electron zero-phonon LF total energy.

    Parameters
    ----------
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    coeff
        Normalized site-basis orbital with shape ``(norb,)``.  Real and
        complex coefficients are both supported.
    shift
        Real coherent displacement with shape ``(norb,)``.
    lam
        Real conditional displacements with shape ``(norb, norb)`` and
        index order ``lam[x, p]``.

    Returns
    -------
    float
        Total energy ``omega * shift @ shift + coeff.conj() @ heff @ coeff``.
    """
    if coeff.ndim != 1:
        raise ValueError("Coefficient array must be 1-dimensional.")
    heff = lf_effective_one_body(tmat, g, omega, shift, lam)
    if coeff.size != heff.shape[0]:
        raise ValueError("Coefficient array length must match effective Hamiltonian size.")
    electronic_energy = np.real(np.vdot(coeff, heff @ coeff))
    boson_constant = omega * np.dot(shift, shift)
    total_energy = electronic_energy + boson_constant
    return float(total_energy)


def lf_hf_full_state(
    lam: np.ndarray,
    shift: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
) -> tuple[float, np.ndarray, float]:
    """Evaluate a one-electron LF-HF state for a full LF parameter matrix.

    Parameters
    ----------
    lam
        Real conditional displacements with shape ``(nmode, norb)`` and
        index order ``lam[x, p]``.  The local Holstein model has one phonon
        mode per electronic site and therefore uses shape ``(L, L)``.
    shift
        Real coherent displacements ``z[x]`` with shape ``(nmode,)``.
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Real local electron--phonon coupling strength.
    omega
        Positive phonon frequency.

    Returns
    -------
    total_energy
        LF-HF total energy including the global boson contribution
        ``omega * shift @ shift``.
    coeff
        Normalized lowest eigenvector of the zero-phonon effective
        one-electron Hamiltonian, with shape ``(norb,)``.
    orbital_energy
        Lowest eigenvalue of the effective one-electron Hamiltonian before
        adding the global boson contribution.

    Notes
    -----
    For one electron the simultaneous row shift
    ``lam[x, p] -> lam[x, p] + c[x]`` and ``shift[x] -> shift[x] + c[x]``
    leaves the total energy and electronic density invariant.
    """
    heff = lf_effective_one_body(tmat, g, omega, shift, lam)
    eigvals, eigvecs = np.linalg.eigh(heff)
    orbital_energy = eigvals[0]
    coeff = eigvecs[:, 0]
    total_energy = omega * shift @ shift + orbital_energy
    return float(total_energy), coeff, float(orbital_energy)


def lf_hf_full_optimize(
    mu0: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
    *,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> scipy_optimize.OptimizeResult:
    """Optimize a one-electron full LF-HF reference.

    Parameters
    ----------
    mu0
        Finite real initial parameters with shape ``(nmode, norb)`` and
        index order ``mu0[x, p]``.  The input array is not modified.
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    gtol
        Positive gradient-norm tolerance passed to BFGS.
    max_cycle
        Positive maximum number of BFGS iterations.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The BFGS result augmented with ``coeff``, ``orbital_energy``, and
        ``shift_residual``.  Its ``fun`` field is recomputed from the final
        parameters, and ``shift_residual`` is the infinity norm of the
        coherent-shift stationarity equation.
    """
    if mu0.ndim != 2 or mu0.shape[0] != mu0.shape[1]:
        raise ValueError("Initial parameter array must be 2-dimensional.")
    for mu_value in mu0.flat:
        if not np.isfinite(mu_value) or not np.isreal(mu_value):
            raise ValueError("Initial parameter array must contain finite real values.")
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    if mu0.shape[0] != tmat.shape[0]:
        raise ValueError("Initial parameter array first dimension must match hopping matrix size.")
    if gtol <= 0:
        raise ValueError("Gradient tolerance must be positive.")
    if max_cycle <= 0:
        raise ValueError("Maximum cycle count must be positive.")
    norb = tmat.shape[0]
    x0 = np.asarray(mu0, dtype=np.float64, copy=True).ravel().copy()

    def cost(flat_mu: np.ndarray) -> float:
        mu = flat_mu.reshape((norb, norb))
        shift = np.zeros(norb)
        return lf_hf_full_state(mu, shift, tmat, g, omega)[0]

    result = scipy_optimize.minimize(
        cost,
        x0,
        method="BFGS",
        jac="3-point",
        options={"gtol": gtol, "maxiter": max_cycle, "disp": False},
    )
    final_shift = np.zeros(norb)
    final_mu = result.x.reshape((norb, norb)).copy()
    final_energy, final_coeff, final_orbital_energy = lf_hf_full_state(
        final_mu, final_shift, tmat, g, omega
    )
    stationary_shift = lf_stationary_shift(final_coeff, final_mu, g, omega)
    shift_residual = float(np.max(np.abs(final_shift - stationary_shift)))
    result["mu"] = final_mu
    result["lam"] = final_mu.copy()
    result["shift"] = final_shift
    result["coeff"] = final_coeff
    result["orbital_energy"] = final_orbital_energy
    result["shift_residual"] = shift_residual
    result["fun"] = final_energy
    return result


def lf_hf_full_multistart(
    mu0s: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
    *,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> tuple[
    scipy_optimize.OptimizeResult,
    list[scipy_optimize.OptimizeResult],
]:
    """Optimize full-matrix one-electron LF-HF from explicit starts.

    Parameters
    ----------
    mu0s
        Finite real gauge-fixed displacements with shape
        ``(nstart, norb, norb)`` and index order ``[start, x, p]``.  At
        least one start is required, and the input array is not modified.
    tmat
        Square site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    gtol
        Positive gradient-norm tolerance passed to every optimization.
    max_cycle
        Positive maximum number of BFGS iterations per starting point.

    Returns
    -------
    best
        Lowest-energy result among the optimizations for which
        ``result.success`` is true.  This is the same object as one entry
        in ``results``.
    results
        Optimization results in the same order as the supplied starts.

    Raises
    ------
    RuntimeError
        If every optimization reports failure.
    """
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    if mu0s.ndim != 3 or mu0s.shape[0] < 1 or mu0s.shape[1:] != tmat.shape:
        raise ValueError(
            "Initial parameter array must be 3-dimensional with shape (nstart, norb, norb)."
        )
    nstart = mu0s.shape[0]
    for mu in mu0s:
        if not np.all(np.isfinite(mu)) or not np.all(np.isreal(mu)):
            raise ValueError("Initial parameter array must contain finite real values.")
    results = []
    for k in range(nstart):
        result = lf_hf_full_optimize(mu0s[k], tmat, g, omega, gtol=gtol, max_cycle=max_cycle)
        results.append(result)
    successful_results = [res for res in results if res.success]
    if not successful_results:
        raise RuntimeError("All optimization attempts failed.")
    best = min(successful_results, key=lambda res: res.fun)
    return best, results


def lf_hf_full_initial_guesses(
    norb: int,
    g: float,
    omega: float,
    *,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Generate reproducible non-Gaussian starts for full-matrix LF-HF.

    The optimized variable is the gauge-fixed displacement
    ``mu[x, p] = lam[x, p] - shift[x]`` with ``shift = 0``.  Starts are
    returned in a fixed order: the coherent-state point, one localized
    polaron point for every site, and uniformly distributed perturbations
    around the coherent-state point.

    Parameters
    ----------
    norb
        Positive number of electronic sites/orbitals.  The local Holstein
        model has the same number of phonon modes.
    g
        Finite real electron--phonon coupling strength.
    omega
        Finite positive phonon frequency.
    nrandom
        Nonnegative number of uniform random starts appended after the
        deterministic starts.
    random_scale
        Finite positive dimensionless half-width of the uniform
        perturbations, whose dimensional scale is ``abs(g) / omega``.
    seed
        Seed passed to an independent :class:`numpy.random.Generator`.

    Returns
    -------
    numpy.ndarray
        Initial gauge-fixed displacements with shape
        ``(1 + norb + nrandom, norb, norb)``, ``float64`` dtype, and index
        order ``[start, x, p]``.  Row zero is the coherent-state point
        ``g / (omega * norb)``.  Row ``1 + s`` is zero except for
        ``mu[s, s] = g / omega``.
    """
    if not isinstance(norb, (int, np.integer)) or isinstance(norb, (bool, np.bool_)) or norb <= 0:
        raise ValueError("Number of orbitals must be a positive integer.")
    if (
        not isinstance(nrandom, (int, np.integer))
        or isinstance(nrandom, (bool, np.bool_))
        or nrandom < 0
    ):
        raise ValueError("Number of random guesses must be a non-negative integer.")
    if not np.isfinite(g) or not np.isreal(g):
        raise ValueError("Electron-phonon coupling g must be finite and real.")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("Phonon frequency omega must be finite, real, and positive.")
    if not np.isfinite(random_scale) or not np.isreal(random_scale) or random_scale <= 0:
        raise ValueError("Random scale must be finite, real, and positive.")
    mu_cs = np.full((norb, norb), g / (omega * norb), dtype=np.float64)
    guesses = np.empty((1 + norb + nrandom, norb, norb), dtype=np.float64)
    guesses[0] = mu_cs
    for s in range(norb):
        mu_s = np.zeros((norb, norb), dtype=np.float64)
        mu_s[s, s] = g / omega
        guesses[s + 1] = mu_s
    rng = np.random.default_rng(seed)
    for i in range(nrandom):
        perturbation = rng.uniform(-random_scale, random_scale, size=(norb, norb))
        guesses[norb + 1 + i] = mu_cs + (abs(g) / omega) * perturbation
    return guesses


def lf_hf_multi_fixed_energy(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    nelec: tuple[int, int],
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Evaluate a fixed unrestricted LF-HF reference in the site basis.

    The local Hubbard--Holstein coupling is ``g * n[x] * (b[x] + b[x]^dag)``
    without a density offset. There is one phonon mode per site. This
    function evaluates supplied orbitals and full LF parameters; it does
    not optimize them.

    Parameters
    ----------
    tmat
        Real symmetric site-basis hopping matrix ``tmat[p, q]`` with shape
        ``(L, L)`` and dtype ``float64``.
    U, g, omega
        Finite real on-site repulsion, local electron--phonon coupling,
        and positive phonon frequency, respectively.
    nelec
        Fixed electron counts ``(neleca, nelecb)``, each between 0 and L.
    occ_a, occ_b
        Real orthonormal occupied-orbital columns ``occ_sigma[p, i]`` in
        the site basis, with shapes ``(L, neleca)`` and ``(L, nelecb)``
        and dtype ``float64``.
    lam
        Full real LF displacement matrix ``lam[x, p]``, with mode row x
        and site column p; shape ``(L, L)``, dtype ``float64``.
    shift
        Real coherent displacement ``shift[x]``; shape ``(L,)``, dtype
        ``float64``.

    Returns
    -------
    energy
        Total LF-HF expectation value for this fixed reference.
    spin_density
        Site density with shape ``(2, L)`` and index order
        ``spin_density[spin, p]``; spin 0 is alpha and spin 1 beta.

    Notes
    -----
    At fixed positive total electron number, adding the same ``c[x]`` to
    every ``lam[x, p]`` and ``N_e * c[x]`` to ``shift[x]`` leaves the
    represented state invariant. Fixing ``shift=0`` is one possible gauge
    for the full LF matrix.
    """
    if (
        not isinstance(tmat, np.ndarray)
        or tmat.ndim != 2
        or tmat.shape[0] == 0
        or tmat.shape[0] != tmat.shape[1]
    ):
        raise ValueError("tmat must have nonempty square shape (L, L)")
    L = tmat.shape[0]
    if (
        not isinstance(nelec, tuple)
        or len(nelec) != 2
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 0
            or n > L
            for n in nelec
        )
    ):
        raise ValueError("nelec must be a tuple (neleca, nelecb) with 0 <= n <= L")
    for name, array, shape in (
        ("tmat", tmat, (L, L)),
        ("occ_a", occ_a, (L, nelec[0])),
        ("occ_b", occ_b, (L, nelec[1])),
        ("lam", lam, (L, L)),
        ("shift", shift, (L,)),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != np.float64
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(f"{name} must be a finite float64 array with shape {shape}")
    if not np.allclose(tmat, tmat.T, rtol=0.0, atol=1e-12):
        raise ValueError("tmat must be symmetric")
    for name, occupied in (("occ_a", occ_a), ("occ_b", occ_b)):
        if not np.allclose(occupied.T @ occupied, np.eye(occupied.shape[1]), rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} columns must be orthonormal")
    for name, value in (("U", U), ("g", g), ("omega", omega)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite real scalar")
    if omega <= 0:
        raise ValueError("omega must be positive")
    D_a = occ_a @ occ_a.T
    D_b = occ_b @ occ_b.T
    rho_a = np.diag(D_a)
    rho_b = np.diag(D_b)
    rho = rho_a + rho_b
    rho_joint = np.stack((rho_a, rho_b), axis=0)
    n_orb = tmat.shape[0]
    Q = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            Q[p, q] = (
                rho[p] * rho[q]
                - D_a[p, q] * D_a[q, p]
                - D_b[p, q] * D_b[q, p]
                + (rho[p] if p == q else 0)
            )
    S = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            S[p, q] = np.exp(-0.5 * np.sum((lam[:, p] - lam[:, q]) ** 2))
    E_t = np.sum(tmat * S * (D_a + D_b).T)
    E_U = U * np.sum(rho_a * rho_b)
    E_ph = omega * (shift.T @ shift - 2 * shift.T @ lam @ rho + np.trace(lam @ Q @ lam.T))
    E_ep = 2 * g * np.sum(shift * rho - np.sum(lam * Q, axis=1))
    E = E_t + E_U + E_ph + E_ep
    return E, rho_joint


def lf_hf_multi_fixed_fock(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    nelec: tuple[int, int],
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build spin-resolved site-basis Fock matrices for a fixed LF reference.

    This is the derivative of ``lf_hf_multi_fixed_energy`` with respect to
    each spin density matrix while ``lam`` and ``shift`` remain fixed. It
    uses the paper's uncentered local Hubbard--Holstein coupling and one
    phonon mode per site. It does not optimize the occupied orbitals.

    Parameters
    ----------
    tmat
        Real symmetric site-basis hopping matrix ``tmat[p, q]``, shape
        ``(L, L)``, dtype ``float64``.
    U, g, omega
        Finite real on-site repulsion and local coupling, followed by a
        positive phonon frequency.
    nelec
        Fixed counts ``(neleca, nelecb)``, each between 0 and L.
    occ_a, occ_b
        Real orthonormal occupied-orbital columns ``occ_sigma[p, i]`` in
        the site basis; shapes ``(L, neleca)`` and ``(L, nelecb)`` and
        dtype ``float64``.
    lam
        Full real LF matrix ``lam[x, p]`` with mode row and site column;
        shape ``(L, L)``, dtype ``float64``.
    shift
        Real coherent displacement ``shift[x]``, shape ``(L,)``, dtype
        ``float64``.

    Returns
    -------
    F_a, F_b
        Real symmetric Fock matrices for alpha and beta electrons, each
        with shape ``(L, L)`` and site index order ``[p, q]``.
    """
    if (
        not isinstance(tmat, np.ndarray)
        or tmat.ndim != 2
        or tmat.shape[0] == 0
        or tmat.shape[0] != tmat.shape[1]
    ):
        raise ValueError("tmat must have nonempty square shape (L, L)")
    L = tmat.shape[0]
    if (
        not isinstance(nelec, tuple)
        or len(nelec) != 2
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 0
            or n > L
            for n in nelec
        )
    ):
        raise ValueError("nelec must be a tuple (neleca, nelecb) with 0 <= n <= L")
    for name, array, shape in (
        ("tmat", tmat, (L, L)),
        ("occ_a", occ_a, (L, nelec[0])),
        ("occ_b", occ_b, (L, nelec[1])),
        ("lam", lam, (L, L)),
        ("shift", shift, (L,)),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != np.float64
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(f"{name} must be a finite float64 array with shape {shape}")
    if not np.allclose(tmat, tmat.T, rtol=0.0, atol=1e-12):
        raise ValueError("tmat must be symmetric")
    for name, occupied in (("occ_a", occ_a), ("occ_b", occ_b)):
        if not np.allclose(occupied.T @ occupied, np.eye(occupied.shape[1]), rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} columns must be orthonormal")
    for name, value in (("U", U), ("g", g), ("omega", omega)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite real scalar")
    if omega <= 0:
        raise ValueError("omega must be positive")
    D_a = occ_a @ occ_a.T
    D_b = occ_b @ occ_b.T
    rho = np.diag(D_a) + np.diag(D_b)
    G = lam.T @ lam
    d = np.diag(G)
    S = np.exp(-0.5 * (d[:, None] + d[None, :] - 2 * G))
    h = tmat * S
    A = omega * G - g * (lam + lam.T)
    v = 2 * g * shift - 2 * omega * lam.T @ shift
    F_a = h + np.diag(v + np.diag(A) + 2 * A @ rho + U * np.diag(D_b)) - 2 * A * D_a
    F_b = h + np.diag(v + np.diag(A) + 2 * A @ rho + U * np.diag(D_a)) - 2 * A * D_b
    return F_a, F_b


def lf_hf_multi_optimize(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    nelec: tuple[int, int],
    mo_coeff_a0: np.ndarray,
    mo_coeff_b0: np.ndarray,
    lam0: np.ndarray,
    *,
    gtol: float = 1e-6,
    max_cycle: int = 1000,
) -> scipy_optimize.OptimizeResult:
    """Jointly optimize unrestricted orbitals and the full LF matrix.

    This is the unrestricted LF-HF variational problem. The coherent shift is
    fixed to zero. Occupied orbitals are the first ``nelec[spin]`` columns
    of each full site-basis orbital matrix. Neither the input orbitals nor
    ``lam0`` are modified.

    Parameters
    ----------
    tmat
        Real symmetric site-basis hopping matrix, shape ``(L, L)``,
        dtype ``float64``.
    U, g, omega
        Finite real Hubbard repulsion and local coupling, followed by a
        positive phonon frequency.
    nelec
        Electron counts ``(neleca, nelecb)``, each between zero and L.
    mo_coeff_a0, mo_coeff_b0
        Initial full real orthogonal orbital matrices, each with shape
        ``(L, L)`` and dtype ``float64``; row is site, column is MO.
    lam0
        Initial full LF matrix ``lam0[x, p]``, shape ``(L, L)`` and
        dtype ``float64``; x is phonon mode, p is electronic site.
    gtol
        Positive BFGS gradient tolerance for the joint parameter vector.
    max_cycle
        Positive maximum number of BFGS iterations per start.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``fun`` is the variational LF-HF energy. ``mo_coeff`` and
        ``mo_energy`` have shapes ``(2, L, L)`` and ``(2, L)`` with alpha
        before beta; occupied MOs precede virtual MOs. ``mo_occ`` has
        shape ``(2, L)`` and contains zeros and ones. ``lam`` has shape
        ``(L, L)``, ``shift`` is the zero vector of length L, and
        ``spin_density`` has shape ``(2, L)``. Also return ``nit``,
        ``orbital_grad_norm``, ``lam_grad_max``, ``success``, and ``message``.
        A numerical optimizer status alone does not establish success:
        the final physical orbital and LF parameter residuals must pass.
    """
    if (
        not isinstance(tmat, np.ndarray)
        or tmat.ndim != 2
        or tmat.shape[0] == 0
        or tmat.shape[0] != tmat.shape[1]
    ):
        raise ValueError("tmat must have nonempty square shape (L, L)")
    L = tmat.shape[0]
    if (
        not isinstance(nelec, tuple)
        or len(nelec) != 2
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 0
            or n > L
            for n in nelec
        )
    ):
        raise ValueError("nelec must be a tuple (neleca, nelecb) with 0 <= n <= L")
    for name, coeff in (("mo_coeff_a0", mo_coeff_a0), ("mo_coeff_b0", mo_coeff_b0)):
        if (
            not isinstance(coeff, np.ndarray)
            or coeff.shape != (L, L)
            or coeff.dtype != np.float64
            or not np.all(np.isfinite(coeff))
        ):
            raise ValueError(f"{name} must be a finite float64 array with shape {(L, L)}")
        if not np.allclose(coeff.T @ coeff, np.eye(L), rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} columns must be orthonormal")
    # Reuse the accepted fixed-reference checks for model data and lam0.
    lf_hf_multi_fixed_fock(
        tmat,
        U,
        g,
        omega,
        nelec,
        mo_coeff_a0[:, : nelec[0]],
        mo_coeff_b0[:, : nelec[1]],
        lam0,
        np.zeros(L, dtype=np.float64),
    )
    if (
        isinstance(gtol, (bool, np.bool_))
        or not isinstance(gtol, (int, float, np.integer, np.floating))
        or not np.isfinite(gtol)
        or gtol <= 0
    ):
        raise ValueError("gtol must be a finite positive real scalar")
    if (
        isinstance(max_cycle, (bool, np.bool_))
        or not isinstance(max_cycle, (int, np.integer))
        or max_cycle <= 0
    ):
        raise ValueError("max_cycle must be a positive integer")
    na, nb = nelec
    nrot_a = na * (L - na)
    nrot_b = nb * (L - nb)
    x0 = np.r_[np.zeros(nrot_a + nrot_b), lam0.ravel()]

    def unpack(x):
        kappa_a = x[:nrot_a].reshape(L - na, na)
        kappa_b = x[nrot_a : nrot_a + nrot_b].reshape(L - nb, nb)
        lam = x[nrot_a + nrot_b :].reshape(L, L)

        K_a = np.zeros((L, L), dtype=np.float64)
        K_b = np.zeros((L, L), dtype=np.float64)
        K_a[na:, :na] = kappa_a
        K_a[:na, na:] = -kappa_a.T
        K_b[nb:, :nb] = kappa_b
        K_b[:nb, nb:] = -kappa_b.T

        # A line search can sample very large rotations. Polar projection
        # removes accumulated matrix-exponential roundoff without changing
        # the exact orthogonal rotation represented by exp(K).
        rotation_a, _ = scipy_linalg.polar(scipy_linalg.expm(K_a))
        rotation_b, _ = scipy_linalg.polar(scipy_linalg.expm(K_b))
        C_a = mo_coeff_a0 @ rotation_a
        C_b = mo_coeff_b0 @ rotation_b
        return C_a, C_b, lam

    shift = np.zeros(L, dtype=np.float64)

    def cost(x):
        try:
            C_a, C_b, lam = unpack(x)
            energy, _ = lf_hf_multi_fixed_energy(
                tmat,
                U,
                g,
                omega,
                nelec,
                C_a[:, :na],
                C_b[:, :nb],
                lam,
                shift,
            )
        except (ValueError, FloatingPointError):
            return 1e100
        return energy if np.isfinite(energy) else 1e100

    result = scipy_optimize.minimize(
        cost,
        x0,
        method="BFGS",
        jac="3-point",
        options={"gtol": gtol, "maxiter": max_cycle},
    )
    C_a, C_b, lam = unpack(result.x)
    _energy, spin_density = lf_hf_multi_fixed_energy(
        tmat,
        U,
        g,
        omega,
        nelec,
        C_a[:, :na],
        C_b[:, :nb],
        lam,
        shift,
    )
    F_a, F_b = lf_hf_multi_fixed_fock(
        tmat,
        U,
        g,
        omega,
        nelec,
        C_a[:, :na],
        C_b[:, :nb],
        lam,
        shift,
    )
    D_a = C_a[:, :na] @ C_a[:, :na].T
    D_b = C_b[:, :nb] @ C_b[:, :nb].T
    orbital_grad_norm = np.hypot(
        np.linalg.norm(F_a @ D_a - D_a @ F_a),
        np.linalg.norm(F_b @ D_b - D_b @ F_b),
    )

    rho = np.diag(D_a + D_b)
    Q = np.outer(rho, rho) - D_a * D_a - D_b * D_b + np.diag(rho)
    G = lam.T @ lam
    diag_G = np.diag(G)
    S = np.exp(-0.5 * (diag_G[:, None] + diag_G[None, :] - 2 * G))
    W = tmat * S * (D_a + D_b)
    lam_gradient = -2 * (lam * W.sum(axis=1)[None, :] - lam @ W) + 2 * omega * lam @ Q - 2 * g * Q
    lam_grad_max = float(np.max(np.abs(lam_gradient)))

    def canonicalize(coeff, fock, nocc):
        fock_mo = coeff.T @ fock @ coeff
        occupied_energy, occupied_rotation = np.linalg.eigh(fock_mo[:nocc, :nocc])
        virtual_energy, virtual_rotation = np.linalg.eigh(fock_mo[nocc:, nocc:])
        rotation = scipy_linalg.block_diag(occupied_rotation, virtual_rotation)
        return coeff @ rotation, np.r_[occupied_energy, virtual_energy]

    C_a, eps_a = canonicalize(C_a, F_a, na)
    C_b, eps_b = canonicalize(C_b, F_b, nb)
    energy_check, spin_density_check = lf_hf_multi_fixed_energy(
        tmat,
        U,
        g,
        omega,
        nelec,
        C_a[:, :na],
        C_b[:, :nb],
        lam,
        shift,
    )
    energy_consistent = bool(
        np.isfinite(energy_check)
        and abs(energy_check - result.fun) <= 1e-9
        and np.allclose(spin_density_check, spin_density, atol=1e-10, rtol=0)
    )
    residual_ok = bool(
        np.isfinite(orbital_grad_norm)
        and np.isfinite(lam_grad_max)
        and orbital_grad_norm <= 1e-5
        and lam_grad_max <= 1e-5
    )
    result.optimizer_success = bool(result.success)
    result.residual_ok = residual_ok
    result.energy_consistent = energy_consistent
    result.success = bool(result.optimizer_success and residual_ok and energy_consistent)
    result.mo_coeff = np.stack((C_a, C_b))
    result.mo_energy = np.stack((eps_a, eps_b))
    result.mo_occ = np.stack((np.arange(L) < na, np.arange(L) < nb)).astype(np.float64)
    result.lam = lam.copy()
    result.shift = shift.copy()
    result.spin_density = spin_density_check
    result.orbital_grad_norm = float(orbital_grad_norm)
    result.lam_grad_max = lam_grad_max
    return result


class LFHF(Calculation):
    """One-start, full-matrix LF-HF in the gauge shift=0.

    kernel(mo_coeff0=None, lam0=None) returns the variational total energy.
    Multi-start selection is explicit in examples, not hidden in this driver.
    """

    def __init__(self, model, *, gtol=None, max_cycle=None):
        super().__init__(model)
        single = sum(model.nelec) == 1
        self.gtol = (1e-8 if single else 1e-6) if gtol is None else gtol
        self.max_cycle = (500 if single else 1000) if max_cycle is None else max_cycle
        _orbitals.validate_options(self.gtol, self.max_cycle)

    def kernel(self, mo_coeff0=None, lam0=None):
        self._reset()
        m = self.model
        coeff0 = _orbitals.initial_orbitals(m, mo_coeff0)
        if lam0 is None:
            lam0 = np.full((m.L, m.L), m.g / (m.omega * m.L), dtype=np.float64)
        lam0 = np.asarray(lam0)
        if lam0.shape != (m.L, m.L) or lam0.dtype != np.float64 or not np.all(np.isfinite(lam0)):
            raise ValueError("lam0 must be finite float64 (L,L)")
        if sum(m.nelec) == 1:
            raw = lf_hf_full_optimize(
                lam0, m.tmat, m.g, m.omega, gtol=self.gtol, max_cycle=self.max_cycle
            )
            eps, coeff = np.linalg.eigh(
                lf_effective_one_body(m.tmat, m.g, m.omega, raw.shift, raw.lam)
            )
            spin = 0 if m.nelec[0] else 1
            orbitals = coeff0.copy()
            orbitals[spin] = coeff
            energies = np.stack((eps, eps))
            density = np.zeros((2, m.L))
            density[spin] = abs(coeff[:, 0]) ** 2
            result = dict(raw)
            result.update(
                energy=float(raw.fun),
                mo_coeff=orbitals,
                mo_energy=energies,
                spin_density=density,
                converged=bool(raw.success),
            )
        else:
            raw = lf_hf_multi_optimize(
                m.tmat,
                m.U,
                m.g,
                m.omega,
                m.nelec,
                coeff0[0],
                coeff0[1],
                lam0,
                gtol=self.gtol,
                max_cycle=self.max_cycle,
            )
            result = dict(raw)
            result.update(energy=float(raw.fun), converged=bool(raw.success))
        self.lam, self.shift = result["lam"], result["shift"]
        self.mo_coeff, self.mo_energy = result["mo_coeff"], result["mo_energy"]
        return self._finish(result, result["energy"], result["converged"])

    def site_density(self):
        require_reference(self)
        return self.result["spin_density"].copy()
