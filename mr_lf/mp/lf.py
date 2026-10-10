"""Single-reference Lang–Firsov second-order perturbation theory."""

from itertools import combinations

import numpy as np

from .._driver import Calculation, require_reference
from ..model import ElectronPhononModel
from ..phonon import lf_total_phonon_configurations, lf_vacuum_coupling_site_matrices
from ..scf.lf import LFHF, lf_effective_one_body, lf_hf_multi_fixed_energy, lf_hf_multi_fixed_fock


def lf_mp2_denominators(
    mo_energy: np.ndarray,
    omega: float,
    occupations: np.ndarray,
    nocc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pure-phonon and electronic-single LF-MP2 denominators.

    Canonical MOs are ordered with occupied orbitals before virtual orbitals.
    For a non-vacuum phonon configuration ``X_k`` with total occupation
    ``N_k``, this function evaluates

    ``pure[k] = -N_k * omega``

    and

    ``singles[k, i, a] = epsilon[i] - epsilon[nocc+a] - N_k * omega``.

    Parameters
    ----------
    mo_energy
        Finite real canonical-MO energies in nondecreasing order with shape
        ``(nmo,)``.
    omega
        Finite positive phonon frequency shared by all modes.
    occupations
        Nonnegative integer non-vacuum configurations with shape
        ``(nconfig, nmode)`` and index order ``[k, x]``.
    nocc
        Number of occupied MOs, satisfying ``1 <= nocc < nmo``.

    Returns
    -------
    total_occupations
        Total phonon occupations ``N_k`` with shape ``(nconfig,)`` and
        ``int64`` dtype.
    pure_denominators
        Pure-phonon denominators with shape ``(nconfig,)``.
    single_denominators
        Electronic-single plus phonon denominators with shape
        ``(nconfig, nocc, nvir)`` and index order ``[k, i, a]``.  The local
        virtual index corresponds to global MO index ``nocc + a``.

    Notes
    -----
    Every returned denominator is strictly negative for a stable canonical
    reference and a non-vacuum phonon configuration.
    """
    if isinstance(nocc, (bool, np.bool_)) or not isinstance(nocc, (int, np.integer)):
        raise TypeError("Number of occupied MOs must be a positive integer.")
    if mo_energy.ndim != 1:
        raise ValueError("MO energy array must be 1-dimensional.")
    for i in range(mo_energy.size):
        if not np.isfinite(mo_energy[i]) or not np.isreal(mo_energy[i]):
            raise ValueError("MO energy array must contain finite real values.")
        if i < mo_energy.size - 1 and mo_energy[i] > mo_energy[i + 1]:
            raise ValueError("MO energy array must be nondecreasing.")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("Phonon frequency omega must be finite, real, and positive.")
    if occupations.ndim != 2:
        raise ValueError("Occupation array must be 2-dimensional.")
    for occ_value in occupations.flat:
        if not np.issubdtype(type(occ_value), np.integer) or occ_value < 0:
            raise ValueError("Occupation array must contain non-negative integers.")
    for total in np.sum(occupations, axis=1):
        if total <= 0:
            raise ValueError("Occupation array must not contain vacuum configurations.")
    nmo = mo_energy.size
    if not (1 <= nocc < nmo):
        raise ValueError("Number of occupied MOs must satisfy 1 <= nocc < nmo.")
    total_occupations = np.sum(occupations, axis=1, dtype=np.int64)
    denominator_pure = -total_occupations * omega
    G = mo_energy[:nocc, None] - mo_energy[nocc:]  # shape (nocc, nvir)
    denominator_singles = G[None, :, :] + denominator_pure[:, None, None]
    return total_occupations, denominator_pure, denominator_singles


def lf_mp2_energy(
    pure_matrix_elements: np.ndarray,
    single_matrix_elements: np.ndarray,
    pure_denominators: np.ndarray,
    single_denominators: np.ndarray,
) -> float:
    """Return the one-electron LF-MP2 second-order correction.

    For each non-vacuum phonon configuration ``X_k``, the pure-phonon
    channel has matrix element ``M[k]`` and denominator ``D[k]``.  The
    electronic-single plus phonon channel has matrix elements ``M[k, i, a]``
    and denominators ``D[k, i, a]``.  This function evaluates

    ``sum_k abs(M_pure[k])**2 / D_pure[k]``

    plus

    ``sum_kia abs(M_single[k, i, a])**2 / D_single[k, i, a]``.

    Parameters
    ----------
    pure_matrix_elements
        Finite real or complex pure-phonon amplitudes with shape
        ``(nconfig,)`` and index order ``[k]``.
    single_matrix_elements
        Finite real or complex electronic-single plus phonon amplitudes with
        shape ``(nconfig, nocc, nvir)`` and index order ``[k, i, a]``.
    pure_denominators
        Finite real strictly negative pure-phonon denominators with shape
        ``(nconfig,)``.
    single_denominators
        Finite real strictly negative electronic-single plus phonon
        denominators with shape ``(nconfig, nocc, nvir)``.

    Returns
    -------
    float
        Total LF-MP2 correction.  It is nonpositive and contains no extra
        spin, symmetry, or factorial prefactor.

    Notes
    -----
    The first axis must describe the same ordered phonon configurations in
    all four inputs.  This is the complete second-order correction for the
    one-electron Fig. 2b model; electronic double excitations do not exist.
    """
    if pure_matrix_elements.ndim != 1:
        raise ValueError("Pure-phonon matrix element array must be 1-dimensional.")
    if single_matrix_elements.ndim != 3:
        raise ValueError("Electronic-single matrix element array must be 3-dimensional.")
    if pure_denominators.ndim != 1:
        raise ValueError("Pure-phonon denominator array must be 1-dimensional.")
    if single_denominators.ndim != 3:
        raise ValueError("Electronic-single denominator array must be 3-dimensional.")
    if pure_matrix_elements.shape[0] != pure_denominators.shape[0]:
        raise ValueError("Pure-phonon matrix elements and denominators must have the same length.")
    if single_matrix_elements.shape != single_denominators.shape:
        raise ValueError(
            "Electronic-single matrix elements and denominators must have the same shape."
        )
    if pure_matrix_elements.shape[0] != single_matrix_elements.shape[0]:
        raise ValueError(
            "Pure-phonon and electronic-single matrix elements must have the same first dimension."
        )
    if (
        not np.all(np.isfinite(pure_denominators))
        or not np.all(np.isreal(pure_denominators))
        or not np.all(pure_denominators < 0)
    ):
        raise ValueError("Pure-phonon denominators must be finite, real, and strictly negative.")
    if (
        not np.all(np.isfinite(single_denominators))
        or not np.all(np.isreal(single_denominators))
        or not np.all(single_denominators < 0)
    ):
        raise ValueError(
            "Electronic-single denominators must be finite, real, and strictly negative."
        )
    if not np.all(np.isfinite(pure_matrix_elements)):
        raise ValueError("Pure-phonon matrix elements must be finite.")
    if not np.all(np.isfinite(single_matrix_elements)):
        raise ValueError("Electronic-single matrix elements must be finite.")
    energy_pure = np.sum(np.abs(pure_matrix_elements) ** 2 / pure_denominators)
    energy_single = np.sum(np.abs(single_matrix_elements) ** 2 / single_denominators)
    total_energy = energy_pure + energy_single
    return float(total_energy)


def lf_vacuum_coupling_mo_elements(
    site_couplings: np.ndarray,
    mo_coeff: np.ndarray,
    nocc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform non-vacuum LF couplings and extract MP2 channels.

    Each site-basis matrix is transformed with the same orthonormal canonical
    MO coefficients according to ``W_mo[k] = C.conj().T @ W_site[k] @ C``.
    Occupied MOs must precede virtual MOs.  For the one-body operator, the
    unchanged-determinant amplitude is the occupied trace, while the
    determinant-single amplitude is ``W_mo[k, a, i]``.

    Parameters
    ----------
    site_couplings
        Non-vacuum site-basis matrices with shape
        ``(nconfig, nsite, nsite)`` and index order ``[k, p, q]``.
        A matrix for fixed configuration need not be Hermitian.
    mo_coeff
        Finite square unitary canonical-MO coefficient matrix with shape
        ``(nsite, nmo)`` and index order ``[p, m]``.  Columns are MOs and
        ``nsite == nmo`` for the complete site basis used here.
    nocc
        Number of occupied MOs, satisfying ``1 <= nocc < nmo``.  The Fig. 2b
        one-electron problem uses ``nocc = 1``.

    Returns
    -------
    mo_couplings
        All transformed matrices with shape ``(nconfig, nmo, nmo)`` and
        index order ``[k, m, n]``.
    pure_phonon
        Unchanged-determinant amplitudes
        ``sum_i W_mo[k, i, i]`` with shape ``(nconfig,)``.
    singles
        Single-excitation amplitudes ``W_mo[k, a, i]`` with shape
        ``(nconfig, nocc, nvir)`` and index order ``[k, i, a]``.  The local
        virtual index ``a`` corresponds to global MO index ``nocc + a``.

    Notes
    -----
    The returned occupied trace is the complete pure-phonon electronic
    matrix element for the one-electron Fig. 2b problem.  Additional
    many-electron LF two-body terms would need separate treatment.
    """
    if site_couplings.ndim != 3:
        raise ValueError("Site-basis coupling array must be 3-dimensional.")
    for site_value in site_couplings.flat:
        if not np.isfinite(site_value):
            raise ValueError("Site-basis coupling array must contain finite values.")
    nconfig, nsite, nsite2 = site_couplings.shape
    if nsite != nsite2:
        raise ValueError("Site-basis coupling matrices must be square.")
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != mo_coeff.shape[1] or mo_coeff.shape[0] != nsite:
        raise ValueError("MO coefficient array must be square and match site-basis size.")
    for mo_value in mo_coeff.flat:
        if not np.isfinite(mo_value):
            raise ValueError("MO coefficient array must contain finite values.")
    nmo = mo_coeff.shape[1]
    if not isinstance(nocc, (int, np.integer)) or isinstance(nocc, (bool, np.bool_)):
        raise TypeError("Number of occupied MOs must be a positive integer.")
    if not (1 <= nocc < nmo):
        raise ValueError("Number of occupied MOs must satisfy 1 <= nocc < nmo.")
    C = mo_coeff
    if not np.allclose(C.conj().T @ C, np.eye(nmo)):
        raise ValueError("MO coefficient array must be unitary.")
    W_mo = np.empty((nconfig, nmo, nmo), dtype=np.result_type(site_couplings, np.complex128))
    for k in range(nconfig):
        W_mo[k] = C.conj().T @ site_couplings[k] @ C
    pure_phonon = np.array(
        [np.trace(W_mo[k, :nocc, :nocc]) for k in range(nconfig)], dtype=np.complex128
    )
    singles = W_mo[:, nocc:, :nocc].transpose(0, 2, 1)
    return W_mo, pure_phonon, singles


def lf_mp2_reference_point(
    tmat: np.ndarray,
    g: float,
    omega: float,
    shift: np.ndarray,
    lam: np.ndarray,
    *,
    max_total: int,
) -> dict[str, object]:
    """Evaluate LF-HF and LF-MP2 for one fixed one-electron LF reference.

    The final zero-phonon effective Hamiltonian is diagonalized to define the
    canonical occupied and virtual MOs.  Every non-vacuum phonon configuration
    satisfying ``sum_x n_x <= max_total`` is then included in the pure-phonon
    and electronic-single LF-MP2 channels.

    Parameters
    ----------
    tmat
        Finite Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Finite real local electron--phonon coupling strength.
    omega
        Finite positive phonon frequency shared by all modes.
    shift
        Finite real coherent displacements with shape ``(norb,)``.  The
        gauge-fixed full-matrix LF-HF calculation passes a zero vector.
    lam
        Finite real conditional displacements with shape ``(norb, norb)`` and
        index order ``lam[x, p]``.
    max_total
        Positive inclusive cutoff on the total phonon occupation.

    Returns
    -------
    dict
        Auditable result with exactly the following entries:

        ``hf_energy``
            Zero-phonon LF-HF energy including ``omega * shift @ shift``.
        ``mp2_correction``
            LF-MP2 second-order correction through ``max_total``.
        ``total_energy``
            Sum of ``hf_energy`` and ``mp2_correction``.
        ``max_total``
            Integer total-phonon cutoff used for this result.
        ``coeff``
            Occupied canonical MO with shape ``(norb,)``.
        ``mo_energy``, ``mo_coeff``
            Canonical energies with shape ``(norb,)`` and site-to-MO
            coefficients with shape ``(norb, norb)``.
        ``occupations``, ``total_occupations``
            Non-vacuum configurations with shape ``(nconfig, norb)`` and
            their totals with shape ``(nconfig,)``.
        ``pure_matrix_elements``, ``pure_denominators``
            Pure-phonon channel arrays with shape ``(nconfig,)``.
        ``single_matrix_elements``, ``single_denominators``
            Electronic-single plus phonon arrays with shape
            ``(nconfig, 1, norb - 1)`` and index order ``[k, i, a]``.

    Notes
    -----
    This routine does not optimize ``lam`` or ``shift``.  Simultaneously
    applying ``lam[x, p] -> lam[x, p] + c[x]`` and
    ``shift[x] -> shift[x] + c[x]`` leaves all returned energies invariant.
    """
    phonon_configurations = lf_total_phonon_configurations(nmode=lam.shape[0], max_total=max_total)
    heff = lf_effective_one_body(tmat, g, omega, shift, lam)
    mo_energy, mo_coeff = np.linalg.eigh(heff)
    nocc = 1
    coeff = mo_coeff[:, 0]
    energy_hf = omega * np.dot(shift, shift) + mo_energy[0]
    site_couplings = lf_vacuum_coupling_site_matrices(
        tmat, g, omega, shift, lam, phonon_configurations
    )
    _, pure_matrix_elements, single_matrix_elements = lf_vacuum_coupling_mo_elements(
        site_couplings,
        mo_coeff,
        nocc,
    )
    total_occupations, pure_denominators, single_denominators = lf_mp2_denominators(
        mo_energy, omega, phonon_configurations, nocc
    )
    mp2_energy = lf_mp2_energy(
        pure_matrix_elements, single_matrix_elements, pure_denominators, single_denominators
    )
    total_energy = energy_hf + mp2_energy
    return {
        "hf_energy": energy_hf,
        "mp2_correction": mp2_energy,
        "total_energy": total_energy,
        "max_total": max_total,
        "coeff": coeff,
        "mo_energy": mo_energy,
        "mo_coeff": mo_coeff,
        "occupations": phonon_configurations,
        "total_occupations": total_occupations,
        "pure_matrix_elements": pure_matrix_elements,
        "pure_denominators": pure_denominators,
        "single_matrix_elements": single_matrix_elements,
        "single_denominators": single_denominators,
    }


def lf_mp2_unrestricted_reference_point(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    nelec: tuple[int, int],
    mo_coeff: np.ndarray,
    mo_energy: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
    *,
    max_total: int,
    max_excited_modes: int = 2,
    denominator_tol: float = 1e-10,
) -> dict[str, object]:
    """Evaluate LF-MP2 on a fixed unrestricted LF-HF reference.

    Supports any positive number of sites and any alpha/beta electron counts.
    The local model has one phonon mode per site. For a non-vacuum phonon
    configuration the transformed Hamiltonian couples the reference only to
    itself and electronic singles; zero-phonon doubles arise from the static
    transformed density interaction. All orbital arrays are real float64.

    Parameters
    ----------
    tmat, U, g, omega, nelec
        Model data in the same convention as lf_hf_multi_fixed_energy.
    mo_coeff
        Canonical full alpha/beta orbital matrices, shape (2, L, L), with
        occupied orbitals before virtual orbitals.
    mo_energy
        Corresponding spin orbital energies, shape (2, L).
    lam, shift
        Full LF matrix of shape (L, L) and coherent displacement of shape
        (L,). The gauge-fixed optimizer returns a zero shift.
    max_total
        Inclusive total phonon occupation cutoff.
    max_excited_modes
        Maximum number of modes with nonzero occupation; default 2.
    denominator_tol
        Any excitation energy at or below this positive threshold raises
        ValueError instead of entering the MP2 sum.

    Returns
    -------
    dict
        HF, separate MP2 channel corrections, total energy, phonon
        configurations, smallest positive denominator, and channel counts.
    """
    if not isinstance(tmat, np.ndarray) or tmat.ndim != 2:
        raise ValueError("tmat must be a square float64 array")
    L = tmat.shape[0]
    if (
        not isinstance(mo_coeff, np.ndarray)
        or mo_coeff.shape != (2, L, L)
        or mo_coeff.dtype != np.float64
        or not np.all(np.isfinite(mo_coeff))
    ):
        raise ValueError("mo_coeff must be finite float64 with shape (2, L, L)")
    if (
        not isinstance(mo_energy, np.ndarray)
        or mo_energy.shape != (2, L)
        or mo_energy.dtype != np.float64
        or not np.all(np.isfinite(mo_energy))
    ):
        raise ValueError("mo_energy must be finite float64 with shape (2, L)")
    for coeff in mo_coeff:
        if not np.allclose(coeff.T @ coeff, np.eye(L), atol=1e-10, rtol=0):
            raise ValueError("mo_coeff columns must be orthonormal")
    if (
        isinstance(max_total, (bool, np.bool_))
        or not isinstance(max_total, (int, np.integer))
        or max_total < 1
    ):
        raise ValueError("max_total must be a positive integer")
    if (
        isinstance(max_excited_modes, (bool, np.bool_))
        or not isinstance(max_excited_modes, (int, np.integer))
        or max_excited_modes < 1
    ):
        raise ValueError("max_excited_modes must be a positive integer")
    if (
        isinstance(denominator_tol, (bool, np.bool_))
        or not isinstance(denominator_tol, (int, float, np.integer, np.floating))
        or not np.isfinite(denominator_tol)
        or denominator_tol <= 0
    ):
        raise ValueError("denominator_tol must be finite and positive")

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
        raise ValueError("nelec must be two electron counts between zero and L")
    na, nb = nelec
    hf_energy, _ = lf_hf_multi_fixed_energy(
        tmat,
        U,
        g,
        omega,
        nelec,
        mo_coeff[0, :, :na],
        mo_coeff[1, :, :nb],
        lam,
        shift,
    )
    fock = lf_hf_multi_fixed_fock(
        tmat,
        U,
        g,
        omega,
        nelec,
        mo_coeff[0, :, :na],
        mo_coeff[1, :, :nb],
        lam,
        shift,
    )
    fock_mo = np.stack([mo_coeff[s].T @ fock[s] @ mo_coeff[s] for s in range(2)])
    for s, nocc in enumerate(nelec):
        for block in (fock_mo[s, :nocc, :nocc], fock_mo[s, nocc:, nocc:]):
            if np.max(np.abs(block - np.diag(np.diag(block))), initial=0.0) > 1e-7:
                raise ValueError("occupied and virtual Fock blocks must be canonical")
        if np.max(np.abs(np.diag(fock_mo[s]) - mo_energy[s]), initial=0.0) > 1e-7:
            raise ValueError("mo_energy must match the canonical Fock diagonal")

    # For one electron the exact zero-phonon Hamiltonian defines the
    # perturbative orbital energies, as in the existing Fig. 2b routine.
    # Its virtual spectrum can differ from the unrestricted Fock spectrum.
    mo_coeff = mo_coeff.copy()
    mo_energy = mo_energy.copy()
    fock_mo = fock_mo.copy()
    if na + nb == 1:
        active_spin = 0 if na else 1
        effective = lf_effective_one_body(tmat, g, omega, shift, lam)
        effective_energy, effective_coeff = np.linalg.eigh(effective)
        overlap = abs(np.dot(effective_coeff[:, 0], mo_coeff[active_spin, :, 0]))
        if overlap < 1 - 1e-5:
            raise ValueError("one-electron reference is not the lowest effective orbital")
        mo_coeff[active_spin] = effective_coeff
        mo_energy[active_spin] = effective_energy
        fock_mo[active_spin] = np.diag(effective_energy)

    occupations = lf_total_phonon_configurations(L, max_total)
    occupations = occupations[np.count_nonzero(occupations, axis=1) <= max_excited_modes]
    totals = occupations.sum(axis=1)
    site_couplings = lf_vacuum_coupling_site_matrices(
        tmat,
        g,
        omega,
        np.zeros_like(shift),
        lam,
        occupations,
    )
    mo_couplings = np.stack(
        [
            np.einsum("pi,kpq,qj->kij", mo_coeff[s], site_couplings, mo_coeff[s], optimize=True)
            for s in range(2)
        ]
    )
    # omega*z_x is a phonon scalar, not omega*z_x times electron number.
    # The shared site coupling builder is a one-electron interface: use
    # z=0 above and add the scalar once, only for one-phonon configurations.
    pure_amplitudes = np.zeros(len(occupations), dtype=np.float64)
    one_phonon = totals == 1
    pure_amplitudes[one_phonon] = omega * (occupations[one_phonon] @ shift)
    phonon_single_correction = 0.0
    zero_single_correction = 0.0
    minimum_denominator = float(omega)
    single_count = 0

    for s, nocc in enumerate(nelec):
        pure_amplitudes += np.trace(
            mo_couplings[s, :, :nocc, :nocc],
            axis1=1,
            axis2=2,
        )
        virtual_energies = mo_energy[s, nocc:]
        occupied_energies = mo_energy[s, :nocc]
        gap = virtual_energies[None, :] - occupied_energies[:, None]
        if gap.size:
            smallest_gap = float(np.min(gap))
            minimum_denominator = min(minimum_denominator, smallest_gap)
            if smallest_gap <= denominator_tol:
                raise ValueError(
                    f"nonpositive or near-zero electronic single denominator: {smallest_gap}"
                )
            phonon_denominator = gap[None, :, :] + omega * totals[:, None, None]
            amplitudes = mo_couplings[s, :, nocc:, :nocc].transpose(0, 2, 1)
            phonon_single_correction -= float(np.sum(np.abs(amplitudes) ** 2 / phonon_denominator))
            zero_amplitudes = fock_mo[s, nocc:, :nocc].T
            zero_single_correction -= float(np.sum(np.abs(zero_amplitudes) ** 2 / gap))
            single_count += nocc * (L - nocc)

    phonon_pure_correction = -float(np.sum(np.abs(pure_amplitudes) ** 2 / (omega * totals)))

    # Site determinant vectors provide the static two-body matrix elements
    # without spin-dependent exchange or excitation-sign bookkeeping.
    site_strings = [
        (
            np.empty((1, 0), dtype=np.int64)
            if n == 0
            else np.asarray(list(combinations(range(L), n)), dtype=np.int64).reshape(-1, n)
        )
        for n in nelec
    ]
    site_bits = []
    for strings in site_strings:
        bits = np.zeros((len(strings), L), dtype=np.float64)
        bits[np.arange(len(strings))[:, None], strings] = 1.0
        site_bits.append(bits)
    nsite = site_bits[0][:, None, :] + site_bits[1][None, :, :]
    G = lam.T @ lam
    A = omega * G - g * (lam + lam.T)
    site_potential = np.einsum("abp,pq,abq->ab", nsite, A, nsite, optimize=True) + U * (
        site_bits[0] @ site_bits[1].T
    )

    def determinant_vector(spin, occupied_mos):
        nocc = nelec[spin]
        if nocc == 0:
            return np.ones(1, dtype=np.float64)
        coeff = mo_coeff[spin]
        return np.linalg.det(coeff[site_strings[spin]][:, :, occupied_mos])

    references = [determinant_vector(s, tuple(range(nelec[s]))) for s in range(2)]
    transitions = [{tuple(range(nelec[s])): references[s] ** 2} for s in range(2)]

    def transition(spin, occupied_mos):
        occupied_mos = tuple(occupied_mos)
        if occupied_mos not in transitions[spin]:
            transitions[spin][occupied_mos] = (
                determinant_vector(spin, occupied_mos) * references[spin]
            )
        return transitions[spin][occupied_mos]

    def excited_occ(spin, holes, particles):
        return tuple(sorted((set(range(nelec[spin])) - set(holes)) | set(particles)))

    def double_term(holes_a, particles_a, holes_b, particles_b):
        delta = (
            np.sum(mo_energy[0, list(particles_a)])
            - np.sum(mo_energy[0, list(holes_a)])
            + np.sum(mo_energy[1, list(particles_b)])
            - np.sum(mo_energy[1, list(holes_b)])
        )
        return float(delta), float(
            np.einsum(
                "a,ab,b->",
                transition(0, excited_occ(0, holes_a, particles_a)),
                site_potential,
                transition(1, excited_occ(1, holes_b, particles_b)),
                optimize=True,
            )
        )

    zero_double_correction = 0.0
    double_count = 0
    for s, nocc in enumerate(nelec):
        for holes in combinations(range(nocc), 2):
            for particles in combinations(range(nocc, L), 2):
                args = (holes, particles, (), ()) if s == 0 else ((), (), holes, particles)
                denominator, amplitude = double_term(*args)
                minimum_denominator = min(minimum_denominator, denominator)
                if denominator <= denominator_tol:
                    raise ValueError(
                        f"nonpositive or near-zero electronic double denominator: {denominator}"
                    )
                zero_double_correction -= amplitude**2 / denominator
                double_count += 1
    for hole_a in range(na):
        for particle_a in range(na, L):
            for hole_b in range(nb):
                for particle_b in range(nb, L):
                    denominator, amplitude = double_term(
                        (hole_a,), (particle_a,), (hole_b,), (particle_b,)
                    )
                    minimum_denominator = min(minimum_denominator, denominator)
                    if denominator <= denominator_tol:
                        raise ValueError(
                            f"nonpositive or near-zero electronic double denominator: {denominator}"
                        )
                    zero_double_correction -= amplitude**2 / denominator
                    double_count += 1

    mp2_correction = (
        phonon_pure_correction
        + phonon_single_correction
        + zero_single_correction
        + zero_double_correction
    )
    return {
        "hf_energy": float(hf_energy),
        "mp2_correction": float(mp2_correction),
        "total_energy": float(hf_energy + mp2_correction),
        "phonon_pure_correction": phonon_pure_correction,
        "phonon_single_correction": phonon_single_correction,
        "zero_single_correction": zero_single_correction,
        "zero_double_correction": float(zero_double_correction),
        "occupations": occupations,
        "max_total": int(max_total),
        "max_excited_modes": int(max_excited_modes),
        "minimum_denominator": float(minimum_denominator),
        "zero_single_count": single_count,
        "zero_double_count": double_count,
    }


class LFMP2(Calculation):
    def __init__(self, hf, *, max_total=10, max_excited_modes=None, denominator_tol=1e-10):
        if not isinstance(hf, LFHF):
            raise TypeError("LFMP2 requires an LFHF reference")
        super().__init__(hf.model)
        self.hf = hf
        self.max_total = max_total
        self.max_excited_modes = (hf.model.nmode if isinstance(hf.model, ElectronPhononModel)
                                 else 2) if max_excited_modes is None else max_excited_modes
        self.denominator_tol = denominator_tol

    def kernel(self):
        self._reset()
        require_reference(self.hf)
        m, hf = self.model, self.hf
        if isinstance(m, ElectronPhononModel):
            from .general_lf import correction

            result = correction(hf, max_total=self.max_total,
                                max_excited_modes=self.max_excited_modes,
                                denominator_tol=self.denominator_tol)
            return self._finish(result, result["total_energy"], True, result["mp2_correction"])
        # The original one-electron all-mode path has its own effective
        # orbital partition. Its cutoff is a total occupation cutoff only.
        if sum(m.nelec) == 1:
            result = lf_mp2_reference_point(
                m.tmat, m.g, m.omega, hf.shift, hf.lam, max_total=self.max_total
            )
            result["max_excited_modes"] = m.L
        else:
            result = lf_mp2_unrestricted_reference_point(
                m.tmat,
                m.U,
                m.g,
                m.omega,
                m.nelec,
                hf.mo_coeff,
                hf.mo_energy,
                hf.lam,
                hf.shift,
                max_total=self.max_total,
                max_excited_modes=self.max_excited_modes,
                denominator_tol=self.denominator_tol,
            )
        if abs(result["hf_energy"] - hf.e_tot) > 1e-7:
            raise ValueError("MP2 reference does not reproduce the supplied HF energy")
        result["cutoff_converged"] = False
        return self._finish(result, result["total_energy"], True, result["mp2_correction"])
