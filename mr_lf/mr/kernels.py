"""Overlap and Hamiltonian blocks between displaced number-state LF frames."""

import numpy as np

from .. import basis
from ..phonon import multimode_displacement_overlap, multimode_phonon_operator_matrices
from ..symmetry import _translation_electron_labels
from . import noci


def lf_frame_phonon_overlap(
    lam_bra: np.ndarray,
    shift_bra: np.ndarray,
    lam_ket: np.ndarray,
    shift_ket: np.ndarray,
    nelec: tuple[int, int],
    bra_occupations: np.ndarray,
    ket_occupations: np.ndarray,
) -> np.ndarray:
    """Return the excited-phonon overlap block between two real LF frames.

    Parameters
    ----------
    lam_bra, lam_ket
        Finite float64 arrays (L, L), L > 0, indexed [phonon mode x,
        electron site p]. Both frames have one phonon mode per site.
    shift_bra, shift_ket
        Finite float64 arrays (L,), indexed by phonon mode x.
    nelec
        Tuple (neleca, nelecb) of integer electron counts in [0, L].
        Both ends use the same orthonormal site-determinant basis in
        basis.make_electron_basis order. Booleans are rejected.
    bra_occupations, ket_occupations
        Nonnegative int64 arrays (nbra, L) and (nket, L), with nonempty
        configuration lists. Each row is a complete phonon configuration;
        vacuum, duplicate rows, and arbitrary input row order are allowed.

    Returns
    -------
    numpy.ndarray
        Float64 tensor (nstra, nstrb, nbra, nstra, nstrb, nket), indexed
        [ia, ib, k, ja, jb, l], where nstr_sigma = comb(L, nelec_sigma).
        It is the overlap of |I> D(eta_bra(I)) |bra_k> and
        |J> D(eta_ket(J)) |ket_l>, with eta = shift - lam @ site_density.
        Elements vanish unless ia == ja and ib == jb. On matching
        determinants the displacement difference is eta_ket - eta_bra.

    Raises
    ------
    TypeError
        For non-array inputs or arrays with wrong dtype.
    ValueError
        For invalid shapes, nonfinite parameters, invalid electron counts,
        empty configuration lists, or negative phonon occupations.

    Notes
    -----
    Inputs must not be modified. This is one unprojected frame-pair block,
    evaluated in the complete phonon space for the supplied configurations.
    No hopping, fermion excitation signs, translation projection or reference
    subtraction is applied here. Different site determinants are orthogonal.
    Only at a solver boundary flatten (alpha address, beta address, phonon
    configuration) in C order. A rectangular bra/ket block is allowed.

    """
    for name, array in (
        ("lam_bra", lam_bra),
        ("lam_ket", lam_ket),
        ("shift_bra", shift_bra),
        ("shift_ket", shift_ket),
    ):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if lam_bra.ndim != 2 or lam_bra.shape[0] == 0 or lam_bra.shape[0] != lam_bra.shape[1]:
        raise ValueError("lam_bra must have nonempty shape (L,L)")
    nsite = lam_bra.shape[0]
    if lam_ket.shape != (nsite, nsite):
        raise ValueError("lam_ket must have the same (L,L) shape as lam_bra")
    for name, shift in (("shift_bra", shift_bra), ("shift_ket", shift_ket)):
        if shift.shape != (nsite,):
            raise ValueError(f"{name} must have shape (L,)")
    if (
        not isinstance(nelec, tuple)
        or len(nelec) != 2
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 0
            or n > nsite
            for n in nelec
        )
    ):
        raise ValueError("nelec must be (neleca,nelecb) with integer counts in [0,L]")
    for name, occupations in (
        ("bra_occupations", bra_occupations),
        ("ket_occupations", ket_occupations),
    ):
        if not isinstance(occupations, np.ndarray) or occupations.dtype != np.int64:
            raise TypeError(f"{name} must be an int64 NumPy array")
        if occupations.ndim != 2 or occupations.shape[0] == 0 or occupations.shape[1] != nsite:
            raise ValueError(f"{name} must have nonempty shape (nconfig,L)")
        if np.any(occupations < 0):
            raise ValueError(f"{name} must contain nonnegative occupations")
    str_a, _ = basis.make_electron_basis(nsite, nelec[0])
    str_b, _ = basis.make_electron_basis(nsite, nelec[1])
    sites = np.arange(nsite, dtype=np.int64)
    occ_a = (str_a[:, None] >> sites[None, :] & 1).astype(np.float64)
    occ_b = (str_b[:, None] >> sites[None, :] & 1).astype(np.float64)
    occupation = occ_a[:, None, :] + occ_b[None, :, :]
    eta_bra = shift_bra - np.einsum("xp,ijp->ijx", lam_bra, occupation)
    eta_ket = shift_ket - np.einsum("xp,ijp->ijx", lam_ket, occupation)
    difference = eta_ket - eta_bra
    S = np.zeros(
        (
            str_a.shape[0],
            str_b.shape[0],
            bra_occupations.shape[0],
            str_a.shape[0],
            str_b.shape[0],
            ket_occupations.shape[0],
        )
    )
    for ia in range(str_a.shape[0]):
        for ib in range(str_b.shape[0]):
            F = multimode_displacement_overlap(difference[ia, ib], bra_occupations, ket_occupations)
            S[ia, ib, :, ia, ib, :] = F
    return S


def lf_frame_diagonal_hamiltonian(
    U: float,
    g: float,
    omega: float,
    lam_bra: np.ndarray,
    shift_bra: np.ndarray,
    lam_ket: np.ndarray,
    shift_ket: np.ndarray,
    nelec: tuple[int, int],
    bra_occupations: np.ndarray,
    ket_occupations: np.ndarray,
) -> np.ndarray:
    """Return the electronic-diagonal Hamiltonian block between two LF frames.

    The operator is U*sum_x n_alpha[x]*n_beta[x]
    + omega*sum_x b[x].dagger*b[x]
    + g*sum_x n[x]*(b[x]+b[x].dagger).
    It uses uncentered density and no phonon zero-point constant, so no
    centered-to-paper energy correction is applied. No one-electron
    matrix, including its site-diagonal entries, is included here.

    Parameters
    ----------
    U, g, omega
        Finite real numeric scalars; omega must be strictly positive.
        Python and NumPy numeric scalars are accepted. Booleans, complex
        values, and array scalars are rejected. U and g may have either sign.
    lam_bra, lam_ket
        Finite float64 arrays (L, L), 0 < L < 64, indexed [phonon mode x,
        electron site p]. There is one phonon mode per electron site.
    shift_bra, shift_ket
        Finite float64 arrays (L,), indexed by phonon mode x.
    nelec
        Tuple (neleca, nelecb) of integer counts in [0, L]; booleans are
        rejected. Site-determinant addresses at both ends follow
        basis.make_electron_basis order.
    bra_occupations, ket_occupations
        Nonnegative int64 arrays (nbra, L) and (nket, L), with nonempty
        lists of complete phonon configurations in the corresponding
        displaced frame. Vacuum, duplicate rows, and arbitrary row order
        are allowed and input row order must be preserved.

    Returns
    -------
    numpy.ndarray
        Float64 tensor (nstra, nstrb, nbra, nstra, nstrb, nket), indexed
        [ia, ib, k, ja, jb, l], where nstr_sigma = comb(L, nelec_sigma).
        The states are |I> D(shift-lam@density(I)) |configuration>.
        Elements vanish unless ia == ja and ib == jb. This condition
        concerns electron determinants; the phonon block need not be
        diagonal. Rectangular bra/ket phonon lists are supported.

    Raises
    ------
    TypeError
        For non-array inputs or wrong array dtypes, or non-real scalar
        parameter types.
    ValueError
        For invalid shapes, nonfinite parameters, nonpositive omega,
        invalid electron counts, or negative phonon occupations.

    Notes
    -----
    Inputs must not be modified. This is one unprojected frame-pair block,
    not an energy or an electron-phonon ground-state solver. It excludes
    hopping, translation projection, reference-space subtraction, and PT2.
    Displaced matrix elements refer to the complete phonon space, with
    only the requested bra/ket configurations selected.

    Reuse multimode_phonon_operator_matrices for each fixed electron
    determinant and its two conditional displacements. Its mode-resolved
    operator blocks already contain spectator-mode overlaps. Multiply the
    Hubbard constant by its overlap block, but do not multiply the phonon
    and coupling blocks by another overlap.
    Only at a solver boundary flatten (alpha address, beta address, phonon
    configuration) in C order.
    """
    parameters = []
    for name, value in (("U", U), ("g", g), ("omega", omega)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real numeric scalar")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be finite in float64") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite in float64")
        parameters.append(value)
    U, g, omega = parameters
    if omega <= 0:
        raise ValueError("omega must be positive")

    for name, array in (
        ("lam_bra", lam_bra),
        ("lam_ket", lam_ket),
        ("shift_bra", shift_bra),
        ("shift_ket", shift_ket),
    ):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if lam_bra.ndim != 2 or not 0 < lam_bra.shape[0] < 64 or lam_bra.shape[0] != lam_bra.shape[1]:
        raise ValueError("lam_bra must have shape (L,L) with 0 < L < 64")
    nsite = lam_bra.shape[0]
    if lam_ket.shape != (nsite, nsite):
        raise ValueError("lam_ket must have the same (L,L) shape as lam_bra")
    for name, shift in (("shift_bra", shift_bra), ("shift_ket", shift_ket)):
        if shift.shape != (nsite,):
            raise ValueError(f"{name} must have shape (L,)")
    if (
        not isinstance(nelec, tuple)
        or len(nelec) != 2
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 0
            or n > nsite
            for n in nelec
        )
    ):
        raise ValueError("nelec must be (neleca,nelecb) with integer counts in [0,L]")
    for name, occupations in (
        ("bra_occupations", bra_occupations),
        ("ket_occupations", ket_occupations),
    ):
        if not isinstance(occupations, np.ndarray) or occupations.dtype != np.int64:
            raise TypeError(f"{name} must be an int64 NumPy array")
        if occupations.ndim != 2 or occupations.shape[0] == 0 or occupations.shape[1] != nsite:
            raise ValueError(f"{name} must have nonempty shape (nconfig,L)")
        if np.any(occupations < 0):
            raise ValueError(f"{name} must contain nonnegative occupations")
    str_a, _ = basis.make_electron_basis(nsite, nelec[0])
    str_b, _ = basis.make_electron_basis(nsite, nelec[1])
    sites = np.arange(nsite, dtype=np.int64)
    occ_a = (str_a[:, None] >> sites[None, :] & 1).astype(np.float64)
    occ_b = (str_b[:, None] >> sites[None, :] & 1).astype(np.float64)
    occupation = occ_a[:, None, :] + occ_b[None, :, :]
    double_occupancy = occ_a[:, None, :] * occ_b[None, :, :]
    eta_bra = shift_bra - np.einsum("xp,ijp->ijx", lam_bra, occupation)
    eta_ket = shift_ket - np.einsum("xp,ijp->ijx", lam_ket, occupation)
    H_d = np.zeros(
        (
            str_a.shape[0],
            str_b.shape[0],
            bra_occupations.shape[0],
            str_a.shape[0],
            str_b.shape[0],
            ket_occupations.shape[0],
        )
    )
    for ia in range(str_a.shape[0]):
        for ib in range(str_b.shape[0]):
            result = multimode_phonon_operator_matrices(
                eta_bra[ia, ib, :], eta_ket[ia, ib, :], bra_occupations, ket_occupations
            )
            S = result["overlap"]
            B = result["annihilation"]
            C = result["creation"]
            N = result["number"]
            H_d[ia, ib, :, ia, ib, :] = (
                U * np.sum(double_occupancy[ia, ib]) * S
                + omega * np.sum(N, axis=0)
                + g * np.einsum("x, xij->ij", occupation[ia, ib, :], B + C)
            )
    return H_d


def lf_frame_hopping_hamiltonian(
    tmat: np.ndarray,
    lam_bra: np.ndarray,
    shift_bra: np.ndarray,
    lam_ket: np.ndarray,
    shift_ket: np.ndarray,
    nelec: tuple[int, int],
    bra_occupations: np.ndarray,
    ket_occupations: np.ndarray,
) -> np.ndarray:
    """Return the one-electron Hamiltonian block between two real LF frames.

    The operator is sum_{p,q,sigma} tmat[p,q]*a[p,sigma].dagger*a[q,sigma].
    It acts as the identity on physical phonons, but the conditional LF
    displacements can differ for the electron determinants at the two ends.
    The name 'hopping' includes all site-diagonal tmat entries as well.

    Parameters
    ----------
    tmat
        Finite float64 array (L,L), indexed [creation site p,
        annihilation site q]. Must be exactly symmetric, matching the
        existing vacuum kernel. Site-diagonal entries are allowed. No
        lattice, bond multiplicity, or boundary condition is inferred.
    lam_bra, lam_ket
        Finite float64 arrays (L,L), 0 < L < 64, indexed [phonon mode x,
        electron site p]. There is one phonon mode per electron site.
    shift_bra, shift_ket
        Finite float64 arrays (L,), indexed by phonon mode x.
    nelec
        Tuple (neleca, nelecb) of integer counts in [0,L]; booleans are
        rejected. Determinant addresses follow basis.make_electron_basis
        order. Alpha and beta electron counts are separately conserved.
    bra_occupations, ket_occupations
        Nonnegative int64 arrays (nbra,L) and (nket,L), with nonempty
        lists of complete configurations in the corresponding displaced
        frame. Vacuum, duplicate rows, and arbitrary row order are allowed.

    Returns
    -------
    numpy.ndarray
        Float64 tensor (nstra,nstrb,nbra,nstra,nstrb,nket), indexed
        [ia,ib,k,ja,jb,l]. Each element equals the electronic one-body
        matrix element <I|Ht|J> times the purely phononic displaced overlap
        <bra_k|D(eta_bra(I)).dagger D(eta_ket(J))|ket_l>.
        eta(I) = shift - lam @ density(I), with density summed over spins.
        Electron determinants can differ by one same-spin excitation or
        coincide. Rectangular bra/ket phonon lists are supported.

    Raises
    ------
    TypeError
        For non-array inputs or wrong array dtypes.
    ValueError
        For invalid shapes, nonfinite parameters, nonsymmetric tmat,
        invalid electron counts, or negative phonon occupations.

    Notes
    -----
    Inputs must not be modified. This is one unprojected frame-pair block;
    Hubbard, phonon energy, and density-phonon coupling belong to
    lf_frame_diagonal_hamiltonian. No energy-convention shift, translation
    projection, reference subtraction, or PT2 response is included.

    Reuse cistring links from basis.make_electron_basis for electronic
    one-body matrix elements. A link row is (create,annihilate,target,sign)
    and belongs to its source ket. Target is the bra address. Accumulate
    links sharing a target/source, including occupied site-diagonal links.
    For a fixed determinant pair reuse multimode_displacement_overlap with
    delta = eta_ket(J) - eta_bra(I). Do not multiply by the full electronic
    overlap from lf_frame_phonon_overlap: that would remove physical hops.
    Only at a solver boundary flatten (alpha address,beta address,phonon
    configuration) in C order.
    """
    for name, array in (
        ("tmat", tmat),
        ("lam_bra", lam_bra),
        ("lam_ket", lam_ket),
        ("shift_bra", shift_bra),
        ("shift_ket", shift_ket),
    ):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if lam_bra.ndim != 2 or not 0 < lam_bra.shape[0] < 64 or lam_bra.shape[0] != lam_bra.shape[1]:
        raise ValueError("lam_bra must have shape (L,L) with 0 < L < 64")
    nsite = lam_bra.shape[0]
    if lam_ket.shape != (nsite, nsite):
        raise ValueError("lam_ket must have the same (L,L) shape as lam_bra")
    if tmat.shape != (nsite, nsite):
        raise ValueError("tmat must have shape (L,L)")
    if not np.array_equal(tmat, tmat.T):
        raise ValueError("tmat must be symmetric")
    for name, shift in (("shift_bra", shift_bra), ("shift_ket", shift_ket)):
        if shift.shape != (nsite,):
            raise ValueError(f"{name} must have shape (L,)")
    if (
        not isinstance(nelec, tuple)
        or len(nelec) != 2
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 0
            or n > nsite
            for n in nelec
        )
    ):
        raise ValueError("nelec must be (neleca,nelecb) with integer counts in [0,L]")
    for name, occupations in (
        ("bra_occupations", bra_occupations),
        ("ket_occupations", ket_occupations),
    ):
        if not isinstance(occupations, np.ndarray) or occupations.dtype != np.int64:
            raise TypeError(f"{name} must be an int64 NumPy array")
        if occupations.ndim != 2 or occupations.shape[0] == 0 or occupations.shape[1] != nsite:
            raise ValueError(f"{name} must have nonempty shape (nconfig,L)")
        if np.any(occupations < 0):
            raise ValueError(f"{name} must contain nonnegative occupations")
    str_a, link_a = basis.make_electron_basis(nsite, nelec[0])
    str_b, link_b = basis.make_electron_basis(nsite, nelec[1])
    sites = np.arange(nsite, dtype=np.int64)
    occ_a = (str_a[:, None] >> sites[None, :] & 1).astype(np.float64)
    occ_b = (str_b[:, None] >> sites[None, :] & 1).astype(np.float64)
    occupation = occ_a[:, None, :] + occ_b[None, :, :]
    eta_bra = shift_bra - np.einsum("xp,ijp->ijx", lam_bra, occupation)
    eta_ket = shift_ket - np.einsum("xp,ijp->ijx", lam_ket, occupation)
    create, annihilate, target, sign = link_a.transpose(2, 0, 1)
    source = np.broadcast_to(np.arange(len(str_a))[:, None], target.shape)
    hopping_a = np.zeros((len(str_a), len(str_a)))
    np.add.at(hopping_a, (target, source), sign * tmat[create, annihilate])
    create, annihilate, target, sign = link_b.transpose(2, 0, 1)
    source = np.broadcast_to(np.arange(len(str_b))[:, None], target.shape)
    hopping_b = np.zeros((len(str_b), len(str_b)))
    np.add.at(hopping_b, (target, source), sign * tmat[create, annihilate])
    identity_a = np.eye(len(str_a))
    identity_b = np.eye(len(str_b))
    hopping = np.einsum("ik,jl->ijkl", hopping_a, identity_b) + np.einsum(
        "ik,jl->ijkl", identity_a, hopping_b
    )
    H_hop = np.zeros(
        (
            str_a.shape[0],
            str_b.shape[0],
            bra_occupations.shape[0],
            str_a.shape[0],
            str_b.shape[0],
            ket_occupations.shape[0],
        )
    )
    for ia in range(str_a.shape[0]):
        for ib in range(str_b.shape[0]):
            for ja in range(str_a.shape[0]):
                for jb in range(str_b.shape[0]):
                    if hopping[ia, ib, ja, jb] != 0:
                        F = multimode_displacement_overlap(
                            eta_ket[ja, jb, :] - eta_bra[ia, ib, :],
                            bra_occupations,
                            ket_occupations,
                        )
                        H_hop[ia, ib, :, ja, jb, :] = hopping[ia, ib, ja, jb] * F
    return H_hop


def _validate_mr_candidate_inputs(
    tmat,
    U,
    g,
    omega,
    lam,
    shift,
    nelec,
    external_occupations,
):
    """Check raw LF candidate inputs without evaluating any model kernels.

    Arrays retain their exact float64/int64 dtype, values, row order and
    duplicates. Scalars are normalized to Python float; return
    (U, g, omega, nframe, nsite). Empty external lists are valid. No
    hopping or electron-phonon matrix elements are evaluated or allocated.
    """
    parameters = []
    for name, value in (("U", U), ("g", g), ("omega", omega)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real numeric scalar")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be finite in float64") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite in float64")
        parameters.append(value)
    U, g, omega = parameters
    if omega <= 0:
        raise ValueError("omega must be positive")

    for name, array in (("tmat", tmat), ("lam", lam), ("shift", shift)):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if (
        lam.ndim != 3
        or lam.shape[0] == 0
        or not 0 < lam.shape[1] < 64
        or lam.shape[1] != lam.shape[2]
    ):
        raise ValueError("lam must have shape (F,L,L) with F > 0 and 0 < L < 64")
    nframe, nsite, _ = lam.shape
    if shift.shape != (nframe, nsite):
        raise ValueError("shift must have shape (F,L)")
    if tmat.shape != (nsite, nsite):
        raise ValueError("tmat must have shape (L,L)")
    if not np.array_equal(tmat, tmat.T):
        raise ValueError("tmat must be symmetric")
    if (
        not isinstance(nelec, tuple)
        or len(nelec) != 2
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 0
            or n > nsite
            for n in nelec
        )
    ):
        raise ValueError("nelec must be (neleca,nelecb) with integer counts in [0,L]")
    if not isinstance(external_occupations, np.ndarray) or external_occupations.dtype != np.int64:
        raise TypeError("external_occupations must be an int64 NumPy array")
    if external_occupations.ndim != 2 or external_occupations.shape[1] != nsite:
        raise ValueError("external_occupations must have shape (nconfig,L)")
    if np.any(external_occupations < 0):
        raise ValueError("external_occupations must contain nonnegative occupations")
    if np.any(~np.any(external_occupations > 0, axis=1)):
        raise ValueError("external_occupations must not contain a vacuum configuration")

    return U, g, omega, nframe, nsite


def build_mr_candidate_blocks(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int],
    external_occupations: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble unprojected reference/candidate overlap and Hamiltonian blocks.

    Reference columns are |A,I> = |I>D(eta_A(I))|vacuum>; candidate
    columns are |A,I,k> = |I>D(eta_A(I))|external_occupations[k]>, with
    eta_A(I) = shift[A] - lam[A] @ density(I). Both sets use the same
    supplied frames and the complete fixed-electron site determinant basis.
    The physical Hamiltonian has site hopping, local Hubbard U, uncentered
    density-phonon coupling g, and common positive frequency omega, without
    a phonon zero-point constant or centered-to-paper energy correction.

    Parameters
    ----------
    tmat
        Finite exactly symmetric float64 array (L,L), indexed [creation
        site, annihilation site]. Site-diagonal entries are included once.
        No lattice, boundary condition, or bond multiplicity is inferred.
    U, g, omega
        Finite real numeric scalars; omega must be strictly positive.
        Boolean, complex, and array scalars are rejected. U and g may have
        either sign. Python and NumPy numeric scalars are accepted.
    lam
        Finite float64 array (F,L,L), F > 0 and 0 < L < 64, indexed
        [frame, phonon mode, electron site]. Duplicate frames are allowed.
    shift
        Finite float64 array (F,L), indexed [frame, phonon mode].
    nelec
        Tuple (neleca,nelecb) of integer counts in [0,L]; booleans are
        rejected. Determinant addresses follow make_electron_basis order.
    external_occupations
        Nonnegative int64 array (nconfig,L), each nonempty-list row having
        strictly positive total occupation. Rows are relative to each LF
        frame. Preserve their order and any duplicates; no enumeration,
        sorting, mode restriction, or rank truncation is performed here.
        An empty array with shape (0,L) is valid and gives zero candidates.
        A vacuum row is rejected because every frame's vacuum is already
        in the complete electronic reference space.

    Returns
    -------
    dict
        Exactly six float64 matrices, where d_e = nstra*nstrb,
        nr = F*d_e and nt = F*d_e*nconfig:

        * 's_rr', 'h_rr': (nr,nr), reference bra/reference ket.
        * 's_rt', 'h_rt': (nr,nt), reference bra/candidate ket.
        * 's_tt', 'h_tt': (nt,nt), candidate bra/candidate ket.

        Reference order is (frame,alpha address,beta address) in C order.
        Candidate order is (frame,alpha address,beta address,configuration)
        in C order. For R and T the column matrices of physical states,
        S blocks are R.T@R, R.T@T, T.T@T, and H blocks insert H between
        those vectors. Reverse cross blocks are obtained by transposition.
        nt == 0 yields (nr,0) cross and (0,0) candidate matrices.

    Raises
    ------
    TypeError
        For non-array inputs or wrong array dtypes, or non-real scalar
        parameter types.
    ValueError
        For invalid shapes, nonfinite parameters, nonsymmetric tmat,
        nonpositive omega, invalid electron counts, negative occupations,
        or any vacuum configuration in the candidate list.

    Notes
    -----
    Inputs must not be modified. This is a dense matrix boundary, so the
    frame-pair tensors are flattened only when placing their blocks into
    the matrices. Reuse noci.lf_multi_frame_overlap and
    noci.lf_multi_frame_hamiltonian for RR, or the equivalent vacuum
    frame-pair kernels. RT uses vacuum bra and excited ket; TT uses the
    supplied occupations on both ends. H frame-pair blocks are the sum
    of lf_frame_diagonal_hamiltonian and lf_frame_hopping_hamiltonian.
    Do not call nonempty-list phonon kernels when nconfig == 0.

    All six blocks are unprojected and use normalized displaced number
    states, without translation-projection scaling. Do not combine them
    with a projected RR block from an existing momentum-sector solver.
    No symmetry projection, reference subtraction, orthogonalization,
    nonlinear frame optimization, energy solve, or PT2 response occurs
    here. Candidates need not be orthogonal to the full multi-frame
    reference even though each has positive relative phonon occupation.
    """
    U, g, omega, nframe, nsite = _validate_mr_candidate_inputs(
        tmat, U, g, omega, lam, shift, nelec, external_occupations
    )

    str_a, _ = basis.make_electron_basis(nsite, nelec[0])
    str_b, _ = basis.make_electron_basis(nsite, nelec[1])
    ndet = len(str_a) * len(str_b)
    nconfig = len(external_occupations)
    candidates_per_frame = ndet * nconfig
    nr, nt = nframe * ndet, nframe * candidates_per_frame
    result = {
        "s_rr": noci.lf_multi_frame_overlap(lam, shift, nelec).reshape(nr, nr),
        "s_rt": np.zeros((nr, nt), dtype=np.float64),
        "s_tt": np.zeros((nt, nt), dtype=np.float64),
        "h_rr": noci.lf_multi_frame_hamiltonian(tmat, U, g, omega, lam, shift, nelec).reshape(
            nr, nr
        ),
        "h_rt": np.zeros((nr, nt), dtype=np.float64),
        "h_tt": np.zeros((nt, nt), dtype=np.float64),
    }
    if nconfig == 0:
        return result

    vacuum = np.zeros((1, nsite), dtype=np.int64)
    for A in range(nframe):
        reference_rows = slice(A * ndet, (A + 1) * ndet)
        candidate_rows = slice(A * candidates_per_frame, (A + 1) * candidates_per_frame)
        for B in range(nframe):
            candidate_columns = slice(B * candidates_per_frame, (B + 1) * candidates_per_frame)
            frame_args = (lam[A], shift[A], lam[B], shift[B], nelec)
            result["s_rt"][reference_rows, candidate_columns] = lf_frame_phonon_overlap(
                *frame_args, vacuum, external_occupations
            ).reshape(ndet, candidates_per_frame)
            h_rt = lf_frame_diagonal_hamiltonian(
                U, g, omega, *frame_args, vacuum, external_occupations
            ) + lf_frame_hopping_hamiltonian(tmat, *frame_args, vacuum, external_occupations)
            result["h_rt"][reference_rows, candidate_columns] = h_rt.reshape(
                ndet, candidates_per_frame
            )
            result["s_tt"][candidate_rows, candidate_columns] = lf_frame_phonon_overlap(
                *frame_args, external_occupations, external_occupations
            ).reshape(candidates_per_frame, candidates_per_frame)
            h_tt = lf_frame_diagonal_hamiltonian(
                U, g, omega, *frame_args, external_occupations, external_occupations
            ) + lf_frame_hopping_hamiltonian(
                tmat, *frame_args, external_occupations, external_occupations
            )
            result["h_tt"][candidate_rows, candidate_columns] = h_tt.reshape(
                candidates_per_frame, candidates_per_frame
            )
    return result


def build_projected_mr_candidate_blocks(
    tmat,
    U,
    g,
    omega,
    lam,
    shift,
    nelec,
    external_occupations,
    *,
    character=1,
):
    """Build six S/H blocks between equally translation-projected LF states.

    Parameters
    ----------
    tmat, U, g, omega, lam, shift, nelec, external_occupations
        Follow build_mr_candidate_blocks. Here finite float64 lam (F,L,L)
        and shift (F,L) specify orbit seeds, not the expanded orbit. tmat
        is exactly symmetric and invariant under simultaneous one-site
        row/column translation within absolute tolerance 1e-12. Coupling
        is uncentered, with no phonon zero-point constant. Occupations are
        int64 (N,L), nonnegative with no vacuum rows. Keep row order and
        duplicates; the list need not be closed under translation.
    character
        Integer +1, or -1 for even L; booleans are rejected.

    Returns
    -------
    dict
        Exactly six finite float64 arrays. Write D=nstra*nstrb,
        nr=F*D, nt=F*D*N. s_rr/h_rr have shape (nr,nr), s_rt/h_rt
        (nr,nt), and s_tt/h_tt (nt,nt). All are bra/ket ordered.
        Reference index is A*D + ia*nstrb + ib. Candidate index is
        (A*D + ia*nstrb + ib)*N + k. Flatten only in C order.
        N=0 is valid and yields (nr,0) and (0,0) arrays. The RR blocks
        match the existing direct translation-projected NOCI kernel.

    Raises
    ------
    TypeError, ValueError
        Same physical input checks as build_mr_candidate_blocks, plus
        ValueError for an invalid character or nonperiodic hopping.
        Validation itself must not evaluate model kernels. Inputs are
        never modified; read-only arrays are supported.

    Notes
    -----
    Each projected column is sum_r character**r*T**r|seed,I,n>/sqrt(L),
    exactly the scale used by lf_translation_projected_noci. The double
    sum reduces to one relative-translation sum, with no additional 1/L.
    For relative translation r, roll the ket lambda on both site axes,
    the ket shift on its mode axis, and ket occupations on axis 1. Gather
    the translated ket determinant on axis 2 of a kernel reshaped as
    (D,Nbra,D,Nket), then multiply by character**r*sign[r] on that axis.

    Write directly into the six result matrices using vacuum/vacuum,
    vacuum/excited, excited/vacuum and excited/excited frame-pair kernels.
    Never allocate a joint (nr+nt,nr+nt) S/H matrix or concatenate vacuum
    and excited lists. Temporary dense arrays span one frame pair only.
    Sum relative translations before placing each half-weighted pair
    and its transpose partner. This preserves the legacy Hermitian
    average: RR/TT average raw blocks with their transpose partners;
    RT averages raw RT with raw TR.T. No full TR matrix or full-size
    symmetrization copy is stored. Dense TT storage still scales as nt**2.

    No expanded-orbit matrices, reference exclusion, rank truncation,
    frame optimization, energy solve or perturbative response occur here.
    """
    U, g, omega, nframe, nsite = _validate_mr_candidate_inputs(
        tmat, U, g, omega, lam, shift, nelec, external_occupations
    )
    if (
        isinstance(character, (bool, np.bool_))
        or not isinstance(character, (int, np.integer))
        or character not in (1, -1)
        or (character == -1 and nsite % 2)
    ):
        raise ValueError("character must be +1, or -1 for even L")
    if not np.allclose(tmat, np.roll(tmat, 1, axis=(0, 1)), atol=1e-12, rtol=0):
        raise ValueError("tmat must be translation invariant")
    address, sign, _ = _translation_electron_labels(nsite, nelec)
    ndet = address.shape[1]
    occupations = {"r": np.zeros((1, nsite), dtype=np.int64), "t": external_occupations}
    frame_sizes = {key: ndet * len(value) for key, value in occupations.items()}
    nr, nt = nframe * ndet, nframe * frame_sizes["t"]
    result = {
        f"{kind}_{suffix}": np.zeros(shape, dtype=np.float64)
        for kind in ("s", "h")
        for suffix, shape in (("rr", (nr, nr)), ("rt", (nr, nt)), ("tt", (nt, nt)))
    }

    for bra, ket in (("r", "r"), ("r", "t"), ("t", "r"), ("t", "t")):
        bra_occ, ket_occ = occupations[bra], occupations[ket]
        nb, nk = len(bra_occ), len(ket_occ)
        if not nb or not nk:
            continue
        bra_size, ket_size = frame_sizes[bra], frame_sizes[ket]
        for a in range(nframe):
            rows = slice(a * bra_size, (a + 1) * bra_size)
            for b in range(nframe):
                cols = slice(b * ket_size, (b + 1) * ket_size)
                pair_s = np.zeros((bra_size, ket_size), dtype=np.float64)
                pair_h = np.zeros_like(pair_s)
                for r in range(nsite):
                    args = (
                        lam[a],
                        shift[a],
                        np.roll(lam[b], r, axis=(0, 1)),
                        np.roll(shift[b], r),
                        nelec,
                        bra_occ,
                        np.roll(ket_occ, r, axis=1),
                    )
                    weight = ((character**r) * sign[r])[None, None, :, None]
                    tensor = lf_frame_phonon_overlap(*args)
                    translated = np.take(tensor.reshape(ndet, nb, ndet, nk), address[r], axis=2)
                    translated *= weight
                    pair_s += translated.reshape(bra_size, ket_size)
                    del tensor, translated

                    tensor = lf_frame_diagonal_hamiltonian(U, g, omega, *args)
                    tensor += lf_frame_hopping_hamiltonian(tmat, *args)
                    translated = np.take(tensor.reshape(ndet, nb, ndet, nk), address[r], axis=2)
                    translated *= weight
                    pair_h += translated.reshape(bra_size, ket_size)
                    del tensor, translated

                # Distribute raw AB/2 and its transpose without storing TR
                # globally or making a second full-size RR/TT matrix.
                for kind, pair in (("s", pair_s), ("h", pair_h)):
                    pair *= 0.5
                    if bra == ket:
                        target = result[f"{kind}_{bra + ket}"]
                        target[rows, cols] += pair
                        target[cols, rows] += pair.T
                    elif bra == "r":
                        result[f"{kind}_rt"][rows, cols] += pair
                    else:
                        result[f"{kind}_rt"][cols, rows] += pair.T
                del pair_s, pair_h, pair
    return result
