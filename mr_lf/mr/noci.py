"""Analytic LF frame kernels and nonorthogonal configuration interaction."""

import numpy as np
from scipy import linalg as scipy_linalg

from .. import basis
from .._driver import Calculation, require_reference
from ..symmetry import lf_translation_orbit


def lf_frame_overlap(
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int] = (1, 0),
) -> np.ndarray:
    """Return the physical-state overlap for one-electron LF frames.

    Frames use real conditional coherent displacements
    ``eta[A, x, p] = shift[A, x] - lam[A, x, p]``.  The result is
    ``S[A, p, B, q] = delta[p, q] * exp(-||eta[A, :, p] -
    eta[B, :, q]||**2 / 2)``.  Different frames may be linearly
    dependent, so the flattened overlap need not be invertible.

    Parameters
    ----------
    lam
        Finite float64 LF parameters with shape ``(K, L, L)`` and
        index order ``[frame, phonon mode, electron site]``.
    shift
        Finite float64 coherent shifts with shape ``(K, L)``
        and index order ``[frame, phonon mode]``.
    nelec
        Reserved for a future electron-sector extension.  This
        implementation describes only the single-alpha-electron
        site basis and does not inspect this argument.

    Returns
    -------
    numpy.ndarray
        Float64 overlap tensor with shape ``(K, L, K, L)`` and
        index order ``[A, p, B, q]``.  At a solver boundary it can
        be reshaped to ``(K*L, K*L)`` with ``a = A*L + p``.

    Raises
    ------
    ValueError
        If an array has incompatible or empty dimensions, or
        contains nonfinite values.
    TypeError
        If an array does not have float64 dtype.
    """
    if lam.ndim != 3 or lam.shape[1] != lam.shape[2]:
        raise ValueError("lam must be a 3D array with shape (n, m, m)")
    K, L, _ = lam.shape
    if K <= 0 or L <= 0:
        raise ValueError("lam must have positive dimensions")
    if shift.ndim != 2 or shift.shape != (K, L):
        raise ValueError("shift must be a 2D array with shape (n, m)")
    if lam.dtype != np.float64 or shift.dtype != np.float64:
        raise TypeError("lam and shift must have float64 dtype")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(shift)):
        raise ValueError("lam and shift must contain only finite values")
    eta = shift[:, :, None] - lam
    S = np.zeros((K, L, K, L), dtype=np.float64)
    for A in range(K):
        for B in range(K):
            for p in range(L):
                S[A, p, B, p] = np.exp(-np.sum((eta[A, :, p] - eta[B, :, p]) ** 2) / 2)
    return S


def lf_frame_hamiltonian(
    tmat: np.ndarray,
    g: float,
    omega: float,
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int] = (1, 0),
) -> np.ndarray:
    """Return the physical Hamiltonian kernel between one-electron LF frames.

    With ``eta[A, x, p] = shift[A, x] - lam[A, x, p]``, each element
    ``H[A, p, B, q]`` is the matrix element of the uncentered
    Hubbard-Holstein Hamiltonian between ``|p>|eta[A, :, p]>`` and
    ``|q>|eta[B, :, q]>``.  The phonon energy sums over every mode;
    the local electron-phonon coupling uses only mode ``p`` when
    ``p == q``.  The boson Hamiltonian has no zero-point constant.

    Parameters
    ----------
    tmat
        Finite real symmetric hopping matrix with float64 dtype and
        shape ``(L, L)``, indexed as ``[p, q]``.  Diagonal entries
        are allowed.
    g
        Finite real local electron-phonon coupling.
    omega
        Finite positive phonon frequency shared by the L modes.
    lam
        Finite float64 LF parameters with shape ``(K, L, L)`` and
        index order ``[frame, phonon mode, electron site]``.
    shift
        Finite float64 coherent shifts with shape ``(K, L)`` and
        index order ``[frame, phonon mode]``.
    nelec
        Reserved for a future electron-sector extension.  This
        implementation describes only the single-alpha-electron
        site basis and does not inspect this argument.

    Returns
    -------
    numpy.ndarray
        Float64 Hamiltonian tensor with shape ``(K, L, K, L)`` and
        index order ``[A, p, B, q]``.  At a solver boundary it can
        be reshaped to ``(K*L, K*L)`` with ``a = A*L + p``.  This
        matrix is paired with the nonorthogonal overlap from
        ``lf_frame_overlap``.

    Raises
    ------
    ValueError
        If dimensions are incompatible or empty, arrays contain
        nonfinite values, ``tmat`` is not symmetric, or the scalar
        parameters violate their finite real and positive-frequency
        requirements.
    TypeError
        If an array does not have float64 dtype.
    """
    if lam.ndim != 3 or lam.shape[1] != lam.shape[2]:
        raise ValueError("lam must be a 3D array with shape (n, m, m)")
    K, L, _ = lam.shape
    if K <= 0 or L <= 0:
        raise ValueError("lam must have positive dimensions")
    if shift.ndim != 2 or shift.shape != (K, L):
        raise ValueError("shift must be a 2D array with shape (n, m)")
    if tmat.ndim != 2 or tmat.shape != (L, L):
        raise ValueError("tmat must be a 2D array with shape (m, m)")
    if lam.dtype != np.float64 or shift.dtype != np.float64 or tmat.dtype != np.float64:
        raise TypeError("lam, shift, and tmat must have float64 dtype")
    if (
        not np.all(np.isfinite(lam))
        or not np.all(np.isfinite(shift))
        or not np.all(np.isfinite(tmat))
    ):
        raise ValueError("lam, shift, and tmat must contain only finite values")
    if not np.array_equal(tmat, tmat.T):
        raise ValueError("tmat must be symmetric")
    if not np.isfinite(g) or not np.isreal(g):
        raise ValueError("g must be a finite real number")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("omega must be a finite real number greater than zero")
    eta = shift[:, :, None] - lam
    H = np.zeros((K, L, K, L), dtype=np.float64)
    for A in range(K):
        for B in range(K):
            for p in range(L):
                for q in range(L):
                    s = np.exp(-np.sum((eta[A, :, p] - eta[B, :, q]) ** 2) / 2)
                    H[A, p, B, q] += s * tmat[p, q]
                    if p == q:
                        H[A, p, B, q] += (
                            omega * np.sum(eta[A, :, p] * eta[B, :, p]) * s
                            + g * (eta[A, p, p] + eta[B, p, p]) * s
                        )
    return H


def lf_noci_lowest(
    hmat: np.ndarray,
    smat: np.ndarray,
    *,
    overlap_cut: float = 1e-10,
    condition_limit: float = 1e8,
) -> tuple[float, np.ndarray, int]:
    """Solve a single- or multi-electron NOCI generalized eigenproblem.

    Kernels are flattened in C order only at this solver boundary. The
    basis address is A*L+p for four-index input, or
    (A*nstra+ia)*nstrb+ib for six-index input. Overlap eigenvectors are
    retained when their eigenvalue exceeds overlap_cut * max(1, s_max).
    If all directions survive and s_max / s_min <= condition_limit,
    scipy.linalg.eigh solves H @ c = E * S @ c directly. Otherwise,
    canonical orthogonalization constructs X with X.T @ S @ X = I
    and solves in the retained span. The electronic and phonon model
    is entirely specified by the supplied H and S kernels.

    Parameters
    ----------
    hmat
        Finite symmetric float64 Hamiltonian kernel. Supported shapes are
        (K, L, K, L), with indices [A, p, B, q], and
        (K, nstra, nstrb, K, nstra, nstrb), with indices
        [A, ia, ib, B, ja, jb]. All dimensions must be positive.
    smat
        Finite symmetric float64 Gram kernel with exactly the same shape
        and ordering as hmat. Repeated frames may make it singular.
    overlap_cut
        Finite positive real scalar controlling retained overlap rank.
        Substantially negative overlap eigenvalues are rejected using
        the same threshold as in the existing single-electron solver.
    condition_limit
        Finite real scalar at least 1, default 1e8. Largest overlap
        condition number permitted for the direct generalized solve.
        This selects the solver path only; overlap_cut alone determines
        which directions are retained.

    Returns
    -------
    energy
        Lowest variational energy in the retained span, as a Python float.
    coeff
        Float64 CI coefficients with shape (K, L) for four-index input or
        (K, nstra, nstrb) for six-index input. They obey
        coeff.ravel() @ S @ coeff.ravel() = 1. Their overall sign is
        arbitrary; individual squared entries are not probabilities in
        this nonorthogonal basis.
    rank
        Number of retained overlap eigenvectors, as a Python int.

    Raises
    ------
    ValueError
        If shapes, finite values, symmetry, cutoff, condition limit, or
        the overlap spectrum violate the solver contract.
    TypeError
        If either kernel is not a NumPy array with float64 dtype.

    Notes
    -----
    No input arrays are modified. After overlap truncation, the relevant
    generalized residual is projected into the retained subspace. The
    full coordinate residual can be larger. This is a dense solver: its
    matrix dimension is K*L or K*nstra*nstrb. No lattice size, electron
    count, coupling convention, or model parameter is hard-coded here.
    The condition limit is a solver selection policy, not a guarantee of
    accuracy for ill-conditioned inputs. Canonical orthogonalization
    without truncation does not remove their intrinsic ill-conditioning.
    """
    for name, array in (("hmat", hmat), ("smat", smat)):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if array.ndim not in (4, 6):
            raise ValueError(f"{name} must be a four- or six-index kernel")
        half = array.ndim // 2
        if any(size == 0 for size in array.shape) or array.shape[:half] != array.shape[half:]:
            raise ValueError(f"{name} must have matching nonempty bra and ket shapes")
        if array.dtype != np.float64:
            raise TypeError(f"{name} must have float64 dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if smat.shape != hmat.shape:
        raise ValueError("smat must have the same shape as hmat")
    if (
        isinstance(overlap_cut, (bool, np.bool_))
        or not isinstance(overlap_cut, (int, float, np.integer, np.floating))
        or not np.isfinite(overlap_cut)
        or overlap_cut <= 0
    ):
        raise ValueError("overlap_cut must be finite, real, and positive")
    if (
        isinstance(condition_limit, (bool, np.bool_))
        or not isinstance(condition_limit, (int, float, np.integer, np.floating))
        or not np.isfinite(condition_limit)
        or condition_limit < 1
    ):
        raise ValueError("condition_limit must be finite, real, and at least 1")

    basis_shape = hmat.shape[: hmat.ndim // 2]
    dimension = int(np.prod(basis_shape))
    hmat_flat = hmat.reshape(dimension, dimension)
    smat_flat = smat.reshape(dimension, dimension)
    if not np.array_equal(hmat_flat, hmat_flat.T) or not np.array_equal(smat_flat, smat_flat.T):
        raise ValueError("hmat and smat must be symmetric")

    s, vectors = np.linalg.eigh(smat_flat)
    threshold = max(float(s[-1]), 1.0) * overlap_cut
    if s[0] < -threshold:
        raise ValueError("smat has eigenvalues below the allowed overlap_cut threshold")
    keep = s > threshold
    if not np.any(keep):
        raise ValueError("No eigenvalues of smat are above the allowed overlap_cut threshold")
    rank = int(np.count_nonzero(keep))
    if rank == dimension and float(s[-1]) / float(s[0]) <= condition_limit:
        energies, eigenvectors = scipy_linalg.eigh(
            hmat_flat,
            smat_flat,
            subset_by_index=[0, 0],
        )
        coeff = eigenvectors[:, 0].reshape(basis_shape)
        return float(energies[0]), coeff, rank
    X = vectors[:, keep] / np.sqrt(s[keep])
    energies, eigenvectors = np.linalg.eigh(X.T @ hmat_flat @ X)
    coeff = (X @ eigenvectors[:, 0]).reshape(basis_shape)
    return float(energies[0]), coeff, rank


def _validate_multi_frame_inputs(
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int],
) -> tuple[int, int]:
    """Validate multi-electron LF frames and return ``(nframe, nsite)``.

    ``lam[A, x, p]`` has frame, phonon-mode, and electron-site indices;
    ``shift[A, x]`` has frame and phonon-mode indices. Both are finite
    float64 arrays. ``nelec`` gives fixed alpha and beta electron counts.
    This helper performs no model contraction and does not alter inputs.
    """
    if (
        not isinstance(lam, np.ndarray)
        or lam.ndim != 3
        or lam.shape[0] == 0
        or lam.shape[1] == 0
        or lam.shape[1] != lam.shape[2]
    ):
        raise ValueError("lam must have nonempty shape (K, L, L)")
    nframe, nsite, _ = lam.shape
    if not isinstance(shift, np.ndarray) or shift.shape != (nframe, nsite):
        raise ValueError("shift must have shape (K, L)")
    if lam.dtype != np.float64 or shift.dtype != np.float64:
        raise TypeError("lam and shift must have float64 dtype")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(shift)):
        raise ValueError("lam and shift must contain only finite values")
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
        raise ValueError("nelec must be (neleca, nelecb), each between 0 and L")
    return nframe, nsite


def lf_multi_frame_overlap(
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int],
) -> np.ndarray:
    """Return the site-determinant Gram tensor of multi-electron LF frames.

    A frame consists of an orthonormal site determinant and its conditional
    multimode coherent state. The local coupling is uncentered. The alpha
    and beta determinant addresses follow ``make_electron_basis`` order.

    Parameters
    ----------
    lam
        Full LF matrices with shape ``(K, L, L)`` and index order
        ``[frame A, phonon mode x, electron site p]``; finite float64.
    shift
        Coherent shifts with shape ``(K, L)`` and order ``[A, x]``;
        finite float64.
    nelec
        Fixed electron counts ``(neleca, nelecb)`` with each count from
        zero through L.

    Returns
    -------
    numpy.ndarray
        Real Gram tensor with shape ``(K, nstra, nstrb, K, nstra, nstrb)``
        and index order ``[A, ia, ib, B, ja, jb]``, where
        ``nstr_sigma = comb(L, nelec_sigma)``. The tensor retains separate
        alpha and beta determinant axes. Its diagonal frame blocks are
        identity matrices on the site determinant space.

    Notes
    -----
    This is the multi-electron extension of ``lf_frame_overlap``. It is
    independent of the phonon Fock cutoff. The corresponding Hamiltonian
    kernel will use the same conditional coherent displacements.
    """
    _, nsite = _validate_multi_frame_inputs(lam, shift, nelec)
    str_a, _ = basis.make_electron_basis(nsite, nelec[0])
    str_b, _ = basis.make_electron_basis(nsite, nelec[1])

    sites = np.arange(nsite, dtype=np.int64)

    occ_a = (str_a[:, None] >> sites[None, :] & 1).astype(np.float64)
    occ_b = (str_b[:, None] >> sites[None, :] & 1).astype(np.float64)
    occupation = occ_a[:, None, :] + occ_b[None, :, :]

    eta = shift[:, None, None, :] - np.einsum("Axp,ijp->Aijx", lam, occupation)

    difference = eta[:, None, :, :, :] - eta[None, :, :, :, :]
    same_det_overlap = np.exp(-0.5 * np.sum(difference**2, axis=-1))
    identity_a = np.eye(len(str_a), dtype=np.float64)
    identity_b = np.eye(len(str_b), dtype=np.float64)

    smat = np.einsum("ABij,ik,jl->AijBkl", same_det_overlap, identity_a, identity_b)
    return smat


def lf_multi_frame_hamiltonian(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int],
) -> np.ndarray:
    """Return the physical Hamiltonian kernel between multi-electron LF frames.

    Uses the uncentered coupling g*n[x]*(b[x] + b[x].dagger), local Hubbard
    interaction U*n_alpha[x]*n_beta[x], and omega*b[x].dagger*b[x] without
    a zero-point constant. The result already uses the paper energy
    convention; no centered-to-paper energy correction is added.

    Parameters
    ----------
    tmat
        Finite float64 hopping matrix with shape (L, L), indexed by
        electron sites [p, q]. Must be exactly symmetric; diagonal entries
        are allowed. No particular lattice or boundary condition is assumed.
    U, g, omega
        Finite real interaction, electron-phonon coupling, and positive
        common phonon frequency.
    lam
        Finite float64 LF matrices with shape (K, L, L) and index order
        [frame A, phonon mode x, electron site p].
    shift
        Finite float64 coherent shifts with shape (K, L), ordered [A, x].
    nelec
        Fixed electron counts (neleca, nelecb), each between zero and L.

    Returns
    -------
    numpy.ndarray
        Float64 tensor of shape (K, nstra, nstrb, K, nstra, nstrb), ordered
        [A, ia, ib, B, ja, jb], matching lf_multi_frame_overlap. Electronic
        determinant addresses use basis.make_electron_basis order.

    Notes
    -----
    Each basis ket is a site determinant times its conditional normalized
    coherent state. Coherent-state matrix elements are evaluated in the
    complete phonon space, so there is no Nmax parameter. Hopping changes
    the electron determinant: its coherent overlap must be retained even
    when the full state overlap from lf_multi_frame_overlap is zero.
    The resulting frame space is variational and need not span the exact
    electron-phonon ground state.
    """
    _nframe, nsite = _validate_multi_frame_inputs(lam, shift, nelec)
    if not isinstance(tmat, np.ndarray) or tmat.shape != (nsite, nsite):
        raise ValueError("tmat must have shape (L, L)")
    if tmat.dtype != np.float64:
        raise TypeError("tmat must have float64 dtype")
    if not np.all(np.isfinite(tmat)):
        raise ValueError("tmat must contain only finite values")
    if not np.array_equal(tmat, tmat.T):
        raise ValueError("tmat must be symmetric")
    for name, value in (("U", U), ("g", g), ("omega", omega)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite real scalar")
    if omega <= 0:
        raise ValueError("omega must be positive")
    str_a, link_a = basis.make_electron_basis(nsite, nelec[0])
    str_b, link_b = basis.make_electron_basis(nsite, nelec[1])

    sites = np.arange(nsite, dtype=np.int64)

    occ_a = (str_a[:, None] >> sites[None, :] & 1).astype(np.float64)
    occ_b = (str_b[:, None] >> sites[None, :] & 1).astype(np.float64)
    occupation = occ_a[:, None, :] + occ_b[None, :, :]

    eta = shift[:, None, None, :] - np.einsum("Axp,ijp->Aijx", lam, occupation)

    double_occ = occ_a @ occ_b.T

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

    eta_dot = np.einsum("Aijx,Bklx->AijBkl", eta, eta)
    eta_norm2 = np.sum(eta**2, axis=-1)

    distance2 = (
        eta_norm2[:, :, :, None, None, None] + eta_norm2[None, None, None, :, :, :] - 2 * eta_dot
    )
    G = np.exp(-0.5 * distance2)
    H_hop = hopping[None, :, :, None, :, :] * G
    same_eta_dot = np.einsum("Aijx,Bijx->ABij", eta, eta)
    density_eta = np.einsum("ijx,Aijx->Aij", occupation, eta)
    V = (
        U * double_occ
        + omega * same_eta_dot
        + g * (density_eta[:, None, :, :] + density_eta[None, :, :, :])
    )
    diagonal_tensor = np.einsum("ABij,ik,jl->AijBkl", V, identity_a, identity_b)
    smat = lf_multi_frame_overlap(lam, shift, nelec)
    return H_hop + diagonal_tensor * smat


def lf_translation_projected_noci(
    tmat,
    U,
    g,
    omega,
    lam,
    shift,
    nelec,
    *,
    character=1,
    electronic_coeff=None,
    overlap_cut=1e-10,
    backend="direct",
    return_projection=False,
    condition_limit=1e8,
):
    """Solve NOCI between translation-projected LF frame families.

    lam: float64 (F,L,L), shift: float64 (F,L), in existing kernel order.
    nelec: (na,nb). Each seed generates L translated frames. Translation
    moves site p to (p+R)%L and reorders each spin determinant with its
    fermionic sign. character=+1 projects to momentum zero; -1 projects
    to momentum pi and requires even L. Other momenta are not implemented.

    Normally all electronic configurations remain independent: projected
    coefficients have shape (F,nstra,nstrb). Optional finite float64
    electronic_coeff of that same shape fixes one electronic wavefunction
    per seed, giving one projected state per seed and coefficients (F,).
    No normalization of individual projected states is assumed; S supplies
    the physical metric. A projected state may vanish and is rank truncated.

    Returns a dict with energy, rank, coeff, expanded_coeff (F*L,na,nb),
    lam/shift for the full orbit, optional projection P (D,M), projected hmat/smat
    (M,M), norm_error and residual_projected. H/S are built with existing
    multi kernels (backend="full") or relative translations (default
    backend="direct"), then solved by lf_noci_lowest. The direct backend
    never builds full orbit H/S. projection is None unless return_projection
    is True; expanded_coeff is constructed without a dense P. Both backends
    use the same roundoff symmetrization and overlap cutoff.
    D=F*L*nstra*nstrb. With full electronic CI, the direct path stores
    O((F*nstra*nstrb)**2) matrix elements; the full path stores
    O((F*L*nstra*nstrb)**2). Inputs are not modified. Translation-invariant tmat
    is required. This is projection of wavefunctions, not averaging lambda.
    """
    F, L = _validate_multi_frame_inputs(lam, shift, nelec)
    if (
        isinstance(character, (bool, np.bool_))
        or not isinstance(character, (int, np.integer))
        or character not in (1, -1)
    ):
        raise ValueError("character must be +1 or -1")
    if character == -1 and L % 2:
        raise ValueError("character=-1 requires even L")
    if (
        not isinstance(tmat, np.ndarray)
        or tmat.shape != (L, L)
        or not np.allclose(tmat, np.roll(tmat, 1, axis=(0, 1)), atol=1e-12, rtol=0)
    ):
        raise ValueError("tmat must be invariant under a one-site translation")
    strings = [basis.make_electron_basis(L, n)[0] for n in nelec]
    na, nb = map(len, strings)
    maps, signs = [], []
    for spin_strings, n in zip(strings, nelec):
        occ = (spin_strings[:, None] >> np.arange(L)) & 1
        mask = (1 << L) - 1
        shifted = np.stack(
            [((spin_strings << R) & mask) | (spin_strings >> (L - R)) for R in range(L)]
        )
        maps.append(np.searchsorted(spin_strings, shifted))
        wraps = np.stack(
            [occ[:, L - R :].sum(axis=1) if R else np.zeros(len(occ), dtype=int) for R in range(L)]
        )
        signs.append(np.where((wraps * (n - wraps)) % 2, -1.0, 1.0))
    orbit = [lf_translation_orbit(lam[A], shift[A]) for A in range(F)]
    orbit_lam = np.concatenate([o[0] for o in orbit])
    orbit_shift = np.concatenate([o[1] for o in orbit])
    d = na * nb
    if backend not in ("direct", "full"):
        raise ValueError("backend must be direct or full")
    if not isinstance(return_projection, (bool, np.bool_)):
        raise TypeError("return_projection must be boolean")
    address = (maps[0][:, :, None] * nb + maps[1][:, None, :]).reshape(L, d)
    sign = (signs[0][:, :, None] * signs[1][:, None, :]).reshape(L, d)
    shape = (F, na, nb)
    if electronic_coeff is not None:
        if (
            not isinstance(electronic_coeff, np.ndarray)
            or electronic_coeff.shape != shape
            or electronic_coeff.dtype != np.float64
            or not np.all(np.isfinite(electronic_coeff))
        ):
            raise ValueError("electronic_coeff must be finite float64 (F,nstra,nstrb)")
        shape = (F,)
    P = None
    if backend == "full" or return_projection:
        P = np.zeros((F * L * d, F * d), dtype=np.float64)
        for A in range(F):
            rows = (A * L + np.arange(L)[:, None]) * d + address
            P[rows, A * d + np.arange(d)[None, :]] = (
                (character ** np.arange(L))[:, None] * sign / np.sqrt(L)
            )
        if electronic_coeff is not None:
            Q = np.zeros((F * d, F))
            Q[np.arange(F * d), np.repeat(np.arange(F), d)] = electronic_coeff.ravel()
            P = P @ Q
    if backend == "full":
        H = lf_multi_frame_hamiltonian(tmat, U, g, omega, orbit_lam, orbit_shift, nelec)
        S = lf_multi_frame_overlap(orbit_lam, orbit_shift, nelec)
        h = P.T @ H.reshape(F * L * d, F * L * d) @ P
        s = P.T @ S.reshape(F * L * d, F * L * d) @ P
    else:
        h, s = _lf_relative_translation_kernels(
            tmat, U, g, omega, lam, shift, nelec, address, sign, character, electronic_coeff
        )
    h, s = (h + h.T) / 2, (s + s.T) / 2
    M = len(h)
    energy, c, rank = lf_noci_lowest(
        h.reshape(M, 1, M, 1),
        s.reshape(M, 1, M, 1),
        overlap_cut=overlap_cut,
        condition_limit=condition_limit,
    )
    c = c.ravel()
    eig, v = np.linalg.eigh(s)
    keep = eig > overlap_cut * max(1.0, eig[-1])
    X = v[:, keep] / np.sqrt(eig[keep])
    seed_coeff = (
        c.reshape(F, d) if electronic_coeff is None else c[:, None] * electronic_coeff.reshape(F, d)
    )
    expanded = np.empty((F, L, d), dtype=np.float64)
    for R in range(L):
        expanded[:, R, address[R]] = seed_coeff * (character**R) * sign[R] / np.sqrt(L)
    return {
        "energy": energy,
        "rank": rank,
        "coeff": c.reshape(shape),
        "expanded_coeff": expanded.reshape(F * L, na, nb),
        "lam": orbit_lam,
        "shift": orbit_shift,
        "projection": P if return_projection else None,
        "hmat": h,
        "smat": s,
        "norm_error": float(abs(c @ s @ c - 1)),
        "residual_projected": float(np.linalg.norm(X.T @ (h @ c - energy * s @ c))),
    }


def _lf_relative_translation_kernels(
    tmat,
    U,
    g,
    omega,
    lam,
    shift,
    nelec,
    address,
    sign,
    character,
    electronic_coeff,
):
    """Accumulate projected matrices one relative translation at a time.

    Matrix boundary indices are (seed, alpha address, beta address), C order.
    All arrays are real float64. A block has shape (F,d,F,d), d=na*nb;
    the full (F*L*d)**2 orbit matrices are never allocated. Fixed-electronic
    states are contracted per translation to (F,F) before accumulation.
    Physical matrix elements are identical to lf_multi_frame_hamiltonian.
    """
    F, L, _ = lam.shape
    # Reuse all physical input validation on a single seed (small d x d).
    lf_multi_frame_hamiltonian(tmat, U, g, omega, lam[:1], shift[:1], nelec)
    bases = [basis.make_electron_basis(L, n) for n in nelec]
    occ = [((ss[:, None] >> np.arange(L)) & 1).astype(np.float64) for ss, _ in bases]
    occupation = (occ[0][:, None, :] + occ[1][None, :, :]).reshape(-1, L)
    double = (occ[0] @ occ[1].T).ravel()
    d = len(occupation)
    hops = []
    for ss, link in bases:
        create, annihilate, target, parity = link.transpose(2, 0, 1)
        source = np.broadcast_to(np.arange(len(ss))[:, None], target.shape)
        hop = np.zeros((len(ss), len(ss)))
        np.add.at(hop, (target, source), parity * tmat[create, annihilate])
        hops.append(hop)
    hopping = np.kron(hops[0], np.eye(len(occ[1]))) + np.kron(np.eye(len(occ[0])), hops[1])
    eta = shift[:, None, :] - np.einsum("Axp,ip->Aix", lam, occupation)
    norms = np.sum(eta**2, axis=-1)
    density_eta = np.einsum("ix,Aix->Ai", occupation, eta)
    size = F * d if electronic_coeff is None else F
    h, s = np.zeros((size, size)), np.zeros((size, size))
    fixed = None if electronic_coeff is None else electronic_coeff.reshape(F, d)
    for R in range(L):
        ket_eta = np.roll(eta, R, axis=-1)
        dot = np.einsum("Aix,Bjx->AiBj", eta, ket_eta)
        gaussian = np.exp(-0.5 * (norms[:, :, None, None] + norms[None, None, :, :] - 2 * dot))
        electronic_overlap = np.eye(d)[:, address[R]]
        weight = (character**R) * sign[R]
        overlap = gaussian * electronic_overlap[None, :, None, :] * weight[None, None, None, :]
        potential = (
            U * double[None, :, None, None]
            + omega * dot
            + g
            * (density_eta[:, :, None, None] + np.einsum("ix,Bjx->iBj", occupation, ket_eta)[None])
        )
        ham = (
            gaussian * hopping[:, address[R]][None, :, None, :] * weight[None, None, None, :]
            + potential * overlap
        )
        if fixed is None:
            h += ham.reshape(size, size)
            s += overlap.reshape(size, size)
        else:
            h += np.einsum("Ai,AiBj,Bj->AB", fixed, ham, fixed, optimize=True)
            s += np.einsum("Ai,AiBj,Bj->AB", fixed, overlap, fixed, optimize=True)
    return h, s


class LFMR(Calculation):
    """Full electronic CI per supplied LF frame; fixed frame parameters.

    Molecular/integral models use optional joint symmetry and irrep labels.
    Lattice character=None: unprojected frames; +1/-1: q=0/pi projection.
    kernel() varies only CI coefficients and returns the lowest MR energy.
    """

    def __init__(
        self, model, lam, shift=None, *, character=None, symmetry=None, irrep=None,
        row=0, determinants=None, overlap_cut=1e-10, condition_limit=1e8
    ):
        super().__init__(model)
        from ..model import ElectronPhononModel

        self.symmetry, self.irrep, self.row = symmetry, irrep, row
        self.determinants = None if determinants is None else np.asarray(determinants).copy()
        self._integral_model = isinstance(model, ElectronPhononModel)
        if character is not None and symmetry is not None:
            raise ValueError("character and symmetry cannot be specified together")
        if self._integral_model:
            from .general_kernels import _validate_inputs

            if character is not None:
                raise ValueError("character is a lattice-only translation projection")
            if (symmetry is None) != (irrep is None):
                raise ValueError("symmetry and irrep must be specified together")
            if symmetry is None and row != 0:
                raise ValueError("row is available only with a point-group projection")
            shift = _validate_inputs(model, lam, shift, determinants)
            self.lam, self.shift = lam.copy(), shift.copy()
            self.character, self.overlap_cut = None, overlap_cut
            self.condition_limit = condition_limit
            return
        if symmetry is not None or irrep is not None or determinants is not None or row != 0:
            raise ValueError("molecular symmetry/determinants require an integral model")
        self.lam = np.asarray(lam).copy()
        if shift is None:
            shift = np.zeros((len(self.lam), model.L), dtype=np.float64)
        self.shift = np.asarray(shift).copy()
        _validate_multi_frame_inputs(self.lam, self.shift, model.nelec)
        if self.lam.shape[1:] != (model.L, model.L):
            raise ValueError("frame sites must match the model")
        if character is not None and (
            isinstance(character, (bool, np.bool_))
            or not isinstance(character, (int, np.integer))
            or character not in (-1, 1)
        ):
            raise ValueError("character must be None, +1 or -1")
        self.character, self.overlap_cut = character, overlap_cut
        self.condition_limit = condition_limit

    def kernel(self):
        self._reset()
        m = self.model
        if self._integral_model:
            from .general_kernels import lf_vacuum_frame_blocks
            from .hermitian import hermitian_noci_result

            if self.symmetry is None:
                blocks = lf_vacuum_frame_blocks(
                    m, self.lam, self.shift, determinants=self.determinants)
                h, s = blocks["hmat"], blocks["smat"]
            else:
                from .point_group_kernels import lf_group_frame_blocks

                h, s = lf_group_frame_blocks(
                    m, self.lam, self.shift, symmetry=self.symmetry, irrep=self.irrep,
                    row=self.row, determinants=self.determinants)
            result = hermitian_noci_result(
                h, s, overlap_cut=self.overlap_cut, condition_limit=self.condition_limit)
            result.update(lam=self.lam.copy(), shift=self.shift.copy(), irrep=self.irrep,
                          row=self.row, determinants=(None if self.determinants is None
                                                    else self.determinants.copy()))
            if self.symmetry is not None:
                d = self.symmetry.group.irrep_matrices[self.irrep].shape[-1]
                result.update(group=self.symmetry.group.name, irrep_dimension=d,
                              partner_rows=tuple(range(d)),
                              partner_convention="same CI coefficients, P[row,n] for each row")
            ok = result["norm_error"] < 1e-10 and result["residual_projected"] < 1e-10
            return self._finish(result, result["energy"], ok)
        if self.character is None:
            h = lf_multi_frame_hamiltonian(m.tmat, m.U, m.g, m.omega, self.lam, self.shift, m.nelec)
            s = lf_multi_frame_overlap(self.lam, self.shift, m.nelec)
            energy, c, rank = lf_noci_lowest(
                h, s, overlap_cut=self.overlap_cut, condition_limit=self.condition_limit
            )
            flat, hm, sm = c.ravel(), h.reshape(c.size, c.size), s.reshape(c.size, c.size)
            vals, vecs = np.linalg.eigh(sm)
            keep = vals > self.overlap_cut * max(1.0, vals[-1])
            x = vecs[:, keep] / np.sqrt(vals[keep])
            spectrum = np.linalg.eigvalsh(x.T @ hm @ x)
            result = {
                "energy": energy,
                "coeff": c,
                "rank": rank,
                "hmat": hm,
                "smat": sm,
                "lam": self.lam.copy(),
                "shift": self.shift.copy(),
                "expanded_coeff": c,
                "norm_error": float(abs(flat @ sm @ flat - 1)),
                "residual_projected": float(np.linalg.norm(x.T @ (hm @ flat - energy * sm @ flat))),
                "reference_eigenvalues": spectrum,
            }
        else:
            result = lf_translation_projected_noci(
                m.tmat,
                m.U,
                m.g,
                m.omega,
                self.lam,
                self.shift,
                m.nelec,
                character=self.character,
                overlap_cut=self.overlap_cut,
                condition_limit=self.condition_limit,
            )
            vals, vecs = np.linalg.eigh(result["smat"])
            keep = vals > self.overlap_cut * max(1.0, vals[-1])
            x = vecs[:, keep] / np.sqrt(vals[keep])
            result["reference_eigenvalues"] = np.linalg.eigvalsh(x.T @ result["hmat"] @ x)
        spectrum = result["reference_eigenvalues"]
        result["reference_gap"] = (
            float(spectrum[1] - spectrum[0]) if len(spectrum) > 1 else float("inf")
        )
        result["near_degenerate"] = bool(result["reference_gap"] <= 1e-8)
        ok = result["norm_error"] < 1e-9 and result["residual_projected"] < 1e-10
        return self._finish(result, result["energy"], ok)

    def site_density(self):
        require_reference(self)
        if self._integral_model:
            raise NotImplementedError(
                "site_density is a lattice observable; molecular projected RDMs are not implemented")
        from ..observables import noci_site_density

        r = self.result
        # Density needs only equal-determinant Gram blocks, so no full
        # expanded-orbit Hamiltonian is constructed here.
        s = lf_multi_frame_overlap(r["lam"], r["shift"], self.model.nelec)
        return noci_site_density(r["expanded_coeff"], s, self.model.L, self.model.nelec)

    def symmetry_diagnostics(self):
        """Contract analytic group expectations and fixed-row irrep purity."""
        from .group_diagnostics import lfmr_symmetry_diagnostics

        return lfmr_symmetry_diagnostics(self)
