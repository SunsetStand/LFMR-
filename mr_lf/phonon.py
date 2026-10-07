"""Number configurations and analytic infinite-oscillator displacement kernels."""

import math

import numpy as np

from .basis import decode


def phonon_configs(L, Nmax):
    if L <= 0 or isinstance(L, (int, np.integer)) == False:
        raise ValueError("L must be positive and an integer")
    if Nmax < 0 or isinstance(Nmax, (int, np.integer)) == False:
        raise ValueError("Nmax must be non-negative and an integer")
    d = Nmax + 1
    P = d**L
    configs = np.zeros((P, L), dtype=np.int64)
    for p in range(P):
        configs[p] = decode(p, L, d)
    return configs


def boson_operators(Nmax):
    if Nmax < 0 or isinstance(Nmax, (int, np.integer)) == False:
        raise ValueError("Nmax must be non-negative and an integer")
    d = Nmax + 1
    b = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if i == j - 1:
                b[i, j] = np.sqrt(j)
    bdag = b.conj().T
    number = bdag @ b
    return b, bdag, number


def lf_total_phonon_configurations(nmode: int, max_total: int) -> np.ndarray:
    """Enumerate non-vacuum configurations under a total-phonon cutoff.

    Parameters
    ----------
    nmode
        Positive number of phonon modes.
    max_total
        Positive inclusive cutoff on ``sum_x occupations[k, x]``.

    Returns
    -------
    numpy.ndarray
        Nonnegative integer occupations with shape ``(nconfig, nmode)``.
        The vacuum is excluded.  Rows are ordered first by increasing total
        occupation and then lexicographically within each total.  Every row
        satisfies ``1 <= occupations[k].sum() <= max_total``.

    Notes
    -----
    This is a collective total-phonon cutoff.  It differs from the exact-ED
    tensor-product convention that independently permits ``0..Nmax`` on
    every mode.  The number of returned rows is
    ``comb(max_total + nmode, nmode) - 1``.
    """
    if not isinstance(nmode, (int, np.integer)) or isinstance(nmode, (bool, np.bool_)):
        raise TypeError("Number of modes must be a integer.")
    if nmode <= 0:
        raise ValueError("Number of modes must be positive.")
    if not isinstance(max_total, (int, np.integer)) or isinstance(max_total, (bool, np.bool_)):
        raise TypeError("Maximum total occupation must be a integer.")
    if max_total <= 0:
        raise ValueError("Maximum total occupation must be positive.")
    occupations = []
    for total in range(1, max_total + 1):
        for config in np.ndindex(*(total + 1,) * nmode):
            if sum(config) == total:
                occupations.append(config)
    return np.array(occupations, dtype=np.int64)


def lf_franck_condon(lam: np.ndarray) -> np.ndarray:
    """Return the zero-phonon Franck--Condon overlap matrix.

    Parameters
    ----------
    lam
        Real conditional displacements with shape ``(nmode, norb)`` and
        index order ``lam[x, p]``.

    Returns
    -------
    numpy.ndarray
        Matrix with shape ``(norb, norb)`` whose elements are
        ``exp(-0.5 * sum_x((lam[x, p] - lam[x, q])**2))``.
    """
    if lam.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    norb = lam.shape[1]
    S = np.zeros((norb, norb), dtype=np.float64)
    for p in range(norb):
        for q in range(norb):
            delta = lam[:, p] - lam[:, q]
            S[p, q] = np.exp(-0.5 * np.sum(delta**2))
    return S


def lf_displacement_vacuum_amplitudes(
    lam: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    """Evaluate vacuum-to-number-state LF displacement amplitudes.

    For the hopping operator ``a_p^dagger a_q``, define the displacement
    vector ``delta[x, p, q] = lam[x, p] - lam[x, q]``.  This function
    evaluates

    ``<X| exp(sum_x delta[x,p,q] * (b_x^dagger-b_x)) |0>``

    for every supplied multimode occupation vector ``X``.

    Parameters
    ----------
    lam
        Finite real LF parameters with shape ``(nmode, norb)`` and index
        order ``lam[x, p]``.
    occupations
        Nonnegative integer phonon occupations with shape
        ``(nconfig, nmode)`` and index order ``occupations[k, x]``.

    Returns
    -------
    numpy.ndarray
        Real amplitudes with shape ``(nconfig, norb, norb)`` and index
        order ``[k, p, q]``.  Configuration ``k`` represents
        ``|X_k> = |n_0, n_1, ...>``.  Odd total phonon occupations retain
        the sign of the displacement components.
    """
    if lam.ndim != 2:
        raise ValueError("LF parameter array must be 2-dimensional.")
    for lam_value in lam.flat:
        if not np.isfinite(lam_value) or not np.isreal(lam_value):
            raise ValueError("LF parameter array must contain finite real values.")
    if occupations.ndim != 2:
        raise ValueError("Occupation array must be 2-dimensional.")
    nmode, norb = lam.shape
    if occupations.shape[1] != nmode:
        raise ValueError(
            "Occupation array second dimension must match LF parameter first dimension."
        )
    for occ_value in occupations.flat:
        if not np.issubdtype(type(occ_value), np.integer) or occ_value < 0:
            raise ValueError("Occupation array must contain non-negative integers.")
    delta = lam[:, :, None] - lam[:, None, :]
    Gpq = np.exp(-0.5 * np.sum(delta**2, axis=0))
    amplitudes = np.empty((occupations.shape[0], norb, norb), dtype=np.float64)
    for k in range(occupations.shape[0]):
        one = np.ones((norb, norb), dtype=np.float64)
        for x in range(nmode):
            n = occupations[k, x]
            if n > 0:
                one *= delta[x] ** n / np.sqrt(math.factorial(int(n)))
        amplitudes[k] = Gpq * one
    return amplitudes


def lf_vacuum_coupling_site_matrices(
    tmat: np.ndarray,
    g: float,
    omega: float,
    shift: np.ndarray,
    lam: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    """Build non-vacuum LF fluctuation couplings in the site basis.

    For every non-vacuum phonon configuration ``X_k``, construct the
    one-electron matrix

    ``W[k, p, q] = <X_k, p| V |0, q>``.

    Its hopping contribution is ``tmat[p, q]`` times the corresponding
    vacuum-to-number-state displacement amplitude.  A configuration with
    exactly one phonon in mode ``x`` additionally receives the diagonal
    residual coupling

    ``omega * shift[x] + g * delta[x, p] - omega * lam[x, p]``.

    Parameters
    ----------
    tmat
        Finite Hermitian site-basis hopping matrix with shape
        ``(norb, norb)``.
    g
        Finite real local Holstein coupling strength.
    omega
        Finite positive phonon frequency shared by all modes.
    shift
        Finite real coherent displacements with shape ``(nmode,)`` and
        index order ``shift[x]``.
    lam
        Finite real LF parameters with shape ``(nmode, norb)`` and index
        order ``lam[x, p]``.  This local Holstein specialization requires
        ``nmode == norb``.
    occupations
        Nonnegative integer configurations with shape ``(nconfig, nmode)``.
        Every row must have a strictly positive total occupation.

    Returns
    -------
    numpy.ndarray
        Site-basis coupling matrices with shape ``(nconfig, norb, norb)``
        and index order ``[k, p, q]``.  The dtype preserves a complex
        ``tmat``.  A matrix for fixed ``X_k`` need not be Hermitian.

    Notes
    -----
    Vacuum configurations are rejected because the zero-phonon block also
    contains static LF terms and the zeroth-order Fock subtraction.  For
    non-vacuum configurations those terms have zero matrix element, so the
    returned matrix is directly the required fluctuation coupling.
    """
    if lam.ndim != 2 or lam.shape[0] != lam.shape[1]:
        raise ValueError("LF parameter array must be 2-dimensional and square.")
    norb = lam.shape[1]
    if (
        tmat.ndim != 2
        or tmat.shape[0] != tmat.shape[1]
        or tmat.shape[0] != norb
        or not np.allclose(tmat, tmat.conj().T)
    ):
        raise ValueError("Hopping matrix must be 2-dimensional, square, and Hermitian.")
    for tmat_value in tmat.flat:
        if not np.isfinite(tmat_value):
            raise ValueError("Hopping matrix must contain finite values.")
    if shift.ndim != 1 or shift.size != norb:
        raise ValueError(
            "Shift array must be 1-dimensional and match LF parameter first dimension."
        )
    for shift_value in shift.flat:
        if not np.isfinite(shift_value) or not np.isreal(shift_value):
            raise ValueError("Shift array must contain finite real values.")
    if occupations.ndim != 2 or occupations.shape[1] != norb:
        raise ValueError(
            "Occupation array must be 2-dimensional with second dimension matching LF parameter first dimension."
        )
    for occ_value in occupations.flat:
        if not np.issubdtype(type(occ_value), np.integer) or occ_value < 0:
            raise ValueError("Occupation array must contain non-negative integers.")
    if np.any(np.sum(occupations, axis=1) == 0):
        raise ValueError("Occupation array must not contain vacuum configurations.")
    if omega <= 0 or not np.isfinite(omega) or not np.isreal(omega):
        raise ValueError("Frequency omega must be positive.")
    if not np.isfinite(g) or not np.isreal(g):
        raise ValueError("Coupling g must be finite and real.")
    amplitudes = lf_displacement_vacuum_amplitudes(lam, occupations)
    W = np.empty((occupations.shape[0], norb, norb), dtype=np.result_type(tmat, np.float64))
    for k in range(occupations.shape[0]):
        W[k] = tmat * amplitudes[k]
        if np.sum(occupations[k]) == 1:
            x = np.argmax(occupations[k])
            residual = np.empty(norb, dtype=np.float64)
            for p in range(norb):
                residual[p] = omega * shift[x] + g * (1 if p == x else 0) - omega * lam[x, p]
            W[k] += np.diag(residual)
    return W


def displacement_number_matrix(
    delta: float,
    max_bra: int,
    max_ket: int,
) -> np.ndarray:
    """Return a block of the single-mode displacement operator in number states.

    Parameters
    ----------
    delta
        Finite real scalar displacement in D(delta) = exp(delta*(b.dag-b)).
        Positive, negative and zero values are allowed; booleans and complex
        values are rejected, including complex values with zero imaginary part.
    max_bra, max_ket
        Nonnegative integer occupation limits, inclusive. Python and NumPy
        integer scalars are accepted; booleans are rejected.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape (max_bra+1, max_ket+1). Entry [m,n] is
        <m|D(delta)|n>: the row is the bra occupation, the column the ket.
        For overlaps of real displaced states use delta = eta_ket-eta_bra.

    Raises
    ------
    TypeError
        If delta is not a real numeric scalar, or an occupation limit is
        not an integer scalar. Boolean inputs are also rejected.
    ValueError
        If delta is nonfinite or a limit is negative.

    Notes
    -----
    This is a submatrix of the infinite-oscillator displacement operator,
    not the exponential of a truncated oscillator matrix. The returned
    finite block need not be orthogonal. First-column vacuum amplitudes
    seed the number-state recurrence; D(-delta) = D(delta).T for real delta.
    High occupations and large displacements require separate numerical
    stability checks; no uniform accuracy guarantee is made there.
    """
    if isinstance(delta, (bool, np.bool_)) or not isinstance(
        delta, (int, float, np.integer, np.floating)
    ):
        raise TypeError("delta must be a real numeric scalar, not boolean or complex")
    try:
        delta = float(delta)
    except (OverflowError, ValueError) as exc:
        raise ValueError("delta must be finite in float64") from exc
    if not np.isfinite(delta):
        raise ValueError("delta must be finite in float64")
    for name, limit in (("max_bra", max_bra), ("max_ket", max_ket)):
        if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)):
            raise TypeError(f"{name} must be an integer scalar, not boolean")
        if limit < 0:
            raise ValueError(f"{name} must be nonnegative")
    max_bra, max_ket = int(max_bra), int(max_ket)

    F = np.zeros((max_bra + 1, max_ket + 1))
    F[0, 0] = np.exp(-0.5 * delta**2)
    for n in range(1, max_bra + 1):
        F[n, 0] = delta / np.sqrt(n) * F[n - 1, 0]
    for m in range(1, max_ket + 1):
        F[0, m] = -delta / np.sqrt(m) * F[0, m - 1]
    for n in range(1, max_bra + 1):
        for m in range(1, max_ket + 1):
            F[n, m] = (np.sqrt(n) * F[n - 1, m - 1] - delta * F[n, m - 1]) / np.sqrt(m)
    return F


def multimode_displacement_overlap(
    delta: np.ndarray,
    bra_occupations: np.ndarray,
    ket_occupations: np.ndarray,
) -> np.ndarray:
    """Return number-state overlaps for a real multimode displacement.

    Parameters
    ----------
    delta
        Finite float64 array (nmode,), nmode > 0. For overlaps of displaced
        states, delta[x] = eta_ket[x] - eta_bra[x].
    bra_occupations, ket_occupations
        Nonnegative int64 arrays (nbra, nmode) and (nket, nmode), with
        nonempty configuration lists. Row k is a full multimode occupation
        vector; columns use the same mode order as delta. Vacuum rows,
        duplicate rows and arbitrary configuration ordering are allowed.

    Returns
    -------
    numpy.ndarray
        Float64 (nbra, nket). Entry [i,j] is the product over modes x of
        <bra_occupations[i,x]|D(delta[x])|ket_occupations[j,x]>.
        Rows/columns follow the input configuration order without sorting.

    Raises
    ------
    TypeError
        For non-array inputs or wrong dtypes (including boolean arrays).
    ValueError
        For wrong shapes, empty inputs, nonfinite displacements or
        negative occupation numbers.

    Notes
    -----
    Inputs must not be modified. Occupation-list truncation is defined by
    the caller; no tensor-product basis is allocated or inferred here.
    Use displacement_number_matrix independently for each mode. This
    computes only the phonon overlap, with no electron determinant overlap,
    translation projection, reference subtraction or Hamiltonian factors.
    """
    if not isinstance(delta, np.ndarray) or delta.dtype != np.float64:
        raise TypeError("delta must be a float64 NumPy array")
    if delta.ndim != 1 or delta.size == 0:
        raise ValueError("delta must have nonempty shape (nmode,)")
    if not np.all(np.isfinite(delta)):
        raise ValueError("delta must contain only finite values")
    for name, occupations in (
        ("bra_occupations", bra_occupations),
        ("ket_occupations", ket_occupations),
    ):
        if not isinstance(occupations, np.ndarray) or occupations.dtype != np.int64:
            raise TypeError(f"{name} must be an int64 NumPy array")
        if occupations.ndim != 2 or occupations.shape[0] == 0 or occupations.shape[1] != delta.size:
            raise ValueError(f"{name} must have nonempty shape (nconfig, nmode)")
        if np.any(occupations < 0):
            raise ValueError(f"{name} must contain nonnegative occupations")
    M = np.zeros((bra_occupations.shape[0], ket_occupations.shape[0], delta.size), dtype=np.float64)
    m_max = np.max(bra_occupations, axis=0)
    n_max = np.max(ket_occupations, axis=0)
    for x in range(delta.size):
        F = displacement_number_matrix(delta[x], m_max[x], n_max[x])
        M[:, :, x] = F[bra_occupations[:, x][:, None], ket_occupations[:, x][None, :]]
    return np.prod(M, axis=2)


def displaced_phonon_operator_matrices(
    eta_bra: float,
    eta_ket: float,
    max_bra: int,
    max_ket: int,
) -> dict[str, np.ndarray]:
    """Return single-mode operator blocks between real displaced number states.

    Parameters
    ----------
    eta_bra, eta_ket
        Finite real scalar displacements of the bra and ket. Python and
        NumPy real numeric scalars are accepted; boolean and complex values
        are rejected. The overlap displacement is eta_ket - eta_bra.
    max_bra, max_ket
        Nonnegative integer occupation limits, inclusive; booleans are
        rejected. The two limits may differ and either may be zero.

    Returns
    -------
    dict
        Four float64 arrays, each with shape (max_bra+1, max_ket+1):
        'overlap', 'annihilation', 'creation', 'number'. Entry [m,n] is
        <m|D(eta_bra).dagger O D(eta_ket)|n>, with O respectively identity,
        b, b.dagger, and b.dagger*b. Rows label bra, columns label ket.
        These are bare physical operators, with no omega, g or zero-point
        constant. No electronic or multimode factors are included.

    Raises
    ------
    TypeError
        If displacements are not real numeric scalars or limits are not
        integer scalars. Boolean inputs are rejected.
    ValueError
        If displacements or their difference are nonfinite in float64,
        or an occupation limit is negative.

    Notes
    -----
    Blocks refer to the complete oscillator space. Their finite outputs
    cannot in general be multiplied to produce a number-operator block.
    In different frames, creation is the transpose of annihilation with
    the two frames and occupation limits exchanged, not the transpose of
    annihilation in the same frame-pair block.

    Use displacement_number_matrix with delta = eta_ket - eta_bra.
    Right-action formulas need one extra ket occupation for b.dagger.
    That auxiliary column must be computed before taking the requested
    output block. High-occupation stability remains a separate question.

    """
    shifts = []
    for name, value in (("eta_bra", eta_bra), ("eta_ket", eta_ket)):
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
        shifts.append(value)
    eta_bra, eta_ket = shifts
    if not np.isfinite(eta_ket - eta_bra):
        raise ValueError("displacement difference must be finite in float64")
    for name, limit in (("max_bra", max_bra), ("max_ket", max_ket)):
        if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)):
            raise TypeError(f"{name} must be an integer scalar")
        if limit < 0:
            raise ValueError(f"{name} must be nonnegative")
    max_bra, max_ket = int(max_bra), int(max_ket)
    delta = eta_ket - eta_bra
    S = displacement_number_matrix(delta, max_bra, max_ket + 1)
    B = np.zeros((max_bra + 1, max_ket + 1))
    for m in range(max_bra + 1):
        for n in range(max_ket + 1):
            B[m, n] = eta_ket * S[m, n]
            B[m, n] += np.sqrt(n) * S[m, n - 1] if n > 0 else 0.0
    C = np.zeros((max_bra + 1, max_ket + 1))
    for m in range(max_bra + 1):
        for n in range(max_ket + 1):
            C[m, n] = np.sqrt(n + 1) * S[m, n + 1] + eta_ket * S[m, n]
    N = np.zeros((max_bra + 1, max_ket + 1))
    for m in range(max_bra + 1):
        for n in range(max_ket + 1):
            N[m, n] = (n + eta_ket**2) * S[m, n] + eta_ket * np.sqrt(n + 1) * S[m, n + 1]
            N[m, n] += eta_ket * np.sqrt(n) * S[m, n - 1] if n > 0 else 0.0
    result = {}
    result["overlap"] = S[: max_bra + 1, : max_ket + 1]
    result["annihilation"] = B
    result["creation"] = C
    result["number"] = N
    return result


def multimode_phonon_operator_matrices(
    eta_bra: np.ndarray,
    eta_ket: np.ndarray,
    bra_occupations: np.ndarray,
    ket_occupations: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return mode-resolved operators between displaced multimode number states.

    Parameters
    ----------
    eta_bra, eta_ket
        Finite float64 arrays (nmode,), nmode > 0, with identical mode
        ordering. Both absolute displacements are required. Their
        difference eta_ket - eta_bra must be finite in float64.
    bra_occupations, ket_occupations
        Nonnegative int64 arrays (nbra, nmode) and (nket, nmode), with
        nonempty configuration lists. Row i specifies a complete number
        configuration in the corresponding displaced frame. Input row
        order is preserved; vacuum and repeated configurations are allowed.

    Returns
    -------
    dict
        Float64 arrays: 'overlap' has shape (nbra, nket), and
        'annihilation', 'creation', 'number' each have shape
        (nmode, nbra, nket), indexed [x, i, j]. Operator entry [x,i,j] is
        <bra_i|D(eta_bra).dagger O_x D(eta_ket)|ket_j>, with O_x equal
        respectively to b_x, b_x.dagger, and b_x.dagger*b_x. The identity
        has one overlap block shared by all modes.

    Raises
    ------
    TypeError
        For non-array inputs or wrong dtypes.
    ValueError
        For invalid shapes, empty configuration lists, negative
        occupations, or nonfinite displacements or displacement difference.

    Notes
    -----
    Inputs must not be modified. Configuration lists define the requested
    subblocks; no complete tensor-product basis is allocated. Each local
    operator matrix element is multiplied by the overlap of every other
    mode. Products excluding the operated mode must be formed directly:
    dividing the full overlap by a single-mode overlap is invalid at zeros.
    For one mode the spectator product equals one.

    Reuse displaced_phonon_operator_matrices for each mode, then gather
    number-state entries according to the supplied configuration lists.
    These are physical phonon operators without omega, g, zero-point
    energy, electronic factors, or translation projection.
    """
    for name, eta in (("eta_bra", eta_bra), ("eta_ket", eta_ket)):
        if not isinstance(eta, np.ndarray) or eta.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if eta.ndim != 1 or eta.size == 0:
            raise ValueError(f"{name} must have nonempty shape (nmode,)")
        if not np.all(np.isfinite(eta)):
            raise ValueError(f"{name} must contain only finite values")
    if eta_ket.shape != eta_bra.shape:
        raise ValueError("eta_ket must have the same shape as eta_bra")
    with np.errstate(over="ignore", invalid="ignore"):
        finite_difference = np.all(np.isfinite(eta_ket - eta_bra))
    if not finite_difference:
        raise ValueError("displacement difference must be finite in float64")
    nmode = eta_bra.size
    for name, occupations in (
        ("bra_occupations", bra_occupations),
        ("ket_occupations", ket_occupations),
    ):
        if not isinstance(occupations, np.ndarray) or occupations.dtype != np.int64:
            raise TypeError(f"{name} must be an int64 NumPy array")
        if occupations.ndim != 2 or occupations.shape[0] == 0 or occupations.shape[1] != nmode:
            raise ValueError(f"{name} must have nonempty shape (nconfig,nmode)")
        if np.any(occupations < 0):
            raise ValueError(f"{name} must contain nonnegative occupations")
    bra_nmax = np.max(bra_occupations, axis=0)
    ket_nmax = np.max(ket_occupations, axis=0)
    nbra = bra_occupations.shape[0]
    nket = ket_occupations.shape[0]
    S = np.zeros((nmode, nbra, nket))
    B = np.zeros_like(S)
    C = np.zeros_like(S)
    N = np.zeros_like(S)
    for x in range(nmode):
        result = displaced_phonon_operator_matrices(
            eta_bra[x], eta_ket[x], bra_nmax[x], ket_nmax[x]
        )
        rows = bra_occupations[:, x][:, None]
        cols = ket_occupations[:, x][None, :]
        S[x] = result["overlap"][rows, cols]
        B[x] = result["annihilation"][rows, cols]
        C[x] = result["creation"][rows, cols]
        N[x] = result["number"][rows, cols]
    total_overlap = np.prod(S, axis=0)
    for x in range(nmode):
        S_prod_else = np.prod(S[np.arange(nmode) != x], axis=0)
        B[x] *= S_prod_else
        C[x] *= S_prod_else
        N[x] *= S_prod_else
    result = {"overlap": total_overlap, "annihilation": B, "creation": C, "number": N}
    return result
