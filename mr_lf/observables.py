"""Spin-resolved density expectations in the physical overlap metric."""

import numpy as np

from .basis import make_electron_basis


def determinant_occupations(L, nelec):
    """Return alpha/beta occupations, [determinant address,site]."""
    return [
        ((make_electron_basis(L, n)[0][:, None] >> np.arange(L)) & 1).astype(np.float64)
        for n in nelec
    ]


def exact_site_density(psi_site, L, nelec):
    """Return normalized float64 (2,L) density from a structured FCI tensor."""
    oa, ob = determinant_occupations(L, nelec)
    shape = (len(oa), len(ob), -1)
    weights = np.sum(abs(np.asarray(psi_site).reshape(shape)) ** 2, axis=-1)
    norm = weights.sum()
    if norm <= 0:
        raise ValueError("wavefunction has zero norm")
    return np.stack((weights.sum(axis=1) @ oa, weights.sum(axis=0) @ ob)) / norm


def noci_site_density(coeff, smat, L, nelec):
    """Return (2,L) density; coeff[A,ia,ib] is not a probability amplitude.

    smat is the unprojected overlap tensor or its flattened square matrix.
    Use expanded_coeff and orbit overlaps for a projected calculation.
    """
    oa, ob = determinant_occupations(L, nelec)
    coeff = np.asarray(coeff)
    if coeff.ndim != 3 or coeff.shape[1:] != (len(oa), len(ob)):
        raise ValueError("coeff must have shape (nframe,nstra,nstrb)")
    c = coeff.ravel()
    sc = np.asarray(smat).reshape(c.size, c.size) @ c
    norm = float(c @ sc)
    if norm <= 0:
        raise ValueError("nonpositive physical norm")
    weights = np.sum(coeff * sc.reshape(coeff.shape), axis=0) / norm
    return np.stack((weights.sum(axis=1) @ oa, weights.sum(axis=0) @ ob))


def symmetry_indicators(spin_density):
    """Charge max(n)-min(n); spin max(abs(n_alpha-n_beta))."""
    density = np.asarray(spin_density)
    if density.ndim != 2 or density.shape[0] != 2:
        raise ValueError("spin_density must have shape (2,L)")
    return {
        "charge_imbalance": float(np.ptp(density.sum(axis=0))),
        "spin_amplitude": float(np.max(np.abs(density[0] - density[1]))),
    }


def lf_noci_site_density(
    coeff: np.ndarray,
    smat: np.ndarray,
) -> np.ndarray:
    """Return electronic site occupations for a real NOCI state.

    For ``|Psi> = sum_{A,p} coeff[A,p] |A,p>``, the occupation
    at site ``p`` is ``sum_{A,B} coeff[A,p] *
    smat[A,p,B,p] * coeff[B,p]``.  Cross-frame overlap terms
    are required even though different electronic sites are
    orthogonal.

    Parameters
    ----------
    coeff
        Finite float64 CI coefficients with shape ``(K, L)``
        and index order ``[frame, electron site]``.  Coefficients
        from ``lf_noci_lowest`` are normalized in the S metric.
    smat
        Finite float64 overlap tensor for the same frames with shape
        ``(K, L, K, L)`` and order ``[A, p, B, q]``.

    Returns
    -------
    numpy.ndarray
        Float64 site occupations with shape ``(L,)`` and order
        ``[electron site]``.  Their sum is
        ``coeff.ravel() @ S @ coeff.ravel()``, hence one for a
        normalized one-electron state.  The function neither
        renormalizes coefficients nor clips the result.

    Raises
    ------
    ValueError
        If dimensions are empty or incompatible, or values are
        nonfinite.
    TypeError
        If either array does not have float64 dtype.
    """
    if coeff.ndim != 2:
        raise ValueError("coeff must be a 2D array")
    K, L = coeff.shape
    if K == 0 or L == 0:
        raise ValueError("coeff must have positive dimensions")
    if smat.shape != (K, L, K, L):
        raise ValueError("smat must have shape (K, L, K, L)")
    if coeff.dtype != np.float64 or smat.dtype != np.float64:
        raise TypeError("coeff and smat must have float64 dtype")
    if not np.all(np.isfinite(coeff)) or not np.all(np.isfinite(smat)):
        raise ValueError("coeff and smat must contain only finite values")
    rho = np.zeros((L,), dtype=np.float64)
    for p in range(L):
        rho[p] = coeff[:, p] @ smat[:, p, :, p] @ coeff[:, p]
    return rho


def lf_noci_site_phonon_moment(
    coeff: np.ndarray,
    smat: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """Return the joint electron-site and phonon-displacement moment.

    For real one-electron NOCI coefficients and conditional displacements
    ``eta[A, x, p] = shift[A, x] - lam[A, x, p]``, compute
    ``M[x, p] = <n_p (b_x + b_x^dagger)>``.  The matrix element sums over
    both frame indices: ``coeff[A, p] * smat[A, p, B, p] *
    coeff[B, p] * (eta[A, x, p] + eta[B, x, p])``.  In the uncentered
    Holstein convention, the coupling energy is ``g * trace(M)``.

    Parameters
    ----------
    coeff
        Finite float64 CI coefficients with shape ``(K, L)`` and index
        order ``[frame, electron site]``.  They are normally normalized
        in the metric supplied by ``smat``.
    smat
        Finite float64 overlap tensor with shape ``(K, L, K, L)`` and
        index order ``[A, p, B, q]`` for the same frames.
    lam
        Finite float64 LF parameters with shape ``(K, L, L)`` and index
        order ``[frame, phonon mode, electron site]``.
    shift
        Finite float64 coherent shifts with shape ``(K, L)`` and index
        order ``[frame, phonon mode]``.

    Returns
    -------
    numpy.ndarray
        Float64 joint moment with shape ``(L, L)`` and index order
        ``[phonon mode x, electron site p]``.  It is not divided by the
        site occupation, and inputs are not renormalized or modified.

    Raises
    ------
    ValueError
        If dimensions are empty or incompatible, or values are nonfinite.
    TypeError
        If an input array does not have float64 dtype.
    """
    if coeff.ndim != 2:
        raise ValueError("coeff must be a 2D array")
    K, L = coeff.shape
    if K == 0 or L == 0:
        raise ValueError("coeff must have positive dimensions")
    if smat.shape != (K, L, K, L):
        raise ValueError("smat must have shape (K, L, K, L)")
    if lam.shape != (K, L, L):
        raise ValueError("lam must have shape (K, L, L)")
    if shift.shape != (K, L):
        raise ValueError("shift must have shape (K, L)")
    if (
        coeff.dtype != np.float64
        or smat.dtype != np.float64
        or lam.dtype != np.float64
        or shift.dtype != np.float64
    ):
        raise TypeError("coeff, smat, lam, and shift must have float64 dtype")
    if (
        not np.all(np.isfinite(coeff))
        or not np.all(np.isfinite(smat))
        or not np.all(np.isfinite(lam))
        or not np.all(np.isfinite(shift))
    ):
        raise ValueError("coeff, smat, lam, and shift must contain only finite values")
    eta = shift[:, :, None] - lam
    M = np.zeros((L, L), dtype=np.float64)
    for x in range(L):
        for p in range(L):
            for A in range(K):
                for B in range(K):
                    M[x, p] += (
                        coeff[A, p] * smat[A, p, B, p] * coeff[B, p] * (eta[A, x, p] + eta[B, x, p])
                    )
    return M
