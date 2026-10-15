"""Canonical NOCI for real or complex structured molecular kernels."""

import numpy as np
from scipy import linalg

from ._metric_precision import metric_norm, metric_product, metric_residual


def hermitian_noci_result(hmat, smat, *, overlap_cut=1e-10, condition_limit=1e8):
    """Return the lowest root, retained spectrum and metric diagnostics.

    Both finite float64/complex128 tensors have matching bra/ket axes.
    Flattening occurs here, at the dense solve boundary. Retain overlap
    eigenvalues s > overlap_cut*max(1,s_max); use conjugate transposes
    throughout. A direct generalized solve is used only when every metric
    direction survives and its condition number is <= condition_limit.
    The reported residual is in the retained canonical space, not the full
    electron/photon Hilbert space.
    """
    for name, value, minimum in (("overlap_cut", overlap_cut, 0),
                                  ("condition_limit", condition_limit, 1)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < minimum
                or (minimum == 0 and value == 0)):
            raise ValueError(f"{name} must be finite and {'positive' if minimum == 0 else '>= 1'}")
    for name, array in (("hmat", hmat), ("smat", smat)):
        if not isinstance(array, np.ndarray) or array.dtype not in (np.float64, np.complex128):
            raise TypeError(f"{name} must be a float64 or complex128 NumPy array")
        half = array.ndim // 2
        if (array.ndim < 2 or array.ndim % 2 or not all(array.shape)
                or array.shape[:half] != array.shape[half:]):
            raise ValueError(f"{name} must have matching nonempty bra and ket shapes")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if hmat.shape != smat.shape:
        raise ValueError("hmat and smat must have identical shapes")
    shape = hmat.shape[:hmat.ndim // 2]
    size = int(np.prod(shape))
    matrices = []
    for name, array in (("hmat", hmat), ("smat", smat)):
        flat = array.reshape(size, size)
        scale = max(1., float(np.max(np.abs(flat))))
        if np.max(np.abs(flat - flat.conj().T)) > 1e-12 * scale:
            raise ValueError(f"{name} must be Hermitian within relative tolerance 1e-12")
        matrices.append(.5 * flat + .5 * flat.conj().T)
    h, s = matrices
    values, vectors = np.linalg.eigh(s)
    threshold = overlap_cut * max(1., float(values[-1]))
    if values[0] < -threshold:
        raise ValueError("smat has negative eigenvalues beyond the overlap threshold")
    keep = values > threshold
    if not np.any(keep):
        raise ValueError("no nonzero projected overlap directions survive overlap_cut")
    x = vectors[:, keep] / np.sqrt(values[keep])
    extended = np.max(np.abs(x)) > 100
    product = metric_product if extended else lambda a, b, c: a.conj().T @ b @ c
    reduced_h, reduced_s = product(x, h, x), product(x, s, x)
    # Even accepted metric eigenvectors lose relative accuracy along small
    # eigenvalues. Solve with their actual metric instead of assuming I.
    spectrum, eigenvectors = linalg.eigh(
        .5 * reduced_h + .5 * reduced_h.conj().T,
        .5 * reduced_s + .5 * reduced_s.conj().T)
    rank = int(np.count_nonzero(keep))
    if rank == size and values[-1] / values[0] <= condition_limit:
        energies, roots = linalg.eigh(h, s, subset_by_index=[0, 0])
        energy, coeff = float(energies[0]), roots[:, 0]
    else:
        energy = float(spectrum[0])
        coeff = np.asarray(x.astype(np.clongdouble) @ eigenvectors[:, 0].astype(np.clongdouble),
                           dtype=np.complex128) if extended else x @ eigenvectors[:, 0]
    norm = metric_norm(coeff, s)
    coeff = coeff / np.sqrt(norm.real)
    norm_error = float(abs(metric_norm(coeff, s) - 1.))
    residual = metric_residual(x, h, s, coeff, energy)
    gap = float(spectrum[1] - spectrum[0]) if rank > 1 else float("inf")
    return {"energy": energy, "coeff": coeff.reshape(shape), "basis_shape": shape,
            "rank": rank, "hmat": h, "smat": s, "norm_error": norm_error,
            "residual_projected": residual, "reference_eigenvalues": spectrum,
            "reference_gap": gap, "near_degenerate": bool(gap <= 1e-8),
            "overlap_eigenvalues": values, "overlap_threshold": float(threshold)}
