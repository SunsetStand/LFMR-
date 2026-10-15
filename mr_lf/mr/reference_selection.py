"""Choose fixed LF frames by their contributions to a projected CI subspace."""

import numpy as np
from scipy import linalg

from ._metric_precision import metric_norm, metric_product, metric_residual


def _pool_kernels(model, lam, shift, *, symmetry, irrep, determinants):
    """Build the bounded dense pool once, including a possibly zero projector."""
    if symmetry is None:
        from .general_kernels import lf_vacuum_frame_blocks

        blocks = lf_vacuum_frame_blocks(model, lam, shift, determinants=determinants)
        h, s = blocks["hmat"], blocks["smat"]
    else:
        from .point_group_kernels import lf_group_frame_blocks

        h, s = lf_group_frame_blocks(model, lam, shift, symmetry=symmetry,
                                     irrep=irrep, determinants=determinants)
    shape = h.shape[:h.ndim // 2]
    dimension = int(np.prod(shape))
    return h.reshape(dimension, dimension), s.reshape(dimension, dimension), shape


def _hermitian(array):
    return .5 * array + .5 * array.conj().T


def _append_basis(basis, smat, start, stop, threshold):
    """Append residualized candidate columns; keep every old column unchanged.

    Only eigenvalues of the new residual Gram block are thresholded. Its
    threshold is fixed from the original candidate block, so adding a frame
    never changes a relative cutoff on directions accepted in earlier steps.
    The extra reorthogonalization and metric normalization act on new columns
    alone. The retained metric is solved explicitly rather than assumed exact.
    """
    residual = np.eye(len(smat), dtype=np.complex128)[:, start:stop].copy()
    old_gram = _hermitian(basis.conj().T @ smat @ basis)

    def remove_old(columns):
        if basis.shape[1]:
            for _ in range(2):
                overlap = basis.conj().T @ smat @ columns
                columns -= basis @ linalg.solve(old_gram, overlap, assume_a="pos")
        return columns

    residual = remove_old(residual)
    values, vectors = np.linalg.eigh(_hermitian(residual.conj().T @ smat @ residual))
    if values[0] < -threshold:
        raise ValueError("candidate residual Gram matrix has a negative eigenvalue")
    keep = values > threshold
    if not np.any(keep):
        return None
    new = remove_old(residual @ (vectors[:, keep] / np.sqrt(values[keep])))
    metric, rotation = np.linalg.eigh(_hermitian(new.conj().T @ smat @ new))
    if metric[0] <= 0 or not np.all(np.isfinite(metric)):
        raise ValueError("new reference directions have an unstable overlap metric")
    new = new @ (rotation / np.sqrt(metric))
    return np.column_stack((basis, new))


def _lowest(hmat, smat, basis):
    """Solve in a retained nested span without any further rank truncation."""
    extended = np.max(np.abs(basis)) > 100.
    product = metric_product if extended else lambda left, matrix, right: left.conj().T @ matrix @ right
    h = _hermitian(product(basis, hmat, basis))
    s = _hermitian(product(basis, smat, basis))
    values, vectors = linalg.eigh(h, s, subset_by_index=[0, 0])
    energy, vector = float(values[0]), vectors[:, 0]
    coeff = np.asarray(np.asarray(basis, dtype=np.clongdouble)
                       @ np.asarray(vector, dtype=np.clongdouble), dtype=np.complex128)
    norm = metric_norm(coeff, smat).real
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("selected reference state has an invalid physical norm")
    coeff, vector = coeff / np.sqrt(norm), vector / np.sqrt(norm)
    residual = metric_residual(basis, hmat, smat, coeff, energy)
    return {"energy": energy, "coeff": coeff, "residual_retained": residual,
            "residual_reduced": float(np.linalg.norm(h @ vector - energy * (s @ vector))),
            "norm_error": float(abs(metric_norm(coeff, smat) - 1.))}


def _candidate_data(hmat, smat, nframe, overlap_cut):
    block = len(smat) // nframe
    empty = np.zeros((len(smat), 0), dtype=np.complex128)
    thresholds, energies = [], []
    for index in range(nframe):
        start, stop = index * block, (index + 1) * block
        values = np.linalg.eigvalsh(smat[start:stop, start:stop])
        threshold = overlap_cut * max(1., float(values[-1]))
        thresholds.append(threshold)
        basis = _append_basis(empty, smat, start, stop, threshold)
        energies.append(None if basis is None else _lowest(hmat, smat, basis)["energy"])
    return block, thresholds, energies


def select_lf_references(model, lam, shift=None, *, symmetry=None, irrep=None,
                         determinants=None, strategy="greedy", energy_tol=1e-8,
                         overlap_cut=1e-10, max_references=None):
    """Select a nested sequence from a supplied, fixed LF frame pool.

    Parameters follow ``LFMR``: lam is float64 (F,nmode,norb), shift is
    float64 (F,nmode) or None, and determinants is an optional common int64
    (D,2) list of alpha/beta bit strings. symmetry and irrep must either both
    be specified or both be None. No frame/orbital optimization is performed.

    ``strategy='unprojected'`` ranks by each unprojected frame CI energy;
    ``'projected'`` ranks by each target-sector single-frame CI energy.
    Both then add frames in that fixed order, skipping zero projectors and
    duplicate subspaces. ``'greedy'`` chooses the largest actual target-sector
    MR energy lowering at each step. Ties are resolved by input frame index.
    After the first accepted frame, stop before accepting a gain strictly
    smaller than energy_tol. Zero energy_tol allows zero-gain new directions.

    New candidate directions are residualized against the retained subspace
    in the physical Gram metric. Their eigenvalues must exceed
    overlap_cut*max(1, s_max(candidate)). Previously accepted directions are
    retained unchanged, so a growing global relative cutoff cannot remove
    them. All strategies solve in the same target symmetry sector throughout.

    Returns a dictionary with chosen_indices; target single_frame_energies
    (None for a zero projector); ranking_energies; energies, ranks, gains,
    retained residuals and norm errors per accepted step; stop_reason; the
    final energy and full-pool complex128 coeff; and the retained basis Q in
    full-pool coefficient coordinates. coeff has the same structured shape as
    LFMR (first axis F); unchosen frames have zero coefficients. An all-zero
    projected pool returns energy=None, rank=0 and zero coefficients.

    This first implementation builds all dense pool S/H kernels once. The
    existing 256-electron-determinant and 1024-total-basis-state caps apply to
    the whole supplied pool, including irrep columns, before selection.
    """
    if strategy not in ("unprojected", "projected", "greedy"):
        raise ValueError("strategy must be 'unprojected', 'projected', or 'greedy'")
    if (symmetry is None) != (irrep is None):
        raise ValueError("symmetry and irrep must be supplied together")
    for name, value, positive in (("energy_tol", energy_tol, False),
                                   ("overlap_cut", overlap_cut, True)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0 or (positive and value == 0)):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if max_references is not None and (
            isinstance(max_references, (bool, np.bool_))
            or not isinstance(max_references, (int, np.integer)) or max_references < 1):
        raise ValueError("max_references must be a positive integer or None")
    hmat, smat, shape = _pool_kernels(model, lam, shift, symmetry=symmetry,
                                     irrep=irrep, determinants=determinants)
    nframe = shape[0]
    block, thresholds, energies = _candidate_data(hmat, smat, nframe, overlap_cut)
    ranking = energies
    if strategy == "unprojected" and symmetry is not None:
        raw_h, raw_s, raw_shape = _pool_kernels(model, lam, shift, symmetry=None,
                                              irrep=None, determinants=determinants)
        _, _, ranking = _candidate_data(raw_h, raw_s, raw_shape[0], overlap_cut)
    order = sorted((i for i in range(nframe) if energies[i] is not None),
                   key=lambda i: (float("inf") if ranking[i] is None else ranking[i], i))
    remaining = list(order)
    chosen, history = [], []
    basis = np.zeros((len(smat), 0), dtype=np.complex128)
    current, stop_reason = None, "zero_projector" if not order else "pool_exhausted"
    limit = nframe if max_references is None else int(max_references)
    while remaining and len(chosen) < limit:
        best, duplicates = None, []
        for index in remaining:
            candidate = _append_basis(basis, smat, index * block, (index + 1) * block,
                                      thresholds[index])
            if candidate is None:
                duplicates.append(index)
                continue
            result = _lowest(hmat, smat, candidate)
            trial = (result["energy"], index, candidate, result)
            if best is None or trial[:2] < best[:2]:
                best = trial
            if strategy != "greedy":
                break
        remaining = [i for i in remaining if i not in duplicates]
        if best is None:
            stop_reason = "no_new_direction"
            break
        _, index, candidate, result = best
        gain = None if current is None else current["energy"] - result["energy"]
        if gain is not None and gain < energy_tol:
            stop_reason = "energy_tolerance"
            break
        basis, current = candidate, result
        chosen.append(index)
        remaining.remove(index)
        history.append({"index": index, "energy": result["energy"],
                        "rank": basis.shape[1], "gain": gain,
                        "residual_retained": result["residual_retained"],
                        "residual_reduced": result["residual_reduced"],
                        "norm_error": result["norm_error"]})
    if len(chosen) == limit and remaining:
        stop_reason = "max_references"
    coeff = np.zeros(len(smat), dtype=np.complex128) if current is None else current["coeff"]
    return {"strategy": strategy, "chosen_indices": chosen,
            "single_frame_energies": energies, "ranking_energies": ranking,
            "energies": [step["energy"] for step in history],
            "ranks": [step["rank"] for step in history],
            "gains": [step["gain"] for step in history],
            "residuals_retained": [step["residual_retained"] for step in history],
            "norm_errors": [step["norm_error"] for step in history], "history": history,
            "stop_reason": stop_reason, "energy": None if current is None else current["energy"],
            "rank": basis.shape[1], "coeff": coeff.reshape(shape), "Q": basis,
            "basis_shape": shape, "candidate_count": nframe,
            "zero_projector_indices": [i for i, energy in enumerate(energies) if energy is None],
            "energy_tol": float(energy_tol), "overlap_cut": float(overlap_cut)}
