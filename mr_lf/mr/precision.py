"""Retained-space metric repair and extended-precision contractions.

Numerical helpers for the existing whole-reference/whole-external PT2
partition. Canonical spectral selection belongs to lf_mrpt; these helpers
never reselect ranks or alter raw float64 kernels. All Hamiltonian group
pairs are retained even though Gram groups must be exactly disjoint.
"""

import numpy as np

from . import linalg


def _symmetric_copy(matrix, name):
    return linalg.symmetric_copy(matrix, name)


def _column_groups(cr, ct, rg, tg):
    """Label columns with exact support in one supplied electronic group."""
    labels = np.empty(cr.shape[1], dtype=np.int64)
    for j in range(cr.shape[1]):
        support = np.concatenate((rg[cr[:, j] != 0.0], tg[ct[:, j] != 0.0]))
        if not len(support) or np.any(support != support[0]):
            raise ValueError("each coefficient column must have exact support in one group")
        labels[j] = support[0]
    return labels


def _validate_boundary(blocks, coefficients, rg, tg, kind):
    if kind not in ("s", "h"):
        raise ValueError("kind must be s or h")
    if not isinstance(blocks, dict) or not isinstance(coefficients, dict):
        raise TypeError("blocks and coefficients must be dictionaries")
    names = (kind + "_rr", kind + "_rt", kind + "_tt")
    for name in names:
        matrix = blocks[name]
        if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be a finite matrix")
    nr, nt = blocks[names[1]].shape
    if nr == 0 or blocks[names[0]].shape != (nr, nr) or blocks[names[2]].shape != (nt, nt):
        raise ValueError("raw blocks must have shapes (nr,nr), (nr,nt), (nt,nt)")
    for name, size, labels in (("reference_groups", nr, rg), ("external_groups", nt, tg)):
        if not isinstance(labels, np.ndarray) or labels.dtype != np.int64:
            raise TypeError(f"{name} must be an int64 NumPy array")
        if labels.shape != (size,) or np.any(labels < 0):
            raise ValueError(f"{name} must have shape ({size},) and nonnegative labels")
    x, cr, ct = (
        coefficients[key] for key in ("x_ref", "external_ref_coeff", "external_candidate_coeff")
    )
    for name, matrix in (
        ("x_ref", x),
        ("external_ref_coeff", cr),
        ("external_candidate_coeff", ct),
    ):
        if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be a finite matrix")
    if (
        x.shape[0] != nr
        or not 1 <= x.shape[1] <= nr
        or cr.shape[0] != nr
        or ct.shape != (nt, cr.shape[1])
        or cr.shape[1] > nt
    ):
        raise ValueError("coefficient shapes must be (nr,r), (nr,s), (nt,s)")
    zero = np.zeros((nt, x.shape[1]), dtype=np.float64)
    ref_labels = _column_groups(x, zero, rg, tg)
    ext_labels = _column_groups(cr, ct, rg, tg)
    rr = _symmetric_copy(blocks[names[0]], names[0])
    tt = _symmetric_copy(blocks[names[2]], names[2])
    rt = blocks[names[1]]
    if kind == "s":
        for matrix, bra, ket in ((rr, rg, rg), (rt, rg, tg), (tt, tg, tg)):
            for group in np.unique(bra):
                if np.any(matrix[np.ix_(bra == group, ket != group)] != 0.0):
                    raise ValueError("extended contraction requires exactly disjoint Gram groups")
    return rr, rt, tt, x, zero, cr, ct, ref_labels, ext_labels


def _contract(
    rr,
    rt,
    tt,
    left_r,
    left_t,
    right_r,
    right_t,
    rg,
    tg,
    left_labels,
    right_labels,
    dtype,
    *,
    disjoint=False,
):
    result = np.zeros((left_r.shape[1], right_r.shape[1]), dtype=dtype)
    for a in np.unique(left_labels):
        ir, it, ic = (np.flatnonzero(labels == a) for labels in (rg, tg, left_labels))
        lr, lt = (
            np.asarray(matrix[np.ix_(rows, ic)], dtype=dtype)
            for matrix, rows in ((left_r, ir), (left_t, it))
        )
        for b in np.unique(right_labels):
            if disjoint and a != b:
                continue
            jr, jt, jc = (np.flatnonzero(labels == b) for labels in (rg, tg, right_labels))
            kr, kt = (
                np.asarray(matrix[np.ix_(rows, jc)], dtype=dtype)
                for matrix, rows in ((right_r, jr), (right_t, jt))
            )
            value = np.zeros((len(ic), len(jc)), dtype=dtype)
            if np.any(lr):
                action = np.zeros((len(ir), len(jc)), dtype=dtype)
                if np.any(kr):
                    action += np.asarray(rr[np.ix_(ir, jr)], dtype=dtype) @ kr
                if np.any(kt):
                    action += np.asarray(rt[np.ix_(ir, jt)], dtype=dtype) @ kt
                value += lr.T @ action
            if np.any(lt):
                action = np.zeros((len(it), len(jc)), dtype=dtype)
                if np.any(kr):
                    action += np.asarray(rt[np.ix_(jr, it)].T, dtype=dtype) @ kr
                if np.any(kt):
                    action += np.asarray(tt[np.ix_(it, jt)], dtype=dtype) @ kt
                value += lt.T @ action
            result[np.ix_(ic, jc)] = value
    return result


def contract_retained_blocks(
    blocks,
    coefficients,
    reference_groups,
    external_groups,
    *,
    kind="h",
    accumulator_dtype=np.longdouble,
):
    """Evaluate exact congruences using sparse column support and H/S actions.

    Raw blocks are float64 (nr,nr), (nr,nt), (nt,nt). Coefficients contain
    x_ref (nr,r), external_ref_coeff (nr,s), external_candidate_coeff
    (nt,s), all finite float64; r > 0, s >= 0. int64 group vectors (nr,),
    (nt,) describe exact electronic support. Each retained column must
    have support in one group. Gram blocks must be exactly disjoint;
    Hamiltonian blocks may couple different groups and all such pairs
    are included. Raw indices retain seed/alpha/beta/config C order.

    kind='h' returns h_reference (r,r), h_coupling (r,s), h_external
    (s,s); kind='s' returns reference_metric, cross_metric, external_metric
    in those shapes. Output dtype is accumulator_dtype (float64 or
    longdouble). Only multiplication/accumulation precision changes;
    float64 input kernels are not regenerated at higher precision.
    Square inputs use the existing symmetry check. Nothing is modified,
    inverted, symmetrized on output, truncated or solved. Type/shape/
    support/group errors raise TypeError/ValueError.
    """
    dtype = np.dtype(accumulator_dtype)
    if dtype not in (np.dtype(np.float64), np.dtype(np.longdouble)):
        raise TypeError("accumulator_dtype must be float64 or longdouble")
    rr, rt, tt, x, zero, cr, ct, xl, yl = _validate_boundary(
        blocks, coefficients, reference_groups, external_groups, kind
    )
    common = (rr, rt, tt)
    ending = (reference_groups, external_groups)
    reference = _contract(*common, x, zero, x, zero, *ending, xl, xl, dtype, disjoint=kind == "s")
    coupling = _contract(*common, x, zero, cr, ct, *ending, xl, yl, dtype, disjoint=kind == "s")
    external = _contract(*common, cr, ct, cr, ct, *ending, yl, yl, dtype, disjoint=kind == "s")
    names = (
        ("h_reference", "h_coupling", "h_external")
        if kind == "h"
        else ("reference_metric", "cross_metric", "external_metric")
    )
    return dict(zip(names, (reference, coupling, external), strict=True))


def recondition_retained_spaces(blocks, geometry, reference_groups, external_groups):
    """Repair metrics inside the already retained P/Q spaces, without reselection.

    Inputs have contract_retained_blocks' float64 shapes and exact group
    support. geometry is the accepted canonical geometry dictionary.
    Outputs are float64 x_ref (nr,r), external_ref_coeff (nr,s),
    external_candidate_coeff (nt,s), and Python int retained ranks.
    The original ranks and spectral cuts are not recomputed.

    With L_R L_R.T = X.T S_RR X, replace X by X L_R**(-T). Remove the
    remaining reference overlap from the external coefficients by solving
    M_R d = X.T S D and subtracting X d from D's reference rows. Then,
    with L_Q L_Q.T = D.T S D, replace D by D L_Q**(-T). All metric
    products accumulate in longdouble; near-identity Cholesky/linear
    solves run in float64, and stored coefficients remain float64.
    These are coordinate changes/projection-roundoff repairs, not a
    new H0, added physical direction or remedy for inaccurate raw kernels.

    Requires genuine extra longdouble precision in this environment;
    otherwise raises RuntimeError. A nonpositive reduced metric raises
    LinAlgError, with no eigenvalue clipping, shift or pseudoinverse.
    Type/shape/support errors follow contract_retained_blocks. Inputs
    and the original geometry are never modified.
    """
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise RuntimeError("this environment has no extra longdouble precision")
    metrics = contract_retained_blocks(
        blocks, geometry, reference_groups, external_groups, kind="s"
    )
    x, cr, ct = (
        geometry[key].copy() for key in ("x_ref", "external_ref_coeff", "external_candidate_coeff")
    )
    xl = _column_groups(x, np.zeros((len(ct), x.shape[1])), reference_groups, external_groups)
    yl = _column_groups(cr, ct, reference_groups, external_groups)
    for group in np.unique(xl):
        columns = np.flatnonzero(xl == group)
        rows = np.flatnonzero(reference_groups == group)
        metric = metrics["reference_metric"][np.ix_(columns, columns)]
        lower = np.linalg.cholesky(np.asarray(0.5 * metric + 0.5 * metric.T, dtype=np.float64))
        whitening = np.linalg.solve(lower, np.eye(len(columns))).T
        x[np.ix_(rows, columns)] = np.asarray(
            np.asarray(x[np.ix_(rows, columns)], dtype=np.longdouble) @ whitening, dtype=np.float64
        )
    repaired = {"x_ref": x, "external_ref_coeff": cr, "external_candidate_coeff": ct}
    metrics = contract_retained_blocks(
        blocks, repaired, reference_groups, external_groups, kind="s"
    )
    for group in np.unique(yl):
        rcols, ecols = np.flatnonzero(xl == group), np.flatnonzero(yl == group)
        rows = np.flatnonzero(reference_groups == group)
        if not len(rcols):
            continue
        correction = np.linalg.solve(
            np.asarray(metrics["reference_metric"][np.ix_(rcols, rcols)], dtype=np.float64),
            np.asarray(metrics["cross_metric"][np.ix_(rcols, ecols)], dtype=np.float64),
        )
        cr[np.ix_(rows, ecols)] = np.asarray(
            np.asarray(cr[np.ix_(rows, ecols)], dtype=np.longdouble)
            - np.asarray(x[np.ix_(rows, rcols)], dtype=np.longdouble) @ correction,
            dtype=np.float64,
        )
    metrics = contract_retained_blocks(
        blocks, repaired, reference_groups, external_groups, kind="s"
    )
    for group in np.unique(yl):
        columns = np.flatnonzero(yl == group)
        metric = metrics["external_metric"][np.ix_(columns, columns)]
        lower = np.linalg.cholesky(np.asarray(0.5 * metric + 0.5 * metric.T, dtype=np.float64))
        whitening = np.linalg.solve(lower, np.eye(len(columns))).T
        for coefficients, labels in ((cr, reference_groups), (ct, external_groups)):
            rows = np.flatnonzero(labels == group)
            coefficients[np.ix_(rows, columns)] = np.asarray(
                np.asarray(coefficients[np.ix_(rows, columns)], dtype=np.longdouble) @ whitening,
                dtype=np.float64,
            )
    repaired.update(reference_rank=x.shape[1], external_rank=ct.shape[1])
    return repaired


def _metric_errors(metrics):
    return {
        "reference_metric_error": linalg.identity_error(metrics["reference_metric"]),
        "external_metric_error": linalg.identity_error(metrics["external_metric"]),
        "cross_metric_error": linalg.max_abs(metrics["cross_metric"]),
    }
