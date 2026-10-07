"""Canonical reference geometry and orthogonal external-space construction."""

import numpy as np

from . import linalg


def _symmetric_copy(matrix: np.ndarray, name: str) -> np.ndarray:
    return linalg.symmetric_copy(matrix, name)


def _check_positive_semidefinite(values: np.ndarray, name: str) -> None:
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    if np.min(values, initial=0.0) < -1e-10 * scale:
        raise ValueError(f"{name} has a significantly negative eigenvalue")


def orthogonalize_mr_spaces(
    s_rr: np.ndarray,
    s_rt: np.ndarray,
    s_tt: np.ndarray,
    *,
    reference_cut: float = 1e-10,
    external_cut: float = 1e-10,
) -> dict[str, object]:
    """Orthogonalize references and remove their retained span from candidates.

    Parameters
    ----------
    s_rr
        Reference Gram matrix R.T @ R, finite float64 (nr, nr), nr > 0.
    s_rt
        Cross overlap R.T @ T, finite float64 (nr, nt). Rows label reference
        bras and columns label external-candidate kets. nt may be zero.
    s_tt
        Candidate Gram matrix T.T @ T, finite float64 (nt, nt).
    reference_cut, external_cut
        Finite positive real spectral cutoffs. At each orthogonalization,
        retain eigenvalues strictly greater than cut * max(1, largest).
        The reference rule matches lf_noci_lowest. These cutoffs change
        the retained spaces; they are not perturbative denominator shifts.

    Returns
    -------
    dict
        All arrays are float64. With retained ranks r and s:

        * x_ref: (nr, r), canonical reference transformation X.
        * ref_subtraction: (nr, nt), B = X @ (X.T @ s_rt).
        * external_metric: (nt, nt), G = s_tt - s_rt.T @ B.
        * x_external: (nt, s), canonical transformation Y of G.
        * external_ref_coeff: (nr, s), -B @ Y.
        * external_candidate_coeff: (nt, s), a copy of Y.
        * reference_eigenvalues: (nr,), full ascending reference spectrum.
        * external_eigenvalues: (nt,), full ascending G spectrum.
        * reference_rank, external_rank: Python ints r and s.

        The physical bases are R @ X and Z = (T - R @ B) @ Y. They obey
        (R @ X).T @ Z = 0 and Z.T @ Z = I on retained spaces. Orthogonality
        to discarded reference directions is not imposed. A zero external
        rank is valid and yields empty-column transformation arrays.

    Raises
    ------
    TypeError
        For non-NumPy inputs or arrays whose dtype is not float64.
    ValueError
        For invalid shapes, nonfinite data, invalid cutoffs, asymmetry,
        incompatible Gram blocks, or zero retained reference rank.

    Notes
    -----
    This is a dense linear-algebra boundary. Flatten physical reference
    tensors in C order (frame, alpha address, beta address); candidates use
    (frame, alpha address, beta address, phonon configuration). The caller
    must use consistent basis order and projection normalization in all
    blocks. Electron counts and physical representations are not inferred.

    Input arrays are not modified. Square blocks are symmetrized only if
    max(abs(S-S.T)) <= 1e-12 * max(1, max(abs(S))). The joint Gram matrix
    and both individual spectra must have no eigenvalues below
    -1e-10 * max(1, max(abs(eigenvalues))). Tiny negative eigenvalues remain
    in the reported spectra but are never retained. The PSD tolerance is
    independent of the rank cutoffs. The joint check costs a dense (nr+nt)
    eigensolve and is intended for the initial small-space implementation.

    No Hamiltonian, energy, explicit physical projector, or inverse is
    constructed. This geometry alone does not define a PT2 partition.
    """
    for name, array in (("s_rr", s_rr), ("s_rt", s_rt), ("s_tt", s_tt)):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a NumPy array with float64 dtype")
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    nr, nt = s_rt.shape
    if nr == 0 or s_rr.shape != (nr, nr) or s_tt.shape != (nt, nt):
        raise ValueError("expected shapes (nr,nr), (nr,nt), (nt,nt) with nr > 0")
    for name, cut in (("reference_cut", reference_cut), ("external_cut", external_cut)):
        if (
            isinstance(cut, (bool, np.bool_))
            or not isinstance(cut, (int, float, np.integer, np.floating))
            or not np.isfinite(cut)
            or cut <= 0
        ):
            raise ValueError(f"{name} must be finite, real, and positive")

    rr = _symmetric_copy(s_rr, "s_rr")
    tt = _symmetric_copy(s_tt, "s_tt")
    joint = np.block([[rr, s_rt], [s_rt.T, tt]])
    _check_positive_semidefinite(np.linalg.eigvalsh(joint), "joint Gram matrix")

    reference_values, reference_vectors = np.linalg.eigh(rr)
    _check_positive_semidefinite(reference_values, "s_rr")
    keep_ref = reference_values > reference_cut * max(1.0, float(reference_values[-1]))
    if not np.any(keep_ref):
        raise ValueError("no reference eigenvalues exceed the reference_cut threshold")
    x_ref = reference_vectors[:, keep_ref] / np.sqrt(reference_values[keep_ref])

    # The subtraction removes the entire retained reference space, not
    # just one target CI state. Parenthesization avoids a dense nr x nr P.
    ref_subtraction = x_ref @ (x_ref.T @ s_rt)
    external_metric = tt - s_rt.T @ ref_subtraction
    external_metric = 0.5 * external_metric + 0.5 * external_metric.T
    external_values, external_vectors = np.linalg.eigh(external_metric)
    _check_positive_semidefinite(external_values, "external metric")
    external_scale = max(1.0, float(np.max(external_values, initial=0.0)))
    keep_ext = external_values > external_cut * external_scale
    x_external = external_vectors[:, keep_ext] / np.sqrt(external_values[keep_ext])

    return {
        "x_ref": x_ref,
        "ref_subtraction": ref_subtraction,
        "external_metric": external_metric,
        "x_external": x_external,
        "external_ref_coeff": -ref_subtraction @ x_external,
        "external_candidate_coeff": x_external.copy(),
        "reference_eigenvalues": reference_values,
        "external_eigenvalues": external_values,
        "reference_rank": int(np.count_nonzero(keep_ref)),
        "external_rank": int(np.count_nonzero(keep_ext)),
    }


def transform_mr_hamiltonian_blocks(
    h_rr: np.ndarray,
    h_rt: np.ndarray,
    h_tt: np.ndarray,
    x_ref: np.ndarray,
    external_ref_coeff: np.ndarray,
    external_candidate_coeff: np.ndarray,
) -> dict[str, np.ndarray]:
    """Represent Hamiltonian blocks in retained reference/external bases.

    Parameters
    ----------
    h_rr, h_rt, h_tt
        Finite float64 arrays with shapes (nr,nr), (nr,nt), (nt,nt),
        respectively. They represent R.T@H@R, R.T@H@T, T.T@H@T in the
        same raw bases used to construct the overlap geometry. nr > 0;
        nt may be zero. h_rr and h_tt must be symmetric within tolerance.
        Hamiltonian spectra may be negative; no PSD check is appropriate.
    x_ref
        Finite float64 (nr,r), with 1 <= r <= nr. Use 'x_ref' from
        orthogonalize_mr_spaces applied to the matching overlap blocks.
        The orthonormal reference columns are Rbar = R@x_ref.
    external_ref_coeff, external_candidate_coeff
        Finite float64 arrays (nr,s) and (nt,s), with 0 <= s <= nt.
        Use the equally named geometry outputs. They define the external
        columns Z = R@external_ref_coeff + T@external_candidate_coeff.
        external_ref_coeff already contains the negative subtraction;
        do not subtract the reference span a second time.

    Returns
    -------
    dict
        Exactly three finite float64 matrices:

        * 'h_reference': (r,r), Rbar.T@H@Rbar.
        * 'h_coupling': (r,s), Rbar.T@H@Z; reference bra/external ket.
        * 'h_external': (s,s), Z.T@H@Z.

        A zero retained external rank yields (r,0) coupling and (0,0)
        external matrices, whether raw candidates are absent or removed
        entirely by the overlap geometry. Column labels are retained
        orthonormal basis indices, rather than individual LF frames or
        determinant addresses.

    Raises
    ------
    TypeError
        For non-NumPy inputs or array dtypes other than float64.
    ValueError
        For nonfinite data, invalid dimensions, incompatible shapes or
        retained ranks, or asymmetry beyond the tolerance.

    Notes
    -----
    Raw reference order is (frame,alpha address,beta address) in C order;
    candidates use (frame,alpha address,beta address,configuration).
    All Hamiltonian/overlap blocks must agree in order, physical model,
    translation sector, and projection normalization. The current
    build_mr_candidate_blocks produces unprojected blocks. This function
    applies no symmetry projection and cannot check these conventions.

    Inputs are not modified. Square Hamiltonian blocks are replaced
    locally by their symmetric parts only when max(abs(H-H.T)) is at
    most 1e-12 * max(1,max(abs(H))). No overlap arrays are accepted here,
    so physical orthonormality is a prerequisite supplied by the geometry
    stage, not something this function can verify from coefficient shapes.

    The core needs only transposes and matrix multiplication. It does not
    diagonalize, invert, reconstruct an explicit physical-space projector,
    select a reference root, or evaluate a perturbative correction.
    """
    for name, array in (
        ("h_rr", h_rr),
        ("h_rt", h_rt),
        ("h_tt", h_tt),
        ("x_ref", x_ref),
        ("external_ref_coeff", external_ref_coeff),
        ("external_candidate_coeff", external_candidate_coeff),
    ):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a NumPy array with float64 dtype")
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")

    nr, nt = h_rt.shape
    if nr == 0 or h_rr.shape != (nr, nr) or h_tt.shape != (nt, nt):
        raise ValueError("expected H shapes (nr,nr), (nr,nt), (nt,nt) with nr > 0")
    if x_ref.shape[0] != nr or not 1 <= x_ref.shape[1] <= nr:
        raise ValueError("x_ref must have shape (nr,r) with 1 <= r <= nr")
    nexternal = external_candidate_coeff.shape[1]
    if (
        external_ref_coeff.shape != (nr, nexternal)
        or external_candidate_coeff.shape[0] != nt
        or nexternal > nt
    ):
        raise ValueError("external coefficients must have shapes (nr,s), (nt,s) with s <= nt")

    h_rr = _symmetric_copy(h_rr, "h_rr")
    h_tt = _symmetric_copy(h_tt, "h_tt")

    h_reference = x_ref.T @ h_rr @ x_ref
    h_coupling = x_ref.T @ (h_rr @ external_ref_coeff + h_rt @ external_candidate_coeff)
    h_external = (
        external_ref_coeff.T @ h_rr @ external_ref_coeff
        + external_ref_coeff.T @ h_rt @ external_candidate_coeff
        + external_candidate_coeff.T @ h_rt.T @ external_ref_coeff
        + external_candidate_coeff.T @ h_tt @ external_candidate_coeff
    )
    return {
        "h_reference": h_reference,
        "h_coupling": h_coupling,
        "h_external": h_external,
    }


def orthogonalize_mr_spaces_blockwise(
    s_rr,
    s_rt,
    s_tt,
    reference_groups,
    external_groups,
    *,
    reference_cut=1e-10,
    external_cut=1e-10,
):
    """Apply the same MR geometry using exact disjoint Gram blocks.

    s_rr/rt/tt have the original float64 matrix contract. Group labels are
    nonnegative int64 vectors (nr,) and (nt,), with every external group
    also present among references. All overlaps across distinct groups
    must be zero within 1e-12*max(1,max(abs(S))); otherwise reject them.
    For the projected site basis, electron translation orbits provide
    these groups. Hamiltonian blocks need not respect this grouping.

    Only the Gram eigensolves are decomposed. Both cutoff scales and the
    PSD check use complete spectra over all groups, exactly as in
    orthogonalize_mr_spaces. Return the same fields, shapes and dtypes.
    The basis within degenerate eigenspaces can differ; compare physical
    projectors and energies, not eigenvector signs or coordinates.
    Full dense Hamiltonian transformations and external responses remain
    unchanged. This reduces eigensolver memory without changing the PT2
    partition. Inputs are not modified; invalid types/shapes/grouping/
    spectra and zero reference rank use the existing failure conventions.

    The owned symmetric TT copy is reused for the returned external_metric
    (nt,nt). Schur subtraction uses at most 128-row products and symmetric
    averaging uses small tile pairs, rather than full-size temporaries.
    Both external transforms remain separate float64 (nt,s) arrays as in
    the original interface; the caller's S blocks are never modified.
    """
    for name, matrix in (("s_rr", s_rr), ("s_rt", s_rt), ("s_tt", s_tt)):
        if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be a finite matrix")
    nr, nt = s_rt.shape
    if nr == 0 or s_rr.shape != (nr, nr) or s_tt.shape != (nt, nt):
        raise ValueError("incompatible Gram shapes or empty reference")
    for name, labels, size in (
        ("reference_groups", reference_groups, nr),
        ("external_groups", external_groups, nt),
    ):
        if not isinstance(labels, np.ndarray) or labels.dtype != np.int64:
            raise TypeError(f"{name} must be an int64 NumPy array")
        if labels.shape != (size,) or np.any(labels < 0):
            raise ValueError(f"{name} must be a nonnegative vector of length {size}")
    if not set(external_groups).issubset(set(reference_groups)):
        raise ValueError("every external group must be a reference group")
    for name, cut in (("reference_cut", reference_cut), ("external_cut", external_cut)):
        if (
            isinstance(cut, (bool, np.bool_))
            or not isinstance(cut, (int, float, np.integer, np.floating))
            or not np.isfinite(cut)
            or cut <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    rr, tt = _symmetric_copy(s_rr, "s_rr"), _symmetric_copy(s_tt, "s_tt")
    for matrix, bra, ket in (
        (rr, reference_groups, reference_groups),
        (s_rt, reference_groups, external_groups),
        (tt, external_groups, external_groups),
    ):
        scale = max(1.0, linalg.max_abs(matrix))
        for group in np.unique(bra):
            rows, columns = np.flatnonzero(bra == group), np.flatnonzero(ket != group)
            for start in range(0, len(rows), linalg.BLOCK_ROWS):
                cross = matrix[np.ix_(rows[start : start + linalg.BLOCK_ROWS], columns)]
                if linalg.max_abs(cross) > 1e-12 * scale:
                    raise ValueError("Gram blocks couple distinct supplied groups")
    parts, joint_values, ref_values = [], [], []
    for group in np.unique(reference_groups):
        ir, it = np.flatnonzero(reference_groups == group), np.flatnonzero(external_groups == group)
        rblock, cross, tblock = rr[np.ix_(ir, ir)], s_rt[np.ix_(ir, it)], tt[np.ix_(it, it)]
        joint_values.append(np.linalg.eigvalsh(np.block([[rblock, cross], [cross.T, tblock]])))
        vals, vecs = np.linalg.eigh(rblock)
        ref_values.append(vals)
        parts.append((ir, it, vals, vecs))
    _check_positive_semidefinite(np.concatenate(joint_values), "joint Gram matrix")
    reference_values = np.sort(np.concatenate(ref_values))
    _check_positive_semidefinite(reference_values, "s_rr")
    threshold = reference_cut * max(1.0, float(reference_values[-1]))
    ref_rank = int(np.count_nonzero(reference_values > threshold))
    if ref_rank == 0:
        raise ValueError("no reference eigenvalues exceed the reference_cut threshold")
    x_ref = np.zeros((nr, ref_rank))
    retained_values, offset = [], 0
    for ir, _it, vals, vecs in parts:
        keep = vals > threshold
        count = int(np.count_nonzero(keep))
        x_ref[np.ix_(ir, np.arange(offset, offset + count))] = vecs[:, keep] / np.sqrt(vals[keep])
        retained_values.extend(vals[keep])
        offset += count
    x_ref = x_ref[:, np.argsort(retained_values)]
    subtraction = x_ref @ (x_ref.T @ s_rt)
    metric = tt
    for start in range(0, nt, linalg.BLOCK_ROWS):
        rows = slice(start, start + linalg.BLOCK_ROWS)
        product = s_rt[:, rows].T @ subtraction
        metric[rows] -= product
        del product
    linalg.symmetrize_in_place(metric)
    external_parts, external_values = [], []
    for _ir, it, _vals, _vecs in parts:
        vals, vecs = np.linalg.eigh(metric[np.ix_(it, it)])
        external_parts.append((it, vals, vecs))
        external_values.append(vals)
    external_values = np.sort(np.concatenate(external_values))
    _check_positive_semidefinite(external_values, "external metric")
    threshold = external_cut * max(1.0, float(np.max(external_values, initial=0.0)))
    ext_rank = int(np.count_nonzero(external_values > threshold))
    x_ext = np.zeros((nt, ext_rank))
    retained_values, offset = [], 0
    for it, vals, vecs in external_parts:
        keep = vals > threshold
        count = int(np.count_nonzero(keep))
        x_ext[np.ix_(it, np.arange(offset, offset + count))] = vecs[:, keep] / np.sqrt(vals[keep])
        retained_values.extend(vals[keep])
        offset += count
    x_ext = x_ext[:, np.argsort(retained_values)]
    return {
        "x_ref": x_ref,
        "ref_subtraction": subtraction,
        "external_metric": metric,
        "x_external": x_ext,
        "external_ref_coeff": -subtraction @ x_ext,
        "external_candidate_coeff": x_ext.copy(),
        "reference_eigenvalues": reference_values,
        "external_eigenvalues": external_values,
        "reference_rank": ref_rank,
        "external_rank": ext_rank,
    }
