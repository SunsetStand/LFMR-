"""Symmetry-reduced projected candidate blocks for high phonon cutoffs.

The projected MRPT2 candidate set contains exactly redundant columns when a
seed frame is invariant under lattice translation, and splits into
noninteracting sectors under alpha/beta exchange when both spin counts are
equal. Removing that redundancy leaves the retained external space seen by
the ground reference root, and hence E0 + E2, unchanged:

* Invariant seed. For translation-invariant frame B, the projected column
  (B, address[r][J], roll(n, r)) equals sign[r][J] * character**r times
  column (B, J, n). Each orbit of (determinant, configuration) pairs is
  replaced by one column scaled by sqrt(orbit size); columns annihilated
  by the projection are dropped. The candidate operator T T^T, and with it
  every nonzero external Gram eigenvalue and retained direction, is
  identical to the complete list.
* Spin exchange. F|A,I,n> = (-1)**(na*nb) |A,FI,n> maps a seed to itself,
  because LF displacements depend on total site density. H, S and the
  translation projection commute with F, so the external Gram and
  Hamiltonian are block diagonal in F parity. Only the parity sector of the
  reference ground root couples to it. For seeds without exact translation
  invariance, candidates are replaced by orthonormal F-adapted pairs of
  that parity. The other parity sector is evaluated for its overlap
  spectrum only, so the external cutoff keeps its complete-list scale.

References are not reduced. Gram groups are electronic translation orbits
merged under F; Hamiltonian couplings between groups are all retained.
"""

import numpy as np

from .. import basis
from ..model import HubbardHolstein
from ..phonon import multimode_displacement_overlap, multimode_phonon_operator_matrices
from ..symmetry import _translation_electron_labels
from . import linalg
from .kernels import _validate_mr_candidate_inputs, build_projected_mr_candidate_blocks


def _electronic_tables(tmat, nsite, nelec):
    str_a, link_a = basis.make_electron_basis(nsite, nelec[0])
    str_b, link_b = basis.make_electron_basis(nsite, nelec[1])
    sites = np.arange(nsite, dtype=np.int64)
    occ_a = (str_a[:, None] >> sites[None, :] & 1).astype(np.float64)
    occ_b = (str_b[:, None] >> sites[None, :] & 1).astype(np.float64)
    density = (occ_a[:, None, :] + occ_b[None, :, :]).reshape(-1, nsite)
    double = (occ_a[:, None, :] * occ_b[None, :, :]).sum(axis=2).reshape(-1)
    # Same construction and orientation (bra, ket) as lf_frame_hopping_hamiltonian.
    hops = []
    for strings, link in ((str_a, link_a), (str_b, link_b)):
        create, annihilate, target, sign = link.transpose(2, 0, 1)
        source = np.broadcast_to(np.arange(len(strings))[:, None], target.shape)
        matrix = np.zeros((len(strings), len(strings)))
        np.add.at(matrix, (target, source), sign * tmat[create, annihilate])
        hops.append(matrix)
    hopping = np.kron(hops[0], np.eye(len(str_b))) + np.kron(np.eye(len(str_a)), hops[1])
    na, nb = len(str_a), len(str_b)
    flip = None
    if nelec[0] == nelec[1]:
        ia, ib = np.divmod(np.arange(na * nb), nb)
        flip = ib * nb + ia
    return density, double, hopping, flip


def translation_invariant(lam, shift):
    """Exact (bitwise) invariance of one frame under every lattice translation."""
    nsite = len(shift)
    return all(
        np.array_equal(np.roll(lam, r, axis=(0, 1)), lam)
        and np.array_equal(np.roll(shift, r), shift)
        for r in range(1, nsite)
    )


def _config_permutations(occupations):
    nsite = occupations.shape[1]
    index = {tuple(row): k for k, row in enumerate(occupations.tolist())}
    perms = np.empty((nsite, len(occupations)), dtype=np.int64)
    for r in range(nsite):
        rolled = np.roll(occupations, r, axis=1)
        try:
            perms[r] = [index[tuple(row)] for row in rolled.tolist()]
        except KeyError as exc:
            raise ValueError("configuration list must be closed under translation") from exc
    return perms


def reduced_candidate_labels(lam, shift, nelec, occupations, character, parity, flip):
    """Electronic labels (seed, [(det, coeff)], config indices, weights).

    Column (label, k) is sum_j coeff_j * weights[k] * P|seed, det_j, configs[k]>.
    parity=None disables spin-exchange adaptation; flip is the determinant
    map I -> FI from _electronic_tables.
    """
    nsite = lam.shape[1]
    address, sign, _ = _translation_electron_labels(nsite, nelec)
    ndet = address.shape[1]
    nconfig = len(occupations)
    if parity is None:
        flip = None
    elif flip is None:
        raise ValueError("spin-exchange parity requires equal spin counts")
    else:
        epsilon = (-1) ** (nelec[0] * nelec[1])
    perms = _config_permutations(occupations) if nconfig else None
    labels, kinds = [], []
    for A in range(len(lam)):
        if nconfig and translation_invariant(lam[A], shift[A]):
            kinds.append("translation_orbit")
            for J in range(ndet):
                keep, weight = [], []
                for k in range(nconfig):
                    images = [(int(address[r, J]), int(perms[r, k])) for r in range(nsite)]
                    if min(images) != (J, k):
                        continue
                    stab = [r for r in range(nsite) if images[r] == (J, k)]
                    if any(character**r * sign[r, J] < 0 for r in stab):
                        continue
                    keep.append(k)
                    weight.append(np.sqrt(nsite / len(stab)))
                if keep:
                    labels.append((A, [(J, 1.0)], np.array(keep), np.array(weight)))
        elif flip is not None:
            kinds.append("spin_exchange")
            for J in range(ndet):
                F = int(flip[J])
                if F == J:
                    if parity == epsilon:
                        labels.append((A, [(J, 1.0)], np.arange(nconfig), np.ones(nconfig)))
                elif J < F:
                    terms = [(J, 1 / np.sqrt(2)), (F, parity * epsilon / np.sqrt(2))]
                    labels.append((A, terms, np.arange(nconfig), np.ones(nconfig)))
        else:
            kinds.append("complete")
            for J in range(ndet):
                labels.append((A, [(J, 1.0)], np.arange(nconfig), np.ones(nconfig)))
    return labels, kinds


def _merged_groups(nsite, nelec, flip):
    """Translation-orbit labels, merged with their spin-exchange images."""
    _, _, groups = _translation_electron_labels(nsite, nelec)
    groups = groups.copy()
    if flip is None:
        return groups
    while True:
        candidate = np.minimum(groups, groups[flip])
        merged = groups.copy()
        for label in np.unique(groups):
            members = groups == label
            merged[members] = candidate[members].min()
        if np.array_equal(merged, groups):
            return groups
        groups = merged


def _pair_blocks(
    model,
    lam,
    shift,
    nelec,
    character,
    tables,
    rows,
    cols,
    occ,
    want_h,
    label_groups=None,
    want_s=True,
):
    """Raw projected <row|O|col> for row/col label lists; no symmetrization.

    With label_groups (one Gram group per label, rows == cols), S is
    returned as {group: (global columns, block)}: overlaps vanish between
    different electronic groups, so only diagonal group blocks are stored.
    """
    nsite = model.L
    address, sign, _ = _translation_electron_labels(nsite, nelec)
    density, double, hopping, _ = tables
    U, g, omega = float(model.U), float(model.g), float(model.omega)
    row_offsets = np.cumsum([0] + [len(lab[2]) for lab in rows])
    col_offsets = np.cumsum([0] + [len(lab[2]) for lab in cols])
    H = np.zeros((row_offsets[-1], col_offsets[-1])) if want_h else None
    if not want_s:
        S, s_target, s_cols = None, None, None
    elif label_groups is None:
        S = np.zeros((row_offsets[-1], col_offsets[-1]))
        s_target = [(S, slice(row_offsets[i], row_offsets[i + 1])) for i in range(len(rows))]
        s_cols = [slice(col_offsets[j], col_offsets[j + 1]) for j in range(len(cols))]
    else:
        S, s_target, s_cols = {}, [None] * len(rows), [None] * len(cols)
        for group in np.unique(label_groups):
            members = np.flatnonzero(label_groups == group)
            local = np.cumsum([0] + [len(rows[i][2]) for i in members])
            columns = np.concatenate(
                [np.arange(row_offsets[i], row_offsets[i + 1]) for i in members]
            )
            S[int(group)] = (columns, np.zeros((local[-1], local[-1])))
            for k, i in enumerate(members):
                s_target[i] = (S[int(group)][1], slice(local[k], local[k + 1]))
                s_cols[i] = slice(local[k], local[k + 1])
    for A in sorted({lab[0] for lab in rows}):
        eta_a = shift[A] - density @ lam[A].T
        for B in sorted({lab[0] for lab in cols}):
            for r in range(nsite):
                lam_b = np.roll(lam[B], r, axis=(0, 1))
                eta_b = np.roll(shift[B], r) - density @ lam_b.T
                for i, (seed_row, rterms, rconf, rweight) in enumerate(rows):
                    if seed_row != A:
                        continue
                    bra = occ[rconf]
                    rs = slice(row_offsets[i], row_offsets[i + 1])
                    for j, (seed, cterms, cconf, cweight) in enumerate(cols):
                        if seed != B:
                            continue
                        ket = np.roll(occ[cconf], r, axis=1)
                        cs = slice(col_offsets[j], col_offsets[j + 1])
                        sblock = hblock = None
                        for I, c in rterms:
                            for J, d in cterms:
                                K = int(address[r, J])
                                factor = c * d * character**r * sign[r, J]
                                if K == I:
                                    if want_h:
                                        ops = multimode_phonon_operator_matrices(
                                            eta_a[I], eta_b[K], bra, ket
                                        )
                                        overlap = ops["overlap"]
                                        h = (
                                            U * double[I] * overlap
                                            + omega * np.sum(ops["number"], axis=0)
                                            + g
                                            * np.einsum(
                                                "x,xij->ij",
                                                density[I],
                                                ops["annihilation"] + ops["creation"],
                                            )
                                        )
                                        hblock = (
                                            factor * h if hblock is None else hblock + factor * h
                                        )
                                        del ops, h
                                    else:
                                        overlap = multimode_displacement_overlap(
                                            eta_b[K] - eta_a[I], bra, ket
                                        )
                                    sblock = (
                                        factor * overlap
                                        if sblock is None
                                        else sblock + factor * overlap
                                    )
                                if want_h and hopping[I, K] != 0:
                                    f = multimode_displacement_overlap(
                                        eta_b[K] - eta_a[I], bra, ket
                                    )
                                    f *= factor * hopping[I, K]
                                    hblock = f if hblock is None else hblock + f
                        scale = rweight[:, None] * cweight[None, :]
                        if sblock is not None and want_s:
                            if label_groups is not None and label_groups[i] != label_groups[j]:
                                raise ValueError("overlap couples distinct Gram groups")
                            target, trows = s_target[i]
                            target[trows, s_cols[j]] += sblock * scale
                        if hblock is not None:
                            H[rs, cs] += hblock * scale
    return S, H


def _symmetrize(matrix):
    linalg.symmetrize_in_place(matrix)
    return matrix


def build_reduced_projected_blocks(
    model,
    lam,
    shift,
    occupations,
    *,
    character,
    parity=None,
    overlap_only=False,
    grouped_overlap=False,
    defer_h=False,
):
    """Six blocks for a symmetry-reduced projected candidate list.

    Returns (blocks, info). blocks follows build_projected_mr_candidate_blocks
    with complete reference rows; info holds reference/external Gram
    groups, the label list and per-seed reduction kinds. overlap_only skips
    every Hamiltonian candidate block (h_rt/h_tt are then None).
    grouped_overlap stores s_tt only as 's_tt_groups' {group: (columns,
    block)}, with 's_tt' set to None, for lean_mrpt2. defer_h (requires
    grouped_overlap) leaves h_tt unassembled: blocks['h_tt_pairs'](rows, cols)
    returns the raw H block between two candidate-label index lists, so the
    solver can contract one Gram-group pair at a time.
    """
    if not isinstance(model, HubbardHolstein):
        raise TypeError("reduced lattice MRPT2 requires a HubbardHolstein model")
    _validate_mr_candidate_inputs(model.tmat, model.U, model.g, model.omega,
                                  lam, shift, model.nelec, occupations)
    nsite, nelec = model.L, model.nelec
    if (isinstance(character, (bool, np.bool_))
            or not isinstance(character, (int, np.integer))
            or character not in (-1, 1) or (character == -1 and nsite % 2)):
        raise ValueError("character must be +1, or -1 for even L")
    if not np.allclose(model.tmat, np.roll(model.tmat, 1, axis=(0, 1)), atol=1e-12, rtol=0):
        raise ValueError("tmat must be translation invariant")
    if parity is not None and (isinstance(parity, (bool, np.bool_))
                               or not isinstance(parity, (int, np.integer))
                               or parity not in (-1, 1)):
        raise ValueError("parity must be None, +1 or -1")
    if parity is not None and nelec[0] != nelec[1]:
        raise ValueError("spin-exchange parity requires equal spin counts")
    for name, flag in (("overlap_only", overlap_only), ("grouped_overlap", grouped_overlap),
                        ("defer_h", defer_h)):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(f"{name} must be bool")
    if defer_h and not grouped_overlap:
        raise ValueError("defer_h requires grouped_overlap")
    occ = occupations.copy()
    if len(np.unique(occ, axis=0)) != len(occ):
        raise ValueError("reduced occupations must be unique")
    if len(occ):
        _config_permutations(occ)
    tables = _electronic_tables(model.tmat, nsite, nelec)
    labels, kinds = reduced_candidate_labels(lam, shift, nelec, occ, character, parity, tables[3])
    groups = _merged_groups(nsite, nelec, tables[3] if parity is not None else None)
    if grouped_overlap:
        # Contiguous Gram groups let the lean solver slice H_TT without copies.
        labels = sorted(labels, key=lambda label: groups[label[1][0][0]])
    reference = build_projected_mr_candidate_blocks(
        model.tmat,
        model.U,
        model.g,
        model.omega,
        lam,
        shift,
        nelec,
        np.empty((0, nsite), dtype=np.int64),
        character=character,
    )
    ndet = len(tables[0])
    vacuum = np.zeros((1, nsite), dtype=np.int64)
    occ_with_vacuum = np.concatenate((vacuum, occ))
    ref_labels = [
        (A, [(J, 1.0)], np.zeros(1, dtype=np.int64), np.ones(1))
        for A in range(len(lam))
        for J in range(ndet)
    ]
    cand_labels = [(A, terms, conf + 1, w) for A, terms, conf, w in labels]
    want_h = not overlap_only
    s_rt, h_rt = _pair_blocks(
        model,
        lam,
        shift,
        nelec,
        character,
        tables,
        ref_labels,
        cand_labels,
        occ_with_vacuum,
        want_h,
    )
    s_tr, h_tr = _pair_blocks(
        model,
        lam,
        shift,
        nelec,
        character,
        tables,
        cand_labels,
        ref_labels,
        occ_with_vacuum,
        want_h,
    )
    s_rt = 0.5 * (s_rt + s_tr.T)
    if want_h:
        h_rt = 0.5 * (h_rt + h_tr.T)
    del s_tr, h_tr
    label_groups = np.array([groups[terms[0][0]] for _, terms, _, _ in labels], dtype=np.int64)
    if defer_h and not grouped_overlap:
        raise ValueError("defer_h requires grouped_overlap")
    s_tt, h_tt = _pair_blocks(
        model,
        lam,
        shift,
        nelec,
        character,
        tables,
        cand_labels,
        cand_labels,
        occ_with_vacuum,
        want_h and not defer_h,
        label_groups if grouped_overlap else None,
    )
    if grouped_overlap:
        for _, block in s_tt.values():
            _symmetrize(block)
    else:
        _symmetrize(s_tt)
    if want_h and not defer_h:
        _symmetrize(h_tt)
    reference_groups = np.tile(groups, len(lam)).astype(np.int64)
    external_groups = (
        np.concatenate(
            [
                np.full(len(conf), groups[terms[0][0]], dtype=np.int64)
                for _, terms, conf, _ in labels
            ]
        )
        if labels
        else np.empty(0, dtype=np.int64)
    )
    blocks = {
        "s_rr": reference["s_rr"],
        "h_rr": reference["h_rr"],
        "s_rt": s_rt,
        "h_rt": h_rt if want_h else None,
        "s_tt": None if grouped_overlap else s_tt,
        "h_tt": h_tt if want_h else None,
    }
    if grouped_overlap:
        blocks["s_tt_groups"] = s_tt
    if defer_h and want_h:

        def h_tt_pairs(rows, cols):
            _, block = _pair_blocks(
                model,
                lam,
                shift,
                nelec,
                character,
                tables,
                [cand_labels[i] for i in rows],
                [cand_labels[j] for j in cols],
                occ_with_vacuum,
                True,
                want_s=False,
            )
            return block

        blocks["h_tt_pairs"] = h_tt_pairs
    info = {
        "reference_groups": reference_groups,
        "external_groups": external_groups,
        "labels": labels,
        "kinds": kinds,
        "parity": parity,
        "label_groups": label_groups,
    }
    return blocks, info


def reference_spin_parity(mr):
    """F parity of the projected MR ground root, from its raw seed coefficients."""
    m = mr.model
    nelec = m.nelec
    if nelec[0] != nelec[1]:
        return None
    _, _, _, flip = _electronic_tables(m.tmat, m.L, nelec)
    s = mr.result["smat"]
    h = mr.result["hmat"]
    vals, vecs = np.linalg.eigh(0.5 * (s + s.T))
    keep = vals > mr.overlap_cut * max(1.0, vals[-1])
    x = vecs[:, keep] / np.sqrt(vals[keep])
    e, u = np.linalg.eigh(x.T @ (0.5 * (h + h.T)) @ x)
    c = x @ u[:, 0]
    ndet = len(flip)
    perm = np.concatenate([A * ndet + flip for A in range(len(c) // ndet)])
    epsilon = (-1) ** (nelec[0] * nelec[1])
    value = epsilon * c @ s[:, perm] @ c / (c @ s @ c)
    parity = int(np.rint(value))
    if abs(value - parity) > 1e-8 or abs(parity) != 1:
        raise ValueError(f"MR ground root has no definite spin-exchange parity: {value}")
    if abs(e[0] - mr.e_tot) > 1e-8:
        raise ValueError("reference ground root differs from the MR energy")
    return parity


def _eigh(matrix, values_only=False):
    """numpy eigh/eigvalsh; on LAPACK nonconvergence retry SciPy evr, then ev."""
    import scipy.linalg

    try:
        return np.linalg.eigvalsh(matrix) if values_only else np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        for driver in ("evr", "ev"):
            try:
                return scipy.linalg.eigh(matrix, eigvals_only=values_only, driver=driver)
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
                continue
        raise


def _columns(index):
    """A slice for a contiguous ascending index array, else the array itself."""
    if len(index) and index[-1] - index[0] + 1 == len(index) and np.all(np.diff(index) == 1):
        return slice(int(index[0]), int(index[-1]) + 1)
    return index


def _reference_geometry(s_rr, s_rt, s_groups, reference_groups, reference_cut):
    """Reference coordinates exactly as orthogonalize_mr_spaces_blockwise."""
    from .space import _check_positive_semidefinite

    rr = linalg.symmetric_copy(s_rr, "s_rr")
    scale = max(1.0, linalg.max_abs(rr))
    for group in np.unique(reference_groups):
        rows = np.flatnonzero(reference_groups == group)
        if (
            linalg.max_abs(rr[np.ix_(rows, np.flatnonzero(reference_groups != group))])
            > 1e-12 * scale
        ):
            raise ValueError("Gram blocks couple distinct supplied groups")
    parts, joint, ref_values = [], [], []
    for group in np.unique(reference_groups):
        ir = np.flatnonzero(reference_groups == group)
        it, tblock = s_groups.get(int(group), (np.empty(0, dtype=np.int64), np.zeros((0, 0))))
        rblock, cross = rr[np.ix_(ir, ir)], s_rt[np.ix_(ir, it)]
        joint.append(_eigh(np.block([[rblock, cross], [cross.T, tblock]]), values_only=True))
        vals, vecs = _eigh(rblock)
        ref_values.append(vals)
        parts.append((ir, vals, vecs))
    _check_positive_semidefinite(np.concatenate(joint), "joint Gram matrix")
    reference_values = np.sort(np.concatenate(ref_values))
    _check_positive_semidefinite(reference_values, "s_rr")
    threshold = reference_cut * max(1.0, float(reference_values[-1]))
    rank = int(np.count_nonzero(reference_values > threshold))
    if rank == 0:
        raise ValueError("no reference eigenvalues exceed the reference_cut threshold")
    x_ref = np.zeros((len(rr), rank))
    retained, offset = [], 0
    for ir, vals, vecs in parts:
        keep = vals > threshold
        count = int(np.count_nonzero(keep))
        x_ref[np.ix_(ir, np.arange(offset, offset + count))] = vecs[:, keep] / np.sqrt(vals[keep])
        retained.extend(vals[keep])
        offset += count
    return rr, x_ref[:, np.argsort(retained)], reference_values


def _external_metrics(s_rt, s_groups, subtraction):
    """Reference-subtracted Gram blocks, group by group."""
    metrics = {}
    for group, (it, tblock) in s_groups.items():
        metric = tblock - s_rt[:, it].T @ subtraction[:, it]
        linalg.symmetrize_in_place(metric)
        metrics[group] = (it, metric)
    return metrics


def external_gram_maximum(blocks, info, reference_cut):
    """Largest reference-subtracted external Gram eigenvalue (grouped overlap)."""
    _, x_ref, _ = _reference_geometry(
        blocks["s_rr"],
        blocks["s_rt"],
        blocks["s_tt_groups"],
        info["reference_groups"],
        reference_cut,
    )
    subtraction = x_ref @ (x_ref.T @ blocks["s_rt"])
    values = [
        _eigh(metric, values_only=True)
        for _, metric in _external_metrics(
            blocks["s_rt"], blocks["s_tt_groups"], subtraction
        ).values()
    ]
    return float(np.max(np.concatenate(values), initial=0.0)) if values else 0.0


def lean_mrpt2(
    blocks,
    info,
    *,
    reference_cut,
    external_cut,
    external_scale=0.0,
    reference_gap_tol=1e-10,
    denominator_tol=1e-10,
    release_h=False,
):
    """float64 MRPT2 equal to mrpt2_from_blocks, with grouped overlap storage.

    blocks comes from build_reduced_projected_blocks(grouped_overlap=True).
    Steps, cut rules, spectral/PSD/asymmetry checks and returned diagnostics
    follow mrpt2_from_blocks + solve_mrpt2_blocks; only storage differs:
    S_TT, its reference-subtracted metric and the external coordinates are
    kept per Gram group; with release_h, S_TT groups and h_tt are removed
    from blocks and released once no longer needed (last solve at this M).
    LAPACK eigensolver nonconvergence is retried with SciPy evr, then ev. external_scale is a lower bound for
    the largest external Gram eigenvalue used by the external cut (the
    complete-list maximum when a parity sector was dropped).
    """
    from .pt2 import _validate_mrpt2_gaps
    from .space import _check_positive_semidefinite

    for name, value in (("reference_cut", reference_cut), ("external_cut", external_cut)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if (isinstance(external_scale, (bool, np.bool_))
            or not isinstance(external_scale, (int, float, np.integer, np.floating))
            or not np.isfinite(external_scale) or external_scale < 0):
        raise ValueError("external_scale must be finite and nonnegative")
    s_rt, h_rr, h_rt = blocks["s_rt"], blocks["h_rr"], blocks["h_rt"]
    s_groups = blocks["s_tt_groups"]
    rg, tg = info["reference_groups"], info["external_groups"]
    nr, nt = s_rt.shape
    scale = max(1.0, linalg.max_abs(s_rt), *(linalg.max_abs(b) for _, b in s_groups.values()))
    for group in np.unique(rg):
        rows, columns = np.flatnonzero(rg == group), np.flatnonzero(tg != group)
        if linalg.max_abs(s_rt[np.ix_(rows, columns)]) > 1e-12 * scale:
            raise ValueError("Gram blocks couple distinct supplied groups")
    rr, x_ref, reference_values = _reference_geometry(
        blocks["s_rr"], s_rt, s_groups, rg, reference_cut
    )
    subtraction = x_ref @ (x_ref.T @ s_rt)
    metrics = _external_metrics(s_rt, s_groups, subtraction)
    parts, values = [], []
    for group, (it, metric) in metrics.items():
        vals, vecs = _eigh(metric)
        parts.append((group, it, vals, vecs))
        values.append(vals)
    external_values = np.sort(np.concatenate(values)) if values else np.empty(0)
    _check_positive_semidefinite(external_values, "external metric")
    threshold = external_cut * max(
        1.0, float(np.max(external_values, initial=0.0)), float(external_scale)
    )
    kept = []
    for group, it, vals, vecs in parts:
        keep = vals > threshold
        kept.append((group, it, vals[keep], vecs[:, keep] / np.sqrt(vals[keep])))
    order = np.argsort(np.concatenate([v for _, _, v, _ in kept]), kind="stable")
    position = np.empty_like(order)
    position[order] = np.arange(len(order))
    s, offset, coords = len(order), 0, []
    for group, it, vals, y in kept:
        coords.append((group, it, position[offset : offset + len(vals)], y))
        offset += len(vals)
    # External coordinates: candidate rows Y (block diagonal), reference rows -B Y.
    cr = np.zeros((nr, s))
    s_rt_ct = np.zeros((nr, s))
    h_rt_ct = np.zeros((nr, s))
    external_metric_error = 0.0
    coeff_scale = np.zeros(s)
    for group, it, cols, y in coords:
        cr[:, cols] = -subtraction[:, it] @ y
        s_rt_ct[:, cols] = s_rt[:, it] @ y
        h_rt_ct[:, cols] = h_rt[:, it] @ y
        coeff_scale[cols] = np.sum(np.abs(y), axis=0)
        metric = metrics[group][1]
        external_metric_error = max(external_metric_error, linalg.identity_error(y.T @ metric @ y))
    del metrics, subtraction
    if release_h:
        del s_groups, blocks["s_tt_groups"]
    coeff_scale = max(1.0, float(np.max(coeff_scale + np.sum(np.abs(cr), axis=0), initial=0.0)))
    h_rr_sym = linalg.symmetric_copy(h_rr, "h_rr")
    h_external = np.zeros((s, s))
    h_tt = blocks.pop("h_tt") if release_h else blocks["h_tt"]
    if h_tt is None:
        # Deferred H_TT: build, legacy-symmetrize and contract one group pair at a time.
        pairs, label_groups = blocks["h_tt_pairs"], info["label_groups"]
        members = {int(g): np.flatnonzero(label_groups == g) for g in np.unique(label_groups)}
        active = [c for c in coords if c[3].shape[1]]
        h_tt_max = 0.0
        for k, (g, _, cols_g, y_g) in enumerate(active):
            for h, _, cols_h, y_h in active[k:]:
                block = pairs(members[g], members[h])
                if g == h:
                    linalg.symmetrize_in_place(block)
                else:
                    block += pairs(members[h], members[g]).T
                    block *= 0.5
                h_tt_max = max(h_tt_max, linalg.max_abs(block))
                value = y_g.T @ block @ y_h
                del block
                h_external[np.ix_(cols_g, cols_h)] = value
                if g != h:
                    h_external[np.ix_(cols_h, cols_g)] = value.T
        hscale = max(1.0, linalg.max_abs(h_rr), linalg.max_abs(h_rt), h_tt_max)
    else:
        hscale = max(1.0, linalg.max_abs(h_rr), linalg.max_abs(h_rt), linalg.max_abs(h_tt))
        if linalg.symmetry_error(h_tt) > 1e-12 * max(1.0, linalg.max_abs(h_tt)):
            raise ValueError("h_tt must be symmetric within relative tolerance 1e-12")
        for _, it_h, cols_h, y_h in coords:
            action = h_tt[:, _columns(it_h)] @ y_h
            for _, it_g, cols_g, y_g in coords:
                h_external[np.ix_(cols_g, cols_h)] = y_g.T @ action[_columns(it_g)]
            del action
    del h_tt
    h_external += cr.T @ h_rr_sym @ cr + cr.T @ h_rt_ct + h_rt_ct.T @ cr
    hblocks = {
        "h_reference": x_ref.T @ h_rr_sym @ x_ref,
        "h_coupling": x_ref.T @ (h_rr_sym @ cr + h_rt_ct),
        "h_external": h_external,
    }
    roundoff_bound = 10 * np.finfo(np.float64).eps * (nr + nt) * hscale * coeff_scale**2
    asymmetries = {}
    for name in ("h_reference", "h_external"):
        matrix = hblocks[name]
        asymmetry = linalg.symmetry_error(matrix)
        if asymmetry > min(roundoff_bound, 1e-7 * max(1.0, linalg.max_abs(matrix))):
            raise ValueError(f"{name} congruence asymmetry exceeds roundoff bound: {asymmetry}")
        asymmetries[name + "_asymmetry"] = asymmetry
        linalg.symmetrize_in_place(matrix)
    reference_energy, reference_vectors = _eigh(hblocks["h_reference"])
    external_eigenvalues = _eigh(h_external, values_only=True)
    denominators = external_eigenvalues - reference_energy[0]
    _validate_mrpt2_gaps(reference_energy, denominators, reference_gap_tol, denominator_tol)
    u0 = reference_vectors[:, 0]
    v = hblocks["h_coupling"].T @ u0
    h_external[np.diag_indices(s)] -= reference_energy[0]
    a = np.linalg.solve(h_external, -v)
    e2 = float(v @ a)
    reference_metric_error = linalg.identity_error(x_ref.T @ rr @ x_ref)
    cross_metric_error = float(np.max(np.abs(x_ref.T @ (rr @ cr + s_rt_ct)), initial=0.0))
    return {
        "reference_energy": float(reference_energy[0]),
        "second_order_energy": e2,
        "corrected_energy": float(reference_energy[0] + e2),
        "reference_gap": float(reference_energy[1] - reference_energy[0])
        if len(reference_energy) > 1
        else float("inf"),
        "minimum_denominator": float(np.min(denominators)) if s else float("inf"),
        "reference_residual": float(
            np.linalg.norm(hblocks["h_reference"] @ u0 - reference_energy[0] * u0)
        ),
        "response_residual": float(np.linalg.norm(h_external @ a + v)),
        "first_order_norm_squared": float(a @ a),
        "reference_rank": x_ref.shape[1],
        "external_rank": s,
        "reference_metric_error": reference_metric_error,
        "external_metric_error": external_metric_error,
        "cross_metric_error": cross_metric_error,
        "reference_overlap_eigenvalues": reference_values,
        "external_overlap_eigenvalues": external_values,
        **asymmetries,
    }
