"""Analytic group expectation values without allocating a group-orbit matrix."""

from math import comb

import numpy as np

from .._driver import require_reference
from .general_kernels import FRAME_SPACE_LIMIT, _validate_inputs
from .point_group_kernels import _check_symmetry, lf_group_relative_blocks


def lf_group_state_diagnostics(model, lam, coeff, shift=None, *, symmetry, irrep,
                                row=0, determinants=None):
    """Check a projected state supplied as arrays, without an LFMR lifecycle.

    ``coeff`` is finite float64/complex128 with structured shape
    (F,d,nstra,nstrb), or (F,d,D) for a selected determinant list. It describes
    sum_AnI coeff[AnI] P[row,n]|A,I>. Frames, symmetry and determinant order
    follow lf_group_frame_blocks. The same dense guards apply before weight
    allocation; the relative-kernel generator checks Hamiltonian invariance.

    For each g return <Psi|R(g)|Psi>/<Psi|Psi>. A fixed irrep row m has
    expectation D_mm(g), including non-Abelian partner states. The central
    character projector gives its representation purity. This independent
    contraction costs O(|G|^2 N^2 + |G|^3 N) arithmetic and O(|G|N+N^2)
    storage (N=frames*determinants); it is optional for larger groups.
    """
    shift = _validate_inputs(model, lam, shift, determinants)
    group = _check_symmetry(model, symmetry)
    if not isinstance(irrep, str) or irrep not in group.irrep_matrices:
        raise ValueError("irrep must label an irreducible representation of symmetry.group")
    dmat = group.irrep_matrices[irrep]
    d = dmat.shape[-1]
    if (isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer))
            or not 0 <= row < d):
        raise ValueError("row must be an integer partner index in [0,irrep dimension)")
    electronic_shape = (tuple(comb(model.norb, n) for n in model.nelec)
                        if determinants is None else (len(determinants),))
    f = len(lam)
    if f * d * int(np.prod(electronic_shape)) > FRAME_SPACE_LIMIT:
        raise NotImplementedError(
            f"group state diagnostics require <= {FRAME_SPACE_LIMIT} basis states")
    if not isinstance(coeff, np.ndarray) or coeff.dtype not in (np.float64, np.complex128):
        raise TypeError("coeff must be a float64 or complex128 NumPy array")
    if coeff.shape != (f, d, *electronic_shape):
        raise ValueError("coeff must match the structured projected frame/irrep/electronic shape")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("coeff must contain only finite values")
    coeff = coeff.reshape(f, d, -1)
    weights = (np.longdouble(d) / group.order * np.einsum(
        "gn,AnI->gAI", dmat[:, row, :].conj().astype(np.clongdouble),
        coeff.astype(np.clongdouble))).reshape(group.order, -1)
    table = group.multiplication
    identity = next(i for i in range(group.order)
                    if np.array_equal(table[i], np.arange(group.order)))
    inverse = np.array([np.flatnonzero(table[g] == identity)[0] for g in range(group.order)])
    expectations = np.zeros(group.order, dtype=np.clongdouble)
    for g, _, overlap in lf_group_relative_blocks(
            model, lam, shift, symmetry=symmetry, determinants=determinants):
        left = weights.conj() @ overlap.reshape(weights.shape[1], weights.shape[1]).astype(np.clongdouble)
        for h in range(group.order):
            addresses = table[table[inverse[h], np.arange(group.order)], g]
            expectations[h] += np.einsum("ai,ai->", left, weights[addresses])
    norm = expectations[identity]
    if abs(norm.imag) > 1e-10 or norm.real <= 0:
        raise ValueError("group contraction gave an invalid projected norm")
    expectations /= norm.real
    characters = np.trace(dmat, axis1=1, axis2=2)
    purity = d / group.order * np.vdot(characters, expectations)
    result = {"group_expectations": np.asarray(expectations, dtype=np.complex128),
              "representation_purity": complex(purity),
              "representation_purity_error": float(abs(purity - 1.)),
              "row_transformation_error": float(np.max(np.abs(
                  expectations - dmat[:, row, row]))),
              "norm_consistency_error": float(abs(norm - 1.))}
    return result


def lfmr_symmetry_diagnostics(reference):
    """Check a converged point-group LFMR state and update its result."""
    require_reference(reference)
    if reference.symmetry is None:
        raise ValueError("symmetry diagnostics require a point-group projected LFMR")
    result = lf_group_state_diagnostics(
        reference.model, reference.lam, reference.result["coeff"], reference.shift,
        symmetry=reference.symmetry, irrep=reference.irrep, row=reference.row,
        determinants=reference.determinants)
    reference.result.update(result)
    return result
