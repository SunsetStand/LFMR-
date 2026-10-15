"""Finite-group projection of coherent LF frames in a common electronic basis.

Group actions rotate the full conditional ket displacement, rather than
assuming the rotated state retains a density-diagonal LF parametrization.
"""

from math import comb

import numpy as np

from .._electronic import ElectronicSpace
from .._selected_electronic import SelectedElectronicSpace
from ..joint_symmetry import JointSymmetry
from ..model import ElectronPhononModel
from ..point_group import apply_electronic_symmetry
from .general_kernels import ELECTRONIC_LIMIT, FRAME_SPACE_LIMIT, _validate_inputs


def _check_symmetry(model, symmetry):
    if not isinstance(symmetry, JointSymmetry):
        raise TypeError("symmetry must be a validated JointSymmetry")
    group = symmetry.group
    for name, action, width in (("orbital_actions", symmetry.orbital_actions, model.norb),
                                ("photon_actions", symmetry.photon_actions, model.nmode)):
        if not isinstance(action, np.ndarray) or action.dtype != np.float64:
            raise TypeError(f"symmetry.{name} must be a float64 NumPy array")
        if action.shape != (group.order, width, width) or not np.all(np.isfinite(action)):
            raise ValueError(f"symmetry.{name} has invalid shape or nonfinite entries")
        if not np.allclose(action.transpose(0, 2, 1) @ action, np.eye(width),
                           atol=1e-12, rtol=0):
            raise ValueError(f"symmetry.{name} must be orthogonal")
        for g in range(group.order):
            if not np.allclose(action[g] @ action, action[group.multiplication[g]],
                               atol=1e-10, rtol=0):
                raise ValueError(f"symmetry.{name} violates group multiplication")
    frequencies = np.diag(model.omega)
    if not np.allclose(symmetry.photon_actions.transpose(0, 2, 1)
                       @ frequencies @ symmetry.photon_actions, frequencies,
                       atol=1e-11, rtol=1e-11):
        raise ValueError("photon group actions must preserve oscillator frequencies")
    return group


def _selected_actions(model, space, orbital_actions):
    """Selected determinant minors, checking closure without full FCI arrays."""
    occupied = [{int(bit): np.flatnonzero((int(bit) >> np.arange(model.norb)) & 1)
                 for bit in space.determinants[:, spin]} for spin in range(2)]
    actions = np.ones((len(orbital_actions), space.size, space.size))
    for g, orbital in enumerate(orbital_actions):
        for spin in range(2):
            minors = {(bra, ket): np.linalg.det(orbital[np.ix_(bra_occ, ket_occ)])
                      for bra, bra_occ in occupied[spin].items()
                      for ket, ket_occ in occupied[spin].items()}
            actions[g] *= np.array([[minors[int(bra), int(ket)]
                                     for ket in space.determinants[:, spin]]
                                    for bra in space.determinants[:, spin]])
        # A unitary full-space action has unit column norms. Missing norm in
        # this exact determinant subblock measures leakage outside the list.
        if not np.allclose(actions[g].T @ actions[g], np.eye(space.size),
                           atol=1e-11, rtol=0):
            raise ValueError("selected determinant list is not closed under electronic symmetry")
    return actions


def _electronic_actions(model, space, symmetry, determinants):
    if determinants is not None:
        return _selected_actions(model, space, symmetry.orbital_actions)
    actions = np.empty((symmetry.group.order, space.size, space.size))
    for ket in range(space.size):
        ci = np.zeros(space.shape)
        ci.flat[ket] = 1.
        for g, orbital in enumerate(symmetry.orbital_actions):
            actions[g, :, ket] = apply_electronic_symmetry(
                ci, model.norb, model.nelec, orbital).ravel()
    return actions


def electronic_actions(model, symmetry, *, determinants=None):
    """Return active float64 T[g,bra,ket] in full/selected determinant order.

    Full FCI uses the PySCF CI wrapper; a selected list uses occupied minors
    and must be invariant. At most 256 determinants are supported. This
    function never enumerates a complete FCI space for selected input.
    """
    if not isinstance(model, ElectronPhononModel):
        raise TypeError("model must be an ElectronPhononModel")
    dummy = np.zeros((1, model.nmode, model.norb), dtype=np.float64)
    _validate_inputs(model, dummy, None, determinants)
    _check_symmetry(model, symmetry)
    space = (ElectronicSpace(model) if determinants is None
             else SelectedElectronicSpace(model, determinants))
    return _electronic_actions(model, space, symmetry, determinants)


def _prepare(model, lam, shift, symmetry, determinants):
    shift = _validate_inputs(model, lam, shift, determinants)
    _check_symmetry(model, symmetry)
    space = (ElectronicSpace(model) if determinants is None
             else SelectedElectronicSpace(model, determinants))
    actions = _electronic_actions(model, space, symmetry, determinants)
    electronic, coupling = space.dense(limit=ELECTRONIC_LIMIT)
    # The matrix-unit reduction requires [H,R]=0. Check it in the actual
    # retained sector, including direct dataclass instances and wrong-model
    # symmetry snapshots, before relying on that reduction.
    for action, photon in zip(actions, symmetry.photon_actions):
        if not np.allclose(electronic @ action, action @ electronic, atol=1e-11, rtol=1e-11):
            raise ValueError("electronic Hamiltonian does not preserve the supplied symmetry")
        rotated = action @ coupling @ action.T
        expected = np.einsum("yx,yij->xij", photon, coupling)
        if not np.allclose(rotated, expected, atol=1e-11, rtol=1e-11):
            raise ValueError("electron-photon coupling does not preserve the supplied symmetry")
    with np.errstate(over="ignore", invalid="ignore"):
        eta = shift[:, None, :] - np.einsum("Axp,Ip->AIx", lam, space.density)
    if not np.all(np.isfinite(eta)):
        raise ValueError("conditional displacements are nonfinite")
    return space, eta, electronic, coupling, actions


def _relative_blocks(model, symmetry, eta, electronic, coupling, actions):
    nframe, ndet = eta.shape[:2]
    shape = (nframe, ndet, nframe, ndet)
    for g, (action, photon) in enumerate(zip(actions, symmetry.photon_actions)):
        hmat = np.empty(shape, dtype=np.complex128)
        smat = np.empty(shape, dtype=np.complex128)
        ht, gt = electronic @ action, coupling @ action
        beta = eta @ photon.T
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for a in range(nframe):
                for b in range(nframe):
                    exponent = np.zeros((ndet, ndet))
                    scalar = np.full((ndet, ndet), model.ecore)
                    pair_h = ht.copy()
                    for x, frequency in enumerate(model.omega):
                        bra, ket = eta[a, :, x, None], beta[b, None, :, x]
                        exponent += (bra - ket)**2
                        scalar += frequency * bra * ket
                        pair_h += gt[x] * (bra + ket)
                    overlap = np.exp(-.5 * exponent)
                    hmat[a, :, b, :] = overlap * (pair_h + action * scalar)
                    smat[a, :, b, :] = overlap * action
        if not np.all(np.isfinite(hmat)) or not np.all(np.isfinite(smat)):
            raise ValueError("relative group kernels contain nonfinite values")
        yield g, hmat, smat


def lf_group_relative_blocks(model, lam, shift=None, *, symmetry, determinants=None):
    """Yield (g,H_g,S_g), each complex128 (F,D,F,D), one operation at a time.

    H_g[AI,BJ]=<AI|H R(g)|BJ> and S_g[AI,BJ]=<AI|R(g)|BJ> use
    analytic infinite-oscillator coherent overlaps. D is the full FCI size
    (alpha address before beta address), or the supplied selected-list size.
    The group images need not admit new diagonal LF parameters. These
    relative matrices are generally not individually Hermitian.
    """
    _, eta, electronic, coupling, actions = _prepare(
        model, lam, shift, symmetry, determinants)
    yield from _relative_blocks(model, symmetry, eta, electronic, coupling, actions)


def _hermitian_mean(matrix, name):
    half = matrix.ndim // 2
    adjoint = matrix.transpose((*range(half, matrix.ndim), *range(half))).conj()
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains nonfinite values")
    scale = max(1., float(np.max(np.abs(matrix))))
    if np.max(np.abs(matrix - adjoint)) > 1e-12 * scale:
        raise ValueError(f"{name} must be Hermitian within relative tolerance 1e-12")
    return .5 * matrix + .5 * adjoint


def lf_group_frame_blocks(model, lam, shift=None, *, symmetry, irrep, row=0,
                          determinants=None):
    r"""Return (H,S) between |A,n,I> = P^irrep[row,n] |A,I>.

    P_mn=d/|G| sum_g D_mn(g)* R(g). For a fixed row, all d columns
    are retained; the matrix-unit identity gives H_AnI,BkJ=<AI|H P_nk|BJ>
    and the corresponding S. Thus H/S do not depend on the chosen row.
    ``row`` selects partner wave functions when reconstructing coefficients.

    lam is float64 (F,nmode,norb), shift is float64 (F,nmode) or None.
    Return complex128 tensors of shape (F,d,nstra,nstrb,F,d,nstra,nstrb),
    or (F,d,D,F,d,D) for selected int64 determinant bit pairs. Both tensors
    are checked before conjugate-transpose averaging. Zero-rank projected
    sectors are returned as zero matrices for the solver to diagnose.

    Guards require D<=256 and F*d*D<=1024, before electronic enumeration
    or square-kernel allocation. No full matrix over all group images is
    formed. Inputs are never changed, and no coherent cutoff is used.
    """
    _validate_inputs(model, lam, shift, determinants)
    group = _check_symmetry(model, symmetry)
    if not isinstance(irrep, str) or irrep not in group.irrep_matrices:
        raise ValueError("irrep must label an irreducible representation of symmetry.group")
    matrices = group.irrep_matrices[irrep]
    dimension = matrices.shape[1]
    if (isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer))
            or not 0 <= row < dimension):
        raise ValueError("row must be an integer partner index in [0,irrep dimension)")
    ndet = (comb(model.norb, model.nelec[0]) * comb(model.norb, model.nelec[1])
            if determinants is None else len(determinants))
    if len(lam) * dimension * ndet > FRAME_SPACE_LIMIT:
        raise NotImplementedError(
            f"group-projected kernels require <= {FRAME_SPACE_LIMIT} basis states")
    space, eta, electronic, coupling, actions = _prepare(
        model, lam, shift, symmetry, determinants)
    shape = (len(lam), dimension, ndet, len(lam), dimension, ndet)
    hmat, smat = np.zeros(shape, dtype=np.complex128), np.zeros(shape, dtype=np.complex128)
    for g, hg, sg in _relative_blocks(model, symmetry, eta, electronic, coupling, actions):
        weight = dimension / group.order * matrices[g].conj()
        hmat += np.einsum("nk,aibj->anibkj", weight, hg)
        smat += np.einsum("nk,aibj->anibkj", weight, sg)
    structured = (len(lam), dimension, *space.shape, len(lam), dimension, *space.shape)
    return (_hermitian_mean(hmat.reshape(structured), "hmat"),
            _hermitian_mean(smat.reshape(structured), "smat"))
