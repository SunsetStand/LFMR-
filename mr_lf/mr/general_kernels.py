"""Dense, small-space vacuum kernels in a common orthonormal electronic basis.

Each frame spans a common full or selected fixed-electron-number space:
|A,I> = |I> D(z_A - lambda_A n(I)) |0>.  The frames are supplied, not
optimized. LFMR exposes these kernels; molecular LFMRPT2 is not implemented.
"""

from math import comb

import numpy as np

from .._electronic import ElectronicSpace
from .._selected_electronic import SelectedElectronicSpace
from ..model import ElectronPhononModel

ELECTRONIC_LIMIT = 256
FRAME_SPACE_LIMIT = 1024


def _validate_inputs(model, lam, shift, determinants):
    """Reject malformed/oversized inputs before enumerating determinants."""
    if not isinstance(model, ElectronPhononModel):
        raise TypeError("model must be an ElectronPhononModel or MolecularEP")
    if not isinstance(lam, np.ndarray) or lam.dtype != np.float64:
        raise TypeError("lam must be a float64 NumPy array")
    if (lam.ndim != 3 or lam.shape[0] < 1
            or lam.shape[1:] != (model.nmode, model.norb)):
        raise ValueError("lam must have shape (F,nmode,norb), F >= 1")
    if not np.all(np.isfinite(lam)):
        raise ValueError("lam must contain only finite values")
    if shift is not None:
        if not isinstance(shift, np.ndarray) or shift.dtype != np.float64:
            raise TypeError("shift must be a float64 NumPy array")
        if shift.shape != (len(lam), model.nmode):
            raise ValueError("shift must have shape (F,nmode)")
        if not np.all(np.isfinite(shift)):
            raise ValueError("shift must contain only finite values")
    if model.norb >= 64:
        raise NotImplementedError("the determinant representation requires norb < 64")
    if determinants is None:
        ndet = comb(model.norb, model.nelec[0]) * comb(model.norb, model.nelec[1])
    else:
        if not isinstance(determinants, np.ndarray) or determinants.dtype != np.int64:
            raise TypeError("determinants must be an int64 NumPy array")
        if (determinants.ndim != 2 or determinants.shape[1] != 2
                or len(determinants) == 0):
            raise ValueError("determinants must have nonempty shape (D,2)")
        ndet = len(determinants)
    if ndet > ELECTRONIC_LIMIT:
        raise NotImplementedError(
            f"vacuum frame kernels require <= {ELECTRONIC_LIMIT} electronic determinants")
    if len(lam) * ndet > FRAME_SPACE_LIMIT:
        raise NotImplementedError(
            f"vacuum frame kernels require <= {FRAME_SPACE_LIMIT} frame basis states")
    if shift is None:
        shift = np.zeros((len(lam), model.nmode), dtype=np.float64)
    return shift


def _hermitian_mean(kernel, name):
    """Check ordered frame pairs before averaging accepted float64 roundoff."""
    if not np.all(np.isfinite(kernel)):
        raise ValueError(f"{name} contains nonfinite matrix elements")
    half = kernel.ndim // 2
    transpose = kernel.transpose((*range(half, kernel.ndim), *range(half)))
    scale = max(1., float(np.max(np.abs(kernel))))
    with np.errstate(over='ignore', invalid='ignore'):
        error = float(np.max(np.abs(kernel - transpose)))
    if not np.isfinite(error) or error > 1e-12 * scale:
        raise ValueError(f"{name} must be Hermitian within relative tolerance 1e-12")
    return .5 * kernel + .5 * transpose


def lf_vacuum_frame_blocks(model, lam, shift=None, *, determinants=None):
    r"""Return vacuum S/H kernels between supplied general-integral LF frames.

    Parameters
    ----------
    model : ElectronPhononModel or MolecularEP
        Real integrals in one common orthonormal electronic basis. Full
        chemists' ERIs use a_p^dagger a_r^dagger a_s a_q with a 1/2 factor.
        Both electronic backends enforce only pair exchange and real Hermitian
        reversal; no additional ERI symmetry or packing is assumed.
    lam : float64 ndarray, shape (F,nmode,norb)
        Finite density-diagonal LF parameters, F >= 1.
    shift : float64 ndarray, shape (F,nmode), optional
        Finite coherent shifts; None means zero. All inputs may be
        read-only or noncontiguous and are never modified.
    determinants : int64 ndarray, shape (D,2), optional
        Ordered alpha/beta bit-string pairs in the model's common orthonormal
        orbital basis, not FCI addresses. Pairs must be unique and have the
        model's electron counts. All frames use this same list. None retains
        the full FCI path; a supplied list uses direct selected contractions
        without enumerating or allocating the complete FCI space.

    Returns
    -------
    dict
        smat and hmat, new exactly symmetric float64 tensors of shape
        (F,nstra,nstrb,F,nstra,nstrb), ordered bra frame, alpha determinant
        address, beta address, ket frame, alpha address, beta address.
        Determinant addresses follow cistring order, not bit-string values.
        With determinants supplied, shape is instead (F,D,F,D), ordered
        bra frame, bra selected address, ket frame, ket selected address.
        Pass this dictionary to lf_noci_lowest; coefficients have shape
        (F,nstra,nstrb) or (F,D), respectively. Flatten only at that solver
        boundary. S need not be invertible when frames are repeated.

    Notes
    -----
    For eta_AI = z_A - lambda_A n(I) and f = exp(-|eta_BJ-eta_AI|^2/2),
    S_AI,BJ = delta_IJ f and
    H_AI,BJ = f [h_el_IJ + sum_x G_x,IJ (eta_AI,x+eta_BJ,x)
                 + delta_IJ (ecore + sum_x omega_x eta_AI,x eta_BJ,x)].
    h_el excludes ecore; the scalar offset is included once as ecore*S.
    The coupling is uncentered and there is no phonon zero-point energy.
    Vacuum displacement matrix elements are analytic in the infinite
    oscillator space: no oscillator cutoff is introduced by this kernel.

    This dense baseline supports at most 256 full or selected determinants and
    1024 total frame basis states; guards precede determinant/link and
    square-matrix allocations. Zero modes and zero electrons are supported.
    An explicit electronic list defines a variational truncation, with no
    correction for excluded configurations. No independent orbital sets,
    automatic CI selection, phonon excitations, symmetry
    projection, reference optimization or PT2 are implemented here.
    Raw Hermitian errors above 1e-12*max(1,max(abs(kernel))) and nonfinite
    results raise rather than being hidden by averaging.
    """
    shift = _validate_inputs(model, lam, shift, determinants)
    space = (ElectronicSpace(model) if determinants is None
             else SelectedElectronicSpace(model, determinants))
    electronic, coupling = space.dense(limit=ELECTRONIC_LIMIT)
    nframe, ndet = len(lam), space.size
    shape = (nframe, *space.shape, nframe, *space.shape)
    smat, hmat = np.zeros(shape), np.zeros(shape)
    diagonal = np.diag_indices(ndet)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        eta = shift[:, None, :] - np.einsum('Axp,Ip->AIx', lam, space.density)
        if not np.all(np.isfinite(eta)):
            raise ValueError("conditional displacements are nonfinite")
        # Evaluate both ordered pairs independently, so Hermitian checks
        # can detect errors before the accepted arithmetic is averaged.
        for a in range(nframe):
            for b in range(nframe):
                exponent = np.zeros((ndet, ndet))
                pair_h = electronic.copy()
                offset = np.full(ndet, model.ecore)
                for x, frequency in enumerate(model.omega):
                    bra, ket = eta[a, :, x], eta[b, :, x]
                    exponent += (ket[None, :] - bra[:, None])**2
                    pair_h += coupling[x] * (bra[:, None] + ket[None, :])
                    offset += frequency * bra * ket
                franck_condon = np.exp(-.5 * exponent)
                pair_h[diagonal] += offset
                pair_h *= franck_condon
                pair_s = np.zeros_like(pair_h)
                pair_s[diagonal] = franck_condon[diagonal]
                axes = (slice(None),) * len(space.shape)
                index = (a, *axes, b, *axes)
                hmat[index] = pair_h.reshape((*space.shape, *space.shape))
                smat[index] = pair_s.reshape((*space.shape, *space.shape))
    return {'smat': _hermitian_mean(smat, 'smat'),
            'hmat': _hermitian_mean(hmat, 'hmat')}
