"""Translation orbits, determinant addresses and fermionic parity."""

import numpy as np

from . import basis


def lf_translation_orbit(
    lam: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the full cyclic orbit of one local LF reference frame.

    Translation ``R`` moves both the physical phonon mode and
    electron-site indices forward on an L-site ring:
    ``lam_orbit[R, x, p] = lam[(x-R) % L, (p-R) % L]`` and
    ``shift_orbit[R, x] = shift[(x-R) % L]``.  The orbit keeps
    all L positions, including duplicates of a symmetric seed.

    Parameters
    ----------
    lam
        Finite float64 LF parameters for one frame with shape
        ``(L, L)`` and index order ``[phonon mode, electron site]``.
    shift
        Finite float64 coherent shift for the same frame with
        shape ``(L,)`` and index order ``[phonon mode]``.

    Returns
    -------
    lam_orbit
        Translated LF parameters with shape ``(L, L, L)`` and
        index order ``[translation R, phonon mode x, electron site p]``.
    shift_orbit
        Translated coherent shifts with shape ``(L, L)`` and
        index order ``[translation R, phonon mode x]``.

    Raises
    ------
    ValueError
        If dimensions are empty or incompatible, or values are
        nonfinite.
    TypeError
        If either array does not have float64 dtype.

    Notes
    -----
    A full LF-HF orbital would also translate as
    ``coeff_R[p] = coeff[(p-R) % L]``.  The NOCI frame basis used
    here already includes every electron site, so electronic CI
    coefficients are optimized later and are not part of this output.
    """
    if lam.ndim != 2 or lam.shape[0] != lam.shape[1] or lam.shape[0] == 0:
        raise ValueError("lam must be a nonempty square 2D array")
    L = lam.shape[0]
    if shift.ndim != 1 or shift.shape != (L,):
        raise ValueError("shift must have shape (L,)")
    if lam.dtype != np.float64 or shift.dtype != np.float64:
        raise TypeError("lam and shift must have float64 dtype")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(shift)):
        raise ValueError("lam and shift must contain only finite values")
    lam_orbit = np.zeros((L, L, L), dtype=np.float64)
    shift_orbit = np.zeros((L, L), dtype=np.float64)
    for R in range(L):
        lam_orbit[R, :, :] = np.roll(lam, shift=R, axis=(0, 1))
        shift_orbit[R, :] = np.roll(shift, shift=R, axis=0)
    return lam_orbit, shift_orbit


def _translation_electron_labels(nsite, nelec):
    """Determinant addresses, fermion signs and electron translation orbits."""
    bases = [basis.make_electron_basis(nsite, n)[0] for n in nelec]
    maps, signs = [], []
    mask = (1 << nsite) - 1
    for strings, number in zip(bases, nelec):
        occupation = (strings[:, None] >> np.arange(nsite)) & 1
        rotated = np.stack(
            [((strings << r) & mask) | (strings >> (nsite - r)) for r in range(nsite)]
        )
        maps.append(np.searchsorted(strings, rotated))
        wrapped = np.stack(
            [
                occupation[:, nsite - r :].sum(axis=1)
                if r
                else np.zeros(len(strings), dtype=np.int64)
                for r in range(nsite)
            ]
        )
        signs.append(np.where((wrapped * (number - wrapped)) % 2, -1.0, 1.0))
    nb = len(bases[1])
    address = (maps[0][:, :, None] * nb + maps[1][:, None, :]).reshape(nsite, -1)
    sign = (signs[0][:, :, None] * signs[1][:, None, :]).reshape(nsite, -1)
    groups = np.min(address, axis=0).astype(np.int64)
    return address, sign, groups
