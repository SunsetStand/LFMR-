"""Real orthogonal orbitals and occupied/virtual canonicalization."""

import numpy as np
from scipy import linalg


def initial_orbitals(model, mo_coeff0=None):
    if mo_coeff0 is None:
        coeff = np.linalg.eigh(model.tmat)[1]
        return np.stack((coeff, coeff))
    coeff = np.asarray(mo_coeff0)
    if coeff.shape != (2, model.L, model.L) or coeff.dtype != np.float64:
        raise ValueError("mo_coeff0 must be float64 (2,L,L), [spin,site,MO]")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("orbitals must be finite")
    for c in coeff:
        if not np.allclose(c.T @ c, np.eye(model.L), atol=1e-10, rtol=0):
            raise ValueError("orbitals must have orthonormal columns")
    return coeff.copy()


def rotate(coeff0, nelec, parameters):
    coeff = []
    offset = 0
    for c0, nocc in zip(coeff0, nelec):
        size = nocc * (len(c0) - nocc)
        kappa = parameters[offset : offset + size].reshape(len(c0) - nocc, nocc)
        generator = np.zeros_like(c0)
        generator[nocc:, :nocc] = kappa
        generator[:nocc, nocc:] = -kappa.T
        rotation, _ = linalg.polar(linalg.expm(generator))
        coeff.append(c0 @ rotation)
        offset += size
    return np.stack(coeff)


def densities(coeff, nelec):
    return np.stack([c[:, :n] @ c[:, :n].T for c, n in zip(coeff, nelec)])


def canonicalize(coeff, fock, nelec):
    orbitals, energies = [], []
    for c, f, n in zip(coeff, fock, nelec):
        fmo = c.T @ f @ c
        eo, ro = np.linalg.eigh(fmo[:n, :n])
        ev, rv = np.linalg.eigh(fmo[n:, n:])
        orbitals.append(c @ linalg.block_diag(ro, rv))
        energies.append(np.r_[eo, ev])
    return np.stack(orbitals), np.stack(energies)


def validate_options(gtol, max_cycle):
    if isinstance(gtol, (bool, np.bool_)) or not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be finite and positive")
    if (
        isinstance(max_cycle, (bool, np.bool_))
        or not isinstance(max_cycle, (int, np.integer))
        or max_cycle < 1
    ):
        raise ValueError("max_cycle must be a positive integer")
