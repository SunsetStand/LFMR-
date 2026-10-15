"""Accurate retained-metric contractions for nearly dependent NOCI columns.

LAPACK still receives complex128 reduced matrices. Extended accumulations
avoid losing the small physical result when large coefficient coordinates
cancel in the original nonorthogonal basis. They do not alter the raw
double-precision kernels, retained span, or overlap cutoff.
"""

import numpy as np


def metric_product(left, matrix, right):
    """Return left.conj().T @ matrix @ right using extended accumulation."""
    left = np.asarray(left, dtype=np.clongdouble)
    matrix = np.asarray(matrix, dtype=np.clongdouble)
    right = np.asarray(right, dtype=np.clongdouble)
    return np.asarray(left.conj().T @ (matrix @ right), dtype=np.complex128)


def metric_norm(coeff, smat):
    """Return complex c†Sc, retaining its imaginary diagnostic component."""
    coeff = np.asarray(coeff, dtype=np.clongdouble)
    smat = np.asarray(smat, dtype=np.clongdouble)
    return complex(np.vdot(coeff, smat @ coeff))


def metric_residual(basis, hmat, smat, coeff, energy):
    """Return ||Q†(Hc-ESc)|| using the actual supplied coefficient vector."""
    basis = np.asarray(basis, dtype=np.clongdouble)
    hmat = np.asarray(hmat, dtype=np.clongdouble)
    smat = np.asarray(smat, dtype=np.clongdouble)
    coeff = np.asarray(coeff, dtype=np.clongdouble)
    residual = basis.conj().T @ (hmat @ coeff - np.longdouble(energy) * (smat @ coeff))
    return float(np.sqrt(np.sum(np.abs(residual)**2)))
