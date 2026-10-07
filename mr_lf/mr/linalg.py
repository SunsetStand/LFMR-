"""Bounded-temporary matrix helpers; no LF model or spectral selection.

Rows remain bra indices and columns ket indices. Callers validate finite
float64/longdouble matrices before using these private helpers. Scans use
at most 128 rows; in-place averaging uses pairs of 128-square tiles.
"""

import numpy as np

BLOCK_ROWS = 128


def max_abs(matrix):
    """Return max(abs(A)), or 0 for empty 2D A, without a full-size copy."""
    maximum = 0.0
    for start in range(0, matrix.shape[0], BLOCK_ROWS):
        maximum = float(
            np.maximum(maximum, np.max(np.abs(matrix[start : start + BLOCK_ROWS]), initial=0.0))
        )
    return maximum


def symmetry_error(matrix):
    """Return max(abs(A-A.T)) for a finite square matrix using row tiles."""
    maximum = 0.0
    for start in range(0, len(matrix), BLOCK_ROWS):
        difference = matrix[start : start + BLOCK_ROWS] - matrix[:, start : start + BLOCK_ROWS].T
        maximum = float(np.maximum(maximum, np.max(np.abs(difference), initial=0.0)))
    return maximum


def identity_error(matrix):
    """Return max(abs(A-I)) for a finite square matrix without a dense I."""
    maximum = 0.0
    for start in range(0, len(matrix), BLOCK_ROWS):
        difference = matrix[start : start + BLOCK_ROWS].copy()
        count = len(difference)
        difference[np.arange(count), start + np.arange(count)] -= 1.0
        maximum = float(np.maximum(maximum, max_abs(difference)))
    return maximum


def symmetric_copy(matrix, name):
    """Check the existing 1e-12 symmetry rule and return an independent mean.

    Input is validated finite float64/longdouble (n,n), optionally empty,
    strided or read-only. Return the same dtype/shape as .5*A+.5*A.T,
    preserving that expression's rounding, including subnormal entries.
    The sole full-size allocation is the result. Never modify input.
    """
    if symmetry_error(matrix) > 1e-12 * max(1.0, max_abs(matrix)):
        raise ValueError(f"{name} must be symmetric within relative tolerance 1e-12")
    result = np.empty_like(matrix)
    for start in range(0, len(matrix), BLOCK_ROWS):
        result[start : start + BLOCK_ROWS] = (
            0.5 * matrix[start : start + BLOCK_ROWS] + 0.5 * matrix[:, start : start + BLOCK_ROWS].T
        )
    return result


def symmetrize_in_place(matrix):
    """Replace an owned writable square A by .5*A+.5*A.T in the same buffer.

    Finite float64/longdouble input has shape (n,n); n may be zero. Read
    both transpose partners before either write. Return None. This helper
    does no tolerance check and is never applied to caller-owned inputs.
    """
    for i in range(0, len(matrix), BLOCK_ROWS):
        rows = slice(i, i + BLOCK_ROWS)
        for j in range(i, len(matrix), BLOCK_ROWS):
            columns = slice(j, j + BLOCK_ROWS)
            average = 0.5 * matrix[rows, columns] + 0.5 * matrix[columns, rows].T
            matrix[rows, columns] = average
            if i != j:
                matrix[columns, rows] = average.T
