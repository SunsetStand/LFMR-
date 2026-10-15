"""Whole-reference/whole-external block-Hamiltonian second-order response."""

import numpy as np

from .._driver import Calculation, require_reference
from ..phonon import lf_total_phonon_configurations
from ..symmetry import _translation_electron_labels
from . import linalg, precision
from .kernels import build_mr_candidate_blocks, build_projected_mr_candidate_blocks
from .noci import LFMR
from .space import (
    _symmetric_copy,
    orthogonalize_mr_spaces_blockwise,
    transform_mr_hamiltonian_blocks,
)


def _validate_mrpt2_gaps(
    reference_eigenvalues: np.ndarray,
    denominators: np.ndarray,
    reference_gap_tol: float,
    denominator_tol: float,
) -> None:
    """Reject nonisolated ground roots and unsafe external denominators.

    Internal guard for ascending reference eigenvalues and external
    energy differences. Arrays must be finite float64 vectors, with at
    least one reference value. Scalar tolerances are validated by the
    public solver. Error messages include the offending spectrum.

    Relative scales use energy differences, so a common additive energy
    constant does not change the criteria. Empty external spectra and
    one-dimensional references impose no gap check in that respective
    space. No shift, clipping, or pseudoinverse is applied.
    """
    for name, values in (
        ("reference_eigenvalues", reference_eigenvalues),
        ("denominators", denominators),
    ):
        if not isinstance(values, np.ndarray) or values.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite vector")
    if reference_eigenvalues.size == 0 or np.any(
        reference_eigenvalues[1:] < reference_eigenvalues[:-1]
    ):
        raise ValueError("reference_eigenvalues must be nonempty and ascending")

    reference_differences = reference_eigenvalues - reference_eigenvalues[0]
    if not np.all(np.isfinite(reference_differences)):
        raise ValueError("reference energy differences overflowed float64")
    if reference_eigenvalues.size > 1:
        threshold = reference_gap_tol * max(1.0, float(np.max(reference_differences)))
        gap = float(reference_differences[1])
        if gap <= threshold:
            spectrum = np.array2string(reference_eigenvalues, precision=17)
            raise ValueError(
                "reference ground state is degenerate or near-degenerate: "
                f"gap={gap}, threshold={threshold}, reference spectrum={spectrum}"
            )
    if denominators.size:
        threshold = denominator_tol * max(1.0, float(np.max(np.abs(denominators))))
        minimum = float(np.min(denominators))
        if minimum <= threshold:
            spectrum = np.array2string(denominators, precision=17)
            raise ValueError(
                "nonpositive or near-zero external denominator: "
                f"minimum={minimum}, threshold={threshold}, denominators={spectrum}"
            )


def solve_mrpt2_blocks(
    h_reference: np.ndarray,
    h_coupling: np.ndarray,
    h_external: np.ndarray,
    *,
    reference_gap_tol: float = 1e-10,
    denominator_tol: float = 1e-10,
) -> dict[str, object]:
    """Compute the ground-root block-Hamiltonian PT2 response and diagnostics.

    Parameters
    ----------
    h_reference
        Finite float64 (r,r), r > 0; the Hamiltonian in the retained
        orthonormal reference space. The lowest root alone is selected.
    h_coupling
        Finite float64 (r,s); reference bra and orthonormal external ket.
    h_external
        Finite float64 (s,s); the complete chosen external Hamiltonian
        block K. s may be zero. Obtain all three inputs from
        transform_mr_hamiltonian_blocks using the same overlap geometry.
    reference_gap_tol, denominator_tol
        Finite positive real relative cutoffs. Reject a ground reference
        gap <= reference_gap_tol*max(1,max(e_ref-e_ref[0])). Reject any
        Delta <= denominator_tol*max(1,max(abs(Delta))), where
        Delta = eigvalsh(K)-E0. A one-root reference has no root-gap check;
        an empty external block has no denominator check. The criteria
        use energy differences and are invariant to a common constant
        energy shift, up to floating-point precision.

    Returns
    -------
    dict
        Exactly these fields (all arrays float64):

        * reference_energy: Python float E0.
        * reference_coeff: (r,), normalized lowest eigenvector u0.
        * reference_eigenvalues: (r,), full ascending reference spectrum.
        * reference_gap: Python float e_ref[1]-E0; inf when r == 1.
        * coupling_vector: (s,), v = h_coupling.T@u0.
        * external_eigenvalues: (s,), ascending spectrum of K.
        * denominators: (s,), external_eigenvalues-E0.
        * minimum_denominator: Python float; inf when s == 0.
        * response_matrix: (s,s), A = K-E0*I.
        * first_order_coeff: (s,), solution of A@a = -v.
        * second_order_energy: Python float v.T@a.
        * corrected_energy: Python float E0+second_order_energy.
        * reference_residual: Python float norm(h_reference@u0-E0*u0).
        * response_residual: Python float norm(A@a+v).
        * first_order_norm_squared: Python float a.T@a.

        Empty external spaces yield empty arrays and zero second-order
        energy, response residual, and first-order norm squared.
        Eigenvector signs are arbitrary; energies and norms are invariant.
        Coefficient indices refer to the retained orthonormal bases. The
        physical states are Psi0=Rbar@u0 and Psi1=Z@a; they are not stored.

    Raises
    ------
    TypeError
        For non-array inputs, array dtypes other than float64, or
        non-real scalar tolerance types.
    ValueError
        For invalid shapes, nonfinite values, invalid tolerances,
        asymmetry, a nonisolated reference ground root, or a nonpositive
        or near-zero external denominator. Spectrum failures include
        the gap/denominator values in their error messages.
    numpy.linalg.LinAlgError
        If an eigensolve or the unregularized linear solve fails.

    Notes
    -----
    This is the proposed whole-reference/whole-external block partition,
    H0=diag(h_reference,K), V containing the two off-diagonal blocks.
    E1=0. It is not asserted to be Cui LF-MP2 or a Fock partition. A
    positive denominator spectrum implies E2 <= 0, but E0+E2 has no
    guaranteed variational upper bound. These diagnostics do not certify
    phonon, overlap-cut, or perturbation-order convergence.

    Input arrays are not modified. Square blocks are locally symmetrized
    only within 1e-12*max(1,max(abs(H))). There is no level shift, cutoff
    of small denominators, pseudoinverse, frame optimization, or symmetry
    projection here. Even a zero source does not exempt an unsafe spectrum
    from the ground-root checks. Projection is handled before this solver boundary.

    Call _validate_mrpt2_gaps(reference_eigenvalues, denominators,
    reference_gap_tol, denominator_tol) after constructing the spectra
    and before solving the response. Solve the linear equation directly;
    do not construct the matrix inverse.
    """
    for name, array in (
        ("h_reference", h_reference),
        ("h_coupling", h_coupling),
        ("h_external", h_external),
    ):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a NumPy array with float64 dtype")
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    r, s = h_coupling.shape
    if r == 0 or h_reference.shape != (r, r) or h_external.shape != (s, s):
        raise ValueError("expected shapes (r,r), (r,s), (s,s) with r > 0")
    tolerances = []
    for name, value in (
        ("reference_gap_tol", reference_gap_tol),
        ("denominator_tol", denominator_tol),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real numeric scalar")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be finite, real, and positive") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite, real, and positive")
        tolerances.append(value)
    reference_gap_tol, denominator_tol = tolerances
    h_reference = _symmetric_copy(h_reference, "h_reference")
    h_external = _symmetric_copy(h_external, "h_external")
    reference_energy, reference_eigenvectors = np.linalg.eigh(h_reference)
    external_eigenvalues = np.linalg.eigvalsh(h_external)
    denominators = external_eigenvalues - reference_energy[0]
    _validate_mrpt2_gaps(
        reference_eigenvalues=reference_energy,
        denominators=denominators,
        reference_gap_tol=reference_gap_tol,
        denominator_tol=denominator_tol,
    )
    coupled_vector = h_coupling.T @ reference_eigenvectors[:, 0]
    A = h_external - np.eye(h_external.shape[0]) * reference_energy[0]
    response_vector = np.linalg.solve(A, -coupled_vector)
    correction_energy = coupled_vector.T @ response_vector
    return {
        "reference_energy": float(reference_energy[0]),
        "reference_coeff": reference_eigenvectors[:, 0],
        "reference_eigenvalues": reference_energy,
        "reference_gap": float(reference_energy[1] - reference_energy[0])
        if r > 1
        else float("inf"),
        "coupling_vector": coupled_vector,
        "external_eigenvalues": external_eigenvalues,
        "denominators": denominators,
        "minimum_denominator": float(np.min(denominators)) if s > 0 else float("inf"),
        "response_matrix": A,
        "first_order_coeff": response_vector,
        "second_order_energy": float(correction_energy),
        "corrected_energy": float(reference_energy[0] + correction_energy),
        "reference_residual": float(
            np.linalg.norm(
                h_reference @ reference_eigenvectors[:, 0]
                - reference_energy[0] * reference_eigenvectors[:, 0]
            )
        ),
        "response_residual": float(np.linalg.norm(A @ response_vector + coupled_vector)),
        "first_order_norm_squared": float(response_vector.T @ response_vector),
    }


def mrpt2_from_blocks(
    blocks,
    nseed,
    nsite,
    nelec,
    nconfig,
    *,
    reference_cut=1e-10,
    external_cut=1e-10,
    return_hamiltonian_blocks=False,
    precision_mode="float64",
    projected=True,
    reference_groups=None,
    external_groups=None,
):
    """Connect six blocks to retained-space geometry and the PT2 solver.

    projected=True groups determinants by translation orbit; False groups
    by determinant address. Both partitions use the same retained-space
    geometry, response equation and optional precision reconditioning.
    Explicit int64 reference_groups (nr,) and external_groups (nt,) replace
    that default labelling, for candidate lists other than the complete
    seed/determinant/configuration product (nconfig is then not used).

    blocks is a dictionary of finite float64 S/H arrays: rr (nr,nr),
    rt (nr,nt), tt (nt,nt). nseed is a positive integer, 0 < nsite < 64,
    nelec is an explicit (neleca,nelecb) tuple in [0,nsite], and nconfig
    is a nonnegative integer. nr=nseed*nstra*nstrb; nt=nr*nconfig.
    Grouping uses exact electronic translation orbits; columns still have
    the documented seed/determinant/configuration order. Return the solver
    fields plus geometry spectra/ranks, orthonormality diagnostics and raw
    seed-basis coefficients for Psi0 and Psi1. No physical optimization or
    convergence declaration occurs here. Gap failures propagate unchanged.

    return_hamiltonian_blocks must be bool. When True, also return a
    'hamiltonian_blocks' dict with float64 'h_reference' (r,r),
    'h_coupling' (r,s), and 'h_external' (s,s). These are the checked,
    roundoff-symmetrized blocks actually passed to solve_mrpt2_blocks;
    geometry is evaluated only once. The default return fields are
    unchanged for precision_mode='float64'. No input arrays are modified.

    precision_mode must be 'float64' (default legacy arithmetic) or
    'reconditioned_longdouble'. The latter reconditions the already kept
    P/Q coordinates, without repeating spectral selection. Gram/H products
    accumulate in longdouble; raw kernels, stored coefficients and checked
    solver blocks remain float64. Requires extra longdouble precision and
    exactly disjoint electronic Gram groups; H may couple all groups.
    Preserve the canonical overlap spectra and retained ranks. Raw state
    coefficients and optional H blocks use the repaired coordinates.
    In the extended mode, the private geometry's dense external_metric and
    duplicate x_external are released after spectral selection; only the
    needed transforms/spectra remain. The standalone geometry API still
    returns its complete original fields. No public result field is lost.

    Only the extended mode adds 'precision_diagnostics': mode, accumulator
    epsilon/significand bits, rank_preserved, before/after physical metric
    errors, metric tolerances and Hamiltonian asymmetry limits. The physical
    errors use the joint raw Gram, rather than the Schur metric alone.
    Reject reference/external metric errors >= 1e-6, cross errors >= 1e-7,
    and reference/response residuals >= 1e-10. Existing roundoff, reference
    gap and denominator checks remain unchanged. No automatic fallback,
    clipping, level shift or convergence declaration is made. Cholesky
    failure propagates as LinAlgError; unsupported precision as RuntimeError.
    These checks concern arithmetic on the supplied float64 blocks, not
    raw-kernel accuracy, phonon truncation or perturbation-order convergence.
    """
    if not isinstance(return_hamiltonian_blocks, (bool, np.bool_)):
        raise TypeError("return_hamiltonian_blocks must be bool")
    if not isinstance(precision_mode, str):
        raise TypeError("precision_mode must be a string")
    if precision_mode not in ("float64", "reconditioned_longdouble"):
        raise ValueError("precision_mode must be float64 or reconditioned_longdouble")
    for name, value, minimum in (("nseed", nseed, 1), ("nsite", nsite, 1), ("nconfig", nconfig, 0)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if nsite >= 64:
        raise ValueError("nsite must be less than 64")
    if (
        not isinstance(nelec, tuple)
        or len(nelec) != 2
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or not 0 <= n <= nsite
            for n in nelec
        )
    ):
        raise ValueError("nelec must be (neleca,nelecb), each between 0 and nsite")
    if not isinstance(blocks, dict):
        raise TypeError("blocks must be a dictionary")
    if any(name not in blocks for name in ("s_rr", "s_rt", "s_tt", "h_rr", "h_rt", "h_tt")):
        raise ValueError("blocks must contain all six overlap/Hamiltonian blocks")
    if not isinstance(projected, (bool, np.bool_)):
        raise TypeError("projected must be bool")
    if (reference_groups is None) != (external_groups is None):
        raise ValueError("supply both reference_groups and external_groups, or neither")
    if reference_groups is None:
        if projected:
            _, _, groups = _translation_electron_labels(nsite, nelec)
        else:
            import math

            groups = np.arange(
                math.comb(nsite, nelec[0]) * math.comb(nsite, nelec[1]), dtype=np.int64
            )
        reference_groups = np.tile(groups, nseed)
        external_groups = np.repeat(reference_groups, nconfig)
    geometry = orthogonalize_mr_spaces_blockwise(
        blocks["s_rr"],
        blocks["s_rt"],
        blocks["s_tt"],
        reference_groups,
        external_groups,
        reference_cut=reference_cut,
        external_cut=external_cut,
    )
    coordinates = geometry
    precision_diagnostics = None
    if precision_mode == "reconditioned_longdouble":
        # These two buffers are not used by raw-Gram contractions or the
        # extended result. Keep their standalone geometry return contract.
        del geometry["external_metric"], geometry["x_external"]
        before = precision.contract_retained_blocks(
            blocks, geometry, reference_groups, external_groups, kind="s"
        )
        coordinates = precision.recondition_retained_spaces(
            blocks, geometry, reference_groups, external_groups
        )
        after = precision.contract_retained_blocks(
            blocks, coordinates, reference_groups, external_groups, kind="s"
        )
        metric_errors = precision._metric_errors(after)
        tolerances = {
            "reference_metric_error": 1e-6,
            "external_metric_error": 1e-6,
            "cross_metric_error": 1e-7,
        }
        for name, tolerance in tolerances.items():
            if not np.isfinite(metric_errors[name]) or metric_errors[name] >= tolerance:
                raise ValueError(
                    f"{name} exceeds unchanged numerical tolerance: {metric_errors[name]}"
                )
        precision_diagnostics = {
            "mode": precision_mode,
            "accumulator_epsilon": float(np.finfo(np.longdouble).eps),
            "accumulator_significand_bits": int(np.finfo(np.longdouble).nmant) + 1,
            "rank_preserved": bool(
                coordinates["reference_rank"] == geometry["reference_rank"]
                and coordinates["external_rank"] == geometry["external_rank"]
            ),
            "before": precision._metric_errors(before),
            "after": metric_errors.copy(),
            "metric_tolerances": tolerances,
            "hamiltonian_asymmetry_limits": {},
        }
        if not precision_diagnostics["rank_preserved"]:
            raise ValueError("precision repair must preserve the canonical retained ranks")
        hblocks = precision.contract_retained_blocks(
            blocks, coordinates, reference_groups, external_groups
        )
    else:
        hblocks = transform_mr_hamiltonian_blocks(
            blocks["h_rr"],
            blocks["h_rt"],
            blocks["h_tt"],
            geometry["x_ref"],
            geometry["external_ref_coeff"],
            geometry["external_candidate_coeff"],
        )
    # The congruence is exactly Hermitian. Near-dependent coordinate
    # columns can amplify GEMM roundoff above the strict input tolerance.
    # Bound that roundoff, record it, and average only within the bound;
    # do not alter spectra or denominators. The absolute scale cap prevents
    # very ill-conditioned transforms from bypassing the diagnostic.
    cr, ct = coordinates["external_ref_coeff"], coordinates["external_candidate_coeff"]
    coeff_scale = max(
        1.0, float(np.max(np.sum(np.abs(cr), axis=0) + np.sum(np.abs(ct), axis=0), initial=0.0))
    )
    hscale = max(1.0, *(linalg.max_abs(blocks[name]) for name in ("h_rr", "h_rt", "h_tt")))
    roundoff_bound = 10 * np.finfo(np.float64).eps * (len(cr) + len(ct)) * hscale * coeff_scale**2
    asymmetries = {}
    for name in ("h_reference", "h_external"):
        matrix = hblocks[name]
        asymmetry = linalg.symmetry_error(matrix)
        scale = max(1.0, linalg.max_abs(matrix))
        if asymmetry > min(roundoff_bound, 1e-7 * scale):
            raise ValueError(f"{name} congruence asymmetry exceeds roundoff bound: {asymmetry}")
        asymmetries[name + "_asymmetry"] = asymmetry
        linalg.symmetrize_in_place(matrix)
        if precision_diagnostics is not None:
            precision_diagnostics["hamiltonian_asymmetry_limits"][name] = min(
                roundoff_bound, 1e-7 * scale
            )
    if precision_diagnostics is not None:
        hblocks = {name: np.asarray(matrix, dtype=np.float64) for name, matrix in hblocks.items()}
    result = solve_mrpt2_blocks(**hblocks)
    x = coordinates["x_ref"]
    if precision_diagnostics is None:
        y = geometry["x_external"]
        metric_errors = {
            "reference_metric_error": float(
                np.max(np.abs(x.T @ blocks["s_rr"] @ x - np.eye(x.shape[1])), initial=0.0)
            ),
            "external_metric_error": float(
                np.max(
                    np.abs(y.T @ geometry["external_metric"] @ y - np.eye(y.shape[1])), initial=0.0
                )
            ),
            "cross_metric_error": float(
                np.max(np.abs(x.T @ (blocks["s_rr"] @ cr + blocks["s_rt"] @ ct)), initial=0.0)
            ),
        }
    else:
        for name in ("reference_residual", "response_residual"):
            if not np.isfinite(result[name]) or result[name] >= 1e-10:
                raise ValueError(f"{name} exceeds unchanged numerical tolerance: {result[name]}")
        result["precision_diagnostics"] = precision_diagnostics
    result.update(asymmetries)
    result.update(metric_errors)
    result.update(
        reference_rank=geometry["reference_rank"],
        external_rank=geometry["external_rank"],
        reference_overlap_eigenvalues=geometry["reference_eigenvalues"],
        external_overlap_eigenvalues=geometry["external_eigenvalues"],
        raw_reference_coeff=x @ result["reference_coeff"],
        raw_first_order_ref_coeff=cr @ result["first_order_coeff"],
        raw_first_order_candidate_coeff=ct @ result["first_order_coeff"],
    )
    if return_hamiltonian_blocks:
        result["hamiltonian_blocks"] = hblocks
    return result


def projected_mrpt2_from_blocks(*args, **kwargs):
    """Projected convenience entry; same fields as mrpt2_from_blocks."""
    return mrpt2_from_blocks(*args, projected=True, **kwargs)


class LFMRPT2(Calculation):
    """Block-Hamiltonian PT2 with all-mode inclusive total-phonon cutoff.

    A single call certifies numerical response checks, not cutoff convergence.
    Frame parameters, symmetry sector and reference cutoff come from mr.
    """

    def __init__(self, mr, max_total, *, external_cut=1e-10, precision_mode="float64"):
        if not isinstance(mr, LFMR):
            raise TypeError("LFMRPT2 requires an LFMR reference")
        super().__init__(mr.model)
        self.mr = mr
        if (
            isinstance(max_total, (bool, np.bool_))
            or not isinstance(max_total, (int, np.integer))
            or max_total < 0
        ):
            raise ValueError("max_total must be a nonnegative integer")
        self.max_total = int(max_total)
        self.external_cut = external_cut
        self.precision_mode = precision_mode

    def kernel(self):
        self._reset()
        require_reference(self.mr)
        m, mr = self.model, self.mr
        occupations = (
            lf_total_phonon_configurations(m.L, self.max_total)
            if self.max_total
            else np.empty((0, m.L), dtype=np.int64)
        )
        args = (m.tmat, m.U, m.g, m.omega, mr.lam, mr.shift, m.nelec, occupations)
        if mr.character is None:
            blocks = build_mr_candidate_blocks(*args)
        else:
            blocks = build_projected_mr_candidate_blocks(*args, character=mr.character)
        result = mrpt2_from_blocks(
            blocks,
            len(mr.lam),
            m.L,
            m.nelec,
            len(occupations),
            reference_cut=mr.overlap_cut,
            external_cut=self.external_cut,
            precision_mode=self.precision_mode,
            projected=mr.character is not None,
        )
        if abs(result["reference_energy"] - mr.e_tot) > 1e-7:
            raise ValueError("MRPT2 retained reference differs from supplied MR reference")
        valid = (
            result["reference_residual"] < 1e-10
            and result["response_residual"] < 1e-10
            and result["reference_metric_error"] < 1e-6
            and result["external_metric_error"] < 1e-6
            and result["cross_metric_error"] < 1e-7
        )
        if not valid:
            raise ValueError("MRPT2 numerical residual or metric check failed")
        result.update(
            max_total=self.max_total,
            max_excited_modes=m.L,
            occupations=occupations,
            character=mr.character,
            reference_cut=mr.overlap_cut,
            external_cut=self.external_cut,
            precision_mode=self.precision_mode,
            numerically_valid=True,
            cutoff_converged=False,
        )
        return self._finish(result, result["corrected_energy"], True, result["second_order_energy"])
