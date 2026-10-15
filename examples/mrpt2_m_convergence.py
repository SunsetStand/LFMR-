"""Raise the MRPT2 phonon cutoff M at one archived point until it converges.

Fixed archived frames, translation-projected sector and reference cut 1e-10
are taken from reference_data. For every M the candidate blocks are built
once and then solved for each requested external overlap cut. A cut is
phonon-converged once two successive increments |E(M)-E(M-1)| fall below
--tol (the historical criterion). Rows are rewritten after every solve, so a
requeued or resubmitted job resumes at the first incomplete M.
"""

import argparse
import json
import os
import resource
import time
from math import comb
from pathlib import Path

import numpy as np

from mr_lf import LFMR
from mr_lf.mr.kernels import build_projected_mr_candidate_blocks
from mr_lf.mr.pt2 import mrpt2_from_blocks
from mr_lf.mr.reduced import (
    build_reduced_projected_blocks,
    external_gram_maximum,
    lean_mrpt2,
    reference_spin_parity,
)
from mr_lf.mr.space import orthogonalize_mr_spaces_blockwise
from mr_lf.phonon import lf_total_phonon_configurations

from .common import frame_point, model_for, read_rows, write_rows

FIELDS = [
    "reference_energy",
    "second_order_energy",
    "corrected_energy",
    "reference_gap",
    "minimum_denominator",
    "reference_residual",
    "response_residual",
    "first_order_norm_squared",
    "reference_rank",
    "external_rank",
    "reference_metric_error",
    "external_metric_error",
    "cross_metric_error",
]


def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def external_overlap_maximum(blocks, groups, reference_cut):
    """Largest external Gram eigenvalue after reference subtraction."""
    geometry = orthogonalize_mr_spaces_blockwise(
        blocks["s_rr"],
        blocks["s_rt"],
        blocks["s_tt"],
        groups["reference_groups"],
        groups["external_groups"],
        reference_cut=reference_cut,
        external_cut=1.0,
    )
    return float(np.max(geometry["external_eigenvalues"], initial=0.0))


def build(algorithm, m, mr, occupations, defer_h=False):
    """Candidate blocks plus solver keywords for the chosen candidate list.

    'reduced' removes exact translation-orbit duplicates of invariant seeds
    and keeps the spin-exchange sector of the MR ground root. The external
    cut is rescaled so that its threshold uses the largest external Gram
    eigenvalue of the complete list (both parity sectors), as in 'full'.
    """
    if algorithm == "full":
        blocks = build_projected_mr_candidate_blocks(
            m.tmat,
            m.U,
            m.g,
            m.omega,
            mr.lam,
            mr.shift,
            m.nelec,
            occupations,
            character=mr.character,
        )
        return blocks, {}, {"cut_scale": 1.0}
    parity = reference_spin_parity(mr)
    lean = algorithm == "lean"
    lam_comp = 0.0
    if parity is not None:
        comp, cinfo = build_reduced_projected_blocks(
            m,
            mr.lam,
            mr.shift,
            occupations,
            character=mr.character,
            parity=-parity,
            overlap_only=True,
            grouped_overlap=lean,
        )
        if "spin_exchange" in cinfo["kinds"]:
            if lean:
                lam_comp = external_gram_maximum(comp, cinfo, mr.overlap_cut)
            else:
                lam_comp = external_overlap_maximum(comp, cinfo, mr.overlap_cut)
        del comp
    if lean:
        blocks, info = build_reduced_projected_blocks(
            m,
            mr.lam,
            mr.shift,
            occupations,
            character=mr.character,
            parity=parity,
            grouped_overlap=True,
            defer_h=defer_h,
        )
        meta = {
            "spin_parity": parity,
            "seed_reductions": "/".join(info["kinds"]),
            "complement_gram_max": lam_comp,
            "cut_scale": 1.0,
        }
        return blocks, info, meta
    blocks, info = build_reduced_projected_blocks(
        m, mr.lam, mr.shift, occupations, character=mr.character, parity=parity
    )
    groups = {k: info[k] for k in ("reference_groups", "external_groups")}
    lam_red = external_overlap_maximum(blocks, groups, mr.overlap_cut)
    meta = {
        "spin_parity": parity,
        "seed_reductions": "/".join(info["kinds"]),
        "external_gram_max": max(lam_red, lam_comp),
        "cut_scale": max(1.0, lam_red, lam_comp) / max(1.0, lam_red),
    }
    return blocks, groups, meta


def solve(blocks, groups, mr, m, nconfig, cut, mode, meta=None, last=False):
    """One numerically checked solve; same acceptance as LFMRPT2.kernel."""
    if "s_tt_groups" in blocks:
        if mode != "float64":
            raise ValueError("the lean solver supports float64 only")
        result = lean_mrpt2(
            blocks,
            groups,
            reference_cut=mr.overlap_cut,
            external_cut=cut,
            external_scale=meta["complement_gram_max"],
            release_h=last,
        )
    else:
        result = mrpt2_from_blocks(
            blocks,
            len(mr.lam),
            m.L,
            m.nelec,
            nconfig,
            reference_cut=mr.overlap_cut,
            external_cut=cut,
            precision_mode=mode,
            projected=True,
            **{k: groups[k] for k in ("reference_groups", "external_groups") if k in groups},
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
    return result


def history(rows, cut):
    """Successful energies by M for one cut, and the converged M if any."""
    ok = {
        int(r["max_total"]): float(r["corrected_energy"])
        for r in rows
        if float(r["external_cut"]) == cut and r["status"] == "ok"
    }
    return ok


def converged_at(energies, tol, consecutive):
    """Smallest M whose last `consecutive` successive increments are < tol."""
    for M in sorted(energies):
        steps = [M - k for k in range(consecutive + 1)]
        if all(s in energies for s in steps) and all(
            abs(energies[steps[k]] - energies[steps[k + 1]]) < tol for k in range(consecutive)
        ):
            return M
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["single", "multi"], required=True)
    p.add_argument("--omega", type=float, default=0.5)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--external-cuts", type=float, nargs="+", default=[1e-8, 1e-10])
    p.add_argument(
        "--primary-cut", type=float, default=1e-8, help="cut whose convergence drives the loop"
    )
    p.add_argument(
        "--precision-mode", choices=["float64", "reconditioned_longdouble"], default="float64"
    )
    p.add_argument(
        "--fallback-longdouble",
        action="store_true",
        help="retry a failed float64 primary-cut solve with reconditioned_longdouble",
    )
    p.add_argument("--algorithm", choices=["full", "reduced", "lean"], default="full")
    p.add_argument(
        "--defer-h",
        action="store_true",
        help="lean only: contract H_TT one Gram-group pair at a time instead of storing it",
    )
    p.add_argument(
        "--retry-failed", action="store_true", help="recompute cuts whose rows at an M all failed"
    )
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--consecutive", type=int, default=2)
    p.add_argument("--min-total", type=int, default=1)
    p.add_argument("--max-total", type=int, default=40)
    p.add_argument(
        "--memory-gib",
        type=float,
        default=None,
        help="stop before an M whose projected peak memory exceeds this (default: Slurm limit)",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.primary_cut not in a.external_cuts:
        p.error("--primary-cut must be one of --external-cuts")
    if a.memory_gib is None and os.environ.get("SLURM_MEM_PER_NODE"):
        a.memory_gib = 0.92 * float(os.environ["SLURM_MEM_PER_NODE"]) / 1024
    m = model_for(a.case, a.omega, a.alpha)
    lam, z, q = frame_point(a.case, a.omega, a.alpha)
    mr = LFMR(m, lam, z, character=q, overlap_cut=1e-10)
    mr.kernel()
    if not mr.converged:
        raise SystemExit("MR reference did not converge")
    rows = read_rows(a.output) if a.output.exists() else []
    base = {
        "case": a.case,
        "omega": a.omega,
        "alpha": a.alpha,
        "character": q,
        "mr_energy": mr.e_tot,
        "reference_cut": mr.overlap_cut,
    }
    status = {"stop_reason": None}
    previous_nt = previous_peak = None
    for M in range(a.min_total, a.max_total + 1):
        done = {
            float(r["external_cut"])
            for r in rows
            if int(r["max_total"]) == M and (r["status"] == "ok" or not a.retry_failed)
        }
        conv = {c: converged_at(history(rows, c), a.tol, a.consecutive) for c in a.external_cuts}
        if conv[a.primary_cut] is not None:
            last = {c: [r for r in rows if float(r["external_cut"]) == c] for c in a.external_cuts}
            secondary_done = all(
                conv[c] is not None or (last[c] and last[c][-1]["status"] != "ok")
                for c in a.external_cuts
                if c != a.primary_cut
            )
            if secondary_done:
                status["stop_reason"] = "converged"
                break
        active = [
            c for c in a.external_cuts if c not in done and (c == a.primary_cut or conv[c] is None)
        ]
        if not active:
            continue
        occupations = lf_total_phonon_configurations(m.L, M)
        nt = len(lam) * comb(m.L, m.nelec[0]) * comb(m.L, m.nelec[1]) * len(occupations)
        if a.algorithm in ("reduced", "lean"):
            # Reduced lists are about 0.42 (multi) of the complete list.
            nt = int(0.45 * nt) if m.nelec[0] == m.nelec[1] and sum(m.nelec) > 1 else nt
        if a.memory_gib and previous_nt and previous_peak:
            projected = previous_peak * (nt / previous_nt) ** 2
            if projected > a.memory_gib:
                status["stop_reason"] = f"memory: M={M} projected {projected:.1f} GiB"
                break
        t0 = time.time()
        blocks, groups, meta = build(a.algorithm, m, mr, occupations, a.defer_h)
        nt = blocks["h_rt"].shape[1]
        build_seconds = time.time() - t0
        for icut, cut in enumerate(active):
            modes = [a.precision_mode]
            if (
                a.fallback_longdouble
                and cut == a.primary_cut
                and a.precision_mode == "float64"
                and a.algorithm != "lean"
            ):
                modes.append("reconditioned_longdouble")
            for mode in modes:
                row = dict(
                    base,
                    max_total=M,
                    nconfig=len(occupations),
                    n_candidates=nt,
                    external_cut=cut,
                    precision_mode=mode,
                    algorithm=a.algorithm,
                    **meta,
                    build_seconds=round(build_seconds, 2),
                )
                t1 = time.time()
                try:
                    res = solve(
                        blocks,
                        groups,
                        mr,
                        m,
                        len(occupations),
                        cut * meta["cut_scale"],
                        mode,
                        meta,
                        last=icut == len(active) - 1,
                    )
                    row.update(status="ok", **{k: res[k] for k in FIELDS})
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    row.update(status="failed", failure=str(exc)[:300])
                row.update(solve_seconds=round(time.time() - t1, 2), peak_gib=round(peak_gib(), 3))
                rows.append(row)
                write_rows(a.output, rows)
                energy = row.get("corrected_energy", float("nan"))
                print(
                    f"M={M} nt={nt} cut={cut:g} {mode} {row['status']} E={energy:.12f} "
                    f"build={build_seconds:.1f}s solve={row['solve_seconds']}s "
                    f"peak={row['peak_gib']}GiB",
                    flush=True,
                )
                if row["status"] == "ok":
                    break
        del blocks
        previous_nt, previous_peak = nt, peak_gib()
    else:
        status["stop_reason"] = f"max_total {a.max_total} reached"
    conv = {c: converged_at(history(rows, c), a.tol, a.consecutive) for c in a.external_cuts}
    status.update(
        base,
        tol=a.tol,
        consecutive=a.consecutive,
        converged_M={f"{c:g}": conv[c] for c in a.external_cuts},
        max_M={
            f"{c:g}": max(
                (int(r["max_total"]) for r in rows if float(r["external_cut"]) == c), default=None
            )
            for c in a.external_cuts
        },
    )
    a.output.with_suffix(".json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
