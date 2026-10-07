"""Recompute a fixed archived point while varying phonon and overlap cutoffs."""

import argparse
from pathlib import Path

import numpy as np
from pyscf import lib

from mr_lf import LFMR, LFMRPT2

from .common import ROOT, frame_point, model_for, write_metadata, write_rows

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["single", "multi"], default="single")
    p.add_argument("--omega", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=2.4)
    p.add_argument("--max-total", type=int, default=3)
    p.add_argument("--external-cut", type=float, default=1e-8)
    p.add_argument("--overlap-sweep", action="store_true")
    p.add_argument(
        "--precision-mode", choices=["float64", "reconditioned_longdouble"], default="float64"
    )
    p.add_argument("--output", type=Path, default=ROOT / "outputs/mrpt2_cutoff.csv")
    a = p.parse_args()
    lib.num_threads(1)
    if a.max_total < 1:
        p.error("--max-total must be at least 1")
    m = model_for(a.case, a.omega, a.alpha)
    lam, z, q = frame_point(a.case, a.omega, a.alpha)
    configurations = [(cut, 1e-10, a.external_cut) for cut in range(1, a.max_total + 1)]
    if a.overlap_sweep:
        configurations += [(a.max_total, r, a.external_cut) for r in (1e-8, 1e-12)]
        configurations += [
            (a.max_total, 1e-10, e) for e in (1e-8, 1e-10, 1e-12) if e != a.external_cut
        ]
    rows = []
    for cut, rc, ec in configurations:
        row = {
            "omega": a.omega,
            "alpha": a.alpha,
            "character": q,
            "max_total": cut,
            "reference_cut": rc,
            "external_cut": ec,
            "precision_mode": a.precision_mode,
        }
        try:
            mr = LFMR(m, lam, z, character=q, overlap_cut=rc)
            mr.kernel()
            pt = LFMRPT2(mr, cut, external_cut=ec, precision_mode=a.precision_mode)
            pt.kernel()
            row.update(
                status="ok",
                energy=pt.e_tot,
                **{
                    k: pt.result[k]
                    for k in [
                        "reference_energy",
                        "second_order_energy",
                        "reference_rank",
                        "external_rank",
                        "reference_residual",
                        "response_residual",
                        "minimum_denominator",
                        "first_order_norm_squared",
                        "numerically_valid",
                    ]
                },
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row.update(status="failed", failure=str(exc))
        rows.append(row)
        write_rows(a.output, rows)
        print(f"M={cut} Rcut={rc:g} Qcut={ec:g} {row['status']}", flush=True)
    write_metadata(
        a.output,
        {
            "case": a.case,
            "cutoff_convergence_established": False,
            "note": "A finite sequence alone does not certify convergence.",
        },
    )
    if any(r["status"] != "ok" for r in rows):
        raise SystemExit("Numerical failures recorded")
