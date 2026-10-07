"""Recompute archived fixed-frame MR/PT2 points with their recorded settings."""

import argparse
from pathlib import Path

from pyscf import lib

from mr_lf import LFMR, LFMRPT2

from .common import REFERENCE, ROOT, frame_point, model_for, read_rows, write_metadata, write_rows

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["single", "multi"], default="single")
    p.add_argument("--omega", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--full", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced_reference.csv")
    a = p.parse_args()
    lib.num_threads(1)
    source = read_rows(
        REFERENCE / ("single_mrpt2.csv" if a.case == "single" else "multi_mrpt2_m5.csv")
    )
    selected = (
        source
        if a.full
        else [
            r
            for r in source
            if float(r["omega"]) == a.omega and abs(float(r["alpha"]) - a.alpha) < 1e-10
        ]
    )
    if not selected:
        p.error("requested point is absent from the reference table")
    rows = []
    for r in selected:
        omega, alpha = float(r["omega"]), float(r["alpha"])
        m = model_for(a.case, omega, alpha)
        lam, z, q = frame_point(a.case, omega, alpha)
        mr = LFMR(m, lam, z, character=q, overlap_cut=float(r["reference_cut"]))
        mr.kernel()
        mode = r.get("precision_mode", "float64")
        pt = LFMRPT2(
            mr, int(r["max_total"]), external_cut=float(r["external_cut"]), precision_mode=mode
        )
        pt.kernel()
        row = {
            "omega": omega,
            "alpha": alpha,
            "character": q,
            "max_total": int(r["max_total"]),
            "energy": pt.e_tot,
            "archived_energy": float(r["corrected_energy"]),
            "difference": pt.e_tot - float(r["corrected_energy"]),
            "reference_residual": pt.result["reference_residual"],
            "response_residual": pt.result["response_residual"],
            "reference_rank": pt.result["reference_rank"],
            "external_rank": pt.result["external_rank"],
            "precision_mode": mode,
        }
        rows.append(row)
        write_rows(a.output, rows)
        print(f"omega={omega:g} alpha={alpha:g} difference={row['difference']:.3e}", flush=True)
    write_metadata(
        a.output,
        {
            "case": a.case,
            "source": "Distributed fixed-frame inputs and archived cutoffs.",
            "cutoff_convergence_asserted": False,
        },
    )
    if any(abs(r["difference"]) > 1e-8 for r in rows):
        raise SystemExit("Energy differs from archive by more than 1e-8")
