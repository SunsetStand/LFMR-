"""Greedily add full translation orbits along the archived LF-frame path."""

import argparse
from pathlib import Path

import numpy as np
from pyscf import lib

from mr_lf import LFMR
from mr_lf.symmetry import lf_translation_orbit

from .common import ROOT, frame_point, model_for, write_metadata, write_rows

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=2.4)
    p.add_argument(
        "--thetas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0]
    )
    p.add_argument("--output", type=Path, default=ROOT / "outputs/mr_frames.csv")
    a = p.parse_args()
    lib.num_threads(1)
    m = model_for("single", 0.5, a.alpha)
    seeds, z, _ = frame_point("single", 0.5, a.alpha)
    rows = []

    def measure(frames, shifts, label, theta=""):
        mr = LFMR(m, frames, shifts)
        mr.kernel()
        if not mr.converged:
            raise RuntimeError("MR residual or normalization check failed")
        rows.append(
            {
                "phase": label,
                "theta": theta,
                "n_frames": len(frames),
                "rank": mr.result["rank"],
                "energy": mr.e_tot,
                "residual": mr.result["residual_projected"],
            }
        )
        write_rows(a.output, rows)
        return mr

    measure(seeds[:1], z[:1], "best LF-HF frame")
    base, shifts = lf_translation_orbit(seeds[0], z[0])
    current = measure(base, shifts, "best orbit")
    cs, cs_z = lf_translation_orbit(seeds[1], z[1])
    base = np.concatenate((base, cs))
    shifts = np.concatenate((shifts, cs_z))
    current = measure(base, shifts, "plus CS-start orbit")
    pending = list(a.thetas)
    while pending:
        eligible = []
        for theta in pending:
            l = (1 - theta) * seeds[1] + theta * seeds[0]
            s = (1 - theta) * z[1] + theta * z[0]
            orbit, oz = lf_translation_orbit(l, s)
            trial = LFMR(m, np.concatenate((base, orbit)), np.concatenate((shifts, oz)))
            trial.kernel()
            if (
                trial.converged
                and trial.result["rank"] > current.result["rank"]
                and current.e_tot - trial.e_tot > 1e-8
            ):
                eligible.append((trial.e_tot, theta, orbit, oz))
        if not eligible:
            break
        _, theta, orbit, oz = min(eligible, key=lambda x: x[0])
        base = np.concatenate((base, orbit))
        shifts = np.concatenate((shifts, oz))
        current = measure(base, shifts, "greedy candidate orbit", theta)
        pending.remove(theta)
    write_metadata(
        a.output,
        {
            "alpha": a.alpha,
            "thetas": a.thetas,
            "path": "lambda(theta)=(1-theta)*CS_start+theta*best_LF_HF",
            "note": "Finite candidate-pool improvement; not convergence to exact ED.",
        },
    )
