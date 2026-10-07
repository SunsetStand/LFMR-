"""Compare fixed LF-HF, one-frame MR, orbit MR and projected MR densities."""

import argparse
from pathlib import Path

import numpy as np
from pyscf import lib

from mr_lf import LFMR
from mr_lf.symmetry import lf_translation_orbit

from .common import (
    REFERENCE,
    ROOT,
    density_fields,
    frame_point,
    model_for,
    read_rows,
    write_metadata,
    write_rows,
)


def run(case, omega, alpha, output):
    lib.num_threads(1)
    m = model_for(case, omega, alpha)
    lam, z, q = frame_point(case, omega, alpha)
    table = read_rows(REFERENCE / f"{case}_electron.csv")
    r = next(
        r for r in table if float(r["omega"]) == omega and abs(float(r["alpha"]) - alpha) < 1e-10
    )
    rows = [
        dict(
            method="LF-HF",
            energy=r["lf_hf_energy"],
            omega=omega,
            alpha=alpha,
            **{k: v for k, v in r.items() if k.startswith("hf_")},
        )
    ]
    orbit = [lf_translation_orbit(l, s) for l, s in zip(lam, z)]
    expanded = np.concatenate([x[0] for x in orbit])
    expanded_z = np.concatenate([x[1] for x in orbit])
    for label, frames, shifts, character in [
        ("one-frame MR", lam[:1], z[:1], None),
        ("translation-orbit MR", expanded, expanded_z, None),
        ("projected MR", lam, z, q),
    ]:
        mr = LFMR(m, frames, shifts, character=character)
        mr.kernel()
        if not mr.converged:
            raise RuntimeError("MR numerical checks failed")
        rows.append(
            dict(
                method=label,
                energy=mr.e_tot,
                omega=omega,
                alpha=alpha,
                rank=mr.result["rank"],
                near_degenerate=mr.result["near_degenerate"],
                retained_residual=mr.result["residual_projected"],
                **density_fields("mr", mr.site_density()),
            )
        )
    write_rows(output, rows)
    write_metadata(
        output,
        {
            "case": case,
            "character": q,
            "frames": "Archived fixed-frame recipe; no frame reoptimization.",
            "ambiguity": "Densities of a near-degenerate selected root are state-dependent.",
        },
    )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["single", "multi"], default="single")
    p.add_argument("--omega", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=2.4)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/symmetry.csv")
    a = p.parse_args()
    run(a.case, a.omega, a.alpha, a.output)
