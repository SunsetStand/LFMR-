"""Compute the new multielectron CS curves without recalculating other methods."""

import argparse
from pathlib import Path

from pyscf import lib

from mr_lf import CSMP2

from .common import (
    REFERENCE,
    ROOT,
    density_fields,
    model_for,
    read_rows,
    write_metadata,
    write_rows,
)
from .scan import best_cs


def run(output):
    lib.num_threads(1)
    rows = []
    previous = None
    last_omega = None
    for r in read_rows(REFERENCE / "multi_electron.csv"):
        omega, alpha = float(r["omega"]), float(r["alpha"])
        if omega != last_omega:
            previous = None
        m = model_for("multi", omega, alpha)
        hf = best_cs(m, previous)
        previous = hf
        last_omega = omega
        row = dict(
            omega=omega,
            alpha=alpha,
            cs_hf_energy=hf.e_tot,
            cs_converged=hf.converged,
            cs_orbital_residual=hf.result["orbital_grad_norm"],
            cs_optimizer_success=hf.result["optimizer_success"],
            **density_fields("cs", hf.site_density()),
        )
        try:
            mp = CSMP2(hf)
            mp.kernel()
            row.update(
                cs_mp2_energy=mp.e_tot,
                cs_mp2_correction=mp.e_corr,
                cs_phonon_single_correction=mp.result["phonon_single_correction"],
                cs_zero_double_correction=mp.result["zero_double_correction"],
                minimum_denominator=mp.result["minimum_denominator"],
                status="ok",
            )
        except (ValueError, RuntimeError) as exc:
            row.update(status="failed", failure=str(exc))
        rows.append(row)
        write_rows(output, rows)
        print(f"CS omega={omega:g} alpha={alpha:g} {row['status']}", flush=True)
    write_metadata(
        output,
        {
            "origin": "Computed using mr_lf.scf.cs and mr_lf.mp.cs",
            "starts": "AFM, CDW, hopping eigenvectors, previous-point orbitals",
            "gtol": 1e-8,
            "max_cycle": 1000,
            "stationarity_tolerance": 1e-7,
            "energy_convention": "paper_uncentered_no_zero_point",
        },
    )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/multi_cs.csv")
    args = p.parse_args()
    rows = run(args.output)
    if any(r["status"] != "ok" for r in rows):
        raise SystemExit("CS-MP2 failures recorded")
