"""Explicit alpha scans; only requested ED/PT2 calculations are performed."""

import argparse
from pathlib import Path

import numpy as np
from pyscf import lib

from mr_lf import CSHF, CSMP2, LFHF, LFMP2, LFMR, LFMRPT2, ExactED
from mr_lf.scf.lf import lf_hf_full_initial_guesses

from .common import (
    ROOT,
    antiferromagnetic_start,
    charge_ordered_start,
    density_fields,
    model_for,
    write_metadata,
    write_rows,
)


def best_cs(model, previous=None):
    starts = (
        [None]
        if sum(model.nelec) == 1
        else [antiferromagnetic_start(), charge_ordered_start(), None]
    )
    if previous is not None:
        starts.append(previous.mo_coeff)
    candidates = []
    for start in starts:
        hf = CSHF(model)
        hf.kernel(start)
        if hf.converged:
            candidates.append(hf)
    if not candidates:
        raise RuntimeError("no converged CS reference from the specified starts")
    return min(candidates, key=lambda c: c.e_tot)


def best_lf(model, nrandom=8, previous=None):
    candidates = []
    if sum(model.nelec) == 1:
        starts = [
            (None, lam)
            for lam in lf_hf_full_initial_guesses(
                model.L, model.g, model.omega, nrandom=nrandom, seed=0
            )
        ]
    else:
        starts = [
            (c, np.eye(model.L) * model.g / model.omega)
            for c in (antiferromagnetic_start(), charge_ordered_start())
        ]
        if previous is not None:
            starts.append((previous.mo_coeff, previous.lam))
    for coeff, lam in starts:
        hf = LFHF(model)
        hf.kernel(coeff, lam)
        candidates.append(hf)
    good = [c for c in candidates if c.converged]
    if not good:
        raise RuntimeError("no converged LF reference from the specified starts")
    if sum(model.nelec) == 1 and not candidates[0].converged:
        raise RuntimeError("the CS-start LF branch needed for the MR recipe failed")
    return min(good, key=lambda c: c.e_tot), candidates


def main(case):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="use the complete published alpha grid")
    parser.add_argument("--alphas", type=float, nargs="+")
    parser.add_argument("--omegas", type=float, nargs="+")
    parser.add_argument("--with-ed", action="store_true")
    parser.add_argument("--with-mrpt2", action="store_true")
    parser.add_argument(
        "--nmax", type=int, default=4, help="fixed per-mode ED cutoff, not a convergence claim"
    )
    parser.add_argument("--ed-convention", choices=["paper", "centered"], default="paper")
    parser.add_argument("--max-total", type=int, default=2, help="inclusive all-mode MRPT2 cutoff")
    parser.add_argument("--lf-max-total", type=int, default=10 if case == "single" else 12)
    parser.add_argument("--lf-max-modes", type=int, default=4)
    parser.add_argument("--nrandom", type=int, default=8)
    parser.add_argument(
        "--precision-mode", choices=["float64", "reconditioned_longdouble"], default="float64"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / f"{case}_scan.csv")
    args = parser.parse_args()
    lib.num_threads(1)
    grid = np.arange(16) * 0.2 if case == "single" else np.arange(11) * 0.4
    alphas = args.alphas if args.alphas is not None else (grid if args.full else [0.0, 0.4])
    omegas = (
        args.omegas
        if args.omegas is not None
        else ([0.5] if case == "single" or not args.full else [0.5, 5.0])
    )
    rows = []
    for omega in omegas:
        previous_cs = previous_lf = None
        for alpha in alphas:
            m = model_for(case, omega, float(alpha))
            row = {
                "omega": m.omega,
                "alpha": m.alpha,
                "g": m.g,
                "L": m.L,
                "neleca": m.nelec[0],
                "nelecb": m.nelec[1],
                "U": m.U,
                "t": -1,
                "energy_convention": "paper_uncentered_no_zero_point",
                "status": "ok",
            }
            try:
                cs = best_cs(m, previous_cs)
                lf, branches = best_lf(m, args.nrandom, previous_lf)
                previous_cs, previous_lf = cs, lf
                row.update(
                    cs_hf_energy=cs.e_tot,
                    lf_hf_energy=lf.e_tot,
                    cs_orbital_residual=cs.result["orbital_grad_norm"],
                    cs_converged=cs.converged,
                    lf_converged=lf.converged,
                    **density_fields("cs", cs.site_density()),
                    **density_fields("hf", lf.site_density()),
                )
                for name, calc in [
                    ("cs_mp2", CSMP2(cs)),
                    (
                        "lf_mp2",
                        LFMP2(lf, max_total=args.lf_max_total, max_excited_modes=args.lf_max_modes),
                    ),
                ]:
                    try:
                        calc.kernel()
                        row[name + "_energy"] = calc.e_tot
                        row[name + "_minimum_denominator"] = calc.result.get(
                            "minimum_denominator", ""
                        )
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        row[name + "_failure"] = str(exc)
                row.update(
                    lf_mp2_max_total=args.lf_max_total,
                    lf_mp2_max_excited_modes=m.L if case == "single" else args.lf_max_modes,
                )
                if case == "single":
                    lam = np.stack((lf.lam, branches[0].lam))
                    z = np.stack((lf.shift, branches[0].shift))
                    characters = [1]
                else:
                    lam = np.stack((lf.lam, np.zeros((m.L, m.L))))
                    z = np.stack((lf.shift, np.full(m.L, -m.g * sum(m.nelec) / (m.L * m.omega))))
                    characters = [-1, 1]
                refs = []
                for q in characters:
                    mr = LFMR(m, lam, z, character=q)
                    mr.kernel()
                    if mr.converged:
                        refs.append(mr)
                if not refs:
                    raise RuntimeError("no converged MR sector")
                mr = min(refs, key=lambda c: c.e_tot)
                row.update(
                    mr_energy=mr.e_tot,
                    character=mr.character,
                    mr_rank=mr.result["rank"],
                    reference_cut=mr.overlap_cut,
                    mr_residual=mr.result["residual_projected"],
                    mr_density_near_degenerate=mr.result["near_degenerate"],
                    **density_fields("mr", mr.site_density()),
                )
                if args.with_mrpt2:
                    pt = LFMRPT2(
                        mr, args.max_total, external_cut=1e-8, precision_mode=args.precision_mode
                    )
                    pt.kernel()
                    row.update(
                        mrpt2_energy=pt.e_tot,
                        mrpt2_max_total=args.max_total,
                        mrpt2_phonon_converged=False,
                        mrpt2_converged=False,
                        external_cut=pt.external_cut,
                        precision_mode=pt.precision_mode,
                        **{
                            k: pt.result[k]
                            for k in [
                                "reference_residual",
                                "response_residual",
                                "reference_rank",
                                "external_rank",
                                "minimum_denominator",
                                "numerically_valid",
                            ]
                        },
                    )
                if args.with_ed:
                    ed = ExactED(
                        m,
                        args.nmax,
                        max_cycle=500,
                        max_space=40,
                        coupling_convention=args.ed_convention,
                        tol_residual=1e-9,
                        lindep=1e-22,
                    )
                    ed.kernel()
                    shift = (
                        -(m.g**2) * sum(m.nelec) ** 2 / (m.L * m.omega)
                        if args.ed_convention == "centered"
                        else 0.0
                    )
                    row.update(
                        ed_energy=ed.e_tot + shift,
                        ed_solved_energy=ed.e_tot,
                        ed_paper_shift=shift,
                        ed_convention=args.ed_convention,
                        ed_Nmax=args.nmax,
                        ed_dimension=ed.result["dimension"],
                        ed_residual=ed.result["residual"],
                        ed_solver_converged=ed.converged,
                        ed_cutoff_converged=False,
                    )
                    if not ed.converged:
                        raise RuntimeError("ED did not meet the independent residual threshold")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                row.update(status="failed", failure=str(exc))
            rows.append(row)
            write_rows(args.output, rows)
            print(f"{case} omega={omega:g} alpha={alpha:g} {row['status']}", flush=True)
    write_metadata(
        args.output,
        {
            "case": case,
            "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "seed": 0,
            "convergence": "Single cutoff; no phonon or overlap-cut convergence assertion.",
        },
    )
    if any(r["status"] != "ok" or any(k.endswith("_failure") for k in r) for r in rows):
        raise SystemExit("One or more methods failed; failures are recorded in the output CSV.")
