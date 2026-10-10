"""H2 with a supplied odd harmonic mode: charge and spin symmetry breaking.

This is a molecular-electronic/artificial-mode benchmark, not an ab initio
vibrational parameter prediction or a cavity Pauli–Fierz Hamiltonian.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyscf import gto, lib, lo, scf
from scipy import linalg, optimize

from mr_lf import LFHF, LFMP2, ExactED, MolecularEP
from mr_lf._electronic import ElectronicSpace
from mr_lf.scf.general_lf import fixed_reference, occupations


def make_model(distance, coupling, omega=.5):
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {distance}", basis="sto-3g",
                unit="Angstrom", spin=0, verbose=0)
    s = mol.intor_symmetric("int1e_ovlp")
    x = lo.orth.lowdin(s)
    inverse_x = x.T @ s
    # Define g*(n_0-n_1)*(b+b†) in the fixed orthonormal basis and map to AO.
    g_ao = inverse_x.T @ np.diag([coupling, -coupling]) @ inverse_x
    return MolecularEP(mol, omega=[omega], coupling_ao=g_ao[None])


def parity_overlap(hf):
    """Overlap with electron-site exchange combined with b -> -b (H2 sector)."""
    space = ElectronicSpace(hf.model)
    c = space.reference_ci(hf.mo_coeff)
    eta = (hf.shift - space.density @ hf.lam.T).reshape(2, 2, 1)
    overlap = np.exp(-.5*np.sum((eta + eta[::-1, ::-1])**2, axis=-1))
    return float(np.sum(c * c[::-1, ::-1] * overlap))


def exact_parity(ed):
    coeff = ed.result["coeff"]
    phases = (-1.)**np.arange(ed.nmax + 1)
    return float(np.sum(coeff * coeff[::-1, ::-1] * phases))


def solve_references(model, *, gtol=1e-8):
    """Compare a symmetric stationary branch against explicit unconstrained starts."""
    base = np.linalg.eigh(model.h1e)[1]
    c0 = np.stack((base, base))
    g = model.coupling[0, 0, 0]
    omega = model.omega[0]

    def symmetric_energy(value):
        return fixed_reference(model, c0, np.array([[value, -value]]))[0]

    bound = max(1., 3*abs(g)/omega)
    symmetric = optimize.minimize_scalar(symmetric_energy, bounds=(-bound, bound),
                                         method="bounded", options={"xatol": 1e-12})
    symmetric_lam = np.array([[symmetric.x, -symmetric.x]])
    _, _, fock = fixed_reference(model, c0, symmetric_lam)
    symmetric_orbital_residual = float(np.linalg.norm(scf.uhf.get_grad(c0, occupations(model), fock)))
    symmetric_lambda_residual = 0.
    for index in np.ndindex(symmetric_lam.shape):
        plus, minus = symmetric_lam.copy(), symmetric_lam.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        gradient = (fixed_reference(model, c0, plus)[0]-fixed_reference(model, c0, minus)[0])/2e-5
        symmetric_lambda_residual = max(symmetric_lambda_residual, abs(gradient))
    if max(symmetric_orbital_residual, symmetric_lambda_residual) > 1e-6:
        raise RuntimeError("the constrained symmetric branch is not stationary in the full space")
    rotation = linalg.expm(np.array([[0., -.001], [.001, 0.]]))
    inverse_rotation = rotation.T
    seeds = [
        ("symmetric", c0, symmetric_lam),
        ("tiny_charge", np.stack((base @ rotation, base @ rotation)), symmetric_lam),
        ("tiny_spin", np.stack((base @ rotation, base @ inverse_rotation)), symmetric_lam),
        ("ionic_left", np.stack((np.eye(2), np.eye(2))), np.array([[g/omega, -g/omega]])),
        ("ionic_right", np.stack((np.eye(2)[:, ::-1], np.eye(2)[:, ::-1])),
         np.array([[g/omega, -g/omega]])),
        ("antiferromagnetic", np.stack((np.eye(2), np.eye(2)[:, ::-1])), symmetric_lam),
    ]
    trials, valid = [], []
    for name, c, lam in seeds:
        hf = LFHF(model, gtol=gtol, max_cycle=1000)
        hf.kernel(mo_coeff0=c, lam0=lam)
        trials.append({
            "seed": name, "energy": hf.e_tot, "converged": hf.converged,
            "parity_overlap": parity_overlap(hf), "spin_square": hf.result["spin_square"],
            "orbital_residual": hf.result["orbital_grad_norm"],
            "lambda_residual": hf.result["lam_grad_max"], "message": hf.result["message"],
        })
        if hf.converged:
            valid.append((name, hf))
    if not valid:
        raise RuntimeError("no LF-HF start passed physical stationarity checks")
    name, best = min(valid, key=lambda item: item[1].e_tot)
    q = np.eye(2)[::-1]
    partner_energy = fixed_reference(model, q @ best.mo_coeff, -best.lam[:, ::-1])[0]
    parity = parity_overlap(best)
    tiny_broken = any(t["seed"].startswith("tiny_") and t["converged"]
                      and t["energy"] < symmetric.fun - 1e-7
                      and 1-abs(t["parity_overlap"]) > 1e-6 for t in trials)
    diagnostics = {
        "symmetric_energy": float(symmetric.fun), "selected_seed": name,
        "symmetric_orbital_residual": symmetric_orbital_residual,
        "symmetric_lambda_residual": symmetric_lambda_residual,
        "energy_gain": float(symmetric.fun - best.e_tot), "parity_overlap": parity,
        "parity_defect": 1-abs(parity), "partner_energy_error": abs(partner_energy-best.e_tot),
        "broken": bool(symmetric.fun-best.e_tot > 1e-7 and 1-abs(parity) > 1e-6),
        "broken_from_tiny_seed": bool(tiny_broken), "trials": trials,
    }
    return best, diagnostics


def run_point(distance, coupling, omega, nmax_values, max_total_values):
    model = make_model(distance, coupling, omega)
    hf, diagnostic = solve_references(model)
    ed_history = []
    previous = None
    for nmax in nmax_values:
        ed = ExactED(model, nmax, max_space=100, max_cycle=600, lindep=1e-22)
        ed.kernel()
        delta = None if previous is None else abs(ed.e_tot-previous)
        ed_history.append({"nmax": nmax, "dimension": ed.result["dimension"],
                           "energy": ed.e_tot, "residual": ed.result["residual"],
                           "numerically_converged": ed.converged, "cutoff_delta": delta})
        if ed.converged and delta is not None and delta < 1e-9:
            break
        previous = ed.e_tot if ed.converged else None
    ed_cutoff = bool(ed.converged and ed_history[-1]["cutoff_delta"] is not None
                     and ed_history[-1]["cutoff_delta"] < 1e-9)
    pt_history = []
    for maximum in max_total_values:
        pt = LFMP2(hf, max_total=maximum)
        try:
            pt.kernel()
            pt_history.append({"max_total": maximum, "energy": pt.e_tot, "correction": pt.e_corr,
                               "minimum_denominator": pt.result["minimum_denominator"], "status": "ok"})
        except ValueError as error:
            pt_history.append({"max_total": maximum, "status": "failed", "message": str(error)})
            break
    pt_ok = all(p["status"] == "ok" for p in pt_history)
    pt_cutoff = bool(pt_ok and len(pt_history) > 1
                     and abs(pt_history[-1]["energy"]-pt_history[-2]["energy"]) < 1e-9)
    density = hf.site_density()
    row = {
        "distance_angstrom": distance, "coupling_hartree": coupling, "omega_hartree": omega,
        "hf_energy": hf.e_tot, "symmetric_energy": diagnostic["symmetric_energy"],
        "energy_gain": diagnostic["energy_gain"], "hf_converged": hf.converged,
        "orbital_residual": hf.result["orbital_grad_norm"], "lambda_residual": hf.result["lam_grad_max"],
        "charge_imbalance": float(np.ptp(density.sum(axis=0))),
        "spin_amplitude": float(np.max(abs(density[0]-density[1]))),
        "spin_square": hf.result["spin_square"], "phonon_mean": float(hf.result["phonon_mean"][0]),
        "parity_overlap": diagnostic["parity_overlap"], "broken": diagnostic["broken"],
        "broken_from_tiny_seed": diagnostic["broken_from_tiny_seed"],
        "partner_energy_error": diagnostic["partner_energy_error"],
        "symmetric_orbital_residual": diagnostic["symmetric_orbital_residual"],
        "symmetric_lambda_residual": diagnostic["symmetric_lambda_residual"],
        "ed_energy": ed.e_tot, "ed_residual": ed.result["residual"], "ed_nmax": ed.nmax,
        "ed_dimension": ed.result["dimension"], "ed_converged": ed.converged,
        "ed_cutoff_converged": ed_cutoff, "ed_parity": exact_parity(ed),
        "mp2_energy": pt_history[-1].get("energy"), "mp2_status": pt_history[-1]["status"],
        "mp2_max_total": pt_history[-1]["max_total"], "mp2_cutoff_converged": pt_cutoff,
    }
    detail = {"parameters": {"norb": 2, "nmode": 1, "nelec": [1, 1], "basis": "sto-3g",
                              "distance_angstrom": distance, "coupling_hartree": coupling,
                              "omega_hartree": omega,
                              "orthogonalization": model.orthogonalization, "ecore": model.ecore,
                              "coupling_convention": "uncentered", "zero_point_energy": False},
              **diagnostic, "ed_history": ed_history, "mp2_history": pt_history}
    return row, detail, hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", nargs="+", type=float, default=[.74, 1.5, 2.0])
    parser.add_argument("--couplings", nargs="+", type=float, default=[0., .2, .4, .6, .8])
    parser.add_argument("--omega", type=float, default=.5)
    parser.add_argument("--nmax", nargs="+", type=int, default=[8, 16, 24, 32, 48, 64])
    parser.add_argument("--max-total", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--output", type=Path, default=Path("outputs/molecular_symmetry"))
    args = parser.parse_args()
    for name, values in (("nmax", args.nmax), ("max-total", args.max_total)):
        if any(v < 0 for v in values) or values != sorted(set(values)):
            parser.error(f"{name} must be distinct, increasing nonnegative cutoffs")
    if (any(not np.isfinite(r) or r <= 0 for r in args.distances)
            or any(not np.isfinite(g) or g < 0 for g in args.couplings)
            or not np.isfinite(args.omega) or args.omega <= 0):
        parser.error("distances must be positive and couplings nonnegative")
    rows, details, frames = [], [], []
    with lib.with_omp_threads(1):
        for distance in args.distances:
            for coupling in args.couplings:
                row, detail, hf = run_point(distance, coupling, args.omega, args.nmax, args.max_total)
                rows.append(row)
                details.append(detail)
                frames.append(hf)
                print(f"R={distance:g} g={coupling:g} HF={hf.e_tot:.10f} "
                      f"broken={row['broken']} EDcut={row['ed_cutoff_converged']} "
                      f"MPcut={row['mp2_cutoff_converged']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "scan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "diagnostics.json").write_text(json.dumps(details, indent=2) + "\n")
    np.savez(args.output / "frames.npz", mo_coeff=np.array([h.mo_coeff for h in frames]),
             lam=np.array([h.lam for h in frames]), shift=np.array([h.shift for h in frames]))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for distance in args.distances:
        selected = [r for r in rows if r["distance_angstrom"] == distance]
        g = [r["coupling_hartree"] for r in selected]
        for axis, field in zip(axes, ("energy_gain", "charge_imbalance", "spin_square")):
            axis.plot(g, [r[field] for r in selected], "o-", label=f"R={distance:g} Å")
    for axis, label in zip(axes, ("Symmetric - best LF-HF (Ha)", "Charge imbalance", "Spin <S²>")):
        axis.set_xlabel("Supplied coupling g (Ha)")
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(args.output / "symmetry.png", dpi=180)
    figure.savefig(args.output / "symmetry.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
