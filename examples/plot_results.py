"""Redraw the main SI figures from CSV files, without modifying input data."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import REFERENCE, ROOT, read_rows

METHODS = [
    ("ed", "Exact ED", "#202020"),
    ("cs_hf", "CS-HF", "#4b85b7"),
    ("cs_mp2", "CS-MP2", "#67a69e"),
    ("lf_hf", "LF-HF", "#de983e"),
    ("lf_mp2", "LF-MP2", "#b36828"),
    ("mr", "LF-MR", "#8056a1"),
    ("mrpt2", "LF-MRPT2", "#c94646"),
]


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def flag(x):
    return str(x).lower() == "true"


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def comparison(case, data, output):
    rows = read_rows(data / f"{case}_electron.csv")
    if case == "multi" and (data / "multi_cs.csv").exists():
        cs = {(float(r["omega"]), float(r["alpha"])): r for r in read_rows(data / "multi_cs.csv")}
        rows = [
            dict(
                r,
                **{
                    k: v
                    for k, v in cs.get((float(r["omega"]), float(r["alpha"])), {}).items()
                    if k not in ("omega", "alpha")
                },
            )
            for r in rows
        ]
    omegas = sorted({float(r["omega"]) for r in rows})
    fig, axes = plt.subplots(
        2,
        len(omegas),
        figsize=(6 * len(omegas), 7),
        squeeze=False,
        sharex="col",
        constrained_layout=True,
    )
    sym, sa = plt.subplots(
        1, len(omegas), figsize=(6 * len(omegas), 3.5), squeeze=False, constrained_layout=True
    )
    for col, omega in enumerate(omegas):
        group = sorted(
            [r for r in rows if float(r["omega"]) == omega], key=lambda r: float(r["alpha"])
        )
        x = np.array([float(r["alpha"]) for r in group])
        ed = np.array([number(r.get("ed_energy")) for r in group])
        for key, label, color in METHODS:
            vals = np.array([number(r.get(key + "_energy")) for r in group])
            if not np.any(np.isfinite(vals)):
                continue
            axes[0, col].plot(x, vals, color=color, label=label, linewidth=1.2)
            if key != "ed":
                axes[1, col].plot(x, vals - ed, color=color, label=label, linewidth=1.2)
            if key == "mrpt2":
                good = np.array([flag(r.get("mrpt2_converged")) for r in group])
                axes[0, col].scatter(
                    x[~good],
                    vals[~good],
                    facecolors="none",
                    edgecolors=color,
                    s=25,
                    label="MRPT2 finite cutoff",
                )
                axes[0, col].scatter(x[good], vals[good], color=color, s=20)
        axes[0, col].set(title=rf"$\omega={omega:g}$", ylabel=r"$E/|t|$")
        axes[1, col].axhline(0, color="black", linewidth=0.6)
        axes[1, col].set(xlabel=r"$\alpha=g^2/\omega$", ylabel=r"$(E-E_{ED})/|t|$")
        axes[0, col].legend(frameon=False, fontsize=8, ncol=2)
        for prefix, label, color in [
            ("cs", "CS-HF", "#4b85b7"),
            ("hf", "LF-HF", "#de983e"),
            ("mr", "LF-MR", "#8056a1"),
        ]:
            for field, style in [("charge_imbalance", "-"), ("spin_amplitude", "--")]:
                if case == "single" and field == "spin_amplitude":
                    continue
                vals = np.array([number(r.get(prefix + "_" + field)) for r in group])
                if prefix == "mr":
                    ambiguous = np.array([flag(r.get("mr_density_near_degenerate")) for r in group])
                    sa[0, col].scatter(
                        x[ambiguous],
                        vals[ambiguous],
                        marker="x",
                        color="gray",
                        s=20,
                        label="near-degenerate MR root" if ambiguous.any() else None,
                    )
                    vals[ambiguous] = np.nan
                if np.any(np.isfinite(vals)):
                    sa[0, col].plot(
                        x, vals, style, color=color, label=f"{label} {field.split('_')[0]}"
                    )
        sa[0, col].set(
            title=rf"$\omega={omega:g}$", xlabel=r"$\alpha=g^2/\omega$", ylabel="Density indicator"
        )
        sa[0, col].legend(frameon=False, fontsize=8)
        for ax in [axes[0, col], axes[1, col], sa[0, col]]:
            ax.grid(alpha=0.2)
    fig.suptitle(
        f"{case.capitalize()} electron sector; open MRPT2 markers denote finite-cutoff results"
    )
    save(fig, output, f"{case}_energies")
    save(sym, output, f"{case}_symmetry")


def convergence(data, output):
    rows = read_rows(data / "mr_frame_convergence.csv")
    x = [int(r["n_frames"]) for r in rows]
    e = [float(r["energy"]) for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True, constrained_layout=True)
    ax[0].plot(x, e, "o-", label="finite candidate pool")
    ax[0].axhline(float(rows[0]["exact_energy"]), color="black", linestyle="--", label="ED")
    ax[0].fill_between(
        x,
        [float(r["energy_cut_loose"]) for r in rows],
        [float(r["energy_cut_tight"]) for r in rows],
        alpha=0.2,
        label="overlap-cut sensitivity",
    )
    ax[0].set_ylabel("Energy")
    ax[0].legend(frameon=False)
    ax[1].plot(x, [int(r["rank"]) for r in rows], "o-", label="retained rank")
    ax[1].plot(x, np.array(x) * 4, "--", label="nominal dimension")
    ax[1].set(xlabel="Number of frames", ylabel="Dimension")
    ax[1].legend(frameon=False)
    save(fig, output, "mr_frame_convergence")
    rows = read_rows(data / "mr_transferred_frame.csv")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = [float(r["alpha"]) for r in rows]
    for field, label in [
        ("energy_base", "target frames"),
        ("energy_augmented", "target + transferred broken orbit"),
    ]:
        ax.plot(x, [float(r[field]) for r in rows], "o-", label=label)
    ax.set(xlabel=r"$\alpha$", ylabel="MR energy")
    ax.legend(frameon=False)
    save(fig, output, "mr_transferred_frame")
    for case in ["single", "multi"]:
        rows = read_rows(data / f"{case}_mrpt2_cutoff.csv")
        if case == "multi":
            rows += read_rows(data / "multi_mrpt2_m5.csv")
        anchors = [2.2, 2.4] if case == "single" else [0.8, 2.4, 3.2]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        keys = sorted(
            {
                (float(r["omega"]), float(r["alpha"]), int(r["character"]))
                for r in rows
                if any(abs(float(r["alpha"]) - a) < 1e-9 for a in anchors)
                and ("selected_character" not in r or r["character"] == r["selected_character"])
            }
        )
        for w, a, q in keys:
            group = sorted(
                [
                    r
                    for r in rows
                    if float(r["omega"]) == w
                    and abs(float(r["alpha"]) - a) < 1e-9
                    and int(r["character"]) == q
                ],
                key=lambda r: int(r["max_total"]),
            )
            values = [
                number(r.get("corrected_energy"))
                if r["status"] == "ok" and flag(r.get("numerically_valid"))
                else np.nan
                for r in group
            ]
            ax.plot(
                [int(r["max_total"]) for r in group],
                values,
                "o-",
                label=rf"$\omega={w:g},\alpha={a:g},q={q:+d}$",
            )
        ax.set(
            xlabel="Inclusive total-phonon cutoff M",
            ylabel="MRPT2 total energy",
            title="Stored finite-cutoff sequence; no extrapolation",
        )
        ax.legend(frameon=False, fontsize=8)
        save(fig, output, f"{case}_mrpt2_cutoff")
        rows = read_rows(data / f"{case}_mrpt2_overlap.csv")
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for w, a, q in keys:
            group = [
                r
                for r in rows
                if float(r["omega"]) == w
                and abs(float(r["alpha"]) - a) < 1e-9
                and int(r["character"]) == q
                and float(r["reference_cut"]) == 1e-10
            ]
            group = sorted(group, key=lambda r: float(r["external_cut"]))
            ax.plot(
                [float(r["external_cut"]) for r in group],
                [
                    number(r.get("corrected_energy"))
                    if r["status"] == "ok" and flag(r.get("numerically_valid"))
                    else np.nan
                    for r in group
                ],
                "o-",
                label=rf"$\omega={w:g},\alpha={a:g}$",
            )
        ax.set_xscale("log")
        ax.set(
            xlabel="External overlap cutoff",
            ylabel="MRPT2 total energy",
            title="Historical overlap sensitivity at the stored cutoff",
        )
        ax.legend(frameon=False, fontsize=8)
        save(fig, output, f"{case}_mrpt2_overlap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    for case in ["single", "multi"]:
        comparison(case, args.data_dir, args.output_dir)
    convergence(args.data_dir, args.output_dir)
    print(f"Figures saved to {args.output_dir}")
