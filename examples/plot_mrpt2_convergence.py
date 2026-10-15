"""Plot phonon-cutoff-converged MRPT2 alpha scans against ED, LF-HF/MP2 and MR.

Reads the per-point CSV histories written by examples.mrpt2_m_convergence
(<runs>/<case>/w<omega>_a<alpha>.csv) and the archived ED/LF/MR columns in
reference_data. For each point and external cut the plotted MRPT2 value is
the row at the largest computed M. A cut is converged when two successive
increments are below --tol. A numerical failure is drawn at the MR energy.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .common import REFERENCE, ROOT, read_rows, write_rows
from .mrpt2_m_convergence import converged_at, history

CUTS = (1e-8, 1e-10)
STYLE = {
    "ed": {"color": "#202020", "lw": 1.6, "label": "Converged ED"},
    "lf_hf": {"color": "#3465a4", "lw": 1.4, "marker": "o", "ms": 3.5, "label": "LF-HF"},
    "lf_mp2": {"color": "#e08a1e", "lw": 1.4, "ls": "--", "marker": "o", "ms": 3.5},
    "mr": {
        "color": "#7e4fa3",
        "lw": 1.4,
        "marker": "o",
        "ms": 3.5,
        "label": "LF-MR (fixed frames)",
    },
    1e-10: {"color": "#e39a9a", "lw": 2.2, "ls": ":", "label": r"MRPT2 (external cut $10^{-10}$)"},
    1e-8: {"color": "#cc3333", "lw": 2.2, "label": r"MRPT2 (external cut $10^{-8}$)"},
}


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def final_rows(path, tol, consecutive):
    """Final row, converged M and failure flag for each cut at one point."""
    rows = read_rows(path)
    out = {}
    for cut in CUTS:
        mine = [r for r in rows if abs(float(r["external_cut"]) - cut) < 1e-3 * cut]
        if not mine:
            continue
        top = max(int(r["max_total"]) for r in mine)
        at_top = [r for r in mine if int(r["max_total"]) == top]
        ok = [r for r in at_top if r["status"] == "ok"]
        row = ok[-1] if ok else at_top[-1]
        conv = converged_at(history(rows, cut), tol, consecutive)
        out[cut] = {
            "row": row,
            "max_total": top,
            "converged_M": conv,
            "failed": row["status"] != "ok",
            "energy": number(row.get("corrected_energy")) if row["status"] == "ok" else np.nan,
        }
    return out


def collect(case, runs, tol, consecutive):
    table = {
        (float(r["omega"]), round(float(r["alpha"]), 6)): r
        for r in read_rows(REFERENCE / f"{case}_electron.csv")
    }
    points = {}
    for path in glob.glob(str(runs / case / "w*_a*.csv")):
        w, a = (float(x[1:]) for x in os.path.basename(path)[:-4].split("_"))
        points[(w, round(a, 6))] = final_rows(path, tol, consecutive)
    return table, points


def summary_rows(case, table, points):
    rows = []
    for (w, a), ref in sorted(table.items()):
        pt = points.get((w, a), {})
        row = {
            "case": case,
            "omega": w,
            "alpha": a,
            "ed_energy": ref["ed_energy"],
            "mr_energy": ref["mr_energy"],
        }
        for cut in CUTS:
            tag = f"cut{-round(np.log10(cut))}"
            info = pt.get(cut)
            if info is None:
                row[f"mrpt2_{tag}_status"] = "missing"
                continue
            r = info["row"]
            row.update(
                {
                    f"mrpt2_{tag}_energy": info["energy"],
                    f"mrpt2_{tag}_max_total": info["max_total"],
                    f"mrpt2_{tag}_converged_M": info["converged_M"] or "",
                    f"mrpt2_{tag}_converged": info["converged_M"] is not None,
                    f"mrpt2_{tag}_status": r["status"],
                    f"mrpt2_{tag}_algorithm": r.get("algorithm") or "full",
                    f"mrpt2_{tag}_precision_mode": r.get("precision_mode", ""),
                    f"mrpt2_{tag}_external_rank": r.get("external_rank", ""),
                    f"mrpt2_{tag}_failure": r.get("failure", ""),
                }
            )
        rows.append(row)
    return rows


def panel(ax_e, ax_d, table, points, omega, lf_mp2_label, linthresh):
    keys = sorted(k for k in table if k[0] == omega)
    x = np.array([k[1] for k in keys])
    col = {f: np.array([number(table[k].get(f)) for k in keys]) for f in table[keys[0]]}
    ed = col["ed_energy"]
    curves = [
        ("ed", ed),
        ("lf_hf", col["lf_hf_energy"]),
        ("lf_mp2", col["lf_mp2_energy"]),
        ("mr", col["mr_energy"]),
    ]
    for key, y in curves:
        style = dict(STYLE[key])
        if key == "lf_mp2":
            style["label"] = lf_mp2_label
        ax_e.plot(x, y, **style)
        if key != "ed":
            ax_d.plot(x, y - ed, **{k: v for k, v in style.items() if k != "label"})
    for cut in (1e-10, 1e-8):
        info = [points.get(k, {}).get(cut) for k in keys]
        y = np.array([i["energy"] if i else np.nan for i in info])
        failed = np.array([bool(i and i["failed"]) for i in info])
        conv = np.array([bool(i and i["converged_M"] is not None) for i in info])
        style = STYLE[cut]
        ax_e.plot(x, y, color=style["color"], lw=style["lw"], ls=style.get("ls", "-"))
        ax_d.plot(x, y - ed, color=style["color"], lw=style["lw"], ls=style.get("ls", "-"))
        mr = col["mr_energy"]
        for ax, base in ((ax_e, np.zeros_like(ed)), (ax_d, ed)):
            ax.scatter(
                x[failed], (mr - base)[failed], marker="x", s=60, color="#e39a9a", lw=1.6, zorder=5
            )
            if cut == 1e-8:
                good = ~failed & np.isfinite(y)
                ax.scatter(
                    x[good & conv],
                    (y - base)[good & conv],
                    marker="D",
                    s=46,
                    color=style["color"],
                    edgecolors=style["color"],
                    zorder=6,
                )
                ax.scatter(
                    x[good & ~conv],
                    (y - base)[good & ~conv],
                    marker="D",
                    s=46,
                    facecolors="white",
                    edgecolors=style["color"],
                    lw=1.5,
                    zorder=6,
                )
    ax_d.axhline(0, color="black", lw=0.8)
    ax_d.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    ax_e.set_title(rf"$\omega={omega:g}$", fontsize=13)
    ax_e.set_ylabel(r"$E/|t|$")
    ax_d.set_ylabel(r"$(E-E_{\rm ED})/|t|$")
    ax_d.set_xlabel(r"$\alpha=g^2/\omega$")
    for ax in (ax_e, ax_d):
        ax.grid(alpha=0.25)


def legend(ax, lf_mp2_label, tol, incomplete, failed):
    """Legend; marker entries appear only for categories present in the figure."""
    handles = [
        Line2D([], [], **{k: v for k, v in STYLE["ed"].items()}),
        Line2D([], [], color=STYLE["lf_hf"]["color"], lw=1.4, label="LF-HF"),
        Line2D([], [], color=STYLE["lf_mp2"]["color"], lw=1.4, ls="--", label=lf_mp2_label),
        Line2D([], [], color=STYLE["mr"]["color"], lw=1.4, label=STYLE["mr"]["label"]),
        Line2D([], [], **STYLE[1e-10]),
        Line2D([], [], **STYLE[1e-8]),
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            ms=7,
            color=STYLE[1e-8]["color"],
            label=rf"MRPT2 $M$-converged ($|\Delta E|<10^{{{int(np.log10(tol))}}}$ twice)",
        ),
    ]
    if incomplete:
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker="D",
                ms=7,
                mfc="white",
                mec=STYLE[1e-8]["color"],
                mew=1.5,
                label="MRPT2 convergence incomplete",
            )
        )
    if failed:
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker="x",
                ms=8,
                mew=1.6,
                color="#e39a9a",
                label="MRPT2 numerical failure (at MR energy)",
            )
        )
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.5)


def figure(case, table, points, output, tol, m_range):
    omegas = sorted({k[0] for k in table})
    first = table[min(table)]
    ne = int(first["neleca"]) + int(first["nelecb"])
    lf_mp2_label = f"LF-MP2 (M={int(float(first['lf_mp2_max_total']))})"
    fig, axes = plt.subplots(
        2,
        len(omegas),
        figsize=(6.2 * len(omegas) + 1.2, 8.6),
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1]},
    )
    infos = [i for p in points.values() for i in p.values()]
    incomplete = any(
        p.get(1e-8) and p[1e-8]["converged_M"] is None and not p[1e-8]["failed"]
        for p in points.values()
    )
    failed = any(i["failed"] for i in infos)
    for c, omega in enumerate(omegas):
        panel(axes[0, c], axes[1, c], table, points, omega, lf_mp2_label, 1e-4)
        legend(axes[0, c], lf_mp2_label, tol, incomplete, failed)
    U = float(first["U"])
    fig.suptitle(
        f"Four-site Hubbard-Holstein: Ne={ne}, U={U:g}, t=-1\n"
        rf"Block-Hamiltonian MRPT2; fixed LF frames, all modes, $M$ converged "
        rf"(${m_range[0]}\leq M^*\leq {m_range[1]}$)",
        fontsize=14,
    )
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    name = f"{case}_mrpt2_m_converged"
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output / f"{name}.png"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--consecutive", type=int, default=2)
    a = p.parse_args()
    for case in ("single", "multi"):
        table, points = collect(case, a.runs, a.tol, a.consecutive)
        if not points:
            continue
        summary = summary_rows(case, table, points)
        write_rows(a.output_dir.parent / f"{case}_mrpt2_m_converged.csv", summary)
        ms = [p[1e-8]["converged_M"] or p[1e-8]["max_total"] for p in points.values() if 1e-8 in p]
        print(figure(case, table, points, a.output_dir, a.tol, (min(ms), max(ms))))
