"""Small IO and fixed-frame helpers shared by the example recipes."""

import csv
import json
from pathlib import Path

import numpy as np

from mr_lf import HubbardHolstein

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_data"


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path, metadata):
    Path(path).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def model_for(case, omega, alpha):
    return HubbardHolstein.ring(
        4,
        nelec=(1, 0) if case == "single" else (2, 2),
        U=0.0 if case == "single" else 4.0,
        omega=omega,
        alpha=alpha,
    )


def frame_point(case, omega, alpha, reference_dir=REFERENCE):
    """Load only supplied frames; perform no optimization or external IO."""
    path = Path(reference_dir) / f"{case}_frames.npz"
    with np.load(path) as z:
        matching = np.isclose(z["alpha"], alpha, atol=1e-12, rtol=0)
        if "omega" in z:
            matching &= np.isclose(z["omega"], omega, atol=1e-12, rtol=0)
        elif omega != 0.5:
            raise ValueError("single-electron archived frames have omega=0.5")
        indices = np.flatnonzero(matching)
        if len(indices) != 1:
            raise ValueError("requested point is not uniquely present in the frame archive")
        i = int(indices[0])
        character = int(z["character"][i]) if "character" in z else 1
        return z["lam"][i].copy(), z["shift"][i].copy(), character


def density_fields(prefix, density):
    from mr_lf.observables import symmetry_indicators

    return {
        **{
            f"{prefix}_rho_{s}_{p}": float(density[s, p])
            for s in range(2)
            for p in range(density.shape[1])
        },
        **{f"{prefix}_{key}": value for key, value in symmetry_indicators(density).items()},
    }


def antiferromagnetic_start(L=4):
    eye = np.eye(L)
    even = list(range(0, L, 2))
    odd = list(range(1, L, 2))
    return np.stack((eye[:, even + odd], eye[:, odd + even]))


def charge_ordered_start(L=4):
    coeff = antiferromagnetic_start(L)[0]
    return np.stack((coeff, coeff))
