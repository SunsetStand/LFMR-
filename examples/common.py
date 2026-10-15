"""Read bundled reference tables for the public plotting example."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_data"


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))
