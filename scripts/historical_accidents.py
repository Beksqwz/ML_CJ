"""Sanity-check historical accident source files.

The current repository already contains the downloaded Astana accident source
data in CSV and Parquet form. This script does not re-download from ArcGIS; it
verifies that the immutable source artifacts required by later stages exist.

Run:
    py scripts/historical_accidents.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = [
    PROJECT_ROOT / "astana_accidents.csv",
    PROJECT_ROOT / "astana_accidents.parquet",
]


def main() -> int:
    missing = [path for path in RAW_FILES if not path.is_file()]
    if missing:
        print("Missing historical accident source files:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("Historical accident source files are present:")
    for path in RAW_FILES:
        print(f"- {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size} bytes)")
    print("These files are treated as immutable inputs for stages 2+.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
