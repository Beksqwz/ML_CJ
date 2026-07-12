"""Fetch one current TomTom flow reading to verify local configuration."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_service import TomTomTrafficService


def main() -> None:
    with (ROOT / "astana_edges.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    segment_id = f"{row['u']}_{row['v']}_{row['key']}"
    print(json.dumps(TomTomTrafficService().get_segment(segment_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
