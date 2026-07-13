"""Collect the OpenWeather 24-hour future context only."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.collect_future_intelligence import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
