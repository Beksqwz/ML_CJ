"""Build the Stage 20A hybrid risk table for one 24-hour prediction window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml_service.hybrid_risk import build_hybrid_risk  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_datetime")
    parser.add_argument("--future-context", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "risk" / "stage20a_hybrid_risk.parquet")
    args = parser.parse_args()
    context = pd.read_parquet(args.future_context) if args.future_context else None
    frame = build_hybrid_risk(args.prediction_datetime, future_context=context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(frame), "output": str(args.output), "status": "ok"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
