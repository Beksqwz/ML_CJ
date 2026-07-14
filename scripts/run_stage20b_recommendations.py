"""Apply the Stage 20B Recommendation Engine to a Stage 20A risk table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from recommendations.stage20b import recommend_stage20b  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Stage 20A Parquet input")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "future_intelligence" / "processed" / "stage20b_recommendations.parquet",
    )
    args = parser.parse_args()
    result = recommend_stage20b(pd.read_parquet(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(result), "output": str(args.output), "status": "ok"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
