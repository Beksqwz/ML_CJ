"""Collect non-model future context from registered providers."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass
from future_intelligence.pipeline import FutureIntelligencePipeline  # noqa: E402
from future_intelligence.storage import save_result  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", default="openweather")
    parser.add_argument("--prediction-datetime", required=True)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "future_intelligence"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    names = tuple(item.strip() for item in args.providers.split(",") if item.strip())
    pipeline = FutureIntelligencePipeline()
    context = pipeline.collect(
        args.prediction_datetime,
        args.horizon_hours,
        names,
        latitude=args.latitude,
        longitude=args.longitude,
        strict=args.strict,
    )
    if not args.dry_run:
        paths = [
            save_result(
                result, args.output_dir, args.prediction_datetime.replace(":", "-")
            )
            for result in pipeline.last_results
        ]
        context["storage"] = [
            {name: str(path) for name, path in saved.items()} for saved in paths
        ]
        report = (
            ROOT / "reports" / "stage15" / "future_intelligence_validation_report.json"
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(context, ensure_ascii=False, indent=2))
    return 0 if context["status"] == "ok" or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
