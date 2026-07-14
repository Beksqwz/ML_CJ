"""Build deterministic, context-only 24-hour Future Intelligence segment features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_intelligence.segment_feature_builder import (  # noqa: E402
    BuilderPaths,
    FutureSegmentFeatureBuilder,
    feature_catalog,
    validate_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-datetime", required=True)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument(
        "--road-network-path", type=Path, default=BuilderPaths().road_network_path
    )
    parser.add_argument(
        "--matches-path", type=Path, default=BuilderPaths().matches_path
    )
    parser.add_argument(
        "--weather-features-path",
        type=Path,
        default=BuilderPaths().weather_features_path,
    )
    parser.add_argument(
        "--gov-events-path", type=Path, default=BuilderPaths().gov_events_path
    )
    parser.add_argument(
        "--ticketon-events-path", type=Path, default=BuilderPaths().ticketon_events_path
    )
    parser.add_argument("--output-dir", type=Path, default=BuilderPaths().output_dir)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = BuilderPaths(
        road_network_path=args.road_network_path,
        matches_path=args.matches_path,
        weather_features_path=args.weather_features_path,
        gov_events_path=args.gov_events_path,
        ticketon_events_path=args.ticketon_events_path,
        output_dir=args.output_dir,
    )
    builder = FutureSegmentFeatureBuilder(paths)
    frame, report = builder.build(args.prediction_datetime, args.horizon_hours)
    validation = {
        key: bool(value)
        for key, value in validate_features(
            frame, set(builder._production_ids())
        ).items()
    }
    reports = ROOT / "reports" / "stage18a"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "input_audit.json").write_text(
        json.dumps(builder.input_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "feature_catalog.json").write_text(
        json.dumps(feature_catalog(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.dry_run:
        builder.save(frame, report)
    (reports / "feature_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "feature_validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(frame),
                "validation": validation,
                "warnings": report["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(validation.values()) or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
