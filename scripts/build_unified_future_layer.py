"""CLI for building the canonical Stage 18B unified Future Feature Layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_intelligence.unified_future_layer import (  # noqa: E402
    UnifiedFutureLayerBuilder,
    UnifiedPaths,
    feature_provenance,
    validate,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment-features-path",
        type=Path,
        default=UnifiedPaths().segment_features_path,
    )
    parser.add_argument("--output-dir", type=Path, default=UnifiedPaths().output_dir)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = UnifiedPaths(
        segment_features_path=args.segment_features_path,
        output_dir=args.output_dir,
    )
    builder = UnifiedFutureLayerBuilder(paths)
    frame, integration, collisions = builder.build()
    validation = validate(frame)
    reports_dir = ROOT / "reports" / "stage18b"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(reports_dir / "integration_report.json", integration)
    _write_json(
        reports_dir / "provider_coverage.json", integration["provider_coverage"]
    )
    _write_json(reports_dir / "collision_report.json", collisions)
    _write_json(
        reports_dir / "feature_provenance.json",
        feature_provenance(frame, args.segment_features_path),
    )
    _write_json(reports_dir / "validation_report.json", validation)

    storage = {} if args.dry_run else builder.save(frame)
    print(json.dumps({"rows": len(frame), "validation": validation, **storage}))
    return 0 if all(validation.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
