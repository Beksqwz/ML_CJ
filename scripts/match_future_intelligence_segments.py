"""Match persisted Future Intelligence events to production road segments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_intelligence.schemas import FutureRecord  # noqa: E402
from future_intelligence.spatial_matching import (  # noqa: E402
    SpatialMatchingEngine,
    SpatialMatchResult,
    save_matching_report,
    save_segment_matches,
)


def _datetime(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).to_pydatetime()


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return value


def load_records(path: Path) -> list[FutureRecord]:
    records = []
    for row in pd.read_parquet(path).to_dict("records"):
        payload = _json(row.get("payload_json"), {})
        warnings = _json(row.get("warnings"), [])
        geometry = _json(row.get("geometry"), None)
        affected = _json(row.get("affected_road_segment_ids"), [])
        records.append(
            FutureRecord(
                source=row["source"],
                source_type=row["source_type"],
                source_version=row["source_version"],
                source_item_id=row.get("source_item_id"),
                source_url=row.get("source_url"),
                collected_at=_datetime(row.get("collected_at")) or datetime.now(),
                published_at=_datetime(row.get("published_at")),
                valid_from=_datetime(row.get("valid_from")),
                valid_to=_datetime(row.get("valid_to")),
                prediction_datetime=_datetime(row.get("prediction_datetime"))
                or datetime.now(),
                horizon_hours=int(row.get("horizon_hours") or 24),
                latitude=None if pd.isna(row.get("latitude")) else row.get("latitude"),
                longitude=None
                if pd.isna(row.get("longitude"))
                else row.get("longitude"),
                geometry=geometry,
                affected_road_segment_ids=affected
                if isinstance(affected, list)
                else [],
                event_type=row.get("event_type"),
                severity=row.get("severity"),
                confidence=None
                if pd.isna(row.get("confidence"))
                else row.get("confidence"),
                is_forecast=bool(row.get("is_forecast", True)),
                is_realtime=bool(row.get("is_realtime", False)),
                is_historical=bool(row.get("is_historical", False)),
                payload=payload,
                warnings=warnings if isinstance(warnings, list) else [],
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gov-input",
        type=Path,
        default=ROOT
        / "data"
        / "future_intelligence"
        / "processed"
        / "gov_kz_road_events.parquet",
    )
    parser.add_argument(
        "--ticketon-input",
        type=Path,
        default=ROOT
        / "data"
        / "future_intelligence"
        / "processed"
        / "ticketon_events.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "future_intelligence",
    )
    parser.add_argument("--ticketon-radius-m", type=float, default=1000.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = SpatialMatchingEngine(ticketon_radius_m=args.ticketon_radius_m)
    combined = SpatialMatchResult()
    for path, provider in (
        (args.gov_input, "gov_kz_repairs"),
        (args.ticketon_input, "ticketon_events"),
    ):
        if not path.exists():
            combined.warnings.append(f"input_not_found:{path.name}")
            continue
        result = engine.match_records(load_records(path), provider)
        combined.matches.extend(result.matches)
        combined.unmatched.extend(result.unmatched)
        combined.warnings.extend(result.warnings)

    summary = {"matches": len(combined.matches), "unmatched": len(combined.unmatched)}
    if not args.dry_run:
        paths, changes = save_segment_matches(combined.matches, args.output_dir)
        report = save_matching_report(
            combined, ROOT / "reports" / "stage17" / "spatial_matching_report.json"
        )
        summary["storage"] = {name: str(path) for name, path in paths.items()}
        summary["changes"] = changes
        summary["report"] = str(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
