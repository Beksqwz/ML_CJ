"""Create a leakage-free accident-to-road base table for ML dataset building.

The source accident tables intentionally keep all original police-report fields.
This script creates a narrower ML staging table that keeps only fields known
before prediction time or needed to build targets.

Run:
    py scripts/create_ml_base_accidents.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "ml_base"

KEEP_COLUMNS = [
    # Stable identifiers / grouping
    "objectid",
    "globalid",
    "area_code",
    # Time needed to create segment-hour targets and time features
    "rta_date",
    "accident_date",
    "accident_datetime",
    "hour",
    "weekday",
    "day",
    "month",
    "year",
    "quarter",
    "season",
    "is_weekend",
    # Location needed for QA and possible spatial aggregation
    "longitude",
    "latitude",
    # Road matching result and road attributes
    "road_segment_id",
    "distance_to_road_m",
    "road_u",
    "road_v",
    "road_key",
    "road_osmid",
    "road_highway",
    "road_lanes",
    "road_name",
    "road_oneway",
    "road_reversed",
    "road_length",
    "road_maxspeed",
    "road_ref",
    "road_bridge",
    "road_tunnel",
    "road_width",
    "road_junction",
    "road_access",
]


def configure_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-free accident-road ML base data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads.parquet",
        help="Matched accident-road Parquet input.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_base.parquet",
        help="Leakage-free ML base Parquet output.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_base.csv",
        help="Leakage-free ML base CSV output.",
    )
    return parser.parse_args()


def make_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def create_ml_base(
    input_path: Path, output_parquet: Path, output_csv: Path
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    dataframe = pd.read_parquet(input_path)
    missing_required = [
        column for column in KEEP_COLUMNS if column not in dataframe.columns
    ]
    if missing_required:
        raise ValueError(
            f"Input is missing required ML base columns: {', '.join(missing_required)}"
        )

    ml_base = dataframe.loc[:, KEEP_COLUMNS].copy()
    if ml_base["objectid"].duplicated().any():
        raise ValueError("objectid must be unique before ML base creation.")
    if ml_base["road_segment_id"].isna().any():
        raise ValueError("road_segment_id must be filled before ML base creation.")
    if ml_base["accident_datetime"].isna().any():
        raise ValueError("accident_datetime must be filled before ML base creation.")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    ml_base.to_parquet(output_parquet, index=False, engine="pyarrow")
    ml_base.to_csv(output_csv, index=False, encoding="utf-8-sig")

    removed_columns = [
        column for column in dataframe.columns if column not in KEEP_COLUMNS
    ]
    likely_leakage_removed = [
        column
        for column in removed_columns
        if column.startswith("fd")
        or column
        in {
            "type_dtp",
            "vehicle_category",
            "is_public_transport",
            "x",
            "y",
            "geometry",
            "load_date",
        }
    ]
    return {
        "input_path": str(input_path),
        "output_parquet": str(output_parquet),
        "output_csv": str(output_csv),
        "rows_before": int(len(dataframe)),
        "columns_before": int(len(dataframe.columns)),
        "rows_after": int(len(ml_base)),
        "columns_after": int(len(ml_base.columns)),
        "kept_columns": KEEP_COLUMNS,
        "removed_columns": removed_columns,
        "likely_leakage_removed": likely_leakage_removed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_report(report_dir: Path, summary: dict[str, Any]) -> None:
    (report_dir / "ml_base_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "ML base accident-road dataset report",
        f"Input: {summary['input_path']}",
        f"Rows before/after: {summary['rows_before']} / {summary['rows_after']}",
        f"Columns before/after: {summary['columns_before']} / {summary['columns_after']}",
        f"Output Parquet: {summary['output_parquet']}",
        f"Output CSV: {summary['output_csv']}",
        "",
        "Removed columns:",
        *[f"- {column}" for column in summary["removed_columns"]],
    ]
    (report_dir / "ml_base_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    configure_logging()
    try:
        args = parse_args()
        report_dir = make_report_dir()
        summary = create_ml_base(
            args.input.resolve(),
            args.output_parquet.resolve(),
            args.output_csv.resolve(),
        )
        write_report(report_dir, summary)
        print(f"Rows: {summary['rows_after']}")
        print(f"Columns: {summary['columns_after']}")
        print(f"Removed columns: {len(summary['removed_columns'])}")
        print(f"Output Parquet: {summary['output_parquet']}")
        print(f"Report: {report_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("ML base creation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
