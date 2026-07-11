"""Filter leakage-free accident-road data by road-match quality.

Run:
    py scripts/filter_ml_base_accidents.py
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
REPORTS_ROOT = PROJECT_ROOT / "reports" / "ml_filtering"


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
        description="Create filtered ML-ready accident-road base data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_base.parquet",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=300.0,
        help="Maximum accepted nearest-road distance in meters.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_filtered.parquet",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_filtered.csv",
    )
    parser.add_argument(
        "--suspicious-parquet",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_suspicious.parquet",
    )
    parser.add_argument(
        "--suspicious-csv",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_suspicious.csv",
    )
    return parser.parse_args()


def make_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def add_match_confidence(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    distance = pd.to_numeric(result["distance_to_road_m"], errors="coerce")
    result["match_confidence"] = pd.cut(
        distance,
        bins=[-float("inf"), 100.0, 300.0, 1000.0, float("inf")],
        labels=["high", "medium", "low", "suspicious"],
    ).astype("string")
    return result


def filter_ml_base(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    data = pd.read_parquet(input_path)
    required = {
        "objectid",
        "distance_to_road_m",
        "road_segment_id",
        "accident_datetime",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    data = add_match_confidence(data)
    distance = pd.to_numeric(data["distance_to_road_m"], errors="coerce")
    valid_mask = (
        data["road_segment_id"].notna()
        & data["accident_datetime"].notna()
        & distance.le(args.max_distance_m)
    )
    filtered = data.loc[valid_mask].copy()
    suspicious = data.loc[~valid_mask].copy()

    for path, dataframe in (
        (args.output_parquet.resolve(), filtered),
        (args.suspicious_parquet.resolve(), suspicious),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(path, index=False, engine="pyarrow")

    filtered.to_csv(args.output_csv.resolve(), index=False, encoding="utf-8-sig")
    suspicious.to_csv(args.suspicious_csv.resolve(), index=False, encoding="utf-8-sig")

    confidence_counts = data["match_confidence"].value_counts(dropna=False).to_dict()
    filtered_confidence_counts = (
        filtered["match_confidence"].value_counts(dropna=False).to_dict()
    )
    return {
        "input_path": str(input_path),
        "max_distance_m": args.max_distance_m,
        "rows_before": int(len(data)),
        "rows_after": int(len(filtered)),
        "rows_suspicious": int(len(suspicious)),
        "columns_after": int(len(filtered.columns)),
        "confidence_counts_before_filter": {
            str(key): int(value) for key, value in confidence_counts.items()
        },
        "confidence_counts_after_filter": {
            str(key): int(value) for key, value in filtered_confidence_counts.items()
        },
        "output_parquet": str(args.output_parquet.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "suspicious_parquet": str(args.suspicious_parquet.resolve()),
        "suspicious_csv": str(args.suspicious_csv.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_report(report_dir: Path, summary: dict[str, Any]) -> None:
    (report_dir / "ml_filtering_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "ML accident-road filtering report",
        f"Input: {summary['input_path']}",
        f"Max distance: {summary['max_distance_m']} m",
        f"Rows before/after/suspicious: {summary['rows_before']} / {summary['rows_after']} / {summary['rows_suspicious']}",
        f"Confidence before: {summary['confidence_counts_before_filter']}",
        f"Confidence after: {summary['confidence_counts_after_filter']}",
        f"Output Parquet: {summary['output_parquet']}",
        f"Suspicious Parquet: {summary['suspicious_parquet']}",
    ]
    (report_dir / "ml_filtering_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    configure_logging()
    try:
        args = parse_args()
        report_dir = make_report_dir()
        summary = filter_ml_base(args)
        write_report(report_dir, summary)
        print(f"Rows kept: {summary['rows_after']}/{summary['rows_before']}")
        print(f"Suspicious rows: {summary['rows_suspicious']}")
        print(f"Output: {summary['output_parquet']}")
        print(f"Report: {report_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("ML filtering failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
