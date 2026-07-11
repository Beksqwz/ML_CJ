"""Normalize road attributes for ML-ready accident-road data.

This keeps raw OSM values for traceability and adds numeric/model-safe fields:
``road_lanes_num``, ``road_lanes_missing``, ``road_maxspeed_kmh``,
``road_maxspeed_missing`` and ``road_name_missing``.

Run:
    py scripts/normalize_road_features.py
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "road_features"

LANES_DEFAULTS = {
    "motorway": 3.0,
    "trunk": 3.0,
    "primary": 3.0,
    "secondary": 2.0,
    "tertiary": 2.0,
    "unclassified": 1.0,
    "residential": 1.0,
    "living_street": 1.0,
    "service": 1.0,
    "busway": 1.0,
}
MAXSPEED_DEFAULTS = {
    "motorway": 90.0,
    "trunk": 80.0,
    "primary": 60.0,
    "secondary": 60.0,
    "tertiary": 50.0,
    "unclassified": 40.0,
    "residential": 40.0,
    "living_street": 20.0,
    "service": 20.0,
    "busway": 40.0,
}


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
        description="Normalize lanes and maxspeed in filtered ML accident-road data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_filtered.parquet",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_ready.parquet",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads_ml_ready.csv",
    )
    return parser.parse_args()


def make_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def is_missing(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "[]"}


def parse_list_like(value: Any) -> list[str]:
    if is_missing(value):
        return []
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if not is_missing(item)]
    return [text]


def highway_tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    for item in parse_list_like(value):
        tokens.extend(
            token.strip().lower()
            for token in re.split(r"[,;/|]", item)
            if token.strip()
        )
    return tokens


def default_for_highway(
    value: Any, defaults: dict[str, float], link_default: float
) -> float:
    tokens = highway_tokens(value)
    if any(token.endswith("_link") for token in tokens):
        return link_default
    for token in tokens:
        if token in defaults:
            return defaults[token]
    return defaults["residential"]


def numeric_values(value: Any) -> list[float]:
    if is_missing(value):
        return []
    values: list[float] = []
    for item in parse_list_like(value):
        values.extend(float(match) for match in re.findall(r"\d+(?:\.\d+)?", item))
    return values


def normalize_lanes(row: pd.Series) -> float:
    values = numeric_values(row.get("road_lanes"))
    if values:
        return max(values)
    return default_for_highway(
        row.get("road_highway"), LANES_DEFAULTS, link_default=1.0
    )


def normalize_maxspeed(row: pd.Series) -> float:
    values = numeric_values(row.get("road_maxspeed"))
    if values:
        return max(values)
    return default_for_highway(
        row.get("road_highway"), MAXSPEED_DEFAULTS, link_default=40.0
    )


def normalize_features(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    data = pd.read_parquet(input_path)
    required = {"road_highway", "road_lanes", "road_maxspeed", "road_name"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input is missing road columns: {', '.join(missing)}")

    result = data.copy()
    result["road_lanes_raw"] = result["road_lanes"]
    result["road_maxspeed_raw"] = result["road_maxspeed"]
    result["road_lanes_missing"] = (
        result["road_lanes"].map(is_missing).astype("boolean")
    )
    result["road_maxspeed_missing"] = (
        result["road_maxspeed"].map(is_missing).astype("boolean")
    )
    result["road_name_missing"] = result["road_name"].map(is_missing).astype("boolean")
    result["road_lanes_num"] = result.apply(normalize_lanes, axis=1).astype("float64")
    result["road_maxspeed_kmh"] = result.apply(normalize_maxspeed, axis=1).astype(
        "float64"
    )

    args.output_parquet.resolve().parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_parquet.resolve(), index=False, engine="pyarrow")
    result.to_csv(args.output_csv.resolve(), index=False, encoding="utf-8-sig")

    summary = {
        "input_path": str(input_path),
        "output_parquet": str(args.output_parquet.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "rows": int(len(result)),
        "columns_before": int(len(data.columns)),
        "columns_after": int(len(result.columns)),
        "road_lanes_missing_before": int(result["road_lanes_missing"].sum()),
        "road_maxspeed_missing_before": int(result["road_maxspeed_missing"].sum()),
        "road_name_missing_before": int(result["road_name_missing"].sum()),
        "road_lanes_num_missing_after": int(result["road_lanes_num"].isna().sum()),
        "road_maxspeed_kmh_missing_after": int(
            result["road_maxspeed_kmh"].isna().sum()
        ),
        "road_lanes_num_distribution": result["road_lanes_num"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict(),
        "road_maxspeed_kmh_distribution": result["road_maxspeed_kmh"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def write_report(report_dir: Path, summary: dict[str, Any]) -> None:
    (report_dir / "road_features_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "Road feature normalization report",
        f"Input: {summary['input_path']}",
        f"Rows: {summary['rows']}",
        f"Columns before/after: {summary['columns_before']} / {summary['columns_after']}",
        f"Missing lanes before: {summary['road_lanes_missing_before']}",
        f"Missing maxspeed before: {summary['road_maxspeed_missing_before']}",
        f"Missing name before: {summary['road_name_missing_before']}",
        f"Missing lanes after: {summary['road_lanes_num_missing_after']}",
        f"Missing maxspeed after: {summary['road_maxspeed_kmh_missing_after']}",
        f"Output Parquet: {summary['output_parquet']}",
    ]
    (report_dir / "road_features_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    configure_logging()
    try:
        args = parse_args()
        report_dir = make_report_dir()
        summary = normalize_features(args)
        write_report(report_dir, summary)
        print(f"Rows: {summary['rows']}")
        print(
            f"Missing lanes before/after: {summary['road_lanes_missing_before']} / {summary['road_lanes_num_missing_after']}"
        )
        print(
            f"Missing maxspeed before/after: {summary['road_maxspeed_missing_before']} / {summary['road_maxspeed_kmh_missing_after']}"
        )
        print(f"Output: {summary['output_parquet']}")
        print(f"Report: {report_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("Road feature normalization failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
