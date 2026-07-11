"""Validate stage-6 training data structure, joins, labels, and leakage guards."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATASET = PROCESSED / "training_dataset_1h.parquet"
DEFAULT_READY = PROCESSED / "accidents_with_roads_ml_ready.parquet"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "stage6" / "validation"
FORBIDDEN_FEATURES = {"type_dtp", "fd1r17", "fd1r17_descrip", "distance_to_road_m", "target_event_count_1h"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stage-6 road-segment / hourly training dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ready", type=Path, default=DEFAULT_READY)
    return parser.parse_args()


def expected_positive_keys(ready: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    events = ready[["road_segment_id", "accident_datetime"]].copy()
    events["event_hour"] = pd.to_datetime(events["accident_datetime"], errors="raise").dt.floor("h")
    return set(zip(events["road_segment_id"].astype(str), events["event_hour"] - pd.Timedelta(hours=1)))


def verify_strict_history(dataset: pd.DataFrame, ready: pd.DataFrame) -> dict[str, object]:
    events = ready[["road_segment_id", "accident_datetime"]].copy()
    events["road_segment_id"] = events["road_segment_id"].astype(str)
    events["event_hour"] = pd.to_datetime(events["accident_datetime"], errors="raise").dt.floor("h")
    counts = events.groupby(["road_segment_id", "event_hour"], as_index=False).size().rename(columns={"size": "count"})
    mismatches = 0
    for segment, indices in dataset.groupby("road_segment_id").groups.items():
        group = counts.loc[counts["road_segment_id"] == segment].sort_values("event_hour")
        event_ns = group["event_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
        cumulative = group["count"].cumsum().to_numpy(dtype=np.int64)
        query = dataset.loc[list(indices), "datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
        positions = np.searchsorted(event_ns, query, side="left")
        expected = np.where(positions > 0, cumulative[positions - 1], 0)
        actual = dataset.loc[list(indices), "segment_accidents_total_prior"].to_numpy(dtype=np.int64)
        mismatches += int((expected != actual).sum())
    city = counts.groupby("event_hour", as_index=False)["count"].sum().sort_values("event_hour")
    city_ns = city["event_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
    city_cumulative = city["count"].cumsum().to_numpy(dtype=np.int64)
    query = dataset["datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
    positions = np.searchsorted(city_ns, query, side="left")
    city_expected = np.where(positions > 0, city_cumulative[positions - 1], 0)
    city_mismatches = int((city_expected != dataset["city_accidents_total_prior"].to_numpy(dtype=np.int64)).sum())
    return {"segment_total_prior_mismatches": mismatches, "city_total_prior_mismatches": city_mismatches, "passed": mismatches == 0 and city_mismatches == 0}


def main() -> int:
    args = parse_args()
    data = pd.read_parquet(args.dataset)
    ready = pd.read_parquet(args.ready)
    required = {"road_segment_id", "datetime_hour", "target_datetime_hour", "target_1h", "segment_accidents_total_prior", "city_accidents_total_prior"}
    missing_required = sorted(required - set(data.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    data["datetime_hour"] = pd.to_datetime(data["datetime_hour"], errors="raise")
    data["target_datetime_hour"] = pd.to_datetime(data["target_datetime_hour"], errors="raise")
    expected = expected_positive_keys(ready)
    actual_positive = set(zip(data.loc[data["target_1h"] == 1, "road_segment_id"].astype(str), data.loc[data["target_1h"] == 1, "datetime_hour"]))
    negative_keys = set(zip(data.loc[data["target_1h"] == 0, "road_segment_id"].astype(str), data.loc[data["target_1h"] == 0, "datetime_hour"]))
    forbidden_present = sorted(FORBIDDEN_FEATURES.intersection(data.columns))
    history = verify_strict_history(data, ready)
    join_columns = {
        "road": ["road_highway", "road_lanes_num", "road_maxspeed_kmh"],
        "poi": [column for column in data.columns if column.startswith("poi_")],
        "calendar": [column for column in data.columns if column.startswith("calendar_")],
        "weather": [column for column in data.columns if column.startswith("weather_")],
    }
    join_missing = {name: int(data[columns].isna().any(axis=1).sum()) if columns else len(data) for name, columns in join_columns.items()}
    positives = int((data["target_1h"] == 1).sum())
    negatives = int((data["target_1h"] == 0).sum())
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "target_balance": {"positive": positives, "negative": negatives, "negative_to_positive": negatives / positives if positives else None},
        "temporal_coverage": {"start": str(data["datetime_hour"].min()), "end": str(data["datetime_hour"].max()), "timezone_interpretation": "Asia/Almaty"},
        "duplicates": {"road_segment_datetime_hour": int(data.duplicated(["road_segment_id", "datetime_hour"]).sum())},
        "joins": {"rows_with_missing_joined_values": join_missing},
        "missing_values": {column: int(data[column].isna().sum()) for column in data.columns},
        "label_checks": {
            "allowed_target_values": sorted(map(int, pd.unique(data["target_1h"]))),
            "positive_keys_expected": int(len(expected)),
            "positive_keys_actual": int(len(actual_positive)),
            "positive_key_mismatches": int(len(expected.symmetric_difference(actual_positive))),
            "negative_positive_overlap": int(len(negative_keys.intersection(expected))),
        },
        "leakage_checks": {
            "forbidden_post_event_columns": forbidden_present,
            "target_timestamp_is_exactly_next_hour": bool((data["target_datetime_hour"] == data["datetime_hour"] + pd.Timedelta(hours=1)).all()),
            "strict_history": history,
        },
    }
    passed = (
        summary["duplicates"]["road_segment_datetime_hour"] == 0
        and set(summary["label_checks"]["allowed_target_values"]) == {0, 1}
        and summary["label_checks"]["positive_key_mismatches"] == 0
        and summary["label_checks"]["negative_positive_overlap"] == 0
        and not forbidden_present
        and summary["leakage_checks"]["target_timestamp_is_exactly_next_hour"]
        and history["passed"]
        and all(value == 0 for value in join_missing.values())
    )
    summary["final_status"] = "READY" if passed else "FAILED"
    report_dir = REPORTS_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "training_dataset_1h_validation_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Status: {summary['final_status']}")
    print(f"Rows: {len(data)}")
    print(f"Report: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
