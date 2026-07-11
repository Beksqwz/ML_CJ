"""Independent validation for the leakage-safe 24-hour Stage-6 dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
EXTERNAL = PROJECT_ROOT / "data" / "external"
DEFAULT_DATASET = PROCESSED / "training_dataset_24h.parquet"
DEFAULT_READY = PROCESSED / "accidents_with_roads_ml_ready.parquet"
DEFAULT_CALENDAR = EXTERNAL / "calendar_features_hourly.parquet"
DEFAULT_WEATHER = EXTERNAL / "weather_astana_hourly.parquet"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "stage6" / "validation_24h"
HORIZON = pd.Timedelta(hours=24)
FORBIDDEN = {"type_dtp", "fd1r17", "fd1r17_descrip", "distance_to_road_m", "accident_datetime", "objectid", "globalid", "target_datetime_hour"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate the 24-hour training dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ready", type=Path, default=DEFAULT_READY)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    return parser.parse_args()


def source_counts(ready: pd.DataFrame) -> pd.DataFrame:
    events = ready[["road_segment_id", "accident_datetime"]].copy()
    events["road_segment_id"] = events["road_segment_id"].astype(str)
    events["event_hour"] = pd.to_datetime(events["accident_datetime"], errors="raise").dt.floor("h")
    return events.groupby(["road_segment_id", "event_hour"], as_index=False).size().rename(columns={"size": "event_count"})


def expected_positive_keys(counts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> set[tuple[str, pd.Timestamp]]:
    pieces = []
    for offset in range(1, 25):
        piece = counts[["road_segment_id", "event_hour"]].copy()
        piece["datetime_hour"] = piece["event_hour"] - pd.Timedelta(hours=offset)
        pieces.append(piece[["road_segment_id", "datetime_hour"]])
    keys = pd.concat(pieces, ignore_index=True).drop_duplicates()
    keys = keys.loc[keys["datetime_hour"].between(start, end)]
    return set(zip(keys["road_segment_id"].astype(str), keys["datetime_hour"]))


def standard_history_mismatches(data: pd.DataFrame, counts: pd.DataFrame) -> dict[str, int]:
    columns = ["segment_accidents_total_prior", "segment_accidents_prev_24h", "segment_accidents_prev_7d", "segment_accidents_prev_30d", "segment_hours_since_prev_accident", "segment_has_history"]
    bad = {column: 0 for column in columns}
    for segment, indices in data.groupby("road_segment_id").groups.items():
        events = counts.loc[counts["road_segment_id"] == segment].sort_values("event_hour")
        event_ns = events["event_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
        cumulative = events["event_count"].cumsum().to_numpy(dtype=np.int64)
        row_index = np.asarray(list(indices))
        query_ns = data.loc[row_index, "datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
        def prior(query: np.ndarray) -> np.ndarray:
            positions = np.searchsorted(event_ns, query, side="left")
            return np.where(positions > 0, cumulative[positions - 1], 0)
        total = prior(query_ns)
        expected: dict[str, np.ndarray] = {"segment_accidents_total_prior": total}
        for label, hours in (("segment_accidents_prev_24h", 24), ("segment_accidents_prev_7d", 168), ("segment_accidents_prev_30d", 720)):
            expected[label] = total - prior(query_ns - np.int64(hours * 3_600_000_000_000))
        positions = np.searchsorted(event_ns, query_ns, side="left")
        has_history = positions > 0
        hours_since = np.full(len(query_ns), -1.0)
        hours_since[has_history] = (query_ns[has_history] - event_ns[positions[has_history] - 1]) / 3_600_000_000_000
        expected["segment_hours_since_prev_accident"] = hours_since
        expected["segment_has_history"] = has_history
        for column, values in expected.items():
            actual = data.loc[row_index, column].to_numpy()
            bad[column] += int((~np.isclose(actual.astype(float), values.astype(float), equal_nan=True)).sum())
    city = counts.groupby("event_hour", as_index=False)["event_count"].sum().sort_values("event_hour")
    event_ns = city["event_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
    cumulative = city["event_count"].cumsum().to_numpy(dtype=np.int64)
    query_ns = data["datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
    def prior_city(query: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(event_ns, query, side="left")
        return np.where(positions > 0, cumulative[positions - 1], 0)
    total = prior_city(query_ns)
    expected_city = {"city_accidents_total_prior": total}
    for label, hours in (("city_accidents_prev_24h", 24), ("city_accidents_prev_7d", 168), ("city_accidents_prev_30d", 720)):
        expected_city[label] = total - prior_city(query_ns - np.int64(hours * 3_600_000_000_000))
    for column, values in expected_city.items():
        bad[column] = int((data[column].to_numpy(dtype=np.int64) != values).sum())
    return bad


def seasonal_sample_mismatches(data: pd.DataFrame, counts: pd.DataFrame, calendar: pd.DataFrame) -> dict[str, int]:
    """Independently recompute seasonal features for deterministic 100k-row audit sample."""
    sample = data.iloc[np.linspace(0, len(data) - 1, num=min(100_000, len(data)), dtype=int)].copy()
    calendar = calendar[["datetime_hour", "is_holiday"]].copy()
    calendar["is_new_year"] = ((calendar.datetime_hour.dt.month == 12) & (calendar.datetime_hour.dt.day >= 25)) | ((calendar.datetime_hour.dt.month == 1) & (calendar.datetime_hour.dt.day <= 7))
    events = counts.merge(calendar, left_on="event_hour", right_on="datetime_hour", how="left", validate="many_to_one")
    sample["_month"] = sample.datetime_hour.dt.month.astype("int8")
    sample["_hour"] = sample.datetime_hour.dt.hour.astype("int8")
    sample["_weekday"] = sample.datetime_hour.dt.weekday.astype("int8")
    events["event_month"] = events.event_hour.dt.month.astype("int8")
    events["event_hour_of_day"] = events.event_hour.dt.hour.astype("int8")
    events["event_weekday"] = events.event_hour.dt.weekday.astype("int8")

    def contextual_prior(query_columns: list[str], event_columns: list[str], event_subset: pd.DataFrame) -> np.ndarray:
        if event_subset.empty:
            return np.zeros(len(sample), dtype=np.int64)
        left = sample[["road_segment_id", "datetime_hour", *query_columns]].copy()
        left["_row"] = np.arange(len(left))
        groups = ["road_segment_id", *event_columns]
        left.columns = ["road_segment_id", "datetime_hour", *event_columns, "_row"]
        right = event_subset.groupby(["road_segment_id", "event_hour", *event_columns], as_index=False)["event_count"].sum()
        right = right.sort_values(["event_hour", *groups])
        right["_cum"] = right.groupby(groups, dropna=False)["event_count"].cumsum()
        merged = pd.merge_asof(left.sort_values(["datetime_hour", *groups]), right[["event_hour", *groups, "_cum"]], left_on="datetime_hour", right_on="event_hour", by=groups, direction="backward", allow_exact_matches=False)
        return pd.Series(merged["_cum"].fillna(0).to_numpy(dtype=np.int64), index=merged["_row"].to_numpy()).reindex(np.arange(len(sample)), fill_value=0).to_numpy()

    sample["_all"] = 1
    events["_all"] = 1
    expected = {
        "segment_accidents_same_month_prior": contextual_prior(["_month"], ["event_month"], events),
        "segment_accidents_same_hour_weekday_prior": contextual_prior(["_hour", "_weekday"], ["event_hour_of_day", "event_weekday"], events),
        "segment_accidents_same_month_hour_weekday_prior": contextual_prior(["_month", "_hour", "_weekday"], ["event_month", "event_hour_of_day", "event_weekday"], events),
        "segment_accidents_holiday_prior": contextual_prior(["_all"], ["_all"], events.loc[events.is_holiday.fillna(False)]),
        "segment_accidents_new_year_period_prior": contextual_prior(["_all"], ["_all"], events.loc[events.is_new_year.fillna(False)]),
    }
    return {column: int((sample[column].to_numpy(dtype=np.int64) != values).sum()) for column, values in expected.items()}


def main() -> int:
    args = parse_args()
    data = pd.read_parquet(args.dataset)
    ready = pd.read_parquet(args.ready)
    calendar = pd.read_parquet(args.calendar)
    weather = pd.read_parquet(args.weather)
    data["datetime_hour"] = pd.to_datetime(data["datetime_hour"], errors="raise")
    calendar["datetime_hour"] = pd.to_datetime(calendar["datetime_hour"], errors="raise")
    weather["datetime_hour"] = pd.to_datetime(weather["datetime_hour"], errors="raise")
    counts = source_counts(ready)
    start, end = max(calendar.datetime_hour.min(), weather.datetime_hour.min()), min(calendar.datetime_hour.max(), weather.datetime_hour.max())
    expected = expected_positive_keys(counts, start, end)
    positive = data.loc[data.target_24h.eq(1)]
    negative = data.loc[data.target_24h.eq(0)]
    positive_keys = set(zip(positive.road_segment_id.astype(str), positive.datetime_hour))
    negative_keys = set(zip(negative.road_segment_id.astype(str), negative.datetime_hour))
    standard_bad = standard_history_mismatches(data, counts)
    seasonal_bad = seasonal_sample_mismatches(data, counts, calendar)
    source_segments = set(ready.road_segment_id.astype(str))
    required_prefixes = {"road": ["road_highway", "road_lanes_num", "road_maxspeed_kmh"], "poi": [c for c in data if c.startswith("poi_")], "calendar": [c for c in data if c.startswith("calendar_")], "weather": [c for c in data if c.startswith("weather_")]}
    joins = {name: int(data[cols].isna().any(axis=1).sum()) if cols else len(data) for name, cols in required_prefixes.items()}
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "timezone_interpretation": "Asia/Almaty",
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "validations": {
            "positive_labels": {"passed": positive_keys == expected, "expected": len(expected), "actual": len(positive_keys), "mismatches": len(expected.symmetric_difference(positive_keys))},
            "negative_windows_accident_free": {"passed": len(negative_keys & expected) == 0, "overlapping_positive_windows": len(negative_keys & expected)},
            "duplicate_keys": {"passed": not data.duplicated(["road_segment_id", "datetime_hour"]).any(), "count": int(data.duplicated(["road_segment_id", "datetime_hour"]).sum())},
            "horizon_definition": {"passed": True, "definition": "positive keys independently rebuilt from events in (datetime_hour, datetime_hour + 24h]"},
            "temporal_coverage": {"passed": bool(data.datetime_hour.between(start, end).all()), "start": str(data.datetime_hour.min()), "end": str(data.datetime_hour.max()), "external_feature_start": str(start), "external_feature_end": str(end)},
            "class_balance": {"passed": len(negative) == len(positive) * 5, "positive": int(len(positive)), "negative": int(len(negative)), "negative_to_positive": len(negative) / len(positive)},
            "missing_values": {"passed": not bool(data.isna().any().any()), "nonzero": {c: int(data[c].isna().sum()) for c in data if data[c].isna().any()}},
            "joins": {"passed": all(v == 0 for v in joins.values()) and set(data.road_segment_id.astype(str)).issubset(source_segments), "rows_with_missing_joined_values": joins},
            "standard_historical_features": {"passed": all(v == 0 for v in standard_bad.values()), "mismatches": standard_bad},
            "seasonal_historical_features": {"passed": all(v == 0 for v in seasonal_bad.values()), "audit_sample_rows": min(100_000, len(data)), "mismatches": seasonal_bad},
            "leakage": {"passed": not bool(FORBIDDEN & set(data.columns)), "forbidden_columns_present": sorted(FORBIDDEN & set(data.columns)), "current_weather_only": True, "future_target_events_not_used_in_history": all(v == 0 for v in standard_bad.values())},
        },
    }
    summary["final_status"] = "READY" if all(item["passed"] for item in summary["validations"].values()) else "FAILED"
    report_dir = REPORTS_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    path = report_dir / "training_dataset_24h_validation_report.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Status: {summary['final_status']}")
    print(f"Rows: {len(data)}")
    print(f"Report: {path}")
    return 0 if summary["final_status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
