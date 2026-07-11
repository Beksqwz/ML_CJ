"""Build a leakage-safe 24-hour road-segment accident forecasting dataset.

Rows are observations available at local Asia/Almaty ``datetime_hour``.
``target_24h`` is one only if an accident occurs strictly after that hour and
not later than 24 hours after it.  No target-window events or future weather
are used as features.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_training_dataset_1h import ROAD_FEATURES, TIMEZONE, prefix_columns, static_segment_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
EXTERNAL = PROJECT_ROOT / "data" / "external"
DEFAULT_READY = PROCESSED / "accidents_with_roads_ml_ready.parquet"
DEFAULT_POI = PROCESSED / "road_segment_poi_features.parquet"
DEFAULT_CALENDAR = EXTERNAL / "calendar_features_hourly.parquet"
DEFAULT_WEATHER = EXTERNAL / "weather_astana_hourly.parquet"
DEFAULT_OUTPUT = PROCESSED / "training_dataset_24h.parquet"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "stage6" / "training_dataset_24h"
RANDOM_SEED = 20260711
HORIZON_HOURS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage-6 24-hour training data without model training.")
    parser.add_argument("--ready", type=Path, default=DEFAULT_READY)
    parser.add_argument("--poi-features", type=Path, default=DEFAULT_POI)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def event_counts(ready: pd.DataFrame) -> pd.DataFrame:
    events = ready[["road_segment_id", "accident_datetime"]].copy()
    events["road_segment_id"] = events["road_segment_id"].astype(str)
    timestamps = pd.to_datetime(events["accident_datetime"], errors="raise")
    if getattr(timestamps.dt, "tz", None) is not None:
        raise ValueError("accident_datetime must be timezone-naive local Asia/Almaty time.")
    events["event_hour"] = timestamps.dt.floor("h")
    return events.groupby(["road_segment_id", "event_hour"], as_index=False).size().rename(columns={"size": "event_count"})


def make_positive_windows(counts: pd.DataFrame, feature_start: pd.Timestamp, feature_end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    pieces = []
    for offset in range(1, HORIZON_HOURS + 1):
        part = counts[["road_segment_id", "event_hour"]].copy()
        part["datetime_hour"] = part["event_hour"] - pd.Timedelta(hours=offset)
        pieces.append(part[["road_segment_id", "datetime_hour"]])
    all_positive = pd.concat(pieces, ignore_index=True).drop_duplicates(["road_segment_id", "datetime_hour"])
    eligible = all_positive.loc[all_positive["datetime_hour"].between(feature_start, feature_end)].copy()
    excluded = int(len(all_positive) - len(eligible))
    eligible["target_24h"] = np.int8(1)
    return eligible, excluded


def sample_negatives(positives: pd.DataFrame, static: pd.DataFrame, ratio: int, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    if ratio < 1:
        raise ValueError("negative-ratio must be at least one.")
    rng = np.random.default_rng(seed)
    highway_by_segment = static.set_index("road_segment_id")["road_highway"].astype(str).to_dict()
    all_segments = np.array(sorted(highway_by_segment), dtype=object)
    groups = {str(key): value["road_segment_id"].astype(str).to_numpy() for key, value in static.groupby("road_highway", dropna=False)}
    positive_keys = set(zip(positives["road_segment_id"].astype(str), positives["datetime_hour"]))
    negative_keys: set[tuple[str, pd.Timestamp]] = set()
    rows: list[tuple[str, pd.Timestamp, np.int8]] = []
    fallback = 0
    for item in positives[["road_segment_id", "datetime_hour"]].itertuples(index=False):
        road_segment_id, timestamp = str(item[0]), item[1]
        candidates_primary = groups.get(str(highway_by_segment.get(road_segment_id, "unknown")), all_segments)
        for _ in range(ratio):
            picked: str | None = None
            for candidates, uses_fallback in ((candidates_primary, False), (all_segments, True)):
                for _attempt in range(100):
                    candidate = str(candidates[rng.integers(len(candidates))])
                    key = (candidate, timestamp)
                    if key not in positive_keys and key not in negative_keys:
                        picked = candidate
                        fallback += int(uses_fallback)
                        break
                if picked is not None:
                    break
            if picked is None:
                raise RuntimeError(f"Cannot sample a negative at {timestamp} without positive-window overlap.")
            negative_keys.add((picked, timestamp))
            rows.append((picked, timestamp, np.int8(0)))
    negatives = pd.DataFrame(rows, columns=["road_segment_id", "datetime_hour", "target_24h"])
    return negatives, {"fallback_draws": fallback, "negative_rows": int(len(negatives))}


def add_standard_history(data: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    defaults = {"segment_accidents_total_prior": 0, "segment_accidents_prev_24h": 0, "segment_accidents_prev_7d": 0, "segment_accidents_prev_30d": 0, "segment_hours_since_prev_accident": -1.0, "segment_has_history": False}
    for column, value in defaults.items():
        result[column] = value
    for segment, indices in result.groupby("road_segment_id").groups.items():
        events = counts.loc[counts["road_segment_id"] == segment].sort_values("event_hour")
        if events.empty:
            continue
        event_ns = events["event_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
        cumulative = events["event_count"].cumsum().to_numpy(dtype=np.int64)
        row_indices = np.asarray(list(indices))
        query_ns = result.loc[row_indices, "datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
        def prior(query: np.ndarray) -> np.ndarray:
            location = np.searchsorted(event_ns, query, side="left")
            return np.where(location > 0, cumulative[location - 1], 0)
        total = prior(query_ns)
        result.loc[row_indices, "segment_accidents_total_prior"] = total
        for label, hours in (("segment_accidents_prev_24h", 24), ("segment_accidents_prev_7d", 168), ("segment_accidents_prev_30d", 720)):
            result.loc[row_indices, label] = total - prior(query_ns - np.int64(hours * 3_600_000_000_000))
        location = np.searchsorted(event_ns, query_ns, side="left")
        has_history = location > 0
        hours_since = np.full(len(query_ns), -1.0)
        hours_since[has_history] = (query_ns[has_history] - event_ns[location[has_history] - 1]) / 3_600_000_000_000
        result.loc[row_indices, "segment_hours_since_prev_accident"] = hours_since
        result.loc[row_indices, "segment_has_history"] = has_history
    city = counts.groupby("event_hour", as_index=False)["event_count"].sum().sort_values("event_hour")
    event_ns = city["event_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
    cumulative = city["event_count"].cumsum().to_numpy(dtype=np.int64)
    query_ns = result["datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
    def city_prior(query: np.ndarray) -> np.ndarray:
        location = np.searchsorted(event_ns, query, side="left")
        return np.where(location > 0, cumulative[location - 1], 0)
    total = city_prior(query_ns)
    result["city_accidents_total_prior"] = total
    for label, hours in (("city_accidents_prev_24h", 24), ("city_accidents_prev_7d", 168), ("city_accidents_prev_30d", 720)):
        result[label] = total - city_prior(query_ns - np.int64(hours * 3_600_000_000_000))
    return result


def cumulative_context_feature(data: pd.DataFrame, events: pd.DataFrame, query_context: list[str], event_context: list[str], output: str) -> pd.Series:
    """Count context-matching events strictly before each row timestamp."""
    left = data[["road_segment_id", "datetime_hour", *query_context]].copy()
    left["_row_id"] = np.arange(len(left))
    right = events[["road_segment_id", "event_hour", "event_count", *event_context]].copy()
    right = right.groupby(["road_segment_id", "event_hour", *event_context], as_index=False)["event_count"].sum()
    group_right = ["road_segment_id", *event_context]
    left.columns = ["road_segment_id", "datetime_hour", *group_right[1:], "_row_id"]
    right = right.sort_values(["event_hour", *group_right])
    right["_cumulative"] = right.groupby(group_right, dropna=False)["event_count"].cumsum()
    left_sorted = left.sort_values(["datetime_hour", *group_right])
    merged = pd.merge_asof(left_sorted, right[["event_hour", *group_right, "_cumulative"]], left_on="datetime_hour", right_on="event_hour", by=group_right, direction="backward", allow_exact_matches=False)
    values = pd.Series(merged["_cumulative"].fillna(0).to_numpy(dtype=np.int64), index=merged["_row_id"].to_numpy(), name=output)
    return values.reindex(np.arange(len(data)), fill_value=0)


def add_seasonal_history(data: pd.DataFrame, counts: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    calendar_context = calendar[["datetime_hour", "is_holiday"]].copy()
    calendar_context["is_new_year_period"] = ((calendar_context["datetime_hour"].dt.month == 12) & (calendar_context["datetime_hour"].dt.day >= 25)) | ((calendar_context["datetime_hour"].dt.month == 1) & (calendar_context["datetime_hour"].dt.day <= 7))
    event_context = counts.merge(calendar_context, left_on="event_hour", right_on="datetime_hour", how="left", validate="many_to_one")
    event_context["event_month"] = event_context["event_hour"].dt.month.astype("int8")
    event_context["event_hour_of_day"] = event_context["event_hour"].dt.hour.astype("int8")
    event_context["event_weekday"] = event_context["event_hour"].dt.weekday.astype("int8")
    result["_month"] = result["datetime_hour"].dt.month.astype("int8")
    result["_hour_of_day"] = result["datetime_hour"].dt.hour.astype("int8")
    result["_weekday"] = result["datetime_hour"].dt.weekday.astype("int8")
    result["segment_accidents_same_month_prior"] = cumulative_context_feature(result, event_context, ["_month"], ["event_month"], "segment_accidents_same_month_prior").to_numpy()
    result["segment_accidents_same_hour_weekday_prior"] = cumulative_context_feature(result, event_context, ["_hour_of_day", "_weekday"], ["event_hour_of_day", "event_weekday"], "segment_accidents_same_hour_weekday_prior").to_numpy()
    result["segment_accidents_same_month_hour_weekday_prior"] = cumulative_context_feature(result, event_context, ["_month", "_hour_of_day", "_weekday"], ["event_month", "event_hour_of_day", "event_weekday"], "segment_accidents_same_month_hour_weekday_prior").to_numpy()
    holiday_events = event_context.loc[event_context["is_holiday"].fillna(False)].copy()
    new_year_events = event_context.loc[event_context["is_new_year_period"].fillna(False)].copy()
    # A constant context is used so these are prior counts of holiday/new-year accidents only.
    result["_all_context"] = 1
    holiday_events["_all_context"] = 1
    new_year_events["_all_context"] = 1
    result["segment_accidents_holiday_prior"] = cumulative_context_feature(result, holiday_events, ["_all_context"], ["_all_context"], "segment_accidents_holiday_prior").to_numpy()
    result["segment_accidents_new_year_period_prior"] = cumulative_context_feature(result, new_year_events, ["_all_context"], ["_all_context"], "segment_accidents_new_year_period_prior").to_numpy()
    return result.drop(columns=["_month", "_hour_of_day", "_weekday", "_all_context"])


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    ready = pd.read_parquet(args.ready)
    static = static_segment_features(ready)
    counts = event_counts(ready)
    calendar = pd.read_parquet(args.calendar)
    weather = pd.read_parquet(args.weather)
    calendar["datetime_hour"] = pd.to_datetime(calendar["datetime_hour"], errors="raise")
    weather["datetime_hour"] = pd.to_datetime(weather["datetime_hour"], errors="raise")
    feature_start = max(calendar["datetime_hour"].min(), weather["datetime_hour"].min())
    feature_end = min(calendar["datetime_hour"].max(), weather["datetime_hour"].max())
    positives, outside_coverage = make_positive_windows(counts, feature_start, feature_end)
    negatives, sampling = sample_negatives(positives, static, args.negative_ratio, args.seed)
    data = pd.concat([positives, negatives], ignore_index=True).sort_values(["datetime_hour", "road_segment_id"]).reset_index(drop=True)
    data = data.merge(static, on="road_segment_id", how="left", validate="many_to_one", indicator="_road_join")
    road_unmatched = int((data.pop("_road_join") != "both").sum())
    poi = pd.read_parquet(args.poi_features)
    data = data.merge(poi, on="road_segment_id", how="left", validate="many_to_one", indicator="_poi_join")
    poi_unmatched = int((data.pop("_poi_join") != "both").sum())
    calendar["is_new_year_period"] = ((calendar["datetime_hour"].dt.month == 12) & (calendar["datetime_hour"].dt.day >= 25)) | ((calendar["datetime_hour"].dt.month == 1) & (calendar["datetime_hour"].dt.day <= 7))
    calendar_for_features = prefix_columns(calendar, "calendar_", {"datetime_hour"})
    data = data.merge(calendar_for_features, on="datetime_hour", how="left", validate="many_to_one", indicator="_calendar_join")
    calendar_unmatched = int((data.pop("_calendar_join") != "both").sum())
    weather = weather.drop(columns=[column for column in ("latitude_source", "longitude_source", "timezone_source") if column in weather])
    weather_for_features = prefix_columns(weather, "weather_", {"datetime_hour"})
    data = data.merge(weather_for_features, on="datetime_hour", how="left", validate="many_to_one", indicator="_weather_join")
    weather_unmatched = int((data.pop("_weather_join") != "both").sum())
    data = add_standard_history(data, counts)
    data = add_seasonal_history(data, counts, calendar)
    data = data.sort_values(["datetime_hour", "road_segment_id"]).reset_index(drop=True)
    summary: dict[str, object] = {
        "timezone_interpretation": TIMEZONE,
        "horizon_hours": HORIZON_HOURS,
        "feature_time_coverage": {"start": str(feature_start), "end": str(feature_end)},
        "positive_windows_excluded_outside_calendar_weather_coverage": outside_coverage,
        "positive_rows": int((data["target_24h"] == 1).sum()),
        "negative_rows": int((data["target_24h"] == 0).sum()),
        "negative_ratio": args.negative_ratio,
        "sampling": sampling,
        "joins": {"road_unmatched": road_unmatched, "poi_unmatched": poi_unmatched, "calendar_unmatched": calendar_unmatched, "weather_unmatched": weather_unmatched},
        "history_policy": "All standard and seasonal accident features use event_hour < datetime_hour. Current-hour weather/calendar only are joined.",
    }
    return data, summary


def write_report(data: pd.DataFrame, summary: dict[str, object], output: Path) -> Path:
    report_dir = REPORTS_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    payload = summary | {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output": str(output.resolve()),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "target_balance": {str(key): int(value) for key, value in data["target_24h"].value_counts().sort_index().items()},
        "temporal_coverage": {"start": str(data["datetime_hour"].min()), "end": str(data["datetime_hour"].max())},
        "duplicate_segment_hour_keys": int(data.duplicated(["road_segment_id", "datetime_hour"]).sum()),
        "missing_values": {column: int(data[column].isna().sum()) for column in data.columns},
        "leakage_policy": {"post_event_fields_excluded": ["type_dtp", "fd1r17", "fd1r17_descrip", "distance_to_road_m"], "future_weather_excluded": True, "target_window_events_excluded_from_history": True},
    }
    path = report_dir / "training_dataset_24h_build_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    data, summary = build(args)
    if data.duplicated(["road_segment_id", "datetime_hour"]).any():
        raise AssertionError("Duplicate road-segment/hour keys.")
    if len(data.loc[data["target_24h"] == 0]) != len(data.loc[data["target_24h"] == 1]) * args.negative_ratio:
        raise AssertionError("Negative ratio was not preserved.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output, index=False, engine="pyarrow")
    report = write_report(data, summary, args.output)
    print(f"Rows: {len(data)}")
    print(f"Target balance: {data['target_24h'].value_counts().sort_index().to_dict()}")
    print(f"Output: {args.output.resolve()}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
