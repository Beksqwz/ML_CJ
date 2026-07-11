"""Create the leakage-safe stage-6 road-segment / hourly training dataset.

Each row represents information available at ``datetime_hour`` for one road
segment. ``target_1h`` is one only when an accident occurs on that segment in
the *following* hour. Historical counters use only events strictly before the
row hour.  Timestamps in this project are timezone-naive local civil time and
are explicitly interpreted as Asia/Almaty.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
EXTERNAL = PROJECT_ROOT / "data" / "external"
DEFAULT_READY = PROCESSED / "accidents_with_roads_ml_ready.parquet"
DEFAULT_POI = PROCESSED / "road_segment_poi_features.parquet"
DEFAULT_CALENDAR = EXTERNAL / "calendar_features_hourly.parquet"
DEFAULT_WEATHER = EXTERNAL / "weather_astana_hourly.parquet"
DEFAULT_OUTPUT = PROCESSED / "training_dataset_1h.parquet"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "stage6" / "training_dataset"
TIMEZONE = "Asia/Almaty"
RANDOM_SEED = 20260711

ROAD_FEATURES = [
    "road_highway",
    "road_lanes_num",
    "road_maxspeed_kmh",
    "road_length",
    "road_oneway",
    "road_lanes_missing",
    "road_maxspeed_missing",
    "road_name_missing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the stage-6 1-hour road-segment dataset without model training.")
    parser.add_argument("--ready", type=Path, default=DEFAULT_READY)
    parser.add_argument("--poi-features", type=Path, default=DEFAULT_POI)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def static_segment_features(ready: pd.DataFrame) -> pd.DataFrame:
    required = {"road_segment_id", *ROAD_FEATURES}
    missing = required - set(ready.columns)
    if missing:
        raise ValueError(f"Ready dataset is missing road fields: {sorted(missing)}")
    static = ready[["road_segment_id", *ROAD_FEATURES]].copy()
    static["road_segment_id"] = static["road_segment_id"].astype(str)
    # Road attributes are static per segment; retain a deterministic first value.
    static = static.sort_values("road_segment_id").drop_duplicates("road_segment_id", keep="first")
    static["road_highway"] = static["road_highway"].fillna("unknown").astype("string")
    return static.reset_index(drop=True)


def positive_observations(ready: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = ready[["road_segment_id", "accident_datetime"]].copy()
    events["road_segment_id"] = events["road_segment_id"].astype(str)
    event_time = pd.to_datetime(events["accident_datetime"], errors="raise")
    if getattr(event_time.dt, "tz", None) is not None:
        raise ValueError("accident_datetime must be timezone-naive Asia/Almaty local time.")
    events["event_datetime_hour"] = event_time.dt.floor("h")
    event_counts = events.groupby(["road_segment_id", "event_datetime_hour"], as_index=False).size().rename(columns={"size": "event_count"})
    positives = event_counts[["road_segment_id", "event_datetime_hour"]].rename(columns={"event_datetime_hour": "target_datetime_hour"})
    positives["datetime_hour"] = positives["target_datetime_hour"] - pd.Timedelta(hours=1)
    positives["target_1h"] = np.int8(1)
    positives = positives[["road_segment_id", "datetime_hour", "target_datetime_hour", "target_1h"]]
    if positives.duplicated(["road_segment_id", "datetime_hour"]).any():
        raise AssertionError("Positive segment-hour keys are not unique.")
    return positives, event_counts


def sample_negatives(
    positives: pd.DataFrame, static: pd.DataFrame, ratio: int, seed: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    if ratio < 1:
        raise ValueError("negative-ratio must be at least 1.")
    rng = np.random.default_rng(seed)
    segment_highway = static.set_index("road_segment_id")["road_highway"].astype(str).to_dict()
    all_segments = np.array(sorted(segment_highway), dtype=object)
    by_highway: dict[str, np.ndarray] = {}
    for highway, group in static.groupby("road_highway", dropna=False):
        by_highway[str(highway)] = group["road_segment_id"].astype(str).to_numpy()

    positives = positives.copy()
    positives["_highway"] = positives["road_segment_id"].map(segment_highway).fillna("unknown").astype(str)
    positive_keys = set(zip(positives["road_segment_id"], positives["datetime_hour"]))
    negative_keys: set[tuple[str, pd.Timestamp]] = set()
    rows: list[dict[str, object]] = []
    fallback_draws = 0
    for positive in positives.to_dict("records"):
        primary = by_highway.get(str(positive["_highway"]), all_segments)
        for _ in range(ratio):
            selected: str | None = None
            for candidates, is_fallback in ((primary, False), (all_segments, True)):
                if len(candidates) == 0:
                    continue
                for _attempt in range(100):
                    candidate = str(candidates[rng.integers(len(candidates))])
                    key = (candidate, positive["datetime_hour"])
                    if key not in positive_keys and key not in negative_keys:
                        selected = candidate
                        if is_fallback:
                            fallback_draws += 1
                        break
                if selected is not None:
                    break
            if selected is None:
                raise RuntimeError(f"Unable to draw a non-overlapping negative for {positive['datetime_hour']}.")
            negative_keys.add((selected, positive["datetime_hour"]))
            rows.append(
                {
                    "road_segment_id": selected,
                    "datetime_hour": positive["datetime_hour"],
                    "target_datetime_hour": positive["target_datetime_hour"],
                    "target_1h": np.int8(0),
                }
            )
    negatives = pd.DataFrame(rows)
    if len(negatives) != len(positives) * ratio:
        raise AssertionError("Negative sampling did not preserve the requested ratio.")
    return negatives, {"fallback_draws": fallback_draws, "negative_rows": int(len(negatives))}


def prior_features(observations: pd.DataFrame, event_counts: pd.DataFrame) -> pd.DataFrame:
    """Add counts strictly before each row hour, never including the target hour."""
    result = observations.copy()
    result["segment_accidents_total_prior"] = 0
    result["segment_accidents_prev_24h"] = 0
    result["segment_accidents_prev_7d"] = 0
    result["segment_accidents_prev_30d"] = 0
    result["segment_hours_since_prev_accident"] = -1.0
    result["segment_has_history"] = False

    for segment, indices in result.groupby("road_segment_id").groups.items():
        event_group = event_counts.loc[event_counts["road_segment_id"] == segment].sort_values("event_datetime_hour")
        if event_group.empty:
            continue
        event_times = event_group["event_datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
        cumulative = event_group["event_count"].cumsum().to_numpy(dtype=np.int64)
        query_indices = np.asarray(list(indices))
        query_times = result.loc[query_indices, "datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()

        def count_before(times: np.ndarray) -> np.ndarray:
            positions = np.searchsorted(event_times, times, side="left")
            return np.where(positions > 0, cumulative[positions - 1], 0)

        total = count_before(query_times)
        result.loc[query_indices, "segment_accidents_total_prior"] = total
        for label, hours in (("segment_accidents_prev_24h", 24), ("segment_accidents_prev_7d", 168), ("segment_accidents_prev_30d", 720)):
            window_ns = np.int64(hours * 3_600_000_000_000)
            result.loc[query_indices, label] = total - count_before(query_times - window_ns)
        positions = np.searchsorted(event_times, query_times, side="left")
        has_history = positions > 0
        hours_since = np.full(len(query_times), -1.0)
        hours_since[has_history] = (query_times[has_history] - event_times[positions[has_history] - 1]) / 3_600_000_000_000
        result.loc[query_indices, "segment_hours_since_prev_accident"] = hours_since
        result.loc[query_indices, "segment_has_history"] = has_history

    city = event_counts.groupby("event_datetime_hour", as_index=False)["event_count"].sum().sort_values("event_datetime_hour")
    event_times = city["event_datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()
    cumulative = city["event_count"].cumsum().to_numpy(dtype=np.int64)
    query_times = result["datetime_hour"].astype("datetime64[ns]").astype("int64").to_numpy()

    def city_count_before(times: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(event_times, times, side="left")
        return np.where(positions > 0, cumulative[positions - 1], 0)

    city_total = city_count_before(query_times)
    result["city_accidents_total_prior"] = city_total
    for label, hours in (("city_accidents_prev_24h", 24), ("city_accidents_prev_7d", 168), ("city_accidents_prev_30d", 720)):
        result[label] = city_total - city_count_before(query_times - np.int64(hours * 3_600_000_000_000))
    return result


def prefix_columns(data: pd.DataFrame, prefix: str, excluded: set[str]) -> pd.DataFrame:
    return data.rename(columns={column: f"{prefix}{column}" for column in data.columns if column not in excluded})


def build_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    ready = pd.read_parquet(args.ready)
    static = static_segment_features(ready)
    positives, event_counts = positive_observations(ready)
    negatives, sampling_summary = sample_negatives(positives, static, args.negative_ratio, args.seed)
    observations = pd.concat([positives, negatives], ignore_index=True).sort_values(["datetime_hour", "road_segment_id"]).reset_index(drop=True)
    observations = observations.merge(static, on="road_segment_id", how="left", validate="many_to_one", indicator="_road_join")
    road_missing = int((observations.pop("_road_join") != "both").sum())

    poi = pd.read_parquet(args.poi_features)
    observations = observations.merge(poi, on="road_segment_id", how="left", validate="many_to_one", indicator="_poi_join")
    poi_missing = int((observations.pop("_poi_join") != "both").sum())

    calendar = pd.read_parquet(args.calendar)
    calendar["datetime_hour"] = pd.to_datetime(calendar["datetime_hour"], errors="raise")
    calendar = prefix_columns(calendar, "calendar_", {"datetime_hour"})
    observations = observations.merge(calendar, on="datetime_hour", how="left", validate="many_to_one", indicator="_calendar_join")
    calendar_missing = int((observations.pop("_calendar_join") != "both").sum())

    weather = pd.read_parquet(args.weather)
    weather["datetime_hour"] = pd.to_datetime(weather["datetime_hour"], errors="raise")
    weather = weather.drop(columns=[column for column in ("latitude_source", "longitude_source", "timezone_source") if column in weather])
    weather = prefix_columns(weather, "weather_", {"datetime_hour"})
    observations = observations.merge(weather, on="datetime_hour", how="left", validate="many_to_one", indicator="_weather_join")
    weather_missing = int((observations.pop("_weather_join") != "both").sum())
    observations = prior_features(observations, event_counts)
    observations = observations.sort_values(["datetime_hour", "road_segment_id"]).reset_index(drop=True)

    join_summary = {"road_unmatched": road_missing, "poi_unmatched": poi_missing, "calendar_unmatched": calendar_missing, "weather_unmatched": weather_missing}
    summary: dict[str, object] = {
        "input_rows": int(len(ready)),
        "positive_rows": int(len(positives)),
        "negative_rows": int(len(negatives)),
        "negative_ratio": args.negative_ratio,
        "sampling": sampling_summary,
        "join": join_summary,
        "timezone_interpretation": TIMEZONE,
        "history_policy": "All historical counts use event_datetime_hour < datetime_hour; target hour and future events are excluded.",
    }
    return observations, summary


def write_report(dataset: pd.DataFrame, summary: dict[str, object], output: Path) -> Path:
    report_dir = REPORTS_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    report = summary | {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output": str(output.resolve()),
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "target_balance": {str(key): int(value) for key, value in dataset["target_1h"].value_counts().sort_index().items()},
        "temporal_coverage": {"start": str(dataset["datetime_hour"].min()), "end": str(dataset["datetime_hour"].max())},
        "duplicate_segment_hour_keys": int(dataset.duplicated(["road_segment_id", "datetime_hour"]).sum()),
        "missing_values": {column: int(dataset[column].isna().sum()) for column in dataset.columns},
        "leakage_checks": {
            "target_is_next_hour": True,
            "history_uses_strictly_prior_events": True,
            "excluded_post_accident_fields": ["type_dtp", "fd1r17", "fd1r17_descrip", "distance_to_road_m"],
            "weather_and_calendar_timestamp": "datetime_hour (observation hour, not future target hour)",
        },
    }
    report_path = report_dir / "training_dataset_1h_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    dataset, summary = build_dataset(args)
    if dataset.duplicated(["road_segment_id", "datetime_hour"]).any():
        raise AssertionError("Training dataset has duplicate segment-hour keys.")
    if set(dataset["target_1h"].unique()) != {0, 1}:
        raise AssertionError("target_1h must contain both zero and one.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False, engine="pyarrow")
    report_path = write_report(dataset, summary, args.output)
    print(f"Rows: {len(dataset)}")
    print(f"Target balance: {dataset['target_1h'].value_counts().sort_index().to_dict()}")
    print(f"Output: {args.output.resolve()}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
