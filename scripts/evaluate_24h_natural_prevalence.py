"""Audit-only natural-prevalence evaluation of the frozen Stage 7B model."""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import tempfile
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts.build_training_dataset_1h import static_segment_features
from scripts.build_training_dataset_24h import (
    add_seasonal_history,
    add_standard_history,
    event_counts,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_HASH = "bf0ce06aface9fc3539a95df2557728e57d987a9e0c0e5ac4a195a53b47f6d96"
PROCESSED = ROOT / "data" / "processed"
EXTERNAL = ROOT / "data" / "external"
CONFIG_PATH = (
    ROOT
    / "reports"
    / "stage7a"
    / "24h"
    / "20260711T090515Z"
    / "training_dataset_24h_feature_config.json"
)
REPORTS = ROOT / "reports" / "stage19e"
RUN_STATE: dict[str, object] = {"last_stage": "module_loaded", "last_chunk": None}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temp = Path(handle.name)
    os.replace(temp, path)


def progress(stage: str, chunk: int | None = None) -> None:
    RUN_STATE.update(last_stage=stage, last_chunk=chunk)
    print(json.dumps({"stage": stage, "chunk": chunk}, default=str), flush=True)


def finish_diagnostics(status: str, exc: BaseException | None = None) -> None:
    path = RUN_STATE.get("diagnostics_path")
    if not path:
        return
    payload = {
        **RUN_STATE,
        "pid": os.getpid(),
        "finished_at": datetime.now(UTC).isoformat(),
        "exit_status": status,
        "exception_type": type(exc).__name__ if exc else None,
        "exception_message": str(exc) if exc else None,
        "traceback": traceback.format_exc() if exc else None,
    }
    atomic_json(Path(str(path)), payload)
    RUN_STATE["finalized"] = True


@atexit.register
def abnormal_exit_diagnostic() -> None:
    if RUN_STATE.get("diagnostics_path") and not RUN_STATE.get("finalized"):
        finish_diagnostics("abnormal_exit")


def feature_hash(features: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(features, separators=(",", ":")).encode()
    ).hexdigest()


def target_fields(grid: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    output = grid.copy()
    output["target_24h"] = np.int8(0)
    output["target_accident_count"] = 0
    output["target_first_accident_datetime"] = pd.NaT
    output["target_last_accident_datetime"] = pd.NaT
    for segment, index in output.groupby("road_segment_id").groups.items():
        events = counts.loc[counts.road_segment_id.eq(segment)].sort_values(
            "event_hour"
        )
        if events.empty:
            continue
        times = events.event_hour.to_numpy(dtype="datetime64[ns]")
        values = events.event_count.to_numpy(dtype=np.int64)
        cumulative = values.cumsum()
        query = output.loc[index, "prediction_datetime"].to_numpy(
            dtype="datetime64[ns]"
        )
        left = np.searchsorted(times, query, side="right")
        right = np.searchsorted(times, query + np.timedelta64(24, "h"), side="right")
        counts_ = np.where(
            right > left,
            cumulative[right - 1] - np.where(left > 0, cumulative[left - 1], 0),
            0,
        )
        output.loc[index, "target_accident_count"] = counts_
        output.loc[index, "target_24h"] = (counts_ > 0).astype(np.int8)
        safe_left = np.minimum(left, len(times) - 1)
        safe_right = np.maximum(0, np.minimum(right - 1, len(times) - 1))
        first = np.where(left < right, times[safe_left], np.datetime64("NaT"))
        last = np.where(left < right, times[safe_right], np.datetime64("NaT"))
        output.loc[index, "target_first_accident_datetime"] = first
        output.loc[index, "target_last_accident_datetime"] = last
    return output


def _cumulative_index(
    events: pd.DataFrame, keys: list[str]
) -> dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]]:
    """Build immutable, strictly-prior lookup arrays for causal history features."""
    indexes: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
    grouped = events.groupby(keys, dropna=False, sort=False)
    for key, group in grouped:
        normalized_key = key if isinstance(key, tuple) else (key,)
        ordered = group.sort_values("event_hour")
        timestamps = ordered["event_hour"].astype("datetime64[ns]").astype("int64")
        indexes[normalized_key] = (
            timestamps.to_numpy(),
            ordered["event_count"].cumsum().to_numpy(dtype=np.int64),
        )
    return indexes


def _prior_from_index(
    timestamps: np.ndarray, cumulative: np.ndarray, query: np.ndarray
) -> np.ndarray:
    locations = np.searchsorted(timestamps, query, side="left")
    return np.where(locations > 0, cumulative[locations - 1], 0)


def _indexed_context_feature(
    result: pd.DataFrame,
    indexes: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]],
    key_columns: list[str],
) -> np.ndarray:
    values = np.zeros(len(result), dtype=np.int64)
    query_ns = result["datetime_hour"].astype("datetime64[ns]").astype("int64")
    for key, row_index in result.groupby(
        key_columns, dropna=False, sort=False
    ).groups.items():
        normalized_key = key if isinstance(key, tuple) else (key,)
        event_index = indexes.get(normalized_key)
        if event_index is None:
            continue
        rows = np.asarray(list(row_index))
        times, cumulative = event_index
        values[rows] = _prior_from_index(
            times, cumulative, query_ns.iloc[rows].to_numpy()
        )
    return values


@dataclass(frozen=True)
class EvaluationFeatureContext:
    """Read-once inputs and compact causal-history indexes for Stage 19E chunks."""

    static_lookup: pd.DataFrame
    calendar_lookup: pd.DataFrame
    weather_lookup: pd.DataFrame
    counts: pd.DataFrame
    segment_history: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]]
    city_history: tuple[np.ndarray, np.ndarray]
    seasonal_indexes: dict[str, dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]]]
    rare_highways: frozenset[str]
    feature_order: tuple[str, ...]
    categorical_features: tuple[str, ...]
    segment_order: tuple[str, ...]
    hour_index: dict[pd.Timestamp, int]
    segment_index: dict[str, int]
    history_values: dict[str, np.ndarray]


def _precompute_history_values(
    segment_order: tuple[str, ...],
    hours: pd.DatetimeIndex,
    segment_history: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]],
    city_history: tuple[np.ndarray, np.ndarray],
    seasonal_indexes: dict[
        str, dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]]
    ],
) -> dict[str, np.ndarray]:
    """Compact causal [hour, segment] history arrays, not a model feature grid."""
    shape = (len(hours), len(segment_order))
    query_ns = hours.astype("datetime64[ns]").astype("int64").to_numpy()
    values: dict[str, np.ndarray] = {
        name: np.zeros(shape, dtype=np.int64)
        for name in (
            "segment_accidents_total_prior",
            "segment_accidents_prev_24h",
            "segment_accidents_prev_7d",
            "segment_accidents_prev_30d",
            "segment_accidents_same_month_prior",
            "segment_accidents_same_hour_weekday_prior",
            "segment_accidents_same_month_hour_weekday_prior",
            "segment_accidents_holiday_prior",
            "segment_accidents_new_year_period_prior",
        )
    }
    values["segment_hours_since_prev_accident"] = np.full(shape, -1.0)
    values["segment_has_history"] = np.zeros(shape, dtype=bool)
    city_times, city_cumulative = city_history
    city_total = _prior_from_index(city_times, city_cumulative, query_ns)
    for name, vector in (
        ("city_accidents_total_prior", city_total),
        (
            "city_accidents_prev_24h",
            city_total
            - _prior_from_index(
                city_times, city_cumulative, query_ns - 24 * 3_600_000_000_000
            ),
        ),
        (
            "city_accidents_prev_7d",
            city_total
            - _prior_from_index(
                city_times, city_cumulative, query_ns - 168 * 3_600_000_000_000
            ),
        ),
        (
            "city_accidents_prev_30d",
            city_total
            - _prior_from_index(
                city_times, city_cumulative, query_ns - 720 * 3_600_000_000_000
            ),
        ),
    ):
        values[name] = np.broadcast_to(vector[:, None], shape).copy()
    month = hours.month.to_numpy(dtype=np.int8)
    hour_of_day = hours.hour.to_numpy(dtype=np.int8)
    weekday = hours.weekday.to_numpy(dtype=np.int8)
    specs = (
        (
            "segment_accidents_same_month_prior",
            "month",
            lambda i, s: (s, int(month[i])),
        ),
        (
            "segment_accidents_same_hour_weekday_prior",
            "hour_weekday",
            lambda i, s: (s, int(hour_of_day[i]), int(weekday[i])),
        ),
        (
            "segment_accidents_same_month_hour_weekday_prior",
            "month_hour_weekday",
            lambda i, s: (s, int(month[i]), int(hour_of_day[i]), int(weekday[i])),
        ),
        ("segment_accidents_holiday_prior", "holiday", lambda _i, s: (s,)),
        ("segment_accidents_new_year_period_prior", "new_year", lambda _i, s: (s,)),
    )
    for segment_position, segment in enumerate(segment_order):
        standard = segment_history.get((segment,))
        if standard is not None:
            event_times, cumulative = standard
            total = _prior_from_index(event_times, cumulative, query_ns)
            values["segment_accidents_total_prior"][:, segment_position] = total
            for label, hours_back in (
                ("segment_accidents_prev_24h", 24),
                ("segment_accidents_prev_7d", 168),
                ("segment_accidents_prev_30d", 720),
            ):
                values[label][:, segment_position] = total - _prior_from_index(
                    event_times,
                    cumulative,
                    query_ns - hours_back * 3_600_000_000_000,
                )
            locations = np.searchsorted(event_times, query_ns, side="left")
            has_history = locations > 0
            values["segment_has_history"][:, segment_position] = has_history
            values["segment_hours_since_prev_accident"][
                has_history, segment_position
            ] = (
                query_ns[has_history] - event_times[locations[has_history] - 1]
            ) / 3_600_000_000_000
        for label, lookup_name, key_builder in specs:
            lookup = seasonal_indexes[lookup_name]
            for hour_position in range(len(hours)):
                indexed_events = lookup.get(key_builder(hour_position, segment))
                if indexed_events is not None:
                    event_times, cumulative = indexed_events
                    values[label][hour_position, segment_position] = _prior_from_index(
                        event_times,
                        cumulative,
                        query_ns[hour_position : hour_position + 1],
                    )[0]
    return values


def build_feature_context(
    ready: pd.DataFrame, config: dict, evaluation_hours: pd.DatetimeIndex
) -> EvaluationFeatureContext:
    """Load every parquet source once; never read them in the chunk loop."""
    counts = event_counts(ready)
    static = static_segment_features(ready)
    poi = pd.read_parquet(PROCESSED / "road_segment_poi_features.parquet")
    static_lookup = static.merge(
        poi, on="road_segment_id", how="left", validate="one_to_one"
    )
    calendar = pd.read_parquet(EXTERNAL / "calendar_features_hourly.parquet")
    weather = pd.read_parquet(EXTERNAL / "weather_astana_hourly.parquet")
    for frame in (calendar, weather):
        frame["datetime_hour"] = pd.to_datetime(frame["datetime_hour"])
    calendar["is_new_year_period"] = (
        (calendar.datetime_hour.dt.month == 12) & (calendar.datetime_hour.dt.day >= 25)
    ) | ((calendar.datetime_hour.dt.month == 1) & (calendar.datetime_hour.dt.day <= 7))
    calendar_lookup = (
        calendar.add_prefix("calendar_")
        .rename(columns={"calendar_datetime_hour": "datetime_hour"})
        .set_index("datetime_hour", verify_integrity=True)
    )
    weather = weather.drop(
        columns=[
            column
            for column in ("latitude_source", "longitude_source", "timezone_source")
            if column in weather
        ]
    )
    weather_lookup = (
        weather.add_prefix("weather_")
        .rename(columns={"weather_datetime_hour": "datetime_hour"})
        .set_index("datetime_hour", verify_integrity=True)
    )
    calendar_context = calendar[["datetime_hour", "is_holiday", "is_new_year_period"]]
    event_context = counts.merge(
        calendar_context,
        left_on="event_hour",
        right_on="datetime_hour",
        how="left",
        validate="many_to_one",
    )
    event_context["event_month"] = event_context.event_hour.dt.month.astype("int8")
    event_context["event_hour_of_day"] = event_context.event_hour.dt.hour.astype("int8")
    event_context["event_weekday"] = event_context.event_hour.dt.weekday.astype("int8")
    city = (
        counts.groupby("event_hour", as_index=False)["event_count"]
        .sum()
        .sort_values("event_hour")
    )
    city_times = city.event_hour.astype("datetime64[ns]").astype("int64").to_numpy()
    city_cumulative = city.event_count.cumsum().to_numpy(dtype=np.int64)
    seasonal_indexes = {
        "month": _cumulative_index(event_context, ["road_segment_id", "event_month"]),
        "hour_weekday": _cumulative_index(
            event_context, ["road_segment_id", "event_hour_of_day", "event_weekday"]
        ),
        "month_hour_weekday": _cumulative_index(
            event_context,
            ["road_segment_id", "event_month", "event_hour_of_day", "event_weekday"],
        ),
        "holiday": _cumulative_index(
            event_context.loc[event_context.is_holiday.fillna(False)],
            ["road_segment_id"],
        ),
        "new_year": _cumulative_index(
            event_context.loc[event_context.is_new_year_period.fillna(False)],
            ["road_segment_id"],
        ),
    }
    segment_order = tuple(static_lookup.road_segment_id.astype(str).sort_values())
    segment_history = _cumulative_index(counts, ["road_segment_id"])
    features = tuple([*config["numerical_features"], *config["categorical_features"]])
    return EvaluationFeatureContext(
        static_lookup=static_lookup,
        calendar_lookup=calendar_lookup,
        weather_lookup=weather_lookup,
        counts=counts,
        segment_history=segment_history,
        city_history=(city_times, city_cumulative),
        seasonal_indexes=seasonal_indexes,
        rare_highways=frozenset(
            config["transforms"]["road_highway_grouping"][
                "rare_categories_mapped_to_OTHER"
            ]
        ),
        feature_order=features,
        categorical_features=tuple(config["categorical_features"]),
        segment_order=segment_order,
        hour_index={time: position for position, time in enumerate(evaluation_hours)},
        segment_index={
            segment: position for position, segment in enumerate(segment_order)
        },
        history_values=_precompute_history_values(
            segment_order,
            evaluation_hours,
            segment_history,
            (city_times, city_cumulative),
            seasonal_indexes,
        ),
    )


def add_indexed_standard_history(
    result: pd.DataFrame, context: EvaluationFeatureContext
) -> pd.DataFrame:
    result["segment_accidents_total_prior"] = _indexed_context_feature(
        result, context.segment_history, ["road_segment_id"]
    )
    query_ns = result.datetime_hour.astype("datetime64[ns]").astype("int64").to_numpy()
    for label, hours in (
        ("segment_accidents_prev_24h", 24),
        ("segment_accidents_prev_7d", 168),
        ("segment_accidents_prev_30d", 720),
    ):
        prior_at_start = np.zeros(len(result), dtype=np.int64)
        for segment, row_index in result.groupby(
            "road_segment_id", sort=False
        ).groups.items():
            event_index = context.segment_history.get((segment,))
            if event_index is None:
                continue
            rows = np.asarray(list(row_index))
            times, cumulative = event_index
            prior_at_start[rows] = _prior_from_index(
                times, cumulative, query_ns[rows] - np.int64(hours * 3_600_000_000_000)
            )
        result[label] = (
            result["segment_accidents_total_prior"].to_numpy() - prior_at_start
        )
    result["segment_hours_since_prev_accident"] = -1.0
    result["segment_has_history"] = False
    for segment, row_index in result.groupby(
        "road_segment_id", sort=False
    ).groups.items():
        event_index = context.segment_history.get((segment,))
        if event_index is None:
            continue
        rows = np.asarray(list(row_index))
        times, _ = event_index
        locations = np.searchsorted(times, query_ns[rows], side="left")
        has_history = locations > 0
        result.loc[rows, "segment_has_history"] = has_history
        hours_since = np.full(len(rows), -1.0)
        hours_since[has_history] = (
            query_ns[rows][has_history] - times[locations[has_history] - 1]
        ) / 3_600_000_000_000
        result.loc[rows, "segment_hours_since_prev_accident"] = hours_since
    city_times, city_cumulative = context.city_history
    city_total = _prior_from_index(city_times, city_cumulative, query_ns)
    result["city_accidents_total_prior"] = city_total
    for label, hours in (
        ("city_accidents_prev_24h", 24),
        ("city_accidents_prev_7d", 168),
        ("city_accidents_prev_30d", 720),
    ):
        result[label] = city_total - _prior_from_index(
            city_times,
            city_cumulative,
            query_ns - np.int64(hours * 3_600_000_000_000),
        )
    return result


def add_indexed_seasonal_history(
    result: pd.DataFrame, context: EvaluationFeatureContext
) -> pd.DataFrame:
    result["_month"] = result.datetime_hour.dt.month.astype("int8")
    result["_hour_of_day"] = result.datetime_hour.dt.hour.astype("int8")
    result["_weekday"] = result.datetime_hour.dt.weekday.astype("int8")
    result["segment_accidents_same_month_prior"] = _indexed_context_feature(
        result, context.seasonal_indexes["month"], ["road_segment_id", "_month"]
    )
    result["segment_accidents_same_hour_weekday_prior"] = _indexed_context_feature(
        result,
        context.seasonal_indexes["hour_weekday"],
        ["road_segment_id", "_hour_of_day", "_weekday"],
    )
    result["segment_accidents_same_month_hour_weekday_prior"] = (
        _indexed_context_feature(
            result,
            context.seasonal_indexes["month_hour_weekday"],
            ["road_segment_id", "_month", "_hour_of_day", "_weekday"],
        )
    )
    result["segment_accidents_holiday_prior"] = _indexed_context_feature(
        result, context.seasonal_indexes["holiday"], ["road_segment_id"]
    )
    result["segment_accidents_new_year_period_prior"] = _indexed_context_feature(
        result, context.seasonal_indexes["new_year"], ["road_segment_id"]
    )
    return result.drop(columns=["_month", "_hour_of_day", "_weekday"])


def build_features_reference(
    grid: pd.DataFrame, ready: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Original Stage-6 construction, retained only for semantic-regression checks."""
    counts = event_counts(ready)
    static = static_segment_features(ready)
    poi = pd.read_parquet(PROCESSED / "road_segment_poi_features.parquet")
    calendar = pd.read_parquet(EXTERNAL / "calendar_features_hourly.parquet")
    weather = pd.read_parquet(EXTERNAL / "weather_astana_hourly.parquet")
    for frame in (calendar, weather):
        frame["datetime_hour"] = pd.to_datetime(frame["datetime_hour"])
    result = (
        grid.rename(columns={"prediction_datetime": "datetime_hour"})
        .merge(static, on="road_segment_id", how="left", validate="many_to_one")
        .merge(poi, on="road_segment_id", how="left", validate="many_to_one")
    )
    calendar["is_new_year_period"] = (
        (calendar.datetime_hour.dt.month == 12) & (calendar.datetime_hour.dt.day >= 25)
    ) | ((calendar.datetime_hour.dt.month == 1) & (calendar.datetime_hour.dt.day <= 7))
    result = result.merge(
        calendar.add_prefix("calendar_").rename(
            columns={"calendar_datetime_hour": "datetime_hour"}
        ),
        on="datetime_hour",
        how="left",
        validate="many_to_one",
    )
    weather = weather.drop(
        columns=[
            column
            for column in ("latitude_source", "longitude_source", "timezone_source")
            if column in weather
        ]
    )
    result = result.merge(
        weather.add_prefix("weather_").rename(
            columns={"weather_datetime_hour": "datetime_hour"}
        ),
        on="datetime_hour",
        how="left",
        validate="many_to_one",
    )
    result = add_standard_history(result, counts)
    result = add_seasonal_history(result, counts, calendar)
    return _apply_frozen_transforms(result, config)


def _apply_frozen_transforms(result: pd.DataFrame, config: dict) -> pd.DataFrame:
    rare = set(
        config["transforms"]["road_highway_grouping"]["rare_categories_mapped_to_OTHER"]
    )
    highway = result.road_highway.astype("string").fillna("UNKNOWN").astype(str)
    result["road_highway"] = highway.where(~highway.isin(rare), "OTHER").astype(
        "string"
    )
    result.loc[result.road_lanes_missing.astype(bool), "road_lanes_num"] = np.nan
    result.loc[result.road_maxspeed_missing.astype(bool), "road_maxspeed_kmh"] = np.nan
    result = result.rename(columns={"datetime_hour": "prediction_datetime"})
    return result


def build_features_for_chunk(
    grid: pd.DataFrame, context: EvaluationFeatureContext, config: dict
) -> pd.DataFrame:
    """Build exactly one chunk while retaining no full-period feature matrix."""
    if len(grid) > 3968 * 6:
        raise ValueError("chunk_rows_exceed_six_hour_memory_bound")
    result = grid.rename(columns={"prediction_datetime": "datetime_hour"}).merge(
        context.static_lookup, on="road_segment_id", how="left", validate="many_to_one"
    )
    chunk_times = pd.DatetimeIndex(result.datetime_hour.unique(), name="datetime_hour")
    calendar = context.calendar_lookup.reindex(chunk_times).reset_index()
    weather = context.weather_lookup.reindex(chunk_times).reset_index()
    result = result.merge(
        calendar, on="datetime_hour", how="left", validate="many_to_one"
    )
    result = result.merge(
        weather, on="datetime_hour", how="left", validate="many_to_one"
    )
    hour_positions = result.datetime_hour.map(context.hour_index).to_numpy(dtype=int)
    segment_positions = (
        result.road_segment_id.astype(str)
        .map(context.segment_index)
        .to_numpy(dtype=int)
    )
    for name, values in context.history_values.items():
        result[name] = values[hour_positions, segment_positions]
    return _apply_frozen_transforms(result, config)


def rank_metrics(frame: pd.DataFrame) -> dict:
    values = []
    for _, hour in frame.groupby("prediction_datetime", sort=False):
        y = hour.target_24h.to_numpy()
        score = hour.raw_model_score.to_numpy()
        order = np.argsort(-score, kind="stable")
        row = {"positives": int(y.sum())}
        for size in (10, 20, 50, 100, 40, 199, 397):
            top = order[:size]
            row[f"precision_{size}"] = float(y[top].mean())
            row[f"recall_{size}"] = float(y[top].sum() / y.sum()) if y.sum() else None
            row[f"lift_{size}"] = float(y[top].mean() / y.mean()) if y.sum() else None
        values.append(row)
    table = pd.DataFrame(values)
    return {
        "hours": len(table),
        "hours_without_positives": int((table.positives == 0).sum()),
        "mean": table.mean(numeric_only=True).to_dict(),
        "median": table.median(numeric_only=True).to_dict(),
        "top_definition": {"top_1pct": 40, "top_5pct": 199, "top_10pct": 397},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-datetime", default="2024-10-01T00:00:00+05:00")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--chunk-hours", type=int, default=6)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "audit" / "stage19e"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int)
    args = parser.parse_args()
    progress("parsed_cli_arguments")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions_parts").mkdir(exist_ok=True)
    start = (
        pd.Timestamp(args.start_datetime).tz_convert(None)
        if pd.Timestamp(args.start_datetime).tzinfo
        else pd.Timestamp(args.start_datetime)
    )
    hours = pd.date_range(start, periods=args.days * 24, freq="h")
    progress("evaluation_interval_selected")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    features = [*config["numerical_features"], *config["categorical_features"]]
    if feature_hash(features) != EXPECTED_HASH:
        raise ValueError("frozen_77_feature_order_hash_mismatch")
    initial_checkpoint = {
        "status": "initialized",
        "evaluation_start": str(hours.min()),
        "evaluation_end": str(hours.max()),
        "chunk_hours": args.chunk_hours,
        "expected_chunks": int(np.ceil(len(hours) / args.chunk_hours)),
        "completed_chunks": [],
        "feature_order_hash": EXPECTED_HASH,
        "parameters": {
            "start": str(hours.min()),
            "end": str(hours.max()),
            "chunk_hours": args.chunk_hours,
            "feature_order_hash": EXPECTED_HASH,
            "model_hash": hashlib.sha256(
                (ROOT / "models" / "production" / "catboost_24h.cbm").read_bytes()
            ).hexdigest(),
        },
    }
    checkpoint_candidate = args.output_dir / "checkpoint.json"
    if not checkpoint_candidate.exists() or args.overwrite:
        atomic_json(checkpoint_candidate, initial_checkpoint)
    progress("checkpoint_initialized")
    ready = pd.read_parquet(PROCESSED / "accidents_with_roads_ml_ready.parquet")
    progress("static_inputs_loaded")
    context = build_feature_context(ready, config, hours)
    segments = (
        context.static_lookup.road_segment_id.astype(str).sort_values().to_numpy()
    )
    if len(segments) != 3968:
        raise ValueError(f"expected_3968_segments_got_{len(segments)}")
    progress("production_segment_ids_loaded")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": len(hours) * len(segments),
                    "timestamps": len(hours),
                    "feature_hash": feature_hash(features),
                },
                indent=2,
            )
        )
        return 0
    started = datetime.now(UTC)
    if args.chunk_hours < 1:
        raise ValueError("chunk_hours_must_be_positive")
    model = CatBoostClassifier()
    model.load_model(ROOT / "models" / "production" / "catboost_24h.cbm")
    progress("model_loaded")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "predictions_parts"
    parts_dir.mkdir(exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.json"
    model_hash = hashlib.sha256(
        (ROOT / "models" / "production" / "catboost_24h.cbm").read_bytes()
    ).hexdigest()
    expected_chunks = int(np.ceil(len(hours) / args.chunk_hours))
    parameters = {
        "start": str(hours.min()),
        "end": str(hours.max()),
        "chunk_hours": args.chunk_hours,
        "feature_order_hash": EXPECTED_HASH,
        "model_hash": model_hash,
    }
    checkpoint = {
        "parameters": parameters,
        "expected_chunks": expected_chunks,
        "completed_chunks": [],
    }
    if args.resume and checkpoint_path.exists() and not args.overwrite:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("parameters") != parameters:
            raise ValueError("incompatible_resume_parameters")
    completed = set(checkpoint.get("completed_chunks", []))
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    hourly_metrics: list[dict] = []
    counts = context.counts
    for chunk_index, begin in enumerate(range(0, len(hours), args.chunk_hours)):
        part_path = parts_dir / f"part_{chunk_index:04d}.parquet"
        if chunk_index in completed and part_path.exists():
            compact = pd.read_parquet(
                part_path,
                columns=[
                    "target_24h",
                    "raw_model_score",
                    "prediction_datetime",
                    "risk_rank_within_hour",
                ],
            )
            labels.append(compact.target_24h.to_numpy(np.uint8))
            scores.append(compact.raw_model_score.to_numpy(np.float32))
            hourly_metrics.append(rank_metrics(compact))
            continue
        chunk_times = hours[begin : begin + args.chunk_hours]
        progress("chunk_start", chunk_index)
        grid = pd.MultiIndex.from_product(
            [chunk_times, segments], names=["prediction_datetime", "road_segment_id"]
        ).to_frame(index=False)
        progress("features_started", chunk_index)
        full = build_features_for_chunk(grid, context, config)
        progress("features_completed", chunk_index)
        targets = target_fields(
            full[["road_segment_id", "prediction_datetime"]], counts
        )
        full = full.merge(
            targets,
            on=["road_segment_id", "prediction_datetime"],
            validate="one_to_one",
        )
        output_base = full[
            [
                "road_segment_id",
                "prediction_datetime",
                "target_24h",
                "target_accident_count",
                "segment_accidents_total_prior",
            ]
        ].copy()
        chunk_scores = np.empty(len(full), dtype=np.float64)
        for _timestamp, positions in full.groupby(
            "prediction_datetime", sort=False
        ).indices.items():
            model_matrix = full.iloc[positions][features].copy()
            for column in config["categorical_features"]:
                model_matrix[column] = (
                    model_matrix[column]
                    .astype("string")
                    .fillna("__MISSING__")
                    .astype(str)
                )
            chunk_scores[positions] = model.predict_proba(model_matrix, thread_count=1)[
                :, 1
            ]
            del model_matrix
        del full, targets
        gc.collect()
        output_base["raw_model_score"] = chunk_scores
        progress("prediction_completed", chunk_index)
        output_base = output_base.sort_values(
            ["prediction_datetime", "road_segment_id"]
        )
        output_base["risk_rank_within_hour"] = (
            output_base.groupby("prediction_datetime")
            .raw_model_score.rank(method="first", ascending=False)
            .astype("int16")
        )
        output_base["risk_percentile_within_hour"] = (
            output_base.groupby("prediction_datetime")
            .raw_model_score.rank(method="max", pct=True, ascending=False)
            .astype("float32")
        )
        output_base["history_tier"] = "uncomputed_audit_pending"
        output_base["baseline_score"] = output_base[
            "segment_accidents_total_prior"
        ].astype("float32")
        output_base["model_version"] = "stage7b-final-v1"
        output_base["feature_order_hash"] = EXPECTED_HASH
        output = output_base[
            [
                "road_segment_id",
                "prediction_datetime",
                "target_24h",
                "target_accident_count",
                "raw_model_score",
                "risk_rank_within_hour",
                "risk_percentile_within_hour",
                "history_tier",
                "baseline_score",
                "model_version",
                "feature_order_hash",
            ]
        ]
        temporary_part = part_path.with_suffix(".parquet.tmp")
        output.to_parquet(temporary_part, index=False)
        os.replace(temporary_part, part_path)
        progress("part_written", chunk_index)
        compact = output[
            [
                "target_24h",
                "raw_model_score",
                "prediction_datetime",
                "risk_rank_within_hour",
            ]
        ]
        labels.append(compact.target_24h.to_numpy(np.uint8))
        scores.append(compact.raw_model_score.to_numpy(np.float32))
        hourly_metrics.append(rank_metrics(compact))
        completed.add(chunk_index)
        checkpoint["completed_chunks"] = sorted(completed)
        checkpoint["status"] = "running"
        checkpoint.setdefault("parts", {})[str(chunk_index)] = {
            "part_path": str(part_path),
            "row_count": int(len(output)),
            "timestamp_start": str(chunk_times.min()),
            "timestamp_end": str(chunk_times.max()),
            "target_checksum": hashlib.sha256(
                compact.target_24h.to_numpy(np.uint8).tobytes()
            ).hexdigest(),
            "prediction_checksum": hashlib.sha256(
                compact.raw_model_score.to_numpy(np.float32).tobytes()
            ).hexdigest(),
        }
        atomic_json(checkpoint_path, checkpoint)
        del grid, chunk_scores, output, output_base, compact
        gc.collect()
        # CatBoost retains native prediction buffers across calls in this audit
        # environment. Reloading the identical frozen model bounds that buffer
        # without changing model bytes, feature order, or prediction semantics.
        model = CatBoostClassifier()
        model.load_model(ROOT / "models" / "production" / "catboost_24h.cbm")
        if args.stop_after_chunks and len(completed) >= args.stop_after_chunks:
            checkpoint["status"] = "stopped_for_test"
            atomic_json(checkpoint_path, checkpoint)
            return 0
    y = np.concatenate(labels)
    score = np.concatenate(scores)
    np.savez_compressed(
        args.output_dir / "compact_metric_arrays.npz", y_true=y, raw_model_score=score
    )
    checkpoint["status"] = "completed"
    atomic_json(checkpoint_path, checkpoint)
    per_hour_means = pd.DataFrame([item["mean"] for item in hourly_metrics]).mean(
        numeric_only=True
    )
    report = {
        "interval": {
            "start": str(hours.min()),
            "end": str(hours.max()),
            "hours": len(hours),
        },
        "rows": int(len(y)),
        "expected_rows": len(hours) * len(segments),
        "segments": len(segments),
        "natural_prevalence": float(y.mean()),
        "positive_rows": int(y.sum()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "top_k_mean_by_hour": {
            "precision_at_10": float(per_hour_means["precision_10"]),
            "precision_at_20": float(per_hour_means["precision_20"]),
            "precision_at_50": float(per_hour_means["precision_50"]),
            "recall_at_1pct": float(per_hour_means["recall_40"]),
            "recall_at_5pct": float(per_hour_means["recall_199"]),
            "recall_at_10pct": float(per_hour_means["recall_397"]),
            "lift_at_1pct": float(per_hour_means["lift_40"]),
            "lift_at_5pct": float(per_hour_means["lift_199"]),
            "lift_at_10pct": float(per_hour_means["lift_397"]),
        },
        "feature_order_hash": EXPECTED_HASH,
        "ranking_by_chunk": hourly_metrics,
        "calibration": "raw score; no natural validation calibration fitted",
        "runtime_seconds": (datetime.now(UTC) - started).total_seconds(),
        "completed_chunks": len(completed),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name in (
        "input_audit.json",
        "grid_validation.json",
        "target_validation.json",
        "feature_reconstruction_audit.json",
        "leakage_audit.json",
        "natural_prevalence_metrics.json",
        "hourly_ranking_metrics.json",
        "calibration_readiness.json",
        "determinism_report.json",
    ):
        (REPORTS / name).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
    (REPORTS / "stage19e_natural_evaluation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (REPORTS / "stage19e_natural_evaluation.md").write_text(
        "# Stage 19E natural-prevalence evaluation\n\nRaw Stage 7B scores only; no probability calibration was fitted.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
