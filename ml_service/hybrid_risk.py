"""Stage 20A hybrid risk contract.

The dynamic engine remains separate from operational context: it returns ranks,
not calibrated accident probabilities.  Historical accident counts are always
computed using records strictly before the requested prediction hour.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ml_service.inference.feature_builder import ROOT, build_features
from ml_service.inference.catboost_explanations import (
    catboost_explanations,
    unavailable_explanations,
)


ENGINE_VERSION = "stage19i_ensemble_v1"
ENGINE_STATUS = "experimental"
EXPECTED_SEGMENTS = 3968
DEFAULT_FUTURE_CONTEXT_PATH = (
    ROOT
    / "data"
    / "future_intelligence"
    / "processed"
    / "unified_future_features_24h.parquet"
)
_KEY = ["road_segment_id", "prediction_datetime"]
ALMATY = ZoneInfo("Asia/Almaty")
TTL_MINUTES = {"weather": 90, "traffic": 20, "repairs": 360, "events": 1440}


class LegacyWeatherSnapshotError(ValueError):
    pass


class WeatherSnapshotMismatchError(ValueError):
    pass


def validate_weather_snapshot(context: pd.DataFrame | None) -> dict[str, object]:
    if context is None or context.empty or "weather_snapshot_version" not in context:
        return {
            "consistent": False,
            "issue": "WEATHER_SNAPSHOT_LEGACY_SCHEMA",
            "version": None,
        }
    row = context.iloc[0]
    version = row.get("weather_snapshot_version")
    for idx in range(len(context)):
        if version and not pd.isna(version):
            break
        version = context.iloc[idx].get("weather_snapshot_version")
    if not version or pd.isna(version):
        for idx in range(len(context)):
            snap = context.iloc[idx].get("weather_snapshot")
            if isinstance(snap, str):
                try:
                    version = json.loads(snap).get("snapshot_version")
                    if version:
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass
    if not version or pd.isna(version):
        source_digest = (
            context.attrs.get("weather_snapshot_version")
            if hasattr(context, "attrs")
            else None
        )
        return {
            "consistent": False,
            "issue": "WEATHER_SNAPSHOT_LEGACY_SCHEMA",
            "version": source_digest,
        }
    required = (
        "weather_forecast_start",
        "weather_forecast_end",
    )
    if any(name not in context or pd.isna(row.get(name)) for name in required):
        return {
            "consistent": False,
            "issue": "WEATHER_SNAPSHOT_MISMATCH",
            "version": version,
        }
    return {"consistent": True, "issue": None, "version": version}


def _weights() -> tuple[float, float]:
    config_path = ROOT / "models" / "final" / "ensemble_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights = config["weights"]
    catboost_weight = float(weights["score_catboost_stage19h"])
    hgb_weight = float(weights["score_hist_gradient_boosting"])
    if not np.isclose(catboost_weight, 0.8) or not np.isclose(hgb_weight, 0.2):
        raise ValueError("stage19i_frozen_weights_invalid")
    return catboost_weight, hgb_weight


def _local_model_hour(value: str | pd.Timestamp) -> pd.Timestamp:
    """Map an API timestamp to the local-naive hour used by archived features."""

    timestamp = pd.Timestamp(value).floor("h")
    return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp


@lru_cache(maxsize=1)
def _canonical_segment_coordinates() -> pd.DataFrame:
    """Return deterministic centroids from the canonical OSM edge source."""
    path = ROOT / "data" / "roads" / "astana_edges.csv"
    if not path.exists():
        return pd.DataFrame(columns=["road_segment_id", "longitude", "latitude"])
    roads = pd.read_csv(path, usecols=["u", "v", "key", "geometry"])

    def centroid(value: object) -> tuple[float | None, float | None]:
        pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", str(value))
        if not pairs:
            return None, None
        points = np.asarray([(float(lon), float(lat)) for lon, lat in pairs])
        lon, lat = float(points[:, 0].mean()), float(points[:, 1].mean())
        return (
            (lon, lat) if 70.0 <= lon <= 72.0 and 50.0 <= lat <= 52.0 else (None, None)
        )

    values = roads["geometry"].map(centroid)
    return pd.DataFrame(
        {
            "road_segment_id": (
                roads["u"].astype(str)
                + "_"
                + roads["v"].astype(str)
                + "_"
                + roads["key"].astype(str)
            ),
            "longitude": values.map(lambda item: item[0]),
            "latitude": values.map(lambda item: item[1]),
        }
    ).drop_duplicates("road_segment_id")


def _weather_override(context: pd.DataFrame | None) -> dict | None:
    """Map only existing frozen weather inputs from a valid future snapshot."""
    if (
        context is None
        or context.empty
        or not bool(context.get("weather_provider_available", pd.Series([0])).iloc[0])
    ):
        return None
    row = context.iloc[0]
    origin = row.get("weather_origin")
    if isinstance(origin, str):
        try:
            origin = json.loads(origin)
        except json.JSONDecodeError:
            origin = None
    if not isinstance(origin, dict):
        origin = {
            "temperature": row.get("weather_origin_temperature"),
            "humidity": row.get("weather_origin_humidity"),
            "rain": row.get("weather_origin_rain"),
            "wind_speed": row.get("weather_origin_wind_speed"),
            "visibility": row.get("weather_origin_visibility"),
        }
    # Never substitute 24-hour aggregates for frozen-model origin-hour inputs.
    if not row.get("weather_snapshot_version") or all(
        pd.isna(value) for value in origin.values()
    ):
        return None
    return {
        "temperature_2m": origin.get("temperature", np.nan),
        "relative_humidity_2m": origin.get("humidity", np.nan),
        "precipitation": origin.get("rain", np.nan),
        "rain": origin.get("rain", np.nan),
        "snowfall": origin.get("snow", np.nan),
        "weather_code": np.nan,
        "cloud_cover": origin.get("clouds", np.nan),
        "wind_speed_10m": origin.get("wind_speed", np.nan),
        "wind_gusts_10m": origin.get("wind_gust", np.nan),
    }


def _component_metadata(
    cat_scores: pd.Series,
    cat_percentiles: pd.Series,
    hgb_scores: pd.Series,
    hgb_percentiles: pd.Series,
    cat_weight: float,
    hgb_weight: float,
) -> list[dict[str, dict[str, float | str]]]:
    """Return transparent component metadata; final ordering stays separate."""

    return [
        {
            "catboost": {
                "score": float(cat_score),
                "score_type": "probability",
                "percentile": float(cat_percentile),
                "weight": round(cat_weight, 12),
            },
            "hgb": {
                "score": float(hgb_score),
                "score_type": "probability",
                "percentile": float(hgb_percentile),
                "weight": round(hgb_weight, 12),
            },
        }
        for cat_score, cat_percentile, hgb_score, hgb_percentile in zip(
            cat_scores,
            cat_percentiles,
            hgb_scores,
            hgb_percentiles,
            strict=True,
        )
    ]


def build_dynamic_risk(
    prediction_datetime: str | pd.Timestamp,
    *,
    weather_override: dict | None = None,
    strict_live_features: bool = False,
) -> pd.DataFrame:
    """Score every known segment with the frozen CatBoost/HGB percentile ensemble."""

    features, _ = build_features(
        _local_model_hour(prediction_datetime),
        "24h",
        weather_override=weather_override,
        strict_live_weather=strict_live_features,
    )
    config_path = (
        ROOT
        / "reports"
        / "stage7a"
        / "24h"
        / "20260711T090515Z"
        / "training_dataset_24h_feature_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ordered_features = [*config["numerical_features"], *config["categorical_features"]]
    categorical_features = config["categorical_features"]
    cat_features = features[ordered_features].copy()
    for column in categorical_features:
        cat_features[column] = (
            cat_features[column].astype("string").fillna("__MISSING__").astype(str)
        )

    catboost = CatBoostClassifier()
    catboost.load_model(ROOT / "models" / "final" / "stage19i_catboost.cbm")
    cat_scores = catboost.predict_proba(cat_features)[:, 1]
    preprocessor = joblib.load(
        ROOT / "models" / "final" / "stage19i_preprocessor.joblib"
    )
    hgb = joblib.load(
        ROOT / "models" / "final" / "stage19i_hist_gradient_boosting.joblib"
    )
    hgb_scores = hgb.predict_proba(preprocessor.transform(features))[:, 1]
    cat_weight, hgb_weight = _weights()

    try:
        explanations, _ = catboost_explanations(
            catboost,
            features,
            ordered_features=ordered_features,
            categorical_features=categorical_features,
            component_weight=cat_weight,
        )
        explanation_warnings: list[list[str]] = [[] for _ in range(len(features))]
    except Exception:
        explanations = unavailable_explanations(len(features), cat_weight)
        explanation_warnings = [
            ["CATBOOST_EXPLANATION_UNAVAILABLE"] for _ in range(len(features))
        ]

    result = features[["road_segment_id", "datetime_hour"]].copy()
    result = result.rename(columns={"datetime_hour": "prediction_datetime"})
    result["prediction_datetime"] = pd.Timestamp(prediction_datetime).floor("h")
    result["road_segment_id"] = result["road_segment_id"].astype(str)
    result["score_catboost_stage19h"] = cat_scores
    result["score_hist_gradient_boosting"] = hgb_scores
    result["enrichment_warnings"] = explanation_warnings
    result["explanation"] = explanations
    # Percentiles are calculated within this one complete prediction hour.
    result["_cat_percentile"] = result["score_catboost_stage19h"].rank(pct=True)
    result["_hgb_percentile"] = result["score_hist_gradient_boosting"].rank(pct=True)
    result["dynamic_score"] = (
        cat_weight * result["_cat_percentile"] + hgb_weight * result["_hgb_percentile"]
    )
    result = result.sort_values("road_segment_id", kind="stable").reset_index(drop=True)
    result["dynamic_rank"] = (
        result["dynamic_score"].rank(method="first", ascending=False).astype("int32")
    )
    result["dynamic_percentile"] = result["dynamic_score"].rank(
        method="average", pct=True
    )
    result["dynamic_engine_version"] = ENGINE_VERSION
    result["dynamic_engine_status"] = ENGINE_STATUS
    result = result.merge(
        _canonical_segment_coordinates(),
        on="road_segment_id",
        how="left",
        validate="one_to_one",
    )
    for coordinate in ("longitude", "latitude"):
        result[coordinate] = (
            result[coordinate].astype(object).where(result[coordinate].notna(), None)
        )
    result["dynamic_risk"] = [
        {
            "score": float(score),
            "score_type": "weighted_percentile_ensemble",
            "rank": int(rank),
            "percentile": float(percentile),
            "population_size": EXPECTED_SEGMENTS,
            "horizon_hours": 24,
            "engine": "stage19i_ensemble",
            "weights": {
                "catboost": round(cat_weight, 12),
                "hgb": round(hgb_weight, 12),
            },
        }
        for score, rank, percentile in zip(
            result["dynamic_score"],
            result["dynamic_rank"],
            result["dynamic_percentile"],
            strict=True,
        )
    ]
    result["model_components"] = _component_metadata(
        result["score_catboost_stage19h"],
        result["_cat_percentile"],
        result["score_hist_gradient_boosting"],
        result["_hgb_percentile"],
        cat_weight,
        hgb_weight,
    )
    return result.drop(columns=["_cat_percentile", "_hgb_percentile"])


def validate_dynamic_risk(
    frame: pd.DataFrame, expected_segments: int = EXPECTED_SEGMENTS
) -> None:
    """Fail closed when the stable Stage 20A dynamic output contract is broken."""

    required = {
        "road_segment_id",
        "prediction_datetime",
        "dynamic_score",
        "dynamic_rank",
        "dynamic_percentile",
        "dynamic_engine_version",
        "dynamic_engine_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"stage20a_dynamic_missing_columns:{sorted(missing)}")
    if (
        len(frame) != expected_segments
        or frame.road_segment_id.nunique() != expected_segments
    ):
        raise ValueError("stage20a_dynamic_segment_grain_invalid")
    if frame.duplicated(_KEY).any() or frame[list(required)].isna().any().any():
        raise ValueError("stage20a_dynamic_null_or_duplicate")
    if not np.isfinite(frame[["dynamic_score", "dynamic_percentile"]].to_numpy()).all():
        raise ValueError("stage20a_dynamic_non_finite")
    if set(frame.dynamic_rank) != set(range(1, expected_segments + 1)):
        raise ValueError("stage20a_dynamic_rank_invalid")


def _historical_hotspot(
    prediction_datetime: str | pd.Timestamp, segments: pd.Series
) -> pd.DataFrame:
    """Return strict-prior all-time and rolling historical accident counts."""

    moment = _local_model_hour(prediction_datetime)
    accidents = pd.read_parquet(
        ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet",
        columns=["road_segment_id", "accident_datetime"],
    )
    accidents["road_segment_id"] = accidents["road_segment_id"].astype(str)
    accidents["event_hour"] = pd.to_datetime(accidents["accident_datetime"]).dt.floor(
        "h"
    )
    prior = accidents.loc[accidents["event_hour"] < moment]
    result = pd.DataFrame({"road_segment_id": segments.astype(str)})
    for name, start in (
        ("historical_accident_count", None),
        ("historical_accident_count_30d", moment - pd.Timedelta(days=30)),
        ("historical_accident_count_90d", moment - pd.Timedelta(days=90)),
        ("historical_accident_count_365d", moment - pd.Timedelta(days=365)),
    ):
        source = prior if start is None else prior.loc[prior["event_hour"] >= start]
        result[name] = (
            result["road_segment_id"]
            .map(source["road_segment_id"].value_counts())
            .fillna(0)
            .astype("int32")
        )
    result["historical_hotspot_score"] = result["historical_accident_count"].astype(
        float
    )
    result = result.sort_values("road_segment_id", kind="stable").reset_index(drop=True)
    result["historical_hotspot_rank"] = (
        result["historical_hotspot_score"]
        .rank(method="first", ascending=False)
        .astype("int32")
    )
    result["historical_hotspot_percentile"] = result["historical_hotspot_score"].rank(
        method="average", pct=True
    )
    return result


def _future_context(
    segments: pd.Series,
    future_context: pd.DataFrame | None,
    prediction_datetime: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize optional Stage 18 context; unavailable providers stay explicit."""

    result = pd.DataFrame({"road_segment_id": segments.astype(str)})
    defaults: dict[str, object] = {
        "weather_context_available": False,
        "traffic_context_available": False,
        "repair_context_available": False,
        "event_context_available": False,
        "future_context_flags": "[]",
        "future_context_warnings": '["future_context_unavailable"]',
        "future_context_confidence": "degraded",
        "provider_degraded": True,
    }
    if future_context is None:
        return result.assign(**defaults)
    if "road_segment_id" not in future_context:
        raise ValueError("stage20a_future_context_missing_road_segment_id")
    source = future_context.copy()
    source["road_segment_id"] = source["road_segment_id"].astype(str)
    if prediction_datetime is not None and "prediction_datetime" in source:
        requested_hour = _local_model_hour(prediction_datetime)
        context_hours = source["prediction_datetime"].map(_local_model_hour)
        source = source.loc[context_hours.eq(requested_hour)].copy()
    if source.duplicated("road_segment_id").any():
        raise ValueError("stage20a_future_context_duplicate_segment")
    source = source.set_index("road_segment_id")

    def enabled(*names: str) -> pd.Series:
        available = next((name for name in names if name in source), None)
        if available is None:
            return pd.Series(False, index=result.index)
        return result["road_segment_id"].map(source[available]).fillna(0).astype(bool)

    result["weather_context_available"] = enabled(
        "weather_provider_available", "weather_available"
    )
    result["traffic_context_available"] = enabled(
        "traffic_context_available", "traffic_provider_available"
    )
    result["repair_context_available"] = enabled(
        "gov_provider_available", "repair_provider_available"
    )
    result["event_context_available"] = enabled(
        "ticketon_provider_available", "event_provider_available"
    )
    degraded = (
        enabled("provider_degraded")
        if "provider_degraded" in source
        else ~(
            result[
                [
                    "weather_context_available",
                    "traffic_context_available",
                    "repair_context_available",
                    "event_context_available",
                ]
            ].any(axis=1)
        )
    )
    result["provider_degraded"] = degraded

    flags: list[str] = []
    for _, row in result.iterrows():
        row_flags = []
        original = (
            source.loc[row["road_segment_id"]]
            if row["road_segment_id"] in source.index
            else pd.Series(dtype=object)
        )
        severity = float(original.get("weather_severity_score", 0) or 0)
        severity = (
            severity if severity <= 1 else severity / (5 if severity <= 5 else 10)
        )
        if row["weather_context_available"] and min(1.0, max(0.0, severity)) >= float(
            os.getenv("WEATHER_SEVERE_THRESHOLD", "0.60")
        ):
            row_flags.append("severe_weather")
        if (
            row["traffic_context_available"]
            and float(original.get("traffic_congestion_score", 0) or 0) >= 0.7
        ):
            row_flags.append("heavy_traffic")
        if float(original.get("repair_active_next_24h", 0) or 0) > 0:
            row_flags.append("road_repair")
        if float(original.get("event_major_next_24h", 0) or 0) > 0:
            row_flags.append("major_event")
        flags.append(json.dumps(row_flags))
    result["future_context_flags"] = flags
    result["future_context_warnings"] = result["provider_degraded"].map(
        lambda value: '["provider_degraded"]' if value else "[]"
    )
    result["future_context_confidence"] = result["provider_degraded"].map(
        lambda value: "degraded" if value else "available"
    )
    # Compact, safe provenance for the API; never include raw provider payloads.
    for output, source_name in (
        ("weather_severity_score", "weather_severity_score"),
        ("weather_collected_at", "weather_collection_timestamp"),
        ("traffic_collected_at", "traffic_collection_timestamp"),
        ("repairs_collected_at", "gov_collection_timestamp"),
        ("events_collected_at", "ticketon_collection_timestamp"),
        ("traffic_severity_score", "traffic_congestion_score"),
        ("repair_active", "repair_active_next_24h"),
        ("event_major", "event_major_next_24h"),
        ("repair_source_id", "repair_source_id"),
        ("repair_title", "repair_title"),
        ("repair_road_name", "repair_road_name"),
        ("repair_start", "repair_start"),
        ("repair_end", "repair_end"),
        ("event_source_id", "event_source_id"),
        ("event_name", "event_name"),
        ("event_venue", "event_venue"),
        ("event_start", "event_start"),
        ("event_end", "event_end"),
    ):
        values = (
            result["road_segment_id"].map(source[source_name])
            if source_name in source
            else np.nan
        )
        if output == "weather_severity_score":
            values = pd.to_numeric(values, errors="coerce")
            values = values.where(
                values <= 1, values / values.where(values <= 5, 10).where(values > 1, 1)
            ).clip(0, 1)
        result[output] = values
    return result


def load_valid_future_context(
    prediction_datetime: str | pd.Timestamp, path: Path = DEFAULT_FUTURE_CONTEXT_PATH
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load a snapshot only when its prediction window and provider TTLs are valid."""
    if not path.exists():
        return None, [
            "FUTURE_CONTEXT_UNAVAILABLE",
            "WEATHER_PROVIDER_DEGRADED",
            "TRAFFIC_PROVIDER_DEGRADED",
            "REPAIRS_PROVIDER_DEGRADED",
            "EVENTS_PROVIDER_DEGRADED",
        ]
    source = pd.read_parquet(path)
    requested = pd.Timestamp(prediction_datetime)
    if requested.tzinfo is None:
        requested = requested.tz_localize(ALMATY)
    starts = pd.to_datetime(source["forecast_window_start"], utc=True)
    ends = pd.to_datetime(source["forecast_window_end"], utc=True)
    selected = source.loc[
        (starts <= requested.tz_convert("UTC")) & (ends > requested.tz_convert("UTC"))
    ].copy()
    if selected.empty:
        return None, [
            "FUTURE_CONTEXT_STALE",
            "WEATHER_PROVIDER_DEGRADED",
            "TRAFFIC_PROVIDER_DEGRADED",
            "REPAIRS_PROVIDER_DEGRADED",
            "EVENTS_PROVIDER_DEGRADED",
        ]
    now = datetime.now(UTC)
    warnings: list[str] = []
    checks = [
        ("weather", "weather_collection_timestamp", "WEATHER_PROVIDER_DEGRADED"),
        ("traffic", "traffic_collection_timestamp", "TRAFFIC_PROVIDER_DEGRADED"),
        ("repairs", "gov_collection_timestamp", "REPAIRS_PROVIDER_DEGRADED"),
        ("events", "ticketon_collection_timestamp", "EVENTS_PROVIDER_DEGRADED"),
    ]
    for name, column, warning in checks:
        ttl = int(os.getenv(f"FUTURE_{name.upper()}_TTL_MINUTES", TTL_MINUTES[name]))
        value = selected[column].iloc[0] if column in selected else None
        try:
            fresh = pd.Timestamp(value).to_pydatetime().astimezone(
                UTC
            ) >= now - timedelta(minutes=ttl)
        except (TypeError, ValueError):
            fresh = False
        if not fresh:
            warnings.append(warning)
            if name == "weather":
                selected["weather_provider_available"] = 0
            if name == "traffic":
                selected["traffic_context_available"] = 0
            if name == "repairs":
                selected["gov_provider_available"] = 0
            if name == "events":
                selected["ticketon_provider_available"] = 0
    if warnings:
        warnings.insert(0, "FUTURE_CONTEXT_STALE")
        selected["provider_degraded"] = 1
    return selected, warnings


def build_hybrid_risk(
    prediction_datetime: str | pd.Timestamp,
    *,
    future_context: pd.DataFrame | None = None,
    strict_live_features: bool = False,
) -> pd.DataFrame:
    """Build the complete one-row-per-segment Stage 20A operational contract."""

    # Stage 18B is the canonical future-context source.  Its absence is valid
    # operationally and is represented by explicit degraded fields below.
    snapshot_warnings: list[str] = []
    if future_context is None:
        future_context, snapshot_warnings = load_valid_future_context(
            prediction_datetime
        )
    consistency = validate_weather_snapshot(future_context)
    if not consistency["consistent"]:
        if strict_live_features:
            error = (
                LegacyWeatherSnapshotError
                if consistency["issue"] == "WEATHER_SNAPSHOT_LEGACY_SCHEMA"
                else WeatherSnapshotMismatchError
            )
            raise error(str(consistency["issue"]))
        snapshot_warnings.append(str(consistency["issue"]))
        future_context = None
    dynamic = build_dynamic_risk(
        prediction_datetime,
        weather_override=_weather_override(future_context),
        strict_live_features=strict_live_features,
    )
    historical = _historical_hotspot(prediction_datetime, dynamic["road_segment_id"])
    future = _future_context(
        dynamic["road_segment_id"], future_context, prediction_datetime
    )
    result = dynamic.merge(historical, on="road_segment_id", validate="one_to_one")
    result = result.merge(future, on="road_segment_id", validate="one_to_one")
    result["future_snapshot_warnings"] = json.dumps(snapshot_warnings)
    result["weather_provider_degraded"] = (
        "WEATHER_PROVIDER_DEGRADED" in snapshot_warnings
    )
    result["ml_weather_snapshot_version"] = (
        consistency["version"] if consistency["consistent"] else None
    )
    result["explanation_weather_snapshot_version"] = (
        consistency["version"] if consistency["consistent"] else None
    )
    result["ml_weather_degraded"] = not bool(consistency["consistent"])
    source = (
        future_context.iloc[0]
        if future_context is not None and not future_context.empty
        else pd.Series(dtype=object)
    )
    for output, column in (
        ("ml_weather_origin_timestamp", "weather_origin_prediction_datetime"),
        ("ml_weather_source_before", "weather_origin_source_before"),
        ("ml_weather_source_after", "weather_origin_source_after"),
        ("ml_weather_interpolated", "weather_origin_interpolated"),
        ("explanation_forecast_start", "weather_forecast_start"),
        ("explanation_forecast_end", "weather_forecast_end"),
        ("explanation_forecast_points_available", "weather_forecast_points_available"),
    ):
        result[output] = source.get(column) if consistency["consistent"] else None
    result["weather_snapshot_consistent"] = bool(consistency["consistent"])
    result["weather_snapshot_issues"] = json.dumps(
        [consistency["issue"]] if consistency["issue"] else []
    )
    validate_hybrid_risk(result)
    return result


def validate_hybrid_risk(
    frame: pd.DataFrame, expected_segments: int = EXPECTED_SEGMENTS
) -> None:
    validate_dynamic_risk(frame, expected_segments)
    required = {
        "historical_accident_count",
        "historical_accident_count_30d",
        "historical_accident_count_90d",
        "historical_accident_count_365d",
        "historical_hotspot_score",
        "historical_hotspot_rank",
        "historical_hotspot_percentile",
        "weather_context_available",
        "traffic_context_available",
        "repair_context_available",
        "event_context_available",
        "future_context_flags",
        "future_context_warnings",
        "future_context_confidence",
        "provider_degraded",
    }
    missing = required - set(frame.columns)
    if missing or frame[list(required)].isna().any().any():
        raise ValueError(f"stage20a_hybrid_required_fields_invalid:{sorted(missing)}")
    if set(frame["historical_hotspot_rank"]) != set(range(1, expected_segments + 1)):
        raise ValueError("stage20a_historical_rank_invalid")
    if (
        (
            frame[
                [
                    "historical_accident_count",
                    "historical_accident_count_30d",
                    "historical_accident_count_90d",
                    "historical_accident_count_365d",
                ]
            ]
            < 0
        )
        .any()
        .any()
    ):
        raise ValueError("stage20a_historical_count_invalid")
