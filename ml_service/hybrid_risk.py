"""Stage 20A hybrid risk contract.

The dynamic engine remains separate from operational context: it returns ranks,
not calibrated accident probabilities.  Historical accident counts are always
computed using records strictly before the requested prediction hour.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ml_service.inference.feature_builder import ROOT, build_features


ENGINE_VERSION = "stage19i_ensemble_v1"
ENGINE_STATUS = "experimental"
EXPECTED_SEGMENTS = 3968
DEFAULT_FUTURE_CONTEXT_PATH = (
    ROOT / "data" / "future_intelligence" / "processed" / "unified_future_features_24h.parquet"
)
_KEY = ["road_segment_id", "prediction_datetime"]


def _weights() -> tuple[float, float]:
    config_path = ROOT / "models" / "stage19i_simple" / "weighted_ensemble_config.json"
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


def build_dynamic_risk(prediction_datetime: str | pd.Timestamp) -> pd.DataFrame:
    """Score every known segment with the frozen CatBoost/HGB percentile ensemble."""

    features, _ = build_features(_local_model_hour(prediction_datetime), "24h")
    config_path = ROOT / "reports" / "stage7a" / "24h" / "20260711T090515Z" / "training_dataset_24h_feature_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ordered_features = [*config["numerical_features"], *config["categorical_features"]]
    categorical_features = config["categorical_features"]
    cat_features = features[ordered_features].copy()
    for column in categorical_features:
        cat_features[column] = cat_features[column].astype("string").fillna("__MISSING__").astype(str)

    catboost = CatBoostClassifier()
    catboost.load_model(ROOT / "models" / "stage19h" / "catboost_candidate.cbm")
    cat_scores = catboost.predict_proba(cat_features)[:, 1]
    preprocessor = joblib.load(ROOT / "models" / "stage19h" / "train_only_preprocessor_v2.joblib")
    hgb = joblib.load(ROOT / "models" / "stage19h" / "hist_gradient_boosting_candidate.joblib")
    hgb_scores = hgb.predict_proba(preprocessor.transform(features))[:, 1]
    cat_weight, hgb_weight = _weights()

    result = features[["road_segment_id", "datetime_hour"]].copy()
    result = result.rename(columns={"datetime_hour": "prediction_datetime"})
    result["prediction_datetime"] = pd.Timestamp(prediction_datetime).floor("h")
    result["road_segment_id"] = result["road_segment_id"].astype(str)
    result["score_catboost_stage19h"] = cat_scores
    result["score_hist_gradient_boosting"] = hgb_scores
    # Percentiles are calculated within this one complete prediction hour.
    result["_cat_percentile"] = result["score_catboost_stage19h"].rank(pct=True)
    result["_hgb_percentile"] = result["score_hist_gradient_boosting"].rank(pct=True)
    result["dynamic_score"] = cat_weight * result["_cat_percentile"] + hgb_weight * result["_hgb_percentile"]
    result = result.sort_values("road_segment_id", kind="stable").reset_index(drop=True)
    result["dynamic_rank"] = result["dynamic_score"].rank(method="first", ascending=False).astype("int32")
    result["dynamic_percentile"] = result["dynamic_score"].rank(method="average", pct=True)
    result["dynamic_engine_version"] = ENGINE_VERSION
    result["dynamic_engine_status"] = ENGINE_STATUS
    return result.drop(columns=["_cat_percentile", "_hgb_percentile"])


def validate_dynamic_risk(frame: pd.DataFrame, expected_segments: int = EXPECTED_SEGMENTS) -> None:
    """Fail closed when the stable Stage 20A dynamic output contract is broken."""

    required = {
        "road_segment_id", "prediction_datetime", "dynamic_score", "dynamic_rank",
        "dynamic_percentile", "dynamic_engine_version", "dynamic_engine_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"stage20a_dynamic_missing_columns:{sorted(missing)}")
    if len(frame) != expected_segments or frame.road_segment_id.nunique() != expected_segments:
        raise ValueError("stage20a_dynamic_segment_grain_invalid")
    if frame.duplicated(_KEY).any() or frame[list(required)].isna().any().any():
        raise ValueError("stage20a_dynamic_null_or_duplicate")
    if not np.isfinite(frame[["dynamic_score", "dynamic_percentile"]].to_numpy()).all():
        raise ValueError("stage20a_dynamic_non_finite")
    if set(frame.dynamic_rank) != set(range(1, expected_segments + 1)):
        raise ValueError("stage20a_dynamic_rank_invalid")


def _historical_hotspot(prediction_datetime: str | pd.Timestamp, segments: pd.Series) -> pd.DataFrame:
    """Return strict-prior all-time and rolling historical accident counts."""

    moment = _local_model_hour(prediction_datetime)
    accidents = pd.read_parquet(
        ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet",
        columns=["road_segment_id", "accident_datetime"],
    )
    accidents["road_segment_id"] = accidents["road_segment_id"].astype(str)
    accidents["event_hour"] = pd.to_datetime(accidents["accident_datetime"]).dt.floor("h")
    prior = accidents.loc[accidents["event_hour"] < moment]
    result = pd.DataFrame({"road_segment_id": segments.astype(str)})
    for name, start in (
        ("historical_accident_count", None),
        ("historical_accident_count_30d", moment - pd.Timedelta(days=30)),
        ("historical_accident_count_90d", moment - pd.Timedelta(days=90)),
        ("historical_accident_count_365d", moment - pd.Timedelta(days=365)),
    ):
        source = prior if start is None else prior.loc[prior["event_hour"] >= start]
        result[name] = result["road_segment_id"].map(source["road_segment_id"].value_counts()).fillna(0).astype("int32")
    result["historical_hotspot_score"] = result["historical_accident_count"].astype(float)
    result = result.sort_values("road_segment_id", kind="stable").reset_index(drop=True)
    result["historical_hotspot_rank"] = result["historical_hotspot_score"].rank(method="first", ascending=False).astype("int32")
    result["historical_hotspot_percentile"] = result["historical_hotspot_score"].rank(method="average", pct=True)
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
        "future_context_warnings": "[\"future_context_unavailable\"]",
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

    result["weather_context_available"] = enabled("weather_provider_available", "weather_available")
    result["traffic_context_available"] = enabled("traffic_context_available", "traffic_provider_available")
    result["repair_context_available"] = enabled("gov_provider_available", "repair_provider_available")
    result["event_context_available"] = enabled("ticketon_provider_available", "event_provider_available")
    degraded = enabled("provider_degraded") if "provider_degraded" in source else ~(result[["weather_context_available", "traffic_context_available", "repair_context_available", "event_context_available"]].any(axis=1))
    result["provider_degraded"] = degraded

    flags: list[str] = []
    for _, row in result.iterrows():
        row_flags = []
        original = source.loc[row["road_segment_id"]] if row["road_segment_id"] in source.index else pd.Series(dtype=object)
        if row["weather_context_available"] and float(original.get("weather_severity_score", 0) or 0) >= 0.6:
            row_flags.append("severe_weather")
        if row["traffic_context_available"] and float(original.get("traffic_congestion_score", 0) or 0) >= 0.7:
            row_flags.append("heavy_traffic")
        if float(original.get("repair_active_next_24h", 0) or 0) > 0:
            row_flags.append("road_repair")
        if float(original.get("event_major_next_24h", 0) or 0) > 0:
            row_flags.append("major_event")
        flags.append(json.dumps(row_flags))
    result["future_context_flags"] = flags
    result["future_context_warnings"] = result["provider_degraded"].map(lambda value: "[\"provider_degraded\"]" if value else "[]")
    result["future_context_confidence"] = result["provider_degraded"].map(lambda value: "degraded" if value else "available")
    return result


def build_hybrid_risk(
    prediction_datetime: str | pd.Timestamp, *, future_context: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the complete one-row-per-segment Stage 20A operational contract."""

    # Stage 18B is the canonical future-context source.  Its absence is valid
    # operationally and is represented by explicit degraded fields below.
    if future_context is None and DEFAULT_FUTURE_CONTEXT_PATH.exists():
        future_context = pd.read_parquet(DEFAULT_FUTURE_CONTEXT_PATH)
    dynamic = build_dynamic_risk(prediction_datetime)
    historical = _historical_hotspot(prediction_datetime, dynamic["road_segment_id"])
    future = _future_context(
        dynamic["road_segment_id"], future_context, prediction_datetime
    )
    result = dynamic.merge(historical, on="road_segment_id", validate="one_to_one")
    result = result.merge(future, on="road_segment_id", validate="one_to_one")
    validate_hybrid_risk(result)
    return result


def validate_hybrid_risk(frame: pd.DataFrame, expected_segments: int = EXPECTED_SEGMENTS) -> None:
    validate_dynamic_risk(frame, expected_segments)
    required = {
        "historical_accident_count", "historical_accident_count_30d", "historical_accident_count_90d",
        "historical_accident_count_365d", "historical_hotspot_score", "historical_hotspot_rank",
        "historical_hotspot_percentile", "weather_context_available", "traffic_context_available",
        "repair_context_available", "event_context_available", "future_context_flags",
        "future_context_warnings", "future_context_confidence", "provider_degraded",
    }
    missing = required - set(frame.columns)
    if missing or frame[list(required)].isna().any().any():
        raise ValueError(f"stage20a_hybrid_required_fields_invalid:{sorted(missing)}")
    if set(frame["historical_hotspot_rank"]) != set(range(1, expected_segments + 1)):
        raise ValueError("stage20a_historical_rank_invalid")
    if (frame[["historical_accident_count", "historical_accident_count_30d", "historical_accident_count_90d", "historical_accident_count_365d"]] < 0).any().any():
        raise ValueError("stage20a_historical_count_invalid")
