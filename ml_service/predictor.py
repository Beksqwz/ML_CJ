"""Backend-friendly façade over frozen models, local SHAP, and recommendation rules."""

from __future__ import annotations
from collections import Counter
import json
from typing import Any, Iterable
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError, Pool
from .inference.export_geojson import _geometry_map
from .inference.feature_builder import _config, build_features
from .inference.risk_thresholds import configured_risk_level, load_risk_thresholds
from recommendations.engine import recommend
from .exceptions import (
    ConfigNotFoundError,
    EmptySegmentListError,
    InvalidDatetimeError,
    ModelNotFoundError,
    UnknownRoadSegmentError,
)
from .registry import ModelRegistry, ROOT
from .traffic import TomTomTrafficService
from .utils import parse_datetime, validate_bbox
from .weather import OpenWeatherService


class AccidentRiskPredictor:
    """Keep final models in memory and serve JSON-ready road-risk predictions.

    Each unseen hour/horizon is built and scored once per instance. Existing
    feature construction, TreeSHAP, thresholds, and recommendation rules are
    reused unchanged.
    """

    def __init__(self) -> None:
        self.registry = ModelRegistry()
        try:
            self.thresholds = load_risk_thresholds()
            self.feature_configs = {
                horizon: _config(horizon)
                for horizon in self.registry.info()["models"]
            }
            calendar = pd.read_parquet(
                ROOT / "data" / "external" / "calendar_features_hourly.parquet",
                columns=["datetime_hour"],
            )
            weather = pd.read_parquet(
                ROOT / "data" / "external" / "weather_astana_hourly.parquet",
                columns=["datetime_hour"],
            )
            self.datetime_min = max(
                pd.Timestamp(calendar.datetime_hour.min()),
                pd.Timestamp(weather.datetime_hour.min()),
            )
            self.datetime_max = min(
                pd.Timestamp(calendar.datetime_hour.max()),
                pd.Timestamp(weather.datetime_hour.max()),
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ConfigNotFoundError(
                "Required feature, calendar, weather, or threshold configuration is unavailable."
            ) from exc
        self.models: dict[str, CatBoostClassifier] = {}
        self._cache: dict[tuple[str, str, bool], tuple[list[dict[str, Any]], pd.DataFrame]] = {}
        self._geometry = _geometry_map(ROOT / "data" / "roads" / "astana_edges.csv")
        for horizon in self.registry.info()["models"]:
            entry = self.registry.get(horizon)
            model = CatBoostClassifier()
            try:
                model.load_model(entry["resolved_path"])
            except (CatBoostError, OSError, RuntimeError) as exc:
                raise ModelNotFoundError(
                    f"Unable to load final model: {entry['resolved_path']}"
                ) from exc
            self.models[horizon] = model

    def _score_city(
        self,
        datetime_hour: str,
        horizon: str,
        *,
        weather_override: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], pd.DataFrame]:
        when = (
            parse_datetime(datetime_hour)
            if weather_override is not None
            else parse_datetime(datetime_hour, self.datetime_min, self.datetime_max)
        )
        key = (str(when), horizon, weather_override is not None)
        if key in self._cache:
            return self._cache[key]
        entry = self.registry.get(horizon)
        try:
            data, _ = build_features(when, horizon, weather_override=weather_override)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ConfigNotFoundError(
                "Feature construction configuration is unavailable for this prediction."
            ) from exc
        cfg = self.feature_configs[horizon]
        features = list(cfg["numerical_features"]) + list(cfg["categorical_features"])
        categories = list(cfg["categorical_features"])
        model = self.models[horizon]
        if model.feature_names_ != features or len(features) != entry["feature_count"]:
            raise InvalidDatetimeError(
                "Model feature contract does not match its final registry entry."
            )
        x = data[features].copy()
        for column in categories:
            x[column] = x[column].astype("string").fillna("__MISSING__").astype(str)
        pool = Pool(x, cat_features=categories, feature_names=features)
        probabilities = model.predict_proba(pool)[:, 1]
        shap = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
        records: list[dict[str, Any]] = []
        for index, row in data.reset_index(drop=True).iterrows():
            local = {
                feature: float(shap[index, position])
                for position, feature in enumerate(features)
            }
            values = {
                feature: (None if pd.isna(row[feature]) else row[feature])
                for feature in features
            }
            explanation = recommend(
                probability=float(probabilities[index]),
                shap_values=local,
                feature_values=values,
                model_horizon=horizon,
                final_model_version=entry["stage"],
            )
            records.append(
                {
                    "road_segment_id": str(row.road_segment_id),
                    "road_name": "UNKNOWN"
                    if pd.isna(row.road_name)
                    else str(row.road_name),
                    "risk_probability": float(probabilities[index]),
                    "risk_level": configured_risk_level(
                        float(probabilities[index]), self.thresholds
                    ),
                    "model_horizon": horizon,
                    "top_positive_factors": explanation["top_positive_factors"],
                    "top_negative_factors": explanation["top_negative_factors"],
                    "recommendations": explanation["recommendations"],
                    "feature_values": values,
                    "final_model_version": entry["stage"],
                    "longitude": float(row.segment_longitude) if "segment_longitude" in row else None,
                    "latitude": float(row.segment_latitude) if "segment_latitude" in row else None,
                }
            )
        self._cache[key] = (records, data.reset_index(drop=True))
        return self._cache[key]

    def _response(
        self,
        records: list[dict[str, Any]],
        frame: pd.DataFrame,
        datetime_hour: str,
        horizon: str,
    ) -> dict[str, Any]:
        features = []
        for record in records:
            geometry = self._geometry.get(record["road_segment_id"])
            if geometry is None:
                raise UnknownRoadSegmentError(
                    f"Missing geometry: {record['road_segment_id']}"
                )
            properties = {
                key: record[key]
                for key in (
                    "road_segment_id",
                    "road_name",
                    "risk_probability",
                    "risk_level",
                    "model_horizon",
                    "top_positive_factors",
                    "recommendations",
                )
            }
            features.append(
                {"type": "Feature", "geometry": geometry, "properties": properties}
            )
        counts = Counter(record["risk_level"] for record in records)
        return {
            "datetime_hour": str(parse_datetime(datetime_hour)),
            "model_horizon": horizon,
            "predictions": records,
            "geojson": {"type": "FeatureCollection", "features": features},
            "summary": {"segments": len(records), "risk_level_counts": dict(counts)},
        }

    def predict_city(self, datetime_hour: str, horizon: str) -> dict[str, Any]:
        """Return all city segments with probabilities, explanations, recommendations, and GeoJSON."""
        records, frame = self._score_city(datetime_hour, horizon)
        return self._response(records, frame, datetime_hour, horizon)

    def predict_segment(
        self, road_segment_id: str, datetime_hour: str, horizon: str
    ) -> dict[str, Any]:
        """Return one known road segment or raise an explicit unknown-segment error."""
        return self.predict_segments([road_segment_id], datetime_hour, horizon)

    def predict_segments(
        self, segment_ids: Iterable[str], datetime_hour: str, horizon: str
    ) -> dict[str, Any]:
        """Return a requested set of known segments while preserving request order."""
        records, frame = self._score_city(datetime_hour, horizon)
        lookup = {
            record["road_segment_id"]: (record, index)
            for index, record in enumerate(records)
        }
        requested = list(segment_ids)
        if not requested:
            raise EmptySegmentListError(
                "segment_ids must contain at least one known road segment."
            )
        missing = [segment for segment in requested if segment not in lookup]
        if missing:
            raise UnknownRoadSegmentError(f"Unknown road segment: {missing[0]}")
        chosen = [lookup[segment] for segment in requested]
        return self._response(
            [item[0] for item in chosen],
            frame.iloc[[item[1] for item in chosen]],
            datetime_hour,
            horizon,
        )

    def predict_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        datetime_hour: str,
        horizon: str,
    ) -> dict[str, Any]:
        """Return segments whose stored representative point lies inside a map bounding box."""
        bounds = validate_bbox(min_lon, min_lat, max_lon, max_lat)
        records, frame = self._score_city(datetime_hour, horizon)
        mask = frame.segment_longitude.between(
            bounds[0], bounds[2]
        ) & frame.segment_latitude.between(bounds[1], bounds[3])
        positions = list(frame.index[mask])
        return self._response(
            [records[i] for i in positions],
            frame.iloc[positions],
            datetime_hour,
            horizon,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Expose final model stages, feature counts, registry version, and display thresholds."""
        return self.registry.info() | {"risk_thresholds": self.thresholds}

    def get_future_context(
        self,
        prediction_datetime: str,
        horizon: str = "24h",
        providers: tuple[str, ...] = ("openweather",),
    ) -> dict[str, Any]:
        """Return independent future context; never pass it to frozen CatBoost models."""
        if horizon != "24h":
            return {
                "status": "degraded",
                "prediction_datetime": prediction_datetime,
                "horizon_hours": 0,
                "providers": [],
                "features": {},
                "coverage": {},
                "warnings": ["future_intelligence_supports_24h_only"],
                "fallback_used": True,
            }
        from future_intelligence.pipeline import FutureIntelligencePipeline

        return FutureIntelligencePipeline().collect(
            prediction_datetime, horizon_hours=24, providers=providers
        )

    def predict_segment_with_live_traffic(
        self,
        road_segment_id: str,
        datetime_hour: str,
        horizon: str,
        traffic: TomTomTrafficService | None = None,
    ) -> dict[str, Any]:
        """Return frozen model risk plus a separate, non-feature traffic reading."""
        response = self.predict_segment(road_segment_id, datetime_hour, horizon)
        response["live_traffic"] = (traffic or TomTomTrafficService()).get_segment(
            road_segment_id
        )
        return response

    def get_live_weather(
        self, weather: OpenWeatherService | None = None
    ) -> dict[str, Any]:
        """Return normalized current weather without exposing credentials."""
        return (weather or OpenWeatherService()).get_current()

    def predict_current_city(
        self, horizon: str, weather: OpenWeatherService | None = None
    ) -> dict[str, Any]:
        """Score the current Astana hour through the production feature builder."""
        live_weather = self.get_live_weather(weather)
        if not live_weather.get("available"):
            return {
                "live_weather": live_weather,
                "predictions": [],
                "summary": {"segments": 0},
            }
        from datetime import datetime
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Almaty")).replace(
            minute=0, second=0, microsecond=0, tzinfo=None
        )
        records, frame = self._score_city(
            str(now), horizon, weather_override=live_weather
        )
        response = self._response(records, frame, str(now), horizon)
        response["live_weather"] = live_weather
        response["weather_mode"] = "openweather_current"
        return response

    def healthcheck(self) -> dict[str, Any]:
        """Report readiness after the constructor has loaded every final model once."""
        models = self.registry.info()["models"]
        if len(self.models) != len(models):
            return {"status": "error"}
        return {
            "status": "ok",
            "models": {horizon: item["stage"] for horizon, item in models.items()},
            "versions": {
                horizon: item["model_version"] for horizon, item in models.items()
            },
            "features": {
                horizon: item["feature_count"] for horizon, item in models.items()
            },
        }
