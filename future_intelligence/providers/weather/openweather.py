"""OpenWeather 5 day / 3 hour forecast provider.

Uses the official `/data/2.5/forecast` endpoint. All features are future
retraining candidates: this module never calls CatBoost or mutates its input.
Thresholds are centralized in WEATHER_THRESHOLDS. Scores count forecast points
meeting a condition (rather than inventing hourly observations).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import time
from typing import Any, Callable

import requests

from future_intelligence.aggregation import numeric, summary
from future_intelligence.providers.weather.base import WeatherForecastProvider
from future_intelligence.schemas import FutureRecord, ProviderMetadata, ProviderResult
from future_intelligence.utils import api_timezone, parse_prediction_datetime
from ml_service.weather import ASTANA, OpenWeatherService


FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
WEATHER_THRESHOLDS = {
    "fog_visibility_m": 1000.0,
    "heavy_rain_mm": 2.5,
    "strong_wind_mps": 10.0,
    "freezing_c": 0.0,
    "storm_weather_ids": (200, 201, 202, 210, 211, 212, 221, 230, 231, 232),
}


class OpenWeatherForecastProvider(WeatherForecastProvider):
    metadata = ProviderMetadata(
        provider_name="openweather",
        provider_version="1.0",
        source_type="weather_forecast",
        supported_horizons=(24,),
        requires_api_key=True,
        update_frequency="OpenWeather 5-day/3-hour forecast refresh",
        spatial_scope="city point (Astana by default)",
    )

    def __init__(
        self,
        service: OpenWeatherService | None = None,
        *,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # Compose the existing client so .env/API-key behavior stays single-sourced.
        self.service = service or OpenWeatherService(timeout_seconds=timeout_seconds)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.service.configured else "degraded",
            "configured": self.service.configured,
            "provider": self.metadata.provider_name,
        }

    def _degraded(
        self, prediction_datetime: datetime, horizon_hours: int, warning: str
    ) -> ProviderResult:
        return ProviderResult(
            metadata=self.metadata,
            raw_records=[],
            normalized_records=[],
            features={},
            coverage={
                "point_count": 0,
                "window_start": prediction_datetime.isoformat(),
                "window_end": (
                    prediction_datetime + timedelta(hours=horizon_hours)
                ).isoformat(),
            },
            warnings=[warning],
            status="degraded",
            fallback_used=True,
        )

    def _request(
        self, latitude: float, longitude: float
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not self.service.configured:
            return None, "openweather_api_key_not_configured"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.service.session.get(
                    FORECAST_URL,
                    params={
                        "lat": latitude,
                        "lon": longitude,
                        "appid": self.service.api_key,
                        "units": "metric",
                    },
                    timeout=self.timeout_seconds,
                )
                status = getattr(response, "status_code", 200)
                if status == 429:
                    if attempt < self.max_retries:
                        self.sleep(self.backoff_seconds * (2**attempt))
                        continue
                    return None, "openweather_rate_limited"
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("list"), list
                ):
                    return None, "openweather_invalid_response"
                return payload, None
            except (requests.Timeout, requests.RequestException, ValueError, TypeError):
                if attempt < self.max_retries:
                    self.sleep(self.backoff_seconds * (2**attempt))
                    continue
                return None, "openweather_request_failed"
        return None, "openweather_request_failed"

    def collect(
        self,
        prediction_datetime: datetime,
        horizon_hours: int,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> ProviderResult:
        del bbox
        prediction_datetime = parse_prediction_datetime(prediction_datetime)
        if horizon_hours != 24:
            return self._degraded(
                prediction_datetime, horizon_hours, "openweather_supports_24h_only"
            )
        latitude = ASTANA["latitude"] if latitude is None else latitude
        longitude = ASTANA["longitude"] if longitude is None else longitude
        payload, warning = self._request(latitude, longitude)
        if payload is None:
            return self._degraded(prediction_datetime, horizon_hours, str(warning))
        records = self.normalize(payload, prediction_datetime, horizon_hours)
        features = self.build_features(records, prediction_datetime, horizon_hours)
        return ProviderResult(
            metadata=self.metadata,
            raw_records=[payload],
            normalized_records=records,
            features=features,
            coverage={
                "point_count": len(records),
                "window_start": prediction_datetime.isoformat(),
                "window_end": (
                    prediction_datetime + timedelta(hours=horizon_hours)
                ).isoformat(),
                "latitude": latitude,
                "longitude": longitude,
            },
            warnings=[],
            status="ok",
            fallback_used=False,
        )

    def normalize(
        self,
        raw_payload: dict[str, Any],
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> list[FutureRecord]:
        start = parse_prediction_datetime(prediction_datetime)
        end = start + timedelta(hours=horizon_hours)
        city = raw_payload.get("city", {})
        offset = api_timezone(city.get("timezone"))
        collected_at = datetime.now(UTC)
        records: list[FutureRecord] = []
        for item in raw_payload.get("list", []):
            timestamp = datetime.fromtimestamp(int(item["dt"]), tz=UTC).astimezone(
                offset
            )
            if not start <= timestamp < end:
                continue
            main, wind = item.get("main", {}), item.get("wind", {})
            rain, snow = item.get("rain", {}), item.get("snow", {})
            weather = (item.get("weather") or [{}])[0]
            payload = {
                "forecast_timestamp": timestamp.isoformat(),
                "temperature": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "temperature_min": main.get("temp_min"),
                "temperature_max": main.get("temp_max"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "sea_level": main.get("sea_level"),
                "ground_level": main.get("grnd_level"),
                "visibility": item.get("visibility"),
                "wind_speed": wind.get("speed"),
                "wind_direction": wind.get("deg"),
                "wind_gust": wind.get("gust"),
                "clouds": item.get("clouds", {}).get("all"),
                "rain": rain.get("3h", rain.get("1h")),
                "snow": snow.get("3h", snow.get("1h")),
                "precipitation_probability": item.get("pop"),
                "weather_main": weather.get("main"),
                "weather_description": weather.get("description"),
                "weather_id": weather.get("id"),
                "sunrise": city.get("sunrise"),
                "sunset": city.get("sunset"),
                "timezone": city.get("timezone"),
            }
            records.append(
                FutureRecord(
                    source="OpenWeather",
                    source_type=self.metadata.source_type,
                    source_version=self.metadata.provider_version,
                    source_item_id=str(item.get("dt")),
                    source_url=FORECAST_URL,
                    collected_at=collected_at,
                    published_at=None,
                    valid_from=timestamp,
                    valid_to=timestamp + timedelta(hours=3),
                    prediction_datetime=start,
                    horizon_hours=horizon_hours,
                    latitude=city.get("coord", {}).get("lat"),
                    longitude=city.get("coord", {}).get("lon"),
                    event_type="weather_forecast",
                    confidence=None,
                    payload=payload,
                )
            )
        return records

    def build_features(
        self,
        normalized_records: list[FutureRecord],
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> dict[str, Any]:
        del horizon_hours
        if not normalized_records:
            return {}
        rows = [record.payload for record in normalized_records]

        def values(name: str) -> list[float]:
            return numeric(row.get(name) for row in rows)

        features: dict[str, Any] = {}
        for source, name in (
            ("temperature", "weather_temperature"),
            ("humidity", "weather_humidity"),
            ("pressure", "weather_pressure"),
            ("wind_speed", "weather_wind_speed"),
            ("visibility", "weather_visibility"),
            ("clouds", "weather_clouds"),
            ("precipitation_probability", "weather_precipitation_probability"),
        ):
            features.update(summary(values(source), name))
        features["weather_temperature_range"] = (
            None
            if not values("temperature")
            else max(values("temperature")) - min(values("temperature"))
        )
        features["weather_wind_gust_max"] = max(values("wind_gust"), default=None)
        features["weather_rain_sum"] = sum(values("rain")) if values("rain") else 0.0
        features["weather_snow_sum"] = sum(values("snow")) if values("snow") else 0.0
        ids = [row.get("weather_id") for row in rows]
        t = WEATHER_THRESHOLDS
        features.update(
            {
                "weather_rain_hours": sum(value > 0 for value in values("rain")) * 3,
                "weather_snow_hours": sum(value > 0 for value in values("snow")) * 3,
                "weather_fog_hours": sum(
                    value is not None and value < t["fog_visibility_m"]
                    for value in (row.get("visibility") for row in rows)
                )
                * 3,
                "weather_heavy_rain_hours": sum(
                    value >= t["heavy_rain_mm"] for value in values("rain")
                )
                * 3,
                "weather_strong_wind_hours": sum(
                    value >= t["strong_wind_mps"] for value in values("wind_speed")
                )
                * 3,
                "weather_freezing_hours": sum(
                    value <= t["freezing_c"] for value in values("temperature")
                )
                * 3,
                "weather_storm_hours": sum(
                    value in t["storm_weather_ids"] for value in ids
                )
                * 3,
                "weather_change_count": sum(
                    ids[index] != ids[index - 1] for index in range(1, len(ids))
                ),
            }
        )
        features["weather_rapid_temperature_drop"] = any(
            values("temperature")[index - 1] - values("temperature")[index] >= 5
            for index in range(1, len(values("temperature")))
        )
        temperatures = values("temperature")
        features["weather_freeze_thaw_transition"] = bool(
            temperatures and min(temperatures) <= 0 < max(temperatures)
        )
        features["weather_is_rain_expected"] = features["weather_rain_hours"] > 0
        features["weather_is_snow_expected"] = features["weather_snow_hours"] > 0
        features["weather_is_fog_expected"] = features["weather_fog_hours"] > 0
        features["weather_is_freezing_expected"] = (
            features["weather_freezing_hours"] > 0
        )
        features["weather_is_strong_wind_expected"] = (
            features["weather_strong_wind_hours"] > 0
        )
        features["weather_is_storm_expected"] = features["weather_storm_hours"] > 0
        features["weather_instability_score"] = (
            features["weather_change_count"]
            + int(features["weather_rapid_temperature_drop"])
            + int(features["weather_freeze_thaw_transition"])
        )
        features["weather_bad_weather_score"] = sum(
            int(features[key])
            for key in (
                "weather_is_rain_expected",
                "weather_is_snow_expected",
                "weather_is_fog_expected",
                "weather_is_strong_wind_expected",
                "weather_is_storm_expected",
            )
        )
        features["weather_road_surface_risk_score"] = (
            int(features["weather_is_rain_expected"])
            + 2 * int(features["weather_is_snow_expected"])
            + int(features["weather_is_freezing_expected"])
        )
        features["weather_visibility_risk_score"] = int(
            features["weather_is_fog_expected"]
        )
        features["weather_winter_risk_score"] = (
            int(features["weather_is_snow_expected"])
            + int(features["weather_is_freezing_expected"])
            + int(features["weather_freeze_thaw_transition"])
        )
        features["weather_driving_conditions_score"] = (
            features["weather_bad_weather_score"]
            + features["weather_road_surface_risk_score"]
            + features["weather_visibility_risk_score"]
        )
        features["weather_severity_score"] = (
            features["weather_driving_conditions_score"]
            + features["weather_instability_score"]
        )
        first = rows[0]
        tz = api_timezone(first.get("timezone"))
        sunrise = first.get("sunrise")
        sunset = first.get("sunset")
        sunrise_dt = (
            datetime.fromtimestamp(sunrise, UTC).astimezone(tz) if sunrise else None
        )
        sunset_dt = (
            datetime.fromtimestamp(sunset, UTC).astimezone(tz) if sunset else None
        )
        features["weather_day_length_hours"] = (
            None
            if not sunrise_dt or not sunset_dt
            else (sunset_dt - sunrise_dt).total_seconds() / 3600
        )
        features["weather_dark_hours"] = (
            None
            if features["weather_day_length_hours"] is None
            else max(0.0, 24 - features["weather_day_length_hours"])
        )
        prediction_datetime = parse_prediction_datetime(prediction_datetime).astimezone(
            tz
        )
        features["weather_hours_until_sunrise"] = (
            None
            if not sunrise_dt
            else max(0.0, (sunrise_dt - prediction_datetime).total_seconds() / 3600)
        )
        features["weather_hours_until_sunset"] = (
            None
            if not sunset_dt
            else max(0.0, (sunset_dt - prediction_datetime).total_seconds() / 3600)
        )
        return features
