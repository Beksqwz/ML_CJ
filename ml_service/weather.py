"""OpenWeather current-conditions integration for Astana."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import requests

from .registry import ROOT


LOGGER = logging.getLogger(__name__)
CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
ASTANA = {"latitude": 51.1694, "longitude": 71.4491}


def _dew_point_celsius(temperature_c: float, humidity_percent: float) -> float:
    """Magnus approximation used when a current API has no dew-point field."""
    a, b = 17.62, 243.12
    gamma = math.log(max(humidity_percent, 0.1) / 100) + (a * temperature_c) / (b + temperature_c)
    return (b * gamma) / (a - gamma)


def _local_api_key() -> str | None:
    """Read the untracked key without loading unrelated environment values."""
    try:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "OPENWEATHER_API_KEY":
                return value.strip().strip('"').strip("'") or None
    except FileNotFoundError:
        pass
    return None


class OpenWeatherService:
    """Fetch current Astana weather and map it to the model's weather fields."""

    def __init__(self, api_key: str | None = None, *, timeout_seconds: float = 10.0,
                 session: requests.Session | None = None) -> None:
        self.api_key = api_key if api_key is not None else (
            os.getenv("OPENWEATHER_API_KEY") or _local_api_key()
        )
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _unavailable(reason: str) -> dict[str, Any]:
        return {"available": False, "reason": reason}

    def get_current(self) -> dict[str, Any]:
        """Return normalized current conditions, never exposing the API key."""
        if not self.configured:
            return self._unavailable("openweather_api_key_not_configured")
        try:
            response = self.session.get(
                CURRENT_WEATHER_URL,
                params={"lat": ASTANA["latitude"], "lon": ASTANA["longitude"],
                        "appid": self.api_key, "units": "metric"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            main, wind = payload["main"], payload.get("wind", {})
            rain, snow = payload.get("rain", {}), payload.get("snow", {})
            condition = (payload.get("weather") or [{}])[0]
            temperature = float(main["temp"])
            humidity = float(main["humidity"])
            return {
                "available": True,
                "source": "OpenWeather Current Weather API",
                "observed_at_unix": payload.get("dt"),
                "temperature_2m": temperature,
                "relative_humidity_2m": humidity,
                "dew_point_2m": _dew_point_celsius(temperature, humidity),
                "surface_pressure": float(main["pressure"]),
                # The current endpoint does not report observed sunshine seconds.
                "sunshine_duration": float("nan"),
                "precipitation": float(rain.get("1h", 0)) + float(snow.get("1h", 0)),
                "rain": float(rain.get("1h", 0)),
                "snowfall": float(snow.get("1h", 0)),
                "weather_code": str(condition.get("id", "0")),
                "cloud_cover": float(payload.get("clouds", {}).get("all", 0)),
                "wind_speed_10m": float(wind.get("speed", 0)),
                "wind_gusts_10m": float(wind.get("gust", wind.get("speed", 0))),
                "description": str(condition.get("description", "")),
            }
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            LOGGER.warning("OpenWeather current conditions unavailable: %s", exc)
            return self._unavailable("openweather_request_failed")
