import unittest
from datetime import datetime, timedelta, timezone

import requests

from future_intelligence.providers.weather.openweather import (
    OpenWeatherForecastProvider,
)
from ml_service.weather import OpenWeatherService


WHEN = datetime(2026, 7, 14, 0, tzinfo=timezone(timedelta(hours=5)))


def fixture():
    def row(hour, temp, code=500, rain=0):
        return {
            "dt": int((WHEN + timedelta(hours=hour)).timestamp()),
            "main": {"temp": temp, "humidity": 60, "pressure": 1000},
            "visibility": 900 if hour == 3 else 5000,
            "wind": {"speed": 11 if hour == 3 else 4, "gust": 12},
            "clouds": {"all": 30},
            "rain": {"3h": rain},
            "pop": 0.5,
            "weather": [{"id": code, "main": "Rain", "description": "rain"}],
        }

    return {
        "city": {
            "timezone": 18000,
            "sunrise": int((WHEN + timedelta(hours=2)).timestamp()),
            "sunset": int((WHEN + timedelta(hours=16)).timestamp()),
            "coord": {"lat": 51.1, "lon": 71.4},
        },
        "list": [row(-3, 9), row(0, 8, 500, 3), row(3, 2, 201), row(23, 3), row(24, 4)],
    }


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class Session:
    def __init__(self, values):
        self.values = iter(values)

    def get(self, *args, **kwargs):
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


class OpenWeatherForecastTests(unittest.TestCase):
    def make(self, values):
        return OpenWeatherForecastProvider(
            OpenWeatherService(api_key="mock", session=Session(values)),
            sleep=lambda _: None,
        )

    def test_successful_collection_window_timezone_and_features(self):
        result = self.make([Response(fixture())]).collect(WHEN, 24)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.normalized_records), 3)
        self.assertTrue(
            all(
                WHEN <= record.valid_from < WHEN + timedelta(hours=24)
                for record in result.normalized_records
            )
        )
        self.assertEqual(result.features["weather_rain_sum"], 3.0)
        self.assertEqual(result.features["weather_strong_wind_hours"], 3)
        self.assertTrue(result.features["weather_is_storm_expected"])

    def test_missing_key_and_timeout_are_explicit_fallbacks(self):
        no_key = OpenWeatherForecastProvider(OpenWeatherService(api_key="")).collect(
            WHEN, 24
        )
        timeout = self.make(
            [requests.Timeout(), requests.Timeout(), requests.Timeout()]
        ).collect(WHEN, 24)
        self.assertTrue(no_key.fallback_used)
        self.assertIn("openweather_api_key_not_configured", no_key.warnings)
        self.assertTrue(timeout.fallback_used)
        self.assertIn("openweather_request_failed", timeout.warnings)

    def test_rate_limit_invalid_json_and_missing_optionals_are_safe(self):
        rate = self.make(
            [Response({}, 429), Response({}, 429), Response({}, 429)]
        ).collect(WHEN, 24)
        invalid = self.make(
            [
                Response(ValueError("bad")),
                Response(ValueError("bad")),
                Response(ValueError("bad")),
            ]
        ).collect(WHEN, 24)
        raw = fixture()
        raw["list"][1].pop("rain")
        raw["list"][1].pop("visibility")
        optional = self.make([Response(raw)]).collect(WHEN, 24)
        self.assertIn("openweather_rate_limited", rate.warnings)
        self.assertIn("openweather_request_failed", invalid.warnings)
        self.assertIn("weather_visibility_min", optional.features)
