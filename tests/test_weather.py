import unittest

from ml_service.weather import OpenWeatherService


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "dt": 1, "main": {"temp": 23.5, "humidity": 44, "pressure": 1005},
            "wind": {"speed": 4.2, "gust": 8.1}, "clouds": {"all": 10},
            "rain": {"1h": 0.3}, "weather": [{"id": 500, "description": "light rain"}],
        }


class FakeSession:
    def get(self, *args, **kwargs):
        return FakeResponse()


class WeatherTests(unittest.TestCase):
    def test_missing_key_is_a_safe_fallback(self):
        result = OpenWeatherService(api_key="").get_current()
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "openweather_api_key_not_configured")

    def test_current_weather_is_normalized(self):
        result = OpenWeatherService(api_key="test", session=FakeSession()).get_current()
        self.assertTrue(result["available"])
        self.assertEqual(result["temperature_2m"], 23.5)
        self.assertEqual(result["precipitation"], 0.3)
        self.assertIn("dew_point_2m", result)
        self.assertEqual(result["surface_pressure"], 1005.0)
