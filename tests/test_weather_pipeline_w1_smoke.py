"""W1 fixture smoke: canonical OpenWeather snapshot reaches 3,968 predictions."""

from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from ml_service.hybrid_risk import build_hybrid_risk
from recommendations.stage20b import recommend_stage20b
from ml_service.runtime_store import PredictionStore
from future_intelligence.weather_snapshot import (
    canonical_snapshot,
    select_origin_weather,
    summarize_24h,
)


class WeatherPipelineW1Smoke(unittest.TestCase):
    def test_fixture_origin_and_prediction(self):
        t = pd.Timestamp("2026-07-15T01:30:00+05:00")
        start = t - pd.Timedelta(hours=1.5)
        points = []
        for i in range(10):
            at = start + pd.Timedelta(hours=3 * i)
            severe = i == 6
            points.append(
                {
                    "forecast_timestamp": at.isoformat(),
                    "temperature": 20 + i,
                    "humidity": 50 + i,
                    "pressure": 1000 + i,
                    "wind_speed": 15 if severe else 3,
                    "wind_gust": 18 if severe else 4,
                    "rain": 3 if severe else 0,
                    "snow": 0,
                    "visibility": 500 if severe else 10000,
                    "clouds": 20,
                    "weather_main": "Rain" if severe else "Clear",
                }
            )
        snapshot = canonical_snapshot(points, "2026-07-15T00:00:00Z", t.isoformat())
        origin = select_origin_weather(snapshot, t.isoformat())
        summary = summarize_24h(snapshot, t.isoformat())
        self.assertTrue(origin["interpolated"])
        self.assertEqual(summary["forecast_points_available"], 8)
        self.assertTrue(summary["severe_weather_expected"])
        # Real scorer receives compact persisted-origin fields; no aggregate is present.
        context = pd.DataFrame(
            {
                "road_segment_id": pd.read_parquet(
                    "data/processed/accidents_with_roads_ml_ready.parquet",
                    columns=["road_segment_id"],
                )
                .road_segment_id.astype(str)
                .drop_duplicates()
                .sort_values()
                .tolist(),
                "prediction_datetime": [t.isoformat()] * 3968,
                "weather_provider_available": [1] * 3968,
                "weather_snapshot_version": [snapshot["snapshot_version"]] * 3968,
                "weather_origin_temperature": [origin["temperature"]] * 3968,
                "weather_origin_humidity": [origin["humidity"]] * 3968,
                "weather_origin_wind_speed": [origin["wind_speed"]] * 3968,
                "weather_origin_rain": [origin["rain"]] * 3968,
                "weather_origin_visibility": [origin["visibility"]] * 3968,
                "weather_origin_source_before": [origin["source_before"]] * 3968,
                "weather_origin_source_after": [origin["source_after"]] * 3968,
                "weather_origin_interpolated": [True] * 3968,
                "weather_severity_score": [summary["max_weather_severity_score"]]
                * 3968,
                "weather_severe_weather_expected": [True] * 3968,
                "weather_forecast_points_available": [9] * 3968,
                "weather_forecast_start": [summary["forecast_start"]] * 3968,
                "weather_forecast_end": [summary["forecast_end"]] * 3968,
                "weather_worst_period_start": [summary["worst_period_start"]] * 3968,
                "weather_worst_period_end": [summary["worst_period_end"]] * 3968,
                "provider_degraded": [0] * 3968,
            }
        )
        result = recommend_stage20b(
            build_hybrid_risk(t.isoformat(), future_context=context)
        )
        self.assertEqual((len(result), result.road_segment_id.nunique()), (3968, 3968))
        self.assertIn("SEVERE_WEATHER", result.iloc[0].reasons)
        p = Path(tempfile.mkdtemp()) / "x.sqlite3"
        store = PredictionStore(p)
        store.save_completed(
            {
                "batch_id": "b",
                "prediction_datetime": t.isoformat(),
                "started_at": "x",
                "completed_at": "y",
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            result,
        )
        self.assertEqual(
            PredictionStore(p).segment(str(result.iloc[0].road_segment_id))[
                "road_segment_id"
            ],
            str(result.iloc[0].road_segment_id),
        )
