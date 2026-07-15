from __future__ import annotations

import unittest

import pandas as pd

from ml_service.hybrid_risk import (
    ENGINE_STATUS,
    ENGINE_VERSION,
    _future_context,
    _local_model_hour,
    validate_dynamic_risk,
    validate_hybrid_risk,
)


class HybridRiskTests(unittest.TestCase):
    def test_offset_timestamp_maps_to_archived_local_hour(self):
        self.assertEqual(
            _local_model_hour("2024-09-30T19:41:00+05:00"),
            pd.Timestamp("2024-09-30T19:00:00"),
        )

    def _valid(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "road_segment_id": ["3", "2", "1"],
                "prediction_datetime": ["2026-07-15T10:00:00+05:00"] * 3,
                "dynamic_score": [0.2, 0.5, 1.0],
                "dynamic_rank": [3, 2, 1],
                "dynamic_percentile": [1 / 3, 2 / 3, 1.0],
                "dynamic_engine_version": [ENGINE_VERSION] * 3,
                "dynamic_engine_status": [ENGINE_STATUS] * 3,
            }
        )

    def test_dynamic_contract_accepts_complete_unique_ranked_output(self):
        validate_dynamic_risk(self._valid(), expected_segments=3)

    def test_dynamic_contract_rejects_duplicate_segment(self):
        frame = self._valid()
        frame.loc[2, "road_segment_id"] = "2"
        with self.assertRaisesRegex(ValueError, "segment_grain_invalid"):
            validate_dynamic_risk(frame, expected_segments=3)

    def test_dynamic_contract_rejects_missing_rank(self):
        frame = self._valid()
        frame.loc[2, "dynamic_rank"] = 2
        with self.assertRaisesRegex(ValueError, "rank_invalid"):
            validate_dynamic_risk(frame, expected_segments=3)

    def test_missing_future_context_is_explicitly_degraded(self):
        context = _future_context(pd.Series(["1", "2", "3"]), None)
        self.assertTrue(context["provider_degraded"].all())
        self.assertTrue(
            context["future_context_warnings"].str.contains("unavailable").all()
        )

    def test_future_context_maps_provider_flags_and_signals(self):
        source = pd.DataFrame(
            {
                "road_segment_id": ["1", "2", "3"],
                "weather_provider_available": [1, 0, 0],
                "traffic_context_available": [1, 0, 0],
                "repair_provider_available": [1, 0, 0],
                "ticketon_provider_available": [1, 0, 0],
                "weather_severity_score": [0.8, 0.0, 0.0],
                "traffic_congestion_score": [0.9, 0.0, 0.0],
                "repair_active_next_24h": [1, 0, 0],
                "event_major_next_24h": [1, 0, 0],
            }
        )
        context = _future_context(pd.Series(["1", "2", "3"]), source)
        self.assertFalse(context.loc[0, "provider_degraded"])
        self.assertEqual(
            context.loc[0, "future_context_flags"],
            '["severe_weather", "heavy_traffic", "road_repair", "major_event"]',
        )

    def test_hybrid_contract_requires_historical_and_context_columns(self):
        frame = self._valid()
        for column, values in {
            "historical_accident_count": [2, 1, 0],
            "historical_accident_count_30d": [1, 1, 0],
            "historical_accident_count_90d": [2, 1, 0],
            "historical_accident_count_365d": [2, 1, 0],
            "historical_hotspot_score": [2.0, 1.0, 0.0],
            "historical_hotspot_rank": [1, 2, 3],
            "historical_hotspot_percentile": [1.0, 2 / 3, 1 / 3],
            "weather_context_available": [False] * 3,
            "traffic_context_available": [False] * 3,
            "repair_context_available": [False] * 3,
            "event_context_available": [False] * 3,
            "future_context_flags": ["[]"] * 3,
            "future_context_warnings": ['["provider_degraded"]'] * 3,
            "future_context_confidence": ["degraded"] * 3,
            "provider_degraded": [True] * 3,
        }.items():
            frame[column] = values
        validate_hybrid_risk(frame, expected_segments=3)


if __name__ == "__main__":
    unittest.main()
