from __future__ import annotations

import unittest

import pandas as pd

from recommendations.stage20b import recommend_stage20b


def _row(**changes: object) -> dict[str, object]:
    row = {
        "road_segment_id": "1", "prediction_datetime": "2026-07-15T10:00:00+05:00",
        "dynamic_rank": 1, "dynamic_percentile": 0.995,
        "historical_hotspot_rank": 10, "historical_hotspot_percentile": 0.998,
        "future_context_flags": '["severe_weather", "heavy_traffic"]',
        "future_context_warnings": "[]", "provider_degraded": False,
    }
    return row | changes


class Stage20BRecommendationTests(unittest.TestCase):
    def test_multi_signal_critical_has_explainable_cautious_plans(self):
        result = recommend_stage20b(pd.DataFrame([_row()])).iloc[0]
        self.assertEqual(result.operational_priority, "critical")
        self.assertIn("DYNAMIC_TOP_1PCT", result.reasons)
        self.assertIn("MULTI_SIGNAL_AGREEMENT", result.reasons)
        self.assertEqual(set(result.possible_plan[0]), {"ru", "kz", "en"})
        self.assertIn("Рассмотреть", result.possible_plan[0]["ru"])

    def test_degraded_low_signal_segment_is_monitor_only(self):
        result = recommend_stage20b(pd.DataFrame([_row(
            dynamic_percentile=0.20, historical_hotspot_rank=300,
            historical_hotspot_percentile=0.20, future_context_flags="[]",
            future_context_warnings='["provider_degraded"]', provider_degraded=True,
        )])).iloc[0]
        self.assertEqual(result.operational_priority, "monitor_only")
        self.assertIn("PROVIDER_DEGRADED", result.reasons)
        self.assertIn("provider_degraded", result.warnings)
        self.assertEqual(result.uncertainty, "high")

    def test_priority_rank_is_deterministic(self):
        result = recommend_stage20b(pd.DataFrame([
            _row(road_segment_id="2", dynamic_percentile=0.60, historical_hotspot_rank=100),
            _row(road_segment_id="1"),
        ]))
        self.assertEqual(result.road_segment_id.tolist(), ["1", "2"])
        self.assertEqual(result.priority_rank.tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
