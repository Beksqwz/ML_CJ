from __future__ import annotations
import unittest
from recommendations.city_action_plan import generate_city_action_plan

T = "2026-01-01T09:00:00+05:00"


def rows():
    r = [
        {
            "road_segment_id": str(i),
            "operational_priority": "low",
            "dynamic_percentile": 0,
            "historical_hotspot_percentile": 0,
            "reasons": [],
            "uncertainty": "low",
        }
        for i in range(3968)
    ]

    def put(i, **x):
        r[i].update(x)

    put(
        0,
        operational_priority="critical",
        dynamic_percentile=0.99,
        historical_hotspot_percentile=0.9,
        road_name="Main",
        reasons=["DYNAMIC_TOP_1PCT", "SEVERE_WEATHER"],
        weather_severity_score=1,
        weather_worst_period_start="w1",
        weather_worst_period_end="w2",
    )
    put(
        1,
        operational_priority="high",
        dynamic_percentile=0.95,
        historical_hotspot_percentile=0.8,
        road_name=" main ",
        reasons=["DYNAMIC_TOP_1PCT"],
    )
    put(
        2,
        operational_priority="high",
        dynamic_percentile=0.8,
        historical_hotspot_percentile=0.7,
        road_name="Main",
        reasons=["MAJOR_EVENT"],
        event_start="e1",
        event_end="e2",
    )
    put(
        3,
        operational_priority="medium",
        dynamic_percentile=0.7,
        historical_hotspot_percentile=0.6,
        road_ref="R1",
        reasons=["ROAD_REPAIR"],
        repair_start="r1",
        repair_end="r2",
    )
    put(
        4,
        operational_priority="medium",
        dynamic_percentile=0.7,
        historical_hotspot_percentile=0.6,
        reasons=["HEAVY_TRAFFIC"],
        traffic_start="t1",
        traffic_end="t2",
    )
    return r


class Core(unittest.TestCase):
    def plan(self, **k):
        return generate_city_action_plan(
            rows(), batch_id="b", prediction_datetime=T, **k
        )

    def test_analyzes_exactly_3968_segments(self):
        self.assertEqual(self.plan()["summary"]["segments_analyzed"], 3968)

    def test_candidate_selection_excludes_low_segments(self):
        p = self.plan()
        ids = {x for a in p["actions"] for x in a["location"]["segment_ids"]}
        self.assertIn("0", ids)
        self.assertNotIn("100", ids)

    def test_same_road_and_action_are_grouped(self):
        a = [
            a
            for a in self.plan(max_actions=50)["actions"]
            if a["action_code"] == "INCREASE_PATROL"
        ][0]
        self.assertEqual(set(a["location"]["segment_ids"]), {"0", "1"})

    def test_different_action_codes_are_not_merged(self):
        codes = {a["action_code"] for a in self.plan(max_actions=50)["actions"]}
        self.assertTrue(
            {
                "INCREASE_PATROL",
                "SPEED_MONITORING",
                "EVENT_TRAFFIC_CONTROL",
                "REPAIR_ZONE_SAFETY_REVIEW",
                "CONGESTION_MONITORING",
            }.issubset(codes)
        )

    def test_location_fallback_order(self):
        p = self.plan(max_actions=50)
        names = {a["location"]["display_name"] for a in p["actions"]}
        self.assertIn("Main", names)
        self.assertIn("R1", names)
        self.assertIn("участок дороги", names)

    def test_time_basis_selection(self):
        by = {
            a["action_code"]: a["recommended_period"]
            for a in self.plan(max_actions=50)["actions"]
        }
        self.assertEqual(by["EVENT_TRAFFIC_CONTROL"]["basis"], "event_period")
        self.assertEqual(by["REPAIR_ZONE_SAFETY_REVIEW"]["basis"], "repair_period")
        self.assertEqual(by["ROAD_SURFACE_CHECK"]["basis"], "weather_period")
        self.assertEqual(by["CONGESTION_MONITORING"]["basis"], "traffic_period")
        self.assertEqual(by["INCREASE_PATROL"]["basis"], "prediction_horizon")

    def test_priority_formula_top_n_and_determinism(self):
        p = self.plan(max_actions=1)
        a = p["actions"][0]
        self.assertAlmostEqual(
            a["action_priority_score"],
            0.4 * 0.99 + 0.25 * 0.9 + 0.2 + 0.1 + 0.05 * min(2 / 5, 1),
        )
        self.assertEqual(a["action_rank"], 1)
        self.assertEqual(p, self.plan(max_actions=1))
        empty = generate_city_action_plan([], batch_id="b", prediction_datetime=T)
        self.assertEqual(empty["actions"], [])

    def test_multilingual_safety_and_zero_external_calls(self):
        for a in self.plan(max_actions=50)["actions"]:
            self.assertTrue(all(a["text"][x] for x in ("ru", "kz", "en")))
            self.assertTrue(a["requires_human_confirmation"])
            self.assertNotIn("probability", str(a).lower())
