import unittest

from recommendations.engine import recommend


def run(shap, values):
    return recommend(
        probability=0.4,
        shap_values=shap,
        feature_values=values,
        model_horizon="1h",
        final_model_version="Stage 7D",
    )


class Stage8BEngineTests(unittest.TestCase):
    def test_twenty_rule_scenarios(self):
        scenarios = [
            (
                {"segment_accidents_total_prior": 0.1},
                {"segment_accidents_total_prior": 2},
                "historical_accidents",
            ),
            (
                {"weather_risk_snow_now": 0.1},
                {"weather_risk_snow_now": 1},
                "weather_hazard",
            ),
            (
                {"weather_precip_sum_prev_24h": 0.1},
                {"weather_precip_sum_prev_24h": 2},
                "weather_hazard",
            ),
            (
                {"weather_risk_freezing_now": 0.1},
                {"weather_risk_freezing_now": 1},
                "weather_hazard",
            ),
            (
                {"road_maxspeed_kmh": 0.1},
                {"road_maxspeed_kmh": 60, "road_maxspeed_missing": False},
                "confirmed_high_speed",
            ),
            (
                {"road_maxspeed_missing": 0.1},
                {"road_maxspeed_missing": True},
                "missing_speed_limit",
            ),
            (
                {"poi_transit_stop_500m": 0.1},
                {"poi_transit_stop_500m": 3},
                "nearby_poi",
            ),
            ({"poi_education_250m": 0.1}, {"poi_education_250m": 4}, "nearby_poi"),
            ({"road_oneway": 0.1}, {"road_oneway": True}, "oneway"),
            ({"road_length": 0.1}, {"road_length": 100}, "road_spatial"),
            ({"segment_latitude": 0.1}, {"segment_latitude": 51}, "road_spatial"),
            ({"segment_longitude": 0.1}, {"segment_longitude": 71}, "road_spatial"),
            ({"road_highway": 0.1}, {"road_highway": "primary"}, "road_spatial"),
            (
                {"calendar_is_rush_hour": 0.1},
                {"calendar_is_rush_hour": True},
                "positive_rush_holiday",
            ),
            (
                {"calendar_is_holiday": 0.1},
                {"calendar_is_holiday": True},
                "positive_rush_holiday",
            ),
            (
                {"segment_accidents_total_prior": -0.1},
                {"segment_accidents_total_prior": 3},
                None,
            ),
            ({"weather_risk_snow_now": -0.1}, {"weather_risk_snow_now": 1}, None),
            ({"road_maxspeed_kmh": 0.1}, {"road_maxspeed_kmh": 50}, None),
            ({"road_maxspeed_missing": 0.1}, {"road_maxspeed_missing": False}, None),
            ({"calendar_is_rush_hour": -0.1}, {"calendar_is_rush_hour": True}, None),
        ]
        for shap, values, expected in scenarios:
            with self.subTest(expected=expected, shap=shap):
                rules = {x["rule"] for x in run(shap, values)["recommendations"]}
                self.assertEqual(expected in rules, expected is not None)

    def test_every_recommendation_has_positive_shap_evidence(self):
        output = run(
            {"road_oneway": 0.2, "road_length": -0.2},
            {"road_oneway": True, "road_length": 100},
        )
        self.assertTrue(
            all(
                x["evidence"]["shap_value"] > 0 and x["human_review_required"]
                for x in output["recommendations"]
            )
        )

    def test_poi_evidence_must_be_the_feature_that_meets_the_threshold(self):
        output = run(
            {"poi_school_500m": 0.2, "poi_transit_stop_500m": -0.1},
            {"poi_school_500m": 2, "poi_transit_stop_500m": 6},
        )
        self.assertNotIn(
            "nearby_poi", {item["rule"] for item in output["recommendations"]}
        )

    def test_rationale_is_russian_noncausal_and_requires_review(self):
        text = run({"road_oneway": 0.2}, {"road_oneway": True})["recommendations"][0][
            "rationale"
        ]
        self.assertNotIn("\ufffd", text)
        self.assertTrue(any("\u0400" <= char <= "\u04ff" for char in text))
        self.assertIn(
            "\u043d\u0435 \u0434\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442",
            text.lower(),
        )
        self.assertIn(
            "\u0442\u0440\u0435\u0431\u0443\u0435\u0442 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 \u0441\u043f\u0435\u0446\u0438\u0430\u043b\u0438\u0441\u0442\u043e\u043c",
            text.lower(),
        )
