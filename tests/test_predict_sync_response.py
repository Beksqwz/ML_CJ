"""Focused HTTP coverage for optional synchronous prediction payloads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

import api_service.app as api
from api_service.app import Runtime, _ordered_public_predictions, create_app
from ml_service.runtime_store import PredictionStore


def _rows(count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rank = index + 1
        rows.append(
            {
                "road_segment_id": f"segment-{index:04d}",
                "longitude": 71.4 + index / 1_000_000,
                "latitude": 51.15 + index / 1_000_000,
                "prediction_datetime": "2026-07-15T08:00:00+05:00",
                "dynamic_score": 1 - index / max(count, 1),
                "dynamic_rank": rank,
                "dynamic_percentile": 1 - index / max(count, 1),
                "dynamic_risk": {
                    "score": 1 - index / max(count, 1),
                    "score_type": "weighted_percentile_ensemble",
                    "rank": rank,
                    "percentile": 1 - index / max(count, 1),
                    "population_size": count,
                    "horizon_hours": 24,
                    "engine": "stage19i_ensemble",
                    "weights": {"catboost": 0.8, "hgb": 0.2},
                },
                "model_components": {
                    "catboost": {
                        "score": 0.8,
                        "score_type": "probability",
                        "percentile": 0.8,
                        "weight": 0.8,
                    },
                    "hgb": {
                        "score": 0.6,
                        "score_type": "probability",
                        "percentile": 0.6,
                        "weight": 0.2,
                    },
                },
                "operational_priority": "high" if index < 3 else "low",
                "priority_rank": rank,
                "historical_hotspot_rank": rank,
                "historical_hotspot_percentile": 0.5,
                "historical_accident_count": 2,
                "reasons": ["DYNAMIC_TOP_5PCT"] if index < 3 else [],
                "warnings": [],
                "uncertainty": "low",
                "possible_plan": [],
                "weather_context_available": True,
                "weather_severity_score": 0.7,
                "weather_provider": "openweather",
                "future_context_flags": '["severe_weather"]',
                "future_context_warnings": "[]",
                "future_context_confidence": "available",
                "provider_degraded": False,
                "weather_worst_period_start": "2026-07-15T12:00:00+05:00",
                "weather_worst_period_end": "2026-07-15T15:00:00+05:00",
                "event_source_id": "event-1",
                "event_name": "Concert",
                "event_venue": "Arena",
                "event_start": "2026-07-15T19:00:00+05:00",
                "event_end": "2026-07-15T22:00:00+05:00",
                "repair_source_id": "repair-1",
                "repair_title": "Lane closure",
                "repair_road_name": "Main street",
                "repair_start": "2026-07-15T08:00:00+05:00",
                "repair_end": "2026-07-15T17:00:00+05:00",
                "explanation": {
                    "explanation_status": "available",
                    "method": "shap",
                    "scope": "catboost_component_only",
                    "component_weight": 0.8,
                    "base_value": 0.1,
                    "top_positive_factors": [
                        {
                            "feature": "road_length",
                            "display_name": {
                                "ru": "Длина участка",
                                "kz": "Учаске ұзындығы",
                                "en": "Segment length",
                            },
                            "feature_value": 100,
                            "shap_value": 0.3,
                        },
                        {"feature": "rush_hour", "feature_value": 1, "shap_value": 0.2},
                        {"feature": "weather", "feature_value": 1, "shap_value": 0.1},
                    ],
                    "top_negative_factors": [
                        {"feature": "history", "feature_value": 0, "shap_value": -0.3},
                        {"feature": "lanes", "feature_value": 3, "shap_value": -0.2},
                        {"feature": "speed", "feature_value": 60, "shap_value": -0.1},
                    ],
                    "disclaimer": {"ru": "x", "kz": "x", "en": "x"},
                    "text": {"ru": "x", "kz": "x", "en": "x"},
                },
            }
        )
    return pd.DataFrame(rows)


class PredictSyncResponseTests(unittest.TestCase):
    headers = {"X-API-Key": "test-key"}
    request_body = {"prediction_datetime": "2026-07-15T08:00:00+05:00"}

    def _client(self, rows: int = 3):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = PredictionStore(Path(temporary.name) / "runtime.sqlite3")
        calls = {"predict": 0}
        frame = _rows(rows)

        def predictor(_: str) -> pd.DataFrame:
            calls["predict"] += 1
            return frame.copy(deep=True)

        runtime = Runtime(predictor=predictor, store=store)
        return TestClient(create_app(api_key="test-key", runtime=runtime)), store, calls

    def _post(self, client: TestClient, **body: object):
        return client.post(
            "/api/v1/predict", headers=self.headers, json=self.request_body | body
        )

    def test_01_default_request_is_compact(self):
        client, _, _ = self._client()
        response = self._post(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["responseMode"], "compact")
        self.assertFalse(response.json()["predictionsIncluded"])
        self.assertNotIn("predictions", response.json())

    def test_02_explicit_compact_has_no_predictions(self):
        client, _, _ = self._client()
        response = self._post(client, response_mode="compact")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("predictions", response.json())

    def test_03_full_returns_predictions(self):
        client, _, _ = self._client()
        response = self._post(client, response_mode="full")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["predictions"]), 3)

    def test_04_full_returns_3968_unique_segments(self):
        client, _, _ = self._client(3968)
        response = self._post(client, response_mode="full")
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["summary"]["segmentsPredicted"], 3968)
        self.assertEqual(
            len({row["road_segment_id"] for row in payload["predictions"]}), 3968
        )

    def test_05_full_order_is_dynamic_rank(self):
        client, _, _ = self._client()
        response = self._post(client, response_mode="full")
        self.assertEqual(
            [row["dynamic_rank"] for row in response.json()["predictions"]], [1, 2, 3]
        )

    def test_06_http_rows_match_persisted_rows(self):
        client, store, _ = self._client()
        payload = self._post(client, response_mode="full").json()
        rows = store.get_prediction_segments_for_batch(payload["batchId"])
        self.assertEqual(
            payload["predictions"],
            _ordered_public_predictions(
                rows, include_explanations=True, max_explanation_factors=3
            ),
        )

    def test_07_dynamic_ensemble_is_primary_score(self):
        client, _, _ = self._client()
        row = self._post(client, response_mode="full").json()["predictions"][0]
        self.assertEqual(row["dynamic_risk"]["score"], row["dynamic_score"])
        self.assertEqual(
            row["dynamic_risk"]["score_type"], "weighted_percentile_ensemble"
        )

    def test_08_nested_dynamic_values_match_flat_values(self):
        client, _, _ = self._client()
        row = self._post(client, response_mode="full").json()["predictions"][0]
        self.assertEqual(row["dynamic_risk"]["rank"], row["dynamic_rank"])
        self.assertEqual(row["dynamic_risk"]["percentile"], row["dynamic_percentile"])

    def test_09_component_metadata_is_preserved(self):
        client, _, _ = self._client()
        components = self._post(client, response_mode="full").json()["predictions"][0][
            "model_components"
        ]
        self.assertEqual(
            (components["catboost"]["weight"], components["hgb"]["weight"]), (0.8, 0.2)
        )

    def test_10_final_score_has_no_probability_name(self):
        client, _, _ = self._client()
        row = self._post(client, response_mode="full").json()["predictions"][0]
        self.assertNotIn("risk_probability", row)
        self.assertNotIn("accident_probability", row)

    def test_11_explanations_are_included_by_default_in_full(self):
        client, _, _ = self._client()
        explanation = self._post(client, response_mode="full").json()["predictions"][0][
            "explanation"
        ]
        self.assertEqual(explanation["scope"], "catboost_component_only")
        self.assertTrue(explanation["top_positive_factors"])

    def test_12_explanations_can_be_excluded(self):
        client, _, _ = self._client()
        explanation = self._post(
            client, response_mode="full", include_explanations=False
        ).json()["predictions"][0]["explanation"]
        self.assertEqual(explanation["explanation_status"], "excluded_by_request")
        self.assertNotIn("top_positive_factors", explanation)

    def test_13_zero_factor_limit_returns_empty_lists(self):
        client, _, _ = self._client()
        explanation = self._post(
            client, response_mode="full", max_explanation_factors=0
        ).json()["predictions"][0]["explanation"]
        self.assertEqual(
            (explanation["top_positive_factors"], explanation["top_negative_factors"]),
            ([], []),
        )

    def test_14_one_factor_limit_truncates_both_lists(self):
        client, _, _ = self._client()
        explanation = self._post(
            client, response_mode="full", max_explanation_factors=1
        ).json()["predictions"][0]["explanation"]
        self.assertLessEqual(len(explanation["top_positive_factors"]), 1)
        self.assertLessEqual(len(explanation["top_negative_factors"]), 1)

    def test_15_three_factor_limit_preserves_top_three_maximum(self):
        client, _, _ = self._client()
        explanation = self._post(
            client, response_mode="full", max_explanation_factors=3
        ).json()["predictions"][0]["explanation"]
        self.assertEqual(len(explanation["top_positive_factors"]), 3)
        self.assertEqual(len(explanation["top_negative_factors"]), 3)

    def test_16_invalid_response_mode_is_rejected(self):
        client, _, _ = self._client()
        self.assertEqual(self._post(client, response_mode="other").status_code, 422)

    def test_17_invalid_factor_limit_is_rejected(self):
        client, _, _ = self._client()
        self.assertEqual(
            self._post(
                client, response_mode="full", max_explanation_factors=4
            ).status_code,
            422,
        )
        self.assertEqual(
            self._post(
                client, response_mode="full", max_explanation_factors=-1
            ).status_code,
            422,
        )

    def test_18_full_response_excludes_full_feature_vector(self):
        client, _, _ = self._client()
        payload = self._post(client, response_mode="full").text
        self.assertNotIn("frozen_feature_vector", payload)

    def test_19_full_response_excludes_full_shap_vector(self):
        client, _, _ = self._client()
        self.assertNotIn("shap_values", self._post(client, response_mode="full").text)

    def test_20_full_response_excludes_sensitive_values(self):
        client, _, _ = self._client()
        self.assertNotIn(
            "raw_provider_payload", self._post(client, response_mode="full").text
        )

    def test_21_prediction_runs_once(self):
        client, _, calls = self._client()
        self._post(client, response_mode="full")
        self.assertEqual(calls["predict"], 1)

    def test_22_full_mode_does_not_call_hybrid_again(self):
        client, _, _ = self._client()
        with patch.object(
            api, "build_hybrid_risk", side_effect=AssertionError("called")
        ):
            self.assertEqual(self._post(client, response_mode="full").status_code, 200)

    def test_23_full_mode_does_not_refresh_providers(self):
        client, _, _ = self._client()
        with patch.object(
            api, "recommend_stage20b", side_effect=AssertionError("called")
        ):
            self.assertEqual(self._post(client, response_mode="full").status_code, 200)

    def test_24_compact_and_full_persist_identical_rows(self):
        client, store, _ = self._client()
        compact = self._post(client).json()
        full = self._post(client, response_mode="full").json()
        self.assertEqual(
            len(store.get_prediction_segments_for_batch(compact["batchId"])),
            len(store.get_prediction_segments_for_batch(full["batchId"])),
        )

    def test_25_response_size_metadata_matches_contract(self):
        client, _, _ = self._client()
        payload = self._post(client, response_mode="full").json()
        self.assertEqual(
            payload["responseSizeBytes"], api._response_size_bytes(payload)
        )

    def test_26_oversize_returns_413_after_persistence(self):
        client, store, _ = self._client()
        with patch.dict("os.environ", {"PREDICT_FULL_RESPONSE_MAX_BYTES": "1"}):
            response = self._post(client, response_mode="full")
        self.assertEqual(response.status_code, 413)
        self.assertIsNotNone(store.batch(response.json()["detail"]["batchId"]))

    def test_27_413_batch_is_readable(self):
        client, store, _ = self._client()
        with patch.dict("os.environ", {"PREDICT_FULL_RESPONSE_MAX_BYTES": "1"}):
            response = self._post(client, response_mode="full")
        batch_id = response.json()["detail"]["batchId"]
        self.assertEqual(len(store.get_prediction_segments_for_batch(batch_id)), 3)

    def test_28_excluded_explanations_are_smaller(self):
        client, _, _ = self._client()
        complete = self._post(
            client, response_mode="full", include_explanations=True
        ).json()
        excluded = self._post(
            client, response_mode="full", include_explanations=False
        ).json()
        self.assertLess(excluded["responseSizeBytes"], complete["responseSizeBytes"])

    def test_29_city_action_plan_endpoint_remains_compatible(self):
        client, _, _ = self._client()
        self._post(client, response_mode="full")
        response = client.post("/api/v1/action-plans", headers=self.headers, json={})
        self.assertEqual(response.status_code, 201)

    def test_30_segment_endpoints_remain_compatible(self):
        client, _, _ = self._client()
        batch = self._post(client, response_mode="full").json()["batchId"]
        self.assertEqual(
            client.get(
                "/api/v1/risk/segment/segment-0000", headers=self.headers
            ).status_code,
            200,
        )
        self.assertEqual(
            client.get(
                f"/api/v1/batches/{batch}/segments/segment-0000", headers=self.headers
            ).status_code,
            200,
        )

    def test_31_backend_sync_returns_only_requested_backend_dto(self):
        client, _, _ = self._client()
        response = self._post(
            client,
            response_mode="backend_sync",
            road_segment_ids=["segment-0002", "segment-0000", "segment-0000"],
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["contractVersion"], "2")
        self.assertEqual(payload["modelHorizon"], "24h")
        self.assertEqual(payload["predictionsCount"], 3)
        self.assertEqual(payload["predictionsReturned"], 2)
        self.assertEqual(
            [row["road_segment_id"] for row in payload["predictions"]],
            ["segment-0000", "segment-0002"],
        )
        row = payload["predictions"][0]
        self.assertEqual(row["risk_score"], 1)
        self.assertEqual(row["risk_level"], "HIGH")
        self.assertIsNone(row["confidence"])
        self.assertIsNone(row["uncertainty"])
        self.assertEqual(row["top_positive_factors"][0]["value"], 100)
        self.assertEqual(
            row["top_positive_factors"][0]["display_name"]["ru"], "Длина участка"
        )
        self.assertEqual(row["future_context"]["signals"][0]["code"], "SEVERE_WEATHER")
        self.assertEqual(row["future_context"]["providers"]["events"]["venue"], "Arena")
        self.assertEqual(
            row["future_context"]["providers"]["repairs"]["title"],
            "Lane closure",
        )
        self.assertNotIn("dynamic_score", row)
        self.assertNotIn("risk_probability", row)

    def test_32_backend_sync_requires_segment_ids(self):
        client, _, _ = self._client()
        response = self._post(client, response_mode="backend_sync")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "BACKEND_SYNC_SEGMENT_IDS_REQUIRED")

    def test_33_backend_sync_rejects_blank_segment_id(self):
        client, _, _ = self._client()
        self.assertEqual(
            self._post(
                client, response_mode="backend_sync", road_segment_ids=[" "]
            ).status_code,
            422,
        )


if __name__ == "__main__":
    unittest.main()
