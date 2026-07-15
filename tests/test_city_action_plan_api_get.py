"""HTTP read/validation coverage for the persisted city action-plan API."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from api_service.app import Runtime, create_app
from ml_service.runtime_store import PredictionStore


class CityActionPlanGetApiTests(unittest.TestCase):
    headers = {"X-API-Key": "test-key"}
    prediction_time = "2026-07-15T09:00:00+05:00"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PredictionStore(Path(self.tmp.name) / "runtime.sqlite3")
        self.client = TestClient(
            create_app(api_key="test-key", runtime=Runtime(store=self.store))
        )

    def tearDown(self) -> None:
        self.client.close()
        self.tmp.cleanup()

    def _save_batch(self, batch_id: str, completed_at: str | None = None) -> None:
        completed_at = completed_at or self.prediction_time
        frame = pd.DataFrame(
            [
                {
                    "road_segment_id": f"{batch_id}-segment",
                    "operational_priority": "high",
                    "dynamic_percentile": 0.98,
                    "dynamic_rank": 1,
                    "historical_hotspot_percentile": 0.75,
                    "historical_hotspot_rank": 2,
                    "reasons": ["DYNAMIC_TOP_5PCT"],
                    "warnings": [],
                    "uncertainty": "low",
                    "road_name": "Абая",
                }
            ]
        )
        self.store.save_completed(
            {
                "batch_id": batch_id,
                "prediction_datetime": self.prediction_time,
                "started_at": completed_at,
                "completed_at": completed_at,
                "execution_time_ms": 1,
                "model_version": "test",
                "warnings": [],
            },
            frame,
        )

    def _completed_plan(
        self, plan_id: str, batch_id: str, generated_at: str
    ) -> dict[str, object]:
        return {
            "plan_id": plan_id,
            "batch_id": batch_id,
            "status": "completed",
            "prediction_datetime": self.prediction_time,
            "horizon_hours": 24,
            "generated_at": generated_at,
            "plan_version": "city_action_plan_v1",
            "request_parameters": {"max_actions": 10, "minimum_priority": "medium"},
            "summary": {
                "segments_analyzed": 1,
                "candidate_segments": 1,
                "groups_created": 1,
                "actions_returned": 1,
            },
            "actions": [
                {
                    "action_id": f"{plan_id}-action",
                    "action_code": "INCREASE_PATROL",
                    "action_priority": "high",
                    "action_priority_score": 0.7,
                    "action_rank": 1,
                    "location": {"display_name": "Абая", "segment_ids": ["segment"]},
                    "recommended_period": {
                        "start": self.prediction_time,
                        "end": self.prediction_time,
                        "basis": "prediction_horizon",
                    },
                    "reason_codes": ["DYNAMIC_TOP_5PCT"],
                    "evidence": {},
                    "text": {"ru": "Рассмотреть", "kz": "Қарастыру", "en": "Consider"},
                    "warnings": [],
                    "requires_human_confirmation": True,
                }
            ],
        }

    def _save_completed_plan(
        self, plan_id: str, batch_id: str, generated_at: str
    ) -> dict[str, object]:
        plan = self._completed_plan(plan_id, batch_id, generated_at)
        self.store.save_completed_action_plan(
            plan, max_actions=10, minimum_priority="medium"
        )
        return plan

    def _save_running_or_failed(self, plan_id: str, status: str) -> None:
        self.store.create_action_plan(
            plan_id=plan_id,
            batch_id="batch-status",
            prediction_datetime=self.prediction_time,
            generated_at="2026-07-15T12:00:00+05:00",
        )
        if status == "failed":
            self.store.save_failed_action_plan(
                plan_id=plan_id,
                batch_id="batch-status",
                prediction_datetime=self.prediction_time,
                error="CITY_ACTION_PLAN_GENERATION_FAILED",
            )

    def test_get_action_plan_by_id(self) -> None:
        persisted = self._save_completed_plan(
            "plan-1", "batch-1", "2026-07-15T10:00:00+05:00"
        )
        response = self.client.get("/api/v1/action-plans/plan-1", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["plan_id"], persisted["plan_id"])
        self.assertEqual(response.json()["batch_id"], persisted["batch_id"])
        self.assertEqual(response.json()["actions"], persisted["actions"])
        self.assertEqual(response.json()["summary"], persisted["summary"])

    def test_get_latest_completed_action_plan(self) -> None:
        self._save_completed_plan("old", "batch-old", "2026-07-15T10:00:00+05:00")
        self._save_completed_plan("new", "batch-new", "2026-07-15T11:00:00+05:00")
        response = self.client.get("/api/v1/action-plans/latest", headers=self.headers)
        self.assertEqual(
            (response.status_code, response.json()["plan_id"]), (200, "new")
        )

    def test_latest_ignores_running_and_failed(self) -> None:
        self._save_completed_plan(
            "completed", "batch-completed", "2026-07-15T10:00:00+05:00"
        )
        self._save_running_or_failed("running", "running")
        self._save_running_or_failed("failed", "failed")
        response = self.client.get("/api/v1/action-plans/latest", headers=self.headers)
        self.assertEqual(
            (response.status_code, response.json()["plan_id"]), (200, "completed")
        )

    def test_get_unknown_or_non_completed_plan_returns_404(self) -> None:
        self._save_running_or_failed("running", "running")
        self._save_running_or_failed("failed", "failed")
        for plan_id in ("unknown", "running", "failed"):
            response = self.client.get(
                f"/api/v1/action-plans/{plan_id}", headers=self.headers
            )
            self.assertEqual(response.status_code, 404)
            self.assertNotIn("error", response.text.lower())

    def test_latest_without_completed_plan_returns_404(self) -> None:
        self._save_running_or_failed("running", "running")
        response = self.client.get("/api/v1/action-plans/latest", headers=self.headers)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "CITY_ACTION_PLAN_NOT_FOUND")

    def test_post_request_validation(self) -> None:
        for body in (
            {"max_actions": 0},
            {"max_actions": 51},
            {"minimum_priority": "low"},
            {"batch_id": "   "},
        ):
            self.assertEqual(
                self.client.post(
                    "/api/v1/action-plans", headers=self.headers, json=body
                ).status_code,
                422,
            )

    def test_post_batch_not_found_cases(self) -> None:
        unknown = self.client.post(
            "/api/v1/action-plans", headers=self.headers, json={"batch_id": "unknown"}
        )
        self.assertEqual(unknown.status_code, 404)
        with self.store._connect() as db:
            db.execute(
                "INSERT INTO prediction_batches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "running",
                    "running",
                    self.prediction_time,
                    24,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None,
                    "[]",
                    None,
                ),
            )
        self.assertEqual(
            self.client.post(
                "/api/v1/action-plans",
                headers=self.headers,
                json={"batch_id": "running"},
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/action-plans", headers=self.headers, json={}
            ).status_code,
            404,
        )

    def test_get_endpoints_are_read_only(self) -> None:
        plan = self._save_completed_plan(
            "readonly", "batch-readonly", "2026-07-15T10:00:00+05:00"
        )
        with (
            patch(
                "api_service.app.generate_city_action_plan", side_effect=AssertionError
            ) as city_plan,
            patch.object(Runtime, "predict", side_effect=AssertionError) as prediction,
        ):
            by_id = self.client.get(
                "/api/v1/action-plans/readonly", headers=self.headers
            )
            latest = self.client.get(
                "/api/v1/action-plans/latest", headers=self.headers
            )
        self.assertEqual((by_id.status_code, latest.status_code), (200, 200))
        self.assertEqual((city_plan.call_count, prediction.call_count), (0, 0))
        self.assertEqual(self.store.get_action_plan("readonly"), plan)


if __name__ == "__main__":
    unittest.main()
