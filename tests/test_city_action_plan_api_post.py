import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from fastapi.testclient import TestClient
from api_service.app import Runtime, create_app
from ml_service.runtime_store import PredictionStore


class A(unittest.TestCase):
    def setUp(self):
        self.s = PredictionStore(Path(tempfile.mkdtemp()) / "x.sqlite3")
        self.c = TestClient(create_app(api_key="k", runtime=Runtime(store=self.s)))
        self.h = {"X-API-Key": "k"}

    def batch(self, b, at, n=2):
        rows = [
            {
                "road_segment_id": f"{b}-{i}",
                "operational_priority": "high" if i == 0 else "low",
                "dynamic_percentile": 0.99 if i == 0 else 0,
                "dynamic_rank": i + 1,
                "historical_hotspot_percentile": 0.8 if i == 0 else 0,
                "historical_hotspot_rank": 1,
                "reasons": ["DYNAMIC_TOP_1PCT"] if i == 0 else [],
                "warnings": [],
                "uncertainty": "low",
                "road_name": "Road",
            }
            for i in range(n)
        ]
        self.s.save_completed(
            {
                "batch_id": b,
                "prediction_datetime": at,
                "started_at": at,
                "completed_at": at,
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            pd.DataFrame(rows),
        )

    def test_post_action_plan_uses_explicit_completed_batch(self):
        self.batch("a", "2026-01-01T00:00:00+05:00")
        r = self.c.post("/api/v1/action-plans", headers=self.h, json={"batch_id": "a"})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.s.get_action_plan(r.json()["plan_id"])["batch_id"], "a")

    def test_post_action_plan_uses_latest_completed_batch(self):
        self.batch("a", "2026-01-01T00:00:00+05:00")
        self.batch("b", "2026-01-02T00:00:00+05:00")
        self.assertEqual(
            self.c.post("/api/v1/action-plans", headers=self.h, json={}).json()[
                "batch_id"
            ],
            "b",
        )

    def test_post_action_plan_handles_3968_segments_without_external_calls(self):
        self.batch("a", "2026-01-01T00:00:00+05:00", 3968)
        r = self.c.post("/api/v1/action-plans", headers=self.h, json={"batch_id": "a"})
        self.assertEqual(
            (r.status_code, r.json()["summary"]["segments_analyzed"]), (201, 3968)
        )
        self.assertNotIn("segments", r.json())

    def test_generation_failure_marks_plan_failed_safely(self):
        self.batch("a", "2026-01-01T00:00:00+05:00")
        with patch(
            "api_service.app.generate_city_action_plan",
            side_effect=RuntimeError("secret=FAKE_SECRET C:\\Users\\Beknur\\private"),
        ):
            r = self.c.post(
                "/api/v1/action-plans", headers=self.h, json={"batch_id": "a"}
            )
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("FAKE_SECRET", r.text)
        self.assertIsNone(self.s.get_latest_action_plan())
