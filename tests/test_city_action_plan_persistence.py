import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from ml_service.runtime_store import PredictionStore
from recommendations.city_action_plan import generate_city_action_plan


class P(unittest.TestCase):
    def store(self):
        return PredictionStore(Path(tempfile.mkdtemp()) / "x.sqlite3")

    def plan(self, pid="p", batch="b", at="2026-01-01T01:00:00+05:00"):
        p = generate_city_action_plan(
            [], batch_id=batch, prediction_datetime="2026-01-01T00:00:00+05:00"
        )
        p.update(
            plan_id=pid,
            generated_at=at,
            actions=[
                {
                    "action_id": "a",
                    "text": {"ru": "текст", "kz": "мәтін", "en": "text"},
                    "action_priority_score": 0.5,
                    "location": {"segment_ids": ["1"]},
                }
            ],
        )
        p["summary"]["actions_returned"] = 1
        return p

    def save(self, s, p, **k):
        s.save_completed_action_plan(p, **k)

    def test_save_and_load_completed_plan(self):
        s = self.store()
        p = self.plan()
        self.save(s, p)
        self.assertEqual(s.get_action_plan("p")["actions"][0]["text"]["ru"], "текст")

    def test_plan_survives_store_reopen(self):
        s = self.store()
        p = self.plan()
        self.save(s, p)
        self.assertEqual(
            PredictionStore(s.path).get_action_plan("p")["actions"], p["actions"]
        )

    def test_latest_completed_ignores_running_and_failed(self):
        s = self.store()
        self.save(s, self.plan())
        s.save_failed_action_plan(
            plan_id="f", batch_id="b", prediction_datetime="x", error="safe"
        )
        self.assertEqual(s.get_latest_action_plan()["plan_id"], "p")

    def test_get_plan_for_correct_batch(self):
        s = self.store()
        self.save(s, self.plan("a", "a"))
        self.save(s, self.plan("b", "b"))
        self.assertEqual(s.get_action_plan_for_batch("a")["plan_id"], "a")

    def test_multiple_plans_for_same_batch(self):
        s = self.store()
        self.save(s, self.plan("old", "b", "2026-01-01T01:00:00+05:00"))
        self.save(s, self.plan("new", "b", "2026-01-01T02:00:00+05:00"))
        self.assertEqual(s.get_action_plan_for_batch("b")["plan_id"], "new")
        self.assertIsNotNone(s.get_action_plan("old"))

    def test_failed_plan_is_not_completed(self):
        s = self.store()
        s.save_failed_action_plan(
            plan_id="f", batch_id="b", prediction_datetime="x", error="safe"
        )
        self.assertEqual(s.get_action_plan("f")["status"], "failed")
        self.assertIsNone(s.get_action_plan_for_batch("b"))

    def test_json_safe_nested_serialization(self):
        s = self.store()
        p = self.plan()
        p["actions"][0]["null"] = None
        self.save(s, p)
        json.dumps(s.get_action_plan("p"))
        self.assertIsNone(s.get_action_plan("p")["actions"][0]["null"])

    def test_sensitive_internal_fields_are_not_persisted(self):
        s = self.store()
        p = self.plan()
        p.update(api_key="SECRET", raw_provider_payload="RAW")
        self.save(s, p)
        self.assertNotIn("SECRET", json.dumps(s.get_action_plan("p")))

    def test_prediction_batch_persistence_regression(self):
        s = self.store()
        f = pd.DataFrame([{"road_segment_id": "x", "priority_rank": 1}])
        s.save_completed(
            {
                "batch_id": "b",
                "prediction_datetime": "x",
                "started_at": "x",
                "completed_at": "x",
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            f,
        )
        self.assertEqual(s.segment("x")["road_segment_id"], "x")

    def test_completed_empty_plan_is_valid(self):
        s = self.store()
        p = generate_city_action_plan([], batch_id="b", prediction_datetime="x")
        p.update(plan_id="e", generated_at="z")
        self.save(s, p)
        self.assertEqual(s.get_action_plan("e")["actions"], [])
