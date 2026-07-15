from __future__ import annotations

import unittest

import pandas as pd
from fastapi.testclient import TestClient

from api_service.app import Runtime, TrainingRuntime, create_app


class ApiServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        def predictor(_: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "road_segment_id": "segment-1",
                        "priority_rank": 1,
                        "operational_priority": "high",
                        "reasons": ["DYNAMIC_TOP_5PCT"],
                    }
                ]
            )

        self.training = TrainingRuntime(lambda _: [{"id": "event-1"}])
        self.client = TestClient(
            create_app(
                api_key="shared-secret",
                runtime=Runtime(predictor),
                training=self.training,
            )
        )
        self.headers = {"X-API-Key": "shared-secret"}

    def test_protected_endpoint_rejects_invalid_key(self):
        self.assertEqual(self.client.get("/api/v1/model-status").status_code, 401)

    def test_predict_and_read_back_top_risk(self):
        response = self.client.post(
            "/api/v1/predict", headers=self.headers, json={"force": False}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(response.json()["predictionsCount"], 1)
        top = self.client.get("/api/v1/risk/top?limit=1", headers=self.headers)
        self.assertEqual(top.status_code, 200)
        self.assertEqual(top.json()[0]["road_segment_id"], "segment-1")
        recommendations = self.client.get(
            "/api/v1/recommendations/top?limit=1", headers=self.headers
        )
        self.assertEqual(recommendations.status_code, 200)

    def test_training_is_idempotent_and_queryable(self):
        payload = {
            "baseDatasetSnapshotId": "snapshot-1",
            "includeConfirmedEventsUntil": "2026-07-14T12:00:00Z",
        }
        headers = self.headers | {"Idempotency-Key": "request-1"}
        first = self.client.post("/api/v1/training", headers=headers, json=payload)
        second = self.client.post("/api/v1/training", headers=headers, json=payload)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["trainingRunId"], second.json()["trainingRunId"])
        status = self.client.get(
            f"/api/v1/training/{first.json()['trainingRunId']}", headers=self.headers
        )
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "QUEUED")

    def test_training_rejects_parallel_run(self):
        body = {
            "baseDatasetSnapshotId": "snapshot",
            "includeConfirmedEventsUntil": "2026-07-14T12:00:00Z",
        }
        self.client.post(
            "/api/v1/training",
            headers=self.headers | {"Idempotency-Key": "one"},
            json=body,
        )
        response = self.client.post(
            "/api/v1/training",
            headers=self.headers | {"Idempotency-Key": "two"},
            json=body,
        )
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
