from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
from api_service.app import Runtime, create_app
from ml_service.runtime_store import PredictionStore

PROVENANCE = {
    "ml_weather_snapshot_version": "s1",
    "ml_weather_origin_timestamp": "2026-01-01T01:00:00+05:00",
    "ml_weather_source_before": "a",
    "ml_weather_source_after": "b",
    "ml_weather_interpolated": True,
    "ml_weather_degraded": False,
    "explanation_weather_snapshot_version": "s1",
    "explanation_forecast_start": "x",
    "explanation_forecast_end": "y",
    "explanation_forecast_points_available": 8,
    "weather_snapshot_consistent": True,
    "weather_snapshot_issues": "[]",
}


class W2APropagationTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "x.sqlite3"
        self.store = PredictionStore(self.path)
        self.row = {
            "road_segment_id": "s",
            "priority_rank": 1,
            "weather_context_available": True,
            "weather_provider": "openweather",
            "weather_snapshot_version": "s1",
            "warnings": [],
            **PROVENANCE,
        }
        self.store.save_completed(
            {
                "batch_id": "b",
                "prediction_datetime": "x",
                "started_at": "x",
                "completed_at": "z",
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            pd.DataFrame([self.row]),
        )
        self.client = TestClient(
            create_app(api_key="k", runtime=Runtime(store=PredictionStore(self.path)))
        )
        self.h = {"X-API-Key": "k"}

    def test_valid_provenance_survives_sqlite_reopen(self):
        row = PredictionStore(self.path).segment("s")
        [self.assertEqual(row[k], v) for k, v in PROVENANCE.items()]

    def test_latest_segment_endpoint_returns_provenance(self):
        r = self.client.get("/api/v1/risk/segment/s", headers=self.h)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.json()["context"]["weather"]["provenance"]["mlSnapshotVersion"], "s1"
        )

    def test_batch_segment_endpoint_returns_provenance(self):
        r = self.client.get("/api/v1/batches/b/segments/s", headers=self.h)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.json()["context"]["weather"]["provenance"]["explanationSnapshotVersion"],
            "s1",
        )

    def test_legacy_warning_survives_persistence_and_api(self):
        self.row.update(
            {
                "road_segment_id": "legacy",
                "warnings": ["WEATHER_SNAPSHOT_LEGACY_SCHEMA"],
                "weather_context_available": False,
                "ml_weather_degraded": True,
                "weather_snapshot_consistent": False,
            }
        )
        self.store.save_completed(
            {
                "batch_id": "legacy",
                "prediction_datetime": "x",
                "started_at": "x",
                "completed_at": "zz",
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            pd.DataFrame([self.row]),
        )
        self.assertIn(
            "WEATHER_SNAPSHOT_LEGACY_SCHEMA",
            PredictionStore(self.path).segment("legacy", "legacy")["warnings"],
        )

    def test_mismatch_warning_survives_persistence_and_api(self):
        self.row.update(
            {
                "road_segment_id": "mismatch",
                "warnings": ["WEATHER_SNAPSHOT_MISMATCH"],
                "ml_weather_degraded": True,
                "weather_snapshot_consistent": False,
                "reasons": [],
            }
        )
        self.store.save_completed(
            {
                "batch_id": "mismatch",
                "prediction_datetime": "x",
                "started_at": "x",
                "completed_at": "zzz",
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            pd.DataFrame([self.row]),
        )
        row = PredictionStore(self.path).segment("mismatch", "mismatch")
        self.assertIn("WEATHER_SNAPSHOT_MISMATCH", row["warnings"])
        self.assertNotIn("SEVERE_WEATHER", row["reasons"])

    def test_public_api_excludes_sensitive_internal_fields(self):
        self.row.update(
            {
                "road_segment_id": "safe",
                "frozen_feature_vector": "SECRET_VECTOR",
                "raw_provider_payload": "RAW",
                "api_key": "KEY",
                "authorization": "AUTH",
                "local_file_path": "C:/secret",
                "provider_secret": "SECRET",
            }
        )
        self.store.save_completed(
            {
                "batch_id": "safe",
                "prediction_datetime": "x",
                "started_at": "x",
                "completed_at": "zzzz",
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            pd.DataFrame([self.row]),
        )
        r = self.client.get("/api/v1/batches/safe/segments/safe", headers=self.h)
        self.assertNotIn("SECRET_VECTOR", r.text)
