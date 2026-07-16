from __future__ import annotations

import tempfile
import unittest
import io
import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from api_service.app import Runtime, create_app
from ml_service.hybrid_risk import _future_context, _weights, load_valid_future_context
from ml_service.inference.feature_builder import ROOT, _live_calendar
from ml_service.runtime_store import PredictionStore
from recommendations.stage20b import recommend_stage20b
from scripts import refresh_future_intelligence as refresh
from scripts import run_future_intelligence_scheduler as scheduler
from scripts.run_future_intelligence_scheduler import run_with_retry


class LiveHardeningTests(unittest.TestCase):
    def test_future_date_calendar_is_deterministic(self):
        value = _live_calendar(pd.Timestamp("2031-01-01T08:00:00"))
        self.assertEqual(value["year"], 2031)
        self.assertTrue(value["is_rush_hour"])
        self.assertIn(value["season"], {"winter", "spring", "summer", "autumn"})

    def test_frozen_stage19i_weights_are_unchanged(self):
        self.assertEqual(_weights()[0], 0.8)
        self.assertAlmostEqual(_weights()[1], 0.2)

    def test_severity_five_is_normalized_before_recommendation_flag(self):
        source = pd.DataFrame(
            {
                "road_segment_id": ["x"],
                "weather_provider_available": [1],
                "traffic_context_available": [0],
                "gov_provider_available": [0],
                "ticketon_provider_available": [0],
                "weather_severity_score": [5],
            }
        )
        result = _future_context(pd.Series(["x"]), source)
        self.assertIn("severe_weather", result.loc[0, "future_context_flags"])
        self.assertAlmostEqual(result.loc[0, "weather_severity_score"], 1.0)

    def test_unavailable_snapshot_returns_explicit_codes(self):
        context, warnings = load_valid_future_context(
            "2031-01-01T00:00:00+05:00", Path("missing.parquet")
        )
        self.assertIsNone(context)
        self.assertIn("FUTURE_CONTEXT_UNAVAILABLE", warnings)
        self.assertIn("WEATHER_PROVIDER_DEGRADED", warnings)

    def test_completed_batch_survives_new_store_instance(self):
        path = Path(tempfile.mkdtemp()) / "runtime.sqlite3"
        store = PredictionStore(path)
        frame = pd.DataFrame([{"road_segment_id": "s1", "priority_rank": 1}])
        store.save_completed(
            {
                "batch_id": "b1",
                "prediction_datetime": "2031-01-01T00:00:00+05:00",
                "started_at": "a",
                "completed_at": "b",
                "execution_time_ms": 1,
                "model_version": "v1",
                "warnings": [],
            },
            frame,
        )
        restored = PredictionStore(path)
        self.assertEqual(restored.latest()["batchId"], "b1")
        self.assertEqual(restored.segment("s1")["road_segment_id"], "s1")

    def test_live_calendar_has_all_frozen_calendar_features(self):
        live = _live_calendar(pd.Timestamp("2031-05-01T12:00:00"))
        required = {
            "year",
            "month",
            "day",
            "hour",
            "weekday",
            "is_weekend",
            "season",
            "is_rush_hour",
            "is_holiday",
            "holiday_name",
            "is_school_break",
        }
        self.assertTrue(required.issubset(live.index))

    def test_only_stage19i_24h_artifacts_are_referenced(self):
        source = (ROOT / "ml_service" / "hybrid_risk.py").read_text(encoding="utf-8")
        self.assertIn("stage19i_catboost.cbm", source)
        self.assertIn("stage19i_hist_gradient_boosting.joblib", source)
        self.assertNotIn("catboost_1h", source)

    def test_weather_override_is_limited_to_existing_contract(self):
        override = {
            "temperature_2m": 20,
            "relative_humidity_2m": 50,
            "precipitation": 0,
        }
        self.assertTrue(
            set(override).issubset(
                {
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "rain",
                    "snowfall",
                    "weather_code",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                }
            )
        )

    def test_strict_weather_uses_clear_error(self):
        from ml_service.inference.feature_builder import build_features

        with self.assertRaisesRegex(ValueError, "LIVE_WEATHER_UNAVAILABLE"):
            build_features("2031-01-01T00:00:00", "24h", strict_live_weather=True)

    def test_severity_ten_normalizes_to_one(self):
        frame = pd.DataFrame(
            {
                "road_segment_id": ["x"],
                "weather_provider_available": [1],
                "weather_severity_score": [10],
            }
        )
        self.assertEqual(
            _future_context(pd.Series(["x"]), frame).loc[0, "weather_severity_score"],
            1.0,
        )

    def test_stale_snapshot_is_not_joined(self):
        context, warnings = load_valid_future_context(
            "2031-01-01T00:00:00+05:00",
            ROOT
            / "data/future_intelligence/processed/unified_future_features_24h.parquet",
        )
        self.assertIsNone(context)
        self.assertIn("FUTURE_CONTEXT_STALE", warnings)

    def test_incomplete_batch_is_not_latest(self):
        path = Path(tempfile.mkdtemp()) / "runtime.sqlite3"
        store = PredictionStore(path)
        with store._connect() as db:
            db.execute(
                "INSERT INTO prediction_batches VALUES ('bad','running',NULL,24,NULL,NULL,NULL,NULL,NULL,NULL,'[]',NULL)"
            )
        self.assertIsNone(PredictionStore(path).latest())

    def test_latest_completed_batch_is_restored(self):
        path = Path(tempfile.mkdtemp()) / "runtime.sqlite3"
        store = PredictionStore(path)
        frame = pd.DataFrame([{"road_segment_id": "s", "priority_rank": 1}])
        store.save_completed(
            {
                "batch_id": "done",
                "prediction_datetime": "x",
                "started_at": "x",
                "completed_at": "z",
                "execution_time_ms": 1,
                "model_version": "v",
                "warnings": [],
            },
            frame,
        )
        self.assertEqual(PredictionStore(path).latest()["status"], "completed")

    def test_possible_plan_remains_available(self):
        frame = pd.DataFrame(
            {
                "road_segment_id": ["x"],
                "prediction_datetime": ["x"],
                "dynamic_rank": [1],
                "dynamic_percentile": [1.0],
                "historical_hotspot_rank": [1],
                "historical_hotspot_percentile": [1.0],
                "future_context_flags": ["[]"],
                "future_context_warnings": ["[]"],
                "provider_degraded": [False],
            }
        )
        self.assertTrue(recommend_stage20b(frame).iloc[0].possible_plan)

    def test_dynamic_score_is_not_named_probability(self):
        self.assertNotIn(
            "probability",
            (ROOT / "ml_service" / "hybrid_risk.py")
            .read_text(encoding="utf-8")
            .split("def build_dynamic_risk")[1]
            .split("def validate_dynamic_risk")[0],
        )

    def test_artifact_hashes_are_frozen(self):
        expected = {
            "stage19i_catboost.cbm": "5b574c92959bf8e17388c0046f46e3ef402f16030bea77fbd3ea945e2efb27fb",
            "stage19i_hist_gradient_boosting.joblib": "409b41c62ba68b976f26f711b174f7903ef124331cde27be2c12819dd13554c9",
        }
        for name, digest in expected.items():
            self.assertEqual(
                sha256((ROOT / "models/final" / name).read_bytes()).hexdigest(),
                digest,
            )

    def test_retry_stops_at_maximum_attempts(self):
        with patch.dict(
            "os.environ",
            {
                "FUTURE_RETRY_MAX_ATTEMPTS": "3",
                "FUTURE_RETRY_INITIAL_SECONDS": "2",
                "FUTURE_RETRY_MAX_SECONDS": "30",
                "FUTURE_RETRY_MULTIPLIER": "2",
            },
        ):
            calls = []
            result = run_with_retry(
                "x", lambda _: calls.append(1) or 1, sleep=lambda _: None
            )
        self.assertEqual((len(calls), result["attempts"]), (3, 3))

    def test_retry_backoff_is_capped(self):
        delays = []
        with patch.dict(
            "os.environ",
            {
                "FUTURE_RETRY_MAX_ATTEMPTS": "4",
                "FUTURE_RETRY_INITIAL_SECONDS": "20",
                "FUTURE_RETRY_MAX_SECONDS": "30",
                "FUTURE_RETRY_MULTIPLIER": "2",
            },
        ):
            run_with_retry("x", lambda _: 1, sleep=delays.append)
        self.assertEqual(delays, [20.0, 30.0, 30.0])

    def test_non_retryable_error_is_not_retried(self):
        calls = []
        result = run_with_retry(
            "x", lambda _: calls.append(1) or 2, sleep=lambda _: None
        )
        self.assertEqual((len(calls), result["status"]), (1, "permanent_error"))

    def test_provider_retries_are_isolated(self):
        calls = []
        first = run_with_retry(
            "bad", lambda _: calls.append("bad") or 2, sleep=lambda _: None
        )
        second = run_with_retry(
            "good", lambda _: calls.append("good") or 0, sleep=lambda _: None
        )
        self.assertEqual(
            (first["status"], second["status"], calls[-1]),
            ("permanent_error", "ok", "good"),
        )

    def test_api_structured_context_and_no_secret(self):
        rt = Runtime(
            lambda _: pd.DataFrame(
                [
                    {
                        "road_segment_id": "s",
                        "priority_rank": 1,
                        "dynamic_score": 1,
                        "dynamic_rank": 1,
                        "dynamic_percentile": 1,
                        "dynamic_engine_version": "v",
                        "dynamic_engine_status": "x",
                    }
                ]
            ),
            store=PredictionStore(Path(tempfile.mkdtemp()) / "x.sqlite3"),
        )
        client = TestClient(create_app(api_key="key", runtime=rt))
        client.post("/api/v1/predict", headers={"X-API-Key": "key"}, json={})
        body = client.get("/api/v1/risk/segment/s", headers={"X-API-Key": "key"}).json()
        self.assertIn("context", body)
        self.assertNotIn("key", str(body).lower())

    def test_refresh_dry_run_invokes_zero_collectors_and_writes(self):
        output = io.StringIO()
        with (
            patch("sys.argv", ["refresh", "--dry-run"]),
            patch("sys.stdout", output),
            patch.object(refresh, "collect_provider") as collect,
            patch.object(refresh, "rebuild_segment_features") as segment_features,
            patch.object(refresh, "rebuild_unified") as rebuild,
        ):
            self.assertEqual(refresh.main(), 0)
        self.assertEqual(
            (collect.call_count, segment_features.call_count, rebuild.call_count),
            (0, 0, 0),
        )
        self.assertEqual(json.loads(output.getvalue())["network_calls"], 0)

    def test_scheduler_dry_run_invokes_zero_collectors_and_writes(self):
        output = io.StringIO()
        with (
            patch("sys.argv", ["scheduler", "--dry-run"]),
            patch("sys.stdout", output),
            patch.object(scheduler, "execute_provider") as execute,
            patch.object(scheduler, "atomic_json") as write,
        ):
            self.assertEqual(scheduler.main(), 0)
        self.assertEqual((execute.call_count, write.call_count), (0, 0))
        self.assertEqual(json.loads(output.getvalue())["writes"], 0)

    def test_graceful_shutdown_skips_backoff_wait(self):
        waits = []
        result = run_with_retry(
            "x", lambda _: 1, sleep=waits.append, active=lambda: False
        )
        self.assertEqual((result["attempts"], waits), (0, []))

    def test_scheduler_once_provider_isolation_fixture(self):
        bad = run_with_retry("bad", lambda _: 2, sleep=lambda _: None)
        good = run_with_retry("good", lambda _: 0, sleep=lambda _: None)
        self.assertEqual((bad["status"], good["status"]), ("permanent_error", "ok"))


if __name__ == "__main__":
    unittest.main()
