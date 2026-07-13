import hashlib
import json
import unittest
from datetime import datetime

from future_intelligence.pipeline import FutureIntelligencePipeline
from future_intelligence.registry import ProviderRegistry, default_registry
from future_intelligence.schemas import ProviderMetadata, ProviderResult
from ml_service.inference.feature_builder import _config
from ml_service.registry import ModelRegistry, ROOT


class MockProvider:
    metadata = ProviderMetadata("mock", "1", "test", (24,), False, "test", "city")

    def collect(self, prediction_datetime, horizon_hours, **kwargs):
        return ProviderResult(self.metadata, [], [], {"mock_signal": 1}, {"points": 1})

    def normalize(self, *args):
        return []

    def build_features(self, *args):
        return {"mock_signal": 1}

    def healthcheck(self):
        return {"status": "ok"}


class FutureIntelligencePipelineTests(unittest.TestCase):
    def test_registry_loads_openweather(self):
        self.assertIn("openweather", default_registry().names())

    def test_pipeline_handles_missing_key_without_raising(self):
        context = FutureIntelligencePipeline().collect("2026-07-14T00:00:00+05:00")
        self.assertEqual(context["status"], "degraded")
        self.assertTrue(context["fallback_used"])

    def test_pipeline_accepts_multiple_mock_providers(self):
        registry = ProviderRegistry()
        registry.register("first", MockProvider)
        registry.register("second", MockProvider)
        context = FutureIntelligencePipeline(registry).collect(
            "2026-07-14T00:00:00+05:00", providers=("first", "second")
        )
        self.assertEqual(len(context["providers"]), 2)
        self.assertTrue(context["fallback_used"])

    def test_frozen_24h_contract_is_unchanged(self):
        entry = ModelRegistry().get("24h")
        features = list(_config("24h")["numerical_features"]) + list(
            _config("24h")["categorical_features"]
        )
        self.assertEqual(entry["stage"], "Stage 7B")
        self.assertEqual(entry["feature_count"], 77)
        self.assertEqual(len(features), 77)
        self.assertEqual(
            hashlib.sha256(
                json.dumps(features, separators=(",", ":")).encode()
            ).hexdigest(),
            "bf0ce06aface9fc3539a95df2557728e57d987a9e0c0e5ac4a195a53b47f6d96",
        )
        self.assertEqual(
            hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest(),
            "0c8e1b88b1cfaf95fb39e395e2fdc54f1b7abda22d8ac00e1d6f561ab9110a0c",
        )

    def test_features_do_not_collide_with_frozen_model_contract(self):
        context = FutureIntelligencePipeline().collect(datetime(2026, 7, 14))
        frozen = set(_config("24h")["numerical_features"]) | set(
            _config("24h")["categorical_features"]
        )
        self.assertFalse(set(context["features"]) & frozen)
