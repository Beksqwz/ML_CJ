import unittest
from datetime import datetime

from future_intelligence.schemas import FutureRecord, ProviderMetadata, ProviderResult
from future_intelligence.validation import validate_result


class FutureIntelligenceSchemaTests(unittest.TestCase):
    def test_universal_schema_validates_required_fields(self):
        metadata = ProviderMetadata(
            "mock", "1", "weather_forecast", (24,), False, "test", "city"
        )
        record = FutureRecord(
            "mock",
            "weather_forecast",
            "1",
            "id",
            None,
            datetime.now(),
            None,
            None,
            None,
            datetime.now(),
            24,
            None,
            None,
            payload={},
        )
        self.assertEqual(
            validate_result(
                ProviderResult(metadata, [], [record], {"weather_signal": 1}, {})
            ),
            [],
        )

    def test_invalid_namespaced_feature_is_reported(self):
        metadata = ProviderMetadata(
            "mock", "1", "weather_forecast", (24,), False, "test", "city"
        )
        self.assertIn(
            "unnamespaced_feature",
            validate_result(ProviderResult(metadata, [], [], {"signal": 1}, {})),
        )
