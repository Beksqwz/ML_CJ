"""Unit coverage for the Stage 18B canonical unified feature layer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from future_intelligence.segment_feature_builder import FEATURE_COLUMNS
from future_intelligence.unified_future_layer import (
    ALLOWED_TRAINING_STATUSES,
    PROVIDER_METADATA_COLUMNS,
    UnifiedFutureLayerBuilder,
    UnifiedPaths,
    feature_provenance,
    validate,
)
from ml_service.inference.feature_builder import _config


def _base_frame(rows: int = 3) -> pd.DataFrame:
    timestamp = "2026-07-15T08:00:00+05:00"
    values: dict[str, object] = {}
    for column in FEATURE_COLUMNS:
        if column == "road_segment_id":
            values[column] = list(range(100, 100 + rows))
        elif column in {
            "prediction_datetime",
            "forecast_window_start",
            "forecast_window_end",
            "generated_at",
        }:
            values[column] = [timestamp] * rows
        elif column.endswith("_available") or column.endswith("_used"):
            values[column] = [0] * rows
        elif "distance" in column or "hours_until" in column:
            values[column] = [None] * rows
        else:
            values[column] = [0] * rows
    values["horizon_hours"] = [24] * rows
    return pd.DataFrame(values, columns=FEATURE_COLUMNS)


class UnifiedFutureLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "future_segment_features_24h.parquet"
        _base_frame().to_parquet(self.source, index=False)
        self.weather = self.root / "openweather_24h_features.parquet"
        self.gov = self.root / "gov_kz_road_events.parquet"
        self.ticketon = self.root / "ticketon_events.parquet"
        self.traffic = self.root / "tomtom_live_traffic.parquet"
        self.paths = UnifiedPaths(
            segment_features_path=self.source,
            weather_path=self.weather,
            gov_path=self.gov,
            ticketon_path=self.ticketon,
            traffic_path=self.traffic,
            output_dir=self.root / "data" / "future_intelligence",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _build(self) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
        return UnifiedFutureLayerBuilder(self.paths, expected_rows=3).build()

    def test_provider_unavailable_is_explicit(self) -> None:
        frame, report, _ = self._build()
        self.assertEqual(frame["weather_provider_available"].iloc[0], 0)
        self.assertEqual(frame["provider_degraded"].iloc[0], 1)
        self.assertEqual(report["provider_coverage"]["weather"]["warning_count"], 1)

    def test_available_provider_with_zero_events_is_not_unavailable(self) -> None:
        pd.DataFrame().to_parquet(self.ticketon, index=False)
        frame, report, _ = self._build()
        self.assertEqual(frame["ticketon_provider_available"].iloc[0], 1)
        self.assertEqual(report["provider_coverage"]["ticketon"]["available"], 1)
        self.assertEqual(frame["event_count_next_24h"].sum(), 0)

    def test_weather_repairs_and_ticketon_columns_are_preserved(self) -> None:
        frame, _, _ = self._build()
        self.assertIn("weather_temperature_mean", frame)
        self.assertIn("repair_active_next_24h", frame)
        self.assertIn("event_count_next_24h", frame)
        self.assertIn("traffic_context_available", frame)

    def test_metadata_and_schema_are_stable(self) -> None:
        first, _, _ = self._build()
        second, _, _ = self._build()
        self.assertEqual(list(first.columns), list(second.columns))
        self.assertTrue(all(column in first for column in PROVIDER_METADATA_COLUMNS))
        self.assertEqual(
            list(first.columns[: len(FEATURE_COLUMNS)]), list(FEATURE_COLUMNS)
        )
        self.assertEqual(
            UnifiedFutureLayerBuilder.checksum(first),
            UnifiedFutureLayerBuilder.checksum(second),
        )

    def test_collision_report_detects_existing_reserved_column(self) -> None:
        frame = _base_frame()
        frame["metadata_generated_at"] = "unexpected"
        frame.to_parquet(self.source, index=False)
        _, _, collisions = self._build()
        self.assertEqual(
            collisions["reserved_metadata_collisions"], ["metadata_generated_at"]
        )

    def test_validation_passes_for_valid_table(self) -> None:
        frame, _, _ = self._build()
        self.assertTrue(all(validate(frame, expected_rows=3).values()))

    def test_storage_writes_readable_json_and_parquet_idempotently(self) -> None:
        frame, _, _ = self._build()
        builder = UnifiedFutureLayerBuilder(self.paths, expected_rows=3)
        first = builder.save(frame)
        second = builder.save(frame)
        self.assertFalse(first["unchanged"])
        self.assertTrue(second["unchanged"])
        self.assertEqual(len(pd.read_parquet(first["parquet"])), 3)
        self.assertEqual(
            len(json.loads(Path(first["json"]).read_text(encoding="utf-8"))), 3
        )

    def test_provenance_has_one_valid_status_per_feature(self) -> None:
        frame, _, _ = self._build()
        items = feature_provenance(frame, self.source)
        self.assertEqual(len(items), len(frame.columns))
        self.assertTrue(
            all(item["training_status"] in ALLOWED_TRAINING_STATUSES for item in items)
        )
        self.assertEqual({item["feature_name"] for item in items}, set(frame.columns))

    def test_invalid_row_grain_is_rejected(self) -> None:
        _base_frame(rows=2).to_parquet(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "stage18a_row_grain_invalid"):
            self._build()

    def test_unified_features_do_not_enter_frozen_24h_contract(self) -> None:
        frame, _, _ = self._build()
        frozen = set(_config("24h")["numerical_features"]) | set(
            _config("24h")["categorical_features"]
        )
        context_columns = set(frame.columns) - {
            "road_segment_id",
            "prediction_datetime",
            "forecast_window_start",
            "forecast_window_end",
            "horizon_hours",
            "generated_at",
        }
        self.assertFalse(context_columns & frozen)


if __name__ == "__main__":
    unittest.main()
