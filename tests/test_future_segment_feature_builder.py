import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from future_intelligence.segment_feature_builder import (
    BuilderPaths,
    FEATURE_COLUMNS,
    FutureSegmentFeatureBuilder,
    feature_catalog,
    validate_features,
)
from future_intelligence.utils import ASTANA_TIMEZONE


WHEN = datetime(2026, 7, 15, 8, tzinfo=ASTANA_TIMEZONE)


def event_row(source, source_item_id, start, end, **extra):
    return {
        "source": source,
        "source_type": "events",
        "source_version": "1",
        "source_item_id": source_item_id,
        "source_url": None,
        "collected_at": WHEN.isoformat(),
        "published_at": None,
        "valid_from": start,
        "valid_to": end,
        "prediction_datetime": WHEN.isoformat(),
        "horizon_hours": 24,
        "latitude": None,
        "longitude": None,
        "geometry": None,
        "affected_road_segment_ids": [],
        "event_type": extra.get("event_type"),
        "severity": extra.get("severity"),
        "confidence": 0.8,
        "is_forecast": True,
        "is_realtime": False,
        "is_historical": False,
        "restriction_type": extra.get("restriction_type"),
        "open_end": extra.get("open_end", False),
        "payload_json": json.dumps(extra.get("payload", {})),
    }


class FutureSegmentFeatureBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        processed = root / "processed"
        processed.mkdir()
        self.road_path = root / "roads.parquet"
        pd.DataFrame({"road_segment_id": ["s1", "s2", "s3"]}).to_parquet(self.road_path)
        self.weather_path = processed / "weather.parquet"
        pd.DataFrame(
            [
                {
                    "prediction_datetime": WHEN.isoformat(),
                    "weather_temperature_mean": 12.5,
                    "weather_rain_sum": 2.0,
                    "weather_visibility_min": 500,
                }
            ]
        ).to_parquet(self.weather_path)
        self.gov_path = processed / "gov.parquet"
        pd.DataFrame(
            [
                event_row(
                    "gov.kz Astana Akimat",
                    "g1",
                    "2026-07-15T10:00:00+05:00",
                    "2026-07-16T10:00:00+05:00",
                    event_type="road_closure",
                    severity="high",
                    restriction_type="full_closure",
                    payload={"restriction_type": "full_closure"},
                ),
                event_row(
                    "gov.kz Astana Akimat",
                    "g2",
                    "2026-07-17T10:00:00+05:00",
                    "2026-07-18T10:00:00+05:00",
                    event_type="road_repair",
                    severity="low",
                ),
                event_row(
                    "gov.kz Astana Akimat",
                    "g3",
                    "2026-07-14T10:00:00+05:00",
                    None,
                    event_type="road_repair",
                    severity="medium",
                    open_end=True,
                ),
            ]
        ).to_parquet(self.gov_path)
        self.ticketon_path = processed / "ticketon.parquet"
        pd.DataFrame(
            [
                event_row(
                    "Ticketon",
                    "e1",
                    "2026-07-15T12:00:00+05:00",
                    "2026-07-15T15:00:00+05:00",
                    event_type="large_concert",
                    payload={
                        "event_severity": 5,
                        "event_intensity_score": 7.5,
                        "venue": "Astana Arena",
                    },
                ),
                event_row(
                    "Ticketon",
                    "e2",
                    "2026-07-17T12:00:00+05:00",
                    "2026-07-17T15:00:00+05:00",
                    event_type="football_match",
                    payload={
                        "event_severity": 5,
                        "event_intensity_score": 5,
                        "venue": "Astana Arena",
                    },
                ),
            ]
        ).to_parquet(self.ticketon_path)
        self.matches_path = processed / "matches.parquet"
        pd.DataFrame(
            [
                {
                    "provider": "gov_kz_repairs",
                    "source_item_id": "g1",
                    "road_segment_id": "s1",
                    "distance_m": 10.0,
                    "match_confidence": 0.8,
                },
                {
                    "provider": "gov_kz_repairs",
                    "source_item_id": "g3",
                    "road_segment_id": "s1",
                    "distance_m": 20.0,
                    "match_confidence": 0.6,
                },
                {
                    "provider": "gov_kz_repairs",
                    "source_item_id": "g2",
                    "road_segment_id": "s2",
                    "distance_m": 5.0,
                    "match_confidence": 0.9,
                },
                {
                    "provider": "ticketon_events",
                    "source_item_id": "e1",
                    "road_segment_id": "s2",
                    "distance_m": 400.0,
                    "match_confidence": 0.9,
                },
                {
                    "provider": "ticketon_events",
                    "source_item_id": "e1",
                    "road_segment_id": "s1",
                    "distance_m": 1500.0,
                    "match_confidence": 0.7,
                },
                {
                    "provider": "ticketon_events",
                    "source_item_id": "e2",
                    "road_segment_id": "s3",
                    "distance_m": 100.0,
                    "match_confidence": 0.9,
                },
                {
                    "provider": "ticketon_events",
                    "source_item_id": "bad",
                    "road_segment_id": "bad_id",
                    "distance_m": -1.0,
                    "match_confidence": 2.0,
                },
            ]
        ).to_parquet(self.matches_path)
        self.paths = BuilderPaths(
            self.road_path,
            self.matches_path,
            self.weather_path,
            self.gov_path,
            self.ticketon_path,
            root,
        )

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return FutureSegmentFeatureBuilder(
            self.paths, expected_segment_count=None
        ).build(WHEN)

    def test_one_row_per_production_segment(self):
        frame, _ = self.build()
        self.assertEqual((len(frame), frame.road_segment_id.nunique()), (3, 3))

    def test_weather_is_broadcast(self):
        frame, _ = self.build()
        self.assertEqual(frame.weather_temperature_mean.tolist(), [12.5, 12.5, 12.5])
        self.assertEqual(frame.weather_available.tolist(), [1, 1, 1])

    def test_repair_window_and_full_closure(self):
        frame, _ = self.build()
        row = frame.set_index("road_segment_id").loc["s1"]
        self.assertEqual(
            (row.repair_event_count_next_24h, row.repair_full_closure_next_24h), (2, 1)
        )

    def test_repair_outside_window_is_excluded(self):
        frame, _ = self.build()
        self.assertEqual(
            frame.set_index("road_segment_id").loc["s2", "repair_event_count_next_24h"],
            0,
        )

    def test_repair_distance_confidence_and_open_end(self):
        row = self.build()[0].set_index("road_segment_id").loc["s1"]
        self.assertEqual(
            (
                row.repair_distance_to_nearest_m,
                row.repair_match_confidence_max,
                row.repair_open_end_count_next_24h,
            ),
            (10.0, 0.8, 1),
        )

    def test_event_window_radius_buckets_and_intensity(self):
        frame, _ = self.build()
        row = frame.set_index("road_segment_id").loc["s2"]
        self.assertEqual(
            (
                row.event_count_next_24h,
                row.event_count_500m_next_24h,
                row.event_count_1000m_next_24h,
            ),
            (1, 1, 1),
        )
        self.assertEqual(
            (row.event_intensity_sum_next_24h, row.event_intensity_max_next_24h),
            (7.5, 7.5),
        )

    def test_event_classification_and_post_event_outflow(self):
        row = self.build()[0].set_index("road_segment_id").loc["s2"]
        self.assertEqual(
            (
                row.event_major_next_24h,
                row.event_concert_next_24h,
                row.event_stadium_next_24h,
                row.event_post_event_outflow_risk,
            ),
            (1, 1, 1, 1),
        )

    def test_no_event_defaults_are_valid(self):
        row = self.build()[0].set_index("road_segment_id").loc["s3"]
        self.assertEqual(
            (
                row.event_count_next_24h,
                row.event_intensity_sum_next_24h,
                pd.isna(row.event_distance_to_nearest_m),
            ),
            (0, 0, True),
        )

    def test_invalid_stage17_match_is_rejected(self):
        _, report = self.build()
        self.assertIn("stage17_invalid_matches_rejected:1", report["warnings"])

    def test_missing_provider_is_explicitly_degraded(self):
        missing = BuilderPaths(
            self.road_path,
            self.matches_path,
            Path("missing.parquet"),
            self.gov_path,
            Path("missing-ticketon.parquet"),
            self.paths.output_dir,
        )
        frame, report = FutureSegmentFeatureBuilder(
            missing, expected_segment_count=None
        ).build(WHEN)
        self.assertEqual(frame.weather_available.iloc[0], 0)
        self.assertEqual(frame.ticketon_provider_available.iloc[0], 0)
        self.assertTrue(report["provider_availability"]["weather"]["warnings"])

    def test_deterministic_storage_and_schema(self):
        builder = FutureSegmentFeatureBuilder(self.paths, expected_segment_count=None)
        frame, report = builder.build(WHEN)
        first = builder.save(frame, report)
        second = builder.save(*builder.build(WHEN))
        self.assertEqual(
            (first["checksum"], second["checksum"], second["unchanged"]),
            (first["checksum"], first["checksum"], True),
        )
        self.assertTrue(pd.read_parquet(first["parquet"]).equals(frame))
        self.assertTrue(json.loads(Path(first["json"]).read_text(encoding="utf-8")))

    def test_catalog_and_validation(self):
        frame, _ = self.build()
        catalog = feature_catalog()
        validation = validate_features(frame, {"s1", "s2", "s3"}, 3)
        self.assertEqual(len(catalog), len(frame.columns) - 6)
        self.assertTrue(
            all(item["training_status"] != "trainable_now" for item in catalog)
        )
        self.assertTrue(all(validation.values()))

    def test_feature_schema_has_unique_weather_summary_columns(self):
        self.assertEqual(len(FEATURE_COLUMNS), len(set(FEATURE_COLUMNS)))
        self.assertEqual(FEATURE_COLUMNS.count("weather_summary_temperature_min"), 1)
        self.assertEqual(FEATURE_COLUMNS.count("weather_summary_temperature_max"), 1)

    def test_frozen_24h_feature_hash_is_unchanged(self):
        from tests.test_future_intelligence_pipeline import _config

        features = list(_config("24h")["numerical_features"]) + list(
            _config("24h")["categorical_features"]
        )
        self.assertEqual(len(features), 77)
        self.assertEqual(
            hashlib.sha256(
                json.dumps(features, separators=(",", ":")).encode()
            ).hexdigest(),
            "bf0ce06aface9fc3539a95df2557728e57d987a9e0c0e5ac4a195a53b47f6d96",
        )
