"""Unit coverage for the historical gov.kz orchestration, without network I/O."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from future_intelligence.history.gov_kz_history import (
    GovKzHistoricalBackfill,
    official_detail_id,
    temporal_metadata,
    training_eligibility,
    unique_detail_urls,
)
from future_intelligence.schemas import FutureRecord
from future_intelligence.spatial_matching import SegmentMatch
from future_intelligence.utils import ASTANA_TIMEZONE


URL = "https://www.gov.kz/memleket/entities/astana/press/news/details/123?lang=ru"


class GovKzHistoryTests(unittest.TestCase):
    def test_only_official_astana_details_are_accepted_and_deduplicated(self):
        urls, dropped = unique_detail_urls([URL, URL, "https://example.test/123"])
        self.assertEqual(urls, [URL])
        self.assertEqual(dropped, 2)
        self.assertEqual(official_detail_id(URL), "123")

    def test_checkpoint_makes_listing_resume_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            history = GovKzHistoricalBackfill(Mock(), output_dir=Path(directory))
            self.assertEqual(
                len(history.register_listing_urls([URL], 1)["accepted_urls"]), 1
            )
            self.assertEqual(
                history.register_listing_urls([URL], 2)["accepted_urls"], []
            )

    def test_event_started_before_publication_has_explicit_leakage_warning(self):
        record = FutureRecord(
            "gov.kz Astana Akimat",
            "repairs",
            "1.0",
            "123",
            URL,
            datetime(2024, 5, 2, tzinfo=ASTANA_TIMEZONE),
            datetime(2024, 5, 2, tzinfo=ASTANA_TIMEZONE),
            datetime(2024, 5, 1, tzinfo=ASTANA_TIMEZONE),
            None,
            datetime(2024, 5, 2, tzinfo=ASTANA_TIMEZONE),
            24,
            None,
            None,
        )
        metadata = temporal_metadata(
            record, datetime(2024, 6, 1, tzinfo=ASTANA_TIMEZONE)
        )
        self.assertIn(
            "event_started_before_publication_use_as_known_at_only",
            metadata["temporal_leakage_warning"],
        )
        self.assertTrue(metadata["as_known_at_supported"])

    def test_training_eligibility_requires_match_and_operational_date(self):
        event = {
            "source_url": URL,
            "relevant": True,
            "published_at": "2024-01-01T00:00:00+05:00",
            "valid_from": "2024-01-02T00:00:00+05:00",
            "valid_to": None,
            "open_end": True,
            "historical_at_audit_time": True,
            "temporal_knowledge_eligible": True,
        }
        eligible, reasons = training_eligibility(event, [])
        self.assertFalse(eligible)
        self.assertIn("no_valid_production_segment_match", reasons)

    def test_canonical_storage_is_idempotent(self):
        event = {
            "source": "gov.kz Astana Akimat",
            "source_item_id": "123",
            "source_url": URL,
            "title": "repair",
            "training_eligible": False,
            "warnings": [],
        }
        match = SegmentMatch(
            "gov_kz_repairs",
            "123",
            "1_2_3",
            0.0,
            "line_intersection",
            0.9,
            "road_from_to_network",
            None,
            None,
            datetime(2024, 1, 1, tzinfo=ASTANA_TIMEZONE),
            datetime(2024, 1, 1, tzinfo=ASTANA_TIMEZONE),
        )
        with tempfile.TemporaryDirectory() as directory:
            history = GovKzHistoricalBackfill(Mock(), output_dir=Path(directory))
            first = history.save([event.copy()], [match])
            second = history.save([event.copy()], [match])
            self.assertEqual(first["writes"]["events"]["new"], 1)
            self.assertEqual(second["writes"]["events"]["unchanged"], 1)
            self.assertEqual(second["writes"]["matches"]["unchanged"], 1)


if __name__ == "__main__":
    unittest.main()
