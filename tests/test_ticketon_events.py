import unittest
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from future_intelligence.providers.events.ticketon import TicketonEventsProvider
from future_intelligence.schemas import ProviderResult
from future_intelligence.storage import save_ticketon_result
from future_intelligence.utils import ASTANA_TIMEZONE
from future_intelligence.validation import validate_result

HTML = """<script type="application/ld+json">{"@context":"https://schema.org","@type":"Event","@id":"event-1","name":"Футбольный матч","category":"Sport","startDate":"2026-07-14T19:00:00+05:00","endDate":"2026-07-14T22:00:00+05:00","url":"/event/1","location":{"@type":"Place","name":"Astana Arena","address":{"streetAddress":"пр. Кабанбай батыра"}}}</script>"""


class TicketonEventsTests(unittest.TestCase):
    def setUp(self):
        self.provider = TicketonEventsProvider(sleep=lambda _: None)

    def test_jsonld_normalizes_known_venue_and_intensity(self):
        parsed = self.provider.parse_listing(HTML)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(
            (
                parsed[0]["event_type"],
                parsed[0]["event_severity"],
                parsed[0]["geocoding_quality"],
            ),
            ("football_match", 5, "local_directory"),
        )
        when = datetime(2026, 7, 14, tzinfo=ASTANA_TIMEZONE)
        records = self.provider.normalize({"events": parsed}, when, 24)
        features = self.provider.build_features(records, when, 24)
        self.assertEqual(
            (
                features["event_major_count_next_24h"],
                features["event_stadium_count_next_24h"],
            ),
            (1, 1),
        )
        self.assertEqual(features["event_intensity_score"], 7.5)

    def test_only_events_overlapping_window_are_retained(self):
        self.assertEqual(
            self.provider.normalize(
                {"events": self.provider.parse_listing(HTML)},
                datetime(2026, 7, 15, tzinfo=ASTANA_TIMEZONE),
                24,
            ),
            [],
        )

    def test_collection_is_safe_on_empty_listing(self):
        self.provider._robots_allow_listing = lambda: True
        self.provider._get = lambda _: "<html></html>"
        result = self.provider.collect(
            datetime(2026, 7, 14, tzinfo=ASTANA_TIMEZONE), 24
        )
        self.assertEqual(result.status, "degraded")
        self.assertTrue(result.fallback_used)
        self.assertEqual(validate_result(result), [])

    def test_robots_uses_only_the_applicable_user_agent_group(self):
        self.provider._get = lambda _: (
            "User-agent: GPTBot\nDisallow: /\nUser-agent: *\nAllow: /\nDisallow: /admin/"
        )
        self.assertTrue(self.provider._robots_allow_listing())

    def test_detail_schema_sub_events_and_listing_links(self):
        detail = """<script type="application/ld+json">{"@type":"ExhibitionEvent","name":"Выставка","url":"/event/a","startDate":"2026-07-14T10:00:00+05:00","location":{"name":"EXPO","address":{"addressLocality":"Астана"}},"subEvent":[{"@type":"ExhibitionEvent","name":"Выставка","url":"/event/a","startDate":"2026-07-14T12:00:00+05:00","location":{"name":"EXPO","address":{"addressLocality":"Астана"}}}]}</script>"""
        self.assertEqual(len(self.provider.parse_listing(detail)), 2)
        self.assertEqual(
            self.provider.discover_event_urls('<a href="/astana/event/test">x</a>'),
            ["https://ticketon.kz/astana/event/test"],
        )

    def test_sub_event_is_preferred_to_its_broad_parent(self):
        detail = """<script type="application/ld+json">{"@type":"ExhibitionEvent","name":"Выставка","startDate":"2026-07-14T10:00:00+05:00","endDate":"2027-01-01T10:00:00+05:00","location":{"name":"EXPO"},"subEvent":{"@type":"ExhibitionEvent","name":"Выставка","startDate":"2026-07-14T10:00:00+05:00","endDate":"2026-07-14T12:00:00+05:00","location":{"name":"EXPO"}}}</script>"""
        parsed = self.provider.parse_listing(detail)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["valid_to"].hour, 12)

    def test_canonical_ticketon_storage_is_idempotent_and_exports_json(self):
        when = datetime(2026, 7, 14, tzinfo=ASTANA_TIMEZONE)
        records = self.provider.normalize(
            {"events": self.provider.parse_listing(HTML)}, when, 24
        )
        result = ProviderResult(
            self.provider.metadata,
            [{"listing_url": "https://ticketon.kz/astana", "event_count": 1}],
            records,
            self.provider.build_features(records, when, 24),
            {"city": "Astana", "records": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths, first = save_ticketon_result(result, output_dir)
            _, second = save_ticketon_result(result, output_dir)
            self.assertEqual(first, {"new": 1, "updated": 0, "unchanged": 0})
            self.assertEqual(second, {"new": 0, "updated": 0, "unchanged": 1})
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertEqual(len(pd.read_parquet(paths["events"])), 1)
            self.assertEqual(
                len(json.loads(paths["json"].read_text(encoding="utf-8"))), 1
            )
            records[0].payload["name"] = "Updated event title"
            _, updated = save_ticketon_result(result, output_dir)
            self.assertEqual(updated, {"new": 0, "updated": 1, "unchanged": 0})
