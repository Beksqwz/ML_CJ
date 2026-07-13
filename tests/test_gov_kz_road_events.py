import unittest
from datetime import datetime
from pathlib import Path

from future_intelligence.providers.repairs.gov_kz import GovKzRoadEventsProvider, BASE
from future_intelligence.utils import ASTANA_TIMEZONE
from future_intelligence.validation import validate_result

FIXTURES = Path(__file__).parent / "fixtures"


def article(title, body):
    return f"<html><body><h1>{title}</h1><time>3 апреля 2026</time><article>{body}</article></body></html>"


class GovKzRoadEventTests(unittest.TestCase):
    def setUp(self):
        self.provider = GovKzRoadEventsProvider(request_delay=0, sleep=lambda _: None)

    def parse(self, title, body, ident="1"):
        return self.provider.parse_article(
            article(title, body),
            f"{BASE}/memleket/entities/astana/press/news/details/{ident}?lang=ru",
            ident,
        )

    def test_listing_parsing(self):
        html = (FIXTURES / "gov_kz_listing.html").read_text(encoding="utf-8")
        self.provider._get = lambda url: (
            "User-agent: *\nAllow: /" if url.endswith("robots.txt") else html
        )
        links, report = self.provider.discover("ru", 1)
        self.assertEqual(len(links), 2)
        self.assertTrue(report["robots_allows_listing"])

    def test_repair_from_to_dates_and_location(self):
        parsed = self.provider.parse_article(
            (FIXTURES / "gov_kz_repair.html").read_text(encoding="utf-8"), "x", "101"
        )
        self.assertTrue(parsed["relevant"])
        self.assertEqual(parsed["event_type"], "traffic_restriction")
        self.assertEqual(parsed["restriction_type"], "partial_closure")
        self.assertIsNotNone(parsed["valid_from"])
        self.assertIsNotNone(parsed["valid_to"])
        self.assertIsNotNone(parsed["location"]["from_street"])

    def test_full_closure_intersection_bridge_and_lane(self):
        full = self.parse(
            "Перекрытие",
            "С 20:00 17 мая полностью перекрыто движение на пересечении улиц А и Б в связи с ремонтом.",
        )
        bridge = self.parse(
            "Мост",
            "С 4 апреля по 4 мая ремонт моста по проспекту Туран, движение ограничено.",
        )
        lane = self.parse(
            "Полоса",
            "С 4 апреля по 4 мая ремонт дороги по улице А, закрытие полосы движения.",
        )
        self.assertEqual(full["event_type"], "intersection_closure")
        self.assertEqual(full["severity"], "critical")
        self.assertEqual(bridge["event_type"], "bridge_repair")
        self.assertEqual(lane["restriction_type"], "single_lane_closure")

    def test_open_end_exact_hours_irrelevant_and_kazakh(self):
        open_end = self.parse(
            "Ремонт", "С 20:00 17 мая ремонт улицы А до завершения ремонтных работ."
        )
        irrelevant = self.parse("Фестиваль", "В городе пройдет концерт и праздник.")
        kazakh = self.parse(
            "Жол жөндеу",
            "4 сәуірден 4 мамырға дейін А көшесінде жол жөндеу жұмыстарына байланысты қозғалысқа шектеу енгізіледі.",
        )
        self.assertTrue(open_end["open_end"])
        self.assertEqual(open_end["valid_from"].hour, 20)
        self.assertFalse(irrelevant["relevant"])
        self.assertTrue(kazakh["relevant"])

    def test_hash_update_and_universal_features_are_deterministic(self):
        first = self.parse(
            "Ремонт", "С 4 апреля по 4 мая ремонт улицы А, движение ограничено.", "7"
        )
        changed = self.parse(
            "Ремонт", "С 4 апреля по 5 мая ремонт улицы А, движение ограничено.", "7"
        )
        self.assertNotEqual(first["content_hash"], changed["content_hash"])
        when = datetime(2026, 4, 4, tzinfo=ASTANA_TIMEZONE)
        records = self.provider.normalize({"parsed": first}, when, 24)
        result = type(
            "R",
            (),
            {
                "normalized_records": records,
                "features": self.provider.build_features(records, when, 24),
            },
        )
        self.assertIn("repair_disruption_score", result.features)
        self.assertEqual(
            validate_result(
                type(
                    "X",
                    (),
                    {
                        "status": "ok",
                        "fallback_used": False,
                        "features": result.features,
                        "normalized_records": records,
                    },
                )()
            ),
            [],
        )
