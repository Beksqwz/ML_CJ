import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from future_intelligence.providers.repairs.discovery import prefilter_listing_card
from future_intelligence.providers.repairs.gov_kz import GovKzRoadEventsProvider
from future_intelligence.utils import ASTANA_TIMEZONE


FIXTURES = Path(__file__).parent / "fixtures"
ROAD_URL = (
    "https://www.gov.kz/memleket/entities/astana/press/news/details/1254889?lang=ru"
)
NON_ROAD_URL = (
    "https://www.gov.kz/memleket/entities/astana/press/news/details/1255243?lang=ru"
)


class FakeOfficialListingSession:
    rendered = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def official_filtered_listing(self, max_pages, max_articles):
        return [ROAD_URL], {
            "pages_scanned": 2,
            "listing_cards_seen": 4,
            "listing_cards_selected": 1,
            "duplicate_urls_removed": 1,
            "listing_card_diagnostics": [
                {"source_url": ROAD_URL, "candidate_selected": True},
                {
                    "source_url": NON_ROAD_URL,
                    "candidate_selected": False,
                    "rejection_reason": "no_strong_road_event_term_or_weak_term_with_road_context",
                },
            ],
        }

    def render_article(self, url):
        self.rendered.append(url)
        return (
            "<html><h1>Улицу К. Аманжолова частично закроют в связи с ремонтными работами</h1>"
            "<time>8 июля 2026</time><article>С 9 июля движение по улице Аманжолова "
            "будет частично перекрыто в связи с ремонтом дороги.</article></html>",
            b"png",
        )


class OfficialFilteredListingTests(unittest.TestCase):
    def test_listing_fixture_has_road_and_nonroad_cards(self):
        html = (FIXTURES / "gov_kz_listing_cards.html").read_text(encoding="utf-8")
        self.assertIn("1254889", html)
        self.assertIn("1255243", html)

    def test_strong_repair_and_restriction_cards_are_selected(self):
        strong = prefilter_listing_card("Средний ремонт дороги на проспекте", "")
        restricted = prefilter_listing_card(
            "Временное ограничение движения", "на улице Абая"
        )
        self.assertTrue(strong["candidate_selected"])
        self.assertTrue(restricted["candidate_selected"])

    def test_weak_generic_repair_without_road_context_is_rejected(self):
        result = prefilter_listing_card("Ремонт в школе завершён", "")
        self.assertFalse(result["candidate_selected"])
        self.assertIsNotNone(result["rejection_reason"])

    def test_kazakh_road_card_is_selected(self):
        result = prefilter_listing_card(
            "Жол жөндеу жұмыстарына байланысты көше жабылады", ""
        )
        self.assertTrue(result["candidate_selected"])

    @patch(
        "future_intelligence.providers.repairs.gov_kz.GovKzPlaywrightArticleSession",
        FakeOfficialListingSession,
    )
    def test_only_prefiltered_candidate_is_rendered_then_parsed(self):
        FakeOfficialListingSession.rendered = []
        provider = GovKzRoadEventsProvider(max_pages=2, max_articles=5, request_delay=0)
        provider._get = lambda _: '<div id="root"></div>'
        result = provider.collect(
            datetime(2026, 7, 8, tzinfo=ASTANA_TIMEZONE),
            24,
            discovery_method="official-filtered",
            force_refresh=True,
        )
        self.assertEqual(FakeOfficialListingSession.rendered, [ROAD_URL])
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.normalized_records), 1)
        self.assertEqual(provider.last_report["listing_cards_seen"], 4)
        self.assertEqual(provider.last_report["detail_pages_rendered"], 1)
        self.assertEqual(provider.last_report["relevant_articles"], 1)

    @patch(
        "future_intelligence.providers.repairs.gov_kz.GovKzPlaywrightArticleSession",
        FakeOfficialListingSession,
    )
    def test_official_listing_path_does_not_require_search(self):
        provider = GovKzRoadEventsProvider(max_pages=2, max_articles=5, request_delay=0)
        provider._get = lambda _: '<div id="root"></div>'
        result = provider.collect(
            datetime(2026, 7, 8, tzinfo=ASTANA_TIMEZONE),
            24,
            discovery_method="official-filtered",
            force_refresh=True,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(
            provider.last_report["selected_method"], "official_filtered_listing"
        )

    @patch(
        "future_intelligence.providers.repairs.gov_kz.GovKzPlaywrightArticleSession",
        FakeOfficialListingSession,
    )
    def test_known_article_is_skipped_without_force_refresh(self):
        FakeOfficialListingSession.rendered = []
        provider = GovKzRoadEventsProvider(max_pages=2, max_articles=5, request_delay=0)
        provider._get = lambda _: '<div id="root"></div>'
        provider._known_source_item_ids = lambda: {"1254889"}
        result = provider.collect(
            datetime(2026, 7, 8, tzinfo=ASTANA_TIMEZONE),
            24,
            discovery_method="official-filtered",
        )
        self.assertEqual(FakeOfficialListingSession.rendered, [])
        self.assertEqual(provider.last_report["already_known_articles_skipped"], 1)
        self.assertEqual(result.normalized_records, [])
