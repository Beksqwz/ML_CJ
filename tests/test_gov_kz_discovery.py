import unittest

from future_intelligence.providers.repairs.discovery import (
    GovKzJsonDiscovery,
    GovKzSitemapDiscovery,
    unique,
    valid,
)


XML = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://www.gov.kz/memleket/entities/astana/press/news/details/101?lang=ru</loc></url><url><loc>https://external.example/news/details/102</loc></url><url><loc>https://www.gov.kz/memleket/entities/other/press/news/details/103</loc></url><url><loc>https://www.gov.kz/memleket/entities/astana/press/news/details/101?lang=ru</loc></url></urlset>"""


class GovKzDiscoveryTests(unittest.TestCase):
    def test_sitemap_discovers_only_valid_official_urls(self):
        result = GovKzSitemapDiscovery(lambda _: XML).discover(max_articles=10)
        self.assertEqual(result.status, "ok")
        self.assertEqual(
            result.article_urls,
            [
                "https://www.gov.kz/memleket/entities/astana/press/news/details/101?lang=ru"
            ],
        )

    def test_empty_and_changed_sitemap_are_safe(self):
        empty = GovKzSitemapDiscovery(
            lambda _: XML.replace("details/101?lang=ru", "not-an-article")
        ).discover(max_articles=10)
        changed = GovKzSitemapDiscovery(lambda _: "<html>changed</html>").discover(
            max_articles=10
        )
        self.assertEqual(empty.status, "empty")
        self.assertIn(changed.status, {"empty", "failed"})

    def test_json_is_explicitly_skipped_until_verified(self):
        result = GovKzJsonDiscovery().discover()
        self.assertEqual(result.status, "skipped")

    def test_url_validation_and_deduplication(self):
        url = (
            "https://www.gov.kz/memleket/entities/astana/press/news/details/99?lang=ru"
        )
        self.assertTrue(valid(url))
        self.assertFalse(
            valid(
                "https://external.example/memleket/entities/astana/press/news/details/99"
            )
        )
        self.assertEqual(unique([url, url], 5), [url])
