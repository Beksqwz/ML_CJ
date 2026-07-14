import unittest

from future_intelligence.history.ticketon_wayback_normalizer import (
    TicketonWaybackNormalizer,
    classify_transport,
)


HTML = """<script type="application/ld+json">{"@context":"https://schema.org","@type":"MusicEvent","name":"Concert","startDate":"2024-05-01T20:00:00+05:00","endDate":"2024-05-01T22:00:00+05:00","location":{"@type":"Place","name":"Astana Arena","address":{"@type":"PostalAddress","addressLocality":"Astana"}},"offers":{"@type":"Offer","price":"5000"}}</script>"""


class TicketonWaybackNormalizerTests(unittest.TestCase):
    def test_normalizes_event_with_existing_venue_coordinate(self):
        events, audit = TicketonWaybackNormalizer().normalize(
            HTML,
            archive_year=2024,
            original_url="https://ticketon.kz/astana/event/test",
        )
        self.assertEqual(audit["normalized_events"], 1)
        self.assertTrue(events[0]["astana_valid"])
        self.assertEqual(events[0]["venue_tier"], "very_high")
        self.assertIsNotNone(events[0]["latitude"])
        self.assertEqual(events[0]["training_eligibility"], "trainable")

    def test_entity_encoded_jsonld_is_supported(self):
        encoded = (
            HTML.replace('<script type="application/ld+json">', "")
            .replace("</script>", "")
            .replace('"', "&quot;")
        )
        events, _ = TicketonWaybackNormalizer().normalize(
            f'<script type="application/ld+json">{encoded}</script>',
            archive_year=2024,
            original_url="https://ticketon.kz/astana/event/test",
        )
        self.assertEqual(len(events), 1)

    def test_transport_exclusions_and_unknown_low(self):
        excluded = classify_transport(
            {"title": "Online workshop", "venue_tier": "unknown"}
        )
        unknown = classify_transport(
            {"title": "Unknown event", "venue_tier": "unknown"}
        )
        self.assertEqual(excluded["transport_class"], "exclude")
        self.assertEqual(unknown["transport_class"], "low")

    def test_archive_jsonld_literal_newline_is_supported(self):
        archived = HTML.replace('"Concert"', '"Concert\nlive"')
        events, _ = TicketonWaybackNormalizer().normalize(
            archived,
            archive_year=2024,
            original_url="https://ticketon.kz/astana/event/test",
        )
        self.assertEqual(len(events), 1)
