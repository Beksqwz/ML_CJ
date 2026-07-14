import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import requests

from future_intelligence.geocoding import (
    AstanaGeocoder,
    GeocodeResult,
    RoadGeometryResolver,
    apply_geocode,
)
from future_intelligence.schemas import FutureRecord


class Response:
    def __init__(self, payload=None, status_code=200):
        self.payload = [] if payload is None else payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self.payload


class Session:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0
        self.requests = []

    def get(self, *args, **kwargs):
        self.calls += 1
        self.requests.append((args, kwargs))
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def astana_response(city="Astana", latitude="51.1", longitude="71.4"):
    return Response(
        [
            {
                "lat": latitude,
                "lon": longitude,
                "address": {"city": city} if city is not None else {},
                "display_name": f"Venue, {city or 'Kazakhstan'}",
            }
        ]
    )


class GeocodingTests(unittest.TestCase):
    def geocoder(self, *actions, max_attempts=3):
        self.session = Session(actions)
        return AstanaGeocoder(
            session=self.session,
            min_interval_seconds=0,
            max_attempts=max_attempts,
            sleep=lambda _: None,
        )

    def test_known_astana_arena_local_directory_hit(self):
        geocoder = self.geocoder()
        result = geocoder.event("Astana Arena", None)
        self.assertEqual(
            (result.quality, result.latitude, self.session.calls),
            ("exact_known_venue", 51.1083, 0),
        )

    def test_nominatim_astana_result_is_accepted_with_bounded_query(self):
        geocoder = self.geocoder(astana_response())
        result = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (
                result.quality,
                result.bbox_valid,
                result.city_valid,
                result.attempt_count,
            ),
            ("nominatim_exact_astana", True, True, 1),
        )
        params = self.session.requests[0][1]["params"]
        self.assertEqual(
            (params["viewbox"], params["bounded"]), ("71.2,51.3,71.6,51.0", 1)
        )

    def test_nominatim_almaty_result_is_rejected(self):
        geocoder = self.geocoder(astana_response("Almaty"))
        result = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (result.latitude, result.quality, result.city_valid),
            (None, "rejected_wrong_city", False),
        )

    def test_result_outside_astana_bbox_is_rejected(self):
        geocoder = self.geocoder(astana_response("Astana", "43.2389", "76.8897"))
        result = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (result.longitude, result.quality, result.bbox_valid),
            (None, "rejected_outside_astana", False),
        )

    def test_bbox_only_result_is_explicitly_lower_confidence(self):
        response = Response([{"lat": "51.1", "lon": "71.4", "address": {}}])
        result = self.geocoder(response).geocode("Unknown Hall")
        self.assertEqual(
            (result.quality, result.confidence, result.city_valid),
            ("nominatim_bbox_only", 0.55, None),
        )
        self.assertIn("city_metadata_missing_bbox_only", result.warnings)

    def test_timeout_retries_to_configured_limit(self):
        geocoder = self.geocoder(
            requests.Timeout(), requests.Timeout(), requests.Timeout()
        )
        result = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (self.session.calls, result.attempt_count, result.latitude), (3, 3, None)
        )
        self.assertIn("nominatim_retry_exhausted", result.warnings)

    def test_http_429_retries(self):
        geocoder = self.geocoder(Response(status_code=429), astana_response())
        result = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (self.session.calls, result.attempt_count, result.quality),
            (2, 2, "nominatim_exact_astana"),
        )

    def test_http_500_retries(self):
        geocoder = self.geocoder(Response(status_code=500), astana_response())
        result = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (self.session.calls, result.attempt_count, result.quality),
            (2, 2, "nominatim_exact_astana"),
        )

    def test_http_400_does_not_retry(self):
        geocoder = self.geocoder(Response(status_code=400), astana_response())
        result = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (self.session.calls, result.attempt_count, result.latitude), (1, 1, None)
        )
        self.assertIn("nominatim_request_failed", result.warnings)

    def test_successful_second_attempt_and_cache(self):
        geocoder = self.geocoder(requests.ConnectionError(), astana_response())
        first = geocoder.geocode("Unknown Hall")
        second = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (self.session.calls, first.attempt_count, second.cache_hit), (2, 2, True)
        )

    def test_rejected_result_is_cached(self):
        geocoder = self.geocoder(astana_response("Almaty"))
        first = geocoder.geocode("Unknown Hall")
        second = geocoder.geocode("Unknown Hall")
        self.assertEqual(
            (first.quality, second.cache_hit, self.session.calls),
            ("rejected_wrong_city", True, 1),
        )

    def test_repair_point_fallback_is_low_confidence_without_segment_id(self):
        geocoder = self.geocoder(astana_response())
        result = geocoder.repair({"road_name": "проспект Туран"})
        self.assertEqual(result.confidence, 0.35)
        self.assertIn("repair_point_fallback_not_segment_geometry", result.warnings)
        record = FutureRecord(
            "x",
            "repairs",
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
        )
        apply_geocode(record, result)
        self.assertEqual(record.affected_road_segment_ids, [])

    def test_result_is_written_as_geojson_point(self):
        record = FutureRecord(
            "x",
            "events",
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
        )
        apply_geocode(
            record,
            GeocodeResult(51.1, 71.4, "nominatim_exact_astana", "nominatim", "x", 0.7),
        )
        self.assertEqual(record.geometry["type"], "Point")

    def test_from_to_uses_network_line_not_street_centre(self):
        text = (
            "u,v,key,name,length,geometry\n"
            '1,2,0,Туран,10,"LINESTRING (71 51, 72 51)"\n'
            '2,3,0,Туран,10,"LINESTRING (72 51, 73 51)"\n'
            '1,9,0,Достык,10,"LINESTRING (71 51, 71 52)"\n'
            '3,8,0,Коргалжын,10,"LINESTRING (73 51, 73 52)"\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roads.csv"
            path.write_text(text, encoding="utf-8")
            result = RoadGeometryResolver(path).repair(
                {
                    "road_name": "проспект Туран",
                    "from_street": "улица Достык",
                    "to_street": "шоссе Коргалжын",
                }
            )
        self.assertEqual(
            (result.quality, result.geometry["type"]),
            ("road_from_to_network", "LineString"),
        )
