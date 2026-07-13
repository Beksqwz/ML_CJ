import unittest
import tempfile
from pathlib import Path

from future_intelligence.geocoding import AstanaGeocoder, RoadGeometryResolver, apply_geocode
from future_intelligence.schemas import FutureRecord
from datetime import datetime


class Response:
    def raise_for_status(self): pass
    def json(self): return [{"lat": "51.1", "lon": "71.4"}]


class Session:
    def __init__(self): self.calls = 0
    def get(self, *args, **kwargs): self.calls += 1; return Response()


class GeocodingTests(unittest.TestCase):
    def test_local_venue_precedes_network(self):
        session = Session(); result = AstanaGeocoder(session=session, sleep=lambda _: None).event("Astana Arena", None)
        self.assertEqual((result.quality, result.latitude, session.calls), ("local_venue_directory", 51.1083, 0))
    def test_nominatim_result_is_cached(self):
        session = Session(); geocoder = AstanaGeocoder(session=session, min_interval_seconds=0, sleep=lambda _: None)
        self.assertEqual(geocoder.event("Unknown Hall", "Туран 1").quality, "nominatim")
        geocoder.event("Unknown Hall", "Туран 1")
        self.assertEqual(session.calls, 1)
    def test_repair_is_explicitly_point_only(self):
        result = AstanaGeocoder(session=Session(), min_interval_seconds=0, sleep=lambda _: None).repair({"road_name": "проспект Туран"})
        self.assertIn("repair_line_geometry_pending", result.warnings)
    def test_result_is_written_as_geojson_point(self):
        record = FutureRecord("x", "events", "1", "id", None, datetime.now(), None, None, None, datetime.now(), 24, None, None)
        apply_geocode(record, AstanaGeocoder(session=Session(), min_interval_seconds=0, sleep=lambda _: None).event("Unknown", "Turan 1"))
        self.assertEqual(record.geometry["type"], "Point")
    def test_from_to_uses_network_line_not_street_centre(self):
        text = "u,v,key,name,length,geometry\n1,2,0,Туран,10,\"LINESTRING (71 51, 72 51)\"\n2,3,0,Туран,10,\"LINESTRING (72 51, 73 51)\"\n1,9,0,Достык,10,\"LINESTRING (71 51, 71 52)\"\n3,8,0,Коргалжын,10,\"LINESTRING (73 51, 73 52)\"\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roads.csv"; path.write_text(text, encoding="utf-8")
            result = RoadGeometryResolver(path).repair({"road_name": "проспект Туран", "from_street": "улица Достык", "to_street": "шоссе Коргалжын"})
        self.assertEqual((result.quality, result.geometry["type"]), ("road_from_to_network", "LineString"))
