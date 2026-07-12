import tempfile
import unittest
from pathlib import Path

from ml_service.traffic import SegmentPoint, TomTomTrafficService


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"flowSegmentData": {"currentSpeed": 20, "freeFlowSpeed": 40, "confidence": 0.8}}


class FakeSession:
    def get(self, *args, **kwargs):
        return FakeResponse()


class TrafficTests(unittest.TestCase):
    def setUp(self):
        self.points = {"s1": SegmentPoint(latitude=51.1, longitude=71.4)}

    def test_missing_key_is_a_safe_fallback(self):
        result = TomTomTrafficService(api_key="", segment_points=self.points).get_segment("s1")
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "tomtom_api_key_not_configured")

    def test_flow_result_and_parquet_snapshot(self):
        service = TomTomTrafficService(api_key="test", session=FakeSession(), segment_points=self.points)
        result = service.get_segment("s1")
        self.assertTrue(result["available"])
        self.assertEqual(result["congestion_ratio"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.parquet"
            service.collect(["s1"], path)
            self.assertTrue(path.is_file())
