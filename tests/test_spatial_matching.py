import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from future_intelligence.schemas import FutureRecord
from future_intelligence.spatial_matching import (
    SpatialMatchingEngine,
    save_segment_matches,
)


def record(
    identifier, *, source="Ticketon", geometry=None, latitude=None, longitude=None
):
    now = datetime(2026, 7, 14, tzinfo=UTC)
    return FutureRecord(
        source,
        "events" if source == "Ticketon" else "repairs",
        "1",
        identifier,
        None,
        now,
        None,
        now,
        None,
        now,
        24,
        latitude,
        longitude,
        geometry=geometry,
        payload={"geocoding_quality": "exact_known_venue"},
    )


class SpatialMatchingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.edges_path = Path(self.directory.name) / "astana_edges.csv"
        self.edges_path.write_text(
            "u,v,key,name,length,geometry\n"
            '1,2,0,Turan,100,"LINESTRING (71.400 51.108, 71.405 51.108)"\n'
            '2,3,0,Turan,100,"LINESTRING (71.405 51.108, 71.410 51.108)"\n'
            '4,5,0,Far,100,"LINESTRING (71.430 51.108, 71.435 51.108)"\n'
            '6,7,0,Outside,100,"LINESTRING (76.880 43.230, 76.890 43.230)"\n',
            encoding="utf-8",
        )
        self.engine = SpatialMatchingEngine(self.edges_path, ticketon_radius_m=1000)

    def tearDown(self):
        self.directory.cleanup()

    def test_known_astana_arena_radius_matching(self):
        result = self.engine.match_record(
            record("arena", latitude=51.1083, longitude=71.4027), "ticketon_events"
        )
        self.assertTrue(result.matches)
        self.assertTrue(
            all(match.match_type == "ticketon_radius" for match in result.matches)
        )
        self.assertTrue(all(match.distance_m <= 1000 for match in result.matches))

    def test_known_repair_line_intersects_production_segment_ids(self):
        result = self.engine.match_record(
            record(
                "line",
                source="gov.kz Astana Akimat",
                geometry={
                    "type": "LineString",
                    "coordinates": [[71.402, 51.108], [71.408, 51.108]],
                },
            ),
            "gov_kz_repairs",
        )
        self.assertEqual(
            {match.road_segment_id for match in result.matches}, {"1_2_0", "2_3_0"}
        )
        self.assertTrue(all(match.distance_m == 0 for match in result.matches))

    def test_known_repair_point_uses_nearest_road_search(self):
        result = self.engine.match_record(
            record(
                "point",
                source="gov.kz Astana Akimat",
                latitude=51.108,
                longitude=71.401,
            ),
            "gov_kz_repairs",
        )
        self.assertEqual([match.road_segment_id for match in result.matches], ["1_2_0"])
        self.assertEqual(result.matches[0].match_type, "point_nearest")

    def test_unknown_geometry_is_explicitly_unmatched(self):
        result = self.engine.match_record(record("unknown"), "ticketon_events")
        self.assertEqual(result.matches, [])
        self.assertEqual(result.unmatched[0]["reason"], "geometry_missing_or_invalid")

    def test_outside_astana_geometry_is_rejected(self):
        result = self.engine.match_record(
            record("outside", latitude=43.2389, longitude=76.8897), "ticketon_events"
        )
        self.assertEqual(result.matches, [])
        self.assertEqual(result.unmatched[0]["reason"], "geometry_outside_astana")

    def test_confidence_is_valid_and_road_ids_are_production_ids(self):
        result = self.engine.match_record(
            record("confidence", latitude=51.1083, longitude=71.4027), "ticketon_events"
        )
        road_ids = {road.road_segment_id for road in self.engine.roads()}
        self.assertTrue(
            all(0 <= match.match_confidence <= 1 for match in result.matches)
        )
        self.assertTrue(
            all(match.road_segment_id in road_ids for match in result.matches)
        )

    def test_storage_deduplicates_and_is_idempotent(self):
        matches = self.engine.match_record(
            record("storage", latitude=51.1083, longitude=71.4027), "ticketon_events"
        ).matches
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths, first = save_segment_matches(matches + matches, output_dir)
            _, second = save_segment_matches(matches, output_dir)
            frame = pd.read_parquet(paths["parquet"])
            exported = json.loads(paths["json"].read_text(encoding="utf-8"))
            changed = [replace(matches[0], distance_m=12.0), *matches[1:]]
            _, updated = save_segment_matches(changed, output_dir)
        self.assertEqual(first["new"], len(matches))
        self.assertEqual(second["unchanged"], len(matches))
        self.assertEqual(
            len(frame),
            frame[["provider", "source_item_id", "road_segment_id"]]
            .drop_duplicates()
            .shape[0],
        )
        self.assertEqual(len(exported), len(frame))
        self.assertEqual(updated["updated"], 1)
