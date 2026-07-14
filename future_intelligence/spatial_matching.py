"""Reusable, context-only matching of Future Intelligence records to road segments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform
from shapely.strtree import STRtree
from shapely import wkt

from future_intelligence.geocoding import ASTANA_BBOX
from future_intelligence.schemas import FutureRecord
from future_intelligence.utils import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
ROAD_EDGES_PATH = ROOT / "data" / "roads" / "astana_edges.csv"
PRODUCTION_SEGMENTS_PATH = (
    ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet"
)
WGS84_TO_METERS = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
MATCH_COLUMNS = [
    "provider",
    "source_item_id",
    "road_segment_id",
    "distance_m",
    "match_type",
    "match_confidence",
    "geometry_quality",
    "valid_from",
    "valid_to",
    "created_at",
    "updated_at",
    "warnings",
    "content_hash",
]


@dataclass(frozen=True)
class SegmentMatch:
    provider: str
    source_item_id: str | None
    road_segment_id: str
    distance_m: float
    match_type: str
    match_confidence: float
    geometry_quality: str | None
    valid_from: datetime | None
    valid_to: datetime | None
    created_at: datetime
    updated_at: datetime
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("valid_from", "valid_to", "created_at", "updated_at"):
            if row[key] is not None:
                row[key] = row[key].isoformat()
        row["warnings"] = list(self.warnings)
        return row


@dataclass(frozen=True)
class RoadSegment:
    road_segment_id: str
    geometry: Any


@dataclass
class SpatialMatchResult:
    matches: list[SegmentMatch] = field(default_factory=list)
    unmatched: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SpatialMatchingEngine:
    """Lazy cached matcher against the single production road-segment network."""

    def __init__(
        self,
        edges_path: Path | None = None,
        production_segments_path: Path | None = None,
        *,
        ticketon_radius_m: float = 1000.0,
        point_radius_m: float = 100.0,
    ) -> None:
        self.edges_path = edges_path or ROAD_EDGES_PATH
        self.production_segments_path = production_segments_path
        self.ticketon_radius_m = ticketon_radius_m
        self.point_radius_m = point_radius_m
        self._roads: list[RoadSegment] | None = None
        self._meter_road_geometries: list[Any] | None = None
        self._road_tree: STRtree | None = None
        self._production_segment_ids: set[str] | None = None

    def production_segment_ids(self) -> set[str] | None:
        if self.production_segments_path is None and self.edges_path != ROAD_EDGES_PATH:
            return None
        path = self.production_segments_path or PRODUCTION_SEGMENTS_PATH
        if self._production_segment_ids is None:
            self._production_segment_ids = set(
                pd.read_parquet(path, columns=["road_segment_id"])["road_segment_id"]
                .astype(str)
                .tolist()
            )
        return self._production_segment_ids

    @staticmethod
    def _inside_astana(geometry: Any) -> bool:
        min_lat, min_lon, max_lat, max_lon = ASTANA_BBOX
        centroid = geometry.centroid
        return min_lon <= centroid.x <= max_lon and min_lat <= centroid.y <= max_lat

    @staticmethod
    def _meter_geometry(geometry: Any) -> Any:
        return transform(WGS84_TO_METERS.transform, geometry)

    def roads(self) -> list[RoadSegment]:
        if self._roads is not None:
            return self._roads
        loaded: list[RoadSegment] = []
        with self.edges_path.open(encoding="utf-8", newline="") as stream:
            for row in pd.read_csv(
                stream, dtype={"u": str, "v": str, "key": str}
            ).to_dict("records"):
                try:
                    geometry = wkt.loads(row["geometry"])
                except (KeyError, TypeError, ValueError):
                    continue
                if geometry.is_empty or not self._inside_astana(geometry):
                    continue
                road_segment_id = f"{row['u']}_{row['v']}_{row['key']}"
                production_ids = self.production_segment_ids()
                if production_ids is not None and road_segment_id not in production_ids:
                    continue
                loaded.append(
                    RoadSegment(
                        road_segment_id,
                        geometry,
                    )
                )
        self._roads = loaded
        return loaded

    def _candidate_roads(
        self, meter_geometry: Any, radius_m: float = 0.0
    ) -> list[RoadSegment]:
        roads = self.roads()
        if self._meter_road_geometries is None:
            self._meter_road_geometries = [
                self._meter_geometry(road.geometry) for road in roads
            ]
            self._road_tree = STRtree(self._meter_road_geometries)
        query_geometry = (
            meter_geometry.buffer(radius_m) if radius_m > 0 else meter_geometry
        )
        indices = self._road_tree.query(query_geometry) if self._road_tree else []
        return [roads[int(index)] for index in indices]

    @staticmethod
    def _geometry_from_record(record: FutureRecord) -> Any | None:
        if record.geometry:
            try:
                geometry = shape(record.geometry)
            except (TypeError, ValueError):
                return None
            return None if geometry.is_empty else geometry
        if record.latitude is not None and record.longitude is not None:
            return Point(record.longitude, record.latitude)
        return None

    @staticmethod
    def _confidence(distance_m: float, radius_m: float, quality: str | None) -> float:
        base = (
            0.90 if quality in {"road_from_to_network", "exact_known_venue"} else 0.75
        )
        if radius_m <= 0:
            return base
        return round(max(0.05, min(1.0, base * (1.0 - distance_m / radius_m))), 4)

    def _make_match(
        self,
        record: FutureRecord,
        provider: str,
        segment: RoadSegment,
        distance_m: float,
        match_type: str,
        quality: str | None,
        radius_m: float,
    ) -> SegmentMatch:
        now = datetime.now(UTC)
        return SegmentMatch(
            provider=provider,
            source_item_id=record.source_item_id,
            road_segment_id=segment.road_segment_id,
            distance_m=round(distance_m, 3),
            match_type=match_type,
            match_confidence=self._confidence(distance_m, radius_m, quality),
            geometry_quality=quality,
            valid_from=record.valid_from,
            valid_to=record.valid_to,
            created_at=now,
            updated_at=now,
        )

    def match_record(
        self, record: FutureRecord, provider: str | None = None
    ) -> SpatialMatchResult:
        provider = provider or record.source
        geometry = self._geometry_from_record(record)
        result = SpatialMatchResult()
        if geometry is None:
            reason = "geometry_missing_or_invalid"
            result.unmatched.append(
                {
                    "provider": provider,
                    "source_item_id": record.source_item_id,
                    "reason": reason,
                }
            )
            result.warnings.append(reason)
            return result
        if not self._inside_astana(geometry):
            reason = "geometry_outside_astana"
            result.unmatched.append(
                {
                    "provider": provider,
                    "source_item_id": record.source_item_id,
                    "reason": reason,
                }
            )
            result.warnings.append(reason)
            return result

        quality = record.payload.get("repair_geometry_quality") or record.payload.get(
            "geocoding_quality"
        )
        meter_geometry = self._meter_geometry(geometry)
        if geometry.geom_type in {"LineString", "MultiLineString"}:
            for road in self._candidate_roads(meter_geometry):
                if meter_geometry.intersects(self._meter_geometry(road.geometry)):
                    result.matches.append(
                        self._make_match(
                            record,
                            provider,
                            road,
                            0.0,
                            "line_intersection",
                            quality,
                            1.0,
                        )
                    )
        elif geometry.geom_type in {"Polygon", "MultiPolygon"}:
            for road in self._candidate_roads(meter_geometry):
                if meter_geometry.intersects(self._meter_geometry(road.geometry)):
                    result.matches.append(
                        self._make_match(
                            record,
                            provider,
                            road,
                            0.0,
                            "polygon_intersection",
                            quality,
                            1.0,
                        )
                    )
        elif geometry.geom_type == "Point":
            radius = (
                self.ticketon_radius_m
                if provider == "ticketon_events" or record.source == "Ticketon"
                else self.point_radius_m
            )
            match_type = (
                "ticketon_radius"
                if radius == self.ticketon_radius_m
                else "point_nearest"
            )
            for road in self._candidate_roads(meter_geometry, radius):
                distance_m = meter_geometry.distance(
                    self._meter_geometry(road.geometry)
                )
                if distance_m <= radius:
                    result.matches.append(
                        self._make_match(
                            record,
                            provider,
                            road,
                            distance_m,
                            match_type,
                            quality,
                            radius,
                        )
                    )
        else:
            reason = f"unsupported_geometry_type:{geometry.geom_type}"
            result.unmatched.append(
                {
                    "provider": provider,
                    "source_item_id": record.source_item_id,
                    "reason": reason,
                }
            )
            result.warnings.append(reason)
            return result

        deduplicated = {match.road_segment_id: match for match in result.matches}
        result.matches = sorted(
            deduplicated.values(), key=lambda item: item.road_segment_id
        )
        record.affected_road_segment_ids = [
            match.road_segment_id for match in result.matches
        ]
        if not result.matches:
            reason = "no_road_segments_in_matching_scope"
            result.unmatched.append(
                {
                    "provider": provider,
                    "source_item_id": record.source_item_id,
                    "reason": reason,
                }
            )
            result.warnings.append(reason)
        return result

    def match_records(
        self, records: Iterable[FutureRecord], provider: str | None = None
    ) -> SpatialMatchResult:
        combined = SpatialMatchResult()
        for record in records:
            item = self.match_record(record, provider)
            combined.matches.extend(item.matches)
            combined.unmatched.extend(item.unmatched)
            combined.warnings.extend(item.warnings)
        return combined


def save_segment_matches(
    matches: Iterable[SegmentMatch], output_dir: Path
) -> tuple[dict[str, Path], dict[str, int]]:
    """Idempotently persist canonical segment matches and a JSON export."""
    processed = output_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    parquet_path = processed / "future_segment_matches.parquet"
    json_path = processed / "future_segment_matches.json"
    incoming = {
        (match.provider, match.source_item_id, match.road_segment_id): match.to_dict()
        for match in matches
        if match.distance_m >= 0 and 0 <= match.match_confidence <= 1
    }
    incoming_events = {
        (provider, source_item_id) for provider, source_item_id, _ in incoming
    }
    previous = (
        pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()
    )
    if not previous.empty:
        previous["_identity"] = list(
            zip(
                previous["provider"],
                previous["source_item_id"],
                previous["road_segment_id"],
            )
        )
    previous_by_identity = (
        previous.set_index("_identity").to_dict("index") if not previous.empty else {}
    )
    new = updated = unchanged = 0
    rows: list[dict[str, Any]] = []
    for identity, row in incoming.items():
        material = {
            key: value
            for key, value in row.items()
            if key not in {"created_at", "updated_at"}
        }
        row["content_hash"] = hashlib.sha256(
            json.dumps(
                to_jsonable(material), ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        existing = previous_by_identity.pop(identity, None)
        if existing is None:
            new += 1
        elif existing.get("content_hash") == row["content_hash"]:
            unchanged += 1
            row["created_at"] = existing.get("created_at", row["created_at"])
            row["updated_at"] = existing.get("updated_at", row["updated_at"])
        else:
            updated += 1
            row["created_at"] = existing.get("created_at", row["created_at"])
        rows.append(row)
    rows.extend(
        value
        for identity, value in previous_by_identity.items()
        if (identity[0], identity[1]) not in incoming_events and "provider" in value
    )
    frame = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    if not frame.empty:
        frame = frame.drop(columns=["_identity"], errors="ignore").drop_duplicates(
            ["provider", "source_item_id", "road_segment_id"], keep="last"
        )
    frame.to_parquet(parquet_path, index=False)
    json_path.write_text(
        frame.to_json(orient="records", force_ascii=False, date_format="iso", indent=2),
        encoding="utf-8",
    )
    return {"parquet": parquet_path, "json": json_path}, {
        "new": new,
        "updated": updated,
        "unchanged": unchanged,
    }


def matching_metrics(result: SpatialMatchResult) -> dict[str, Any]:
    matches = result.matches
    events = {(match.provider, match.source_item_id) for match in matches}
    unmatched_events = {
        (row.get("provider"), row.get("source_item_id")) for row in result.unmatched
    }
    distances = [match.distance_m for match in matches]
    confidences = [match.match_confidence for match in matches]
    providers = sorted(
        {match.provider for match in matches}
        | {row.get("provider") for row in result.unmatched}
    )
    return {
        "providers": providers,
        "records": len(events | unmatched_events),
        "matched": len(events),
        "unmatched": len(unmatched_events),
        "coverage": len(events) / len(events | unmatched_events)
        if (events | unmatched_events)
        else 0.0,
        "average_matches_per_event": len(matches) / len(events) if events else 0.0,
        "distance_m": {
            "min": min(distances) if distances else None,
            "mean": sum(distances) / len(distances) if distances else None,
            "max": max(distances) if distances else None,
        },
        "match_confidence": {
            "min": min(confidences) if confidences else None,
            "mean": sum(confidences) / len(confidences) if confidences else None,
            "max": max(confidences) if confidences else None,
        },
        "warnings": result.warnings,
    }


def save_matching_report(result: SpatialMatchResult, report_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(matching_metrics(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path
