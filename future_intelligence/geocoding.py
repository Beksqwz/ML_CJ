"""Auditable Astana geocoding for future-context records.

No road segment is inferred here.  A point result is only a geocoded source
location; linear repair geometry and road matching are separate later stages.
"""
from __future__ import annotations

import re
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
import networkx as nx
from shapely import wkt
from shapely.geometry import mapping
from shapely.ops import linemerge, unary_union
from future_intelligence.schemas import FutureRecord

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# (latitude, longitude, documented capacity).  Keep this compact and reviewable.
KNOWN_ASTANA_VENUES = {
    "astana arena": (51.1083, 71.4027, 30000),
    "барыс арена": (51.1156, 71.4446, 12000),
    "barys arena": (51.1156, 71.4446, 12000),
    "expo": (51.0894, 71.4184, 5000),
    "конгресс-центр": (51.0891, 71.4180, 3000),
    "congress centre": (51.0891, 71.4180, 3000),
}


@dataclass(frozen=True)
class GeocodeResult:
    latitude: float | None
    longitude: float | None
    quality: str
    source: str
    query: str | None
    confidence: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairGeometryResult:
    geometry: dict[str, Any] | None
    quality: str
    confidence: float
    warnings: tuple[str, ...] = ()


def _text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _road_key(value: str | None) -> str:
    value = _text(value).lower()
    value = re.sub(r"^(?:улиц[ауыые]?|ул\.?|проспект[а-еом]?|пр\.?|пр-т|шоссе|ш\.?)\s*", "", value)
    value = re.sub(r"\b(?:көшесі|кошеси|даңғылы|дангылы|даңғыл|улица|street|avenue)\b", " ", value)
    value = re.sub(r"\b(?:от|до|from|to)\b.*", " ", value)
    value = value.translate(str.maketrans({"ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "і": "и"}))
    return re.sub(r"[^\wа-я]+", "", value)


class RoadGeometryResolver:
    """Build repair line geometry from the local OSM road graph.

    It never replaces an explicit from/to section with a street centroid.  If
    the boundary streets cannot be resolved, the result remains an explicitly
    lower-confidence whole-road geometry rather than a fabricated subsection.
    """

    def __init__(self, edges_path: Path | None = None) -> None:
        self.edges_path = edges_path or Path(__file__).resolve().parents[1] / "data" / "roads" / "astana_edges.csv"
        self._edges: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._edges is not None: return self._edges
        edges = []
        with self.edges_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("geometry") or not _road_key(row.get("name")): continue
                try: geometry = wkt.loads(row["geometry"])
                except Exception: continue
                edges.append({"u": str(row["u"]), "v": str(row["v"]), "name": _road_key(row.get("name")), "length": float(row.get("length") or geometry.length), "geometry": geometry})
        self._edges = edges
        return edges

    def repair(self, location: dict[str, Any]) -> RepairGeometryResult:
        road, start, end = (_road_key(location.get(key)) for key in ("road_name", "from_street", "to_street"))
        if not road:
            return RepairGeometryResult(None, "not_geocoded", 0, ("repair_road_name_missing",))
        edges = self._load()
        def same(left: str, right: str) -> bool:
            left, right = left.rstrip("аыи"), right.rstrip("аыи")
            return left in right or right in left
        main = [edge for edge in edges if same(road, edge["name"])]
        if not main:
            return RepairGeometryResult(None, "not_geocoded", 0, ("repair_road_name_not_found",))
        if not start or not end:
            return RepairGeometryResult(mapping(linemerge(unary_union([edge["geometry"] for edge in main]))), "road_name_geometry", .45, ("repair_from_to_missing",))
        nodes_by_street = {}
        for boundary in (start, end):
            boundary_nodes = {node for edge in edges if same(boundary, edge["name"]) for node in (edge["u"], edge["v"])}
            nodes_by_street[boundary] = {node for edge in main for node in (edge["u"], edge["v"]) if node in boundary_nodes}
        starts, ends = nodes_by_street[start], nodes_by_street[end]
        if not starts or not ends:
            return RepairGeometryResult(mapping(linemerge(unary_union([edge["geometry"] for edge in main]))), "road_name_geometry", .45, ("repair_boundary_street_not_found",))
        graph = nx.Graph()
        best: dict[frozenset[str], dict[str, Any]] = {}
        for edge in main:
            key = frozenset((edge["u"], edge["v"]))
            if key not in best or edge["length"] < best[key]["length"]: best[key] = edge
        for edge in best.values(): graph.add_edge(edge["u"], edge["v"], weight=edge["length"])
        candidates = []
        for source in starts:
            for target in ends:
                try: candidates.append(nx.shortest_path(graph, source, target, weight="weight"))
                except (nx.NetworkXNoPath, nx.NodeNotFound): pass
        if not candidates:
            return RepairGeometryResult(mapping(linemerge(unary_union([edge["geometry"] for edge in main]))), "road_name_geometry", .45, ("repair_boundary_path_not_found",))
        path = min(candidates, key=lambda nodes: sum(graph[a][b]["weight"] for a, b in zip(nodes, nodes[1:])))
        lines = [best[frozenset((a, b))]["geometry"] for a, b in zip(path, path[1:])]
        return RepairGeometryResult(mapping(linemerge(unary_union(lines))), "road_from_to_network", .90)


class AstanaGeocoder:
    """Local venue lookup first, then one polite Nominatim fallback per query."""

    def __init__(self, *, session: requests.Session | None = None, timeout_seconds: float = 10, min_interval_seconds: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        self.session, self.timeout_seconds = session or requests.Session(), timeout_seconds
        self.min_interval_seconds, self.sleep, self._last_request = min_interval_seconds, sleep, 0.0
        self._cache: dict[str, GeocodeResult] = {}

    def local_venue(self, venue: str | None) -> GeocodeResult | None:
        value = _text(venue).lower()
        match = next((data for name, data in KNOWN_ASTANA_VENUES.items() if name in value), None)
        if not match:
            return None
        return GeocodeResult(match[0], match[1], "local_venue_directory", "local", venue, .98)

    def geocode(self, query: str | None) -> GeocodeResult:
        query = _text(query)
        if not query:
            return GeocodeResult(None, None, "not_geocoded", "none", None, 0, ("empty_geocode_query",))
        if query in self._cache:
            return self._cache[query]
        wait = self.min_interval_seconds - (time.monotonic() - self._last_request)
        if wait > 0: self.sleep(wait)
        try:
            response = self.session.get(NOMINATIM_URL, params={"q": query + ", Astana, Kazakhstan", "format": "jsonv2", "limit": 1, "countrycodes": "kz"}, headers={"User-Agent": "AstanaFutureIntelligence/1.0 (public geocoding)"}, timeout=self.timeout_seconds)
            self._last_request = time.monotonic(); response.raise_for_status(); items = response.json()
            item = items[0] if isinstance(items, list) and items else None
            result = GeocodeResult(float(item["lat"]), float(item["lon"]), "nominatim", "nominatim", query, .70) if item else GeocodeResult(None, None, "not_geocoded", "nominatim", query, 0, ("nominatim_no_result",))
        except (requests.RequestException, ValueError, KeyError, TypeError):
            result = GeocodeResult(None, None, "not_geocoded", "nominatim", query, 0, ("nominatim_request_failed",))
        self._cache[query] = result
        return result

    def event(self, venue: str | None, address: str | None) -> GeocodeResult:
        return self.local_venue(venue) or self.geocode(", ".join(filter(None, (_text(venue), _text(address)))) or None)

    def repair(self, location: dict[str, Any]) -> GeocodeResult:
        # This intentionally returns a point only. from/to geometry comes later.
        query = location.get("intersection_streets") or location.get("road_name") or location.get("address")
        result = self.geocode(query)
        if result.latitude is not None:
            return GeocodeResult(result.latitude, result.longitude, result.quality, result.source, result.query, result.confidence, result.warnings + ("repair_line_geometry_pending",))
        return result


def apply_geocode(record: FutureRecord, result: GeocodeResult) -> None:
    """Attach transparent geocoding provenance to a mutable universal record."""
    record.payload.update({"geocoding_quality": result.quality, "geocoding_source": result.source, "geocoding_query": result.query})
    record.warnings.extend(result.warnings)
    if result.latitude is None or result.longitude is None:
        return
    record.latitude, record.longitude = result.latitude, result.longitude
    record.geometry = {"type": "Point", "coordinates": [result.longitude, result.latitude]}
    record.confidence = max(record.confidence or 0, result.confidence)
