"""Optional TomTom flow integration, deliberately separate from ML features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from shapely import wkt

from .registry import ROOT


LOGGER = logging.getLogger(__name__)
FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/14/json"
SNAPSHOT_COLUMNS = [
    "timestamp", "road_segment_id", "coordinates", "current_speed",
    "free_flow_speed", "congestion_ratio", "confidence",
]


def _local_api_key() -> str | None:
    try:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "TOMTOM_API_KEY":
                return value.strip().strip('"').strip("'") or None
    except FileNotFoundError:
        pass
    return None


@dataclass(frozen=True)
class SegmentPoint:
    latitude: float
    longitude: float


def _segment_points(
    edges_path: Path = ROOT / "data" / "roads" / "astana_edges.csv",
) -> dict[str, SegmentPoint]:
    """Map existing source-of-truth road segments to their WGS84 midpoints."""
    edges = pd.read_csv(edges_path, usecols=["u", "v", "key", "geometry"])
    points: dict[str, SegmentPoint] = {}
    for row in edges.itertuples(index=False):
        midpoint = wkt.loads(str(row.geometry)).interpolate(0.5, normalized=True)
        points[f"{row.u}_{row.v}_{row.key}"] = SegmentPoint(
            latitude=float(midpoint.y), longitude=float(midpoint.x)
        )
    return points


class TomTomTrafficService:
    """Fetch traffic readings without changing frozen model features or probabilities."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 10.0,
        session: requests.Session | None = None,
        segment_points: dict[str, SegmentPoint] | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else (
            os.getenv("TOMTOM_API_KEY") or _local_api_key()
        )
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self._points = segment_points

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _point_for(self, road_segment_id: str) -> SegmentPoint | None:
        if self._points is None:
            self._points = _segment_points()
        return self._points.get(str(road_segment_id))

    @staticmethod
    def _unavailable(
        segment_id: str, reason: str, point: SegmentPoint | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "road_segment_id": str(segment_id), "available": False, "reason": reason,
            "current_speed": None, "free_flow_speed": None,
            "congestion_ratio": None, "confidence": None,
        }
        if point:
            result["query_coordinates"] = {
                "latitude": point.latitude, "longitude": point.longitude,
            }
        return result

    def get_segment(self, road_segment_id: str) -> dict[str, Any]:
        point = self._point_for(road_segment_id)
        if point is None:
            return self._unavailable(road_segment_id, "unknown_road_segment")
        if not self.configured:
            return self._unavailable(road_segment_id, "tomtom_api_key_not_configured", point)
        try:
            response = self.session.get(
                FLOW_URL,
                params={
                    "key": self.api_key,
                    "point": f"{point.latitude},{point.longitude}",
                    "unit": "kmph",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            flow = response.json().get("flowSegmentData")
            if not isinstance(flow, dict):
                return self._unavailable(road_segment_id, "no_flow_data", point)
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("TomTom live traffic unavailable for %s: %s", road_segment_id, exc)
            return self._unavailable(road_segment_id, "tomtom_request_failed", point)

        current, free_flow = flow.get("currentSpeed"), flow.get("freeFlowSpeed")
        return {
            "road_segment_id": str(road_segment_id), "available": True,
            "query_coordinates": {"latitude": point.latitude, "longitude": point.longitude},
            "current_speed": None if current is None else float(current),
            "free_flow_speed": None if free_flow is None else float(free_flow),
            "congestion_ratio": None if not free_flow else float(current) / float(free_flow),
            "confidence": None if flow.get("confidence") is None else float(flow["confidence"]),
            "road_closure": flow.get("roadClosure"), "tomtom_frc": flow.get("frc"),
            "flow_coordinates": flow.get("coordinates", {}).get("coordinate", []),
        }

    def collect(self, road_segment_ids: Iterable[str], output_path: Path) -> list[dict[str, Any]]:
        captured_at = datetime.now(timezone.utc).isoformat()
        readings = [self.get_segment(segment_id) for segment_id in road_segment_ids]
        rows = [
            {
                "timestamp": captured_at, "road_segment_id": item["road_segment_id"],
                "coordinates": None if not item.get("query_coordinates") else (
                    f"{item['query_coordinates']['latitude']},{item['query_coordinates']['longitude']}"
                ),
                "current_speed": item["current_speed"],
                "free_flow_speed": item["free_flow_speed"],
                "congestion_ratio": item["congestion_ratio"], "confidence": item["confidence"],
            }
            for item in readings
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
        if output_path.exists():
            snapshot = pd.concat([pd.read_parquet(output_path), snapshot], ignore_index=True)
        snapshot.to_parquet(output_path, index=False)
        return readings
