"""Bounded TomTom live-traffic provider for Future Intelligence context only."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from future_intelligence.providers.base import FutureIntelligenceProvider
from future_intelligence.schemas import FutureRecord, ProviderMetadata, ProviderResult
from future_intelligence.utils import parse_prediction_datetime
from ml_service.traffic import SegmentPoint, TomTomTrafficService

ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_SEGMENTS_PATH = (
    ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet"
)


class TomTomFutureTrafficProvider(FutureIntelligenceProvider):
    """Collect a conservative live-flow sample without changing frozen inference."""

    metadata = ProviderMetadata(
        provider_name="tomtom",
        provider_version="1.0",
        source_type="traffic",
        supported_horizons=(24,),
        requires_api_key=True,
        update_frequency="live bounded snapshot",
        spatial_scope="Astana production road segments",
    )

    def __init__(
        self,
        service: TomTomTrafficService | None = None,
        *,
        max_segments: int | None = None,
    ) -> None:
        self.max_segments = max_segments or int(
            os.getenv("TOMTOM_FUTURE_MAX_SEGMENTS", "20")
        )
        self.service = service or TomTomTrafficService(
            segment_points=self._production_points()
        )

    @staticmethod
    def _production_points() -> dict[str, SegmentPoint]:
        roads = pd.read_parquet(
            PRODUCTION_SEGMENTS_PATH,
            columns=["road_segment_id", "latitude", "longitude"],
        ).dropna(subset=["road_segment_id", "latitude", "longitude"])
        grouped = roads.groupby("road_segment_id", sort=True)[
            ["latitude", "longitude"]
        ].median()
        return {
            str(segment_id): SegmentPoint(float(row.latitude), float(row.longitude))
            for segment_id, row in grouped.head(100).iterrows()
        }

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.service.configured else "degraded",
            "configured": self.service.configured,
            "provider": self.metadata.provider_name,
            "max_segments": self.max_segments,
        }

    def collect(
        self,
        prediction_datetime: datetime,
        horizon_hours: int,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> ProviderResult:
        del latitude, longitude, bbox
        when = parse_prediction_datetime(prediction_datetime)
        if horizon_hours != 24:
            return self._degraded(
                when, horizon_hours, "tomtom_supports_24h_context_only"
            )
        if not self.service.configured:
            return self._degraded(when, horizon_hours, "tomtom_api_key_not_configured")
        segment_ids = sorted(self.service._points or {})[: self.max_segments]
        readings = [self.service.get_segment(segment_id) for segment_id in segment_ids]
        records = self.normalize({"readings": readings}, when, horizon_hours)
        available = sum(record.payload.get("available", False) for record in records)
        return ProviderResult(
            metadata=self.metadata,
            raw_records=readings,
            normalized_records=records,
            features={
                "traffic_context_available": int(available > 0),
                "traffic_segments_sampled": len(records),
                "traffic_segments_available": available,
            },
            coverage={
                "segments_sampled": len(records),
                "segments_available": available,
                "prediction_datetime": when.isoformat(),
            },
            warnings=[] if available else ["tomtom_no_flow_data"],
            status="ok" if available else "degraded",
            fallback_used=available == 0,
        )

    def _degraded(
        self, when: datetime, horizon_hours: int, warning: str
    ) -> ProviderResult:
        return ProviderResult(
            self.metadata,
            [],
            [],
            {
                "traffic_context_available": 0,
                "traffic_segments_sampled": 0,
                "traffic_segments_available": 0,
            },
            {
                "segments_sampled": 0,
                "segments_available": 0,
                "prediction_datetime": when.isoformat(),
            },
            [warning],
            "degraded",
            True,
        )

    def normalize(
        self,
        raw_payload: dict[str, Any],
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> list[FutureRecord]:
        collected = datetime.now(UTC)
        records: list[FutureRecord] = []
        for reading in raw_payload.get("readings", []):
            point = reading.get("query_coordinates") or {}
            segment_id = str(reading.get("road_segment_id"))
            records.append(
                FutureRecord(
                    source="TomTom",
                    source_type="traffic",
                    source_version=self.metadata.provider_version,
                    source_item_id=f"{segment_id}:{prediction_datetime.isoformat()}",
                    source_url="https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/14/json",
                    collected_at=collected,
                    published_at=collected,
                    valid_from=collected,
                    valid_to=None,
                    prediction_datetime=prediction_datetime,
                    horizon_hours=horizon_hours,
                    latitude=point.get("latitude"),
                    longitude=point.get("longitude"),
                    affected_road_segment_ids=[segment_id],
                    event_type="traffic_flow",
                    confidence=reading.get("confidence"),
                    is_forecast=False,
                    is_realtime=True,
                    is_historical=False,
                    payload=reading,
                    warnings=[]
                    if reading.get("available")
                    else [str(reading.get("reason", "tomtom_no_flow_data"))],
                )
            )
        return records

    def build_features(
        self,
        normalized_records: list,
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> dict[str, Any]:
        del prediction_datetime, horizon_hours
        available = sum(
            record.payload.get("available", False) for record in normalized_records
        )
        return {
            "traffic_context_available": int(available > 0),
            "traffic_segments_sampled": len(normalized_records),
            "traffic_segments_available": available,
        }
