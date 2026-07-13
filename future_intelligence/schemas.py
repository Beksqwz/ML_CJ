"""Typed, JSON-ready contracts for future intelligence providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ProviderMetadata:
    provider_name: str
    provider_version: str
    source_type: str
    supported_horizons: tuple[int, ...]
    requires_api_key: bool
    update_frequency: str
    spatial_scope: str


@dataclass
class FutureRecord:
    source: str
    source_type: str
    source_version: str
    source_item_id: str | None
    source_url: str | None
    collected_at: datetime
    published_at: datetime | None
    valid_from: datetime | None
    valid_to: datetime | None
    prediction_datetime: datetime
    horizon_hours: int
    latitude: float | None
    longitude: float | None
    geometry: dict[str, Any] | None = None
    affected_road_segment_ids: list[str] = field(default_factory=list)
    event_type: str | None = None
    severity: str | None = None
    confidence: float | None = None
    is_forecast: bool = True
    is_realtime: bool = False
    is_historical: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "collected_at",
            "published_at",
            "valid_from",
            "valid_to",
            "prediction_datetime",
        ):
            if value[key] is not None:
                value[key] = value[key].isoformat()
        return value


@dataclass
class ProviderResult:
    metadata: ProviderMetadata
    raw_records: list[dict[str, Any]]
    normalized_records: list[FutureRecord]
    features: dict[str, Any]
    coverage: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"
    fallback_used: bool = False

    def to_context(self) -> dict[str, Any]:
        return {
            "provider": asdict(self.metadata),
            "status": self.status,
            "features": self.features,
            "coverage": self.coverage,
            "warnings": self.warnings,
            "fallback_used": self.fallback_used,
            "records": [record.to_dict() for record in self.normalized_records],
        }
