"""Stable provider contract; raw schemas remain provider-specific."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from future_intelligence.schemas import ProviderMetadata, ProviderResult


class FutureIntelligenceProvider(ABC):
    metadata: ProviderMetadata

    @abstractmethod
    def collect(
        self,
        prediction_datetime: datetime,
        horizon_hours: int,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> ProviderResult:
        """Collect, normalize and aggregate one provider response."""

    @abstractmethod
    def normalize(
        self,
        raw_payload: dict[str, Any],
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> list:
        """Convert provider-specific payload to universal records."""

    @abstractmethod
    def build_features(
        self,
        normalized_records: list,
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> dict[str, Any]:
        """Build namespaced future-training candidates only."""

    @abstractmethod
    def healthcheck(self) -> dict[str, Any]:
        """Return a safe, JSON-ready readiness status."""
