"""Orchestrate one or many independent future providers."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from future_intelligence.registry import ProviderRegistry, default_registry
from future_intelligence.schemas import ProviderResult
from future_intelligence.utils import parse_prediction_datetime
from future_intelligence.validation import validate_result


class FutureIntelligencePipeline:
    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self.registry = registry or default_registry()
        self.last_results: list[ProviderResult] = []

    def collect(
        self,
        prediction_datetime: str | datetime,
        horizon_hours: int = 24,
        providers: Iterable[str] = ("openweather",),
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        strict: bool = False,
    ) -> dict:
        when = parse_prediction_datetime(prediction_datetime)
        results: list[ProviderResult] = []
        for name in providers:
            provider = self.registry.create(name)
            result = provider.collect(
                when, horizon_hours, latitude=latitude, longitude=longitude
            )
            issues = validate_result(result)
            if issues:
                result.warnings.extend(issues)
                result.status = "degraded"
                result.fallback_used = True
            if strict and result.status != "ok":
                raise RuntimeError(
                    f"Provider {name} is {result.status}: {result.warnings}"
                )
            results.append(result)
        self.last_results = results
        features: dict = {}
        warnings: list[str] = []
        for result in results:
            collisions = set(features).intersection(result.features)
            if collisions:
                result.status = "degraded"
                result.fallback_used = True
                result.warnings.append(f"feature_collision:{sorted(collisions)}")
            features.update(result.features)
            warnings.extend(result.warnings)
        return {
            "status": "ok"
            if results and all(result.status == "ok" for result in results)
            else "degraded",
            "prediction_datetime": when.isoformat(),
            "horizon_hours": horizon_hours,
            "providers": [result.to_context() for result in results],
            "features": features,
            "coverage": {
                result.metadata.provider_name: result.coverage for result in results
            },
            "warnings": warnings,
            "fallback_used": any(result.fallback_used for result in results),
        }
