"""Registry keeps provider addition independent from pipeline orchestration."""

from __future__ import annotations

from typing import Callable

from future_intelligence.providers.base import FutureIntelligenceProvider
from future_intelligence.providers.weather.openweather import (
    OpenWeatherForecastProvider,
)
from future_intelligence.providers.repairs.gov_kz import GovKzRoadEventsProvider
from future_intelligence.providers.events.ticketon import TicketonEventsProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], FutureIntelligenceProvider]] = {}

    def register(
        self, name: str, factory: Callable[[], FutureIntelligenceProvider]
    ) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> FutureIntelligenceProvider:
        try:
            return self._factories[name]()
        except KeyError as exc:
            raise ValueError(f"Unknown future intelligence provider: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("openweather", OpenWeatherForecastProvider)
    registry.register("gov_kz_repairs", GovKzRoadEventsProvider)
    registry.register("ticketon_events", TicketonEventsProvider)
    return registry
