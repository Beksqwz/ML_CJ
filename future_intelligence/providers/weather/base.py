"""Extension point for weather forecast providers."""

from future_intelligence.providers.base import FutureIntelligenceProvider


class WeatherForecastProvider(FutureIntelligenceProvider):
    """Marker base class for weather providers."""
