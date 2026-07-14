"""Stable backend entry point for city and segment accident-risk predictions.

Keep optional live-provider dependencies lazy: the standalone inference API only
needs ``hybrid_risk`` and must not import traffic/weather client stacks at boot.
"""

from __future__ import annotations

from typing import Any

__all__ = ["AccidentRiskPredictor", "TomTomTrafficService", "OpenWeatherService"]


def __getattr__(name: str) -> Any:
    """Load legacy provider classes only for callers that explicitly request them."""
    if name == "AccidentRiskPredictor":
        from .predictor import AccidentRiskPredictor

        return AccidentRiskPredictor
    if name == "TomTomTrafficService":
        from .traffic import TomTomTrafficService

        return TomTomTrafficService
    if name == "OpenWeatherService":
        from .weather import OpenWeatherService

        return OpenWeatherService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
