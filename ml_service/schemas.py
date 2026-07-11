"""Small typed aliases documenting the JSON-ready public response shape."""

from __future__ import annotations
from typing import Any, TypedDict


class PredictionRecord(TypedDict):
    road_segment_id: str
    road_name: str
    risk_probability: float
    risk_level: str
    model_horizon: str
    top_positive_factors: list[dict[str, Any]]
    top_negative_factors: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    final_model_version: str


class CityPredictionResponse(TypedDict):
    datetime_hour: str
    model_horizon: str
    predictions: list[PredictionRecord]
    geojson: dict[str, Any]
    summary: dict[str, Any]
