"""Shared parsing and serialization helpers for the public ML service API."""

from __future__ import annotations
import math
import pandas as pd
from .exceptions import InvalidBBoxError, InvalidDatetimeError


def parse_datetime(
    value: str, minimum: pd.Timestamp | None = None, maximum: pd.Timestamp | None = None
) -> pd.Timestamp:
    """Parse an hourly local timestamp accepted by the underlying feature builder."""
    try:
        result = pd.Timestamp(value).floor("h")
    except (TypeError, ValueError) as exc:
        raise InvalidDatetimeError(f"Invalid datetime_hour: {value}") from exc
    if pd.isna(result):
        raise InvalidDatetimeError(f"Invalid datetime_hour: {value}")
    if minimum is not None and maximum is not None and not minimum <= result <= maximum:
        raise InvalidDatetimeError(
            f"datetime_hour is outside available source coverage: {result}"
        )
    return result


def validate_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> tuple[float, float, float, float]:
    """Validate geographic bounds in longitude/latitude order."""
    values = tuple(float(v) for v in (min_lon, min_lat, max_lon, max_lat))
    if not all(math.isfinite(v) for v in values) or not (
        -180 <= values[0] < values[2] <= 180 and -90 <= values[1] < values[3] <= 90
    ):
        raise InvalidBBoxError(
            "Bounding box must satisfy min_lon < max_lon and min_lat < max_lat."
        )
    return values
