"""Canonical, safe weather forecast snapshot helpers for 3-hour OpenWeather data."""

from __future__ import annotations
import hashlib
from datetime import timedelta
from typing import Any
import pandas as pd
from future_intelligence.utils import ASTANA_TIMEZONE

NUMERIC = (
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_gust",
    "rain",
    "snow",
    "visibility",
    "clouds",
    "precipitation_probability",
)


def canonical_snapshot(
    records: list[dict[str, Any]], collected_at: str, prediction_datetime: str
) -> dict[str, Any]:
    points = sorted(records, key=lambda x: x["forecast_timestamp"])
    digest = hashlib.sha256((str(points) + str(collected_at)).encode()).hexdigest()[:16]
    return {
        "snapshot_version": f"openweather-{digest}",
        "provider": "openweather",
        "collected_at": collected_at,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "timezone": str(ASTANA_TIMEZONE),
        "valid_from": points[0]["forecast_timestamp"] if points else None,
        "valid_until": (
            pd.Timestamp(points[-1]["forecast_timestamp"]) + timedelta(hours=3)
        ).isoformat()
        if points
        else None,
        "source_step_hours": 3,
        "forecast_points": points,
    }


def select_origin_weather(
    snapshot: dict[str, Any], prediction_datetime: str, max_gap_hours: float = 3
) -> dict[str, Any] | None:
    target = (
        pd.Timestamp(prediction_datetime).tz_convert(ASTANA_TIMEZONE)
        if pd.Timestamp(prediction_datetime).tzinfo
        else pd.Timestamp(prediction_datetime).tz_localize(ASTANA_TIMEZONE)
    )
    points = snapshot.get("forecast_points", [])
    if not points:
        return None
    dated = [(pd.Timestamp(p["forecast_timestamp"]), p) for p in points]
    before = max((x for x in dated if x[0] <= target), default=None, key=lambda x: x[0])
    after = min((x for x in dated if x[0] >= target), default=None, key=lambda x: x[0])
    if (
        not before
        or not after
        or (target - before[0]).total_seconds() / 3600 > max_gap_hours
        or (after[0] - target).total_seconds() / 3600 > max_gap_hours
    ):
        return None
    fraction = (
        0
        if after[0] == before[0]
        else (target - before[0]).total_seconds()
        / (after[0] - before[0]).total_seconds()
    )
    value = {
        "prediction_datetime": target.isoformat(),
        "source_before": before[0].isoformat(),
        "source_after": after[0].isoformat(),
        "interpolated": bool(fraction),
    }
    for key in NUMERIC:
        a, b = before[1].get(key), after[1].get(key)
        value[key] = (
            None
            if a is None or b is None
            else float(a) + (float(b) - float(a)) * fraction
        )
    value["weather_condition"] = (before if fraction <= 0.5 else after)[1].get(
        "weather_main"
    )
    value["snapshot_version"] = snapshot["snapshot_version"]
    return value


def summarize_24h(snapshot: dict[str, Any], prediction_datetime: str) -> dict[str, Any]:
    start = pd.Timestamp(prediction_datetime)
    end = start + timedelta(hours=24)
    points = [
        p
        for p in snapshot.get("forecast_points", [])
        if start <= pd.Timestamp(p["forecast_timestamp"]) <= end
    ]

    def severity(point: dict[str, Any]) -> float:
        signals = (
            int((point.get("rain") or 0) > 0)
            + int((point.get("snow") or 0) > 0) * 2
            + int((point.get("visibility") or 1e9) < 1000)
            + int((point.get("wind_speed") or 0) >= 10)
        )
        return min(1.0, signals / 5)

    worst = max(points, key=severity, default=None)
    return {
        "forecast_start": start.isoformat(),
        "forecast_end": end.isoformat(),
        "forecast_points_available": len(points),
        "expected_points": 9,
        "forecast_complete": len(points) >= 9,
        "source_step_hours": 3,
        "max_weather_severity_score": severity(worst) if worst else 0.0,
        "severe_weather_expected": bool(worst and severity(worst) >= 0.6),
        "worst_period_start": worst.get("forecast_timestamp") if worst else None,
        "worst_period_end": (
            pd.Timestamp(worst["forecast_timestamp"]) + timedelta(hours=3)
        ).isoformat()
        if worst
        else None,
        "precipitation_expected": any((p.get("rain") or 0) > 0 for p in points),
        "snow_expected": any((p.get("snow") or 0) > 0 for p in points),
        "heavy_rain_expected": any((p.get("rain") or 0) >= 2.5 for p in points),
        "minimum_visibility_m": min(
            (p.get("visibility") for p in points if p.get("visibility") is not None),
            default=None,
        ),
        "maximum_wind_speed": max(
            (p.get("wind_speed") for p in points if p.get("wind_speed") is not None),
            default=None,
        ),
        "temperature_min": min(
            (p.get("temperature") for p in points if p.get("temperature") is not None),
            default=None,
        ),
        "temperature_max": max(
            (p.get("temperature") for p in points if p.get("temperature") is not None),
            default=None,
        ),
    }
