"""Validation that is deliberately separate from frozen-model feature validation."""

from __future__ import annotations

from future_intelligence.schemas import FutureRecord, ProviderResult


REQUIRED_RECORD_FIELDS = (
    "source",
    "source_type",
    "source_version",
    "collected_at",
    "prediction_datetime",
    "horizon_hours",
    "is_forecast",
    "is_realtime",
    "is_historical",
    "payload",
    "warnings",
)


def validate_record(record: FutureRecord) -> list[str]:
    data = record.to_dict()
    return [
        field
        for field in REQUIRED_RECORD_FIELDS
        if field not in data or data[field] is None
    ]


def validate_result(result: ProviderResult) -> list[str]:
    issues = []
    if result.status not in {"ok", "degraded"}:
        issues.append("invalid_status")
    if result.status == "degraded" and not result.fallback_used:
        issues.append("degraded_without_explicit_fallback")
    allowed_prefixes = ("weather_", "repair_", "traffic_", "transit_", "event_")
    if any(not key.startswith(allowed_prefixes) for key in result.features):
        issues.append("unnamespaced_feature")
    for record in result.normalized_records:
        issues.extend(f"record:{field}" for field in validate_record(record))
    return issues
