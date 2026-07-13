"""Timezone and JSON helpers shared by future providers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


ASTANA_TIMEZONE = ZoneInfo("Asia/Almaty")


def parse_prediction_datetime(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ASTANA_TIMEZONE)
    return parsed


def api_timezone(offset_seconds: int | None) -> timezone:
    return timezone(timedelta(seconds=int(offset_seconds or 0)))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
