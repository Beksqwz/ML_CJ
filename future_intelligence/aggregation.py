"""Small deterministic aggregation helpers; null inputs remain null."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable


def numeric(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values if value is not None]


def summary(values: Iterable[object], prefix: str) -> dict[str, float | None]:
    items = numeric(values)
    if not items:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_std": None,
        }
    return {
        f"{prefix}_mean": mean(items),
        f"{prefix}_min": min(items),
        f"{prefix}_max": max(items),
        f"{prefix}_std": pstdev(items),
    }
