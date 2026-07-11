"""Expose versioned operational display levels shared by inference and export.

These ranges label probabilities for the interface; they are not model training
or binary-classification thresholds.
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "config" / "risk_thresholds.json"


def load_risk_thresholds() -> dict[str, object]:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    levels = payload["levels"]
    if levels[0]["min_inclusive"] != 0.0 or levels[-1]["max_exclusive"] <= 1.0:
        raise ValueError("Risk threshold ranges must cover [0, 1].")
    for previous, current in zip(levels, levels[1:]):
        if previous["max_exclusive"] != current["min_inclusive"]:
            raise ValueError(
                "Risk threshold ranges must be adjacent and non-overlapping."
            )
    return payload


def configured_risk_level(
    probability: float, payload: dict[str, object] | None = None
) -> str:
    """Return the display level for a probability under the versioned config."""
    payload = payload or load_risk_thresholds()
    for item in payload["levels"]:
        if item["min_inclusive"] <= probability < item["max_exclusive"]:
            return str(item["level"])
    raise ValueError(f"Probability outside [0, 1]: {probability}")


def load() -> dict[str, object]:
    """Backward-compatible alias for :func:`load_risk_thresholds`."""
    return load_risk_thresholds()


def level(probability: float, payload: dict[str, object] | None = None) -> str:
    """Backward-compatible alias for :func:`configured_risk_level`."""
    return configured_risk_level(probability, payload)
