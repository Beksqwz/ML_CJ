"""Single source of truth for Stage 8C operational display risk levels."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "config" / "risk_thresholds.json"

def load() -> dict:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    levels = payload["levels"]
    if levels[0]["min_inclusive"] != 0.0 or levels[-1]["max_exclusive"] <= 1.0:
        raise ValueError("Risk threshold ranges must cover [0, 1].")
    for previous, current in zip(levels, levels[1:]):
        if previous["max_exclusive"] != current["min_inclusive"]:
            raise ValueError("Risk threshold ranges must be adjacent and non-overlapping.")
    return payload

def level(probability: float, payload: dict | None = None) -> str:
    payload = payload or load()
    for item in payload["levels"]:
        if item["min_inclusive"] <= probability < item["max_exclusive"]:
            return str(item["level"])
    raise ValueError(f"Probability outside [0, 1]: {probability}")
