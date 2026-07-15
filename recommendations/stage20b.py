"""Deterministic Stage 20B operational recommendation engine.

This module consumes the Stage 20A contract only.  It neither trains a model
nor describes a risk score as an accident probability.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "monitor_only": 4}
REQUIRED_COLUMNS = {
    "road_segment_id",
    "prediction_datetime",
    "dynamic_rank",
    "dynamic_percentile",
    "historical_hotspot_rank",
    "historical_hotspot_percentile",
    "future_context_flags",
    "future_context_warnings",
    "provider_degraded",
}
STRONG_SIGNALS = {"severe_weather", "heavy_traffic", "road_repair", "major_event"}

PLAN_TEMPLATES = {
    "patrol": {
        "ru": "Рассмотреть временное патрулирование участка.",
        "kz": "Учаскеде уақытша патрульдеуді қарастыру.",
        "en": "Consider temporary patrol coverage for this segment.",
    },
    "speed": {
        "ru": "Рассмотреть контроль скоростного режима после проверки обстановки.",
        "kz": "Жағдайды тексергеннен кейін жылдамдық режимін бақылауды қарастыру.",
        "en": "Consider speed monitoring after reviewing local conditions.",
    },
    "repair": {
        "ru": "Проверить безопасность ремонтной зоны и временную организацию движения.",
        "kz": "Жөндеу аймағының қауіпсіздігін және қозғалысты уақытша ұйымдастыруды тексеру.",
        "en": "Review repair-zone safety and temporary traffic arrangements.",
    },
    "event": {
        "ru": "Рассмотреть контроль транспортного потока около мероприятия.",
        "kz": "Іс-шара маңындағы көлік ағынын бақылауды қарастыру.",
        "en": "Consider traffic-flow monitoring near the event.",
    },
    "weather": {
        "ru": "Проверить покрытие, освещение и информирование участников движения.",
        "kz": "Жол жабынын, жарықтандыруды және жол қозғалысына қатысушыларды хабардар етуді тексеру.",
        "en": "Review road surface, lighting, and traveller information.",
    },
    "engineering": {
        "ru": "Передать участок на инженерную проверку знаков, разметки и освещения.",
        "kz": "Учаскені белгілерді, жолақтарды және жарықтандыруды инженерлік тексеруге беру.",
        "en": "Refer the segment for an engineering review of signs, markings, and lighting.",
    },
    "monitor": {
        "ru": "Продолжить мониторинг: контекстные данные провайдеров неполны.",
        "kz": "Мониторингті жалғастыру: провайдерлердің контекстік деректері толық емес.",
        "en": "Continue monitoring: provider context data are incomplete.",
    },
}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(str(value))
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _reasons(row: pd.Series) -> tuple[list[str], set[str]]:
    reasons: list[str] = []
    if float(row.dynamic_percentile) >= 0.99:
        reasons.append("DYNAMIC_TOP_1PCT")
    elif float(row.dynamic_percentile) >= 0.95:
        reasons.append("DYNAMIC_TOP_5PCT")
    if int(row.historical_hotspot_rank) <= 20:
        reasons.append("HOTSPOT_TOP_20")
    elif int(row.historical_hotspot_rank) <= 50:
        reasons.append("HOTSPOT_TOP_50")
    flags = set(_as_list(row.future_context_flags))
    mapping = {
        "severe_weather": "SEVERE_WEATHER",
        "heavy_traffic": "HEAVY_TRAFFIC",
        "road_repair": "ROAD_REPAIR",
        "major_event": "MAJOR_EVENT",
    }
    reasons.extend(code for flag, code in mapping.items() if flag in flags)
    if (
        float(row.dynamic_percentile) >= 0.95
        and float(row.historical_hotspot_percentile) < 0.50
    ):
        reasons.append("MODEL_DISAGREEMENT")
    independent = (
        int(float(row.dynamic_percentile) >= 0.95)
        + int(int(row.historical_hotspot_rank) <= 50)
        + len(flags & STRONG_SIGNALS)
    )
    if independent >= 3:
        reasons.append("MULTI_SIGNAL_AGREEMENT")
    if bool(row.provider_degraded):
        reasons.append("PROVIDER_DEGRADED")
    return reasons, flags


def _priority(row: pd.Series, reasons: list[str], flags: set[str]) -> str:
    dynamic_top_1 = "DYNAMIC_TOP_1PCT" in reasons
    dynamic_top_5 = dynamic_top_1 or "DYNAMIC_TOP_5PCT" in reasons
    hotspot_top_20 = "HOTSPOT_TOP_20" in reasons
    hotspot_top_50 = hotspot_top_20 or "HOTSPOT_TOP_50" in reasons
    strong = bool(flags & STRONG_SIGNALS)
    if dynamic_top_1 and (
        (hotspot_top_20 and strong) or "MULTI_SIGNAL_AGREEMENT" in reasons
    ):
        return "critical"
    if dynamic_top_5 or (hotspot_top_50 and strong) or strong:
        return "high"
    if hotspot_top_50 or "MODEL_DISAGREEMENT" in reasons:
        return "medium"
    if bool(row.provider_degraded):
        return "monitor_only"
    return "low"


def _plans(reasons: list[str]) -> list[dict[str, str]]:
    keys: list[str] = []
    if "DYNAMIC_TOP_1PCT" in reasons or "DYNAMIC_TOP_5PCT" in reasons:
        keys.extend(["patrol", "speed"])
    if "HOTSPOT_TOP_20" in reasons or "HOTSPOT_TOP_50" in reasons:
        keys.append("engineering")
    if "SEVERE_WEATHER" in reasons:
        keys.append("weather")
    if "ROAD_REPAIR" in reasons:
        keys.append("repair")
    if "MAJOR_EVENT" in reasons or "HEAVY_TRAFFIC" in reasons:
        keys.append("event")
    if "PROVIDER_DEGRADED" in reasons:
        keys.append("monitor")
    return [PLAN_TEMPLATES[key] for key in dict.fromkeys(keys)]


def recommend_stage20b(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic operational priorities to a valid Stage 20A table."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"stage20b_missing_columns:{sorted(missing)}")
    if frame.duplicated(["road_segment_id", "prediction_datetime"]).any():
        raise ValueError("stage20b_duplicate_segment_prediction")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        reasons, flags = _reasons(row)
        priority = _priority(row, reasons, flags)
        warnings = _as_list(row.future_context_warnings)
        if "PROVIDER_DEGRADED" in reasons and "provider_degraded" not in warnings:
            warnings.append("provider_degraded")
        uncertainty = (
            "high"
            if bool(row.provider_degraded)
            else ("medium" if "MODEL_DISAGREEMENT" in reasons else "low")
        )
        rows.append(
            {
                "operational_priority": priority,
                "reasons": reasons,
                "possible_plan": _plans(reasons),
                "uncertainty": uncertainty,
                "warnings": warnings,
            }
        )
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result = result.sort_values(
        [
            "operational_priority",
            "dynamic_rank",
            "historical_hotspot_rank",
            "road_segment_id",
        ],
        key=lambda values: (
            values.map(PRIORITY_ORDER)
            if values.name == "operational_priority"
            else values
        ),
        kind="stable",
    ).reset_index(drop=True)
    result["priority_rank"] = range(1, len(result) + 1)
    return result
