"""Compact, CatBoost-component-only explanations for the frozen 24-hour model."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

CATBOOST_SHAP_BATCH_SIZE = 512
MAX_POSITIVE_FACTORS = 3
MAX_NEGATIVE_FACTORS = 3
MIN_ABSOLUTE_SHAP_VALUE = 1e-8

_DISCLAIMERS = {
    "ru": "Объяснение относится только к CatBoost-компоненту итогового ансамбля.",
    "kz": "Түсіндірме қорытынды ансамбльдің тек CatBoost компонентіне қатысты.",
    "en": "This explanation applies only to the CatBoost component of the final ensemble.",
}
_DISPLAY_NAMES = {
    "road_lanes_num": {"ru": "Число полос", "kz": "Жолақ саны", "en": "Lane count"},
    "road_maxspeed_kmh": {
        "ru": "Скоростной режим",
        "kz": "Жылдамдық режимі",
        "en": "Speed limit",
    },
    "road_length": {
        "ru": "Длина участка",
        "kz": "Учаске ұзындығы",
        "en": "Segment length",
    },
    "calendar_is_rush_hour": {
        "ru": "Час пик",
        "kz": "Қарбалас уақыт",
        "en": "Rush hour",
    },
    "weather_risk_adverse_now": {
        "ru": "Неблагоприятная погода",
        "kz": "Қолайсыз ауа райы",
        "en": "Adverse weather",
    },
}


def _json_safe(value: Any) -> Any:
    """Convert selected factor values only; never serialize a whole feature row."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return None
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return str(value)


def _display_name(feature: str) -> dict[str, str]:
    if feature in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[feature]
    label = feature.replace("_", " ")
    return {"ru": label, "kz": label, "en": label}


def _factor(feature: str, value: Any, shap_value: float) -> dict[str, Any]:
    return {
        "feature": feature,
        "display_name": _display_name(feature),
        "feature_value": _json_safe(value),
        "shap_value": float(shap_value),
    }


def _text(
    positive: list[dict[str, Any]], negative: list[dict[str, Any]]
) -> dict[str, str]:
    positive_names = ", ".join(item["display_name"]["ru"] for item in positive)
    negative_names = ", ".join(item["display_name"]["ru"] for item in negative)
    ru = "Объяснение CatBoost-компонента; итоговое ранжирование выполняет ансамбль."
    if positive_names:
        ru += f" Повышающий вклад: {positive_names}."
    if negative_names:
        ru += f" Снижающий вклад: {negative_names}."
    return {
        "ru": ru,
        "kz": "Түсіндірме CatBoost компонентіне қатысты; қорытынды ранжирлеуді ансамбль орындайды.",
        "en": "This explains the CatBoost component; final ranking is performed by the ensemble.",
    }


def unavailable_explanations(
    count: int, component_weight: float
) -> list[dict[str, Any]]:
    return [
        {
            "method": "shap",
            "scope": "catboost_component_only",
            "component_weight": component_weight,
            "base_value": None,
            "top_positive_factors": [],
            "top_negative_factors": [],
            "disclaimer": _DISCLAIMERS,
            "text": _text([], []),
            "explanation_status": "unavailable",
        }
        for _ in range(count)
    ]


def catboost_explanations(
    model: Any,
    feature_frame: pd.DataFrame,
    *,
    ordered_features: list[str],
    categorical_features: list[str],
    component_weight: float,
    batch_size: int = CATBOOST_SHAP_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], int]:
    """Return compact per-row explanations and the number of batched model calls."""
    explanations: list[dict[str, Any]] = []
    calls = 0
    for start in range(0, len(feature_frame), batch_size):
        batch = feature_frame.iloc[start : start + batch_size][ordered_features].copy()
        for column in categorical_features:
            batch[column] = (
                batch[column].astype("string").fillna("__MISSING__").astype(str)
            )
        shap_values = model.get_feature_importance(
            Pool(batch, cat_features=categorical_features), type="ShapValues"
        )
        calls += 1
        for position, vector in enumerate(np.asarray(shap_values)):
            contributions, base_value = vector[:-1], vector[-1]
            pairs = [
                (ordered_features[index], float(value))
                for index, value in enumerate(contributions)
                if abs(float(value)) >= MIN_ABSOLUTE_SHAP_VALUE
            ]
            positive = sorted(
                (item for item in pairs if item[1] > 0),
                key=lambda item: item[1],
                reverse=True,
            )[:MAX_POSITIVE_FACTORS]
            negative = sorted(
                (item for item in pairs if item[1] < 0), key=lambda item: item[1]
            )[:MAX_NEGATIVE_FACTORS]
            values = batch.iloc[position]
            positive_factors = [
                _factor(name, values[name], value) for name, value in positive
            ]
            negative_factors = [
                _factor(name, values[name], value) for name, value in negative
            ]
            explanations.append(
                {
                    "method": "shap",
                    "scope": "catboost_component_only",
                    "component_weight": component_weight,
                    "base_value": _json_safe(base_value),
                    "top_positive_factors": positive_factors,
                    "top_negative_factors": negative_factors,
                    "disclaimer": _DISCLAIMERS,
                    "text": _text(positive_factors, negative_factors),
                    "explanation_status": "available",
                }
            )
    return explanations, calls
