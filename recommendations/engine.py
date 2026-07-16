"""Apply human-review rules to local model evidence after scoring.

The engine is separate from CatBoost. It requires positive relevant SHAP
evidence, describes model association rather than causality, and never trains.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .templates import rationale


RULES_PATH = Path(__file__).with_name("rules.yaml")


def load_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    # JSON is a strict subset of YAML; rules.yaml deliberately uses that subset.
    return json.loads(path.read_text(encoding="utf-8"))


def _truth(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes", "да"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def risk_level(probability: float, levels: dict[str, float]) -> str:
    if probability >= levels["high"]:
        return "high"
    if probability >= levels["medium"]:
        return "medium"
    if probability >= levels["low"]:
        return "low"
    return "minimal"


def _priority(probability: float) -> str:
    if probability >= 0.35:
        return "critical"
    if probability >= 0.20:
        return "high"
    return "medium"


def _positive(shap: dict[str, float], feature: str, minimum: float) -> float | None:
    value = _number(shap.get(feature), float("nan"))
    return value if value > minimum else None


def _add(
    items: list[dict[str, Any]],
    probability: float,
    category: str,
    rule: str,
    feature: str,
    shap_value: float,
    detail: str,
) -> None:
    items.append(
        {
            "priority": _priority(probability),
            "category": category,
            "rule": rule,
            "rationale": rationale(feature, shap_value, detail),
            "human_review_required": True,
            "evidence": {"feature": feature, "shap_value": shap_value},
        }
    )


_FEATURE_NAMES: dict[str, str] = {
    "road_lanes_num": "Число полос",
    "road_maxspeed_kmh": "Скоростной режим (км/ч)",
    "road_length": "Длина участка (м)",
    "segment_longitude": "Долгота",
    "segment_latitude": "Широта",
    "road_highway": "Тип дороги",
    "road_oneway": "Одностороннее движение",
    "road_lanes_missing": "Число полос не указано",
    "road_maxspeed_missing": "Скорость не указана",
    "road_name_missing": "Название дороги не указано",
}


def _feature_label(feature: str) -> str:
    if feature in _FEATURE_NAMES:
        return _FEATURE_NAMES[feature]
    if feature.startswith("weather_risk_"):
        risk = feature.replace("weather_risk_", "").replace("_now", "")
        return {"precip": "Осадки", "snow": "Снег", "freezing": "Гололёд", "high_wind": "Сильный ветер", "adverse": "Неблагоприятная погода"}.get(risk, feature)
    if feature.startswith("weather_") and "_prev_" in feature:
        parts = feature.replace("weather_", "").split("_prev_")
        metric = parts[0].replace("_", " ")
        hours = parts[1].replace("h", "")
        return f"Погода: {metric} за {hours}ч"
    if feature.startswith("weather_"):
        return "Погода: " + feature.replace("weather_", "").replace("_", " ")
    if feature.startswith("poi_"):
        parts = feature.replace("poi_", "").rsplit("_", 1)
        cat = parts[0].replace("_", " ")
        radius = parts[1].replace("m", "")
        labels = {"crossing": "Переходы", "education": "Образование", "emergency": "Скорая помощь", "healthcare": "Медицина", "other": "Прочее", "traffic_signal": "Светофоры", "transit_stop": "Остановки", "total": "Всего POI"}
        return f"{labels.get(cat, cat)} ({radius}м)"
    if feature.startswith("segment_accidents"):
        return "ДТП: " + feature.replace("segment_accidents_", "").replace("_", " ")
    if feature.startswith("city_accidents"):
        return "ДТП (город): " + feature.replace("city_accidents_", "").replace("_", " ")
    if feature.startswith("calendar_"):
        return "Календарь: " + feature.replace("calendar_", "").replace("_", " ")
    if feature.startswith("segment_"):
        return "Участок: " + feature.replace("segment_", "").replace("_", " ")
    return feature.replace("_", " ")


def _format_value(feature: str, value: object) -> str:
    if value is None:
        return "неизвестно"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        abs_v = abs(float(value))
        if abs_v < 1 and "_missing" not in feature:
            return f"{float(value):.2f}"
        if abs_v == int(abs_v):
            return str(int(value))
        return f"{float(value):.1f}"
    return str(value)


def factor_text(feature: str, value: object, shap_value: float) -> str:
    direction = "Повышает" if shap_value > 0 else "Снижает"
    impact = f"{abs(shap_value) * 100:.0f}%"
    label = _feature_label(feature)
    val_str = _format_value(feature, value)
    return f"{label}: {val_str}. {direction} риск на {impact}."


def recommend(
    *,
    probability: float,
    shap_values: dict[str, float],
    feature_values: dict[str, Any],
    model_horizon: str,
    final_model_version: str,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate recommendations only when a rule has a real positive local SHAP value."""
    rules = rules or load_rules()
    minimum = float(rules["positive_shap_min"])
    recommendations: list[dict[str, Any]] = []
    positive = sorted(
        (
            (name, _number(value))
            for name, value in shap_values.items()
            if _number(value) > 0
        ),
        key=lambda x: -x[1],
    )[:5]
    negative = sorted(
        (
            (name, _number(value))
            for name, value in shap_values.items()
            if _number(value) < 0
        ),
        key=lambda x: x[1],
    )[:5]

    historical = _positive(shap_values, "segment_accidents_total_prior", minimum)
    if historical is not None and _number(
        feature_values.get("segment_accidents_total_prior")
    ) >= float(rules["historical_accidents_min"]):
        _add(
            recommendations,
            probability,
            "inspection",
            "historical_accidents",
            "segment_accidents_total_prior",
            historical,
            "Проверить участок и актуальность схемы организации движения из-за модельной связи с предшествующими происшествиями.",
        )

    # Independent structural signal: unlike SHAP-derived rules, it intentionally
    # covers segments without crash history and does not require a high ML score.
    structural_names = (
        "speed_infra_mismatch",
        "pedestrian_exposure_gap",
        "narrow_fast_road",
        "junction_complexity_unregulated",
        "visibility_risk",
    )
    structural_hits = [
        name for name in structural_names if _truth(feature_values.get(name))
    ]
    if (
        _number(feature_values.get("segment_accidents_total_prior"))
        < float(rules["historical_accidents_min"])
        and structural_hits
    ):
        recommendations.append(
            {
                "priority": "medium",
                "category": "inspection",
                "rule": "structural_risk_flag",
                "rationale": "Сегмент не имеет истории ДТП, но выявлены инфраструктурные факторы риска: "
                + ", ".join(structural_hits)
                + ". Рекомендуется выездная проверка.",
                "human_review_required": True,
                "evidence": {"features": structural_hits, "source": "structural_rule"},
            }
        )

    weather_candidates = [
        name
        for name in shap_values
        if name.startswith("weather_")
        and any(
            token in name
            for token in ("precip", "snow", "freezing", "ice", "risk_adverse")
        )
    ]
    for name in weather_candidates:
        value = _positive(shap_values, name, minimum)
        if value is not None and (
            _truth(feature_values.get(name)) or _number(feature_values.get(name)) > 0
        ):
            _add(
                recommendations,
                probability,
                "operational",
                "weather_hazard",
                name,
                value,
                "Рассмотреть оперативный осмотр покрытия, информирование и готовность дорожных служб при погодном риске.",
            )
            break

    speed = _number(feature_values.get("road_maxspeed_kmh"), -1)
    speed_shap = _positive(shap_values, "road_maxspeed_kmh", minimum)
    if (
        speed_shap is not None
        and speed >= float(rules["high_speed_kmh"])
        and not _truth(feature_values.get("road_maxspeed_missing"))
    ):
        _add(
            recommendations,
            probability,
            "long_term",
            "confirmed_high_speed",
            "road_maxspeed_kmh",
            speed_shap,
            "Проверить соответствие подтверждённого скоростного режима условиям участка и целесообразность инженерной оценки скорости.",
        )

    missing_shap = _positive(shap_values, "road_maxspeed_missing", minimum)
    if missing_shap is not None and _truth(feature_values.get("road_maxspeed_missing")):
        _add(
            recommendations,
            probability,
            "inspection",
            "missing_speed_limit",
            "road_maxspeed_missing",
            missing_shap,
            "Уточнить и верифицировать сведения об ограничении скорости в реестре и на местности.",
        )

    poi_features = [name for name in shap_values if name.startswith("poi_")]
    poi_evidence = [
        (name, _positive(shap_values, name, minimum)) for name in poi_features
    ]
    poi_evidence = [
        (name, value)
        for name, value in poi_evidence
        if value is not None
        and _number(feature_values.get(name)) >= float(rules["poi_nearby_min"])
    ]
    if poi_evidence:
        name, value = max(poi_evidence, key=lambda x: x[1])
        _add(
            recommendations,
            probability,
            "inspection",
            "nearby_poi",
            name,
            value,
            "Рассмотреть полевой аудит переходов, остановок, школ и других точек притяжения рядом с участком.",
        )

    oneway = _positive(shap_values, "road_oneway", minimum)
    if oneway is not None and _truth(feature_values.get("road_oneway")):
        _add(
            recommendations,
            probability,
            "inspection",
            "oneway",
            "road_oneway",
            oneway,
            "Проверить читаемость знаков, разметку и конфликтные манёвры на одностороннем участке.",
        )

    road_features = (
        "road_length",
        "segment_latitude",
        "segment_longitude",
        "road_highway",
    )
    max_positive = positive[0][1] if positive else 0.0
    for name in road_features:
        value = _positive(shap_values, name, minimum)
        if value is not None and value >= max_positive * float(
            rules["road_spatial_importance_fraction"]
        ):
            _add(
                recommendations,
                probability,
                "long_term",
                "road_spatial",
                name,
                value,
                "Включить участок в приоритизацию инженерного обследования с учётом дорожной и пространственной важности для модели.",
            )
            break

    for name in ("calendar_is_rush_hour", "calendar_is_holiday"):
        value = _positive(shap_values, name, minimum)
        if value is not None and _truth(feature_values.get(name)):
            _add(
                recommendations,
                probability,
                "operational",
                "positive_rush_holiday",
                name,
                value,
                "Рассмотреть усиление наблюдения и информирования в соответствующий период, после проверки специалистом.",
            )
            break

    return {
        "risk_probability": float(probability),
        "risk_level": risk_level(float(probability), rules["risk_levels"]),
        "top_positive_factors": [
            {"feature": n, "shap_value": v,
             "value": feature_values.get(n),
             "text": factor_text(n, feature_values.get(n), v)}
            for n, v in positive
        ],
        "top_negative_factors": [
            {"feature": n, "shap_value": v,
             "value": feature_values.get(n),
             "text": factor_text(n, feature_values.get(n), v)}
            for n, v in negative
        ],
        "recommendations": recommendations,
        "model_horizon": model_horizon,
        "final_model_version": final_model_version,
        "human_decision_note": "Рекомендации поддерживают анализ; окончательное решение остаётся за человеком.",
    }
