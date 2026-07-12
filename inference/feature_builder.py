"""Assemble one leakage-safe feature row per known segment for a requested hour.

The builder sits between source-derived tables and frozen models. Historical
events are strictly prior and weather windows are shifted; it is not a trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _config(horizon: str) -> dict:
    path = (
        ROOT / "reports" / "stage7d" / "1h" / "stage7d_feature_config.json"
        if horizon == "1h"
        else sorted(
            (ROOT / "reports" / "stage7a" / "24h").glob(
                "*/training_dataset_*_feature_config.json"
            )
        )[-1]
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _weather_enriched() -> pd.DataFrame:
    w = (
        pd.read_parquet(ROOT / "data" / "external" / "weather_astana_hourly.parquet")
        .sort_values("datetime_hour")
        .copy()
    )
    w["weather_risk_precip_now"] = (w.precipitation.fillna(0) > 0).astype("int8")
    w["weather_risk_snow_now"] = (w.snowfall.fillna(0) > 0).astype("int8")
    w["weather_risk_freezing_now"] = (w.temperature_2m <= 0).astype("int8")
    w["weather_risk_high_wind_now"] = (
        (w.wind_speed_10m >= 10) | (w.wind_gusts_10m >= 15)
    ).astype("int8")
    w["weather_risk_adverse_now"] = (
        w[
            [
                "weather_risk_precip_now",
                "weather_risk_snow_now",
                "weather_risk_freezing_now",
                "weather_risk_high_wind_now",
            ]
        ].sum(axis=1)
        > 0
    ).astype("int8")
    for n in (3, 6, 24):
        w[f"weather_precip_sum_prev_{n}h"] = (
            w.precipitation.fillna(0).shift(1).rolling(n, min_periods=1).sum()
        )
        w[f"weather_snow_sum_prev_{n}h"] = (
            w.snowfall.fillna(0).shift(1).rolling(n, min_periods=1).sum()
        )
        w[f"weather_wind_mean_prev_{n}h"] = (
            w.wind_speed_10m.shift(1).rolling(n, min_periods=1).mean()
        )
        w[f"weather_adverse_hours_prev_{n}h"] = (
            w.weather_risk_adverse_now.shift(1).rolling(n, min_periods=1).sum()
        )
    w["weather_temperature_change_1h"] = w.temperature_2m - w.temperature_2m.shift(1)
    w["weather_temperature_change_3h"] = w.temperature_2m - w.temperature_2m.shift(3)
    last = w.datetime_hour.where(w.weather_risk_precip_now.eq(1)).ffill()
    w["weather_hours_since_precip"] = (
        (w.datetime_hour - last).dt.total_seconds().div(3600).fillna(9999.0)
    )
    return w


def _history(frame: pd.DataFrame, when: pd.Timestamp, horizon: str) -> pd.DataFrame:
    ready = pd.read_parquet(
        ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet",
        columns=["road_segment_id", "accident_datetime"],
    )
    events = ready.assign(
        event_hour=pd.to_datetime(ready.accident_datetime).dt.floor("h")
    )
    events = events.loc[
        events.event_hour < when
    ].copy()  # strict prior: no target/current/future events
    frame["segment_accidents_total_prior"] = (
        frame.road_segment_id.map(events.road_segment_id.value_counts())
        .fillna(0)
        .astype(int)
    )
    for name, hours in (("24h", 24), ("7d", 168), ("30d", 720)):
        recent = events.loc[
            events.event_hour >= when - pd.Timedelta(hours=hours)
        ].road_segment_id.value_counts()
        frame[f"segment_accidents_prev_{name}"] = (
            frame.road_segment_id.map(recent).fillna(0).astype(int)
        )
    last = events.groupby("road_segment_id").event_hour.max()
    frame["segment_hours_since_prev_accident"] = frame.road_segment_id.map(
        (when - last).dt.total_seconds().div(3600)
    ).fillna(-1.0)
    frame["segment_has_history"] = frame.segment_accidents_total_prior.gt(0)
    city = events.event_hour.value_counts().sort_index()
    frame["city_accidents_total_prior"] = int(len(events))
    for name, hours in (("24h", 24), ("7d", 168), ("30d", 720)):
        frame[f"city_accidents_prev_{name}"] = int(
            city.loc[city.index >= when - pd.Timedelta(hours=hours)].sum()
        )
    if horizon == "24h":
        ec = events.assign(
            month=events.event_hour.dt.month,
            hour=events.event_hour.dt.hour,
            weekday=events.event_hour.dt.weekday,
        )

        def count(mask: pd.Series) -> pd.Series:
            return (
                frame.road_segment_id.map(ec.loc[mask].road_segment_id.value_counts())
                .fillna(0)
                .astype(int)
            )

        frame["segment_accidents_same_month_prior"] = count(ec.month.eq(when.month))
        frame["segment_accidents_same_hour_weekday_prior"] = count(
            ec.hour.eq(when.hour) & ec.weekday.eq(when.weekday())
        )
        frame["segment_accidents_same_month_hour_weekday_prior"] = count(
            ec.month.eq(when.month)
            & ec.hour.eq(when.hour)
            & ec.weekday.eq(when.weekday())
        )
        cal = pd.read_parquet(
            ROOT / "data" / "external" / "calendar_features_hourly.parquet",
            columns=["datetime_hour", "is_holiday"],
        )
        ec = ec.merge(cal, left_on="event_hour", right_on="datetime_hour", how="left")
        frame["segment_accidents_holiday_prior"] = count(ec.is_holiday.fillna(False))
        ny = (ec.event_hour.dt.month.eq(12) & ec.event_hour.dt.day.ge(25)) | (
            ec.event_hour.dt.month.eq(1) & ec.event_hour.dt.day.le(7)
        )
        frame["segment_accidents_new_year_period_prior"] = count(ny)
    return frame


def build_features(
    datetime_hour: str | pd.Timestamp, horizon: str, *, weather_override: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Return ordered model features and segment metadata available at the requested hour."""
    when = pd.Timestamp(datetime_hour).floor("h")
    cfg = _config(horizon)
    features = list(cfg["numerical_features"]) + list(cfg["categorical_features"])
    ready = pd.read_parquet(
        ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet"
    )
    static_cols = [
        "road_segment_id",
        "road_highway",
        "road_lanes_num",
        "road_maxspeed_kmh",
        "road_length",
        "road_oneway",
        "road_lanes_missing",
        "road_maxspeed_missing",
        "road_name_missing",
        "road_name",
    ]
    static = (
        ready[static_cols]
        .sort_values("road_segment_id")
        .drop_duplicates("road_segment_id")
    )
    poi = pd.read_parquet(
        ROOT / "data" / "processed" / "road_segment_poi_features.parquet"
    )
    data = static.merge(poi, on="road_segment_id", how="inner", validate="one_to_one")
    data["datetime_hour"] = when
    cal = pd.read_parquet(
        ROOT / "data" / "external" / "calendar_features_hourly.parquet"
    )
    selected_calendar = cal.loc[cal.datetime_hour.eq(when)]
    if selected_calendar.empty:
        # Live inference may be beyond the archived calendar table. These are
        # the same deterministic calendar rules used by the generator; holiday
        # names are intentionally left empty when no year-specific table exists.
        month, day, hour, weekday = when.month, when.day, when.hour, when.weekday()
        c = pd.Series({
            "year": when.year, "month": month, "day": day, "hour": hour, "weekday": weekday,
            "is_weekend": weekday >= 5, "is_holiday": False, "holiday_name": "",
            "is_day_before_holiday": False, "is_day_after_holiday": False,
            "is_rush_hour": hour in (7, 8, 9, 17, 18, 19),
            "season": {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}[month],
            "is_school_year": month in (9, 10, 11, 12, 1, 2, 3, 4, 5),
            "is_school_summer_break": month in (6, 7, 8),
            "is_school_winter_break": (month == 12 and day >= 29) or (month == 1 and day <= 8),
            "is_school_spring_break": month == 3 and 21 <= day <= 31,
            "is_school_autumn_break": month == 10 and day >= 28,
        })
        c["is_school_break"] = any((c["is_school_summer_break"], c["is_school_winter_break"], c["is_school_spring_break"], c["is_school_autumn_break"]))
    else:
        c = selected_calendar.iloc[0]
    mapping = {
        "calendar_year": "year",
        "calendar_month": "month",
        "calendar_day": "day",
        "calendar_hour": "hour",
        "calendar_weekday": "weekday",
        "calendar_is_weekend": "is_weekend",
        "calendar_is_holiday": "is_holiday",
        "calendar_holiday_name": "holiday_name",
        "calendar_is_day_before_holiday": "is_day_before_holiday",
        "calendar_is_day_after_holiday": "is_day_after_holiday",
        "calendar_is_rush_hour": "is_rush_hour",
        "calendar_season": "season",
        "calendar_is_school_year": "is_school_year",
        "calendar_is_school_summer_break": "is_school_summer_break",
        "calendar_is_school_winter_break": "is_school_winter_break",
        "calendar_is_school_spring_break": "is_school_spring_break",
        "calendar_is_school_autumn_break": "is_school_autumn_break",
        "calendar_is_school_break": "is_school_break",
    }
    for out, source in mapping.items():
        data[out] = c[source]
    if "calendar_is_new_year_period" in features:
        data["calendar_is_new_year_period"] = bool(
            (when.month == 12 and when.day >= 25) or (when.month == 1 and when.day <= 7)
        )
    if weather_override is None:
        w = _weather_enriched().loc[lambda x: x.datetime_hour.eq(when)].iloc[0]
    else:
        # OpenWeather provides observed current conditions. Its current endpoint
        # has no complete 24-hour history, so history-derived features remain
        # missing rather than being fabricated from old archive data.
        w = pd.Series(weather_override).copy()
        w["weather_risk_precip_now"] = int(float(w.get("precipitation", 0)) > 0)
        w["weather_risk_snow_now"] = int(float(w.get("snowfall", 0)) > 0)
        w["weather_risk_freezing_now"] = int(float(w["temperature_2m"]) <= 0)
        w["weather_risk_high_wind_now"] = int(
            float(w.get("wind_speed_10m", 0)) >= 10 or float(w.get("wind_gusts_10m", 0)) >= 15
        )
        w["weather_risk_adverse_now"] = int(any(w[name] for name in (
            "weather_risk_precip_now", "weather_risk_snow_now", "weather_risk_freezing_now", "weather_risk_high_wind_now"
        )))
    weather_map = {
        "weather_temperature_2m": "temperature_2m",
        "weather_relative_humidity_2m": "relative_humidity_2m",
        "weather_dew_point_2m": "dew_point_2m",
        "weather_surface_pressure": "surface_pressure",
        "weather_precipitation": "precipitation",
        "weather_rain": "rain",
        "weather_snowfall": "snowfall",
        "weather_weather_code": "weather_code",
        "weather_cloud_cover": "cloud_cover",
        "weather_sunshine_duration": "sunshine_duration",
        "weather_wind_speed_10m": "wind_speed_10m",
        "weather_wind_gusts_10m": "wind_gusts_10m",
    }
    for out, source in weather_map.items():
        data[out] = w[source]
    for f in features:
        if f.startswith("weather_") and f not in weather_map:
            # CatBoost accepts missing values. This is preferable to inventing
            # prior-hour readings when only current OpenWeather data exists.
            data[f] = w[f] if f in w.index else np.nan
    data = _history(data, when, horizon)
    if horizon == "1h":
        data["weather_interaction_adverse_rush_hour"] = (
            data.weather_risk_adverse_now * data.calendar_is_rush_hour.astype(bool)
        )
        data["weather_interaction_adverse_maxspeed"] = (
            data.weather_risk_adverse_now * data.road_maxspeed_kmh.fillna(0)
        )
        data["weather_interaction_precip_history"] = (
            data.weather_precip_sum_prev_24h * data.segment_accidents_total_prior
        )
        data["weather_interaction_adverse_history"] = (
            data.weather_risk_adverse_now * data.segment_accidents_prev_30d
        )
        data["weather_interaction_adverse_road_highway"] = (
            data.weather_risk_adverse_now.astype(str)
            + "__"
            + data.road_highway.astype(str)
        )
    transform_cfg = (
        cfg
        if horizon == "24h"
        else json.loads(
            sorted(
                (ROOT / "reports" / "stage7a" / "1h").glob(
                    "*/training_dataset_*_feature_config.json"
                )
            )[-1].read_text(encoding="utf-8")
        )
    )
    rare = set(
        transform_cfg.get("transforms", {})
        .get("road_highway_grouping", {})
        .get("rare_categories_mapped_to_OTHER", [])
    )
    data["road_highway"] = data.road_highway.astype(str).where(
        ~data.road_highway.astype(str).isin(rare), "OTHER"
    )
    data.loc[data.road_lanes_missing.astype(bool), "road_lanes_num"] = np.nan
    data.loc[data.road_maxspeed_missing.astype(bool), "road_maxspeed_kmh"] = np.nan
    missing = [f for f in features if f not in data]
    if missing:
        raise ValueError(f"Feature builder missing: {missing}")
    entirely_missing = data[features].isna().all(axis=0)
    unsupported_missing = [
        feature for feature in features
        if entirely_missing[feature] and not (weather_override is not None and feature.startswith("weather_"))
    ]
    if unsupported_missing:
        raise ValueError(f"A required feature is entirely missing: {unsupported_missing}")
    return data, {
        "datetime_hour": str(when),
        "segments": len(data),
        "features": len(features),
        "horizon": horizon,
        "causality": "Events are strict-prior; rolling weather windows are shifted by one hour.",
    }
