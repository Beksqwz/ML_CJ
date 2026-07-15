"""Deterministic, context-only 24-hour Future Intelligence segment features."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from future_intelligence.spatial_matching import PRODUCTION_SEGMENTS_PATH
from future_intelligence.utils import ASTANA_TIMEZONE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "future_intelligence"
WEATHER_COLUMNS = (
    "weather_temperature_mean",
    "weather_temperature_min",
    "weather_temperature_max",
    "weather_temperature_range",
    "weather_humidity_mean",
    "weather_precipitation_probability_max",
    "weather_rain_sum",
    "weather_snow_sum",
    "weather_visibility_min",
    "weather_wind_speed_max",
    "weather_wind_gust_max",
    "weather_freeze_thaw_transition",
    "weather_road_surface_risk_score",
    "weather_visibility_risk_score",
    "weather_severity_score",
    # Context-only canonical snapshot fields; never added to frozen ML input.
    "weather_snapshot",
    "weather_origin",
    "weather_summary_24h",
)
WEATHER_CONTEXT_COLUMNS = (
    "weather_snapshot_version",
    "weather_provider",
    "weather_collected_at",
    "weather_valid_from",
    "weather_valid_until",
    "weather_source_step_hours",
    "weather_stale",
    "weather_origin_prediction_datetime",
    "weather_origin_source_before",
    "weather_origin_source_after",
    "weather_origin_interpolated",
    "weather_origin_temperature",
    "weather_origin_humidity",
    "weather_origin_pressure",
    "weather_origin_wind_speed",
    "weather_origin_rain",
    "weather_origin_visibility",
    "weather_origin_weather_condition",
    "weather_forecast_start",
    "weather_forecast_end",
    "weather_forecast_points_available",
    "weather_expected_points",
    "weather_forecast_complete",
    "weather_max_weather_severity_score",
    "weather_severe_weather_expected",
    "weather_worst_period_start",
    "weather_worst_period_end",
    "weather_precipitation_expected",
    "weather_snow_expected",
    "weather_heavy_rain_expected",
    "weather_minimum_visibility_m",
    "weather_maximum_wind_speed",
    "weather_summary_temperature_min",
    "weather_summary_temperature_max",
)
REPAIR_COLUMNS = (
    "repair_active_next_24h",
    "repair_event_count_next_24h",
    "repair_full_closure_next_24h",
    "repair_partial_closure_next_24h",
    "repair_lane_closure_next_24h",
    "repair_intersection_closure_next_24h",
    "repair_bridge_event_next_24h",
    "repair_high_severity_count_next_24h",
    "repair_disruption_score_next_24h",
    "repair_distance_to_nearest_m",
    "repair_match_confidence_max",
    "repair_open_end_count_next_24h",
    "repair_hours_until_nearest_start",
)
EVENT_COLUMNS = (
    "event_count_next_24h",
    "event_count_500m_next_24h",
    "event_count_1000m_next_24h",
    "event_count_2000m_next_24h",
    "event_major_next_24h",
    "event_stadium_next_24h",
    "event_concert_next_24h",
    "event_sports_next_24h",
    "event_intensity_sum_next_24h",
    "event_intensity_max_next_24h",
    "event_distance_to_nearest_m",
    "event_hours_until_nearest_start",
    "event_post_event_outflow_risk",
    "event_match_confidence_max",
)
BASE_COLUMNS = (
    "road_segment_id",
    "prediction_datetime",
    "forecast_window_start",
    "forecast_window_end",
    "horizon_hours",
    "generated_at",
)
FEATURE_COLUMNS = (
    *BASE_COLUMNS,
    *WEATHER_COLUMNS,
    *WEATHER_CONTEXT_COLUMNS,
    "weather_available",
    "weather_fallback_used",
    *REPAIR_COLUMNS,
    "repair_provider_available",
    *EVENT_COLUMNS,
    "ticketon_provider_available",
)

_DUPLICATE_FEATURE_COLUMNS = sorted(
    {name for name in FEATURE_COLUMNS if FEATURE_COLUMNS.count(name) > 1}
)
if _DUPLICATE_FEATURE_COLUMNS:
    raise ValueError(f"duplicate_feature_columns: {_DUPLICATE_FEATURE_COLUMNS}")


@dataclass(frozen=True)
class BuilderPaths:
    road_network_path: Path = PRODUCTION_SEGMENTS_PATH
    matches_path: Path = (
        DEFAULT_OUTPUT_DIR / "processed" / "future_segment_matches.parquet"
    )
    weather_features_path: Path = (
        DEFAULT_OUTPUT_DIR / "processed" / "openweather_24h_features.parquet"
    )
    gov_events_path: Path = (
        DEFAULT_OUTPUT_DIR / "processed" / "gov_kz_road_events.parquet"
    )
    ticketon_events_path: Path = (
        DEFAULT_OUTPUT_DIR / "processed" / "ticketon_events.parquet"
    )
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        return parsed.tz_localize(ASTANA_TIMEZONE).to_pydatetime()
    return parsed.tz_convert(ASTANA_TIMEZONE).to_pydatetime()


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return value


def _overlaps(
    start: datetime | None,
    end: datetime | None,
    window_start: datetime,
    window_end: datetime,
    *,
    open_end: bool = False,
) -> bool:
    if start is None:
        return False
    if end is None:
        return start < window_end if open_end else window_start <= start < window_end
    return start < window_end and end > window_start


def _severity_weight(value: Any) -> float:
    return {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}.get(
        str(value).lower(), 0.0
    )


class FutureSegmentFeatureBuilder:
    """Builds separate 24-hour context features; never mutates frozen model input."""

    def __init__(
        self,
        paths: BuilderPaths | None = None,
        *,
        expected_segment_count: int | None = 3968,
    ) -> None:
        self.paths = paths or BuilderPaths()
        self.expected_segment_count = expected_segment_count
        self.warnings: list[str] = []
        self.input_audit: dict[str, Any] = {}

    @staticmethod
    def _hash(path: Path) -> str | None:
        if not path.exists():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _audit_file(self, name: str, path: Path) -> dict[str, Any]:
        report: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": self._hash(path),
        }
        if path.exists() and path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            report.update(
                {
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "providers": frame["provider"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                    if "provider" in frame
                    else [],
                    "prediction_datetimes": frame["prediction_datetime"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                    if "prediction_datetime" in frame
                    else [],
                }
            )
        self.input_audit[name] = report
        return report

    def audit_inputs(self) -> dict[str, Any]:
        for name, path in {
            "road_network": self.paths.road_network_path,
            "matches": self.paths.matches_path,
            "weather": self.paths.weather_features_path,
            "gov_events": self.paths.gov_events_path,
            "ticketon_events": self.paths.ticketon_events_path,
        }.items():
            self._audit_file(name, path)
        return self.input_audit

    def _production_ids(self) -> list[str]:
        frame = pd.read_parquet(
            self.paths.road_network_path, columns=["road_segment_id"]
        )
        ids = sorted(frame["road_segment_id"].astype(str).drop_duplicates().tolist())
        if (
            self.expected_segment_count is not None
            and len(ids) != self.expected_segment_count
        ):
            raise ValueError(
                f"production_segment_count_expected:{self.expected_segment_count}:got:{len(ids)}"
            )
        return ids

    def _weather(
        self, prediction_datetime: datetime
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        values = {
            column: None for column in (*WEATHER_COLUMNS, *WEATHER_CONTEXT_COLUMNS)
        }
        metadata = {"available": 0, "fallback_used": 1, "coverage": {}, "warnings": []}
        path = self.paths.weather_features_path
        if not path.exists():
            metadata["warnings"].append("weather_features_file_missing")
            return values, metadata
        frame = pd.read_parquet(path)
        if frame.empty or "prediction_datetime" not in frame:
            metadata["warnings"].append("weather_features_missing_prediction_datetime")
            return values, metadata
        candidate = frame.loc[
            frame["prediction_datetime"].astype(str) == prediction_datetime.isoformat()
        ]
        if candidate.empty:
            metadata["warnings"].append("weather_features_requested_window_missing")
            return values, metadata
        row = candidate.iloc[-1]
        for column in (*WEATHER_COLUMNS, *WEATHER_CONTEXT_COLUMNS):
            if column in row and pd.notna(row[column]):
                values[column] = (
                    row[column].item() if hasattr(row[column], "item") else row[column]
                )
        metadata.update(
            {"available": 1, "fallback_used": 0, "coverage": {"rows": len(candidate)}}
        )
        return values, metadata

    def _load_events(self, path: Path, provider: str) -> dict[str, dict[str, Any]]:
        if not path.exists():
            self.warnings.append(f"{provider}_events_file_missing")
            return {}
        events = {}
        for row in pd.read_parquet(path).to_dict("records"):
            source_id = row.get("source_item_id")
            if source_id is None or (
                isinstance(source_id, float) and math.isnan(source_id)
            ):
                continue
            payload = _json(row.get("payload_json"), {})
            events[str(source_id)] = row | {"payload": payload}
        return events

    def _valid_matches(self, production_ids: set[str]) -> pd.DataFrame:
        path = self.paths.matches_path
        if not path.exists():
            self.warnings.append("stage17_matches_file_missing")
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        required = {
            "provider",
            "source_item_id",
            "road_segment_id",
            "distance_m",
            "match_confidence",
        }
        if not required.issubset(frame.columns):
            self.warnings.append("stage17_matches_schema_invalid")
            return pd.DataFrame()
        valid = frame.copy()
        valid["road_segment_id"] = valid["road_segment_id"].astype(str)
        invalid = (
            ~valid["road_segment_id"].isin(production_ids)
            | valid["source_item_id"].isna()
            | valid["distance_m"].notna() & (valid["distance_m"] < 0)
            | valid["match_confidence"].isna()
            | (valid["match_confidence"] < 0)
            | (valid["match_confidence"] > 1)
            | ~valid["provider"].isin(["gov_kz_repairs", "ticketon_events"])
        )
        if invalid.any():
            self.warnings.append(
                f"stage17_invalid_matches_rejected:{int(invalid.sum())}"
            )
        valid = valid.loc[~invalid].drop_duplicates(
            ["provider", "source_item_id", "road_segment_id"], keep="last"
        )
        return valid

    @staticmethod
    def _zero_repair() -> dict[str, Any]:
        values = {column: 0 for column in REPAIR_COLUMNS}
        values["repair_distance_to_nearest_m"] = None
        values["repair_hours_until_nearest_start"] = None
        return values

    @staticmethod
    def _zero_event() -> dict[str, Any]:
        values = {column: 0 for column in EVENT_COLUMNS}
        values["event_distance_to_nearest_m"] = None
        values["event_hours_until_nearest_start"] = None
        return values

    def _repair_features(
        self, rows: pd.DataFrame, start: datetime, end: datetime
    ) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for segment_id, group in rows.groupby("road_segment_id"):
            events = group.to_dict("records")
            unique_ids = {str(event["source_item_id"]) for event in events}
            restrictions = [
                str(event["restriction_type"] or "unknown") for event in events
            ]
            event_types = [str(event["event_type"] or "") for event in events]
            distances = [
                float(event["distance_m"])
                for event in events
                if pd.notna(event["distance_m"])
            ]
            confidences = [float(event["match_confidence"]) for event in events]
            starts = [
                event["valid_from"]
                for event in events
                if event["valid_from"] is not None
            ]
            score = sum(
                _severity_weight(event["severity"])
                * float(event["match_confidence"])
                * (
                    2.0
                    if event["restriction_type"] == "full_closure"
                    else 1.5
                    if event["restriction_type"]
                    in {"partial_closure", "intersection_closure"}
                    else 1.25
                    if event["restriction_type"]
                    in {"single_lane_closure", "lane_narrowing"}
                    else 1.0
                )
                for event in events
            )
            output[str(segment_id)] = {
                "repair_active_next_24h": int(bool(unique_ids)),
                "repair_event_count_next_24h": len(unique_ids),
                "repair_full_closure_next_24h": sum(
                    value == "full_closure" for value in restrictions
                ),
                "repair_partial_closure_next_24h": sum(
                    value == "partial_closure" for value in restrictions
                ),
                "repair_lane_closure_next_24h": sum(
                    value in {"single_lane_closure", "lane_narrowing"}
                    for value in restrictions
                ),
                "repair_intersection_closure_next_24h": sum(
                    value == "intersection_closure" for value in restrictions
                ),
                "repair_bridge_event_next_24h": sum(
                    value == "bridge_repair" for value in event_types
                ),
                "repair_high_severity_count_next_24h": sum(
                    _severity_weight(event["severity"]) >= 3 for event in events
                ),
                "repair_disruption_score_next_24h": round(score, 4),
                "repair_distance_to_nearest_m": min(distances) if distances else None,
                "repair_match_confidence_max": max(confidences) if confidences else 0,
                "repair_open_end_count_next_24h": sum(
                    bool(event["open_end"]) for event in events
                ),
                "repair_hours_until_nearest_start": min(
                    max(0.0, (item - start).total_seconds() / 3600) for item in starts
                )
                if starts
                else None,
            }
        return output

    def _event_features(
        self, rows: pd.DataFrame, start: datetime, end: datetime
    ) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for segment_id, group in rows.groupby("road_segment_id"):
            events = group.to_dict("records")
            unique = {str(event["source_item_id"]): event for event in events}
            events = list(unique.values())
            distances = [
                float(event["distance_m"])
                for event in events
                if pd.notna(event["distance_m"])
            ]
            intensities = [
                float(event["event_intensity_score"] or 0) for event in events
            ]
            starts = [
                event["valid_from"]
                for event in events
                if event["valid_from"] is not None
            ]
            types = [str(event["event_type"] or "") for event in events]
            venues = [str(event["venue"] or "").lower() for event in events]
            output[str(segment_id)] = {
                "event_count_next_24h": len(events),
                "event_count_500m_next_24h": sum(value <= 500 for value in distances),
                "event_count_1000m_next_24h": sum(value <= 1000 for value in distances),
                "event_count_2000m_next_24h": sum(value <= 2000 for value in distances),
                "event_major_next_24h": sum(
                    float(event["event_severity"] or 0) >= 4 for event in events
                ),
                "event_stadium_next_24h": sum(
                    "arena" in venue or "stadium" in venue for venue in venues
                ),
                "event_concert_next_24h": sum(
                    value == "large_concert" for value in types
                ),
                "event_sports_next_24h": sum(
                    value in {"football_match", "hockey_match", "sports_event"}
                    for value in types
                ),
                "event_intensity_sum_next_24h": round(sum(intensities), 4),
                "event_intensity_max_next_24h": max(intensities, default=0),
                "event_distance_to_nearest_m": min(distances) if distances else None,
                "event_hours_until_nearest_start": min(
                    max(0.0, (item - start).total_seconds() / 3600) for item in starts
                )
                if starts
                else None,
                "event_post_event_outflow_risk": int(
                    any(
                        event["valid_to"] is not None
                        and start <= event["valid_to"] < end
                        and float(event["event_severity"] or 0) >= 4
                        for event in events
                    )
                ),
                "event_match_confidence_max": max(
                    float(event["match_confidence"]) for event in events
                ),
            }
        return output

    def build(
        self, prediction_datetime: str | datetime, horizon_hours: int = 24
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if horizon_hours != 24:
            raise ValueError("stage18a_supports_24h_only")
        start = _parse_datetime(prediction_datetime)
        if start is None:
            raise ValueError("prediction_datetime_required")
        end = start + timedelta(hours=24)
        started = time.perf_counter()
        self.warnings = []
        audit = self.audit_inputs()
        ids = self._production_ids()
        weather, weather_meta = self._weather(start)
        matches = self._valid_matches(set(ids))
        gov_events = self._load_events(self.paths.gov_events_path, "gov_kz")
        ticketon_events = self._load_events(self.paths.ticketon_events_path, "ticketon")

        repair_rows = []
        event_rows = []
        for match in matches.to_dict("records"):
            source_id = str(match["source_item_id"])
            if match["provider"] == "gov_kz_repairs":
                event = gov_events.get(source_id)
                if event is None:
                    self.warnings.append(f"gov_event_missing_for_match:{source_id}")
                    continue
                payload = event["payload"]
                valid_from = _parse_datetime(event.get("valid_from"))
                valid_to = _parse_datetime(event.get("valid_to"))
                open_end = bool(event.get("open_end") or payload.get("open_end"))
                if not _overlaps(valid_from, valid_to, start, end, open_end=open_end):
                    continue
                repair_rows.append(
                    match
                    | {
                        "valid_from": valid_from,
                        "valid_to": valid_to,
                        "restriction_type": event.get("restriction_type")
                        or payload.get("restriction_type"),
                        "event_type": event.get("event_type"),
                        "severity": event.get("severity"),
                        "open_end": open_end,
                    }
                )
            elif match["provider"] == "ticketon_events":
                event = ticketon_events.get(source_id)
                if event is None:
                    self.warnings.append(
                        f"ticketon_event_missing_for_match:{source_id}"
                    )
                    continue
                payload = event["payload"]
                valid_from = _parse_datetime(event.get("valid_from"))
                valid_to = _parse_datetime(event.get("valid_to"))
                if not _overlaps(valid_from, valid_to, start, end):
                    continue
                event_rows.append(
                    match
                    | {
                        "valid_from": valid_from,
                        "valid_to": valid_to,
                        "event_type": event.get("event_type"),
                        "event_severity": payload.get("event_severity", 0),
                        "event_intensity_score": payload.get(
                            "event_intensity_score", 0
                        ),
                        "venue": payload.get("venue"),
                    }
                )

        repairs = (
            self._repair_features(pd.DataFrame(repair_rows), start, end)
            if repair_rows
            else {}
        )
        events = (
            self._event_features(pd.DataFrame(event_rows), start, end)
            if event_rows
            else {}
        )
        base = {
            "prediction_datetime": start.isoformat(),
            "forecast_window_start": start.isoformat(),
            "forecast_window_end": end.isoformat(),
            "horizon_hours": 24,
            "generated_at": start.isoformat(),
        }
        rows = []
        for segment_id in ids:
            rows.append(
                base
                | {"road_segment_id": segment_id}
                | weather
                | {
                    "weather_available": weather_meta["available"],
                    "weather_fallback_used": weather_meta["fallback_used"],
                }
                | self._zero_repair()
                | repairs.get(segment_id, {})
                | {"repair_provider_available": int(bool(gov_events))}
                | self._zero_event()
                | events.get(segment_id, {})
                | {"ticketon_provider_available": int(bool(ticketon_events))}
            )
        frame = pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS)
        report = {
            "prediction_datetime": start.isoformat(),
            "forecast_window": [start.isoformat(), end.isoformat()],
            "production_segment_count": len(ids),
            "output_row_count": len(frame),
            "input_audit": audit,
            "provider_availability": {
                "weather": weather_meta,
                "gov_kz": int(bool(gov_events)),
                "ticketon": int(bool(ticketon_events)),
            },
            "repair_records_considered": len(repair_rows),
            "ticketon_records_considered": len(event_rows),
            "stage17_matches_considered": len(matches),
            "matched_segments_by_provider": matches.groupby("provider")[
                "road_segment_id"
            ]
            .nunique()
            .to_dict()
            if not matches.empty
            else {},
            "segments_with_repairs": len(repairs),
            "segments_with_events": len(events),
            "feature_count_by_group": {
                "weather": len(WEATHER_COLUMNS) + 2,
                "repair": len(REPAIR_COLUMNS) + 1,
                "event": len(EVENT_COLUMNS) + 1,
            },
            "missing_value_rates": {
                column: float(frame[column].isna().mean()) for column in frame.columns
            },
            "build_duration_seconds": round(time.perf_counter() - started, 4),
            "warnings": sorted(set(self.warnings + weather_meta["warnings"])),
        }
        return frame, report

    @staticmethod
    def checksum(frame: pd.DataFrame) -> str:
        material = frame.sort_values("road_segment_id").to_json(
            orient="records", date_format="iso", force_ascii=False
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def save(
        self, frame: pd.DataFrame, report: dict[str, Any]
    ) -> dict[str, Path | bool | str]:
        processed = self.paths.output_dir / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        parquet_path = processed / "future_segment_features_24h.parquet"
        json_path = processed / "future_segment_features_24h.json"
        checksum = self.checksum(frame)
        unchanged = (
            parquet_path.exists()
            and self.checksum(
                pd.read_parquet(parquet_path).reindex(columns=FEATURE_COLUMNS)
            )
            == checksum
        )
        if not unchanged:
            frame.to_parquet(parquet_path, index=False)
            json_path.write_text(
                frame.to_json(
                    orient="records", force_ascii=False, date_format="iso", indent=2
                ),
                encoding="utf-8",
            )
        report["output_paths"] = {"parquet": str(parquet_path), "json": str(json_path)}
        report["output_checksum"] = checksum
        report["unchanged"] = unchanged
        return {
            "parquet": parquet_path,
            "json": json_path,
            "checksum": checksum,
            "unchanged": unchanged,
        }


def feature_catalog() -> list[dict[str, Any]]:
    catalog = []
    for name in WEATHER_COLUMNS:
        catalog.append(
            {
                "feature_name": name,
                "provider": "openweather",
                "group": "weather",
                "dtype": "float_or_bool",
                "default_value": None,
                "nullable": True,
                "formula": "validated city-level 24h forecast aggregate",
                "temporal_window": "[prediction_datetime, +24h)",
                "spatial_scope": "city_broadcast",
                "training_status": "requires_historical_backfill",
                "description": "Future weather candidate; separate from frozen model.",
            }
        )
    for name in WEATHER_CONTEXT_COLUMNS:
        catalog.append(
            {
                "feature_name": name,
                "provider": "openweather",
                "group": "weather_context",
                "dtype": "context_metadata",
                "default_value": None,
                "nullable": True,
                "formula": "canonical weather snapshot provenance or 24h summary",
                "temporal_window": "[prediction_datetime, +24h]",
                "spatial_scope": "city_broadcast",
                "training_status": "context_only",
                "description": "Operational weather context; excluded from frozen ML input.",
            }
        )
    for name in ("weather_available", "weather_fallback_used"):
        catalog.append(
            {
                "feature_name": name,
                "provider": "openweather",
                "group": "availability",
                "dtype": "int",
                "default_value": 0,
                "nullable": False,
                "formula": "provider availability/fallback flag",
                "temporal_window": "build",
                "spatial_scope": "city_broadcast",
                "training_status": "context_only",
                "description": "Explicit weather availability metadata.",
            }
        )
    for name in REPAIR_COLUMNS + ("repair_provider_available",):
        catalog.append(
            {
                "feature_name": name,
                "provider": "gov_kz_repairs",
                "group": "repair",
                "dtype": "float_or_int",
                "default_value": 0
                if "distance" not in name and "hours" not in name
                else None,
                "nullable": "distance" in name or "hours" in name,
                "formula": "Stage 17 matched repair events overlapping the 24h window",
                "temporal_window": "[prediction_datetime, +24h)",
                "spatial_scope": "matched_segment",
                "training_status": "requires_historical_backfill",
                "description": "Future repair/closure context only.",
            }
        )
    for name in EVENT_COLUMNS + ("ticketon_provider_available",):
        catalog.append(
            {
                "feature_name": name,
                "provider": "ticketon_events",
                "group": "event",
                "dtype": "float_or_int",
                "default_value": 0
                if "distance" not in name and "hours" not in name
                else None,
                "nullable": "distance" in name or "hours" in name,
                "formula": "Stage 17 matched Ticketon events overlapping the 24h window",
                "temporal_window": "[prediction_datetime, +24h)",
                "spatial_scope": "matched_segment",
                "training_status": "requires_historical_backfill",
                "description": "Future event context only; not attendance.",
            }
        )
    return catalog


def validate_features(
    frame: pd.DataFrame, production_ids: set[str], expected_segment_count: int = 3968
) -> dict[str, Any]:
    count_columns = [
        column
        for column in REPAIR_COLUMNS + EVENT_COLUMNS
        if "count" in column or column.endswith("_next_24h") or column.endswith("_risk")
    ]
    numeric = frame.select_dtypes(include="number")
    return {
        "row_count_3968": len(frame) == expected_segment_count,
        "unique_segments": frame["road_segment_id"].nunique() == len(frame),
        "primary_key_duplicates": int(
            frame.duplicated(
                ["road_segment_id", "prediction_datetime", "horizon_hours"]
            ).sum()
        )
        == 0,
        "all_ids_in_production": frame["road_segment_id"].isin(production_ids).all(),
        "timezone_aware": all(
            _parse_datetime(value).tzinfo is not None
            for value in frame["prediction_datetime"]
        ),
        "window_24h": all(
            (_parse_datetime(end) - _parse_datetime(start)).total_seconds() == 86400
            for start, end in zip(
                frame["forecast_window_start"], frame["forecast_window_end"]
            )
        ),
        "no_infinite_numeric": not bool(
            numeric.map(
                lambda value: isinstance(value, (int, float)) and math.isinf(value)
            )
            .any()
            .any()
        ),
        "nonnegative_count_features": all(
            (frame[column].fillna(0) >= 0).all()
            for column in count_columns
            if column in frame
        ),
        "confidence_in_range": frame["repair_match_confidence_max"].between(0, 1).all()
        and frame["event_match_confidence_max"].between(0, 1).all(),
        "nonnegative_distances": all(
            (frame[column].dropna() >= 0).all()
            for column in (
                "repair_distance_to_nearest_m",
                "event_distance_to_nearest_m",
            )
        ),
        "stable_schema": list(frame.columns) == list(FEATURE_COLUMNS),
    }
