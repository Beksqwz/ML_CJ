"""Canonical, context-only unified Future Feature Layer for the 24-hour horizon.

This module deliberately consumes the already validated Stage 18A segment table.
It does not participate in frozen-model feature construction or inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from future_intelligence.segment_feature_builder import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = ROOT / "data" / "future_intelligence" / "processed"
PROVIDER_METADATA_COLUMNS = (
    "weather_provider_available",
    "ticketon_provider_available",
    "gov_provider_available",
    "weather_collection_timestamp",
    "ticketon_collection_timestamp",
    "gov_collection_timestamp",
    "provider_warning_count",
    "provider_degraded",
    "metadata_generated_at",
)
ALLOWED_TRAINING_STATUSES = {
    "trainable_now",
    "requires_historical_backfill",
    "context_only",
    "collect_for_future",
}


@dataclass(frozen=True)
class UnifiedPaths:
    """File locations consumed and produced by the Stage 18B layer."""

    segment_features_path: Path = (
        DEFAULT_PROCESSED / "future_segment_features_24h.parquet"
    )
    weather_path: Path = DEFAULT_PROCESSED / "openweather_24h_features.parquet"
    gov_path: Path = DEFAULT_PROCESSED / "gov_kz_road_events.parquet"
    ticketon_path: Path = DEFAULT_PROCESSED / "ticketon_events.parquet"
    traffic_path: Path = DEFAULT_PROCESSED / "tomtom_live_traffic.parquet"
    output_dir: Path = ROOT / "data" / "future_intelligence"


def _file_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()


class UnifiedFutureLayerBuilder:
    """Integrate provider context into one stable, separate feature table."""

    def __init__(
        self, paths: UnifiedPaths | None = None, *, expected_rows: int = 3968
    ) -> None:
        self.paths = paths or UnifiedPaths()
        self.expected_rows = expected_rows

    def provider_coverage(self, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Report collection availability independently from zero event counts."""

        definitions = {
            "weather": {
                "path": self.paths.weather_path,
                "stage18a_flag": "weather_available",
            },
            "gov": {
                "path": self.paths.gov_path,
                "stage18a_flag": "repair_provider_available",
            },
            "ticketon": {
                "path": self.paths.ticketon_path,
                "stage18a_flag": "ticketon_provider_available",
            },
            "traffic": {
                "path": self.paths.traffic_path,
                "stage18a_flag": "traffic_context_available",
            },
        }
        coverage: dict[str, dict[str, Any]] = {}
        for provider, definition in definitions.items():
            path = definition["path"]
            flag = definition["stage18a_flag"]
            table_available = bool(path.exists())
            stage18a_available = (
                bool(frame[flag].fillna(0).astype(bool).any())
                if flag in frame
                else False
            )
            # The provider table is the collection-level source of truth.  The
            # Stage 18A flag is retained as diagnostic evidence, not used to
            # confuse an available-but-zero-event provider with an outage.
            available = int(table_available)
            coverage[provider] = {
                "available": available,
                "source_file": str(path),
                "source_file_exists": table_available,
                "stage18a_availability_evidence": stage18a_available,
                "collection_timestamp": _file_timestamp(path),
                "warning_count": 0 if available else 1,
                "degraded": int(not available),
            }
        return coverage

    def build(self) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        """Return the unified table, integration report and collision report."""

        if not self.paths.segment_features_path.exists():
            raise FileNotFoundError("stage18a_segment_features_missing")

        raw = pd.read_parquet(self.paths.segment_features_path)
        duplicate_input_columns = raw.columns[raw.columns.duplicated()].tolist()
        missing_base_columns = [
            column for column in FEATURE_COLUMNS if column not in raw
        ]
        if missing_base_columns:
            raise ValueError(f"stage18a_schema_missing_columns:{missing_base_columns}")
        base = raw.loc[:, FEATURE_COLUMNS].copy()
        if (
            len(base) != self.expected_rows
            or base["road_segment_id"].nunique() != self.expected_rows
        ):
            raise ValueError("stage18a_row_grain_invalid")

        base_columns = {
            "road_segment_id",
            "prediction_datetime",
            "forecast_window_start",
            "forecast_window_end",
            "horizon_hours",
            "generated_at",
            # This is a deliberately retained Stage 18A availability field.
            # It belongs to provider metadata rather than the event namespace.
            "ticketon_provider_available",
        }
        collisions = {
            "duplicate_input_columns": duplicate_input_columns,
            # ``ticketon_provider_available`` is an intentional Stage 18A
            # availability flag retained at its established schema position.
            "reserved_metadata_collisions": [
                column
                for column in raw.columns
                if column
                in set(PROVIDER_METADATA_COLUMNS) - {"ticketon_provider_available"}
            ],
            "namespace_violations": [
                column
                for column in base.columns
                if column not in base_columns
                and not column.startswith(("weather_", "repair_", "event_", "traffic_"))
            ],
            "dtype_mismatches": [],
            "nullability_conflicts": [],
        }
        coverage = self.provider_coverage(base)
        metadata = {
            "weather_provider_available": coverage["weather"]["available"],
            "ticketon_provider_available": coverage["ticketon"]["available"],
            "gov_provider_available": coverage["gov"]["available"],
            "weather_collection_timestamp": coverage["weather"]["collection_timestamp"],
            "ticketon_collection_timestamp": coverage["ticketon"][
                "collection_timestamp"
            ],
            "traffic_context_available": coverage["traffic"]["available"],
            "gov_collection_timestamp": coverage["gov"]["collection_timestamp"],
            "provider_warning_count": sum(
                item["warning_count"] for item in coverage.values()
            ),
            "provider_degraded": int(
                any(item["degraded"] for item in coverage.values())
            ),
            # Stage 18A's deterministic generation timestamp is used instead
            # of the current clock so identical inputs remain identical.
            "metadata_generated_at": base["generated_at"],
        }
        unified = base.assign(**metadata)
        # Operational severity is a separate explanation contract, always 0..1.
        # Preserve provenance so a provider-scale issue remains diagnosable.
        if "weather_severity_score" in unified:
            original = pd.to_numeric(unified["weather_severity_score"], errors="coerce")
            unified["weather_severity_original_value"] = original
            maximum = (
                float(original.max(skipna=True)) if original.notna().any() else 1.0
            )
            scale = 1 if maximum <= 1 else (5 if maximum <= 5 else 10)
            unified["weather_severity_original_scale"] = scale
            normalized = original / scale
            out_of_range = normalized.notna() & ~normalized.between(0, 1)
            unified["weather_severity_data_quality_warning"] = out_of_range.astype(
                "int8"
            )
            unified["weather_severity_score"] = normalized.clip(0, 1)
        report = {
            "rows": len(unified),
            "columns": len(unified.columns),
            "primary_key": ["road_segment_id", "prediction_datetime", "horizon_hours"],
            "provider_coverage": coverage,
            "feature_counts_by_namespace": self.feature_counts(unified),
            "collisions": collisions,
        }
        return unified, report, collisions

    @staticmethod
    def feature_counts(frame: pd.DataFrame) -> dict[str, int]:
        prefixes = ("weather_", "repair_", "event_", "traffic_", "metadata_")
        counts = {prefix[:-1]: 0 for prefix in prefixes}
        for column in frame.columns:
            for prefix in prefixes:
                if column.startswith(prefix):
                    counts[prefix[:-1]] += 1
                    break
        return counts

    @staticmethod
    def checksum(frame: pd.DataFrame) -> str:
        canonical = frame.sort_values("road_segment_id").to_json(
            orient="records", force_ascii=False, date_format="iso"
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Persist stable Parquet and JSON exports without duplicate rows."""

        processed = self.paths.output_dir / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        parquet_path = processed / "unified_future_features_24h.parquet"
        json_path = processed / "unified_future_features_24h.json"
        checksum = self.checksum(frame)
        unchanged = False
        if parquet_path.exists():
            previous = pd.read_parquet(parquet_path).reindex(columns=frame.columns)
            unchanged = self.checksum(previous) == checksum
        if not unchanged:
            frame.to_parquet(parquet_path, index=False)
            json_path.write_text(
                frame.to_json(
                    orient="records", force_ascii=False, date_format="iso", indent=2
                ),
                encoding="utf-8",
            )
        return {
            "parquet": str(parquet_path),
            "json": str(json_path),
            "checksum": checksum,
            "unchanged": unchanged,
        }


def feature_provenance(frame: pd.DataFrame, source_file: Path) -> list[dict[str, Any]]:
    """Return one provenance record per canonical output feature."""

    catalog_path = ROOT / "reports" / "stage18a" / "feature_catalog.json"
    stage18a_catalog: dict[str, dict[str, Any]] = {}
    if catalog_path.exists():
        for item in json.loads(catalog_path.read_text(encoding="utf-8")):
            stage18a_catalog[item["feature_name"]] = item

    items: list[dict[str, Any]] = []
    for name in frame.columns:
        inherited = stage18a_catalog.get(name, {})
        training_status = inherited.get("training_status", "context_only")
        if training_status not in ALLOWED_TRAINING_STATUSES:
            training_status = "context_only"
        items.append(
            {
                "feature_name": name,
                "provider": inherited.get(
                    "provider",
                    "metadata" if name in PROVIDER_METADATA_COLUMNS else "stage18a",
                ),
                "source_file": str(source_file),
                "generation_timestamp": _file_timestamp(source_file),
                "training_status": training_status,
                "description": inherited.get(
                    "description",
                    "Unified Future Feature Layer metadata or inherited Stage 18A feature.",
                ),
            }
        )
    return items


def validate(frame: pd.DataFrame, expected_rows: int = 3968) -> dict[str, bool]:
    """Validate row grain, namespaces and safe numeric/timestamp properties."""

    numeric = frame.select_dtypes(include="number")
    finite = numeric.map(
        lambda value: not isinstance(value, (int, float)) or not math.isinf(value)
    )
    primary_key = ["road_segment_id", "prediction_datetime", "horizon_hours"]
    timestamps = [
        "prediction_datetime",
        "forecast_window_start",
        "forecast_window_end",
        "generated_at",
        "metadata_generated_at",
    ]
    return {
        "row_count": bool(len(frame) == expected_rows),
        "unique_segments": bool(frame["road_segment_id"].nunique() == expected_rows),
        "duplicate_primary_keys": bool(not frame.duplicated(primary_key).any()),
        "duplicate_columns": bool(not frame.columns.duplicated().any()),
        "stable_base_schema": bool(
            list(frame.columns[: len(FEATURE_COLUMNS)]) == list(FEATURE_COLUMNS)
        ),
        "timezone_aware": bool(
            all("+" in str(value) for column in timestamps for value in frame[column])
        ),
        "no_infinite_values": bool(finite.all().all()),
        "provider_metadata_present": bool(
            all(column in frame for column in PROVIDER_METADATA_COLUMNS)
        ),
    }
