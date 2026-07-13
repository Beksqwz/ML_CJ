"""Persist safe normalized and engineered outputs; never persist API credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from future_intelligence.schemas import ProviderResult
from future_intelligence.utils import to_jsonable


def save_result(
    result: ProviderResult, output_dir: Path, prediction_datetime: str
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw" / result.metadata.provider_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_raw = [
        {
            key: value
            for key, value in raw.items()
            if key not in {"appid", "api_key", "key"}
        }
        for raw in result.raw_records
    ]
    raw_path = raw_dir / f"{prediction_datetime}.json"
    raw_path.write_text(
        json.dumps(to_jsonable(safe_raw), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    normalized = pd.DataFrame(
        [record.to_dict() for record in result.normalized_records]
    )
    points_path = output_dir / "processed" / "openweather_forecast_points.parquet"
    points_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(points_path, index=False)
    feature_row = {
        "prediction_datetime": prediction_datetime,
        "road_segment_id": None,
        "provider": result.metadata.provider_name,
        **result.features,
    }
    feature_path = output_dir / "processed" / "openweather_24h_features.parquet"
    pd.DataFrame([feature_row]).to_parquet(feature_path, index=False)
    universal_path = output_dir / "processed" / "future_intelligence_features.parquet"
    pd.DataFrame([feature_row]).to_parquet(universal_path, index=False)
    return {
        "raw": raw_path,
        "points": points_path,
        "features": feature_path,
        "universal": universal_path,
    }


def save_gov_kz_result(
    result: ProviderResult, output_dir: Path
) -> tuple[dict[str, Path], dict[str, int]]:
    """Upsert gov.kz rows by source/item id and content hash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw" / "gov_kz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "gov_kz_collection.json"
    raw_path.write_text(
        json.dumps(to_jsonable(result.raw_records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    event_path = output_dir / "processed" / "gov_kz_road_events.parquet"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "processed" / "gov_kz_road_events.json"
    rows = []
    for record in result.normalized_records:
        item = record.to_dict()
        payload = item.pop("payload")
        location = payload.get("location", {})
        rows.append(
            item
            | {
                "title": payload.get("title"),
                "description": payload.get("description"),
                "restriction_type": payload.get("restriction_type"),
                "language": payload.get("language"),
                "content_hash": payload.get("content_hash"),
                "open_end": payload.get("open_end"),
                "road_name": location.get("road_name"),
                "from_street": location.get("from_street"),
                "to_street": location.get("to_street"),
                "intersection_streets": location.get("intersection_streets"),
                "payload_json": json.dumps(to_jsonable(payload), ensure_ascii=False),
            }
        )
    previous = pd.read_parquet(event_path) if event_path.exists() else pd.DataFrame()
    old_hash = {
        (row["source"], row["source_item_id"]): row["content_hash"]
        for _, row in previous.iterrows()
        if "content_hash" in previous.columns
    }
    new = updated = unchanged = 0
    kept = previous.copy()
    for row in rows:
        identity = (row["source"], row["source_item_id"])
        if identity not in old_hash:
            new += 1
            kept = pd.concat([kept, pd.DataFrame([row])], ignore_index=True)
        elif old_hash[identity] == row["content_hash"]:
            unchanged += 1
        else:
            updated += 1
            kept = kept.loc[
                ~((kept.source == identity[0]) & (kept.source_item_id == identity[1]))
            ]
            kept = pd.concat([kept, pd.DataFrame([row])], ignore_index=True)
    kept.to_parquet(event_path, index=False)
    json_path.write_text(
        kept.to_json(orient="records", force_ascii=False, date_format="iso", indent=2),
        encoding="utf-8",
    )
    feature_path = output_dir / "processed" / "gov_kz_repair_features.parquet"
    pd.DataFrame(
        [
            {
                "prediction_datetime": result.normalized_records[
                    0
                ].prediction_datetime.isoformat()
                if result.normalized_records
                else None,
                "road_segment_id": None,
                "provider": result.metadata.provider_name,
                **result.features,
            }
        ]
    ).to_parquet(feature_path, index=False)
    return {
        "raw": raw_path,
        "events": event_path,
        "json": json_path,
        "features": feature_path,
    }, {"new": new, "updated": updated, "unchanged": unchanged}
