"""Persist safe normalized and engineered outputs; never persist API credentials."""

from __future__ import annotations

import json
import hashlib
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
    safe_timestamp = prediction_datetime.replace(":", "-")
    raw_path = raw_dir / f"{safe_timestamp}.json"
    raw_path.write_text(
        json.dumps(to_jsonable(safe_raw), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    normalized = pd.DataFrame(
        [record.to_dict() for record in result.normalized_records]
    )
    provider = result.metadata.provider_name
    points_path = output_dir / "processed" / f"{provider}_records.parquet"
    points_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(points_path, index=False)
    feature_row = {
        "prediction_datetime": prediction_datetime,
        "road_segment_id": None,
        "provider": result.metadata.provider_name,
        **result.features,
    }
    feature_path = output_dir / "processed" / f"{provider}_24h_features.parquet"
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


def save_ticketon_result(
    result: ProviderResult, output_dir: Path
) -> tuple[dict[str, Path], dict[str, int]]:
    """Upsert canonical Ticketon events by source identity and content hash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw" / "ticketon"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "ticketon_collection.json"
    safe_raw = [
        {
            key: value
            for key, value in row.items()
            if key not in {"appid", "api_key", "key"}
        }
        for row in result.raw_records
    ]
    raw_path.write_text(
        json.dumps(to_jsonable(safe_raw), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    event_path = output_dir / "processed" / "ticketon_events.parquet"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "processed" / "ticketon_events.json"
    rows_by_identity: dict[tuple[str, str | None], dict] = {}
    for record in result.normalized_records:
        item = record.to_dict()
        payload = item.pop("payload")
        content = {
            key: item.get(key)
            for key in (
                "source",
                "source_item_id",
                "source_url",
                "valid_from",
                "valid_to",
                "event_type",
                "severity",
            )
        } | {
            key: payload.get(key)
            for key in (
                "name",
                "category",
                "venue",
                "address",
                "city",
                "event_severity",
                "event_intensity_score",
                "venue_capacity",
            )
        }
        content_hash = hashlib.sha256(
            json.dumps(to_jsonable(content), ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        row = item | {
            "name": payload.get("name"),
            "category": payload.get("category"),
            "venue": payload.get("venue"),
            "address": payload.get("address"),
            "city": payload.get("city"),
            "event_severity": payload.get("event_severity"),
            "event_intensity_score": payload.get("event_intensity_score"),
            "venue_capacity": payload.get("venue_capacity"),
            "content_hash": content_hash,
            "payload_json": json.dumps(to_jsonable(payload), ensure_ascii=False),
        }
        rows_by_identity[(row["source"], row["source_item_id"])] = row

    previous = pd.read_parquet(event_path) if event_path.exists() else pd.DataFrame()
    old_hash = {
        (row["source"], row["source_item_id"]): row["content_hash"]
        for _, row in previous.iterrows()
        if "content_hash" in previous.columns
    }
    kept = previous.copy()
    new = updated = unchanged = 0
    for identity, row in rows_by_identity.items():
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
    feature_path = output_dir / "processed" / "ticketon_event_features.parquet"
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


def save_tomtom_result(
    result: ProviderResult, output_dir: Path
) -> tuple[dict[str, Path], dict[str, int]]:
    """Upsert bounded TomTom context by segment and prediction timestamp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw" / "tomtom"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "tomtom_collection.json"
    raw_path.write_text(
        json.dumps(to_jsonable(result.raw_records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    event_path = output_dir / "processed" / "tomtom_live_traffic.parquet"
    json_path = output_dir / "processed" / "tomtom_live_traffic.json"
    rows = []
    for record in result.normalized_records:
        item = record.to_dict()
        payload = item.pop("payload")
        content_hash = hashlib.sha256(
            json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        rows.append(
            item
            | {
                "road_segment_id": (record.affected_road_segment_ids or [None])[0],
                "content_hash": content_hash,
                "payload_json": json.dumps(to_jsonable(payload), ensure_ascii=False),
            }
        )
    previous = pd.read_parquet(event_path) if event_path.exists() else pd.DataFrame()
    old_hash = (
        {
            (row["source_item_id"], row["road_segment_id"]): row["content_hash"]
            for _, row in previous.iterrows()
        }
        if not previous.empty
        else {}
    )
    kept = previous.copy()
    new = updated = unchanged = 0
    for row in rows:
        identity = (row["source_item_id"], row["road_segment_id"])
        if identity not in old_hash:
            new += 1
            kept = pd.concat([kept, pd.DataFrame([row])], ignore_index=True)
        elif old_hash[identity] == row["content_hash"]:
            unchanged += 1
        else:
            updated += 1
            kept = kept.loc[
                ~(
                    (kept.source_item_id == identity[0])
                    & (kept.road_segment_id == identity[1])
                )
            ]
            kept = pd.concat([kept, pd.DataFrame([row])], ignore_index=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(event_path, index=False)
    json_path.write_text(
        kept.to_json(orient="records", force_ascii=False, date_format="iso", indent=2),
        encoding="utf-8",
    )
    return {"raw": raw_path, "events": event_path, "json": json_path}, {
        "new": new,
        "updated": updated,
        "unchanged": unchanged,
    }
