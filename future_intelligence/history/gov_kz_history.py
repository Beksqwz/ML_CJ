"""Leakage-aware historical gov.kz orchestration using the live parser.

This module deliberately does not parse articles itself.  It calls the live
``GovKzRoadEventsProvider.parse_article`` and then records historical-only
provenance, eligibility and idempotent event/match snapshots.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from future_intelligence.geocoding import apply_geocode
from future_intelligence.providers.repairs.gov_kz import GovKzRoadEventsProvider
from future_intelligence.schemas import FutureRecord
from future_intelligence.spatial_matching import SegmentMatch, SpatialMatchingEngine
from future_intelligence.utils import ASTANA_TIMEZONE, to_jsonable

DETAIL_URL = re.compile(
    r"^https://www\.gov\.kz/memleket/entities/astana/press/news/details/(\d+)(?:\?.*)?$",
    re.IGNORECASE,
)
ARTICLE_IDENTITY = ("source", "source_item_id")
MATCH_IDENTITY = ("source", "source_item_id", "road_segment_id")


def official_detail_id(url: str) -> str | None:
    """Return an official Astana detail ID, never accepting another entity."""
    match = DETAIL_URL.match(url.strip())
    return match.group(1) if match else None


def unique_detail_urls(urls: Iterable[str]) -> tuple[list[str], int]:
    accepted: list[str] = []
    seen: set[str] = set()
    rejected = 0
    for url in urls:
        item_id = official_detail_id(url)
        if item_id is None:
            rejected += 1
        elif item_id not in seen:
            seen.add(item_id)
            accepted.append(url)
        else:
            rejected += 1
    return accepted, rejected


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def _content_hash(row: dict[str, Any]) -> str:
    # Collection and match timestamps describe this run, not the canonical
    # historical event.  Excluding them is required for idempotent resumes.
    material = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "content_hash",
            "collected_at",
            "created_at",
            "updated_at",
            "prediction_datetime",
        }
    }
    return hashlib.sha256(
        json.dumps(_parquet_safe(material), ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _parquet_safe(row: dict[str, Any]) -> dict[str, Any]:
    """Keep nested provenance JSON-readable without Arrow empty-struct traps."""

    def value_for_parquet(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
        # Existing Parquet snapshots hold nested values as JSON strings.  Parse
        # them back for the hash so a read/write cycle stays semantically equal.
        if isinstance(value, str) and value[:1] in {"{", "["}:
            try:
                return json.dumps(json.loads(value), ensure_ascii=False, sort_keys=True)
            except json.JSONDecodeError:
                pass
        return None if pd.isna(value) else value

    return {key: value_for_parquet(value) for key, value in row.items()}


def temporal_metadata(record: FutureRecord, audit_at: datetime) -> dict[str, Any]:
    """Express what was knowable at any historical prediction timestamp."""
    warnings: list[str] = []
    published = record.published_at
    start = record.valid_from
    if published is None:
        warnings.append("published_at_missing")
    if start is None:
        warnings.append("valid_from_missing")
    if published and start and start < published:
        warnings.append("event_started_before_publication_use_as_known_at_only")
    if start and start > audit_at:
        warnings.append("event_not_yet_historical_at_audit_time")
    return {
        "known_from": _iso(published),
        "known_until": None,
        "as_known_at_supported": published is not None,
        "temporal_leakage_warning": warnings,
        "temporal_knowledge_eligible": published is not None and start is not None,
        "historical_at_audit_time": start is not None and start <= audit_at,
    }


def training_eligibility(
    event: dict[str, Any], matches: Iterable[SegmentMatch]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not official_detail_id(str(event.get("source_url") or "")):
        reasons.append("not_official_astana_detail_url")
    if not event.get("relevant"):
        reasons.append("not_relevant_road_event")
    if not event.get("published_at"):
        reasons.append("published_at_missing")
    if not event.get("valid_from"):
        reasons.append("valid_from_missing")
    if not event.get("valid_to") and not event.get("open_end"):
        reasons.append("valid_to_missing_without_open_end")
    if not event.get("historical_at_audit_time"):
        reasons.append("not_historical_at_audit_time")
    if not event.get("temporal_knowledge_eligible"):
        reasons.append("temporal_knowledge_not_defensible")
    if not list(matches):
        reasons.append("no_valid_production_segment_match")
    return not reasons, reasons


@dataclass
class HistoryWriteResult:
    new: int = 0
    updated: int = 0
    unchanged: int = 0


class GovKzHistoricalBackfill:
    """Small resumable historical pilot; live parser and matcher stay canonical."""

    def __init__(
        self,
        provider: GovKzRoadEventsProvider,
        matcher: SpatialMatchingEngine | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.provider = provider
        self.matcher = matcher or SpatialMatchingEngine()
        self.output_dir = output_dir or (
            Path(__file__).resolve().parents[2]
            / "data"
            / "future_intelligence"
            / "history"
            / "gov_kz"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "gov_kz_history_checkpoints.json"

    def load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {"seen_source_item_ids": [], "pages": []}
        return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

    def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoint_path.write_text(
            json.dumps(to_jsonable(checkpoint), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def register_listing_urls(
        self, urls: Iterable[str], listing_page: int | None = None
    ) -> dict[str, Any]:
        checkpoint = self.load_checkpoint()
        accepted, rejected = unique_detail_urls(urls)
        seen = set(checkpoint.get("seen_source_item_ids", []))
        new = [url for url in accepted if official_detail_id(url) not in seen]
        seen.update(official_detail_id(url) for url in new)
        checkpoint["seen_source_item_ids"] = sorted(seen)
        if listing_page is not None:
            checkpoint.setdefault("pages", []).append(
                {
                    "page": listing_page,
                    "new_ids": len(new),
                    "recorded_at": _iso(datetime.now(UTC)),
                }
            )
        self.save_checkpoint(checkpoint)
        return {
            "accepted_urls": new,
            "duplicates_or_invalid": rejected + (len(accepted) - len(new)),
            "checkpoint": checkpoint,
        }

    def record_from_html(
        self,
        html: str,
        url: str,
        *,
        language: str = "ru",
        audit_at: datetime | None = None,
    ) -> tuple[dict[str, Any], list[SegmentMatch]]:
        """Normalize rendered/request HTML through the existing live parser."""
        item_id = official_detail_id(url)
        if item_id is None:
            raise ValueError("official_astana_detail_url_required")
        audit_at = (audit_at or datetime.now(ASTANA_TIMEZONE)).astimezone(
            ASTANA_TIMEZONE
        )
        parsed = self.provider.parse_article(html, url, item_id, language)
        records = self.provider.normalize({"parsed": parsed}, audit_at, 24)
        record = records[0]
        record.is_historical = True
        point = self.provider.geocoder.repair(record.payload["location"])
        apply_geocode(record, point)
        line = self.provider.road_geometry.repair(record.payload["location"])
        record.payload["repair_geometry_quality"] = line.quality
        record.warnings.extend(line.warnings)
        if line.geometry is not None:
            record.geometry = line.geometry
            record.confidence = max(record.confidence or 0.0, line.confidence)
        matches = self.matcher.match_record(record, provider="gov_kz_repairs").matches
        event = record.to_dict()
        event.update(
            {
                "title": parsed["title"],
                "description": parsed["description"],
                "restriction_type": parsed["restriction_type"],
                "location": parsed["location"],
                "relevant": parsed["relevant"],
                "relevance_score": parsed["relevance_score"],
                "open_end": parsed["open_end"],
                "date_confidence": "open_end"
                if parsed["open_end"]
                else "parsed"
                if parsed["valid_from"]
                else "missing",
                "date_extraction_method": "live_gov_kz_parse_dates",
                "date_warnings": parsed["warnings"],
                **temporal_metadata(record, audit_at),
            }
        )
        eligible, reasons = training_eligibility(event, matches)
        event["training_eligible"] = eligible
        event["training_rejection_reasons"] = reasons
        event["overall_confidence"] = record.confidence
        event["content_hash"] = _content_hash(event)
        return event, matches

    @staticmethod
    def _upsert(
        path: Path, incoming: list[dict[str, Any]], identity: tuple[str, ...]
    ) -> HistoryWriteResult:
        # An empty pilot is valid: never leave a zero-byte parquet that makes
        # a later resume fail before it can collect anything.
        try:
            previous = (
                pd.read_parquet(path)
                if path.exists() and path.stat().st_size
                else pd.DataFrame()
            )
        except (OSError, ValueError):
            previous = pd.DataFrame()
        existing = {}
        if not previous.empty:
            for row in previous.to_dict("records"):
                existing[tuple(str(row.get(key)) for key in identity)] = row
        result = HistoryWriteResult()
        for row in incoming:
            row["content_hash"] = _content_hash(row)
            key = tuple(str(row.get(field)) for field in identity)
            old = existing.get(key)
            if old is None:
                result.new += 1
            elif _content_hash(old) == row["content_hash"]:
                result.unchanged += 1
            else:
                result.updated += 1
            existing[key] = row
        columns = list(
            dict.fromkeys(
                [
                    *identity,
                    "content_hash",
                    *(key for row in incoming for key in row),
                ]
            )
        )
        pd.DataFrame(
            [_parquet_safe(row) for row in existing.values()], columns=columns
        ).to_parquet(path, index=False)
        return result

    def save(
        self, events: list[dict[str, Any]], matches: list[SegmentMatch]
    ) -> dict[str, Any]:
        article_rows = [
            {
                key: event.get(key)
                for key in (
                    "source",
                    "source_item_id",
                    "source_url",
                    "title",
                    "description",
                    "published_at",
                    "content_hash",
                    "relevant",
                    "relevance_score",
                )
            }
            for event in events
        ]
        match_rows = []
        for match in matches:
            row = match.to_dict()
            row["source"] = "gov.kz Astana Akimat"
            match_rows.append(row)
        paths = {
            "articles": self.output_dir / "gov_kz_historical_articles.parquet",
            "events": self.output_dir / "gov_kz_historical_events.parquet",
            "matches": self.output_dir / "gov_kz_historical_segment_matches.parquet",
            "rejected": self.output_dir / "gov_kz_historical_rejected.parquet",
        }
        results = {
            "articles": self._upsert(paths["articles"], article_rows, ARTICLE_IDENTITY),
            "events": self._upsert(paths["events"], events, ARTICLE_IDENTITY),
            "matches": self._upsert(paths["matches"], match_rows, MATCH_IDENTITY),
            "rejected": self._upsert(
                paths["rejected"],
                [event for event in events if not event["training_eligible"]],
                ARTICLE_IDENTITY,
            ),
        }
        (self.output_dir / "gov_kz_historical_articles.json").write_text(
            json.dumps(to_jsonable(article_rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "paths": {key: str(value) for key, value in paths.items()},
            "writes": {key: result.__dict__ for key, result in results.items()},
        }
