"""Normalize archived Ticketon JSON-LD without changing the live provider."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from html import unescape
from typing import Any

from future_intelligence.utils import ASTANA_TIMEZONE

# Aliases are audit-only. Coordinates exist only where the existing directory
# already supplies them; missing coordinates stay null.
HISTORICAL_VENUES: dict[str, dict[str, Any]] = {
    "astana arena": {"tier": "very_high", "coordinates": (51.1083, 71.4027)},
    "барыс арена": {"tier": "very_high", "coordinates": (51.1156, 71.4446)},
    "barys arena": {"tier": "very_high", "coordinates": (51.1156, 71.4446)},
    "central concert hall kazakhstan": {"tier": "high", "coordinates": None},
    "central concert hall": {"tier": "high", "coordinates": None},
    "qazaqconcert": {"tier": "high", "coordinates": None},
    "дворец мира и согласия": {"tier": "high", "coordinates": None},
    "palace of peace and reconciliation": {"tier": "high", "coordinates": None},
    "дворец независимости": {"tier": "high", "coordinates": None},
    "palace of independence": {"tier": "high", "coordinates": None},
    "expo congress center": {"tier": "high", "coordinates": None},
    "congress center": {"tier": "high", "coordinates": None},
    "expo": {"tier": "medium", "coordinates": (51.0894, 71.4184)},
}


def _clean(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", unescape(str(value or ""))).strip()
    return text or None


def _datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (
        parsed.replace(tzinfo=ASTANA_TIMEZONE)
        if parsed.tzinfo is None
        else parsed.astimezone(ASTANA_TIMEZONE)
    )


def _event_nodes(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [node for item in value for node in _event_nodes(item)]
    if not isinstance(value, dict):
        return []
    types = value.get("@type", [])
    types = types if isinstance(types, list) else [types]
    current = (
        [value] if any(str(item).lower().endswith("event") for item in types) else []
    )
    return (
        current
        + _event_nodes(value.get("@graph", []))
        + _event_nodes(value.get("itemListElement", []))
        + _event_nodes(value.get("subEvent", []))
    )


def _jsonld(html: str) -> tuple[list[Any], int, int]:
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>',
        html,
        re.IGNORECASE,
    )
    parsed: list[Any] = []
    failures = 0
    for block in blocks:
        for candidate in (block.strip(), unescape(block).strip()):
            try:
                # Archived Ticketon JSON-LD can contain literal newlines in
                # description strings.  Keep this tolerance archive-only.
                parsed.append(json.loads(candidate, strict=False))
                break
            except json.JSONDecodeError:
                continue
        else:
            failures += 1
    return parsed, len(blocks), failures


def _venue_info(venue: str | None) -> dict[str, Any]:
    value = (venue or "").casefold()
    for alias, metadata in HISTORICAL_VENUES.items():
        if alias.casefold() in value:
            return metadata | {"alias": alias}
    return {"tier": "unknown", "coordinates": None, "alias": None}


def classify_transport(event: dict[str, Any]) -> dict[str, Any]:
    """Deterministic audit label; it neither deletes nor changes live records."""

    text = " ".join(
        str(event.get(key) or "")
        for key in ("title", "category", "venue", "description")
    ).casefold()
    exclusions = {
        "cinema": ("cinema", "кино"),
        "workshop": ("workshop", "мастер-класс"),
        "museum_session": ("museum", "музей"),
        "online": ("online", "онлайн"),
        "children_recurring": ("детск", "children", "малыш"),
    }
    for reason, terms in exclusions.items():
        if any(term in text for term in terms):
            return {
                "is_transport_relevant": False,
                "transport_impact_score": 0,
                "transport_class": "exclude",
                "exclusion_reason": reason,
            }
    tier = event.get("venue_tier", "unknown")
    if any(
        term in text
        for term in ("football", "хоккей", "match", "матч", "concert", "концерт")
    ):
        score = 5 if tier == "very_high" else 4 if tier == "high" else 3
    elif any(term in text for term in ("festival", "фестиваль", "forum", "форум")):
        score = 4 if tier in {"high", "very_high"} else 3
    else:
        score = 2 if tier in {"high", "very_high"} else 1
    return {
        "is_transport_relevant": score >= 3,
        "transport_impact_score": score,
        "transport_class": {
            1: "low",
            2: "medium",
            3: "high",
            4: "high",
            5: "very_high",
        }[score],
        "exclusion_reason": None,
    }


class TicketonWaybackNormalizer:
    """Archive-only parser producing a Ticketon-compatible audit record."""

    def normalize(
        self, html: str, *, archive_year: int, original_url: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        documents, blocks, failures = _jsonld(html)
        events = [node for document in documents for node in _event_nodes(document)]
        output: list[dict[str, Any]] = []
        for node in events:
            location = node.get("location") or {}
            location = (
                location[0] if isinstance(location, list) and location else location
            )
            location = location if isinstance(location, dict) else {}
            address_data = location.get("address") or {}
            city = (
                _clean(address_data.get("addressLocality"))
                if isinstance(address_data, dict)
                else None
            )
            address = (
                _clean(address_data.get("streetAddress"))
                if isinstance(address_data, dict)
                else _clean(address_data)
            )
            title = _clean(node.get("name"))
            start = _datetime(node.get("startDate"))
            if not title or start is None:
                continue
            venue = _clean(location.get("name"))
            info = _venue_info(venue)
            coordinates = info["coordinates"]
            source_identity = f"{original_url}|{start.isoformat()}|{title}"
            item = {
                "source": "Ticketon Wayback",
                "source_item_id": hashlib.sha256(
                    source_identity.encode("utf-8")
                ).hexdigest()[:24],
                "source_url": original_url,
                "archive_year": archive_year,
                "title": title,
                "start_datetime": start.isoformat(),
                "end_datetime": _datetime(node.get("endDate")).isoformat()
                if _datetime(node.get("endDate"))
                else None,
                "venue": venue,
                "city": city,
                "address": address,
                "category": _clean(node.get("category") or node.get("@type")),
                "description": _clean(node.get("description")),
                "price": (node.get("offers") or {}).get("price")
                if isinstance(node.get("offers"), dict)
                else None,
                "sold_out": str(
                    (node.get("offers") or {}).get("availability", "")
                ).endswith("SoldOut")
                if isinstance(node.get("offers"), dict)
                else None,
                "latitude": coordinates[0] if coordinates else None,
                "longitude": coordinates[1] if coordinates else None,
                "venue_tier": info["tier"],
                "venue_alias": info["alias"],
            }
            item["astana_valid"] = bool(
                city and ("астана" in city.casefold() or "astana" in city.casefold())
            )
            item |= classify_transport(item)
            item["training_eligibility"] = (
                "trainable"
                if (
                    item["astana_valid"]
                    and item["latitude"] is not None
                    and item["is_transport_relevant"]
                )
                else "context_only"
                if item["astana_valid"] and item["start_datetime"]
                else "rejected"
            )
            output.append(item)
        unique = {
            (item["source_item_id"], item["start_datetime"]): item for item in output
        }
        return list(unique.values()), {
            "jsonld_blocks": blocks,
            "jsonld_parse_failures": failures,
            "event_nodes": len(events),
            "normalized_events": len(unique),
        }
