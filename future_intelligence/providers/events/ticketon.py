"""Public Ticketon event collector for Astana; output is future context only."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime, timedelta
from html import unescape
from typing import Any, Callable
from urllib.parse import urljoin

import requests

from future_intelligence.providers.events.base import EventsProvider
from future_intelligence.geocoding import AstanaGeocoder, apply_geocode
from future_intelligence.schemas import FutureRecord, ProviderMetadata, ProviderResult
from future_intelligence.utils import ASTANA_TIMEZONE, parse_prediction_datetime

TICKETON_ASTANA_URL = "https://ticketon.kz/astana"
TICKETON_ROBOTS_URL = "https://ticketon.kz/robots.txt"
# Local, auditable venue directory. Unknown venues deliberately remain ungeocoded.
KNOWN_VENUES = {
    "astana arena": (51.1083, 71.4027, 30000),
    "барыс арена": (51.1156, 71.4446, 12000),
    "barys arena": (51.1156, 71.4446, 12000),
    "expo": (51.0894, 71.4184, 5000),
    "конгресс-центр": (51.0891, 71.4180, 3000),
    "congress centre": (51.0891, 71.4180, 3000),
}


def _clean(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", unescape(str(value or ""))).strip()
    return text or None


def _events(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [node for item in value for node in _events(item)]
    if not isinstance(value, dict):
        return []
    types = value.get("@type", [])
    types = types if isinstance(types, list) else [types]
    found = [value] if any(str(t).lower().endswith("event") for t in types) else []
    # Prefer published session-level sub-events to their broad parent event.
    return (
        _events(value.get("@graph", []))
        + _events(value.get("itemListElement", []))
        + _events(value.get("subEvent", []))
        + found
    )


def _datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return (
        parsed.replace(tzinfo=ASTANA_TIMEZONE)
        if parsed.tzinfo is None
        else parsed.astimezone(ASTANA_TIMEZONE)
    )


def _classify(name: str, category: str | None, venue: str | None) -> tuple[str, int]:
    value = " ".join(filter(None, (name, category, venue))).lower()
    if any(
        x in value for x in ("футбол", "football", "хоккей", "hockey", "матч", "match")
    ):
        return "football_match", 5
    if any(x in value for x in ("концерт", "concert", "шоу", "show")):
        return "large_concert", 5
    if any(x in value for x in ("фестив", "festival")):
        return "festival", 4
    if any(x in value for x in ("выставк", "exhibition", "expo")):
        return "exhibition", 2
    if any(x in value for x in ("театр", "theatre", "спектакл")):
        return "theatre", 1
    return "public_event", 2


def _capacity_multiplier(capacity: int | None) -> float:
    if capacity is None:
        return 1.0
    if capacity >= 30000:
        return 1.5
    if capacity >= 10000:
        return 1.3
    if capacity >= 5000:
        return 1.15
    return 1.0


class TicketonEventsProvider(EventsProvider):
    metadata = ProviderMetadata(
        "ticketon_events", "1.0", "events", (24,), False, "on-demand", "Astana"
    )

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        geocoder: AstanaGeocoder | None = None,
        timeout_seconds: float = 10,
        max_retries: int = 2,
        request_delay: float = 0.25,
        max_event_pages: int = 30,
        sleep: Callable[[float], None] = time.sleep,
        listing_url: str = TICKETON_ASTANA_URL,
    ):
        (
            self.session,
            self.timeout_seconds,
            self.max_retries,
            self.request_delay,
            self.max_event_pages,
            self.sleep,
            self.listing_url,
        ) = (
            session or requests.Session(),
            timeout_seconds,
            max_retries,
            request_delay,
            max_event_pages,
            sleep,
            listing_url,
        )
        self.geocoder = geocoder or AstanaGeocoder(session=self.session, sleep=sleep)

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": self.metadata.provider_name,
            "listing": self.listing_url,
        }

    def _get(self, url: str) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout_seconds,
                    headers={
                        "User-Agent": "AstanaFutureIntelligence/1.0 (+public-data)"
                    },
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                self.sleep(0.25 * (2**attempt))
        raise RuntimeError("unreachable")

    def _robots_allow_listing(self) -> bool | None:
        try:
            robots = self._get(TICKETON_ROBOTS_URL)
        except requests.RequestException:
            return None
        # robots.txt rules apply to the current user-agent group, not to every
        # group in the file.  Ticketon, for example, disallows `/` for several
        # named bots while allowing the generic `*` group.
        active_agents: list[str] = []
        rules: list[tuple[list[str], str]] = []
        for raw in robots.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key, value = key.lower(), value.lower()
            if key == "user-agent":
                active_agents = [value]
            elif key in {"allow", "disallow"} and active_agents:
                rules.append((active_agents[:], f"{key}:{value}"))
        applicable = [
            rule
            for agents, rule in rules
            if "*" in agents or "astanafutureintelligence" in agents
        ]
        disallowed = [
            rule.removeprefix("disallow:")
            for rule in applicable
            if rule.startswith("disallow:")
        ]
        allowed = [
            rule.removeprefix("allow:")
            for rule in applicable
            if rule.startswith("allow:")
        ]
        # Longest matching path wins; an equal Allow wins over Disallow.
        path = "/astana"
        matches = [
            (len(item), False) for item in disallowed if item and path.startswith(item)
        ] + [(len(item), True) for item in allowed if item and path.startswith(item)]
        return True if not matches else max(matches)[1]

    def parse_listing(self, html: str) -> list[dict[str, Any]]:
        output = []
        pattern = (
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>'
        )
        for match in re.finditer(pattern, html, re.I):
            try:
                nodes = _events(json.loads(unescape(match.group(1)).strip()))
            except json.JSONDecodeError:
                continue
            for node in nodes:
                name, start = _clean(node.get("name")), _datetime(node.get("startDate"))
                location = node.get("location") or {}
                location = (
                    location[0] if isinstance(location, list) and location else location
                )
                location = location if isinstance(location, dict) else {}
                venue, category = (
                    _clean(location.get("name")),
                    _clean(node.get("category") or node.get("@type")),
                )
                address_data = location.get("address") or {}
                city = (
                    _clean(address_data.get("addressLocality"))
                    if isinstance(address_data, dict)
                    else None
                )
                address = _clean(
                    address_data
                    if isinstance(address_data, str)
                    else address_data.get("streetAddress")
                )
                if not name or not start:
                    continue
                match_venue = next(
                    (
                        data
                        for key, data in KNOWN_VENUES.items()
                        if key in (venue or "").lower()
                    ),
                    None,
                )
                event_type, severity = _classify(name, category, venue)
                url = urljoin(
                    self.listing_url, _clean(node.get("url")) or self.listing_url
                )
                identity = (_clean(node.get("@id")) or url) + "|" + start.isoformat()
                capacity = match_venue[2] if match_venue else None
                output.append(
                    {
                        "source_item_id": hashlib.sha256(identity.encode()).hexdigest()[
                            :24
                        ],
                        "source_url": url,
                        "name": name,
                        "category": category,
                        "venue": venue,
                        "address": address,
                        "city": city or "Astana",
                        "valid_from": start,
                        "valid_to": _datetime(node.get("endDate")),
                        "event_type": event_type,
                        "event_severity": severity,
                        "event_intensity_score": severity
                        * _capacity_multiplier(capacity),
                        "latitude": match_venue[0] if match_venue else None,
                        "longitude": match_venue[1] if match_venue else None,
                        "venue_capacity": capacity,
                        "geocoding_quality": "local_directory"
                        if match_venue
                        else "not_geocoded",
                    }
                )
        # Parent Event and its first subEvent often describe the same session
        # with different end dates.  One road-impact record per venue/start is
        # the correct grain; child events were emitted first above.
        unique: dict[tuple[str, str | None, datetime], dict[str, Any]] = {}
        for item in output:
            # `_events` emits the detailed sub-event first.  Keep that first
            # record rather than overwriting it with its broad parent event.
            unique.setdefault((item["name"], item["venue"], item["valid_from"]), item)
        return list(unique.values())

    def discover_event_urls(self, html: str) -> list[str]:
        """Extract only public Astana event detail links from the listing."""
        urls = re.findall(r'href=["\'](?P<url>/astana/event/[^"\'?#]+)', html, re.I)
        return list(dict.fromkeys(urljoin(self.listing_url, value) for value in urls))

    def normalize(
        self,
        raw_payload: dict[str, Any],
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> list[FutureRecord]:
        start, end, now = (
            parse_prediction_datetime(prediction_datetime),
            parse_prediction_datetime(prediction_datetime)
            + timedelta(hours=horizon_hours),
            datetime.now(UTC),
        )
        result = []
        for item in raw_payload.get("events", []):
            if (
                "астана" not in item["city"].lower()
                and "astana" not in item["city"].lower()
            ):
                continue
            if not (
                item["valid_from"] < end
                and (item["valid_to"] is None or item["valid_to"] > start)
            ):
                continue
            payload = {
                key: item[key]
                for key in (
                    "name",
                    "category",
                    "venue",
                    "address",
                    "city",
                    "event_severity",
                    "event_intensity_score",
                    "venue_capacity",
                    "geocoding_quality",
                )
            }
            result.append(
                FutureRecord(
                    "Ticketon",
                    "events",
                    "1.0",
                    item["source_item_id"],
                    item["source_url"],
                    now,
                    None,
                    item["valid_from"],
                    item["valid_to"],
                    start,
                    horizon_hours,
                    item["latitude"],
                    item["longitude"],
                    event_type=item["event_type"],
                    severity=str(item["event_severity"]),
                    confidence=0.9 if item["latitude"] is not None else 0.7,
                    payload=payload,
                )
            )
        return result

    def build_features(
        self,
        records: list[FutureRecord],
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> dict[str, Any]:
        del horizon_hours
        scores = [int(r.payload["event_severity"]) for r in records]
        intensity = [float(r.payload["event_intensity_score"]) for r in records]
        starts = [r.valid_from for r in records if r.valid_from]
        stadiums = {"astana arena", "барыс арена", "barys arena"}
        return {
            "event_count_next_24h": len(records),
            "event_major_count_next_24h": sum(x >= 4 for x in scores),
            "event_stadium_count_next_24h": sum(
                (r.payload.get("venue") or "").lower() in stadiums for r in records
            ),
            "event_intensity_score": sum(intensity),
            "event_hours_until_nearest": min(
                ((x - prediction_datetime).total_seconds() / 3600 for x in starts),
                default=None,
            ),
            "event_geocoded_count": sum(r.latitude is not None for r in records),
        }

    def _degraded(self, warning: str) -> ProviderResult:
        return ProviderResult(
            self.metadata,
            [],
            [],
            {},
            {"city": "Astana", "records": 0},
            [warning],
            "degraded",
            True,
        )

    def collect(
        self,
        prediction_datetime: datetime,
        horizon_hours: int,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> ProviderResult:
        del latitude, longitude, bbox
        when = parse_prediction_datetime(prediction_datetime)
        if horizon_hours != 24:
            return self._degraded("ticketon_supports_24h_only")
        allowed = self._robots_allow_listing()
        if allowed is False:
            return self._degraded("ticketon_robots_disallow_listing")
        try:
            listing_html = self._get(self.listing_url)
        except requests.RequestException:
            return self._degraded("ticketon_request_failed")
        parsed = self.parse_listing(listing_html)
        detail_errors = 0
        # The listing contains cards but not their Event JSON-LD.  Detail pages
        # do, so follow a bounded set of public links with a polite delay.
        if not parsed:
            for url in self.discover_event_urls(listing_html)[: self.max_event_pages]:
                try:
                    parsed.extend(self.parse_listing(self._get(url)))
                except requests.RequestException:
                    detail_errors += 1
                self.sleep(self.request_delay)
            parsed = list({item["source_item_id"]: item for item in parsed}.values())
        records = self.normalize({"events": parsed}, when, horizon_hours)
        for record in records:
            apply_geocode(
                record,
                self.geocoder.event(
                    record.payload.get("venue"), record.payload.get("address")
                ),
            )
        warnings = ([] if parsed else ["ticketon_no_parseable_event_metadata"]) + (
            ["ticketon_robots_unavailable"] if allowed is None else []
        )
        if detail_errors:
            warnings.append(f"ticketon_detail_requests_failed:{detail_errors}")
        return ProviderResult(
            self.metadata,
            [{"listing_url": self.listing_url, "event_count": len(parsed)}],
            records,
            self.build_features(records, when, horizon_hours),
            {
                "city": "Astana",
                "listing_events": len(parsed),
                "records_in_window": len(records),
                "detail_pages_failed": detail_errors,
            },
            warnings,
            "ok" if parsed else "degraded",
            not bool(parsed),
        )
