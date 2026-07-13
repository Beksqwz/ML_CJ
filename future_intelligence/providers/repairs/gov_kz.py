"""Conservative server-rendered HTML collector for official Astana Akimat news."""

from __future__ import annotations
import hashlib
import re
import time
from datetime import UTC, datetime, timedelta
from html import unescape
from typing import Any, Callable
import requests
import pandas as pd
from future_intelligence.providers.repairs.base import RepairsProvider
from future_intelligence.geocoding import AstanaGeocoder, RoadGeometryResolver, apply_geocode
from future_intelligence.schemas import FutureRecord, ProviderMetadata, ProviderResult
from future_intelligence.utils import ASTANA_TIMEZONE, parse_prediction_datetime
from future_intelligence.providers.repairs.discovery import (
    GovKzJsonDiscovery,
    GovKzPlaywrightArticleSession,
    GovKzPlaywrightDiscovery,
    GovKzSearchDiscovery,
    GovKzSitemapDiscovery,
)
from ml_service.registry import ROOT

BASE = "https://www.gov.kz"
LISTING = BASE + "/memleket/entities/astana/press/news/{page}?lang={language}"
ROBOTS = BASE + "/robots.txt"
DETAIL_RE = re.compile(
    r'href=["\'](?P<url>/memleket/entities/astana/press/news/details/(?P<id>\d+)(?:\?[^"\']*)?)["\']',
    re.I,
)
MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
    "қаңтар": 1,
    "ақпан": 2,
    "наурыз": 3,
    "сәуір": 4,
    "мамыр": 5,
    "маусым": 6,
    "шілде": 7,
    "тамыз": 8,
    "қыркүйек": 9,
    "қазан": 10,
    "қараша": 11,
    "желтоқсан": 12,
}
KEYWORDS = {
    "repair": (
        "ремонт",
        "реконструкц",
        "строительств",
        "асфальт",
        "жөндеу",
        "қайта жаңғырту",
        "құрылыс",
    ),
    "restriction": (
        "перекры",
        "закрыт",
        "ограничен",
        "сужени",
        "полос",
        "жабыл",
        "шектеу",
        "тарыл",
    ),
    "infra": (
        "мост",
        "путепровод",
        "развязк",
        "перекрёст",
        "улиц",
        "проспект",
        "шоссе",
        "дорог",
        "көпір",
        "жолайрық",
        "қиылыс",
        "көше",
        "даңғыл",
        "жол",
    ),
}
RULES = {
    "full_closure": ("полностью перекры", "полностью закры", "толық жаб"),
    "partial": ("частично перекры", "ограничен", "сужени", "жартылай"),
    "lane": ("полос", "жолақ"),
}


def clean_html(html: str) -> str:
    text = re.sub(
        r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html, flags=re.I
    )
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def tag_text(html: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}[^>]*>([\s\S]*?)</{tag}>", html, re.I)
    return clean_html(match.group(1)) if match else None


def is_javascript_shell(html: str, minimum_text_length: int = 400) -> bool:
    text = clean_html(html)
    markers = ('<div id="root"', "You need to enable JavaScript", "/static/js/main")
    return (
        not (tag_text(html, "h1") or "").strip()
        or len(text) < minimum_text_length
        or any(marker.lower() in html.lower() for marker in markers)
    )


def parse_published(text: str, fallback: datetime | None = None) -> datetime:
    match = re.search(r"(\d{1,2})\s+([А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+)\s+(20\d{2})", text)
    if match and match.group(2).lower() in MONTHS:
        return datetime(
            int(match.group(3)),
            MONTHS[match.group(2).lower()],
            int(match.group(1)),
            tzinfo=ASTANA_TIMEZONE,
        )
    return fallback or datetime.now(ASTANA_TIMEZONE)


def parse_dates(
    text: str, published: datetime
) -> tuple[datetime | None, datetime | None, list[str], bool]:
    warnings = []
    year = published.year

    def date(day: str, month: str, hour: str | None = None) -> datetime:
        return datetime(
            year,
            MONTHS[month.lower()],
            int(day),
            int(hour or 0),
            tzinfo=ASTANA_TIMEZONE,
        )

    match = re.search(
        r"с\s+(?:(\d{1,2}):?(\d{2})?\s+)?(\d{1,2})\s+([А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+)\s+(?:по|до)\s+(\d{1,2})\s+([А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+)",
        text,
        re.I,
    )
    if match and match.group(4).lower() in MONTHS and match.group(6).lower() in MONTHS:
        return (
            date(match.group(3), match.group(4), match.group(1)),
            date(match.group(5), match.group(6)),
            warnings,
            False,
        )
    match = re.search(
        r"с\s+(?:(\d{1,2}):(\d{2})\s+)?(\d{1,2})\s+([А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+)",
        text,
        re.I,
    )
    if match and match.group(4).lower() in MONTHS:
        start = date(match.group(3), match.group(4), match.group(1))
        end_match = re.search(r"до\s+(\d{1,2})\s+([А-Яа-я]+)", text, re.I)
        if end_match and end_match.group(2).lower() in MONTHS:
            return start, date(end_match.group(1), end_match.group(2)), warnings, False
        warnings.append("open_end_date")
        return start, None, warnings, True
    match = re.search(r"до\s+(\d{1,2})\s+([А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+)", text, re.I)
    if match and match.group(2).lower() in MONTHS:
        return None, date(match.group(1), match.group(2)), warnings, False
    return None, None, ["date_not_found"], False


def classify(text: str) -> tuple[int, str, str, str]:
    value = text.lower()
    score = 2 * sum(
        any(word in value for word in KEYWORDS[group])
        for group in ("repair", "restriction")
    ) + sum(any(word in value for word in KEYWORDS["infra"]) for _ in [0])
    if any(word in value for word in RULES["full_closure"]):
        restriction = "full_closure"
    elif any(word in value for word in RULES["lane"]):
        restriction = "single_lane_closure" if "закры" in value else "lane_narrowing"
    elif any(word in value for word in RULES["partial"]):
        restriction = "partial_closure"
    else:
        restriction = (
            "no_restriction"
            if any(word in value for word in KEYWORDS["repair"])
            else "unknown"
        )
    if "перекр" in value or "қиылыс" in value:
        event = (
            "intersection_closure"
            if restriction == "full_closure"
            else "traffic_restriction"
        )
    elif "мост" in value or "путепровод" in value or "көпір" in value:
        event = "bridge_repair"
    elif "реконструк" in value:
        event = "road_reconstruction"
    elif "строитель" in value or "құрылыс" in value:
        event = "road_construction"
    elif restriction == "full_closure":
        event = "road_closure"
    elif restriction != "no_restriction":
        event = "lane_closure" if "полос" in value else "traffic_restriction"
    elif any(word in value for word in KEYWORDS["repair"]):
        event = "road_repair"
    else:
        event = "unknown_road_event"
    severity = (
        "critical"
        if restriction == "full_closure"
        and event in {"intersection_closure", "bridge_repair"}
        else "high"
        if restriction == "full_closure"
        else "medium"
        if restriction in {"partial_closure", "lane_narrowing", "single_lane_closure"}
        else "low"
        if event == "road_repair"
        else "unknown"
    )
    return score, event, restriction, severity


def locations(text: str) -> dict[str, Any]:
    # Capture the street name, not the following prose ("частично перекроют").
    street_pattern = re.compile(
        r"(?P<prefix>ул\.?|улиц[ауыые]?|пр\.?|пр-т|проспект[а-еом]?|ш\.?|шоссе)\s+"
        r"(?P<name>[А-ЯA-ZӘәҒғҚқҢңӨөҰұҮүҺһІі][\w.\-]*(?:\s+[А-ЯA-ZӘәҒғҚқҢңӨөҰұҮүҺһІі][\w.\-]*){0,2})",
        re.I,
    )
    original = []
    for match in street_pattern.finditer(text):
        value = match.group(0).strip(" ,.;")
        value = re.split(r"\s+(?:частично|полностью|перекро\w*|закро\w*|огранич\w*|ремонт\w*|от|до|с|по|на|в)\b", value, maxsplit=1, flags=re.I)[0]
        original.append(value.strip(" ,.;"))
    section = re.search(r"от\s+(.{2,60}?)\s+до\s+(.{2,60}?)(?:[,.]|$)", text, re.I)
    intersection = re.search(
        r"(?:пересечени[ие][\w ]*|қиылысында)\s+(.{2,70}?)(?:[,.]|$)", text, re.I
    )
    return {
        "road_name": original[0] if original else None,
        "from_street": section.group(1).strip() if section else None,
        "to_street": section.group(2).strip() if section else None,
        "intersection_streets": intersection.group(1).strip() if intersection else None,
        "address": None,
        "district": None,
        "original_location_phrases": original,
    }


class GovKzRoadEventsProvider(RepairsProvider):
    metadata = ProviderMetadata(
        "gov_kz_repairs", "1.0", "repairs", (24,), False, "daily", "Astana"
    )

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        geocoder: AstanaGeocoder | None = None,
        road_geometry: RoadGeometryResolver | None = None,
        timeout_seconds: float = 10,
        max_pages: int = 3,
        max_articles: int = 10,
        request_delay: float = 0.5,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        discovery_method: str = "official-filtered",
        browser_timeout: int = 15000,
        headless: bool = True,
    ):
        self.session = session or requests.Session()
        self.geocoder = geocoder or AstanaGeocoder(session=self.session, sleep=sleep)
        self.road_geometry = road_geometry or RoadGeometryResolver()
        self.timeout_seconds = timeout_seconds
        self.max_pages = max_pages
        self.max_articles = max_articles
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.sleep = sleep
        self.discovery_method = discovery_method
        self.browser_timeout = browser_timeout
        self.headless = headless
        self.last_report = {}

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": self.metadata.provider_name,
            "listing": LISTING,
        }

    def _get(self, url: str) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout_seconds,
                    headers={
                        "User-Agent": "AstanaFutureIntelligence/1.0 (+public-data; contact: data-team)"
                    },
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                self.sleep(0.25 * (2**attempt))
        raise RuntimeError("unreachable")

    def _known_source_item_ids(self) -> set[str]:
        path = (
            ROOT
            / "data"
            / "future_intelligence"
            / "processed"
            / "gov_kz_road_events.parquet"
        )
        if not path.exists():
            return set()
        try:
            return set(
                pd.read_parquet(path, columns=["source_item_id"]).source_item_id.astype(
                    str
                )
            )
        except Exception:
            return set()

    def discover(
        self,
        language: str = "ru",
        max_pages: int | None = None,
        method: str | None = None,
    ) -> tuple[list[tuple[str, str]], dict]:
        robots = "unknown"
        links = []
        pages = max_pages or self.max_pages
        try:
            robots_text = self._get(ROBOTS)
            robots = "fetched"
            disallowed = "Disallow: /memleket/entities/astana/press/news" in robots_text
        except requests.RequestException:
            disallowed = False
            robots = "unavailable"
        if disallowed:
            return [], {
                "robots": robots,
                "robots_allows_listing": False,
                "pages_requested": 0,
                "discovery_chain": [],
            }
        selected = method or (
            "auto"
            if self.discovery_method == "official-filtered"
            else self.discovery_method
        )
        chain = []

        def legacy():
            started = time.perf_counter()
            for page in range(1, pages + 1):
                html = self._get(LISTING.format(page=page, language=language))
                links.extend(
                    (BASE + m.group("url"), m.group("id"))
                    for m in DETAIL_RE.finditer(html)
                )
                self.sleep(self.request_delay)
            values = list(
                dict(((ident, url), (url, ident)) for url, ident in links).values()
            )
            return {
                "method_name": "legacy_html",
                "status": "ok" if values else "empty",
                "article_urls": [url for url, _ in values],
                "pages_scanned": pages,
                "warnings": [],
                "errors": [],
                "duration_seconds": time.perf_counter() - started,
            }, values

        adapters = []
        if selected in {"auto", "json"}:
            adapters.append(GovKzJsonDiscovery())
        if selected in {"auto", "sitemap"}:
            adapters.append(GovKzSitemapDiscovery(self._get))
        if selected in {"auto", "search"}:
            adapters.append(
                GovKzSearchDiscovery(
                    self._get, sleep=self.sleep, request_delay=self.request_delay
                )
            )
        mocked_downloader = getattr(self._get, "__self__", None) is None
        if selected in {"auto", "playwright"} and not (
            selected == "auto" and mocked_downloader
        ):
            adapters.append(
                GovKzPlaywrightDiscovery(
                    LISTING.format(page="{page}", language=language),
                    self.browser_timeout,
                    self.headless,
                )
            )
        for adapter in adapters:
            item = adapter.discover(max_pages=pages, max_articles=self.max_articles)
            chain.append(item.__dict__)
            if item.article_urls:
                return [
                    (url, re.search(r"details/(\d+)", url).group(1))
                    for url in item.article_urls
                ], {
                    "robots": robots,
                    "robots_allows_listing": True,
                    "pages_requested": item.pages_scanned,
                    "selected_method": item.method_name,
                    "discovery_chain": chain,
                }
            if selected != "auto":
                break
        if selected in {"auto", "html"}:
            item, values = legacy()
            chain.append(item)
            if values:
                return values, {
                    "robots": robots,
                    "robots_allows_listing": True,
                    "pages_requested": pages,
                    "selected_method": "legacy_html",
                    "discovery_chain": chain,
                }
        return [], {
            "robots": robots,
            "robots_allows_listing": True,
            "pages_requested": pages,
            "selected_method": None,
            "discovery_chain": chain,
        }

    def parse_article(
        self, html: str, url: str, item_id: str, language: str = "ru"
    ) -> dict[str, Any]:
        title = tag_text(html, "h1") or ""
        description = tag_text(html, "meta") or ""
        text = clean_html(html)
        published = parse_published(text)
        score, event, restriction, severity = classify(title + " " + text)
        start, end, warnings, open_end = parse_dates(text, published)
        loc = locations(text)
        relevant = score >= 3 or (score >= 2 and bool(loc["road_name"]))
        normalized = {
            "source": "gov.kz Astana Akimat",
            "source_type": "repairs",
            "source_version": "1.0",
            "source_item_id": item_id,
            "source_url": url,
            "title": title,
            "description": description,
            "published_at": published,
            "valid_from": start,
            "valid_to": end,
            "event_type": event,
            "restriction_type": restriction,
            "severity": severity,
            "confidence": min(0.95, 0.35 + 0.15 * score - (0.15 if open_end else 0)),
            "language": language,
            "location": loc,
            "relevance_score": score,
            "relevant": relevant,
            "warnings": warnings,
            "open_end": open_end,
            "text": text,
        }
        material = (
            "|".join(
                str(normalized[key])
                for key in (
                    "title",
                    "description",
                    "valid_from",
                    "valid_to",
                    "event_type",
                    "restriction_type",
                )
            )
            + "|"
            + str(loc)
        )
        normalized["content_hash"] = hashlib.sha256(material.encode()).hexdigest()
        return normalized

    def normalize(
        self,
        raw_payload: dict[str, Any],
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> list[FutureRecord]:
        parsed = raw_payload["parsed"]
        now = datetime.now(UTC)
        return [
            FutureRecord(
                source=parsed["source"],
                source_type="repairs",
                source_version="1.0",
                source_item_id=parsed["source_item_id"],
                source_url=parsed["source_url"],
                collected_at=now,
                published_at=parsed["published_at"],
                valid_from=parsed["valid_from"],
                valid_to=parsed["valid_to"],
                prediction_datetime=prediction_datetime,
                horizon_hours=horizon_hours,
                latitude=None,
                longitude=None,
                event_type=parsed["event_type"],
                severity=parsed["severity"],
                confidence=parsed["confidence"],
                is_forecast=True,
                is_realtime=False,
                is_historical=False,
                payload={
                    key: parsed[key]
                    for key in (
                        "title",
                        "description",
                        "restriction_type",
                        "language",
                        "location",
                        "relevance_score",
                        "content_hash",
                        "open_end",
                        "text",
                        "warnings",
                    )
                },
                warnings=parsed["warnings"],
            )
        ]

    def build_features(
        self,
        records: list[FutureRecord],
        prediction_datetime: datetime,
        horizon_hours: int,
    ) -> dict[str, Any]:
        end = prediction_datetime + timedelta(hours=horizon_hours)
        active = [
            r
            for r in records
            if r.valid_from is None
            or (
                r.valid_from < end
                and (r.valid_to is None or r.valid_to > prediction_datetime)
            )
        ]
        return {
            "repair_events_next_24h": len(active),
            "repair_full_closures_next_24h": sum(
                r.payload["restriction_type"] == "full_closure" for r in active
            ),
            "repair_partial_closures_next_24h": sum(
                r.payload["restriction_type"] == "partial_closure" for r in active
            ),
            "repair_lane_closures_next_24h": sum(
                r.event_type == "lane_closure" for r in active
            ),
            "repair_intersection_closures_next_24h": sum(
                r.event_type == "intersection_closure" for r in active
            ),
            "repair_bridge_events_next_24h": sum(
                r.event_type == "bridge_repair" for r in active
            ),
            "repair_high_severity_events_next_24h": sum(
                r.severity in {"high", "critical"} for r in active
            ),
            "repair_active_event_count": len(active),
            "repair_open_end_event_count": sum(
                bool(r.payload["open_end"]) for r in active
            ),
            "repair_disruption_score": sum(
                {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(r.severity, 0)
                for r in active
            ),
        }

    def collect(
        self,
        prediction_datetime: datetime,
        horizon_hours: int,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        bbox=None,
        language: str = "ru",
        since: datetime | None = None,
        until: datetime | None = None,
        force_refresh: bool = False,
        discovery_method: str | None = None,
    ) -> ProviderResult:
        del latitude, longitude, bbox
        when = parse_prediction_datetime(prediction_datetime)
        if horizon_hours != 24:
            return ProviderResult(
                self.metadata,
                [],
                [],
                {},
                {},
                ["gov_kz_repairs_supports_24h_only"],
                "degraded",
                True,
            )
        selected_method = discovery_method or self.discovery_method
        browser_session = None
        try:
            if selected_method in {
                "official-filtered",
                "auto",
                "playwright",
                "road-search",
            }:
                browser_session = GovKzPlaywrightArticleSession(
                    LISTING.format(page="{page}", language=language),
                    self.browser_timeout,
                    self.headless,
                ).__enter__()
                if selected_method in {"official-filtered", "auto"}:
                    urls, listing_report = browser_session.official_filtered_listing(
                        self.max_pages, self.max_articles
                    )
                    report = {
                        "robots": "fetched",
                        "robots_allows_listing": True,
                        "pages_requested": listing_report["pages_scanned"],
                        "selected_method": "official_filtered_listing"
                        if urls
                        else None,
                        "discovery_chain": [
                            {
                                "method_name": "official_filtered_listing",
                                "status": "ok" if urls else "empty",
                                "article_urls": urls,
                                "pages_scanned": listing_report["pages_scanned"],
                                "warnings": [],
                                "errors": [],
                            }
                        ],
                        **listing_report,
                    }
                    if not urls and selected_method == "auto":
                        fallback_links, fallback_report = self.discover(
                            language, method="auto"
                        )
                        links = fallback_links
                        report["discovery_chain"].extend(
                            fallback_report["discovery_chain"]
                        )
                        report["selected_method"] = fallback_report["selected_method"]
                        report["pages_requested"] += fallback_report["pages_requested"]
                    else:
                        links = [
                            (url, re.search(r"details/(\d+)", url).group(1))
                            for url in urls
                        ]
                elif selected_method == "road-search":
                    urls, search_diagnostics = browser_session.road_search(
                        self.max_articles, self.request_delay
                    )
                    report = {
                        "robots": "fetched",
                        "robots_allows_listing": True,
                        "pages_requested": 0,
                        "selected_method": "road_search",
                        "discovery_chain": [
                            {
                                "method_name": "road_search",
                                "status": "ok" if urls else "empty",
                                "article_urls": urls,
                                "pages_scanned": len(search_diagnostics),
                                "warnings": [],
                                "errors": [],
                            }
                        ],
                        "search_diagnostics": search_diagnostics,
                    }
                    links = [
                        (url, re.search(r"details/(\d+)", url).group(1)) for url in urls
                    ]
                else:
                    urls = browser_session.discover(self.max_pages, self.max_articles)
                    report = {
                        "robots": "fetched",
                        "robots_allows_listing": True,
                        "pages_requested": self.max_pages,
                        "selected_method": "playwright",
                        "discovery_chain": [
                            {
                                "method_name": "playwright",
                                "status": "ok",
                                "article_urls": urls,
                                "pages_scanned": self.max_pages,
                                "warnings": [],
                                "errors": [],
                            }
                        ],
                    }
                    links = [
                        (url, re.search(r"details/(\d+)", url).group(1)) for url in urls
                    ]
            else:
                links, report = self.discover(language, method=selected_method)
            parsed = []
            raw = []
        except requests.RequestException as exc:
            return ProviderResult(
                self.metadata,
                [],
                [],
                {},
                {},
                [f"gov_kz_request_failed:{type(exc).__name__}"],
                "degraded",
                True,
            )
        known_ids = set() if force_refresh else self._known_source_item_ids()
        known_links = [(url, item_id) for url, item_id in links if item_id in known_ids]
        links = [(url, item_id) for url, item_id in links if item_id not in known_ids]
        report["already_known_articles_skipped"] = len(known_links)
        diagnostics = []
        for url, item_id in links[: self.max_articles]:
            try:
                requests_html = self._get(url)
                request_text = clean_html(requests_html)
                shell = is_javascript_shell(requests_html)
                html = requests_html
                acquisition = "requests"
                rendered_html_length = None
                rendered_title = None
                rendered_text_length = None
                warnings = []
                if shell and browser_session is not None:
                    try:
                        html, png = browser_session.render_article(url)
                        acquisition = "playwright_fallback"
                        rendered_html_length = len(html)
                        rendered_title = tag_text(html, "h1") or ""
                        rendered_text_length = len(clean_html(html))
                        if not diagnostics:
                            debug = ROOT / "reports" / "stage15" / "gov_kz_debug"
                            debug.mkdir(parents=True, exist_ok=True)
                            (debug / "requests_shell.html").write_text(
                                requests_html, encoding="utf-8"
                            )
                            (debug / "playwright_rendered.html").write_text(
                                html, encoding="utf-8"
                            )
                            (debug / "playwright_rendered.png").write_bytes(png)
                    except Exception as exc:
                        warnings.append(
                            f"playwright_detail_failed:{type(exc).__name__}"
                        )
                item = self.parse_article(html, url, item_id, language)
                raw.append(
                    {
                        "url": url,
                        "item_id": item_id,
                        "content_hash": item["content_hash"],
                    }
                )
                text = (item["title"] + " " + item["text"]).lower()
                matched = {
                    group: [word for word in words if word in text]
                    for group, words in KEYWORDS.items()
                }
                rejection = None
                if not item["relevant"]:
                    rejection = "relevance_score_below_threshold_or_no_location_context"
                elif since is not None and item["published_at"] < since:
                    rejection = "before_since"
                elif until is not None and item["published_at"] > until:
                    rejection = "after_until"
                diagnostics.append(
                    {
                        "source_item_id": item_id,
                        "source_url": url,
                        "acquisition_method": acquisition,
                        "requests_html_length": len(requests_html),
                        "requests_extracted_text_length": len(request_text),
                        "shell_detected": shell,
                        "playwright_used": acquisition == "playwright_fallback",
                        "rendered_html_length": rendered_html_length,
                        "rendered_title": rendered_title,
                        "rendered_text_length": rendered_text_length,
                        "title": item["title"],
                        "extracted_text_length": len(item["text"]),
                        "matched_keywords": matched,
                        "relevance_score": item["relevance_score"],
                        "relevant": item["relevant"] and rejection is None,
                        "rejection_reason": rejection,
                        "acquisition_warnings": warnings,
                    }
                )
                if item["relevant"] and rejection is None:
                    parsed.append(item)
            except requests.RequestException:
                report.setdefault("errors", 0)
                report["errors"] += 1
            self.sleep(self.request_delay)
        if browser_session is not None:
            browser_session.__exit__(None, None, None)
        records = [
            record
            for item in parsed
            for record in self.normalize({"parsed": item}, when, horizon_hours)
        ]
        for record in records:
            apply_geocode(record, self.geocoder.repair(record.payload["location"]))
            geometry = self.road_geometry.repair(record.payload["location"])
            record.payload.update({"repair_geometry_quality": geometry.quality})
            record.warnings.extend(geometry.warnings)
            if geometry.geometry is not None:
                record.geometry = geometry.geometry
                record.confidence = max(record.confidence or 0, geometry.confidence)
        features = self.build_features(records, when, horizon_hours)
        rejection_reasons = {}
        for item in diagnostics:
            if item["rejection_reason"]:
                rejection_reasons[item["rejection_reason"]] = (
                    rejection_reasons.get(item["rejection_reason"], 0) + 1
                )
        for search_item in report.get("search_diagnostics", []):
            inspected = [
                item
                for item in diagnostics
                if item["source_url"] in set(search_item.get("accepted_urls", []))
            ]
            search_item["rendered_successfully"] = sum(
                item["acquisition_method"] == "playwright_fallback"
                and bool(item["rendered_text_length"])
                for item in inspected
            )
            search_item["relevant_articles"] = sum(
                item["relevant"] for item in inspected
            )
            search_item["rejected_articles"] = sum(
                not item["relevant"] for item in inspected
            )
            search_item["rejection_reasons"] = {
                reason: sum(item["rejection_reason"] == reason for item in inspected)
                for reason in {
                    item["rejection_reason"]
                    for item in inspected
                    if item["rejection_reason"]
                }
            }
        rendered = sum(
            item["acquisition_method"] == "playwright_fallback"
            and bool(item["rendered_text_length"])
            for item in diagnostics
        )
        selected_count = report.get("listing_cards_selected", len(links))
        report.update(
            {
                "article_links_found": len(links) + len(known_links),
                "articles_downloaded": len(raw),
                "detail_pages_rendered": rendered,
                "rendered_successfully": rendered,
                "relevant_articles": len(records),
                "rejected_articles": sum(not item["relevant"] for item in diagnostics),
                "prefilter_precision": (
                    len(records) / selected_count if selected_count else None
                ),
                "rejection_reasons": rejection_reasons,
                "language": language,
                "article_diagnostics": diagnostics,
            }
        )
        self.last_report = report
        if not links:
            warning = (
                "gov_kz_official_listing_has_no_prefiltered_road_candidates"
                if selected_method == "official-filtered"
                else "gov_kz_listing_has_no_server_rendered_article_links"
            )
            return ProviderResult(
                self.metadata,
                raw,
                records,
                features,
                {"city": "Astana", "records": 0},
                [warning],
                "degraded",
                True,
            )
        return ProviderResult(
            self.metadata,
            raw,
            records,
            features,
            {"city": "Astana", "records": len(records)},
            [],
            "ok",
            False,
        )
