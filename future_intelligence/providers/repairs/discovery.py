"""Ordered, safe discovery adapters for official gov.kz article URLs."""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from xml.etree import ElementTree

import requests


BASE = "https://www.gov.kz"
SITEMAP = BASE + "/sitemap.xml"
PATH = re.compile(r"^/memleket/entities/astana/press/news/details/\d+$")


@dataclass
class DiscoveryResult:
    method_name: str
    status: str
    article_urls: list[str] = field(default_factory=list)
    pages_scanned: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


def valid(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "www.gov.kz"
        and bool(PATH.match(parsed.path))
    )


def unique(urls: list[str], max_articles: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if valid(url) and url not in seen:
            output.append(url)
            seen.add(url)
        if len(output) >= max_articles:
            break
    return output


# Listing-card terms are deliberately lighter than the full article parser.
# A generic word such as "ремонт" alone is not enough: it needs road context.
PREFILTER_STRONG = (
    "дорожный ремонт",
    "средний ремонт",
    "капитальный ремонт",
    "реконструкция",
    "асфальт",
    "дорожно-строительные работы",
    "строительство дороги",
    "перекрытие",
    "перекроют",
    "ограничение движения",
    "закрытие движения",
    "частичное ограничение",
    "сужение",
    "закрытие полосы",
    "изменение схемы движения",
    "жол жөндеу",
    "жолды жөндеу",
    "жол жабылады",
    "қозғалыс шектеледі",
    "қозғалысқа шектеу",
    "жол құрылысы",
    "жолдарды қайта жаңғырту",
)
PREFILTER_WEAK = ("ремонт", "строительство", "жөндеу", "құрылыс")
PREFILTER_ROAD_CONTEXT = (
    "улица",
    "ул.",
    "проспект",
    "пр-т",
    "шоссе",
    "дорога",
    "перекрёсток",
    "перекресток",
    "развязка",
    "мост",
    "путепровод",
    "проезжая часть",
    "транспорт",
    "көше",
    "даңғыл",
    "жол",
    "қиылыс",
    "көпір",
    "жолайрық",
    "көлік",
)


def normalize_card_text(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", normalized).strip().lower()


def prefilter_listing_card(title: str, snippet: str = "") -> dict:
    """Return transparent, non-final candidate decision for a rendered card."""
    text = normalize_card_text(f"{title} {snippet}")
    strong = [term for term in PREFILTER_STRONG if term in text]
    weak = [term for term in PREFILTER_WEAK if term in text]
    road = [term for term in PREFILTER_ROAD_CONTEXT if term in text]
    selected = bool(strong) or bool(weak and road)
    return {
        "prefilter_matched_keywords": {
            "strong": strong,
            "weak": weak,
            "road_context": road,
        },
        "prefilter_score": len(strong) * 2 + len(weak) + len(road),
        "candidate_selected": selected,
        "rejection_reason": (
            None
            if selected
            else "no_strong_road_event_term_or_weak_term_with_road_context"
        ),
    }


class GovKzJsonDiscovery:
    """No endpoint is used until confirmed from official browser activity."""

    def discover(self, **kwargs) -> DiscoveryResult:
        return DiscoveryResult(
            "json_api",
            "skipped",
            warnings=["no_verified_public_gov_kz_xhr_endpoint_discovered"],
        )


class GovKzSitemapDiscovery:
    def __init__(self, get: Callable[[str], str], sitemap_url: str = SITEMAP):
        self.get = get
        self.sitemap_url = sitemap_url

    def discover(self, *, max_articles: int, **kwargs) -> DiscoveryResult:
        started = time.perf_counter()
        try:
            root = ElementTree.fromstring(self.get(self.sitemap_url))
            urls = [
                element.text
                for element in root.iter(
                    "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                )
                if element.text
            ]
            found = unique(urls, max_articles)
            return DiscoveryResult(
                "sitemap_or_feed",
                "ok" if found else "empty",
                found,
                1,
                ["sitemap_contains_no_astana_detail_article_urls"] if not found else [],
                duration_seconds=time.perf_counter() - started,
            )
        except (requests.RequestException, ElementTree.ParseError) as exc:
            return DiscoveryResult(
                "sitemap_or_feed",
                "failed",
                errors=[type(exc).__name__],
                duration_seconds=time.perf_counter() - started,
            )


class GovKzSearchDiscovery:
    """Discover only official detail URLs from public search-result pages."""

    QUERIES_RU = (
        "ремонт дороги",
        "перекрытие дороги",
        "ограничение движения",
        "реконструкция",
    )
    QUERIES_KK = ("жол жөндеу", "жол жабылуы", "қозғалыс шектеуі", "қайта жаңғырту")

    def __init__(
        self,
        get: Callable[[str], str],
        search_url: str = "https://www.bing.com/search?q=",
        sleep: Callable[[float], None] = time.sleep,
        request_delay: float = 0.5,
    ):
        self.get = get
        self.search_url = search_url
        self.sleep = sleep
        self.request_delay = request_delay

    def discover(
        self, *, max_articles: int, language: str = "ru", **kwargs
    ) -> DiscoveryResult:
        started = time.perf_counter()
        found: list[str] = []
        errors: list[str] = []
        queries = self.QUERIES_KK if language == "kk" else self.QUERIES_RU
        for phrase in queries:
            query = f"site:gov.kz/memleket/entities/astana/press/news {phrase}"
            try:
                html = unquote(self.get(self.search_url + quote_plus(query)))
                found.extend(
                    re.findall(
                        r"https?://www\.gov\.kz/memleket/entities/astana/press/news/details/\d+(?:\?[^\"'<>\s&]+)?",
                        html,
                        re.I,
                    )
                )
            except requests.RequestException as exc:
                errors.append(type(exc).__name__)
            if phrase != queries[-1]:
                self.sleep(self.request_delay)
        urls = unique(found, max_articles)
        return DiscoveryResult(
            "search_discovery",
            "ok" if urls else ("failed" if errors else "empty"),
            urls,
            len(queries),
            errors=errors,
            warnings=[
                "search_results_are_discovery_only; detail_urls_are_still_validated_as_official"
            ]
            if urls
            else [],
            duration_seconds=time.perf_counter() - started,
        )


class GovKzPlaywrightDiscovery:
    def __init__(
        self, listing_url: str, timeout_ms: int = 15000, headless: bool = True
    ):
        self.listing_url = listing_url
        self.timeout_ms = timeout_ms
        self.headless = headless

    def discover(
        self, *, max_pages: int, max_articles: int, **kwargs
    ) -> DiscoveryResult:
        started = time.perf_counter()
        try:
            return asyncio.run(self._run(max_pages, max_articles, started))
        except ModuleNotFoundError:
            return DiscoveryResult(
                "playwright",
                "unavailable",
                warnings=["playwright_not_installed"],
                duration_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            return DiscoveryResult(
                "playwright",
                "failed",
                errors=[f"{type(exc).__name__}:{exc}"],
                duration_seconds=time.perf_counter() - started,
            )

    async def _run(
        self, max_pages: int, max_articles: int, started: float
    ) -> DiscoveryResult:
        from playwright.async_api import async_playwright

        urls: list[str] = []
        async with async_playwright() as api:
            browser = await api.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                user_agent="AstanaFutureIntelligence/1.0 (+public-data; contact: data-team)"
            )
            page = await context.new_page()
            await page.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.resource_type in {"image", "font", "media"}
                    else route.continue_()
                ),
            )
            for number in range(1, max_pages + 1):
                await page.goto(
                    self.listing_url.format(page=number),
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                )
                await page.wait_for_selector(
                    "a[href*='/memleket/entities/astana/press/news/details/']",
                    timeout=self.timeout_ms,
                )
                urls.extend(
                    await page.locator(
                        "a[href*='/memleket/entities/astana/press/news/details/']"
                    ).evaluate_all("links => links.map(a => a.href)")
                )
                if len(unique(urls, max_articles)) >= max_articles:
                    break
            await context.close()
            await browser.close()
        found = unique(urls, max_articles)
        return DiscoveryResult(
            "playwright",
            "ok" if found else "empty",
            found,
            max_pages,
            duration_seconds=time.perf_counter() - started,
        )


class GovKzPlaywrightArticleSession:
    """One reusable Chromium browser/context for discovery and detail fallback."""

    def __init__(
        self, listing_url: str, timeout_ms: int = 15000, headless: bool = True
    ):
        self.listing_url = listing_url
        self.timeout_ms = timeout_ms
        self.headless = headless
        self._api = self._browser = self._context = self._page = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._api = sync_playwright().start()
        self._browser = self._api.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent="AstanaFutureIntelligence/1.0 (+public-data; contact: data-team)"
        )
        self._context.route(
            "**/*",
            lambda route: (
                route.abort()
                if route.request.resource_type in {"image", "font", "media"}
                else route.continue_()
            ),
        )
        self._page = self._context.new_page()
        return self

    def __exit__(self, *args):
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._api:
            self._api.stop()

    def discover(self, max_pages: int, max_articles: int):
        urls: list[str] = []
        for number in range(1, max_pages + 1):
            self._page.goto(
                self.listing_url.format(page=number),
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )
            self._page.wait_for_selector(
                "a[href*='/memleket/entities/astana/press/news/details/']",
                timeout=self.timeout_ms,
            )
            urls.extend(
                self._page.locator(
                    "a[href*='/memleket/entities/astana/press/news/details/']"
                ).evaluate_all("links => links.map(a => a.href)")
            )
            if len(unique(urls, max_articles)) >= max_articles:
                break
        return unique(urls, max_articles)

    def official_filtered_listing(self, max_pages: int, max_articles: int):
        """Read rendered official listing cards and keep only likely road events."""
        selected: list[str] = []
        diagnostics: list[dict] = []
        seen_ids: set[str] = set()
        seen_urls: set[str] = set()
        duplicates = 0
        pages_scanned = 0
        card_selector = ".card-news-list__item"
        for number in range(1, max_pages + 1):
            self._page.goto(
                self.listing_url.format(page=number),
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )
            try:
                self._page.wait_for_selector(
                    "a[href*='/memleket/entities/astana/press/news/details/']",
                    timeout=self.timeout_ms,
                )
            except Exception:
                break
            cards = self._page.locator(card_selector)
            if not cards.count():
                cards = self._page.locator(
                    "a[href*='/memleket/entities/astana/press/news/details/']"
                )
            rendered = cards.evaluate_all("""cards => cards.map(card => {
                const link=card.matches('a[href*="/details/"]') ? card : card.querySelector('a[href*="/details/"]');
                if (!link) return null;
                const title=(link.getAttribute('title') || link.innerText || '').trim();
                const all=(card.innerText || '').trim();
                const date=((card.querySelector('time, .categories__item.color_grey') || {}).innerText || '').trim();
                return {url: link.href, title, snippet: all.replace(title, '').trim(), visible_date: date};
            }).filter(Boolean)""")
            if not rendered:
                break
            pages_scanned += 1
            new_ids = 0
            for card in rendered:
                url = card["url"]
                match = re.search(r"details/(\d+)", url)
                if not match or not valid(url):
                    continue
                item_id = match.group(1)
                base = {
                    "page_number": number,
                    "source_item_id": item_id,
                    "source_url": url,
                    "card_title": card["title"],
                    "card_snippet": card["snippet"],
                    "visible_published_at": card["visible_date"],
                }
                if item_id in seen_ids or url in seen_urls:
                    diagnostics.append(
                        base
                        | {
                            "prefilter_matched_keywords": {},
                            "prefilter_score": 0,
                            "candidate_selected": False,
                            "rejection_reason": "duplicate_listing_card",
                        }
                    )
                    duplicates += 1
                    continue
                seen_ids.add(item_id)
                seen_urls.add(url)
                new_ids += 1
                decision = prefilter_listing_card(card["title"], card["snippet"])
                diagnostics.append(base | decision)
                if decision["candidate_selected"] and len(selected) < max_articles:
                    selected.append(url)
            if not new_ids or len(selected) >= max_articles:
                break
        return unique(selected, max_articles), {
            "pages_scanned": pages_scanned,
            "listing_cards_seen": len(diagnostics),
            "listing_cards_selected": len(selected),
            "duplicate_urls_removed": duplicates,
            "listing_card_diagnostics": diagnostics,
        }

    def road_search(self, max_articles: int, request_delay: float = 0.5):
        """Search narrowly for road announcements; never enumerate the news listing."""
        queries = (
            "site:gov.kz/memleket/entities/astana/press/news ремонт дороги Астана",
            "site:gov.kz/memleket/entities/astana/press/news перекрытие дороги Астана",
            "site:gov.kz/memleket/entities/astana/press/news ограничение движения Астана",
            "site:gov.kz/memleket/entities/astana/press/news дорожные работы Астана",
            "site:gov.kz/memleket/entities/astana/press/news транспортная развязка Астана",
            "site:gov.kz/memleket/entities/astana/press/news ремонт моста Астана",
        )
        accepted: list[str] = []
        diagnostics: list[dict] = []
        seen: set[str] = set()
        for index, query in enumerate(queries):
            warning = None
            try:
                self._page.goto(
                    "https://search.brave.com/search?q=" + quote_plus(query),
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                )
                try:
                    self._page.wait_for_selector(
                        "a[href*='gov.kz/memleket/entities/astana/press/news/details/']",
                        timeout=min(self.timeout_ms, 5000),
                    )
                except Exception:
                    warning = "search_result_links_not_rendered"
                hrefs = self._page.locator("a").evaluate_all(
                    "links => links.map(a => a.href)"
                )
                hrefs.extend(
                    re.findall(
                        r"https?://www\.gov\.kz/memleket/entities/astana/press/news/details/\d+(?:\?[^\"'< >]+)?",
                        unquote(self._page.content()),
                        re.I,
                    )
                )
                found: list[str] = []
                for href in hrefs:
                    parsed = urlparse(href)
                    candidate = (
                        parse_qs(parsed.query).get("q", [href])[0]
                        if parsed.netloc.endswith("google.com")
                        and parsed.path == "/url"
                        else href
                    )
                    if valid(candidate):
                        found.append(candidate)
                query_urls = unique(found, max_articles)
                new_urls = [url for url in query_urls if url not in seen]
                duplicates = (
                    len(found) - len(query_urls) + (len(query_urls) - len(new_urls))
                )
                accepted.extend(new_urls)
                seen.update(query_urls)
                diagnostics.append(
                    {
                        "search_query": query,
                        "urls_found": len(found),
                        "urls_accepted": len(new_urls),
                        "duplicates_removed": duplicates,
                        "found_urls": query_urls,
                        "accepted_urls": new_urls,
                        "warning": warning,
                    }
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "search_query": query,
                        "urls_found": 0,
                        "urls_accepted": 0,
                        "duplicates_removed": 0,
                        "found_urls": [],
                        "accepted_urls": [],
                        "warning": f"search_failed:{type(exc).__name__}",
                    }
                )
            if index < len(queries) - 1:
                time.sleep(request_delay)
        return unique(accepted, max_articles), diagnostics

    def render_article(self, url: str):
        self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        selectors = ("article h1", "h1", "[itemprop='headline']")
        last = None
        for selector in selectors:
            try:
                self._page.wait_for_selector(selector, timeout=self.timeout_ms)
                last = selector
                break
            except Exception:
                pass
        if last is None:
            raise TimeoutError("rendered_article_title_not_found")
        return self._page.content(), self._page.screenshot(full_page=True)
