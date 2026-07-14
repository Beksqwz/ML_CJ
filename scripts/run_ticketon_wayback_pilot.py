"""Run a bounded, audit-only Wayback pilot for historical Ticketon pages."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_intelligence.history.ticketon_wayback_normalizer import (  # noqa: E402
    TicketonWaybackNormalizer,
)

CDX_URL = "https://web.archive.org/cdx/search/cdx"
FALLBACK_SEEDS = (
    {
        "timestamp": "20230716075946",
        "original": "https://ticketon.kz/astana/event/2000s-hits-tutti-beats",
    },
    {
        "timestamp": "20240203001217",
        "original": "https://ticketon.kz/astana/event/13-komediya-ckz",
    },
    {
        "timestamp": "20250324101923",
        "original": "https://ticketon.kz/astana/event/100-letie-nurgisy-tlendieva-astana",
    },
)


def _canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _bucket(url: str) -> str:
    text = url.casefold()
    if any(
        value in text for value in ("football", "hockey", "match", "futbol", "barys")
    ):
        return "sports"
    if any(value in text for value in ("concert", "koncert", "music", "studio")):
        return "concerts"
    if "fest" in text:
        return "festivals"
    return "other"


def discover(session: requests.Session, limit: int) -> list[dict]:
    params = {
        "url": "ticketon.kz/astana/event/*",
        "output": "json",
        "filter": ["statuscode:200", "mimetype:text/html"],
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "collapse": "digest",
        "limit": limit * 4,
    }
    response = None
    for attempt in range(3):
        response = session.get(CDX_URL, params=params, timeout=30)
        if response.status_code != 503:
            break
        time.sleep(1.0 * (attempt + 1))
    assert response is not None
    response.raise_for_status()
    rows = response.json()
    header, values = rows[0], rows[1:]
    deduplicated: dict[str, dict] = {}
    for row in values:
        item = dict(zip(header, row, strict=True))
        url = _canonical_url(item["original"])
        if item["timestamp"][:4] in {"2023", "2024", "2025"}:
            deduplicated.setdefault(url, item | {"original": url})
    selected: list[dict] = []
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for item in deduplicated.values():
        groups[(item["timestamp"][:4], _bucket(item["original"]))].append(item)
    while len(selected) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 100:
        parser.error("--max-pages must be between 1 and 100")
    session = requests.Session()
    session.headers["User-Agent"] = "ML-CJ-Stage19B-Audit/1.0 (+public research)"
    normalizer = TicketonWaybackNormalizer()
    warnings: list[str] = []
    try:
        pages = (
            list(FALLBACK_SEEDS[: args.max_pages])
            if args.seed_only
            else discover(session, args.max_pages)
        )
        if args.seed_only:
            warnings = ["using_stage19a_verified_wayback_seed_sample"]
    except (requests.RequestException, ValueError, IndexError) as exc:
        pages = list(FALLBACK_SEEDS[: args.max_pages])
        warnings = [
            f"cdx_discovery_failed:{type(exc).__name__}",
            "using_stage19a_verified_wayback_seed_sample",
        ]
    diagnostics, records = [], []
    for page in pages:
        replay = (
            f"https://web.archive.org/web/{page['timestamp']}id_/{page['original']}"
        )
        try:
            response = session.get(replay, timeout=30)
            html = response.text
            parsed, detail = normalizer.normalize(
                html,
                archive_year=int(page["timestamp"][:4]),
                original_url=page["original"],
            )
            records.extend(parsed)
            diagnostics.append(
                {
                    "original_url": page["original"],
                    "archive_timestamp": page["timestamp"],
                    "http_status": response.status_code,
                    "rendered": response.ok,
                    **detail,
                }
            )
        except requests.RequestException as exc:
            diagnostics.append(
                {
                    "original_url": page["original"],
                    "archive_timestamp": page["timestamp"],
                    "http_status": None,
                    "rendered": False,
                    "warning": type(exc).__name__,
                }
            )
        time.sleep(args.request_delay)
    years = collections.Counter(str(item["archive_year"]) for item in records)
    metrics = {
        "archive_pages_checked": len(diagnostics),
        "archive_pages_rendered": sum(item["rendered"] for item in diagnostics),
        "archive_pages_parsed": sum(
            item.get("normalized_events", 0) > 0 for item in diagnostics
        ),
        "jsonld_success_rate": sum(
            item.get("jsonld_blocks", 0) > 0 for item in diagnostics
        )
        / len(diagnostics)
        if diagnostics
        else 0,
        "exact_datetime_rate": sum(bool(item["start_datetime"]) for item in records)
        / len(records)
        if records
        else 0,
        "venue_rate": sum(bool(item["venue"]) for item in records) / len(records)
        if records
        else 0,
        "astana_rate": sum(item["astana_valid"] for item in records) / len(records)
        if records
        else 0,
        "geocoding_rate": sum(item["latitude"] is not None for item in records)
        / len(records)
        if records
        else 0,
        "major_venue_rate": sum(
            item["venue_tier"] in {"high", "very_high"} for item in records
        )
        / len(records)
        if records
        else 0,
        "transport_relevant_count": sum(
            item["is_transport_relevant"] for item in records
        ),
        "duplicate_rate": 1
        - len({item["source_item_id"] for item in records}) / len(records)
        if records
        else 0,
        "usable_by_year": dict(years),
    }
    reports = ROOT / "reports" / "stage19b"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "pages": diagnostics,
        "records": records,
        "metrics": metrics,
        "warnings": warnings,
    }
    (reports / "ticketon_wayback_pilot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "ticketon_transport_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "ticketon_parser_comparison.json").write_text(
        json.dumps(
            {
                "live_parser_archive_sample_success": 0,
                "wayback_normalizer_records": len(records),
                "archive_variant_support": "JSON-LD Event documents and HTML-entity decoded blocks",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports / "ticketon_training_eligibility.json").write_text(
        json.dumps(
            collections.Counter(item["training_eligibility"] for item in records),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports / "ticketon_wayback_pilot.md").write_text(
        f"# Ticketon Wayback pilot\n\nPages checked: {metrics['archive_pages_checked']}; normalized records: {len(records)}; transport-relevant: {metrics['transport_relevant_count']}.\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
