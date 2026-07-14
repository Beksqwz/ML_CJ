"""Run a bounded, leakage-aware official gov.kz historical backfill pilot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from future_intelligence.history.gov_kz_history import (  # noqa: E402
    GovKzHistoricalBackfill,
    temporal_metadata,
    training_eligibility,
)
from future_intelligence.providers.repairs.gov_kz import GovKzRoadEventsProvider  # noqa: E402
from future_intelligence.utils import (  # noqa: E402
    ASTANA_TIMEZONE,
    parse_prediction_datetime,
    to_jsonable,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2022-01-01T00:00:00+05:00")
    parser.add_argument("--end-date", default=datetime.now(ASTANA_TIMEZONE).isoformat())
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--max-articles", type=int, default=100)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    start = parse_prediction_datetime(args.start_date)
    end = parse_prediction_datetime(args.end_date)
    provider = GovKzRoadEventsProvider(
        max_pages=args.max_pages,
        max_articles=args.max_articles,
        request_delay=args.request_delay,
        discovery_method="official-filtered",
    )
    result = provider.collect(
        end,
        24,
        since=start,
        until=end,
        force_refresh=args.force_refresh,
        discovery_method="official-filtered",
    )
    history = GovKzHistoricalBackfill(provider)
    checkpoint = history.load_checkpoint()
    resumed_ids = (
        set(checkpoint.get("seen_source_item_ids", [])) if args.resume else set()
    )
    events = []
    matches = []
    # The live provider already fetched safely and parsed canonical records.  Reuse
    # its records instead of scraping a second detail parser.
    for record in result.normalized_records:
        if str(record.source_item_id) in resumed_ids:
            continue
        record.is_historical = True
        event = record.to_dict() | {
            "title": record.payload.get("title"),
            "description": record.payload.get("description"),
            "restriction_type": record.payload.get("restriction_type"),
            "location": record.payload.get("location"),
            "relevant": True,
            "relevance_score": record.payload.get("relevance_score"),
            "open_end": record.payload.get("open_end", False),
            "date_confidence": "open_end"
            if record.payload.get("open_end")
            else "parsed"
            if record.valid_from
            else "missing",
            "date_extraction_method": "live_gov_kz_parse_dates",
            "date_warnings": record.payload.get("warnings", []),
        }
        event.update(temporal_metadata(record, end))
        found = history.matcher.match_record(record, provider="gov_kz_repairs").matches
        eligible, reasons = training_eligibility(event, found)
        event.update(
            training_eligible=eligible,
            training_rejection_reasons=reasons,
            overall_confidence=record.confidence,
        )
        events.append(event)
        matches.extend(found)
    discovered_urls = [
        item.get("source_url")
        for item in provider.last_report.get("article_diagnostics", [])
        if item.get("source_url")
    ]
    checkpoint_result = history.register_listing_urls(discovered_urls)
    storage = {} if args.dry_run else history.save(events, matches)
    years = {}
    for event in events:
        published = str(event.get("published_at") or "")
        year = published[:4] if published[:4].isdigit() else "unknown"
        years[year] = years.get(year, 0) + 1
    eligible = sum(event["training_eligible"] for event in events)
    blockers = []
    if len(years) < 3:
        blockers.append(
            "bounded_official_listing_pilot_did_not_prove_multi_year_coverage"
        )
    if not eligible:
        blockers.append("no_leakage_safe_historical_training_eligible_events_collected")
    report = {
        "status": result.status,
        "backfill_status": "READY" if not blockers else "NOT_READY",
        "blockers": blockers,
        "target_range": {"start": start.isoformat(), "end": end.isoformat()},
        "pilot_only": True,
        "discovery": provider.last_report,
        "records_parsed": len(events),
        "training_eligible": eligible,
        "matches": len(matches),
        "coverage_by_year": years,
        "date_extraction_rate": (
            sum(bool(event.get("valid_from")) for event in events) / len(events)
            if events
            else 0.0
        ),
        "spatial_match_rate": (
            len({match.source_item_id for match in matches}) / len(events)
            if events
            else 0.0
        ),
        "leakage_safe_rate": eligible / len(events) if events else 0.0,
        "storage": storage,
        "resume": {
            "enabled": args.resume,
            "records_skipped_from_checkpoint": len(result.normalized_records)
            - len(events),
            "checkpoint_urls_added": len(checkpoint_result["accepted_urls"]),
        },
        "warnings": result.warnings,
    }
    reports = ROOT / "reports" / "stage19c"
    reports.mkdir(parents=True, exist_ok=True)
    reports_payloads = {
        "gov_kz_history_input_audit.json": {
            "reused_live_provider": provider.metadata.provider_name,
            "date_extraction": "GovKzRoadEventsProvider.parse_dates",
            "geometry": "AstanaGeocoder + RoadGeometryResolver",
            "spatial_matching": "SpatialMatchingEngine",
            "storage_identity": ["source", "source_item_id"],
            "archive_gap": "No verified historical date navigation was exposed in this bounded official-listing pilot.",
        },
        "gov_kz_history_discovery_report.json": report,
        "gov_kz_history_parser_report.json": report,
        "gov_kz_history_date_quality.json": {
            "records": len(events),
            "with_valid_from": sum(bool(item.get("valid_from")) for item in events),
            "open_end": sum(bool(item.get("open_end")) for item in events),
        },
        "gov_kz_history_spatial_quality.json": {
            "records": len(events),
            "segment_matches": len(matches),
            "matched_events": len({item.source_item_id for item in matches}),
        },
        "gov_kz_history_training_eligibility.json": {
            "eligible": sum(item["training_eligible"] for item in events),
            "rejected": sum(not item["training_eligible"] for item in events),
        },
        "gov_kz_history_leakage_audit.json": {
            "as_known_at_mode_required": True,
            "known_from": "published_at",
            "records_with_warnings": sum(
                bool(item.get("temporal_leakage_warning")) for item in events
            ),
        },
        "gov_kz_history_backfill_summary.json": report,
    }
    for name, payload in reports_payloads.items():
        (reports / name).write_text(
            json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (reports / "gov_kz_history_backfill_summary.md").write_text(
        "# Stage 19C gov.kz historical pilot\n\n"
        f"Official pages scanned: {provider.last_report.get('pages_requested', 0)}\n\n"
        f"Relevant records parsed: {len(events)}\n\n"
        f"Training eligible: {sum(item['training_eligible'] for item in events)}\n\n"
        "This is a bounded pilot, not a claim of multi-year coverage. Historical features must be built only in as-known-at mode from `published_at`.\n",
        encoding="utf-8",
    )
    print(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2))
    return 1 if args.strict and not events else 0


if __name__ == "__main__":
    raise SystemExit(main())
