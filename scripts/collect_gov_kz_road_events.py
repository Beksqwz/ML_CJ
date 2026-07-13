"""Conservatively collect official Astana Akimat road-event announcements."""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass
from future_intelligence.providers.repairs.gov_kz import GovKzRoadEventsProvider  # noqa: E402
from future_intelligence.storage import save_gov_kz_result  # noqa: E402
from future_intelligence.utils import parse_prediction_datetime  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-pages", type=int, default=3)
    p.add_argument("--max-articles", type=int, default=10)
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--language", choices=("ru", "kk"), default="ru")
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "future_intelligence"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--request-delay", type=float, default=0.5)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument(
        "--prediction-datetime", default=datetime.now().astimezone().isoformat()
    )
    p.add_argument(
        "--discovery-method",
        choices=(
            "official-filtered",
            "auto",
            "json",
            "sitemap",
            "search",
            "road-search",
            "playwright",
            "html",
        ),
        default="official-filtered",
    )
    p.add_argument("--browser-timeout", type=int, default=15000)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--debug-network", action="store_true")
    p.add_argument("--save-rendered-html", action="store_true")
    a = p.parse_args()
    provider = GovKzRoadEventsProvider(
        max_pages=a.max_pages,
        max_articles=a.max_articles,
        request_delay=a.request_delay,
        discovery_method=a.discovery_method,
        browser_timeout=a.browser_timeout,
        headless=a.headless,
    )
    result = provider.collect(
        parse_prediction_datetime(a.prediction_datetime),
        24,
        language=a.language,
        since=parse_prediction_datetime(a.since) if a.since else None,
        until=parse_prediction_datetime(a.until) if a.until else None,
        force_refresh=a.force_refresh,
        discovery_method=a.discovery_method,
    )
    report = {
        "provider": result.to_context(),
        "collection": provider.last_report,
        "debug_network": "disabled"
        if not a.debug_network
        else "no sensitive headers recorded",
        "save_rendered_html": False
        if not a.save_rendered_html
        else "not available without successful Playwright discovery",
    }
    if not a.dry_run:
        paths, changes = save_gov_kz_result(result, a.output_dir)
        report["storage"] = {k: str(v) for k, v in paths.items()}
        report["changes"] = changes
    (ROOT / "reports" / "stage15").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "stage15" / "gov_kz_collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.status == "ok" or not a.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
