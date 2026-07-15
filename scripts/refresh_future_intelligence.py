"""Refresh Future Intelligence outside the HTTP prediction process."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
KNOWN_PROVIDERS = ("openweather", "tomtom", "gov_kz_repairs", "ticketon_events")


def validate_providers(names: list[str]) -> None:
    unknown = sorted(set(names) - set(KNOWN_PROVIDERS))
    if unknown:
        raise ValueError(f"unknown_future_providers:{unknown}")


def collect_provider(
    provider: str, prediction_datetime: str, runner: Callable = subprocess.run
) -> int:
    """Run one isolated collector.  The caller owns retries and never logs output."""
    command = [
        sys.executable,
        str(ROOT / "scripts" / "collect_future_intelligence.py"),
        "--providers",
        provider,
        "--prediction-datetime",
        prediction_datetime,
    ]
    return runner(command, cwd=ROOT, text=True, capture_output=True).returncode


def rebuild_unified(runner: Callable = subprocess.run) -> int:
    command = [sys.executable, str(ROOT / "scripts" / "build_unified_future_layer.py")]
    return runner(command, cwd=ROOT, text=True, capture_output=True).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", default=",".join(KNOWN_PROVIDERS))
    parser.add_argument(
        "--prediction-datetime", default=datetime.now().astimezone().isoformat()
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names = [item.strip() for item in args.providers.split(",") if item.strip()]
    try:
        validate_providers(names)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "mode": "dry_run" if args.dry_run else "live",
                    "configuration_valid": False,
                    "error": str(exc),
                }
            )
        )
        return 2
    if args.dry_run:
        # Deliberately no subprocess, filesystem write, key lookup, or provider import.
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "network_calls": 0,
                    "writes": 0,
                    "providers_due": names,
                    "unified_layer_rebuild_planned": True,
                    "configuration_valid": True,
                }
            )
        )
        return 0
    outcomes = {
        provider: collect_provider(provider, args.prediction_datetime)
        for provider in names
    }
    unified = rebuild_unified()
    print(
        json.dumps(
            {
                "mode": "live",
                "providers": {
                    key: "ok" if value == 0 else "degraded"
                    for key, value in outcomes.items()
                },
                "unified": "ok" if unified == 0 else "degraded",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
