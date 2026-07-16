"""Standalone Future Intelligence scheduler; never run inside API workers."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
INTERVALS = {
    "tomtom": "FUTURE_TRAFFIC_INTERVAL_MINUTES",
    "openweather": "FUTURE_WEATHER_INTERVAL_MINUTES",
    "gov_kz_repairs": "FUTURE_REPAIRS_INTERVAL_MINUTES",
    "ticketon_events": "FUTURE_EVENTS_INTERVAL_MINUTES",
}
DEFAULTS = {
    "tomtom": 10,
    "openweather": 30,
    "gov_kz_repairs": 180,
    "ticketon_events": 720,
}
running = True


def retry_policy() -> tuple[int, float, float, float]:
    return (
        int(os.getenv("FUTURE_RETRY_MAX_ATTEMPTS", "3")),
        float(os.getenv("FUTURE_RETRY_INITIAL_SECONDS", "2")),
        float(os.getenv("FUTURE_RETRY_MAX_SECONDS", "30")),
        float(os.getenv("FUTURE_RETRY_MULTIPLIER", "2")),
    )


def retryable(returncode: int) -> bool:
    # argparse/configuration errors are permanent; provider/network errors are retried.
    return returncode not in (0, 2)


def run_with_retry(
    provider: str,
    execute: Callable[[str], int],
    *,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = lambda: 0.0,
    active: Callable[[], bool] = lambda: running,
) -> dict[str, object]:
    maximum, initial, ceiling, multiplier = retry_policy()
    attempts, delay = 0, initial
    while attempts < maximum and active():
        attempts += 1
        code = execute(provider)
        if code == 0:
            return {
                "provider": provider,
                "status": "ok",
                "attempts": attempts,
                "returncode": code,
            }
        if not retryable(code):
            return {
                "provider": provider,
                "status": "permanent_error",
                "attempts": attempts,
                "returncode": code,
            }
        if attempts < maximum and active():
            sleep(min(ceiling, delay) + max(0.0, jitter()))
            delay = min(ceiling, delay * multiplier)
    return {
        "provider": provider,
        "status": "degraded",
        "attempts": attempts,
        "returncode": code if attempts else None,
    }


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def execute_provider(provider: str) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "refresh_future_intelligence.py"),
        "--providers",
        provider,
    ]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True).returncode


def synchronize_api() -> dict[str, object]:
    """Ask the API to persist a new batch and operational plan after refresh."""

    if not os.getenv("FUTURE_SYNC_API_URL") or not os.getenv("ML_SERVICE_API_KEY"):
        return {"status": "disabled", "reason": "sync_configuration_missing"}
    command = [sys.executable, str(ROOT / "scripts" / "sync_future_layer_to_api.py")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {"status": "degraded", "error": "sync_response_invalid"}
    if result.returncode:
        payload["status"] = "degraded"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "data" / "runtime" / "future_scheduler_state.json",
    )
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "network_calls": 0,
                    "writes": 0,
                    "providers_due": list(INTERVALS),
                    "unified_layer_rebuild_planned": True,
                    "configuration_valid": all(
                        int(os.getenv(name, DEFAULTS[key])) > 0
                        for key, name in INTERVALS.items()
                    ),
                }
            )
        )
        return 0

    def stop(*_: object) -> None:
        global running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    due = {name: 0.0 for name in INTERVALS}
    while running:
        now = time.monotonic()
        names = [name for name in INTERVALS if now >= due[name]]
        if names:
            outcomes = [
                run_with_retry(
                    name, execute_provider, jitter=lambda: random.uniform(0, 0.1)
                )
                for name in names
            ]
            for name in names:
                due[name] = now + 60 * int(os.getenv(INTERVALS[name], DEFAULTS[name]))
            api_sync = (
                synchronize_api()
                if any(outcome["status"] == "ok" for outcome in outcomes)
                else {"status": "skipped", "reason": "no_provider_refreshed"}
            )
            atomic_json(
                args.state,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "providers": outcomes,
                    "api_sync": api_sync,
                },
            )
        if args.once:
            return 0
        time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
