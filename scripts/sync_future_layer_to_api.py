"""Create a fresh prediction batch and city plan after Future Layer refresh."""

from __future__ import annotations

import json
import os

import httpx


def main() -> int:
    base_url = os.getenv("FUTURE_SYNC_API_URL", "").rstrip("/")
    api_key = os.getenv("ML_SERVICE_API_KEY", "")
    if not base_url or not api_key:
        print(
            json.dumps({"status": "disabled", "reason": "sync_configuration_missing"})
        )
        return 0

    headers = {"X-API-Key": api_key}
    timeout = float(os.getenv("FUTURE_SYNC_API_TIMEOUT_SECONDS", "180"))
    max_actions = int(os.getenv("FUTURE_SYNC_MAX_ACTIONS", "10"))
    minimum_priority = os.getenv("FUTURE_SYNC_MINIMUM_PRIORITY", "medium")
    try:
        with httpx.Client(timeout=timeout) as client:
            prediction = client.post(
                f"{base_url}/api/v1/predict",
                headers=headers,
                json={"response_mode": "compact"},
            )
            prediction.raise_for_status()
            batch_id = prediction.json().get("batchId")
            if not batch_id:
                raise ValueError("prediction_batch_id_missing")
            plan = client.post(
                f"{base_url}/api/v1/action-plans",
                headers=headers,
                json={
                    "batch_id": batch_id,
                    "max_actions": max_actions,
                    "minimum_priority": minimum_priority,
                },
            )
            plan.raise_for_status()
    except (httpx.HTTPError, ValueError) as exc:
        print(json.dumps({"status": "degraded", "error": str(exc)}))
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "batch_id": batch_id,
                "plan_id": plan.json().get("plan_id"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
