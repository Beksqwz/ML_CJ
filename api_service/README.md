# Docker ML Service

Build from the repository root:

```bash
docker build -f api_service/Dockerfile -t rrai-ml-service .
docker run --rm -p 7860:7860 -e ML_SERVICE_API_KEY=change-me -e ML_PREDICTION_DATETIME=2024-09-30T19:00:00+05:00 rrai-ml-service
```

For a Hugging Face Docker Space, use this project as the Space source (including
the required frozen `models/` and source datasets) and place this Dockerfile at
the Space repository root. Set `ML_SERVICE_API_KEY` as a Space secret. Do not
commit `.env` or API keys.

`POST /api/v1/training` implements the authenticated, idempotent queue contract
and fetches approved events from `BACKEND_URL`. The actual retraining worker is
intentionally offline: it must validate a snapshot and approve a candidate model
before any active-model change.

## Production live operation

Run two processes (never start the scheduler in each API web worker):

```bash
uvicorn api_service.app:app --host 0.0.0.0 --port 7860
python scripts/run_future_intelligence_scheduler.py
```

For a deployment check, refresh once without network writes using
`python scripts/refresh_future_intelligence.py --dry-run`, or run one scheduler
cycle with `python scripts/run_future_intelligence_scheduler.py --once --dry-run`.
The service predicts the Stage19I **24-hour ranking** for `(t, t+24h]`; it is
not an accident probability. Live calendar fields are generated in Asia/Almaty
when archival coverage ends. Weather/Future Intelligence are optional: stale or
unavailable providers produce warnings and degraded uncertainty, not a failed
dynamic prediction. TTL defaults are traffic 20m, weather 90m, repairs 6h and
events 24h. Completed batches are durable in `data/runtime/ml_service.sqlite3`
and can be read through `/api/v1/batches/latest` or by batch id after restart.
Provider collection retries are isolated and bounded (three attempts by default,
2-second exponential backoff capped at 30 seconds). Configure them with
`FUTURE_RETRY_MAX_ATTEMPTS`, `FUTURE_RETRY_INITIAL_SECONDS`,
`FUTURE_RETRY_MAX_SECONDS`, and `FUTURE_RETRY_MULTIPLIER`. Both worker CLIs
support a fully offline `--dry-run`: it validates scheduling and provider names
without collectors, network calls, writes, credentials, or waiting.

## Operator plan flows

Segment inspection and the city-wide plan are separate operations. A map click uses
`GET /api/v1/risk/segment/{road_segment_id}` to explain one segment: its dynamic
and historical ranks, available weather/traffic/repair/event context, warnings and
uncertainty. It is not a city action plan.

The dashboard button **«Сформировать возможный план»** calls:

```http
POST /api/v1/action-plans
Content-Type: application/json

{"batch_id": null, "max_actions": 10, "minimum_priority": "medium"}
```

The service reads an existing completed prediction batch, produces and persists a
compact Top-N city plan, and returns it. The client can later read the stored plan
with `GET /api/v1/action-plans/latest` or `GET /api/v1/action-plans/{plan_id}`.
No ML prediction is re-run and no provider is refreshed. `action_priority_score`
orders operational actions; it is not an accident probability. Every action needs
human/operator confirmation. `/api/v1/recommendations/top` remains a segment
ranking endpoint, not the city-plan endpoint.

## Synchronous prediction response modes

`POST /api/v1/predict` remains compact by default and first persists the completed
24-hour prediction batch. A compact request is either omitted or explicit:

```json
{"response_mode": "compact"}
```

It returns batch metadata only, including `responseMode: "compact"`,
`predictionsIncluded: false`, and `responseSizeBytes`. To receive the already
persisted 3,968 compact segment results in the same HTTP response, request:

```json
{
  "response_mode": "full",
  "include_explanations": true,
  "max_explanation_factors": 3
}
```

Full mode returns `predictions` ordered by final dynamic rank. The main
`dynamic_score` / `dynamic_risk.score` is the Stage19I weighted-percentile
ensemble ordering score, not an accident probability. `model_components` keeps
the CatBoost and HGB scores separately. The optional SHAP factors explain only
the CatBoost component, not the full ensemble; set `include_explanations` to
`false` or `max_explanation_factors` to `0..3` to reduce the response.

The service does not rerun prediction, SHAP, or provider collection to form a
full response: it reads the completed batch just written to SQLite. Full payloads
are bounded by `PREDICT_FULL_RESPONSE_MAX_BYTES` (default 16 MiB). If the bound
is exceeded, the batch remains completed and the API returns HTTP 413 with its
`batchId`; retry compact mode or use segment/batch lookup endpoints. No raw
feature vectors, full SHAP vectors, provider payloads, credentials, or local
paths are returned.
