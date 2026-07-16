# Docker deployment with Future Layer

`docker compose up -d --build` starts two persistent services:

- `ml-api` serves the ML HTTP API on port 7860.
- `future-worker` refreshes Future Intelligence, then requests a new prediction
  batch and city action plan from `ml-api`.

Both use `restart: unless-stopped`, so Docker starts them again after a server
reboot and restarts them after a failure.

## One-time server setup

1. Copy `.env.example` to `.env` and set `ML_SERVICE_API_KEY`,
   `OPENWEATHER_API_KEY` and `TOMTOM_API_KEY`.
2. Ensure the deployment has the initial Stage 18A source table at
   `data/future_intelligence/processed/future_segment_features_24h.parquet`.
   It is intentionally local runtime data and is not committed to Git.
3. Start the stack:

   ```bash
   docker compose up -d --build
   ```

4. Inspect live operation:

   ```bash
   docker compose ps
   docker compose logs -f future-worker
   cat data/runtime/future_scheduler_state.json
   ```

The state file contains provider outcomes and `api_sync`, including the generated
prediction `batch_id` and `plan_id`. The website should show this state as the
Future Layer freshness/status indicator for operators and judges.

## Data flow

After any successful worker cycle, it calls the API with the same
`ML_SERVICE_API_KEY`. The API persists a new batch using the current unified
Future Layer and creates an action plan for that exact batch. The frontend must
display `GET /api/v1/action-plans/latest`, not `/api/v1/recommendations/top`, for
the operational plan. `backend_sync` contract version 2 supplies the separate,
display-ready `future_context` for a selected road segment.
