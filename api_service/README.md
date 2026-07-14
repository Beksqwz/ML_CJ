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
