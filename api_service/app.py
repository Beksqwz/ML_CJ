"""HTTP boundary for the RRAI ML service."""

from __future__ import annotations

import hmac
import os
import time
import uuid
from datetime import UTC, datetime
from zoneinfo import ZoneInfo
from threading import Lock
from typing import Callable, Literal

import httpx
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from ml_service.hybrid_risk import ENGINE_VERSION, build_hybrid_risk
from recommendations.stage20b import recommend_stage20b
from recommendations.city_action_plan import generate_city_action_plan
from ml_service.runtime_store import PredictionStore


class PredictRequest(BaseModel):
    force: bool = False
    prediction_datetime: datetime | None = None
    strict_live_features: bool = False


class ActionPlanRequest(BaseModel):
    batch_id: str | None = None
    max_actions: int = Field(default=10, ge=1, le=50)
    minimum_priority: Literal["critical", "high", "medium"] = "medium"

    @field_validator("batch_id")
    @classmethod
    def validate_batch_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("batch_id must not be blank")
        return value


class TrainingRequest(BaseModel):
    baseDatasetSnapshotId: str
    includeConfirmedEventsUntil: datetime


class Runtime:
    def __init__(
        self,
        predictor: Callable[[str], pd.DataFrame] | None = None,
        store: PredictionStore | None = None,
    ) -> None:
        self.predictor = predictor or self._predict
        self.latest: pd.DataFrame | None = None
        self.batch_id: str | None = None
        self.running = False
        self.lock = Lock()
        self.store = store or PredictionStore()

    @staticmethod
    def _predict(prediction_datetime: str) -> pd.DataFrame:
        return recommend_stage20b(
            build_hybrid_risk(
                prediction_datetime,
                strict_live_features=os.getenv(
                    "ML_STRICT_LIVE_FEATURES", "false"
                ).lower()
                == "true",
            )
        )

    def predict(
        self,
        prediction_datetime: datetime | None = None,
        strict_live_features: bool = False,
    ) -> tuple[str, pd.DataFrame, int]:
        with self.lock:
            if self.running:
                raise RuntimeError("PREDICTION_ALREADY_RUNNING")
            self.running = True
        started = time.perf_counter()
        try:
            # A deterministic override supports audited backfills and Docker smoke runs.
            at = (
                (
                    prediction_datetime
                    or datetime.fromisoformat(
                        os.getenv(
                            "ML_PREDICTION_DATETIME",
                            datetime.now(ZoneInfo("Asia/Almaty")).isoformat(),
                        )
                    )
                )
                .astimezone(ZoneInfo("Asia/Almaty"))
                .isoformat()
            )
            try:
                frame = (
                    self.predictor(at)
                    if not strict_live_features
                    else recommend_stage20b(
                        build_hybrid_risk(at, strict_live_features=True)
                    )
                )
            except ValueError as exc:
                if str(exc).startswith("LIVE_WEATHER_UNAVAILABLE"):
                    raise RuntimeError("LIVE_WEATHER_UNAVAILABLE") from exc
                raise
            self.latest = frame
            self.batch_id = str(uuid.uuid4())
            elapsed = round((time.perf_counter() - started) * 1000)
            completed = datetime.now(UTC).isoformat()
            self.store.save_completed(
                {
                    "batch_id": self.batch_id,
                    "prediction_datetime": at,
                    "started_at": completed,
                    "completed_at": completed,
                    "execution_time_ms": elapsed,
                    "model_version": str(
                        uuid.uuid5(uuid.NAMESPACE_URL, ENGINE_VERSION)
                    ),
                    "warnings": [],
                },
                frame,
            )
            return self.batch_id, frame, elapsed
        finally:
            with self.lock:
                self.running = False


class BackendLiveEventsClient:
    """Read privacy-minimised approved events from the Backend contract."""

    def __init__(
        self, base_url: str, api_key: str, client: httpx.Client | None = None
    ) -> None:
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self.client = client or httpx.Client(timeout=20)

    def fetch_until(self, until: datetime) -> list[dict[str, object]]:
        page, events = 1, []
        while True:
            response = self.client.get(
                f"{self.base_url}/api/v1/internal/live-events-for-training",
                headers={"X-API-Key": self.api_key},
                params={"until": until.isoformat(), "page": page, "limit": 1000},
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success"):
                raise ValueError("backend_live_events_unsuccessful")
            batch = payload.get("data", [])
            events.extend(batch)
            meta = payload.get("meta", {})
            if page * int(meta.get("limit", 1000)) >= int(
                meta.get("total", len(events))
            ):
                return events
            page += 1


class TrainingRuntime:
    """Idempotent queue boundary; the approved offline worker owns training."""

    def __init__(
        self, fetch_events: Callable[[datetime], list[dict[str, object]]] | None = None
    ) -> None:
        self.fetch_events = fetch_events
        self.jobs: dict[str, dict[str, object]] = {}
        self.idempotency: dict[str, str] = {}
        self.lock = Lock()

    def start(self, request: TrainingRequest, key: str) -> str:
        with self.lock:
            if key in self.idempotency:
                return self.idempotency[key]
            if any(
                job["status"] in {"QUEUED", "RUNNING"} for job in self.jobs.values()
            ):
                raise RuntimeError("TRAINING_ALREADY_RUNNING")
            if self.fetch_events is None:
                raise ValueError("DATA_INSUFFICIENT")
            events = self.fetch_events(request.includeConfirmedEventsUntil)
            if len(events) < int(os.getenv("ML_MIN_TRAINING_EVENTS", "1")):
                raise ValueError("DATA_INSUFFICIENT")
            run_id = str(uuid.uuid4())
            self.jobs[run_id] = {
                "status": "QUEUED",
                "startedAt": None,
                "completedAt": None,
                "modelVersionId": None,
                "errorSummary": None,
                "baseDatasetSnapshotId": request.baseDatasetSnapshotId,
                "confirmedEventsCount": len(events),
            }
            self.idempotency[key] = run_id
            return run_id


def create_app(
    *,
    api_key: str | None = None,
    runtime: Runtime | None = None,
    training: TrainingRuntime | None = None,
) -> FastAPI:
    secret = api_key if api_key is not None else os.getenv("ML_SERVICE_API_KEY", "")
    runtime = runtime or Runtime()
    training = training or TrainingRuntime(
        BackendLiveEventsClient(os.environ["BACKEND_URL"], secret).fetch_until
        if os.getenv("BACKEND_URL") and secret
        else None
    )
    app = FastAPI(title="RRAI ML Service", version="1.0.0")

    def authenticated(x_api_key: str | None = Header(default=None)) -> None:
        if not secret or not x_api_key or not hmac.compare_digest(x_api_key, secret):
            raise HTTPException(status_code=401, detail="INVALID_API_KEY")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/model-status", dependencies=[Depends(authenticated)])
    def model_status() -> dict[str, object]:
        return {
            "activeModelVersionId": str(uuid.uuid5(uuid.NAMESPACE_URL, ENGINE_VERSION)),
            "modelName": "Stage19I CatBoost + HistGradientBoosting",
            "trainedAt": None,
            "metrics": {"prAuc": 0.28420836316515546, "rocAuc": 0.6463727582214999},
            "featureSchemaVersion": 1,
        }

    @app.get("/model-info", dependencies=[Depends(authenticated)])
    def model_info() -> dict[str, object]:
        return model_status()

    @app.post("/api/v1/predict", dependencies=[Depends(authenticated)])
    def predict(_: PredictRequest) -> dict[str, object]:
        try:
            batch_id, frame, elapsed = runtime.predict(
                _.prediction_datetime, _.strict_live_features
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="PREDICTION_FAILED") from exc
        return {
            "status": "completed",
            "batchId": batch_id,
            "predictionsCount": len(frame),
            "executionTimeMs": elapsed,
            "modelVersionId": str(uuid.uuid5(uuid.NAMESPACE_URL, ENGINE_VERSION)),
            "completedAt": datetime.now(UTC).isoformat(),
        }

    @app.post(
        "/api/v1/action-plans", status_code=201, dependencies=[Depends(authenticated)]
    )
    def create_action_plan(request: ActionPlanRequest) -> dict[str, object]:
        batch = (
            runtime.store.batch(request.batch_id)
            if request.batch_id
            else runtime.store.latest()
        )
        if batch is None or batch["status"] != "completed":
            raise HTTPException(
                status_code=404, detail="COMPLETED_PREDICTION_BATCH_NOT_FOUND"
            )
        segments = runtime.store.get_prediction_segments_for_batch(batch["batchId"])
        if not segments:
            raise HTTPException(
                status_code=404, detail="COMPLETED_PREDICTION_BATCH_NOT_FOUND"
            )
        plan_id = str(uuid.uuid4())
        generated_at = datetime.now(UTC).isoformat()
        runtime.store.create_action_plan(
            plan_id=plan_id,
            batch_id=batch["batchId"],
            prediction_datetime=batch["predictionDatetime"],
            horizon_hours=batch["horizonHours"],
            max_actions=request.max_actions,
            minimum_priority=request.minimum_priority,
            generated_at=generated_at,
        )
        try:
            plan = generate_city_action_plan(
                segments,
                batch_id=batch["batchId"],
                prediction_datetime=batch["predictionDatetime"],
                horizon_hours=batch["horizonHours"],
                max_actions=request.max_actions,
                minimum_priority=request.minimum_priority,
            )
            plan.update(
                {
                    "plan_id": plan_id,
                    "status": "completed",
                    "generated_at": generated_at,
                    "request_parameters": {
                        "max_actions": request.max_actions,
                        "minimum_priority": request.minimum_priority,
                    },
                }
            )
            runtime.store.save_completed_action_plan(
                plan,
                max_actions=request.max_actions,
                minimum_priority=request.minimum_priority,
            )
            return plan
        except Exception:
            runtime.store.save_failed_action_plan(
                plan_id=plan_id,
                batch_id=batch["batchId"],
                prediction_datetime=batch["predictionDatetime"],
                error="CITY_ACTION_PLAN_GENERATION_FAILED",
            )
            raise HTTPException(
                status_code=500, detail="CITY_ACTION_PLAN_GENERATION_FAILED"
            )

    @app.get("/api/v1/action-plans/latest", dependencies=[Depends(authenticated)])
    def latest_action_plan() -> dict[str, object]:
        plan = runtime.store.get_latest_action_plan()
        if plan is None:
            raise HTTPException(status_code=404, detail="CITY_ACTION_PLAN_NOT_FOUND")
        return plan

    @app.get("/api/v1/action-plans/{plan_id}", dependencies=[Depends(authenticated)])
    def get_action_plan(plan_id: str) -> dict[str, object]:
        if not plan_id.strip():
            raise HTTPException(status_code=404, detail="CITY_ACTION_PLAN_NOT_FOUND")
        plan = runtime.store.get_action_plan(plan_id)
        if plan is None or plan.get("status") != "completed":
            raise HTTPException(status_code=404, detail="CITY_ACTION_PLAN_NOT_FOUND")
        return plan

    @app.post(
        "/api/v1/training", status_code=202, dependencies=[Depends(authenticated)]
    )
    def start_training(
        request: TrainingRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, str]:
        if not idempotency_key:
            raise HTTPException(status_code=422, detail="IDEMPOTENCY_KEY_REQUIRED")
        try:
            run_id = training.start(request, idempotency_key)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "accepted", "trainingRunId": run_id}

    @app.get(
        "/api/v1/training/{training_run_id}", dependencies=[Depends(authenticated)]
    )
    def training_status(training_run_id: str) -> dict[str, object]:
        job = training.jobs.get(training_run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="TRAINING_RUN_NOT_FOUND")
        return job

    @app.get("/api/v1/risk/top", dependencies=[Depends(authenticated)])
    def risk_top(limit: int = 50) -> list[dict[str, object]]:
        if runtime.latest is None and runtime.store.latest() is None:
            raise HTTPException(status_code=422, detail="ML_MODEL_NOT_READY")
        return (
            runtime.latest.nsmallest(max(1, min(limit, 500)), "priority_rank").to_dict(
                "records"
            )
            if runtime.latest is not None
            else runtime.store.top(max(1, min(limit, 500)))
        )

    @app.get("/api/v1/recommendations/top", dependencies=[Depends(authenticated)])
    def recommendations_top(limit: int = 50) -> list[dict[str, object]]:
        return risk_top(limit)

    @app.get(
        "/api/v1/risk/segment/{road_segment_id}", dependencies=[Depends(authenticated)]
    )
    def risk_segment(road_segment_id: str) -> dict[str, object]:
        if runtime.latest is None and runtime.store.latest() is None:
            raise HTTPException(status_code=422, detail="ML_MODEL_NOT_READY")
        row = (
            runtime.latest.loc[
                runtime.latest.road_segment_id.astype(str).eq(road_segment_id)
            ]
            .iloc[0]
            .to_dict()
            if runtime.latest is not None
            and not runtime.latest.loc[
                runtime.latest.road_segment_id.astype(str).eq(road_segment_id)
            ].empty
            else runtime.store.segment(road_segment_id)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="SEGMENT_NOT_FOUND")
        return row | {
            "dynamic_risk": {
                "score": row.get("dynamic_score"),
                "rank": row.get("dynamic_rank"),
                "percentile": row.get("dynamic_percentile"),
                "engine_version": row.get("dynamic_engine_version"),
                "status": row.get("dynamic_engine_status"),
            },
            "historical_hotspot": {
                "rank": row.get("historical_hotspot_rank"),
                "percentile": row.get("historical_hotspot_percentile"),
                "accident_count": row.get("historical_accident_count"),
                "accident_count_30d": row.get("historical_accident_count_30d"),
                "accident_count_90d": row.get("historical_accident_count_90d"),
                "accident_count_365d": row.get("historical_accident_count_365d"),
            },
            "context": {
                "weather": {
                    "available": row.get("weather_context_available"),
                    "severity_score": row.get("weather_severity_score"),
                    "collected_at": row.get("weather_collected_at"),
                    "provider": row.get("weather_provider", "openweather"),
                    "snapshotVersion": row.get("weather_snapshot_version"),
                    "validFrom": row.get("weather_valid_from"),
                    "validUntil": row.get("weather_valid_until"),
                    "sourceStepHours": row.get("weather_source_step_hours"),
                    "consistent": row.get("weather_snapshot_consistent", False),
                    "degraded": row.get("ml_weather_degraded", True),
                    "originHour": {
                        "predictionDatetime": row.get(
                            "weather_origin_prediction_datetime"
                        ),
                        "sourceBefore": row.get("weather_origin_source_before"),
                        "sourceAfter": row.get("weather_origin_source_after"),
                        "interpolated": row.get("weather_origin_interpolated"),
                        "temperature": row.get("weather_origin_temperature"),
                        "humidity": row.get("weather_origin_humidity"),
                        "pressure": row.get("weather_origin_pressure"),
                        "windSpeed": row.get("weather_origin_wind_speed"),
                        "precipitation": row.get("weather_origin_rain"),
                        "visibility": row.get("weather_origin_visibility"),
                        "weatherCondition": row.get("weather_origin_weather_condition"),
                    },
                    "summary24h": {
                        "forecastStart": row.get("weather_forecast_start"),
                        "forecastEnd": row.get("weather_forecast_end"),
                        "forecastPointsAvailable": row.get(
                            "weather_forecast_points_available"
                        ),
                        "expectedPoints": row.get("weather_expected_points"),
                        "forecastComplete": row.get("weather_forecast_complete"),
                        "maxSeverityScore": row.get(
                            "weather_max_weather_severity_score"
                        ),
                        "severeWeatherExpected": row.get(
                            "weather_severe_weather_expected"
                        ),
                        "worstPeriodStart": row.get("weather_worst_period_start"),
                        "worstPeriodEnd": row.get("weather_worst_period_end"),
                        "snowExpected": row.get("weather_snow_expected"),
                        "heavyRainExpected": row.get("weather_heavy_rain_expected"),
                        "minimumVisibilityM": row.get("weather_minimum_visibility_m"),
                        "maximumWindSpeed": row.get("weather_maximum_wind_speed"),
                        "temperatureMin": row.get("weather_summary_temperature_min"),
                        "temperatureMax": row.get("weather_summary_temperature_max"),
                    },
                    "provenance": {
                        "mlSnapshotVersion": row.get("ml_weather_snapshot_version"),
                        "explanationSnapshotVersion": row.get(
                            "explanation_weather_snapshot_version"
                        ),
                        "originTimestamp": row.get("ml_weather_origin_timestamp"),
                        "sourceBefore": row.get("ml_weather_source_before"),
                        "sourceAfter": row.get("ml_weather_source_after"),
                        "interpolated": row.get("ml_weather_interpolated"),
                        "forecastStart": row.get("explanation_forecast_start"),
                        "forecastEnd": row.get("explanation_forecast_end"),
                        "forecastPointsAvailable": row.get(
                            "explanation_forecast_points_available"
                        ),
                    },
                },
                "traffic": {
                    "available": row.get("traffic_context_available"),
                    "severity_score": row.get("traffic_severity_score"),
                    "collected_at": row.get("traffic_collected_at"),
                    "provider": "tomtom",
                },
                "repairs": {
                    "available": row.get("repair_context_available"),
                    "active": row.get("repair_active"),
                },
                "events": {
                    "available": row.get("event_context_available"),
                    "major": row.get("event_major"),
                },
            },
        }

    @app.get("/api/v1/batches/latest", dependencies=[Depends(authenticated)])
    def latest_batch() -> dict[str, object]:
        value = runtime.store.latest()
        if value is None:
            raise HTTPException(status_code=404, detail="BATCH_NOT_FOUND")
        return value

    @app.get("/api/v1/batches/{batch_id}", dependencies=[Depends(authenticated)])
    def batch(batch_id: str) -> dict[str, object]:
        value = runtime.store.batch(batch_id)
        if value is None:
            raise HTTPException(status_code=404, detail="BATCH_NOT_FOUND")
        return value

    @app.get(
        "/api/v1/batches/{batch_id}/segments/{road_segment_id}",
        dependencies=[Depends(authenticated)],
    )
    def batch_segment(batch_id: str, road_segment_id: str) -> dict[str, object]:
        value = runtime.store.segment(road_segment_id, batch_id)
        if value is None:
            raise HTTPException(status_code=404, detail="SEGMENT_NOT_FOUND")
        value["context"] = value.get("context", {}) | {
            "weather": {
                "available": value.get("weather_context_available", False),
                "consistent": value.get("weather_snapshot_consistent", False),
                "degraded": value.get("ml_weather_degraded", True),
                "snapshotVersion": value.get("weather_snapshot_version"),
                "provenance": {
                    "mlSnapshotVersion": value.get("ml_weather_snapshot_version"),
                    "explanationSnapshotVersion": value.get(
                        "explanation_weather_snapshot_version"
                    ),
                    "originTimestamp": value.get("ml_weather_origin_timestamp"),
                    "sourceBefore": value.get("ml_weather_source_before"),
                    "sourceAfter": value.get("ml_weather_source_after"),
                    "interpolated": value.get("ml_weather_interpolated"),
                    "forecastStart": value.get("explanation_forecast_start"),
                    "forecastEnd": value.get("explanation_forecast_end"),
                    "forecastPointsAvailable": value.get(
                        "explanation_forecast_points_available"
                    ),
                },
            }
        }
        for key in (
            "frozen_feature_vector",
            "raw_provider_payload",
            "api_key",
            "authorization",
            "local_file_path",
            "provider_secret",
        ):
            value.pop(key, None)
        return value

    return app


app = create_app()
