"""HTTP boundary for the RRAI ML service."""

from __future__ import annotations

import hmac
import os
import time
import uuid
from datetime import UTC, datetime
from threading import Lock
from typing import Callable

import httpx
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from ml_service.hybrid_risk import ENGINE_VERSION, build_hybrid_risk
from recommendations.stage20b import recommend_stage20b

try:
    from ml_service import AccidentRiskPredictor
    _predictor = AccidentRiskPredictor()
    import threading
        def _warm():
            try:
                at = os.getenv("ML_PREDICTION_DATETIME", datetime.now(UTC).isoformat())
                if "+" in at:
                    at = at[:at.rindex("+")]
                elif at.endswith("Z"):
                    at = at[:-1]
            _predictor.predict_city(at, "1h")
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()
except Exception:
    _predictor = None


class PredictRequest(BaseModel):
    force: bool = False


class TrainingRequest(BaseModel):
    baseDatasetSnapshotId: str
    includeConfirmedEventsUntil: datetime


class Runtime:
    def __init__(self, predictor: Callable[[str], pd.DataFrame] | None = None) -> None:
        self.predictor = predictor or self._predict
        self.latest: pd.DataFrame | None = None
        self.batch_id: str | None = None
        self.running = False
        self.lock = Lock()

    @staticmethod
    def _predict(prediction_datetime: str) -> pd.DataFrame:
        return recommend_stage20b(build_hybrid_risk(prediction_datetime))

    def predict(self) -> tuple[str, pd.DataFrame, int]:
        with self.lock:
            if self.running:
                raise RuntimeError("PREDICTION_ALREADY_RUNNING")
            self.running = True
        started = time.perf_counter()
        try:
            # A deterministic override supports audited backfills and Docker smoke runs.
            at = os.getenv("ML_PREDICTION_DATETIME", datetime.now(UTC).isoformat())
            frame = self.predictor(at)
            self.latest = frame
            self.batch_id = str(uuid.uuid4())
            return self.batch_id, frame, round((time.perf_counter() - started) * 1000)
        finally:
            with self.lock:
                self.running = False


class BackendLiveEventsClient:
    """Read privacy-minimised approved events from the Backend contract."""

    def __init__(self, base_url: str, api_key: str, client: httpx.Client | None = None) -> None:
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
            if page * int(meta.get("limit", 1000)) >= int(meta.get("total", len(events))):
                return events
            page += 1


class TrainingRuntime:
    """Idempotent queue boundary; the approved offline worker owns training."""

    def __init__(self, fetch_events: Callable[[datetime], list[dict[str, object]]] | None = None) -> None:
        self.fetch_events = fetch_events
        self.jobs: dict[str, dict[str, object]] = {}
        self.idempotency: dict[str, str] = {}
        self.lock = Lock()

    def start(self, request: TrainingRequest, key: str) -> str:
        with self.lock:
            if key in self.idempotency:
                return self.idempotency[key]
            if any(job["status"] in {"QUEUED", "RUNNING"} for job in self.jobs.values()):
                raise RuntimeError("TRAINING_ALREADY_RUNNING")
            if self.fetch_events is None:
                raise ValueError("DATA_INSUFFICIENT")
            events = self.fetch_events(request.includeConfirmedEventsUntil)
            if len(events) < int(os.getenv("ML_MIN_TRAINING_EVENTS", "1")):
                raise ValueError("DATA_INSUFFICIENT")
            run_id = str(uuid.uuid4())
            self.jobs[run_id] = {
                "status": "QUEUED", "startedAt": None, "completedAt": None,
                "modelVersionId": None, "errorSummary": None,
                "baseDatasetSnapshotId": request.baseDatasetSnapshotId,
                "confirmedEventsCount": len(events),
            }
            self.idempotency[key] = run_id
            return run_id


def create_app(*, api_key: str | None = None, runtime: Runtime | None = None, training: TrainingRuntime | None = None) -> FastAPI:
    secret = api_key if api_key is not None else os.getenv("ML_SERVICE_API_KEY", "")
    runtime = runtime or Runtime()
    training = training or TrainingRuntime(
        BackendLiveEventsClient(os.environ["BACKEND_URL"], secret).fetch_until
        if os.getenv("BACKEND_URL") and secret else None
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

    def _risk_level(probability: float) -> str:
        if probability >= 0.35:
            return "CRITICAL"
        if probability >= 0.20:
            return "HIGH"
        if probability >= 0.10:
            return "MEDIUM"
        return "LOW"

    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return default

    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return default

    @app.post("/api/v1/predict", dependencies=[Depends(authenticated)])
    def predict(_: PredictRequest) -> dict[str, object]:
        try:
            batch_id, frame, elapsed = runtime.predict()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="PREDICTION_FAILED") from exc

        # Fetch SHAP factors + feature_values from CatBoost predictor (pipeline #1)
        shap_lookup: dict[str, dict[str, object]] = {}
        if _predictor is not None:
            try:
                at = os.getenv("ML_PREDICTION_DATETIME", datetime.now(UTC).isoformat())
                # Strip timezone — predictor data is tz-naive
                if "+" in at:
                    at = at[:at.rindex("+")]
                elif at.endswith("Z"):
                    at = at[:-1]
                city_result = _predictor.predict_city(at, "1h")
                for rec in city_result.get("predictions", []):
                    shap_lookup[str(rec["road_segment_id"])] = {
                        "top_positive_factors": rec.get("top_positive_factors", []),
                        "top_negative_factors": rec.get("top_negative_factors", []),
                        "feature_values": rec.get("feature_values", {}),
                        "longitude": rec.get("longitude"),
                        "latitude": rec.get("latitude"),
                    }
            except Exception:
                pass

        predictions = []
        for _, row in frame.iterrows():
            seg_id = str(row.road_segment_id)
            shap = shap_lookup.get(seg_id, {})
            prob = _safe_float(row.get("score_catboost_stage19h") or row.get("dynamic_score", 0))
            pred = {
                "road_segment_id": seg_id,
                "risk_probability": prob,
                "risk_level": _risk_level(prob),
                "top_positive_factors": shap.get("top_positive_factors") or row.get("top_positive_factors") or [],
                "top_negative_factors": shap.get("top_negative_factors") or row.get("top_negative_factors") or [],
                "feature_values": shap.get("feature_values") or row.get("feature_values") or {},
                "reasons": row.get("reasons") or [],
                "possible_plan": row.get("possible_plan") or [],
                "uncertainty": _safe_float(row.get("uncertainty")),
                "warnings": row.get("warnings") or [],
                "priority_rank": _safe_int(row.get("priority_rank"), 999),
                "longitude": _safe_float(shap.get("longitude") or row.get("longitude")),
                "latitude": _safe_float(shap.get("latitude") or row.get("latitude")),
            }
            predictions.append(pred)
        return {
            "status": "completed", "batchId": batch_id,
            "predictionsCount": len(frame), "executionTimeMs": elapsed,
            "modelVersionId": str(uuid.uuid5(uuid.NAMESPACE_URL, ENGINE_VERSION)),
            "completedAt": datetime.now(UTC).isoformat(),
            "predictions": predictions,
        }

    @app.post("/api/v1/training", status_code=202, dependencies=[Depends(authenticated)])
    def start_training(request: TrainingRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, str]:
        if not idempotency_key:
            raise HTTPException(status_code=422, detail="IDEMPOTENCY_KEY_REQUIRED")
        try:
            run_id = training.start(request, idempotency_key)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "accepted", "trainingRunId": run_id}

    @app.get("/api/v1/training/{training_run_id}", dependencies=[Depends(authenticated)])
    def training_status(training_run_id: str) -> dict[str, object]:
        job = training.jobs.get(training_run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="TRAINING_RUN_NOT_FOUND")
        return job

    @app.get("/api/v1/risk/top", dependencies=[Depends(authenticated)])
    def risk_top(limit: int = 50) -> list[dict[str, object]]:
        if runtime.latest is None:
            raise HTTPException(status_code=422, detail="ML_MODEL_NOT_READY")
        return runtime.latest.nsmallest(max(1, min(limit, 500)), "priority_rank").to_dict("records")

    @app.get("/api/v1/recommendations/top", dependencies=[Depends(authenticated)])
    def recommendations_top(limit: int = 50) -> list[dict[str, object]]:
        return risk_top(limit)

    @app.get("/api/v1/risk/segment/{road_segment_id}", dependencies=[Depends(authenticated)])
    def risk_segment(road_segment_id: str) -> dict[str, object]:
        if runtime.latest is None:
            raise HTTPException(status_code=422, detail="ML_MODEL_NOT_READY")
        matches = runtime.latest.loc[runtime.latest.road_segment_id.astype(str).eq(road_segment_id)]
        if matches.empty:
            raise HTTPException(status_code=404, detail="SEGMENT_NOT_FOUND")
        return matches.iloc[0].to_dict()

    return app


app = create_app()
