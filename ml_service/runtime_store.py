"""Small durable store for completed ML prediction batches."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class _RuntimeConnection(sqlite3.Connection):
    """Commit/rollback and close connections used as RuntimeStore contexts."""

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            super().__exit__(exc_type, exc, traceback)
        finally:
            self.close()


class PredictionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(
            os.getenv(
                "ML_RUNTIME_DB_PATH", ROOT / "data" / "runtime" / "ml_service.sqlite3"
            )
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS prediction_batches (
              batch_id TEXT PRIMARY KEY, status TEXT NOT NULL, prediction_datetime TEXT,
              horizon_hours INTEGER NOT NULL, started_at TEXT, completed_at TEXT,
              execution_time_ms INTEGER, segment_count INTEGER, model_version TEXT,
              future_snapshot_version TEXT, warnings TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS prediction_segments (
              batch_id TEXT NOT NULL, road_segment_id TEXT NOT NULL, result_json TEXT NOT NULL,
              PRIMARY KEY(batch_id, road_segment_id),
              FOREIGN KEY(batch_id) REFERENCES prediction_batches(batch_id));
            CREATE INDEX IF NOT EXISTS idx_segments_road ON prediction_segments(road_segment_id);
            CREATE INDEX IF NOT EXISTS idx_batches_status_time ON prediction_batches(status, completed_at DESC);
            CREATE TABLE IF NOT EXISTS city_action_plans (
              plan_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, status TEXT NOT NULL,
              prediction_datetime TEXT NOT NULL, horizon_hours INTEGER NOT NULL,
              plan_version TEXT NOT NULL, generated_at TEXT NOT NULL, max_actions INTEGER,
              minimum_priority TEXT, segments_analyzed INTEGER, candidate_segments INTEGER,
              groups_created INTEGER, actions_returned INTEGER, plan_json TEXT,
              warnings_json TEXT, error TEXT, created_at TEXT NOT NULL, completed_at TEXT);
            CREATE INDEX IF NOT EXISTS idx_city_plans_batch ON city_action_plans(batch_id);
            CREATE INDEX IF NOT EXISTS idx_city_plans_status_generated ON city_action_plans(status, generated_at DESC);
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, factory=_RuntimeConnection)

    def save_completed(self, metadata: dict[str, Any], frame: pd.DataFrame) -> None:
        if frame.empty or frame.road_segment_id.duplicated().any():
            raise ValueError("invalid_completed_batch")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO prediction_batches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    metadata["batch_id"],
                    "completed",
                    metadata["prediction_datetime"],
                    24,
                    metadata["started_at"],
                    metadata["completed_at"],
                    metadata["execution_time_ms"],
                    len(frame),
                    metadata["model_version"],
                    metadata.get("future_snapshot_version"),
                    json.dumps(metadata.get("warnings", [])),
                    None,
                ),
            )
            db.executemany(
                "INSERT INTO prediction_segments VALUES (?,?,?)",
                [
                    (
                        metadata["batch_id"],
                        str(row.road_segment_id),
                        json.dumps(row.to_dict(), default=str, ensure_ascii=False),
                    )
                    for _, row in frame.iterrows()
                ],
            )

    def latest(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT batch_id,status,prediction_datetime,horizon_hours,started_at,completed_at,execution_time_ms,segment_count,model_version,future_snapshot_version,warnings,error FROM prediction_batches WHERE status='completed' ORDER BY completed_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        keys = [
            "batchId",
            "status",
            "predictionDatetime",
            "horizonHours",
            "startedAt",
            "completedAt",
            "executionTimeMs",
            "segmentCount",
            "modelVersionId",
            "futureSnapshotVersion",
            "warnings",
            "error",
        ]
        value = dict(zip(keys, row))
        value["warnings"] = json.loads(value["warnings"] or "[]")
        return value

    def batch(self, batch_id: str) -> dict[str, Any] | None:
        latest = self.latest()
        if latest and latest["batchId"] == batch_id:
            return latest
        with self._connect() as db:
            row = db.execute(
                "SELECT batch_id,status,prediction_datetime,horizon_hours,started_at,completed_at,execution_time_ms,segment_count,model_version,future_snapshot_version,warnings,error FROM prediction_batches WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
        if row is None:
            return None
        keys = [
            "batchId",
            "status",
            "predictionDatetime",
            "horizonHours",
            "startedAt",
            "completedAt",
            "executionTimeMs",
            "segmentCount",
            "modelVersionId",
            "futureSnapshotVersion",
            "warnings",
            "error",
        ]
        value = dict(zip(keys, row))
        value["warnings"] = json.loads(value["warnings"] or "[]")
        return value

    def segment(
        self, road_segment_id: str, batch_id: str | None = None
    ) -> dict[str, Any] | None:
        batch = self.batch(batch_id) if batch_id else self.latest()
        if not batch:
            return None
        with self._connect() as db:
            row = db.execute(
                "SELECT result_json FROM prediction_segments WHERE batch_id=? AND road_segment_id=?",
                (batch["batchId"], str(road_segment_id)),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def top(self, limit: int) -> list[dict[str, Any]]:
        batch = self.latest()
        if not batch:
            return []
        with self._connect() as db:
            rows = db.execute(
                "SELECT result_json FROM prediction_segments WHERE batch_id=?",
                (batch["batchId"],),
            ).fetchall()
        values = [json.loads(row[0]) for row in rows]
        return sorted(values, key=lambda item: item.get("priority_rank", 10**9))[:limit]

    def get_prediction_segments_for_batch(self, batch_id: str) -> list[dict[str, Any]]:
        batch = self.batch(batch_id)
        if batch is None or batch["status"] != "completed":
            return []
        with self._connect() as db:
            rows = db.execute(
                "SELECT result_json FROM prediction_segments WHERE batch_id=? ORDER BY road_segment_id",
                (batch_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def create_action_plan(
        self,
        *,
        plan_id: str,
        batch_id: str,
        prediction_datetime: str,
        horizon_hours: int = 24,
        max_actions: int | None = None,
        minimum_priority: str | None = None,
        generated_at: str | None = None,
    ) -> None:
        """Create the durable running record used by city-plan orchestration."""
        now = generated_at or prediction_datetime
        with self._connect() as db:
            db.execute(
                "INSERT INTO city_action_plans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    plan_id,
                    batch_id,
                    "running",
                    prediction_datetime,
                    horizon_hours,
                    "city_action_plan_v1",
                    now,
                    max_actions,
                    minimum_priority,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "[]",
                    None,
                    now,
                    None,
                ),
            )

    def save_completed_action_plan(
        self,
        plan: dict[str, Any],
        *,
        max_actions: int | None = None,
        minimum_priority: str | None = None,
    ) -> None:
        safe = {
            key: value
            for key, value in plan.items()
            if key
            not in {
                "frozen_feature_vector",
                "raw_provider_payload",
                "api_key",
                "authorization",
                "local_file_path",
                "provider_secret",
            }
        }
        summary = safe.get("summary", {})
        now = safe.get("generated_at") or safe.get("prediction_datetime")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR REPLACE INTO city_action_plans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    safe["plan_id"],
                    safe["batch_id"],
                    "completed",
                    safe["prediction_datetime"],
                    safe.get("horizon_hours", 24),
                    safe.get("plan_version", "city_action_plan_v1"),
                    now,
                    max_actions,
                    minimum_priority,
                    summary.get("segments_analyzed"),
                    summary.get("candidate_segments"),
                    summary.get("groups_created"),
                    summary.get("actions_returned"),
                    json.dumps(safe, default=str, ensure_ascii=False),
                    json.dumps(safe.get("warnings", [])),
                    None,
                    now,
                    now,
                ),
            )

    def save_failed_action_plan(
        self, *, plan_id: str, batch_id: str, prediction_datetime: str, error: str
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO city_action_plans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    plan_id,
                    batch_id,
                    "failed",
                    prediction_datetime,
                    24,
                    "city_action_plan_v1",
                    prediction_datetime,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "[]",
                    error,
                    prediction_datetime,
                    None,
                ),
            )

    def get_action_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT status,plan_json,error FROM city_action_plans WHERE plan_id=?",
                (plan_id,),
            ).fetchone()
        if not row:
            return None
        return (
            json.loads(row[1])
            if row[1]
            else {"plan_id": plan_id, "status": row[0], "error": row[2]}
        ) | {"status": row[0]}

    def get_latest_action_plan(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT plan_id FROM city_action_plans WHERE status='completed' ORDER BY generated_at DESC LIMIT 1"
            ).fetchone()
        return self.get_action_plan(row[0]) if row else None

    def get_action_plan_for_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT plan_id FROM city_action_plans WHERE batch_id=? AND status='completed' ORDER BY generated_at DESC LIMIT 1",
                (batch_id,),
            ).fetchone()
        return self.get_action_plan(row[0]) if row else None
