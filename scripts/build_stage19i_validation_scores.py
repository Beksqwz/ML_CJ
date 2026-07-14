"""Build aligned, non-production Stage 19I validation score vectors."""

from __future__ import annotations

import hashlib
import json
import time
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models" / "stage19h"
VALIDATION = (
    ROOT / "data" / "processed" / "stage7a" / "training_dataset_24h_validation.parquet"
)
OUTPUT = ROOT / "data" / "audit" / "stage19i_simple" / "validation_scores.parquet"
REPORT = ROOT / "reports" / "stage19i_simple" / "validation_alignment.json"


def digest(value: object) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(value, index=False).values.tobytes()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = json.loads(
        next(
            (ROOT / "reports" / "stage7a").rglob(
                "training_dataset_24h_feature_config.json"
            )
        ).read_text()
    )
    features = [*config["numerical_features"], *config["categorical_features"]]
    frame = pd.read_parquet(VALIDATION)
    cat = CatBoostClassifier()
    cat.load_model(MODELS / "catboost_candidate.cbm")
    pre = joblib.load(MODELS / "train_only_preprocessor_v2.joblib")
    numeric = pre.transform(frame)
    xgb = XGBClassifier()
    xgb.load_model(MODELS / "xgboost_candidate.json")
    hgb = joblib.load(MODELS / "hist_gradient_boosting_candidate.joblib")
    log = joblib.load(MODELS / "logistic_regression_candidate.joblib")
    lgbm = lgb.Booster(model_file=str(MODELS / "lightgbm_candidate.txt"))
    cat_frame = frame.loc[:, features].copy()
    for column in config["categorical_features"]:
        cat_frame[column] = (
            cat_frame[column].astype("string").fillna("__MISSING__").astype(str)
        )
    started = time.perf_counter()
    scores = {
        "score_catboost_stage19h": cat.predict_proba(cat_frame)[:, 1],
        "score_lightgbm": lgbm.predict(numeric),
        "score_xgboost": xgb.predict_proba(numeric)[:, 1],
        "score_hist_gradient_boosting": hgb.predict_proba(numeric)[:, 1],
        "score_logistic_regression": log.predict_proba(numeric)[:, 1],
    }
    output = pd.DataFrame(
        {
            "row_id": np.arange(len(frame), dtype=np.int64),
            "datetime_hour": frame.datetime_hour,
            "target_24h": frame.target_24h,
            **scores,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False)
    checksums = {name: digest(output[name]) for name in output.columns}
    report = {
        "source": str(VALIDATION),
        "source_sha256": file_hash(VALIDATION),
        "row_count": len(output),
        "feature_count": len(features),
        "score_columns": list(scores),
        "checksums": checksums,
        "missing_values": output.isna().sum().to_dict(),
        "finite_scores": {
            name: bool(np.isfinite(output[name]).all()) for name in scores
        },
        "row_id_consecutive": bool(
            np.array_equal(output.row_id.to_numpy(), np.arange(len(output)))
        ),
        "runtime_seconds": time.perf_counter() - started,
        "model_hashes": {p.name: file_hash(p) for p in MODELS.glob("*candidate.*")},
        "warnings": [
            "row_id_is_artifact_local_not_road_identity",
            "logistic_regression_max_iter_300_convergence_warning",
        ],
        "ready_for_validation_only_weight_search": True,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT.with_suffix(".md")).write_text(
        "# Stage 19I validation alignment\n\n" + json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
