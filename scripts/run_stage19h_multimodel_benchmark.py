"""Fair, audit-only Stage 19H natural-prevalence tabular benchmark."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from ml_service.preprocessing.train_only import TrainOnlyPreprocessor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_24h_natural_prevalence import (  # noqa: E402
    CONFIG_PATH,
    EXPECTED_HASH,
    PROCESSED,
    build_feature_context,
    build_features_for_chunk,
    feature_hash,
    target_fields,
)
from run_stage19f_hotspot_baseline import accident_indexes, historical_scores  # noqa: E402


SEED = 20260711
MODELS = ROOT / "models" / "stage19h"
REPORTS = ROOT / "reports" / "stage19h"
SPLITS = ROOT / "data" / "processed" / "stage7a"
MODEL_PATH = ROOT / "models" / "production" / "catboost_24h.cbm"
WINDOWS = {
    "autumn": pd.Timestamp("2024-09-30 19:00:00"),
    "winter": pd.Timestamp("2025-01-01 00:00:00"),
    "spring": pd.Timestamp("2025-04-01 00:00:00"),
}
HOURS, CHUNK_HOURS = 168, 6


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    hashed = pd.util.hash_pandas_object(frame.loc[:, columns], index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


class _LegacyTrainOnlyPreprocessor:
    """Train-only ordinal categories and numeric medians for non-CatBoost models."""

    def __init__(self, numeric: list[str], categorical: list[str]) -> None:
        self.numeric, self.categorical = numeric, categorical
        self.medians: dict[str, float] = {}
        self.codebooks: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyPreprocessor":
        self.medians = {
            column: float(pd.to_numeric(frame[column], errors="coerce").median())
            for column in self.numeric
        }
        self.codebooks = {}
        for column in self.categorical:
            values = frame[column].astype("string").fillna("__MISSING__")
            self.codebooks[column] = {
                str(value): index for index, value in enumerate(pd.unique(values))
            }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in self.numeric:
            result[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(self.medians[column])
                .astype("float32")
            )
        for column in self.categorical:
            values = frame[column].astype("string").fillna("__MISSING__").astype(str)
            result[column] = (
                values.map(self.codebooks[column]).fillna(-1).astype("float32")
            )
        return result.loc[:, [*self.numeric, *self.categorical]]


def catboost_frame(
    frame: pd.DataFrame, features: list[str], categorical: list[str]
) -> pd.DataFrame:
    result = frame.loc[:, features].copy()
    for column in categorical:
        result[column] = (
            result[column].astype("string").fillna("__MISSING__").astype(str)
        )
    return result


def per_hour_records(
    frame: pd.DataFrame, score: np.ndarray
) -> tuple[list[dict[str, float]], set[str], dict[str, int]]:
    table = frame[["road_segment_id", "prediction_datetime", "target_24h"]].copy()
    table["score"] = score
    records, seen, frequency = [], set(), {}
    for _, hour in table.groupby("prediction_datetime", sort=True):
        ranked = hour.sort_values(
            ["score", "road_segment_id"], ascending=[False, True], kind="stable"
        )
        positives = int(ranked.target_24h.sum())
        row: dict[str, float] = {"positives": float(positives)}
        for size in (10, 20, 50, 40, 199, 397):
            values = ranked.head(size).target_24h.to_numpy(dtype=np.int8)
            precision = float(values.mean())
            row[f"precision_{size}"] = precision
            row[f"recall_{size}"] = (
                float(values.sum() / positives) if positives else np.nan
            )
            row[f"lift_{size}"] = (
                precision / (positives / len(ranked)) if positives else np.nan
            )
        for segment in ranked.head(10).road_segment_id.astype(str):
            seen.add(segment)
            frequency[segment] = frequency.get(segment, 0) + 1
        records.append(row)
    return records, seen, frequency


def aggregate_records(records: list[dict[str, float]]) -> dict[str, float]:
    mean = pd.DataFrame(records).mean(numeric_only=True)
    return {
        "precision_at_10": float(mean["precision_10"]),
        "precision_at_20": float(mean["precision_20"]),
        "precision_at_50": float(mean["precision_50"]),
        "recall_at_1pct": float(mean["recall_40"]),
        "recall_at_5pct": float(mean["recall_199"]),
        "recall_at_10pct": float(mean["recall_397"]),
        "lift_at_1pct": float(mean["lift_40"]),
        "lift_at_5pct": float(mean["lift_199"]),
        "lift_at_10pct": float(mean["lift_397"]),
    }


def metrics(
    labels: np.ndarray, scores: np.ndarray, records: list[dict[str, float]]
) -> dict[str, float]:
    return {
        "natural_prevalence": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "pr_auc_over_prevalence": float(
            average_precision_score(labels, scores) / labels.mean()
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        **aggregate_records(records),
    }


def load_splits(features: list[str], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [*features, target]
    return (
        pd.read_parquet(SPLITS / "training_dataset_24h_train.parquet", columns=columns),
        pd.read_parquet(
            SPLITS / "training_dataset_24h_validation.parquet", columns=columns
        ),
    )


def train_candidates(
    features: list[str], numeric: list[str], categorical: list[str], target: str
) -> tuple[dict[str, object], dict[str, dict]]:
    MODELS.mkdir(parents=True, exist_ok=True)
    train, validation = load_splits(features, target)
    audit = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_checksum": frame_hash(train, [*features, target]),
        "validation_checksum": frame_hash(validation, [*features, target]),
        "target_train": train[target].value_counts().to_dict(),
        "target_validation": validation[target].value_counts().to_dict(),
        "feature_count": len(features),
        "categorical_features": categorical,
    }
    preprocessor = TrainOnlyPreprocessor(numeric, categorical).fit(train)
    joblib.dump(preprocessor, MODELS / "train_only_preprocessor.joblib")
    x_train_num = preprocessor.transform(train)
    y_train, y_validation = (
        train[target].to_numpy(np.int8),
        validation[target].to_numpy(np.int8),
    )
    candidates: dict[str, object] = {}
    summary: dict[str, dict] = {}

    def record(name: str, model: object, path: Path, started: float, predict) -> None:
        score = predict(validation)
        summary[name] = {
            "validation_pr_auc": float(average_precision_score(y_validation, score)),
            "validation_roc_auc": float(roc_auc_score(y_validation, score)),
            "training_seconds": time.perf_counter() - started,
            "model_path": str(path),
            "model_size_bytes": path.stat().st_size,
            "tuning_trials": 1,
            "tuning_policy": "one fixed validation trial; identical bounded budget for CatBoost, LightGBM and XGBoost",
        }
        candidates[name] = model

    started = time.perf_counter()
    cat = CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=SEED,
        allow_writing_files=False,
        thread_count=1,
    )
    cat.fit(
        catboost_frame(train, features, categorical),
        y_train,
        cat_features=categorical,
        eval_set=(catboost_frame(validation, features, categorical), y_validation),
        early_stopping_rounds=75,
        verbose=False,
    )
    cat_path = MODELS / "catboost_candidate.cbm"
    cat.save_model(cat_path)
    record(
        "catboost_candidate",
        cat,
        cat_path,
        started,
        lambda frame: cat.predict_proba(catboost_frame(frame, features, categorical))[
            :, 1
        ],
    )
    del train

    boosting = {
        "lightgbm": (
            LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=7,
                min_child_samples=50,
                reg_lambda=5.0,
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            ),
            MODELS / "lightgbm_candidate.txt",
        ),
        "xgboost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                random_state=SEED,
                n_jobs=-1,
                tree_method="hist",
            ),
            MODELS / "xgboost_candidate.json",
        ),
    }
    for name, (model, path) in boosting.items():
        started = time.perf_counter()
        model.fit(x_train_num, y_train)
        if name == "lightgbm":
            model.booster_.save_model(str(path))
        else:
            model.save_model(path)
        record(
            name,
            model,
            path,
            started,
            lambda frame, m=model: m.predict_proba(preprocessor.transform(frame))[:, 1],
        )
    for name, model, path in (
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=63,
                l2_regularization=5.0,
                random_state=SEED,
            ),
            MODELS / "hist_gradient_boosting_candidate.joblib",
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1,
                class_weight=None,
            ),
            MODELS / "extra_trees_candidate.joblib",
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=300, C=1.0, solver="lbfgs", n_jobs=-1, random_state=SEED
            ),
            MODELS / "logistic_regression_candidate.joblib",
        ),
    ):
        started = time.perf_counter()
        model.fit(x_train_num, y_train)
        joblib.dump(model, path)
        record(
            name,
            model,
            path,
            started,
            lambda frame, m=model: m.predict_proba(preprocessor.transform(frame))[:, 1],
        )
    return candidates, {
        "input_audit": audit,
        "training": summary,
        "preprocessor": preprocessor,
    }


def evaluate_windows(
    candidates: dict[str, object],
    preprocessor: TrainOnlyPreprocessor,
    features: list[str],
    categorical: list[str],
    config: dict,
    target: str,
) -> tuple[dict, dict]:
    ready = pd.read_parquet(PROCESSED / "accidents_with_roads_ml_ready.parquet")
    indexes = accident_indexes()
    production = CatBoostClassifier()
    production.load_model(MODEL_PATH)
    all_models = {"stage7b_frozen_reference": production, **candidates}
    window_metrics, stability = {}, {}
    for window_name, start in WINDOWS.items():
        hours = pd.date_range(start, periods=HOURS, freq="h")
        context = build_feature_context(ready, config, hours)
        labels: list[np.ndarray] = []
        scores = {name: [] for name in [*all_models, "historical_baseline"]}
        records = {name: [] for name in scores}
        top_sets = {name: set() for name in scores}
        top_frequency = {name: {} for name in scores}
        for begin in range(0, HOURS, CHUNK_HOURS):
            chunk_hours = hours[begin : begin + CHUNK_HOURS]
            grid = pd.MultiIndex.from_product(
                [chunk_hours, context.segment_order],
                names=["prediction_datetime", "road_segment_id"],
            ).to_frame(index=False)
            raw = build_features_for_chunk(grid, context, config)
            targets = target_fields(
                raw[["road_segment_id", "prediction_datetime"]], context.counts
            )
            frame = raw[["road_segment_id", "prediction_datetime"]].merge(
                targets,
                on=["road_segment_id", "prediction_datetime"],
                validate="one_to_one",
            )
            numeric = preprocessor.transform(raw)
            raw_cat = catboost_frame(raw, features, categorical)
            chunk_scores = {
                "stage7b_frozen_reference": production.predict_proba(
                    raw_cat, thread_count=1
                )[:, 1],
                "catboost_candidate": candidates["catboost_candidate"].predict_proba(
                    raw_cat, thread_count=1
                )[:, 1],
                "lightgbm": candidates["lightgbm"].predict_proba(numeric)[:, 1],
                "xgboost": candidates["xgboost"].predict_proba(numeric)[:, 1],
                "hist_gradient_boosting": candidates[
                    "hist_gradient_boosting"
                ].predict_proba(numeric)[:, 1],
                "extra_trees": candidates["extra_trees"].predict_proba(numeric)[:, 1],
                "logistic_regression": candidates["logistic_regression"].predict_proba(
                    numeric
                )[:, 1],
                "historical_baseline": historical_scores(frame, indexes),
            }
            labels.append(frame.target_24h.to_numpy(np.uint8))
            for name, score in chunk_scores.items():
                rec, observed, frequency = per_hour_records(frame, score)
                scores[name].append(score.astype("float32"))
                records[name].extend(rec)
                top_sets[name].update(observed)
                for segment, count in frequency.items():
                    top_frequency[name][segment] = (
                        top_frequency[name].get(segment, 0) + count
                    )
        y = np.concatenate(labels)
        window_metrics[window_name] = {
            name: metrics(y, np.concatenate(value), records[name])
            for name, value in scores.items()
        }
        stability[window_name] = {
            name: {
                "unique_top10_segments": len(top_sets[name]),
                "permanently_top10_segments": sum(
                    count == HOURS for count in top_frequency[name].values()
                ),
            }
            for name in top_sets
        }
    return window_metrics, stability


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    numeric = list(config["numerical_features"])
    categorical = list(config["categorical_features"])
    features = [*numeric, *categorical]
    target = str(config["target_column"])
    if len(features) != 77 or feature_hash(features) != EXPECTED_HASH:
        raise ValueError("frozen_77_feature_contract_failed")
    if (
        file_hash(MODEL_PATH)
        != "0c8e1b88b1cfaf95fb39e395e2fdc54f1b7abda22d8ac00e1d6f561ab9110a0c"
    ):
        raise ValueError("production_model_hash_changed")
    candidates, trained = train_candidates(features, numeric, categorical, target)
    seasonal, stability = evaluate_windows(
        candidates, trained["preprocessor"], features, categorical, config, target
    )
    means = {
        model: {
            metric: float(
                np.mean([seasonal[window][model][metric] for window in WINDOWS])
            )
            for metric in seasonal["autumn"][model]
        }
        for model in seasonal["autumn"]
    }
    primary = {
        name: values["recall_at_5pct"]
        for name, values in means.items()
        if name not in {"historical_baseline", "stage7b_frozen_reference"}
    }
    winner = max(primary, key=primary.get)
    frozen = means["stage7b_frozen_reference"]
    replacement = (
        primary[winner] >= frozen["recall_at_5pct"] * 1.03
        and means[winner]["lift_at_5pct"] >= frozen["lift_at_5pct"]
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    dump(REPORTS / "input_audit.json", trained["input_audit"])
    dump(
        REPORTS / "environment.json",
        {
            "python": platform.python_version(),
            "catboost": __import__("catboost").__version__,
            "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "transformer": "skipped: PyTorch/pytorch_tabular unavailable",
        },
    )
    dump(REPORTS / "training_summary.json", trained["training"])
    dump(REPORTS / "seasonal_natural_metrics.json", seasonal)
    dump(REPORTS / "top10_stability_comparison.json", stability)
    dump(REPORTS / "operational_ranking_comparison.json", means)
    dump(
        REPORTS / "baseline_comparison.json",
        {
            "historical_baseline": means["historical_baseline"],
            "models": {
                name: values
                for name, values in means.items()
                if name != "historical_baseline"
            },
        },
    )
    dump(REPORTS / "efficiency_comparison.json", trained["training"])
    dump(
        REPORTS / "pareto_analysis.json",
        {
            "note": "No automatic production replacement; compare mean Recall@5%, Lift@5%, PR-AUC/prevalence, ROC-AUC, size and latency."
        },
    )
    dump(
        REPORTS / "winner_decision.json",
        {
            "best_experimental_model": winner,
            "replacement_criteria_met": replacement,
            "production_stage7b_remains_unchanged": True,
            "hybrid_recommendation": "Evaluate a later hotspot-plus-ML hybrid because baseline remains a required operational comparator.",
        },
    )
    summary = {
        "mean_metrics": means,
        "winner": winner,
        "replacement_criteria_met": replacement,
        "windows": {name: str(start) for name, start in WINDOWS.items()},
    }
    dump(REPORTS / "summary.json", summary)
    (REPORTS / "summary.md").write_text(
        "# Stage 19H\n\n" + json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
