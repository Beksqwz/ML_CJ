"""Stage 10A: isolated XGBoost benchmark against frozen final CatBoost models.

This script reads only the already prepared Stage 7B/7D splits.  It writes new
Stage 10 artifacts and never changes an existing model, preprocessing pipeline,
or production component.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260711  # Same seed used by the Stage 7 CatBoost training scripts.
MODELS = ROOT / "models" / "stage10_experiments"
REPORTS = ROOT / "reports" / "stage10"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} under {folder}")
    return matches[-1]


def source(horizon: str) -> tuple[dict[str, object], dict[str, Path], dict[str, object], str]:
    """Return the frozen feature config, splits and appropriate final CatBoost report."""
    if horizon == "1h":
        config_path = ROOT / "reports" / "stage7d" / "1h" / "stage7d_feature_config.json"
        split_dir = ROOT / "data" / "processed" / "stage7d"
        cat_report = load_json(ROOT / "reports" / "stage7d" / "1h" / "stage7d_weather_experiment_report.json")
        baseline = {
            "metrics": cat_report["experimental_metrics"],
            "calibration": {
                split: cat_report["experimental_metrics"][split]["calibration"]
                for split in ("validation", "test")
            },
            "best_iteration": cat_report["best_iteration"],
            "model_path": cat_report["model_path"],
        }
        name = "Stage 7D CatBoost weather experiment (final 1h model)"
    else:
        config_path = latest(ROOT / "reports" / "stage7a" / "24h", "*/training_dataset_24h_feature_config.json")
        split_dir = ROOT / "data" / "processed" / "stage7a"
        baseline = load_json(latest(ROOT / "reports" / "stage7b" / "24h", "*/catboost_24h_report.json"))
        name = "Stage 7B CatBoost (final 24h model)"
    config = load_json(config_path)
    return (
        config,
        {s: split_dir / f"training_dataset_{horizon}_{s}.parquet" for s in ("train", "validation", "test")},
        baseline,
        name,
    )


def encode_categories(splits: dict[str, pd.DataFrame], categorical: list[str]) -> dict[str, pd.DataFrame]:
    """Train-only ordinal encoding; unseen validation/test values get -1.

    This is required by XGBoost's numeric matrix interface and does not add,
    remove, reorder, or derive model features.
    """
    encoded = {name: frame.copy() for name, frame in splits.items()}
    for column in categorical:
        train_values = encoded["train"][column].astype("string").fillna("__MISSING__")
        codes = {value: i for i, value in enumerate(pd.unique(train_values))}
        for frame in encoded.values():
            values = frame[column].astype("string").fillna("__MISSING__")
            frame[column] = values.map(codes).fillna(-1).astype("int32")
    return encoded


def ece_10(y: np.ndarray, score: np.ndarray) -> tuple[float, list[dict[str, object]]]:
    bins = np.clip(np.digitize(score, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    ece = 0.0
    result = []
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any():
            predicted, observed = float(score[mask].mean()), float(y[mask].mean())
            ece += abs(predicted - observed) * int(mask.sum())
            result.append({"bin": bin_id, "rows": int(mask.sum()), "mean_prediction": predicted, "observed_rate": observed})
    return float(ece / len(y)), result


def measure(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = (score >= threshold).astype(np.int8)
    top = np.argsort(-score, kind="stable")[: max(1, int(np.ceil(0.10 * len(y))))]
    ece, bins = ece_10(y, score)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "lift_at_top_10pct": float(y[top].mean() / y.mean()),
        "brier_score": float(brier_score_loss(y, score)),
        "expected_calibration_error_10_bins": ece,
        "calibration_bins": bins,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def cat_value(baseline: dict[str, object], split: str, metric: str) -> float:
    if metric in {"brier_score", "expected_calibration_error_10_bins"}:
        return float(baseline["calibration"][split][metric])
    return float(baseline["metrics"][split][metric])


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Find maximum-F1 validation threshold in O(n log n), without test access."""
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order]
    true_positive = np.cumsum(sorted_y)
    predicted_positive = np.arange(1, len(y) + 1)
    false_positive = predicted_positive - true_positive
    false_negative = int(y.sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=float), where=denominator != 0)
    # A threshold changes only at the final item in each equal-score group.
    candidates = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(sorted_score) - 1]
    return float(sorted_score[candidates[np.argmax(f1[candidates])]])


def enrich_catboost_baseline(baseline, raw, feature_order, categorical, target):
    """Read the frozen CatBoost only to fill fields Stage 7D did not report."""
    model = CatBoostClassifier()
    model_path = Path(str(baseline["model_path"]))
    model.load_model(model_path)
    x = {name: frame.loc[:, feature_order].copy() for name, frame in raw.items()}
    for frame in x.values():
        for column in categorical:
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    y = {name: frame[target].to_numpy(np.int8) for name, frame in raw.items()}
    started = time.perf_counter()
    validation_score = model.predict_proba(x["validation"])[:, 1]
    validation_prediction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    test_score = model.predict_proba(x["test"])[:, 1]
    test_prediction_seconds = time.perf_counter() - started
    threshold = float(baseline.get("threshold", {}).get("value", best_f1_threshold(y["validation"], validation_score)))
    for split, score in (("validation", validation_score), ("test", test_score)):
        calculated = measure(y[split], score, threshold)
        baseline.setdefault("metrics", {}).setdefault(split, {}).update({key: calculated[key] for key in ("roc_auc", "precision", "recall", "f1")})
        baseline.setdefault("calibration", {}).setdefault(split, {}).update({"brier_score": calculated["brier_score"], "expected_calibration_error_10_bins": calculated["expected_calibration_error_10_bins"]})
    return {"prediction_validation_seconds": validation_prediction_seconds, "prediction_test_seconds": test_prediction_seconds, "model_size_bytes": model_path.stat().st_size}


def run(horizon: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    config, paths, catboost, catboost_name = source(horizon)
    feature_order = list(config["numerical_features"]) + list(config["categorical_features"])
    categorical = list(config["categorical_features"])
    expected_count = 97 if horizon == "1h" else 77
    if len(feature_order) != expected_count or len(feature_order) != len(set(feature_order)):
        raise ValueError(f"{horizon}: feature config does not match the required {expected_count} unique features")
    raw = {name: pd.read_parquet(path) for name, path in paths.items()}
    for name, frame in raw.items():
        missing = set(feature_order + [config["target_column"]]) - set(frame.columns)
        if missing:
            raise ValueError(f"{horizon}/{name} missing columns: {sorted(missing)}")
    catboost_resources = enrich_catboost_baseline(catboost, raw, feature_order, categorical, str(config["target_column"]))
    prepared = encode_categories(raw, categorical)
    target = str(config["target_column"])
    x = {name: frame.loc[:, feature_order] for name, frame in prepared.items()}
    y = {name: frame[target].to_numpy(np.int8) for name, frame in prepared.items()}
    params = {
        "objective": "binary:logistic", "eval_metric": "aucpr", "n_estimators": 1500,
        "learning_rate": 0.05, "max_depth": 7, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0,
        "random_state": SEED, "n_jobs": -1, "tree_method": "hist",
        "early_stopping_rounds": 100,
    }
    model = XGBClassifier(**params)
    started = time.perf_counter()
    model.fit(x["train"], y["train"], eval_set=[(x["validation"], y["validation"])], verbose=False)
    training_seconds = time.perf_counter() - started
    threshold_scores = model.predict_proba(x["validation"])[:, 1]
    # The validation split alone determines the F1 threshold, matching Stage 7B's protocol.
    threshold = best_f1_threshold(y["validation"], threshold_scores)
    prediction_seconds: dict[str, float] = {}
    scores: dict[str, np.ndarray] = {"validation": threshold_scores}
    for split in ("validation", "test"):
        started = time.perf_counter()
        # Make a fresh inference call for an honest per-split prediction benchmark.
        score = model.predict_proba(x[split])[:, 1]
        prediction_seconds[split] = time.perf_counter() - started
        scores[split] = score
    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / f"xgboost_{horizon}.json"
    model.save_model(model_path)
    importance = [
        {"feature": feature, "importance_gain": float(value)}
        for feature, value in sorted(zip(feature_order, model.feature_importances_), key=lambda item: item[1], reverse=True)
    ]
    metrics = {split: measure(y[split], scores[split], threshold) for split in ("validation", "test")}
    validation_test_gap = metrics["validation"]["pr_auc"] - metrics["test"]["pr_auc"]
    cat_gap = cat_value(catboost, "validation", "pr_auc") - cat_value(catboost, "test", "pr_auc")
    test_ranking = all(metrics["test"][key] >= cat_value(catboost, "test", key) - 1e-12 for key in ("pr_auc", "lift_at_top_10pct"))
    calibration = all(metrics["test"][key] <= cat_value(catboost, "test", key) + 1e-12 for key in ("brier_score", "expected_calibration_error_10_bins"))
    no_overfitting = validation_test_gap <= max(0.0, cat_gap) + 1e-12
    winner = "XGBoost" if test_ranking and calibration and no_overfitting else "CatBoost"
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(), "stage": "10A", "horizon": horizon,
        "isolation": "Independent experiment; no existing pipeline, Stage 7 assets, CatBoost, SHAP, inference, API, or final models were modified.",
        "dataset": {"split_source": "Stage 7D weather experiment" if horizon == "1h" else "Stage 7B final dataset", "paths": {k: str(v.resolve()) for k, v in paths.items()}},
        "feature_contract": {"feature_count": len(feature_order), "feature_order": feature_order, "categorical_encoding": "Train-only ordinal mapping; unknown validation/test categories map to -1; feature set and order are unchanged."},
        "parameters": params, "early_stopping": {"split": "validation", "best_iteration": int(model.best_iteration), "best_score_aucpr": float(model.best_score)},
        "threshold": {"rule": "maximum validation F1", "value": threshold}, "metrics": metrics,
        "timing_seconds": {"training": training_seconds, "prediction": prediction_seconds}, "model": {"path": str(model_path.resolve()), "size_bytes": model_path.stat().st_size},
        "feature_importance_gain": importance, "catboost_baseline": {"name": catboost_name, "model_path": catboost.get("model_path"), "metrics": {split: {metric: cat_value(catboost, split, metric) for metric in ("pr_auc", "roc_auc", "lift_at_top_10pct", "brier_score", "expected_calibration_error_10_bins")} for split in ("validation", "test")}, "resources": {"training_time_seconds": None, "training_time_note": "Not recorded in frozen Stage 7 report; retraining is intentionally prohibited for this isolated experiment.", **catboost_resources}},
        "decision": {"winner": winner, "test_pr_auc_not_worse": test_ranking, "test_calibration_not_worse": calibration, "no_overfitting": no_overfitting, "validation_test_pr_auc_gap": validation_test_gap, "catboost_validation_test_pr_auc_gap": cat_gap,
            "reason_ru": "XGBoost удовлетворяет всем критериям победы." if winner == "XGBoost" else "XGBoost отклонён: не выполнены все обязательные тестовые критерии (ранжирование, калибровка и отсутствие признаков переобучения)."},
    }
    rows = []
    for split in ("validation", "test"):
        for metric in ("pr_auc", "roc_auc", "lift_at_top_10pct", "brier_score", "expected_calibration_error_10_bins"):
            rows.append({"horizon": horizon, "split": split, "metric": metric, "catboost": cat_value(catboost, split, metric), "xgboost": metrics[split][metric]})
    for metric, cb, xgb in (("training_time_seconds", None, training_seconds), ("prediction_time_validation_seconds", catboost_resources["prediction_validation_seconds"], prediction_seconds["validation"]), ("prediction_time_test_seconds", catboost_resources["prediction_test_seconds"], prediction_seconds["test"]), ("model_size_bytes", catboost_resources["model_size_bytes"], model_path.stat().st_size)):
        rows.append({"horizon": horizon, "split": "resource", "metric": metric, "catboost": cb, "xgboost": xgb})
    return report, rows


def main() -> None:
    reports, rows = {}, []
    for horizon in ("1h", "24h"):
        report, comparison_rows = run(horizon)
        reports[horizon] = report
        rows.extend(comparison_rows)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"xgboost_report_{horizon}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = {"generated_at_utc": datetime.now(UTC).isoformat(), "stage": "10A", "comparison_rows": rows, "winner_by_horizon": {h: reports[h]["decision"] for h in reports}, "overall_conclusion_ru": "CatBoost remains recommended as the current final model unless XGBoost wins every required criterion for a horizon."}
    (REPORTS / "comparison_catboost_vs_xgboost.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(REPORTS / "comparison_catboost_vs_xgboost.csv", index=False)


if __name__ == "__main__":
    main()
