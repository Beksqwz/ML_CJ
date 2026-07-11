"""Train one Stage 7B CatBoost model for a prepared 1h or 24h split set.

This script deliberately does not run SHAP or a recommendation engine.
GPU is attempted first and falls back to CPU if CatBoost cannot use it.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = PROJECT_ROOT / "data" / "processed" / "stage7a"
MODELS_ROOT = PROJECT_ROOT / "models" / "stage7b"
PREDICTIONS_ROOT = PROJECT_ROOT / "data" / "processed" / "stage7b" / "predictions"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "stage7b"
SEED = 20260711


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one CatBoost Stage 7B model with temporal Stage 7A splits.")
    parser.add_argument("--label", required=True, choices=("1h", "24h"))
    parser.add_argument("--iterations", type=int, default=1500)
    return parser.parse_args()


def load_feature_config(label: str) -> tuple[dict[str, object], Path]:
    candidates = sorted((PROJECT_ROOT / "reports" / "stage7a" / label).glob("*/training_dataset_*_feature_config.json"))
    if not candidates:
        raise FileNotFoundError(f"No Stage 7A feature config found for {label}.")
    path = candidates[-1]
    return json.loads(path.read_text(encoding="utf-8")), path


def load_baseline_report(label: str) -> tuple[dict[str, object], Path]:
    candidates = sorted((PROJECT_ROOT / "reports" / "stage7a" / label).glob("*/training_dataset_*_stage7a_report.json"))
    if not candidates:
        raise FileNotFoundError(f"No Stage 7A report found for {label}.")
    path = candidates[-1]
    return json.loads(path.read_text(encoding="utf-8")), path


def prepare_features(frame: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    result = frame[features].copy()
    for column in categorical:
        # CatBoost accepts string categories; numeric NaN values remain NaN by design.
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result


def pr_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    groups = pd.DataFrame({"score": score, "target": y}).groupby("score")["target"].agg(["count", "sum"]).sort_index(ascending=False)
    cumulative_positive = groups["sum"].cumsum()
    precision = cumulative_positive / groups["count"].cumsum()
    recall = cumulative_positive / positives
    return float((precision * recall.diff().fillna(recall)).sum())


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    groups = pd.DataFrame({"score": score, "target": y}).groupby("score")["target"].agg(["count", "sum"]).sort_index(ascending=False)
    tp = groups["sum"].cumsum().to_numpy(dtype=float)
    fp = (groups["count"].cumsum() - groups["sum"].cumsum()).to_numpy(dtype=float)
    fn = float(y.sum()) - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    index = int(np.argmax(f1))
    return float(groups.index.to_numpy(dtype=float)[index]), float(f1[index])


def metric_summary(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = score >= threshold
    tp = int((predicted & (y == 1)).sum())
    fp = int((predicted & (y == 0)).sum())
    fn = int(((~predicted) & (y == 1)).sum())
    tn = int(((~predicted) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    top_n = max(1, int(np.ceil(len(y) * 0.10)))
    top = np.argsort(-score, kind="stable")[:top_n]
    prevalence = float(y.mean())
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_rate": prevalence,
        "pr_auc": pr_auc(y, score),
        "roc_auc": roc_auc(y, score),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_at_top_10pct": float(y[top].sum() / y.sum()) if y.sum() else None,
        "lift_at_top_10pct": float(y[top].mean() / prevalence) if prevalence else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def calibration_summary(y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.digitize(score, edges, right=True) - 1, 0, 9)
    rows: list[dict[str, object]] = []
    weighted_error = 0.0
    for bin_index in range(10):
        mask = bins == bin_index
        if not mask.any():
            rows.append({"bin": bin_index, "rows": 0, "mean_prediction": None, "observed_rate": None})
            continue
        mean_prediction = float(score[mask].mean())
        observed_rate = float(y[mask].mean())
        weighted_error += abs(mean_prediction - observed_rate) * int(mask.sum())
        rows.append({"bin": bin_index, "rows": int(mask.sum()), "mean_prediction": mean_prediction, "observed_rate": observed_rate})
    return {"brier_score": float(np.mean((score - y) ** 2)), "expected_calibration_error_10_bins": weighted_error / len(y), "bins": rows}


def comparison(model_metrics: dict[str, object], baseline_metrics: dict[str, object]) -> dict[str, object]:
    keys = ("pr_auc", "roc_auc", "precision", "recall", "f1", "recall_at_top_10pct", "lift_at_top_10pct")
    result: dict[str, object] = {}
    for key in keys:
        model_value = model_metrics.get(key)
        baseline_value = baseline_metrics.get(key)
        if model_value is None or baseline_value is None:
            result[key] = {"absolute": None, "relative_percent": None}
            continue
        absolute = float(model_value) - float(baseline_value)
        result[key] = {"absolute": absolute, "relative_percent": absolute / float(baseline_value) * 100 if baseline_value else None}
    return result


def train_with_gpu_fallback(params: dict[str, object], train_x: pd.DataFrame, train_y: np.ndarray, valid_x: pd.DataFrame, valid_y: np.ndarray, cat_indexes: list[int]) -> tuple[CatBoostClassifier, dict[str, object]]:
    gpu_params = params | {"task_type": "GPU", "devices": "0"}
    try:
        model = CatBoostClassifier(**gpu_params)
        model.fit(train_x, train_y, cat_features=cat_indexes, eval_set=(valid_x, valid_y), early_stopping_rounds=100, verbose=100)
        return model, {"device": "GPU", "fallback_reason": None, "effective_params": gpu_params}
    except (CatBoostError, RuntimeError) as exc:
        cpu_params = params | {"task_type": "CPU", "thread_count": -1}
        model = CatBoostClassifier(**cpu_params)
        model.fit(train_x, train_y, cat_features=cat_indexes, eval_set=(valid_x, valid_y), early_stopping_rounds=100, verbose=100)
        return model, {"device": "CPU", "fallback_reason": str(exc), "effective_params": cpu_params}


def save_predictions(label: str, split: str, frame: pd.DataFrame, target: str, probability: np.ndarray) -> str:
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({"datetime_hour": frame["datetime_hour"], target: frame[target].astype("int8"), "prediction_probability": probability})
    path = PREDICTIONS_ROOT / f"training_dataset_{label}_{split}_predictions.parquet"
    output.to_parquet(path, index=False, engine="pyarrow")
    return str(path.resolve())


def main() -> int:
    args = parse_args()
    label = args.label
    target = f"target_{label}"
    config, config_path = load_feature_config(label)
    baseline_report, baseline_path = load_baseline_report(label)
    features = [*config["numerical_features"], *config["categorical_features"]]
    categorical = list(config["categorical_features"])
    cat_indexes = [features.index(column) for column in categorical]
    splits = {name: pd.read_parquet(SPLIT_ROOT / f"training_dataset_{label}_{name}.parquet") for name in ("train", "validation", "test")}
    for name, frame in splits.items():
        expected = {"datetime_hour", target, *features}
        missing = expected - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split lacks columns: {sorted(missing)}")
    train_x = prepare_features(splits["train"], features, categorical)
    valid_x = prepare_features(splits["validation"], features, categorical)
    test_x = prepare_features(splits["test"], features, categorical)
    train_y = splits["train"][target].to_numpy(dtype=np.int8)
    valid_y = splits["validation"][target].to_numpy(dtype=np.int8)
    test_y = splits["test"][target].to_numpy(dtype=np.int8)
    params: dict[str, object] = {
        "iterations": args.iterations,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 5.0,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": SEED,
        "allow_writing_files": False,
    }
    model, training = train_with_gpu_fallback(params, train_x, train_y, valid_x, valid_y, cat_indexes)
    prediction = {"train": model.predict_proba(train_x)[:, 1], "validation": model.predict_proba(valid_x)[:, 1], "test": model.predict_proba(test_x)[:, 1]}
    threshold, validation_f1 = best_f1_threshold(valid_y, prediction["validation"])
    model_metrics = {name: metric_summary(labels, prediction[name], threshold) for name, labels in (("train", train_y), ("validation", valid_y), ("test", test_y))}
    calibration = {name: calibration_summary(labels, prediction[name]) for name, labels in (("validation", valid_y), ("test", test_y))}
    predictions = {name: save_predictions(label, name, splits[name], target, prediction[name]) for name in ("validation", "test")}
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_ROOT / f"catboost_{label}.cbm"
    model.save_model(model_path)
    report_dir = REPORTS_ROOT / label / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    feature_path = report_dir / f"catboost_{label}_feature_list.json"
    params_path = report_dir / f"catboost_{label}_parameters.json"
    threshold_path = report_dir / f"catboost_{label}_threshold.json"
    report_path = report_dir / f"catboost_{label}_report.json"
    feature_path.write_text(json.dumps({"features": features, "categorical_features": categorical, "numerical_features": config["numerical_features"], "excluded_features": config["excluded_from_model_features"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    params_path.write_text(json.dumps(training, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    threshold_path.write_text(json.dumps({"rule": "Maximum F1 selected exclusively on validation probabilities.", "threshold": threshold, "validation_f1_at_threshold": validation_f1}, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline_metrics = baseline_report["baseline"]["metrics"]
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "horizon": label,
        "stage7a_feature_config": str(config_path.resolve()),
        "stage7a_baseline_report": str(baseline_path.resolve()),
        "model_path": str(model_path.resolve()),
        "device": training["device"],
        "gpu_fallback_reason": training["fallback_reason"],
        "best_iteration": int(model.get_best_iteration()),
        "best_score": model.get_best_score(),
        "threshold": {"rule": "maximum validation F1", "value": threshold},
        "metrics": model_metrics,
        "calibration": calibration,
        "baseline_comparison": {name: comparison(model_metrics[name], baseline_metrics[name]) for name in ("validation", "test")},
        "overfitting_check": {
            "train_validation_pr_auc_gap": float(model_metrics["train"]["pr_auc"] - model_metrics["validation"]["pr_auc"]),
            "validation_test_pr_auc_gap": float(model_metrics["validation"]["pr_auc"] - model_metrics["test"]["pr_auc"]),
            "train_validation_roc_auc_gap": float(model_metrics["train"]["roc_auc"] - model_metrics["validation"]["roc_auc"]),
            "interpretation": "Large positive train-validation gaps indicate possible overfitting; validation-test gaps indicate temporal generalization drift.",
        },
        "prediction_files": predictions,
        "training_exclusions": config["excluded_from_model_features"],
        "not_run": ["SHAP", "recommendation engine", "git commit"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Horizon: {label}")
    print(f"Device: {training['device']}")
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Validation PR-AUC: {model_metrics['validation']['pr_auc']}")
    print(f"Test PR-AUC: {model_metrics['test']['pr_auc']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
