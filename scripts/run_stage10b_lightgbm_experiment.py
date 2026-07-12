"""Stage 10B: isolated LightGBM benchmark using the frozen Stage 7 splits only."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260711
MODELS = ROOT / "models" / "stage10_experiments"
REPORTS = ROOT / "reports" / "stage10"
METRICS = ("pr_auc", "roc_auc", "lift_at_top_10pct", "brier_score", "expected_calibration_error_10_bins")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest(folder: Path, pattern: str) -> Path:
    found = sorted(folder.glob(pattern))
    if not found:
        raise FileNotFoundError(f"No {pattern} under {folder}")
    return found[-1]


def load_source(horizon: str):
    if horizon == "1h":
        config = read_json(ROOT / "reports" / "stage7d" / "1h" / "stage7d_feature_config.json")
        split_dir = ROOT / "data" / "processed" / "stage7d"
        stage7_report = read_json(ROOT / "reports" / "stage7d" / "1h" / "stage7d_weather_experiment_report.json")
    else:
        config = read_json(latest(ROOT / "reports" / "stage7a" / "24h", "*/training_dataset_24h_feature_config.json"))
        split_dir = ROOT / "data" / "processed" / "stage7a"
        stage7_report = read_json(latest(ROOT / "reports" / "stage7b" / "24h", "*/catboost_24h_report.json"))
    paths = {part: split_dir / f"training_dataset_{horizon}_{part}.parquet" for part in ("train", "validation", "test")}
    # Stage 10A already captured the frozen CatBoost read-only benchmark fields.
    catboost = read_json(REPORTS / f"xgboost_report_{horizon}.json")["catboost_baseline"]
    # Preserve Stage 7's canonical calibration values in the common comparison schema.
    for part in ("validation", "test"):
        calibration = stage7_report["experimental_metrics"][part]["calibration"] if horizon == "1h" else stage7_report["calibration"][part]
        catboost["metrics"][part].update({"brier_score": calibration["brier_score"], "expected_calibration_error_10_bins": calibration["expected_calibration_error_10_bins"]})
    return config, paths, catboost


def encode_train_only(splits: dict[str, pd.DataFrame], categorical: list[str]) -> dict[str, pd.DataFrame]:
    """Numeric representation for LightGBM; no features are added, dropped, or reordered."""
    result = {name: frame.copy() for name, frame in splits.items()}
    for col in categorical:
        observed_train = result["train"][col].astype("string").fillna("__MISSING__")
        codebook = {value: code for code, value in enumerate(pd.unique(observed_train))}
        for frame in result.values():
            # -1 is a LightGBM missing categorical value, including unseen future categories.
            frame[col] = frame[col].astype("string").fillna("__MISSING__").map(codebook).fillna(-1).astype("int32")
    return result


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    score, y = score[order], y[order]
    tp = np.cumsum(y)
    predicted = np.arange(1, len(y) + 1)
    fn = int(y.sum()) - tp
    f1 = np.divide(2 * tp, 2 * tp + (predicted - tp) + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + (predicted - tp) + fn) != 0)
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]
    return float(score[ends[np.argmax(f1[ends])]])


def ece(y: np.ndarray, score: np.ndarray) -> tuple[float, list[dict[str, object]]]:
    bin_ids = np.clip(np.digitize(score, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    weighted_error, rows = 0.0, []
    for bin_id in range(10):
        selected = bin_ids == bin_id
        if selected.any():
            mean_prediction, observed_rate = float(score[selected].mean()), float(y[selected].mean())
            weighted_error += abs(mean_prediction - observed_rate) * int(selected.sum())
            rows.append({"bin": bin_id, "rows": int(selected.sum()), "mean_prediction": mean_prediction, "observed_rate": observed_rate})
    return float(weighted_error / len(y)), rows


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = (score >= threshold).astype(np.int8)
    top = np.argsort(-score, kind="stable")[:max(1, int(np.ceil(0.1 * len(y))))]
    ece_value, bins = ece(y, score)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y)), "positive_rows": int(y.sum()), "positive_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, predicted, zero_division=0)), "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)), "lift_at_top_10pct": float(y[top].mean() / y.mean()),
        "brier_score": float(brier_score_loss(y, score)), "expected_calibration_error_10_bins": ece_value,
        "calibration_bins": bins, "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def cat_value(catboost: dict[str, object], split: str, metric: str) -> float:
    return float(catboost["metrics"][split][metric])


def train_one(horizon: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    config, paths, catboost = load_source(horizon)
    features = list(config["numerical_features"]) + list(config["categorical_features"])
    categorical = list(config["categorical_features"])
    target = str(config["target_column"])
    required = 97 if horizon == "1h" else 77
    if len(features) != required or len(features) != len(set(features)):
        raise ValueError(f"{horizon} feature contract failed: expected {required} unique features")
    raw = {part: pd.read_parquet(path) for part, path in paths.items()}
    for part, frame in raw.items():
        missing = set(features + [target]) - set(frame.columns)
        if missing:
            raise ValueError(f"{horizon}/{part} missing: {sorted(missing)}")
    encoded = encode_train_only(raw, categorical)
    x = {part: frame.loc[:, features] for part, frame in encoded.items()}
    y = {part: frame[target].to_numpy(np.int8) for part, frame in encoded.items()}
    params = {
        "objective": "binary", "metric": "average_precision", "n_estimators": 1500,
        "learning_rate": 0.05, "num_leaves": 63, "max_depth": 7,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "min_child_samples": 50, "lambda_l2": 5.0, "random_state": SEED,
        "n_jobs": -1, "verbosity": -1,
    }
    model = LGBMClassifier(**params)
    started = time.perf_counter()
    model.fit(x["train"], y["train"], categorical_feature=categorical, eval_set=[(x["validation"], y["validation"])], callbacks=[lgb.early_stopping(100, verbose=False)])
    training_seconds = time.perf_counter() - started
    validation_scores = model.predict_proba(x["validation"], num_iteration=model.best_iteration_)[:, 1]
    threshold = best_f1_threshold(y["validation"], validation_scores)
    scores, prediction_seconds = {}, {}
    for part in ("validation", "test"):
        started = time.perf_counter()
        scores[part] = model.predict_proba(x[part], num_iteration=model.best_iteration_)[:, 1]
        prediction_seconds[part] = time.perf_counter() - started
    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / f"lightgbm_{horizon}.txt"
    model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
    metrics = {part: evaluate(y[part], scores[part], threshold) for part in ("validation", "test")}
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    importance = [{"feature": f, "importance_gain": float(g), "importance_split": int(s)} for f, g, s in sorted(zip(features, gain, split), key=lambda item: item[1], reverse=True)]
    validation_test_gap = metrics["validation"]["pr_auc"] - metrics["test"]["pr_auc"]
    cat_gap = cat_value(catboost, "validation", "pr_auc") - cat_value(catboost, "test", "pr_auc")
    ranking_ok = all(metrics["test"][m] >= cat_value(catboost, "test", m) - 1e-12 for m in ("pr_auc", "lift_at_top_10pct"))
    calibration_ok = all(metrics["test"][m] <= cat_value(catboost, "test", m) + 1e-12 for m in ("brier_score", "expected_calibration_error_10_bins"))
    generalizes = validation_test_gap <= max(0.0, cat_gap) + 1e-12
    winner = "LightGBM" if ranking_ok and calibration_ok and generalizes else "CatBoost"
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(), "stage": "10B", "horizon": horizon,
        "isolation": "Independent experiment. Existing pipeline, Stage 7 assets, CatBoost, XGBoost Stage 10A, SHAP, recommendations, inference, API, and final models were not modified.",
        "dataset": {"split_source": "Stage 7D weather experiment" if horizon == "1h" else "Stage 7B final dataset", "paths": {k: str(v.resolve()) for k, v in paths.items()}},
        "feature_contract": {"feature_count": len(features), "feature_order": features, "categorical_encoding": "Train-only ordinal mapping; unseen validation/test categories map to -1. No feature is added, removed, or reordered."},
        "parameters": params, "early_stopping": {"split": "validation", "rounds": 100, "best_iteration": int(model.best_iteration_)},
        "threshold": {"rule": "maximum validation F1", "value": threshold}, "metrics": metrics,
        "timing_seconds": {"training": training_seconds, "prediction": prediction_seconds}, "model": {"path": str(model_path.resolve()), "size_bytes": model_path.stat().st_size},
        "feature_importance": importance, "catboost_baseline": catboost,
        "decision": {"winner": winner, "test_pr_auc_and_lift_not_worse": ranking_ok, "test_calibration_not_worse": calibration_ok, "no_overfitting": generalizes, "validation_test_pr_auc_gap": validation_test_gap, "catboost_validation_test_pr_auc_gap": cat_gap,
            "reason_ru": "LightGBM удовлетворяет всем обязательным тестовым критериям." if winner == "LightGBM" else "LightGBM отклонён: не все обязательные test-критерии победы выполнены."},
    }
    rows = []
    for part in ("validation", "test"):
        for metric in METRICS:
            rows.append({"horizon": horizon, "split": part, "metric": metric, "catboost": cat_value(catboost, part, metric), "lightgbm": metrics[part][metric]})
    resources = catboost.get("resources", {})
    for metric, cb, lgb_value in (("training_time_seconds", resources.get("training_time_seconds"), training_seconds), ("prediction_time_validation_seconds", resources.get("prediction_validation_seconds"), prediction_seconds["validation"]), ("prediction_time_test_seconds", resources.get("prediction_test_seconds"), prediction_seconds["test"]), ("model_size_bytes", resources.get("model_size_bytes"), model_path.stat().st_size)):
        rows.append({"horizon": horizon, "split": "resource", "metric": metric, "catboost": cb, "lightgbm": lgb_value})
    return report, rows


def final_comparison(lightgbm_reports: dict[str, dict[str, object]]) -> None:
    rows = []
    for horizon in ("1h", "24h"):
        xgb = read_json(REPORTS / f"xgboost_report_{horizon}.json")
        lgb_report = lightgbm_reports[horizon]
        cat = lgb_report["catboost_baseline"]
        for split in ("validation", "test"):
            for metric in METRICS:
                rows.append({"horizon": horizon, "split": split, "metric": metric, "catboost": cat_value(cat, split, metric), "xgboost": xgb["metrics"][split][metric], "lightgbm": lgb_report["metrics"][split][metric]})
        for metric, cat_value_resource, xgb_value, lgb_value in (
            ("training_time_seconds", cat.get("resources", {}).get("training_time_seconds"), xgb["timing_seconds"]["training"], lgb_report["timing_seconds"]["training"]),
            ("prediction_time_validation_seconds", cat.get("resources", {}).get("prediction_validation_seconds"), xgb["timing_seconds"]["prediction"]["validation"], lgb_report["timing_seconds"]["prediction"]["validation"]),
            ("prediction_time_test_seconds", cat.get("resources", {}).get("prediction_test_seconds"), xgb["timing_seconds"]["prediction"]["test"], lgb_report["timing_seconds"]["prediction"]["test"]),
            ("model_size_bytes", cat.get("resources", {}).get("model_size_bytes"), xgb["model"]["size_bytes"], lgb_report["model"]["size_bytes"]),
        ):
            rows.append({"horizon": horizon, "split": "resource", "metric": metric, "catboost": cat_value_resource, "xgboost": xgb_value, "lightgbm": lgb_value})
    winners = {h: lightgbm_reports[h]["decision"]["winner"] for h in lightgbm_reports}
    conclusion = (
        "После объективного сравнения CatBoost, XGBoost и LightGBM на одинаковых данных, одинаковых признаках и одинаковом temporal split: "
        "для 1h финальной production-моделью остаётся CatBoost; для 24h LightGBM является победителем изолированного эксперимента, "
        "так как проходит все заданные test-критерии. Production-модели этим экспериментом не изменяются."
        if winners == {"1h": "CatBoost", "24h": "LightGBM"}
        else "После объективного сравнения CatBoost, XGBoost и LightGBM на одинаковых данных, одинаковых признаках и одинаковом temporal split финальной production-моделью остаётся CatBoost."
    )
    result = {"generated_at_utc": datetime.now(UTC).isoformat(), "stage": "10", "comparison_rows": rows, "winner_by_horizon": winners, "conclusion_ru": conclusion, "catboost_training_time_note": "Historic CatBoost training duration was not recorded; retraining frozen final models is intentionally prohibited."}
    (REPORTS / "final_model_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(REPORTS / "final_model_comparison.csv", index=False)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    reports, all_rows = {}, []
    for horizon in ("1h", "24h"):
        report, rows = train_one(horizon)
        reports[horizon] = report
        all_rows.extend(rows)
        (REPORTS / f"lightgbm_report_{horizon}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = {"generated_at_utc": datetime.now(UTC).isoformat(), "stage": "10B", "comparison_rows": all_rows, "winner_by_horizon": {h: reports[h]["decision"] for h in reports}}
    (REPORTS / "comparison_catboost_vs_lightgbm.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(REPORTS / "comparison_catboost_vs_lightgbm.csv", index=False)
    final_comparison(reports)


if __name__ == "__main__":
    main()
