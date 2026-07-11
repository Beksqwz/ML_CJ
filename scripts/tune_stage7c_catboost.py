"""Limited, reproducible Stage 7C CatBoost tuning without test-set selection."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError

from train_stage7b_catboost import (
    best_f1_threshold,
    calibration_summary,
    metric_summary,
    prepare_features,
    pr_auc,
    roc_auc,
    save_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = PROJECT_ROOT / "data" / "processed" / "stage7a"
STAGE7B_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "stage7b" / "predictions"
MODELS_ROOT = PROJECT_ROOT / "models" / "stage7c"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "stage7c"
SEED = 20260711
SEEDS = (20260711, 20260712, 20260713)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run limited Stage 7C CatBoost tuning for one horizon."
    )
    parser.add_argument("--label", required=True, choices=("1h", "24h"))
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def latest_feature_config(label: str) -> tuple[dict[str, object], Path]:
    paths = sorted(
        (PROJECT_ROOT / "reports" / "stage7a" / label).glob(
            "*/training_dataset_*_feature_config.json"
        )
    )
    if not paths:
        raise FileNotFoundError(f"Stage 7A config not found for {label}")
    path = paths[-1]
    return json.loads(path.read_text(encoding="utf-8")), path


def trial_grid() -> list[dict[str, object]]:
    """Twelve intentional configurations, including one explicit class-weight trial."""
    return [
        {
            "name": "d5_fast",
            "depth": 5,
            "learning_rate": 0.07,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 128,
        },
        {
            "name": "d6_balanced",
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 128,
        },
        {
            "name": "d7_baseline_like",
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 128,
        },
        {
            "name": "d8_slow_regularized",
            "depth": 8,
            "learning_rate": 0.03,
            "l2_leaf_reg": 10.0,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 128,
        },
        {
            "name": "d6_high_border",
            "depth": 6,
            "learning_rate": 0.07,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
            "border_count": 254,
        },
        {
            "name": "d7_low_lr",
            "depth": 7,
            "learning_rate": 0.03,
            "l2_leaf_reg": 12.0,
            "random_strength": 0.25,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 254,
        },
        {
            "name": "d5_strong_reg",
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 15.0,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.0,
            "border_count": 128,
        },
        {
            "name": "d8_low_lr",
            "depth": 8,
            "learning_rate": 0.02,
            "l2_leaf_reg": 20.0,
            "random_strength": 0.75,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.5,
            "border_count": 254,
        },
        {
            "name": "d6_small_border",
            "depth": 6,
            "learning_rate": 0.04,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.25,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 64,
        },
        {
            "name": "d7_noisy",
            "depth": 7,
            "learning_rate": 0.06,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 254,
        },
        {
            "name": "d6_bernoulli",
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 8.0,
            "random_strength": 0.5,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "border_count": 128,
        },
        {
            "name": "d7_class_weight_variant",
            "depth": 7,
            "learning_rate": 0.04,
            "l2_leaf_reg": 8.0,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 128,
            "scale_pos_weight": 2.0,
        },
    ]


def base_params(
    config: dict[str, object], seed: int, iterations: int
) -> dict[str, object]:
    return {
        "iterations": iterations,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": seed,
        "allow_writing_files": False,
        "task_type": "GPU",
        "devices": "0",
        **{key: value for key, value in config.items() if key != "name"},
    }


def fit(
    params: dict[str, object],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    valid_y: np.ndarray,
    cat_indexes: list[int],
) -> CatBoostClassifier:
    model = CatBoostClassifier(**params)
    model.fit(
        train_x,
        train_y,
        cat_features=cat_indexes,
        eval_set=(valid_x, valid_y),
        early_stopping_rounds=100,
        verbose=False,
    )
    return model


def validation_metrics(
    model: CatBoostClassifier,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    valid_y: np.ndarray,
) -> dict[str, float | int | None]:
    train_score = model.predict_proba(train_x)[:, 1]
    valid_score = model.predict_proba(valid_x)[:, 1]
    top_n = max(1, int(np.ceil(len(valid_y) * 0.10)))
    top = np.argsort(-valid_score, kind="stable")[:top_n]
    prevalence = float(valid_y.mean())
    return {
        "train_pr_auc": pr_auc(train_y, train_score),
        "validation_pr_auc": pr_auc(valid_y, valid_score),
        "validation_roc_auc": roc_auc(valid_y, valid_score),
        "validation_recall_at_top_10pct": float(valid_y[top].sum() / valid_y.sum()),
        "validation_lift_at_top_10pct": float(valid_y[top].mean() / prevalence),
        "train_validation_pr_auc_gap": float(
            pr_auc(train_y, train_score) - pr_auc(valid_y, valid_score)
        ),
    }


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    return (
        float(row["validation_pr_auc"]),
        float(row["validation_recall_at_top_10pct"]),
        float(row["validation_lift_at_top_10pct"]),
        float(row["validation_roc_auc"]),
        -float(row["train_validation_pr_auc_gap"]),
    )


def stage7b_validation_pr_auc(label: str) -> float:
    target = f"target_{label}"
    path = (
        STAGE7B_PREDICTIONS / f"training_dataset_{label}_validation_predictions.parquet"
    )
    data = pd.read_parquet(path)
    return float(
        pr_auc(
            data[target].to_numpy(dtype=np.int8),
            data["prediction_probability"].to_numpy(dtype=float),
        )
    )


def stage7b_test_metrics(label: str, threshold: float) -> dict[str, object]:
    target = f"target_{label}"
    path = STAGE7B_PREDICTIONS / f"training_dataset_{label}_test_predictions.parquet"
    data = pd.read_parquet(path)
    return metric_summary(
        data[target].to_numpy(dtype=np.int8),
        data["prediction_probability"].to_numpy(dtype=float),
        threshold,
    )


def differences(
    candidate: dict[str, object], baseline: dict[str, object]
) -> dict[str, object]:
    keys = (
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "recall_at_top_10pct",
        "lift_at_top_10pct",
    )
    output: dict[str, object] = {}
    for key in keys:
        absolute = float(candidate[key]) - float(baseline[key])
        output[key] = {
            "absolute": absolute,
            "relative_percent": absolute / float(baseline[key]) * 100
            if baseline[key]
            else None,
        }
    return output


def main() -> int:
    args = parse_args()
    label = args.label
    target = f"target_{label}"
    config, config_path = latest_feature_config(label)
    features = [*config["numerical_features"], *config["categorical_features"]]
    categorical = list(config["categorical_features"])
    cat_indexes = [features.index(column) for column in categorical]
    # Test is intentionally not loaded before this point in the program; tuning uses train/validation only.
    train = pd.read_parquet(SPLIT_ROOT / f"training_dataset_{label}_train.parquet")
    validation = pd.read_parquet(
        SPLIT_ROOT / f"training_dataset_{label}_validation.parquet"
    )
    train_x = prepare_features(train, features, categorical)
    valid_x = prepare_features(validation, features, categorical)
    train_y = train[target].to_numpy(dtype=np.int8)
    valid_y = validation[target].to_numpy(dtype=np.int8)
    trials: list[dict[str, object]] = []
    models: dict[str, CatBoostClassifier] = {}
    for number, trial in enumerate(trial_grid(), start=1):
        started = time.perf_counter()
        params = base_params(trial, SEED, args.iterations)
        try:
            model = fit(params, train_x, train_y, valid_x, valid_y, cat_indexes)
            metrics = validation_metrics(model, train_x, train_y, valid_x, valid_y)
            result = {
                "trial": number,
                "name": trial["name"],
                "status": "completed",
                "params": params,
                "best_iteration": int(model.get_best_iteration()),
                "training_seconds": time.perf_counter() - started,
                **metrics,
            }
            models[str(trial["name"])] = model
        except (CatBoostError, RuntimeError) as exc:
            result = {
                "trial": number,
                "name": trial["name"],
                "status": "failed",
                "params": params,
                "best_iteration": None,
                "training_seconds": time.perf_counter() - started,
                "error": str(exc),
            }
        trials.append(result)
        print(
            f"{label} trial {number}/{len(trial_grid())}: {trial['name']} — {result['status']}"
        )
    completed = [trial for trial in trials if trial["status"] == "completed"]
    if not completed:
        raise RuntimeError("All tuning trials failed.")
    winner = max(completed, key=rank_key)
    winner_model = models[str(winner["name"])]
    # Stability is validation-only and includes the selected seed plus two independent reruns.
    stability: list[dict[str, object]] = []
    for seed in SEEDS:
        if seed == SEED:
            model = winner_model
        else:
            params = dict(winner["params"]) | {"random_seed": seed}
            model = fit(params, train_x, train_y, valid_x, valid_y, cat_indexes)
        result = validation_metrics(model, train_x, train_y, valid_x, valid_y)
        stability.append(
            {"seed": seed, "best_iteration": int(model.get_best_iteration()), **result}
        )
    stability_pr = np.array([float(item["validation_pr_auc"]) for item in stability])
    stability_std = float(stability_pr.std(ddof=0))
    threshold, validation_f1 = best_f1_threshold(
        valid_y, winner_model.predict_proba(valid_x)[:, 1]
    )
    stage7b_validation = stage7b_validation_pr_auc(label)
    validation_improvement = float(winner["validation_pr_auc"]) / stage7b_validation - 1
    # The test split is read for the first time only after winner, stability and threshold are fixed.
    test = pd.read_parquet(SPLIT_ROOT / f"training_dataset_{label}_test.parquet")
    test_x = prepare_features(test, features, categorical)
    test_y = test[target].to_numpy(dtype=np.int8)
    candidate_validation_score = winner_model.predict_proba(valid_x)[:, 1]
    candidate_test_score = winner_model.predict_proba(test_x)[:, 1]
    candidate_validation = metric_summary(
        valid_y, candidate_validation_score, threshold
    )
    candidate_test = metric_summary(test_y, candidate_test_score, threshold)
    baseline_test = stage7b_test_metrics(label, threshold)
    stable = stability_std <= 0.01
    improved_enough = validation_improvement >= 0.01
    test_not_worse = float(candidate_test["pr_auc"]) >= float(baseline_test["pr_auc"])
    accepted = bool(improved_enough and stable and test_not_worse)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_model_path = MODELS_ROOT / f"catboost_{label}_tuned_candidate.cbm"
    winner_model.save_model(candidate_model_path)
    selected_path: str | None = None
    if accepted:
        path = MODELS_ROOT / f"catboost_{label}_selected_stage7c.cbm"
        shutil.copy2(candidate_model_path, path)
        selected_path = str(path.resolve())
    report_dir = REPORTS_ROOT / label / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    csv_path = report_dir / f"catboost_{label}_tuning_experiments.csv"
    json_path = report_dir / f"catboost_{label}_tuning_experiments.json"
    report_path = report_dir / f"catboost_{label}_tuning_report.json"
    pd.DataFrame(
        [
            {
                **{key: value for key, value in trial.items() if key != "params"},
                "params_json": json.dumps(trial["params"], ensure_ascii=False),
            }
            for trial in trials
        ]
    ).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(trials, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    validation_prediction_path = save_predictions(
        label, "stage7c_validation", validation, target, candidate_validation_score
    )
    test_prediction_path = save_predictions(
        label, "stage7c_test", test, target, candidate_test_score
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "horizon": label,
        "stage7a_feature_config": str(config_path.resolve()),
        "test_access_policy": "Test split was loaded only after tuning winner, stability and validation threshold were fixed.",
        "search": {
            "trial_count": len(trials),
            "seed": SEED,
            "selection_rule": "max validation PR-AUC; then Recall@Top-10%, Lift@Top-10%, ROC-AUC, and smaller train-validation PR-AUC gap.",
        },
        "winner": winner,
        "stability_three_seeds": {
            "runs": stability,
            "pr_auc_mean": float(stability_pr.mean()),
            "pr_auc_std": stability_std,
            "stable_threshold_std_max": 0.01,
        },
        "threshold": {
            "rule": "maximum F1 on validation only",
            "value": threshold,
            "validation_f1": validation_f1,
        },
        "candidate_metrics": {
            "validation": candidate_validation,
            "test": candidate_test,
            "calibration": {
                "validation": calibration_summary(valid_y, candidate_validation_score),
                "test": calibration_summary(test_y, candidate_test_score),
            },
        },
        "stage7b": {
            "validation_pr_auc": stage7b_validation,
            "test_metrics_recomputed_with_candidate_threshold": baseline_test,
        },
        "comparison_with_stage7b": {
            "validation_pr_auc_absolute": float(winner["validation_pr_auc"])
            - stage7b_validation,
            "validation_pr_auc_relative_percent": validation_improvement * 100,
            "test": differences(candidate_test, baseline_test),
        },
        "selection": {
            "stage7c_accepted": accepted,
            "reason": "Stage 7C accepted only when validation PR-AUC improves by at least 1%, 3-seed std <= 0.01, and candidate test PR-AUC is not lower than Stage 7B.",
            "improved_enough": improved_enough,
            "stable": stable,
            "test_not_worse": test_not_worse,
            "final_model": selected_path
            if accepted
            else "Stage 7B remains final; tuned candidate is retained for audit only.",
        },
        "candidate_model_path": str(candidate_model_path.resolve()),
        "prediction_files": {
            "validation": validation_prediction_path,
            "test": test_prediction_path,
        },
        "not_run": ["SHAP", "recommendation engine", "git commit"],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Horizon: {label}")
    print(f"Winner: {winner['name']}")
    print(f"Validation PR-AUC: {winner['validation_pr_auc']}")
    print(f"Test PR-AUC: {candidate_test['pr_auc']}")
    print(f"Stage 7C accepted: {accepted}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
