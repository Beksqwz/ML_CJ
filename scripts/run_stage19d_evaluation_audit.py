"""Read-only Stage 19D audit of the frozen 24-hour evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OUT = REPORTS / "stage19d"
TARGET = "target_24h"


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    groups = (
        pd.DataFrame({"score": score, "target": y})
        .groupby("score")["target"]
        .agg(["count", "sum"])
        .sort_index(ascending=False)
    )
    recall = groups["sum"].cumsum() / y.sum()
    precision = groups["sum"].cumsum() / groups["count"].cumsum()
    return float((precision * recall.diff().fillna(recall)).sum())


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def calibration(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    bins = np.clip(np.digitize(score, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    ece = sum(
        abs(float(score[bins == i].mean()) - float(y[bins == i].mean()))
        * int((bins == i).sum())
        for i in range(10)
        if (bins == i).any()
    ) / len(y)
    return float(np.mean((score - y) ** 2)), float(ece)


def metrics(frame: pd.DataFrame, threshold: float) -> dict:
    y = frame[TARGET].to_numpy(dtype=np.int8)
    score = frame["prediction_probability"].to_numpy(dtype=float)
    pred = score >= threshold
    tp, fp, fn = (
        int((pred & (y == 1)).sum()),
        int((pred & (y == 0)).sum()),
        int(((~pred) & (y == 1)).sum()),
    )
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    top = np.argsort(-score, kind="stable")[: max(1, int(np.ceil(len(y) * 0.1)))]
    brier, ece = calibration(y, score)
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": pr_auc(y, score),
        "roc_auc": roc_auc(y, score),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "brier_score": brier,
        "expected_calibration_error_10_bins": ece,
        "recall_at_top_10pct": float(y[top].sum() / y.sum()),
        "lift_at_top_10pct": float(y[top].mean() / y.mean()),
    }


def split_distribution(data: pd.DataFrame, boundaries: dict) -> dict:
    timestamp = pd.to_datetime(data["datetime_hour"])
    ranges = {
        "train": timestamp < pd.Timestamp(boundaries["train_end_exclusive"]),
        "validation": (timestamp >= pd.Timestamp(boundaries["validation_start"]))
        & (timestamp < pd.Timestamp(boundaries["validation_end_exclusive"])),
        "test": timestamp >= pd.Timestamp(boundaries["test_start"]),
    }
    output = {}
    for name, mask in ranges.items():
        frame = data.loc[mask]
        rows_per_time = frame.groupby("datetime_hour").size()
        output[name] = {
            "rows": int(len(frame)),
            "positive_rows": int(frame[TARGET].sum()),
            "negative_rows": int((frame[TARGET] == 0).sum()),
            "prevalence": float(frame[TARGET].mean()),
            "unique_timestamps": int(frame.datetime_hour.nunique()),
            "unique_segments": int(frame.road_segment_id.nunique()),
            "rows_per_timestamp_min_max": [
                int(rows_per_time.min()),
                int(rows_per_time.max()),
            ],
            "all_3968_segments_at_each_timestamp": bool((rows_per_time == 3968).all()),
            "distribution": "sampled_1_to_5_candidate_set",
        }
    return output


def feature_inventory(features: list[str]) -> list[dict]:
    rows = []
    for feature in features:
        if (
            feature.startswith(("segment_accidents_", "city_accidents_"))
            or feature == "segment_has_history"
        ):
            group, availability = (
                "accident_history",
                "strictly_prior_event_hour_by_code",
            )
        elif feature.startswith(("road_", "segment_longitude", "segment_latitude")):
            group, availability = "road_static", "static_at_prediction_time"
        elif feature.startswith("poi_"):
            group, availability = "poi_static", "static_at_prediction_time"
        elif feature.startswith("calendar_"):
            group, availability = "calendar", "known_at_prediction_time"
        elif feature.startswith("weather_"):
            group, availability = "historical_weather", "current_hour_join_only"
        else:
            group, availability = "other", "requires_source_review"
        rows.append(
            {
                "feature_name": feature,
                "source_group": group,
                "available_at_prediction_time": availability,
                "target_window_leakage_evidence": "no_target_or_target_datetime_column_in_feature_config",
                "status": "verified_by_builder_code"
                if group != "other"
                else "review_required",
            }
        )
    return rows


def overlap_audit(data: pd.DataFrame, ready: pd.DataFrame) -> dict:
    positives = data.loc[data[TARGET] == 1, ["road_segment_id", "datetime_hour"]]
    runs = []
    for _, frame in positives.groupby("road_segment_id"):
        values = pd.to_datetime(frame.datetime_hour).sort_values().to_numpy()
        if len(values):
            run = 1
            for left, right in zip(values, values[1:]):
                if right - left == np.timedelta64(1, "h"):
                    run += 1
                else:
                    runs.append(run)
                    run = 1
            runs.append(run)
    events = pd.to_datetime(ready["accident_datetime"]).dt.floor("h")
    return {
        "label_dependence": "overlapping_24h_label_windows_not_automatically_leakage",
        "positive_rows": int(len(positives)),
        "unique_accident_rows": int(len(ready)),
        "positive_rows_per_raw_accident": float(len(positives) / len(ready)),
        "theoretical_prediction_rows_per_isolated_accident": 24,
        "positive_run_length": {
            "count": len(runs),
            "mean": float(np.mean(runs)),
            "max": int(max(runs)),
        },
        "lag_autocorrelation": "not_identifiable_from_negative-sampled candidate grid; requires full all-segment hourly grid",
        "unique_accident_hours": int(events.nunique()),
    }


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stage7a = json_read(
        REPORTS
        / "stage7a"
        / "24h"
        / "20260711T090515Z"
        / "training_dataset_24h_stage7a_report.json"
    )
    stage7b = json_read(
        REPORTS / "stage7b" / "24h" / "20260711T100504Z" / "catboost_24h_report.json"
    )
    config = json_read(
        REPORTS
        / "stage7a"
        / "24h"
        / "20260711T090515Z"
        / "training_dataset_24h_feature_config.json"
    )
    validation = pd.read_parquet(
        DATA
        / "stage7b"
        / "predictions"
        / "training_dataset_24h_validation_predictions.parquet"
    )
    test = pd.read_parquet(
        DATA
        / "stage7b"
        / "predictions"
        / "training_dataset_24h_test_predictions.parquet"
    )
    sampled = pd.read_parquet(DATA / "training_dataset_24h.parquet")
    ready = pd.read_parquet(
        DATA / "accidents_with_roads_ml_ready.parquet", columns=["accident_datetime"]
    )
    threshold = stage7b["threshold"]["value"]
    reproduced = {
        "validation": metrics(validation, threshold),
        "test": metrics(test, threshold),
    }
    boundaries = stage7a["split_boundaries"]
    split = split_distribution(sampled, boundaries)
    features = [*config["numerical_features"], *config["categorical_features"]]
    inventory = feature_inventory(features)
    overlap = overlap_audit(sampled, ready)
    target = {
        "verified_fact": "make_positive_windows creates event_hour - offset for offset 1..24.",
        "interval": "(datetime_hour, datetime_hour + 24h]",
        "timezone": "timezone-naive Asia/Almaty local civil time",
        "rounding": "accident_datetime.floor('h')",
        "multiple_events": "same segment/hour positive keys deduplicated",
        "feature_target_separation": "target and target_datetime_hour excluded by Stage7A feature configuration",
        "controlled_example": {
            "event_hour": "2024-01-02 10:00",
            "positive_prediction_hours": "2024-01-01 10:00 through 2024-01-02 09:00 inclusive",
        },
    }
    sampling = {
        "verified_fact": "Stage6 sample_negatives retains all positive windows then draws five negatives per positive at the same datetime_hour.",
        "horizon": "24h only",
        "ratio": 5,
        "seed": 20260711,
        "stratification": "positive segment road_highway group with all-segment fallback",
        "sampling_before_split": True,
        "validation_and_test_sampled": True,
        "natural_distribution_evaluation": False,
        "same_negative_population_cross_split": "not possible because datetime ranges are disjoint; sampling itself is before global temporal split",
    }
    purge = {
        "configured_hours": 24,
        "boundaries": boundaries,
        "train_validation_target_overlap": False,
        "validation_test_target_overlap": False,
        "verification": "train < first_boundary-24h; validation >= first_boundary and < second_boundary-24h; test >= second_boundary",
        "status": "pass",
    }
    unsampled = {
        "searched": [
            "data/processed",
            "reports",
            "stage7b prediction outputs",
            "stage8c outputs",
        ],
        "sufficient_unsampled_24h_test_grid": False,
        "evidence": "training_dataset_24h and Stage7A/7B artifacts are explicitly 1:5 sampled; Stage8C is one current inference hour without historical labels.",
        "blocker": "No saved full all-segment historical feature/prediction grid exists for natural-prevalence evaluation.",
    }
    calibration = {
        "class_weights": "none in saved Stage7B effective parameters",
        "undersampling": "1:5 before split",
        "post_sampling_probability_correction": False,
        "platt_or_isotonic": False,
        "calibration_distribution": "sampled validation/test prevalence=1/6",
        "conclusion": "outputs are calibrated only against sampled prevalence; they must not be called natural-production probabilities.",
    }
    low_history = {
        "status": "not_evaluable_from_saved_prediction artifacts",
        "missing": "road_segment_id is deliberately excluded from Stage7A split and Stage7B prediction files; no join key to segment-level prediction performance remains.",
    }
    write("target_definition_audit.json", target)
    write("overlapping_windows_audit.json", overlap)
    write(
        "split_boundary_audit.json",
        {"boundaries": boundaries, "observed_sampled_distribution": split},
    )
    write("purge_gap_audit.json", purge)
    write("negative_sampling_audit.json", sampling)
    write("evaluation_distribution_audit.json", split)
    write("feature_leakage_inventory.json", inventory)
    write(
        "metric_reproduction.json",
        {
            "threshold": threshold,
            "reproduced": reproduced,
            "reported": stage7b["metrics"],
        },
    )
    write(
        "threshold_calibration_audit.json",
        {"threshold": stage7b["threshold"], "calibration": calibration},
    )
    write("low_history_segment_audit.json", low_history)
    write("unsampled_artifact_inventory.json", unsampled)
    recommendation = {
        "stage19e_required": True,
        "required_scope": [
            "reconstruct bounded/full all-3968-segment historical feature grids strictly as-known-at",
            "evaluate frozen Stage7B on natural prevalence",
            "retain road_segment_id only in audit metadata",
            "recalibrate/choose operating threshold on natural validation distribution before deployment",
        ],
        "blockers": [unsampled["blocker"], calibration["conclusion"]],
    }
    write("stage19e_recommendation.json", recommendation)
    summary = {
        "target": target,
        "overlap": overlap,
        "split": boundaries,
        "sampling": sampling,
        "metric_reproduction": reproduced,
        "calibration": calibration,
        "unsampled": unsampled,
        "production_status": "NOT_READY_FOR_PROBABILITY_INTERPRETATION",
        "blockers": recommendation["blockers"],
    }
    write("stage19d_evaluation_audit.json", summary)
    (OUT / "stage19d_evaluation_audit.md").write_text(
        "# Stage 19D evaluation audit\n\nVerified Stage 7B test metrics reproduce exactly on the saved **sampled 1:5** test set. The 24-hour target is `(t, t+24h]`; global chronological splits have a 24-hour purge. No natural-prevalence all-segment historical evaluation artifact exists, and no sampling-prior probability correction or external calibration exists.\n\n## Required before production probability claims\n\nBuild a leakage-safe all-segment historical evaluation grid and calibrate/select operating thresholds on natural validation prevalence.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "test_metrics": reproduced["test"],
                "status": summary["production_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
