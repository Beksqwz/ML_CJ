"""Prepare leakage-safe chronological Stage 7A splits and historical baseline.

This script never trains a ML model.  It creates temporal train/validation/test
Parquet copies, feature configurations, and baseline metrics derived from train
targets only.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT = PROCESSED / "stage7a"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "stage7a"
DEFAULT_DATASETS = {
    "1h": PROCESSED / "training_dataset_1h.parquet",
    "24h": PROCESSED / "training_dataset_24h.parquet",
}
RARE_CATEGORY_MIN_TRAIN_ROWS = 100
SMOOTHING_ALPHA = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create chronological Stage 7A data splits and baseline reports.")
    parser.add_argument("--dataset", type=Path, help="Input training_dataset Parquet.")
    parser.add_argument("--horizon-hours", type=int, choices=(1, 24), help="Forecast horizon of the input dataset.")
    parser.add_argument("--label", choices=("1h", "24h"), help="Shortcut for a standard Stage 6 dataset.")
    parser.add_argument("--rare-min-train-rows", type=int, default=RARE_CATEGORY_MIN_TRAIN_ROWS)
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> tuple[Path, int, str]:
    if args.label:
        return DEFAULT_DATASETS[args.label], int(args.label.removesuffix("h")), args.label
    if args.dataset is None or args.horizon_hours is None:
        raise ValueError("Pass --label 1h|24h or both --dataset and --horizon-hours.")
    return args.dataset, args.horizon_hours, f"{args.horizon_hours}h"


def target_column(data: pd.DataFrame, horizon: int) -> str:
    target = f"target_{horizon}h"
    if target not in data.columns:
        raise ValueError(f"Dataset has no expected target column: {target}")
    return target


def chronological_splits(data: pd.DataFrame, horizon_hours: int) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    timestamps = pd.Series(pd.to_datetime(data["datetime_hour"], errors="raise").unique()).sort_values().reset_index(drop=True)
    first_boundary = timestamps.iloc[int(len(timestamps) * 0.70)]
    second_boundary = timestamps.iloc[int(len(timestamps) * 0.85)]
    horizon = pd.Timedelta(hours=horizon_hours)
    timestamp = pd.to_datetime(data["datetime_hour"])
    # Purge each partition tail, so its labels never enter the next partition's calendar interval.
    train = data.loc[timestamp < first_boundary - horizon].copy()
    validation = data.loc[(timestamp >= first_boundary) & (timestamp < second_boundary - horizon)].copy()
    test = data.loc[timestamp >= second_boundary].copy()
    dropped = int(len(data) - len(train) - len(validation) - len(test))
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One chronological split is empty.")
    boundaries = {
        "split_time_axis": "datetime_hour interpreted as Asia/Almaty local civil time",
        "initial_train_validation_boundary": str(first_boundary),
        "initial_validation_test_boundary": str(second_boundary),
        "purge_horizon_hours": horizon_hours,
        "train_end_exclusive": str(first_boundary - horizon),
        "validation_start": str(first_boundary),
        "validation_end_exclusive": str(second_boundary - horizon),
        "test_start": str(second_boundary),
        "purged_boundary_rows": dropped,
    }
    return {"train": train, "validation": validation, "test": test}, boundaries


def feature_configuration(data: pd.DataFrame, target: str) -> dict[str, object]:
    excluded = {
        target,
        "target_1h",
        "target_24h",
        "target_datetime_hour",
        "datetime_hour",
        "calendar_date",
        "road_segment_id",
        "objectid",
        "globalid",
        "accident_datetime",
        "event_hour",
        "type_dtp",
        "fd1r17",
        "fd1r17_descrip",
        "distance_to_road_m",
    }
    present_excluded = sorted(excluded.intersection(data.columns))
    candidates = [column for column in data.columns if column not in excluded]
    categorical: list[str] = []
    numerical: list[str] = []
    for column in candidates:
        dtype = data[column].dtype
        if (
            pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or column in {"weather_weather_code", "calendar_year", "calendar_month", "calendar_day", "calendar_hour", "calendar_weekday"}
            or column.startswith("calendar_is_")
            or column.endswith("_missing")
            or column in {"calendar_season", "calendar_holiday_name", "road_highway", "road_oneway", "segment_has_history"}
        ):
            categorical.append(column)
        else:
            numerical.append(column)
    return {
        "target_column": target,
        "excluded_from_model_features": present_excluded,
        "metadata_columns": [column for column in ("datetime_hour", "road_segment_id") if column in data.columns],
        "numerical_features": numerical,
        "categorical_features": categorical,
        "road_missing_value_policy": {
            "road_lanes_num": "Set to NaN when road_lanes_missing=True; retain road_lanes_missing categorical flag.",
            "road_maxspeed_kmh": "Set to NaN when road_maxspeed_missing=True; retain road_maxspeed_missing categorical flag.",
        },
    }


def apply_train_only_transforms(splits: dict[str, pd.DataFrame], config: dict[str, object], rare_min: int) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    train = splits["train"]
    frequency = train["road_highway"].astype("string").fillna("UNKNOWN").value_counts()
    rare = sorted(frequency.loc[frequency < rare_min].index.astype(str).tolist())
    known = set(frequency.index.astype(str)) - set(rare)
    transformed: dict[str, pd.DataFrame] = {}
    for name, frame in splits.items():
        result = frame.copy()
        road_highway = result["road_highway"].astype("string").fillna("UNKNOWN").astype(str)
        result["road_highway"] = road_highway.where(road_highway.isin(known), "OTHER").astype("string")
        result.loc[result["road_lanes_missing"].astype(bool), "road_lanes_num"] = np.nan
        result.loc[result["road_maxspeed_missing"].astype(bool), "road_maxspeed_kmh"] = np.nan
        transformed[name] = result
    transform = {
        "road_highway_grouping": {
            "source": "train only",
            "rare_min_train_rows": rare_min,
            "rare_categories_mapped_to_OTHER": rare,
            "train_category_counts_before_grouping": {str(key): int(value) for key, value in frequency.items()},
        },
        "missing_numeric_values_masked": ["road_lanes_num", "road_maxspeed_kmh"],
    }
    return transformed, transform


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pr_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    grouped = pd.DataFrame({"score": score, "target": y}).groupby("score", sort=False)["target"].agg(["count", "sum"]).sort_index(ascending=False)
    cumulative_positive = grouped["sum"].cumsum()
    precision = cumulative_positive / grouped["count"].cumsum()
    recall = cumulative_positive / positives
    return float((precision * recall.diff().fillna(recall)).sum())


def best_train_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    grouped = pd.DataFrame({"score": score, "target": y}).groupby("score")["target"].agg(["count", "sum"]).sort_index(ascending=False)
    true_positive = grouped["sum"].cumsum().to_numpy(dtype=float)
    false_positive = (grouped["count"].cumsum() - grouped["sum"].cumsum()).to_numpy(dtype=float)
    false_negative = float(y.sum()) - true_positive
    f1 = np.divide(2 * true_positive, 2 * true_positive + false_positive + false_negative, out=np.zeros_like(true_positive), where=(2 * true_positive + false_positive + false_negative) > 0)
    return float(grouped.index.to_numpy(dtype=float)[int(np.argmax(f1))])


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | None]:
    predicted = score >= threshold
    true_positive = int(((predicted) & (y == 1)).sum())
    false_positive = int(((predicted) & (y == 0)).sum())
    false_negative = int(((~predicted) & (y == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    top_n = max(1, int(np.ceil(len(y) * 0.10)))
    top_index = np.argsort(-score, kind="stable")[:top_n]
    top_positive_rate = float(y[top_index].mean())
    prevalence = float(y.mean())
    return {
        "pr_auc": pr_auc(y, score),
        "roc_auc": roc_auc(y, score),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_at_top_10pct": float(y[top_index].sum() / y.sum()) if y.sum() else None,
        "lift_at_top_10pct": top_positive_rate / prevalence if prevalence else None,
        "predicted_positive_rows": int(predicted.sum()),
        "top_10pct_rows": top_n,
    }


def historical_baseline(splits: dict[str, pd.DataFrame], target: str) -> dict[str, object]:
    train = splits["train"]
    global_rate = float(train[target].mean())
    grouped = train.groupby("road_segment_id")[target].agg(["sum", "count"])
    segment_rate = ((grouped["sum"] + SMOOTHING_ALPHA * global_rate) / (grouped["count"] + SMOOTHING_ALPHA)).to_dict()
    train_score = train["road_segment_id"].map(segment_rate).fillna(global_rate).to_numpy(dtype=float)
    threshold = best_train_f1_threshold(train[target].to_numpy(dtype=np.int8), train_score)
    payload: dict[str, object] = {
        "name": "train_only_smoothed_segment_historical_rate",
        "formula": "(train_segment_positive_count + alpha * train_global_positive_rate) / (train_segment_row_count + alpha)",
        "smoothing_alpha": SMOOTHING_ALPHA,
        "train_global_positive_rate": global_rate,
        "train_segments_with_rate": int(len(segment_rate)),
        "classification_threshold_selected_on_train_f1": threshold,
        "metrics": {},
    }
    for name in ("validation", "test"):
        frame = splits[name]
        score = frame["road_segment_id"].map(segment_rate).fillna(global_rate).to_numpy(dtype=float)
        payload["metrics"][name] = metrics(frame[target].to_numpy(dtype=np.int8), score, threshold)
        payload["metrics"][name]["rows"] = int(len(frame))
        payload["metrics"][name]["positive_rows"] = int(frame[target].sum())
        payload["metrics"][name]["unseen_segments_fallback_to_global_rate"] = int((~frame["road_segment_id"].isin(segment_rate)).sum())
    return payload


def save_splits(splits: dict[str, pd.DataFrame], target: str, label: str, config: dict[str, object]) -> dict[str, str]:
    feature_columns = [*config["numerical_features"], *config["categorical_features"]]
    output_paths: dict[str, str] = {}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        # datetime_hour and target remain for temporal audit/supervised training; identifiers are not retained.
        output = frame[["datetime_hour", target, *feature_columns]].copy()
        path = OUTPUT_ROOT / f"training_dataset_{label}_{name}.parquet"
        output.to_parquet(path, index=False, engine="pyarrow")
        output_paths[name] = str(path.resolve())
    return output_paths


def main() -> int:
    args = parse_args()
    dataset_path, horizon, label = resolve_args(args)
    data = pd.read_parquet(dataset_path)
    target = target_column(data, horizon)
    splits, boundaries = chronological_splits(data, horizon)
    config = feature_configuration(data, target)
    transformed, transform = apply_train_only_transforms(splits, config, args.rare_min_train_rows)
    baseline = historical_baseline(transformed, target)
    output_paths = save_splits(transformed, target, label, config)
    report_dir = REPORTS_ROOT / label / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    config_path = report_dir / f"training_dataset_{label}_feature_config.json"
    report_path = report_dir / f"training_dataset_{label}_stage7a_report.json"
    config_payload = config | {"transforms": transform, "output_split_paths": output_paths}
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": str(dataset_path.resolve()),
        "horizon_hours": horizon,
        "timezone_interpretation": "Asia/Almaty",
        "source_rows": int(len(data)),
        "split_boundaries": boundaries,
        "splits": {name: {"rows": int(len(frame)), "positive_rows": int(frame[target].sum()), "negative_rows": int((frame[target] == 0).sum()), "start": str(frame["datetime_hour"].min()), "end": str(frame["datetime_hour"].max())} for name, frame in transformed.items()},
        "feature_config": str(config_path.resolve()),
        "output_split_paths": output_paths,
        "baseline": baseline,
        "model_training": "Not run. CatBoost and SHAP were intentionally not invoked.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Horizon: {label}")
    print(f"Report: {report_path}")
    for name, values in report["splits"].items():
        print(f"{name}: {values['rows']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
