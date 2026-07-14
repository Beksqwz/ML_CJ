"""Stage 19G seasonal robustness audit for frozen Stage 7B versus hotspot history."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_24h_natural_prevalence import (  # noqa: E402
    EXPECTED_HASH,
    CONFIG_PATH,
    PROCESSED,
    build_feature_context,
    build_features_for_chunk,
    feature_hash,
    target_fields,
)
from run_stage19f_hotspot_baseline import (  # noqa: E402
    accident_indexes,
    historical_scores,
    metric_bundle,
    ranking_metrics,
)


REPORTS = ROOT / "reports" / "stage19g"
MODEL_PATH = ROOT / "models" / "production" / "catboost_24h.cbm"
AUTUMN_PARTS = (
    ROOT
    / "data"
    / "audit"
    / "stage19e"
    / "minimal_full_run_fixed"
    / "predictions_parts"
)
STAGE19F = ROOT / "reports" / "stage19f" / "stage19f_summary.json"
TEST_START = pd.Timestamp("2024-09-14 12:00:00")
HOURS = 168
CHUNK_HOURS = 6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_candidates(
    accidents: pd.DataFrame, start_after: pd.Timestamp, months: set[int]
) -> pd.Timestamp:
    latest_start = accidents.event_hour.max() - pd.Timedelta(hours=24 + HOURS - 1)
    candidates = pd.date_range(start_after.ceil("D"), latest_start.floor("D"), freq="D")
    for candidate in candidates:
        if candidate.month not in months:
            continue
        event_count = accidents.loc[
            accidents.event_hour.gt(candidate)
            & accidents.event_hour.le(candidate + pd.Timedelta(hours=HOURS + 23)),
            "event_count",
        ].sum()
        if event_count:
            return candidate
    raise ValueError(f"no_positive_window_for_months_{sorted(months)}")


def record_metrics(records: list[dict[str, object]]) -> dict[str, float]:
    table = pd.DataFrame(records)
    mean = table.mean(numeric_only=True)
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


def window_ranking(frame: pd.DataFrame, score_column: str) -> list[dict[str, object]]:
    """Per-hour metrics with explicit NaN for hours having no positives."""
    records = []
    for timestamp, hour in frame.groupby("prediction_datetime", sort=True):
        ranked = hour.sort_values(
            [score_column, "road_segment_id"], ascending=[False, True], kind="stable"
        )
        positives = int(ranked.target_24h.sum())
        row: dict[str, object] = {
            "prediction_datetime": str(timestamp),
            "positives": positives,
        }
        for size in (10, 20, 50, 40, 199, 397):
            selected = ranked.head(size).target_24h.to_numpy(dtype=np.int8)
            precision = float(selected.mean())
            row[f"precision_{size}"] = precision
            row[f"recall_{size}"] = (
                float(selected.sum() / positives) if positives else np.nan
            )
            row[f"lift_{size}"] = (
                precision / (positives / len(ranked)) if positives else np.nan
            )
        records.append(row)
    return records


def top10_sets(frame: pd.DataFrame) -> tuple[set[str], dict[str, int]]:
    observed: set[str] = set()
    frequency: dict[str, int] = {}
    for _, hour in frame.groupby("prediction_datetime", sort=False):
        ranked = hour.sort_values(
            ["raw_model_score", "road_segment_id"],
            ascending=[False, True],
            kind="stable",
        )
        for segment in ranked.head(10).road_segment_id.astype(str):
            observed.add(segment)
            frequency[segment] = frequency.get(segment, 0) + 1
    return observed, frequency


def evaluate_existing_autumn(
    indexes: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, set[str], dict[str, int]]:
    parts = sorted(AUTUMN_PARTS.glob("part_*.parquet"))
    if len(parts) != 28:
        raise ValueError("missing_completed_autumn_stage19e_parts")
    frames = []
    for part in parts:
        frame = pd.read_parquet(
            part,
            columns=[
                "road_segment_id",
                "prediction_datetime",
                "target_24h",
                "raw_model_score",
            ],
        )
        frame["baseline_score"] = historical_scores(frame, indexes)
        frames.append(frame)
    grid = pd.concat(frames, ignore_index=True)
    cat_rank, cat_records = ranking_metrics(grid, "raw_model_score")
    baseline_rank, _ = ranking_metrics(grid, "baseline_score")
    y = grid.target_24h.to_numpy(dtype=np.uint8)
    return (
        {
            "catboost": metric_bundle(y, grid.raw_model_score.to_numpy(), cat_rank),
            "baseline": metric_bundle(y, grid.baseline_score.to_numpy(), baseline_rank),
            "rows": len(grid),
            "timestamps": int(grid.prediction_datetime.nunique()),
        },
        *top10_sets(grid),
    )


def evaluate_window(
    start: pd.Timestamp,
    config: dict,
    ready: pd.DataFrame,
    indexes: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, set[str], dict[str, int]]:
    hours = pd.date_range(start, periods=HOURS, freq="h")
    context = build_feature_context(ready, config, hours)
    labels: list[np.ndarray] = []
    cat_scores: list[np.ndarray] = []
    baseline_scores: list[np.ndarray] = []
    cat_records: list[dict[str, object]] = []
    baseline_records: list[dict[str, object]] = []
    observed: set[str] = set()
    frequency: dict[str, int] = {}
    for begin in range(0, HOURS, CHUNK_HOURS):
        chunk_times = hours[begin : begin + CHUNK_HOURS]
        grid = pd.MultiIndex.from_product(
            [chunk_times, context.segment_order],
            names=["prediction_datetime", "road_segment_id"],
        ).to_frame(index=False)
        features = build_features_for_chunk(grid, context, config)
        targets = target_fields(
            features[["road_segment_id", "prediction_datetime"]], context.counts
        )
        frame = features[["road_segment_id", "prediction_datetime"]].merge(
            targets,
            on=["road_segment_id", "prediction_datetime"],
            validate="one_to_one",
        )
        scores = np.empty(len(features), dtype=np.float64)
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        for _, positions in features.groupby(
            "prediction_datetime", sort=False
        ).indices.items():
            matrix = features.iloc[positions][list(context.feature_order)].copy()
            for column in context.categorical_features:
                matrix[column] = (
                    matrix[column].astype("string").fillna("__MISSING__").astype(str)
                )
            scores[positions] = model.predict_proba(matrix, thread_count=1)[:, 1]
        frame["raw_model_score"] = scores
        frame["baseline_score"] = historical_scores(frame, indexes)
        cat_part_records = window_ranking(frame, "raw_model_score")
        baseline_part_records = window_ranking(frame, "baseline_score")
        cat_records.extend(cat_part_records)
        baseline_records.extend(baseline_part_records)
        top_set, top_frequency = top10_sets(frame)
        observed.update(top_set)
        for segment, count in top_frequency.items():
            frequency[segment] = frequency.get(segment, 0) + count
        labels.append(frame.target_24h.to_numpy(dtype=np.uint8))
        cat_scores.append(frame.raw_model_score.to_numpy(dtype=np.float32))
        baseline_scores.append(frame.baseline_score.to_numpy(dtype=np.int64))
    y = np.concatenate(labels)
    return (
        {
            "catboost": metric_bundle(
                y, np.concatenate(cat_scores), record_metrics(cat_records)
            ),
            "baseline": metric_bundle(
                y, np.concatenate(baseline_scores), record_metrics(baseline_records)
            ),
            "rows": len(y),
            "timestamps": HOURS,
        },
        observed,
        frequency,
    )


def comparison(
    catboost: dict[str, float], baseline: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    rows = {}
    for name, value in catboost.items():
        base = baseline[name]
        diff = value - base
        rows[name] = {
            "catboost": value,
            "baseline": base,
            "absolute_difference": diff,
            "relative_improvement_pct": diff / base * 100 if base else None,
            "winner": "catboost" if diff > 0 else "baseline" if diff < 0 else "tie",
        }
    return rows


def robustness(
    window_metrics: dict[str, dict],
) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for model_name in ("catboost", "baseline"):
        metric_names = window_metrics[next(iter(window_metrics))][model_name].keys()
        result[model_name] = {}
        for metric in metric_names:
            values = np.array(
                [window_metrics[name][model_name][metric] for name in window_metrics],
                dtype=float,
            )
            result[model_name][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "coefficient_of_variation": float(
                    values.std(ddof=0) / abs(values.mean())
                )
                if values.mean()
                else None,
            }
    return result


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        feature_hash([*config["numerical_features"], *config["categorical_features"]])
        != EXPECTED_HASH
    ):
        raise ValueError("frozen_feature_hash_changed")
    if (
        sha256(MODEL_PATH)
        != "0c8e1b88b1cfaf95fb39e395e2fdc54f1b7abda22d8ac00e1d6f561ab9110a0c"
    ):
        raise ValueError("frozen_model_hash_changed")
    ready = pd.read_parquet(PROCESSED / "accidents_with_roads_ml_ready.parquet")
    counts = pd.DataFrame()
    counts["event_hour"] = pd.to_datetime(ready.accident_datetime).dt.floor("h")
    counts["event_count"] = 1
    indexes = accident_indexes()
    autumn_start = pd.Timestamp("2024-09-30 19:00:00")
    winter_start = selection_candidates(
        counts, autumn_start + pd.Timedelta(hours=HOURS), {1, 2}
    )
    spring_start = selection_candidates(
        counts, winter_start + pd.Timedelta(hours=HOURS), {4, 5, 6}
    )
    windows = {"autumn": autumn_start, "winter": winter_start, "spring": spring_start}
    autumn, previous_set, previous_frequency = evaluate_existing_autumn(indexes)
    evaluations = {"autumn": autumn}
    stability = {
        "autumn": {
            "unique_top10_segments": len(previous_set),
            "overlap_with_previous": None,
            "jaccard_with_previous": None,
            "permanently_dominant_segments": sum(
                value == HOURS for value in previous_frequency.values()
            ),
            "new_top10_segments": len(previous_set),
        }
    }
    for name in ("winter", "spring"):
        evaluation, current_set, current_frequency = evaluate_window(
            windows[name], config, ready, indexes
        )
        evaluations[name] = evaluation
        overlap = previous_set & current_set
        stability[name] = {
            "unique_top10_segments": len(current_set),
            "overlap_with_previous": len(overlap),
            "jaccard_with_previous": len(overlap) / len(previous_set | current_set),
            "permanently_dominant_segments": sum(
                value == HOURS for value in current_frequency.values()
            ),
            "new_top10_segments": len(current_set - previous_set),
        }
        previous_set, previous_frequency = current_set, current_frequency
    comparisons = {
        name: comparison(item["catboost"], item["baseline"])
        for name, item in evaluations.items()
    }
    wins = {"catboost": 0, "baseline": 0, "tie": 0}
    for item in comparisons.values():
        for metric in item.values():
            wins[metric["winner"]] += 1
    REPORTS.mkdir(parents=True, exist_ok=True)
    selection = {
        name: {
            "start": str(start),
            "end": str(start + pd.Timedelta(hours=HOURS - 1)),
            "hours": HOURS,
        }
        for name, start in windows.items()
    }
    for filename, payload in (
        ("window_selection.json", selection),
        ("window_metrics.json", evaluations),
        (
            "baseline_comparison.json",
            {"per_window": comparisons, "wins_losses_ties": wins},
        ),
        ("robustness_statistics.json", robustness(evaluations)),
        ("top10_stability.json", stability),
    ):
        (REPORTS / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    seasonal = {
        "best_roc_auc": max(
            evaluations, key=lambda name: evaluations[name]["catboost"]["roc_auc"]
        ),
        "worst_roc_auc": min(
            evaluations, key=lambda name: evaluations[name]["catboost"]["roc_auc"]
        ),
        "catboost_consistent_over_baseline": all(
            item["roc_auc"]["winner"] == "catboost" for item in comparisons.values()
        ),
    }
    (REPORTS / "seasonal_analysis.json").write_text(
        json.dumps(seasonal, indent=2), encoding="utf-8"
    )
    summary = {
        "windows": selection,
        "metrics": evaluations,
        "comparison": comparisons,
        "top10_stability": stability,
        "seasonal_analysis": seasonal,
    }
    (REPORTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (REPORTS / "summary.md").write_text(
        "# Stage 19G seasonal robustness\n\n" + json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
