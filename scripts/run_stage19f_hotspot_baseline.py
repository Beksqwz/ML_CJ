"""Leakage-safe Stage 19F comparison against historical hotspot frequency."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = (
    ROOT
    / "data"
    / "audit"
    / "stage19e"
    / "minimal_full_run_fixed"
    / "predictions_parts"
)
READY_PATH = ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet"
REPORTS = ROOT / "reports" / "stage19f"
MODEL_PATH = ROOT / "models" / "production" / "catboost_24h.cbm"
FEATURE_CONFIG = (
    ROOT
    / "reports"
    / "stage7a"
    / "24h"
    / "20260711T090515Z"
    / "training_dataset_24h_feature_config.json"
)
EXPECTED_MODEL_HASH = "0c8e1b88b1cfaf95fb39e395e2fdc54f1b7abda22d8ac00e1d6f561ab9110a0c"
EXPECTED_FEATURE_HASH = (
    "bf0ce06aface9fc3539a95df2557728e57d987a9e0c0e5ac4a195a53b47f6d96"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def feature_hash() -> str:
    config = json.loads(FEATURE_CONFIG.read_text(encoding="utf-8"))
    features = [*config["numerical_features"], *config["categorical_features"]]
    return sha256_bytes(json.dumps(features, separators=(",", ":")).encode())


def accident_indexes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    accidents = pd.read_parquet(
        READY_PATH, columns=["road_segment_id", "accident_datetime"]
    )
    accidents["road_segment_id"] = accidents["road_segment_id"].astype(str)
    accidents["event_hour"] = pd.to_datetime(accidents["accident_datetime"]).dt.floor(
        "h"
    )
    counts = (
        accidents.groupby(["road_segment_id", "event_hour"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    indexes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for segment, group in counts.groupby("road_segment_id", sort=False):
        ordered = group.sort_values("event_hour")
        indexes[str(segment)] = (
            ordered.event_hour.astype("datetime64[ns]").astype("int64").to_numpy(),
            ordered.event_count.cumsum().to_numpy(dtype=np.int64),
        )
    return indexes


def historical_scores(
    frame: pd.DataFrame, indexes: dict[str, tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    scores = np.zeros(len(frame), dtype=np.int64)
    for segment, positions in frame.groupby(
        "road_segment_id", sort=False
    ).indices.items():
        event_index = indexes.get(str(segment))
        if event_index is None:
            continue
        times, cumulative = event_index
        selected = np.asarray(positions)
        query = (
            frame.iloc[selected]
            .prediction_datetime.astype("datetime64[ns]")
            .astype("int64")
            .to_numpy()
        )
        location = np.searchsorted(times, query, side="left")
        scores[selected] = np.where(location > 0, cumulative[location - 1], 0)
    return scores


def order_for_score(hour: pd.DataFrame, score_column: str) -> pd.DataFrame:
    return hour.sort_values(
        [score_column, "road_segment_id"], ascending=[False, True], kind="stable"
    )


def ranking_metrics(
    frame: pd.DataFrame, score_column: str
) -> tuple[dict[str, float], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for timestamp, hour in frame.groupby("prediction_datetime", sort=True):
        ranked = order_for_score(hour, score_column)
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
                float(selected.sum() / positives) if positives else None
            )
            row[f"lift_{size}"] = (
                precision / (positives / len(ranked)) if positives else None
            )
        records.append(row)
    table = pd.DataFrame(records)
    mean = table.mean(numeric_only=True)
    return (
        {
            "precision_at_10": float(mean["precision_10"]),
            "precision_at_20": float(mean["precision_20"]),
            "precision_at_50": float(mean["precision_50"]),
            "recall_at_1pct": float(mean["recall_40"]),
            "recall_at_5pct": float(mean["recall_199"]),
            "recall_at_10pct": float(mean["recall_397"]),
            "lift_at_1pct": float(mean["lift_40"]),
            "lift_at_5pct": float(mean["lift_199"]),
            "lift_at_10pct": float(mean["lift_397"]),
        },
        records,
    )


def metric_bundle(
    y: np.ndarray, scores: np.ndarray, ranking: dict[str, float]
) -> dict[str, float]:
    return {
        "natural_prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        **ranking,
    }


def comparison(
    catboost: dict[str, float], baseline: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for name, catboost_value in catboost.items():
        baseline_value = baseline[name]
        difference = catboost_value - baseline_value
        result[name] = {
            "catboost": catboost_value,
            "baseline": baseline_value,
            "absolute_difference": difference,
            "relative_improvement_pct": (difference / baseline_value * 100)
            if baseline_value
            else None,
        }
    return result


def precision10_investigation(frame: pd.DataFrame) -> dict[str, object]:
    top_counts: dict[str, int] = {}
    cutoff_ties: list[int] = []
    precision_with_ties: list[float] = []
    positive_top20 = 0
    positive_top50 = 0
    for _, hour in frame.groupby("prediction_datetime", sort=True):
        ranked = order_for_score(hour, "raw_model_score")
        top10 = ranked.head(10)
        for segment in top10.road_segment_id.astype(str):
            top_counts[segment] = top_counts.get(segment, 0) + 1
        cutoff = top10.raw_model_score.iloc[-1]
        tied = ranked.loc[ranked.raw_model_score.eq(cutoff)]
        cutoff_ties.append(len(tied))
        selected = ranked.loc[ranked.raw_model_score.ge(cutoff)]
        precision_with_ties.append(float(selected.target_24h.mean()))
        positive_top20 += int(ranked.head(20).target_24h.sum())
        positive_top50 += int(ranked.head(50).target_24h.sum())
    top10_positive_rows = int(
        frame.groupby("prediction_datetime")
        .apply(
            lambda x: order_for_score(x, "raw_model_score").head(10).target_24h.sum()
        )
        .sum()
    )
    prevalence = float(frame.target_24h.mean())
    return {
        "unique_segments_in_top10": len(top_counts),
        "top10_positive_rows": top10_positive_rows,
        "positive_rows_in_top20": positive_top20,
        "positive_rows_in_top50": positive_top50,
        "cutoff_tie_size": {
            "maximum": max(cutoff_ties),
            "mean": float(np.mean(cutoff_ties)),
            "median": float(np.median(cutoff_ties)),
        },
        "deterministic_tie_break": "road_segment_id ascending after score descending",
        "precision_at_10_with_all_cutoff_ties": float(np.mean(precision_with_ties)),
        "natural_prevalence": prevalence,
        "expected_positive_rows_in_top10_at_random": float(
            prevalence * 10 * frame.prediction_datetime.nunique()
        ),
        "top10_segment_frequency": dict(
            sorted(top_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def main() -> int:
    parts = sorted(PARTS_DIR.glob("part_*.parquet"))
    if len(parts) != 28:
        raise ValueError(f"expected_28_stage19e_parts_got_{len(parts)}")
    model_hash = sha256_bytes(MODEL_PATH.read_bytes())
    frozen_feature_hash = feature_hash()
    if (
        model_hash != EXPECTED_MODEL_HASH
        or frozen_feature_hash != EXPECTED_FEATURE_HASH
    ):
        raise ValueError("production_fingerprint_changed")
    indexes = accident_indexes()
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
    grid = pd.concat(frames, ignore_index=True).sort_values(
        ["prediction_datetime", "road_segment_id"]
    )
    if (
        len(grid) != 666624
        or grid.duplicated(["road_segment_id", "prediction_datetime"]).any()
    ):
        raise ValueError("stage19e_grid_integrity_failure")
    cat_ranking, _ = ranking_metrics(grid, "raw_model_score")
    baseline_ranking, _ = ranking_metrics(grid, "baseline_score")
    y = grid.target_24h.to_numpy(dtype=np.uint8)
    catboost_metrics = metric_bundle(y, grid.raw_model_score.to_numpy(), cat_ranking)
    baseline_metrics = metric_bundle(
        y, grid.baseline_score.to_numpy(), baseline_ranking
    )
    precision = precision10_investigation(grid)
    grid["model_rank"] = grid.groupby("prediction_datetime")["raw_model_score"].rank(
        method="first", ascending=False
    )
    top_rows = []
    for segment, group in grid.groupby("road_segment_id", sort=False):
        top_rows.append(
            {
                "road_segment_id": segment,
                "times_in_top10": int((group.model_rank <= 10).sum()),
                "times_in_top20": int((group.model_rank <= 20).sum()),
                "times_in_top50": int((group.model_rank <= 50).sum()),
                "historical_accident_count": int(group.baseline_score.max()),
                "positive_rows": int(group.target_24h.sum()),
                "mean_model_score": float(group.raw_model_score.mean()),
                "mean_baseline_score": float(group.baseline_score.mean()),
            }
        )
    top_segments = (
        pd.DataFrame(top_rows)
        .sort_values(
            ["times_in_top10", "times_in_top20", "times_in_top50", "road_segment_id"],
            ascending=[False, False, False, True],
        )
        .head(30)
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "baseline_metrics.json", baseline_metrics)
    write_json(
        REPORTS / "comparison_table.json",
        comparison(catboost_metrics, baseline_metrics),
    )
    write_json(REPORTS / "precision10_investigation.json", precision)
    write_json(
        REPORTS / "stage19f_summary.json",
        {
            "grid": {
                "rows": len(grid),
                "timestamps": int(grid.prediction_datetime.nunique()),
                "segments": int(grid.road_segment_id.nunique()),
            },
            "production_fingerprint": {
                "model_hash": model_hash,
                "feature_order_hash": frozen_feature_hash,
            },
            "catboost": catboost_metrics,
            "baseline": baseline_metrics,
            "comparison": comparison(catboost_metrics, baseline_metrics),
            "precision10_investigation": precision,
        },
    )
    top_segments.to_csv(
        REPORTS / "top_ranked_segments.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    (REPORTS / "baseline_metrics.md").write_text(
        "# Stage 19F baseline metrics\n\n" + json.dumps(baseline_metrics, indent=2),
        encoding="utf-8",
    )
    (REPORTS / "comparison_table.md").write_text(
        "# Stage 19F comparison\n\n"
        + json.dumps(comparison(catboost_metrics, baseline_metrics), indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "catboost": catboost_metrics,
                "baseline": baseline_metrics,
                "precision10": precision,
                "rows": len(grid),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
