"""Reconstruct Stage 19H audit artifacts from its completed benchmark run."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "stage19h"
MODELS = ROOT / "models" / "stage19h"
AUDIT = ROOT / "data" / "audit" / "stage19h"
PRODUCTION = ROOT / "models" / "production" / "catboost_24h.cbm"
FEATURE_HASH = "bf0ce06aface9fc3539a95df2557728e57d987a9e0c0e5ac4a195a53b47f6d96"
WINDOWS = {
    "autumn": "2024-09-30T19:00:00+05:00/2024-10-07T18:00:00+05:00",
    "winter": "2025-01-01T00:00:00+05:00/2025-01-07T23:00:00+05:00",
    "spring": "2025-04-01T00:00:00+05:00/2025-04-07T23:00:00+05:00",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    seasonal = json.loads((REPORTS / "seasonal_natural_metrics.json").read_text())
    training = json.loads((REPORTS / "training_summary.json").read_text())
    stability = json.loads((REPORTS / "top10_stability_comparison.json").read_text())
    models = list(seasonal["autumn"])
    metrics = (
        "pr_auc",
        "roc_auc",
        "recall_at_5pct",
        "lift_at_5pct",
        "precision_at_20",
        "precision_at_50",
    )
    robust = {
        model: {
            metric: {
                key: float(
                    func([seasonal[season][model][metric] for season in seasonal])
                )
                for key, func in (
                    ("mean", np.mean),
                    ("std", np.std),
                    ("min", np.min),
                    ("max", np.max),
                )
            }
            for metric in metrics
        }
        for model in models
    }
    sizes = {"stage7b_frozen_reference": PRODUCTION.stat().st_size}
    hashes = {"stage7b_frozen_reference": sha(PRODUCTION)}
    for name, info in training.items():
        path = Path(info["model_path"])
        sizes[name], hashes[name] = path.stat().st_size, sha(path)
    warnings = {name: [] for name in models}
    warnings["logistic_regression"].append("lbfgs_max_iter_300_convergence_warning")
    warnings["extra_trees"].append("artifact_size_1GB_operationally_impractical")
    efficiency = {
        name: {
            "artifact_size_bytes": sizes.get(name),
            "training_seconds": training.get(name, {}).get("training_seconds"),
            "seasonal_inference_seconds": None,
            "warnings": warnings[name] or None,
            "operational_suitability": "not_recommended"
            if name in {"extra_trees", "logistic_regression"}
            else "measured",
        }
        for name in models
    }
    stability["cross_season_note"] = (
        "Exact per-segment cross-season sets were not persisted; null rather than fabricated."
    )
    for model in models:
        stability.setdefault("cross_season", {})[model] = {
            "pairwise_jaccard": None,
            "three_season_intersection": None,
            "concentration_rate": float(
                np.mean(
                    [
                        stability[s][model]["permanently_top10_segments"] / 10
                        for s in seasonal
                    ]
                )
            ),
        }
    completed = models
    checkpoint = {
        "status": "completed",
        "completed_models": completed,
        "completed_windows": list(seasonal),
        "model_hashes": hashes,
        "feature_hash": FEATURE_HASH,
        "metric_checksums": {
            s: hashlib.sha256(
                json.dumps(seasonal[s], sort_keys=True).encode()
            ).hexdigest()
            for s in seasonal
        },
        "created_at": datetime.now(UTC).isoformat(),
        "reconstructed_from_completed_run": True,
    }
    dump(AUDIT / "evaluation_checkpoint.json", checkpoint)
    dump(
        REPORTS / "validation_metrics.json",
        {"validation": training, "feature_count": 77, "feature_hash": FEATURE_HASH},
    )
    dump(REPORTS / "robustness_summary.json", robust)
    dump(REPORTS / "top10_stability_comparison.json", stability)
    dump(REPORTS / "efficiency_comparison.json", efficiency)
    decision = {
        "best_production_model": "stage7b_frozen_reference",
        "best_experimental_tree_candidate": "catboost_candidate",
        "best_small_budget_comparator": "historical_baseline / unconverged_logistic_regression",
        "replacement_approved": False,
        "replacement_reason": "No experimental candidate improves frozen Recall@5% by 3% while preserving Lift@5%.",
        "recommended_next_step": "hybrid operational ranking study, not more model-family training",
    }
    dump(REPORTS / "winner_decision.json", decision)
    summary = {
        "decision": decision,
        "robustness": robust,
        "checkpoint": str(AUDIT / "evaluation_checkpoint.json"),
    }
    dump(REPORTS / "summary.json", summary)
    (REPORTS / "summary.md").write_text(
        "# Stage 19H finalization\n\n" + json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
