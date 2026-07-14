"""Validation-only frozen simple blend selection for Stage 19I."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data/audit/stage19i_simple/validation_scores.parquet"
M = ROOT / "models/stage19i_simple"
R = ROOT / "reports/stage19i_simple"
COLS = [
    "score_catboost_stage19h",
    "score_hist_gradient_boosting",
    "score_logistic_regression",
]


def metric(y, s):
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def main():
    d = pd.read_parquet(IN)
    x = d[COLS].rank(pct=True)
    y = d.target_24h.to_numpy()
    order = np.argsort(pd.to_datetime(d.datetime_hour).to_numpy(), kind="stable")
    folds = np.array_split(order, 3)

    def rec(w):
        s = x.to_numpy() @ np.array(w)
        a = metric(y, s)
        subs = [metric(y[z], s[z]) for z in folds]
        return {
            "weights": dict(zip(COLS, map(float, w))),
            "validation": a,
            "subperiods": subs,
            "minimum_subperiod_pr_auc": min(q["pr_auc"] for q in subs),
            "std_subperiod_pr_auc": float(np.std([q["pr_auc"] for q in subs])),
        }

    two = [rec([i / 20, 1 - i / 20, 0]) for i in range(21)]
    three = [
        rec([a / 10, b / 10, 1 - (a + b) / 10])
        for a in range(11)
        for b in range(11 - a)
    ]

    def key(q):
        return (
            q["validation"]["pr_auc"],
            q["minimum_subperiod_pr_auc"],
            q["validation"]["roc_auc"],
            -abs(q["weights"][COLS[0]] - 0.5),
        )

    best2 = max(two, key=key)
    best3 = max(three, key=key)
    accept = (
        best3["validation"]["pr_auc"] >= best2["validation"]["pr_auc"] * 1.01
        and best3["minimum_subperiod_pr_auc"] >= best2["minimum_subperiod_pr_auc"]
    )
    chosen = best3 if accept else best2
    M.mkdir(parents=True, exist_ok=True)
    R.mkdir(parents=True, exist_ok=True)
    norm = {
        "method": "validation_global_percentile_rank",
        "source": str(IN),
        "columns": COLS,
    }
    equal = {
        "components": COLS[:2],
        "weights": {COLS[0]: 0.5, COLS[1]: 0.5},
        "normalization": norm["method"],
        "status": "frozen",
    }
    frozen = {
        **chosen,
        "normalization": norm["method"],
        "status": "frozen",
        "selection_rule": "PR-AUC,min subperiod PR-AUC,ROC-AUC,equal proximity",
    }
    frozen["configuration_checksum"] = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode()
    ).hexdigest()
    for p, v in [
        (M / "normalization_config.json", norm),
        (M / "equal_ensemble_config.json", equal),
        (M / "weighted_ensemble_config.json", frozen),
        (
            M / "component_manifest.json",
            {
                "components": COLS,
                "logistic_warning": "max_iter_300_convergence_warning",
            },
        ),
        (
            R / "weighted_search.json",
            {
                "two_model": two,
                "three_model": three,
                "best_two": best2,
                "best_three": best3,
                "three_model_accepted": accept,
            },
        ),
        (
            R / "validation_subperiods.json",
            {"subperiod_sizes": [len(z) for z in folds]},
        ),
        (R / "frozen_ensemble_configs.json", {"equal": equal, "weighted": frozen}),
    ]:
        p.write_text(json.dumps(v, indent=2), encoding="utf-8")
    (R / "weight_search_summary.md").write_text(
        "# Stage 19I weight search\n\n"
        + json.dumps(
            {"best_two": best2, "best_three": best3, "three_accepted": accept}, indent=2
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
