"""Derive Stage 8B demonstration payloads from stored Stage 8A local evidence.

The script consumes frozen explanations and does not retrain or recompute SHAP.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from recommendations.engine import recommend

OUT = ROOT / "reports" / "stage8b"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = {
        "1h": "Stage 7D / catboost_1h_weather_experiment.cbm",
        "24h": "Stage 7B / catboost_24h.cbm",
    }
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "models": models,
        "horizons": {},
    }
    for horizon, version in models.items():
        base = ROOT / "reports" / "stage8a" / horizon
        local = json.loads(
            (base / "local_explanations.json").read_text(encoding="utf-8")
        )
        shap = pd.read_parquet(base / "shap_values.parquet").set_index(
            "sample_row_index"
        )
        rows = []
        for category, examples in local.items():
            for example in examples:
                index = int(example["sample_row_index"])
                shap_values = {
                    name: float(value)
                    for name, value in shap.loc[index].items()
                    if name
                    not in {
                        "datetime_hour",
                        f"target_{horizon}",
                        "prediction_probability",
                        "raw_prediction",
                        "base_value_log_odds",
                    }
                }
                output = recommend(
                    probability=float(example["probability"]),
                    shap_values=shap_values,
                    feature_values=example["feature_values"],
                    model_horizon=horizon,
                    final_model_version=version,
                )
                output["demo_category"] = category
                output["sample_row_index"] = index
                output["true_label"] = int(example["true_label"])
                rows.append(output)
        for item in rows:
            for rec in item["recommendations"]:
                if rec["evidence"]["shap_value"] <= 0:
                    raise AssertionError(
                        "Recommendation without positive local SHAP evidence"
                    )
        (OUT / f"recommendations_{horizon}_demo.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        flat = pd.DataFrame(
            [
                {
                    **{
                        k: v
                        for k, v in row.items()
                        if k
                        not in {
                            "recommendations",
                            "top_positive_factors",
                            "top_negative_factors",
                        }
                    },
                    "top_positive_factors_json": json.dumps(
                        row["top_positive_factors"], ensure_ascii=False
                    ),
                    "top_negative_factors_json": json.dumps(
                        row["top_negative_factors"], ensure_ascii=False
                    ),
                    "recommendations_json": json.dumps(
                        row["recommendations"], ensure_ascii=False
                    ),
                }
                for row in rows
            ]
        )
        flat.to_parquet(OUT / f"recommendations_{horizon}_demo.parquet", index=False)
        report["horizons"][horizon] = {
            "demo_rows": len(rows),
            "recommendations": sum(len(x["recommendations"]) for x in rows),
            "all_recommendations_positive_shap_evidence": True,
        }
    (OUT / "stage8b_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
