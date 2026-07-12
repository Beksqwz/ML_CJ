"""Stage 11 parallel reconstruction with frozen Stage 7A boundaries and row-key audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_stage7a_splits import apply_train_only_transforms, feature_configuration
from run_stage7d_weather_experiment import add_features, risk_features

OUT = ROOT / "data" / "processed" / "stage11"
REPORTS = ROOT / "reports" / "stage11"


def fixed_splits(data: pd.DataFrame, boundaries: dict[str, object]) -> dict[str, pd.DataFrame]:
    ts = pd.to_datetime(data["datetime_hour"], errors="raise")
    train_end = pd.Timestamp(boundaries["train_end_exclusive"])
    validation_start = pd.Timestamp(boundaries["validation_start"])
    validation_end = pd.Timestamp(boundaries["validation_end_exclusive"])
    test_start = pd.Timestamp(boundaries["test_start"])
    return {
        "train": data.loc[ts < train_end].copy(),
        "validation": data.loc[(ts >= validation_start) & (ts < validation_end)].copy(),
        "test": data.loc[ts >= test_start].copy(),
    }


def audit(rebuilt: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Verify exact row keys and representative old features before graph columns exist."""
    checks: dict[str, object] = {}
    for split, frame in rebuilt.items():
        frozen = pd.read_parquet(ROOT / "data" / "processed" / "stage7d" / f"training_dataset_1h_{split}.parquet")
        keys = ["datetime_hour", "target_1h", "segment_accidents_total_prior", "road_length"]
        left = frame[keys].sort_values(keys[:2]).reset_index(drop=True)
        right = frozen[keys].sort_values(keys[:2]).reset_index(drop=True)
        checks[split] = {
            "rows_rebuilt": len(frame), "rows_stage7d": len(frozen),
            "target_matches": bool(left["target_1h"].equals(right["target_1h"])),
            "segment_accidents_total_prior_matches": bool(left["segment_accidents_total_prior"].equals(right["segment_accidents_total_prior"])),
            "road_length_matches": bool(np.allclose(left["road_length"], right["road_length"], equal_nan=True)),
        }
    return checks


def main() -> None:
    report7a = json.loads(next((ROOT / "reports" / "stage7a" / "1h").glob("*/training_dataset_1h_stage7a_report.json")).read_text(encoding="utf-8"))
    raw = pd.read_parquet(ROOT / "data" / "processed" / "training_dataset_1h.parquet")
    splits = fixed_splits(raw, report7a["split_boundaries"])
    config = feature_configuration(raw, "target_1h")
    transformed, transform_info = apply_train_only_transforms(splits, config, 100)
    weather, derived, _ = risk_features()
    enhanced = {}
    for name, frame in transformed.items():
        enhanced[name], _, _ = add_features(frame, weather, derived)
    checks = audit(enhanced)
    if not all(all(v for k, v in item.items() if k.endswith("matches")) and item["rows_rebuilt"] == item["rows_stage7d"] for item in checks.values()):
        raise AssertionError(f"Stage 11 reconstruction audit failed: {checks}")
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    for name, frame in enhanced.items():
        frame.to_parquet(OUT / f"training_dataset_1h_{name}_with_keys.parquet", index=False)
    report = {
        "stage": "11", "horizon": "1h", "status": "parallel_split_reconstruction_complete",
        "parallel_split_not_modification": True, "source_dataset": str((ROOT / "data" / "processed" / "training_dataset_1h.parquet").resolve()),
        "frozen_stage7a_boundaries": report7a["split_boundaries"], "row_level_reconstruction_audit": checks,
        "road_segment_id": "Retained only as a join key; excluded from model features.",
        "stage7a_train_only_transform": transform_info,
        "next_step": "Add causal graph features, then train and evaluate the isolated candidate.",
    }
    (REPORTS / "stage11_reconstruction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

if __name__ == "__main__": main()
