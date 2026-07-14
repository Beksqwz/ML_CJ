"""One-time safe migration of the legacy __main__ Stage 19H joblib."""

from __future__ import annotations

import __main__
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml_service.preprocessing.train_only import TrainOnlyPreprocessor  # noqa: E402

MODELS = ROOT / "models" / "stage19h"
LEGACY = MODELS / "train_only_preprocessor.joblib"
V2 = MODELS / "train_only_preprocessor_v2.joblib"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    setattr(__main__, "TrainOnlyPreprocessor", TrainOnlyPreprocessor)
    legacy = joblib.load(LEGACY)
    v2 = TrainOnlyPreprocessor(list(legacy.numeric), list(legacy.categorical))
    v2.medians = dict(legacy.medians)
    v2.codebooks = {k: dict(v) for k, v in legacy.codebooks.items()}
    joblib.dump(v2, V2)
    meta = {
        "stable_class_module": "ml_service.preprocessing.train_only.TrainOnlyPreprocessor",
        "legacy_artifact_path": str(LEGACY),
        "legacy_sha256": sha(LEGACY),
        "migrated_artifact_sha256": sha(V2),
        "migration_method": "scoped __main__ compatibility shim; exact fitted-state transfer",
        "categorical_columns": v2.categorical,
        "numerical_columns": v2.numeric,
        "output_feature_count": len(v2.numeric) + len(v2.categorical),
        "created_at": datetime.now(UTC).isoformat(),
        "compatibility_status": "state_transferred",
        "warnings": [],
    }
    (MODELS / "train_only_preprocessor_v2.metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
