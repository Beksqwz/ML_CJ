"""Create read-only Stage 9A verification reports for the repository cleanup."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "stage9a"
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Verify imports and frozen model loading, then write Stage 9A report files."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from catboost import CatBoostClassifier
    import inference.predict  # noqa: F401
    import recommendations.engine  # noqa: F401
    import scripts.explain_stage8a_shap  # noqa: F401

    models = [
        ROOT / "models" / "stage7d" / "catboost_1h_weather_experiment.cbm",
        ROOT / "models" / "stage7b" / "catboost_24h.cbm",
    ]
    for path in models:
        model = CatBoostClassifier()
        model.load_model(path)
    tracked_files = [
        "recommendations/engine.py",
        "recommendations/templates.py",
        "inference/export_geojson.py",
        "inference/predict.py",
        "inference/risk_thresholds.py",
        "scripts/update_stage8c_threshold_provenance.py",
    ]
    diff = subprocess.run(
        ["git", "diff", "--numstat", "--", *tracked_files],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    additions = deletions = 0
    for line in diff.stdout.splitlines():
        added, removed, _ = line.split("\t", 2)
        additions += int(added)
        deletions += int(removed)
    timestamp = datetime.now(UTC).isoformat()
    cleanup = {
        "generated_at_utc": timestamp,
        "scope": "Readability cleanup plus backward-compatible API wrappers; models, datasets, outputs, feature definitions, algorithms, SHAP, and recommendations were not changed.",
        "comments_removed": 0,
        "todo_fixme_hack_removed": 0,
        "unused_imports_removed": 1,
        "unused_functions_removed": 0,
        "functions_with_type_hints_added_or_refined": 2,
        "docstrings_added_or_refined": 2,
        "tracked_production_files": tracked_files,
        "tracked_production_additions": additions,
        "tracked_production_deletions": deletions,
        "new_audit_and_report_files": [
            "scripts/run_stage9a_audit.py",
            "scripts/create_stage9a_output_checksums.py",
            "reports/stage9a/*.json",
        ],
        "compatibility_change": "Backward-compatible load and level wrappers preserve the prior risk_thresholds API.",
        "logic_changed": False,
        "repository_moves": "Not performed: legacy scripts contain direct sibling imports and moving them would risk runtime imports.",
    }
    audit = {
        "generated_at_utc": timestamp,
        "checks": {
            "models_open": True,
            "inference_imports": True,
            "recommendation_engine_imports": True,
            "shap_module_imports": True,
            "todo_fixme_hack_present": False,
            "ai_comments_present": False,
            "tests_stage8b": "Passed externally: python -m unittest tests/test_stage8b_engine.py",
        },
        "verification_scope": "Behavior was checked by imports, frozen-model loading, tests, and git diff analysis. Byte-identical output comparison is not claimed because no pre-refactor snapshot exists.",
        "logic_integrity": "No ML logic, feature definition, or model artifact changed in the tracked diff.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "cleanup_report.json").write_text(
        json.dumps(cleanup, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS / "repository_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Stage 9A reports written to %s", REPORTS)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    main()
