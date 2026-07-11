"""Exercise the Stage 8C path at a fixed historical hour for reproducible demos.

It writes demo reports and exports; it is not a live scheduling entry point.
"""

from __future__ import annotations
import json, sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from inference.predict import run


def main() -> None:
    output = ROOT / "reports" / "stage8c" / "demo_20220908T150000"
    summary = run("2022-09-08 15:00:00", output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "build_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "builds": {h: v["build"] for h, v in summary["horizons"].items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    (output / "validation_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "validation": {
                    h: v["validation"] for h, v in summary["horizons"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    (output / "prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
