"""Create a post-refactor SHA-256 snapshot of immutable Stage 8C demo outputs."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "reports" / "stage8c" / "demo_20220908T150000"
REPORTS = ROOT / "reports" / "stage9a"
FILES = (
    "predictions_current_1h.json",
    "predictions_current_24h.json",
    "risk_map_1h.geojson",
    "risk_map_24h.geojson",
)
LOGGER = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of an existing output file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Write a post-refactor snapshot, not a claimed pre-refactor comparison."""
    checksums = {
        name: {
            "sha256": sha256(OUTPUTS / name),
            "bytes": (OUTPUTS / name).stat().st_size,
        }
        for name in FILES
    }
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_kind": "post_refactor_baseline",
        "limitation": "No pre-refactor checksum snapshot exists; this file does not claim a before/after comparison.",
        "future_use": "Use this snapshot as the baseline for subsequent output-integrity checks.",
        "files": checksums,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "output_checksums.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Stage 8C output checksums written to %s", REPORTS)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    main()
