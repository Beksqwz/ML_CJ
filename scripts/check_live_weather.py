"""Print current OpenWeather conditions and a compact live model summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_service import AccidentRiskPredictor  # noqa: E402


def main() -> None:
    result = AccidentRiskPredictor().predict_current_city("1h")
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in ("weather_mode", "live_weather", "summary")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
