"""Validate stage-5 external data sources and write a manifest.

Run:
    py scripts/validate_external_data.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "external_data"


def configure_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stage-5 external data.")
    parser.add_argument(
        "--calendar",
        type=Path,
        default=EXTERNAL_ROOT / "calendar_features_hourly.parquet",
    )
    parser.add_argument(
        "--weather", type=Path, default=EXTERNAL_ROOT / "weather_astana_hourly.parquet"
    )
    parser.add_argument(
        "--pois", type=Path, default=EXTERNAL_ROOT / "pois_astana_osm.parquet"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXTERNAL_ROOT / "stage5_external_data_manifest.json",
    )
    return parser.parse_args()


def make_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / "validation" / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def validate(args: argparse.Namespace) -> dict[str, Any]:
    calendar = pd.read_parquet(args.calendar)
    weather = pd.read_parquet(args.weather)
    pois = pd.read_parquet(args.pois)

    calendar_hours = pd.to_datetime(calendar["datetime_hour"])
    weather_hours = pd.to_datetime(weather["datetime_hour"])
    common_hours = set(calendar_hours).intersection(set(weather_hours))

    weather_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calendar": {
            "path": str(args.calendar.resolve()),
            "rows": int(len(calendar)),
            "columns": list(calendar.columns),
            "start": str(calendar_hours.min()),
            "end": str(calendar_hours.max()),
            "duplicate_hours": int(calendar_hours.duplicated().sum()),
            "holiday_hours": int(calendar["is_holiday"].sum()),
            "school_break_hours": int(calendar["is_school_break"].sum())
            if "is_school_break" in calendar
            else None,
        },
        "weather": {
            "path": str(args.weather.resolve()),
            "rows": int(len(weather)),
            "columns": list(weather.columns),
            "start": str(weather_hours.min()),
            "end": str(weather_hours.max()),
            "duplicate_hours": int(weather_hours.duplicated().sum()),
            "missing_values": {
                column: int(weather[column].isna().sum())
                for column in weather_vars
                if column in weather.columns
            },
        },
        "pois": {
            "path": str(args.pois.resolve()),
            "rows": int(len(pois)),
            "columns": list(pois.columns),
            "missing_coordinates": int(
                pois["latitude"].isna().sum() + pois["longitude"].isna().sum()
            ),
            "category_counts": {
                str(key): int(value)
                for key, value in pois["poi_category"].value_counts().to_dict().items()
            },
        },
        "hour_alignment": {
            "calendar_weather_common_hours": int(len(common_hours)),
            "calendar_only_hours": int(len(set(calendar_hours) - set(weather_hours))),
            "weather_only_hours": int(len(set(weather_hours) - set(calendar_hours))),
        },
        "known_limitations": [
            "Weather is city-level hourly ERA5 data for central Astana, not road-segment microclimate.",
            "Visibility is not included in the selected Open-Meteo Archive variable set; weather_code/cloud_cover are available instead.",
            "School calendar columns are heuristic and should be replaced if official year-specific school break data is obtained.",
            "POIs come from OpenStreetMap and may be incomplete or unevenly tagged.",
        ],
    }
    return summary


def write_report(
    report_dir: Path, manifest_path: Path, summary: dict[str, Any]
) -> None:
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "external_data_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "Stage 5 external data validation",
        f"Calendar rows: {summary['calendar']['rows']}",
        f"Weather rows: {summary['weather']['rows']}",
        f"POI rows: {summary['pois']['rows']}",
        f"Calendar/weather common hours: {summary['hour_alignment']['calendar_weather_common_hours']}",
        f"Manifest: {manifest_path.resolve()}",
    ]
    (report_dir / "external_data_validation_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    configure_logging()
    try:
        args = parse_args()
        report_dir = make_report_dir()
        summary = validate(args)
        write_report(report_dir, args.manifest.resolve(), summary)
        print(f"Calendar rows: {summary['calendar']['rows']}")
        print(f"Weather rows: {summary['weather']['rows']}")
        print(f"POI rows: {summary['pois']['rows']}")
        print(
            f"Common hours: {summary['hour_alignment']['calendar_weather_common_hours']}"
        )
        print(f"Manifest: {args.manifest.resolve()}")
        print(f"Report: {report_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("External data validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
