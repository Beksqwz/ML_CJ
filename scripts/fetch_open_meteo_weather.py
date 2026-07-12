"""Fetch hourly historical weather for Astana from Open-Meteo Archive API.

Run:
    py scripts/fetch_open_meteo_weather.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "external_data"
DEFAULT_ACCIDENTS = (
    PROJECT_ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet"
)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ASTANA_LAT = 51.1694
ASTANA_LON = 71.4491
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "sunshine_duration",
    "wind_speed_10m",
    "wind_gusts_10m",
]


def configure_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch hourly Astana weather from Open-Meteo."
    )
    parser.add_argument("--accidents", type=Path, default=DEFAULT_ACCIDENTS)
    parser.add_argument("--start", type=str, help="Optional start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, help="Optional end date YYYY-MM-DD.")
    parser.add_argument("--latitude", type=float, default=ASTANA_LAT)
    parser.add_argument("--longitude", type=float, default=ASTANA_LON)
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=EXTERNAL_ROOT / "weather_astana_hourly.parquet",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=EXTERNAL_ROOT / "weather_astana_hourly.csv"
    )
    return parser.parse_args()


def resolve_range(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    if args.start and args.end:
        return pd.Timestamp(args.start), pd.Timestamp(args.end)
    accidents = pd.read_parquet(args.accidents)
    timestamps = pd.to_datetime(accidents["accident_datetime"])
    return timestamps.min().floor("D"), timestamps.max().ceil("D")


def year_chunks(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start_date
    while current <= end_date:
        chunk_end = min(pd.Timestamp(year=current.year, month=12, day=31), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks


def request_weather(
    start_date: pd.Timestamp, end_date: pd.Timestamp, latitude: float, longitude: float
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Almaty",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
        "cell_selection": "land",
        # ERA5 gives consistent long historical coverage, useful for 2011-2026 ML.
        "models": "era5",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo response has no hourly data: {payload}")
    hourly = payload["hourly"]
    dataframe = pd.DataFrame(hourly)
    dataframe["datetime_hour"] = pd.to_datetime(dataframe["time"])
    dataframe = dataframe.drop(columns=["time"])
    dataframe["latitude_source"] = payload.get("latitude")
    dataframe["longitude_source"] = payload.get("longitude")
    dataframe["timezone_source"] = payload.get("timezone")
    return dataframe


def make_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / "weather" / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def write_report(report_dir: Path, summary: dict[str, Any]) -> None:
    (report_dir / "weather_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "Open-Meteo weather fetch report",
        f"Rows: {summary['rows']}",
        f"Range: {summary['start']} to {summary['end']}",
        f"Missing values: {summary['missing_values']}",
        f"Output Parquet: {summary['output_parquet']}",
    ]
    (report_dir / "weather_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    configure_logging()
    try:
        args = parse_args()
        start_date, end_date = resolve_range(args)
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in year_chunks(start_date, end_date):
            LOGGER.info(
                "Fetching weather %s to %s", chunk_start.date(), chunk_end.date()
            )
            frames.append(
                request_weather(chunk_start, chunk_end, args.latitude, args.longitude)
            )
            time.sleep(0.5)
        weather = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["datetime_hour"])
            .sort_values("datetime_hour")
        )
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        weather.to_parquet(args.output_parquet, index=False, engine="pyarrow")
        weather.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        report_dir = make_report_dir()
        missing_values = {
            column: int(weather[column].isna().sum()) for column in HOURLY_VARIABLES
        }
        summary = {
            "source": "Open-Meteo Historical Weather API /v1/archive",
            "source_url": OPEN_METEO_URL,
            "latitude": args.latitude,
            "longitude": args.longitude,
            "timezone": "Asia/Almaty",
            "model": "era5",
            "hourly_variables": HOURLY_VARIABLES,
            "start": str(start_date.date()),
            "end": str(end_date.date()),
            "rows": int(len(weather)),
            "missing_values": missing_values,
            "output_parquet": str(args.output_parquet.resolve()),
            "output_csv": str(args.output_csv.resolve()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_report(report_dir, summary)
        print(f"Rows: {summary['rows']}")
        print(f"Missing values: {missing_values}")
        print(f"Output: {summary['output_parquet']}")
        print(f"Report: {report_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("Weather fetch failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
