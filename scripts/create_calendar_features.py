"""Create hourly Kazakhstan calendar features for RoadRisk ML.

Run:
    py scripts/create_calendar_features.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import holidays
import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "external_data"
DEFAULT_ACCIDENTS = PROJECT_ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet"


def configure_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hourly Kazakhstan calendar features.")
    parser.add_argument("--accidents", type=Path, default=DEFAULT_ACCIDENTS)
    parser.add_argument("--start", type=str, help="Optional start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, help="Optional end date YYYY-MM-DD.")
    parser.add_argument("--output-parquet", type=Path, default=EXTERNAL_ROOT / "calendar_features_hourly.parquet")
    parser.add_argument("--output-csv", type=Path, default=EXTERNAL_ROOT / "calendar_features_hourly.csv")
    return parser.parse_args()


def resolve_range(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    if args.start and args.end:
        return pd.Timestamp(args.start), pd.Timestamp(args.end)
    accidents = pd.read_parquet(args.accidents)
    timestamps = pd.to_datetime(accidents["accident_datetime"])
    return timestamps.min().floor("D"), timestamps.max().ceil("D")


def make_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / "calendar" / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def create_calendar(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    hours = pd.date_range(start=start_date, end=end_date + pd.Timedelta(hours=23), freq="h")
    years = range(int(start_date.year), int(end_date.year) + 1)
    kz_holidays = holidays.country_holidays("KZ", years=years, observed=True, language="en_US")
    holiday_dates = {pd.Timestamp(day).date(): name for day, name in kz_holidays.items()}

    data = pd.DataFrame({"datetime_hour": hours})
    data["date"] = data["datetime_hour"].dt.date.astype("string")
    data["year"] = data["datetime_hour"].dt.year.astype("Int64")
    data["month"] = data["datetime_hour"].dt.month.astype("Int64")
    data["day"] = data["datetime_hour"].dt.day.astype("Int64")
    data["hour"] = data["datetime_hour"].dt.hour.astype("Int64")
    data["weekday"] = data["datetime_hour"].dt.weekday.astype("Int64")
    data["is_weekend"] = data["weekday"].ge(5).astype("boolean")
    data["is_holiday"] = data["datetime_hour"].dt.date.map(lambda day: day in holiday_dates).astype("boolean")
    data["holiday_name"] = data["datetime_hour"].dt.date.map(lambda day: holiday_dates.get(day, "")).astype("string")
    data["is_day_before_holiday"] = (data["datetime_hour"].dt.date + pd.Timedelta(days=1)).map(
        lambda day: day in holiday_dates
    ).astype("boolean")
    data["is_day_after_holiday"] = (data["datetime_hour"].dt.date - pd.Timedelta(days=1)).map(
        lambda day: day in holiday_dates
    ).astype("boolean")
    data["is_rush_hour"] = data["hour"].isin([7, 8, 9, 17, 18, 19]).astype("boolean")
    data["season"] = data["month"].map(
        {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    ).astype("string")
    # Heuristic school calendar features for Kazakhstan-style academic years.
    # Exact break dates can be replaced later with official year-specific data.
    data["is_school_year"] = data["month"].isin([9, 10, 11, 12, 1, 2, 3, 4, 5]).astype("boolean")
    data["is_school_summer_break"] = data["month"].isin([6, 7, 8]).astype("boolean")
    data["is_school_winter_break"] = (
        ((data["month"] == 12) & data["day"].ge(29)) | ((data["month"] == 1) & data["day"].le(8))
    ).astype("boolean")
    data["is_school_spring_break"] = ((data["month"] == 3) & data["day"].between(21, 31)).astype("boolean")
    data["is_school_autumn_break"] = ((data["month"] == 10) & data["day"].ge(28)).astype("boolean")
    data["is_school_break"] = (
        data["is_school_summer_break"]
        | data["is_school_winter_break"]
        | data["is_school_spring_break"]
        | data["is_school_autumn_break"]
    ).astype("boolean")
    return data


def write_report(report_dir: Path, summary: dict[str, Any]) -> None:
    (report_dir / "calendar_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "Calendar features report",
        f"Rows: {summary['rows']}",
        f"Range: {summary['start']} to {summary['end']}",
        f"Holiday hours: {summary['holiday_hours']}",
        f"Output Parquet: {summary['output_parquet']}",
    ]
    (report_dir / "calendar_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_logging()
    try:
        args = parse_args()
        start_date, end_date = resolve_range(args)
        calendar = create_calendar(start_date, end_date)
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        calendar.to_parquet(args.output_parquet, index=False, engine="pyarrow")
        calendar.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        report_dir = make_report_dir()
        summary = {
            "source": "python-holidays country_holidays('KZ', observed=True)",
            "start": str(start_date.date()),
            "end": str(end_date.date()),
            "rows": int(len(calendar)),
            "holiday_hours": int(calendar["is_holiday"].sum()),
            "school_break_hours": int(calendar["is_school_break"].sum()),
            "unique_holidays": sorted(name for name in calendar["holiday_name"].dropna().unique().tolist() if name),
            "output_parquet": str(args.output_parquet.resolve()),
            "output_csv": str(args.output_csv.resolve()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_report(report_dir, summary)
        print(f"Rows: {summary['rows']}")
        print(f"Holiday hours: {summary['holiday_hours']}")
        print(f"Output: {summary['output_parquet']}")
        print(f"Report: {report_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("Calendar feature creation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
