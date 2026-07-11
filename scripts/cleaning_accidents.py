"""Clean historical Astana accident datasets without altering their source files.

Run:
    python scripts/cleaning_accidents.py
    python scripts/cleaning_accidents.py --input path/to/accidents.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyarrow  # Required Parquet engine dependency; imported deliberately for deployment checks.
from shapely.geometry import Point


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "data_cleaning"
SUPPORTED_SUFFIXES = {".csv", ".parquet"}
ASTANA_REGION_BOUNDS = (69.0, 50.0, 74.0, 53.0)  # min_lon, min_lat, max_lon, max_lat
LOCAL_TIMEZONE = "Asia/Almaty"


@dataclass(frozen=True)
class CleaningMetrics:
    """Counts and outputs produced by one cleaning run."""

    input_path: str
    rows_before: int
    rows_after: int
    full_duplicates_removed: int
    objectid_duplicates_removed: int
    missing_objectid_removed: int
    missing_coordinates_removed: int
    missing_datetime_removed: int
    suspicious_coordinates: int
    created_columns: list[str]
    output_parquet: str
    output_csv: str
    suspicious_coordinates_report: str


def configure_logging() -> None:
    """Configure concise logging for command-line use."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Clean historical Astana accident CSV or Parquet data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Source CSV or Parquet file. Defaults to astana_accidents.parquet.",
    )
    return parser.parse_args()


def resolve_input(input_path: Path | None) -> Path:
    """Resolve an explicit dataset or use the project's standard Parquet/CSV pair."""
    if input_path is not None:
        path = input_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Only CSV and Parquet are supported, received: {path.suffix}"
            )
        return path

    for name in ("astana_accidents.parquet", "astana_accidents.csv"):
        path = PROJECT_ROOT / name
        if path.is_file():
            LOGGER.info(
                "Automatically selected input: %s", path.relative_to(PROJECT_ROOT)
            )
            return path
    raise FileNotFoundError(
        "Neither astana_accidents.parquet nor astana_accidents.csv exists in the project root."
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a source dataset into a new in-memory dataframe."""
    LOGGER.info("Reading %s", path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path, engine="pyarrow")


def require_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Fail early with one actionable message for missing required source fields."""
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Input dataset is missing required columns: {', '.join(missing)}"
        )


def remove_duplicates(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Remove complete duplicates first, then repeated non-empty object identifiers."""
    before = len(dataframe)
    deduplicated = dataframe.drop_duplicates().copy()
    full_removed = before - len(deduplicated)

    objectid_present = deduplicated["objectid"].notna() & deduplicated[
        "objectid"
    ].astype("string").str.strip().ne("")
    duplicate_objectids = objectid_present & deduplicated.duplicated(
        subset=["objectid"], keep="first"
    )
    objectid_removed = int(duplicate_objectids.sum())
    deduplicated = deduplicated.loc[~duplicate_objectids].copy()
    LOGGER.info(
        "Removed duplicates: full=%d, objectid=%d", full_removed, objectid_removed
    )
    return deduplicated, full_removed, objectid_removed


def create_datetime_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Derive local accident date/time from millisecond timestamps and HH:MM(:SS) values."""
    result = dataframe.copy()
    date_values = pd.to_datetime(
        pd.to_numeric(result["rta_date"], errors="coerce"),
        unit="ms",
        errors="coerce",
        utc=True,
    )
    local_dates = date_values.dt.tz_convert(LOCAL_TIMEZONE)
    result["accident_date"] = local_dates.dt.normalize().dt.tz_localize(None)

    time_values = result["fd1r05p1"].astype("string").str.strip().replace("", pd.NA)
    date_text = result["accident_date"].dt.strftime("%Y-%m-%d")
    result["accident_datetime"] = pd.to_datetime(
        date_text + " " + time_values, errors="coerce"
    )
    return result


def create_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add reproducible calendar features from the cleaned local accident datetime."""
    result = dataframe.copy()
    dates = result["accident_datetime"].dt
    result["hour"] = dates.hour.astype("Int64")
    result["weekday"] = dates.weekday.astype("Int64")
    result["day"] = dates.day.astype("Int64")
    result["month"] = dates.month.astype("Int64")
    result["year"] = dates.year.astype("Int64")
    result["quarter"] = dates.quarter.astype("Int64")
    result["season"] = (
        result["month"]
        .map(
            {
                12: "winter",
                1: "winter",
                2: "winter",
                3: "spring",
                4: "spring",
                5: "spring",
                6: "summer",
                7: "summer",
                8: "summer",
                9: "autumn",
                10: "autumn",
                11: "autumn",
            }
        )
        .astype("string")
    )
    result["is_weekend"] = dates.weekday.ge(5).astype("boolean")
    return result


def create_geodataframe(dataframe: pd.DataFrame) -> gpd.GeoDataFrame:
    """Create WGS84 point geometry and latitude/longitude from EPSG:3857 x/y values."""
    result = dataframe.copy()
    x = pd.to_numeric(result["x"], errors="coerce")
    y = pd.to_numeric(result["y"], errors="coerce")
    valid_mercator = x.between(-20_037_508.35, 20_037_508.35) & y.between(
        -20_048_966.1, 20_048_966.1
    )
    geometry_3857 = gpd.GeoSeries(
        [
            Point(x_value, y_value) if is_valid else None
            for x_value, y_value, is_valid in zip(x, y, valid_mercator, strict=True)
        ],
        index=result.index,
        crs="EPSG:3857",
    )
    geodataframe = gpd.GeoDataFrame(
        result, geometry=geometry_3857, crs="EPSG:3857"
    ).to_crs("EPSG:4326")
    geodataframe["longitude"] = geodataframe.geometry.x
    geodataframe["latitude"] = geodataframe.geometry.y
    return geodataframe


def remove_missing_required(
    dataframe: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, int, int, int]:
    """Drop records lacking an identifier, usable coordinates, or complete accident timestamp."""
    objectid_missing = dataframe["objectid"].isna() | dataframe["objectid"].astype(
        "string"
    ).str.strip().eq("")
    without_objectid = dataframe.loc[~objectid_missing].copy()
    missing_objectid_removed = int(objectid_missing.sum())

    coordinate_missing = (
        without_objectid["x"].isna()
        | without_objectid["y"].isna()
        | without_objectid.geometry.isna()
    )
    without_coordinates = without_objectid.loc[~coordinate_missing].copy()
    missing_coordinates_removed = int(coordinate_missing.sum())

    datetime_missing = without_coordinates["accident_datetime"].isna()
    cleaned = without_coordinates.loc[~datetime_missing].copy()
    missing_datetime_removed = int(datetime_missing.sum())
    LOGGER.info(
        "Removed incomplete records: objectid=%d, coordinates=%d, datetime=%d",
        missing_objectid_removed,
        missing_coordinates_removed,
        missing_datetime_removed,
    )
    return (
        cleaned,
        missing_objectid_removed,
        missing_coordinates_removed,
        missing_datetime_removed,
    )


def write_suspicious_coordinates(
    dataframe: gpd.GeoDataFrame, report_dir: Path
) -> tuple[Path, int]:
    """Write but retain WGS84 points outside the broad Astana-area boundary."""
    longitude, latitude = dataframe["longitude"], dataframe["latitude"]
    in_region = longitude.between(
        ASTANA_REGION_BOUNDS[0], ASTANA_REGION_BOUNDS[2]
    ) & latitude.between(ASTANA_REGION_BOUNDS[1], ASTANA_REGION_BOUNDS[3])
    suspicious = dataframe.loc[~in_region].copy()
    report_path = report_dir / "suspicious_coordinates.csv"
    suspicious.to_csv(report_path, index=False, encoding="utf-8-sig")
    LOGGER.info(
        "Suspicious Astana-area coordinates retained and reported: %d", len(suspicious)
    )
    return report_path, len(suspicious)


def make_report_dir() -> Path:
    """Create a unique timestamped report directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def save_outputs(dataframe: gpd.GeoDataFrame) -> tuple[Path, Path]:
    """Persist clean data in GeoParquet and CSV forms under data/processed."""
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_ROOT / "astana_accidents_clean.parquet"
    csv_path = PROCESSED_ROOT / "astana_accidents_clean.csv"
    dataframe.to_parquet(parquet_path, index=False, engine="pyarrow")
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return parquet_path, csv_path


def write_reports(report_dir: Path, metrics: CleaningMetrics) -> None:
    """Create human- and machine-readable cleaning reports."""
    payload: dict[str, Any] = asdict(metrics) | {
        "generated_at_utc": datetime.now(UTC).isoformat()
    }
    (report_dir / "cleaning_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "Historical accidents data cleaning report",
        f"Input: {metrics.input_path}",
        f"Rows before cleaning: {metrics.rows_before}",
        f"Rows after cleaning: {metrics.rows_after}",
        f"Full duplicates removed: {metrics.full_duplicates_removed}",
        f"Duplicate objectid rows removed: {metrics.objectid_duplicates_removed}",
        f"Rows removed without objectid: {metrics.missing_objectid_removed}",
        f"Rows removed without coordinates: {metrics.missing_coordinates_removed}",
        f"Rows removed without accident_datetime: {metrics.missing_datetime_removed}",
        f"Suspicious coordinates retained: {metrics.suspicious_coordinates}",
        f"Created columns: {', '.join(metrics.created_columns)}",
        f"Clean Parquet: {metrics.output_parquet}",
        f"Clean CSV: {metrics.output_csv}",
        f"Suspicious coordinate report: {metrics.suspicious_coordinates_report}",
    ]
    (report_dir / "cleaning_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Run the complete, non-destructive cleaning pipeline."""
    configure_logging()
    try:
        input_path = resolve_input(parse_args().input)
        source = load_dataset(input_path)
        if source.empty:
            raise ValueError("Input dataset contains no rows.")
        require_columns(source, ("objectid", "rta_date", "fd1r05p1", "x", "y"))
        rows_before = len(source)
        cleaned, full_duplicates, objectid_duplicates = remove_duplicates(source.copy())
        cleaned = create_datetime_columns(cleaned)
        cleaned = create_time_features(cleaned)
        geodataframe = create_geodataframe(cleaned)
        (
            cleaned_geodataframe,
            missing_objectid,
            missing_coordinates,
            missing_datetime,
        ) = remove_missing_required(geodataframe)
        report_dir = make_report_dir()
        suspicious_report, suspicious_count = write_suspicious_coordinates(
            cleaned_geodataframe, report_dir
        )
        parquet_path, csv_path = save_outputs(cleaned_geodataframe)
        metrics = CleaningMetrics(
            input_path=str(input_path),
            rows_before=rows_before,
            rows_after=len(cleaned_geodataframe),
            full_duplicates_removed=full_duplicates,
            objectid_duplicates_removed=objectid_duplicates,
            missing_objectid_removed=missing_objectid,
            missing_coordinates_removed=missing_coordinates,
            missing_datetime_removed=missing_datetime,
            suspicious_coordinates=suspicious_count,
            created_columns=[
                "accident_date",
                "accident_datetime",
                "hour",
                "weekday",
                "day",
                "month",
                "year",
                "quarter",
                "season",
                "is_weekend",
                "latitude",
                "longitude",
                "geometry",
            ],
            output_parquet=str(parquet_path),
            output_csv=str(csv_path),
            suspicious_coordinates_report=str(suspicious_report),
        )
        write_reports(report_dir, metrics)
        print(f"Rows before / after: {metrics.rows_before} / {metrics.rows_after}")
        print(f"Report: {report_dir}")
        print("Next validation command:")
        print(
            "python scripts/validate_historical_accidents.py --input data/processed/astana_accidents_clean.parquet"
        )
        return 0
    except Exception as exc:
        LOGGER.exception("Cleaning failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
