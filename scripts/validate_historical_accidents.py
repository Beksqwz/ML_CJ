"""Read-only quality validation for historical accident datasets.

Run:
    python scripts/validate_historical_accidents.py
    python scripts/validate_historical_accidents.py --input path/to/dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pyproj import Transformer


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports" / "data_validation"
SUPPORTED_SUFFIXES = {".csv", ".parquet"}
ASTANA_LON, ASTANA_LAT = 71.4491, 51.1694
# Deliberately broad: it includes Astana and intercity roads around it.
ASTANA_REGION_BOUNDS = (69.0, 50.0, 74.0, 53.0)  # min_lon, min_lat, max_lon, max_lat
TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$")


@dataclass(frozen=True)
class CheckResult:
    """Machine-readable outcome for one validation check."""

    name: str
    status: str
    message: str
    metrics: dict[str, Any]


def configure_logging() -> None:
    """Configure console logging for a command-line validation run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate historical accident CSV or Parquet data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a CSV or Parquet dataset. If omitted, a dataset is discovered automatically.",
    )
    return parser.parse_args()


def discover_input() -> Path:
    """Locate a likely downloader output without relying on absolute paths.

    A future ``scripts/historical_accidents.py`` can save anywhere under the project;
    discovery first prefers files named like its expected output, then any dataset
    outside generated reports. Parquet is preferred over an equivalent CSV.
    """
    candidates = [
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and REPORTS_ROOT not in path.parents
        and ".venv" not in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            "No CSV or Parquet input was found. Pass the dataset explicitly with --input."
        )

    def ranking(path: Path) -> tuple[int, int, float]:
        name = path.stem.lower()
        historical_priority = int("historical" in name and "accident" in name)
        accident_priority = int("accident" in name or "dtp" in name)
        parquet_priority = int(path.suffix.lower() == ".parquet")
        return (
            historical_priority * 4 + accident_priority * 2 + parquet_priority,
            parquet_priority,
            path.stat().st_mtime,
        )

    selected = max(candidates, key=ranking)
    LOGGER.info("Automatically selected input: %s", selected.relative_to(PROJECT_ROOT))
    return selected


def resolve_input(input_path: Path | None) -> Path:
    """Validate an explicit input path or discover one."""
    path = (input_path if input_path is not None else discover_input()).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Only CSV and Parquet are supported, received: {path.suffix}")
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a dataset without altering it."""
    LOGGER.info("Reading %s", path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


def make_report_dir(input_path: Path) -> Path:
    """Create a timestamped directory for newly generated validation artefacts."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / f"{input_path.stem}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def result(name: str, status: str, message: str, **metrics: Any) -> CheckResult:
    """Build a consistently structured check result."""
    return CheckResult(name=name, status=status, message=message, metrics=metrics)


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Match a candidate column case-insensitively."""
    lookup = {column.lower(): column for column in columns}
    return next(
        (
            lookup[candidate.lower()]
            for candidate in candidates
            if candidate.lower() in lookup
        ),
        None,
    )


def profile_columns(dataframe: pd.DataFrame, report_dir: Path) -> None:
    """Write a compact profile and missing-value table for every source column."""
    rows: list[dict[str, Any]] = []
    total_rows = len(dataframe)
    for column in dataframe.columns:
        series = dataframe[column]
        non_null = int(series.notna().sum())
        examples = [
            str(value) for value in series.dropna().drop_duplicates().head(5).tolist()
        ]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": total_rows,
                "non_null_count": non_null,
                "missing_count": total_rows - non_null,
                "missing_percent": round((total_rows - non_null) * 100 / total_rows, 3)
                if total_rows
                else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": " | ".join(examples),
            }
        )
    profile = pd.DataFrame(rows)
    profile.to_csv(report_dir / "column_profile.csv", index=False, encoding="utf-8-sig")
    profile.loc[profile["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    ).to_csv(report_dir / "missing_values.csv", index=False, encoding="utf-8-sig")


def check_duplicates(dataframe: pd.DataFrame, report_dir: Path) -> list[CheckResult]:
    """Detect complete and identifier-level duplicates and store representative rows."""
    results: list[CheckResult] = []
    examples: list[pd.DataFrame] = []
    full_mask = dataframe.duplicated(keep=False)
    full_count = int(full_mask.sum())
    if full_count:
        duplicate_rows = dataframe.loc[full_mask].copy()
        duplicate_rows.insert(0, "duplicate_check", "full_row")
        examples.append(duplicate_rows.head(100))
        results.append(
            result(
                "full_duplicates",
                "WARNING",
                "Full duplicate rows found.",
                rows=full_count,
            )
        )
    else:
        results.append(
            result("full_duplicates", "PASS", "No full duplicate rows found.", rows=0)
        )

    for identifier in ("objectid", "fd1id", "globalid"):
        column = first_existing(dataframe.columns, [identifier])
        if column is None:
            results.append(
                result(
                    f"duplicates_{identifier}",
                    "SKIPPED",
                    f"Column {identifier} is absent.",
                )
            )
            continue
        non_empty = dataframe[column].notna() & dataframe[column].astype(
            "string"
        ).str.strip().ne("")
        duplicate_mask = dataframe.loc[non_empty, column].duplicated(keep=False)
        duplicate_index = duplicate_mask.index[duplicate_mask]
        count = len(duplicate_index)
        if count:
            duplicate_rows = dataframe.loc[duplicate_index].copy()
            duplicate_rows.insert(0, "duplicate_check", f"duplicate_{column}")
            examples.append(duplicate_rows.head(100))
            results.append(
                result(
                    f"duplicates_{identifier}",
                    "WARNING",
                    f"Duplicate {column} values found.",
                    rows=count,
                )
            )
        else:
            results.append(
                result(
                    f"duplicates_{identifier}",
                    "PASS",
                    f"No duplicate non-empty {column} values.",
                    rows=0,
                )
            )

    output = (
        pd.concat(examples, ignore_index=True)
        if examples
        else dataframe.head(0).assign(duplicate_check=pd.Series(dtype="string"))
    )
    output.to_csv(
        report_dir / "duplicate_examples.csv", index=False, encoding="utf-8-sig"
    )
    return results


def compare_date_part(
    dataframe: pd.DataFrame,
    source_column: str,
    datetime_values: pd.Series,
    accessor: str,
    check_name: str,
    source_label: str,
    severity_on_mismatch: str,
) -> CheckResult:
    """Compare a numeric date-part field with a parsed datetime series."""
    source = pd.to_numeric(dataframe[source_column], errors="coerce")
    datetime_part = getattr(datetime_values.dt, accessor)
    comparable = source.notna() & datetime_part.notna()
    mismatches = int((source.loc[comparable] != datetime_part.loc[comparable]).sum())
    status = severity_on_mismatch if mismatches else "PASS"
    return result(
        check_name,
        status,
        f"Compared {source_column} with {accessor} extracted from {source_label}.",
        compared=int(comparable.sum()),
        mismatches=mismatches,
    )


def check_dates(
    dataframe: pd.DataFrame, report_dir: Path
) -> tuple[list[CheckResult], pd.Series | None]:
    """Validate timestamps and use cleaned datetime fields when they are available."""
    checks: list[CheckResult] = []
    parsed_rta: pd.Series | None = None
    for requested_name in ("rta_date", "load_date"):
        column = first_existing(dataframe.columns, [requested_name])
        if column is None:
            checks.append(
                result(
                    f"{requested_name}_timestamp",
                    "SKIPPED",
                    f"Column {requested_name} is absent.",
                )
            )
            continue
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        parsed = pd.to_datetime(
            numeric, unit="ms", errors="coerce", utc=True
        ).dt.tz_convert("Asia/Almaty")
        invalid = int(numeric.notna().sum() - parsed.notna().sum())
        date_min = parsed.min()
        date_max = parsed.max()
        status = "WARNING" if invalid else "PASS"
        checks.append(
            result(
                f"{requested_name}_timestamp",
                status,
                f"Parsed {requested_name} as Unix milliseconds.",
                non_null=int(parsed.notna().sum()),
                invalid=invalid,
                min=str(date_min) if pd.notna(date_min) else None,
                max=str(date_max) if pd.notna(date_max) else None,
            )
        )
        if requested_name == "rta_date":
            parsed_rta = parsed

    if parsed_rta is None:
        pd.DataFrame(columns=["year", "accident_count"]).to_csv(
            report_dir / "year_distribution.csv", index=False
        )
        pd.DataFrame(columns=["month", "accident_count"]).to_csv(
            report_dir / "month_distribution.csv", index=False
        )
        return checks, None

    valid_dates = parsed_rta.dropna()
    valid_dates.dt.year.value_counts().sort_index().rename_axis("year").reset_index(
        name="accident_count"
    ).to_csv(report_dir / "year_distribution.csv", index=False, encoding="utf-8-sig")
    valid_dates.dt.month.value_counts().sort_index().rename_axis("month").reset_index(
        name="accident_count"
    ).to_csv(report_dir / "month_distribution.csv", index=False, encoding="utf-8-sig")
    accident_datetime_column = first_existing(dataframe.columns, ["accident_datetime"])
    cleaned_year_column = first_existing(dataframe.columns, ["year"])
    cleaned_month_column = first_existing(dataframe.columns, ["month"])
    if accident_datetime_column and cleaned_year_column and cleaned_month_column:
        accident_datetime = pd.to_datetime(
            dataframe[accident_datetime_column], errors="coerce"
        )
        checks.append(
            compare_date_part(
                dataframe,
                cleaned_year_column,
                accident_datetime,
                "year",
                "year_matches_accident_datetime",
                accident_datetime_column,
                "WARNING",
            )
        )
        checks.append(
            compare_date_part(
                dataframe,
                cleaned_month_column,
                accident_datetime,
                "month",
                "month_matches_accident_datetime",
                accident_datetime_column,
                "WARNING",
            )
        )
    else:
        for expected, accessor in (("yr", "year"), ("period", "month")):
            column = first_existing(dataframe.columns, [expected])
            if column is None:
                checks.append(
                    result(
                        f"{expected}_matches_rta_date",
                        "SKIPPED",
                        f"Column {expected} is absent.",
                    )
                )
                continue
            checks.append(
                compare_date_part(
                    dataframe,
                    column,
                    parsed_rta,
                    accessor,
                    f"{expected}_matches_rta_date",
                    "rta_date",
                    "WARNING",
                )
            )

    # Preserve raw source-date discrepancies as context, never as cleaned-data failures.
    for expected, accessor in (("yr", "year"), ("period", "month")):
        column = first_existing(dataframe.columns, [expected])
        if column is None:
            continue
        checks.append(
            compare_date_part(
                dataframe,
                column,
                parsed_rta,
                accessor,
                f"source_{expected}_matches_rta_date",
                "rta_date",
                "INFO",
            )
        )
    return checks, parsed_rta


def check_time(dataframe: pd.DataFrame, report_dir: Path) -> list[CheckResult]:
    """Validate HH:MM(/:SS) time strings and write hourly distribution when possible."""
    column = first_existing(dataframe.columns, ["fd1r05p1"])
    if column is None:
        return [result("time_format", "SKIPPED", "Column fd1r05p1 is absent.")]
    values = dataframe[column].dropna().astype("string").str.strip()
    values = values[values.ne("")]
    valid_mask = values.str.match(TIME_PATTERN, na=False)
    invalid = int((~valid_mask).sum())
    if valid_mask.any():
        hours = values.loc[valid_mask].str.slice(0, 2).astype(int)
        hours.value_counts().sort_index().rename_axis("hour").reset_index(
            name="accident_count"
        ).to_csv(
            report_dir / "hour_distribution.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=["hour", "accident_count"]).to_csv(
            report_dir / "hour_distribution.csv", index=False
        )
    return [
        result(
            "time_format",
            "WARNING" if invalid else "PASS",
            "Validated fd1r05p1 as 24-hour time.",
            non_empty=int(len(values)),
            invalid=invalid,
        )
    ]


def check_coordinates(
    dataframe: pd.DataFrame, report_dir: Path
) -> tuple[list[CheckResult], int]:
    """Validate coordinates and check their temporary WGS84 location near Astana."""
    x_column = first_existing(dataframe.columns, ["x"])
    y_column = first_existing(dataframe.columns, ["y"])
    lon_column = first_existing(dataframe.columns, ["longitude", "lon"])
    lat_column = first_existing(dataframe.columns, ["latitude", "lat"])
    if x_column and y_column:
        x = pd.to_numeric(dataframe[x_column], errors="coerce")
        y = pd.to_numeric(dataframe[y_column], errors="coerce")
        valid = (
            x.notna()
            & y.notna()
            & x.between(-20_037_508.35, 20_037_508.35)
            & y.between(-20_048_966.1, 20_048_966.1)
        )
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        longitude, latitude = transformer.transform(
            x.where(valid).to_numpy(), y.where(valid).to_numpy()
        )
        longitude, latitude = (
            pd.Series(longitude, index=dataframe.index),
            pd.Series(latitude, index=dataframe.index),
        )
        source_crs = "EPSG:3857"
    elif lon_column and lat_column:
        longitude = pd.to_numeric(dataframe[lon_column], errors="coerce")
        latitude = pd.to_numeric(dataframe[lat_column], errors="coerce")
        valid = (
            longitude.notna()
            & latitude.notna()
            & longitude.between(-180, 180)
            & latitude.between(-90, 90)
        )
        source_crs = "EPSG:4326"
    else:
        pd.DataFrame(columns=["reason"]).to_csv(
            report_dir / "suspicious_coordinates.csv", index=False
        )
        return [
            result(
                "coordinates",
                "SKIPPED",
                "No x/y or longitude/latitude coordinate pair found.",
            )
        ], 0

    region_ok = longitude.between(
        ASTANA_REGION_BOUNDS[0], ASTANA_REGION_BOUNDS[2]
    ) & latitude.between(ASTANA_REGION_BOUNDS[1], ASTANA_REGION_BOUNDS[3])
    suspicious = (~valid) | (valid & ~region_ok)
    suspicious_rows = dataframe.loc[suspicious].copy()
    suspicious_rows.insert(0, "reason", "outside_astana_region_or_invalid_coordinate")
    suspicious_rows.insert(1, "longitude_wgs84", longitude.loc[suspicious].round(6))
    suspicious_rows.insert(2, "latitude_wgs84", latitude.loc[suspicious].round(6))
    suspicious_rows.to_csv(
        report_dir / "suspicious_coordinates.csv", index=False, encoding="utf-8-sig"
    )
    count = int(suspicious.sum())
    status = "ERROR" if int(valid.sum()) == 0 else ("WARNING" if count else "PASS")
    return [
        result(
            "coordinates",
            status,
            f"Validated {source_crs} coordinates and checked temporary EPSG:4326 positions near Astana.",
            valid=int(valid.sum()),
            suspicious=count,
            total=int(len(dataframe)),
        )
    ], count


def check_numeric_values(dataframe: pd.DataFrame) -> list[CheckResult]:
    """Find NaN/infinite values and implausible negative accident count fields."""
    numeric_columns = dataframe.select_dtypes(include="number").columns
    infinite_count = 0
    negative_columns: dict[str, int] = {}
    for column in numeric_columns:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        infinite_count += int((~numeric.isna() & ~numeric.map(math.isfinite)).sum())
        lowered = column.lower()
        if any(
            token in lowered
            for token in (
                "dead",
                "fatal",
                "killed",
                "injur",
                "victim",
                "погиб",
                "ранен",
            )
        ):
            negatives = int((numeric < 0).sum())
            if negatives:
                negative_columns[column] = negatives
    status = "WARNING" if infinite_count or negative_columns else "PASS"
    return [
        result(
            "numeric_values",
            status,
            "Checked numeric values for infinities and negative casualty counts.",
            numeric_columns=int(len(numeric_columns)),
            infinite_values=infinite_count,
            negative_counts=negative_columns,
        )
    ]


def categorical_summary(dataframe: pd.DataFrame) -> list[CheckResult]:
    """Summarise frequent values for important semantic and coded category columns."""
    preferred = [
        "type_dtp",
        "vehicle_category",
        "area_code",
        "fd1r07p1",
        "fd1r17",
        "fd1r06p1",
        "fd1r08p1",
    ]
    available = [column for column in preferred if column in dataframe.columns]
    if not available:
        return [
            result(
                "categorical_values",
                "SKIPPED",
                "No configured categorical fields are present.",
            )
        ]
    frequent = {
        column: dataframe[column].value_counts(dropna=False).head(10).to_dict()
        for column in available
    }
    return [
        result(
            "categorical_values",
            "PASS",
            "Computed ten most frequent values for important categories.",
            frequent_values=frequent,
        )
    ]


def check_ml_leakage(dataframe: pd.DataFrame) -> list[CheckResult]:
    """Flag fields likely known only after an accident and unsafe as prediction features."""
    leakage_tokens = (
        "type_dtp",
        "death",
        "dead",
        "fatal",
        "injur",
        "victim",
        "offender",
        "culprit",
        "violation",
        "guilt",
        "погиб",
        "ранен",
        "винов",
        "наруш",
    )
    columns = [
        column
        for column in dataframe.columns
        if any(token in column.lower() for token in leakage_tokens)
    ]
    if not columns:
        return [
            result(
                "ml_leakage",
                "SKIPPED",
                "No semantically identifiable post-accident fields were found by name.",
            )
        ]
    return [
        result(
            "ml_leakage",
            "WARNING",
            "Potential target leakage fields detected; exclude them unless they are available before prediction time.",
            columns=columns,
        )
    ]


def write_text_report(
    report_dir: Path,
    input_path: Path,
    dataframe: pd.DataFrame,
    checks: list[CheckResult],
    final_status: str,
) -> None:
    """Write a readable report including all statuses, metrics, and category frequencies."""
    lines = [
        "Historical accidents data validation report",
        f"Input: {input_path}",
        f"Generated (UTC): {datetime.now(UTC).isoformat()}",
        f"Rows: {len(dataframe)}",
        f"Columns: {len(dataframe.columns)}",
        f"Final status: {final_status}",
        "",
        "Checks:",
    ]
    for check in checks:
        lines.append(f"[{check.status}] {check.name}: {check.message}")
        if check.metrics:
            lines.append(
                json.dumps(check.metrics, ensure_ascii=False, default=str, indent=2)
            )
    (report_dir / "validation_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def final_status(checks: list[CheckResult]) -> str:
    """Map check severities to the requested readiness status."""
    statuses = {check.status for check in checks}
    if "ERROR" in statuses:
        return "NOT READY"
    if "WARNING" in statuses:
        return "READY WITH WARNINGS"
    return "READY"


def main() -> int:
    """Run all non-mutating checks and persist a complete validation report."""
    configure_logging()
    try:
        input_path = resolve_input(parse_args().input)
        report_dir = make_report_dir(input_path)
        dataframe = load_dataset(input_path)
        if dataframe.empty:
            raise ValueError("Input dataset contains no rows.")

        profile_columns(dataframe, report_dir)
        checks: list[CheckResult] = [
            result(
                "dataset_structure",
                "PASS",
                "Dataset loaded successfully.",
                rows=int(len(dataframe)),
                columns=int(len(dataframe.columns)),
                file_size_bytes=input_path.stat().st_size,
                dtypes={
                    column: str(dtype) for column, dtype in dataframe.dtypes.items()
                },
            )
        ]
        checks.extend(check_duplicates(dataframe, report_dir))
        date_checks, parsed_rta = check_dates(dataframe, report_dir)
        checks.extend(date_checks)
        checks.extend(check_time(dataframe, report_dir))
        coordinate_checks, suspicious_coordinates = check_coordinates(
            dataframe, report_dir
        )
        checks.extend(coordinate_checks)
        checks.extend(check_numeric_values(dataframe))
        checks.extend(categorical_summary(dataframe))
        checks.extend(check_ml_leakage(dataframe))

        outcome = final_status(checks)
        payload = {
            "input_path": str(input_path),
            "report_directory": str(report_dir),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "final_status": outcome,
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "date_range": {
                "min": str(parsed_rta.min())
                if parsed_rta is not None and pd.notna(parsed_rta.min())
                else None,
                "max": str(parsed_rta.max())
                if parsed_rta is not None and pd.notna(parsed_rta.max())
                else None,
            },
            "checks": [asdict(check) for check in checks],
        }
        (report_dir / "validation_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
        write_text_report(report_dir, input_path, dataframe, checks, outcome)
        dataframe.head(100).to_csv(
            report_dir / "sample_rows.csv", index=False, encoding="utf-8-sig"
        )

        warning_count = sum(check.status == "WARNING" for check in checks)
        error_count = sum(check.status == "ERROR" for check in checks)
        duplicate_count = sum(
            int(check.metrics.get("rows", 0))
            for check in checks
            if check.name.startswith("duplicates_") or check.name == "full_duplicates"
        )
        print(f"Rows / columns: {len(dataframe)} / {len(dataframe.columns)}")
        print(
            f"Date range: {payload['date_range']['min']} — {payload['date_range']['max']}"
        )
        print(f"Duplicate rows (check totals): {duplicate_count}")
        print(f"Suspicious coordinates: {suspicious_coordinates}")
        print(f"Errors / warnings: {error_count} / {warning_count}")
        print(f"Final status: {outcome}")
        print(f"Report: {report_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("Validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
