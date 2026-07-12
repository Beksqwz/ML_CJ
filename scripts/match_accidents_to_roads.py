"""Match cleaned Astana accidents to nearest road segments.

This script intentionally works with the standard library so it can run even
when geopandas/pandas are unavailable. If pandas + pyarrow are installed, it
also writes a Parquet copy of the matched CSV.

Run:
    py scripts/match_accidents_to_roads.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "road_matching"

ASTANA_LAT0 = 51.1694
ASTANA_LON0 = 71.4491
EARTH_RADIUS_M = 6_371_008.8
GRID_CELL_M = 500.0
ROAD_ATTRIBUTE_COLUMNS = (
    "u",
    "v",
    "key",
    "osmid",
    "highway",
    "lanes",
    "name",
    "oneway",
    "reversed",
    "length",
    "maxspeed",
    "ref",
    "bridge",
    "tunnel",
    "width",
    "junction",
    "access",
)
LINESTRING_RE = re.compile(r"^LINESTRING\s*\((.*)\)$", re.IGNORECASE)


@dataclass(frozen=True)
class RoadSegment:
    """Projected road segment with source attributes."""

    segment_id: str
    attributes: dict[str, str]
    points_m: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class MatchingSummary:
    """Machine-readable matching output."""

    accidents_input: str
    roads_input: str
    rows: int
    road_segments: int
    matched_rows: int
    unmatched_rows: int
    distance_min_m: float | None
    distance_p50_m: float | None
    distance_p90_m: float | None
    distance_p95_m: float | None
    distance_max_m: float | None
    rows_over_50m: int
    rows_over_100m: int
    rows_over_200m: int
    output_csv: str
    output_parquet: str | None
    far_matches_report: str


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
        description="Attach each cleaned accident to the nearest Astana road segment."
    )
    parser.add_argument(
        "--accidents",
        type=Path,
        default=PROCESSED_ROOT / "astana_accidents_clean.csv",
        help="Cleaned accidents CSV. Defaults to data/processed/astana_accidents_clean.csv.",
    )
    parser.add_argument(
        "--roads",
        type=Path,
        default=PROJECT_ROOT / "data" / "roads" / "astana_edges.csv",
        help="Road edges CSV with WKT LINESTRING geometries. Defaults to astana_edges.csv.",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=PROCESSED_ROOT / "accidents_with_roads.csv"
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=PROCESSED_ROOT / "accidents_with_roads.parquet",
    )
    return parser.parse_args()


def project_lonlat(lon: float, lat: float) -> tuple[float, float]:
    """Project lon/lat near Astana to local meters with low distortion for nearest-road matching."""
    x = (
        EARTH_RADIUS_M
        * math.radians(lon - ASTANA_LON0)
        * math.cos(math.radians(ASTANA_LAT0))
    )
    y = EARTH_RADIUS_M * math.radians(lat - ASTANA_LAT0)
    return x, y


def parse_linestring_wkt(value: str) -> tuple[tuple[float, float], ...] | None:
    match = LINESTRING_RE.match((value or "").strip())
    if not match:
        return None
    points: list[tuple[float, float]] = []
    for pair in match.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            return None
        try:
            lon, lat = float(parts[0]), float(parts[1])
        except ValueError:
            return None
        points.append(project_lonlat(lon, lat))
    return tuple(points) if len(points) >= 2 else None


def bbox_for(
    points: Iterable[tuple[float, float]],
) -> tuple[float, float, float, float]:
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def road_id(row: dict[str, str], fallback_index: int) -> str:
    u, v, key = (
        row.get("u", "").strip(),
        row.get("v", "").strip(),
        row.get("key", "").strip(),
    )
    if u and v and key:
        return f"{u}_{v}_{key}"
    return f"road_{fallback_index}"


def load_roads(path: Path) -> list[RoadSegment]:
    if not path.is_file():
        raise FileNotFoundError(f"Road edges file does not exist: {path}")

    roads: list[RoadSegment] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "geometry" not in reader.fieldnames:
            raise ValueError(
                "Road CSV must contain a geometry column with LINESTRING WKT."
            )
        for index, row in enumerate(reader):
            points_m = parse_linestring_wkt(row.get("geometry", ""))
            if points_m is None:
                continue
            attributes = {
                column: row.get(column, "") for column in ROAD_ATTRIBUTE_COLUMNS
            }
            roads.append(
                RoadSegment(
                    segment_id=road_id(row, index),
                    attributes=attributes,
                    points_m=points_m,
                    bbox=bbox_for(points_m),
                )
            )
    if not roads:
        raise ValueError(f"No valid road LINESTRING geometries were read from {path}")
    LOGGER.info("Loaded %d road segments", len(roads))
    return roads


def cell_range(min_value: float, max_value: float) -> range:
    return range(
        math.floor(min_value / GRID_CELL_M), math.floor(max_value / GRID_CELL_M) + 1
    )


def build_spatial_index(roads: list[RoadSegment]) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    for index, road in enumerate(roads):
        min_x, min_y, max_x, max_y = road.bbox
        for cell_x in cell_range(min_x, max_x):
            for cell_y in cell_range(min_y, max_y):
                grid.setdefault((cell_x, cell_y), []).append(index)
    LOGGER.info("Built spatial index with %d occupied cells", len(grid))
    return grid


def point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def point_polyline_distance(
    px: float, py: float, points: tuple[tuple[float, float], ...]
) -> float:
    best = math.inf
    for first, second in zip(points, points[1:]):
        distance = point_segment_distance(
            px, py, first[0], first[1], second[0], second[1]
        )
        if distance < best:
            best = distance
    return best


def candidate_indices(
    px: float, py: float, grid: dict[tuple[int, int], list[int]], max_ring: int = 10
) -> set[int]:
    origin_x, origin_y = math.floor(px / GRID_CELL_M), math.floor(py / GRID_CELL_M)
    candidates: set[int] = set()
    for ring in range(max_ring + 1):
        for cell_x in range(origin_x - ring, origin_x + ring + 1):
            for cell_y in range(origin_y - ring, origin_y + ring + 1):
                if max(abs(cell_x - origin_x), abs(cell_y - origin_y)) != ring:
                    continue
                candidates.update(grid.get((cell_x, cell_y), ()))
    return candidates


def nearest_road(
    lon: float,
    lat: float,
    roads: list[RoadSegment],
    grid: dict[tuple[int, int], list[int]],
) -> tuple[RoadSegment | None, float | None]:
    px, py = project_lonlat(lon, lat)
    candidates = candidate_indices(px, py, grid)
    if not candidates:
        candidates = set(range(len(roads)))

    best_road: RoadSegment | None = None
    best_distance = math.inf
    for road_index in candidates:
        road = roads[road_index]
        distance = point_polyline_distance(px, py, road.points_m)
        if distance < best_distance:
            best_road, best_distance = road, distance
    return best_road, best_distance if math.isfinite(best_distance) else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def try_write_parquet(csv_path: Path, parquet_path: Path) -> Path | None:
    try:
        import pandas as pd  # type: ignore
        import pyarrow  # noqa: F401  # type: ignore
    except Exception as exc:
        LOGGER.warning(
            "Skipping Parquet output because pandas/pyarrow are unavailable: %s", exc
        )
        return None

    dataframe = pd.read_csv(csv_path, low_memory=False)
    dataframe.to_parquet(parquet_path, index=False, engine="pyarrow")
    return parquet_path


def make_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def write_text_report(report_dir: Path, summary: MatchingSummary) -> None:
    lines = [
        "Road matching report",
        f"Accidents input: {summary.accidents_input}",
        f"Roads input: {summary.roads_input}",
        f"Rows: {summary.rows}",
        f"Road segments: {summary.road_segments}",
        f"Matched rows: {summary.matched_rows}",
        f"Unmatched rows: {summary.unmatched_rows}",
        f"Distance p50/p90/p95/max m: {summary.distance_p50_m} / {summary.distance_p90_m} / {summary.distance_p95_m} / {summary.distance_max_m}",
        f"Rows over 50/100/200m: {summary.rows_over_50m} / {summary.rows_over_100m} / {summary.rows_over_200m}",
        f"Output CSV: {summary.output_csv}",
        f"Output Parquet: {summary.output_parquet}",
        f"Far matches report: {summary.far_matches_report}",
    ]
    (report_dir / "road_matching_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def match_accidents(args: argparse.Namespace) -> MatchingSummary:
    accidents_path = args.accidents.resolve()
    roads_path = args.roads.resolve()
    output_csv = args.output_csv.resolve()
    output_parquet = args.output_parquet.resolve()
    if not accidents_path.is_file():
        raise FileNotFoundError(f"Clean accidents CSV does not exist: {accidents_path}")

    roads = load_roads(roads_path)
    grid = build_spatial_index(roads)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_dir = make_report_dir()
    far_matches_path = report_dir / "far_matches_over_100m.csv"

    distances: list[float] = []
    rows = 0
    matched = 0
    far_rows: list[dict[str, str]] = []

    with (
        accidents_path.open("r", encoding="utf-8-sig", newline="") as input_file,
        output_csv.open("w", encoding="utf-8-sig", newline="") as output_file,
    ):
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError("Clean accidents CSV has no header.")
        required = {"longitude", "latitude"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Clean accidents CSV is missing required columns: {', '.join(missing)}"
            )

        road_fields = [
            "road_segment_id",
            "distance_to_road_m",
            *[f"road_{column}" for column in ROAD_ATTRIBUTE_COLUMNS],
        ]
        writer = csv.DictWriter(
            output_file, fieldnames=[*reader.fieldnames, *road_fields]
        )
        writer.writeheader()

        for row in reader:
            rows += 1
            try:
                lon, lat = float(row["longitude"]), float(row["latitude"])
            except (TypeError, ValueError):
                road, distance = None, None
            else:
                road, distance = nearest_road(lon, lat, roads, grid)

            if road is not None and distance is not None:
                matched += 1
                distances.append(distance)
                row["road_segment_id"] = road.segment_id
                row["distance_to_road_m"] = f"{distance:.3f}"
                for column in ROAD_ATTRIBUTE_COLUMNS:
                    row[f"road_{column}"] = road.attributes.get(column, "")
                if distance > 100:
                    far_rows.append(row.copy())
            else:
                row["road_segment_id"] = ""
                row["distance_to_road_m"] = ""
                for column in ROAD_ATTRIBUTE_COLUMNS:
                    row[f"road_{column}"] = ""
            writer.writerow(row)

    with far_matches_path.open("w", encoding="utf-8-sig", newline="") as far_file:
        fieldnames = [
            *reader.fieldnames,
            "road_segment_id",
            "distance_to_road_m",
            *[f"road_{column}" for column in ROAD_ATTRIBUTE_COLUMNS],
        ]
        writer = csv.DictWriter(far_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(far_rows)

    parquet_written = try_write_parquet(output_csv, output_parquet)
    summary = MatchingSummary(
        accidents_input=str(accidents_path),
        roads_input=str(roads_path),
        rows=rows,
        road_segments=len(roads),
        matched_rows=matched,
        unmatched_rows=rows - matched,
        distance_min_m=round(min(distances), 3) if distances else None,
        distance_p50_m=round(percentile(distances, 0.50), 3) if distances else None,
        distance_p90_m=round(percentile(distances, 0.90), 3) if distances else None,
        distance_p95_m=round(percentile(distances, 0.95), 3) if distances else None,
        distance_max_m=round(max(distances), 3) if distances else None,
        rows_over_50m=sum(distance > 50 for distance in distances),
        rows_over_100m=sum(distance > 100 for distance in distances),
        rows_over_200m=sum(distance > 200 for distance in distances),
        output_csv=str(output_csv),
        output_parquet=str(parquet_written) if parquet_written else None,
        far_matches_report=str(far_matches_path),
    )
    (report_dir / "road_matching_summary.json").write_text(
        json.dumps(
            asdict(summary)
            | {"generated_at_utc": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_text_report(report_dir, summary)
    return summary


def main() -> int:
    configure_logging()
    try:
        summary = match_accidents(parse_args())
        print(f"Rows matched: {summary.matched_rows}/{summary.rows}")
        print(
            f"Distance p50/p90/p95/max m: {summary.distance_p50_m}/{summary.distance_p90_m}/{summary.distance_p95_m}/{summary.distance_max_m}"
        )
        print(f"Rows over 100m: {summary.rows_over_100m}")
        print(f"Output CSV: {summary.output_csv}")
        print(f"Output Parquet: {summary.output_parquet}")
        return 0
    except Exception as exc:
        LOGGER.exception("Road matching failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
