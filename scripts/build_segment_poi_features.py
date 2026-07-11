"""Build static OpenStreetMap POI features for road segments.

The segment coordinates come from the road geometry, not accident observations,
so these features are safe to use at any forecast hour.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READY = PROJECT_ROOT / "data" / "processed" / "accidents_with_roads_ml_ready.parquet"
DEFAULT_ROADS = PROJECT_ROOT / "astana_edges.csv"
DEFAULT_POIS = PROJECT_ROOT / "data" / "external" / "pois_astana_osm.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "road_segment_poi_features.parquet"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "stage6" / "poi_features"
RADII_METERS = (100, 250, 500)
EARTH_RADIUS_M = 6_371_008.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create static POI features for ML road segments.")
    parser.add_argument("--ready", type=Path, default=DEFAULT_READY)
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROADS)
    parser.add_argument("--pois", type=Path, default=DEFAULT_POIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def segment_id(row: dict[str, str]) -> str:
    return f"{row['u'].strip()}_{row['v'].strip()}_{row['key'].strip()}"


def parse_linestring_midpoint(value: str) -> tuple[float, float] | None:
    text = (value or "").strip()
    if not text.upper().startswith("LINESTRING"):
        return None
    try:
        body = text[text.index("(") + 1 : text.rindex(")")]
        points = [tuple(map(float, part.strip().split()[:2])) for part in body.split(",")]
    except (ValueError, IndexError):
        return None
    if len(points) < 2:
        return None
    lengths = [math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]) for i in range(len(points) - 1)]
    half = sum(lengths) / 2
    covered = 0.0
    for index, length in enumerate(lengths):
        if covered + length >= half and length:
            fraction = (half - covered) / length
            lon = points[index][0] + fraction * (points[index + 1][0] - points[index][0])
            lat = points[index][1] + fraction * (points[index + 1][1] - points[index][1])
            return lon, lat
        covered += length
    return points[-1]


def load_segment_centres(roads_path: Path, wanted_segments: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with roads_path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            identifier = segment_id(row)
            if identifier not in wanted_segments:
                continue
            centre = parse_linestring_midpoint(row.get("geometry", ""))
            if centre is not None:
                rows.append({"road_segment_id": identifier, "segment_longitude": centre[0], "segment_latitude": centre[1]})
    return pd.DataFrame(rows).drop_duplicates("road_segment_id")


def haversine_m(lon: float, lat: float, poi_lon: np.ndarray, poi_lat: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(poi_lat)
    lon2 = np.radians(poi_lon)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def build_features(centres: pd.DataFrame, pois: pd.DataFrame) -> pd.DataFrame:
    categories = sorted(pois["poi_category"].astype(str).unique())
    poi_lon = pois["longitude"].to_numpy(dtype=float)
    poi_lat = pois["latitude"].to_numpy(dtype=float)
    poi_categories = pois["poi_category"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for centre in centres.itertuples(index=False):
        distances = haversine_m(centre.segment_longitude, centre.segment_latitude, poi_lon, poi_lat)
        feature_row: dict[str, object] = {
            "road_segment_id": centre.road_segment_id,
            "segment_longitude": centre.segment_longitude,
            "segment_latitude": centre.segment_latitude,
        }
        for radius in RADII_METERS:
            within = distances <= radius
            feature_row[f"poi_total_{radius}m"] = int(within.sum())
            for category in categories:
                feature_row[f"poi_{category}_{radius}m"] = int((within & (poi_categories == category)).sum())
        rows.append(feature_row)
    return pd.DataFrame(rows).sort_values("road_segment_id").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    ready = pd.read_parquet(args.ready)
    pois = pd.read_parquet(args.pois)
    wanted = set(ready["road_segment_id"].astype(str).unique())
    centres = load_segment_centres(args.roads, wanted)
    missing_geometry = sorted(wanted - set(centres["road_segment_id"]))
    if missing_geometry:
        raise ValueError(f"Road geometry is missing for {len(missing_geometry)} ML segments.")
    features = build_features(centres, pois)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False, engine="pyarrow")
    report_dir = REPORTS_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "segments": int(len(features)),
        "pois": int(len(pois)),
        "radii_m": list(RADII_METERS),
        "output": str(args.output.resolve()),
        "missing_values": {column: int(features[column].isna().sum()) for column in features.columns},
        "feature_columns": list(features.columns),
    }
    (report_dir / "poi_features_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Segments: {len(features)}")
    print(f"Output: {args.output.resolve()}")
    print(f"Report: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
