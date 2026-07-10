"""Fetch Astana infrastructure POIs from OpenStreetMap Overpass API.

Run:
    py scripts/fetch_osm_pois.py
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
import requests


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "external_data"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def configure_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OSM POIs for Astana.")
    parser.add_argument("--geojson", type=Path, default=PROJECT_ROOT / "astana.geojson")
    parser.add_argument("--output-parquet", type=Path, default=EXTERNAL_ROOT / "pois_astana_osm.parquet")
    parser.add_argument("--output-csv", type=Path, default=EXTERNAL_ROOT / "pois_astana_osm.csv")
    return parser.parse_args()


def read_bbox(path: Path) -> tuple[float, float, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    properties = payload["features"][0]["properties"] if payload.get("features") else payload.get("properties", {})
    west = float(properties["bbox_west"])
    south = float(properties["bbox_south"])
    east = float(properties["bbox_east"])
    north = float(properties["bbox_north"])
    return south, west, north, east


def build_query(bbox: tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    bbox_text = f"{south},{west},{north},{east}"
    return f"""
[out:json][timeout:180];
(
  node["amenity"~"school|kindergarten|college|university|hospital|clinic|police|fire_station"]({bbox_text});
  way["amenity"~"school|kindergarten|college|university|hospital|clinic|police|fire_station"]({bbox_text});
  relation["amenity"~"school|kindergarten|college|university|hospital|clinic|police|fire_station"]({bbox_text});
  node["highway"~"bus_stop|traffic_signals|crossing"]({bbox_text});
  way["highway"="crossing"]({bbox_text});
  node["public_transport"~"platform|stop_position"]({bbox_text});
  way["public_transport"="platform"]({bbox_text});
);
out center tags;
"""


def classify(tags: dict[str, Any]) -> str:
    amenity = str(tags.get("amenity", "")).lower()
    highway = str(tags.get("highway", "")).lower()
    public_transport = str(tags.get("public_transport", "")).lower()
    if amenity in {"school", "kindergarten", "college", "university"}:
        return "education"
    if amenity in {"hospital", "clinic"}:
        return "healthcare"
    if amenity in {"police", "fire_station"}:
        return "emergency"
    if highway == "bus_stop" or public_transport in {"platform", "stop_position"}:
        return "transit_stop"
    if highway == "traffic_signals":
        return "traffic_signal"
    if highway == "crossing":
        return "crossing"
    return "other"


def fetch_pois(bbox: tuple[float, float, float, float]) -> tuple[pd.DataFrame, str]:
    query = build_query(bbox)
    headers = {"User-Agent": "RoadRisk-AI-GovTech-Camp/1.0"}
    response: requests.Response | None = None
    source_url = ""
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            response = requests.get(url, params={"data": query}, headers=headers, timeout=240)
            response.raise_for_status()
            source_url = url
            break
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Overpass endpoint failed: %s (%s)", url, exc)
    if response is None or not source_url:
        raise RuntimeError(f"All Overpass endpoints failed: {last_error}")
    elements = response.json().get("elements", [])
    rows: list[dict[str, Any]] = []
    for element in elements:
        tags = element.get("tags", {})
        center = element.get("center", {})
        lat = element.get("lat", center.get("lat"))
        lon = element.get("lon", center.get("lon"))
        if lat is None or lon is None:
            continue
        rows.append(
            {
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "poi_category": classify(tags),
                "amenity": tags.get("amenity", ""),
                "highway": tags.get("highway", ""),
                "public_transport": tags.get("public_transport", ""),
                "name": tags.get("name", ""),
                "latitude": float(lat),
                "longitude": float(lon),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["osm_type", "osm_id"]), source_url


def make_report_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_ROOT / "pois" / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def write_report(report_dir: Path, summary: dict[str, Any]) -> None:
    (report_dir / "pois_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "OSM POI fetch report",
        f"Rows: {summary['rows']}",
        f"Category counts: {summary['category_counts']}",
        f"Output Parquet: {summary['output_parquet']}",
    ]
    (report_dir / "pois_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_logging()
    try:
        args = parse_args()
        bbox = read_bbox(args.geojson)
        pois, source_url = fetch_pois(bbox)
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        pois.to_parquet(args.output_parquet, index=False, engine="pyarrow")
        pois.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        report_dir = make_report_dir()
        summary = {
            "source": "OpenStreetMap via Overpass API",
            "source_url": source_url,
            "bbox_south_west_north_east": bbox,
            "rows": int(len(pois)),
            "category_counts": {str(key): int(value) for key, value in pois["poi_category"].value_counts().to_dict().items()},
            "output_parquet": str(args.output_parquet.resolve()),
            "output_csv": str(args.output_csv.resolve()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_report(report_dir, summary)
        print(f"Rows: {summary['rows']}")
        print(f"Category counts: {summary['category_counts']}")
        print(f"Output: {summary['output_parquet']}")
        print(f"Report: {report_dir}")
        return 0
    except Exception as exc:
        LOGGER.exception("OSM POI fetch failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
