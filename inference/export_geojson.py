"""GeoJSON and backend JSON exporters for Stage 8C predictions."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .risk_thresholds import configured_risk_level, load_risk_thresholds

def _geometry_map(edges_path: Path) -> dict[str, dict]:
    edges = pd.read_csv(edges_path)
    result = {}
    for row in edges.itertuples(index=False):
        key = f"{row.u}_{row.v}_{row.key}"
        text = str(row.geometry).replace("LINESTRING", "").replace("(", "").replace(")", "").strip()
        coords = [[float(x), float(y)] for x, y in (pair.split() for pair in text.split(","))]
        result[key] = {"type": "LineString", "coordinates": coords}
    return result

def export(predictions: list[dict], horizon: str, output_dir: Path, edges_path: Path) -> tuple[Path, Path]:
    threshold_config = load_risk_thresholds()
    geometry = _geometry_map(edges_path); features=[]
    for row in predictions:
        if row["risk_level"] != configured_risk_level(float(row["risk_probability"]), threshold_config):
            raise ValueError("Risk level does not match the configured operational display threshold.")
        segment=str(row["road_segment_id"])
        if segment not in geometry: raise ValueError(f"No geometry for {segment}")
        properties={k: row[k] for k in ("road_segment_id","road_name","risk_probability","risk_level","model_horizon","top_positive_factors","recommendations")}
        features.append({"type":"Feature","geometry":geometry[segment],"properties":properties})
    geo={"type":"FeatureCollection","features":features}; output_dir.mkdir(parents=True,exist_ok=True)
    gp=output_dir/f"risk_map_{horizon}.geojson"; jp=output_dir/f"predictions_current_{horizon}.json"
    gp.write_text(json.dumps(geo,ensure_ascii=False),encoding="utf-8"); jp.write_text(json.dumps(predictions,ensure_ascii=False),encoding="utf-8")
    return gp,jp
