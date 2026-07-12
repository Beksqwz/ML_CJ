"""Coordinate feature assembly, frozen-model scoring, local explanations, and exports.

This is the Stage 8C runtime boundary. It never trains models and keeps full
SHAP arrays in memory only while selecting per-segment explanation factors.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from recommendations.engine import recommend
from .feature_builder import ROOT, _config, build_features
from .export_geojson import export
from .risk_thresholds import configured_risk_level, load_risk_thresholds


def run(datetime_hour: str, output_dir: Path) -> dict:
    threshold_config = load_risk_thresholds()
    summary = {
        "datetime_hour": datetime_hour,
        "risk_threshold_config_path": str(
            (ROOT / "config" / "risk_thresholds.json").resolve()
        ),
        "risk_threshold_config_version": threshold_config["version"],
        "risk_thresholds": threshold_config["levels"],
        "horizons": {},
    }
    for h, model_rel, version in (
        ("1h", "models/production/catboost_1h_weather_experiment.cbm", "Stage 7D"),
        ("24h", "models/production/catboost_24h.cbm", "Stage 7B"),
    ):
        started = time.perf_counter()
        data, build = build_features(datetime_hour, h)
        cfg = _config(h)
        feats = list(cfg["numerical_features"]) + list(cfg["categorical_features"])
        cats = list(cfg["categorical_features"])
        x = data[feats].copy()
        for c in cats:
            x[c] = x[c].astype("string").fillna("__MISSING__").astype(str)
        model = CatBoostClassifier()
        model.load_model(ROOT / model_rel)
        if model.feature_names_ != feats:
            raise ValueError(f"{h} feature order mismatch")
        pool = Pool(x, cat_features=cats, feature_names=feats)
        prob = model.predict_proba(pool)[:, 1]
        values = model.get_feature_importance(pool, type="ShapValues")
        shap = values[:, :-1]
        rows = []
        for i, record in data.reset_index(drop=True).iterrows():
            local = {f: float(shap[i, j]) for j, f in enumerate(feats)}
            rec = recommend(
                probability=float(prob[i]),
                shap_values=local,
                feature_values={
                    f: (None if str(record[f]) == "<NA>" else record[f]) for f in feats
                },
                model_horizon=h,
                final_model_version=version,
            )
            rows.append(
                {
                    "road_segment_id": str(record.road_segment_id),
                    "road_name": "UNKNOWN"
                    if pd.isna(record.road_name)
                    else str(record.road_name),
                    "risk_probability": float(prob[i]),
                    "risk_level": configured_risk_level(
                        float(prob[i]), threshold_config
                    ),
                    "model_horizon": h,
                    "top_positive_factors": rec["top_positive_factors"],
                    "top_negative_factors": rec["top_negative_factors"],
                    "recommendations": rec["recommendations"],
                    "final_model_version": version,
                }
            )
        gp, jp = export(rows, h, output_dir, ROOT / "data" / "roads" / "astana_edges.csv")
        elapsed = time.perf_counter() - started
        geo = json.loads(gp.read_text(encoding="utf8"))
        ready_segments = pd.read_parquet(
            ROOT / "data/processed/accidents_with_roads_ml_ready.parquet",
            columns=["road_segment_id"],
        ).road_segment_id.nunique()
        required = (
            "road_segment_id",
            "road_name",
            "risk_probability",
            "risk_level",
            "model_horizon",
            "top_positive_factors",
            "recommendations",
        )
        validation = {
            "segments": len(rows),
            "ml_ready_unique_segments": int(ready_segments),
            "unique_segments": len({r["road_segment_id"] for r in rows}),
            "probabilities_in_range": all(
                0 <= r["risk_probability"] <= 1 for r in rows
            ),
            "risk_levels_match_config": all(
                r["risk_level"]
                == configured_risk_level(float(r["risk_probability"]), threshold_config)
                for r in rows
            ),
            "risk_threshold_config_path": str(
                (ROOT / "config" / "risk_thresholds.json").resolve()
            ),
            "risk_threshold_config_version": threshold_config["version"],
            "required_export_fields_complete": all(
                all(r.get(k) is not None for k in required) for r in rows
            ),
            "recommendations_have_positive_evidence": all(
                all(x["evidence"]["shap_value"] > 0 for x in r["recommendations"])
                for r in rows
            ),
            "geojson_feature_count": len(geo["features"]),
            "geojson_valid": geo.get("type") == "FeatureCollection"
            and all(
                f.get("geometry", {}).get("type") == "LineString"
                and len(f["geometry"].get("coordinates", [])) >= 2
                for f in geo["features"]
            ),
        }
        if not (
            validation["segments"]
            == validation["ml_ready_unique_segments"]
            == validation["unique_segments"]
            == validation["geojson_feature_count"]
            and validation["probabilities_in_range"]
            and validation["risk_levels_match_config"]
            and validation["required_export_fields_complete"]
            and validation["recommendations_have_positive_evidence"]
            and validation["geojson_valid"]
        ):
            raise ValueError(f"Validation failed {h}: {validation}")
        summary["horizons"][h] = {
            "build": build,
            "validation": validation,
            "seconds": elapsed,
            "mean_seconds_per_segment": elapsed / len(rows),
            "geojson": str(gp),
            "json": str(jp),
            "geojson_bytes": gp.stat().st_size,
            "json_bytes": jp.stat().st_size,
            "high": sum(r["risk_level"] == "HIGH" for r in rows),
            "critical": sum(r["risk_level"] == "CRITICAL" for r in rows),
        }
    return summary
