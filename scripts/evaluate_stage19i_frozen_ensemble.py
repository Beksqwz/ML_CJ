"""Stage 19I frozen simple-ensemble smoke evaluator."""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scripts.evaluate_24h_natural_prevalence import (
    CONFIG_PATH,
    EXPECTED_HASH,
    PROCESSED,
    build_feature_context,
    build_features_for_chunk,
    feature_hash,
    target_fields,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-dir", default="data/audit/stage19i_simple/smoke")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cfg = json.loads(
        (ROOT / "models/stage19i_simple/weighted_ensemble_config.json").read_text()
    )
    w = cfg["weights"]
    catboost_weight = float(w["score_catboost_stage19h"])
    hgb_weight = float(w["score_hist_gradient_boosting"])
    assert catboost_weight >= 0 and hgb_weight >= 0
    assert math.isclose(catboost_weight, 0.8, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(hgb_weight, 0.2, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(catboost_weight + hgb_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    c = json.loads(CONFIG_PATH.read_text())
    f = [*c["numerical_features"], *c["categorical_features"]]
    assert feature_hash(f) == EXPECTED_HASH
    hours = pd.date_range("2024-09-30 19:00:00", periods=6, freq="h")
    ready = pd.read_parquet(PROCESSED / "accidents_with_roads_ml_ready.parquet")
    ctx = build_feature_context(ready, c, hours)
    grid = pd.MultiIndex.from_product(
        [hours, ctx.segment_order], names=["prediction_datetime", "road_segment_id"]
    ).to_frame(index=False)
    raw = build_features_for_chunk(grid, ctx, c)
    targets = target_fields(
        raw[["road_segment_id", "prediction_datetime"]], ctx.counts
    )[["road_segment_id", "prediction_datetime", "target_24h"]]
    cat = CatBoostClassifier()
    cat.load_model(ROOT / "models/stage19h/catboost_candidate.cbm")
    cf = raw[f].copy()
    for col in c["categorical_features"]:
        cf[col] = cf[col].astype("string").fillna("__MISSING__").astype(str)
    cs = cat.predict_proba(cf)[:, 1]
    pre = joblib.load(ROOT / "models/stage19h/train_only_preprocessor_v2.joblib")
    hgb = joblib.load(ROOT / "models/stage19h/hist_gradient_boosting_candidate.joblib")
    hs = hgb.predict_proba(pre.transform(raw))[:, 1]
    o = raw[["road_segment_id", "prediction_datetime"]].merge(
        targets, on=["road_segment_id", "prediction_datetime"]
    )
    o["score_catboost_stage19h"] = cs
    o["score_hist_gradient_boosting"] = hs
    o["rank_percentile_catboost"] = o.groupby("prediction_datetime")[
        "score_catboost_stage19h"
    ].rank(pct=True)
    o["rank_percentile_hgb"] = o.groupby("prediction_datetime")[
        "score_hist_gradient_boosting"
    ].rank(pct=True)
    o["ensemble_score"] = (
        catboost_weight * o.rank_percentile_catboost
        + hgb_weight * o.rank_percentile_hgb
    )
    o["ensemble_rank_within_hour"] = (
        o.groupby("prediction_datetime")["ensemble_score"]
        .rank(method="first", ascending=False)
        .astype("int32")
    )
    out = ROOT / a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out / "part_0000.parquet", index=False)
    assert (
        len(o) == 23808
        and not o.duplicated(["road_segment_id", "prediction_datetime"]).any()
        and np.isfinite(
            o[
                [
                    "ensemble_score",
                    "score_catboost_stage19h",
                    "score_hist_gradient_boosting",
                ]
            ]
        )
        .all()
        .all()
    )


if __name__ == "__main__":
    main()
