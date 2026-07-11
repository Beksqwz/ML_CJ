"""Stage 7D: causal weather-feature experiment, isolated from Stage 7A/7B assets.

Copies the already frozen Stage 7A splits, adds weather information available at
the row timestamp (and strict-past rolling windows), trains experimental CatBoost
models, and compares them to frozen Stage 7B.  It never overwrites Stage 7A/7B.
"""
from __future__ import annotations

import json
import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError, Pool

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260711
EXPERIMENT = "stage7d"
OUT_DATA = ROOT / "data" / "processed" / EXPERIMENT
OUT_MODELS = ROOT / "models" / EXPERIMENT
OUT_REPORTS = ROOT / "reports" / EXPERIMENT


def latest(folder: Path, pattern: str) -> Path:
    result = sorted(folder.glob(pattern))
    if not result: raise FileNotFoundError(f"No {pattern} under {folder}")
    return result[-1]


def risk_features() -> tuple[pd.DataFrame, list[str], list[str]]:
    """One feature row per hour; all *_prev_* windows exclude the current hour."""
    weather = pd.read_parquet(ROOT / "data" / "external" / "weather_astana_hourly.parquet").sort_values("datetime_hour").copy()
    weather["datetime_hour"] = pd.to_datetime(weather["datetime_hour"])
    if weather["datetime_hour"].duplicated().any() or not weather["datetime_hour"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Weather series must be unique and continuous hourly for causal rolling features.")
    precip = weather["precipitation"].fillna(0.0); snow = weather["snowfall"].fillna(0.0)
    weather["weather_risk_precip_now"] = (precip > 0).astype("int8")
    weather["weather_risk_snow_now"] = (snow > 0).astype("int8")
    weather["weather_risk_freezing_now"] = (weather["temperature_2m"] <= 0).astype("int8")
    weather["weather_risk_high_wind_now"] = ((weather["wind_speed_10m"] >= 10) | (weather["wind_gusts_10m"] >= 15)).astype("int8")
    weather["weather_risk_adverse_now"] = ((weather[["weather_risk_precip_now", "weather_risk_snow_now", "weather_risk_freezing_now", "weather_risk_high_wind_now"]].sum(axis=1)) > 0).astype("int8")
    for hours in (3, 6, 24):
        # shift(1): data in the forecasting hour is never used in these aggregates.
        weather[f"weather_precip_sum_prev_{hours}h"] = precip.shift(1).rolling(hours, min_periods=1).sum()
        weather[f"weather_snow_sum_prev_{hours}h"] = snow.shift(1).rolling(hours, min_periods=1).sum()
        weather[f"weather_wind_mean_prev_{hours}h"] = weather["wind_speed_10m"].shift(1).rolling(hours, min_periods=1).mean()
        weather[f"weather_adverse_hours_prev_{hours}h"] = weather["weather_risk_adverse_now"].shift(1).rolling(hours, min_periods=1).sum()
    weather["weather_temperature_change_1h"] = weather["temperature_2m"] - weather["temperature_2m"].shift(1)
    weather["weather_temperature_change_3h"] = weather["temperature_2m"] - weather["temperature_2m"].shift(3)
    # Difference in hours to the latest precipitation at or before now: current observed weather is allowed.
    last = weather["datetime_hour"].where(weather["weather_risk_precip_now"].eq(1)).ffill()
    weather["weather_hours_since_precip"] = (weather["datetime_hour"] - last).dt.total_seconds().div(3600)
    weather["weather_hours_since_precip"] = weather["weather_hours_since_precip"].fillna(9_999.0)
    numeric = [c for c in weather if c.startswith("weather_") and c not in {"weather_weather_code"} and c not in {"weather_risk_road_highway"}]
    # The existing weather fields are intentionally not duplicated; only newly derived fields are appended later.
    derived = [c for c in numeric if c not in {"weather_temperature_2m", "weather_relative_humidity_2m", "weather_precipitation", "weather_rain", "weather_snowfall", "weather_cloud_cover", "weather_wind_speed_10m", "weather_wind_gusts_10m"}]
    return weather[["datetime_hour", *derived]], derived, []


def add_features(frame: pd.DataFrame, weather: pd.DataFrame, derived: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    result = frame.merge(weather, on="datetime_hour", how="left", validate="many_to_one")
    if result[derived].isna().all(axis=None): raise ValueError("Weather feature merge failed")
    # Current weather and prior-only aggregates are global hourly information known at prediction time.
    result["weather_interaction_adverse_rush_hour"] = result["weather_risk_adverse_now"] * result["calendar_is_rush_hour"].astype(str).str.lower().isin(["true", "1"]).astype("int8")
    result["weather_interaction_adverse_maxspeed"] = result["weather_risk_adverse_now"] * result["road_maxspeed_kmh"].fillna(0)
    result["weather_interaction_precip_history"] = result["weather_precip_sum_prev_24h"] * result["segment_accidents_total_prior"].fillna(0)
    result["weather_interaction_adverse_history"] = result["weather_risk_adverse_now"] * result["segment_accidents_prev_30d"].fillna(0)
    # Explicit road-type interaction; its components are each known at the row time.
    result["weather_interaction_adverse_road_highway"] = (result["weather_risk_adverse_now"].astype(str) + "__" + result["road_highway"].astype("string").fillna("UNKNOWN").astype(str)).astype("string")
    numeric = [*derived, "weather_interaction_adverse_rush_hour", "weather_interaction_adverse_maxspeed", "weather_interaction_precip_history", "weather_interaction_adverse_history"]
    categorical = ["weather_interaction_adverse_road_highway"]
    if result[numeric].isna().any().any():
        # Initial history is the only legitimate missing case; numeric CatBoost NaN handling is retained.
        pass
    return result, numeric, categorical


def prep(frame: pd.DataFrame, features: list[str], cats: list[str]) -> pd.DataFrame:
    x = frame[features].copy()
    for c in cats: x[c] = x[c].astype("string").fillna("__MISSING__").astype(str)
    return x


def pr_auc(y, score):
    g = pd.DataFrame({"score": score, "y": y}).groupby("score")["y"].agg(["count", "sum"]).sort_index(ascending=False)
    recall = g["sum"].cumsum() / y.sum(); precision = g["sum"].cumsum() / g["count"].cumsum()
    return float((precision * recall.diff().fillna(recall)).sum())


def calibration(y, score):
    bins = np.clip(np.digitize(score, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    ece = 0.; rows = []
    for b in range(10):
        m = bins == b
        if m.any():
            mp, obs = float(score[m].mean()), float(y[m].mean()); ece += abs(mp - obs) * m.sum()
            rows.append({"bin": b, "rows": int(m.sum()), "mean_prediction": mp, "observed_rate": obs})
    return {"brier_score": float(np.mean((score-y)**2)), "expected_calibration_error_10_bins": float(ece/len(y)), "bins": rows}


def metrics(y, score):
    top = np.argsort(-score, kind="stable")[:max(1, int(np.ceil(.1 * len(y))))]
    return {"pr_auc": pr_auc(y, score), "lift_at_top_10pct": float(y[top].mean()/y.mean()), "recall_at_top_10pct": float(y[top].sum()/y.sum()), "calibration": calibration(y, score)}


def stratified(frame, target, n):
    rng=np.random.default_rng(SEED); y=frame[target].to_numpy(); p=np.flatnonzero(y==1); q=np.flatnonzero(y==0); np_=round(n*len(p)/len(y))
    ids=np.r_[rng.choice(p, min(len(p),np_), replace=False),rng.choice(q, min(len(q),n-np_),replace=False)]
    return frame.iloc[np.sort(ids)]


def weather_shap_share(model, frame, target, features, cats):
    sample = stratified(frame, target, 5_000 if target == "target_1h" else 10_000)
    x=prep(sample, features, cats); vals=model.get_feature_importance(Pool(x, cat_features=cats), type="ShapValues")[:, :-1]
    all_abs=np.abs(vals).mean(axis=0); weather_abs=all_abs[[f.startswith("weather_") for f in features]].sum()
    return {"sample_rows": len(sample), "weather_mean_abs_shap": float(weather_abs), "all_mean_abs_shap": float(all_abs.sum()), "weather_share_percent": float(100*weather_abs/all_abs.sum())}


def train(horizon, splits, features, cats):
    target=f"target_{horizon}"; x={k:prep(v,features,cats) for k,v in splits.items()}; y={k:v[target].to_numpy(np.int8) for k,v in splits.items()}
    params={"iterations":1500,"learning_rate":.05,"depth":7,"l2_leaf_reg":5.,"loss_function":"Logloss","eval_metric":"PRAUC","random_seed":SEED,"allow_writing_files":False}
    cat_idx=[features.index(c) for c in cats]
    try:
        model=CatBoostClassifier(**(params|{"task_type":"GPU","devices":"0"})); model.fit(x["train"],y["train"],cat_features=cat_idx,eval_set=(x["validation"],y["validation"]),early_stopping_rounds=100,verbose=100); device="GPU"
    except (CatBoostError, RuntimeError) as exc:
        model=CatBoostClassifier(**(params|{"task_type":"CPU","thread_count":-1})); model.fit(x["train"],y["train"],cat_features=cat_idx,eval_set=(x["validation"],y["validation"]),early_stopping_rounds=100,verbose=100); device=f"CPU fallback: {exc}"
    prediction={k:model.predict_proba(x[k])[:,1] for k in ("validation","test")}
    return model, device, {k:metrics(y[k], prediction[k]) for k in prediction}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", choices=("1h", "24h"), nargs="*")
    parser.add_argument("--comparison-only", action="store_true")
    args = parser.parse_args()
    if args.comparison_only:
        reports = {h: json.loads((OUT_REPORTS / h / "stage7d_weather_experiment_report.json").read_text(encoding="utf-8")) for h in ("1h", "24h")}
        (OUT_REPORTS / "stage7d_comparison.json").write_text(json.dumps({"generated_at_utc": datetime.now(UTC).isoformat(), "1h": reports["1h"], "24h": reports["24h"]}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return
    weather, derived, _ = risk_features(); all_reports={}
    horizons = args.horizon or ("1h", "24h")
    for h in horizons:
        target=f"target_{h}"; config_path=latest(ROOT/"reports"/"stage7a"/h, "*/training_dataset_*_feature_config.json"); config=json.loads(config_path.read_text(encoding="utf-8"))
        old={name:pd.read_parquet(ROOT/"data"/"processed"/"stage7a"/f"training_dataset_{h}_{name}.parquet") for name in ("train","validation","test")}
        enhanced={}; new_numeric=new_cat=None
        for name, frame in old.items(): enhanced[name], new_numeric, new_cat=add_features(frame,weather,derived)
        features=list(config["numerical_features"])+new_numeric+list(config["categorical_features"])+new_cat; cats=list(config["categorical_features"])+new_cat
        # Persist distinct experimental split copies, preserving original row order/timestamps exactly.
        OUT_DATA.mkdir(parents=True, exist_ok=True)
        for name, frame in enhanced.items(): frame.to_parquet(OUT_DATA/f"training_dataset_{h}_{name}.parquet",index=False)
        config_out = {"target_column": target, "numerical_features": list(config["numerical_features"])+new_numeric, "categorical_features": cats,
                      "excluded_from_model_features": config["excluded_from_model_features"], "source_stage7a_config": str(config_path.resolve()),
                      "new_weather_numerical_features": new_numeric, "new_weather_categorical_features": new_cat}
        d=OUT_REPORTS/h; d.mkdir(parents=True,exist_ok=True)
        (d/"stage7d_feature_config.json").write_text(json.dumps(config_out,ensure_ascii=False,indent=2),encoding="utf-8")
        model, device, scores=train(h,enhanced,features,cats); OUT_MODELS.mkdir(parents=True,exist_ok=True); model.save_model(OUT_MODELS/f"catboost_{h}_weather_experiment.cbm")
        frozen=CatBoostClassifier(); frozen.load_model(ROOT/"models"/"stage7b"/f"catboost_{h}.cbm")
        base_features=list(config["numerical_features"])+list(config["categorical_features"]); base_cats=list(config["categorical_features"])
        base_shap=weather_shap_share(frozen,old["test"],target,base_features,base_cats); exp_shap=weather_shap_share(model,enhanced["test"],target,features,cats)
        baseline_report=json.loads(latest(ROOT/"reports"/"stage7b"/h,"*/catboost_*_report.json").read_text(encoding="utf-8"))
        ranking_ok=all(scores[s][m] >= baseline_report["metrics"][s][m] - 1e-12 for s in ("validation","test") for m in ("pr_auc","lift_at_top_10pct"))
        calibration_ok=all(scores[s]["calibration"][m] <= baseline_report["calibration"][s][m] + 1e-12 for s in ("validation","test") for m in ("brier_score","expected_calibration_error_10_bins"))
        quality_ok=ranking_ok and calibration_ok
        accepted=bool(quality_ok and exp_shap["weather_share_percent"] > base_shap["weather_share_percent"])
        report={"generated_at_utc":datetime.now(UTC).isoformat(),"horizon":h,"experiment":"causal weather signal enrichment; no artificial model weighting","source_stage7a_config":str(config_path.resolve()),"source_weather":str((ROOT/"data"/"external"/"weather_astana_hourly.parquet").resolve()),"causality":{"current_weather":"Observed at datetime_hour and therefore available at prediction time.","rolling":"All rolling aggregates apply shift(1), so they use only hours strictly earlier than datetime_hour.","future_actual_weather_used":False},"new_numerical_features":new_numeric,"new_categorical_features":new_cat,"split_boundaries_preserved":all(enhanced[k]["datetime_hour"].equals(old[k]["datetime_hour"]) for k in old),"model_path":str((OUT_MODELS/f"catboost_{h}_weather_experiment.cbm").resolve()),"device":device,"best_iteration":int(model.get_best_iteration()),"experimental_metrics":scores,"stage7b_metrics":{s:{m:baseline_report["metrics"][s][m] for m in ("pr_auc","lift_at_top_10pct")} | {"calibration":baseline_report["calibration"][s]} for s in ("validation","test")},"weather_shap": {"stage7b":base_shap,"stage7d":exp_shap},"acceptance_rule":"Accepted only if validation and test PR-AUC, Lift@10%, Brier score and ECE do not decline and weather SHAP share increases naturally.","ranking_not_degraded":ranking_ok,"calibration_not_degraded":calibration_ok,"quality_not_degraded":quality_ok,"accepted_as_candidate":accepted,"decision_ru":"Кандидат принят." if accepted else "Кандидат не принят: правило качества и естественного роста weather SHAP не выполнено."}
        (d/"stage7d_weather_experiment_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); all_reports[h]=report
    if set(horizons) == {"1h", "24h"}:
        (OUT_REPORTS/"stage7d_comparison.json").write_text(json.dumps({"generated_at_utc":datetime.now(UTC).isoformat(),"1h":all_reports["1h"],"24h":all_reports["24h"]},ensure_ascii=False,indent=2,default=str),encoding="utf-8")

if __name__ == "__main__": main()
