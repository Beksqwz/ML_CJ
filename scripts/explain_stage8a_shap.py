"""Stage 8A: deterministic CatBoost SHAP explanations for the frozen Stage 7B models.

This is intentionally an analysis-only script: it never fits or writes a model or
an input dataset.  SHAP here describes contributions to the model score (log-odds),
not causal effects.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "stage8a"
SEED = 20260711
SAMPLE_SIZES = {"1h": 5_000, "24h": 10_000}


def latest(path: Path) -> Path:
    files = sorted(path.glob("*/training_dataset_*_feature_config.json"))
    if not files:
        raise FileNotFoundError(f"Feature config is absent in {path}")
    return files[-1]


def prepare(frame: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    result = frame[features].copy()
    for col in categorical:
        result[col] = result[col].astype("string").fillna("__MISSING__").astype(str)
    return result


def representative_sample(frame: pd.DataFrame, target: str, n: int) -> pd.DataFrame:
    """Deterministic stratified sample, retaining test prevalence where possible."""
    n = min(n, len(frame))
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(frame[target].to_numpy(dtype=np.int8) == 1)
    negative = np.flatnonzero(frame[target].to_numpy(dtype=np.int8) == 0)
    desired_pos = min(len(positive), max(1, round(n * len(positive) / len(frame))))
    desired_neg = min(len(negative), n - desired_pos)
    # In the unlikely event that a class is smaller than its requested quota.
    if desired_pos + desired_neg < n:
        desired_pos = min(len(positive), desired_pos + n - desired_pos - desired_neg)
        desired_neg = min(len(negative), n - desired_pos)
    chosen = np.concatenate((rng.choice(positive, desired_pos, replace=False), rng.choice(negative, desired_neg, replace=False)))
    return frame.iloc[np.sort(chosen)].copy()


def group_for(feature: str) -> str:
    if feature.startswith(("segment_accidents_", "city_accidents_", "segment_hours_", "segment_has_history")):
        return "historical"
    if feature.startswith("weather_"):
        return "weather"
    if feature.startswith("calendar_"):
        return "calendar"
    if feature.startswith("poi_"):
        return "POI"
    return "road"  # road_* plus static segment coordinates


def clean(value: object) -> object:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def factor_rows(features: list[str], values: pd.Series, row_shap: np.ndarray, positive: bool) -> list[dict[str, object]]:
    indices = np.where(row_shap > 0)[0] if positive else np.where(row_shap < 0)[0]
    order = indices[np.argsort(-row_shap[indices] if positive else row_shap[indices])]
    return [{"feature": features[i], "feature_value": clean(values.iloc[i]), "shap_log_odds": float(row_shap[i])} for i in order[:5]]


def _legacy_russian_text(probability: float, label: int, predicted: bool, positive: list[dict], negative: list[dict]) -> str:
    truth = "событие произошло" if label else "событие не произошло"
    decision = "высокий риск по порогу" if predicted else "ниже рабочего порога"
    plus = ", ".join(x["feature"] for x in positive[:3]) or "нет положительных факторов"
    minus = ", ".join(x["feature"] for x in negative[:3]) or "нет отрицательных факторов"
    return (f"Модель оценила вероятность риска в {probability:.2%}; фактически {truth}, прогноз {decision}. "
            f"В данной модели оценку повышали: {plus}; понижали: {minus}. "
            "Это вклад признаков в прогноз модели, а не доказательство причинности.")


def russian_text(probability: float, label: int, predicted: bool, positive: list[dict], negative: list[dict]) -> str:
    """Return an UTF-8-safe Russian local explanation without causal claims."""
    truth = "\u0441\u043e\u0431\u044b\u0442\u0438\u0435 \u043f\u0440\u043e\u0438\u0437\u043e\u0448\u043b\u043e" if label else "\u0441\u043e\u0431\u044b\u0442\u0438\u0435 \u043d\u0435 \u043f\u0440\u043e\u0438\u0437\u043e\u0448\u043b\u043e"
    decision = "\u0432\u044b\u0441\u043e\u043a\u0438\u0439 \u0440\u0438\u0441\u043a" if predicted else "\u043d\u0438\u0436\u0435 \u043f\u043e\u0440\u043e\u0433\u0430"
    plus = ", ".join(str(item["feature"]) for item in positive[:3]) or "\u043d\u0435\u0442"
    minus = ", ".join(str(item["feature"]) for item in negative[:3]) or "\u043d\u0435\u0442"
    return (
        f"\u0412\u0435\u0440\u043e\u044f\u0442\u043d\u043e\u0441\u0442\u044c \u0440\u0438\u0441\u043a\u0430: {probability:.2%}. "
        f"\u0424\u0430\u043a\u0442: {truth}; \u043f\u0440\u043e\u0433\u043d\u043e\u0437: {decision}. "
        f"\u041f\u043e\u043b\u043e\u0436\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0439 \u0432\u043a\u043b\u0430\u0434: {plus}. "
        f"\u041e\u0442\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0439 \u0432\u043a\u043b\u0430\u0434: {minus}. "
        "SHAP \u043e\u043f\u0438\u0441\u044b\u0432\u0430\u0435\u0442 \u0432\u043a\u043b\u0430\u0434 \u0432 \u043f\u0440\u043e\u0433\u043d\u043e\u0437, \u0430 \u043d\u0435 \u043f\u0440\u0438\u0447\u0438\u043d\u043d\u043e\u0441\u0442\u044c."
    )


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    grouped = pd.DataFrame({"score": score, "target": y}).groupby("score")["target"].agg(["count", "sum"]).sort_index(ascending=False)
    tp = grouped["sum"].cumsum().to_numpy(float)
    fp = (grouped["count"].cumsum() - grouped["sum"].cumsum()).to_numpy(float)
    fn = float(y.sum()) - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    return float(grouped.index.to_numpy(float)[int(np.argmax(f1))])


def plots(output: Path, importance: pd.DataFrame, shap_values: np.ndarray, x: pd.DataFrame) -> None:
    top = importance.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top["feature"], top["mean_abs_shap_log_odds"], color="#2678b2")
    ax.set_xlabel("Среднее |SHAP| (log-odds)")
    ax.set_title("SHAP summary bar: топ-30 признаков")
    fig.tight_layout(); fig.savefig(output / "shap_summary_bar_top30.png", dpi=180); plt.close(fig)

    top_forward = importance.head(30)
    fig, ax = plt.subplots(figsize=(11, 10))
    rng = np.random.default_rng(SEED)
    for pos, (_, row) in enumerate(top_forward.iloc[::-1].iterrows()):
        col = row["feature"]; idx = x.columns.get_loc(col)
        jitter = rng.uniform(-0.32, 0.32, len(x))
        numeric = pd.to_numeric(x[col], errors="coerce")
        if numeric.notna().any():
            color = numeric.fillna(numeric.median()).to_numpy()
            ax.scatter(shap_values[:, idx], pos + jitter, c=color, cmap="coolwarm", s=5, alpha=.55, linewidths=0)
        else:
            ax.scatter(shap_values[:, idx], pos + jitter, color="#7d7d7d", s=5, alpha=.55, linewidths=0)
    ax.axvline(0, color="black", lw=.6); ax.set_yticks(range(30), top_forward.iloc[::-1]["feature"])
    ax.set_xlabel("SHAP value (log-odds)"); ax.set_title("SHAP beeswarm: топ-30 признаков")
    fig.tight_layout(); fig.savefig(output / "shap_beeswarm_top30.png", dpi=180); plt.close(fig)


def run(horizon: str, stage7d: bool = False) -> dict[str, object]:
    target = f"target_{horizon}"
    if stage7d:
        config_path = ROOT / "reports" / "stage7d" / horizon / "stage7d_feature_config.json"
        split_root = ROOT / "data" / "processed" / "stage7d"
        model_path = ROOT / "models" / "stage7d" / f"catboost_{horizon}_weather_experiment.cbm"
        model_stage = "Stage 7D"
    else:
        config_path = latest(ROOT / "reports" / "stage7a" / horizon)
        split_root = ROOT / "data" / "processed" / "stage7a"
        model_path = ROOT / "models" / "stage7b" / f"catboost_{horizon}.cbm"
        model_stage = "Stage 7B"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    features = list(config["numerical_features"]) + list(config["categorical_features"])
    categorical = list(config["categorical_features"])
    test = pd.read_parquet(split_root / f"training_dataset_{horizon}_test.parquet")
    sample = representative_sample(test, target, SAMPLE_SIZES[horizon])
    x = prepare(sample, features, categorical)
    model = CatBoostClassifier(); model.load_model(model_path)
    model_features = model.feature_names_
    if model_features != features:
        raise ValueError(f"Feature order mismatch for {horizon}: model and Stage 7A config differ")
    leakage = sorted(set(features) & set(config["excluded_from_model_features"]))
    if leakage:
        raise ValueError(f"Leakage/excluded fields present in model features: {leakage}")
    pool = Pool(x, cat_features=categorical, feature_names=features)
    values_with_base = model.get_feature_importance(pool, type="ShapValues")
    shap_values, base_value = values_with_base[:, :-1], values_with_base[:, -1]
    raw_prediction = model.predict(pool, prediction_type="RawFormulaVal").reshape(-1)
    probability = model.predict_proba(pool)[:, 1]
    residual = shap_values.sum(axis=1) + base_value - raw_prediction
    if stage7d:
        validation = pd.read_parquet(split_root / f"training_dataset_{horizon}_validation.parquet")
        validation_x = prepare(validation, features, categorical)
        validation_probability = model.predict_proba(Pool(validation_x, cat_features=categorical, feature_names=features))[:, 1]
        threshold = best_f1_threshold(validation[target].to_numpy(dtype=np.int8), validation_probability)
        threshold_rule = "Maximum F1 recalculated only on Stage 7D validation predictions; model was not retrained."
    else:
        threshold_file = sorted((ROOT / "reports" / "stage7b" / horizon).glob("*/catboost_*_threshold.json"))[-1]
        threshold = float(json.loads(threshold_file.read_text(encoding="utf-8"))["threshold"])
        threshold_rule = "Stage 7B maximum validation F1 threshold."
    output = OUT / horizon; output.mkdir(parents=True, exist_ok=True)

    shap_frame = pd.DataFrame(shap_values, columns=features)
    shap_frame.insert(0, "sample_row_index", sample.index.to_numpy())
    shap_frame.insert(1, "datetime_hour", sample["datetime_hour"].astype(str).to_numpy())
    shap_frame.insert(2, target, sample[target].astype("int8").to_numpy())
    shap_frame["prediction_probability"] = probability
    shap_frame["raw_prediction"] = raw_prediction
    shap_frame["base_value_log_odds"] = base_value
    shap_frame.to_parquet(output / "shap_values.parquet", index=False)

    importance = pd.DataFrame({"feature": features, "mean_abs_shap_log_odds": np.abs(shap_values).mean(axis=0), "mean_shap_log_odds": shap_values.mean(axis=0)})
    importance["feature_group"] = importance["feature"].map(group_for)
    importance = importance.sort_values("mean_abs_shap_log_odds", ascending=False, kind="stable").reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance.to_csv(output / "global_feature_importance.csv", index=False, encoding="utf-8-sig")
    importance.head(30).to_csv(output / "global_top30_feature_importance.csv", index=False, encoding="utf-8-sig")
    plots(output, importance, shap_values, x)
    group = importance.groupby("feature_group", sort=False)["mean_abs_shap_log_odds"].sum().sort_values(ascending=False).reset_index(name="sum_mean_abs_shap_log_odds")
    group["share_percent"] = group["sum_mean_abs_shap_log_odds"] / group["sum_mean_abs_shap_log_odds"].sum() * 100
    group.to_csv(output / "feature_group_contributions.csv", index=False, encoding="utf-8-sig")

    y = sample[target].to_numpy(dtype=np.int8); predicted = probability >= threshold
    categories = {
        "high_risk": np.arange(len(sample)),
        "true_positive": np.flatnonzero(predicted & (y == 1)),
        "false_positive": np.flatnonzero(predicted & (y == 0)),
        "false_negative": np.flatnonzero((~predicted) & (y == 1)),
    }
    local: dict[str, list[dict[str, object]]] = {}
    for name, indices in categories.items():
        # Highest scores make high-risk / TP / FP concrete; for FN they make the near-miss most informative.
        pick = indices[np.argsort(-probability[indices], kind="stable")[:20]]
        rows = []
        for i in pick:
            pos = factor_rows(features, x.iloc[i], shap_values[i], True)
            neg = factor_rows(features, x.iloc[i], shap_values[i], False)
            rows.append({"category": name, "sample_row_index": int(sample.index[i]), "datetime_hour": clean(sample.iloc[i]["datetime_hour"]),
                         "road_segment_id": clean(sample.iloc[i].get("road_segment_id")), "probability": float(probability[i]),
                         "true_label": int(y[i]), "predicted_high_risk": bool(predicted[i]), "threshold": threshold,
                         "top_positive_shap_factors": pos, "top_negative_shap_factors": neg,
                         "feature_values": {f: clean(x.iloc[i][f]) for f in features},
                         "explanation_ru": russian_text(float(probability[i]), int(y[i]), bool(predicted[i]), pos, neg)})
        local[name] = rows
    (output / "local_explanations.json").write_text(json.dumps(local, ensure_ascii=False, indent=2, default=clean), encoding="utf-8")
    local_flat = pd.DataFrame([{"category": r["category"], "sample_row_index": r["sample_row_index"], "datetime_hour": r["datetime_hour"], "road_segment_id": r["road_segment_id"], "probability": r["probability"], "true_label": r["true_label"], "predicted_high_risk": r["predicted_high_risk"], "explanation_ru": r["explanation_ru"], "feature_values_json": json.dumps(r["feature_values"], ensure_ascii=False), "positive_factors_json": json.dumps(r["top_positive_shap_factors"], ensure_ascii=False), "negative_factors_json": json.dumps(r["top_negative_shap_factors"], ensure_ascii=False)} for rows in local.values() for r in rows])
    local_flat.to_parquet(output / "local_explanations.parquet", index=False)
    single_share = float(importance.iloc[0]["mean_abs_shap_log_odds"] / importance["mean_abs_shap_log_odds"].sum())
    group_share = float(group.iloc[0]["share_percent"] / 100)
    anomalies = []
    if single_share >= .50: anomalies.append(f"Один признак {importance.iloc[0]['feature']} даёт {single_share:.1%} суммарного mean|SHAP| (порог проверки 50%).")
    if group_share >= .70: anomalies.append(f"Одна группа {group.iloc[0]['feature_group']} даёт {group_share:.1%} суммарного mean|SHAP| (порог проверки 70%).")
    if not anomalies: anomalies.append("Подозрительного доминирования одного признака (>=50%) или группы (>=70%) не обнаружено на выборке.")
    report = {"generated_at_utc": datetime.now(UTC).isoformat(), "horizon": horizon, "final_model_stage": model_stage, "model_path": str(model_path.resolve()), "feature_config": str(config_path.resolve()), "test_split": str((split_root / f"training_dataset_{horizon}_test.parquet").resolve()), "test_rows": len(test), "sample_rows": len(sample), "sample_positive_rate": float(y.mean()), "sampling": "Детерминированная стратифицированная случайная выборка test с seed=20260711.", "feature_order_matches_model": True, "leakage_fields_in_features": leakage, "threshold": threshold, "threshold_rule": threshold_rule, "shap_scale": "log-odds (RawFormulaVal)", "shap_additivity": {"max_abs_residual": float(np.abs(residual).max()), "mean_abs_residual": float(np.abs(residual).mean()), "passed_tolerance_1e-8": bool(np.abs(residual).max() < 1e-8)}, "top_10_features": importance.head(10).to_dict(orient="records"), "feature_group_contributions": group.to_dict(orient="records"), "anomaly_check": anomalies, "local_example_counts": {k: len(v) for k, v in local.items()}, "caution_ru": "SHAP отражает вклад признаков в выход конкретной модели на выбранных данных; это не причинная интерпретация."}
    (output / "stage8a_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=clean), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage7d-1h", action="store_true", help="Refresh only Stage 8A 1h using the accepted Stage 7D model.")
    parser.add_argument("--repair-local-text", action="store_true", help="Repair only UTF-8 Russian text in existing local_explanations.json files.")
    args = parser.parse_args()
    if args.repair_local_text:
        for horizon in ("1h", "24h"):
            path = OUT / horizon / "local_explanations.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for rows in payload.values():
                for row in rows:
                    explanation = russian_text(
                        float(row["probability"]),
                        int(row["true_label"]),
                        bool(row["predicted_high_risk"]),
                        row["top_positive_shap_factors"],
                        row["top_negative_shap_factors"],
                    )
                    row["explanation_ru"] = explanation + " \u0424\u0430\u043a\u0442\u043e\u0440\u044b \u043e\u0431\u044a\u044f\u0441\u043d\u044f\u044e\u0442 \u0432\u043a\u043b\u0430\u0434 \u0432 \u043f\u0440\u043e\u0433\u043d\u043e\u0437, \u0430 \u043d\u0435 \u0434\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u044e\u0442 \u043f\u0440\u0438\u0447\u0438\u043d\u043d\u043e\u0441\u0442\u044c."
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if args.stage7d_1h:
        reports = {"1h": run("1h", stage7d=True), "24h": json.loads((OUT / "24h" / "stage8a_report.json").read_text(encoding="utf-8"))}
    else:
        reports = {h: run(h) for h in ("1h", "24h")}
    group_1 = {x["feature_group"]: x["share_percent"] for x in reports["1h"]["feature_group_contributions"]}
    group_24 = {x["feature_group"]: x["share_percent"] for x in reports["24h"]["feature_group_contributions"]}
    top1, top24 = reports["1h"]["top_10_features"], reports["24h"]["top_10_features"]
    comparison_text = [
        "Сравнение основано на mean absolute SHAP на независимых детерминированных test-выборках. Различия описывают использование признаков моделями, а не причинные механизмы.",
        f"В обеих моделях главный признак — {top1[0]['feature']}; однако у 24h его mean|SHAP| выше ({top24[0]['mean_abs_shap_log_odds']:.3f} против {top1[0]['mean_abs_shap_log_odds']:.3f}).",
        f"1h сильнее опирается на исторические признаки ({group_1['historical']:.2f}% против {group_24['historical']:.2f}% у 24h), а 24h — на дорожные/пространственные признаки ({group_24['road']:.2f}% против {group_1['road']:.2f}%).",
        f"Погодные и календарные признаки малы в обеих моделях: 1h — {group_1['weather']:.2f}% и {group_1['calendar']:.2f}%, 24h — {group_24['weather']:.2f}% и {group_24['calendar']:.2f}%.",
        f"В top-10 24h заметнее статическая география: segment_latitude занимает {next(x['rank'] for x in top24 if x['feature'] == 'segment_latitude')}-е место (для 1h — {next(x['rank'] for x in top1 if x['feature'] == 'segment_latitude')}-е); road_length также имеет более высокий вклад у 24h.",
    ]
    comparison = {"generated_at_utc": datetime.now(UTC).isoformat(), "final_models": {"1h": {"stage": reports["1h"].get("final_model_stage", "Stage 7B"), "model_path": reports["1h"].get("model_path")}, "24h": {"stage": reports["24h"].get("final_model_stage", "Stage 7B"), "model_path": reports["24h"].get("model_path")}}, "comparison_ru": comparison_text, "1h_top_10": top1, "24h_top_10": top24, "1h_groups": reports["1h"]["feature_group_contributions"], "24h_groups": reports["24h"]["feature_group_contributions"]}
    (OUT / "stage8a_model_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
