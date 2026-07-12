# Road Risk Forecasting

This repository forecasts road-segment risk in Astana for one-hour and twenty-four-hour horizons. It combines road geometry, nearby points of interest, calendar context, observed weather, and strictly prior accident history. The system produces model probabilities, concise SHAP explanations, review-required recommendations, and map-ready GeoJSON.

## Pipeline

```text
Raw Data
  ↓
Validation
  ↓
Cleaning
  ↓
Feature Engineering
  ↓
CatBoost
  ↓
SHAP
  ↓
Recommendation Engine
  ↓
Inference
  ↓
GeoJSON
```

The pipeline uses chronological splits and strict-prior accident counters. It does not use target-window events or future weather when constructing a prediction row.

## Repository structure

- `config/` — versioned operational configuration, including display risk thresholds.
- `data/` — source-derived and processed datasets; training data is not rewritten by inference.
- `docs/` — architecture, data pipeline, and model notes.
- `inference/` — feature assembly, batch prediction, local TreeSHAP, and export code.
- `models/` — frozen CatBoost artifacts.
- `recommendations/` — rule-based, human-reviewed recommendation logic.
- `reports/` — validation, explainability, inference, and cleanup reports.
- `scripts/` — reproducible stage runners and maintenance utilities.
- `tests/` — focused rule-engine tests.

## Machine learning

Two frozen CatBoost classifiers are used in production:

- **1h:** Stage 7D weather experiment, `models/stage7d/catboost_1h_weather_experiment.cbm`, with 97 features. It adds causal weather rolling features to the Stage 7A feature set.
- **24h:** Stage 7B baseline, `models/stage7b/catboost_24h.cbm`, with 77 features.

Feature engineering covers road attributes, POI counts, calendar context, weather, and strictly prior historical accident counters. SHAP values describe each model's local contribution to a prediction; they do not establish cause and effect.

## Recommendation engine

The Stage 8B engine is rule-based. A recommendation requires matching local positive SHAP evidence and retains that evidence in its payload. Recommendations are grouped as operational, inspection, or long-term actions. They support review rather than replace professional judgment.

## Inference and exports

Stage 8C builds features for every known road segment at a requested hour, scores both horizons, computes only local SHAP factors needed for explanations, and emits JSON plus GeoJSON `FeatureCollection` files. Display levels use the versioned operational ranges in `config/risk_thresholds.json`; those ranges are not CatBoost classification thresholds.

Example fixed-time demo:

```powershell
python scripts/run_stage8c_inference.py
```

The demo uses a fixed historical hour, not the current date.

## Live weather (OpenWeather)

For a current-hour prediction, the backend can replace the archived weather row
with live observations from OpenWeather. Put `OPENWEATHER_API_KEY` in the
untracked `.env` file (copy `.env.example` first), then call:

```python
from ml_service import AccidentRiskPredictor

result = AccidentRiskPredictor().predict_current_city(horizon="1h")
```

`result["live_weather"]` contains the normalized current conditions and source.
If the key or API is unavailable, the call returns no predictions and an explicit
reason; it never silently falls back to the static historical weather file.

## Live traffic (TomTom, optional)

TomTom Flow Segment Data is integrated as an **independent live indicator**. For
each `road_segment_id`, the service calculates the geometrical midpoint of the
stored OSM line and asks TomTom for the closest traffic-flow segment at that
WGS84 point. The response exposes `current_speed`, `free_flow_speed`,
`congestion_ratio` (`current/free-flow`), and `confidence`. The TomTom segment
can differ slightly from the OSM segment, so this is a point-near-segment
enrichment rather than an identity join.

It is intentionally not a CatBoost feature and is never written into a training
dataset: there is no historical traffic series aligned with the historical ДТП
labels. The backend method keeps the two signals separate:

```python
from ml_service import AccidentRiskPredictor

result = AccidentRiskPredictor().predict_segment_with_live_traffic(
    "2744171408_2744219355_0", "2022-09-08 15:00:00", "1h"
)
# result["predictions"][0]["risk_probability"] is frozen-model risk
# result["live_traffic"]["congestion_ratio"] is current traffic only
```

Create an untracked local `.env` from `.env.example` and set `TOMTOM_API_KEY`
(or export it in the process environment). Then run a bounded collection for
the top model-risk segments:

```powershell
$env:TOMTOM_API_KEY = "your-key"
.\.venv\Scripts\python.exe scripts\collect_tomtom_live_traffic.py --datetime 2022-09-08T15:00:00 --top-n 100 --interval-seconds 3600 --iterations 1
```

Snapshots append to `data/live_traffic/tomtom_flow_snapshots.parquet` with
`timestamp`, `road_segment_id`, WGS84 query `coordinates` (`lat,lon`), `current_speed`,
`free_flow_speed`, `congestion_ratio`, and `confidence`. Missing key, no local
road match, TomTom coverage gaps, timeouts, and API errors return
`available: false` with a reason; model inference still succeeds.

The current TomTom Freemium allowance is 2,500 free **non-tile** requests per
day. Flow Segment Data counts as one request per queried segment: 100 segments
hourly is 2,400/day; 50 segments every 15 minutes is 4,800/day and exceeds the
free cap. Watch actual dashboard usage, handle HTTP 429, and use a paid plan or
a lower frequency before enabling a continuous production collector. Traffic
coverage and confidence vary by road; do not interpret congestion as a causal
change in accident risk.

## Reproducibility runbook

Use the repository virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest tests\test_stage8b_engine.py
```

The ordinary demo path assumes that the prepared datasets, frozen models, and Stage 8A artifacts already exist. It does not retrain a model:

```powershell
.\.venv\Scripts\python.exe scripts\generate_stage8b_recommendations.py
.\.venv\Scripts\python.exe scripts\run_stage8c_inference.py
```

For explainability maintenance, the Stage 7D 1h model can refresh its Stage 8A analysis without training. The regular Stage 8A script refreshes both frozen Stage 7B models when invoked without an option:

```powershell
.\.venv\Scripts\python.exe scripts\explain_stage8a_shap.py --stage7d-1h
.\.venv\Scripts\python.exe scripts\explain_stage8a_shap.py
```

The following commands are for a full data preparation or retraining workflow and require the earlier stage inputs to be present. The final production choices remain **1h Stage 7D** and **24h Stage 7B**.

```powershell
# Rebuild chronological Stage 7A splits from the standard prepared datasets.
.\.venv\Scripts\python.exe scripts\prepare_stage7a_splits.py --label 1h
.\.venv\Scripts\python.exe scripts\prepare_stage7a_splits.py --label 24h

# Train Stage 7B candidates from the prepared splits.
.\.venv\Scripts\python.exe scripts\train_stage7b_catboost.py --label 1h
.\.venv\Scripts\python.exe scripts\train_stage7b_catboost.py --label 24h

# Run the causal weather-feature experiment used for the final 1h Stage 7D model.
.\.venv\Scripts\python.exe scripts\run_stage7d_weather_experiment.py --horizon 1h
```

`run_stage7d_weather_experiment.py --comparison-only` rewrites only the Stage 7D comparison report from existing experiment reports. All training and experiment commands may create new model or report artifacts; they are not required for the normal demo run.

## Limitations

The models are trained on available matched-accident and road data. Coverage, road metadata quality, weather source quality, and temporal drift affect predictions. Probabilities are decision-support signals, not guarantees. SHAP and the recommendation rules identify model-supported associations, not interventions proven to reduce accidents.
