# Backend integration

Final production models are **1h → Stage 7D** and **24h → Stage 7B**. Backend
callers do not need to import CatBoost or stage-specific modules.

```python
from ml_service import AccidentRiskPredictor

predictor = AccidentRiskPredictor()
city = predictor.predict_city("2022-09-08 15:00:00", "1h")
segment = predictor.predict_segment("2744171408_2744219355_0", "2022-09-08 15:00:00", "24h")
bbox = predictor.predict_bbox(71.0, 51.0, 72.0, 52.0, "2022-09-08 15:00:00", "1h")
```

`predict_city(datetime_hour, horizon)` returns every known segment. `predict_segment(road_segment_id, datetime_hour, horizon)` returns one segment. `predict_segments(segment_ids, datetime_hour, horizon)` serves an explicit list, and `predict_bbox(min_lon, min_lat, max_lon, max_lat, datetime_hour, horizon)` filters by segment representative point. Horizons are `"1h"` and `"24h"`.

Each response is a plain Python dictionary with `datetime_hour`, `model_horizon`, `predictions`, `geojson`, and `summary`. Prediction records contain `risk_probability`, `risk_level`, local positive and negative factors, and human-review recommendations. `get_model_info()` exposes the final model stages, feature counts, and operational display thresholds. `healthcheck()` returns `status`, model stages, model versions, and feature counts after models load.

Invalid inputs raise library exceptions: `InvalidHorizonError`, `InvalidDatetimeError`, `InvalidBBoxError`, `UnknownRoadSegmentError`, `EmptySegmentListError`, `RegistryNotFoundError`, `ConfigNotFoundError`, or `ModelNotFoundError`. The library loads final models, feature configs, and thresholds in the constructor and caches each requested city hour/horizon for the lifetime of the predictor instance.

## HTTP backend synchronization contract

`POST /api/v1/predict` supports a dedicated `backend_sync` response mode. It is
for the RRAI Worker only; existing `compact` and `full` callers retain their
previous responses.

```json
{
  "force": false,
  "response_mode": "backend_sync",
  "road_segment_ids": ["250783683_5078375733_0"]
}
```

`road_segment_ids` is required, accepts at most 6,000 non-blank IDs, and is
deduplicated. The service still calculates the city prediction once, then
returns only the requested known segments. The response contains
`contractVersion: "2"`, `modelVersionId`, `modelHorizon: "24h"`, `generatedAt`,
and `predictions`.

Each prediction uses `risk_score` for the Stage19I ranking score. It is not an
accident probability. `operational_priority` is mapped to the backend enum
`LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`; `monitor_only` maps to `LOW`.
Uncalibrated uncertainty labels are emitted as `null`, and `confidence` is
always `null` until ML provides a documented confidence metric.

Each prediction also contains `future_context`. It holds multilingual, display-ready
weather, traffic, repair and event signals for the next 24 hours, their provider
availability and warnings. This is operational context: it must be displayed in a
separate UI section and does **not** change the frozen ML score or SHAP explanation.
Factors additionally retain a multilingual `display_name`; clients must display it
instead of the technical `feature` key.
