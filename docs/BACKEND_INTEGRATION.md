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

For optional live traffic, call `predict_segment_with_live_traffic(...)`. Its
`live_traffic` object is intentionally a sibling of `predictions`, never a
model feature: `risk_probability` and `congestion_ratio` must be displayed as
separate indicators. If `TOMTOM_API_KEY` is absent or TomTom does not return a
flow segment, this object has `available: false` and model output remains valid.

Invalid inputs raise library exceptions: `InvalidHorizonError`, `InvalidDatetimeError`, `InvalidBBoxError`, `UnknownRoadSegmentError`, `EmptySegmentListError`, `RegistryNotFoundError`, `ConfigNotFoundError`, or `ModelNotFoundError`. The library loads final models, feature configs, and thresholds in the constructor and caches each requested city hour/horizon for the lifetime of the predictor instance.
