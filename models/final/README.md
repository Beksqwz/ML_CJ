# Deployed final model: Stage19I ensemble

This directory is the only model bundle used by `api_service.app` for live
24-hour predictions.

- `stage19i_catboost.cbm` — CatBoost component.
- `stage19i_hist_gradient_boosting.joblib` — HGB component.
- `stage19i_preprocessor.joblib` — frozen HGB preprocessor.
- `ensemble_config.json` — frozen percentile-rank weights: CatBoost 0.8, HGB 0.2.

The `models/stage19h/` and `models/stage19i_simple/` directories remain
research/training provenance. `models/legacy_registry/` supports the older
`AccidentRiskPredictor` library path and is not the live API model.
