# Models

## 1h — Stage 7D

The one-hour production model is `catboost_1h_weather_experiment.cbm`. It uses 97 ordered features and was trained as a weather-signal experiment on the Stage 7A temporal split. In addition to road, POI, calendar, weather, and historical features, it includes strictly past weather rolling windows and model interactions defined before inference.

## 24h — Stage 7B

The twenty-four-hour production model is `catboost_24h.cbm`. It uses 77 ordered features from the Stage 7A configuration. It relies on the same core feature families and includes 24-hour seasonal historical context.

## Explainability and operation

CatBoost TreeSHAP is used for global analysis and local explanation. Recommendation rules consume only positive local evidence relevant to their trigger. Stage 8C keeps full SHAP arrays in memory only long enough to select local factors; exports contain concise explanations instead.

Model probabilities should be read with calibration, data coverage, and temporal drift in mind. Neither SHAP values nor recommendations prove a causal mechanism.
