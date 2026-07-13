# Stage 15.0 Production Audit

**Audit mode:** read-only. No production logic, feature engineering, inference, models, or registry entries were changed. No Git commit was created.

## Production status

**PRODUCTION STATUS: READY FOR OPENWEATHER INTEGRATION**

The current registry is internally consistent and loads exactly two production models: Stage 7D (1h, 97 features) and Stage 7B (24h, 77 features). In particular, the required 24h Stage 7B model is loaded from `models/production/catboost_24h.cbm` and matches the registry feature count.

## Passed checks

- Required production folders, Stage 8C and Stage 10–14 report folders exist; no required folder is missing.
- Model paths exist, CatBoost models load, and every model feature list exactly matches its feature-builder configuration and registry count.
- Historical feature build at `2022-09-08 15:00:00` produced 3,968 rows for both horizons. There are no missing required columns, duplicate columns, entirely-null features, or feature-order mismatches.
- Stage 8C completed successfully in 12.14 s. It generated valid JSON and GeoJSON for 3,968 segments per horizon. All probabilities were in range and all risk levels matched `config/risk_thresholds.json`.
- Recommendation rules load and execute; structural-rule tests pass.
- SHAP dimensions are valid: 1h `3968 x 98` (97 features plus expected value), 24h `3968 x 78` (77 plus expected value). All tested SHAP and expected values are finite.
- Healthcheck returned `ok`, with both registered models loaded and the expected feature counts.
- Core calendar, weather, ML-ready, POI and road datasets have no missing timestamps/road IDs where applicable, no duplicate columns, and no broken ML-ready-to-road references.
- All 10 unit tests passed; no failures or skips. `pip check` reported no broken requirements and runtime imports/compile checks passed.

## Performance

| Metric | 1h | 24h |
| --- | ---: | ---: |
| Model load | 0.535 s | 0.501 s |
| Stage 8C horizon runtime | 6.002 s | 5.063 s |
| Model size | 267,584 B | 113,272 B |
| JSON output | 11,379,564 B | 9,585,973 B |
| GeoJSON output | 10,567,811 B | 8,771,441 B |

Feature-build peak Python allocations were 41.9 MB (1h) and 40.6 MB (24h). A separate complete Stage 8C run peaked at 114.1 MB of Python allocations. These are Python allocation measurements, not a process RSS limit.

## Warnings and recommendations before the next integration

1. The working tree already contains uncommitted live weather/traffic changes. Establish a reviewed baseline before further integration work; this audit did not alter them.
2. Install the existing linting tool (`ruff`) in the execution environment before release if unused-import enforcement is required. The package is not currently installed, so that static check was not performed.
3. Investigate the small number of duplicate full rows in historical Stage 7A/7D training artifacts before any future retraining. They do not affect frozen production inference.
4. Continue preserving the feature lists and model hashes in `production_fingerprint.json`. Any OpenWeather Forecast integration should add an explicit compatibility check against those fingerprints before calling CatBoost.

The machine-readable results are in `production_audit.json`; the freeze baseline, ordered feature names, and model hashes are in `production_fingerprint.json`.
