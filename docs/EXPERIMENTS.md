# ML experiment trail

| Stage | Experiment | Test PR-AUC | Decision |
|---|---|---:|---|
| 7B | CatBoost 24h baseline | 0.25563 | Production 24h |
| 7D | Causal weather CatBoost 1h | 0.15801 | Production 1h |
| 10 | XGBoost/LightGBM benchmark | 0.16048 XGB (1h) | No automatic replacement |
| 11 | Graph-neighbor features | 0.15696 | Rejected |
| 12 | Spatial cleanup and retune | 0.16053 | Rejected: validation rule not met |
| 13 | Exposure, Poisson, weights, ensemble | See `reports/stage13/` | Rejected |
| 14 | Structural OSM composites | 0.15858 | Rejected: validation rule not met |

Reports are immutable research artifacts under `reports/`.
