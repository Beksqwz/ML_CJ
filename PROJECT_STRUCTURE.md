# Project structure

`config/` holds versioned runtime configuration. `data/` contains external inputs and processed datasets. `docs/` carries human-facing technical documentation. `inference/` exposes the production prediction path. `models/` stores frozen CatBoost artifacts. `recommendations/` contains the explainable rule layer. `reports/` preserves stage evidence and generated operational outputs. `scripts/` keeps reproducible stage entry points. `tests/` covers recommendation behaviour.

Large source files at the repository root are geographic and road-network inputs retained for reproducibility. They are not rewritten by the production inference service.
