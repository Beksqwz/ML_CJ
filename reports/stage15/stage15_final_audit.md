# Stage 15 final pre-commit audit

Status: ready.

The Stage 15 Future Intelligence layer remains separate from the frozen Stage 7B CatBoost input. `py_compile`, Compileall, Ruff formatting, Ruff linting, and all 36 unit tests passed (1.259 s wall time). Pyright/Pylance CLI is not installed; Python parsing, import validation and Ruff provided the structural validation.

The frozen 24h model SHA-256 is `0c8e1b88b1cfaf95fb39e395e2fdc54f1b7abda22d8ac00e1d6f561ab9110a0c`; its 77 ordered features hash to `bf0ce06aface9fc3539a95df2557728e57d987a9e0c0e5ac4a195a53b47f6d96`. The frozen legacy probability checksum remains `29c7b5af0e44cd4a0749947b9efec62f8d40375598d02bfedacd98823f4f7f68`.

OpenWeather mock tests passed, including missing-key degraded fallback and exact 24-hour window handling. Its deterministic aggregation produces 59 namespaced candidate features, which remain separate from the 77 frozen model features. No real OpenWeather call was made because no API key is configured.

The conservative gov.kz dry-run rendered one official listing page: 10 cards, one selected candidate and one rendered detail. The request shell had 181 extracted characters; the Playwright-rendered detail had 2,152 characters and relevance score 5. The validated five-page collection found 50 cards, selected four candidates, stored four relevant official events, achieved prefilter precision 1.0, and has no duplicate canonical IDs. A force-refresh repeat reported `new=0`, `updated=0`, `unchanged=4`.

Audit-only corrections made during this pass: mechanical Ruff formatting, removal of two unused storage locals, explicit optional Playwright pin, documentation refresh, and ignore rules for generated Future Intelligence output and debug artifacts. No model, feature order, registry entry, SHAP behavior, recommendation rule, or production inference probability was changed.

Generated `data/future_intelligence` output and `reports/stage15/gov_kz_debug/` are ignored. Commit only sanitized source, fixtures, tests, documentation and selected compact reports.
