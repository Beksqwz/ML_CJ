# Future Intelligence Layer

`future_intelligence` is a separate future-context layer for the frozen 24-hour Stage 7B model. It never alters the CatBoost feature matrix, model registry, SHAP calculation, recommendation rules, or legacy Stage 8C outputs. Its namespaced fields are candidates for a future retraining project only.

## Architecture

- `providers/base.py` defines the collect/normalize/build_features/healthcheck contract.
- `providers/weather/openweather.py` is the first implementation and composes the existing `ml_service.weather.OpenWeatherService` for its API-key, `.env`, session, and timeout behavior.
- `registry.py` maps provider names to factories; adding Open-Meteo, TomTom, gov.kz repairs, CTS, or Ticketon requires only a provider implementation and one registration.
- `pipeline.py` orchestrates independently normalized provider results and detects feature-name collisions.
- `schemas.py` provides the universal record envelope; `storage.py` writes safe raw, normalized, provider and universal feature outputs.

## OpenWeather configuration and endpoint

Copy `.env.example` to `.env` and set only `OPENWEATHER_API_KEY`. The provider calls OpenWeather's official 5 day / 3 hour forecast endpoint: `https://api.openweathermap.org/data/2.5/forecast`. No key is written to reports, raw files, or logs.

Example:

```powershell
python scripts/collect_future_intelligence.py --providers openweather --prediction-datetime "2026-07-14T00:00:00+05:00" --horizon-hours 24
```

Use `--dry-run` to avoid writes and `--strict` to make a degraded provider status fail the command.

## Data lifecycle and fallback

Provider raw responses are stored below `data/future_intelligence/raw/<provider>/` with credential fields removed. Normalized OpenWeather points and city-level 24-hour features are stored in Parquet beneath `data/future_intelligence/processed/`. The universal feature table includes `prediction_datetime`, a nullable `road_segment_id` (city-wide values may later be broadcast), provider name, and namespaced feature columns.

Only forecast points in `[prediction_datetime, prediction_datetime + 24h)` are used. On a missing key, timeout, rate limit, malformed response, or unsupported horizon, the provider returns `status="degraded"`, `fallback_used=true`, and warnings. It returns no fabricated forecast features and never exposes raw network exceptions to callers.

## Feature definitions

All OpenWeather outputs start with `weather_`. Mean/min/max/std are computed over included 3-hour forecast points. Rain and snow are sums of supplied forecast precipitation; `*_hours` count qualifying 3-hour points times three. Fog is visibility under 1,000 m, heavy rain is at least 2.5 mm/point, strong wind is at least 10 m/s, freezing is at most 0 C, and storm IDs use OpenWeather 2xx thunderstorm codes. `weather_change_count` counts adjacent weather-code changes; instability combines that count with a >=5 C adjacent temperature drop and any freeze/thaw crossing. Surface, visibility, winter, driving-condition, and severity scores are transparent sums of the documented boolean conditions.

Sunrise/sunset fields are converted with the API's timezone offset. Missing optional API fields remain null (or zero only for a documented absent precipitation amount); no historical observation is substituted.

## Future extensions

Open-Meteo belongs in `providers/weather/`; TomTom in `providers/traffic/`; gov.kz repairs in `providers/repairs/`; CTS in `providers/transit/`; Ticketon, stadium, and other public events in `providers/events/`. Implement the base contract, return a `ProviderResult`, namespace features, and register a factory. None of these providers may become a frozen model input without a separately versioned retraining and production-approval process.

## gov.kz Astana road events

`gov_kz_repairs` is a public, API-key-free provider registered beside OpenWeather. Its default `official-filtered` discovery renders the official Astana Akimat news listing at `https://www.gov.kz/memleket/entities/astana/press/news/{page}?lang={language}`, extracts visible card title/snippet/date, and prefilters likely repair or restriction cards before rendering any detail page. It follows only official article links matching `/memleket/entities/astana/press/news/details/<ID>`. It checks `https://www.gov.kz/robots.txt`, sends a descriptive User-Agent, uses a timeout, retry backoff, a delay between requests, and defaults to three pages and ten articles.

```powershell
python scripts/collect_gov_kz_road_events.py --discovery-method official-filtered --max-pages 3 --max-articles 20 --language ru --dry-run
```

The parser uses Russian/Kazakh repair, restriction, and infrastructure keyword groups; a transparent relevance score requires more than a single weak generic term. It extracts publication/event dates, location phrases, from/to sections, intersections, event/restriction class, severity, confidence, and a content hash. Unknown end dates stay null with `open_end=true`; no coordinates or road segments are fabricated.

Canonical records are upserted by `(source, source_item_id)` and `content_hash` into `gov_kz_road_events.parquet`, with a JSON export and `gov_kz_repair_features.parquet`. Current candidate features are city-level counts and `repair_disruption_score`; road-level distance and segment flags remain reserved for a separate geocoding/road-matching stage.

The gov.kz public listing serves a JavaScript application to the conservative HTTP client. `official-filtered` therefore uses one reusable headless Playwright browser/context to render listing cards and only candidate details. It first tries the conservative detail request, detects a JavaScript shell, then passes `page.content()` from the same browser session into the existing parser. If rendered cards or a candidate detail cannot be obtained, it returns explicit degraded status; it does not use undocumented endpoints, CAPTCHA solving, proxy rotation, or fabricated records. `road-search` remains optional and is degraded if a search engine blocks automation.

Auto discovery is ordered: `official-filtered`, verified public JSON/XHR endpoint, official sitemap/feed, optional road search, unfiltered Playwright listing, then legacy server-rendered HTML. JSON is deliberately skipped until it is verified from official browser network activity. Playwright is an optional runtime dependency; install the package from `requirements.txt` and run `playwright install chromium`. It is not required for fixture parsing or any frozen inference path.

Generated raw/processed collection outputs remain local runtime data. Sanitized fixture HTML under `tests/fixtures/` is appropriate for source control; large rendered debug HTML and screenshots are intentionally ignored. Current limitations are missing road coordinates and segment matching. A later stage may add geocoding and road matching, then CTS and Ticketon providers, without changing the frozen Stage 7B input.
