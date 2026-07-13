# Stage 15B — gov.kz Road Events Parser Audit

## Source and safe collection

The provider uses the official Astana Akimat listing `https://www.gov.kz/memleket/entities/astana/press/news/{page}?lang={language}` and only follows official detail paths matching `/memleket/entities/astana/press/news/details/<ID>`. It fetches `https://www.gov.kz/robots.txt` before discovery, identifies itself with a descriptive User-Agent, applies a timeout, exponential retry, configurable delay, and conservative defaults of one page and ten articles.

A real dry-run requested one Russian listing page with a two-article maximum. `robots.txt` was fetched and the provider did not find a matching disallow rule. The listing returned a JavaScript application shell to the non-browser HTTP client and contained zero server-rendered detail links. Therefore it returned **degraded**, `fallback_used=true`, and zero records. This is deliberate: it did not use a private endpoint, authentication bypass, CAPTCHA workaround, or invented event data.

## Parser contract

The parser retains title, cleaned article text, published date, event dates, original location phrases, classification evidence, relevance score, warnings and content hash in the universal record payload. Dates are timezone-aware (`Asia/Almaty`). Unknown end dates remain null with `open_end=true`. Latitude, longitude, geometry and road segment IDs are intentionally null until a separate geocoding/road matching stage.

Relevance requires repair/restriction evidence plus infrastructure/location context, rather than a single generic word. It supports Russian and Kazakh keyword groups. Classifications are `road_repair`, `road_construction`, `road_reconstruction`, `road_closure`, `lane_closure`, `bridge_repair`, `intersection_closure`, `traffic_restriction`, and `unknown_road_event`; restrictions and operational severity follow the documented transparent rules in the provider.

## Storage and features

Records are upserted by `(source, source_item_id)` and a content hash over normalized title, description, dates, event/restriction classes and locations. A mock storage run produced `new=1`; an identical rerun produced `unchanged=1`. Outputs are `gov_kz_road_events.parquet`, `gov_kz_road_events.json` and `gov_kz_repair_features.parquet`.

Candidate features are city-level only: `repair_events_next_24h`, full/partial/lane/intersection/bridge counts, high-severity count, active/open-end counts and `repair_disruption_score`. They are not CatBoost features. Future `repair_active_on_segment`, distance and nearby-segment features require validated coordinates and explicit road matching.

## Validation

25 unit tests pass, including listing parsing, dates, timezone, locations, repair/closure/lane/intersection/bridge cases, irrelevant article rejection, Kazakh keywords, update detection, universal schema validation, idempotency and frozen-model regression. Stage 8C 24h checksum, model hash, 77-feature order and OpenWeather registry entry are unchanged.
