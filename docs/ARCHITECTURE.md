# Architecture

```mermaid
flowchart TD
    A[Raw accident and road data] --> B[Validation and cleaning]
    B --> C[Road, POI, calendar, weather and history features]
    C --> D[Temporal train / validation / test splits]
    D --> E[CatBoost 1h Stage 7D]
    D --> F[CatBoost 24h Stage 7B]
    E --> G[Local TreeSHAP]
    F --> G
    G --> H[Stage 8B rule-based recommendations]
    H --> I[Stage 8C batch inference]
    I --> J[JSON API payload]
    I --> K[GeoJSON risk map]
```

The inference path receives a prediction hour, builds only information available at that hour, and keeps explanations local to each road segment. Recommendations remain separate from the models and always require human review.
