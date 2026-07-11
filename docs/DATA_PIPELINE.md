# Data pipeline

Stages 1–6 ingest accident records, validate source structure, clean coordinates and timestamps, match events to road segments, normalize road attributes, derive POI features, and join hourly calendar and weather context.

Historical counters use only accidents strictly earlier than `datetime_hour`. Weather rolling windows for the 1h experiment are shifted so they do not include later observations. Target fields, technical identifiers, and post-event attributes are excluded from model features.

Stage 7A creates chronological train, validation, and test partitions with a horizon-specific purge at boundaries. Stage 7B trains the 24h model and Stage 7D evaluates the causal weather-feature experiment for 1h. Later stages explain frozen models, generate human-reviewed recommendations, and provide batch inference exports.
