# Stage 19D evaluation audit

Verified Stage 7B test metrics reproduce exactly on the saved **sampled 1:5** test set. The 24-hour target is `(t, t+24h]`; global chronological splits have a 24-hour purge. No natural-prevalence all-segment historical evaluation artifact exists, and no sampling-prior probability correction or external calibration exists.

## Required before production probability claims

Build a leakage-safe all-segment historical evaluation grid and calibrate/select operating thresholds on natural validation prevalence.
