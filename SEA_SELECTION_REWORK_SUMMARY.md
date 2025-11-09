# Sea Selection Rework - Implementation Summary

## Overview
The sea rubric now evaluates candidates with daypart-aware sky normalization,
a wave score derived from sea cache meters, and staged scoring that widens
constraints without dropping the pool. The pipeline keeps evening sunsets in
scope, rewards fresh captures, and logs every stage of the decision so operators
can audit degradations.

## Key Changes

### 1. Daypart + Weather Tag Normalization
- `data_access.py` parses EXIF capture timestamps and recognition tags to assign
  `captured_at`, `doy`, and a `daypart` (`morning`, `day`, `evening`, `night`).
- `sea_selection.NormalizedSky` stores the normalized daypart and weather tag,
  mapping `sunset` / `dusk` / `twilight` / `evening` hints to
  `daypart="evening"` while deriving the sky bucket (`sunny`,
  `partly_cloudy`, etc.) from tags or live cloud cover.
- Evening assets compatible with clear daytime weather are treated as valid
  matches, allowing sunsets to participate when the forecast reports clear or
  mostly clear skies.

### 2. Wave Score Unification
- `main.wave_m_to_score` converts cached sea wave heights (in meters) to an
  integer score from 0 to 10 using 0.2 m increments and clamping at both ends.
  Example: `0.05 m → 0`, `0.45 m → 2`, `1.6 m → 8`, `≥2.0 m → 10`.
- Vision already reports `sea_wave_score` on the same 0–10 scale; the selector
  compares the target score from `sea_cache` to each asset’s score.

### 3. Soft-penalty Stage Scoring
- `sea_selection.StageConfig` describes the B0/B1/B2/AN stages with wave
  tolerance, sky requirements, and penalty weights.
- `_publish_sea` builds a wave corridor for each stage and calls
  `evaluate_stage_candidate`, which always returns a numeric score instead of
  short-circuiting on mismatches. Penalties cover wave deltas, out-of-corridor
  overshoot, calm-day caps, sky visibility, sky compatibility, and seasonal
  mismatch. Bonuses reward fresh captures, sky matches, and visible skies.
- Stages widen tolerances gradually:
  - **B0** – Strict: requires season match, visible sky, known sky bucket, and
    ±1 wave tolerance.
  - **B1** – Relaxed sky visibility requirement, ±1.8 wave tolerance.
  - **B2** – Drops season requirement, allows missing wave scores, ±2.5 wave
    tolerance.
  - **AN** – Emergency: accepts false/unknown sky, ±3.5 wave tolerance.
- If every stage produces an empty ranking, a final age-priority fallback picks
  the least recently used candidate instead of returning `None`.

### 4. Logging and Observability
- Each stage logs its corridor, pool size, and top five candidates with detailed
  score components. Pool counts accumulate across stages so operators can see
  how degradation progresses.
- Final selection logs include the chosen stage, score breakdown, target wave
  score, actual photo score, normalized sky token, and whether clear-guard rules
  detected a critical mismatch.

### 5. Caption Generation Updates
- `_generate_sea_caption` receives the normalized sky summary, wave height in
  meters, and the 0–10 score so captions render phrases such as "почти ясно" and
  "штиль" consistently with the new metrics.

### 6. Database & Freshness
- Migration `0028_assets_capture_fields.py` adds `captured_at`, `doy`, and
  `daypart` columns with supporting indexes.
- Candidate loading prioritizes captures from today and yesterday using the new
  metadata, with graceful fallback to post dates when EXIF timestamps are
  missing.

### 7. Test Coverage
- `tests/test_sea_wave_score.py` validates the meter-to-score mapping and sky
  compatibility helpers.
- Integration tests in `tests/test_sea_selection.py` verify that an evening calm
  sunset wins against stormy or stale assets and that degradation keeps returning
  the best-scoring candidate instead of collapsing to storms.
- Migration fixtures in `tests/test_vision_results.py` and
  `tests/test_data_access_capture_fields.py` exercise the new capture columns.

## Acceptance Criteria
- Evening sunsets stay eligible under clear forecasts thanks to daypart-aware
  normalization.
- Wave comparisons use the unified 0–10 scale from both sea cache and vision
  outputs.
- Stage degradation only loosens penalties; pools are never dropped entirely.
- Captions and logs surface the normalized sky, target/actual wave scores, and
  capture freshness required by the rubric.
- All lint, type, and test suites pass via GitHub Actions (`ruff`, `black`,
  `mypy`, `pytest`).
