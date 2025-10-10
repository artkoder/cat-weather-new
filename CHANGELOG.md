# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Ingest now preserves EXIF capture timestamps, season detection prefers those capture times when available, and operator
  captions surface the photo's capture date and time for easier context.
- Operator style brief that puts a stronger emphasis on a lively, conversational tone and captures the explicit prompting
  instructions we now provide for the `gpt-4o` writers.
- YAML-driven pattern and ban list loader for the `flowers` generator, enabling operators to expand copy templates without
  touching the codebase.
- Flowers rubric preview workflow that lets operators review candidates, regenerate media or captions, capture manual
  instructions, and choose the final publication destinations, backed by automated tests covering regeneration, instruction
  capture, and delivery flows.
- Supabase integration that mirrors OpenAI token usage into `token_usage` via `SUPABASE_URL`/`SUPABASE_KEY`, including strict JSON serialization of metadata and structured `log_token_usage` log lines mirroring the payload when inserts succeed or fail.
- `/help` command that delivers a Markdown help guide with links to admin-interface workflows and manual button instructions.
- Dual asset channel support with new `/set_weather_assets_channel` and `/set_recognition_channel` commands plus migration `0014_split_asset_channels.sql` that creates the recognition table and marks asset origins.
- Regression coverage that simulates both channels to ensure weather posts ignore recognition-only assets.
- Vision schema enhancements that capture framing, detailed weather, seasonal context and architectural style metadata, and an automatic conversion path that renders document uploads into photo assets before publication.

### Changed
- Flowers rubric prompts now weave daypart weather context into the greeting, retain per-photo descriptions, and spell out the
  latest `gpt-4o` response rules, with regression coverage in `tests/test_rubrics.py` and `tests/test_vision_results.py`.
- Weather automation replaces raw `weather_code` values with localized text conditions before publication so operators no
  longer need to translate numeric codes manually.
- Expanded the stop-word list for caption linting to catch additional filler phrases during operator review.
- Flowers caption generator now tracks recently used templates to avoid repeating the same copy across consecutive posts.
- Updated the OpenAI Responses payload to use the latest `response_format` schema and
  ensured token usage metrics persist to Supabase with non-null totals.
- OpenAI payloads now run through `strictify_schema`, enforcing required keys and null-tolerant
  types to eliminate `invalid_json_schema` errors while normalizing nested objects and arrays so
  operators can see why retries no longer fail with missing-property complaints.
- `/set_assets_channel` now updates both channel roles for backward compatibility, while `publish_weather` only copies from the weather storage channel and leaves source messages untouched.
- `guess_arch` overlays now scale to 10–16% of the shortest image side so custom PNG badges stay proportional across photo sizes.

- Telegram file downloads now stream directly to disk during ingest and vision jobs, reducing memory usage and ensuring GPS extraction reads from the stored files instead of in-memory buffers.

## [1.3.0] - 2024-05-17
### Added
- Automatic recognition pipeline that classifies ingested assets with OpenAI `gpt-4o-mini`, stores architectural metadata and detects flowers for downstream rubrics.
- Persistent asynchronous job queue that schedules recognition, rubric publication and manual overrides with retry/backoff semantics.
- Daily rubrics «Цветы» and «Угадай» that assemble carousels and quizzes from recognized assets and clean up consumed media.
- Token accounting with per-model daily quotas to prevent OpenAI overages and surface usage to administrators.
- New database migrations (`0008_vision_enhancements.sql`, `0009_token_usage.sql`, `0012_core_schema.sql`) required to support recognition storage, queue persistence, rubric history and token usage tracking; apply them before deploying this release.

## [1.2.0] - 2024-04-02
### Added
- Initial public release with Telegram ingestion, weather automation and manual rubric tooling.
