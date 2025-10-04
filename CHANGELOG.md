# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `/help` command that delivers a Markdown help guide with links to admin-interface workflows and manual button instructions.

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
