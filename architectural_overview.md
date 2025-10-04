# Architectural Overview

This document outlines the components that power the asset-driven rubric pipeline.

## 1. Introduction
The bot ingests posts from a dedicated Telegram channel, enriches them with EXIF metadata and OpenAI recognition results, and schedules rubric publications to downstream channels. A SQLite database stores all state, and an internal job queue coordinates long-running work.

## 2. Database
SQLite tables track users, channels, assets, token usage, job queue entries and rubric configuration. Asset rows retain EXIF-derived coordinates, reverse-geocoded location names, and the recognition payload so rubrics can filter by category. Token consumption is persisted per OpenAI model to enforce daily limits.

## 3. Services
### 3.1 Bot
The aiohttp bot handles Telegram webhooks, user commands and inline callbacks. It also manages the background scheduler loop that processes due jobs and rubric schedules.

### 3.2 Asset ingest
Incoming photos from the assets channel trigger an ingest job. The job downloads the original media, validates EXIF GPS coordinates, stores the file locally, reverse-geocodes coordinates via Nominatim, and updates the asset record with derived metadata.

### 3.3 Recognition
A follow-up `vision` job classifies each asset with OpenAI (`gpt-4o-mini`) to extract a primary category, architecture view, flower varieties and weather description. The job respects model quotas and logs token usage for auditing.

### 3.4 Rubric engine
Configured rubrics (`flowers`, `guess_arch`) pull classified assets, render text using OpenAI where required, overlay numbers on quiz photos, and publish carousels back to Telegram channels. After a successful run the consumed assets and temporary overlays are removed.

### 3.5 Admin interface
Superadmins manage onboarding, asset channel binding, rubric schedules and manual triggers directly from Telegram commands and inline buttons. The interface exposes quick actions such as «Run now» and displays remaining OpenAI quota.

## 4. Job Queue
`jobs.py` implements a persistent queue with exponential backoff, deduplication and concurrency control. It loads due jobs every second, dispatches handlers (`ingest`, `vision`, `publish_rubric`) and records failure history for support investigations.

## 5. User Stories
- **US-1**: Asset ingestion validates EXIF metadata, reverse-geocodes coordinates and notifies authors when data is missing.
- **US-2**: Recognition jobs enrich assets with rubric categories, architecture details and flower varieties using OpenAI.
- **US-3**: The rubric engine automatically assembles and publishes `flowers` and `guess_arch` drops when schedules mature.
- **US-4**: Administrators can audit OpenAI usage, enqueue manual runs and monitor rubric status within Telegram.
- **US-5**: Operators rely on the durable job queue to recover from transient failures without losing or duplicating posts.

## 6. Legacy
Weather-specific schedulers, sea temperature placeholders and related commands remain available for historic channels. Their behaviour is unchanged but they are no longer part of the primary rubric workflow; see `docs/weather.md` for details.
