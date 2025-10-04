# Telegram Scheduler Bot

## Summary
[Полную историю изменений см. в `CHANGELOG.md`](CHANGELOG.md).

- **Asset ingestion**. The bot listens to the configured assets channel, stores each message as an asset record and downloads the original media to local storage. Every photo must contain GPS EXIF data so the bot can resolve the city through Nominatim; authors without coordinates receive an automatic reminder.
- **Recognition pipeline**. After ingestion the asynchronous job queue schedules a `vision` task that classifies the photo with OpenAI `gpt-4o-mini`, storing the rubric category, architectural details and detected flowers while respecting per-model daily token quotas configured via environment variables.
- **Rubric automation**. Two daily rubrics are supported out of the box: `flowers` creates a carousel with greetings for cities detected in flower assets, while `guess_arch` prepares a numbered architecture quiz with optional overlays and weather context. Both rubrics consume recognized assets and clean them up after publishing.
- **Admin workflow**. Superadmins manage user access, asset channel binding and rubric schedules directly inside Telegram via commands and inline buttons. The admin interface also exposes manual approval queues and quick status messages for rubric runs.
- **Operations guardrails**. OpenAI usage is rate-limited per model, reverse geocoding calls Nominatim with throttling, and each rubric publication is persisted with metadata for auditing through the admin tools.

## Commands
### Access & governance
- `/start` – registers the requester and assigns the first superadmin on first launch.
- `/pending`, `/approve <id>`, `/reject <id>` – manage the onboarding queue from the Telegram admin interface.
- `/add_user <id>`, `/remove_user <id>`, `/list_users` – grant or revoke long-term access to the scheduler and rubric tools.

### Asset management
- `/set_assets_channel` – bind the private storage channel used for ASSETS ingestion; only posts created after this command are captured.
- `/history` and `/scheduled` – inspect previously published posts and queued schedules, including rubric drops copied from the assets channel.

### Rubrics & quotas
- `/list_weather_channels` – repurposed admin dashboard that now shows rubric schedules, remaining OpenAI daily quota and lets admins toggle individual runs (messages labelled accordingly).
- `/weather now` – forces a refresh of cached weather that is embedded into the `guess_arch` rubric intro; other weather commands remain available under Legacy behavior.
- Inline buttons labelled «Run now» next to rubric schedules enqueue an immediate `publish_rubric` job using the background queue while preserving the regular cadence.

## User Stories
### Implemented
- **US-1**: As a photographer I can drop photos into the ASSETS channel and the bot ingests them with EXIF validation, notifying me if coordinates are missing so the rubric team always knows the shooting location.
- **US-2**: As a curator I can rely on automatic recognition to pre-fill categories, architecture notes and flower types for every asset, reducing manual tagging before rubric publication.
- **US-3**: As a rubric editor I can configure daily schedules for `flowers` and `guess_arch`, ensure enough classified assets are available and publish them automatically or on demand from Telegram.
- **US-4**: As an administrator I can monitor OpenAI token consumption, review rubric history and audit which assets were consumed, all through the bot’s inline admin interface.
- **US-5**: As an operations engineer I have a persistent job queue with retry/backoff semantics and manual controls so rubric jobs can be re-run safely without duplicate posts.

### Planned
- Automatic triage for non-photo assets and support for additional rubrics once the asset taxonomy is expanded.

## Environment Setup
### Required environment variables
- `TELEGRAM_BOT_TOKEN` – Telegram bot API token used for both webhook registration and admin interactions.
- `WEBHOOK_URL` – public HTTPS base used to register `/webhook` on startup.
- `DB_PATH` – path to the SQLite database (default `/data/bot.db`).
- `ASSET_STORAGE_DIR` – optional override for local media storage. Defaults to `/tmp/bot_assets`; ensure the directory persists across deployments if you expect re-ingestion safeguards.
- `TZ_OFFSET` – default timezone applied to schedules until users pick their own offset.
- `SCHED_INTERVAL_SEC` – polling cadence for the scheduler loop (default `30`).
- `OPENAI_API_KEY` – key used by the recognition pipeline and rubric copy generators; when missing, related jobs are skipped automatically.
- `OPENAI_DAILY_TOKEN_LIMIT`, `OPENAI_DAILY_TOKEN_LIMIT_4O`, `OPENAI_DAILY_TOKEN_LIMIT_4O_MINI` – optional per-model quotas that gate new OpenAI jobs until the next UTC reset.

### External services
- **Nominatim** – the bot queries `https://nominatim.openstreetmap.org/reverse` and rate-limits calls to one request per second. Set `User-Agent` friendly values in the code if you fork, and consider running your own Nominatim instance for higher throughput.
- **OpenAI Responses API** – outbound requests target the `/responses` endpoint; ensure outbound egress is permitted from your hosting environment.

### Local assets & overlays
- Store overlay PNGs for the `guess_arch` rubric inside the directory referenced by the rubric config (default `main.py` → `overlays`). Files named `1.png`, `2.png`, etc., are overlaid on published photos; the bot auto-generates numeric badges when files are missing.
- Persist asset storage on a volume so ingestion and recognition jobs can retry without re-downloading media. Temporary directories will break overlay generation and duplicate detection after restarts.

## Job Queue and Manual Rubrics
- The SQLite-backed job queue starts automatically when the bot boots, loads due jobs every second and executes handlers concurrently according to `JobQueue(concurrency=...)` settings. Failed jobs retry with exponential backoff up to five attempts before being marked as `failed`. Inspect `jobs_queue` via any SQLite tool for troubleshooting.
- To enqueue a rubric manually without waiting for the scheduler, run the following snippet with valid environment variables:
  ```bash
  python - <<'PY'
  import asyncio, os
  from main import Bot

  async def trigger():
      bot = Bot(os.environ["TELEGRAM_BOT_TOKEN"], os.environ.get("DB_PATH", "/data/bot.db"))
      await bot.start()
      # replace channel id with the target Telegram channel id
      await bot.enqueue_rubric("flowers", channel_id=-1001234567890)
      await bot.close()

  asyncio.run(trigger())
  PY
  ```
  Replace `flowers` with `guess_arch` as needed. The job queue deduplicates identical pending payloads, so repeated runs are safe.
- For ad-hoc publication bypassing the queue entirely, call `publish_rubric` inside the same context; it returns `True` on success and records the run in `posts_history` for later review.

## Deployment
The bot targets Fly.io and exposes a single aiohttp application on port `8080`. Ensure your Fly.io service terminates TLS on port `443` and forwards traffic to the container port so Telegram can reach the webhook. The provided `fly.toml` already contains the required configuration.

### Local run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the bot:
   ```bash
   python main.py
   ```

Provision Fly.io secrets (`TELEGRAM_BOT_TOKEN`, `WEBHOOK_URL`, `OPENAI_API_KEY`, token limits) before the first deployment.

## Legacy
Historical documentation for the weather scheduler, including sea temperature handling and template placeholders, now lives in `docs/weather.md`. Those features remain in the codebase for backward compatibility but are no longer part of the primary rubric workflow.
