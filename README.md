# Telegram Scheduler Bot

## Summary
[Полную историю изменений см. в `CHANGELOG.md`](CHANGELOG.md).

- [API reference](api/docs/index.html)

Closes #123.

## API Contract

The backend consumes the public API contract via the `api/contract` git submodule pinned to the release tag `v1.0.0`.

### Bumping the contract version

1. Sync submodules locally: `git submodule update --init --recursive`.
2. Enter the contract directory: `cd api/contract`.
3. Fetch upstream tags: `git fetch --tags`.
4. Checkout the new release tag (for example `git checkout v1.1.0`).
5. Return to the repository root: `cd ../..`.
6. Stage the updated submodule pointer: `git add api/contract`.
7. Update any documentation or CI checks that reference the previous tag.
8. Commit the change with a message that includes the new contract version.

- **Asset ingestion**. The bot listens to the dedicated recognition channel for new submissions while weather-ready assets live in a separate storage channel. Every photo must contain GPS EXIF data so the bot can resolve the city through Nominatim; authors without coordinates receive an automatic reminder. The pipeline also persists the original EXIF capture timestamp, reuses it when inferring seasonal context and surfaces the recorded date inside the operator info block.
- **Recognition pipeline**. After ingestion the asynchronous job queue schedules a `vision` task that classifies the photo with OpenAI `gpt-4o-mini`, storing the rubric category alongside architectural style, framing notes, seasonal context, detailed weather and detected flowers while respecting per-model daily token quotas configured via environment variables. Before each OpenAI call the bot strictifies the JSON schema—enforcing `required` lists, propagating `null`-permitted types and setting `additionalProperties: false`—so `/v1/responses` stays happy when `strict: true` is enabled. Operators should expect nullable values in payload fields that were previously plain primitives. Document uploads are automatically rendered into photo assets before they reach the publishing queue.
- **Rubric automation**. Two daily rubrics are supported out of the box: `flowers` builds its palette strictly from vision outputs, threads daypart weather context into the greeting, preserves per-photo descriptions, injects weather as numeric metrics plus plain-language conditions (no raw provider codes) and spells out the latest `gpt-4o` prompting rules directly in the request payload; `guess_arch` prepares a numbered architecture quiz with optional overlays and weather context. Both rubrics consume recognized assets and clean them up after publishing, auto-initialize on first run and are fully managed through the `/rubrics` inline dashboard.
- **Admin workflow**. Superadmins manage user access, asset channel binding and rubric schedules directly inside Telegram via commands and inline buttons. The admin interface also exposes manual approval queues and quick status messages for rubric runs.
- **Operations guardrails**. OpenAI usage is rate-limited per model, reverse geocoding calls Nominatim with throttling, and each rubric publication is persisted with metadata for auditing through the admin tools.

## Operations

### Healthcheck

- `GET /v1/health` returns the service status for Fly.io/Kubernetes probes, uptime monitoring and e2e tests.
- A `200 OK` response means the SQLite connection, job queue metrics and Telegram `getMe` call all succeeded.
- A `207 Multi-Status` response indicates the bot is running in dry-run mode (`TELEGRAM_BOT_TOKEN=dummy`); Telegram connectivity is skipped but database and queue checks must still pass.
- A `503 Service Unavailable` response surfaces failures from any mandatory dependency and includes a short `error` string in the corresponding `checks` entry.
- The payload includes the resolved `version` (from `APP_VERSION` or `CHANGELOG.md`), UTC timestamp (`now`), process uptime (`uptime_s`), per-check latencies and queue counters (`pending`, `active`, `failed`).
- Each call emits a `HEALTH ... status=...` log line summarizing the probe so operators can trace latency regressions or dependency failures quickly.

### Storage schema

- **devices** — stores hardware devices linked to Telegram users. Tracks creation time, optional last_seen timestamp and revocation markers.
- **pairing_tokens** — short-lived attach codes that pair a device with a user-defined name. Tokens expire after 10 minutes and are burned on first use.
- **nonces** — per-device anti-replay tokens with a 10-minute TTL. Old entries are automatically removed when they expire.
- **uploads** — device upload ledger keyed by `(device_id, idempotency_key)` to enforce idempotency. Status transitions follow `queued → processing → done|failed`; failed rows keep the last error and completed entries remain for at most 24 hours to prevent duplicate retries.

The job queue starts a periodic cleanup task that runs every five minutes. It purges expired pairing tokens and nonces and removes upload rows whose idempotency window (24 hours) has elapsed, keeping the database lean without manual intervention.

### Running migrations

Apply schema migrations whenever the service boots or before running tests:

```bash
python - <<'PY'
import sqlite3
from main import apply_migrations

conn = sqlite3.connect("/data/bot.db")
conn.row_factory = sqlite3.Row
apply_migrations(conn)
conn.commit()
conn.close()
PY
```

Replace `/data/bot.db` with the desired SQLite path (for local development you can keep it under `./data`). The same helper executes SQL and Python-based migrations in order and records their application in `schema_migrations`.

## Operator Interface

### Access & governance
- `/start` регистрирует первого супер-админа и показывает статус повторным пользователям.
- `/pending` открывает очередь заявок с кнопками `Approve`/`Reject`; ручное подтверждение больше не требуется.
- `/list_users` выводит текущих администраторов и операторов.
- `/tz` запускает кнопочный выбор часового пояса для личных уведомлений и расписаний.
- `/help` повторяет памятку с описанием всех кнопочных сценариев.

### Рубрики
- При инициализации бот автоматически создаёт `flowers` и `guess_arch` в выключенном состоянии и без расписаний.
- `/rubrics` показывает карточки каждой рубрики. Внутри доступны кнопки:
  - `Включить/Выключить` — мгновенное переключение статуса.
  - `Канал` и `Тест-канал` — список каналов с пагинацией и поиском, всё через inline-кнопки.
  - `Добавить расписание` — пошаговый мастер (выбор времени, минут, дней недели и канала), который также позволяет редактировать или отключать отдельные слоты.
  - `▶️ Запустить` и `🧪 Тест` — сперва показывают запрос подтверждения, затем ставят задачу `publish_rubric` в очередь и отправляют номер задания.

### Мастер расписаний
- Временной шаг сначала предлагает часы, затем минуты; выбор отображается прямо в тексте карточки.
- Список дней недели отмечает активные значения галочками, а кнопки `Все`, `Очистить` и `Готово` завершают выбор без ввода текста.
- При переключении на «Канал» мастер открывает тот же канал-пикер, что и карточка рубрики, с возможностью вернуться назад одним нажатием.
- Кнопки `Сохранить` и `Отмена` фиксируют изменения и сразу возвращают к карточке рубрики.

### Каналы, погода и история
- `/channels` — аудит подключённых каналов.
- `/set_weather_assets_channel` и `/set_recognition_channel` — кнопочный выбор каналов для хранения ассетов и входного распознавания.
- `/setup_weather` и `/list_weather_channels` — управление погодными рассылками с кнопками `Run now`/`Stop` и отметками последнего запуска.
- `/weather`, `/history`, `/scheduled` — статусные отчёты с возможностью отменять или переносить публикации прямо из inline-интерфейса.
- `/amber` — отдельный раздел для каналов Янтарного проекта, также полностью кнопочный.

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
- `ASSETS_DEBUG_EXIF` – when set to a truthy value, replies to recognized messages with a raw EXIF dump for debugging; defaults to disabled (`0`).
- `PORT` – aiohttp listener used when calling `web.run_app`; defaults to `8080` and must align with the port exposed by your hosting provider.
- `4O_API_KEY` – key used by the recognition pipeline and rubric copy generators; when missing, related jobs are skipped automatically.
- `OPENAI_DAILY_TOKEN_LIMIT`, `OPENAI_DAILY_TOKEN_LIMIT_GPT_4O` (`OPENAI_DAILY_TOKEN_LIMIT_4O` legacy), `OPENAI_DAILY_TOKEN_LIMIT_GPT_4O_MINI` (`OPENAI_DAILY_TOKEN_LIMIT_4O_MINI` legacy) – optional per-model quotas that gate new OpenAI jobs until the next UTC reset.
- `PORT` – HTTP port that `web.run_app` listens on (default `8080`). Ensure it matches the port exposed by your proxy or hosting platform (Fly.io, Docker, etc.) so inbound requests reach the app.
- `SUPABASE_URL`, `SUPABASE_KEY` – optional credentials for the Supabase project that receives OpenAI token usage events. When configured the bot mirrors SQLite usage rows into the `token_usage` table for centralized analytics, storing `bot`, `model`, prompt/completion/total token counts, `request_id`, `endpoint` (`responses`), strictly JSON-serialized `meta` and timestamp `at` in UTC ISO8601. Each Supabase call also emits a structured `log_token_usage` record – identical for successes and failures – so log shippers can archive the same payloads even when Supabase is unreachable.

### External services
- **Nominatim** – the bot queries `https://nominatim.openstreetmap.org/reverse` and rate-limits calls to one request per second. Set `User-Agent` friendly values in the code if you fork, and consider running your own Nominatim instance for higher throughput.
- **OpenAI Responses API** – outbound requests target the `/responses` endpoint; ensure outbound egress is permitted from your hosting environment.
- **Supabase REST** – if `SUPABASE_URL`/`SUPABASE_KEY` are set, the bot posts token usage metrics to `rest/v1/token_usage` using the Supabase service role key.

### Local assets & overlays
- Store overlay PNGs for the `guess_arch` rubric inside the directory referenced by the rubric config (default `main.py` → `overlays`). Files named `1.png`, `2.png`, etc., are overlaid on published photos; the bot auto-generates numeric badges when files are missing.
- Keep badge graphics roughly within 10–16% of the shorter photo side. The bot rescales custom overlays into this window before compositing so numbers stay legible without hiding too much of the picture.
- Reserve a safe zone in the top-left corner: standard frames leave 24 px of padding from both edges, while compact assets with a shorter side under 480 px shrink the offset to 12 px. Keep critical details outside this area so auto-numbering never covers the subject.
- Persist asset storage on a volume so ingestion and recognition jobs can retry without re-downloading media. Temporary directories will break overlay generation and duplicate detection after restarts.

## Job Queue and Manual Rubrics
- The SQLite-backed job queue starts automatically when the bot boots, loads due jobs every second and executes handlers concurrently according to `JobQueue(concurrency=...)` settings. Failed jobs retry with exponential backoff up to five attempts before being marked as `failed`. Inspect `jobs_queue` via any SQLite tool for troubleshooting.
- Для ручного запуска откройте карточку рубрики и нажмите `▶️ Запустить` (боевой слот) или `🧪 Тест` (тестовый слот). Теперь бот не публикует пост сразу: в операторском чате появляется превью дропа с кнопками для выбора города публикации, перегенерации фотографий, перегенерации подписи, добавления дополнительных инструкций и финальной отправки в боевой или тестовый канал. Погодный блок в превью использует тот же формат, что и публикация — числовые метрики рядом с текстовыми условиями без кодов. Перед подтверждением публикации операторы сверяются с этим README и `docs/weather.md`, чтобы соблюдать актуальные требования к погодной подаче.
- Ручной запуск проходит по короткому сценарию: (1) инициируйте запуск кнопкой на карточке; (2) изучите превью в чате оператора; (3) при необходимости используйте кнопки перегенерации и `Добавить инструкции`, чтобы уточнить текст — ввод осуществляется прямо в модальном ответе Telegram; (4) выберите назначение через кнопки доставки в тестовый или основной канал.
- The job queue deduplicates identical pending payloads, so repeated runs from the same button are safe.
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

Provision Fly.io secrets (`TELEGRAM_BOT_TOKEN`, `WEBHOOK_URL`, `4O_API_KEY`, token limits) before the first deployment.

## Legacy
Historical documentation for the weather scheduler, including sea temperature handling and template placeholders, now lives in `docs/weather.md`. Those features remain in the codebase for backward compatibility but are no longer part of the primary rubric workflow.
