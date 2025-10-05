# Telegram Scheduler Bot

## Summary
[Полную историю изменений см. в `CHANGELOG.md`](CHANGELOG.md).

- **Asset ingestion**. The bot listens to the dedicated recognition channel for new submissions while weather-ready assets live in a separate storage channel. Every photo must contain GPS EXIF data so the bot can resolve the city through Nominatim; authors without coordinates receive an automatic reminder.
- **Recognition pipeline**. After ingestion the asynchronous job queue schedules a `vision` task that classifies the photo with OpenAI `gpt-4o-mini`, storing the rubric category, architectural details and detected flowers while respecting per-model daily token quotas configured via environment variables.
- **Rubric automation**. Two daily rubrics are supported out of the box: `flowers` creates a carousel with greetings for cities detected in flower assets, while `guess_arch` prepares a numbered architecture quiz with optional overlays and weather context. Both rubrics consume recognized assets and clean them up after publishing, auto-initialize on first run and are fully managed through the `/rubrics` inline dashboard.
- **Admin workflow**. Superadmins manage user access, asset channel binding and rubric schedules directly inside Telegram via commands and inline buttons. The admin interface also exposes manual approval queues and quick status messages for rubric runs.
- **Operations guardrails**. OpenAI usage is rate-limited per model, reverse geocoding calls Nominatim with throttling, and each rubric publication is persisted with metadata for auditing through the admin tools.

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
  - `Удалить рубрику` — кнопка внизу карточки, удаляющая запись после подтверждения Telegram.

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
- `PORT` – aiohttp listener used when calling `web.run_app`; defaults to `8080` and must align with the port exposed by your hosting provider.
- `OPENAI_API_KEY` – key used by the recognition pipeline and rubric copy generators; when missing, related jobs are skipped automatically.
- `OPENAI_DAILY_TOKEN_LIMIT`, `OPENAI_DAILY_TOKEN_LIMIT_4O`, `OPENAI_DAILY_TOKEN_LIMIT_4O_MINI` – optional per-model quotas that gate new OpenAI jobs until the next UTC reset.
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
- Для ручного запуска откройте карточку рубрики в интерфейсе оператора и воспользуйтесь кнопками `▶️ Запустить` (боевой канал) или `🧪 Тест` (тестовый канал). Бот поставит задачу `publish_rubric` через очередь и сразу пришлёт уведомление с номером задания, поэтому можно проверять статус без выхода из Telegram.
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

Provision Fly.io secrets (`TELEGRAM_BOT_TOKEN`, `WEBHOOK_URL`, `OPENAI_API_KEY`, token limits) before the first deployment.

## Legacy
Historical documentation for the weather scheduler, including sea temperature handling and template placeholders, now lives in `docs/weather.md`. Those features remain in the codebase for backward compatibility but are no longer part of the primary rubric workflow.
