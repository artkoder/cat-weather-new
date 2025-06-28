# Telegram Scheduler Bot

This bot allows authorized users to schedule posts to their Telegram channels.

## Features
- User authorization with superadmin.
- Channel tracking where bot is admin.
- Schedule message forwarding to one or more channels with inline interface. The bot forwards the original post so views and custom emoji are preserved. It must be a member of the source channel.
- If forwarding fails (e.g., bot not in source), the message is copied instead.
- View posting history.
- User lists show clickable usernames for easy profile access.
- Local timezone support for scheduling.
- Configurable scheduler interval.
- Add inline buttons to existing posts.
- Remove inline buttons from existing posts.
- Weather updates from Open-Meteo roughly every 30 minutes with the raw response logged. Admins
  can view the latest data or force an update with `/weather now`. The `/weather` command lists
  the cached weather and sea temperature for all registered locations.
- Register channel posts with custom templates for automatic weather updates,
  including sea temperature, working with both text and caption posts.


## Commands
- /start - register or access bot
- /pending - list pending users (admin)
- /approve <id> - approve user
- /reject <id> - reject user
- /list_users - list approved users
- /remove_user <id> - remove user
- /channels - list channels (admin)
- /scheduled - show scheduled posts with target channel names
- /history - recent posts
- /tz <offset> - set timezone offset (e.g., +02:00)
- /addbutton <post_url> <text> <url> - add inline button to existing post (button text may contain spaces)
- /delbutton <post_url> - remove all buttons from an existing post

- /addcity <name> <lat> <lon> - add a city for weather checks (admin, coordinates

  may include six or more decimal places and may be separated with a comma)
- /addsea <name> <lat> <lon> - add a sea location for water temperature checks
  (comma separator allowed)

- /cities - list cities with inline delete buttons (admin). Coordinates are shown
  with six decimal places.



## User Stories

### Done
- **US-1**: Registration of the first superadmin.
- **US-2**: User registration queue with limits and admin approval flow.
- **US-3**: Superadmin manages pending and approved users. Rejected users cannot
  register again. Pending and approved lists display clickable usernames with
  inline approval buttons.
- **US-4**: Channel listener events and `/channels` command.
- **US-5**: Post scheduling interface with channel selection, cancellation and rescheduling. Scheduled list shows the post preview or link along with the target channel name and time in HH:MM DD.MM.YYYY format.
- **US-6**: Scheduler forwards queued posts at the correct local time. If forwarding fails because the bot is not a member, it falls back to copying. Interval is configurable and all actions are logged.
- **US-8**: `/addbutton <post_url> <text> <url>` adds an inline button to an existing channel post. Update logged with INFO level.
- **US-9**: `/delbutton <post_url>` removes inline buttons from an existing channel post.
- **US-10**: Admin adds a city with `/addcity`.
- **US-11**: Admin views and removes cities with `/cities`.
- **US-12**: Periodic weather collection from Open-Meteo with up to three retries on failure.
- **US-13**: Admin requests last weather check info and can force an update.
- **US-14**: Admin registers a weather post for updates, including sea temperature.
- **US-15**: Automatic weather post updates with current weather and sea temperature.
- **US-16**: Admin lists registered posts showing the rendered weather and sea data for all registered seas.




### In Progress
- **US-7**: Logging of all operations.

### Planned

## Deployment
The bot is designed for Fly.io using a webhook on `/webhook` and listens on port `8080`.
For Telegram to reach the webhook over HTTPS, the Fly.io service must expose port `443` with TLS termination enabled. This is configured in `fly.toml`.

### Environment Variables
- `TELEGRAM_BOT_TOKEN` – Telegram bot API token.

- `WEBHOOK_URL` – external HTTPS URL of the deployed application. Used to register the Telegram webhook.

- `DB_PATH` – path to the SQLite database (default `bot.db`).
- `FLY_API_TOKEN` – token for automated Fly deployments.
- `TZ_OFFSET` – default timezone offset like `+02:00`.
- `SCHED_INTERVAL_SEC` – scheduler check interval in seconds (default `30`).

### Запуск локально
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Запустите бота:
   ```bash
   python main.py
   ```

> Fly.io secrets `TELEGRAM_BOT_TOKEN` и `FLY_API_TOKEN` должны быть заданы перед запуском.


### Деплой на Fly.io

1. Запустить приложение в первый раз (из CLI, однократно):

```bash
fly launch
fly volumes create sched_db --size 1


```

2. После этого любой push в ветку `main` будет автоматически триггерить деплой.

3. Все секреты устанавливаются через Fly.io UI или CLI:

```bash
fly secrets set TELEGRAM_BOT_TOKEN=xxx
fly secrets set WEBHOOK_URL=https://<app-name>.fly.dev/
```

The `fly.toml` file should expose port `443` so that Telegram can connect over HTTPS.

## CI/CD
Каждый push в main запускает GitHub Actions → flyctl deploy → Fly.io.

