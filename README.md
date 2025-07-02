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
- Daily weather posts use images from a dedicated private channel selected with
  `/set_assets_channel`.

- Forecast periods (morning/day/evening/night) are averaged from hourly data and
  rounded to whole degrees for smoother values.


## Commands
- /start - register or access bot
- /pending - list pending users (admin)
- /approve <id> - approve user
- /reject <id> - reject user
- /add_user <id> - manually add a user (superadmin)
- /list_users - list approved users
- /remove_user <id> - remove user
- /channels - list channels (admin)
- /scheduled - show scheduled posts with target channel names
- /history - recent posts
- /tz <offset> - set timezone offset (e.g., +02:00). Affects daily weather schedules immediately
- /addbutton <post_url> <text> <url> - add inline button to existing post (button text may contain spaces)
- /delbutton <post_url> - remove all buttons from an existing post and clear stored weather buttons

- /addcity <name> <lat> <lon> - add a city for weather checks (admin, coordinates

  may include six or more decimal places and may be separated with a comma)
- /addsea <name> <lat> <lon> - add a sea location for water temperature checks
  (comma separator allowed)

- /cities - list cities with inline delete buttons (admin). Coordinates are shown
  with six decimal places.
- /seas - list sea locations with inline delete buttons (admin).
- /weather [now] - show cached weather; append `now` to refresh data
- /regweather <post_url> <template> - register a post for weather updates

 - /addweatherbutton <post_url> <text> [url] - attach a button linking to the latest forecast. Text supports the same placeholders as templates. Multiple weather buttons share one row

- /weatherposts [update] - list registered weather posts with a 'Stop weather' button on each; append `update` to refresh
- /setup_weather - interactive wizard to add a daily forecast channel
- /list_weather_channels - show configured weather channels with action buttons
- /set_assets_channel - choose the channel used for weather assets


`/list_weather_channels` displays the last publication time adjusted to your
current `/tz` setting. When using the "Run now" button, the bot attempts to copy
the next available asset. The run is not recorded, so the regular scheduled post
for that day will still happen. If no unused asset exists, it replies with

"No asset to publish".

### Asset channel
Images and caption templates are stored in a private channel
`@kotopogoda_assets`. Choose this channel with `/set_assets_channel` **before**
uploading assets. Only posts sent after the bot becomes an admin are captured.

If you edit a post in this channel, the bot updates the stored template.
Used posts are deleted automatically after publishing so the channel always
contains only fresh assets.




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
- **US-8.1**: `/addbutton` appends a new button without removing existing ones.

- **US-9**: `/delbutton <post_url>` removes all inline buttons from an existing channel post and deletes stored weather button data.

- **US-10**: Admin adds a city with `/addcity`.
- **US-11**: Admin views and removes cities with `/cities`.
- **US-12**: Periodic weather collection from Open-Meteo with up to three retries on failure.
- **US-13**: Admin requests last weather check info and can force an update.
- **US-14**: Admin registers a weather post for updates, including sea temperature.

 - **US-14.1**: `/addweatherbutton <post_url> <text> [url]` attaches a button linking to the latest `#котопогода`. Multiple weather buttons are placed on one row. `/weatherposts` lists these posts with a remove option.

- **US-15**: Automatic weather post updates with current weather and sea temperature.
- **US-16**: Admin lists registered posts showing the rendered weather and sea data for all registered seas.
- **US-16.1**: Admin stops weather updates for a post using the "Stop weather" button shown in `/weatherposts`.
- **US-17**: Admin adds a channel for daily weather posts and specifies the publication time with `/setup_weather`.
- **US-18**: Content manager uploads images with templates to `@kotopogoda_assets`; used posts disappear after publishing.
- **US-19**: Admin views the list of weather channels and can send a post immediately with «Run now» or remove a channel with «Stop».
- **US-20**: The bot publishes the weather once per day for each configured channel at the set time.




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

