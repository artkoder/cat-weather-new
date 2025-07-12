# Weather Extension

This document describes the weather feature set for the Telegram scheduler bot.

Weather for each city is queried from the Open-Meteo API approximately every 30
minutes and stored in the `weather_cache` table. The bot logs both the raw HTTP
response and the parsed weather information. The request looks like:

```
https://api.open-meteo.com/v1/forecast?latitude=<lat>&longitude=<lon>&current=temperature_2m,weather_code,wind_speed_10m,is_day&timezone=auto
```

Sea conditions use the marine API endpoint:

```
https://api.open-meteo.com/v1/marine?latitude=<lat>&longitude=<lon>&current=wave_height,wind_speed_10m,sea_surface_temperature&hourly=wave_height,wind_speed_10m,sea_surface_temperature&forecast_days=2&timezone=auto
```

### Storm rating

The bot looks at the wave height in meters and wind speed at 10 m above the
sea surface. When both values stay below **0.5 m** and **5 m/s** the sea is
considered calm and `{id|seastorm}` prints the water temperature just like
`{id|seatemperature}`. If either metric rises above those values the placeholder
expands to `шторм`. A wave of **1.5 m** or more or wind of **10 m/s** or more
produces `сильный шторм`.

The bot continues working even if a query fails. When a request fails, it is
retried up to three times with a one‑minute pause between attempts. After that,
no further requests are made for that city until the next scheduled half hour.



## Commands

- `/addcity <name> <lat> <lon>` – add a city to the database. Only superadmins can
  execute this command. Latitude and longitude must be valid floating point numbers
  and may include six or more digits after the decimal point. Coordinates may be
  separated with a comma.
- `/cities` – list registered cities. Each entry has an inline *Delete* button that
  removes the city from the list. Coordinates are displayed with six decimal digits
  to reflect the stored precision.

- `/seas` – list sea locations with inline *Delete* buttons.
- `/addsea <name> <lat> <lon>` – add a sea location for water temperature checks.
  Coordinates may also be separated with a comma.
- `/weather` – show the last collected weather for all cities and sea locations. Only superadmins may


  request this information. Append `now` to force a fresh API request for both
  weather and sea data before displaying results.
- `/regweather <post_url> <template>` – register a channel post for automatic
  weather updates. The template may include placeholders like

  `{<city_id>|temperature}` or `{<city_id>|wind}` mixed with text. Water

  temperature can be inserted with `{<sea_id>|seatemperature}` which expands to
  the sea emoji followed by the current temperature like `🌊 15.1°C`. Storm
  conditions are available with `{<sea_id>|seastorm}`. When waves are below
  0.5 m and wind below 5 m/s it behaves like `{<sea_id>|seatemperature}`.
  Values above that show `шторм`, and a wave higher than 1.5 m or wind faster
  than 10 m/s shows `сильный шторм`.

  Message text already containing a weather header separated by `∙` is stripped
  when registering so only the original text remains.

 - `/addweatherbutton <post_url> <text> [url]` – add a button linking to the latest forecast. Button text supports the same placeholders as templates. Provide the URL manually if no forecast exists yet. Multiple weather buttons appear on the same row.
- `/delbutton <post_url>` – remove all buttons from a post and delete any stored weather button data so they do not reappear.


- `/weatherposts` – list registered weather posts. Append `update` to refresh all posts immediately. Each entry shows the post link and current weather header with a *Stop weather* button. Pressing the button removes the weather header and stops further updates.

### Templates

Placeholders are replaced with cached values when updating posts. If no data is

available the post is left unchanged and a log entry is written. Posts can be
plain text or contain media with a caption—the bot will edit either field as
needed. The rendered header is prepended to the original text or caption
separated by the `∙` character for reliable replacement on each update.




## Database schema

```
CREATE TABLE IF NOT EXISTS cities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS weather_cache_day (
    city_id INTEGER NOT NULL,
    day DATE NOT NULL,
    temperature REAL,
    weather_code INTEGER,
    wind_speed REAL,
    PRIMARY KEY (city_id, day)
);

CREATE TABLE IF NOT EXISTS weather_cache_hour (
    city_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    temperature REAL,
    weather_code INTEGER,
    wind_speed REAL,
    is_day INTEGER,
    PRIMARY KEY (city_id, timestamp)
);



CREATE TABLE IF NOT EXISTS weather_posts (
    id INTEGER PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    template TEXT NOT NULL,
    base_text TEXT,

    base_caption TEXT,
    reply_markup TEXT,

    UNIQUE(chat_id, message_id)
);

CREATE TABLE IF NOT EXISTS seas (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS sea_cache (
    sea_id INTEGER PRIMARY KEY,
    updated TEXT,
    current REAL,
    morning REAL,
    day REAL,
    evening REAL,
    night REAL
);
```

## Open-Meteo example response

```json
{
  "latitude": 55.75,
  "longitude": 37.62,
  "current": {
    "temperature_2m": 20.5,
    "weather_code": 1,
    "wind_speed_10m": 3.5
  }
}
```

## WMO code to emoji

| Code | Emoji |
|-----:|:------|
| 0 | ☀️ (🌙 at night) |
| 1 | 🌤 |
| 2 | ⛅ |
| 3 | ☁️ |
| 45 | 🌫 |
| 48 | 🌫 |
| 51 | 🌦 |
| 53 | 🌦 |
| 55 | 🌦 |
| 61 | 🌧 |
| 63 | 🌧 |
| 65 | 🌧 |
| 71 | ❄️ |
| 73 | ❄️ |
| 75 | ❄️ |
| 80 | 🌦 |
| 81 | 🌦 |
| 82 | 🌧 |
| 95 | ⛈ |
| 96 | ⛈ |
| 99 | ⛈ |
```
