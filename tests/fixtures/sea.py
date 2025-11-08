from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path


def seed_sea_environment(
    bot,
    *,
    sea_id: int = 1,
    sea_name: str = "Балтика",
    sea_lat: float = 54.95,
    sea_lon: float = 20.2,
    wave: float = 0.3,
    water_temp: float | None = 8.5,
    city_id: int = 101,
    city_name: str = "Зеленоградск",
    city_lat: float = 54.9604,
    city_lon: float = 20.4721,
    wind_speed: float = 5.0,
    wind_gust: float | None = None,
    cloud_cover: float = 35.0,
    timestamp: datetime | None = None,
) -> str:
    """Seed reference sea, city, and weather rows for integration tests."""

    ts = (timestamp or datetime.utcnow()).isoformat()
    wind_speed_ms = wind_speed
    wind_speed_kmh = wind_speed_ms * 3.6 if wind_speed_ms is not None else None
    gust_ms = wind_gust
    gust_kmh = gust_ms * 3.6 if gust_ms is not None else None
    wind_units = "m/s" if wind_speed_ms is not None else None
    gust_units = "m/s" if gust_ms is not None else None

    bot.db.execute(
        "INSERT OR REPLACE INTO seas (id, name, lat, lon) VALUES (?, ?, ?, ?)",
        (sea_id, sea_name, sea_lat, sea_lon),
    )
    bot.db.execute(
        """
        INSERT OR REPLACE INTO sea_cache (
            sea_id, updated, current, morning, day, evening, night,
            wave, morning_wave, day_wave, evening_wave, night_wave
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sea_id, ts, water_temp, None, None, None, None, wave, None, None, None, None),
    )
    bot.db.execute(
        """
        INSERT OR REPLACE INTO sea_conditions (
            sea_id, updated, wave_height_m, wind_speed_10m_ms, wind_speed_10m_kmh,
            wind_gusts_10m_ms, wind_gusts_10m_kmh, wind_units, wind_gusts_units,
            wind_time_ref, cloud_cover_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sea_id,
            ts,
            wave,
            wind_speed_ms,
            wind_speed_kmh,
            gust_ms,
            gust_kmh,
            wind_units,
            gust_units,
            ts,
            cloud_cover,
        ),
    )
    bot.db.execute(
        "INSERT OR REPLACE INTO cities (id, name, lat, lon) VALUES (?, ?, ?, ?)",
        (city_id, city_name, city_lat, city_lon),
    )
    bot.db.execute(
        """
        INSERT OR REPLACE INTO weather_cache_hour (
            city_id, timestamp, temperature, weather_code, wind_speed, is_day
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (city_id, ts, 10.0, 1, wind_speed, 1),
    )
    bot.db.commit()
    return ts


def create_stub_image(directory: Path, name: str) -> Path:
    """Create a tiny placeholder image file for Telegram payloads."""

    path = directory / name
    path.write_bytes(b"stub-image-data")
    return path


def create_sea_asset(
    bot,
    *,
    rubric_id: int,
    message_id: int,
    file_name: str,
    local_path: Path,
    tags: Sequence[str] | None = None,
    categories: Iterable[str] | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    channel_id: int = -1000,
    sea_wave_score: float = 2.0,
    photo_sky: str = "partly_cloudy",
    is_sunset: bool = False,
    season_guess: str = "summer",
    sky_visible: bool = True,
) -> str:
    """Persist a sea asset with the provided metadata and local source."""

    file_meta = {
        "file_id": f"file-{message_id}",
        "file_unique_id": f"unique-{message_id}",
        "file_name": file_name,
        "mime_type": "image/jpeg",
        "width": 1080,
        "height": 1080,
    }
    asset_id = bot.data.save_asset(
        channel_id,
        message_id,
        template=None,
        hashtags=None,
        tg_chat_id=channel_id,
        caption=None,
        kind="photo",
        file_meta=file_meta,
        rubric_id=rubric_id,
        origin="sea_fixture",
    )
    vision_payload = {
        "tags": list(tags or []),
        "is_sea": True,
        "sea_wave_score": sea_wave_score,
        "photo_sky": photo_sky,
        "sky_visible": sky_visible,
        "is_sunset": is_sunset,
        "season_guess": season_guess,
    }
    bot.data.update_asset(
        asset_id,
        vision_category="sea",
        vision_results=vision_payload,
        local_path=str(local_path),
        latitude=latitude,
        longitude=longitude,
        rubric_id=rubric_id,
    )
    if categories:
        bot.data.update_asset_categories_merge(asset_id, categories)
    bot.db.commit()
    return asset_id
