from __future__ import annotations

from weather_migration import migrate_weather_publish_channels


def run(conn):
    migrate_weather_publish_channels(conn)
