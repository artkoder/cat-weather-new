from __future__ import annotations

from typing import Any

from weather_migration import migrate_weather_publish_channels


def run(conn: Any) -> None:
    migrate_weather_publish_channels(conn)
