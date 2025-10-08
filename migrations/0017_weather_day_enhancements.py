from __future__ import annotations

import sqlite3

def run(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info(weather_cache_day)")
    existing = {row[1] for row in cursor.fetchall()}
    columns = [
        ("temp_min", "REAL"),
        ("temp_max", "REAL"),
        ("precipitation", "REAL"),
        ("wind_speed_max", "REAL"),
    ]
    for name, col_type in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE weather_cache_day ADD COLUMN {name} {col_type}")
