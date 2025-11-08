"""Add photo metadata columns used for sea selection."""

import sqlite3


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
        return True
    except sqlite3.OperationalError:
        return False


def run(conn: sqlite3.Connection) -> None:
    columns = [
        ("photo_doy", "INTEGER"),
        ("photo_wave", "REAL"),
        ("sky_visible", "TEXT"),
    ]
    for name, definition in columns:
        if not _column_exists(conn, "assets", name):
            conn.execute(f"ALTER TABLE assets ADD COLUMN {name} {definition}")
            print(f"Added column {name} to assets")
        else:
            print(f"Column {name} already exists in assets")
    if _column_exists(conn, "assets", "shot_doy") and _column_exists(conn, "assets", "photo_doy"):
        conn.execute(
            """
            UPDATE assets
               SET photo_doy = COALESCE(photo_doy, shot_doy)
             WHERE shot_doy IS NOT NULL
               AND (photo_doy IS NULL OR photo_doy = '')
            """
        )
        print("Backfilled photo_doy with existing shot_doy values")
    conn.commit()
