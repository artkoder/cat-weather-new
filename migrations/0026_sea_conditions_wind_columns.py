"""Migration 0026: add missing sea_conditions columns for enhanced wind data."""

from __future__ import annotations

import sqlite3


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    try:
        conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
        return True
    except sqlite3.OperationalError:
        return False


def run(conn: sqlite3.Connection) -> None:
    """Add missing columns to sea_conditions table for enhanced wind tracking."""

    # List of columns to add with their definitions
    columns_to_add = [
        ("wind_speed_10m_kmh", "REAL"),
        ("wind_gusts_10m_ms", "REAL"),
        ("wind_gusts_10m_kmh", "REAL"),
        ("wind_units", "TEXT"),
        ("wind_gusts_units", "TEXT"),
        ("wind_time_ref", "TEXT"),
    ]

    for column_name, column_def in columns_to_add:
        if not _column_exists(conn, "sea_conditions", column_name):
            conn.execute(f"ALTER TABLE sea_conditions ADD COLUMN {column_name} {column_def}")
            print(f"Added column {column_name} to sea_conditions")
        else:
            print(f"Column {column_name} already exists in sea_conditions")

    conn.commit()
