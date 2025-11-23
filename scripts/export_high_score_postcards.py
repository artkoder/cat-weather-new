"""Export postcard-ready assets (scores 7 and 8) into a CSV dump."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from postcard_export import export_high_score_assets  # noqa: E402

DEFAULT_DB_FALLBACK = "/data/bot.db"
DEFAULT_OUTPUT = "postcard_photos_dump.csv"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump all postcard assets regardless of postcard_score. "
            "Scores are resolved both from the assets table and the latest vision result."
        )
    )
    parser.add_argument(
        "--db-path",
        help=("Path to the SQLite database. Defaults to DB_PATH env or /data/bot.db."),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Destination CSV file (defaults to postcard_photos_dump.csv in the CWD)."
        ),
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help=("Pretty-print JSON blobs for readability before writing them to the CSV."),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_db_path(candidate: str | None) -> str:
    if candidate:
        return candidate
    env_value = os.getenv("DB_PATH")
    if env_value:
        return env_value
    return DEFAULT_DB_FALLBACK


def _connect(db_path: str) -> sqlite3.Connection:
    uri = db_path.startswith("file:")
    conn = sqlite3.connect(db_path, uri=uri)
    conn.row_factory = sqlite3.Row
    return conn


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = _resolve_db_path(args.db_path)
    output_path = Path(args.output).expanduser().resolve()
    try:
        conn = _connect(db_path)
    except sqlite3.Error as exc:
        print(f"Failed to open database at {db_path}: {exc}")
        return 1
    try:
        row_count = export_high_score_assets(
            conn,
            output_path=output_path,
            pretty_json=bool(args.pretty_json),
        )
    finally:
        conn.close()
    print(f"Exported {row_count} record(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
