import logging
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def dump_database(db_path: str | Path) -> Path:
    """
    Create a full binary dump of the SQLite database using the backup API.

    Args:
        db_path: Path to the source database file.

    Returns:
        Path to the temporary file containing the backup.
    """
    source_path = Path(db_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Database file not found at {source_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Create a temporary file for the backup
    # We use delete=False so we can close it and then send it/read it
    fd, temp_path_str = tempfile.mkstemp(suffix=".db", prefix=f"dump_{timestamp}_")
    # Close the file descriptor immediately, we'll open it via sqlite3 or open()
    os.close(fd)

    dest_path = Path(temp_path_str)

    logging.info("Starting database dump from %s to %s", source_path, dest_path)

    src_conn = None
    dst_conn = None

    try:
        # Open connections
        src_conn = sqlite3.connect(source_path)
        dst_conn = sqlite3.connect(dest_path)

        # Use the SQLite backup API
        # pages=-1 means copy all pages at once (or chunked if we wanted progress)
        src_conn.backup(dst_conn)

        logging.info("Database dump completed successfully")

        return dest_path

    except Exception as e:
        logging.exception("Database dump failed")
        # Cleanup on failure
        if dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        raise e

    finally:
        if dst_conn:
            dst_conn.close()
        if src_conn:
            src_conn.close()
