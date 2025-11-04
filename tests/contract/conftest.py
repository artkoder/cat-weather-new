import os
from pathlib import Path


def pytest_sessionstart(session):
    # Set a safe DB path for tests BEFORE imports
    if "DB_PATH" not in os.environ:
        tmp_dir = Path(session.config.rootpath) / ".pytest_db"
        tmp_dir.mkdir(exist_ok=True)
        os.environ["DB_PATH"] = str(tmp_dir / "bot.db")
