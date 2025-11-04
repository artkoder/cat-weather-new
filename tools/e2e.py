"""CLI helpers for end-to-end testing flows."""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from data_access import create_pairing_token

_PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_DEFAULT_DB_PATH = "/data/bot.db"


@dataclass(slots=True)
class CreatePairingOptions:
    user_id: int
    device_name: str
    ttl: int
    code: str | None
    db_path: str


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _require_e2e_mode() -> None:
    if not _env_bool("E2E_MODE"):
        raise RuntimeError(
            "E2E helpers are disabled. Set E2E_MODE=true in the environment to enable them."
        )


def _generate_pairing_code(rng: random.Random) -> str:
    length = rng.randint(6, 8)
    return "".join(rng.choice(_PAIRING_ALPHABET) for _ in range(length))


def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _create_pairing(options: CreatePairingOptions) -> str:
    _require_e2e_mode()
    rng = random.SystemRandom()
    code = options.code or _generate_pairing_code(rng)
    ttl = max(int(options.ttl), 0)
    conn = _open_db(options.db_path)
    try:
        create_pairing_token(
            conn,
            code=code,
            user_id=options.user_id,
            device_name=options.device_name,
            ttl_sec=ttl,
        )
        conn.commit()
    finally:
        conn.close()
    return code


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E helper utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create-pairing", help="Create a temporary pairing code for device attachment"
    )
    create.add_argument("--user-id", type=int, required=True, help="Telegram user identifier")
    create.add_argument(
        "--device-name",
        required=True,
        help="Label that will be associated with the generated device",
    )
    create.add_argument(
        "--ttl",
        type=int,
        default=int(os.getenv("PAIRING_TOKEN_TTL_SECONDS", "600") or 600),
        help="Lifetime of the pairing code in seconds (default: env PAIRING_TOKEN_TTL_SECONDS or 600)",
    )
    create.add_argument(
        "--code",
        help="Optional explicit pairing code to use instead of generating a random one",
    )
    create.add_argument(
        "--db-path",
        default=os.getenv("DB_PATH", _DEFAULT_DB_PATH),
        help="Path to the SQLite database (default: env DB_PATH or /data/bot.db)",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.command == "create-pairing":
            options = CreatePairingOptions(
                user_id=args.user_id,
                device_name=args.device_name,
                ttl=args.ttl,
                code=args.code,
                db_path=args.db_path,
            )
            code = _create_pairing(options)
            print(code)
            return 0
        print("Unknown command", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - surfaced to caller
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
