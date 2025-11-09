from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import parse_qs, urlparse

_PAIRING_TOKEN_PATTERN = re.compile(r"^[A-Z2-9]{6,8}$")


class PairingTokenError(ValueError):
    """Raised when a pairing token cannot be normalized."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _extract_deeplink_token(raw: str) -> str:
    try:
        parsed = urlparse(raw)
    except ValueError as exc:  # pragma: no cover - malformed URIs raise ValueError
        raise PairingTokenError("Не удалось разобрать deeplink.") from exc

    if parsed.scheme.lower() != "catweather" or parsed.netloc.lower() != "pair":
        raise PairingTokenError("Неизвестный deeplink для привязки устройства.")

    query = parse_qs(parsed.query)
    values = query.get("token") or query.get("code") or []
    if not values:
        raise PairingTokenError("В deeplink отсутствует параметр token.")

    return values[0]


def _normalize_raw_token(raw: str) -> str:
    candidate = raw.strip()
    if not candidate:
        raise PairingTokenError("Токен не может быть пустым.")

    if candidate.upper().startswith("PAIR:"):
        candidate = candidate[5:]
    elif candidate.lower().startswith("catweather://"):
        candidate = _extract_deeplink_token(candidate)

    candidate = candidate.strip()
    if not candidate:
        raise PairingTokenError("Токен не может быть пустым.")

    normalized = candidate.upper()
    if not _PAIRING_TOKEN_PATTERN.fullmatch(normalized):
        raise PairingTokenError("Недопустимый формат токена: ожидаем 6–8 символов A-Z и 2-9.")

    return normalized


def normalize_pairing_token(payload: Mapping[str, object]) -> str:
    token_value = payload.get("token")
    code_value = payload.get("code")

    raw_value: str | None = None
    if isinstance(token_value, str) and token_value.strip():
        raw_value = token_value
    elif isinstance(code_value, str) and code_value.strip():
        raw_value = code_value

    if raw_value is None:
        raise PairingTokenError("Поле token обязательно.")

    return _normalize_raw_token(raw_value)


__all__ = ["PairingTokenError", "normalize_pairing_token"]
