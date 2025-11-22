from __future__ import annotations

import logging
import math
from typing import Any, Literal, NamedTuple

import httpx

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
_OVERPASS_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    "User-Agent": "cat-weather-new/1.0 (+https://github.com/catforecaster/cat-weather-new)",
}

WaterKind = Literal["sea", "lake", "lagoon", "river", "other"]


class WaterInfo(NamedTuple):
    kind: WaterKind
    name_ru: str | None


class NationalParkInfo(NamedTuple):
    osm_name_ru: str
    short_name: str
    hashtag: str


class SettlementInfo(NamedTuple):
    name: str
    distance_m: float


async def overpass_query(query: str, timeout: int = 25) -> dict[str, Any]:
    logging.info("OVERPASS request timeout=%s query=%s", timeout, query)
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_OVERPASS_HEADERS) as client:
            response = await client.post(
                OVERPASS_ENDPOINT,
                data={"data": query},
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - network errors are rare in tests
        logging.error("OVERPASS http_error %s", exc)
        raise RuntimeError("Overpass request failed") from exc
    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - unexpected payload
        logging.error("OVERPASS json_decode_error", exc_info=True)
        raise RuntimeError("Overpass JSON decode failed") from exc


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_earth * c


def _extract_name(tags: dict[str, Any]) -> str | None:
    if not isinstance(tags, dict):
        return None
    candidates = (tags.get("name:ru"), tags.get("name"))
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        text = candidate.strip()
        if not text:
            continue
        for delimiter in (";", "/"):
            if delimiter in text:
                text = text.split(delimiter, 1)[0].strip()
        if text:
            return text
    return None


def _extract_center(element: dict[str, Any]) -> tuple[float, float] | None:
    if not isinstance(element, dict):
        return None
    center = element.get("center")
    if isinstance(center, dict):
        lat = center.get("lat")
        lon = center.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
    lat = element.get("lat")
    lon = element.get("lon")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return float(lat), float(lon)
    return None


def _format_coord(value: float) -> str:
    return f"{value:.6f}"


def _normalize_radius(radius_m: int) -> int:
    try:
        value = int(radius_m)
    except (TypeError, ValueError):
        value = 0
    return max(1, value)


async def find_water_body(lat: float, lon: float, radius_m: int = 300) -> WaterInfo | None:
    radius = _normalize_radius(radius_m)
    lat_text = _format_coord(lat)
    lon_text = _format_coord(lon)
    coastline_query = (
        "[out:json][timeout:25];"
        "("  # noqa: PIE807 - easier to read multi-line query
        f'way(around:{radius},{lat_text},{lon_text})["natural"="coastline"];'
        f'relation(around:{radius},{lat_text},{lon_text})["natural"="coastline"];'
        ");"
        "out tags center;"
    )
    coastline_response = await overpass_query(coastline_query)
    elements = coastline_response.get("elements")
    if isinstance(elements, list) and elements:
        return WaterInfo(kind="sea", name_ru="Балтийское море")
    water_query = (
        "[out:json][timeout:25];"
        "("  # noqa: PIE807
        f'way(around:{radius},{lat_text},{lon_text})["natural"="water"];'
        f'relation(around:{radius},{lat_text},{lon_text})["natural"="water"];'
        ");"
        "out tags center;"
    )
    water_response = await overpass_query(water_query)
    elements = water_response.get("elements")
    if not isinstance(elements, list) or not elements:
        return None
    best_item: tuple[float, WaterInfo] | None = None
    for element in elements:
        tags = element.get("tags") if isinstance(element, dict) else None
        if not isinstance(tags, dict):
            continue
        center = _extract_center(element)
        if not center:
            continue
        name_ru = _extract_name(tags)
        water_value = str(tags.get("water") or "").strip().casefold()
        if water_value == "lake":
            kind: WaterKind = "lake"
        elif water_value == "lagoon":
            kind = "lagoon"
        elif water_value in {"river", "riverbank"}:
            kind = "river"
        else:
            kind = "other"
        distance = _haversine_distance_m(lat, lon, center[0], center[1])
        water_info = WaterInfo(kind=kind, name_ru=name_ru)
        if best_item is None or distance < best_item[0]:
            best_item = (distance, water_info)
    return best_item[1] if best_item else None


async def find_national_park(lat: float, lon: float) -> NationalParkInfo | None:
    lat_text = _format_coord(lat)
    lon_text = _format_coord(lon)
    query = (
        "[out:json][timeout:25];"
        f"is_in({lat_text},{lon_text})->.a;"
        'rel(pivot.a)["boundary"~"protected_area|national_park"];'
        "out tags;"
    )
    response = await overpass_query(query)
    elements = response.get("elements")
    if not isinstance(elements, list):
        return None
    for element in elements:
        tags = element.get("tags") if isinstance(element, dict) else None
        if not isinstance(tags, dict):
            continue
        name = _extract_name(tags)
        if not name:
            continue
        lowered = name.casefold()
        if "куршск" in lowered and "коса" in lowered:
            return NationalParkInfo(name, "Куршская коса", "#КуршскаяКоса")
        if "балтийск" in lowered and "коса" in lowered:
            return NationalParkInfo(name, "Балтийская коса", "#БалтийскаяКоса")
        if "виштынец" in lowered:
            return NationalParkInfo(name, "Виштынецкий", "#Виштынец")
    return None


async def find_nearest_settlement(lat: float, lon: float, radius_m: int) -> SettlementInfo | None:
    radius = _normalize_radius(radius_m)
    lat_text = _format_coord(lat)
    lon_text = _format_coord(lon)
    query = (
        "[out:json][timeout:25];"
        "("  # noqa: PIE807
        f'node(around:{radius},{lat_text},{lon_text})["place"~"city|town|village|hamlet"];'
        ");"
        "out center tags;"
    )
    response = await overpass_query(query)
    elements = response.get("elements")
    if not isinstance(elements, list) or not elements:
        return None
    best: SettlementInfo | None = None
    best_distance = float("inf")
    for element in elements:
        tags = element.get("tags") if isinstance(element, dict) else None
        if not isinstance(tags, dict):
            continue
        name = _extract_name(tags)
        if not name:
            continue
        center = _extract_center(element)
        if not center:
            continue
        distance = _haversine_distance_m(lat, lon, center[0], center[1])
        if distance < best_distance:
            best_distance = distance
            best = SettlementInfo(name=name, distance_m=distance)
    return best


__all__ = [
    "WaterInfo",
    "NationalParkInfo",
    "SettlementInfo",
    "overpass_query",
    "find_water_body",
    "find_national_park",
    "find_nearest_settlement",
]
