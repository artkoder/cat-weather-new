from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from osm_utils import find_national_park, find_nearest_settlement, overpass_query

_BALTIC_SEA_NAME = "Балтийское море"
_WATER_SEARCH_RADIUS_M = 300
_SETTLEMENT_RADIUS_DEFAULT_M = 3000
_SETTLEMENT_RADIUS_IN_PARK_M = 500
_WATER_TAG_HINTS = {"water", "sea", "lake", "river", "ocean"}
_SEA_TAG_HINTS = {"sea", "ocean"}


@dataclass(slots=True)
class GeoContext:
    main_place: str | None
    water_name: str | None
    water_kind: str | None
    national_park: str | None


@dataclass(slots=True)
class _WaterCandidate:
    distance_m: float
    name: str | None
    kind: str | None


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_tag_keys(asset_tags: Sequence[str] | None) -> set[str]:
    normalized: set[str] = set()
    if not asset_tags:
        return normalized
    for tag in asset_tags:
        if tag is None:
            continue
        text = str(tag).strip().casefold()
        if text:
            normalized.add(text)
    return normalized


def _format_coord(value: float) -> str:
    return f"{value:.6f}"


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_earth * c


def _extract_center(element: dict[str, Any]) -> tuple[float, float] | None:
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


def _extract_name(tags: dict[str, Any]) -> str | None:
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


def _normalize_kind_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().casefold()
    return text or None


def _extract_water_kind(tags: dict[str, Any]) -> str | None:
    for key in ("water", "natural"):
        kind = _normalize_kind_value(tags.get(key))
        if not kind:
            continue
        if kind == "riverbank":
            return "river"
        if kind == "ocean":
            return "sea"
        return kind
    return None


def _closest_distance(elements: Any, lat: float, lon: float) -> float | None:
    if not isinstance(elements, list):
        return None
    best_distance: float | None = None
    for element in elements:
        if not isinstance(element, dict):
            continue
        center = _extract_center(element)
        if not center:
            continue
        distance = _haversine_distance_m(lat, lon, center[0], center[1])
        if best_distance is None or distance < best_distance:
            best_distance = distance
    return best_distance


def _extract_water_candidate(element: dict[str, Any], lat: float, lon: float) -> _WaterCandidate | None:
    if not isinstance(element, dict):
        return None
    tags = element.get("tags")
    if not isinstance(tags, dict):
        return None
    center = _extract_center(element)
    if not center:
        return None
    distance = _haversine_distance_m(lat, lon, center[0], center[1])
    name = _extract_name(tags)
    kind = _extract_water_kind(tags)
    return _WaterCandidate(distance_m=distance, name=name, kind=kind)


async def _fetch_coastline_distance(lat: float, lon: float) -> float | None:
    lat_text = _format_coord(lat)
    lon_text = _format_coord(lon)
    query = (
        "[out:json][timeout:25];"
        "("
        f'way(around:{_WATER_SEARCH_RADIUS_M},{lat_text},{lon_text})["natural"="coastline"];'
        f'relation(around:{_WATER_SEARCH_RADIUS_M},{lat_text},{lon_text})["natural"="coastline"];'
        ");"
        "out tags center;"
    )
    response = await overpass_query(query)
    return _closest_distance(response.get("elements"), lat, lon)


async def _fetch_nearest_water(lat: float, lon: float) -> _WaterCandidate | None:
    lat_text = _format_coord(lat)
    lon_text = _format_coord(lon)
    query = (
        "[out:json][timeout:25];"
        "("
        f'way(around:{_WATER_SEARCH_RADIUS_M},{lat_text},{lon_text})["natural"="water"];'
        f'relation(around:{_WATER_SEARCH_RADIUS_M},{lat_text},{lon_text})["natural"="water"];'
        f'way(around:{_WATER_SEARCH_RADIUS_M},{lat_text},{lon_text})["water"];'
        f'relation(around:{_WATER_SEARCH_RADIUS_M},{lat_text},{lon_text})["water"];'
        ");"
        "out tags center;"
    )
    response = await overpass_query(query)
    elements = response.get("elements")
    if not isinstance(elements, list):
        return None
    best: _WaterCandidate | None = None
    for element in elements:
        candidate = _extract_water_candidate(element, lat, lon)
        if not candidate:
            continue
        if best is None or candidate.distance_m < best.distance_m:
            best = candidate
    return best


def _select_water_result(
    *,
    has_sea_tag: bool,
    coastline_distance: float | None,
    water_candidate: _WaterCandidate | None,
) -> tuple[str | None, str | None]:
    if coastline_distance is None and water_candidate is None:
        return None, None
    if coastline_distance is not None and water_candidate is None:
        if has_sea_tag:
            return _BALTIC_SEA_NAME, "sea"
        return None, None
    if water_candidate and coastline_distance is None:
        return water_candidate.name, water_candidate.kind
    if not water_candidate:
        return None, None
    if coastline_distance is None:
        return water_candidate.name, water_candidate.kind
    if water_candidate.distance_m < coastline_distance:
        return water_candidate.name, water_candidate.kind
    if has_sea_tag and coastline_distance <= water_candidate.distance_m:
        return _BALTIC_SEA_NAME, "sea"
    return water_candidate.name, water_candidate.kind


async def _detect_water(
    lat: float,
    lon: float,
    normalized_tags: set[str],
) -> tuple[str | None, str | None]:
    if not normalized_tags & _WATER_TAG_HINTS:
        return None, None
    has_sea_tag = bool(normalized_tags & _SEA_TAG_HINTS)
    try:
        coastline_distance, water_candidate = await _gather_water_candidates(lat, lon)
    except Exception:  # pragma: no cover - network errors are rare
        logging.exception("GEO_CONTEXT water_error lat=%.5f lon=%.5f", lat, lon)
        return None, None
    return _select_water_result(
        has_sea_tag=has_sea_tag,
        coastline_distance=coastline_distance,
        water_candidate=water_candidate,
    )


async def _gather_water_candidates(
    lat: float,
    lon: float,
) -> tuple[float | None, _WaterCandidate | None]:
    coastline_distance = await _fetch_coastline_distance(lat, lon)
    water_candidate = await _fetch_nearest_water(lat, lon)
    return coastline_distance, water_candidate


async def build_geo_context_for_asset(
    *,
    lat: float | None,
    lon: float | None,
    asset_city: str | None,
    asset_tags: Sequence[str] | None,
) -> GeoContext:
    main_place = _normalize_text(asset_city)
    normalized_tags = _normalize_tag_keys(asset_tags)
    if lat is None or lon is None:
        return GeoContext(
            main_place=main_place,
            water_name=None,
            water_kind=None,
            national_park=None,
        )
    national_park_name: str | None = None
    try:
        park = await find_national_park(lat, lon)
    except Exception:  # pragma: no cover - network errors are rare
        logging.exception("GEO_CONTEXT national_park_error lat=%.5f lon=%.5f", lat, lon)
        park = None
    if park:
        national_park_name = _normalize_text(getattr(park, "short_name", None))
    settlement_name: str | None = None
    if not main_place:
        radius = _SETTLEMENT_RADIUS_IN_PARK_M if national_park_name else _SETTLEMENT_RADIUS_DEFAULT_M
        try:
            settlement = await find_nearest_settlement(lat, lon, radius)
        except Exception:  # pragma: no cover - network errors are rare
            logging.exception("GEO_CONTEXT settlement_error lat=%.5f lon=%.5f", lat, lon)
            settlement = None
        if settlement:
            settlement_name = _normalize_text(getattr(settlement, "name", None))
    water_name: str | None = None
    water_kind: str | None = None
    if normalized_tags & _WATER_TAG_HINTS:
        water_name, water_kind = await _detect_water(lat, lon, normalized_tags)
    return GeoContext(
        main_place=main_place or settlement_name,
        water_name=water_name,
        water_kind=water_kind,
        national_park=national_park_name,
    )


__all__ = ["GeoContext", "build_geo_context_for_asset"]
