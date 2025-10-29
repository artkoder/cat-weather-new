from __future__ import annotations

import io
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import piexif
from PIL import ExifTags, Image

ExifreadFieldType: Any | None = None

try:  # pragma: no cover - optional dependency
    import exifread  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing in some environments
    exifread = None  # type: ignore[assignment]
else:  # pragma: no cover - optional dependency helpers
    try:
        from exifread.core.ifd_tag import FieldType as ExifreadFieldType  # type: ignore
    except Exception:
        ExifreadFieldType = None

_HEIF_REGISTERED = False
_HEIF_IMPORT_FAILED = False


@dataclass(slots=True)
class PhotoMeta:
    captured_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    make: str | None = None
    model: str | None = None
    orientation: int | None = None
    raw_exif: dict[str, Any] = field(default_factory=dict)
    raw_gps: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"


def extract_metadata_from_file(
    path_or_bytes: str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO,
) -> tuple[PhotoMeta, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Extract metadata from image bytes or path.

    Returns a tuple of ``(PhotoMeta, exif_payload, gps_payload, exif_ifds)``.
    """

    data, origin = _read_source_bytes(path_or_bytes)
    if not data:
        return PhotoMeta(source="empty"), {}, {}, {"0th": {}, "Exif": {}, "GPS": {}}

    format_hint = _detect_image_format(data)
    if format_hint == "HEIC":
        _ensure_heif_registered()

    exif_payload: dict[str, Any] = {}
    gps_info: dict[str, Any] = {}
    exif_ifds: dict[str, dict[str, Any]] = {}
    source = "unknown"

    exif_dict: dict[str, Any] | None = None
    exif_bytes: bytes | None = None
    pil_exif_cached: Mapping[int, Any] | None = None

    with Image.open(io.BytesIO(data)) as image:
        try:
            image.load()
        except Exception:
            logging.exception("Failed to load image for metadata extraction")
        exif_bytes = getattr(image, "info", {}).get("exif")
        pil_exif_cached = image.getexif() if hasattr(image, "getexif") else None

    piexif_used_embedded_bytes = False
    piexif_used_full_image = False

    if exif_bytes:
        try:
            exif_dict = piexif.load(exif_bytes)
            source = "piexif"
            piexif_used_embedded_bytes = True
        except Exception:
            logging.debug("piexif failed on embedded bytes", exc_info=True)
            exif_dict = None

    if exif_dict is None and format_hint in {"JPEG", "TIFF", "WEBP"}:
        try:
            exif_dict = piexif.load(data)
            source = "piexif"
            piexif_used_full_image = True
        except Exception:
            logging.debug("piexif failed on full image", exc_info=True)
            exif_dict = None

    if exif_dict:
        exif_payload, gps_info, exif_ifds = _extract_from_piexif(exif_dict)
        gps_issue = _validate_gps_payload(gps_info)
        if gps_issue:
            _clear_gps_blocks(exif_payload, exif_ifds)
            gps_info = {}

            retry_issue = gps_issue
            if piexif_used_embedded_bytes and not piexif_used_full_image:
                logging.warning(
                    "piexif returned invalid GPS metadata (%s) from embedded EXIF; retrying with full image bytes",
                    gps_issue,
                )
                try:
                    retry_exif_dict = piexif.load(data)
                    piexif_used_full_image = True
                except Exception:
                    logging.debug("piexif failed on full image during GPS retry", exc_info=True)
                else:
                    retry_payload, retry_gps, retry_ifds = _extract_from_piexif(retry_exif_dict)
                    retry_issue = _validate_gps_payload(retry_gps)
                    if retry_issue:
                        _clear_gps_blocks(retry_payload, retry_ifds)
                        exif_payload = _merge_exif_payloads(exif_payload, retry_payload)
                        exif_ifds = _merge_exif_ifds(
                            exif_ifds,
                            {name: values for name, values in retry_ifds.items() if name != "GPS"},
                        )
                    else:
                        exif_payload = retry_payload
                        gps_info = retry_gps
                        exif_ifds = retry_ifds
                        retry_issue = None

            if retry_issue:
                logging.warning(
                    "piexif returned invalid GPS metadata (%s); retrying with fallback parser",
                    retry_issue,
                )
                fallback_payload: dict[str, Any] = {}
                fallback_gps: dict[str, Any] = {}
                fallback_ifds: dict[str, dict[str, Any]] = {}
                fallback_source: str | None = None

                if pil_exif_cached:
                    pillow_exif, pillow_gps, pillow_ifds = _extract_from_pillow_exif(pil_exif_cached)
                    if pillow_gps:
                        fallback_payload = dict(pillow_exif)
                        fallback_ifds = {name: dict(values) for name, values in pillow_ifds.items()}
                        fallback_gps = dict(pillow_gps)
                        fallback_source = "pillow"
                        pillow_issue = _validate_gps_payload(fallback_gps)
                        if pillow_issue:
                            logging.warning(
                                "pillow returned invalid GPS metadata (%s); continuing to next fallback",
                                pillow_issue,
                            )
                            fallback_gps.clear()
                            fallback_source = None
                            _clear_gps_blocks(fallback_payload, fallback_ifds)

                if retry_issue and (not fallback_gps) and exifread is not None:
                    try:
                        exifread_payload, exifread_gps, exifread_ifds = _extract_with_exifread(data)
                    except Exception:
                        logging.debug("exifread failed during GPS fallback", exc_info=True)
                    else:
                        if exifread_payload or exifread_gps:
                            fallback_payload = dict(exifread_payload)
                            fallback_gps = dict(exifread_gps)
                            fallback_ifds = {
                                name: dict(values) for name, values in exifread_ifds.items()
                            }
                            fallback_source = "exifread"
                            exifread_issue = _validate_gps_payload(fallback_gps)
                            if exifread_issue:
                                logging.warning(
                                    "exifread returned invalid GPS metadata (%s); ignoring fallback",
                                    exifread_issue,
                                )
                                fallback_gps.clear()
                                fallback_source = None
                                _clear_gps_blocks(fallback_payload, fallback_ifds)

                if fallback_payload:
                    exif_payload = _merge_exif_payloads(exif_payload, fallback_payload)
                if fallback_ifds:
                    exif_ifds = _merge_exif_ifds(exif_ifds, fallback_ifds)
                if fallback_gps:
                    gps_info = dict(fallback_gps)
                    exif_payload["GPSInfo"] = gps_info
                    exif_payload["GPS"] = gps_info
                    exif_ifds.setdefault("GPS", {})
                    exif_ifds["GPS"].update(fallback_ifds.get("GPS", {}))
                    if fallback_source:
                        source = fallback_source
        if (not gps_info) and pil_exif_cached:
            pillow_exif, pillow_gps, pillow_ifds = _extract_from_pillow_exif(pil_exif_cached)
            if pillow_gps and not _validate_gps_payload(pillow_gps):
                gps_info = pillow_gps
                merged_ifds: dict[str, dict[str, Any]] = {}
                merged_ifds.update(exif_ifds)
                for name, values in pillow_ifds.items():
                    existing = merged_ifds.get(name) or {}
                    updated = dict(existing)
                    updated.update(values)
                    merged_ifds[name] = updated
                exif_ifds = merged_ifds
                if not exif_payload.get("GPSInfo"):
                    exif_payload["GPSInfo"] = pillow_gps
                if not exif_payload.get("GPS"):
                    exif_payload["GPS"] = pillow_gps
                for key, value in pillow_exif.items():
                    exif_payload.setdefault(key, value)
                source = "pillow"
    else:
        with Image.open(io.BytesIO(data)) as image:
            pil_exif = image.getexif() if hasattr(image, "getexif") else None
            exif_payload, gps_info, exif_ifds = _extract_from_pillow_exif(pil_exif)
            if exif_payload:
                source = "pillow"

    if not exif_payload and exifread is not None:
        try:
            exif_payload, gps_info, exif_ifds = _extract_with_exifread(data)
            if exif_payload or gps_info:
                source = "exifread"
        except Exception:
            logging.debug("exifread failed", exc_info=True)

    photo_meta = _build_photo_meta(
        exif_payload=exif_payload,
        gps_info=gps_info,
        exif_ifds=exif_ifds,
        source=source,
    )

    gps_payload = _build_gps_payload(photo_meta)

    for required_ifd in ("0th", "Exif", "GPS"):
        exif_ifds.setdefault(required_ifd, {})

    return photo_meta, exif_payload, gps_payload, exif_ifds


def _read_source_bytes(
    path_or_bytes: str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO,
) -> tuple[bytes, str | None]:
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        return (bytes(path_or_bytes), None)
    if isinstance(path_or_bytes, (str, os.PathLike)):
        path = Path(path_or_bytes)
        return (path.read_bytes(), str(path))
    if hasattr(path_or_bytes, "read"):
        data = path_or_bytes.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return (data or b"", getattr(path_or_bytes, "name", None))
    raise TypeError(f"Unsupported input type: {type(path_or_bytes)!r}")


def _detect_image_format(data: bytes) -> str | None:
    if len(data) >= 12 and data[:4] == b"\xff\xd8\xff\xe0" or data[:4] == b"\xff\xd8\xff\xe1":
        return "JPEG"
    if len(data) >= 4 and data[:4] in {b"II*\x00", b"MM\x00*"}:
        return "TIFF"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "HEIC"
    return None


def _ensure_heif_registered() -> None:
    global _HEIF_REGISTERED, _HEIF_IMPORT_FAILED
    if _HEIF_REGISTERED or _HEIF_IMPORT_FAILED:
        return
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
        _HEIF_REGISTERED = True
    except Exception:
        logging.debug("Unable to register pillow_heif", exc_info=True)
        _HEIF_IMPORT_FAILED = True


def _extract_from_piexif(
    exif_dict: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    exif_payload: dict[str, Any] = {}
    gps_info: dict[str, Any] = {}
    exif_ifds: dict[str, dict[str, Any]] = {}

    for ifd_name in ("0th", "Exif", "1st"):
        source_ifd = exif_dict.get(ifd_name)
        if isinstance(source_ifd, Mapping):
            exif_ifds[ifd_name] = _serialize_exif_ifd(ifd_name, source_ifd)
            tag_map = piexif.TAGS.get(ifd_name, {})
            for tag_id, raw_value in source_ifd.items():
                tag_info = tag_map.get(tag_id) or {}
                tag_name = tag_info.get("name") or ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_payload[tag_name] = _normalize_exif_value(raw_value)

    gps_ifd = exif_dict.get("GPS")
    if isinstance(gps_ifd, Mapping):
        exif_ifds["GPS"] = _serialize_exif_ifd("GPS", gps_ifd)
        gps_info = {
            ExifTags.GPSTAGS.get(int(sub_id), str(sub_id)): _normalize_exif_value(sub_val)
            for sub_id, sub_val in gps_ifd.items()
        }
        exif_payload["GPSInfo"] = gps_info
        exif_payload["GPS"] = gps_info

    return exif_payload, gps_info, exif_ifds


def _validate_gps_payload(gps_info: Mapping[str, Any] | None) -> str | None:
    if not gps_info:
        return None

    lat_values = gps_info.get("GPSLatitude")
    lon_values = gps_info.get("GPSLongitude")
    lat_ref = _normalize_gps_ref(gps_info.get("GPSLatitudeRef"))
    lon_ref = _normalize_gps_ref(gps_info.get("GPSLongitudeRef"))
    valid_refs = {"N", "S", "E", "W"}

    if lat_values and (not lat_ref or lat_ref.strip().upper()[:1] not in valid_refs):
        return "invalid latitude reference"
    if lon_values and (not lon_ref or lon_ref.strip().upper()[:1] not in valid_refs):
        return "invalid longitude reference"

    if not lat_values and not lon_values:
        return None

    latitude, longitude = _extract_gps_decimal(gps_info)
    if lat_values and latitude is None:
        return "unable to decode latitude"
    if lon_values and longitude is None:
        return "unable to decode longitude"
    return None


def _merge_exif_payloads(
    base: Mapping[str, Any], updates: Mapping[str, Any]
) -> dict[str, Any]:
    if not updates:
        return dict(base)
    merged = dict(base)
    for key, value in updates.items():
        if key in {"GPS", "GPSInfo"}:
            merged[key] = value
        elif key not in merged or merged[key] in (None, "", {}):
            merged[key] = value
    return merged


def _merge_exif_ifds(
    base: Mapping[str, Mapping[str, Any]], updates: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not updates:
        return {key: dict(value) for key, value in base.items()}
    merged: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in base.items()
    }
    for ifd_name, ifd_values in updates.items():
        existing = merged.get(ifd_name, {})
        combined = dict(existing)
        combined.update(ifd_values)
        merged[ifd_name] = combined
    return merged


def _clear_gps_blocks(
    exif_payload: dict[str, Any], exif_ifds: dict[str, dict[str, Any]]
) -> None:
    exif_payload.pop("GPSInfo", None)
    exif_payload.pop("GPS", None)
    gps_ifd = exif_ifds.pop("GPS", None)
    if isinstance(gps_ifd, dict):  # pragma: no cover - defensive
        gps_ifd.clear()


def _extract_from_pillow_exif(
    exif_data: Mapping[int, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    if not exif_data:
        return {}, {}, {}

    exif_payload: dict[str, Any] = {}
    gps_info: dict[str, Any] = {}
    exif_ifds: dict[str, dict[str, Any]] = {}

    for tag_id, raw_value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if tag_name == "GPSInfo" and isinstance(raw_value, Mapping):
            gps_info = {
                ExifTags.GPSTAGS.get(sub_id, str(sub_id)): _normalize_exif_value(sub_val)
                for sub_id, sub_val in raw_value.items()
            }
            exif_payload["GPSInfo"] = gps_info
            exif_payload["GPS"] = gps_info
            exif_ifds["GPS"] = {
                ExifTags.GPSTAGS.get(sub_id, str(sub_id)): _serialize_exif_raw_value(sub_val)
                for sub_id, sub_val in raw_value.items()
            }
        else:
            exif_payload[tag_name] = _normalize_exif_value(raw_value)

    if gps_info and "GPS" not in exif_ifds:
        exif_ifds["GPS"] = {k: _serialize_exif_raw_value(v) for k, v in gps_info.items()}

    return exif_payload, gps_info, exif_ifds


def _is_ascii_exif_field(field: Any) -> bool:
    field_type = getattr(field, "field_type", None)
    if ExifreadFieldType is not None:
        try:
            if field_type == ExifreadFieldType.ASCII:
                return True
        except Exception:  # pragma: no cover - defensive against unexpected field types
            pass
    if isinstance(field_type, str):
        return field_type.upper() == "ASCII"
    if isinstance(field_type, int):
        return field_type == 2
    return False


def _should_use_ascii_printable(tag_name: str, field: Any, printable_value: Any) -> bool:
    if not isinstance(tag_name, str):
        return False
    if not isinstance(printable_value, str) or not printable_value:
        return False
    if not (tag_name.startswith("GPS ") or tag_name.startswith("EXIF ")):
        return False
    return _is_ascii_exif_field(field)


def _extract_with_exifread(
    data: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    if exifread is None:  # pragma: no cover - guarded earlier
        return {}, {}, {}

    tags = exifread.process_file(io.BytesIO(data), details=False)
    exif_payload: dict[str, Any] = {}
    gps_info: dict[str, Any] = {}
    exif_ifds: dict[str, dict[str, Any]] = {}

    for tag_name, field in tags.items():
        values: Any
        if hasattr(field, "values"):
            values = field.values
        else:
            values = field

        printable_attr = getattr(field, "printable", values)
        printable_normalized: Any | None = None
        if isinstance(printable_attr, str):
            stripped_printable = printable_attr.replace("\x00", "").strip()
            if stripped_printable:
                printable_normalized = _normalize_exif_value(stripped_printable)

        normalized = _normalize_exif_value(values)
        raw_serialized = _serialize_exif_raw_value(values)

        if (
            printable_normalized is not None
            and _should_use_ascii_printable(tag_name, field, printable_normalized)
        ):
            normalized = printable_normalized
            raw_serialized = _serialize_exif_raw_value(printable_normalized)
            printable_value: Any = printable_normalized
        else:
            printable_value = printable_attr

        if tag_name.startswith("GPS "):
            key = tag_name[4:]
            gps_info[key] = normalized
            exif_ifds.setdefault("GPS", {})[key] = raw_serialized
        elif tag_name.startswith("EXIF "):
            key = tag_name[5:]
            exif_payload[key] = normalized
            exif_ifds.setdefault("Exif", {})[key] = raw_serialized
        elif tag_name.startswith("Image "):
            key = tag_name[6:]
            exif_payload[key] = normalized
            exif_ifds.setdefault("0th", {})[key] = raw_serialized
        else:
            exif_payload[tag_name] = printable_value

    if gps_info:
        exif_payload["GPSInfo"] = gps_info
        exif_payload["GPS"] = gps_info

    return exif_payload, gps_info, exif_ifds


def _build_photo_meta(
    *,
    exif_payload: Mapping[str, Any],
    gps_info: Mapping[str, Any],
    exif_ifds: Mapping[str, Mapping[str, Any]],
    source: str,
) -> PhotoMeta:
    photo = PhotoMeta(source=source or "unknown")

    photo.make = _as_optional_str(exif_payload.get("Make"))
    photo.model = _as_optional_str(exif_payload.get("Model"))
    photo.orientation = _as_optional_int(exif_payload.get("Orientation"))

    latitude, longitude = _extract_gps_decimal(dict(gps_info)) if gps_info else (None, None)
    if latitude is not None:
        photo.latitude = round(latitude, 7)
    if longitude is not None:
        photo.longitude = round(longitude, 7)

    altitude = _extract_altitude(gps_info)
    if altitude is not None:
        photo.altitude = altitude

    captured_at = _select_captured_at(exif_payload, gps_info)
    photo.captured_at = captured_at

    photo.raw_exif = {
        key: dict(value) for key, value in exif_ifds.items() if key != "GPS"
    }
    photo.raw_gps = dict(exif_ifds.get("GPS", {}))

    return photo


def _build_gps_payload(photo: PhotoMeta) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if photo.latitude is not None:
        payload["latitude"] = photo.latitude
    if photo.longitude is not None:
        payload["longitude"] = photo.longitude
    if photo.altitude is not None:
        payload["altitude"] = photo.altitude
    if photo.captured_at is not None:
        payload["captured_at"] = photo.captured_at.isoformat()
    return payload


def _extract_altitude(gps_info: Mapping[str, Any]) -> float | None:
    altitude = gps_info.get("GPSAltitude")
    if altitude is None:
        return None

    value = _to_float_ratio(altitude)
    if value is None:
        try:
            value = float(altitude)
        except Exception:
            return None

    ref = gps_info.get("GPSAltitudeRef")
    ref_value = None
    if isinstance(ref, (bytes, bytearray)) and ref:
        ref_value = ref[0]
    elif isinstance(ref, (Sequence, list, tuple)) and ref:
        first = ref[0]
        if isinstance(first, (int, float)):
            ref_value = int(first)
    elif isinstance(ref, str):
        ref = ref.strip()
        if ref.isdigit():
            ref_value = int(ref)
    elif isinstance(ref, (int, float)):
        ref_value = int(ref)

    if ref_value == 1:
        value *= -1.0

    return value


def _select_captured_at(
    exif_payload: Mapping[str, Any], gps_info: Mapping[str, Any]
) -> datetime | None:
    gps_capture = _combine_gps_timestamp(gps_info)
    if gps_capture:
        return gps_capture

    candidates = [
        exif_payload.get("DateTimeOriginal"),
        exif_payload.get("DateTimeDigitized"),
        exif_payload.get("DateTime"),
    ]
    for value in candidates:
        dt = _parse_exif_datetime(value)
        if dt:
            return dt
    return None


def _combine_gps_timestamp(gps_info: Mapping[str, Any]) -> datetime | None:
    date_raw = gps_info.get("GPSDateStamp")
    time_raw = gps_info.get("GPSTimeStamp")
    if not date_raw or not time_raw:
        return None

    date_text = _as_optional_str(date_raw)
    if not date_text:
        return None

    date_text = date_text.replace("-", ":")
    parts = [part for part in date_text.split(":") if part]
    if len(parts) != 3:
        return None
    try:
        year, month, day = [int(part) for part in parts]
    except Exception:
        return None

    hours, minutes, seconds = _normalize_gps_time(time_raw)
    if hours is None:
        return None

    microsecond = int(round((seconds - int(seconds)) * 1_000_000))
    seconds_int = int(seconds)
    return datetime(
        year,
        month,
        day,
        int(hours),
        int(minutes),
        seconds_int,
        microsecond,
        tzinfo=timezone.utc,
    )


def _normalize_gps_time(value: Any) -> tuple[float | None, float | None, float | None]:
    if value is None:
        return (None, None, None)

    values: Sequence[Any]
    if isinstance(value, str):
        tokens = [token for token in re.split(r"[\s:]+", value.strip()) if token]
        values = tokens
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = [value]

    numbers: list[float] = []
    for part in values:
        ratio = _to_float_ratio(part)
        if ratio is None:
            try:
                ratio = float(part)
            except Exception:
                return (None, None, None)
        numbers.append(ratio)

    while len(numbers) < 3:
        numbers.append(0.0)

    hours, minutes, seconds = numbers[:3]
    if hours is None or minutes is None or seconds is None:
        return (None, None, None)

    return (hours, minutes, seconds)


def _parse_exif_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip().strip("\x00")
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc)
    return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        for encoding in ("utf-8", "ascii", "latin-1"):
            try:
                decoded = value.decode(encoding, errors="ignore").strip()
            except Exception:
                continue
            if decoded:
                return decoded
        return None
    text = str(value).strip().strip("\x00")
    return text or None


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return None


_RATIO_SPLIT_PATTERN = re.compile(r"[\s,]+")


def _parse_ratio_token(token: str) -> float | None:
    stripped = token.strip()
    if not stripped:
        return None
    if "/" in stripped:
        numerator_text, denominator_text = stripped.split("/", 1)
        try:
            numerator_value = float(numerator_text)
            denominator_value = float(denominator_text)
        except Exception:
            return None
        if denominator_value == 0:
            return None
        return numerator_value / denominator_value
    try:
        return float(stripped)
    except Exception:
        return None


def _extract_ratio_numbers(text: str) -> list[float] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if "/" not in stripped and "," not in stripped:
        return None
    tokens = [token for token in _RATIO_SPLIT_PATTERN.split(stripped) if token]
    if not tokens:
        return None
    numbers: list[float] = []
    for token in tokens:
        parsed = _parse_ratio_token(token)
        if parsed is None:
            return None
        numbers.append(parsed)
    return numbers


def _to_float_ratio(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        ratio_numbers = _extract_ratio_numbers(stripped)
        if ratio_numbers and len(ratio_numbers) == 1:
            return ratio_numbers[0]
        if "/" in stripped:
            parts = stripped.split("/", 1)
            if len(parts) == 2:
                try:
                    numerator_value = float(parts[0])
                    denominator_value = float(parts[1])
                except Exception:
                    return None
                if denominator_value == 0:
                    return None
                return numerator_value / denominator_value
        try:
            return float(stripped)
        except Exception:
            return None
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(numerator, int) and isinstance(denominator, int):
        if denominator == 0:
            return None
        return numerator / denominator
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        if len(value) == 2:
            first, second = value
            first_value = _to_float_ratio(first)
            if first_value is None and isinstance(first, (int, float)):
                first_value = float(first)
            second_value = _to_float_ratio(second)
            if second_value is None and isinstance(second, (int, float)):
                second_value = float(second)
            if (
                first_value is not None
                and second_value not in {None, 0}
                and isinstance(second_value, (int, float))
                and second_value != 0
            ):
                return first_value / float(second_value)
        # Attempt to compute a simple average if sequence is already flattened
        flattened: list[float] = []
        for item in value:
            item_ratio = _to_float_ratio(item)
            if item_ratio is not None:
                flattened.append(item_ratio)
            elif isinstance(item, (int, float)):
                flattened.append(float(item))
        if len(flattened) == 2 and flattened[1] != 0:
            return flattened[0] / flattened[1]
    return None


def _normalize_exif_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        for encoding in ("utf-8", "ascii", "latin-1"):
            try:
                decoded = value.decode(encoding, errors="ignore").strip("\x00").strip()
            except Exception:
                continue
            if decoded:
                return decoded
        return value.hex()
    if isinstance(value, str):
        value = value.strip().strip("\x00")
        ratio_numbers = _extract_ratio_numbers(value)
        if ratio_numbers:
            return ratio_numbers if len(ratio_numbers) > 1 else ratio_numbers[0]
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_normalize_exif_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _normalize_exif_value(v) for k, v in value.items()}
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(numerator, int) and isinstance(denominator, int):
        return [numerator, denominator]
    try:
        return float(value)
    except Exception:
        return str(value)


def _serialize_exif_raw_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(numerator, int) and isinstance(denominator, int):
        return [numerator, denominator]
    if isinstance(value, tuple):
        return [_serialize_exif_raw_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_exif_raw_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_serialize_exif_raw_value(item) for item in value)
    if isinstance(value, Mapping):
        return {str(k): _serialize_exif_raw_value(v) for k, v in value.items()}
    return str(value)


def _serialize_exif_ifd(
    ifd_name: str, source_ifd: Mapping[int, Any]
) -> dict[str, Any]:
    tag_map = piexif.TAGS.get(ifd_name, {})
    result: dict[str, Any] = {}
    for tag_id, raw_value in source_ifd.items():
        if ifd_name == "GPS":
            tag_name = ExifTags.GPSTAGS.get(int(tag_id), str(tag_id))
        else:
            tag_info = tag_map.get(tag_id) or {}
            tag_name = tag_info.get("name") or ExifTags.TAGS.get(tag_id, str(tag_id))
        result[tag_name] = _serialize_exif_raw_value(raw_value)
    return result


def _normalize_gps_ref(ref: Any) -> str | None:
    if ref is None:
        return None
    if isinstance(ref, (bytes, bytearray)):
        for encoding in ("utf-8", "ascii", "latin-1"):
            try:
                decoded = ref.decode(encoding, errors="ignore").strip()
            except Exception:
                continue
            if decoded:
                ref = decoded
                break
        else:
            return None
    text = str(ref).strip()
    if not text:
        return None
    if len(text) >= 3 and text[0] == "b" and text[1] in {'"', "'"} and text[-1] == text[1]:
        inner = text[2:-1].strip()
        if inner:
            text = inner
    return text or None


def _extract_gps_decimal(gps_info: Mapping[str, Any]) -> tuple[float | None, float | None]:
    def _coerce_sequence(value: Any) -> list[Any]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            ratio_numbers = _extract_ratio_numbers(stripped)
            if ratio_numbers:
                return ratio_numbers
            if re.search(r"[\s,]", stripped):
                tokens = [token for token in re.split(r"[\s,]+", stripped) if token]
                if tokens:
                    return tokens
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        return [value]

    def _to_decimal(values: Any, ref: str | None) -> float | None:
        if not values:
            return None
        raw_parts = _coerce_sequence(values)
        numeric_parts: list[float] = []
        for raw_part in raw_parts:
            if isinstance(raw_part, str):
                ratio_values = _extract_ratio_numbers(raw_part)
                if ratio_values:
                    numeric_parts.extend(ratio_values)
                    continue
            normalized = _normalize_exif_value(raw_part)
            number: float | None = None
            if isinstance(normalized, (int, float)):
                number = float(normalized)
            else:
                ratio_value = _to_float_ratio(normalized)
                if ratio_value is None:
                    ratio_value = _to_float_ratio(raw_part)
                if ratio_value is not None:
                    number = ratio_value
            if number is None:
                try:
                    number = float(normalized)
                except Exception:
                    return None
            numeric_parts.append(number)
        while len(numeric_parts) < 3:
            numeric_parts.append(0.0)
        decimal = numeric_parts[0] + numeric_parts[1] / 60.0 + numeric_parts[2] / 3600.0
        if ref:
            ref_letter = ref.strip().upper()[:1]
            if ref_letter in {"S", "W"}:
                decimal *= -1.0
        return decimal

    lat_ref = _normalize_gps_ref(gps_info.get("GPSLatitudeRef"))
    lon_ref = _normalize_gps_ref(gps_info.get("GPSLongitudeRef"))
    lat = _to_decimal(gps_info.get("GPSLatitude") or [], lat_ref)
    lon = _to_decimal(gps_info.get("GPSLongitude") or [], lon_ref)
    return lat, lon


__all__ = ["PhotoMeta", "extract_metadata_from_file"]
