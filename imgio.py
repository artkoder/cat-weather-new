from __future__ import annotations

import gc
import io
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

try:  # pragma: no cover - optional dependency
    from PIL import ImageCms  # type: ignore
except Exception:  # pragma: no cover - fallback when LittleCMS is unavailable
    ImageCms = None  # type: ignore[assignment]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.warning("Invalid %s=%s, using %s", name, raw, default)
        return default
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logging.warning("Invalid %s=%s, using %s", name, raw, default)
        return default
    return value


def _convert_to_srgb(image: Image.Image) -> Image.Image:
    icc_profile = image.info.get("icc_profile") if hasattr(image, "info") else None
    if icc_profile and ImageCms is not None:
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            dst_profile = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(
                image, src_profile, dst_profile, outputMode="RGB"
            )
        except Exception:
            logging.exception("Failed to convert ICC profile to sRGB")
            converted = image.convert("RGB")
    elif image.mode != "RGB":
        converted = image.convert("RGB")
    else:
        converted = image.copy()
    if converted is image:
        converted = image.copy()
    return converted


def _resize_if_needed(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_side:
        return image
    scale = max_side / float(current_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:  # pragma: no cover - Pillow < 9 compatibility
        resample = Image.LANCZOS  # type: ignore[attr-defined]
    resized = image.resize((new_width, new_height), resample)
    image.close()
    return resized


def _prepare_image(src_path: Path, *, max_side: int | None = None) -> Image.Image:
    with Image.open(src_path) as original:
        transposed = ImageOps.exif_transpose(original)
        converted = _convert_to_srgb(transposed)
        if converted is not transposed:
            transposed.close()
        image = converted
        if max_side is not None:
            image = _resize_if_needed(image, max_side)
        return image


def _create_temp_file(prefix: str) -> Path:
    fd, temp_name = tempfile.mkstemp(prefix=prefix, suffix=".jpg")
    os.close(fd)
    return Path(temp_name)


def _iter_quality_steps(min_quality: int) -> list[int]:
    base_steps = [92, 88, 86]
    qualities = [q for q in base_steps if q >= min_quality]
    if min_quality > max(qualities, default=0):
        qualities.append(min_quality)
    if min_quality not in qualities:
        qualities.append(min_quality)
    # ensure descending order and remove duplicates while preserving order
    seen: set[int] = set()
    ordered: list[int] = []
    for value in qualities:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    ordered.sort(reverse=True)
    if ordered[-1] != min_quality:
        ordered.append(min_quality)
    return ordered


def ensure_photo_file_for_telegram(
    src_path: str | os.PathLike[str] | Path,
) -> Path:
    source_path = Path(src_path)
    target_mb = _env_float("ASSETS_PUB_TARGET_MB", 9.5)
    min_quality = _env_int("ASSETS_PUB_MIN_QUALITY", 86)
    target_bytes = int(target_mb * 1024 * 1024)
    try:
        if source_path.suffix.lower() in {".jpg", ".jpeg"}:
            if source_path.stat().st_size <= target_bytes:
                return source_path
    except FileNotFoundError:
        raise
    image = _prepare_image(source_path, max_side=None)
    temp_path = _create_temp_file("tg-")
    try:
        for quality in _iter_quality_steps(min_quality):
            image.save(
                temp_path,
                format="JPEG",
                quality=quality,
                progressive=True,
                exif=b"",
            )
            if temp_path.stat().st_size <= target_bytes:
                break
        else:
            logging.warning(
                "Compressed photo remains above %s bytes at min quality %s", target_bytes, min_quality
            )
    finally:
        image.close()
        gc.collect()
    return temp_path

