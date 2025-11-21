from __future__ import annotations

from pathlib import Path
from typing import Final

from PIL import Image

_RESAMPLING = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS  # type: ignore[attr-defined]
_MAX_WIDTH_RATIO: Final[float] = 0.8

WATERMARK_PATH = Path(__file__).resolve().parent / "assets" / "watermarks" / "LoveKaliningrad.png"

__all__ = ["WATERMARK_PATH", "add_love_kaliningrad_watermark"]


def _load_watermark_image() -> Image.Image:
    with Image.open(WATERMARK_PATH) as watermark_src:
        watermark = watermark_src.convert("RGBA").copy()
    return watermark


def _resize_watermark_if_needed(watermark: Image.Image, base_size: tuple[int, int]) -> Image.Image:
    base_width, base_height = base_size
    max_width = max(1, int(base_width * _MAX_WIDTH_RATIO))
    scale = 1.0
    if watermark.width > max_width:
        scale = min(scale, max_width / float(watermark.width))
    if watermark.height > base_height:
        scale = min(scale, base_height / float(watermark.height))
    if scale < 1.0:
        new_width = max(1, int(round(watermark.width * scale)))
        new_height = max(1, int(round(watermark.height * scale)))
        resized = watermark.resize((new_width, new_height), _RESAMPLING)
        watermark.close()
        return resized
    return watermark


def add_love_kaliningrad_watermark(image: Image.Image) -> Image.Image:
    """Overlay the LoveKaliningrad watermark on a copy of the provided image."""

    if image.mode in {"RGB", "RGBA"}:
        working = image.copy()
    else:
        working = image.convert("RGB")
    try:
        base_rgba = working.convert("RGBA")
    finally:
        working.close()
    watermark = _resize_watermark_if_needed(_load_watermark_image(), base_rgba.size)
    composed = base_rgba.copy()
    offset_x = max(0, (base_rgba.width - watermark.width) // 2)
    offset_y = max(0, base_rgba.height - watermark.height)
    composed.paste(watermark, (offset_x, offset_y), watermark)
    result = composed.convert("RGB")
    watermark.close()
    composed.close()
    base_rgba.close()
    return result
