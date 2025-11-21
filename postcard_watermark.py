from __future__ import annotations

from pathlib import Path

from PIL import Image

WATERMARK_PATH = Path(__file__).resolve().parent / "assets" / "watermarks" / "LoveKaliningrad.png"

__all__ = ["WATERMARK_PATH", "add_love_kaliningrad_watermark"]


def _load_watermark_image() -> Image.Image:
    with Image.open(WATERMARK_PATH) as watermark_src:
        watermark = watermark_src.convert("RGBA").copy()
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

    watermark = _load_watermark_image()
    composed = base_rgba.copy()
    offset_x = (base_rgba.width - watermark.width) // 2
    offset_y = base_rgba.height - watermark.height
    composed.paste(watermark, (offset_x, offset_y), watermark)
    result = composed.convert("RGB")
    watermark.close()
    composed.close()
    base_rgba.close()
    return result
