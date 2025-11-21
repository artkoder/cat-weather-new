import os
import sys

import pytest
from PIL import Image, ImageChops

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import postcard_watermark  # noqa: E402
from postcard_watermark import WATERMARK_PATH, add_love_kaliningrad_watermark  # noqa: E402


def test_love_kaliningrad_watermark_exists() -> None:
    assert (
        WATERMARK_PATH.is_file()
    ), "LoveKaliningrad watermark is missing. Place assets/watermarks/LoveKaliningrad.png manually."


def test_add_love_kaliningrad_watermark_modifies_bottom_center() -> None:
    if not WATERMARK_PATH.is_file():
        pytest.skip("LoveKaliningrad watermark asset is required for this test.")
    base_image = Image.new("RGB", (1920, 2560), color=(24, 48, 96))
    result_image: Image.Image | None = None
    try:
        result_image = add_love_kaliningrad_watermark(base_image)
        assert result_image.size == base_image.size
        diff = ImageChops.difference(base_image, result_image).convert("L")
        width, height = diff.size
        region_height = max(10, height // 6)
        region_width = max(10, width // 5)
        left = max(0, (width - region_width) // 2)
        bottom_region = diff.crop((left, height - region_height, left + region_width, height))
        extrema = bottom_region.getextrema()
        assert extrema is not None
        assert extrema[1] > 0, "Bottom-center pixels should change after applying the watermark"
    finally:
        base_image.close()
        if result_image is not None:
            result_image.close()


def test_add_love_kaliningrad_watermark_keeps_large_size(monkeypatch: pytest.MonkeyPatch) -> None:
    base_image = Image.new("RGB", (320, 200), color=(0, 0, 0))
    watermark_size = (400, 120)
    result_image: Image.Image | None = None

    def fake_load() -> Image.Image:
        return Image.new("RGBA", watermark_size, color=(255, 255, 255, 255))

    monkeypatch.setattr(postcard_watermark, "_load_watermark_image", fake_load, raising=False)

    try:
        result_image = add_love_kaliningrad_watermark(base_image)
        diff = ImageChops.difference(base_image, result_image).convert("L")
        bbox = diff.getbbox()
        assert bbox is not None
        assert bbox == (0, base_image.height - watermark_size[1], base_image.width, base_image.height)
    finally:
        base_image.close()
        if result_image is not None:
            result_image.close()


def test_add_love_kaliningrad_watermark_allows_negative_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_image = Image.new("RGB", (200, 120), color=(4, 8, 12))
    watermark_size = (260, 40)
    result_image: Image.Image | None = None

    def fake_load() -> Image.Image:
        return Image.new("RGBA", watermark_size, color=(255, 0, 0, 255))

    monkeypatch.setattr(postcard_watermark, "_load_watermark_image", fake_load, raising=False)

    recorded_box: tuple[int, int] | None = None
    original_paste = Image.Image.paste

    def capture_paste(
        self: Image.Image, img: Image.Image, box: object = None, mask: Image.Image | None = None
    ) -> Image.Image | None:
        nonlocal recorded_box
        if mask is img and recorded_box is None:
            if isinstance(box, tuple) and len(box) == 2:
                x, y = box
                if isinstance(x, int) and isinstance(y, int):
                    recorded_box = (x, y)
        return original_paste(self, img, box, mask)

    monkeypatch.setattr(Image.Image, "paste", capture_paste, raising=False)

    try:
        result_image = add_love_kaliningrad_watermark(base_image)
        assert recorded_box == (
            (base_image.width - watermark_size[0]) // 2,
            base_image.height - watermark_size[1],
        )
    finally:
        base_image.close()
        if result_image is not None:
            result_image.close()
