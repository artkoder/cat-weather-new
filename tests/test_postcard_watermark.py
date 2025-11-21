import os
import sys

import pytest
from PIL import Image, ImageChops

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
