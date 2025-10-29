from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metadata.extractor import extract_metadata_from_file


class _FakeExifField:
    def __init__(self, *, printable: str, values: Any, field_type: Any) -> None:
        self.printable = printable
        self.values = values
        self.field_type = field_type


def test_exifread_ascii_printable_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("exifread")
    from exifread.core.ifd_tag import FieldType

    image_path = tmp_path / "bare.jpg"
    Image.new("RGB", (4, 4), color=(0, 128, 255)).save(image_path, format="JPEG")

    def _broken_load(*args: Any, **kwargs: Any):  # type: ignore[override]
        raise RuntimeError("piexif unavailable")

    monkeypatch.setattr("metadata.extractor.piexif.load", _broken_load)

    fake_tags = {
        "GPS GPSLatitudeRef": _FakeExifField(
            printable="N",
            values=[78, 0],
            field_type=FieldType.ASCII,
        ),
        "GPS GPSLatitude": _FakeExifField(
            printable="55/1 30/1 0/1",
            values=[(55, 1), (30, 1), (0, 1)],
            field_type=FieldType.RATIO,
        ),
        "GPS GPSLongitudeRef": _FakeExifField(
            printable="E",
            values=[69, 0],
            field_type=FieldType.ASCII,
        ),
        "GPS GPSLongitude": _FakeExifField(
            printable="37/1 36/1 0/1",
            values=[(37, 1), (36, 1), (0, 1)],
            field_type=FieldType.RATIO,
        ),
    }

    def _fake_process_file(*args: Any, **kwargs: Any):  # type: ignore[override]
        return fake_tags

    monkeypatch.setattr("metadata.extractor.exifread.process_file", _fake_process_file)

    photo_meta, exif_payload, gps_payload, _ = extract_metadata_from_file(image_path.read_bytes())

    assert photo_meta.source == "exifread"
    assert photo_meta.latitude == pytest.approx(55.5, rel=1e-7)
    assert photo_meta.longitude == pytest.approx(37.6, rel=1e-7)
    assert gps_payload["latitude"] == pytest.approx(55.5, rel=1e-7)
    assert gps_payload["longitude"] == pytest.approx(37.6, rel=1e-7)
    assert exif_payload["GPS"]["GPSLatitudeRef"] == "N"
    assert exif_payload["GPS"]["GPSLongitudeRef"] == "E"
