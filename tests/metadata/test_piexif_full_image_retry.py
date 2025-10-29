from __future__ import annotations

import copy
import io
import os
import sys
from pathlib import Path

import piexif
import pytest
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from metadata.extractor import extract_metadata_from_file


@pytest.fixture()
def mobile_like_photo_bytes(tmp_path: Path) -> bytes:
    image_path = tmp_path / "mobile_photo.jpg"
    image = Image.new("RGB", (200, 100), color=(180, 140, 90))
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "PhoneMaker".encode("utf-8"),
            piexif.ImageIFD.Model: "PhoneCam".encode("utf-8"),
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2024:05:06 07:08:09",
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: [(37, 1), (46, 1), (5000, 100)],
            piexif.GPSIFD.GPSLongitudeRef: "E",
            piexif.GPSIFD.GPSLongitude: [(122, 1), (25, 1), (600, 100)],
            piexif.GPSIFD.GPSAltitudeRef: b"\x00",
            piexif.GPSIFD.GPSAltitude: (523, 10),
        },
        "1st": {},
        "thumbnail": None,
    }
    image.save(image_path, format="JPEG", exif=piexif.dump(exif_dict))
    return image_path.read_bytes()


def test_retries_full_image_bytes_before_fallback(
    mobile_like_photo_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = mobile_like_photo_bytes
    with Image.open(io.BytesIO(data)) as image:
        embedded_exif = image.info.get("exif")
    assert embedded_exif

    original_load = piexif.load

    def _patched_load(payload, *args, **kwargs):  # type: ignore[override]
        loaded = original_load(payload, *args, **kwargs)
        if isinstance(payload, (bytes, bytearray)) and payload == embedded_exif:
            corrupted = copy.deepcopy(loaded)
            gps_ifd = copy.deepcopy(corrupted.get("GPS") or {})
            if gps_ifd:
                gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = b"\x00"
                gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = b"\x00"
                gps_ifd[piexif.GPSIFD.GPSLatitude] = [(0, 1), (0, 1), (0, 1)]
                gps_ifd[piexif.GPSIFD.GPSLongitude] = [(0, 1), (0, 1), (0, 1)]
                gps_ifd[piexif.GPSIFD.GPSAltitude] = (0, 1)
                corrupted["GPS"] = gps_ifd
            return corrupted
        return loaded

    monkeypatch.setattr(piexif, "load", _patched_load)

    photo_meta, exif_payload, gps_payload, exif_ifds = extract_metadata_from_file(data)

    assert photo_meta.source == "piexif"
    assert photo_meta.latitude == pytest.approx(37.7805556, rel=1e-7)
    assert photo_meta.longitude == pytest.approx(122.4183333, rel=1e-7)
    assert photo_meta.altitude == pytest.approx(52.3, rel=1e-7)
    assert gps_payload["latitude"] == pytest.approx(photo_meta.latitude, rel=1e-7)
    assert gps_payload["longitude"] == pytest.approx(photo_meta.longitude, rel=1e-7)
    assert gps_payload["altitude"] == pytest.approx(photo_meta.altitude, rel=1e-7)
    assert exif_payload.get("Make") == "PhoneMaker"
    assert exif_ifds.get("GPS", {}).get("GPSLatitude") == [
        [37, 1],
        [46, 1],
        [5000, 100],
    ]
