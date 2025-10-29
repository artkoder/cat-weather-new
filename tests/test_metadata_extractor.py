from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import piexif
import pytest
from PIL import Image

from metadata.extractor import PhotoMeta, extract_metadata_from_file
from tests.fixtures.ingestion_utils import create_sample_image


def _save_with_exif(path: Path, *, fmt: str, exif_dict: dict[str, Any]) -> Path:
    image = Image.new("RGB", (120, 80), color=(12, 34, 56))
    exif_bytes = piexif.dump(exif_dict)
    image.save(path, format=fmt, exif=exif_bytes)
    return path


def _create_heic_with_exif(path: Path) -> Path:
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    if hasattr(pillow_heif, "register_heif_writer"):
        pillow_heif.register_heif_writer()
    else:  # pragma: no cover - depends on optional dependency version
        pytest.skip("pillow_heif writer support not available")

    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "HeifMaker".encode("utf-8"),
            piexif.ImageIFD.Model: "HeifCam".encode("utf-8"),
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2024:01:02 03:04:05",
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "S",
            piexif.GPSIFD.GPSLatitude: [(13, 1), (15, 1), (0, 1)],
            piexif.GPSIFD.GPSLongitudeRef: "W",
            piexif.GPSIFD.GPSLongitude: [(77, 1), (30, 1), (0, 1)],
            piexif.GPSIFD.GPSAltitude: (150, 1),
            piexif.GPSIFD.GPSAltitudeRef: b"\x00",
        },
        "1st": {},
        "thumbnail": None,
    }
    return _save_with_exif(path, fmt="HEIF", exif_dict=exif_dict)


def _create_webp_with_exif(path: Path) -> Path:
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "WebpMaker".encode("utf-8"),
            piexif.ImageIFD.Model: "WebpCam".encode("utf-8"),
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2024:04:05 06:07:08",
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: [(40, 1), (30, 1), (0, 1)],
            piexif.GPSIFD.GPSLongitudeRef: "E",
            piexif.GPSIFD.GPSLongitude: [(70, 1), (10, 1), (0, 1)],
        },
        "1st": {},
        "thumbnail": None,
    }
    return _save_with_exif(path, fmt="WEBP", exif_dict=exif_dict)


def _create_gps_timestamp_exif() -> dict[str, Any]:
    return {
        "0th": {
            piexif.ImageIFD.Make: "GPSMaker".encode("utf-8"),
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2024:06:01 02:03:04",
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: [(10, 1), (20, 1), (300, 100)],
            piexif.GPSIFD.GPSLongitudeRef: "E",
            piexif.GPSIFD.GPSLongitude: [(20, 1), (40, 1), (500, 100)],
            piexif.GPSIFD.GPSAltitude: (1234, 100),
            piexif.GPSIFD.GPSAltitudeRef: b"\x01",
            piexif.GPSIFD.GPSDateStamp: "2024:06:02",
            piexif.GPSIFD.GPSTimeStamp: [(12, 1), (34, 1), (56789, 1000)],
        },
        "1st": {},
        "thumbnail": None,
    }


def test_extract_metadata_from_jpeg_returns_photo_meta(tmp_path: Path) -> None:
    image_path = create_sample_image(tmp_path / "sample.jpg")
    photo_meta, exif_payload, gps_payload, exif_ifds = extract_metadata_from_file(
        image_path.read_bytes()
    )

    assert isinstance(photo_meta, PhotoMeta)
    assert photo_meta.latitude == pytest.approx(55.5, rel=1e-7)
    assert photo_meta.longitude == pytest.approx(37.6, rel=1e-7)
    assert photo_meta.source in {"piexif", "pillow"}
    assert "GPS" in exif_ifds and exif_ifds["GPS"]
    assert gps_payload["latitude"] == pytest.approx(55.5, rel=1e-7)
    assert gps_payload["longitude"] == pytest.approx(37.6, rel=1e-7)
    assert "GPS" in exif_payload


def test_extract_metadata_from_heic(tmp_path: Path) -> None:
    heic_path = _create_heic_with_exif(tmp_path / "sample.heic")
    photo_meta, exif_payload, gps_payload, _ = extract_metadata_from_file(heic_path.read_bytes())

    assert photo_meta.make == "HeifMaker"
    assert photo_meta.model == "HeifCam"
    assert photo_meta.latitude == pytest.approx(-13.25, rel=1e-7)
    assert photo_meta.longitude == pytest.approx(-77.5, rel=1e-7)
    assert gps_payload["latitude"] == pytest.approx(-13.25, rel=1e-7)
    assert gps_payload["longitude"] == pytest.approx(-77.5, rel=1e-7)


def test_extract_metadata_from_webp(tmp_path: Path) -> None:
    webp_path = _create_webp_with_exif(tmp_path / "sample.webp")
    photo_meta, exif_payload, gps_payload, _ = extract_metadata_from_file(webp_path)

    assert photo_meta.make == "WebpMaker"
    assert photo_meta.model == "WebpCam"
    assert "DateTimeOriginal" in exif_payload
    assert gps_payload["latitude"] == pytest.approx(40.5, rel=1e-7)
    assert gps_payload["longitude"] == pytest.approx(70.1666667, rel=1e-7)


def test_extract_metadata_handles_negative_altitude(tmp_path: Path) -> None:
    exif_dict = _create_gps_timestamp_exif()
    image_path = _save_with_exif(tmp_path / "gps.jpg", fmt="JPEG", exif_dict=exif_dict)

    photo_meta, _, gps_payload, _ = extract_metadata_from_file(image_path.read_bytes())

    assert photo_meta.altitude is not None
    assert photo_meta.altitude < 0
    assert gps_payload["altitude"] == photo_meta.altitude


def test_extract_metadata_combines_gps_timestamp(tmp_path: Path) -> None:
    exif_dict = _create_gps_timestamp_exif()
    image_path = _save_with_exif(tmp_path / "gps_time.jpg", fmt="JPEG", exif_dict=exif_dict)

    photo_meta, _, gps_payload, _ = extract_metadata_from_file(image_path)

    assert photo_meta.captured_at is not None
    assert photo_meta.captured_at.tzinfo == timezone.utc
    assert gps_payload["captured_at"] == photo_meta.captured_at.isoformat()
    expected = datetime(2024, 6, 2, 12, 34, 56, 789000, tzinfo=timezone.utc)
    assert photo_meta.captured_at == expected


def test_extract_metadata_falls_back_to_exifread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = create_sample_image(tmp_path / "fallback.jpg")

    original_load = piexif.load

    def _failing_load(*args: Any, **kwargs: Any):  # type: ignore[override]
        raise RuntimeError("broken piexif")

    monkeypatch.setattr(piexif, "load", _failing_load)
    photo_meta, exif_payload, _, _ = extract_metadata_from_file(image_path)

    monkeypatch.setattr(piexif, "load", original_load)

    assert photo_meta.source in {"exifread", "pillow"}
    assert exif_payload


def test_extract_metadata_recovers_from_invalid_piexif_gps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("exifread")

    image_path = create_sample_image(tmp_path / "fallback-invalid-gps.jpg")

    original_load = piexif.load

    def _corrupting_load(*args: Any, **kwargs: Any):  # type: ignore[override]
        exif_dict = original_load(*args, **kwargs)
        mutated = dict(exif_dict)
        gps_ifd = dict(mutated.get("GPS") or {})
        if gps_ifd:
            gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = b"00"
            gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = b"00"
            gps_ifd[piexif.GPSIFD.GPSLatitude] = [(0, 0)]
            gps_ifd[piexif.GPSIFD.GPSLongitude] = [(0, 0)]
            mutated["GPS"] = gps_ifd
        return mutated

    monkeypatch.setattr(piexif, "load", _corrupting_load)

    photo_meta, exif_payload, _, exif_ifds = extract_metadata_from_file(image_path)

    assert photo_meta.source == "exifread"
    assert photo_meta.latitude == pytest.approx(55.5, rel=1e-7)
    assert photo_meta.longitude == pytest.approx(37.6, rel=1e-7)
    assert photo_meta.raw_gps
    assert exif_payload.get("GPS", {}).get("GPSLatitudeRef") == "N"
    assert exif_payload.get("GPS", {}).get("GPSLongitudeRef") == "E"
    assert exif_ifds.get("GPS")
    assert exif_ifds["GPS"].get("GPSLatitudeRef") == "N"
    assert exif_ifds["GPS"].get("GPSLongitudeRef") == "E"
    assert photo_meta.raw_gps.get("GPSLatitudeRef") == "N"
    assert photo_meta.raw_gps.get("GPSLongitudeRef") == "E"
