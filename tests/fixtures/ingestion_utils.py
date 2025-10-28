from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib
import json

import piexif
from PIL import Image

from openai_client import OpenAIResponse


DEFAULT_VISION_PAYLOAD: dict[str, Any] = {
    "caption": "Солнечный кот во дворе",
    "objects": ["кот"],
    "tags": ["animals", "sunny", "pet"],
    "weather_image": "sunny",
    "season_guess": "summer",
    "categories": ["animals", "sunny"],
    "safety": {"nsfw": False, "reason": "безопасно"},
}


def create_sample_image(path: Path) -> Path:
    """Create a deterministic JPEG image with EXIF metadata for tests."""

    image = Image.new("RGB", (640, 480), color=(10, 20, 30))
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "UnitTest".encode("utf-8"),
            piexif.ImageIFD.DateTime: "2023:12:24 15:30:45",
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2023:12:24 15:30:45",
            piexif.ExifIFD.DateTimeDigitized: "2023:12:24 15:30:45",
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: [(55, 1), (30, 1), (0, 1)],
            piexif.GPSIFD.GPSLongitudeRef: "E",
            piexif.GPSIFD.GPSLongitude: [(37, 1), (36, 1), (0, 1)],
        },
        "1st": {},
        "thumbnail": None,
    }
    exif_bytes = piexif.dump(exif_dict)
    image.save(path, format="JPEG", exif=exif_bytes)
    return path


def create_sample_image_without_gps(path: Path) -> Path:
    """Create a JPEG image with EXIF metadata but without GPS tags."""

    image = Image.new("RGB", (640, 480), color=(10, 20, 30))
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "UnitTest".encode("utf-8"),
            piexif.ImageIFD.DateTime: "2023:12:24 15:30:45",
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2023:12:24 15:30:45",
            piexif.ExifIFD.DateTimeDigitized: "2023:12:24 15:30:45",
        },
        "1st": {},
        "thumbnail": None,
    }
    exif_bytes = piexif.dump(exif_dict)
    image.save(path, format="JPEG", exif=exif_bytes)
    return path


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StorageStub:
    def __init__(self, mapping: dict[str, Path]) -> None:
        self.mapping = mapping
        self.get_calls: list[str] = []

    async def put_stream(self, *, key: str, stream, content_type: str) -> str:  # pragma: no cover - not used
        raise NotImplementedError

    async def get_url(self, *, key: str) -> str:
        self.get_calls.append(key)
        try:
            return self.mapping[key].as_uri()
        except KeyError as exc:  # pragma: no cover - defensive
            raise FileNotFoundError(key) from exc


class TelegramStub:
    def __init__(self, *, start_message_id: int = 100) -> None:
        self.next_message_id = start_message_id
        self.calls: list[dict[str, Any]] = []

    async def send_photo(
        self,
        *,
        chat_id: int,
        photo: Path,
        caption: str | None = None,
    ) -> dict[str, Any]:
        message_id = self.next_message_id
        self.next_message_id += 1
        with Image.open(photo) as img:
            width, height = img.size
        file_id = f"photo-{message_id}"
        photo_sizes = [
            {
                "file_id": f"{file_id}-s",
                "file_unique_id": f"{file_id}-s-uniq",
                "width": max(1, width // 2),
                "height": max(1, height // 2),
                "file_size": max(1, (width // 2) * (height // 2) * 2),
                "mime_type": "image/jpeg",
            },
            {
                "file_id": file_id,
                "file_unique_id": f"{file_id}-uniq",
                "width": width,
                "height": height,
                "file_size": max(1, width * height * 3),
                "mime_type": "image/jpeg",
            },
        ]
        call = {
            "chat_id": chat_id,
            "photo": Path(photo),
            "caption": caption,
            "message_id": message_id,
            "photo_sizes": photo_sizes,
            "file_id": file_id,
        }
        self.calls.append(call)
        return {"message_id": message_id, "chat": {"id": chat_id}, "photo": photo_sizes}


@dataclass
class OpenAIStub:
    payload: dict[str, Any] = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = json.loads(json.dumps(DEFAULT_VISION_PAYLOAD))
        self.calls: list[dict[str, Any]] = []

    async def classify_image(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return OpenAIResponse(
            self.payload,
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "request_id": "req-test",
                "endpoint": "/v1/responses",
            },
        )
