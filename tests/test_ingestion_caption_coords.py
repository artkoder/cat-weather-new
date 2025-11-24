import os
import sys
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ingestion
from ingestion import (
    ImageMetadataResult,
    IngestionCallbacks,
    IngestionContext,
    IngestionFile,
    IngestionInputs,
    IngestionVisionConfig,
    UploadMetricsRecorder,
)
from metadata.extractor import PhotoMeta


class DummyTelegram:
    async def send_document(
        self,
        *,
        chat_id: int,
        document,
        file_name: str,
        caption: str | None = None,
        content_type: str | None = None,
    ) -> dict:
        del chat_id, document, file_name, caption, content_type
        return {"message_id": 1, "document": {"file_id": "doc", "file_unique_id": "uniq"}}

    async def send_photo(
        self,
        *,
        chat_id: int,
        photo,
        caption: str | None = None,
    ) -> dict:
        del chat_id, photo, caption
        return {
            "message_id": 1,
            "photo": [
                {
                    "file_id": "doc",
                    "file_unique_id": "uniq",
                    "width": 640,
                    "height": 480,
                }
            ],
        }


@pytest.mark.asyncio
async def test_caption_coordinates_fills_missing_metadata(monkeypatch, tmp_path):
    metrics = UploadMetricsRecorder()

    image_path = tmp_path / "caption-geo.jpg"
    image_path.write_bytes(b"fake-image-data")

    metadata_result = ImageMetadataResult(
        mime_type="image/jpeg",
        width=640,
        height=480,
        exif={},
        gps={},
        exif_ifds={},
        photo=PhotoMeta(captured_at=None),
    )

    def fake_extract_image_metadata(_path, *, skip_gps: bool = False):  # noqa: ARG001, ANN001
        del skip_gps
        return metadata_result

    monkeypatch.setattr(ingestion, "extract_image_metadata", fake_extract_image_metadata)

    collected_payload: dict[str, object] = {}

    def fake_create_asset(payload: dict[str, object]) -> str:
        collected_payload.update(payload)
        return "asset-geo"

    capture_local = datetime(2024, 5, 21, 18, 32, tzinfo=ZoneInfo("Europe/Kaliningrad"))
    gps_override = {
        "latitude": 54.72,
        "longitude": 20.51,
        "captured_at": capture_local.isoformat(),
    }

    ingestion_inputs = IngestionInputs(
        source="telegram",
        channel_id=-100500,
        file=IngestionFile(path=image_path, cleanup=False),
        upload_id="upload-geo",
        file_ref="file-ref-geo",
        vision=IngestionVisionConfig(enabled=False),
        gps=gps_override,
    )

    context = IngestionContext(
        telegram=DummyTelegram(),
        metrics=metrics,
        openai=None,
        supabase=None,
        token_logger=None,
    )

    callbacks = IngestionCallbacks(create_asset=fake_create_asset)

    result = await ingestion._ingest_photo_internal(  # type: ignore[attr-defined]
        ingestion_inputs,
        context,
        callbacks,
    )

    expected_timestamp = int(capture_local.astimezone(UTC).timestamp())
    expected_doy = capture_local.timetuple().tm_yday

    assert collected_payload["shot_at_utc"] == expected_timestamp
    assert collected_payload["shot_doy"] == expected_doy
    assert collected_payload.get("photo_doy") == expected_doy
    assert result.gps["captured_at"] == capture_local.isoformat()
