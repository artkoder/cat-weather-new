import os
import sys
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any
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
    ) -> dict[str, Any]:
        return {
            "message_id": 321,
            "document": {
                "file_id": "doc-321",
                "file_unique_id": "uniq-321",
                "file_name": file_name,
            },
        }

    async def send_photo(
        self,
        *,
        chat_id: int,
        photo,
        caption: str | None = None,
    ) -> dict[str, Any]:
        return {
            "message_id": 321,
            "photo": [
                {
                    "file_id": "doc-321",
                    "file_unique_id": "uniq-321",
                    "width": 800,
                    "height": 600,
                }
            ],
        }


@pytest.mark.asyncio
async def test_ingestion_sets_shot_fields_with_timezone(monkeypatch, tmp_path):
    metrics = UploadMetricsRecorder()

    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image-data")

    kaliningrad_tz = ZoneInfo("Europe/Kaliningrad")
    captured_local = datetime(2020, 3, 1, 1, 30, tzinfo=kaliningrad_tz)
    photo_meta = PhotoMeta(captured_at=captured_local)
    metadata_result = ImageMetadataResult(
        mime_type="image/jpeg",
        width=1024,
        height=768,
        exif={"DateTimeOriginal": "2020:03:01 01:30:00"},
        gps={},
        exif_ifds={},
        photo=photo_meta,
    )

    def fake_extract_image_metadata(_path, *, skip_gps: bool = False):  # noqa: ANN001
        return metadata_result

    monkeypatch.setattr(ingestion, "extract_image_metadata", fake_extract_image_metadata)

    collected_payload: dict[str, Any] = {}

    def fake_create_asset(payload: dict[str, Any]) -> str:
        collected_payload.update(payload)
        return "asset-1"

    ingestion_inputs = IngestionInputs(
        source="mobile",
        channel_id=-100500,
        file=IngestionFile(path=image_path, cleanup=False),
        upload_id="upload-1",
        file_ref="file-ref-1",
        vision=IngestionVisionConfig(enabled=False),
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

    expected_dt = captured_local.astimezone(UTC)
    expected_timestamp = int(expected_dt.timestamp())
    expected_doy = captured_local.timetuple().tm_yday

    assert collected_payload["shot_at_utc"] == expected_timestamp
    assert collected_payload["shot_doy"] == expected_doy
    assert result.asset_id == "asset-1"


@pytest.mark.asyncio
async def test_exif_shot_doy_saved(monkeypatch, tmp_path):
    metrics = UploadMetricsRecorder()

    image_path = tmp_path / "sample-exif.jpg"
    image_path.write_bytes(b"fake-image-data")

    photo_meta = PhotoMeta(captured_at=None)
    metadata_result = ImageMetadataResult(
        mime_type="image/jpeg",
        width=800,
        height=600,
        exif={"DateTimeOriginal": "2025:11:06 12:38:00"},
        gps={},
        exif_ifds={},
        photo=photo_meta,
    )

    def fake_extract_image_metadata(_path, *, skip_gps: bool = False):  # noqa: ANN001
        return metadata_result

    monkeypatch.setattr(ingestion, "extract_image_metadata", fake_extract_image_metadata)

    collected_payload: dict[str, Any] = {}

    def fake_create_asset(payload: dict[str, Any]) -> str:
        collected_payload.update(payload)
        return "asset-exif"

    ingestion_inputs = IngestionInputs(
        source="mobile",
        channel_id=-12345,
        file=IngestionFile(path=image_path, cleanup=False),
        upload_id="upload-exif",
        file_ref="file-ref-exif",
        vision=IngestionVisionConfig(enabled=False),
    )

    context = IngestionContext(
        telegram=DummyTelegram(),
        metrics=metrics,
        openai=None,
        supabase=None,
        token_logger=None,
    )

    callbacks = IngestionCallbacks(create_asset=fake_create_asset)

    await ingestion._ingest_photo_internal(  # type: ignore[attr-defined]
        ingestion_inputs,
        context,
        callbacks,
    )

    assert collected_payload["shot_doy"] == 310
