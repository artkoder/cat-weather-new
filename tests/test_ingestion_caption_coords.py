import os
import sys
from datetime import UTC, datetime
from typing import Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ingestion
from ingestion import (  # noqa: E402
    ImageMetadataResult,
    IngestionCallbacks,
    IngestionContext,
    IngestionFile,
    IngestionInputs,
    IngestionVisionConfig,
    UploadMetricsRecorder,
)
from metadata.extractor import PhotoMeta  # noqa: E402


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
            "message_id": 42,
            "document": {
                "file_id": "doc-42",
                "file_unique_id": "uniq-42",
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
            "message_id": 42,
            "photo": [
                {
                    "file_id": "doc-42",
                    "file_unique_id": "uniq-42",
                    "width": 800,
                    "height": 600,
                }
            ],
        }


@pytest.mark.asyncio
async def test_caption_coordinates_fills_missing_metadata(monkeypatch, tmp_path):
    metrics = UploadMetricsRecorder()
    image_path = tmp_path / "caption-source.jpg"
    image_path.write_bytes(b"fake-bytes")

    metadata_result = ImageMetadataResult(
        mime_type="image/jpeg",
        width=800,
        height=600,
        exif={},
        gps={},
        exif_ifds={},
        photo=PhotoMeta(),
    )

    def fake_extract_image_metadata(_path, *, skip_gps: bool = False):  # noqa: ANN001
        return metadata_result

    monkeypatch.setattr(ingestion, "extract_image_metadata", fake_extract_image_metadata)

    async def fake_extract_caption_geo_metadata(**kwargs):  # noqa: ANN001
        fake_extract_caption_geo_metadata.calls.append(kwargs)
        return {
            "latitude": 59.9386,
            "longitude": 30.3141,
            "captured_at_iso": "2024-05-23T10:15:00+03:00",
        }

    fake_extract_caption_geo_metadata.calls = []  # type: ignore[attr-defined]
    monkeypatch.setattr(ingestion, "_extract_caption_geo_metadata", fake_extract_caption_geo_metadata)
    monkeypatch.setattr(ingestion, "_CAPTION_COORDS_MODEL", "gpt-4o-mini")

    captured_payload: dict[str, Any] = {}

    def fake_create_asset(payload: dict[str, Any]) -> str:
        captured_payload.update(payload)
        return "asset-from-caption"

    ingestion_inputs = IngestionInputs(
        source="telegram",
        channel_id=-100500,
        file=IngestionFile(path=image_path, cleanup=False),
        upload_id="upload-caption",
        asset_id="asset-existing",
        file_ref="file-ref-caption",
        caption="Снято 23.05.2024 в 10:15 МСК\nКоординаты: 59.9386, 30.3141",
        vision=IngestionVisionConfig(enabled=False),
    )

    context = IngestionContext(
        telegram=DummyTelegram(),
        metrics=metrics,
        openai=object(),
        supabase=None,
        token_logger=None,
    )

    callbacks = IngestionCallbacks(create_asset=fake_create_asset)

    result = await ingestion._ingest_photo_internal(  # type: ignore[attr-defined]
        ingestion_inputs,
        context,
        callbacks,
    )

    assert fake_extract_caption_geo_metadata.calls, "caption helper should be invoked"
    assert pytest.approx(result.gps["latitude"], rel=1e-6) == 59.9386
    assert pytest.approx(result.gps["longitude"], rel=1e-6) == 30.3141
    assert result.gps["captured_at"] == "2024-05-23T10:15:00+03:00"
    assert "Дата съёмки" in result.caption
    assert "Координаты" in result.caption

    expected_timestamp = int(datetime(2024, 5, 23, 7, 15, tzinfo=UTC).timestamp())
    assert captured_payload["shot_at_utc"] == expected_timestamp
    assert captured_payload["shot_doy"] == datetime(2024, 5, 23).timetuple().tm_yday


@pytest.mark.asyncio
async def test_caption_coordinates_skipped_without_caption(monkeypatch, tmp_path):
    metrics = UploadMetricsRecorder()
    image_path = tmp_path / "caption-missing.jpg"
    image_path.write_bytes(b"fake-bytes")

    metadata_result = ImageMetadataResult(
        mime_type="image/jpeg",
        width=800,
        height=600,
        exif={},
        gps={},
        exif_ifds={},
        photo=PhotoMeta(),
    )

    def fake_extract_image_metadata(_path, *, skip_gps: bool = False):  # noqa: ANN001
        return metadata_result

    monkeypatch.setattr(ingestion, "extract_image_metadata", fake_extract_image_metadata)

    async def should_not_be_called(**kwargs):  # noqa: ANN001
        raise AssertionError("caption parser should not run when caption is missing")

    monkeypatch.setattr(ingestion, "_extract_caption_geo_metadata", should_not_be_called)
    monkeypatch.setattr(ingestion, "_CAPTION_COORDS_MODEL", "gpt-4o-mini")

    def fake_create_asset(payload: dict[str, Any]) -> str:
        return "asset-no-caption"

    ingestion_inputs = IngestionInputs(
        source="telegram",
        channel_id=-100600,
        file=IngestionFile(path=image_path, cleanup=False),
        upload_id="upload-no-caption",
        file_ref="file-ref-missing",
        vision=IngestionVisionConfig(enabled=False),
    )

    context = IngestionContext(
        telegram=DummyTelegram(),
        metrics=metrics,
        openai=object(),
        supabase=None,
        token_logger=None,
    )

    callbacks = IngestionCallbacks(create_asset=fake_create_asset)

    result = await ingestion._ingest_photo_internal(  # type: ignore[attr-defined]
        ingestion_inputs,
        context,
        callbacks,
    )

    assert result.gps.get("latitude") is None
    assert result.gps.get("longitude") is None
    assert result.gps.get("captured_at") is None
    assert result.caption == "Новая загрузка готова к обработке."
