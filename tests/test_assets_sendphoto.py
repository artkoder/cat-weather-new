"""Tests for assets publishing via sendPhoto with fallback."""

import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion import (
    IngestionCallbacks,
    IngestionContext,
    IngestionFile,
    IngestionInputs,
    IngestionVisionConfig,
    UploadIngestionContext,
    UploadMetricsRecorder,
    _get_assets_upload_mode,
    _truncate_caption,
    ingest_photo,
)
from tests.fixtures.ingestion_utils import (
    OpenAIStub,
    TelegramStub,
    create_sample_image,
)


def test_get_assets_upload_mode_default() -> None:
    """Test that default mode is 'photo'."""
    os.environ.pop("ASSETS_UPLOAD_MODE", None)
    mode = _get_assets_upload_mode()
    assert mode == "photo"


def test_get_assets_upload_mode_document() -> None:
    """Test that mode can be set to 'document'."""
    os.environ["ASSETS_UPLOAD_MODE"] = "document"
    try:
        mode = _get_assets_upload_mode()
        assert mode == "document"
    finally:
        os.environ.pop("ASSETS_UPLOAD_MODE", None)


def test_get_assets_upload_mode_invalid() -> None:
    """Test that invalid mode falls back to 'photo'."""
    os.environ["ASSETS_UPLOAD_MODE"] = "invalid"
    try:
        mode = _get_assets_upload_mode()
        assert mode == "photo"
    finally:
        os.environ.pop("ASSETS_UPLOAD_MODE", None)


def test_truncate_caption_short() -> None:
    """Test that short captions are not truncated."""
    caption = "Short caption"
    result = _truncate_caption(caption, max_length=1024)
    assert result == caption


def test_truncate_caption_exact() -> None:
    """Test that captions at exact length are not truncated."""
    caption = "x" * 1024
    result = _truncate_caption(caption, max_length=1024)
    assert result == caption


def test_truncate_caption_long() -> None:
    """Test that long captions are truncated."""
    caption = "x" * 1500
    result = _truncate_caption(caption, max_length=1024)
    assert result is not None
    assert len(result) <= 1024
    assert result.endswith("…")


def test_truncate_caption_soft_boundary() -> None:
    """Test that truncation respects word boundaries."""
    caption = "word " * 300  # Creates a long caption with spaces
    result = _truncate_caption(caption, max_length=1024)
    assert result is not None
    assert len(result) <= 1024
    assert result.endswith("…")
    # Should not end with a partial word (except the ellipsis)
    assert not result[:-1].endswith(" ")


def test_truncate_caption_none() -> None:
    """Test that None captions are handled."""
    result = _truncate_caption(None, max_length=1024)
    assert result is None


def test_truncate_caption_empty() -> None:
    """Test that empty captions are handled."""
    result = _truncate_caption("", max_length=1024)
    assert result == ""


@pytest.mark.asyncio
async def test_ingest_mobile_uses_sendphoto_by_default(tmp_path: Path) -> None:
    """Test that mobile uploads use sendPhoto by default."""
    os.environ.pop("ASSETS_UPLOAD_MODE", None)

    image_path = create_sample_image(tmp_path / "test.jpg")
    telegram = TelegramStub()
    metrics = UploadMetricsRecorder()

    from api.uploads import UploadsConfig

    context_obj = UploadIngestionContext(
        upload_id="upload-123",
        storage_key="ref-123",
        metrics=metrics,
        source="mobile",
    )

    config = UploadsConfig(assets_channel_id=-100123456789, vision_enabled=False)

    # Mock DataAccess - not used for this test
    class MockDataAccess:
        def save_asset(self, *args, **kwargs):
            return "asset-123"

        def create_asset(self, *args, **kwargs):
            return "asset-123"

    data = MockDataAccess()

    result = await ingest_photo(
        data=data,
        telegram=telegram,
        openai=None,
        supabase=None,
        config=config,
        context=context_obj,
        file_path=image_path,
        cleanup_file=False,
    )

    # Verify sendPhoto was called
    assert len(telegram.calls) == 1
    call = telegram.calls[0]
    assert call["method"] == "sendPhoto"
    assert result.message_id == call["message_id"]


@pytest.mark.asyncio
async def test_ingest_mobile_uses_senddocument_when_configured(tmp_path: Path) -> None:
    """Test that mobile uploads use sendDocument when ASSETS_UPLOAD_MODE=document."""
    os.environ["ASSETS_UPLOAD_MODE"] = "document"
    try:
        image_path = create_sample_image(tmp_path / "test.jpg")
        telegram = TelegramStub()
        metrics = UploadMetricsRecorder()

        from api.uploads import UploadsConfig

        context_obj = UploadIngestionContext(
            upload_id="upload-123",
            storage_key="ref-123",
            metrics=metrics,
            source="mobile",
        )

        config = UploadsConfig(assets_channel_id=-100123456789, vision_enabled=False)

        # Mock DataAccess - not used for this test
        class MockDataAccess:
            def save_asset(self, *args, **kwargs):
                return "asset-123"

            def create_asset(self, *args, **kwargs):
                return "asset-123"

        data = MockDataAccess()

        result = await ingest_photo(
            data=data,
            telegram=telegram,
            openai=None,
            supabase=None,
            config=config,
            context=context_obj,
            file_path=image_path,
            cleanup_file=False,
        )

        # Verify sendDocument was called
        assert len(telegram.calls) == 1
        call = telegram.calls[0]
        assert call["method"] == "sendDocument"
        assert result.message_id == call["message_id"]
    finally:
        os.environ.pop("ASSETS_UPLOAD_MODE", None)


class FailingSendPhoto(TelegramStub):
    """Telegram stub that fails sendPhoto but succeeds sendDocument."""

    async def send_photo(
        self, *, chat_id: int, photo: Path, caption: str | None = None
    ) -> dict[str, Any]:
        raise ValueError("sendPhoto failed")


@pytest.mark.asyncio
async def test_ingest_mobile_fallback_to_senddocument(tmp_path: Path) -> None:
    """Test that mobile uploads fallback to sendDocument when sendPhoto fails."""
    os.environ.pop("ASSETS_UPLOAD_MODE", None)

    image_path = create_sample_image(tmp_path / "test.jpg")
    telegram = FailingSendPhoto()
    metrics = UploadMetricsRecorder()

    from api.uploads import UploadsConfig

    context_obj = UploadIngestionContext(
        upload_id="upload-123",
        storage_key="ref-123",
        metrics=metrics,
        source="mobile",
    )

    config = UploadsConfig(assets_channel_id=-100123456789, vision_enabled=False)

    # Mock DataAccess - not used for this test
    class MockDataAccess:
        def save_asset(self, *args, **kwargs):
            return "asset-123"

        def create_asset(self, *args, **kwargs):
            return "asset-123"

    data = MockDataAccess()

    result = await ingest_photo(
        data=data,
        telegram=telegram,
        openai=None,
        supabase=None,
        config=config,
        context=context_obj,
        file_path=image_path,
        cleanup_file=False,
    )

    # Verify sendDocument was called after sendPhoto failed
    assert len(telegram.calls) == 1
    call = telegram.calls[0]
    assert call["method"] == "sendDocument"
    assert result.message_id == call["message_id"]
    # Verify fallback metric was incremented
    assert metrics.counters.get("assets_publish_fallback", 0) > 0
