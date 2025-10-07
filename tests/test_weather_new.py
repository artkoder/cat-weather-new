import json
import os
import sys
import types
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import copy
import piexif
import pytest
import sqlite3
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import Bot, Job
from openai_client import OpenAIResponse

os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'dummy')


async def _run_vision_job_collect_calls(
    tmp_path,
    monkeypatch,
    *,
    flag_enabled: bool,
    asset_kind: str = 'photo',
    asset_file_name: str = 'sample.jpg',
    asset_mime: str = 'image/jpeg',
    metadata: dict[str, Any] | None = None,
    vision_overrides: dict[str, Any] | None = None,
    exif_month: int | None = None,
):
    import main as main_module

    monkeypatch.setattr(main_module, 'ASSETS_DEBUG_EXIF', flag_enabled)

    bot = Bot('test-token', str(tmp_path / 'db.sqlite'))
    bot.asset_storage = tmp_path

    img = Image.new('RGB', (10, 10), color='white')
    buffer = BytesIO()
    exif_bytes: bytes | None = None
    if exif_month is not None:
        exif_dict = {
            'Exif': {
                piexif.ExifIFD.DateTimeOriginal: f"2023:{exif_month:02d}:15 12:00:00".encode('utf-8')
            }
        }
        exif_bytes = piexif.dump(exif_dict)
    if exif_bytes:
        img.save(buffer, format='JPEG', exif=exif_bytes)
    else:
        img.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()

    calls: list[dict[str, Any]] = []
    recognized_mid = 777

    async def fake_api_request(method, data=None, *, files=None):  # type: ignore[override]
        calls.append({'method': method, 'data': data, 'files': files})
        if method in {'copyMessage', 'sendPhoto', 'sendDocument'}:
            return {'ok': True, 'result': {'message_id': recognized_mid}}
        if method == 'sendMessage':
            return {'ok': True, 'result': {'message_id': recognized_mid + 1}}
        if method == 'deleteMessage':
            return {'ok': True, 'result': True}
        return {'ok': True, 'result': {}}

    async def fake_api_request_multipart(method, data=None, *, files=None):  # type: ignore[override]
        normalized_files = None
        if files:
            normalized_files = {}
            for name, (filename, fh, _content_type) in files.items():
                payload = fh.read()
                try:
                    fh.seek(0)
                except Exception:
                    pass
                normalized_files[name] = (filename, payload)
        calls.append({'method': method, 'data': data, 'files': normalized_files})
        if method == 'sendPhoto':
            return {'ok': True, 'result': {'message_id': recognized_mid}}
        return {'ok': True, 'result': {}}

    async def fake_download(file_id, dest_path=None):  # type: ignore[override]
        assert dest_path is not None
        path = Path(dest_path)
        path.write_bytes(image_bytes)
        return path

    class DummyOpenAI:
        def __init__(self):
            self.api_key = 'test-key'

        async def classify_image(self, **kwargs):
            payload = {
                'arch_view': False,
                'caption': 'кот',
                'objects': ['кот'],
                'is_outdoor': False,
                'framing': 'close_up',
                'guess_country': None,
                'guess_city': None,
                'location_confidence': 0.0,
                'landmarks': [],
                'tags': ['animals', 'sunny', 'pet'],
                'architecture_close_up': False,
                'architecture_wide': False,
                'weather_image': 'sunny',
                'season_guess': None,
                'arch_style': None,
                'safety': {'nsfw': False, 'reason': 'безопасно'},
            }
            if vision_overrides:
                payload.update(copy.deepcopy(vision_overrides))
            return OpenAIResponse(
                payload,
                {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15,
                    'request_id': 'req-vision',
                    'endpoint': '/v1/responses',
                },
            )

    async def fake_record(self, *args, **kwargs):  # type: ignore[override]
        return None

    def fake_enforce(self, *args, **kwargs):  # type: ignore[override]
        return None

    supabase_calls: list[dict[str, Any]] = []

    async def fake_insert(self, **kwargs):  # type: ignore[override]
        supabase_calls.append(kwargs)
        return False, {'mock': True}, 'disabled'

    bot.api_request = fake_api_request  # type: ignore[assignment]
    bot.api_request_multipart = fake_api_request_multipart  # type: ignore[assignment]
    bot._download_file = fake_download  # type: ignore[assignment]
    bot.openai = DummyOpenAI()
    bot._record_openai_usage = types.MethodType(fake_record, bot)  # type: ignore[assignment]
    bot._enforce_openai_limit = types.MethodType(fake_enforce, bot)  # type: ignore[assignment]
    bot.supabase.insert_token_usage = types.MethodType(  # type: ignore[assignment]
        fake_insert, bot.supabase
    )

    file_meta = {
        'file_id': 'ph_large',
        'file_unique_id': 'uniq_large',
        'file_name': asset_file_name,
        'mime_type': asset_mime,
        'file_size': len(image_bytes),
        'width': 1920,
        'height': 1080,
    }

    asset_id = bot.data.save_asset(
        channel_id=-100123,
        message_id=123,
        template=None,
        hashtags='#test',
        tg_chat_id=-100123,
        caption='Исходный пост',
        kind=asset_kind,
        file_meta=file_meta,
        metadata=metadata,
        origin='recognition',
    )

    job = Job(
        id=1,
        name='vision',
        payload={'asset_id': asset_id},
        status='queued',
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_vision(job)

    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.recognized_message_id == recognized_mid

    await bot.close()

    return calls, asset, supabase_calls


@pytest.mark.asyncio
async def test_job_vision_skips_exif_debug_by_default(tmp_path, monkeypatch):
    calls, _, _ = await _run_vision_job_collect_calls(
        tmp_path, monkeypatch, flag_enabled=False
    )
    assert any(call['method'] == 'copyMessage' for call in calls)
    assert not any(call['method'] == 'sendMessage' for call in calls)
    delete_calls = [call for call in calls if call['method'] == 'deleteMessage']
    assert delete_calls, 'original photo message should be deleted after publishing'


@pytest.mark.asyncio
async def test_job_vision_sends_exif_debug_when_flag_enabled(tmp_path, monkeypatch):
    calls, _, _ = await _run_vision_job_collect_calls(
        tmp_path, monkeypatch, flag_enabled=True
    )
    assert any(call['method'] == 'copyMessage' for call in calls)
    assert any(call['method'] == 'sendMessage' for call in calls)


@pytest.mark.asyncio
async def test_job_vision_converts_document_to_photo_and_deletes_original(
    tmp_path, monkeypatch
):
    calls, _, _ = await _run_vision_job_collect_calls(
        tmp_path,
        monkeypatch,
        flag_enabled=False,
        asset_kind='document',
        asset_file_name='sample.png',
        asset_mime='image/png',
    )

    send_photo_calls = [call for call in calls if call['method'] == 'sendPhoto']
    assert send_photo_calls, 'converted photo should be uploaded'
    send_photo = send_photo_calls[0]
    assert send_photo['data'] is not None
    assert 'photo' not in send_photo['data']
    assert send_photo['files'] is not None and 'photo' in send_photo['files']
    filename, blob = send_photo['files']['photo']
    assert filename.endswith('.jpg')
    assert blob.startswith(b'\xff\xd8')
    assert len(blob) <= int(10 * 1024 * 1024)

    delete_calls = [call for call in calls if call['method'] == 'deleteMessage']
    assert delete_calls, 'original document message should be deleted'

    copy_calls = [call for call in calls if call['method'] == 'copyMessage']
    assert not copy_calls


@pytest.mark.asyncio
async def test_ingest_extracts_gps_for_convertible_document(tmp_path, monkeypatch):
    bot = Bot('test-token', str(tmp_path / 'db.sqlite'))
    bot.asset_storage = tmp_path

    img = Image.new('RGB', (20, 20), color='white')
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b'N',
        piexif.GPSIFD.GPSLatitude: [(55, 1), (45, 1), (30, 1)],
        piexif.GPSIFD.GPSLongitudeRef: b'E',
        piexif.GPSIFD.GPSLongitude: [(37, 1), (36, 1), (56, 1)],
    }
    exif_bytes = piexif.dump({'GPS': gps_ifd})
    buffer = BytesIO()
    img.save(buffer, format='JPEG', exif=exif_bytes)
    image_bytes = buffer.getvalue()

    calls: list[dict[str, Any]] = []
    recognized_mid = 999

    async def fake_api_request(method, data=None, *, files=None):  # type: ignore[override]
        calls.append({'method': method, 'data': data, 'files': files})
        if method in {'copyMessage', 'sendPhoto', 'sendDocument'}:
            return {'ok': True, 'result': {'message_id': recognized_mid}}
        if method == 'sendMessage':
            return {'ok': True, 'result': {'message_id': recognized_mid + 1}}
        if method == 'deleteMessage':
            return {'ok': True, 'result': True}
        return {'ok': True, 'result': {}}

    async def fake_api_request_multipart(method, data=None, *, files=None):  # type: ignore[override]
        normalized_files = None
        if files:
            normalized_files = {}
            for name, (filename, fh, _content_type) in files.items():
                payload = fh.read()
                try:
                    fh.seek(0)
                except Exception:
                    pass
                normalized_files[name] = (filename, payload)
        calls.append({'method': method, 'data': data, 'files': normalized_files})
        if method == 'sendPhoto':
            return {'ok': True, 'result': {'message_id': recognized_mid}}
        return {'ok': True, 'result': {}}

    async def fake_download(self, file_id, dest_path=None):  # type: ignore[override]
        assert dest_path is not None
        path = Path(dest_path)
        path.write_bytes(image_bytes)
        return path

    async def fake_reverse_geocode(self, lat, lon):  # type: ignore[override]
        return {'city': 'Москва', 'country': 'Россия'}

    class DummyOpenAI:
        def __init__(self):
            self.api_key = 'test-key'

        async def classify_image(self, **kwargs):
            return OpenAIResponse(
                {
                    'arch_view': False,
                    'caption': 'кот',
                    'objects': ['кот'],
                    'is_outdoor': False,
                    'framing': 'close_up',
                    'guess_country': None,
                    'guess_city': None,
                    'location_confidence': 0.0,
                    'landmarks': [],
                    'tags': ['animals', 'sunny', 'pet'],
                    'architecture_close_up': False,
                    'architecture_wide': False,
                    'weather_image': 'sunny',
                    'season_guess': None,
                    'arch_style': None,
                    'safety': {'nsfw': False, 'reason': 'безопасно'},
                },
                {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15,
                    'request_id': 'req-vision',
                    'endpoint': '/v1/responses',
                },
            )

    async def fake_record(self, *args, **kwargs):  # type: ignore[override]
        return None

    def fake_enforce(self, *args, **kwargs):  # type: ignore[override]
        return None

    async def fake_insert(self, **kwargs):  # type: ignore[override]
        return False, {'mock': True}, 'disabled'

    bot.api_request = fake_api_request  # type: ignore[assignment]
    bot.api_request_multipart = fake_api_request_multipart  # type: ignore[assignment]
    bot._download_file = types.MethodType(fake_download, bot)  # type: ignore[assignment]
    bot._reverse_geocode = types.MethodType(fake_reverse_geocode, bot)  # type: ignore[assignment]
    bot.openai = DummyOpenAI()
    bot._record_openai_usage = types.MethodType(fake_record, bot)  # type: ignore[assignment]
    bot._enforce_openai_limit = types.MethodType(fake_enforce, bot)  # type: ignore[assignment]
    bot.supabase.insert_token_usage = types.MethodType(  # type: ignore[assignment]
        fake_insert, bot.supabase
    )

    file_meta = {
        'file_id': 'doc_large',
        'file_unique_id': 'uniq_doc',
        'file_name': 'convertible.jpg',
        'mime_type': 'image/jpeg',
        'file_size': len(image_bytes),
        'width': 1920,
        'height': 1080,
    }

    asset_id = bot.data.save_asset(
        channel_id=-100123,
        message_id=123,
        template=None,
        hashtags='#test',
        tg_chat_id=-100123,
        caption='Исходный документ',
        kind='document',
        file_meta=file_meta,
        author_user_id=4242,
        origin='recognition',
    )

    ingest_job = Job(
        id=1,
        name='ingest',
        payload={'asset_id': asset_id},
        status='queued',
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_ingest(ingest_job)

    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.exif_present is True
    assert asset.latitude == pytest.approx(55.7583, rel=1e-4)
    assert asset.longitude == pytest.approx(37.6156, rel=1e-4)
    assert asset.city == 'Москва'
    assert asset.country == 'Россия'

    vision_job = Job(
        id=2,
        name='vision',
        payload={'asset_id': asset_id},
        status='queued',
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    await bot._job_vision(vision_job)

    send_photo_calls = [call for call in calls if call['method'] == 'sendPhoto']
    assert send_photo_calls, 'converted document should be published as photo'
    caption_payload = send_photo_calls[0]['data']
    assert caption_payload is not None and 'caption' in caption_payload
    caption_text = caption_payload['caption']
    assert caption_text and 'Москва' in caption_text
    assert 'Адрес (EXIF)' in caption_text

    assert not any(
        call['method'] == 'sendMessage'
        and call.get('data', {}).get('text') == 'В изображении отсутствуют EXIF-данные с координатами.'
        for call in calls
    )

    await bot.close()

async def test_job_vision_enriches_weather_season_and_style(tmp_path, monkeypatch):
    overrides = {
        'weather_image': 'cloudy',
        'season_guess': 'spring',
        'arch_style': {'label': 'Gothic', 'confidence': 0.65},
    }
    metadata = {'exif_weather': {'enum': 'rainy'}}

    calls, asset, supabase_calls = await _run_vision_job_collect_calls(
        tmp_path,
        monkeypatch,
        flag_enabled=False,
        vision_overrides=overrides,
        metadata=metadata,
        exif_month=11,
    )

    assert any(call['method'] == 'copyMessage' for call in calls)

    assert asset.vision_results is not None
    vision = asset.vision_results
    assert vision['weather_final'] == 'rain'
    assert vision['weather_final_display'] == 'дождь'
    assert vision['season_final'] == 'autumn'
    assert vision['season_final_display'] == 'осень'
    assert vision['arch_style'] == {'label': 'Gothic', 'confidence': 0.65}
    assert 'rain' in vision['tags']

    assert asset.vision_caption is not None
    assert 'Погода: дождь' in asset.vision_caption
    assert 'Сезон: осень' in asset.vision_caption
    assert 'Стиль: Gothic (≈65%)' in asset.vision_caption

    assert supabase_calls, 'supabase logging should be attempted'
    meta = supabase_calls[0]['meta']
    assert meta['weather_final'] == 'rain'
    assert meta['season_final'] == 'autumn'
    assert meta['arch_style'] == {'label': 'Gothic', 'confidence': 0.65}


@pytest.mark.asyncio
async def test_job_vision_caption_entities_utf16_length(tmp_path, monkeypatch):
    overrides = {
        'caption': '⚠️ тест',
        'tags': ['animals', 'sunny'],
    }

    calls, _, _ = await _run_vision_job_collect_calls(
        tmp_path,
        monkeypatch,
        flag_enabled=False,
        vision_overrides=overrides,
    )

    publish_calls = [
        call
        for call in calls
        if call['method'] in {'copyMessage', 'sendPhoto', 'sendDocument'}
    ]
    assert publish_calls, 'vision job should publish recognition results'
    payload = publish_calls[0]['data']
    assert payload is not None
    caption_text = payload.get('caption')
    assert isinstance(caption_text, str) and caption_text
    caption_entities = payload.get('caption_entities')
    assert isinstance(caption_entities, list) and caption_entities
    expected_length = len(caption_text.encode('utf-16-le')) // 2
    assert caption_entities[0]['length'] == expected_length


@pytest.mark.asyncio
async def test_asset_selection(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.add_asset(1, '#дождь', 'cap')
    bot.add_asset(2, '', 'cap2')
    a = bot.next_asset({'#дождь'})
    assert a['message_id'] == 1
    a2 = bot.next_asset(None)
    assert a2['message_id'] == 2
    await bot.close()

@pytest.mark.asyncio
async def test_render_date(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    tomorrow = datetime.utcnow().date() + timedelta(days=1)
    tpl = 'date {next-day-date} {next-day-month}'
    result = bot._render_template(tpl)
    assert str(tomorrow.day) in result
    await bot.close()

@pytest.mark.asyncio
async def test_weather_scheduler_publish(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_weather_assets_channel(-100)
    calls = []
    async def dummy(method, data=None):
        calls.append((method, data))
        return {'ok': True}
    bot.api_request = dummy  # type: ignore
    bot.add_asset(1, '', 'hi')
    bot.add_weather_channel(-100, (datetime.utcnow() + timedelta(minutes=-1)).strftime('%H:%M'))
    await bot.process_weather_channels()
    assert any(c[0]=='copyMessage' for c in calls)
    await bot.close()



@pytest.mark.asyncio
async def test_publish_weather_uses_migrated_legacy_assets(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE asset_images (
            message_id INTEGER PRIMARY KEY,
            hashtags TEXT,
            template TEXT,
            used_at TEXT
        );
        INSERT INTO asset_images (message_id, hashtags, template, used_at)
        VALUES (101, '#дождь #котопогода', 'legacy caption', '2024-01-01T00:00:00');
        CREATE TABLE IF NOT EXISTS asset_channel (
            channel_id INTEGER PRIMARY KEY
        );
        INSERT INTO asset_channel (channel_id) VALUES (-100123);
        """
    )
    conn.commit()
    conn.close()

    bot = Bot('dummy', str(db_path))

    calls = []

    async def fake_api_request(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        return {'ok': True, 'result': {'message_id': 777}}

    bot.api_request = fake_api_request  # type: ignore[assignment]

    pre_row = bot.db.execute(
        "SELECT hashtags, categories, channel_id, tg_chat_id FROM assets WHERE message_id=?",
        (101,),
    ).fetchone()
    assert pre_row is not None
    pre_categories = json.loads(pre_row['categories'])
    assert '#дождь' in pre_categories and '#котопогода' in pre_categories
    assert pre_row['channel_id'] == -100123
    assert pre_row['tg_chat_id'] == -100123

    ok = await bot.publish_weather(-100500, {'#дождь'}, record=False)
    assert ok

    copy_calls = [call for call in calls if call[0] == 'copyMessage']
    assert copy_calls
    copy_payload = copy_calls[0][1]
    assert copy_payload['message_id'] == 101
    assert copy_payload['from_chat_id'] == -100123

    delete_calls = [payload for method, payload in calls if method == 'deleteMessage']
    assert delete_calls and delete_calls[0]['message_id'] == 101

    legacy_table = bot.db.execute(
        "SELECT name FROM sqlite_master WHERE name='asset_images'",
    ).fetchone()
    assert legacy_table is None

    row = bot.db.execute(
        "SELECT hashtags, categories, channel_id, tg_chat_id FROM assets WHERE message_id=?",
        (101,),
    ).fetchone()
    assert row is None

    await bot.close()


@pytest.mark.asyncio
async def test_publish_weather_retries_when_source_missing(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))

    first_id = bot.data.save_asset(
        channel_id=-100001,
        message_id=101,
        template='first template',
        hashtags='#first',
        tg_chat_id=-100001,
        caption='first caption',
        kind='photo',
        file_meta={'file_id': 'f1', 'file_unique_id': 'u1'},
        origin='weather',
    )

    second_id = bot.data.save_asset(
        channel_id=-100002,
        message_id=102,
        template='second template',
        hashtags='#second',
        tg_chat_id=-100002,
        caption='second caption',
        kind='photo',
        file_meta={'file_id': 'f2', 'file_unique_id': 'u2'},
        origin='weather',
    )

    copy_calls: list[dict[str, Any]] = []
    delete_calls: list[dict[str, Any]] = []
    responses = [
        {
            'ok': False,
            'error_code': 400,
            'description': 'Bad Request: message to copy not found',
        },
        {'ok': True, 'result': {'message_id': 999}},
    ]

    async def fake_api_request(method, data=None, *, files=None):  # type: ignore[override]
        if method == 'copyMessage':
            copy_calls.append(data)
            if responses:
                return responses.pop(0)
            return {'ok': True, 'result': {'message_id': 1000}}
        if method == 'deleteMessage':
            delete_calls.append(data)
            return {'ok': True, 'result': True}
        return {'ok': True, 'result': {}}

    bot.api_request = fake_api_request  # type: ignore[assignment]

    ok = await bot.publish_weather(-400001, None, record=False)
    assert ok

    assert len(copy_calls) == 2
    assert copy_calls[0]['message_id'] == 101
    assert copy_calls[1]['message_id'] == 102

    assert bot.data.get_asset(first_id) is None
    assert bot.data.get_asset(second_id) is None
    assert delete_calls and delete_calls[0]['message_id'] == 102

    await bot.close()


@pytest.mark.asyncio

async def test_handle_asset_message(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_weather_assets_channel(-100123)
    msg = {
        'message_id': 10,
        'chat': {'id': -100123},
        'caption': '#котопогода #дождь cap'
    }
    await bot.handle_message(msg)

    a = bot.next_asset({'#дождь'})
    assert a['message_id'] == 10
    await bot.close()


@pytest.mark.asyncio

async def test_edit_asset(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_weather_assets_channel(-100123)
    msg = {
        'message_id': 11,
        'chat': {'id': -100123},
        'caption': '#котопогода old'
    }
    await bot.handle_message(msg)
    edit = {
        'message_id': 11,
        'chat': {'id': -100123},
        'caption': '#котопогода #новый new'
    }
    await bot.handle_edited_message(edit)
    a = bot.next_asset({'#новый'})
    assert a and a['message_id'] == 11
    await bot.close()


@pytest.mark.asyncio

async def test_photo_triggers_ingest_and_vision(tmp_path, caplog):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_recognition_channel(-100123)
    message = {
        'message_id': 21,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -100123},
        'caption': '#котопогода свежий кадр',
        'from': {'id': 777, 'username': 'catlover'},
        'photo': [
            {'file_id': 'small', 'file_unique_id': 'uniq_small', 'file_size': 10, 'width': 90, 'height': 90},
            {'file_id': 'large', 'file_unique_id': 'uniq_large', 'file_size': 20, 'width': 1920, 'height': 1080},
        ],
    }
    with caplog.at_level('INFO'):
        await bot.handle_message(message)
    rows = bot.db.execute(
        "SELECT id, name, payload FROM jobs_queue ORDER BY id"
    ).fetchall()
    assert rows and rows[0]['name'] == 'ingest'
    payload = json.loads(rows[0]['payload'])
    asset_id = payload['asset_id']
    asset = bot.data.get_asset(asset_id)
    assert asset.kind == 'photo'
    assert asset.file_id == 'large'
    assert asset.author_user_id == 777
    await bot.handle_message(message)
    ingest_jobs = bot.db.execute(
        "SELECT COUNT(*) FROM jobs_queue WHERE name='ingest'"
    ).fetchone()[0]
    assert ingest_jobs == 1
    job = Job(
        id=rows[0]['id'],
        name='ingest',
        payload=payload,
        status='queued',
        attempts=0,
        available_at=None,
        last_error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    with caplog.at_level('INFO'):
        await bot._job_ingest(job)
    vision_rows = bot.db.execute(
        "SELECT name, payload FROM jobs_queue WHERE name='vision'"
    ).fetchall()
    assert vision_rows
    vision_payload = json.loads(vision_rows[0]['payload'])
    assert vision_payload['asset_id'] == asset_id
    ingest_messages = [rec.message for rec in caplog.records if 'Scheduled ingest job' in rec.message]
    ingest_log = bool(ingest_messages)
    vision_log = any('queued for vision job' in rec.message for rec in caplog.records)
    assert ingest_log and vision_log
    assert any('reason=new_message' in msg for msg in ingest_messages)
    await bot.close()


@pytest.mark.asyncio

async def test_recognized_message_skips_reingest(tmp_path):
    bot = Bot('test-token', str(tmp_path / 'db.sqlite'))
    bot.set_recognition_channel(-100123)

    img = Image.new('RGB', (10, 10), color='white')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()

    recognized_mid = 555

    async def fake_download(file_id, dest_path=None):  # type: ignore[override]
        assert dest_path is not None
        Path(dest_path).write_bytes(image_bytes)
        return Path(dest_path)

    async def fake_api_request(method, data=None, *, files=None):  # type: ignore[override]
        if method == 'copyMessage':
            return {'ok': True, 'result': {'message_id': recognized_mid}}
        if method == 'sendPhoto':
            return {'ok': True, 'result': {'message_id': recognized_mid}}
        return {'ok': True, 'result': {}}

    async def fake_api_request_multipart(method, data=None, *, files=None):  # type: ignore[override]
        return await fake_api_request(method, data=data, files=files)

    class DummyOpenAI:
        def __init__(self):
            self.api_key = 'test-key'

        async def classify_image(self, **kwargs):
            return OpenAIResponse(
                {
                    'arch_view': False,
                    'caption': 'кот',
                    'objects': ['кот'],
                    'is_outdoor': False,
                    'framing': 'close_up',
                    'guess_country': None,
                    'guess_city': None,
                    'location_confidence': 0.0,
                    'landmarks': [],
                    'tags': ['animals', 'sunny', 'pet'],
                    'architecture_close_up': False,
                    'architecture_wide': False,
                    'weather_image': 'sunny',
                    'season_guess': None,
                    'arch_style': None,
                    'safety': {'nsfw': False, 'reason': 'безопасно'},
                },
                {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15,
                    'request_id': 'req-1',
                    'endpoint': '/v1/responses',
                },
            )

    bot._download_file = fake_download  # type: ignore[assignment]
    bot.api_request = fake_api_request  # type: ignore[assignment]
    bot.api_request_multipart = fake_api_request_multipart  # type: ignore[assignment]
    bot.openai = DummyOpenAI()

    message = {
        'message_id': 42,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -100123},
        'caption': '#котопогода исходник',
        'photo': [
            {
                'file_id': 'ph_small',
                'file_unique_id': 'uniq_small',
                'file_size': 10,
                'width': 320,
                'height': 200,
            },
            {
                'file_id': 'ph_large',
                'file_unique_id': 'uniq_large',
                'file_size': 30,
                'width': 1920,
                'height': 1080,
            },
        ],
    }

    await bot.handle_message(message)

    ingest_row = bot.db.execute(
        "SELECT * FROM jobs_queue WHERE name='ingest' ORDER BY id LIMIT 1"
    ).fetchone()
    assert ingest_row is not None
    ingest_payload = json.loads(ingest_row['payload']) if ingest_row['payload'] else {}
    ingest_job = Job(
        id=ingest_row['id'],
        name=ingest_row['name'],
        payload=ingest_payload,
        status=ingest_row['status'],
        attempts=ingest_row['attempts'],
        available_at=datetime.fromisoformat(ingest_row['available_at'])
        if ingest_row['available_at']
        else None,
        last_error=ingest_row['last_error'],
        created_at=datetime.fromisoformat(ingest_row['created_at']),
        updated_at=datetime.fromisoformat(ingest_row['updated_at']),
    )

    await bot._job_ingest(ingest_job)

    vision_row = bot.db.execute(
        "SELECT * FROM jobs_queue WHERE name='vision' ORDER BY id LIMIT 1"
    ).fetchone()
    assert vision_row is not None
    vision_payload = json.loads(vision_row['payload']) if vision_row['payload'] else {}
    vision_job = Job(
        id=vision_row['id'],
        name=vision_row['name'],
        payload=vision_payload,
        status=vision_row['status'],
        attempts=vision_row['attempts'],
        available_at=datetime.fromisoformat(vision_row['available_at'])
        if vision_row['available_at']
        else None,
        last_error=vision_row['last_error'],
        created_at=datetime.fromisoformat(vision_row['created_at']),
        updated_at=datetime.fromisoformat(vision_row['updated_at']),
    )

    await bot._job_vision(vision_job)

    asset_id = ingest_payload['asset_id']
    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.recognized_message_id == recognized_mid

    bot.db.execute('DELETE FROM jobs_queue')
    bot.db.commit()

    asset_count = bot.db.execute('SELECT COUNT(*) FROM assets').fetchone()[0]

    recognized_message = {
        'message_id': recognized_mid,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -100123},
        'caption': 'Распознано: кот\nУверенность в локации: 0%\nОбстановка: солнечно\nНа улице: нет\nАрхитектура: нет\nОбъекты: кот\nТеги: animals, sunny, pet\nБезопасность: безопасно',
        'photo': [
            {
                'file_id': 'vision_small',
                'file_unique_id': 'vision_small_unique',
                'file_size': 12,
                'width': 320,
                'height': 200,
            },
            {
                'file_id': 'vision_large',
                'file_unique_id': 'vision_large_unique',
                'file_size': 34,
                'width': 1920,
                'height': 1080,
            },
        ],
    }

    await bot.handle_message(recognized_message)

    ingest_jobs = bot.db.execute(
        "SELECT COUNT(*) FROM jobs_queue WHERE name='ingest'"
    ).fetchone()[0]
    assert ingest_jobs == 0
    asset_count_after = bot.db.execute('SELECT COUNT(*) FROM assets').fetchone()[0]
    assert asset_count_after == asset_count

    await bot.close()


@pytest.mark.asyncio
async def test_recognized_edit_skips_reingest(tmp_path):
    bot = Bot('test-token', str(tmp_path / 'db.sqlite'))
    bot.set_recognition_channel(-100123)

    img = Image.new('RGB', (10, 10), color='white')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()

    recognized_mid = 777

    async def fake_download(file_id, dest_path=None):  # type: ignore[override]
        assert dest_path is not None
        Path(dest_path).write_bytes(image_bytes)
        return Path(dest_path)

    async def fake_api_request(method, data=None, *, files=None):  # type: ignore[override]
        if method == 'copyMessage':
            return {'ok': True, 'result': {'message_id': recognized_mid}}
        if method == 'sendPhoto':
            return {'ok': True, 'result': {'message_id': recognized_mid}}
        return {'ok': True, 'result': {}}

    class DummyOpenAI:
        def __init__(self):
            self.api_key = 'test-key'

        async def classify_image(self, **kwargs):
            return OpenAIResponse(
                {
                    'arch_view': False,
                    'caption': 'кот',
                    'objects': ['кот'],
                    'is_outdoor': False,
                    'framing': 'close_up',
                    'guess_country': None,
                    'guess_city': None,
                    'location_confidence': 0.0,
                    'landmarks': [],
                    'tags': ['animals', 'sunny', 'pet'],
                    'architecture_close_up': False,
                    'architecture_wide': False,
                    'weather_image': 'sunny',
                    'season_guess': None,
                    'arch_style': None,
                    'safety': {'nsfw': False, 'reason': 'безопасно'},
                },
                {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15,
                    'request_id': 'req-2',
                    'endpoint': '/v1/responses',
                },
            )

    bot._download_file = fake_download  # type: ignore[assignment]
    bot.api_request = fake_api_request  # type: ignore[assignment]
    bot.openai = DummyOpenAI()

    message = {
        'message_id': 99,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -100123},
        'caption': '#котопогода исходник',
        'photo': [
            {
                'file_id': 'ph_small',
                'file_unique_id': 'uniq_small',
                'file_size': 10,
                'width': 320,
                'height': 200,
            },
            {
                'file_id': 'ph_large',
                'file_unique_id': 'uniq_large',
                'file_size': 30,
                'width': 1920,
                'height': 1080,
            },
        ],
    }

    await bot.handle_message(message)

    ingest_row = bot.db.execute(
        "SELECT * FROM jobs_queue WHERE name='ingest' ORDER BY id LIMIT 1"
    ).fetchone()
    assert ingest_row is not None
    ingest_payload = json.loads(ingest_row['payload']) if ingest_row['payload'] else {}
    ingest_job = Job(
        id=ingest_row['id'],
        name=ingest_row['name'],
        payload=ingest_payload,
        status=ingest_row['status'],
        attempts=ingest_row['attempts'],
        available_at=datetime.fromisoformat(ingest_row['available_at'])
        if ingest_row['available_at']
        else None,
        last_error=ingest_row['last_error'],
        created_at=datetime.fromisoformat(ingest_row['created_at']),
        updated_at=datetime.fromisoformat(ingest_row['updated_at']),
    )

    await bot._job_ingest(ingest_job)

    vision_row = bot.db.execute(
        "SELECT * FROM jobs_queue WHERE name='vision' ORDER BY id LIMIT 1"
    ).fetchone()
    assert vision_row is not None
    vision_payload = json.loads(vision_row['payload']) if vision_row['payload'] else {}
    vision_job = Job(
        id=vision_row['id'],
        name=vision_row['name'],
        payload=vision_payload,
        status=vision_row['status'],
        attempts=vision_row['attempts'],
        available_at=datetime.fromisoformat(vision_row['available_at'])
        if vision_row['available_at']
        else None,
        last_error=vision_row['last_error'],
        created_at=datetime.fromisoformat(vision_row['created_at']),
        updated_at=datetime.fromisoformat(vision_row['updated_at']),
    )

    await bot._job_vision(vision_job)

    asset_id = ingest_payload['asset_id']
    asset = bot.data.get_asset(asset_id)
    assert asset is not None
    assert asset.recognized_message_id == recognized_mid

    bot.db.execute('DELETE FROM jobs_queue')
    bot.db.commit()

    asset_count = bot.db.execute('SELECT COUNT(*) FROM assets').fetchone()[0]

    edited_message = {
        'message_id': recognized_mid,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -100123},
        'caption': 'Распознано: кот (ред.)',
        'photo': [
            {
                'file_id': 'vision_small',
                'file_unique_id': 'vision_small_unique',
                'file_size': 12,
                'width': 320,
                'height': 200,
            },
            {
                'file_id': 'vision_large',
                'file_unique_id': 'vision_large_unique',
                'file_size': 34,
                'width': 1920,
                'height': 1080,
            },
        ],
    }

    await bot.handle_edited_message(edited_message)

    ingest_jobs = bot.db.execute(
        "SELECT COUNT(*) FROM jobs_queue WHERE name='ingest'"
    ).fetchone()[0]
    assert ingest_jobs == 0
    asset_count_after = bot.db.execute('SELECT COUNT(*) FROM assets').fetchone()[0]
    assert asset_count_after == asset_count

    await bot.close()


@pytest.mark.asyncio
async def test_weather_and_recognition_channels_do_not_mix(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_weather_assets_channel(-200001)
    bot.set_recognition_channel(-300001)

    calls: list[tuple[str, dict | None]] = []

    async def fake_api_request(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        return {'ok': True, 'result': {'message_id': 900}}

    bot.api_request = fake_api_request  # type: ignore[assignment]

    recognition_message = {
        'message_id': 11,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -300001},
        'caption': '#распознавание кот',
        'photo': [
            {
                'file_id': 'rec_photo',
                'file_unique_id': 'rec_unique',
                'file_size': 10,
                'width': 640,
                'height': 480,
            }
        ],
    }

    weather_message = {
        'message_id': 22,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -200001},
        'caption': '#погода готово',
    }

    await bot.handle_message(recognition_message)
    await bot.handle_message(weather_message)

    row = bot.db.execute(
        "SELECT origin FROM assets WHERE tg_chat_id=? AND message_id=?",
        (-300001, 11),
    ).fetchone()
    assert row is not None and row['origin'] == 'recognition'

    weather_asset_row = bot.db.execute(
        "SELECT id FROM assets WHERE tg_chat_id=? AND message_id=?",
        (-200001, 22),
    ).fetchone()
    assert weather_asset_row is not None
    weather_asset_id = weather_asset_row['id']

    ok = await bot.publish_weather(-400001, None, record=False)
    assert ok

    copy_calls = [payload for method, payload in calls if method == 'copyMessage']
    assert copy_calls and copy_calls[0]['message_id'] == 22
    assert copy_calls[0]['from_chat_id'] == -200001
    delete_calls = [payload for method, payload in calls if method == 'deleteMessage']
    assert delete_calls and delete_calls[0]['message_id'] == 22

    assert bot.data.get_asset(weather_asset_id) is None

    await bot.close()


@pytest.mark.asyncio
async def test_weather_publish_survives_shared_channel_then_splits(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_weather_assets_channel(-200001)

    calls: list[tuple[str, dict | None]] = []

    async def fake_api_request(method, data=None, *, files=None):  # type: ignore[override]
        calls.append((method, data))
        return {'ok': True, 'result': {'message_id': 700}}

    bot.api_request = fake_api_request  # type: ignore[assignment]

    weather_message = {
        'message_id': 33,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -200001},
        'caption': '#погода шаблон',
    }

    await bot.handle_message(weather_message)

    weather_row = bot.db.execute(
        "SELECT origin, id FROM assets WHERE tg_chat_id=? AND message_id=?",
        (-200001, 33),
    ).fetchone()
    assert weather_row is not None and weather_row['origin'] == 'weather'
    weather_asset_id = weather_row['id']

    ok = await bot.publish_weather(-400001, None, record=False)
    assert ok

    copy_calls = [payload for method, payload in calls if method == 'copyMessage']
    assert copy_calls and copy_calls[0]['message_id'] == 33
    assert copy_calls[0]['from_chat_id'] == -200001

    delete_calls = [payload for method, payload in calls if method == 'deleteMessage']
    assert delete_calls and delete_calls[0]['message_id'] == 33

    assert bot.data.get_asset(weather_asset_id) is None

    pre_ingest_jobs = bot.db.execute(
        "SELECT COUNT(*) FROM jobs_queue WHERE name='ingest'",
    ).fetchone()[0]
    assert pre_ingest_jobs == 0

    bot.set_recognition_channel(-300001)

    recognition_message = {
        'message_id': 44,
        'date': int(datetime.utcnow().timestamp()),
        'chat': {'id': -300001},
        'caption': '#распознавание кот',
        'photo': [
            {
                'file_id': 'rec_photo',
                'file_unique_id': 'rec_unique',
                'file_size': 10,
                'width': 640,
                'height': 480,
            }
        ],
    }

    await bot.handle_message(recognition_message)

    recognition_row = bot.db.execute(
        "SELECT origin FROM assets WHERE tg_chat_id=? AND message_id=?",
        (-300001, 44),
    ).fetchone()
    assert recognition_row is not None and recognition_row['origin'] == 'recognition'

    ingest_jobs = bot.db.execute(
        "SELECT COUNT(*) FROM jobs_queue WHERE name='ingest'",
    ).fetchone()[0]
    assert ingest_jobs == 1

    await bot.close()


@pytest.mark.asyncio

async def test_template_russian_and_period(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    # insert cached weather and sea data
    bot.db.execute(
        "INSERT INTO weather_cache_hour (city_id, timestamp, temperature, weather_code, wind_speed, is_day)"
        " VALUES (1, ?, 20.0, 1, 5.0, 1)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.execute(

        "INSERT INTO weather_cache_period (city_id, updated, morning_temp, morning_code, morning_wind, day_temp, day_code, day_wind, evening_temp, evening_code, evening_wind, night_temp, night_code, night_wind)"
        " VALUES (1, ?, 21.0, 1, 4.0, 22.0, 2, 5.0, 23.0, 3, 6.0, 24.0, 4, 7.0)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.execute(

        "INSERT INTO sea_cache (sea_id, updated, current, morning, day, evening, night)"
        " VALUES (1, ?, 15.0, 15.1, 15.2, 15.3, 15.4)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()
    tpl = '{next-day-date} {next-day-month} {1|nm-temp} {1|nd-seatemperature}'
    result = bot._render_template(tpl)

    assert '15.' in result and '21\u00B0C' in result
    months = ['января','февраля','марта','апреля','мая','июня','июля','августа','сентября','октября','ноября','декабря']
    assert any(m in result for m in months)
    await bot.close()


@pytest.mark.asyncio
async def test_seastorm_render(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))

    bot.db.execute(
        "INSERT INTO sea_cache (sea_id, updated, current, morning, day, evening, night, wave, morning_wave, day_wave, evening_wave, night_wave)"
        " VALUES (1, ?, 15.0, 15.1, 15.2, 15.3, 15.4, 0.2, 0.4, 0.6, 1.6, 0.3)",
        (datetime.utcnow().isoformat(),),
    )
    bot.db.commit()

    assert bot._render_template('{1|seastorm}') == '\U0001F30A 15.0\u00B0C'
    assert bot._render_template('{1|nd-seastorm}') == '\U0001F30A шторм'
    assert bot._render_template('{1|ny-seastorm}') == '\U0001F30A сильный шторм'
    await bot.close()


def test_strip_header():
    assert Bot.strip_header('🌊 16°C∙text') == 'text'
    assert Bot.strip_header('prefix ∙ data') == 'prefix ∙ data'



@pytest.mark.asyncio
async def test_migrate_legacy_weather_channels(tmp_path):
    db_path = tmp_path / 'db.sqlite'
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE weather_publish_channels ("
        " channel_id INTEGER PRIMARY KEY,"
        " post_time TEXT NOT NULL,"
        " last_published_at TEXT"
        ")"
    )
    conn.execute(
        "INSERT INTO weather_publish_channels (channel_id, post_time, last_published_at) VALUES (?, ?, ?)",
        (-100500, '10:30', '2024-01-01T09:00:00')
    )
    conn.commit()
    conn.close()

    bot = Bot('dummy', str(db_path))
    channels = bot.list_weather_channels()
    assert channels and channels[0]['channel_id'] == -100500
    assert channels[0]['post_time'] == '10:30'
    assert channels[0]['last_published_at'] == '2024-01-01T09:00:00'
    assert (
        bot.db.execute("SELECT name FROM sqlite_master WHERE name='weather_publish_channels'").fetchone()
        is None
    )
    await bot.close()
