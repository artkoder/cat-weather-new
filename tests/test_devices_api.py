import sys
from pathlib import Path

import logging
import os
import re

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.rate_limit import TokenBucketLimiter, create_rate_limit_middleware
from data_access import create_pairing_token
from main import Bot, attach_device


async def _make_app(bot: Bot) -> web.Application:
    app = web.Application(middlewares=[create_rate_limit_middleware()])
    app['bot'] = bot
    limit = int(os.getenv('RL_ATTACH_USER_PER_MIN', '3'))
    window = int(os.getenv('RL_ATTACH_USER_WINDOW_SEC', '60'))
    app['attach_user_rate_limiter'] = TokenBucketLimiter(limit, window)
    app.router.add_post('/v1/devices/attach', attach_device)
    return app


@pytest.mark.asyncio
async def test_attach_device_success(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_asset_channel(12345)
    await bot.start()

    create_pairing_token(bot.db, code='CATPAA', user_id=101, device_name='Office Pixel')
    bot.db.commit()

    app = await _make_app(bot)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            resp = await client.post(
                '/v1/devices/attach',
                json={'token': 'catpaa', 'name': 'Pixel 8'},
            )
            payload = await resp.json()

    assert resp.status == 200
    assert payload['name'] == 'Pixel 8'
    assert payload['device_id']
    assert payload['device_secret']
    assert payload['id'] == payload['device_id']
    assert payload['secret'] == payload['device_secret']
    assert re.fullmatch(r'[0-9a-f]{64}', payload['device_secret'])

    row = bot.db.execute(
        'SELECT user_id, name, secret FROM devices WHERE id=?', (payload['device_id'],)
    ).fetchone()
    assert row is not None
    assert row['user_id'] == 101
    assert row['name'] == 'Pixel 8'
    assert row['secret'] == payload['device_secret']

    token_row = bot.db.execute(
        'SELECT used_at FROM pairing_tokens WHERE code=?', ('CATPAA',)
    ).fetchone()
    assert token_row is not None and token_row['used_at'] is not None

    await bot.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'submitted_code, token_code, user_id',
    [
        ('PAIR:WWF5LL', 'WWF5LL', 202),
        ('catweather://pair?code=TR7X9Z', 'TR7X9Z', 303),
        ('catweather://pair?token=XY2Z34', 'XY2Z34', 404),
    ],
)
async def test_attach_device_accepts_prefixed_payloads(
    tmp_path, submitted_code, token_code, user_id
):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_asset_channel(54321)
    await bot.start()

    create_pairing_token(bot.db, code=token_code, user_id=user_id, device_name='Office Pixel')
    bot.db.commit()

    app = await _make_app(bot)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            resp = await client.post('/v1/devices/attach', json={'code': submitted_code})
            payload = await resp.json()

    assert resp.status == 200
    assert payload['name'] == 'Office Pixel'
    assert payload['device_id']
    assert payload['device_secret']
    assert payload['id'] == payload['device_id']
    assert payload['secret'] == payload['device_secret']
    assert re.fullmatch(r'[0-9a-f]{64}', payload['device_secret'])

    row = bot.db.execute(
        'SELECT user_id, name FROM devices WHERE id=?', (payload['device_id'],)
    ).fetchone()
    assert row is not None
    assert row['user_id'] == user_id
    assert row['name'] == 'Office Pixel'

    token_row = bot.db.execute(
        'SELECT used_at FROM pairing_tokens WHERE code=?', (token_code,)
    ).fetchone()
    assert token_row is not None and token_row['used_at'] is not None

    await bot.close()


@pytest.mark.asyncio
async def test_attach_device_emits_mobile_log(tmp_path, caplog):
    caplog.set_level(logging.INFO)

    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    bot.set_asset_channel(12345)
    await bot.start()

    create_pairing_token(bot.db, code='CATPAA', user_id=101, device_name='Office Pixel')
    bot.db.commit()

    app = await _make_app(bot)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            resp = await client.post(
                '/v1/devices/attach',
                json={'token': 'CATPAA', 'name': 'Pixel 8'},
            )
            payload = await resp.json()

    assert resp.status == 200

    attach_event = next(
        record for record in caplog.records if record.message == 'MOBILE_ATTACH_OK'
    )
    assert attach_event.device_id == payload['device_id']
    assert attach_event.user_id == 101
    assert attach_event.device_name == 'Pixel 8'
    assert attach_event.source == 'mobile'
    assert isinstance(attach_event.timestamp, str) and attach_event.timestamp

    await bot.close()


@pytest.mark.asyncio
async def test_attach_device_rate_limited(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    await bot.start()

    app = await _make_app(bot)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            for offset in range(3):
                code = f'CATP{chr(65 + offset)}A'
                create_pairing_token(
                    bot.db,
                    code=code,
                    user_id=7,
                    device_name='Android',
                )
                bot.db.commit()
                resp = await client.post(
                    '/v1/devices/attach',
                    json={'code': code},
                )
                assert resp.status == 200
            create_pairing_token(
                bot.db,
                code='CATPDA',
                user_id=7,
                device_name='Android',
            )
            bot.db.commit()
            resp = await client.post(
                '/v1/devices/attach',
                json={'code': 'CATPDA'},
            )
            assert resp.status == 429
            payload = await resp.json()
            assert payload['error'] == 'rate_limited'

    await bot.close()


@pytest.mark.asyncio
async def test_attach_device_ip_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setenv('RL_ATTACH_IP_PER_MIN', '2')
    monkeypatch.setenv('RL_ATTACH_USER_PER_MIN', '100')
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    await bot.start()

    app = await _make_app(bot)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            for offset in range(2):
                code = f'CATZ{chr(65 + offset)}A'
                create_pairing_token(
                    bot.db,
                    code=code,
                    user_id=offset + 1,
                    device_name='Android',
                )
                bot.db.commit()
                resp = await client.post('/v1/devices/attach', json={'code': code})
                assert resp.status == 200
            create_pairing_token(
                bot.db,
                code='CATZQA',
                user_id=99,
                device_name='Android',
            )
            bot.db.commit()
            resp = await client.post('/v1/devices/attach', json={'code': 'CATZQA'})
            assert resp.status == 429
            payload = await resp.json()
            assert payload['error'] == 'rate_limited'

    await bot.close()


@pytest.mark.asyncio
async def test_attach_device_invalid_code(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    await bot.start()

    app = await _make_app(bot)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            resp = await client.post(
                '/v1/devices/attach',
                json={'token': 'bad-code'},
            )
            assert resp.status == 400
            payload = await resp.json()
            assert payload['error'] == 'invalid_token'
            assert (
                payload['message']
                == 'Недопустимый формат токена: ожидаем 6–8 символов A-Z и 2-9.'
            )

    await bot.close()


@pytest.mark.asyncio
async def test_attach_device_rejects_consumed_token(tmp_path):
    bot = Bot('dummy', str(tmp_path / 'db.sqlite'))
    await bot.start()

    create_pairing_token(bot.db, code='CATPZZ', user_id=17, device_name='Spare iPhone')
    bot.db.commit()

    app = await _make_app(bot)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            first = await client.post('/v1/devices/attach', json={'token': 'CATPZZ'})
            assert first.status == 200

            second = await client.post('/v1/devices/attach', json={'token': 'CATPZZ'})
            assert second.status == 400
            payload = await second.json()

    assert payload == {
        'error': 'invalid_token',
        'message': 'Токен недействителен или срок его действия истёк.',
    }

    await bot.close()
