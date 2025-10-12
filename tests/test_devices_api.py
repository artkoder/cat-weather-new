import sys
from pathlib import Path

import os

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
    await bot.start()

    create_pairing_token(bot.db, code='CATPAA', user_id=101, device_name='Office Pixel')
    bot.db.commit()

    app = await _make_app(bot)

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            resp = await client.post(
                '/v1/devices/attach',
                json={'code': 'catpaa', 'name': 'Pixel 8'},
            )
            payload = await resp.json()

    assert resp.status == 200
    assert payload['name'] == 'Pixel 8'
    assert payload['id']
    assert payload['secret']
    assert '=' not in payload['secret']

    row = bot.db.execute(
        'SELECT user_id, name, secret FROM devices WHERE id=?', (payload['id'],)
    ).fetchone()
    assert row is not None
    assert row['user_id'] == 101
    assert row['name'] == 'Pixel 8'
    assert row['secret'] == payload['secret']

    token_row = bot.db.execute(
        'SELECT used_at FROM pairing_tokens WHERE code=?', ('CATPAA',)
    ).fetchone()
    assert token_row is not None and token_row['used_at'] is not None

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
                json={'code': 'bad-code'},
            )
            assert resp.status == 400
            payload = await resp.json()
            assert payload['error'] == 'invalid_or_expired_code'

    await bot.close()
