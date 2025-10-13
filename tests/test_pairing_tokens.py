import pytest

from api.pairing import PairingTokenError, normalize_pairing_token


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"token": "abc234"}, "ABC234"),
        ({"token": "  catpaa  "}, "CATPAA"),
        ({"code": "pair:wwf5ll"}, "WWF5LL"),
        ({"token": "PAIR:kkm2zz"}, "KKM2ZZ"),
        ({"token": "catweather://pair?token=XY2Z34"}, "XY2Z34"),
        ({"code": "catweather://pair?code=XY2Z34"}, "XY2Z34"),
        ({"token": "catweather://pair?token=qp2pqr&extra=1"}, "QP2PQR"),
    ],
)
def test_normalize_pairing_token_success(payload, expected):
    assert normalize_pairing_token(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"token": ""},
        {"token": "   "},
        {"code": "   "},
        {"token": "pair:"},
        {"token": "catweather://pair"},
        {"token": "catweather://pair?foo=bar"},
        {"token": "catweather://example?token=ABC123"},
        {"token": "http://pair?token=ABC123"},
        {"token": "bad-code"},
        {"token": "abc"},
        {"token": "abcd1234"},
    ],
)
def test_normalize_pairing_token_errors(payload):
    with pytest.raises(PairingTokenError):
        normalize_pairing_token(payload)
