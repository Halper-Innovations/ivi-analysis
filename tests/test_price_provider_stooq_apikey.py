from __future__ import annotations

import json

import requests

from app.config import get_config


class _Resp:
    status_code = 200
    text = "Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,9.0,100\n"

    def raise_for_status(self):
        return None


def test_stooq_download_includes_apikey_when_set(monkeypatch):
    monkeypatch.setenv("VOE_STOOQ_APIKEY", "TESTKEY123")
    get_config.cache_clear()
    from app.market.price_provider import StooqProvider

    captured = {}

    def _fake_get(url, params=None, timeout=None, headers=None):
        captured["url"] = url
        captured["params"] = dict(params or {})
        return _Resp()

    monkeypatch.setattr("app.market.price_provider.requests.get", _fake_get)
    prov = StooqProvider()
    text, request_url, status = prov._download_daily_history("aapl.us")
    assert captured["params"]["s"] == "aapl.us"
    assert captured["params"]["i"] == "d"
    assert captured["params"]["apikey"] == "TESTKEY123"   # default param name
    assert request_url == "https://stooq.com/q/d/l/?s=aapl.us&i=d"
    assert "TESTKEY123" not in request_url
    assert status == 200


def test_stooq_download_no_apikey_by_default(monkeypatch):
    monkeypatch.delenv("VOE_STOOQ_APIKEY", raising=False)
    get_config.cache_clear()
    from app.market.price_provider import StooqProvider

    captured = {}

    def _fake_get(url, params=None, timeout=None, headers=None):
        captured["params"] = dict(params or {})
        return _Resp()

    monkeypatch.setattr("app.market.price_provider.requests.get", _fake_get)
    prov = StooqProvider()
    prov._download_daily_history("aapl.us")
    assert "apikey" not in captured["params"]
    assert captured["params"] == {"s": "aapl.us", "i": "d"}   # behavior unchanged


def test_stooq_history_url_override(monkeypatch):
    monkeypatch.setenv("VOE_STOOQ_HISTORY_URL", "https://example.test/custom/")
    get_config.cache_clear()
    from app.market.price_provider import StooqProvider

    captured = {}

    def _fake_get(url, params=None, timeout=None, headers=None):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr("app.market.price_provider.requests.get", _fake_get)
    prov = StooqProvider()
    prov._download_daily_history("aapl.us")
    assert captured["url"] == "https://example.test/custom/"


def test_stooq_custom_apikey_name_is_redacted_from_error_diagnostic(monkeypatch):
    monkeypatch.setenv("VOE_STOOQ_APIKEY", "CUSTOM-KEY-789")
    monkeypatch.setenv("VOE_STOOQ_APIKEY_PARAM", "vendor_secret")
    # The mocked transport is only reached with the network switch on.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    get_config.cache_clear()
    from app.market.price_provider import StooqProvider

    def _fake_get(url, params=None, timeout=None, headers=None):
        del timeout, headers
        error = requests.HTTPError(
            f"401 client error for {url}?s={params['s']}&vendor_secret={params['vendor_secret']}"
        )
        error.response = type("Response", (), {"status_code": 401})()
        raise error

    monkeypatch.setattr("app.market.price_provider.requests.get", _fake_get)
    provider = StooqProvider()
    assert provider.get_price_asof("AAPL", "2026-02-13") is None
    diagnostic = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diagnostic is not None
    assert diagnostic["result"]["reason_code"] == "HTTP_4XX"
    assert "CUSTOM-KEY-789" not in json.dumps(diagnostic, sort_keys=True)
    assert diagnostic["provider_attempts"][0]["error_detail"] == (
        "401 client error for https://stooq.com/q/d/l/?s=aapl.us&vendor_secret=REDACTED"
    )


def test_stooq_secondary_keeps_www_default(monkeypatch):
    monkeypatch.delenv("VOE_STOOQ_HISTORY_URL", raising=False)
    get_config.cache_clear()
    from app.market.price_provider import StooqSecondaryProvider
    prov = StooqSecondaryProvider()
    assert prov.history_url == "https://www.stooq.com/q/d/l/"


def test_stooq_secondary_respects_history_url_override(monkeypatch):
    monkeypatch.setenv("VOE_STOOQ_HISTORY_URL", "https://example.test/custom/")
    get_config.cache_clear()
    from app.market.price_provider import StooqSecondaryProvider
    prov = StooqSecondaryProvider()
    assert prov.history_url == "https://example.test/custom/"
