from __future__ import annotations

import json

import requests

import app.util.http as http_util
from app.config import get_config
from app.market.price_provider import (
    ChainedPriceProvider,
    EODHDProvider,
    build_price_provider,
    get_default_provider,
)


_WINDOW_PAYLOAD = [
    {"date": "2026-02-11", "open": 10.1, "high": 10.4, "low": 10.0, "close": 10.25, "adjusted_close": 10.2, "volume": 900},
    {"date": "2026-02-12", "open": 10.3, "high": 10.8, "low": 10.2, "close": 10.75, "adjusted_close": 10.7, "volume": 1000},
    {"date": "2026-02-13", "open": 10.8, "high": 11.6, "low": 10.7, "close": 11.5, "adjusted_close": 11.25, "volume": 1100},
]

_STOOQ_CSV = "Date,Open,High,Low,Close,Volume\n2026-02-13,100,102,99,101.25,1000000\n"


class _FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            exc = requests.HTTPError(f"{self.status_code} client error")
            exc.response = self
            raise exc


def _init_cfg(monkeypatch, tmp_path, *, apikey: str | None = "EODKEY123"):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    overrides_path = tmp_path / "price_symbol_overrides.csv"
    overrides_path.write_text("ticker,stooq_symbol\n", encoding="utf-8")
    monkeypatch.setenv("VOE_PRICE_SYMBOL_OVERRIDES_PATH", str(overrides_path))
    monkeypatch.setenv("VOE_PRICE_FALLBACK_DAYS", "7")
    monkeypatch.setenv("VOE_QUOTE_TTL_SECONDS", "86400")
    monkeypatch.delenv("VOE_NET_PROVIDER", raising=False)
    monkeypatch.delenv("VOE_PRICE_PROVIDER", raising=False)
    monkeypatch.delenv("VOE_MAX_REQUESTS_EODHD_DOMAIN", raising=False)
    if apikey is None:
        monkeypatch.delenv("VOE_EODHD_APIKEY", raising=False)
    else:
        monkeypatch.setenv("VOE_EODHD_APIKEY", apikey)
    # Domain budget counters are process-global; reset for hermetic tests.
    monkeypatch.setitem(http_util._GLOBAL_DOMAIN_COUNTS, "eodhd.com", 0)
    monkeypatch.setitem(http_util._GLOBAL_DOMAIN_COUNTS, "stooq.com", 0)
    get_config.cache_clear()
    return get_config()


def _install_fake_get(monkeypatch, response, captured: dict | None = None, calls: dict | None = None):
    def _fake_get(url, params=None, timeout=None, headers=None):
        if calls is not None:
            calls["count"] = calls.get("count", 0) + 1
        if captured is not None:
            captured["url"] = url
            captured["params"] = dict(params or {})
        return response

    monkeypatch.setattr("app.market.price_provider.requests.get", _fake_get)


def test_eodhd_exact_date_hit(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    captured: dict = {}
    _install_fake_get(monkeypatch, _FakeResponse(_WINDOW_PAYLOAD), captured)
    provider = EODHDProvider(cfg)

    snapshot = provider.get_price_asof("AAPL", "2026-02-13")
    assert snapshot is not None
    assert snapshot.ticker == "AAPL"
    assert snapshot.price == 11.25  # adjusted_close, not raw close 11.5
    assert snapshot.as_of_date == "2026-02-13"
    assert snapshot.confidence == "HIGH"
    assert snapshot.source == "eodhd"
    assert snapshot.url == (
        "https://eodhd.com/api/eod/AAPL.US?from=2026-02-06&to=2026-02-13&fmt=json"
    )

    assert captured["url"] == "https://eodhd.com/api/eod/AAPL.US"
    assert captured["params"] == {
        "api_token": "EODKEY123",
        "from": "2026-02-06",
        "to": "2026-02-13",
        "fmt": "json",
    }

    diag = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diag is not None
    assert diag["resolved_symbol"] == "AAPL.US"
    assert diag["result"]["reason_code"] == "PROVIDER_OK"
    assert diag["market_day"]["requested_day_type"] == "TRADING"
    assert diag["provider_attempts"][-1]["provider"] == "eodhd"
    assert diag["provider_attempts"][-1]["status"] == "PROVIDER_OK"
    assert diag["provider_attempts"][-1]["url"] == (
        "https://eodhd.com/api/eod/AAPL.US?from=2026-02-06&to=2026-02-13&fmt=json"
    )
    assert "EODKEY123" not in json.dumps(diag, sort_keys=True)

    # Resolved price persists to the shared disk cache pathway.
    cache_text = (cfg.cache_dir / "prices" / "AAPL.json").read_text(encoding="utf-8")
    assert "EODKEY123" not in cache_text
    cache_payload = json.loads(cache_text)
    assert any(
        row.get("requested_as_of_date") == "2026-02-13" and row.get("source") == "eodhd"
        for row in cache_payload["entries"]
    )


def test_eodhd_weekend_at_or_before_selection(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    captured: dict = {}
    _install_fake_get(monkeypatch, _FakeResponse(_WINDOW_PAYLOAD), captured)
    provider = EODHDProvider(cfg)

    # 2026-02-15 is a Sunday; the last close at-or-before is Friday 2026-02-13.
    snapshot = provider.get_price_asof("AAPL", "2026-02-15")
    assert snapshot is not None
    assert snapshot.as_of_date == "2026-02-13"
    assert snapshot.price == 11.25
    assert snapshot.confidence == "MEDIUM"
    assert captured["params"]["from"] == "2026-02-08"
    assert captured["params"]["to"] == "2026-02-15"

    diag = provider.get_last_diagnostic("AAPL", "2026-02-15")
    assert diag is not None
    assert diag["market_day"]["requested_day_type"] == "NON_TRADING"
    assert diag["market_day"]["fallback_days_checked"] == 3
    assert diag["asof_final_used"] == "2026-02-13"
    assert diag["result"]["reason_code"] == "PROVIDER_OK"


def test_eodhd_uses_close_when_adjusted_close_absent(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = [{"date": "2026-02-13", "open": 12.0, "high": 12.6, "low": 11.9, "close": 12.5, "volume": 100}]
    _install_fake_get(monkeypatch, _FakeResponse(payload))
    provider = EODHDProvider(cfg)

    snapshot = provider.get_price_asof("AAPL", "2026-02-13")
    assert snapshot is not None
    assert snapshot.price == 12.5


def test_eodhd_empty_payload(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _install_fake_get(monkeypatch, _FakeResponse([]))
    provider = EODHDProvider(cfg)

    snapshot = provider.get_price_asof("GONE", "2026-02-13")
    assert snapshot is None
    diag = provider.get_last_diagnostic("GONE", "2026-02-13")
    assert diag is not None
    # Zero-row payloads classify as SYMBOL_NOT_FOUND, mirroring the Stooq convention.
    assert diag["result"]["reason_code"] == "SYMBOL_NOT_FOUND"
    assert diag["provider_attempts"][-1]["status"] == "SYMBOL_NOT_FOUND"


def test_eodhd_no_row_at_or_before_date(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = [{"date": "2026-01-30", "open": 9.0, "high": 9.3, "low": 8.9, "close": 9.1, "adjusted_close": 9.1, "volume": 50}]
    _install_fake_get(monkeypatch, _FakeResponse(payload))
    provider = EODHDProvider(cfg)

    snapshot = provider.get_price_asof("AAPL", "2026-02-13")
    assert snapshot is None
    diag = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diag is not None
    assert diag["result"]["reason_code"] == "NON_TRADING_DAY_NO_FALLBACK"
    assert diag["market_day"]["fallback_days_checked"] == 8


def test_eodhd_http_error_reason_code(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _install_fake_get(monkeypatch, _FakeResponse({"error": "payment required"}, status_code=402))
    provider = EODHDProvider(cfg)

    snapshot = provider.get_price_asof("AAPL", "2026-02-13")
    assert snapshot is None
    diag = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diag is not None
    assert diag["result"]["reason_code"] == "HTTP_4XX"
    assert diag["result"]["retryable"] is False
    assert diag["provider_attempts"][-1]["status"] == "HTTP_4XX"
    assert diag["provider_attempts"][-1]["http_status"] == 402
    assert diag["total_attempts"] == 1


def test_eodhd_budget_exhaustion(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_MAX_REQUESTS_EODHD_DOMAIN", "0")
    get_config.cache_clear()
    cfg = get_config()
    calls: dict = {}
    _install_fake_get(monkeypatch, _FakeResponse(_WINDOW_PAYLOAD), calls=calls)
    provider = EODHDProvider(cfg)

    snapshot = provider.get_price_asof("AAPL", "2026-02-13")
    assert snapshot is None
    assert calls.get("count", 0) == 0  # budget gate fires before any HTTP request
    diag = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diag is not None
    assert diag["result"]["reason_code"] == "BUDGET_EXHAUSTED"
    assert diag["result"]["retryable"] is False
    assert diag["provider_attempts"][-1]["status"] == "BUDGET_EXHAUSTED"


def test_eodhd_leads_chain_when_key_set(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "stooq")
    get_config.cache_clear()
    cfg = get_config()

    provider = get_default_provider(cfg)
    assert isinstance(provider, ChainedPriceProvider)
    assert [p.provider_name for p in provider.providers] == ["eodhd", "stooq", "stooq_secondary"]

    def _fake_get(url, params=None, timeout=None, headers=None):
        if "/api/eod/" in str(url):
            return _FakeResponse(_WINDOW_PAYLOAD)
        return _FakeResponse(_STOOQ_CSV)

    monkeypatch.setattr("app.market.price_provider.requests.get", _fake_get)
    snapshot = provider.get_price_asof("AAPL", "2026-02-13")
    assert snapshot is not None
    assert snapshot.source == "eodhd"
    assert snapshot.price == 11.25

    diag = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diag is not None
    assert [row["provider"] for row in diag["provider_attempts"]] == ["eodhd"]


def test_eodhd_chain_positions_other_modes(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "auto")
    monkeypatch.delenv("VOE_STOOQ_APIKEY", raising=False)
    get_config.cache_clear()
    provider = get_default_provider(get_config())
    # Keyless Stooq is refused upstream, so "auto" only chains it with a key.
    assert [p.provider_name for p in provider.providers] == ["eodhd", "yahoo"]

    monkeypatch.setenv("VOE_STOOQ_APIKEY", "STOOQKEY123")
    get_config.cache_clear()
    provider = get_default_provider(get_config())
    assert [p.provider_name for p in provider.providers] == ["eodhd", "yahoo", "stooq"]
    monkeypatch.delenv("VOE_STOOQ_APIKEY", raising=False)

    monkeypatch.setenv("VOE_PRICE_PROVIDER", "yahoo")
    get_config.cache_clear()
    provider = get_default_provider(get_config())
    assert [p.provider_name for p in provider.providers] == ["eodhd", "yahoo"]


def test_eodhd_inert_without_apikey(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path, apikey=None)
    monkeypatch.delenv("VOE_STOOQ_APIKEY", raising=False)
    get_config.cache_clear()
    cfg = get_config()

    # The shipped default is "auto": with no keys it is Yahoo alone.
    provider = build_price_provider(cfg=cfg, with_prices=True)
    assert isinstance(provider, ChainedPriceProvider)
    assert [p.provider_name for p in provider.providers] == ["yahoo"]

    # "disabled" + with_prices chain assembly is unchanged.
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
    get_config.cache_clear()
    cfg = get_config()
    provider = build_price_provider(cfg=cfg, with_prices=True)
    assert isinstance(provider, ChainedPriceProvider)
    assert [p.provider_name for p in provider.providers] == ["stooq", "stooq_secondary"]

    monkeypatch.setenv("VOE_PRICE_PROVIDER", "stooq")
    get_config.cache_clear()
    cfg = get_config()
    provider = get_default_provider(cfg)
    assert [p.provider_name for p in provider.providers] == ["stooq", "stooq_secondary"]

    monkeypatch.setenv("VOE_PRICE_PROVIDER", "yahoo")
    get_config.cache_clear()
    bare = get_default_provider(get_config())
    assert bare.provider_name == "yahoo"

    # Lookup through the chain produces no EODHD attempts in diagnostics.
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "stooq")
    get_config.cache_clear()
    cfg = get_config()
    provider = get_default_provider(cfg)
    _install_fake_get(monkeypatch, _FakeResponse(_STOOQ_CSV))
    snapshot = provider.get_price_asof("AAPL", "2026-02-13")
    assert snapshot is not None
    assert snapshot.source == "stooq"
    assert snapshot.price == 101.25
    diag = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diag is not None
    attempt_providers = [row["provider"] for row in diag["provider_attempts"]]
    assert attempt_providers == ["stooq"]
    assert "eodhd" not in attempt_providers


def test_eodhd_snapshot_carries_raw_close(monkeypatch, tmp_path):
    """Split-basis re-basing: adjusted_close is split/dividend-adjusted to the
    download date, but deploy classification must compare the per-share
    anchor (as-of share basis) against what the market actually quoted at T.
    The snapshot now retains the raw close alongside the adjusted price."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    _install_fake_get(monkeypatch, _FakeResponse(_WINDOW_PAYLOAD))
    provider = EODHDProvider(cfg)

    snapshot = provider.get_price_asof("AAPL", "2026-02-13")
    assert snapshot is not None
    assert snapshot.price == 11.25  # adjusted_close (returns basis)
    assert snapshot.raw_price == 11.5  # raw close (classification basis)


def test_eodhd_raw_close_survives_cache_round_trip(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    calls: dict = {}
    _install_fake_get(monkeypatch, _FakeResponse(_WINDOW_PAYLOAD), calls=calls)
    provider = EODHDProvider(cfg)
    first = provider.get_price_asof("AAPL", "2026-02-13")
    assert first is not None and first.raw_price == 11.5
    n_after_first = calls.get("count", 0)

    # Fresh provider instance -> disk cache hit, no new network call.
    provider2 = EODHDProvider(cfg)
    second = provider2.get_price_asof("AAPL", "2026-02-13")
    assert second is not None
    assert calls.get("count", 0) == n_after_first
    assert second.price == 11.25
    assert second.raw_price == 11.5
