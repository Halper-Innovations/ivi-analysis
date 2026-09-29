from __future__ import annotations

import json

import requests

from app.market.price_provider import (
    ChainedPriceProvider,
    PriceSnapshot,
    StooqProvider,
    classify_price_error,
    resolve_price_from_historical_runs,
    write_prices_for_run,
)
from app.util.http import DomainBudgetExceeded


def _init_cfg(monkeypatch, tmp_path, *, overrides_csv: str | None = None):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    overrides_path = tmp_path / "price_symbol_overrides.csv"
    if overrides_csv is None:
        overrides_csv = "ticker,stooq_symbol\n"
    overrides_path.write_text(overrides_csv, encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_PRICE_SYMBOL_OVERRIDES_PATH", str(overrides_path))
    monkeypatch.setenv("VOE_PRICE_FALLBACK_DAYS", "7")
    monkeypatch.setenv("VOE_QUOTE_TTL_SECONDS", "86400")
    # These tests exercise network-enabled provider paths; every transport
    # is mocked per test (and the conftest blocks real sockets).
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def test_stooq_provider_cache_roundtrip(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = StooqProvider(cfg)
    csv_payload = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-12,100,102,99,101.25,1000000",
            "2026-02-11,98,100,97,99.50,900000",
        ]
    )

    monkeypatch.setattr(
        provider,
        "_download_daily_history",
        lambda _ticker: (csv_payload, "https://stooq.example/test.csv", 200),
    )

    first = provider.get_price_asof("AAPL", "2026-02-13")
    assert first is not None
    assert first.ticker == "AAPL"
    assert first.price == 101.25
    assert first.as_of_date == "2026-02-12"
    assert first.confidence == "MEDIUM"

    # Second call should use deterministic disk cache and avoid fetch.
    monkeypatch.setattr(
        provider,
        "_download_daily_history",
        lambda _ticker: (_ for _ in ()).throw(RuntimeError("network should not be called")),
    )
    second = provider.get_price_asof("AAPL", "2026-02-13")
    assert second == first

    diag = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diag is not None
    assert diag["result"]["reason_code"] == "CACHE_HIT"
    assert diag["cache"]["hit"] is True
    assert diag["provider_attempts"][0]["status"] == "CACHE_HIT"

    cache_path = cfg.cache_dir / "prices" / "AAPL.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "AAPL"
    assert any(
        row.get("requested_as_of_date") == "2026-02-13" and row.get("source") == "stooq"
        for row in (payload.get("entries") or [])
    )


def test_stooq_provider_weekend_fallback_and_attempt_log(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = StooqProvider(cfg)
    csv_payload = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-13,120,121,118,119.00,1000",
            "2026-02-12,118,120,117,118.00,900",
        ]
    )
    monkeypatch.setattr(
        provider,
        "_download_daily_history",
        lambda symbol: (csv_payload, f"https://stooq.example/{symbol}.csv", 200),
    )
    snapshot = provider.get_price_asof("AAPL", "2026-02-14")
    assert snapshot is not None
    assert snapshot.as_of_date == "2026-02-13"
    assert snapshot.price == 119.0
    assert snapshot.confidence == "MEDIUM"

    diag = provider.get_last_diagnostic("AAPL", "2026-02-14")
    assert diag is not None
    assert diag["resolved_symbol"] == "aapl.us"
    assert diag["attempted_symbols"] == ["aapl.us"]
    assert diag["asof_final_used"] == "2026-02-13"
    assert diag["market_day"]["requested_day_type"] == "NON_TRADING"
    assert diag["market_day"]["fallback_days_checked"] == 2
    assert diag["market_day"]["asof_used"] == "2026-02-13"
    assert diag["result"]["status"] == "OK"
    assert diag["result"]["reason_code"] == "PROVIDER_OK"
    assert diag["output_fields"]["price_asof_used"] == "2026-02-13"
    assert diag["output_fields"]["asof_final_used"] == "2026-02-13"
    assert diag["provider_attempts"] == [
        {
            "provider": "stooq",
            "status": "PROVIDER_OK",
            "http_status": 200,
            "url": "https://stooq.example/aapl.us.csv",
            "took_ms": diag["provider_attempts"][0]["took_ms"],
        }
    ]


def test_stooq_provider_applies_symbol_override(monkeypatch, tmp_path):
    cfg = _init_cfg(
        monkeypatch,
        tmp_path,
        overrides_csv="ticker,stooq_symbol\nBRK.B,brk-b.us\n",
    )
    provider = StooqProvider(cfg)
    seen_symbols: list[str] = []
    csv_payload = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-13,400,405,398,402.25,1000",
        ]
    )

    def _fake_download(symbol: str):
        seen_symbols.append(symbol)
        return csv_payload, "https://stooq.example/brk-b.us.csv", 200

    monkeypatch.setattr(provider, "_download_daily_history", _fake_download)
    snapshot = provider.get_price_asof("BRK.B", "2026-02-13")
    assert snapshot is not None
    assert snapshot.price == 402.25
    assert seen_symbols == ["brk-b.us"]

    diag = provider.get_last_diagnostic("BRK.B", "2026-02-13")
    assert diag is not None
    assert diag["resolved_symbol"] == "brk-b.us"
    assert diag["result"]["reason_code"] == "PROVIDER_OK"


def test_stooq_provider_walkback_deterministic(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = StooqProvider(cfg, fallback_days=3)
    csv_payload = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-13,301,305,299,304.50,1200",
            "2026-02-12,299,302,296,300.00,1300",
        ]
    )
    monkeypatch.setattr(
        provider,
        "_download_daily_history",
        lambda symbol: (csv_payload, f"https://stooq.example/{symbol}.csv", 200),
    )

    snapshot = provider.get_price_asof("MSFT", "2026-02-14")
    assert snapshot is not None
    assert snapshot.as_of_date == "2026-02-13"
    assert snapshot.price == 304.5
    diag = provider.get_last_diagnostic("MSFT", "2026-02-14")
    assert diag is not None
    assert diag["result"]["reason_code"] == "PROVIDER_OK"
    assert diag["market_day"]["asof_used"] == "2026-02-13"
    assert diag["output_fields"]["price_asof_used"] == "2026-02-13"
    assert diag["asof_final_used"] == "2026-02-13"


def test_stooq_provider_refreshes_stale_cache(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = StooqProvider(cfg)
    csv_payload_initial = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-13,100,102,99,101.25,1000000",
        ]
    )
    csv_payload_refreshed = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-13,100,102,99,111.25,1000000",
        ]
    )

    monkeypatch.setattr(
        provider,
        "_download_daily_history",
        lambda _ticker: (csv_payload_initial, "https://stooq.example/test.csv", 200),
    )
    first = provider.get_price_asof("AAPL", "2026-02-13")
    assert first is not None
    assert first.price == 101.25

    cache_path = cfg.cache_dir / "prices" / "AAPL.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    for entry in payload.get("entries") or []:
        snapshot = entry.get("snapshot") if isinstance(entry, dict) else None
        if isinstance(snapshot, dict):
            snapshot["retrieved_at"] = "2000-01-01T00:00:00+00:00"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        provider,
        "_download_daily_history",
        lambda _ticker: (csv_payload_refreshed, "https://stooq.example/test-refreshed.csv", 200),
    )
    second = provider.get_price_asof("AAPL", "2026-02-13")
    assert second is not None
    assert second.price == 111.25

    diag = provider.get_last_diagnostic("AAPL", "2026-02-13")
    assert diag is not None
    assert diag["result"]["reason_code"] == "PROVIDER_OK"
    assert diag["cache"]["hit"] is False


def test_symbol_override_uses_date_window(monkeypatch, tmp_path):
    cfg = _init_cfg(
        monkeypatch,
        tmp_path,
        overrides_csv=(
            "ticker,stooq_symbol,valid_from,valid_to\n"
            "ABC,abc_old.us,2020-01-01,2025-12-31\n"
            "ABC,abc_new.us,2026-01-01,\n"
        ),
    )
    provider = StooqProvider(cfg)
    seen_symbols: list[str] = []
    csv_payload = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-13,11,12,10,11.25,100",
        ]
    )

    def _fake_download(symbol: str):
        seen_symbols.append(symbol)
        return csv_payload, f"https://stooq.example/{symbol}.csv", 200

    monkeypatch.setattr(provider, "_download_daily_history", _fake_download)
    snapshot = provider.get_price_asof("ABC", "2026-02-14")
    assert snapshot is not None
    assert seen_symbols == ["abc_new.us"]
    diag = provider.get_last_diagnostic("ABC", "2026-02-14")
    assert diag is not None
    assert diag["resolved_symbol"] == "abc_new.us"
    assert diag["attempted_symbols"] == ["abc_new.us", "abc.us"]


def test_classify_price_error_mapping():
    code, retryable, suggestion, detail = classify_price_error(requests.exceptions.ConnectionError("Name resolution failed"))
    assert code == "DNS_FAILURE"
    assert retryable is True
    assert "connectivity" in suggestion
    assert "Name resolution failed" in detail

    response = requests.Response()
    response.status_code = 429
    http_exc = requests.HTTPError("too many requests")
    http_exc.response = response
    code, retryable, suggestion, _ = classify_price_error(http_exc)
    assert code == "RATE_LIMIT"
    assert retryable is True
    assert "budget" in suggestion

    code, retryable, suggestion, _ = classify_price_error(DomainBudgetExceeded("budget exhausted"))
    assert code == "BUDGET_EXHAUSTED"
    assert retryable is False
    assert "budget" in suggestion


def test_provider_redundancy_primary_retryable_secondary_success():
    class _PrimaryFailDNS:
        provider_name = "primary"

        def __init__(self):
            self._diag = None

        def get_price_asof(self, ticker: str, as_of_date: str):
            self._diag = {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "provider_attempts": [{"provider": "primary", "status": "DNS_FAILURE"}],
                "result": {
                    "status": "UNKNOWN",
                    "reason_code": "DNS_FAILURE",
                    "reason_detail": "dns",
                    "retryable": True,
                    "error_detail": "dns",
                },
                "total_attempts": 1,
                "retry_count": 0,
            }
            return None

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return self._diag

    class _SecondaryOK:
        provider_name = "secondary"

        def __init__(self):
            self._diag = None

        def get_price_asof(self, ticker: str, as_of_date: str):
            self._diag = {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "provider_attempts": [{"provider": "secondary", "status": "PROVIDER_OK"}],
                "result": {
                    "status": "OK",
                    "reason_code": "PROVIDER_OK",
                    "reason_detail": "ok",
                    "retryable": False,
                    "error_detail": "",
                },
                "output_fields": {
                    "current_price": 33.0,
                    "price_asof_used": as_of_date,
                    "asof_final_used": as_of_date,
                    "price_source": "secondary",
                    "confidence": "HIGH",
                },
                "total_attempts": 1,
                "retry_count": 0,
            }
            return PriceSnapshot(
                ticker=ticker,
                as_of_date=as_of_date,
                price=33.0,
                source="secondary",
                retrieved_at="2026-02-13T00:00:00+00:00",
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return self._diag

    chained = ChainedPriceProvider([_PrimaryFailDNS(), _SecondaryOK()])
    snapshot = chained.get_price_asof("MSFT", "2026-02-13")
    assert snapshot is not None
    assert snapshot.price == 33.0
    diag = chained.get_last_diagnostic("MSFT", "2026-02-13")
    assert diag is not None
    statuses = [row.get("status") for row in diag.get("provider_attempts") or []]
    assert statuses == ["DNS_FAILURE", "PROVIDER_OK"]


def test_provider_redundancy_secondary_tried_after_terminal_primary():
    # Providers have different symbology, so a terminal (non-retryable)
    # miss at the primary no longer short-circuits the chain — every
    # provider is tried until one succeeds.
    calls = {"secondary": 0}

    class _PrimaryTerminal:
        provider_name = "primary"

        def __init__(self):
            self._diag = None

        def get_price_asof(self, ticker: str, as_of_date: str):
            self._diag = {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "provider_attempts": [{"provider": "primary", "status": "SYMBOL_NOT_FOUND"}],
                "result": {
                    "status": "UNKNOWN",
                    "reason_code": "SYMBOL_NOT_FOUND",
                    "reason_detail": "terminal",
                    "retryable": False,
                    "error_detail": "terminal",
                },
                "total_attempts": 1,
                "retry_count": 0,
            }
            return None

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return self._diag

    class _SecondaryMiss:
        provider_name = "secondary"

        def get_price_asof(self, ticker: str, as_of_date: str):
            calls["secondary"] += 1
            return None

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "provider_attempts": [{"provider": "secondary", "status": "PROVIDER_NO_DATA"}],
                "result": {"status": "UNKNOWN", "reason_code": "PROVIDER_NO_DATA", "retryable": False},
                "total_attempts": 1,
                "retry_count": 0,
            }

    chained = ChainedPriceProvider([_PrimaryTerminal(), _SecondaryMiss()])
    snapshot = chained.get_price_asof("MSFT", "2026-02-13")
    assert snapshot is None
    assert calls["secondary"] == 1
    diag = chained.get_last_diagnostic("MSFT", "2026-02-13")
    assert diag is not None
    statuses = [row.get("status") for row in diag.get("provider_attempts") or []]
    assert statuses == ["SYMBOL_NOT_FOUND", "PROVIDER_NO_DATA"]


def test_resolve_price_from_historical_runs_from_sector_artifacts(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    sectors_dir = cfg.sectors_dir
    old_run = sectors_dir / "aaa_old_run"
    new_run = sectors_dir / "zzz_new_run"
    old_run.mkdir(parents=True, exist_ok=True)
    new_run.mkdir(parents=True, exist_ok=True)

    (old_run / "price_coverage.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "ticker": "MSFT",
                        "result": {"status": "OK", "reason_code": "CACHE_HIT"},
                        "output_fields": {"current_price": 200.0, "price_asof_used": "2026-02-12", "confidence": "MEDIUM"},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (new_run / "valuation_MSFT.json").write_text(
        json.dumps(
            {
                "ticker": "MSFT",
                "as_of_date": "2026-02-14",
                "input_snapshot": {
                    "current_price": 305.5,
                    "price_asof_used": "2026-02-13",
                    "current_price_source": "provider_live_fetch",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (new_run / "price_coverage.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "ticker": "MSFT",
                        "result": {"status": "OK", "reason_code": "CACHE_HIT"},
                        "output_fields": {"current_price": 999.0, "price_asof_used": "2026-02-13", "confidence": "HIGH"},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    snapshot = resolve_price_from_historical_runs(
        ticker="MSFT",
        as_of_date="2026-02-14",
        sectors_dir=sectors_dir,
        max_runs=50,
    )
    assert snapshot is not None
    assert snapshot.ticker == "MSFT"
    assert snapshot.as_of_date == "2026-02-13"
    assert snapshot.price == 305.5
    assert snapshot.source == "historical_run_artifacts"
    assert snapshot.confidence == "MEDIUM"


def test_write_prices_for_run_respects_disabled_network(monkeypatch, tmp_path):
    _cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()

    def _unexpected_download(*_args, **_kwargs):
        raise AssertionError("network download should be skipped in disabled-network mode")

    monkeypatch.setattr("app.market.price_provider.StooqProvider._download_daily_history", _unexpected_download)
    monkeypatch.setattr("app.market.price_provider.StooqSecondaryProvider._download_daily_history", _unexpected_download)
    payload = write_prices_for_run(
        tickers=["AAA"],
        as_of_date="2026-02-14",
        run_id="offline_price_run",
        local_only=False,
        cfg=cfg,
    )

    assert payload["local_only"] is False
    assert payload["provider_effective"] == "stooq+fallback"
    assert payload["reason_counts"]["OFFLINE_NO_CACHE"] == 1


def test_stooq_network_gate_is_net_provider_only(
    monkeypatch,
    tmp_path,
) -> None:
    # VOE_NET_PROVIDER is the only network switch: a disabled LLM provider
    # no longer implies offline, and no per-provider override exists.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    cfg = _init_cfg(monkeypatch, tmp_path)
    csv_payload = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-07-17,100,102,99,101.25,1000000",
        ]
    )
    downloaded_symbols: list[str] = []
    allowed = StooqProvider(cfg, max_retries=0)

    def download(symbol: str):
        downloaded_symbols.append(symbol)
        return csv_payload, "https://stooq.com/q/d/l/?s=allow.us&i=d", 200

    monkeypatch.setattr(allowed, "_download_daily_history", download)
    allowed_snapshot = allowed.get_price_asof("ALLOW", "2026-07-17")

    assert allowed_snapshot is not None
    assert allowed_snapshot.price == 101.25
    assert allowed_snapshot.source == "stooq"
    assert downloaded_symbols == ["allow.us"]

    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    hard_guarded = StooqProvider(cfg, max_retries=0)
    monkeypatch.setattr(
        hard_guarded,
        "_download_daily_history",
        lambda _symbol: (_ for _ in ()).throw(
            AssertionError("VOE_NET_PROVIDER=disabled must remain authoritative")
        ),
    )
    assert hard_guarded.get_price_asof("BLOCK", "2026-07-17") is None
    hard_diag = hard_guarded.get_last_diagnostic("BLOCK", "2026-07-17")
    assert hard_diag is not None
    assert hard_diag["result"]["reason_code"] == "OFFLINE_NO_CACHE"


def test_history_memo_disabled_by_default_downloads_per_anchor(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = StooqProvider(cfg)
    csv_payload = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-12,100,102,99,101.25,1000000",
            "2026-02-11,98,100,97,99.50,900000",
        ]
    )
    calls = {"n": 0}

    def _download(_symbol):
        calls["n"] += 1
        return csv_payload, "https://stooq.example/test.csv", 200

    monkeypatch.setattr(provider, "_download_daily_history", _download)

    first = provider.get_price_asof("AAPL", "2026-02-12")
    second = provider.get_price_asof("AAPL", "2026-02-11")
    assert first is not None and first.price == 101.25
    assert second is not None and second.price == 99.50
    assert calls["n"] == 2


def test_history_memo_enabled_downloads_once_per_symbol(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = StooqProvider(cfg)
    provider.enable_history_memo()
    csv_payload = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-02-12,100,102,99,101.25,1000000",
            "2026-02-11,98,100,97,99.50,900000",
        ]
    )
    calls = {"n": 0}

    def _download(_symbol):
        calls["n"] += 1
        return csv_payload, "https://stooq.example/test.csv", 200

    monkeypatch.setattr(provider, "_download_daily_history", _download)

    first = provider.get_price_asof("AAPL", "2026-02-12")
    second = provider.get_price_asof("AAPL", "2026-02-11")
    assert first is not None and first.price == 101.25
    assert first.volume == 1000000.0
    assert second is not None and second.price == 99.50
    assert second.volume == 900000.0
    assert calls["n"] == 1


def test_history_memo_single_slot_replaced_across_symbols(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = StooqProvider(cfg)
    provider.enable_history_memo()
    payload_by_symbol = {
        "aapl.us": "Date,Open,High,Low,Close,Volume\n2026-02-12,100,102,99,101.25,1000000",
        "msft.us": "Date,Open,High,Low,Close,Volume\n2026-02-12,200,202,199,201.50,2000000",
    }
    calls = []

    def _download(symbol):
        calls.append(symbol)
        return payload_by_symbol[symbol], f"https://stooq.example/{symbol}.csv", 200

    monkeypatch.setattr(provider, "_download_daily_history", _download)

    aapl = provider.get_price_asof("AAPL", "2026-02-12")
    msft = provider.get_price_asof("MSFT", "2026-02-12")
    assert aapl is not None and aapl.price == 101.25
    assert msft is not None and msft.price == 201.50
    assert msft.volume == 2000000.0
    assert calls == ["aapl.us", "msft.us"]


def test_chained_provider_delegates_history_memo():
    inner = StooqProvider.__new__(StooqProvider)
    inner._history_memo_enabled = False
    chain = ChainedPriceProvider([inner])
    chain.enable_history_memo()
    assert inner._history_memo_enabled is True


def test_history_memo_negative_memo_hard_failure(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    provider = StooqProvider(cfg)
    provider.enable_history_memo()
    calls = {"n": 0}

    def _download(_symbol):
        calls["n"] += 1
        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError("429 rate limited", response=response)

    monkeypatch.setattr(provider, "_download_daily_history", _download)

    first = provider.get_price_asof("AAPL", "2026-02-12")
    second = provider.get_price_asof("AAPL", "2026-02-11")
    assert first is None
    assert second is None
    assert calls["n"] == 1
