from __future__ import annotations

import json

import pytest

from app.calibration import picks_benchmarks
from app.config import get_config
from app.market.price_provider import (
    ChainedPriceProvider,
    EODHDProvider,
    PriceSnapshot,
    YahooFinanceProvider,
)


@pytest.mark.parametrize("disabled", ["disabled", "off", "none"])
def test_disabled_refuses_before_factory(isolated_data_root, monkeypatch, disabled):
    cfg = get_config().model_copy(
        update={"price_provider": disabled, "eodhd_apikey": "test-only-key"}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("The disabled provider must not construct a fallback.")

    monkeypatch.setattr(picks_benchmarks, "get_default_provider", forbidden)
    with pytest.raises(ValueError, match="configured price provider is disabled"):
        picks_benchmarks.refresh_benchmarks(cfg, {"2026-09-09"})


def test_eodhd_raw_quotes_preserve_config_and_database(isolated_data_root, monkeypatch):
    cfg = get_config().model_copy(
        update={
            "price_provider": "yahoo",
            "eodhd_apikey": "test-only-key",
            "net_provider": "enabled",
        }
    )
    cfg.db_path.write_bytes(b"unchanged owner database")
    original_factory = picks_benchmarks.get_default_provider
    constructed = []

    def factory(provider_cfg, **kwargs):
        assert provider_cfg.eodhd_apikey == "test-only-key"
        assert provider_cfg.price_provider == "yahoo"
        assert provider_cfg.db_path != cfg.db_path
        assert kwargs == {"max_retries": 0}
        constructed.append(provider_cfg.cache_dir)
        return original_factory(provider_cfg, **kwargs)

    def download(self, symbol):
        requested = self._window_target.isoformat()
        raw = 100.0 if requested == "2026-06-12" else 110.0
        return json.dumps([{"date": requested, "close": raw, "adjusted_close": raw - 5}]), "", 200

    monkeypatch.setattr(picks_benchmarks, "get_default_provider", factory)
    monkeypatch.setattr(EODHDProvider, "_download_daily_history", download)
    # A disabled LLM provider does not imply offline; only VOE_NET_PROVIDER does.
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    result = picks_benchmarks.refresh_benchmarks(cfg, {"2026-06-12", "2026-09-09"})

    assert result.request_count == 4
    assert result.request_count_unit == "provider anchor calls; HTTP requests not measured"
    assert result.failures == []
    assert result.prices[("IWM", "2026-06-12")].price == 100.0
    assert result.prices[("SPY", "2026-09-09")].price == 110.0
    assert result.prices[("SPY", "2026-09-09")].provider == "eodhd"
    assert result.prices[("SPY", "2026-09-09")].price_basis == "UNADJUSTED"
    assert cfg.db_path.read_bytes() == b"unchanged owner database"
    assert not (cfg.cache_dir / "prices").exists()
    assert all(not directory.exists() for directory in constructed)


def test_failed_primary_counts_and_yahoo_uses_unadjusted(isolated_data_root, monkeypatch):
    cfg = get_config().model_copy(update={"price_provider": "yahoo", "net_provider": "enabled"})

    class FailedProvider:
        provider_name = "primary"

        def get_price_asof(self, ticker, requested):
            raise RuntimeError("never report this test-only-secret")

    def factory(provider_cfg, **kwargs):
        return ChainedPriceProvider([FailedProvider(), YahooFinanceProvider(provider_cfg)])

    def get_price(self, ticker, requested):
        assert self.auto_adjust is False
        return PriceSnapshot(ticker, "2026-09-08", 120, source="yahoo")

    monkeypatch.setattr(picks_benchmarks, "get_default_provider", factory)
    monkeypatch.setattr(YahooFinanceProvider, "get_price_asof", get_price)
    result = picks_benchmarks.refresh_benchmarks(cfg, {"2026-09-09"})
    assert result.request_count == 4
    assert result.failures == []
    assert result.prices[("IWM", "2026-09-09")].as_of_date == "2026-09-08"
    assert result.prices[("IWM", "2026-09-09")].price == 120.0


@pytest.mark.parametrize(
    ("used_date", "raw", "expected"),
    [
        ("2026-09-09", None, "RAW_BASIS_UNAVAILABLE"),
        ("2026-09-10", 100, "INVALID_ANCHOR"),
        ("2026-08-01", 100, "INVALID_ANCHOR"),
        ("2026-09-09", float("nan"), "RAW_BASIS_UNAVAILABLE"),
    ],
)
def test_unknown_basis_and_invalid_anchors_rejected(
    isolated_data_root, monkeypatch, used_date, raw, expected
):
    cfg = get_config().model_copy(update={"price_provider": "auto", "net_provider": "enabled"})

    class FakeProvider:
        provider_name = "unknown"

        def get_price_asof(self, ticker, requested):
            return PriceSnapshot(ticker, used_date, 100, raw_price=raw)

    monkeypatch.setattr(picks_benchmarks, "get_default_provider", lambda *a, **k: FakeProvider())
    result = picks_benchmarks.refresh_benchmarks(cfg, {"2026-09-09"})
    assert result.prices == {}
    assert result.request_count == 2
    assert result.failures == [
        f"IWM on 2026-09-09: unknown: {expected}",
        f"SPY on 2026-09-09: unknown: {expected}",
    ]


def test_stooq_unknown_basis_refuses_without_network(isolated_data_root, monkeypatch):
    cfg = get_config().model_copy(
        update={"price_provider": "stooq", "eodhd_apikey": None, "net_provider": "enabled"}
    )
    result = picks_benchmarks.refresh_benchmarks(cfg, {"2026-09-09"})
    assert result.request_count == 0
    assert result.prices == {}
    assert result.failures == [
        "IWM on 2026-09-09: stooq: RAW_BASIS_UNAVAILABLE; stooq_secondary: RAW_BASIS_UNAVAILABLE",
        "SPY on 2026-09-09: stooq: RAW_BASIS_UNAVAILABLE; stooq_secondary: RAW_BASIS_UNAVAILABLE",
    ]


def test_empty_dates_do_not_construct_provider(isolated_data_root, monkeypatch):
    cfg = get_config().model_copy(update={"price_provider": "yahoo", "net_provider": "enabled"})

    def forbidden(*args, **kwargs):
        pytest.fail("No dates require no provider.")

    monkeypatch.setattr(picks_benchmarks, "get_default_provider", forbidden)
    result = picks_benchmarks.refresh_benchmarks(cfg, set())
    assert result.request_count == 0
    assert result.prices == {}


def test_disabled_network_refuses(isolated_data_root):
    cfg = get_config().model_copy(update={"price_provider": "yahoo", "net_provider": "disabled"})
    with pytest.raises(ValueError, match="network access is disabled"):
        picks_benchmarks.refresh_benchmarks(cfg, {"2026-09-09"})
