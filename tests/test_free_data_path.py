"""The keyless path: SEC facts and free quotes with no LLM and no paid key.

Contract under test:
- VOE_NET_PROVIDER is the only network switch; VOE_LLM_PROVIDER=disabled
  does not make SEC or price fetches offline.
- Valuation quotes follow price_provider (+ the network switch), not
  safe_mode, and state their price basis instead of assuming it.
- ``ivi value`` composes fetch -> facts -> writer -> quote and reports
  degraded states explicitly.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from app.config import get_config
from app.db import get_db, init_db
from app.market import company_facts_provider as cfp
from app.market.price_provider import PriceSnapshot, YahooFinanceProvider
from app.valuation import price_provider as vpp
from app.valuation import quick_value


def _cfg(monkeypatch, **env: str):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


class _Response:
    status_code = 200

    def json(self):
        return {"cik": 21344, "entityName": "Test Co", "facts": {"us-gaap": {}}}


class _Session:
    def __init__(self, calls: list[str]):
        self.calls = calls
        self.headers: dict[str, str] = {}

    def get(self, url, timeout=None, **kwargs):
        self.calls.append(url)
        return _Response()


def _install_sec_session(monkeypatch) -> list[str]:
    calls: list[str] = []
    original = cfp.HttpClient.__init__

    def init(self, cfg=None):
        original(self, cfg)
        self.session = _Session(calls)

    monkeypatch.setattr(cfp.HttpClient, "__init__", init)
    return calls


# --- the network switch ------------------------------------------------------


def test_llm_disabled_does_not_block_companyfacts_fetch(isolated_data_root, monkeypatch):
    cfg = _cfg(
        monkeypatch,
        VOE_LLM_PROVIDER="disabled",
        VOE_NET_PROVIDER="enabled",
        VOE_SEC_USER_AGENT="IVI tests qa@ivi.test",
    )
    calls = _install_sec_session(monkeypatch)

    result = cfp.fetch_company_facts("21344", cfg=cfg)

    assert result["status"] == "OK"
    assert result["reason_code"] == "FETCH_OK"
    assert calls == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000021344.json"]


def test_fresh_companyfacts_cache_is_served_without_a_request(isolated_data_root, monkeypatch):
    cfg = _cfg(
        monkeypatch,
        VOE_NET_PROVIDER="enabled",
        VOE_SEC_USER_AGENT="IVI tests qa@ivi.test",
    )
    calls = _install_sec_session(monkeypatch)
    first = cfp.fetch_company_facts("21344", cfg=cfg)
    second = cfp.fetch_company_facts("21344", cfg=cfg)

    assert first["reason_code"] == "FETCH_OK"
    assert second["reason_code"] == "CACHE_HIT"
    assert second["reason_detail"] == "Resolved from a companyfacts cache fetched within the last 24h."
    assert second["companyfacts"] == first["companyfacts"]
    assert len(calls) == 1

    # A cache older than the TTL is refreshed.
    path = cfp.companyfacts_cache_path("21344", cfg=cfg)
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    stale = datetime.now(timezone.utc) - timedelta(seconds=cfp.COMPANYFACTS_CACHE_TTL_SECONDS + 60)
    wrapper["retrieved_at"] = stale.isoformat()
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    third = cfp.fetch_company_facts("21344", cfg=cfg)
    assert third["reason_code"] == "FETCH_OK"
    assert len(calls) == 2


def test_net_disabled_blocks_companyfacts_fetch(isolated_data_root, monkeypatch):
    cfg = _cfg(monkeypatch, VOE_LLM_PROVIDER="openai", VOE_NET_PROVIDER="disabled")
    calls = _install_sec_session(monkeypatch)

    result = cfp.fetch_company_facts("21344", cfg=cfg)

    assert result["status"] == "MISSING"
    assert result["reason_code"] == "OFFLINE_NO_CACHE"
    assert calls == []


def test_yahoo_honors_the_network_switch(isolated_data_root, monkeypatch):
    cfg = _cfg(monkeypatch, VOE_NET_PROVIDER="disabled")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("yfinance must not be reached with the network switch off")

    monkeypatch.setattr("yfinance.Ticker", forbidden)
    provider = YahooFinanceProvider(cfg)

    assert provider.get_price_asof("KO", "2026-09-25") is None
    assert provider.get_last_diagnostic("KO", "2026-09-25") == {
        "status": "OFFLINE_NO_CACHE",
        "provider": "yahoo",
        "ticker": "KO",
    }


# --- valuation quote selection -------------------------------------------------


def test_valuation_provider_selection_ignores_safe_mode(isolated_data_root, monkeypatch):
    cfg = _cfg(monkeypatch, VOE_NET_PROVIDER="enabled", VOE_SAFE_MODE="true")
    monkeypatch.delenv("VOE_EODHD_APIKEY", raising=False)
    monkeypatch.delenv("VOE_STOOQ_APIKEY", raising=False)

    auto = vpp.get_default_provider(cfg.model_copy(update={"price_provider": "auto"}))
    assert isinstance(auto, vpp.ChainedQuoteProvider)
    assert [p.provider_name for p in auto.providers] == ["yahoo"]
    yahoo_leaf = auto.providers[0]
    assert isinstance(yahoo_leaf, vpp.MarketQuoteProvider)
    assert yahoo_leaf.source.auto_adjust is False

    keyed = vpp.get_default_provider(
        cfg.model_copy(update={"price_provider": "auto", "eodhd_apikey": "k", "stooq_apikey": "s"})
    )
    assert [p.provider_name for p in keyed.providers] == ["eodhd", "yahoo", "stooq"]

    pinned = vpp.get_default_provider(cfg.model_copy(update={"price_provider": "stooq"}))
    assert pinned.provider_name == "stooq"

    for update, reason in (
        ({"price_provider": "disabled"}, "price_provider_disabled"),
        ({"price_provider": "eodhd", "eodhd_apikey": None}, "eodhd_selected_without_api_key"),
        ({"price_provider": "auto", "net_provider": "disabled"}, "network_disabled"),
    ):
        monkeypatch.setenv("VOE_NET_PROVIDER", update.get("net_provider", "enabled"))
        provider = vpp.get_default_provider(cfg.model_copy(update=update))
        assert isinstance(provider, vpp.DisabledPriceProvider)
        assert provider.get_quote("KO", "2026-09-25").provenance == {
            "mode": "offline_disabled",
            "reason": reason,
        }


class _FakeSource:
    def __init__(self, name: str, snapshot: PriceSnapshot | None, diagnostic=None):
        self.provider_name = name
        self.snapshot = snapshot
        self.diagnostic = diagnostic or {}
        self.calls = 0
        self.auto_adjust = False

    def get_price_asof(self, ticker, as_of_date):
        self.calls += 1
        return self.snapshot

    def get_last_diagnostic(self, ticker, as_of_date):
        return self.diagnostic


def test_recent_yahoo_close_is_an_unadjusted_quote_and_is_cached(isolated_data_root, monkeypatch):
    cfg = _cfg(monkeypatch)
    today = date.today().isoformat()
    source = _FakeSource("yahoo", PriceSnapshot("KO", today, 87.18, source="yahoo", confidence="HIGH"))
    provider = vpp.MarketQuoteProvider(cfg, source)

    quote = provider.get_quote("KO", today)
    again = provider.get_quote("KO", today)

    assert quote.status == "OK"
    assert quote.price == 87.18
    assert quote.price_basis == "UNADJUSTED"
    assert quote.provenance["history_auto_adjust"] is False
    assert quote.provenance["quote_date"] == today
    assert again.quote_snapshot_id == quote.quote_snapshot_id
    assert source.calls == 1
    with get_db() as conn:
        row = conn.execute(
            "SELECT provider, status, price_basis FROM price_quotes WHERE ticker = 'KO'"
        ).fetchone()
    assert tuple(row) == ("yahoo", "OK", "UNADJUSTED")


def test_historical_free_close_is_not_labeled_unadjusted(isolated_data_root, monkeypatch):
    # Yahoo and Stooq restate history for later splits; an old close cannot be
    # shown to be the raw quote, so it is refused rather than relabeled.
    cfg = _cfg(monkeypatch)
    old = (date.today() - timedelta(days=400)).isoformat()
    provider = vpp.MarketQuoteProvider(
        cfg, _FakeSource("stooq", PriceSnapshot("KO", old, 60.0, source="stooq"))
    )

    quote = provider.get_quote("KO", old)

    assert quote.status == "UNKNOWN"
    assert quote.price is None
    assert quote.provenance["reason"] == "RAW_CLOSE_UNPROVEN_FOR_HISTORICAL_DATE"
    assert quote.provenance["provider_price"] == 60.0


def test_eodhd_quote_uses_the_raw_close(isolated_data_root, monkeypatch):
    cfg = _cfg(monkeypatch)
    old = "2020-06-01"
    with_raw = vpp.MarketQuoteProvider(
        cfg,
        _FakeSource("eodhd", PriceSnapshot("KO", old, 40.0, source="eodhd", raw_price=45.5)),
    )
    without_raw = vpp.MarketQuoteProvider(
        cfg, _FakeSource("eodhd", PriceSnapshot("PEP", old, 40.0, source="eodhd"))
    )

    assert with_raw.get_quote("KO", old).price == 45.5
    missing = without_raw.get_quote("PEP", old)
    assert missing.price is None
    assert missing.provenance["reason"] == "EODHD_RAW_CLOSE_UNAVAILABLE"


def test_chain_falls_through_and_reports_every_attempt(isolated_data_root, monkeypatch):
    cfg = _cfg(monkeypatch)
    today = date.today().isoformat()
    dead = vpp.MarketQuoteProvider(
        cfg,
        _FakeSource("eodhd", None, {"result": {"reason_code": "SYMBOL_NOT_FOUND"}}),
    )
    live = vpp.MarketQuoteProvider(
        cfg, _FakeSource("yahoo", PriceSnapshot("KO", today, 87.18, source="yahoo"))
    )
    also_dead = vpp.MarketQuoteProvider(cfg, _FakeSource("yahoo", None, {"status": "ERROR"}))

    assert vpp.ChainedQuoteProvider(cfg, [dead, live]).get_quote("KO", today).provider == "yahoo"
    failed = vpp.ChainedQuoteProvider(cfg, [dead, also_dead]).get_quote("PEP", today)
    assert failed.status == "UNKNOWN"
    assert failed.provenance == {
        "mode": "live",
        "reason": "NO_SOURCE_RETURNED_A_QUOTE",
        "attempts": [
            {"provider": "eodhd", "status": "UNKNOWN", "reason": "SYMBOL_NOT_FOUND"},
            {"provider": "yahoo", "status": "UNKNOWN", "reason": "ERROR"},
        ],
    }


# --- ivi value --------------------------------------------------------------------


def _insert_row(conn, ticker: str, as_of: str, method: str, outputs: dict) -> None:
    conn.execute(
        """
        INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json,
                               warnings_json, created_at)
        VALUES (?, ?, ?, '{}', ?, '[]', ?)
        """,
        (ticker, as_of, method, json.dumps(outputs), datetime.now(timezone.utc).isoformat()),
    )


def test_value_command_composes_the_pipeline(isolated_data_root, monkeypatch):
    _cfg(monkeypatch)
    calls: list[tuple] = []
    monkeypatch.setattr(
        "app.valuation.facts.resolve_cik_for_ticker", lambda ticker, cfg=None: "0000021344"
    )
    monkeypatch.setattr(
        "app.market.company_facts_provider.fetch_company_facts",
        lambda cik, cfg=None: {"status": "OK", "reason_code": "FETCH_OK", "size_bytes": 5_000_000},
    )
    def fake_facts(ticker, years_back=10):
        calls.append(("facts", ticker, years_back))
        with get_db() as conn:
            conn.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
                "line_item, value, units, source_url, fetched_at, filed_date, form, accession) "
                "VALUES (?, 2025, 'FY', '2025-12-31', 'revenue', 1.0, 'USD_millions', '', "
                "'2026-09-25T00:00:00+00:00', '2026-02-20', '10-K', 'x')",
                (ticker,),
            )

    monkeypatch.setattr("app.ingest.facts_writer.ensure_all_facts", fake_facts)

    def fake_writer(ticker, as_of_date, provider, **kwargs):
        calls.append(("writer", ticker, as_of_date, kwargs["issuer_cik"], kwargs["force_refresh"]))
        with get_db() as conn:
            _insert_row(conn, ticker, as_of_date, "dcf", {"status": "OK", "low": 40.0, "base": 50.0, "high": 60.0})
            _insert_row(conn, ticker, as_of_date, "graham", {"status": "OK", "value_per_share": 20.0})
            _insert_row(conn, ticker, as_of_date, "epv", {"status": "METHOD_INSUFFICIENT_DATA", "flags": ["NET_DEBT_UNKNOWN"]})
            _insert_row(
                conn,
                ticker,
                as_of_date,
                "scorecard",
                {
                    "signal": "PASS",
                    "pricing_zone": "FAIR",
                    "pricing_zone_detail": {
                        "current_price": 40.0,
                        "current_price_source": "yahoo",
                        "current_price_as_of_date": as_of_date,
                        "current_price_basis": "UNADJUSTED",
                    },
                },
            )
        return []

    monkeypatch.setattr("app.valuation.valuation_writer.ensure_valuation", fake_writer)

    result = CliRunner().invoke(
        __import__("app.cli", fromlist=["app"]).app, ["value", "ko", "--as-of", "2026-09-25"]
    )

    assert result.exit_code == 0, result.output
    assert calls == [("facts", "KO", 10), ("writer", "KO", "2026-09-25", "0000021344", True)]
    lines = result.output.splitlines()
    assert lines[0] == "KO  as of 2026-09-25  (CIK 0000021344)"
    assert lines[2] == "Price: 40.00 (yahoo, close of 2026-09-25, basis UNADJUSTED)"
    assert "DCF (base)                       50.00              +20%  OK" in lines
    assert "Graham number                    20.00             -100%  OK" in lines
    assert any(
        line.startswith("Earnings power (EPV)") and "METHOD_INSUFFICIENT_DATA (NET_DEBT_UNKNOWN)" in line
        for line in lines
    )
    assert "not bound to an audited run artifact" in result.output


def test_value_command_reports_a_failed_facts_fetch(isolated_data_root, monkeypatch):
    _cfg(monkeypatch)
    monkeypatch.setattr(
        "app.valuation.facts.resolve_cik_for_ticker", lambda ticker, cfg=None: "0000021344"
    )
    monkeypatch.setattr(
        "app.market.company_facts_provider.fetch_company_facts",
        lambda cik, cfg=None: {
            "status": "MISSING",
            "reason_code": "SEC_USER_AGENT_INVALID",
            "reason_detail": "Set VOE_SEC_USER_AGENT.",
        },
    )

    summary = quick_value.run_value("KO", as_of_date="2026-09-25")

    assert summary["status"] == "FAILED"
    assert summary["reason"] == "COMPANYFACTS_SEC_USER_AGENT_INVALID"
    assert quick_value.render_value_summary(summary).splitlines()[-1] == (
        "FAILED: COMPANYFACTS_SEC_USER_AGENT_INVALID - Set VOE_SEC_USER_AGENT."
    )


@pytest.mark.parametrize("ticker", ["NOPE"])
def test_value_command_reports_unknown_ticker(isolated_data_root, monkeypatch, ticker):
    # Online: offline with no cached ticker list is reported as OFFLINE_NOT_CACHED.
    _cfg(monkeypatch, VOE_NET_PROVIDER="enabled")
    monkeypatch.setattr("app.valuation.facts.resolve_cik_for_ticker", lambda t, cfg=None: None)

    summary = quick_value.run_value(ticker, as_of_date="2026-09-25")

    assert summary["status"] == "FAILED"
    assert summary["reason"] == "CIK_NOT_FOUND"


def test_value_command_refuses_a_registrant_with_no_annual_report(isolated_data_root, monkeypatch):
    """The 2026-09-29 basket: SEC's ticker map moved XOM to CIK 0002115436,
    ExxonMobil Holdings Corp, a holding company whose companyfacts starts at its
    first 10-Q (0.1 MB). With no annual report there is nothing to value; the
    command says so instead of printing seven unrelated method gaps, and never
    borrows the predecessor registrant's history (SEC data does not link the
    two CIKs)."""
    _cfg(monkeypatch)
    monkeypatch.setattr(
        "app.valuation.facts.resolve_cik_for_ticker", lambda ticker, cfg=None: "0002115436"
    )
    monkeypatch.setattr(
        "app.market.company_facts_provider.fetch_company_facts",
        lambda cik, cfg=None: {"status": "OK", "reason_code": "FETCH_OK", "size_bytes": 100_000},
    )

    def quarterly_only(ticker, years_back=10):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
                "line_item, value, units, source_url, fetched_at, filed_date, form, accession) "
                "VALUES (?, 2026, 'Q2', '2026-06-30', 'revenue', 1.0, 'USD_millions', '', "
                "'2026-09-29T00:00:00+00:00', '2026-08-03', '10-Q', 'x')",
                (ticker,),
            )

    monkeypatch.setattr("app.ingest.facts_writer.ensure_all_facts", quarterly_only)
    writer_calls: list[str] = []
    monkeypatch.setattr(
        "app.valuation.valuation_writer.ensure_valuation",
        lambda ticker, *args, **kwargs: writer_calls.append(ticker),
    )

    summary = quick_value.run_value("XOM", as_of_date="2026-09-29")

    assert summary["status"] == "FAILED"
    assert summary["reason"] == "NO_ANNUAL_REPORT"
    assert writer_calls == []
    assert quick_value.render_value_summary(summary).splitlines()[-1] == (
        "FAILED: NO_ANNUAL_REPORT - SEC companyfacts for CIK 0002115436 holds no annual report "
        "(10-K, 20-F or 40-F) filed by 2026-09-29. A newly formed registrant, such as a holding "
        "company that replaced an older registrant in a reorganization, starts a new filing "
        "history under its new CIK, and SEC data does not link it to the predecessor's; re-run "
        "once it has filed an annual report."
    )


def test_value_offline_with_nothing_cached_says_offline(isolated_data_root, monkeypatch):
    _cfg(monkeypatch, VOE_NET_PROVIDER="disabled")

    summary = quick_value.run_value("KO", as_of_date="2026-09-25")

    assert summary["status"] == "FAILED"
    assert summary["reason"] == "OFFLINE_NOT_CACHED"
    assert summary["detail"] == (
        "Offline (VOE_NET_PROVIDER=disabled) and the SEC ticker list is not cached, "
        "so KO cannot be looked up."
    )


def test_value_old_as_of_ingests_back_from_the_as_of_date(isolated_data_root, monkeypatch):
    """The ingest window counted back from today, so `--as-of 2012-06-30`
    kept only fiscal years from today-10 on and reported NO_ANNUAL_REPORT.
    The window must reach ten years before the as-of date."""
    _cfg(monkeypatch)
    monkeypatch.setattr("app.valuation.facts.resolve_cik_for_ticker", lambda t, cfg=None: "0000021344")
    monkeypatch.setattr(
        "app.market.company_facts_provider.fetch_company_facts",
        lambda cik, cfg=None: {"status": "OK", "reason_code": "CACHE_HIT"},
    )
    monkeypatch.setattr("app.util.issuer_classification.ensure_registrant_sic", lambda **k: None)
    seen: list[int] = []
    monkeypatch.setattr(
        "app.ingest.facts_writer.ensure_all_facts",
        lambda ticker, years_back=10: seen.append(years_back),
    )

    quick_value.run_value("KO", as_of_date="2012-06-30", years_back=10)

    assert seen and date.today().year - seen[0] == 2002
