from app.config import AppConfig
from app.valuation.facts import (
    UNKNOWN,
    _to_mshares,
    _to_musd,
    clear_facts_row_cache,
    resolve_cik_for_ticker,
    resolve_financial_facts_asof,
)


def test_companyfacts_cache_only_is_noncreating_and_never_constructs_http(
    monkeypatch,
    tmp_path,
):
    from app.config import get_config
    from app.market.company_facts_provider import fetch_company_facts

    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    get_config.cache_clear()
    cfg = AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "engine.db",
        cache_dir=tmp_path / "cache",
    )

    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("cache-only companyfacts attempted HTTP or a cache write")

    monkeypatch.setattr(
        "app.market.company_facts_provider.HttpClient",
        unexpected_io,
    )
    monkeypatch.setattr(
        "app.market.company_facts_provider._write_cached_companyfacts",
        unexpected_io,
    )

    result = fetch_company_facts("1", cfg=cfg, cache_only=True)

    assert result["status"] == "MISSING"
    assert result["reason_code"] == "OFFLINE_NO_CACHE"
    assert result["network_attempted"] is False
    assert result["attempts_made"] == 0
    assert not (cfg.cache_dir / "companyfacts").exists()
    get_config.cache_clear()


def test_raw_usd_below_one_million_is_always_normalized_once():
    assert _to_musd(900_000.0, "USD") == 0.9
    assert _to_musd(90_000_000.0, "USD") == 90.0


def test_raw_shares_below_one_million_are_always_normalized_once():
    assert _to_mshares(500_000.0, "shares") == 0.5
    assert _to_mshares(50_000_000.0, "shares") == 50.0


def test_missing_or_ambiguous_units_fail_closed():
    assert _to_musd(900_000.0, None) == UNKNOWN
    assert _to_musd(0.9, "USD_millions") == UNKNOWN
    assert _to_mshares(500_000.0, None) == UNKNOWN
    assert _to_mshares(0.5, "shares_millions") == UNKNOWN


def test_universe_cik_cache_is_scoped_to_active_universe_root(
    monkeypatch,
    tmp_path,
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_universe = first_root / "universe.csv"
    second_universe = second_root / "universe.csv"
    first_universe.write_text(
        "ticker,cik,name\nROOT,111,First Root Issuer\n",
        encoding="utf-8",
    )
    second_universe.write_text(
        "ticker,cik,name\nROOT,222,Second Root Issuer\n",
        encoding="utf-8",
    )
    first_cfg = AppConfig(
        data_dir=first_root,
        db_path=first_root / "engine.db",
        universe_path=first_universe,
        cache_dir=first_root / "cache",
    )
    second_cfg = AppConfig(
        data_dir=second_root,
        db_path=second_root / "engine.db",
        universe_path=second_universe,
        cache_dir=second_root / "cache",
    )

    def unavailable_db(*_args, **_kwargs):
        raise RuntimeError("database intentionally unavailable")

    monkeypatch.setattr("app.valuation.facts.get_db", unavailable_db)
    clear_facts_row_cache()
    first = resolve_cik_for_ticker(
        "ROOT",
        cfg=first_cfg,
        refresh_if_missing=False,
    )
    second = resolve_cik_for_ticker(
        "ROOT",
        cfg=second_cfg,
        refresh_if_missing=False,
    )
    clear_facts_row_cache()

    assert first == "0000000111"
    assert second == "0000000222"


def test_financial_facts_cache_is_scoped_to_active_data_roots(
    monkeypatch,
    tmp_path,
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_cfg = AppConfig(
        data_dir=first_root,
        db_path=first_root / "engine.db",
        universe_path=first_root / "universe.csv",
        cache_dir=first_root / "cache",
    )
    second_cfg = AppConfig(
        data_dir=second_root,
        db_path=second_root / "engine.db",
        universe_path=second_root / "universe.csv",
        cache_dir=second_root / "cache",
    )
    fetch_roots: list[str] = []

    monkeypatch.setattr(
        "app.valuation.facts.resolve_cik_for_ticker",
        lambda *_args, **_kwargs: "0000000001",
    )

    def fake_fetch(_cik, **kwargs):
        cfg = kwargs["cfg"]
        cache_root = str(cfg.cache_dir)
        fetch_roots.append(cache_root)
        return {
            "status": "OK",
            "reason_code": "CACHE_HIT",
            "reason_detail": "Literal test cache.",
            "source_resolution": "companyfacts_cache",
            "source_url": ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"),
            "cache_path": str(cfg.cache_dir / "0000000001.json"),
            "companyfacts": {"cache_root": cache_root},
            "network_attempted": False,
        }

    monkeypatch.setattr(
        "app.valuation.facts.fetch_company_facts",
        fake_fetch,
    )

    def fake_extract(payload, _as_of_date):
        raw_shares = (
            10_000_000.0 if payload["cache_root"] == str(first_cfg.cache_dir) else 20_000_000.0
        )
        return {
            "shares_outstanding_asof": {
                "value": raw_shares,
                "unit": "shares",
                "fact_end_date": "2025-12-31",
                "filed_date": "2026-02-01",
                "derived_from": ["companyfacts.dei.shares"],
            },
            "cfo_asof": None,
            "capex_asof": None,
            "fcf_asof": None,
        }

    monkeypatch.setattr(
        "app.valuation.facts.extract_company_facts_asof",
        fake_extract,
    )

    clear_facts_row_cache()
    first = resolve_financial_facts_asof(
        ticker="AAA",
        as_of_date="2026-02-13",
        cfg=first_cfg,
    )
    second = resolve_financial_facts_asof(
        ticker="AAA",
        as_of_date="2026-02-13",
        cfg=second_cfg,
    )
    clear_facts_row_cache()

    assert first["shares_value"] == 10.0
    assert second["shares_value"] == 20.0
    assert fetch_roots == [
        str(first_cfg.cache_dir),
        str(second_cfg.cache_dir),
    ]
