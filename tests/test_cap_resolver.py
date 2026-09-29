from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.autonomous import cap_resolver
from app.autonomous.cap_resolver import (
    CAP_SOURCE_ASOF_COMPANYFACTS,
    CAP_SOURCE_EODHD_FUNDAMENTALS,
    CAP_SOURCE_STALE_SHARES,
    CAP_SOURCE_TERMINAL_EXCHANGE,
    CAP_SOURCE_TERMINAL_LOCAL,
    CAP_SOURCE_TERMINAL_SEARCH,
    CAP_SOURCE_UNKNOWN,
    SECURITY_ROLE_ADR,
    SECURITY_ROLE_PRIMARY,
    SECURITY_ROLE_SECONDARY_CLASS,
    SECURITY_ROLE_SECONDARY_SECURITY,
    TerminalCapEvidence,
    UNKNOWN_CAP_LABEL,
    band_for_market_cap,
    classify_market_cap_for_band_filter,
    resolve_security_identity,
)
from app.config import AppConfig
from app.market.price_provider import PriceSnapshot
from tests.financial_integrity_helpers import (
    materialized_split_proof as _materialized_split_proof,
)


COMPANYFACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS companyfacts_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    period_type TEXT NOT NULL DEFAULT 'FY',
    period_end TEXT NOT NULL,
    line_item TEXT NOT NULL,
    value REAL,
    units TEXT,
    source_url TEXT,
    fetched_at TEXT NOT NULL,
    filed_date TEXT,
    form TEXT,
    accession TEXT,
    UNIQUE(ticker, fiscal_year, period_type, line_item)
)
"""


def _seed_db(tmp_path: Path, rows: list[tuple[str, int, str, str, str, float]]) -> Path:
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(COMPANYFACTS_SCHEMA)
    for ticker, fiscal_year, period_type, period_end, line_item, value in rows:
        conn.execute(
            """
            INSERT INTO companyfacts_facts
                (ticker, fiscal_year, period_type, period_end, line_item, value,
                 units, source_url, fetched_at, filed_date, form, accession)
            VALUES (?, ?, ?, ?, ?, ?, 'shares_millions',
                    'https://data.sec.gov/api/xbrl/companyfacts/test.json',
                    '2026-06-11T00:00:00+00:00', '2026-06-01', '10-K', 'test')
            """,
            (ticker, fiscal_year, period_type, period_end, line_item, value),
        )
    conn.commit()
    conn.close()
    return db_path


def _strict_miss(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        lambda **kwargs: (None, {"market_cap_reason_code": "SHARES_UNKNOWN"}),
    )


def _unadjusted_quote(
    price: float,
    *,
    ticker: str,
    issuer_cik: str = "0000000001",
    quote_as_of: str = "2026-06-10",
    shares_as_of: str = "2025-12-31",
    source: str = "fixture_quote",
    source_url: str = "https://example.test/quote",
) -> dict[str, object]:
    split_proof = _materialized_split_proof(
        {
            "ticker": ticker.upper(),
            "status": "PASS",
            "period_start": shares_as_of,
            "period_end": quote_as_of,
            "verified_as_of": quote_as_of,
            "issuer_cik": issuer_cik,
            "source": "fixture_corporate_actions",
            "source_reference": f"https://eodhd.com/api/splits/{ticker.upper()}",
        }
    )
    return {
        "price": price,
        "raw_price": price,
        "price_basis": "UNADJUSTED",
        "split_adjustment_factor": 1.0,
        "as_of_date": quote_as_of,
        "currency": "USD",
        "source": source,
        "source_url": source_url,
        "no_intervening_split_proof": split_proof,
    }


def _authoritative_primary_identity(
    ticker: str,
    *,
    identity_as_of_date: str = "2026-06-10",
) -> dict[str, object]:
    upper = ticker.upper()
    return {
        "issuer_primary_ticker": upper,
        "issuer_listed_tickers": [upper],
        "security_role": "PRIMARY",
        "is_secondary_class": False,
        "is_adr": False,
        "identity_source": "sec_submissions_exchange_binding",
        "identity_source_url": (f"https://data.sec.gov/submissions/CIK0000000001.json#{upper}"),
        "identity_as_of_date": identity_as_of_date,
        "identity_confidence": "HIGH",
    }


def test_band_for_market_cap_boundaries():
    # millions USD; lower-inclusive, upper-exclusive (2026-06-11 redefinition).
    assert band_for_market_cap(150.0) == "micro"
    assert band_for_market_cap(499.0) == "micro"
    assert band_for_market_cap(500.0) == "small"
    assert band_for_market_cap(2_499.0) == "small"
    assert band_for_market_cap(2_500.0) == "mid"
    assert band_for_market_cap(9_999.0) == "mid"
    assert band_for_market_cap(10_000.0) == "large_cap"
    assert band_for_market_cap(199_999.99) == "large_cap"
    assert band_for_market_cap(200_000.0) == "mega_cap"
    assert band_for_market_cap(None) is None
    assert band_for_market_cap(0.0) is None


def test_tier1_strict_asof_companyfacts_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        lambda **kwargs: (
            1234.0,
            {
                "shares_outstanding": 10.0,
                "shares_asof_used": "2026-06-10",
                "shares_source_resolution": "sec_companyfacts",
                "shares_source_url": (
                    "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
                ),
                "derived_from": ["/tmp/companyfacts/CIK0000000001.json"],
            },
        ),
    )
    result = classify_market_cap_for_band_filter(
        "ABCD",
        as_of_date="2026-06-11",
        asof_price=123.4,
        asof_price_provenance=_unadjusted_quote(
            123.4,
            ticker="ABCD",
            quote_as_of="2026-06-11",
            shares_as_of="2026-06-10",
        ),
        db_path=tmp_path / "missing.db",
        price_lookup=lambda ticker, as_of: None,
        identity_evidence=_authoritative_primary_identity("ABCD"),
    )
    assert result.market_cap_mm == 1234.0
    assert result.cap_source == CAP_SOURCE_ASOF_COMPANYFACTS
    assert result.cap_band == "small"
    assert result.shares_mm == 10.0
    assert result.shares_period_end == "2026-06-10"
    assert result.cap_source_kind == "SEC"
    assert result.cap_source_url == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    )
    assert result.in_band(500.0, 2_500.0) is True
    assert result.in_band(0.0, 500.0) is False


def test_v1_tier1_uses_explicit_config_and_database_root(
    monkeypatch,
    tmp_path,
):
    from app.valuation import shares as shares_module

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_cfg = AppConfig(
        data_dir=first_root,
        db_path=first_root / "engine.db",
        universe_path=first_root / "universe.csv",
        cache_dir=first_root / "cache",
        sectors_dir=first_root / "outputs" / "sectors",
        dossiers_dir=first_root / "outputs" / "dossiers",
    )
    second_cfg = AppConfig(
        data_dir=second_root,
        db_path=second_root / "copied-engine.db",
        universe_path=second_root / "universe.csv",
        cache_dir=second_root / "cache",
        sectors_dir=second_root / "outputs" / "sectors",
        dossiers_dir=second_root / "outputs" / "dossiers",
    )
    observed_cache_roots: list[Path] = []
    monkeypatch.setattr(shares_module, "get_config", lambda: first_cfg)

    def fake_facts(
        *,
        ticker,
        as_of_date,
        run_id=None,
        refresh=False,
        cfg=None,
        **_kwargs,
    ):
        del ticker, as_of_date, run_id, refresh
        observed_cache_roots.append(cfg.cache_dir)
        shares = 10.0 if cfg.cache_dir == first_cfg.cache_dir else 20.0
        return {
            "status": "PARTIAL",
            "shares_value": shares,
            "shares_raw_value": shares * 1_000_000.0,
            "shares_input_unit": "shares",
            "shares_output_unit": "shares_millions",
            "shares_asof_used": "2025-12-31",
            "shares_filed_date": "2026-02-01",
            "source_resolution": "companyfacts_cache",
            "source_url": ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"),
            "derived_from": [f"cache:{cfg.cache_dir}"],
            "fetch_reason_code": "CACHE_HIT",
            "fetch_reason_detail": "Controlled offline cache.",
        }

    monkeypatch.setattr(
        shares_module,
        "resolve_financial_facts_asof",
        fake_facts,
    )

    result = classify_market_cap_for_band_filter(
        "ROOT",
        as_of_date="2026-06-10",
        asof_price=10.0,
        asof_price_provenance=_unadjusted_quote(
            10.0,
            ticker="ROOT",
            quote_as_of="2026-06-10",
            shares_as_of="2025-12-31",
        ),
        db_path=second_cfg.db_path,
        cfg=second_cfg,
        pipeline_version="v1",
        identity_evidence=_authoritative_primary_identity("ROOT"),
    )

    assert observed_cache_roots == [second_cfg.cache_dir]
    assert result.shares_mm == 20.0
    assert result.market_cap_mm == 200.0
    assert result.cap_source == CAP_SOURCE_ASOF_COMPANYFACTS


def test_vnom_profile_resolves_via_stale_shares_and_exits_micro_band(monkeypatch, tmp_path):
    # VNOM profile: strict as-of cap uncomputable, but companyfacts holds a
    # last-known share count. 58.0mm shares x $155.00 = $8,990mm -> mid band,
    # so the name must never appear as in-band output for micro (0-500).
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [
            ("VNOM", 2025, "FY", "2025-12-31", "shares_outstanding", 57.0),
            ("VNOM", 2026, "Q1", "2026-03-31", "shares_outstanding", 58.0),
        ],
    )
    quote = _unadjusted_quote(155.0, ticker="VNOM", shares_as_of="2026-03-31")
    result = classify_market_cap_for_band_filter(
        "VNOM",
        as_of_date="2026-06-11",
        asof_price=155.0,
        asof_price_provenance=quote,
        current_price=quote,
        db_path=db_path,
        identity_evidence=_authoritative_primary_identity("VNOM"),
    )
    assert result.market_cap_mm == 8990.0
    assert result.cap_source == CAP_SOURCE_STALE_SHARES
    assert result.cap_band == "mid"
    assert result.shares_mm == 58.0
    assert result.shares_period_end == "2026-03-31"
    assert result.in_band(0.0, 500.0) is False
    assert result.band_label == "mid"


def test_stale_companyfacts_raw_share_unit_is_normalized_without_magnitude_inference(
    monkeypatch,
    tmp_path,
):
    _strict_miss(monkeypatch)
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(COMPANYFACTS_SCHEMA)
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            )
            VALUES(
                'RAWU', 2026, 'Q1', '2026-03-31', 'shares_outstanding',
                58000000, 'shares',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                '2026-05-01T00:00:00+00:00', '2026-05-01', '10-Q',
                '0000000001-26-000001'
            )
            """
        )

    quote = _unadjusted_quote(10.0, ticker="RAWU", shares_as_of="2026-03-31")
    result = classify_market_cap_for_band_filter(
        "RAWU",
        as_of_date="2026-06-11",
        asof_price=10.0,
        asof_price_provenance=quote,
        current_price=quote,
        db_path=db_path,
        identity_evidence=_authoritative_primary_identity("RAWU"),
    )

    assert result.market_cap_mm == 580.0
    assert result.shares_mm == 58.0
    assert result.raw_shares_outstanding_mm == 58.0
    assert result.raw_shares_source_value == 58_000_000.0
    assert result.raw_shares_source_unit == "shares"


def test_stale_shares_respects_asof_date(monkeypatch, tmp_path):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("FUTR", 2027, "Q1", "2027-01-01", "shares_outstanding", 99.0)],
    )
    result = classify_market_cap_for_band_filter(
        "FUTR",
        as_of_date="2026-06-11",
        asof_price=10.0,
        current_price=10.0,
        db_path=db_path,
        identity_evidence=_authoritative_primary_identity("FUTR"),
    )
    assert result.cap_source == CAP_SOURCE_UNKNOWN
    assert result.market_cap_mm is None
    assert result.detail == "no_companyfacts_shares"


def test_v1_candidate_band_uses_only_shares_filed_as_of_scan(
    monkeypatch,
    tmp_path,
):
    _strict_miss(monkeypatch)
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(COMPANYFACTS_SCHEMA)
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            )
            VALUES(
                'PITB', ?, ?, ?, 'shares_outstanding', ?,
                'shares_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/test.json',
                '2026-08-01T00:00:00+00:00', ?, '10-Q', ?
            )
            """,
            [
                (2025, "FY", "2025-12-31", 20.0, "2026-02-15", "visible"),
                (2026, "Q1", "2026-03-31", 200.0, "2026-08-01", "future"),
                (2026, "Q2", "2026-05-31", 300.0, None, "undated"),
            ],
        )

    quote = _unadjusted_quote(20.0, ticker="PITB")
    result = classify_market_cap_for_band_filter(
        "PITB",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance=quote,
        current_price=quote,
        db_path=db_path,
        pipeline_version="v1",
        identity_evidence=_authoritative_primary_identity("PITB"),
    )

    assert result.market_cap_mm == 400.0
    assert result.cap_band == "micro"
    assert result.shares_mm == 20.0
    assert result.shares_period_end == "2025-12-31"
    assert result.shares_filed_date == "2026-02-15"
    assert result.in_band(0.0, 500.0) is True
    assert result.in_band(500.0, 10_000.0) is False


def test_v1_ambiguous_split_basis_cannot_claim_pre_gate_candidate_band(
    monkeypatch,
    tmp_path,
):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("BASIS", 2025, "FY", "2025-12-31", "shares_outstanding", 10.0)],
    )

    result = classify_market_cap_for_band_filter(
        "BASIS",
        as_of_date="2026-06-11",
        asof_price=25.0,
        current_price={
            "price": 25.0,
            "raw_price": 100.0,
            "as_of_date": "2026-06-10",
            "currency": "USD",
            "source": "ambiguous_adjusted_quote",
        },
        price_lookup=lambda _ticker, _as_of: None,
        db_path=db_path,
        pipeline_version="v1",
        identity_evidence=_authoritative_primary_identity("BASIS"),
    )

    assert result.market_cap_mm is None
    assert result.cap_band is None
    assert result.band_label == UNKNOWN_CAP_LABEL
    assert result.in_band(0.0, 500.0) is None
    assert result.in_band(500.0, 2_500.0) is None
    assert result.detail == "shares_known_price_missing"


def test_unknown_when_no_shares_anywhere(monkeypatch, tmp_path):
    _strict_miss(monkeypatch)
    db_path = _seed_db(tmp_path, [])
    result = classify_market_cap_for_band_filter(
        "NOPE",
        as_of_date="2026-06-11",
        asof_price=5.0,
        current_price=5.0,
        db_path=db_path,
    )
    assert result.cap_source == CAP_SOURCE_UNKNOWN
    assert result.cap_band is None
    assert result.band_label == UNKNOWN_CAP_LABEL
    assert result.in_band(0.0, 500.0) is None


def test_tier3_uses_injected_price_lookup(monkeypatch, tmp_path):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("TINY", 2025, "FY", "2025-12-31", "shares_outstanding", 2.0)],
    )
    result = classify_market_cap_for_band_filter(
        "TINY",
        as_of_date="2026-06-11",
        asof_price=None,
        db_path=db_path,
        price_lookup=lambda ticker, as_of: _unadjusted_quote(10.0, ticker=ticker),
        identity_evidence=_authoritative_primary_identity("TINY"),
    )
    assert result.market_cap_mm == 20.0
    assert result.cap_source == CAP_SOURCE_STALE_SHARES
    assert result.cap_band == "micro"
    assert result.detail == "shares_period_end=2025-12-31:price_origin=provider"


def test_tier3_falls_back_to_asof_price_when_provider_has_nothing(monkeypatch, tmp_path):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("OFLN", 2025, "FY", "2025-12-31", "shares_outstanding", 2.0)],
    )
    result = classify_market_cap_for_band_filter(
        "OFLN",
        as_of_date="2026-06-11",
        asof_price=12.0,
        asof_price_provenance=_unadjusted_quote(12.0, ticker="OFLN"),
        db_path=db_path,
        price_lookup=lambda ticker, as_of: None,
        identity_evidence=_authoritative_primary_identity("OFLN"),
    )
    assert result.market_cap_mm == 24.0
    assert result.cap_source == CAP_SOURCE_STALE_SHARES
    assert result.detail == "shares_period_end=2025-12-31:price_origin=asof_price_fallback"


def test_eodhd_fundamentals_tier_is_inert_by_default(monkeypatch, tmp_path):
    _strict_miss(monkeypatch)

    def _explode(ticker, cfg):
        raise AssertionError("EODHD fundamentals tier must not be called by default")

    monkeypatch.setattr(cap_resolver, "_eodhd_fundamentals_market_cap_mm", _explode)
    db_path = _seed_db(tmp_path, [])
    result = classify_market_cap_for_band_filter(
        "INRT",
        as_of_date="2026-06-11",
        asof_price=5.0,
        current_price=5.0,
        db_path=db_path,
    )
    assert result.cap_source == CAP_SOURCE_UNKNOWN


def test_eodhd_fundamentals_tier_activates_only_with_config_flag(monkeypatch, tmp_path):
    _strict_miss(monkeypatch)
    monkeypatch.setattr(
        cap_resolver,
        "_eodhd_fundamentals_market_cap_mm",
        lambda ticker, cfg: (9_000.0, "OK"),
    )
    cfg = AppConfig(cap_eodhd_fundamentals_enabled=True)
    db_path = _seed_db(tmp_path, [])
    result = classify_market_cap_for_band_filter(
        "UPGR",
        as_of_date="2026-06-11",
        asof_price=5.0,
        current_price=5.0,
        db_path=db_path,
        cfg=cfg,
    )
    assert result.market_cap_mm == 9_000.0
    assert result.cap_source == CAP_SOURCE_EODHD_FUNDAMENTALS
    assert result.cap_band == "mid"


def test_to_dict_carries_band_label():
    classification = cap_resolver.CapClassification(
        ticker="ZZZZ",
        as_of_date="2026-06-11",
        market_cap_mm=None,
        cap_source=CAP_SOURCE_UNKNOWN,
        cap_band=None,
    )
    payload = classification.to_dict()
    assert payload["band_label"] == "UNKNOWN_CAP"
    assert payload["cap_source"] == "unknown"


def test_stale_shares_too_old_cannot_claim_a_band(monkeypatch, tmp_path):
    # IBTA class: the only cached share count predates the as-of date by more
    # than the staleness window (e.g. a pre-IPO cover page). A band claim off
    # archaeology is worse than an honest UNKNOWN_CAP.
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("IBTA", 2023, "FY", "2023-12-31", "shares_outstanding", 9.207337)],
    )
    result = classify_market_cap_for_band_filter(
        "IBTA",
        as_of_date="2026-06-11",
        asof_price=31.74,
        current_price=31.74,
        db_path=db_path,
        identity_evidence=_authoritative_primary_identity("IBTA"),
    )
    assert result.cap_source == CAP_SOURCE_UNKNOWN
    assert result.market_cap_mm is None
    assert result.cap_band is None
    assert result.detail == "stale_shares_too_old:2023-12-31"
    assert result.shares_period_end == "2023-12-31"


def test_stale_shares_within_window_still_resolves(monkeypatch, tmp_path):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("FRSH", 2025, "FY", "2025-12-31", "shares_outstanding", 142.53)],
    )
    quote = _unadjusted_quote(44.78, ticker="FRSH")
    result = classify_market_cap_for_band_filter(
        "FRSH",
        as_of_date="2026-06-11",
        asof_price=44.78,
        asof_price_provenance=quote,
        current_price=quote,
        db_path=db_path,
        identity_evidence=_authoritative_primary_identity("FRSH"),
    )
    assert result.cap_source == CAP_SOURCE_STALE_SHARES
    assert round(result.market_cap_mm, 1) == 6382.5
    assert result.cap_band == "mid"


def test_split_adjusted_quote_normalizes_companyfacts_shares_before_market_cap(
    monkeypatch,
    tmp_path,
):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("SPLT", 2025, "FY", "2025-12-31", "shares_outstanding", 20.0)],
    )
    adjusted_quote = {
        "price": 25.0,
        "raw_price": 100.0,
        "price_basis": "SPLIT_ADJUSTED",
        "split_adjustment_factor": 4.0,
        "split_effective_date": "2026-01-15",
        "as_of_date": "2026-06-11",
        "currency": "USD",
        "source": "adjusted_quote_fixture",
        "source_url": "https://example.test/SPLT",
        "split_event": _materialized_split_proof(
            {
                "ticker": "SPLT",
                "factor": 4.0,
                "effective_date": "2026-01-15",
                "filed_date": "2026-01-10",
                "source": "issuer_split_filing",
                "source_reference": (
                    "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/split-event.htm"
                ),
            }
        ),
    }

    result = classify_market_cap_for_band_filter(
        "SPLT",
        as_of_date="2026-06-11",
        asof_price=25.0,
        asof_price_provenance=adjusted_quote,
        current_price=adjusted_quote,
        db_path=db_path,
        identity_evidence=_authoritative_primary_identity("SPLT"),
    )

    assert result.market_cap_mm == 2000.0
    assert result.cap_source == CAP_SOURCE_STALE_SHARES
    assert result.price_used == 25.0
    assert result.price_basis == "SPLIT_ADJUSTED"
    assert result.shares_mm == 80.0
    assert result.raw_shares_outstanding_mm == 20.0
    assert result.shares_basis == "SPLIT_ADJUSTED"
    assert result.cap_band == "small"
    assert result.in_band(500.0, 2_500.0) is True


def test_cap_resolution_rejects_split_proof_bound_to_different_issuer(
    monkeypatch,
    tmp_path,
):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("BOUND", 2025, "FY", "2025-12-31", "shares_outstanding", 20.0)],
    )
    quote = _unadjusted_quote(
        25.0,
        ticker="BOUND",
        issuer_cik="0000000002",
    )

    result = classify_market_cap_for_band_filter(
        "BOUND",
        as_of_date="2026-06-11",
        asof_price=25.0,
        asof_price_provenance=quote,
        current_price=quote,
        price_lookup=lambda _ticker, _as_of: None,
        db_path=db_path,
        identity_evidence={
            **_authoritative_primary_identity("BOUND"),
            "issuer_cik": "0000000001",
        },
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN


@pytest.mark.parametrize(
    "source_reference",
    [
        "https://example.test/SPLT/invented-split",
        "https://www.sec.gov/fake-split",
    ],
)
def test_split_adjusted_quote_rejects_invented_split_reference(
    monkeypatch,
    tmp_path,
    source_reference,
):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("SPLT", 2025, "FY", "2025-12-31", "shares_outstanding", 20.0)],
    )
    invented_quote = {
        "price": 25.0,
        "raw_price": 100.0,
        "price_basis": "SPLIT_ADJUSTED",
        "split_adjustment_factor": 4.0,
        "split_effective_date": "2026-01-15",
        "as_of_date": "2026-06-11",
        "currency": "USD",
        "source": "adjusted_quote_fixture",
        "source_url": "https://example.test/SPLT",
        "split_event": {
            "factor": 4.0,
            "effective_date": "2026-01-15",
            "filed_date": "2026-01-10",
            "source": "invented_split",
            "ticker": "SPLT",
            "source_reference": source_reference,
        },
    }

    result = classify_market_cap_for_band_filter(
        "SPLT",
        as_of_date="2026-06-11",
        asof_price=25.0,
        asof_price_provenance=invented_quote,
        current_price=invented_quote,
        price_lookup=lambda _ticker, _as_of: None,
        db_path=db_path,
        identity_evidence=_authoritative_primary_identity("SPLT"),
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN


def test_split_adjusted_quote_with_conflicting_factor_fails_closed(
    monkeypatch,
    tmp_path,
):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("SPLT", 2025, "FY", "2025-12-31", "shares_outstanding", 20.0)],
    )
    conflicting_quote = {
        "price": 25.0,
        "raw_price": 100.0,
        "price_basis": "SPLIT_ADJUSTED",
        "split_adjustment_factor": 2.0,
        "split_effective_date": "2026-01-15",
        "as_of_date": "2026-06-11",
        "currency": "USD",
        "source": "adjusted_quote_fixture",
        "source_url": "https://example.test/SPLT",
        "split_event": _materialized_split_proof(
            {
                "ticker": "SPLT",
                "factor": 2.0,
                "effective_date": "2026-01-15",
                "filed_date": "2026-01-10",
                "source": "issuer_split_filing",
                "source_reference": (
                    "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/split-event.htm"
                ),
            }
        ),
    }

    result = classify_market_cap_for_band_filter(
        "SPLT",
        as_of_date="2026-06-11",
        asof_price=25.0,
        asof_price_provenance=conflicting_quote,
        current_price=conflicting_quote,
        price_lookup=lambda _ticker, _as_of: None,
        db_path=db_path,
        identity_evidence=_authoritative_primary_identity("SPLT"),
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN
    assert result.price_basis is None
    assert result.detail == "shares_known_price_missing"


def test_known_adr_without_authoritative_ratio_refuses_issuer_shares_times_quote(
    monkeypatch, tmp_path
):
    def _unsafe_call(**kwargs):
        raise AssertionError("strict shares x ADR quote must not run without a ratio")

    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        _unsafe_call,
    )
    db_path = _seed_db(
        tmp_path,
        [("ADRX", 2025, "FY", "2025-12-31", "shares_outstanding", 100.0)],
    )

    result = classify_market_cap_for_band_filter(
        "ADRX",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance={"as_of_date": "2026-06-10", "currency": "USD"},
        current_price={"price": 20.0, "as_of_date": "2026-06-10", "currency": "USD"},
        db_path=db_path,
        pipeline_version="v2",
        identity_evidence={
            "issuer_cik": "1234567",
            "issuer_primary_ticker": "ADRX",
            "security_role": "ADR",
            "is_adr": True,
            "identity_source_url": "https://www.sec.gov/Archives/example-20f.htm",
            "identity_confidence": "HIGH",
        },
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN
    assert result.security_role == SECURITY_ROLE_ADR
    assert result.is_adr is True
    assert result.price_used == 20.0
    assert result.detail == (
        "UNSAFE_ISSUER_SHARES_ADR_RATIO_MISSING:terminal_direct_cap_evidence_unavailable"
    )


def test_adr_ratio_adjusts_issuer_wide_shares_cap_and_preserves_price_provenance(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        lambda **kwargs: (
            2_000.0,
            {
                "shares_outstanding": 100.0,
                "shares_asof_used": "2026-03-31",
                "shares_source_resolution": "sec_companyfacts",
            },
        ),
    )
    db_path = _seed_db(
        tmp_path,
        [("ADRR", 2026, "Q1", "2026-03-31", "shares_outstanding", 100.0)],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE companyfacts_facts SET source_url = ? WHERE ticker = 'ADRR'",
            ("https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json",),
        )
    result = classify_market_cap_for_band_filter(
        "ADRR",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance={
            **_unadjusted_quote(
                20.0,
                ticker="ADRR",
                issuer_cik="0001234567",
                shares_as_of="2026-03-31",
                source="nasdaq_close",
                source_url="https://api.nasdaq.com/example",
            ),
            "as_of_date": "2026-06-10",
            "source": "nasdaq_close",
            "source_url": "https://api.nasdaq.com/example",
            "currency": "USD",
            "confidence": "HIGH",
        },
        db_path=db_path,
        pipeline_version="v2",
        identity_evidence={
            "issuer_cik": "1234567",
            "issuer_primary_ticker": "ADRR",
            "is_adr": True,
            "adr_ratio": 2.0,
            "ratio_source_url": (
                "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/issuer-20f.htm"
            ),
            "ratio_source_accession": "0001234567-26-000001",
            "ratio_security_symbol": "ADRR",
            "identity_source": "issuer_filing",
            "identity_source_url": (
                "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/issuer-20f.htm"
            ),
            "identity_as_of_date": "2026-04-01",
            "identity_confidence": "HIGH",
        },
    )

    assert result.market_cap_mm == 1_000.0
    assert result.cap_source == CAP_SOURCE_ASOF_COMPANYFACTS
    assert result.adr_ratio == 2.0
    assert result.ratio_source_accession == "0001234567-26-000001"
    assert result.ratio_security_symbol == "ADRR"
    assert result.price_used == 20.0
    assert result.price_as_of_date == "2026-06-10"
    assert result.price_source == "nasdaq_close"
    assert result.price_source_url == "https://api.nasdaq.com/example"
    assert result.price_confidence == "HIGH"
    assert result.cap_effective_as_of_date == "2026-06-10"
    assert "quote_to_issuer_share_ratio=2.0" in str(result.detail)


def test_secondary_class_without_authoritative_ratio_is_not_derived(monkeypatch, tmp_path):
    def _unsafe_call(**kwargs):
        raise AssertionError("strict shares x class quote must not run without a ratio")

    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        _unsafe_call,
    )
    db_path = _seed_db(
        tmp_path,
        [("DUALB", 2025, "FY", "2025-12-31", "shares_outstanding", 50.0)],
    )
    result = classify_market_cap_for_band_filter(
        "DUALB",
        as_of_date="2026-06-11",
        asof_price=100.0,
        current_price=100.0,
        db_path=db_path,
        pipeline_version="v2",
        identity_evidence={
            "issuer_cik": "7654321",
            "issuer_primary_ticker": "DUALA",
            "issuer_listed_tickers": ["DUALA", "DUALB"],
            "security_role": "SECONDARY_CLASS",
            "identity_source_url": "https://data.sec.gov/submissions/CIK0007654321.json",
            "identity_confidence": "HIGH",
        },
    )

    assert result.market_cap_mm is None
    assert result.security_role == SECURITY_ROLE_SECONDARY_CLASS
    assert result.is_secondary_class is True
    assert result.detail == (
        "UNSAFE_ISSUER_SHARES_SECONDARY_CLASS_RATIO_MISSING:"
        "terminal_direct_cap_evidence_unavailable"
    )


def test_search_terminal_direct_cap_resolves_unsafe_adr_with_required_provenance(
    monkeypatch, tmp_path
):
    def _unsafe_call(**kwargs):
        raise AssertionError("ADR shares multiplication must remain bypassed")

    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        _unsafe_call,
    )
    db_path = _seed_db(
        tmp_path,
        [("ADRS", 2025, "FY", "2025-12-31", "shares_outstanding", 100.0)],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE companies (ticker TEXT, cik TEXT)")
        conn.execute("INSERT INTO companies VALUES ('ADRS', '1234567')")
    seen: dict[str, object] = {}

    def lookup(ticker, as_of_date, identity):
        seen.update(ticker=ticker, as_of_date=as_of_date, is_adr=identity.is_adr)
        return TerminalCapEvidence(
            ticker="ADRS",
            market_cap_mm=12_500.0,
            source_kind="SEARCH",
            source_name="exchange_screener_search",
            source_url="https://exchange.example.com/security/ADRS",
            as_of_date="2026-06-10",
            confidence="HIGH",
            issuer_cik="1234567",
            detail="direct issuer market cap from dated exchange result",
        )

    result = classify_market_cap_for_band_filter(
        "ADRS",
        as_of_date="2026-06-11",
        asof_price=20.0,
        current_price=20.0,
        db_path=db_path,
        identity_evidence={"issuer_cik": "1234567", "is_adr": True},
        terminal_cap_lookup=lookup,
        pipeline_version="v2",
    )

    assert seen == {"ticker": "ADRS", "as_of_date": "2026-06-11", "is_adr": True}
    assert result.market_cap_mm == 12_500.0
    assert result.cap_source == CAP_SOURCE_TERMINAL_SEARCH
    assert result.cap_source_kind == "SEARCH"
    assert result.cap_source_url == "https://exchange.example.com/security/ADRS"
    assert result.cap_effective_as_of_date == "2026-06-10"
    assert result.cap_confidence == "HIGH"


@pytest.mark.parametrize("missing_field", ["source_url", "as_of_date", "confidence"])
def test_search_terminal_mapping_requires_url_asof_and_confidence(
    monkeypatch, tmp_path, missing_field
):
    _strict_miss(monkeypatch)
    db_path = _seed_db(tmp_path, [])

    def lookup(ticker, as_of, identity):
        payload = {
            "ticker": "MISS",
            "market_cap_mm": 11_000.0,
            "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
            "market_cap_units": "USD_millions",
            "source_kind": "SEARCH",
            "source_name": "search_result",
            "source_url": "https://exchange.example.com/MISS",
            "as_of_date": "2026-06-11",
            "confidence": "HIGH",
        }
        del payload[missing_field]
        return payload

    result = classify_market_cap_for_band_filter(
        "MISS",
        as_of_date="2026-06-11",
        db_path=db_path,
        price_lookup=lambda ticker, as_of: None,
        terminal_cap_lookup=lookup,
        pipeline_version="v2",
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN
    assert result.detail == (
        "UNSAFE_ISSUER_SHARES_SECURITY_ROLE_UNRESOLVED:terminal_direct_cap_evidence_unavailable"
    )


def test_explicit_direct_local_market_cap_row_precedes_derived_tiers(tmp_path):
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE market_caps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            effective_as_of_date TEXT NOT NULL,
            market_cap REAL,
            market_cap_status TEXT NOT NULL,
            provider TEXT,
            source_url TEXT,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE TABLE companies (ticker TEXT, cik TEXT)")
    conn.execute("INSERT INTO companies VALUES ('ZXLO', '1234567')")
    conn.execute(
        """
        INSERT INTO market_caps(
            ticker, effective_as_of_date, market_cap, market_cap_status,
            provider, source_url, payload_json
        ) VALUES (?, ?, ?, 'OK', ?, ?, ?)
        """,
        (
            "ZXLO",
            "2026-06-10",
            14_000.0,
            "authoritative_exchange_file",
            "https://exchange.example.com/daily-market-cap.csv",
            """{
              "ticker": "ZXLO",
              "market_cap_mm": 14000.0,
              "market_cap_basis": "DIRECT_ISSUER_MARKET_CAP",
              "market_cap_units": "USD_millions",
              "market_cap_source_kind": "LOCAL_AUTHORITATIVE",
              "market_cap_confidence": "HIGH",
              "issuer_cik": "1234567",
              "local_authority_schema": "VOE_TERMINAL_CAP_EVIDENCE_V1",
              "record_provenance": "PERSISTED_RUN_ARTIFACT",
              "source_name": "cached_exchange_file",
              "as_of_date": "2026-06-10"
            }""",
        ),
    )
    conn.commit()
    conn.close()

    def _search_must_not_run(ticker, as_of_date, identity):
        raise AssertionError("local direct-cap evidence must precede search fallback")

    result = classify_market_cap_for_band_filter(
        "ZXLO",
        as_of_date="2026-06-11",
        db_path=db_path,
        price_lookup=lambda ticker, as_of: PriceSnapshot(
            ticker=ticker,
            as_of_date=as_of,
            price=10.0,
            source="fixed_asof_exchange_close",
            retrieved_at="2026-06-11T12:00:00Z",
            url="https://exchange.example.com/ZXLO/close",
            confidence="HIGH",
        ),
        terminal_cap_lookup=_search_must_not_run,
        pipeline_version="v2",
    )

    assert result.market_cap_mm == 14_000.0
    assert result.cap_source == CAP_SOURCE_TERMINAL_LOCAL
    assert result.cap_band == "large_cap"
    assert result.cap_source_url == "https://exchange.example.com/daily-market-cap.csv"
    assert result.cap_effective_as_of_date == "2026-06-10"
    assert result.cap_confidence == "HIGH"
    assert result.price_used == 10.0
    assert result.price_as_of_date == "2026-06-11"
    assert result.price_source == "fixed_asof_exchange_close"
    assert result.price_source_url == "https://exchange.example.com/ZXLO/close"
    assert result.price_confidence == "HIGH"


def test_cached_20f_resolves_adr_ratio_as_authoritative_identity(tmp_path):
    filing_path = tmp_path / "issuer-20f.htm"
    filing_path.write_text(
        "<html><body><table><tr><td>American Depositary Shares</td>"
        "<td>Trading Symbol FADR</td></tr></table>"
        "Each ADS represents two Class A ordinary shares.</body></html>",
        encoding="utf-8",
    )
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE companies (ticker TEXT, cik TEXT);
        CREATE TABLE filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            cik TEXT,
            accession TEXT,
            form_type TEXT,
            filing_date TEXT,
            local_path TEXT,
            primary_doc_url TEXT,
            status TEXT
        );
        """
    )
    conn.execute("INSERT INTO companies(ticker, cik) VALUES ('FADR', '1234567')")
    conn.execute(
        """
        INSERT INTO filings(
            ticker, cik, accession, form_type, filing_date, local_path,
            primary_doc_url, status
        ) VALUES (
            'FADR', '1234567', '0001234567-26-000001', '20-F',
            '2026-04-01', ?, ?, 'downloaded'
        )
        """,
        (
            str(filing_path),
            "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/issuer-20f.htm",
        ),
    )
    conn.commit()
    conn.close()

    identity = resolve_security_identity(
        "FADR",
        as_of_date="2026-06-11",
        db_path=db_path,
    )

    assert identity.issuer_cik == "0001234567"
    assert identity.security_role == SECURITY_ROLE_ADR
    assert identity.is_adr is True
    assert identity.adr_ratio == 2.0
    assert identity.ratio_source_url == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/issuer-20f.htm"
    )
    assert identity.ratio_source_accession == "0001234567-26-000001"
    assert identity.ratio_security_symbol == "FADR"
    assert identity.identity_source == "sec_cached_annual_filing"
    assert identity.identity_as_of_date == "2026-04-01"
    assert identity.identity_confidence == "HIGH"


def test_cached_20f_amendment_without_identity_falls_back_to_full_annual(tmp_path):
    amendment_path = tmp_path / "issuer-20f-a.htm"
    amendment_path.write_text(
        "<html><body>Exhibit index amendment only.</body></html>",
        encoding="utf-8",
    )
    full_path = tmp_path / "issuer-20f.htm"
    full_path.write_text(
        "<html><body><table><tr><td>American Depositary Shares</td>"
        "<td>Trading Symbol FAMD</td></tr></table>"
        "Each ADS represents three ordinary shares.</body></html>",
        encoding="utf-8",
    )
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE companies (ticker TEXT, cik TEXT);
            CREATE TABLE filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, cik TEXT, accession TEXT, form_type TEXT,
                filing_date TEXT, local_path TEXT, primary_doc_url TEXT,
                status TEXT
            );
            INSERT INTO companies(ticker, cik) VALUES ('FAMD', '1234567');
            """
        )
        conn.execute(
            """INSERT INTO filings(
                   ticker, cik, accession, form_type, filing_date, local_path,
                   primary_doc_url, status
               ) VALUES (
                   'FAMD', '1234567', '0001234567-26-000001', '20-F',
                   '2026-03-01', ?, ?, 'downloaded'
               )""",
            (
                str(full_path),
                "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/full-20f.htm",
            ),
        )
        conn.execute(
            """INSERT INTO filings(
                   ticker, cik, accession, form_type, filing_date, local_path,
                   primary_doc_url, status
               ) VALUES (
                   'FAMD', '1234567', '0001234567-26-000002', '20-F/A',
                   '2026-04-01', ?, ?, 'downloaded'
               )""",
            (
                str(amendment_path),
                "https://www.sec.gov/Archives/edgar/data/1234567/"
                "000123456726000002/amended-20f.htm",
            ),
        )

    identity = resolve_security_identity(
        "FAMD",
        as_of_date="2026-06-11",
        db_path=db_path,
    )

    assert identity.security_role == SECURITY_ROLE_ADR
    assert identity.is_adr is True
    assert identity.adr_ratio == 3.0
    assert identity.ratio_source_url == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/full-20f.htm"
    )
    assert identity.identity_as_of_date == "2026-03-01"


def test_foreign_annual_adr_ratio_does_not_attach_to_unrelated_primary_symbol(
    tmp_path,
):
    filing_path = tmp_path / "multi-security-20f.htm"
    filing_path.write_text(
        "<html><body><table>"
        "<tr><td>Ordinary Shares</td><td>Trading Symbol PORD</td></tr>"
        "<tr><td>American Depositary Shares</td><td>Trading Symbol PADR</td></tr>"
        "</table>Each ADS represents two ordinary shares.</body></html>",
        encoding="utf-8",
    )
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE companies (ticker TEXT, cik TEXT);
            CREATE TABLE filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, cik TEXT, accession TEXT, form_type TEXT,
                filing_date TEXT, local_path TEXT, primary_doc_url TEXT,
                status TEXT
            );
            INSERT INTO companies VALUES ('PORD', '1234567');
            """
        )
        conn.execute(
            """
            INSERT INTO filings VALUES (
                NULL, 'PORD', '1234567', '0001234567-26-000001', '20-F',
                '2026-04-01', ?, ?, 'downloaded'
            )
            """,
            (
                str(filing_path),
                "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/issuer-20f.htm",
            ),
        )

    identity = resolve_security_identity(
        "PORD",
        as_of_date="2026-06-11",
        db_path=db_path,
    )

    assert identity.security_role == SECURITY_ROLE_SECONDARY_SECURITY
    assert identity.is_adr is None
    assert identity.adr_ratio is None
    assert identity.ratio_source_url is None


@pytest.mark.parametrize(
    ("ratio_source_url", "ratio_accession", "ratio_symbol"),
    [
        (
            "https://random-blog.example/Archives/edgar/data/1234567/"
            "000123456726000001/issuer-20f.htm",
            "0001234567-26-000001",
            "BIND",
        ),
        (
            "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000002/issuer-20f.htm",
            "0001234567-26-000001",
            "BIND",
        ),
        (
            "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/issuer-20f.htm",
            "0001234567-26-000001",
            "OTHER",
        ),
    ],
)
def test_adr_ratio_requires_trusted_host_accession_and_security_binding(
    tmp_path,
    ratio_source_url,
    ratio_accession,
    ratio_symbol,
):
    db_path = _seed_db(
        tmp_path,
        [("BIND", 2025, "FY", "2025-12-31", "shares_outstanding", 100.0)],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE companyfacts_facts SET source_url = ? WHERE ticker = 'BIND'",
            ("https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json",),
        )

    result = classify_market_cap_for_band_filter(
        "BIND",
        as_of_date="2026-06-11",
        asof_price=20.0,
        db_path=db_path,
        pipeline_version="v2",
        identity_evidence={
            "issuer_cik": "1234567",
            "is_adr": True,
            "adr_ratio": 2.0,
            "ratio_source_url": ratio_source_url,
            "ratio_source_accession": ratio_accession,
            "ratio_security_symbol": ratio_symbol,
            "identity_source": "issuer_filing",
            "identity_as_of_date": "2026-04-01",
            "identity_confidence": "HIGH",
        },
    )

    assert result.market_cap_mm is None
    assert result.detail == (
        "UNSAFE_ISSUER_SHARES_ADR_RATIO_MISSING:terminal_direct_cap_evidence_unavailable"
    )


def test_multi_class_submission_order_never_declares_first_symbol_primary(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    submissions = cache_dir / "submissions"
    submissions.mkdir(parents=True)
    submissions.joinpath("0001234567.json").write_text(
        '{"tickers":["DUALB","DUALA"]}',
        encoding="utf-8",
    )
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE companies (ticker TEXT, cik TEXT)")
    conn.execute("INSERT INTO companies(ticker, cik) VALUES ('DUALB', '1234567')")
    conn.commit()
    conn.close()

    identity = resolve_security_identity(
        "DUALB",
        as_of_date="2026-06-11",
        db_path=db_path,
        cfg=AppConfig(cache_dir=cache_dir),
    )

    assert identity.issuer_primary_ticker is None
    assert identity.security_role == SECURITY_ROLE_SECONDARY_CLASS
    assert identity.is_secondary_class is True


@pytest.mark.parametrize(
    ("ratio_url", "identity_as_of"),
    [
        ("file:///tmp/depositary-agreement", "2026-04-01"),
        ("https://www.sec.gov/Archives/example-20f.htm", "2026-07-01"),
    ],
)
def test_adr_ratio_rejects_non_http_or_future_authority(
    monkeypatch, tmp_path, ratio_url, identity_as_of
):
    def _unsafe_call(**kwargs):
        raise AssertionError("untrusted ADR ratio must not reach shares arithmetic")

    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        _unsafe_call,
    )
    result = classify_market_cap_for_band_filter(
        "ZXAD",
        as_of_date="2026-06-11",
        asof_price=20.0,
        db_path=tmp_path / "missing.db",
        pipeline_version="v2",
        identity_evidence={
            "issuer_cik": "1234567",
            "is_adr": True,
            "adr_ratio": 2.0,
            "ratio_source_url": ratio_url,
            "identity_source": "issuer_filing",
            "identity_as_of_date": identity_as_of,
            "identity_confidence": "HIGH",
        },
    )

    assert result.market_cap_mm is None
    assert result.detail == (
        "UNSAFE_ISSUER_SHARES_ADR_RATIO_MISSING:terminal_direct_cap_evidence_unavailable"
    )


def test_price_snapshot_without_basis_preserves_provenance_but_not_cap(
    monkeypatch,
    tmp_path,
):
    _strict_miss(monkeypatch)
    db_path = _seed_db(
        tmp_path,
        [("PRCE", 2025, "FY", "2025-12-31", "shares_outstanding", 2.0)],
    )
    result = classify_market_cap_for_band_filter(
        "PRCE",
        as_of_date="2026-06-11",
        db_path=db_path,
        price_lookup=lambda ticker, as_of: PriceSnapshot(
            ticker="PRCE",
            as_of_date="2026-06-10",
            price=11.0,
            source="exchange_cache",
            retrieved_at="2026-06-11T12:00:00Z",
            url="https://exchange.example.com/PRCE/close",
            confidence="HIGH",
        ),
        identity_evidence=_authoritative_primary_identity("PRCE"),
    )

    assert result.market_cap_mm is None
    assert result.cap_band is None
    assert result.in_band(0.0, 500.0) is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN
    assert result.price_used == 11.0
    assert result.price_as_of_date == "2026-06-10"
    assert result.price_source == "exchange_cache"
    assert result.price_source_url == "https://exchange.example.com/PRCE/close"
    assert result.price_confidence == "HIGH"
    assert result.cap_effective_as_of_date is None
    assert result.detail == "shares_known_price_missing"


def test_enb_primary_exchange_listing_is_not_confused_with_otc_preferred_aliases(
    tmp_path,
):
    cache_dir = tmp_path / "cache"
    submissions = cache_dir / "submissions"
    submissions.mkdir(parents=True)
    submissions.joinpath("0000895728.json").write_text(
        """{
          "tickers": ["ENB", "EBBNF", "EBBGF", "ENBFF", "ENBGF"],
          "exchanges": ["NYSE", "OTC", "OTC", "OTC", "OTC"]
        }""",
        encoding="utf-8",
    )
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE companies (ticker TEXT, cik TEXT);
            CREATE TABLE filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, cik TEXT, form_type TEXT, filing_date TEXT,
                local_path TEXT, primary_doc_url TEXT, status TEXT
            );
            INSERT INTO companies(ticker, cik) VALUES ('ENB', '895728');
            INSERT INTO filings(
                ticker, cik, form_type, filing_date, local_path, primary_doc_url
            ) VALUES (
                'ENB', '895728', '40-F', '2026-02-14', NULL,
                'https://www.sec.gov/Archives/edgar/data/895728/example-40f.htm'
            );
            """
        )

    identity = resolve_security_identity(
        "ENB",
        as_of_date="2026-06-11",
        db_path=db_path,
        cfg=AppConfig(cache_dir=cache_dir),
    )

    assert identity.issuer_primary_ticker == "ENB"
    assert identity.security_role == SECURITY_ROLE_PRIMARY
    assert identity.is_secondary_class is False
    assert identity.identity_source == "sec_submissions_exchange_binding"


def test_cached_single_ticker_20f_without_ratio_blocks_issuer_shares_arithmetic(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "cache"
    submissions = cache_dir / "submissions"
    submissions.mkdir(parents=True)
    submissions.joinpath("0001234567.json").write_text(
        '{"tickers":["FSEC"],"exchanges":["Nasdaq"]}',
        encoding="utf-8",
    )
    db_path = _seed_db(
        tmp_path,
        [("FSEC", 2025, "FY", "2025-12-31", "shares_outstanding", 100.0)],
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE companies (ticker TEXT, cik TEXT);
            CREATE TABLE filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, cik TEXT, form_type TEXT, filing_date TEXT,
                local_path TEXT, primary_doc_url TEXT, status TEXT
            );
            INSERT INTO companies(ticker, cik) VALUES ('FSEC', '1234567');
            INSERT INTO filings(
                ticker, cik, form_type, filing_date, local_path, primary_doc_url,
                status
            ) VALUES (
                'FSEC', '1234567', '20-F', '2026-03-31', NULL,
                'https://www.sec.gov/Archives/edgar/data/1234567/example-20f.htm',
                'downloaded'
            );
            """
        )

    def _unsafe_call(**kwargs):
        raise AssertionError("foreign security quote basis must be proven first")

    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        _unsafe_call,
    )
    result = classify_market_cap_for_band_filter(
        "FSEC",
        as_of_date="2026-06-11",
        asof_price=20.0,
        current_price=20.0,
        db_path=db_path,
        cfg=AppConfig(cache_dir=cache_dir),
        pipeline_version="v2",
    )

    assert result.market_cap_mm is None
    assert result.security_role == SECURITY_ROLE_SECONDARY_SECURITY
    assert result.identity_source == "sec_cached_foreign_annual_filing"
    assert result.detail == (
        "UNSAFE_ISSUER_SHARES_SECURITY_ROLE_UNRESOLVED:terminal_direct_cap_evidence_unavailable"
    )

    direct = classify_market_cap_for_band_filter(
        "FSEC",
        as_of_date="2026-06-11",
        asof_price=20.0,
        current_price=20.0,
        db_path=db_path,
        cfg=AppConfig(cache_dir=cache_dir),
        terminal_cap_lookup=lambda ticker, as_of_date, identity: TerminalCapEvidence(
            ticker="FSEC",
            market_cap_mm=14_000.0,
            source_kind="EXCHANGE",
            source_name="dated exchange issuer cap",
            source_url="https://www.nasdaq.com/market-activity/stocks/FSEC",
            as_of_date="2026-06-10",
            confidence="HIGH",
            issuer_cik="1234567",
        ),
        pipeline_version="v2",
    )

    assert direct.market_cap_mm == 14_000.0
    assert direct.cap_source == CAP_SOURCE_TERMINAL_EXCHANGE
    assert direct.security_role == SECURITY_ROLE_SECONDARY_SECURITY


def test_v1_rejects_downloaded_foreign_annual_as_primary_quote_identity(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE companies (ticker TEXT, cik TEXT);
            CREATE TABLE filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, cik TEXT, form_type TEXT, filing_date TEXT,
                local_path TEXT, primary_doc_url TEXT, status TEXT
            );
            INSERT INTO companies VALUES ('FDWN', '1234567');
            INSERT INTO filings VALUES (
                NULL, 'FDWN', '1234567', '20-F', '2026-03-31', NULL,
                'https://www.sec.gov/Archives/edgar/data/1234567/issuer-20f.htm',
                'downloaded'
            );
            """
        )
    monkeypatch.setattr(
        "app.valuation.shares.resolve_market_cap_from_price_asof",
        lambda **kwargs: (
            2_000.0,
            {
                "shares_outstanding": 100.0,
                "shares_asof_used": "2026-03-31",
                "shares_source_resolution": "legacy",
            },
        ),
    )

    v2_identity = resolve_security_identity(
        "FDWN",
        as_of_date="2026-06-11",
        db_path=db_path,
    )
    v1_result = classify_market_cap_for_band_filter(
        "FDWN",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance=_unadjusted_quote(
            20.0,
            ticker="FDWN",
            shares_as_of="2026-03-31",
        ),
        db_path=db_path,
        pipeline_version="v1",
    )

    assert v2_identity.security_role == SECURITY_ROLE_SECONDARY_SECURITY
    assert v2_identity.identity_source == "sec_cached_foreign_annual_filing"
    assert v1_result.market_cap_mm is None
    assert v1_result.cap_source == CAP_SOURCE_UNKNOWN
    assert v1_result.scope_status == "NEEDS_DATA"
    assert v1_result.scope_reason == "UNSAFE_ISSUER_SHARES_SECURITY_ROLE_UNRESOLVED"
    assert v1_result.security_role == SECURITY_ROLE_SECONDARY_SECURITY
    assert v1_result.identity_source == "sec_cached_foreign_annual_filing"


def test_v2_adr_uses_issuer_alias_vintage_shares_visible_as_of_scan(tmp_path):
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE companies (ticker TEXT, cik TEXT);
            INSERT INTO companies VALUES ('VADR', '1234567');
            INSERT INTO companies VALUES ('VPRI', '1234567');
            CREATE TABLE companyfacts_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, fiscal_year INTEGER, period_type TEXT,
                period_end TEXT, line_item TEXT, value REAL, units TEXT,
                source_url TEXT, fetched_at TEXT, filed_date TEXT, form TEXT,
                accession TEXT
            );
            CREATE TABLE companyfacts_vintages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT, fiscal_year INTEGER, period_type TEXT,
                period_end TEXT, line_item TEXT, value REAL, units TEXT,
                filed_date TEXT, form TEXT, accession TEXT, recorded_at TEXT,
                issuer_cik TEXT, source_url TEXT
            );
            INSERT INTO companyfacts_facts VALUES (
                NULL, 'VPRI', 2025, 'FY', '2025-12-31',
                'shares_outstanding', 130.0, 'shares_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json',
                '2026-08-01T00:00:00+00:00', '2026-08-01', '20-F/A',
                '0001234567-26-000002'
            );
            INSERT INTO companyfacts_vintages VALUES (
                NULL, 'VPRI', 2025, 'FY', '2025-12-31',
                'shares_outstanding', 100.0, 'shares_millions', '2026-03-01',
                '20-F', '0001234567-26-000001',
                '2026-03-01T00:00:00+00:00', '0001234567',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json'
            );
            INSERT INTO companyfacts_vintages VALUES (
                NULL, 'VADR', 2025, 'FY', '2025-12-31',
                'shares_outstanding', 999.0, 'shares_millions', '2026-02-01',
                '20-F', '0009999999-26-000001',
                '2026-02-01T00:00:00+00:00', '0009999999',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0009999999.json'
            );
            """
        )

    result = classify_market_cap_for_band_filter(
        "VADR",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance=_unadjusted_quote(
            20.0,
            ticker="VADR",
            issuer_cik="0001234567",
        ),
        db_path=db_path,
        pipeline_version="v2",
        identity_evidence={
            "issuer_cik": "1234567",
            "issuer_primary_ticker": "VPRI",
            "issuer_listed_tickers": ["VPRI", "VADR"],
            "is_adr": True,
            "adr_ratio": 2.0,
            "ratio_source_url": (
                "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/issuer-20f.htm"
            ),
            "ratio_source_accession": "0001234567-26-000001",
            "ratio_security_symbol": "VADR",
            "identity_source": "issuer_filing",
            "identity_as_of_date": "2026-03-01",
            "identity_confidence": "HIGH",
        },
    )

    assert result.market_cap_mm == 1_000.0
    assert result.shares_mm == 100.0
    assert result.shares_period_end == "2025-12-31"
    assert result.cap_source == CAP_SOURCE_ASOF_COMPANYFACTS
    assert "quote_to_issuer_share_ratio=2.0" in str(result.detail)


@pytest.mark.parametrize(
    ("currency", "expected_detail"),
    [
        (None, "CAP_PRICE_CURRENCY_UNRESOLVED"),
        ("EUR", "NON_USD_CAP_PRICE_UNSUPPORTED:EUR"),
    ],
)
def test_v2_shares_times_price_rejects_missing_or_non_usd_currency(
    tmp_path,
    currency,
    expected_detail,
):
    db_path = _seed_db(
        tmp_path,
        [("CURR", 2025, "FY", "2025-12-31", "shares_outstanding", 100.0)],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE companyfacts_facts SET source_url = ? WHERE ticker = 'CURR'",
            ("https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json",),
        )
    price = {
        "price": 20.0,
        "as_of_date": "2026-06-10",
        "source": "exchange_close",
    }
    if currency is not None:
        price["currency"] = currency

    result = classify_market_cap_for_band_filter(
        "CURR",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance=price,
        current_price=price,
        price_lookup=lambda ticker, as_of: None,
        db_path=db_path,
        pipeline_version="v2",
        identity_evidence={
            "issuer_cik": "1234567",
            "issuer_primary_ticker": "CURR",
            "security_role": "PRIMARY",
            "is_secondary_class": False,
            "is_adr": False,
            "identity_source_url": ("https://data.sec.gov/submissions/CIK0001234567.json"),
            "identity_as_of_date": "2026-04-01",
            "identity_confidence": "HIGH",
        },
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN
    assert result.price_used is None
    assert result.detail == expected_detail


def test_v2_shares_times_price_accepts_explicit_usd_currency(tmp_path):
    db_path = _seed_db(
        tmp_path,
        [("USDP", 2025, "FY", "2025-12-31", "shares_outstanding", 100.0)],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE companyfacts_facts SET source_url = ? WHERE ticker = 'USDP'",
            ("https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json",),
        )

    result = classify_market_cap_for_band_filter(
        "USDP",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance={
            **_unadjusted_quote(
                20.0,
                ticker="USDP",
                issuer_cik="0001234567",
            ),
            "as_of_date": "2026-06-10",
            "source": "exchange_close",
            "currency": "USD",
        },
        db_path=db_path,
        pipeline_version="v2",
        identity_evidence={
            "issuer_cik": "1234567",
            "issuer_primary_ticker": "USDP",
            "security_role": "PRIMARY",
            "is_secondary_class": False,
            "is_adr": False,
            "identity_source_url": ("https://data.sec.gov/submissions/CIK0001234567.json"),
            "identity_as_of_date": "2026-04-01",
            "identity_confidence": "HIGH",
        },
    )

    assert result.market_cap_mm == 2_000.0
    assert result.price_used == 20.0
    assert result.price_currency == "USD"
    assert result.cap_source == CAP_SOURCE_ASOF_COMPANYFACTS


def test_direct_issuer_cap_can_resolve_with_non_usd_security_quote(tmp_path):
    result = classify_market_cap_for_band_filter(
        "DCAP",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance={
            "as_of_date": "2026-06-10",
            "source": "foreign_exchange_close",
            "currency": "EUR",
        },
        db_path=tmp_path / "missing.db",
        identity_evidence={"issuer_cik": "1234567"},
        terminal_cap_lookup=lambda ticker, as_of, identity: TerminalCapEvidence(
            ticker="DCAP",
            market_cap_mm=25_000.0,
            source_kind="SEARCH",
            source_name="dated direct issuer cap",
            source_url="https://issuer.example/DCAP",
            as_of_date="2026-06-10",
            confidence="HIGH",
            issuer_cik="1234567",
        ),
        pipeline_version="v2",
    )

    assert result.market_cap_mm == 25_000.0
    assert result.cap_source == CAP_SOURCE_TERMINAL_SEARCH
    assert result.price_used == 20.0
    assert result.price_currency == "EUR"


def test_terminal_direct_cap_with_mismatched_issuer_cik_is_rejected(tmp_path):
    def lookup(ticker, as_of_date, identity):
        return TerminalCapEvidence(
            ticker="ZXCI",
            market_cap_mm=15_000.0,
            source_kind="SEARCH",
            source_name="issuer market data",
            source_url="https://exchange.example.com/ZXCI",
            as_of_date="2026-06-10",
            confidence="HIGH",
            issuer_cik="7654321",
        )

    result = classify_market_cap_for_band_filter(
        "ZXCI",
        as_of_date="2026-06-11",
        db_path=tmp_path / "missing.db",
        price_lookup=lambda ticker, as_of: None,
        identity_evidence={"issuer_cik": "1234567"},
        terminal_cap_lookup=lookup,
        pipeline_version="v2",
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN


def test_terminal_direct_cap_cannot_set_scope_without_resolved_issuer_cik(tmp_path):
    result = classify_market_cap_for_band_filter(
        "NOCI",
        as_of_date="2026-06-11",
        db_path=tmp_path / "missing.db",
        price_lookup=lambda ticker, as_of: None,
        terminal_cap_lookup=lambda ticker, as_of_date, identity: TerminalCapEvidence(
            ticker="NOCI",
            market_cap_mm=15_000.0,
            source_kind="SEARCH",
            source_name="issuer market data",
            source_url="https://issuer.example/NOCI",
            as_of_date="2026-06-10",
            confidence="HIGH",
        ),
        pipeline_version="v2",
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN


def test_v2_rejects_future_dated_price_from_every_price_lane(tmp_path):
    db_path = _seed_db(
        tmp_path,
        [("PITP", 2026, "Q1", "2026-03-31", "shares_outstanding", 100.0)],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE companyfacts_facts SET source_url = ? WHERE ticker = 'PITP'",
            ("https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json",),
        )
    identity = {
        "issuer_cik": "1234567",
        "issuer_primary_ticker": "PITP",
        "security_role": "PRIMARY",
        "is_adr": False,
        "is_secondary_class": False,
        "identity_source_url": "https://www.sec.gov/Archives/example.htm",
        "identity_as_of_date": "2026-04-01",
        "identity_confidence": "HIGH",
    }

    result = classify_market_cap_for_band_filter(
        "PITP",
        as_of_date="2026-06-11",
        asof_price=20.0,
        asof_price_provenance={"as_of_date": "2099-01-01", "source": "future"},
        current_price={"price": 20.0, "as_of_date": "2099-01-01"},
        price_lookup=lambda ticker, as_of: {
            "price": 20.0,
            "as_of_date": "2099-01-01",
            "source": "future_provider",
        },
        db_path=db_path,
        identity_evidence=identity,
        pipeline_version="v2",
    )

    assert result.market_cap_mm is None
    assert result.price_used is None
    assert result.detail == "shares_known_price_missing"


def test_v2_rejects_share_fact_filed_after_scan_date(tmp_path):
    db_path = _seed_db(
        tmp_path,
        [("PITF", 2025, "FY", "2025-12-31", "shares_outstanding", 100.0)],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE companyfacts_facts SET filed_date = '2099-01-01' WHERE ticker = 'PITF'"
        )

    result = classify_market_cap_for_band_filter(
        "PITF",
        as_of_date="2026-06-11",
        asof_price=20.0,
        db_path=db_path,
        identity_evidence={
            "issuer_primary_ticker": "PITF",
            "security_role": "PRIMARY",
            "is_adr": False,
            "is_secondary_class": False,
            "identity_source_url": "https://www.sec.gov/Archives/example.htm",
            "identity_as_of_date": "2026-04-01",
            "identity_confidence": "HIGH",
        },
        pipeline_version="v2",
    )

    assert result.market_cap_mm is None
    assert result.shares_mm is None
    assert result.detail == "no_companyfacts_shares"


def test_terminal_authority_label_cannot_spoof_url_host(tmp_path):
    result = classify_market_cap_for_band_filter(
        "SPUF",
        as_of_date="2026-06-11",
        db_path=tmp_path / "missing.db",
        terminal_cap_lookup=lambda ticker, as_of, identity: TerminalCapEvidence(
            ticker="SPUF",
            market_cap_mm=25_000.0,
            source_kind="SEC",
            source_name="caller_claimed_sec",
            source_url="https://random-blog.example/SPUF",
            as_of_date="2026-06-10",
            confidence="HIGH",
            issuer_cik="1234567",
        ),
        identity_evidence={"issuer_cik": "1234567"},
        pipeline_version="v2",
    )

    assert result.market_cap_mm is None
    assert result.cap_source == CAP_SOURCE_UNKNOWN


def test_terminal_cap_discards_future_embedded_price(tmp_path):
    result = classify_market_cap_for_band_filter(
        "DPRC",
        as_of_date="2026-06-11",
        db_path=tmp_path / "missing.db",
        price_lookup=lambda ticker, as_of: None,
        terminal_cap_lookup=lambda ticker, as_of, identity: TerminalCapEvidence(
            ticker="DPRC",
            market_cap_mm=25_000.0,
            source_kind="SEARCH",
            source_name="dated issuer market data",
            source_url="https://market-data.example/DPRC",
            as_of_date="2026-06-10",
            confidence="HIGH",
            issuer_cik="1234567",
            price_used=50.0,
            price_source="future_quote",
            price_as_of_date="2099-01-01",
            price_confidence="HIGH",
        ),
        identity_evidence={"issuer_cik": "1234567"},
        pipeline_version="v2",
    )

    assert result.market_cap_mm == 25_000.0
    assert result.price_used is None
    assert result.price_as_of_date is None


def test_submission_foreign_form_blocks_single_ticker_primary_inference(tmp_path):
    cache_dir = tmp_path / "cache"
    submissions = cache_dir / "submissions"
    submissions.mkdir(parents=True)
    submissions.joinpath("0001234567.json").write_text(
        """{
          "tickers": ["FONE"],
          "exchanges": ["Nasdaq"],
          "filings": {"recent": {
            "form": ["20-F"],
            "filingDate": ["2026-03-01"]
          }}
        }""",
        encoding="utf-8",
    )
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE companies (ticker TEXT, cik TEXT)")
        conn.execute("INSERT INTO companies VALUES ('FONE', '1234567')")

    identity = resolve_security_identity(
        "FONE",
        as_of_date="2026-06-11",
        db_path=db_path,
        cfg=AppConfig(cache_dir=cache_dir),
        identity_evidence={
            "security_role": "PRIMARY",
            "is_adr": False,
            "is_secondary_class": False,
        },
    )

    assert identity.issuer_primary_ticker is None
    assert identity.security_role == SECURITY_ROLE_SECONDARY_SECURITY
    assert identity.identity_source == "sec_submissions_foreign_security_unresolved"
