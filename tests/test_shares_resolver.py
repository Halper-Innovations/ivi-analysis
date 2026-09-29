from __future__ import annotations

import json
import sqlite3
from urllib.parse import urldefrag

import pytest

from app.db import init_db
from app.market.company_facts_provider import companyfacts_cache_path
from app.valuation.shares import (
    resolve_market_cap_from_price_asof,
    resolve_shares_asof,
    write_shares_coverage_for_run,
)
from tests.financial_integrity_helpers import (
    materialized_split_proof as _materialized_split_proof,
)


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SEC_USER_AGENT", "IVI tests qa@ivi.test")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _write_dossier(
    path,
    *,
    cfg=None,
    ticker: str,
    as_of_date: str,
    shares: float,
    shares_unit: str | None = "shares_millions",
    shares_filed_date: str | None = None,
    include_filed_date: bool = True,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    filed_date = (shares_filed_date or as_of_date) if include_filed_date else None
    normalized_shares = float(shares) / 1_000_000.0 if shares_unit == "shares" else float(shares)
    source_reference = (
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
        f"#{path.parent.parent.name}-{ticker}-shares"
    )
    payload = {
        "ticker": ticker,
        "run_id": path.parent.parent.name,
        "as_of_date": as_of_date,
        "raw_shares_source_value": float(shares),
        "raw_shares_source_unit": shares_unit,
        "raw_shares_outstanding_mm": normalized_shares,
        "shares_outstanding_mm": normalized_shares,
        "shares_basis": "UNADJUSTED",
        "split_adjustment_factor": 1.0,
        "shares_period_end": as_of_date,
        "shares_filed_date": filed_date,
        "shares_source": "SEC_COMPANYFACTS",
        "shares_source_reference": source_reference,
        "time_series": {
            "standardized_rows": [
                {"year": 2025, "shares_outstanding": normalized_shares},
            ],
            "standardized_row_traces": {
                "2025": {
                    "shares_outstanding": {
                        "derived_from": [
                            f"dossiers.{path.parent.parent.name}.{ticker}.shares_outstanding"
                        ],
                        "citations": [],
                        "value": float(shares),
                        "unit": shares_unit,
                        "period_end": as_of_date,
                        "filed_date": filed_date,
                        "source": "SEC_COMPANYFACTS",
                        "source_reference": source_reference,
                    }
                }
            },
        },
    }
    if shares_unit is not None:
        payload["shares_unit"] = shares_unit
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if cfg is not None and shares_unit is not None and filed_date is not None:
        _seed_companyfacts_share_backing(
            cfg,
            ticker=ticker,
            value=float(shares),
            unit=shares_unit,
            period_end=as_of_date,
            filed_date=filed_date,
            source_reference=source_reference,
            period_type=f"ARTIFACT_{path.parent.parent.name}",
        )


def _seed_companyfacts_share_backing(
    cfg,
    *,
    ticker: str,
    value: float,
    unit: str,
    period_end: str,
    filed_date: str,
    source_reference: str,
    period_type: str,
) -> None:
    with sqlite3.connect(cfg.db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO companies(ticker, cik, name, created_at)
            VALUES (?, '0000000001', ?, '2026-01-01T00:00:00+00:00')
            """,
            (ticker, ticker),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            )
            VALUES (?, ?, ?, ?, 'shares_outstanding', ?, ?, ?,
                    '2026-05-01T00:00:00+00:00', ?, '10-Q', ?)
            """,
            (
                ticker,
                int(period_end[:4]),
                period_type,
                period_end,
                value,
                unit,
                urldefrag(source_reference)[0],
                filed_date,
                f"artifact-{period_type}",
            ),
        )


def test_shares_resolver_prefers_current_run_fundamentals(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "shares_current_precedence"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fundamentals_AAA.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "run_id": run_id,
                "as_of_date": "2026-02-14",
                "shares_outstanding_latest": 111.0,
                "raw_shares_source_value": 111.0,
                "raw_shares_source_unit": "shares_millions",
                "raw_shares_outstanding_mm": 111.0,
                "shares_outstanding_mm": 111.0,
                "shares_unit": "shares_millions",
                "shares_basis": "UNADJUSTED",
                "split_adjustment_factor": 1.0,
                "shares_period_end": "2025-12-31",
                "shares_filed_date": "2026-02-01",
                "shares_source": "SEC_COMPANYFACTS",
                "shares_source_reference": (
                    "https://data.sec.gov/api/xbrl/companyfacts/"
                    "CIK0000000001.json#fundamentals-AAA-shares"
                ),
                "shares_trace": {
                    "value": 111.0,
                    "unit": "shares_millions",
                    "period_end": "2025-12-31",
                    "filed_date": "2026-02-01",
                    "source": "SEC_COMPANYFACTS",
                    "source_reference": (
                        "https://data.sec.gov/api/xbrl/companyfacts/"
                        "CIK0000000001.json#fundamentals-AAA-shares"
                    ),
                },
                "rows": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_dossier(
        cfg.dossiers_dir / "hist_older" / "AAA" / "dossier.json",
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        shares=222.0,
    )
    _seed_companyfacts_share_backing(
        cfg,
        ticker="AAA",
        value=111.0,
        unit="shares_millions",
        period_end="2025-12-31",
        filed_date="2026-02-01",
        source_reference=(
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json#fundamentals-AAA-shares"
        ),
        period_type="ARTIFACT_CURRENT_FUNDAMENTALS",
    )

    value, coverage = resolve_shares_asof(ticker="AAA", as_of_date="2026-02-14", run_id=run_id)
    assert value == 111.0
    assert coverage["shares_status"] == "OK"
    assert coverage["shares_reason_code"] == "OK"
    assert coverage["shares_source"] == "SEC_COMPANYFACTS"
    assert coverage["shares_source_resolution"] == "current_run_fundamentals"
    assert coverage["derived_from"]


def test_shares_resolver_historical_tie_break_is_deterministic(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "shares_historical_tie_break"
    (cfg.sectors_dir / run_id).mkdir(parents=True, exist_ok=True)
    _write_dossier(
        cfg.dossiers_dir / "hist_b" / "AAA" / "dossier.json",
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        shares=150.0,
    )
    _write_dossier(
        cfg.dossiers_dir / "hist_a" / "AAA" / "dossier.json",
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        shares=175.0,
    )

    value, coverage = resolve_shares_asof(ticker="AAA", as_of_date="2026-02-14", run_id=run_id)
    assert value == 175.0
    assert coverage["shares_status"] == "OK"
    assert coverage["shares_reason_code"] == "HISTORICAL_DOSSIER_HIT"
    assert coverage["shares_source"] == "SEC_COMPANYFACTS"
    assert coverage["shares_source_resolution"] == "historical_dossier"

    summary = write_shares_coverage_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        output_dir=cfg.sectors_dir / run_id,
        cfg=cfg,
    )
    assert summary["ticker_count"] == 2
    assert summary["reason_counts"]["HISTORICAL_DOSSIER_HIT"] == 1
    assert summary["reason_counts"]["NO_HISTORICAL_DOSSIER"] == 1


def test_artifact_shares_require_explicit_unit_and_normalize_raw_shares(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_dossier(
        cfg.dossiers_dir / "hist_raw_shares" / "AAA" / "dossier.json",
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-03-31",
        shares=20_000_000.0,
        shares_unit="shares",
        shares_filed_date="2026-05-01",
    )

    value, coverage = resolve_shares_asof(
        ticker="AAA",
        as_of_date="2026-07-21",
        run_id=None,
        cfg=cfg,
        companyfacts_cache_only=True,
    )

    assert value == 20.0
    assert coverage["shares_status"] == "OK"
    assert coverage["shares_unit"] == "shares_millions"
    assert coverage["shares_declared_source_unit"] == "shares"
    assert coverage["shares_asof_used"] == "2026-03-31"
    assert coverage["shares_filed_date"] == "2026-05-01"
    assert coverage["raw_shares_source_value"] == 20_000_000.0
    assert coverage["raw_shares_source_unit"] == "shares"
    assert coverage["raw_shares_outstanding_mm"] == 20.0

    market_cap, cap_coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-07-21",
        price=100.0,
        cfg=cfg,
        companyfacts_cache_only=True,
        require_split_lineage=True,
        quote_lineage={
            "as_of_date": "2026-07-21",
            "currency": "USD",
            "unit": "USD_per_share",
            "source": "fixture_quote",
            "source_url": "https://example.test/AAA/quote",
            "price_basis": "UNADJUSTED",
            "raw_price": 100.0,
            "split_adjustment_factor": 1.0,
            "split_effective_date": None,
            "no_intervening_split_proof": _materialized_split_proof(
                {
                    "ticker": "AAA",
                    "status": "PASS",
                    "period_start": "2026-03-31",
                    "period_end": "2026-07-21",
                    "verified_as_of": "2026-07-21",
                    "source": "fixture_actions",
                    "source_reference": "https://eodhd.com/api/splits/AAA",
                }
            ),
        },
    )

    assert market_cap == 2_000.0
    assert cap_coverage["market_cap_status"] == "OK"


def test_artifact_shares_reject_raw_unit_value_stored_in_millions_field(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    dossier_path = cfg.dossiers_dir / "hist_conflated_raw" / "AAA" / "dossier.json"
    _write_dossier(
        dossier_path,
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-03-31",
        shares=20_000_000.0,
        shares_unit="shares",
        shares_filed_date="2026-05-01",
    )
    payload = json.loads(dossier_path.read_text(encoding="utf-8"))
    payload["raw_shares_outstanding_mm"] = 20_000_000.0
    dossier_path.write_text(json.dumps(payload), encoding="utf-8")

    value, coverage = resolve_shares_asof(
        ticker="AAA",
        as_of_date="2026-07-21",
        run_id=None,
        cfg=cfg,
        companyfacts_cache_only=True,
    )

    assert value is None
    assert coverage["shares_reason_code"] == "SHARES_SPLIT_LINEAGE_CONFLICT"


@pytest.mark.parametrize(
    ("shares_unit", "shares_filed_date", "include_filed_date", "expected_reason"),
    [
        (None, "2026-05-01", True, "SHARES_UNIT_MISSING"),
        ("shares_millions", None, False, "SHARES_FILED_PROVENANCE_MISSING"),
        (
            "shares_millions",
            "2026-07-22",
            True,
            "SHARES_FILED_PROVENANCE_CONFLICT",
        ),
    ],
)
def test_artifact_shares_reject_missing_or_post_asof_provenance(
    monkeypatch,
    tmp_path,
    shares_unit,
    shares_filed_date,
    include_filed_date,
    expected_reason,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_dossier(
        cfg.dossiers_dir / "hist_invalid" / "AAA" / "dossier.json",
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-03-31",
        shares=20.0,
        shares_unit=shares_unit,
        shares_filed_date=shares_filed_date,
        include_filed_date=include_filed_date,
    )

    value, coverage = resolve_shares_asof(
        ticker="AAA",
        as_of_date="2026-07-21",
        run_id=None,
        cfg=cfg,
        companyfacts_cache_only=True,
    )

    assert value is None
    assert coverage["shares_reason_code"] == expected_reason
    assert coverage["shares_lineage_reason"] == expected_reason


def test_artifact_shares_reject_self_attested_non_sec_source_reference(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    dossier_path = cfg.dossiers_dir / "hist_untrusted" / "AAA" / "dossier.json"
    _write_dossier(
        dossier_path,
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-03-31",
        shares=20.0,
        shares_filed_date="2026-05-01",
    )
    payload = json.loads(dossier_path.read_text(encoding="utf-8"))
    payload["time_series"]["standardized_row_traces"]["2025"]["shares_outstanding"][
        "source_reference"
    ] = "fixture:self-attested-shares"
    dossier_path.write_text(json.dumps(payload), encoding="utf-8")

    value, coverage = resolve_shares_asof(
        ticker="AAA",
        as_of_date="2026-07-21",
        run_id=None,
        cfg=cfg,
        companyfacts_cache_only=True,
    )

    assert value is None
    assert coverage["shares_reason_code"] == "SHARES_SOURCE_REFERENCE_UNTRUSTED"


def test_artifact_shares_reject_invented_sec_reference_without_exact_backing(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    dossier_path = cfg.dossiers_dir / "hist_fake_sec" / "AAA" / "dossier.json"
    _write_dossier(
        dossier_path,
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-03-31",
        shares=20.0,
        shares_filed_date="2026-05-01",
    )
    payload = json.loads(dossier_path.read_text(encoding="utf-8"))
    payload["time_series"]["standardized_row_traces"]["2025"]["shares_outstanding"][
        "source_reference"
    ] = "https://data.sec.gov/api/xbrl/companyfacts/CIK9999999999.json#invented-share-observation"
    dossier_path.write_text(json.dumps(payload), encoding="utf-8")

    value, coverage = resolve_shares_asof(
        ticker="AAA",
        as_of_date="2026-07-21",
        run_id=None,
        cfg=cfg,
        companyfacts_cache_only=True,
    )

    assert value is None
    assert coverage["shares_reason_code"] == "SHARES_SOURCE_REFERENCE_UNVERIFIED"


def test_artifact_shares_reject_db_row_bound_to_wrong_issuer_cik(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    dossier_path = cfg.dossiers_dir / "hist_wrong_cik" / "AAA" / "dossier.json"
    _write_dossier(
        dossier_path,
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-03-31",
        shares=20.0,
        shares_filed_date="2026-05-01",
    )
    wrong_reference = (
        "https://data.sec.gov/api/xbrl/companyfacts/"
        "CIK9999999999.json#misattributed-share-observation"
    )
    payload = json.loads(dossier_path.read_text(encoding="utf-8"))
    payload["time_series"]["standardized_row_traces"]["2025"]["shares_outstanding"][
        "source_reference"
    ] = wrong_reference
    dossier_path.write_text(json.dumps(payload), encoding="utf-8")
    with sqlite3.connect(cfg.db_path) as conn:
        conn.execute(
            """
            UPDATE companyfacts_facts
            SET source_url = ?
            WHERE ticker = 'AAA'
              AND period_type = 'ARTIFACT_hist_wrong_cik'
            """,
            (urldefrag(wrong_reference)[0],),
        )

    value, coverage = resolve_shares_asof(
        ticker="AAA",
        as_of_date="2026-07-21",
        run_id=None,
        cfg=cfg,
        companyfacts_cache_only=True,
    )

    assert value is None
    assert coverage["shares_reason_code"] == "SHARES_SOURCE_REFERENCE_UNVERIFIED"


def test_market_cap_from_price_asof_uses_requested_historical_shares(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_dossier(
        cfg.dossiers_dir / "hist_older" / "AAA" / "dossier.json",
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-02-13",
        shares=100.0,
    )
    _write_dossier(
        cfg.dossiers_dir / "hist_newer" / "AAA" / "dossier.json",
        cfg=cfg,
        ticker="AAA",
        as_of_date="2026-03-13",
        shares=200.0,
    )

    older_value, older_coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-02-14",
        price=10.0,
        run_id=None,
    )
    newer_value, newer_coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-03-14",
        price=10.0,
        run_id=None,
    )

    assert older_value == 1000.0
    assert older_coverage["shares_asof_used"] == "2026-02-13"
    assert newer_value == 2000.0
    assert newer_coverage["shares_asof_used"] == "2026-03-13"


def test_market_cap_from_price_asof_rejects_circular_shares_source(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    def _fake_resolve_shares_asof(*, ticker, as_of_date, run_id):
        return 125.0, {
            "shares_status": "OK",
            "shares_reason_code": "DERIVED_FROM_MKTCAP_PRICE",
            "shares_source": "derived_market_cap_price",
            "shares_source_resolution": "derived_market_cap_price",
            "shares_asof_used": as_of_date,
            "derived_from": ["derived:shares=market_cap/current_price"],
        }

    monkeypatch.setattr("app.valuation.shares.resolve_shares_asof", _fake_resolve_shares_asof)

    value, coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-02-14",
        price=20.0,
        run_id=None,
    )

    assert value is None
    assert coverage["market_cap_status"] == "UNKNOWN"
    assert coverage["market_cap_reason_code"] == "CIRCULAR_SHARES_SOURCE"


def test_market_cap_normalizes_a_complete_four_for_one_split_lineage(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.valuation.facts.resolve_cik_for_ticker",
        lambda *_args, **_kwargs: "0000000001",
    )
    cache_path = companyfacts_cache_path("0000000001", cfg=cfg)
    cache_path.write_text(
        json.dumps(
            {
                "source_url": ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"),
                "companyfacts": {
                    "facts": {
                        "dei": {
                            "EntityCommonStockSharesOutstanding": {
                                "units": {
                                    "shares": [
                                        {
                                            "val": 20_000_000,
                                            "end": "2025-12-31",
                                            "filed": "2026-01-05",
                                            "form": "10-K",
                                            "accn": "0000000001-26-000001",
                                        }
                                    ]
                                }
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    raw_value, raw_coverage = resolve_shares_asof(
        ticker="AAA",
        as_of_date="2026-06-11",
        run_id=None,
    )
    assert raw_value == 20.0, raw_coverage
    assert raw_coverage["shares_lineage_status"] == "NEEDS_DATA"
    assert raw_coverage["shares_basis"] is None
    assert raw_coverage["shares_split_adjustment_factor"] is None
    assert raw_coverage["shares_source_url"] == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    )

    value, coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-06-11",
        price=25.0,
        quote_lineage={
            "as_of_date": "2026-06-11",
            "currency": "USD",
            "unit": "USD_per_share",
            "source": "adjusted_quote_fixture",
            "source_url": "https://example.test/AAA",
            "price_basis": "SPLIT_ADJUSTED",
            "raw_price": 100.0,
            "split_adjustment_factor": 4.0,
            "split_effective_date": "2026-01-15",
            "split_event": _materialized_split_proof(
                {
                    "ticker": "AAA",
                    "factor": 4.0,
                    "effective_date": "2026-01-15",
                    "filed_date": "2026-01-10",
                    "source": "issuer_split_filing",
                    "source_reference": (
                        "https://www.sec.gov/Archives/edgar/data/1/"
                        "000000000126000001/split-event.htm"
                    ),
                }
            ),
        },
        require_split_lineage=True,
    )

    assert value == 2000.0
    assert coverage["raw_shares_outstanding_mm"] == 20.0
    assert coverage["normalized_shares_outstanding_mm"] == 80.0
    assert coverage["shares_basis"] == "SPLIT_ADJUSTED"
    assert coverage["split_lineage_proof"]["factor"] == 4.0
    assert coverage["shares_source_url"] == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    )
    assert coverage["market_cap_status"] == "OK"


def test_market_cap_rejects_a_four_for_one_factor_mismatch(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.valuation.shares.resolve_shares_asof",
        lambda **_kwargs: (
            20.0,
            {
                "shares_status": "OK",
                "shares_reason_code": "OK",
                "shares_source": "sec_companyfacts",
                "shares_source_url": "https://example.test/companyfacts/AAA",
                "shares_source_resolution": "sec_companyfacts",
                "shares_asof_used": "2025-12-31",
                "shares_filed_date": "2026-02-01",
                "shares_lineage_status": "PASS",
                "raw_shares_source_value": 20_000_000.0,
                "raw_shares_source_unit": "shares",
                "raw_shares_outstanding_mm": 20.0,
                "normalized_shares_outstanding_mm": 20.0,
                "shares_unit": "shares_millions",
                "shares_basis": "UNADJUSTED",
                "shares_split_adjustment_factor": 1.0,
                "derived_from": ["companyfacts.test.shares"],
            },
        ),
    )

    value, coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-06-11",
        price=25.0,
        quote_lineage={
            "as_of_date": "2026-06-11",
            "currency": "USD",
            "unit": "USD_per_share",
            "source": "adjusted_quote_fixture",
            "source_url": "https://example.test/AAA",
            "price_basis": "SPLIT_ADJUSTED",
            "raw_price": 100.0,
            "split_adjustment_factor": 2.0,
            "split_effective_date": "2026-01-15",
            "split_event": _materialized_split_proof(
                {
                    "ticker": "AAA",
                    "factor": 2.0,
                    "effective_date": "2026-01-15",
                    "filed_date": "2026-01-10",
                    "source": "issuer_split_filing",
                    "source_reference": (
                        "https://www.sec.gov/Archives/edgar/data/1/"
                        "000000000126000001/split-event.htm"
                    ),
                }
            ),
        },
        require_split_lineage=True,
    )

    assert value is None
    assert coverage["market_cap_status"] == "UNKNOWN"
    assert coverage["market_cap_reason_code"] == "SPLIT_LINEAGE_CONFLICT"


def test_unadjusted_quote_requires_no_intervening_split_proof(
    monkeypatch,
    tmp_path,
):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.valuation.shares.resolve_shares_asof",
        lambda **_kwargs: (
            20.0,
            {
                "shares_status": "OK",
                "shares_reason_code": "COMPANYFACTS_HIT",
                "shares_source": "sec_companyfacts",
                "shares_source_url": "https://example.test/companyfacts/AAA",
                "shares_source_resolution": "sec_companyfacts",
                "shares_asof_used": "2025-12-31",
                "shares_filed_date": "2026-02-01",
                "raw_shares_source_value": 20_000_000.0,
                "raw_shares_source_unit": "shares",
                "raw_shares_outstanding_mm": 20.0,
                "shares_unit": "shares_millions",
                "derived_from": ["https://example.test/companyfacts/AAA"],
            },
        ),
    )
    quote = {
        "as_of_date": "2026-06-11",
        "currency": "USD",
        "unit": "USD_per_share",
        "source": "unadjusted_quote_fixture",
        "source_url": "https://example.test/AAA/quote",
        "price_basis": "UNADJUSTED",
        "raw_price": 100.0,
        "split_adjustment_factor": 1.0,
        "split_effective_date": None,
    }

    value, coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-06-11",
        price=100.0,
        quote_lineage=quote,
        require_split_lineage=True,
    )

    assert value is None
    assert coverage["market_cap_reason_code"] == "SPLIT_LINEAGE_CONFLICT"

    quote["no_intervening_split_proof"] = _materialized_split_proof(
        {
            "ticker": "AAA",
            "status": "PASS",
            "period_start": "2025-12-31",
            "period_end": "2026-06-11",
            "verified_as_of": "2026-06-10",
            "source": "issuer_actions_ledger",
            "source_reference": "https://eodhd.com/api/splits/AAA",
        }
    )
    value, coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-06-11",
        price=100.0,
        quote_lineage=quote,
        require_split_lineage=True,
    )

    assert value is None
    assert coverage["market_cap_reason_code"] == "SPLIT_LINEAGE_CONFLICT"

    quote["no_intervening_split_proof"] = _materialized_split_proof(
        {
            "ticker": "AAA",
            "status": "PASS",
            "period_start": "2025-12-31",
            "period_end": "2026-06-11",
            "verified_as_of": "2026-06-11",
            "source": "issuer_actions_ledger",
            "source_reference": "https://eodhd.com/api/splits/AAA",
        }
    )
    value, coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-06-11",
        price=100.0,
        quote_lineage=quote,
        require_split_lineage=True,
    )

    assert value == 2_000.0
    assert coverage["shares_basis"] == "UNADJUSTED"
    assert coverage["split_lineage_proof"]["status"] == "PASS"


def test_share_resolution_rejects_split_proof_bound_to_different_issuer(
    monkeypatch,
    tmp_path,
):
    _init_cfg(monkeypatch, tmp_path)
    companyfacts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    monkeypatch.setattr(
        "app.valuation.shares.resolve_shares_asof",
        lambda **_kwargs: (
            20.0,
            {
                "shares_status": "OK",
                "shares_reason_code": "COMPANYFACTS_HIT",
                "shares_source": "sec_companyfacts",
                "shares_source_url": companyfacts_url,
                "shares_source_resolution": "sec_companyfacts",
                "shares_asof_used": "2025-12-31",
                "shares_filed_date": "2026-02-01",
                "raw_shares_source_value": 20_000_000.0,
                "raw_shares_source_unit": "shares",
                "raw_shares_outstanding_mm": 20.0,
                "shares_unit": "shares_millions",
                "derived_from": [companyfacts_url],
            },
        ),
    )
    proof = _materialized_split_proof(
        {
            "ticker": "AAA",
            "status": "PASS",
            "period_start": "2025-12-31",
            "period_end": "2026-06-11",
            "verified_as_of": "2026-06-11",
            "issuer_cik": "0000000002",
            "source": "wrong_issuer_actions_ledger",
            "source_reference": "https://eodhd.com/api/splits/AAA",
        }
    )

    value, coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-06-11",
        price=100.0,
        quote_lineage={
            "as_of_date": "2026-06-11",
            "currency": "USD",
            "unit": "USD_per_share",
            "source": "unadjusted_quote_fixture",
            "source_url": "https://example.test/AAA/quote",
            "price_basis": "UNADJUSTED",
            "raw_price": 100.0,
            "split_adjustment_factor": 1.0,
            "split_effective_date": None,
            "no_intervening_split_proof": proof,
        },
        require_split_lineage=True,
    )

    assert value is None
    assert coverage["market_cap_reason_code"] == "SPLIT_LINEAGE_CONFLICT"


def test_same_day_four_for_one_split_cannot_self_attest_unadjusted_basis(
    monkeypatch,
    tmp_path,
):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.valuation.shares.resolve_shares_asof",
        lambda **_kwargs: (
            20.0,
            {
                "shares_status": "OK",
                "shares_reason_code": "COMPANYFACTS_HIT",
                "shares_source": "sec_companyfacts",
                "shares_source_url": "https://example.test/companyfacts/AAA",
                "shares_source_resolution": "sec_companyfacts",
                "shares_asof_used": "2026-01-15",
                "shares_filed_date": "2026-01-15",
                "raw_shares_source_value": 20_000_000.0,
                "raw_shares_source_unit": "shares",
                "raw_shares_outstanding_mm": 20.0,
                "shares_unit": "shares_millions",
                "derived_from": ["https://example.test/companyfacts/AAA"],
            },
        ),
    )

    value, coverage = resolve_market_cap_from_price_asof(
        ticker="AAA",
        as_of_date="2026-01-15",
        price=25.0,
        quote_lineage={
            "as_of_date": "2026-01-15",
            "currency": "USD",
            "unit": "USD_per_share",
            "source": "same_day_quote_fixture",
            "source_url": "https://example.test/AAA/quote",
            "price_basis": "UNADJUSTED",
            "raw_price": 25.0,
            "split_adjustment_factor": 1.0,
            "split_effective_date": None,
            "split_event": _materialized_split_proof(
                {
                    "ticker": "AAA",
                    "factor": 4.0,
                    "effective_date": "2026-01-15",
                    "filed_date": "2026-01-10",
                    "source": "issuer_split_filing",
                    "source_reference": (
                        "https://www.sec.gov/Archives/edgar/data/1/"
                        "000000000126000001/split-event.htm"
                    ),
                }
            ),
        },
        require_split_lineage=True,
    )

    assert value is None
    assert coverage["market_cap_reason_code"] == "SPLIT_LINEAGE_CONFLICT"
