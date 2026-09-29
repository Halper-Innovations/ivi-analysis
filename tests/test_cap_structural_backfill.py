from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.db import init_db
from app.watchlist.cap_structural_backfill import backfill_watchlist_cap_and_structural
from app.watchlist.contract import WatchlistEntry
from app.watchlist.store import add_or_update, add_price_snapshot, get_latest
from tests.financial_integrity_helpers import materialized_no_split_proof


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())
    return db_path


def _add_companyfacts(db_path: Path, rows: list[tuple[str, int, str, str, str, float]]) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
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
    )
    ticker_ciks = {
        ticker: f"{index:010d}"
        for index, ticker in enumerate(
            sorted({str(row[0]).upper() for row in rows}),
            start=1,
        )
    }
    for ticker, fiscal_year, period_type, period_end, line_item, value in rows:
        unit = "shares_millions" if line_item == "shares_outstanding" else "USD_millions"
        conn.execute(
            "INSERT INTO companyfacts_facts (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at, filed_date, form, accession)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, '2026-06-11T00:00:00+00:00', '2026-06-01', '10-K', 'test-accession')",
            (
                ticker,
                fiscal_year,
                period_type,
                period_end,
                line_item,
                value,
                unit,
                (
                    "https://data.sec.gov/api/xbrl/companyfacts/"
                    f"CIK{ticker_ciks[str(ticker).upper()]}.json"
                ),
            ),
        )
    for ticker, cik in ticker_ciks.items():
        conn.execute(
            """
            INSERT OR REPLACE INTO companies(ticker, cik, name, created_at)
            VALUES (?, ?, ?, '2026-06-01T00:00:00+00:00')
            """,
            (ticker, cik, f"{ticker} fixture"),
        )
    conn.commit()
    conn.close()
    from app.config import get_config

    submissions_dir = get_config().cache_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    for ticker, cik in ticker_ciks.items():
        (submissions_dir / f"{cik}.json").write_text(
            json.dumps(
                {
                    "tickers": [ticker],
                    "exchanges": ["NYSE"],
                    "filings": {
                        "recent": {
                            "form": ["10-Q"],
                            "filingDate": ["2026-06-01"],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )


def _entry(ticker: str, *, status: str = "ACTIVE", price: float = 100.0) -> WatchlistEntry:
    return WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade="WATCHLIST_ONLY",
        confidence="MODERATE",
        conviction_source="company_autonomy",
        valuation_anchor_method="DCF",
        valuation_anchor_value=price * 1.2,
        buy_price_target=price * 0.8,
        current_price_at_addition=price,
        source_run_id="run_backfill_test",
        source_sector="energy",
        added_at="2026-06-01T12:00:00+00:00",
    )


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
    shares_as_of: str = "2026-03-31",
) -> dict[str, object]:
    return {
        "price": price,
        "raw_price": price,
        "price_basis": "UNADJUSTED",
        "split_adjustment_factor": 1.0,
        "as_of_date": "2026-06-10",
        "currency": "USD",
        "source": "fixture_quote",
        "source_url": "https://example.test/quote",
        "no_intervening_split_proof": materialized_no_split_proof(
            ticker=ticker,
            period_start=shares_as_of,
            period_end="2026-06-10",
            issuer_cik=issuer_cik,
        ),
    }


def test_backfill_quarantines_gtec_and_resolves_vnom(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _strict_miss(monkeypatch)

    gtec_id = add_or_update(_entry("GTEC", price=0.66), db_path=db_path)
    vnom_id = add_or_update(_entry("VNOM", price=155.0), db_path=db_path)
    add_price_snapshot(
        gtec_id, price=0.66, checked_at="2026-06-10T12:00:00Z", source="test", db_path=db_path
    )
    add_price_snapshot(
        vnom_id, price=155.0, checked_at="2026-06-10T12:00:00Z", source="test", db_path=db_path
    )

    _add_companyfacts(
        db_path,
        [
            # GTEC: computable $13.6M cap + NI/OCF divergence + penny price.
            ("GTEC", 2026, "Q1", "2026-03-31", "shares_outstanding", 20.606),
            ("GTEC", 2025, "FY", "2025-12-31", "net_income", 2.0),
            ("GTEC", 2025, "FY", "2025-12-31", "cfo", -5.0),
            # VNOM: resolves to a real (mid-band) cap via stale shares.
            ("VNOM", 2026, "Q1", "2026-03-31", "shares_outstanding", 58.0),
        ],
    )

    report = backfill_watchlist_cap_and_structural(
        db_path=db_path,
        as_of_date="2026-06-11",
        price_lookup=lambda ticker, _as_of: _unadjusted_quote(
            0.66 if ticker == "GTEC" else 155.0,
            ticker=ticker,
            issuer_cik="0000000001" if ticker == "GTEC" else "0000000002",
        ),
    )

    gtec = get_latest("GTEC", db_path=db_path)
    assert gtec is not None
    assert gtec.status == "QUARANTINE"
    assert gtec.status_reason == (
        "QUARANTINE_STRUCTURAL:PENNY_FLOOR;"
        "QUARANTINE_STRUCTURAL:NANO_FLOOR;"
        "QUARANTINE_STRUCTURAL:EARNINGS_QUALITY_DIVERGENCE"
    )
    assert gtec.cap_source == "stale_shares"
    assert round(gtec.market_cap_mm, 5) == 13.59996
    assert gtec.cap_band == "micro"

    vnom = get_latest("VNOM", db_path=db_path)
    assert vnom is not None
    assert vnom.status == "ACTIVE"
    assert vnom.cap_source == "stale_shares"
    assert vnom.market_cap_mm == 8990.0
    assert vnom.cap_band == "mid"

    assert report["counts"]["rows_examined"] == 2
    assert report["counts"]["rows_changed"] == 2
    assert report["counts"]["tickers_quarantined_structural"] == 1
    assert report["counts"]["cap_source"] == {"stale_shares": 2}
    by_ticker = {row["ticker"]: row for row in report["changed_rows"]}
    assert by_ticker["GTEC"]["new_status"] == "QUARANTINE"
    assert by_ticker["GTEC"]["structural_codes"] == [
        "PENNY_FLOOR",
        "NANO_FLOOR",
        "EARNINGS_QUALITY_DIVERGENCE",
    ]
    assert by_ticker["VNOM"]["new_status"] == "ACTIVE"
    assert by_ticker["VNOM"]["changes"]["cap_band"]["new"] == "mid"


def test_backfill_dry_run_writes_nothing(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _strict_miss(monkeypatch)

    gtec_id = add_or_update(_entry("GTEC", price=0.66), db_path=db_path)
    add_price_snapshot(
        gtec_id, price=0.66, checked_at="2026-06-10T12:00:00Z", source="test", db_path=db_path
    )

    report = backfill_watchlist_cap_and_structural(
        db_path=db_path,
        as_of_date="2026-06-11",
        dry_run=True,
        price_lookup=lambda ticker, _as_of: _unadjusted_quote(0.66, ticker=ticker),
    )
    assert report["dry_run"] is True
    assert report["counts"]["rows_changed"] == 1

    row = get_latest("GTEC", db_path=db_path)
    assert row is not None
    assert row.status == "ACTIVE"
    assert row.cap_source is None


def test_census_cap_resolution_counts(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _strict_miss(monkeypatch)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sector_inference (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            inferred_sector TEXT,
            score REAL NOT NULL DEFAULT 0,
            derived_from TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date)
        );
        CREATE TABLE IF NOT EXISTS valuations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            method TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            valuation_writer_version TEXT,
            quality_gate_verdict TEXT,
            confidence_class TEXT,
            gate_reason_codes TEXT,
            valuation_headwinds TEXT,
            valuation_supports TEXT,
            source_run_id TEXT,
            source_artifact_path TEXT,
            source_artifact_sha256 TEXT,
            financial_integrity_fingerprint TEXT,
            UNIQUE(ticker, as_of_date, method)
        );
        """
    )
    for ticker in ("RSLV", "NPRC"):
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES (?, '2026-06-01', 'energy', '2026-06-01T00:00:00+00:00')",
            (ticker,),
        )
    conn.commit()
    conn.close()
    _add_companyfacts(
        db_path,
        [("RSLV", 2026, "Q1", "2026-03-31", "shares_outstanding", 10.0)],
    )

    from app.autonomous.cap_census import census_cap_resolution

    report = census_cap_resolution(
        as_of_date="2026-06-11",
        db_path=db_path,
        price_lookup=lambda ticker, _as_of: (
            _unadjusted_quote(12.0, ticker=ticker) if ticker == "RSLV" else None
        ),
    )
    assert report["counts"]["census_tickers"] == 2
    assert report["counts"]["strict_resolved"] == 0
    assert report["counts"]["stale_shares_resolved"] == 1
    assert report["counts"]["still_unknown"] == 1
    assert report["counts"]["stale_shares_by_band"] == {"micro": 1}
    assert report["counts"]["unknown_reasons"] == {
        (
            "UNSAFE_ISSUER_SHARES_SECURITY_ROLE_UNRESOLVED:terminal_direct_cap_evidence_unavailable"
        ): 1
    }
    assert report["stale_shares_resolved"] == [
        {
            "ticker": "RSLV",
            "market_cap_mm": 120.0,
            "cap_band": "micro",
            "detail": "shares_period_end=2026-03-31:price_origin=provider",
        }
    ]
    assert report["still_unknown"] == [
        {
            "ticker": "NPRC",
            "reason": (
                "UNSAFE_ISSUER_SHARES_SECURITY_ROLE_UNRESOLVED:"
                "terminal_direct_cap_evidence_unavailable"
            ),
        }
    ]
