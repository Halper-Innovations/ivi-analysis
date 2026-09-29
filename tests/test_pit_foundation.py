"""PIT foundation: provenance through the normalizer, vintage archival,
flag-gated filed_date as-of reads, cache backfill."""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.config import get_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def _env(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    from app.db import init_db

    init_db()
    return db_path


def _raw_payload(*, filed="2026-02-15", val=1_000_000.0, accn="0001-26-000123", form="10-K"):
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "val": val,
                                "fy": 2025,
                                "fp": "FY",
                                "form": form,
                                "filed": filed,
                                "accn": accn,
                            }
                        ]
                    }
                }
            }
        }
    }


def test_normalizer_carries_filing_provenance():
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    rows = normalize_annual_facts_from_raw(_raw_payload(), cik="0000000001", years_back=10)
    revenue = [row for row in rows if row["line_item"] == "revenue"]
    assert revenue, rows
    row = revenue[0]
    assert row["filed_date"] == "2026-02-15"
    assert row["form"] == "10-K"
    assert row["accession"] == "0001-26-000123"


def _seed_fact(db_path, *, filed_date=None, value=1.0):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
        "line_item, value, units, source_url, fetched_at, filed_date) "
        "VALUES ('AAA', 2025, 'FY', '2025-12-31', 'revenue', ?, 'USD_millions', 'u', "
        "'2026-01-01T00:00:00+00:00', ?)",
        (value, filed_date),
    )
    conn.commit()
    conn.close()


def test_pit_flag_gates_asof_reads(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.db import connect
    from app.util.financial_data_access import companyfacts_rows

    _seed_fact(db_path, filed_date="2026-02-15", value=1.0)
    conn = connect(db_path)

    # Flag OFF (default): period_end visibility — the fact leaks before filing.
    rows = companyfacts_rows(
        conn, "AAA", columns=("line_item", "value"), as_of_date="2026-01-15"
    )
    assert len(rows) == 1

    # Flag ON: invisible before filed_date, visible after.
    monkeypatch.setenv("VOE_PIT_FILED_ASOF", "true")
    rows = companyfacts_rows(
        conn, "AAA", columns=("line_item", "value"), as_of_date="2026-01-15"
    )
    assert rows == []
    rows = companyfacts_rows(
        conn, "AAA", columns=("line_item", "value"), as_of_date="2026-02-15"
    )
    assert len(rows) == 1
    conn.close()


def test_pit_flag_null_filed_rows_use_90d_lag(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.db import connect
    from app.util.financial_data_access import companyfacts_rows

    _seed_fact(db_path, filed_date=None, value=1.0)
    monkeypatch.setenv("VOE_PIT_FILED_ASOF", "true")
    conn = connect(db_path)
    # period_end 2025-12-31: invisible at +80d, visible at +95d.
    assert (
        companyfacts_rows(conn, "AAA", columns=("value",), as_of_date="2026-03-21")
        == []
    )
    assert (
        len(companyfacts_rows(conn, "AAA", columns=("value",), as_of_date="2026-04-05"))
        == 1
    )
    conn.close()


def test_pit_backfill_stamps_rows_and_seeds_vintages(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    cache_dir = tmp_path / "cache" / "companyfacts"
    cache_dir.mkdir(parents=True)
    (cache_dir / "0000000001.json").write_text(json.dumps(_raw_payload()))
    # Row exists with the SAME value but no provenance (pre-A6 ingest).
    _seed_fact(db_path, filed_date=None, value=1.0)  # 1_000_000 USD -> 1.0 millions

    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda refresh_if_missing=False: {"AAA": "1"},
    )
    from app.ops.pit_backfill import run_pit_backfill

    counts = run_pit_backfill(db_path=db_path)
    assert counts["payloads_matched"] == 1
    assert counts["rows_stamped"] == 1
    assert counts["vintages_added"] >= 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT filed_date, form, accession FROM companyfacts_facts WHERE ticker='AAA'"
    ).fetchone()
    vintage = conn.execute(
        "SELECT filed_date, value FROM companyfacts_vintages WHERE ticker='AAA' AND line_item='revenue'"
    ).fetchone()
    conn.close()
    assert row["filed_date"] == "2026-02-15"
    assert row["accession"] == "0001-26-000123"
    assert vintage["filed_date"] == "2026-02-15"

    # Idempotent: a second run stamps nothing new and dedupes vintages.
    counts2 = run_pit_backfill(db_path=db_path)
    assert counts2["rows_stamped"] == 0
    assert counts2["vintages_added"] == 0


def test_pit_backfill_counts_value_mismatch_never_guesses(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    cache_dir = tmp_path / "cache" / "companyfacts"
    cache_dir.mkdir(parents=True)
    (cache_dir / "0000000001.json").write_text(json.dumps(_raw_payload(val=2_000_000.0)))
    _seed_fact(db_path, filed_date=None, value=1.0)  # stored 1.0mm vs cache 2.0mm

    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda refresh_if_missing=False: {"AAA": "1"},
    )
    from app.ops.pit_backfill import run_pit_backfill

    counts = run_pit_backfill(db_path=db_path)
    assert counts["rows_stamped"] == 0
    assert counts["value_mismatches"] == 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT filed_date FROM companyfacts_facts WHERE ticker='AAA'"
    ).fetchone()
    conn.close()
    assert not (row["filed_date"] or "")
