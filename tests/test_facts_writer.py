from __future__ import annotations

import json
from unittest.mock import patch

from app.db import get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _gc
    _gc.cache_clear()
    cfg = _gc()
    init_db(cfg)
    return cfg


_FAKE_FACTS = [
    {"line_item": "revenue", "fiscal_year": 2024, "period_end": "2024-09-28",
     "value": 391035.0, "units": "USD_millions", "source_url": "https://data.sec.gov"},
    {"line_item": "net_income", "fiscal_year": 2024, "period_end": "2024-09-28",
     "value": 93736.0, "units": "USD_millions", "source_url": "https://data.sec.gov"},
]


def test_ensure_facts_writes_rows(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", return_value=_FAKE_FACTS):
        from app.ingest.facts_writer import ensure_facts
        ensure_facts("AAPL", years_back=2)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT line_item, value FROM companyfacts_facts WHERE ticker='AAPL'"
        ).fetchall()
    assert len(rows) == 2
    by_item = {r["line_item"]: r["value"] for r in rows}
    assert by_item["revenue"] == 391035.0


def test_ensure_facts_no_duplicates(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", return_value=_FAKE_FACTS):
        from app.ingest.facts_writer import ensure_facts
        ensure_facts("AAPL", years_back=2)
        ensure_facts("AAPL", years_back=2)
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) as n FROM companyfacts_facts WHERE ticker='AAPL'"
        ).fetchone()["n"]
    assert count == 2  # not 4


def test_ensure_facts_skips_if_fresh(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    fetch_calls = []
    def fake_fetch(cik, years_back):
        fetch_calls.append(cik)
        return _FAKE_FACTS

    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", side_effect=fake_fetch):
        from app.ingest.facts_writer import ensure_facts
        ensure_facts("AAPL", years_back=2)   # first: fetches
        ensure_facts("AAPL", years_back=2)   # second: should skip (data is fresh)
    assert len(fetch_calls) == 1


def test_ensure_facts_fetches_annual_when_only_quarterly_rows_are_fresh(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            )
            VALUES('AAPL', 2025, 'Q1', '2024-12-28', 'revenue', 94930.0, 'USD_millions', 'https://data.sec.gov', ?)
            """,
            (now,),
        )

    fetch_calls = []

    def fake_fetch(cik, years_back):
        fetch_calls.append(cik)
        return _FAKE_FACTS

    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", side_effect=fake_fetch):
        from app.ingest.facts_writer import ensure_facts

        ensure_facts("AAPL", years_back=2)

    assert fetch_calls == ["0000320193"]
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT period_type, value
            FROM companyfacts_facts
            WHERE ticker='AAPL' AND fiscal_year=2024 AND line_item='revenue'
            """
        ).fetchone()
    assert row["period_type"] == "FY"
    assert row["value"] == 391035.0


def test_ensure_facts_uses_local_companyfacts_cache(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    cache_path = cfg.cache_dir / "companyfacts" / "0000320193.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "companyfacts": {
                    "facts": {
                        "us-gaap": {
                            "Revenues": {
                                "units": {
                                    "USD": [
                                        {
                                            "form": "10-K",
                                            "fy": 2024,
                                            "start": "2023-09-30",
                                            "end": "2024-09-28",
                                            "val": 391035000000,
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", side_effect=AssertionError("network fetch should not run")):
        from app.ingest.facts_writer import ensure_facts

        ensure_facts("AAPL", years_back=2)

    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM companyfacts_facts WHERE ticker='AAPL' AND fiscal_year=2024 AND line_item='revenue'"
        ).fetchone()
    assert row is not None
    assert row["value"] == 391035.0


# ---------------------------------------------------------------------------
# ensure_quarterly_facts tests
# ---------------------------------------------------------------------------

_FAKE_QUARTERLY_FACTS = [
    {"line_item": "revenue", "fiscal_year": 2025, "period_type": "Q1",
     "period_end": "2024-12-28", "value": 94930.0, "units": "USD_millions",
     "source_url": "https://data.sec.gov"},
    {"line_item": "revenue", "fiscal_year": 2025, "period_type": "Q2",
     "period_end": "2025-03-29", "value": 95000.0, "units": "USD_millions",
     "source_url": "https://data.sec.gov"},
]


def test_ensure_quarterly_facts_writes_rows(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_quarterly_facts", return_value=_FAKE_QUARTERLY_FACTS):
        from app.ingest.facts_writer import ensure_quarterly_facts
        ensure_quarterly_facts("AAPL", years_back=2)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT line_item, value, period_type FROM companyfacts_facts "
            "WHERE ticker='AAPL' AND period_type != 'FY'"
        ).fetchall()
    assert len(rows) == 2
    periods = {r["period_type"] for r in rows}
    assert periods == {"Q1", "Q2"}


def test_quarterly_and_annual_coexist(monkeypatch, tmp_path):
    """Annual and quarterly rows for the same fiscal year + line item can both exist."""
    _init_temp_db(monkeypatch, tmp_path)
    annual = [{"line_item": "revenue", "fiscal_year": 2025, "period_type": "FY",
               "period_end": "2025-09-27", "value": 400000.0, "units": "USD_millions",
               "source_url": "https://data.sec.gov"}]
    quarterly = [{"line_item": "revenue", "fiscal_year": 2025, "period_type": "Q1",
                  "period_end": "2024-12-28", "value": 94930.0, "units": "USD_millions",
                  "source_url": "https://data.sec.gov"}]
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", return_value=annual), \
         patch("app.ingest.facts_writer.fetch_quarterly_facts", return_value=quarterly):
        from app.ingest.facts_writer import ensure_facts, ensure_quarterly_facts
        ensure_facts("AAPL", years_back=2)
        ensure_quarterly_facts("AAPL", years_back=2)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT period_type, value FROM companyfacts_facts "
            "WHERE ticker='AAPL' AND line_item='revenue' AND fiscal_year=2025"
        ).fetchall()
    assert len(rows) == 2
    by_period = {r["period_type"]: r["value"] for r in rows}
    assert by_period["FY"] == 400000.0
    assert by_period["Q1"] == 94930.0


def test_ensure_annual_facts_writes_period_type_fy(monkeypatch, tmp_path):
    """ensure_facts (annual) should write period_type='FY' to the DB."""
    _init_temp_db(monkeypatch, tmp_path)
    annual_with_pt = [
        {"line_item": "revenue", "fiscal_year": 2024, "period_type": "FY",
         "period_end": "2024-09-28", "value": 391035.0, "units": "USD_millions",
         "source_url": "https://data.sec.gov"},
    ]
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", return_value=annual_with_pt):
        from app.ingest.facts_writer import ensure_facts
        ensure_facts("AAPL", years_back=2)
    with get_db() as conn:
        row = conn.execute(
            "SELECT period_type FROM companyfacts_facts WHERE ticker='AAPL' LIMIT 1"
        ).fetchone()
    assert row["period_type"] == "FY"


# ── stale-row purge on refresh ────────────────────────────────────────────────


def _seed(conn, ticker, fy, period_type, line_item, value=1.0):
    conn.execute(
        """INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end,
               line_item, value, units, source_url, fetched_at)
           VALUES(?, ?, ?, ?, ?, ?, 'USD_millions', 'u', '2000-01-01T00:00:00+00:00')""",
        (ticker, fy, period_type, f"{fy}-12-31", line_item, value),
    )


def _keys(ticker):
    with get_db() as conn:
        return {
            (r["fiscal_year"], r["period_type"], r["line_item"])
            for r in conn.execute(
                "SELECT fiscal_year, period_type, line_item FROM companyfacts_facts WHERE ticker=?",
                (ticker,),
            )
        }


def test_annual_refresh_removes_rows_the_normalizer_no_longer_emits(monkeypatch, tmp_path):
    """A guard-refused share count / debt total left over from an older ingest must not
    survive a refresh; other tickers, quarterly rows, line items outside the normalizer's
    map and years before the refresh window are untouched."""
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed(conn, "AAPL", 2024, "FY", "shares_outstanding")  # now refused
        _seed(conn, "AAPL", 2024, "FY", "total_debt")  # now UNKNOWN (not emitted)
        _seed(conn, "AAPL", 2024, "FY", "revenue", 5.0)  # re-emitted, replaced
        _seed(conn, "AAPL", 2024, "Q1", "shares_outstanding")  # quarterly: not this refresh
        _seed(conn, "AAPL", 2024, "FY", "custom_metric")  # not a normalizer line item
        _seed(conn, "AAPL", 1990, "FY", "shares_outstanding")  # before the window
        _seed(conn, "MSFT", 2024, "FY", "shares_outstanding")  # other ticker
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", return_value=_FAKE_FACTS):
        from app.ingest.facts_writer import ensure_facts
        ensure_facts("AAPL", years_back=2)
    assert _keys("AAPL") == {
        (2024, "FY", "revenue"),
        (2024, "FY", "net_income"),
        (2024, "Q1", "shares_outstanding"),
        (2024, "FY", "custom_metric"),
        (1990, "FY", "shares_outstanding"),
    }
    assert _keys("MSFT") == {(2024, "FY", "shares_outstanding")}
    with get_db() as conn:
        v = conn.execute(
            "SELECT value FROM companyfacts_facts WHERE ticker='AAPL' AND line_item='revenue'"
        ).fetchone()["value"]
    assert v == 391035.0


def test_quarterly_refresh_removes_relabelled_quarter_rows(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    q = [{"line_item": "revenue", "fiscal_year": 2025, "period_type": "Q2",
          "period_end": "2025-03-29", "value": 95359.0, "units": "USD_millions",
          "source_url": "u"}]
    with get_db() as conn:
        _seed(conn, "AAPL", 2025, "Q1", "revenue")  # old mislabel, no longer emitted
        _seed(conn, "AAPL", 2025, "FY", "revenue")  # annual row untouched
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_quarterly_facts", return_value=q), \
         patch("app.ingest.facts_writer._quarterly_facts_from_local_cache", return_value=[]):
        from app.ingest.facts_writer import ensure_quarterly_facts
        ensure_quarterly_facts("AAPL", years_back=2)
    assert _keys("AAPL") == {(2025, "Q2", "revenue"), (2025, "FY", "revenue")}


def test_empty_fetch_never_wipes_existing_rows(monkeypatch, tmp_path):
    """No emitted facts is indistinguishable from a failed fetch: keep what is stored."""
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed(conn, "AAPL", 2024, "FY", "revenue")
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", return_value=[]), \
         patch("app.ingest.facts_writer._facts_from_local_cache", return_value=[]):
        from app.ingest.facts_writer import ensure_facts
        ensure_facts("AAPL", years_back=2)
    assert _keys("AAPL") == {(2024, "FY", "revenue")}


def test_refresh_is_atomic_when_an_insert_fails(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed(conn, "AAPL", 2024, "FY", "shares_outstanding")
    bad = _FAKE_FACTS + [{"line_item": "cfo", "fiscal_year": 2024, "period_end": "2024-09-28",
                          "value": 1.0, "source_url": "u"}]  # no "units": KeyError mid-write
    import pytest
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer.fetch_annual_facts", return_value=bad), \
         patch("app.ingest.facts_writer._facts_from_local_cache", return_value=[]):
        from app.ingest.facts_writer import ensure_facts
        with pytest.raises(Exception):
            ensure_facts("AAPL", years_back=2)
    # the purge was rolled back with the failed write
    assert _keys("AAPL") == {(2024, "FY", "shares_outstanding")}


def test_refresh_keeps_point_in_time_rows_the_repair_writer_wrote(monkeypatch, tmp_path):
    """The purge deleted every normalizer-mapped row in the window, including the rows
    app/autonomous/companyfacts_repair.py writes for a fixed as-of date. It now deletes
    only rows the facts writer owns (written_by 'facts_writer', or NULL from before the
    marker); the repair writer marks its rows and stores their source_tags."""
    cfg = _init_temp_db(monkeypatch, tmp_path)
    from app.autonomous import companyfacts_repair as repair

    sti = {
        "fiscal_year": 2024,
        "period_type": "FY",
        "period_end": "2024-09-28",
        "line_item": "short_term_investments",
        "value": 35228.0,
        "units": "USD_millions",
        "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "filed_date": "2024-11-01",
        "form": "10-K",
        "accession": "0000320193-24-000123",
        "source_tags": "MarketableSecuritiesCurrent",
    }
    monkeypatch.setattr(repair, "normalize_annual_facts_from_raw", lambda *a, **k: [sti])
    result = repair.repair_issuer_annual_companyfacts(
        "AAPL",
        issuer_cik="320193",
        as_of_date="2025-01-15",
        db_path=cfg.db_path,
        fetcher=lambda *a, **k: {
            "status": "OK",
            "reason_code": "FETCH_OK",
            "source_url": sti["source_url"],
            "companyfacts": {"facts": {"us-gaap": {}}},
        },
    )
    assert result["rows_written"] == 1
    with get_db() as conn:
        _seed(conn, "AAPL", 2024, "FY", "shares_outstanding")  # legacy, now refused
    # A refresh once the TTL has passed (the repair row just written is itself fresh).
    with patch("app.ingest.facts_writer.resolve", return_value="0000320193"), \
         patch("app.ingest.facts_writer._is_fresh", return_value=False), \
         patch("app.ingest.facts_writer.fetch_annual_facts", return_value=_FAKE_FACTS), \
         patch("app.ingest.facts_writer._facts_from_local_cache", return_value=[]):
        from app.ingest.facts_writer import ensure_facts
        ensure_facts("AAPL", years_back=2)
    with get_db() as conn:
        rows = {
            r["line_item"]: (r["value"], r["written_by"], r["source_tags"])
            for r in conn.execute(
                "SELECT line_item, value, written_by, source_tags FROM companyfacts_facts "
                "WHERE ticker='AAPL'"
            )
        }
    assert rows == {
        "revenue": (391035.0, "facts_writer", None),
        "net_income": (93736.0, "facts_writer", None),
        "short_term_investments": (35228.0, "companyfacts_repair", "MarketableSecuritiesCurrent"),
    }
