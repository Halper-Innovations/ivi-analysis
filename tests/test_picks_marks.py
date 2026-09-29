from __future__ import annotations

import csv
import hashlib
import sqlite3
from contextlib import closing
from datetime import date

import pytest
from typer.testing import CliRunner

from app.calibration.picks_marks import (
    build_report,
    dollar_line,
    refresh_dates,
    render_markdown,
    summaries,
    write_report,
)
from app.cli_investor import investor_app
from app.config import get_config
from app.db import connect, init_db
from app.holdings import ensure_holdings_schema
from app.watchlist.schema import ensure_watchlist_schema

TODAY = date(2026, 9, 9)
ENTRY = "2026-07-22"


def insert(conn, table, **values):
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values()))


def buy(conn, ticker="AAA", **changes):
    values = dict(
        ticker=ticker,
        as_of_date=ENTRY,
        entry_date=ENTRY,
        run_id="autonomous_sector_fixture",
        decision="BUY",
        conviction=4,
        horizon_days=365,
        entry_price=100.0,
        entry_price_source="watchlist_population",
        benchmark_symbol="IWM",
        buy_price_target=150.0,
        created_at=ENTRY,
        updated_at=ENTRY,
    )
    values.update(changes)
    insert(conn, "ticker_outcomes", **values)


def wl(conn, ticker="AAA", **changes):
    values = dict(
        ticker=ticker,
        added_at=ENTRY + "T01:00:00Z",
        status="ACTIVE",
        conviction_grade="WATCHLIST_ONLY",
        current_price_at_addition=100.0,
        buy_price_target=150.0,
        cap_band="small",
        source_run_id=f"fixture-{conn.execute('SELECT COUNT(*) FROM watchlist').fetchone()[0]}",
    )
    values.update(changes)
    insert(conn, "watchlist", **values)


def quote(conn, ticker="AAA", on="2026-09-09", price=120.0, **changes):
    values = dict(
        ticker=ticker,
        as_of_date=on,
        price=price,
        provider="yahoo",
        currency="USD",
        price_basis="UNADJUSTED",
        split_adjustment_factor=1.0,
        status="OK",
        fetched_at=on + "T14:00:00Z",
        expires_at=on,
        raw_json="{}",
        quote_hash="fixture",
    )
    values.update(changes)
    insert(conn, "price_quotes", **values)


def limit(conn):
    insert(
        conn,
        "recommendation_ledger",
        recommendation_id="fixture-limit",
        ticker="AAA",
        recommendation_type="BUY_AT_LIMIT",
        record_vintage="LIVE",
        model_id="fixture",
        model_vintage="fixture",
        thesis_reference="fixture",
        thesis_summary="fixture",
        trigger_price=200.0,
        trigger_price_source="raw quote",
        trigger_price_as_of=ENTRY,
        trigger_price_age_seconds=0,
        target_price=250.0,
        target_price_source="fixture",
        conviction_grade="ACTIONABLE",
        capacity_class="THIN",
        pre_mortem="fixture",
        risk_flags_json="[]",
        policy_hash="fixture",
        source_run_id="fixture",
        horizons_json="[365]",
        benchmark_symbol="IWM",
        staked_at=ENTRY + "T02:00:00Z",
        recorded_at=ENTRY,
    )


@pytest.fixture
def database(isolated_data_root, monkeypatch):
    class ReportDate(date):
        @classmethod
        def today(cls):
            return TODAY

    monkeypatch.setattr("app.cli_investor.date", ReportDate)
    init_db()
    ensure_watchlist_schema()
    with closing(connect()) as conn:
        ensure_holdings_schema(conn)
        # Verify real schema names before inserting, including nullable basis fields.
        assert [r["name"] for r in conn.execute("PRAGMA table_info(price_quotes)")] == [
            "id",
            "ticker",
            "provider",
            "as_of_date",
            "price",
            "currency",
            "price_basis",
            "split_adjustment_factor",
            "split_effective_date",
            "source_url",
            "status",
            "fetched_at",
            "expires_at",
            "raw_json",
            "quote_hash",
            "volume",
        ]
        for table, expected in {
            "ticker_outcomes": {
                "entry_date",
                "entry_price",
                "entry_price_source",
                "buy_price_target",
                "benchmark_symbol",
                "horizon_days",
                "cap_category",
            },
            "watchlist": {
                "added_at",
                "current_price_at_addition",
                "status",
                "status_reason",
                "conviction_source",
                "source_sector",
                "cap_band",
                "event_pending",
            },
            "recommendation_ledger": {
                "staked_at",
                "trigger_price",
                "target_price",
                "benchmark_symbol",
            },
        }.items():
            actual = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            assert actual & expected == expected
        buy(conn)
        buy(conn, "BBB")
        wl(conn)
        wl(conn, "BBB")
        quote(conn)
        quote(conn, "BBB", price=80)
        quote(conn, "IWM", ENTRY, 50)
        quote(conn, "IWM", price=55)
        conn.commit()
        yield conn


def test_exact_returns_excess_aggregates_and_dollars(database):
    report = build_report(database, today=TODAY)
    a, b = report["rows"]
    assert a["return_pct"] == 20.0
    assert b["return_pct"] == -20.0
    assert a["benchmark_return_pct"] == 10.0
    assert a["excess_return_pct"] == 10.0
    assert b["excess_return_pct"] == -30.0
    assert a["distance_to_buy_target_pct"] == 25.0
    assert a["days_held"] == 49
    assert a["horizon_days"] == 365
    assert a["price_basis"] == "UNADJUSTED"
    assert a["flags"] == "CLOSE_UNCONFIRMED"
    assert summaries(report)[0] == {
        "group_by": "source",
        "group": "BUY",
        "count": 2,
        "priced_count": 2,
        "mean_return_pct": 0.0,
        "median_return_pct": 0.0,
        "excess_count": 2,
        "positive_excess_pct": 50.0,
    }
    assert dollar_line(report) == (
        "$1,000 in each BUY decision at entry is worth $2,000.00 today "
        "($2,000.00 invested; hypothetical quote marks, excluding distributions)."
    )


def test_deduplication_preserves_all_source_rows_status_and_pearl(database):
    buy(database, run_id="autonomous_sector_second", entry_price=50)
    wl(
        database,
        conviction_grade="ACTIONABLE",
        status="REMOVED",
        status_reason="duplicate",
        added_at=ENTRY + "T03:00:00Z",
    )
    wl(
        database,
        conviction_grade="ACTIONABLE",
        status="QUARANTINE",
        status_reason="bad anchor",
        conviction_source="pearl_analyst_review",
        event_pending="EVENT_PENDING:MERGER",
        added_at=ENTRY + "T04:00:00Z",
        cap_band="large_cap",
    )
    limit(database)
    quote(database, "SPY", ENTRY, 100)
    quote(database, "SPY", price=100)
    report = build_report(database, today=TODAY)
    assert len(report["rows"]) == 6
    assert sum(r["counted"] for r in report["rows"]) == 4
    action = [r for r in report["rows"] if r["source"] == "ACTIONABLE"]
    assert [r["status"] for r in action] == ["REMOVED", "QUARANTINE"]
    assert [r["counted"] for r in action] == [False, True]
    assert action[0]["flags"] == "DUPLICATE_NOT_COUNTED;CLOSE_UNCONFIRMED"
    assert action[1]["benchmark_symbol"] == "SPY"
    assert action[1]["event_flag"] == "EVENT_PENDING:MERGER"
    assert action[1]["conviction_source"] == "pearl_analyst_review"
    assert action[1]["status_reason"] == "bad anchor"
    rec = next(r for r in report["rows"] if r["source"] == "BUY_AT_LIMIT")
    assert rec["entry_price"] == 200.0
    assert rec["buy_target"] == 250.0
    assert rec["return_pct"] == -40.0
    # The explicitly stored IWM overrides the newly large-cap watchlist metadata.
    assert rec["benchmark_symbol"] == "IWM"
    assert dollar_line(report).startswith("$1,000 in each BUY decision at entry is worth $3,200.00")


def test_stale_benchmark_is_dated_and_not_current_excess(database):
    database.execute("DELETE FROM price_quotes WHERE ticker = 'IWM' AND as_of_date='2026-09-09'")
    quote(database, "IWM", "2026-08-28", 55)
    row = build_report(database, today=TODAY)["rows"][0]
    assert row["benchmark_exit_date"] == "2026-08-28"
    assert row["benchmark_return_pct"] == 10.0
    assert row["excess_return_pct"] is None
    assert row["flags"] == "CLOSE_UNCONFIRMED;BENCHMARK_STALE"


@pytest.mark.parametrize("basis", [None, "ADJUSTED"])
def test_unlabeled_or_adjusted_history_never_mixes_with_raw(database, basis):
    database.execute("UPDATE price_quotes SET price_basis=? WHERE ticker='IWM'", (basis,))
    database.execute("UPDATE price_quotes SET price_basis=? WHERE ticker='AAA'", (basis,))
    row = build_report(database, today=TODAY)["rows"][0]
    assert row["return_pct"] is None
    assert row["benchmark_return_pct"] is None
    assert row["excess_return_pct"] is None
    assert row["distance_to_buy_target_pct"] is None
    assert row["price_basis"] == (basis or "UNKNOWN")
    suffix = "UNKNOWN" if basis is None else "MISMATCH"
    assert row["flags"] == f"PRICE_BASIS_{suffix};CLOSE_UNCONFIRMED;BENCHMARK_BASIS_{suffix}"


@pytest.mark.parametrize(
    "changes",
    [
        {"split_adjustment_factor": 2.0},
        {"split_adjustment_factor": None, "split_effective_date": "2026-08-01"},
    ],
)
def test_splits_require_check_without_inventing_share_entitlement(database, changes):
    quote(database, on="2026-08-01", **changes)
    report = build_report(database, today=TODAY)
    row = report["rows"][0]
    assert row["flags"] == "SPLIT_CHECK;CLOSE_UNCONFIRMED"
    assert row["return_pct"] is None
    assert row["distance_to_buy_target_pct"] is None
    assert summaries(report)[0]["count"] == 2
    assert summaries(report)[0]["priced_count"] == 1
    assert dollar_line(report) == (
        "$1,000 in each BUY decision at entry: total value cannot be established "
        "(1/2 priced; priced subset worth $800.00)."
    )


def test_latest_quote_ignores_future_and_failed_but_not_unknown_basis(database):
    quote(database, on="2026-09-10", price=999)
    quote(database, on="2026-09-09", price=999, provider="failure", status="ERROR")
    assert build_report(database, today=TODAY)["rows"][0]["latest_close"] == 120
    database.execute(
        "UPDATE price_quotes SET as_of_date='2026-09-08' WHERE ticker='AAA' AND provider='yahoo' AND as_of_date='2026-09-09'"
    )
    quote(database, price=125, price_basis=None, provider="eodhd")
    row = build_report(database, today=TODAY)["rows"][0]
    assert row["latest_close"] == 125
    assert row["quote_date"] == "2026-09-09"
    assert row["return_pct"] is None


def test_missing_prices_entries_and_non_usd_do_not_disappear(database):
    buy(database, "MISSING", entry_price=None)
    buy(database, "UNDATED", entry_date=None)
    quote(database, "UNDATED", price=50)
    database.execute("UPDATE price_quotes SET currency='CAD' WHERE ticker='BBB'")
    report = build_report(database, today=TODAY)
    assert len(report["rows"]) == 4
    assert sum(r["return_pct"] is not None for r in report["rows"]) == 1
    assert "CURRENCY_CHECK" in next(r for r in report["rows"] if r["ticker"] == "BBB")["flags"]


def test_journal_and_csv_markdown_preserve_all_fields(database, isolated_data_root):
    buy(
        database,
        "PASSER",
        decision="PASS",
        run_id="journal_live",
        entry_date=None,
        as_of_date="2026-06-12",
    )
    report = build_report(database, today=TODAY)
    assert report["journal"] == [
        {"ticker": "PASSER", "decision": "PASS", "entry_date": "2026-06-12"}
    ]
    md, csv_path = write_report(report, isolated_data_root / "outputs")
    assert md.read_text() == render_markdown(report)
    assert "| PASSER | 2026-06-12 | PASS |" in md.read_text()
    assert "Recorded holdings: 0. No holdings or live BUY decisions are recorded." in md.read_text()
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["return_pct"] == "20.0"
    assert rows[0]["benchmark_exit_basis"] == "UNADJUSTED"
    assert len(rows) == 2
    assert (md.parent / "report.csv").exists()
    assert (md.parent / "journal.csv").exists()
    assert (md.parent / "summaries.csv").exists()


def test_readonly_connection_rejects_write_and_missing_db(database, isolated_data_root):
    database.commit()
    path = get_config().db_path
    with closing(connect(path, read_only=True)) as conn:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM ticker_outcomes")
        assert len(build_report(conn, today=TODAY)["rows"]) == 2
    missing = isolated_data_root / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        connect(missing, read_only=True)
    assert not missing.exists()


def test_cli_default_is_offline_and_never_changes_database(database, monkeypatch):
    from app.calibration import picks_benchmarks

    database.commit()
    path = get_config().db_path
    database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    def forbidden(*a, **kw):
        pytest.fail("default marks cannot construct a provider")

    monkeypatch.setattr(picks_benchmarks, "get_default_provider", forbidden)
    runner = CliRunner()
    result = runner.invoke(investor_app, ["marks"])
    assert result.exit_code == 0, result.output
    assert "Network requests: 0 (local prices only)." in result.output
    assert "$1,000 in each BUY decision at entry is worth $2,000.00 today" in result.output
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert (
        database.execute("SELECT DISTINCT outcome_status FROM ticker_outcomes").fetchall()[0][0]
        == "OPEN"
    )


def test_cli_disabled_refresh_refuses(database, monkeypatch):
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
    get_config.cache_clear()
    result = CliRunner().invoke(investor_app, ["marks", "--refresh-benchmarks"])
    assert result.exit_code == 1
    assert "configured price provider is disabled" in result.output


def test_refresh_overlays_raw_anchors_only_in_memory(database):
    from app.calibration.picks_benchmarks import BenchmarkPrice

    database.execute("UPDATE price_quotes SET price_basis=NULL WHERE ticker='IWM'")
    refreshed = {
        ("IWM", on): BenchmarkPrice("IWM", on, on, price, "eodhd")
        for on, price in [(ENTRY, 50), ("2026-09-09", 55)]
    }
    report = build_report(database, today=TODAY, refreshed_prices=refreshed)
    assert report["rows"][0]["excess_return_pct"] == 10.0
    assert (
        database.execute("SELECT price_basis FROM price_quotes WHERE ticker='IWM'").fetchone()[0]
        is None
    )
    assert refresh_dates(database, TODAY) == {ENTRY, "2026-09-09"}


def test_prior_benchmark_entry_withholds_same_window_excess(database):
    database.execute(
        "UPDATE price_quotes SET as_of_date='2026-07-21' WHERE ticker='IWM' AND as_of_date=?",
        (ENTRY,),
    )
    row = build_report(database, today=TODAY)["rows"][0]
    assert row["benchmark_return_pct"] == 10.0
    assert row["excess_return_pct"] is None
    assert row["flags"] == "CLOSE_UNCONFIRMED;BENCHMARK_ENTRY_PRIOR_DATE"


def test_partial_refresh_different_provider_does_not_form_benchmark_pair(database):
    from app.calibration.picks_benchmarks import BenchmarkPrice

    refreshed = {("IWM", ENTRY): BenchmarkPrice("IWM", ENTRY, ENTRY, 50, "eodhd")}
    row = build_report(database, today=TODAY, refreshed_prices=refreshed)["rows"][0]
    assert row["benchmark_return_pct"] is None
    assert row["excess_return_pct"] is None
    assert row["flags"] == "CLOSE_UNCONFIRMED;BENCHMARK_PROVIDER_MISMATCH"


def test_cli_refresh_reports_count_and_uses_in_memory_anchors(database, monkeypatch):
    from app.calibration import picks_benchmarks

    database.execute("UPDATE price_quotes SET price_basis=NULL WHERE ticker='IWM'")
    database.commit()
    called = []

    def fake_refresh(cfg, dates):
        called.append(dates)
        return picks_benchmarks.BenchmarkRefresh(
            prices={
                ("IWM", on): picks_benchmarks.BenchmarkPrice("IWM", on, on, p, "eodhd")
                for on, p in [(ENTRY, 50), ("2026-09-09", 55)]
            },
            request_count=4,
            failures=["SPY on 2026-09-09: fixture NO_DATA"],
        )

    monkeypatch.setattr(picks_benchmarks, "refresh_benchmarks", fake_refresh)
    result = CliRunner().invoke(investor_app, ["marks", "--refresh-benchmarks"])
    assert result.exit_code == 0, result.output
    assert len(called) == 1
    assert (
        "Benchmark refresh: 4 provider anchor calls; HTTP requests not measured." in result.output
    )
    assert "SPY on 2026-09-09: fixture NO_DATA" in result.output
    assert "| +20.00% |" in result.output
    assert (
        database.execute("SELECT price_basis FROM price_quotes WHERE ticker='IWM'").fetchone()[0]
        is None
    )


def test_no_picks_and_existing_holdings_are_reported_truthfully(database):
    database.execute("DELETE FROM ticker_outcomes")
    insert(
        database, "holdings", ticker="HELD", entry_date=ENTRY, created_at=ENTRY, updated_at=ENTRY
    )
    report = build_report(database, today=TODAY)
    assert report["rows"] == []
    assert (
        dollar_line(report) == "No BUY decisions are recorded; no hypothetical investment to mark."
    )
    assert "Recorded holdings: 1." in render_markdown(report)
    assert "No holdings or live BUY decisions are recorded." not in render_markdown(report)


def test_split_after_last_usable_quote_withholds_today_value_and_target(database):
    database.execute("UPDATE price_quotes SET as_of_date='2026-09-08' WHERE ticker='AAA'")
    quote(
        database,
        price=None,
        status="ERROR",
        split_effective_date="2026-09-09",
        split_adjustment_factor=2.0,
    )
    report = build_report(database, today=TODAY)
    row = report["rows"][0]
    assert row["quote_date"] == "2026-09-08"
    assert "SPLIT_CHECK" in row["flags"]
    assert "PRICE_STALE" in row["flags"]
    assert row["return_pct"] is None
    assert row["distance_to_buy_target_pct"] is None
    assert "total value cannot be established (1/2 priced" in dollar_line(report)
