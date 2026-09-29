from __future__ import annotations

import sqlite3

import pytest

from app.backtest.inference import (
    FEW_CLUSTER_WARNING,
    cluster_bootstrap_contrast,
    format_bootstrap_report,
    load_closed_outcomes,
)


def _row(ticker, as_of_date, grade, excess, *, status="CHEAP_VS_EXPECTATIONS", realized=None):
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "grade": grade,
        "status": status,
        "excess_return_pct": excess,
        "realized_return_pct": realized if realized is not None else (excess + 1.0 if excess is not None else None),
    }


def _main_rows():
    """3 clusters; treat=[10,6 | 8 | 2,4] control=[2 | -2,0 | 4].

    By hand: mean contrast = 6.0 - 1.0 = 5.0; median contrast = 6.0 - 1.0 = 5.0.
    Per-cluster: 8-2=6.0, 8-(-1)=9.0, 3-4=-1.0. Leave-one-out: 4.0, 2.5, 8.0.
    """
    return [
        _row("AAA", "2024-01-02", "DEPLOY_READY", 10.0),
        _row("BBB", "2024-01-02", "DEPLOY_READY", 6.0),
        _row("CCC", "2024-07-01", "DEPLOY_READY", 8.0),
        _row("DDD", "2025-01-02", "DEPLOY_READY", 2.0),
        _row("EEE", "2025-01-02", "DEPLOY_READY", 4.0),
        _row("FFF", "2024-01-02", "NOT_CHEAP", 2.0),
        _row("GGG", "2024-07-01", "NOT_CHEAP", -2.0),
        _row("HHH", "2024-07-01", "NOT_CHEAP", 0.0),
        _row("III", "2025-01-02", "NOT_CHEAP", 4.0),
    ]


@pytest.fixture()
def outcomes_db(tmp_path):
    """Handcrafted ticker_outcomes table mirroring the real column names."""
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE ticker_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            outcome_status TEXT NOT NULL,
            realized_return_pct REAL,
            excess_return_pct REAL,
            grade TEXT,
            status TEXT,
            benchmark_symbol TEXT
        )
        """
    )
    insert = (
        "INSERT INTO ticker_outcomes(ticker, as_of_date, run_id, horizon_days, outcome_status, "
        "realized_return_pct, excess_return_pct, grade, status, benchmark_symbol) "
        "VALUES (?, ?, ?, 365, ?, ?, ?, ?, ?, 'SPY')"
    )
    for r in _main_rows():
        conn.execute(
            insert,
            (
                r["ticker"], r["as_of_date"], "test_run", "CLOSED",
                r["realized_return_pct"], r["excess_return_pct"], r["grade"], r["status"],
            ),
        )
    # Excluded rows: OPEN, NULL excess, other run_id
    conn.execute(insert, ("OPN", "2024-01-02", "test_run", "OPEN", None, 9.0, "DEPLOY_READY", "X"))
    conn.execute(insert, ("NUL", "2024-01-02", "test_run", "CLOSED", 3.0, None, "DEPLOY_READY", "X"))
    conn.execute(insert, ("OTH", "2024-01-02", "other_run", "CLOSED", 3.0, 3.0, "DEPLOY_READY", "X"))
    conn.commit()
    conn.close()
    return db_path


def test_load_closed_outcomes_filters_and_fields(outcomes_db):
    rows = load_closed_outcomes("test_run", db_path=outcomes_db)
    assert len(rows) == 9
    tickers = {r["ticker"] for r in rows}
    assert "OPN" not in tickers and "NUL" not in tickers and "OTH" not in tickers
    first = rows[0]
    assert set(first) == {
        "ticker", "as_of_date", "grade", "status", "excess_return_pct", "realized_return_pct",
    }
    assert first["ticker"] == "AAA"  # ordered by as_of_date, ticker
    assert first["excess_return_pct"] == 10.0


def test_observed_contrasts_exact_literals():
    result = cluster_bootstrap_contrast(_main_rows(), n_boot=10, seed=1)
    assert result["observed_mean_contrast"] == 5.0
    assert result["observed_median_contrast"] == 5.0
    assert result["n_treat"] == 5
    assert result["n_control"] == 4
    assert result["n_clusters"] == 3
    assert result["per_cluster_contrasts"] == {
        "2024-01-02": 6.0,
        "2024-07-01": 9.0,
        "2025-01-02": -1.0,
    }
    assert result["leave_one_out_contrasts"] == {
        "2024-01-02": 4.0,
        "2024-07-01": 2.5,
        "2025-01-02": 8.0,
    }


def test_bootstrap_deterministic_exact_literals():
    # Literals captured from one run at n_boot=200, seed=7 (do not recompute).
    result = cluster_bootstrap_contrast(_main_rows(), n_boot=200, seed=7)
    assert result["boot_se_mean"] == 2.3519
    assert result["boot_se_median"] == 2.5432
    assert result["ci95_mean"] == (1.275, 9.0)
    assert result["ci95_median"] == (-0.025, 9.0)
    assert result["n_boot_effective"] == 200
    assert result["skipped_draws"] == 0


def test_same_seed_identical_different_seed_differs():
    a = cluster_bootstrap_contrast(_main_rows(), n_boot=200, seed=7)
    b = cluster_bootstrap_contrast(_main_rows(), n_boot=200, seed=7)
    assert a == b
    c = cluster_bootstrap_contrast(_main_rows(), n_boot=200, seed=8)
    assert c["boot_se_mean"] != a["boot_se_mean"]


def test_cohort_empty_clusters_skip_draws():
    # One all-treat cluster + one all-control cluster: any draw that repeats a
    # single cluster has an empty cohort and must be skipped.
    rows = [
        _row("AAA", "2024-01-02", "DEPLOY_READY", 5.0),
        _row("BBB", "2024-01-02", "DEPLOY_READY", 7.0),
        _row("CCC", "2024-07-01", "NOT_CHEAP", 1.0),
        _row("DDD", "2024-07-01", "NOT_CHEAP", 3.0),
    ]
    result = cluster_bootstrap_contrast(rows, n_boot=100, seed=3)
    assert result["observed_mean_contrast"] == 4.0
    assert result["per_cluster_contrasts"] == {"2024-01-02": None, "2024-07-01": None}
    assert result["leave_one_out_contrasts"] == {"2024-01-02": None, "2024-07-01": None}
    # Literals captured from one run at seed=3 (do not recompute).
    assert result["skipped_draws"] == 46
    assert result["n_boot_effective"] == 54
    assert result["n_boot_effective"] + result["skipped_draws"] == 100


def test_raises_when_cohort_missing_entirely():
    rows = [_row("AAA", "2024-01-02", "DEPLOY_READY", 5.0)]
    with pytest.raises(ValueError, match="NOT_CHEAP"):
        cluster_bootstrap_contrast(rows, n_boot=10, seed=1)


def test_report_includes_few_cluster_warning():
    result = cluster_bootstrap_contrast(_main_rows(), n_boot=200, seed=7)
    report = format_bootstrap_report(result, "test_run")
    assert "# Cluster Bootstrap Contrast (test_run)" in report
    assert FEW_CLUSTER_WARNING.format(n=3) in report
    assert "unreliable with so few clusters" in report
    assert "observed mean contrast: 5.0" in report
    assert "2024-07-01: 9.0" in report


def test_report_omits_warning_at_five_clusters():
    dates = ["2024-01-02", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-02"]
    rows = []
    for i, d in enumerate(dates):
        rows.append(_row(f"T{i}", d, "DEPLOY_READY", 5.0 + i))
        rows.append(_row(f"C{i}", d, "NOT_CHEAP", 1.0 + i))
    result = cluster_bootstrap_contrast(rows, n_boot=50, seed=1)
    assert result["n_clusters"] == 5
    report = format_bootstrap_report(result, "test_run")
    assert "unreliable with so few clusters" not in report
