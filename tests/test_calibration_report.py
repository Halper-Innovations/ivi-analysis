# tests/test_calibration_report.py
"""calibration_report segments CLOSED ticker_outcomes by grade and status.

Pure aggregation (``_segment_stats``) is tested with literal arithmetic; the
report builder is tested against an env-redirected tmp db (VOE_DB_PATH) seeded
with CLOSED rows only — no network, no LLM, no live price fetch. Expected values
are hand-computed literals, never recomputed from the production logic.
"""
from __future__ import annotations

from pathlib import Path

from app.calibration.calibration_report import (
    _segment_stats,
    build_calibration_report,
    latest_grade_status_report,
)
from app.db import get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


def _insert_closed(
    *,
    ticker: str,
    run_id: str,
    grade: str,
    status: str,
    realized: float,
    excess: float | None = None,
    benchmark: float | None = None,
    reached: int | None = None,
) -> None:
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO ticker_outcomes(
                ticker, as_of_date, run_id, decision, conviction, horizon_days,
                thesis_tags_json, notes, outcome_status, close_date,
                realized_return_pct, benchmark_return_pct, excess_return_pct,
                reached_buy_target, entry_price, entry_date, grade, status,
                benchmark_symbol, created_at, updated_at
            ) VALUES(?, ?, ?, 'BUY', 4, 365, '[]', '', 'CLOSED', '2026-05-17',
                ?, ?, ?, ?, 100.0, '2025-05-17', ?, ?, 'SPY', ?, ?)
            """,
            (
                ticker.upper(),
                "2025-05-17",
                run_id,
                realized,
                benchmark,
                excess,
                reached,
                grade,
                status,
                now,
                now,
            ),
        )


def test_segment_stats_basic_four_rows() -> None:
    rows = [
        {"realized_return_pct": 20.0, "excess_return_pct": None},
        {"realized_return_pct": -10.0, "excess_return_pct": None},
        {"realized_return_pct": 5.0, "excess_return_pct": None},
        {"realized_return_pct": -5.0, "excess_return_pct": None},
    ]
    stats = _segment_stats(rows)
    assert stats["n"] == 4
    assert stats["hit_rate"] == 0.5
    assert stats["avg_return"] == 2.5
    assert stats["median_return"] == 0.0


def test_segment_stats_avg_excess() -> None:
    rows = [
        {"realized_return_pct": 20.0, "excess_return_pct": 10.0},
        {"realized_return_pct": -10.0, "excess_return_pct": -4.0},
    ]
    stats = _segment_stats(rows)
    assert stats["avg_excess"] == 3.0


def test_build_calibration_report_segments_by_grade(monkeypatch, tmp_path) -> None:
    _init_temp_db(monkeypatch, tmp_path)
    _insert_closed(ticker="AAA", run_id="r1", grade="ACTIONABLE", status="DEPLOY_READY", realized=20.0)
    _insert_closed(ticker="BBB", run_id="r1", grade="ACTIONABLE", status="ACTIVE", realized=-5.0)
    _insert_closed(ticker="CCC", run_id="r1", grade="WATCHLIST_ONLY", status="ACTIVE", realized=2.0)

    report = build_calibration_report("2026-05-30")

    assert report["by_grade"]["ACTIONABLE"]["n"] == 2
    assert report["by_grade"]["ACTIONABLE"]["hit_rate"] == 0.5
    assert report["by_grade"]["WATCHLIST_ONLY"]["n"] == 1
    assert report["by_grade"]["WATCHLIST_ONLY"]["hit_rate"] == 1.0


def test_build_calibration_report_upserts_single_row(monkeypatch, tmp_path) -> None:
    _init_temp_db(monkeypatch, tmp_path)
    _insert_closed(ticker="AAA", run_id="r1", grade="ACTIONABLE", status="DEPLOY_READY", realized=20.0)

    build_calibration_report("2026-05-30")
    build_calibration_report("2026-05-30")

    with get_db() as conn:
        rows = conn.execute(
            "SELECT run_id FROM calibration_reports WHERE run_id = ?",
            ("calibration_report_2026-05-30",),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["run_id"] == "calibration_report_2026-05-30"


def test_segment_stats_excess_hit_rate_is_headline() -> None:
    # excess > 0 in 3 of 4 rows -> excess_hit_rate 0.75; absolute hit_rate 0.5.
    rows = [
        {"realized_return_pct": 20.0, "excess_return_pct": 10.0, "reached_buy_target": None},
        {"realized_return_pct": -10.0, "excess_return_pct": 4.0, "reached_buy_target": None},
        {"realized_return_pct": 5.0, "excess_return_pct": 2.0, "reached_buy_target": None},
        {"realized_return_pct": -5.0, "excess_return_pct": -3.0, "reached_buy_target": None},
    ]
    stats = _segment_stats(rows)
    assert stats["n"] == 4
    assert stats["excess_hit_rate"] == 0.75
    assert stats["hit_rate"] == 0.5
    assert stats["target_hit_rate"] is None


def test_segment_stats_target_hit_rate() -> None:
    # 2 of 3 rows carry a reached flag; 1 of those reached -> 0.5 over 2.
    rows = [
        {"realized_return_pct": 1.0, "excess_return_pct": None, "reached_buy_target": 1},
        {"realized_return_pct": 1.0, "excess_return_pct": None, "reached_buy_target": 0},
        {"realized_return_pct": 1.0, "excess_return_pct": None, "reached_buy_target": None},
    ]
    stats = _segment_stats(rows)
    assert stats["target_hit_rate"] == 0.5


def test_segment_stats_avoid_inverts_sign() -> None:
    # AVOID rows: a "hit" is an underperformance. realized -5 / +10; excess -3 / +2.
    # Inverted absolute hits: realized<0 -> 1 of 2 = 0.5.
    # Inverted excess hits: excess<0 -> 1 of 2 = 0.5.
    rows = [
        {"realized_return_pct": -5.0, "excess_return_pct": -3.0, "reached_buy_target": None, "invert": True},
        {"realized_return_pct": 10.0, "excess_return_pct": 2.0, "reached_buy_target": None, "invert": True},
    ]
    stats = _segment_stats(rows)
    assert stats["hit_rate"] == 0.5
    assert stats["excess_hit_rate"] == 0.5
    # avg_return / avg_excess are raw margins, never sign-flipped.
    assert stats["avg_return"] == 2.5
    assert stats["avg_excess"] == -0.5


def test_segment_stats_decoupled_filtering_consistent_denominators() -> None:
    # A row with realized=None but excess present must NOT contribute to n / hit_rate,
    # yet still counts toward avg_excess / excess_hit_rate (separate denominators).
    rows = [
        {"realized_return_pct": 10.0, "excess_return_pct": None, "reached_buy_target": None},
        {"realized_return_pct": None, "excess_return_pct": 4.0, "reached_buy_target": None},
    ]
    stats = _segment_stats(rows)
    assert stats["n"] == 1
    assert stats["hit_rate"] == 1.0
    assert stats["excess_hit_rate"] == 1.0
    assert stats["avg_excess"] == 4.0


def test_build_report_headline_and_avoid_inversion(monkeypatch, tmp_path) -> None:
    _init_temp_db(monkeypatch, tmp_path)
    # ACTIONABLE: beats benchmark (excess>0) and absolute>0, reached target.
    _insert_closed(
        ticker="AAA", run_id="r1", grade="ACTIONABLE", status="DEPLOY_READY",
        realized=20.0, excess=8.0, benchmark=12.0, reached=1,
    )
    # AVOID that correctly fell: realized<0 and excess<0 -> inverted hit.
    _insert_closed(
        ticker="BAD", run_id="r1", grade="AVOID", status="ACTIVE",
        realized=-15.0, excess=-9.0, benchmark=-6.0, reached=0,
    )

    report = build_calibration_report("2026-05-30")

    assert report["headline_hit_metric"] == "excess_return_pct>0"
    assert report["avoid_sign_inverted"] is True
    assert report["by_grade"]["ACTIONABLE"]["excess_hit_rate"] == 1.0
    assert report["by_grade"]["ACTIONABLE"]["target_hit_rate"] == 1.0
    # AVOID fell, so the inverted excess/absolute hit both count as correct.
    assert report["by_grade"]["AVOID"]["excess_hit_rate"] == 1.0
    assert report["by_grade"]["AVOID"]["hit_rate"] == 1.0
    # Overall excess_hit_rate: ACTIONABLE beat (+8 -> hit) + AVOID inverted (-9 -> hit) = 2/2.
    assert report["overall"]["excess_hit_rate"] == 1.0


def test_build_calibration_report_excludes_backtest_rows(monkeypatch, tmp_path) -> None:
    """Backtest rows (entry_price_source='historical_backtest') must never appear in the live report."""
    _init_temp_db(monkeypatch, tmp_path)
    # LIVE row: entry_price_source=NULL, grade=ACTIONABLE
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO ticker_outcomes(
                ticker, as_of_date, run_id, decision, conviction, horizon_days,
                thesis_tags_json, notes, outcome_status, close_date,
                realized_return_pct, benchmark_return_pct, excess_return_pct,
                reached_buy_target, entry_price, entry_date, grade, status,
                benchmark_symbol, entry_price_source, created_at, updated_at
            ) VALUES('LIVE1', '2025-05-17', 'r_live', 'BUY', 4, 365, '[]', '',
                'CLOSED', '2026-05-17', 12.0, 5.0, 7.0, 1, 100.0, '2025-05-17',
                'ACTIONABLE', 'DEPLOY_READY', 'SPY', NULL, ?, ?)
            """,
            (now, now),
        )
        # BACKTEST row: entry_price_source='historical_backtest', grade=DEPLOY_READY
        conn.execute(
            """
            INSERT INTO ticker_outcomes(
                ticker, as_of_date, run_id, decision, conviction, horizon_days,
                thesis_tags_json, notes, outcome_status, close_date,
                realized_return_pct, benchmark_return_pct, excess_return_pct,
                reached_buy_target, entry_price, entry_date, grade, status,
                benchmark_symbol, entry_price_source, created_at, updated_at
            ) VALUES('BT1', '2025-05-17', 'r_bt', 'BUY', 4, 365, '[]', '',
                'CLOSED', '2026-05-17', 25.0, 10.0, 15.0, 1, 100.0, '2025-05-17',
                'DEPLOY_READY', 'ACTIVE', 'IWM', 'historical_backtest', ?, ?)
            """,
            (now, now),
        )

    report = build_calibration_report("2026-06-01")

    # Overall n must be 1 (the live row only, not the backtest row)
    assert report["overall"]["n"] == 1
    # DEPLOY_READY grade must not appear in by_grade (it belongs to the backtest row)
    assert "DEPLOY_READY" not in report["by_grade"]
    # ACTIONABLE must appear with n=1
    assert report["by_grade"]["ACTIONABLE"]["n"] == 1


def test_latest_grade_status_report_ignores_discovery_family(monkeypatch, tmp_path) -> None:
    _init_temp_db(monkeypatch, tmp_path)
    _insert_closed(
        ticker="AAA", run_id="r1", grade="ACTIONABLE", status="DEPLOY_READY", realized=20.0,
    )
    build_calibration_report("2026-05-30")
    # Insert a discovery-calibration row with a later created_at to prove the
    # prefix filter, not recency, decides what latest_grade_status_report returns.
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO calibration_reports(run_id, as_of_date, report_json, report_path, created_at)
            VALUES('calibration_99991231T000000Z', NULL, '{"family":"discovery"}', '/tmp/x.json', ?)
            """,
            (now,),
        )

    latest = latest_grade_status_report()
    assert latest is not None
    assert latest["run_id"] == "calibration_report_2026-05-30"
    assert latest["report_family"] == "grade_status_calibration"
