# tests/test_decision_ledger.py
"""decision_ledger.snapshot_decision persists a verdict into ticker_outcomes.

Snapshots every emitted verdict at decision time so the calibration loop can later
resolve realized + excess returns. Tests use an env-redirected tmp db (VOE_DB_PATH)
and fixture WatchlistEntry objects only — no network, no LLM.
"""
from __future__ import annotations

from pathlib import Path

from app.calibration.decision_ledger import select_benchmark_symbol, snapshot_decision
from app.db import get_db, init_db
from app.watchlist.contract import WatchlistEntry


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


def _entry(*, ticker: str = "AAA", current_price_at_addition: float | None = 42.0) -> WatchlistEntry:
    return WatchlistEntry(
        ticker=ticker,
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        conviction_source="company_autonomy",
        buy_price_target=30.0,
        current_price_at_addition=current_price_at_addition,
        source_run_id="r1",
        source_sector="industrial_tech",
        added_at="2026-05-17T12:00:00+00:00",
    )


def _rows(ticker: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM ticker_outcomes WHERE ticker = ?", (ticker.upper(),)
        ).fetchall()
    return [dict(r) for r in rows]


def test_snapshot_actionable_writes_one_buy_row(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    snapshot_decision(
        _entry(),
        grade="ACTIONABLE",
        confidence="HIGH",
        status="DEPLOY_READY",
        run_id="r1",
        as_of_date="2026-05-17",
        horizon_days=365,
        market_cap_focus="large_cap",
    )
    rows = _rows("AAA")
    assert len(rows) == 1
    row = rows[0]
    assert row["decision"] == "BUY"
    assert row["conviction"] == 4
    assert row["entry_price"] == 42.0
    assert row["grade"] == "ACTIONABLE"
    assert row["status"] == "DEPLOY_READY"
    assert row["benchmark_symbol"] == "SPY"
    assert row["buy_price_target"] == 30.0
    assert row["outcome_status"] == "OPEN"


def test_select_benchmark_symbol_cap_aware():
    # SPY for large-cap, IWM for small/mid (and everything else).
    assert select_benchmark_symbol("large_cap") == "SPY"
    assert select_benchmark_symbol("large_cap_financials") == "SPY"
    assert select_benchmark_symbol("small_cap") == "IWM"
    assert select_benchmark_symbol("mid_cap") == "IWM"
    assert select_benchmark_symbol("smid_cap") == "IWM"
    assert select_benchmark_symbol(None) == "IWM"
    # Per-name category wins over the sweep focus.
    assert select_benchmark_symbol("small_cap", "large_cap") == "SPY"
    assert select_benchmark_symbol("large_cap", "small_cap") == "IWM"


def test_snapshot_selects_iwm_for_small_cap(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    snapshot_decision(
        _entry(ticker="SML"),
        grade="ACTIONABLE",
        confidence="HIGH",
        status="DEPLOY_READY",
        run_id="r1",
        as_of_date="2026-05-17",
        horizon_days=365,
        market_cap_focus="small_cap",
    )
    assert _rows("SML")[0]["benchmark_symbol"] == "IWM"


def test_snapshot_explicit_benchmark_overrides_cap(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    snapshot_decision(
        _entry(ticker="OVR"),
        grade="ACTIONABLE",
        confidence="HIGH",
        status="DEPLOY_READY",
        run_id="r1",
        as_of_date="2026-05-17",
        horizon_days=365,
        market_cap_focus="small_cap",
        benchmark_symbol="SPY",
    )
    assert _rows("OVR")[0]["benchmark_symbol"] == "SPY"


def test_snapshot_grade_decision_mapping(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    snapshot_decision(
        _entry(ticker="WAT"),
        grade="WATCHLIST_ONLY",
        confidence="HIGH",
        status="ACTIVE",
        run_id="r1",
        as_of_date="2026-05-17",
        horizon_days=365,
    )
    snapshot_decision(
        _entry(ticker="AVD"),
        grade="AVOID",
        confidence="HIGH",
        status="ACTIVE",
        run_id="r1",
        as_of_date="2026-05-17",
        horizon_days=365,
    )
    assert _rows("WAT")[0]["decision"] == "WATCH"
    assert _rows("AVD")[0]["decision"] == "PASS"


def test_snapshot_none_price_writes_nothing(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    result = snapshot_decision(
        _entry(ticker="NUL", current_price_at_addition=None),
        grade="ACTIONABLE",
        confidence="HIGH",
        status="DEPLOY_READY",
        run_id="r1",
        as_of_date="2026-05-17",
        horizon_days=365,
    )
    assert result is None
    assert len(_rows("NUL")) == 0


def test_snapshot_idempotent_upsert_overwrites_grade(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    snapshot_decision(
        _entry(ticker="UPS"),
        grade="ACTIONABLE",
        confidence="HIGH",
        status="DEPLOY_READY",
        run_id="r1",
        as_of_date="2026-05-17",
        horizon_days=365,
    )
    snapshot_decision(
        _entry(ticker="UPS"),
        grade="WATCHLIST_ONLY",
        confidence="HIGH",
        status="ACTIVE",
        run_id="r1",
        as_of_date="2026-05-17",
        horizon_days=365,
    )
    rows = _rows("UPS")
    assert len(rows) == 1
    assert rows[0]["grade"] == "WATCHLIST_ONLY"
    assert rows[0]["decision"] == "WATCH"
