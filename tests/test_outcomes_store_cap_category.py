from __future__ import annotations
from app.config import get_config
from app.db import get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def test_add_outcome_persists_cap_category(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.outcomes.store import add_outcome

    add_outcome(
        ticker="AAA",
        as_of_date="2024-01-02",
        run_id="h1_sample_h365",
        decision="BUY",
        conviction=2,
        horizon_days=365,
        entry_price=60.0,
        entry_price_source="historical_backtest",
        entry_date="2024-01-02",
        grade="DEPLOY_READY",
        status="CHEAP_VS_EXPECTATIONS",
        benchmark_symbol="SPY",
        buy_price_target=75.0,
        cap_category="large_cap",
    )
    with get_db() as conn:
        row = conn.execute(
            "SELECT cap_category FROM ticker_outcomes WHERE ticker='AAA' AND run_id='h1_sample_h365'"
        ).fetchone()
    assert row["cap_category"] == "large_cap"


def test_add_outcome_cap_category_defaults_to_none(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.outcomes.store import add_outcome

    add_outcome(
        ticker="BBB",
        as_of_date="2024-01-02",
        run_id="h1_sample_h90",
        decision="BUY",
        conviction=2,
        horizon_days=90,
        entry_price=60.0,
        entry_price_source="historical_backtest",
        entry_date="2024-01-02",
        grade="DEPLOY_READY",
        status="CHEAP_VS_EXPECTATIONS",
        benchmark_symbol="SPY",
        buy_price_target=75.0,
    )
    with get_db() as conn:
        row = conn.execute(
            "SELECT cap_category FROM ticker_outcomes WHERE ticker='BBB' AND run_id='h1_sample_h90'"
        ).fetchone()
    assert row["cap_category"] is None
