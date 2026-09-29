from __future__ import annotations
from dataclasses import dataclass
from app.config import get_config
from app.db import get_db, init_db


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


@dataclass
class _Snap:
    ticker: str
    as_of_date: str
    price: float


class _DelistedProvider:
    """Prices the benchmark but never the name (simulates a delisted ticker)."""

    def get_price_asof(self, ticker, as_of_date):
        if ticker in ("IWM", "SPY"):
            return _Snap(ticker, as_of_date, 100.0)
        return None


def test_matured_unresolvable_row_is_marked_unresolved(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.outcomes.store import add_outcome
    from app.calibration.return_resolver import resolve_open_outcomes

    add_outcome(
        ticker="DEAD",
        as_of_date="2022-01-03",
        run_id="h1_sample_h365",
        decision="BUY",
        conviction=2,
        horizon_days=365,
        entry_price=10.0,
        entry_price_source="historical_backtest",
        entry_date="2022-01-03",
        grade="DEPLOY_READY",
        status="CHEAP_VS_EXPECTATIONS",
        benchmark_symbol="IWM",
        buy_price_target=8.0,
        cap_category="micro",
    )

    # Without the grace flag: stays OPEN (legacy behavior).
    summary = resolve_open_outcomes("2026-06-05", provider=_DelistedProvider())
    assert summary.skipped_no_price == 1
    with get_db() as conn:
        st = conn.execute("SELECT outcome_status FROM ticker_outcomes WHERE ticker='DEAD'").fetchone()
    assert st["outcome_status"] == "OPEN"

    # With grace=0: a clearly-matured unresolvable row is closed UNRESOLVED.
    summary = resolve_open_outcomes(
        "2026-06-05", provider=_DelistedProvider(), unresolved_grace_days=0
    )
    assert summary.unresolved == 1
    with get_db() as conn:
        st = conn.execute("SELECT outcome_status FROM ticker_outcomes WHERE ticker='DEAD'").fetchone()
    assert st["outcome_status"] == "UNRESOLVED"
