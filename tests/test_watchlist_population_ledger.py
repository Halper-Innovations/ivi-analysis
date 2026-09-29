# tests/test_watchlist_population_ledger.py
"""populate_from_sector_artifact snapshots each verdict into ticker_outcomes.

Every watchlist entry emitted by a sector run must also write a ticker_outcomes
row (the decision ledger) so the calibration loop can later resolve realized +
excess returns. The ledger write is NON-FATAL: a failure inside snapshot_decision
must never abort watchlist population.

Fixture/monkeypatch only — no network, no LLM.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
)
from app.db import get_db, init_db
from app.watchlist.store import populate_from_sector_artifact


@pytest.fixture(autouse=True)
def _legacy_packet_fixture_bypasses_new_integrity_gate(monkeypatch):
    monkeypatch.setattr(
        "app.watchlist.store.artifact_decision_eligibility", lambda payload: "PASS"
    )


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


def _packet(
    ticker: str,
    *,
    current_price: float,
    buy_price_target: float | None = 80.0,
) -> SectorCompanyFinancialPacket:
    valuation = {"anchor_method": "DCF", "valuation_anchor": 106.67}
    if buy_price_target is not None:
        valuation["buy_price_target"] = buy_price_target
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        current_price=current_price,
        valuation=valuation,
    )


def _sector_artifact() -> AutonomousSectorFinancialRunArtifact:
    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_industrial_tech_test",
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        objective="Populate a watchlist from finalist candidates.",
        as_of_date="2026-05-08",
        created_at="2026-05-08T12:00:00Z",
        completed_at="2026-05-08T12:01:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker=None,
        confidence="MODERATE",
        candidate_selection={"selected_tickers": ["AAA", "BBB", "CCC"]},
        company_packets=[
            _packet("AAA", current_price=70.0, buy_price_target=80.0),
            _packet("BBB", current_price=95.0, buy_price_target=80.0),
            _packet("CCC", current_price=55.0, buy_price_target=60.0),
        ],
        relative_ranking=[
            {
                "ticker": "AAA",
                "company_autonomy_verdict": "ACTIONABLE",
                "company_autonomy_confidence": "HIGH",
                "positioning_summary": "AAA clears the buy-price test.",
            },
            {
                "ticker": "BBB",
                "company_autonomy_verdict": "WATCHLIST_ONLY",
                "company_autonomy_confidence": "MODERATE",
                "positioning_summary": "BBB belongs on the watchlist above the buy-price target.",
            },
            {
                "ticker": "CCC",
                "company_autonomy_verdict": "AVOID",
                "positioning_summary": "CCC is rejected by guardrails.",
            },
        ],
    )


def _outcome_rows() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM ticker_outcomes ORDER BY ticker"
        ).fetchall()
    return [dict(r) for r in rows]


def test_population_writes_one_outcome_row_per_emitted_verdict(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    result = populate_from_sector_artifact(_sector_artifact(), db_path=db_path)
    # Only ACTIONABLE / WATCHLIST_ONLY are balloted onto the watchlist as rows.
    assert result.added_or_updated == 2

    rows = _outcome_rows()
    # But EVERY emitted verdict — including AVOID — is recorded in the
    # decision ledger so grade predictiveness (incl. the AVOID sign-inverted
    # segment) is measurable by the automated loop, so the
    # ledger records ALL grades.
    assert len(rows) == 3
    by_ticker = {r["ticker"]: r for r in rows}
    assert set(by_ticker) == {"AAA", "BBB", "CCC"}
    assert by_ticker["AAA"]["grade"] == "ACTIONABLE"
    assert by_ticker["AAA"]["decision"] == "BUY"
    assert by_ticker["BBB"]["grade"] == "WATCHLIST_ONLY"
    assert by_ticker["BBB"]["decision"] == "WATCH"
    assert by_ticker["CCC"]["grade"] == "AVOID"
    assert by_ticker["CCC"]["decision"] == "PASS"
    # The mid_cap sweep selects the small/mid benchmark (IWM), not the
    # hardcoded SPY, and the per-name buy target is snapshotted for the
    # reached-buy-target metric.
    assert by_ticker["AAA"]["benchmark_symbol"] == "IWM"
    assert by_ticker["BBB"]["benchmark_symbol"] == "IWM"
    assert by_ticker["AAA"]["buy_price_target"] == 80.0


def test_population_survives_ledger_failure(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr("app.watchlist.store.snapshot_decision", _boom)

    result = populate_from_sector_artifact(_sector_artifact(), db_path=db_path)

    # Watchlist rows are still written even though the ledger raised.
    assert result.added_or_updated == 2
    # And no outcome rows were committed because every snapshot raised.
    assert len(_outcome_rows()) == 0
