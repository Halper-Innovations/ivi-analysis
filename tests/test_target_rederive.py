from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from app.db import init_db
from app.watchlist.contract import WatchlistEntry
from app.watchlist.store import add_or_update, get_latest


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


def _seed_row(
    db_path,
    *,
    ticker: str,
    status: str = "DEPLOY_READY",
    conviction_grade: str = "ACTIONABLE",
    confidence: str = "MODERATE",
    anchor_method: str = "dcf",
    anchor_value: float = 40.13,
    buy_price_target: float = 27.31,
) -> int:
    return add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            conviction_grade=conviction_grade,
            confidence=confidence,
            conviction_source="sector_final_decision",
            valuation_anchor_method=anchor_method,
            valuation_anchor_value=anchor_value,
            buy_price_target=buy_price_target,
            current_price_at_addition=17.63,
            thesis_text=f"{ticker} thesis.",
            source_run_id=f"run_{ticker}",
            source_sector="advertising_tech",
            added_at="2026-06-01T12:00:00Z",
        ),
        db_path=db_path,
    )


def _patch_packet_chain(monkeypatch, fixtures: dict[str, dict]):
    """Route the production chain to per-ticker fakes.

    ``fixtures[ticker]`` -> the fin-packet ``valuation`` dict (plus optional
    ``trend`` for the quality context). A ticker mapped to an Exception
    instance raises from packet assembly.
    """
    from app.watchlist import target_rederive

    def fake_assemble(ticker: str, *, filing_risk_use_llm: bool = True):
        fixture = fixtures[ticker.upper()]
        if isinstance(fixture, Exception):
            raise fixture
        return SimpleNamespace(
            ticker=ticker.upper(),
            raw_quality_ctx={"revenue_trend_class": fixture.get("trend") or "FLAT"},
        )

    def fake_build(signal_packet, *, sector=None, as_of_date=None, cap_classification=None):
        fixture = fixtures[signal_packet.ticker]
        return SimpleNamespace(
            ticker=signal_packet.ticker,
            valuation=fixture["valuation"],
        )

    monkeypatch.setattr(target_rederive, "assemble_signal_packet", fake_assemble)
    monkeypatch.setattr(target_rederive, "build_sector_company_financial_packet", fake_build)


def _rederive_history_rows(db_path, watchlist_id: int) -> list[dict]:
    """History rows written by the rederive itself (seeding writes 'created')."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM watchlist_history WHERE watchlist_id = ? AND source = 'target_rederive' ORDER BY id",
            (watchlist_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def test_dry_run_reports_decline_cap_retarget_without_writing(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_row(db_path, ticker="CRTO")
    # Production today: decline-capped to the EPV basis ($28) — the stale row
    # still carries the DCF-anchored $27.31 target off a $40.13 anchor.
    _patch_packet_chain(
        monkeypatch,
        {
            "CRTO": {
                "trend": "DECLINING",
                "valuation": {"anchor_method": "epv", "valuation_anchor": 28.0},
            }
        },
    )
    from app.watchlist.target_rederive import rederive_watchlist_targets

    report = rederive_watchlist_targets(db_path=db_path, apply=False)

    assert report["applied"] is False
    assert report["counts"] == {"RETARGET": 1}
    row = report["rows"][0]
    assert row["ticker"] == "CRTO"
    assert row["revenue_trend_class"] == "DECLINING"
    assert row["new_anchor_method"] == "epv"
    # (ACTIONABLE, MODERATE) base 0.12 + 0.20 * neutral dispersion 0.25 = 0.17
    assert row["new_buy_target"] == pytest.approx(23.24, abs=0.01)
    # Dry run: stored row untouched, no history.
    entry = get_latest("CRTO", db_path=db_path)
    assert entry.buy_price_target == 27.31
    assert _rederive_history_rows(db_path, row["watchlist_id"]) == []


def test_apply_updates_row_and_writes_history(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _seed_row(db_path, ticker="CRTO")
    _patch_packet_chain(
        monkeypatch,
        {
            "CRTO": {
                "trend": "DECLINING",
                "valuation": {"anchor_method": "epv", "valuation_anchor": 28.0},
            }
        },
    )
    from app.watchlist.target_rederive import rederive_watchlist_targets

    report = rederive_watchlist_targets(db_path=db_path, apply=True)

    assert report["applied"] is True
    assert report["counts"] == {"RETARGET": 1}
    entry = get_latest("CRTO", db_path=db_path)
    assert entry.valuation_anchor_method == "epv"
    assert entry.valuation_anchor_value == 28.0
    assert entry.buy_price_target == pytest.approx(23.24, abs=0.01)
    history = _rederive_history_rows(db_path, row_id)
    changed_fields = {row["field_name"] for row in history}
    assert changed_fields == {
        "buy_price_target",
        "valuation_anchor_method",
        "valuation_anchor_value",
    }
    assert all(row["source"] == "target_rederive" for row in history)
    target_row = next(row for row in history if row["field_name"] == "buy_price_target")
    assert target_row["old_value"] == "27.31"


def test_would_null_rows_are_never_mutated(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _seed_row(db_path, ticker="GATE")
    _patch_packet_chain(
        monkeypatch,
        {
            "GATE": {
                "valuation": {
                    "anchor_method": None,
                    "valuation_anchor": None,
                    "gate_action": "BLOCK",
                }
            }
        },
    )
    from app.watchlist.target_rederive import rederive_watchlist_targets

    report = rederive_watchlist_targets(db_path=db_path, apply=True)

    assert report["counts"] == {"WOULD_NULL": 1}
    assert report["rows"][0]["detail"] == "gate_action=BLOCK"
    entry = get_latest("GATE", db_path=db_path)
    assert entry.buy_price_target == 27.31
    assert entry.valuation_anchor_method == "dcf"
    assert _rederive_history_rows(db_path, row_id) == []


def test_unchanged_within_tolerance_is_not_retargeted(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    # Stored target 33.30; derived = 40.13 * (1 - 0.17) = 33.3079 — within 0.5%.
    _seed_row(db_path, ticker="SAME", buy_price_target=33.30)
    _patch_packet_chain(
        monkeypatch,
        {"SAME": {"valuation": {"anchor_method": "dcf", "valuation_anchor": 40.13}}},
    )
    from app.watchlist.target_rederive import rederive_watchlist_targets

    report = rederive_watchlist_targets(db_path=db_path, apply=True)

    assert report["counts"] == {"UNCHANGED": 1}
    entry = get_latest("SAME", db_path=db_path)
    assert entry.buy_price_target == 33.30


def test_assembly_error_is_reported_and_other_rows_proceed(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_row(db_path, ticker="BOOM")
    _seed_row(db_path, ticker="OKAY", buy_price_target=20.0)
    _patch_packet_chain(
        monkeypatch,
        {
            "BOOM": RuntimeError("no facts"),
            "OKAY": {"valuation": {"anchor_method": "epv", "valuation_anchor": 30.0}},
        },
    )
    from app.watchlist.target_rederive import rederive_watchlist_targets

    report = rederive_watchlist_targets(db_path=db_path, apply=True)

    assert report["counts"] == {"ERROR": 1, "RETARGET": 1}
    boom = next(row for row in report["rows"] if row["ticker"] == "BOOM")
    assert boom["disposition"] == "ERROR"
    assert "RuntimeError" in boom["detail"]
    okay = get_latest("OKAY", db_path=db_path)
    assert okay.buy_price_target == pytest.approx(24.90, abs=0.01)


def test_removed_rows_are_out_of_scope(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_row(db_path, ticker="GONE", status="REMOVED")
    _patch_packet_chain(monkeypatch, {})
    from app.watchlist.target_rederive import rederive_watchlist_targets

    report = rederive_watchlist_targets(db_path=db_path, apply=True)

    assert report["counts"] == {}
    assert report["rows"] == []
