"""Disposition ledger: at-target sync, typed closes, journal wiring,
EVENT_PENDING enforcement, event-disposal ledger rows."""

from __future__ import annotations

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
    # Existing ledger tests model an already-audited decision book. Dedicated
    # regressions below replace this stub to exercise fail-closed eligibility.
    from app.watchlist import dispositions

    monkeypatch.setattr(
        dispositions,
        "watchlist_row_is_decision_eligible",
        lambda row, *_args, **_kwargs: (
            bool(str(dict(row).get("source_run_id") or "").strip())
            and bool(str(dict(row).get("ticker") or "").strip())
        ),
    )
    return db_path


def _add_row(
    db_path,
    ticker="AAA",
    status="DEPLOY_READY",
    grade="ACTIONABLE",
    event_pending=None,
    source_run_id="run1",
):
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update

    row_id = add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            conviction_grade=grade,
            buy_price_target=10.0,
            source_run_id=source_run_id,
            added_at="2026-07-01T00:00:00+00:00",
        ),
        db_path=db_path,
    )
    if event_pending:
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE watchlist SET event_pending = ? WHERE id = ?", (event_pending, row_id))
        conn.commit()
        conn.close()
    return row_id


def test_sync_opens_one_disposition_per_at_target_name(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.dispositions import open_dispositions, sync_at_target_dispositions

    _add_row(db_path, "AAA", "DEPLOY_READY")
    _add_row(db_path, "BBB", "ACTIVE")
    _add_row(db_path, "EVT", "DEPLOY_READY", event_pending="EVENT_PENDING:MERGER")
    _add_row(db_path, "AVD", "DEPLOY_READY", grade="AVOID")

    first = sync_at_target_dispositions(db_path)
    second = sync_at_target_dispositions(db_path)

    assert first["opened"] == 1
    assert second["opened"] == 0
    rows = open_dispositions(db_path)
    assert [row["ticker"] for row in rows] == ["AAA"]
    assert rows[0]["kind"] == "AT_TARGET"
    assert rows[0]["source_state"] == "CURRENT"
    assert rows[0]["is_current_actionable"] is True


def test_superseded_open_disposition_is_labeled_and_new_authority_opens(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.dispositions import (
        close_disposition,
        open_dispositions,
        sync_at_target_dispositions,
    )

    old_watchlist_id = _add_row(
        db_path,
        "AAA",
        "DEPLOY_READY",
        grade="ACTIONABLE",
        source_run_id="run_old",
    )
    assert sync_at_target_dispositions(db_path)["opened"] == 1
    new_watchlist_id = _add_row(
        db_path,
        "AAA",
        "DEPLOY_READY",
        grade="WATCHLIST_ONLY",
        source_run_id="run_new",
    )
    assert sync_at_target_dispositions(db_path)["opened"] == 1

    rows = open_dispositions(db_path)

    assert [
        (
            row["watchlist_id"],
            row["source_state"],
            row["is_current_actionable"],
        )
        for row in rows
    ] == [
        (old_watchlist_id, "SUPERSEDED_STALE_SOURCE", False),
        (new_watchlist_id, "CURRENT", False),
    ]
    assert rows[0]["resolution_required"] == (
        "Explicit PASSED or DEFERRED closure required; the source assessment "
        "is no longer authoritative."
    )

    with pytest.raises(ValueError, match="PASSED or DEFERRED"):
        close_disposition(
            ticker="AAA",
            status="ACTED",
            operator="owner",
            reason_code="ENTERED_POSITION",
            rationale="Attempted stale-source action.",
            disposition_id=rows[0]["id"],
            db_path=db_path,
        )

    closed = close_disposition(
        ticker="AAA",
        status="DEFERRED",
        operator="owner",
        reason_code="DEFERRED_PENDING_REVIEW",
        rationale="The source assessment was superseded.",
        disposition_id=rows[0]["id"],
        db_path=db_path,
    )
    assert closed["status"] == "DEFERRED"


def test_unusable_manifest_blocks_new_and_current_actionable_dispositions(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.lineage import (
        watchlist_row_is_decision_eligible as central_row_eligibility,
    )
    from app.watchlist import dispositions

    monkeypatch.setattr(
        dispositions,
        "watchlist_row_is_decision_eligible",
        lambda _row, *_args, **_kwargs: True,
    )
    _add_row(db_path, "AAA", "DEPLOY_READY", source_run_id="run_audited")
    assert dispositions.sync_at_target_dispositions(db_path)["opened"] == 1

    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(tmp_path / "missing-financial-integrity-manifest.json"),
    )
    monkeypatch.setattr(
        dispositions,
        "watchlist_row_is_decision_eligible",
        central_row_eligibility,
    )
    _add_row(db_path, "BBB", "DEPLOY_READY", source_run_id="run_unaudited")

    assert dispositions.sync_at_target_dispositions(db_path)["opened"] == 0
    rows = dispositions.open_dispositions(db_path)
    assert [row["ticker"] for row in rows] == ["AAA"]
    assert rows[0]["source_state"] == "CURRENT_FINANCIAL_INTEGRITY_BLOCKED"
    assert rows[0]["financial_integrity_eligible"] is False
    assert rows[0]["is_current_actionable"] is False


def test_latest_invalid_source_does_not_resurrect_older_actionable_state(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist import dispositions

    monkeypatch.setattr(
        dispositions,
        "watchlist_row_is_decision_eligible",
        lambda row, *_args, **_kwargs: dict(row).get("source_run_id") == "run_old",
    )
    old_watchlist_id = _add_row(
        db_path,
        "AAA",
        "DEPLOY_READY",
        grade="ACTIONABLE",
        source_run_id="run_old",
    )
    assert dispositions.sync_at_target_dispositions(db_path)["opened"] == 1

    new_watchlist_id = _add_row(
        db_path,
        "AAA",
        "DEPLOY_READY",
        grade="ACTIONABLE",
        source_run_id="run_invalid_new",
    )

    assert dispositions.sync_at_target_dispositions(db_path)["opened"] == 0
    rows = dispositions.open_dispositions(db_path)
    assert len(rows) == 1
    assert rows[0]["watchlist_id"] == old_watchlist_id
    assert rows[0]["current_watchlist_id"] == new_watchlist_id
    assert rows[0]["source_state"] == "SUPERSEDED_STALE_SOURCE"
    assert rows[0]["financial_integrity_eligible"] is False
    assert rows[0]["is_current_actionable"] is False


def test_acted_close_rejects_noneligible_current_source(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist import dispositions

    monkeypatch.setattr(
        dispositions,
        "watchlist_row_is_decision_eligible",
        lambda _row, *_args, **_kwargs: True,
    )
    _add_row(db_path, "AAA", "DEPLOY_READY", source_run_id="run_audited")
    assert dispositions.sync_at_target_dispositions(db_path)["opened"] == 1
    open_row = dispositions.open_dispositions(db_path)[0]

    monkeypatch.setattr(
        dispositions,
        "watchlist_row_is_decision_eligible",
        lambda _row, *_args, **_kwargs: False,
    )
    with pytest.raises(ValueError, match="financial-integrity decision-eligible"):
        dispositions.close_disposition(
            ticker="AAA",
            status="ACTED",
            operator="owner",
            reason_code="ENTERED_POSITION",
            rationale="Would act if the source were eligible.",
            disposition_id=open_row["id"],
            db_path=db_path,
        )

    conn = sqlite3.connect(str(db_path))
    status_row = conn.execute(
        "SELECT status FROM dispositions WHERE id = ?", (open_row["id"],)
    ).fetchone()
    assert status_row[0] == "OPEN"
    assert conn.execute("SELECT COUNT(*) FROM ticker_outcomes").fetchone()[0] == 0
    conn.close()


def test_close_writes_journal_row_and_is_append_only(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.dispositions import (
        close_disposition,
        journal_entry_count,
        open_dispositions,
        sync_at_target_dispositions,
    )

    _add_row(db_path, "AAA", "DEPLOY_READY")
    sync_at_target_dispositions(db_path)

    result = close_disposition(
        ticker="AAA",
        status="PASSED",
        operator="owner",
        reason_code="PASSED_VALUATION",
        rationale="Anchor confidence too low at this print.",
        db_path=db_path,
    )

    assert result["status"] == "PASSED"
    assert result["journal"] is not None
    assert result["journal"]["decision"] == "PASS"
    assert open_dispositions(db_path) == []
    assert journal_entry_count(db_path) == 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    outcome = conn.execute(
        "SELECT decision, run_id, notes FROM ticker_outcomes WHERE ticker='AAA'"
    ).fetchone()
    conn.close()
    assert outcome["run_id"] == "journal_live"
    assert outcome["decision"] == "PASS"

    # Re-deciding the same disposition id is refused (append-only ledger).
    with pytest.raises(ValueError, match="append-only"):
        close_disposition(
            ticker="AAA",
            status="ACTED",
            operator="owner",
            reason_code="ENTERED_POSITION",
            rationale="changed my mind",
            disposition_id=result["id"],
            db_path=db_path,
        )


def test_acted_close_refused_while_event_pending(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.dispositions import close_disposition

    _add_row(db_path, "EVT", "DEPLOY_READY", event_pending="EVENT_PENDING:MERGER")

    with pytest.raises(ValueError, match="EVENT_PENDING:MERGER"):
        close_disposition(
            ticker="EVT",
            status="ACTED",
            operator="owner",
            reason_code="ENTERED_POSITION",
            rationale="buying anyway",
            db_path=db_path,
        )
    # PASSED is allowed — declining a name never needs the event resolved.
    result = close_disposition(
        ticker="EVT",
        status="PASSED",
        operator="owner",
        reason_code="PASSED_EVENT_RISK",
        rationale="open merger paper; not paying up for deal risk",
        db_path=db_path,
    )
    assert result["status"] == "PASSED"


def test_manual_journal_without_open_disposition(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.dispositions import close_disposition

    _add_row(db_path, "MNL", "ACTIVE")
    result = close_disposition(
        ticker="MNL",
        status="DEFERRED",
        operator="owner",
        reason_code="DEFERRED_PENDING_REVIEW",
        rationale="waiting on the 10-K",
        db_path=db_path,
    )
    assert result["kind"] == "MANUAL"
    assert result["journal"]["decision"] == "WATCH"


def test_typed_reason_and_operator_required(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.dispositions import close_disposition

    _add_row(db_path, "AAA", "DEPLOY_READY")
    with pytest.raises(ValueError, match="reason_code"):
        close_disposition(
            ticker="AAA",
            status="PASSED",
            operator="owner",
            reason_code="FREEFORM",
            rationale="x",
            db_path=db_path,
        )
    with pytest.raises(ValueError, match="operator"):
        close_disposition(
            ticker="AAA",
            status="PASSED",
            operator="  ",
            reason_code="PASSED_VALUATION",
            rationale="x",
            db_path=db_path,
        )


def test_event_disposal_lands_in_ledger(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.dispositions import record_event_disposal

    row = record_event_disposal(
        ticker="AAA",
        event_id=42,
        operator="owner",
        reason_code="EVENT_REVIEWED",
        note="merger paper reviewed; terms immaterial",
        db_path=db_path,
    )
    assert row["kind"] == "EVENT_DISPOSAL"
    assert row["status"] == "PASSED"
    assert row["event_id"] == 42
    assert row["operator"] == "owner"


def test_digest_surfaces_open_dispositions(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.dispositions import sync_at_target_dispositions
    from app.watchlist.digest import render_digest

    _add_row(db_path, "AAA", "DEPLOY_READY")
    sync_at_target_dispositions(db_path)

    digest = render_digest(days_back=1, db_path=db_path)
    assert "## Open Dispositions (decision required)" in digest
    section = digest.split("## Open Dispositions (decision required)", 1)[1].split("\n## ", 1)[0]
    assert "AAA" in section
