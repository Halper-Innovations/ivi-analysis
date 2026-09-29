from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.watchlist.contract import WatchlistEntry
from app.watchlist.digest import (
    NO_ACTIONABLE_AT_TARGET,
    NO_WINDOW_ACTIVITY,
    render_digest,
    write_digest,
)
from app.watchlist.store import add_or_update, add_price_snapshot, remove


runner = CliRunner()


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


def _now() -> datetime:
    return datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)


def _entry(
    db_path: Path,
    *,
    ticker: str,
    status: str = "ACTIVE",
    buy_price_target: float = 75.0,
    current_price_at_addition: float = 90.0,
    added_at: str = "2026-05-01T12:00:00+00:00",
    source_sector: str = "industrial_tech",
    source_run_id: str | None = None,
    conviction_grade: str = "WATCHLIST_ONLY",
    confidence: str | None = "MODERATE",
    conviction_source: str | None = "company_autonomy",
) -> int:
    return add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            conviction_grade=conviction_grade,
            confidence=confidence,
            conviction_source=conviction_source,
            valuation_anchor_method="DCF",
            valuation_anchor_value=100.0,
            buy_price_target=buy_price_target,
            current_price_at_addition=current_price_at_addition,
            thesis_text=f"{ticker} belongs on the watchlist.",
            key_risks=["Margin compression"],
            falsifiers=["Revenue decline accelerates"],
            open_questions=["Can working capital normalize?"],
            source_run_id=source_run_id or f"sector_run_{ticker}",
            source_sector=source_sector,
            added_at=added_at,
        ),
        db_path=db_path,
    )


def _history(
    db_path: Path,
    *,
    watchlist_id: int,
    field_name: str,
    old_value: str | None,
    new_value: str,
    source: str,
    changed_at: str = "2026-05-10T11:00:00+00:00",
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO watchlist_history (
                watchlist_id, changed_at, field_name, old_value, new_value, source, source_run_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (watchlist_id, changed_at, field_name, old_value, new_value, source, "test_run"),
        )
        conn.commit()
    finally:
        conn.close()


def test_render_digest_with_no_transitions_keeps_all_empty_sections(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="AAA", added_at="2026-05-01T12:00:00+00:00")

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert digest.index("## Review Today") < digest.index("## Newly Deploy-Ready")
    assert "## Newly Deploy-Ready\n(none in this window)" in digest
    assert "## Newly Contradicted\n(none in this window)" in digest
    assert "## Newly Added\n(none in this window)" in digest
    assert "## Recent Re-Evaluations\n(none in this window)" in digest
    assert "## Price Data Quality Issues\n(none in this window)" in digest
    assert "## Active Watchlist Summary" in digest
    assert "| ACTIVE | 1 |" in digest
    assert "| PRICE_DATA_SUSPECT | 0 |" in digest


def test_review_today_renders_first_and_excludes_price_suspect(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(
        db_path,
        ticker="AAA",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        conviction_source="sector_final_decision",
        buy_price_target=100.0,
        current_price_at_addition=90.0,
    )
    _entry(
        db_path,
        ticker="BAD",
        status="PRICE_DATA_SUSPECT",
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        conviction_source="company_autonomy",
        buy_price_target=100.0,
        current_price_at_addition=10.0,
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert digest.index("## Review Today") < digest.index("## Newly Deploy-Ready")
    assert (
        "| AAA | DEPLOY_READY | ACTIONABLE | HIGH | sector_final_decision | $90.00 | $100.00 | -10.0% | UNKNOWN_CAP | — | industrial_tech | n/a |"
        in digest
    )
    review_section = digest.split("## Review Today", 1)[1].split("## Newly Deploy-Ready", 1)[0]
    assert "| BAD |" not in review_section


def test_render_digest_lists_new_deploy_ready_transition(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _entry(db_path, ticker="AAA", status="DEPLOY_READY")
    _history(
        db_path,
        watchlist_id=row_id,
        field_name="status",
        old_value="ACTIVE",
        new_value="DEPLOY_READY",
        source="trigger",
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Newly Deploy-Ready" in digest
    assert "## Newly Deploy-Ready\n(none in this window)" in digest
    assert "| AAA | 2026-05-10T11:00:00+00:00 | ACTIVE | DEPLOY_READY |" not in digest


def test_digest_does_not_resurrect_authorized_history_behind_unaudited_latest_row(
    monkeypatch, tmp_path
):
    from app.watchlist import digest as digest_module

    db_path = _init_temp_db(monkeypatch, tmp_path)
    old_id = _entry(
        db_path,
        ticker="AAA",
        status="DEPLOY_READY",
        source_run_id="run_old_authorized",
    )
    _history(
        db_path,
        watchlist_id=old_id,
        field_name="status",
        old_value="ACTIVE",
        new_value="DEPLOY_READY",
        source="trigger",
    )
    _entry(
        db_path,
        ticker="AAA",
        status="DEPLOY_READY",
        source_run_id="run_new_unaudited",
        added_at="2026-05-10T11:30:00+00:00",
    )
    monkeypatch.setattr(
        digest_module,
        "watchlist_row_is_decision_eligible",
        lambda row, *_args, source_run_field="source_run_id", **_kwargs: (
            dict(row).get(source_run_field) == "run_old_authorized"
            and dict(row).get("ticker") == "AAA"
        ),
    )
    monkeypatch.setattr(
        "app.watchlist.store.watchlist_row_is_decision_eligible",
        lambda row, *_args, **_kwargs: (
            dict(row).get("source_run_id") == "run_old_authorized"
            and dict(row).get("ticker") == "AAA"
        ),
    )

    rendered = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "| AAA |" not in rendered
    assert "## Newly Deploy-Ready\n(none in this window)" in rendered


def test_render_digest_lists_new_contradicted_transition(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _entry(db_path, ticker="BBB", status="CONTRADICTED")
    _history(
        db_path,
        watchlist_id=row_id,
        field_name="status",
        old_value="ACTIVE",
        new_value="CONTRADICTED",
        source="reevaluation",
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Newly Contradicted" in digest
    assert "## Newly Contradicted\n(none in this window)" in digest
    assert "| BBB | 2026-05-10T11:00:00+00:00 | ACTIVE | CONTRADICTED |" not in digest


def test_render_digest_lists_new_entry(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="CCC", added_at="2026-05-10T10:00:00+00:00")

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Newly Added" in digest
    assert (
        "| CCC | ACTIVE | WATCHLIST_ONLY | MODERATE | company_autonomy | $75.00 | UNKNOWN_CAP | industrial_tech | 2026-05-10T10:00:00+00:00 |"
        in digest
    )


def test_removed_entry_added_inside_window_is_not_newly_added(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="DEL", added_at="2026-05-10T10:00:00+00:00")
    remove("DEL", "test cleanup", db_path=db_path)

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Newly Added\n(none in this window)" in digest
    # Mutable history rows have no exact source binding, so the auxiliary
    # removal section must fail closed.
    exits = digest.split("## Newly Removed (Universe Exits)", 1)[1].split("\n## ", 1)[0]
    assert "| DEL |" not in exits
    assert "| DEL |" not in digest.replace(exits, "")


def test_removed_entry_with_deploy_ready_transition_is_not_newly_deploy_ready(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _entry(
        db_path, ticker="DRP", status="DEPLOY_READY", added_at="2026-05-01T10:00:00+00:00"
    )
    _history(
        db_path,
        watchlist_id=row_id,
        field_name="status",
        old_value="ACTIVE",
        new_value="DEPLOY_READY",
        source="trigger",
    )
    remove("DRP", "test cleanup", db_path=db_path)

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Newly Deploy-Ready\n(none in this window)" in digest
    # Mutable history rows have no exact source binding.
    exits = digest.split("## Newly Removed (Universe Exits)", 1)[1].split("\n## ", 1)[0]
    assert "| DRP |" not in exits
    assert "| DRP |" not in digest.replace(exits, "")


def test_removed_entry_with_contradicted_transition_is_not_newly_contradicted(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _entry(
        db_path, ticker="CON", status="CONTRADICTED", added_at="2026-05-01T10:00:00+00:00"
    )
    _history(
        db_path,
        watchlist_id=row_id,
        field_name="status",
        old_value="ACTIVE",
        new_value="CONTRADICTED",
        source="reevaluation",
    )
    remove("CON", "test cleanup", db_path=db_path)

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Newly Contradicted\n(none in this window)" in digest
    # Mutable history rows have no exact source binding.
    exits = digest.split("## Newly Removed (Universe Exits)", 1)[1].split("\n## ", 1)[0]
    assert "| CON |" not in exits
    assert "| CON |" not in digest.replace(exits, "")


def test_entry_added_before_window_is_not_newly_added(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="OLD", added_at="2026-05-01T10:00:00+00:00")

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Newly Added\n(none in this window)" in digest
    newly_added_section = digest.split("## Newly Added", 1)[1].split("## Recent Re-Evaluations", 1)[
        0
    ]
    assert "| OLD | ACTIVE | WATCHLIST_ONLY |" not in newly_added_section


def test_top_active_entries_sorted_by_distance_to_buy(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    aaa_id = _entry(db_path, ticker="AAA", buy_price_target=100.0, current_price_at_addition=130.0)
    bbb_id = _entry(db_path, ticker="BBB", buy_price_target=100.0, current_price_at_addition=110.0)
    add_price_snapshot(
        aaa_id, price=125.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )
    add_price_snapshot(
        bbb_id, price=95.0, checked_at="2026-05-10T11:00:00+00:00", source="stooq", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    bbb_index = digest.index("| BBB | ACTIVE | $95.00 | $100.00 | -5.0% |")
    aaa_index = digest.index("| AAA | ACTIVE | $125.00 | $100.00 | 25.0% |")
    assert bbb_index < aaa_index


def test_buy_now_section_renders_before_review_today(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _entry(
        db_path,
        ticker="AAA",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        buy_price_target=100.0,
        current_price_at_addition=90.0,
    )
    add_price_snapshot(
        row_id, price=90.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## At Buy Target (Review)" in digest
    assert digest.index("## At Buy Target (Review)") < digest.index("## Review Today")


def test_buy_now_orders_most_through_target_first(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    a_id = _entry(
        db_path,
        ticker="AAA",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    b_id = _entry(
        db_path,
        ticker="BBB",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    add_price_snapshot(
        a_id, price=50.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )
    add_price_snapshot(
        b_id, price=90.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    buy_now_section = digest.split("## At Buy Target (Review)", 1)[1].split("## Review Today", 1)[0]
    a_index = buy_now_section.index("| AAA | ACTIONABLE | $50.00 | $100.00 | -50.0% |")
    b_index = buy_now_section.index("| BBB | ACTIONABLE | $90.00 | $100.00 | -10.0% |")
    assert a_index < b_index


def test_buy_now_includes_imperative_line(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _entry(
        db_path,
        ticker="CRTO",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        buy_price_target=32.19,
        current_price_at_addition=999.0,
    )
    add_price_snapshot(
        row_id, price=18.33, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert (
        "- AT TARGET CRTO: $18.33 <= target $32.19 (ACTIONABLE)"
        " — review trigger, not a buy signal\n"
        "  - (price basis: snapshot 2026-05-10 via yahoo, 0d old)\n"
        "  - (capacity: ADV_UNKNOWN — no volume history; tradeability unassessed)" in digest
    )


def test_buy_now_excludes_avoid_grade(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _entry(
        db_path,
        ticker="AVD",
        status="DEPLOY_READY",
        conviction_grade="AVOID",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    add_price_snapshot(
        row_id, price=50.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    buy_now_section = digest.split("## At Buy Target (Review)", 1)[1].split("## Review Today", 1)[0]
    assert "AVD" not in buy_now_section
    assert "AT TARGET AVD" not in digest


def test_within_reach_section_excludes_deploy_ready_rows(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    dr_id = _entry(
        db_path,
        ticker="DRX",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    act_id = _entry(
        db_path,
        ticker="ACT",
        status="ACTIVE",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    add_price_snapshot(
        dr_id, price=50.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )
    add_price_snapshot(
        act_id, price=110.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Within Reach (above target)" in digest
    within_reach_section = digest.split("## Within Reach (above target)", 1)[1]
    assert "| DRX |" not in within_reach_section
    assert "| ACT | ACTIVE | $110.00 | $100.00 | 10.0% |" in within_reach_section


def test_buy_now_empty_when_no_deploy_ready(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(
        db_path,
        ticker="ACT",
        status="ACTIVE",
        buy_price_target=100.0,
        current_price_at_addition=110.0,
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert f"## At Buy Target (Review)\n{NO_WINDOW_ACTIVITY}" in digest


def test_price_data_quality_issues_lists_suspect_entries(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(
        db_path,
        ticker="BKNG",
        status="PRICE_DATA_SUSPECT",
        buy_price_target=3250.15,
        source_sector="internet_services",
    )
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE watchlist SET status_reason = ? WHERE ticker = ?",
            (
                "PRICE_DATA_SUSPECT:latest_price=154.13:historical_fiscal_year_median=3000.00",
                "BKNG",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Price Data Quality Issues" in digest
    assert (
        "| BKNG | PRICE_DATA_SUSPECT | $3250.15 | internet_services | "
        "PRICE_DATA_SUSPECT:latest_price=154.13:historical_fiscal_year_median=3000.00 |"
    ) in digest
    assert "| PRICE_DATA_SUSPECT | 1 |" in digest


def test_recent_reevaluations_excludes_no_new_evidence(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    aaa_id = _entry(db_path, ticker="AAA")
    bbb_id = _entry(db_path, ticker="BBB")
    _history(
        db_path,
        watchlist_id=aaa_id,
        field_name="reevaluation",
        old_value=None,
        new_value="NO_NEW_EVIDENCE",
        source="reevaluation",
    )
    _history(
        db_path,
        watchlist_id=bbb_id,
        field_name="reevaluation",
        old_value=None,
        new_value="CONTRADICTED",
        source="reevaluation",
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Recent Re-Evaluations\n(none in this window)" in digest
    assert "| BBB | 2026-05-10T11:00:00+00:00 | CONTRADICTED |" not in digest
    assert "NO_NEW_EVIDENCE" not in digest


@pytest.mark.financial_integrity_contract
def test_digest_suppresses_unfingerprinted_held_exit_rows(monkeypatch, tmp_path):
    from app.holdings import ensure_holdings_schema
    from tests.test_classic_postwrite_authorization import _baseline_manifest

    _baseline_manifest(monkeypatch, tmp_path)
    db_path = _init_temp_db(monkeypatch, tmp_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_holdings_schema(conn)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(exit_signals)").fetchall()}
    assert {"holding_id", "ticker", "signal_type", "evidence_json", "status"} <= columns
    conn.execute(
        """
        INSERT INTO exit_signals(
            holding_id, ticker, signal_type, as_of_date, detected_at,
            evidence_json, status
        ) VALUES (
            1, 'FORGED_EXIT', 'THESIS_BREAK',
            '2026-05-10', '2026-05-10T11:00:00+00:00',
            '{"fabricated":true}', 'OPEN'
        )
        """
    )
    conn.commit()
    conn.close()

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "FORGED_EXIT" not in digest


def test_write_digest_writes_to_explicit_path(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="AAA")
    output_path = tmp_path / "digest.md"

    result = write_digest(days_back=1, output_path=output_path, now=_now(), db_path=db_path)

    assert result.path == output_path
    assert output_path.read_text(encoding="utf-8") == result.markdown
    assert result.markdown.startswith("# IVI Watchlist Daily Digest")


def test_cli_digest_check_prices_first_runs_before_render(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="AAA")
    output_path = tmp_path / "daily.md"
    calls: list[str] = []
    from app.watchlist import digest as digest_module

    original_render = digest_module.render_digest

    def fake_check(*args, **kwargs):
        calls.append("trigger")
        return []

    def fake_render(*args, **kwargs):
        calls.append("render")
        return original_render(*args, **kwargs)

    monkeypatch.setattr("app.watchlist.digest.check_watchlist_triggers", fake_check)
    monkeypatch.setattr("app.watchlist.digest.render_digest", fake_render)

    result = runner.invoke(
        app, ["watchlist", "digest", "--check-prices-first", "--output", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    assert calls == ["trigger", "render"]
    assert result.output == output_path.read_text(encoding="utf-8")
    assert result.output.startswith("# IVI Watchlist Daily Digest")


def test_buy_now_actionable_grade_outranks_deeper_memoless_discount(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    deep_id = _entry(
        db_path,
        ticker="DEEP",
        status="DEPLOY_READY",
        conviction_grade="WATCHLIST_ONLY",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    memo_id = _entry(
        db_path,
        ticker="MEMO",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    add_price_snapshot(
        deep_id, price=50.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )
    add_price_snapshot(
        memo_id, price=90.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    section = digest.split("## At Buy Target (Review)", 1)[1].split("## Review Today", 1)[0]
    memo_idx = section.index("| MEMO | ACTIONABLE | $90.00 | $100.00 | -10.0% |")
    deep_idx = section.index("| DEEP | WATCHLIST_ONLY | $50.00 | $100.00 | -50.0% |")
    assert memo_idx < deep_idx


def test_digest_ends_with_current_actionable_queue(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    row_id = _entry(
        db_path,
        ticker="MEMO",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    add_price_snapshot(
        row_id, price=90.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    footer = digest.split("## Current Actionable Queue", 1)[1]
    assert (
        "AT TARGET MEMO: $90.00 <= target $100.00 (ACTIONABLE)"
        " — review trigger, not a buy signal" in footer
    )
    # The queue is the digest's closing section: nothing renders after it.
    assert "##" not in footer


def test_actionable_queue_excludes_memoless_rows_and_reports_empty(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    deep_id = _entry(
        db_path,
        ticker="DEEP",
        status="DEPLOY_READY",
        conviction_grade="WATCHLIST_ONLY",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    add_price_snapshot(
        deep_id, price=50.0, checked_at="2026-05-10T11:00:00+00:00", source="yahoo", db_path=db_path
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    footer = digest.split("## Current Actionable Queue", 1)[1]
    assert NO_ACTIONABLE_AT_TARGET in footer
    assert "AT TARGET DEEP" not in footer


def test_actionable_queue_caps_at_three_and_counts_blocked(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    # Four presentable ACTIONABLE names at target, increasing distance.
    for offset, ticker in enumerate(["AAAA", "BBBB", "CCCC", "DDDD"]):
        row_id = _entry(
            db_path,
            ticker=ticker,
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            buy_price_target=100.0,
            current_price_at_addition=999.0,
        )
        add_price_snapshot(
            row_id,
            price=50.0 + offset,
            checked_at="2026-05-10T11:00:00+00:00",
            source="yahoo",
            db_path=db_path,
        )
    blocked_id = _entry(
        db_path,
        ticker="EEEE",
        status="DEPLOY_READY",
        conviction_grade="ACTIONABLE",
        buy_price_target=100.0,
        current_price_at_addition=999.0,
    )
    add_price_snapshot(
        blocked_id,
        price=55.0,
        checked_at="2026-05-10T11:00:00+00:00",
        source="yahoo",
        db_path=db_path,
    )
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE watchlist SET event_pending = 'EVENT_PENDING:MERGER' WHERE id = ?",
            (blocked_id,),
        )
        conn.commit()
    finally:
        conn.close()

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    footer = digest.split("## Current Actionable Queue", 1)[1]
    assert "AT TARGET AAAA" in footer
    assert "AT TARGET BBBB" in footer
    assert "AT TARGET CCCC" in footer
    assert "AT TARGET DDDD" not in footer
    assert "AT TARGET EEEE" not in footer
    assert "(+1 more at target; 1 blocked pending event review — see sections above)" in footer


def test_render_compact_signal_table_orders_live_signals_first():
    from app.watchlist.digest import render_compact_signal_table

    rows = [
        {
            "ticker": "WAITER",
            "status": "ACTIVE",
            "presented_status": "ACTIVE",
            "conviction_grade": "ACTIONABLE",
            "confidence": "HIGH",
            "latest_price": 100.0,
            "buy_price_target": 80.0,
            "distance_from_buy_pct": 25.0,
        },
        {
            "ticker": "BLOCKED",
            "status": "DEPLOY_READY",
            "presented_status": "EVENT_PENDING",
            "conviction_grade": "ACTIONABLE",
            "confidence": "LOW",
            "latest_price": 70.0,
            "buy_price_target": 80.0,
            "distance_from_buy_pct": -12.5,
        },
        {
            "ticker": "BUYME",
            "status": "DEPLOY_READY",
            "presented_status": "DEPLOY_READY",
            "conviction_grade": "WATCHLIST_ONLY",
            "confidence": "MODERATE",
            "latest_price": 60.0,
            "buy_price_target": 80.0,
            "distance_from_buy_pct": -25.0,
            "cap_band": "small_cap",
            "source_sector": "industrials",
        },
    ]

    lines = render_compact_signal_table(rows)

    assert lines[0] == (
        "| Signal | Ticker | Conviction | Confidence | Price | Buy Target "
        "| Distance From Buy | Band | Source Sector | Status | Disposition "
        "| Decision Basis | Validation |"
    )
    assert lines[2] == (
        "| REVIEW AT TARGET | BUYME | WATCHLIST_ONLY | MODERATE | $60.00 | $80.00 "
        "| -25.0% | small_cap | industrials | DEPLOY_READY | n/a | n/a | n/a |"
    )
    assert lines[3].startswith("| EVENT HOLD | BLOCKED | ACTIONABLE | LOW |")
    assert lines[4].startswith("| WAIT | WAITER | ACTIONABLE | HIGH |")


def test_compact_signal_table_keeps_v2_screen_rows_noninvestable():
    from app.watchlist.digest import render_compact_signal_table

    lines = render_compact_signal_table(
        [
            {
                "ticker": "SCREEN",
                "status": "ACTIVE",
                "presented_status": "ACTIVE",
                "conviction_grade": "WATCHLIST_ONLY",
                "confidence": "MODERATE",
                "latest_price": 60.0,
                "buy_price_target": 80.0,
                "distance_from_buy_pct": -25.0,
                "pipeline_version": "v2",
                "candidate_disposition": "READY_FOR_UNDERWRITING",
                "decision_basis": "SCREEN",
                "selection_validation_status": None,
            }
        ]
    )

    assert lines[2] == (
        "| SCREEN RESEARCH | SCREEN | WATCHLIST_ONLY | MODERATE | $60.00 | n/a "
        "| n/a | UNKNOWN_CAP | n/a | ACTIVE | READY_FOR_UNDERWRITING | SCREEN | n/a |"
    )
    assert "REVIEW AT TARGET" not in lines[2]
    assert "$80.00" not in lines[2]


def test_digest_includes_signal_board_section(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _entry(db_path, ticker="AAA", status="DEPLOY_READY", conviction_grade="ACTIONABLE")
    _entry(db_path, ticker="BBB", status="ACTIVE")

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    assert "## Signal Board" in digest
    assert "| REVIEW AT TARGET | AAA | ACTIONABLE |" in digest
    assert "| WAIT | BBB | WATCHLIST_ONLY |" in digest
    # The closer must remain the final section.
    assert digest.rstrip().split("## ")[-1].startswith("Current Actionable Queue")


def test_full_digest_hides_targets_for_persisted_v2_research_rows(monkeypatch, tmp_path):
    from app.watchlist.store import record_trigger_status_change

    db_path = _init_temp_db(monkeypatch, tmp_path)
    for entry in (
        WatchlistEntry(
            ticker="READYV2",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="sector_screen",
            valuation_anchor_method="DCF",
            valuation_anchor_value=100.0,
            buy_price_target=80.0,
            current_price_at_addition=60.0,
            source_run_id="autonomous_sector_v2_ready_digest",
            source_sector="energy",
            added_at="2026-05-10T08:00:00+00:00",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        WatchlistEntry(
            ticker="NEEDSV2",
            status="UNCERTAIN",
            conviction_grade="DATA_INCOMPLETE",
            confidence="LOW",
            conviction_source="sector_screen",
            valuation_anchor_method="DCF",
            valuation_anchor_value=90.0,
            buy_price_target=70.0,
            current_price_at_addition=55.0,
            source_run_id="autonomous_sector_v2_needs_digest",
            source_sector="energy",
            added_at="2026-05-10T08:00:00+00:00",
            pipeline_version="v2",
            candidate_disposition="NEEDS_DATA",
            decision_basis="SCREEN",
        ),
    ):
        add_or_update(entry, db_path=db_path)
    record_trigger_status_change(
        "NEEDSV2",
        status="PRICE_DATA_SUSPECT",
        reason="PRICE_DATA_SUSPECT:test fixture",
        db_path=db_path,
    )

    digest = render_digest(days_back=1, now=_now(), db_path=db_path)

    research_lines = [
        line for line in digest.splitlines() if "READYV2" in line or "NEEDSV2" in line
    ]
    assert research_lines
    assert all("$80.00" not in line for line in research_lines)
    assert all("$70.00" not in line for line in research_lines)
    assert all("-25.0%" not in line for line in research_lines)
    assert "| SCREEN RESEARCH | READYV2 |" in digest
    assert "| READYV2 | READY_FOR_UNDERWRITING | SCREEN | n/a | no |" in digest
    assert "| NEEDSV2 | NEEDS_DATA | SCREEN | n/a | no |" in digest
    assert "## Within Reach (above target)\n(none in this window)" in digest
