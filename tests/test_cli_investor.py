from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from typer.testing import CliRunner

from app.db import init_db
from app.cli import app
from app.watchlist.contract import WatchlistEntry
from app.watchlist.store import add_or_update, add_price_snapshot, get_latest


runner = CliRunner()


def _init_temp_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


def _add_deploy_ready(
    db_path,
    *,
    ticker: str,
    buy_price_target: float,
    snapshot_price: float,
    conviction_grade: str = "ACTIONABLE",
    confidence: str = "MODERATE",
    source_run_id: str | None = None,
    checked_at: str = "2026-05-28T00:00:00Z",
    status: str = "DEPLOY_READY",
    with_snapshot: bool = True,
) -> None:
    add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            conviction_grade=conviction_grade,
            confidence=confidence,
            conviction_source="sector_final_decision",
            valuation_anchor_method="DCF",
            valuation_anchor_value=buy_price_target / 0.75,
            buy_price_target=buy_price_target,
            current_price_at_addition=snapshot_price,
            thesis_text=f"{ticker} is below the buy-price target.",
            falsifiers=[f"{ticker} loses its core customer base."],
            source_run_id=source_run_id or f"sector_run_{ticker}",
            source_sector="advertising_tech",
            added_at="2026-05-09T12:00:00Z",
            last_evaluated_at="2026-07-21T10:00:00+00:00",
        ),
        db_path=db_path,
    )
    entry = get_latest(ticker, db_path=db_path)
    assert entry is not None and entry.id is not None
    if with_snapshot:
        add_price_snapshot(
            entry.id,
            price=snapshot_price,
            checked_at=checked_at,
            source="yahoo",
            db_path=db_path,
        )


def _today_candidate_row(
    ticker: str = "READY",
    *,
    confidence: str = "HIGH",
    distance: float = -10.0,
) -> dict:
    return {
        "id": ord(ticker[0]),
        "ticker": ticker,
        "status": "DEPLOY_READY",
        "conviction_grade": "ACTIONABLE",
        "confidence": confidence,
        "latest_price": 100.0 + distance,
        "latest_price_checked_at": "2026-07-21T11:00:00+00:00",
        "buy_price_target": 100.0,
        "distance_from_buy_pct": distance,
        "last_evaluated_at": "2026-07-20T12:00:00+00:00",
        "source_sector": "industrial_tech",
        "cap_band": "small_cap",
        "adv_dollar_20d": 1_000_000.0,
        "capacity_class": "MODERATE",
        "falsifiers": [f"{ticker} falsifier"],
    }


def test_investor_buy_now_emits_imperative_for_deploy_ready(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _add_deploy_ready(db_path, ticker="CRTO", buy_price_target=32.19, snapshot_price=18.33)

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    assert "AT TARGET CRTO: $18.33 <= target $32.19" in result.output


def test_investor_buy_now_includes_buy_confirmed(monkeypatch, tmp_path):
    # BUY_CONFIRMED (catalyst-confirmed, price <= target) is strictly more
    # actionable than DEPLOY_READY and must appear in `investor buy-now`.
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _add_deploy_ready(
        db_path,
        ticker="CNFM",
        buy_price_target=32.19,
        snapshot_price=18.33,
        status="BUY_CONFIRMED",
    )

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    assert "AT TARGET CNFM: $18.33 <= target $32.19" in result.output


def test_investor_buy_now_sorts_most_through_target_first(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    # A: distance -50% (snapshot 50 vs target 100); B: distance -10% (snapshot 90 vs target 100)
    _add_deploy_ready(db_path, ticker="AAA", buy_price_target=100.0, snapshot_price=50.0)
    _add_deploy_ready(db_path, ticker="BBB", buy_price_target=100.0, snapshot_price=90.0)

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    assert result.output.index("AT TARGET AAA:") < result.output.index("AT TARGET BBB:")


def test_investor_buy_now_excludes_avoid_grade(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _add_deploy_ready(db_path, ticker="GOOD", buy_price_target=100.0, snapshot_price=50.0)
    _add_deploy_ready(
        db_path,
        ticker="AVOIDED",
        buy_price_target=100.0,
        snapshot_price=10.0,
        conviction_grade="AVOID",
    )

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    assert "AT TARGET GOOD:" in result.output
    assert "AVOIDED" not in result.output


def test_investor_memo_renders_decision_block(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _add_deploy_ready(db_path, ticker="CRTO", buy_price_target=32.19, snapshot_price=18.33)

    result = runner.invoke(app, ["investor", "memo", "CRTO"])

    assert result.exit_code == 0, result.output
    assert "## Decision" in result.output
    assert "- ACTION: Review at target" in result.output


def test_investor_help_lists_core_subcommands_and_today(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)

    result = runner.invoke(app, ["investor", "--help"])

    assert result.exit_code == 0, result.output
    assert "buy-now" in result.output
    assert "ideas" in result.output
    assert "memo" in result.output
    assert "watchlist" in result.output
    assert "today" in result.output


def test_render_investor_today_is_pure_capped_and_confidence_sorted():
    from app.cli_investor import render_investor_today
    from app.ops.data_health import DataHealth

    markdown = render_investor_today(
        queue_rows=[
            _today_candidate_row("A", confidence="MODERATE", distance=-50.0),
            _today_candidate_row("B", confidence="HIGH", distance=-10.0),
            _today_candidate_row("C", confidence="HIGH", distance=-20.0),
            _today_candidate_row("D", confidence="HIGH", distance=-5.0),
        ],
        health=DataHealth(
            checks=[{"name": "db_readable", "ok": True, "detail": "fixture"}]
        ),
        open_rows=[],
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
        price_max_age_days=5,
        adv_floor=250_000.0,
    )

    assert markdown.index("### C — REVIEW AT TARGET") < markdown.index(
        "### B — REVIEW AT TARGET"
    )
    assert markdown.index("### B — REVIEW AT TARGET") < markdown.index(
        "### D — REVIEW AT TARGET"
    )
    assert "### A — REVIEW AT TARGET" not in markdown
    assert "review trigger, not a buy signal." in markdown
    assert "BUY NOW" not in markdown
    assert len(markdown.splitlines()) <= 60


def test_investor_today_nonblocking_health_failure_is_amber_and_does_not_gate():
    from app.cli_investor import render_investor_today
    from app.ops.data_health import DataHealth

    markdown = render_investor_today(
        queue_rows=[_today_candidate_row()],
        health=DataHealth(
            checks=[
                {
                    "name": "backup_fresh",
                    "ok": False,
                    "detail": "backup status=FAILED (0d old)",
                }
            ]
        ),
        open_rows=[],
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
        price_max_age_days=5,
        adv_floor=250_000.0,
    )

    assert "- Data health: AMBER" in markdown
    assert "### READY — REVIEW AT TARGET" in markdown
    assert "- DATA_HEALTH_RED: 0" in markdown
    assert (
        "- OPERATIONS_WARNING: 1 — backup_fresh: backup status=FAILED (0d old)"
        in markdown
    )


def test_investor_today_blocking_health_failure_remains_red_and_gates():
    from app.cli_investor import render_investor_today
    from app.ops.data_health import DataHealth

    markdown = render_investor_today(
        queue_rows=[_today_candidate_row()],
        health=DataHealth(
            checks=[
                {
                    "name": "engine_db_exists",
                    "ok": False,
                    "detail": "missing books of record",
                }
            ],
            blocking=True,
        ),
        open_rows=[],
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
        price_max_age_days=5,
        adv_floor=250_000.0,
    )

    assert "- Data health: RED" in markdown
    assert "No review-ready candidates today." in markdown
    assert "### READY — REVIEW AT TARGET" not in markdown
    assert "- DATA_HEALTH_RED: 1 — engine_db_exists: missing books of record" in markdown
    assert "- OPERATIONS_WARNING: 0" in markdown


def test_investor_today_writes_exact_markdown_and_prints_stdout(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    # The price-freshness gate measures against the real clock, so the snapshot
    # timestamp has to track it. A pinned literal silently rots into a
    # STALE_PRICE blocker once it ages past the trigger window.
    _add_deploy_ready(
        db_path,
        ticker="TRUTH",
        buy_price_target=100.0,
        snapshot_price=90.0,
        confidence="HIGH",
        checked_at=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    )
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE watchlist SET adv_dollar_20d = 1000000.0, "
            "adv_dollar_60d = 1200000.0, adv_asof = '2026-07-21', "
            "capacity_class = 'MODERATE' WHERE ticker = 'TRUTH'"
        )
        conn.commit()
    finally:
        conn.close()
    output_path = tmp_path / "nested" / "investor_today.md"

    result = runner.invoke(
        app, ["investor", "today", "--output", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    assert "### TRUTH — REVIEW AT TARGET" in result.output
    assert "BUY NOW" not in result.output
    assert output_path.read_text(encoding="utf-8") == result.output
    assert result.output.endswith("\n")


def test_investor_today_zero_candidates_is_truthful(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="WAIT",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="LOW",
            source_run_id="wait_run",
            added_at="2026-07-21T00:00:00+00:00",
        ),
        db_path=db_path,
    )

    result = runner.invoke(app, ["investor", "today"])

    assert result.exit_code == 0, result.output
    assert "No review-ready candidates today." in result.output


def test_root_help_lists_investor(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "investor" in result.output


def test_is_snapshot_stale_boundary():
    # Pin the staleness boundary (7 trading days ~= 9 calendar days).
    from datetime import datetime, timezone

    from app.cli_investor import _is_snapshot_stale

    now = datetime(2026, 5, 15, tzinfo=timezone.utc)
    assert _is_snapshot_stale("2026-05-01T00:00:00Z", now=now) is True   # 14 days
    assert _is_snapshot_stale("2026-05-10T00:00:00Z", now=now) is False  # 5 days
    assert _is_snapshot_stale(None, now=now) is False


def test_investor_buy_now_emits_stale_caveat_for_old_snapshot(monkeypatch, tmp_path):
    # An old snapshot must surface the staleness caveat in buy-now output.
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _add_deploy_ready(
        db_path,
        ticker="OLD",
        buy_price_target=32.19,
        snapshot_price=18.33,
        checked_at="2025-01-01T00:00:00Z",
    )

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    assert "older than 7 trading days" in result.output


def test_investor_buy_now_caveats_when_no_snapshot(monkeypatch, tmp_path):
    # A DEPLOY_READY row with NO price snapshot renders the imperative off the
    # unverified addition price; surface a caveat instead of presenting it as live.
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _add_deploy_ready(
        db_path,
        ticker="NOSNAP",
        buy_price_target=32.19,
        snapshot_price=18.33,
        with_snapshot=False,
    )

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    assert "AT TARGET NOSNAP:" in result.output
    assert "no recent price snapshot" in result.output


def test_investor_buy_now_actionable_outranks_deeper_memoless_discount(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    # Memo-less name 50% through target; ACTIONABLE name only 10% through.
    _add_deploy_ready(
        db_path,
        ticker="DEEP",
        buy_price_target=100.0,
        snapshot_price=50.0,
        conviction_grade="WATCHLIST_ONLY",
    )
    _add_deploy_ready(db_path, ticker="MEMO", buy_price_target=100.0, snapshot_price=90.0)

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    actionable_header = result.output.index("## Actionable (conviction memo)")
    divider = result.output.index("## No Conviction Memo (price triggers only)")
    memo_line = result.output.index("AT TARGET MEMO: $90.00 <= target $100.00 (ACTIONABLE)")
    deep_line = result.output.index("AT TARGET DEEP: $50.00 <= target $100.00 (WATCHLIST_ONLY)")
    assert actionable_header < memo_line < divider < deep_line


def test_investor_buy_now_omits_divider_when_all_rows_actionable(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _add_deploy_ready(db_path, ticker="MEMO", buy_price_target=100.0, snapshot_price=90.0)

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    assert "## Actionable (conviction memo)" in result.output
    assert "## No Conviction Memo" not in result.output


def test_investor_buy_now_says_so_when_no_actionable_grade_at_target(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _add_deploy_ready(
        db_path,
        ticker="DEEP",
        buy_price_target=100.0,
        snapshot_price=50.0,
        conviction_grade="WATCHLIST_ONLY",
    )

    result = runner.invoke(app, ["investor", "buy-now"])

    assert result.exit_code == 0, result.output
    assert "No ACTIONABLE-grade names at target." in result.output
    assert result.output.index("No ACTIONABLE-grade names at target.") < result.output.index(
        "AT TARGET DEEP:"
    )


def test_investor_watchlist_hides_v2_screen_target_and_shows_provenance(
    monkeypatch, tmp_path
):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="SCREEN",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="sector_screen",
            valuation_anchor_method="DCF",
            valuation_anchor_value=100.0,
            buy_price_target=80.0,
            current_price_at_addition=60.0,
            source_run_id="autonomous_sector_v2_screen",
            source_sector="energy",
            added_at="2026-07-16T00:00:00Z",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )

    result = runner.invoke(app, ["investor", "watchlist"])

    assert result.exit_code == 0, result.output
    assert "| Pipeline | Disposition | Decision Basis | Validation |" in result.output
    row = next(line for line in result.output.splitlines() if "| SCREEN |" in line)
    assert "$60.00 | n/a | n/a | v2 | READY_FOR_UNDERWRITING | SCREEN | n/a" in row
    assert "$80.00" not in row


def test_investor_memo_renders_v2_screen_and_needs_data_as_research_only(
    monkeypatch, tmp_path
):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
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
            source_run_id="autonomous_sector_v2_ready_memo",
            source_sector="energy",
            added_at="2026-07-16T00:00:00Z",
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
            source_run_id="autonomous_sector_v2_needs_memo",
            source_sector="energy",
            added_at="2026-07-16T00:00:00Z",
            pipeline_version="v2",
            candidate_disposition="NEEDS_DATA",
            decision_basis="SCREEN",
        ),
    ):
        add_or_update(entry, db_path=db_path)

    for ticker, disposition, hidden_target in (
        ("READYV2", "READY_FOR_UNDERWRITING", "$80.00"),
        ("NEEDSV2", "NEEDS_DATA", "$70.00"),
    ):
        result = runner.invoke(app, ["investor", "memo", ticker])

        assert result.exit_code == 0, result.output
        assert "- ACTION: Research only" in result.output
        assert "- PIPELINE: v2" in result.output
        assert f"- CANDIDATE DISPOSITION: {disposition}" in result.output
        assert "- DECISION BASIS: SCREEN" in result.output
        assert "- SELECTION VALIDATION: n/a" in result.output
        assert "- INVESTABLE: no" in result.output
        assert "BUY PRICE TARGET" not in result.output
        assert "% TO TARGET" not in result.output
        assert "Wait for" not in result.output
        assert hidden_target not in result.output


def test_investor_journal_preview_hides_v2_screen_buy_target(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="SCREEN",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="sector_screen",
            valuation_anchor_method="DCF",
            valuation_anchor_value=100.0,
            buy_price_target=80.0,
            current_price_at_addition=60.0,
            source_run_id="autonomous_sector_v2_journal",
            source_sector="energy",
            added_at="2026-07-16T00:00:00Z",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )

    result = runner.invoke(
        app,
        [
            "investor",
            "journal",
            "SCREEN",
            "--action",
            "deferred",
            "--reason",
            "DEFERRED_PENDING_REVIEW",
            "--rationale",
            "Underwriting has not started.",
        ],
        input="n\n",
    )

    assert result.exit_code == 1, result.output
    assert "- price trigger: disabled until validated underwriting" in result.output
    assert "- PIPELINE: v2" in result.output
    assert "- CANDIDATE DISPOSITION: READY_FOR_UNDERWRITING" in result.output
    assert "- DECISION BASIS: SCREEN" in result.output
    assert "- INVESTABLE: no" in result.output
    assert "buy target" not in result.output.lower()
    assert "$80.00" not in result.output
