from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from app.db import init_db
from app.cli import app
from app.market.price_provider import PriceSnapshot
from app.watchlist.contract import WatchlistEntry
from app.watchlist.digest import DigestWriteResult
from app.watchlist.store import add_or_update, add_price_snapshot, get_history, get_latest
from app.watchlist.triggers import WatchlistTriggerResult


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


def _seed_watchlist(db_path):
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="company_autonomy",
            valuation_anchor_method="DCF",
            valuation_anchor_value=120.0,
            buy_price_target=90.0,
            current_price_at_addition=100.0,
            thesis_text="AAA is good but needs a better entry price.",
            key_risks=["Gross margin declined 310 bps."],
            falsifiers=["Revenue contraction persists."],
            open_questions=["Can backlog convert to revenue?"],
            source_run_id="sector_run_1",
            source_sector="industrial_tech",
            added_at="2026-05-08T12:00:00Z",
        ),
        db_path=db_path,
    )
    add_or_update(
        WatchlistEntry(
            ticker="BBB",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="LOW",
            conviction_source="sector_final_decision",
            valuation_anchor_method="EPV",
            valuation_anchor_value=80.0,
            buy_price_target=60.0,
            current_price_at_addition=55.0,
            thesis_text="BBB is below the buy-price target.",
            key_risks=["Cash conversion cycle lengthened to 98 days."],
            falsifiers=["Free cash flow turns negative."],
            open_questions=["Is working-capital drag temporary?"],
            source_run_id="sector_run_2",
            source_sector="industrial_tech",
            added_at="2026-05-09T12:00:00Z",
        ),
        db_path=db_path,
    )


def _insert_refresh_filing(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1234567890",
                "AAA",
                "0000000000-26-000010",
                "10-Q",
                "2026-05-03",
                "2026-03-31",
                "https://example.com/0000000000-26-000010.htm",
                None,
                "parsed",
                "2026-05-03T12:00:00Z",
                "2026-05-03T12:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _write_counterfactual_artifact(
    data_dir: Path,
    *,
    run_id: str,
    ticker: str,
    base_return: float,
    hard_blockers: list[str],
    confidence_caps: list[str],
    downside_return: float = -0.02,
) -> Path:
    run_dir = data_dir / "outputs" / "runs" / "autonomous_sector" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "autonomous_sector_run.json"
    payload = {
        "run_id": run_id,
        "relative_ranking": [
            {
                "ticker": ticker,
                "audit_status": "BLOCKED" if hard_blockers else "WATCHLIST_ONLY",
                "best_base_annualized_return": base_return,
                "downside_annualized_return": downside_return,
                "hard_blockers": hard_blockers,
                "confidence_caps": confidence_caps,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_watchlist_list_outputs_latest_entries(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)

    result = runner.invoke(app, ["watchlist", "list", "--status", "DEPLOY_READY"])

    assert result.exit_code == 0, result.output
    assert (
        "| Ticker | Status | Conviction | Confidence | Conviction Source | Scan Family | Valuation Anchor |"
        in result.output
    )
    assert (
        "| BBB | DEPLOY_READY | ACTIONABLE | LOW | sector_final_decision | normal | EPV $80.00 | $60.00 | $55.00 | -8.3% | industrial_tech | 2026-05-09T12:00:00Z |"
        in result.output
    )
    assert "AAA" not in result.output


def test_watchlist_list_scan_family_filter_preserves_family_latest(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            conviction_source="pearl_scan",
            scan_family="pearl",
            valuation_anchor_method="pearl_p15",
            valuation_anchor_value=110.0,
            buy_price_target=110.0,
            current_price_at_addition=90.0,
            thesis_text="Deep-pass row for the same ticker.",
            source_run_id="pearl_scan_test",
            source_sector="pearl",
            added_at="2026-05-10T12:00:00Z",
        ),
        db_path=db_path,
    )

    normal = runner.invoke(app, ["watchlist", "list", "--scan-family", "normal"])
    pearl = runner.invoke(app, ["watchlist", "list", "--scan-family", "pearl"])

    assert normal.exit_code == 0, normal.output
    assert (
        "| AAA | ACTIVE | WATCHLIST_ONLY | MODERATE | company_autonomy | normal |" in normal.output
    )
    assert pearl.exit_code == 0, pearl.output
    assert "| AAA | DEPLOY_READY | ACTIONABLE | HIGH | pearl_scan | pearl |" in pearl.output
    assert "BBB" not in pearl.output


def test_watchlist_show_outputs_markdown_detail(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)

    result = runner.invoke(app, ["watchlist", "show", "AAA"])

    assert result.exit_code == 0, result.output
    assert "# AAA Watchlist Entry" in result.output
    assert "- Confidence: MODERATE" in result.output
    assert "- Conviction source: company_autonomy" in result.output
    assert "- Scan family: normal" in result.output
    assert "- Buy price target: $90.00" in result.output
    assert "AAA is good but needs a better entry price." in result.output
    assert "- Gross margin declined 310 bps." in result.output
    assert "## History" in result.output


def test_v2_screen_row_masks_price_trigger_fields_across_classic_watchlist_surfaces(
    monkeypatch, tmp_path
):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="VTS",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="LOW",
            conviction_source="sector_screen",
            valuation_anchor_method="DCF",
            valuation_anchor_value=100.0,
            buy_price_target=75.0,
            current_price_at_addition=90.0,
            thesis_text="Deterministic screen only; underwriting has not completed.",
            source_run_id="autonomous_sector_v2_screen_surface",
            source_sector="energy",
            added_at="2026-07-16T12:00:00Z",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )

    listed = runner.invoke(app, ["watchlist", "list"])
    shown = runner.invoke(app, ["watchlist", "show", "VTS"])
    queued = runner.invoke(app, ["watchlist", "queue", "--limit", "10"])

    assert listed.exit_code == 0, listed.output
    assert shown.exit_code == 0, shown.output
    assert queued.exit_code == 0, queued.output
    assert (
        "| VTS | ACTIVE | WATCHLIST_ONLY | LOW | sector_screen | normal | DCF $100.00 | n/a | $90.00 | n/a |"
        in listed.output
    )
    assert "$75.00" not in listed.output
    assert "+20.0%" not in listed.output
    assert "- Investable: no — research only" in shown.output
    assert "Buy price target" not in shown.output
    assert "Distance from buy" not in shown.output
    assert "$75.00" not in shown.output
    assert (
        "| VTS | ACTIVE | WATCHLIST_ONLY | LOW | sector_screen | normal | $90.00 | n/a | n/a |"
        in queued.output
    )
    assert "$75.00" not in queued.output
    assert "+20.0%" not in queued.output
    for output in (listed.output, shown.output, queued.output):
        assert "BUY NOW" not in output
        assert "AT TARGET" not in output
        assert "Wait for $" not in output


def test_watchlist_add_rejects_before_price_fetch_or_write(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    price_calls: list[str] = []
    monkeypatch.setattr(
        "app.cli._fetch_watchlist_manual_add_price",
        lambda ticker: price_calls.append(ticker),
    )

    result = runner.invoke(
        app,
        [
            "watchlist",
            "add",
            "xyz",
            "--thesis",
            "Manual thesis with a clear valuation setup.",
            "--buy-price",
            "100.00",
            "--source-note",
            "manual source note",
        ],
    )
    latest = get_latest("XYZ", db_path=db_path)
    history = get_history("XYZ", db_path=db_path)

    assert result.exit_code == 1
    assert "exact authorized run/ticker lineage" in result.output
    assert "no price was fetched and no entry was created" in result.output
    assert price_calls == []
    assert latest is None
    assert history == []


def test_watchlist_add_without_required_args_errors(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)

    result = runner.invoke(app, ["watchlist", "add", "XYZ", "--buy-price", "100.00"])

    # A non-zero exit is the behavior under test (missing required --thesis).
    # The exact "Missing option" text is Click-version-dependent (newer renders a
    # Rich Usage panel), so asserting the message would be flaky across versions.
    assert result.exit_code != 0


def test_watchlist_add_existing_ticker_preserves_prior_row(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)
    price_calls: list[str] = []
    monkeypatch.setattr(
        "app.cli._fetch_watchlist_manual_add_price",
        lambda ticker: price_calls.append(ticker),
    )

    first = runner.invoke(
        app,
        ["watchlist", "add", "AAA", "--thesis", "First manual thesis.", "--buy-price", "100.00"],
    )
    second = runner.invoke(
        app,
        ["watchlist", "add", "AAA", "--thesis", "Second manual thesis.", "--buy-price", "110.00"],
    )
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT source_run_id, thesis_text FROM watchlist WHERE ticker = ? ORDER BY id",
            ("AAA",),
        ).fetchall()
    finally:
        conn.close()

    assert first.exit_code == 1
    assert second.exit_code == 1
    assert price_calls == []
    assert rows == [("sector_run_1", "AAA is good but needs a better entry price.")]


def test_watchlist_add_price_unavailable_does_not_create_entry(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.cli._fetch_watchlist_manual_add_price",
        lambda ticker: (_ for _ in ()).throw(AssertionError(f"unexpected price call: {ticker}")),
    )

    result = runner.invoke(
        app,
        ["watchlist", "add", "XYZ", "--thesis", "Manual thesis.", "--buy-price", "100.00"],
    )
    latest = get_latest("XYZ", db_path=db_path)

    assert result.exit_code == 1
    assert "exact authorized run/ticker lineage" in result.output
    assert latest is None


def test_watchlist_remove_soft_deletes_entry(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)

    result = runner.invoke(app, ["watchlist", "remove", "AAA", "--reason", "No longer fits."])
    latest = get_latest("AAA", db_path=db_path)
    removed_list = runner.invoke(app, ["watchlist", "list", "--status", "REMOVED"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "Removed AAA from the active watchlist."
    assert latest is not None
    assert latest.status == "REMOVED"
    assert latest.status_reason == "No longer fits."
    assert removed_list.exit_code == 0, removed_list.output
    assert (
        "| AAA | REMOVED | WATCHLIST_ONLY | MODERATE | company_autonomy | normal | DCF $120.00 | $90.00 | $100.00 | +11.1% | industrial_tech | 2026-05-08T12:00:00Z |"
        in removed_list.output
    )


def test_watchlist_stats_outputs_counts(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)

    result = runner.invoke(app, ["watchlist", "stats"])

    assert result.exit_code == 0, result.output
    assert "# Watchlist Stats" in result.output
    assert "- Mode: current unique tickers" in result.output
    assert "- Total entries: 2" in result.output
    assert "- Unique tickers: 2" in result.output
    assert "- All non-removed rows: 2" in result.output
    assert "- Duplicate non-removed rows: 0" in result.output
    assert "| ACTIVE | 1 |" in result.output
    assert "| DEPLOY_READY | 1 |" in result.output
    assert "| industrial_tech | 2 |" in result.output


def test_watchlist_stats_all_rows_preserves_historical_row_totals(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            conviction_source="relative_ranking",
            buy_price_target=85.0,
            current_price_at_addition=80.0,
            thesis_text="Latest duplicate row.",
            source_run_id="sector_run_3",
            source_sector="industrial_tech",
            added_at="2026-05-10T12:00:00Z",
        ),
        db_path=db_path,
    )

    current = runner.invoke(app, ["watchlist", "stats"])
    all_rows = runner.invoke(app, ["watchlist", "stats", "--all-rows"])

    assert current.exit_code == 0, current.output
    assert "- Mode: current unique tickers" in current.output
    assert "- Total entries: 2" in current.output
    assert "- All non-removed rows: 3" in current.output
    assert "- Duplicate non-removed rows: 1" in current.output
    assert all_rows.exit_code == 0, all_rows.output
    assert "- Mode: all rows" in all_rows.output
    assert "- Total entries: 3" in all_rows.output
    assert "| DEPLOY_READY | 2 |" in all_rows.output


def test_watchlist_queue_outputs_ranked_current_rows(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)
    bbb = get_latest("BBB", db_path=db_path)
    assert bbb is not None
    add_price_snapshot(
        bbb.id or 0, price=50.0, checked_at="2026-05-11T00:00:00Z", source="test", db_path=db_path
    )
    add_or_update(
        WatchlistEntry(
            ticker="CCC",
            status="PRICE_DATA_SUSPECT",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            conviction_source="company_autonomy",
            buy_price_target=100.0,
            current_price_at_addition=20.0,
            thesis_text="Suspect price row.",
            source_run_id="sector_run_suspect",
            source_sector="industrial_tech",
            added_at="2026-05-10T12:00:00Z",
        ),
        db_path=db_path,
    )

    default = runner.invoke(app, ["watchlist", "queue", "--limit", "10"])
    with_suspect = runner.invoke(
        app, ["watchlist", "queue", "--limit", "10", "--include-price-suspect"]
    )

    assert default.exit_code == 0, default.output
    assert (
        "| Ticker | Status | Conviction | Confidence | Conviction Source | Scan Family | Latest/Add Price |"
        in default.output
    )
    assert default.output.index(
        "| BBB | DEPLOY_READY | ACTIONABLE | LOW | sector_final_decision | normal | $50.00 | $60.00 | -16.7% |"
    ) < default.output.index(
        "| AAA | ACTIVE | WATCHLIST_ONLY | MODERATE | company_autonomy | normal | $100.00 | $90.00 | +11.1% |"
    )
    assert "CCC" not in default.output
    assert with_suspect.exit_code == 0, with_suspect.output
    assert (
        "| CCC | PRICE_DATA_SUSPECT | ACTIONABLE | HIGH | company_autonomy | normal | $20.00 | $100.00 | -80.0% |"
        in with_suspect.output
    )


def test_watchlist_refresh_dry_run_reports_without_mutation(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)
    _insert_refresh_filing(db_path)
    monkeypatch.setattr(
        "app.watchlist.reevaluation.load_current_event_context",
        lambda *args, **kwargs: SimpleNamespace(ordered_documents=[]),
    )

    result = runner.invoke(
        app, ["watchlist", "refresh", "AAA", "--since", "2026-05-01", "--dry-run"]
    )
    latest = get_latest("AAA", db_path=db_path)

    assert result.exit_code == 0, result.output
    assert "| AAA | ACTIVE | ACTIVE | DRY_RUN_NEW_EVIDENCE | $0.00 |" in result.output
    assert "Total estimated LLM cost: $0.00" in result.output
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert latest.last_evaluated_at is None


def test_watchlist_refresh_max_cost_skips_before_llm(monkeypatch, tmp_path):
    from tests.test_watchlist_reevaluation import (
        _init_temp_db,
        _seed_canonical_financial_context,
    )

    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_watchlist(db_path)
    _seed_canonical_financial_context(
        db_path,
        ticker="AAA",
        as_of_date=date.today().isoformat(),
    )
    _insert_refresh_filing(db_path)
    monkeypatch.setattr(
        "app.watchlist.reevaluation.load_current_event_context",
        lambda *args, **kwargs: SimpleNamespace(ordered_documents=[]),
    )
    monkeypatch.setattr(
        "app.watchlist.reevaluation.estimate_preflight_cost_usd", lambda *args, **kwargs: 0.50
    )

    def fail_llm(prompt):
        raise AssertionError("LLM should be skipped by max-cost cap")

    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", fail_llm)

    result = runner.invoke(
        app, ["watchlist", "refresh", "AAA", "--since", "2026-05-01", "--max-cost-usd", "0.10"]
    )
    latest = get_latest("AAA", db_path=db_path)

    assert result.exit_code == 0, result.output
    assert "| AAA | ACTIVE | ACTIVE | BUDGET_SKIPPED | $0.00 |" in result.output
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert latest.last_evaluated_at is None


def test_watchlist_check_triggers_dry_run_reports_without_mutation(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)

    today = date.today().isoformat()
    monkeypatch.setattr(
        "app.watchlist.triggers._fetch_latest_price",
        lambda ticker: PriceSnapshot(
            ticker=ticker,
            as_of_date=today,
            price=85.0,
            currency="USD",
            source="yahoo",
            retrieved_at=f"{today}T12:00:00+00:00",
            confidence="HIGH",
        ),
    )

    result = runner.invoke(app, ["watchlist", "check-triggers", "AAA", "--dry-run"])
    latest = get_latest("AAA", db_path=db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        snapshot_count = conn.execute("SELECT COUNT(*) FROM watchlist_price_snapshots").fetchone()[
            0
        ]
    finally:
        conn.close()

    assert result.exit_code == 0, result.output
    # A fresh at-target crossing is pending confirmation until the second
    # consecutive heartbeat (the CLI renders the warning in the transition cell).
    assert (
        "| AAA | ACTIVE | $85.00 | $90.00 | ACTIVE | "
        "AT_TARGET_PENDING_CONFIRMATION:price=85.00:target=90.00:confirms_next_heartbeat |"
    ) in result.output
    assert "(dry run — no watchlist rows or price snapshots updated)" in result.output
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert snapshot_count == 0


def test_watchlist_daily_default_runs_triggers_then_digest(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    output_path = tmp_path / "daily.md"
    calls: list[str] = []

    def fake_check():
        calls.append("trigger")
        return [
            WatchlistTriggerResult(
                ticker="AAA",
                prior_status="ACTIVE",
                new_status="DEPLOY_READY",
                latest_price=85.0,
                buy_price_target=90.0,
                transition="crossed below buy-target",
            )
        ]

    def fake_write(
        *, output_path: str | Path | None = None, check_prices_first: bool = False, **kwargs
    ):
        calls.append("digest")
        assert check_prices_first is False
        path = Path(output_path or tmp_path / "digest.md")
        path.write_text("# fake digest\n", encoding="utf-8")
        return DigestWriteResult(markdown="# fake digest\n", path=path)

    monkeypatch.setattr("app.watchlist.triggers.check_watchlist_triggers", fake_check)
    monkeypatch.setattr("app.watchlist.digest.write_digest", fake_write)

    result = runner.invoke(app, ["watchlist", "daily", "--digest-output", str(output_path)])

    assert result.exit_code == 0, result.output
    assert calls == ["trigger", "digest"]
    assert result.output.index("## Price Trigger Check") < result.output.index("Digest written:")
    assert (
        "| AAA | ACTIVE | $85.00 | $90.00 | DEPLOY_READY | crossed below buy-target |"
        in result.output
    )
    assert "# fake digest" in result.output


def test_watchlist_daily_no_prices_skips_trigger_check(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    output_path = tmp_path / "daily.md"
    calls: list[str] = []

    def fail_check():
        raise AssertionError("trigger check should be skipped")

    def fake_write(
        *, output_path: str | Path | None = None, check_prices_first: bool = False, **kwargs
    ):
        calls.append("digest")
        assert check_prices_first is False
        path = Path(output_path or tmp_path / "digest.md")
        path.write_text("# digest only\n", encoding="utf-8")
        return DigestWriteResult(markdown="# digest only\n", path=path)

    monkeypatch.setattr("app.watchlist.triggers.check_watchlist_triggers", fail_check)
    monkeypatch.setattr("app.watchlist.digest.write_digest", fake_write)

    result = runner.invoke(
        app, ["watchlist", "daily", "--no-prices", "--digest-output", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    assert calls == ["digest"]
    assert "## Price Trigger Check" not in result.output
    assert "# digest only" in result.output


def test_watchlist_daily_no_digest_runs_only_trigger_check(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    calls: list[str] = []

    def fake_check():
        calls.append("trigger")
        return []

    def fail_write(**kwargs):
        raise AssertionError("digest render should be skipped")

    monkeypatch.setattr("app.watchlist.triggers.check_watchlist_triggers", fake_check)
    monkeypatch.setattr("app.watchlist.digest.write_digest", fail_write)

    result = runner.invoke(app, ["watchlist", "daily", "--no-digest"])

    assert result.exit_code == 0, result.output
    assert calls == ["trigger"]
    assert "| n/a | n/a | n/a | n/a | n/a | no eligible entries |" in result.output
    assert "Digest written:" not in result.output


def test_watchlist_daily_rejects_no_work_flags(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)

    result = runner.invoke(app, ["watchlist", "daily", "--no-prices", "--no-digest"])

    assert result.exit_code == 1
    assert "Nothing to do: --no-prices and --no-digest cannot both be set." in result.output


def test_watchlist_daily_default_writes_digest_to_configured_output(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(tmp_path / "data" / "engine.db")

    monkeypatch.setattr(
        "app.watchlist.triggers._fetch_latest_price",
        lambda ticker: PriceSnapshot(
            ticker=ticker,
            as_of_date="2026-05-10",
            price=85.0 if ticker == "AAA" else 55.0,
            currency="USD",
            source="yahoo",
            retrieved_at="2026-05-10T12:00:00+00:00",
            confidence="HIGH",
        ),
    )

    result = runner.invoke(app, ["watchlist", "daily"])

    digest_paths = sorted((tmp_path / "data" / "outputs" / "digests").glob("digest_*.md"))
    assert result.exit_code == 0, result.output
    assert len(digest_paths) == 1
    assert digest_paths[0].read_text(encoding="utf-8").startswith("# IVI Watchlist Daily Digest")
    assert f"Digest written: {digest_paths[0]}" in result.output


def test_watchlist_actionable_counterfactual_respects_allow_evidence_caps(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    data_dir = tmp_path / "data"
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="DEPLOY_READY",
            conviction_grade="WATCHLIST_ONLY",
            buy_price_target=90.0,
            current_price_at_addition=80.0,
            thesis_text="Evidence gap only.",
            source_run_id="sector_run_gap",
            source_sector="capital_markets",
            added_at="2026-05-12T12:00:00Z",
        ),
        db_path=db_path,
    )
    _write_counterfactual_artifact(
        data_dir,
        run_id="sector_run_gap",
        ticker="AAA",
        base_return=0.13,
        hard_blockers=["NO_FILING"],
        confidence_caps=["CURRENT_EVENTS_UNAVAILABLE"],
    )
    output_path = tmp_path / "counterfactual.md"

    result = runner.invoke(
        app,
        [
            "watchlist",
            "actionable-counterfactual",
            "--hurdle-pct",
            "12",
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "| AAA | DEPLOY_READY | WATCHLIST_ONLY | WATCHLIST_ONLY | +13.0% | -2.0% |" in result.output
    )
    assert "Would shift to ACTIONABLE: 0" in result.output

    result = runner.invoke(
        app,
        [
            "watchlist",
            "actionable-counterfactual",
            "--allow-evidence-caps",
            "--hurdle-pct",
            "12",
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "| AAA | DEPLOY_READY | WATCHLIST_ONLY | ACTIONABLE | +13.0% | -2.0% |" in result.output
    assert "Would shift to ACTIONABLE: 1" in result.output
    assert output_path.read_text(encoding="utf-8").startswith(
        "# Watchlist Actionability Counterfactual"
    )


def test_watchlist_actionable_counterfactual_respects_hurdle_pct(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    data_dir = tmp_path / "data"
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            thesis_text="Needs lower hurdle.",
            source_run_id="sector_run_hurdle",
            source_sector="industrial_tech",
            added_at="2026-05-12T12:00:00Z",
        ),
        db_path=db_path,
    )
    _write_counterfactual_artifact(
        data_dir,
        run_id="sector_run_hurdle",
        ticker="AAA",
        base_return=0.095,
        hard_blockers=["BASE_RETURN_BELOW_12PCT_HURDLE"],
        confidence_caps=[],
    )

    result = runner.invoke(
        app,
        ["watchlist", "actionable-counterfactual", "--allow-evidence-caps", "--hurdle-pct", "10"],
    )
    assert result.exit_code == 0, result.output
    assert "| AAA | ACTIVE | WATCHLIST_ONLY | WATCHLIST_ONLY | +9.5% | -2.0% |" in result.output
    assert "Would shift to ACTIONABLE: 0" in result.output

    result = runner.invoke(
        app,
        ["watchlist", "actionable-counterfactual", "--allow-evidence-caps", "--hurdle-pct", "9"],
    )
    assert result.exit_code == 0, result.output
    assert "| AAA | ACTIVE | WATCHLIST_ONLY | ACTIONABLE | +9.5% | -2.0% |" in result.output
    assert "Would shift to ACTIONABLE: 1" in result.output


def test_watchlist_actionable_counterfactual_business_quality_always_blocks(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    data_dir = tmp_path / "data"
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="DEPLOY_READY",
            conviction_grade="WATCHLIST_ONLY",
            thesis_text="Business risk remains.",
            source_run_id="sector_run_business",
            source_sector="utilities",
            added_at="2026-05-12T12:00:00Z",
        ),
        db_path=db_path,
    )
    _write_counterfactual_artifact(
        data_dir,
        run_id="sector_run_business",
        ticker="AAA",
        base_return=0.30,
        hard_blockers=[],
        confidence_caps=["ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK"],
    )

    result = runner.invoke(
        app,
        ["watchlist", "actionable-counterfactual", "--allow-evidence-caps", "--hurdle-pct", "9"],
    )

    assert result.exit_code == 0, result.output
    assert (
        "| AAA | DEPLOY_READY | WATCHLIST_ONLY | WATCHLIST_ONLY | +30.0% | -2.0% |" in result.output
    )
    assert "business-quality signals: ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK" in result.output
    assert "Would shift to ACTIONABLE: 0" in result.output


def test_watchlist_queue_compact_renders_signal_board(monkeypatch, tmp_path):
    db_path = _init_temp_data_dir(monkeypatch, tmp_path)
    _seed_watchlist(db_path)

    result = runner.invoke(app, ["watchlist", "queue", "--compact"])

    assert result.exit_code == 0, result.output
    assert (
        "| Signal | Ticker | Conviction | Confidence | Price | Buy Target | "
        "Distance From Buy | Band | Source Sector | Status | Disposition | "
        "Decision Basis | Validation |"
    ) in result.output
    assert "| WAIT | AAA | WATCHLIST_ONLY | MODERATE |" in result.output
