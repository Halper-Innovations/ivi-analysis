from __future__ import annotations

import json
from datetime import date
import sqlite3
from pathlib import Path

import pytest

from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
)
from app.watchlist.contract import WatchlistEntry
from app.watchlist.schema import ensure_watchlist_schema
from app.watchlist.store import (
    add_price_snapshot,
    add_or_update,
    backfill_confidence_from_artifacts,
    get_history,
    get_latest,
    list_active,
    mark_status,
    populate_from_sector_artifact,
    record_trigger_status_change,
    remove,
    stats,
    watchlist_queue,
)


@pytest.fixture(autouse=True)
def _legacy_packet_fixture_bypasses_new_integrity_gate(monkeypatch):
    monkeypatch.setattr("app.watchlist.store.artifact_decision_eligibility", lambda payload: "PASS")


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    return db_path


def _entry(
    *,
    ticker: str = "AAA",
    source_run_id: str = "sector_run_1",
    added_at: str = "2026-05-08T12:00:00+00:00",
    status: str = "ACTIVE",
    buy_price_target: float = 80.0,
    current_price_at_addition: float = 100.0,
    confidence: str | None = "MODERATE",
    conviction_grade: str = "WATCHLIST_ONLY",
    conviction_source: str | None = "company_autonomy",
    scan_family: str = "normal",
) -> WatchlistEntry:
    return WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade=conviction_grade,
        confidence=confidence,
        conviction_source=conviction_source,
        scan_family=scan_family,
        valuation_anchor_method="DCF",
        valuation_anchor_value=106.67,
        buy_price_target=buy_price_target,
        current_price_at_addition=current_price_at_addition,
        thesis_text="Durable candidate with a buy-price anchor.",
        key_risks=["Margin compression"],
        falsifiers=["Revenue decline persists"],
        open_questions=["Can margins normalize?"],
        source_run_id=source_run_id,
        source_sector="industrial_tech",
        added_at=added_at,
    )


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


# Populate's add-time DEPLOY_READY refuses artifacts older than the
# price-staleness ceiling, so artifact fixtures are dated relative to today.
_ARTIFACT_ASOF = date.today().isoformat()


def _sector_artifact() -> AutonomousSectorFinancialRunArtifact:
    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_industrial_tech_test",
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        objective="Populate a watchlist from finalist candidates.",
        as_of_date=_ARTIFACT_ASOF,
        created_at=f"{_ARTIFACT_ASOF}T12:00:00Z",
        completed_at=f"{_ARTIFACT_ASOF}T12:01:00Z",
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
        memo_body={
            "candidates": {
                "AAA": {
                    "thesis": "AAA is actionable because price is below the buy anchor.",
                    "key_risks": ["Backlog declines"],
                    "falsifiers": ["Margins keep falling"],
                    "open_questions": ["Is the trough cyclical?"],
                },
                "BBB": {
                    "thesis": "BBB remains watchlist-only until price comes to value.",
                    "key_risks": ["Working capital lengthens"],
                    "falsifiers": ["FCF yield deteriorates"],
                    "open_questions": ["Can conversion normalize?"],
                },
            }
        },
    )


def test_schema_migration_creates_tables_idempotently(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    ensure_watchlist_schema(db_path)
    ensure_watchlist_schema(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        watchlist_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()
        ]
        history_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(watchlist_history)").fetchall()
        ]
    finally:
        conn.close()
    assert watchlist_columns == [
        "id",
        "ticker",
        "status",
        "conviction_grade",
        "confidence",
        "conviction_source",
        "scan_family",
        "valuation_anchor_method",
        "valuation_anchor_value",
        "buy_price_target",
        "current_price_at_addition",
        "thesis_text",
        "key_risks_json",
        "falsifiers_json",
        "open_questions_json",
        "source_run_id",
        "source_sector",
        "added_at",
        "last_evaluated_at",
        "current_event_watermark_json",
        "status_reason",
        "market_cap_mm",
        "cap_source",
        "cap_band",
        "cap_asof",
        "event_pending",
        "adv_dollar_20d",
        "adv_dollar_60d",
        "adv_asof",
        "capacity_class",
        "pipeline_version",
        "candidate_disposition",
        "decision_basis",
        "selection_validation_status",
    ]
    assert history_columns == [
        "id",
        "watchlist_id",
        "changed_at",
        "field_name",
        "old_value",
        "new_value",
        "source",
        "source_run_id",
    ]


def test_add_or_update_new_ticker_creates_entry_and_history(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    row_id = add_or_update(_entry(), db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)
    history = get_history("AAA", db_path=db_path)

    assert row_id == 1
    assert latest is not None
    assert latest.ticker == "AAA"
    assert latest.status == "ACTIVE"
    assert latest.confidence == "MODERATE"
    assert latest.conviction_source == "company_autonomy"
    assert latest.scan_family == "normal"
    assert latest.buy_price_target == 80.0
    assert history[0]["field_name"] == "created"
    assert history[0]["new_value"] == "ACTIVE"


def test_add_or_update_same_run_is_idempotent(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    first_id = add_or_update(_entry(), db_path=db_path)
    second_id = add_or_update(_entry(), db_path=db_path)
    history = get_history("AAA", db_path=db_path)

    assert first_id == 1
    assert second_id == 1
    assert len(history) == 1


def test_add_or_update_different_run_creates_latest_active_entry(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    first_id = add_or_update(
        _entry(source_run_id="sector_run_1", added_at="2026-05-08T12:00:00Z"), db_path=db_path
    )
    second_id = add_or_update(
        _entry(source_run_id="sector_run_2", added_at="2026-05-09T12:00:00Z"), db_path=db_path
    )
    entries = list_active(db_path=db_path)

    assert first_id == 1
    assert second_id == 2
    assert len(entries) == 1
    assert entries[0].ticker == "AAA"
    assert entries[0].source_run_id == "sector_run_2"


def test_list_active_scan_family_filter_applies_before_latest_grouping(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    add_or_update(
        _entry(
            ticker="AAA",
            source_run_id="normal_run",
            added_at="2026-05-08T12:00:00Z",
            scan_family="normal",
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker="AAA",
            source_run_id="pearl_run",
            added_at="2026-05-09T12:00:00Z",
            conviction_source="pearl_scan",
            scan_family="pearl",
        ),
        db_path=db_path,
    )

    normal_rows = list_active(scan_family="normal", db_path=db_path)
    pearl_rows = list_active(scan_family="pearl", db_path=db_path)

    assert [(row.ticker, row.source_run_id, row.scan_family) for row in normal_rows] == [
        ("AAA", "normal_run", "normal")
    ]
    assert [(row.ticker, row.source_run_id, row.scan_family) for row in pearl_rows] == [
        ("AAA", "pearl_run", "pearl")
    ]


def test_add_or_update_does_not_revert_trigger_set_deploy_ready(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    add_or_update(_entry(status="ACTIVE"), db_path=db_path)
    # Trigger path flips the existing (ticker, source_run_id) row to DEPLOY_READY.
    mark_status(
        "AAA",
        "DEPLOY_READY",
        "Price crossed the buy target.",
        "trigger",
        "price_check_1",
        db_path=db_path,
    )

    # Re-populating the SAME (ticker, source_run_id) with a recomputed ACTIVE status must not revert it.
    add_or_update(_entry(status="ACTIVE"), db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.status == "DEPLOY_READY"
    assert latest.status_reason == "Price crossed the buy target."


def test_add_or_update_does_not_revert_price_data_suspect(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    add_or_update(_entry(status="ACTIVE"), db_path=db_path)
    mark_status(
        "AAA",
        "PRICE_DATA_SUSPECT",
        "PRICE_DATA_SUSPECT:latest_price=0.00",
        "trigger",
        "price_check_1",
        db_path=db_path,
    )

    add_or_update(_entry(status="ACTIVE"), db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.status == "PRICE_DATA_SUSPECT"
    assert latest.status_reason == "PRICE_DATA_SUSPECT:latest_price=0.00"


def test_mark_status_logs_history(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(_entry(), db_path=db_path)

    mark_status(
        "AAA",
        "DEPLOY_READY",
        "Price crossed the buy target.",
        "trigger",
        "price_check_1",
        db_path=db_path,
    )
    latest = get_latest("AAA", db_path=db_path)
    history = get_history("AAA", db_path=db_path)

    assert latest is not None
    assert latest.status == "DEPLOY_READY"
    assert latest.status_reason == "Price crossed the buy target."
    assert [row["field_name"] for row in history] == ["created", "status", "status_reason"]
    assert history[1]["old_value"] == "ACTIVE"
    assert history[1]["new_value"] == "DEPLOY_READY"


def test_remove_soft_deletes_latest_entry(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(_entry(), db_path=db_path)

    remove("AAA", "No longer fits the mandate.", db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)
    active_entries = list_active(db_path=db_path)
    removed_entries = list_active(status="REMOVED", db_path=db_path)

    assert latest is not None
    assert latest.status == "REMOVED"
    assert latest.status_reason == "No longer fits the mandate."
    assert active_entries == []
    assert len(removed_entries) == 1
    assert removed_entries[0].ticker == "AAA"


def test_populate_from_sector_artifact_adds_only_actionable_watchlist_candidates(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    result = populate_from_sector_artifact(_sector_artifact(), db_path=db_path)
    entries = list_active(db_path=db_path)
    aaa = get_latest("AAA", db_path=db_path)
    bbb = get_latest("BBB", db_path=db_path)
    ccc = get_latest("CCC", db_path=db_path)

    assert result.added_or_updated == 2
    assert result.skipped == 1
    assert result.skipped_reasons == {"CCC": "VERDICT_AVOID"}
    assert [entry.ticker for entry in entries] == ["BBB", "AAA"]
    assert aaa is not None
    assert aaa.status == "DEPLOY_READY"
    assert aaa.conviction_grade == "ACTIONABLE"
    assert aaa.confidence == "HIGH"
    assert aaa.conviction_source == "company_autonomy"
    assert aaa.scan_family == "normal"
    assert aaa.thesis_text == "AAA is actionable because price is below the buy anchor."
    assert aaa.key_risks == ["Backlog declines"]
    assert bbb is not None
    assert bbb.status == "ACTIVE"
    assert bbb.conviction_grade == "WATCHLIST_ONLY"
    assert bbb.confidence == "MODERATE"
    assert bbb.conviction_source == "company_autonomy"
    assert bbb.scan_family == "normal"
    assert ccc is None


def test_populate_runs_grade_time_adverse_check_on_actionable_only(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    calls: list[list[str]] = []

    def fake_check(conn, tickers, **kwargs):
        calls.append(list(tickers))
        return {"checked": list(tickers), "events_created": 0, "flagged": {}}

    import app.events.grade_time as grade_time

    monkeypatch.setattr(grade_time, "grade_time_adverse_check", fake_check)
    result = populate_from_sector_artifact(_sector_artifact(), db_path=db_path)
    # AAA is the only ACTIONABLE candidate; BBB (WATCHLIST_ONLY) and CCC
    # (AVOID) never trigger the grade-time check.
    assert calls == [["AAA"]]
    assert result.adverse_check == {"checked": ["AAA"], "events_created": 0, "flagged": {}}


def test_populate_from_sector_artifact_derives_watchlist_confidence_from_caps(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    artifact = AutonomousSectorFinancialRunArtifact.from_dict(
        {
            **_sector_artifact().to_dict(),
            "run_id": "autonomous_sector_industrial_tech_caps",
            "candidate_selection": {"selected_tickers": ["BBB"]},
            "relative_ranking": [
                {
                    "ticker": "BBB",
                    "company_autonomy_verdict": "WATCHLIST_ONLY",
                    "confidence_caps": [
                        "ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK",
                        "STRUCTURALLY_WEAK_ECONOMICS",
                    ],
                    "positioning_summary": "BBB is watchlist-only with two business-quality confidence caps.",
                }
            ],
        }
    )

    result = populate_from_sector_artifact(artifact, db_path=db_path)
    latest = get_latest("BBB", db_path=db_path)

    assert result.added_or_updated == 1
    assert result.skipped == 0
    assert latest is not None
    assert latest.conviction_grade == "WATCHLIST_ONLY"
    assert latest.confidence == "LOW"
    assert latest.conviction_source == "company_autonomy"


@pytest.mark.parametrize(
    ("caps", "expected"),
    [
        ([], None),
        (["CURRENT_EVENTS_UNAVAILABLE"], None),  # evidence-quality cap only
        (["ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK"], "MODERATE"),
        (["ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK", "STRUCTURALLY_WEAK_ECONOMICS"], "LOW"),
    ],
)
def test_confidence_inferred_from_caps_never_claims_high(monkeypatch, tmp_path, caps, expected):
    """Absence of caps is not evidence of strength: with no explicit confidence label the
    ranking's caps can only lower confidence, so nothing here comes back HIGH (it used to,
    which also selected the thinnest margin-of-safety base for an ACTIONABLE name)."""
    from app.watchlist.store import _confidence_from_business_caps

    assert _confidence_from_business_caps(caps) == expected

    db_path = _init_temp_db(monkeypatch, tmp_path)
    artifact = AutonomousSectorFinancialRunArtifact.from_dict(
        {
            **_sector_artifact().to_dict(),
            "run_id": "autonomous_sector_industrial_tech_nocaps",
            "candidate_selection": {"selected_tickers": ["BBB"]},
            "relative_ranking": [
                {
                    "ticker": "BBB",
                    "company_autonomy_verdict": "WATCHLIST_ONLY",
                    "confidence_caps": caps,
                    "positioning_summary": "BBB is watchlist-only; no explicit confidence label.",
                }
            ],
        }
    )
    populate_from_sector_artifact(artifact, db_path=db_path)
    latest = get_latest("BBB", db_path=db_path)
    assert latest is not None
    assert latest.confidence == expected


def test_populate_from_sector_artifact_selected_ticker_overrides_no_winner_ranking(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_enterprise_software_selected",
        sector="enterprise_software",
        market_cap_focus="mid_cap",
        objective="Populate selected finalist despite stale ranking verdict.",
        as_of_date=_ARTIFACT_ASOF,
        created_at=f"{_ARTIFACT_ASOF}T12:00:00Z",
        completed_at=f"{_ARTIFACT_ASOF}T12:10:00Z",
        status="COMPLETED",
        final_verdict="SELECTED",
        selected_ticker="AAA",
        confidence="LOW",
        candidate_selection={"selected_tickers": ["AAA"]},
        company_packets=[_packet("AAA", current_price=70.0, buy_price_target=80.0)],
        relative_ranking=[
            {
                "ticker": "AAA",
                "company_autonomy_verdict": "NO_WINNER",
                "positioning_summary": "AAA is the final selected ticker after sector audit.",
            }
        ],
        memo_body={
            "candidates": {
                "AAA": {
                    "thesis": "AAA is selected by the final sector decision.",
                    "key_risks": ["Execution risk"],
                    "falsifiers": ["Growth decelerates"],
                    "open_questions": ["Can margins hold?"],
                }
            }
        },
    )

    result = populate_from_sector_artifact(artifact, db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)

    assert result.added_or_updated == 1
    assert result.skipped == 0
    assert result.skipped_reasons == {}
    assert latest is not None
    assert latest.conviction_grade == "ACTIONABLE"
    assert latest.confidence == "LOW"
    assert latest.conviction_source == "sector_final_decision"
    assert latest.status == "DEPLOY_READY"
    assert latest.source_sector == "enterprise_software"


def test_backfill_confidence_from_artifacts_populates_existing_actionable_rows(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            ticker="AAA",
            status="DEPLOY_READY",
            source_run_id="autonomous_sector_industrial_tech_backfill",
            confidence=None,
            conviction_source=None,
            conviction_grade="ACTIONABLE",
        ),
        db_path=db_path,
    )
    run_dir = (
        tmp_path
        / "data"
        / "outputs"
        / "runs"
        / "autonomous_sector"
        / "autonomous_sector_industrial_tech_backfill"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "autonomous_sector_run.json").write_text(
        json.dumps(
            {
                **_sector_artifact().to_dict(),
                "run_id": "autonomous_sector_industrial_tech_backfill",
                "relative_ranking": [
                    {
                        "ticker": "AAA",
                        "company_autonomy_verdict": "ACTIONABLE",
                        "company_autonomy_confidence": "LOW",
                        "positioning_summary": "AAA is actionable but low-confidence.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = backfill_confidence_from_artifacts(db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)
    history = get_history("AAA", db_path=db_path)

    assert result == {
        "updated": [
            {
                "id": 1,
                "ticker": "AAA",
                "confidence": "LOW",
                "conviction_source": "company_autonomy",
                "updated_fields": ["confidence", "conviction_source"],
            }
        ],
        "skipped": {},
        "updated_count": 1,
    }
    assert latest is not None
    assert latest.confidence == "LOW"
    assert latest.conviction_source == "company_autonomy"
    assert [row["field_name"] for row in history[-2:]] == ["confidence", "conviction_source"]
    assert history[-1]["source"] == "decision_clarity_backfill"


def test_backfill_confidence_from_artifacts_populates_existing_watchlist_only_rows(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            ticker="BBB",
            status="ACTIVE",
            source_run_id="autonomous_sector_industrial_tech_watchlist_backfill",
            confidence=None,
            conviction_source=None,
            conviction_grade="WATCHLIST_ONLY",
        ),
        db_path=db_path,
    )
    run_dir = (
        tmp_path
        / "data"
        / "outputs"
        / "runs"
        / "autonomous_sector"
        / "autonomous_sector_industrial_tech_watchlist_backfill"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "autonomous_sector_run.json").write_text(
        json.dumps(
            {
                **_sector_artifact().to_dict(),
                "run_id": "autonomous_sector_industrial_tech_watchlist_backfill",
                "relative_ranking": [
                    {
                        "ticker": "BBB",
                        "company_autonomy_verdict": "WATCHLIST_ONLY",
                        "company_autonomy_confidence": "MODERATE",
                        "positioning_summary": "BBB is watchlist-only with moderate confidence.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = backfill_confidence_from_artifacts(db_path=db_path)
    latest = get_latest("BBB", db_path=db_path)
    history = get_history("BBB", db_path=db_path)

    assert result == {
        "updated": [
            {
                "id": 1,
                "ticker": "BBB",
                "confidence": "MODERATE",
                "conviction_source": "company_autonomy",
                "updated_fields": ["confidence", "conviction_source"],
            }
        ],
        "skipped": {},
        "updated_count": 1,
    }
    assert latest is not None
    assert latest.confidence == "MODERATE"
    assert latest.conviction_source == "company_autonomy"
    assert [row["field_name"] for row in history[-2:]] == ["confidence", "conviction_source"]
    assert history[-1]["source"] == "decision_clarity_backfill"


def test_stats_defaults_to_latest_unique_rows(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            ticker="AAA", source_run_id="run_1", status="ACTIVE", added_at="2026-05-01T00:00:00Z"
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker="AAA",
            source_run_id="run_2",
            status="DEPLOY_READY",
            added_at="2026-05-02T00:00:00Z",
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker="BBB", source_run_id="run_3", status="ACTIVE", added_at="2026-05-03T00:00:00Z"
        ),
        db_path=db_path,
    )

    current = stats(db_path=db_path)
    historical = stats(db_path=db_path, all_rows=True)

    assert current["mode"] == "current_unique"
    assert current["total_entries"] == 2
    assert current["all_rows_total"] == 3
    assert current["unique_tickers"] == 2
    assert current["duplicate_rows"] == 1
    assert current["status_counts"] == {"ACTIVE": 1, "DEPLOY_READY": 1}
    assert historical["mode"] == "all_rows"
    assert historical["total_entries"] == 3
    assert historical["status_counts"] == {"ACTIVE": 2, "DEPLOY_READY": 1}


def test_watchlist_queue_sorts_by_decision_priority(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            ticker="AAA",
            source_run_id="run_watch_deploy",
            status="DEPLOY_READY",
            conviction_grade="WATCHLIST_ONLY",
            confidence="HIGH",
            buy_price_target=80.0,
            current_price_at_addition=70.0,
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker="BBB",
            source_run_id="run_action_active",
            status="ACTIVE",
            conviction_grade="ACTIONABLE",
            confidence="LOW",
            buy_price_target=80.0,
            current_price_at_addition=100.0,
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker="CCC",
            source_run_id="run_action_deploy_moderate",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="MODERATE",
            buy_price_target=100.0,
            current_price_at_addition=90.0,
        ),
        db_path=db_path,
    )
    ddd_id = add_or_update(
        _entry(
            ticker="DDD",
            source_run_id="run_action_deploy_high",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            buy_price_target=100.0,
            current_price_at_addition=95.0,
        ),
        db_path=db_path,
    )
    add_price_snapshot(
        ddd_id, price=85.0, checked_at="2026-05-17T00:00:00Z", source="test", db_path=db_path
    )
    add_or_update(
        _entry(
            ticker="EEE",
            source_run_id="run_price_suspect",
            status="PRICE_DATA_SUSPECT",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            buy_price_target=100.0,
            current_price_at_addition=20.0,
        ),
        db_path=db_path,
    )

    default_rows = watchlist_queue(limit=10, db_path=db_path)
    with_suspect = watchlist_queue(limit=10, include_price_suspect=True, db_path=db_path)

    assert [row["ticker"] for row in default_rows] == ["DDD", "CCC", "BBB", "AAA"]
    assert default_rows[0]["latest_price"] == 85.0
    assert default_rows[0]["distance_from_buy_pct"] == -15.0
    assert [row["ticker"] for row in with_suspect] == ["DDD", "CCC", "BBB", "EEE", "AAA"]


def test_watchlist_queue_scan_family_filter_applies_before_latest_grouping(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            ticker="AAA",
            source_run_id="normal_run",
            added_at="2026-05-08T12:00:00Z",
            scan_family="normal",
            buy_price_target=80.0,
            current_price_at_addition=100.0,
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker="AAA",
            source_run_id="pearl_run",
            added_at="2026-05-09T12:00:00Z",
            conviction_source="pearl_scan",
            scan_family="pearl",
            buy_price_target=120.0,
            current_price_at_addition=90.0,
        ),
        db_path=db_path,
    )

    normal_rows = watchlist_queue(limit=10, scan_family="normal", db_path=db_path)
    pearl_rows = watchlist_queue(limit=10, scan_family="pearl", db_path=db_path)

    assert [
        (row["ticker"], row["scan_family"], row["buy_price_target"]) for row in normal_rows
    ] == [("AAA", "normal", 80.0)]
    assert [(row["ticker"], row["scan_family"], row["buy_price_target"]) for row in pearl_rows] == [
        ("AAA", "pearl", 120.0)
    ]


def test_watchlist_queue_ranks_buy_confirmed_above_active(monkeypatch, tmp_path):
    # BUY_CONFIRMED (catalyst-confirmed, within reach) is at least as
    # actionable as DEPLOY_READY and must outrank ACTIVE at the same conviction +
    # confidence. status is sort key #2 (before distance), so with both rows
    # ACTIONABLE/HIGH the status priority alone decides the order.
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            ticker="CONF",
            source_run_id="run_conf",
            status="BUY_CONFIRMED",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            buy_price_target=100.0,
            current_price_at_addition=80.0,
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker="ACTV",
            source_run_id="run_actv",
            status="ACTIVE",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            buy_price_target=100.0,
            current_price_at_addition=120.0,
        ),
        db_path=db_path,
    )

    tickers = [row["ticker"] for row in watchlist_queue(limit=10, db_path=db_path)]

    assert "CONF" in tickers
    assert tickers.index("CONF") < tickers.index("ACTV")


def test_watchlist_queue_uses_most_recent_snapshot_timestamp(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    watchlist_id = add_or_update(
        _entry(
            ticker="AAA",
            source_run_id="run_snapshot_order",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            buy_price_target=80.0,
            current_price_at_addition=95.0,
        ),
        db_path=db_path,
    )
    add_price_snapshot(
        watchlist_id,
        price=70.0,
        checked_at="2026-05-19T12:00:00Z",
        source="fresh_snapshot",
        db_path=db_path,
    )
    add_price_snapshot(
        watchlist_id,
        price=90.0,
        checked_at="2026-05-18T12:00:00Z",
        source="older_snapshot_inserted_later",
        db_path=db_path,
    )

    rows = watchlist_queue(limit=5, db_path=db_path)

    assert rows == [
        {
            "id": 1,
            "ticker": "AAA",
            "status": "ACTIVE",
            "presented_status": "ACTIVE",
            "event_pending": None,
            "conviction_grade": "WATCHLIST_ONLY",
            "confidence": "MODERATE",
            "conviction_source": "company_autonomy",
            "pipeline_version": None,
            "candidate_disposition": None,
            "decision_basis": None,
            "selection_validation_status": None,
            "price_trigger_eligible": True,
            "scan_family": "normal",
            "latest_price": 70.0,
            "latest_price_basis": "SNAPSHOT",
            "price_at_addition": 95.0,
            "latest_price_source": "fresh_snapshot",
            "latest_price_checked_at": "2026-05-19T12:00:00Z",
            "buy_price_target": 80.0,
            "distance_from_buy_pct": -12.5,
            "valuation_anchor_method": "DCF",
            "valuation_anchor_value": 106.67,
            "source_sector": "industrial_tech",
            "status_reason": None,
            "added_at": "2026-05-08T12:00:00+00:00",
            "last_evaluated_at": None,
            "falsifiers": ["Revenue decline persists"],
            "market_cap_mm": None,
            "cap_source": None,
            "cap_band": None,
            "cap_band_label": "UNKNOWN_CAP",
            "adv_dollar_20d": None,
            "adv_dollar_60d": None,
            "adv_asof": None,
            "capacity_class": "ADV_UNKNOWN",
            "cheapness": "n/a",
        }
    ]


def test_watchlist_queue_says_when_the_price_is_only_the_price_at_addition(monkeypatch, tmp_path):
    """A row with no price snapshot still shows its addition price (investor buy-now renders
    it with a no-snapshot caveat), but the row now says it is the price at addition rather
    than passing it off as the latest price; a row with a snapshot says SNAPSHOT."""
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(ticker="NOSNAP", buy_price_target=80.0, current_price_at_addition=95.0),
        db_path=db_path,
    )
    snapped_id = add_or_update(
        _entry(ticker="SNAPPED", buy_price_target=80.0, current_price_at_addition=95.0),
        db_path=db_path,
    )
    add_price_snapshot(
        snapped_id, price=70.0, checked_at="2026-05-19T12:00:00Z", source="test", db_path=db_path
    )
    rows = {row["ticker"]: row for row in watchlist_queue(limit=5, db_path=db_path)}

    nosnap = rows["NOSNAP"]
    assert nosnap["latest_price"] == 95.0
    assert nosnap["latest_price_basis"] == "PRICE_AT_ADDITION"
    assert nosnap["latest_price_checked_at"] is None
    assert nosnap["latest_price_source"] is None
    assert nosnap["price_at_addition"] == 95.0
    assert rows["SNAPPED"]["latest_price"] == 70.0
    assert rows["SNAPPED"]["latest_price_basis"] == "SNAPSHOT"
    assert rows["SNAPPED"]["price_at_addition"] == 95.0


# ---------------------------------------------------------------------------
# Per-name margin-of-safety discount wired into watchlist persistence.
# ---------------------------------------------------------------------------


def _discount_packet(
    ticker: str,
    *,
    current_price: float,
    anchor: float,
    intrinsic_low: float | None = None,
    intrinsic_high: float | None = None,
    buy_below_price: float | None = None,
) -> SectorCompanyFinancialPacket:
    valuation: dict = {"anchor_method": "DCF", "valuation_anchor": anchor}
    if intrinsic_low is not None:
        valuation["intrinsic_range_low"] = intrinsic_low
    if intrinsic_high is not None:
        valuation["intrinsic_range_high"] = intrinsic_high
    if buy_below_price is not None:
        valuation["buy_below_price"] = buy_below_price
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        current_price=current_price,
        valuation=valuation,
    )


def _discount_artifact(
    *,
    ticker: str,
    packet: SectorCompanyFinancialPacket,
    verdict: str,
    confidence: str,
) -> AutonomousSectorFinancialRunArtifact:
    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_discount_test",
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        objective="Exercise the per-name margin-of-safety discount.",
        as_of_date=_ARTIFACT_ASOF,
        created_at=f"{_ARTIFACT_ASOF}T12:00:00Z",
        completed_at=f"{_ARTIFACT_ASOF}T12:01:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker=None,
        confidence="MODERATE",
        candidate_selection={"selected_tickers": [ticker]},
        company_packets=[packet],
        relative_ranking=[
            {
                "ticker": ticker,
                "company_autonomy_verdict": verdict,
                "company_autonomy_confidence": confidence,
                "positioning_summary": f"{ticker} fixture for discount wiring.",
            }
        ],
        memo_body={"candidates": {}},
    )


def test_populate_actionable_high_uses_12pct_discount(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    packet = _discount_packet(
        "AAA",
        current_price=70.0,
        anchor=100.0,
        intrinsic_low=95.0,
        intrinsic_high=105.0,
    )
    artifact = _discount_artifact(
        ticker="AAA", packet=packet, verdict="ACTIONABLE", confidence="HIGH"
    )

    populate_from_sector_artifact(artifact, db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.conviction_grade == "ACTIONABLE"
    assert latest.confidence == "HIGH"
    assert latest.buy_price_target == 88.0


def test_populate_watchlist_low_uses_40pct_discount(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    packet = _discount_packet(
        "BBB",
        current_price=70.0,
        anchor=80.0,
        intrinsic_low=40.0,
        intrinsic_high=120.0,
    )
    artifact = _discount_artifact(
        ticker="BBB", packet=packet, verdict="WATCHLIST_ONLY", confidence="LOW"
    )

    populate_from_sector_artifact(artifact, db_path=db_path)
    latest = get_latest("BBB", db_path=db_path)

    assert latest is not None
    assert latest.conviction_grade == "WATCHLIST_ONLY"
    assert latest.confidence == "LOW"
    assert latest.buy_price_target == 48.0


def test_populate_explicit_buy_below_price_still_wins(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    packet = _discount_packet(
        "CCC",
        current_price=60.0,
        anchor=100.0,
        intrinsic_low=95.0,
        intrinsic_high=105.0,
        buy_below_price=70.0,
    )
    artifact = _discount_artifact(
        ticker="CCC", packet=packet, verdict="ACTIONABLE", confidence="HIGH"
    )

    populate_from_sector_artifact(artifact, db_path=db_path)
    latest = get_latest("CCC", db_path=db_path)

    assert latest is not None
    assert latest.buy_price_target == 70.0


def test_populate_no_range_watchlist_moderate_is_legacy_neutral(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    packet = _discount_packet(
        "DDD",
        current_price=70.0,
        anchor=106.67,
    )
    artifact = _discount_artifact(
        ticker="DDD", packet=packet, verdict="WATCHLIST_ONLY", confidence="MODERATE"
    )

    populate_from_sector_artifact(artifact, db_path=db_path)
    latest = get_latest("DDD", db_path=db_path)

    assert latest is not None
    assert latest.buy_price_target is not None
    assert round(latest.buy_price_target, 4) == 80.0025


def test_store_and_sector_report_buy_price_target_agree():
    from app.autonomous import sector_report
    from app.watchlist import store

    packet = _discount_packet(
        "EEE",
        current_price=70.0,
        anchor=100.0,
        intrinsic_low=90.0,
        intrinsic_high=110.0,
    )

    store_value = store._buy_price_target(packet, conviction_grade="ACTIONABLE", confidence="HIGH")
    report_value = sector_report._buy_price_target(
        packet, conviction_grade="ACTIONABLE", confidence="HIGH"
    )

    assert store_value == report_value
    assert store_value == 86.0


def test_memo_buy_price_threads_live_volatility(monkeypatch):
    # The memo's _buy_price_target must consult live realized volatility
    # (same call the store makes), not hardcode realized_volatility=None — else
    # memo$ diverges from DB$ the moment the vol weight is raised above 0.
    from app.autonomous import sector_report

    captured: dict[str, object] = {}

    def _spy_compute_buy_target(
        anchor,
        *,
        conviction_grade=None,
        confidence=None,
        intrinsic_low=None,
        intrinsic_high=None,
        realized_volatility=None,
        **_,
    ):
        captured["rv"] = realized_volatility
        return anchor

    monkeypatch.setattr(sector_report, "compute_buy_target", _spy_compute_buy_target)
    monkeypatch.setattr(
        sector_report,
        "realized_volatility",
        lambda ticker, db_path=None: 0.42,
        raising=False,
    )

    packet = _discount_packet(
        "VOLT", current_price=70.0, anchor=100.0, intrinsic_low=90.0, intrinsic_high=110.0
    )
    sector_report._buy_price_target(packet, conviction_grade="ACTIONABLE", confidence="HIGH")

    assert captured["rv"] == 0.42


def test_memo_render_buy_price_matches_persisted_db_value(monkeypatch, tmp_path):
    # The memo's table AND candidate-section buy-price lines must
    # show the grade-scaled value the DB persists for a graded ACTIONABLE/HIGH
    # name with an intrinsic range — NOT the neutral anchor*0.75. Before the fix
    # the memo called _buy_price_target(packet) without grade/confidence and
    # rendered $75.00 while the DB stored $88.00.
    from app.autonomous import sector_report

    db_path = _init_temp_db(monkeypatch, tmp_path)
    packet = _discount_packet(
        "AAA",
        current_price=70.0,
        anchor=100.0,
        intrinsic_low=95.0,
        intrinsic_high=105.0,
    )
    artifact = _discount_artifact(
        ticker="AAA", packet=packet, verdict="ACTIONABLE", confidence="HIGH"
    )

    populate_from_sector_artifact(artifact, db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None
    assert latest.buy_price_target == 88.0

    table_lines = sector_report._render_candidate_table(artifact)
    aaa_row = next(line for line in table_lines if line.startswith("| AAA |"))
    # Grade-scaled buy target appears; neither the legacy flat $75.00 nor the
    # pre-fix neutral-base $78.00 (0.20 base + 0.20*0.10 dispersion) leaks in.
    assert "$88.00" in aaa_row
    assert "$78.00" not in aaa_row
    assert "$75.00" not in aaa_row

    section_lines = sector_report._render_candidate_section(artifact, "AAA", packet)
    buy_line = next(line for line in section_lines if "buy-price target:" in line)
    assert "buy-price target: $88.00" in buy_line
    assert "$78.00" not in buy_line
    assert "$75.00" not in buy_line


def test_current_row_newer_watchlist_only_low_supersedes_actionable_high(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            source_run_id="run_strong",
            added_at="2026-05-17T01:54:06+00:00",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            source_run_id="run_weak",
            added_at="2026-06-13T20:42:04+00:00",
            conviction_grade="WATCHLIST_ONLY",
            confidence="LOW",
        ),
        db_path=db_path,
    )

    latest = get_latest("AAA", db_path=db_path)
    active = list_active(db_path=db_path)
    queue = watchlist_queue(db_path=db_path)

    assert latest is not None
    assert latest.conviction_grade == "WATCHLIST_ONLY"
    assert latest.confidence == "LOW"
    assert latest.source_run_id == "run_weak"
    assert [e.source_run_id for e in active] == ["run_weak"]
    assert [row["conviction_grade"] for row in queue] == ["WATCHLIST_ONLY"]


def test_current_row_same_grade_uses_newer_assessment_even_with_lower_confidence(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(source_run_id="run_moderate", confidence="MODERATE"),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            source_run_id="run_low",
            added_at="2026-06-13T20:42:04+00:00",
            confidence="LOW",
        ),
        db_path=db_path,
    )

    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.source_run_id == "run_low"
    assert latest.confidence == "LOW"


def test_newer_active_status_supersedes_older_deploy_ready(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(source_run_id="run_old", status="DEPLOY_READY"),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            source_run_id="run_new",
            added_at="2026-06-13T20:42:04+00:00",
            status="ACTIVE",
        ),
        db_path=db_path,
    )

    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.source_run_id == "run_new"
    assert latest.status == "ACTIVE"


def test_current_row_equal_strength_prefers_newest(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(_entry(source_run_id="run_old"), db_path=db_path)
    add_or_update(
        _entry(source_run_id="run_new", added_at="2026-06-13T20:42:04+00:00"),
        db_path=db_path,
    )

    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.source_run_id == "run_new"


def test_current_row_newest_avoid_supersedes_stronger_history(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            source_run_id="run_strong",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            source_run_id="run_avoid",
            added_at="2026-06-13T20:42:04+00:00",
            conviction_grade="AVOID",
            confidence="LOW",
        ),
        db_path=db_path,
    )

    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.conviction_grade == "AVOID"
    assert latest.source_run_id == "run_avoid"


@pytest.mark.parametrize(
    ("ticker", "new_status"),
    [
        ("GPOR", "DEPLOY_READY"),
        ("BOSC", "ACTIVE"),
        ("TDW", "ACTIVE"),
        ("BAM", "ACTIVE"),
    ],
)
def test_named_newer_downgrade_regressions_select_new_assessment(
    monkeypatch, tmp_path, ticker, new_status
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            ticker=ticker,
            source_run_id=f"{ticker.lower()}_old_actionable",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker=ticker,
            source_run_id=f"{ticker.lower()}_new_watchlist",
            added_at="2026-07-20T00:00:00+00:00",
            status=new_status,
            conviction_grade="WATCHLIST_ONLY",
            confidence="LOW",
        ),
        db_path=db_path,
    )

    latest = get_latest(ticker, db_path=db_path)

    assert latest is not None
    assert latest.source_run_id == f"{ticker.lower()}_new_watchlist"
    assert latest.conviction_grade == "WATCHLIST_ONLY"
    assert latest.confidence == "LOW"
    assert latest.status == new_status


def test_sector_pipeline_version_filtering_remains_fail_closed(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            ticker="UPGRADE",
            source_run_id="autonomous_sector_old_v1",
            conviction_grade="ACTIONABLE",
        ),
        db_path=db_path,
    )
    add_or_update(
        WatchlistEntry(
            ticker="UPGRADE",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            conviction_source="sector_screen",
            source_run_id="autonomous_sector_new_v2",
            source_sector="energy",
            added_at="2026-07-20T00:00:00+00:00",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )
    add_or_update(
        WatchlistEntry(
            ticker="ROLLBACK",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            conviction_source="sector_screen",
            source_run_id="autonomous_sector_old_v2",
            source_sector="energy",
            added_at="2026-07-19T00:00:00+00:00",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            ticker="ROLLBACK",
            source_run_id="autonomous_sector_new_v1",
            added_at="2026-07-20T00:00:00+00:00",
            conviction_grade="ACTIONABLE",
        ),
        db_path=db_path,
    )

    assert get_latest("UPGRADE", db_path=db_path).source_run_id == ("autonomous_sector_new_v2")
    assert get_latest("ROLLBACK", db_path=db_path).source_run_id == ("autonomous_sector_new_v1")


def test_trigger_and_status_mutations_target_authoritative_newer_row(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    old_id = add_or_update(
        _entry(
            source_run_id="run_old",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
        ),
        db_path=db_path,
    )
    new_id = add_or_update(
        _entry(
            source_run_id="run_new",
            added_at="2026-07-20T00:00:00+00:00",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
        ),
        db_path=db_path,
    )

    record_trigger_status_change(
        "AAA",
        status="DEPLOY_READY",
        reason="Price crossed the review target.",
        db_path=db_path,
    )
    mark_status(
        "AAA",
        "CONTRADICTED",
        "New evidence contradicts the thesis.",
        "reevaluation",
        db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT id, status, status_reason FROM watchlist ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        (old_id, "DEPLOY_READY", None),
        (new_id, "CONTRADICTED", "New evidence contradicts the thesis."),
    ]


def test_remove_marks_only_authoritative_row_and_preserves_history(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            source_run_id="run_strong",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(
            source_run_id="run_weak",
            added_at="2026-06-13T20:42:04+00:00",
            confidence="LOW",
        ),
        db_path=db_path,
    )

    remove("AAA", "No longer fits.", db_path=db_path)

    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None
    assert latest.status == "REMOVED"
    assert list_active(db_path=db_path) == []
    assert watchlist_queue(db_path=db_path) == []
    conn = sqlite3.connect(str(db_path))
    try:
        statuses = [
            row[0] for row in conn.execute("SELECT status FROM watchlist WHERE ticker='AAA'")
        ]
    finally:
        conn.close()
    assert statuses == ["ACTIVE", "REMOVED"]


def test_legacy_newest_row_removed_keeps_ticker_hidden(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        _entry(
            source_run_id="run_strong",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
        ),
        db_path=db_path,
    )
    add_or_update(
        _entry(source_run_id="run_weak", added_at="2026-06-13T20:42:04+00:00"),
        db_path=db_path,
    )
    # Legacy removals flipped only the newest row; the older duplicate must
    # not resurface as the current row.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE watchlist SET status='REMOVED' WHERE ticker='AAA' AND source_run_id='run_weak'"
        )
        conn.commit()
    finally:
        conn.close()

    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None
    assert latest.status == "REMOVED"
    assert list_active(db_path=db_path) == []
    assert watchlist_queue(db_path=db_path) == []


def test_populate_flags_grade_narrative_mismatch_in_status_reason(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    artifact = _sector_artifact()
    artifact.memo_body["candidates"]["AAA"]["thesis"] = (
        "AAA compounds capital well, but returns sit below the sector's hurdle, "
        "which caps conviction and justifies watchlist positioning."
    )

    populate_from_sector_artifact(artifact, db_path=db_path)

    entry = get_latest("AAA", db_path=db_path)
    assert entry is not None
    assert entry.conviction_grade == "ACTIONABLE"
    assert entry.status_reason is not None
    assert "GRADE_NARRATIVE_MISMATCH" in entry.status_reason
    assert "thesis argues WATCHLIST_ONLY" in entry.status_reason


def test_populate_flags_ungrounded_missing_data_claim(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    artifact = _sector_artifact()
    artifact.memo_body["candidates"]["AAA"]["thesis"] = (
        "AAA is actionable, though the case is clouded by missing cash-flow "
        "and working-capital inputs."
    )

    populate_from_sector_artifact(artifact, db_path=db_path)

    entry = get_latest("AAA", db_path=db_path)
    assert entry is not None
    assert entry.status_reason is not None
    assert "UNGROUNDED_MISSING_DATA_CLAIM" in entry.status_reason


def test_populate_consistent_entry_carries_no_consistency_flag(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)

    populate_from_sector_artifact(_sector_artifact(), db_path=db_path)

    entry = get_latest("AAA", db_path=db_path)
    assert entry is not None
    assert entry.status_reason is None
