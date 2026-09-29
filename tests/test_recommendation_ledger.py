from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.calibration.recommendation_ledger import (
    MISSING_PRE_MORTEM,
    MISSING_RISK_FLAG,
    RecommendationDraft,
    get_recommendation,
    preview_open_at_target_seed,
    seed_open_at_target_dispositions,
    stake_recommendation,
)
from app.cli import app
from app.config import get_config
from app.db import connect, init_db
from app.watchlist.schema import ensure_watchlist_schema


runner = CliRunner()


@pytest.fixture
def ledger_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema(db_path)
    yield db_path
    get_config.cache_clear()


def _draft(
    recommendation_id: str,
    *,
    ticker: str = "AAA",
    trigger_price: float = 80.0,
    trigger_price_as_of: str = "2026-07-28T16:00:00+00:00",
    trigger_price_age_seconds: int = 3600,
    staked_at: str = "2026-07-28T17:00:00+00:00",
    corrects_recommendation_id: str | None = None,
    correction_reason: str | None = None,
) -> RecommendationDraft:
    return RecommendationDraft(
        recommendation_id=recommendation_id,
        ticker=ticker,
        recommendation_type="BUY_AT_LIMIT",
        model_id="deepseek",
        model_vintage="deepseek-chat-v3.1-2026-07",
        thesis_reference="run:relight-001#memo:AAA",
        thesis_summary="Cash generation supports the audited valuation at a disciplined entry.",
        trigger_price=trigger_price,
        trigger_price_source="eodhd:quote-001",
        trigger_price_as_of=trigger_price_as_of,
        trigger_price_age_seconds=trigger_price_age_seconds,
        target_price=75.0,
        target_price_source="audited_memo:dcf_discount_band",
        conviction_grade="ACTIONABLE",
        capacity_class="MODERATE",
        adv_dollar_20d=5_000_000.0,
        adv_as_of="2026-07-28",
        pre_mortem="Margins contract and the apparent discount proves cyclical.",
        risk_flags=("MARGIN_CONTRACTION", "CYCLICAL_DEMAND"),
        policy_hash="a" * 64,
        source_run_id="relight-001",
        horizons=(90, 365),
        benchmark_symbol="IWM",
        staked_at=staked_at,
        corrects_recommendation_id=corrects_recommendation_id,
        correction_reason=correction_reason,
    )


def test_schema_is_new_table_only_with_complete_resolution_fields(ledger_db: Path) -> None:
    conn = sqlite3.connect(ledger_db)
    try:
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(recommendation_ledger)")
        }
    finally:
        conn.close()

    assert columns == {
        "id",
        "recommendation_id",
        "ticker",
        "recommendation_type",
        "record_vintage",
        "model_id",
        "model_vintage",
        "thesis_reference",
        "thesis_summary",
        "trigger_price",
        "trigger_price_source",
        "trigger_price_as_of",
        "trigger_price_age_seconds",
        "target_price",
        "target_price_source",
        "conviction_grade",
        "capacity_class",
        "adv_dollar_20d",
        "adv_as_of",
        "pre_mortem",
        "risk_flags_json",
        "policy_hash",
        "source_run_id",
        "horizons_json",
        "benchmark_symbol",
        "staked_at",
        "recorded_at",
        "source_disposition_id",
        "corrects_recommendation_id",
        "correction_reason",
    }


def test_staked_rows_are_immutable_and_corrections_append(ledger_db: Path) -> None:
    original = stake_recommendation(_draft("rec-001"), db_path=ledger_db)
    assert original["record_vintage"] == "LIVE"
    assert original["risk_flags"] == ["MARGIN_CONTRACTION", "CYCLICAL_DEMAND"]
    assert original["horizons"] == [90, 365]
    assert original["source_disposition_id"] is None

    conn = sqlite3.connect(ledger_db)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE recommendation_ledger SET target_price = 74.0 "
                "WHERE recommendation_id = 'rec-001'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "DELETE FROM recommendation_ledger WHERE recommendation_id = 'rec-001'"
            )
        conn.rollback()
    finally:
        conn.close()

    correction = stake_recommendation(
        _draft(
            "rec-002",
            trigger_price=79.0,
            trigger_price_as_of="2026-07-28T18:00:00+00:00",
            trigger_price_age_seconds=3600,
            staked_at="2026-07-28T19:00:00+00:00",
            corrects_recommendation_id="rec-001",
            correction_reason="Correct the trigger quote after provenance review.",
        ),
        db_path=ledger_db,
    )
    assert correction["corrects_recommendation_id"] == "rec-001"
    assert correction["correction_reason"] == (
        "Correct the trigger quote after provenance review."
    )
    assert get_recommendation("rec-001", db_path=ledger_db)["trigger_price"] == 80.0

    with pytest.raises(ValueError, match="keep the predecessor ticker"):
        stake_recommendation(
            _draft(
                "rec-003",
                ticker="BBB",
                corrects_recommendation_id="rec-001",
                correction_reason="Wrong issuer.",
            ),
            db_path=ledger_db,
        )


def _insert_seed_source(
    conn: sqlite3.Connection,
    *,
    watchlist_id: int,
    disposition_id: int,
    ticker: str,
    grade: str,
    target: float,
    trigger: float,
    risks: list[str],
    falsifiers: list[str],
) -> None:
    conn.execute(
        """
        INSERT INTO watchlist(
            id, ticker, status, conviction_grade, confidence, conviction_source,
            valuation_anchor_method, valuation_anchor_value, buy_price_target,
            current_price_at_addition, thesis_text, key_risks_json,
            falsifiers_json, open_questions_json, source_run_id, source_sector,
            added_at, cap_band, adv_dollar_20d, adv_asof, capacity_class
        )
        VALUES (?, ?, 'DEPLOY_READY', ?, 'MODERATE', 'company_autonomy',
                'DCF', 100.0, ?, 90.0, ?, ?, ?, '[]', ?, 'industrial_tech',
                '2026-07-01T12:00:00+00:00', 'mid', 2500000.0,
                '2026-07-27', 'MODERATE')
        """,
        (
            watchlist_id,
            ticker,
            grade,
            target,
            f"{ticker} has an audited historical thesis.",
            json.dumps(risks),
            json.dumps(falsifiers),
            f"historical-run-{ticker.lower()}",
        ),
    )
    conn.execute(
        """
        INSERT INTO watchlist_price_snapshots(
            id, watchlist_id, price, checked_at, source
        )
        VALUES (?, ?, ?, '2026-07-28T16:55:00+00:00', 'eodhd')
        """,
        (watchlist_id, watchlist_id, trigger),
    )
    conn.execute(
        """
        INSERT INTO dispositions(
            id, ticker, watchlist_id, kind, status, opened_at, opened_by,
            trigger_snapshot_json
        )
        VALUES (?, ?, ?, 'AT_TARGET', 'OPEN',
                '2026-07-28T17:00:00+00:00', 'watchlist_daily', ?)
        """,
        (
            disposition_id,
            ticker,
            watchlist_id,
            json.dumps(
                {
                    "watchlist_status": "DEPLOY_READY",
                    "conviction_grade": grade,
                    "buy_price_target": target,
                }
            ),
        ),
    )


def test_seed_import_pragmas_sources_and_never_creates_live_rows(ledger_db: Path) -> None:
    conn = connect(ledger_db)
    try:
        _insert_seed_source(
            conn,
            watchlist_id=10,
            disposition_id=20,
            ticker="AAA",
            grade="ACTIONABLE",
            target=75.0,
            trigger=72.0,
            risks=["CUSTOMER_CONCENTRATION"],
            falsifiers=["Customer concentration rises above the audited bound."],
        )
        _insert_seed_source(
            conn,
            watchlist_id=11,
            disposition_id=21,
            ticker="BBB",
            grade="WATCHLIST_ONLY",
            target=55.0,
            trigger=57.0,
            risks=[],
            falsifiers=[],
        )
        conn.commit()

        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        preview = preview_open_at_target_seed(conn=conn)
        conn.set_trace_callback(None)

        first_source_select = next(
            index
            for index, statement in enumerate(statements)
            if "FROM dispositions d" in statement
        )
        for table in ("dispositions", "watchlist", "watchlist_price_snapshots"):
            pragma_index = next(
                index
                for index, statement in enumerate(statements)
                if statement.strip().lower() == f"pragma table_info({table})"
            )
            assert pragma_index < first_source_select

        assert len(preview) == 2
        assert [row["record_vintage"] for row in preview] == ["SEED", "SEED"]
        assert [row["recommendation_type"] for row in preview] == [
            "BUY_AT_LIMIT",
            "WATCH",
        ]
        assert preview[0]["trigger_price_age_seconds"] == 300
        assert preview[0]["target_price"] == 75.0

        first = seed_open_at_target_dispositions(conn=conn)
        second = seed_open_at_target_dispositions(conn=conn)
        conn.commit()
        assert first == {"source_rows": 2, "inserted": 2, "existing": 0}
        assert second == {"source_rows": 2, "inserted": 0, "existing": 2}

        rows = conn.execute(
            """
            SELECT recommendation_id, record_vintage, risk_flags_json, pre_mortem
            FROM recommendation_ledger
            ORDER BY source_disposition_id
            """
        ).fetchall()
        assert len(rows) == 2
        assert all(row["record_vintage"] == "SEED" for row in rows)
        assert json.loads(rows[1]["risk_flags_json"]) == [MISSING_RISK_FLAG]
        assert rows[1]["pre_mortem"] == MISSING_PRE_MORTEM
        live_count = conn.execute(
            "SELECT COUNT(*) FROM recommendation_ledger WHERE record_vintage = 'LIVE'"
        ).fetchone()[0]
        assert live_count == 0
    finally:
        conn.close()


def test_cli_list_show_and_seed_preview_are_read_only_surfaces(ledger_db: Path) -> None:
    stake_recommendation(_draft("rec-cli"), db_path=ledger_db)

    listed = runner.invoke(
        app,
        ["recommendation-list", "--ticker", "AAA", "--vintage", "LIVE"],
    )
    assert listed.exit_code == 0
    listed_payload = json.loads(listed.stdout)
    assert listed_payload["count"] == 1
    assert listed_payload["rows"][0]["recommendation_id"] == "rec-cli"

    shown = runner.invoke(app, ["recommendation-show", "rec-cli"])
    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["target_price"] == 75.0

    preview = runner.invoke(app, ["recommendation-seed-preview"])
    assert preview.exit_code == 0
    assert json.loads(preview.stdout) == {
        "source_rows": 0,
        "record_vintage": "SEED",
        "type_counts": {},
        "preview_count": 0,
        "rows": [],
    }
