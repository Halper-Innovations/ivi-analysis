"""Outcomes read model + /api/outcomes.

The method scoreboard must agree with the outcome-tracker resolver: fixture
rows are produced by ``evaluate_method_outcome`` itself (the pure resolver
evaluator) and persisted in the exact column shape ``_persist_outcome``
writes, so the scoreboard's aggregation is proven against resolver output,
not a parallel reimplementation.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.calibration.calibration_report import build_calibration_report
from app.calibration.outcome_tracker import evaluate_method_outcome
from app.db import init_db
from app.watchlist.schema import ensure_watchlist_schema
from app.web.main import app
from app.web.readmodel import outcomes as outcomes_model

client = TestClient(app)


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    ensure_watchlist_schema(db_path)
    return cfg


def _insert_outcome(conn, *, ticker: str, as_of_date: str, run_id: str, **overrides):
    fields = {
        "decision": "BUY",
        "conviction": 3,
        "horizon_days": 180,
        "outcome_status": "CLOSED",
        "close_date": None,
        "entry_price": None,
        "realized_return_pct": None,
        "benchmark_return_pct": None,
        "excess_return_pct": None,
        **overrides,
    }
    cursor = conn.execute(
        """
        INSERT INTO ticker_outcomes (ticker, as_of_date, run_id, decision, conviction,
                                     horizon_days, outcome_status, close_date, entry_price,
                                     realized_return_pct, benchmark_return_pct,
                                     excess_return_pct, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            as_of_date,
            run_id,
            fields["decision"],
            fields["conviction"],
            fields["horizon_days"],
            fields["outcome_status"],
            fields["close_date"],
            fields["entry_price"],
            fields["realized_return_pct"],
            fields["benchmark_return_pct"],
            fields["excess_return_pct"],
            f"{as_of_date}T00:00:00+00:00",
            f"{as_of_date}T00:00:00+00:00",
        ),
    )
    return cursor.lastrowid


def test_method_scoreboard_matches_resolver_fixtures(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    conn = sqlite3.connect(cfg.db_path)

    # Resolver-evaluated fixtures: entry 100 → exit 120 (+20%).
    # dcf predicts 150 (UNDERVALUED, +20% > +10% confirm) → CORRECT
    # epv predicts 100 (FAIRLY_VALUED) → INCONCLUSIVE
    # graham predicts 60 (OVERVALUED, +20% > +15% disconfirm) → INCORRECT
    dcf = evaluate_method_outcome("dcf", 150.0, 100.0, 120.0)
    epv = evaluate_method_outcome("epv", 100.0, 100.0, 120.0)
    graham = evaluate_method_outcome("graham", 60.0, 100.0, 120.0, unadjusted_source=True)
    assert dcf.outcome == "CORRECT"
    assert epv.outcome == "INCONCLUSIVE"
    assert graham.outcome == "INCORRECT"

    def source_outcome(ticker: str, as_of_date: str) -> int:
        valuation = conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at, source_run_id,
                source_artifact_path, source_artifact_sha256,
                financial_integrity_fingerprint
            ) VALUES (
                ?, ?, 'deep_research', '{}', '{}', '[]', ?,
                ?, ?, ?, ?
            )
            """,
            (
                ticker,
                as_of_date,
                f"{as_of_date}T00:00:00+00:00",
                f"source_{ticker}",
                f"/fixture/{ticker}.json",
                "a" * 64,
                f"fingerprint_{ticker}",
            ),
        )
        outcome = conn.execute(
            """
            INSERT INTO deep_research_outcomes(
                ticker, source_as_of_date, source_valuation_id, source_run_id,
                source_artifact_path, source_artifact_sha256,
                source_valuation_fingerprint, horizon_days,
                target_exit_date, scan_date, entry_price, exit_price,
                exit_as_of_date, exit_source, price_change_pct, thesis_verdict,
                verdict_outcome, created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, 90,
                '2026-07-01', '2026-07-02', 100.0, 120.0,
                      '2026-07-01', 'fixture', 20.0, 'UNDERVALUED',
                'CORRECT', '2026-07-02T00:00:00+00:00'
            )
            """,
            (
                ticker,
                as_of_date,
                valuation.lastrowid,
                f"source_{ticker}",
                f"/fixture/{ticker}.json",
                "a" * 64,
                f"fingerprint_{ticker}",
            ),
        )
        return int(outcome.lastrowid)

    outcome_id = source_outcome("AAA", "2026-01-10")
    for method_outcome in (dcf, epv, graham):
        # _persist_outcome stores the report's entry price alongside.
        conn.execute(
            """INSERT OR REPLACE INTO deep_research_method_outcomes
               (outcome_id, method, predicted_value, entry_price, direction,
                outcome, unadjusted_source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                outcome_id,
                method_outcome.method,
                method_outcome.predicted_value,
                100.0,
                method_outcome.direction,
                method_outcome.outcome,
                1 if method_outcome.unadjusted_source else 0,
                "2026-07-01T00:00:00+00:00",
            ),
        )
    # A second month: dcf misses (predicts 150, price falls 20%).
    dcf_miss = evaluate_method_outcome("dcf", 150.0, 100.0, 80.0)
    assert dcf_miss.outcome == "INCORRECT"
    outcome_id_2 = source_outcome("BBB", "2026-02-05")
    conn.execute(
        """INSERT OR REPLACE INTO deep_research_method_outcomes
           (outcome_id, method, predicted_value, entry_price, direction,
            outcome, unadjusted_source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            outcome_id_2,
            "dcf",
            150.0,
            100.0,
            dcf_miss.direction,
            dcf_miss.outcome,
            0,
            "2026-07-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    with sqlite3.connect(cfg.db_path) as read_conn:
        read_conn.row_factory = sqlite3.Row
        board = outcomes_model.method_scoreboard(read_conn)

    assert board["methods"] == [
        {
            "method": "dcf",
            "n": 2,
            "correct": 1,
            "incorrect": 1,
            "inconclusive": 0,
            "hit_rate": 0.5,
            "unadjusted_source": False,
        },
        {
            "method": "epv",
            "n": 1,
            "correct": 0,
            "incorrect": 0,
            "inconclusive": 1,
            "hit_rate": None,
            "unadjusted_source": False,
        },
        {
            "method": "graham",
            "n": 1,
            "correct": 0,
            "incorrect": 1,
            "inconclusive": 0,
            "hit_rate": 0.0,
            "unadjusted_source": True,
        },
    ]
    assert board["monthly"] == [
        {"month": "2026-01", "method": "dcf", "n": 1, "correct": 1, "incorrect": 0},
        {"month": "2026-01", "method": "epv", "n": 1, "correct": 0, "incorrect": 0},
        {"month": "2026-01", "method": "graham", "n": 1, "correct": 0, "incorrect": 1},
        {"month": "2026-02", "method": "dcf", "n": 1, "correct": 0, "incorrect": 1},
    ]


def test_realized_returns_slices_and_histogram(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    conn = sqlite3.connect(cfg.db_path)
    _insert_outcome(
        conn,
        ticker="AAA",
        as_of_date="2026-01-10",
        run_id="r1",
        decision="BUY",
        conviction=4,
        excess_return_pct=8.0,
        realized_return_pct=12.0,
    )
    _insert_outcome(
        conn,
        ticker="BBB",
        as_of_date="2026-01-11",
        run_id="r1",
        decision="BUY",
        conviction=4,
        excess_return_pct=25.0,
        realized_return_pct=30.0,
    )
    _insert_outcome(
        conn,
        ticker="CCC",
        as_of_date="2026-01-12",
        run_id="r1",
        decision="WATCH",
        conviction=2,
        excess_return_pct=-12.0,
        realized_return_pct=-8.0,
    )
    _insert_outcome(
        conn,
        ticker="DDD",
        as_of_date="2026-01-13",
        run_id="r1",
        decision="WATCH",
        conviction=2,
        excess_return_pct=-60.0,
        realized_return_pct=-55.0,
    )
    # Excluded: still open / closed without an excess record.
    _insert_outcome(
        conn,
        ticker="EEE",
        as_of_date="2026-01-14",
        run_id="r1",
        outcome_status="OPEN",
        excess_return_pct=99.0,
    )
    _insert_outcome(conn, ticker="FFF", as_of_date="2026-01-15", run_id="r1")
    conn.commit()
    conn.close()

    with sqlite3.connect(cfg.db_path) as read_conn:
        read_conn.row_factory = sqlite3.Row
        returns = outcomes_model.realized_returns(read_conn)

    # Open rows never leak into the record: EEE's +99 excess is absent.
    assert returns["overall"] == {
        "n": 4,
        "avg_excess": -9.75,
        "median_excess": -2.0,
        "hit_rate": 0.5,
        "avg_realized": -5.25,
    }
    non_empty = [b for b in returns["histogram"] if b["count"] > 0]
    assert non_empty == [
        {"low": None, "high": -50.0, "count": 1},  # -60
        {"low": -20.0, "high": -10.0, "count": 1},  # -12
        {"low": 0.0, "high": 10.0, "count": 1},  # +8
        {"low": 20.0, "high": 30.0, "count": 1},  # +25
    ]
    assert returns["by_decision"] == [
        {
            "decision": "BUY",
            "n": 2,
            "avg_excess": 16.5,
            "median_excess": 16.5,
            "hit_rate": 1.0,
            "avg_realized": 21.0,
        },
        {
            "decision": "WATCH",
            "n": 2,
            "avg_excess": -36.0,
            "median_excess": -36.0,
            "hit_rate": 0.0,
            "avg_realized": -31.5,
        },
    ]
    assert returns["by_conviction"] == [
        {
            "conviction": 2,
            "n": 2,
            "avg_excess": -36.0,
            "median_excess": -36.0,
            "hit_rate": 0.0,
            "avg_realized": -31.5,
        },
        {
            "conviction": 4,
            "n": 2,
            "avg_excess": 16.5,
            "median_excess": 16.5,
            "hit_rate": 1.0,
            "avg_realized": 21.0,
        },
    ]


def test_journal_ledger_and_goal(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO dispositions (ticker, kind, status, opened_at, opened_by,
                                  trigger_snapshot_json)
        VALUES ('AAA', 'AT_TARGET', 'OPEN', '2026-07-10T00:00:00+00:00', 'system', '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO dispositions (ticker, kind, status, opened_at, opened_by,
                                  trigger_snapshot_json, decided_at, operator,
                                  reason_code, rationale)
        VALUES ('TDW', 'EVENT_DISPOSAL', 'PASSED', '2026-07-01T00:00:00+00:00', 'owner',
                '{}', '2026-07-02T00:00:00+00:00', 'owner', 'EVENT_REVIEWED', 'Routine 8-K.')
        """
    )
    conn.execute(
        """
        INSERT INTO dispositions (ticker, kind, status, opened_at, opened_by,
                                  trigger_snapshot_json, decided_at, operator,
                                  reason_code, rationale, intended_size)
        VALUES ('NCSM', 'AT_TARGET', 'PASSED', '2026-06-01T00:00:00+00:00', 'owner',
                '{}', '2026-06-12T00:00:00+00:00', 'owner', 'THESIS_BROKEN',
                'Acquisition announced.', 'none')
        """
    )
    conn.commit()
    conn.close()

    with sqlite3.connect(cfg.db_path) as read_conn:
        read_conn.row_factory = sqlite3.Row
        ledger = outcomes_model.journal(read_conn)

    assert ledger["decided_count"] == 2
    assert ledger["open_count"] == 1
    assert ledger["goal"] == 6
    assert ledger["goal_deadline"] == "2026-09-09"
    assert [entry["ticker"] for entry in ledger["entries"]] == ["TDW", "NCSM"]
    assert ledger["entries"][0]["reason_code"] == "EVENT_REVIEWED"
    assert ledger["entries"][1]["intended_size"] == "none"


def test_calibration_reports_unpacked(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    report_payload = {
        "run_id": "calibration_report_2026-06-02",
        "source_run_ids": ["r1"],
        "source_outcomes": [
            {
                "schema_version": "ticker_outcome_report_binding_v1",
                "outcome_state": {"id": 1, "run_id": "r1", "ticker": "AAA"},
                "financial_integrity_fingerprint": "legacy-test-binding",
            }
        ],
        "report_family": "grade_status_calibration",
        "headline_hit_metric": "excess_return_pct>0",
        "overall": {
            "n": 12,
            "hit_rate": 0.5833,
            "excess_hit_rate": 0.5,
            "avg_return": 3.2,
            "avg_excess": 1.1,
            "median_return": 2.0,
            "target_hit_rate": 0.25,
        },
        "by_grade": {"DEPLOY_READY": {"n": 4}},
        "by_status": {},
    }
    report_path = cfg.calibration_dir / "calibration_report_2026-06-02.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO calibration_reports (run_id, as_of_date, report_json, report_path, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "calibration_report_2026-06-02",
            "2026-06-02",
            json.dumps(report_payload),
            str(report_path),
            "2026-06-03T04:07:01+00:00",
        ),
    )
    conn.commit()
    conn.close()

    with sqlite3.connect(cfg.db_path) as read_conn:
        read_conn.row_factory = sqlite3.Row
        reports = outcomes_model.calibration(read_conn)

    assert len(reports) == 1
    report = reports[0]
    assert report["run_id"] == "calibration_report_2026-06-02"
    assert report["report_family"] == "grade_status_calibration"
    assert report["headline_hit_metric"] == "excess_return_pct>0"
    assert report["overall"]["n"] == 12
    assert report["overall"]["hit_rate"] == 0.5833
    assert report["by_grade"] == {"DEPLOY_READY": {"n": 4}}


@pytest.mark.financial_integrity_contract
def test_calibration_invalid_newest_family_does_not_resurrect_older_report(
    monkeypatch,
    tmp_path,
):
    from tests.test_classic_postwrite_authorization import _baseline_manifest

    _baseline_manifest(monkeypatch, tmp_path)
    cfg = _init_temp_env(monkeypatch, tmp_path)
    old_payload = build_calibration_report("2026-06-01")
    newest_path = cfg.calibration_dir / "calibration_report_2026-06-02.json"
    with sqlite3.connect(cfg.db_path) as old_conn:
        old_conn.row_factory = sqlite3.Row
        old_reports = outcomes_model.calibration(old_conn)
    assert [report["run_id"] for report in old_reports] == [old_payload["run_id"]]

    newest_path.write_text("{invalid json", encoding="utf-8")
    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO calibration_reports(
            run_id, as_of_date, report_json, report_path, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            "calibration_report_2026-06-02",
            "2026-06-02",
            "{}",
            str(newest_path),
            "2099-06-02T12:00:00+00:00",
        ),
    )
    conn.commit()
    conn.row_factory = sqlite3.Row

    assert outcomes_model.calibration(conn) == []
    conn.close()


def test_api_outcomes_end_to_end(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    conn = sqlite3.connect(cfg.db_path)
    _insert_outcome(
        conn,
        ticker="AAA",
        as_of_date="2026-01-10",
        run_id="r1",
        decision="BUY",
        conviction=3,
        excess_return_pct=5.0,
        realized_return_pct=9.0,
    )
    conn.commit()
    conn.close()

    response = client.get("/api/outcomes")
    assert response.status_code == 200
    body = response.json()
    assert body["scoreboard"]["methods"] == []
    assert body["returns"]["overall"]["n"] == 1
    assert body["returns"]["overall"]["avg_excess"] == 5.0
    assert body["journal"]["decided_count"] == 0
    assert body["journal"]["goal"] == 6
    assert body["calibration"] == []
