"""Company depth tabs: /api/company/{ticker}/(research|dossier|decisions).

Hermetic tmp-DB fixtures; exact-literal assertions. Research unpacks the
latest ``deep_research`` valuation row; the dossier rides the Reader's
allowlisted renderer; decisions mirror the dispositions store rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.autonomous.artifact_financial_audit import INVALID, PASS
from app.db import init_db
from app.watchlist.schema import ensure_watchlist_schema
from app.web.main import app
from app.web.readmodel import company_depth

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
    # Legacy endpoint examples below model an already-audited product book.
    # Dedicated tests in this module override these gates to exercise the
    # missing, invalid, and stale-current fail-closed cases.
    monkeypatch.setattr(
        company_depth,
        "financial_integrity_manifest_is_usable",
        lambda: True,
    )
    monkeypatch.setattr(
        company_depth,
        "artifact_decision_eligibility",
        lambda _path: PASS,
    )
    for row_gate in (
        "valuation_row_is_decision_eligible",
        "watchlist_row_is_decision_eligible",
        "outcome_row_is_decision_eligible",
    ):
        monkeypatch.setattr(
            company_depth,
            row_gate,
            lambda *_args, **_kwargs: True,
        )
    return cfg


_RESEARCH_OUTPUTS = {
    "status": "OK",
    "conviction_class": "MODERATE",
    "conviction_score": 55,
    "gate_action": "PROCEED",
    "tension_type": "GROWTH_VS_EARNINGS_POWER",
    "methods_agree": False,
    "consensus_strength": 4,
    "method_count": 4,
    "hypotheses_generated": 2,
    "report_path": "data/outputs/research/AAA_report.md",
    "thesis": {
        "original_dcf": 180.5,
        "adjusted_dcf": 112.85,
        "original_epv": 24.86,
        "adjusted_epv": 24.86,
        "original_graham": 135.39,
        "adjusted_intrinsic_mid": 68.85,
        "adjusted_margin_of_safety": -3.16,
        "adjusted_value_floored": False,
        "current_price": 286.82,
        "average_coverage": 0.375,
        "high_priority_unresolved": 1,
        "hypotheses_confirmed": 1,
        "hypotheses_contradicted": 0,
        "hypotheses_partially_confirmed": 0,
        "hypotheses_inconclusive": 1,
        "adjustments": [
            {
                "hypothesis_source": "GROWTH_VS_EARNINGS_POWER",
                "hypothesis_claim": "DCF assumes growth EPV does not support.",
                "hypothesis_direction": "BEARISH",
                "hypothesis_status": "CONFIRMED",
                "affected_method": "dcf",
                "adjustment_magnitude": -67.71,
                "adjustment_confidence": "FACT_CALIBRATED",
                "calibration_detail": "concentration 75% -> 3.0pp growth haircut",
            }
        ],
        "unresolved": [
            {
                "description": "retention metrics",
                "importance": "SUPPORTING",
                "unresolved_reason": "NOT_FOUND",
                "hypothesis_priority": "HIGH",
                "hypothesis_direction": "BEARISH",
            }
        ],
    },
    "analyst_notes": {
        "ticker": "AAA",
        "positives": [
            {
                "category": "POSITIVE",
                "claim": "Cash generation is strong.",
                "direction": "BULLISH",
                "severity": "HIGH",
                "suggested_adjustment": "Raise margin expectations.",
                "validation_status": "VERIFIED",
                "citations": [
                    {"section": "MDA", "excerpt": "operating cash flow rose", "block_id": "x:1"}
                ],
            }
        ],
        "risks": [],
        "surprises": [],
        "adjustment_triggers": [],
        "overall_assessment": "Momentum with concentration risk.",
        "filing_sections_read": ["MDA", "risk_factors"],
    },
    "citations": [
        {
            "citation_id": "C1",
            "section": "risk_factors",
            "excerpt": "Revenue increased 12%",
            "relevance": "revenue by segment",
            "hypothesis_source": "GROWTH_VS_EARNINGS_POWER",
            "source_form_type": "10-Q",
            "source_filing_date": "2026-04-30",
            "source_title": None,
            "source_url": None,
        }
    ],
}


def _seed_research(cfg) -> None:
    import sqlite3

    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json,
                                warnings_json, created_at)
        VALUES (?, ?, ?, '{}', ?, '[]', ?)
        """,
        (
            "AAA",
            "2026-07-15",
            "deep_research",
            json.dumps(_RESEARCH_OUTPUTS),
            "2026-07-16T04:00:00+00:00",
        ),
    )
    # An older pass that must lose the latest-row selection.
    conn.execute(
        """
        INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json,
                                warnings_json, created_at)
        VALUES (?, ?, ?, '{}', ?, '[]', ?)
        """,
        (
            "AAA",
            "2026-06-01",
            "deep_research",
            json.dumps({"status": "STALE"}),
            "2026-06-01T04:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO evidence_items (evidence_id, ticker, as_of_date, run_id, source_type,
                                    source_url, source_title, source_published_at,
                                    retrieved_at, excerpt_text, excerpt_hash,
                                    citations_json, item_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?)
        """,
        (
            "ev-1",
            "AAA",
            "2026-07-15",
            "run-1",
            "sec_filing",
            "https://example.test/10q",
            "Q2 filing",
            "2026-04-30",
            "2026-07-15T00:00:00+00:00",
            "Revenue increased 12%",
            "h1",
            "i1",
            "2026-07-15T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()


def test_research_unpacks_latest_deep_research_row(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_research(cfg)

    response = client.get("/api/company/AAA/research")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["as_of_date"] == "2026-07-15"
    assert body["status"] == "OK"
    assert body["conviction_class"] == "MODERATE"
    assert body["gate_action"] == "PROCEED"
    assert body["tension_type"] == "GROWTH_VS_EARNINGS_POWER"
    assert body["thesis"]["original_dcf"] == 180.5
    assert body["thesis"]["adjusted_dcf"] == 112.85
    assert body["thesis"]["adjusted_margin_of_safety"] == -3.16
    assert body["thesis"]["current_price"] == 286.82
    assert len(body["adjustments"]) == 1
    assert body["adjustments"][0]["affected_method"] == "dcf"
    assert body["adjustments"][0]["adjustment_magnitude"] == -67.71
    assert body["adjustments"][0]["adjustment_confidence"] == "FACT_CALIBRATED"
    assert len(body["unresolved"]) == 1
    assert body["unresolved"][0]["unresolved_reason"] == "NOT_FOUND"
    notes = body["analyst_notes"]
    assert notes["overall_assessment"] == "Momentum with concentration risk."
    assert len(notes["positives"]) == 1
    assert notes["positives"][0]["claim"] == "Cash generation is strong."
    assert notes["positives"][0]["citations"] == [
        {"section": "MDA", "excerpt": "operating cash flow rose"}
    ]
    assert notes["filing_sections_read"] == ["MDA", "risk_factors"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["citation_id"] == "C1"
    assert body["citations"][0]["source_form_type"] == "10-Q"
    assert body["report_path"] == "data/outputs/research/AAA_report.md"
    assert body["report_available"] is False
    assert len(body["evidence"]) == 1
    assert body["evidence"][0]["evidence_id"] == "ev-1"
    assert body["evidence"][0]["source_type"] == "sec_filing"
    assert body["evidence"][0]["excerpt"] == "Revenue increased 12%"


def test_research_absent_and_unknown_ticker(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    import sqlite3

    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json,
                                warnings_json, created_at)
        VALUES ('BBB', '2026-07-01', 'dcf', '{}', '{}', '[]', '2026-07-01T00:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()

    response = client.get("/api/company/BBB/research")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["reason"] == "NEVER_RESEARCHED"

    assert client.get("/api/company/ZZZ/research").status_code == 404


def test_research_and_decisions_fail_closed_when_manifest_unusable(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_research(cfg)
    monkeypatch.setattr(
        company_depth,
        "financial_integrity_manifest_is_usable",
        lambda: False,
    )

    def unexpected_gate(*_args, **_kwargs):
        raise AssertionError("row-level gates must not run without a usable manifest")

    for gate in (
        "artifact_decision_eligibility",
        "valuation_row_is_decision_eligible",
        "watchlist_row_is_decision_eligible",
        "outcome_row_is_decision_eligible",
    ):
        monkeypatch.setattr(company_depth, gate, unexpected_gate)

    research_response = client.get("/api/company/AAA/research")
    assert research_response.status_code == 200
    assert research_response.json()["available"] is False
    assert research_response.json()["reason"] == "FINANCIAL_INTEGRITY_AUDIT_UNAVAILABLE"
    assert research_response.json()["thesis"] is None
    assert research_response.json()["evidence"] == []

    decisions_response = client.get("/api/company/AAA/decisions")
    assert decisions_response.status_code == 200
    assert decisions_response.json()["dispositions"] == []
    assert decisions_response.json()["outcomes"] == []
    assert decisions_response.json()["outcomes_total"] == 0
    assert decisions_response.json()["outcomes_open"] == 0
    assert decisions_response.json()["outcomes_closed"] == 0


def test_research_gates_exact_latest_report_without_older_resurrection(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_research(cfg)
    latest_report = Path(str(_RESEARCH_OUTPUTS["report_path"]))
    older_report = Path("data/outputs/research/AAA_older_report.md")
    latest_report.parent.mkdir(parents=True, exist_ok=True)
    latest_report.write_text("# Latest\n", encoding="utf-8")
    older_report.write_text("# Older\n", encoding="utf-8")

    import sqlite3

    older_outputs = {"status": "STALE", "report_path": str(older_report)}
    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        UPDATE valuations
        SET outputs_json = ?
        WHERE ticker = 'AAA' AND as_of_date = '2026-06-01'
          AND method = 'deep_research'
        """,
        (json.dumps(older_outputs),),
    )
    conn.commit()
    conn.close()

    checked_paths: list[Path] = []

    def only_older_is_eligible(path):
        checked_paths.append(Path(path))
        return PASS if Path(path) == older_report else INVALID

    monkeypatch.setattr(
        company_depth,
        "artifact_decision_eligibility",
        only_older_is_eligible,
    )
    response = client.get("/api/company/AAA/research")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["reason"] == "FINANCIAL_INTEGRITY_REPORT_BLOCKED"
    assert checked_paths == [latest_report]

    monkeypatch.setattr(
        company_depth,
        "artifact_decision_eligibility",
        lambda path: PASS if Path(path) == latest_report else INVALID,
    )
    response = client.get("/api/company/AAA/research")
    assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.json()["as_of_date"] == "2026-07-15"
    assert response.json()["report_path"] == str(latest_report)
    assert response.json()["report_available"] is True


def test_dossier_renders_newest_with_claims(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    import sqlite3

    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json,
                                warnings_json, created_at)
        VALUES ('AAA', '2026-07-01', 'dcf', '{}', '{}', '[]', '2026-07-01T00:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()

    import os

    outputs = Path(cfg.outputs_dir)
    old_dir = outputs / "dossiers" / "run_old" / "AAA"
    new_dir = outputs / "dossiers" / "run_new" / "AAA"
    for directory in (old_dir, new_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (old_dir / "dossier.md").write_text("# AAA Dossier (old)\n", encoding="utf-8")
    (new_dir / "dossier.md").write_text("# AAA Dossier (new)\n\n## Risks\ntext\n", encoding="utf-8")
    (new_dir / "dossier.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "2025_revenue",
                        "label": "revenue::2025",
                        "value": 123.0,
                        "unit": "USD_millions",
                        "citations": [
                            {
                                "source_url": "https://example.test/facts",
                                "snippet": "revenue was 123",
                                "section_label": "financial_statements",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    os.utime(old_dir / "dossier.md", (1_000_000_000, 1_000_000_000))

    response = client.get("/api/company/AAA/dossier")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["run_label"] == "run_new"
    assert body["others"] == 1
    assert body["title"] == "AAA Dossier (new)"
    assert "<h2" in body["html"]
    assert body["claims"] == [
        {
            "claim_id": "2025_revenue",
            "label": "revenue::2025",
            "value": 123.0,
            "unit": "USD_millions",
            "citations": [
                {
                    "source_url": "https://example.test/facts",
                    "snippet": "revenue was 123",
                    "section_label": "financial_statements",
                }
            ],
        }
    ]

    # A ticker with no dossier is an honest absence, not an error.
    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json,
                                warnings_json, created_at)
        VALUES ('BBB', '2026-07-01', 'dcf', '{}', '{}', '[]', '2026-07-01T00:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()
    body = client.get("/api/company/BBB/dossier").json()
    assert body["available"] is False
    assert body["reason"] == "NO_DOSSIER"


def test_decisions_lists_dispositions_and_outcomes(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    import sqlite3

    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json,
                                warnings_json, created_at)
        VALUES ('AAA', '2026-07-01', 'dcf', '{}', '{}', '[]', '2026-07-01T00:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO dispositions (ticker, kind, status, opened_at, opened_by,
                                  trigger_snapshot_json, pre_mortem)
        VALUES ('AAA', 'AT_TARGET', 'OPEN', '2026-07-10T00:00:00+00:00', 'system',
                '{"price": 78.5}', 'We are wrong if margins mean-revert.')
        """
    )
    conn.execute(
        """
        INSERT INTO dispositions (ticker, kind, status, opened_at, opened_by,
                                  trigger_snapshot_json, decided_at, operator,
                                  reason_code, rationale, event_id)
        VALUES ('AAA', 'EVENT_DISPOSAL', 'PASSED', '2026-07-01T00:00:00+00:00', 'owner',
                '{}', '2026-07-02T00:00:00+00:00', 'owner', 'EVENT_REVIEWED',
                'Routine 8-K, no thesis impact.', 42)
        """
    )
    conn.execute(
        """
        INSERT INTO ticker_outcomes (ticker, as_of_date, run_id, decision, conviction,
                                     horizon_days, outcome_status, close_date, grade,
                                     entry_price, realized_return_pct,
                                     benchmark_return_pct, excess_return_pct,
                                     reached_buy_target, created_at, updated_at)
        VALUES ('AAA', '2026-01-10', 'run-a', 'BUY', 4, 180, 'CLOSED', '2026-07-09',
                'DEPLOY_READY', 100.0, 12.5, 4.5, 8.0, 1,
                '2026-01-10T00:00:00+00:00', '2026-07-09T00:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO ticker_outcomes (ticker, as_of_date, run_id, decision, conviction,
                                     horizon_days, outcome_status, created_at, updated_at)
        VALUES ('AAA', '2026-06-10', 'run-b', 'WATCH', 2, 180, 'OPEN',
                '2026-06-10T00:00:00+00:00', '2026-06-10T00:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()

    response = client.get("/api/company/AAA/decisions")
    assert response.status_code == 200
    body = response.json()

    assert len(body["dispositions"]) == 2
    open_row, decided_row = body["dispositions"]
    assert open_row["kind"] == "AT_TARGET"
    assert open_row["status"] == "OPEN"
    assert open_row["trigger"] == {"price": 78.5}
    assert open_row["pre_mortem"] == "We are wrong if margins mean-revert."
    assert open_row["journal_command"] == (
        f"ivi investor journal AAA --disposition-id {open_row['id']} "
        '--action acted|passed|deferred --reason <CODE> --rationale "<why>"'
    )
    assert decided_row["status"] == "PASSED"
    assert decided_row["reason_code"] == "EVENT_REVIEWED"
    assert decided_row["rationale"] == "Routine 8-K, no thesis impact."
    assert decided_row["event_id"] == 42
    assert decided_row["journal_command"] is None

    assert body["outcomes_total"] == 2
    assert body["outcomes_open"] == 1
    assert body["outcomes_closed"] == 1
    newest, oldest = body["outcomes"]
    assert newest["as_of_date"] == "2026-06-10"
    assert newest["decision"] == "WATCH"
    assert newest["outcome_status"] == "OPEN"
    assert newest["reached_buy_target"] is None
    assert oldest["as_of_date"] == "2026-01-10"
    assert oldest["realized_return_pct"] == 12.5
    assert oldest["benchmark_return_pct"] == 4.5
    assert oldest["excess_return_pct"] == 8.0
    assert oldest["reached_buy_target"] is True


def test_decisions_filter_source_runs_after_latest_selection(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    import sqlite3

    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json,
                                warnings_json, created_at)
        VALUES ('AAA', '2026-07-01', 'dcf', '{}', '{}', '[]',
                '2026-07-01T00:00:00+00:00')
        """
    )
    conn.executemany(
        """
        INSERT INTO watchlist(
            id, ticker, status, scan_family, source_run_id, added_at
        ) VALUES (?, 'AAA', 'WATCH', 'normal', ?, ?)
        """,
        [
            (101, "run-old-pass", "2026-07-01T00:00:00+00:00"),
            (102, "run-middle-invalid", "2026-07-02T00:00:00+00:00"),
            (103, "run-latest-pass", "2026-07-03T00:00:00+00:00"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO dispositions(
            ticker, watchlist_id, kind, status, opened_at, opened_by,
            trigger_snapshot_json
        ) VALUES ('AAA', ?, 'AT_TARGET', ?, ?, 'system', '{}')
        """,
        [
            (101, "PASSED", "2026-07-01T00:00:00+00:00"),
            (102, "PASSED", "2026-07-02T00:00:00+00:00"),
            (103, "OPEN", "2026-07-03T00:00:00+00:00"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO ticker_outcomes(
            ticker, as_of_date, run_id, decision, conviction, horizon_days,
            outcome_status, created_at, updated_at
        ) VALUES ('AAA', ?, ?, ?, 3, 180, ?, ?, ?)
        """,
        [
            (
                "2026-07-01",
                "run-old-pass",
                "PASS",
                "CLOSED",
                "2026-07-01T00:00:00+00:00",
                "2026-07-01T00:00:00+00:00",
            ),
            (
                "2026-07-02",
                "run-middle-invalid",
                "WATCH",
                "CLOSED",
                "2026-07-02T00:00:00+00:00",
                "2026-07-02T00:00:00+00:00",
            ),
            (
                "2026-07-03",
                "run-latest-pass",
                "BUY",
                "OPEN",
                "2026-07-03T00:00:00+00:00",
                "2026-07-03T00:00:00+00:00",
            ),
        ],
    )
    conn.commit()
    conn.close()

    eligible_watchlist_rows = {
        ("AAA", "run-old-pass", "2026-07-01T00:00:00+00:00"),
        ("AAA", "run-latest-pass", "2026-07-03T00:00:00+00:00"),
    }
    eligible_outcome_rows = {
        ("AAA", "run-old-pass", "2026-07-01"),
        ("AAA", "run-latest-pass", "2026-07-03"),
    }

    def watchlist_row_is_eligible(
        row,
        *,
        source_run_field="source_run_id",
        ticker=None,
        **_kwargs,
    ):
        identity = (
            str(ticker or row["ticker"]).upper(),
            str(row[source_run_field]),
            str(row["added_at"]),
        )
        return identity in eligible_watchlist_rows

    def outcome_row_is_eligible(row, *_args, **_kwargs):
        identity = (
            str(row["ticker"]).upper(),
            str(row["run_id"]),
            str(row["as_of_date"]),
        )
        return identity in eligible_outcome_rows

    monkeypatch.setattr(
        company_depth,
        "watchlist_row_is_decision_eligible",
        watchlist_row_is_eligible,
    )
    monkeypatch.setattr(
        company_depth,
        "outcome_row_is_decision_eligible",
        outcome_row_is_eligible,
    )

    response = client.get("/api/company/AAA/decisions")
    assert response.status_code == 200
    body = response.json()
    assert [row["status"] for row in body["dispositions"]] == ["OPEN", "PASSED"]
    assert [row["as_of_date"] for row in body["outcomes"]] == [
        "2026-07-03",
        "2026-07-01",
    ]
    assert body["outcomes_total"] == 2
    assert body["outcomes_open"] == 1
    assert body["outcomes_closed"] == 1

    # If the newest row loses authorization, the older PASS history must not
    # become the apparent current decision.
    eligible_watchlist_rows.remove(("AAA", "run-latest-pass", "2026-07-03T00:00:00+00:00"))
    eligible_outcome_rows.remove(("AAA", "run-latest-pass", "2026-07-03"))
    response = client.get("/api/company/AAA/decisions")
    assert response.status_code == 200
    assert response.json()["dispositions"] == []
    assert response.json()["outcomes"] == []
    assert response.json()["outcomes_total"] == 0
