from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.autonomous.artifact_financial_audit import (
    audit_artifact_tree,
    write_run_financial_authorization,
)
from app.autonomous.sector_report import render_autonomous_sector_report
from app.config import get_config
from app.web.readmodel import run_detail as run_detail_model
from app.web.readmodel.run_detail import (
    load_run_detail,
    load_run_report,
)
from app.web.readmodel.runs_index import UI_SCHEMA_SQL, refresh_index

# Shapes mirror the real producers: candidate_dispositions / screen_result /
# gate_evaluations from the v2 sector contract, lane_usage from
# SectorLaneUsageLedger.summary(), provider_usage from provider_usage_meta().
V2_PAYLOAD = {
    "run_id": "all_sector_v2_20260715_utilities",
    "contract_version": "autonomous_sector_financial_run_v2",
    "pipeline_version": "v2",
    "sector": "utilities",
    "market_cap_focus": "large_and_mega",
    "scan_family": "normal",
    "as_of_date": "2026-07-15",
    "created_at": "2026-07-17T09:31:34+00:00",
    "completed_at": "2026-07-17T09:32:31+00:00",
    "status": "COMPLETED",
    "execution_status": "COMPLETED",
    "decision_status": "COMPLETE",
    "final_verdict": "SELECTED",
    "selected_ticker": "AEE",
    "confidence": "MEDIUM",
    "memo_body": "Selected AEE on regulated-return durability.",
    "admitted_tickers": ["AEE", "DUK"],
    "final_decision": {
        "confidence": "MEDIUM",
        "confidence_cap_reasons": ["Rate-case timing unresolved"],
    },
    "candidate_selection": {
        "source": "sector_scan_db",
        "requested_tickers": [],
        "excluded_tickers": ["EXCL"],
        "loaded_tickers": ["AEE", "DUK", "BADCO", "FRGN"],
        "selected_tickers": ["AEE", "DUK"],
        "warnings": ["One candidate lost to cap reclassification"],
    },
    "candidate_dispositions": [
        {
            "ticker": "AEE",
            "primary_ticker": "AEE",
            "terminal_state": "UNDERWRITTEN",
            "last_completed_stage": "UNDERWRITING",
            "scope_status": "IN_SCOPE",
            "screen_status": "PASS",
            "review_status": "REVIEWED",
            "reason_codes": [],
            "security_type": "COMMON",
            "is_adr": False,
            "is_secondary_class": False,
            "frontier_status": "ON_FRONTIER",
            "frontier_dominated_by": None,
            "underwriting_verdict": "ACTIONABLE",
            "underwriting_confidence": "MEDIUM",
            "watchlist_eligible": True,
            "screen_result": {
                "contract_id": "utilities",
                "status": "PASS",
                "reason_codes": [],
                "gate_evaluations": [
                    {
                        "rule_id": "NON_PRIMARY_LISTING",
                        "contract_id": "utilities",
                        "status": "PASS",
                        "applicable": True,
                        "observed_value": "PRIMARY_OR_AUTHORIZED_SHARE_CLASS",
                        "threshold": "PRIMARY_OR_AUTHORIZED_SHARE_CLASS",
                        "reason_code": None,
                        "evidence_ref_id": "sec-company-ticker-registry:AEE:2026-07-15",
                        "evidence_url": "https://www.sec.gov/files/company_tickers_exchange.json",
                        "notes": [],
                    }
                ],
            },
        },
        {
            "ticker": "BADCO",
            "primary_ticker": "BADCO",
            "terminal_state": "SCREENED_OUT",
            "last_completed_stage": "DETERMINISTIC_SCREEN",
            "scope_status": "IN_SCOPE",
            "screen_status": "FAIL",
            "review_status": None,
            "reason_codes": ["PENNY_FLOOR"],
            "security_type": "COMMON",
            "is_adr": False,
            "is_secondary_class": False,
            "frontier_status": None,
            "frontier_dominated_by": None,
            "underwriting_verdict": None,
            "underwriting_confidence": None,
            "watchlist_eligible": False,
            "screen_result": {
                "contract_id": "utilities",
                "status": "FAIL",
                "reason_codes": ["PENNY_FLOOR"],
                "gate_evaluations": [
                    {
                        "rule_id": "PENNY_FLOOR",
                        "contract_id": "utilities",
                        "status": "FAIL",
                        "applicable": True,
                        "observed_value": 0.42,
                        "threshold": 1.0,
                        "reason_code": "PENNY_FLOOR",
                        "evidence_ref_id": "price-snapshot:BADCO:2026-07-15",
                        "evidence_url": "https://example.com/evidence/badco",
                        "notes": ["Close below the penny floor for 30 sessions"],
                    },
                    {
                        "rule_id": "NON_PRIMARY_LISTING",
                        "contract_id": "utilities",
                        "status": "PASS",
                        "applicable": True,
                        "observed_value": "PRIMARY_OR_AUTHORIZED_SHARE_CLASS",
                        "threshold": "PRIMARY_OR_AUTHORIZED_SHARE_CLASS",
                        "reason_code": None,
                        "evidence_ref_id": "sec-company-ticker-registry:BADCO:2026-07-15",
                        "evidence_url": None,
                        "notes": [],
                    },
                ],
            },
        },
        {
            "ticker": "FRGN",
            "primary_ticker": "FRGN",
            "terminal_state": "OUT_OF_SCOPE",
            "last_completed_stage": "SCOPE",
            "scope_status": "OUT_OF_SCOPE",
            "screen_status": None,
            "review_status": None,
            "reason_codes": ["FOREIGN_PRIVATE_ISSUER"],
            "security_type": "ADR",
            "is_adr": True,
            "is_secondary_class": False,
            "screen_result": None,
            "watchlist_eligible": False,
        },
        {
            "ticker": "DUK",
            "primary_ticker": "DUK",
            "terminal_state": "NEEDS_DATA",
            "last_completed_stage": "DATA_ASSEMBLY",
            "scope_status": "IN_SCOPE",
            "screen_status": "PASS",
            "review_status": None,
            "reason_codes": ["MISSING_QUARTERLY_FACTS"],
            "security_type": "COMMON",
            "is_adr": False,
            "is_secondary_class": False,
            "screen_result": {
                "contract_id": "utilities",
                "status": "PASS",
                "reason_codes": [],
                "gate_evaluations": [],
            },
            "watchlist_eligible": False,
        },
    ],
    "lane_usage": {
        "currency": "USD",
        "cost_unit": "microdollars",
        "lanes": {
            "parent_research": {
                "tool_call_attempts": 4,
                "tool_calls_ok": 4,
                "tool_calls_failed": 0,
                "provider_call_attempts": 2,
                "provider_calls_ok": 2,
                "provider_calls_failed": 0,
                "input_tokens": 9000,
                "cached_input_tokens": 1000,
                "output_tokens": 2000,
                "cost_microdollars": 1000000,
                "cost_usd": "1.000000",
            },
            "company_underwriting": {
                "tool_call_attempts": 1,
                "tool_calls_ok": 1,
                "tool_calls_failed": 0,
                "provider_call_attempts": 1,
                "provider_calls_ok": 1,
                "provider_calls_failed": 0,
                "input_tokens": 4000,
                "cached_input_tokens": 0,
                "output_tokens": 800,
                "cost_microdollars": 234567,
                "cost_usd": "0.234567",
            },
        },
        "aggregate": {
            "tool_call_attempts": 5,
            "tool_calls_ok": 5,
            "tool_calls_failed": 0,
            "provider_call_attempts": 3,
            "provider_calls_ok": 3,
            "provider_calls_failed": 0,
            "input_tokens": 13000,
            "cached_input_tokens": 1000,
            "output_tokens": 2800,
            "cost_microdollars": 1234567,
            "cost_usd": "1.234567",
        },
        "aggregate_reconciles": True,
    },
    "lane_budget": {
        "artifact_type": "autonomous_sector_lane_budget_policy_v1",
        "lanes": {
            "parent_research": {
                "max_cost_microdollars": 2000000,
                "max_cost_usd": "2.000000",
            },
            "company_underwriting": {
                "max_cost_microdollars": 1500000,
                "max_cost_usd": "1.500000",
            },
        },
    },
    "provider_usage": [
        {
            "status": "OK",
            "lane": "parent_research",
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "schema_name": "sector_synthesis",
            "input_tokens": 9000,
            "cached_input_tokens": 1000,
            "output_tokens": 2000,
            "reserved_output_tokens": 0,
            "estimated_tokens": False,
            "cost_estimate_usd": 1.0,
        },
        {
            "status": "OK",
            "lane": "company_underwriting",
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "schema_name": "company_underwriting",
            "input_tokens": 4000,
            "cached_input_tokens": 0,
            "output_tokens": 800,
            "reserved_output_tokens": 0,
            "estimated_tokens": False,
            "cost_estimate_usd": 0.234567,
        },
        {
            "status": "FAILED",
            "lane": "company_underwriting",
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "schema_name": "company_underwriting",
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reserved_output_tokens": 0,
            "estimated_tokens": True,
            "cost_estimate_usd": 0.0001,
        },
    ],
}

V1_PAYLOAD = {
    "run_id": "autonomous_sector_biotech_20260101_abc123",
    "contract_version": "autonomous_sector_financial_run_v1",
    "sector": "biotech",
    "market_cap_focus": "mid_cap",
    "scan_family": "normal",
    "as_of_date": "2026-01-01",
    "created_at": "2026-01-01T10:00:00Z",
    "completed_at": "2026-01-01T11:00:00Z",
    "status": "COMPLETED",
    "final_verdict": "NO_SELECTION",
    "selected_ticker": None,
    "no_selection_reason": "No finalist cleared underwriting.",
    "confidence": None,
    "memo_body": "",
    "candidate_selection": {
        "source": "sector_scan_db",
        "requested_tickers": [],
        "excluded_tickers": ["ZZZU"],
        "loaded_tickers": ["AAA", "BBB", "CCC"],
        "selected_tickers": ["AAA", "BBB", "CCC"],
        "warnings": [],
    },
    "company_packets": [
        {
            "ticker": "AAA",
            "blockers": [],
            "data_quality_status": "OK",
            "financial_status": "OK",
            "model_fit_status": "OK",
            "market_cap_mm": 1500.0,
            "market_cap_category": "mid_cap",
            "valuation": {
                "anchor_method": "DCF",
                "current_price": 42.0,
                "discount_to_anchor": -0.18,
            },
        },
        {
            "ticker": "BBB",
            "blockers": ["MISSING_VALUATION"],
            "data_quality_status": "DEGRADED",
            "financial_status": "OK",
            "model_fit_status": "POOR_FIT",
            "market_cap_mm": 900.0,
            "market_cap_category": "small_cap",
            "valuation": {},
        },
    ],
    "relative_ranking": [
        {
            "rank": 1,
            "ticker": "AAA",
            "audit_status": "WATCHLIST_ONLY",
            "actionable": False,
            "buy_candidate": True,
            "buy_candidate_reason": None,
            "hard_blockers": [],
            "cross_sectional_rank": 1,
            "cross_sectional_percentile": 0.91,
            "best_base_annualized_return": 0.064,
            "positioning_summary": "Cheapest durable franchise in the cohort.",
        },
        {
            "rank": 2,
            "ticker": "BBB",
            "audit_status": "AVOID",
            "actionable": False,
            "buy_candidate": False,
            "buy_candidate_reason": "Valuation missing",
            "hard_blockers": ["MISSING_VALUATION"],
            "cross_sectional_rank": 2,
            "cross_sectional_percentile": 0.42,
            "best_base_annualized_return": -0.021,
            "positioning_summary": None,
        },
    ],
}


def _ui_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(UI_SCHEMA_SQL)
    return conn


def _write_run(runs_dir: Path, subdir: str, name: str, payload: dict, report: str | None) -> None:
    run_dir = runs_dir / subdir / name
    run_dir.mkdir(parents=True)
    (run_dir / "autonomous_sector_run.json").write_text(json.dumps(payload), encoding="utf-8")
    if report is not None:
        (run_dir / "autonomous_sector_report.md").write_text(report, encoding="utf-8")


def _indexed_tree(tmp_path: Path) -> tuple[sqlite3.Connection, Path]:
    runs_dir = tmp_path / "runs"
    _write_run(
        runs_dir,
        "all_sector_v2_replay",
        "utilities",
        V2_PAYLOAD,
        "# Utilities run\n\n| a | b |\n| - | - |\n| 1 | 2 |\n",
    )
    _write_run(
        runs_dir,
        "autonomous_sector",
        "autonomous_sector_biotech_20260101_abc123",
        V1_PAYLOAD,
        None,
    )
    broken = runs_dir / "autonomous_sector" / "autonomous_sector_broken_20260102_zzz999"
    broken.mkdir(parents=True)
    (broken / "autonomous_sector_run.json").write_text("{not json", encoding="utf-8")
    conn = _ui_conn()
    refresh_index(conn, runs_dir=runs_dir)
    return conn, runs_dir


def test_v2_detail_builds_funnel_in_pipeline_order(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "all_sector_v2_20260715_utilities")
    assert detail is not None
    assert detail["quarantined"] is False
    assert [(s["key"], s["count"]) for s in detail["funnel"]] == [
        ("OUT_OF_SCOPE", 1),
        ("SCREENED_OUT", 1),
        ("NEEDS_DATA", 1),
        ("UNDERWRITTEN", 1),
    ]
    screened = next(s for s in detail["funnel"] if s["key"] == "SCREENED_OUT")
    assert screened["label"] == "screened out"
    assert screened["tickers"] == ["BADCO"]


def test_v2_detail_gate_xray_for_screened_out_candidate(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "all_sector_v2_20260715_utilities")
    badco = next(c for c in detail["candidates"] if c["ticker"] == "BADCO")
    assert badco["terminal_state"] == "SCREENED_OUT"
    assert badco["reason_codes"] == ["PENNY_FLOOR"]
    assert badco["failed_gates"] == 1
    failing = next(g for g in badco["gates"] if g["status"] == "FAIL")
    assert failing["rule_id"] == "PENNY_FLOOR"
    assert failing["observed_value"] == "0.42"
    assert failing["threshold"] == "1"
    assert failing["reason_code"] == "PENNY_FLOOR"
    assert failing["evidence_url"] == "https://example.com/evidence/badco"
    assert failing["notes"] == ["Close below the penny floor for 30 sessions"]
    passing = next(g for g in badco["gates"] if g["status"] == "PASS")
    assert passing["observed_value"] == "PRIMARY_OR_AUTHORIZED_SHARE_CLASS"


def test_v2_detail_candidates_ordered_by_funnel_stage(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "all_sector_v2_20260715_utilities")
    assert [c["ticker"] for c in detail["candidates"]] == ["FRGN", "BADCO", "DUK", "AEE"]
    aee = detail["candidates"][-1]
    assert aee["underwriting_verdict"] == "ACTIONABLE"
    assert aee["underwriting_confidence"] == "MEDIUM"
    assert aee["watchlist_eligible"] is True
    assert aee["frontier_status"] == "ON_FRONTIER"


def test_v2_detail_costs_reconcile_lanes_budget_and_providers(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "all_sector_v2_20260715_utilities")
    costs = detail["costs"]
    assert costs["available"] is True
    assert costs["aggregate_cost_microdollars"] == 1234567
    assert costs["aggregate_cost_usd"] == 1.234567
    lanes = {row["lane"]: row for row in costs["lanes"]}
    assert lanes["parent_research"]["cost_microdollars"] == 1000000
    assert lanes["parent_research"]["cost_usd"] == 1.0
    assert lanes["parent_research"]["max_cost_microdollars"] == 2000000
    assert lanes["parent_research"]["max_cost_usd"] == 2.0
    assert lanes["parent_research"]["input_tokens"] == 9000
    assert lanes["company_underwriting"]["provider_call_attempts"] == 1
    assert costs["providers"] == [
        {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "calls": 3,
            "ok_calls": 2,
            "failed_calls": 1,
            "input_tokens": 13100,
            "cached_input_tokens": 1000,
            "output_tokens": 2800,
            "cost_estimate_usd": 1.234667,
        }
    ]


def test_v2_detail_decision_and_selection_blocks(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "all_sector_v2_20260715_utilities")
    assert detail["decision"] == {
        "status": "COMPLETED",
        "execution_status": "COMPLETED",
        "decision_status": "COMPLETE",
        "final_verdict": "SELECTED",
        "selected_ticker": "AEE",
        "no_selection_reason": None,
        "confidence": "MEDIUM",
        "confidence_cap_reasons": ["Rate-case timing unresolved"],
        "memo_present": True,
    }
    assert detail["selection"]["excluded_tickers"] == ["EXCL"]
    assert detail["selection"]["admitted_tickers"] == ["AEE", "DUK"]
    assert detail["selection"]["warnings"] == ["One candidate lost to cap reclassification"]


def test_v1_detail_builds_selection_funnel_without_dispositions(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "autonomous_sector_biotech_20260101_abc123")
    assert detail is not None
    assert detail["candidates"] == []
    assert [(s["key"], s["count"]) for s in detail["funnel"]] == [
        ("LOADED", 3),
        ("EXAMINED", 2),
        ("RANKED", 2),
        ("SELECTED", 0),
    ]
    assert detail["funnel"][0]["tickers"] == ["AAA", "BBB", "CCC"]
    assert detail["decision"]["final_verdict"] == "NO_SELECTION"
    assert detail["decision"]["no_selection_reason"] == "No finalist cleared underwriting."
    assert detail["decision"]["memo_present"] is False


def test_v1_detail_ranking_and_packets(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "autonomous_sector_biotech_20260101_abc123")
    assert [r["ticker"] for r in detail["ranking"]] == ["AAA", "BBB"]
    top = detail["ranking"][0]
    assert top["audit_status"] == "WATCHLIST_ONLY"
    assert top["buy_candidate"] is True
    assert top["best_base_annualized_return"] == 0.064
    bbb = detail["ranking"][1]
    assert bbb["hard_blockers"] == ["MISSING_VALUATION"]
    packets = {p["ticker"]: p for p in detail["packets"]}
    assert packets["AAA"]["anchor_method"] == "DCF"
    assert packets["AAA"]["discount_to_anchor"] == -0.18
    assert packets["BBB"]["blockers"] == ["MISSING_VALUATION"]
    assert packets["BBB"]["anchor_method"] is None


def test_v1_costs_unavailable(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "autonomous_sector_biotech_20260101_abc123")
    assert detail["costs"] == {
        "available": False,
        "aggregate_cost_microdollars": None,
        "aggregate_cost_usd": None,
        "lanes": [],
        "providers": [],
    }


def test_quarantined_run_returns_error_not_crash(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    detail = load_run_detail(conn, "autonomous_sector_broken_20260102_zzz999")
    assert detail is not None
    assert detail["quarantined"] is True
    assert detail["parse_error"].startswith("JSONDecodeError:")
    assert detail["funnel"] == []
    assert detail["candidates"] == []


def test_unknown_run_id_returns_none(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    assert load_run_detail(conn, "no_such_run") is None
    assert load_run_report(conn, "no_such_run") is None


def test_watchlist_links_join_source_run_id(monkeypatch, tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    monkeypatch.setattr(
        run_detail_model,
        "watchlist_row_is_decision_eligible",
        lambda row: row["status"] == "ACTIVE",
    )
    engine = sqlite3.connect(":memory:")
    engine.row_factory = sqlite3.Row
    engine.execute(
        """
        CREATE TABLE watchlist (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            status TEXT,
            conviction_grade TEXT,
            source_run_id TEXT,
            added_at TEXT
        )
        """
    )
    engine.executemany(
        """
        INSERT INTO watchlist(
            ticker, status, conviction_grade, source_run_id, added_at
        ) VALUES(?, ?, ?, ?, ?)
        """,
        [
            (
                "AEE",
                "ACTIVE",
                "ACTIONABLE",
                "all_sector_v2_20260715_utilities",
                "2026-07-17T10:00:00+00:00",
            ),
            (
                "OTHER",
                "ACTIVE",
                "AVOID",
                "some_other_run",
                "2026-07-17T10:00:00+00:00",
            ),
        ],
    )
    detail = load_run_detail(conn, "all_sector_v2_20260715_utilities", engine_conn=engine)
    assert detail["watchlist_rows"] == [
        {
            "watchlist_id": 1,
            "ticker": "AEE",
            "status": "ACTIVE",
            "conviction_grade": "ACTIONABLE",
        }
    ]


def test_watchlist_links_reject_unauthorized_newest_row_without_fallback(
    monkeypatch,
    tmp_path,
):
    conn, _ = _indexed_tree(tmp_path)
    monkeypatch.setattr(
        run_detail_model,
        "watchlist_row_is_decision_eligible",
        lambda row: row["status"] == "ACTIVE" and row["conviction_grade"] == "ACTIONABLE",
    )
    engine = sqlite3.connect(":memory:")
    engine.row_factory = sqlite3.Row
    engine.execute(
        """
        CREATE TABLE watchlist (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            status TEXT,
            conviction_grade TEXT,
            source_run_id TEXT,
            added_at TEXT
        )
        """
    )
    engine.executemany(
        """
        INSERT INTO watchlist(
            ticker, status, conviction_grade, source_run_id, added_at
        ) VALUES(?, ?, ?, ?, ?)
        """,
        [
            (
                "AEE",
                "ACTIVE",
                "ACTIONABLE",
                "all_sector_v2_20260715_utilities",
                "2026-07-17T10:00:00+00:00",
            ),
            (
                "AEE",
                "DEPLOY_READY",
                "BUY",
                "all_sector_v2_20260715_utilities",
                "2026-07-18T10:00:00+00:00",
            ),
        ],
    )

    detail = load_run_detail(
        conn,
        "all_sector_v2_20260715_utilities",
        engine_conn=engine,
    )
    assert detail is not None
    assert detail["watchlist_rows"] == []


def test_report_renders_markdown_tables(tmp_path):
    conn, _ = _indexed_tree(tmp_path)
    report = load_run_report(conn, "all_sector_v2_20260715_utilities")
    assert report is not None
    assert report["run_id"] == "all_sector_v2_20260715_utilities"
    assert "<table>" in report["html"]
    assert "<h1>Utilities run</h1>" in report["html"]
    assert report["bytes"] == 47
    # v1 fixture has no report on disk
    assert load_run_report(conn, "autonomous_sector_biotech_20260101_abc123") is None


def test_detail_parses_authorized_run_bytes_not_swapped_path(monkeypatch, tmp_path):
    conn, runs_dir = _indexed_tree(tmp_path)
    run_path = runs_dir / "all_sector_v2_replay" / "utilities" / "autonomous_sector_run.json"
    authorized_bytes = run_path.read_bytes()
    forged_payload = {
        **V2_PAYLOAD,
        "selected_ticker": "FORGED",
        "final_decision": {
            **V2_PAYLOAD["final_decision"],
            "selected_ticker": "FORGED",
        },
    }

    def authorize_then_swap(path):
        assert Path(path) == run_path
        run_path.write_text(json.dumps(forged_payload), encoding="utf-8")
        return "PASS", authorized_bytes

    monkeypatch.setattr(
        run_detail_model,
        "authorized_artifact_bytes",
        authorize_then_swap,
    )
    detail = load_run_detail(conn, "all_sector_v2_20260715_utilities")
    assert detail is not None
    assert detail["decision"]["selected_ticker"] == "AEE"
    assert detail["decision"]["selected_ticker"] != "FORGED"


def test_report_renders_authorized_markdown_bytes_not_swapped_path(monkeypatch, tmp_path):
    conn, runs_dir = _indexed_tree(tmp_path)
    run_path = runs_dir / "all_sector_v2_replay" / "utilities" / "autonomous_sector_run.json"
    report_path = run_path.parent / "autonomous_sector_report.md"
    run_bytes = run_path.read_bytes()
    report_bytes = report_path.read_bytes()

    def authorize_then_swap(path):
        candidate = Path(path)
        if candidate == run_path:
            return "PASS", run_bytes
        assert candidate == report_path
        report_path.write_text("# FORGED REPORT\n", encoding="utf-8")
        return "PASS", report_bytes

    monkeypatch.setattr(
        run_detail_model,
        "authorized_artifact_bytes",
        authorize_then_swap,
    )
    report = load_run_report(conn, "all_sector_v2_20260715_utilities")
    assert report is not None
    assert "Utilities run" in report["html"]
    assert "FORGED REPORT" not in report["html"]


def test_direct_loaders_rederive_summary_from_current_authorized_bytes(tmp_path):
    conn, runs_dir = _indexed_tree(tmp_path)
    run_path = runs_dir / "all_sector_v2_replay" / "utilities" / "autonomous_sector_run.json"
    changed_payload = {
        **V2_PAYLOAD,
        "selected_ticker": "DUK",
        "final_decision": {
            **V2_PAYLOAD["final_decision"],
            "selected_ticker": "DUK",
        },
    }
    run_path.write_text(json.dumps(changed_payload), encoding="utf-8")

    detail = load_run_detail(conn, "all_sector_v2_20260715_utilities")
    assert detail is not None
    assert detail["summary"]["decision_eligible"] == 1
    assert detail["summary"]["final_verdict"] == "SELECTED"
    assert detail["summary"]["selected_ticker"] == "DUK"
    assert detail["decision"]["selected_ticker"] == "DUK"
    assert detail["funnel"]

    report = load_run_report(conn, "all_sector_v2_20260715_utilities")
    assert report is not None
    assert report["decision_eligible"] is True
    assert "Utilities run" in report["html"]


@pytest.mark.financial_integrity_contract
def test_bare_run_id_selects_newest_current_bytes_then_suppresses_invalid(
    monkeypatch,
    tmp_path,
):
    from tests.test_classic_postwrite_authorization import (
        _baseline_manifest,
        _valid_classic_artifact,
    )

    _baseline_manifest(monkeypatch, tmp_path)
    cfg = get_config()
    runs_dir = cfg.runs_dir
    run_id = "autonomous_sector_energy_20260724_duplicate"
    artifact = _valid_classic_artifact(run_id)
    artifact.created_at = "2026-07-17T09:31:34+00:00"
    run_dir = runs_dir / "autonomous_sector" / run_id
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "autonomous_sector_run.json"
    report_path = run_dir / "autonomous_sector_report.md"
    artifact_path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(
        render_autonomous_sector_report(artifact),
        encoding="utf-8",
    )
    write_run_financial_authorization(
        artifact_path,
        report_path,
        published_parent=run_dir,
    )
    reports = audit_artifact_tree(
        runs_root=cfg.runs_dir / "autonomous_sector",
        analyst_outputs_root=cfg.outputs_dir / "analyst_outputs",
        scans_root=cfg.outputs_dir / "scans",
        research_outputs_root=cfg.research_dir,
        watchlist_report_roots=[cfg.outputs_dir / "digests"],
        analysis_dir=tmp_path / "current_audit",
        generated_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(reports.manifest_json),
    )
    get_config.cache_clear()
    newer_artifact = _valid_classic_artifact(run_id)
    newer_artifact.created_at = "2026-07-18T09:31:34+00:00"
    newer_dir = runs_dir / "unreviewed_replay" / run_id
    newer_dir.mkdir(parents=True)
    newer_path = newer_dir / "autonomous_sector_run.json"
    newer_path.write_text(
        json.dumps(newer_artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (newer_dir / "autonomous_sector_report.md").write_text(
        render_autonomous_sector_report(newer_artifact),
        encoding="utf-8",
    )
    conn = _ui_conn()
    refresh_index(conn, runs_dir=runs_dir)
    old_path = artifact_path
    newer_path.write_text("{invalid newest bytes", encoding="utf-8")
    os.utime(newer_path, ns=(1_577_836_800_000_000_000, 1_577_836_800_000_000_000))
    conn.execute(
        """
        UPDATE run_index
        SET created_at = ?, slug = ?
        WHERE path = ?
        """,
        (
            "2099-01-01T00:00:00+00:00",
            "forged-cache-prefers-old",
            str(old_path),
        ),
    )

    detail = load_run_detail(conn, run_id)
    assert detail is not None
    assert detail["summary"]["path"] == str(newer_path)
    assert detail["summary"]["decision_eligible"] == 0
    assert detail["summary"]["integrity_status"] == "UNAUDITED"
    assert detail["decision"]["selected_ticker"] is None

    report = load_run_report(conn, run_id)
    assert report is not None
    assert report["decision_eligible"] is False
    assert "Historical artifact" in report["html"]

    old_slug = f"autonomous_sector/{run_id}"
    exact_old = load_run_detail(conn, old_slug)
    assert exact_old is not None
    assert exact_old["summary"]["path"] == str(old_path)
    assert exact_old["summary"]["decision_eligible"] == 1

    exact_old_report = load_run_report(conn, old_slug)
    assert exact_old_report is not None
    assert exact_old_report["decision_eligible"] is True
