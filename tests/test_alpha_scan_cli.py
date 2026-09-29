from __future__ import annotations

import json
import os
from datetime import date

import pytest
from typer.testing import CliRunner

import app.alpha.publication as alpha_publication_module
from app.alpha.publication import alpha_publication_is_authorized
from app.alpha.schemas import ComparisonRound, SectorAlphaReport, TickerSignalPacket
from app.autonomous.financial_integrity import stable_quote_hash
from app.cli import app
from app.db import get_db, init_db, utc_now_iso
from tests.financial_integrity_helpers import (
    authorize_valuation_rows,
    canonicalize_financial_packet,
    materialized_no_split_proof,
)


runner = CliRunner()


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _bind_integrity(packet: TickerSignalPacket, *, as_of_date: str) -> None:
    packet.current_price_unit = "USD_per_share"
    packet.current_price_as_of_date = as_of_date
    packet.current_price_currency = "USD"
    packet.current_price_source = "test_quote"
    packet.price_basis = "UNADJUSTED"
    packet.raw_price = packet.current_price
    packet.split_adjustment_factor = 1.0
    packet.split_lineage_proof = {
        "status": "PASS",
        "period_start": as_of_date,
        "period_end": as_of_date,
        "verified_as_of": as_of_date,
        "source": "fixture_corporate_actions_ledger",
        "source_reference": f"fixture://corporate-actions/{packet.ticker}",
    }
    packet.shares_outstanding_mm = 10.0
    packet.shares_unit = "shares_millions"
    packet.shares_basis = "UNADJUSTED"
    packet.shares_as_of_date = as_of_date
    packet.shares_source = "test_filing"
    packet.issuer_quote_ratio = 1.0
    packet.market_cap_mm = float(packet.current_price or 0.0) * 10.0
    packet.market_cap_unit = "USD_millions"
    packet.market_cap_source = "derived_from_quote_and_shares"
    packet.market_cap_method = "price_times_shares"
    packet.market_cap_effective_as_of_date = as_of_date
    packet.quote_snapshot_id = stable_quote_hash(
        ticker=packet.ticker,
        price=packet.current_price,
        as_of_date=as_of_date,
        currency="USD",
        source="test_quote",
        price_basis="UNADJUSTED",
        raw_price=packet.current_price,
        split_adjustment_factor=1.0,
    )
    packet.cap_stage_price = packet.current_price
    packet.cap_stage_price_as_of_date = as_of_date
    packet.cap_stage_price_currency = "USD"
    packet.cap_stage_price_source = "test_quote"
    packet.cap_stage_quote_snapshot_id = packet.quote_snapshot_id
    canonicalize_financial_packet(packet, as_of_date=as_of_date, shares_mm=10.0)


def test_alpha_scan_summary_and_artifacts_share_the_same_winner(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    packets = {
        "AAA": TickerSignalPacket(
            ticker="AAA",
            dcf_value=140.0,
            epv_value=120.0,
            current_price=100.0,
            gate_verdict="PROCEED",
            confidence_class="HIGH",
            moat_score=4,
            moat_classification="MODERATE_MOAT",
        ),
        "BBB": TickerSignalPacket(
            ticker="BBB",
            dcf_value=160.0,
            epv_value=130.0,
            current_price=100.0,
            gate_verdict="PROCEED",
            confidence_class="HIGH",
            moat_score=5,
            moat_classification="WIDE_MOAT",
        ),
    }
    for packet in packets.values():
        _bind_integrity(packet, as_of_date=date.today().isoformat())
    fiscal_year = date.today().year - 1
    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
                VALUES(
                    'BBB', ?, 'FY', ?, ?, ?, 'USD_millions',
                    'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                    ?, ?, '10-K', '0000000001-26-000001'
                )
            """,
            [
                (
                    fiscal_year,
                    f"{fiscal_year}-12-31",
                    line_item,
                    value,
                    utc_now_iso(),
                    date.today().isoformat(),
                )
                for line_item, value in (
                    ("revenue", 200.0),
                    ("operating_income", 40.0),
                    ("net_income", 30.0),
                    ("cfo", 50.0),
                    ("capex", 10.0),
                )
            ],
        )
    report = SectorAlphaReport(
        sector="software",
        total_candidates=2,
        rounds=[
            ComparisonRound(
                round_number=1,
                candidates_entering=["AAA", "BBB"],
                candidates_eliminated=[],
                candidates_remaining=["AAA", "BBB"],
                reasoning="Consensus shortlist.",
                elimination_criteria="Consensus score",
            )
        ],
        winner="BBB",
        winner_thesis="BBB clears the post-investigation screen with the strongest setup.",
        winner_conviction="HIGH",
        runner_up="AAA",
        runner_up_thesis="AAA still looks attractive but ranks second after investigation.",
        key_risk="Execution slippage.",
        falsification_trigger="Margin erosion.",
        time_horizon="12 months",
        signal_packets={
            "AAA": {"research_report": {}},
            "BBB": {"research_report": {}},
        },
        pre_investigation_winner="AAA",
        pre_investigation_runner_up="BBB",
        selection_basis="llm_decision",
        investigation_previews=[
            {
                "ticker": "AAA",
                "pre_investigation_rank": 1,
                "consensus_score": 20.0,
                "conviction_score": 54,
                "conviction_class": "MODERATE",
                "adjusted_mos": 0.12,
                "hard_blockers": 1,
                "open_questions": 1,
                "status": "OK",
                "verdict": "WATCH",
                "report_path": "data/outputs/research/AAA.md",
            },
            {
                "ticker": "BBB",
                "pre_investigation_rank": 2,
                "consensus_score": 19.0,
                "conviction_score": 77,
                "conviction_class": "HIGH",
                "adjusted_mos": 0.22,
                "hard_blockers": 0,
                "open_questions": 0,
                "status": "OK",
                "verdict": "PROCEED",
                "report_path": "data/outputs/research/BBB.md",
            },
        ],
        prior_ranking=[
            {
                "ticker": "AAA",
                "consensus_rank": 1,
                "consensus_score": 20.0,
                "hard_block_reasons": [],
            },
            {
                "ticker": "BBB",
                "consensus_rank": 2,
                "consensus_score": 19.0,
                "hard_block_reasons": [],
            },
        ],
        investigation_plan={
            "summary": "Investigate both shortlisted names.",
            "selected_targets": [
                {
                    "ticker": "AAA",
                    "reason": "Check blocker load.",
                    "evidence_gaps": ["Need filing follow-up."],
                },
                {
                    "ticker": "BBB",
                    "reason": "Check best upside.",
                    "evidence_gaps": ["Need peer confirmation."],
                },
            ],
            "skipped_candidates": [],
        },
        candidate_investigations=[
            {
                "ticker": "AAA",
                "consensus_rank": 1,
                "consensus_score": 20.0,
                "verdict": "WATCH",
                "confidence": "MODERATE",
                "key_findings": ["AAA still has blocker load."],
                "open_questions": ["Need follow-up."],
                "key_risk": "Execution slippage.",
                "falsification_trigger": "Margin erosion.",
                "reasoning_trace": "AAA stayed second.",
                "tool_call_counts": {"fetch_filing_section": 1},
                "tool_transcript": [],
                "num_turns": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "termination_reason": "fallback_summary",
                "error": None,
                "investigation_mode": "fallback_tool_bundle",
                "hard_block_reasons": [],
                "eligible_for_selection": True,
                "requested_focus": {
                    "reason": "Check blocker load.",
                    "evidence_gaps": ["Need filing follow-up."],
                },
            },
            {
                "ticker": "BBB",
                "consensus_rank": 2,
                "consensus_score": 19.0,
                "verdict": "PROCEED",
                "confidence": "HIGH",
                "key_findings": ["BBB has the strongest setup."],
                "open_questions": [],
                "key_risk": "Execution slippage.",
                "falsification_trigger": "Margin erosion.",
                "reasoning_trace": "BBB won the final decision.",
                "tool_call_counts": {"compare_peer_metric": 1},
                "tool_transcript": [],
                "num_turns": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "termination_reason": "fallback_summary",
                "error": None,
                "investigation_mode": "fallback_tool_bundle",
                "hard_block_reasons": [],
                "eligible_for_selection": True,
                "requested_focus": {
                    "reason": "Check best upside.",
                    "evidence_gaps": ["Need peer confirmation."],
                },
            },
        ],
        llm_decision_trace={
            "decision_trace": "BBB won after the final LLM decision.",
            "decision_mode": "llm",
            "rejected_candidates": [
                {"ticker": "AAA", "reason": "Still second after investigation."}
            ],
        },
        tool_budget_summary={
            "provider_mode": "fallback_tool_bundle",
            "investigated_candidates": 2,
            "total_tool_calls": 2,
            "total_turns": 2,
            "total_cost_usd": 0.0,
        },
    )

    monkeypatch.setattr(
        "app.valuation.peer_context._load_sector_tickers", lambda sector: ["AAA", "BBB"]
    )
    monkeypatch.setattr(
        "app.alpha.signal_assembler.assemble_sector_packets",
        lambda tickers, **kwargs: packets,
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_sector_comparison",
        lambda sector, packets, deterministic=False, integrity_scope=None: report,
    )

    result = runner.invoke(app, ["alpha-scan", "--sector", "software", "--skip-quarterly"])

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output[result.output.index("{") :])
    assert summary["financial_integrity_status"] == "PASS"
    assert len(summary["financial_integrity_scope_fingerprint"]) == 64
    assert summary["selection_basis"] == "llm_decision"
    assert summary["pre_investigation_winner"] == "AAA"
    assert summary["winner"] == "BBB"
    assert summary["winner_conviction"] == "HIGH"
    assert summary["investigation_plan"]["selected_targets"][0]["ticker"] == "AAA"
    assert summary["candidate_investigations"][1]["ticker"] == "BBB"

    report_json = cfg.data_dir / "outputs" / "alpha" / "alpha_software.json"
    report_md = cfg.data_dir / "outputs" / "alpha" / "alpha_software_report.md"
    authorization = (
        cfg.data_dir / "outputs" / "alpha" / "alpha_software.financial_integrity_authorization.json"
    )

    assert report_json.exists()
    assert report_md.exists()
    assert authorization.exists()
    assert alpha_publication_is_authorized(authorization)

    saved = json.loads(report_json.read_text(encoding="utf-8"))
    saved_md = report_md.read_text(encoding="utf-8")

    binding = saved["financial_integrity_binding"]
    assert binding["status"] == "PASS"
    assert len(binding["decision_scope_fingerprint"]) == 64
    assert len(binding["publication_scope_fingerprint"]) == 64
    assert binding["decision_scope_manifest"]["context"] == "alpha_scan:software"
    assert binding["decision_scope_manifest"]["packets"] == binding["scope_manifest"]["packets"]
    assert binding["scope_manifest"]["scenarios"]
    winner_scenario = next(
        item for item in binding["scope_manifest"]["scenarios"] if item["ticker"] == "BBB"
    )
    assert len(winner_scenario["financial_inputs"]["companyfacts_rows"]) == 5
    assert winner_scenario["financial_inputs"]["rendered_financials"][0]["revenue"] == 200.0
    assert saved["pre_investigation_winner"] == "AAA"
    assert saved["winner"] == "BBB"
    assert "## Winner: BBB" in saved_md
    assert f"| {fiscal_year} | 200 | 40 | 20.0% | 30 | 50 | 40 |" in saved_md
    assert "**Consensus leader before investigation:** AAA" in saved_md
    assert "## LLM Investigation Plan" in saved_md

    original_json_bytes = report_json.read_bytes()
    tampered_json_bytes = original_json_bytes.replace(
        b'"winner": "BBB"',
        b'"winner": "BBC"',
        1,
    )
    assert tampered_json_bytes != original_json_bytes
    assert len(tampered_json_bytes) == len(original_json_bytes)
    original_snapshot_reader = alpha_publication_module._read_exact_file_snapshot
    replaced_current_json = False

    def _replace_json_after_snapshot(path):
        nonlocal replaced_current_json
        snapshot = original_snapshot_reader(path)
        if not replaced_current_json and str(path) == str(report_json):
            replacement = report_json.with_name(".alpha_software.swap.tmp")
            replacement.write_bytes(tampered_json_bytes)
            replacement.replace(report_json)
            replaced_current_json = True
        return snapshot

    monkeypatch.setattr(
        alpha_publication_module,
        "_read_exact_file_snapshot",
        _replace_json_after_snapshot,
    )
    assert not alpha_publication_is_authorized(authorization)
    assert report_json.read_bytes() == tampered_json_bytes
    monkeypatch.setattr(
        alpha_publication_module,
        "_read_exact_file_snapshot",
        original_snapshot_reader,
    )
    report_json.write_bytes(original_json_bytes)
    assert alpha_publication_is_authorized(authorization)

    original_stat = report_md.stat()
    tampered = saved_md.replace("## Winner: BBB", "## Winner: BBC", 1)
    assert len(tampered.encode("utf-8")) == len(saved_md.encode("utf-8"))
    report_md.write_text(tampered, encoding="utf-8")
    os.utime(
        report_md,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert not alpha_publication_is_authorized(authorization)


@pytest.mark.financial_integrity_contract
def test_alpha_scan_real_assembler_context_reaches_gate_with_canonical_inputs(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    cfg = _init_temp_db(monkeypatch, tmp_path)
    as_of_date = date.today().isoformat()
    ticker = "RCTX"
    cik = "0000004321"
    source_url = "https://example.test/quote/RCTX"
    scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {
            "current_price": 50.0,
            "current_price_as_of_date": as_of_date,
            "current_price_currency": "USD",
            "current_price_source": "fixture_quote",
            "current_price_source_url": source_url,
            "current_price_basis": "UNADJUSTED",
            "split_adjustment_factor": 1.0,
            "no_intervening_split_proof": materialized_no_split_proof(
                ticker=ticker,
                period_start="2025-12-31",
                period_end=as_of_date,
                issuer_cik=cik,
            ),
            "dcf_base": 75.0,
            "epv_adjusted": 65.0,
            "gate_action": "PROCEED",
        },
        "quality_context": {
            "gate_action": "PROCEED",
            "confidence_class": "MODERATE",
        },
    }
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (ticker, cik, "Real Context Fixture", utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at
            )
            VALUES(?, ?, 'scorecard', '{}', ?, '[]', ?)
            """,
            (ticker, as_of_date, json.dumps(scorecard), utc_now_iso()),
        )
        scorecard_row_id = int(
            conn.execute("SELECT last_insert_rowid() AS row_id").fetchone()["row_id"]
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                ?, 2025, 'FY', '2025-12-31', 'shares_outstanding',
                10.0, 'shares_millions', ?, ?, '2026-02-01', '10-K',
                '0000004321-26-000001'
            )
            """,
            (
                ticker,
                "https://data.sec.gov/api/xbrl/companyfacts/CIK0000004321.json",
                utc_now_iso(),
            ),
        )
    source_artifact = authorize_valuation_rows(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        run_id="alpha_real_assembler_context",
        row_ids=[scorecard_row_id],
        issuer_ciks={ticker: cik},
    )
    with get_db() as conn:
        scorecard_row = conn.execute(
            "SELECT * FROM valuations WHERE id = ?",
            (scorecard_row_id,),
        ).fetchone()
    from app.valuation.lineage import valuation_row_is_decision_eligible

    assert source_artifact.is_file()
    assert not valuation_row_is_decision_eligible({})
    assert valuation_row_is_decision_eligible(
        scorecard_row,
        expected_issuer_cik=cik,
        expected_issuer_aliases=(ticker,),
        require_exact_issuer_binding=True,
    )
    submissions_dir = cfg.cache_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    (submissions_dir / f"{cik}.json").write_text(
        json.dumps(
            {
                "tickers": [ticker],
                "exchanges": ["NYSE"],
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "filingDate": ["2026-02-01"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def unexpected_companyfacts_io(*_args, **_kwargs):
        raise AssertionError("canonical V1 attempted companyfacts HTTP or cache mutation")

    monkeypatch.setattr(
        "app.market.company_facts_provider.HttpClient",
        unexpected_companyfacts_io,
    )
    monkeypatch.setattr(
        "app.market.company_facts_provider._write_cached_companyfacts",
        unexpected_companyfacts_io,
    )
    captured: dict[str, object] = {}

    def _compare(sector, packets, *, deterministic, integrity_scope):
        packet = packets[ticker]
        captured["packet"] = packet
        captured["scope"] = integrity_scope
        assert deterministic is True
        return SectorAlphaReport(
            sector=sector,
            total_candidates=1,
            rounds=[],
            winner=ticker,
            winner_thesis="Canonical real-assembler fixture.",
            winner_conviction="MODERATE",
            runner_up=None,
            runner_up_thesis="",
            key_risk="Fixture risk.",
            falsification_trigger="Fixture trigger.",
            time_horizon="12 months",
            signal_packets={ticker: packet.to_summary_dict()},
            selection_basis="consensus",
        )

    monkeypatch.setattr(
        "app.valuation.peer_context._load_sector_tickers",
        lambda sector: [ticker],
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_sector_comparison",
        _compare,
    )

    result = runner.invoke(
        app,
        ["alpha-scan", "--sector", "software", "--skip-quarterly", "--deterministic"],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output[result.output.index("{") :])
    assert summary["financial_integrity_status"] == "PASS"
    assert len(summary["financial_integrity_scope_fingerprint"]) == 64
    packet = captured["packet"]
    expected_snapshot_id = stable_quote_hash(
        ticker=ticker,
        price=50.0,
        as_of_date=as_of_date,
        currency="USD",
        source="fixture_quote",
        source_url=source_url,
        price_basis="UNADJUSTED",
        raw_price=50.0,
        split_adjustment_factor=1.0,
    )
    assert packet.current_price == 50.0
    assert packet.dcf_value == 75.0
    assert packet.epv_value == 65.0
    assert packet.pricing_zone == "MARGIN_OF_SAFETY"
    assert packet.gate_verdict == "PROCEED"
    assert packet.raw_valuation == scorecard
    assert packet.market_cap_mm == 500.0
    assert packet.shares_outstanding_mm == 10.0
    assert packet.raw_shares_outstanding_mm == 10.0
    assert packet.price_basis == "UNADJUSTED"
    assert packet.shares_basis == "UNADJUSTED"
    assert packet.quote_snapshot_id == expected_snapshot_id
    assert captured["scope"].packets[0] is packet
    assert not (cfg.cache_dir / "companyfacts").exists()


def test_alpha_scan_viability_counts_insurance_value_as_valuation(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    packets = {
        "INSR": TickerSignalPacket(
            ticker="INSR",
            insurance_value=50.0,
            insurance_method="insurance_common",
            current_price=25.0,
            gate_verdict="PROCEED",
            confidence_class="HIGH",
        )
    }
    for packet in packets.values():
        _bind_integrity(packet, as_of_date=date.today().isoformat())
    report = SectorAlphaReport(
        sector="insurance",
        total_candidates=1,
        rounds=[],
        winner="INSR",
        winner_thesis="Insurance valuation is canonical.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Risk.",
        falsification_trigger="Trigger.",
        time_horizon="12 months",
        signal_packets={"INSR": packets["INSR"].to_summary_dict()},
        selection_basis="consensus",
    )

    monkeypatch.setattr("app.valuation.peer_context._load_sector_tickers", lambda sector: ["INSR"])
    monkeypatch.setattr(
        "app.alpha.signal_assembler.assemble_sector_packets",
        lambda tickers, **kwargs: packets,
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_sector_comparison",
        lambda sector, packets, deterministic=False, integrity_scope=None: report,
    )

    result = runner.invoke(app, ["alpha-scan", "--sector", "insurance", "--skip-quarterly"])

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output[result.output.index("{") :])
    assert summary["viable_candidates"] == 1
    assert summary["attrition"] == {}
