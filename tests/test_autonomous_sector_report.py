from __future__ import annotations

import json
import sqlite3

from app.autonomous.output_store import persist_autonomous_sector_run
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    MARKET_CAP_UNIT_USD_MILLIONS,
    PRICE_BASIS_UNADJUSTED,
    PRICE_UNIT_USD_PER_SHARE,
    SHARES_UNIT_MILLIONS,
    canonical_metric_trace,
    require_financial_integrity_scope,
    stable_quote_hash,
)
from app.autonomous.run_contract import BeliefUpdate, EvidenceReference, ToolCallRecord
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorFinalDecision,
    SectorFinancialFramework,
    SectorResearchQuestion,
)
from app.autonomous.sector_report import render_autonomous_sector_report
from app.autonomous.sector_runtime import (
    _financial_integrity_run_binding,
    enrich_sector_artifact_memo_body,
)
from app.autonomous.sweep_delta import v1_terminal_coverage_from_artifact
from tests.financial_integrity_helpers import canonicalize_financial_packet


class MemoBodyProvider:
    provider_name = "test"

    def __init__(
        self,
        payload: dict | None = None,
        exc: Exception | None = None,
        payloads_by_schema: dict[str, dict] | None = None,
        exc_by_schema: dict[str, Exception] | None = None,
    ):
        self.payload = payload or {}
        self.exc = exc
        self.payloads_by_schema = payloads_by_schema or {}
        self.exc_by_schema = exc_by_schema or {}
        self.calls: list[dict] = []

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls.append(kwargs)
        schema_name = str(kwargs.get("schema_name") or "")
        if schema_name in self.exc_by_schema:
            raise self.exc_by_schema[schema_name]
        if self.exc:
            raise self.exc
        return json.dumps(self.payloads_by_schema.get(schema_name, self.payload))


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def _input_provenance(
    inputs: dict[str, float],
    *,
    as_of_date: str,
) -> dict[str, dict[str, object]]:
    units = {
        "current_price": PRICE_UNIT_USD_PER_SHARE,
        "estimated_future_value_per_share": PRICE_UNIT_USD_PER_SHARE,
        "horizon_years": "years",
        "shares_outstanding_mm": SHARES_UNIT_MILLIONS,
        "issuer_quote_ratio": "ratio",
        "free_cash_flow_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
        "market_cap_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    }
    provenance = {
        name: {
            "value": value,
            "unit": units[name],
            "source": "fixture_financial_contract",
            "period_end": as_of_date,
            "filed_date": as_of_date,
            "source_reference": f"https://example.test/financial-contract/{name}",
        }
        for name, value in inputs.items()
    }
    if "shares_outstanding_mm" in provenance:
        provenance["shares_outstanding_mm"].update(
            {
                "raw_source_value": float(inputs["shares_outstanding_mm"]) * 1_000_000.0,
                "raw_source_unit": "shares",
                "normalized_value": float(inputs["shares_outstanding_mm"]),
                "normalized_unit": SHARES_UNIT_MILLIONS,
                "split_adjustment_factor": 1.0,
            }
        )
    return provenance


def _bind_valid_financial_context(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> AutonomousSectorFinancialRunArtifact:
    packets_by_ticker = {str(packet.ticker).upper(): packet for packet in artifact.company_packets}
    for packet in artifact.company_packets:
        ticker = str(packet.ticker).upper()
        price = float(packet.current_price or 50.0)
        shares_mm = 10.0
        market_cap_mm = price * shares_mm
        source_url = f"https://example.test/{ticker.lower()}/quote"
        snapshot_id = stable_quote_hash(
            ticker=ticker,
            price=price,
            as_of_date=artifact.as_of_date,
            currency="USD",
            source="fixture_quote",
            source_url=source_url,
            price_basis=PRICE_BASIS_UNADJUSTED,
            raw_price=price,
            split_adjustment_factor=1.0,
        )
        packet.current_price = price
        packet.current_price_unit = PRICE_UNIT_USD_PER_SHARE
        packet.current_price_as_of_date = artifact.as_of_date
        packet.current_price_currency = "USD"
        packet.current_price_source = "fixture_quote"
        packet.current_price_source_url = source_url
        packet.quote_snapshot_id = snapshot_id
        packet.price_basis = PRICE_BASIS_UNADJUSTED
        packet.raw_price = price
        packet.split_adjustment_factor = 1.0
        packet.split_lineage_proof = {
            "status": "PASS",
            "period_start": artifact.as_of_date,
            "period_end": artifact.as_of_date,
            "verified_as_of": artifact.as_of_date,
            "source": "fixture_corporate_actions",
            "source_reference": (f"https://example.test/{ticker.lower()}/corporate-actions"),
        }
        packet.shares_outstanding_mm = shares_mm
        packet.raw_shares_outstanding_mm = shares_mm
        packet.shares_unit = SHARES_UNIT_MILLIONS
        packet.shares_basis = PRICE_BASIS_UNADJUSTED
        packet.shares_as_of_date = artifact.as_of_date
        packet.shares_filed_date = artifact.as_of_date
        packet.shares_source = "fixture_companyfacts"
        packet.shares_source_url = f"https://example.test/{ticker.lower()}/companyfacts"
        packet.issuer_quote_ratio = 1.0
        packet.market_cap_mm = market_cap_mm
        packet.market_cap_unit = MARKET_CAP_UNIT_USD_MILLIONS
        packet.market_cap_source = "derived_from_quote_and_shares"
        packet.market_cap_effective_as_of_date = artifact.as_of_date
        packet.market_cap_method = "price_times_shares"
        packet.cap_stage_price = price
        packet.cap_stage_price_as_of_date = artifact.as_of_date
        packet.cap_stage_price_currency = "USD"
        packet.cap_stage_price_source = "fixture_quote"
        packet.cap_stage_price_source_url = source_url
        packet.cap_stage_quote_snapshot_id = snapshot_id
        canonicalize_financial_packet(
            packet,
            as_of_date=artifact.as_of_date,
            shares_mm=shares_mm,
        )
        packet.financial_integrity_status = "PASS"
        packet.financial_integrity_violations = []
        snapshot_id = str(packet.quote_snapshot_id)
        market_cap_mm = float(packet.market_cap_mm)
        cap_inputs = {
            "current_price": price,
            "shares_outstanding_mm": shares_mm,
            "issuer_quote_ratio": 1.0,
        }
        packet.metric_traces["market_cap_mm"] = canonical_metric_trace(
            metric="market_cap_mm",
            formula=("current_price * shares_outstanding_mm / issuer_quote_ratio"),
            inputs=cap_inputs,
            output=market_cap_mm,
            output_unit=MARKET_CAP_UNIT_USD_MILLIONS,
            recomputed_output=market_cap_mm,
            quote_snapshot_id=snapshot_id,
            input_provenance=_input_provenance(
                cap_inputs,
                as_of_date=artifact.as_of_date,
            ),
        )
        fcf_yield = (packet.valuation or {}).get("fcf_yield")
        if isinstance(fcf_yield, (int, float)):
            fcf_inputs = {
                "free_cash_flow_usd_millions": float(fcf_yield) * market_cap_mm,
                "market_cap_usd_millions": market_cap_mm,
            }
            packet.metric_traces["fcf_yield"] = canonical_metric_trace(
                metric="fcf_yield",
                formula=("free_cash_flow_usd_millions / market_cap_usd_millions"),
                inputs=fcf_inputs,
                output=float(fcf_yield),
                output_unit="ratio",
                recomputed_output=float(fcf_yield),
                quote_snapshot_id=snapshot_id,
                input_provenance=_input_provenance(
                    fcf_inputs,
                    as_of_date=artifact.as_of_date,
                ),
            )

    for scenario in artifact.expected_return_scenarios:
        packet = packets_by_ticker.get(str(scenario.ticker).upper())
        if packet is None or scenario.annualized_return is None:
            continue
        scenario.current_price = packet.current_price
        scenario.current_price_unit = PRICE_UNIT_USD_PER_SHARE
        scenario.quote_snapshot_id = packet.quote_snapshot_id
        scenario.price_basis = packet.price_basis
        scenario_inputs = {
            "estimated_future_value_per_share": float(scenario.estimated_future_value_per_share),
            "current_price": float(scenario.current_price),
            "horizon_years": float(scenario.horizon_years),
        }
        scenario.metric_trace = canonical_metric_trace(
            metric="annualized_return",
            formula=(
                "round((estimated_future_value_per_share / current_price) "
                "** (1 / horizon_years) - 1, 6)"
            ),
            inputs=scenario_inputs,
            output=float(scenario.annualized_return),
            output_unit="annualized_ratio",
            recomputed_output=float(scenario.annualized_return),
            quote_snapshot_id=packet.quote_snapshot_id,
            input_provenance=_input_provenance(
                scenario_inputs,
                as_of_date=artifact.as_of_date,
            ),
        )
        scenario.financial_integrity_status = "PASS"
        scenario.financial_integrity_violations = []
    scope = FinancialIntegrityScope(
        context="autonomous_sector_pre_provider",
        run_as_of_date=artifact.as_of_date,
        packets=tuple(artifact.company_packets),
        scenarios=tuple(artifact.expected_return_scenarios),
    )
    gate = require_financial_integrity_scope(scope)
    artifact.candidate_selection["financial_integrity"] = gate.to_dict()
    artifact.candidate_selection["financial_integrity_binding"] = _financial_integrity_run_binding(
        scope,
        scope_fingerprint=gate.scope_fingerprint,
    )
    return artifact


def _artifact() -> AutonomousSectorFinancialRunArtifact:
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_test_20260426_report",
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        objective="Pick the strongest financially underwritten candidate.",
        as_of_date="2026-04-26",
        created_at="2026-04-26T20:00:00Z",
        completed_at="2026-04-26T20:03:00Z",
        status="COMPLETED",
        final_verdict="SELECTED",
        selected_ticker="AAA",
        confidence="MODERATE",
        candidate_selection={
            "source": "sector_scan_db",
            "ranking_basis": "consensus_pre_rank",
            "selected_tickers": ["AAA", "BBB"],
            "loaded_tickers": ["AAA", "BBB", "CCC"],
            "excluded_tickers": ["CCC"],
            "warnings": ["PRE_RANK_USED"],
        },
        framework=SectorFinancialFramework(
            sector="specialty_manufacturing",
            market_cap_focus="small_cap",
            horizon_years=[5, 10],
            economic_model="Small-cap financial underwriting.",
            selected_value_drivers=["per_share_growth", "cash_conversion"],
            selected_metrics=["base_case_annualized_return"],
            valid_valuation_methods=["dcf", "epv"],
            required_evidence=["scenario_math", "kpi_trends"],
        ),
        company_packets=[
            SectorCompanyFinancialPacket(
                ticker="AAA",
                financial_status="Financially Viable",
                model_fit_status="VALID_GENERIC",
                data_quality_status="OK",
                current_price=50.0,
                business_quality={
                    "piotroski_f_score": 7,
                    "piotroski_interpretation": "moderate",
                    "piotroski_missing_inputs": [],
                    "piotroski_status": "OK",
                    "beneish_m_score": -2.1,
                    "beneish_interpretation": "no manipulation flag",
                    "beneish_missing_inputs": [],
                    "beneish_status": "OK",
                    "altman_z_score": 3.4,
                    "altman_interpretation": "safe",
                    "altman_missing_inputs": [],
                    "altman_status": "OK",
                },
                valuation={
                    "anchor_method": "dcf",
                    "valuation_anchor": 100.0,
                    "discount_to_anchor": 0.5,
                    "fcf_yield": 0.05,
                    "peer_relative_valuation": {
                        "status": "OK",
                        "peer_count": 6,
                        "metrics": {
                            "ev_ebitda": {
                                "stock_value": 4.0,
                                "sector_median": 3.5,
                                "sector_q1": 2.0,
                                "sector_q3": 5.0,
                                "percentile_rank": 66.7,
                            },
                            "fcf_yield": {
                                "stock_value": 0.05,
                                "sector_median": 0.035,
                                "sector_q1": 0.02,
                                "sector_q3": 0.06,
                                "percentile_rank": 66.7,
                            },
                        },
                    },
                },
                returns_on_capital={
                    "roic": 0.11538461538461539,
                    "roic_wacc_spread": 0.015384615384615385,
                    "incremental_roic_3y": 0.03666666666666665,
                    "roic_trajectory_5y": [
                        {"fiscal_year": 2022, "roic": 0.18285714285714286},
                        {"fiscal_year": 2023, "roic": 0.158},
                        {"fiscal_year": 2024, "roic": 0.13454545454545455},
                        {"fiscal_year": 2025, "roic": 0.11538461538461539},
                    ],
                    "roic_not_computable_reasons": [],
                },
                blockers=[],
                confidence_caps=["CYCLE_DURABILITY_UNRESOLVED"],
            ),
            SectorCompanyFinancialPacket(
                ticker="BBB",
                financial_status="Data Insufficient",
                model_fit_status="UNKNOWN",
                data_quality_status="MISSING_VALUATION",
                current_price=70.0,
                valuation={
                    "anchor_method": None,
                    "valuation_anchor": None,
                    "discount_to_anchor": None,
                    "fcf_yield": None,
                },
                returns_on_capital={
                    "roic": None,
                    "roic_wacc_spread": None,
                    "incremental_roic_3y": None,
                    "roic_trajectory_5y": [],
                    "roic_not_computable_reasons": ["ROIC_NOT_COMPUTABLE"],
                },
                blockers=["MISSING_VALUATION"],
                confidence_caps=[],
            ),
        ],
        expected_return_scenarios=[
            SectorExpectedReturnScenario(
                scenario_id="AAA_downside_5Y",
                ticker="AAA",
                scenario_name="downside",
                horizon_years=5,
                current_price=50.0,
                estimated_future_value_per_share=58.754852,
                annualized_return=0.032796,
            ),
            SectorExpectedReturnScenario(
                scenario_id="AAA_base_5Y",
                ticker="AAA",
                scenario_name="base",
                horizon_years=5,
                current_price=50.0,
                estimated_future_value_per_share=146.932808,
                annualized_return=0.240594,
            ),
            SectorExpectedReturnScenario(
                scenario_id="AAA_upside_5Y",
                ticker="AAA",
                scenario_name="upside",
                horizon_years=5,
                current_price=50.0,
                estimated_future_value_per_share=176.319369,
                annualized_return=0.286667,
            ),
        ],
        research_questions=[
            SectorResearchQuestion(
                question_id="Q1",
                question="Which candidate clears the expected-return hurdle?",
                financial_pillar="Expected return",
                expected_decision_impact="Determines whether selection is possible.",
                priority="HIGH",
                status="ANSWERED",
                target_tickers=["AAA", "BBB"],
                planned_tools=["rank_expected_return_cases", "fetch_kpi_trends"],
                evidence_ref_ids=["E1", "E2"],
            )
        ],
        tool_calls=[
            ToolCallRecord(
                call_id="TC1",
                tool_name="rank_expected_return_cases",
                tool_input={"horizon_years": 5},
                rationale="Rank base cases.",
                status="OK",
                question_id="Q1",
                evidence_ref_ids=["E1"],
            )
        ],
        evidence=[
            EvidenceReference(
                evidence_id="E1",
                source_type="tool_output",
                source_label="rank_expected_return_cases",
                summary="AAA ranked first on base-case expected return.",
                ticker="AAA",
                confidence="MODERATE",
            )
        ],
        belief_updates=[
            BeliefUpdate(
                update_id="BU1",
                question_id="Q1",
                ticker="AAA",
                prior_belief="AAA needed scenario support.",
                updated_belief="AAA clears the hurdle.",
                direction="BULLISH",
                confidence_after="MODERATE",
                summary="Expected-return evidence supports AAA.",
                evidence_ref_ids=["E1"],
                remaining_uncertainty=["Cycle durability."],
            )
        ],
        final_decision=SectorFinalDecision(
            verdict="SELECTED",
            confidence="MODERATE",
            selected_ticker="AAA",
            expected_annualized_return_range="20-25%",
            thesis="AAA has the strongest underwritten return case.",
            key_risk="Margins may normalize lower.",
            downside_case="Downside remains above the current price only if the anchor holds.",
            falsifiers=["Base-case return falls below hurdle."],
            why_selected_over_finalists=["AAA beats BBB on expected return."],
            confidence_cap_reasons=["CYCLE_DURABILITY_UNRESOLVED"],
            evidence_ref_ids=["E1"],
        ),
        selection_audit={
            "status": "PASS",
            "selected_ticker": "AAA",
            "actionable": True,
            "confidence_ceiling": "HIGH",
            "hard_blockers": [],
            "confidence_caps": [],
            "expected_return_evidence_count": 1,
            "company_specific_evidence_count": 2,
            "base_return_hurdle": 0.12,
            "selected_return_cushion_hurdle": 0.15,
            "best_base_annualized_return": 0.240594,
            "base_return_margin_over_hurdle": 0.120594,
            "best_base_horizon_years": 5,
            "downside_annualized_return": 0.032796,
            "downside_horizon_years": 5,
            "downside_evidence_status": "PRESENT",
            "capital_loss_underwriting_status": "TEMPORARY_WEAKNESS_SUPPORTED",
            "capital_loss_impairment_class": "TEMPORARY_WEAKNESS",
            "capital_loss_underwriting_caution": "POSSIBLE_CYCLE_DISTORTION",
            "capital_loss_reason_codes": ["CYCLICAL_TROUGH", "CYCLICAL_SUPPORT"],
            "selected_company_evidence_tools": ["analyze_liquidity_stress", "fetch_kpi_trends"],
            "selected_evidence_pillars": ["liquidity", "quality"],
            "selected_risk_evidence_tools": ["analyze_liquidity_stress"],
            "framework_required_evidence": ["scenario_math", "kpi_trends"],
            "framework_required_evidence_covered": ["scenario_math", "kpi_trends"],
            "framework_required_evidence_missing": [],
            "framework_required_evidence_coverage_ratio": 1.0,
            "capital_structure_resolution_status": "CAPITAL_STRUCTURE_RESOLVED_CLEAR",
            "capital_structure_resolution_summary": "AAA refinancing and covenant status are clear.",
            "capital_structure_terms_status": "STRUCTURED_TERMS_EXTRACTED",
            "maturity_schedule_status": "MATURITY_SCHEDULE_EXTRACTED",
            "maturity_schedule": [
                {"year": 2027, "amount": 25.0, "unit": "million"},
                {"year": 2028, "maturity_date": "2028-09-30", "amount": 40.0, "unit": "million"},
                {"year": None, "period": "thereafter", "amount": 45.0, "unit": "million"},
            ],
            "covenant_status": "COMPLIANCE_EVIDENCED",
            "covenant_terms_status": "COVENANT_TERMS_EXTRACTED",
            "covenant_terms": [
                {
                    "metric": "consolidated net leverage ratio",
                    "condition": "MAXIMUM",
                    "threshold": 3.5,
                    "unit": "ratio",
                },
                {
                    "metric": "liquidity",
                    "condition": "MINIMUM",
                    "threshold": 20.0,
                    "unit": "million",
                },
            ],
            "refinancing_timeline_status": "RESOLVED_CURRENT_EVIDENCE",
            "refinancing_timeline_needs": [],
            "refinancing_timeline_summary": "AAA refinancing and covenant status are clear.",
            "debt_due_within_12mo": False,
            "going_concern_language": False,
            "no_assurance_financing": False,
            "cash_runway_quarters": 12.0,
            "return_cushion_status": "CLEAR",
            "notes": ["Selection audit passed; candidate remains actionable."],
            "final_verdict_after_audit": "SELECTED",
        },
        company_autonomy_attempted=True,
        company_autonomy_status="COMPLETED",
        company_autonomy_notes=["Nested company autonomy completed for AAA."],
        company_autonomy_runs=[
            {
                "ticker": "AAA",
                "status": "COMPLETED",
                "final_verdict": "ACTIONABLE",
                "confidence": "MODERATE",
                "tool_calls": 2,
                "evidence_references": 2,
                "degraded_states": [],
            }
        ],
        relative_ranking=[
            {
                "rank": 1,
                "ticker": "AAA",
                "best_base_annualized_return": 0.240594,
                "downside_annualized_return": 0.032796,
                "capital_loss_underwriting_status": "TEMPORARY_WEAKNESS_SUPPORTED",
                "refinancing_timeline_status": "RESOLVED_CURRENT_EVIDENCE",
                "audit_status": "PASS",
                "actionable": True,
                "company_autonomy_verdict": "ACTIONABLE",
                "confidence_caps": [],
                "hard_blockers": [],
                "positioning_summary": "Best-positioned and actionable under current guardrails.",
            },
            {
                "rank": 2,
                "ticker": "BBB",
                "best_base_annualized_return": 0.08,
                "downside_annualized_return": -0.12,
                "capital_loss_underwriting_status": "PERMANENT_LOSS_BLOCKED",
                "refinancing_timeline_status": "NEAR_TERM_MATURITY_UNRESOLVED",
                "audit_status": "BLOCKED",
                "actionable": False,
                "company_autonomy_verdict": None,
                "confidence_caps": [],
                "hard_blockers": ["MISSING_VALUATION"],
                "positioning_summary": "Not actionable under current guardrails: MISSING_VALUATION.",
            },
        ],
        audit_notes=["Final decision made after deterministic scenario review."],
    )
    return _bind_valid_financial_context(artifact)


def _init_temp_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    return get_config()


def test_render_autonomous_sector_report_includes_decision_and_audit_sections():
    report = render_autonomous_sector_report(_artifact())

    assert "# specialty_manufacturing Sector Research Memo" in report
    assert "## Executive Conclusion" in report
    assert "The sector run selects **AAA**" in report
    assert "## Candidate Coverage" in report
    assert "separate pipeline stages, not interchangeable 'reviewed' counts" in report
    assert "candidate source was sector_scan_db" in report
    assert "## Methodology" in report
    assert "Small-cap financial underwriting." in report
    assert "## Top Candidates" in report
    assert (
        "| Ticker | Company | Market Cap | Conviction Grade | ROIC | ROIC−WACC Spread | FCF Yield | Valuation Anchor | Buy-Price Target | Current Price | Distance From Buy |"
        in report
    )
    # The memo renders the grade-scaled buy target the DB
    # persists (ACTIONABLE/MODERATE, no intrinsic range -> base 0.12 + neutral
    # dispersion 0.20*0.25 = 0.17 discount -> 100*0.83 = $83.00), NOT the legacy
    # neutral anchor*0.75 = $75.00. Distance recomputes to (50-83)/83 = -39.8%.
    assert (
        "| AAA | AAA | $500.0M | MODERATE | 11.5% | 1.5% | 5.0% | dcf $100.00 | $83.00 | $50.00 | -39.8% |"
        in report
    )
    assert "## Per-Candidate Sections" in report
    assert "### AAA — AAA" in report
    assert "**5-Year Financial Table**" in report
    assert "**Business Quality**" in report
    assert (
        "- Quality scores: Piotroski F=7/9 (moderate); Beneish M=-2.10 (no manipulation flag); Altman Z''=3.40 (safe)."
        in report
    )
    assert (
        "- Returns on capital: latest ROIC 11.5%; ROIC−WACC spread 1.5%; incremental 3Y ROIC 3.7%."
        in report
    )
    assert "- 5-year ROIC trajectory: 2022: 18.3%; 2023: 15.8%; 2024: 13.5%; 2025: 11.5%." in report
    assert "**Valuation And Expected-Return Cases**" in report
    assert "Peer-relative valuation: EV/EBITDA 4.0x versus sector median 3.5x" in report
    assert "FCF yield 5.0% versus sector median 3.5%" in report
    assert "**Key Risks**" in report
    assert "**Thesis**" in report
    assert "**Falsifiers**" in report
    assert "**Conviction Grade With Reasoning**" in report
    assert "## Selection / Watchlist / No-Selection Decision" in report
    assert "**Binding result:** Selected." in report
    assert "- Base-case return falls below hurdle." in report
    assert "## Audit Appendix" in report
    assert "### Run Metadata" in report
    assert "| Final Verdict | SELECTED |" in report
    assert "### Candidate Selection" in report
    assert "| Source | sector_scan_db |" in report
    assert "| Selected Tickers | AAA, BBB |" in report
    assert "### Financial Framework" in report
    assert "| Economic Model | Small-cap financial underwriting. |" in report
    assert "### Selection Audit" in report
    assert "| Audit Status | PASS |" in report
    assert "| Actionable | true |" in report
    assert "| Base Return Hurdle | 12.0% |" in report
    assert "| Best Base Annualized Return | 24.1% |" in report
    assert "| Capital Structure Resolution Status | CAPITAL_STRUCTURE_RESOLVED_CLEAR |" in report
    assert (
        "| Maturity Schedule | 2027: 25 million; 2028-09-30: 40 million; thereafter: 45 million |"
        in report
    )
    assert (
        "| Covenant Terms | consolidated net leverage ratio: MAXIMUM 3.5 ratio; liquidity: MINIMUM 20 million |"
        in report
    )
    assert "| Audit Gap Repair Attempted | false |" in report
    assert "| Watchlist Resolution Attempted | false |" in report
    assert "| Company Autonomy Attempted | true |" in report
    assert "| Company Autonomy Status | COMPLETED |" in report
    assert "### Company Autonomy" in report
    assert "| AAA | COMPLETED | ACTIONABLE | MODERATE | 2 | 2 | N/A |" in report
    assert "### Relative Ranking" in report
    assert (
        "| 1 | AAA | 24.1% | 3.3% | PASS | true | ACTIONABLE | N/A | N/A | Best-positioned and actionable under current guardrails. |"
        in report
    )
    assert "### Company Financial Packets" in report
    assert (
        "| AAA | Financially Viable | VALID_GENERIC | OK | $50.00 | dcf | $100.00 | 50.0% | N/A | CYCLE_DURABILITY_UNRESOLVED |"
        in report
    )
    assert "### Expected Return Scenarios" in report
    assert "| AAA | 5Y | 3.3% | 24.1% | 28.7% | $58.75 | $146.93 | $176.32 |" in report
    assert "### Research Questions" in report
    assert (
        "| Q1 | ANSWERED | HIGH | Expected return | AAA, BBB | rank_expected_return_cases, fetch_kpi_trends | Which candidate clears the expected-return hurdle? |"
        in report
    )
    assert "### Tool Calls" in report
    assert "| TC1 | Q1 | rank_expected_return_cases | OK | E1 | N/A |" in report
    assert "### Evidence References" in report
    assert (
        "| E1 | rank_expected_return_cases | AAA | MODERATE | AAA ranked first on base-case expected return. |"
        in report
    )
    assert "### Belief Updates" in report
    assert (
        "| BU1 | Q1 | AAA | BULLISH | MODERATE | Expected-return evidence supports AAA. | Cycle durability. |"
        in report
    )
    assert "### AI Working Notes And Runtime Audit" in report
    assert "### Guardrail Outcome" not in report
    assert "These notes are pre/post-processing audit trail, not the final decision." not in report


def test_render_autonomous_sector_report_distinguishes_computed_and_declared_metrics():
    artifact = _artifact()
    assert artifact.framework is not None
    artifact.framework.selected_metrics = [
        "revenue_cagr_5y",
        "gross_margin",
        "cash_conversion_cycle",
        "share_count_cagr",
        "book_to_bill_or_backlog_growth",
        "maintenance_capex_to_sales",
    ]
    artifact.company_packets[0].business_quality = {
        "revenue_cagr_5y": 0.12,
        "gross_margin": 0.38,
        "gross_margin_trajectory_5y": [
            {"fiscal_year": 2024, "gross_margin": 0.35, "not_computable_reasons": []},
            {"fiscal_year": 2025, "gross_margin": 0.38, "not_computable_reasons": []},
        ],
    }
    artifact.company_packets[0].cash_conversion = {
        "cash_conversion_cycle": 54.75,
        "days_sales_outstanding": 36.5,
        "days_inventory_outstanding": 54.75,
        "days_payable_outstanding": 36.5,
        "cash_conversion_cycle_trajectory_5y": [
            {"fiscal_year": 2024, "cash_conversion_cycle": 60.0, "not_computable_reasons": []},
            {"fiscal_year": 2025, "cash_conversion_cycle": 54.75, "not_computable_reasons": []},
        ],
    }
    artifact.company_packets[0].capital_allocation = {
        "share_count_cagr": -0.02,
        "share_count_cagr_direction": "BUYBACKS",
        "share_count_oldest": 100.0,
        "share_count_latest": 92.24,
    }

    report = render_autonomous_sector_report(artifact)

    assert (
        "Used metrics: revenue_cagr_5y, gross_margin, cash_conversion_cycle, share_count_cagr."
        in report
    )
    assert (
        "Declared but not yet implemented: book_to_bill_or_backlog_growth, maintenance_capex_to_sales."
        in report
    )
    aaa_section = _between(report, "### AAA — AAA", "### BBB — BBB")
    assert (
        "- Gross margin: latest 38.0%; 5-year trajectory 2024: 35.0%; 2025: 38.0%." in aaa_section
    )
    assert (
        "- Cash conversion cycle: latest 54.8 days; DSO 36.5 days; DIO 54.8 days; DPO 36.5 days;"
        in aaa_section
    )
    assert "- Share count CAGR: -2.0% 5Y (buybacks; shares 100.00 to 92.24)." in aaa_section


def test_render_autonomous_sector_report_surfaces_audit_renderer_gap_metrics():
    artifact = _artifact()
    assert artifact.framework is not None
    artifact.framework.selected_metrics = [
        "free_cash_flow_margin",
        "inventory_days",
        "inventory_turnover",
    ]
    artifact.company_packets[0].cash_conversion = {
        "fcf_margin": 0.1234,
        "days_inventory_outstanding": 45.625,
        "cash_conversion_cycle_not_computable_reasons": [],
    }
    artifact.company_packets[1].cash_conversion = {
        "fcf_margin": None,
        "days_inventory_outstanding": None,
        "cash_conversion_cycle_not_computable_reasons": [
            "CCC_NOT_COMPUTABLE",
            "INVENTORY_MISSING",
            "COGS_MISSING",
        ],
    }

    report = render_autonomous_sector_report(artifact)

    assert "Used metrics: free_cash_flow_margin, inventory_days, inventory_turnover." in report
    aaa_section = _between(report, "### AAA — AAA", "### BBB — BBB")
    assert "- Free cash flow margin: 12.3%." in aaa_section
    assert "- Inventory days: 45.6 days." in aaa_section
    assert "- Inventory turnover: 8.0x." in aaa_section
    bbb_section = report.split("### BBB — BBB", 1)[1]
    assert "- Free cash flow margin: N/A (free cash flow margin is not computable)." in bbb_section
    assert (
        "- Inventory days: N/A (cash conversion cycle is not computable, inventory is missing, COGS is missing)."
        in bbb_section
    )
    assert (
        "- Inventory turnover: N/A (cash conversion cycle is not computable, inventory is missing, COGS is missing)."
        in bbb_section
    )


def test_render_autonomous_sector_report_surfaces_normalized_operating_margin():
    artifact = _artifact()
    assert artifact.framework is not None
    artifact.framework.selected_metrics = ["normalized_operating_margin"]
    artifact.company_packets[0].business_quality = {
        "normalized_operating_margin": 0.209,
        "latest_operating_margin": 0.008,
        "normalized_operating_margin_status": "OK",
        "normalized_operating_margin_not_computable_reasons": [],
    }
    artifact.company_packets[1].business_quality = {
        "normalized_operating_margin": None,
        "latest_operating_margin": 0.05,
        "normalized_operating_margin_status": "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY",
        "normalized_operating_margin_not_computable_reasons": [
            "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY",
            "OPERATING_INCOME_MISSING",
        ],
    }

    report = render_autonomous_sector_report(artifact)

    assert "Used metrics: normalized_operating_margin." in report
    aaa_section = _between(report, "### AAA — AAA", "### BBB — BBB")
    assert "- Normalized operating margin (5Y avg): 20.9% vs latest 0.8%." in aaa_section
    bbb_section = report.split("### BBB — BBB", 1)[1]
    assert (
        "- Normalized operating margin (5Y avg): N/A "
        "(normalized margin history is insufficient, operating income is missing)."
    ) in bbb_section


def test_render_autonomous_sector_report_adds_historical_multiple_bands_line():
    artifact = _artifact()
    artifact.company_packets[0].valuation["historical_multiples"] = {
        "ev_to_ebitda": {
            "current_value": 14.2,
            "range_min": 8.1,
            "range_median": 11.7,
            "range_max": 18.5,
            "current_percentile": 78.0,
            "years_of_history": 10,
            "status": "OK",
            "history": [],
        },
        "pe": {
            "current_value": 18.0,
            "range_min": 10.0,
            "range_median": 15.0,
            "range_max": 25.0,
            "current_percentile": 60.0,
            "years_of_history": 6,
            "status": "INSUFFICIENT_HISTORY",
            "history": [],
        },
        "price_to_book": {
            "current_value": 2.4,
            "range_min": 1.2,
            "range_median": 2.0,
            "range_max": 3.0,
            "current_percentile": 70.0,
            "years_of_history": 10,
            "status": "OK",
            "history": [],
        },
        "fcf_yield": {
            "current_value": 0.045,
            "range_min": 0.01,
            "range_median": 0.03,
            "range_max": 0.08,
            "current_percentile": 72.0,
            "years_of_history": 10,
            "status": "OK",
            "history": [],
        },
    }

    report = render_autonomous_sector_report(artifact)

    aaa_section = _between(report, "### AAA — AAA", "### BBB — BBB")
    assert (
        "Historical multiples: EV/EBITDA at the 78.0th percentile of its 10-year range "
        "(current 14.2x vs range 8.1x – 18.5x, median 11.7x); "
        "P/E at the 60.0th percentile of its available 6-year range "
        "(current 18.0x vs range 10.0x – 25.0x, median 15.0x) "
        "(full 10y history not available); "
        "P/B at the 70.0th percentile of its 10-year range "
        "(current 2.4x vs range 1.2x – 3.0x, median 2.0x); "
        "FCF yield at the 72.0th percentile of its 10-year range "
        "(current 4.5% vs range 1.0% – 8.0%, median 3.0%) (high yield = cheap)."
    ) in aaa_section


def test_render_sector_report_exposes_freshness_diagnostics_in_evidence_references():
    artifact = _artifact()
    artifact.evidence.append(
        EvidenceReference(
            evidence_id="E2",
            source_type="tool_output",
            source_label="fetch_current_events",
            summary=(
                "No current-event documents were available from configured company-controlled sources "
                "(metadata=universe_members, homepage=present, ir_rss=missing)."
            ),
            ticker="AAA",
            confidence="LOW",
        )
    )
    artifact.evidence.append(
        EvidenceReference(
            evidence_id="E3",
            source_type="tool_output",
            source_label="fetch_recent_filing_context",
            summary=(
                "1 readable latest annual filing document(s) available as fresh filing evidence "
                "(source_strategy=section:risk_factors)."
            ),
            ticker="AAA",
            confidence="MODERATE",
        )
    )

    report = render_autonomous_sector_report(artifact)

    assert (
        "| E2 | fetch_current_events | AAA | LOW | "
        "No current-event documents were available from configured company-controlled sources "
        "(metadata=universe_members, homepage=present, ir_rss=missing). |"
    ) in report
    assert (
        "| E3 | fetch_recent_filing_context | AAA | MODERATE | "
        "1 readable latest annual filing document(s) available as fresh filing evidence "
        "(source_strategy=section:risk_factors). |"
    ) in report


def test_render_watchlist_report_labels_focus_candidate_as_non_actionable():
    artifact = _artifact()
    artifact.final_verdict = "WATCHLIST"
    artifact.confidence = "MODERATE"
    artifact.degraded_states = ["SELECTION_AUDIT_WATCHLIST_ONLY"]
    artifact.watchlist_resolution_attempted = True
    artifact.watchlist_resolution_status = "UNRESOLVED_WATCHLIST"
    artifact.watchlist_resolution_notes = [
        "Follow-up evidence did not clear all watchlist audit caps."
    ]
    artifact.selection_audit = {
        "status": "WATCHLIST_ONLY",
        "selected_ticker": "AAA",
        "actionable": False,
        "confidence_ceiling": "MODERATE",
        "hard_blockers": [],
        "confidence_caps": ["STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE"],
        "expected_return_evidence_count": 1,
        "company_specific_evidence_count": 1,
        "base_return_hurdle": 0.12,
        "best_base_annualized_return": 0.240594,
        "best_base_horizon_years": 5,
        "notes": ["Selection audit found non-binding evidence caps; candidate is watchlist-only."],
        "final_verdict_after_audit": "WATCHLIST",
    }
    artifact.final_decision = SectorFinalDecision(
        verdict="WATCHLIST",
        confidence="MODERATE",
        selected_ticker="AAA",
        expected_annualized_return_range="20-25%",
        thesis="AAA is financially interesting but not actionable yet.",
        key_risk="The filing evidence is stale.",
        downside_case="The return case needs fresher confirmation.",
        falsifiers=["Fresh evidence weakens the return case."],
        confidence_cap_reasons=["STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE"],
        evidence_ref_ids=["E1"],
    )

    report = render_autonomous_sector_report(artifact)

    assert "| Final Verdict | WATCHLIST |" in report
    assert "## Executive Conclusion" in report
    assert "remains a watchlist candidate" in report
    assert "## Selection / Watchlist / No-Selection Decision" in report
    assert "The audited result is **WATCHLIST**, not an actionable sector selection." in report
    assert "| Audit Status | WATCHLIST_ONLY |" in report
    assert "| Actionable | false |" in report
    assert "| Watchlist Resolution Attempted | true |" in report
    assert "| Watchlist Resolution Status | UNRESOLVED_WATCHLIST |" in report
    assert (
        "| Watchlist Resolution Notes | Follow-up evidence did not clear all watchlist audit caps. |"
        in report
    )
    assert "| Alternate Finalist Audit Attempted | false |" in report
    assert "| Confidence Caps | STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE |" in report
    assert "### Guardrail Outcome" not in report


def test_render_no_selection_report_makes_guardrail_outcome_dominant():
    artifact = _artifact()
    artifact.final_verdict = "NO_SELECTION"
    artifact.selected_ticker = None
    artifact.confidence = None
    artifact.no_selection_reason = "No company selected because unresolved evidence gaps remained."
    artifact.degraded_states = ["STRUCTURED_DECISION_INCOMPLETE", "NO_CURRENT_EVENTS"]
    artifact.audit_gap_repair_attempted = True
    artifact.audit_gap_repair_status = "UNRESOLVED_BLOCKED"
    artifact.audit_gap_repair_notes = ["Repair evidence did not clear the binding audit blockers."]
    artifact.no_selection_finalist_audit_attempted = True
    artifact.no_selection_finalist_audit_status = "BLOCKED"
    artifact.no_selection_finalist_audit_focus_ticker = "AAA"
    artifact.no_selection_finalist_audit_notes = [
        "Audited AAA after provider no-selection: highest available base-case expected return from deterministic candidate selection.",
        "No company selected because finalist AAA failed selection audit: MISSING_COMPANY_SPECIFIC_EVIDENCE.",
    ]
    artifact.no_selection_finalist_resolution_attempted = True
    artifact.no_selection_finalist_resolution_status = "UNRESOLVED_NO_SELECTION"
    artifact.no_selection_finalist_resolution_notes = [
        "Follow-up evidence did not clear all no-selection finalist audit caps."
    ]
    artifact.alternate_finalist_audit_attempted = True
    artifact.alternate_finalist_audit_status = "NO_ALTERNATE_PASSED"
    artifact.alternate_finalist_audit_notes = [
        "No alternate finalist cleared the deterministic selection audit."
    ]
    artifact.alternate_finalist_audit_results = [
        {
            "ticker": "BBB",
            "source": "provider_rejected_finalist",
            "audit_status": "BLOCKED",
            "hard_blockers": ["BASE_RETURN_BELOW_12PCT_HURDLE"],
            "confidence_caps": [],
            "best_base_annualized_return": 0.08,
        }
    ]
    artifact.final_decision = SectorFinalDecision(
        verdict="NO_SELECTION",
        confidence=None,
        selected_ticker=None,
        expected_annualized_return_range=None,
        thesis="No sector selection was made.",
        key_risk="Evidence quality was insufficient.",
        downside_case="No underwritten downside case was finalized.",
        no_selection_reason="No company selected because unresolved evidence gaps remained.",
        selection_blockers=["STRUCTURED_DECISION_INCOMPLETE"],
    )
    artifact.selection_audit = {
        "status": "BLOCKED",
        "selected_ticker": "AAA",
        "actionable": False,
        "confidence_ceiling": None,
        "hard_blockers": ["MISSING_COMPANY_SPECIFIC_EVIDENCE"],
        "confidence_caps": [],
        "expected_return_evidence_count": 1,
        "company_specific_evidence_count": 0,
        "base_return_hurdle": 0.12,
        "best_base_annualized_return": 0.240594,
        "best_base_horizon_years": 5,
        "notes": ["Selection audit found binding hard blockers."],
        "final_verdict_after_audit": "NO_SELECTION",
    }
    artifact.audit_notes = [
        "Provider working note: SELECT AAA at half position.",
        "Runtime guardrails are binding over provider working notes; top-level final_verdict and final_decision fields are the source of truth.",
    ]

    report = render_autonomous_sector_report(artifact)

    assert "## Selection / Watchlist / No-Selection Decision" in report
    assert "## Audit Appendix" in report
    assert "### Guardrail Outcome" in report
    assert "### Selection Audit" in report
    assert "| Audit Status | BLOCKED |" in report
    assert "| Audit Gap Repair Attempted | true |" in report
    assert "| Audit Gap Repair Status | UNRESOLVED_BLOCKED |" in report
    assert (
        "| Audit Gap Repair Notes | Repair evidence did not clear the binding audit blockers. |"
        in report
    )
    assert "| No-Selection Finalist Audit Attempted | true |" in report
    assert "| No-Selection Finalist Audit Status | BLOCKED |" in report
    assert "| No-Selection Finalist Audit Focus | AAA |" in report
    assert (
        "| No-Selection Finalist Audit Notes | "
        "Audited AAA after provider no-selection: highest available base-case expected return from deterministic candidate selection., "
        "No company selected because finalist AAA failed selection audit: MISSING_COMPANY_SPECIFIC_EVIDENCE. |"
    ) in report
    assert "| No-Selection Finalist Resolution Attempted | true |" in report
    assert "| No-Selection Finalist Resolution Status | UNRESOLVED_NO_SELECTION |" in report
    assert (
        "| No-Selection Finalist Resolution Notes | "
        "Follow-up evidence did not clear all no-selection finalist audit caps. |"
    ) in report
    assert "| Alternate Finalist Audit Attempted | true |" in report
    assert "| Alternate Finalist Audit Status | NO_ALTERNATE_PASSED |" in report
    assert (
        "| Alternate Finalist Audit Notes | No alternate finalist cleared the deterministic selection audit. |"
        in report
    )
    assert (
        "| Alternate Finalist Audit Results | "
        "BBB: BLOCKED (source=provider_rejected_finalist; base=8.0%; "
        "blockers=BASE_RETURN_BELOW_12PCT_HURDLE; caps=N/A) |"
    ) in report
    assert "The binding artifact result is **NO_SELECTION**." in report
    assert "- Selected ticker: N/A" in report
    assert "- Confidence: N/A" in report
    assert (
        "- No-selection reason: No company selected because unresolved evidence gaps remained."
        in report
    )
    assert "### AI Working Notes And Runtime Audit" in report
    assert "These notes are pre/post-processing audit trail, not the final decision." in report
    assert "- Provider working note: SELECT AAA at half position." in report
    assert report.index("### Selection Audit") < report.index("### Guardrail Outcome")
    assert report.index("### Guardrail Outcome") < report.index(
        "### AI Working Notes And Runtime Audit"
    )


def test_render_provider_failure_states_are_visible_before_audit_notes():
    artifact = _artifact()
    artifact.final_verdict = "NO_SELECTION"
    artifact.selected_ticker = None
    artifact.confidence = None
    artifact.degraded_states = ["LLM_PROVIDER_INVALID_JSON", "LLM_PROVIDER_TIMEOUT"]
    artifact.no_selection_reason = "LLM provider failed during first sector planning turn."
    artifact.selection_audit = {}
    artifact.final_decision = SectorFinalDecision(
        verdict="NO_SELECTION",
        confidence=None,
        selected_ticker=None,
        expected_annualized_return_range=None,
        thesis="No selection was made because provider planning failed before tool execution.",
        key_risk="Forcing a selection without deterministic evidence would overstate conviction.",
        downside_case="No downside case was underwritten.",
        no_selection_reason="LLM provider failed during first sector planning turn.",
        selection_blockers=["LLM_PROVIDER_INVALID_JSON"],
    )
    artifact.audit_notes = [
        "Provider structured-output failure on sector turn 1.",
        "Stopped before tool execution without forcing a sector selection.",
    ]

    report = render_autonomous_sector_report(artifact)

    assert "### Degraded States" in report
    assert "- LLM_PROVIDER_INVALID_JSON" in report
    assert "- LLM_PROVIDER_TIMEOUT" in report
    assert "### Guardrail Outcome" in report
    assert "- No-selection reason: LLM provider failed during first sector planning turn." in report
    assert report.index("### Degraded States") < report.index(
        "### AI Working Notes And Runtime Audit"
    )


def test_render_deterministic_finalization_fallback_is_visible_with_audit():
    artifact = _artifact()
    artifact.final_verdict = "NO_SELECTION"
    artifact.selected_ticker = None
    artifact.confidence = None
    artifact.degraded_states = [
        "LLM_PROVIDER_TIMEOUT",
        "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
    ]
    artifact.no_selection_reason = (
        "LLM provider final decision was unavailable after deterministic evidence collection; "
        "the runtime will attempt deterministic finalist audit instead of forcing a selection."
    )
    artifact.no_selection_finalist_audit_attempted = True
    artifact.no_selection_finalist_audit_status = "BLOCKED"
    artifact.no_selection_finalist_audit_focus_ticker = "AAA"
    artifact.no_selection_finalist_audit_notes = [
        "Audited AAA after provider no-selection: provider finalist with the strongest available base-case expected return.",
        "No company selected because finalist AAA failed selection audit: NO_FILING.",
    ]
    artifact.alternate_finalist_audit_attempted = True
    artifact.alternate_finalist_audit_status = "NO_ALTERNATE_PASSED"
    artifact.alternate_finalist_audit_notes = [
        "No alternate finalist cleared the deterministic selection audit."
    ]
    artifact.alternate_finalist_audit_results = [
        {
            "ticker": "BBB",
            "source": "provider_rejected_finalist",
            "audit_status": "WATCHLIST_ONLY",
            "hard_blockers": [],
            "confidence_caps": ["THIN_RETURN_CUSHION"],
            "best_base_annualized_return": 0.13,
        }
    ]
    artifact.selection_audit = {
        "status": "BLOCKED",
        "selected_ticker": "AAA",
        "actionable": False,
        "confidence_ceiling": None,
        "hard_blockers": ["NO_FILING"],
        "confidence_caps": [],
        "expected_return_evidence_count": 1,
        "company_specific_evidence_count": 2,
        "base_return_hurdle": 0.12,
        "best_base_annualized_return": 0.22,
        "best_base_horizon_years": 5,
        "notes": ["Selection audit found binding hard blockers."],
        "final_verdict_after_audit": "NO_SELECTION",
    }
    artifact.final_decision = SectorFinalDecision(
        verdict="NO_SELECTION",
        confidence=None,
        selected_ticker=None,
        expected_annualized_return_range=None,
        thesis="No provider-authored final decision was available; deterministic audit governed the outcome.",
        key_risk="Provider final-decision failure left evidence interpretation incomplete.",
        downside_case="No provider-authored downside synthesis was available.",
        no_selection_reason=artifact.no_selection_reason,
        selection_blockers=[
            "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
            "NO_SELECTION_FINALIST_AUDIT_BLOCKED",
            "NO_FILING",
        ],
    )
    artifact.audit_notes = [
        "Deterministic finalization fallback ran after provider final-decision failure."
    ]

    report = render_autonomous_sector_report(artifact)

    assert "### Degraded States" in report
    assert "- LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE" in report
    assert "### Selection Audit" in report
    assert "| No-Selection Finalist Audit Attempted | true |" in report
    assert "| Alternate Finalist Audit Attempted | true |" in report
    assert (
        "BBB: WATCHLIST_ONLY (source=provider_rejected_finalist; base=13.0%; "
        "blockers=N/A; caps=THIN_RETURN_CUSHION)"
    ) in report
    assert "### AI Working Notes And Runtime Audit" in report
    assert (
        "- Deterministic finalization fallback ran after provider final-decision failure." in report
    )
    assert report.index("### Degraded States") < report.index(
        "### AI Working Notes And Runtime Audit"
    )


def test_cohort_comparison_renders_with_provider_mocked():
    artifact = _artifact()
    provider = MemoBodyProvider(
        payloads_by_schema={
            "autonomous_sector_cohort_comparison": {
                "paragraphs": [
                    "AAA is the cleanest asymmetric candidate: 24.1% base return, 11.5% ROIC, and 5.0% FCF yield beat BBB's incomplete packet.",
                    "BBB remains useful as a contrast case because missing valuation evidence prevents the apparent downside setup from becoming investable.",
                ],
                "generation_notes": ["Cohort comparison written."],
            },
            "autonomous_sector_triage_surprises": {
                "items": [
                    "AAA ranks first deterministically and first in the memo order. That matters because the AI and packet math are aligned.",
                ],
                "generation_notes": ["Triage written."],
            },
            "autonomous_sector_candidate_memo_aaa": {
                "ticker": "AAA",
                "thesis": "AAA is a financially viable specialty-manufacturing candidate with the best return case in the cohort. The tension is attractive valuation against unresolved cycle durability. Revenue quality needs deeper validation, but the packet shows 24.1% base-case annualized return and 5.0% FCF yield. The verdict stays moderate because the return case clears the hurdle while cycle evidence caps confidence.",
                "key_risks": [
                    "Cycle durability remains unresolved despite a 24.1% base-case return.",
                    "FCF yield of 5.0% needs confirmation through a full-cycle downturn.",
                    "The $100.00 valuation anchor depends on normalized cash conversion holding.",
                ],
                "falsifiers": [
                    "Base-case annualized return falls below 12.0% after fresh evidence.",
                    "Cash conversion weakens enough to invalidate the $100.00 anchor.",
                ],
                "open_questions": [
                    "What end-market evidence supports normalized demand?",
                    "How resilient is margin through a cycle?",
                    "What would lift confidence above moderate?",
                ],
            },
            "autonomous_sector_candidate_memo_bbb": {
                "ticker": "BBB",
                "thesis": "BBB is a lower-quality contrast candidate with incomplete valuation evidence. The central tension is whether missing data hides upside or simply confirms weak investability. The packet shows 8.0% base-case return and missing valuation support. The verdict remains avoid because the evidence base does not clear the memo bar.",
                "key_risks": [
                    "Base return is only 8.0%.",
                    "Valuation evidence is missing.",
                    "Downside return is -12.0%.",
                ],
                "falsifiers": ["Fresh valuation support moves base return above 12.0%."],
                "open_questions": [
                    "Can valuation evidence be recovered?",
                    "Why is downside return so weak?",
                    "Is missing data temporary?",
                ],
            },
        }
    )

    enrich_sector_artifact_memo_body(artifact, provider=provider)
    report = render_autonomous_sector_report(artifact)

    assert provider.calls[0]["schema_name"] == "autonomous_sector_cohort_comparison"
    assert {call["schema_name"] for call in provider.calls} >= {
        "autonomous_sector_triage_surprises",
        "autonomous_sector_candidate_memo_aaa",
        "autonomous_sector_candidate_memo_bbb",
    }
    assert artifact.memo_body["status"] == "LLM_GENERATED"
    assert artifact.memo_body["usage"]["input_tokens"] > 0
    assert len(artifact.provider_usage) == 4
    assert {row["schema_name"] for row in artifact.provider_usage} == {
        "autonomous_sector_cohort_comparison",
        "autonomous_sector_triage_surprises",
        "autonomous_sector_candidate_memo_aaa",
        "autonomous_sector_candidate_memo_bbb",
    }
    assert all(row["lane"] == "parent_research" for row in artifact.provider_usage)
    assert [row["provider_call_id"] for row in artifact.provider_usage] == [
        "P1",
        "P2",
        "P3",
        "P4",
    ]
    assert "## Cohort Comparison" in report
    assert "AAA is the cleanest asymmetric candidate: 24.1% base return" in report
    assert "## Triage Surprises" in report
    assert "- AAA ranks first deterministically and first in the memo order." in report
    assert "The tension is attractive valuation against unresolved cycle durability." in report
    assert "**Open Questions**" in report


def test_cohort_and_triage_prompt_preflight_skips_oversize_prompts(monkeypatch):
    artifact = _artifact()
    provider = MemoBodyProvider(
        payloads_by_schema={
            "autonomous_sector_candidate_memo_aaa": {
                "ticker": "AAA",
                "thesis": "AAA remains viable, but the memo prompt guard skipped cohort synthesis.",
                "key_risks": ["Cycle risk remains measurable at a 24.1% base return."],
                "falsifiers": ["Base return falls below 12.0%."],
                "open_questions": ["What evidence resolves cycle durability?"],
            },
            "autonomous_sector_candidate_memo_bbb": {
                "ticker": "BBB",
                "thesis": "BBB remains an incomplete contrast candidate.",
                "key_risks": ["Missing valuation data limits confidence."],
                "falsifiers": ["Recovered valuation data clears the hurdle."],
                "open_questions": ["Can missing valuation data be recovered?"],
            },
        }
    )
    monkeypatch.setattr("app.autonomous.sector_runtime.SECTOR_MEMO_PROMPT_PREFLIGHT_MAX_TOKENS", 1)

    enrich_sector_artifact_memo_body(artifact, provider=provider)

    called_schemas = {call["schema_name"] for call in provider.calls}
    assert "autonomous_sector_cohort_comparison" not in called_schemas
    assert "autonomous_sector_triage_surprises" not in called_schemas
    assert called_schemas == {
        "autonomous_sector_candidate_memo_aaa",
        "autonomous_sector_candidate_memo_bbb",
    }
    assert "COHORT_PROMPT_OVERSIZE" in artifact.memo_body["degraded_states"]
    assert (
        "cohort_comparison_fallback:COHORT_PROMPT_OVERSIZE" in artifact.memo_body["degraded_states"]
    )
    assert (
        "triage_surprises_fallback:COHORT_PROMPT_OVERSIZE" in artifact.memo_body["degraded_states"]
    )
    assert artifact.memo_body["cohort_comparison"]["source"] == "deterministic_fallback"
    assert artifact.memo_body["triage_surprises"]["source"] == "deterministic_fallback"


def test_cohort_comparison_falls_back_deterministically_when_provider_raises():
    artifact = _artifact()
    provider = MemoBodyProvider(exc=TimeoutError("memo body timed out"))

    enrich_sector_artifact_memo_body(artifact, provider=provider)
    report = render_autonomous_sector_report(artifact)

    assert artifact.memo_body["status"] == "DEGRADED_FALLBACK"
    assert set(artifact.memo_body["degraded_states"]) == {
        "cohort_comparison_fallback:LLM_PROVIDER_TIMEOUT",
        "triage_surprises_fallback:LLM_PROVIDER_TIMEOUT",
        "candidate_thesis_fallback:AAA:LLM_PROVIDER_TIMEOUT",
        "candidate_thesis_fallback:BBB:LLM_PROVIDER_TIMEOUT",
    }
    assert "## Cohort Comparison" in report
    assert (
        "**DEGRADED_STATE:** LLM_PROVIDER_TIMEOUT; deterministic fallback uses packet values."
        in report
    )
    assert "Among the 2 displayed candidates" in report
    assert "AAA has the highest latest ROIC at 11.5%" in report


def test_candidate_memo_body_falls_back_per_ticker_when_one_call_fails():
    artifact = _artifact()
    provider = MemoBodyProvider(
        payloads_by_schema={
            "autonomous_sector_cohort_comparison": {
                "paragraphs": [
                    "AAA has the cleaner return case, while BBB remains a missing-data contrast."
                ],
                "generation_notes": [],
            },
            "autonomous_sector_triage_surprises": {
                "items": [
                    "BBB is lower ranked and missing valuation support. That matters because the packet cannot underwrite upside."
                ],
                "generation_notes": [],
            },
            "autonomous_sector_candidate_memo_aaa": {
                "ticker": "AAA",
                "thesis": "AAA is the cleaner cohort candidate. The central tension is attractive return versus cycle durability. The packet shows 24.1% base-case return and 5.0% FCF yield. The verdict remains moderate because the case clears the hurdle but needs better cycle evidence.",
                "key_risks": ["Cycle durability remains unresolved at 24.1% base-case return."],
                "falsifiers": ["Base return falls below 12.0%."],
                "open_questions": ["What evidence resolves cycle durability?"],
            },
        },
        exc_by_schema={"autonomous_sector_candidate_memo_bbb": TimeoutError("BBB timed out")},
    )

    enrich_sector_artifact_memo_body(artifact, provider=provider)
    report = render_autonomous_sector_report(artifact)

    assert artifact.memo_body["status"] == "PARTIAL_LLM_GENERATED"
    assert artifact.memo_body["candidates"]["AAA"]["source"] == "llm"
    assert artifact.memo_body["candidates"]["BBB"]["source"] == "deterministic_fallback"
    assert (
        "candidate_thesis_fallback:BBB:LLM_PROVIDER_TIMEOUT"
        in artifact.memo_body["degraded_states"]
    )
    assert "AAA is the cleaner cohort candidate." in report
    bbb_section = _between(report, "### BBB — BBB", "**Falsifiers**")
    assert (
        "**DEGRADED_STATE:** LLM_PROVIDER_TIMEOUT; deterministic fallback uses packet values."
        in bbb_section
    )


def test_candidate_memo_ticker_mismatch_fails_coverage_binding():
    artifact = _artifact()
    provider = MemoBodyProvider(
        payloads_by_schema={
            "autonomous_sector_cohort_comparison": {"paragraphs": ["AAA leads."]},
            "autonomous_sector_triage_surprises": {"items": ["No surprises."]},
            "autonomous_sector_candidate_memo_aaa": {
                "ticker": "BBB",
                "thesis": "Wrong issuer response.",
                "key_risks": ["Wrong issuer."],
                "falsifiers": ["Ticker identity matches."],
                "open_questions": ["Why did identity drift?"],
            },
            "autonomous_sector_candidate_memo_bbb": {
                "ticker": "BBB",
                "thesis": "BBB remains a contrast candidate.",
                "key_risks": ["Return is below hurdle."],
                "falsifiers": ["Return clears hurdle."],
                "open_questions": ["What changes the return case?"],
            },
        }
    )

    enrich_sector_artifact_memo_body(artifact, provider=provider)
    projection = v1_terminal_coverage_from_artifact(artifact)

    assert artifact.memo_body["candidates"]["AAA"]["source"] == ("deterministic_fallback")
    assert artifact.memo_body["candidates"]["AAA"]["degraded_state"] == (
        "CANDIDATE_MEMO_TICKER_MISMATCH"
    )
    assert projection["llm_candidate_review_completed"] == ["BBB"]
    assert projection["llm_candidate_review_failed"] == ["AAA"]


def test_empty_candidate_memo_content_fails_coverage_binding():
    artifact = _artifact()
    provider = MemoBodyProvider(
        payloads_by_schema={
            "autonomous_sector_cohort_comparison": {"paragraphs": ["AAA leads."]},
            "autonomous_sector_triage_surprises": {"items": ["No surprises."]},
            "autonomous_sector_candidate_memo_aaa": {
                "ticker": "AAA",
                "thesis": "",
                "key_risks": [],
                "falsifiers": [],
                "open_questions": [],
            },
            "autonomous_sector_candidate_memo_bbb": {
                "ticker": "BBB",
                "thesis": "BBB remains a contrast candidate.",
                "key_risks": ["Return is below hurdle."],
                "falsifiers": ["Return clears hurdle."],
                "open_questions": ["What changes the return case?"],
            },
        }
    )

    enrich_sector_artifact_memo_body(artifact, provider=provider)
    projection = v1_terminal_coverage_from_artifact(artifact)

    assert artifact.memo_body["candidates"]["AAA"]["source"] == ("deterministic_fallback")
    assert artifact.memo_body["candidates"]["AAA"]["degraded_state"] == (
        "CANDIDATE_MEMO_CONTENT_INCOMPLETE"
    )
    assert projection["llm_candidate_review_completed"] == ["BBB"]
    assert projection["llm_candidate_review_failed"] == ["AAA"]


def test_candidate_memo_review_attempts_every_packet_beyond_shared_context_cap():
    artifact = _artifact()
    tickers = [f"T{index:02d}" for index in range(26)]
    artifact.candidate_selection["selected_tickers"] = tickers
    artifact.candidate_selection["loaded_tickers"] = tickers
    artifact.company_packets = [
        SectorCompanyFinancialPacket(
            ticker=ticker,
            financial_status="Financially Viable",
            model_fit_status="VALID_GENERIC",
            data_quality_status="OK",
        )
        for ticker in tickers
    ]
    artifact.expected_return_scenarios = []
    _bind_valid_financial_context(artifact)
    provider = MemoBodyProvider(
        payloads_by_schema={
            "autonomous_sector_cohort_comparison": {"paragraphs": ["Cohort compared."]},
            "autonomous_sector_triage_surprises": {"items": ["No surprises."]},
            **{
                f"autonomous_sector_candidate_memo_{ticker.lower()}": {
                    "ticker": ticker,
                    "thesis": f"{ticker} candidate review.",
                    "key_risks": ["Execution risk."],
                    "falsifiers": ["Execution improves."],
                    "open_questions": ["What changes the case?"],
                }
                for ticker in tickers
            },
        }
    )

    enrich_sector_artifact_memo_body(artifact, provider=provider)

    candidate_schemas = {
        call["schema_name"]
        for call in provider.calls
        if str(call["schema_name"]).startswith("autonomous_sector_candidate_memo_")
    }
    assert len(candidate_schemas) == 26
    assert set(artifact.memo_body["candidates"]) == set(tickers)
    assert len(v1_terminal_coverage_from_artifact(artifact)["llm_candidate_review_completed"]) == 26


def test_triage_surprises_section_catches_rank_divergence():
    artifact = _artifact()
    artifact.candidate_selection["selected_tickers"] = ["BBB", "AAA"]

    report = render_autonomous_sector_report(artifact)

    assert "## Triage Surprises" in report
    assert (
        "- BBB appears #1 in the displayed finalist order but #2 in deterministic base-return rank. "
        "That matters because its base-case return is 8.0%"
    ) in report


def test_candidate_thesis_fallback_uses_numbers_not_enum_codes():
    artifact = _artifact()

    report = render_autonomous_sector_report(artifact)
    bbb_section = _between(report, "### BBB — BBB", "**Falsifiers**")

    assert "The base case projects 8.0% annualized" in bbb_section
    assert "base-case return remains below the 12% hurdle" in bbb_section
    assert "MISSING_VALUATION" not in bbb_section
    assert "BASE_RETURN_BELOW_12PCT_HURDLE" not in bbb_section


def test_enum_codes_remain_in_appendix_selection_audit_not_memo_body_thesis():
    artifact = _artifact()
    artifact.selection_audit["hard_blockers"] = ["NO_FILING"]

    report = render_autonomous_sector_report(artifact)
    aaa_thesis = _between(report, "### AAA — AAA", "**Falsifiers**")

    assert "NO_FILING" not in aaa_thesis
    assert "### Selection Audit" in report
    assert "| Hard Blockers | NO_FILING |" in report
    assert report.index("| Hard Blockers | NO_FILING |") > report.index("### Selection Audit")


def test_v2_incomplete_report_never_publishes_false_no_selection():
    artifact = _artifact()
    artifact.pipeline_version = "v2"
    artifact.execution_status = "COMPLETED"
    artifact.decision_status = "INCOMPLETE"
    artifact.provisional_final_decision = artifact.final_decision
    artifact.final_decision = None
    artifact.final_verdict = None
    artifact.selected_ticker = None

    report = render_autonomous_sector_report(artifact)
    executive = _between(report, "## Executive Conclusion", "## Candidate Coverage")
    decision = _between(
        report,
        "## Selection / Watchlist / No-Selection Decision",
        "## Candidate Disposition Funnel",
    )

    assert "Decision Incomplete" in executive
    assert "No company cleared" not in executive
    assert "Provisional thesis (non-binding)" in decision
    assert "Provisional key risk (non-binding)" in decision
    assert "Provisional downside case (non-binding)" in decision


def test_persist_autonomous_sector_run_writes_json_and_markdown(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    artifact = _artifact()
    monkeypatch.setattr(
        "app.autonomous.output_store.write_run_financial_authorization",
        lambda artifact_path, _report_path, **_kwargs: (
            artifact_path.parent / "financial_integrity_authorization.json"
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.output_store.bind_authorized_valuation_rows",
        lambda **_kwargs: 0,
    )

    paths = persist_autonomous_sector_run(artifact)

    assert paths.artifact_json.name == "autonomous_sector_run.json"
    assert paths.report_md.name == "autonomous_sector_report.md"
    assert paths.artifact_json.exists()
    assert paths.report_md.exists()
    payload = json.loads(paths.artifact_json.read_text(encoding="utf-8"))
    report = paths.report_md.read_text(encoding="utf-8")
    assert payload["run_id"] == "autonomous_sector_test_20260426_report"
    assert payload["candidate_selection"]["source"] == "sector_scan_db"
    assert "# specialty_manufacturing Sector Research Memo" in report
    assert "The sector run selects **AAA**" in report


def test_valuation_anchor_rejects_non_positive_value():
    # A non-positive DCF/EPV anchor must yield None (n/a), matching the store
    # (store._valuation_anchor guards value > 0), rather than rendering a bogus
    # negative/zero buy-price target for a name the store would have skipped.
    from app.autonomous import sector_report
    from app.autonomous.sector_contract import SectorCompanyFinancialPacket

    packet = SectorCompanyFinancialPacket(
        ticker="NEG",
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        current_price=10.0,
        valuation={"anchor_method": "DCF", "valuation_anchor": -5.0},
    )

    assert sector_report._valuation_anchor(packet) is None


def test_financial_snapshot_excludes_post_asof_and_undated_facts(
    monkeypatch,
    tmp_path,
):
    from app.autonomous import sector_report

    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE companyfacts_facts (
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                period_type TEXT NOT NULL,
                period_end TEXT,
                filed_date TEXT,
                line_item TEXT NOT NULL,
                value REAL,
                units TEXT,
                source_url TEXT,
                form TEXT,
                accession TEXT,
                UNIQUE(ticker, fiscal_year, period_type, line_item)
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, filed_date,
                line_item, value, units, source_url, form, accession
            )
            VALUES(
                'AAA', ?, 'FY', ?, ?, 'revenue', ?, 'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                '10-K', ?
            )
            """,
            [
                (2023, "2023-12-31", "2024-02-01", 100.0, "0000000001-24-000001"),
                (2024, "2024-12-31", "2026-03-01", 999.0, "0000000001-26-000001"),
                (2025, "2025-12-31", None, 888.0, "0000000001-26-000002"),
            ],
        )
    monkeypatch.setattr(sector_report, "_db_path", lambda: db_path)

    lines = sector_report._render_financial_snapshot(
        "AAA",
        as_of_date="2026-02-13",
    )
    rendered = "\n".join(lines)

    assert "| 2023 | 100.00 |" in rendered
    assert "999.00" not in rendered
    assert "888.00" not in rendered
