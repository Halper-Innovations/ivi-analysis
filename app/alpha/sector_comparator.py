"""LLM-first sector comparator.

Deterministic consensus ranking remains the evidence prior and shortlist
generator. The default shared path then lets the LLM choose which shortlisted
candidates deserve deeper work, investigate them with deterministic tools, and
make the canonical final winner selection.
"""

from __future__ import annotations

import logging
from typing import Any

from app.alpha.consensus_ranker import RankedEntry, rank_by_consensus
from app.alpha.llm_runtime import (
    AlphaInvestigationConfig,
    alpha_llm_available,
    decide_alpha_winner,
    plan_alpha_investigations,
    run_candidate_investigation,
)
from app.alpha.llm_tools import AlphaToolContext, default_as_of_date
from app.alpha.schemas import ComparisonRound, SectorAlphaReport, TickerSignalPacket
from app.alpha.solvency_scanner import going_concern_asserted
from app.autonomous.financial_integrity import FinancialIntegrityScope

logger = logging.getLogger(__name__)

PRIOR_SHORTLIST_COUNT = 8
MAX_INVESTIGATION_TARGETS = 5
_CONFIDENCE_TO_SCORE = {"HIGH": 80, "MODERATE": 60, "LOW": 35}


def _deterministic_thesis(entry: RankedEntry, pkt: TickerSignalPacket) -> str:
    parts = [f"{entry.ticker}:"]
    if pkt.insurance_value and pkt.current_price:
        parts.append(
            f"Insurance anchor ${pkt.insurance_value:.0f} vs price ${pkt.current_price:.0f}."
        )
    if pkt.dcf_value and pkt.current_price:
        discount = pkt.discount_to_dcf_pct()
        parts.append(f"DCF ${pkt.dcf_value:.0f} vs price ${pkt.current_price:.0f}")
        if discount is not None:
            parts.append(f"({discount:.0%} discount).")
    parts.append(f"Consensus score {entry.consensus_score:+.1f}.")
    parts.append(
        f"Gate: {pkt.gate_verdict}, Moat: {pkt.moat_classification} ({pkt.moat_score}), "
        f"Downside: {pkt.downside_risk_class}."
    )
    if entry.adjustments:
        parts.append(f"Adjustments: {', '.join(entry.adjustments)}.")
    return " ".join(parts)


def _explicit_distress(packet: TickerSignalPacket) -> bool:
    research = packet.research_report if isinstance(packet.research_report, dict) else {}
    solvency = research.get("solvency") if isinstance(research.get("solvency"), dict) else {}
    return going_concern_asserted(solvency) or bool(solvency.get("no_assurance_financing"))


def _hard_block_reasons(packet: TickerSignalPacket) -> list[str]:
    reasons: list[str] = []
    has_valuation = any(
        isinstance(value, (int, float))
        for value in (
            packet.insurance_value,
            packet.dcf_value,
            packet.epv_value,
            packet.graham_value,
            packet.ncav_value,
        )
    )
    if packet.current_price is None or not has_valuation:
        reasons.append("critical_data_missing")
    if (packet.solvency_risk or "").upper() == "CRITICAL":
        reasons.append("solvency_critical")
    if _explicit_distress(packet):
        reasons.append("explicit_distress")
    if packet.model_status == "MODEL_BLOCKED" and packet.model_blockers:
        if "critical_data_missing" not in reasons:
            reasons.append("critical_data_missing")
    return reasons


def _prior_candidate_payload(
    entry: RankedEntry,
    packet: TickerSignalPacket,
    *,
    consensus_rank: int,
) -> dict[str, Any]:
    research = packet.research_report if isinstance(packet.research_report, dict) else {}
    anomaly_list = research.get("anomalies") if isinstance(research.get("anomalies"), list) else []
    solvency = research.get("solvency") if isinstance(research.get("solvency"), dict) else {}
    return {
        "ticker": entry.ticker,
        "consensus_rank": consensus_rank,
        "consensus_score": entry.consensus_score,
        "method_discounts": entry.method_discounts,
        "methods_with_value": entry.methods_with_value,
        "methods_agreeing": entry.methods_agreeing,
        "ranking_adjustments": list(entry.adjustments),
        "discount_to_dcf_pct": packet.discount_to_dcf_pct(),
        "gate_verdict": packet.gate_verdict,
        "margin_of_safety_verdict": packet.margin_of_safety_verdict,
        "insurance_method": packet.insurance_method,
        "insurance_value": packet.insurance_value,
        "security_type": packet.security_type,
        "issuer_type": packet.issuer_type,
        "insurance_subtype": packet.insurance_subtype,
        "model_status": packet.model_status,
        "model_blockers": list(packet.model_blockers or []),
        "model_fit_warnings": list(packet.model_fit_warnings or []),
        "method_tension_type": packet.method_tension_type,
        "growth_dependency_ratio": packet.growth_dependency_ratio,
        "consensus_direction": packet.consensus_direction,
        "solvency_risk": packet.solvency_risk,
        "anomaly_count": len(anomaly_list),
        "filing_risk_signals": dict(packet.filing_risk_signals or {}),
        "quarterly_revenue_trend": packet.quarterly_revenue_trend,
        "peer_position": packet.peer_position,
        "hard_block_reasons": _hard_block_reasons(packet),
    }


def _preview_from_investigation(
    investigation: dict[str, Any],
    *,
    consensus_rank: int,
    consensus_score: float,
    packet: TickerSignalPacket,
) -> dict[str, Any]:
    confidence = str(investigation.get("confidence") or "LOW").upper()
    conviction_score = _CONFIDENCE_TO_SCORE.get(confidence, 35)
    hard_block_reasons = [
        str(item) for item in (investigation.get("hard_block_reasons") or []) if str(item).strip()
    ]
    return {
        "ticker": str(investigation.get("ticker") or ""),
        "pre_investigation_rank": consensus_rank,
        "consensus_score": consensus_score,
        "conviction_score": max(0, conviction_score - (10 * len(hard_block_reasons))),
        "conviction_class": confidence,
        "adjusted_mos": packet.discount_to_dcf_pct(),
        "hard_blockers": len(hard_block_reasons),
        "open_questions": len(investigation.get("open_questions") or []),
        "status": "OK",
        "verdict": investigation.get("verdict"),
        "report_path": None,
        "artifact_path": None,
        "investigation_mode": investigation.get("investigation_mode"),
    }


def _tool_budget_summary(
    *,
    investigation_plan: dict[str, Any],
    candidate_investigations: list[dict[str, Any]],
    final_decision: dict[str, Any],
    provider_mode: str,
    per_candidate_cap_usd: float,
) -> dict[str, Any]:
    planner_cost = float(investigation_plan.get("cost_usd") or 0.0)
    candidate_cost = sum(float(item.get("cost_usd") or 0.0) for item in candidate_investigations)
    final_cost = float(final_decision.get("cost_usd") or 0.0)
    total_cost = planner_cost + candidate_cost + final_cost
    total_turns = sum(int(item.get("num_turns") or 0) for item in candidate_investigations)
    total_tool_calls = sum(
        sum(int(count) for count in (item.get("tool_call_counts") or {}).values())
        for item in candidate_investigations
    )
    return {
        "provider_mode": provider_mode,
        "budget_scope": "STAGE_LOCAL_REQUEST_CAPS",
        "whole_comparison_cap_usd": None,
        "planned_candidates": len(candidate_investigations),
        "investigated_candidates": len(candidate_investigations),
        "per_candidate_cap_usd": per_candidate_cap_usd,
        "planner_request_cap_usd": investigation_plan.get("budget_usd"),
        "final_decision_request_cap_usd": final_decision.get("budget_usd"),
        "total_cost_usd": round(total_cost, 4),
        "planner_cost_usd": round(planner_cost, 4),
        "candidate_cost_usd": round(candidate_cost, 4),
        "final_decision_cost_usd": round(final_cost, 4),
        "physical_provider_calls": (
            int(investigation_plan.get("physical_calls") or 0)
            + sum(int(item.get("physical_calls") or 0) for item in candidate_investigations)
            + int(final_decision.get("physical_calls") or 0)
        ),
        "total_turns": total_turns,
        "total_tool_calls": total_tool_calls,
    }


def _inject_preview_placeholders(
    report: SectorAlphaReport,
    *,
    packets: dict[str, TickerSignalPacket],
) -> SectorAlphaReport:
    report.investigation_previews = [
        _preview_from_investigation(
            investigation,
            consensus_rank=int(
                next(
                    (
                        item.get("consensus_rank")
                        for item in report.prior_ranking
                        if item.get("ticker") == investigation.get("ticker")
                    ),
                    0,
                )
            ),
            consensus_score=float(
                next(
                    (
                        item.get("consensus_score")
                        for item in report.prior_ranking
                        if item.get("ticker") == investigation.get("ticker")
                    ),
                    0.0,
                )
            ),
            packet=packets[str(investigation.get("ticker") or "")],
        )
        for investigation in report.candidate_investigations
        if str(investigation.get("ticker") or "") in packets
    ]
    return report


def run_sector_comparison(
    sector: str,
    packets: dict[str, TickerSignalPacket],
    *,
    deterministic: bool = False,
    integrity_scope: FinancialIntegrityScope | None = None,
) -> SectorAlphaReport:
    if integrity_scope is not None:
        integrity_scope = FinancialIntegrityScope(
            context=integrity_scope.context,
            run_as_of_date=integrity_scope.run_as_of_date,
            packets=tuple(packets.values()),
            scenarios=integrity_scope.scenarios,
        )
    total_candidates = len(packets)
    rounds: list[ComparisonRound] = []
    signal_packets = {ticker: packet.to_summary_dict() for ticker, packet in packets.items()}

    all_tickers = list(packets.keys())
    blocked = [
        ticker for ticker in all_tickers if (packets[ticker].gate_verdict or "").upper() == "BLOCK"
    ]
    no_data = [
        ticker
        for ticker in all_tickers
        if ticker not in blocked
        and packets[ticker].dcf_value is None
        and packets[ticker].insurance_value is None
        and packets[ticker].gate_verdict is None
    ]
    excluded = blocked + no_data
    viable = [ticker for ticker in all_tickers if ticker not in excluded]

    if excluded:
        reasons = []
        if blocked:
            reasons.append(f"{len(blocked)} BLOCKED")
        if no_data:
            reasons.append(f"{len(no_data)} without valuation data")
        rounds.append(
            ComparisonRound(
                round_number=0,
                candidates_entering=all_tickers,
                candidates_eliminated=excluded,
                candidates_remaining=viable,
                reasoning=f"Pre-filtered: {', '.join(reasons)}",
                elimination_criteria="BLOCK gate verdict or missing valuation data",
            )
        )

    if not viable:
        return SectorAlphaReport(
            sector=sector,
            total_candidates=total_candidates,
            rounds=rounds,
            winner=None,
            winner_thesis="No viable candidates — all tickers blocked by quality gate.",
            winner_conviction="LOW",
            runner_up=None,
            runner_up_thesis="",
            key_risk="All candidates blocked.",
            falsification_trigger="N/A",
            time_horizon="N/A",
            signal_packets=signal_packets,
            selection_basis="no_viable_candidates",
        )

    if len(viable) == 1:
        winner_ticker = viable[0]
        return SectorAlphaReport(
            sector=sector,
            total_candidates=total_candidates,
            rounds=rounds,
            winner=winner_ticker,
            winner_thesis=f"{winner_ticker} is the only viable candidate after pre-filtering.",
            winner_conviction="LOW",
            runner_up=None,
            runner_up_thesis="",
            key_risk="Single candidate — no comparative analysis possible.",
            falsification_trigger="N/A",
            time_horizon="12 months",
            signal_packets=signal_packets,
            selection_basis="single_candidate",
        )

    viable_packets = {ticker: packets[ticker] for ticker in viable}
    ranking_result = rank_by_consensus(viable_packets)
    ranked = ranking_result.ranked if ranking_result.ranked else ranking_result.ranked_insufficient
    if not ranked:
        return SectorAlphaReport(
            sector=sector,
            total_candidates=total_candidates,
            rounds=rounds,
            winner=None,
            winner_thesis="No tickers with computable discount.",
            winner_conviction="LOW",
            runner_up=None,
            runner_up_thesis="",
            key_risk="N/A",
            falsification_trigger="N/A",
            time_horizon="N/A",
            signal_packets=signal_packets,
            selection_basis="no_discount_candidates",
        )

    prior_ranked = ranked[:PRIOR_SHORTLIST_COUNT]
    eliminated_by_rank = ranked[PRIOR_SHORTLIST_COUNT:]
    rounds.append(
        ComparisonRound(
            round_number=1,
            candidates_entering=viable,
            candidates_eliminated=[entry.ticker for entry in eliminated_by_rank],
            candidates_remaining=[entry.ticker for entry in prior_ranked],
            reasoning="Consensus prior ranking selected the shortlist for LLM investigation.",
            elimination_criteria="Consensus score — prior only, not the final chooser",
        )
    )

    for idx, entry in enumerate(ranked, start=1):
        signal_bucket = signal_packets.get(entry.ticker, {})
        signal_bucket["consensus_rank"] = idx
        signal_bucket["consensus_score"] = entry.consensus_score
        signal_bucket["consensus_method_discounts"] = entry.method_discounts
        signal_bucket["consensus_adjustments"] = entry.adjustments
        signal_bucket["ranking_tier"] = (
            "INSUFFICIENT" if entry in ranking_result.ranked_insufficient else "FULL"
        )
        signal_packets[entry.ticker] = signal_bucket
    for entry in ranking_result.ranked_insufficient:
        signal_bucket = signal_packets.get(entry.ticker, {})
        signal_bucket["ranking_tier"] = "INSUFFICIENT"
        signal_bucket["consensus_score"] = entry.consensus_score
        signal_bucket["consensus_method_discounts"] = entry.method_discounts
        signal_bucket["consensus_adjustments"] = entry.adjustments
        signal_packets[entry.ticker] = signal_bucket

    prior_ranking = [
        _prior_candidate_payload(entry, packets[entry.ticker], consensus_rank=index)
        for index, entry in enumerate(prior_ranked, start=1)
    ]
    pre_investigation_winner = prior_ranked[0].ticker if prior_ranked else None
    pre_investigation_runner_up = prior_ranked[1].ticker if len(prior_ranked) > 1 else None

    if deterministic or not alpha_llm_available():
        winner_entry = prior_ranked[0]
        runner_entry = prior_ranked[1] if len(prior_ranked) > 1 else None
        winner_ticker = winner_entry.ticker
        runner_ticker = runner_entry.ticker if runner_entry else None
        return SectorAlphaReport(
            sector=sector,
            total_candidates=total_candidates,
            rounds=rounds,
            winner=winner_ticker,
            winner_thesis=_deterministic_thesis(winner_entry, packets[winner_ticker]),
            winner_conviction="MODERATE" if winner_entry.consensus_score > 20 else "LOW",
            runner_up=runner_ticker,
            runner_up_thesis=_deterministic_thesis(runner_entry, packets[runner_ticker])
            if runner_entry and runner_ticker
            else "",
            key_risk="Deterministic ranking — no LLM-led shortlist or investigation applied.",
            falsification_trigger="Enable an LLM provider for the full LLM-first alpha workflow.",
            time_horizon="12 months",
            signal_packets=signal_packets,
            selection_basis="consensus",
            prior_ranking=prior_ranking,
        )

    hard_blocked_candidates = [
        {
            "ticker": item["ticker"],
            "consensus_rank": item["consensus_rank"],
            "hard_block_reasons": list(item.get("hard_block_reasons") or []),
        }
        for item in prior_ranking
        if item.get("hard_block_reasons")
    ]

    investigation_cfg = AlphaInvestigationConfig()
    investigation_plan = plan_alpha_investigations(
        sector=sector,
        prior_candidates=prior_ranking,
        hard_blocked_candidates=hard_blocked_candidates,
        max_targets=MAX_INVESTIGATION_TARGETS,
        integrity_scope=integrity_scope,
        max_cost_usd=investigation_cfg.max_cost_usd,
    )
    selected_targets = [
        item
        for item in (investigation_plan.get("selected_targets") or [])
        if isinstance(item, dict) and str(item.get("ticker") or "").strip()
    ]
    selected_tickers = [str(item.get("ticker") or "").upper() for item in selected_targets]
    rounds.append(
        ComparisonRound(
            round_number=len(rounds) + 1,
            candidates_entering=[item["ticker"] for item in prior_ranking],
            candidates_eliminated=[
                item["ticker"] for item in prior_ranking if item["ticker"] not in selected_tickers
            ],
            candidates_remaining=selected_tickers,
            reasoning=str(
                investigation_plan.get("summary")
                or "LLM selected which candidates deserved deeper investigation."
            ),
            elimination_criteria="LLM investigation planning over deterministic shortlist",
        )
    )

    ranking_map = {entry.ticker: entry for entry in ranked}
    candidate_investigations: list[dict[str, Any]] = []
    as_of_date = (
        integrity_scope.run_as_of_date if integrity_scope is not None else default_as_of_date()
    )
    provider_mode = "fallback_tool_bundle"
    for item in selected_targets:
        ticker = str(item.get("ticker") or "").upper()
        if ticker not in packets or ticker not in ranking_map:
            continue
        prior_entry = ranking_map[ticker]
        prior_payload = next(
            (candidate for candidate in prior_ranking if candidate.get("ticker") == ticker), {}
        )
        hard_block_reasons = [
            str(reason)
            for reason in (prior_payload.get("hard_block_reasons") or [])
            if str(reason).strip()
        ]
        ctx = AlphaToolContext(
            sector=sector,
            ticker=ticker,
            packet=packets[ticker],
            as_of_date=as_of_date,
            consensus_rank=int(prior_payload.get("consensus_rank") or 0) or None,
            consensus_score=float(prior_entry.consensus_score),
        )
        investigation = run_candidate_investigation(
            ctx=ctx,
            investigation_request=item,
            hard_block_reasons=hard_block_reasons,
            config=investigation_cfg,
            integrity_scope=integrity_scope,
        )
        provider_mode = str(investigation.get("investigation_mode") or provider_mode)
        investigation["consensus_rank"] = prior_payload.get("consensus_rank")
        investigation["consensus_score"] = prior_entry.consensus_score
        candidate_investigations.append(investigation)
        signal_bucket = signal_packets.get(ticker, {})
        signal_bucket["candidate_investigation"] = {
            "verdict": investigation.get("verdict"),
            "confidence": investigation.get("confidence"),
            "key_findings": investigation.get("key_findings"),
            "hard_block_reasons": investigation.get("hard_block_reasons"),
            "model_validity": investigation.get("model_validity"),
            "selection_blockers": investigation.get("selection_blockers"),
            "confidence_cap_reasons": investigation.get("confidence_cap_reasons"),
            "eligible_for_selection": investigation.get("eligible_for_selection"),
        }
        signal_packets[ticker] = signal_bucket

    final_decision = decide_alpha_winner(
        sector=sector,
        prior_ranking=prior_ranking,
        investigation_plan=investigation_plan,
        candidate_investigations=candidate_investigations,
        hard_blocked_candidates=hard_blocked_candidates,
        integrity_scope=integrity_scope,
        max_cost_usd=investigation_cfg.max_cost_usd,
    )
    tool_budget_summary = _tool_budget_summary(
        investigation_plan=investigation_plan,
        candidate_investigations=candidate_investigations,
        final_decision=final_decision,
        provider_mode=provider_mode,
        per_candidate_cap_usd=investigation_cfg.max_cost_usd,
    )
    winner = final_decision.get("winner")
    runner_up = final_decision.get("runner_up")
    winner_investigation = next(
        (item for item in candidate_investigations if item.get("ticker") == winner),
        {},
    )
    winner_conviction = str(final_decision.get("winner_conviction") or "LOW").upper()
    investigation_confidence = str(winner_investigation.get("confidence") or "").upper()
    if investigation_confidence and _CONFIDENCE_TO_SCORE.get(
        investigation_confidence, 0
    ) < _CONFIDENCE_TO_SCORE.get(winner_conviction, 0):
        winner_conviction = investigation_confidence
    if winner:
        remaining = [winner] + ([runner_up] if runner_up else [])
    else:
        remaining = []
    rounds.append(
        ComparisonRound(
            round_number=len(rounds) + 1,
            candidates_entering=[item.get("ticker") for item in candidate_investigations],
            candidates_eliminated=[
                ticker
                for ticker in [item.get("ticker") for item in candidate_investigations]
                if ticker and ticker not in remaining
            ],
            candidates_remaining=remaining,
            reasoning=str(
                final_decision.get("decision_trace")
                or "LLM made the final alpha choice from investigated candidates."
            ),
            elimination_criteria="LLM final choice from investigated eligible candidates",
        )
    )

    report = SectorAlphaReport(
        sector=sector,
        total_candidates=total_candidates,
        rounds=rounds,
        winner=winner,
        winner_thesis=str(final_decision.get("winner_thesis") or ""),
        winner_conviction=winner_conviction,
        runner_up=runner_up,
        runner_up_thesis=str(final_decision.get("runner_up_thesis") or ""),
        key_risk=str(final_decision.get("key_risk") or ""),
        falsification_trigger=str(final_decision.get("falsification_trigger") or ""),
        time_horizon=str(final_decision.get("time_horizon") or "12 months"),
        signal_packets=signal_packets,
        pre_investigation_winner=pre_investigation_winner,
        pre_investigation_runner_up=pre_investigation_runner_up,
        selection_basis="llm_decision" if winner else "llm_no_winner",
        prior_ranking=prior_ranking,
        investigation_plan=investigation_plan,
        candidate_investigations=candidate_investigations,
        hard_blocked_candidates=hard_blocked_candidates,
        llm_decision_trace={
            "decision_trace": final_decision.get("decision_trace"),
            "decision_mode": final_decision.get("decision_mode"),
            "rejected_candidates": final_decision.get("rejected_candidates") or [],
            "selected_winner": winner,
            "selected_runner_up": runner_up,
            "eligible_candidates": [
                item.get("ticker")
                for item in candidate_investigations
                if item.get("eligible_for_selection") is True
            ],
            "selection_blocked_candidates": [
                {
                    "ticker": item.get("ticker"),
                    "verdict": item.get("verdict"),
                    "model_validity": item.get("model_validity"),
                    "selection_blockers": item.get("selection_blockers")
                    or item.get("hard_block_reasons")
                    or [],
                    "confidence_cap_reasons": item.get("confidence_cap_reasons") or [],
                }
                for item in candidate_investigations
                if item.get("eligible_for_selection") is not True
            ],
        },
        tool_budget_summary=tool_budget_summary,
    )
    report = _inject_preview_placeholders(report, packets=packets)
    if report.winner is None:
        report.winner_thesis = report.winner_thesis or (
            "The LLM investigation did not identify an eligible candidate with enough evidence-backed upside."
        )
    return report
