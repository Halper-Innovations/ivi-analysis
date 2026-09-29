"""Tests for app.alpha.sector_comparator."""

from __future__ import annotations

from app.alpha.schemas import SectorAlphaReport, TickerSignalPacket
from app.alpha.sector_comparator import _tool_budget_summary


def _make_packet(
    ticker: str,
    dcf: float,
    *,
    gate: str = "PROCEED",
    moat: int = 3,
    price: float = 100.0,
    solvency_risk: str | None = None,
    going_concern: bool = False,
) -> TickerSignalPacket:
    return TickerSignalPacket(
        ticker=ticker,
        dcf_value=dcf,
        epv_value=dcf * 0.8,
        current_price=price,
        gate_verdict=gate,
        confidence_class="HIGH",
        moat_score=moat,
        moat_classification="MODERATE_MOAT",
        margin_of_safety_verdict="UNDERVALUED" if dcf > price else "OVERVALUED",
        research_report={
            "anomalies": [],
            "solvency": {
                "risk": solvency_risk or "LOW",
                "signals": [],
                "details": "",
                "going_concern_language": going_concern,
                # A going-concern hard block needs the filed, blockable assertion behind
                # the flag: the bare flag alone no longer counts as distress.
                "going_concern_assertions": (
                    [
                        {
                            "subject": "REGISTRANT",
                            "assertion_mode": "AFFIRMATIVE_CURRENT",
                            "blockable": True,
                            "excerpt": "There is substantial doubt about our ability to "
                            "continue as a going concern.",
                        }
                    ]
                    if going_concern
                    else []
                ),
                "no_assurance_financing": False,
            },
        },
        solvency_risk=solvency_risk,
    )


def test_tool_budget_summary_includes_planner_candidate_and_final_costs():
    summary = _tool_budget_summary(
        investigation_plan={"cost_usd": 0.11, "physical_calls": 1, "budget_usd": 1.25},
        candidate_investigations=[
            {"cost_usd": 0.22, "physical_calls": 1, "num_turns": 1},
            {"cost_usd": 0.33, "physical_calls": 2, "num_turns": 2},
        ],
        final_decision={"cost_usd": 0.44, "physical_calls": 1, "budget_usd": 1.25},
        provider_mode="openai",
        per_candidate_cap_usd=1.25,
    )

    assert summary["planner_cost_usd"] == 0.11
    assert summary["candidate_cost_usd"] == 0.55
    assert summary["final_decision_cost_usd"] == 0.44
    assert summary["total_cost_usd"] == 1.1
    assert summary["physical_provider_calls"] == 5
    assert summary["budget_scope"] == "STAGE_LOCAL_REQUEST_CAPS"
    assert summary["whole_comparison_cap_usd"] is None
    assert summary["planner_request_cap_usd"] == 1.25
    assert summary["final_decision_request_cap_usd"] == 1.25


def test_comparator_produces_llm_first_report(monkeypatch):
    packets = {
        "CRM": _make_packet("CRM", dcf=400.0, price=250.0, moat=3),
        "MSFT": _make_packet("MSFT", dcf=500.0, price=400.0, moat=5),
        "ORCL": _make_packet("ORCL", dcf=200.0, price=180.0, moat=2),
    }

    monkeypatch.setattr("app.alpha.sector_comparator.alpha_llm_available", lambda: True)
    monkeypatch.setattr(
        "app.alpha.sector_comparator.plan_alpha_investigations",
        lambda **kwargs: {
            "summary": "Investigate the two most interesting names only.",
            "selected_targets": [
                {
                    "ticker": "CRM",
                    "reason": "Highest prior upside but method tension needs checking.",
                    "evidence_gaps": ["Need dilution check."],
                },
                {
                    "ticker": "MSFT",
                    "reason": "Stronger quality could overcome lower prior discount.",
                    "evidence_gaps": ["Need peer and liquidity confirmation."],
                },
            ],
            "skipped_candidates": [{"ticker": "ORCL", "reason": "Lower upside."}],
            "planner_mode": "llm",
        },
    )
    investigations = {
        "CRM": {
            "ticker": "CRM",
            "verdict": "WATCH",
            "confidence": "MODERATE",
            "key_findings": ["Valuation upside is real but dilution still needs monitoring."],
            "open_questions": ["Need another filing cycle for margin durability."],
            "key_risk": "Competitive pressure.",
            "falsification_trigger": "If margins compress again.",
            "reasoning_trace": "CRM has upside, but evidence is not as complete as MSFT.",
            "tool_call_counts": {"analyze_dilution": 1},
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
                "reason": "Need dilution check.",
                "evidence_gaps": ["Need dilution check."],
            },
        },
        "MSFT": {
            "ticker": "MSFT",
            "verdict": "PROCEED",
            "confidence": "HIGH",
            "key_findings": [
                "Quality and peer position justify choosing MSFT despite lower prior discount."
            ],
            "open_questions": [],
            "key_risk": "Cloud growth deceleration.",
            "falsification_trigger": "If growth falls below sector median for two years.",
            "reasoning_trace": "MSFT became the best risk-adjusted candidate after deeper evidence review.",
            "tool_call_counts": {"compare_peer_metric": 2},
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
                "reason": "Need peer and liquidity confirmation.",
                "evidence_gaps": ["Need peer and liquidity confirmation."],
            },
        },
    }
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_candidate_investigation",
        lambda **kwargs: investigations[kwargs["ctx"].ticker],
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.decide_alpha_winner",
        lambda **kwargs: {
            "winner": "MSFT",
            "runner_up": "CRM",
            "winner_thesis": "MSFT best combines quality with sufficient upside.",
            "winner_conviction": "HIGH",
            "runner_up_thesis": "CRM is interesting but more fragile.",
            "key_risk": "Cloud growth deceleration.",
            "falsification_trigger": "If growth falls below sector median for two years.",
            "time_horizon": "12 months",
            "decision_trace": "MSFT won because the evidence closed more favorably after investigation.",
            "rejected_candidates": [
                {"ticker": "ORCL", "reason": "Lower upside and weaker evidence."}
            ],
            "decision_mode": "llm",
        },
    )

    from app.alpha.sector_comparator import run_sector_comparison

    report = run_sector_comparison("enterprise_software", packets)

    assert isinstance(report, SectorAlphaReport)
    assert report.pre_investigation_winner == "CRM"
    assert report.winner == "MSFT"
    assert report.runner_up == "CRM"
    assert report.selection_basis == "llm_decision"
    assert len(report.prior_ranking) == 3
    assert len(report.candidate_investigations) == 2
    assert report.investigation_plan["selected_targets"][0]["ticker"] == "CRM"
    assert report.llm_decision_trace["decision_mode"] == "llm"
    assert report.tool_budget_summary["investigated_candidates"] == 2
    assert report.investigation_previews[0]["ticker"] in {"CRM", "MSFT"}


def test_hard_blocked_candidates_visible_but_ineligible(monkeypatch):
    packets = {
        "SAFE": _make_packet("SAFE", dcf=300.0, price=200.0),
        "DIST": _make_packet(
            "DIST", dcf=320.0, price=150.0, solvency_risk="CRITICAL", going_concern=True
        ),
    }

    monkeypatch.setattr("app.alpha.sector_comparator.alpha_llm_available", lambda: True)
    monkeypatch.setattr(
        "app.alpha.sector_comparator.plan_alpha_investigations",
        lambda **kwargs: {
            "summary": "Investigate both names because the distressed candidate still affects relative judgment.",
            "selected_targets": [
                {
                    "ticker": "DIST",
                    "reason": "High upside but distress risk.",
                    "evidence_gaps": ["Need liquidity check."],
                },
                {
                    "ticker": "SAFE",
                    "reason": "Baseline quality candidate.",
                    "evidence_gaps": ["Need peer context."],
                },
            ],
            "skipped_candidates": [],
            "planner_mode": "llm",
        },
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_candidate_investigation",
        lambda **kwargs: {
            "ticker": kwargs["ctx"].ticker,
            "verdict": "AVOID" if kwargs["ctx"].ticker == "DIST" else "PROCEED",
            "confidence": "HIGH" if kwargs["ctx"].ticker == "SAFE" else "MODERATE",
            "key_findings": ["Finding"],
            "open_questions": [],
            "key_risk": "Risk",
            "falsification_trigger": "Trigger",
            "reasoning_trace": "Trace",
            "tool_call_counts": {},
            "tool_transcript": [],
            "num_turns": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "termination_reason": "fallback_summary",
            "error": None,
            "investigation_mode": "fallback_tool_bundle",
            "hard_block_reasons": list(kwargs["hard_block_reasons"]),
            "eligible_for_selection": not kwargs["hard_block_reasons"],
            "requested_focus": {"reason": "reason", "evidence_gaps": []},
        },
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.decide_alpha_winner",
        lambda **kwargs: {
            "winner": "SAFE",
            "runner_up": None,
            "winner_thesis": "SAFE is the only eligible winner.",
            "winner_conviction": "HIGH",
            "runner_up_thesis": "",
            "key_risk": "Risk",
            "falsification_trigger": "Trigger",
            "time_horizon": "12 months",
            "decision_trace": "DIST stayed visible but ineligible due to hard blocks.",
            "rejected_candidates": [{"ticker": "DIST", "reason": "Hard-blocked by distress."}],
            "decision_mode": "llm",
        },
    )

    from app.alpha.sector_comparator import run_sector_comparison

    report = run_sector_comparison("enterprise_software", packets)

    assert report.winner == "SAFE"
    assert report.hard_blocked_candidates == [
        {
            "ticker": "DIST",
            "consensus_rank": 2,
            "hard_block_reasons": ["solvency_critical", "explicit_distress"],
        }
    ]
    assert report.candidate_investigations[0]["ticker"] == "DIST"
    assert report.candidate_investigations[0]["eligible_for_selection"] is False
    assert report.llm_decision_trace["eligible_candidates"] == ["SAFE"]


def test_llm_no_winner_outcome(monkeypatch):
    packets = {
        "AAA": _make_packet("AAA", dcf=180.0, price=120.0),
        "BBB": _make_packet("BBB", dcf=170.0, price=115.0),
    }

    monkeypatch.setattr("app.alpha.sector_comparator.alpha_llm_available", lambda: True)
    monkeypatch.setattr(
        "app.alpha.sector_comparator.plan_alpha_investigations",
        lambda **kwargs: {
            "summary": "Investigate both names because neither prior is decisive.",
            "selected_targets": [
                {
                    "ticker": "AAA",
                    "reason": "Need more work.",
                    "evidence_gaps": ["Need liquidity check."],
                },
                {
                    "ticker": "BBB",
                    "reason": "Need more work.",
                    "evidence_gaps": ["Need peer check."],
                },
            ],
            "skipped_candidates": [],
            "planner_mode": "llm",
        },
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_candidate_investigation",
        lambda **kwargs: {
            "ticker": kwargs["ctx"].ticker,
            "verdict": "WATCH",
            "confidence": "LOW",
            "key_findings": ["Evidence stayed mixed."],
            "open_questions": ["Still unresolved."],
            "key_risk": "Risk",
            "falsification_trigger": "Trigger",
            "reasoning_trace": "Trace",
            "tool_call_counts": {},
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
            "requested_focus": {"reason": "reason", "evidence_gaps": []},
        },
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.decide_alpha_winner",
        lambda **kwargs: {
            "winner": None,
            "runner_up": None,
            "winner_thesis": "",
            "winner_conviction": "LOW",
            "runner_up_thesis": "",
            "key_risk": "No name cleared the bar.",
            "falsification_trigger": "Re-run when evidence changes.",
            "time_horizon": "N/A",
            "decision_trace": "No winner because both names stayed too unresolved.",
            "rejected_candidates": [
                {"ticker": "AAA", "reason": "Too unresolved."},
                {"ticker": "BBB", "reason": "Too unresolved."},
            ],
            "decision_mode": "llm",
        },
    )

    from app.alpha.sector_comparator import run_sector_comparison

    report = run_sector_comparison("enterprise_software", packets)

    assert report.winner is None
    assert report.selection_basis == "llm_no_winner"
    assert "did not identify an eligible candidate" in report.winner_thesis.lower()


def test_deterministic_mode_skips_llm(monkeypatch):
    packets = {
        "CRM": _make_packet("CRM", dcf=400.0, price=250.0, moat=3),
        "MSFT": _make_packet("MSFT", dcf=500.0, price=400.0, moat=5),
    }

    monkeypatch.setattr("app.alpha.sector_comparator.alpha_llm_available", lambda: True)
    called = {"planner": False}

    def _mark_called(**kwargs):
        called["planner"] = True
        return {}

    monkeypatch.setattr("app.alpha.sector_comparator.plan_alpha_investigations", _mark_called)

    from app.alpha.sector_comparator import run_sector_comparison

    report = run_sector_comparison("enterprise_software", packets, deterministic=True)

    assert called["planner"] is False
    assert report.winner == "CRM"
    assert report.runner_up == "MSFT"
    assert report.selection_basis == "consensus"


def test_insufficient_tickers_tagged(monkeypatch):
    packets = {
        "FULL": _make_packet("FULL", dcf=150.0, price=100.0, gate="PROCEED"),
        "SPARSE": TickerSignalPacket(ticker="SPARSE", dcf_value=200.0, current_price=80.0),
    }

    monkeypatch.setattr("app.alpha.sector_comparator.alpha_llm_available", lambda: False)

    from app.alpha.sector_comparator import run_sector_comparison

    report = run_sector_comparison("test", packets)

    assert report.winner == "FULL"
    assert report.signal_packets["SPARSE"]["ranking_tier"] == "INSUFFICIENT"
