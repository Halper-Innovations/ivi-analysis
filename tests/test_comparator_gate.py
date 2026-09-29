"""Gate-focused tests for the LLM-first sector comparator."""
from __future__ import annotations

from app.alpha.schemas import TickerSignalPacket


def _packet(
    ticker: str,
    *,
    dcf: float,
    price: float = 100.0,
    gate: str = "PROCEED",
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
        moat_score=4,
        moat_classification="MODERATE_MOAT",
        research_report={
            "anomalies": [],
            "solvency": {
                "risk": solvency_risk or "LOW",
                "signals": [],
                "details": "",
                "going_concern_language": going_concern,
                # A going-concern hard block needs the filed, blockable assertion behind the
                # flag; the bare flag alone no longer counts as explicit distress.
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


def test_explicit_distress_candidate_stays_visible_but_hard_blocked(monkeypatch):
    packets = {
        "SAFE": _packet("SAFE", dcf=220.0, price=150.0),
        "DIST": _packet("DIST", dcf=260.0, price=120.0, solvency_risk="CRITICAL", going_concern=True),
    }

    monkeypatch.setattr("app.alpha.sector_comparator.alpha_llm_available", lambda: True)
    monkeypatch.setattr(
        "app.alpha.sector_comparator.plan_alpha_investigations",
        lambda **kwargs: {
            "summary": "Investigate both names.",
            "selected_targets": [
                {"ticker": "DIST", "reason": "Stress case.", "evidence_gaps": ["Need liquidity detail."]},
                {"ticker": "SAFE", "reason": "Baseline case.", "evidence_gaps": ["Need KPI check."]},
            ],
            "skipped_candidates": [],
        },
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_candidate_investigation",
        lambda **kwargs: {
            "ticker": kwargs["ctx"].ticker,
            "verdict": "AVOID" if kwargs["ctx"].ticker == "DIST" else "PROCEED",
            "confidence": "MODERATE",
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
            "requested_focus": {"reason": "x", "evidence_gaps": []},
        },
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.decide_alpha_winner",
        lambda **kwargs: {
            "winner": "SAFE",
            "runner_up": None,
            "winner_thesis": "SAFE is the only eligible winner.",
            "winner_conviction": "MODERATE",
            "runner_up_thesis": "",
            "key_risk": "Risk",
            "falsification_trigger": "Trigger",
            "time_horizon": "12 months",
            "decision_trace": "Hard-blocked distressed candidate stayed visible but could not win.",
            "rejected_candidates": [{"ticker": "DIST", "reason": "Hard-blocked."}],
            "decision_mode": "llm",
        },
    )

    from app.alpha.sector_comparator import run_sector_comparison

    report = run_sector_comparison("enterprise_software", packets)

    assert report.winner == "SAFE"
    assert report.hard_blocked_candidates[0]["ticker"] == "DIST"
    assert report.hard_blocked_candidates[0]["hard_block_reasons"] == [
        "solvency_critical",
        "explicit_distress",
    ]


def test_sparse_candidate_remains_tagged_insufficient_in_consensus_fallback(monkeypatch):
    packets = {
        "FULL": _packet("FULL", dcf=160.0, price=100.0),
        "SPARSE": TickerSignalPacket(ticker="SPARSE", dcf_value=220.0, current_price=80.0),
    }

    monkeypatch.setattr("app.alpha.sector_comparator.alpha_llm_available", lambda: False)

    from app.alpha.sector_comparator import run_sector_comparison

    report = run_sector_comparison("enterprise_software", packets)

    assert report.winner == "FULL"
    assert report.signal_packets["SPARSE"]["ranking_tier"] == "INSUFFICIENT"
