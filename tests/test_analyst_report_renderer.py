"""Decision-first header in the analyst memo (render_report + render_summary).

The analyst renderer lives in :mod:`app.analyst.report_renderer` (distinct from
the research diagnostic renderer in :mod:`app.research.report_renderer`). These
tests assert the verdict-to-action mapping for the analyst
memo: BUY -> "Buy now", WATCH -> "Wait", STAY_AWAY -> "Pass".
"""
from __future__ import annotations

from app.analyst.report_renderer import render_report, render_summary
from app.analyst.thesis_contract import (
    AnalysisReport,
    Falsifier,
    ValuationConclusion,
)


def _make_report(**overrides) -> AnalysisReport:
    defaults = dict(
        analysis_id="a1",
        ticker="TEST",
        as_of_date="2026-04-17",
        generated_at="2026-04-17T12:00:00+00:00",
        verdict="BUY",
        research_gate="PROCEED",
        confidence_label="MODERATE",
        confidence_score=60,
        thesis_summary="A representative thesis summary.",
        valuation=ValuationConclusion(
            price=18.33,
            base_case_value=42.0,
            bear_case_value=20.0,
            bull_case_value=60.0,
            margin_of_safety=0.4,
        ),
    )
    defaults.update(overrides)
    return AnalysisReport(**defaults)


def _section_index(markdown: str, header: str) -> int:
    return markdown.index(header)


class TestDecisionHeaderInReport:
    def test_decision_block_appears_before_thesis(self):
        md = render_report(_make_report(verdict="BUY"))
        assert "## Decision" in md
        assert _section_index(md, "## Decision") < _section_index(md, "## Thesis")

    def test_buy_verdict_renders_buy_now_action(self):
        md = render_report(_make_report(verdict="BUY"))
        assert "- ACTION: Buy now" in md
        assert "**Research Gate:** PROCEED (pipeline provenance)" in md

    def test_stay_away_verdict_renders_pass_action(self):
        md = render_report(_make_report(verdict="STAY_AWAY"))
        assert "- ACTION: Pass" in md

    def test_falsifier_renders_what_would_change_my_mind(self):
        report = _make_report(
            verdict="BUY",
            falsifiers=[
                Falsifier(
                    description="Free cash flow turns negative for two consecutive years",
                    trigger_type="FUNDAMENTAL",
                )
            ],
        )
        md = render_report(report)
        assert (
            "- WHAT WOULD CHANGE MY MIND: "
            "Free cash flow turns negative for two consecutive years"
        ) in md


class TestDecisionHeaderInSummary:
    def test_watch_verdict_summary_second_line_is_action_wait(self):
        summary = render_summary(_make_report(verdict="WATCH"))
        lines = summary.split("\n")
        assert lines[1] == "Action: Wait"
