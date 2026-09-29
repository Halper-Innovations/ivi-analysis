import pytest
from pydantic import ValidationError

from app.agent.analyst_agent import Hypothesis, ResearchPlanStep


def test_hypothesis_schema_accepts_valid_payload():
    h = Hypothesis(
        hypothesis_id="H1",
        direction="LONG",
        claim="Example claim",
        valuation_anchor="DCF range from packet",
        confidence="LOW",
        citations=[{"source_url": "https://www.sec.gov/example", "snippet": "sample"}],
    )
    assert h.direction == "LONG"


def test_hypothesis_schema_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="H1",
            direction="LONG",
            claim="Example claim",
            valuation_anchor="anchor",
            confidence="CERTAIN",
            citations=[],
        )


def test_research_plan_step_enforces_edgar_only_source():
    step = ResearchPlanStep(
        step_id="R1",
        action="Pull latest 10-Q",
        rationale="Check trend",
        expected_artifact="parsed filing",
        section_targets=["MD&A"],
        keywords=["revenue"],
        disconfirmation_check="Reject if revenue drops.",
        metric_ties=["revenue"],
        allowed_source="EDGAR",
    )
    assert step.allowed_source == "EDGAR"
