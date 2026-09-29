"""Going-concern and material-weakness flags need an assertion, not a keyword.

Both false positives below were raised for a healthy large-cap issuer: the
auditor's standard ICFR paragraph and a climate-risk sentence.
"""

from __future__ import annotations

import pytest

from app.parse.extractors.footnotes_signals import (
    asserted_risk_excerpt,
    extract_footnote_signals,
)
from app.research.schemas import EvidenceItem
from app.research.signals import compute_research_signals

AUDITOR_BOILERPLATE = (
    "Our audit included obtaining an understanding of internal control over financial "
    "reporting, assessing the risk that a material weakness exists, testing and evaluating "
    "the design and operating effectiveness of internal control based on the assessed risk."
)
CLIMATE_SENTENCE = (
    "In addition, ongoing concern over climate change is expected to continue to result in "
    "additional legal or regulatory requirements."
)


def _fact_types(text: str) -> set[str]:
    return {fact["fact_type"] for fact in extract_footnote_signals(text, "https://www.sec.gov/x")}


def test_boilerplate_and_substring_hits_raise_no_flag():
    assert _fact_types(f"{AUDITOR_BOILERPLATE} {CLIMATE_SENTENCE}") == set()


@pytest.mark.parametrize(
    "text",
    [
        "We did not identify any material weaknesses in our internal control over financial reporting.",
        "There were no material weaknesses as of December 31, 2025.",
        "If we fail to maintain effective internal controls, we could identify a material weakness.",
        "The material weakness previously reported has been fully remediated.",
        "A material weakness is a deficiency, or a combination of deficiencies, in internal control.",
    ],
)
def test_non_assertions_are_not_material_weakness_findings(text):
    assert asserted_risk_excerpt("material_weakness", text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Management identified a material weakness in our internal control over financial reporting.",
        "A material weakness was identified related to revenue cut-off controls.",
        "Our internal control over financial reporting was not effective because of the material "
        "weakness described above.",
        "The material weakness has not yet been remediated.",
    ],
)
def test_affirmative_material_weakness_is_flagged(text):
    assert asserted_risk_excerpt("material_weakness", text) is not None
    assert "material_weakness" in _fact_types(text)


def test_going_concern_uses_the_attributed_detector():
    negated = "Management believes substantial doubt about going concern does not exist."
    hypothetical = "If we cannot raise capital, there could be substantial doubt about our ability to continue as a going concern."
    asserted = "There is substantial doubt about our ability to continue as a going concern."

    assert asserted_risk_excerpt("going_concern", CLIMATE_SENTENCE) is None
    assert asserted_risk_excerpt("going_concern", negated) is None
    assert asserted_risk_excerpt("going_concern", hypothetical) is None
    assert asserted_risk_excerpt("going_concern", asserted) == asserted
    assert _fact_types(asserted) == {"going_concern"}


def _item(excerpt: str) -> EvidenceItem:
    return EvidenceItem(
        id="ev-1",
        ticker="KO",
        as_of_date="2026-09-25",
        hash="h1",
        source_type="EDGAR",
        source_title="10-K",
        source_url="https://www.sec.gov/x",
        source_published_at="2026-02-20T00:00:00+00:00",
        retrieved_at="2026-02-20T00:00:00+00:00",
        excerpt_text=excerpt,
    )


def test_research_signals_apply_the_same_rule():
    quiet = compute_research_signals(
        ticker="KO",
        as_of_date="2026-09-25",
        run_id="run",
        evidence_items=[_item(f"{AUDITOR_BOILERPLATE} {CLIMATE_SENTENCE}")],
    )
    loud = compute_research_signals(
        ticker="KO",
        as_of_date="2026-09-25",
        run_id="run",
        evidence_items=[_item("Management identified a material weakness in revenue controls.")],
    )

    assert quiet.sentiment_flags == []
    assert quiet.summary["risk_flags_present"] is False
    assert loud.sentiment_flags == ["material_weakness"]
