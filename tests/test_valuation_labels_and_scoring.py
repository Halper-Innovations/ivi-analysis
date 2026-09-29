"""Scores and labels must say what the numbers are.

Residual cases that no other test covered. Each docstring derives the correct answer by hand; none is copied from
implementation output.
"""

from __future__ import annotations

from app.score.rubric import score_packet
from app.valuation.anchor_policy import select_anchor
from app.valuation.expectations_gap import compute_expectations_gap


# ── Rubric: adverse accounting events are not points ──────────────────────────


def test_restatement_auditor_change_and_material_weakness_do_not_add_points():
    """The same empty packet with and without three adverse accounting events.

    A restatement, an auditor change and a material weakness are evidence against
    the filings' reliability. Policy call (2026-09-29, conservative): they earn no
    points (special_situations 0.0) rather than +7.5; penalising them is left to the
    change-momentum penalty that already exists for new negative flags.
    """
    base = {"fundamentals": {}, "valuations": {}}
    adverse = {
        **base,
        "extracted_facts": [
            {"fact_type": "restatement"},
            {"fact_type": "auditor_change"},
            {"fact_type": "material_weakness"},
        ],
    }
    sub_base, total_base, _, _ = score_packet(base)
    sub_adverse, total_adverse, _, _ = score_packet(adverse)
    assert sub_adverse["special_situations"] == 0.0
    assert total_adverse == total_base


# ── Expectations gap: the saturated-low gap is an upper bound ──────────────────


def test_saturated_low_gap_is_labelled_an_upper_bound_not_a_floor():
    """Implied growth saturates at the -25% bound with a positive margin.

    The true implied growth is at or below -25%, so the gap (implied - supportable)
    is at or below -0.25 - 0.05 = -0.30: -0.30 is an UPPER bound. The payload must
    say so and must not call it a floor.
    """
    gap = compute_expectations_gap(
        -0.25,
        0.05,
        True,
        saturated_bound="LOW",
        margin_sign="POSITIVE",
    )
    assert gap["gap"] == -0.3
    assert gap["gap_is_upper_bound"] is True
    assert "gap_is_floor" not in gap


# ── Anchor: a provenance lens cannot take over an anchor-eligible method's name ─


def test_an_extra_candidate_named_like_a_core_method_cannot_replace_it():
    """DCF 50 and a provenance lens that arrives under the key "dcf" with 500.

    Lenses are provenance only and never anchor. The anchor stays the DCF's 50 and
    the candidates table keeps 50 for dcf.
    """
    selection = select_anchor(dcf=50.0, extra_candidates={"dcf": 500.0})
    assert selection.value == 50.0
    assert selection.candidates["dcf"] == 50.0
