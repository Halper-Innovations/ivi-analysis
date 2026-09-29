from __future__ import annotations

import pytest

from app.synthesis.schemas import SignalEvidence, VariantPerception, VariantPerceptionReport


def test_variant_perception_schema_accepts_valid_payload() -> None:
    signal = SignalEvidence(
        source="VALUATION",
        signal_type="DCF_DISCOUNT",
        direction="SUPPORTS_UNDERVALUED",
        strength="HIGH",
        summary="DCF indicates a material discount to price.",
        derived_from=["valuations.dcf.outputs.base"],
    )
    perception = VariantPerception(
        perception_id="ABC_2026-03-22_undervalued_1",
        ticker="ABC",
        as_of_date="2026-03-22",
        thesis="The market is underestimating the company's growth durability.",
        direction="UNDERVALUED",
        confidence="MEDIUM",
        implied_vs_estimated={
            "market_implied_growth": 0.10,
            "estimated_fair_growth": 0.14,
            "gap_pct": 0.40,
        },
        supporting_signals=[signal],
        contradicting_signals=[],
        testable_prediction="Revenue growth should accelerate within the next two annual filings.",
        time_horizon="MEDIUM",
        catalyst="A filing showing stronger backlog conversion.",
        risk="Growth could stall and invalidate the thesis.",
        derived_from=["valuations.dcf.outputs.base"],
        generated_at="2026-03-23T00:00:00+00:00",
    )
    report = VariantPerceptionReport(
        run_id="run_1",
        ticker="ABC",
        as_of_date="2026-03-22",
        perceptions=[perception],
        signal_summary={"total_signals": 1},
        data_quality={"available_sources": ["VALUATION"]},
    )

    assert report.perceptions[0].direction == "UNDERVALUED"


def test_variant_perception_schema_rejects_missing_supporting_signals() -> None:
    with pytest.raises(ValueError):
        VariantPerception(
            perception_id="ABC_2026-03-22_undervalued_1",
            ticker="ABC",
            as_of_date="2026-03-22",
            thesis="Unsupported thesis.",
            direction="UNDERVALUED",
            confidence="LOW",
            implied_vs_estimated={
                "market_implied_growth": 0.10,
                "estimated_fair_growth": 0.11,
                "gap_pct": 0.10,
            },
            supporting_signals=[],
            contradicting_signals=[],
            testable_prediction="Prediction.",
            time_horizon="SHORT",
            catalyst="Catalyst.",
            risk="Risk.",
            derived_from=[],
            generated_at="2026-03-23T00:00:00+00:00",
        )


def test_variant_perception_schema_rejects_incomplete_growth_gap() -> None:
    signal = SignalEvidence(
        source="PATTERN",
        signal_type="deferred_revenue_leading_indicator",
        direction="SUPPORTS_UNDERVALUED",
        strength="MEDIUM",
        summary="Pattern support.",
        derived_from=["pattern.ref"],
    )
    with pytest.raises(ValueError):
        VariantPerception(
            perception_id="ABC_2026-03-22_undervalued_1",
            ticker="ABC",
            as_of_date="2026-03-22",
            thesis="Incomplete growth gap.",
            direction="UNDERVALUED",
            confidence="LOW",
            implied_vs_estimated={"market_implied_growth": 0.10},
            supporting_signals=[signal],
            contradicting_signals=[],
            testable_prediction="Prediction.",
            time_horizon="SHORT",
            catalyst="Catalyst.",
            risk="Risk.",
            derived_from=[],
            generated_at="2026-03-23T00:00:00+00:00",
        )
