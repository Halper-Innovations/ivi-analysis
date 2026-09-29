from __future__ import annotations

from app.diff.schemas import FilingDiffReport
from app.patterns.schemas import PatternHit, PatternResult, PatternScanReport
from app.synthesis.schemas import SignalEvidence, VariantPerception
from app.synthesis.variant_builder import _adjust_perception_confidence, build_variant_perceptions


def _medium_perception() -> VariantPerception:
    return VariantPerception(
        perception_id="p1",
        ticker="AAA",
        as_of_date="2026-03-22",
        thesis="Test",
        direction="UNDERVALUED",
        confidence="MEDIUM",
        implied_vs_estimated={"market_implied_growth": 0.1, "estimated_fair_growth": 0.2, "gap_pct": 0.1},
        supporting_signals=[
            SignalEvidence(
                source="PATTERN",
                signal_type="pattern_x",
                direction="SUPPORTS_UNDERVALUED",
                strength="HIGH",
                summary="x",
                derived_from=[],
            )
        ],
        contradicting_signals=[],
        testable_prediction="x",
        time_horizon="SHORT",
        catalyst="x",
        risk="x",
        derived_from=[],
        generated_at="2026-03-22T10:00:00+00:00",
    )


def _weights(*, hit_rate: float, sample_size: int) -> dict:
    return {
        "pattern_weights": {
            "pattern_x": {
                "pattern_id": "pattern_x",
                "hit_rate": hit_rate,
                "sample_size": sample_size,
                "last_calibrated": "2027-03-22T00:00:00+00:00",
            }
        }
    }


# FIX 3: confidence must NOT be upgraded/downgraded on a thin decisive sample.
# A high hit_rate backed by < 3 decisive resolutions must leave MEDIUM unchanged.
def test_confidence_not_upgraded_on_thin_sample() -> None:
    perception = _medium_perception()
    adjusted = _adjust_perception_confidence(perception, _weights(hit_rate=1.0, sample_size=2))
    assert adjusted.confidence == "MEDIUM"


def test_confidence_not_downgraded_on_thin_sample() -> None:
    perception = _medium_perception()
    adjusted = _adjust_perception_confidence(perception, _weights(hit_rate=0.0, sample_size=2))
    assert adjusted.confidence == "MEDIUM"


def test_confidence_upgraded_with_sufficient_decisive_sample() -> None:
    perception = _medium_perception()
    adjusted = _adjust_perception_confidence(perception, _weights(hit_rate=0.8, sample_size=3))
    assert adjusted.confidence == "HIGH"


def test_confidence_downgraded_with_sufficient_decisive_sample() -> None:
    perception = _medium_perception()
    adjusted = _adjust_perception_confidence(perception, _weights(hit_rate=0.2, sample_size=3))
    assert adjusted.confidence == "LOW"


def _valuation_payload(
    *,
    price: float = 100.0,
    implied_growth: float = 0.10,
    dcf_base: float = 160.0,
    scorecard_signal: str = "FAIRLY_VALUED",
    gaap_stance: str = "WATCH",
    adjusted_stance: str = "WATCH",
    tech_divergence: float | None = None,
) -> dict:
    tech_outputs = {}
    if tech_divergence is not None:
        tech_outputs = {
            "tech_valuation_divergence": tech_divergence,
            "category_classification": {"category": "ENTERPRISE_SOFTWARE", "derived_from": ["category.ref"]},
            "rnd_adjustment": {"derived_from": ["rnd.ref"]},
        }
    return {
        "dcf": {"inputs": {}, "outputs": {"base": dcf_base}},
        "scorecard": {
            "inputs": {},
            "outputs": {
                "signal": scorecard_signal,
                "type": "EARNINGS_DRIVEN",
                "track_comparison": {
                    "gaap_stance": gaap_stance,
                    "adjusted_stance": adjusted_stance,
                    "stance_differs": gaap_stance != adjusted_stance,
                },
            },
        },
        "reverse_dcf": {
            "inputs": {"price": price},
            "outputs": {"outputs": {"implied_growth": implied_growth}},
        },
        "tech_adjustment": {"inputs": {}, "outputs": tech_outputs},
    }


def _pattern_report(*, ticker: str, pattern_id: str, hit_rate: float = 0.67, sample_size: int = 12, outcome_details: str = "Revenue growth accelerated after deferred revenue outpaced recognized revenue.") -> PatternScanReport:
    return PatternScanReport(
        run_id="run_1",
        scan_date="2026-03-23T00:00:00+00:00",
        peer_set_size=12,
        peer_set_tickers=[ticker],
        pattern_results=[
            PatternResult(
                pattern_id=pattern_id,
                hypothesis="Pattern hypothesis.",
                hit_count=1,
                confirmed_count=8 if sample_size else 0,
                unconfirmed_count=0,
                hit_rate=hit_rate,
                sample_size=sample_size,
                hits=[
                    PatternHit(
                        ticker=ticker,
                        pattern_id=pattern_id,
                        years_detected=[2023, 2024],
                        detection_strength=0.8,
                        outcome_confirmed=True,
                        outcome_value=0.05,
                        outcome_details=outcome_details,
                        derived_from=["pattern.hit.ref"],
                    )
                ],
            )
        ],
        patterns_with_signal=[pattern_id] if hit_rate > 0.5 and sample_size >= 3 else [],
    )


def _diff_report(*, ticker: str, change_type: str, materiality: str, summary: str) -> FilingDiffReport:
    return FilingDiffReport(
        ticker=ticker,
        run_id="run_1",
        as_of_date="2026-03-22",
        years_compared=[(2024, 2025)],
        changes=[
            {
                "section": "md_and_a",
                "fiscal_year_from": 2024,
                "fiscal_year_to": 2025,
                "change_type": change_type,
                "materiality": materiality,
                "summary": summary,
            }
        ],
        sections_compared=["md_and_a"],
        sections_skipped=[],
        section_diagnostics=[],
        truncation_applied=False,
        llm_enabled=False,
        created_at="2026-03-23T00:00:00+00:00",
    )


def test_two_converging_undervaluation_signals_produce_medium_confidence() -> None:
    report = build_variant_perceptions(
        ticker="ABC",
        as_of_date="2026-03-22",
        run_id="run_1",
        valuation_payload=_valuation_payload(),
        pattern_report=_pattern_report(ticker="ABC", pattern_id="deferred_revenue_leading_indicator"),
        persist=False,
    )

    assert len(report.perceptions) == 1
    perception = report.perceptions[0]
    assert perception.direction == "UNDERVALUED"
    assert perception.confidence == "MEDIUM"
    assert "Revenue growth should accelerate" in perception.testable_prediction
    assert "backlog conversion" in perception.catalyst


def test_three_converging_undervaluation_signals_produce_high_confidence() -> None:
    report = build_variant_perceptions(
        ticker="ABC",
        as_of_date="2026-03-22",
        run_id="run_1",
        valuation_payload=_valuation_payload(),
        pattern_report=_pattern_report(ticker="ABC", pattern_id="deferred_revenue_leading_indicator"),
        diff_report=_diff_report(
            ticker="ABC",
            change_type="STRATEGIC_SIGNAL",
            materiality="HIGH",
            summary="Management disclosed that the major investment phase is completed and efficiency should improve.",
        ),
        persist=False,
    )

    assert len(report.perceptions) == 1
    assert report.perceptions[0].confidence == "HIGH"


def test_contradicting_signals_are_preserved_not_ignored() -> None:
    report = build_variant_perceptions(
        ticker="ABC",
        as_of_date="2026-03-22",
        run_id="run_1",
        valuation_payload=_valuation_payload(),
        pattern_report=_pattern_report(ticker="ABC", pattern_id="deferred_revenue_leading_indicator"),
        diff_report=_diff_report(
            ticker="ABC",
            change_type="RISK_SIGNAL",
            materiality="MEDIUM",
            summary="The filing added competitive and pricing pressure language in the core segment.",
        ),
        persist=False,
    )

    assert len(report.perceptions) == 1
    perception = report.perceptions[0]
    assert perception.direction == "UNDERVALUED"
    assert perception.contradicting_signals
    assert any(signal.source == "FILING_DIFF" for signal in perception.contradicting_signals)


def test_single_signal_source_does_not_generate_perception() -> None:
    report = build_variant_perceptions(
        ticker="ABC",
        as_of_date="2026-03-22",
        run_id="run_1",
        valuation_payload=_valuation_payload(),
        persist=False,
    )

    assert report.perceptions == []


def test_missing_data_sources_are_reported_gracefully() -> None:
    report = build_variant_perceptions(
        ticker="ABC",
        as_of_date="2026-03-22",
        run_id="run_1",
        valuation_payload={},
        pattern_report=None,
        diff_report=None,
        intangible_payload=None,
        persist=False,
    )

    assert report.perceptions == []
    assert "VALUATION" in report.data_quality["missing_sources"]
    assert "FILING_DIFF" in report.data_quality["missing_sources"]


def test_overvaluation_convergence_with_high_implied_growth_and_weak_intangibles() -> None:
    report = build_variant_perceptions(
        ticker="XYZ",
        as_of_date="2026-03-22",
        run_id="run_1",
        valuation_payload=_valuation_payload(price=100.0, implied_growth=0.18, dcf_base=70.0, scorecard_signal="OVERVALUED"),
        pattern_report=_pattern_report(
            ticker="XYZ",
            pattern_id="gross_margin_regime_change",
            outcome_details="Operating margin deteriorated after the gross margin regime shifted downward.",
        ),
        intangible_payload={
            "ticker": "XYZ",
            "intangible_economics_total": 7.0,
            "rnd_productivity_score": 0.8,
            "owner_value_capture_score": 1.5,
            "gross_margin_durability_score": 2.0,
            "intangible_economics_reason_codes": ["REASON_WEAK_PER_SHARE_CAPTURE"],
            "derived_from": ["intangible.ref"],
        },
        persist=False,
    )

    assert len(report.perceptions) == 1
    perception = report.perceptions[0]
    assert perception.direction == "OVERVALUED"
    assert perception.confidence in {"MEDIUM", "HIGH"}
    assert "margin should deteriorate" in perception.testable_prediction.lower() or "revenue growth should fall below" in perception.testable_prediction.lower()


# --- valuation signal extraction: one piece of evidence per model family --------------


def _signal_types(valuations: dict) -> list[str]:
    from app.synthesis.variant_builder import _extract_valuation_signals

    signals, _, _ = _extract_valuation_signals(ticker="TEST", valuations=valuations)
    return [signal.signal_type for signal in signals]


def test_adjusted_method_on_the_same_side_as_its_twin_is_not_a_second_signal() -> None:
    """Signal extractor double counting: the R&D-adjusted DCF/EPV is the
    same model re-run, so a DISCOUNT on both counted the one finding twice and doubled
    the HIGH-strength count and the fair-growth adjustment. Only the plain method speaks."""
    valuations = {
        "dcf": {"outputs": {"base": 60.0}},
        "dcf_adjusted": {"outputs": {"base": 70.0}},
        "epv": {"outputs": {"value_per_share": 40.0}},
        "epv_adjusted": {"outputs": {"value_per_share": 45.0}},
        "reverse_dcf": {"inputs": {"price": 20.0}, "outputs": {}},
    }
    assert _signal_types(valuations) == ["DCF_DISCOUNT", "EPV_DISCOUNT"]


def test_adjusted_method_is_kept_when_its_twin_is_silent_or_disagrees() -> None:
    price = {"reverse_dcf": {"inputs": {"price": 20.0}, "outputs": {}}}
    # plain DCF within the dead band (upside 10%), adjusted is a real discount (upside 100%)
    silent_twin = {"dcf": {"outputs": {"base": 22.0}}, "dcf_adjusted": {"outputs": {"base": 40.0}}, **price}
    assert _signal_types(silent_twin) == ["DCF_ADJUSTED_DISCOUNT"]
    # plain says premium, adjusted says discount: opposite sides are both evidence
    disagreeing = {"dcf": {"outputs": {"base": 10.0}}, "dcf_adjusted": {"outputs": {"base": 40.0}}, **price}
    assert _signal_types(disagreeing) == ["DCF_PREMIUM", "DCF_ADJUSTED_DISCOUNT"]


def test_durable_dcf_substitution_applies_to_the_plain_dcf_only() -> None:
    """The scorecard's durable base corrects the plain dcf. A dcf_adjusted row keeps its own
    measured value (here 12 against a price of 10, inside the dead band), and a dcf row absent from the run is
    not conjured from the scorecard at all."""
    detail = {"dcf_base": 30.0, "dcf_raw_base": 50.0}
    scorecard = {"scorecard": {"outputs": {"pricing_zone_detail": detail}}}
    price = {"reverse_dcf": {"inputs": {"price": 10.0}, "outputs": {}}}
    assert _signal_types(
        {"dcf": {"outputs": {"base": 50.0}}, "dcf_adjusted": {"outputs": {"base": 12.0}}, **scorecard, **price}
    ) == ["DCF_DISCOUNT"]
    assert _signal_types({**scorecard, **price}) == []
