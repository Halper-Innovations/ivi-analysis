"""Tests for app.research.thesis_updater."""
from __future__ import annotations


class TestParseFact:
    """Tests for _parse_fact — structured_fact string to numeric value."""

    def test_percentage_with_percent_sign(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("35%", "pct") == 0.35

    def test_percentage_without_percent_sign(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("0.35", "pct") == 0.35

    def test_percentage_points(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("12pp", "pp") == 12.0

    def test_plain_number_as_pp(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("12", "pp") == 12.0

    def test_years(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("4", "years") == 4.0

    def test_ratio_with_x(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("3.5x", "ratio") == 3.5

    def test_none_returns_none(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact(None, "pct") is None

    def test_unparseable_returns_none(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("not a number", "pct") is None

    def test_empty_string_returns_none(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("", "pct") is None

    def test_negative_percentage(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("-2%", "pct") == -0.02

    def test_negative_percentage_points(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("-3pp", "pp") == -3.0

    def test_negative_large_percentage(self):
        from app.research.thesis_updater import _parse_fact
        assert _parse_fact("-15%", "pct") == -0.15


class TestSensitivityHelpers:
    """Tests for _dcf_growth_impact and _epv_margin_impact."""

    def test_dcf_growth_impact_known_values(self):
        from app.research.thesis_updater import _dcf_growth_impact, ValuationInputs
        inputs = ValuationInputs(dcf=100.0, epv=50.0, graham=None, price=80.0, wacc=0.10, terminal_growth=0.02)
        # 100 * 2.0 * 0.01 / (0.10 - 0.02) = 100 * 0.02 / 0.08 = 25.0
        result = _dcf_growth_impact(2.0, inputs)
        assert abs(result - 25.0) < 0.01

    def test_dcf_growth_impact_missing_dcf(self):
        from app.research.thesis_updater import _dcf_growth_impact, ValuationInputs
        inputs = ValuationInputs(dcf=None, epv=50.0, graham=None, price=80.0, wacc=0.10, terminal_growth=0.02)
        assert _dcf_growth_impact(2.0, inputs) == 0.0

    def test_dcf_growth_impact_narrow_spread(self):
        from app.research.thesis_updater import _dcf_growth_impact, ValuationInputs
        inputs = ValuationInputs(dcf=100.0, epv=50.0, graham=None, price=80.0, wacc=0.024, terminal_growth=0.02)
        # spread = 0.004, below 0.005 guard — should return 0.0
        assert _dcf_growth_impact(2.0, inputs) == 0.0

    def test_epv_margin_impact_known_values(self):
        from app.research.thesis_updater import _epv_margin_impact, ValuationInputs
        inputs = ValuationInputs(dcf=100.0, epv=60.0, graham=None, price=80.0, wacc=0.10, terminal_growth=0.02)
        # 60 * 3.0 * 0.01 / 0.10 = 60 * 0.03 / 0.10 = 18.0
        result = _epv_margin_impact(3.0, inputs)
        assert abs(result - 18.0) < 0.01

    def test_epv_margin_impact_missing_epv(self):
        from app.research.thesis_updater import _epv_margin_impact, ValuationInputs
        inputs = ValuationInputs(dcf=100.0, epv=None, graham=None, price=80.0, wacc=0.10, terminal_growth=0.02)
        assert _epv_margin_impact(3.0, inputs) == 0.0

    def test_epv_margin_impact_zero_wacc(self):
        from app.research.thesis_updater import _epv_margin_impact, ValuationInputs
        inputs = ValuationInputs(dcf=100.0, epv=60.0, graham=None, price=80.0, wacc=0.0, terminal_growth=0.02)
        assert _epv_margin_impact(3.0, inputs) == 0.0


class TestExtractValuationInputs:
    """Tests for _extract_valuation_inputs — scorecard dict to ValuationInputs."""

    def test_full_scorecard(self):
        from app.research.thesis_updater import _extract_valuation_inputs
        scorecard = {
            "pricing_zone_detail": {
                "current_price": 80.0,
                "dcf_base": 100.0,
                "epv_adjusted": 60.0,
                "terminal_growth_used": 0.02,
            },
            "discounts": {"dcf": 0.25, "epv": -0.25, "graham": 0.10},
            "wacc_detail": {"adjusted_wacc": 0.095},
        }
        inputs = _extract_valuation_inputs(scorecard)
        assert inputs.dcf == 100.0
        assert inputs.epv == 60.0
        assert inputs.price == 80.0
        assert inputs.wacc == 0.095
        assert inputs.terminal_growth == 0.02
        # Graham inverts the TEXTBOOK discount: 80 / (1 - 0.10) = 88.89
        # (audit: graham-discount-inversion — the old 80/1.1 sign-inverted it)
        assert inputs.graham is not None
        assert abs(inputs.graham - 88.888888) < 0.01

    def test_missing_dcf(self):
        from app.research.thesis_updater import _extract_valuation_inputs
        scorecard = {
            "pricing_zone_detail": {"current_price": 80.0, "epv_adjusted": 60.0, "terminal_growth_used": 0.02},
            "wacc_detail": {"adjusted_wacc": 0.095},
        }
        inputs = _extract_valuation_inputs(scorecard)
        assert inputs.dcf is None
        assert inputs.epv == 60.0

    def test_empty_scorecard(self):
        from app.research.thesis_updater import _extract_valuation_inputs
        inputs = _extract_valuation_inputs({})
        assert inputs.dcf is None
        assert inputs.epv is None
        assert inputs.price is None
        assert inputs.wacc is None
        assert inputs.graham is None


from app.research.hypothesis_generator import Hypothesis, EvidenceNeed
from app.research.evidence_searcher import EvidenceResult, EvidenceItemResult


def _make_evidence_result(
    source: str,
    status: str,
    direction: str = "BEARISH",
    priority: str = "HIGH",
    impact_estimate: float | None = None,
    calibration_context: dict | None = None,
    items: list[EvidenceItemResult] | None = None,
) -> EvidenceResult:
    """Helper to build EvidenceResult for tests."""
    hyp = Hypothesis(
        claim="Test claim",
        direction=direction,
        evidence_needed=[EvidenceNeed("test_001_test", "test need", "REQUIRED")],
        falsification="Test falsification",
        priority=priority,
        source=source,
        impact_estimate=impact_estimate,
        calibration_context=calibration_context,
    )
    return EvidenceResult(
        hypothesis=hyp,
        evidence_item_results=items or [],
        hypothesis_status=status,
        coverage_score=0.5,
        classification_method="LLM",
    )


def _make_item(need_id: str, status: str, structured_fact: str | None = None) -> EvidenceItemResult:
    """Helper to build EvidenceItemResult for tests."""
    return EvidenceItemResult(
        need_id=need_id,
        needed="test need",
        importance="REQUIRED",
        status=status,
        classification_method="LLM",
        citations=[],
        excerpt="test excerpt",
        structured_fact=structured_fact,
        reasoning_short="test reasoning",
        candidates_considered=3,
        top_candidate_score=0.8,
        candidate_rankings=[],
    )


def _standard_inputs():
    from app.research.thesis_updater import ValuationInputs
    return ValuationInputs(dcf=100.0, epv=60.0, graham=None, price=80.0, wacc=0.10, terminal_growth=0.02)


class TestCalibrateGrowthTension:
    """Tests for _calibrate_growth_tension — GROWTH_VS_EARNINGS_POWER."""

    def test_confirmed_with_concentration_fact(self):
        """Fact-calibrated: concentration 35% -> 3pp band -> FACT_CALIBRATED."""
        from app.research.thesis_updater import _calibrate_growth_tension
        er = _make_evidence_result(
            "GROWTH_VS_EARNINGS_POWER", "CONFIRMED",
            calibration_context={"growth_dependency_ratio": 0.81, "dcf_epv_gap": 171.0},
            items=[_make_item("growth_ep_002_customer_concentration", "CONFIRMS", "35%")],
        )
        inputs = _standard_inputs()
        adjustments = _calibrate_growth_tension(er, inputs, {})
        assert len(adjustments) == 1
        adj = adjustments[0]
        assert adj.affected_method == "dcf"
        assert adj.adjustment_magnitude < 0  # bearish
        assert adj.adjustment_confidence == "FACT_CALIBRATED"
        assert "35%" in adj.structured_facts_used

    def test_confirmed_with_calibration_context_fallback(self):
        """No structured_fact, uses calibration_context growth_dependency_ratio for band selection."""
        from app.research.thesis_updater import _calibrate_growth_tension
        er = _make_evidence_result(
            "GROWTH_VS_EARNINGS_POWER", "CONFIRMED",
            calibration_context={"growth_dependency_ratio": 0.81, "dcf_epv_gap": 171.0},
        )
        inputs = _standard_inputs()
        adjustments = _calibrate_growth_tension(er, inputs, {})
        assert len(adjustments) == 1
        adj = adjustments[0]
        assert adj.affected_method == "dcf"
        assert adj.adjustment_confidence == "HEURISTIC"
        assert adj.adjustment_magnitude < 0
        # 81% dependency -> 3.5pp band: 100 * 0.035 / 0.08 = 43.75
        assert abs(adj.adjustment_magnitude + 43.75) < 0.01

    def test_confirmed_heuristic_default(self):
        """No fact, no calibration_context -> heuristic 2pp."""
        from app.research.thesis_updater import _calibrate_growth_tension
        er = _make_evidence_result("GROWTH_VS_EARNINGS_POWER", "CONFIRMED")
        inputs = _standard_inputs()
        adjustments = _calibrate_growth_tension(er, inputs, {})
        assert len(adjustments) == 1
        assert adjustments[0].adjustment_confidence == "HEURISTIC"
        # 2pp heuristic: 100 * 0.02 / 0.08 = 25.0
        assert abs(adjustments[0].adjustment_magnitude + 25.0) < 0.01

    def test_contradicted_returns_empty(self):
        from app.research.thesis_updater import _calibrate_growth_tension
        er = _make_evidence_result("GROWTH_VS_EARNINGS_POWER", "CONTRADICTED")
        assert _calibrate_growth_tension(er, _standard_inputs(), {}) == []

    def test_inconclusive_returns_empty(self):
        from app.research.thesis_updater import _calibrate_growth_tension
        er = _make_evidence_result("GROWTH_VS_EARNINGS_POWER", "INCONCLUSIVE")
        assert _calibrate_growth_tension(er, _standard_inputs(), {}) == []

    def test_partially_confirmed_always_heuristic(self):
        """PARTIALLY_CONFIRMED always gets HEURISTIC even with fact."""
        from app.research.thesis_updater import _calibrate_growth_tension
        er = _make_evidence_result(
            "GROWTH_VS_EARNINGS_POWER", "PARTIALLY_CONFIRMED",
            items=[_make_item("growth_ep_002_customer_concentration", "CONFIRMS", "35%")],
        )
        adjustments = _calibrate_growth_tension(er, _standard_inputs(), {})
        assert len(adjustments) == 1
        assert adjustments[0].adjustment_confidence == "HEURISTIC"

    def test_evidence_trace_only_includes_consumed_items(self):
        """Adjustment should only reference the specific item that drove calibration, not unrelated items."""
        from app.research.thesis_updater import _calibrate_growth_tension
        er = _make_evidence_result(
            "GROWTH_VS_EARNINGS_POWER", "CONFIRMED",
            items=[
                _make_item("growth_ep_001_revenue_segment", "CONFIRMS", "$500M"),
                _make_item("growth_ep_002_customer_concentration", "CONFIRMS", "35%"),
                _make_item("growth_ep_003_mgmt_guidance", "CONFIRMS", "expects 5% growth"),
            ],
        )
        adjustments = _calibrate_growth_tension(er, _standard_inputs(), {})
        assert len(adjustments) == 1
        adj = adjustments[0]
        # Only customer_concentration should appear — that's the item the calibrator used
        assert adj.evidence_item_ids == ["growth_ep_002_customer_concentration"]
        assert adj.structured_facts_used == ["35%"]

    def test_contradicts_item_not_used_for_calibration(self):
        """A CONTRADICTS item should not drive fact-calibrated adjustment even if hypothesis is CONFIRMED."""
        from app.research.thesis_updater import _calibrate_growth_tension
        er = _make_evidence_result(
            "GROWTH_VS_EARNINGS_POWER", "CONFIRMED",
            items=[_make_item("growth_ep_002_customer_concentration", "CONTRADICTS", "35%")],
        )
        adjustments = _calibrate_growth_tension(er, _standard_inputs(), {})
        assert len(adjustments) == 1
        # Should NOT be FACT_CALIBRATED — the only item is CONTRADICTS
        assert adjustments[0].adjustment_confidence == "HEURISTIC"
        assert adjustments[0].evidence_item_ids == []
        assert adjustments[0].structured_facts_used == []


class TestCalibrateNegativeOE:
    """Tests for _calibrate_negative_oe — NEGATIVE_OWNER_EARNINGS (dual method)."""

    def test_confirmed_with_years_from_calibration_context(self):
        """Should emit 2 records (DCF + EPV) using calibration_context years."""
        from app.research.thesis_updater import _calibrate_negative_oe
        er = _make_evidence_result(
            "NEGATIVE_OWNER_EARNINGS", "CONFIRMED",
            calibration_context={"negative_oe_years": 3},
        )
        inputs = _standard_inputs()
        adjustments = _calibrate_negative_oe(er, inputs, {})
        assert len(adjustments) == 2
        methods = {a.affected_method for a in adjustments}
        assert methods == {"dcf", "epv"}
        dcf_adj = next(a for a in adjustments if a.affected_method == "dcf")
        epv_adj = next(a for a in adjustments if a.affected_method == "epv")
        assert dcf_adj.adjustment_magnitude < 0
        assert epv_adj.adjustment_magnitude < 0
        # 3yr: DCF 3pp growth haircut = 100 * 0.03 / 0.08 = 37.5
        # 3yr: EPV 4.5pp margin haircut = 60 * 0.045 / 0.10 = 27.0
        assert abs(dcf_adj.adjustment_magnitude - (-37.5)) < 0.01
        assert abs(epv_adj.adjustment_magnitude - (-27.0)) < 0.01

    def test_confirmed_heuristic_no_context(self):
        """No calibration_context -> heuristic defaults (DCF 2.5pp, EPV 3pp)."""
        from app.research.thesis_updater import _calibrate_negative_oe
        er = _make_evidence_result("NEGATIVE_OWNER_EARNINGS", "CONFIRMED")
        inputs = _standard_inputs()
        adjustments = _calibrate_negative_oe(er, inputs, {})
        assert len(adjustments) == 2
        assert all(a.adjustment_confidence == "HEURISTIC" for a in adjustments)

    def test_contradicted_returns_empty(self):
        from app.research.thesis_updater import _calibrate_negative_oe
        er = _make_evidence_result("NEGATIVE_OWNER_EARNINGS", "CONTRADICTED")
        assert _calibrate_negative_oe(er, _standard_inputs(), {}) == []

    def test_sbc_fact_increases_epv_haircut(self):
        """SBC > 10% should increase EPV haircut by 1pp and trace the SBC item."""
        from app.research.thesis_updater import _calibrate_negative_oe
        er = _make_evidence_result(
            "NEGATIVE_OWNER_EARNINGS", "CONFIRMED",
            calibration_context={"negative_oe_years": 3},
            items=[_make_item("neg_oe_002_sbc", "CONFIRMS", "12%")],
        )
        adjustments = _calibrate_negative_oe(er, _standard_inputs(), {})
        epv_adj = next(a for a in adjustments if a.affected_method == "epv")
        # Base: 3yr -> 4.5pp. SBC 12% > 10% -> +1pp = 5.5pp
        # 60 * 0.055 / 0.10 = 33.0
        assert abs(epv_adj.adjustment_magnitude + 33.0) < 0.01
        assert epv_adj.adjustment_confidence == "HEURISTIC"  # base from calibration_context, fact only modulates
        assert "neg_oe_002_sbc" in epv_adj.evidence_item_ids
        assert "12%" in epv_adj.structured_facts_used

    def test_capex_fact_reduces_dcf_haircut(self):
        """Growth capex > 60% should reduce DCF haircut by 1pp."""
        from app.research.thesis_updater import _calibrate_negative_oe
        er = _make_evidence_result(
            "NEGATIVE_OWNER_EARNINGS", "CONFIRMED",
            calibration_context={"negative_oe_years": 3},
            items=[_make_item("neg_oe_001_capex", "CONFIRMS", "65%")],
        )
        adjustments = _calibrate_negative_oe(er, _standard_inputs(), {})
        dcf_adj = next(a for a in adjustments if a.affected_method == "dcf")
        # Base: 3yr -> 3pp. Capex 65% > 60% -> -1pp = 2pp
        # 100 * 0.02 / 0.08 = 25.0
        assert abs(dcf_adj.adjustment_magnitude + 25.0) < 0.01
        assert "neg_oe_001_capex" in dcf_adj.evidence_item_ids

    def test_dcf_detail_does_not_leak_into_epv(self):
        """Capex modulation detail should appear in DCF record only, not EPV."""
        from app.research.thesis_updater import _calibrate_negative_oe
        er = _make_evidence_result(
            "NEGATIVE_OWNER_EARNINGS", "CONFIRMED",
            calibration_context={"negative_oe_years": 3},
            items=[_make_item("neg_oe_001_capex", "CONFIRMS", "65%")],
        )
        adjustments = _calibrate_negative_oe(er, _standard_inputs(), {})
        dcf_adj = next(a for a in adjustments if a.affected_method == "dcf")
        epv_adj = next(a for a in adjustments if a.affected_method == "epv")
        assert "capex" in dcf_adj.calibration_detail
        assert "capex" not in epv_adj.calibration_detail

    def test_contradicts_capex_not_used_for_modulation(self):
        """A CONTRADICTS capex item should not modulate the DCF haircut."""
        from app.research.thesis_updater import _calibrate_negative_oe
        er = _make_evidence_result(
            "NEGATIVE_OWNER_EARNINGS", "CONFIRMED",
            calibration_context={"negative_oe_years": 3},
            items=[_make_item("neg_oe_001_capex", "CONTRADICTS", "65%")],
        )
        adjustments = _calibrate_negative_oe(er, _standard_inputs(), {})
        dcf_adj = next(a for a in adjustments if a.affected_method == "dcf")
        # CONTRADICTS should not trigger capex modulation -> base 3pp stays
        # 100 * 0.03 / 0.08 = 37.5
        assert abs(dcf_adj.adjustment_magnitude + 37.5) < 0.01
        assert dcf_adj.evidence_item_ids == []


class TestCalibratePersistentBurn:
    """Tests for _calibrate_persistent_burn — PERSISTENT_CASH_BURN (dual method)."""

    def test_confirmed_with_burn_years(self):
        from app.research.thesis_updater import _calibrate_persistent_burn
        er = _make_evidence_result(
            "PERSISTENT_CASH_BURN", "CONFIRMED",
            calibration_context={"burn_years": 3},
        )
        adjustments = _calibrate_persistent_burn(er, _standard_inputs(), {})
        assert len(adjustments) == 2
        methods = {a.affected_method for a in adjustments}
        assert methods == {"dcf", "epv"}
        dcf_adj = next(a for a in adjustments if a.affected_method == "dcf")
        epv_adj = next(a for a in adjustments if a.affected_method == "epv")
        # DCF hit harder than EPV for cash burn (3yr: DCF 3.5pp, EPV 2.5pp)
        assert abs(dcf_adj.adjustment_magnitude) > abs(epv_adj.adjustment_magnitude)

    def test_low_growth_capex_increases_dcf_haircut(self):
        """Growth capex < 30% should increase DCF haircut by 1pp."""
        from app.research.thesis_updater import _calibrate_persistent_burn
        er = _make_evidence_result(
            "PERSISTENT_CASH_BURN", "CONFIRMED",
            calibration_context={"burn_years": 3},
            items=[_make_item("persistent_cash_burn_001_capex_breakdown_growth_vs_main", "CONFIRMS", "20%")],
        )
        adjustments = _calibrate_persistent_burn(er, _standard_inputs(), {})
        dcf_adj = next(a for a in adjustments if a.affected_method == "dcf")
        # Base: 3yr -> 3.5pp. Capex 20% < 30% -> +1pp = 4.5pp
        # 100 * 0.045 / 0.08 = 56.25
        assert abs(dcf_adj.adjustment_magnitude + 56.25) < 0.01
        assert dcf_adj.adjustment_confidence == "HEURISTIC"  # base from calibration_context, fact only modulates

    def test_rd_alone_does_not_reduce_haircuts(self):
        """R&D fact without corroborating growth capex should NOT reduce haircuts."""
        from app.research.thesis_updater import _calibrate_persistent_burn
        er = _make_evidence_result(
            "PERSISTENT_CASH_BURN", "CONFIRMED",
            calibration_context={"burn_years": 3},
            items=[_make_item("persistent_cash_burn_002_r&d_as_%_of_revenue_trend", "CONFIRMS", "15%")],
        )
        adjustments = _calibrate_persistent_burn(er, _standard_inputs(), {})
        dcf_adj = next(a for a in adjustments if a.affected_method == "dcf")
        # No capex_breakdown fact -> R&D modulation does not fire -> base 3.5pp stays
        # 100 * 0.035 / 0.08 = 43.75
        assert abs(dcf_adj.adjustment_magnitude + 43.75) < 0.01

    def test_rd_plus_growth_capex_reduces_both_haircuts(self):
        """R&D confirms + growth capex >= 30% should reduce both haircuts by 1pp."""
        from app.research.thesis_updater import _calibrate_persistent_burn
        er = _make_evidence_result(
            "PERSISTENT_CASH_BURN", "CONFIRMED",
            calibration_context={"burn_years": 3},
            items=[
                _make_item("persistent_cash_burn_001_capex_breakdown_growth_vs_main", "CONFIRMS", "45%"),
                _make_item("persistent_cash_burn_002_r&d_as_%_of_revenue_trend", "CONFIRMS", "15%"),
            ],
        )
        adjustments = _calibrate_persistent_burn(er, _standard_inputs(), {})
        dcf_adj = next(a for a in adjustments if a.affected_method == "dcf")
        epv_adj = next(a for a in adjustments if a.affected_method == "epv")
        # capex 45% >= 30% so no DCF +1pp penalty. R&D + growth capex -> both -1pp
        # Base: 3yr -> DCF 3.5pp, EPV 2.5pp. Both -1pp = DCF 2.5pp, EPV 1.5pp
        # DCF: 100 * 0.025 / 0.08 = 31.25
        assert abs(dcf_adj.adjustment_magnitude + 31.25) < 0.01
        # EPV: 60 * 0.015 / 0.10 = 9.0
        assert abs(epv_adj.adjustment_magnitude + 9.0) < 0.01


class TestCalibrateMarginCollapse:
    """Tests for _calibrate_margin_collapse — MARGIN_COLLAPSE."""

    def test_confirmed_with_calibration_context(self):
        """calibration_context margin_decline_pp -> band selection."""
        from app.research.thesis_updater import _calibrate_margin_collapse
        er = _make_evidence_result(
            "MARGIN_COLLAPSE", "CONFIRMED",
            calibration_context={"margin_decline_pp": 12.0},
        )
        adjustments = _calibrate_margin_collapse(er, _standard_inputs(), {})
        assert len(adjustments) == 1
        assert adjustments[0].affected_method == "epv"
        assert adjustments[0].adjustment_magnitude < 0
        # 12pp > 10pp -> 5pp band: 60 * 0.05 / 0.10 = 30.0
        assert abs(adjustments[0].adjustment_magnitude + 30.0) < 0.01

    def test_confirmed_with_structured_fact_over_context(self):
        """structured_fact should take priority over calibration_context."""
        from app.research.thesis_updater import _calibrate_margin_collapse
        er = _make_evidence_result(
            "MARGIN_COLLAPSE", "CONFIRMED",
            calibration_context={"margin_decline_pp": 12.0},
            items=[_make_item("margin_collapse_001_segment_margins", "CONFIRMS", "7pp")],
        )
        adjustments = _calibrate_margin_collapse(er, _standard_inputs(), {})
        assert len(adjustments) == 1
        # 7pp is in 5-10pp band -> 3pp haircut: 60 * 0.03 / 0.10 = 18.0
        assert abs(adjustments[0].adjustment_magnitude + 18.0) < 0.01
        assert adjustments[0].adjustment_confidence == "FACT_CALIBRATED"


class TestCalibrateRevenueDecline:
    """Tests for _calibrate_revenue_decline — REVENUE_DECLINE_FROM_PEAK."""

    def test_confirmed_with_calibration_context(self):
        from app.research.thesis_updater import _calibrate_revenue_decline
        er = _make_evidence_result(
            "REVENUE_DECLINE_FROM_PEAK", "CONFIRMED",
            calibration_context={"peak_decline_pct": 0.32},
        )
        adjustments = _calibrate_revenue_decline(er, _standard_inputs(), {})
        assert len(adjustments) == 1
        assert adjustments[0].affected_method == "dcf"
        # 32% in 25-40% band -> 2.5pp: 100 * 0.025 / 0.08 = 31.25
        assert abs(adjustments[0].adjustment_magnitude + 31.25) < 0.01

    def test_confirmed_heuristic_default(self):
        from app.research.thesis_updater import _calibrate_revenue_decline
        er = _make_evidence_result("REVENUE_DECLINE_FROM_PEAK", "CONFIRMED")
        adjustments = _calibrate_revenue_decline(er, _standard_inputs(), {})
        assert len(adjustments) == 1
        assert adjustments[0].adjustment_confidence == "HEURISTIC"
        # Heuristic 2pp: 100 * 0.02 / 0.08 = 25.0
        assert abs(adjustments[0].adjustment_magnitude + 25.0) < 0.01

    def test_filing_segment_pct_not_used_as_decline(self):
        """A segment composition fact like '40%' must NOT be treated as 40% decline."""
        from app.research.thesis_updater import _calibrate_revenue_decline
        er = _make_evidence_result(
            "REVENUE_DECLINE_FROM_PEAK", "CONFIRMED",
            calibration_context={"peak_decline_pct": 0.32},
            items=[_make_item("revenue_decline_from_peak_001_revenue_by_segment", "CONFIRMS", "40%")],
        )
        adjustments = _calibrate_revenue_decline(er, _standard_inputs(), {})
        assert len(adjustments) == 1
        # Should use calibration_context (32%), not the filing fact (40%)
        # 32% -> 25-40% band -> 2.5pp: 100 * 0.025 / 0.08 = 31.25
        assert abs(adjustments[0].adjustment_magnitude + 31.25) < 0.01


class TestUpdateThesis:
    """Tests for the public API update_thesis()."""

    def _scorecard(self, dcf=100.0, epv=60.0, price=80.0, wacc=0.10, tg=0.02, graham_disc=0.10):
        sc = {
            "pricing_zone_detail": {"current_price": price, "terminal_growth_used": tg},
            "wacc_detail": {"adjusted_wacc": wacc},
            "discounts": {},
        }
        if dcf is not None:
            sc["pricing_zone_detail"]["dcf_base"] = dcf
        if epv is not None:
            sc["pricing_zone_detail"]["epv_adjusted"] = epv
        if graham_disc is not None:
            sc["discounts"]["graham"] = graham_disc
        return sc

    def _tensions(self):
        return {"tension_type": "NONE", "method_values": {"dcf": 100.0, "epv": 60.0}}

    def test_empty_evidence_returns_no_evidence(self):
        from app.research.thesis_updater import update_thesis
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [], iteration=0)
        assert result.status == "NO_EVIDENCE"
        assert result.original_dcf == 100.0
        assert result.adjustments == []
        assert result.hypotheses_confirmed == 0

    def test_no_dcf_no_epv_returns_no_valuation(self):
        from app.research.thesis_updater import update_thesis
        sc = self._scorecard(dcf=None, epv=None)
        er = _make_evidence_result("GROWTH_VS_EARNINGS_POWER", "CONFIRMED")
        result = update_thesis("TEST", sc, self._tensions(), [er])
        assert result.status == "NO_VALUATION"

    def test_confirmed_bearish_reduces_dcf(self):
        from app.research.thesis_updater import update_thesis
        er = _make_evidence_result(
            "GROWTH_VS_EARNINGS_POWER", "CONFIRMED",
            items=[_make_item("growth_ep_002_customer_concentration", "CONFIRMS", "35%")],
        )
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [er])
        assert result.status == "OK"
        assert result.adjusted_dcf is not None
        assert result.adjusted_dcf < 100.0  # reduced from original
        assert len(result.adjustments) == 1
        assert result.hypotheses_confirmed == 1

    def test_contradicted_no_adjustment(self):
        from app.research.thesis_updater import update_thesis
        er = _make_evidence_result("GROWTH_VS_EARNINGS_POWER", "CONTRADICTED")
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [er])
        assert result.status == "OK"
        assert result.adjustments == []
        assert result.adjusted_dcf == 100.0
        assert result.hypotheses_contradicted == 1

    def test_class3_source_no_adjustment(self):
        from app.research.thesis_updater import update_thesis
        er = _make_evidence_result("DEBT_SPIKE", "CONFIRMED")
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [er])
        assert result.status == "OK"
        assert result.adjustments == []
        assert result.hypotheses_confirmed == 1  # still counted

    def test_adjusted_intrinsic_mid_both_methods(self):
        from app.research.thesis_updater import update_thesis
        er = _make_evidence_result(
            "NEGATIVE_OWNER_EARNINGS", "CONFIRMED",
            calibration_context={"negative_oe_years": 3},
        )
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [er])
        assert result.adjusted_dcf is not None
        assert result.adjusted_epv is not None
        assert result.adjusted_intrinsic_mid is not None
        # Mid should be average of adjusted DCF and adjusted EPV
        expected_mid = (result.adjusted_dcf + result.adjusted_epv) / 2
        assert abs(result.adjusted_intrinsic_mid - expected_mid) < 0.01

    def test_dcf_only_no_epv(self):
        from app.research.thesis_updater import update_thesis
        sc = self._scorecard(epv=None)
        er = _make_evidence_result("GROWTH_VS_EARNINGS_POWER", "CONFIRMED")
        result = update_thesis("TEST", sc, self._tensions(), [er])
        assert result.status == "OK"
        assert result.adjusted_dcf is not None
        assert result.adjusted_epv is None
        assert result.adjusted_intrinsic_mid == result.adjusted_dcf

    def test_floor_at_zero(self):
        """Massive adjustment should clamp to 0, not go negative."""
        from app.research.thesis_updater import update_thesis
        er = _make_evidence_result(
            "REVENUE_DECLINE_FROM_PEAK", "CONFIRMED",
            calibration_context={"peak_decline_pct": 0.90},
        )
        # 90% decline -> 4pp band: 100 * 0.04 / 0.08 = 50, but multiple could stack
        sc = self._scorecard(dcf=10.0)  # small DCF, big haircut
        result = update_thesis("TEST", sc, self._tensions(), [er])
        assert result.adjusted_dcf >= 0.0

    def test_unresolved_evidence_collected(self):
        from app.research.thesis_updater import update_thesis
        items = [
            _make_item("growth_ep_002_customer_concentration", "NOT_FOUND"),
            _make_item("growth_ep_003_mgmt_guidance", "INCONCLUSIVE"),
        ]
        er = _make_evidence_result(
            "GROWTH_VS_EARNINGS_POWER", "INCONCLUSIVE",
            items=items,
        )
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [er])
        assert len(result.unresolved) == 2
        reasons = {u.unresolved_reason for u in result.unresolved}
        assert "NOT_FOUND" in reasons
        assert "INCONCLUSIVE" in reasons

    def test_average_coverage(self):
        from app.research.thesis_updater import update_thesis
        er1 = _make_evidence_result("GROWTH_VS_EARNINGS_POWER", "CONFIRMED")
        er1_replaced = EvidenceResult(
            hypothesis=er1.hypothesis, evidence_item_results=er1.evidence_item_results,
            hypothesis_status=er1.hypothesis_status, coverage_score=0.8,
            classification_method=er1.classification_method,
        )
        er2 = _make_evidence_result("DEBT_SPIKE", "INCONCLUSIVE")
        er2_replaced = EvidenceResult(
            hypothesis=er2.hypothesis, evidence_item_results=er2.evidence_item_results,
            hypothesis_status=er2.hypothesis_status, coverage_score=0.4,
            classification_method=er2.classification_method,
        )
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [er1_replaced, er2_replaced])
        assert abs(result.average_coverage - 0.6) < 0.01

    def test_margin_of_safety_computed(self):
        from app.research.thesis_updater import update_thesis
        er = _make_evidence_result("GROWTH_VS_EARNINGS_POWER", "CONTRADICTED")
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [er])
        # No adjustments, so adjusted_intrinsic_mid = (100 + 60) / 2 = 80
        # MOS = (80 - 80) / 80 = 0.0
        assert result.adjusted_margin_of_safety is not None
        assert abs(result.adjusted_margin_of_safety - 0.0) < 0.01

    def test_high_priority_unresolved_count(self):
        from app.research.thesis_updater import update_thesis
        items_high = [_make_item("growth_ep_001_revenue_segment", "NOT_FOUND")]
        er_high = _make_evidence_result(
            "GROWTH_VS_EARNINGS_POWER", "INCONCLUSIVE", priority="HIGH", items=items_high,
        )
        items_low = [_make_item("comp_disrupt_001_competitor", "INCONCLUSIVE")]
        er_low = _make_evidence_result(
            "COMPETITIVE_DISRUPTION", "INCONCLUSIVE", priority="LOW", items=items_low,
        )
        result = update_thesis("TEST", self._scorecard(), self._tensions(), [er_high, er_low])
        assert result.high_priority_unresolved == 1


class TestAdjustedValueFloorFlag:
    """FIX 5: deeply-negative adjusted theses are floored to 0.0 but flagged."""

    def test_floor_flag_set_when_adjusted_dcf_would_be_negative(self):
        from app.research.thesis_updater import (
            _compute_adjusted_valuations,
            ThesisAdjustment,
            ValuationInputs,
        )
        inputs = ValuationInputs(dcf=10.0, epv=10.0, graham=None, price=5.0)
        # A large negative DCF adjustment that drives raw adjusted DCF well below zero.
        adj = ThesisAdjustment(
            hypothesis_source="GROWTH_VS_EARNINGS_POWER",
            hypothesis_claim="stress",
            hypothesis_direction="bearish",
            hypothesis_status="CONFIRMED",
            affected_method="dcf",
            adjustment_magnitude=-50.0,
            adjustment_confidence="HEURISTIC",
            calibration_detail="stress",
            evidence_item_ids=[],
            structured_facts_used=[],
        )
        result = _compute_adjusted_valuations(inputs, [adj])
        # 5-tuple now: (adj_dcf, adj_epv, adj_mid, adj_mos, floored)
        assert len(result) == 5
        adj_dcf, adj_epv, adj_mid, adj_mos, floored = result
        assert adj_dcf == 0.0
        assert adj_epv == 10.0
        assert floored is True

    def test_floor_flag_false_when_no_truncation(self):
        from app.research.thesis_updater import (
            _compute_adjusted_valuations,
            ValuationInputs,
        )
        inputs = ValuationInputs(dcf=100.0, epv=60.0, graham=None, price=80.0)
        adj_dcf, adj_epv, adj_mid, adj_mos, floored = _compute_adjusted_valuations(inputs, [])
        assert adj_dcf == 100.0
        assert adj_epv == 60.0
        assert floored is False

    def test_thesis_result_exposes_floored_flag(self):
        from app.research.thesis_updater import update_thesis
        scorecard = {
            "pricing_zone_detail": {
                "current_price": 80.0,
                "dcf_base": 100.0,
                "epv_adjusted": 60.0,
                "terminal_growth_used": 0.02,
            },
            "discounts": {"dcf": 0.25, "epv": -0.25},
            "wacc_detail": {"adjusted_wacc": 0.095},
        }
        result = update_thesis("TEST", scorecard, {}, [])
        # No adjustments => nothing floored.
        assert result.adjusted_value_floored is False


class TestThesisMosConventionTextbook:
    """FIX 6: thesis adjusted MoS uses the textbook (intrinsic - price)/intrinsic convention."""

    def test_textbook_mos_literal_value(self):
        from app.research.thesis_updater import (
            _compute_adjusted_valuations,
            ValuationInputs,
        )
        # intrinsic mid = (150 + 150)/2 = 150, price = 100
        # textbook MoS = (150 - 100) / 150 = 0.3333...
        inputs = ValuationInputs(dcf=150.0, epv=150.0, graham=None, price=100.0)
        adj_dcf, adj_epv, adj_mid, adj_mos, floored = _compute_adjusted_valuations(inputs, [])
        assert adj_mid == 150.0
        assert abs(adj_mos - (1.0 / 3.0)) < 1e-12
