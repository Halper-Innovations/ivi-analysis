"""Tests for app.valuation.reverse_dcf."""

from __future__ import annotations

from app.fundamentals.normalize import UNKNOWN
from app.valuation.reverse_dcf import implied_growth_from_price


def _base_kwargs(**overrides):
    kwargs = dict(
        market_price=100.0,
        shares_outstanding=1_000_000.0,
        net_debt=0.0,
        base_revenue=500_000_000.0,
        margin=0.20,
        tax_rate=0.25,
        reinvestment_rate=0.35,
        discount_rate=0.10,
        terminal_growth=0.02,
    )
    kwargs.update(overrides)
    return kwargs


class TestReverseDcfTerminalSpread:
    """FIX 3: invalid discount/terminal-growth spread must surface, not silently clamp."""

    def test_discount_rate_equal_to_terminal_growth_returns_unknown(self):
        result, warnings = implied_growth_from_price(
            **_base_kwargs(discount_rate=0.02, terminal_growth=0.02)
        )
        assert result["implied_growth"] == UNKNOWN
        assert result["feasibility_gap_score"] == UNKNOWN
        assert any("discount" in w.lower() and "terminal" in w.lower() for w in warnings)

    def test_discount_rate_below_terminal_growth_returns_unknown(self):
        result, warnings = implied_growth_from_price(
            **_base_kwargs(discount_rate=0.01, terminal_growth=0.03)
        )
        assert result["implied_growth"] == UNKNOWN
        assert result["feasibility_gap_score"] == UNKNOWN
        assert any("discount" in w.lower() and "terminal" in w.lower() for w in warnings)

    def test_valid_spread_still_solves(self):
        result, _warnings = implied_growth_from_price(
            **_base_kwargs(discount_rate=0.10, terminal_growth=0.02)
        )
        assert result["implied_growth"] != UNKNOWN
        assert isinstance(result["implied_growth"], float)


class TestReverseDcfNetDebtIntegrity:
    def test_missing_net_debt_does_not_become_zero(self):
        result, warnings = implied_growth_from_price(
            **_base_kwargs(net_debt=None, market_price=2000.0)
        )

        assert result == {
            "implied_growth": UNKNOWN,
            "feasibility_gap_score": UNKNOWN,
        }
        assert warnings == ["Net debt UNKNOWN"]

    def test_explicit_zero_net_debt_remains_usable(self):
        result, warnings = implied_growth_from_price(
            **_base_kwargs(net_debt=0.0, market_price=2000.0)
        )

        assert isinstance(result["implied_growth"], float)
        assert result["target_ev"] == 2_000_000_000.0
        assert "Net debt UNKNOWN" not in warnings


class TestReverseDcfSaturation:
    """Widen the solver ceiling and expose an implied_growth_saturated flag."""

    def test_interior_solve_is_not_saturated(self):
        # market_price=100 saturates at the -0.25 floor; the interior band for this
        # fixture is ~1500-4000, so use 2000 (solves to ~+0.32258, not saturated).
        result, _warnings = implied_growth_from_price(**_base_kwargs(market_price=2000.0))
        assert result["implied_growth_saturated"] is False
        assert isinstance(result["implied_growth"], float)
        assert -0.249 < result["implied_growth"] < 0.599

    def test_overpriced_case_saturates_at_new_high_bound(self):
        result, _warnings = implied_growth_from_price(**_base_kwargs(market_price=10000.0))
        assert result["implied_growth"] >= 0.599
        assert result["implied_growth_saturated"] is True

    def test_invalid_spread_unknown_path_has_no_saturation_flag(self):
        result, _warnings = implied_growth_from_price(
            **_base_kwargs(discount_rate=0.02, terminal_growth=0.02)
        )
        assert result["implied_growth"] == UNKNOWN
        assert result.get("implied_growth_saturated") is None

    def test_deeply_undervalued_case_saturates_at_low_bound(self):
        result, _warnings = implied_growth_from_price(**_base_kwargs(market_price=1.0))
        assert result["implied_growth"] <= -0.249
        assert result["implied_growth_saturated"] is True


class TestSaturatedBoundLabelMatchesValue:
    """Queue defect (2026-08-23): the saturation label contradicted the value.

    The label used to be picked from the price side of the model, so a solve
    whose direction inverted returned the LOW growth bound (-0.25) under the
    label 'HIGH'. The label now names the bound the returned value sits at.
    """

    def test_queue_reproduction_exact_call_no_longer_returns_a_number(self):
        # The queue's call, verbatim and positional. tax_rate=1.5 is not a tax
        # rate the model can represent, so there is no implied growth to label.
        result, warnings = implied_growth_from_price(
            100.0, 1_000_000.0, 0.0, 500_000.0, 0.2, tax_rate=1.5
        )

        assert result == {
            "implied_growth": UNKNOWN,
            "feasibility_gap_score": UNKNOWN,
        }
        assert warnings == ["Effective tax rate outside 0-100%"]

    def test_queue_reproduction_with_a_valid_tax_rate_labels_high_and_returns_high(self):
        # Same price, same company, a tax rate the model can represent: the
        # price really is above everything the model can produce, so the label
        # is HIGH and the value is the HIGH bound. Both agree.
        result, warnings = implied_growth_from_price(
            100.0, 1_000_000.0, 0.0, 500_000.0, 0.2, tax_rate=0.25
        )

        assert result["implied_growth"] == 0.60
        assert result["implied_growth_saturated"] is True
        assert result["implied_growth_saturated_bound"] == "HIGH"
        assert result["margin_sign"] == "POSITIVE"
        assert warnings == ["Implied growth solve saturated at bound"]

    def test_money_loser_saturation_labels_the_bound_its_value_sits_at(self):
        # The root defect with no out-of-range input at all: a negative margin
        # makes enterprise value DECREASE with growth, so a price above
        # everything the model can produce lands on the LOW growth bound. This
        # used to return -0.25 labelled 'HIGH'.
        result, warnings = implied_growth_from_price(1.0, 100.0, 0.0, 1000.0, -0.10)

        assert result["implied_growth"] == -0.25
        assert result["implied_growth_saturated_bound"] == "LOW"
        assert result["margin_sign"] == "NEGATIVE"
        assert warnings == ["Implied growth solve saturated at bound"]

    def test_deep_cheap_positive_margin_still_labels_low(self):
        # Regression guard: the ordinary max-cheap case is unchanged, so the
        # CHEAP-floor bucket in expectations_gap (saturated LOW + POSITIVE
        # margin) keeps firing on exactly the cohort it fired on before.
        result, _warnings = implied_growth_from_price(4.0, 100.0, 0.0, 1000.0, 0.30)

        assert result["implied_growth"] == -0.25
        assert result["implied_growth_saturated_bound"] == "LOW"
        assert result["margin_sign"] == "POSITIVE"

    def test_overpriced_positive_margin_still_labels_high(self):
        result, _warnings = implied_growth_from_price(**_base_kwargs(market_price=10000.0))

        assert result["implied_growth"] == 0.60
        assert result["implied_growth_saturated_bound"] == "HIGH"
        assert result["margin_sign"] == "POSITIVE"

    def test_label_and_value_never_disagree_across_every_saturating_direction(self):
        # One case per quadrant of (curve direction) x (price side). The value
        # and the label come from a single boolean, so this is exhaustive.
        cases = [
            # (kwargs, expected growth, expected label)
            (
                dict(
                    market_price=4.0,
                    shares_outstanding=100.0,
                    net_debt=0.0,
                    base_revenue=1000.0,
                    margin=0.30,
                ),
                -0.25,
                "LOW",
            ),
            (
                dict(
                    market_price=10000.0,
                    shares_outstanding=1_000_000.0,
                    net_debt=0.0,
                    base_revenue=500_000_000.0,
                    margin=0.20,
                ),
                0.60,
                "HIGH",
            ),
            (
                dict(
                    market_price=1.0,
                    shares_outstanding=100.0,
                    net_debt=0.0,
                    base_revenue=1000.0,
                    margin=-0.10,
                ),
                -0.25,
                "LOW",
            ),
            (
                dict(
                    market_price=1.0,
                    shares_outstanding=100.0,
                    net_debt=-1_000_000.0,
                    base_revenue=1000.0,
                    margin=-0.10,
                ),
                0.60,
                "HIGH",
            ),
        ]
        for kwargs, expected_growth, expected_label in cases:
            result, _warnings = implied_growth_from_price(**kwargs)
            assert result["implied_growth_saturated"] is True
            assert result["implied_growth"] == expected_growth
            assert result["implied_growth_saturated_bound"] == expected_label


class TestOutOfRangeRateInputsAreRefused:
    """A rate at or above 100% inverts the sign of every modeled cash flow.

    dcf_lite clamps its own tax rate because it has no solver whose direction
    can invert; here the inversion is the failure, so the answer is a refusal
    rather than a repaired number nobody asked for.
    """

    def test_tax_rate_above_one_is_refused(self):
        result, warnings = implied_growth_from_price(**_base_kwargs(tax_rate=1.5))

        assert result == {"implied_growth": UNKNOWN, "feasibility_gap_score": UNKNOWN}
        assert warnings == ["Effective tax rate outside 0-100%"]

    def test_tax_rate_exactly_one_is_refused(self):
        result, warnings = implied_growth_from_price(**_base_kwargs(tax_rate=1.0))

        assert result["implied_growth"] == UNKNOWN
        assert warnings == ["Effective tax rate outside 0-100%"]

    def test_negative_tax_rate_is_refused(self):
        result, warnings = implied_growth_from_price(**_base_kwargs(tax_rate=-0.01))

        assert result["implied_growth"] == UNKNOWN
        assert warnings == ["Effective tax rate outside 0-100%"]

    def test_infinite_tax_rate_is_refused(self):
        result, warnings = implied_growth_from_price(**_base_kwargs(tax_rate=float("inf")))

        assert result["implied_growth"] == UNKNOWN
        assert warnings == ["Effective tax rate outside 0-100%"]

    def test_high_but_representable_tax_rate_still_solves(self):
        # 99% is punishing, not impossible: the model stays sign-stable, so it
        # gets an answer rather than a refusal.
        result, warnings = implied_growth_from_price(100.0, 100.0, 0.0, 1000.0, 0.2, tax_rate=0.99)

        assert result["implied_growth"] == 0.60
        assert result["implied_growth_saturated_bound"] == "HIGH"
        assert warnings == ["Implied growth solve saturated at bound"]

    def test_reinvestment_rate_above_one_is_refused(self):
        result, warnings = implied_growth_from_price(
            100.0, 1_000_000.0, 0.0, 500_000.0, 0.2, reinvestment_rate=1.5
        )

        assert result == {"implied_growth": UNKNOWN, "feasibility_gap_score": UNKNOWN}
        assert warnings == ["Reinvestment rate outside 0-100%"]


class TestSolverDegenerateCases:
    """The rest of the solver family: no unique root, non-finite math, crashes."""

    def test_zero_margin_no_longer_fabricates_an_interior_solve(self):
        # Every growth rate produces the same enterprise value, so no growth
        # rate is the implied one. Bisection used to run anyway and return the
        # low bracket end as a genuine solve with feasibility_gap_score 0.0.
        result, warnings = implied_growth_from_price(0.0, 100.0, 0.0, 1000.0, 0.0)

        assert result == {"implied_growth": UNKNOWN, "feasibility_gap_score": UNKNOWN}
        assert warnings == ["Modeled cash flow does not respond to growth"]

    def test_zero_base_revenue_is_refused_not_solved(self):
        result, warnings = implied_growth_from_price(10.0, 100.0, 0.0, 0.0, 0.20)

        assert result["implied_growth"] == UNKNOWN
        assert warnings == ["Modeled cash flow does not respond to growth"]

    def test_nan_margin_no_longer_returns_a_fabricated_number(self):
        # NaN compares False against everything, so the bisection used to walk
        # to +0.60 and report it unsaturated with a feasibility score of 49.99.
        result, warnings = implied_growth_from_price(100.0, 100.0, 0.0, 1000.0, float("nan"))

        assert result == {"implied_growth": UNKNOWN, "feasibility_gap_score": UNKNOWN}
        assert warnings == ["Non-finite price, share count, revenue or margin"]

    def test_discount_rate_of_negative_one_no_longer_raises(self):
        # (1 + discount_rate) is the compounding base; at -1.0 it is zero and
        # the module raised ZeroDivisionError into its caller.
        result, warnings = implied_growth_from_price(
            100.0, 100.0, 0.0, 1000.0, 0.2, discount_rate=-1.0, terminal_growth=-2.0
        )

        assert result == {"implied_growth": UNKNOWN, "feasibility_gap_score": UNKNOWN}
        assert warnings == ["Invalid discount rate"]

    def test_growth_above_the_discount_rate_is_a_real_solve_not_a_singularity(self):
        # The terminal denominator is (discount_rate - terminal_growth), never
        # (discount_rate - g), so the search crossing g = 10% is not a pole.
        # This one solves at 32.3% with a 10% discount rate.
        result, _warnings = implied_growth_from_price(**_base_kwargs(market_price=2000.0))

        assert result["implied_growth"] == 0.32257702579824427
        assert result["implied_growth_saturated"] is False
        assert result["feasibility_gap_score"] == 22.257702579824425

    def test_negative_margin_interior_root_still_solves(self):
        # A cash-rich money-loser with a well-defined root near zero growth:
        # the direction-aware bisection still finds it.
        result, warnings = implied_growth_from_price(2.2926, 100.0, -800.0, 1000.0, -0.10)

        assert result["implied_growth"] == -9.248090262324675e-07
        assert result["implied_growth_saturated"] is False
        assert result["margin_sign"] == "NEGATIVE"
        assert warnings == []
