"""Tests for app.valuation.nonrecurring_filter."""

from __future__ import annotations


def test_restructuring_detected():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    facts = {
        "restructuring_charges": [(2024, 50.0), (2025, 30.0)],
        "operating_income": [(2023, 200.0), (2024, 150.0), (2025, 180.0)],
        "net_income": [(2023, 150.0), (2024, 100.0), (2025, 130.0)],
        "revenue": [(2023, 1000.0), (2024, 1020.0), (2025, 1050.0)],
    }
    result = detect_nonrecurring_items(facts)
    assert result["has_nonrecurring"] is True
    assert "RESTRUCTURING_CHARGE_DETECTED" in result["nonrecurring_flags"]
    assert 2024 in result["nonrecurring_years"]
    assert 2025 in result["nonrecurring_years"]
    assert result["restructuring_by_year"][2024] == 50.0
    assert result["restructuring_by_year"][2025] == 30.0
    assert result["adjusted_operating_income"][2024] == 200.0
    assert result["adjusted_operating_income"][2025] == 210.0
    # The charge is pre-tax: net income gets it back net of the 25% statutory fallback
    # (no tax lines on file), 100 + 50 * 0.75 and 130 + 30 * 0.75.
    assert result["adjusted_net_income"][2024] == 137.5
    assert result["adjusted_net_income"][2025] == 152.5


def test_oi_spike_without_revenue_change():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    facts = {
        "operating_income": [(2023, 100.0), (2024, 140.0), (2025, 105.0)],
        "revenue": [(2023, 1000.0), (2024, 1030.0), (2025, 1050.0)],
        "net_income": [(2023, 70.0), (2024, 100.0), (2025, 75.0)],
    }
    result = detect_nonrecurring_items(facts)
    assert result["has_nonrecurring"] is True
    assert "POSSIBLE_NONRECURRING_OI_SPIKE" in result["nonrecurring_flags"]
    assert 2024 in result["nonrecurring_years"]


def test_goodwill_impairment_detected():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    facts = {
        "goodwill": [(2023, 500.0), (2024, 430.0), (2025, 420.0)],
        "operating_income": [(2023, 100.0), (2024, 95.0), (2025, 98.0)],
        "revenue": [(2023, 1000.0), (2024, 980.0), (2025, 1010.0)],
        "net_income": [(2023, 70.0), (2024, 65.0), (2025, 68.0)],
    }
    result = detect_nonrecurring_items(facts)
    assert result["has_nonrecurring"] is True
    assert "GOODWILL_IMPAIRMENT_LIKELY" in result["nonrecurring_flags"]
    assert 2024 in result["goodwill_impairment_years"]


def test_clean_company_no_flags():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    facts = {
        "operating_income": [(2023, 100.0), (2024, 108.0), (2025, 115.0)],
        "revenue": [(2023, 1000.0), (2024, 1080.0), (2025, 1160.0)],
        "net_income": [(2023, 70.0), (2024, 76.0), (2025, 81.0)],
        "goodwill": [(2023, 500.0), (2024, 500.0), (2025, 500.0)],
    }
    result = detect_nonrecurring_items(facts)
    assert result["has_nonrecurring"] is False
    assert result["nonrecurring_flags"] == []
    assert result["nonrecurring_years"] == []


def test_missing_data_graceful():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    result = detect_nonrecurring_items({})
    assert result["has_nonrecurring"] is False
    assert result["nonrecurring_flags"] == []
    assert result["adjusted_operating_income"] == {}
    assert result["adjusted_net_income"] == {}


def test_adjusted_oi_adds_back_restructuring():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    facts = {
        "restructuring_charges": [(2025, 20.0)],
        "operating_income": [(2023, 100.0), (2024, 105.0), (2025, 90.0)],
        "net_income": [(2023, 70.0), (2024, 74.0), (2025, 60.0)],
        "revenue": [(2023, 1000.0), (2024, 1020.0), (2025, 1050.0)],
    }
    result = detect_nonrecurring_items(facts)
    assert result["adjusted_operating_income"][2023] == 100.0
    assert result["adjusted_operating_income"][2024] == 105.0
    assert result["adjusted_operating_income"][2025] == 110.0
    # 60 + 20 * (1 - 0.25 statutory fallback): the pre-tax charge is taxed before it
    # is added to an after-tax line (was 80.0, the full charge).
    assert result["adjusted_net_income"][2025] == 75.0


def test_addback_uses_the_same_year_effective_rate_and_says_so():
    """A usable same-period effective rate (tax 10 on pre-tax 100 = 10%) beats the fallback."""
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    result = detect_nonrecurring_items(
        {
            "net_income": [(2025, 100.0)],
            "operating_income": [(2025, 120.0)],
            "revenue": [(2025, 1000.0)],
            "restructuring_charges": [(2025, 50.0)],
            "income_tax_expense": [(2025, 10.0)],
            "pretax_income": [(2025, 100.0)],
        }
    )
    assert result["adjusted_operating_income"][2025] == 170.0
    assert result["adjusted_net_income"][2025] == 145.0  # 100 + 50 * (1 - 0.10)
    assert result["restructuring_tax_rate_by_year"] == {2025: 0.1}
    assert result["restructuring_tax_rate_source_by_year"] == {2025: "EFFECTIVE_SAME_PERIOD"}
    assert result["statutory_tax_rate_fallback"] == 0.25


def test_addback_ignores_another_years_effective_rate():
    """Only the charge's own fiscal year supplies the rate; 2024 tax lines do not apply to 2025."""
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    result = detect_nonrecurring_items(
        {
            "net_income": [(2025, 100.0), (2024, 90.0)],
            "operating_income": [(2025, 120.0), (2024, 110.0)],
            "revenue": [(2025, 1000.0), (2024, 980.0)],
            "restructuring_charges": [(2025, 40.0)],
            "income_tax_expense": [(2024, 10.0)],
            "pretax_income": [(2024, 100.0)],
        }
    )
    assert result["adjusted_net_income"][2025] == 130.0  # 100 + 40 * 0.75
    assert result["restructuring_tax_rate_source_by_year"] == {2025: "STATUTORY_FALLBACK"}


def test_unusable_effective_rate_falls_back_to_statutory():
    """A loss year (pre-tax <= 0) and a rate above 50% are denominators, not tax rates."""
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    result = detect_nonrecurring_items(
        {
            "net_income": [(2025, 100.0), (2024, -20.0)],
            "operating_income": [(2025, 120.0), (2024, -10.0)],
            "revenue": [(2025, 1000.0), (2024, 990.0)],
            "restructuring_charges": [(2025, 40.0), (2024, 40.0)],
            "income_tax_expense": [(2025, 60.0), (2024, 5.0)],
            "pretax_income": [(2025, 100.0), (2024, -30.0)],
        }
    )
    assert result["restructuring_tax_rate_source_by_year"] == {
        2024: "STATUTORY_FALLBACK",
        2025: "STATUTORY_FALLBACK",
    }
    assert result["adjusted_net_income"][2025] == 130.0
    assert result["adjusted_net_income"][2024] == 10.0  # -20 + 40 * 0.75


def test_a_year_without_a_charge_is_not_taxed_or_rated():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    result = detect_nonrecurring_items(
        {
            "net_income": [(2025, 100.0), (2024, 90.0)],
            "operating_income": [(2025, 120.0), (2024, 110.0)],
            "revenue": [(2025, 1000.0), (2024, 980.0)],
            "restructuring_charges": [(2025, 40.0)],
        }
    )
    assert result["adjusted_net_income"][2024] == 90.0
    assert list(result["restructuring_tax_rate_by_year"]) == [2025]


def test_old_charge_does_not_flag_today_but_keeps_its_own_year_addback():
    """The window is the three latest fiscal years on file (2023-2025 here)."""
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    result = detect_nonrecurring_items(
        {
            "net_income": [(2025, 100.0), (2009, 80.0)],
            "operating_income": [(2025, 120.0), (2009, 90.0)],
            "revenue": [(2025, 1000.0), (2009, 800.0)],
            "restructuring_charges": [(2009, 8.0)],
        }
    )
    assert result["has_nonrecurring"] is False
    assert result["nonrecurring_flags"] == []
    assert result["nonrecurring_years"] == []
    assert result["nonrecurring_window_first_year"] == 2023
    assert result["nonrecurring_window_last_year"] == 2025
    assert result["nonrecurring_window_years"] == 3
    # History is still adjusted: the 2009 add-back belongs to 2009.
    assert result["adjusted_operating_income"][2009] == 98.0
    assert result["adjusted_net_income"][2009] == 86.0  # 80 + 8 * 0.75
    assert result["restructuring_by_year"] == {2009: 8.0}


def test_window_boundary_is_three_fiscal_years_inclusive():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    base = {
        "net_income": [(2025, 100.0), (2024, 100.0), (2023, 100.0), (2022, 100.0)],
        "operating_income": [(2025, 120.0), (2024, 120.0), (2023, 120.0), (2022, 120.0)],
        "revenue": [(2025, 1000.0), (2024, 1000.0), (2023, 1000.0), (2022, 1000.0)],
    }
    inside = detect_nonrecurring_items({**base, "restructuring_charges": [(2023, 5.0)]})
    outside = detect_nonrecurring_items({**base, "restructuring_charges": [(2022, 5.0)]})
    assert inside["has_nonrecurring"] is True
    assert inside["nonrecurring_years"] == [2023]
    assert outside["has_nonrecurring"] is False


def test_old_oi_spike_and_old_goodwill_writedown_do_not_flag_today():
    from app.valuation.nonrecurring_filter import detect_nonrecurring_items
    result = detect_nonrecurring_items(
        {
            "operating_income": [(2009, 100.0), (2010, 150.0), (2011, 105.0), (2023, 100.0), (2024, 102.0), (2025, 104.0)],
            "revenue": [(2009, 1000.0), (2010, 1030.0), (2011, 1050.0), (2023, 1000.0), (2024, 1010.0), (2025, 1020.0)],
            "net_income": [(2009, 70.0), (2010, 100.0), (2011, 75.0), (2023, 70.0), (2024, 71.0), (2025, 72.0)],
            "goodwill": [(2009, 500.0), (2010, 400.0), (2023, 400.0), (2024, 400.0), (2025, 400.0)],
        }
    )
    assert result["has_nonrecurring"] is False
    assert result["nonrecurring_years"] == []
    assert result["goodwill_impairment_years"] == []
