"""Amortization and revenue must cover the same years in their ratio."""

from app.valuation.intangible_amort import compute_series


def _series(revenues):
    return compute_series(
        ticker="FIXTURE",
        years_data=[
            {"fiscal_year": 2023, "intangible_amortization": 8.0},
            {"fiscal_year": 2024, "d_and_a": 100.0, "capex": 100.0, "intangible_assets": 0.0},
        ],
        revenue_by_year=revenues,
    )


def test_amort_materiality_counts_zero_charge_year_in_same_period_ratio():
    # (8 + 0) / (100 + 100) = .04, below the unchanged .05 threshold.
    result = _series({2023: 100.0, 2024: 100.0})
    assert result.average_addback == 4.0
    assert result.average_addback_pct_of_revenue == 0.04
    assert result.is_materially_distorted is False


def test_amort_without_matching_revenue_cannot_enter_ratio_numerator():
    result = _series({2024: 100.0})
    assert result.average_addback_pct_of_revenue == 0.0
    assert result.is_materially_distorted is False


def test_amort_single_matched_charge_year_retains_material_ratio():
    result = _series({2023: 100.0})
    assert result.average_addback_pct_of_revenue == 0.08
    assert result.is_materially_distorted is True


def test_missing_addback_year_does_not_dilute_known_materiality():
    result = compute_series(
        ticker="FIXTURE",
        years_data=[
            {"fiscal_year": 2023, "intangible_amortization": 8.0},
            {"fiscal_year": 2024},
        ],
        revenue_by_year={2023: 100.0, 2024: 100.0},
    )
    assert result.average_addback == 8.0
    assert result.average_addback_pct_of_revenue == 0.08
    assert result.is_materially_distorted is True


def test_missing_intangible_basis_does_not_count_as_a_known_zero():
    result = compute_series(
        ticker="FIXTURE",
        years_data=[
            {"fiscal_year": 2023, "intangible_amortization": 8.0},
            {"fiscal_year": 2024, "d_and_a": 50.0, "capex": 10.0},
        ],
        revenue_by_year={2023: 100.0, 2024: 100.0},
    )
    assert result.average_addback_pct_of_revenue == 0.08
    assert result.is_materially_distorted is True


def test_reported_zero_amortization_is_not_replaced_with_a_positive_estimate():
    result = compute_series(
        ticker="FIXTURE",
        years_data=[
            {
                "fiscal_year": 2024,
                "intangible_amortization": 0.0,
                "d_and_a": 200.0,
                "capex": 100.0,
                "intangible_assets": 1000.0,
            },
        ],
        revenue_by_year={2024: 100.0},
    )
    assert result.years[0].addback == 0.0
    assert result.years[0].reason == "DIRECT_FROM_AMORTIZATION_OF_INTANGIBLE_ASSETS"
    assert result.average_addback_pct_of_revenue == 0.0
    assert result.is_materially_distorted is False
