"""Observed nonpositive earnings and absent earnings require distinct reasons."""

from app.valuation.intrinsic_discipline import _normalized_earnings_power


def _power(rows):
    return _normalized_earnings_power(
        owner_payload={},
        series_map={"owner_earnings": rows, "fcf": rows, "cfo": rows},
        owner_quality_payload={},
        intangible_payload={},
        maintenance_capex_payload={},
    )


def test_intrinsic_observed_losses_report_negative_earnings():
    result = _power([{"year": y, "value": -10.0} for y in [2022, 2023, 2024]])
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["NEGATIVE_NORMALIZED_EARNINGS"]


def test_intrinsic_missing_earnings_remain_missing():
    result = _power([{"year": y, "value": "UNKNOWN"} for y in [2022, 2023, 2024]])
    assert result["normalized_earnings_power_reason_codes"] == ["INSUFFICIENT_NORMALIZED_INPUTS"]


def test_intrinsic_zero_earnings_are_not_labelled_negative():
    result = _power([{"year": y, "value": 0.0} for y in [2022, 2023, 2024]])
    assert "NEGATIVE_NORMALIZED_EARNINGS" not in result["normalized_earnings_power_reason_codes"]
