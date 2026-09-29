from __future__ import annotations

from app.patterns.catalog import get_pattern_definition


def _series(metric: str, values: dict[int, float]) -> list[dict]:
    return [
        {"year": year, "value": value, "derived_from": [f"{metric}.{year}"]}
        for year, value in sorted(values.items())
    ]


def test_deferred_revenue_pattern_detects_and_confirms_revenue_acceleration():
    definition = get_pattern_definition("deferred_revenue_leading_indicator")
    assert definition is not None
    data = {
        "revenue": _series("revenue", {2020: 100.0, 2021: 110.0, 2022: 121.0, 2023: 155.0, 2024: 195.0}),
        "deferred_revenue": _series("deferred_revenue", {2020: 80.0, 2021: 112.0, 2022: 151.2, 2023: 190.0, 2024: 220.0}),
    }

    detection = definition.detection_fn(data)
    outcome = definition.outcome_fn(data, detection)

    assert detection["present"] is True
    assert detection["years_detected"] == [2021, 2022]
    assert detection["detection_strength"] > 0
    assert outcome["outcome_confirmed"] is True
    assert outcome["outcome_value"] > 0


def test_deferred_revenue_pattern_does_not_false_positive_on_normal_growth():
    definition = get_pattern_definition("deferred_revenue_leading_indicator")
    assert definition is not None
    data = {
        "revenue": _series("revenue", {2020: 100.0, 2021: 110.0, 2022: 121.0, 2023: 133.1, 2024: 146.4}),
        "deferred_revenue": _series("deferred_revenue", {2020: 90.0, 2021: 98.0, 2022: 108.0, 2023: 119.0, 2024: 131.0}),
    }

    detection = definition.detection_fn(data)

    assert detection["present"] is False
    assert detection["years_detected"] == []


def test_patterns_require_minimum_history_and_metrics():
    definition = get_pattern_definition("capex_to_depreciation_divergence")
    assert definition is not None
    data = {
        "capex": _series("capex", {2023: 100.0, 2024: 120.0}),
        "depreciation": _series("depreciation", {2023: 50.0, 2024: 55.0}),
        "operating_income": _series("operating_income", {2023: 80.0, 2024: 90.0}),
        "revenue": _series("revenue", {2023: 400.0, 2024: 450.0}),
    }

    detection = definition.detection_fn(data)

    assert detection["present"] is False
    assert detection["reason"] == "INSUFFICIENT_HISTORY"
