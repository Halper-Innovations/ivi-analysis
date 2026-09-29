"""Hand-derived, scratch-only regressions for one formula defect."""

from app.valuation import maintenance_capex_discipline


def test_maintenance_depreciation_ratio_accepts_capex_outflow_sign():
    """Depreciation / capex magnitude = 20/abs(-20)=1, a fully known alignment ratio."""
    result = maintenance_capex_discipline.compute_maintenance_capex_discipline(
        "FIXTURE",
        "2025-03-01",
        fundamentals={
            "rows": [
                {"year": 2024, "cfo": 100.0, "revenue": 200.0, "capex": -20.0, "depreciation": 20.0}
            ]
        },
    )
    assert result["depreciation_to_capex_median"] == 1.0


def test_positive_capex_and_zero_denominator_keep_existing_meaning():
    def compute(capex):
        return maintenance_capex_discipline.compute_maintenance_capex_discipline(
            "FIXTURE",
            "2025-03-01",
            fundamentals={"rows": [{"year": 2024, "depreciation": 20.0, "capex": capex}]},
        )

    assert compute(20.0)["depreciation_to_capex_median"] == 1.0
    assert compute(0.0)["depreciation_to_capex_median"] == "UNKNOWN"
