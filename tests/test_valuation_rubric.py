from app.valuation.fundamentals import UNKNOWN
from app.valuation.rubric import compute_value_first_score


def _fundamentals(*, revenue: float | str) -> dict:
    return {
        "rows": [
            {
                "year": 2025,
                "revenue": revenue,
                "op_margin": 0.20,
                "fcf_margin": 0.15,
                "gross_margin": 0.50,
                "net_debt": 100.0,
            }
        ],
        "derived_signals": {},
    }


def test_positive_net_debt_missing_revenue_is_a_quality_data_gap():
    result = compute_value_first_score(
        fundamentals=_fundamentals(revenue=UNKNOWN),
        valuation={},
    )
    assert result["quality_score"] == 20.0
    assert "QUALITY_MISSING_REVENUE_FOR_LEVERAGE" in result["gaps"]


def test_positive_net_debt_explicit_revenue_scores_leverage():
    result = compute_value_first_score(
        fundamentals=_fundamentals(revenue=1000.0),
        valuation={},
    )
    assert result["quality_score"] == 23.0
    assert "QUALITY_MISSING_REVENUE_FOR_LEVERAGE" not in result["gaps"]
