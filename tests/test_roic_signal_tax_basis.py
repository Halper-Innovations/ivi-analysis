"""app/valuation/valuation_writer.py::_roic_signal must compare an after-tax return to WACC.

The old code compared a pre-tax return to it.
``roic = avg_oi / avg_ic`` (valuation_writer.py:2173-2180) uses operating income
before tax, then ``ratio = roic / _WACC`` (2181) grades it against a discount rate
that is after tax. The same module taxes operating income at ``_TAX_RATE`` (0.21)
for its earnings-power value (valuation_writer.py:1238). Observed: operating income
12 on capital 100 → ``roic_proxy`` 0.12, ratio 1.2, ROIC_MODEST — a business that
earns 9.48% after tax against a 10% hurdle is graded as clearing it. Koller and
Greenwald both define the return as after-tax operating profit over capital.
"""

from __future__ import annotations

import math

from app.valuation.valuation_writer import _roic_signal


def _facts(operating_income: float) -> dict:
    years = (2025, 2024, 2023)
    return {
        "operating_income": [(y, operating_income) for y in years],
        "total_debt": [(y, 40.0) for y in years],
        "equity": [(y, 60.0) for y in years],
    }


def test_the_return_is_after_tax_before_it_meets_the_after_tax_hurdle():
    out = _roic_signal(_facts(12.0))
    assert out["status"] == "OK"
    assert math.isclose(out["roic_proxy"], 0.0948, rel_tol=1e-9)
    assert math.isclose(out["roic_wacc_ratio"], 0.948, rel_tol=1e-9)
    assert out["signal"] == "ROIC_MARGINAL"


def test_a_business_clearing_the_hurdle_after_tax_still_reads_strong():
    out = _roic_signal(_facts(25.0))
    assert math.isclose(out["roic_proxy"], 0.1975, rel_tol=1e-9)
    assert out["signal"] == "ROIC_STRONG"


def test_a_loss_is_not_given_a_tax_shield():
    out = _roic_signal(_facts(-10.0))
    assert math.isclose(out["roic_proxy"], -0.10, rel_tol=1e-9)
    assert out["signal"] == "ROIC_DESTROYING"


def test_the_return_is_taxed_at_the_issuers_own_normalized_rate():
    """Operating income 12 on capital 100, taxed at the issuer's 10% (five years of
    tax 1 on pre-tax 10): NOPAT 10.8, return 0.108, ratio 1.08 -> ROIC_MODEST, and
    the rate and its source are recorded. Without a usable history the 21% fallback
    applies and is recorded as STATUTORY_DEFAULT."""
    facts = _facts(12.0)
    facts["income_tax_expense"] = [(y, 1.0) for y in (2025, 2024, 2023)]
    facts["pretax_income"] = [(y, 10.0) for y in (2025, 2024, 2023)]
    out = _roic_signal(facts)
    assert math.isclose(out["roic_proxy"], 0.108, rel_tol=1e-9)
    assert out["signal"] == "ROIC_MODEST"
    assert out["tax_rate"] == 0.1
    assert out["tax_rate_source"] == "ISSUER"

    fallback = _roic_signal(_facts(12.0))
    assert fallback["tax_rate"] == 0.21
    assert fallback["tax_rate_source"] == "STATUTORY_DEFAULT"
