from __future__ import annotations

import pytest

import app.autonomous.sector_financial_packets as sfp
from app.autonomous.sector_financial_packets import (
    _effective_tax_rate,
    _gross_margin_year_row,
    _returns_on_capital_metrics,
    _roic_row,
)


# Full ROIC inputs engineered to yield exactly 0.20:
#   tax_rate = 20 / 100 = 0.20 (within [0, 0.50] clamp)
#   nopat    = operating_income * (1 - 0.20) = 100 * 0.80 = 80
#   invested = total_debt + equity - cash    = 300 + 200 - 100 = 400
#   roic     = 80 / 400 = 0.20
_FULL_ROIC_INPUTS_020 = {
    "operating_income": 100.0,
    "income_tax_expense": 20.0,
    "pretax_income": 100.0,
    "total_debt": 300.0,
    "equity": 200.0,
    "cash": 100.0,
}

# Full ROIC inputs engineered to yield exactly 0.18:
#   nopat    = 90 * 0.80 = 72
#   invested = 300 + 200 - 100 = 400
#   roic     = 72 / 400 = 0.18
_FULL_ROIC_INPUTS_018 = {
    "operating_income": 90.0,
    "income_tax_expense": 20.0,
    "pretax_income": 100.0,
    "total_debt": 300.0,
    "equity": 200.0,
    "cash": 100.0,
}


def test_effective_tax_rate_canonical_line_items():
    rate = _effective_tax_rate({"income_tax_expense": 29749.0, "pretax_income": 123485.0})
    assert rate == pytest.approx(0.240914, abs=1e-5)


def test_roic_row_canonical_inputs():
    row = _roic_row(
        {
            "operating_income": 123216.0,
            "income_tax_expense": 29749.0,
            "pretax_income": 123485.0,
            "total_debt": 106629.0,
            "equity": 56950.0,
            "cash": 29943.0,
        }
    )
    assert row["roic"] == pytest.approx(0.6999, rel=0.01)
    assert row["reasons"] == []


def test_roic_row_missing_pretax_flags_tax_rate_missing():
    row = _roic_row(
        {
            "operating_income": 123216.0,
            "income_tax_expense": 29749.0,
            "total_debt": 106629.0,
            "equity": 56950.0,
            "cash": 29943.0,
        }
    )
    assert row["roic"] is None
    assert "TRAILING_EFFECTIVE_TAX_RATE_MISSING" in row["reasons"]


def test_gross_margin_year_row_via_cost_of_revenue():
    row = _gross_margin_year_row(2024, {"revenue": 391035.0, "cost_of_revenue": 210352.0})
    assert row["gross_margin"] == pytest.approx(0.4621, rel=0.001)
    assert row["cost_of_revenue_source"] == "reported_cost_of_revenue"


def test_gross_margin_year_row_prefers_gross_profit():
    row = _gross_margin_year_row(2024, {"revenue": 391035.0, "gross_profit": 180683.0})
    assert row["gross_margin"] == pytest.approx(0.4621, rel=0.001)
    assert row["cost_of_revenue_source"] == "gross_profit"


def test_complete_year_headline_falls_back_to_prior_complete_year(monkeypatch):
    by_year = {
        2023: dict(_FULL_ROIC_INPUTS_020),
        2024: {"operating_income": 10.0},
    }
    monkeypatch.setattr(sfp, "_annual_fact_rows", lambda ticker, **_kwargs: by_year)
    metrics = _returns_on_capital_metrics("TEST", as_of_date=None)
    assert metrics["roic"] == 0.20
    assert "LATEST_YEAR_STUB_USED_PRIOR_COMPLETE_YEAR" in metrics["roic_not_computable_reasons"]


def test_complete_year_headline_uses_latest_when_fully_populated(monkeypatch):
    by_year = {
        2023: dict(_FULL_ROIC_INPUTS_020),
        2024: dict(_FULL_ROIC_INPUTS_018),
    }
    monkeypatch.setattr(sfp, "_annual_fact_rows", lambda ticker, **_kwargs: by_year)
    metrics = _returns_on_capital_metrics("TEST", as_of_date=None)
    assert metrics["roic"] == 0.18
    assert "LATEST_YEAR_STUB_USED_PRIOR_COMPLETE_YEAR" not in metrics["roic_not_computable_reasons"]


def test_complete_year_no_complete_year_yields_none_with_full_trajectory(monkeypatch):
    by_year = {
        2023: {"operating_income": 10.0},
        2024: {"operating_income": 12.0},
    }
    monkeypatch.setattr(sfp, "_annual_fact_rows", lambda ticker, **_kwargs: by_year)
    metrics = _returns_on_capital_metrics("TEST", as_of_date=None)
    assert metrics["roic"] is None
    assert len(metrics["roic_trajectory_5y"]) == 2


def test_roic_row_financial_issuer_uses_equity_basis():
    # Financial issuer (deposits + loans present), no total_debt:
    #   tax_rate = 10 / 50 = 0.20
    #   nopat    = 50 * (1 - 0.20) = 40
    #   invested = equity = 500
    #   roic     = 40 / 500 = 0.08
    row = _roic_row(
        {
            "operating_income": 50.0,
            "income_tax_expense": 10.0,
            "pretax_income": 50.0,
            "equity": 500.0,
            "cash": 100.0,
            "deposits": 800.0,
            "loans": 700.0,
        }
    )
    assert row["roic"] == pytest.approx(0.08, rel=0.01)
    assert row["invested_capital_basis"] == "equity_only_financial"
    # Basis annotation lives in a dedicated field, NOT in the failure-reasons
    # stream (a successful ROIC must never be labeled "not computable").
    assert "FINANCIAL_ISSUER_EQUITY_BASIS" in row["basis_notes"]
    assert "FINANCIAL_ISSUER_EQUITY_BASIS" not in row["reasons"]
    assert row["reasons"] == []


def test_returns_on_capital_financial_issuer_headline_basis_labeled(monkeypatch):
    # Financial issuer (deposits + loans, no total_debt) computes a valid ROIC.
    # The headline metrics dict MUST report the equity-only financial basis at
    # the consumed level, carry the basis annotation in a dedicated field, and
    # NOT contaminate roic_not_computable_reasons with the basis annotation.
    #   tax_rate = 10 / 50 = 0.20
    #   nopat    = 50 * (1 - 0.20) = 40
    #   invested = equity = 500
    #   roic     = 40 / 500 = 0.08
    by_year = {
        2024: {
            "operating_income": 50.0,
            "income_tax_expense": 10.0,
            "pretax_income": 50.0,
            "equity": 500.0,
            "cash": 100.0,
            "deposits": 800.0,
            "loans": 700.0,
        }
    }
    monkeypatch.setattr(sfp, "_annual_fact_rows", lambda ticker, **_kwargs: by_year)
    metrics = _returns_on_capital_metrics("FINL", as_of_date=None)
    assert metrics["roic"] == pytest.approx(0.08, rel=0.01)
    assert metrics["invested_capital_basis"] == "equity_only_financial"
    assert metrics["roic_basis_notes"] == ["FINANCIAL_ISSUER_EQUITY_BASIS"]
    assert "FINANCIAL_ISSUER_EQUITY_BASIS" not in metrics["roic_not_computable_reasons"]
    assert metrics["roic_trajectory_5y"][-1]["invested_capital_basis"] == "equity_only_financial"
    assert metrics["roic_trajectory_5y"][-1]["basis_notes"] == ["FINANCIAL_ISSUER_EQUITY_BASIS"]


def test_roic_row_without_financial_line_items_flags_total_debt_missing():
    # Same inputs WITHOUT financial line_items (no deposits/loans) and no total_debt
    # => operating issuer => TOTAL_DEBT_MISSING and roic None.
    row = _roic_row(
        {
            "operating_income": 50.0,
            "income_tax_expense": 10.0,
            "pretax_income": 50.0,
            "equity": 500.0,
            "cash": 100.0,
        }
    )
    assert row["roic"] is None
    assert "TOTAL_DEBT_MISSING" in row["reasons"]


def test_gross_margin_year_row_financial_issuer_not_applicable():
    row = _gross_margin_year_row(
        2024,
        {
            "revenue": 1000.0,
            "deposits": 800.0,
            "loans": 700.0,
            "provision_for_credit_losses": 5.0,
        },
    )
    assert row["gross_margin"] is None
    assert row["not_computable_reasons"] == ["GROSS_MARGIN_NOT_APPLICABLE_FINANCIAL_ISSUER"]
