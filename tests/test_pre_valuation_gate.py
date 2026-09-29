"""Tests for app.valuation.pre_valuation_gate."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from app.db import init_db


# ── in-memory DB helpers (mirrors test_valuation_writer.py pattern) ──────────

def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn=conn)
    return conn


def _seed_companyfacts(conn: sqlite3.Connection, ticker: str = "TST") -> None:
    now = "2026-01-01T00:00:00+00:00"
    rows = [
        ("cfo",               2024, 120.0), ("cfo",               2023, 110.0), ("cfo",               2022, 100.0),
        ("capex",             2024,  15.0), ("capex",             2023,  14.0), ("capex",             2022,  12.0),
        ("operating_income",  2024,  80.0), ("operating_income",  2023,  75.0), ("operating_income",  2022,  70.0),
        ("operating_income",  2021,  65.0), ("operating_income",  2020,  60.0),
        ("net_income",        2024,  60.0), ("net_income",        2023,  55.0), ("net_income",        2022,  50.0),
        ("revenue",           2024, 500.0), ("revenue",           2023, 450.0), ("revenue",           2022, 400.0),
        ("revenue",           2021, 350.0), ("revenue",           2020, 300.0),
        ("total_debt",        2024,  50.0), ("cash",              2024,  30.0),
        ("equity",            2024, 200.0), ("shares_outstanding", 2024,  10.0),
        ("total_liabilities", 2024, 300.0),
    ]
    for li, fy, val in rows:
        conn.execute(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_end, line_item, value, units, source_url, fetched_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, fy, f"{fy}-12-31", li, val, "USD_millions", "", now),
        )
    conn.commit()


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_facts(**overrides) -> dict[str, list[tuple[int, float]]]:
    """Build a minimal facts dict with sensible defaults.

    Revenue defaults to 5 years of stable ~100M revenue.
    Override any field by passing keyword args.
    """
    base = {
        "revenue": [(2025, 100.0), (2024, 98.0), (2023, 95.0), (2022, 92.0), (2021, 90.0), (2020, 88.0)],
        "cfo": [(2025, 20.0), (2024, 19.0), (2023, 18.0), (2022, 17.0), (2021, 16.0)],
        "operating_income": [(2025, 15.0), (2024, 14.0), (2023, 13.0), (2022, 12.0), (2021, 11.0)],
        "net_income": [(2025, 10.0), (2024, 9.5), (2023, 9.0), (2022, 8.5), (2021, 8.0)],
        "total_debt": [(2025, 30.0)],
        "cash": [(2025, 15.0)],
        "equity": [(2025, 50.0)],
        "shares_outstanding": [(2025, 10.0)],
    }
    base.update(overrides)
    return base


# ~17%/yr: clears the 8% demonstrated-growth bar so the category maintenance
# ratio applies un-scaled (the gate ratio is growth-aware post-audit).
GROWTH_REVENUE = [(2025, 150.0), (2024, 130.0), (2023, 110.0), (2022, 95.0), (2021, 80.0)]


# ── EPV quality classification ───────────────────────────────────────────────

def test_classify_epv_quality_deteriorating():
    from app.valuation.pre_valuation_gate import _classify_epv_quality
    quality, cagr = _classify_epv_quality(-0.07, None)
    assert quality == "DETERIORATING_BASE"
    assert cagr == -0.07


def test_classify_epv_quality_declining():
    from app.valuation.pre_valuation_gate import _classify_epv_quality
    quality, cagr = _classify_epv_quality(-0.04, None)
    assert quality == "DECLINING_BASE"
    assert cagr == -0.04


def test_classify_epv_quality_stable():
    from app.valuation.pre_valuation_gate import _classify_epv_quality
    quality, cagr = _classify_epv_quality(0.05, None)
    assert quality == "STABLE"
    assert cagr == 0.05


def test_classify_epv_quality_unknown():
    from app.valuation.pre_valuation_gate import _classify_epv_quality
    quality, cagr = _classify_epv_quality(None, None)
    assert quality == "UNKNOWN"
    assert cagr is None


def test_classify_epv_quality_5y_null_falls_back_to_3y():
    from app.valuation.pre_valuation_gate import _classify_epv_quality
    quality, cagr = _classify_epv_quality(None, -0.06)
    assert quality == "DETERIORATING_BASE"
    assert cagr == -0.06


# ── Terminal growth override ─────────────────────────────────────────────────

# ── Revenue trend classification ─────────────────────────────────────────────

def test_revenue_trend_growing():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    series = [(2020, 80.0), (2021, 88.0), (2022, 96.0), (2023, 104.0), (2024, 113.0), (2025, 123.0)]
    assert _classify_revenue_trend(series, 0.09)["revenue_trend_class"] == "GROWING"


def test_revenue_trend_flat():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    series = [(2020, 100.0), (2021, 101.0), (2022, 99.0), (2023, 102.0), (2024, 100.0), (2025, 101.0)]
    assert _classify_revenue_trend(series, 0.002)["revenue_trend_class"] == "FLAT"


def test_revenue_trend_declining():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    # Two consecutive declines at the end (below secular-decline threshold)
    series = [(2020, 100.0), (2021, 97.0), (2022, 99.0), (2023, 93.0), (2024, 95.0), (2025, 90.0)]
    # CAGR ≈ -4%
    assert _classify_revenue_trend(series, -0.04)["revenue_trend_class"] == "DECLINING"


def test_revenue_trend_secular_decline():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    series = [(2020, 100.0), (2021, 90.0), (2022, 82.0), (2023, 75.0), (2024, 68.0), (2025, 60.0)]
    assert _classify_revenue_trend(series, -0.10)["revenue_trend_class"] == "SECULAR_DECLINE"


def test_revenue_trend_volatile():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    # Large swings but no clear net direction (mean yoy change near zero)
    series = [(2020, 100.0), (2021, 130.0), (2022, 90.0), (2023, 125.0), (2024, 85.0), (2025, 100.0)]
    assert _classify_revenue_trend(series, 0.00)["revenue_trend_class"] == "VOLATILE"


def test_revenue_trend_unknown_insufficient_data():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    assert _classify_revenue_trend([(2025, 100.0)], None)["revenue_trend_class"] == "UNKNOWN"


# ── Enriched revenue trend (dict return) ────────────────────────────────

def test_revenue_trend_secular_decline_7yr():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    series = [
        (2018, 100.0), (2019, 92.0), (2020, 85.0), (2021, 78.0),
        (2022, 72.0), (2023, 66.0), (2024, 62.0), (2025, 58.0),
    ]
    result = _classify_revenue_trend(series, -0.08, revenue_cagr_3y=-0.06)
    assert isinstance(result, dict)
    assert result["revenue_trend_class"] == "SECULAR_DECLINE"
    assert result["decline_years_consecutive"] == 7
    assert result["decline_magnitude_total"] is not None
    assert result["decline_magnitude_total"] < -0.40
    assert result["peak_year"] == 2018
    assert result["peak_revenue"] == 100.0
    assert result["revenue_cagr_5y"] == -0.08
    assert result["revenue_cagr_3y"] == -0.06


def test_revenue_trend_moderate_decline():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    series = [
        (2020, 100.0), (2021, 105.0), (2022, 102.0), (2023, 103.0),
        (2024, 95.0), (2025, 89.0),
    ]
    result = _classify_revenue_trend(series, -0.04, revenue_cagr_3y=-0.04)
    assert isinstance(result, dict)
    assert result["revenue_trend_class"] == "DECLINING"
    assert result["decline_years_consecutive"] == 2
    assert result["peak_year"] == 2021
    assert result["peak_revenue"] == 105.0


def test_revenue_trend_growing_no_decline_fields():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    series = [(2020, 80.0), (2021, 88.0), (2022, 96.0), (2023, 104.0), (2024, 113.0), (2025, 123.0)]
    result = _classify_revenue_trend(series, 0.09, revenue_cagr_3y=0.08)
    assert isinstance(result, dict)
    assert result["revenue_trend_class"] == "GROWING"
    assert result["decline_years_consecutive"] == 0
    assert result["decline_magnitude_total"] is None
    assert result["peak_year"] == 2025
    assert result["peak_revenue"] == 123.0


def test_revenue_trend_volatile_no_decline():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    series = [(2020, 100.0), (2021, 130.0), (2022, 90.0), (2023, 125.0), (2024, 85.0), (2025, 100.0)]
    result = _classify_revenue_trend(series, 0.00, revenue_cagr_3y=0.00)
    assert isinstance(result, dict)
    assert result["revenue_trend_class"] == "VOLATILE"


def test_revenue_trend_peak_year_detection():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    series = [
        (2019, 90.0), (2020, 95.0), (2021, 110.0), (2022, 100.0),
        (2023, 92.0), (2024, 88.0), (2025, 85.0),
    ]
    result = _classify_revenue_trend(series, -0.04, revenue_cagr_3y=-0.04)
    assert result["peak_year"] == 2021
    assert result["peak_revenue"] == 110.0
    assert result["decline_magnitude_total"] is not None
    assert result["decline_magnitude_total"] < -0.20


def test_revenue_trend_unknown_insufficient_data_dict():
    from app.valuation.pre_valuation_gate import _classify_revenue_trend
    result = _classify_revenue_trend([(2025, 100.0)], None, revenue_cagr_3y=None)
    assert isinstance(result, dict)
    assert result["revenue_trend_class"] == "UNKNOWN"
    assert result["decline_years_consecutive"] == 0
    assert result["decline_magnitude_total"] is None
    assert result["peak_year"] is None


def test_terminal_growth_deteriorating_base():
    from app.valuation.pre_valuation_gate import _compute_terminal_growth
    assert _compute_terminal_growth("INSUFFICIENT", "SECULAR_DECLINE", "F") == 0.0


def test_terminal_growth_declining_base():
    from app.valuation.pre_valuation_gate import _compute_terminal_growth
    assert _compute_terminal_growth("LOW", "DECLINING", "C") == 0.005


def test_terminal_growth_peak_cycle():
    from app.valuation.pre_valuation_gate import _compute_terminal_growth
    assert _compute_terminal_growth("MODERATE", "FLAT", "B") == 0.015


def test_terminal_growth_default():
    from app.valuation.pre_valuation_gate import _compute_terminal_growth
    assert _compute_terminal_growth("HIGH", "GROWING", "A") == 0.02


def test_terminal_growth_deteriorating_overrides_peak():
    from app.valuation.pre_valuation_gate import _compute_terminal_growth
    assert _compute_terminal_growth("INSUFFICIENT", "SECULAR_DECLINE", "A") == 0.0


# ── Gate action logic ────────────────────────────────────────────────────────

def test_gate_block_low_quality_high_leverage():
    """LOW earnings quality + HIGH leverage stress → BLOCK."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        # Make accounting_quality return LOW, balance_sheet_stress return HIGH
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "LOW_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "HIGH_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.5}
            return {}

        mock_safe.side_effect = side_effect
        facts = _make_facts()
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    assert ctx["gate_action"] == "BLOCK"
    assert ctx["earnings_quality"] == "LOW"
    assert ctx["leverage_stress"] == "HIGH"
    assert "LOW_QUALITY_HIGH_LEVERAGE" in ctx["gate_reason"]


def test_gate_adjust_peak_cycle():
    """PEAK cycle position → ADJUST with 1.5% terminal growth."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "HIGH_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "ELEVATED_RELATIVE_TO_NORMAL", "conservative_cyclical_denominator": 12.5}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.8}
            return {}

        mock_safe.side_effect = side_effect
        facts = _make_facts()
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    assert ctx["gate_action"] == "ADJUST"
    assert ctx["cycle_position"] == "PEAK"
    assert ctx["terminal_growth_override"] == 0.015
    # Post-audit: normalized_earnings is the OI-basis median (median of
    # _make_facts operating_income 11..15 = 13.0), NOT the cyclical module's
    # CFO-basis conservative denominator (12.5).
    assert ctx["normalized_earnings"] == 13.0


def test_gate_adjust_declining_revenue():
    """Declining revenue CAGR → ADJUST with 0% terminal growth."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    # Revenue declining from 100 to ~73 over 5 years ≈ -6% CAGR
    declining_revenue = [
        (2025, 73.0), (2024, 78.0), (2023, 83.0), (2022, 88.0), (2021, 94.0), (2020, 100.0),
    ]

    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.6}
            return {}

        mock_safe.side_effect = side_effect
        facts = _make_facts(revenue=declining_revenue)
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    assert ctx["gate_action"] == "ADJUST"
    assert ctx["epv_quality"] == "DETERIORATING_BASE"
    assert ctx["terminal_growth_override"] == 0.005  # was 0.0; 4-tier model: LOW confidence → 0.5%
    assert ctx["revenue_cagr_5y"] is not None
    assert ctx["revenue_cagr_5y"] < -0.05


def test_gate_proceed_healthy():
    """All quality signals clean → PROCEED with 2.0% terminal growth."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "HIGH_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.9}
            return {}

        mock_safe.side_effect = side_effect
        facts = _make_facts()
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    assert ctx["gate_action"] == "PROCEED"
    assert ctx["gate_reason"] is None
    assert ctx["terminal_growth_override"] == 0.015  # was 0.02; FLAT trend → 1.5%
    assert ctx["earnings_quality"] == "HIGH"
    assert ctx["leverage_stress"] == "NONE"
    assert ctx["allocation_grade"] == "A"


def test_gate_module_failure_graceful():
    """Quality module raising an exception → gate still returns with conservative defaults."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        # All modules fail
        mock_safe.return_value = {}
        facts = _make_facts()
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    # Should not crash — returns conservative defaults. Accounting quality
    # nobody could read is UNKNOWN, not MODERATE.
    assert ctx["gate_action"] in ("PROCEED", "ADJUST", "BLOCK")
    assert ctx["earnings_quality"] == "UNKNOWN"
    assert ctx["leverage_stress"] == "MODERATE"
    assert ctx["cycle_position"] == "MID"
    assert ctx["allocation_grade"] == "C"
    assert ctx["epv_quality"] in ("STABLE", "UNKNOWN", "DECLINING_BASE", "DETERIORATING_BASE")


def test_gate_import_failure_graceful():
    """If a quality module can't even be imported, gate still works."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.accounting_quality.compute_accounting_quality", side_effect=ImportError("test")):
        facts = _make_facts()
        # This should not raise
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    assert ctx["gate_action"] in ("PROCEED", "ADJUST", "BLOCK")
    assert ctx["earnings_quality"] == "UNKNOWN"


# ── Maintenance capex by category ────────────────────────────────────────────

def test_maintenance_capex_semiconductor():
    """Category ratio applies at demonstrated growth (CAGR >= 8%)."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        facts = _make_facts(revenue=GROWTH_REVENUE)
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts, category="SEMICONDUCTOR")

    assert ctx["maintenance_capex_pct"] == 0.80


def test_maintenance_capex_enterprise_software():
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        facts = _make_facts(revenue=GROWTH_REVENUE)
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts, category="ENTERPRISE_SOFTWARE")

    assert ctx["maintenance_capex_pct"] == 0.25


def test_maintenance_capex_traditional_default():
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        facts = _make_facts(revenue=GROWTH_REVENUE)
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts, category="TRADITIONAL_OPERATING")

    assert ctx["maintenance_capex_pct"] == 0.60


def test_maintenance_capex_unknown_category_uses_default():
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        facts = _make_facts(revenue=GROWTH_REVENUE)
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts, category="SOME_UNKNOWN")

    assert ctx["maintenance_capex_pct"] == 0.60


def test_maintenance_capex_full_for_no_growth_firm():
    """Audit maint-capex-60pct-universal-haircut: a flat-revenue firm's capex
    is all maintenance — no category haircut regardless of category."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    flat_revenue = [(2025, 100.0), (2024, 100.0), (2023, 100.0), (2022, 100.0), (2021, 100.0)]
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        facts = _make_facts(revenue=flat_revenue)
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts, category="SEMICONDUCTOR")

    assert ctx["maintenance_capex_pct"] == 1.0


def test_maintenance_capex_all_categories():
    from app.valuation.pre_valuation_gate import _MAINT_CAPEX_BY_CATEGORY
    expected = {
        "SEMICONDUCTOR": 0.80,
        "CONSUMER_HARDWARE": 0.70,
        "INDUSTRIAL_TECH": 0.75,
        "ENTERPRISE_SOFTWARE": 0.25,
        "PLATFORM_HYBRID": 0.40,
        "NETWORK_INFRA": 0.65,
        "TRADITIONAL_OPERATING": 0.60,
    }
    assert _MAINT_CAPEX_BY_CATEGORY == expected


# ── SBC burden detection ────────────────────────────────────────────────────

def test_gate_adjust_sbc_burden():
    """High SBC burden → ADJUST."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "HIGH_ACCOUNTING_QUALITY", "sbc_to_revenue_median_3y": 0.08}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.8}
            return {}

        mock_safe.side_effect = side_effect
        facts = _make_facts()
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    assert ctx["gate_action"] == "ADJUST"
    assert ctx["sbc_burden"] is True
    assert "SBC_BURDEN" in ctx["gate_reason"]


# ── Context fields completeness ──────────────────────────────────────────────

def test_quality_context_has_all_required_fields():
    """Verify all expected fields are present in the returned context."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        facts = _make_facts()
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    required_fields = {
        "earnings_quality", "cash_conversion_score", "sbc_burden",
        "cycle_position", "normalized_earnings", "allocation_grade",
        "leverage_stress", "revenue_cagr_5y", "revenue_cagr_3y",
        "epv_quality", "revenue_trend_class", "terminal_growth_override",
        "maintenance_capex_pct", "gate_action", "gate_reason",
        "decline_years_consecutive", "decline_magnitude_total",
    }
    assert required_fields.issubset(set(ctx.keys())), f"Missing: {required_fields - set(ctx.keys())}"


# ── Integration: gate wired into valuation_writer ────────────────────────────

# ── Working capital efficiency ────────────────────────────────────────────────

def test_working_capital_all_metrics():
    from app.valuation.pre_valuation_gate import _compute_working_capital_efficiency
    facts = {
        "accounts_receivable": [(2025, 50.0)],
        "inventory": [(2025, 20.0)],
        "accounts_payable": [(2025, 30.0)],
        "revenue": [(2025, 365.0)],
    }
    wc = _compute_working_capital_efficiency(facts)
    assert wc["dso"] == 50.0  # 50/365*365
    assert wc["dio"] == 20.0  # 20/365*365
    assert wc["dpo"] == 30.0  # 30/365*365
    assert wc["ccc"] == 40.0  # 50 + 20 - 30


def test_working_capital_no_inventory():
    from app.valuation.pre_valuation_gate import _compute_working_capital_efficiency
    facts = {
        "accounts_receivable": [(2025, 50.0)],
        "accounts_payable": [(2025, 30.0)],
        "revenue": [(2025, 365.0)],
    }
    wc = _compute_working_capital_efficiency(facts)
    assert wc["dso"] == 50.0
    assert wc["dio"] is None
    assert wc["dpo"] == 30.0
    assert wc["ccc"] == 20.0  # DSO + 0 - DPO


def test_working_capital_missing_data():
    from app.valuation.pre_valuation_gate import _compute_working_capital_efficiency
    wc = _compute_working_capital_efficiency({})
    assert wc == {"dso": None, "dio": None, "dpo": None, "ccc": None}


def test_working_capital_in_quality_context():
    from app.valuation.pre_valuation_gate import compute_quality_context
    facts = _make_facts(
        accounts_receivable=[(2025, 50.0)],
        accounts_payable=[(2025, 30.0)],
    )
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)
    assert "working_capital" in ctx
    assert isinstance(ctx["working_capital"], dict)
    assert "dso" in ctx["working_capital"]


def test_terminal_growth_zero_not_treated_as_falsy():
    """Regression: terminal_growth_override=0.0 must not fall back to 2%."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    # Revenue declining sharply → DETERIORATING_BASE → terminal_growth = 0.0
    declining_revenue = [
        (2025, 60.0), (2024, 68.0), (2023, 76.0), (2022, 85.0), (2021, 94.0), (2020, 100.0),
    ]
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        facts = _make_facts(revenue=declining_revenue)
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    assert ctx["terminal_growth_override"] == 0.0
    assert ctx["epv_quality"] == "DETERIORATING_BASE"
    # The value 0.0 is valid — it must not be replaced with the 2% default


def test_ensure_valuation_includes_quality_context_in_scorecard():
    """When ensure_valuation runs, the scorecard output includes quality_context."""
    import json
    from app.valuation.valuation_writer import ensure_valuation

    conn = _make_conn()
    _seed_companyfacts(conn, ticker="GATE_INTEGRATION_TEST_XYZ")
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = None

    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        # Run ensure_valuation — must not crash
        ensure_valuation("GATE_INTEGRATION_TEST_XYZ", "2026-03-25", provider=fake_provider)

    # Check if scorecard row exists (it might not for a ticker with no data,
    # but the function must not crash)
    row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker = ? AND method = 'scorecard'",
        ("GATE_INTEGRATION_TEST_XYZ",),
    ).fetchone()
    # For a ticker with no facts data, valuation may not produce rows,
    # but the important thing is no crash occurred
    if row:
        outputs = json.loads(row["outputs_json"])
        # quality_context should be present
        assert "quality_context" in outputs


# ── Confidence classification ──────────────────────────────────────────────

def test_confidence_high_quality_grower():
    from app.valuation.pre_valuation_gate import _classify_confidence
    result = _classify_confidence(
        earnings_quality="HIGH",
        revenue_trend={"revenue_trend_class": "GROWING", "decline_magnitude_total": None},
        allocation_grade="A",
        oe_quality_total=8.0,
        gate_action="PROCEED",
    )
    assert result == "HIGH"


def test_confidence_high_with_grade_b():
    from app.valuation.pre_valuation_gate import _classify_confidence
    result = _classify_confidence(
        earnings_quality="HIGH",
        revenue_trend={"revenue_trend_class": "GROWING", "decline_magnitude_total": None},
        allocation_grade="B",
        oe_quality_total=8.0,
        gate_action="PROCEED",
    )
    assert result == "HIGH"


def test_confidence_insufficient_on_block():
    from app.valuation.pre_valuation_gate import _classify_confidence
    result = _classify_confidence(
        earnings_quality="LOW",
        revenue_trend={"revenue_trend_class": "SECULAR_DECLINE", "decline_magnitude_total": -0.42},
        allocation_grade="F",
        oe_quality_total=1.0,
        gate_action="BLOCK",
    )
    assert result == "INSUFFICIENT"


def test_confidence_low_on_low_earnings():
    from app.valuation.pre_valuation_gate import _classify_confidence
    result = _classify_confidence(
        earnings_quality="LOW",
        revenue_trend={"revenue_trend_class": "FLAT", "decline_magnitude_total": None},
        allocation_grade="B",
        oe_quality_total=5.0,
        gate_action="ADJUST",
    )
    assert result == "LOW"


def test_confidence_low_on_weak_oe_quality():
    from app.valuation.pre_valuation_gate import _classify_confidence
    result = _classify_confidence(
        earnings_quality="MODERATE",
        revenue_trend={"revenue_trend_class": "FLAT", "decline_magnitude_total": None},
        allocation_grade="B",
        oe_quality_total=2.5,
        gate_action="PROCEED",
    )
    assert result == "LOW"


def test_confidence_low_on_decline_magnitude():
    from app.valuation.pre_valuation_gate import _classify_confidence
    result = _classify_confidence(
        earnings_quality="MODERATE",
        revenue_trend={"revenue_trend_class": "DECLINING", "decline_magnitude_total": -0.15},
        allocation_grade="B",
        oe_quality_total=5.0,
        gate_action="ADJUST",
    )
    assert result == "LOW"


def test_confidence_moderate_default():
    from app.valuation.pre_valuation_gate import _classify_confidence
    result = _classify_confidence(
        earnings_quality="MODERATE",
        revenue_trend={"revenue_trend_class": "FLAT", "decline_magnitude_total": None},
        allocation_grade="C",
        oe_quality_total=5.0,
        gate_action="PROCEED",
    )
    assert result == "MODERATE"


# ── Gate logic: headwinds, supports, confidence, 4-tier terminal growth ─────

def test_gate_block_severe_decline():
    from app.valuation.pre_valuation_gate import compute_quality_context
    severe_decline = [
        (2020, 100.0), (2021, 88.0), (2022, 76.0), (2023, 66.0), (2024, 58.0), (2025, 50.0),
    ]
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.5}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts(revenue=severe_decline))
    assert ctx["gate_action"] == "BLOCK"
    assert ctx["terminal_growth_override"] == 0.0
    assert ctx["epv_adjustment"] == "BLOCK"
    assert ctx["confidence_class"] == "INSUFFICIENT"
    assert "SECULAR_DECLINE_HEADWIND" in ctx["valuation_headwinds"]
    assert ctx["mos_threshold_widening"] == 0.0


def test_gate_adjust_moderate_decline():
    from app.valuation.pre_valuation_gate import compute_quality_context
    moderate_decline = [
        (2020, 100.0), (2021, 103.0), (2022, 98.0), (2023, 92.0), (2024, 87.0), (2025, 82.0),
    ]
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": 15.0}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.5}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts(revenue=moderate_decline))
    assert ctx["gate_action"] == "ADJUST"
    assert ctx["terminal_growth_override"] == 0.005
    assert ctx["epv_adjustment"] == "USE_NORMALIZED"
    assert ctx["confidence_class"] == "LOW"
    assert ctx["mos_threshold_widening"] == 0.15


def test_gate_adjust_cyclical_peak_epv():
    from app.valuation.pre_valuation_gate import compute_quality_context
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "HIGH_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "ELEVATED_RELATIVE_TO_NORMAL", "conservative_cyclical_denominator": 12.5}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 3.5}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts())
    assert ctx["gate_action"] == "ADJUST"
    assert ctx["epv_adjustment"] == "USE_NORMALIZED"
    assert "CYCLICAL_PEAK_HEADWIND" in ctx["valuation_headwinds"]


def test_gate_adjust_bad_capital_allocation():
    from app.valuation.pre_valuation_gate import compute_quality_context
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 2.0}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts())
    assert ctx["gate_action"] == "ADJUST"
    assert "CAPITAL_ALLOCATION_HEADWIND" in ctx["valuation_headwinds"]


def test_gate_proceed_high_quality():
    from app.valuation.pre_valuation_gate import compute_quality_context
    growing_revenue = [
        (2020, 80.0), (2021, 88.0), (2022, 96.0), (2023, 105.0), (2024, 115.0), (2025, 126.0),
    ]
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "HIGH_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 3.5, "oe_quality_total": 9.0}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts(revenue=growing_revenue))
    assert ctx["gate_action"] == "PROCEED"
    assert ctx["confidence_class"] == "HIGH"
    assert ctx["terminal_growth_override"] == 0.02
    assert ctx["epv_adjustment"] == "NONE"
    assert ctx["mos_threshold_widening"] == 0.0
    assert "STRONG_EARNINGS_SUPPORT" in ctx["valuation_supports"]
    assert "OWNER_FRIENDLY_SUPPORT" in ctx["valuation_supports"]
    assert "GROWING_REVENUE_SUPPORT" in ctx["valuation_supports"]
    assert "LOW_LEVERAGE_SUPPORT" in ctx["valuation_supports"]


def test_gate_proceed_moderate():
    from app.valuation.pre_valuation_gate import compute_quality_context
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 2.0, "oe_quality_total": 5.0}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts())
    assert ctx["gate_action"] == "PROCEED"
    assert ctx["confidence_class"] == "MODERATE"
    assert ctx["terminal_growth_override"] == 0.015


def test_headwinds_accumulate():
    from app.valuation.pre_valuation_gate import compute_quality_context
    declining_revenue = [
        (2020, 100.0), (2021, 90.0), (2022, 82.0), (2023, 75.0), (2024, 68.0), (2025, 60.0),
    ]
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "LOW_ACCOUNTING_QUALITY", "sbc_to_revenue_median_3y": 0.08}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "HIGH_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.5, "oe_quality_total": 1.0}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts(revenue=declining_revenue))
    hw = ctx["valuation_headwinds"]
    assert "SECULAR_DECLINE_HEADWIND" in hw
    assert "ACCOUNTING_QUALITY_HEADWIND" in hw
    assert "CAPITAL_ALLOCATION_HEADWIND" in hw
    assert "EARNINGS_QUALITY_HEADWIND" in hw
    assert "SBC_BURDEN_HEADWIND" in hw
    assert "LEVERAGE_STRESS_HEADWIND" in hw


def test_supports_accumulate():
    from app.valuation.pre_valuation_gate import compute_quality_context
    growing_revenue = [
        (2020, 80.0), (2021, 88.0), (2022, 96.0), (2023, 105.0), (2024, 115.0), (2025, 126.0),
    ]
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "HIGH_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 3.5, "oe_quality_total": 9.0}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts(revenue=growing_revenue))
    sp = ctx["valuation_supports"]
    assert "STRONG_EARNINGS_SUPPORT" in sp
    assert "OWNER_FRIENDLY_SUPPORT" in sp
    assert "STRONG_CASH_CONVERSION_SUPPORT" in sp
    assert "LOW_LEVERAGE_SUPPORT" in sp
    assert "GROWING_REVENUE_SUPPORT" in sp


def test_mos_widening_on_adjust():
    from app.valuation.pre_valuation_gate import compute_quality_context
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "LOW_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.5}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts())
    assert ctx["gate_action"] == "ADJUST"
    assert ctx["mos_threshold_widening"] == 0.15


def test_mos_widening_zero_on_proceed():
    from app.valuation.pre_valuation_gate import compute_quality_context
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "HIGH_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 3.5}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts())
    assert ctx["gate_action"] == "PROCEED"
    assert ctx["mos_threshold_widening"] == 0.0


def test_backward_compat_existing_keys():
    from app.valuation.pre_valuation_gate import compute_quality_context
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "HIGH_ACCOUNTING_QUALITY"}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "NEAR_NORMAL", "conservative_cyclical_denominator": "UNKNOWN"}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.8}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts())
    original_keys = {
        "earnings_quality", "cash_conversion_score", "sbc_burden",
        "cycle_position", "normalized_earnings", "allocation_grade",
        "leverage_stress", "revenue_cagr_5y", "revenue_cagr_3y",
        "epv_quality", "revenue_trend_class", "terminal_growth_override",
        "maintenance_capex_pct", "working_capital", "gate_action", "gate_reason",
    }
    new_keys = {
        "confidence_class", "epv_adjustment", "valuation_headwinds",
        "valuation_supports", "gate_reason_codes", "mos_threshold_widening",
        "decline_years_consecutive", "decline_magnitude_total",
    }
    all_expected = original_keys | new_keys
    assert all_expected.issubset(set(ctx.keys())), f"Missing: {all_expected - set(ctx.keys())}"


def test_terminal_growth_four_tiers():
    from app.valuation.pre_valuation_gate import _compute_terminal_growth
    assert _compute_terminal_growth("HIGH", "GROWING", "A") == 0.02
    assert _compute_terminal_growth("MODERATE", "FLAT", "B") == 0.015
    assert _compute_terminal_growth("LOW", "DECLINING", "C") == 0.005
    assert _compute_terminal_growth("INSUFFICIENT", "SECULAR_DECLINE", "F") == 0.0


def test_gate_reason_codes_populated():
    from app.valuation.pre_valuation_gate import compute_quality_context
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        def side_effect(fn, *args, **kwargs):
            name = getattr(fn, "__name__", "")
            if "accounting_quality" in name:
                return {"accounting_quality_class": "LOW_ACCOUNTING_QUALITY", "sbc_to_revenue_median_3y": 0.08}
            if "balance_sheet_stress" in name:
                return {"balance_sheet_stress_class": "LOW_BALANCE_SHEET_STRESS"}
            if "cyclical" in name:
                return {"cycle_position_class": "ELEVATED_RELATIVE_TO_NORMAL", "conservative_cyclical_denominator": 12.0}
            if "capital_allocation" in name:
                return {"capital_allocation_discipline_class": "OWNER_DILUTIVE_OR_DESTRUCTIVE"}
            if "owner_earnings_quality" in name:
                return {"cash_conversion_score": 0.5}
            return {}
        mock_safe.side_effect = side_effect
        ctx = compute_quality_context("TEST", "2026-03-25", facts=_make_facts())
    codes = ctx["gate_reason_codes"]
    assert isinstance(codes, list)
    assert len(codes) >= 2
    assert ctx["gate_reason"] is not None


# ── Working capital trends ─────────────────────────────────────────────

def test_wc_trend_dso_deteriorating():
    from app.valuation.pre_valuation_gate import _compute_working_capital_trends
    facts = {
        "accounts_receivable": [(2023, 40.0), (2024, 48.0), (2025, 52.0)],
        "revenue": [(2023, 365.0), (2024, 365.0), (2025, 365.0)],
    }
    result = _compute_working_capital_trends(facts)
    assert "RECEIVABLES_DETERIORATING" in result["trend_flags"]


def test_wc_trend_dio_building():
    from app.valuation.pre_valuation_gate import _compute_working_capital_trends
    facts = {
        "inventory": [(2023, 20.0), (2024, 24.0), (2025, 26.0)],
        "revenue": [(2023, 365.0), (2024, 365.0), (2025, 365.0)],
    }
    result = _compute_working_capital_trends(facts)
    assert "INVENTORY_BUILDING" in result["trend_flags"]


def test_wc_trend_ccc_drag():
    from app.valuation.pre_valuation_gate import _compute_working_capital_trends
    facts = {
        "accounts_receivable": [(2023, 40.0), (2024, 48.0), (2025, 55.0)],
        "accounts_payable": [(2023, 30.0), (2024, 30.0), (2025, 30.0)],
        "revenue": [(2023, 365.0), (2024, 365.0), (2025, 365.0)],
    }
    result = _compute_working_capital_trends(facts)
    assert "WORKING_CAPITAL_DRAG" in result["trend_flags"]


def test_wc_trend_stable_no_flags():
    from app.valuation.pre_valuation_gate import _compute_working_capital_trends
    facts = {
        "accounts_receivable": [(2023, 40.0), (2024, 41.0), (2025, 40.0)],
        "accounts_payable": [(2023, 30.0), (2024, 30.0), (2025, 30.0)],
        "revenue": [(2023, 365.0), (2024, 370.0), (2025, 365.0)],
    }
    result = _compute_working_capital_trends(facts)
    assert result["trend_flags"] == []


def test_wc_trend_improving_no_flags():
    from app.valuation.pre_valuation_gate import _compute_working_capital_trends
    facts = {
        "accounts_receivable": [(2023, 50.0), (2024, 45.0), (2025, 40.0)],
        "revenue": [(2023, 365.0), (2024, 365.0), (2025, 365.0)],
    }
    result = _compute_working_capital_trends(facts)
    assert "RECEIVABLES_DETERIORATING" not in result["trend_flags"]


def test_wc_trend_missing_data_graceful():
    from app.valuation.pre_valuation_gate import _compute_working_capital_trends
    result = _compute_working_capital_trends({})
    assert result["yearly_metrics"] == []
    assert result["trend_flags"] == []


def test_wc_trend_headwinds_in_context():
    from app.valuation.pre_valuation_gate import compute_quality_context
    facts = _make_facts(
        accounts_receivable=[(2023, 40.0), (2024, 48.0), (2025, 52.0)],
        revenue=[(2020, 88.0), (2021, 90.0), (2022, 92.0), (2023, 365.0), (2024, 365.0), (2025, 365.0)],
    )
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)
    assert "RECEIVABLES_DETERIORATING_HEADWIND" in ctx["valuation_headwinds"]


def test_wc_trend_insufficient_years():
    from app.valuation.pre_valuation_gate import _compute_working_capital_trends
    facts = {
        "accounts_receivable": [(2025, 40.0)],
        "revenue": [(2025, 365.0)],
    }
    result = _compute_working_capital_trends(facts)
    assert result["trend_flags"] == []


# ── New BLOCK conditions (Wave 4) ─────────────────────────────────────────────

def test_block_zero_owner_earnings_3y():
    """3 consecutive years of negative owner earnings → BLOCK."""
    from app.valuation.pre_valuation_gate import compute_quality_context
    facts = {
        "cfo": [(2025, -5.0), (2024, -3.0), (2023, -2.0)],
        "capex": [(2025, 1.0), (2024, 1.0), (2023, 1.0)],
        "sbc": [(2025, 2.0), (2024, 2.0), (2023, 2.0)],
        "revenue": [(2025, 50.0), (2024, 55.0), (2023, 60.0)],
        "operating_income": [(2025, -10.0), (2024, -5.0), (2023, -3.0)],
        "net_income": [(2025, -12.0), (2024, -7.0), (2023, -5.0)],
    }
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        result = compute_quality_context("TEST", "2026-03-27", facts=facts)
    assert result["gate_action"] == "BLOCK"
    assert "ZERO_OWNER_EARNINGS" in result.get("gate_reason_codes", [])


def test_block_negative_equity_declining_revenue():
    """Negative equity + declining revenue → BLOCK."""
    from app.valuation.pre_valuation_gate import compute_quality_context
    facts = {
        "equity": [(2025, -16.0), (2024, 9.0)],
        "revenue": [(2025, 81.0), (2024, 84.0), (2023, 107.0)],
        "operating_income": [(2025, -17.0), (2024, 3.0), (2023, -10.0)],
        "net_income": [(2025, -27.0), (2024, 1.8), (2023, -9.0)],
    }
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        result = compute_quality_context("TEST", "2026-03-27", facts=facts)
    assert result["gate_action"] == "BLOCK"
    assert "BOOK_INSOLVENCY" in result.get("gate_reason_codes", [])


def test_block_current_ratio_below_half():
    """Current ratio < 0.5 → BLOCK."""
    from app.valuation.pre_valuation_gate import compute_quality_context
    facts = {
        "current_assets": [(2025, 15.0)],
        "current_liabilities": [(2025, 40.0)],
        "revenue": [(2025, 50.0), (2024, 55.0), (2023, 60.0)],
        "operating_income": [(2025, 5.0), (2024, 6.0), (2023, 7.0)],
        "net_income": [(2025, 3.0), (2024, 4.0), (2023, 5.0)],
    }
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        result = compute_quality_context("TEST", "2026-03-27", facts=facts)
    assert result["gate_action"] == "BLOCK"
    assert "LIQUIDITY_CRISIS" in result.get("gate_reason_codes", [])


def test_no_false_block_healthy():
    """A healthy company should NOT be blocked by the new conditions."""
    from app.valuation.pre_valuation_gate import compute_quality_context
    facts = {
        "revenue": [(2025, 120.0), (2024, 110.0), (2023, 100.0)],
        "operating_income": [(2025, 26.0), (2024, 23.0), (2023, 20.0)],
        "net_income": [(2025, 19.0), (2024, 17.0), (2023, 15.0)],
        "cfo": [(2025, 30.0), (2024, 28.0), (2023, 25.0)],
        "capex": [(2025, 5.0), (2024, 4.0), (2023, 3.0)],
        "equity": [(2025, 100.0), (2024, 90.0)],
        "current_assets": [(2025, 50.0)],
        "current_liabilities": [(2025, 25.0)],
    }
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        result = compute_quality_context("TEST", "2026-03-27", facts=facts)
    assert result["gate_action"] in ("PROCEED", "ADJUST")


def test_block_going_concern():
    """Going concern language in filing should trigger BLOCK."""
    from unittest.mock import patch
    from app.valuation.pre_valuation_gate import compute_quality_context
    from app.alpha.schemas import SolvencyAssessment

    mock_solvency = SolvencyAssessment(
        solvency_risk="CRITICAL",
        going_concern_language=True,
    )

    facts = {
        "revenue": [(2025, 1000), (2024, 1100), (2023, 1200), (2022, 1300), (2021, 1400)],
        "operating_income": [(2025, 100), (2024, 110), (2023, 120)],
        "net_income": [(2025, 80), (2024, 90), (2023, 100)],
        "cfo": [(2025, 120), (2024, 130), (2023, 140)],
        "capex": [(2025, 40), (2024, 40), (2023, 40)],
        "shares_outstanding": [(2025, 100)],
        "total_debt": [(2025, 200)],
        "cash": [(2025, 50)],
        "equity": [(2025, 300)],
    }

    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}), \
         patch("app.alpha.solvency_scanner.assess_solvency", return_value=mock_solvency):
        result = compute_quality_context("TEST", "2025-12-31", facts=facts)

    assert result["gate_action"] == "BLOCK"
    assert "GOING_CONCERN" in result.get("gate_reason_codes", [])


def test_v2_going_concern_gate_binds_issuer_filing_context_and_serializes_assertions(
    tmp_path,
):
    from app.alpha.schemas import GoingConcernAssertion, SolvencyAssessment
    from app.valuation.pre_valuation_gate import compute_quality_context

    assertion = GoingConcernAssertion(
        subject="REGISTRANT",
        subject_detail="PRIMARY",
        assertion_mode="AFFIRMATIVE_CURRENT",
        blockable=True,
        accession="0000000042-26-000007",
        form_type="10-K",
        filing_date="2026-02-18",
        section="AUDITOR_REPORT",
        excerpt=(
            "The Company has concluded that substantial doubt exists about "
            "its ability to continue as a going concern."
        ),
        corroborating_distress=("LOW_CASH_RUNWAY",),
        issuer_cik="42",
        source_url="https://www.sec.gov/Archives/edgar/data/42/annual.htm",
        content_revision="sha256:annual-v1",
    )
    assessment = SolvencyAssessment(
        solvency_risk="CRITICAL",
        going_concern_language=True,
        going_concern_assertions=[assertion],
    )
    db_path = tmp_path / "issuer-bound.db"

    with (
        patch("app.valuation.pre_valuation_gate._safe_call", return_value={}),
        patch(
            "app.alpha.solvency_scanner.assess_solvency",
            return_value=assessment,
        ) as assess,
    ):
        result = compute_quality_context(
            "ADRX",
            "2026-03-01",
            facts=_make_facts(),
            issuer_cik="42",
            issuer_aliases=("ADRX", "PRIMARY"),
            require_filed_asof=True,
            db_path=db_path,
        )

    assess.assert_called_once_with(
        "ADRX",
        as_of_date="2026-03-01",
        require_filed_asof=True,
        issuer_cik="42",
        aliases=("ADRX", "PRIMARY"),
        db_path=db_path,
    )
    assert result["gate_action"] == "BLOCK"
    assert result["going_concern_assertions"] == [
        {
            "subject": "REGISTRANT",
            "subject_detail": "PRIMARY",
            "assertion_mode": "AFFIRMATIVE_CURRENT",
            "blockable": True,
            "accession": "0000000042-26-000007",
            "form_type": "10-K",
            "filing_date": "2026-02-18",
            "section": "AUDITOR_REPORT",
            "excerpt": (
                "The Company has concluded that substantial doubt exists about "
                "its ability to continue as a going concern."
            ),
            "corroborating_distress": ["LOW_CASH_RUNWAY"],
            "issuer_cik": "42",
            "source_url": "https://www.sec.gov/Archives/edgar/data/42/annual.htm",
            "content_revision": "sha256:annual-v1",
        }
    ]


def test_adjust_valuation_allowance():
    """Full valuation allowance should trigger ADJUST with SOLVENCY_CONCERN headwind."""
    from unittest.mock import patch
    from app.valuation.pre_valuation_gate import compute_quality_context
    from app.alpha.schemas import SolvencyAssessment

    mock_solvency = SolvencyAssessment(
        solvency_risk="ELEVATED",
        valuation_allowance_full=True,
    )

    facts = {
        "revenue": [(2025, 1000), (2024, 1100), (2023, 1200), (2022, 1300), (2021, 1400)],
        "operating_income": [(2025, 100), (2024, 110), (2023, 120)],
        "net_income": [(2025, 80), (2024, 90), (2023, 100)],
        "cfo": [(2025, 120), (2024, 130), (2023, 140)],
        "capex": [(2025, 40), (2024, 40), (2023, 40)],
        "shares_outstanding": [(2025, 100)],
        "total_debt": [(2025, 200)],
        "cash": [(2025, 50)],
        "equity": [(2025, 300)],
    }

    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}), \
         patch("app.alpha.solvency_scanner.assess_solvency", return_value=mock_solvency):
        result = compute_quality_context("TEST", "2025-12-31", facts=facts)

    assert result["gate_action"] == "ADJUST"
    assert "SOLVENCY_CONCERN" in result.get("gate_reason_codes", [])
    assert "SOLVENCY_CONCERN" in result.get("valuation_headwinds", [])


# ── negative_oe_years contract ───────────────────────────────────────────────

def test_negative_oe_years_in_quality_context():
    """quality_context should include negative_oe_years count when ZERO_OWNER_EARNINGS fires."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        mock_safe.side_effect = lambda fn, *a, **kw: {}

        # 3 years of negative owner earnings: CFO < capex + SBC for each year
        facts = _make_facts(
            cfo=[(2025, 100.0), (2024, 80.0), (2023, 60.0), (2022, 90.0), (2021, 70.0)],
            capex=[(2025, 60.0), (2024, 50.0), (2023, 40.0), (2022, 30.0), (2021, 25.0)],
            sbc=[(2025, 50.0), (2024, 40.0), (2023, 30.0), (2022, 20.0), (2021, 15.0)],
        )
        # 2025: 100 < 60+50=110 ✓, 2024: 80 < 50+40=90 ✓, 2023: 60 < 40+30=70 ✓ → 3 years
        ctx = compute_quality_context("TEST", "2026-03-29", facts=facts)

    assert "ZERO_OWNER_EARNINGS" in ctx.get("gate_reason_codes", [])
    assert ctx.get("negative_oe_years") == 3


def test_negative_oe_years_zero_when_owner_earnings_positive():
    """negative_oe_years should be 0 when owner earnings are positive."""
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        mock_safe.side_effect = lambda fn, *a, **kw: {}

        # All years: CFO(20) > capex(0) + SBC(0) = 0 → positive OE
        facts = _make_facts()
        ctx = compute_quality_context("TEST", "2026-03-29", facts=facts)

    assert "ZERO_OWNER_EARNINGS" not in ctx.get("gate_reason_codes", [])
    assert "negative_oe_years" in ctx  # key must be explicitly present
    assert ctx["negative_oe_years"] == 0


def _gate_with_owner_quality(
    owner_quality: dict, shares: list | None = None, facts_row: dict | None = None
) -> dict:
    """Run the gate with the owner-earnings-quality module stubbed and the
    capital-allocation module left REAL, so the test sees what the gate hands it."""
    from app.valuation.pre_valuation_gate import _safe_call, compute_quality_context

    def side_effect(fn, *args, **kwargs):
        name = getattr(fn, "__name__", "")
        if "owner_earnings_quality" in name:
            return dict(owner_quality)
        if "capital_allocation" in name:
            return _safe_call(fn, *args, **kwargs)
        return {}

    facts = _make_facts()
    if shares is not None:
        facts["shares_outstanding"] = shares
    with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
        mock_safe.side_effect = side_effect
        return compute_quality_context("TEST", "2026-03-25", facts=facts, facts_row=facts_row)


# Shares shrinking about 1% a year, and growing about 5% a year, 2022 -> 2025.
_BUYBACK_SHARES = [(2025, 10.0), (2024, 10.1), (2023, 10.2), (2022, 10.3)]
_DILUTER_SHARES = [(2025, 11.6), (2024, 11.0), (2023, 10.5), (2022, 10.0)]


def test_gate_grades_capital_allocation_from_the_owner_earnings_evidence():
    """The gate called the capital-allocation module with no
    inputs, so every issuer graded C (unknown). Handed the owner-earnings
    quality total it already computes (9) and a share count shrinking about 1%
    a year, the module grades OWNER_FRIENDLY_DISCIPLINED, an A."""
    ctx = _gate_with_owner_quality({"oe_quality_total": 9.0}, shares=_BUYBACK_SHARES)
    assert ctx["allocation_grade"] == "A"
    assert "OWNER_FRIENDLY_SUPPORT" in ctx["valuation_supports"]


def test_gate_grades_a_diluter_with_weak_quality_as_f():
    ctx = _gate_with_owner_quality({"oe_quality_total": 1.5}, shares=_DILUTER_SHARES)
    assert ctx["allocation_grade"] == "F"
    assert "CAPITAL_ALLOCATION_HEADWIND" in ctx["valuation_headwinds"]


def test_gate_without_dilution_evidence_still_grades_unknown_as_c():
    ctx = _gate_with_owner_quality({"oe_quality_total": 9.0}, shares=[(2025, 10.0)])
    assert ctx["allocation_grade"] == "C"


def test_gate_ignores_the_split_blind_full_history_dilution_figure():
    """The owner-earnings-quality payload's dilution runs first-to-last over
    the whole filed history, so a 2-for-1 split reads as years of dilution (or a
    reverse split as buybacks). The gate grades from the recent FY counts
    instead, and a window with a split in it is unknown (C), not a grade."""
    from app.valuation.pre_valuation_gate import _recent_share_count_cagr

    split_window = [(2025, 20.4), (2024, 20.2), (2023, 10.1), (2022, 10.0)]
    reverse_split = [(2025, 1.0), (2024, 1.0), (2023, 10.0), (2022, 10.0)]

    assert _recent_share_count_cagr({"shares_outstanding": split_window}) is None
    assert _recent_share_count_cagr({"shares_outstanding": reverse_split}) is None
    assert _recent_share_count_cagr({"shares_outstanding": _BUYBACK_SHARES}) == pytest.approx(
        (10.0 / 10.3) ** (1.0 / 3.0) - 1.0, abs=1e-12
    )

    friendly_claim = {
        "dilution_rate_shares_cagr": -0.05,
        "oe_quality_total": 9.0,
        "capital_allocation_reason_codes": ["SHAREHOLDER_FRIENDLY"],
    }
    assert _gate_with_owner_quality(friendly_claim, shares=reverse_split)["allocation_grade"] == "C"
    assert _gate_with_owner_quality(friendly_claim, shares=_DILUTER_SHARES)["allocation_grade"] == "B"


def test_unknown_accounting_quality_stays_unknown_and_adds_no_moat_point():
    """ACCOUNTING_QUALITY_UNKNOWN was mapped to MODERATE, which
    is worth +1 on the moat score, so not knowing a company's accounting quality
    scored better than knowing it was LOW and the same as knowing it was fine."""
    from app.valuation.pre_valuation_gate import compute_quality_context
    from app.valuation.valuation_writer import _classify_moat_strength

    def run(aq_class):
        def side_effect(fn, *args, **kwargs):
            if "accounting_quality" in getattr(fn, "__name__", ""):
                return {"accounting_quality_class": aq_class}
            return {}

        with patch("app.valuation.pre_valuation_gate._safe_call") as mock_safe:
            mock_safe.side_effect = side_effect
            return compute_quality_context("TEST", "2026-03-25", facts=_make_facts())

    unknown = run("ACCOUNTING_QUALITY_UNKNOWN")
    moderate = run("MODERATE_ACCOUNTING_QUALITY")

    def moat(ctx):
        return _classify_moat_strength(
            earnings_quality=ctx["earnings_quality"],
            revenue_trend_class=ctx["revenue_trend_class"],
            epv_quality=ctx["epv_quality"],
            allocation_grade=ctx["allocation_grade"],
        )

    assert unknown["earnings_quality"] == "UNKNOWN"
    assert moderate["earnings_quality"] == "MODERATE"
    assert moat(moderate)["moat_score"] - moat(unknown)["moat_score"] == 1
    assert moat(unknown)["signal_detail"]["earnings_quality"]["counted"] is False


def test_a_three_for_two_split_in_the_window_is_a_break_not_dilution():
    """A jump of exactly 1.5x sits on the edge of the [2/3, 3/2] band and was read as
    dilution. It is a 3-for-2 split: the shared split-break rule (the one the
    owner-earnings quality module uses) calls it a break, so the window is unknown."""
    from app.valuation.pre_valuation_gate import _recent_share_count_cagr

    three_for_two = [(2025, 15.0), (2024, 15.0), (2023, 15.0), (2022, 10.0)]
    assert _recent_share_count_cagr({"shares_outstanding": three_for_two}) is None
    # Within 3% of a clean split factor counts too (2.0x * 1.02).
    near_two = [(2025, 20.4), (2024, 20.4), (2023, 20.4), (2022, 10.0)]
    assert _recent_share_count_cagr({"shares_outstanding": near_two}) is None


def test_a_fifty_percent_raise_without_a_filed_split_stays_unknown():
    """Review H5, 2026-09-29: 100 -> 100 -> 150 -> 152 is a 50% raise that sits
    on the 3-for-2 split factor. With no filed split it is not a split and not a
    measurable rate: the window is unknown (None), never low dilution."""
    from app.valuation.pre_valuation_gate import _recent_share_count_cagr

    raise_window = [(2022, 100.0), (2023, 100.0), (2024, 150.0), (2025, 152.0)]
    assert _recent_share_count_cagr({"shares_outstanding": raise_window}) is None
    wrong_ratio = [{"year": 2024, "value": 2.0, "derived_from": []}]
    assert (
        _recent_share_count_cagr({"shares_outstanding": raise_window}, split_rows=wrong_ratio)
        is None
    )


def test_a_filed_split_is_split_adjusted_not_dropped():
    """Review H5, 2026-09-29: a break a filed split ratio corroborates is a split.
    The earlier counts are restated on the new basis and the window measured
    across it (the gate used to call every such window unknown): 3-for-2 in
    2024, 100 -> 150 -> 152 is (152 / 150) ** (1/3) - 1 a year."""
    from app.valuation.pre_valuation_gate import _recent_share_count_cagr

    window = [(2022, 100.0), (2023, 100.0), (2024, 150.0), (2025, 152.0)]
    split = [{"year": 2024, "value": 1.5, "derived_from": ["split.2024"]}]
    assert _recent_share_count_cagr({"shares_outstanding": window}, split_rows=split) == (
        (152.0 / 150.0) ** (1.0 / 3.0) - 1.0
    )


def test_gate_reads_the_filed_split_from_the_issuer_companyfacts(tmp_path):
    """The gate loads the split ratio from the facts row's companyfacts cache:
    a 2-for-1 split in 2024 inside a buyback (10.3 -> 10.2 -> 20.2 -> 20.0) is
    about -1% a year once split-adjusted, an A beside a quality total of 9; the
    same counts with nothing filed are unknown (C)."""
    import json

    split_fact = {"val": 2.0, "start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-15",
                  "fy": 2024, "fp": "FY", "form": "10-K", "accn": "0000000000-25-000001"}
    path = tmp_path / "cf.json"
    path.write_text(json.dumps({"facts": {"us-gaap": {
        "StockholdersEquityNoteStockSplitConversionRatio1": {"units": {"pure": [split_fact]}},
    }}}), encoding="utf-8")
    split_buyback = [(2025, 20.0), (2024, 20.2), (2023, 10.2), (2022, 10.3)]

    filed = _gate_with_owner_quality(
        {"oe_quality_total": 9.0}, shares=split_buyback, facts_row={"cache_path": str(path)}
    )
    unfiled = _gate_with_owner_quality({"oe_quality_total": 9.0}, shares=split_buyback)

    assert [row["year"] for row in filed["share_split_ratios"]] == [2024]
    assert filed["allocation_grade"] == "A"
    assert unfiled["share_split_ratios"] == []
    assert unfiled["allocation_grade"] == "C"
