"""Four accounting-quality behaviours in app/valuation/accounting_quality.py, pinned.

Each test uses a synthetic fundamentals series and asserts the correct answer.
"""

from __future__ import annotations

from app.valuation.accounting_quality import (
    UNKNOWN,
    _compute_accruals_signals,
    compute_accounting_quality,
)

OK_STATUS = {"facts_status": "OK", "shares_status": "OK", "fcf_status": "OK"}


# ── one profitable year was enough for HIGH quality ─────────────────


def test_one_profitable_year_is_not_a_three_year_median():
    """Two loss years and one profitable year leave a single usable CFO/NI and
    FCF/NI ratio. A "median" over one row is that row: it used to read 1.2 and
    1.0, fire STRONG_CFO, STRONG_FCF and CASH_EARNINGS_CONSISTENT, and call the
    company HIGH_ACCOUNTING_QUALITY. A median now needs a majority of the
    three-year window (two usable years); with one it is UNKNOWN."""
    rows = [
        {"year": 2023, "net_income": -50.0, "cfo": 10.0, "fcf": 5.0},
        {"year": 2024, "net_income": -40.0, "cfo": 12.0, "fcf": 6.0},
        {"year": 2025, "net_income": 100.0, "cfo": 120.0, "fcf": 100.0},
    ]
    out = compute_accounting_quality("TEST", "2026-03-01", fundamentals={"rows": rows}, **OK_STATUS)
    assert out["cfo_to_net_income_median_3y"] == UNKNOWN
    assert out["fcf_to_net_income_median_3y"] == UNKNOWN
    assert "CASH_EARNINGS_CONSISTENT" not in out["cash_earnings_support_signals"]
    assert out["accounting_quality_class"] == "ACCOUNTING_QUALITY_UNKNOWN"


def test_two_profitable_years_of_three_still_give_a_median():
    """Control: two usable years of the three are a majority; the median is taken."""
    rows = [
        {"year": 2023, "net_income": -50.0, "cfo": 10.0, "fcf": 5.0},
        {"year": 2024, "net_income": 100.0, "cfo": 110.0, "fcf": 90.0},
        {"year": 2025, "net_income": 100.0, "cfo": 120.0, "fcf": 100.0},
    ]
    out = compute_accounting_quality("TEST", "2026-03-01", fundamentals={"rows": rows}, **OK_STATUS)
    assert out["cfo_to_net_income_median_3y"] == 1.15
    assert out["fcf_to_net_income_median_3y"] == 0.95


# ── receivables growth measured over a different window ────────────


def test_a_recent_receivables_build_is_not_hidden_by_old_history():
    """Revenue grows 10% a year 2016-2025. Receivables sat at 100 until 2023 and
    then grew 50% a year to 225 in 2025. First-to-last over every row reads
    receivables +9.4% a year against revenue +10% and granted
    WORKING_CAPITAL_DISCIPLINE_PRESENT. Over the same three recent years the
    conversion ratios use, receivables grow 50% against revenue's 10%: a drain."""
    receivables = {2023: 100.0, 2024: 150.0, 2025: 225.0}
    rows = [
        {
            "year": year,
            "revenue": 100.0 * 1.10 ** (year - 2016),
            "accounts_receivable": receivables.get(year, 100.0),
        }
        for year in range(2016, 2026)
    ]
    out = compute_accounting_quality("TEST", "2026-03-01", fundamentals={"rows": rows}, **OK_STATUS)
    assert round(out["receivables_growth_vs_revenue_growth_proxy"], 4) == 0.4
    assert "WORKING_CAPITAL_DRAIN" in out["cash_earnings_headwind_signals"]
    assert "WORKING_CAPITAL_DISCIPLINE_PRESENT" not in out["cash_earnings_support_signals"]


def test_receivables_and_revenue_growth_use_the_same_years():
    """Receivables reported only for 2023-2025 were compared with revenue growth
    from 2016: two different windows. Both now run 2023 to 2025."""
    rows = [
        {"year": year, "revenue": 100.0 * 1.10 ** (year - 2016)} for year in range(2016, 2023)
    ] + [
        {"year": 2023, "revenue": 150.0, "accounts_receivable": 100.0},
        {"year": 2024, "revenue": 150.0, "accounts_receivable": 100.0},
        {"year": 2025, "revenue": 150.0, "accounts_receivable": 100.0},
    ]
    out = compute_accounting_quality("TEST", "2026-03-01", fundamentals={"rows": rows}, **OK_STATUS)
    assert out["receivables_growth_vs_revenue_growth_proxy"] == 0.0


# ── stale conservative and recent aggressive accruals cancelled ────


def test_stale_conservative_accruals_do_not_offset_recent_aggressive_ones():
    """Accrual ratio 0.0 for 2018-2022 and 0.15 for 2023-2025. Counted over the
    whole history both ACCRUALS_AGGRESSIVE (3 years > 0.10) and
    ACCRUALS_CONSERVATIVE (5 years < 0.02) fired, a +1 support against the +1
    headwind. Over the recent three-year window only the aggressive signal stands."""
    rows = [
        {"year": year, "net_income": 100.0, "cfo": 100.0 if year <= 2022 else -50.0,
         "total_assets": 1000.0}
        for year in range(2018, 2026)
    ]
    result = _compute_accruals_signals(rows)
    assert result["accruals_signals"] == ["ACCRUALS_AGGRESSIVE"]
    assert [item["year"] for item in result["accrual_ratios"]] == [2023, 2024, 2025]


# ── blocked evidence was classified whenever any signal fired ──────


def test_blocked_evidence_is_not_classified_because_a_signal_fired():
    """Facts status not OK marks the evidence blocked, but the block only applied
    when no signal fired: this strong series came back HIGH_ACCOUNTING_QUALITY on
    blocked facts. Blocked evidence is ACCOUNTING_QUALITY_UNKNOWN; the signals
    stay listed for the reader.

    Policy call: only blocks on inputs the module reads (facts status, free
    cash flow status with no FCF ratio, fewer than two rows) are now binding.
    The share status is not an input, keeps its old role (it withholds a class
    only when nothing fired), and the valuation gate, which passes no share
    status, is unaffected."""
    rows = [
        {"year": year, "net_income": 100.0, "cfo": 120.0, "fcf": 100.0}
        for year in (2023, 2024, 2025)
    ]
    out = compute_accounting_quality(
        "TEST",
        "2026-03-01",
        fundamentals={"rows": rows},
        facts_status="STALE",
        shares_status="OK",
        fcf_status="OK",
    )
    assert out["accounting_quality_class"] == "ACCOUNTING_QUALITY_UNKNOWN"
    assert out["primary_accounting_caution"] == "ACCOUNTING_QUALITY_UNCLEAR"
    assert "STRONG_CFO_TO_EARNINGS_CONVERSION" in out["cash_earnings_support_signals"]
    assert "ACCOUNTING_EVIDENCE_THIN" in out["accounting_quality_reason_codes"]
