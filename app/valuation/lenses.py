"""Complementary valuation lenses for microcap quality-value.

Three deterministic, point-in-time-safe lenses computed from the as-of facts
dict and persisted per (ticker, as_of, method) like the existing methods:

  - ev_ebit:        EV/EBIT multiple anchor (enterprise convention)
  - fcf_yield:      FCF-yield anchor (equity convention — CFO is post-interest)
  - tangible_floor: tangible-asset / liquidation floor (corrected NCAV +
                    tangible-book variants)

Design rules:
  - NO lens gates anything. They feed the scorecard's per-method table and
    select_anchor() provenance (extra_candidates) only; the canonical anchor
    rule (max positive DCF/EPV, graham->ncav fallback) is unchanged.
  - Zero extra provider calls: every input is an as-of fact; the only price
    used is the one ensure_valuation already has (for warnings only).
  - Conventions are labeled on every output (textbook-vs-upside and
    enterprise-vs-equity confusion produced real bugs — see
    app/valuation/mos_conventions.py).

Considered and REJECTED for v1 (documented per the goal):
  - Own-history EV/EBIT percentile and peer EV/EBIT percentile: both need
    historical/peer PRICES inside the writer; the backtest harness would multiply
    provider load 5-10x per name-date. The static category multiple table is
    deterministic and auditable; own-history/peer enrichment can be layered
    once the price cache covers it. The current implied multiple IS computed
    (vs the anchor multiple) so richness is still visible.
  - P/E lens: duplicates the Graham EPS basis and is the multiple most
    distorted by non-operating items in microcaps.
  - EV/Sales: no profitability discipline — rewards unprofitable revenue in
    exactly the population where that is the failure mode.
  - DDM: microcap universe rarely pays meaningful dividends.
"""
from __future__ import annotations

import statistics
from typing import Any

from app.valuation.adjacent_years import trailing_adjacent_run

# Conservative through-cycle EV/EBIT anchor multiples by company category.
# Static and documented by design (see module docstring). Default 8x is the
# long-run small-cap median paid for steady operating earnings; quality
# software economics merit more, hardware/cyclicals less.
EV_EBIT_MULTIPLE_BY_CATEGORY: dict[str, float] = {
    "ENTERPRISE_SOFTWARE": 12.0,
    "PLATFORM_HYBRID": 10.0,
    "NETWORK_INFRA": 9.0,
    "SEMICONDUCTOR": 8.0,
    "INDUSTRIAL_TECH": 8.0,
    "CONSUMER_HARDWARE": 7.0,
    "TRADITIONAL_OPERATING": 8.0,
}
DEFAULT_EV_EBIT_MULTIPLE = 8.0

# Required levered-FCF yields by category (equity convention).
REQUIRED_FCF_YIELD_BY_CATEGORY: dict[str, float] = {
    "ENTERPRISE_SOFTWARE": 0.07,
    "PLATFORM_HYBRID": 0.075,
    "NETWORK_INFRA": 0.08,
    "SEMICONDUCTOR": 0.09,
    "INDUSTRIAL_TECH": 0.08,
    "CONSUMER_HARDWARE": 0.09,
    "TRADITIONAL_OPERATING": 0.08,
}
DEFAULT_REQUIRED_FCF_YIELD = 0.08

# Graham 2/3 discipline on tangible book for the liquidation floor.
TANGIBLE_BOOK_FLOOR_HAIRCUT = 0.65

_MIN_OBS = 3


def _series(facts: dict, field: str, n: int = 5) -> list[tuple[int, float]]:
    """Newest-first (year, value) pairs: at most ``n``, and only the run of
    calendar-adjacent fiscal years ending at the latest year. A gap inside the
    window used to let an old year in as if it were recent."""
    rows = sorted(facts.get(field) or [], key=lambda x: x[0], reverse=True)
    run = trailing_adjacent_run([(int(y), float(v)) for y, v in rows])
    return list(reversed(run))[:n]


def _value_for_year(facts: dict, field: str, year: int) -> float | None:
    for y, v in facts.get(field) or []:
        if int(y) == year:
            return float(v)
    return None


def ev_ebit_anchor(
    facts: dict,
    *,
    shares: float,
    bridge_deduction: float | None,
    category: str,
    current_price: float | None = None,
) -> dict[str, Any]:
    """EV/EBIT multiple anchor: conservative category multiple x normalized
    EBIT, bridged to equity with the SAME deduction stack as DCF/EPV
    (lease-exclusive net debt + senior claims)."""
    flags: list[str] = []
    oi = _series(facts, "operating_income")
    if len(oi) < _MIN_OBS or shares <= 0 or not isinstance(bridge_deduction, (int, float)):
        gap = len(facts.get("operating_income") or []) >= _MIN_OBS
        return {"status": "METHOD_INSUFFICIENT_DATA", "value_per_share": None,
                "flags": ["INSUFFICIENT_INPUTS"] + (["NON_ADJACENT_YEARS"] if gap else []),
                "basis": {}}

    # Median, negatives included — same normalization basis as the gate.
    ebit_normalized = float(statistics.median([v for _, v in oi]))
    multiple = float(EV_EBIT_MULTIPLE_BY_CATEGORY.get(category, DEFAULT_EV_EBIT_MULTIPLE))
    basis: dict[str, Any] = {
        "ebit_normalized": ebit_normalized,
        "ebit_basis": "median of last-5 FY operating income (negatives included)",
        "multiple": multiple,
        "multiple_source": f"static category table ({category})",
        "bridge_deduction": float(bridge_deduction),
        "convention": "ENTERPRISE (EV/EBIT); equity = multiple x EBIT - net debt - senior claims",
    }
    if ebit_normalized <= 0:
        return {"status": "EV_EBIT_NOT_MEANINGFUL", "value_per_share": None,
                "flags": flags + ["NEGATIVE_NORMALIZED_EBIT"], "basis": basis}

    equity_value = multiple * ebit_normalized - float(bridge_deduction)
    value_per_share = equity_value / shares

    if isinstance(current_price, (int, float)) and current_price > 0:
        implied = (float(current_price) * shares + float(bridge_deduction)) / ebit_normalized
        basis["implied_current_multiple"] = implied
        if implied > multiple * 1.25:
            flags.append("EV_EBIT_RICH_VS_ANCHOR_MULTIPLE")
        elif implied < multiple * 0.75:
            flags.append("EV_EBIT_CHEAP_VS_ANCHOR_MULTIPLE")

    status = "OK" if value_per_share > 0 else "EV_EBIT_NEGATIVE_EQUITY"
    return {"status": status, "value_per_share": value_per_share, "flags": flags, "basis": basis}


def fcf_yield_anchor(
    facts: dict,
    *,
    shares: float,
    category: str,
) -> dict[str, Any]:
    """FCF-yield anchor: normalized levered FCF capitalized at the required
    yield. EQUITY convention — CFO is post-interest, so the value is equity
    directly with NO net-debt deduction (that would charge debt twice)."""
    flags: list[str] = []
    cfo = dict(_series(facts, "cfo"))
    capex = dict(_series(facts, "capex"))
    common_years = list(
        reversed(trailing_adjacent_run(sorted(set(cfo) & set(capex)), lambda year: year))
    )[:5]
    if len(common_years) < _MIN_OBS or shares <= 0:
        gap = (
            len({int(y) for y, _ in facts.get("cfo") or []} & {int(y) for y, _ in facts.get("capex") or []})
            >= _MIN_OBS
        )
        return {"status": "METHOD_INSUFFICIENT_DATA", "value_per_share": None,
                "flags": ["INSUFFICIENT_INPUTS"] + (["NON_ADJACENT_YEARS"] if gap else []),
                "basis": {}}

    fcf_values = [cfo[y] - abs(capex[y]) for y in common_years]
    fcf_normalized = float(statistics.median(fcf_values))
    required_yield = float(REQUIRED_FCF_YIELD_BY_CATEGORY.get(category, DEFAULT_REQUIRED_FCF_YIELD))
    basis: dict[str, Any] = {
        "fcf_normalized": fcf_normalized,
        "fcf_basis": "median of last-5 common-year (CFO - abs(capex)), negatives included",
        "required_yield": required_yield,
        "convention": "EQUITY (levered FCF / required yield); CFO is post-interest — no net-debt deduction",
    }
    if fcf_normalized <= 0:
        return {"status": "FCF_YIELD_NOT_MEANINGFUL", "value_per_share": None,
                "flags": flags + ["NEGATIVE_NORMALIZED_FCF"], "basis": basis}

    value_per_share = (fcf_normalized / required_yield) / shares
    return {"status": "OK", "value_per_share": value_per_share, "flags": flags, "basis": basis}


def tangible_floor(
    facts: dict,
    *,
    shares: float,
    ncav_value_per_share: float | None = None,
) -> dict[str, Any]:
    """Tangible-asset / liquidation floor: the better of corrected NCAV and
    Graham's 2/3-of-tangible-book discipline. Goodwill/intangibles excluded;
    preferred deducted; noncontrolling interest deducted from consolidated
    equity only (parent-only equity already excludes it)."""
    flags: list[str] = []
    equity_series = _series(facts, "equity", n=1)
    if not equity_series or shares <= 0:
        return {"status": "METHOD_INSUFFICIENT_DATA", "value_per_share": None,
                "flags": ["INSUFFICIENT_INPUTS"], "basis": {}}
    year, equity = equity_series[0]

    goodwill = _value_for_year(facts, "goodwill", year)
    intangibles = _value_for_year(facts, "intangible_assets", year)
    preferred = _value_for_year(facts, "preferred_equity", year)
    nci = _value_for_year(facts, "noncontrolling_interest", year)
    # A deduction the company files in other years but not for the equity year is
    # unknown for that year, not zero: assuming zero would lift the floor by the
    # whole goodwill or intangible balance. Only a company that never filed the
    # line keeps the (flagged) zero assumption.
    missing_year_flags = [
        flag
        for field, value, flag in (
            ("goodwill", goodwill, "GOODWILL_YEAR_MISSING"),
            ("intangible_assets", intangibles, "INTANGIBLES_YEAR_MISSING"),
        )
        if value is None and facts.get(field)
    ]
    if missing_year_flags:
        return {"status": "METHOD_INSUFFICIENT_DATA", "value_per_share": None,
                "flags": missing_year_flags, "basis": {"fiscal_year": year}}
    if goodwill is None:
        flags.append("GOODWILL_MISSING_ZERO_ASSUMED")
    if intangibles is None:
        flags.append("INTANGIBLES_MISSING_ZERO_ASSUMED")

    # NCI is senior to common only inside CONSOLIDATED equity. When the filing's
    # consolidated total is on file and differs from "equity", "equity" is the
    # parent-only figure, which already excludes NCI: deducting it again would
    # understate the floor. Without that proof the (conservative) deduction stays.
    consolidated = _value_for_year(facts, "equity_including_nci", year)
    parent_only = consolidated is not None and abs(consolidated - float(equity)) > 1e-9
    if parent_only:
        nci = None
        flags.append("EQUITY_PARENT_ONLY_NCI_NOT_DEDUCTED")
    tangible_equity = (
        float(equity)
        - max(0.0, goodwill or 0.0)
        - max(0.0, intangibles or 0.0)
        - max(0.0, preferred or 0.0)
        - max(0.0, nci or 0.0)
    )
    tangible_bvps = tangible_equity / shares

    candidates = [TANGIBLE_BOOK_FLOOR_HAIRCUT * tangible_bvps]
    if isinstance(ncav_value_per_share, (int, float)):
        candidates.append(float(ncav_value_per_share))
    floor = max(candidates)

    basis: dict[str, Any] = {
        "fiscal_year": year,
        "tangible_book_per_share": tangible_bvps,
        "ncav_value_per_share": ncav_value_per_share,
        "haircut": TANGIBLE_BOOK_FLOOR_HAIRCUT,
        "floor_rule": "max(corrected NCAV, 0.65 x tangible BVPS)",
        "convention": "EQUITY liquidation floor; goodwill/intangibles excluded, preferred + NCI deducted",
    }
    if floor <= 0:
        return {"status": "NO_TANGIBLE_FLOOR", "value_per_share": None,
                "flags": flags + ["NO_POSITIVE_FLOOR"], "basis": basis}
    return {"status": "OK", "value_per_share": floor, "flags": flags, "basis": basis}
