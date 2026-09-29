"""Pre-valuation quality gate.

Consumes outputs from existing quality modules and produces a
ValuationQualityContext dict that valuation methods use to adjust
inputs or block computation entirely.

Design rules:
  - dict-in / dict-out — stateless pure function
  - never crashes the pipeline — every quality module call is wrapped
  - conservative defaults on failure (MODERATE, not HIGH)
  - dependency flows one way: valuation_writer → pre_valuation_gate → quality modules
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.valuation.share_splits import (
    is_share_count_break,
    split_adjust_share_series,
    split_ratio_rows_from_companyfacts,
)

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

UNKNOWN = "UNKNOWN"

# Gate actions
PROCEED = "PROCEED"
ADJUST = "ADJUST"
BLOCK = "BLOCK"

# Confidence classes
CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MODERATE = "MODERATE"
CONFIDENCE_LOW = "LOW"
CONFIDENCE_INSUFFICIENT = "INSUFFICIENT"

# EPV adjustments
EPV_ADJ_NONE = "NONE"
EPV_ADJ_USE_NORMALIZED = "USE_NORMALIZED"
EPV_ADJ_BLOCK = "BLOCK"

# ── Headwind signals ──────────────────────────────────────────────────────
SECULAR_DECLINE_HEADWIND = "SECULAR_DECLINE_HEADWIND"
ACCOUNTING_QUALITY_HEADWIND = "ACCOUNTING_QUALITY_HEADWIND"
CAPITAL_ALLOCATION_HEADWIND = "CAPITAL_ALLOCATION_HEADWIND"
EARNINGS_QUALITY_HEADWIND = "EARNINGS_QUALITY_HEADWIND"
SBC_BURDEN_HEADWIND = "SBC_BURDEN_HEADWIND"
LEVERAGE_STRESS_HEADWIND = "LEVERAGE_STRESS_HEADWIND"
CYCLICAL_PEAK_HEADWIND = "CYCLICAL_PEAK_HEADWIND"

# ── Support signals ───────────────────────────────────────────────────────
STRONG_EARNINGS_SUPPORT = "STRONG_EARNINGS_SUPPORT"
OWNER_FRIENDLY_SUPPORT = "OWNER_FRIENDLY_SUPPORT"
STRONG_CASH_CONVERSION_SUPPORT = "STRONG_CASH_CONVERSION_SUPPORT"
LOW_LEVERAGE_SUPPORT = "LOW_LEVERAGE_SUPPORT"
GROWING_REVENUE_SUPPORT = "GROWING_REVENUE_SUPPORT"
TROUGH_EARNINGS_CONSERVATIVE = "TROUGH_EARNINGS_CONSERVATIVE"

# ── Working capital trend headwinds ───────────────────────────────────
RECEIVABLES_DETERIORATING_HEADWIND = "RECEIVABLES_DETERIORATING_HEADWIND"
INVENTORY_BUILDING_HEADWIND = "INVENTORY_BUILDING_HEADWIND"
WORKING_CAPITAL_DRAG_HEADWIND = "WORKING_CAPITAL_DRAG_HEADWIND"

# ── SBC and depreciation headwinds ────────────────────────────────────
SBC_ACCELERATING_HEADWIND = "SBC_ACCELERATING_HEADWIND"
SBC_BURDEN_EXTREME_HEADWIND = "SBC_BURDEN_EXTREME_HEADWIND"
NET_DILUTION_HEADWIND = "NET_DILUTION_HEADWIND"
DEPRECIATION_RATE_DECLINING_HEADWIND = "DEPRECIATION_RATE_DECLINING_HEADWIND"
CAPEX_BELOW_DEPRECIATION_HEADWIND = "CAPEX_BELOW_DEPRECIATION_HEADWIND"

# ── Peer-relative headwinds/supports ──────────────────────────────────
PEER_LEADER_SUPPORT = "PEER_LEADER_SUPPORT"
PEER_LAGGARD_HEADWIND = "PEER_LAGGARD_HEADWIND"

# Earnings quality (mapped from accounting_quality_class)
_AQ_MAP: dict[str, str] = {
    "HIGH_ACCOUNTING_QUALITY": "HIGH",
    "MODERATE_ACCOUNTING_QUALITY": "MODERATE",
    "LOW_ACCOUNTING_QUALITY": "LOW",
    # Unknown stays UNKNOWN. Mapping it to MODERATE gave every issuer whose
    # accounting quality could not be read the moat point a MODERATE reading
    # earns. The gate's own rules key on LOW and HIGH only.
    "ACCOUNTING_QUALITY_UNKNOWN": UNKNOWN,
}

# Balance sheet stress (mapped from balance_sheet_stress_class)
_BS_MAP: dict[str, str] = {
    "LOW_BALANCE_SHEET_STRESS": "NONE",
    "MODERATE_BALANCE_SHEET_STRESS": "MODERATE",
    "HIGH_BALANCE_SHEET_STRESS": "HIGH",
    "BALANCE_SHEET_STRESS_UNKNOWN": "MODERATE",
}

# Cycle position (mapped from cycle_position_class)
_CP_MAP: dict[str, str] = {
    "DEPRESSED_RELATIVE_TO_NORMAL": "TROUGH",
    "NEAR_NORMAL": "MID",
    "ELEVATED_RELATIVE_TO_NORMAL": "PEAK",
    "CYCLE_POSITION_UNKNOWN": "MID",
}

# Capital allocation discipline (mapped to letter grade)
_CA_MAP: dict[str, str] = {
    "OWNER_FRIENDLY_DISCIPLINED": "A",
    "MIXED_CAPITAL_ALLOCATION": "B",
    "OWNER_DILUTIVE_OR_DESTRUCTIVE": "F",
    "CAPITAL_ALLOCATION_UNKNOWN": "C",
}

# Maintenance capex ratios by tech category
_MAINT_CAPEX_BY_CATEGORY: dict[str, float] = {
    "SEMICONDUCTOR": 0.80,
    "CONSUMER_HARDWARE": 0.70,
    "INDUSTRIAL_TECH": 0.75,
    "ENTERPRISE_SOFTWARE": 0.25,
    "PLATFORM_HYBRID": 0.40,
    "NETWORK_INFRA": 0.65,
    "TRADITIONAL_OPERATING": 0.60,
}
_DEFAULT_MAINT_CAPEX = 0.60

# Terminal growth by quality profile
_TERMINAL_GROWTH_DEFAULT = 0.02


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_call(fn, *args, **kwargs) -> dict[str, Any]:
    """Call a quality module function, returning {} on any failure."""
    try:
        result = fn(*args, **kwargs)
        return result if isinstance(result, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre_valuation_gate: quality module %s failed: %s", getattr(fn, "__name__", fn), exc)
        return {}


def _quality_module_rows(
    facts: dict[str, list[tuple[int, float]]],
) -> list[dict[str, Any]]:
    """Project the gate's own fact series into the per-year row contract the
    quality modules read.

    The gate holds every series accounting quality and balance-sheet stress
    need — net income, cash flow, capex, revenue, assets, receivables,
    inventory, stock compensation, debt and cash — and used to call both
    modules with none of it, so both answered UNKNOWN for every issuer and the
    gate rendered that as "MODERATE". Two figures are derived here rather than
    invented: free cash flow as cash flow less the MAGNITUDE of capex (the
    definition in app/valuation/fcf.py) and net debt as total debt less cash
    and short-term investments (the valuation's own net-debt basis). Everything else is passed
    through as filed. Values are USD millions, the unit _load_facts returns and
    the unit balance_sheet_stress expects by default.
    """
    by_year: dict[int, dict[str, Any]] = {}
    for line_item, series in (facts or {}).items():
        for year, value in series or []:
            if not isinstance(value, (int, float)):
                continue
            by_year.setdefault(int(year), {})[str(line_item)] = float(value)
    rows: list[dict[str, Any]] = []
    for year in sorted(by_year):
        source = by_year[year]
        row: dict[str, Any] = {"year": year}
        for key in (
            "revenue",
            "net_income",
            "cfo",
            "capex",
            "total_assets",
            "accounts_receivable",
            "inventory",
            "sbc",
            "total_debt",
            "cash",
            "equity",
        ):
            if key in source:
                row[key] = source[key]
        if "cfo" in source and "capex" in source:
            row["fcf"] = source["cfo"] - abs(source["capex"])
        if "total_debt" in source and "cash" in source:
            # Short-term investments count as cash, as the valuation's own net
            # debt counts them (valuation_writer._cash_like_for_year); without
            # them the stress class saw a cash-rich issuer as indebted.
            short_term = source.get("short_term_investments")
            cash_like = source["cash"] + (
                short_term if isinstance(short_term, float) and short_term > 0 else 0.0
            )
            row["net_debt"] = source["total_debt"] - cash_like
        rows.append(row)
    return rows


def _classify_confidence(
    *,
    earnings_quality: str,
    revenue_trend: dict,
    allocation_grade: str,
    oe_quality_total: float | None,
    gate_action: str,
) -> str:
    """Classify valuation confidence based on quality signals.

    Returns one of: HIGH, MODERATE, LOW, INSUFFICIENT.
    """
    if gate_action == BLOCK:
        return CONFIDENCE_INSUFFICIENT
    if earnings_quality == "LOW":
        return CONFIDENCE_LOW
    if isinstance(oe_quality_total, (int, float)) and oe_quality_total < 3.0:
        return CONFIDENCE_LOW
    decline_mag = revenue_trend.get("decline_magnitude_total")
    if isinstance(decline_mag, (int, float)) and decline_mag < -0.10:
        return CONFIDENCE_LOW
    if (
        earnings_quality == "HIGH"
        and revenue_trend.get("revenue_trend_class") == "GROWING"
        and allocation_grade in ("A", "B")
    ):
        return CONFIDENCE_HIGH
    return CONFIDENCE_MODERATE


def _classify_epv_quality(
    revenue_cagr_5y: float | None,
    revenue_cagr_3y: float | None,
) -> tuple[str, float | None]:
    """Classify EPV input quality based on revenue CAGR.

    Returns (epv_quality, cagr_used).
    """
    cagr = revenue_cagr_5y if revenue_cagr_5y is not None else revenue_cagr_3y
    if cagr is None:
        return UNKNOWN, None
    if cagr < -0.05:
        return "DETERIORATING_BASE", cagr
    if cagr < -0.03:
        return "DECLINING_BASE", cagr
    return "STABLE", cagr


# Share-count evidence for the capital-allocation grade: a recent window, and
# no year-over-year break that a filed split does not explain. The break rule
# and the split corroboration live in app.valuation.share_splits (shared with
# the owner-earnings-quality module, the evidence packet and SBC trajectory).
_DILUTION_WINDOW_YEARS = 3
_is_share_count_break = is_share_count_break


# Owner-earnings-quality reason codes that are read off its own dilution figure.
_DILUTION_DERIVED_REASON_CODES = frozenset({"EXCESS_DILUTION", "SHAREHOLDER_FRIENDLY"})


def _recent_share_count_cagr(
    facts: dict[str, list[tuple[int, float]]],
    *,
    split_rows: list[dict[str, Any]] | None = None,
) -> float | None:
    """Annual change in the FY share count over the newest three years, or None.

    Needs the four newest fiscal years (three intervals) with positive counts.
    A year-over-year ratio outside [2/3, 3/2], or within 3% of a clean split
    factor (a 3-for-2 split is exactly 1.5), is a break. A break a filed split
    ratio (``split_rows``, from the issuer's companyfacts) corroborates is
    split-adjusted and the window measured across it; any other break — a
    large raise, a buyback of a third of the stock, a mis-scaled count, or a
    split nobody filed — leaves the rate unknown. The owner-earnings-quality
    payload's own dilution figure is not handed to the grade.
    """
    series = sorted(
        (
            (int(year), float(value))
            for year, value in facts.get("shares_outstanding") or []
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        key=lambda item: item[0],
        reverse=True,
    )[: _DILUTION_WINDOW_YEARS + 1]
    if len(series) < _DILUTION_WINDOW_YEARS + 1 or any(v <= 0 for _, v in series):
        return None
    ordered, _split_years, uncorroborated = split_adjust_share_series(series, split_rows)
    if uncorroborated:
        return None
    years = ordered[-1][0] - ordered[0][0]
    if years <= 0:
        return None
    return float((ordered[-1][1] / ordered[0][1]) ** (1.0 / years) - 1.0)


def _growth_aware_maintenance_ratio(
    category_ratio: float,
    revenue_cagr: float | None,
) -> float:
    """Greenwald-style maintenance fraction: a no-growth firm's capex is all
    maintenance. At/below terminal growth (2%) the full capex is maintenance
    (ratio 1.0); the category haircut only applies at demonstrated growth
    (>=8%), with linear scaling between. Unknown growth keeps the category
    ratio (audit: maint-capex-60pct-universal-haircut)."""
    if not isinstance(revenue_cagr, (int, float)):
        return float(category_ratio)
    if revenue_cagr <= 0.02:
        return 1.0
    if revenue_cagr >= 0.08:
        return float(category_ratio)
    growth_fraction = (float(revenue_cagr) - 0.02) / 0.06
    return 1.0 + (float(category_ratio) - 1.0) * growth_fraction


def _classify_revenue_trend(
    revenue_series: list[tuple[int, float]],
    revenue_cagr_5y: float | None,
    revenue_cagr_3y: float | None = None,
) -> dict[str, Any]:
    """Classify revenue trend and compute decline metrics.

    Returns dict with: revenue_trend_class, revenue_cagr_5y, revenue_cagr_3y,
    decline_years_consecutive, decline_magnitude_total, peak_year, peak_revenue,
    reason_codes.
    """
    base: dict[str, Any] = {
        "revenue_trend_class": UNKNOWN,
        "revenue_cagr_5y": revenue_cagr_5y,
        "revenue_cagr_3y": revenue_cagr_3y,
        "decline_years_consecutive": 0,
        "decline_magnitude_total": None,
        "peak_year": None,
        "peak_revenue": None,
        "reason_codes": [],
    }

    if len(revenue_series) < 3:
        return base

    # Sort chronologically
    sorted_series = sorted(revenue_series, key=lambda x: x[0])
    revenues = [v for _, v in sorted_series]
    years = [yr for yr, _ in sorted_series]

    # Peak detection
    peak_idx = max(range(len(revenues)), key=lambda i: revenues[i])
    base["peak_year"] = years[peak_idx]
    base["peak_revenue"] = revenues[peak_idx]

    # One-year spike awareness (audit: secular-decline-block-single-spike-
    # peak): a peak >= 2.0x BOTH adjacent years is a non-recurring revenue
    # event (licensing/milestone), not a level the business fell from.
    # Decline magnitude is then measured against the highest NON-spike year
    # so a steadily growing company is not blocked for "secular decline".
    if 0 < peak_idx < len(revenues) - 1:
        peak_is_spike = (
            revenues[peak_idx - 1] > 0
            and revenues[peak_idx + 1] > 0
            and revenues[peak_idx] >= 2.0 * revenues[peak_idx - 1]
            and revenues[peak_idx] >= 2.0 * revenues[peak_idx + 1]
        )
    else:
        # Window edge: a spike that slides
        # to the first/last slot of the window has one visible neighbor —
        # requiring two re-blocked the same growing company for one year.
        # One-sided test at 2.0x; genuine-decline peaks are not 2x their
        # neighbor. Multi-year (adjacent) spikes remain undetected by design.
        neighbor_idx = 1 if peak_idx == 0 else len(revenues) - 2
        peak_is_spike = (
            revenues[neighbor_idx] > 0
            and revenues[peak_idx] >= 2.0 * revenues[neighbor_idx]
        )
    base["peak_is_spike"] = peak_is_spike
    decline_reference = (
        max(v for i, v in enumerate(revenues) if i != peak_idx)
        if peak_is_spike
        else revenues[peak_idx]
    )

    # Decline magnitude: (latest - reference peak) / reference peak — only if
    # latest is below the reference
    latest_revenue = revenues[-1]
    peak_revenue = decline_reference
    if peak_revenue > 0 and latest_revenue < peak_revenue:
        base["decline_magnitude_total"] = (latest_revenue - peak_revenue) / peak_revenue

    # Consecutive decline years: count from the end of the series backward
    consecutive = 0
    for i in range(len(revenues) - 1, 0, -1):
        if revenues[i] < revenues[i - 1]:
            consecutive += 1
        else:
            break
    base["decline_years_consecutive"] = consecutive

    # Reason codes
    if consecutive >= 3:
        base["reason_codes"].append(f"CONSECUTIVE_DECLINE_{consecutive}YR")
    if base["decline_magnitude_total"] is not None and base["decline_magnitude_total"] < -0.20:
        base["reason_codes"].append(f"PEAK_{base['peak_year']}")

    # ── Classification ─────────────────────────────────────────────────────
    # Check volatility first
    if len(revenues) >= 3:
        yoy_changes = [
            (revenues[i] - revenues[i - 1]) / abs(revenues[i - 1])
            for i in range(1, len(revenues))
            if abs(revenues[i - 1]) > 1e-9
        ]
        if yoy_changes:
            import statistics
            mean_chg = statistics.mean(yoy_changes)
            if len(yoy_changes) >= 2:
                stdev_chg = statistics.stdev(yoy_changes)
                if stdev_chg > 0.15 and abs(mean_chg) < 0.05:
                    base["revenue_trend_class"] = "VOLATILE"
                    return base

    # Secular decline: >=3 consecutive OR >20% cumulative drop
    decline_mag = base["decline_magnitude_total"]
    if consecutive >= 3 or (isinstance(decline_mag, (int, float)) and decline_mag < -0.20):
        base["revenue_trend_class"] = "SECULAR_DECLINE"
        return base

    # Use CAGR for remaining classifications
    cagr = revenue_cagr_5y
    if cagr is None:
        return base  # stays UNKNOWN
    if cagr > 0.05:
        base["revenue_trend_class"] = "GROWING"
    elif cagr >= -0.03:
        base["revenue_trend_class"] = "FLAT"
    elif cagr >= -0.05:
        base["revenue_trend_class"] = "DECLINING"
    else:
        base["revenue_trend_class"] = "SECULAR_DECLINE"

    return base


def _compute_working_capital_efficiency(
    facts: dict[str, list[tuple[int, float]]],
) -> dict[str, float | None]:
    """Compute working capital efficiency metrics for the latest common year.

    Returns dict with: dso (days sales outstanding), dio (days inventory outstanding),
    dpo (days payable outstanding), ccc (cash conversion cycle = DSO + DIO - DPO).
    """
    from app.valuation.valuation_writer import _get_value_for_year, _latest_common_year

    result: dict[str, float | None] = {"dso": None, "dio": None, "dpo": None, "ccc": None}

    # DSO: (AR / Revenue) * 365
    year = _latest_common_year(facts, "accounts_receivable", "revenue")
    if year:
        ar = _get_value_for_year(facts, "accounts_receivable", year)
        rev = _get_value_for_year(facts, "revenue", year)
        if isinstance(ar, (int, float)) and isinstance(rev, (int, float)) and rev > 0:
            result["dso"] = round(float(ar) / float(rev) * 365, 1)

    # DIO: (Inventory / Revenue) * 365 (using revenue as proxy for COGS)
    year = _latest_common_year(facts, "inventory", "revenue")
    if year:
        inv = _get_value_for_year(facts, "inventory", year)
        rev = _get_value_for_year(facts, "revenue", year)
        if isinstance(inv, (int, float)) and isinstance(rev, (int, float)) and rev > 0:
            result["dio"] = round(float(inv) / float(rev) * 365, 1)

    # DPO: (AP / Revenue) * 365 (using revenue as proxy for COGS)
    year = _latest_common_year(facts, "accounts_payable", "revenue")
    if year:
        ap = _get_value_for_year(facts, "accounts_payable", year)
        rev = _get_value_for_year(facts, "revenue", year)
        if isinstance(ap, (int, float)) and isinstance(rev, (int, float)) and rev > 0:
            result["dpo"] = round(float(ap) / float(rev) * 365, 1)

    # CCC = DSO + DIO - DPO
    if result["dso"] is not None and result["dpo"] is not None:
        dio = result["dio"] or 0.0
        result["ccc"] = round(result["dso"] + dio - result["dpo"], 1)

    return result


def _compute_working_capital_trends(
    facts: dict[str, list[tuple[int, float]]],
) -> dict[str, Any]:
    """Compute working capital trend metrics over multiple years.

    Returns dict with yearly_metrics (per-year DSO/DIO/DPO/CCC) and trend_flags.
    """
    from app.valuation.valuation_writer import _get_value_for_year

    yearly_metrics: list[dict[str, Any]] = []
    trend_flags: list[str] = []

    rev_entries = facts.get("revenue") or []
    if len(rev_entries) < 2:
        return {"yearly_metrics": [], "trend_flags": []}

    years = sorted({yr for yr, _ in rev_entries})

    for year in years:
        rev = _get_value_for_year(facts, "revenue", year)
        if not isinstance(rev, (int, float)) or rev <= 0:
            continue

        ar = _get_value_for_year(facts, "accounts_receivable", year)
        inv = _get_value_for_year(facts, "inventory", year)
        ap = _get_value_for_year(facts, "accounts_payable", year)

        dso = round(float(ar) / float(rev) * 365, 1) if isinstance(ar, (int, float)) else None
        dio = round(float(inv) / float(rev) * 365, 1) if isinstance(inv, (int, float)) else None
        dpo = round(float(ap) / float(rev) * 365, 1) if isinstance(ap, (int, float)) else None

        ccc: float | None = None
        if dso is not None and dpo is not None:
            ccc = round(dso + (dio or 0.0) - dpo, 1)

        yearly_metrics.append({"year": year, "dso": dso, "dio": dio, "dpo": dpo, "ccc": ccc})

    if len(yearly_metrics) < 3:
        return {"yearly_metrics": yearly_metrics, "trend_flags": []}

    def _first_last_with(key: str) -> tuple[dict | None, dict | None]:
        """Find oldest and latest yearly_metrics entries that have a non-None value for key."""
        with_key = [m for m in yearly_metrics if isinstance(m.get(key), (int, float))]
        if len(with_key) < 2:
            return None, None
        return with_key[0], with_key[-1]

    # DSO trend: > 15% increase
    dso_oldest, dso_latest = _first_last_with("dso")
    if (
        dso_oldest is not None
        and dso_latest is not None
        and dso_oldest["dso"] > 0
    ):
        dso_change = (dso_latest["dso"] - dso_oldest["dso"]) / dso_oldest["dso"]
        if dso_change > 0.15:
            trend_flags.append("RECEIVABLES_DETERIORATING")

    # DIO trend: > 20% increase
    dio_oldest, dio_latest = _first_last_with("dio")
    if (
        dio_oldest is not None
        and dio_latest is not None
        and dio_oldest["dio"] > 0
    ):
        dio_change = (dio_latest["dio"] - dio_oldest["dio"]) / dio_oldest["dio"]
        if dio_change > 0.20:
            trend_flags.append("INVENTORY_BUILDING")

    # CCC trend: > 10 days increase
    ccc_oldest, ccc_latest = _first_last_with("ccc")
    if ccc_oldest is not None and ccc_latest is not None:
        ccc_change = ccc_latest["ccc"] - ccc_oldest["ccc"]
        if ccc_change > 10.0:
            trend_flags.append("WORKING_CAPITAL_DRAG")

    return {"yearly_metrics": yearly_metrics, "trend_flags": trend_flags}


def _compute_terminal_growth(
    confidence_class: str,
    revenue_trend_class: str,
    allocation_grade: str,
) -> float:
    """Select terminal growth rate based on 4-tier quality profile.

    Tier 1: HIGH confidence + GROWING + A allocation → 2.0%
    Tier 2: MODERATE confidence OR FLAT revenue       → 1.5%
    Tier 3: LOW confidence OR decline 10-30%          → 0.5%
    Tier 4: INSUFFICIENT (BLOCK) OR decline >30%      → 0.0%
    """
    if confidence_class == "INSUFFICIENT":
        return 0.0
    if confidence_class == "LOW":
        return 0.005
    if (
        confidence_class == "HIGH"
        and revenue_trend_class == "GROWING"
        and allocation_grade == "A"
    ):
        return 0.02  # _TERMINAL_GROWTH_DEFAULT
    return 0.015  # MODERATE default


# ── main entry point ──────────────────────────────────────────────────────────

def compute_quality_context(
    ticker: str,
    as_of_date: str,
    *,
    facts: dict[str, list[tuple[int, float]]],
    facts_row: dict[str, Any] | None = None,
    category: str = "TRADITIONAL_OPERATING",
    cfg: Any | None = None,
    issuer_cik: str | None = None,
    issuer_aliases: tuple[str, ...] = (),
    require_filed_asof: bool = False,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute pre-valuation quality context for a ticker.

    Parameters
    ----------
    ticker : str
        Uppercase ticker symbol.
    as_of_date : str
        Valuation date.
    facts : dict
        Raw companyfacts data from _load_facts() — {line_item: [(year, value), ...]}.
    facts_row : dict or None
        Resolved facts row from resolve_financial_facts_asof().
    category : str
        Tech category from classify_company_category().
    cfg : AppConfig or None
        Application config.
    issuer_cik, issuer_aliases, require_filed_asof, db_path
        Issuer-bound point-in-time inputs for v2 solvency evidence. Legacy
        callers omit them and retain the historical ticker-only scan.

    Returns
    -------
    dict with ValuationQualityContext fields.
    """
    from app.valuation.valuation_writer import _n_years, _revenue_cagr

    # ── Revenue CAGR & EPV quality ────────────────────────────────────────────
    rev_series = _n_years(facts, "revenue", n=6)
    revenue_cagr_5y = _revenue_cagr(rev_series, n=6)
    revenue_cagr_3y = _revenue_cagr(rev_series, n=4)
    epv_quality, _cagr_used = _classify_epv_quality(revenue_cagr_5y, revenue_cagr_3y)
    revenue_trend = _classify_revenue_trend(rev_series, revenue_cagr_5y, revenue_cagr_3y=revenue_cagr_3y)
    revenue_trend_class = revenue_trend["revenue_trend_class"]

    # ── Accounting quality ────────────────────────────────────────────────────
    earnings_quality = UNKNOWN
    acct_payload: dict[str, Any] = {}
    try:
        from app.valuation.accounting_quality import compute_accounting_quality
        acct_payload = _safe_call(
            compute_accounting_quality,
            ticker, as_of_date,
            fundamentals={"ticker": ticker, "rows": _quality_module_rows(facts)},
            facts_status="OK" if facts else UNKNOWN,
            cfg=cfg,
        )
        raw_class = str(acct_payload.get("accounting_quality_class") or "")
        earnings_quality = _AQ_MAP.get(raw_class, UNKNOWN)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre_valuation_gate: accounting_quality import failed: %s", exc)

    # ── Owner earnings quality ────────────────────────────────────────────────
    cash_conversion_score: float | None = None
    sbc_burden = False
    oe_q: dict[str, Any] = {}
    try:
        from app.valuation.owner_earnings_quality import compute_owner_earnings_quality
        oe_q = _safe_call(
            compute_owner_earnings_quality,
            ticker, as_of_date,
            facts_row=facts_row,
            cfg=cfg,
        )
        raw_score = oe_q.get("cash_conversion_score")
        if isinstance(raw_score, (int, float)):
            cash_conversion_score = float(raw_score)
        # SBC burden: check for high SBC flag or ratio
        sbc_ratio = oe_q.get("sbc_to_revenue_median_3y") if isinstance(oe_q.get("sbc_to_revenue_median_3y"), (int, float)) else None
        if sbc_ratio is not None and sbc_ratio > 0.05:
            sbc_burden = True
        # Also check acct_payload for SBC signal
        if not sbc_burden and isinstance(acct_payload.get("sbc_to_revenue_median_3y"), (int, float)):
            if float(acct_payload["sbc_to_revenue_median_3y"]) > 0.05:
                sbc_burden = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre_valuation_gate: owner_earnings_quality import failed: %s", exc)

    # ── Owner earnings quality total ───────────────────────────────────────────
    oe_quality_total: float | None = None
    if isinstance(oe_q, dict):
        raw_oet = oe_q.get("oe_quality_total")
        if isinstance(raw_oet, (int, float)):
            oe_quality_total = float(raw_oet)

    # ── Cyclical normalization ────────────────────────────────────────────────
    cycle_position = "MID"
    normalized_earnings: float | None = None
    try:
        from app.valuation.cyclical_normalization import compute_cyclical_normalization
        # Build series dicts from facts
        oe_series_raw = facts.get("cfo") or []
        oe_series = [{"year": yr, "value": v} for yr, v in oe_series_raw]
        fcf_raw = facts.get("fcf") or []
        fcf_series = [{"year": yr, "value": v} for yr, v in fcf_raw]
        cfo_raw = facts.get("cfo") or []
        cfo_series = [{"year": yr, "value": v} for yr, v in cfo_raw]

        cycl_payload = _safe_call(
            compute_cyclical_normalization,
            ticker, as_of_date,
            owner_earnings_series=oe_series,
            fcf_series=fcf_series,
            cfo_series=cfo_series,
        )
        raw_cp = str(cycl_payload.get("cycle_position_class") or "")
        cycle_position = _CP_MAP.get(raw_cp, "MID")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre_valuation_gate: cyclical_normalization import failed: %s", exc)

    # normalized_earnings on the SAME basis as the EPV input (operating
    # income): median of the last 5 FY values, negatives INCLUDED (audit:
    # use-normalized-feeds-cfo-median-as-operating-income). The previous
    # positive-CFO median was after-tax and D&A-inclusive, then got re-taxed
    # in _epv — inflating the "conservative" EPV ~40-60% for capital-intensive
    # cyclicals. Cycle POSITION detection still uses the CFO-based payload.
    oi_norm_window = sorted(
        facts.get("operating_income") or [], key=lambda x: x[0], reverse=True
    )[:5]
    if oi_norm_window:
        import statistics
        normalized_earnings = float(statistics.median([v for _, v in oi_norm_window]))

    # Guard: strong secular growers should not be classified as PEAK.
    # The cyclical normalizer can misread steep growth trajectories as elevated.
    if (
        cycle_position == "PEAK"
        and revenue_trend_class == "GROWING"
        and isinstance(revenue_cagr_5y, (int, float))
        and revenue_cagr_5y > 0.08
    ):
        cycle_position = "MID"
        logger.info("pre_valuation_gate: overrode PEAK→MID for strong grower (CAGR=%.1f%%)", revenue_cagr_5y * 100)

    # ── Capital allocation discipline ─────────────────────────────────────────
    allocation_grade = "C"
    split_rows: list[dict[str, Any]] = []
    try:
        from app.valuation.capital_allocation_discipline import compute_capital_allocation_discipline
        # Hand the module the evidence it grades from. Called with no inputs
        # it returned UNKNOWN for every issuer, so every grade was C and the
        # moat score could never count allocation. The quality
        # figures come from the owner-earnings-quality payload; the dilution
        # rate comes from the valuation's own recent FY share counts, guarded
        # against splits, and replaces the payload's split-blind full-history
        # figure. Without a usable recent rate the module answers UNKNOWN (C)
        # rather than guess.
        owner_quality_for_ca: dict[str, Any] | None = None
        if isinstance(oe_q, dict):
            owner_quality_for_ca = {
                **oe_q,
                "dilution_rate_shares_cagr": UNKNOWN,
                "capital_allocation_reason_codes": [
                    code
                    for code in (oe_q.get("capital_allocation_reason_codes") or [])
                    if str(code) not in _DILUTION_DERIVED_REASON_CODES
                ],
            }
        if isinstance(facts_row, dict):
            from app.valuation.owner_earnings import _load_companyfacts_payload

            split_rows = split_ratio_rows_from_companyfacts(
                _load_companyfacts_payload(facts_row), as_of_date
            )
        share_count_cagr = _recent_share_count_cagr(facts, split_rows=split_rows)
        ca_payload = _safe_call(
            compute_capital_allocation_discipline,
            ticker, as_of_date,
            owner_quality_payload=owner_quality_for_ca,
            shares_payload=(
                {"share_count_cagr": share_count_cagr} if share_count_cagr is not None else None
            ),
            facts_status="OK" if facts else UNKNOWN,
            shares_status="OK" if share_count_cagr is not None else UNKNOWN,
            cfg=cfg,
        )
        raw_ca = str(ca_payload.get("capital_allocation_discipline_class") or "")
        allocation_grade = _CA_MAP.get(raw_ca, "C")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre_valuation_gate: capital_allocation import failed: %s", exc)

    # ── Balance sheet stress ──────────────────────────────────────────────────
    leverage_stress = "MODERATE"
    try:
        from app.valuation.balance_sheet_stress import compute_balance_sheet_stress
        bs_payload = _safe_call(
            compute_balance_sheet_stress,
            ticker, as_of_date,
            fundamentals={"ticker": ticker, "rows": _quality_module_rows(facts)},
            facts_status="OK" if facts else UNKNOWN,
            cfg=cfg,
        )
        raw_bs = str(bs_payload.get("balance_sheet_stress_class") or "")
        leverage_stress = _BS_MAP.get(raw_bs, "MODERATE")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre_valuation_gate: balance_sheet_stress import failed: %s", exc)

    # ── Working capital efficiency ────────────────────────────────────────────
    working_capital = _compute_working_capital_efficiency(facts)

    # ── Working capital trends ─────────────────────────────────────────────
    working_capital_trends = _compute_working_capital_trends(facts)

    # ── Maintenance capex ratio (growth-aware) ────────────────────────────────
    maintenance_capex_pct = _growth_aware_maintenance_ratio(
        _MAINT_CAPEX_BY_CATEGORY.get(category, _DEFAULT_MAINT_CAPEX),
        revenue_cagr_5y if isinstance(revenue_cagr_5y, (int, float)) else revenue_cagr_3y,
    )

    # ── Headwinds and supports ─────────────────────────────────────────────────
    decline_mag = revenue_trend.get("decline_magnitude_total")
    decline_years = revenue_trend.get("decline_years_consecutive", 0)

    headwinds: list[str] = []
    supports: list[str] = []

    # Revenue headwinds / supports
    if isinstance(decline_mag, (int, float)) and decline_mag < -0.10:
        headwinds.append(SECULAR_DECLINE_HEADWIND)
    elif revenue_trend_class == "GROWING":
        supports.append(GROWING_REVENUE_SUPPORT)

    # Accounting quality
    if earnings_quality == "LOW":
        headwinds.append(ACCOUNTING_QUALITY_HEADWIND)

    # Earnings quality (owner earnings) headwind / support
    if isinstance(cash_conversion_score, (int, float)) and cash_conversion_score < 1.0:
        headwinds.append(EARNINGS_QUALITY_HEADWIND)
    elif isinstance(cash_conversion_score, (int, float)) and cash_conversion_score >= 3.0:
        supports.append(STRONG_EARNINGS_SUPPORT)

    # Cash conversion support
    if isinstance(cash_conversion_score, (int, float)) and cash_conversion_score >= 3.0:
        supports.append(STRONG_CASH_CONVERSION_SUPPORT)

    # SBC burden
    if sbc_burden:
        headwinds.append(SBC_BURDEN_HEADWIND)

    # Capital allocation
    if allocation_grade == "F":
        headwinds.append(CAPITAL_ALLOCATION_HEADWIND)
    elif allocation_grade == "A":
        supports.append(OWNER_FRIENDLY_SUPPORT)

    # Leverage stress
    if leverage_stress == "HIGH":
        headwinds.append(LEVERAGE_STRESS_HEADWIND)
    elif leverage_stress == "NONE":
        supports.append(LOW_LEVERAGE_SUPPORT)

    # Cyclical peak
    if cycle_position == "PEAK":
        headwinds.append(CYCLICAL_PEAK_HEADWIND)
    elif cycle_position == "TROUGH":
        supports.append(TROUGH_EARNINGS_CONSERVATIVE)

    # Working capital trend headwinds
    for flag in working_capital_trends.get("trend_flags", []):
        if flag == "RECEIVABLES_DETERIORATING":
            headwinds.append(RECEIVABLES_DETERIORATING_HEADWIND)
        elif flag == "INVENTORY_BUILDING":
            headwinds.append(INVENTORY_BUILDING_HEADWIND)
        elif flag == "WORKING_CAPITAL_DRAG":
            headwinds.append(WORKING_CAPITAL_DRAG_HEADWIND)

    # ── Gate action ───────────────────────────────────────────────────────────
    gate_action = PROCEED
    gate_reason: str | None = None
    gate_reason_codes: list[str] = []

    # BLOCK conditions. The magnitude-only test needs corroboration: the
    # classifier itself must call it SECULAR_DECLINE or the latest year must
    # actually be declining — a VOLATILE series with a deep-but-recovering
    # trough should not hard-block (audit: secular-decline-block-single-
    # spike-peak).
    revenue_declining_now = int(revenue_trend.get("decline_years_consecutive") or 0) >= 1
    if (
        isinstance(decline_mag, (int, float))
        and decline_mag < -0.30
        and (revenue_trend_class == "SECULAR_DECLINE" or revenue_declining_now)
    ):
        gate_action = BLOCK
        gate_reason_codes.append("SEVERE_SECULAR_DECLINE")
    elif earnings_quality == "LOW" and leverage_stress == "HIGH":
        gate_action = BLOCK
        gate_reason_codes.append("LOW_QUALITY_HIGH_LEVERAGE")

    # BLOCK: Zero owner earnings — CFO below MAINTENANCE capex + SBC for the
    # 3 most recent CFO years. Charging FULL capex here while the canonical
    # owner-earnings convention charges maintenance-only excluded
    # growth-capex-heavy names whose own valuation stack computes strongly
    # positive owner earnings (audit: zero-oe-gate-full-capex-vs-
    # maintenance). maintenance_capex_pct is the growth-aware category ratio
    # computed above (1.0 for no-growth firms — their capex IS maintenance).
    cfo_series = facts.get("cfo") or []
    capex_series = facts.get("capex") or []
    sbc_series = facts.get("sbc") or []
    negative_oe_years = 0
    if gate_action != BLOCK and len(cfo_series) >= 3:
        recent_cfo = sorted(cfo_series, key=lambda x: x[0], reverse=True)[:3]
        for yr, cfo_val in recent_cfo:
            # Capex as a MAGNITUDE: on a filer-negated series the right-hand
            # side falls below zero and this block can never fire (the gate
            # companion of the writer's negated-capex owner earnings).
            capex_val = abs(float(next((v for y, v in capex_series if y == yr), 0) or 0.0))
            sbc_val = next((v for y, v in sbc_series if y == yr), 0)
            if cfo_val < (maintenance_capex_pct * capex_val + sbc_val):
                negative_oe_years += 1
        if negative_oe_years >= 3:
            gate_action = BLOCK
            gate_reason_codes.append("ZERO_OWNER_EARNINGS")

    # BLOCK: Book insolvency — latest equity < 0 AND revenue declining
    equity_series = facts.get("equity") or []
    if gate_action != BLOCK and equity_series:
        latest_equity = sorted(equity_series, key=lambda x: x[0], reverse=True)[0][1]
        rev_declining = (
            (isinstance(revenue_cagr_5y, (int, float)) and revenue_cagr_5y < 0)
            or revenue_trend_class in ("DECLINING", "SECULAR_DECLINE")
        )
        if latest_equity < 0 and rev_declining:
            gate_action = BLOCK
            gate_reason_codes.append("BOOK_INSOLVENCY")

    # BLOCK: Liquidity crisis — current ratio < 0.5 computed from the latest
    # COMMON fiscal year only. Independent latest-year reads mixed fiscal
    # years whenever XBRL tag coverage differed, spuriously blocking healthy
    # names (audit: liquidity-crisis-cross-year-ratio). No common year -> the
    # check is skipped, never computed on unalignable data.
    ca_series = facts.get("current_assets") or []
    cl_series = facts.get("current_liabilities") or []
    if gate_action != BLOCK and ca_series and cl_series:
        from app.valuation.valuation_writer import _get_value_for_year, _latest_common_year

        liq_year = _latest_common_year(facts, "current_assets", "current_liabilities")
        if liq_year is not None:
            latest_ca = _get_value_for_year(facts, "current_assets", liq_year)
            latest_cl = _get_value_for_year(facts, "current_liabilities", liq_year)
            if (
                isinstance(latest_ca, (int, float))
                and isinstance(latest_cl, (int, float))
                and latest_cl > 0
                and (latest_ca / latest_cl) < 0.5
            ):
                gate_action = BLOCK
                gate_reason_codes.append("LIQUIDITY_CRISIS")

    # BLOCK: Going concern language in filing (via solvency_scanner)
    # ADJUST: Full valuation allowance → SOLVENCY_CONCERN headwind
    going_concern_assertions: list[dict[str, Any]] = []
    try:
        from app.alpha.solvency_scanner import assess_solvency
        if require_filed_asof or issuer_cik is not None or issuer_aliases or db_path is not None:
            solvency = assess_solvency(
                ticker,
                as_of_date=as_of_date,
                require_filed_asof=require_filed_asof,
                issuer_cik=issuer_cik,
                aliases=issuer_aliases,
                db_path=db_path,
            )
        else:
            solvency = assess_solvency(ticker)
        going_concern_assertions = [
            assertion.to_dict() for assertion in solvency.going_concern_assertions
        ]
        if solvency.going_concern_language and gate_action != BLOCK:
            gate_action = BLOCK
            gate_reason_codes.append("GOING_CONCERN")
            headwinds.append("GOING_CONCERN")
        if solvency.valuation_allowance_full and gate_action != BLOCK:
            headwinds.append("SOLVENCY_CONCERN")
            if gate_action == PROCEED:
                gate_action = ADJUST
                gate_reason_codes.append("SOLVENCY_CONCERN")
    except Exception as exc:
        logger.warning("pre_valuation_gate: solvency_scanner failed: %s", exc)

    # ADJUST conditions (only if not already BLOCK)
    if gate_action != BLOCK:
        adjust_reasons: list[str] = []
        if isinstance(decline_mag, (int, float)) and -0.30 <= decline_mag < -0.10:
            adjust_reasons.append("MODERATE_DECLINE")
        if cycle_position == "PEAK":
            adjust_reasons.append("PEAK_CYCLE")
        if epv_quality in ("DECLINING_BASE", "DETERIORATING_BASE"):
            adjust_reasons.append(f"EPV_{epv_quality}")
        if earnings_quality == "LOW":
            adjust_reasons.append("LOW_EARNINGS_QUALITY")
        if sbc_burden:
            adjust_reasons.append("SBC_BURDEN")
        if allocation_grade == "F":
            adjust_reasons.append("DESTRUCTIVE_ALLOCATION")
        if adjust_reasons:
            gate_action = ADJUST
            gate_reason_codes.extend(adjust_reasons)

    # Build human-readable gate_reason
    if gate_reason_codes:
        gate_reason = "; ".join(gate_reason_codes)

    # ── Confidence classification ──────────────────────────────────────────────
    confidence_class = _classify_confidence(
        earnings_quality=earnings_quality,
        revenue_trend=revenue_trend,
        allocation_grade=allocation_grade,
        oe_quality_total=oe_quality_total,
        gate_action=gate_action,
    )

    # ── Terminal growth override (4-tier) ──────────────────────────────────────
    terminal_growth_override = _compute_terminal_growth(
        confidence_class, revenue_trend_class, allocation_grade,
    )

    # ── EPV adjustment ─────────────────────────────────────────────────────────
    if gate_action == BLOCK:
        epv_adjustment = EPV_ADJ_BLOCK
    elif (
        (isinstance(decline_mag, (int, float)) and -0.30 <= decline_mag < -0.10)
        or cycle_position == "PEAK"
    ):
        epv_adjustment = EPV_ADJ_USE_NORMALIZED
    else:
        epv_adjustment = EPV_ADJ_NONE

    # ── MOS threshold widening ─────────────────────────────────────────────────
    mos_threshold_widening = 0.15 if gate_action == ADJUST else 0.0

    return {
        "earnings_quality": earnings_quality,
        "cash_conversion_score": cash_conversion_score,
        "sbc_burden": sbc_burden,
        "cycle_position": cycle_position,
        "normalized_earnings": normalized_earnings,
        "allocation_grade": allocation_grade,
        # Filed split ratios (companyfacts), for callers that measure the share
        # count on the same split-corroborated basis (SBC trajectory).
        "share_split_ratios": split_rows,
        "leverage_stress": leverage_stress,
        "revenue_cagr_5y": revenue_cagr_5y,
        "revenue_cagr_3y": revenue_cagr_3y,
        "epv_quality": epv_quality,
        "revenue_trend_class": revenue_trend_class,
        "decline_years_consecutive": decline_years,
        "decline_magnitude_total": decline_mag,
        "terminal_growth_override": terminal_growth_override,
        "maintenance_capex_pct": maintenance_capex_pct,
        "working_capital": working_capital,
        "working_capital_trends": working_capital_trends,
        "gate_action": gate_action,
        "gate_reason": gate_reason,
        "confidence_class": confidence_class,
        "epv_adjustment": epv_adjustment,
        "valuation_headwinds": headwinds,
        "valuation_supports": supports,
        "gate_reason_codes": gate_reason_codes,
        "mos_threshold_widening": mos_threshold_widening,
        "negative_oe_years": negative_oe_years,
        "going_concern_assertions": going_concern_assertions,
    }
