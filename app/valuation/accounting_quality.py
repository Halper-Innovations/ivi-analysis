from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"
OK = "OK"

HIGH_ACCOUNTING_QUALITY = "HIGH_ACCOUNTING_QUALITY"
MODERATE_ACCOUNTING_QUALITY = "MODERATE_ACCOUNTING_QUALITY"
LOW_ACCOUNTING_QUALITY = "LOW_ACCOUNTING_QUALITY"
ACCOUNTING_QUALITY_UNKNOWN = "ACCOUNTING_QUALITY_UNKNOWN"

CASH_EARNINGS_SUPPORTIVE = "CASH_EARNINGS_SUPPORTIVE"
ACCOUNTING_QUALITY_MIXED = "ACCOUNTING_QUALITY_MIXED"
CASH_EARNINGS_HEADWIND = "CASH_EARNINGS_HEADWIND"
ACCOUNTING_QUALITY_UNCLEAR = "ACCOUNTING_QUALITY_UNCLEAR"

SIG_STRONG_CFO_TO_EARNINGS_CONVERSION = "STRONG_CFO_TO_EARNINGS_CONVERSION"
SIG_STRONG_FCF_TO_EARNINGS_CONVERSION = "STRONG_FCF_TO_EARNINGS_CONVERSION"
SIG_OWNER_EARNINGS_SUPPORT_REPORTED_RESULTS = "OWNER_EARNINGS_SUPPORT_REPORTED_RESULTS"
SIG_WORKING_CAPITAL_DISCIPLINE_PRESENT = "WORKING_CAPITAL_DISCIPLINE_PRESENT"
SIG_LOW_SBC_BURDEN = "LOW_SBC_BURDEN"
SIG_CASH_EARNINGS_CONSISTENT = "CASH_EARNINGS_CONSISTENT"

SIG_WEAK_CFO_TO_EARNINGS_CONVERSION = "WEAK_CFO_TO_EARNINGS_CONVERSION"
SIG_WEAK_FCF_TO_EARNINGS_CONVERSION = "WEAK_FCF_TO_EARNINGS_CONVERSION"
SIG_ACCRUAL_HEAVY_EARNINGS = "ACCRUAL_HEAVY_EARNINGS"
SIG_WORKING_CAPITAL_DRAIN = "WORKING_CAPITAL_DRAIN"
SIG_SBC_BURDEN_HEADWIND = "SBC_BURDEN_HEADWIND"
SIG_CASH_EARNINGS_DIVERGENCE = "CASH_EARNINGS_DIVERGENCE"
SIG_REPORTED_EARNINGS_NOT_OWNER_RELEVANT = "REPORTED_EARNINGS_NOT_OWNER_RELEVANT"

SIG_ACCRUALS_AGGRESSIVE = "ACCRUALS_AGGRESSIVE"
SIG_ACCRUALS_RISING = "ACCRUALS_RISING"
SIG_ACCRUALS_CONSERVATIVE = "ACCRUALS_CONSERVATIVE"

REASON_MISSING_ACCOUNTING_INPUTS = "MISSING_ACCOUNTING_INPUTS"
REASON_INSUFFICIENT_ACCOUNTING_HISTORY = "INSUFFICIENT_ACCOUNTING_HISTORY"
REASON_ACCOUNTING_EVIDENCE_THIN = "ACCOUNTING_EVIDENCE_THIN"
REASON_HIGH_ACCOUNTING_QUALITY_SUPPORT = "HIGH_ACCOUNTING_QUALITY_SUPPORT"
REASON_LOW_ACCOUNTING_QUALITY_HEADWIND = "LOW_ACCOUNTING_QUALITY_HEADWIND"
REASON_ACCRUAL_HEAVY_EARNINGS_HEADWIND = "ACCRUAL_HEAVY_EARNINGS_HEADWIND"
REASON_WEAK_CASH_CONVERSION_HEADWIND = "WEAK_CASH_CONVERSION_HEADWIND"

# The recent window every ratio, growth and accrual test is judged over, and
# the usable years a median needs (a majority of that window).
_RECENT_WINDOW_YEARS = 3
_MIN_RATIO_YEARS = 2

_CLASS_ORDER = {
    HIGH_ACCOUNTING_QUALITY: 0,
    MODERATE_ACCOUNTING_QUALITY: 1,
    ACCOUNTING_QUALITY_UNKNOWN: 2,
    LOW_ACCOUNTING_QUALITY: 3,
}


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _compute_accruals_signals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accrual ratio signals from time-series rows.

    accrual_ratio = (net_income - cfo) / total_assets per year.

    Returns dict with accrual_ratios (per-year) and accruals_signals (flag list).
    """
    accrual_ratios: list[dict[str, Any]] = []
    signals: list[str] = []

    # Judged over the same recent window as the conversion ratios. Counted over
    # the whole history, five stale conservative years and three recent
    # aggressive ones fired both signals, and their +1 and -1 cancelled.
    recent_years = _recent_years(rows)
    for row in rows:
        if int(row.get("year") or 0) not in recent_years:
            continue
        ni = row.get("net_income")
        cfo = row.get("cfo")
        ta = row.get("total_assets")
        if not (_is_num(ni) and _is_num(cfo) and _is_num(ta) and float(ta) > 0):
            continue
        ratio = (float(ni) - float(cfo)) / float(ta)
        accrual_ratios.append({"year": row.get("year"), "ratio": round(ratio, 4)})

    if not accrual_ratios:
        return {"accrual_ratios": [], "accruals_signals": []}

    ratios = [ar["ratio"] for ar in accrual_ratios]

    # ACCRUALS_AGGRESSIVE: ratio > 0.10 for 2+ years
    aggressive_count = sum(1 for r in ratios if r > 0.10)
    if aggressive_count >= 2:
        signals.append(SIG_ACCRUALS_AGGRESSIVE)

    # ACCRUALS_RISING: ratio > 0.05 AND trending up (latest > median of earlier)
    if not signals and len(ratios) >= 2:
        latest = ratios[-1]
        if latest > 0.05:
            med = median(ratios[:-1]) if len(ratios) > 2 else ratios[0]
            if latest > med:
                signals.append(SIG_ACCRUALS_RISING)

    # ACCRUALS_CONSERVATIVE: ratio < 0.02 for 3+ years
    conservative_count = sum(1 for r in ratios if r < 0.02)
    if conservative_count >= 3:
        signals.append(SIG_ACCRUALS_CONSERVATIVE)

    return {"accrual_ratios": accrual_ratios, "accruals_signals": signals}


def _coalesce_status(*values: Any, fallback: str = UNKNOWN) -> str:
    for value in values:
        token = str(value or "").strip().upper()
        if token:
            return token
    return fallback


def _series_rows(fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("year") or 0))
    return rows


def _fundamentals_rows_from_owner_payload(owner_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in owner_payload.get("series") or []:
        if not isinstance(row, dict):
            continue
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        rows.append(
            {
                "year": year,
                "cfo": row.get("cfo", UNKNOWN),
                "capex": row.get("capex", UNKNOWN),
                "owner_earnings": row.get("owner_earnings", UNKNOWN),
            }
        )
    rows.sort(key=lambda row: int(row.get("year") or 0))
    return rows


def _fundamentals_for_score_row(score_row: dict[str, Any]) -> dict[str, Any]:
    fundamentals = (
        score_row.get("fundamentals_detail")
        if isinstance(score_row.get("fundamentals_detail"), dict)
        else {}
    )
    if isinstance(fundamentals, dict) and isinstance(fundamentals.get("rows"), list) and fundamentals.get("rows"):
        return fundamentals
    owner_payload = (
        score_row.get("owner_earnings_detail")
        if isinstance(score_row.get("owner_earnings_detail"), dict)
        else {}
    )
    owner_rows = _fundamentals_rows_from_owner_payload(owner_payload)
    if owner_rows:
        return {
            "ticker": str(score_row.get("ticker") or "").upper(),
            "rows": owner_rows,
            "derived_from": [str(ref) for ref in (owner_payload.get("derived_from") or []) if str(ref).strip()],
        }
    return {}


def _collect_derived_from(*payloads: dict[str, Any], row_refs: list[Any] | None = None) -> list[str]:
    refs: list[Any] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        refs.extend(payload.get("derived_from") or [])
        claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
        for claim in claims.values():
            if isinstance(claim, dict):
                refs.extend(claim.get("derived_from") or [])
    refs.extend(row_refs or [])
    return _dedupe(refs)


def _recent_years(rows: list[dict[str, Any]], *, window: int = _RECENT_WINDOW_YEARS) -> set[int]:
    """The last ``window`` fiscal years the series covers, whatever each row carries."""
    years = sorted({int(row.get("year") or 0) for row in rows} - {0})
    return set(years[-max(1, int(window)) :])


def _recent_common_ratios(
    rows: list[dict[str, Any]],
    numerator_key: str,
    denominator_key: str,
    *,
    window: int = 3,
    require_positive_denominator: bool = True,
) -> tuple[list[float], list[str]]:
    # The window is taken FIRST, over the most recent years of the series, and
    # only then are the unusable years dropped. Filtering first made "the last
    # three years" mean "the last three years with usable figures, however
    # old": a company profitable to 2021 and loss-making 2022-2024 was scored
    # in 2025 on its 2019-2021 profits and came back HIGH quality, and so was
    # one whose 2022-2024 cash flows were simply missing. A loss year or a
    # missing figure now shrinks the sample honestly instead of importing a
    # stale one.
    recent_years = _recent_years(rows, window=window)
    candidates: list[tuple[int, Any, Any]] = []
    for row in rows:
        year = int(row.get("year") or 0)
        numerator = row.get(numerator_key, UNKNOWN)
        denominator = row.get(denominator_key, UNKNOWN)
        if year not in recent_years or not _is_num(numerator) or not _is_num(denominator):
            continue
        candidates.append((year, float(numerator), float(denominator)))
    candidates = sorted(candidates, key=lambda item: item[0])

    ratios: list[float] = []
    refs: list[str] = []
    for year, numerator_value, denominator_value in candidates:
        if denominator_value == 0.0:
            continue
        if require_positive_denominator and denominator_value <= 0.0:
            continue
        ratios.append(numerator_value / denominator_value)
        refs.extend(
            [
                f"fundamentals.rows[{year}].{numerator_key}",
                f"fundamentals.rows[{year}].{denominator_key}",
            ]
        )
    return ratios, _dedupe(refs)


def _paired_recent_cagr(
    rows: list[dict[str, Any]],
    key: str,
    base_key: str,
) -> tuple[float | str, float | str, list[str]]:
    """Growth of ``key`` and of ``base_key`` over the SAME recent years.

    First-to-last over every row compared receivables growth since 2016 with
    conversion ratios over the last three years, and a receivables series that
    started later than revenue was compared with revenue over a different span.
    Both series now run from the first to the last recent year that carries
    both figures.
    """
    recent_years = _recent_years(rows)
    paired: list[tuple[int, float, float]] = []
    for row in rows:
        year = int(row.get("year") or 0)
        value = row.get(key, UNKNOWN)
        base = row.get(base_key, UNKNOWN)
        if year not in recent_years or not _is_num(value) or not _is_num(base):
            continue
        paired.append((year, float(value), float(base)))
    paired.sort(key=lambda item: item[0])
    if len(paired) < 2:
        return UNKNOWN, UNKNOWN, []
    (start_year, start_value, start_base), (end_year, end_value, end_base) = paired[0], paired[-1]
    refs = _dedupe(
        [f"fundamentals.rows[{year}].{name}" for year in (start_year, end_year) for name in (key, base_key)]
    )
    periods = max(1, int(end_year - start_year))

    def _cagr(start: float, end: float) -> float | str:
        if start <= 0.0 or end <= 0.0:
            return UNKNOWN
        return (end / start) ** (1.0 / float(periods)) - 1.0

    return _cagr(start_value, end_value), _cagr(start_base, end_base), refs


def _latest_median(values: list[float]) -> float | str:
    # A median over one usable year is that year: one profitable year after
    # two losses used to read as a three-year median and earn HIGH quality.
    # The median needs a majority of the window.
    if len(values) < _MIN_RATIO_YEARS:
        return UNKNOWN
    return float(median(values))


def _summary_for_class(*, cls: str, support_signals: list[str], headwind_signals: list[str]) -> str:
    if cls == HIGH_ACCOUNTING_QUALITY:
        return "reported economics appear cash-backed and owner-relevant across conversion and discipline signals"
    if cls == MODERATE_ACCOUNTING_QUALITY:
        return "accounting quality is acceptable but mixed; reported strength is only partially cash-backed"
    if cls == LOW_ACCOUNTING_QUALITY:
        return "reported economics are not sufficiently cash-backed; conversion and owner relevance are weak"
    if headwind_signals:
        return "accounting quality is unclear with visible headwinds and insufficient evidence"
    if support_signals:
        return "some cash-earnings support is present, but evidence remains too thin for confidence"
    return "evidence too thin to judge whether reported economics are genuinely cash-backed"


def compute_accounting_quality(
    ticker: str,
    as_of_date: str,
    *,
    fundamentals: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    capital_allocation_discipline_payload: dict[str, Any] | None = None,
    reinvestment_efficiency_payload: dict[str, Any] | None = None,
    price_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    fcf_status: Any = UNKNOWN,
    row_derived_from: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg

    ticker_norm = str(ticker or "").strip().upper()
    fundamentals = fundamentals if isinstance(fundamentals, dict) else {}
    owner_quality_payload = owner_quality_payload if isinstance(owner_quality_payload, dict) else {}
    cap_alloc_payload = (
        capital_allocation_discipline_payload
        if isinstance(capital_allocation_discipline_payload, dict)
        else {}
    )
    reinvestment_payload = (
        reinvestment_efficiency_payload
        if isinstance(reinvestment_efficiency_payload, dict)
        else {}
    )
    rows = _series_rows(fundamentals)

    cfo_to_ni_values, cfo_to_ni_refs = _recent_common_ratios(rows, "cfo", "net_income")
    fcf_to_ni_values, fcf_to_ni_refs = _recent_common_ratios(rows, "fcf", "net_income")
    owner_to_ni_values, owner_to_ni_refs = _recent_common_ratios(rows, "owner_earnings", "net_income")
    receivables_growth, receivables_revenue_growth, receivables_growth_refs = _paired_recent_cagr(
        rows, "accounts_receivable", "revenue"
    )
    inventory_growth, inventory_revenue_growth, inventory_growth_refs = _paired_recent_cagr(
        rows, "inventory", "revenue"
    )
    # "sbc" is the name the ingest actually writes (companyfacts_facts.line_item);
    # neither "sbc_total" nor "stock_based_compensation" is ever written, so this
    # test could not fire on any production row until 2026-09-02 — the data was
    # there and the key was wrong. All three names are read, most specific first.
    for sbc_key in ("sbc_total", "stock_based_compensation", "sbc"):
        sbc_to_revenue_values, sbc_to_revenue_refs = _recent_common_ratios(
            rows, sbc_key, "revenue"
        )
        if sbc_to_revenue_values:
            break

    cfo_to_ni = _latest_median(cfo_to_ni_values)
    fcf_to_ni = _latest_median(fcf_to_ni_values)
    owner_to_ni = _latest_median(owner_to_ni_values)
    sbc_to_revenue = _latest_median(sbc_to_revenue_values)

    cash_conversion_score = owner_quality_payload.get("cash_conversion_score", UNKNOWN)
    oe_quality_total = owner_quality_payload.get("oe_quality_total", UNKNOWN)
    cap_alloc_class = str(
        cap_alloc_payload.get("capital_allocation_discipline_class") or "CAPITAL_ALLOCATION_UNKNOWN"
    ).upper()
    reinvestment_class = str(
        reinvestment_payload.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
    ).upper()

    support_signals: list[str] = []
    headwind_signals: list[str] = []
    reason_codes: list[str] = []

    if _is_num(cfo_to_ni) and float(cfo_to_ni) >= 0.95:
        support_signals.append(SIG_STRONG_CFO_TO_EARNINGS_CONVERSION)
    elif _is_num(cfo_to_ni) and float(cfo_to_ni) < 0.75:
        headwind_signals.extend([SIG_WEAK_CFO_TO_EARNINGS_CONVERSION, SIG_ACCRUAL_HEAVY_EARNINGS])

    if _is_num(fcf_to_ni) and float(fcf_to_ni) >= 0.75:
        support_signals.append(SIG_STRONG_FCF_TO_EARNINGS_CONVERSION)
    elif _is_num(fcf_to_ni) and float(fcf_to_ni) < 0.50:
        headwind_signals.append(SIG_WEAK_FCF_TO_EARNINGS_CONVERSION)

    if _is_num(owner_to_ni) and float(owner_to_ni) >= 0.80:
        support_signals.append(SIG_OWNER_EARNINGS_SUPPORT_REPORTED_RESULTS)
    elif _is_num(owner_to_ni) and float(owner_to_ni) < 0.60:
        headwind_signals.append(SIG_REPORTED_EARNINGS_NOT_OWNER_RELEVANT)

    # Each balance is compared with revenue growth over its own paired years.
    receivables_pair = (
        (float(receivables_growth), float(receivables_revenue_growth))
        if _is_num(receivables_growth) and _is_num(receivables_revenue_growth)
        else None
    )
    inventory_pair = (
        (float(inventory_growth), float(inventory_revenue_growth))
        if _is_num(inventory_growth) and _is_num(inventory_revenue_growth)
        else None
    )
    working_capital_pairs = [pair for pair in (receivables_pair, inventory_pair) if pair is not None]
    if any(
        revenue_growth > 0.03 and balance_growth > revenue_growth + 0.05
        for balance_growth, revenue_growth in working_capital_pairs
    ):
        headwind_signals.append(SIG_WORKING_CAPITAL_DRAIN)
    elif any(
        revenue_growth >= 0.0 and balance_growth <= revenue_growth
        for balance_growth, revenue_growth in working_capital_pairs
    ):
        support_signals.append(SIG_WORKING_CAPITAL_DISCIPLINE_PRESENT)

    if _is_num(sbc_to_revenue) and float(sbc_to_revenue) <= 0.03:
        support_signals.append(SIG_LOW_SBC_BURDEN)
    elif _is_num(sbc_to_revenue) and float(sbc_to_revenue) >= 0.05:
        headwind_signals.append(SIG_SBC_BURDEN_HEADWIND)

    if (
        _is_num(cfo_to_ni)
        and _is_num(fcf_to_ni)
        and float(cfo_to_ni) >= 0.90
        and float(fcf_to_ni) >= 0.70
    ):
        support_signals.append(SIG_CASH_EARNINGS_CONSISTENT)
    elif (
        _is_num(cfo_to_ni)
        and _is_num(fcf_to_ni)
        and (float(cfo_to_ni) < 0.80 or float(fcf_to_ni) < 0.60)
    ):
        headwind_signals.append(SIG_CASH_EARNINGS_DIVERGENCE)

    if _is_num(cash_conversion_score) and float(cash_conversion_score) >= 3.0:
        support_signals.append(SIG_CASH_EARNINGS_CONSISTENT)
    elif _is_num(cash_conversion_score) and float(cash_conversion_score) <= 1.0:
        headwind_signals.append(SIG_WEAK_FCF_TO_EARNINGS_CONVERSION)

    # The guard compares a REINVESTMENT class, so it must name a reinvestment
    # constant: against LOW_ACCOUNTING_QUALITY it was true for every input and
    # the support point was granted to a disciplined allocator whose
    # reinvestment efficiency had just been called LOW two lines below.
    if (
        cap_alloc_class == "OWNER_FRIENDLY_DISCIPLINED"
        and reinvestment_class != "LOW_REINVESTMENT_EFFICIENCY"
    ):
        support_signals.append(SIG_OWNER_EARNINGS_SUPPORT_REPORTED_RESULTS)
    if reinvestment_class == "LOW_REINVESTMENT_EFFICIENCY":
        headwind_signals.append(SIG_REPORTED_EARNINGS_NOT_OWNER_RELEVANT)

    # Accruals ratio signals
    accruals_result = _compute_accruals_signals(rows)
    for sig in accruals_result.get("accruals_signals", []):
        if sig == SIG_ACCRUALS_AGGRESSIVE:
            headwind_signals.append(sig)
        elif sig == SIG_ACCRUALS_RISING:
            headwind_signals.append(sig)
        elif sig == SIG_ACCRUALS_CONSERVATIVE:
            support_signals.append(sig)

    evidence_known = any(
        _is_num(value)
        for value in [cfo_to_ni, fcf_to_ni, owner_to_ni, sbc_to_revenue]
    )
    # A block on an input this module reads (the facts, the free cash flow it
    # divides, or a history too short for a window) withholds the class
    # whatever fired: it used to apply only when no signal fired, so a strong
    # series on blocked facts came back HIGH. The signals stay listed for the
    # reader. The share status is not an input here (no share figure enters
    # any test), so it keeps its old role and withholds a class only when
    # nothing fired.
    evidence_blocked = (
        _coalesce_status(facts_status) != OK
        or (_coalesce_status(fcf_status) != OK and not _is_num(fcf_to_ni))
        or len(rows) < 2
    )
    shares_unresolved = _coalesce_status(shares_status) != OK

    if not evidence_known:
        reason_codes.extend([REASON_MISSING_ACCOUNTING_INPUTS, REASON_ACCOUNTING_EVIDENCE_THIN])
        cls = ACCOUNTING_QUALITY_UNKNOWN
        caution = ACCOUNTING_QUALITY_UNCLEAR
    elif evidence_blocked or (shares_unresolved and not support_signals and not headwind_signals):
        reason_codes.extend([REASON_INSUFFICIENT_ACCOUNTING_HISTORY, REASON_ACCOUNTING_EVIDENCE_THIN])
        cls = ACCOUNTING_QUALITY_UNKNOWN
        caution = ACCOUNTING_QUALITY_UNCLEAR
    else:
        support_strength = 0
        headwind_strength = 0
        if SIG_STRONG_CFO_TO_EARNINGS_CONVERSION in support_signals:
            support_strength += 2
        if SIG_STRONG_FCF_TO_EARNINGS_CONVERSION in support_signals:
            support_strength += 2
        if SIG_OWNER_EARNINGS_SUPPORT_REPORTED_RESULTS in support_signals:
            support_strength += 1
        if SIG_WORKING_CAPITAL_DISCIPLINE_PRESENT in support_signals:
            support_strength += 1
        if SIG_LOW_SBC_BURDEN in support_signals:
            support_strength += 1
        if SIG_CASH_EARNINGS_CONSISTENT in support_signals:
            support_strength += 1
        if _is_num(oe_quality_total) and float(oe_quality_total) >= 8.0:
            support_strength += 1
        if SIG_ACCRUALS_CONSERVATIVE in support_signals:
            support_strength += 1

        if SIG_WEAK_CFO_TO_EARNINGS_CONVERSION in headwind_signals:
            headwind_strength += 2
        if SIG_ACCRUAL_HEAVY_EARNINGS in headwind_signals:
            headwind_strength += 2
        if SIG_WEAK_FCF_TO_EARNINGS_CONVERSION in headwind_signals:
            headwind_strength += 2
        if SIG_WORKING_CAPITAL_DRAIN in headwind_signals:
            headwind_strength += 1
        if SIG_SBC_BURDEN_HEADWIND in headwind_signals:
            headwind_strength += 1
        if SIG_CASH_EARNINGS_DIVERGENCE in headwind_signals:
            headwind_strength += 1
        if SIG_REPORTED_EARNINGS_NOT_OWNER_RELEVANT in headwind_signals:
            headwind_strength += 1
        if SIG_ACCRUALS_AGGRESSIVE in headwind_signals:
            headwind_strength += 1

        if support_strength >= 4 and headwind_strength == 0:
            cls = HIGH_ACCOUNTING_QUALITY
            caution = CASH_EARNINGS_SUPPORTIVE
            reason_codes.append(REASON_HIGH_ACCOUNTING_QUALITY_SUPPORT)
        elif headwind_strength >= 3 and support_strength <= 2:
            cls = LOW_ACCOUNTING_QUALITY
            caution = CASH_EARNINGS_HEADWIND
            reason_codes.extend(
                [
                    REASON_LOW_ACCOUNTING_QUALITY_HEADWIND,
                    REASON_WEAK_CASH_CONVERSION_HEADWIND,
                ]
            )
        else:
            cls = MODERATE_ACCOUNTING_QUALITY
            caution = ACCOUNTING_QUALITY_MIXED

    if SIG_ACCRUAL_HEAVY_EARNINGS in headwind_signals:
        reason_codes.append(REASON_ACCRUAL_HEAVY_EARNINGS_HEADWIND)
    if any(
        signal in headwind_signals
        for signal in {
            SIG_WEAK_CFO_TO_EARNINGS_CONVERSION,
            SIG_WEAK_FCF_TO_EARNINGS_CONVERSION,
            SIG_CASH_EARNINGS_DIVERGENCE,
        }
    ):
        reason_codes.append(REASON_WEAK_CASH_CONVERSION_HEADWIND)
    if cls == ACCOUNTING_QUALITY_UNKNOWN:
        reason_codes.append(REASON_ACCOUNTING_EVIDENCE_THIN)

    support_signals = _dedupe(support_signals)
    headwind_signals = _dedupe(headwind_signals)
    reason_codes = _dedupe(reason_codes)
    derived_from = _collect_derived_from(
        fundamentals,
        owner_quality_payload,
        cap_alloc_payload,
        reinvestment_payload,
        row_refs=row_derived_from,
    )
    derived_from = _dedupe(
        derived_from
        + cfo_to_ni_refs
        + fcf_to_ni_refs
        + owner_to_ni_refs
        + receivables_growth_refs
        + inventory_growth_refs
        + sbc_to_revenue_refs
    )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "cfo_to_net_income_median_3y": _to_num(cfo_to_ni),
        "fcf_to_net_income_median_3y": _to_num(fcf_to_ni),
        "owner_earnings_to_net_income_median_3y": _to_num(owner_to_ni),
        "receivables_growth_vs_revenue_growth_proxy": (
            _to_num(receivables_pair[0] - receivables_pair[1])
            if receivables_pair is not None
            else UNKNOWN
        ),
        "inventory_growth_vs_revenue_growth_proxy": (
            _to_num(inventory_pair[0] - inventory_pair[1])
            if inventory_pair is not None
            else UNKNOWN
        ),
        "sbc_to_revenue_median_3y": _to_num(sbc_to_revenue),
        "accounting_quality_class": cls,
        "accounting_quality_reason_codes": reason_codes,
        "cash_earnings_support_signals": support_signals,
        "cash_earnings_headwind_signals": headwind_signals,
        "primary_accounting_caution": caution,
        "cash_earnings_discipline_summary": _summary_for_class(
            cls=cls,
            support_signals=support_signals,
            headwind_signals=headwind_signals,
        ),
        "derived_from": derived_from,
        "claims": {
            "accounting_quality_class": {
                "value": cls,
                "status": OK if cls != ACCOUNTING_QUALITY_UNKNOWN else UNKNOWN,
                "reason_code": reason_codes[0] if reason_codes else UNKNOWN,
                "derived_from": derived_from,
            },
            "primary_accounting_caution": {
                "value": caution,
                "status": OK if caution != ACCOUNTING_QUALITY_UNCLEAR else UNKNOWN,
                "reason_code": reason_codes[0] if reason_codes else UNKNOWN,
                "derived_from": derived_from,
            },
        },
        "generated_at": utc_now_iso(),
    }


def write_accounting_quality_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg

    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("accounting_quality_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("accounting_quality_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    caution_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    known_count = 0
    unknown_count = 0

    for ticker in sorted({str(value or "").strip().upper() for value in tickers if str(value or "").strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            score_row = next(
                (
                    row
                    for row in (scoreboard_rows or [])
                    if isinstance(row, dict) and str(row.get("ticker") or "").strip().upper() == ticker
                ),
                {},
            )
            detail = compute_accounting_quality(
                ticker=ticker,
                as_of_date=as_of_date,
                fundamentals=_fundamentals_for_score_row(score_row),
                owner_quality_payload=score_row.get("owner_earnings_quality_detail")
                if isinstance(score_row.get("owner_earnings_quality_detail"), dict)
                else {},
                capital_allocation_discipline_payload=score_row.get("capital_allocation_discipline_detail")
                if isinstance(score_row.get("capital_allocation_discipline_detail"), dict)
                else {},
                reinvestment_efficiency_payload=score_row.get("reinvestment_efficiency_detail")
                if isinstance(score_row.get("reinvestment_efficiency_detail"), dict)
                else {},
                price_status=score_row.get("price_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                fcf_status=score_row.get("fcf_status", UNKNOWN),
                row_derived_from=list(score_row.get("derived_from") or []),
            )

        cls = str(detail.get("accounting_quality_class") or ACCOUNTING_QUALITY_UNKNOWN)
        caution = str(detail.get("primary_accounting_caution") or ACCOUNTING_QUALITY_UNCLEAR)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        caution_counts[caution] = caution_counts.get(caution, 0) + 1
        if cls == ACCOUNTING_QUALITY_UNKNOWN:
            unknown_count += 1
        else:
            known_count += 1
        for code in detail.get("accounting_quality_reason_codes") or []:
            token = str(code or "").strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1
        rows.append(
            {
                "ticker": ticker,
                "accounting_quality_class": cls,
                "accounting_quality_reason_codes": [
                    str(code)
                    for code in (detail.get("accounting_quality_reason_codes") or [])
                    if str(code).strip()
                ],
                "cash_earnings_support_signals": [
                    str(code)
                    for code in (detail.get("cash_earnings_support_signals") or [])
                    if str(code).strip()
                ],
                "cash_earnings_headwind_signals": [
                    str(code)
                    for code in (detail.get("cash_earnings_headwind_signals") or [])
                    if str(code).strip()
                ],
                "primary_accounting_caution": caution,
                "cash_earnings_discipline_summary": str(
                    detail.get("cash_earnings_discipline_summary") or ""
                ),
                "derived_from": [
                    str(ref) for ref in (detail.get("derived_from") or []) if str(ref).strip()
                ],
            }
        )

    def _class_rows(value: str) -> list[dict[str, Any]]:
        subset = [row for row in rows if str(row.get("accounting_quality_class") or "") == value]
        subset.sort(key=lambda row: str(row.get("ticker") or ""))
        return subset

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "known_count": known_count,
        "unknown_count": unknown_count,
        "counts_by_accounting_quality_class": dict(
            sorted(class_counts.items(), key=lambda item: (_CLASS_ORDER.get(item[0], 99), item[0]))
        ),
        "counts_by_primary_accounting_caution": dict(
            sorted(caution_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ),
        "top_10_high_accounting_quality": [
            {
                "ticker": str(row.get("ticker") or ""),
                "accounting_quality_reason_codes": [
                    str(code)
                    for code in (row.get("accounting_quality_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows(HIGH_ACCOUNTING_QUALITY)[:10]
        ],
        "top_10_low_accounting_quality": [
            {
                "ticker": str(row.get("ticker") or ""),
                "accounting_quality_reason_codes": [
                    str(code)
                    for code in (row.get("accounting_quality_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows(LOW_ACCOUNTING_QUALITY)[:10]
        ],
        "most_common_accounting_quality_reason_codes": [
            {"reason_code": str(name), "count": int(count)}
            for name, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ][:20],
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["accounting_quality_path"] = str(output_path)
    return payload


def _accounting_quality_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "accounting_quality.json",
        cfg.sectors_dir / run_id / "accounting_quality.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_accounting_quality(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _accounting_quality_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "accounting_quality_path": str(path) if path is not None else "",
        }

    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "known_count": int(payload.get("known_count") or 0),
        "unknown_count": int(payload.get("unknown_count") or 0),
        "counts_by_accounting_quality_class": (
            payload.get("counts_by_accounting_quality_class")
            if isinstance(payload.get("counts_by_accounting_quality_class"), dict)
            else {}
        ),
        "counts_by_primary_accounting_caution": (
            payload.get("counts_by_primary_accounting_caution")
            if isinstance(payload.get("counts_by_primary_accounting_caution"), dict)
            else {}
        ),
        "top_10_high_accounting_quality": [
            row for row in (payload.get("top_10_high_accounting_quality") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_low_accounting_quality": [
            row for row in (payload.get("top_10_low_accounting_quality") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "most_common_accounting_quality_reason_codes": [
            row
            for row in (payload.get("most_common_accounting_quality_reason_codes") or [])
            if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "accounting_quality_path": str(path),
    }
