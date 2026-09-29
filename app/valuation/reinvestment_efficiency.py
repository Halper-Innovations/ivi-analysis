from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.valuation.maintenance_capex_discipline import (
    HIGH_ASSET_INTENSITY,
    LOW_ASSET_INTENSITY,
    LOW_MAINTENANCE_CAPEX_CREDIBILITY,
    MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN,
    REASON_HIGH_ASSET_INTENSITY_HEADWIND,
    REASON_LOW_ASSET_INTENSITY_SUPPORT,
    REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND,
    REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN,
)


UNKNOWN = "UNKNOWN"
OK = "OK"

HIGH_REINVESTMENT_EFFICIENCY = "HIGH_REINVESTMENT_EFFICIENCY"
MODERATE_REINVESTMENT_EFFICIENCY = "MODERATE_REINVESTMENT_EFFICIENCY"
LOW_REINVESTMENT_EFFICIENCY = "LOW_REINVESTMENT_EFFICIENCY"
REINVESTMENT_EFFICIENCY_UNKNOWN = "REINVESTMENT_EFFICIENCY_UNKNOWN"

REINVESTMENT_APPEARS_PRODUCTIVE = "REINVESTMENT_APPEARS_PRODUCTIVE"
REINVESTMENT_MIXED = "REINVESTMENT_MIXED"
REINVESTMENT_HEADWIND = "REINVESTMENT_HEADWIND"
REINVESTMENT_UNCLEAR = "REINVESTMENT_UNCLEAR"

SIG_OWNER_EARNINGS_GROWTH_PRESENT = "OWNER_EARNINGS_GROWTH_PRESENT"
SIG_FCF_GROWTH_PRESENT = "FCF_GROWTH_PRESENT"
SIG_REVENUE_GROWTH_WITH_MARGIN_STABILITY = "REVENUE_GROWTH_WITH_MARGIN_STABILITY"
SIG_CAPEX_LIGHT_SCALING_PRESENT = "CAPEX_LIGHT_SCALING_PRESENT"
SIG_PER_SHARE_PROGRESS_PRESENT = "PER_SHARE_PROGRESS_PRESENT"
SIG_GROSS_MARGIN_DURABILITY_SUPPORTS_REINVESTMENT = "GROSS_MARGIN_DURABILITY_SUPPORTS_REINVESTMENT"
SIG_LOW_CAPITAL_BURDEN_RELATIVE_TO_GROWTH = "LOW_CAPITAL_BURDEN_RELATIVE_TO_GROWTH"

SIG_REVENUE_GROWTH_WITHOUT_OWNER_OUTCOME = "REVENUE_GROWTH_WITHOUT_OWNER_OUTCOME"
SIG_HIGH_CAPEX_BURDEN = "HIGH_CAPEX_BURDEN"
SIG_CFO_GROWTH_WEAK_RELATIVE_TO_REINVESTMENT = "CFO_GROWTH_WEAK_RELATIVE_TO_REINVESTMENT"
SIG_FCF_GROWTH_WEAK_RELATIVE_TO_REINVESTMENT = "FCF_GROWTH_WEAK_RELATIVE_TO_REINVESTMENT"
SIG_PER_SHARE_PROGRESS_ABSENT = "PER_SHARE_PROGRESS_ABSENT"
SIG_CAPITAL_HUNGRY_GROWTH = "CAPITAL_HUNGRY_GROWTH"
SIG_REINVESTMENT_PRODUCTIVITY_UNCLEAR = "REINVESTMENT_PRODUCTIVITY_UNCLEAR"

REASON_MISSING_REINVESTMENT_INPUTS = "MISSING_REINVESTMENT_INPUTS"
REASON_INSUFFICIENT_REINVESTMENT_HISTORY = "INSUFFICIENT_REINVESTMENT_HISTORY"
REASON_REINVESTMENT_EVIDENCE_THIN = "REINVESTMENT_EVIDENCE_THIN"
REASON_PRODUCTIVE_REINVESTMENT_SUPPORT = "PRODUCTIVE_REINVESTMENT_SUPPORT"
REASON_REINVESTMENT_MIXED = "REINVESTMENT_MIXED"
REASON_CAPITAL_HUNGRY_GROWTH_HEADWIND = "CAPITAL_HUNGRY_GROWTH_HEADWIND"
REASON_GROWTH_WITHOUT_OWNER_OUTCOME = "GROWTH_WITHOUT_OWNER_OUTCOME"

_CLASS_ORDER = {
    HIGH_REINVESTMENT_EFFICIENCY: 0,
    MODERATE_REINVESTMENT_EFFICIENCY: 1,
    REINVESTMENT_EFFICIENCY_UNKNOWN: 2,
    LOW_REINVESTMENT_EFFICIENCY: 3,
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
    series = owner_payload.get("series") if isinstance(owner_payload.get("series"), list) else []
    for row in series:
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
        refs = [str(ref) for ref in (owner_payload.get("derived_from") or []) if str(ref).strip()]
        return {
            "ticker": str(score_row.get("ticker") or "").upper(),
            "rows": owner_rows,
            "derived_from": refs,
        }
    return {}


_CAGR_WINDOW_YEARS = 5
_BURDEN_WINDOW_YEARS = 3


def _series_cagr(rows: list[dict[str, Any]], key: str) -> tuple[float | str, list[str]]:
    numeric = []
    refs: list[str] = []
    for row in rows:
        value = row.get(key, UNKNOWN)
        if not _is_num(value):
            continue
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        numeric.append((year, float(value)))
    # The output is labelled a five-year rate: compound over the last five
    # intervals, not from the first row on file.
    numeric = sorted(numeric, key=lambda item: item[0])[-(_CAGR_WINDOW_YEARS + 1) :]
    refs = [f"fundamentals.rows[{year}].{key}" for year, _ in numeric]
    if len(numeric) < 2:
        return UNKNOWN, _dedupe(refs)
    start_year, start_value = numeric[0]
    end_year, end_value = numeric[-1]
    periods = max(1, int(end_year - start_year))
    if start_value <= 0.0 or end_value <= 0.0:
        return UNKNOWN, _dedupe(refs)
    cagr = (end_value / start_value) ** (1.0 / float(periods)) - 1.0
    return float(cagr), _dedupe(refs)


def _capex_burden(rows: list[dict[str, Any]]) -> tuple[float | str, list[str]]:
    dated: list[tuple[int, float]] = []
    latest_year = max((int(row.get("year") or 0) for row in rows), default=0)
    for row in rows:
        year = int(row.get("year") or 0)
        # The published metric is a three-year median. Missing or unusable
        # recent years must not pull older, cheaper capex into that window.
        if year < latest_year - 2:
            continue
        capex = row.get("capex", row.get("capex_total", UNKNOWN))
        cfo = row.get("cfo", UNKNOWN)
        if year <= 0 or not _is_num(capex) or not _is_num(cfo):
            continue
        if float(cfo) <= 0.0:
            continue
        dated.append((year, abs(float(capex)) / float(cfo)))
    # The output is labelled a three-year median: the last three years with a
    # formable ratio, not every row on file.
    dated = sorted(dated, key=lambda item: item[0])[-_BURDEN_WINDOW_YEARS:]
    refs = [ref for year, _ in dated for ref in (f"fundamentals.rows[{year}].capex", f"fundamentals.rows[{year}].cfo")]
    if not dated:
        return UNKNOWN, _dedupe(refs)
    ratios = sorted(ratio for _, ratio in dated)
    mid = len(ratios) // 2
    median = ratios[mid] if len(ratios) % 2 == 1 else (ratios[mid - 1] + ratios[mid]) / 2.0
    return float(median), _dedupe(refs)


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


def _summary_for_class(
    *,
    cls: str,
    support_signals: list[str],
    headwind_signals: list[str],
) -> str:
    if cls == HIGH_REINVESTMENT_EFFICIENCY:
        return "reinvestment appears productive and owner-relevant across growth, burden, and capture signals"
    if cls == MODERATE_REINVESTMENT_EFFICIENCY:
        return "reinvestment evidence is mixed: some productive signals, but owner-outcome support is partial"
    if cls == LOW_REINVESTMENT_EFFICIENCY:
        return "reinvestment appears capital-consuming relative to owner outcomes and per-share capture"
    if headwind_signals:
        return "reinvestment quality is unclear with visible headwinds and thin evidence"
    if support_signals:
        return "reinvestment appears partially constructive, but evidence remains too thin for confidence"
    return "evidence too thin to assess reinvestment productivity responsibly"


def compute_reinvestment_efficiency(
    ticker: str,
    as_of_date: str,
    *,
    fundamentals: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    maintenance_capex_payload: dict[str, Any] | None = None,
    capital_allocation_discipline_payload: dict[str, Any] | None = None,
    evidence_sufficiency_payload: dict[str, Any] | None = None,
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
    intangible_payload = intangible_payload if isinstance(intangible_payload, dict) else {}
    maintenance_capex_payload = (
        maintenance_capex_payload if isinstance(maintenance_capex_payload, dict) else {}
    )
    cap_alloc_payload = (
        capital_allocation_discipline_payload
        if isinstance(capital_allocation_discipline_payload, dict)
        else {}
    )
    evidence_sufficiency_payload = (
        evidence_sufficiency_payload
        if isinstance(evidence_sufficiency_payload, dict)
        else {}
    )

    rows = _series_rows(fundamentals)
    revenue_cagr, revenue_cagr_refs = _series_cagr(rows, "revenue")
    cfo_cagr, cfo_cagr_refs = _series_cagr(rows, "cfo")
    fcf_cagr, fcf_cagr_refs = _series_cagr(rows, "fcf")
    shares_cagr, shares_cagr_refs = _series_cagr(rows, "shares_outstanding")
    capex_burden, capex_burden_refs = _capex_burden(rows)

    owner_value_capture_score = intangible_payload.get("owner_value_capture_score", UNKNOWN)
    owner_value_capture_reasons = {
        str(code)
        for code in (intangible_payload.get("owner_value_capture_reason_codes") or [])
        if str(code).strip()
    }
    gross_margin_durability_score = intangible_payload.get("gross_margin_durability_score", UNKNOWN)
    rnd_productivity_score = intangible_payload.get("rnd_productivity_score", UNKNOWN)
    sga_leverage_score = intangible_payload.get("sga_leverage_score", UNKNOWN)
    oe_quality_total = owner_quality_payload.get("oe_quality_total", UNKNOWN)
    owner_stability_score = owner_quality_payload.get("owner_earnings_stability_score", UNKNOWN)
    cash_conversion_score = owner_quality_payload.get("cash_conversion_score", UNKNOWN)
    maintenance_capex_class = str(
        maintenance_capex_payload.get("maintenance_capex_credibility_class")
        or MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
    ).upper()
    asset_intensity_class = str(
        maintenance_capex_payload.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
    ).upper()

    cap_alloc_class = str(
        cap_alloc_payload.get("capital_allocation_discipline_class") or "CAPITAL_ALLOCATION_UNKNOWN"
    ).upper()
    cap_alloc_headwinds = {
        str(code)
        for code in (cap_alloc_payload.get("per_share_headwind_signals") or [])
        if str(code).strip()
    }
    cap_alloc_supports = {
        str(code)
        for code in (cap_alloc_payload.get("per_share_support_signals") or [])
        if str(code).strip()
    }

    evidence_class = str(
        evidence_sufficiency_payload.get("evidence_sufficiency_class") or "SUFFICIENCY_UNKNOWN"
    ).upper()
    facts_ok = _coalesce_status(facts_status) == OK and evidence_class != "INSUFFICIENT_FOR_MOS"
    price_ok = _coalesce_status(price_status) == OK
    shares_ok = _coalesce_status(shares_status) == OK
    fcf_ok = _coalesce_status(fcf_status) == OK

    support_signals: list[str] = []
    headwind_signals: list[str] = []
    reason_codes: list[str] = []

    growth_known_count = 0
    if _is_num(revenue_cagr):
        growth_known_count += 1
    if _is_num(cfo_cagr):
        growth_known_count += 1
    if _is_num(fcf_cagr):
        growth_known_count += 1

    if _is_num(fcf_cagr) and float(fcf_cagr) > 0.03:
        support_signals.append(SIG_FCF_GROWTH_PRESENT)
    if _is_num(cfo_cagr) and float(cfo_cagr) > 0.03:
        support_signals.append(SIG_OWNER_EARNINGS_GROWTH_PRESENT)
    if _is_num(revenue_cagr) and float(revenue_cagr) > 0.03 and (
        (_is_num(gross_margin_durability_score) and float(gross_margin_durability_score) >= 3.0)
        or (_is_num(cash_conversion_score) and float(cash_conversion_score) >= 2.0)
    ):
        support_signals.append(SIG_REVENUE_GROWTH_WITH_MARGIN_STABILITY)
    if _is_num(capex_burden) and float(capex_burden) <= 0.35:
        support_signals.extend([SIG_CAPEX_LIGHT_SCALING_PRESENT, SIG_LOW_CAPITAL_BURDEN_RELATIVE_TO_GROWTH])
    if _is_num(owner_value_capture_score) and float(owner_value_capture_score) >= 3.0:
        support_signals.append(SIG_PER_SHARE_PROGRESS_PRESENT)
    if _is_num(gross_margin_durability_score) and float(gross_margin_durability_score) >= 4.0:
        support_signals.append(SIG_GROSS_MARGIN_DURABILITY_SUPPORTS_REINVESTMENT)

    per_share_progress_absent = False
    if _is_num(owner_value_capture_score) and float(owner_value_capture_score) <= 1.0:
        per_share_progress_absent = True
    if "WEAK_PER_SHARE_CAPTURE" in owner_value_capture_reasons:
        per_share_progress_absent = True
    if _is_num(shares_cagr) and float(shares_cagr) > 0.02 and (
        (_is_num(fcf_cagr) and float(fcf_cagr) <= float(shares_cagr))
        or not _is_num(fcf_cagr)
    ):
        per_share_progress_absent = True

    if _is_num(revenue_cagr) and float(revenue_cagr) > 0.03:
        if (not _is_num(cfo_cagr) or float(cfo_cagr) <= 0.0) and (not _is_num(fcf_cagr) or float(fcf_cagr) <= 0.0):
            headwind_signals.append(SIG_REVENUE_GROWTH_WITHOUT_OWNER_OUTCOME)
    if _is_num(capex_burden) and float(capex_burden) >= 0.60:
        headwind_signals.append(SIG_HIGH_CAPEX_BURDEN)
    if _is_num(cfo_cagr) and float(cfo_cagr) <= 0.0 and _is_num(revenue_cagr) and float(revenue_cagr) > 0.03:
        headwind_signals.append(SIG_CFO_GROWTH_WEAK_RELATIVE_TO_REINVESTMENT)
    if _is_num(fcf_cagr) and float(fcf_cagr) <= 0.0 and _is_num(revenue_cagr) and float(revenue_cagr) > 0.03:
        headwind_signals.append(SIG_FCF_GROWTH_WEAK_RELATIVE_TO_REINVESTMENT)
    if per_share_progress_absent:
        headwind_signals.append(SIG_PER_SHARE_PROGRESS_ABSENT)
    if _is_num(capex_burden) and float(capex_burden) >= 0.80:
        headwind_signals.append(SIG_CAPITAL_HUNGRY_GROWTH)
    if asset_intensity_class == LOW_ASSET_INTENSITY:
        support_signals.append(SIG_CAPEX_LIGHT_SCALING_PRESENT)
    if asset_intensity_class == HIGH_ASSET_INTENSITY:
        headwind_signals.extend([SIG_HIGH_CAPEX_BURDEN, SIG_CAPITAL_HUNGRY_GROWTH])
    if maintenance_capex_class == LOW_MAINTENANCE_CAPEX_CREDIBILITY:
        headwind_signals.append(SIG_HIGH_CAPEX_BURDEN)

    if not facts_ok or not shares_ok or (not price_ok and not fcf_ok):
        reason_codes.extend([REASON_MISSING_REINVESTMENT_INPUTS, REASON_REINVESTMENT_EVIDENCE_THIN])
        cls = REINVESTMENT_EFFICIENCY_UNKNOWN
        caution = REINVESTMENT_UNCLEAR
    elif len(rows) < 2 or growth_known_count == 0:
        reason_codes.extend([REASON_INSUFFICIENT_REINVESTMENT_HISTORY, REASON_REINVESTMENT_EVIDENCE_THIN])
        cls = REINVESTMENT_EFFICIENCY_UNKNOWN
        caution = REINVESTMENT_UNCLEAR
    else:
        productive_strength = 0
        headwind_strength = 0

        if len(set(support_signals) & {SIG_FCF_GROWTH_PRESENT, SIG_OWNER_EARNINGS_GROWTH_PRESENT}) > 0:
            productive_strength += 1
        if SIG_REVENUE_GROWTH_WITH_MARGIN_STABILITY in support_signals:
            productive_strength += 1
        if SIG_LOW_CAPITAL_BURDEN_RELATIVE_TO_GROWTH in support_signals:
            productive_strength += 1
        if SIG_PER_SHARE_PROGRESS_PRESENT in support_signals:
            productive_strength += 1
        if _is_num(rnd_productivity_score) and float(rnd_productivity_score) >= 3.0:
            productive_strength += 1
        if _is_num(sga_leverage_score) and float(sga_leverage_score) >= 3.0:
            productive_strength += 1
        if _is_num(oe_quality_total) and float(oe_quality_total) >= 8.0:
            productive_strength += 1
        if cap_alloc_class == "OWNER_FRIENDLY_DISCIPLINED" or "OWNER_VALUE_CAPTURE_PRESENT" in cap_alloc_supports:
            productive_strength += 1

        if any(
            signal in headwind_signals
            for signal in {
                SIG_REVENUE_GROWTH_WITHOUT_OWNER_OUTCOME,
                SIG_HIGH_CAPEX_BURDEN,
                SIG_CFO_GROWTH_WEAK_RELATIVE_TO_REINVESTMENT,
                SIG_FCF_GROWTH_WEAK_RELATIVE_TO_REINVESTMENT,
                SIG_PER_SHARE_PROGRESS_ABSENT,
                SIG_CAPITAL_HUNGRY_GROWTH,
            }
        ):
            headwind_strength += 2
        if cap_alloc_class == "OWNER_DILUTIVE_OR_DESTRUCTIVE":
            headwind_strength += 2
        if "HIGH_DILUTION" in cap_alloc_headwinds:
            headwind_strength += 1
        if "DILUTION_DESTROYS_OWNER_VALUE" in cap_alloc_headwinds:
            headwind_strength += 1
        if _is_num(owner_stability_score) and float(owner_stability_score) < 2.0:
            headwind_strength += 1

        if productive_strength >= 4 and headwind_strength == 0:
            cls = HIGH_REINVESTMENT_EFFICIENCY
            caution = REINVESTMENT_APPEARS_PRODUCTIVE
            reason_codes.append(REASON_PRODUCTIVE_REINVESTMENT_SUPPORT)
        elif headwind_strength >= 3 and productive_strength <= 2:
            cls = LOW_REINVESTMENT_EFFICIENCY
            caution = REINVESTMENT_HEADWIND
            reason_codes.extend([REASON_CAPITAL_HUNGRY_GROWTH_HEADWIND, REASON_GROWTH_WITHOUT_OWNER_OUTCOME])
        else:
            cls = MODERATE_REINVESTMENT_EFFICIENCY
            caution = REINVESTMENT_MIXED
            reason_codes.append(REASON_REINVESTMENT_MIXED)

    if cls == REINVESTMENT_EFFICIENCY_UNKNOWN:
        headwind_signals.append(SIG_REINVESTMENT_PRODUCTIVITY_UNCLEAR)

    support_signals = _dedupe(support_signals)
    headwind_signals = _dedupe(headwind_signals)
    reason_codes = _dedupe(
        reason_codes
        + (
            [REASON_GROWTH_WITHOUT_OWNER_OUTCOME]
            if SIG_REVENUE_GROWTH_WITHOUT_OWNER_OUTCOME in headwind_signals
            else []
        )
        + (
            [REASON_CAPITAL_HUNGRY_GROWTH_HEADWIND]
            if SIG_CAPITAL_HUNGRY_GROWTH in headwind_signals or SIG_HIGH_CAPEX_BURDEN in headwind_signals
            else []
        )
        + ([REASON_LOW_ASSET_INTENSITY_SUPPORT] if asset_intensity_class == LOW_ASSET_INTENSITY else [])
        + ([REASON_HIGH_ASSET_INTENSITY_HEADWIND] if asset_intensity_class == HIGH_ASSET_INTENSITY else [])
        + (
            [REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND]
            if maintenance_capex_class == LOW_MAINTENANCE_CAPEX_CREDIBILITY
            else []
        )
        + (
            [REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN]
            if maintenance_capex_class == MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
            else []
        )
        + ([REASON_REINVESTMENT_EVIDENCE_THIN] if cls == REINVESTMENT_EFFICIENCY_UNKNOWN else [])
    )

    derived_from = _collect_derived_from(
        fundamentals,
        owner_quality_payload,
        intangible_payload,
        maintenance_capex_payload,
        cap_alloc_payload,
        evidence_sufficiency_payload,
        row_refs=row_derived_from,
    )
    derived_from = _dedupe(
        derived_from
        + revenue_cagr_refs
        + cfo_cagr_refs
        + fcf_cagr_refs
        + shares_cagr_refs
        + capex_burden_refs
    )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "revenue_cagr_5y_proxy": _to_num(revenue_cagr),
        "cfo_cagr_5y_proxy": _to_num(cfo_cagr),
        "fcf_cagr_5y_proxy": _to_num(fcf_cagr),
        "shares_cagr_5y_proxy": _to_num(shares_cagr),
        "capex_burden_vs_cfo_median_3y": _to_num(capex_burden),
        "reinvestment_efficiency_class": cls,
        "reinvestment_efficiency_reason_codes": reason_codes,
        "reinvestment_support_signals": support_signals,
        "reinvestment_headwind_signals": headwind_signals,
        "primary_reinvestment_caution": caution,
        "reinvestment_efficiency_summary": _summary_for_class(
            cls=cls,
            support_signals=support_signals,
            headwind_signals=headwind_signals,
        ),
        "derived_from": derived_from,
        "claims": {
            "reinvestment_efficiency_class": {
                "value": cls,
                "status": OK if cls != REINVESTMENT_EFFICIENCY_UNKNOWN else UNKNOWN,
                "reason_code": reason_codes[0] if reason_codes else UNKNOWN,
                "derived_from": derived_from,
            },
            "primary_reinvestment_caution": {
                "value": caution,
                "status": OK if caution != REINVESTMENT_UNCLEAR else UNKNOWN,
                "reason_code": reason_codes[0] if reason_codes else UNKNOWN,
                "derived_from": derived_from,
            },
        },
        "generated_at": utc_now_iso(),
    }


def write_reinvestment_efficiency_for_run(
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
        str(row.get("ticker") or "").upper(): row.get("reinvestment_efficiency_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("reinvestment_efficiency_detail"), dict)
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
            detail = compute_reinvestment_efficiency(
                ticker=ticker,
                as_of_date=as_of_date,
                fundamentals=_fundamentals_for_score_row(score_row),
                owner_quality_payload=score_row.get("owner_earnings_quality_detail")
                if isinstance(score_row.get("owner_earnings_quality_detail"), dict)
                else {},
                intangible_payload=score_row.get("intangible_economics_detail")
                if isinstance(score_row.get("intangible_economics_detail"), dict)
                else {},
                maintenance_capex_payload=score_row.get("maintenance_capex_discipline_detail")
                if isinstance(score_row.get("maintenance_capex_discipline_detail"), dict)
                else {},
                capital_allocation_discipline_payload=score_row.get("capital_allocation_discipline_detail")
                if isinstance(score_row.get("capital_allocation_discipline_detail"), dict)
                else {},
                evidence_sufficiency_payload=score_row.get("evidence_sufficiency_detail")
                if isinstance(score_row.get("evidence_sufficiency_detail"), dict)
                else {},
                price_status=score_row.get("price_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                fcf_status=score_row.get("fcf_status", UNKNOWN),
                row_derived_from=list(score_row.get("derived_from") or []),
            )

        cls = str(detail.get("reinvestment_efficiency_class") or REINVESTMENT_EFFICIENCY_UNKNOWN)
        caution = str(detail.get("primary_reinvestment_caution") or REINVESTMENT_UNCLEAR)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        caution_counts[caution] = caution_counts.get(caution, 0) + 1
        if cls == REINVESTMENT_EFFICIENCY_UNKNOWN:
            unknown_count += 1
        else:
            known_count += 1

        for code in (detail.get("reinvestment_efficiency_reason_codes") or []):
            token = str(code or "").strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1

        rows.append(
            {
                "ticker": ticker,
                "reinvestment_efficiency_class": cls,
                "reinvestment_efficiency_reason_codes": [
                    str(code)
                    for code in (detail.get("reinvestment_efficiency_reason_codes") or [])
                    if str(code).strip()
                ],
                "reinvestment_support_signals": [
                    str(signal)
                    for signal in (detail.get("reinvestment_support_signals") or [])
                    if str(signal).strip()
                ],
                "reinvestment_headwind_signals": [
                    str(signal)
                    for signal in (detail.get("reinvestment_headwind_signals") or [])
                    if str(signal).strip()
                ],
                "primary_reinvestment_caution": caution,
                "reinvestment_efficiency_summary": str(detail.get("reinvestment_efficiency_summary") or ""),
                "derived_from": [str(ref) for ref in (detail.get("derived_from") or []) if str(ref).strip()],
            }
        )

    def _class_rows(value: str) -> list[dict[str, Any]]:
        subset = [
            row
            for row in rows
            if str(row.get("reinvestment_efficiency_class") or REINVESTMENT_EFFICIENCY_UNKNOWN) == value
        ]
        subset.sort(key=lambda row: str(row.get("ticker") or ""))
        return subset

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "known_count": int(known_count),
        "unknown_count": int(unknown_count),
        "counts_by_reinvestment_efficiency_class": dict(
            sorted(class_counts.items(), key=lambda item: (_CLASS_ORDER.get(item[0], 99), item[0]))
        ),
        "counts_by_primary_reinvestment_caution": dict(
            sorted(caution_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ),
        "top_10_high_reinvestment_efficiency": [
            {
                "ticker": str(row.get("ticker") or ""),
                "reinvestment_efficiency_reason_codes": [
                    str(code)
                    for code in (row.get("reinvestment_efficiency_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows(HIGH_REINVESTMENT_EFFICIENCY)[:10]
        ],
        "top_10_low_reinvestment_efficiency": [
            {
                "ticker": str(row.get("ticker") or ""),
                "reinvestment_efficiency_reason_codes": [
                    str(code)
                    for code in (row.get("reinvestment_efficiency_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows(LOW_REINVESTMENT_EFFICIENCY)[:10]
        ],
        "most_common_reinvestment_reason_codes": [
            {"reason_code": str(name), "count": int(count)}
            for name, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ][:20],
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["reinvestment_efficiency_path"] = str(output_path)
    return payload


def _reinvestment_efficiency_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "reinvestment_efficiency.json",
        cfg.sectors_dir / run_id / "reinvestment_efficiency.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_reinvestment_efficiency(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _reinvestment_efficiency_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "reinvestment_efficiency_path": str(path) if path is not None else "",
        }

    payload = _safe_json(path)
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    high_rows = [
        row for row in rows if str(row.get("reinvestment_efficiency_class") or "") == HIGH_REINVESTMENT_EFFICIENCY
    ]
    low_rows = [
        row for row in rows if str(row.get("reinvestment_efficiency_class") or "") == LOW_REINVESTMENT_EFFICIENCY
    ]
    caution_counts = payload.get("counts_by_primary_reinvestment_caution")
    reason_counts = payload.get("most_common_reinvestment_reason_codes")

    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "known_count": int(payload.get("known_count") or 0),
        "unknown_count": int(payload.get("unknown_count") or 0),
        "counts_by_reinvestment_efficiency_class": (
            payload.get("counts_by_reinvestment_efficiency_class")
            if isinstance(payload.get("counts_by_reinvestment_efficiency_class"), dict)
            else {}
        ),
        "counts_by_primary_reinvestment_caution": caution_counts if isinstance(caution_counts, dict) else {},
        "top_10_high_reinvestment_efficiency": [
            {
                "ticker": str(row.get("ticker") or ""),
                "reinvestment_efficiency_reason_codes": [
                    str(code)
                    for code in (row.get("reinvestment_efficiency_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in high_rows[: max(1, int(top_n))]
        ],
        "top_10_low_reinvestment_efficiency": [
            {
                "ticker": str(row.get("ticker") or ""),
                "reinvestment_efficiency_reason_codes": [
                    str(code)
                    for code in (row.get("reinvestment_efficiency_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in low_rows[: max(1, int(top_n))]
        ],
        "most_common_reinvestment_reason_codes": [
            {"reason_code": str(item.get("reason_code") or ""), "count": int(item.get("count") or 0)}
            for item in (reason_counts if isinstance(reason_counts, list) else [])
            if isinstance(item, dict)
        ][: max(1, int(top_n))],
        "reinvestment_efficiency_path": str(path),
    }
