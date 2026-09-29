from __future__ import annotations

import json
from pathlib import Path
from statistics import pstdev
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"
OK = "OK"

HIGH_RETURNS_PERSISTENCE = "HIGH_RETURNS_PERSISTENCE"
MODERATE_RETURNS_PERSISTENCE = "MODERATE_RETURNS_PERSISTENCE"
LOW_RETURNS_PERSISTENCE = "LOW_RETURNS_PERSISTENCE"
RETURNS_PERSISTENCE_UNKNOWN = "RETURNS_PERSISTENCE_UNKNOWN"

RETURNS_DURABILITY_SUPPORTIVE = "RETURNS_DURABILITY_SUPPORTIVE"
RETURNS_DURABILITY_MIXED = "RETURNS_DURABILITY_MIXED"
RETURNS_DURABILITY_HEADWIND = "RETURNS_DURABILITY_HEADWIND"
RETURNS_DURABILITY_UNCLEAR = "RETURNS_DURABILITY_UNCLEAR"

SIG_HIGH_RETURN_ON_CAPITAL_PRESENT = "HIGH_RETURN_ON_CAPITAL_PRESENT"
SIG_RETURNS_STABILITY_PRESENT = "RETURNS_STABILITY_PRESENT"
SIG_INCREMENTAL_RETURNS_ACCEPTABLE = "INCREMENTAL_RETURNS_ACCEPTABLE"
SIG_GROSS_MARGIN_DURABILITY_SUPPORTS_RETURNS = "GROSS_MARGIN_DURABILITY_SUPPORTS_RETURNS"
SIG_REINVESTMENT_EFFICIENCY_SUPPORTS_RETURNS = "REINVESTMENT_EFFICIENCY_SUPPORTS_RETURNS"
SIG_PER_SHARE_VALUE_CAPTURE_SUPPORTS_RETURNS = "PER_SHARE_VALUE_CAPTURE_SUPPORTS_RETURNS"

SIG_LOW_RETURN_ON_CAPITAL = "LOW_RETURN_ON_CAPITAL"
SIG_RETURNS_VOLATILITY_HEADWIND = "RETURNS_VOLATILITY_HEADWIND"
SIG_INCREMENTAL_RETURNS_DETERIORATING = "INCREMENTAL_RETURNS_DETERIORATING"
SIG_CAPITAL_INTENSITY_ERODES_RETURNS = "CAPITAL_INTENSITY_ERODES_RETURNS"
SIG_RETURNS_DEPEND_ON_FAVORABLE_CYCLE = "RETURNS_DEPEND_ON_FAVORABLE_CYCLE"
SIG_RETURNS_EVIDENCE_THIN = "RETURNS_EVIDENCE_THIN"

REASON_MISSING_RETURNS_INPUTS = "MISSING_RETURNS_INPUTS"
REASON_RETURN_ON_CAPITAL_UNKNOWN = "RETURN_ON_CAPITAL_EQUITY_ASSETS_ALL_UNKNOWN"
REASON_RETURNS_EVIDENCE_THIN = "RETURNS_EVIDENCE_THIN"
REASON_HIGH_RETURNS_PERSISTENCE_SUPPORT = "HIGH_RETURNS_PERSISTENCE_SUPPORT"
REASON_LOW_RETURNS_PERSISTENCE_HEADWIND = "LOW_RETURNS_PERSISTENCE_HEADWIND"
REASON_INCREMENTAL_RETURNS_DETERIORATION = "INCREMENTAL_RETURNS_DETERIORATION"
REASON_RETURNS_DURABILITY_MIXED = "RETURNS_DURABILITY_MIXED"
REASON_RETURNS_DURABILITY_UNKNOWN = "RETURNS_DURABILITY_UNKNOWN"
REASON_RETURNS_SERIES_ALIGNMENT_UNKNOWN = "RETURNS_SERIES_ALIGNMENT_UNKNOWN"

_CLASS_ORDER = {
    HIGH_RETURNS_PERSISTENCE: 0,
    MODERATE_RETURNS_PERSISTENCE: 1,
    RETURNS_PERSISTENCE_UNKNOWN: 2,
    LOW_RETURNS_PERSISTENCE: 3,
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


def _coalesce_num(*values: Any) -> float | str:
    for value in values:
        if _is_num(value):
            return float(value)
    return UNKNOWN


def _series_rows(fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("year") or 0))
    return rows


def _trace_refs_from_row(row: dict[str, Any], key: str) -> list[str]:
    year = int(row.get("year") or 0)
    metric = str(key).strip()
    if year <= 0 or not metric:
        return []
    return [f"fundamentals.rows[{year}].{metric}"]


def _latest_metric(rows: list[dict[str, Any]], *keys: str) -> tuple[float | str, list[str]]:
    for row in sorted(rows, key=lambda item: int(item.get("year") or 0), reverse=True):
        for key in keys:
            value = row.get(key, UNKNOWN)
            if _is_num(value):
                return float(value), _trace_refs_from_row(row, key)
    return UNKNOWN, []


def _series_values(rows: list[dict[str, Any]], *keys: str, window: int = 5) -> tuple[list[float], list[str]]:
    collected: list[tuple[int, float, list[str]]] = []
    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        for key in keys:
            value = row.get(key, UNKNOWN)
            if _is_num(value):
                collected.append((year, float(value), _trace_refs_from_row(row, key)))
                break
    collected = sorted(collected, key=lambda item: item[0])[-max(1, int(window)) :]
    return [value for _year, value, _refs in collected], _dedupe(
        [ref for _year, _value, refs in collected for ref in refs]
    )


def _series_cagr(rows: list[dict[str, Any]], *keys: str) -> tuple[float | str, list[str]]:
    points: list[tuple[int, float, list[str]]] = []
    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        for key in keys:
            value = row.get(key, UNKNOWN)
            if _is_num(value):
                points.append((year, float(value), _trace_refs_from_row(row, key)))
                break
    if len(points) < 2:
        return UNKNOWN, _dedupe([ref for _year, _value, refs in points for ref in refs])
    start_year, start_value, start_refs = points[0]
    end_year, end_value, end_refs = points[-1]
    periods = int(end_year - start_year)
    if periods <= 0 or start_value <= 0.0 or end_value <= 0.0:
        return UNKNOWN, _dedupe(start_refs + end_refs)
    return (
        (end_value / start_value) ** (1.0 / float(periods)) - 1.0,
        _dedupe(start_refs + end_refs),
    )


def _claim(*, value: Any, refs: list[Any], reason_code: str) -> dict[str, Any]:
    derived = _dedupe(refs)
    if isinstance(value, str):
        token = str(value).strip()
        if token.upper() in {RETURNS_PERSISTENCE_UNKNOWN, RETURNS_DURABILITY_UNCLEAR}:
            return {
                "value": token, "status": UNKNOWN,
                "reason_code": str(reason_code or UNKNOWN), "derived_from": derived,
            }
        if token and token.upper() != UNKNOWN:
            return {"value": token, "status": OK, "reason_code": OK, "derived_from": derived}
    if _is_num(value):
        return {"value": float(value), "status": OK, "reason_code": OK, "derived_from": derived}
    return {
        "value": UNKNOWN,
        "status": UNKNOWN,
        "reason_code": str(reason_code or UNKNOWN),
        "derived_from": derived,
    }


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


def _summary_for_class(*, cls: str, headwinds: list[str], supports: list[str]) -> str:
    if cls == HIGH_RETURNS_PERSISTENCE:
        return "business economics appear durable; returns look meaningfully above mediocre levels and reasonably repeatable"
    if cls == MODERATE_RETURNS_PERSISTENCE:
        return "returns durability looks decent but mixed; economics are acceptable without being clearly persistent"
    if cls == LOW_RETURNS_PERSISTENCE:
        return "returns look too weak, unstable, or deteriorating to treat current economics as durable"
    if headwinds:
        return "durability is unclear with visible returns headwinds and limited support"
    if supports:
        return "some durability support exists, but evidence is too thin to judge repeatable returns honestly"
    return "evidence too thin to judge whether returns on capital are durable or merely temporary"


def compute_returns_persistence(
    ticker: str,
    as_of_date: str,
    *,
    fundamentals: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    reinvestment_efficiency_payload: dict[str, Any] | None = None,
    capital_allocation_discipline_payload: dict[str, Any] | None = None,
    revenue_dependence_payload: dict[str, Any] | None = None,
    roic_proxy: Any = UNKNOWN,
    roe_proxy: Any = UNKNOWN,
    roa_proxy: Any = UNKNOWN,
    return_on_retained_earnings: Any = UNKNOWN,
    revenue_cagr_proxy: Any = UNKNOWN,
    invested_capital_cagr_proxy: Any = UNKNOWN,
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
    reinvestment_efficiency_payload = (
        reinvestment_efficiency_payload if isinstance(reinvestment_efficiency_payload, dict) else {}
    )
    capital_allocation_discipline_payload = (
        capital_allocation_discipline_payload
        if isinstance(capital_allocation_discipline_payload, dict)
        else {}
    )
    revenue_dependence_payload = (
        revenue_dependence_payload if isinstance(revenue_dependence_payload, dict) else {}
    )
    rows = _series_rows(fundamentals)

    roic_series, roic_series_refs = _series_values(
        rows,
        "roic_proxy",
        "roic",
        "return_on_invested_capital",
    )
    roe_series, roe_series_refs = _series_values(rows, "roe_proxy", "roe", "return_on_equity")
    roa_series, roa_series_refs = _series_values(rows, "roa_proxy", "roa", "return_on_assets")

    latest_roic, latest_roic_refs = _latest_metric(rows, "roic_proxy", "roic", "return_on_invested_capital")
    latest_roe, latest_roe_refs = _latest_metric(rows, "roe_proxy", "roe", "return_on_equity")
    latest_roa, latest_roa_refs = _latest_metric(rows, "roa_proxy", "roa", "return_on_assets")

    roic_value = _coalesce_num(roic_proxy, latest_roic)
    roe_value = _coalesce_num(roe_proxy, latest_roe)
    roa_value = _coalesce_num(roa_proxy, latest_roa)
    retained_returns = _coalesce_num(return_on_retained_earnings)
    revenue_cagr = _coalesce_num(revenue_cagr_proxy, _series_cagr(rows, "revenue")[0])
    invested_capital_cagr = _coalesce_num(
        invested_capital_cagr_proxy,
        _series_cagr(rows, "invested_capital", "capital_employed", "net_operating_assets")[0],
    )

    if _is_num(roic_proxy):
        if not roic_series:
            roic_series = [float(roic_proxy)]
            roic_series_refs = _dedupe(roic_series_refs + list(row_derived_from or []))
        latest_roic_refs = _dedupe(latest_roic_refs + list(row_derived_from or []))
    if _is_num(roe_proxy):
        if not roe_series:
            roe_series = [float(roe_proxy)]
            roe_series_refs = _dedupe(roe_series_refs + list(row_derived_from or []))
        latest_roe_refs = _dedupe(latest_roe_refs + list(row_derived_from or []))
    if _is_num(roa_proxy):
        if not roa_series:
            roa_series = [float(roa_proxy)]
            roa_series_refs = _dedupe(roa_series_refs + list(row_derived_from or []))
        latest_roa_refs = _dedupe(latest_roa_refs + list(row_derived_from or []))

    chosen_series = roic_series or roe_series or roa_series
    chosen_series_refs = roic_series_refs or roe_series_refs or roa_series_refs
    chosen_proxy = roic_proxy if roic_series else (roe_proxy if roe_series else roa_proxy)
    # The scalar has no fiscal date. A contradiction cannot be assigned
    # to the latest annual row or treated as an additional annual year.
    # Keep the current level, but withhold stale stability support.
    series_alignment_unknown = bool(
        chosen_series and _is_num(chosen_proxy)
        and float(chosen_proxy) != chosen_series[-1]
    )
    latest_return_proxy = _coalesce_num(roic_value, roe_value, roa_value)

    support_signals: list[str] = []
    headwind_signals: list[str] = []
    reason_codes: list[str] = []
    if series_alignment_unknown:
        reason_codes.append(REASON_RETURNS_SERIES_ALIGNMENT_UNKNOWN)

    gross_margin_durability_score = intangible_payload.get("gross_margin_durability_score", UNKNOWN)
    cycle_resilience_score = intangible_payload.get("cycle_resilience_score", UNKNOWN)
    balance_sheet_optionality_score = intangible_payload.get("balance_sheet_optionality_score", UNKNOWN)
    owner_value_capture_score = intangible_payload.get("owner_value_capture_score", UNKNOWN)
    owner_value_capture_reasons = {
        str(code)
        for code in (intangible_payload.get("owner_value_capture_reason_codes") or [])
        if str(code).strip()
    }

    reinvestment_class = str(
        reinvestment_efficiency_payload.get("reinvestment_efficiency_class")
        or "REINVESTMENT_EFFICIENCY_UNKNOWN"
    ).upper()
    reinvestment_reasons = {
        str(code)
        for code in (reinvestment_efficiency_payload.get("reinvestment_efficiency_reason_codes") or [])
        if str(code).strip()
    }

    cap_alloc_class = str(
        capital_allocation_discipline_payload.get("capital_allocation_discipline_class")
        or "CAPITAL_ALLOCATION_UNKNOWN"
    ).upper()
    revenue_dependence_class = str(
        revenue_dependence_payload.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
    ).upper()
    revenue_dependence_headwinds = {
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_headwind_signals") or [])
        if str(code).strip()
    }

    if (
        (_is_num(roic_value) and float(roic_value) >= 0.15)
        or (_is_num(roe_value) and float(roe_value) >= 0.16)
        or (_is_num(roa_value) and float(roa_value) >= 0.08)
    ):
        support_signals.append(SIG_HIGH_RETURN_ON_CAPITAL_PRESENT)
    elif (
        (_is_num(roic_value) and float(roic_value) < 0.08)
        or (_is_num(roe_value) and float(roe_value) < 0.10)
        or (_is_num(roa_value) and float(roa_value) < 0.04)
    ):
        headwind_signals.append(SIG_LOW_RETURN_ON_CAPITAL)

    if len(chosen_series) >= 3:
        returns_volatility = float(pstdev(chosen_series))
        if (
            not series_alignment_unknown
            and returns_volatility <= 0.03 and min(chosen_series) >= 0.10
        ):
            support_signals.append(SIG_RETURNS_STABILITY_PRESENT)
        elif returns_volatility >= 0.07 or (max(chosen_series) - min(chosen_series)) >= 0.12:
            headwind_signals.append(SIG_RETURNS_VOLATILITY_HEADWIND)

    capital_spread = UNKNOWN
    if _is_num(revenue_cagr) and _is_num(invested_capital_cagr):
        capital_spread = float(revenue_cagr) - float(invested_capital_cagr)
        if capital_spread >= -0.02:
            support_signals.append(SIG_INCREMENTAL_RETURNS_ACCEPTABLE)
        elif capital_spread < -0.05:
            headwind_signals.append(SIG_INCREMENTAL_RETURNS_DETERIORATING)

    if _is_num(retained_returns) and float(retained_returns) >= 0.12:
        support_signals.append(SIG_INCREMENTAL_RETURNS_ACCEPTABLE)
    elif _is_num(retained_returns) and float(retained_returns) < 0.05:
        headwind_signals.append(SIG_INCREMENTAL_RETURNS_DETERIORATING)

    if _is_num(gross_margin_durability_score) and float(gross_margin_durability_score) >= 4.0:
        support_signals.append(SIG_GROSS_MARGIN_DURABILITY_SUPPORTS_RETURNS)
    if _is_num(cycle_resilience_score) and float(cycle_resilience_score) <= 2.0:
        headwind_signals.append(SIG_RETURNS_DEPEND_ON_FAVORABLE_CYCLE)
    if _is_num(balance_sheet_optionality_score) and float(balance_sheet_optionality_score) <= 2.0:
        headwind_signals.append(SIG_RETURNS_DEPEND_ON_FAVORABLE_CYCLE)

    if reinvestment_class == "HIGH_REINVESTMENT_EFFICIENCY":
        support_signals.append(SIG_REINVESTMENT_EFFICIENCY_SUPPORTS_RETURNS)
    elif reinvestment_class == "LOW_REINVESTMENT_EFFICIENCY":
        headwind_signals.extend(
            [SIG_INCREMENTAL_RETURNS_DETERIORATING, SIG_CAPITAL_INTENSITY_ERODES_RETURNS]
        )
    if {
        "CAPITAL_HUNGRY_GROWTH_HEADWIND",
        "GROWTH_WITHOUT_OWNER_OUTCOME",
    } & reinvestment_reasons:
        headwind_signals.extend(
            [SIG_INCREMENTAL_RETURNS_DETERIORATING, SIG_CAPITAL_INTENSITY_ERODES_RETURNS]
        )

    if (
        cap_alloc_class == "OWNER_FRIENDLY_DISCIPLINED"
        or (_is_num(owner_value_capture_score) and float(owner_value_capture_score) >= 3.0)
    ):
        support_signals.append(SIG_PER_SHARE_VALUE_CAPTURE_SUPPORTS_RETURNS)
    elif (
        cap_alloc_class == "OWNER_DILUTIVE_OR_DESTRUCTIVE"
        or "WEAK_PER_SHARE_CAPTURE" in owner_value_capture_reasons
    ):
        headwind_signals.append(SIG_CAPITAL_INTENSITY_ERODES_RETURNS)

    # A returns class is a claim about returns. Growth, capital-spread and intangible
    # sub-scores are supporting context, never a substitute for a measured level of
    # return: with return on capital, equity and assets all unknown the class is
    # UNKNOWN, whatever else is on file.
    return_level_known = any(_is_num(value) for value in [roic_value, roe_value, roa_value])
    evidence_known = return_level_known
    evidence_blocked = any(
        str(status or UNKNOWN).upper() != OK
        for status in [facts_status, price_status, shares_status]
    )

    if not evidence_known:
        headwind_signals.append(SIG_RETURNS_EVIDENCE_THIN)
        cls = RETURNS_PERSISTENCE_UNKNOWN
        caution = RETURNS_DURABILITY_UNCLEAR
        reason_codes.extend([REASON_MISSING_RETURNS_INPUTS, REASON_RETURNS_DURABILITY_UNKNOWN])
        reason_codes.append(REASON_RETURN_ON_CAPITAL_UNKNOWN)
    else:
        support_strength = 0
        headwind_strength = 0
        if SIG_HIGH_RETURN_ON_CAPITAL_PRESENT in support_signals:
            support_strength += 2
        if SIG_RETURNS_STABILITY_PRESENT in support_signals:
            support_strength += 2
        if SIG_INCREMENTAL_RETURNS_ACCEPTABLE in support_signals:
            support_strength += 1
        if SIG_GROSS_MARGIN_DURABILITY_SUPPORTS_RETURNS in support_signals:
            support_strength += 1
        if SIG_REINVESTMENT_EFFICIENCY_SUPPORTS_RETURNS in support_signals:
            support_strength += 1
        if SIG_PER_SHARE_VALUE_CAPTURE_SUPPORTS_RETURNS in support_signals:
            support_strength += 1

        if SIG_LOW_RETURN_ON_CAPITAL in headwind_signals:
            headwind_strength += 2
        if SIG_RETURNS_VOLATILITY_HEADWIND in headwind_signals:
            headwind_strength += 2
        if SIG_INCREMENTAL_RETURNS_DETERIORATING in headwind_signals:
            headwind_strength += 2
        if SIG_CAPITAL_INTENSITY_ERODES_RETURNS in headwind_signals:
            headwind_strength += 1
        if SIG_RETURNS_DEPEND_ON_FAVORABLE_CYCLE in headwind_signals:
            headwind_strength += 1

        if support_strength >= 4 and headwind_strength <= 1:
            cls = HIGH_RETURNS_PERSISTENCE
            caution = RETURNS_DURABILITY_SUPPORTIVE
            reason_codes.append(REASON_HIGH_RETURNS_PERSISTENCE_SUPPORT)
        elif headwind_strength >= 3 or (
            SIG_LOW_RETURN_ON_CAPITAL in headwind_signals
            and (
                SIG_RETURNS_VOLATILITY_HEADWIND in headwind_signals
                or SIG_INCREMENTAL_RETURNS_DETERIORATING in headwind_signals
            )
        ):
            cls = LOW_RETURNS_PERSISTENCE
            caution = RETURNS_DURABILITY_HEADWIND
            reason_codes.append(REASON_LOW_RETURNS_PERSISTENCE_HEADWIND)
        else:
            cls = MODERATE_RETURNS_PERSISTENCE
            caution = RETURNS_DURABILITY_MIXED
            reason_codes.append(REASON_RETURNS_DURABILITY_MIXED)

        if evidence_blocked and cls == MODERATE_RETURNS_PERSISTENCE:
            headwind_signals.append(SIG_RETURNS_EVIDENCE_THIN)

        if revenue_dependence_class == "HIGH_REVENUE_DEPENDENCE_RISK":
            if cls == HIGH_RETURNS_PERSISTENCE:
                cls = MODERATE_RETURNS_PERSISTENCE
                caution = RETURNS_DURABILITY_MIXED
                if REASON_HIGH_RETURNS_PERSISTENCE_SUPPORT in reason_codes:
                    reason_codes.remove(REASON_HIGH_RETURNS_PERSISTENCE_SUPPORT)
                reason_codes.append(REASON_RETURNS_DURABILITY_MIXED)
            reason_codes.append("HIGH_REVENUE_DEPENDENCE_HEADWIND")
            if {
                "SINGLE_CUSTOMER_CONCENTRATION",
                "TOP_CUSTOMER_DOMINANCE",
            } & revenue_dependence_headwinds:
                reason_codes.append("CUSTOMER_CONCENTRATION_HEADWIND")
            if "NARROW_CHANNEL_DEPENDENCE" in revenue_dependence_headwinds:
                reason_codes.append("CHANNEL_DEPENDENCE_HEADWIND")
        elif revenue_dependence_class == "REVENUE_DEPENDENCE_UNKNOWN":
            reason_codes.append("REVENUE_DEPENDENCE_UNKNOWN")

    if SIG_INCREMENTAL_RETURNS_DETERIORATING in headwind_signals:
        reason_codes.append(REASON_INCREMENTAL_RETURNS_DETERIORATION)
    if cls == RETURNS_PERSISTENCE_UNKNOWN:
        reason_codes.append(REASON_RETURNS_DURABILITY_UNKNOWN)
    if SIG_RETURNS_EVIDENCE_THIN in headwind_signals:
        reason_codes.append(REASON_RETURNS_EVIDENCE_THIN)

    support_signals = _dedupe(support_signals)
    headwind_signals = _dedupe(headwind_signals)
    reason_codes = _dedupe(reason_codes)
    derived_from = _collect_derived_from(
        owner_quality_payload,
        intangible_payload,
        reinvestment_efficiency_payload,
        capital_allocation_discipline_payload,
        revenue_dependence_payload,
        row_refs=(
            list(fundamentals.get("derived_from") or [])
            + chosen_series_refs
            + latest_roic_refs
            + latest_roe_refs
            + latest_roa_refs
            + list(row_derived_from or [])
        ),
    )

    payload = {
        "ticker": ticker_norm,
        "as_of_date": str(as_of_date or ""),
        "returns_persistence_class": cls,
        "returns_persistence_reason_codes": reason_codes,
        "returns_support_signals": support_signals,
        "returns_headwind_signals": headwind_signals,
        "primary_returns_caution": caution,
        "economic_durability_summary": _summary_for_class(
            cls=cls,
            headwinds=headwind_signals,
            supports=support_signals,
        ),
        "latest_roic_proxy": _to_num(roic_value),
        "latest_roe_proxy": _to_num(roe_value),
        "latest_roa_proxy": _to_num(roa_value),
        "return_on_retained_earnings": _to_num(retained_returns),
        "revenue_cagr_proxy": _to_num(revenue_cagr),
        "invested_capital_cagr_proxy": _to_num(invested_capital_cagr),
        "derived_from": derived_from,
        "claims": {
            "returns_persistence_class": _claim(
                value=cls,
                refs=derived_from,
                reason_code=REASON_RETURNS_DURABILITY_UNKNOWN,
            ),
        },
        "generated_at": utc_now_iso(),
    }
    return payload


def write_returns_persistence_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    rows_in = [row for row in (scoreboard_rows or []) if isinstance(row, dict)]
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("returns_persistence_detail")
        for row in rows_in
        if isinstance(row.get("returns_persistence_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    counts_by_class: dict[str, int] = {}
    counts_by_caution: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for ticker in sorted({str(token).strip().upper() for token in tickers if str(token).strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            score_row = next((row for row in rows_in if str(row.get("ticker") or "").upper() == ticker), {})
            detail = compute_returns_persistence(
                ticker=ticker,
                as_of_date=as_of_date,
                fundamentals=score_row.get("fundamentals_detail")
                if isinstance(score_row.get("fundamentals_detail"), dict)
                else {},
                owner_quality_payload=score_row.get("owner_earnings_quality_detail")
                if isinstance(score_row.get("owner_earnings_quality_detail"), dict)
                else {},
                intangible_payload=score_row.get("intangible_economics_detail")
                if isinstance(score_row.get("intangible_economics_detail"), dict)
                else {},
                reinvestment_efficiency_payload=score_row.get("reinvestment_efficiency_detail")
                if isinstance(score_row.get("reinvestment_efficiency_detail"), dict)
                else {},
                capital_allocation_discipline_payload=score_row.get("capital_allocation_discipline_detail")
                if isinstance(score_row.get("capital_allocation_discipline_detail"), dict)
                else {},
                revenue_dependence_payload=score_row.get("revenue_dependence_detail")
                if isinstance(score_row.get("revenue_dependence_detail"), dict)
                else {},
                roic_proxy=(score_row.get("metric_values") or {}).get("roic_proxy", UNKNOWN)
                if isinstance(score_row.get("metric_values"), dict)
                else UNKNOWN,
                row_derived_from=list(score_row.get("derived_from") or []),
            )
        cls = str(detail.get("returns_persistence_class") or RETURNS_PERSISTENCE_UNKNOWN)
        caution = str(detail.get("primary_returns_caution") or RETURNS_DURABILITY_UNCLEAR)
        counts_by_class[cls] = int(counts_by_class.get(cls) or 0) + 1
        counts_by_caution[caution] = int(counts_by_caution.get(caution) or 0) + 1
        for code in [str(code) for code in (detail.get("returns_persistence_reason_codes") or []) if str(code).strip()]:
            reason_counts[code] = int(reason_counts.get(code) or 0) + 1
        rows.append(
            {
                "ticker": ticker,
                "returns_persistence_class": cls,
                "returns_persistence_reason_codes": [
                    str(code)
                    for code in (detail.get("returns_persistence_reason_codes") or [])
                    if str(code).strip()
                ],
                "returns_support_signals": [
                    str(code) for code in (detail.get("returns_support_signals") or []) if str(code).strip()
                ],
                "returns_headwind_signals": [
                    str(code) for code in (detail.get("returns_headwind_signals") or []) if str(code).strip()
                ],
                "primary_returns_caution": caution,
                "economic_durability_summary": str(detail.get("economic_durability_summary") or ""),
                "derived_from": [str(ref) for ref in (detail.get("derived_from") or []) if str(ref).strip()],
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            _CLASS_ORDER.get(
                str(row.get("returns_persistence_class") or RETURNS_PERSISTENCE_UNKNOWN),
                len(_CLASS_ORDER),
            ),
            str(row.get("ticker") or ""),
        ),
    )

    def _class_rows(value: str) -> list[dict[str, Any]]:
        return [
            row
            for row in rows_sorted
            if str(row.get("returns_persistence_class") or RETURNS_PERSISTENCE_UNKNOWN) == value
        ]

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "generated_at": utc_now_iso(),
        "ticker_count": len(rows_sorted),
        "rows": rows_sorted,
        "counts_by_returns_persistence_class": dict(sorted(counts_by_class.items())),
        "counts_by_primary_returns_caution": dict(sorted(counts_by_caution.items())),
        "top_10_high_returns_persistence": [
            {
                "ticker": row.get("ticker", ""),
                "returns_persistence_reason_codes": [
                    str(code)
                    for code in (row.get("returns_persistence_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows(HIGH_RETURNS_PERSISTENCE)[:10]
        ],
        "top_10_low_returns_persistence": [
            {
                "ticker": row.get("ticker", ""),
                "returns_persistence_reason_codes": [
                    str(code)
                    for code in (row.get("returns_persistence_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows(LOW_RETURNS_PERSISTENCE)[:10]
        ],
        "most_common_returns_persistence_reason_codes": [
            {"reason_code": code, "count": count}
            for code, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:10]
        ],
    }
    payload["returns_persistence_path"] = str(output_path)
    _json_write(output_path, payload)
    return payload


def _returns_persistence_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "returns_persistence.json",
        cfg.sectors_dir / run_id / "returns_persistence.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def open_returns_persistence(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _returns_persistence_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "returns_persistence_path": str(path) if path is not None else "",
        }

    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_returns_persistence_class": (
            payload.get("counts_by_returns_persistence_class")
            if isinstance(payload.get("counts_by_returns_persistence_class"), dict)
            else {}
        ),
        "counts_by_primary_returns_caution": (
            payload.get("counts_by_primary_returns_caution")
            if isinstance(payload.get("counts_by_primary_returns_caution"), dict)
            else {}
        ),
        "top_10_high_returns_persistence": [
            row for row in (payload.get("top_10_high_returns_persistence") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_low_returns_persistence": [
            row for row in (payload.get("top_10_low_returns_persistence") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "most_common_returns_persistence_reason_codes": [
            row
            for row in (payload.get("most_common_returns_persistence_reason_codes") or [])
            if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "returns_persistence_path": str(path),
    }
