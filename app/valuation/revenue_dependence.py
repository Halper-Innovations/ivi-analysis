from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"

LOW_REVENUE_DEPENDENCE_RISK = "LOW_REVENUE_DEPENDENCE_RISK"
MODERATE_REVENUE_DEPENDENCE_RISK = "MODERATE_REVENUE_DEPENDENCE_RISK"
HIGH_REVENUE_DEPENDENCE_RISK = "HIGH_REVENUE_DEPENDENCE_RISK"
REVENUE_DEPENDENCE_UNKNOWN = "REVENUE_DEPENDENCE_UNKNOWN"

REVENUE_BASE_SUPPORTIVE = "REVENUE_BASE_SUPPORTIVE"
REVENUE_BASE_MIXED = "REVENUE_BASE_MIXED"
REVENUE_BASE_HEADWIND = "REVENUE_BASE_HEADWIND"
REVENUE_BASE_UNCLEAR = "REVENUE_BASE_UNCLEAR"

SIG_NO_MAJOR_CUSTOMER_CONCENTRATION_DISCLOSED = "NO_MAJOR_CUSTOMER_CONCENTRATION_DISCLOSED"
SIG_DIVERSIFIED_REVENUE_BASE = "DIVERSIFIED_REVENUE_BASE"
SIG_MULTIPLE_END_MARKETS_PRESENT = "MULTIPLE_END_MARKETS_PRESENT"
SIG_CHANNEL_DIVERSITY_PRESENT = "CHANNEL_DIVERSITY_PRESENT"
SIG_CONCENTRATION_NOT_DOMINANT = "CONCENTRATION_NOT_DOMINANT"

SIG_SINGLE_CUSTOMER_CONCENTRATION = "SINGLE_CUSTOMER_CONCENTRATION"
SIG_TOP_CUSTOMER_DOMINANCE = "TOP_CUSTOMER_DOMINANCE"
SIG_NARROW_CHANNEL_DEPENDENCE = "NARROW_CHANNEL_DEPENDENCE"
SIG_NARROW_END_MARKET_DEPENDENCE = "NARROW_END_MARKET_DEPENDENCE"
SIG_REVENUE_BASE_FRAGILITY = "REVENUE_BASE_FRAGILITY"
SIG_CONCENTRATION_DISCLOSURE_THIN = "CONCENTRATION_DISCLOSURE_THIN"

REASON_LOW_REVENUE_DEPENDENCE_SUPPORT = "LOW_REVENUE_DEPENDENCE_SUPPORT"
REASON_HIGH_REVENUE_DEPENDENCE_HEADWIND = "HIGH_REVENUE_DEPENDENCE_HEADWIND"
REASON_CUSTOMER_CONCENTRATION_HEADWIND = "CUSTOMER_CONCENTRATION_HEADWIND"
REASON_CHANNEL_DEPENDENCE_HEADWIND = "CHANNEL_DEPENDENCE_HEADWIND"
REASON_REVENUE_DEPENDENCE_UNKNOWN = "REVENUE_DEPENDENCE_UNKNOWN"
REASON_MISSING_REVENUE_DEPENDENCE_INPUTS = "MISSING_REVENUE_DEPENDENCE_INPUTS"
REASON_REVENUE_BASE_MIXED = "REVENUE_BASE_MIXED"

_CLASS_ORDER = {
    LOW_REVENUE_DEPENDENCE_RISK: 0,
    MODERATE_REVENUE_DEPENDENCE_RISK: 1,
    REVENUE_DEPENDENCE_UNKNOWN: 2,
    HIGH_REVENUE_DEPENDENCE_RISK: 3,
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


def _claim(*, value: Any, refs: list[Any], reason_code: str) -> dict[str, Any]:
    derived = _dedupe(refs)
    token = str(value or "").strip()
    if token and token.upper() != UNKNOWN:
        return {"value": token, "status": "OK", "reason_code": "OK", "derived_from": derived}
    return {
        "value": UNKNOWN,
        "status": UNKNOWN,
        "reason_code": str(reason_code or UNKNOWN),
        "derived_from": derived,
    }


def _summary_for_class(*, cls: str, headwinds: list[str], supports: list[str]) -> str:
    if cls == LOW_REVENUE_DEPENDENCE_RISK:
        return "revenue base appears reasonably diversified without a dominant customer, channel, or end market"
    if cls == MODERATE_REVENUE_DEPENDENCE_RISK:
        return "revenue dependence looks mixed; some concentration exists but does not clearly dominate the case"
    if cls == HIGH_REVENUE_DEPENDENCE_RISK:
        return "revenue base appears fragile because too much depends on a narrow set of customers, channels, or end markets"
    if headwinds:
        return "concentration risk is unclear, with some fragility signs but insufficient evidence for a clean judgment"
    if supports:
        return "some diversification support exists, but disclosure remains too thin to judge concentration honestly"
    return "evidence too thin to judge whether the revenue base is diversified or narrowly dependent"


def compute_revenue_dependence(
    ticker: str,
    as_of_date: str,
    *,
    fundamentals: dict[str, Any] | None = None,
    returns_persistence_payload: dict[str, Any] | None = None,
    cyclical_normalization_payload: dict[str, Any] | None = None,
    impairment_classification_payload: dict[str, Any] | None = None,
    normalization_credibility_payload: dict[str, Any] | None = None,
    customer_concentration_pct: Any = UNKNOWN,
    top_customer_pct: Any = UNKNOWN,
    top_channel_pct: Any = UNKNOWN,
    top_end_market_pct: Any = UNKNOWN,
    segment_count: Any = UNKNOWN,
    channel_count: Any = UNKNOWN,
    end_market_count: Any = UNKNOWN,
    customer_concentration_flag: Any = UNKNOWN,
    price_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    row_derived_from: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg

    ticker_norm = str(ticker or "").strip().upper()
    fundamentals = fundamentals if isinstance(fundamentals, dict) else {}
    returns_persistence_payload = (
        returns_persistence_payload if isinstance(returns_persistence_payload, dict) else {}
    )
    cyclical_normalization_payload = (
        cyclical_normalization_payload if isinstance(cyclical_normalization_payload, dict) else {}
    )
    impairment_classification_payload = (
        impairment_classification_payload if isinstance(impairment_classification_payload, dict) else {}
    )
    normalization_credibility_payload = (
        normalization_credibility_payload if isinstance(normalization_credibility_payload, dict) else {}
    )
    rows = _series_rows(fundamentals)

    latest_customer_pct, latest_customer_refs = _latest_metric(
        rows,
        "customer_concentration_pct",
        "top_customer_pct",
    )
    latest_channel_pct, latest_channel_refs = _latest_metric(rows, "top_channel_pct")
    latest_end_market_pct, latest_end_market_refs = _latest_metric(rows, "top_end_market_pct")
    latest_segment_count, latest_segment_refs = _latest_metric(rows, "segment_count")
    latest_channel_count, latest_channel_count_refs = _latest_metric(rows, "channel_count")
    latest_end_market_count, latest_end_market_count_refs = _latest_metric(rows, "end_market_count")

    customer_pct_value = _coalesce_num(customer_concentration_pct, top_customer_pct, latest_customer_pct)
    channel_pct_value = _coalesce_num(top_channel_pct, latest_channel_pct)
    end_market_pct_value = _coalesce_num(top_end_market_pct, latest_end_market_pct)
    segment_count_value = _coalesce_num(segment_count, latest_segment_count)
    channel_count_value = _coalesce_num(channel_count, latest_channel_count)
    end_market_count_value = _coalesce_num(end_market_count, latest_end_market_count)

    support_signals: list[str] = []
    headwind_signals: list[str] = []
    reason_codes: list[str] = []

    customer_flag = bool(customer_concentration_flag) if isinstance(customer_concentration_flag, bool) else False
    returns_class = str(
        returns_persistence_payload.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
    ).upper()
    cyclical_risk_class = str(
        cyclical_normalization_payload.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN"
    ).upper()
    impairment_class = str(
        impairment_classification_payload.get("impairment_class_primary") or "IMPAIRMENT_UNKNOWN"
    ).upper()
    normalization_class = str(
        normalization_credibility_payload.get("normalization_credibility_class")
        or "NORMALIZATION_CREDIBILITY_UNKNOWN"
    ).upper()

    evidence_known = any(
        _is_num(value)
        for value in [
            customer_pct_value,
            channel_pct_value,
            end_market_pct_value,
            segment_count_value,
            channel_count_value,
            end_market_count_value,
        ]
    ) or customer_flag

    if _is_num(customer_pct_value):
        if float(customer_pct_value) >= 0.25:
            headwind_signals.extend([SIG_SINGLE_CUSTOMER_CONCENTRATION, SIG_TOP_CUSTOMER_DOMINANCE, SIG_REVENUE_BASE_FRAGILITY])
        elif float(customer_pct_value) >= 0.10:
            headwind_signals.append(SIG_SINGLE_CUSTOMER_CONCENTRATION)
        else:
            support_signals.append(SIG_NO_MAJOR_CUSTOMER_CONCENTRATION_DISCLOSED)
    elif customer_flag:
        headwind_signals.append(SIG_SINGLE_CUSTOMER_CONCENTRATION)

    if _is_num(channel_pct_value):
        if float(channel_pct_value) >= 0.40:
            headwind_signals.extend([SIG_NARROW_CHANNEL_DEPENDENCE, SIG_REVENUE_BASE_FRAGILITY])
        elif float(channel_pct_value) < 0.20:
            support_signals.append(SIG_CHANNEL_DIVERSITY_PRESENT)

    if _is_num(end_market_pct_value):
        if float(end_market_pct_value) >= 0.50:
            headwind_signals.extend([SIG_NARROW_END_MARKET_DEPENDENCE, SIG_REVENUE_BASE_FRAGILITY])
        elif float(end_market_pct_value) < 0.30:
            support_signals.append(SIG_MULTIPLE_END_MARKETS_PRESENT)

    if _is_num(segment_count_value) and float(segment_count_value) >= 3.0:
        support_signals.append(SIG_DIVERSIFIED_REVENUE_BASE)
    if _is_num(channel_count_value) and float(channel_count_value) >= 3.0:
        support_signals.append(SIG_CHANNEL_DIVERSITY_PRESENT)
    if _is_num(end_market_count_value) and float(end_market_count_value) >= 3.0:
        support_signals.append(SIG_MULTIPLE_END_MARKETS_PRESENT)

    if support_signals and not headwind_signals:
        support_signals.append(SIG_CONCENTRATION_NOT_DOMINANT)

    if not evidence_known:
        headwind_signals.append(SIG_CONCENTRATION_DISCLOSURE_THIN)
        cls = REVENUE_DEPENDENCE_UNKNOWN
        caution = REVENUE_BASE_UNCLEAR
        reason_codes.extend([REASON_MISSING_REVENUE_DEPENDENCE_INPUTS, REASON_REVENUE_DEPENDENCE_UNKNOWN])
    else:
        headwind_strength = 0
        support_strength = 0
        if SIG_SINGLE_CUSTOMER_CONCENTRATION in headwind_signals:
            headwind_strength += 2
        if SIG_TOP_CUSTOMER_DOMINANCE in headwind_signals:
            headwind_strength += 2
        if SIG_NARROW_CHANNEL_DEPENDENCE in headwind_signals:
            headwind_strength += 2
        if SIG_NARROW_END_MARKET_DEPENDENCE in headwind_signals:
            headwind_strength += 2
        if SIG_REVENUE_BASE_FRAGILITY in headwind_signals:
            headwind_strength += 1

        if SIG_DIVERSIFIED_REVENUE_BASE in support_signals:
            support_strength += 2
        if SIG_CHANNEL_DIVERSITY_PRESENT in support_signals:
            support_strength += 1
        if SIG_MULTIPLE_END_MARKETS_PRESENT in support_signals:
            support_strength += 1
        if SIG_NO_MAJOR_CUSTOMER_CONCENTRATION_DISCLOSED in support_signals:
            support_strength += 1
        if SIG_CONCENTRATION_NOT_DOMINANT in support_signals:
            support_strength += 1

        if headwind_strength >= 3:
            cls = HIGH_REVENUE_DEPENDENCE_RISK
            caution = REVENUE_BASE_HEADWIND
            reason_codes.append(REASON_HIGH_REVENUE_DEPENDENCE_HEADWIND)
        elif support_strength >= 3 and headwind_strength == 0:
            cls = LOW_REVENUE_DEPENDENCE_RISK
            caution = REVENUE_BASE_SUPPORTIVE
            reason_codes.append(REASON_LOW_REVENUE_DEPENDENCE_SUPPORT)
        else:
            cls = MODERATE_REVENUE_DEPENDENCE_RISK
            caution = REVENUE_BASE_MIXED
            reason_codes.append(REASON_REVENUE_BASE_MIXED)

        if SIG_SINGLE_CUSTOMER_CONCENTRATION in headwind_signals or SIG_TOP_CUSTOMER_DOMINANCE in headwind_signals:
            reason_codes.append(REASON_CUSTOMER_CONCENTRATION_HEADWIND)
        if SIG_NARROW_CHANNEL_DEPENDENCE in headwind_signals:
            reason_codes.append(REASON_CHANNEL_DEPENDENCE_HEADWIND)

        if (
            cls == MODERATE_REVENUE_DEPENDENCE_RISK
            and returns_class == "HIGH_RETURNS_PERSISTENCE"
            and cyclical_risk_class not in {"PEAK_EARNINGS_RISK", "TROUGH_EARNINGS_RISK"}
            and impairment_class not in {"CLEAR_IMPAIRMENT", "PROBABLE_IMPAIRMENT"}
            and normalization_class != "LOW_NORMALIZATION_CREDIBILITY"
        ):
            support_signals.append(SIG_CONCENTRATION_NOT_DOMINANT)

    if SIG_CONCENTRATION_DISCLOSURE_THIN in headwind_signals:
        reason_codes.append(REASON_REVENUE_DEPENDENCE_UNKNOWN)

    support_signals = _dedupe(support_signals)
    headwind_signals = _dedupe(headwind_signals)
    reason_codes = _dedupe(reason_codes)
    derived_from = _collect_derived_from(
        returns_persistence_payload,
        cyclical_normalization_payload,
        impairment_classification_payload,
        normalization_credibility_payload,
        row_refs=(
            list(fundamentals.get("derived_from") or [])
            + latest_customer_refs
            + latest_channel_refs
            + latest_end_market_refs
            + latest_segment_refs
            + latest_channel_count_refs
            + latest_end_market_count_refs
            + list(row_derived_from or [])
        ),
    )

    payload = {
        "ticker": ticker_norm,
        "as_of_date": str(as_of_date or ""),
        "revenue_dependence_risk_class": cls,
        "revenue_dependence_risk_reason_codes": reason_codes,
        "revenue_dependence_support_signals": support_signals,
        "revenue_dependence_headwind_signals": headwind_signals,
        "primary_revenue_dependence_caution": caution,
        "revenue_fragility_summary": _summary_for_class(cls=cls, headwinds=headwind_signals, supports=support_signals),
        "customer_concentration_pct": _to_num(customer_pct_value),
        "top_channel_pct": _to_num(channel_pct_value),
        "top_end_market_pct": _to_num(end_market_pct_value),
        "segment_count": _to_num(segment_count_value),
        "channel_count": _to_num(channel_count_value),
        "end_market_count": _to_num(end_market_count_value),
        "derived_from": derived_from,
        "claims": {
            "revenue_dependence_risk_class": _claim(
                value=cls,
                refs=derived_from,
                reason_code=REASON_REVENUE_DEPENDENCE_UNKNOWN,
            ),
        },
        "generated_at": utc_now_iso(),
    }
    return payload


def write_revenue_dependence_for_run(
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

    rows_in = [row for row in (scoreboard_rows or []) if isinstance(row, dict)]
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("revenue_dependence_detail")
        for row in rows_in
        if isinstance(row.get("revenue_dependence_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    counts_by_class: dict[str, int] = {}
    counts_by_caution: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for ticker in sorted({str(token).strip().upper() for token in tickers if str(token).strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            score_row = next((row for row in rows_in if str(row.get("ticker") or "").upper() == ticker), {})
            detail = compute_revenue_dependence(
                ticker=ticker,
                as_of_date=as_of_date,
                fundamentals=score_row.get("fundamentals_detail")
                if isinstance(score_row.get("fundamentals_detail"), dict)
                else {},
                returns_persistence_payload=score_row.get("returns_persistence_detail")
                if isinstance(score_row.get("returns_persistence_detail"), dict)
                else {},
                cyclical_normalization_payload=score_row.get("cyclical_normalization_detail")
                if isinstance(score_row.get("cyclical_normalization_detail"), dict)
                else {},
                impairment_classification_payload=score_row.get("impairment_classification_detail")
                if isinstance(score_row.get("impairment_classification_detail"), dict)
                else {},
                normalization_credibility_payload=score_row.get("normalization_credibility_detail")
                if isinstance(score_row.get("normalization_credibility_detail"), dict)
                else {},
                customer_concentration_pct=score_row.get("customer_concentration_pct", UNKNOWN),
                top_customer_pct=score_row.get("top_customer_pct", UNKNOWN),
                top_channel_pct=score_row.get("top_channel_pct", UNKNOWN),
                top_end_market_pct=score_row.get("top_end_market_pct", UNKNOWN),
                segment_count=score_row.get("segment_count", UNKNOWN),
                channel_count=score_row.get("channel_count", UNKNOWN),
                end_market_count=score_row.get("end_market_count", UNKNOWN),
                customer_concentration_flag=score_row.get("customer_concentration_flag", UNKNOWN),
                price_status=score_row.get("price_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                row_derived_from=list(score_row.get("derived_from") or []),
            )
        cls = str(detail.get("revenue_dependence_risk_class") or REVENUE_DEPENDENCE_UNKNOWN)
        caution = str(detail.get("primary_revenue_dependence_caution") or REVENUE_BASE_UNCLEAR)
        counts_by_class[cls] = int(counts_by_class.get(cls) or 0) + 1
        counts_by_caution[caution] = int(counts_by_caution.get(caution) or 0) + 1
        for code in [
            str(code)
            for code in (detail.get("revenue_dependence_risk_reason_codes") or [])
            if str(code).strip()
        ]:
            reason_counts[code] = int(reason_counts.get(code) or 0) + 1
        rows.append(
            {
                "ticker": ticker,
                "revenue_dependence_risk_class": cls,
                "revenue_dependence_risk_reason_codes": [
                    str(code)
                    for code in (detail.get("revenue_dependence_risk_reason_codes") or [])
                    if str(code).strip()
                ],
                "revenue_dependence_support_signals": [
                    str(code)
                    for code in (detail.get("revenue_dependence_support_signals") or [])
                    if str(code).strip()
                ],
                "revenue_dependence_headwind_signals": [
                    str(code)
                    for code in (detail.get("revenue_dependence_headwind_signals") or [])
                    if str(code).strip()
                ],
                "primary_revenue_dependence_caution": caution,
                "revenue_fragility_summary": str(detail.get("revenue_fragility_summary") or ""),
                "derived_from": [
                    str(ref)
                    for ref in (detail.get("derived_from") or [])
                    if str(ref).strip()
                ],
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            _CLASS_ORDER.get(
                str(row.get("revenue_dependence_risk_class") or REVENUE_DEPENDENCE_UNKNOWN),
                len(_CLASS_ORDER),
            ),
            str(row.get("ticker") or ""),
        ),
    )

    def _class_rows(value: str) -> list[dict[str, Any]]:
        return [
            row
            for row in rows_sorted
            if str(row.get("revenue_dependence_risk_class") or REVENUE_DEPENDENCE_UNKNOWN) == value
        ]

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "generated_at": utc_now_iso(),
        "ticker_count": len(rows_sorted),
        "rows": rows_sorted,
        "counts_by_revenue_dependence_risk_class": dict(sorted(counts_by_class.items())),
        "counts_by_primary_revenue_dependence_caution": dict(sorted(counts_by_caution.items())),
        "top_10_low_revenue_dependence_risk": [
            {
                "ticker": row.get("ticker", ""),
                "revenue_dependence_risk_reason_codes": [
                    str(code)
                    for code in (row.get("revenue_dependence_risk_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows(LOW_REVENUE_DEPENDENCE_RISK)[:10]
        ],
        "top_10_high_revenue_dependence_risk": [
            {
                "ticker": row.get("ticker", ""),
                "revenue_dependence_risk_reason_codes": [
                    str(code)
                    for code in (row.get("revenue_dependence_risk_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows(HIGH_REVENUE_DEPENDENCE_RISK)[:10]
        ],
        "most_common_revenue_dependence_reason_codes": [
            {"reason_code": code, "count": count}
            for code, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:10]
        ],
        "revenue_dependence_path": str(output_path),
    }
    _json_write(output_path, payload)
    return payload


def _revenue_dependence_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "revenue_dependence.json",
        cfg.sectors_dir / run_id / "revenue_dependence.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def open_revenue_dependence(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _revenue_dependence_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "revenue_dependence_path": str(path) if path is not None else "",
        }

    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_revenue_dependence_risk_class": (
            payload.get("counts_by_revenue_dependence_risk_class")
            if isinstance(payload.get("counts_by_revenue_dependence_risk_class"), dict)
            else {}
        ),
        "counts_by_primary_revenue_dependence_caution": (
            payload.get("counts_by_primary_revenue_dependence_caution")
            if isinstance(payload.get("counts_by_primary_revenue_dependence_caution"), dict)
            else {}
        ),
        "top_10_low_revenue_dependence_risk": [
            row for row in (payload.get("top_10_low_revenue_dependence_risk") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_high_revenue_dependence_risk": [
            row for row in (payload.get("top_10_high_revenue_dependence_risk") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "most_common_revenue_dependence_reason_codes": [
            row for row in (payload.get("most_common_revenue_dependence_reason_codes") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "revenue_dependence_path": str(path),
    }
