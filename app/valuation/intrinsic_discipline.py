from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.market.company_facts_extract import FCF_DIRECT_TAG_PRIORITY, SHARES_TAG_PRIORITY
from app.valuation.owner_earnings import (
    DEFAULT_MAINT_CAPEX_RATIO,
    _annual_series_for_priority,
    _dedupe_refs,
    _load_companyfacts_payload,
    compute_owner_earnings_series,
)
from app.valuation.maintenance_capex_discipline import (
    LOW_MAINTENANCE_CAPEX_CREDIBILITY,
    MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN,
    REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND,
    REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN,
    REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE,
)


UNKNOWN = "UNKNOWN"
OK = "OK"
# Facts rows report USD and shares in millions; the earnings series here are whole units.
_MILLION = 1_000_000.0
LOW_CONFIDENCE = "LOW_CONFIDENCE"

REASON_OWNER_EARNINGS_SELECTED = "OWNER_EARNINGS_SELECTED"
REASON_FCF_SELECTED = "FCF_SELECTED"
REASON_CFO_PROXY_SELECTED = "CFO_PROXY_SELECTED"
REASON_INSUFFICIENT_NORMALIZED_INPUTS = "INSUFFICIENT_NORMALIZED_INPUTS"
REASON_NEGATIVE_NORMALIZED_EARNINGS = "NEGATIVE_NORMALIZED_EARNINGS"
REASON_CYCLICAL_NORMALIZATION_LOW_CONFIDENCE = "CYCLICAL_NORMALIZATION_LOW_CONFIDENCE"

REASON_FLOOR_FROM_NETNET = "FLOOR_FROM_NETNET"
REASON_FLOOR_FROM_EPV = "FLOOR_FROM_EPV"
REASON_FLOOR_FROM_NORMALIZED_EARNINGS_POWER = "FLOOR_FROM_NORMALIZED_EARNINGS_POWER"
REASON_FLOOR_FROM_EXISTING_CONSERVATIVE = "FLOOR_FROM_EXISTING_CONSERVATIVE"
REASON_BASE_FROM_NORMALIZED_EARNINGS_POWER = "BASE_FROM_NORMALIZED_EARNINGS_POWER"
REASON_BASE_FROM_EXISTING_INTRINSIC = "BASE_FROM_EXISTING_INTRINSIC"
REASON_BASE_FROM_EPV = "BASE_FROM_EPV"
REASON_CEILING_FROM_NORMALIZED_EARNINGS_POWER = "CEILING_FROM_NORMALIZED_EARNINGS_POWER"
REASON_CEILING_FROM_EXISTING_INTRINSIC = "CEILING_FROM_EXISTING_INTRINSIC"
REASON_RANGE_LOW_CONFIDENCE = "RANGE_LOW_CONFIDENCE"
REASON_RANGE_UNKNOWN_MISSING_PRICE = "RANGE_UNKNOWN_MISSING_PRICE"
REASON_RANGE_UNKNOWN_MISSING_SHARES = "RANGE_UNKNOWN_MISSING_SHARES"
REASON_RANGE_UNKNOWN_MISSING_EARNINGS_SUPPORT = "RANGE_UNKNOWN_MISSING_EARNINGS_SUPPORT"

MOS_DEEP_VALUE_SUPPORT = "DEEP_VALUE_SUPPORT"
MOS_ADEQUATE = "ADEQUATE_MARGIN_OF_SAFETY"
MOS_MODEST = "MODEST_MARGIN_OF_SAFETY"
MOS_NONE = "NO_MARGIN_OF_SAFETY"
MOS_UNKNOWN = "MOS_UNKNOWN"

SUPPORT_ASSET = "ASSET_SUPPORT"
SUPPORT_EARNINGS = "EARNINGS_POWER_SUPPORT"
SUPPORT_BALANCE_SHEET = "BALANCE_SHEET_SUPPORT"
SUPPORT_LIMITED = "LIMITED_SUPPORT"
SUPPORT_UNKNOWN = "UNKNOWN_SUPPORT"

_CFO_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]
_CAPEX_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
]


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


def _claim(
    *,
    value: Any,
    refs: list[Any],
    reason_code: str,
    status: str | None = None,
) -> dict[str, Any]:
    derived = _dedupe_refs([str(ref) for ref in refs if str(ref).strip()])
    if _is_num(value):
        effective_status = str(status or OK).upper()
        effective_reason = OK if effective_status == OK else str(reason_code or UNKNOWN)
        return {
            "value": float(value),
            "status": effective_status,
            "reason_code": effective_reason,
            "derived_from": derived,
        }
    return {
        "value": UNKNOWN,
        "status": str(status or UNKNOWN).upper(),
        "reason_code": str(reason_code or UNKNOWN),
        "derived_from": derived,
    }


def _series_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        refs = [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()]
        out[year] = {
            "year": year,
            "value": float(row.get("value")) if _is_num(row.get("value")) else UNKNOWN,
            "derived_from": refs,
        }
    return out


def _trace_refs(trace_map: dict[str, Any], year: int, metric: str, fallback: str) -> list[str]:
    year_bucket = trace_map.get(str(year)) if isinstance(trace_map, dict) else {}
    metric_bucket = year_bucket.get(metric) if isinstance(year_bucket, dict) else {}
    if isinstance(metric_bucket, dict):
        refs = [str(ref) for ref in (metric_bucket.get("derived_from") or []) if str(ref).strip()]
        if refs:
            return refs
    return [fallback]


def _fundamentals_rows(fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    return sorted(rows, key=lambda row: int(row.get("year") or 0))


def _fundamentals_trace_map(fundamentals: dict[str, Any]) -> dict[str, Any]:
    return fundamentals.get("row_traces") if isinstance(fundamentals.get("row_traces"), dict) else {}


def _owner_series_from_owner_payload(owner_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in [row for row in (owner_payload.get("series") or []) if isinstance(row, dict)]:
        year = int(raw.get("year") or 0)
        owner_value = raw.get("owner_earnings", UNKNOWN)
        if year <= 0:
            continue
        refs = [str(ref) for ref in (raw.get("derived_from") or []) if str(ref).strip()]
        if not refs:
            refs = [str(ref) for ref in (owner_payload.get("derived_from") or []) if str(ref).strip()]
        rows.append({"year": year, "value": _to_num(owner_value), "derived_from": refs})
    return rows


def _series_from_companyfacts(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...],
) -> list[dict[str, Any]]:
    series = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=priority,
        expected_unit_exact=expected_unit_exact,
    )
    return [
        {
            "year": int(row.get("year") or 0),
            "value": float(row.get("value")) if _is_num(row.get("value")) else UNKNOWN,
            "derived_from": [str(row.get("ref"))] if str(row.get("ref") or "").strip() else [],
        }
        for row in series
        if int(row.get("year") or 0) > 0
    ]


def _build_series_from_fundamentals(fundamentals: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = _fundamentals_rows(fundamentals)
    traces = _fundamentals_trace_map(fundamentals)

    owner_rows: list[dict[str, Any]] = []
    cfo_rows: list[dict[str, Any]] = []
    fcf_rows: list[dict[str, Any]] = []
    shares_rows: list[dict[str, Any]] = []
    net_debt_rows: list[dict[str, Any]] = []

    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        cfo = row.get("cfo", UNKNOWN)
        capex = row.get("capex", UNKNOWN)
        fcf = row.get("fcf", UNKNOWN)
        shares = row.get("shares_outstanding", UNKNOWN)
        net_debt = row.get("net_debt", UNKNOWN)
        cfo_rows.append(
            {
                "year": year,
                "value": _to_num(cfo),
                "derived_from": _trace_refs(traces, year, "cfo", f"fundamentals.rows[{year}].cfo"),
            }
        )
        fcf_rows.append(
            {
                "year": year,
                "value": _to_num(fcf),
                "derived_from": _trace_refs(traces, year, "fcf", f"fundamentals.rows[{year}].fcf"),
            }
        )
        if _is_num(shares):
            shares_rows.append(
                {
                    "year": year,
                    "value": float(shares),
                    "derived_from": _trace_refs(traces, year, "shares_outstanding", f"fundamentals.rows[{year}].shares_outstanding"),
                }
            )
        if _is_num(net_debt):
            net_debt_rows.append(
                {
                    "year": year,
                    "value": float(net_debt),
                    "derived_from": _trace_refs(traces, year, "net_debt", f"fundamentals.rows[{year}].net_debt"),
                }
            )
        owner_rows.append(
            {
                "year": year,
                "value": float(cfo) - (float(DEFAULT_MAINT_CAPEX_RATIO) * abs(float(capex)))
                if _is_num(cfo) and _is_num(capex) else UNKNOWN,
                "derived_from": _dedupe_refs(
                    _trace_refs(traces, year, "cfo", f"fundamentals.rows[{year}].cfo")
                    + _trace_refs(traces, year, "capex", f"fundamentals.rows[{year}].capex")
                    + [f"derived:maintenance_capex_ratio={float(DEFAULT_MAINT_CAPEX_RATIO):.2f}"]
                ),
            }
        )
    return {
        "owner_earnings": owner_rows,
        "cfo": cfo_rows,
        "fcf": fcf_rows,
        "shares": shares_rows,
        "net_debt": net_debt_rows,
    }


def _build_series_from_facts(
    *,
    as_of_date: str,
    facts_row: dict[str, Any],
    owner_payload: dict[str, Any] | None,
    net_debt_value: Any,
    net_debt_refs: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    companyfacts = _load_companyfacts_payload(facts_row)
    owner_rows = _owner_series_from_owner_payload(owner_payload or {})
    cfo_rows: list[dict[str, Any]] = []
    fcf_rows: list[dict[str, Any]] = []
    shares_rows: list[dict[str, Any]] = []
    has_cfo_history = False
    has_fcf_history = False
    if companyfacts:
        cfo_rows = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=_CFO_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
        fcf_rows = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=FCF_DIRECT_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
        # Missing-row padding must not disable the existing scalar fallback
        # when this raw metric never had any observations in the first place.
        has_cfo_history = bool(cfo_rows)
        has_fcf_history = bool(fcf_rows)
        capex_rows = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=_CAPEX_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
        cfo_by_year = _series_map(cfo_rows)
        capex_by_year = _series_map(capex_rows)
        if not fcf_rows:
            derived_fcf: list[dict[str, Any]] = []
            for year in sorted(set(cfo_by_year) | set(capex_by_year)):
                cfo_row = cfo_by_year.get(year, {})
                capex_row = capex_by_year.get(year, {})
                derived_fcf.append(
                    {
                        "year": year,
                        "value": float(cfo_row["value"]) - abs(float(capex_row["value"]))
                        if _is_num(cfo_row.get("value")) and _is_num(capex_row.get("value"))
                        else UNKNOWN,
                        "derived_from": _dedupe_refs(
                            list(cfo_row.get("derived_from") or [])
                            + list(capex_row.get("derived_from") or [])
                            + ["derived:fcf=cfo-abs(capex)"]
                        ),
                    }
                )
            fcf_rows = derived_fcf
            has_fcf_history = any(_is_num(row.get("value")) for row in fcf_rows)
        # Select the recent financial years before filtering each metric's value.
        # A year declared by another cash-flow source is evidence of a missing
        # input here, not permission to pull an older profit into the window.
        declared_years = sorted({
            int(row["year"])
            for row in cfo_rows + capex_rows + fcf_rows + owner_rows
            if 0 < int(row.get("year") or 0) < 9999
        })
        fcf_by_year = _series_map(fcf_rows)
        cfo_rows = [
            cfo_by_year.get(year, {"year": year, "value": UNKNOWN, "derived_from": []})
            for year in declared_years
        ]
        fcf_rows = [
            fcf_by_year.get(year, {"year": year, "value": UNKNOWN, "derived_from": []})
            for year in declared_years
        ]
        shares_rows = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=SHARES_TAG_PRIORITY,
            expected_unit_exact=("shares",),
        )
    facts_refs = [str(ref) for ref in (facts_row.get("derived_from") or []) if str(ref).strip()]
    # The facts row reports USD millions and shares in millions
    # (app.valuation.facts); the companyfacts series and the owner-earnings series
    # above are whole dollars and whole shares. Scale the scalar fallbacks up.
    if not has_cfo_history and _is_num(facts_row.get("cfo_value")):
        cfo_rows = [{"year": 9999, "value": float(facts_row["cfo_value"]) * _MILLION, "derived_from": facts_refs}]
    if not has_fcf_history and _is_num(facts_row.get("fcf_value")):
        fcf_rows = [{"year": 9999, "value": float(facts_row["fcf_value"]) * _MILLION, "derived_from": facts_refs}]
    if not shares_rows and _is_num(facts_row.get("shares_value")):
        shares_rows = [{"year": 9999, "value": float(facts_row["shares_value"]) * _MILLION, "derived_from": facts_refs}]
    net_debt_rows: list[dict[str, Any]] = []
    if _is_num(net_debt_value):
        net_debt_rows = [{"year": 9999, "value": float(net_debt_value), "derived_from": [str(ref) for ref in net_debt_refs if str(ref).strip()]}]
    return {
        "owner_earnings": owner_rows,
        "cfo": cfo_rows,
        "fcf": fcf_rows,
        "shares": shares_rows,
        "net_debt": net_debt_rows,
    }


def _median_positive(rows: list[dict[str, Any]], *, window: int = 3) -> tuple[float | str, list[str], int]:
    window = max(1, int(window))
    ordered = sorted(rows, key=lambda row: int(row.get("year") or 0))
    dated_years = [
        int(row.get("year") or 0) for row in ordered if 0 < int(row.get("year") or 0) < 9999
    ]
    if dated_years:
        # The window is the latest fiscal year and its (window - 1) predecessors.
        # Missing years inside it stay missing; they never pull older rows in, so a
        # "3y" figure cannot be built from years 2018, 2021 and 2024.
        latest_year = max(dated_years)
        recent = [
            row for row in ordered
            if latest_year - window < int(row.get("year") or 0) <= latest_year
        ]
    else:
        # Undated scalar fallback rows (year 9999 / unknown) keep the row-count rule.
        recent = ordered[-window:]
    # The median runs over EVERY measured year in the window, losses included:
    # dropping the non-positive years first made [-500, -400, 100] a normalized
    # 100. A normalized figure also needs at least two profitable years; one
    # good year beside losses (or a lone scalar) is not a normal level.
    numeric = [row for row in recent if _is_num(row.get("value"))]
    positive_count = len([row for row in numeric if float(row["value"]) > 0.0])
    if positive_count < 2:
        return UNKNOWN, _dedupe_refs([ref for row in recent for ref in (row.get("derived_from") or [])]), 0
    return (
        float(median([float(row["value"]) for row in numeric])),
        _dedupe_refs([ref for row in numeric for ref in (row.get("derived_from") or [])]),
        len(numeric),
    )


def _latest_numeric(rows: list[dict[str, Any]]) -> tuple[float | str, list[str]]:
    numeric = [row for row in rows if _is_num(row.get("value"))]
    if not numeric:
        return UNKNOWN, _dedupe_refs([ref for row in rows for ref in (row.get("derived_from") or [])])
    latest = sorted(numeric, key=lambda row: int(row.get("year") or 0))[-1]
    return float(latest["value"]), [str(ref) for ref in (latest.get("derived_from") or []) if str(ref).strip()]


def _selected_quality_support(
    *,
    owner_quality_payload: dict[str, Any] | None,
    intangible_payload: dict[str, Any] | None,
) -> tuple[bool, bool]:
    owner_quality = owner_quality_payload or {}
    intangible = intangible_payload or {}
    stability = owner_quality.get("owner_earnings_stability_score", UNKNOWN)
    oe_total = owner_quality.get("oe_quality_total", UNKNOWN)
    cycle = intangible.get("cycle_resilience_score", UNKNOWN)
    quality_ok = (_is_num(stability) and float(stability) >= 3.0) or (_is_num(oe_total) and float(oe_total) >= 6.0)
    low_confidence = (
        (_is_num(stability) and float(stability) <= 1.0)
        or (_is_num(cycle) and float(cycle) <= 1.0)
    )
    return quality_ok, low_confidence


def _latest_fiscal_year(rows: list[dict[str, Any]]) -> int:
    # 9999 is the existing undated scalar fallback, not a financial period.
    return max(
        (int(row.get("year") or 0) for row in rows
         if 0 < int(row.get("year") or 0) < 9999),
        default=0,
    )


def _normalized_earnings_power(
    *,
    owner_payload: dict[str, Any] | None,
    series_map: dict[str, list[dict[str, Any]]],
    owner_quality_payload: dict[str, Any] | None,
    intangible_payload: dict[str, Any] | None,
    maintenance_capex_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    owner_payload = owner_payload or {}
    owner_summary = owner_payload.get("summary") if isinstance(owner_payload.get("summary"), dict) else {}
    owner_from_summary = owner_summary.get("owner_earnings_normalized_3y", UNKNOWN)
    owner_summary_method = str(owner_summary.get("owner_earnings_normalized_method") or REASON_INSUFFICIENT_NORMALIZED_INPUTS)
    owner_refs = [str(ref) for ref in (owner_payload.get("derived_from") or []) if str(ref).strip()]
    latest_financial_year = _latest_fiscal_year([
        row for metric in ("owner_earnings", "fcf", "cfo")
        for row in (series_map.get(metric) or [])
    ])
    owner_series_year = _latest_fiscal_year(series_map.get("owner_earnings") or [])
    if owner_series_year and owner_series_year < latest_financial_year:
        # The facts path obtains this series from the owner-earnings payload itself.
        # Rejecting only the summary would re-admit the same stale source here.
        series_map = {**series_map, "owner_earnings": []}
    summary_year = _latest_fiscal_year([
        row for row in _owner_series_from_owner_payload(owner_payload)
        if _is_num(row.get("value"))
    ])
    try:
        summary_asof_year = date.fromisoformat(str(owner_payload.get("as_of_date") or "")).year
    except ValueError:
        summary_asof_year = 0
    if (
        (summary_year and summary_year < latest_financial_year)
        or (summary_asof_year and summary_asof_year < latest_financial_year)
        # A summary with neither a series year nor an as-of date cannot be shown
        # to cover the dated financial years on file, so it cannot override them.
        or (latest_financial_year and not summary_year and not summary_asof_year)
    ):
        owner_from_summary = UNKNOWN
    owner_value, owner_series_refs, owner_points = _median_positive(series_map.get("owner_earnings") or [], window=3)
    if _is_num(owner_from_summary) and float(owner_from_summary) > 0.0:
        owner_value = float(owner_from_summary)
        owner_series_refs = owner_refs or owner_series_refs
        owner_points = int(owner_summary.get("owner_earnings_points") or owner_points or 0)

    fcf_value, fcf_refs, fcf_points = _median_positive(series_map.get("fcf") or [], window=3)
    cfo_value, cfo_refs, cfo_points = _median_positive(series_map.get("cfo") or [], window=3)
    quality_ok, quality_low_confidence = _selected_quality_support(
        owner_quality_payload=owner_quality_payload,
        intangible_payload=intangible_payload,
    )
    maintenance_capex_payload = (
        maintenance_capex_payload if isinstance(maintenance_capex_payload, dict) else {}
    )
    maintenance_capex_class = str(
        maintenance_capex_payload.get("maintenance_capex_credibility_class")
        or MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
    ).upper()
    maintenance_refs = [
        str(ref) for ref in (maintenance_capex_payload.get("derived_from") or []) if str(ref).strip()
    ]

    positive_candidates: list[dict[str, Any]] = []
    if _is_num(owner_value) and float(owner_value) > 0.0:
        positive_candidates.append(
            {
                "method": REASON_OWNER_EARNINGS_SELECTED,
                "value": float(owner_value),
                "refs": owner_series_refs,
                "points": owner_points,
            }
        )
    if _is_num(fcf_value) and float(fcf_value) > 0.0:
        positive_candidates.append(
            {
                "method": REASON_FCF_SELECTED,
                "value": float(fcf_value),
                "refs": fcf_refs,
                "points": fcf_points,
            }
        )
    if _is_num(cfo_value) and float(cfo_value) > 0.0:
        positive_candidates.append(
            {
                "method": REASON_CFO_PROXY_SELECTED,
                "value": float(cfo_value),
                "refs": cfo_refs,
                "points": cfo_points,
            }
        )

    selected: dict[str, Any] | None = None
    if positive_candidates:
        positive_candidates = sorted(
            positive_candidates,
            key=lambda item: (
                float(item.get("value") or 0.0),
                0 if str(item.get("method")) == REASON_OWNER_EARNINGS_SELECTED else (1 if str(item.get("method")) == REASON_FCF_SELECTED else 2),
            ),
        )
        owner_candidate = next((item for item in positive_candidates if item["method"] == REASON_OWNER_EARNINGS_SELECTED), None)
        alt_floor = min(
            [float(item["value"]) for item in positive_candidates if item["method"] != REASON_OWNER_EARNINGS_SELECTED],
            default=UNKNOWN,
        )
        if (
            isinstance(owner_candidate, dict)
            and quality_ok
            and (not _is_num(alt_floor) or float(owner_candidate["value"]) <= float(alt_floor) * 1.20)
        ):
            selected = owner_candidate
        else:
            selected = positive_candidates[0]

    reason_codes: list[str] = []
    status = OK
    method_used = UNKNOWN
    value = UNKNOWN
    refs: list[str] = []

    if isinstance(selected, dict):
        method_used = str(selected.get("method") or UNKNOWN)
        value = float(selected.get("value"))
        refs = [str(ref) for ref in (selected.get("refs") or []) if str(ref).strip()]
        reason_codes.append(method_used)
        candidate_values = [float(item["value"]) for item in positive_candidates]
        material_disagreement = (
            len(candidate_values) >= 2
            and min(candidate_values) > 0.0
            and (max(candidate_values) / min(candidate_values)) >= 1.5
        )
        if method_used == REASON_CFO_PROXY_SELECTED or material_disagreement or quality_low_confidence:
            status = LOW_CONFIDENCE
            reason_codes.append(REASON_CYCLICAL_NORMALIZATION_LOW_CONFIDENCE)
        if method_used == REASON_OWNER_EARNINGS_SELECTED and str(owner_summary_method).upper() not in {"", UNKNOWN}:
            refs = _dedupe_refs(refs + [f"owner_earnings.summary.method={owner_summary_method}"])
        if method_used == REASON_OWNER_EARNINGS_SELECTED and maintenance_capex_class == LOW_MAINTENANCE_CAPEX_CREDIBILITY:
            status = LOW_CONFIDENCE
            reason_codes.extend(
                [
                    REASON_CYCLICAL_NORMALIZATION_LOW_CONFIDENCE,
                    REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND,
                    REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE,
                ]
            )
            refs = _dedupe_refs(refs + maintenance_refs)
        elif method_used == REASON_OWNER_EARNINGS_SELECTED and maintenance_capex_class == MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN:
            reason_codes.append(REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN)
            refs = _dedupe_refs(refs + maintenance_refs)
    else:
        # Positive-only normalization cannot supply a negative candidate. Inspect
        # the same recent observations to distinguish measured losses from absence.
        recent_rows = [
            row
            for metric in ("owner_earnings", "fcf", "cfo")
            for row in sorted(
                series_map.get(metric) or [], key=lambda item: int(item.get("year") or 0)
            )[-3:]
        ]
        numeric_candidates = [row["value"] for row in recent_rows if _is_num(row.get("value"))]
        if _is_num(owner_from_summary):
            numeric_candidates.append(owner_from_summary)
        if numeric_candidates and max(float(value) for value in numeric_candidates) < 0.0:
            reason_codes.append(REASON_NEGATIVE_NORMALIZED_EARNINGS)
        else:
            reason_codes.append(REASON_INSUFFICIENT_NORMALIZED_INPUTS)

    claim = _claim(
        value=value,
        refs=refs,
        reason_code=reason_codes[0] if reason_codes else REASON_INSUFFICIENT_NORMALIZED_INPUTS,
        status=status if _is_num(value) else UNKNOWN,
    )
    return {
        "normalized_earnings_power_value": claim.get("value", UNKNOWN),
        "normalized_earnings_power_method_used": method_used,
        "normalized_earnings_power_status": str(claim.get("status") or UNKNOWN),
        "normalized_earnings_power_reason_codes": _dedupe_refs(reason_codes),
        "normalized_earnings_power_derived_from": list(claim.get("derived_from") or []),
        "claims": {"normalized_earnings_power_value": claim},
    }


def _per_share_value(
    *,
    earnings_power_value: Any,
    shares_value: Any,
    net_debt_value: Any,
    multiple: float,
) -> float | str:
    # The normalized stream (owner earnings, FCF, CFO) is LEVERED: US GAAP puts
    # cash interest inside operating cash flow, so it is already an equity-level
    # cash flow and a multiple of it is an equity value. Subtracting net debt as
    # well counted the debt twice (app/valuation/owner_earnings.py header).
    # ``net_debt_value`` is kept in the signature for the callers' provenance.
    del net_debt_value
    if not _is_num(earnings_power_value) or float(earnings_power_value) <= 0.0:
        return UNKNOWN
    if not _is_num(shares_value) or float(shares_value) <= 0.0:
        return UNKNOWN
    equity_value = float(earnings_power_value) * float(multiple)
    return equity_value / float(shares_value)


def _first_reason(claim: dict[str, Any], fallback: str) -> str:
    return str(claim.get("reason_code") or fallback)


def _valuation_range(
    *,
    normalized_payload: dict[str, Any],
    shares_value: Any,
    shares_refs: list[Any],
    net_debt_value: Any,
    net_debt_refs: list[Any],
    price_value: Any,
    price_refs: list[Any],
    epv_per_share: Any,
    epv_refs: list[Any],
    netnet_per_share: Any,
    netnet_refs: list[Any],
    existing_intrinsic_base: Any,
    existing_intrinsic_base_refs: list[Any],
    existing_intrinsic_conservative: Any,
    existing_intrinsic_conservative_refs: list[Any],
    existing_intrinsic_ceiling: Any,
    existing_intrinsic_ceiling_refs: list[Any],
) -> dict[str, Any]:
    normalized_value = normalized_payload.get("normalized_earnings_power_value", UNKNOWN)
    normalized_refs = list(normalized_payload.get("normalized_earnings_power_derived_from") or [])
    normalized_status = str(normalized_payload.get("normalized_earnings_power_status") or UNKNOWN)
    normalized_floor = _per_share_value(
        earnings_power_value=normalized_value,
        shares_value=shares_value,
        net_debt_value=net_debt_value,
        multiple=8.0,
    )
    normalized_base = _per_share_value(
        earnings_power_value=normalized_value,
        shares_value=shares_value,
        net_debt_value=net_debt_value,
        multiple=10.0,
    )
    normalized_ceiling = _per_share_value(
        earnings_power_value=normalized_value,
        shares_value=shares_value,
        net_debt_value=net_debt_value,
        multiple=12.0,
    )
    normalized_range_refs = _dedupe_refs(
        normalized_refs
        + [str(ref) for ref in shares_refs if str(ref).strip()]
        + [str(ref) for ref in net_debt_refs if str(ref).strip()]
        + [
            "intrinsic_discipline.multiple.floor=8.0x",
            "intrinsic_discipline.multiple.base=10.0x",
            "intrinsic_discipline.multiple.ceiling=12.0x",
        ]
    )

    floor_candidates: list[tuple[float, str, list[str]]] = []
    if _is_num(netnet_per_share) and float(netnet_per_share) > 0.0:
        floor_candidates.append((float(netnet_per_share), REASON_FLOOR_FROM_NETNET, [str(ref) for ref in netnet_refs if str(ref).strip()]))
    if _is_num(epv_per_share) and float(epv_per_share) > 0.0:
        floor_candidates.append((float(epv_per_share), REASON_FLOOR_FROM_EPV, [str(ref) for ref in epv_refs if str(ref).strip()]))
    if _is_num(existing_intrinsic_conservative) and float(existing_intrinsic_conservative) > 0.0:
        floor_candidates.append(
            (
                float(existing_intrinsic_conservative),
                REASON_FLOOR_FROM_EXISTING_CONSERVATIVE,
                [str(ref) for ref in existing_intrinsic_conservative_refs if str(ref).strip()],
            )
        )
    if _is_num(normalized_floor) and float(normalized_floor) > 0.0:
        floor_candidates.append((float(normalized_floor), REASON_FLOOR_FROM_NORMALIZED_EARNINGS_POWER, normalized_range_refs))

    floor_value = min((value for value, _reason, _refs in floor_candidates), default=UNKNOWN)
    floor_candidate = next((item for item in floor_candidates if _is_num(floor_value) and float(item[0]) == float(floor_value)), None)
    floor_reason = floor_candidate[1] if isinstance(floor_candidate, tuple) else REASON_RANGE_UNKNOWN_MISSING_EARNINGS_SUPPORT
    floor_refs = list(floor_candidate[2]) if isinstance(floor_candidate, tuple) else []

    if _is_num(normalized_base) and float(normalized_base) > 0.0:
        base_value = float(normalized_base)
        base_reason = REASON_BASE_FROM_NORMALIZED_EARNINGS_POWER
        base_refs = list(normalized_range_refs)
    elif _is_num(existing_intrinsic_base) and float(existing_intrinsic_base) > 0.0:
        base_value = float(existing_intrinsic_base)
        base_reason = REASON_BASE_FROM_EXISTING_INTRINSIC
        base_refs = [str(ref) for ref in existing_intrinsic_base_refs if str(ref).strip()]
    elif _is_num(epv_per_share) and float(epv_per_share) > 0.0:
        base_value = float(epv_per_share)
        base_reason = REASON_BASE_FROM_EPV
        base_refs = [str(ref) for ref in epv_refs if str(ref).strip()]
    elif _is_num(floor_value):
        base_value = float(floor_value)
        base_reason = floor_reason
        base_refs = list(floor_refs)
    else:
        base_value = UNKNOWN
        base_reason = REASON_RANGE_UNKNOWN_MISSING_EARNINGS_SUPPORT
        base_refs = []

    if _is_num(floor_value) and _is_num(base_value) and float(base_value) < float(floor_value):
        base_value = float(floor_value)
        base_reason = floor_reason
        base_refs = list(floor_refs)

    ceiling_candidates: list[tuple[float, str, list[str]]] = []
    if _is_num(normalized_ceiling) and float(normalized_ceiling) > 0.0:
        ceiling_candidates.append((float(normalized_ceiling), REASON_CEILING_FROM_NORMALIZED_EARNINGS_POWER, normalized_range_refs))
    if _is_num(existing_intrinsic_ceiling) and float(existing_intrinsic_ceiling) > 0.0:
        ceiling_candidates.append(
            (
                float(existing_intrinsic_ceiling),
                REASON_CEILING_FROM_EXISTING_INTRINSIC,
                [str(ref) for ref in existing_intrinsic_ceiling_refs if str(ref).strip()],
            )
        )
    if _is_num(existing_intrinsic_base) and float(existing_intrinsic_base) > 0.0:
        ceiling_candidates.append(
            (
                float(existing_intrinsic_base),
                REASON_CEILING_FROM_EXISTING_INTRINSIC,
                [str(ref) for ref in existing_intrinsic_base_refs if str(ref).strip()],
            )
        )
    if _is_num(epv_per_share) and float(epv_per_share) > 0.0:
        ceiling_candidates.append((float(epv_per_share), REASON_BASE_FROM_EPV, [str(ref) for ref in epv_refs if str(ref).strip()]))

    if _is_num(base_value):
        eligible = [item for item in ceiling_candidates if float(item[0]) >= float(base_value)]
        if eligible:
            ceiling_candidate = sorted(eligible, key=lambda item: (float(item[0]), item[1]))[0]
            ceiling_value, ceiling_reason, ceiling_refs = ceiling_candidate
        else:
            ceiling_value, ceiling_reason, ceiling_refs = float(base_value), base_reason, list(base_refs)
    else:
        ceiling_value, ceiling_reason, ceiling_refs = UNKNOWN, REASON_RANGE_UNKNOWN_MISSING_EARNINGS_SUPPORT, []

    reason_codes: list[str] = []
    if _is_num(floor_value):
        reason_codes.append(floor_reason)
    if _is_num(base_value):
        reason_codes.append(base_reason)
    if _is_num(ceiling_value):
        reason_codes.append(ceiling_reason)

    status = OK
    if not (_is_num(floor_value) or _is_num(base_value) or _is_num(ceiling_value)):
        status = UNKNOWN
        if not _is_num(shares_value):
            reason_codes.append(REASON_RANGE_UNKNOWN_MISSING_SHARES)
        elif not (_is_num(normalized_value) or _is_num(epv_per_share) or _is_num(netnet_per_share) or _is_num(existing_intrinsic_base)):
            reason_codes.append(REASON_RANGE_UNKNOWN_MISSING_EARNINGS_SUPPORT)
    elif normalized_status == LOW_CONFIDENCE or len(set(reason_codes)) <= 1:
        status = LOW_CONFIDENCE
        reason_codes.append(REASON_RANGE_LOW_CONFIDENCE)
    if not _is_num(price_value):
        reason_codes.append(REASON_RANGE_UNKNOWN_MISSING_PRICE)

    claims = {
        "intrinsic_floor": _claim(
            value=floor_value,
            refs=floor_refs,
            reason_code=floor_reason,
            status=status if _is_num(floor_value) and status != UNKNOWN else (OK if _is_num(floor_value) else UNKNOWN),
        ),
        "intrinsic_base": _claim(
            value=base_value,
            refs=base_refs,
            reason_code=base_reason,
            status=status if _is_num(base_value) and status != UNKNOWN else (OK if _is_num(base_value) else UNKNOWN),
        ),
        "intrinsic_ceiling": _claim(
            value=ceiling_value,
            refs=ceiling_refs,
            reason_code=ceiling_reason,
            status=status if _is_num(ceiling_value) and status != UNKNOWN else (OK if _is_num(ceiling_value) else UNKNOWN),
        ),
    }
    return {
        "intrinsic_floor": claims["intrinsic_floor"].get("value", UNKNOWN),
        "intrinsic_base": claims["intrinsic_base"].get("value", UNKNOWN),
        "intrinsic_ceiling": claims["intrinsic_ceiling"].get("value", UNKNOWN),
        "valuation_range_status": status,
        "valuation_range_reason_codes": _dedupe_refs(reason_codes),
        "valuation_range_derived_from": _dedupe_refs(
            list(claims["intrinsic_floor"].get("derived_from") or [])
            + list(claims["intrinsic_base"].get("derived_from") or [])
            + list(claims["intrinsic_ceiling"].get("derived_from") or [])
            + [str(ref) for ref in price_refs if str(ref).strip()]
        ),
        "claims": claims,
    }


def _margin_of_safety(
    *,
    intrinsic_floor: Any,
    intrinsic_base: Any,
    price_value: Any,
    price_refs: list[Any],
) -> dict[str, Any]:
    """Compute mos_to_floor / mos_to_base.

    CONVENTION: these are UPSIDE RATIOS (intrinsic/price - 1), NOT the
    textbook (intrinsic-price)/intrinsic used by the scorecard — see
    app/valuation/mos_conventions.py (audit: dual-mos-convention-same-name).

    CONVENTION (FIX 6): these use the UPSIDE-RATIO convention,
        mos = intrinsic / price - 1
    (the fractional upside from price to intrinsic value), matching graham_dodd's
    mos_epv. This is NOT the textbook margin-of-safety (intrinsic - price)/intrinsic
    used in valuation_writer._compute_pricing_zone and thesis_updater. The 0.50 /
    0.25 / 0.10 classification thresholds below are calibrated to this convention.
    For intrinsic=150, price=100 this yields 0.50 (textbook would yield 0.333...).
    """
    mos_to_floor = (
        (float(intrinsic_floor) / float(price_value)) - 1.0  # upside-ratio
        if _is_num(intrinsic_floor) and _is_num(price_value) and float(price_value) > 0.0
        else UNKNOWN
    )
    mos_to_base = (
        (float(intrinsic_base) / float(price_value)) - 1.0  # upside-ratio
        if _is_num(intrinsic_base) and _is_num(price_value) and float(price_value) > 0.0
        else UNKNOWN
    )
    if not _is_num(mos_to_floor):
        classification = MOS_UNKNOWN
    elif float(mos_to_floor) >= 0.50:
        classification = MOS_DEEP_VALUE_SUPPORT
    elif float(mos_to_floor) >= 0.25:
        classification = MOS_ADEQUATE
    elif float(mos_to_floor) >= 0.10:
        classification = MOS_MODEST
    else:
        classification = MOS_NONE
    claims = {
        "mos_to_floor": _claim(
            value=mos_to_floor,
            refs=[str(ref) for ref in price_refs if str(ref).strip()] + ["intrinsic_discipline.claims.intrinsic_floor.value"],
            reason_code=MOS_UNKNOWN,
        ),
        "mos_to_base": _claim(
            value=mos_to_base,
            refs=[str(ref) for ref in price_refs if str(ref).strip()] + ["intrinsic_discipline.claims.intrinsic_base.value"],
            reason_code=MOS_UNKNOWN,
        ),
    }
    return {
        "mos_to_floor": claims["mos_to_floor"].get("value", UNKNOWN),
        "mos_to_base": claims["mos_to_base"].get("value", UNKNOWN),
        "mos_classification": classification,
        "claims": claims,
    }


def _downside_support(
    *,
    netnet_per_share: Any,
    epv_per_share: Any,
    intrinsic_floor: Any,
    net_debt_value: Any,
) -> dict[str, Any]:
    if _is_num(netnet_per_share) and float(netnet_per_share) > 0.0:
        support_type = SUPPORT_ASSET
        support_status = OK
        reason_codes = [SUPPORT_ASSET]
    elif _is_num(epv_per_share) and float(epv_per_share) > 0.0:
        support_type = SUPPORT_EARNINGS
        support_status = OK
        reason_codes = [SUPPORT_EARNINGS]
    elif _is_num(net_debt_value) and float(net_debt_value) <= 0.0:
        support_type = SUPPORT_BALANCE_SHEET
        support_status = OK
        reason_codes = [SUPPORT_BALANCE_SHEET]
    elif _is_num(intrinsic_floor) and float(intrinsic_floor) > 0.0:
        support_type = SUPPORT_LIMITED
        support_status = LOW_CONFIDENCE
        reason_codes = [SUPPORT_LIMITED]
    else:
        support_type = SUPPORT_UNKNOWN
        support_status = UNKNOWN
        reason_codes = [SUPPORT_UNKNOWN]
    return {
        "downside_support_type": support_type,
        "downside_support_status": support_status,
        "downside_support_reason_codes": reason_codes,
    }


def _build_intrinsic_payload(
    *,
    ticker: str,
    as_of_date: str,
    owner_payload: dict[str, Any] | None,
    owner_quality_payload: dict[str, Any] | None,
    intangible_payload: dict[str, Any] | None,
    maintenance_capex_payload: dict[str, Any] | None,
    series_map: dict[str, list[dict[str, Any]]],
    price_value: Any,
    price_refs: list[Any],
    shares_value: Any,
    shares_refs: list[Any],
    net_debt_value: Any,
    net_debt_refs: list[Any],
    epv_per_share: Any,
    epv_refs: list[Any],
    netnet_per_share: Any,
    netnet_refs: list[Any],
    existing_intrinsic_base: Any,
    existing_intrinsic_base_refs: list[Any],
    existing_intrinsic_conservative: Any,
    existing_intrinsic_conservative_refs: list[Any],
    existing_intrinsic_ceiling: Any,
    existing_intrinsic_ceiling_refs: list[Any],
) -> dict[str, Any]:
    normalized = _normalized_earnings_power(
        owner_payload=owner_payload,
        series_map=series_map,
        owner_quality_payload=owner_quality_payload,
        intangible_payload=intangible_payload,
        maintenance_capex_payload=maintenance_capex_payload,
    )
    valuation_range = _valuation_range(
        normalized_payload=normalized,
        shares_value=shares_value,
        shares_refs=shares_refs,
        net_debt_value=net_debt_value,
        net_debt_refs=net_debt_refs,
        price_value=price_value,
        price_refs=price_refs,
        epv_per_share=epv_per_share,
        epv_refs=epv_refs,
        netnet_per_share=netnet_per_share,
        netnet_refs=netnet_refs,
        existing_intrinsic_base=existing_intrinsic_base,
        existing_intrinsic_base_refs=existing_intrinsic_base_refs,
        existing_intrinsic_conservative=existing_intrinsic_conservative,
        existing_intrinsic_conservative_refs=existing_intrinsic_conservative_refs,
        existing_intrinsic_ceiling=existing_intrinsic_ceiling,
        existing_intrinsic_ceiling_refs=existing_intrinsic_ceiling_refs,
    )
    margin = _margin_of_safety(
        intrinsic_floor=valuation_range.get("intrinsic_floor", UNKNOWN),
        intrinsic_base=valuation_range.get("intrinsic_base", UNKNOWN),
        price_value=price_value,
        price_refs=price_refs,
    )
    downside = _downside_support(
        netnet_per_share=netnet_per_share,
        epv_per_share=epv_per_share,
        intrinsic_floor=valuation_range.get("intrinsic_floor", UNKNOWN),
        net_debt_value=net_debt_value,
    )
    claims = {
        **(normalized.get("claims") if isinstance(normalized.get("claims"), dict) else {}),
        **(valuation_range.get("claims") if isinstance(valuation_range.get("claims"), dict) else {}),
        **(margin.get("claims") if isinstance(margin.get("claims"), dict) else {}),
    }
    derived = _dedupe_refs(
        list(normalized.get("normalized_earnings_power_derived_from") or [])
        + list(valuation_range.get("valuation_range_derived_from") or [])
        + [str(ref) for ref in ((maintenance_capex_payload or {}).get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in price_refs if str(ref).strip()]
        + [str(ref) for ref in shares_refs if str(ref).strip()]
        + [str(ref) for ref in net_debt_refs if str(ref).strip()]
        + [str(ref) for ref in epv_refs if str(ref).strip()]
        + [str(ref) for ref in netnet_refs if str(ref).strip()]
    )
    return {
        "ticker": str(ticker or "").strip().upper(),
        "as_of_date": str(as_of_date or ""),
        "normalized_earnings_power_value": normalized.get("normalized_earnings_power_value", UNKNOWN),
        "normalized_earnings_power_method_used": normalized.get("normalized_earnings_power_method_used", UNKNOWN),
        "normalized_earnings_power_status": normalized.get("normalized_earnings_power_status", UNKNOWN),
        "normalized_earnings_power_reason_codes": [
            str(code)
            for code in (normalized.get("normalized_earnings_power_reason_codes") or [])
            if str(code).strip()
        ],
        "normalized_earnings_power_derived_from": list(normalized.get("normalized_earnings_power_derived_from") or []),
        "intrinsic_floor": valuation_range.get("intrinsic_floor", UNKNOWN),
        "intrinsic_base": valuation_range.get("intrinsic_base", UNKNOWN),
        "intrinsic_ceiling": valuation_range.get("intrinsic_ceiling", UNKNOWN),
        "valuation_range_status": valuation_range.get("valuation_range_status", UNKNOWN),
        "valuation_range_reason_codes": [
            str(code)
            for code in (valuation_range.get("valuation_range_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_range_derived_from": list(valuation_range.get("valuation_range_derived_from") or []),
        "mos_to_floor": margin.get("mos_to_floor", UNKNOWN),
        "mos_to_base": margin.get("mos_to_base", UNKNOWN),
        "mos_classification": margin.get("mos_classification", MOS_UNKNOWN),
        "downside_support_type": downside.get("downside_support_type", SUPPORT_UNKNOWN),
        "downside_support_status": downside.get("downside_support_status", UNKNOWN),
        "downside_support_reason_codes": [
            str(code)
            for code in (downside.get("downside_support_reason_codes") or [])
            if str(code).strip()
        ],
        "claims": claims,
        "derived_from": derived,
        "generated_at": utc_now_iso(),
    }


def compute_intrinsic_discipline(
    ticker: str,
    as_of_date: str,
    *,
    run_id: str | None = None,
    facts_row: dict[str, Any] | None = None,
    owner_payload: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    maintenance_capex_payload: dict[str, Any] | None = None,
    fundamentals: dict[str, Any] | None = None,
    cyclical_normalization_payload: dict[str, Any] | None = None,
    price_value: Any = UNKNOWN,
    price_refs: list[Any] | None = None,
    shares_value: Any = UNKNOWN,
    shares_refs: list[Any] | None = None,
    net_debt_value: Any = UNKNOWN,
    net_debt_refs: list[Any] | None = None,
    epv_per_share: Any = UNKNOWN,
    epv_refs: list[Any] | None = None,
    netnet_per_share: Any = UNKNOWN,
    netnet_refs: list[Any] | None = None,
    existing_intrinsic_base: Any = UNKNOWN,
    existing_intrinsic_base_refs: list[Any] | None = None,
    existing_intrinsic_conservative: Any = UNKNOWN,
    existing_intrinsic_conservative_refs: list[Any] | None = None,
    existing_intrinsic_ceiling: Any = UNKNOWN,
    existing_intrinsic_ceiling_refs: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    price_refs = list(price_refs or [])
    shares_refs = list(shares_refs or [])
    net_debt_refs = list(net_debt_refs or [])
    epv_refs = list(epv_refs or [])
    netnet_refs = list(netnet_refs or [])
    existing_intrinsic_base_refs = list(existing_intrinsic_base_refs or [])
    existing_intrinsic_conservative_refs = list(existing_intrinsic_conservative_refs or [])
    existing_intrinsic_ceiling_refs = list(existing_intrinsic_ceiling_refs or [])

    if isinstance(fundamentals, dict) and fundamentals:
        series_map = _build_series_from_fundamentals(fundamentals)
        if not _is_num(shares_value):
            shares_value, shares_refs = _latest_numeric(series_map.get("shares") or [])
        if not _is_num(net_debt_value):
            net_debt_value, net_debt_refs = _latest_numeric(series_map.get("net_debt") or [])
        if not isinstance(owner_payload, dict):
            owner_payload = {}
    else:
        base_facts_row = facts_row if isinstance(facts_row, dict) else {}
        # On the facts path an explicit share count comes from a facts row or a
        # scout score row, both in millions, while every earnings series here is
        # whole dollars: convert to whole shares before any per-share division.
        if _is_num(shares_value):
            shares_value = float(shares_value) * _MILLION
        if not isinstance(owner_payload, dict):
            owner_payload = compute_owner_earnings_series(
                ticker=ticker_norm,
                as_of_date=as_of_date,
                years_back=5,
                run_id=run_id,
                facts_row=base_facts_row if base_facts_row else None,
                maintenance_capex_ratio=DEFAULT_MAINT_CAPEX_RATIO,
                cfg=cfg,
            )
        series_map = _build_series_from_facts(
            as_of_date=as_of_date,
            facts_row=base_facts_row,
            owner_payload=owner_payload,
            net_debt_value=net_debt_value,
            net_debt_refs=net_debt_refs,
        )
        if not _is_num(shares_value):
            shares_value, shares_refs = _latest_numeric(series_map.get("shares") or [])
        if not _is_num(net_debt_value):
            net_debt_value, net_debt_refs = _latest_numeric(series_map.get("net_debt") or [])

    result = _build_intrinsic_payload(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        owner_payload=owner_payload,
        owner_quality_payload=owner_quality_payload,
        intangible_payload=intangible_payload,
        maintenance_capex_payload=maintenance_capex_payload,
        series_map=series_map,
        price_value=price_value,
        price_refs=price_refs,
        shares_value=shares_value,
        shares_refs=shares_refs,
        net_debt_value=net_debt_value,
        net_debt_refs=net_debt_refs,
        epv_per_share=epv_per_share,
        epv_refs=epv_refs,
        netnet_per_share=netnet_per_share,
        netnet_refs=netnet_refs,
        existing_intrinsic_base=existing_intrinsic_base,
        existing_intrinsic_base_refs=existing_intrinsic_base_refs,
        existing_intrinsic_conservative=existing_intrinsic_conservative,
        existing_intrinsic_conservative_refs=existing_intrinsic_conservative_refs,
        existing_intrinsic_ceiling=existing_intrinsic_ceiling,
        existing_intrinsic_ceiling_refs=existing_intrinsic_ceiling_refs,
    )
    # Attach cycle awareness from cyclical_normalization_payload (optional)
    cycle_payload = cyclical_normalization_payload if isinstance(cyclical_normalization_payload, dict) else {}
    cyclical_profile_class = str(cycle_payload.get("cyclical_profile_class") or "CYCLICALITY_UNKNOWN")
    cyclical_valuation_risk_class = str(cycle_payload.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN")
    result["intrinsic_cycle_awareness_status"] = f"{cyclical_profile_class}/{cyclical_valuation_risk_class}"
    result["cyclical_normalization_detail"] = cycle_payload if cycle_payload else {}
    return result


def write_intrinsic_discipline_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    facts_rows_by_ticker: dict[str, dict[str, Any]] | None = None,
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    owner_earnings_quality_by_ticker: dict[str, dict[str, Any]] | None = None,
    intangible_economics_by_ticker: dict[str, dict[str, Any]] | None = None,
    maintenance_capex_by_ticker: dict[str, dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("intrinsic_discipline_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("intrinsic_discipline_detail"), dict)
    }
    owner_quality_lookup = dict(owner_earnings_quality_by_ticker or {})
    intangible_lookup = dict(intangible_economics_by_ticker or {})
    rows: list[dict[str, Any]] = []

    for ticker in sorted({str(token or "").strip().upper() for token in tickers if str(token or "").strip()}):
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
            metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
            inputs_used = score_row.get("inputs_used") if isinstance(score_row.get("inputs_used"), dict) else {}
            gd = score_row.get("graham_dodd_detail") if isinstance(score_row.get("graham_dodd_detail"), dict) else {}
            owner_quality_payload = owner_quality_lookup.get(ticker)
            if not isinstance(owner_quality_payload, dict) and isinstance(score_row.get("owner_earnings_quality_detail"), dict):
                owner_quality_payload = score_row.get("owner_earnings_quality_detail")
            intangible_payload = intangible_lookup.get(ticker)
            if not isinstance(intangible_payload, dict) and isinstance(score_row.get("intangible_economics_detail"), dict):
                intangible_payload = score_row.get("intangible_economics_detail")
            maintenance_capex_payload = (maintenance_capex_by_ticker or {}).get(ticker)
            if not isinstance(maintenance_capex_payload, dict) and isinstance(
                score_row.get("maintenance_capex_discipline_detail"),
                dict,
            ):
                maintenance_capex_payload = score_row.get("maintenance_capex_discipline_detail")
            detail = compute_intrinsic_discipline(
                ticker=ticker,
                as_of_date=as_of_date,
                run_id=run_id,
                facts_row=(facts_rows_by_ticker or {}).get(ticker),
                owner_payload=score_row.get("owner_earnings_detail")
                if isinstance(score_row.get("owner_earnings_detail"), dict)
                else score_row.get("owner_payload")
                if isinstance(score_row.get("owner_payload"), dict)
                else None,
                owner_quality_payload=owner_quality_payload if isinstance(owner_quality_payload, dict) else None,
                intangible_payload=intangible_payload if isinstance(intangible_payload, dict) else None,
                maintenance_capex_payload=maintenance_capex_payload
                if isinstance(maintenance_capex_payload, dict)
                else None,
                fundamentals=(fundamentals_by_ticker or {}).get(ticker),
                price_value=metric_values.get("current_price", (inputs_used.get("current_price") or {}).get("value", UNKNOWN)),
                price_refs=list((inputs_used.get("current_price") or {}).get("derived_from") or []),
                shares_value=metric_values.get("shares_outstanding", (inputs_used.get("shares_outstanding") or {}).get("value", UNKNOWN)),
                shares_refs=list((inputs_used.get("shares_outstanding") or {}).get("derived_from") or []),
                net_debt_value=metric_values.get("net_debt_proxy", (inputs_used.get("net_debt_proxy") or {}).get("value", UNKNOWN)),
                net_debt_refs=list((inputs_used.get("net_debt_proxy") or {}).get("derived_from") or []),
                epv_per_share=gd.get("epv_per_share", metric_values.get("epv_per_share", UNKNOWN)),
                epv_refs=list(((gd.get("claims") or {}).get("epv_per_share") or {}).get("derived_from") or []),
                netnet_per_share=gd.get("netnet_per_share", metric_values.get("netnet_per_share", UNKNOWN)),
                netnet_refs=list(((gd.get("claims") or {}).get("netnet_per_share") or {}).get("derived_from") or []),
                existing_intrinsic_base=metric_values.get("intrinsic_per_share_base", metric_values.get("intrinsic_per_share_proxy", UNKNOWN)),
                existing_intrinsic_base_refs=list((score_row.get("derived_from") or [])),
                existing_intrinsic_conservative=metric_values.get("intrinsic_per_share_conservative", UNKNOWN),
                existing_intrinsic_conservative_refs=list((score_row.get("derived_from") or [])),
                existing_intrinsic_ceiling=metric_values.get("intrinsic_per_share_proxy", metric_values.get("intrinsic_per_share_base", UNKNOWN)),
                existing_intrinsic_ceiling_refs=list((score_row.get("derived_from") or [])),
                cfg=cfg,
            )
        rows.append(detail)

    known_count = len([row for row in rows if _is_num(row.get("intrinsic_base"))])
    unknown_count = len(rows) - known_count
    top_by_mos = sorted(
        rows,
        key=lambda row: (
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            str(row.get("ticker") or ""),
        ),
    )
    margin_rows = [
        row
        for row in rows
        if str(row.get("mos_classification") or "") in {MOS_DEEP_VALUE_SUPPORT, MOS_ADEQUATE}
    ]
    negative_reason_counts: dict[str, int] = {}
    downside_support_counts: dict[str, int] = {}
    for row in rows:
        for code in (
            list(row.get("normalized_earnings_power_reason_codes") or [])
            + list(row.get("valuation_range_reason_codes") or [])
            + list(row.get("downside_support_reason_codes") or [])
        ):
            token = str(code or "").strip()
            if not token or token in {REASON_OWNER_EARNINGS_SELECTED, REASON_FCF_SELECTED, REASON_CFO_PROXY_SELECTED, SUPPORT_ASSET, SUPPORT_EARNINGS, SUPPORT_BALANCE_SHEET}:
                continue
            negative_reason_counts[token] = negative_reason_counts.get(token, 0) + 1
        support_type = str(row.get("downside_support_type") or SUPPORT_UNKNOWN)
        downside_support_counts[support_type] = downside_support_counts.get(support_type, 0) + 1
    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "known_count": known_count,
        "unknown_count": unknown_count,
        "top_10_by_mos_to_floor": [
            {
                "ticker": str(row.get("ticker") or ""),
                "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
                "mos_to_base": row.get("mos_to_base", UNKNOWN),
                "mos_classification": str(row.get("mos_classification") or MOS_UNKNOWN),
                "intrinsic_floor": row.get("intrinsic_floor", UNKNOWN),
                "intrinsic_base": row.get("intrinsic_base", UNKNOWN),
                "downside_support_type": str(row.get("downside_support_type") or SUPPORT_UNKNOWN),
            }
            for row in top_by_mos[:10]
        ],
        "top_10_with_margin_of_safety": [
            {
                "ticker": str(row.get("ticker") or ""),
                "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
                "mos_classification": str(row.get("mos_classification") or MOS_UNKNOWN),
                "downside_support_type": str(row.get("downside_support_type") or SUPPORT_UNKNOWN),
            }
            for row in sorted(
                margin_rows,
                key=lambda row: (
                    -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
                    str(row.get("ticker") or ""),
                ),
            )[:10]
        ],
        "negative_reason_counts": dict(sorted(negative_reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))),
        "downside_support_counts": dict(sorted(downside_support_counts.items(), key=lambda item: (str(item[0])))),
        "limited_support_count": int(downside_support_counts.get(SUPPORT_LIMITED, 0)),
        "unknown_support_count": int(downside_support_counts.get(SUPPORT_UNKNOWN, 0)),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["intrinsic_discipline_path"] = str(output_path)
    return payload


def _intrinsic_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "intrinsic_discipline.json",
        cfg.sectors_dir / run_id / "intrinsic_discipline.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_intrinsic_discipline(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _intrinsic_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "intrinsic_discipline_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or len(rows)),
        "known_count": int(payload.get("known_count") or 0),
        "unknown_count": int(payload.get("unknown_count") or 0),
        "top_10_by_mos_to_floor": [
            row for row in (payload.get("top_10_by_mos_to_floor") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_with_margin_of_safety": [
            row for row in (payload.get("top_10_with_margin_of_safety") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "negative_reason_counts": payload.get("negative_reason_counts")
        if isinstance(payload.get("negative_reason_counts"), dict)
        else {},
        "downside_support_counts": payload.get("downside_support_counts")
        if isinstance(payload.get("downside_support_counts"), dict)
        else {},
        "limited_support_count": int(payload.get("limited_support_count") or 0),
        "unknown_support_count": int(payload.get("unknown_support_count") or 0),
        "intrinsic_discipline_path": str(path),
    }
