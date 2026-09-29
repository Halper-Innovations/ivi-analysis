from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.valuation.adjacent_years import trailing_adjacent_run
from app.valuation.facts import resolve_financial_facts_asof


UNKNOWN = "UNKNOWN"
DEFAULT_MAINT_CAPEX_RATIO = 0.60

# What this module actually computes, stated so no consumer has to infer it.
#
# It is NOT Buffett's owner earnings (net income + D&A + other non-cash charges
# - maintenance capex - incremental working capital). It is a cash-flow proxy:
# reported CFO less a fixed fraction of reported capex. Everything CFO already
# contains is inherited with CFO's own sign convention:
#   * depreciation, amortisation and other non-cash charges are added back;
#   * impairments and write-offs are added back (they are non-cash);
#   * stock-based compensation is ADDED BACK (treated as non-cash, not as an
#     expense) - note app/valuation/valuation_writer.py subtracts SBC instead,
#     so the two definitions disagree and must not be compared;
#   * deferred tax movements are added back;
#   * the ACTUAL change in working capital is deducted with the correct sign
#     (a build in working capital is a use of cash and reduces CFO), but it is
#     the realised change, not the "incremental working capital required";
#   * cash interest paid is DEDUCTED, because US GAAP classifies it as
#     operating. The result is therefore a LEVERED stream. Capitalising it at
#     an enterprise multiple and then subtracting net debt counts the debt
#     twice.
#
# Companyfacts "val" is used unscaled, so this series is in whole dollars while
# app/valuation/facts.py and app/valuation/fcf.py report USD millions. Consumers
# that mix the two are out by a factor of a million.

REASON_MISSING_CFO = "MISSING_CFO"
REASON_MISSING_CAPEX = "MISSING_CAPEX"
REASON_NEGATIVE_CFO = "NEGATIVE_CFO"
REASON_NEGATIVE_OWNER_EARNINGS = "NEGATIVE_OWNER_EARNINGS"
REASON_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
REASON_NON_ADJACENT_YEARS_DROPPED = "NON_ADJACENT_YEARS_DROPPED"
REASON_FACTS_PAYLOAD_MISSING = "FACTS_PAYLOAD_MISSING"
REASON_PERIOD_MISMATCH = "PERIOD_MISMATCH"
REASON_PERIOD_NOT_ANNUAL = "PERIOD_NOT_ANNUAL"

# A duration fact covering this many days or more is an annual period. Filers
# report fiscal years of 52-53 weeks and the occasional transition year, so the
# band is generous at both ends; anything shorter is an interim figure.
_ANNUAL_PERIOD_MIN_DAYS = 330
_ANNUAL_PERIOD_MAX_DAYS = 400
# How far two paired periods may differ in length before they are not the same
# period at all. A full-year cash flow against a nine-month capex is ~90 days
# apart and used to be subtracted as if both covered the year.
_PERIOD_PAIRING_TOLERANCE_DAYS = 45

_CFO_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]
_CAPEX_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
]


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_yyyy_mm_dd(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except Exception:
        return None


def _dedupe_refs(refs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        token = str(ref).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _load_companyfacts_payload(facts_row: dict[str, Any]) -> dict[str, Any]:
    cache_path = str(facts_row.get("cache_path") or "").strip()
    if not cache_path:
        return {}
    payload = _safe_json(Path(cache_path))
    if isinstance(payload.get("companyfacts"), dict):
        return payload.get("companyfacts")  # type: ignore[return-value]
    if isinstance(payload.get("facts"), dict):
        return payload  # type: ignore[return-value]
    return {}


def _facts_node(companyfacts: dict[str, Any], taxonomy: str, tag: str) -> dict[str, Any]:
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    tax = facts.get(taxonomy) if isinstance(facts, dict) else {}
    if not isinstance(tax, dict):
        return {}
    node = tax.get(tag)
    return node if isinstance(node, dict) else {}


def _fact_rows(
    *,
    companyfacts: dict[str, Any],
    taxonomy: str,
    tag: str,
    as_of_date: str,
    expected_unit_exact: tuple[str, ...] = ("usd",),
) -> list[dict[str, Any]]:
    tag_node = _facts_node(companyfacts, taxonomy, tag)
    units = tag_node.get("units") if isinstance(tag_node.get("units"), dict) else {}
    if not isinstance(units, dict):
        return []
    asof_dt = _parse_yyyy_mm_dd(as_of_date)
    if asof_dt is None:
        return []
    expected = {token.lower() for token in expected_unit_exact}
    out: list[dict[str, Any]] = []
    for unit in sorted(units.keys()):
        if expected and str(unit).strip().lower() not in expected:
            continue
        rows = units.get(unit)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("val")
            if not _is_num(value):
                continue
            end_date = str(row.get("end") or "")
            end_dt = _parse_yyyy_mm_dd(end_date)
            if end_dt is None or end_dt > asof_dt:
                continue
            filed = str(row.get("filed") or "")
            filed_dt = _parse_yyyy_mm_dd(filed)
            if filed_dt is None or filed_dt > asof_dt:
                continue
            accn = str(row.get("accn") or "")
            accn_fragment = f",accn={accn}" if accn else ""
            out.append(
                {
                    "value": float(value),
                    "end_date": end_date,
                    "start_date": str(row.get("start") or ""),
                    "filed": filed,
                    "form": str(row.get("form") or ""),
                    "fp": str(row.get("fp") or ""),
                    "unit": str(unit),
                    "ref": (
                        f"companyfacts.{taxonomy}.{tag}"
                        f"[end_date={end_date},filed={filed},unit={unit}{accn_fragment}]"
                    ),
                }
            )
    out.sort(key=lambda item: (str(item.get("end_date") or ""), str(item.get("filed") or "")))
    return out


def _period_days(row: dict[str, Any]) -> int | None:
    """Length of the period a duration fact covers, when the filing states it."""
    start = _parse_yyyy_mm_dd(str(row.get("start_date") or ""))
    end = _parse_yyyy_mm_dd(str(row.get("end_date") or ""))
    if start is None or end is None:
        return None
    days = (end - start).days
    return days if days > 0 else None


def _is_annual_row(row: dict[str, Any]) -> bool:
    """Whether a row covers a full year, by its own dates where it states them."""
    days = _period_days(row)
    if days is not None:
        return _ANNUAL_PERIOD_MIN_DAYS <= days <= _ANNUAL_PERIOD_MAX_DAYS
    fp = str(row.get("fp") or "").strip().upper()
    form = str(row.get("form") or "").strip().upper()
    return fp == "FY" or form in {"10-K", "20-F", "40-F", "10-K/A", "20-F/A", "40-F/A"}


def _is_interim_row(row: dict[str, Any]) -> bool:
    """Whether a row is positively known to cover less (or more) than a year.

    A row that states its own dates is judged by them; one that does not is
    judged by its fiscal-period and form labels. A row with neither carries no
    evidence either way and is not called interim here.
    """
    days = _period_days(row)
    if days is not None:
        return not (_ANNUAL_PERIOD_MIN_DAYS <= days <= _ANNUAL_PERIOD_MAX_DAYS)
    fp = str(row.get("fp") or "").strip().upper()
    form = str(row.get("form") or "").strip().upper()
    return fp in {"Q1", "Q2", "Q3", "Q4", "H1", "H2"} or form in {"10-Q", "10-Q/A"}


def _annual_series_for_priority(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...] = ("usd",),
) -> list[dict[str, Any]]:
    def _annual_preference(row: dict[str, Any]) -> tuple[int, str, str]:
        fp = str(row.get("fp") or "").strip().upper()
        form = str(row.get("form") or "").strip().upper()
        # Preference is by the period the fact actually covers, falling back to
        # the fiscal-period and form labels when the filing states no start
        # date. A 10-K carries interim durations too.
        if _is_annual_row(row):
            return (0, fp, form)
        return (1, fp, form)

    by_year: dict[int, dict[str, Any]] = {}
    for taxonomy, tag in priority:
        rows = _fact_rows(
            companyfacts=companyfacts,
            taxonomy=taxonomy,
            tag=tag,
            as_of_date=as_of_date,
            expected_unit_exact=expected_unit_exact,
        )
        for row in rows:
            end_dt = _parse_yyyy_mm_dd(str(row.get("end_date") or ""))
            year = end_dt.year if end_dt else 0
            if year <= 0:
                continue
            candidate = {**row, "year": year, "taxonomy": taxonomy, "tag": tag}
            existing = by_year.get(year)
            if existing is None:
                by_year[year] = candidate
                continue
            key_candidate = (
                _annual_preference(candidate),
                str(candidate.get("end_date") or ""),
                str(candidate.get("filed") or ""),
                str(candidate.get("taxonomy") or ""),
                str(candidate.get("tag") or ""),
            )
            key_existing = (
                _annual_preference(existing),
                str(existing.get("end_date") or ""),
                str(existing.get("filed") or ""),
                str(existing.get("taxonomy") or ""),
                str(existing.get("tag") or ""),
            )
            if key_candidate < key_existing:
                by_year[year] = candidate
    annual_rows = [row for year, row in sorted(by_year.items()) if _annual_preference(row)[0] == 0]
    return annual_rows if annual_rows else [by_year[year] for year in sorted(by_year.keys())]


def _normalize_recent_values(values: list[float]) -> tuple[float | str, str, int]:
    if not values:
        return UNKNOWN, REASON_INSUFFICIENT_HISTORY, 0
    recent = [float(v) for v in values[-3:]]
    if len(recent) >= 3:
        ordered = sorted(recent)
        return float(ordered[len(ordered) // 2]), "MEDIAN_3Y", len(recent)
    if len(recent) >= 2:
        return float(sum(recent) / float(len(recent))), f"AVG_{len(recent)}Y", len(recent)
    return float(recent[-1]), "LATEST", 1


def compute_owner_earnings_series(
    ticker: str,
    as_of_date: str,
    years_back: int = 5,
    *,
    run_id: str | None = None,
    facts_row: dict[str, Any] | None = None,
    maintenance_capex_ratio: float = DEFAULT_MAINT_CAPEX_RATIO,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    ratio = max(0.0, min(2.0, float(maintenance_capex_ratio)))
    base_facts_row = (
        facts_row
        if isinstance(facts_row, dict)
        else resolve_financial_facts_asof(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            run_id=run_id,
            refresh=False,
            cfg=cfg,
        )
    )
    companyfacts = _load_companyfacts_payload(base_facts_row)
    base_refs = [str(ref) for ref in (base_facts_row.get("derived_from") or []) if str(ref).strip()]
    if not companyfacts:
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "years_back": max(1, int(years_back)),
            "maintenance_capex_ratio": ratio,
            "status": "UNKNOWN",
            "reason_codes": [
                REASON_FACTS_PAYLOAD_MISSING,
                REASON_MISSING_CFO,
                REASON_MISSING_CAPEX,
            ],
            "series": [],
            "summary": {
                "owner_earnings_normalized_3y": UNKNOWN,
                "owner_earnings_normalized_method": REASON_INSUFFICIENT_HISTORY,
                "owner_earnings_latest": UNKNOWN,
                "owner_earnings_points": 0,
            },
            "derived_from": _dedupe_refs(base_refs),
        }

    cfo_series = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CFO_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    capex_series = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CAPEX_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    cfo_by_year = {int(row.get("year", 0)): row for row in cfo_series}
    capex_by_year = {int(row.get("year", 0)): row for row in capex_series}
    years = sorted(set(cfo_by_year.keys()).union(set(capex_by_year.keys())))
    # Only the calendar-adjacent run ending at the latest year: a gap must not
    # let an older year stand in for a recent one in the normalized figure.
    all_years_count = len(years)
    years = trailing_adjacent_run(years, lambda year: year)
    gap_dropped_years = len(years) < all_years_count
    if years_back > 0 and len(years) > int(years_back):
        years = years[-int(years_back) :]

    rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    owner_numeric: list[float] = []
    owner_numeric_years: list[tuple[int, float]] = []
    owner_refs: list[str] = []

    for year in years:
        cfo_row = cfo_by_year.get(year)
        capex_row = capex_by_year.get(year)
        row_reasons: list[str] = []
        cfo_value = cfo_row.get("value") if isinstance(cfo_row, dict) else UNKNOWN
        capex_value = capex_row.get("value") if isinstance(capex_row, dict) else UNKNOWN
        if not _is_num(cfo_value):
            row_reasons.append(REASON_MISSING_CFO)
        if not _is_num(capex_value):
            row_reasons.append(REASON_MISSING_CAPEX)
        # Both sides of the subtraction must cover the same period. Bucketing by
        # calendar year alone let a full-year cash flow pair with a nine-month
        # capex and called the difference a year of owner earnings.
        # Equal lengths are not enough either: an April-March cash flow and a
        # January-December capex are both twelve months and still different
        # years, so the two periods must also END on the same date.
        cfo_days = _period_days(cfo_row) if isinstance(cfo_row, dict) else None
        capex_days = _period_days(capex_row) if isinstance(capex_row, dict) else None
        cfo_end = str((cfo_row or {}).get("end_date") or "")
        capex_end = str((capex_row or {}).get("end_date") or "")
        period_mismatch = (
            cfo_days is not None
            and capex_days is not None
            and abs(cfo_days - capex_days) > _PERIOD_PAIRING_TOLERANCE_DAYS
        ) or (bool(cfo_end) and bool(capex_end) and cfo_end != capex_end)
        if period_mismatch:
            row_reasons.append(REASON_PERIOD_MISMATCH)
        # A matching pair of quarters is still a quarter: this series is a year
        # of owner earnings per row, so an interim period on either side leaves
        # the year unknown rather than publishing a quarter as the year.
        not_annual = any(
            isinstance(side, dict) and _is_interim_row(side) for side in (cfo_row, capex_row)
        )
        if not_annual:
            row_reasons.append(REASON_PERIOD_NOT_ANNUAL)
        unpairable = period_mismatch or not_annual
        if _is_num(cfo_value) and float(cfo_value) <= 0:
            row_reasons.append(REASON_NEGATIVE_CFO)
        # Capex as a MAGNITUDE: a filer that reports the element negated would
        # otherwise turn the deduction below into an addition (owner earnings
        # above CFO). The reported value keeps its filed sign in "capex".
        maint_capex_proxy = (
            abs(float(capex_value)) * ratio
            if _is_num(capex_value) and not unpairable
            else UNKNOWN
        )
        owner_earnings = (
            float(cfo_value) - float(maint_capex_proxy)
            if _is_num(cfo_value) and _is_num(maint_capex_proxy)
            else UNKNOWN
        )
        if _is_num(owner_earnings) and float(owner_earnings) < 0:
            row_reasons.append(REASON_NEGATIVE_OWNER_EARNINGS)
        if _is_num(owner_earnings):
            owner_numeric.append(float(owner_earnings))
            owner_numeric_years.append((int(year), float(owner_earnings)))
        cfo_ref = str((cfo_row or {}).get("ref") or "")
        capex_ref = str((capex_row or {}).get("ref") or "")
        refs = _dedupe_refs(
            [
                cfo_ref,
                capex_ref,
                f"derived:maintenance_capex_proxy=capex*{ratio:.4f}",
                "derived:owner_earnings=cfo-maintenance_capex_proxy",
            ]
        )
        owner_refs.extend(refs)
        reason_codes.extend(row_reasons)
        rows.append(
            {
                "year": int(year),
                "cfo": _to_num(cfo_value),
                "capex": _to_num(capex_value),
                "maintenance_capex_proxy": _to_num(maint_capex_proxy),
                "owner_earnings": _to_num(owner_earnings),
                "proxy_flags": {
                    "maintenance_capex_proxy_applied": bool(_is_num(capex_value)),
                },
                "reason_codes": sorted(set(row_reasons)),
                "derived_from": refs,
            }
        )

    # A year with no owner-earnings figure is a gap too: normalize over the
    # adjacent run ending at the latest known year, never across the hole.
    if gap_dropped_years or len(trailing_adjacent_run(owner_numeric_years)) < len(owner_numeric_years):
        reason_codes.append(REASON_NON_ADJACENT_YEARS_DROPPED)
    norm_value, norm_method, norm_points = _normalize_recent_values(
        [value for _year, value in trailing_adjacent_run(owner_numeric_years)]
    )
    if not _is_num(norm_value):
        reason_codes.append(REASON_INSUFFICIENT_HISTORY)

    latest_owner = UNKNOWN
    for row in reversed(rows):
        value = row.get("owner_earnings")
        if _is_num(value):
            latest_owner = float(value)
            break

    final_reasons = sorted(set(reason_codes))
    status = "OK" if _is_num(norm_value) else "UNKNOWN"
    maintenance_burden_rows = [
        value
        for _year, value in trailing_adjacent_run(
            [
                (
                    int(row["year"]),
                    abs(float(row.get("maintenance_capex_proxy"))) / float(row.get("cfo")),
                )
                for row in rows
                if _is_num(row.get("maintenance_capex_proxy"))
                and _is_num(row.get("cfo"))
                and float(row.get("cfo")) > 0.0
            ]
        )
    ]
    maintenance_burden_median = (
        float(_normalize_recent_values(maintenance_burden_rows)[0])
        if maintenance_burden_rows
        else UNKNOWN
    )
    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "years_back": max(1, int(years_back)),
        "maintenance_capex_ratio": ratio,
        "status": status,
        "reason_codes": final_reasons,
        "series": rows,
        "summary": {
            "owner_earnings_normalized_3y": _to_num(norm_value),
            "owner_earnings_normalized_method": norm_method,
            "owner_earnings_latest": _to_num(latest_owner),
            "owner_earnings_points": int(norm_points),
            "maintenance_capex_proxy_to_cfo_median": _to_num(maintenance_burden_median),
        },
        "derived_from": _dedupe_refs(
            base_refs + owner_refs + [f"facts_coverage.rows[{ticker_norm}]"]
        ),
    }
