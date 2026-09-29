from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.market.company_facts_extract import extract_shares_outstanding_asof
from app.market.shares_guard import GUARD_REFUSED
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.owner_earnings import REASON_INSUFFICIENT_HISTORY, compute_owner_earnings_series


UNKNOWN = "UNKNOWN"
STATUS_OK = "OK"
STATUS_UNKNOWN = "UNKNOWN"

REASON_OK = "OK"
REASON_MISSING_EARNINGS_STREAM = "MISSING_EARNINGS_STREAM"
REASON_MISSING_SHARES = "MISSING_SHARES"
# Facts rows (app.valuation.facts) report shares in millions.
_SHARES_PER_MILLION = 1_000_000.0
REASON_INVALID_DISCOUNT_RATE = "INVALID_DISCOUNT_RATE"
REASON_NEGATIVE_EARNINGS_STREAM = "NEGATIVE_EARNINGS_STREAM"
REASON_MISSING_CURRENT_ASSETS = "MISSING_CURRENT_ASSETS"
REASON_MISSING_TOTAL_LIABILITIES = "MISSING_TOTAL_LIABILITIES"
REASON_PRICE_UNKNOWN = "PRICE_UNKNOWN"
REASON_FACTS_PAYLOAD_MISSING = "FACTS_PAYLOAD_MISSING"
REASON_PREFERRED_STOCK_ASSUMED_ZERO = "PREFERRED_STOCK_ASSUMED_ZERO"

_FCF_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "FreeCashFlow"),
]
_CFO_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]
_CAPEX_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
]
_CURRENT_ASSETS_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "AssetsCurrent"),
    ("us-gaap", "CurrentAssets"),
]
_TOTAL_LIABILITIES_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "Liabilities"),
]
_LIABILITIES_COMPONENT_TAGS: list[tuple[str, str]] = [
    ("us-gaap", "LiabilitiesCurrent"),
    ("us-gaap", "LiabilitiesNoncurrent"),
]
# Forms whose companyfacts rows carry a full fiscal year. Mirrors
# owner_earnings._annual_series_for_priority so the two modules agree on what
# "the annual figure" is.
_ANNUAL_FORMS: frozenset[str] = frozenset({"10-K", "20-F", "40-F", "10-K/A", "20-F/A", "40-F/A"})
# Normalization looks back this many CONSECUTIVE fiscal years from the latest
# year present — not "the last N values that happen to exist".
_NORMALIZATION_WINDOW_YEARS = 3

_PREFERRED_STOCK_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "PreferredStockValue"),
    ("us-gaap", "PreferredStockCarryingAmount"),
    ("us-gaap", "RedeemablePreferredStockCarryingAmount"),
]


def _is_num(value: Any) -> bool:
    # NaN / +-inf are NOT numbers here. json.loads accepts the bare NaN and
    # Infinity tokens, so a corrupt companyfacts cache can carry one; it then
    # slips past every "> 0" guard below and publishes epv_status=OK with a NaN
    # epv_value (and a payload json.dumps(allow_nan=False) cannot serialize).
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return True
    return math.isfinite(value)


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


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
    expected_unit_exact: tuple[str, ...],
) -> list[dict[str, Any]]:
    tag_node = _facts_node(companyfacts, taxonomy, tag)
    units = tag_node.get("units") if isinstance(tag_node.get("units"), dict) else {}
    if not isinstance(units, dict):
        return []
    asof_dt = _parse_yyyy_mm_dd(as_of_date)
    if asof_dt is None:
        return []
    expected = {token.lower() for token in expected_unit_exact}
    rows_out: list[dict[str, Any]] = []
    for unit in sorted(units.keys()):
        unit_norm = str(unit).strip().lower()
        if expected and unit_norm not in expected:
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
            rows_out.append(
                {
                    "value": float(value),
                    "end_date": end_date,
                    "filed": filed,
                    "form": str(row.get("form") or ""),
                    "fp": str(row.get("fp") or ""),
                    "unit": str(unit),
                    "taxonomy": taxonomy,
                    "tag": tag,
                    "ref": (
                        f"companyfacts.{taxonomy}.{tag}"
                        f"[end_date={end_date},filed={filed},unit={unit}{accn_fragment}]"
                    ),
                }
            )
    rows_out.sort(key=lambda item: (str(item.get("end_date") or ""), str(item.get("filed") or "")))
    return rows_out


def _latest_fact(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for taxonomy, tag in priority:
        rows = _fact_rows(
            companyfacts=companyfacts,
            taxonomy=taxonomy,
            tag=tag,
            as_of_date=as_of_date,
            expected_unit_exact=expected_unit_exact,
        )
        if rows:
            candidates.append(rows[-1])
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            str(item.get("end_date") or ""),
            str(item.get("filed") or ""),
            str(item.get("taxonomy") or ""),
            str(item.get("tag") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def _is_annual_row(row: dict[str, Any]) -> bool:
    fp = str(row.get("fp") or "").strip().upper()
    form = str(row.get("form") or "").strip().upper()
    return fp == "FY" or form in _ANNUAL_FORMS


def _annual_series(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...],
) -> list[dict[str, Any]]:
    """One row per fiscal year, preferring rows that actually cover a full year.

    Rows are bucketed by the CALENDAR year of `end`. A filer whose fiscal year
    ends mid-year files 10-Q rows with later `end` dates inside that same
    calendar year, so a plain latest-`end` contest hands back a 3- or 6-month
    interim figure as "the year" (audit: interim-ytd-as-annual). Annual rows win
    the contest, and if any annual row exists the interim-only years are dropped
    rather than filled with a partial period.
    """

    def _contest_key(row: dict[str, Any]) -> tuple[int, str, str, str, str]:
        return (
            1 if _is_annual_row(row) else 0,
            str(row.get("end_date") or ""),
            str(row.get("filed") or ""),
            str(row.get("taxonomy") or ""),
            str(row.get("tag") or ""),
        )

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
            year_dt = _parse_yyyy_mm_dd(str(row.get("end_date") or ""))
            year = year_dt.year if year_dt else 0
            if year <= 0:
                continue
            candidate = {**row, "year": year}
            existing = by_year.get(year)
            if existing is None:
                by_year[year] = candidate
                continue
            if _contest_key(candidate) > _contest_key(existing):
                by_year[year] = candidate
    ordered = [by_year[year] for year in sorted(by_year.keys())]
    annual_only = [row for row in ordered if _is_annual_row(row)]
    return annual_only if annual_only else ordered


def _recent_window_values(pairs: list[tuple[int, float]]) -> list[float]:
    """Values whose fiscal year falls inside the last N consecutive years.

    `_annual_series` yields only the years a filer actually tagged, so slicing
    the last three VALUES silently splices non-adjacent years together and still
    labels the result MEDIAN_3Y (audit: normalization-window-not-contiguous).
    Anchoring on the latest year present makes a gap shrink the sample — and the
    method label that reports it — instead of importing a stale year.
    """
    if not pairs:
        return []
    latest_year = max(int(year) for year, _value in pairs)
    floor_year = latest_year - (_NORMALIZATION_WINDOW_YEARS - 1)
    return [
        float(value)
        for year, value in sorted(pairs, key=lambda item: int(item[0]))
        if floor_year <= int(year) <= latest_year
    ]


def _normalize_recent(values: list[float]) -> tuple[float | str, str, int]:
    if not values:
        return UNKNOWN, REASON_INSUFFICIENT_HISTORY, 0
    recent = [float(v) for v in values[-3:]]
    if len(recent) >= 3:
        ordered = sorted(recent)
        return float(ordered[len(ordered) // 2]), "MEDIAN_3Y", len(recent)
    if len(recent) >= 2:
        return float(sum(recent) / float(len(recent))), f"AVG_{len(recent)}Y", len(recent)
    return float(recent[-1]), "LATEST", 1


def _resolve_shares(
    *,
    facts_row: dict[str, Any],
    companyfacts: dict[str, Any],
    as_of_date: str,
) -> tuple[float | str, list[str], str]:
    facts_refs = [str(ref) for ref in (facts_row.get("derived_from") or []) if str(ref).strip()]
    shares_value = facts_row.get("shares_value", UNKNOWN)
    shares_status = str(facts_row.get("shares_status") or "UNKNOWN").upper()
    if shares_status == "OK" and _is_num(shares_value) and float(shares_value) > 0:
        # The facts row carries shares in MILLIONS (``shares_output_unit ==
        # "shares_millions"``, app.valuation.facts), while earnings and balances here
        # are whole dollars: dividing by the raw row value made every per-share
        # figure a million times too high. The companyfacts fallback below already
        # returns whole shares.
        return float(shares_value) * _SHARES_PER_MILLION, facts_refs, REASON_OK
    # The facts row's count went through the share-count guard; a count it refused is not
    # re-picked here by a second, unguarded chooser. The fallback below is the same guarded
    # chooser, for a caller whose facts row did not resolve the payload.
    facts_guard = facts_row.get("shares_guard")
    if isinstance(facts_guard, dict) and facts_guard.get("outcome") == GUARD_REFUSED:
        return UNKNOWN, facts_refs, str(facts_guard.get("reason_code") or REASON_MISSING_SHARES)
    share_fact, share_guard = extract_shares_outstanding_asof(companyfacts, as_of_date)
    if (
        isinstance(share_fact, dict)
        and _is_num(share_fact.get("value"))
        and float(share_fact.get("value")) > 0
    ):
        refs = _dedupe_refs(facts_refs + list(share_fact.get("derived_from") or []))
        return float(share_fact["value"]), refs, REASON_OK
    if share_guard.get("outcome") == GUARD_REFUSED:
        return UNKNOWN, facts_refs, str(share_guard.get("reason_code") or REASON_MISSING_SHARES)
    return UNKNOWN, facts_refs, REASON_MISSING_SHARES


def _resolve_fcf_normalized(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
) -> tuple[float | str, str, list[str]]:
    direct = _annual_series(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_FCF_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    cfo = _annual_series(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CFO_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    capex = _annual_series(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CAPEX_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    refs: list[str] = [str(row.get("ref") or "") for row in direct]
    direct_pairs: list[tuple[int, float]] = [
        (int(row.get("year", 0)), float(row.get("value")))
        for row in direct
        if _is_num(row.get("value")) and int(row.get("year", 0)) > 0
    ]
    values: list[float] = _recent_window_values(direct_pairs)
    if values:
        normalized, method, _points = _normalize_recent(values)
        return normalized, f"FCF_{method}", _dedupe_refs(refs)

    cfo_by_year = {int(row.get("year", 0)): row for row in cfo}
    capex_by_year = {int(row.get("year", 0)): row for row in capex}
    derived_pairs: list[tuple[int, float]] = []
    for year in sorted(set(cfo_by_year.keys()).intersection(set(capex_by_year.keys()))):
        cfo_row = cfo_by_year[year]
        capex_row = capex_by_year[year]
        if not (_is_num(cfo_row.get("value")) and _is_num(capex_row.get("value"))):
            continue
        derived_pairs.append((int(year), float(cfo_row["value"]) - float(capex_row["value"])))
        refs.extend(
            [
                str(cfo_row.get("ref") or ""),
                str(capex_row.get("ref") or ""),
                "derived:fcf=cfo-capex",
            ]
        )
    normalized, method, _points = _normalize_recent(_recent_window_values(derived_pairs))
    return normalized, f"FCF_{method}", _dedupe_refs(refs)


def _format_input(value: Any, refs: list[str], *, reason_code: str | None = None) -> dict[str, Any]:
    payload = {
        "value": _to_num(value),
        "derived_from": _dedupe_refs([str(ref) for ref in refs if str(ref).strip()]),
    }
    if reason_code:
        payload["reason_code"] = str(reason_code)
    return payload


def compute_graham_dodd_overlay(
    *,
    ticker: str,
    as_of_date: str,
    price_value: float | str,
    price_status: str,
    discount_rate: float,
    run_id: str | None = None,
    facts_row: dict[str, Any] | None = None,
    owner_payload: dict[str, Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    facts_payload = (
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
    owner = (
        owner_payload
        if isinstance(owner_payload, dict)
        else compute_owner_earnings_series(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            years_back=5,
            run_id=run_id,
            cfg=cfg,
        )
    )
    companyfacts = _load_companyfacts_payload(facts_payload)
    base_refs = [str(ref) for ref in (facts_payload.get("derived_from") or []) if str(ref).strip()]
    owner_refs = [str(ref) for ref in (owner.get("derived_from") or []) if str(ref).strip()]

    shares_value, shares_refs, shares_reason = _resolve_shares(
        facts_row=facts_payload,
        companyfacts=companyfacts,
        as_of_date=as_of_date,
    )

    discount = float(discount_rate) if _is_num(discount_rate) else 0.0
    valid_discount = discount > 0.0

    owner_summary = owner.get("summary") if isinstance(owner.get("summary"), dict) else {}
    owner_norm = owner_summary.get("owner_earnings_normalized_3y", UNKNOWN)
    owner_method = str(
        owner_summary.get("owner_earnings_normalized_method") or REASON_INSUFFICIENT_HISTORY
    )
    fcf_norm, fcf_method, fcf_refs = _resolve_fcf_normalized(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
    )

    earnings_stream = owner_norm if _is_num(owner_norm) else fcf_norm
    earnings_source = f"OWNER_EARNINGS_{owner_method}" if _is_num(owner_norm) else fcf_method
    earnings_refs = owner_refs if _is_num(owner_norm) else fcf_refs

    epv_status = STATUS_UNKNOWN
    epv_reason = REASON_MISSING_EARNINGS_STREAM
    epv_value: float | str = UNKNOWN
    epv_per_share: float | str = UNKNOWN
    if not valid_discount:
        epv_reason = REASON_INVALID_DISCOUNT_RATE
    elif not _is_num(earnings_stream):
        epv_reason = REASON_MISSING_EARNINGS_STREAM
    elif float(earnings_stream) < 0:
        epv_reason = REASON_NEGATIVE_EARNINGS_STREAM
    elif not (_is_num(shares_value) and float(shares_value) > 0):
        epv_reason = REASON_MISSING_SHARES
    else:
        epv_value = float(earnings_stream) / discount
        epv_per_share = float(epv_value) / float(shares_value)
        epv_status = STATUS_OK
        epv_reason = REASON_OK

    current_assets_fact = _latest_fact(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CURRENT_ASSETS_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    total_liabilities_fact = _latest_fact(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_TOTAL_LIABILITIES_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    if not isinstance(total_liabilities_fact, dict):
        liab_current = _latest_fact(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=[_LIABILITIES_COMPONENT_TAGS[0]],
            expected_unit_exact=("usd",),
        )
        liab_noncurrent = _latest_fact(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=[_LIABILITIES_COMPONENT_TAGS[1]],
            expected_unit_exact=("usd",),
        )
        if (
            isinstance(liab_current, dict)
            and isinstance(liab_noncurrent, dict)
            and _is_num(liab_current.get("value"))
            and _is_num(liab_noncurrent.get("value"))
        ):
            total_liabilities_fact = {
                "value": float(liab_current["value"]) + float(liab_noncurrent["value"]),
                "ref": "derived:total_liabilities=liabilities_current+liabilities_noncurrent",
                "end_date": max(
                    str(liab_current.get("end_date") or ""),
                    str(liab_noncurrent.get("end_date") or ""),
                ),
                "tag": "LiabilitiesCurrent_plus_LiabilitiesNoncurrent",
            }
    preferred_fact = _latest_fact(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_PREFERRED_STOCK_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )

    current_assets = (
        current_assets_fact.get("value") if isinstance(current_assets_fact, dict) else UNKNOWN
    )
    total_liabilities = (
        total_liabilities_fact.get("value") if isinstance(total_liabilities_fact, dict) else UNKNOWN
    )
    preferred_found = isinstance(preferred_fact, dict) and _is_num(preferred_fact.get("value"))
    # Zero is the right NCAV term when no preferred is outstanding, but the
    # reason code must not report an untagged stake as a read fact.
    preferred_stock = preferred_fact.get("value") if preferred_found else 0.0
    current_assets_refs = (
        [str(current_assets_fact.get("ref") or "")] if isinstance(current_assets_fact, dict) else []
    )
    total_liabilities_refs = (
        [str(total_liabilities_fact.get("ref") or "")]
        if isinstance(total_liabilities_fact, dict)
        else []
    )
    preferred_refs = (
        [str(preferred_fact.get("ref") or "")] if isinstance(preferred_fact, dict) else []
    )

    netnet_status = STATUS_UNKNOWN
    netnet_reason = REASON_MISSING_CURRENT_ASSETS
    netnet_value: float | str = UNKNOWN
    netnet_per_share: float | str = UNKNOWN
    if not _is_num(current_assets):
        netnet_reason = REASON_MISSING_CURRENT_ASSETS
    elif not _is_num(total_liabilities):
        netnet_reason = REASON_MISSING_TOTAL_LIABILITIES
    elif not (_is_num(shares_value) and float(shares_value) > 0):
        netnet_reason = REASON_MISSING_SHARES
    else:
        netnet_value = float(current_assets) - float(total_liabilities) - float(preferred_stock)
        netnet_per_share = float(netnet_value) / float(shares_value)
        netnet_status = STATUS_OK
        netnet_reason = REASON_OK

    # See app/valuation/mos_conventions.py for the canonical convention split.
    # CONVENTION (FIX 6): mos_epv / mos_netnet here use the UPSIDE-RATIO convention,
    #   mos = intrinsic / price - 1
    # i.e. the fractional upside from price to intrinsic value. This is NOT the
    # textbook margin-of-safety (intrinsic - price)/intrinsic used in
    # valuation_writer._compute_pricing_zone and thesis_updater. Despite the shared
    # "margin_of_safety"/"mos_*" naming, the two conventions differ:
    # for intrinsic=150, price=100 this yields 0.50, while textbook yields 0.333...
    mos_epv: float | str = UNKNOWN
    mos_netnet: float | str = UNKNOWN
    mos_epv_reason = REASON_PRICE_UNKNOWN
    mos_netnet_reason = REASON_PRICE_UNKNOWN
    if _is_num(price_value) and float(price_value) > 0:
        if _is_num(epv_per_share):
            mos_epv = (float(epv_per_share) / float(price_value)) - 1.0  # upside-ratio
            mos_epv_reason = REASON_OK
        else:
            mos_epv_reason = epv_reason
        if _is_num(netnet_per_share):
            mos_netnet = (float(netnet_per_share) / float(price_value)) - 1.0  # upside-ratio
            mos_netnet_reason = REASON_OK
        else:
            mos_netnet_reason = netnet_reason

    gd_value_status = (
        STATUS_OK if (epv_status == STATUS_OK or netnet_status == STATUS_OK) else STATUS_UNKNOWN
    )
    reason_order = [
        epv_reason,
        netnet_reason,
        mos_epv_reason if mos_epv_reason != REASON_OK else "",
        mos_netnet_reason if mos_netnet_reason != REASON_OK else "",
        REASON_FACTS_PAYLOAD_MISSING if not companyfacts else "",
    ]
    gd_primary_reason = (
        REASON_OK
        if gd_value_status == STATUS_OK
        else next(
            (
                str(reason).upper()
                for reason in reason_order
                if str(reason).strip() and str(reason).upper() != REASON_OK
            ),
            REASON_FACTS_PAYLOAD_MISSING,
        )
    )

    inputs_used = {
        "current_price": _format_input(
            price_value,
            [f"prices_summary.rows[{ticker_norm}]"],
            reason_code=str(price_status or "").upper(),
        ),
        "shares_outstanding": _format_input(shares_value, shares_refs, reason_code=shares_reason),
        "normalized_cash_earnings": _format_input(
            earnings_stream, earnings_refs, reason_code=earnings_source
        ),
        "current_assets": _format_input(
            current_assets,
            current_assets_refs,
            reason_code=REASON_OK if _is_num(current_assets) else REASON_MISSING_CURRENT_ASSETS,
        ),
        "total_liabilities": _format_input(
            total_liabilities,
            total_liabilities_refs,
            reason_code=REASON_OK
            if _is_num(total_liabilities)
            else REASON_MISSING_TOTAL_LIABILITIES,
        ),
        "preferred_stock": _format_input(
            preferred_stock,
            preferred_refs,
            reason_code=REASON_OK if preferred_found else REASON_PREFERRED_STOCK_ASSUMED_ZERO,
        ),
    }

    derived_from = _dedupe_refs(
        base_refs
        + owner_refs
        + fcf_refs
        + shares_refs
        + current_assets_refs
        + total_liabilities_refs
        + preferred_refs
        + [f"facts_coverage.rows[{ticker_norm}]"]
        + [
            "derived:epv=normalized_cash_earnings/discount_rate",
            "derived:netnet=current_assets-total_liabilities-preferred_stock",
        ]
    )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "epv_status": epv_status,
        "epv_reason_code": epv_reason,
        "epv_value": _to_num(epv_value),
        "epv_per_share": _to_num(epv_per_share),
        "netnet_status": netnet_status,
        "netnet_reason_code": netnet_reason,
        "netnet_value": _to_num(netnet_value),
        "netnet_per_share": _to_num(netnet_per_share),
        "mos_epv": _to_num(mos_epv),
        "mos_netnet": _to_num(mos_netnet),
        "mos_epv_reason_code": mos_epv_reason,
        "mos_netnet_reason_code": mos_netnet_reason,
        "gd_value_status": gd_value_status,
        "gd_primary_reason_code": gd_primary_reason,
        "inputs_used": inputs_used,
        "thresholds_used": {
            "gd_discount_rate": float(discount) if valid_discount else discount_rate,
        },
        "derived_from": derived_from,
        "generated_at": utc_now_iso(),
    }
