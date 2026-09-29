from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.valuation.intrinsic_discipline import (
    MOS_UNKNOWN,
    REASON_BASE_FROM_EPV,
    REASON_FLOOR_FROM_EPV,
    REASON_FLOOR_FROM_NETNET,
    REASON_OWNER_EARNINGS_SELECTED,
    SUPPORT_ASSET,
    SUPPORT_BALANCE_SHEET,
    SUPPORT_EARNINGS,
)
from app.valuation.valuation_confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_UNKNOWN,
    FRAGILITY_HIGH,
    SUPPORT_EPV,
    SUPPORT_NETNET,
    SUPPORT_NORMALIZED,
    SUPPORT_OWNER,
)
from app.valuation.value_type import (
    VALUE_TYPE_ASSET_BACKED,
    VALUE_TYPE_EARNINGS_POWER,
    VALUE_TYPE_QUALITY,
)


UNKNOWN = "UNKNOWN"
OK = "OK"

CONSISTENCY_CONSISTENT = "CONSISTENT"
CONSISTENCY_INCONSISTENT = "INCONSISTENT"
CONSISTENCY_UNKNOWN = "CONSISTENCY_UNKNOWN"

INTEGRITY_OK = "INTEGRITY_OK"
INTEGRITY_WARNING = "INTEGRITY_WARNING"
INTEGRITY_SUSPECT = "INTEGRITY_SUSPECT"
INTEGRITY_UNKNOWN = "INTEGRITY_UNKNOWN"

REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER = "IDENTICAL_INTRINSIC_RANGE_CLUSTER"
REASON_IDENTICAL_NORMALIZED_EARNINGS_CLUSTER = "IDENTICAL_NORMALIZED_EARNINGS_CLUSTER"
REASON_IDENTICAL_MOS_CLUSTER = "IDENTICAL_MOS_CLUSTER"
REASON_IDENTICAL_SUPPORT_BUNDLE_CLUSTER = "IDENTICAL_SUPPORT_BUNDLE_CLUSTER"
REASON_POSSIBLE_DEFAULT_VALUE_REUSE = "POSSIBLE_DEFAULT_VALUE_REUSE"
REASON_POSSIBLE_STALE_ARTIFACT_REUSE = "POSSIBLE_STALE_ARTIFACT_REUSE"
REASON_METHOD_OWNER_EARNINGS_WITHOUT_OWNER_SUPPORT = "METHOD_OWNER_EARNINGS_WITHOUT_OWNER_SUPPORT"
REASON_NETNET_FLOOR_WITHOUT_NETNET_SUPPORT = "NETNET_FLOOR_WITHOUT_NETNET_SUPPORT"
REASON_EPV_FLOOR_WITHOUT_EPV_SUPPORT = "EPV_FLOOR_WITHOUT_EPV_SUPPORT"
REASON_MOS_WITHOUT_PRICE_OR_VALUE_SUPPORT = "MOS_WITHOUT_PRICE_OR_VALUE_SUPPORT"
REASON_HIGH_CONFIDENCE_WITH_HIGH_FRAGILITY = "HIGH_CONFIDENCE_WITH_HIGH_FRAGILITY"
REASON_VALUE_TYPE_SUPPORT_MISMATCH = "VALUE_TYPE_SUPPORT_MISMATCH"
REASON_CONSISTENCY_CONTEXT_INSUFFICIENT = "CONSISTENCY_CONTEXT_INSUFFICIENT"
REASON_INTEGRITY_REDUCED_BY_ISOLATED_PROVENANCE = "INTEGRITY_REDUCED_BY_ISOLATED_PROVENANCE"

_INTEGRITY_ORDER = {
    INTEGRITY_OK: 0,
    INTEGRITY_WARNING: 1,
    INTEGRITY_SUSPECT: 2,
    INTEGRITY_UNKNOWN: 3,
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


def _claim(*, value: Any, refs: list[Any], reason_code: str, status: str | None = None) -> dict[str, Any]:
    derived = _dedupe(refs)
    if _is_num(value):
        return {
            "value": float(value),
            "status": str(status or OK).upper(),
            "reason_code": str(reason_code or OK),
            "derived_from": derived,
        }
    token = str(value or "").strip()
    if token and token.upper() != UNKNOWN:
        return {
            "value": token,
            "status": str(status or OK).upper(),
            "reason_code": str(reason_code or OK),
            "derived_from": derived,
        }
    return {
        "value": UNKNOWN,
        "status": str(status or UNKNOWN).upper(),
        "reason_code": str(reason_code or UNKNOWN),
        "derived_from": derived,
    }


def _claim_refs(payload: dict[str, Any], key: str) -> list[str]:
    claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
    claim = claims.get(key) if isinstance(claims, dict) else {}
    if not isinstance(claim, dict):
        return []
    return [str(ref) for ref in (claim.get("derived_from") or []) if str(ref).strip()]


def _status_token(value: Any, fallback: str = UNKNOWN) -> str:
    return str(value or fallback).strip().upper() or fallback


def _support_types(payload: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (payload.get("valuation_support_types_present") or [])
        if str(value).strip()
    }


def _range_reason_codes(payload: dict[str, Any]) -> set[str]:
    return {
        str(code)
        for code in (payload.get("valuation_range_reason_codes") or [])
        if str(code).strip()
    }


def _provenance_bundle(
    *,
    intrinsic_payload: dict[str, Any],
    valuation_confidence_payload: dict[str, Any],
    value_type_payload: dict[str, Any],
) -> list[str]:
    return _dedupe(
        _claim_refs(intrinsic_payload, "normalized_earnings_power_value")
        + _claim_refs(intrinsic_payload, "intrinsic_floor")
        + _claim_refs(intrinsic_payload, "intrinsic_base")
        + _claim_refs(intrinsic_payload, "intrinsic_ceiling")
        + _claim_refs(valuation_confidence_payload, "valuation_support_count")
        + _claim_refs(valuation_confidence_payload, "valuation_confidence_class")
        + _claim_refs(value_type_payload, "value_type_primary")
        + [str(ref) for ref in (intrinsic_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (valuation_confidence_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (value_type_payload.get("value_type_derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (value_type_payload.get("derived_from") or []) if str(ref).strip()]
    )


def _fingerprint_summary(
    *,
    intrinsic_payload: dict[str, Any],
    valuation_confidence_payload: dict[str, Any],
    value_type_payload: dict[str, Any],
    price_status: Any,
    shares_status: Any,
    fcf_status: Any,
    facts_status: Any,
    valuation_status: Any,
) -> dict[str, Any]:
    support_types = sorted(_support_types(valuation_confidence_payload))
    summary = {
        "price_status": _status_token(price_status),
        "shares_status": _status_token(shares_status),
        "fcf_status": _status_token(fcf_status),
        "facts_status": _status_token(facts_status),
        "valuation_status": _status_token(valuation_status),
        "normalized_earnings_power_method_used": str(
            intrinsic_payload.get("normalized_earnings_power_method_used") or UNKNOWN
        ),
        "normalized_earnings_power_status": _status_token(
            intrinsic_payload.get("normalized_earnings_power_status")
        ),
        "valuation_support_types_present": support_types,
        "downside_support_type": str(intrinsic_payload.get("downside_support_type") or UNKNOWN),
        "value_type_primary": str(value_type_payload.get("value_type_primary") or UNKNOWN),
        "valuation_range_reason_codes": sorted(_range_reason_codes(intrinsic_payload)),
        "valuation_confidence_reason_codes": [
            str(code)
            for code in (valuation_confidence_payload.get("valuation_confidence_reason_codes") or [])
            if str(code).strip()
        ],
        "derived_from_bundle_summary": _provenance_bundle(
            intrinsic_payload=intrinsic_payload,
            valuation_confidence_payload=valuation_confidence_payload,
            value_type_payload=value_type_payload,
        )[:16],
    }
    return summary


def _fingerprint_token(summary: dict[str, Any]) -> str:
    blob = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def compute_valuation_integrity(
    ticker: str,
    as_of_date: str,
    *,
    intrinsic_payload: dict[str, Any] | None = None,
    valuation_confidence_payload: dict[str, Any] | None = None,
    value_type_payload: dict[str, Any] | None = None,
    price_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    fcf_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    valuation_status: Any = UNKNOWN,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg
    intrinsic_payload = intrinsic_payload if isinstance(intrinsic_payload, dict) else {}
    valuation_confidence_payload = (
        valuation_confidence_payload if isinstance(valuation_confidence_payload, dict) else {}
    )
    value_type_payload = value_type_payload if isinstance(value_type_payload, dict) else {}

    ticker_norm = str(ticker or "").strip().upper()
    support_types = _support_types(valuation_confidence_payload)
    support_count = int(valuation_confidence_payload.get("valuation_support_count") or 0)
    confidence_class = _status_token(
        valuation_confidence_payload.get("valuation_confidence_class"),
        CONFIDENCE_UNKNOWN,
    )
    fragility_status = _status_token(
        valuation_confidence_payload.get("valuation_fragility_status"),
        UNKNOWN,
    )
    downside_support_type = str(intrinsic_payload.get("downside_support_type") or UNKNOWN)
    value_type_primary = str(value_type_payload.get("value_type_primary") or UNKNOWN)
    normalized_method = str(intrinsic_payload.get("normalized_earnings_power_method_used") or UNKNOWN)
    mos_classification = str(intrinsic_payload.get("mos_classification") or MOS_UNKNOWN)
    mos_to_floor = intrinsic_payload.get("mos_to_floor", UNKNOWN)
    mos_to_base = intrinsic_payload.get("mos_to_base", UNKNOWN)
    range_reasons = _range_reason_codes(intrinsic_payload)

    consistency_reason_codes: list[str] = []
    if normalized_method == REASON_OWNER_EARNINGS_SELECTED and SUPPORT_OWNER not in support_types:
        consistency_reason_codes.append(REASON_METHOD_OWNER_EARNINGS_WITHOUT_OWNER_SUPPORT)
    if REASON_FLOOR_FROM_NETNET in range_reasons and SUPPORT_NETNET not in support_types:
        consistency_reason_codes.append(REASON_NETNET_FLOOR_WITHOUT_NETNET_SUPPORT)
    if (
        REASON_FLOOR_FROM_EPV in range_reasons or REASON_BASE_FROM_EPV in range_reasons
    ) and SUPPORT_EPV not in support_types:
        consistency_reason_codes.append(REASON_EPV_FLOOR_WITHOUT_EPV_SUPPORT)
    if mos_classification != MOS_UNKNOWN and (
        _status_token(price_status) != OK or (not _is_num(mos_to_floor) and not _is_num(mos_to_base))
    ):
        consistency_reason_codes.append(REASON_MOS_WITHOUT_PRICE_OR_VALUE_SUPPORT)
    if confidence_class == CONFIDENCE_HIGH and (
        fragility_status == FRAGILITY_HIGH or support_count <= 1
    ):
        consistency_reason_codes.append(REASON_HIGH_CONFIDENCE_WITH_HIGH_FRAGILITY)
    if value_type_primary == VALUE_TYPE_EARNINGS_POWER and not (
        downside_support_type == SUPPORT_EARNINGS
        or bool(support_types & {SUPPORT_EPV, SUPPORT_NORMALIZED, SUPPORT_OWNER})
    ):
        consistency_reason_codes.append(REASON_VALUE_TYPE_SUPPORT_MISMATCH)
    if value_type_primary == VALUE_TYPE_ASSET_BACKED and not (
        downside_support_type in {SUPPORT_ASSET, SUPPORT_BALANCE_SHEET}
        or SUPPORT_NETNET in support_types
    ):
        consistency_reason_codes.append(REASON_VALUE_TYPE_SUPPORT_MISMATCH)
    if value_type_primary == VALUE_TYPE_QUALITY and not (
        support_count >= 2 and confidence_class not in {CONFIDENCE_UNKNOWN}
    ):
        consistency_reason_codes.append(REASON_VALUE_TYPE_SUPPORT_MISMATCH)
    consistency_reason_codes = _dedupe(consistency_reason_codes)

    if consistency_reason_codes:
        consistency_status = CONSISTENCY_INCONSISTENT
    elif support_count == 0 and confidence_class == CONFIDENCE_UNKNOWN:
        consistency_status = CONSISTENCY_UNKNOWN
        consistency_reason_codes = [REASON_CONSISTENCY_CONTEXT_INSUFFICIENT]
    else:
        consistency_status = CONSISTENCY_CONSISTENT

    provenance_summary = _fingerprint_summary(
        intrinsic_payload=intrinsic_payload,
        valuation_confidence_payload=valuation_confidence_payload,
        value_type_payload=value_type_payload,
        price_status=price_status,
        shares_status=shares_status,
        fcf_status=fcf_status,
        facts_status=facts_status,
        valuation_status=valuation_status,
    )
    fingerprint = _fingerprint_token(provenance_summary)
    derived_from = provenance_summary.get("derived_from_bundle_summary") or []
    integrity_class = (
        INTEGRITY_SUSPECT
        if consistency_status == CONSISTENCY_INCONSISTENT
        else (INTEGRITY_UNKNOWN if support_count == 0 else INTEGRITY_OK)
    )
    integrity_reason_codes = list(consistency_reason_codes)
    if integrity_class == INTEGRITY_OK:
        integrity_reason_codes.append(REASON_INTEGRITY_REDUCED_BY_ISOLATED_PROVENANCE)

    claims = {
        "valuation_integrity_class": _claim(
            value=integrity_class,
            refs=derived_from,
            reason_code=integrity_reason_codes[0] if integrity_reason_codes else UNKNOWN,
            status=OK if integrity_class != INTEGRITY_UNKNOWN else UNKNOWN,
        ),
        "valuation_consistency_status": _claim(
            value=consistency_status,
            refs=derived_from,
            reason_code=consistency_reason_codes[0] if consistency_reason_codes else UNKNOWN,
            status=OK if consistency_status != CONSISTENCY_UNKNOWN else UNKNOWN,
        ),
    }
    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "valuation_input_fingerprint": fingerprint,
        "valuation_input_provenance_summary": provenance_summary,
        "valuation_integrity_flags": [],
        "valuation_uniformity_group_id": None,
        "valuation_uniformity_reason_codes": [],
        "valuation_consistency_status": consistency_status,
        "valuation_consistency_reason_codes": consistency_reason_codes,
        "valuation_integrity_class": integrity_class,
        "valuation_integrity_reason_codes": _dedupe(integrity_reason_codes),
        "derived_from": derived_from,
        "claims": claims,
        "generated_at": utc_now_iso(),
    }


def _integrity_class_rank(value: str) -> int:
    return _INTEGRITY_ORDER.get(str(value or INTEGRITY_UNKNOWN), len(_INTEGRITY_ORDER))


def _exact_numeric_key(*values: Any) -> tuple[float, ...] | None:
    if not values or not all(_is_num(value) for value in values):
        return None
    return tuple(round(float(value), 10) for value in values)


def _bundle_key(summary: dict[str, Any]) -> str:
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))


def _attach_group(
    *,
    rows_by_ticker: dict[str, dict[str, Any]],
    tickers: list[str],
    group_id: str,
    reason_code: str,
) -> None:
    for ticker in tickers:
        row = rows_by_ticker.get(ticker)
        if not isinstance(row, dict):
            continue
        flags = _dedupe(list(row.get("valuation_integrity_flags") or []) + [reason_code])
        reasons = _dedupe(list(row.get("valuation_uniformity_reason_codes") or []) + [reason_code])
        row["valuation_integrity_flags"] = flags
        row["valuation_uniformity_reason_codes"] = reasons
        if not row.get("valuation_uniformity_group_id"):
            row["valuation_uniformity_group_id"] = group_id


def write_valuation_integrity_for_run(
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
    scoreboard_rows = [row for row in (scoreboard_rows or []) if isinstance(row, dict)]
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("valuation_integrity_detail")
        for row in scoreboard_rows
        if isinstance(row.get("valuation_integrity_detail"), dict)
    }
    rows: list[dict[str, Any]] = []
    rows_by_ticker: dict[str, dict[str, Any]] = {}

    for ticker in sorted({str(token or "").strip().upper() for token in tickers if str(token or "").strip()}):
        detail = detail_by_ticker.get(ticker)
        score_row = next(
            (
                row
                for row in scoreboard_rows
                if str(row.get("ticker") or "").strip().upper() == ticker
            ),
            {},
        )
        if not isinstance(detail, dict):
            detail = compute_valuation_integrity(
                ticker=ticker,
                as_of_date=as_of_date,
                intrinsic_payload=score_row.get("intrinsic_discipline_detail")
                if isinstance(score_row.get("intrinsic_discipline_detail"), dict)
                else {},
                valuation_confidence_payload=score_row.get("valuation_confidence_detail")
                if isinstance(score_row.get("valuation_confidence_detail"), dict)
                else {},
                value_type_payload=score_row.get("value_type_detail")
                if isinstance(score_row.get("value_type_detail"), dict)
                else {},
                price_status=score_row.get("price_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                fcf_status=score_row.get("fcf_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                valuation_status=score_row.get("valuation_status", UNKNOWN),
            )
        rows.append(detail)
        rows_by_ticker[ticker] = detail

    uniformity_groups: list[dict[str, Any]] = []
    counters = defaultdict(int)

    def register_groups(
        *,
        groups: dict[Any, list[str]],
        reason_code: str,
        prefix: str,
        min_count: int,
    ) -> None:
        for group_key, members in groups.items():
            member_set = sorted({str(ticker) for ticker in members if str(ticker).strip()})
            if len(member_set) < min_count:
                continue
            counters[prefix] += 1
            group_id = f"{prefix}_{counters[prefix]:03d}"
            uniformity_groups.append(
                {
                    "group_id": group_id,
                    "reason_code": reason_code,
                    "size": len(member_set),
                    "tickers": member_set,
                    "group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
                }
            )
            _attach_group(
                rows_by_ticker=rows_by_ticker,
                tickers=member_set,
                group_id=group_id,
                reason_code=reason_code,
            )

    range_groups: dict[tuple[float, ...], list[str]] = defaultdict(list)
    normalized_groups: dict[tuple[float, ...], list[str]] = defaultdict(list)
    mos_groups: dict[tuple[float, ...], list[str]] = defaultdict(list)
    bundle_groups: dict[str, list[str]] = defaultdict(list)
    fingerprint_groups: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        ticker = str(row.get("ticker") or "")
        score_row = next(
            (
                candidate
                for candidate in scoreboard_rows
                if str(candidate.get("ticker") or "").strip().upper() == ticker
            ),
            {},
        )
        intrinsic_payload = (
            score_row.get("intrinsic_discipline_detail")
            if isinstance(score_row.get("intrinsic_discipline_detail"), dict)
            else {}
        )
        valuation_confidence_payload = (
            score_row.get("valuation_confidence_detail")
            if isinstance(score_row.get("valuation_confidence_detail"), dict)
            else {}
        )
        range_key = _exact_numeric_key(
            intrinsic_payload.get("intrinsic_floor", UNKNOWN),
            intrinsic_payload.get("intrinsic_base", UNKNOWN),
            intrinsic_payload.get("intrinsic_ceiling", UNKNOWN),
        )
        if range_key is not None:
            range_groups[range_key].append(ticker)
        normalized_key = _exact_numeric_key(
            intrinsic_payload.get("normalized_earnings_power_value", UNKNOWN)
        )
        if normalized_key is not None:
            normalized_groups[normalized_key].append(ticker)
        mos_key = _exact_numeric_key(
            intrinsic_payload.get("mos_to_floor", UNKNOWN),
            intrinsic_payload.get("mos_to_base", UNKNOWN),
        )
        if mos_key is not None:
            mos_groups[mos_key].append(ticker)
        bundle_groups[_bundle_key(row.get("valuation_input_provenance_summary") or {})].append(ticker)
        fingerprint_groups[str(row.get("valuation_input_fingerprint") or "")].append(ticker)

    register_groups(
        groups=range_groups,
        reason_code=REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER,
        prefix="RANGE",
        min_count=2,
    )
    register_groups(
        groups=normalized_groups,
        reason_code=REASON_IDENTICAL_NORMALIZED_EARNINGS_CLUSTER,
        prefix="NORM",
        min_count=2,
    )
    register_groups(
        groups=mos_groups,
        reason_code=REASON_IDENTICAL_MOS_CLUSTER,
        prefix="MOS",
        min_count=3,
    )
    register_groups(
        groups=bundle_groups,
        reason_code=REASON_IDENTICAL_SUPPORT_BUNDLE_CLUSTER,
        prefix="BUNDLE",
        min_count=2,
    )

    for fingerprint, members in fingerprint_groups.items():
        member_set = sorted({str(ticker) for ticker in members if str(ticker).strip()})
        if not fingerprint or len(member_set) < 2:
            continue
        counters["FPRINT"] += 1
        group_id = f"FPRINT_{counters['FPRINT']:03d}"
        uniformity_groups.append(
            {
                "group_id": group_id,
                "reason_code": REASON_POSSIBLE_STALE_ARTIFACT_REUSE,
                "size": len(member_set),
                "tickers": member_set,
                "group_key": fingerprint,
            }
        )
        _attach_group(
            rows_by_ticker=rows_by_ticker,
            tickers=member_set,
            group_id=group_id,
            reason_code=REASON_POSSIBLE_STALE_ARTIFACT_REUSE,
        )

    for row in rows:
        uniformity_reasons = {
            str(code)
            for code in (row.get("valuation_uniformity_reason_codes") or [])
            if str(code).strip()
        }
        if {
            REASON_IDENTICAL_INTRINSIC_RANGE_CLUSTER,
            REASON_IDENTICAL_NORMALIZED_EARNINGS_CLUSTER,
        }.issubset(uniformity_reasons):
            uniformity_reasons.add(REASON_POSSIBLE_DEFAULT_VALUE_REUSE)
        consistency_status = str(row.get("valuation_consistency_status") or CONSISTENCY_UNKNOWN)
        support_summary = row.get("valuation_input_provenance_summary") or {}
        support_types = support_summary.get("valuation_support_types_present") or []
        if consistency_status == CONSISTENCY_INCONSISTENT or {
            REASON_POSSIBLE_DEFAULT_VALUE_REUSE,
            REASON_POSSIBLE_STALE_ARTIFACT_REUSE,
        } & uniformity_reasons:
            integrity_class = INTEGRITY_SUSPECT
        elif uniformity_reasons:
            integrity_class = INTEGRITY_WARNING
        elif not support_types:
            integrity_class = INTEGRITY_UNKNOWN
        else:
            integrity_class = INTEGRITY_OK
        integrity_reasons = _dedupe(
            list(row.get("valuation_consistency_reason_codes") or [])
            + sorted(uniformity_reasons)
            + (
                [REASON_INTEGRITY_REDUCED_BY_ISOLATED_PROVENANCE]
                if integrity_class == INTEGRITY_OK
                else []
            )
        )
        row["valuation_uniformity_reason_codes"] = sorted(uniformity_reasons)
        row["valuation_integrity_flags"] = _dedupe(
            list(row.get("valuation_integrity_flags") or []) + sorted(uniformity_reasons)
        )
        row["valuation_integrity_class"] = integrity_class
        row["valuation_integrity_reason_codes"] = integrity_reasons
        row["claims"] = row.get("claims") if isinstance(row.get("claims"), dict) else {}
        row["claims"]["valuation_integrity_class"] = _claim(
            value=integrity_class,
            refs=row.get("derived_from") or [],
            reason_code=integrity_reasons[0] if integrity_reasons else UNKNOWN,
            status=OK if integrity_class != INTEGRITY_UNKNOWN else UNKNOWN,
        )

    counts_by_class: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        integrity_class = str(row.get("valuation_integrity_class") or INTEGRITY_UNKNOWN)
        counts_by_class[integrity_class] = counts_by_class.get(integrity_class, 0) + 1
        for code in (row.get("valuation_integrity_reason_codes") or []):
            token = str(code or "").strip()
            if not token or token == REASON_INTEGRITY_REDUCED_BY_ISOLATED_PROVENANCE:
                continue
            reason_counts[token] = reason_counts.get(token, 0) + 1

    suspect_rows = [
        row for row in rows if str(row.get("valuation_integrity_class") or "") == INTEGRITY_SUSPECT
    ]
    suspect_rows.sort(
        key=lambda row: (
            _integrity_class_rank(str(row.get("valuation_integrity_class") or INTEGRITY_UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    )
    uniformity_groups.sort(
        key=lambda row: (
            str(row.get("reason_code") or ""),
            -int(row.get("size") or 0),
            str(row.get("group_id") or ""),
        )
    )

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "counts_by_integrity_class": dict(
            sorted(counts_by_class.items(), key=lambda item: (_integrity_class_rank(item[0]), item[0]))
        ),
        "suspicious_uniformity_groups": uniformity_groups,
        "top_10_integrity_suspect": [
            {
                "ticker": str(row.get("ticker") or ""),
                "valuation_integrity_class": str(row.get("valuation_integrity_class") or INTEGRITY_UNKNOWN),
                "valuation_uniformity_group_id": row.get("valuation_uniformity_group_id"),
                "valuation_integrity_reason_codes": [
                    str(code)
                    for code in (row.get("valuation_integrity_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in suspect_rows[:10]
        ],
        "integrity_reason_counts": dict(
            sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ),
        "exact_uniformity_cluster_count": int(len(uniformity_groups)),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["valuation_integrity_path"] = str(output_path)
    return payload


def _valuation_integrity_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "valuation_integrity.json",
        cfg.sectors_dir / run_id / "valuation_integrity.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_valuation_integrity(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _valuation_integrity_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "valuation_integrity_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_integrity_class": payload.get("counts_by_integrity_class")
        if isinstance(payload.get("counts_by_integrity_class"), dict)
        else {},
        "suspicious_uniformity_groups": [
            row for row in (payload.get("suspicious_uniformity_groups") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_integrity_suspect": [
            row for row in (payload.get("top_10_integrity_suspect") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "integrity_reason_counts": payload.get("integrity_reason_counts")
        if isinstance(payload.get("integrity_reason_counts"), dict)
        else {},
        "exact_uniformity_cluster_count": int(payload.get("exact_uniformity_cluster_count") or 0),
        "valuation_integrity_path": str(path),
    }
