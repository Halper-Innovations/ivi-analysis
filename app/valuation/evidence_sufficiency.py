from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.valuation.intrinsic_discipline import MOS_ADEQUATE, MOS_DEEP_VALUE_SUPPORT, MOS_MODEST, MOS_NONE, MOS_UNKNOWN
from app.valuation.valuation_confidence import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
from app.valuation.valuation_integrity import INTEGRITY_OK, INTEGRITY_WARNING


UNKNOWN = "UNKNOWN"
OK = "OK"

SUFFICIENCY_SUFFICIENT = "SUFFICIENT_FOR_MOS"
SUFFICIENCY_PARTIAL = "PARTIAL_FOR_MOS"
SUFFICIENCY_INSUFFICIENT = "INSUFFICIENT_FOR_MOS"
SUFFICIENCY_UNKNOWN = "SUFFICIENCY_UNKNOWN"

MOS_CONFIRMED_ABSENT = "MOS_CONFIRMED_ABSENT"
MOS_CONFIRMED_PRESENT = "MOS_CONFIRMED_PRESENT"
MOS_WEAK = "MOS_WEAK"
MOS_UNASSESSABLE = "MOS_UNASSESSABLE"
MOS_UNKNOWN_STATUS = "MOS_UNKNOWN"

REASON_PRICE_AVAILABLE = "PRICE_AVAILABLE"
REASON_SHARES_AVAILABLE = "SHARES_AVAILABLE"
REASON_FACTS_AVAILABLE = "FACTS_AVAILABLE"
REASON_MULTI_SUPPORT_PRESENT = "MULTI_SUPPORT_PRESENT"
REASON_MISSING_PRICE = "MISSING_PRICE"
REASON_MISSING_SHARES = "MISSING_SHARES"
REASON_MISSING_FACTS = "MISSING_FACTS"
REASON_SINGLE_SUPPORT_ONLY = "SINGLE_SUPPORT_ONLY"
REASON_SUPPORTS_TOO_THIN = "SUPPORTS_TOO_THIN"
REASON_INTEGRITY_TOO_WEAK = "INTEGRITY_TOO_WEAK"
REASON_CONFIDENCE_TOO_WEAK = "CONFIDENCE_TOO_WEAK"

REASON_NO_MOS_WITH_SUFFICIENT_EVIDENCE = "NO_MOS_WITH_SUFFICIENT_EVIDENCE"
REASON_MOS_BLOCKED_BY_MISSING_PRICE = "MOS_BLOCKED_BY_MISSING_PRICE"
REASON_MOS_BLOCKED_BY_MISSING_FACTS = "MOS_BLOCKED_BY_MISSING_FACTS"
REASON_MOS_BLOCKED_BY_MISSING_SHARES = "MOS_BLOCKED_BY_MISSING_SHARES"
REASON_MOS_BLOCKED_BY_LOW_CONFIDENCE_SUPPORT = "MOS_BLOCKED_BY_LOW_CONFIDENCE_SUPPORT"
REASON_MOS_BLOCKED_BY_INTEGRITY_HEADWIND = "MOS_BLOCKED_BY_INTEGRITY_HEADWIND"
REASON_MOS_PRESENT_WITH_SUFFICIENT_EVIDENCE = "MOS_PRESENT_WITH_SUFFICIENT_EVIDENCE"
REASON_MOS_WEAK_WITH_SUFFICIENT_EVIDENCE = "MOS_WEAK_WITH_SUFFICIENT_EVIDENCE"

_SUFFICIENCY_ORDER = {
    SUFFICIENCY_SUFFICIENT: 0,
    SUFFICIENCY_PARTIAL: 1,
    SUFFICIENCY_INSUFFICIENT: 2,
    SUFFICIENCY_UNKNOWN: 3,
}

_MOS_ASSESSMENT_ORDER = {
    MOS_CONFIRMED_PRESENT: 0,
    MOS_WEAK: 1,
    MOS_CONFIRMED_ABSENT: 2,
    MOS_UNASSESSABLE: 3,
    MOS_UNKNOWN_STATUS: 4,
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
    token = str(value or "").strip()
    if _is_num(value):
        token = str(float(value))
    if token and token.upper() != UNKNOWN:
        return {
            "value": value,
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


def _payload_refs(*payloads: dict[str, Any], row_refs: list[Any] | None = None) -> list[str]:
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


def _coalesce_status(*values: Any, fallback: str = UNKNOWN) -> str:
    for value in values:
        token = str(value or "").strip().upper()
        if token:
            return token
    return fallback


def compute_evidence_sufficiency(
    ticker: str,
    as_of_date: str,
    *,
    intrinsic_payload: dict[str, Any] | None = None,
    valuation_confidence_payload: dict[str, Any] | None = None,
    valuation_integrity_payload: dict[str, Any] | None = None,
    price_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    fcf_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    valuation_status: Any = UNKNOWN,
    facts_blocker_class: str = "FACTS_OK",
    fail_due_to_missing_evidence: bool = False,
    primary_fail_domain: str | None = None,
    row_derived_from: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg
    intrinsic_payload = intrinsic_payload if isinstance(intrinsic_payload, dict) else {}
    valuation_confidence_payload = (
        valuation_confidence_payload if isinstance(valuation_confidence_payload, dict) else {}
    )
    valuation_integrity_payload = (
        valuation_integrity_payload if isinstance(valuation_integrity_payload, dict) else {}
    )

    ticker_norm = str(ticker or "").strip().upper()
    price_ok = _coalesce_status(price_status) == OK
    shares_ok = _coalesce_status(shares_status) == OK
    fcf_ok = _coalesce_status(fcf_status) == OK
    facts_ok = _coalesce_status(facts_status) == OK and str(facts_blocker_class or "FACTS_OK").upper() == "FACTS_OK"
    valuation_ok = _coalesce_status(valuation_status) == OK

    support_count = int(valuation_confidence_payload.get("valuation_support_count") or 0)
    confidence_class = _coalesce_status(
        valuation_confidence_payload.get("valuation_confidence_class"),
        UNKNOWN,
    )
    integrity_class = _coalesce_status(
        valuation_integrity_payload.get("valuation_integrity_class"),
        UNKNOWN,
    )
    mos_classification = _coalesce_status(
        intrinsic_payload.get("mos_classification"),
        MOS_UNKNOWN,
    )
    mos_to_floor = intrinsic_payload.get("mos_to_floor", UNKNOWN)
    has_numeric_mos = _is_num(mos_to_floor) or mos_classification in {
        MOS_DEEP_VALUE_SUPPORT,
        MOS_ADEQUATE,
        MOS_MODEST,
        MOS_NONE,
    }
    primary_fail_domain = _coalesce_status(primary_fail_domain, UNKNOWN)

    sufficiency_reasons = _dedupe(
        [
            REASON_PRICE_AVAILABLE if price_ok else REASON_MISSING_PRICE,
            REASON_SHARES_AVAILABLE if shares_ok else REASON_MISSING_SHARES,
            REASON_FACTS_AVAILABLE if facts_ok else REASON_MISSING_FACTS,
            REASON_MULTI_SUPPORT_PRESENT if support_count >= 2 else "",
            REASON_SINGLE_SUPPORT_ONLY if support_count == 1 else "",
            REASON_SUPPORTS_TOO_THIN if support_count <= 0 else "",
            REASON_CONFIDENCE_TOO_WEAK if confidence_class not in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM} else "",
            REASON_INTEGRITY_TOO_WEAK if integrity_class not in {INTEGRITY_OK, INTEGRITY_WARNING} else "",
        ]
    )

    critical_missing = (not price_ok) or (not shares_ok)
    degraded_inputs = (not facts_ok) or (not valuation_ok) or (not fcf_ok)
    support_too_thin = support_count <= 0 or confidence_class not in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}
    integrity_too_weak = integrity_class not in {INTEGRITY_OK, INTEGRITY_WARNING}

    if critical_missing or (
        fail_due_to_missing_evidence and primary_fail_domain == "EVIDENCE" and (not has_numeric_mos or degraded_inputs)
    ):
        evidence_sufficiency_class = SUFFICIENCY_INSUFFICIENT
    elif price_ok and shares_ok and valuation_ok and facts_ok and support_count >= 2 and not integrity_too_weak and has_numeric_mos:
        evidence_sufficiency_class = SUFFICIENCY_SUFFICIENT
    elif price_ok and shares_ok and valuation_ok and (has_numeric_mos or support_count >= 1):
        evidence_sufficiency_class = SUFFICIENCY_PARTIAL
    elif fail_due_to_missing_evidence or degraded_inputs or support_too_thin:
        evidence_sufficiency_class = SUFFICIENCY_INSUFFICIENT
    else:
        evidence_sufficiency_class = SUFFICIENCY_UNKNOWN

    guardrail_reasons: list[str] = []
    if evidence_sufficiency_class == SUFFICIENCY_SUFFICIENT:
        if mos_classification in {MOS_DEEP_VALUE_SUPPORT, MOS_ADEQUATE}:
            mos_assessment_status = MOS_CONFIRMED_PRESENT
            guardrail_reasons.append(REASON_MOS_PRESENT_WITH_SUFFICIENT_EVIDENCE)
        elif mos_classification == MOS_MODEST:
            mos_assessment_status = MOS_WEAK
            guardrail_reasons.append(REASON_MOS_WEAK_WITH_SUFFICIENT_EVIDENCE)
        elif mos_classification == MOS_NONE:
            mos_assessment_status = MOS_CONFIRMED_ABSENT
            guardrail_reasons.append(REASON_NO_MOS_WITH_SUFFICIENT_EVIDENCE)
        else:
            mos_assessment_status = MOS_UNKNOWN_STATUS
    else:
        if not price_ok:
            mos_assessment_status = MOS_UNASSESSABLE
            guardrail_reasons.append(REASON_MOS_BLOCKED_BY_MISSING_PRICE)
        elif not shares_ok:
            mos_assessment_status = MOS_UNASSESSABLE
            guardrail_reasons.append(REASON_MOS_BLOCKED_BY_MISSING_SHARES)
        elif not facts_ok:
            mos_assessment_status = MOS_UNASSESSABLE
            guardrail_reasons.append(REASON_MOS_BLOCKED_BY_MISSING_FACTS)
        elif confidence_class not in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM} or support_count <= 1:
            mos_assessment_status = MOS_UNASSESSABLE
            guardrail_reasons.append(REASON_MOS_BLOCKED_BY_LOW_CONFIDENCE_SUPPORT)
        elif integrity_too_weak:
            mos_assessment_status = MOS_UNASSESSABLE
            guardrail_reasons.append(REASON_MOS_BLOCKED_BY_INTEGRITY_HEADWIND)
        elif evidence_sufficiency_class == SUFFICIENCY_PARTIAL:
            mos_assessment_status = MOS_UNASSESSABLE
            guardrail_reasons.append(REASON_MOS_BLOCKED_BY_LOW_CONFIDENCE_SUPPORT)
        else:
            mos_assessment_status = MOS_UNKNOWN_STATUS

    derived_from = _payload_refs(
        intrinsic_payload,
        valuation_confidence_payload,
        valuation_integrity_payload,
        row_refs=row_derived_from,
    )
    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "evidence_sufficiency_class": evidence_sufficiency_class,
        "evidence_sufficiency_reason_codes": sufficiency_reasons,
        "mos_assessment_status": mos_assessment_status,
        "mos_guardrail_reason_codes": _dedupe(guardrail_reasons),
        "derived_from": derived_from,
        "claims": {
            "evidence_sufficiency_class": _claim(
                value=evidence_sufficiency_class,
                refs=derived_from,
                reason_code=sufficiency_reasons[0] if sufficiency_reasons else UNKNOWN,
                status=OK if evidence_sufficiency_class != SUFFICIENCY_UNKNOWN else UNKNOWN,
            ),
            "mos_assessment_status": _claim(
                value=mos_assessment_status,
                refs=derived_from,
                reason_code=guardrail_reasons[0] if guardrail_reasons else UNKNOWN,
                status=OK if mos_assessment_status not in {MOS_UNKNOWN_STATUS, MOS_UNASSESSABLE} else UNKNOWN,
            ),
        },
        "generated_at": utc_now_iso(),
    }


def write_evidence_sufficiency_for_run(
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
        str(row.get("ticker") or "").upper(): row.get("evidence_sufficiency_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("evidence_sufficiency_detail"), dict)
    }
    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    sufficiency_counts: dict[str, int] = {}
    mos_counts: dict[str, int] = {}

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
            detail = compute_evidence_sufficiency(
                ticker=ticker,
                as_of_date=as_of_date,
                intrinsic_payload=score_row.get("intrinsic_discipline_detail")
                if isinstance(score_row.get("intrinsic_discipline_detail"), dict)
                else {},
                valuation_confidence_payload=score_row.get("valuation_confidence_detail")
                if isinstance(score_row.get("valuation_confidence_detail"), dict)
                else {},
                valuation_integrity_payload=score_row.get("valuation_integrity_detail")
                if isinstance(score_row.get("valuation_integrity_detail"), dict)
                else {},
                price_status=score_row.get("price_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                fcf_status=score_row.get("fcf_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                valuation_status=score_row.get("valuation_status", UNKNOWN),
                facts_blocker_class=str(score_row.get("facts_blocker_class") or "FACTS_OK"),
                fail_due_to_missing_evidence=bool(score_row.get("fail_due_to_missing_evidence", False)),
                primary_fail_domain=str(score_row.get("primary_fail_domain") or UNKNOWN),
                row_derived_from=list(score_row.get("derived_from") or []),
            )
        rows.append(detail)
        sufficiency = str(detail.get("evidence_sufficiency_class") or SUFFICIENCY_UNKNOWN)
        mos_status = str(detail.get("mos_assessment_status") or MOS_UNKNOWN_STATUS)
        sufficiency_counts[sufficiency] = sufficiency_counts.get(sufficiency, 0) + 1
        mos_counts[mos_status] = mos_counts.get(mos_status, 0) + 1
        for code in detail.get("evidence_sufficiency_reason_codes") or []:
            token = str(code or "").strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1

    def _subset_rows(mos_status: str) -> list[dict[str, Any]]:
        subset = [
            row for row in rows if str(row.get("mos_assessment_status") or MOS_UNKNOWN_STATUS) == mos_status
        ]
        subset.sort(key=lambda row: (str(row.get("ticker") or ""),))
        return [
            {
                "ticker": str(row.get("ticker") or ""),
                "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or SUFFICIENCY_UNKNOWN),
                "mos_assessment_status": str(row.get("mos_assessment_status") or MOS_UNKNOWN_STATUS),
            }
            for row in subset
        ]

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "counts_by_evidence_sufficiency_class": dict(
            sorted(sufficiency_counts.items(), key=lambda item: (_SUFFICIENCY_ORDER.get(item[0], 99), item[0]))
        ),
        "counts_by_mos_assessment_status": dict(
            sorted(mos_counts.items(), key=lambda item: (_MOS_ASSESSMENT_ORDER.get(item[0], 99), item[0]))
        ),
        "top_mos_confirmed_absent": _subset_rows(MOS_CONFIRMED_ABSENT)[:10],
        "top_mos_unassessable": _subset_rows(MOS_UNASSESSABLE)[:10],
        "sufficiency_reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-int(item[1]), item[0]))),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["evidence_sufficiency_path"] = str(output_path)
    return payload


def _evidence_sufficiency_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "evidence_sufficiency.json",
        cfg.sectors_dir / run_id / "evidence_sufficiency.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_evidence_sufficiency(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _evidence_sufficiency_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "evidence_sufficiency_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_evidence_sufficiency_class": payload.get("counts_by_evidence_sufficiency_class")
        if isinstance(payload.get("counts_by_evidence_sufficiency_class"), dict)
        else {},
        "counts_by_mos_assessment_status": payload.get("counts_by_mos_assessment_status")
        if isinstance(payload.get("counts_by_mos_assessment_status"), dict)
        else {},
        "top_mos_confirmed_absent": [
            row for row in (payload.get("top_mos_confirmed_absent") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_mos_unassessable": [
            row for row in (payload.get("top_mos_unassessable") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "sufficiency_reason_counts": payload.get("sufficiency_reason_counts")
        if isinstance(payload.get("sufficiency_reason_counts"), dict)
        else {},
        "evidence_sufficiency_path": str(path),
    }
