"""
Normalization / Recovery Credibility v1

Deterministic per-ticker assessment of how credible the hypothesis of
normalized earnings recovery is, based on available underwriting evidence.

Doctrine:
- No technical indicators, no ML, no macro forecasting
- No narrative turnaround scoring, no aggressive optimism
- UNKNOWN remains UNKNOWN with explicit reason codes
- "Temporary weakness" must not automatically imply "likely recovery"
- Credible normalization requires measurable support
- This layer describes support for recoverability, not predicted timing
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.valuation.revenue_dependence import (
    HIGH_REVENUE_DEPENDENCE_RISK,
    REVENUE_DEPENDENCE_UNKNOWN,
)


UNKNOWN = "UNKNOWN"
OK = "OK"

# ── Normalization credibility classes ─────────────────────────────────────────

HIGH_NORMALIZATION_CREDIBILITY = "HIGH_NORMALIZATION_CREDIBILITY"
MODERATE_NORMALIZATION_CREDIBILITY = "MODERATE_NORMALIZATION_CREDIBILITY"
LOW_NORMALIZATION_CREDIBILITY = "LOW_NORMALIZATION_CREDIBILITY"
NORMALIZATION_CREDIBILITY_UNKNOWN = "NORMALIZATION_CREDIBILITY_UNKNOWN"

# Ordering: best (most credible) → worst
CREDIBILITY_ORDER = {
    HIGH_NORMALIZATION_CREDIBILITY: 0,
    MODERATE_NORMALIZATION_CREDIBILITY: 1,
    NORMALIZATION_CREDIBILITY_UNKNOWN: 2,
    LOW_NORMALIZATION_CREDIBILITY: 3,
}

# ── Primary normalization caution classes ─────────────────────────────────────

CAUTION_APPEARS_CREDIBLE = "NORMALIZATION_APPEARS_CREDIBLE"
CAUTION_PARTIALLY_SUPPORTED = "NORMALIZATION_PARTIALLY_SUPPORTED"
CAUTION_TOO_THEORETICAL = "NORMALIZATION_TOO_THEORETICAL"
CAUTION_BLOCKED_BY_EVIDENCE = "NORMALIZATION_BLOCKED_BY_EVIDENCE"
CAUTION_UNCLEAR = "NORMALIZATION_UNCLEAR"

# ── Recovery support signal constants ─────────────────────────────────────────

SIG_CYCLE_RESILIENCE_PRESENT = "CYCLE_RESILIENCE_PRESENT"
SIG_BALANCE_SHEET_OPTIONALITY_PRESENT = "BALANCE_SHEET_OPTIONALITY_PRESENT"
SIG_EARNINGS_POWER_SUPPORT_PRESENT = "EARNINGS_POWER_SUPPORT_PRESENT"
SIG_TROUGH_EARNINGS_RISK_PRESENT = "TROUGH_EARNINGS_RISK_PRESENT"
SIG_GROSS_MARGIN_DURABILITY_PRESENT = "GROSS_MARGIN_DURABILITY_PRESENT"
SIG_OWNER_EARNINGS_STABILITY_PRESENT = "OWNER_EARNINGS_STABILITY_PRESENT"
SIG_LOW_IMPAIRMENT_RISK = "LOW_IMPAIRMENT_RISK"
SIG_MOS_PRESENT = "MOS_PRESENT"

# ── Recovery headwind signal constants ────────────────────────────────────────

SIG_PEAK_EARNINGS_RISK = "PEAK_EARNINGS_RISK"
SIG_LOW_CONFIDENCE_VALUE = "LOW_CONFIDENCE_VALUE"
SIG_HIGH_FRAGILITY = "HIGH_FRAGILITY"
SIG_INTEGRITY_WARNING = "INTEGRITY_WARNING"
SIG_PROBABLE_IMPAIRMENT = "PROBABLE_IMPAIRMENT"
SIG_IMPAIRMENT_RISK_UNKNOWN = "IMPAIRMENT_RISK_UNKNOWN"
SIG_CLEAR_IMPAIRMENT = "CLEAR_IMPAIRMENT"
SIG_STRUCTURALLY_WEAK_ECONOMICS = "STRUCTURALLY_WEAK_ECONOMICS"
SIG_MISSING_EVIDENCE = "MISSING_EVIDENCE"
SIG_MOS_UNASSESSABLE = "MOS_UNASSESSABLE"

# ── Reason codes ──────────────────────────────────────────────────────────────

REASON_CYCLICAL_TROUGH_SUPPORTED = "CYCLICAL_TROUGH_WITH_RESILIENCE"
REASON_CYCLE_RESILIENCE_CONFIRMED = "CYCLE_RESILIENCE_CONFIRMED"
REASON_BALANCE_SHEET_SUPPORT = "BALANCE_SHEET_SUPPORT_PRESENT"
REASON_EARNINGS_POWER_ANCHORED = "EARNINGS_POWER_ANCHORED"
REASON_GROSS_MARGIN_DURABLE = "GROSS_MARGIN_DURABILITY_CONFIRMED"
REASON_OWNER_EARNINGS_STABLE = "OWNER_EARNINGS_STABILITY_CONFIRMED"
REASON_PARTIAL_SUPPORT_ONLY = "PARTIAL_NORMALIZATION_SUPPORT"
REASON_NO_CYCLICAL_CONTEXT = "NO_CYCLICAL_CONTEXT"
REASON_IMPAIRMENT_DOMINANT = "IMPAIRMENT_SIGNALS_DOMINANT"
REASON_IMPAIRMENT_CLASS_UNKNOWN = "IMPAIRMENT_CLASS_UNKNOWN"
REASON_CLEAR_IMPAIRMENT_PRESENT = "CLEAR_IMPAIRMENT_PRESENT"
REASON_PROBABLE_IMPAIRMENT_PRESENT = "PROBABLE_IMPAIRMENT_PRESENT"
REASON_FRAGILITY_HIGH = "HIGH_FRAGILITY_UNDERMINES_CASE"
REASON_INTEGRITY_ISSUE = "INTEGRITY_ISSUE"
REASON_EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT_FOR_CREDIBILITY"
REASON_EVIDENCE_MISSING_PRICE = "EVIDENCE_MISSING_PRICE"
REASON_EVIDENCE_MISSING_FACTS = "EVIDENCE_MISSING_FACTS"
REASON_SIGNALS_CONFLICT = "CONFLICTING_SIGNALS"
REASON_CYCLICAL_PEAK_WRONG_DIRECTION = "CYCLICAL_PEAK_NOT_TROUGH"
REASON_LOW_CONFIDENCE = "LOW_CONFIDENCE_UNDERWRITING"
REASON_NO_SUPPORT_FOR_CLAIM = "NO_MEASURABLE_SUPPORT_FOR_NORMALIZATION"


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dedupe_refs(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
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


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect_derived_from(*payloads: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        refs.extend([str(ref) for ref in (payload.get("derived_from") or []) if str(ref).strip()])
    return _dedupe_refs(refs)


def compute_normalization_credibility(
    ticker: str,
    as_of_date: str,
    *,
    cyclical_normalization_payload: dict[str, Any] | None = None,
    impairment_classification_payload: dict[str, Any] | None = None,
    evidence_sufficiency_payload: dict[str, Any] | None = None,
    valuation_confidence_payload: dict[str, Any] | None = None,
    valuation_integrity_payload: dict[str, Any] | None = None,
    intrinsic_payload: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    value_type_payload: dict[str, Any] | None = None,
    revenue_dependence_payload: dict[str, Any] | None = None,
    price_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """
    Compute normalization / recovery credibility for a single ticker.

    Conservative underwriting: this layer describes evidence that supports the
    hypothesis that current weakness is temporary and that normalized economics
    are achievable. It does NOT predict recovery timing or catalog catalysts.

    UNKNOWN remains UNKNOWN. Evidence gaps are not treated as credible.
    """
    cfg = cfg or get_config()
    del cfg

    ticker_norm = str(ticker or "").strip().upper()
    cyc = cyclical_normalization_payload if isinstance(cyclical_normalization_payload, dict) else {}
    imp = impairment_classification_payload if isinstance(impairment_classification_payload, dict) else {}
    evs = evidence_sufficiency_payload if isinstance(evidence_sufficiency_payload, dict) else {}
    conf = valuation_confidence_payload if isinstance(valuation_confidence_payload, dict) else {}
    integrity = valuation_integrity_payload if isinstance(valuation_integrity_payload, dict) else {}
    intrinsic = intrinsic_payload if isinstance(intrinsic_payload, dict) else {}
    oe_quality = owner_quality_payload if isinstance(owner_quality_payload, dict) else {}
    intangible = intangible_payload if isinstance(intangible_payload, dict) else {}
    value_type = value_type_payload if isinstance(value_type_payload, dict) else {}
    revenue_dependence = revenue_dependence_payload if isinstance(revenue_dependence_payload, dict) else {}

    # ── Extract cyclical context ───────────────────────────────────────────────
    cyclical_profile_class = str(cyc.get("cyclical_profile_class") or "CYCLICALITY_UNKNOWN").upper()
    cycle_position_class = str(cyc.get("cycle_position_class") or "CYCLE_POSITION_UNKNOWN").upper()
    cyclical_valuation_risk_class = str(cyc.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN").upper()

    cyclical_active = cyclical_profile_class in {"CLEARLY_CYCLICAL", "MODERATELY_CYCLICAL"}
    cyclical_trough = cyclical_active and cycle_position_class == "DEPRESSED_RELATIVE_TO_NORMAL"
    cyclical_peak = cyclical_active and cycle_position_class == "ELEVATED_RELATIVE_TO_NORMAL"
    trough_valuation_risk = cyclical_valuation_risk_class == "TROUGH_EARNINGS_RISK"

    # ── Extract impairment class ──────────────────────────────────────────────
    impairment_class = str(imp.get("impairment_class_primary") or "IMPAIRMENT_UNKNOWN").upper()
    clear_impairment = impairment_class == "CLEAR_IMPAIRMENT"
    probable_impairment = impairment_class == "PROBABLE_IMPAIRMENT"
    temporary_weakness = impairment_class == "TEMPORARY_WEAKNESS"
    structurally_weak = impairment_class == "STRUCTURALLY_WEAK_NOT_IMPAIRED"
    impairment_dominant = clear_impairment or probable_impairment
    # LOW_IMPAIRMENT_RISK needs a classification that actually says so. An absent
    # payload, IMPAIRMENT_UNKNOWN, EVIDENCE_DEGRADED_NOT_ASSESSABLE or any string this
    # module does not recognise is a gap, not evidence of low risk.
    impairment_unassessed = impairment_class not in {
        "CLEAR_IMPAIRMENT",
        "PROBABLE_IMPAIRMENT",
        "TEMPORARY_WEAKNESS",
        "STRUCTURALLY_WEAK_NOT_IMPAIRED",
    }

    # ── Extract evidence quality ───────────────────────────────────────────────
    evidence_sufficiency_class = str(evs.get("evidence_sufficiency_class") or "SUFFICIENCY_UNKNOWN").upper()
    mos_assessment_status = str(evs.get("mos_assessment_status") or "MOS_UNKNOWN").upper()
    evidence_sufficient = evidence_sufficiency_class in {"SUFFICIENT_FOR_MOS", "PARTIAL_FOR_MOS"}
    evidence_insufficient = evidence_sufficiency_class in {"INSUFFICIENT_FOR_MOS", "SUFFICIENCY_UNKNOWN"}
    mos_present = mos_assessment_status == "MOS_CONFIRMED_PRESENT"
    mos_unassessable = mos_assessment_status == "MOS_UNASSESSABLE"

    # Evidence availability
    price_ok = str(price_status or "").upper() == "OK"
    facts_ok = str(facts_status or "").upper() == "OK"
    evidence_blocked = not price_ok or not facts_ok

    # ── Extract valuation confidence ──────────────────────────────────────────
    confidence_class = str(conf.get("valuation_confidence_class") or "CONFIDENCE_UNKNOWN").upper()
    fragility_status = str(conf.get("valuation_fragility_status") or "FRAGILITY_UNKNOWN").upper()
    high_confidence = confidence_class in {"HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE"}
    low_confidence = confidence_class in {"LOW_CONFIDENCE", "CONFIDENCE_UNKNOWN"}
    high_fragility = fragility_status == "HIGH_FRAGILITY"

    # ── Extract valuation integrity ────────────────────────────────────────────
    integrity_class = str(integrity.get("valuation_integrity_class") or "INTEGRITY_UNKNOWN").upper()
    integrity_issue = integrity_class in {"INTEGRITY_SUSPECT", "INTEGRITY_WARNING"}

    # ── Extract downside support ───────────────────────────────────────────────
    downside_support_type = str(intrinsic.get("downside_support_type") or "UNKNOWN_SUPPORT").upper()
    mos_classification = str(intrinsic.get("mos_classification") or UNKNOWN).upper()
    real_support_present = downside_support_type in {
        "ASSET_SUPPORT", "EARNINGS_POWER_SUPPORT", "BALANCE_SHEET_SUPPORT",
    }
    mos_real = mos_classification in {"DEEP_VALUE_SUPPORT", "ADEQUATE_MARGIN_OF_SAFETY", "MODEST_MARGIN_OF_SAFETY"}

    # ── Extract OE quality / stability ────────────────────────────────────────
    oe_quality_total = oe_quality.get("oe_quality_total", UNKNOWN)
    owner_earnings_stability_score = oe_quality.get("owner_earnings_stability_score", UNKNOWN)
    oe_quality_ok = _is_num(oe_quality_total) and float(oe_quality_total) >= 6.0
    owner_earnings_stable = (
        _is_num(owner_earnings_stability_score) and float(owner_earnings_stability_score) >= 3.0
    )
    owner_earnings_weak = _is_num(oe_quality_total) and float(oe_quality_total) < 4.0

    # ── Extract intangible / resilience signals ────────────────────────────────
    cycle_resilience_score = intangible.get("cycle_resilience_score", UNKNOWN)
    gross_margin_durability_score = intangible.get("gross_margin_durability_score", UNKNOWN)
    balance_sheet_optionality_score = intangible.get("balance_sheet_optionality_score", UNKNOWN)
    cycle_resilience_present = (
        _is_num(cycle_resilience_score) and float(cycle_resilience_score) >= 3.0
    )
    gross_margin_durable = (
        _is_num(gross_margin_durability_score) and float(gross_margin_durability_score) >= 3.0
    )
    balance_sheet_optionality_present = (
        _is_num(balance_sheet_optionality_score) and float(balance_sheet_optionality_score) >= 3.0
    )

    # ── Extract value type ────────────────────────────────────────────────────
    value_type_primary = str(value_type.get("value_type_primary") or "UNKNOWN_VALUE_TYPE").upper()
    is_cyclical_value_type = value_type_primary == "CYCLICAL_VALUE"
    is_fragile_value_type = value_type_primary == "FRAGILE_VALUE"
    revenue_dependence_class = str(
        revenue_dependence.get("revenue_dependence_risk_class") or REVENUE_DEPENDENCE_UNKNOWN
    ).upper()
    revenue_dependence_headwinds = {
        str(code)
        for code in (revenue_dependence.get("revenue_dependence_headwind_signals") or [])
        if str(code).strip()
    }

    # ── Build recovery support signals ────────────────────────────────────────
    recovery_support_signals: list[str] = []
    if cycle_resilience_present:
        recovery_support_signals.append(SIG_CYCLE_RESILIENCE_PRESENT)
    if balance_sheet_optionality_present or downside_support_type in {
        "BALANCE_SHEET_SUPPORT", "ASSET_SUPPORT"
    }:
        recovery_support_signals.append(SIG_BALANCE_SHEET_OPTIONALITY_PRESENT)
    if real_support_present or downside_support_type == "EARNINGS_POWER_SUPPORT":
        recovery_support_signals.append(SIG_EARNINGS_POWER_SUPPORT_PRESENT)
    if trough_valuation_risk or cyclical_trough:
        recovery_support_signals.append(SIG_TROUGH_EARNINGS_RISK_PRESENT)
    if gross_margin_durable:
        recovery_support_signals.append(SIG_GROSS_MARGIN_DURABILITY_PRESENT)
    if owner_earnings_stable:
        recovery_support_signals.append(SIG_OWNER_EARNINGS_STABILITY_PRESENT)
    if temporary_weakness:
        recovery_support_signals.append(SIG_LOW_IMPAIRMENT_RISK)
    if mos_present or mos_real:
        recovery_support_signals.append(SIG_MOS_PRESENT)

    # ── Build recovery headwind signals ───────────────────────────────────────
    recovery_headwind_signals: list[str] = []
    if cyclical_peak:
        recovery_headwind_signals.append(SIG_PEAK_EARNINGS_RISK)
    if low_confidence:
        recovery_headwind_signals.append(SIG_LOW_CONFIDENCE_VALUE)
    if high_fragility:
        recovery_headwind_signals.append(SIG_HIGH_FRAGILITY)
    if integrity_issue:
        recovery_headwind_signals.append(SIG_INTEGRITY_WARNING)
    if probable_impairment:
        recovery_headwind_signals.append(SIG_PROBABLE_IMPAIRMENT)
    if clear_impairment:
        recovery_headwind_signals.append(SIG_CLEAR_IMPAIRMENT)
    if impairment_unassessed:
        recovery_headwind_signals.append(SIG_IMPAIRMENT_RISK_UNKNOWN)
    if structurally_weak or owner_earnings_weak:
        recovery_headwind_signals.append(SIG_STRUCTURALLY_WEAK_ECONOMICS)
    if evidence_blocked or evidence_insufficient:
        recovery_headwind_signals.append(SIG_MISSING_EVIDENCE)
    if mos_unassessable:
        recovery_headwind_signals.append(SIG_MOS_UNASSESSABLE)
    if revenue_dependence_class == HIGH_REVENUE_DEPENDENCE_RISK:
        recovery_headwind_signals.append("HIGH_REVENUE_DEPENDENCE_HEADWIND")
        if {
            "SINGLE_CUSTOMER_CONCENTRATION",
            "TOP_CUSTOMER_DOMINANCE",
            "NARROW_CHANNEL_DEPENDENCE",
            "NARROW_END_MARKET_DEPENDENCE",
        } & revenue_dependence_headwinds:
            recovery_headwind_signals.append("RECOVERY_DEPENDS_ON_NARROW_REVENUE_BASE")
    elif revenue_dependence_class == REVENUE_DEPENDENCE_UNKNOWN:
        recovery_headwind_signals.append("REVENUE_DEPENDENCE_UNKNOWN")

    # ── Classification logic ──────────────────────────────────────────────────
    reason_codes: list[str] = []

    # NORMALIZATION_CREDIBILITY_UNKNOWN: evidence too thin or missing
    if evidence_blocked:
        credibility_class = NORMALIZATION_CREDIBILITY_UNKNOWN
        reason_codes.append(REASON_EVIDENCE_INSUFFICIENT)
        if not price_ok:
            reason_codes.append(REASON_EVIDENCE_MISSING_PRICE)
        if not facts_ok:
            reason_codes.append(REASON_EVIDENCE_MISSING_FACTS)

    # LOW: impairment dominant — normalization is not credible without recovery evidence
    elif impairment_dominant:
        credibility_class = LOW_NORMALIZATION_CREDIBILITY
        reason_codes.append(REASON_IMPAIRMENT_DOMINANT)
        if clear_impairment:
            reason_codes.append(REASON_CLEAR_IMPAIRMENT_PRESENT)
        elif probable_impairment:
            reason_codes.append(REASON_PROBABLE_IMPAIRMENT_PRESENT)

    # LOW: cyclical but at peak — wrong direction for normalization
    elif cyclical_peak:
        credibility_class = LOW_NORMALIZATION_CREDIBILITY
        reason_codes.append(REASON_CYCLICAL_PEAK_WRONG_DIRECTION)
        if high_fragility:
            reason_codes.append(REASON_FRAGILITY_HIGH)

    # HIGH: strong cyclical trough + resilience + real support + sufficient evidence
    elif (
        cyclical_trough
        and cycle_resilience_present
        and real_support_present
        and not impairment_dominant
        and evidence_sufficient
        and not high_fragility
    ):
        credibility_class = HIGH_NORMALIZATION_CREDIBILITY
        reason_codes.append(REASON_CYCLICAL_TROUGH_SUPPORTED)
        reason_codes.append(REASON_CYCLE_RESILIENCE_CONFIRMED)
        if balance_sheet_optionality_present:
            reason_codes.append(REASON_BALANCE_SHEET_SUPPORT)
        if gross_margin_durable:
            reason_codes.append(REASON_GROSS_MARGIN_DURABLE)
        if owner_earnings_stable:
            reason_codes.append(REASON_OWNER_EARNINGS_STABLE)

    # HIGH: cyclical trough + at least one resilience signal + adequate evidence
    elif (
        cyclical_trough
        and (cycle_resilience_present or balance_sheet_optionality_present)
        and not impairment_dominant
        and evidence_sufficient
        and not high_fragility
        and not (low_confidence and not real_support_present)
    ):
        credibility_class = HIGH_NORMALIZATION_CREDIBILITY
        reason_codes.append(REASON_CYCLICAL_TROUGH_SUPPORTED)
        if cycle_resilience_present:
            reason_codes.append(REASON_CYCLE_RESILIENCE_CONFIRMED)
        if balance_sheet_optionality_present:
            reason_codes.append(REASON_BALANCE_SHEET_SUPPORT)

    # MODERATE: cyclical/temporary with some support but mixed confidence
    elif (
        (cyclical_trough or temporary_weakness)
        and (cycle_resilience_present or real_support_present or balance_sheet_optionality_present)
        and not impairment_dominant
    ):
        credibility_class = MODERATE_NORMALIZATION_CREDIBILITY
        reason_codes.append(REASON_PARTIAL_SUPPORT_ONLY)
        if cyclical_trough:
            reason_codes.append(REASON_CYCLICAL_TROUGH_SUPPORTED)
        if cycle_resilience_present:
            reason_codes.append(REASON_CYCLE_RESILIENCE_CONFIRMED)
        if evidence_insufficient:
            reason_codes.append(REASON_EVIDENCE_INSUFFICIENT)
        if high_fragility:
            reason_codes.append(REASON_FRAGILITY_HIGH)
        if integrity_issue:
            reason_codes.append(REASON_INTEGRITY_ISSUE)

    # MODERATE: cyclical business not at peak with some support
    elif (
        cyclical_active
        and not cyclical_peak
        and not impairment_dominant
        and (real_support_present or cycle_resilience_present)
        and not high_fragility
    ):
        credibility_class = MODERATE_NORMALIZATION_CREDIBILITY
        reason_codes.append(REASON_PARTIAL_SUPPORT_ONLY)
        if cycle_resilience_present:
            reason_codes.append(REASON_CYCLE_RESILIENCE_CONFIRMED)

    # MODERATE: cyclical value type with adequate base
    elif (
        is_cyclical_value_type
        and not impairment_dominant
        and not high_fragility
        and not evidence_blocked
    ):
        credibility_class = MODERATE_NORMALIZATION_CREDIBILITY
        reason_codes.append(REASON_PARTIAL_SUPPORT_ONLY)
        reason_codes.append(REASON_EARNINGS_POWER_ANCHORED)

    # LOW: structurally weak or fragile without cyclical context
    elif (
        structurally_weak
        or (low_confidence and high_fragility)
        or (owner_earnings_weak and not cyclical_trough)
    ):
        credibility_class = LOW_NORMALIZATION_CREDIBILITY
        if structurally_weak:
            reason_codes.append(REASON_NO_SUPPORT_FOR_CLAIM)
        if low_confidence and high_fragility:
            reason_codes.append(REASON_FRAGILITY_HIGH)
            reason_codes.append(REASON_LOW_CONFIDENCE)
        if owner_earnings_weak and not cyclical_trough:
            reason_codes.append(REASON_NO_CYCLICAL_CONTEXT)

    # LOW: no measurable support for normalization claim
    elif (
        not cyclical_active
        and not temporary_weakness
        and not real_support_present
        and not cycle_resilience_present
        and owner_earnings_weak
    ):
        credibility_class = LOW_NORMALIZATION_CREDIBILITY
        reason_codes.append(REASON_NO_CYCLICAL_CONTEXT)
        reason_codes.append(REASON_NO_SUPPORT_FOR_CLAIM)

    # UNKNOWN: evidence insufficient or signals conflict
    elif evidence_insufficient and not cyclical_trough:
        credibility_class = NORMALIZATION_CREDIBILITY_UNKNOWN
        reason_codes.append(REASON_EVIDENCE_INSUFFICIENT)

    # UNKNOWN fallback: mixed/conflicting signals
    else:
        credibility_class = NORMALIZATION_CREDIBILITY_UNKNOWN
        reason_codes.append(REASON_SIGNALS_CONFLICT)

    if impairment_unassessed:
        reason_codes.append(REASON_IMPAIRMENT_CLASS_UNKNOWN)

    if revenue_dependence_class == HIGH_REVENUE_DEPENDENCE_RISK:
        reason_codes.append("HIGH_REVENUE_DEPENDENCE_HEADWIND")
        if credibility_class == HIGH_NORMALIZATION_CREDIBILITY:
            credibility_class = MODERATE_NORMALIZATION_CREDIBILITY
            reason_codes.append(REASON_PARTIAL_SUPPORT_ONLY)
        elif credibility_class == MODERATE_NORMALIZATION_CREDIBILITY:
            credibility_class = LOW_NORMALIZATION_CREDIBILITY
            reason_codes.append(REASON_SIGNALS_CONFLICT)
    elif revenue_dependence_class == REVENUE_DEPENDENCE_UNKNOWN:
        reason_codes.append("REVENUE_DEPENDENCE_UNKNOWN")

    # ── Primary normalization caution ─────────────────────────────────────────
    if credibility_class == HIGH_NORMALIZATION_CREDIBILITY:
        primary_normalization_caution = CAUTION_APPEARS_CREDIBLE
    elif credibility_class == MODERATE_NORMALIZATION_CREDIBILITY:
        primary_normalization_caution = CAUTION_PARTIALLY_SUPPORTED
    elif credibility_class == LOW_NORMALIZATION_CREDIBILITY:
        if evidence_blocked:
            primary_normalization_caution = CAUTION_BLOCKED_BY_EVIDENCE
        else:
            primary_normalization_caution = CAUTION_TOO_THEORETICAL
    elif credibility_class == NORMALIZATION_CREDIBILITY_UNKNOWN:
        if evidence_blocked or evidence_insufficient:
            primary_normalization_caution = CAUTION_BLOCKED_BY_EVIDENCE
        else:
            primary_normalization_caution = CAUTION_UNCLEAR
    else:
        primary_normalization_caution = CAUTION_UNCLEAR

    # ── Support summary ───────────────────────────────────────────────────────
    normalization_support_summary = _build_support_summary(
        credibility_class=credibility_class,
        cyclical_trough=cyclical_trough,
        cycle_resilience_present=cycle_resilience_present,
        balance_sheet_optionality_present=balance_sheet_optionality_present,
        gross_margin_durable=gross_margin_durable,
        owner_earnings_stable=owner_earnings_stable,
        real_support_present=real_support_present,
        impairment_dominant=impairment_dominant,
        clear_impairment=clear_impairment,
        probable_impairment=probable_impairment,
        structurally_weak=structurally_weak,
        evidence_blocked=evidence_blocked,
        evidence_insufficient=evidence_insufficient,
        high_fragility=high_fragility,
        low_confidence=low_confidence,
        cyclical_peak=cyclical_peak,
    )

    derived_from = _collect_derived_from(
        cyc, imp, evs, conf, integrity, intrinsic, oe_quality, intangible, value_type, revenue_dependence
    )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "normalization_credibility_class": credibility_class,
        "normalization_credibility_reason_codes": _dedupe_refs(reason_codes),
        "recovery_support_signals": _dedupe_refs(recovery_support_signals),
        "recovery_headwind_signals": _dedupe_refs(recovery_headwind_signals),
        "primary_normalization_caution": primary_normalization_caution,
        "normalization_support_summary": normalization_support_summary,
        "derived_from": derived_from,
        "generated_at": utc_now_iso(),
    }


def _build_support_summary(
    *,
    credibility_class: str,
    cyclical_trough: bool,
    cycle_resilience_present: bool,
    balance_sheet_optionality_present: bool,
    gross_margin_durable: bool,
    owner_earnings_stable: bool,
    real_support_present: bool,
    impairment_dominant: bool,
    clear_impairment: bool,
    probable_impairment: bool,
    structurally_weak: bool,
    evidence_blocked: bool,
    evidence_insufficient: bool,
    high_fragility: bool,
    low_confidence: bool,
    cyclical_peak: bool,
) -> str:
    if evidence_blocked:
        return "blocked by missing evidence (price or facts unavailable)"
    if impairment_dominant:
        label = "clear impairment" if clear_impairment else "probable impairment"
        return f"weak due to {label} — normalization not credible without evidence of recovery"
    if cyclical_peak:
        return "weak due to cyclical peak position — wrong direction for normalization support"
    if credibility_class == HIGH_NORMALIZATION_CREDIBILITY:
        parts: list[str] = []
        if cyclical_trough:
            parts.append("cyclical trough")
        if cycle_resilience_present:
            parts.append("cycle resilience")
        if balance_sheet_optionality_present:
            parts.append("balance sheet optionality")
        if gross_margin_durable:
            parts.append("gross margin durability")
        if owner_earnings_stable:
            parts.append("owner earnings stability")
        if real_support_present and not any("support" in p for p in parts):
            parts.append("earnings power support")
        return "supported by " + " + ".join(parts) if parts else "supported — credible normalization case with real support"
    if credibility_class == MODERATE_NORMALIZATION_CREDIBILITY:
        support_parts: list[str] = []
        if cyclical_trough:
            support_parts.append("cyclical trough")
        if cycle_resilience_present:
            support_parts.append("cycle resilience")
        if balance_sheet_optionality_present:
            support_parts.append("balance sheet")
        weakness_parts: list[str] = []
        if evidence_insufficient:
            weakness_parts.append("evidence gap")
        if high_fragility:
            weakness_parts.append("high fragility")
        if low_confidence:
            weakness_parts.append("low confidence")
        support_text = " + ".join(support_parts) if support_parts else "partial support"
        weakness_text = " + ".join(weakness_parts) if weakness_parts else "mixed signals"
        return f"mixed case ({support_text}); limited by {weakness_text}"
    if credibility_class == LOW_NORMALIZATION_CREDIBILITY:
        if structurally_weak:
            return "weak due to structurally weak economics — normalization purely theoretical"
        if high_fragility and low_confidence:
            return "weak due to high fragility + low confidence — insufficient underwriting base"
        return "weak — normalization mostly theoretical without measurable supporting evidence"
    # UNKNOWN
    if evidence_insufficient:
        return "unknown — insufficient evidence to assess normalization credibility"
    return "unknown — conflicting signals prevent credibility assessment"


def write_normalization_credibility_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Write normalization_credibility.json artifact for a universe or sector run."""
    cfg = cfg or get_config()
    del cfg

    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("normalization_credibility_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("normalization_credibility_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    counts_by_class: dict[str, int] = {}
    counts_by_caution: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

    for ticker in sorted({str(t or "").strip().upper() for t in tickers if str(t or "").strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            # Compute on the fly from scoreboard row
            score_row = next(
                (
                    r
                    for r in (scoreboard_rows or [])
                    if isinstance(r, dict) and str(r.get("ticker") or "").strip().upper() == ticker
                ),
                {},
            )
            price_status = str(
                score_row.get("price_status", UNKNOWN)
            )
            facts_status = str(
                (score_row.get("facts_status") or (score_row.get("cyclical_normalization_detail") or {}).get("as_of_date") or UNKNOWN)
            )
            # Get facts status from cyclical_normalization_detail if not on row directly
            _cyc_detail = score_row.get("cyclical_normalization_detail")
            _imp_detail = score_row.get("impairment_classification_detail")
            _evs_detail = score_row.get("evidence_sufficiency_detail")
            _conf_detail = score_row.get("valuation_confidence_detail")
            _int_detail = score_row.get("valuation_integrity_detail")
            _intrinsic_detail = score_row.get("intrinsic_discipline_detail")
            _oe_detail = score_row.get("owner_earnings_quality_detail")
            _intangible_detail = score_row.get("intangible_economics_detail")
            _vt_detail = score_row.get("value_type_detail")
            detail = compute_normalization_credibility(
                ticker=ticker,
                as_of_date=as_of_date,
                cyclical_normalization_payload=_cyc_detail if isinstance(_cyc_detail, dict) else score_row,
                impairment_classification_payload=_imp_detail if isinstance(_imp_detail, dict) else score_row,
                evidence_sufficiency_payload=_evs_detail if isinstance(_evs_detail, dict) else score_row,
                valuation_confidence_payload=_conf_detail if isinstance(_conf_detail, dict) else score_row,
                valuation_integrity_payload=_int_detail if isinstance(_int_detail, dict) else score_row,
                intrinsic_payload=_intrinsic_detail if isinstance(_intrinsic_detail, dict) else score_row,
                owner_quality_payload=_oe_detail if isinstance(_oe_detail, dict) else score_row,
                intangible_payload=_intangible_detail if isinstance(_intangible_detail, dict) else score_row,
                value_type_payload=_vt_detail if isinstance(_vt_detail, dict) else score_row,
                revenue_dependence_payload=score_row.get("revenue_dependence_detail")
                if isinstance(score_row.get("revenue_dependence_detail"), dict)
                else {},
                price_status=score_row.get("price_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
            )

        credibility_class = str(detail.get("normalization_credibility_class") or NORMALIZATION_CREDIBILITY_UNKNOWN)
        caution = str(detail.get("primary_normalization_caution") or CAUTION_UNCLEAR)
        counts_by_class[credibility_class] = counts_by_class.get(credibility_class, 0) + 1
        counts_by_caution[caution] = counts_by_caution.get(caution, 0) + 1
        for code in (detail.get("normalization_credibility_reason_codes") or []):
            token = str(code or "").strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1

        rows.append({
            "ticker": ticker,
            "normalization_credibility_class": credibility_class,
            "normalization_credibility_reason_codes": list(detail.get("normalization_credibility_reason_codes") or []),
            "recovery_support_signals": list(detail.get("recovery_support_signals") or []),
            "recovery_headwind_signals": list(detail.get("recovery_headwind_signals") or []),
            "primary_normalization_caution": caution,
            "normalization_support_summary": str(detail.get("normalization_support_summary") or ""),
            "derived_from": list(detail.get("derived_from") or []),
        })

    def _class_rows(cls: str) -> list[dict[str, Any]]:
        return [r for r in rows if str(r.get("normalization_credibility_class") or "") == cls]

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "counts_by_normalization_credibility_class": dict(
            sorted(counts_by_class.items(), key=lambda kv: (CREDIBILITY_ORDER.get(kv[0], 99), kv[0]))
        ),
        "counts_by_primary_normalization_caution": dict(
            sorted(counts_by_caution.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "top_10_high_normalization_credibility": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("normalization_credibility_reason_codes") or [])}
            for r in _class_rows(HIGH_NORMALIZATION_CREDIBILITY)[:10]
        ],
        "top_10_low_normalization_credibility": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("normalization_credibility_reason_codes") or [])}
            for r in _class_rows(LOW_NORMALIZATION_CREDIBILITY)[:10]
        ],
        "top_10_normalization_blocked_by_evidence": [
            {"ticker": str(r.get("ticker") or ""), "caution": str(r.get("primary_normalization_caution") or "")}
            for r in rows
            if str(r.get("primary_normalization_caution") or "") == CAUTION_BLOCKED_BY_EVIDENCE
        ][:10],
        "most_common_reason_codes": [
            {"reason_code": str(k), "count": int(v)}
            for k, v in sorted(reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        ][:15],
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    return payload


def _normalization_credibility_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "normalization_credibility.json",
        cfg.sectors_dir / run_id / "normalization_credibility.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_normalization_credibility(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    """Open and summarize the normalization_credibility.json artifact for a run."""
    path = _normalization_credibility_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "normalization_credibility_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    rows = [r for r in (payload.get("rows") or []) if isinstance(r, dict)]

    def _class_rows(cls: str) -> list[dict[str, Any]]:
        return [r for r in rows if str(r.get("normalization_credibility_class") or "") == cls]

    reason_data = payload.get("most_common_reason_codes") or []

    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_normalization_credibility_class": (
            payload.get("counts_by_normalization_credibility_class")
            if isinstance(payload.get("counts_by_normalization_credibility_class"), dict)
            else {}
        ),
        "counts_by_primary_normalization_caution": (
            payload.get("counts_by_primary_normalization_caution")
            if isinstance(payload.get("counts_by_primary_normalization_caution"), dict)
            else {}
        ),
        "top_high_normalization_credibility": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("normalization_credibility_reason_codes") or [])}
            for r in _class_rows(HIGH_NORMALIZATION_CREDIBILITY)[: max(1, int(top_n))]
        ],
        "top_low_normalization_credibility": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("normalization_credibility_reason_codes") or [])}
            for r in _class_rows(LOW_NORMALIZATION_CREDIBILITY)[: max(1, int(top_n))]
        ],
        "top_normalization_blocked_by_evidence": (
            payload.get("top_10_normalization_blocked_by_evidence") or []
        )[: max(1, int(top_n))],
        "most_common_normalization_credibility_reason_codes": [
            {"reason_code": str(item.get("reason_code") or ""), "count": int(item.get("count") or 0)}
            for item in reason_data
            if isinstance(item, dict)
        ][: max(1, int(top_n))],
        "normalization_credibility_path": str(path),
    }
