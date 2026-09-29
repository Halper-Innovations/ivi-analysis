"""
Business Impairment vs Temporary Weakness Classification v1

Deterministic classification layer that distinguishes:
1. True business impairment (CLEAR_IMPAIRMENT, PROBABLE_IMPAIRMENT)
2. Temporary cyclical weakness (TEMPORARY_WEAKNESS)
3. Temporary evidence degradation (EVIDENCE_DEGRADED_NOT_ASSESSABLE)
4. Structurally weak economics, not quite impaired (STRUCTURALLY_WEAK_NOT_IMPAIRED)
5. Cannot classify responsibly (IMPAIRMENT_UNKNOWN)

Doctrine:
- No technical indicators, no ML, no narrative scoring, no macro forecasting
- No "cycle will turn soon" claims — no aggressive optimism
- UNKNOWN remains UNKNOWN with explicit reason codes
- True impairment must not be softened into "temporary"
- Temporary weakness must not be mislabeled as structural when evidence says otherwise
- This layer describes underwriting reality, not predicted recovery
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.valuation.balance_sheet_stress import (
    BALANCE_SHEET_STRESS_UNKNOWN,
    HIGH_BALANCE_SHEET_STRESS,
    LOW_BALANCE_SHEET_STRESS,
)
from app.valuation.cyclical_normalization import (
    CLEARLY_CYCLICAL,
    CYCLICALITY_UNKNOWN,
    DEPRESSED_RELATIVE_TO_NORMAL,
    MODERATELY_CYCLICAL,
    TROUGH_EARNINGS_RISK,
)
from app.valuation.evidence_sufficiency import (
    MOS_CONFIRMED_ABSENT,
    MOS_CONFIRMED_PRESENT,
    MOS_UNASSESSABLE,
    MOS_UNKNOWN_STATUS,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_UNKNOWN,
)
from app.valuation.intrinsic_discipline import (
    SUPPORT_ASSET,
    SUPPORT_BALANCE_SHEET,
    SUPPORT_EARNINGS,
    SUPPORT_UNKNOWN,
)
from app.valuation.valuation_confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_UNKNOWN,
    FRAGILITY_HIGH,
)
from app.valuation.valuation_integrity import (
    INTEGRITY_SUSPECT,
    INTEGRITY_UNKNOWN,
    INTEGRITY_WARNING,
)


UNKNOWN = "UNKNOWN"
OK = "OK"

# ─── Impairment class constants ───────────────────────────────────────────────

CLEAR_IMPAIRMENT = "CLEAR_IMPAIRMENT"
PROBABLE_IMPAIRMENT = "PROBABLE_IMPAIRMENT"
TEMPORARY_WEAKNESS = "TEMPORARY_WEAKNESS"
EVIDENCE_DEGRADED_NOT_ASSESSABLE = "EVIDENCE_DEGRADED_NOT_ASSESSABLE"
STRUCTURALLY_WEAK_NOT_IMPAIRED = "STRUCTURALLY_WEAK_NOT_IMPAIRED"
IMPAIRMENT_UNKNOWN = "IMPAIRMENT_UNKNOWN"

# Ordering for ranking (best → worst impairment context)
IMPAIRMENT_ORDER = {
    TEMPORARY_WEAKNESS: 0,
    EVIDENCE_DEGRADED_NOT_ASSESSABLE: 1,
    STRUCTURALLY_WEAK_NOT_IMPAIRED: 2,
    PROBABLE_IMPAIRMENT: 3,
    CLEAR_IMPAIRMENT: 4,
    IMPAIRMENT_UNKNOWN: 5,
}

# ─── Reason codes ─────────────────────────────────────────────────────────────

REASON_ECONOMIC_WEAKNESS_CONFIRMED = "ECONOMIC_WEAKNESS_CONFIRMED"
REASON_NO_MOS_CONFIRMED = "NO_MOS_CONFIRMED"
REASON_LOW_CONFIDENCE = "LOW_CONFIDENCE"
REASON_PERSISTENT_NEGATIVE_OWNER_EARNINGS = "PERSISTENT_NEGATIVE_OWNER_EARNINGS"
REASON_INTEGRITY_SUSPECT = "INTEGRITY_SUSPECT"
REASON_HIGH_LEVERAGE = "HIGH_LEVERAGE"
REASON_QUALITY_WEAKNESS = "QUALITY_WEAKNESS"
REASON_CYCLICAL_TROUGH = "CYCLICAL_TROUGH"
REASON_CYCLICAL_SUPPORT = "CYCLICAL_SUPPORT"
REASON_CYCLE_RESILIENCE_SUPPORT = "CYCLE_RESILIENCE_SUPPORT"
REASON_EVIDENCE_GAP = "EVIDENCE_GAP"
REASON_EVIDENCE_MISSING_PRICE = "EVIDENCE_MISSING_PRICE"
REASON_EVIDENCE_MISSING_FACTS = "EVIDENCE_MISSING_FACTS"
REASON_EVIDENCE_MISSING_SHARES = "EVIDENCE_MISSING_SHARES"
REASON_STRUCTURALLY_WEAK = "STRUCTURALLY_WEAK"
REASON_HIGH_FRAGILITY = "HIGH_FRAGILITY"
REASON_INTEGRITY_WARNING = "INTEGRITY_WARNING"
REASON_INSUFFICIENT_FOR_CLASSIFICATION = "INSUFFICIENT_FOR_CLASSIFICATION"

# ─── Signal constants (support / rebuttal) ────────────────────────────────────

SIG_PERSISTENT_NEGATIVE_OWNER_EARNINGS = "PERSISTENT_NEGATIVE_OWNER_EARNINGS"
SIG_HIGH_LEVERAGE_PRESSURE = "HIGH_LEVERAGE_PRESSURE"
SIG_NO_MOS_CONFIRMED = "NO_MOS_CONFIRMED"
SIG_LOW_CONFIDENCE_VALUE = "LOW_CONFIDENCE_VALUE"
SIG_HIGH_FRAGILITY = "HIGH_FRAGILITY"
SIG_STRUCTURAL_QUALITY_WEAKNESS = "STRUCTURAL_QUALITY_WEAKNESS"
SIG_INTEGRITY_SUSPECT = "INTEGRITY_SUSPECT"

SIG_BALANCE_SHEET_SUPPORT_PRESENT = "BALANCE_SHEET_SUPPORT_PRESENT"
SIG_ASSET_SUPPORT_PRESENT = "ASSET_SUPPORT_PRESENT"
SIG_EARNINGS_POWER_SUPPORT_PRESENT = "EARNINGS_POWER_SUPPORT_PRESENT"
SIG_CYCLE_RESILIENCE_PRESENT = "CYCLE_RESILIENCE_PRESENT"
SIG_TROUGH_EARNINGS_RISK = "TROUGH_EARNINGS_RISK"
SIG_EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
SIG_MOS_UNASSESSABLE = "MOS_UNASSESSABLE"

# ─── Underwriting caution classes ─────────────────────────────────────────────

CAUTION_POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT = "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT"
CAUTION_POSSIBLE_CYCLE_DISTORTION = "POSSIBLE_CYCLE_DISTORTION"
CAUTION_POSSIBLE_EVIDENCE_GAP = "POSSIBLE_EVIDENCE_GAP"
CAUTION_POSSIBLE_DENOMINATOR_PROBLEM = "POSSIBLE_DENOMINATOR_PROBLEM"
CAUTION_LOW_QUALITY_ECONOMICS = "LOW_QUALITY_ECONOMICS"
CAUTION_UNKNOWN = "CAUTION_UNKNOWN"


# ─── Helpers ──────────────────────────────────────────────────────────────────

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
    return _dedupe_refs(refs)


# ─── Core classification function ─────────────────────────────────────────────

def compute_impairment_classification(
    ticker: str,
    as_of_date: str,
    *,
    intrinsic_payload: dict[str, Any] | None = None,
    evidence_sufficiency_payload: dict[str, Any] | None = None,
    valuation_confidence_payload: dict[str, Any] | None = None,
    valuation_integrity_payload: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    balance_sheet_stress_payload: dict[str, Any] | None = None,
    cyclical_normalization_payload: dict[str, Any] | None = None,
    price_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    fcf_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    fail_due_to_missing_evidence: bool = False,
    fail_due_to_economic_weakness: bool = False,
    primary_fail_domain: str | None = None,
    row_derived_from: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """
    Deterministic per-ticker impairment vs temporary weakness classification.

    Returns a structured dict with:
    - impairment_class_primary: one of the six class constants
    - impairment_class_reason_codes: list of why this class was assigned
    - weakness_source_* flags: boolean source decomposition
    - impairment_support_signals: evidence that impairment is real
    - impairment_rebuttal_signals: evidence against impairment
    - primary_underwriting_caution: sober caution label (not a recommendation)
    """
    cfg = cfg or get_config()
    del cfg

    intrinsic_payload = intrinsic_payload if isinstance(intrinsic_payload, dict) else {}
    evidence_sufficiency_payload = (
        evidence_sufficiency_payload if isinstance(evidence_sufficiency_payload, dict) else {}
    )
    valuation_confidence_payload = (
        valuation_confidence_payload if isinstance(valuation_confidence_payload, dict) else {}
    )
    valuation_integrity_payload = (
        valuation_integrity_payload if isinstance(valuation_integrity_payload, dict) else {}
    )
    owner_quality_payload = owner_quality_payload if isinstance(owner_quality_payload, dict) else {}
    intangible_payload = intangible_payload if isinstance(intangible_payload, dict) else {}
    balance_sheet_stress_payload = (
        balance_sheet_stress_payload if isinstance(balance_sheet_stress_payload, dict) else {}
    )
    cyclical_normalization_payload = (
        cyclical_normalization_payload if isinstance(cyclical_normalization_payload, dict) else {}
    )

    ticker_norm = str(ticker or "").strip().upper()
    primary_fail_domain_norm = str(primary_fail_domain or UNKNOWN).upper()

    # ── Extract signals from payloads ──────────────────────────────────────────
    mos_assessment_status = str(
        evidence_sufficiency_payload.get("mos_assessment_status") or MOS_UNKNOWN_STATUS
    ).upper()
    evidence_sufficiency_class = str(
        evidence_sufficiency_payload.get("evidence_sufficiency_class") or SUFFICIENCY_UNKNOWN
    ).upper()

    valuation_confidence_class = str(
        valuation_confidence_payload.get("valuation_confidence_class") or CONFIDENCE_UNKNOWN
    ).upper()
    valuation_fragility_status = str(
        valuation_confidence_payload.get("valuation_fragility_status") or "FRAGILITY_UNKNOWN"
    ).upper()
    support_count = int(valuation_confidence_payload.get("valuation_support_count") or 0)

    valuation_integrity_class = str(
        valuation_integrity_payload.get("valuation_integrity_class") or INTEGRITY_UNKNOWN
    ).upper()

    downside_support_type = str(
        intrinsic_payload.get("downside_support_type") or SUPPORT_UNKNOWN
    ).upper()
    mos_classification = str(intrinsic_payload.get("mos_classification") or UNKNOWN).upper()

    oe_quality_total = owner_quality_payload.get("oe_quality_total", UNKNOWN)
    oe_quality_reason_codes: set[str] = {
        str(code)
        for code in (owner_quality_payload.get("oe_quality_reason_codes") or [])
        if str(code).strip()
    }
    capital_allocation_score = owner_quality_payload.get("capital_allocation_score", UNKNOWN)

    cycle_resilience_score = intangible_payload.get("cycle_resilience_score", UNKNOWN)
    intangible_total = intangible_payload.get("intangible_economics_total", UNKNOWN)

    cyclical_profile_class = str(
        cyclical_normalization_payload.get("cyclical_profile_class") or CYCLICALITY_UNKNOWN
    ).upper()
    cycle_position_class = str(
        cyclical_normalization_payload.get("cycle_position_class") or "CYCLE_POSITION_UNKNOWN"
    ).upper()
    cyclical_valuation_risk_class = str(
        cyclical_normalization_payload.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN"
    ).upper()

    # ── Boolean helper conditions ──────────────────────────────────────────────
    no_mos_confirmed = mos_assessment_status == MOS_CONFIRMED_ABSENT
    mos_real = mos_assessment_status == MOS_CONFIRMED_PRESENT
    mos_unassessable = mos_assessment_status == MOS_UNASSESSABLE

    low_confidence = valuation_confidence_class in {CONFIDENCE_LOW, CONFIDENCE_UNKNOWN}
    high_confidence = valuation_confidence_class in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}
    high_fragility = valuation_fragility_status == FRAGILITY_HIGH
    integrity_suspect = valuation_integrity_class == INTEGRITY_SUSPECT
    integrity_issue = valuation_integrity_class in {INTEGRITY_SUSPECT, INTEGRITY_WARNING}

    support_present = downside_support_type in {SUPPORT_ASSET, SUPPORT_EARNINGS, SUPPORT_BALANCE_SHEET}

    quality_known = _is_num(oe_quality_total)
    quality_very_weak = quality_known and float(oe_quality_total) < 3.0
    quality_weak = quality_known and float(oe_quality_total) < 5.0

    # Persistent negative OE: either reason code or very low quality score
    persistent_negative_oe = (
        "NEGATIVE_OWNER_EARNINGS_SERIES" in oe_quality_reason_codes
        or quality_very_weak
    )

    # High leverage: debt accumulation reason code or very low capital allocation
    high_leverage = (
        "DEBT_ACCUMULATION" in oe_quality_reason_codes
        or (_is_num(capital_allocation_score) and float(capital_allocation_score) <= 1.0)
    )
    balance_sheet_stress_class = str(
        balance_sheet_stress_payload.get("balance_sheet_stress_class") or BALANCE_SHEET_STRESS_UNKNOWN
    ).upper()
    refinancing_risk_class = str(
        balance_sheet_stress_payload.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
    ).upper()
    balance_sheet_headwinds = {
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_headwind_signals") or [])
        if str(code).strip()
    }
    low_balance_sheet_stress = balance_sheet_stress_class == LOW_BALANCE_SHEET_STRESS
    high_balance_sheet_stress = balance_sheet_stress_class == HIGH_BALANCE_SHEET_STRESS
    high_leverage = high_leverage or high_balance_sheet_stress

    # Cycle resilience
    cycle_resilience_present = (
        _is_num(cycle_resilience_score) and float(cycle_resilience_score) >= 3.0
    )

    # Evidence gap: multiple signals
    price_ok = str(price_status or "").upper() == "OK"
    facts_ok = str(facts_status or "").upper() == "OK"
    shares_ok = str(shares_status or "").upper() == "OK"
    evidence_gap = (
        bool(fail_due_to_missing_evidence)
        or primary_fail_domain_norm == "EVIDENCE"
        or evidence_sufficiency_class in {SUFFICIENCY_INSUFFICIENT, SUFFICIENCY_UNKNOWN}
        or not price_ok
        or not facts_ok
    )

    # Cyclical trough: cyclical business at depressed earnings position
    cyclical_trough = (
        cyclical_profile_class in {CLEARLY_CYCLICAL, MODERATELY_CYCLICAL}
        and cycle_position_class == DEPRESSED_RELATIVE_TO_NORMAL
    )

    # Strong economic signal: overrides pure evidence degradation
    strong_economic_signal = (
        bool(fail_due_to_economic_weakness)
        or primary_fail_domain_norm == "ECONOMICS"
        or (persistent_negative_oe and no_mos_confirmed and low_confidence)
        or (integrity_suspect and no_mos_confirmed and low_confidence and not support_present)
    )

    # ── Weakness source decomposition ─────────────────────────────────────────
    weakness_source_evidence = evidence_gap or mos_unassessable
    weakness_source_cyclical = cyclical_profile_class in {CLEARLY_CYCLICAL, MODERATELY_CYCLICAL}
    weakness_source_balance_sheet = high_leverage
    weakness_source_owner_earnings = persistent_negative_oe or quality_very_weak
    weakness_source_valuation = no_mos_confirmed or low_confidence or high_fragility
    weakness_source_quality = quality_weak
    weakness_source_integrity = integrity_issue

    # ── Build impairment support signals (evidence FOR impairment) ─────────────
    impairment_support_signals: list[str] = []
    if persistent_negative_oe:
        impairment_support_signals.append(SIG_PERSISTENT_NEGATIVE_OWNER_EARNINGS)
    if high_leverage:
        impairment_support_signals.append(SIG_HIGH_LEVERAGE_PRESSURE)
    if refinancing_risk_class == "HIGH_REFINANCING_RISK":
        impairment_support_signals.append(SIG_HIGH_LEVERAGE_PRESSURE)
    if no_mos_confirmed:
        impairment_support_signals.append(SIG_NO_MOS_CONFIRMED)
    if valuation_confidence_class == CONFIDENCE_LOW:
        impairment_support_signals.append(SIG_LOW_CONFIDENCE_VALUE)
    if high_fragility:
        impairment_support_signals.append(SIG_HIGH_FRAGILITY)
    if quality_weak:
        impairment_support_signals.append(SIG_STRUCTURAL_QUALITY_WEAKNESS)
    if integrity_suspect:
        impairment_support_signals.append(SIG_INTEGRITY_SUSPECT)

    # ── Build impairment rebuttal signals (evidence AGAINST impairment) ────────
    impairment_rebuttal_signals: list[str] = []
    if downside_support_type == SUPPORT_BALANCE_SHEET:
        impairment_rebuttal_signals.append(SIG_BALANCE_SHEET_SUPPORT_PRESENT)
    if downside_support_type == SUPPORT_ASSET:
        impairment_rebuttal_signals.append(SIG_ASSET_SUPPORT_PRESENT)
    if downside_support_type == SUPPORT_EARNINGS:
        impairment_rebuttal_signals.append(SIG_EARNINGS_POWER_SUPPORT_PRESENT)
    if cycle_resilience_present:
        impairment_rebuttal_signals.append(SIG_CYCLE_RESILIENCE_PRESENT)
    if cyclical_valuation_risk_class == TROUGH_EARNINGS_RISK:
        impairment_rebuttal_signals.append(SIG_TROUGH_EARNINGS_RISK)
    if low_balance_sheet_stress:
        impairment_rebuttal_signals.append(SIG_BALANCE_SHEET_SUPPORT_PRESENT)
    if evidence_sufficiency_class in {SUFFICIENCY_INSUFFICIENT, SUFFICIENCY_UNKNOWN}:
        impairment_rebuttal_signals.append(SIG_EVIDENCE_INSUFFICIENT)
    if mos_unassessable:
        impairment_rebuttal_signals.append(SIG_MOS_UNASSESSABLE)

    # ── Classification logic ───────────────────────────────────────────────────
    # Order: CLEAR_IMPAIRMENT → PROBABLE_IMPAIRMENT → EVIDENCE_DEGRADED →
    #        TEMPORARY_WEAKNESS → STRUCTURALLY_WEAK_NOT_IMPAIRED →
    #        EVIDENCE_DEGRADED (fallback) → IMPAIRMENT_UNKNOWN

    reason_codes: list[str] = []

    # CLEAR_IMPAIRMENT: multiple simultaneous strong signals
    if (
        (bool(fail_due_to_economic_weakness) or primary_fail_domain_norm == "ECONOMICS")
        and no_mos_confirmed
        and low_confidence
    ):
        impairment_class = CLEAR_IMPAIRMENT
        reason_codes = [REASON_ECONOMIC_WEAKNESS_CONFIRMED, REASON_NO_MOS_CONFIRMED, REASON_LOW_CONFIDENCE]
        if high_leverage or "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in balance_sheet_headwinds:
            reason_codes.append(REASON_HIGH_LEVERAGE)
        if quality_weak:
            reason_codes.append(REASON_QUALITY_WEAKNESS)

    elif (
        persistent_negative_oe
        and no_mos_confirmed
        and low_confidence
        and quality_very_weak
    ):
        impairment_class = CLEAR_IMPAIRMENT
        reason_codes = [
            REASON_PERSISTENT_NEGATIVE_OWNER_EARNINGS,
            REASON_NO_MOS_CONFIRMED,
            REASON_LOW_CONFIDENCE,
        ]
        if high_leverage:
            reason_codes.append(REASON_HIGH_LEVERAGE)

    elif (
        integrity_suspect
        and no_mos_confirmed
        and low_confidence
        and (quality_weak or not support_present)
    ):
        impairment_class = CLEAR_IMPAIRMENT
        reason_codes = [REASON_INTEGRITY_SUSPECT, REASON_NO_MOS_CONFIRMED, REASON_LOW_CONFIDENCE]
        if quality_weak:
            reason_codes.append(REASON_QUALITY_WEAKNESS)

    # PROBABLE_IMPAIRMENT: weight of evidence suggests impairment but not as clear
    elif (
        no_mos_confirmed
        and low_confidence
        and quality_weak
        and (high_leverage or integrity_issue or high_fragility)
    ):
        impairment_class = PROBABLE_IMPAIRMENT
        reason_codes = [REASON_NO_MOS_CONFIRMED, REASON_LOW_CONFIDENCE, REASON_QUALITY_WEAKNESS]
        if high_leverage:
            reason_codes.append(REASON_HIGH_LEVERAGE)
        if integrity_issue:
            reason_codes.append(REASON_INTEGRITY_SUSPECT if integrity_suspect else REASON_INTEGRITY_WARNING)
        if high_fragility:
            reason_codes.append(REASON_HIGH_FRAGILITY)

    elif (
        quality_very_weak
        and no_mos_confirmed
        and high_fragility
        and not cyclical_trough
    ):
        impairment_class = PROBABLE_IMPAIRMENT
        reason_codes = [
            REASON_PERSISTENT_NEGATIVE_OWNER_EARNINGS,
            REASON_NO_MOS_CONFIRMED,
            REASON_HIGH_FRAGILITY,
        ]

    elif (
        persistent_negative_oe
        and no_mos_confirmed
        and (high_leverage or integrity_issue)
        and not cyclical_trough
        and not strong_economic_signal
    ):
        impairment_class = PROBABLE_IMPAIRMENT
        reason_codes = [REASON_PERSISTENT_NEGATIVE_OWNER_EARNINGS, REASON_NO_MOS_CONFIRMED]
        if high_leverage:
            reason_codes.append(REASON_HIGH_LEVERAGE)
        if integrity_issue:
            reason_codes.append(REASON_INTEGRITY_SUSPECT if integrity_suspect else REASON_INTEGRITY_WARNING)

    # EVIDENCE_DEGRADED_NOT_ASSESSABLE: evidence gap is primary — no strong economic signal
    elif (
        evidence_gap
        and not strong_economic_signal
        and (mos_unassessable or evidence_sufficiency_class in {SUFFICIENCY_INSUFFICIENT, SUFFICIENCY_UNKNOWN})
    ):
        impairment_class = EVIDENCE_DEGRADED_NOT_ASSESSABLE
        reason_codes = [REASON_EVIDENCE_GAP]
        if not price_ok:
            reason_codes.append(REASON_EVIDENCE_MISSING_PRICE)
        if not facts_ok:
            reason_codes.append(REASON_EVIDENCE_MISSING_FACTS)
        if not shares_ok:
            reason_codes.append(REASON_EVIDENCE_MISSING_SHARES)

    # TEMPORARY_WEAKNESS: cyclical trough with real support present
    elif (
        cyclical_trough
        and (support_present or cycle_resilience_present)
        and not (no_mos_confirmed and low_confidence and quality_weak)
    ):
        impairment_class = TEMPORARY_WEAKNESS
        reason_codes = [REASON_CYCLICAL_TROUGH]
        if support_present:
            reason_codes.append(REASON_CYCLICAL_SUPPORT)
        if cycle_resilience_present:
            reason_codes.append(REASON_CYCLE_RESILIENCE_SUPPORT)

    # STRUCTURALLY_WEAK_NOT_IMPAIRED: weak economics, evidence adequate, not impaired
    elif (
        (quality_weak or no_mos_confirmed or low_confidence)
        and evidence_sufficiency_class not in {SUFFICIENCY_INSUFFICIENT, SUFFICIENCY_UNKNOWN}
        and not evidence_gap
    ):
        impairment_class = STRUCTURALLY_WEAK_NOT_IMPAIRED
        reason_codes = [REASON_STRUCTURALLY_WEAK]
        if no_mos_confirmed:
            reason_codes.append(REASON_NO_MOS_CONFIRMED)
        if low_confidence:
            reason_codes.append(REASON_LOW_CONFIDENCE)
        if quality_weak:
            reason_codes.append(REASON_QUALITY_WEAKNESS)
        if high_fragility:
            reason_codes.append(REASON_HIGH_FRAGILITY)

    # EVIDENCE_DEGRADED fallback: evidence gap prevents responsible judgment
    elif evidence_gap:
        impairment_class = EVIDENCE_DEGRADED_NOT_ASSESSABLE
        reason_codes = [REASON_EVIDENCE_GAP, REASON_INSUFFICIENT_FOR_CLASSIFICATION]

    # IMPAIRMENT_UNKNOWN: insufficient signals to classify
    else:
        impairment_class = IMPAIRMENT_UNKNOWN
        reason_codes = [REASON_INSUFFICIENT_FOR_CLASSIFICATION]
        if not quality_known:
            reason_codes.append(REASON_QUALITY_WEAKNESS)

    # ── Primary underwriting caution ──────────────────────────────────────────
    if impairment_class == CLEAR_IMPAIRMENT:
        primary_underwriting_caution = CAUTION_POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT
    elif impairment_class == PROBABLE_IMPAIRMENT:
        primary_underwriting_caution = CAUTION_POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT
    elif impairment_class == TEMPORARY_WEAKNESS:
        if cyclical_valuation_risk_class == TROUGH_EARNINGS_RISK:
            primary_underwriting_caution = CAUTION_POSSIBLE_CYCLE_DISTORTION
        else:
            primary_underwriting_caution = CAUTION_POSSIBLE_DENOMINATOR_PROBLEM
    elif impairment_class == EVIDENCE_DEGRADED_NOT_ASSESSABLE:
        primary_underwriting_caution = CAUTION_POSSIBLE_EVIDENCE_GAP
    elif impairment_class == STRUCTURALLY_WEAK_NOT_IMPAIRED:
        primary_underwriting_caution = CAUTION_LOW_QUALITY_ECONOMICS
    else:
        primary_underwriting_caution = CAUTION_UNKNOWN

    derived_from = _payload_refs(
        intrinsic_payload,
        evidence_sufficiency_payload,
        valuation_confidence_payload,
        valuation_integrity_payload,
        owner_quality_payload,
        intangible_payload,
        balance_sheet_stress_payload,
        cyclical_normalization_payload,
        row_refs=row_derived_from,
    )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "impairment_class_primary": impairment_class,
        "impairment_class_reason_codes": _dedupe_refs(reason_codes),
        # Weakness source decomposition
        "weakness_source_evidence": bool(weakness_source_evidence),
        "weakness_source_cyclical": bool(weakness_source_cyclical),
        "weakness_source_balance_sheet": bool(weakness_source_balance_sheet),
        "weakness_source_owner_earnings": bool(weakness_source_owner_earnings),
        "weakness_source_valuation": bool(weakness_source_valuation),
        "weakness_source_quality": bool(weakness_source_quality),
        "weakness_source_integrity": bool(weakness_source_integrity),
        # Support / rebuttal signals
        "impairment_support_signals": impairment_support_signals,
        "impairment_rebuttal_signals": impairment_rebuttal_signals,
        # Underwriting caution
        "primary_underwriting_caution": primary_underwriting_caution,
        # Derived from provenance
        "derived_from": derived_from,
        "generated_at": utc_now_iso(),
    }


# ─── Run-scoped artifact writer ───────────────────────────────────────────────

def write_impairment_classification_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Write impairment_classification.json for a universe or sector run."""
    cfg = cfg or get_config()
    del cfg

    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("impairment_classification_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("impairment_classification_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    counts_by_class: dict[str, int] = {}
    counts_by_caution: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

    for ticker in sorted({str(t or "").strip().upper() for t in tickers if str(t or "").strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            score_row = next(
                (
                    r
                    for r in (scoreboard_rows or [])
                    if isinstance(r, dict) and str(r.get("ticker") or "").strip().upper() == ticker
                ),
                {},
            )
            detail = compute_impairment_classification(
                ticker=ticker,
                as_of_date=as_of_date,
                intrinsic_payload=score_row.get("intrinsic_discipline_detail")
                if isinstance(score_row.get("intrinsic_discipline_detail"), dict)
                else score_row,
                evidence_sufficiency_payload=score_row.get("evidence_sufficiency_detail")
                if isinstance(score_row.get("evidence_sufficiency_detail"), dict)
                else score_row,
                valuation_confidence_payload=score_row.get("valuation_confidence_detail")
                if isinstance(score_row.get("valuation_confidence_detail"), dict)
                else score_row,
                valuation_integrity_payload=score_row.get("valuation_integrity_detail")
                if isinstance(score_row.get("valuation_integrity_detail"), dict)
                else score_row,
                owner_quality_payload=score_row.get("owner_earnings_quality_detail")
                if isinstance(score_row.get("owner_earnings_quality_detail"), dict)
                else score_row,
                intangible_payload=score_row.get("intangible_economics_detail")
                if isinstance(score_row.get("intangible_economics_detail"), dict)
                else score_row,
                balance_sheet_stress_payload=score_row.get("balance_sheet_stress_detail")
                if isinstance(score_row.get("balance_sheet_stress_detail"), dict)
                else {},
                cyclical_normalization_payload=score_row.get("cyclical_normalization_detail")
                if isinstance(score_row.get("cyclical_normalization_detail"), dict)
                else score_row,
                price_status=score_row.get("price_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                fcf_status=score_row.get("fcf_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                fail_due_to_missing_evidence=bool(score_row.get("fail_due_to_missing_evidence", False)),
                fail_due_to_economic_weakness=bool(score_row.get("fail_due_to_economic_weakness", False)),
                primary_fail_domain=str(score_row.get("primary_fail_domain") or UNKNOWN),
                row_derived_from=list(score_row.get("derived_from") or []),
            )

        impairment_class = str(detail.get("impairment_class_primary") or IMPAIRMENT_UNKNOWN)
        caution = str(detail.get("primary_underwriting_caution") or CAUTION_UNKNOWN)
        counts_by_class[impairment_class] = counts_by_class.get(impairment_class, 0) + 1
        counts_by_caution[caution] = counts_by_caution.get(caution, 0) + 1
        for code in (detail.get("impairment_class_reason_codes") or []):
            token = str(code or "").strip()
            if not token:
                continue
            reason_counts[token] = reason_counts.get(token, 0) + 1

        rows.append(detail)

    def _class_rows(cls: str) -> list[dict[str, Any]]:
        return [
            r for r in rows if str(r.get("impairment_class_primary") or IMPAIRMENT_UNKNOWN) == cls
        ]

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "counts_by_impairment_class": dict(
            sorted(counts_by_class.items(), key=lambda kv: (IMPAIRMENT_ORDER.get(kv[0], 99), kv[0]))
        ),
        "counts_by_underwriting_caution": dict(
            sorted(counts_by_caution.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "top_10_clear_impairment": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("impairment_class_reason_codes") or [])}
            for r in _class_rows(CLEAR_IMPAIRMENT)[:10]
        ],
        "top_10_temporary_weakness": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("impairment_class_reason_codes") or [])}
            for r in _class_rows(TEMPORARY_WEAKNESS)[:10]
        ],
        "top_10_evidence_degraded": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("impairment_class_reason_codes") or [])}
            for r in _class_rows(EVIDENCE_DEGRADED_NOT_ASSESSABLE)[:10]
        ],
        "impairment_reason_counts": dict(
            sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["impairment_classification_path"] = str(output_path)
    return payload


# ─── Path resolver + open ─────────────────────────────────────────────────────

def _impairment_classification_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "impairment_classification.json",
        cfg.sectors_dir / run_id / "impairment_classification.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_impairment_classification(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    """Open and summarize the impairment_classification.json artifact for a run."""
    path = _impairment_classification_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "impairment_classification_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    rows = [r for r in (payload.get("rows") or []) if isinstance(r, dict)]

    def _class_rows(cls: str) -> list[dict[str, Any]]:
        return [r for r in rows if str(r.get("impairment_class_primary") or IMPAIRMENT_UNKNOWN) == cls]

    reason_counts = payload.get("impairment_reason_counts") or {}

    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_impairment_class": payload.get("counts_by_impairment_class")
        if isinstance(payload.get("counts_by_impairment_class"), dict)
        else {},
        "counts_by_underwriting_caution": payload.get("counts_by_underwriting_caution")
        if isinstance(payload.get("counts_by_underwriting_caution"), dict)
        else {},
        "top_10_clear_impairment": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("impairment_class_reason_codes") or [])}
            for r in _class_rows(CLEAR_IMPAIRMENT)[: max(1, int(top_n))]
        ],
        "top_10_temporary_weakness": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("impairment_class_reason_codes") or [])}
            for r in _class_rows(TEMPORARY_WEAKNESS)[: max(1, int(top_n))]
        ],
        "top_10_evidence_degraded": [
            {"ticker": str(r.get("ticker") or ""), "reason_codes": list(r.get("impairment_class_reason_codes") or [])}
            for r in _class_rows(EVIDENCE_DEGRADED_NOT_ASSESSABLE)[: max(1, int(top_n))]
        ],
        "most_common_reason_codes": [
            {"reason_code": str(k), "count": int(v)}
            for k, v in sorted(reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        ][: max(1, int(top_n))],
        "impairment_classification_path": str(path),
    }
