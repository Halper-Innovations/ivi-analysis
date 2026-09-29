"""Tests for app.valuation.impairment_classification"""
from __future__ import annotations


from app.valuation.impairment_classification import (
    CAUTION_LOW_QUALITY_ECONOMICS,
    CAUTION_POSSIBLE_CYCLE_DISTORTION,
    CAUTION_POSSIBLE_EVIDENCE_GAP,
    CAUTION_POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT,
    CLEAR_IMPAIRMENT,
    EVIDENCE_DEGRADED_NOT_ASSESSABLE,
    IMPAIRMENT_UNKNOWN,
    PROBABLE_IMPAIRMENT,
    STRUCTURALLY_WEAK_NOT_IMPAIRED,
    TEMPORARY_WEAKNESS,
    compute_impairment_classification,
)


def _base_intrinsic(support_type: str = "LIMITED") -> dict:
    return {"downside_support_type": support_type, "mos_classification": "MOS_UNKNOWN"}


def _base_evidence(mos_status: str = "MOS_UNKNOWN", sufficiency: str = "PARTIAL_FOR_MOS") -> dict:
    return {
        "mos_assessment_status": mos_status,
        "evidence_sufficiency_class": sufficiency,
    }


def _base_confidence(cls: str = "MEDIUM_CONFIDENCE", fragility: str = "LOW_FRAGILITY", support_count: int = 2) -> dict:
    return {
        "valuation_confidence_class": cls,
        "valuation_fragility_status": fragility,
        "valuation_support_count": support_count,
    }


def _base_integrity(cls: str = "INTEGRITY_OK") -> dict:
    return {"valuation_integrity_class": cls}


def _base_oe_quality(total: float = 6.0, reason_codes: list | None = None) -> dict:
    return {
        "oe_quality_total": total,
        "oe_quality_reason_codes": reason_codes or [],
        "capital_allocation_score": 5.0,
    }


def _base_intangible(cycle_resilience: float = 2.0, total: float = 5.0) -> dict:
    return {
        "cycle_resilience_score": cycle_resilience,
        "intangible_economics_total": total,
    }


def _base_cyclical(profile: str = "LOW_CYCLICALITY", position: str = "NORMAL_EARNINGS", risk: str = "CYCLE_RISK_LOW") -> dict:
    return {
        "cyclical_profile_class": profile,
        "cycle_position_class": position,
        "cyclical_valuation_risk_class": risk,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. CLEAR_IMPAIRMENT
# ──────────────────────────────────────────────────────────────────────────────

def test_clear_impairment_persistent_negative_oe_with_no_mos():
    result = compute_impairment_classification(
        "TEST",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("LIMITED"),
        evidence_sufficiency_payload=_base_evidence("MOS_CONFIRMED_ABSENT", "SUFFICIENT_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("LOW_CONFIDENCE"),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        owner_quality_payload=_base_oe_quality(1.5, ["NEGATIVE_OWNER_EARNINGS_SERIES"]),
        intangible_payload=_base_intangible(1.0),
        cyclical_normalization_payload=_base_cyclical(),
        price_status="OK",
        facts_status="OK",
        fail_due_to_economic_weakness=True,
        primary_fail_domain="ECONOMICS",
    )
    assert result["impairment_class_primary"] == CLEAR_IMPAIRMENT
    assert result["primary_underwriting_caution"] == CAUTION_POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT
    assert len(result["impairment_class_reason_codes"]) > 0


def test_clear_impairment_high_leverage_no_support_integrity_suspect():
    result = compute_impairment_classification(
        "BADCO",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("LIMITED"),
        evidence_sufficiency_payload=_base_evidence("MOS_CONFIRMED_ABSENT", "SUFFICIENT_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("LOW_CONFIDENCE", "HIGH_FRAGILITY"),
        valuation_integrity_payload=_base_integrity("INTEGRITY_SUSPECT"),
        owner_quality_payload=_base_oe_quality(1.0, ["DEBT_ACCUMULATION"]),
        intangible_payload=_base_intangible(0.5),
        cyclical_normalization_payload=_base_cyclical(),
        price_status="OK",
        facts_status="OK",
        fail_due_to_economic_weakness=True,
        primary_fail_domain="ECONOMICS",
    )
    assert result["impairment_class_primary"] == CLEAR_IMPAIRMENT
    assert CAUTION_POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT == result["primary_underwriting_caution"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. PROBABLE_IMPAIRMENT
# ──────────────────────────────────────────────────────────────────────────────

def test_probable_impairment_no_mos_low_confidence_high_fragility():
    """Hits the first PROBABLE_IMPAIRMENT branch: no_mos + low_confidence + quality_weak + high_fragility."""
    result = compute_impairment_classification(
        "WEAKCO",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("LIMITED"),
        # PARTIAL_FOR_MOS: evidence_gap=False, not CLEAR_IMPAIRMENT via evidence
        evidence_sufficiency_payload=_base_evidence("MOS_CONFIRMED_ABSENT", "PARTIAL_FOR_MOS"),
        # LOW_CONFIDENCE + HIGH_FRAGILITY → quality_weak branch with high_fragility
        valuation_confidence_payload=_base_confidence("LOW_CONFIDENCE", "HIGH_FRAGILITY"),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        # oe_quality=3.5: quality_weak=True but quality_very_weak=False → no persistent_negative_oe
        owner_quality_payload=_base_oe_quality(3.5, []),
        intangible_payload=_base_intangible(1.0),
        cyclical_normalization_payload=_base_cyclical(),
        price_status="OK",
        facts_status="OK",
        fail_due_to_missing_evidence=False,
        fail_due_to_economic_weakness=False,
    )
    assert result["impairment_class_primary"] == PROBABLE_IMPAIRMENT
    assert result["primary_underwriting_caution"] in {
        CAUTION_POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT,
        CAUTION_LOW_QUALITY_ECONOMICS,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. TEMPORARY_WEAKNESS
# ──────────────────────────────────────────────────────────────────────────────

def test_temporary_weakness_cyclical_trough_with_resilience():
    result = compute_impairment_classification(
        "CYCCO",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("EARNINGS"),
        evidence_sufficiency_payload=_base_evidence("MOS_CONFIRMED_PRESENT", "SUFFICIENT_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("MEDIUM_CONFIDENCE", "LOW_FRAGILITY", 3),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        owner_quality_payload=_base_oe_quality(6.0),
        intangible_payload=_base_intangible(4.0),  # cycle_resilience >= 3.0
        cyclical_normalization_payload=_base_cyclical(
            profile="CLEARLY_CYCLICAL",
            position="DEPRESSED_RELATIVE_TO_NORMAL",
            risk="TROUGH_EARNINGS_RISK",
        ),
        price_status="OK",
        facts_status="OK",
        fail_due_to_missing_evidence=False,
        fail_due_to_economic_weakness=False,
    )
    assert result["impairment_class_primary"] == TEMPORARY_WEAKNESS
    assert result["primary_underwriting_caution"] == CAUTION_POSSIBLE_CYCLE_DISTORTION


# ──────────────────────────────────────────────────────────────────────────────
# 4. EVIDENCE_DEGRADED_NOT_ASSESSABLE
# ──────────────────────────────────────────────────────────────────────────────

def test_evidence_degraded_not_assessable_missing_price():
    result = compute_impairment_classification(
        "NODATACO",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("LIMITED"),
        evidence_sufficiency_payload=_base_evidence("MOS_UNASSESSABLE", "INSUFFICIENT_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("LOW_CONFIDENCE"),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        owner_quality_payload=_base_oe_quality(5.5),
        intangible_payload=_base_intangible(3.0),
        cyclical_normalization_payload=_base_cyclical(),
        price_status="MISSING",
        facts_status="OK",
        fail_due_to_missing_evidence=True,
        fail_due_to_economic_weakness=False,
    )
    assert result["impairment_class_primary"] == EVIDENCE_DEGRADED_NOT_ASSESSABLE
    assert result["primary_underwriting_caution"] == CAUTION_POSSIBLE_EVIDENCE_GAP


def test_evidence_degraded_does_not_collapse_to_not_investable_automatically():
    """EVIDENCE_DEGRADED must not automatically become CLEAR_IMPAIRMENT."""
    result = compute_impairment_classification(
        "GAPPYCO",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("BALANCE_SHEET"),
        evidence_sufficiency_payload=_base_evidence("MOS_UNASSESSABLE", "INSUFFICIENT_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("LOW_CONFIDENCE"),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        owner_quality_payload=_base_oe_quality(6.5),  # quality is ok
        intangible_payload=_base_intangible(3.5),
        cyclical_normalization_payload=_base_cyclical(),
        price_status="MISSING",
        facts_status="OK",
        fail_due_to_missing_evidence=True,
        fail_due_to_economic_weakness=False,  # NOT economic weakness
    )
    # Should NOT be CLEAR_IMPAIRMENT just because price is missing
    assert result["impairment_class_primary"] != CLEAR_IMPAIRMENT
    assert result["impairment_class_primary"] in {
        EVIDENCE_DEGRADED_NOT_ASSESSABLE,
        TEMPORARY_WEAKNESS,
        STRUCTURALLY_WEAK_NOT_IMPAIRED,
        IMPAIRMENT_UNKNOWN,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. STRUCTURALLY_WEAK_NOT_IMPAIRED
# ──────────────────────────────────────────────────────────────────────────────

def test_structurally_weak_not_impaired_quality_weak_but_no_impairment_signal():
    result = compute_impairment_classification(
        "WEAKECO",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("BALANCE_SHEET"),
        evidence_sufficiency_payload=_base_evidence("MOS_WEAK", "PARTIAL_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("MEDIUM_CONFIDENCE"),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        owner_quality_payload=_base_oe_quality(3.5),  # weak but not very_weak
        intangible_payload=_base_intangible(1.5),
        cyclical_normalization_payload=_base_cyclical(),
        price_status="OK",
        facts_status="OK",
        fail_due_to_missing_evidence=False,
        fail_due_to_economic_weakness=False,
    )
    assert result["impairment_class_primary"] == STRUCTURALLY_WEAK_NOT_IMPAIRED
    assert result["primary_underwriting_caution"] == CAUTION_LOW_QUALITY_ECONOMICS


# ──────────────────────────────────────────────────────────────────────────────
# 6. Investment readiness integration
# ──────────────────────────────────────────────────────────────────────────────

def test_clear_impairment_adds_structural_blocker_in_investment_readiness():
    from app.valuation.investment_readiness import (
        BLOCKER_IMPAIRMENT_CONFIRMED,
        compute_investment_readiness,
    )

    imp_payload = compute_impairment_classification(
        "IMPAIRED",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("LIMITED"),
        evidence_sufficiency_payload=_base_evidence("MOS_CONFIRMED_ABSENT", "SUFFICIENT_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("LOW_CONFIDENCE"),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        owner_quality_payload=_base_oe_quality(1.5, ["NEGATIVE_OWNER_EARNINGS_SERIES"]),
        intangible_payload=_base_intangible(1.0),
        cyclical_normalization_payload=_base_cyclical(),
        price_status="OK",
        facts_status="OK",
        fail_due_to_economic_weakness=True,
        primary_fail_domain="ECONOMICS",
    )
    assert imp_payload["impairment_class_primary"] == CLEAR_IMPAIRMENT

    readiness = compute_investment_readiness(
        "IMPAIRED",
        "2024-01-01",
        impairment_classification_payload=imp_payload,
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    blockers = readiness.get("blocker_stack_all") or []
    assert BLOCKER_IMPAIRMENT_CONFIRMED in blockers
    assert readiness.get("blocker_stack_structural") is True


def test_temporary_weakness_does_not_add_structural_blocker():
    from app.valuation.investment_readiness import (
        BLOCKER_IMPAIRMENT_CONFIRMED,
        compute_investment_readiness,
    )

    imp_payload = compute_impairment_classification(
        "CYCLECO",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("EARNINGS"),
        evidence_sufficiency_payload=_base_evidence("MOS_CONFIRMED_PRESENT", "SUFFICIENT_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("MEDIUM_CONFIDENCE", "LOW_FRAGILITY", 3),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        owner_quality_payload=_base_oe_quality(6.0),
        intangible_payload=_base_intangible(4.0),
        cyclical_normalization_payload=_base_cyclical(
            profile="CLEARLY_CYCLICAL",
            position="DEPRESSED_RELATIVE_TO_NORMAL",
            risk="TROUGH_EARNINGS_RISK",
        ),
        price_status="OK",
        facts_status="OK",
    )
    assert imp_payload["impairment_class_primary"] == TEMPORARY_WEAKNESS

    readiness = compute_investment_readiness(
        "CYCLECO",
        "2024-01-01",
        impairment_classification_payload=imp_payload,
        price_status="OK",
        facts_status="OK",
        shares_status="OK",
        fcf_status="OK",
    )
    blockers = readiness.get("blocker_stack_all") or []
    assert BLOCKER_IMPAIRMENT_CONFIRMED not in blockers


# ──────────────────────────────────────────────────────────────────────────────
# 7. Cyclical / value-type interaction
# ──────────────────────────────────────────────────────────────────────────────

def test_cyclical_trough_without_resilience_falls_back_to_probable_or_structurally_weak():
    """Cyclical trough with NO cycle resilience should not get TEMPORARY_WEAKNESS."""
    result = compute_impairment_classification(
        "WEAKCY",
        "2024-01-01",
        intrinsic_payload=_base_intrinsic("LIMITED"),
        evidence_sufficiency_payload=_base_evidence("MOS_CONFIRMED_ABSENT", "PARTIAL_FOR_MOS"),
        valuation_confidence_payload=_base_confidence("LOW_CONFIDENCE"),
        valuation_integrity_payload=_base_integrity("INTEGRITY_OK"),
        owner_quality_payload=_base_oe_quality(2.5, ["NEGATIVE_OWNER_EARNINGS_SERIES"]),
        intangible_payload=_base_intangible(0.5),  # low resilience
        cyclical_normalization_payload=_base_cyclical(
            profile="CLEARLY_CYCLICAL",
            position="DEPRESSED_RELATIVE_TO_NORMAL",
        ),
        price_status="OK",
        facts_status="OK",
        fail_due_to_economic_weakness=False,
    )
    # Without resilience and with negative OE + no MOS, should not be TEMPORARY_WEAKNESS
    assert result["impairment_class_primary"] != TEMPORARY_WEAKNESS


# ──────────────────────────────────────────────────────────────────────────────
# 8. Memo pack section
# ──────────────────────────────────────────────────────────────────────────────

def test_memo_pack_includes_business_impairment_section():
    """build_investment_memo dict should include business_impairment_classification."""
    from app.universe.memo_pack import build_investment_memo

    # Minimal shortlist_row with impairment fields populated
    shortlist_row = {
        "ticker": "TESTTICKER",
        "sector": "Technology",
        "appearances_count": 1,
        "latest_value_gate_status": "PASS",
        "as_of_date": "2024-01-01",
        "impairment_class_primary": TEMPORARY_WEAKNESS,
        "primary_underwriting_caution": CAUTION_POSSIBLE_CYCLE_DISTORTION,
        "impairment_class_reason_codes": ["CYCLICAL_TROUGH"],
        "impairment_classification_detail": {
            "impairment_class_primary": TEMPORARY_WEAKNESS,
            "primary_underwriting_caution": CAUTION_POSSIBLE_CYCLE_DISTORTION,
            "impairment_class_reason_codes": ["CYCLICAL_TROUGH"],
            "weakness_source_flags": {"weakness_source_cyclical": True},
            "support_signals": ["CYCLE_RESILIENCE_PRESENT"],
            "rebuttal_signals": [],
        },
    }
    memo = build_investment_memo(
        "TESTTICKER",
        universe_run_id="test_run",
        batch_run_id="test_batch",
        sources={"shortlist_row": shortlist_row},
    )
    assert "business_impairment_classification" in memo
    bic = memo["business_impairment_classification"]
    assert bic["impairment_class_primary"] == TEMPORARY_WEAKNESS
    assert bic["primary_underwriting_caution"] == CAUTION_POSSIBLE_CYCLE_DISTORTION


# ──────────────────────────────────────────────────────────────────────────────
# 9. Promotion / escalation visibility without overriding FAIL
# ──────────────────────────────────────────────────────────────────────────────

def test_impairment_headwind_added_to_risk_flags_without_overriding_fail():
    """IMPAIRMENT_HEADWIND should appear in risk_flags but not change a FAIL gate."""
    from app.universe.promotion import _build_risk_flags, classify_priority_lane

    row = {
        "impairment_class_primary": CLEAR_IMPAIRMENT,
        "latest_value_gate_status": "FAIL",
        "appearances_count": 2,
        "investment_readiness_class": "NOT_INVESTABLE",
        "fail_due_to_economic_weakness": True,
        "facts_blocker_terminal": False,
        "facts_blocker_retryable": False,
        "facts_blocker_partial_usable": False,
        "facts_blocker_class": "FACTS_OK",
    }
    flags = _build_risk_flags(row)
    assert "IMPAIRMENT_HEADWIND" in flags
    # Lane should still be DEPRIORITIZED (FAIL)
    lane = classify_priority_lane(row)
    assert lane in {"LANE_4_DEPRIORITIZED", "LANE_3_MONITOR"}


def test_temporary_weakness_support_added_to_strength_flags():
    from app.universe.promotion import _build_strength_flags

    row = {
        "impairment_class_primary": TEMPORARY_WEAKNESS,
        "latest_value_gate_status": "PASS",
        "appearances_count": 1,
        "investment_readiness_class": "INVESTABLE_NOW",
    }
    flags = _build_strength_flags(row)
    assert "TEMPORARY_WEAKNESS_SUPPORT" in flags
