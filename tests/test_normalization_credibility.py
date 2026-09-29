"""Tests for normalization / recovery credibility module."""
from __future__ import annotations

import json

import pytest

from app.valuation.normalization_credibility import (
    HIGH_NORMALIZATION_CREDIBILITY,
    MODERATE_NORMALIZATION_CREDIBILITY,
    LOW_NORMALIZATION_CREDIBILITY,
    NORMALIZATION_CREDIBILITY_UNKNOWN,
    CAUTION_APPEARS_CREDIBLE,
    CAUTION_PARTIALLY_SUPPORTED,
    CAUTION_TOO_THEORETICAL,
    CAUTION_BLOCKED_BY_EVIDENCE,
    SIG_CYCLE_RESILIENCE_PRESENT,
    SIG_TROUGH_EARNINGS_RISK_PRESENT,
    SIG_CLEAR_IMPAIRMENT,
    SIG_PROBABLE_IMPAIRMENT,
    SIG_MISSING_EVIDENCE,
    SIG_IMPAIRMENT_RISK_UNKNOWN,
    SIG_LOW_IMPAIRMENT_RISK,
    compute_normalization_credibility,
    write_normalization_credibility_for_run,
    open_normalization_credibility,
)


def _cyc(profile="CLEARLY_CYCLICAL", position="DEPRESSED_RELATIVE_TO_NORMAL", risk="TROUGH_EARNINGS_RISK"):
    return {
        "cyclical_profile_class": profile,
        "cycle_position_class": position,
        "cyclical_valuation_risk_class": risk,
        "derived_from": ["companyfacts/AAPL"],
    }


def _imp(cls="TEMPORARY_WEAKNESS"):
    return {
        "impairment_class_primary": cls,
        "impairment_support_signals": [],
        "impairment_rebuttal_signals": ["CYCLE_RESILIENCE_PRESENT"],
        "derived_from": ["companyfacts/AAPL"],
    }


def _evs(sufficiency="SUFFICIENT_FOR_MOS", mos="MOS_CONFIRMED_PRESENT"):
    return {
        "evidence_sufficiency_class": sufficiency,
        "mos_assessment_status": mos,
        "derived_from": ["companyfacts/AAPL", "prices/AAPL"],
    }


def _conf(confidence="HIGH_CONFIDENCE", fragility="LOW_FRAGILITY"):
    return {
        "valuation_confidence_class": confidence,
        "valuation_fragility_status": fragility,
        "derived_from": [],
    }


def _intangible(cycle_res=4.0, gmd=4.0, bso=3.5):
    return {
        "cycle_resilience_score": cycle_res,
        "gross_margin_durability_score": gmd,
        "balance_sheet_optionality_score": bso,
        "intangible_economics_total": 9.0,
        "derived_from": [],
    }


def _oe_quality(total=8.0, stability=4.0):
    return {
        "oe_quality_total": total,
        "owner_earnings_stability_score": stability,
        "derived_from": [],
    }


def _intrinsic(support="EARNINGS_POWER_SUPPORT", mos_cls="ADEQUATE_MARGIN_OF_SAFETY"):
    return {
        "downside_support_type": support,
        "mos_classification": mos_cls,
        "derived_from": [],
    }


class TestHighNormalizationCredibility:
    def test_cyclical_trough_with_full_support(self):
        result = compute_normalization_credibility(
            "AAPL", "2025-01-01",
            cyclical_normalization_payload=_cyc(),
            impairment_classification_payload=_imp("TEMPORARY_WEAKNESS"),
            evidence_sufficiency_payload=_evs(),
            valuation_confidence_payload=_conf(),
            intangible_payload=_intangible(),
            owner_quality_payload=_oe_quality(),
            intrinsic_payload=_intrinsic(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["normalization_credibility_class"] == HIGH_NORMALIZATION_CREDIBILITY
        assert result["primary_normalization_caution"] == CAUTION_APPEARS_CREDIBLE
        assert SIG_CYCLE_RESILIENCE_PRESENT in result["recovery_support_signals"]
        assert SIG_TROUGH_EARNINGS_RISK_PRESENT in result["recovery_support_signals"]
        assert len(result["normalization_credibility_reason_codes"]) > 0

    def test_cyclical_trough_balance_sheet_support(self):
        result = compute_normalization_credibility(
            "MSFT", "2025-01-01",
            cyclical_normalization_payload=_cyc(),
            impairment_classification_payload=_imp("TEMPORARY_WEAKNESS"),
            evidence_sufficiency_payload=_evs(),
            valuation_confidence_payload=_conf("MEDIUM_CONFIDENCE", "MODERATE_FRAGILITY"),
            intangible_payload=_intangible(cycle_res=3.5, gmd=2.0, bso=4.0),
            owner_quality_payload=_oe_quality(total=7.0),
            intrinsic_payload=_intrinsic("BALANCE_SHEET_SUPPORT"),
            price_status="OK",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == HIGH_NORMALIZATION_CREDIBILITY


class TestModerateNormalizationCredibility:
    def test_cyclical_trough_mixed_support(self):
        result = compute_normalization_credibility(
            "GOOGL", "2025-01-01",
            cyclical_normalization_payload=_cyc(),
            impairment_classification_payload=_imp("TEMPORARY_WEAKNESS"),
            evidence_sufficiency_payload=_evs("PARTIAL_FOR_MOS", "MOS_WEAK"),
            valuation_confidence_payload=_conf("LOW_CONFIDENCE", "MODERATE_FRAGILITY"),
            intangible_payload=_intangible(cycle_res=3.0, gmd=2.0, bso=2.0),
            owner_quality_payload=_oe_quality(total=6.0, stability=3.0),
            price_status="OK",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == MODERATE_NORMALIZATION_CREDIBILITY
        assert result["primary_normalization_caution"] == CAUTION_PARTIALLY_SUPPORTED

    def test_cyclical_active_not_at_trough_with_resilience(self):
        result = compute_normalization_credibility(
            "META", "2025-01-01",
            cyclical_normalization_payload=_cyc(position="NEAR_NORMAL", risk="MID_CYCLE_REASONABLE"),
            impairment_classification_payload=_imp("IMPAIRMENT_UNKNOWN"),
            evidence_sufficiency_payload=_evs("PARTIAL_FOR_MOS", "MOS_WEAK"),
            valuation_confidence_payload=_conf("MEDIUM_CONFIDENCE", "LOW_FRAGILITY"),
            intangible_payload=_intangible(cycle_res=4.0),
            intrinsic_payload=_intrinsic("EARNINGS_POWER_SUPPORT"),
            price_status="OK",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == MODERATE_NORMALIZATION_CREDIBILITY


class TestLowNormalizationCredibility:
    def test_impairment_dominant_clear(self):
        result = compute_normalization_credibility(
            "FAIL1", "2025-01-01",
            cyclical_normalization_payload=_cyc(profile="LOW_CYCLICALITY", position="NEAR_NORMAL"),
            impairment_classification_payload=_imp("CLEAR_IMPAIRMENT"),
            evidence_sufficiency_payload=_evs("SUFFICIENT_FOR_MOS", "MOS_CONFIRMED_ABSENT"),
            valuation_confidence_payload=_conf("LOW_CONFIDENCE", "HIGH_FRAGILITY"),
            price_status="OK",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == LOW_NORMALIZATION_CREDIBILITY
        assert result["primary_normalization_caution"] == CAUTION_TOO_THEORETICAL
        assert SIG_CLEAR_IMPAIRMENT in result["recovery_headwind_signals"]

    def test_probable_impairment_headwind(self):
        result = compute_normalization_credibility(
            "FAIL2", "2025-01-01",
            impairment_classification_payload=_imp("PROBABLE_IMPAIRMENT"),
            evidence_sufficiency_payload=_evs("PARTIAL_FOR_MOS", "MOS_CONFIRMED_ABSENT"),
            valuation_confidence_payload=_conf("LOW_CONFIDENCE", "HIGH_FRAGILITY"),
            price_status="OK",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == LOW_NORMALIZATION_CREDIBILITY
        assert SIG_PROBABLE_IMPAIRMENT in result["recovery_headwind_signals"]

    def test_cyclical_peak_wrong_direction(self):
        result = compute_normalization_credibility(
            "PEAK1", "2025-01-01",
            cyclical_normalization_payload=_cyc(
                profile="CLEARLY_CYCLICAL",
                position="ELEVATED_RELATIVE_TO_NORMAL",
                risk="PEAK_EARNINGS_RISK"
            ),
            impairment_classification_payload=_imp("IMPAIRMENT_UNKNOWN"),
            price_status="OK",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == LOW_NORMALIZATION_CREDIBILITY


class TestUnknownOrBlockedByEvidence:
    def test_missing_price_blocks_credibility(self):
        result = compute_normalization_credibility(
            "NOPRICE", "2025-01-01",
            cyclical_normalization_payload=_cyc(),
            impairment_classification_payload=_imp("TEMPORARY_WEAKNESS"),
            price_status="MISSING",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == NORMALIZATION_CREDIBILITY_UNKNOWN
        assert result["primary_normalization_caution"] == CAUTION_BLOCKED_BY_EVIDENCE
        assert SIG_MISSING_EVIDENCE in result["recovery_headwind_signals"]

    def test_missing_facts_blocks_credibility(self):
        result = compute_normalization_credibility(
            "NOFACTS", "2025-01-01",
            cyclical_normalization_payload=_cyc(),
            price_status="OK",
            facts_status="MISSING",
        )
        assert result["normalization_credibility_class"] == NORMALIZATION_CREDIBILITY_UNKNOWN
        assert result["primary_normalization_caution"] == CAUTION_BLOCKED_BY_EVIDENCE

    def test_insufficient_evidence_without_cyclical_context(self):
        result = compute_normalization_credibility(
            "NOTICKS", "2025-01-01",
            evidence_sufficiency_payload=_evs("INSUFFICIENCY_FOR_MOS", "MOS_UNKNOWN"),
            cyclical_normalization_payload={"cyclical_profile_class": "CYCLICALITY_UNKNOWN"},
            price_status="OK",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == NORMALIZATION_CREDIBILITY_UNKNOWN


class TestImpairmentHeadwindSuppressesCredibility:
    def test_clear_impairment_suppresses_cyclical_signals(self):
        """Even with strong cyclical signals, clear impairment forces LOW credibility."""
        result = compute_normalization_credibility(
            "IMPAIRED", "2025-01-01",
            cyclical_normalization_payload=_cyc(),
            impairment_classification_payload=_imp("CLEAR_IMPAIRMENT"),
            evidence_sufficiency_payload=_evs("SUFFICIENT_FOR_MOS", "MOS_CONFIRMED_PRESENT"),
            valuation_confidence_payload=_conf("HIGH_CONFIDENCE", "LOW_FRAGILITY"),
            intangible_payload=_intangible(cycle_res=5.0),
            price_status="OK",
            facts_status="OK",
        )
        assert result["normalization_credibility_class"] == LOW_NORMALIZATION_CREDIBILITY
        assert SIG_CLEAR_IMPAIRMENT in result["recovery_headwind_signals"]


class TestWriteNormalizationCredibilityForRun:
    def test_write_creates_artifact(self, tmp_path):
        output_path = tmp_path / "normalization_credibility.json"
        rows = [
            {
                "ticker": "AAPL",
                "normalization_credibility_detail": {
                    "ticker": "AAPL",
                    "as_of_date": "2025-01-01",
                    "normalization_credibility_class": HIGH_NORMALIZATION_CREDIBILITY,
                    "normalization_credibility_reason_codes": ["CYCLICAL_TROUGH_WITH_RESILIENCE"],
                    "recovery_support_signals": [SIG_CYCLE_RESILIENCE_PRESENT],
                    "recovery_headwind_signals": [],
                    "primary_normalization_caution": CAUTION_APPEARS_CREDIBLE,
                    "normalization_support_summary": "supported by cycle resilience",
                    "derived_from": [],
                },
            },
            {
                "ticker": "GOOGL",
                "normalization_credibility_detail": {
                    "ticker": "GOOGL",
                    "as_of_date": "2025-01-01",
                    "normalization_credibility_class": LOW_NORMALIZATION_CREDIBILITY,
                    "normalization_credibility_reason_codes": ["IMPAIRMENT_SIGNALS_DOMINANT"],
                    "recovery_support_signals": [],
                    "recovery_headwind_signals": [SIG_CLEAR_IMPAIRMENT],
                    "primary_normalization_caution": CAUTION_TOO_THEORETICAL,
                    "normalization_support_summary": "weak due to clear impairment",
                    "derived_from": [],
                },
            },
        ]
        result = write_normalization_credibility_for_run(
            run_id="test_run",
            as_of_date="2025-01-01",
            tickers=["AAPL", "GOOGL"],
            output_path=output_path,
            scoreboard_rows=rows,
        )
        assert output_path.exists()
        payload = json.loads(output_path.read_text())
        assert payload["ticker_count"] == 2
        assert HIGH_NORMALIZATION_CREDIBILITY in payload["counts_by_normalization_credibility_class"]
        assert LOW_NORMALIZATION_CREDIBILITY in payload["counts_by_normalization_credibility_class"]
        assert len(payload["top_10_high_normalization_credibility"]) == 1
        assert len(payload["top_10_low_normalization_credibility"]) == 1


class TestOpenNormalizationCredibility:
    def test_open_missing_returns_missing_status(self):
        result = open_normalization_credibility(run_id="nonexistent_run_xyz")
        assert result["status"] == "MISSING"

    def test_open_existing_returns_ok_status(self, tmp_path, monkeypatch):
        payload = {
            "run_id": "test",
            "as_of_date": "2025-01-01",
            "ticker_count": 1,
            "counts_by_normalization_credibility_class": {HIGH_NORMALIZATION_CREDIBILITY: 1},
            "counts_by_primary_normalization_caution": {CAUTION_APPEARS_CREDIBLE: 1},
            "top_10_high_normalization_credibility": [{"ticker": "AAPL", "reason_codes": []}],
            "top_10_low_normalization_credibility": [],
            "top_10_normalization_blocked_by_evidence": [],
            "most_common_reason_codes": [],
            "rows": [
                {
                    "ticker": "AAPL",
                    "normalization_credibility_class": HIGH_NORMALIZATION_CREDIBILITY,
                    "normalization_credibility_reason_codes": [],
                    "recovery_support_signals": [],
                    "recovery_headwind_signals": [],
                    "primary_normalization_caution": CAUTION_APPEARS_CREDIBLE,
                    "normalization_support_summary": "supported",
                    "derived_from": [],
                }
            ],
            "generated_at": "2025-01-01T00:00:00Z",
        }
        run_path = tmp_path / "universe" / "test_run" / "normalization_credibility.json"
        run_path.parent.mkdir(parents=True)
        run_path.write_text(json.dumps(payload))

        from app.config import get_config
        original_config = get_config()

        monkeypatch.setattr(
            "app.valuation.normalization_credibility._normalization_credibility_path",
            lambda run_id: run_path,
        )
        result = open_normalization_credibility(run_id="test_run")
        assert result["status"] == "OK"
        assert result["ticker_count"] == 1


def _credibility(**overrides):
    kwargs = dict(
        cyclical_normalization_payload=_cyc(),
        evidence_sufficiency_payload=_evs(),
        valuation_confidence_payload=_conf(),
        intangible_payload=_intangible(),
        intrinsic_payload=_intrinsic(),
        price_status="OK",
        facts_status="OK",
    )
    kwargs.update(overrides)
    return compute_normalization_credibility("IMP", "2025-01-01", **kwargs)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        "not a dict",
        _imp("IMPAIRMENT_UNKNOWN"),
        _imp("EVIDENCE_DEGRADED_NOT_ASSESSABLE"),
        _imp("A_CLASS_THIS_MODULE_DOES_NOT_KNOW"),
        {"impairment_class_primary": ""},
        {"impairment_class_primary": None},
    ],
)
def test_absent_or_unclassifiable_impairment_is_not_evidence_of_low_impairment_risk(payload):
    result = _credibility(impairment_classification_payload=payload)
    assert SIG_LOW_IMPAIRMENT_RISK not in result["recovery_support_signals"]
    assert SIG_IMPAIRMENT_RISK_UNKNOWN in result["recovery_headwind_signals"]
    assert "IMPAIRMENT_CLASS_UNKNOWN" in result["normalization_credibility_reason_codes"]


def test_a_temporary_weakness_classification_is_the_evidence_for_low_impairment_risk():
    result = _credibility(impairment_classification_payload=_imp("TEMPORARY_WEAKNESS"))
    assert SIG_LOW_IMPAIRMENT_RISK in result["recovery_support_signals"]
    assert SIG_IMPAIRMENT_RISK_UNKNOWN not in result["recovery_headwind_signals"]
    assert "IMPAIRMENT_CLASS_UNKNOWN" not in result["normalization_credibility_reason_codes"]


@pytest.mark.parametrize(
    "cls", ["CLEAR_IMPAIRMENT", "PROBABLE_IMPAIRMENT", "STRUCTURALLY_WEAK_NOT_IMPAIRED"]
)
def test_an_assessed_impaired_or_weak_class_is_neither_low_risk_nor_unknown(cls):
    result = _credibility(impairment_classification_payload=_imp(cls))
    assert SIG_LOW_IMPAIRMENT_RISK not in result["recovery_support_signals"]
    assert SIG_IMPAIRMENT_RISK_UNKNOWN not in result["recovery_headwind_signals"]
    assert "IMPAIRMENT_CLASS_UNKNOWN" not in result["normalization_credibility_reason_codes"]
