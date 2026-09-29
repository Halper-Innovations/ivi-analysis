"""Tests for per-share capital allocation discipline module."""
from __future__ import annotations

import json
import pytest

from app.valuation.capital_allocation_discipline import (
    OWNER_FRIENDLY_DISCIPLINED,
    MIXED_CAPITAL_ALLOCATION,
    OWNER_DILUTIVE_OR_DESTRUCTIVE,
    CAPITAL_ALLOCATION_UNKNOWN,
    CAUTION_SUPPORTIVE,
    CAUTION_MIXED,
    CAUTION_HEADWIND,
    CAUTION_UNCLEAR,
    SIG_LOW_DILUTION,
    SIG_HIGH_DILUTION,
    SIG_SHARE_COUNT_DISCIPLINE,
    SIG_SHARE_COUNT_HEADWIND,
    REASON_EVIDENCE_BLOCKED_MISSING_FACTS,
    REASON_EVIDENCE_BLOCKED_MISSING_SHARES,
    REASON_HIGH_DILUTION_DOMINANT,
    REASON_HIGH_DILUTION_OFFSET_BY_QUALITY,
    REASON_LOW_DILUTION_STRONG_QUALITY,
    REASON_VERY_LOW_QUALITY_DESTRUCTIVE,
    REASON_BUYBACKS_QUALITY_CONFIRMED,
    compute_capital_allocation_discipline,
    write_capital_allocation_discipline_for_run,
    open_capital_allocation_discipline,
)


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _oe_quality(
    dilution_rate=0.01,  # fraction: 0.01 = 1% annually
    oe_quality_total=9.0,
    capital_allocation_score=3.5,
    stability=4.0,
    reason_codes=None,
):
    """Build a minimal owner_earnings_quality payload."""
    return {
        "dilution_rate_shares_cagr": dilution_rate,
        "oe_quality_total": oe_quality_total,
        "capital_allocation_score": capital_allocation_score,
        "owner_earnings_stability_score": stability,
        "capital_allocation_reason_codes": reason_codes or [],
        "derived_from": ["companyfacts/AAPL"],
    }


def _intrinsic(support="EARNINGS_POWER_SUPPORT", mos_cls="ADEQUATE_MARGIN_OF_SAFETY"):
    return {
        "downside_support_type": support,
        "mos_classification": mos_cls,
        "derived_from": [],
    }


def _evs(sufficiency="SUFFICIENT_FOR_MOS"):
    return {
        "evidence_sufficiency_class": sufficiency,
        "derived_from": ["companyfacts/AAPL", "prices/AAPL"],
    }


def _conf(confidence="HIGH_CONFIDENCE"):
    return {
        "valuation_confidence_class": confidence,
        "derived_from": [],
    }


# ── Test classes ───────────────────────────────────────────────────────────────

class TestOwnerFriendlyDisciplined:
    def test_low_dilution_strong_quality(self):
        """Low dilution + high quality score → OWNER_FRIENDLY_DISCIPLINED."""
        result = compute_capital_allocation_discipline(
            "AAPL", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.005, oe_quality_total=10.0),
            intrinsic_payload=_intrinsic(),
            evidence_sufficiency_payload=_evs(),
            valuation_confidence_payload=_conf(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == OWNER_FRIENDLY_DISCIPLINED
        assert result["primary_capital_allocation_caution"] == CAUTION_SUPPORTIVE
        assert REASON_LOW_DILUTION_STRONG_QUALITY in result["capital_allocation_discipline_reason_codes"]
        assert SIG_LOW_DILUTION in result["per_share_support_signals"]

    def test_buybacks_with_quality(self):
        """Negative dilution (buybacks) + quality → OWNER_FRIENDLY, BUYBACKS reason code."""
        result = compute_capital_allocation_discipline(
            "MSFT", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=-0.02, oe_quality_total=11.0),
            evidence_sufficiency_payload=_evs(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == OWNER_FRIENDLY_DISCIPLINED
        assert REASON_BUYBACKS_QUALITY_CONFIRMED in result["capital_allocation_discipline_reason_codes"]
        assert SIG_SHARE_COUNT_DISCIPLINE in result["per_share_support_signals"]

    def test_shareholder_friendly_reason_code_boosts(self):
        """SHAREHOLDER_FRIENDLY in capital_allocation_reason_codes should add support."""
        result = compute_capital_allocation_discipline(
            "GOOGL", "2025-01-01",
            owner_quality_payload=_oe_quality(
                dilution_rate=0.008,
                oe_quality_total=9.0,
                reason_codes=["SHAREHOLDER_FRIENDLY"],
            ),
            evidence_sufficiency_payload=_evs(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == OWNER_FRIENDLY_DISCIPLINED


class TestMixedCapitalAllocation:
    def test_high_dilution_offset_by_quality(self):
        """High dilution but strong quality → MIXED."""
        result = compute_capital_allocation_discipline(
            "META", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.05, oe_quality_total=8.0),
            evidence_sufficiency_payload=_evs(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == MIXED_CAPITAL_ALLOCATION
        assert result["primary_capital_allocation_caution"] == CAUTION_MIXED
        assert REASON_HIGH_DILUTION_OFFSET_BY_QUALITY in result["capital_allocation_discipline_reason_codes"]
        assert SIG_HIGH_DILUTION in result["per_share_headwind_signals"]

    def test_moderate_dilution_with_quality(self):
        """Moderate dilution (1-3%) + quality present → MIXED."""
        result = compute_capital_allocation_discipline(
            "NVDA", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.02, oe_quality_total=6.0),
            evidence_sufficiency_payload=_evs(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == MIXED_CAPITAL_ALLOCATION
        assert SIG_SHARE_COUNT_HEADWIND in result["per_share_headwind_signals"]

    def test_low_dilution_quality_unknown(self):
        """Low dilution but quality unknown → MIXED (lean friendly)."""
        result = compute_capital_allocation_discipline(
            "ORCL", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.005, oe_quality_total="UNKNOWN"),
            evidence_sufficiency_payload=_evs(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == MIXED_CAPITAL_ALLOCATION


class TestOwnerDilutiveOrDestructive:
    def test_high_dilution_no_quality(self):
        """High dilution (>3%) + no quality → OWNER_DILUTIVE_OR_DESTRUCTIVE."""
        result = compute_capital_allocation_discipline(
            "DILUTE1", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.08, oe_quality_total=1.0),
            evidence_sufficiency_payload=_evs(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == OWNER_DILUTIVE_OR_DESTRUCTIVE
        assert result["primary_capital_allocation_caution"] == CAUTION_HEADWIND
        assert REASON_VERY_LOW_QUALITY_DESTRUCTIVE in result["capital_allocation_discipline_reason_codes"]

    def test_excess_dilution_reason_code_forces_destructive(self):
        """EXCESS_DILUTION in reason codes forces HIGH dilution path."""
        result = compute_capital_allocation_discipline(
            "DILUTE2", "2025-01-01",
            owner_quality_payload=_oe_quality(
                dilution_rate=0.01,  # Would be low, but...
                oe_quality_total=1.5,
                reason_codes=["EXCESS_DILUTION"],
            ),
            evidence_sufficiency_payload=_evs(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == OWNER_DILUTIVE_OR_DESTRUCTIVE

    def test_very_low_quality_is_destructive(self):
        """Very low oe_quality_total (<2.0) forces OWNER_DILUTIVE_OR_DESTRUCTIVE."""
        result = compute_capital_allocation_discipline(
            "WEAK1", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.005, oe_quality_total=1.5),
            evidence_sufficiency_payload=_evs(),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == OWNER_DILUTIVE_OR_DESTRUCTIVE
        assert REASON_VERY_LOW_QUALITY_DESTRUCTIVE in result["capital_allocation_discipline_reason_codes"]


class TestUnknownOrEvidenceBlocked:
    def test_missing_facts_blocks_classification(self):
        """facts_status != OK → CAPITAL_ALLOCATION_UNKNOWN with EVIDENCE_BLOCKED_MISSING_FACTS."""
        result = compute_capital_allocation_discipline(
            "NOFACTS", "2025-01-01",
            owner_quality_payload=_oe_quality(),
            price_status="OK",
            facts_status="MISSING",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == CAPITAL_ALLOCATION_UNKNOWN
        assert result["primary_capital_allocation_caution"] == CAUTION_UNCLEAR
        assert REASON_EVIDENCE_BLOCKED_MISSING_FACTS in result["capital_allocation_discipline_reason_codes"]

    def test_missing_shares_with_no_dilution_info_blocks(self):
        """shares_status != OK and no dilution data → CAPITAL_ALLOCATION_UNKNOWN."""
        result = compute_capital_allocation_discipline(
            "NOSHARES", "2025-01-01",
            owner_quality_payload={"oe_quality_total": 9.0, "derived_from": []},
            price_status="OK",
            facts_status="OK",
            shares_status="MISSING",
        )
        assert result["capital_allocation_discipline_class"] == CAPITAL_ALLOCATION_UNKNOWN
        assert REASON_EVIDENCE_BLOCKED_MISSING_SHARES in result["capital_allocation_discipline_reason_codes"]

    def test_all_unknown_inputs(self):
        """No owner quality, no dilution data → CAPITAL_ALLOCATION_UNKNOWN."""
        result = compute_capital_allocation_discipline(
            "EMPTY", "2025-01-01",
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == CAPITAL_ALLOCATION_UNKNOWN


class TestDilutionRateConversion:
    def test_dilution_rate_converted_from_fraction(self):
        """dilution_rate_shares_cagr is a fraction; multiply by 100 for pct thresholds."""
        # 0.04 fraction = 4% → above the 3% HIGH_DILUTION threshold
        result = compute_capital_allocation_discipline(
            "FRAC1", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.04, oe_quality_total=9.0),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        # 4% dilution + high quality → MIXED (high dilution offset by quality)
        assert result["capital_allocation_discipline_class"] == MIXED_CAPITAL_ALLOCATION
        assert result["dilution_rate_annual_pct"] == pytest.approx(4.0)

    def test_low_dilution_below_threshold(self):
        """0.005 fraction = 0.5% → below 1% threshold → low dilution."""
        result = compute_capital_allocation_discipline(
            "FRAC2", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.005, oe_quality_total=8.5),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == OWNER_FRIENDLY_DISCIPLINED
        assert result["dilution_rate_annual_pct"] == pytest.approx(0.5)


class TestWriteCapitalAllocationDisciplineForRun:
    def test_write_creates_artifact(self, tmp_path):
        """write_capital_allocation_discipline_for_run creates correct JSON artifact."""
        output_path = tmp_path / "capital_allocation_discipline.json"
        rows = [
            {
                "ticker": "AAPL",
                "capital_allocation_discipline_detail": {
                    "ticker": "AAPL",
                    "as_of_date": "2025-01-01",
                    "capital_allocation_discipline_class": OWNER_FRIENDLY_DISCIPLINED,
                    "capital_allocation_discipline_reason_codes": [REASON_LOW_DILUTION_STRONG_QUALITY],
                    "per_share_support_signals": [SIG_LOW_DILUTION],
                    "per_share_headwind_signals": [],
                    "primary_capital_allocation_caution": CAUTION_SUPPORTIVE,
                    "per_share_value_capture_summary": "owner-friendly: low dilution + strong quality",
                    "dilution_rate_annual_pct": 0.5,
                    "oe_quality_total": 10.0,
                    "derived_from": [],
                },
            },
            {
                "ticker": "DILUTE",
                "capital_allocation_discipline_detail": {
                    "ticker": "DILUTE",
                    "as_of_date": "2025-01-01",
                    "capital_allocation_discipline_class": OWNER_DILUTIVE_OR_DESTRUCTIVE,
                    "capital_allocation_discipline_reason_codes": [REASON_HIGH_DILUTION_DOMINANT],
                    "per_share_support_signals": [],
                    "per_share_headwind_signals": [SIG_HIGH_DILUTION, SIG_SHARE_COUNT_HEADWIND],
                    "primary_capital_allocation_caution": CAUTION_HEADWIND,
                    "per_share_value_capture_summary": "destructive: high dilution",
                    "dilution_rate_annual_pct": 7.0,
                    "oe_quality_total": 2.0,
                    "derived_from": [],
                },
            },
        ]
        result = write_capital_allocation_discipline_for_run(
            run_id="test_run",
            as_of_date="2025-01-01",
            tickers=["AAPL", "DILUTE"],
            output_path=output_path,
            scoreboard_rows=rows,
        )
        assert output_path.exists()
        payload = json.loads(output_path.read_text())
        assert payload["ticker_count"] == 2
        assert OWNER_FRIENDLY_DISCIPLINED in payload["counts_by_capital_allocation_discipline_class"]
        assert OWNER_DILUTIVE_OR_DESTRUCTIVE in payload["counts_by_capital_allocation_discipline_class"]
        assert len(payload["top_10_owner_friendly_disciplined"]) == 1
        assert len(payload["top_10_owner_dilutive_or_destructive"]) == 1

    def test_write_computes_on_the_fly_when_no_detail(self, tmp_path):
        """write function computes discipline on-the-fly when detail not pre-computed."""
        output_path = tmp_path / "capital_allocation_discipline.json"
        rows = [
            {
                "ticker": "LIVE",
                "dilution_rate_shares_cagr": 0.008,
                "oe_quality_total": 9.0,
                "capital_allocation_score": 3.5,
                "owner_earnings_stability_score": 4.0,
                "capital_allocation_reason_codes": [],
                "facts_status": "OK",
                "shares_status": "OK",
                "price_status": "OK",
            }
        ]
        result = write_capital_allocation_discipline_for_run(
            run_id="test_live",
            as_of_date="2025-01-01",
            tickers=["LIVE"],
            output_path=output_path,
            scoreboard_rows=rows,
        )
        assert output_path.exists()
        payload = json.loads(output_path.read_text())
        assert payload["ticker_count"] == 1


class TestOpenCapitalAllocationDiscipline:
    def test_open_missing_returns_missing_status(self):
        """open_capital_allocation_discipline returns MISSING for nonexistent run."""
        result = open_capital_allocation_discipline(run_id="nonexistent_run_xyz_cap_alloc")
        assert result["status"] == "MISSING"

    def test_open_existing_returns_ok_status(self, tmp_path, monkeypatch):
        """open_capital_allocation_discipline returns OK when artifact exists."""
        payload = {
            "run_id": "test",
            "as_of_date": "2025-01-01",
            "ticker_count": 1,
            "counts_by_capital_allocation_discipline_class": {OWNER_FRIENDLY_DISCIPLINED: 1},
            "counts_by_primary_capital_allocation_caution": {CAUTION_SUPPORTIVE: 1},
            "top_10_owner_friendly_disciplined": [{"ticker": "AAPL", "reason_codes": []}],
            "top_10_owner_dilutive_or_destructive": [],
            "top_10_capital_allocation_headwind": [],
            "most_common_reason_codes": [],
            "rows": [
                {
                    "ticker": "AAPL",
                    "capital_allocation_discipline_class": OWNER_FRIENDLY_DISCIPLINED,
                    "capital_allocation_discipline_reason_codes": [],
                    "per_share_support_signals": [SIG_LOW_DILUTION],
                    "per_share_headwind_signals": [],
                    "primary_capital_allocation_caution": CAUTION_SUPPORTIVE,
                    "per_share_value_capture_summary": "owner-friendly",
                    "derived_from": [],
                }
            ],
            "generated_at": "2025-01-01T00:00:00Z",
        }
        run_path = tmp_path / "universe" / "test_run" / "capital_allocation_discipline.json"
        run_path.parent.mkdir(parents=True)
        run_path.write_text(json.dumps(payload))

        monkeypatch.setattr(
            "app.valuation.capital_allocation_discipline._capital_allocation_discipline_path",
            lambda run_id: run_path,
        )
        result = open_capital_allocation_discipline(run_id="test_run")
        assert result["status"] == "OK"
        assert result["ticker_count"] == 1
        assert OWNER_FRIENDLY_DISCIPLINED in result["counts_by_capital_allocation_discipline_class"]


class TestNonFiniteDilutionRate:
    """A NaN dilution rate compares False against every tier boundary and used to land in
    LOW_DILUTION (and OWNER_FRIENDLY_DISCIPLINED); inf and bool are not measurements either."""

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf"), True, False, "n/a"]
    )
    def test_non_finite_dilution_rate_is_unknown_not_low_dilution(self, bad):
        result = compute_capital_allocation_discipline(
            "NANDIL", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=bad, oe_quality_total=10.0),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == CAPITAL_ALLOCATION_UNKNOWN
        assert result["dilution_rate_annual_pct"] == "UNKNOWN"
        assert result["effective_dilution_pct"] == "UNKNOWN"
        assert SIG_LOW_DILUTION not in result["per_share_support_signals"]
        assert SIG_SHARE_COUNT_DISCIPLINE not in result["per_share_support_signals"]
        assert SIG_HIGH_DILUTION not in result["per_share_headwind_signals"]

    def test_the_result_never_carries_a_bare_nan_literal(self):
        result = compute_capital_allocation_discipline(
            "NANJSON", "2025-01-01",
            owner_quality_payload=_oe_quality(
                dilution_rate=float("nan"),
                oe_quality_total=float("nan"),
                capital_allocation_score=float("inf"),
            ),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["oe_quality_total"] == "UNKNOWN"
        assert result["capital_allocation_score"] == "UNKNOWN"
        text = json.dumps(result, allow_nan=False)  # raises ValueError on NaN/Infinity
        assert "NaN" not in text and "Infinity" not in text

    def test_non_finite_dilution_falls_back_to_a_finite_share_count_cagr(self):
        """The shares payload is the existing second source when the rate is missing."""
        result = compute_capital_allocation_discipline(
            "NANFALL", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=float("nan"), oe_quality_total=10.0),
            shares_payload={"share_count_cagr": 0.05},
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["effective_dilution_pct"] == pytest.approx(5.0)
        assert SIG_HIGH_DILUTION in result["per_share_headwind_signals"]

    def test_a_finite_rate_is_unchanged(self):
        result = compute_capital_allocation_discipline(
            "FINITE", "2025-01-01",
            owner_quality_payload=_oe_quality(dilution_rate=0.005, oe_quality_total=10.0),
            price_status="OK",
            facts_status="OK",
            shares_status="OK",
        )
        assert result["capital_allocation_discipline_class"] == OWNER_FRIENDLY_DISCIPLINED
        assert result["oe_quality_total"] == 10.0
