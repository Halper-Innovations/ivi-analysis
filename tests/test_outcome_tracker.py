"""Tests for app.calibration.outcome_tracker."""

from __future__ import annotations

import sqlite3

from app.db import init_db
from app.calibration.outcome_tracker import (
    MOS_UNDERVALUED_THRESHOLD,
    MOS_OVERVALUED_THRESHOLD,
    METHOD_NEUTRAL_BAND,
    CONFIRM_THRESHOLD_PCT,
    DISCONFIRM_THRESHOLD_PCT,
    SUPPORTED_METHODS,
    ReportOutcome,
    MethodOutcome,
    ScanResult,
    evaluate_report_outcome,
    evaluate_method_outcome,
    evaluate_methods,
)
from app.research.deep_research import ResearchReport
from app.research.thesis_updater import ThesisResult


def _make_thesis(
    *,
    current_price=100.0,
    adjusted_dcf=120.0,
    adjusted_epv=90.0,
    original_graham=80.0,
    adjusted_intrinsic_mid=100.0,
    adjusted_margin_of_safety=0.0,
) -> ThesisResult:
    return ThesisResult(
        ticker="TEST",
        iteration=0,
        status="OK",
        original_dcf=adjusted_dcf,
        original_epv=adjusted_epv,
        original_graham=original_graham,
        current_price=current_price,
        adjustments=[],
        adjusted_dcf=adjusted_dcf,
        adjusted_epv=adjusted_epv,
        adjusted_intrinsic_mid=adjusted_intrinsic_mid,
        adjusted_margin_of_safety=adjusted_margin_of_safety,
        hypotheses_confirmed=2,
        hypotheses_contradicted=1,
        hypotheses_partially_confirmed=0,
        hypotheses_inconclusive=1,
        hypotheses_unclassified=0,
        average_coverage=0.6,
        unresolved=[],
        high_priority_unresolved=0,
    )


def _make_report(
    *, thesis=None, conviction_score=67, conviction_class="MODERATE"
) -> ResearchReport:
    return ResearchReport(
        ticker="TEST",
        as_of_date="2026-01-01",
        status="OK",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:01:00+00:00",
        scorecard_present=True,
        filing_present=True,
        filing_date="2025-11-15",
        form_type="10-K",
        anomaly_count=0,
        solvency_status=None,
        filing_risk_status=None,
        gate_action="PROCEED",
        investigation_ran=True,
        hypotheses_generated=4,
        thesis=thesis,
        researchable_items=[],
        not_researchable_items=[],
        total_adjustments=0,
        fact_calibrated_count=0,
        heuristic_count=0,
        methods_agree=True,
        consensus_strength=2,
        method_count=2,
        tension_type="NONE",
        conviction_score=conviction_score,
        conviction_class=conviction_class,
    )


class TestOutcomeTrackerSchema:
    def test_tables_created(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db(conn=conn)
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        assert "deep_research_outcomes" in tables
        assert "deep_research_method_outcomes" in tables
        conn.close()

    def test_parent_check_constraints(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db(conn=conn)
        import pytest

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO deep_research_outcomes
                   (ticker, source_as_of_date, horizon_days,
                    target_exit_date, scan_date,
                    entry_price, exit_price,
                    exit_as_of_date, exit_source,
                    price_change_pct,
                    thesis_verdict, verdict_outcome, created_at)
                   VALUES ('TEST', '2026-01-01', 90,
                           '2026-04-01', '2026-04-02',
                           100.0, 110.0,
                           '2026-04-01', 'stooq',
                           10.0,
                           'BAD_VERDICT', 'CORRECT', '2026-04-02T00:00:00')"""
            )
        conn.close()

    def test_child_cascade_delete(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db(conn=conn)
        conn.execute(
            """INSERT INTO deep_research_outcomes
               (ticker, source_as_of_date, horizon_days,
                target_exit_date, scan_date,
                entry_price, exit_price,
                exit_as_of_date, exit_source,
                price_change_pct,
                thesis_verdict, verdict_outcome, created_at)
               VALUES ('TEST', '2026-01-01', 90,
                       '2026-04-01', '2026-04-02',
                       100.0, 110.0,
                       '2026-04-01', 'stooq',
                       10.0,
                       'UNDERVALUED', 'CORRECT', '2026-04-02T00:00:00')"""
        )
        parent_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO deep_research_method_outcomes
               (outcome_id, method, predicted_value, entry_price,
                direction, outcome, unadjusted_source, created_at)
               VALUES (?, 'dcf', 120.0, 100.0,
                       'UNDERVALUED', 'CORRECT', 0, '2026-04-02T00:00:00')""",
            (parent_id,),
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM deep_research_method_outcomes").fetchone()[0] == 1
        conn.execute("DELETE FROM deep_research_outcomes WHERE id = ?", (parent_id,))
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM deep_research_method_outcomes").fetchone()[0] == 0
        conn.close()


class TestConstantsAndDataclasses:
    def test_threshold_constants(self):
        assert MOS_UNDERVALUED_THRESHOLD == 0.03
        assert MOS_OVERVALUED_THRESHOLD == -0.03
        assert METHOD_NEUTRAL_BAND == 0.03
        assert CONFIRM_THRESHOLD_PCT == 10.0
        assert DISCONFIRM_THRESHOLD_PCT == 15.0

    def test_supported_methods(self):
        assert SUPPORTED_METHODS == {"dcf", "epv", "graham"}

    def test_report_outcome_construction(self):
        ro = ReportOutcome(
            thesis_verdict="UNDERVALUED",
            verdict_outcome="CORRECT",
            price_change_pct=15.0,
            conviction_score=75,
            conviction_class="HIGH",
        )
        assert ro.thesis_verdict == "UNDERVALUED"
        assert ro.conviction_score == 75

    def test_method_outcome_construction(self):
        mo = MethodOutcome(
            method="dcf",
            predicted_value=120.0,
            direction="UNDERVALUED",
            outcome="CORRECT",
            unadjusted_source=False,
        )
        assert mo.method == "dcf"
        assert mo.unadjusted_source is False

    def test_method_outcome_graham_unadjusted(self):
        mo = MethodOutcome(
            method="graham",
            predicted_value=90.0,
            direction="OVERVALUED",
            outcome="INCONCLUSIVE",
            unadjusted_source=True,
        )
        assert mo.unadjusted_source is True

    def test_scan_result_construction(self):
        sr = ScanResult(
            scan_date="2026-04-01",
            horizon_days=90,
            total_eligible=10,
            evaluated=8,
            skipped_no_price=2,
            outcomes=[],
        )
        assert sr.evaluated == 8
        assert sr.skipped_no_price == 2


class TestDeriveThesisVerdict:
    def test_undervalued(self):
        from app.calibration.outcome_tracker import _derive_thesis_verdict

        assert _derive_thesis_verdict(0.10) == "UNDERVALUED"

    def test_overvalued(self):
        from app.calibration.outcome_tracker import _derive_thesis_verdict

        assert _derive_thesis_verdict(-0.10) == "OVERVALUED"

    def test_fairly_valued_positive_edge(self):
        from app.calibration.outcome_tracker import _derive_thesis_verdict

        assert _derive_thesis_verdict(0.03) == "FAIRLY_VALUED"

    def test_fairly_valued_negative_edge(self):
        from app.calibration.outcome_tracker import _derive_thesis_verdict

        assert _derive_thesis_verdict(-0.03) == "FAIRLY_VALUED"

    def test_fairly_valued_zero(self):
        from app.calibration.outcome_tracker import _derive_thesis_verdict

        assert _derive_thesis_verdict(0.0) == "FAIRLY_VALUED"

    def test_just_above_threshold(self):
        from app.calibration.outcome_tracker import _derive_thesis_verdict

        assert _derive_thesis_verdict(0.031) == "UNDERVALUED"

    def test_just_below_threshold(self):
        from app.calibration.outcome_tracker import _derive_thesis_verdict

        assert _derive_thesis_verdict(-0.031) == "OVERVALUED"


class TestResolveDirection:
    def test_undervalued_confirmed(self):
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("UNDERVALUED", 12.0) == "CORRECT"

    def test_undervalued_disconfirmed(self):
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("UNDERVALUED", -16.0) == "INCORRECT"

    def test_undervalued_inconclusive(self):
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("UNDERVALUED", 5.0) == "INCONCLUSIVE"

    def test_overvalued_confirmed(self):
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("OVERVALUED", -12.0) == "CORRECT"

    def test_overvalued_disconfirmed(self):
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("OVERVALUED", 16.0) == "INCORRECT"

    def test_overvalued_inconclusive(self):
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("OVERVALUED", -5.0) == "INCONCLUSIVE"

    def test_fairly_valued_always_inconclusive(self):
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("FAIRLY_VALUED", 50.0) == "INCONCLUSIVE"
        assert _resolve_direction("FAIRLY_VALUED", -50.0) == "INCONCLUSIVE"

    def test_boundary_confirm_not_triggered(self):
        """Exactly 10% does not cross the >10% threshold."""
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("UNDERVALUED", 10.0) == "INCONCLUSIVE"

    def test_boundary_disconfirm_not_triggered(self):
        """Exactly -15% does not cross the <-15% threshold."""
        from app.calibration.outcome_tracker import _resolve_direction

        assert _resolve_direction("UNDERVALUED", -15.0) == "INCONCLUSIVE"


class TestMethodDirection:
    def test_undervalued(self):
        from app.calibration.outcome_tracker import _method_direction

        assert _method_direction(110.0, 100.0) == "UNDERVALUED"

    def test_overvalued(self):
        from app.calibration.outcome_tracker import _method_direction

        assert _method_direction(90.0, 100.0) == "OVERVALUED"

    def test_neutral_band(self):
        from app.calibration.outcome_tracker import _method_direction

        assert _method_direction(102.0, 100.0) == "FAIRLY_VALUED"

    def test_neutral_band_negative(self):
        from app.calibration.outcome_tracker import _method_direction

        assert _method_direction(98.0, 100.0) == "FAIRLY_VALUED"

    def test_exact_boundary_positive(self):
        """Exactly 3% ratio -> FAIRLY_VALUED (not > threshold)."""
        from app.calibration.outcome_tracker import _method_direction

        assert _method_direction(103.0, 100.0) == "FAIRLY_VALUED"

    def test_just_above_boundary(self):
        from app.calibration.outcome_tracker import _method_direction

        assert _method_direction(103.1, 100.0) == "UNDERVALUED"


class TestEvaluateReportOutcome:
    def test_undervalued_correct(self):
        thesis = _make_thesis(adjusted_margin_of_safety=0.15, current_price=100.0)
        report = _make_report(thesis=thesis)
        ro = evaluate_report_outcome(report, entry_price=100.0, exit_price=115.0)
        assert ro.thesis_verdict == "UNDERVALUED"
        assert ro.verdict_outcome == "CORRECT"
        assert ro.price_change_pct == 15.0
        assert ro.conviction_score == 67

    def test_overvalued_correct(self):
        thesis = _make_thesis(adjusted_margin_of_safety=-0.20, current_price=100.0)
        report = _make_report(thesis=thesis)
        ro = evaluate_report_outcome(report, entry_price=100.0, exit_price=88.0)
        assert ro.thesis_verdict == "OVERVALUED"
        assert ro.verdict_outcome == "CORRECT"
        assert ro.price_change_pct == -12.0

    def test_fairly_valued_always_inconclusive(self):
        thesis = _make_thesis(adjusted_margin_of_safety=0.01, current_price=100.0)
        report = _make_report(thesis=thesis)
        ro = evaluate_report_outcome(report, entry_price=100.0, exit_price=150.0)
        assert ro.thesis_verdict == "FAIRLY_VALUED"
        assert ro.verdict_outcome == "INCONCLUSIVE"


class TestEvaluateMethodOutcome:
    def test_dcf_undervalued_correct(self):
        mo = evaluate_method_outcome("dcf", 120.0, 100.0, 115.0, unadjusted_source=False)
        assert mo.direction == "UNDERVALUED"
        assert mo.outcome == "CORRECT"
        assert mo.unadjusted_source is False

    def test_graham_overvalued_with_flag(self):
        mo = evaluate_method_outcome("graham", 80.0, 100.0, 88.0, unadjusted_source=True)
        assert mo.direction == "OVERVALUED"
        assert mo.outcome == "CORRECT"
        assert mo.unadjusted_source is True

    def test_neutral_band_inconclusive(self):
        mo = evaluate_method_outcome("epv", 102.0, 100.0, 115.0, unadjusted_source=False)
        assert mo.direction == "FAIRLY_VALUED"
        assert mo.outcome == "INCONCLUSIVE"


class TestEvaluateMethods:
    def test_all_three_methods(self):
        thesis = _make_thesis(
            current_price=100.0,
            adjusted_dcf=120.0,
            adjusted_epv=90.0,
            original_graham=80.0,
        )
        report = _make_report(thesis=thesis)
        methods = evaluate_methods(report, entry_price=100.0, exit_price=115.0)
        assert len(methods) == 3
        by_name = {m.method: m for m in methods}
        assert by_name["dcf"].direction == "UNDERVALUED"
        assert by_name["epv"].direction == "OVERVALUED"
        assert by_name["graham"].direction == "OVERVALUED"
        assert by_name["graham"].unadjusted_source is True
        assert by_name["dcf"].unadjusted_source is False

    def test_none_method_omitted(self):
        thesis = _make_thesis(
            current_price=100.0,
            adjusted_dcf=120.0,
            adjusted_epv=None,
            original_graham=None,
        )
        report = _make_report(thesis=thesis)
        methods = evaluate_methods(report, entry_price=100.0, exit_price=115.0)
        assert len(methods) == 1
        assert methods[0].method == "dcf"

    def test_neutral_band_method_included_as_fairly_valued(self):
        thesis = _make_thesis(
            current_price=100.0,
            adjusted_dcf=102.0,
            adjusted_epv=98.0,
            original_graham=None,
        )
        report = _make_report(thesis=thesis)
        methods = evaluate_methods(report, entry_price=100.0, exit_price=115.0)
        assert len(methods) == 2
        for m in methods:
            assert m.direction == "FAIRLY_VALUED"
            assert m.outcome == "INCONCLUSIVE"


class TestDeserializationRoundTrip:
    def test_from_dict_preserves_outcome_fields(self):
        """Construct report via from_dict (simulating DB load) and verify
        all fields needed by outcome evaluation survive round-trip."""
        from dataclasses import asdict

        thesis = _make_thesis(
            current_price=100.0,
            adjusted_dcf=120.0,
            adjusted_epv=90.0,
            original_graham=80.0,
            adjusted_intrinsic_mid=95.0,
            adjusted_margin_of_safety=0.15,
        )
        report = _make_report(thesis=thesis, conviction_score=75, conviction_class="HIGH")
        data = asdict(report)
        reconstructed = ResearchReport.from_dict(data)

        assert reconstructed.thesis.current_price == 100.0
        assert reconstructed.thesis.adjusted_margin_of_safety == 0.15
        assert reconstructed.thesis.adjusted_dcf == 120.0
        assert reconstructed.thesis.adjusted_epv == 90.0
        assert reconstructed.thesis.original_graham == 80.0
        assert reconstructed.thesis.adjusted_intrinsic_mid == 95.0
        assert reconstructed.conviction_score == 75
        assert reconstructed.conviction_class == "HIGH"

        # Evaluate from reconstructed — should produce same result
        ro_original = evaluate_report_outcome(report, 100.0, 115.0)
        ro_reconstructed = evaluate_report_outcome(reconstructed, 100.0, 115.0)
        assert ro_original.thesis_verdict == ro_reconstructed.thesis_verdict
        assert ro_original.verdict_outcome == ro_reconstructed.verdict_outcome

        methods_original = evaluate_methods(report, 100.0, 115.0)
        methods_reconstructed = evaluate_methods(reconstructed, 100.0, 115.0)
        assert len(methods_original) == len(methods_reconstructed)
        for mo, mr in zip(methods_original, methods_reconstructed, strict=False):
            assert mo.method == mr.method
            assert mo.direction == mr.direction
            assert mo.outcome == mr.outcome


# ---------------------------------------------------------------------------
# Task 5: Orchestration tests
# ---------------------------------------------------------------------------

import json
from dataclasses import asdict
from unittest.mock import patch, MagicMock


def _make_db_row(*, ticker="TEST", as_of_date="2026-01-01", valuation_id=42, thesis=None):
    """Build a mock valuations row dict."""
    if thesis is None:
        thesis = _make_thesis(
            current_price=100.0,
            adjusted_dcf=120.0,
            adjusted_epv=90.0,
            original_graham=80.0,
            adjusted_intrinsic_mid=105.0,
            adjusted_margin_of_safety=0.05,
        )
    report = _make_report(thesis=thesis)
    return {
        "id": valuation_id,
        "ticker": ticker,
        "as_of_date": as_of_date,
        "method": "deep_research",
        "outputs_json": json.dumps(asdict(report), default=str),
    }


class TestComputeTargetExitDate:
    def test_90_days(self):
        from app.calibration.outcome_tracker import _compute_target_exit_date

        assert _compute_target_exit_date("2026-01-01", 90) == "2026-04-01"

    def test_180_days(self):
        from app.calibration.outcome_tracker import _compute_target_exit_date

        assert _compute_target_exit_date("2026-01-01", 180) == "2026-06-30"


class TestLoadEligibleReports:
    @patch("app.calibration.outcome_tracker.get_db")
    def test_filters_by_age_and_deserializes(self, mock_get_db):
        from app.calibration.outcome_tracker import _load_eligible_reports

        row = _make_db_row(as_of_date="2026-01-01", valuation_id=42)
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [row]
        mock_get_db.return_value.__enter__ = lambda s: mock_conn
        mock_get_db.return_value.__exit__ = lambda s, *a: None

        results = _load_eligible_reports("2026-04-01", 90)
        assert len(results) == 1
        report, vid, aod = results[0]
        assert report.ticker == "TEST"
        assert vid == 42
        assert aod == "2026-01-01"

    @patch("app.calibration.outcome_tracker.get_db")
    def test_skips_report_with_none_thesis(self, mock_get_db):
        from app.calibration.outcome_tracker import _load_eligible_reports

        report = _make_report(thesis=None)
        report.status = "NO_SCORECARD"
        row = {
            "id": 1,
            "ticker": "BAD",
            "as_of_date": "2026-01-01",
            "method": "deep_research",
            "outputs_json": json.dumps(asdict(report), default=str),
        }
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [row]
        mock_get_db.return_value.__enter__ = lambda s: mock_conn
        mock_get_db.return_value.__exit__ = lambda s, *a: None

        results = _load_eligible_reports("2026-04-01", 90)
        assert len(results) == 0

    @patch("app.calibration.outcome_tracker.get_db")
    def test_skips_report_with_zero_price(self, mock_get_db):
        from app.calibration.outcome_tracker import _load_eligible_reports

        thesis = _make_thesis(current_price=0.0)
        row = _make_db_row(thesis=thesis)
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [row]
        mock_get_db.return_value.__enter__ = lambda s: mock_conn
        mock_get_db.return_value.__exit__ = lambda s, *a: None

        results = _load_eligible_reports("2026-04-01", 90)
        assert len(results) == 0


class TestPersistOutcome:
    def test_inserts_parent_and_children(self, tmp_path):
        from app.calibration.outcome_tracker import _persist_outcome

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db(conn=conn)
        conn.execute(
            """
            INSERT INTO valuations(
                id, ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at, source_run_id,
                source_artifact_path, source_artifact_sha256,
                financial_integrity_fingerprint
            ) VALUES (
                42, 'TEST', '2026-01-01', 'deep_research', '{}', '{}', '[]',
                '2026-01-01T00:00:00Z', 'research_TEST_exact',
                '/fixture/research_TEST_exact.json', ?, 'valuation_fp_exact'
            )
            """,
            ("a" * 64,),
        )

        ro = ReportOutcome(
            thesis_verdict="UNDERVALUED",
            verdict_outcome="CORRECT",
            price_change_pct=15.0,
            conviction_score=75,
            conviction_class="HIGH",
        )
        methods = [
            MethodOutcome("dcf", 120.0, "UNDERVALUED", "CORRECT", False),
            MethodOutcome("epv", 90.0, "OVERVALUED", "INCORRECT", False),
            MethodOutcome("graham", 80.0, "OVERVALUED", "INCORRECT", True),
        ]

        _persist_outcome(
            conn,
            "TEST",
            "2026-01-01",
            "2026-04-01",
            "2026-04-02",
            90,
            100.0,
            115.0,
            "2026-04-01",
            "stooq",
            ro,
            methods,
            source_valuation_id=42,
        )
        conn.commit()

        parent = conn.execute("SELECT * FROM deep_research_outcomes").fetchone()
        assert parent["ticker"] == "TEST"
        assert parent["thesis_verdict"] == "UNDERVALUED"
        assert parent["source_valuation_id"] == 42
        assert parent["source_run_id"] == "research_TEST_exact"
        assert parent["source_artifact_path"] == "/fixture/research_TEST_exact.json"
        assert parent["source_artifact_sha256"] == "a" * 64
        assert parent["source_valuation_fingerprint"] == "valuation_fp_exact"
        assert parent["target_exit_date"] == "2026-04-01"
        assert parent["exit_as_of_date"] == "2026-04-01"
        assert parent["exit_source"] == "stooq"

        children = conn.execute(
            "SELECT * FROM deep_research_method_outcomes ORDER BY method"
        ).fetchall()
        assert len(children) == 3
        assert children[0]["method"] == "dcf"
        assert children[1]["method"] == "epv"
        assert children[2]["method"] == "graham"
        assert children[2]["unadjusted_source"] == 1
        conn.close()

    def test_upsert_replaces_parent_and_children(self, tmp_path):
        from app.calibration.outcome_tracker import _persist_outcome

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db(conn=conn)

        ro1 = ReportOutcome("UNDERVALUED", "CORRECT", 15.0, 75, "HIGH")
        m1 = [MethodOutcome("dcf", 120.0, "UNDERVALUED", "CORRECT", False)]
        _persist_outcome(
            conn,
            "TEST",
            "2026-01-01",
            "2026-04-01",
            "2026-04-02",
            90,
            100.0,
            115.0,
            "2026-04-01",
            "stooq",
            ro1,
            m1,
        )
        conn.commit()

        # Re-run on a later scan_date — same target_exit_date, same price
        ro2 = ReportOutcome("UNDERVALUED", "CORRECT", 15.0, 75, "HIGH")
        m2 = [
            MethodOutcome("dcf", 120.0, "UNDERVALUED", "CORRECT", False),
            MethodOutcome("epv", 90.0, "OVERVALUED", "INCORRECT", False),
        ]
        _persist_outcome(
            conn,
            "TEST",
            "2026-01-01",
            "2026-04-01",
            "2026-05-01",
            90,
            100.0,
            115.0,
            "2026-04-01",
            "stooq",
            ro2,
            m2,
        )
        conn.commit()

        parents = conn.execute("SELECT * FROM deep_research_outcomes").fetchall()
        assert len(parents) == 1
        assert parents[0]["target_exit_date"] == "2026-04-01"
        assert parents[0]["scan_date"] == "2026-05-01"

        children = conn.execute("SELECT * FROM deep_research_method_outcomes").fetchall()
        assert len(children) == 2
        conn.close()


class TestScanResearchOutcomes:
    @patch("app.calibration.outcome_tracker._persist_outcome")
    @patch("app.calibration.outcome_tracker._get_exit_snapshot")
    @patch("app.calibration.outcome_tracker._load_eligible_reports")
    def test_basic_scan(self, mock_load, mock_snapshot, mock_persist):
        from app.calibration.outcome_tracker import scan_research_outcomes

        thesis = _make_thesis(
            current_price=100.0,
            adjusted_dcf=120.0,
            adjusted_epv=90.0,
            original_graham=80.0,
            adjusted_intrinsic_mid=105.0,
            adjusted_margin_of_safety=0.15,
        )
        report = _make_report(thesis=thesis)
        mock_load.return_value = [(report, 42, "2026-01-01")]
        mock_snapshot.return_value = MagicMock(price=115.0, as_of_date="2026-04-01", source="stooq")

        result = scan_research_outcomes("2026-04-01", horizon_days=90)
        assert result.total_eligible == 1
        assert result.evaluated == 1
        assert result.skipped_no_price == 0
        assert len(result.outcomes) == 1
        assert result.outcomes[0].ticker == "TEST"
        assert result.outcomes[0].thesis_verdict == "UNDERVALUED"
        assert result.outcomes[0].target_exit_date == "2026-04-01"
        assert result.outcomes[0].exit_as_of_date == "2026-04-01"
        assert mock_persist.call_count == 1

    @patch("app.calibration.outcome_tracker._persist_outcome")
    @patch("app.calibration.outcome_tracker._get_exit_snapshot")
    @patch("app.calibration.outcome_tracker._load_eligible_reports")
    def test_skips_no_price(self, mock_load, mock_snapshot, mock_persist):
        from app.calibration.outcome_tracker import scan_research_outcomes

        thesis = _make_thesis(current_price=100.0, adjusted_margin_of_safety=0.15)
        report = _make_report(thesis=thesis)
        mock_load.return_value = [(report, 42, "2026-01-01")]
        mock_snapshot.return_value = None

        result = scan_research_outcomes("2026-04-01", horizon_days=90)
        assert result.total_eligible == 1
        assert result.evaluated == 0
        assert result.skipped_no_price == 1
        assert mock_persist.call_count == 0

    @patch("app.calibration.outcome_tracker._persist_outcome")
    @patch("app.calibration.outcome_tracker._get_exit_snapshot")
    @patch("app.calibration.outcome_tracker._load_eligible_reports")
    def test_empty_scan(self, mock_load, mock_snapshot, mock_persist):
        from app.calibration.outcome_tracker import scan_research_outcomes

        mock_load.return_value = []
        result = scan_research_outcomes("2026-04-01", horizon_days=90)
        assert result.total_eligible == 0
        assert result.evaluated == 0
        assert result.skipped_no_price == 0
        assert result.outcomes == []


# ---------------------------------------------------------------------------
# Task 6: Provenance independence tests
# ---------------------------------------------------------------------------


class TestProvenanceIndependence:
    def test_source_valuation_id_change_does_not_duplicate(self, tmp_path):
        """If deep_research reruns (changing valuations.id), outcome tracker
        should upsert cleanly — no duplicate parent rows."""
        from app.calibration.outcome_tracker import _persist_outcome

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db(conn=conn)

        ro = ReportOutcome("UNDERVALUED", "CORRECT", 15.0, 75, "HIGH")
        methods = [MethodOutcome("dcf", 120.0, "UNDERVALUED", "CORRECT", False)]

        # First evaluation with source_valuation_id=42
        _persist_outcome(
            conn,
            "TEST",
            "2026-01-01",
            "2026-04-01",
            "2026-04-01",
            90,
            100.0,
            115.0,
            "2026-04-01",
            "stooq",
            ro,
            methods,
            source_valuation_id=42,
        )
        conn.commit()

        # Deep research reruns — new source_valuation_id=99
        _persist_outcome(
            conn,
            "TEST",
            "2026-01-01",
            "2026-04-01",
            "2026-04-02",
            90,
            100.0,
            115.0,
            "2026-04-01",
            "stooq",
            ro,
            methods,
            source_valuation_id=99,
        )
        conn.commit()

        parents = conn.execute("SELECT * FROM deep_research_outcomes").fetchall()
        assert len(parents) == 1  # upsert, not duplicate
        assert parents[0]["source_valuation_id"] == 99
        conn.close()

    def test_outcome_lookup_by_canonical_identity(self, tmp_path):
        """Outcomes are looked up by (ticker, source_as_of_date, horizon_days),
        not by source_valuation_id."""
        from app.calibration.outcome_tracker import _persist_outcome

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        init_db(conn=conn)

        ro = ReportOutcome("OVERVALUED", "INCONCLUSIVE", -5.0, 50, "MODERATE")
        methods = [MethodOutcome("epv", 85.0, "OVERVALUED", "INCONCLUSIVE", False)]
        _persist_outcome(
            conn,
            "AAPL",
            "2026-01-15",
            "2026-04-15",
            "2026-04-15",
            90,
            150.0,
            142.5,
            "2026-04-15",
            "stooq",
            ro,
            methods,
            source_valuation_id=None,
        )
        conn.commit()

        # Lookup by canonical identity — no reference to source_valuation_id
        row = conn.execute(
            "SELECT * FROM deep_research_outcomes WHERE ticker = ? AND source_as_of_date = ? AND horizon_days = ?",
            ("AAPL", "2026-01-15", 90),
        ).fetchone()
        assert row is not None
        assert row["thesis_verdict"] == "OVERVALUED"
        conn.close()


# ---------------------------------------------------------------------------
# Task 7: Fixed-horizon anchoring and exit provenance regression tests
# ---------------------------------------------------------------------------


class TestFixedHorizonAnchoring:
    @patch("app.calibration.outcome_tracker._persist_outcome")
    @patch("app.calibration.outcome_tracker._get_exit_snapshot")
    @patch("app.calibration.outcome_tracker._load_eligible_reports")
    def test_same_thesis_different_scan_dates_same_target(
        self,
        mock_load,
        mock_snapshot,
        mock_persist,
    ):
        """Running scan on day 100 and day 200 for the same thesis with
        horizon_days=90 must produce identical target_exit_date and exit price.
        The second run must NOT produce a different-horizon price."""
        from app.calibration.outcome_tracker import scan_research_outcomes

        thesis = _make_thesis(
            current_price=100.0,
            adjusted_dcf=120.0,
            adjusted_epv=90.0,
            original_graham=80.0,
            adjusted_intrinsic_mid=105.0,
            adjusted_margin_of_safety=0.15,
        )
        report = _make_report(thesis=thesis)
        mock_load.return_value = [(report, 42, "2026-01-01")]
        mock_snapshot.return_value = MagicMock(
            price=115.0,
            as_of_date="2026-04-01",
            source="stooq",
        )

        # First scan on day ~100
        r1 = scan_research_outcomes("2026-04-10", horizon_days=90)
        # Second scan on day ~200
        r2 = scan_research_outcomes("2026-07-20", horizon_days=90)

        # Both scans should request the same target_exit_date
        assert r1.outcomes[0].target_exit_date == "2026-04-01"
        assert r2.outcomes[0].target_exit_date == "2026-04-01"

        # Verify _get_exit_snapshot was called with target date, not scan_date
        for call in mock_snapshot.call_args_list:
            assert call[0][1] == "2026-04-01"  # target_exit_date, not scan_date


class TestExitProvenance:
    @patch("app.calibration.outcome_tracker._persist_outcome")
    @patch("app.calibration.outcome_tracker._get_exit_snapshot")
    @patch("app.calibration.outcome_tracker._load_eligible_reports")
    def test_weekend_target_persists_actual_snapshot_date(
        self,
        mock_load,
        mock_snapshot,
        mock_persist,
    ):
        """When target_exit_date falls on a weekend, the provider returns a
        snapshot for the prior trading day. The persisted exit_as_of_date must
        match the snapshot's actual date, not the target."""
        from app.calibration.outcome_tracker import scan_research_outcomes

        thesis = _make_thesis(
            current_price=100.0,
            adjusted_margin_of_safety=0.15,
        )
        report = _make_report(thesis=thesis)
        mock_load.return_value = [(report, 42, "2026-01-03")]

        # target_exit_date = 2026-01-03 + 90 = 2026-04-03 (Friday)
        # But simulate provider returning prior Thursday due to holiday
        mock_snapshot.return_value = MagicMock(
            price=112.0,
            as_of_date="2026-04-02",
            source="stooq",
        )

        result = scan_research_outcomes("2026-04-10", horizon_days=90)
        assert len(result.outcomes) == 1
        assert result.outcomes[0].target_exit_date == "2026-04-03"
        assert result.outcomes[0].exit_as_of_date == "2026-04-02"  # actual snapshot date

        # Verify _persist_outcome was called with correct provenance
        persist_call = mock_persist.call_args
        # Positional args: conn, ticker, source_as_of_date, target_exit_date,
        #   scan_date, horizon_days, entry_price, exit_price,
        #   exit_as_of_date, exit_source, ...
        assert persist_call[0][3] == "2026-04-03"  # target_exit_date
        assert persist_call[0][8] == "2026-04-02"  # exit_as_of_date (actual)
        assert persist_call[0][9] == "stooq"  # exit_source


# ---------------------------------------------------------------------------
# Boundary hardening tests
# ---------------------------------------------------------------------------


class TestEligibilityMissingMOS:
    @patch("app.calibration.outcome_tracker.get_db")
    def test_skips_report_with_none_adjusted_mos(self, mock_get_db):
        """A report with adjusted_intrinsic_mid present but
        adjusted_margin_of_safety=None should be skipped, not crash."""
        from app.calibration.outcome_tracker import _load_eligible_reports

        thesis = _make_thesis(
            current_price=100.0,
            adjusted_intrinsic_mid=105.0,
            adjusted_margin_of_safety=None,
        )
        row = _make_db_row(thesis=thesis)
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [row]
        mock_get_db.return_value.__enter__ = lambda s: mock_conn
        mock_get_db.return_value.__exit__ = lambda s, *a: None

        results = _load_eligible_reports("2026-04-01", 90)
        assert len(results) == 0


class TestInvalidHorizonDays:
    def test_zero_horizon_raises(self):
        import pytest
        from app.calibration.outcome_tracker import scan_research_outcomes

        with pytest.raises(ValueError, match="horizon_days must be positive"):
            scan_research_outcomes("2026-04-01", horizon_days=0)

    def test_negative_horizon_raises(self):
        import pytest
        from app.calibration.outcome_tracker import scan_research_outcomes

        with pytest.raises(ValueError, match="horizon_days must be positive"):
            scan_research_outcomes("2026-04-01", horizon_days=-30)
