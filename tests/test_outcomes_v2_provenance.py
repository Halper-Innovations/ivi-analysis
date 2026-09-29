from __future__ import annotations

import pytest

from app.config import get_config
from app.db import init_db
from app.outcomes.store import add_outcome


def _init(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    init_db()


def _write_v2(**overrides):
    values = {
        "ticker": "AAA",
        "as_of_date": "2026-07-16",
        "run_id": "autonomous_sector_v2_outcome",
        "decision": "WATCH",
        "conviction": 2,
        "horizon_days": 365,
        "grade": "WATCHLIST_ONLY",
        "status": "ACTIVE",
        "pipeline_version": "v2",
        "candidate_disposition": "READY_FOR_UNDERWRITING",
        "decision_basis": "SCREEN",
        "source_sector": "energy",
    }
    values.update(overrides)
    return add_outcome(**values)


def test_direct_v2_outcome_accepts_literal_screen_provenance(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    row = _write_v2()

    assert row["pipeline_version"] == "v2"
    assert row["candidate_disposition"] == "READY_FOR_UNDERWRITING"
    assert row["decision_basis"] == "SCREEN"
    assert row["source_sector"] == "energy"


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_disposition": None},
        {"decision_basis": None},
        {"source_sector": None},
        {"candidate_disposition": "SCREENED_OUT", "grade": "WATCHLIST_ONLY"},
        {"decision_basis": "UNDERWRITING"},
        {"grade": "ACTIONABLE"},
        {"decision": "BUY"},
        {"status": "DEPLOY_READY"},
        {"selection_validation_status": "VALIDATED"},
        {
            "candidate_disposition": "UNDERWRITTEN",
            "decision_basis": "VALIDATED_UNDERWRITING",
            "grade": "ACTIONABLE",
            "selection_validation_status": "NOT_VALIDATED",
        },
    ],
)
def test_direct_v2_outcome_rejects_inconsistent_provenance(
    monkeypatch, tmp_path, overrides
):
    _init(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="v2"):
        _write_v2(**overrides)


def test_direct_v2_outcome_accepts_validated_actionable(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    row = _write_v2(
        decision="BUY",
        grade="ACTIONABLE",
        status="DEPLOY_READY",
        candidate_disposition="UNDERWRITTEN",
        decision_basis="VALIDATED_UNDERWRITING",
        selection_validation_status="VALIDATED",
    )

    assert row["decision"] == "BUY"
    assert row["grade"] == "ACTIONABLE"
    assert row["selection_validation_status"] == "VALIDATED"


def test_direct_v2_outcome_accepts_screened_out_measurement(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    row = _write_v2(
        decision="PASS",
        grade="AVOID",
        status=None,
        candidate_disposition="SCREENED_OUT",
        decision_basis="SCREEN",
    )

    assert row["candidate_disposition"] == "SCREENED_OUT"
    assert row["decision_basis"] == "SCREEN"
    assert row["grade"] == "AVOID"


def test_direct_v2_outcome_accepts_noninvestable_safety_statuses(
    monkeypatch, tmp_path
):
    _init(monkeypatch, tmp_path)

    screen = _write_v2(status="QUARANTINE")
    validated_but_quarantined = _write_v2(
        ticker="BBB",
        run_id="autonomous_sector_v2_quarantined_actionable",
        decision="WATCH",
        grade="ACTIONABLE",
        status="QUARANTINE",
        candidate_disposition="UNDERWRITTEN",
        decision_basis="VALIDATED_UNDERWRITING",
        selection_validation_status="VALIDATED",
    )

    assert screen["status"] == "QUARANTINE"
    assert validated_but_quarantined["decision"] == "WATCH"
    assert validated_but_quarantined["grade"] == "ACTIONABLE"


def test_legacy_shaped_upsert_cannot_corrupt_existing_v2_provenance(
    monkeypatch, tmp_path
):
    _init(monkeypatch, tmp_path)
    original = _write_v2()

    with pytest.raises(ValueError, match="v2"):
        add_outcome(
            ticker="AAA",
            as_of_date="2026-07-16",
            run_id="autonomous_sector_v2_outcome",
            decision="BUY",
            conviction=5,
            horizon_days=365,
            grade="ACTIONABLE",
            status="DEPLOY_READY",
        )

    assert original["candidate_disposition"] == "READY_FOR_UNDERWRITING"
    # A failed merged-row validation happens before the archive or UPSERT.
    from app.db import get_db

    with get_db() as conn:
        row = conn.execute(
            "SELECT decision, grade, status, pipeline_version, "
            "candidate_disposition, decision_basis FROM ticker_outcomes "
            "WHERE ticker='AAA' AND run_id='autonomous_sector_v2_outcome'"
        ).fetchone()
    assert tuple(row) == (
        "WATCH",
        "WATCHLIST_ONLY",
        "ACTIVE",
        "v2",
        "READY_FOR_UNDERWRITING",
        "SCREEN",
    )
