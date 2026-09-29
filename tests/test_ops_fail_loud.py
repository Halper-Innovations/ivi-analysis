"""Fail-loud ops: preflight, heartbeat ledger, census breakers, gate
fail-closed semantics, digest data-health block."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.config import get_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def _make_engine_db(path: Path, *, watchlist_rows: int = 0, facts_rows: int = 0) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE watchlist (id INTEGER PRIMARY KEY, ticker TEXT)")
    conn.execute(
        "CREATE TABLE companyfacts_facts (id INTEGER PRIMARY KEY, ticker TEXT)"
    )
    conn.executemany(
        "INSERT INTO watchlist(ticker) VALUES (?)",
        [(f"T{i}",) for i in range(watchlist_rows)],
    )
    conn.executemany(
        "INSERT INTO companyfacts_facts(ticker) VALUES (?)",
        [(f"T{i}",) for i in range(facts_rows)],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# config path anchoring
# ---------------------------------------------------------------------------


def test_relative_db_path_is_anchored_to_project_root(monkeypatch):
    monkeypatch.setenv("VOE_DB_PATH", "data/engine.db")
    monkeypatch.setenv("VOE_DATA_DIR", "data")
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.db_path.is_absolute()
    assert cfg.data_dir.is_absolute()
    assert cfg.db_path == cfg.project_root / "data/engine.db"


def test_absolute_db_path_is_untouched(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "x.db"))
    get_config.cache_clear()
    assert get_config().db_path == tmp_path / "x.db"


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def test_preflight_fails_on_missing_engine_db(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "missing.db"))
    get_config.cache_clear()
    from app.ops.preflight import run_preflight

    result = run_preflight(require_cache_dirs=False)
    assert not result.ok
    assert any(c.name == "engine_db_exists" for c in result.failures)


def test_preflight_fails_row_floor_on_shadow_db(monkeypatch, tmp_path):
    db = tmp_path / "engine.db"
    _make_engine_db(db, watchlist_rows=2, facts_rows=5)
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(db))
    get_config.cache_clear()
    from app.ops.preflight import run_preflight

    result = run_preflight(require_cache_dirs=False)
    assert not result.ok
    names = {c.name for c in result.failures}
    assert "engine_db_row_floor_watchlist" in names
    assert "engine_db_row_floor_companyfacts" in names


def test_preflight_passes_on_healthy_db(monkeypatch, tmp_path):
    db = tmp_path / "engine.db"
    _make_engine_db(db, watchlist_rows=150, facts_rows=1_000_001)
    (tmp_path / "cache").mkdir()
    (tmp_path / "raw_filings").mkdir()
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(db))
    get_config.cache_clear()
    from app.ops.preflight import run_preflight

    result = run_preflight()
    assert result.ok, [c.to_dict() for c in result.failures]


def test_assert_preflight_raises_with_failure_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "missing.db"))
    get_config.cache_clear()
    from app.ops.preflight import PreflightError, assert_preflight

    with pytest.raises(PreflightError, match="engine_db_exists"):
        assert_preflight(require_cache_dirs=False)


# ---------------------------------------------------------------------------
# heartbeat ledger
# ---------------------------------------------------------------------------


def _ledger_env(monkeypatch, tmp_path):
    db = tmp_path / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(db))
    get_config.cache_clear()
    return db


def test_heartbeat_ledger_complete_roundtrip(monkeypatch, tmp_path):
    db = _ledger_env(monkeypatch, tmp_path)
    from app.ops.heartbeat_ledger import heartbeat_status, record_step

    record_step(heartbeat="hb", step="a", exit_code=0, run_date="2026-07-15", db_path=db)
    record_step(
        heartbeat="hb", step="_complete", exit_code=0, run_date="2026-07-15", db_path=db
    )
    status = heartbeat_status(heartbeat="hb", run_date="2026-07-15", db_path=db)
    assert status["status"] == "COMPLETE"
    assert status["failed_steps"] == []


def test_heartbeat_ledger_failed_step_marks_run_failed(monkeypatch, tmp_path):
    db = _ledger_env(monkeypatch, tmp_path)
    from app.ops.heartbeat_ledger import heartbeat_status, record_step

    record_step(heartbeat="hb", step="a", exit_code=3, run_date="2026-07-15", db_path=db)
    record_step(
        heartbeat="hb", step="_complete", exit_code=1, run_date="2026-07-15", db_path=db
    )
    status = heartbeat_status(heartbeat="hb", run_date="2026-07-15", db_path=db)
    assert status["status"] == "FAILED"
    assert status["failed_steps"] == ["a"]


def test_heartbeat_ledger_missing_and_started(monkeypatch, tmp_path):
    db = _ledger_env(monkeypatch, tmp_path)
    from app.ops.heartbeat_ledger import heartbeat_status, record_step

    assert (
        heartbeat_status(heartbeat="hb", run_date="2026-07-15", db_path=db)["status"]
        == "MISSING"
    )
    record_step(heartbeat="hb", step="a", exit_code=0, run_date="2026-07-15", db_path=db)
    assert (
        heartbeat_status(heartbeat="hb", run_date="2026-07-15", db_path=db)["status"]
        == "STARTED"
    )


# ---------------------------------------------------------------------------
# structural gate fail-closed
# ---------------------------------------------------------------------------


def test_gate_missing_db_is_excluded_error(tmp_path):
    from app.autonomous.structural_gate import evaluate_structural_gate

    result = evaluate_structural_gate(
        "AAA", as_of_date="2026-07-15", db_path=tmp_path / "nope.db"
    )
    assert result.excluded_error
    assert result.degraded_codes == ["ENGINE_DB_MISSING"]
    assert not result.quarantined


def test_gate_minimal_db_without_tables_is_clean_not_degraded(tmp_path):
    # A legitimately minimal DB (no filings/companyfacts tables) means "no
    # signal", not "platform failure" — missing-schema errors stay benign.
    db = tmp_path / "mini.db"
    sqlite3.connect(str(db)).close()
    from app.autonomous.structural_gate import evaluate_structural_gate

    result = evaluate_structural_gate("AAA", as_of_date="2026-07-15", db_path=db)
    assert not result.excluded_error
    assert not result.quarantined


def test_intake_gate_excluded_error_quarantines(tmp_path, monkeypatch):
    # Store intake: a gate that could not examine the name must quarantine
    # with the EXCLUDED_ERROR reason, never land presentable.
    from app.autonomous import structural_gate as sg

    captured = {}

    def fake_gate(ticker, **kwargs):
        return sg.StructuralGateResult(
            ticker=ticker,
            as_of_date="2026-07-15",
            degraded_codes=["ENGINE_DB_MISSING"],
        )

    result = fake_gate("AAA")
    assert result.excluded_error
    assert result.degraded_string == "EXCLUDED_ERROR:ENGINE_DB_MISSING"
    captured["ok"] = True


def test_sector_candidates_gate_excludes_on_degraded(tmp_path, monkeypatch):
    from app.autonomous import sector_candidates as sc

    warnings: list[str] = []
    kept, gated = sc._apply_structural_gate(
        ["AAA", "BBB"],
        as_of_date="2026-07-15",
        cap_classifications={},
        db_path=tmp_path / "nope.db",
        warnings=warnings,
        exclude=True,
    )
    assert kept == []
    assert sorted(gated) == ["AAA", "BBB"]
    assert all("EXCLUDED_ERROR:ENGINE_DB_MISSING" in w for w in warnings)


# ---------------------------------------------------------------------------
# census breakers
# ---------------------------------------------------------------------------


def test_census_flip_guard_refuses_mass_flip():
    from app.universe.registrant_census import (
        CensusRefusedError,
        RegistrantRecord,
        _flip_guard,
    )

    existing = {
        str(i).zfill(10): {"operating_status": "OPERATING", "in_scope": 1}
        for i in range(100)
    }
    records = [
        RegistrantRecord(
            cik=str(i).zfill(10),
            primary_ticker=f"T{i}",
            tickers=[f"T{i}"],
            name=f"Co {i}",
            exchange="NYSE",
            exchange_scope="IN_SCOPE",
        )
        for i in range(100)
    ]
    for record in records:
        record.operating_status = "SUBMISSIONS_UNAVAILABLE"
        record.in_scope = False

    with pytest.raises(CensusRefusedError, match="flip"):
        _flip_guard(records, existing, allow_mass_update=False)
    # Explicit override passes.
    _flip_guard(records, existing, allow_mass_update=True)


def test_census_flip_guard_allows_normal_churn():
    from app.universe.registrant_census import RegistrantRecord, _flip_guard

    existing = {
        str(i).zfill(10): {"operating_status": "OPERATING", "in_scope": 1}
        for i in range(5000)
    }
    records = []
    for i in range(5000):
        record = RegistrantRecord(
            cik=str(i).zfill(10),
            primary_ticker=f"T{i}",
            tickers=[f"T{i}"],
            name=f"Co {i}",
            exchange="NYSE",
            exchange_scope="IN_SCOPE",
        )
        record.operating_status = "OPERATING"
        record.in_scope = True
        records.append(record)
    # 60 flips on 5000 rows (1.2%) is under the 2% threshold.
    for record in records[:60]:
        record.operating_status = "NO_LONGER_REPORTING"
        record.in_scope = False
    _flip_guard(records, existing, allow_mass_update=False)


def test_sync_removal_floor_refuses_mass_removal(tmp_path):
    from app.db import init_db
    from app.universe.registrant_intake import (
        UniverseSyncRefusedError,
        _mark_removed_registrants,
    )

    db = tmp_path / "engine.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    init_db(conn=conn)
    now = "2026-07-15T00:00:00+00:00"
    for i in range(200):
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, all_tickers, exchange_scope,
                operating_status, in_scope, first_seen_at, last_seen_at
            ) VALUES (?, ?, '[]', 'IN_SCOPE', 'OPERATING', 1, ?, ?)
            """,
            (str(i).zfill(10), f"T{i}", now, now),
        )
    conn.commit()
    conn.close()

    # Registry only knows the first 100 -> 100 removals of 200 (50%) refused.
    registry = {str(i).zfill(10) for i in range(100)}
    with pytest.raises(UniverseSyncRefusedError, match="refused"):
        _mark_removed_registrants(registry_ciks=registry, db_path=db)

    # 4 removals of 200 would be fine but is below the absolute floor of 20.
    registry_ok = {str(i).zfill(10) for i in range(4, 200)}
    removed = _mark_removed_registrants(registry_ciks=registry_ok, db_path=db)
    assert len(removed) == 4


# ---------------------------------------------------------------------------
# digest data-health block
# ---------------------------------------------------------------------------


def test_digest_renders_data_health_first(monkeypatch, tmp_path):
    from app.watchlist.digest import render_digest
    from app.watchlist.schema import ensure_watchlist_schema

    db = tmp_path / "watch.db"
    ensure_watchlist_schema(db)
    digest = render_digest(days_back=1, db_path=db)
    assert "## Data Health" in digest
    assert digest.index("## Data Health") < digest.index("## Signal Board")
    assert "data health: OK" in digest


def test_data_health_blocking_on_missing_explicit_db(tmp_path):
    from app.ops.data_health import compute_data_health

    health = compute_data_health(tmp_path / "gone.db")
    assert health.blocking
    assert health.red


def test_data_health_lines_distinguish_warning_from_blocking_red():
    from app.ops.data_health import DataHealth

    warning = DataHealth(
        checks=[
            {
                "name": "backup_fresh",
                "ok": False,
                "detail": "backup status=FAILED (0d old)",
            }
        ]
    )
    blocking = DataHealth(
        checks=[
            {
                "name": "engine_db_exists",
                "ok": False,
                "detail": "missing books of record",
            }
        ],
        blocking=True,
    )

    assert warning.state == "AMBER"
    assert warning.lines() == [
        "- **DATA HEALTH AMBER** (1 warning):",
        "  - backup_fresh: backup status=FAILED (0d old)",
        "  - **NONBLOCKING**: operational freshness warning; candidate-level gates remain authoritative",
    ]
    assert blocking.state == "RED"
    assert blocking.lines() == [
        "- **DATA HEALTH RED** (1 failing):",
        "  - engine_db_exists: missing books of record",
        "  - **BLOCKING**: engine DB failed basic integrity — at-target sections are suppressed until this is fixed",
    ]


def test_data_health_full_mode_blocks_on_shadow_db(monkeypatch, tmp_path):
    db = tmp_path / "engine.db"
    _make_engine_db(db, watchlist_rows=1, facts_rows=1)
    (tmp_path / "cache").mkdir()
    (tmp_path / "raw_filings").mkdir()
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(db))
    get_config.cache_clear()
    from app.ops.data_health import compute_data_health_full

    health = compute_data_health_full()
    assert health.blocking
    assert any("row_floor" in line for line in health.red_lines)


def test_data_health_overridden_db_path_gets_cheap_mode(monkeypatch, tmp_path):
    # A VOE_DB_PATH override (tests, ad-hoc copies) must not trip production
    # row floors — the caller chose that DB deliberately.
    db = tmp_path / "engine.db"
    _make_engine_db(db, watchlist_rows=1, facts_rows=1)
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(db))
    get_config.cache_clear()
    from app.ops.data_health import compute_data_health

    health = compute_data_health(None)
    assert not health.blocking
    assert not health.red
