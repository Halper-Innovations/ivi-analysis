"""Tests for app.sector.scan_preflight — health gate before launching a scan."""

from __future__ import annotations



from app.db import get_db, init_db, utc_now_iso
from app.sector.scan_preflight import (
    PreflightResult,
    format_preflight_report,
    run_preflight,
)


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _gc

    _gc.cache_clear()
    cfg = _gc()
    init_db(cfg)
    return cfg, db_path


def _seed_facts(ticker: str, items: list[tuple[int, str, float]]) -> None:
    """items: list of (fiscal_year, line_item, value)."""
    now = utc_now_iso()
    with get_db() as conn:
        for fy, li, v in items:
            conn.execute(
                """INSERT OR REPLACE INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item, value,
                    units, source_url, fetched_at)
                   VALUES (?, ?, 'FY', ?, ?, ?, 'USD_millions', '', ?)""",
                (ticker, fy, f"{fy}-12-31", li, v, now),
            )


def test_preflight_passes_when_healthy(monkeypatch, tmp_path):
    """All sampled tickers have all core fields → pass."""
    cfg, db_path = _init_temp_db(monkeypatch, tmp_path)
    healthy = ("AAA", "BBB", "CCC", "DDD", "EEE")
    for t in healthy:
        _seed_facts(t, [
            (2025, "revenue", 1000.0),
            (2025, "cfo", 200.0),
            (2025, "cash", 100.0),
            (2025, "total_debt", 300.0),
        ])
    result = run_preflight(
        sector="test", tickers=list(healthy), db_path=db_path, sample_size=10,
    )
    assert result.passed
    assert result.pass_count == 5
    assert result.fail_count == 0
    assert result.pass_rate == 1.0


def test_preflight_passes_when_some_fields_missing_but_2plus_present(monkeypatch, tmp_path):
    """Tickers with 3 of 4 core fields still pass — the gate is checking the
    tool layer, not the universe completeness."""
    cfg, db_path = _init_temp_db(monkeypatch, tmp_path)
    full_set = [f"T{i}" for i in range(10)]
    # All 10 missing total_debt but have the other 3
    for t in full_set:
        _seed_facts(t, [
            (2025, "revenue", 100.0), (2025, "cfo", 20.0), (2025, "cash", 10.0),
        ])
    result = run_preflight(
        sector="test", tickers=full_set, db_path=db_path, sample_size=10,
    )
    # 3/4 fields per ticker — passes the >= 2 threshold
    assert result.passed
    assert result.failures_by_field.get("total_debt") == 10  # but field IS missing


def test_preflight_fails_when_systemic_data_failure(monkeypatch, tmp_path):
    """Most tickers have 0 or 1 of 4 fields → systemic failure (e.g., the
    ingestion pipeline broke). The gate must catch this."""
    cfg, db_path = _init_temp_db(monkeypatch, tmp_path)
    full_set = [f"T{i}" for i in range(10)]
    # 2 healthy
    for t in full_set[:2]:
        _seed_facts(t, [
            (2025, "revenue", 100.0), (2025, "cfo", 20.0),
            (2025, "cash", 10.0), (2025, "total_debt", 30.0),
        ])
    # 8 only have revenue (1/4 → fails per-ticker check)
    for t in full_set[2:]:
        _seed_facts(t, [(2025, "revenue", 100.0)])
    result = run_preflight(
        sector="test", tickers=full_set, db_path=db_path, sample_size=10,
    )
    # 2 pass, 8 fail → 20% pass rate < 50% threshold → FAIL
    assert not result.passed
    assert result.pass_rate == 0.2


def test_preflight_fails_when_data_is_stale(monkeypatch, tmp_path):
    """All ticker data > 3 years old → all stale → 0% pass → FAIL."""
    cfg, db_path = _init_temp_db(monkeypatch, tmp_path)
    full_set = ["X1", "X2", "X3", "X4", "X5"]
    for t in full_set:
        _seed_facts(t, [
            (2018, "revenue", 100.0), (2018, "cfo", 20.0),
            (2018, "cash", 10.0), (2018, "total_debt", 30.0),
        ])
    result = run_preflight(
        sector="test", tickers=full_set, db_path=db_path, sample_size=5,
        max_data_age_years=3,
    )
    assert not result.passed
    assert result.stale_count == 5


def test_preflight_samples_when_universe_large(monkeypatch, tmp_path):
    """500 tickers, sample_size=20 → only 20 sampled."""
    cfg, db_path = _init_temp_db(monkeypatch, tmp_path)
    universe = [f"T{i:03d}" for i in range(500)]
    for t in universe:
        _seed_facts(t, [
            (2025, "revenue", 100.0), (2025, "cfo", 20.0),
            (2025, "cash", 10.0), (2025, "total_debt", 30.0),
        ])
    result = run_preflight(
        sector="test", tickers=universe, db_path=db_path,
        sample_size=20, seed=42,
    )
    assert len(result.sampled_tickers) == 20
    assert result.passed


def test_preflight_handles_missing_ticker_gracefully(monkeypatch, tmp_path):
    """Tickers with NO companyfacts data should be flagged as missing all fields."""
    cfg, db_path = _init_temp_db(monkeypatch, tmp_path)
    # Only one ticker has data; the other has nothing in the DB
    _seed_facts("HAS_DATA", [
        (2025, "revenue", 100.0), (2025, "cfo", 20.0),
        (2025, "cash", 10.0), (2025, "total_debt", 30.0),
    ])
    result = run_preflight(
        sector="test", tickers=["HAS_DATA", "NO_DATA"], db_path=db_path,
        sample_size=2,
    )
    assert result.pass_count == 1
    assert result.fail_count == 1
    no_data = next(r for r in result.per_ticker if r.ticker == "NO_DATA")
    assert sorted(no_data.missing_fields) == ["cash", "cfo", "revenue", "total_debt"]
    assert "no companyfacts data" in no_data.note.lower()


def test_preflight_pre_revenue_biotech_passes(monkeypatch, tmp_path):
    """Pre-revenue clinical-stage biotech with cash + total_debt only should
    PASS — those two fields are enough to confirm the data layer works for
    that ticker even though the company has no revenue or cfo yet."""
    cfg, db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_facts("BIOTECH", [
        (2025, "cash", 200.0),
        (2025, "total_debt", 50.0),
    ])
    result = run_preflight(
        sector="test", tickers=["BIOTECH"], db_path=db_path, sample_size=1,
    )
    assert result.passed
    biotech = result.per_ticker[0]
    assert biotech.passed
    assert sorted(biotech.missing_fields) == ["cfo", "revenue"]


def test_format_preflight_report_renders():
    """Smoke-test the report formatter doesn't blow up on a real result."""
    result = PreflightResult(
        sector="test",
        sampled_tickers=["A", "B"],
        pass_count=1,
        fail_count=1,
        pass_rate=0.5,
        pass_threshold=0.9,
        passed=False,
        failures_by_field={"total_debt": 1},
    )
    text = format_preflight_report(result)
    assert "test" in text
    assert "FAILED" in text
    assert "total_debt" in text


def test_threshold_is_configurable(monkeypatch, tmp_path):
    """Same data, different threshold → different verdict.

    Default threshold is 50%. We construct a sample with 40% passing
    (4 healthy + 6 with only 1 field) to verify both behaviors.
    """
    cfg, db_path = _init_temp_db(monkeypatch, tmp_path)
    full_set = [f"T{i}" for i in range(10)]
    # 4 healthy
    for t in full_set[:4]:
        _seed_facts(t, [
            (2025, "revenue", 100.0), (2025, "cfo", 20.0),
            (2025, "cash", 10.0), (2025, "total_debt", 30.0),
        ])
    # 6 broken (only 1 field)
    for t in full_set[4:]:
        _seed_facts(t, [(2025, "revenue", 100.0)])

    # 40% pass — fails default 50% threshold
    r1 = run_preflight(sector="test", tickers=full_set, db_path=db_path, sample_size=10)
    assert not r1.passed
    assert r1.pass_rate == 0.4
    # 40% pass — passes a 30% threshold
    r2 = run_preflight(
        sector="test", tickers=full_set, db_path=db_path, sample_size=10,
        pass_threshold=0.3,
    )
    assert r2.passed
