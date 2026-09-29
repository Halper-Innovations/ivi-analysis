"""Tests for app.alpha.anomaly_detector."""

from __future__ import annotations

from unittest.mock import patch

from app.db import connect, get_db, init_db, utc_now_iso


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
    return cfg


def _seed_facts(ticker, facts_list):
    """Insert facts. Each item: (fiscal_year, period_type, line_item, value)."""
    now = utc_now_iso()
    with get_db() as conn:
        for fy, pt, li, val in facts_list:
            conn.execute(
                """INSERT OR REPLACE INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item, value,
                    units, source_url, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, fy, pt, f"{fy}-12-31", li, val, "USD_millions", "", now),
            )


def test_q4_earnings_bomb(monkeypatch, tmp_path):
    """Detect when Q4 implied net income is wildly negative vs Q1-Q3."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST",
        [
            (2025, "FY", "net_income", -27.0),
            (2025, "Q1", "net_income", -2.0),
            (2025, "Q2", "net_income", 2.0),
            (2025, "Q3", "net_income", 0.5),
        ],
    )
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("TEST")
    q4_bombs = [a for a in anomalies if a.anomaly_type == "Q4_EARNINGS_BOMB"]
    assert len(q4_bombs) == 1
    assert q4_bombs[0].severity == "HIGH"
    assert "Q4" in q4_bombs[0].description


def test_negative_equity(monkeypatch, tmp_path):
    """Detect negative stockholders' equity."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST",
        [
            (2025, "FY", "equity", -16.0),
            (2024, "FY", "equity", 10.0),
        ],
    )
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("TEST")
    neg_eq = [a for a in anomalies if a.anomaly_type == "NEGATIVE_EQUITY"]
    assert len(neg_eq) == 1
    assert neg_eq[0].severity == "HIGH"


def test_margin_collapse(monkeypatch, tmp_path):
    """Detect operating margin decline > 10pp over 2 years."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST",
        [
            (2023, "FY", "revenue", 100.0),
            (2023, "FY", "operating_income", 20.0),
            (2024, "FY", "revenue", 100.0),
            (2024, "FY", "operating_income", 12.0),
            (2025, "FY", "revenue", 100.0),
            (2025, "FY", "operating_income", 5.0),
        ],
    )
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("TEST")
    mc = [a for a in anomalies if a.anomaly_type == "MARGIN_COLLAPSE"]
    assert len(mc) == 1


def test_intangible_asset_jump(monkeypatch, tmp_path):
    """Detect intangible asset increase > 3x YoY."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST",
        [
            (2024, "FY", "intangible_assets", 1.0),
            (2025, "FY", "intangible_assets", 8.7),
        ],
    )
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("TEST")
    jumps = [a for a in anomalies if a.anomaly_type == "INTANGIBLE_ASSET_JUMP"]
    assert len(jumps) == 1
    assert "acquisition" in jumps[0].question.lower() or "capitaliz" in jumps[0].question.lower()


def test_revenue_decline(monkeypatch, tmp_path):
    """Detect revenue decline > 20% from peak."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST",
        [
            (2021, "FY", "revenue", 107.0),
            (2022, "FY", "revenue", 74.0),
            (2023, "FY", "revenue", 61.0),
            (2024, "FY", "revenue", 84.0),
            (2025, "FY", "revenue", 81.0),
        ],
    )
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("TEST")
    rd = [a for a in anomalies if a.anomaly_type == "REVENUE_DECLINE_FROM_PEAK"]
    assert len(rd) == 1


def test_cash_burn(monkeypatch, tmp_path):
    """Detect persistent negative CFO."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST",
        [
            (2023, "FY", "cfo", -3.0),
            (2024, "FY", "cfo", -2.0),
            (2025, "FY", "cfo", -1.0),
            (2025, "FY", "cash", 8.0),
        ],
    )
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("TEST")
    cb = [a for a in anomalies if a.anomaly_type == "PERSISTENT_CASH_BURN"]
    assert len(cb) == 1
    assert "runway" in cb[0].question.lower() or "cash" in cb[0].question.lower()


def test_cash_burn_preserves_missing_cash_as_unavailable(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST",
        [
            (2023, "FY", "cfo", -3.0),
            (2024, "FY", "cfo", -2.0),
            (2025, "FY", "cfo", -1.0),
        ],
    )
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("TEST")
    cash_burn = next(item for item in anomalies if item.anomaly_type == "PERSISTENT_CASH_BURN")
    assert cash_burn.data["cash"] is None
    assert "Cash: unavailable" in cash_burn.description
    assert "Cash: $0.0M" not in cash_burn.description


def test_no_data_returns_empty(monkeypatch, tmp_path):
    """Ticker with no data should return empty anomaly list."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("ZZZZ")
    assert anomalies == []


def test_healthy_company_few_anomalies(monkeypatch, tmp_path):
    """A stable, profitable company should produce zero or few anomalies."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "GOOD",
        [
            (2023, "FY", "revenue", 100.0),
            (2023, "FY", "operating_income", 20.0),
            (2023, "FY", "net_income", 15.0),
            (2023, "FY", "cfo", 25.0),
            (2023, "FY", "equity", 80.0),
            (2023, "FY", "cash", 30.0),
            (2024, "FY", "revenue", 110.0),
            (2024, "FY", "operating_income", 23.0),
            (2024, "FY", "net_income", 17.0),
            (2024, "FY", "cfo", 28.0),
            (2024, "FY", "equity", 90.0),
            (2024, "FY", "cash", 35.0),
            (2025, "FY", "revenue", 120.0),
            (2025, "FY", "operating_income", 26.0),
            (2025, "FY", "net_income", 19.0),
            (2025, "FY", "cfo", 30.0),
            (2025, "FY", "equity", 100.0),
            (2025, "FY", "cash", 40.0),
        ],
    )
    from app.alpha.anomaly_detector import detect_anomalies

    anomalies = detect_anomalies("GOOD")
    assert len(anomalies) <= 1  # maybe a minor one at most


def test_v2_anomalies_bind_known_cik_exclude_future_and_use_explicit_db(tmp_path):
    db_path = tmp_path / "issuer-context.db"
    conn = connect(db_path)
    try:
        init_db(conn=conn)
        now = utc_now_iso()
        rows = (
            ("ORD", 2024, "FY", "2024-12-31", "equity", 100.0, "2025-02-15", 42),
            ("ORD", 2026, "FY", "2026-12-31", "equity", -500.0, "2027-02-15", 42),
            ("ADR", 2024, "FY", "2024-12-31", "equity", -900.0, "2025-02-15", 43),
        )
        for ticker, year, period_type, period_end, line_item, value, filed_date, cik in rows:
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date)
                   VALUES (?, ?, ?, ?, ?, ?, 'USD_millions', ?, ?, ?)""",
                (
                    ticker,
                    year,
                    period_type,
                    period_end,
                    line_item,
                    value,
                    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                    now,
                    filed_date,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    from app.alpha.anomaly_detector import detect_anomalies

    with patch(
        "app.alpha.anomaly_detector.get_db",
        side_effect=AssertionError("explicit db_path must not open the configured DB"),
    ):
        anomalies = detect_anomalies(
            "ADR",
            as_of_date="2025-06-30",
            require_filed_asof=True,
            issuer_cik="42",
            aliases=("ADR", "ORD"),
            db_path=db_path,
        )

    assert [item.anomaly_type for item in anomalies] == []


def test_v1_anomalies_remain_exact_ticker(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts("LEGACY", [(2025, "FY", "equity", 100.0)])
    _seed_facts("LEGACY.A", [(2025, "FY", "equity", -900.0)])

    from app.alpha.anomaly_detector import detect_anomalies

    assert [item.anomaly_type for item in detect_anomalies("LEGACY")] == []
