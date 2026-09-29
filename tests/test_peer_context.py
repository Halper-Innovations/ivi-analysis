"""Tests for app.valuation.peer_context."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _init_temp_db(monkeypatch, tmp_path):
    """Set up a temporary database for tests that need DB access."""
    from app.db import init_db

    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _mock_sector_universe(sector):
    """Return a fake peer list for testing."""
    return [
        {"ticker": "PEER_A", "cik": "0000001", "company_name": "Peer A", "sic": 3674},
        {"ticker": "PEER_B", "cik": "0000002", "company_name": "Peer B", "sic": 3674},
        {"ticker": "PEER_C", "cik": "0000003", "company_name": "Peer C", "sic": 3674},
        {"ticker": "PEER_D", "cik": "0000004", "company_name": "Peer D", "sic": 3674},
        {"ticker": "PEER_E", "cik": "0000005", "company_name": "Peer E", "sic": 3674},
        {"ticker": "PEER_F", "cik": "0000006", "company_name": "Peer F", "sic": 3674},
    ]


def _insert_filed_companyfact(
    conn,
    *,
    ticker,
    fiscal_year,
    line_item,
    value,
    period_end,
    filed_date,
):
    accession = f"fixture-{ticker.lower()}-{fiscal_year}-{line_item}"
    conn.execute(
        """
        INSERT OR IGNORE INTO companyfacts_facts(
            ticker, fiscal_year, period_type, period_end, filed_date, form,
            accession, line_item, value, units, source_url, fetched_at
        ) VALUES (?, ?, 'FY', ?, ?, '10-K', ?, ?, ?, 'USD_millions',
                  'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                  ?)
        """,
        (
            ticker,
            fiscal_year,
            period_end,
            filed_date,
            accession,
            line_item,
            float(value),
            f"{filed_date}T00:00:00Z",
        ),
    )


def _seed_peer_facts(conn):
    """Insert companyfacts rows for mock peers."""
    peers = {
        "PEER_A": {
            "revenue": [(2021, 100), (2025, 150)],
            "operating_income": [(2024, 24), (2025, 30)],
            "total_assets": [(2024, 180), (2025, 200)],
        },
        "PEER_B": {
            "revenue": [(2021, 200), (2025, 280)],
            "operating_income": [(2024, 40), (2025, 50)],
            "total_assets": [(2024, 360), (2025, 400)],
        },
        "PEER_C": {
            "revenue": [(2021, 80), (2025, 100)],
            "operating_income": [(2024, 12), (2025, 15)],
            "total_assets": [(2024, 135), (2025, 150)],
        },
        "PEER_D": {
            "revenue": [(2021, 120), (2025, 180)],
            "operating_income": [(2024, 32), (2025, 40)],
            "total_assets": [(2024, 225), (2025, 250)],
        },
        "PEER_E": {
            "revenue": [(2021, 90), (2025, 110)],
            "operating_income": [(2024, 16), (2025, 20)],
            "total_assets": [(2024, 162), (2025, 180)],
        },
        "PEER_F": {
            "revenue": [(2021, 150), (2025, 200)],
            "operating_income": [(2024, 36), (2025, 45)],
            "total_assets": [(2024, 315), (2025, 350)],
        },
    }
    for ticker, items in peers.items():
        for line_item, series in items.items():
            for year, value in series:
                _insert_filed_companyfact(
                    conn,
                    ticker=ticker,
                    fiscal_year=year,
                    line_item=line_item,
                    value=value,
                    period_end=f"{year}-12-31",
                    filed_date=f"{year + 1}-02-15",
                )
    conn.commit()


def test_sector_medians_computed(monkeypatch, tmp_path):
    """Sector with 6 peers -> medians computed, peer_count = 6."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.valuation.peer_context import compute_sector_medians
    from app.db import get_db

    with get_db() as conn:
        _seed_peer_facts(conn)

    with patch(
        "app.valuation.peer_context._load_sector_tickers",
        return_value=["PEER_A", "PEER_B", "PEER_C", "PEER_D", "PEER_E", "PEER_F"],
    ):
        result = compute_sector_medians("semiconductors", "2026-03-27", use_cache=False)

    assert result["peer_count"] == 6
    assert result["medians"]["roic"] is not None
    assert result["medians"]["operating_margin"] is not None
    assert result["medians"]["revenue_growth_5y"] is not None
    assert result["medians"]["roic"] > 0
    assert result["medians"]["operating_margin"] > 0


def test_sector_medians_insufficient_peers(monkeypatch, tmp_path):
    """Sector with < 5 peers -> None medians."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.valuation.peer_context import compute_sector_medians

    with patch("app.valuation.peer_context._load_sector_tickers", return_value=["ONLY_ONE"]):
        result = compute_sector_medians("energy", "2026-03-27", use_cache=False)

    assert result["peer_count"] < 5
    assert result["medians"]["roic"] is None


def test_sector_medians_cache_hit(tmp_path):
    """Cached result returned without recomputing."""
    import json
    from app.valuation.peer_context import compute_sector_medians

    # Pre-populate cache
    cache_data = {
        "sector": "test_sector",
        "peer_count": 10,
        "peer_tickers": ["A", "B"],
        "medians": {"roic": 0.15, "operating_margin": 0.20, "revenue_growth_5y": 0.08},
        "computed_at": "2026-03-26T23:00:00Z",
    }

    with patch("app.valuation.peer_context._cache_path", return_value=tmp_path / "test.json"):
        (tmp_path / "test.json").write_text(json.dumps(cache_data))
        with patch("app.valuation.peer_context._cache_is_fresh", return_value=True):
            result = compute_sector_medians("test_sector", "2026-03-27")

    assert result["peer_count"] == 10
    assert result["medians"]["roic"] == 0.15


def test_peer_relative_ok(monkeypatch, tmp_path):
    """Ticker in known sector with peers -> status OK, position classified."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.valuation.peer_context import compute_peer_relative_metrics
    from app.db import get_db

    with get_db() as conn:
        _seed_peer_facts(conn)
        # Add the target ticker
        for item, series in {
            "revenue": [(2021, 300), (2025, 500)],
            "operating_income": [(2025, 100)],
            "total_assets": [(2025, 400)],
        }.items():
            for year, value in series:
                _insert_filed_companyfact(
                    conn,
                    ticker="TARGET",
                    fiscal_year=year,
                    line_item=item,
                    value=value,
                    period_end=f"{year}-12-31",
                    filed_date=f"{year + 1}-02-15",
                )
        conn.commit()

    with patch("app.valuation.peer_context._detect_sector", return_value="semiconductors"):
        with patch(
            "app.valuation.peer_context._load_sector_tickers",
            return_value=["PEER_A", "PEER_B", "PEER_C", "PEER_D", "PEER_E", "PEER_F"],
        ):
            result = compute_peer_relative_metrics("TARGET", "2026-03-27")

    assert result["status"] == "OK"
    assert result["sector"] == "semiconductors"
    assert result["relative_position"] in (
        "LEADER",
        "ABOVE_AVERAGE",
        "AVERAGE",
        "BELOW_AVERAGE",
        "LAGGARD",
    )
    assert result["ticker_metrics"]["roic"] is not None
    assert result["relative_ratios"]["roic_vs_median"] is not None


def test_peer_relative_no_sector():
    """Ticker not in any sector -> NO_SECTOR_MATCH."""
    from app.valuation.peer_context import compute_peer_relative_metrics

    with patch("app.valuation.peer_context._detect_sector", return_value=None):
        result = compute_peer_relative_metrics("UNKNOWN_TICKER", "2026-03-27")

    assert result["status"] == "NO_SECTOR_MATCH"


def test_peer_relative_insufficient_peers():
    """Sector with < 5 peers -> INSUFFICIENT_PEERS."""
    from app.valuation.peer_context import compute_peer_relative_metrics

    with patch("app.valuation.peer_context._detect_sector", return_value="energy"):
        with patch(
            "app.valuation.peer_context.compute_sector_medians",
            return_value={
                "sector": "energy",
                "peer_count": 3,
                "peer_tickers": ["A", "B", "C"],
                "medians": {"roic": None, "operating_margin": None, "revenue_growth_5y": None},
                "computed_at": "2026-01-01",
            },
        ):
            result = compute_peer_relative_metrics("TEST", "2026-03-27")

    assert result["status"] == "INSUFFICIENT_PEERS"
    assert result["peer_count"] == 3


def test_peer_relative_leader(monkeypatch, tmp_path):
    """Ticker significantly outperforming peers -> LEADER."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.valuation.peer_context import compute_peer_relative_metrics
    from app.db import get_db

    with get_db() as conn:
        _seed_peer_facts(conn)
        # Add a strong ticker -- 2x peer medians
        for item, series in {
            "revenue": [(2021, 100), (2025, 300)],
            "operating_income": [(2025, 120)],
            "total_assets": [(2025, 200)],
        }.items():
            for year, value in series:
                _insert_filed_companyfact(
                    conn,
                    ticker="STRONG",
                    fiscal_year=year,
                    line_item=item,
                    value=value,
                    period_end=f"{year}-12-31",
                    filed_date=f"{year + 1}-02-15",
                )
        conn.commit()

    with patch("app.valuation.peer_context._detect_sector", return_value="semiconductors"):
        with patch(
            "app.valuation.peer_context._load_sector_tickers",
            return_value=["PEER_A", "PEER_B", "PEER_C", "PEER_D", "PEER_E", "PEER_F"],
        ):
            result = compute_peer_relative_metrics("STRONG", "2026-03-27")

    assert result["status"] == "OK"
    assert result["relative_position"] == "LEADER"


def test_peer_relative_respects_as_of_date(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    from app.valuation.peer_context import compute_peer_relative_metrics
    from app.db import get_db

    with get_db() as conn:
        _seed_peer_facts(conn)
        for item, series in {
            "revenue": [(2021, 250), (2025, 500)],
            "operating_income": [(2024, 40), (2025, 100)],
            "total_assets": [(2024, 200), (2025, 400)],
        }.items():
            for year, value in series:
                _insert_filed_companyfact(
                    conn,
                    ticker="TARGET",
                    fiscal_year=year,
                    line_item=item,
                    value=value,
                    period_end=f"{year}-12-31",
                    filed_date=f"{year + 1}-02-15",
                )
        conn.commit()

    with patch("app.valuation.peer_context._detect_sector", return_value="semiconductors"):
        with patch(
            "app.valuation.peer_context._load_sector_tickers",
            return_value=["PEER_A", "PEER_B", "PEER_C", "PEER_D", "PEER_E", "PEER_F"],
        ):
            older = compute_peer_relative_metrics("TARGET", "2025-03-27")
            newer = compute_peer_relative_metrics("TARGET", "2026-03-27")

    assert older["status"] == "OK"
    assert newer["status"] == "OK"
    assert older["ticker_metrics"]["roic"] == 0.2
    assert newer["ticker_metrics"]["roic"] == 0.25


def test_sector_medians_cache_is_asof_specific(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    from app.valuation.peer_context import compute_sector_medians
    from app.db import get_db

    with get_db() as conn:
        _seed_peer_facts(conn)

    cache_dir = tmp_path / "sector_medians"
    with patch("app.valuation.peer_context._cache_dir", return_value=cache_dir):
        with patch(
            "app.valuation.peer_context._load_sector_tickers",
            return_value=["PEER_A", "PEER_B", "PEER_C", "PEER_D", "PEER_E", "PEER_F"],
        ):
            compute_sector_medians("semiconductors", "2025-03-27", use_cache=False)
            compute_sector_medians("semiconductors", "2026-03-27", use_cache=False)

    assert (cache_dir / "semiconductors__2025-03-27.json").exists()
    assert (cache_dir / "semiconductors__2026-03-27.json").exists()


def test_peer_context_integration():
    """Peer context attached to scorecard quality_context when wired in."""
    from app.valuation.peer_context import compute_peer_relative_metrics

    with patch("app.valuation.peer_context._detect_sector", return_value=None):
        result = compute_peer_relative_metrics("INTEGRATION_TEST", "2026-03-27")

    # Just verify the function returns a dict with status
    assert isinstance(result, dict)
    assert "status" in result


@pytest.mark.parametrize(
    ("metric", "ratio_key"),
    [
        ("ev_ebitda", "ev_ebitda_vs_median"),
        ("ev_ebit", "ev_ebit_vs_median"),
        ("ev_sales", "ev_sales_vs_median"),
        ("p_e", "p_e_vs_median"),
        ("p_b", "p_b_vs_median"),
        ("fcf_yield", "fcf_yield_vs_median"),
        ("dividend_yield", "dividend_yield_vs_median"),
    ],
)
def test_peer_relative_metric_distribution_math_for_supported_valuation_metrics(metric, ratio_key):
    from app.valuation.peer_context import compute_peer_relative_metrics

    peer_values = {
        "PEER_A": 1.0,
        "PEER_B": 2.0,
        "PEER_C": 3.0,
        "PEER_D": 4.0,
        "PEER_E": 5.0,
        "PEER_F": 6.0,
        "TARGET": 4.0,
    }

    def fake_metrics(ticker, as_of_date):
        return {metric: peer_values[ticker]}

    with patch("app.valuation.peer_context._detect_sector", return_value="semiconductors"):
        with patch(
            "app.valuation.peer_context._load_sector_tickers",
            return_value=["PEER_A", "PEER_B", "PEER_C", "PEER_D", "PEER_E", "PEER_F"],
        ):
            with patch("app.valuation.peer_context._cache_is_fresh", return_value=False):
                with patch(
                    "app.valuation.peer_context._compute_ticker_metrics", side_effect=fake_metrics
                ):
                    result = compute_peer_relative_metrics("TARGET", "2026-03-27")

    assert result["status"] == "OK"
    assert result["ticker_metrics"][metric] == 4.0
    assert result["sector_medians"][metric] == 3.5
    assert result["sector_q1"][metric] == 2.0
    assert result["sector_q3"][metric] == 5.0
    assert result["relative_ratios"][ratio_key] == 1.14
    assert result["percentile_ranks"][metric] == 66.7
    assert result["peer_set_used"] == ["PEER_A", "PEER_B", "PEER_C", "PEER_D", "PEER_E", "PEER_F"]


def _seed_peer_metric_inputs(
    conn,
    *,
    ticker,
    debt=100.0,
    cash=20.0,
    filed_date="2025-02-15",
):
    values = {
        "operating_income": 100.0,
        "depreciation_amortization": 20.0,
        "revenue": 500.0,
        "total_debt": debt,
        "cash": cash,
    }
    for line_item, value in values.items():
        if value is None:
            continue
        _insert_filed_companyfact(
            conn,
            ticker=ticker,
            fiscal_year=2024,
            line_item=line_item,
            value=value,
            period_end="2024-12-31",
            filed_date=filed_date,
        )
    conn.execute(
        """
        INSERT INTO market_caps(
            ticker, effective_as_of_date, run_id, run_as_of_date,
            market_cap, market_cap_unit, market_cap_status, fetched_at,
            payload_json, created_at
        ) VALUES (?, '2026-03-01', 'peer-test', '2026-03-27',
                  1000.0, 'USD_millions', 'OK',
                  '2026-03-01T00:00:00Z', '{}',
                  '2026-03-01T00:00:00Z')
        """,
        (ticker,),
    )


def test_peer_metrics_reject_post_asof_facts(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    from app.db import get_db
    from app.valuation.peer_context import _compute_ticker_metrics

    with get_db() as conn:
        _seed_peer_metric_inputs(conn, ticker="PIT")
        for line_item, value in {
            "operating_income": 900.0,
            "depreciation_amortization": 100.0,
            "revenue": 1200.0,
            "total_debt": 900.0,
            "cash": 0.0,
        }.items():
            _insert_filed_companyfact(
                conn,
                ticker="PIT",
                fiscal_year=2025,
                line_item=line_item,
                value=value,
                period_end="2025-12-31",
                filed_date="2026-04-15",
            )
        conn.commit()

    metrics = _compute_ticker_metrics("PIT", as_of_date="2026-03-27")

    assert metrics["ev_ebitda"] == 9.0
    assert metrics["ev_ebit"] == 10.8
    assert metrics["ev_sales"] == 2.16


@pytest.mark.parametrize(
    ("debt", "cash"),
    [(None, 20.0), (100.0, None)],
)
def test_peer_metrics_omit_ev_when_debt_or_cash_is_missing(
    monkeypatch,
    tmp_path,
    debt,
    cash,
):
    _init_temp_db(monkeypatch, tmp_path)
    from app.db import get_db
    from app.valuation.peer_context import _compute_ticker_metrics

    with get_db() as conn:
        _seed_peer_metric_inputs(conn, ticker="MISS", debt=debt, cash=cash)
        conn.commit()

    metrics = _compute_ticker_metrics("MISS", as_of_date="2026-03-27")

    assert metrics["ev_ebitda"] is None
    assert metrics["ev_ebit"] is None
    assert metrics["ev_sales"] is None


def test_peer_metrics_accept_explicit_reported_zero_debt_and_cash(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    from app.db import get_db
    from app.valuation.peer_context import _compute_ticker_metrics

    with get_db() as conn:
        _seed_peer_metric_inputs(conn, ticker="ZERO", debt=0.0, cash=0.0)
        conn.commit()

    metrics = _compute_ticker_metrics("ZERO", as_of_date="2026-03-27")

    assert metrics["ev_ebitda"] == 8.3333
    assert metrics["ev_ebit"] == 10.0
    assert metrics["ev_sales"] == 2.0
