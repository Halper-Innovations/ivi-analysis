import json

from app.db import get_db, init_db
from app.fundamentals.metrics import (
    compute_all_fundamentals,
    compute_fundamentals_for_ticker,
    compute_metrics_from_financials,
)
from app.fundamentals.normalize import UNKNOWN


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


def _insert_filing(
    conn,
    *,
    cik: str,
    ticker: str,
    accession: str,
    form_type: str,
    filing_date: str,
    period_end: str,
    status: str,
) -> int:
    conn.execute(
        """
        INSERT INTO filings(
            cik, ticker, accession, form_type, filing_date, period_end,
            primary_doc_url, status, created_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cik,
            ticker,
            accession,
            form_type,
            filing_date,
            period_end,
            "https://example.com/filing",
            status,
            "2026-03-20T00:00:00+00:00",
            "2026-03-20T00:00:00+00:00",
        ),
    )
    row = conn.execute("SELECT id FROM filings WHERE accession = ?", (accession,)).fetchone()
    return int(row["id"])


def _insert_financial_map(
    conn,
    filing_id: int,
    *,
    revenue: float,
    gross_profit: float,
    operating_income: float,
    net_income: float,
    cfo: float,
    capex: float,
    cash: float,
    total_debt: float,
) -> None:
    rows = [
        ("income_statement", "revenue", revenue),
        ("income_statement", "gross_profit", gross_profit),
        ("income_statement", "operating_income", operating_income),
        ("income_statement", "net_income", net_income),
        ("cash_flow_statement", "cfo", cfo),
        ("cash_flow_statement", "capex", capex),
        ("balance_sheet", "cash", cash),
        ("balance_sheet", "total_debt", total_debt),
    ]
    for statement_type, line_item, value in rows:
        conn.execute(
            """
            INSERT INTO financials(filing_id, statement_type, line_item, value, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (filing_id, statement_type, line_item, value, "2026-03-20T00:00:00+00:00"),
        )


def test_compute_metrics_basic_fields():
    fin = {
        "revenue": 1000.0,
        "gross_profit": 400.0,
        "operating_income": 120.0,
        "net_income": 80.0,
        "cfo": 150.0,
        "capex": 40.0,
        "cash": 200.0,
        "total_debt": 500.0,
    }

    metrics, flags = compute_metrics_from_financials(fin, liquidity_score=3)

    assert metrics["gross_margin"] == 0.4
    assert metrics["operating_margin"] == 0.12
    assert metrics["fcf"] == 110.0
    assert metrics["fcf_margin"] == 0.11
    assert metrics["net_debt"] == 300.0
    assert metrics["liquidity_stress_score"] == 3
    assert flags["has_revenue"] is True


def test_compute_metrics_unknown_when_missing_inputs():
    metrics, flags = compute_metrics_from_financials({"revenue": None}, liquidity_score=0)
    assert metrics["revenue"] == UNKNOWN
    assert metrics["gross_margin"] == UNKNOWN
    assert metrics["fcf"] == UNKNOWN
    assert flags["has_revenue"] is False


def test_compute_metrics_treats_bank_like_issuer_as_sector_limited_for_fcf():
    fin = {
        "revenue": 1000.0,
        "cfo": -200.0,
        "capex": 20.0,
        "cash": 300.0,
        "total_debt": 100.0,
        "deposits": 2500.0,
        "loans": 1400.0,
        "total_assets": 3900.0,
        "allowance_for_credit_losses": 35.0,
        "provision_for_credit_losses": 12.0,
        "net_charge_offs": 4.0,
        "nonaccrual_loans": 20.0,
    }

    metrics, flags = compute_metrics_from_financials(fin, liquidity_score=4)

    assert metrics["issuer_classification"] == "financial"
    assert metrics["fcf_applicability"] == "sector_limited"
    assert metrics["fcf"] == UNKNOWN
    assert metrics["fcf_margin"] == UNKNOWN
    assert metrics["deposits"] == 2500.0
    assert metrics["loans"] == 1400.0
    assert metrics["deposits_to_assets"] == 2500.0 / 3900.0
    assert metrics["loans_to_deposits"] == 1400.0 / 2500.0
    assert metrics["allowance_to_loans"] == 35.0 / 1400.0
    assert metrics["provision_to_loans"] == 12.0 / 1400.0
    assert metrics["net_charge_offs_to_loans"] == 4.0 / 1400.0
    assert metrics["nonaccrual_loans"] == 20.0
    assert metrics["debt_to_assets"] == 100.0 / 3900.0
    assert flags["issuer_classification"] == "financial"


def test_compute_fundamentals_for_ticker_falls_back_to_companyfacts_rows(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, filed_date,
                line_item, value, units, source_url, fetched_at, form,
                accession
            )
            VALUES(?, ?, 'FY', ?, '2026-02-15', ?, ?, ?, ?,
                   '2026-03-20T00:00:00+00:00', '10-K',
                   '0000000001-26-000001')
            """,
            [
                (
                    "XOM",
                    2025,
                    "2025-12-31",
                    "revenue",
                    100.0,
                    "USD_millions",
                    "https://data.sec.gov",
                ),
                (
                    "XOM",
                    2025,
                    "2025-12-31",
                    "gross_profit",
                    30.0,
                    "USD_millions",
                    "https://data.sec.gov",
                ),
                (
                    "XOM",
                    2025,
                    "2025-12-31",
                    "operating_income",
                    20.0,
                    "USD_millions",
                    "https://data.sec.gov",
                ),
                (
                    "XOM",
                    2025,
                    "2025-12-31",
                    "net_income",
                    15.0,
                    "USD_millions",
                    "https://data.sec.gov",
                ),
                ("XOM", 2025, "2025-12-31", "cfo", 25.0, "USD_millions", "https://data.sec.gov"),
                ("XOM", 2025, "2025-12-31", "capex", 5.0, "USD_millions", "https://data.sec.gov"),
                ("XOM", 2025, "2025-12-31", "cash", 10.0, "USD_millions", "https://data.sec.gov"),
                (
                    "XOM",
                    2025,
                    "2025-12-31",
                    "total_debt",
                    30.0,
                    "USD_millions",
                    "https://data.sec.gov",
                ),
            ],
        )

    assert compute_fundamentals_for_ticker("XOM", as_of_date="2026-03-19") is True
    with get_db() as conn:
        row = conn.execute(
            "SELECT metrics_json, quality_flags_json FROM fundamentals WHERE ticker='XOM' AND as_of_date='2026-03-19'"
        ).fetchone()
    assert row is not None
    assert '"fcf": 20.0' in row["metrics_json"]
    assert '"source_resolution": "companyfacts_fallback"' in row["quality_flags_json"]


def test_compute_fundamentals_rejects_companyfacts_filed_after_asof(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.fundamentals.metrics.ensure_facts",
        lambda _ticker: None,
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, filed_date,
                line_item, value, units, source_url, fetched_at, form,
                accession
            )
            VALUES(
                'LATE', 2025, 'FY', '2025-12-31', '2026-07-01',
                'revenue', 999.0, 'USD_millions',
                'https://data.sec.gov/late',
                '2026-07-01T00:00:00+00:00', '10-K',
                '0000000002-26-000001'
            )
            """
        )

    assert (
        compute_fundamentals_for_ticker(
            "LATE",
            as_of_date="2026-06-30",
        )
        is False
    )
    with get_db() as conn:
        row = conn.execute("SELECT 1 FROM fundamentals WHERE ticker = 'LATE'").fetchone()
    assert row is None


def test_compute_fundamentals_selects_latest_filing_visible_asof(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        visible_id = _insert_filing(
            conn,
            cik="0000000003",
            ticker="PIT",
            accession="0003-visible",
            form_type="10-K",
            filing_date="2025-02-15",
            period_end="2024-12-31",
            status="parsed",
        )
        _insert_financial_map(
            conn,
            visible_id,
            revenue=100.0,
            gross_profit=40.0,
            operating_income=20.0,
            net_income=15.0,
            cfo=18.0,
            capex=2.0,
            cash=5.0,
            total_debt=7.0,
        )
        future_id = _insert_filing(
            conn,
            cik="0000000003",
            ticker="PIT",
            accession="0003-future",
            form_type="10-K",
            filing_date="2026-02-15",
            period_end="2025-12-31",
            status="parsed",
        )
        _insert_financial_map(
            conn,
            future_id,
            revenue=999.0,
            gross_profit=900.0,
            operating_income=800.0,
            net_income=700.0,
            cfo=600.0,
            capex=1.0,
            cash=500.0,
            total_debt=0.0,
        )

    assert compute_fundamentals_for_ticker(
        "PIT",
        as_of_date="2025-12-31",
    )
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT metrics_json
            FROM fundamentals
            WHERE ticker = 'PIT'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    assert json.loads(row["metrics_json"])["revenue"] == 100.0


def test_compute_fundamentals_for_ticker_ignores_newer_8k(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        filing_id = _insert_filing(
            conn,
            cik="0000000001",
            ticker="TST",
            accession="0001-10k",
            form_type="10-K",
            filing_date="2025-02-01",
            period_end="2024-12-31",
            status="parsed",
        )
        _insert_financial_map(
            conn,
            filing_id,
            revenue=1000.0,
            gross_profit=400.0,
            operating_income=200.0,
            net_income=150.0,
            cfo=180.0,
            capex=20.0,
            cash=50.0,
            total_debt=70.0,
        )
        _insert_filing(
            conn,
            cik="0000000001",
            ticker="TST",
            accession="0001-8k",
            form_type="8-K",
            filing_date="2025-03-01",
            period_end="2025-03-01",
            status="parsed",
        )

    assert compute_fundamentals_for_ticker("TST") is True

    with get_db() as conn:
        row = conn.execute(
            "SELECT as_of_date, metrics_json FROM fundamentals WHERE ticker = 'TST' ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert row["as_of_date"] == "2025-02-01"
    metrics = json.loads(row["metrics_json"])
    assert metrics["revenue"] == 1000.0
    assert metrics["gross_margin"] == 0.4


def test_compute_all_fundamentals_uses_ok_status_supported_filings(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    with get_db() as conn:
        filing_id = _insert_filing(
            conn,
            cik="0000000002",
            ticker="OKY",
            accession="0002-10k",
            form_type="10-K",
            filing_date="2025-02-15",
            period_end="2024-12-31",
            status="OK",
        )
        _insert_financial_map(
            conn,
            filing_id,
            revenue=900.0,
            gross_profit=360.0,
            operating_income=180.0,
            net_income=140.0,
            cfo=170.0,
            capex=30.0,
            cash=80.0,
            total_debt=110.0,
        )
        _insert_filing(
            conn,
            cik="0000000002",
            ticker="OKY",
            accession="0002-8k",
            form_type="8-K",
            filing_date="2025-03-05",
            period_end="2025-03-05",
            status="parsed",
        )

    assert compute_all_fundamentals() == 1

    with get_db() as conn:
        row = conn.execute(
            "SELECT as_of_date, metrics_json FROM fundamentals WHERE ticker = 'OKY' ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert row["as_of_date"] == "2025-02-15"
    metrics = json.loads(row["metrics_json"])
    assert metrics["revenue"] == 900.0
    assert metrics["fcf"] == 140.0
