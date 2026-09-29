from __future__ import annotations

import json
from types import SimpleNamespace

from app.db import get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _seed_fundamentals(ticker: str) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(
                ticker, as_of_date, metrics_json, quality_flags_json,
                created_at
            )
            VALUES(?, '2025-12-31', ?, '[]', ?)
            """,
            (
                ticker,
                json.dumps(
                    {
                        "revenue": 100.0,
                        "operating_margin": 0.2,
                        "fcf": 15.0,
                        "net_debt": 5.0,
                    }
                ),
                utc_now_iso(),
            ),
        )


def _seed_companyfacts_shares(
    ticker: str,
    *,
    fiscal_year: int,
    period_end: str,
    value: float,
    filed_date: str,
    accession: str,
    source_url: str,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                ?, ?, 'FY', ?, 'shares_outstanding', ?, 'shares_millions',
                ?, ?, ?, '10-K', ?
            )
            """,
            (
                ticker,
                fiscal_year,
                period_end,
                value,
                source_url,
                utc_now_iso(),
                filed_date,
                accession,
            ),
        )


class _Provider:
    def __init__(self) -> None:
        self.calls = 0

    def get_quote(self, ticker: str, as_of_date: str):
        self.calls += 1
        return SimpleNamespace(
            ticker=ticker,
            as_of_date=as_of_date,
            price=10.0,
            provider="fixture",
            status="OK",
            source_url="https://example.test/quote",
            fetched_at=f"{as_of_date}T12:00:00+00:00",
        )


def test_sanity_shares_are_strictly_filed_asof_and_source_proven(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_companyfacts_shares(
        "SAFE",
        fiscal_year=2025,
        period_end="2025-12-31",
        value=10.0,
        filed_date="2026-02-01",
        accession="safe-2026-000001",
        source_url="https://data.sec.gov/safe",
    )
    _seed_companyfacts_shares(
        "SAFE",
        fiscal_year=2026,
        period_end="2026-02-28",
        value=999.0,
        filed_date="2026-04-15",
        accession="safe-2026-000002",
        source_url="https://data.sec.gov/safe",
    )
    from app.valuation.sanity_checks import _latest_shares

    with get_db() as conn:
        evidence = _latest_shares(
            conn,
            "SAFE",
            run_as_of_date="2026-03-15",
        )

    assert evidence == {
        "value": 10.0,
        "units": "shares_millions",
        "raw_value": 10.0,
        "raw_units": "shares_millions",
        "period_end": "2025-12-31",
        "filed_date": "2026-02-01",
        "accession": "safe-2026-000001",
        "source_url": "https://data.sec.gov/safe",
        "source": "companyfacts",
    }


def test_sanity_valuation_refuses_unproven_shares_before_quote_or_write(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_fundamentals("UNPROVEN")
    _seed_companyfacts_shares(
        "UNPROVEN",
        fiscal_year=2025,
        period_end="2025-12-31",
        value=10.0,
        filed_date="2026-02-01",
        accession="unproven-2026-000001",
        source_url="",
    )
    provider = _Provider()
    from app.valuation.sanity_checks import run_valuation_for_ticker

    result = run_valuation_for_ticker(
        "UNPROVEN",
        provider=provider,
        run_as_of_date="2026-03-15",
    )

    assert result is False
    assert provider.calls == 0
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM valuations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM valuations_measurement").fetchone()[0] == 0


def test_sanity_outputs_are_measurements_not_current_product_rows(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_fundamentals("MEASURE")
    _seed_companyfacts_shares(
        "MEASURE",
        fiscal_year=2025,
        period_end="2025-12-31",
        value=10.0,
        filed_date="2026-02-01",
        accession="measure-2026-000001",
        source_url="https://data.sec.gov/measure",
    )
    provider = _Provider()
    from app.valuation.sanity_checks import run_valuation_for_ticker

    result = run_valuation_for_ticker(
        "MEASURE",
        provider=provider,
        run_as_of_date="2026-03-15",
    )

    assert result is True
    assert provider.calls == 1
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM valuations").fetchone()[0] == 0
        rows = conn.execute(
            """
            SELECT method, inputs_json
            FROM valuations_measurement
            WHERE ticker = 'MEASURE'
            ORDER BY method
            """
        ).fetchall()
    assert [row["method"] for row in rows] == [
        "dcf_lite",
        "multiples",
        "reverse_dcf",
    ]
    for row in rows:
        inputs = json.loads(row["inputs_json"])
        assert inputs["shares_outstanding"] == 10.0
        assert inputs["shares_evidence"]["accession"] == "measure-2026-000001"
        assert inputs["shares_evidence"]["filed_date"] == "2026-02-01"
        assert inputs["shares_evidence"]["source_url"] == "https://data.sec.gov/measure"
