from __future__ import annotations

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


def _seed_facts(ticker: str, facts: list[tuple[int, str, float, str]]) -> None:
    now = utc_now_iso()
    with get_db() as conn:
        for fiscal_year, line_item, value, period_end in facts:
            units = "shares_millions" if line_item == "shares_outstanding" else "USD_millions"
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession
                )
                VALUES(
                    ?, ?, 'FY', ?, ?, ?, ?,
                    'https://example.test/companyfacts', ?, '2026-02-01',
                    '10-K', ?
                )
                """,
                (
                    ticker,
                    fiscal_year,
                    period_end,
                    line_item,
                    value,
                    units,
                    now,
                    f"{ticker}-2026-000001",
                ),
            )


def _seed_company_and_filing(ticker: str, *, name: str, text: str, tmp_path) -> None:
    now = utc_now_iso()
    path = tmp_path / f"{ticker.lower()}_10k.txt"
    path.write_text(text, encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO companies(ticker, cik, name, created_at) VALUES(?, '0000000001', ?, ?)",
            (ticker, name, now),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            )
            VALUES('0000000001', ?, ?, '10-K', '2026-02-01', '2025-12-31', 'https://example.test/10k', ?, 'OK', ?, ?)
            """,
            (ticker, f"{ticker}-2026", str(path), now, now),
        )


def test_common_residual_income_valuation_uses_literal_expected_values(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "ICMN",
        [
            (2025, "equity", 1000.0, "2025-12-31"),
            (2025, "shares_outstanding", 100.0, "2025-12-31"),
            (2025, "net_income", 100.0, "2025-12-31"),
            (2024, "net_income", 120.0, "2024-12-31"),
            (2023, "net_income", 80.0, "2023-12-31"),
        ],
    )

    from app.insurance.valuation import calculate_insurance_common_valuation

    result = calculate_insurance_common_valuation(
        "ICMN", as_of_date="2026-02-01", current_price=8.0
    )

    assert result["status"] == "OK"
    assert result["adjusted_book_value_per_share"] == 10.0
    assert result["average_net_income"] == 100.0
    assert result["normalized_roe"] == 0.1
    assert result["justified_price_to_book"] == 1.0
    assert result["residual_income_value_per_share"] == 10.0
    assert result["valuation_anchor"] == 10.0
    assert result["upside_to_anchor"] == 0.2


def test_common_valuation_missing_book_blocks_model(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "IBLK",
        [
            (2025, "shares_outstanding", 100.0, "2025-12-31"),
            (2025, "net_income", 100.0, "2025-12-31"),
        ],
    )

    from app.insurance.valuation import calculate_insurance_common_valuation

    result = calculate_insurance_common_valuation(
        "IBLK", as_of_date="2026-02-01", current_price=8.0
    )

    assert result["status"] == "MODEL_BLOCKED"
    assert result["valuation_anchor"] is None
    assert result["reason_codes"] == ["MISSING_OR_NONPOSITIVE_BOOK_VALUE"]


def test_preferred_valuation_uses_literal_yield_values(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company_and_filing(
        "IPRF",
        name="Example Insurer Depositary Shares 6.000% Non-Cumulative Preferred Series A",
        text=(
            "Each depositary share represents preferred stock with a $25 liquidation preference. "
            "The dividend rate is 6.000%. The preferred stock is non-cumulative and redeemable "
            "on or after January 1, 2030."
        ),
        tmp_path=tmp_path,
    )

    from app.insurance.valuation import calculate_insurance_preferred_valuation

    result = calculate_insurance_preferred_valuation(
        "IPRF", as_of_date="2026-02-01", current_price=20.0
    )

    assert result["status"] == "OK"
    assert result["valuation_anchor"] == 25.0
    assert result["preferred_terms"]["coupon_rate"] == 0.06
    assert result["preferred_terms"]["liquidation_preference"] == 25.0
    assert result["preferred_terms"]["cumulative"] is False
    assert result["preferred_terms"]["annual_coupon"] == 1.5
    assert result["current_yield"] == 0.075
    assert result["yield_to_worst"] == 0.075
    assert result["upside_to_liquidation_preference"] == 0.2


def test_preferred_valuation_missing_terms_blocks_model(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company_and_filing(
        "IPRX",
        name="Example Insurer Preferred Series B",
        text="Preferred terms are not available in this cached excerpt.",
        tmp_path=tmp_path,
    )

    from app.insurance.valuation import calculate_insurance_preferred_valuation

    result = calculate_insurance_preferred_valuation(
        "IPRX", as_of_date="2026-02-01", current_price=20.0
    )

    assert result["status"] == "MODEL_BLOCKED"
    assert result["valuation_anchor"] is None
    assert result["reason_codes"] == [
        "PREFERRED_COUPON_MISSING",
        "LIQUIDATION_PREFERENCE_MISSING",
        "CUMULATIVE_STATUS_MISSING",
    ]
