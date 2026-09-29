# tests/test_companyfacts_quality_tags.py
"""New TAG_MAP families for income_tax_expense, pretax_income,
cost_of_revenue, and the re-ingest writer path that materializes them into
companyfacts_facts.

Exact-literal assertions over normalize_annual_facts_from_raw fixtures and over
the ensure_facts() re-ingest path. No live network/LLM; all inputs are inline
us-gaap payloads written to a temp cache and CIK resolution is monkeypatched.

NOTE: live-DB coverage-percentage checks (>=80% tax, >=50% ROIC, >=45%
gross-margin over the cached universe) are
operator-run queries against data/engine.db and are intentionally OUT OF SCOPE for
this automated suite. The test here verifies the deterministic core: that the
existing re-ingest mechanism (ensure_facts -> normalize_annual_facts_from_raw ->
companyfacts_facts upsert) now materializes the three new line_items end-to-end
with exact literal values.
"""
from __future__ import annotations

import json

import pytest


def _single_tag_payload(tag: str, val: float, end: str, start: str) -> dict:
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000000001-24-000001",
                                "end": end,
                                "start": start,
                                "val": val,
                                "form": "10-K",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                }
            }
        }
    }


def _row_for(facts: list[dict], line_item: str) -> dict | None:
    rows = [f for f in facts if f["line_item"] == line_item]
    return rows[0] if rows else None


def test_income_tax_expense_normalizes_to_millions():
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = _single_tag_payload(
        "IncomeTaxExpenseBenefit", 29_749_000_000.0, "2024-09-28", "2023-09-30"
    )
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    row = _row_for(facts, "income_tax_expense")
    assert row is not None
    assert row["value"] == pytest.approx(29749.0, rel=0.001)


def test_pretax_income_normalizes_to_millions():
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = _single_tag_payload(
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        123_485_000_000.0,
        "2024-09-28",
        "2023-09-30",
    )
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    row = _row_for(facts, "pretax_income")
    assert row is not None
    assert row["value"] == pytest.approx(123485.0, rel=0.001)


def test_cost_of_revenue_normalizes_to_millions():
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = _single_tag_payload(
        "CostOfGoodsAndServicesSold", 210_352_000_000.0, "2024-09-28", "2023-09-30"
    )
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    row = _row_for(facts, "cost_of_revenue")
    assert row is not None
    assert row["value"] == pytest.approx(210352.0, rel=0.001)


def test_cost_of_revenue_first_tag_wins():
    """CostOfRevenue is ranked first, so it wins over CostOfGoodsAndServicesSold."""
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = {
        "facts": {
            "us-gaap": {
                "CostOfRevenue": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000000001-24-000001",
                                "end": "2024-09-28",
                                "start": "2023-09-30",
                                "val": 100_000_000.0,
                                "form": "10-K",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
                "CostOfGoodsAndServicesSold": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000000001-24-000001",
                                "end": "2024-09-28",
                                "start": "2023-09-30",
                                "val": 999_000_000.0,
                                "form": "10-K",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
            }
        }
    }
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    row = _row_for(facts, "cost_of_revenue")
    assert row is not None
    assert row["value"] == pytest.approx(100.0, rel=0.001)


def test_negative_tax_benefit_retained():
    """IncomeTaxExpenseBenefit can be negative (a benefit) and must not be dropped."""
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = _single_tag_payload(
        "IncomeTaxExpenseBenefit", -5_000_000_000.0, "2024-09-28", "2023-09-30"
    )
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    row = _row_for(facts, "income_tax_expense")
    assert row is not None
    assert row["value"] == pytest.approx(-5000.0, rel=0.001)


def test_negative_pretax_loss_retained():
    """pretax_income can be negative for loss years and must not be dropped."""
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = _single_tag_payload(
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        -2_000_000_000.0,
        "2024-09-28",
        "2023-09-30",
    )
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    row = _row_for(facts, "pretax_income")
    assert row is not None
    assert row["value"] == pytest.approx(-2000.0, rel=0.001)


def _three_tag_payload() -> dict:
    """Raw companyfacts payload carrying all three quality tags for FY2024."""
    return {
        "facts": {
            "us-gaap": {
                "IncomeTaxExpenseBenefit": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000000001-24-000001",
                                "end": "2024-09-28",
                                "start": "2023-09-30",
                                "val": 29_749_000_000.0,
                                "form": "10-K",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000000001-24-000001",
                                "end": "2024-09-28",
                                "start": "2023-09-30",
                                "val": 123_485_000_000.0,
                                "form": "10-K",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
                "CostOfGoodsAndServicesSold": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000000001-24-000001",
                                "end": "2024-09-28",
                                "start": "2023-09-30",
                                "val": 210_352_000_000.0,
                                "form": "10-K",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
            }
        }
    }


def test_reingest_materializes_new_line_items_into_companyfacts_facts(tmp_path, monkeypatch):
    """Core: ensure_facts() re-ingest writes the three new line_items.

    Fixture-only end-to-end through the real writer path (no network/LLM): a raw
    companyfacts payload is written to a temp cache, CIK resolution is patched,
    and ensure_facts() upserts into a fresh temp companyfacts_facts. Asserts the
    three new line_items materialize with exact literal millions values.
    """
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "engine.db"))

    from app.config import get_config
    from app.market.company_facts_provider import companyfacts_cache_path

    get_config.cache_clear()

    from app.db import init_db
    from app.ingest import facts_writer

    init_db()

    cik = "0000000001"
    cache_path = companyfacts_cache_path(cik)
    cache_path.write_text(json.dumps(_three_tag_payload()), encoding="utf-8")

    monkeypatch.setattr(facts_writer, "resolve", lambda ticker: cik)

    facts_writer.ensure_facts("TESTCO", years_back=10)

    from app.db import get_db

    with get_db() as conn:
        rows = {
            r["line_item"]: r["value"]
            for r in conn.execute(
                "SELECT line_item, value FROM companyfacts_facts "
                "WHERE ticker = 'TESTCO' AND period_type = 'FY' AND fiscal_year = 2024"
            ).fetchall()
        }

    present = sorted(
        set(rows) & {"income_tax_expense", "pretax_income", "cost_of_revenue"}
    )
    assert present == ["cost_of_revenue", "income_tax_expense", "pretax_income"]
    assert rows["income_tax_expense"] == pytest.approx(29749.0, rel=0.001)
    assert rows["pretax_income"] == pytest.approx(123485.0, rel=0.001)
    assert rows["cost_of_revenue"] == pytest.approx(210352.0, rel=0.001)

    get_config.cache_clear()
