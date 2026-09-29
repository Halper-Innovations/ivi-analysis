from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

from app.db import init_db
from app.valuation.price_provider import PriceQuote
from app.valuation.tech_category import ENTERPRISE_SOFTWARE, TRADITIONAL_OPERATING


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn=conn)
    return conn


def _seed_companyfacts(conn: sqlite3.Connection, ticker: str = "TST") -> None:
    now = "2026-01-01T00:00:00+00:00"
    rows = [
        ("cfo", 2024, 120.0),
        ("cfo", 2023, 110.0),
        ("cfo", 2022, 100.0),
        ("capex", 2024, 15.0),
        ("capex", 2023, 14.0),
        ("capex", 2022, 12.0),
        ("operating_income", 2024, 80.0),
        ("operating_income", 2023, 75.0),
        ("operating_income", 2022, 70.0),
        ("operating_income", 2021, 65.0),
        ("operating_income", 2020, 60.0),
        ("net_income", 2024, 60.0),
        ("net_income", 2023, 55.0),
        ("net_income", 2022, 50.0),
        ("revenue", 2024, 500.0),
        ("revenue", 2023, 450.0),
        ("revenue", 2022, 400.0),
        ("revenue", 2021, 350.0),
        ("revenue", 2020, 300.0),
        ("total_debt", 2024, 50.0),
        ("cash", 2024, 30.0),
        ("preferred_equity", 2024, 0.0),
        ("noncontrolling_interest", 2024, 0.0),
        ("equity", 2024, 200.0),
        ("shares_outstanding", 2024, 10.0),
        ("total_liabilities", 2024, 300.0),
        ("sbc", 2024, 5.0),
    ]
    for line_item, fiscal_year, value in rows:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, accession
            )
            VALUES(?, ?, 'FY', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                fiscal_year,
                f"{fiscal_year}-12-31",
                line_item,
                value,
                "USD_millions",
                f"https://example.test/companyfacts/{ticker}/{line_item}",
                now,
                f"{fiscal_year + 1}-02-15",
                f"{ticker}-{fiscal_year}",
            ),
        )
    conn.commit()


def _fake_quote(price: float | None, ticker: str = "TST") -> PriceQuote:
    now = "2026-01-01T00:00:00+00:00"
    return PriceQuote(
        ticker=ticker,
        as_of_date="2026-03-19",
        price=price,
        currency="USD",
        provider="fake",
        status="OK" if price else "UNKNOWN",
        source_url="",
        fetched_at=now,
        expires_at=now,
        provenance={},
    )


def test_ensure_valuation_writes_adjusted_methods_for_non_traditional_category():
    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    adjusted_payload = {
        "status": "OK",
        "category": ENTERPRISE_SOFTWARE,
        "amortization_life": 3,
        "rnd_adjustment": 20.0,
        "adjusted_owner_earnings": 122.6,
        "time_series": [
            {"year": 2020, "adjusted_operating_income": 70.0},
            {"year": 2021, "adjusted_operating_income": 75.0},
            {"year": 2022, "adjusted_operating_income": 80.0},
            {"year": 2023, "adjusted_operating_income": 85.0},
            {"year": 2024, "adjusted_operating_income": 90.0},
        ],
        "flags": [],
        # Post-audit unit guard: the writer only merges payloads whose inputs
        # were confirmed scaled to millions.
        "guardrails": {
            "input_unit": "USD",
            "output_unit": "USD_millions",
            "input_unit_scale": 1_000_000.0,
        },
    }

    with (
        patch("app.valuation.valuation_writer.get_db") as mock_db,
        patch(
            "app.valuation.valuation_writer.resolve_financial_facts_asof",
            return_value={"cache_path": "", "derived_from": ["facts.TST"]},
        ),
        patch(
            "app.valuation.valuation_writer.classify_company_category",
            return_value={
                "category": ENTERPRISE_SOFTWARE,
                "confidence": "HIGH",
                "derived_from": ["category.TST"],
            },
        ),
        patch(
            "app.valuation.valuation_writer.compute_rnd_adjusted_earnings",
            return_value=adjusted_payload,
        ),
    ):
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    methods = {
        row["method"]
        for row in conn.execute("SELECT method FROM valuations WHERE ticker='TST'").fetchall()
    }
    assert "dcf_adjusted" in methods
    assert "epv_adjusted" in methods
    assert "tech_adjustment" in methods

    dcf_adjusted_row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker='TST' AND method='dcf_adjusted'"
    ).fetchone()
    dcf_adjusted = json.loads(dcf_adjusted_row["outputs_json"])
    assert dcf_adjusted["adjustment_metadata"]["category"] == ENTERPRISE_SOFTWARE
    assert dcf_adjusted["adjustment_metadata"]["amortization_life"] == 3
    assert dcf_adjusted["adjustment_metadata"]["rnd_adjustment"] == 20.0

    scorecard_row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker='TST' AND method='scorecard'"
    ).fetchone()
    scorecard = json.loads(scorecard_row["outputs_json"])
    assert "tech_valuation_divergence" in scorecard
    assert "track_comparison" in scorecard


def test_tech_adjustment_persists_post_guard_unit_rejection():
    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)
    ambiguous_payload = {
        "status": "OK",
        "category": ENTERPRISE_SOFTWARE,
        "amortization_life": 3,
        "rnd_adjustment": 20.0,
        "adjusted_owner_earnings": 122.6,
        "time_series": [
            {"year": 2024, "rnd_adjustment": 20.0},
        ],
        "flags": [],
        "guardrails": {"input_unit_scale": 1_000_000.0},
    }

    with (
        patch("app.valuation.valuation_writer.get_db") as mock_db,
        patch(
            "app.valuation.valuation_writer.resolve_financial_facts_asof",
            return_value={"cache_path": "", "derived_from": ["facts.TST"]},
        ),
        patch(
            "app.valuation.valuation_writer.classify_company_category",
            return_value={
                "category": ENTERPRISE_SOFTWARE,
                "confidence": "HIGH",
                "derived_from": ["category.TST"],
            },
        ),
        patch(
            "app.valuation.valuation_writer.compute_rnd_adjusted_earnings",
            return_value=ambiguous_payload,
        ),
    ):
        mock_db.return_value.__enter__ = lambda _self: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    methods = {
        row["method"]
        for row in conn.execute("SELECT method FROM valuations WHERE ticker='TST'").fetchall()
    }
    tech = json.loads(
        conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker='TST' AND method='tech_adjustment'"
        ).fetchone()["outputs_json"]
    )

    assert "dcf_adjusted" in methods
    assert "epv_adjusted" in methods
    assert tech["rnd_adjustment"]["status"] == "RND_UNIT_SCALE_AMBIGUOUS"
    dcf_adjusted = json.loads(
        conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker='TST' AND method='dcf_adjusted'"
        ).fetchone()["outputs_json"]
    )
    assert dcf_adjusted["status"] == "METHOD_INSUFFICIENT_DATA"
    assert dcf_adjusted["adjustment_metadata"]["status"] == ("RND_UNIT_SCALE_AMBIGUOUS")


def test_ensure_valuation_keeps_only_gaap_methods_for_traditional_operating():
    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    with (
        patch("app.valuation.valuation_writer.get_db") as mock_db,
        patch(
            "app.valuation.valuation_writer.resolve_financial_facts_asof",
            return_value={"cache_path": "", "derived_from": ["facts.TST"]},
        ),
        patch(
            "app.valuation.valuation_writer.classify_company_category",
            return_value={
                "category": TRADITIONAL_OPERATING,
                "confidence": "HIGH",
                "derived_from": ["category.TST"],
            },
        ),
        patch("app.valuation.valuation_writer.compute_rnd_adjusted_earnings") as compute_adjusted,
    ):
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    compute_adjusted.assert_not_called()
    methods = {
        row["method"]
        for row in conn.execute("SELECT method FROM valuations WHERE ticker='TST'").fetchall()
    }
    assert "dcf_adjusted" not in methods
    assert "epv_adjusted" not in methods
    assert "tech_adjustment" not in methods


def test_scorecard_flags_implausible_adjustment_divergence():
    conn = _make_conn()
    _seed_companyfacts(conn)
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0)

    adjusted_payload = {
        "status": "OK",
        "category": ENTERPRISE_SOFTWARE,
        "amortization_life": 3,
        "rnd_adjustment": 20.0,
        "adjusted_owner_earnings": 122_600_000.0,
        "time_series": [
            {"year": 2020, "adjusted_operating_income": 70_000_000.0},
            {"year": 2021, "adjusted_operating_income": 75_000_000.0},
            {"year": 2022, "adjusted_operating_income": 80_000_000.0},
            {"year": 2023, "adjusted_operating_income": 85_000_000.0},
            {"year": 2024, "adjusted_operating_income": 90_000_000.0},
        ],
        "flags": [],
        # Post-audit unit guard: the writer only merges payloads whose inputs
        # were confirmed scaled to millions.
        "guardrails": {
            "input_unit": "USD",
            "output_unit": "USD_millions",
            "input_unit_scale": 1_000_000.0,
        },
    }

    with (
        patch("app.valuation.valuation_writer.get_db") as mock_db,
        patch(
            "app.valuation.valuation_writer.resolve_financial_facts_asof",
            return_value={"cache_path": "", "derived_from": ["facts.TST"]},
        ),
        patch(
            "app.valuation.valuation_writer.classify_company_category",
            return_value={
                "category": ENTERPRISE_SOFTWARE,
                "confidence": "HIGH",
                "derived_from": ["category.TST"],
            },
        ),
        patch(
            "app.valuation.valuation_writer.compute_rnd_adjusted_earnings",
            return_value=adjusted_payload,
        ),
    ):
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("TST", "2026-03-19", provider=fake_provider)

    scorecard_row = conn.execute(
        "SELECT outputs_json FROM valuations WHERE ticker='TST' AND method='scorecard'"
    ).fetchone()
    scorecard = json.loads(scorecard_row["outputs_json"])
    assert scorecard["tech_valuation_divergence"] is None
    assert scorecard["tech_valuation_divergence_flag"] == "ADJUSTMENT_IMPLAUSIBLE"


def test_dcf_adjusted_uses_durable_revenue_series_on_spike():
    """Review ANCHOR-4: dcf_adjusted was built on the RAW (spike-inflated)
    revenue series while the generic dcf_base inherited the durable
    correction — and select_anchor prefers the sector-specific adjusted
    anchor, re-importing the spike inflation on exactly the
    licensing/upfront-heavy R&D names the spike detector targets. When
    rev_series_durable is set, dcf_adjusted must be computed from it."""
    conn = _make_conn()
    now = "2026-01-01T00:00:00+00:00"
    rows = [
        ("cfo", 2024, 120.0),
        ("cfo", 2023, 110.0),
        ("cfo", 2022, 100.0),
        ("capex", 2024, 15.0),
        ("capex", 2023, 14.0),
        ("capex", 2022, 12.0),
        ("operating_income", 2024, 80.0),
        ("operating_income", 2023, 75.0),
        ("operating_income", 2022, 70.0),
        ("operating_income", 2021, 65.0),
        ("operating_income", 2020, 60.0),
        ("net_income", 2024, 60.0),
        ("net_income", 2023, 55.0),
        ("net_income", 2022, 50.0),
        # Latest-year licensing spike: 450 -> 1300 (2.9x, +850M) => durable base 450
        ("revenue", 2024, 1300.0),
        ("revenue", 2023, 450.0),
        ("revenue", 2022, 400.0),
        ("revenue", 2021, 350.0),
        ("revenue", 2020, 300.0),
        ("total_debt", 2024, 50.0),
        ("cash", 2024, 30.0),
        ("preferred_equity", 2024, 0.0),
        ("noncontrolling_interest", 2024, 0.0),
        ("equity", 2024, 200.0),
        ("shares_outstanding", 2024, 10.0),
        ("total_liabilities", 2024, 300.0),
        ("sbc", 2024, 5.0),
    ]
    for line_item, fiscal_year, value in rows:
        conn.execute(
            """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, accession
                )
                VALUES(?, ?, 'FY', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SPK",
                fiscal_year,
                f"{fiscal_year}-12-31",
                line_item,
                value,
                "USD_millions",
                f"https://example.test/companyfacts/SPK/{line_item}",
                now,
                f"{fiscal_year + 1}-02-15",
                f"SPK-{fiscal_year}",
            ),
        )
    conn.commit()
    fake_provider = MagicMock()
    fake_provider.get_quote.return_value = _fake_quote(150.0, ticker="SPK")

    adjusted_payload = {
        "status": "OK",
        "category": ENTERPRISE_SOFTWARE,
        "amortization_life": 3,
        "rnd_adjustment": 20.0,
        "adjusted_owner_earnings": 122.6,
        "time_series": [
            {"year": 2022, "adjusted_operating_income": 80.0},
            {"year": 2023, "adjusted_operating_income": 85.0},
            {"year": 2024, "adjusted_operating_income": 90.0},
        ],
        "flags": [],
        "guardrails": {
            "input_unit": "USD",
            "output_unit": "USD_millions",
            "input_unit_scale": 1_000_000.0,
        },
    }

    with (
        patch("app.valuation.valuation_writer.get_db") as mock_db,
        patch(
            "app.valuation.valuation_writer.resolve_financial_facts_asof",
            return_value={"cache_path": "", "derived_from": ["facts.SPK"]},
        ),
        patch(
            "app.valuation.valuation_writer.classify_company_category",
            return_value={
                "category": ENTERPRISE_SOFTWARE,
                "confidence": "HIGH",
                "derived_from": ["category.SPK"],
            },
        ),
        patch(
            "app.valuation.valuation_writer.compute_rnd_adjusted_earnings",
            return_value=adjusted_payload,
        ),
    ):
        mock_db.return_value.__enter__ = lambda s: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        from app.valuation.valuation_writer import ensure_valuation

        ensure_valuation("SPK", "2026-03-19", provider=fake_provider)

    dcf_adjusted = json.loads(
        conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker='SPK' AND method='dcf_adjusted'"
        ).fetchone()["outputs_json"]
    )
    assert "REVENUE_SERIES_DURABLE_BASE_APPLIED" in (dcf_adjusted.get("flags") or [])

    # And the GAAP durable dcf (quality_context.dcf_durable) must agree on
    # basis: both use the spike-corrected series, so the adjusted base must
    # be derived from the same durable growth inputs (not the raw 2.9x CAGR).
    scorecard = json.loads(
        conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker='SPK' AND method='scorecard'"
        ).fetchone()["outputs_json"]
    )
    qc = scorecard.get("quality_context") or {}
    assert (qc.get("dcf_durable") or {}).get("base") is not None
