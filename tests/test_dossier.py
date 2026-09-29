from __future__ import annotations

import json
import math
from pathlib import Path

from app.db import get_db, init_db, utc_now_iso
from app.dossier.collector import DossierFiling
from app.dossier.extractors import extract_annual_items
from app.dossier.peer_report import build_peer_report
from app.dossier.sections import (
    event_category_for_8k_section,
    item_code_for_8k_section,
    segment_10k_sections,
    segment_10q_sections,
    segment_8k_sections,
)
from app.dossier.time_series import build_time_series


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _seed_filing_rows(filing_id: int):
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO filings(
                id, cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url,
                local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(?, '1', 'TEST', '0001-2025', '10-K', '2025-12-31', '2025-12-31',
                     'https://www.sec.gov/Archives/edgar/data/1/0001/doc.htm',
                     'tests/fixtures/dossier_10k_sample_2025.html', 'h', '2026-02-13', 'parsed', ?, ?)
            """,
            (filing_id, now, now),
        )
        for line_item, value in [
            ("revenue", 120000.0),
            ("gross_profit", 72000.0),
            ("operating_income", 24000.0),
            ("cfo", 26000.0),
            ("capex", 6000.0),
            ("cash", 18000.0),
            ("total_debt", 15000.0),
        ]:
            conn.execute(
                """
                INSERT INTO companyfacts_facts
                    (ticker, fiscal_year, period_type, period_end, line_item,
                     value, units, source_url, fetched_at, filed_date, form,
                     accession)
                VALUES (
                    'TEST', 2025, 'FY', '2025-12-31', ?, ?,
                    'USD_millions', 'https://data.sec.gov', ?,
                    '2025-12-31', '10-K', '0001-2025'
                )
                ON CONFLICT(ticker, fiscal_year, period_type, line_item) DO NOTHING
                """,
                (line_item, value, now),
            )
        conn.execute(
            """
            INSERT INTO extracted_facts(
                filing_id, fact_type, value_json, source_url, snippet, section_label, created_at
            ) VALUES
            (?, 'segments_signal', '{"segment_names":["Cloud","Subscription"]}', 'https://www.sec.gov/doc', 'segment snippet', 'segment_info', ?),
            (?, 'sbc_dilution_signal', '{"present":true}', 'https://www.sec.gov/doc', 'sbc snippet', 'equity_dilution', ?),
            (?, 'customer_concentration_signal', '{"customer_pct":12}', 'https://www.sec.gov/doc', 'customer snippet', 'customer_concentration', ?),
            (?, 'shares_outstanding', '{"value":1000000}', 'https://www.sec.gov/doc', 'shares snippet', 'cover_page', ?)
            """,
            (filing_id, now, filing_id, now, filing_id, now, filing_id, now),
        )


def test_dossier_section_splitting_and_extractor_stability(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(1)
    text = (Path(__file__).parent / "fixtures/dossier_10k_sample_2025.html").read_text(
        encoding="utf-8"
    )
    sections = segment_10k_sections(text)
    labels = {s.section_label for s in sections}
    assert {"business", "risk_factors", "md_and_a", "financial_statements", "notes"}.issubset(
        labels
    )

    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(Path(__file__).parent / "fixtures/dossier_10k_sample_2025.html"),
        filing_id=1,
    )
    items = extract_annual_items(filing=filing, sections=sections)
    metric_map = {item["metric"]: item["value"] for item in items}
    assert metric_map["gross_margin"] == 0.6
    assert metric_map["operating_margin"] == 0.2
    assert metric_map["fcf_margin"] == (26000.0 - 6000.0) / 120000.0
    assert metric_map["r_and_d_total"] == 18000.0
    assert metric_map["sales_marketing_total"] == 22000.0
    assert metric_map["g_and_a_total"] == 9000.0
    assert metric_map["deferred_revenue_amount"] == 40000.0
    assert metric_map["rpo_amount"] == 120000.0
    assert metric_map["share_repurchases_amount"] == 5000.0
    assert metric_map["dividends_paid_amount"] == 3000.0
    assert metric_map["risk_factor_keyword_count"] == 1
    assert metric_map["acquisition_mentions_count"] == 1
    assert metric_map["deferred_revenue_mention"] == 1
    assert metric_map["rpo_mention"] == 1
    assert metric_map["customer_concentration_present"] == 1
    assert metric_map["customer_concentration_pct"] == 0.12
    assert metric_map["segment_count"] == 2
    assert metric_map["sbc_dilution_indicator"] == 1
    segment_names_item = next(item for item in items if item["metric"] == "segment_names")
    assert segment_names_item["value"] == ["Cloud", "Subscription"]


def test_dossier_prefers_companyfacts_shares_over_cover_page(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(2)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts
                (ticker, fiscal_year, period_type, period_end, line_item,
                 value, units, source_url, fetched_at, filed_date, form,
                 accession)
            VALUES (
                'TEST', 2025, 'FY', '2025-12-31',
                'shares_outstanding', 1234.0, 'shares_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                ?, '2025-12-31', '10-K', '0001-2025'
            )
            ON CONFLICT(ticker, fiscal_year, period_type, line_item) DO UPDATE SET
                value=excluded.value, source_url=excluded.source_url, fetched_at=excluded.fetched_at
            """,
            (now,),
        )

    text = (Path(__file__).parent / "fixtures/dossier_10k_sample_2025.html").read_text(
        encoding="utf-8"
    )
    sections = segment_10k_sections(text)
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(Path(__file__).parent / "fixtures/dossier_10k_sample_2025.html"),
        filing_id=2,
    )
    items = extract_annual_items(filing=filing, sections=sections)
    shares_item = next(item for item in items if item["metric"] == "shares_outstanding")
    assert shares_item["value"] == 1234.0
    assert shares_item["section_label"] == "financial_statements"
    assert shares_item["derived_from"] == ["financials.shares_outstanding"]
    assert (
        shares_item["source_url"] == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    )


def test_dossier_time_series_derived_from_traces():
    items = [
        {"year": 2024, "metric": "revenue", "value": 100.0},
        {"year": 2025, "metric": "revenue", "value": 120.0},
        {"year": 2024, "metric": "gross_margin", "value": 0.50},
        {"year": 2025, "metric": "gross_margin", "value": 0.60},
        {"year": 2024, "metric": "operating_margin", "value": 0.15},
        {"year": 2025, "metric": "operating_margin", "value": 0.20},
        {"year": 2024, "metric": "fcf_margin", "value": 0.10},
        {"year": 2025, "metric": "fcf_margin", "value": 0.17},
        {"year": 2024, "metric": "shares_outstanding", "value": 1000.0},
        {"year": 2025, "metric": "shares_outstanding", "value": 1100.0},
        {"year": 2024, "metric": "gross_profit_dollars", "value": 50.0},
        {"year": 2025, "metric": "gross_profit_dollars", "value": 72.0},
        {"year": 2024, "metric": "risk_factor_keyword_count", "value": 4},
        {"year": 2025, "metric": "risk_factor_keyword_count", "value": 6},
        {"year": 2024, "metric": "r_and_d_total", "value": 15.0},
        {"year": 2025, "metric": "r_and_d_total", "value": 18.0},
        {"year": 2024, "metric": "deferred_revenue_amount", "value": 35.0},
        {"year": 2025, "metric": "deferred_revenue_amount", "value": 40.0},
        {"year": 2024, "metric": "rpo_amount", "value": 100.0},
        {"year": 2025, "metric": "rpo_amount", "value": 120.0},
        {"year": 2024, "metric": "customer_concentration_pct", "value": 0.10},
        {"year": 2025, "metric": "customer_concentration_pct", "value": 0.12},
        {"year": 2024, "metric": "segment_count", "value": 2},
        {"year": 2025, "metric": "segment_count", "value": 3},
    ]
    ts = build_time_series(items)
    assert ts["years"] == [2024, 2025]
    assert ts["derived_signals"]
    latest_std = ts["standardized_rows"][-1]
    assert latest_std["r_and_d_total"] == 18.0
    assert latest_std["risk_factor_keyword_count"] == 6
    assert latest_std["acquisition_mentions_count"] == "UNKNOWN"
    assert latest_std["deferred_revenue_amount"] == 40.0
    assert latest_std["rpo_amount"] == 120.0
    assert latest_std["customer_concentration_pct"] == 0.12
    assert latest_std["segment_count"] == 3
    latest_trace = ts["standardized_row_traces"]["2025"]
    assert latest_trace["risk_factor_keyword_count"]["citations"] == []
    assert latest_trace["acquisition_mentions_count"]["derived_from"] == []
    signal_map = {signal["signal"]: signal["value"] for signal in ts["derived_signals"]}
    assert signal_map["r_and_d_intensity_latest"] == 18.0 / 120.0
    assert signal_map["r_and_d_intensity_delta"] == (18.0 / 120.0) - (15.0 / 100.0)
    assert signal_map["segment_count_delta"] == 1.0
    assert math.isclose(signal_map["customer_concentration_delta"], 0.02, rel_tol=0.0, abs_tol=1e-9)
    assert signal_map["deferred_revenue_to_revenue_latest"] == 40.0 / 120.0
    assert signal_map["rpo_to_revenue_latest"] == 120.0 / 120.0
    for signal in ts["derived_signals"]:
        assert signal.get("derived_from")


def test_dossier_expense_extraction_fails_safe_on_ambiguous_rows(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(3)
    text = """
    <html><body>
    <h2>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</h2>
    <table>
    <tr><td>Research and Development and Sales and Marketing</td><td>18000</td><td>22000</td></tr>
    </table>
    </body></html>
    """
    sections = segment_10k_sections(text)
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(Path(__file__).parent / "fixtures/dossier_10k_sample_2025.html"),
        filing_id=3,
    )
    items = extract_annual_items(filing=filing, sections=sections)
    metric_map = {item["metric"]: item["value"] for item in items}
    assert metric_map["r_and_d_total"] == "UNKNOWN"
    assert metric_map["sales_marketing_total"] == "UNKNOWN"


def test_dossier_normalizes_filing_expense_units_to_millions(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(30)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts
                (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
            VALUES ('TEST', 2025, 'FY', '2025-12-31', 'revenue', 707.63, 'USD_millions',
                    'https://data.sec.gov', ?)
            ON CONFLICT(ticker, fiscal_year, period_type, line_item) DO UPDATE SET
                value=excluded.value, source_url=excluded.source_url, fetched_at=excluded.fetched_at
            """,
            (now,),
        )
    html = """
    <html><body>
    <h2>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</h2>
    <table>
    <caption>Amounts in thousands</caption>
    <tr><td>Research and development</td><td>187708</td></tr>
    <tr><td>Sales and marketing</td><td>223511</td></tr>
    <tr><td>General and administrative</td><td>115289</td></tr>
    </table>
    </body></html>
    """
    temp = tmp_path / "normalized_expenses.html"
    temp.write_text(html, encoding="utf-8")
    sections = segment_10k_sections(html)
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(temp),
        filing_id=30,
    )
    items = extract_annual_items(filing=filing, sections=sections)
    metric_map = {item["metric"]: item["value"] for item in items}
    assert math.isclose(metric_map["r_and_d_total"], 187.708, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(metric_map["sales_marketing_total"], 223.511, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(metric_map["g_and_a_total"], 115.289, rel_tol=0.0, abs_tol=1e-9)


def test_dossier_preserves_sub_million_expense_with_explicit_dollar_unit(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(31)
    html = """
    <html><body>
    <h2>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</h2>
    <table>
    <caption>Amounts in dollars</caption>
    <tr><td>Research and development</td><td>900000</td></tr>
    </table>
    </body></html>
    """
    temp = tmp_path / "sub_million_expense.html"
    temp.write_text(html, encoding="utf-8")
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(temp),
        filing_id=31,
    )
    items = extract_annual_items(filing=filing, sections=segment_10k_sections(html))
    metric_map = {item["metric"]: item["value"] for item in items}
    assert metric_map["r_and_d_total"] == 0.9


def test_dossier_rejects_expense_without_explicit_unit(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(32)
    html = """
    <html><body>
    <h2>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</h2>
    <table>
    <tr><td>Research and development</td><td>900000</td></tr>
    </table>
    </body></html>
    """
    temp = tmp_path / "unitless_expense.html"
    temp.write_text(html, encoding="utf-8")
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(temp),
        filing_id=32,
    )
    items = extract_annual_items(filing=filing, sections=segment_10k_sections(html))
    metric_map = {item["metric"]: item["value"] for item in items}
    assert metric_map["r_and_d_total"] == "UNKNOWN"


def test_dossier_customer_concentration_negative_disclosure_sets_present_zero(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(4)
    with get_db() as conn:
        conn.execute(
            "DELETE FROM extracted_facts WHERE filing_id = ? AND fact_type = 'customer_concentration_signal'",
            (4,),
        )
    html = """
    <html><body>
    <h2>ITEM 1. BUSINESS</h2><p>Example.</p>
    <h2>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</h2><p>Example.</p>
    <h2>Notes to Consolidated Financial Statements</h2>
    <p>No sales to an individual customer or country other than the United States accounted for more than 10% of revenue for fiscal years 2025, 2024, or 2023.</p>
    </body></html>
    """
    temp = tmp_path / "customer_negative.html"
    temp.write_text(html, encoding="utf-8")
    sections = segment_10k_sections(html)
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(temp),
        filing_id=4,
    )
    items = extract_annual_items(filing=filing, sections=sections)
    metric_map = {item["metric"]: item["value"] for item in items}
    assert metric_map["customer_concentration_present"] == 0
    assert metric_map["customer_concentration_pct"] == "UNKNOWN"


def test_dossier_fallback_item_section_slicing_for_risk_and_mdna(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(5)
    html = """
    <html><body>
    <div id="item_1_business">ITEM 1. BUSINESS Example.</div>
    <div id="item_1a_risk_factors">
      ITEM 1A. RISK FACTORS
      Competition is intense. Cybersecurity incidents could disrupt operations.
    </div>
    <div id="item_1b_unresolved_staff_comments">ITEM 1B. UNRESOLVED STAFF COMMENTS</div>
    <div id="item_7_management_s_discussion_and_analysis">
      ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
      We acquired a small business this year as part of our acquisition strategy.
    </div>
    <div id="item_7a_quantitative_and_qualitative_disclosures_about_market_risk">ITEM 7A. MARKET RISK</div>
    <div id="item_8_financial_statements">ITEM 8. FINANCIAL STATEMENTS</div>
    </body></html>
    """
    temp = tmp_path / "fallback_items.html"
    temp.write_text(html, encoding="utf-8")
    sections = segment_10k_sections("<html><body><div>stub</div></body></html>")
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(temp),
        filing_id=5,
    )
    items = extract_annual_items(filing=filing, sections=sections)
    metric_map = {item["metric"]: item["value"] for item in items}
    assert metric_map["risk_factor_keyword_count"] == 2
    assert metric_map["acquisition_mentions_count"] == 1


def test_dossier_acquisition_mentions_excludes_traffic_and_customer_acquisition(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(6)
    html = """
    <html><body>
    <div id="item_7_management_s_discussion_and_analysis">
      ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
      Search advertising revenue increased due to traffic acquisition costs.
      Sales efficiency improved through lower customer acquisition costs.
      We acquired a complementary software business during the year.
    </div>
    <div id="item_7a_quantitative_and_qualitative_disclosures_about_market_risk">ITEM 7A. MARKET RISK</div>
    <div id="item_8_financial_statements">ITEM 8. FINANCIAL STATEMENTS</div>
    </body></html>
    """
    temp = tmp_path / "acquisition_filter.html"
    temp.write_text(html, encoding="utf-8")
    sections = segment_10k_sections("<html><body><div>stub</div></body></html>")
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(temp),
        filing_id=6,
    )
    items = extract_annual_items(filing=filing, sections=sections)
    metric_map = {item["metric"]: item["value"] for item in items}
    assert metric_map["acquisition_mentions_count"] == 1


def test_dossier_segment_names_fall_back_to_business_text(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing_rows(5)
    with get_db() as conn:
        conn.execute(
            "DELETE FROM extracted_facts WHERE filing_id = ? AND fact_type = 'segments_signal'",
            (5,),
        )
    text = (Path(__file__).parent / "fixtures/dossier_10k_sample_2025.html").read_text(
        encoding="utf-8"
    )
    sections = segment_10k_sections(text)
    filing = DossierFiling(
        ticker="TEST",
        cik="1",
        accession="0001-2025",
        form_type="10-K",
        filing_date="2025-12-31",
        period_end="2025-12-31",
        primary_doc_url="https://www.sec.gov/doc",
        local_path=str(Path(__file__).parent / "fixtures/dossier_10k_sample_2025.html"),
        filing_id=5,
    )
    items = extract_annual_items(filing=filing, sections=sections)
    metric_map = {item["metric"]: item["value"] for item in items}
    assert metric_map["segment_names"] == ["Cloud Platform", "Subscription"]
    assert metric_map["segment_count"] == 2


def test_peer_report_includes_all_tickers_and_citations(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    dossiers = [
        {
            "ticker": "AAA",
            "time_series": {
                "derived_signals": [
                    {"signal": "revenue_cagr_proxy", "value": 0.10, "derived_from": ["x"]},
                    {"signal": "gross_margin_delta", "value": 0.05, "derived_from": ["x"]},
                    {"signal": "operating_margin_delta", "value": 0.04, "derived_from": ["x"]},
                    {"signal": "fcf_margin_delta", "value": 0.03, "derived_from": ["x"]},
                    {"signal": "dilution_rate_proxy", "value": 0.01, "derived_from": ["x"]},
                ]
            },
            "claims": [
                {
                    "label": "revenue::2025",
                    "value": 120.0,
                    "citations": [
                        {
                            "source_url": "https://www.sec.gov/a",
                            "snippet": "rev",
                            "section_label": "financial_statements",
                        }
                    ],
                    "derived_from": ["x"],
                }
            ],
            "items": [],
        },
        {
            "ticker": "BBB",
            "time_series": {
                "derived_signals": [
                    {"signal": "revenue_cagr_proxy", "value": 0.03, "derived_from": ["x"]},
                    {"signal": "gross_margin_delta", "value": 0.01, "derived_from": ["x"]},
                    {"signal": "operating_margin_delta", "value": 0.01, "derived_from": ["x"]},
                    {"signal": "fcf_margin_delta", "value": 0.00, "derived_from": ["x"]},
                    {"signal": "dilution_rate_proxy", "value": 0.08, "derived_from": ["x"]},
                ]
            },
            "claims": [
                {
                    "label": "revenue::2025",
                    "value": 90.0,
                    "citations": [
                        {
                            "source_url": "https://www.sec.gov/b",
                            "snippet": "rev",
                            "section_label": "financial_statements",
                        }
                    ],
                    "derived_from": ["x"],
                }
            ],
            "items": [],
        },
    ]
    summary = build_peer_report(
        run_id="dossier_test",
        as_of_date="2026-02-13",
        dossiers=dossiers,
    )
    rankings_path = Path(summary["peer_rankings_path"])
    assert rankings_path.exists()
    payload = json.loads(rankings_path.read_text(encoding="utf-8"))
    assert payload["tickers"] == ["AAA", "BBB"]
    for row in payload["rankings"]:
        assert row["top_differentiators"]
        assert row["top_differentiators"][0]["citation"]["source_url"].startswith(
            "https://www.sec.gov/"
        )
    assert Path(cfg.dossiers_dir / "dossier_test" / "peer_report.md").exists()


def test_dossier_section_splitting_handles_realistic_html_anchor_headings():
    html = """
    <html><body>
    <p id="item_1_business"><span>ITEM 1. B</span><span>USINESS</span></p>
    <p>Business body text.</p>
    <p id="item_1a_risk_factors"><span>ITEM 1A. RIS</span><span>K FACTORS</span></p>
    <p>Risk factor body text.</p>
    <p id="item_7_managements_discussion_analysis_f"><span>Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</span></p>
    <p>MD&amp;A body text.</p>
    <p id="item_8_financial_statements_and_supplementary_data"><span>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</span></p>
    <p>Financial statements body text.</p>
    </body></html>
    """
    sections = segment_10k_sections(html)
    labels = {s.section_label for s in sections}
    assert {"business", "risk_factors", "md_and_a", "financial_statements"}.issubset(labels)


def test_dossier_section_skips_table_of_contents_picks_body():
    """Regression for HRMY-style filings: table-of-contents entries appear
    BEFORE the actual section bodies. The previous parser picked the TOC
    position (smallest index) and produced 471-char "business" spans
    containing only TOC links. The fix uses the LAST text-pattern match for
    each label so the body is preferred. Confirmed against a real HRMY 10-K
    where business went from 471 → 173,275 chars and risk_factors from
    5,551 → 645,390 chars after this change.
    """
    html = (
        "<html><body>"
        # --- Table of contents at the top (TOC links to sections) ---
        '<div class="toc">'
        '<p><a href="#item_1">ITEM 1. BUSINESS.</a></p>'
        '<p><a href="#item_1a">ITEM 1A. RISK FACTORS.</a></p>'
        '<p><a href="#item_7">ITEM 7. MANAGEMENT\'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</a></p>'
        "</div>"
        # Filler content between TOC and body
        + ("<p>Forward-looking statements and other front matter.</p>" * 50)
        +
        # --- Actual body sections ---
        '<p style="margin-top:24pt;"><b>Item&#160;1. Business.</b></p>'
        + (
            "<p>Real business body text. This is the actual content of the business section, "
            "describing the company products, markets, and competitive position. </p>" * 30
        )
        + '<p style="margin-top:24pt;"><b>Item&#160;1A. Risk Factors.</b></p>'
        + (
            "<p>Real risk factor text. The company faces material risks from competition, "
            "regulation, and macroeconomic conditions. </p>" * 30
        )
        + '<p style="margin-top:24pt;"><b>Item&#160;7. Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</b></p>'
        + ("<p>Real MD&A text describing operating results year over year. </p>" * 30)
        + "</body></html>"
    )
    sections = segment_10k_sections(html)
    by_label = {s.section_label: s for s in sections}

    # All three sections must be detected
    assert "business" in by_label
    assert "risk_factors" in by_label
    assert "md_and_a" in by_label

    # Critically: each section's text must contain real body content, not
    # just TOC entries. Body sections in this fixture are >= 1500 chars;
    # the TOC entries are only ~50 chars each.
    assert len(by_label["business"].text) > 500, (
        f"business section is {len(by_label['business'].text)} chars — "
        "looks like the parser picked the TOC entry, not the body"
    )
    assert "Real business body text" in by_label["business"].text
    assert "Real risk factor text" in by_label["risk_factors"].text
    assert "Real MD&A text" in by_label["md_and_a"].text


def test_dossier_notes_section_prefers_uppercase_header_over_cross_references():
    """JAZZ-style filings sprinkle 'Notes to Consolidated Financial
    Statements' as cross-references throughout the body (e.g.,
    'see Notes to Consolidated Financial Statements, included in Part IV').
    The actual section header is typically UPPERCASE in a styled div.
    The parser must pick the uppercase header even when many lower-case
    cross-references appear after it.
    """
    html = (
        "<html><body>"
        "<p>Cover page</p>"
        # 100 cross-references in mixed case — these would win under
        # last-match logic and produce a fragment of garbage.
        + (
            "<p>see Notes to Consolidated Financial Statements, "
            "included in Part IV of this Annual Report.</p>" * 100
        )
        +
        # The actual section header in uppercase
        '<div style="text-align:center"><b>NOTES TO CONSOLIDATED FINANCIAL STATEMENTS</b></div>'
        + ("<p>Real notes content goes here. " * 50)
        + "</body></html>"
    )
    sections = segment_10k_sections(html)
    by_label = {s.section_label: s for s in sections}
    assert "notes" in by_label
    assert "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS" in by_label["notes"].text
    assert "Real notes content goes here" in by_label["notes"].text


def test_dossier_notes_pattern_handles_optional_the():
    """ANIP-style filings write 'notes to THE consolidated financial
    statements' (with 'the' in the middle). Pattern must allow it."""
    html = (
        "<html><body>"
        "<p>Some prose.</p>"
        "<div><b>Notes to the consolidated financial statements</b></div>"
        + ("<p>Note 1. Description of the business. " * 30)
        + "</body></html>"
    )
    sections = segment_10k_sections(html)
    by_label = {s.section_label: s for s in sections}
    assert "notes" in by_label
    assert "Note 1" in by_label["notes"].text


def test_dossier_section_handles_html_nbsp_entity_in_item_header():
    """Body section headers in EDGAR filings frequently use `&#160;` (HTML
    non-breaking space) between 'Item' and the number/title. The previous
    `\\s+` patterns wouldn't match this entity, so the parser missed body
    headers entirely and fell back to TOC mentions. The fix makes the
    whitespace gap explicitly accept `&#160;` and `&nbsp;`.
    """
    html = (
        "<html><body>"
        "<p>Cover page</p>"
        "<p><b>Item&#160;1. Business.</b></p>"
        "<p>Body of business section with substantial content. " * 20 + "</p>"
        "<p><b>Item&nbsp;1A. Risk Factors.</b></p>"
        "<p>Body of risk factors section with substantial content. " * 20 + "</p>"
        "</body></html>"
    )
    sections = segment_10k_sections(html)
    by_label = {s.section_label: s for s in sections}
    assert "business" in by_label
    assert "risk_factors" in by_label
    assert "Body of business section" in by_label["business"].text
    assert "Body of risk factors section" in by_label["risk_factors"].text


def test_dossier_section_handles_dash_and_colon_item_title_separators():
    """Readable filing section recovery should not depend on period-separated
    headings; EDGAR HTML often uses dash, em dash, en dash, or colon separators.
    """
    html = (
        "<html><body>"
        '<div class="toc">Item 1 - Business Item 1A - Risk Factors Item 7 - MD&amp;A</div>'
        "<h1>Item 1 — Business</h1>"
        + ("<p>Dash-separated business body with real operating context. </p>" * 20)
        + "<h1>Item 1A – Risk Factors</h1>"
        + ("<p>Dash-separated risk body with competitive and regulatory risks. </p>" * 20)
        + "<h1>Item 7: Management’s Discussion and Analysis of Financial Condition and Results of Operations</h1>"
        + ("<p>Colon-separated MD&A body with revenue and margin commentary. </p>" * 20)
        + "<h1>Item 8 - Financial Statements and Supplementary Data</h1>"
        + ("<p>Dash-separated financial statement body. </p>" * 20)
        + "</body></html>"
    )
    sections = segment_10k_sections(html)
    by_label = {s.section_label: s for s in sections}

    assert "Dash-separated business body" in by_label["business"].text
    assert "Dash-separated risk body" in by_label["risk_factors"].text
    assert "Colon-separated MD&A body" in by_label["md_and_a"].text
    assert "Dash-separated financial statement body" in by_label["financial_statements"].text


def test_dossier_10q_sections_handle_dash_item_title_separators():
    html = (
        "<html><body>"
        '<div class="toc">Item 1 - Financial Statements Item 2 - MD&amp;A Item 1A - Risk Factors</div>'
        "<h1>Item 1 — Financial Statements</h1>"
        + ("<p>Quarterly financial statement body with balance sheet details. </p>" * 20)
        + "<h1>Item 2 - Management’s Discussion and Analysis of Financial Condition and Results of Operations</h1>"
        + ("<p>Quarterly MD&A body with operating trends. </p>" * 20)
        + "<h1>Item 1A: Risk Factors</h1>"
        + ("<p>Quarterly risk-factor body with updated risks. </p>" * 20)
        + "</body></html>"
    )
    sections = segment_10q_sections(html)
    by_label = {s.section_label: s for s in sections}

    assert "Quarterly financial statement body" in by_label["financial_statements"].text
    assert "Quarterly MD&A body" in by_label["md_and_a"].text
    assert "Quarterly risk-factor body" in by_label["risk_factors"].text


def test_dossier_8k_sections_pick_body_item_102_over_table_of_contents():
    html = (
        "<html><body>"
        '<div class="toc">Item 1.02 Termination of a Material Definitive Agreement '
        "Item 7.01 Regulation FD Disclosure Item 9.01 Financial Statements and Exhibits</div>"
        "<h1>Item 1.02 Termination of a Material Definitive Agreement.</h1>"
        + (
            "<p>The company terminated a material supply agreement after counterparty performance deteriorated. "
            "Management said the termination could disrupt deliveries and create transition costs.</p>"
            * 10
        )
        + "<h1>Item 7.01 Regulation FD Disclosure.</h1>"
        + (
            "<p>The investor presentation includes supplemental overview material for reference.</p>"
            * 10
        )
        + "<h1>Item 9.01 Financial Statements and Exhibits.</h1>"
        "<p>Exhibit 99.1 Press release.</p>"
        "</body></html>"
    )

    sections = segment_8k_sections(html)
    by_label = {s.section_label: s for s in sections}

    assert item_code_for_8k_section("item_102") == "1.02"
    assert event_category_for_8k_section("item_102") == "agreement_termination"
    assert "terminated a material supply agreement" in by_label["item_102"].text
    assert "Table of Contents" not in by_label["item_102"].text
    assert "investor presentation includes supplemental" not in by_label["item_102"].text
