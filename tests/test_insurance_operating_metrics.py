from __future__ import annotations

import json

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


def _seed_filing(ticker: str, *, text: str, tmp_path, cik: str = "0000000123") -> None:
    now = utc_now_iso()
    path = tmp_path / f"{ticker.lower()}_10k.txt"
    path.write_text(text, encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            )
            VALUES(?, ?, ?, '10-K', '2026-02-01', '2025-12-31', 'https://example.test/10k', ?, 'OK', ?, ?)
            """,
            (cik, ticker, f"{ticker}-2026", str(path), now, now),
        )


def _seed_packet_inputs(ticker: str, *, text: str, tmp_path) -> None:
    now = utc_now_iso()
    cik = "0000000123"
    submission_path = tmp_path / "data" / "cache" / "submissions" / f"{cik.zfill(10)}.json"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(
        json.dumps(
            {
                "name": "Example Property and Casualty Insurance Corp",
                "tickers": [ticker],
                "exchanges": ["NYSE"],
            }
        ),
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            "INSERT INTO companies(ticker, cik, name, created_at) VALUES(?, ?, 'Example Property and Casualty Insurance Corp', ?)",
            (ticker, cik, now),
        )
        conn.execute(
            """
            INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at)
            VALUES(?, '2026-02-01', 'insurance', 1.0, '[]', ?)
            """,
            (ticker, now),
        )
        conn.execute(
            """
            INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
            VALUES(?, '2026-02-01', 'scorecard', '{}', ?, '[]', ?)
            """,
            (
                ticker,
                json.dumps({"pricing_zone_detail": {"current_price": 20.0}}),
                now,
            ),
        )
        for fiscal_year, line_item, value in [
            (2025, "equity", 1000.0),
            (2025, "shares_outstanding", 100.0),
            (2025, "net_income", 120.0),
            (2024, "net_income", 100.0),
            (2023, "net_income", 80.0),
        ]:
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
                    'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000123.json',
                    ?, '2026-02-01',
                    '10-K', ?
                )
                """,
                (
                    ticker,
                    fiscal_year,
                    f"{fiscal_year}-12-31",
                    line_item,
                    value,
                    units,
                    now,
                    f"{ticker}-2026",
                ),
            )
    _seed_filing(ticker, text=text, tmp_path=tmp_path, cik=cik)


def _pc_text() -> str:
    return """
    Item 1. Business
    We write property and casualty insurance with catastrophe exposure from hurricanes and floods.
    Our reinsurance program includes quota share reinsurance, excess-of-loss protection, and facultative cover.
    Item 7. Management's Discussion and Analysis
    For the year ended December 31, 2025, the combined ratio was 92.4%.
    The loss ratio was 61.3% and the expense ratio was 31.1%.
    Results benefited from favorable prior year reserve development.
    Item 7A. Quantitative and Qualitative Disclosures About Market Risk
    """


def _mortgage_text() -> str:
    return """
    Item 1. Business
    We are a private mortgage insurer subject to the PMIERs.
    As of December 31, 2025, our PMIERs available assets exceeded our risk-based required assets by 70%.
    As of December 31, 2025, we had issued master policies with 2,193 customers.
    As of December 31, 2025, we had $221.4 billion of primary IIF and $59.3 billion of primary RIF.
    For the year ended December 31, 2025, we generated NIW of $48.9 billion.
    We use QSR Transactions, XOL Transactions and ILN Transactions to manage risk.
    Item 7. Management's Discussion and Analysis
    New insurance written $48,900 $46,044 $40,473
    Insurance-in-force (1) $221,448 $210,183 $197,029
    Risk-in-force (1) $59,313 $56,113 $51,796
    Policies in force (count) (1) 684,058 659,567 629,690
    Loans in default (count) (1) 7,661 6,642 5,099
    Default rate (1) 1.12 %1.01 %0.81 %
    Risk-in-force on defaulted loans (1) $656 $545 $408
    Annual persistency (4) 83.4 %84.6 %86.1 %
    Quarterly run-off (5) 5.1 %4.5 %3.4 %
    We paid 445 claims totaling $25.9 million during the year.
    Our provision for claims and claim expenses benefited from favorable prior year development.
    """


def test_money_normalization_requires_explicit_units_without_magnitude_guess():
    from app.insurance.operating_metrics import _normalize_money_to_billions

    assert _normalize_money_to_billions("2500", None) is None
    assert _normalize_money_to_billions("900000", "USD") == 0.0009
    assert _normalize_money_to_billions("900", "million") == 0.9
    assert _normalize_money_to_billions("2", "unsupported") is None


def test_pc_operating_metrics_extracts_underwriting_ratios_and_risk_terms(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing("PCAA", text=_pc_text(), tmp_path=tmp_path)

    from app.insurance.operating_metrics import build_pc_operating_metrics

    result = build_pc_operating_metrics(
        "PCAA",
        as_of_date="2026-02-01",
        routing={"issuer_type": "insurance_underwriter", "insurance_subtype": "pc_insurer"},
    )

    assert result["status"] == "OK"
    assert result["confidence"] == "HIGH"
    assert result["combined_ratio"] == 0.924
    assert result["loss_ratio"] == 0.613
    assert result["expense_ratio"] == 0.311
    assert result["combined_ratio_assessment"] == "STRONG_UNDERWRITING_PROFIT"
    assert result["reserve_development"]["status"] == "FAVORABLE"
    assert result["reinsurance_program"]["structures"] == [
        "quota_share",
        "excess_of_loss",
        "facultative",
    ]
    assert result["catastrophe_exposure"]["terms"] == ["catastrophe", "hurricane", "flood"]
    assert result["missing_components"] == []


def test_mortgage_insurance_operating_metrics_extract_pmier_and_credit_metrics(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_filing("MORT", text=_mortgage_text(), tmp_path=tmp_path)

    from app.insurance.operating_metrics import build_insurance_operating_metrics

    result = build_insurance_operating_metrics(
        "MORT",
        as_of_date="2026-02-01",
        routing={
            "issuer_type": "insurance_underwriter",
            "insurance_subtype": "title_mortgage_specialty",
        },
    )

    assert result["status"] == "OK"
    assert result["confidence"] == "HIGH"
    assert result["metric_family"] == "mortgage_insurance"
    assert result["pmier_excess_ratio"] == 0.7
    assert result["pmier_available_to_required_ratio"] == 1.7
    assert result["primary_iif_billion"] == 221.4
    assert result["primary_rif_billion"] == 59.3
    assert result["new_insurance_written_billion"] == 48.9
    assert result["customer_count"] == 2193
    assert result["policies_in_force"] == 684058
    assert result["loans_in_default"] == 7661
    assert result["default_rate"] == 0.0112
    assert result["rif_on_defaulted_loans_billion"] == 0.656
    assert result["annual_persistency"] == 0.834
    assert result["quarterly_runoff"] == 0.051
    assert result["claims_paid_count"] == 445
    assert result["claims_paid_million"] == 25.9
    assert result["credit_capital_assessment"] == "STRONG_CAPITAL_AND_CREDIT_PROFILE"
    assert result["reserve_development"]["status"] == "FAVORABLE"
    assert result["reinsurance_program"]["structures"] == [
        "quota_share",
        "excess_of_loss",
        "insurance_linked_notes",
    ]
    assert result["missing_components"] == []


def test_pc_operating_metrics_reads_deeper_filing_and_derives_expense_ratio(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    filing_text = (
        "Opening inline XBRL and table metadata. " * 20_000
        + """
        Results of Operations
        Key Ratios
        Attritional loss ratio - current year 54.4%
        Catastrophe loss ratio - current year 8.4%
        Loss and loss adjustment expense ratio 59.7%
        Acquisition cost ratio 24.0%
        Other underwriting expense ratio 9.2%
        Combined ratio 92.9%
        Prior Year Reserve Development
        Favorable prior year reserve development indicates that current estimates are lower.
        Our reinsurance purchases include a variety of quota share and excess of loss treaties.
        """
    )
    _seed_filing("DEEP", text=filing_text, tmp_path=tmp_path)

    from app.insurance.operating_metrics import build_pc_operating_metrics

    result = build_pc_operating_metrics(
        "DEEP",
        as_of_date="2026-02-01",
        routing={"issuer_type": "insurance_underwriter", "insurance_subtype": "pc_insurer"},
    )

    assert result["status"] == "OK"
    assert result["combined_ratio"] == 0.929
    assert result["loss_ratio"] == 0.597
    assert result["expense_ratio"] == 0.332
    assert result["source_labels"] == {
        "combined_ratio": "combined ratio",
        "loss_ratio": "loss and loss adjustment expense ratio",
        "expense_ratio": "derived_combined_minus_loss",
    }
    assert result["reserve_development"]["status"] == "FAVORABLE"
    assert result["reinsurance_program"]["structures"] == ["quota_share", "excess_of_loss"]
    assert result["missing_components"] == []


def test_pc_operating_metrics_are_not_applied_to_life_subtype(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    from app.insurance.operating_metrics import build_pc_operating_metrics

    result = build_pc_operating_metrics(
        "LIFE",
        as_of_date="2026-02-01",
        routing={"issuer_type": "insurance_underwriter", "insurance_subtype": "life_annuity"},
    )

    assert result == {
        "status": "NOT_APPLICABLE",
        "ticker": "LIFE",
        "reason_codes": ["NOT_PC_INSURANCE_SUBTYPE"],
    }


def test_insurance_packet_includes_pc_operating_metrics(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_packet_inputs("PCAA", text=_pc_text(), tmp_path=tmp_path)

    from app.insurance.packet import build_insurance_packet

    packet = build_insurance_packet(
        "PCAA",
        as_of_date="2026-02-01",
        persist=False,
        current_price_override=20.0,
        issuer_cik="123",
        aliases=("PCAA",),
        db_path=cfg.db_path,
        cfg=cfg,
    )

    assert packet["model_status"] == "OK"
    assert packet["routing"]["insurance_subtype"] == "pc_insurer"
    assert packet["operating_metrics"]["combined_ratio"] == 0.924
    assert packet["operating_metrics"]["combined_ratio_assessment"] == "STRONG_UNDERWRITING_PROFIT"
    assert "PC_OPERATING_METRICS_LIMITED" not in packet["model_fit_warnings"]


def test_alpha_report_renders_pc_operating_metrics(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    insurance_packet = {
        "model_status": "OK",
        "generic_valuation_valid": False,
        "routing": {
            "security_type": "common",
            "security_identity_status": "VERIFIED_COMMON_PRIMARY_TICKER",
            "issuer_type": "insurance_underwriter",
            "insurance_subtype": "pc_insurer",
            "accounting_regime": "US_GAAP",
        },
        "valuation": {
            "method": "insurance_common",
            "valuation_anchor": 34.0,
            "current_price": 20.0,
            "adjusted_book_value_per_share": 22.0,
            "normalized_roe": 0.14,
            "justified_price_to_book": 1.55,
        },
        "operating_metrics": {
            "status": "OK",
            "confidence": "HIGH",
            "combined_ratio": 0.924,
            "loss_ratio": 0.613,
            "expense_ratio": 0.311,
            "combined_ratio_assessment": "STRONG_UNDERWRITING_PROFIT",
            "reserve_development": {"status": "FAVORABLE"},
            "reinsurance_program": {"structures": ["quota_share", "excess_of_loss"]},
            "catastrophe_exposure": {"terms": ["catastrophe", "hurricane"]},
            "missing_components": [],
        },
        "model_blockers": [],
        "model_fit_warnings": ["GENERIC_DCF_EPV_SUPPRESSED"],
    }

    from app.alpha.report_writer import generate_alpha_report
    from app.alpha.schemas import SectorAlphaReport

    report = SectorAlphaReport(
        sector="insurance",
        total_candidates=1,
        rounds=[],
        winner="PCAA",
        winner_thesis="P&C metrics should be visible.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Reserve adequacy.",
        falsification_trigger="Adverse development.",
        time_horizon="12 months",
        signal_packets={"PCAA": {"insurance_packet": insurance_packet}},
        selection_basis="llm_decision",
    )
    path = generate_alpha_report(report, tmp_path / "alpha_report.md")
    text = path.read_text(encoding="utf-8")

    assert "**P&C Operating Metrics**" in text
    assert "| Combined Ratio | 92.4% |" in text
    assert "| Loss Ratio | 61.3% |" in text
    assert "| Expense Ratio | 31.1% |" in text
    assert "| Reserve Development | FAVORABLE |" in text


def test_alpha_report_renders_mortgage_insurance_metrics(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    insurance_packet = {
        "model_status": "OK",
        "generic_valuation_valid": False,
        "routing": {
            "security_type": "common",
            "security_identity_status": "VERIFIED_COMMON_PRIMARY_TICKER",
            "issuer_type": "insurance_underwriter",
            "insurance_subtype": "title_mortgage_specialty",
            "accounting_regime": "US_GAAP",
        },
        "valuation": {
            "method": "insurance_common",
            "valuation_anchor": 47.54,
            "current_price": 39.22,
            "adjusted_book_value_per_share": 34.09,
            "normalized_roe": 0.132,
            "justified_price_to_book": 1.39,
        },
        "operating_metrics": {
            "status": "OK",
            "confidence": "HIGH",
            "metric_family": "mortgage_insurance",
            "pmier_excess_ratio": 0.7,
            "pmier_available_to_required_ratio": 1.7,
            "primary_iif_billion": 221.4,
            "primary_rif_billion": 59.3,
            "new_insurance_written_billion": 48.9,
            "default_rate": 0.0112,
            "loans_in_default": 7661,
            "policies_in_force": 684058,
            "annual_persistency": 0.834,
            "claims_paid_count": 445,
            "claims_paid_million": 25.9,
            "credit_capital_assessment": "STRONG_CAPITAL_AND_CREDIT_PROFILE",
            "reserve_development": {"status": "FAVORABLE"},
            "reinsurance_program": {"structures": ["quota_share", "excess_of_loss"]},
            "missing_components": [],
        },
        "model_blockers": [],
        "model_fit_warnings": ["GENERIC_DCF_EPV_SUPPRESSED"],
    }

    from app.alpha.report_writer import generate_alpha_report
    from app.alpha.schemas import SectorAlphaReport

    report = SectorAlphaReport(
        sector="insurance",
        total_candidates=1,
        rounds=[],
        winner="MORT",
        winner_thesis="Mortgage metrics should be visible.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Housing credit.",
        falsification_trigger="PMIERs deterioration.",
        time_horizon="12 months",
        signal_packets={"MORT": {"insurance_packet": insurance_packet}},
        selection_basis="llm_decision",
    )
    path = generate_alpha_report(report, tmp_path / "alpha_report.md")
    text = path.read_text(encoding="utf-8")

    assert "**Mortgage Insurance Metrics**" in text
    assert "| PMIERs Excess Ratio | 70.0% |" in text
    assert "| PMIERs Available / Required | 1.70x |" in text
    assert "| Primary IIF | $221.4B |" in text
    assert "| Default Rate | 1.1% |" in text
    assert "| Credit / Capital Assessment | STRONG_CAPITAL_AND_CREDIT_PROFILE |" in text
