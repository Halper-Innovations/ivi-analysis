from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _legacy_packet_fixture_bypasses_new_integrity_gate(monkeypatch):
    monkeypatch.setattr("app.watchlist.store.artifact_decision_eligibility", lambda payload: "PASS")


from app.autonomous.structural_gate import (
    STRUCTURAL_CODES,
    STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE,
    STRUCTURAL_GOING_CONCERN,
    STRUCTURAL_NANO_FLOOR,
    STRUCTURAL_PENNY_FLOOR,
    evaluate_going_concern_filing_text,
    evaluate_structural_gate,
)


def _seed_db(
    tmp_path: Path,
    *,
    companyfacts: list[tuple[str, int, str, str, str, float]] = (),
    filings: list[tuple[str, str, str, str | None, str | None]] = (),
) -> Path:
    """Minimal engine DB: companyfacts_facts + filings (db.py shapes)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS companyfacts_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL DEFAULT 'FY',
            period_end TEXT NOT NULL,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            source_url TEXT,
            fetched_at TEXT NOT NULL,
            filed_date TEXT,
            form TEXT,
            accession TEXT,
            UNIQUE(ticker, fiscal_year, period_type, line_item)
        );
        CREATE TABLE IF NOT EXISTS filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cik TEXT NOT NULL,
            ticker TEXT,
            accession TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_date TEXT,
            period_end TEXT,
            primary_doc_url TEXT NOT NULL,
            local_path TEXT,
            hash TEXT,
            ingested_as_of TEXT,
            status TEXT NOT NULL DEFAULT 'new',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(cik, accession)
        );
        """
    )
    for ticker, fiscal_year, period_type, period_end, line_item, value in companyfacts:
        conn.execute(
            "INSERT INTO companyfacts_facts (ticker, fiscal_year, period_type, period_end, "
            "line_item, value, units, source_url, fetched_at, filed_date, form, accession)"
            " VALUES (?, ?, ?, ?, ?, ?, 'USD_millions', "
            "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json', "
            "'2026-06-11T00:00:00+00:00', '2026-02-15', '10-K', 'facts-acc')",
            (ticker, fiscal_year, period_type, period_end, line_item, value),
        )
    for idx, (ticker, form_type, filing_date, period_end, local_path) in enumerate(filings):
        conn.execute(
            "INSERT INTO filings (cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url, local_path, status, created_at, updated_at)"
            " VALUES ('0000000001', ?, ?, ?, ?, ?, 'test://doc', ?, 'OK', '2026-06-11T00:00:00+00:00', '2026-06-11T00:00:00+00:00')",
            (ticker, f"acc-{idx}", form_type, filing_date, period_end, local_path),
        )
    conn.commit()
    conn.close()
    return db_path


def _no_submissions(_ticker: str) -> dict | None:
    return None


def _submissions_with_8k(items: str, filed: str) -> dict:
    return {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-000001"],
                "form": ["8-K"],
                "filingDate": [filed],
                "items": [items],
            }
        }
    }


def _complete_submissions() -> dict:
    return {
        "filings": {
            "recent": {
                "accessionNumber": [],
                "form": [],
                "filingDate": [],
                "items": [],
            }
        }
    }


def _seed_complete_v2_sources(
    tmp_path: Path,
    *,
    ticker: str,
    filing_text: str,
    include_earnings: bool = True,
) -> Path:
    filing_path = tmp_path / f"{ticker.lower()}-10k.html"
    filing_path.write_text(
        f"<html><body>{filing_text}</body></html>",
        encoding="utf-8",
    )
    companyfacts = (
        [
            (ticker, 2025, "FY", "2025-12-31", "net_income", 10.0),
            (ticker, 2025, "FY", "2025-12-31", "cfo", 20.0),
        ]
        if include_earnings
        else []
    )
    return _seed_db(
        tmp_path,
        companyfacts=companyfacts,
        filings=[
            (ticker, "10-K", "2026-02-15", "2025-12-31", str(filing_path)),
        ],
    )


def _evaluate_complete_v2(
    tmp_path: Path,
    *,
    ticker: str,
    filing_text: str = "The company has sufficient liquidity for the next twelve months.",
    include_earnings: bool = True,
    applicable_rule_ids: set[str] | None = None,
):
    db_path = _seed_complete_v2_sources(
        tmp_path,
        ticker=ticker,
        filing_text=filing_text,
        include_earnings=include_earnings,
    )
    return evaluate_structural_gate(
        ticker,
        as_of_date="2026-06-11",
        price=25.0,
        market_cap_mm=50_000.0,
        db_path=db_path,
        submissions_loader=lambda _ticker: _complete_submissions(),
        ticker_registry_loader=lambda: [("0000000001", ticker)],
        pipeline_version="v2",
        sector_contract_id="test-sector-v2",
        applicable_rule_ids=applicable_rule_ids,
        price_evidence_ref=f"price:{ticker}:2026-06-11",
        price_evidence_url=f"test://price/{ticker}",
        cap_evidence_ref=f"market-cap:{ticker}:2026-06-11",
        cap_evidence_url=f"test://market-cap/{ticker}",
    )


_STRUCTURAL_GOING_CONCERN_FALSE_POSITIVES = (
    (
        "IQV",
        "There is ongoing concern from privacy advocates, regulators and others "
        "regarding data protection and privacy issues.",
        "39e6e55a80dea5241a5adacba1c5e4d2f6c73245d02a7924ad2443b0761e667e",
    ),
    (
        "ICE",
        "Starting as of September 30, 2023, Bakkt has disclosed that it is "
        "monitoring its ability to continue as a going concern, and such disclosures "
        "have continued in Bakkt's most recent SEC filings. The carrying value of "
        "our equity method investment in Bakkt was $9 million.",
        "51478c6d23cbdfc5df630e496fb31d3ea8768a6929ee60e4d9a5f94506823f10",
    ),
    (
        "KO",
        "Ongoing concern over climate change is expected to continue to result in "
        "additional legal or regulatory requirements.",
        "4b3ff79236da0abaa2e7f6feea47b148ee1bba22d94791fad0743361b31f43e8",
    ),
    (
        "APA",
        "The corresponding obligations of such parties may increase substantially, "
        "thereby causing a significant impact on the counterparties' solvency and "
        "ability to continue as a going concern.",
        "8c72efe9d57d9e5e19c12b37f1dbd41cdbe49db8ad8ac99cef4948214ba8657e",
    ),
    (
        "PLTR",
        "One or more of our partners may suffer a bankruptcy or other economic "
        "hardship that negatively affects its ability to continue as a going concern.",
        "b3411686ff5f50b62627666aae1190cf43fcce6caec1702cb6107b009d26ad05",
    ),
    (
        "HPE",
        "Notes to Consolidated Financial Statements. We consider market conditions, "
        "the ability to operate as a going concern, and other factors which indicate "
        "that the carrying amount of the investment might not be recoverable.",
        "a3e733998eaa0af77b0a7ed2955df458ac8af9f70d3ed4996ef121e790c98eb8",
    ),
    (
        "BSX",
        "We consider a significant adverse change in the environment of an investee "
        "or a significant doubt about an investee's ability to continue as a going "
        "concern. If we identify an impairment indicator, we estimate fair value.",
        "4ded87b18594f1dfa90afd9c92f3fb44fca09ba14cdea2e9710b6e261d81a8c4",
    ),
    (
        "AVGO",
        "We evaluate the earnings performance, credit rating, asset quality, business "
        "prospects of the investee, and financial indicators of the investee's ability "
        "to continue as a going concern.",
        "be075218bfad23e5ad80770be97cc1a430e2b27277e7bab7aab9b15d71ec7493",
    ),
)


def test_penny_floor_exact_reason(tmp_path):
    result = evaluate_structural_gate(
        "PNNY",
        as_of_date="2026-06-11",
        price=0.99,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
    )
    assert result.quarantined is True
    assert result.reasons == ["QUARANTINE_STRUCTURAL:PENNY_FLOOR"]
    assert result.details["PENNY_FLOOR"] == "price=0.99"

    clean = evaluate_structural_gate(
        "PNNY",
        as_of_date="2026-06-11",
        price=1.00,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
    )
    assert clean.quarantined is False
    assert clean.reasons == []


def test_nano_floor_exact_reason(tmp_path):
    result = evaluate_structural_gate(
        "NANO",
        as_of_date="2026-06-11",
        price=5.00,
        market_cap_mm=24.9,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
    )
    assert result.reasons == ["QUARANTINE_STRUCTURAL:NANO_FLOOR"]
    assert result.details["NANO_FLOOR"] == "market_cap_mm=24.9"

    clean = evaluate_structural_gate(
        "NANO",
        as_of_date="2026-06-11",
        price=5.00,
        market_cap_mm=25.0,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
    )
    assert clean.quarantined is False


def test_earnings_quality_divergence_latest_fy_point_in_time(tmp_path):
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            ("DIVG", 2025, "FY", "2025-12-31", "net_income", 12.3),
            ("DIVG", 2025, "FY", "2025-12-31", "cfo", -4.5),
        ],
        filings=[("DIVG", "10-K", "2026-03-01", "2025-12-31", None)],
    )
    result = evaluate_structural_gate(
        "DIVG",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=db_path,
        submissions_loader=_no_submissions,
    )
    assert result.reasons == ["QUARANTINE_STRUCTURAL:EARNINGS_QUALITY_DIVERGENCE"]
    assert result.details["EARNINGS_QUALITY_DIVERGENCE"] == "basis=FY2025:net_income=12.3:cfo=-4.5"

    # Point-in-time: before the 10-K filed date the FY facts are invisible.
    pre_filing = evaluate_structural_gate(
        "DIVG",
        as_of_date="2026-01-15",
        price=5.00,
        db_path=db_path,
        submissions_loader=_no_submissions,
    )
    assert pre_filing.quarantined is False


@pytest.mark.parametrize("filed_date", (None, "2026-07-01"))
def test_earnings_quality_missing_or_post_asof_filing_date_fails_closed(
    tmp_path,
    filed_date,
):
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            ("PITX", 2025, "FY", "2025-12-31", "net_income", 12.3),
            ("PITX", 2025, "FY", "2025-12-31", "cfo", -4.5),
        ],
    )
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE companyfacts_facts SET filed_date = ? WHERE ticker = 'PITX'",
        (filed_date,),
    )
    conn.commit()
    conn.close()

    result = evaluate_structural_gate(
        "PITX",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=db_path,
        submissions_loader=_no_submissions,
    )

    assert result.quarantined is False
    assert STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE not in result.triggered_codes
    assert result.details.get(STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE) is None


def test_earnings_quality_no_trigger_when_cash_flow_positive(tmp_path):
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            ("CLEN", 2025, "FY", "2025-12-31", "net_income", 12.3),
            ("CLEN", 2025, "FY", "2025-12-31", "cfo", 8.0),
        ],
    )
    result = evaluate_structural_gate(
        "CLEN",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=db_path,
        submissions_loader=_no_submissions,
    )
    assert result.quarantined is False


def test_earnings_quality_divergence_ttm_basis(tmp_path):
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            # Latest FY is clean; the divergence only shows in the TTM window.
            ("TTMD", 2025, "FY", "2025-06-30", "net_income", 3.0),
            ("TTMD", 2025, "FY", "2025-06-30", "cfo", 1.0),
            ("TTMD", 2025, "Q1", "2025-09-30", "net_income", 1.0),
            ("TTMD", 2025, "Q1", "2025-09-30", "cfo", -0.5),
            ("TTMD", 2025, "Q2", "2025-12-31", "net_income", 1.0),
            ("TTMD", 2025, "Q2", "2025-12-31", "cfo", -0.5),
            ("TTMD", 2025, "Q3", "2026-03-31", "net_income", 1.0),
            ("TTMD", 2025, "Q3", "2026-03-31", "cfo", -0.5),
            ("TTMD", 2024, "Q4", "2025-06-30", "net_income", 1.0),
            ("TTMD", 2024, "Q4", "2025-06-30", "cfo", -0.5),
        ],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE companyfacts_facts
            SET filed_date = '2026-05-15'
            WHERE ticker = 'TTMD' AND period_type = 'Q3'
            """
        )
    result = evaluate_structural_gate(
        "TTMD",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=db_path,
        submissions_loader=_no_submissions,
    )
    assert result.reasons == ["QUARANTINE_STRUCTURAL:EARNINGS_QUALITY_DIVERGENCE"]
    assert result.details["EARNINGS_QUALITY_DIVERGENCE"].startswith("basis=TTM:")
    assert "net_income=4.0:cfo=-2.0" in result.details["EARNINGS_QUALITY_DIVERGENCE"]


def test_going_concern_affirmative_language_triggers(tmp_path):
    doc = tmp_path / "tenk.html"
    doc.write_text(
        "<html><body><p>These conditions raise substantial doubt about the "
        "Company and our ability to continue as a <span>going concern</span>.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    db_path = _seed_db(
        tmp_path,
        filings=[("GONC", "10-K", "2026-02-15", "2025-12-31", str(doc))],
    )
    result = evaluate_structural_gate(
        "GONC",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=db_path,
        submissions_loader=_no_submissions,
    )
    assert result.reasons == ["QUARANTINE_STRUCTURAL:GOING_CONCERN"]
    assert result.details["GOING_CONCERN"].startswith(
        "10-K filed 2026-02-15: subject=REGISTRANT; "
        "mode=AFFIRMATIVE_CURRENT; section=ANNUAL_FILING_OTHER;"
    )


def test_going_concern_hypothetical_risk_factor_does_not_trigger(tmp_path):
    doc = tmp_path / "tenk.html"
    doc.write_text(
        "<html><body><p>If we are unable to obtain additional funding, this "
        "could raise substantial doubt about our ability to continue as a "
        "going concern.</p></body></html>",
        encoding="utf-8",
    )
    db_path = _seed_db(
        tmp_path,
        filings=[("HYPO", "10-K", "2026-02-15", "2025-12-31", str(doc))],
    )
    result = evaluate_structural_gate(
        "HYPO",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=db_path,
        submissions_loader=_no_submissions,
    )
    assert result.quarantined is False


def test_v2_clean_complete_screen_passes_with_full_rule_records(tmp_path):
    result = _evaluate_complete_v2(tmp_path, ticker="CLEAN")

    assert result.contract_id == "test-sector-v2"
    assert result.quarantined is False
    assert result.excluded_error is False
    assert result.screen_result is not None
    assert result.screen_result.status == "PASS"
    assert result.screen_result.reason_codes == []
    assert result.screen_result.required_rule_ids == list(STRUCTURAL_CODES)
    assert [item.rule_id for item in result.gate_evaluations] == list(STRUCTURAL_CODES)
    assert [item.status for item in result.gate_evaluations] == ["PASS"] * len(STRUCTURAL_CODES)
    assert result.screen_result.gate_evaluations == result.gate_evaluations
    for evaluation in result.gate_evaluations:
        assert evaluation.contract_id == "test-sector-v2"
        assert evaluation.applicable is True
        assert evaluation.observed_value is not None
        assert evaluation.threshold is not None
        assert evaluation.evidence_ref_id is not None or evaluation.evidence_url is not None


def test_v2_missing_sources_are_incomplete_not_business_rejections(tmp_path):
    result = evaluate_structural_gate(
        "MISS",
        as_of_date="2026-06-11",
        price=None,
        market_cap_mm=None,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
        ticker_registry_loader=lambda: None,
        pipeline_version="v2",
        sector_contract_id="test-sector-v2",
    )

    assert result.quarantined is False
    assert result.degraded_codes == ["ENGINE_DB_MISSING"]
    assert result.screen_result is not None
    assert result.screen_result.status == "INCOMPLETE"
    assert {item.rule_id: item.status for item in result.gate_evaluations} == {
        code: "INCOMPLETE" for code in STRUCTURAL_CODES
    }
    assert {item.rule_id: item.reason_code for item in result.gate_evaluations} == {
        "NON_PRIMARY_LISTING": "SEC_TICKER_REGISTRY_UNAVAILABLE",
        "DELISTING_NOTICE": "SEC_SUBMISSIONS_UNAVAILABLE",
        "PENNY_FLOOR": "SCREEN_PRICE_UNAVAILABLE",
        "NANO_FLOOR": "SCREEN_MARKET_CAP_UNAVAILABLE",
        "EARNINGS_QUALITY_DIVERGENCE": "ENGINE_DB_MISSING",
        "GOING_CONCERN": "ENGINE_DB_MISSING",
    }
    assert "QUARANTINE_STRUCTURAL" not in " ".join(result.screen_result.reason_codes)


def test_v2_sourced_failure_preserves_fail_when_other_sources_are_missing(tmp_path):
    result = evaluate_structural_gate(
        "FAIL",
        as_of_date="2026-06-11",
        price=0.50,
        market_cap_mm=None,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
        ticker_registry_loader=lambda: None,
        pipeline_version="v2",
        sector_contract_id="test-sector-v2",
        price_evidence_ref="price:FAIL:2026-06-11",
        price_evidence_url="test://price/FAIL",
    )

    assert result.quarantined is True
    assert result.triggered_codes == [STRUCTURAL_PENNY_FLOOR]
    assert result.screen_result is not None
    assert result.screen_result.status == "FAIL"
    evaluations = {item.rule_id: item for item in result.gate_evaluations}
    assert evaluations[STRUCTURAL_PENNY_FLOOR].status == "FAIL"
    assert evaluations[STRUCTURAL_PENNY_FLOOR].observed_value == 0.5
    assert evaluations[STRUCTURAL_PENNY_FLOOR].reason_code == ("QUARANTINE_STRUCTURAL:PENNY_FLOOR")
    assert evaluations[STRUCTURAL_NANO_FLOOR].status == "INCOMPLETE"
    assert evaluations[STRUCTURAL_GOING_CONCERN].status == "INCOMPLETE"
    assert "QUARANTINE_STRUCTURAL:PENNY_FLOOR" in result.screen_result.reason_codes
    assert "ENGINE_DB_MISSING" in result.screen_result.reason_codes


def test_v2_financial_contract_marks_earnings_quality_rule_not_applicable(tmp_path):
    applicable = set(STRUCTURAL_CODES) - {STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE}
    result = _evaluate_complete_v2(
        tmp_path,
        ticker="BANK",
        include_earnings=False,
        applicable_rule_ids=applicable,
    )

    assert result.screen_result is not None
    assert result.screen_result.status == "PASS"
    evaluations = {item.rule_id: item for item in result.gate_evaluations}
    earnings = evaluations[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE]
    assert earnings.status == "NOT_APPLICABLE"
    assert earnings.applicable is False
    assert earnings.reason_code == "SECTOR_RULE_NOT_APPLICABLE"
    assert all(
        evaluation.status == "PASS"
        for rule_id, evaluation in evaluations.items()
        if rule_id != STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE
    )


@pytest.mark.parametrize(
    ("ticker", "filing_text", "expected_revision"),
    _STRUCTURAL_GOING_CONCERN_FALSE_POSITIVES,
    ids=("IQV", "ICE", "KO", "APA", "PLTR", "HPE", "BSX", "AVGO"),
)
def test_v2_named_going_concern_false_positives_do_not_block(
    tmp_path,
    ticker,
    filing_text,
    expected_revision,
):
    result = _evaluate_complete_v2(
        tmp_path,
        ticker=ticker,
        filing_text=filing_text,
    )

    assert STRUCTURAL_GOING_CONCERN not in result.triggered_codes
    assert result.screen_result is not None
    assert result.screen_result.status == "PASS"
    going_concern = next(
        item for item in result.gate_evaluations if item.rule_id == STRUCTURAL_GOING_CONCERN
    )
    assert going_concern.status == "PASS"
    assert going_concern.observed_value["assertion"] == ("NO_BLOCKABLE_ATTRIBUTED_ASSERTION")
    assert going_concern.observed_value["accession"] == "acc-0"
    assert going_concern.observed_value["form_type"] == "10-K"
    assert going_concern.observed_value["filing_date"] == "2026-02-15"
    assert going_concern.observed_value["issuer_cik"] == "1"
    assert going_concern.observed_value["source_url"] == "test://doc"
    assert going_concern.observed_value["content_revision"] == expected_revision
    assert going_concern.evidence_ref_id == f"sec-filing:{ticker}:acc-0"
    assert going_concern.evidence_url == "test://doc"
    assert all(
        assertion["blockable"] is False for assertion in going_concern.observed_value["assertions"]
    )


@pytest.mark.parametrize(
    ("ticker", "filing_text", "expected_subject", "expected_revision"),
    (
        (
            "REGP",
            "These conditions raise substantial doubt about our ability to continue "
            "as a going concern.",
            "REGISTRANT",
            "4bd39566c08c23262daff27c1d01364a4485f604489482f78a59ae5c7024e1f4",
        ),
        (
            "SUBP",
            "These conditions raise substantial doubt about our wholly owned "
            "consolidated subsidiary's ability to continue as a going concern.",
            "CONSOLIDATED_SUBSIDIARY",
            "a6a1106fc03011133695ef4b8defb5d77837d000cd87b80dce6abbd8a1307ff3",
        ),
    ),
    ids=("registrant", "consolidated-subsidiary"),
)
def test_v2_attributed_true_positive_going_concern_assertions_block(
    tmp_path,
    ticker,
    filing_text,
    expected_subject,
    expected_revision,
):
    result = _evaluate_complete_v2(
        tmp_path,
        ticker=ticker,
        filing_text=filing_text,
    )

    assert result.triggered_codes == [STRUCTURAL_GOING_CONCERN]
    assert result.screen_result is not None
    assert result.screen_result.status == "FAIL"
    going_concern = next(
        item for item in result.gate_evaluations if item.rule_id == STRUCTURAL_GOING_CONCERN
    )
    assert going_concern.status == "FAIL"
    assert going_concern.reason_code == "QUARANTINE_STRUCTURAL:GOING_CONCERN"
    assert going_concern.observed_value["subject"] == expected_subject
    assert going_concern.observed_value["assertion_mode"] == "AFFIRMATIVE_CURRENT"
    assert going_concern.observed_value["blockable"] is True
    assert going_concern.observed_value["accession"] == "acc-0"
    assert going_concern.observed_value["form_type"] == "10-K"
    assert going_concern.observed_value["filing_date"] == "2026-02-15"
    assert going_concern.observed_value["issuer_cik"] == "1"
    assert going_concern.observed_value["source_url"] == "test://doc"
    assert going_concern.observed_value["content_revision"] == expected_revision
    assert going_concern.observed_value["excerpt"] == filing_text
    assert going_concern.evidence_ref_id == f"sec-filing:{ticker}:acc-0"
    assert going_concern.evidence_url == "test://doc"


def test_pure_going_concern_evaluator_uses_runtime_detector_and_gate_contract():
    filing_text = (
        "<html><body>These conditions raise substantial doubt about our ability "
        "to continue as a going concern.</body></html>"
    )

    result = evaluate_going_concern_filing_text(
        "regp",
        filing_text,
        accession="acc-pure",
        form_type="10-K",
        filing_date="2026-02-16",
        issuer_cik="0000000001",
        source_url="test://pure-doc",
    )

    assert result["status"] == "FAIL"
    assert result["reason_code"] == "QUARANTINE_STRUCTURAL:GOING_CONCERN"
    assert result["evidence_ref_id"] == "sec-filing:REGP:acc-pure"
    assert result["evidence_url"] == "test://pure-doc"
    assert result["observed_value"]["subject"] == "REGISTRANT"
    assert result["observed_value"]["content_revision"] == (
        "4bd39566c08c23262daff27c1d01364a4485f604489482f78a59ae5c7024e1f4"
    )


@pytest.mark.parametrize(
    "filing_text",
    (
        "The Company has incurred recurring losses from operations, which raises "
        "substantial doubt about its ability to continue as a going concern.",
        "Management's plans may not be successfully implemented, and substantial "
        "doubt exists about the Company's ability to continue as a going concern.",
        "If the Company does not raise additional capital, there is substantial "
        "doubt about its ability to continue as a going concern.",
        "These conditions raise substantial doubt about the ability of the Company "
        "to continue as a going concern.",
        "There is substantial doubt about the entity's ability to continue as a going concern.",
    ),
    ids=(
        "as2415-raises-its-ability",
        "asc205-40-plans-may-doubt-exists",
        "conditional-capital-there-is-doubt",
        "ability-of-the-company",
        "entity-ability",
    ),
)
def test_gc_audit_canonical_registrant_phrasings_block(filing_text):
    """2026-07-20 audit GC-1: canonical distress phrasings must FAIL the gate."""

    result = evaluate_going_concern_filing_text(
        "GCFN",
        filing_text,
        accession="acc-gc1",
        form_type="10-K",
        filing_date="2026-02-15",
    )

    assert result["status"] == "FAIL"
    assert result["reason_code"] == "QUARANTINE_STRUCTURAL:GOING_CONCERN"
    assert result["observed_value"]["subject"] == "REGISTRANT"
    assert result["observed_value"]["assertion_mode"] == "AFFIRMATIVE_CURRENT"
    assert result["observed_value"]["blockable"] is True


def test_gc_audit_plural_negation_passes():
    """2026-07-20 audit GC-3: plural "do not raise" negation must PASS the gate."""

    result = evaluate_going_concern_filing_text(
        "GCFP",
        "The conditions and events described above do not raise substantial doubt "
        "about our ability to continue as a going concern.",
        accession="acc-gc3",
        form_type="10-K",
        filing_date="2026-02-15",
    )

    assert result["status"] == "PASS"
    assert result["reason_code"] is None
    assertions = result["observed_value"]["assertions"]
    assert len(assertions) == 1
    assert assertions[0]["subject"] == "REGISTRANT"
    assert assertions[0]["assertion_mode"] == "NEGATED"
    assert assertions[0]["blockable"] is False


@pytest.mark.parametrize(
    ("requested_ticker", "primary_ticker"),
    (("ISSR", "ISSR.PRIMARY"), ("ISSR.B", "ISSR")),
    ids=("adr", "secondary-class"),
)
def test_v2_issuer_identity_resolves_primary_ticker_facts_and_filing(
    tmp_path,
    requested_ticker,
    primary_ticker,
):
    filing_path = tmp_path / "primary-10k.html"
    filing_path.write_text(
        "<html><body>These conditions raise substantial doubt about our ability "
        "to continue as a going concern.</body></html>",
        encoding="utf-8",
    )
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            (primary_ticker, 2025, "FY", "2025-12-31", "net_income", 10.0),
            (primary_ticker, 2025, "FY", "2025-12-31", "cfo", -5.0),
        ],
        filings=[
            (
                primary_ticker,
                "10-K",
                "2026-02-15",
                "2025-12-31",
                str(filing_path),
            )
        ],
    )

    result = evaluate_structural_gate(
        requested_ticker,
        as_of_date="2026-06-11",
        price=25.0,
        market_cap_mm=50_000.0,
        db_path=db_path,
        submissions_loader=lambda _ticker: _complete_submissions(),
        ticker_registry_loader=lambda: [("0000000001", requested_ticker)],
        pipeline_version="v2",
        issuer_cik="0000000001",
        aliases=(primary_ticker,),
        sector_contract_id="test-sector-v2",
        applicable_rule_ids={
            STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE,
            STRUCTURAL_GOING_CONCERN,
        },
    )

    assert result.triggered_codes == [
        STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE,
        STRUCTURAL_GOING_CONCERN,
    ]
    assert result.screen_result is not None
    assert result.screen_result.status == "FAIL"
    evaluations = {item.rule_id: item for item in result.gate_evaluations}
    assert evaluations[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE].status == "FAIL"
    assert evaluations[STRUCTURAL_GOING_CONCERN].status == "FAIL"
    going_concern = evaluations[STRUCTURAL_GOING_CONCERN].observed_value
    assert going_concern["subject"] == "REGISTRANT"
    assert going_concern["issuer_cik"] == "1"
    assert evaluations[STRUCTURAL_GOING_CONCERN].evidence_ref_id == (
        f"sec-filing:{requested_ticker}:acc-0"
    )


def test_v2_known_cik_never_uses_conflicting_alias_from_another_issuer(tmp_path):
    filing_path = tmp_path / "conflicting-10k.html"
    filing_path.write_text(
        "<html><body>These conditions raise substantial doubt about our ability "
        "to continue as a going concern.</body></html>",
        encoding="utf-8",
    )
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            ("CONFLICT", 2025, "FY", "2025-12-31", "net_income", 10.0),
            ("CONFLICT", 2025, "FY", "2025-12-31", "cfo", -5.0),
        ],
        filings=[("CONFLICT", "10-K", "2026-02-15", "2025-12-31", str(filing_path))],
    )
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE filings SET cik = '0000000002'")
        conn.execute(
            "UPDATE companyfacts_facts SET source_url = "
            "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000002.json'"
        )

    result = evaluate_structural_gate(
        "SAFE.B",
        as_of_date="2026-06-11",
        price=25.0,
        market_cap_mm=50_000.0,
        db_path=db_path,
        submissions_loader=lambda _ticker: _complete_submissions(),
        ticker_registry_loader=lambda: [("0000000001", "SAFE.B")],
        pipeline_version="v2",
        issuer_cik="0000000001",
        aliases=("CONFLICT",),
        sector_contract_id="test-sector-v2",
        applicable_rule_ids={
            STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE,
            STRUCTURAL_GOING_CONCERN,
        },
    )

    assert result.triggered_codes == []
    assert result.screen_result is not None
    assert result.screen_result.status == "INCOMPLETE"
    evaluations = {item.rule_id: item for item in result.gate_evaluations}
    assert evaluations[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE].reason_code == (
        "EARNINGS_QUALITY_FACTS_UNAVAILABLE"
    )
    assert evaluations[STRUCTURAL_GOING_CONCERN].reason_code == (
        "ANNUAL_OR_QUARTERLY_FILING_TEXT_UNAVAILABLE"
    )


def test_v2_issuer_alias_sources_remain_strictly_point_in_time(tmp_path):
    filing_path = tmp_path / "future-10k.html"
    filing_path.write_text(
        "<html><body>These conditions raise substantial doubt about our ability "
        "to continue as a going concern.</body></html>",
        encoding="utf-8",
    )
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            ("PRIMARY", 2025, "FY", "2025-12-31", "net_income", 10.0),
            ("PRIMARY", 2025, "FY", "2025-12-31", "cfo", -5.0),
        ],
        filings=[("PRIMARY", "10-K", "2026-02-15", "2025-12-31", str(filing_path))],
    )

    result = evaluate_structural_gate(
        "PRIMARY.B",
        as_of_date="2026-01-31",
        price=25.0,
        market_cap_mm=50_000.0,
        db_path=db_path,
        submissions_loader=lambda _ticker: _complete_submissions(),
        ticker_registry_loader=lambda: [("0000000001", "PRIMARY.B")],
        pipeline_version="v2",
        issuer_cik="0000000001",
        aliases=("PRIMARY",),
        sector_contract_id="test-sector-v2",
        applicable_rule_ids={
            STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE,
            STRUCTURAL_GOING_CONCERN,
        },
    )

    assert result.triggered_codes == []
    assert result.screen_result is not None
    assert result.screen_result.status == "INCOMPLETE"
    assert {
        item.rule_id: item.reason_code
        for item in result.gate_evaluations
        if item.status == "INCOMPLETE"
    } == {
        STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE: "EARNINGS_QUALITY_FACTS_UNAVAILABLE",
        STRUCTURAL_GOING_CONCERN: "ANNUAL_OR_QUARTERLY_FILING_TEXT_UNAVAILABLE",
    }


def test_v1_preserves_exact_ticker_reads_when_issuer_identity_is_supplied(tmp_path):
    filing_path = tmp_path / "primary-v1-10k.html"
    filing_path.write_text(
        "<html><body>These conditions raise substantial doubt about our ability "
        "to continue as a going concern.</body></html>",
        encoding="utf-8",
    )
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            ("PRIMARY", 2025, "FY", "2025-12-31", "net_income", 10.0),
            ("PRIMARY", 2025, "FY", "2025-12-31", "cfo", -5.0),
        ],
        filings=[("PRIMARY", "10-K", "2026-02-15", "2025-12-31", str(filing_path))],
    )

    result = evaluate_structural_gate(
        "PRIMARY.B",
        as_of_date="2026-06-11",
        price=25.0,
        market_cap_mm=50_000.0,
        db_path=db_path,
        submissions_loader=lambda _ticker: _complete_submissions(),
        ticker_registry_loader=lambda: [("0000000001", "PRIMARY.B")],
        pipeline_version="v1",
        issuer_cik="0000000001",
        aliases=("PRIMARY",),
    )

    assert result.quarantined is False
    assert STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE not in result.triggered_codes
    assert STRUCTURAL_GOING_CONCERN not in result.triggered_codes
    assert result.gate_evaluations == []
    assert result.screen_result is None


def test_delisting_notice_trailing_window_and_items(tmp_path):
    in_window = evaluate_structural_gate(
        "DLST",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=tmp_path / "missing.db",
        submissions_loader=lambda _t: _submissions_with_8k("3.01,8.01", "2025-08-15"),
    )
    assert in_window.reasons == ["QUARANTINE_STRUCTURAL:DELISTING_NOTICE"]
    assert in_window.details["DELISTING_NOTICE"] == (
        "8-K item 3.01 filed 2025-08-15 (1 in trailing 12m); cure_detection=not_available_v1"
    )

    out_of_window = evaluate_structural_gate(
        "DLST",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=tmp_path / "missing.db",
        submissions_loader=lambda _t: _submissions_with_8k("3.01", "2025-06-10"),
    )
    assert out_of_window.quarantined is False

    other_item = evaluate_structural_gate(
        "DLST",
        as_of_date="2026-06-11",
        price=5.00,
        db_path=tmp_path / "missing.db",
        submissions_loader=lambda _t: _submissions_with_8k("2.02,9.01", "2025-08-15"),
    )
    assert other_item.quarantined is False


def test_gtec_profile_fixture_fires_all_structural_triggers(tmp_path):
    """GTEC class: $0.66 price, $13.6M computable cap, delisting clock,
    positive reported NI against negative OCF."""
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            ("GTEC", 2025, "FY", "2025-12-31", "net_income", 2.0),
            ("GTEC", 2025, "FY", "2025-12-31", "cfo", -5.0),
        ],
        filings=[("GTEC", "10-K", "2026-03-20", "2025-12-31", None)],
    )
    result = evaluate_structural_gate(
        "GTEC",
        as_of_date="2026-06-11",
        price=0.66,
        market_cap_mm=13.6,
        db_path=db_path,
        submissions_loader=lambda _t: _submissions_with_8k("3.01", "2025-11-04"),
    )
    assert result.quarantined is True
    assert result.reasons == [
        "QUARANTINE_STRUCTURAL:DELISTING_NOTICE",
        "QUARANTINE_STRUCTURAL:PENNY_FLOOR",
        "QUARANTINE_STRUCTURAL:NANO_FLOOR",
        "QUARANTINE_STRUCTURAL:EARNINGS_QUALITY_DIVERGENCE",
    ]
    assert result.details["PENNY_FLOOR"] == "price=0.66"
    assert result.details["NANO_FLOOR"] == "market_cap_mm=13.6"
    assert result.reason_string == (
        "QUARANTINE_STRUCTURAL:DELISTING_NOTICE;"
        "QUARANTINE_STRUCTURAL:PENNY_FLOOR;"
        "QUARANTINE_STRUCTURAL:NANO_FLOOR;"
        "QUARANTINE_STRUCTURAL:EARNINGS_QUALITY_DIVERGENCE"
    )


def test_gate_never_quarantines_on_missing_data(tmp_path):
    result = evaluate_structural_gate(
        "BLNK",
        as_of_date="2026-06-11",
        price=None,
        market_cap_mm=None,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
    )
    assert result.quarantined is False
    assert result.reasons == []
    assert result.reason_string == ""


# AMG's SEC registry entry: common stock first, then exchange-traded notes.
_AMG_REGISTRY = [
    ("1004434", "AMG"),
    ("1004434", "MGR"),
    ("1004434", "MGRB"),
    ("1004434", "MGRD"),
    ("1004434", "MGRE"),
]


def test_non_primary_listing_quarantines_exchange_traded_note(tmp_path):
    result = evaluate_structural_gate(
        "MGRB",
        as_of_date="2026-07-03",
        price=16.30,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
        ticker_registry_loader=lambda: list(_AMG_REGISTRY),
    )
    assert result.quarantined is True
    assert result.reasons == ["QUARANTINE_STRUCTURAL:NON_PRIMARY_LISTING"]
    assert result.details["NON_PRIMARY_LISTING"] == "primary=AMG"


def test_non_primary_listing_keeps_primary_ticker(tmp_path):
    result = evaluate_structural_gate(
        "AMG",
        as_of_date="2026-07-03",
        price=150.0,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
        ticker_registry_loader=lambda: list(_AMG_REGISTRY),
    )
    assert result.quarantined is False
    assert result.reasons == []


def test_non_primary_listing_allows_share_class_variants(tmp_path):
    registry = [
        ("1652044", "GOOGL"),
        ("1652044", "GOOG"),
        ("1067983", "BRK-A"),
        ("1067983", "BRK-B"),
        ("920760", "LEN"),
        ("920760", "LEN.B"),
    ]
    for ticker in ("GOOG", "BRK-B", "LEN.B"):
        result = evaluate_structural_gate(
            ticker,
            as_of_date="2026-07-03",
            price=100.0,
            db_path=tmp_path / "missing.db",
            submissions_loader=_no_submissions,
            ticker_registry_loader=lambda: list(registry),
        )
        assert result.quarantined is False, ticker
        assert result.reasons == []


def test_non_primary_listing_no_registry_never_quarantines(tmp_path):
    result = evaluate_structural_gate(
        "MGRB",
        as_of_date="2026-07-03",
        price=16.30,
        db_path=tmp_path / "missing.db",
        submissions_loader=_no_submissions,
        ticker_registry_loader=lambda: None,
    )
    assert result.quarantined is False
    assert result.reasons == []


# ---------------------------------------------------------------------------
# Wiring: pre-LLM in sweep candidate loading + watchlist intake
# ---------------------------------------------------------------------------


def _init_temp_env(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    return db_path


def test_sweep_candidate_loading_excludes_quarantined_pre_llm(monkeypatch, tmp_path):
    db_path = _init_temp_env(monkeypatch, tmp_path)
    # Fail-closed: a MISSING engine DB now excludes every name as
    # EXCLUDED_ERROR, so this selection-routing test runs against a minimal
    # (empty but present) DB — missing schema stays benign.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(str(db_path)).close()
    from app.autonomous.cap_resolver import CapClassification

    classifications = {
        "GTEC": CapClassification(
            ticker="GTEC",
            as_of_date="2026-06-11",
            market_cap_mm=13.6,
            cap_source="stale_shares",
            cap_band="micro",
            price_used=0.66,
        ),
        "AAA": CapClassification(
            ticker="AAA",
            as_of_date="2026-06-11",
            market_cap_mm=100.0,
            cap_source="asof_companyfacts",
            cap_band="micro",
            price_used=5.0,
        ),
    }
    rows = [("GTEC", "2026-06-11", {}), ("AAA", "2026-06-11", {})]
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: (rows, classifications),
    )
    monkeypatch.setattr("app.sector.scan.pre_rank_sector", lambda **kwargs: [])

    from app.autonomous.sector_candidates import resolve_sector_candidate_tickers

    selection = resolve_sector_candidate_tickers(
        sector="energy",
        market_cap_focus="micro_cap",
        filing_risk_use_llm=False,
        as_of_date="2026-06-11",
    )
    assert selection.selected_tickers == ["AAA"]
    assert "GTEC" in selection.excluded_tickers
    assert "QUARANTINE_STRUCTURAL:PENNY_FLOOR:GTEC" in selection.warnings
    assert "QUARANTINE_STRUCTURAL:NANO_FLOOR:GTEC" in selection.warnings


def test_explicit_ticker_path_warns_but_keeps_quarantined_name(monkeypatch, tmp_path):
    db_path = _init_temp_env(monkeypatch, tmp_path)
    _seed_db(
        db_path.parent,
        companyfacts=[
            ("GTEC", 2025, "FY", "2025-12-31", "net_income", 2.0),
            ("GTEC", 2025, "FY", "2025-12-31", "cfo", -5.0),
        ],
    )

    from app.autonomous.sector_candidates import resolve_sector_candidate_tickers

    selection = resolve_sector_candidate_tickers(
        sector="energy",
        explicit_tickers=["GTEC"],
        market_cap_focus="micro_cap",
        as_of_date="2026-06-11",
        db_path=db_path,
    )
    assert selection.selected_tickers == ["GTEC"]
    assert "QUARANTINE_STRUCTURAL:EARNINGS_QUALITY_DIVERGENCE:GTEC" in selection.warnings


def test_intake_quarantines_structural_candidate_with_literal_reason(monkeypatch, tmp_path):
    db_path = _init_temp_env(monkeypatch, tmp_path)
    from app.autonomous.sector_contract import (
        AutonomousSectorFinancialRunArtifact,
        SectorCompanyFinancialPacket,
    )
    from app.watchlist.store import get_latest, populate_from_sector_artifact

    packet = SectorCompanyFinancialPacket(
        ticker="GTEC",
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        current_price=0.66,
        valuation={
            "anchor_method": "DCF",
            "valuation_anchor": 0.70,
            "buy_price_target": 0.50,
        },
    )
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_energy_structural_test",
        sector="energy",
        market_cap_focus="micro_cap",
        objective="Structural gate intake test.",
        as_of_date="2026-06-11",
        created_at="2026-06-11T12:00:00Z",
        completed_at="2026-06-11T12:01:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker=None,
        confidence="MODERATE",
        candidate_selection={"selected_tickers": ["GTEC"]},
        company_packets=[packet],
        relative_ranking=[
            {
                "ticker": "GTEC",
                "company_autonomy_verdict": "WATCHLIST_ONLY",
                "company_autonomy_confidence": "LOW",
            }
        ],
    )

    result = populate_from_sector_artifact(artifact, db_path=db_path)
    assert result.added_or_updated == 1

    row = get_latest("GTEC", db_path=db_path)
    assert row is not None
    assert row.status == "QUARANTINE"
    assert row.status_reason == "QUARANTINE_STRUCTURAL:PENNY_FLOOR"


def test_ttm_basis_rejects_non_consecutive_quarters(tmp_path):
    # Sparse quarterly CFO (YTD-only 10-Qs) leaves only Q1 rows across years;
    # stitching them into a "TTM" would be a seasonal artifact, not a signal.
    db_path = _seed_db(
        tmp_path,
        companyfacts=[
            ("SPRS", 2022, "Q1", "2022-03-31", "net_income", 100.0),
            ("SPRS", 2022, "Q1", "2022-03-31", "cfo", -20.0),
            ("SPRS", 2023, "Q1", "2023-03-31", "net_income", 100.0),
            ("SPRS", 2023, "Q1", "2023-03-31", "cfo", -20.0),
            ("SPRS", 2024, "Q1", "2024-03-31", "net_income", 100.0),
            ("SPRS", 2024, "Q1", "2024-03-31", "cfo", -20.0),
            ("SPRS", 2025, "Q1", "2025-03-31", "net_income", 100.0),
            ("SPRS", 2025, "Q1", "2025-03-31", "cfo", -20.0),
        ],
    )
    result = evaluate_structural_gate(
        "SPRS",
        as_of_date="2026-06-11",
        price=25.00,
        db_path=db_path,
        submissions_loader=_no_submissions,
    )
    assert result.quarantined is False
