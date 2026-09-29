from pathlib import Path

from app.parse.extractors.advanced_signals import (
    extract_cash_flow_quality_signals,
    extract_customer_concentration_signal,
    extract_debt_maturity_and_covenants,
    extract_non_gaap_reconciliation_signals,
    extract_sbc_dilution_signals,
    extract_segment_signals,
)
from app.parse.extractors.cover_page import extract_cover_page_facts
from app.parse.extractors.footnotes_signals import extract_footnote_signals


FIXTURE = Path(__file__).parent / "fixtures/sample_10q.html"


def test_fixture_cover_page_extraction_stability():
    text = FIXTURE.read_text(encoding="utf-8")
    facts = extract_cover_page_facts(text, "https://www.sec.gov/fake")
    fact_types = {f["fact_type"] for f in facts}
    assert "shares_outstanding" in fact_types
    shares = [f for f in facts if f["fact_type"] == "shares_outstanding"][0]
    assert shares["value_json"]["value"] == 12345678


def test_fixture_footnote_signal_extraction_stability():
    text = FIXTURE.read_text(encoding="utf-8")
    facts = extract_footnote_signals(text, "https://www.sec.gov/fake")
    extracted = {f["fact_type"] for f in facts}
    # The fixture says "substantial doubt about going concern does not exist":
    # a negation is not a going-concern finding (the keyword alone used to be).
    assert "going_concern" not in extracted
    assert "covenant_pressure" in extracted
    assert "refinancing_need" in extracted
    assert "dilution_risk" in extracted


def test_fixture_advanced_signal_extraction_stability():
    text = FIXTURE.read_text(encoding="utf-8")
    seg = extract_segment_signals(text, "https://www.sec.gov/fake")
    sbc = extract_sbc_dilution_signals(text, "https://www.sec.gov/fake")
    debt = extract_debt_maturity_and_covenants(text, "https://www.sec.gov/fake")
    conc = extract_customer_concentration_signal(text, "https://www.sec.gov/fake")
    cfq = extract_cash_flow_quality_signals(text, "https://www.sec.gov/fake")
    ngaap = extract_non_gaap_reconciliation_signals(text, "https://www.sec.gov/fake")

    assert any(item["fact_type"] == "segments_signal" for item in seg)
    assert any(item["fact_type"] == "sbc_dilution_signal" for item in sbc)
    assert any(item["fact_type"] == "debt_maturity_signal" for item in debt)
    assert any(item["fact_type"] == "covenant_signal" for item in debt)
    assert any(item["fact_type"] == "customer_concentration_signal" for item in conc)
    assert any(item["fact_type"] == "cash_flow_quality_signal" for item in cfq)
    assert any(item["fact_type"] == "non_gaap_reconciliation_signal" for item in ngaap)
