"""
test_discovery_parser_table.py — RETIRED

extract_financial_rows is retired. Numerical financial data is now sourced
from the SEC XBRL companyfacts API (companyfacts_facts table).
See: app/ingest/companyfacts.py and app/ingest/facts_writer.py
"""

import pytest


@pytest.mark.skip(reason="extract_financial_rows retired; data now sourced from companyfacts_facts table")
def test_table_financial_extraction_for_discovery_metrics():
    pass
