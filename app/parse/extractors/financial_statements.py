"""
financial_statements.py — RETIRED

Numerical financial data is now sourced from the SEC XBRL companyfacts API.
See: app/ingest/companyfacts.py and app/ingest/facts_writer.py

This module is intentionally non-functional. If you are seeing this error,
a caller is still importing extract_financial_rows — remove that call.
"""


def extract_financial_rows(*args, **kwargs):
    raise NotImplementedError(
        "extract_financial_rows is retired. "
        "Use app.ingest.facts_writer.ensure_facts() + companyfacts_facts table instead."
    )
