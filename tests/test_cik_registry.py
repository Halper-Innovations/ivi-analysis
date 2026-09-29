# tests/test_cik_registry.py
from __future__ import annotations
import pytest
from unittest.mock import patch


def _mock_mapping():
    return {"AAPL": "320193", "MSFT": "789019", "NVDA": "1045810"}


def test_resolve_known_ticker():
    with patch("app.ingest.cik_registry.load_ticker_cik_map", return_value=_mock_mapping()):
        from app.ingest.cik_registry import resolve
        cik = resolve("AAPL")
    assert cik == "0000320193"  # zero-padded to 10 digits


def test_resolve_case_insensitive():
    with patch("app.ingest.cik_registry.load_ticker_cik_map", return_value=_mock_mapping()):
        from app.ingest.cik_registry import resolve
        cik = resolve("aapl")
    assert cik == "0000320193"


def test_resolve_unknown_ticker_raises():
    with patch("app.ingest.cik_registry.load_ticker_cik_map", return_value=_mock_mapping()):
        from app.ingest.cik_registry import resolve, CIKNotFoundError
        with pytest.raises(CIKNotFoundError, match="ZZZZ"):
            resolve("ZZZZ")


def test_validate_batch_full_coverage():
    with patch("app.ingest.cik_registry.load_ticker_cik_map", return_value=_mock_mapping()):
        from app.ingest.cik_registry import validate_batch
        report = validate_batch(["AAPL", "MSFT"])
    assert report.coverage_pct == 100.0
    assert report.unresolved == []


def test_validate_batch_partial_coverage():
    with patch("app.ingest.cik_registry.load_ticker_cik_map", return_value=_mock_mapping()):
        from app.ingest.cik_registry import validate_batch
        report = validate_batch(["AAPL", "ZZZZ"])
    assert report.coverage_pct == 50.0
    assert "ZZZZ" in report.unresolved
