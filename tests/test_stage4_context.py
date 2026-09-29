from __future__ import annotations

from app.discover.stage4_context import _normalize_line_items


def test_normalize_line_items_uses_dei_cover_page_shares():
    normalized, translations = _normalize_line_items(["EntityCommonStockSharesOutstanding"])

    assert normalized == ["shares_outstanding"]
    assert translations == {"EntityCommonStockSharesOutstanding": "shares_outstanding"}


def test_normalize_line_items_does_not_treat_weighted_average_shares_as_outstanding():
    normalized, translations = _normalize_line_items(["WeightedAverageNumberOfSharesOutstandingBasic"])

    assert normalized == ["WeightedAverageNumberOfSharesOutstandingBasic"]
    assert translations == {}
