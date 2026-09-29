"""Exact-literal regressions for the valuation math."""

from app.valuation import returns_persistence

def test_returns_unknown_class_claim_has_unknown_status():
    """Zero measured returns and no supporting evidence cannot constitute a known class."""
    result = returns_persistence.compute_returns_persistence("FIXTURE", "2025-03-01")
    assert result["claims"]["returns_persistence_class"]["status"] == "UNKNOWN"
    assert result["claims"]["returns_persistence_class"]["value"] == "RETURNS_PERSISTENCE_UNKNOWN"
    assert result["primary_returns_caution"] == "RETURNS_DURABILITY_UNCLEAR"

