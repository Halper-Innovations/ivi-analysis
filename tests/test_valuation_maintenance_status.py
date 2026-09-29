"""Exact-literal regressions for known valuation edge cases."""

from app.valuation import maintenance_capex_discipline

def test_known_maintenance_unknown_claim_is_unknown():
    """No capex evidence gives no asset-intensity verdict; status must be UNKNOWN."""
    result = maintenance_capex_discipline.compute_maintenance_capex_discipline("FIXTURE", "2025-03-01")
    assert result["claims"]["asset_intensity_class"]["status"] == "UNKNOWN"
    assert result["claims"]["asset_intensity_class"]["value"] == "ASSET_INTENSITY_UNKNOWN"
    assert result["claims"]["maintenance_capex_credibility_class"]["value"] == "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
    assert result["claims"]["maintenance_capex_credibility_class"]["status"] == "UNKNOWN"

