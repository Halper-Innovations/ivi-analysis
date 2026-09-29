from __future__ import annotations


def test_insurance_peer_context_uses_exact_subtype(monkeypatch):
    from app.insurance import peer_context

    subtype_by_ticker = {
        "TARGET": "pc_insurer",
        "PCA": "pc_insurer",
        "PCB": "pc_insurer",
        "PCC": "pc_insurer",
        "LIFE": "life_annuity",
    }
    metrics_by_ticker = {
        "TARGET": {"roic": 0.20, "operating_margin": 0.30, "revenue_growth_5y": 0.12},
        "PCA": {"roic": 0.08, "operating_margin": 0.15, "revenue_growth_5y": 0.04},
        "PCB": {"roic": 0.10, "operating_margin": 0.20, "revenue_growth_5y": 0.05},
        "PCC": {"roic": 0.12, "operating_margin": 0.25, "revenue_growth_5y": 0.06},
        "LIFE": {"roic": 0.50, "operating_margin": 0.80, "revenue_growth_5y": 0.50},
    }

    monkeypatch.setattr(peer_context, "_load_sector_tickers", lambda sector: ["PCA", "PCB", "PCC", "LIFE"])
    monkeypatch.setattr(peer_context, "_insurance_subtype_for", lambda ticker, as_of_date: subtype_by_ticker.get(ticker))
    monkeypatch.setattr(peer_context, "_metrics_dict", lambda ticker, as_of_date: metrics_by_ticker[ticker])

    result = peer_context.compute_insurance_subtype_peer_relative_metrics(
        "TARGET",
        "2026-04-25",
        fallback_to_sector=False,
    )

    assert result["status"] == "OK"
    assert result["peer_scope"] == "insurance_subtype"
    assert result["peer_group"] == "insurance:pc_insurer"
    assert result["insurance_subtype"] == "pc_insurer"
    assert result["peer_count"] == 3
    assert result["peer_tickers"] == ["PCA", "PCB", "PCC"]
    assert result["ticker_metrics"] == {
        "roic": 0.20,
        "operating_margin": 0.30,
        "revenue_growth_5y": 0.12,
    }
    assert result["sector_medians"] == {
        "roic": 0.10,
        "operating_margin": 0.20,
        "revenue_growth_5y": 0.05,
    }
    assert result["relative_ratios"] == {
        "roic_vs_median": 2.0,
        "operating_margin_vs_median": 1.5,
        "revenue_growth_vs_median": 2.4,
    }
    assert result["relative_position"] == "LEADER"


def test_insurance_peer_context_falls_back_when_subtype_cohort_is_thin(monkeypatch):
    from app.insurance import peer_context

    monkeypatch.setattr(peer_context, "_load_sector_tickers", lambda sector: ["PCA"])
    monkeypatch.setattr(peer_context, "_insurance_subtype_for", lambda ticker, as_of_date: "pc_insurer")
    monkeypatch.setattr(
        peer_context,
        "_metrics_dict",
        lambda ticker, as_of_date: {"roic": 0.10, "operating_margin": 0.20, "revenue_growth_5y": 0.05},
    )
    monkeypatch.setattr(
        peer_context,
        "compute_peer_relative_metrics",
        lambda ticker, as_of_date: {
            "status": "OK",
            "sector": "insurance",
            "peer_count": 99,
            "relative_position": "AVERAGE",
            "relative_ratios": {"roic_vs_median": 1.0},
        },
    )

    result = peer_context.compute_insurance_subtype_peer_relative_metrics("TARGET", "2026-04-25")

    assert result["status"] == "OK"
    assert result["sector"] == "insurance"
    assert result["peer_scope"] == "broad_sector_fallback"
    assert result["peer_group"] == "insurance"
    assert result["insurance_subtype"] == "pc_insurer"
    assert result["fallback_reason"] == "insufficient_subtype_peers"
    assert result["peer_count"] == 99
    assert result["relative_ratios"] == {"roic_vs_median": 1.0}
