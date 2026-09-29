from app.report.gaps import _missing_key_metrics, _parse_warnings, _recommended_actions


def test_missing_key_metrics_uses_bank_native_requirements_for_financial_issuers():
    packet = {
        "fundamentals": {
            "issuer_classification": "financial",
            "revenue": 100.0,
            "operating_margin": "UNKNOWN",
            "fcf": "UNKNOWN",
            "net_debt": "UNKNOWN",
            "deposits": 2500.0,
            "loans": 1400.0,
            "total_assets": 3900.0,
            "allowance_for_credit_losses": "UNKNOWN",
        },
        "financials": [
            {"line_item": "deposits", "citation": {"snippet": "Deposits 2500"}},
            {"line_item": "loans", "citation": {"snippet": "Loans 1400"}},
            {"line_item": "total_assets", "citation": {"snippet": "Assets 3900"}},
        ],
        "filings_used": [{"form_type": "10-K"}],
    }

    assert _missing_key_metrics(packet) == [
        "allowance_for_credit_losses",
        "provision_for_credit_losses",
        "net_charge_offs",
        "nonaccrual_loans",
    ]


def test_parse_warnings_skips_capex_requirement_for_financial_issuers():
    packet = {
        "fundamentals": {"issuer_classification": "financial"},
        "financials": [
            {"line_item": "revenue"},
            {"line_item": "cfo"},
            {"line_item": "cash"},
            {"line_item": "total_debt"},
            {"line_item": "deposits"},
            {"line_item": "loans"},
            {"line_item": "total_assets"},
        ],
        "filings_used": [{"form_type": "10-K"}],
    }

    warnings = _parse_warnings(packet)
    assert not warnings


def test_recommended_actions_use_bank_native_wording_for_financial_issuers():
    packet = {
        "fundamentals": {
            "issuer_classification": "financial",
            "revenue": 100.0,
            "deposits": 2500.0,
            "loans": None,
            "total_assets": 3900.0,
        },
        "financials": [],
        "filings_used": [{"form_type": "10-K"}],
    }

    actions = _recommended_actions(
        packet,
        research=None,
        missing_metrics=["loans", "allowance_for_credit_losses", "provision_for_credit_losses"],
        missing_price=False,
        missing_filings=False,
        claim_warnings=[],
    )

    assert any("bank credit and funding metrics" in action for action in actions)
