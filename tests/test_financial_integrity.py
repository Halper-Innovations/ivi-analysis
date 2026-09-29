from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.autonomous.financial_integrity import (
    INVALID_FINANCIAL_INPUT,
    MARKET_CAP_UNIT_USD_MILLIONS,
    NEEDS_DATA,
    PRICE_BASIS_UNADJUSTED,
    PRICE_UNIT_USD_PER_SHARE,
    SHARES_BASIS_ISSUER_REPORTED,
    SHARES_UNIT_MILLIONS,
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    authoritative_split_proof_reference,
    canonical_metric_trace,
    stable_quote_hash,
    validate_financial_integrity_scope,
)
from app.autonomous.sector_financial_packets import (
    _canonical_current_valuation_metrics,
    _derived_trace_provenance,
    _fcf_yield_metrics,
    build_sector_company_financial_packet,
)
from app.autonomous.sector_report import _fmt_market_cap, _market_cap
from app.autonomous.sector_contract import SectorCompanyFinancialPacket
from app.autonomous.sector_scenarios import build_expected_return_scenarios
from app.alpha.schemas import TickerSignalPacket
from app.autonomous.v1_financial_context import (
    bind_v1_financial_scope,
    financial_input_scenario,
)
from tests.financial_integrity_helpers import (
    materialized_split_proof as _materialized_split_proof,
)


def _trace_input_provenance(inputs: dict[str, float]) -> dict[str, dict[str, object]]:
    units = {
        "current_price": PRICE_UNIT_USD_PER_SHARE,
        "estimated_future_value_per_share": PRICE_UNIT_USD_PER_SHARE,
        "horizon_years": "years",
        "shares_outstanding_mm": SHARES_UNIT_MILLIONS,
        "issuer_quote_ratio": "ratio",
    }
    provenance: dict[str, dict[str, object]] = {}
    for name, value in inputs.items():
        if name in {"current_price", "estimated_future_value_per_share", "horizon_years"}:
            period_end = filed_date = "2026-07-21"
            source = "fixture_quote_or_scenario"
        elif name == "shares_outstanding_mm":
            period_end = "2026-03-31"
            filed_date = "2026-05-01"
            source = "fixture_companyfacts"
        elif name == "issuer_quote_ratio":
            period_end = filed_date = "2026-07-21"
            source = "fixture_security_identity"
        else:
            period_end = "2025-12-31"
            filed_date = "2026-02-15"
            source = "fixture_companyfacts_or_derived"
        provenance[name] = {
            "value": value,
            "unit": units.get(name, MARKET_CAP_UNIT_USD_MILLIONS),
            "source": source,
            "period_end": period_end,
            "filed_date": filed_date,
            "source_reference": f"fixture:{name}",
        }
        if name == "shares_outstanding_mm":
            provenance[name].update(
                {
                    "raw_source_value": float(value) * 1_000_000.0,
                    "raw_source_unit": "shares",
                    "normalized_value": value,
                    "normalized_unit": SHARES_UNIT_MILLIONS,
                    "split_adjustment_factor": 1.0,
                }
            )
    return provenance


def _trace(
    metric: str,
    output: float,
    snapshot_id: str,
    *,
    inputs: dict[str, float] | None = None,
) -> dict[str, object]:
    formulas = {
        "market_cap_mm": "current_price * shares_outstanding_mm / issuer_quote_ratio",
        "fcf_yield": "free_cash_flow_usd_millions / market_cap_usd_millions",
        "enterprise_value": ("market_cap_usd_millions + debt_usd_millions - cash_usd_millions"),
        "ev_to_ebitda": ("enterprise_value_usd_millions / ebitda_usd_millions"),
        "pe": "market_cap_usd_millions / net_income_usd_millions",
        "price_to_book": "market_cap_usd_millions / equity_usd_millions",
    }
    trace_inputs = inputs or {"literal": output}
    return canonical_metric_trace(
        metric=metric,
        formula=formulas[metric],
        inputs=trace_inputs,
        output=output,
        recomputed_output=output,
        output_unit=(
            MARKET_CAP_UNIT_USD_MILLIONS
            if metric in {"market_cap_mm", "enterprise_value"}
            else "ratio"
        ),
        quote_snapshot_id=snapshot_id,
        input_provenance=_trace_input_provenance(trace_inputs),
    )


def _valid_packet() -> dict[str, object]:
    quote = {
        "ticker": "MEGA",
        "price": 100.0,
        "as_of_date": "2026-07-21",
        "currency": "USD",
        "source": "fixture_quote",
        "source_url": "https://example.test/mega",
        "price_basis": PRICE_BASIS_UNADJUSTED,
        "raw_price": 100.0,
        "split_adjustment_factor": 1.0,
        "split_effective_date": None,
        "split_lineage_proof": _materialized_split_proof(
            {
                "ticker": "MEGA",
                "status": "PASS",
                "period_start": "2026-03-31",
                "period_end": "2026-07-21",
                "verified_as_of": "2026-07-21",
                "source": "fixture_corporate_actions",
                "source_reference": "https://eodhd.com/api/splits/MEGA",
            }
        ),
    }
    snapshot_id = stable_quote_hash(quote)
    return {
        "ticker": "MEGA",
        "market_cap_mm": 2_000.0,
        "market_cap_unit": MARKET_CAP_UNIT_USD_MILLIONS,
        "market_cap_source": "price_times_shares",
        "market_cap_effective_as_of_date": "2026-07-21",
        "market_cap_method": "price_times_shares_divided_by_issuer_quote_ratio",
        "current_price": 100.0,
        "current_price_unit": PRICE_UNIT_USD_PER_SHARE,
        "current_price_as_of_date": "2026-07-21",
        "current_price_currency": "USD",
        "current_price_source": "fixture_quote",
        "current_price_source_url": "https://example.test/mega",
        "quote_snapshot_id": snapshot_id,
        "price_basis": PRICE_BASIS_UNADJUSTED,
        "raw_price": 100.0,
        "shares_outstanding_mm": 20.0,
        "raw_shares_outstanding_mm": 20.0,
        "raw_shares_source_value": 20_000_000.0,
        "raw_shares_source_unit": "shares",
        "shares_unit": SHARES_UNIT_MILLIONS,
        "shares_basis": SHARES_BASIS_ISSUER_REPORTED,
        "shares_as_of_date": "2026-03-31",
        "shares_filed_date": "2026-05-01",
        "shares_source": "fixture_companyfacts",
        "shares_source_url": "https://example.test/mega/companyfacts",
        "issuer_quote_ratio": 1.0,
        "issuer_primary_ticker": "MEGA",
        "issuer_listed_tickers": ["MEGA"],
        "security_role": "PRIMARY",
        "is_secondary_class": False,
        "is_adr": False,
        "identity_source": "sec_submissions_exchange_binding",
        "identity_source_url": ("https://data.sec.gov/submissions/CIK0000000001.json"),
        "identity_as_of_date": "2026-07-21",
        "identity_confidence": "HIGH",
        "split_adjustment_factor": 1.0,
        "split_effective_date": None,
        "split_lineage_proof": dict(quote["split_lineage_proof"]),
        "cap_stage_price": 100.0,
        "cap_stage_price_as_of_date": "2026-07-21",
        "cap_stage_price_currency": "USD",
        "cap_stage_price_source": "fixture_quote",
        "cap_stage_price_source_url": "https://example.test/mega",
        "cap_stage_quote_snapshot_id": snapshot_id,
        "valuation": {
            "fcf_yield": 0.05,
            "enterprise_value": 1_900.0,
            "ev_to_ebitda": 9.5,
            "pe": 20.0,
            "price_to_book": 4.0,
        },
        "metric_traces": {
            "market_cap_mm": _trace(
                "market_cap_mm",
                2_000.0,
                snapshot_id,
                inputs={
                    "current_price": 100.0,
                    "shares_outstanding_mm": 20.0,
                    "issuer_quote_ratio": 1.0,
                },
            ),
            "fcf_yield": _trace(
                "fcf_yield",
                0.05,
                snapshot_id,
                inputs={
                    "free_cash_flow_usd_millions": 100.0,
                    "market_cap_usd_millions": 2_000.0,
                },
            ),
            "enterprise_value": _trace(
                "enterprise_value",
                1_900.0,
                snapshot_id,
                inputs={
                    "market_cap_usd_millions": 2_000.0,
                    "debt_usd_millions": 100.0,
                    "cash_usd_millions": 200.0,
                },
            ),
            "ev_to_ebitda": _trace(
                "ev_to_ebitda",
                9.5,
                snapshot_id,
                inputs={
                    "enterprise_value_usd_millions": 1_900.0,
                    "ebitda_usd_millions": 200.0,
                },
            ),
            "pe": _trace(
                "pe",
                20.0,
                snapshot_id,
                inputs={
                    "market_cap_usd_millions": 2_000.0,
                    "net_income_usd_millions": 100.0,
                },
            ),
            "price_to_book": _trace(
                "price_to_book",
                4.0,
                snapshot_id,
                inputs={
                    "market_cap_usd_millions": 2_000.0,
                    "equity_usd_millions": 500.0,
                },
            ),
        },
    }


def _bind_no_split_proof_to_issuer(
    packet: dict[str, object],
    issuer_cik: str,
) -> None:
    packet["split_lineage_proof"] = _materialized_split_proof(
        {
            "ticker": str(packet["ticker"]),
            "status": "PASS",
            "period_start": str(packet["shares_as_of_date"]),
            "period_end": str(packet["current_price_as_of_date"]),
            "verified_as_of": str(packet["current_price_as_of_date"]),
            "issuer_cik": issuer_cik,
            "source": "fixture_corporate_actions",
            "source_reference": f"https://eodhd.com/api/splits/{packet['ticker']}",
        }
    )


def test_valid_literal_unit_quote_and_metric_contract_passes() -> None:
    packet = _valid_packet()
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            context="literal_contract",
            run_as_of_date="2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == "PASS"
    assert result.passed is True
    assert result.ticker_snapshot_ids == {"MEGA": packet["quote_snapshot_id"]}


def test_raw_share_source_unit_must_reconcile_to_normalized_millions() -> None:
    packet = _valid_packet()
    packet["raw_shares_source_value"] = 20.0

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            context="raw_share_source_normalization_conflict",
            run_as_of_date="2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "RAW_SHARES_SOURCE_NORMALIZATION_CONFLICT" in {item.code for item in result.violations}


def test_issuer_share_cap_requires_authoritative_primary_quote_identity() -> None:
    packet = _valid_packet()
    packet.update(
        {
            "issuer_primary_ticker": "MEGA",
            "security_role": "PRIMARY",
            "is_secondary_class": False,
            "is_adr": False,
            "identity_source": "v1_legacy_unverified",
            "identity_source_url": None,
            "identity_as_of_date": None,
            "identity_confidence": None,
        }
    )

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            context="unverified_primary_identity",
            run_as_of_date="2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == NEEDS_DATA
    assert "CAP_SECURITY_IDENTITY_UNVERIFIED" in {item.code for item in result.violations}


@pytest.mark.parametrize(
    ("security_role", "is_adr", "is_secondary_class", "ratio_field"),
    [
        ("ADR", True, False, "adr_ratio"),
        ("SECONDARY_CLASS", False, True, "share_class_ratio"),
    ],
)
def test_issuer_share_cap_rejects_unbound_security_ratio_reference(
    security_role: str,
    is_adr: bool,
    is_secondary_class: bool,
    ratio_field: str,
) -> None:
    packet = _valid_packet()
    packet.update(
        {
            "issuer_cik": "0001234567",
            "issuer_primary_ticker": "OTHER",
            "security_role": security_role,
            "is_secondary_class": is_secondary_class,
            "is_adr": is_adr,
            ratio_field: 1.0,
            "identity_source": "issuer_filing",
            "identity_source_url": "https://data.sec.gov/not-a-filing.json",
            "ratio_source_url": "https://data.sec.gov/not-a-filing.json",
            "ratio_source_accession": None,
            "ratio_security_symbol": "MEGA",
        }
    )
    _bind_no_split_proof_to_issuer(packet, "0001234567")

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            context=f"unbound_{security_role.lower()}_ratio",
            run_as_of_date="2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == NEEDS_DATA
    assert "CAP_SECURITY_RATIO_UNVERIFIED" in {item.code for item in result.violations}


def test_issuer_share_cap_accepts_sec_ratio_bound_to_issuer_accession_and_symbol() -> None:
    packet = _valid_packet()
    filing_url = "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/issuer-20f.htm"
    packet.update(
        {
            "issuer_cik": "0001234567",
            "issuer_primary_ticker": "OTHER",
            "security_role": "ADR",
            "is_secondary_class": False,
            "is_adr": True,
            "adr_ratio": 1.0,
            "identity_source": "issuer_filing",
            "identity_source_url": filing_url,
            "ratio_source_url": filing_url,
            "ratio_source_accession": "0001234567-26-000001",
            "ratio_security_symbol": "MEGA",
        }
    )
    _bind_no_split_proof_to_issuer(packet, "0001234567")

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            context="bound_adr_ratio",
            run_as_of_date="2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == "PASS"


@pytest.mark.parametrize(
    ("case_name", "mutations", "recompute_hash", "expected_code"),
    [
        (
            "currency",
            {
                "current_price_currency": "EUR",
                "cap_stage_price_currency": "EUR",
            },
            True,
            "QUOTE_CURRENCY_INVALID",
        ),
        (
            "unit",
            {"current_price_unit": "USD"},
            True,
            "QUOTE_UNIT_INVALID",
        ),
        (
            "source",
            {
                "current_price_source": "",
                "cap_stage_price_source": "",
            },
            True,
            "QUOTE_SOURCE_MISSING",
        ),
        (
            "date",
            {
                "current_price_as_of_date": "2026-07-23",
                "cap_stage_price_as_of_date": "2026-07-23",
            },
            True,
            "QUOTE_ASOF_INVALID",
        ),
        (
            "raw_price",
            {"raw_price": 99.0},
            True,
            "UNADJUSTED_QUOTE_RAW_PRICE_MISMATCH",
        ),
        (
            "split_factor",
            {"split_adjustment_factor": 0.0},
            True,
            "SPLIT_FACTOR_INVALID",
        ),
        (
            "split_effective_date",
            {"split_effective_date": "2026-07-23"},
            True,
            "SPLIT_EFFECTIVE_DATE_INVALID",
        ),
        (
            "recomputed_hash",
            {"quote_snapshot_id": "f" * 64},
            False,
            "QUOTE_SNAPSHOT_ID_MISMATCH",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_quote_field_gate_matrix_rejects_literal_invalid_values(
    case_name: str,
    mutations: dict[str, object],
    recompute_hash: bool,
    expected_code: str,
) -> None:
    packet = _valid_packet()
    packet.update(mutations)
    if recompute_hash:
        snapshot_id = stable_quote_hash(
            ticker=packet["ticker"],
            price=packet["current_price"],
            as_of_date=packet["current_price_as_of_date"],
            currency=packet["current_price_currency"],
            source=packet["current_price_source"],
            source_url=packet["current_price_source_url"],
            price_basis=packet["price_basis"],
            raw_price=packet["raw_price"],
            split_adjustment_factor=packet["split_adjustment_factor"],
            split_effective_date=packet["split_effective_date"],
        )
        packet["quote_snapshot_id"] = snapshot_id
        packet["cap_stage_quote_snapshot_id"] = snapshot_id
        for trace in packet["metric_traces"].values():
            trace["quote_snapshot_id"] = snapshot_id

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            context=f"quote_field_matrix:{case_name}",
            run_as_of_date="2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status in {INVALID_FINANCIAL_INPUT, NEEDS_DATA}
    assert expected_code in {item.code for item in result.violations}


def test_bound_v1_scope_rejects_arbitrary_object_without_stringifying() -> None:
    packet = _valid_packet()
    scenario = financial_input_scenario(
        packet,
        financial_inputs={"supported": True},
    )
    scenario["financial_inputs"]["unsupported"] = object()

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        bind_v1_financial_scope(
            context="strict_json",
            run_as_of_date="2026-07-22",
            packets=(packet,),
            scenarios=(scenario,),
        )

    assert {item.code for item in exc_info.value.violations} == {
        "BOUND_FINANCIAL_INPUT_NOT_CANONICAL_JSON"
    }


def test_bound_v1_scope_normalizes_dataclass_and_rejects_mutation() -> None:
    @dataclass(frozen=True)
    class Inputs:
        label: str
        value: float

    packet = _valid_packet()
    scenario = financial_input_scenario(
        packet,
        financial_inputs={"inputs": Inputs(label="price", value=100.0)},
    )
    scope = bind_v1_financial_scope(
        context="dataclass_and_mutation",
        run_as_of_date="2026-07-22",
        packets=(packet,),
        scenarios=(scenario,),
    )
    assert scope.require().passed is True

    scope.scenarios[0]["financial_inputs"]["inputs"]["value"] = 101.0
    with pytest.raises(InvalidFinancialInputError) as exc_info:
        scope.require()
    assert {item.code for item in exc_info.value.violations} == {"BOUND_FINANCIAL_INPUT_MUTATED"}


def test_bound_v1_scope_rejects_nested_nonfinite_value() -> None:
    packet = _valid_packet()
    scenario = financial_input_scenario(
        packet,
        financial_inputs={"nested": [{"price": float("inf")}]},
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        bind_v1_financial_scope(
            context="nested_nonfinite",
            run_as_of_date="2026-07-22",
            packets=(packet,),
            scenarios=(scenario,),
        )

    assert {item.code for item in exc_info.value.violations} == {"BOUND_FINANCIAL_INPUT_NON_FINITE"}


def test_known_metric_trace_missing_input_provenance_is_needs_data() -> None:
    packet = _valid_packet()
    packet["metric_traces"]["enterprise_value"].pop("input_provenance")

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "missing_trace_input_provenance",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == NEEDS_DATA
    assert "METRIC_TRACE_INPUT_PROVENANCE_MISSING" in {item.code for item in result.violations}


def test_known_metric_trace_post_asof_filing_provenance_is_invalid() -> None:
    packet = _valid_packet()
    packet["metric_traces"]["pe"]["input_provenance"]["net_income_usd_millions"]["filed_date"] = (
        "2026-07-23"
    )

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "post_asof_trace_input_provenance",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "METRIC_TRACE_INPUT_PROVENANCE_ASOF_INVALID" in {item.code for item in result.violations}


def test_known_metric_trace_input_value_must_match_provenance() -> None:
    packet = _valid_packet()
    packet["metric_traces"]["market_cap_mm"]["input_provenance"]["current_price"]["value"] = 99.0

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "mismatched_trace_input_provenance",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "METRIC_TRACE_INPUT_PROVENANCE_VALUE_MISMATCH" in {
        item.code for item in result.violations
    }


def test_derived_trace_component_post_asof_filing_is_invalid() -> None:
    packet = _valid_packet()
    packet["metric_traces"]["enterprise_value"]["input_provenance"]["debt_usd_millions"][
        "components"
    ] = {
        "debt_usd_millions": {
            "value": 100.0,
            "unit": MARKET_CAP_UNIT_USD_MILLIONS,
            "source": "fixture_companyfacts",
            "period_end": "2025-12-31",
            "filed_date": "2026-07-23",
            "source_reference": "fixture:debt_component",
        }
    }

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "post_asof_derived_component",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    violation = next(
        item
        for item in result.violations
        if item.code == "METRIC_TRACE_INPUT_PROVENANCE_ASOF_INVALID"
        and ".components.debt_usd_millions" in item.field
    )
    assert violation.source_values["filed_date"] == "2026-07-23"


def test_derived_trace_component_period_after_filing_is_invalid() -> None:
    packet = _valid_packet()
    packet["metric_traces"]["enterprise_value"]["input_provenance"]["cash_usd_millions"][
        "components"
    ] = {
        "cash_usd_millions": {
            "value": 200.0,
            "unit": MARKET_CAP_UNIT_USD_MILLIONS,
            "source": "fixture_companyfacts",
            "period_end": "2026-03-31",
            "filed_date": "2026-02-15",
            "source_reference": "fixture:cash_component",
        }
    }

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "impossible_derived_component_chronology",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert any(
        item.code == "METRIC_TRACE_INPUT_PROVENANCE_CHRONOLOGY_INVALID"
        and item.source_values
        == {
            "period_end": "2026-03-31",
            "filed_date": "2026-02-15",
        }
        for item in result.violations
    )


def test_derived_trace_component_incomplete_provenance_is_needs_data() -> None:
    packet = _valid_packet()
    packet["metric_traces"]["fcf_yield"]["input_provenance"]["free_cash_flow_usd_millions"][
        "components"
    ] = {
        "cfo_usd_millions": {
            "value": 125.0,
            "unit": MARKET_CAP_UNIT_USD_MILLIONS,
            "source": "fixture_companyfacts",
            "period_end": "2025-12-31",
            "filed_date": "2026-02-15",
        }
    }

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "incomplete_derived_component",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == NEEDS_DATA
    violation = next(
        item
        for item in result.violations
        if item.code == "METRIC_TRACE_INPUT_PROVENANCE_MISSING"
        and ".components.cfo_usd_millions" in item.field
    )
    assert violation.source_values["missing_fields"] == ["source_reference"]


def test_derived_trace_provenance_does_not_launder_impossible_chronology() -> None:
    provenance = _derived_trace_provenance(
        100.0,
        unit=MARKET_CAP_UNIT_USD_MILLIONS,
        source="DERIVED_FREE_CASH_FLOW",
        components={
            "cfo_usd_millions": {
                "value": 125.0,
                "unit": MARKET_CAP_UNIT_USD_MILLIONS,
                "source": "fixture_companyfacts",
                "period_end": "2026-03-31",
                "filed_date": "2026-02-15",
                "source_reference": "fixture:cfo",
            }
        },
    )

    assert provenance == {}


def test_same_day_unadjusted_operands_still_require_external_split_proof() -> None:
    packet = _valid_packet()
    packet["shares_as_of_date"] = packet["current_price_as_of_date"]
    packet["shares_filed_date"] = packet["current_price_as_of_date"]
    packet["split_lineage_proof"] = None

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "same_day_unproven_split_basis",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == NEEDS_DATA
    assert "NO_INTERVENING_SPLIT_PROOF_MISSING" in {item.code for item in result.violations}


def test_no_split_proof_cannot_claim_coverage_after_its_verification_date() -> None:
    packet = _valid_packet()
    packet["split_lineage_proof"]["verified_as_of"] = "2026-07-20"

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "future_coverage_no_split_proof",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "NO_INTERVENING_SPLIT_PROOF_INVALID" in {item.code for item in result.violations}


def test_no_split_proof_cannot_self_authorize_with_fake_sec_url() -> None:
    packet = _valid_packet()
    packet["split_lineage_proof"] = {
        "ticker": "MEGA",
        "status": "PASS",
        "period_start": "2026-03-31",
        "period_end": "2026-07-21",
        "verified_as_of": "2026-07-21",
        "source": "invented_sec_proof",
        "source_reference": "https://www.sec.gov/fake-split",
    }

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "fake_sec_no_split_proof",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "NO_INTERVENING_SPLIT_PROOF_INVALID" in {item.code for item in result.violations}


def test_trusted_host_proof_in_arbitrary_temp_file_is_rejected(tmp_path: Path) -> None:
    record = {
        "ticker": "MEGA",
        "status": "PASS",
        "period_start": "2026-03-31",
        "period_end": "2026-07-21",
        "verified_as_of": "2026-07-21",
        "source": "invented_trusted_host_proof",
        "source_reference": "https://eodhd.com/api/splits/MEGA",
    }
    canonical_proof = _materialized_split_proof(record)
    envelope_bytes = Path(str(canonical_proof["materialized_path"])).read_bytes()
    arbitrary_path = (tmp_path / "caller-authored-proof.json").resolve()
    arbitrary_path.write_bytes(envelope_bytes)
    proof = {
        **canonical_proof,
        "materialized_path": str(arbitrary_path),
    }

    assert not authoritative_split_proof_reference(proof, expected_ticker="MEGA")

    packet = _valid_packet()
    packet["split_lineage_proof"] = proof
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "trusted_host_arbitrary_file",
            "2026-07-22",
            packets=(packet,),
        )
    )
    assert result.status == INVALID_FINANCIAL_INPUT
    assert "NO_INTERVENING_SPLIT_PROOF_INVALID" in {item.code for item in result.violations}


def test_no_split_proof_rejects_raw_provider_row_inside_covered_interval() -> None:
    proof = _materialized_split_proof(
        {
            "ticker": "MEGA",
            "status": "PASS",
            "period_start": "2026-03-31",
            "period_end": "2026-07-21",
            "verified_as_of": "2026-07-21",
            "source": "fixture_corporate_actions",
            "source_reference": "https://eodhd.com/api/splits/MEGA",
        },
        raw_payload_override=[{"date": "2026-05-15", "split": "4/1"}],
    )

    assert not authoritative_split_proof_reference(
        proof,
        expected_ticker="MEGA",
        expected_issuer_cik="0000000001",
    )

    packet = _valid_packet()
    packet["split_lineage_proof"] = proof
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "false_no_split_over_raw_event",
            "2026-07-22",
            packets=(packet,),
        )
    )
    assert result.status == INVALID_FINANCIAL_INPUT
    assert "NO_INTERVENING_SPLIT_PROOF_INVALID" in {item.code for item in result.violations}


def test_split_proof_rejects_expected_issuer_cik_mismatch() -> None:
    packet = _valid_packet()

    assert not authoritative_split_proof_reference(
        packet["split_lineage_proof"],
        expected_ticker="MEGA",
        expected_issuer_cik="0001234567",
    )


def test_missing_provenance_is_needs_data_with_exact_terminal_keys() -> None:
    packet = _valid_packet()
    packet["current_price_source"] = None
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("missing_source", "2026-07-22", packets=(packet,))
    )

    assert result.status == NEEDS_DATA
    violation = next(item for item in result.violations if item.code == "QUOTE_SOURCE_MISSING")
    payload = violation.to_dict()
    assert {
        "ticker",
        "field",
        "source_values",
        "expected_relationship",
        "observed_relationship",
        "reason",
    }.issubset(payload)
    assert payload["ticker"] == "MEGA"
    assert payload["field"] == "current_price_source"


def test_cap_stage_and_current_quote_mismatch_is_invalid() -> None:
    packet = _valid_packet()
    packet["current_price"] = 101.0
    packet["quote_snapshot_id"] = stable_quote_hash(
        ticker="MEGA",
        price=101.0,
        as_of_date="2026-07-21",
        currency="USD",
        source="fixture_quote",
        source_url="https://example.test/mega",
        price_basis=PRICE_BASIS_UNADJUSTED,
        raw_price=100.0,
        split_adjustment_factor=1.0,
    )
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("quote_mismatch", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "QUOTE_SNAPSHOT_MISMATCH" in {item.code for item in result.violations}


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_financial_value_is_invalid(bad_value: float) -> None:
    packet = _valid_packet()
    packet["market_cap_mm"] = bad_value
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("nonfinite", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "MARKET_CAP_INVALID" in {item.code for item in result.violations}


def test_split_adjusted_price_with_issuer_reported_shares_is_invalid() -> None:
    packet = _valid_packet()
    packet["price_basis"] = "SPLIT_ADJUSTED"
    packet["split_adjustment_factor"] = 10.0
    packet["split_effective_date"] = "2024-06-10"
    packet["quote_snapshot_id"] = stable_quote_hash(
        ticker="MEGA",
        price=100.0,
        as_of_date="2026-07-21",
        currency="USD",
        source="fixture_quote",
        source_url="https://example.test/mega",
        price_basis="SPLIT_ADJUSTED",
        raw_price=100.0,
        split_adjustment_factor=10.0,
        split_effective_date="2024-06-10",
    )
    packet["cap_stage_quote_snapshot_id"] = packet["quote_snapshot_id"]
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("split_basis", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "SPLIT_BASIS_MISMATCH" in {item.code for item in result.violations}


def test_unadjusted_price_with_split_adjusted_shares_is_invalid() -> None:
    packet = _valid_packet()
    packet["shares_basis"] = "SPLIT_ADJUSTED"
    packet["split_effective_date"] = "2024-06-10"
    packet["quote_snapshot_id"] = stable_quote_hash(
        ticker="MEGA",
        price=100.0,
        as_of_date="2026-07-21",
        currency="USD",
        source="fixture_quote",
        source_url="https://example.test/mega",
        price_basis=PRICE_BASIS_UNADJUSTED,
        raw_price=100.0,
        split_adjustment_factor=1.0,
        split_effective_date="2024-06-10",
    )
    packet["cap_stage_quote_snapshot_id"] = packet["quote_snapshot_id"]

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("reverse_split_basis", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "SPLIT_BASIS_MISMATCH" in {item.code for item in result.violations}


def _four_for_one_split_packet(*, split_effective_date: str) -> dict[str, object]:
    packet = _valid_packet()
    packet["current_price"] = 25.0
    packet["cap_stage_price"] = 25.0
    packet["price_basis"] = "SPLIT_ADJUSTED"
    packet["raw_price"] = 100.0
    packet["shares_outstanding_mm"] = 80.0
    packet["raw_shares_outstanding_mm"] = 20.0
    packet["shares_basis"] = "SPLIT_ADJUSTED"
    packet["split_adjustment_factor"] = 4.0
    packet["split_effective_date"] = split_effective_date
    packet["split_lineage_proof"] = _materialized_split_proof(
        {
            "ticker": "MEGA",
            "factor": 4.0,
            "effective_date": split_effective_date,
            "filed_date": "2025-06-01",
            "source": "fixture_split_event",
            "source_reference": (
                "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/split-event.htm"
            ),
        }
    )
    snapshot_id = stable_quote_hash(
        ticker="MEGA",
        price=25.0,
        as_of_date="2026-07-21",
        currency="USD",
        source="fixture_quote",
        source_url="https://example.test/mega",
        price_basis="SPLIT_ADJUSTED",
        raw_price=100.0,
        split_adjustment_factor=4.0,
        split_effective_date=split_effective_date,
    )
    packet["quote_snapshot_id"] = snapshot_id
    packet["cap_stage_quote_snapshot_id"] = snapshot_id
    packet["metric_traces"]["market_cap_mm"] = _trace(
        "market_cap_mm",
        2_000.0,
        snapshot_id,
        inputs={
            "current_price": 25.0,
            "shares_outstanding_mm": 80.0,
            "issuer_quote_ratio": 1.0,
        },
    )
    for name, trace in packet["metric_traces"].items():
        if name != "market_cap_mm":
            trace["quote_snapshot_id"] = snapshot_id
    return packet


def test_four_for_one_split_normalizes_both_price_and_shares_and_passes() -> None:
    packet = _four_for_one_split_packet(split_effective_date="2026-06-10")

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("four_for_one_split", "2026-07-22", packets=(packet,))
    )

    assert result.status == "PASS"
    assert result.passed is True


def test_split_event_material_retrieved_after_run_as_of_is_rejected() -> None:
    packet = _four_for_one_split_packet(split_effective_date="2026-06-10")
    packet["split_lineage_proof"] = _materialized_split_proof(
        {
            "ticker": "MEGA",
            "factor": 4.0,
            "effective_date": "2026-06-10",
            "filed_date": "2025-06-01",
            "retrieved_at": "2026-07-23",
            "source": "future_materialized_split_event",
            "source_reference": (
                "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/split-event.htm"
            ),
        }
    )

    assert not authoritative_split_proof_reference(
        packet["split_lineage_proof"],
        expected_ticker="MEGA",
        expected_issuer_cik="0000000001",
        expected_as_of_date="2026-07-22",
    )
    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "future_materialized_split_event",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == NEEDS_DATA
    assert "SPLIT_EVENT_PROOF_MISSING" in {item.code for item in result.violations}


def test_four_for_one_split_rejects_post_split_source_shares() -> None:
    packet = _four_for_one_split_packet(split_effective_date="2025-06-10")

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "four_for_one_split_double_adjustment",
            "2026-07-22",
            packets=(packet,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert result.passed is False
    assert "SPLIT_CHRONOLOGY_INVALID" in {violation.code for violation in result.violations}


@pytest.mark.parametrize("split_date", ["not-a-date", "2026-07-23"])
def test_split_effective_date_must_be_real_and_not_future(split_date: str) -> None:
    packet = _valid_packet()
    packet["price_basis"] = "SPLIT_ADJUSTED"
    packet["shares_basis"] = "SPLIT_ADJUSTED"
    packet["raw_price"] = 1_000.0
    packet["split_adjustment_factor"] = 10.0
    packet["split_effective_date"] = split_date
    packet["quote_snapshot_id"] = stable_quote_hash(
        ticker="MEGA",
        price=100.0,
        as_of_date="2026-07-21",
        currency="USD",
        source="fixture_quote",
        source_url="https://example.test/mega",
        price_basis="SPLIT_ADJUSTED",
        raw_price=1_000.0,
        split_adjustment_factor=10.0,
        split_effective_date=split_date,
    )
    packet["cap_stage_quote_snapshot_id"] = packet["quote_snapshot_id"]

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("split_date", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "SPLIT_EFFECTIVE_DATE_INVALID" in {item.code for item in result.violations}


def test_v1_mega_cap_fcf_yield_preserves_usd_millions(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_rows",
        lambda *_args, **_kwargs: {2025: {"cfo": 118_254.0, "capex": 21_578.0}},
    )

    result = _fcf_yield_metrics(
        "MEGA",
        as_of_date="2026-07-22",
        v2_data_plane=False,
        market_cap_override_mm=5_163_750.0,
        market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
        quote_snapshot_id="fixture-snapshot",
    )

    assert result["market_cap"] == 5_163_750.0
    assert result["market_cap_unit"] == MARKET_CAP_UNIT_USD_MILLIONS
    assert result["ttm_fcf"] == 96_676.0
    assert result["fcf_yield"] == 96_676.0 / 5_163_750.0


def test_v1_market_cap_override_without_literal_unit_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_rows",
        lambda *_args, **_kwargs: {2025: {"cfo": 120.0, "capex": 20.0}},
    )

    result = _fcf_yield_metrics(
        "UNIT",
        as_of_date="2026-07-22",
        v2_data_plane=False,
        market_cap_override_mm=2_000.0,
        market_cap_unit=None,
    )

    assert result["fcf_yield"] is None
    assert result["fcf_yield_reasons"] == ["MARKET_CAP_MISSING"]


def test_legitimate_negative_enterprise_value_and_multiple_reconcile() -> None:
    metrics = _canonical_current_valuation_metrics(
        "CASH",
        as_of_date="2026-07-22",
        market_cap_mm=100.0,
        market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
        quote_snapshot_id="cash-snapshot",
        fundamental_values={
            "total_debt": 0.0,
            "cash": 200.0,
            "operating_income": 15.0,
            "depreciation_amortization": 5.0,
            "net_income": 10.0,
            "equity": 50.0,
            "cfo": 12.0,
            "capex": 2.0,
        },
    )

    assert metrics["enterprise_value"] == -100.0
    assert metrics["ev_to_ebitda"] == -5.0
    assert metrics["metric_traces"]["enterprise_value"]["reconciles"] is True
    assert metrics["metric_traces"]["ev_to_ebitda"]["reconciles"] is True


@pytest.mark.parametrize(
    ("missing_field", "expected_reason"),
    [
        ("total_debt", "MISSING_OR_UNPROVEN_TOTAL_DEBT"),
        ("cash", "MISSING_OR_UNPROVEN_CASH"),
    ],
)
def test_missing_debt_or_cash_never_becomes_zero(
    missing_field: str,
    expected_reason: str,
) -> None:
    values = {
        "total_debt": 0.0,
        "cash": 200.0,
        "operating_income": 15.0,
        "depreciation_amortization": 5.0,
        "net_income": 10.0,
        "equity": 50.0,
        "cfo": 12.0,
        "capex": 2.0,
    }
    del values[missing_field]

    metrics = _canonical_current_valuation_metrics(
        "MISSING",
        as_of_date="2026-07-22",
        market_cap_mm=100.0,
        market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
        quote_snapshot_id="missing-balance-sheet-input",
        fundamental_values=values,
    )

    assert metrics["status"] == NEEDS_DATA
    assert metrics["reason"] == expected_reason
    assert metrics["enterprise_value"] is None
    assert metrics["ev_to_ebitda"] is None
    assert "enterprise_value" not in metrics["metric_traces"]


def test_report_uses_packet_market_cap_mm_without_database_inference(monkeypatch) -> None:
    packet = SectorCompanyFinancialPacket(
        ticker="REPORT",
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status="OK",
        market_cap_mm=459_455.0,
        market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
        current_price=579.43,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_report._fetch_market_cap",
        lambda _ticker: pytest.fail("renderer must not query a second market cap"),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_report._latest_shares_outstanding",
        lambda _ticker: pytest.fail("renderer must not infer cap from shares"),
    )

    assert _market_cap(packet, "REPORT") == 459_455.0
    assert _fmt_market_cap(_market_cap(packet, "REPORT")) == "$459.5B"


def test_active_v1_packet_uses_cap_quote_and_persists_formula_lineage(monkeypatch) -> None:
    quote_snapshot_id = stable_quote_hash(
        ticker="ACTIVE",
        price=100.0,
        as_of_date="2026-07-21",
        currency="USD",
        source="fixture_cap_quote",
        source_url="https://example.test/active",
        price_basis=PRICE_BASIS_UNADJUSTED,
        raw_price=100.0,
        split_adjustment_factor=1.0,
    )
    facts = {
        2025: {
            "cfo": 120.0,
            "capex": 20.0,
            "total_debt": 100.0,
            "cash": 200.0,
            "operating_income": 150.0,
            "depreciation_amortization": 50.0,
            "net_income": 100.0,
            "equity": 500.0,
        }
    }
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_rows",
        lambda *_args, **_kwargs: facts,
    )
    fact_provenance = {
        line_item: {
            "value": value,
            "unit": "USD_millions",
            "source": "SEC_COMPANYFACTS",
            "period_end": "2025-12-31",
            "filed_date": "2026-02-15",
            "source_reference": (f"https://data.sec.gov/api/xbrl/companyfacts/{line_item}"),
        }
        for line_item, value in facts[2025].items()
    }
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._packet_annual_fact_provenance_rows",
        lambda *_args, **_kwargs: {2025: fact_provenance},
    )
    source = TickerSignalPacket(
        ticker="ACTIVE",
        current_price=50.0,
        dcf_value=150.0,
        pricing_zone="MARGIN_OF_SAFETY",
        raw_valuation={
            "pricing_zone_detail": {
                "current_price": 50.0,
                "dcf_base": 150.0,
            }
        },
    )

    packet = build_sector_company_financial_packet(
        source,
        as_of_date="2026-07-22",
        pipeline_version="v1",
        cap_classification={
            "market_cap_mm": 2_000.0,
            "market_cap_unit": MARKET_CAP_UNIT_USD_MILLIONS,
            "cap_source": "stale_shares",
            "cap_source_kind": "SEC",
            "cap_method": "price_times_shares_divided_by_issuer_quote_ratio",
            "price_used": 100.0,
            "price_as_of_date": "2026-07-21",
            "price_currency": "USD",
            "price_source": "fixture_cap_quote",
            "price_source_url": "https://example.test/active",
            "quote_snapshot_id": quote_snapshot_id,
            "price_basis": PRICE_BASIS_UNADJUSTED,
            "raw_price": 100.0,
            "split_adjustment_factor": 1.0,
            "shares_mm": 20.0,
            "raw_shares_outstanding_mm": 20.0,
            "raw_shares_source_value": 20_000_000.0,
            "raw_shares_source_unit": "shares",
            "shares_unit": SHARES_UNIT_MILLIONS,
            "shares_basis": SHARES_BASIS_ISSUER_REPORTED,
            "shares_period_end": "2026-03-31",
            "shares_filed_date": "2026-05-01",
            "shares_source": "SEC_COMPANYFACTS",
            "shares_source_url": ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"),
            "issuer_quote_ratio": 1.0,
        },
    )

    assert packet.current_price == 100.0
    assert packet.cap_stage_price == 100.0
    assert packet.quote_snapshot_id == quote_snapshot_id
    assert packet.cap_stage_quote_snapshot_id == quote_snapshot_id
    assert packet.market_cap_mm == 2_000.0
    assert packet.market_cap_unit == MARKET_CAP_UNIT_USD_MILLIONS
    assert packet.metric_traces["market_cap_mm"]["reconciles"] is True
    shares_trace = packet.metric_traces["market_cap_mm"]["input_provenance"][
        "shares_outstanding_mm"
    ]
    assert shares_trace["raw_source_value"] == 20_000_000.0
    assert shares_trace["raw_source_unit"] == "shares"
    assert shares_trace["normalized_value"] == 20.0
    assert shares_trace["normalized_unit"] == SHARES_UNIT_MILLIONS
    assert packet.metric_traces["fcf_yield"]["reconciles"] is True
    assert packet.valuation["fcf_yield"] == 0.05


def test_scenario_inherits_quote_identity_and_reconciling_trace() -> None:
    packet = SectorCompanyFinancialPacket(
        ticker="SCENARIO",
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status="OK",
        current_price=100.0,
        current_price_unit=PRICE_UNIT_USD_PER_SHARE,
        quote_snapshot_id="scenario-snapshot",
        price_basis=PRICE_BASIS_UNADJUSTED,
        valuation={"anchor_method": "dcf", "valuation_anchor": 125.0},
    )

    scenarios = build_expected_return_scenarios(packet, horizons=[5])

    assert len(scenarios) == 3
    assert {item.quote_snapshot_id for item in scenarios} == {"scenario-snapshot"}
    assert {item.current_price_unit for item in scenarios} == {PRICE_UNIT_USD_PER_SHARE}
    assert all(item.metric_trace["reconciles"] is True for item in scenarios)


def test_stored_metric_must_equal_trace_output() -> None:
    packet = _valid_packet()
    packet["valuation"]["fcf_yield"] = 0.06

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("stored_metric_mismatch", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "DERIVED_METRIC_TRACE_OUTPUT_MISMATCH" in {item.code for item in result.violations}


def test_self_consistent_false_trace_fails_independent_recomputation() -> None:
    packet = _valid_packet()
    packet["valuation"]["fcf_yield"] = 0.25
    false_trace = dict(packet["metric_traces"]["fcf_yield"])
    false_trace["output"] = 0.25
    false_trace["recomputed_output"] = 0.25
    false_trace["reconciles"] = True
    packet["metric_traces"]["fcf_yield"] = false_trace

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("false_trace", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "METRIC_TRACE_INDEPENDENT_RECOMPUTATION_FAILED" in {
        item.code for item in result.violations
    }


def test_arbitrary_formula_text_cannot_claim_valid_metric_lineage() -> None:
    packet = _valid_packet()
    trace = dict(packet["metric_traces"]["fcf_yield"])
    trace["formula"] = "trust the copied result"
    packet["metric_traces"]["fcf_yield"] = trace

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("formula_text", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "METRIC_TRACE_FORMULA_INVALID" in {item.code for item in result.violations}


def test_future_dated_cap_and_share_provenance_is_invalid() -> None:
    packet = _valid_packet()
    packet["market_cap_effective_as_of_date"] = "2026-07-23"
    packet["shares_as_of_date"] = "2026-07-24"

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope("future_provenance", "2026-07-22", packets=(packet,))
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert {"MARKET_CAP_ASOF_INVALID", "SHARES_ASOF_INVALID"}.issubset(
        {item.code for item in result.violations}
    )


def test_scenario_basis_must_equal_packet_basis() -> None:
    packet = _valid_packet()
    future_value = 100.0 * (1.1**5)
    scenario = {
        "ticker": "MEGA",
        "current_price": 100.0,
        "current_price_unit": PRICE_UNIT_USD_PER_SHARE,
        "price_basis": "SPLIT_ADJUSTED",
        "quote_snapshot_id": packet["quote_snapshot_id"],
        "annualized_return": 0.1,
        "metric_trace": canonical_metric_trace(
            metric="annualized_return",
            formula=(
                "round((estimated_future_value_per_share / current_price) "
                "** (1 / horizon_years) - 1, 6)"
            ),
            inputs={
                "estimated_future_value_per_share": future_value,
                "current_price": 100.0,
                "horizon_years": 5,
            },
            output=0.1,
            recomputed_output=0.1,
            output_unit="annualized_ratio",
            quote_snapshot_id=str(packet["quote_snapshot_id"]),
            input_provenance=_trace_input_provenance(
                {
                    "estimated_future_value_per_share": future_value,
                    "current_price": 100.0,
                    "horizon_years": 5,
                }
            ),
        ),
    }

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "scenario_basis",
            "2026-07-22",
            packets=(packet,),
            scenarios=(scenario,),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "SCENARIO_PRICE_BASIS_MISMATCH" in {item.code for item in result.violations}


def test_duplicate_ticker_packets_cannot_carry_different_snapshots() -> None:
    first = _valid_packet()
    duplicate = copy.deepcopy(first)
    duplicate["quote_snapshot_id"] = "b" * 64

    result = validate_financial_integrity_scope(
        FinancialIntegrityScope(
            "duplicate_ticker",
            "2026-07-22",
            packets=(first, duplicate),
        )
    )

    assert result.status == INVALID_FINANCIAL_INPUT
    assert "DUPLICATE_TICKER_QUOTE_SNAPSHOT_CONFLICT" in {item.code for item in result.violations}
