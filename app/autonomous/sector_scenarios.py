"""Deterministic expected-return scenarios for autonomous sector analysis."""

from __future__ import annotations

from typing import Any

from app.autonomous.financial_integrity import canonical_metric_trace
from app.autonomous.sector_contract import (
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
)


DEFAULT_HORIZONS = (5, 10)
SCENARIO_SPECS = (
    ("downside", 0.65),
    ("base", 1.00),
    ("upside", 1.20),
)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _optional_float(value: Any) -> float | None:
    return float(value) if _is_num(value) else None


def _round_float(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if _is_num(value):
            return float(value)
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _annualized_return(future_value: float, current_price: float, horizon_years: int) -> float:
    if current_price <= 0 or future_value <= 0 or horizon_years <= 0:
        return 0.0
    return (future_value / current_price) ** (1 / horizon_years) - 1


def _price(packet: SectorCompanyFinancialPacket) -> float | None:
    return _first_float(packet.current_price, packet.valuation.get("current_price"))


def _anchor(packet: SectorCompanyFinancialPacket) -> float | None:
    return _optional_float(packet.valuation.get("valuation_anchor"))


def _anchor_method(packet: SectorCompanyFinancialPacket) -> str | None:
    method = packet.valuation.get("anchor_method")
    return str(method) if method else None


def _source_growth(packet: SectorCompanyFinancialPacket) -> tuple[float, str, list[str]]:
    growth = _first_float(
        packet.business_quality.get("revenue_cagr_5y"),
        packet.reinvestment.get("revenue_cagr_5y"),
        packet.business_quality.get("revenue_cagr_3y"),
        packet.reinvestment.get("revenue_cagr_3y"),
    )
    if growth is None:
        return 0.0, "default_zero", ["GROWTH_ASSUMPTION_DEFAULTED_TO_ZERO"]
    return growth, "historical_revenue_cagr", []


def _share_count_cagr(packet: SectorCompanyFinancialPacket) -> float:
    value = _first_float(
        packet.capital_allocation.get("dilution_rate_shares_cagr"),
        packet.capital_allocation.get("share_count_cagr"),
        packet.reinvestment.get("share_count_cagr"),
    )
    return value if value is not None else 0.0


def _scenario_growth(scenario_name: str, source_growth: float, share_count_cagr: float) -> float:
    dilution_penalty = max(share_count_cagr, 0.0)
    if scenario_name == "downside":
        return _clamp(min(source_growth, 0.0) - dilution_penalty, -0.10, 0.05)
    if scenario_name == "upside":
        return _clamp(source_growth - share_count_cagr, -0.05, 0.18)
    return _clamp(source_growth - dilution_penalty, -0.05, 0.12)


def _normalized_operating_margin(packet: SectorCompanyFinancialPacket) -> float | None:
    return _first_float(
        packet.business_quality.get("operating_margin"),
        packet.valuation.get("normalized_operating_margin"),
        packet.returns_on_capital.get("operating_margin"),
    )


def _owner_earnings_per_share(packet: SectorCompanyFinancialPacket) -> float | None:
    return _first_float(
        packet.cash_conversion.get("owner_earnings_per_share"),
        packet.valuation.get("owner_earnings_per_share"),
    )


def _unsupported_assumptions(
    packet: SectorCompanyFinancialPacket,
    base_unsupported: list[str],
) -> list[str]:
    unsupported = list(base_unsupported)
    if packet.blockers:
        unsupported.append("BLOCKERS_PRESENT")
    if packet.model_fit_status not in {"VALID_GENERIC", "VALID_SECTOR_SPECIFIC"}:
        unsupported.append("MODEL_FIT_NOT_FULLY_VALIDATED")
    if packet.data_quality_status != "OK":
        unsupported.append(f"DATA_QUALITY_{packet.data_quality_status}")
    return list(dict.fromkeys(unsupported))


def build_expected_return_scenarios(
    packet: SectorCompanyFinancialPacket,
    *,
    horizons: list[int] | tuple[int, ...] | None = None,
) -> list[SectorExpectedReturnScenario]:
    """Build downside/base/upside return scenarios from one financial packet.

    The v1 engine is intentionally conservative and auditable: it compounds the
    packet's valuation anchor by a scenario-specific per-share growth rate and a
    terminal anchor multiplier. The LLM can critique these assumptions later,
    but the math itself stays deterministic.
    """

    current_price = _price(packet)
    valuation_anchor = _anchor(packet)
    if (
        current_price is None
        or current_price <= 0
        or valuation_anchor is None
        or valuation_anchor <= 0
    ):
        return []

    horizon_values = [int(item) for item in (horizons or DEFAULT_HORIZONS) if int(item) > 0]
    if not horizon_values:
        return []

    source_growth, growth_source, base_unsupported = _source_growth(packet)
    share_count_cagr = _share_count_cagr(packet)
    normalized_margin = _normalized_operating_margin(packet)
    owner_earnings = _owner_earnings_per_share(packet)
    unsupported = _unsupported_assumptions(packet, base_unsupported)
    anchor_method = _anchor_method(packet)

    scenarios: list[SectorExpectedReturnScenario] = []
    for horizon_years in horizon_values:
        for scenario_name, terminal_anchor_multiplier in SCENARIO_SPECS:
            scenario_growth = _scenario_growth(scenario_name, source_growth, share_count_cagr)
            future_value = (
                valuation_anchor
                * terminal_anchor_multiplier
                * ((1 + scenario_growth) ** horizon_years)
            )
            annualized = _annualized_return(future_value, current_price, horizon_years)
            rounded_current_price = _round_float(current_price)
            rounded_future_value = _round_float(future_value)
            rounded_annualized = _round_float(annualized)
            scenario_id = f"{packet.ticker}_{scenario_name}_{horizon_years}Y"
            scenarios.append(
                SectorExpectedReturnScenario(
                    scenario_id=scenario_id,
                    ticker=packet.ticker,
                    scenario_name=scenario_name,
                    horizon_years=horizon_years,
                    current_price=rounded_current_price,
                    estimated_future_value_per_share=rounded_future_value,
                    annualized_return=rounded_annualized,
                    current_price_unit=packet.current_price_unit,
                    quote_snapshot_id=packet.quote_snapshot_id,
                    price_basis=packet.price_basis,
                    metric_trace=canonical_metric_trace(
                        metric="annualized_return",
                        formula=(
                            "round((estimated_future_value_per_share / current_price) "
                            "** (1 / horizon_years) - 1, 6)"
                        ),
                        inputs={
                            "estimated_future_value_per_share": future_value,
                            "current_price": current_price,
                            "horizon_years": horizon_years,
                        },
                        output=rounded_annualized,
                        recomputed_output=round(
                            (future_value / current_price) ** (1 / horizon_years) - 1,
                            6,
                        ),
                        output_unit="annualized_ratio",
                        quote_snapshot_id=packet.quote_snapshot_id,
                        input_provenance={
                            "estimated_future_value_per_share": {
                                "value": future_value,
                                "unit": "USD_per_share",
                                "source": "DETERMINISTIC_SCENARIO_MODEL",
                                "period_end": packet.current_price_as_of_date,
                                "filed_date": packet.current_price_as_of_date,
                                "source_reference": scenario_id,
                            },
                            "current_price": {
                                "value": current_price,
                                "unit": packet.current_price_unit,
                                "source": packet.current_price_source,
                                "period_end": packet.current_price_as_of_date,
                                "filed_date": packet.current_price_as_of_date,
                                "source_reference": packet.current_price_source_url,
                            },
                            "horizon_years": {
                                "value": horizon_years,
                                "unit": "years",
                                "source": "DETERMINISTIC_SCENARIO_MODEL",
                                "period_end": packet.current_price_as_of_date,
                                "filed_date": packet.current_price_as_of_date,
                                "source_reference": scenario_id,
                            },
                        },
                    ),
                    revenue_cagr=_round_float(scenario_growth),
                    normalized_operating_margin=_round_float(normalized_margin),
                    owner_earnings_per_share=_round_float(owner_earnings),
                    terminal_multiple=_round_float(terminal_anchor_multiplier),
                    share_count_cagr=_round_float(share_count_cagr),
                    downside_value_per_share=(
                        _round_float(future_value) if scenario_name == "downside" else None
                    ),
                    assumptions={
                        "valuation_anchor": _round_float(valuation_anchor),
                        "anchor_method": anchor_method,
                        "source_revenue_cagr": _round_float(source_growth),
                        "growth_source": growth_source,
                        "scenario_growth_rate": _round_float(scenario_growth),
                        "terminal_anchor_multiplier": _round_float(terminal_anchor_multiplier),
                        "share_count_cagr": _round_float(share_count_cagr),
                        "formula": "valuation_anchor * terminal_anchor_multiplier * (1 + scenario_growth_rate) ** horizon_years",
                    },
                    key_sensitivities=[
                        "valuation_anchor",
                        "scenario_growth_rate",
                        "terminal_anchor_multiplier",
                        "share_count_cagr",
                    ],
                    unsupported_assumptions=unsupported,
                    evidence_ref_ids=list(packet.evidence_ref_ids),
                )
            )
    return scenarios


def build_expected_return_scenarios_for_packets(
    packets: list[SectorCompanyFinancialPacket],
    *,
    horizons: list[int] | tuple[int, ...] | None = None,
) -> dict[str, list[SectorExpectedReturnScenario]]:
    """Build expected-return scenarios keyed by ticker."""

    return {
        packet.ticker: build_expected_return_scenarios(packet, horizons=horizons)
        for packet in packets
    }


__all__ = [
    "build_expected_return_scenarios",
    "build_expected_return_scenarios_for_packets",
]
