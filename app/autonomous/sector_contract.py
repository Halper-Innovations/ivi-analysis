"""Pure contracts for autonomous sector-level financial analysis.

The sector contract extends the single-candidate autonomous run shape to the
larger job: compare many companies in one sector, underwrite 5-10 year
per-share return potential, and either select the best candidate or stop with
no selection. This module is intentionally pure:

- no I/O
- no config access
- no DB access
- no LLM/provider imports
- repeated fields default to empty lists
- ``to_dict`` returns JSON-safe dictionaries
- ``from_dict`` reconstructs nested dataclasses exactly
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from app.autonomous.run_contract import BeliefUpdate, EvidenceReference, ToolCallRecord


SECTOR_CONTRACT_VERSION_V1 = "autonomous_sector_financial_run_v1"
SECTOR_CONTRACT_VERSION_V2 = "autonomous_sector_financial_run_v2"
# Backwards-compatible default. V2 is rollout-gated and must always be chosen
# explicitly by the runtime or the cap-band allowlist.
SECTOR_CONTRACT_VERSION = SECTOR_CONTRACT_VERSION_V1

SECTOR_PIPELINE_VERSION_V1 = "v1"
SECTOR_PIPELINE_VERSION_V2 = "v2"

V2_EXECUTION_STATUSES = frozenset({"COMPLETED", "FAILED"})
V2_DECISION_STATUSES = frozenset({"COMPLETE", "INCOMPLETE"})
V2_FINAL_VERDICTS = frozenset({"SELECTED", "WATCHLIST", "NO_SELECTION"})
V2_TERMINAL_STATES = frozenset(
    {
        "OUT_OF_SCOPE",
        "DEFERRED_BY_BOUND",
        "SCREENED_OUT",
        "NEEDS_DATA",
        "READY_FOR_UNDERWRITING",
        "UNDERWRITTEN",
    }
)
V2_SCOPE_STATUSES = frozenset({"IN_SCOPE", "OUT_OF_SCOPE"})
V2_SCREEN_STATUSES = frozenset({"NOT_RUN", "PASS", "FAIL", "INCOMPLETE"})
V2_REVIEW_STATUSES = frozenset({"NOT_REQUIRED", "NOT_STARTED", "INCOMPLETE", "COMPLETED", "FAILED"})
V2_GATE_STATUSES = frozenset({"NOT_APPLICABLE", "PASS", "FAIL", "INCOMPLETE"})
V2_UNDERWRITING_STATUSES = frozenset(
    {"NOT_REQUIRED", "NOT_STARTED", "INCOMPLETE", "COMPLETED", "FAILED"}
)
V2_COMPLETED_UNDERWRITING_VERDICTS = frozenset({"ACTIONABLE", "WATCHLIST_ONLY", "AVOID"})
V2_FRONTIER_STATUSES = frozenset({"NOT_ELIGIBLE", "LIVE", "REVIEWED", "DOMINATED", "UNRESOLVED"})
V2_SELECTION_VALIDATION_STATUSES = frozenset(
    {"NOT_REQUIRED", "NOT_ATTEMPTED", "INCOMPLETE", "VALIDATED", "CONTRADICTED"}
)
V2_AFFIRMATIVE_VALIDATOR_VERDICTS = frozenset({"ACTIONABLE", "CONFIRMED_ACTIONABLE", "VALIDATED"})
V2_DECISION_FAILURE_STATES = frozenset(
    {
        "BUDGET_EXHAUSTED",
        "TURN_BUDGET_EXHAUSTED",
        "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
        "LLM_PROVIDER_UNAVAILABLE",
        "LLM_COST_BUDGET_EXCEEDED",
        "LLM_RETRY_BUDGET_EXCEEDED",
    }
)


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_json_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def canonical_v2_signal_packet_snapshot(packet: Any) -> dict[str, Any]:
    """Return the exact JSON input projection bound to every v2 child lane.

    Runtime signal packets are dataclasses today, while offline replay reads the
    persisted mapping.  Canonicalizing both through JSON gives the contract one
    immutable representation instead of trusting a caller-supplied digest map.
    """

    source = packet if isinstance(packet, Mapping) else getattr(packet, "__dict__", None)
    if not isinstance(source, Mapping):
        raise ValueError("v2 signal packet snapshot must be a mapping or dataclass instance")
    snapshot = json.loads(json.dumps(dict(source), sort_keys=True, default=str))
    if not isinstance(snapshot, dict) or not str(snapshot.get("ticker") or "").strip():
        raise ValueError("v2 signal packet snapshot requires ticker identity")
    return snapshot


def v2_signal_packet_snapshot_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Fingerprint a persisted canonical signal-packet snapshot."""

    return _canonical_json_fingerprint(canonical_v2_signal_packet_snapshot(snapshot))


def _mapping_signal_snapshot(value: Any, ticker: str) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    snapshot = value.get(str(ticker or "").strip().upper())
    return dict(snapshot) if isinstance(snapshot, Mapping) else None


_SELECTION_VALIDATION_TERMINAL_FIELDS = (
    "status",
    "selected_ticker",
    "validator_run_id",
    "validator_verdict",
    "reason_codes",
    "evidence_ref_ids",
    "evidence",
    "tool_calls",
    "provider_usage",
    "notes",
    "source_binding",
)


def selection_validation_terminal_ledger_fingerprint(
    payload: Mapping[str, Any],
) -> str:
    """Bind the complete terminal validator ledger without trusting child IDs."""

    projection = {
        field_name: payload.get(field_name) for field_name in _SELECTION_VALIDATION_TERMINAL_FIELDS
    }
    return _canonical_json_fingerprint(projection)


def build_v2_canonical_child_source_bindings(
    *,
    sector: str,
    as_of_date: str,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    signal_packet_snapshots: Mapping[str, Mapping[str, Any]],
    frontier_candidate_tickers: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the immutable child-input binding shared by runtime and replay."""

    ordered_tickers = [str(packet.ticker).strip().upper() for packet in company_packets]
    if any(not ticker for ticker in ordered_tickers) or len(ordered_tickers) != len(
        set(ordered_tickers)
    ):
        raise ValueError("v2 source binding requires unique complete company packets")
    frontier_tickers = (
        [str(ticker).strip().upper() for ticker in frontier_candidate_tickers]
        if frontier_candidate_tickers is not None
        else list(ordered_tickers)
    )
    if (
        any(not ticker for ticker in frontier_tickers)
        or len(frontier_tickers) != len(set(frontier_tickers))
        or not set(frontier_tickers).issubset(ordered_tickers)
    ):
        raise ValueError(
            "v2 frontier candidate bindings must be a unique subset of company packets"
        )
    normalized_snapshots = {
        str(ticker).strip().upper(): canonical_v2_signal_packet_snapshot(snapshot)
        for ticker, snapshot in signal_packet_snapshots.items()
        if str(ticker).strip().upper() in ordered_tickers
    }
    if set(normalized_snapshots) != set(ordered_tickers):
        raise ValueError(
            "v2 source binding requires one persisted signal snapshot per company packet"
        )
    for ticker, snapshot in normalized_snapshots.items():
        if str(snapshot.get("ticker") or "").strip().upper() != ticker:
            raise ValueError("v2 signal packet snapshot ticker identity mismatch")
    ordered_snapshots = {ticker: normalized_snapshots[ticker] for ticker in ordered_tickers}
    ordered_signals = {
        ticker: v2_signal_packet_snapshot_fingerprint(ordered_snapshots[ticker])
        for ticker in ordered_tickers
    }
    company_fingerprints = {
        str(packet.ticker).strip().upper(): _canonical_json_fingerprint(packet.to_dict())
        for packet in company_packets
    }
    scenario_rows = [
        scenario.to_dict()
        for scenario in scenarios
        if str(scenario.ticker).strip().upper() in set(ordered_tickers)
    ]
    cohort_fingerprint = _canonical_json_fingerprint(
        {
            "sector": sector,
            "as_of_date": as_of_date,
            "ordered_tickers": ordered_tickers,
            "company_packet_fingerprints": company_fingerprints,
            "signal_packet_fingerprints": ordered_signals,
            "expected_return_scenarios": scenario_rows,
            "frontier_candidate_tickers": frontier_tickers,
        }
    )
    return {
        ticker: {
            "artifact_type": "v2_canonical_child_source_binding_v1",
            "pipeline_version": SECTOR_PIPELINE_VERSION_V2,
            "sector": sector,
            "ticker": ticker,
            "as_of_date": as_of_date,
            "signal_packet_fingerprint": ordered_signals[ticker],
            "cohort_signal_packet_fingerprints": dict(ordered_signals),
            "company_packet_fingerprint": company_fingerprints[ticker],
            "cohort_fingerprint": cohort_fingerprint,
            "cohort_tickers": list(ordered_tickers),
            "frontier_candidate_tickers": list(frontier_tickers),
        }
        for ticker in frontier_tickers
    }


def _validate_selection_source_binding(
    source_binding: dict[str, Any],
    *,
    selected_ticker: str,
) -> None:
    ticker = str(selected_ticker or "").strip().upper()
    cohort_tickers = source_binding.get("cohort_tickers")
    frontier_candidate_tickers = source_binding.get("frontier_candidate_tickers")
    normalized_cohort = (
        [str(item).strip().upper() for item in cohort_tickers]
        if isinstance(cohort_tickers, list)
        else []
    )
    normalized_frontier = (
        [str(item).strip().upper() for item in frontier_candidate_tickers]
        if isinstance(frontier_candidate_tickers, list)
        else []
    )
    cohort_signals = source_binding.get("cohort_signal_packet_fingerprints")
    normalized_signals = (
        {
            str(key).strip().upper(): str(value or "").strip().lower()
            for key, value in cohort_signals.items()
        }
        if isinstance(cohort_signals, dict)
        else {}
    )
    valid = (
        source_binding.get("artifact_type") == "v2_canonical_child_source_binding_v1"
        and source_binding.get("pipeline_version") == SECTOR_PIPELINE_VERSION_V2
        and bool(str(source_binding.get("sector") or "").strip())
        and str(source_binding.get("ticker") or "").strip().upper() == ticker
        and bool(str(source_binding.get("as_of_date") or "").strip())
        and _is_sha256(source_binding.get("signal_packet_fingerprint"))
        and normalized_signals.get(ticker)
        == str(source_binding.get("signal_packet_fingerprint") or "").strip().lower()
        and all(_is_sha256(value) for value in normalized_signals.values())
        and _is_sha256(source_binding.get("company_packet_fingerprint"))
        and _is_sha256(source_binding.get("cohort_fingerprint"))
        and isinstance(cohort_tickers, list)
        and ticker in normalized_cohort
        and len(normalized_cohort) == len(set(normalized_cohort))
        and isinstance(frontier_candidate_tickers, list)
        and ticker in normalized_frontier
        and len(normalized_frontier) == len(set(normalized_frontier))
        and set(normalized_frontier).issubset(normalized_cohort)
    )
    if not valid:
        raise ValueError(
            "VALIDATED/CONTRADICTED selection requires a canonical selected-company source_binding"
        )


def _v2_blocking_run_states(states: list[str]) -> list[str]:
    normalized = {str(state).strip().upper() for state in states}
    blockers = sorted(normalized & V2_DECISION_FAILURE_STATES)
    recovered = bool(normalized & {"LLM_PROVIDER_TURN_RECOVERED", "LLM_PROVIDER_JSON_RECOVERED"})
    if not recovered:
        blockers.extend(
            sorted(
                state
                for state in normalized
                if state.startswith("LLM_PROVIDER_")
                and not state.endswith("_RECOVERED")
                and state not in blockers
            )
        )
    return list(dict.fromkeys(blockers))


@dataclass
class SectorFinancialFramework:
    """Sector-specific financial lens chosen before company comparison."""

    sector: str
    market_cap_focus: str
    horizon_years: list[int]
    economic_model: str
    framework_contract_id: str | None = None
    selected_value_drivers: list[str] = field(default_factory=list)
    selected_metrics: list[str] = field(default_factory=list)
    valid_valuation_methods: list[str] = field(default_factory=list)
    invalid_valuation_methods: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    applicable_screen_rule_ids: list[str] = field(default_factory=list)
    normalization_policy: dict[str, Any] = field(default_factory=dict)
    hurdle_rate_policy: dict[str, Any] = field(default_factory=dict)
    weighting_policy: dict[str, Any] = field(default_factory=dict)
    sector_specific_risks: list[str] = field(default_factory=list)
    contract_version: str = SECTOR_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectorFinancialFramework":
        return cls(
            sector=data["sector"],
            market_cap_focus=data["market_cap_focus"],
            horizon_years=[int(item) for item in data.get("horizon_years", [])],
            economic_model=data["economic_model"],
            framework_contract_id=data.get("framework_contract_id"),
            selected_value_drivers=list(data.get("selected_value_drivers", [])),
            selected_metrics=list(data.get("selected_metrics", [])),
            valid_valuation_methods=list(data.get("valid_valuation_methods", [])),
            invalid_valuation_methods=list(data.get("invalid_valuation_methods", [])),
            required_evidence=list(data.get("required_evidence", [])),
            applicable_screen_rule_ids=list(data.get("applicable_screen_rule_ids", [])),
            normalization_policy=dict(data.get("normalization_policy", {})),
            hurdle_rate_policy=dict(data.get("hurdle_rate_policy", {})),
            weighting_policy=dict(data.get("weighting_policy", {})),
            sector_specific_risks=list(data.get("sector_specific_risks", [])),
            contract_version=data.get("contract_version", SECTOR_CONTRACT_VERSION),
        )


@dataclass
class SectorCompanyFinancialPacket:
    """Comparable financial packet for one company in a sector run."""

    ticker: str
    financial_status: str
    model_fit_status: str
    data_quality_status: str
    market_cap_category: str | None = None
    # Band-filter chain provenance (app/autonomous/cap_resolver.py): cap in
    # millions and which tier resolved it. A None category with a non-None
    # source of "unknown" means the name must render as UNKNOWN_CAP and can
    # never be presented as in-band output.
    market_cap_mm: float | None = None
    market_cap_unit: str | None = None
    market_cap_source: str | None = None
    market_cap_effective_as_of_date: str | None = None
    market_cap_source_kind: str | None = None
    market_cap_source_name: str | None = None
    market_cap_source_url: str | None = None
    market_cap_confidence: str | None = None
    # Exact quote used by the cap stage. This is intentionally distinct from
    # ``current_price`` (the valuation packet's price) so a later stage cannot
    # silently replace the scope-admission price or its provenance.
    cap_stage_price: float | None = None
    cap_stage_price_as_of_date: str | None = None
    cap_stage_price_currency: str | None = None
    cap_stage_price_source: str | None = None
    cap_stage_price_source_url: str | None = None
    cap_stage_price_confidence: str | None = None
    cap_stage_quote_snapshot_id: str | None = None
    issuer_cik: str | None = None
    issuer_primary_ticker: str | None = None
    issuer_listed_tickers: list[str] = field(default_factory=list)
    security_role: str | None = None
    is_secondary_class: bool | None = None
    is_adr: bool | None = None
    adr_ratio: float | None = None
    share_class_ratio: float | None = None
    identity_source: str | None = None
    identity_source_url: str | None = None
    identity_as_of_date: str | None = None
    identity_confidence: str | None = None
    ratio_source_url: str | None = None
    ratio_source_accession: str | None = None
    ratio_security_symbol: str | None = None
    current_price: float | None = None
    current_price_unit: str | None = None
    current_price_as_of_date: str | None = None
    current_price_currency: str | None = None
    current_price_source: str | None = None
    current_price_source_url: str | None = None
    current_price_confidence: str | None = None
    quote_snapshot_id: str | None = None
    price_basis: str | None = None
    raw_price: float | None = None
    shares_outstanding_mm: float | None = None
    raw_shares_outstanding_mm: float | None = None
    raw_shares_source_value: float | None = None
    raw_shares_source_unit: str | None = None
    shares_unit: str | None = None
    shares_basis: str | None = None
    shares_as_of_date: str | None = None
    shares_filed_date: str | None = None
    shares_source: str | None = None
    shares_source_url: str | None = None
    issuer_quote_ratio: float | None = None
    split_adjustment_factor: float | None = None
    split_effective_date: str | None = None
    split_lineage_proof: dict[str, Any] | None = None
    market_cap_method: str | None = None
    market_cap_derivation: dict[str, Any] = field(default_factory=dict)
    metric_traces: dict[str, Any] = field(default_factory=dict)
    financial_integrity_status: str | None = None
    financial_integrity_violations: list[dict[str, Any]] = field(default_factory=list)
    business_quality: dict[str, Any] = field(default_factory=dict)
    reinvestment: dict[str, Any] = field(default_factory=dict)
    returns_on_capital: dict[str, Any] = field(default_factory=dict)
    cash_conversion: dict[str, Any] = field(default_factory=dict)
    balance_sheet: dict[str, Any] = field(default_factory=dict)
    capital_allocation: dict[str, Any] = field(default_factory=dict)
    accounting_quality: dict[str, Any] = field(default_factory=dict)
    valuation: dict[str, Any] = field(default_factory=dict)
    expected_return: dict[str, Any] = field(default_factory=dict)
    score_components: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    confidence_caps: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectorCompanyFinancialPacket":
        price = data.get("current_price")
        return cls(
            ticker=data["ticker"],
            financial_status=data["financial_status"],
            model_fit_status=data["model_fit_status"],
            data_quality_status=data["data_quality_status"],
            market_cap_category=data.get("market_cap_category"),
            market_cap_mm=(
                float(data["market_cap_mm"])
                if isinstance(data.get("market_cap_mm"), (int, float))
                else None
            ),
            market_cap_unit=data.get("market_cap_unit"),
            market_cap_source=data.get("market_cap_source"),
            market_cap_effective_as_of_date=data.get("market_cap_effective_as_of_date"),
            market_cap_source_kind=data.get("market_cap_source_kind"),
            market_cap_source_name=data.get("market_cap_source_name"),
            market_cap_source_url=data.get("market_cap_source_url"),
            market_cap_confidence=data.get("market_cap_confidence"),
            cap_stage_price=(
                float(data["cap_stage_price"])
                if isinstance(data.get("cap_stage_price"), (int, float))
                else None
            ),
            cap_stage_price_as_of_date=data.get("cap_stage_price_as_of_date"),
            cap_stage_price_currency=data.get("cap_stage_price_currency"),
            cap_stage_price_source=data.get("cap_stage_price_source"),
            cap_stage_price_source_url=data.get("cap_stage_price_source_url"),
            cap_stage_price_confidence=data.get("cap_stage_price_confidence"),
            cap_stage_quote_snapshot_id=data.get("cap_stage_quote_snapshot_id"),
            issuer_cik=data.get("issuer_cik"),
            issuer_primary_ticker=data.get("issuer_primary_ticker"),
            issuer_listed_tickers=[
                str(item).upper()
                for item in data.get("issuer_listed_tickers", [])
                if str(item).strip()
            ],
            security_role=data.get("security_role"),
            is_secondary_class=(
                bool(data["is_secondary_class"])
                if isinstance(data.get("is_secondary_class"), bool)
                else None
            ),
            is_adr=(bool(data["is_adr"]) if isinstance(data.get("is_adr"), bool) else None),
            adr_ratio=(
                float(data["adr_ratio"])
                if isinstance(data.get("adr_ratio"), (int, float))
                else None
            ),
            share_class_ratio=(
                float(data["share_class_ratio"])
                if isinstance(data.get("share_class_ratio"), (int, float))
                else None
            ),
            identity_source=data.get("identity_source"),
            identity_source_url=data.get("identity_source_url"),
            identity_as_of_date=data.get("identity_as_of_date"),
            identity_confidence=data.get("identity_confidence"),
            ratio_source_url=data.get("ratio_source_url"),
            ratio_source_accession=data.get("ratio_source_accession"),
            ratio_security_symbol=data.get("ratio_security_symbol"),
            current_price=float(price) if price is not None else None,
            current_price_unit=data.get("current_price_unit"),
            current_price_as_of_date=data.get("current_price_as_of_date"),
            current_price_currency=data.get("current_price_currency"),
            current_price_source=data.get("current_price_source"),
            current_price_source_url=data.get("current_price_source_url"),
            current_price_confidence=data.get("current_price_confidence"),
            quote_snapshot_id=data.get("quote_snapshot_id"),
            price_basis=data.get("price_basis"),
            raw_price=(
                float(data["raw_price"])
                if isinstance(data.get("raw_price"), (int, float))
                else None
            ),
            shares_outstanding_mm=(
                float(data["shares_outstanding_mm"])
                if isinstance(data.get("shares_outstanding_mm"), (int, float))
                else None
            ),
            raw_shares_outstanding_mm=(
                float(data["raw_shares_outstanding_mm"])
                if isinstance(data.get("raw_shares_outstanding_mm"), (int, float))
                else None
            ),
            raw_shares_source_value=(
                float(data["raw_shares_source_value"])
                if isinstance(data.get("raw_shares_source_value"), (int, float))
                else None
            ),
            raw_shares_source_unit=data.get("raw_shares_source_unit"),
            shares_unit=data.get("shares_unit"),
            shares_basis=data.get("shares_basis"),
            shares_as_of_date=data.get("shares_as_of_date"),
            shares_filed_date=data.get("shares_filed_date"),
            shares_source=data.get("shares_source"),
            shares_source_url=data.get("shares_source_url"),
            issuer_quote_ratio=(
                float(data["issuer_quote_ratio"])
                if isinstance(data.get("issuer_quote_ratio"), (int, float))
                else None
            ),
            split_adjustment_factor=(
                float(data["split_adjustment_factor"])
                if isinstance(data.get("split_adjustment_factor"), (int, float))
                else None
            ),
            split_effective_date=data.get("split_effective_date"),
            split_lineage_proof=(
                dict(data["split_lineage_proof"])
                if isinstance(data.get("split_lineage_proof"), dict)
                else None
            ),
            market_cap_method=data.get("market_cap_method"),
            market_cap_derivation=dict(data.get("market_cap_derivation", {})),
            metric_traces=dict(data.get("metric_traces", {})),
            financial_integrity_status=data.get("financial_integrity_status"),
            financial_integrity_violations=[
                dict(item)
                for item in data.get("financial_integrity_violations", [])
                if isinstance(item, dict)
            ],
            business_quality=dict(data.get("business_quality", {})),
            reinvestment=dict(data.get("reinvestment", {})),
            returns_on_capital=dict(data.get("returns_on_capital", {})),
            cash_conversion=dict(data.get("cash_conversion", {})),
            balance_sheet=dict(data.get("balance_sheet", {})),
            capital_allocation=dict(data.get("capital_allocation", {})),
            accounting_quality=dict(data.get("accounting_quality", {})),
            valuation=dict(data.get("valuation", {})),
            expected_return=dict(data.get("expected_return", {})),
            score_components=dict(data.get("score_components", {})),
            blockers=list(data.get("blockers", [])),
            confidence_caps=list(data.get("confidence_caps", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
        )


@dataclass
class SectorExpectedReturnScenario:
    """Downside/base/upside expected-return case for a sector finalist."""

    scenario_id: str
    ticker: str
    scenario_name: str
    horizon_years: int
    current_price: float | None
    estimated_future_value_per_share: float | None
    annualized_return: float | None
    current_price_unit: str | None = None
    quote_snapshot_id: str | None = None
    price_basis: str | None = None
    metric_trace: dict[str, Any] = field(default_factory=dict)
    financial_integrity_status: str | None = None
    financial_integrity_violations: list[dict[str, Any]] = field(default_factory=list)
    revenue_cagr: float | None = None
    normalized_operating_margin: float | None = None
    owner_earnings_per_share: float | None = None
    terminal_multiple: float | None = None
    share_count_cagr: float | None = None
    downside_value_per_share: float | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)
    key_sensitivities: list[str] = field(default_factory=list)
    unsupported_assumptions: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectorExpectedReturnScenario":
        return cls(
            scenario_id=data["scenario_id"],
            ticker=data["ticker"],
            scenario_name=data["scenario_name"],
            horizon_years=int(data["horizon_years"]),
            current_price=_optional_float(data.get("current_price")),
            estimated_future_value_per_share=_optional_float(
                data.get("estimated_future_value_per_share")
            ),
            annualized_return=_optional_float(data.get("annualized_return")),
            current_price_unit=data.get("current_price_unit"),
            quote_snapshot_id=data.get("quote_snapshot_id"),
            price_basis=data.get("price_basis"),
            metric_trace=dict(data.get("metric_trace", {})),
            financial_integrity_status=data.get("financial_integrity_status"),
            financial_integrity_violations=[
                dict(item)
                for item in data.get("financial_integrity_violations", [])
                if isinstance(item, dict)
            ],
            revenue_cagr=_optional_float(data.get("revenue_cagr")),
            normalized_operating_margin=_optional_float(data.get("normalized_operating_margin")),
            owner_earnings_per_share=_optional_float(data.get("owner_earnings_per_share")),
            terminal_multiple=_optional_float(data.get("terminal_multiple")),
            share_count_cagr=_optional_float(data.get("share_count_cagr")),
            downside_value_per_share=_optional_float(data.get("downside_value_per_share")),
            assumptions=dict(data.get("assumptions", {})),
            key_sensitivities=list(data.get("key_sensitivities", [])),
            unsupported_assumptions=list(data.get("unsupported_assumptions", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
        )


@dataclass
class SectorResearchQuestion:
    """Financial research question selected for sector-level comparison."""

    question_id: str
    question: str
    financial_pillar: str
    expected_decision_impact: str
    priority: str
    status: str
    target_tickers: list[str] = field(default_factory=list)
    planned_tools: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectorResearchQuestion":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            financial_pillar=data["financial_pillar"],
            expected_decision_impact=data["expected_decision_impact"],
            priority=data["priority"],
            status=data["status"],
            target_tickers=list(data.get("target_tickers", [])),
            planned_tools=list(data.get("planned_tools", [])),
            depends_on=list(data.get("depends_on", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
        )


@dataclass
class SectorFinalDecision:
    """Final sector-level selection or no-selection decision."""

    verdict: str
    confidence: str | None
    selected_ticker: str | None
    expected_annualized_return_range: str | None
    thesis: str
    key_risk: str
    downside_case: str
    no_selection_reason: str | None = None
    falsifiers: list[str] = field(default_factory=list)
    why_selected_over_finalists: list[str] = field(default_factory=list)
    rejected_finalists: list[dict[str, Any]] = field(default_factory=list)
    selection_blockers: list[str] = field(default_factory=list)
    confidence_cap_reasons: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)
    data_resolution_needed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectorFinalDecision":
        return cls(
            verdict=data["verdict"],
            confidence=data.get("confidence"),
            selected_ticker=data.get("selected_ticker"),
            expected_annualized_return_range=data.get("expected_annualized_return_range"),
            thesis=data["thesis"],
            key_risk=data["key_risk"],
            downside_case=data["downside_case"],
            no_selection_reason=data.get("no_selection_reason"),
            falsifiers=list(data.get("falsifiers", [])),
            why_selected_over_finalists=list(data.get("why_selected_over_finalists", [])),
            rejected_finalists=[dict(item) for item in data.get("rejected_finalists", [])],
            selection_blockers=list(data.get("selection_blockers", [])),
            confidence_cap_reasons=list(data.get("confidence_cap_reasons", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
            data_resolution_needed=list(data.get("data_resolution_needed", [])),
        )


@dataclass
class GateEvaluation:
    """One deterministic, sector-contract gate evaluation.

    A failing gate is only constructible when it is applicable and carries a
    concrete observation, threshold, and evidence reference.  This makes the
    v2 ``SCREENED_OUT`` state provenance-backed by construction instead of a
    projection of a generic blocker string.
    """

    contract_id: str
    rule_id: str
    status: str
    applicable: bool
    observed_value: Any = None
    threshold: Any = None
    evidence_ref_id: str | None = None
    evidence_url: str | None = None
    reason_code: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.contract_id = str(self.contract_id or "").strip()
        self.rule_id = str(self.rule_id or "").strip().upper()
        self.status = str(self.status or "").strip().upper()
        self.evidence_ref_id = str(self.evidence_ref_id or "").strip() or None
        self.evidence_url = str(self.evidence_url or "").strip() or None
        self.reason_code = str(self.reason_code or "").strip().upper() or None
        if not self.contract_id:
            raise ValueError("gate evaluation contract_id is required")
        if not self.rule_id:
            raise ValueError("gate evaluation rule_id is required")
        if self.status not in V2_GATE_STATUSES:
            raise ValueError(f"invalid gate evaluation status: {self.status}")
        if self.status == "NOT_APPLICABLE" and self.applicable:
            raise ValueError("NOT_APPLICABLE gate evaluation cannot be applicable")
        if self.status != "NOT_APPLICABLE" and not self.applicable:
            raise ValueError("an inapplicable gate must use NOT_APPLICABLE status")
        if self.status in {"PASS", "FAIL"} and (
            self.observed_value is None
            or self.threshold is None
            or not (self.evidence_ref_id or self.evidence_url)
        ):
            raise ValueError(
                "a completed gate requires observed_value, threshold, and an evidence reference"
            )
        if self.status == "INCOMPLETE" and not self.reason_code:
            raise ValueError("an incomplete gate requires a reason_code")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GateEvaluation":
        return cls(
            contract_id=data["contract_id"],
            rule_id=data["rule_id"],
            status=data["status"],
            applicable=bool(data.get("applicable", False)),
            observed_value=data.get("observed_value"),
            threshold=data.get("threshold"),
            evidence_ref_id=data.get("evidence_ref_id"),
            evidence_url=data.get("evidence_url"),
            reason_code=data.get("reason_code"),
            notes=list(data.get("notes", [])),
        )


@dataclass
class ScreenResult:
    """Deterministic screen state, separate from company underwriting."""

    contract_id: str
    status: str
    gate_evaluations: list[GateEvaluation] = field(default_factory=list)
    required_rule_ids: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.contract_id = str(self.contract_id or "").strip()
        self.status = str(self.status or "").strip().upper()
        if not self.contract_id:
            raise ValueError("screen result contract_id is required")
        if self.status not in V2_SCREEN_STATUSES:
            raise ValueError(f"invalid screen result status: {self.status}")
        self.reason_codes = list(
            dict.fromkeys(
                str(item).strip().upper() for item in self.reason_codes if str(item).strip()
            )
        )
        self.required_rule_ids = list(
            dict.fromkeys(
                str(item).strip().upper() for item in self.required_rule_ids if str(item).strip()
            )
        )
        derived_refs = [
            item.evidence_ref_id
            for item in self.gate_evaluations
            if item.evidence_ref_id is not None
        ]
        self.evidence_ref_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in [*self.evidence_ref_ids, *derived_refs]
                if str(item).strip()
            )
        )
        if any(item.contract_id != self.contract_id for item in self.gate_evaluations):
            raise ValueError("screen result gate contract IDs must match the screen contract")
        evaluated_rule_ids = [item.rule_id for item in self.gate_evaluations]
        if len(evaluated_rule_ids) != len(set(evaluated_rule_ids)):
            raise ValueError("screen result must evaluate each gate rule exactly once")
        if self.required_rule_ids and set(evaluated_rule_ids) != set(self.required_rule_ids):
            raise ValueError("screen result must cover every required gate rule exactly once")
        gate_statuses = {item.status for item in self.gate_evaluations}
        if self.status == "FAIL" and "FAIL" not in gate_statuses:
            raise ValueError("failed screen result requires a failed gate evaluation")
        if self.status == "INCOMPLETE" and "INCOMPLETE" not in gate_statuses:
            raise ValueError("incomplete screen result requires an incomplete gate evaluation")
        if self.status == "PASS" and gate_statuses & {"FAIL", "INCOMPLETE"}:
            raise ValueError("passing screen result cannot contain failed or incomplete gates")
        if self.status == "PASS" and not self.gate_evaluations:
            raise ValueError("passing screen result requires recorded gate evaluations")
        if self.status == "NOT_RUN" and self.gate_evaluations:
            raise ValueError("NOT_RUN screen result cannot contain gate evaluations")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScreenResult":
        return cls(
            contract_id=data["contract_id"],
            status=data["status"],
            gate_evaluations=[
                GateEvaluation.from_dict(item) for item in data.get("gate_evaluations", [])
            ],
            required_rule_ids=list(data.get("required_rule_ids", [])),
            reason_codes=list(data.get("reason_codes", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
        )


@dataclass
class UnderwritingResult:
    """Company-review state; it never doubles as a deterministic screen."""

    status: str
    verdict: str | None = None
    confidence: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)
    tool_call_ids: list[str] = field(default_factory=list)
    child_run_id: str | None = None

    def __post_init__(self) -> None:
        self.status = str(self.status or "").strip().upper()
        self.verdict = str(self.verdict or "").strip().upper() or None
        self.confidence = str(self.confidence or "").strip().upper() or None
        self.child_run_id = str(self.child_run_id or "").strip() or None
        self.reason_codes = list(
            dict.fromkeys(
                str(item).strip().upper() for item in self.reason_codes if str(item).strip()
            )
        )
        self.evidence_ref_ids = list(
            dict.fromkeys(str(item).strip() for item in self.evidence_ref_ids if str(item).strip())
        )
        self.tool_call_ids = list(
            dict.fromkeys(str(item).strip() for item in self.tool_call_ids if str(item).strip())
        )
        if self.status not in V2_UNDERWRITING_STATUSES:
            raise ValueError(f"invalid underwriting result status: {self.status}")
        if self.status in {"NOT_REQUIRED", "NOT_STARTED", "INCOMPLETE", "FAILED"} and (
            self.verdict is not None
        ):
            raise ValueError(f"{self.status} underwriting cannot carry a verdict")
        if self.status in {"INCOMPLETE", "FAILED"} and not self.reason_codes:
            raise ValueError(f"{self.status.lower()} underwriting requires a reason_code")
        if self.status == "COMPLETED" and (
            self.verdict not in V2_COMPLETED_UNDERWRITING_VERDICTS
            or not self.child_run_id
            or not self.evidence_ref_ids
            or not self.tool_call_ids
        ):
            raise ValueError(
                "completed underwriting requires an investable verdict, child run, "
                "decision-linked evidence, and successful tool calls"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnderwritingResult":
        return cls(
            status=data["status"],
            verdict=data.get("verdict"),
            confidence=data.get("confidence"),
            reason_codes=list(data.get("reason_codes", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
            tool_call_ids=list(data.get("tool_call_ids", [])),
            child_run_id=data.get("child_run_id"),
        )


@dataclass
class CandidateDisposition:
    """Truthful terminal state for one security discovered by a v2 scan."""

    ticker: str
    terminal_state: str
    scope_status: str
    screen_status: str
    review_status: str
    underwriting_verdict: str | None = None
    underwriting_confidence: str | None = None
    watchlist_eligible: bool = False
    issuer_key: str | None = None
    issuer_cik: str | None = None
    primary_ticker: str | None = None
    security_type: str | None = None
    is_secondary_class: bool | None = None
    is_adr: bool | None = None
    adr_ratio: float | None = None
    share_class_ratio: float | None = None
    identity_source_url: str | None = None
    ratio_source_url: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)
    last_completed_stage: str | None = None
    frontier_status: str | None = None
    frontier_dominated_by: list[str] = field(default_factory=list)
    screen_result: ScreenResult | None = None
    underwriting_result: UnderwritingResult | None = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.strip().upper()
        if self.underwriting_verdict is not None:
            self.underwriting_verdict = self.underwriting_verdict.strip().upper() or None
        if self.underwriting_confidence is not None:
            self.underwriting_confidence = self.underwriting_confidence.strip().upper() or None
        if isinstance(self.screen_result, dict):
            self.screen_result = ScreenResult.from_dict(self.screen_result)
        if isinstance(self.underwriting_result, dict):
            self.underwriting_result = UnderwritingResult.from_dict(self.underwriting_result)
        if not self.ticker:
            raise ValueError("candidate disposition ticker is required")
        if self.terminal_state not in V2_TERMINAL_STATES:
            raise ValueError(f"invalid candidate terminal_state: {self.terminal_state}")
        if self.scope_status not in V2_SCOPE_STATUSES:
            raise ValueError(f"invalid candidate scope_status: {self.scope_status}")
        if self.screen_status not in V2_SCREEN_STATUSES:
            raise ValueError(f"invalid candidate screen_status: {self.screen_status}")
        if self.review_status not in V2_REVIEW_STATUSES:
            raise ValueError(f"invalid candidate review_status: {self.review_status}")
        if self.frontier_status is not None:
            self.frontier_status = str(self.frontier_status).strip().upper() or None
        if self.frontier_status is not None and self.frontier_status not in V2_FRONTIER_STATUSES:
            raise ValueError(f"invalid candidate frontier_status: {self.frontier_status}")
        self.frontier_dominated_by = list(
            dict.fromkeys(
                str(item).strip().upper()
                for item in self.frontier_dominated_by
                if str(item).strip()
            )
        )
        if self.frontier_status == "DOMINATED" and not self.frontier_dominated_by:
            raise ValueError("DOMINATED frontier status requires at least one dominator")
        if self.frontier_status != "DOMINATED" and self.frontier_dominated_by:
            raise ValueError("frontier_dominated_by is only legal for DOMINATED candidates")
        if self.screen_result is not None and self.screen_result.status != self.screen_status:
            raise ValueError("candidate screen_status must match screen_result.status")
        underwriting_status_matches = self.underwriting_result is None or (
            self.underwriting_result.status == self.review_status
        )
        if self.underwriting_result is not None and (
            not underwriting_status_matches
            or self.underwriting_result.verdict != self.underwriting_verdict
        ):
            raise ValueError("candidate review status/verdict must match underwriting_result")
        if self.terminal_state == "OUT_OF_SCOPE" and self.scope_status != "OUT_OF_SCOPE":
            raise ValueError("OUT_OF_SCOPE disposition requires scope_status=OUT_OF_SCOPE")
        if self.terminal_state != "OUT_OF_SCOPE" and self.scope_status != "IN_SCOPE":
            raise ValueError("an admitted disposition requires scope_status=IN_SCOPE")
        if self.terminal_state == "OUT_OF_SCOPE" and (
            self.screen_status != "NOT_RUN"
            or self.review_status != "NOT_REQUIRED"
            or self.underwriting_verdict is not None
            or self.watchlist_eligible
            or not self.reason_codes
        ):
            raise ValueError("OUT_OF_SCOPE disposition has incompatible pipeline state")
        if self.terminal_state == "SCREENED_OUT" and (
            self.screen_status != "FAIL"
            or self.review_status != "NOT_REQUIRED"
            or self.underwriting_verdict is not None
            or self.watchlist_eligible
            or not self.reason_codes
        ):
            raise ValueError("SCREENED_OUT disposition has incompatible pipeline state")
        if self.terminal_state == "SCREENED_OUT" and (
            self.screen_result is None or self.screen_result.status != "FAIL"
        ):
            raise ValueError("SCREENED_OUT requires a source-backed failed ScreenResult")
        if self.terminal_state == "DEFERRED_BY_BOUND" and (
            self.scope_status != "IN_SCOPE"
            or self.screen_status != "NOT_RUN"
            or self.review_status != "NOT_REQUIRED"
            or self.underwriting_verdict is not None
            or self.watchlist_eligible
            or self.frontier_status != "NOT_ELIGIBLE"
            or "DEFERRED_BY_EXECUTION_BOUND" not in self.reason_codes
        ):
            raise ValueError("DEFERRED_BY_BOUND disposition has incompatible pipeline state")
        if self.terminal_state == "READY_FOR_UNDERWRITING" and (
            self.screen_status != "PASS"
            or self.review_status != "NOT_STARTED"
            or self.underwriting_verdict is not None
            or not self.watchlist_eligible
        ):
            raise ValueError("READY_FOR_UNDERWRITING disposition has incompatible pipeline state")
        if self.terminal_state == "NEEDS_DATA":
            valid_review_state = (
                (self.review_status == "NOT_STARTED" and self.underwriting_verdict is None)
                or (self.review_status == "INCOMPLETE" and self.underwriting_verdict is None)
                or (self.review_status == "FAILED" and self.underwriting_verdict is None)
                or (
                    self.review_status == "COMPLETED"
                    and self.underwriting_verdict == "DATA_INCOMPLETE"
                )
            )
            valid_screen_state = (
                self.review_status == "NOT_STARTED" and self.screen_status in {"PASS", "INCOMPLETE"}
            ) or (
                self.review_status in {"INCOMPLETE", "FAILED", "COMPLETED"}
                and self.screen_status in {"PASS", "INCOMPLETE"}
            )
            if not valid_screen_state or not valid_review_state or not self.reason_codes:
                raise ValueError("NEEDS_DATA disposition has incompatible pipeline state")
        if self.terminal_state == "UNDERWRITTEN":
            allowed_verdicts = {
                "ACTIONABLE",
                "WATCHLIST_ONLY",
                "AVOID",
            }
            if (
                self.screen_status != "PASS"
                or self.review_status != "COMPLETED"
                or self.underwriting_verdict not in allowed_verdicts
                or self.watchlist_eligible
                != (self.underwriting_verdict in {"ACTIONABLE", "WATCHLIST_ONLY"})
            ):
                raise ValueError("UNDERWRITTEN disposition has incompatible pipeline state")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateDisposition":
        return cls(
            ticker=data["ticker"],
            terminal_state=data["terminal_state"],
            scope_status=data["scope_status"],
            screen_status=data["screen_status"],
            review_status=data["review_status"],
            underwriting_verdict=data.get("underwriting_verdict"),
            underwriting_confidence=data.get("underwriting_confidence"),
            watchlist_eligible=bool(data.get("watchlist_eligible", False)),
            issuer_key=data.get("issuer_key"),
            issuer_cik=data.get("issuer_cik"),
            primary_ticker=data.get("primary_ticker"),
            security_type=data.get("security_type"),
            is_secondary_class=(
                bool(data["is_secondary_class"])
                if isinstance(data.get("is_secondary_class"), bool)
                else None
            ),
            is_adr=(bool(data["is_adr"]) if isinstance(data.get("is_adr"), bool) else None),
            adr_ratio=(
                float(data["adr_ratio"])
                if isinstance(data.get("adr_ratio"), (int, float))
                and not isinstance(data.get("adr_ratio"), bool)
                else None
            ),
            share_class_ratio=(
                float(data["share_class_ratio"])
                if isinstance(data.get("share_class_ratio"), (int, float))
                and not isinstance(data.get("share_class_ratio"), bool)
                else None
            ),
            identity_source_url=data.get("identity_source_url"),
            ratio_source_url=data.get("ratio_source_url"),
            reason_codes=list(data.get("reason_codes", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
            last_completed_stage=data.get("last_completed_stage"),
            frontier_status=data.get("frontier_status"),
            frontier_dominated_by=list(data.get("frontier_dominated_by", [])),
            screen_result=(
                ScreenResult.from_dict(data["screen_result"])
                if isinstance(data.get("screen_result"), dict)
                else None
            ),
            underwriting_result=(
                UnderwritingResult.from_dict(data["underwriting_result"])
                if isinstance(data.get("underwriting_result"), dict)
                else None
            ),
        )


@dataclass
class SectorSelectionValidation:
    """Independent validation state for a provisional selected company."""

    status: str
    selected_ticker: str | None = None
    validator_run_id: str | None = None
    validator_verdict: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    evidence_ref_ids: list[str] = field(default_factory=list)
    evidence: list[EvidenceReference] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    provider_usage: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_binding: dict[str, Any] = field(default_factory=dict)
    terminal_ledger_fingerprint: str | None = None

    def _terminal_ledger_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_ticker": self.selected_ticker,
            "validator_run_id": self.validator_run_id,
            "validator_verdict": self.validator_verdict,
            "reason_codes": list(self.reason_codes),
            "evidence_ref_ids": list(self.evidence_ref_ids),
            "evidence": [item.to_dict() for item in self.evidence],
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "provider_usage": [dict(item) for item in self.provider_usage],
            "notes": list(self.notes),
            "source_binding": dict(self.source_binding),
        }

    def __post_init__(self) -> None:
        self.source_binding = dict(self.source_binding or {})
        if self.status not in V2_SELECTION_VALIDATION_STATUSES:
            raise ValueError(f"invalid selection validation status: {self.status}")
        if self.selected_ticker is not None:
            self.selected_ticker = self.selected_ticker.strip().upper() or None
        if self.validator_run_id is not None:
            self.validator_run_id = self.validator_run_id.strip() or None
        if self.validator_verdict is not None:
            self.validator_verdict = self.validator_verdict.strip().upper() or None
        if self.status in {"VALIDATED", "CONTRADICTED"} and not self.selected_ticker:
            raise ValueError(f"{self.status} validation requires selected_ticker")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("selection validation evidence IDs must be unique")
        tool_call_ids = [item.call_id for item in self.tool_calls]
        if len(tool_call_ids) != len(set(tool_call_ids)):
            raise ValueError("selection validation tool call IDs must be unique")
        unknown_refs = set(self.evidence_ref_ids) - set(evidence_ids)
        if unknown_refs:
            raise ValueError("selection validation contains dangling evidence references")
        tool_unknown_refs = {
            ref
            for call in self.tool_calls
            for ref in call.evidence_ref_ids
            if ref not in set(evidence_ids)
        }
        if tool_unknown_refs:
            raise ValueError("selection validation tool calls contain dangling evidence references")
        if self.status == "VALIDATED":
            evidence_by_id = {item.evidence_id: item for item in self.evidence}
            tool_calls_by_id = {item.call_id: item for item in self.tool_calls}
            selected_ticker = str(self.selected_ticker or "").strip().upper()
            has_successful_challenge = bool(self.evidence_ref_ids) and all(
                bool(evidence_by_id[ref].tool_call_id)
                and evidence_by_id[ref].tool_call_id in tool_calls_by_id
                and str(evidence_by_id[ref].ticker or "").strip().upper() == selected_ticker
                and str(evidence_by_id[ref].confidence or "").strip().upper()
                in {"MODERATE", "HIGH"}
                and tool_calls_by_id[str(evidence_by_id[ref].tool_call_id)].status == "OK"
                and tool_calls_by_id[str(evidence_by_id[ref].tool_call_id)].lane
                == "selected_company_validation"
                and ref in tool_calls_by_id[str(evidence_by_id[ref].tool_call_id)].evidence_ref_ids
                for ref in self.evidence_ref_ids
            )
            if (
                not self.validator_run_id
                or self.validator_verdict not in V2_AFFIRMATIVE_VALIDATOR_VERDICTS
                or not has_successful_challenge
            ):
                raise ValueError(
                    "VALIDATED selection requires an identified affirmative "
                    "validator and successful challenge evidence"
                )
        if self.status in {"VALIDATED", "CONTRADICTED"}:
            _validate_selection_source_binding(
                self.source_binding,
                selected_ticker=str(self.selected_ticker or ""),
            )
            namespace = f"{self.validator_run_id}:"
            if not self.validator_run_id:
                raise ValueError(f"{self.status} validation requires validator_run_id")
            if any(not item.call_id.startswith(namespace) for item in self.tool_calls):
                raise ValueError("selection validation tool IDs must be validator namespaced")
            if any(
                not item.evidence_id.startswith(namespace)
                or (item.tool_call_id is not None and not item.tool_call_id.startswith(namespace))
                for item in self.evidence
            ):
                raise ValueError("selection validation evidence IDs must be validator namespaced")
            if not self.provider_usage or any(
                str(row.get("validator_run_id") or "") != self.validator_run_id
                or not str(row.get("provider_call_id") or "").startswith(namespace)
                for row in self.provider_usage
            ):
                raise ValueError(
                    "completed selection validation requires validator-bound provider usage"
                )
            expected_terminal_fingerprint = selection_validation_terminal_ledger_fingerprint(
                self._terminal_ledger_payload()
            )
            if self.terminal_ledger_fingerprint is None:
                self.terminal_ledger_fingerprint = expected_terminal_fingerprint
            elif (
                not _is_sha256(self.terminal_ledger_fingerprint)
                or str(self.terminal_ledger_fingerprint).strip().lower()
                != expected_terminal_fingerprint
            ):
                raise ValueError(
                    "completed selection validation requires an exact terminal ledger fingerprint"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectorSelectionValidation":
        if (
            str(data.get("status") or "").strip().upper() in {"VALIDATED", "CONTRADICTED"}
            and "terminal_ledger_fingerprint" not in data
        ):
            raise ValueError("persisted completed validation requires terminal_ledger_fingerprint")
        return cls(
            status=data["status"],
            selected_ticker=data.get("selected_ticker"),
            validator_run_id=data.get("validator_run_id"),
            validator_verdict=data.get("validator_verdict"),
            reason_codes=list(data.get("reason_codes", [])),
            evidence_ref_ids=list(data.get("evidence_ref_ids", [])),
            evidence=[
                EvidenceReference.from_dict(item)
                for item in data.get("evidence", [])
                if isinstance(item, dict)
            ],
            tool_calls=[
                ToolCallRecord.from_dict(item)
                for item in data.get("tool_calls", [])
                if isinstance(item, dict)
            ],
            provider_usage=[
                dict(item) for item in data.get("provider_usage", []) if isinstance(item, dict)
            ],
            notes=list(data.get("notes", [])),
            source_binding=dict(data.get("source_binding", {})),
            terminal_ledger_fingerprint=data.get("terminal_ledger_fingerprint"),
        )


@dataclass
class AutonomousSectorFinancialRunArtifact:
    """Complete audit artifact for an autonomous sector financial run."""

    run_id: str
    sector: str
    market_cap_focus: str
    objective: str
    as_of_date: str
    created_at: str
    status: str
    final_verdict: str | None
    selected_ticker: str | None
    confidence: str | None
    scan_family: str = "normal"
    pipeline_version: str = SECTOR_PIPELINE_VERSION_V1
    execution_status: str | None = None
    decision_status: str | None = None
    admitted_tickers: tuple[str, ...] = field(default_factory=tuple)
    candidate_dispositions: list[CandidateDisposition] = field(default_factory=list)
    selection_validation: SectorSelectionValidation | None = None
    competitive_frontier: dict[str, Any] = field(default_factory=dict)
    lane_budget: dict[str, Any] = field(default_factory=dict)
    lane_usage: dict[str, Any] = field(default_factory=dict)
    provider_usage: list[dict[str, Any]] = field(default_factory=list)
    provisional_final_decision: SectorFinalDecision | None = None
    completed_at: str | None = None
    framework: SectorFinancialFramework | None = None
    candidate_selection: dict[str, Any] = field(default_factory=dict)
    company_packets: list[SectorCompanyFinancialPacket] = field(default_factory=list)
    research_questions: list[SectorResearchQuestion] = field(default_factory=list)
    expected_return_scenarios: list[SectorExpectedReturnScenario] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    evidence: list[EvidenceReference] = field(default_factory=list)
    belief_updates: list[BeliefUpdate] = field(default_factory=list)
    final_decision: SectorFinalDecision | None = None
    selection_audit: dict[str, Any] = field(default_factory=dict)
    framework_evidence_preflight: list[dict[str, Any]] = field(default_factory=list)
    final_decision_prompt_context: dict[str, Any] = field(default_factory=dict)
    audit_gap_repair_attempted: bool = False
    audit_gap_repair_status: str | None = None
    audit_gap_repair_notes: list[str] = field(default_factory=list)
    watchlist_resolution_attempted: bool = False
    watchlist_resolution_status: str | None = None
    watchlist_resolution_notes: list[str] = field(default_factory=list)
    no_selection_finalist_audit_attempted: bool = False
    no_selection_finalist_audit_status: str | None = None
    no_selection_finalist_audit_focus_ticker: str | None = None
    no_selection_finalist_audit_notes: list[str] = field(default_factory=list)
    no_selection_finalist_resolution_attempted: bool = False
    no_selection_finalist_resolution_status: str | None = None
    no_selection_finalist_resolution_notes: list[str] = field(default_factory=list)
    alternate_finalist_audit_attempted: bool = False
    alternate_finalist_audit_status: str | None = None
    alternate_finalist_audit_notes: list[str] = field(default_factory=list)
    alternate_finalist_audit_results: list[dict[str, Any]] = field(default_factory=list)
    company_autonomy_attempted: bool = False
    company_autonomy_status: str | None = None
    company_autonomy_notes: list[str] = field(default_factory=list)
    company_autonomy_runs: list[dict[str, Any]] = field(default_factory=list)
    company_autonomy_decision_trace: dict[str, Any] = field(default_factory=dict)
    relative_ranking: list[dict[str, Any]] = field(default_factory=list)
    memo_body: dict[str, Any] = field(default_factory=dict)
    no_selection_reason: str | None = None
    degraded_states: list[str] = field(default_factory=list)
    audit_notes: list[str] = field(default_factory=list)
    contract_version: str = SECTOR_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.pipeline_version == SECTOR_PIPELINE_VERSION_V2:
            self._validate_v2()

    def _validate_v2(self) -> None:
        if self.contract_version != SECTOR_CONTRACT_VERSION_V2:
            raise ValueError("v2 pipeline requires autonomous_sector_financial_run_v2 contract")
        if self.execution_status not in V2_EXECUTION_STATUSES:
            raise ValueError(f"invalid v2 execution_status: {self.execution_status}")
        if self.decision_status not in V2_DECISION_STATUSES:
            raise ValueError(f"invalid v2 decision_status: {self.decision_status}")
        if self.status != self.execution_status:
            raise ValueError("v2 legacy status must equal execution_status")
        if self.final_verdict not in V2_FINAL_VERDICTS | {None}:
            raise ValueError(f"invalid v2 final_verdict: {self.final_verdict}")

        admitted = [ticker.strip().upper() for ticker in self.admitted_tickers]
        if any(not ticker for ticker in admitted) or len(admitted) != len(set(admitted)):
            raise ValueError("v2 admitted_tickers must be non-empty unique ticker symbols")
        self.admitted_tickers = tuple(admitted)
        if self.selected_ticker is not None:
            self.selected_ticker = self.selected_ticker.strip().upper() or None

        dispositions = [item.ticker for item in self.candidate_dispositions]
        if len(dispositions) != len(set(dispositions)):
            raise ValueError("v2 candidate_dispositions must contain one row per ticker")
        in_scope = {
            item.ticker for item in self.candidate_dispositions if item.scope_status == "IN_SCOPE"
        }
        if in_scope != set(admitted):
            raise ValueError(
                "v2 admitted_tickers must exactly equal in-scope candidate dispositions"
            )
        discovered: list[str] = list(admitted)
        classifications = self.candidate_selection.get("cap_classifications")
        if isinstance(classifications, dict):
            discovered.extend(str(ticker).strip().upper() for ticker in classifications)
        for field_name in (
            "requested_tickers",
            "loaded_tickers",
            "excluded_tickers",
            "selected_tickers_before_financial_history_filter",
            "selected_tickers_before_framework_filter",
            "selected_tickers",
            "membership_tickers",
            "execution_tickers",
            "deferred_by_bound_tickers",
        ):
            values = self.candidate_selection.get(field_name)
            if isinstance(values, (list, tuple, set)):
                discovered.extend(
                    str(ticker).strip().upper() for ticker in values if str(ticker).strip()
                )
        discovered_set = set(discovered)
        if discovered_set and set(dispositions) != discovered_set:
            raise ValueError("v2 candidate_dispositions must reconcile every discovered security")
        if bool(self.candidate_selection.get("execution_bound_frozen")):
            membership = [
                str(ticker).strip().upper()
                for ticker in self.candidate_selection.get("membership_tickers") or []
                if str(ticker).strip()
            ]
            execution = [
                str(ticker).strip().upper()
                for ticker in self.candidate_selection.get("execution_tickers") or []
                if str(ticker).strip()
            ]
            deferred = [
                str(ticker).strip().upper()
                for ticker in self.candidate_selection.get("deferred_by_bound_tickers") or []
                if str(ticker).strip()
            ]
            if (
                self.candidate_selection.get("selected_tickers") != membership
                or [*execution, *deferred] != membership
                or set(execution) & set(deferred)
            ):
                raise ValueError("v2 execution-bound ledgers must partition membership")
            if self.candidate_selection.get(
                "membership_fingerprint"
            ) != _canonical_json_fingerprint(membership) or self.candidate_selection.get(
                "execution_fingerprint"
            ) != _canonical_json_fingerprint(execution):
                raise ValueError("v2 execution-bound ledger fingerprint mismatch")
            disposition_deferred = {
                disposition.ticker
                for disposition in self.candidate_dispositions
                if disposition.terminal_state == "DEFERRED_BY_BOUND"
            }
            if disposition_deferred != set(deferred):
                raise ValueError(
                    "v2 DEFERRED_BY_BOUND dispositions must match the frozen deferred ledger"
                )

        if self.execution_status == "FAILED":
            if self.decision_status != "INCOMPLETE":
                raise ValueError("failed v2 execution requires an incomplete decision")
            if self.final_verdict is not None or self.selected_ticker is not None:
                raise ValueError("failed v2 execution cannot publish a final verdict")

        if self.decision_status == "INCOMPLETE" and self.final_verdict is not None:
            raise ValueError("incomplete v2 decision cannot publish a final verdict")
        if self.decision_status == "COMPLETE" and self.final_verdict is None:
            raise ValueError("complete v2 decision requires a final verdict")
        if self.decision_status == "COMPLETE" and _v2_blocking_run_states(self.degraded_states):
            raise ValueError("complete v2 decision cannot carry unresolved run failures")
        if self.decision_status == "COMPLETE":
            eligible_dispositions = [
                item
                for item in self.candidate_dispositions
                if item.scope_status == "IN_SCOPE"
                and item.screen_status == "PASS"
                and item.terminal_state in {"READY_FOR_UNDERWRITING", "UNDERWRITTEN"}
            ]
            if eligible_dispositions:
                from app.autonomous.competitive_frontier import (
                    validate_closed_competitive_frontier_payload,
                )

                expected_candidates = [item.ticker for item in eligible_dispositions]
                expected_reviewed = [
                    item.ticker
                    for item in eligible_dispositions
                    if item.review_status == "COMPLETED"
                ]
                try:
                    expected_candidate_set = set(expected_candidates)
                    frontier_state = validate_closed_competitive_frontier_payload(
                        self.competitive_frontier,
                        expected_candidate_tickers=expected_candidates,
                        expected_reviewed_tickers=expected_reviewed,
                        source_company_packets=[
                            packet
                            for packet in self.company_packets
                            if packet.ticker.strip().upper() in expected_candidate_set
                        ],
                        source_scenarios=[
                            scenario
                            for scenario in self.expected_return_scenarios
                            if scenario.ticker.strip().upper() in expected_candidate_set
                        ],
                    )
                except ValueError as exc:
                    raise ValueError(
                        "complete v2 decision requires a structurally valid closed "
                        f"competitive frontier: {exc}"
                    ) from exc

                reviewed = set(frontier_state.reviewed_tickers)
                dominated = set(frontier_state.dominated_tickers)
                dominators = frontier_state.dominators_by_ticker
                for disposition in eligible_dispositions:
                    if disposition.ticker in reviewed:
                        if disposition.frontier_status != "REVIEWED":
                            raise ValueError(
                                "reviewed competitive-frontier candidates must carry "
                                "frontier_status=REVIEWED"
                            )
                        continue
                    if disposition.ticker not in dominated:
                        raise ValueError(
                            "closed competitive frontier contains an unreviewed live candidate"
                        )
                    if disposition.frontier_status != "DOMINATED" or set(
                        disposition.frontier_dominated_by
                    ) != set(dominators.get(disposition.ticker, ())):
                        raise ValueError(
                            "dominated candidate disposition does not match competitive frontier"
                        )
            unresolved_candidates = [
                item.ticker
                for item in self.candidate_dispositions
                if item.scope_status == "IN_SCOPE"
                and (
                    item.terminal_state in {"NEEDS_DATA", "DEFERRED_BY_BOUND"}
                    or (
                        item.terminal_state == "READY_FOR_UNDERWRITING"
                        and item.frontier_status != "DOMINATED"
                    )
                )
            ]
            if unresolved_candidates:
                raise ValueError("complete v2 decision cannot carry unresolved admitted candidates")
            dominated_unreviewed = [
                item.ticker
                for item in self.candidate_dispositions
                if item.scope_status == "IN_SCOPE"
                and item.terminal_state == "READY_FOR_UNDERWRITING"
                and item.frontier_status == "DOMINATED"
            ]
            if dominated_unreviewed:
                frontier = self.competitive_frontier
                if (
                    not isinstance(frontier, dict)
                    or str(frontier.get("status") or "").upper() != "CLOSED"
                    or frontier.get("unresolved_tickers")
                    or not set(dominated_unreviewed).issubset(
                        {
                            str(item).strip().upper()
                            for item in frontier.get("dominated_tickers") or []
                        }
                    )
                ):
                    raise ValueError(
                        "dominated unreviewed candidates require a closed competitive frontier"
                    )

        self._validate_v2_signal_snapshot_bindings()

        validation = self.selection_validation
        if validation is not None and validation.status in {"VALIDATED", "CONTRADICTED"}:
            selected = str(validation.selected_ticker or "").strip().upper()
            underwriting_child_run_ids = {
                str(item.underwriting_result.child_run_id or "").strip()
                for item in self.candidate_dispositions
                if item.underwriting_result is not None
                and str(item.underwriting_result.child_run_id or "").strip()
            }
            underwriting_child_run_ids.update(
                str(run.get("run_id") or "").strip()
                for run in self.company_autonomy_runs
                if isinstance(run, Mapping) and str(run.get("run_id") or "").strip()
            )
            if validation.validator_run_id in underwriting_child_run_ids:
                raise ValueError(
                    "selection validator_run_id must differ from every underwriting child_run_id"
                )
            source_bindings = self.competitive_frontier.get("source_bindings")
            expected_binding = (
                source_bindings.get(selected) if isinstance(source_bindings, dict) else None
            )
            if (
                not isinstance(expected_binding, dict)
                or validation.source_binding != expected_binding
            ):
                raise ValueError(
                    "selection validation source_binding must exactly match the selected "
                    "competitive-frontier source binding"
                )
            if (
                str(expected_binding.get("sector") or "").strip().lower()
                != str(self.sector or "").strip().lower()
                or str(expected_binding.get("as_of_date") or "") != self.as_of_date
            ):
                raise ValueError(
                    "selection validation source_binding must match the sector artifact identity"
                )
            expected_cohort = [
                str(item.ticker or "").strip().upper()
                for item in self.company_packets
                if str(item.ticker or "").strip()
            ]
            if expected_binding.get("cohort_tickers") != expected_cohort:
                raise ValueError(
                    "selection validation source_binding cohort must match the complete "
                    "company-packet rank order"
                )
            frontier_candidates = self.competitive_frontier.get("candidates")
            expected_frontier = [
                str(item.get("ticker") or "").strip().upper()
                for item in frontier_candidates or []
                if isinstance(item, dict) and str(item.get("ticker") or "").strip()
            ]
            if expected_binding.get("frontier_candidate_tickers") != expected_frontier:
                raise ValueError(
                    "selection validation source_binding frontier candidates must match "
                    "frontier rank order"
                )
            rebuilt_bindings = build_v2_canonical_child_source_bindings(
                sector=self.sector,
                as_of_date=self.as_of_date,
                company_packets=self.company_packets,
                scenarios=self.expected_return_scenarios,
                signal_packet_snapshots=dict(
                    self.competitive_frontier.get("signal_packet_snapshots") or {}
                ),
                frontier_candidate_tickers=expected_frontier,
            )
            if rebuilt_bindings.get(selected) != expected_binding:
                raise ValueError("selection validation source_binding cohort fingerprint mismatch")
            packet = next(
                (
                    item
                    for item in self.company_packets
                    if str(item.ticker or "").strip().upper() == selected
                ),
                None,
            )
            if packet is None or expected_binding.get(
                "company_packet_fingerprint"
            ) != _canonical_json_fingerprint(packet.to_dict()):
                raise ValueError(
                    "selection validation source_binding company packet fingerprint mismatch"
                )

        if self.final_verdict == "NO_SELECTION":
            if self.decision_status != "COMPLETE" or self.selected_ticker is not None:
                raise ValueError("NO_SELECTION requires a complete decision and no ticker")
            unresolved_or_positive = [
                item.ticker
                for item in self.candidate_dispositions
                if item.scope_status == "IN_SCOPE"
                and not (
                    item.terminal_state == "SCREENED_OUT"
                    or (
                        item.terminal_state == "READY_FOR_UNDERWRITING"
                        and item.frontier_status == "DOMINATED"
                    )
                    or (
                        item.terminal_state == "UNDERWRITTEN"
                        and item.underwriting_verdict == "AVOID"
                    )
                )
            ]
            if unresolved_or_positive:
                raise ValueError(
                    "NO_SELECTION requires every admitted candidate to be "
                    "decisively screened out or underwritten negative"
                )
        if self.final_verdict is None and self.selected_ticker is not None:
            raise ValueError("a v2 selected_ticker requires a final verdict")
        if self.final_verdict in {"SELECTED", "WATCHLIST"} and not self.selected_ticker:
            raise ValueError(f"{self.final_verdict} requires selected_ticker")
        if self.final_verdict in {"SELECTED", "WATCHLIST"} and self.selected_ticker not in admitted:
            raise ValueError(f"{self.final_verdict} ticker must be admitted")
        if self.final_verdict == "WATCHLIST":
            disposition = next(
                (
                    item
                    for item in self.candidate_dispositions
                    if item.ticker == self.selected_ticker
                ),
                None,
            )
            if (
                disposition is None
                or disposition.terminal_state != "UNDERWRITTEN"
                or disposition.underwriting_verdict != "WATCHLIST_ONLY"
            ):
                raise ValueError("WATCHLIST requires completed WATCHLIST_ONLY underwriting")
            if any(
                item.scope_status == "IN_SCOPE"
                and item.terminal_state == "UNDERWRITTEN"
                and item.underwriting_verdict == "ACTIONABLE"
                for item in self.candidate_dispositions
            ):
                raise ValueError("WATCHLIST cannot be final while an actionable competitor remains")
        if self.final_verdict == "SELECTED":
            self._validate_v2_selected(admitted)
        self._validate_v2_underwriting_source_bindings()

    def _validate_v2_signal_snapshot_bindings(self) -> None:
        """Rebuild every frontier binding from persisted cohort input snapshots."""

        candidates = self.competitive_frontier.get("candidates")
        source_bindings = self.competitive_frontier.get("source_bindings")
        if not candidates and not source_bindings:
            return
        if not isinstance(candidates, list) or not isinstance(source_bindings, dict):
            raise ValueError("v2 competitive frontier requires candidates and source bindings")
        cohort_tickers = [
            str(packet.ticker or "").strip().upper() for packet in self.company_packets
        ]
        frontier_tickers = [
            str(row.get("ticker") or "").strip().upper()
            for row in candidates
            if isinstance(row, dict)
        ]
        raw_snapshots = self.competitive_frontier.get("signal_packet_snapshots")
        if not isinstance(raw_snapshots, dict):
            raise ValueError("v2 frontier requires persisted signal_packet_snapshots")
        normalized_snapshots = {
            str(ticker).strip().upper(): canonical_v2_signal_packet_snapshot(snapshot)
            for ticker, snapshot in raw_snapshots.items()
            if isinstance(snapshot, Mapping)
        }
        if set(normalized_snapshots) != set(cohort_tickers):
            raise ValueError(
                "v2 signal_packet_snapshots must cover the complete company-packet cohort"
            )
        rebuilt = build_v2_canonical_child_source_bindings(
            sector=self.sector,
            as_of_date=self.as_of_date,
            company_packets=self.company_packets,
            scenarios=self.expected_return_scenarios,
            signal_packet_snapshots=normalized_snapshots,
            frontier_candidate_tickers=frontier_tickers,
        )
        if source_bindings != rebuilt:
            raise ValueError(
                "v2 frontier source bindings must derive from persisted signal snapshots"
            )

    def _validate_v2_underwriting_source_bindings(self) -> None:
        """Bind every completed underwriting result to its canonical child inputs."""

        source_bindings = self.competitive_frontier.get("source_bindings")
        if not isinstance(source_bindings, dict):
            source_bindings = {}
        expected_cohort = [
            str(item.ticker or "").strip().upper()
            for item in self.company_packets
            if str(item.ticker or "").strip()
        ]
        frontier_candidates = self.competitive_frontier.get("candidates")
        expected_frontier = [
            str(item.get("ticker") or "").strip().upper()
            for item in frontier_candidates or []
            if isinstance(item, dict) and str(item.get("ticker") or "").strip()
        ]
        packets_by_ticker = {
            str(item.ticker or "").strip().upper(): item
            for item in self.company_packets
            if str(item.ticker or "").strip()
        }

        for disposition in self.candidate_dispositions:
            if disposition.terminal_state != "UNDERWRITTEN":
                continue
            ticker = disposition.ticker
            underwriting = disposition.underwriting_result
            if underwriting is None or underwriting.status != "COMPLETED":
                raise ValueError(
                    f"UNDERWRITTEN {ticker} requires a structured completed underwriting_result"
                )
            expected_binding = source_bindings.get(ticker)
            if not isinstance(expected_binding, dict):
                raise ValueError(
                    f"UNDERWRITTEN {ticker} requires a competitive-frontier source_binding"
                )
            _validate_selection_source_binding(expected_binding, selected_ticker=ticker)
            if (
                str(expected_binding.get("sector") or "").strip().lower()
                != str(self.sector or "").strip().lower()
                or str(expected_binding.get("as_of_date") or "") != self.as_of_date
                or expected_binding.get("cohort_tickers") != expected_cohort
                or expected_binding.get("frontier_candidate_tickers") != expected_frontier
            ):
                raise ValueError(
                    f"UNDERWRITTEN {ticker} source_binding does not match the sector cohort"
                )
            packet = packets_by_ticker.get(ticker)
            if packet is None or expected_binding.get(
                "company_packet_fingerprint"
            ) != _canonical_json_fingerprint(packet.to_dict()):
                raise ValueError(
                    f"UNDERWRITTEN {ticker} source_binding company packet fingerprint mismatch"
                )
            rebuilt_bindings = build_v2_canonical_child_source_bindings(
                sector=self.sector,
                as_of_date=self.as_of_date,
                company_packets=self.company_packets,
                scenarios=self.expected_return_scenarios,
                signal_packet_snapshots=dict(
                    self.competitive_frontier.get("signal_packet_snapshots") or {}
                ),
                frontier_candidate_tickers=expected_frontier,
            )
            if rebuilt_bindings.get(ticker) != expected_binding:
                raise ValueError(
                    f"UNDERWRITTEN {ticker} source_binding cohort fingerprint mismatch"
                )

            matching_runs = [
                run
                for run in self.company_autonomy_runs
                if isinstance(run, dict)
                and str(run.get("ticker") or "").strip().upper() == ticker
                and str(run.get("run_id") or "").strip() == underwriting.child_run_id
            ]
            if len(matching_runs) != 1:
                raise ValueError(
                    f"UNDERWRITTEN {ticker} must bind to exactly one company child run"
                )
            run = matching_runs[0]
            if run.get("source_binding") != expected_binding:
                raise ValueError(
                    f"UNDERWRITTEN {ticker} child source_binding must match the competitive frontier"
                )

            raw_artifact = run.get("artifact")
            attempts = run.get("attempts")
            if not isinstance(raw_artifact, dict) or not isinstance(attempts, list) or not attempts:
                raise ValueError(
                    f"UNDERWRITTEN {ticker} child run requires artifact and attempt ledgers"
                )
            if not isinstance(attempts[-1], dict) or raw_artifact != attempts[-1]:
                raise ValueError(
                    f"UNDERWRITTEN {ticker} terminal attempt must match child artifact ledger"
                )
            nested_artifacts = [raw_artifact, *attempts]
            for nested in nested_artifacts:
                request = nested.get("request")
                candidate_scope = (
                    request.get("candidate_scope") if isinstance(request, dict) else None
                )
                nested_binding = (
                    candidate_scope.get("source_binding")
                    if isinstance(candidate_scope, dict)
                    else None
                )
                nested_snapshot = (
                    candidate_scope.get("signal_packet_snapshot")
                    if isinstance(candidate_scope, dict)
                    else None
                )
                if nested_binding != expected_binding:
                    raise ValueError(
                        f"UNDERWRITTEN {ticker} nested child source_binding must match the competitive frontier"
                    )
                if (
                    not isinstance(request, dict)
                    or str(request.get("run_id") or "").strip() != underwriting.child_run_id
                    or str(request.get("as_of_date") or "") != self.as_of_date
                    or not isinstance(candidate_scope, dict)
                    or str(candidate_scope.get("mode") or "") != "single_candidate"
                    or [str(item).strip().upper() for item in candidate_scope.get("tickers") or []]
                    != [ticker]
                    or nested_snapshot
                    != _mapping_signal_snapshot(
                        self.competitive_frontier.get("signal_packet_snapshots"),
                        ticker,
                    )
                ):
                    raise ValueError(f"UNDERWRITTEN {ticker} nested request identity mismatch")

    def _validate_v2_selected(self, admitted: list[str]) -> None:
        if self.decision_status != "COMPLETE":
            raise ValueError("SELECTED requires decision_status=COMPLETE")
        selected = (self.selected_ticker or "").strip().upper()
        if selected not in admitted:
            raise ValueError("SELECTED ticker must be admitted")
        disposition = next(
            (item for item in self.candidate_dispositions if item.ticker == selected),
            None,
        )
        if disposition is None or disposition.terminal_state != "UNDERWRITTEN":
            raise ValueError("SELECTED requires completed underwriting")
        if disposition.underwriting_verdict != "ACTIONABLE":
            raise ValueError("SELECTED requires ACTIONABLE underwriting")
        validation = self.selection_validation
        if (
            validation is None
            or validation.status != "VALIDATED"
            or validation.selected_ticker != selected
        ):
            raise ValueError("SELECTED requires matching VALIDATED selection validation")
        self.selected_ticker = selected

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["admitted_tickers"] = list(self.admitted_tickers)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutonomousSectorFinancialRunArtifact":
        framework_data = data.get("framework")
        decision_data = data.get("final_decision")
        provisional_decision_data = data.get("provisional_final_decision")
        validation_data = data.get("selection_validation")
        return cls(
            run_id=data["run_id"],
            sector=data["sector"],
            market_cap_focus=data["market_cap_focus"],
            objective=data["objective"],
            as_of_date=data["as_of_date"],
            created_at=data["created_at"],
            status=data["status"],
            final_verdict=data.get("final_verdict"),
            selected_ticker=data.get("selected_ticker"),
            confidence=data.get("confidence"),
            scan_family=data.get("scan_family", "normal"),
            pipeline_version=data.get("pipeline_version", SECTOR_PIPELINE_VERSION_V1),
            execution_status=data.get("execution_status"),
            decision_status=data.get("decision_status"),
            admitted_tickers=tuple(data.get("admitted_tickers", [])),
            candidate_dispositions=[
                CandidateDisposition.from_dict(item)
                for item in data.get("candidate_dispositions", [])
            ],
            selection_validation=(
                SectorSelectionValidation.from_dict(validation_data)
                if isinstance(validation_data, dict)
                else None
            ),
            competitive_frontier=dict(data.get("competitive_frontier", {})),
            lane_budget=dict(data.get("lane_budget", {})),
            lane_usage=dict(data.get("lane_usage", {})),
            provider_usage=[
                dict(item) for item in data.get("provider_usage", []) if isinstance(item, dict)
            ],
            provisional_final_decision=(
                SectorFinalDecision.from_dict(provisional_decision_data)
                if isinstance(provisional_decision_data, dict)
                else None
            ),
            completed_at=data.get("completed_at"),
            framework=SectorFinancialFramework.from_dict(framework_data)
            if isinstance(framework_data, dict)
            else None,
            candidate_selection=dict(data.get("candidate_selection", {})),
            company_packets=[
                SectorCompanyFinancialPacket.from_dict(item)
                for item in data.get("company_packets", [])
            ],
            research_questions=[
                SectorResearchQuestion.from_dict(item)
                for item in data.get("research_questions", [])
            ],
            expected_return_scenarios=[
                SectorExpectedReturnScenario.from_dict(item)
                for item in data.get("expected_return_scenarios", [])
            ],
            tool_calls=[ToolCallRecord.from_dict(item) for item in data.get("tool_calls", [])],
            evidence=[EvidenceReference.from_dict(item) for item in data.get("evidence", [])],
            belief_updates=[
                BeliefUpdate.from_dict(item) for item in data.get("belief_updates", [])
            ],
            final_decision=SectorFinalDecision.from_dict(decision_data)
            if isinstance(decision_data, dict)
            else None,
            selection_audit=dict(data.get("selection_audit", {})),
            framework_evidence_preflight=[
                dict(item) for item in data.get("framework_evidence_preflight", [])
            ],
            final_decision_prompt_context=dict(data.get("final_decision_prompt_context", {})),
            audit_gap_repair_attempted=bool(data.get("audit_gap_repair_attempted", False)),
            audit_gap_repair_status=data.get("audit_gap_repair_status"),
            audit_gap_repair_notes=list(data.get("audit_gap_repair_notes", [])),
            watchlist_resolution_attempted=bool(data.get("watchlist_resolution_attempted", False)),
            watchlist_resolution_status=data.get("watchlist_resolution_status"),
            watchlist_resolution_notes=list(data.get("watchlist_resolution_notes", [])),
            no_selection_finalist_audit_attempted=bool(
                data.get("no_selection_finalist_audit_attempted", False)
            ),
            no_selection_finalist_audit_status=data.get("no_selection_finalist_audit_status"),
            no_selection_finalist_audit_focus_ticker=data.get(
                "no_selection_finalist_audit_focus_ticker"
            ),
            no_selection_finalist_audit_notes=list(
                data.get("no_selection_finalist_audit_notes", [])
            ),
            no_selection_finalist_resolution_attempted=bool(
                data.get("no_selection_finalist_resolution_attempted", False)
            ),
            no_selection_finalist_resolution_status=data.get(
                "no_selection_finalist_resolution_status"
            ),
            no_selection_finalist_resolution_notes=list(
                data.get("no_selection_finalist_resolution_notes", [])
            ),
            alternate_finalist_audit_attempted=bool(
                data.get("alternate_finalist_audit_attempted", False)
            ),
            alternate_finalist_audit_status=data.get("alternate_finalist_audit_status"),
            alternate_finalist_audit_notes=list(data.get("alternate_finalist_audit_notes", [])),
            alternate_finalist_audit_results=[
                dict(item) for item in data.get("alternate_finalist_audit_results", [])
            ],
            company_autonomy_attempted=bool(data.get("company_autonomy_attempted", False)),
            company_autonomy_status=data.get("company_autonomy_status"),
            company_autonomy_notes=list(data.get("company_autonomy_notes", [])),
            company_autonomy_runs=[dict(item) for item in data.get("company_autonomy_runs", [])],
            company_autonomy_decision_trace=dict(data.get("company_autonomy_decision_trace", {})),
            relative_ranking=[dict(item) for item in data.get("relative_ranking", [])],
            memo_body=dict(data.get("memo_body", {})),
            no_selection_reason=data.get("no_selection_reason"),
            degraded_states=list(data.get("degraded_states", [])),
            audit_notes=list(data.get("audit_notes", [])),
            contract_version=data.get("contract_version", SECTOR_CONTRACT_VERSION),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "AutonomousSectorFinancialRunArtifact",
    "CandidateDisposition",
    "GateEvaluation",
    "SECTOR_CONTRACT_VERSION",
    "SECTOR_CONTRACT_VERSION_V1",
    "SECTOR_CONTRACT_VERSION_V2",
    "SECTOR_PIPELINE_VERSION_V1",
    "SECTOR_PIPELINE_VERSION_V2",
    "SectorCompanyFinancialPacket",
    "SectorExpectedReturnScenario",
    "SectorFinalDecision",
    "SectorFinancialFramework",
    "SectorResearchQuestion",
    "SectorSelectionValidation",
    "ScreenResult",
    "UnderwritingResult",
]
