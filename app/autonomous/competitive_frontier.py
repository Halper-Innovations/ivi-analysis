"""Pure deterministic ranking and competitive-frontier state for sector scans.

This module deliberately performs no database, network, or LLM work.  It turns
an already-built packet/scenario cohort into a stable pre-rank, identifies the
Pareto frontier on deterministic score and reliable base return, and exposes a
bounded review cursor.  Runtime orchestration can consume the state later
without making prompt order or frontier closure depend on input order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from app.alpha.cross_sectional_ranker import FactorVector, rank_cross_sectional
from app.autonomous.sector_contract import (
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
)

COMPETITIVE_FRONTIER_SCHEMA_VERSION = "competitive_frontier_v1"
DEFAULT_FRONTIER_BATCH_SIZE = 3
DEFAULT_FRONTIER_TOP_N = 25

_SCORE_KEYS = (
    "deterministic_score",
    "cross_sectional_score",
    "composite_score",
    "score",
)
_SCORE_STATUSES = frozenset({"AVAILABLE", "MISSING"})
_RETURN_RELIABILITY_STATUSES = frozenset({"RELIABLE", "UNRELIABLE", "MISSING"})
_RANKING_TIERS = frozenset({"COMPLETE", "SCORE_ONLY", "RETURN_ONLY", "UNRESOLVED"})
_RANKING_TIER_ORDER = {
    "COMPLETE": 0,
    "SCORE_ONLY": 1,
    "RETURN_ONLY": 2,
    "UNRESOLVED": 3,
}


def _normalized_ticker(value: Any) -> str:
    ticker = str(value or "").strip().upper()
    if not ticker:
        raise ValueError("competitive-frontier ticker is required")
    return ticker


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


@dataclass(frozen=True)
class RankedFrontierCandidate:
    """One stable rank row with explicit metric availability and reliability."""

    ticker: str
    rank: int
    ranking_tier: str
    deterministic_score: float | None
    deterministic_score_status: str
    deterministic_score_source: str
    deterministic_score_reason_codes: tuple[str, ...]
    base_annualized_return: float | None
    base_return_reliability: str
    base_return_scenario_id: str | None
    base_return_horizon_years: int | None
    base_return_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", _normalized_ticker(self.ticker))
        if self.rank < 1:
            raise ValueError("competitive-frontier rank must be positive")
        if self.ranking_tier not in _RANKING_TIERS:
            raise ValueError(f"invalid competitive-frontier ranking tier: {self.ranking_tier}")
        if self.deterministic_score_status not in _SCORE_STATUSES:
            raise ValueError(
                "invalid competitive-frontier deterministic score status: "
                f"{self.deterministic_score_status}"
            )
        if self.base_return_reliability not in _RETURN_RELIABILITY_STATUSES:
            raise ValueError(
                "invalid competitive-frontier base return reliability: "
                f"{self.base_return_reliability}"
            )
        if self.deterministic_score_status == "AVAILABLE" and self.deterministic_score is None:
            raise ValueError("AVAILABLE deterministic score requires a finite value")
        if self.deterministic_score_status == "MISSING" and self.deterministic_score is not None:
            raise ValueError("MISSING deterministic score cannot carry a value")
        if self.base_return_reliability in {"RELIABLE", "UNRELIABLE"}:
            if self.base_annualized_return is None:
                raise ValueError("resolved base return reliability requires a finite value")
        elif self.base_annualized_return is not None:
            raise ValueError("MISSING base return cannot carry a value")

    @property
    def is_comparable(self) -> bool:
        return (
            self.deterministic_score_status == "AVAILABLE"
            and self.base_return_reliability == "RELIABLE"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "rank": self.rank,
            "ranking_tier": self.ranking_tier,
            "deterministic_score": self.deterministic_score,
            "deterministic_score_status": self.deterministic_score_status,
            "deterministic_score_source": self.deterministic_score_source,
            "deterministic_score_reason_codes": list(self.deterministic_score_reason_codes),
            "base_annualized_return": self.base_annualized_return,
            "base_return_reliability": self.base_return_reliability,
            "base_return_scenario_id": self.base_return_scenario_id,
            "base_return_horizon_years": self.base_return_horizon_years,
            "base_return_reason_codes": list(self.base_return_reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RankedFrontierCandidate:
        return cls(
            ticker=str(data["ticker"]),
            rank=int(data["rank"]),
            ranking_tier=str(data["ranking_tier"]),
            deterministic_score=_finite_float(data.get("deterministic_score")),
            deterministic_score_status=str(data["deterministic_score_status"]),
            deterministic_score_source=str(data["deterministic_score_source"]),
            deterministic_score_reason_codes=tuple(
                str(item) for item in data.get("deterministic_score_reason_codes", [])
            ),
            base_annualized_return=_finite_float(data.get("base_annualized_return")),
            base_return_reliability=str(data["base_return_reliability"]),
            base_return_scenario_id=(
                str(data["base_return_scenario_id"])
                if data.get("base_return_scenario_id") is not None
                else None
            ),
            base_return_horizon_years=(
                int(data["base_return_horizon_years"])
                if data.get("base_return_horizon_years") is not None
                else None
            ),
            base_return_reason_codes=tuple(
                str(item) for item in data.get("base_return_reason_codes", [])
            ),
        )


@dataclass(frozen=True)
class FrontierClosureCertificate:
    """Deterministic proof of whether the live competitive frontier is closed."""

    status: str
    closed: bool
    frontier_tickers: tuple[str, ...]
    reviewed_tickers: tuple[str, ...]
    pending_tickers: tuple[str, ...]
    dominated_tickers: tuple[str, ...]
    unresolved_tickers: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "closed": self.closed,
            "frontier_tickers": list(self.frontier_tickers),
            "reviewed_tickers": list(self.reviewed_tickers),
            "pending_tickers": list(self.pending_tickers),
            "dominated_tickers": list(self.dominated_tickers),
            "unresolved_tickers": list(self.unresolved_tickers),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class CompetitiveFrontierState:
    """Stable review state over a complete ranked sector candidate cohort."""

    candidates: tuple[RankedFrontierCandidate, ...]
    reviewed_tickers: tuple[str, ...] = ()
    top_n: int = DEFAULT_FRONTIER_TOP_N
    batch_size: int = DEFAULT_FRONTIER_BATCH_SIZE

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("competitive-frontier top_n must be positive")
        if self.batch_size < 1:
            raise ValueError("competitive-frontier batch_size must be positive")
        candidate_tickers = [candidate.ticker for candidate in self.candidates]
        if len(candidate_tickers) != len(set(candidate_tickers)):
            raise ValueError("competitive-frontier candidates must have unique tickers")
        if [candidate.rank for candidate in self.candidates] != list(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError(
                "competitive-frontier candidates must be stored in contiguous rank order"
            )

        reviewed = {_normalized_ticker(ticker) for ticker in self.reviewed_tickers}
        unknown = reviewed - set(candidate_tickers)
        if unknown:
            raise ValueError(
                "competitive-frontier reviewed tickers are outside the candidate pool: "
                + ", ".join(sorted(unknown))
            )
        ordered_reviewed = tuple(ticker for ticker in candidate_tickers if ticker in reviewed)
        object.__setattr__(self, "reviewed_tickers", ordered_reviewed)

    @property
    def top_tickers(self) -> tuple[str, ...]:
        return tuple(candidate.ticker for candidate in self.candidates[: self.top_n])

    @property
    def unresolved_tickers(self) -> tuple[str, ...]:
        return tuple(
            candidate.ticker for candidate in self.candidates if not candidate.is_comparable
        )

    @property
    def frontier_tickers(self) -> tuple[str, ...]:
        comparable = [candidate for candidate in self.candidates if candidate.is_comparable]
        return tuple(
            candidate.ticker
            for candidate in comparable
            if not any(
                _dominates(other, candidate)
                for other in comparable
                if other.ticker != candidate.ticker
            )
        )

    @property
    def dominated_tickers(self) -> tuple[str, ...]:
        frontier = set(self.frontier_tickers)
        return tuple(
            candidate.ticker
            for candidate in self.candidates
            if candidate.is_comparable and candidate.ticker not in frontier
        )

    @property
    def dominators_by_ticker(self) -> dict[str, tuple[str, ...]]:
        comparable = [candidate for candidate in self.candidates if candidate.is_comparable]
        return {
            candidate.ticker: tuple(
                other.ticker
                for other in comparable
                if other.ticker != candidate.ticker and _dominates(other, candidate)
            )
            for candidate in comparable
            if candidate.ticker in set(self.dominated_tickers)
        }

    @property
    def pending_tickers(self) -> tuple[str, ...]:
        reviewed = set(self.reviewed_tickers)
        return tuple(ticker for ticker in self.frontier_tickers if ticker not in reviewed)

    @property
    def next_batch(self) -> tuple[str, ...]:
        return self.pending_tickers[: self.batch_size]

    @property
    def needs_continuation(self) -> bool:
        return bool(self.pending_tickers)

    @property
    def closure_certificate(self) -> FrontierClosureCertificate:
        reasons: list[str] = []
        if self.pending_tickers:
            reasons.append("UNREVIEWED_NONDOMINATED_CANDIDATES")
        if self.unresolved_tickers:
            reasons.append("UNRESOLVED_FRONTIER_METRICS")
        closed = not reasons
        return FrontierClosureCertificate(
            status="CLOSED" if closed else "OPEN",
            closed=closed,
            frontier_tickers=self.frontier_tickers,
            reviewed_tickers=self.reviewed_tickers,
            pending_tickers=self.pending_tickers,
            dominated_tickers=self.dominated_tickers,
            unresolved_tickers=self.unresolved_tickers,
            reason_codes=tuple(reasons),
        )

    def mark_reviewed(self, tickers: Iterable[str]) -> CompetitiveFrontierState:
        reviewed = set(self.reviewed_tickers)
        reviewed.update(_normalized_ticker(ticker) for ticker in tickers)
        return replace(self, reviewed_tickers=tuple(reviewed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPETITIVE_FRONTIER_SCHEMA_VERSION,
            "top_n": self.top_n,
            "batch_size": self.batch_size,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reviewed_tickers": list(self.reviewed_tickers),
            "top_tickers": list(self.top_tickers),
            "frontier_tickers": list(self.frontier_tickers),
            "pending_tickers": list(self.pending_tickers),
            "dominated_tickers": list(self.dominated_tickers),
            "unresolved_tickers": list(self.unresolved_tickers),
            "dominators_by_ticker": {
                ticker: list(dominators) for ticker, dominators in self.dominators_by_ticker.items()
            },
            "next_batch": list(self.next_batch),
            "needs_continuation": self.needs_continuation,
            "closure_certificate": self.closure_certificate.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompetitiveFrontierState:
        if data.get("schema_version") != COMPETITIVE_FRONTIER_SCHEMA_VERSION:
            raise ValueError("unsupported competitive-frontier schema version")
        return cls(
            candidates=tuple(
                RankedFrontierCandidate.from_dict(item)
                for item in data.get("candidates", [])
                if isinstance(item, Mapping)
            ),
            reviewed_tickers=tuple(str(item) for item in data.get("reviewed_tickers", [])),
            top_n=int(data.get("top_n", DEFAULT_FRONTIER_TOP_N)),
            batch_size=int(data.get("batch_size", DEFAULT_FRONTIER_BATCH_SIZE)),
        )


@dataclass(frozen=True)
class _BaseReturnMetric:
    value: float | None
    reliability: str
    scenario_id: str | None
    horizon_years: int | None
    reason_codes: tuple[str, ...]


def _packet_factor_vector(packet: SectorCompanyFinancialPacket) -> FactorVector:
    valuation = packet.valuation or {}
    returns_on_capital = packet.returns_on_capital or {}
    cash_conversion = packet.cash_conversion or {}
    quality = _finite_float(returns_on_capital.get("roic"))
    if quality is None:
        quality = _finite_float(returns_on_capital.get("roic_wacc_spread"))
    if quality is None:
        quality = _finite_float(cash_conversion.get("fcf_margin"))
    return FactorVector(
        ticker=_normalized_ticker(packet.ticker),
        value=_finite_float(valuation.get("discount_to_anchor")),
        quality=quality,
        gap=_finite_float(valuation.get("implied_growth")),
    )


def _derived_cross_sectional_scores(
    packets: Sequence[SectorCompanyFinancialPacket],
) -> dict[str, float | None]:
    # Ticker sorting neutralizes the legacy ranker's original-order tie break.
    vectors = [
        _packet_factor_vector(packet)
        for packet in sorted(packets, key=lambda item: item.ticker.upper())
    ]
    return {row.ticker: _finite_float(row.composite) for row in rank_cross_sectional(vectors)}


def _deterministic_score(
    packet: SectorCompanyFinancialPacket,
    *,
    score_overrides: Mapping[str, float | None],
    derived_scores: Mapping[str, float | None],
) -> tuple[float | None, str, tuple[str, ...]]:
    ticker = _normalized_ticker(packet.ticker)
    if ticker in score_overrides:
        value = _finite_float(score_overrides[ticker])
        if value is not None:
            return value, "OVERRIDE", ()
        return None, "OVERRIDE_MISSING", ("DETERMINISTIC_SCORE_MISSING",)

    components = packet.score_components or {}
    for key in _SCORE_KEYS:
        value = _finite_float(components.get(key))
        if value is not None:
            return value, f"PACKET_SCORE_COMPONENT:{key}", ()

    derived = _finite_float(derived_scores.get(ticker))
    if derived is not None:
        return derived, "CROSS_SECTIONAL_FACTORS", ()
    return None, "MISSING", ("DETERMINISTIC_SCORE_MISSING",)


def _base_return_metric(
    ticker: str,
    scenarios: Sequence[SectorExpectedReturnScenario],
) -> _BaseReturnMetric:
    base_scenarios = [
        scenario
        for scenario in scenarios
        if _normalized_ticker(scenario.ticker) == ticker
        and str(scenario.scenario_name or "").strip().lower() == "base"
    ]
    if not base_scenarios:
        return _BaseReturnMetric(
            value=None,
            reliability="MISSING",
            scenario_id=None,
            horizon_years=None,
            reason_codes=("BASE_SCENARIO_MISSING",),
        )

    numeric = [
        (scenario, value)
        for scenario in base_scenarios
        if (value := _finite_float(scenario.annualized_return)) is not None
    ]
    if not numeric:
        return _BaseReturnMetric(
            value=None,
            reliability="MISSING",
            scenario_id=None,
            horizon_years=None,
            reason_codes=("BASE_ANNUALIZED_RETURN_MISSING",),
        )

    reliable = [item for item in numeric if not item[0].unsupported_assumptions]
    candidates = reliable or numeric
    selected, value = sorted(
        candidates,
        key=lambda item: (
            -item[1],
            int(item[0].horizon_years),
            str(item[0].scenario_id),
        ),
    )[0]
    is_reliable = bool(reliable)
    return _BaseReturnMetric(
        value=value,
        reliability="RELIABLE" if is_reliable else "UNRELIABLE",
        scenario_id=str(selected.scenario_id),
        horizon_years=int(selected.horizon_years),
        reason_codes=() if is_reliable else ("UNSUPPORTED_BASE_RETURN_ASSUMPTIONS",),
    )


def _ranking_tier(*, score_available: bool, return_reliable: bool) -> str:
    if score_available and return_reliable:
        return "COMPLETE"
    if score_available:
        return "SCORE_ONLY"
    if return_reliable:
        return "RETURN_ONLY"
    return "UNRESOLVED"


def _ranking_key(candidate: RankedFrontierCandidate) -> tuple[int, float, float, str]:
    score = candidate.deterministic_score
    reliable_return = (
        candidate.base_annualized_return
        if candidate.base_return_reliability == "RELIABLE"
        else None
    )
    return (
        _RANKING_TIER_ORDER[candidate.ranking_tier],
        -(score if score is not None else 0.0),
        -(reliable_return if reliable_return is not None else 0.0),
        candidate.ticker,
    )


def rank_competitive_candidates(
    company_packets: Sequence[SectorCompanyFinancialPacket],
    scenarios: Sequence[SectorExpectedReturnScenario],
    *,
    deterministic_scores: Mapping[str, float | None] | None = None,
) -> tuple[RankedFrontierCandidate, ...]:
    """Rank every packet deterministically, retaining incomplete candidates."""

    packets_by_ticker: dict[str, SectorCompanyFinancialPacket] = {}
    for packet in company_packets:
        ticker = _normalized_ticker(packet.ticker)
        if ticker in packets_by_ticker:
            raise ValueError(f"duplicate competitive-frontier packet ticker: {ticker}")
        packets_by_ticker[ticker] = packet

    score_overrides = {
        _normalized_ticker(ticker): value for ticker, value in (deterministic_scores or {}).items()
    }
    unknown_overrides = set(score_overrides) - set(packets_by_ticker)
    if unknown_overrides:
        raise ValueError(
            "competitive-frontier score overrides are outside the packet pool: "
            + ", ".join(sorted(unknown_overrides))
        )

    packets = list(packets_by_ticker.values())
    derived_scores = _derived_cross_sectional_scores(packets)
    unranked: list[RankedFrontierCandidate] = []
    for ticker, packet in packets_by_ticker.items():
        score, score_source, score_reasons = _deterministic_score(
            packet,
            score_overrides=score_overrides,
            derived_scores=derived_scores,
        )
        base_return = _base_return_metric(ticker, scenarios)
        score_status = "AVAILABLE" if score is not None else "MISSING"
        unranked.append(
            RankedFrontierCandidate(
                ticker=ticker,
                rank=1,
                ranking_tier=_ranking_tier(
                    score_available=score is not None,
                    return_reliable=base_return.reliability == "RELIABLE",
                ),
                deterministic_score=score,
                deterministic_score_status=score_status,
                deterministic_score_source=score_source,
                deterministic_score_reason_codes=score_reasons,
                base_annualized_return=base_return.value,
                base_return_reliability=base_return.reliability,
                base_return_scenario_id=base_return.scenario_id,
                base_return_horizon_years=base_return.horizon_years,
                base_return_reason_codes=base_return.reason_codes,
            )
        )

    return tuple(
        replace(candidate, rank=rank)
        for rank, candidate in enumerate(sorted(unranked, key=_ranking_key), start=1)
    )


def _dominates(
    left: RankedFrontierCandidate,
    right: RankedFrontierCandidate,
) -> bool:
    if not left.is_comparable or not right.is_comparable:
        return False
    left_score = left.deterministic_score
    right_score = right.deterministic_score
    left_return = left.base_annualized_return
    right_return = right.base_annualized_return
    if None in {left_score, right_score, left_return, right_return}:
        return False
    return (
        left_score >= right_score
        and left_return >= right_return
        and (left_score > right_score or left_return > right_return)
    )


def build_competitive_frontier(
    company_packets: Sequence[SectorCompanyFinancialPacket],
    scenarios: Sequence[SectorExpectedReturnScenario],
    *,
    deterministic_scores: Mapping[str, float | None] | None = None,
    reviewed_tickers: Iterable[str] = (),
    top_n: int = DEFAULT_FRONTIER_TOP_N,
    batch_size: int = DEFAULT_FRONTIER_BATCH_SIZE,
) -> CompetitiveFrontierState:
    """Build a pure ranked frontier and its bounded company-review cursor."""

    return CompetitiveFrontierState(
        candidates=rank_competitive_candidates(
            company_packets,
            scenarios,
            deterministic_scores=deterministic_scores,
        ),
        reviewed_tickers=tuple(reviewed_tickers),
        top_n=top_n,
        batch_size=batch_size,
    )


def validate_closed_competitive_frontier_payload(
    payload: Mapping[str, Any],
    *,
    expected_candidate_tickers: Iterable[str] | None = None,
    expected_reviewed_tickers: Iterable[str] | None = None,
    source_company_packets: Sequence[SectorCompanyFinancialPacket] | None = None,
    source_scenarios: Sequence[SectorExpectedReturnScenario] | None = None,
    deterministic_scores: Mapping[str, float | None] | None = None,
) -> CompetitiveFrontierState:
    """Validate the persisted proof used to close a v2 sector decision.

    ``CompetitiveFrontierState.from_dict`` deliberately rebuilds derived
    fields rather than trusting them.  When source packets and scenarios are
    supplied, this validator goes further: it independently rebuilds candidate
    metrics, rank, dominance, and closure from those canonical artifact inputs.
    The persisted frontier therefore cannot validate a coherent but forged set
    of candidate metric rows.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("competitive frontier must be a mapping")
    expected_candidate_values = (
        tuple(expected_candidate_tickers)
        if expected_candidate_tickers is not None
        else None
    )
    expected_reviewed_values = (
        tuple(expected_reviewed_tickers)
        if expected_reviewed_tickers is not None
        else None
    )
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not all(
        isinstance(item, Mapping) for item in raw_candidates
    ):
        raise ValueError("competitive frontier candidates must be a list of mappings")
    try:
        state = CompetitiveFrontierState.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid competitive frontier state: {exc}") from exc
    if len(state.candidates) != len(raw_candidates):
        raise ValueError("competitive frontier candidate rows did not deserialize exactly")

    if (source_company_packets is None) != (source_scenarios is None):
        raise ValueError(
            "competitive frontier source packets and scenarios must be supplied together"
        )

    if source_company_packets is not None and source_scenarios is not None:
        source_tickers = tuple(
            _normalized_ticker(packet.ticker) for packet in source_company_packets
        )
        if len(source_tickers) != len(set(source_tickers)):
            raise ValueError("competitive frontier source packets must have unique tickers")
        expected_source_tickers = (
            {_normalized_ticker(item) for item in expected_candidate_values}
            if expected_candidate_values is not None
            else set(source_tickers)
        )
        if set(source_tickers) != expected_source_tickers:
            raise ValueError(
                "competitive frontier source packets do not exactly reconcile eligible candidates"
            )
        scenario_tickers = {
            _normalized_ticker(scenario.ticker) for scenario in source_scenarios
        }
        if scenario_tickers - expected_source_tickers:
            raise ValueError(
                "competitive frontier source scenarios contain ineligible candidates"
            )
        source_reviewed = (
            expected_reviewed_values
            if expected_reviewed_values is not None
            else state.reviewed_tickers
        )
        try:
            source_state = build_competitive_frontier(
                source_company_packets,
                source_scenarios,
                deterministic_scores=deterministic_scores,
                reviewed_tickers=source_reviewed,
                top_n=DEFAULT_FRONTIER_TOP_N,
                batch_size=DEFAULT_FRONTIER_BATCH_SIZE,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid competitive frontier source cohort: {exc}"
            ) from exc
        canonical = source_state.to_dict()
    else:
        source_state = state
        canonical = state.to_dict()
    derived_fields = (
        "schema_version",
        "top_n",
        "batch_size",
        "candidates",
        "reviewed_tickers",
        "top_tickers",
        "frontier_tickers",
        "pending_tickers",
        "dominated_tickers",
        "unresolved_tickers",
        "dominators_by_ticker",
        "next_batch",
        "needs_continuation",
        "closure_certificate",
    )
    for field_name in derived_fields:
        if field_name not in payload or payload.get(field_name) != canonical[field_name]:
            raise ValueError(
                f"competitive frontier field {field_name} does not match rebuilt "
                "source-backed state"
            )

    state = source_state
    candidate_tickers = tuple(candidate.ticker for candidate in state.candidates)
    reviewed_tickers = tuple(state.reviewed_tickers)
    if expected_candidate_values is not None:
        expected = {_normalized_ticker(item) for item in expected_candidate_values}
        if set(candidate_tickers) != expected:
            raise ValueError(
                "competitive frontier candidate pool does not reconcile eligible candidates"
            )
    if expected_reviewed_values is not None:
        expected = {_normalized_ticker(item) for item in expected_reviewed_values}
        if set(reviewed_tickers) != expected:
            raise ValueError(
                "competitive frontier reviewed set does not reconcile completed underwriting"
            )

    minimum_required = min(DEFAULT_FRONTIER_BATCH_SIZE, len(candidate_tickers))
    if payload.get("minimum_reviews_required") != minimum_required:
        raise ValueError("competitive frontier minimum review count is inconsistent")
    if payload.get("successful_review_count") != len(reviewed_tickers):
        raise ValueError("competitive frontier successful review count is inconsistent")

    attempted_raw = payload.get("attempted_tickers")
    failed_raw = payload.get("failed_review_tickers")
    if not isinstance(attempted_raw, list) or not isinstance(failed_raw, list):
        raise ValueError("competitive frontier attempt fields must be lists")
    attempted = tuple(_normalized_ticker(item) for item in attempted_raw)
    failed = tuple(_normalized_ticker(item) for item in failed_raw)
    if len(attempted) != len(set(attempted)) or len(failed) != len(set(failed)):
        raise ValueError("competitive frontier attempt fields must contain unique tickers")
    candidate_set = set(candidate_tickers)
    reviewed_set = set(reviewed_tickers)
    attempted_set = set(attempted)
    failed_set = set(failed)
    if attempted_set - candidate_set or failed_set - candidate_set:
        raise ValueError("competitive frontier attempts must stay inside the candidate pool")
    if attempted_set != reviewed_set | failed_set or failed_set != attempted_set - reviewed_set:
        raise ValueError("competitive frontier attempts do not reconcile reviews and failures")

    certificate = state.closure_certificate
    if (
        str(payload.get("status") or "").upper() != "CLOSED"
        or not certificate.closed
        or certificate.status != "CLOSED"
        or len(reviewed_tickers) < minimum_required
        or failed
    ):
        raise ValueError("competitive frontier is not closed")
    return state


__all__ = [
    "COMPETITIVE_FRONTIER_SCHEMA_VERSION",
    "DEFAULT_FRONTIER_BATCH_SIZE",
    "DEFAULT_FRONTIER_TOP_N",
    "CompetitiveFrontierState",
    "FrontierClosureCertificate",
    "RankedFrontierCandidate",
    "build_competitive_frontier",
    "rank_competitive_candidates",
    "validate_closed_competitive_frontier_payload",
]
