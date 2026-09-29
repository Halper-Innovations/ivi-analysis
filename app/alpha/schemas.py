"""Data structures for the recursive alpha engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickerSignalPacket:
    """All available signals for one ticker, assembled for comparison."""

    ticker: str
    # Valuation
    dcf_value: float | None = None
    epv_value: float | None = None
    graham_value: float | None = None
    ncav_value: float | None = None
    insurance_value: float | None = None
    insurance_method: str | None = None
    current_price: float | None = None
    current_price_unit: str | None = None
    current_price_as_of_date: str | None = None
    current_price_currency: str | None = None
    current_price_source: str | None = None
    current_price_source_url: str | None = None
    quote_snapshot_id: str | None = None
    price_basis: str | None = None
    raw_price: float | None = None
    split_adjustment_factor: float | None = None
    split_effective_date: str | None = None
    split_lineage_proof: dict[str, Any] | None = None
    market_cap_mm: float | None = None
    market_cap_unit: str | None = None
    market_cap_source: str | None = None
    market_cap_effective_as_of_date: str | None = None
    market_cap_source_kind: str | None = None
    market_cap_source_name: str | None = None
    market_cap_source_url: str | None = None
    market_cap_confidence: str | None = None
    market_cap_method: str | None = None
    market_cap_derivation: dict[str, Any] = field(default_factory=dict)
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
    cap_scope_status: str | None = None
    cap_scope_reason: str | None = None
    cap_stage_price: float | None = None
    cap_stage_price_as_of_date: str | None = None
    cap_stage_price_currency: str | None = None
    cap_stage_price_source: str | None = None
    cap_stage_price_source_url: str | None = None
    cap_stage_quote_snapshot_id: str | None = None
    metric_traces: dict[str, Any] = field(default_factory=dict)
    financial_integrity_status: str | None = None
    financial_integrity_violations: list[dict[str, Any]] = field(default_factory=list)
    # Scorecard pricing zone (MARGIN_OF_SAFETY / GROWTH_DEPENDENT /
    # SPECULATIVE_PREMIUM / VALUATION_ANOMALY / ...) — rankers must exclude
    # VALUATION_ANOMALY names (audit: anomaly-zone-still-anchors-in-backtest).
    pricing_zone: str | None = None
    implied_growth: float | None = None
    reverse_dcf_feasibility: str | None = None
    margin_of_safety_verdict: str | None = None
    # Quality gate
    gate_verdict: str | None = None
    confidence_class: str | None = None
    moat_score: int | None = None
    moat_classification: str | None = None
    downside_risk_class: str | None = None
    valuation_headwinds: list[str] = field(default_factory=list)
    valuation_supports: list[str] = field(default_factory=list)
    valuation_provenance_status: str | None = None
    valuation_provenance_blockers: list[str] = field(default_factory=list)
    # Peer context
    peer_position: str | None = None
    roic_vs_median: float | None = None
    op_margin_vs_median: float | None = None
    revenue_growth_vs_median: float | None = None
    # Filing diff signals
    filing_diff_changes: list[dict[str, Any]] = field(default_factory=list)
    high_materiality_changes: int = 0
    # Pattern hits
    pattern_hits: list[dict[str, Any]] = field(default_factory=list)
    patterns_with_signal: list[str] = field(default_factory=list)
    # Filing risk scan
    filing_risk_signals: dict[str, str] = field(default_factory=dict)
    filing_risk_status: str | None = None  # OK / NO_FILING / KEYWORD_FALLBACK / ERROR
    filing_risk_metadata: dict[str, Any] = field(default_factory=dict)
    # Deep research
    research_report: dict[str, Any] = field(default_factory=dict)
    research_status: str | None = None  # OK / NO_FILING / LLM_DISABLED / ERROR
    solvency_risk: str | None = None  # CRITICAL / ELEVATED / LOW / UNKNOWN
    anomaly_count: int = 0
    # Method tension
    method_tension_type: str | None = (
        None  # NONE / GROWTH_VS_EARNINGS_POWER / ASSET_VS_EARNINGS / INSUFFICIENT_METHODS
    )
    growth_dependency_ratio: float | None = None
    methods_agree: bool | None = None
    consensus_direction: str | None = None  # UNDERVALUED / OVERVALUED / MIXED / FAIR / UNKNOWN
    intrinsic_range_low: float | None = None
    intrinsic_range_high: float | None = None
    # Quarterly freshness
    latest_quarterly_revenue: float | None = None
    latest_quarterly_period: str | None = None
    quarterly_revenue_trend: str | None = None  # ACCELERATING / DECELERATING / STABLE / UNKNOWN
    # Security / insurance routing
    security_type: str | None = None
    issuer_type: str | None = None
    insurance_subtype: str | None = None
    insurance_packet: dict[str, Any] = field(default_factory=dict)
    insurance_valuation: dict[str, Any] = field(default_factory=dict)
    model_status: str | None = None
    model_blockers: list[str] = field(default_factory=list)
    model_fit_warnings: list[str] = field(default_factory=list)
    # Raw data for LLM context
    raw_valuation: dict[str, Any] = field(default_factory=dict)
    raw_quality_ctx: dict[str, Any] = field(default_factory=dict)

    def discount_to_dcf_pct(self) -> float | None:
        if self.current_price and self.dcf_value and self.dcf_value > 0:
            return (self.dcf_value - self.current_price) / self.dcf_value
        return None

    def to_summary_dict(self) -> dict[str, Any]:
        """Compact dict for LLM context — only populated fields."""
        d: dict[str, Any] = {"ticker": self.ticker}
        for k, v in self.__dict__.items():
            if k == "ticker":
                continue
            if v is None or v == [] or v == {} or v == 0:
                continue
            if k.startswith("raw_"):
                continue
            d[k] = v
        return d


@dataclass
class ComparisonRound:
    """One round of the elimination process."""

    round_number: int
    candidates_entering: list[str]
    candidates_eliminated: list[str]
    candidates_remaining: list[str]
    reasoning: str
    elimination_criteria: str


@dataclass
class SectorAlphaReport:
    """Final output of the recursive sector comparison."""

    sector: str
    total_candidates: int
    rounds: list[ComparisonRound]
    winner: str | None
    winner_thesis: str
    winner_conviction: str  # HIGH / MODERATE / LOW
    runner_up: str | None
    runner_up_thesis: str
    key_risk: str
    falsification_trigger: str
    time_horizon: str
    signal_packets: dict[str, dict[str, Any]]  # ticker -> summary_dict
    pre_investigation_winner: str | None = None
    pre_investigation_runner_up: str | None = None
    selection_basis: str = "consensus"
    investigation_previews: list[dict[str, Any]] = field(default_factory=list)
    prior_ranking: list[dict[str, Any]] = field(default_factory=list)
    investigation_plan: dict[str, Any] = field(default_factory=dict)
    candidate_investigations: list[dict[str, Any]] = field(default_factory=list)
    hard_blocked_candidates: list[dict[str, Any]] = field(default_factory=list)
    llm_decision_trace: dict[str, Any] = field(default_factory=dict)
    tool_budget_summary: dict[str, Any] = field(default_factory=dict)
    financial_integrity_binding: dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """A quantitative anomaly detected in the financial data."""

    anomaly_type: str  # e.g. Q4_EARNINGS_BOMB, NEGATIVE_EQUITY, MARGIN_COLLAPSE
    severity: str  # HIGH / MODERATE / LOW
    description: str  # Human-readable
    question: str  # What the filing should explain
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Investigation:
    """Result of searching the filing text to answer an anomaly's question."""

    anomaly_type: str
    question: str
    answer: str
    evidence_excerpt: str
    follow_up: str | None = None


@dataclass(frozen=True)
class GoingConcernAssertion:
    """One filing-backed, subject-attributed going-concern assertion.

    ``blockable`` is deliberately persisted rather than recomputed by every
    consumer.  It is true only for a current affirmative assertion about the
    registrant, consolidated group, or a consolidated subsidiary.
    """

    subject: str
    assertion_mode: str
    accession: str | None
    form_type: str | None
    filing_date: str | None
    section: str
    excerpt: str
    corroborating_distress: tuple[str, ...] = ()
    blockable: bool = False
    subject_detail: str | None = None
    issuer_cik: str | None = None
    source_url: str | None = None
    content_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe assertion payload for run artifacts."""

        return {
            "subject": self.subject,
            "subject_detail": self.subject_detail,
            "assertion_mode": self.assertion_mode,
            "blockable": self.blockable,
            "accession": self.accession,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "section": self.section,
            "excerpt": self.excerpt,
            "corroborating_distress": list(self.corroborating_distress),
            "issuer_cik": self.issuer_cik,
            "source_url": self.source_url,
            "content_revision": self.content_revision,
        }


@dataclass
class SolvencyAssessment:
    """Going-concern and solvency risk signals."""

    solvency_risk: str  # CRITICAL / ELEVATED / LOW / UNKNOWN
    negative_equity: bool = False
    current_ratio: float | None = None
    current_ratio_distressed: bool = False
    debt_due_within_12mo: bool = False
    going_concern_language: bool = False
    valuation_allowance_full: bool = False
    no_assurance_financing: bool = False
    cash_runway_quarters: float | None = None
    going_concern_assertions: list[GoingConcernAssertion] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    details: str = ""


@dataclass
class ResearchReport:
    """Complete research output for one ticker."""

    ticker: str
    anomalies: list[Anomaly]
    investigations: list[Investigation]
    solvency: SolvencyAssessment
    summary: str
    status: str  # OK / NO_FILING / LLM_DISABLED / ERROR
