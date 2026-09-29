"""Pydantic response models — the API contract the SPA builds against."""

from __future__ import annotations


from pydantic import BaseModel


class WatchlistRow(BaseModel):
    id: int
    ticker: str
    status: str | None
    presented_status: str | None
    event_pending: str | None
    conviction_grade: str | None
    confidence: str | None
    conviction_source: str | None
    pipeline_version: str | None
    candidate_disposition: str | None
    decision_basis: str | None
    selection_validation_status: str | None
    price_trigger_eligible: bool
    scan_family: str | None
    latest_price: float | None
    latest_price_source: str | None
    latest_price_checked_at: str | None
    buy_price_target: float | None
    distance_from_buy_pct: float | None
    valuation_anchor_method: str | None
    valuation_anchor_value: float | None
    source_sector: str | None
    status_reason: str | None
    added_at: str | None
    last_evaluated_at: str | None
    falsifiers: list[str]
    market_cap_mm: float | None
    cap_source: str | None
    cap_band: str | None
    cap_band_label: str
    adv_dollar_20d: float | None
    adv_dollar_60d: float | None
    adv_asof: str | None
    capacity_class: str
    cheapness: str


class WatchlistResponse(BaseModel):
    rows: list[WatchlistRow]
    total: int
    generated_at: str


class SearchResult(BaseModel):
    """One ⌘K match: a ticker/name from the registrant census or watchlist.

    ``covered`` marks names the platform can render a company page for;
    census-only rows surface as known-but-uncovered."""

    ticker: str
    name: str | None
    sector: str | None
    covered: bool
    presented_status: str | None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    generated_at: str


class RunSummary(BaseModel):
    run_id: str
    slug: str | None = None
    path: str
    kind: str
    contract_version: str | None
    pipeline_version: str | None
    sector: str | None
    market_cap_focus: str | None
    scan_family: str | None
    as_of_date: str | None
    created_at: str | None
    completed_at: str | None
    status: str | None
    execution_status: str | None
    decision_status: str | None
    final_verdict: str | None
    selected_ticker: str | None
    no_selection_reason: str | None
    examined_count: int | None
    disposition_counts_json: str | None
    cost_microdollars: int | None
    cost_usd: float | None
    report_path: str | None
    parse_error: str | None
    integrity_status: str
    decision_eligible: bool
    indexed_at: str


class RunsIndexStats(BaseModel):
    discovered: int
    parsed: int
    unchanged: int
    quarantined: int
    removed: int


class RunsResponse(BaseModel):
    runs: list[RunSummary]
    total: int
    index: RunsIndexStats | None
    generated_at: str


class HealthCheck(BaseModel):
    name: str
    ok: bool
    detail: str


class HealthBlock(BaseModel):
    state: str  # GREEN | AMBER | RED — DataHealth.state verbatim
    blocking: bool
    checks: list[HealthCheck]


class DecisionCard(BaseModel):
    id: int
    ticker: str
    kind: str
    opened_at: str | None
    opened_by: str | None
    trigger: dict
    pre_mortem: str | None
    watchlist_id: int | None
    watchlist_status: str | None
    conviction_grade: str | None
    confidence: str | None
    source_sector: str | None
    source_state: str | None
    is_current_actionable: bool
    buy_price_target: float | None
    latest_price: float | None
    latest_price_checked_at: str | None
    price_age_hours: float | None
    distance_from_buy_pct: float | None
    falsifiers: list[str]
    journal_command: str


class WaterlineItem(BaseModel):
    watchlist_id: int
    ticker: str
    presented_status: str | None
    conviction_grade: str | None
    confidence: str | None
    latest_price: float | None
    latest_price_checked_at: str | None
    price_age_hours: float | None
    price_suspect: bool
    buy_price_target: float | None
    distance_from_buy_pct: float | None
    valuation_anchor_method: str | None
    valuation_anchor_value: float | None
    source_sector: str | None
    wake: list[float]


class WaterlineDeeperName(BaseModel):
    ticker: str
    distance_from_buy_pct: float | None


class WaterlineDeeper(BaseModel):
    """Submerged names beyond the strip's cut — the seabed footer."""

    count: int
    names: list[WaterlineDeeperName]


class GateBlockedItem(BaseModel):
    watchlist_id: int
    ticker: str
    event_pending: str | None
    conviction_grade: str | None
    distance_from_buy_pct: float | None
    latest_price: float | None
    buy_price_target: float | None


class DigestBlock(BaseModel):
    date: str
    previous_date: str | None
    path: str
    html: str


class TodayResponse(BaseModel):
    health: HealthBlock
    decisions: list[DecisionCard]
    waterline: list[WaterlineItem]
    waterline_deeper: WaterlineDeeper
    gate_blocked: list[GateBlockedItem]
    digest: DigestBlock | None
    generated_at: str


class WaccAdjustment(BaseModel):
    code: str
    delta: float | None
    reason: str


class WaccDetail(BaseModel):
    baseline_wacc: float | None
    adjusted_wacc: float | None
    adjustments: list[WaccAdjustment]


class FairValue(BaseModel):
    kind: str  # band | line | not_a_price | none
    value: float | None = None
    low: float | None = None
    base: float | None = None
    high: float | None = None


class MethodCard(BaseModel):
    method: str
    as_of_date: str
    created_at: str | None
    status: str | None
    fair_value: FairValue
    flags: list[str]
    wacc: WaccDetail | None
    quality_gate_verdict: str | None
    confidence_class: str | None
    gate_reason_codes: list[str]
    headwinds: list[str]
    supports: list[str]
    extras: dict


class GaugeShelf(BaseModel):
    method: str
    label: str
    value: float | None = None
    low: float | None = None
    base: float | None = None
    high: float | None = None
    emphasized: bool


class EvolutionPoint(BaseModel):
    method: str
    as_of_date: str
    value: float
    archived: bool
    pre_hardening: bool


class PriceBlock(BaseModel):
    latest: float | None
    checked_at: str | None
    source: str | None
    age_hours: float | None
    wake: list[float]


class CompanyProfile(BaseModel):
    watchlist_id: int
    thesis_text: str | None
    key_risks: list[str]
    open_questions: list[str]
    valuation_anchor_method: str | None
    valuation_anchor_value: float | None
    current_price_at_addition: float | None
    source_run_id: str | None
    cap_asof: str | None
    added_at: str | None


class CompanyAudit(BaseModel):
    """Whether the page's valuations are audited, and what to do if they are not."""

    state: str  # AUDITED | NOT_AUDITED | NO_VALUATION
    reason: str | None = None
    message: str
    command: str | None = None


class CompanyResponse(BaseModel):
    ticker: str
    as_of: str | None = None
    watchlist: WatchlistRow | None
    profile: CompanyProfile | None
    valuations: list[MethodCard]
    shelves: list[GaugeShelf]
    evolution: list[EvolutionPoint]
    evolution_dropped_pre_split: int = 0
    evolution_rebased_points: int = 0
    prices: PriceBlock
    generated_at: str
    audit: CompanyAudit | None = None
    # Lineage-free rows (`ivi value`): reference only, never decision-eligible.
    research_valuations: list[MethodCard] = []


class TtmValue(BaseModel):
    value: float
    basis: str  # ttm | fy | quarter_end
    period_end: str
    reason: str


class FundamentalsResponse(BaseModel):
    basis: str  # fy | ttm
    as_of: str | None = None
    # FY payload
    units: str | None = None
    fiscal_years: list[int] | None = None
    series: dict[str, list[float | None]] | None = None
    derived: dict[str, list[float | None]] | None = None
    # TTM payload
    available: bool | None = None
    reason: str | None = None
    values: dict[str, TtmValue] | None = None


class RunFunnelStage(BaseModel):
    key: str
    label: str
    count: int
    tickers: list[str]


class RunGate(BaseModel):
    rule_id: str
    status: str
    applicable: bool
    observed_value: str | None
    threshold: str | None
    reason_code: str | None
    evidence_url: str | None
    notes: list[str]


class RunCandidate(BaseModel):
    ticker: str
    primary_ticker: str | None
    terminal_state: str
    last_completed_stage: str | None
    scope_status: str | None
    screen_status: str | None
    review_status: str | None
    reason_codes: list[str]
    security_type: str | None
    is_adr: bool | None
    is_secondary_class: bool | None
    frontier_status: str | None
    frontier_dominated_by: str | None
    underwriting_verdict: str | None
    underwriting_confidence: str | None
    watchlist_eligible: bool | None
    gates: list[RunGate]
    failed_gates: int


class RunRankingRow(BaseModel):
    rank: int | None
    ticker: str
    audit_status: str | None
    actionable: bool | None
    buy_candidate: bool | None
    buy_candidate_reason: str | None
    hard_blockers: list[str]
    cross_sectional_rank: int | None
    cross_sectional_percentile: float | None
    best_base_annualized_return: float | None
    positioning_summary: str | None


class RunPacket(BaseModel):
    ticker: str
    blockers: list[str]
    data_quality_status: str | None
    financial_status: str | None
    model_fit_status: str | None
    market_cap_mm: float | None
    market_cap_category: str | None
    anchor_method: str | None
    current_price: float | None
    discount_to_anchor: float | None


class RunSelection(BaseModel):
    source: str | None
    requested_tickers: list[str]
    excluded_tickers: list[str]
    loaded_tickers: list[str]
    selected_tickers: list[str]
    admitted_tickers: list[str]
    warnings: list[str]


class RunDecision(BaseModel):
    status: str | None
    execution_status: str | None
    decision_status: str | None
    final_verdict: str | None
    selected_ticker: str | None
    no_selection_reason: str | None
    confidence: str | None
    confidence_cap_reasons: list[str]
    memo_present: bool


class RunLane(BaseModel):
    lane: str
    cost_microdollars: int | None
    cost_usd: float | None
    max_cost_microdollars: int | None
    max_cost_usd: float | None
    tool_call_attempts: int | None
    provider_call_attempts: int | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None


class RunProvider(BaseModel):
    provider: str
    model: str
    calls: int
    ok_calls: int
    failed_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_estimate_usd: float


class RunCosts(BaseModel):
    available: bool
    aggregate_cost_microdollars: int | None
    aggregate_cost_usd: float | None
    lanes: list[RunLane]
    providers: list[RunProvider]


class RunWatchlistLink(BaseModel):
    watchlist_id: int
    ticker: str
    status: str | None
    conviction_grade: str | None


class RunDetailResponse(BaseModel):
    summary: RunSummary
    quarantined: bool
    parse_error: str | None
    funnel: list[RunFunnelStage]
    candidates: list[RunCandidate]
    ranking: list[RunRankingRow]
    packets: list[RunPacket]
    selection: RunSelection
    decision: RunDecision
    costs: RunCosts
    watchlist_rows: list[RunWatchlistLink]
    report_available: bool
    generated_at: str


class RunReportResponse(BaseModel):
    run_id: str
    slug: str
    path: str
    bytes: int
    html: str
    integrity_status: str
    decision_eligible: bool
    generated_at: str


class SweepGroup(BaseModel):
    week: str
    week_start: str
    week_end: str
    band: str
    runs: int
    sectors: list[str]
    scan_families: list[str]
    pipelines: list[str]
    verdicts: dict[str, int]
    cost_microdollars: int | None
    cost_usd: float | None
    cost_known_runs: int
    watchlist_rows: int


class SweepsResponse(BaseModel):
    groups: list[SweepGroup]
    generated_at: str


class CoverageCell(BaseModel):
    sector: str
    band: str
    tickers: int
    last_loaded_at: str | None
    age_days: float | None
    run_count: int
    pipelines: list[str]


class CoverageResponse(BaseModel):
    sectors: list[str]
    bands: list[str]
    cells: list[CoverageCell]
    generated_at: str


class CoverageCellRun(BaseModel):
    run_id: str
    slug: str | None
    source: str
    pipeline_version: str | None
    loaded_at: str | None
    tickers: int


class CoverageCellDetail(BaseModel):
    sector: str
    band: str
    tickers: list[str]
    runs: list[CoverageCellRun]
    generated_at: str


class MetaResponse(BaseModel):
    """Boot-time facts for the shell: is IVI online, and of what."""

    app: str
    engine_db_path: str
    engine_db_present: bool
    runs_indexed: int
    watchlist_rows: int | None
    generated_at: str


class OfflineDetail(BaseModel):
    precondition: str
    detail: str


# --- Phase 3: events, ops deck, reader --------------------------------------


class EventFiling(BaseModel):
    accession: str
    form_type: str | None
    filing_date: str | None
    role: str | None


class EventCard(BaseModel):
    id: int
    lane: str  # opportunity | queue_protection
    event_type: str
    status: str
    company_name: str | None
    ticker: str | None
    ticker_state: str | None
    cik: str
    detection_date: str | None
    qualification_date: str | None
    age_days: int | None
    expiry_reason: str | None
    detail: dict
    filings: list[EventFiling]
    dispose_command: str | None


class EventsColumn(BaseModel):
    status: str
    total: int
    cards: list[EventCard]


class EventsLane(BaseModel):
    lane: str
    total: int
    open_total: int
    type_counts: dict[str, int]
    columns: list[EventsColumn]


class ScanDay(BaseModel):
    date: str
    status: str  # OK | FAILED | MISSING | …
    mode: str | None
    index_rows: int | None
    candidate_rows: int | None
    events_created: int | None
    events_updated: int | None


class ScanStrip(BaseModel):
    days: list[ScanDay]  # newest first, business days only
    gap_count: int
    window_business_days: int


class EventsResponse(BaseModel):
    lanes: list[EventsLane]
    unknown_ticker_total: int
    scan_strip: ScanStrip
    generated_at: str


class HeartbeatStep(BaseModel):
    step: str
    status: str
    exit_code: int | None
    detail: str | None
    recorded_at: str | None


class HeartbeatDay(BaseModel):
    run_date: str
    status: str  # COMPLETE | FAILED | STARTED | MISSING
    failed_steps: list[str]
    steps: list[HeartbeatStep]
    log_name: str | None


class HeartbeatRow(BaseModel):
    heartbeat: str
    days: list[HeartbeatDay]  # aligned with HeartbeatsResponse.dates


class HeartbeatsResponse(BaseModel):
    dates: list[str]  # newest first
    heartbeats: list[HeartbeatRow]
    generated_at: str


class BackupLog(BaseModel):
    log_name: str
    date: str | None
    size: int


class BackupsResponse(BaseModel):
    ok: bool
    detail: str  # check_backup_fresh verbatim
    ceiling_days: int
    age_days: int | None
    status: str | None
    status_detail: str | None
    date: str | None
    duration_s: float | None
    finished_at: str | None
    history: list[BackupLog]
    generated_at: str


class CostWeek(BaseModel):
    week: str  # ISO year-Www
    scan_cost_usd: float
    scan_runs: int
    research_cost_usd: float
    research_calls: int
    total_usd: float


class CostBucket(BaseModel):
    key: str
    cost_usd: float
    runs: int | None = None
    calls: int | None = None
    cached_calls: int | None = None


class CostsResponse(BaseModel):
    weeks: list[CostWeek]
    by_sector: list[CostBucket]
    by_band: list[CostBucket]
    by_model: list[CostBucket]
    runs_with_cost: int
    runs_without_cost: int
    total_scan_usd: float
    total_research_usd: float
    generated_at: str


class OpsHealthResponse(BaseModel):
    health: HealthBlock
    generated_at: str


class LogFile(BaseModel):
    name: str
    size: int
    mtime: str


class LogsResponse(BaseModel):
    files: list[LogFile]
    generated_at: str


class LogTailResponse(BaseModel):
    name: str
    lines: list[str]
    truncated: bool
    size: int
    mtime: str
    generated_at: str


class DeadLetterRow(BaseModel):
    id: int
    job_id: int | None
    job_type: str | None
    attempts: int | None
    error_type: str | None
    error_message: str | None
    moved_at: str | None


class ResearchGapRow(BaseModel):
    ticker: str
    as_of_date: str | None
    run_id: str | None
    recency_days_min: int | None
    item_count_30d: int | None
    risk_flags: list[str]
    freshness_bucket: str
    flow_bucket: str
    created_at: str | None


class LegacyRunRow(BaseModel):
    run_id: str
    as_of_date: str | None
    generated_at: str | None
    status: str | None


class ConsolesResponse(BaseModel):
    backlog_size: int
    deadletters: list[DeadLetterRow]
    research_gaps: list[ResearchGapRow]
    legacy_runs: list[LegacyRunRow]
    generated_at: str


class ReaderItem(BaseModel):
    family: str
    title: str
    path: str
    mtime: str
    meta: dict


class ReaderFamily(BaseModel):
    family: str
    label: str
    total: int
    items: list[ReaderItem]


class ReaderLibraryResponse(BaseModel):
    families: list[ReaderFamily]
    generated_at: str


class TocEntry(BaseModel):
    level: int
    text: str
    anchor: str


class ClaimCitation(BaseModel):
    source_url: str
    snippet: str
    section_label: str


class ArtifactClaim(BaseModel):
    claim_id: str
    label: str
    value: float | int | str | bool | None
    unit: str | None
    citations: list[ClaimCitation]


class ReaderArtifactResponse(BaseModel):
    path: str
    title: str
    html: str
    toc: list[TocEntry]
    claims: list[ArtifactClaim] | None
    size: int
    mtime: str
    integrity_status: str
    decision_eligible: bool
    generated_at: str


class NoteCitation(BaseModel):
    section: str
    excerpt: str


class AnalystNote(BaseModel):
    claim: str
    direction: str | None
    severity: str | None
    suggested_adjustment: str | None
    validation_status: str | None
    citations: list[NoteCitation]


class AnalystNotes(BaseModel):
    positives: list[AnalystNote]
    risks: list[AnalystNote]
    surprises: list[AnalystNote]
    adjustment_triggers: list[AnalystNote]
    overall_assessment: str | None
    filing_sections_read: list[str]


class ThesisBlock(BaseModel):
    original_dcf: float | None
    adjusted_dcf: float | None
    original_epv: float | None
    adjusted_epv: float | None
    original_graham: float | None
    adjusted_intrinsic_mid: float | None
    adjusted_margin_of_safety: float | None
    adjusted_value_floored: bool | None
    current_price: float | None
    average_coverage: float | None
    high_priority_unresolved: int | None
    hypotheses_confirmed: int | None
    hypotheses_contradicted: int | None
    hypotheses_partially_confirmed: int | None
    hypotheses_inconclusive: int | None


class ResearchAdjustment(BaseModel):
    hypothesis_source: str | None
    hypothesis_claim: str | None
    hypothesis_direction: str | None
    hypothesis_status: str | None
    affected_method: str | None
    adjustment_magnitude: float | None
    adjustment_confidence: str | None
    calibration_detail: str | None


class ResearchUnresolved(BaseModel):
    description: str | None
    importance: str | None
    unresolved_reason: str | None
    hypothesis_priority: str | None
    hypothesis_direction: str | None


class ResearchCitation(BaseModel):
    citation_id: str | None
    section: str | None
    excerpt: str
    relevance: str | None
    hypothesis_source: str | None
    source_form_type: str | None
    source_filing_date: str | None
    source_title: str | None
    source_url: str | None


class EvidenceItem(BaseModel):
    evidence_id: str | None
    as_of_date: str | None
    source_type: str | None
    source_title: str | None
    source_url: str | None
    source_published_at: str | None
    excerpt: str | None


class CompanyResearchResponse(BaseModel):
    available: bool
    reason: str | None = None
    as_of_date: str | None = None
    created_at: str | None = None
    status: str | None = None
    conviction_class: str | None = None
    conviction_score: float | None = None
    gate_action: str | None = None
    tension_type: str | None = None
    methods_agree: bool | None = None
    consensus_strength: int | None = None
    method_count: int | None = None
    hypotheses_generated: int | None = None
    thesis: ThesisBlock | None = None
    adjustments: list[ResearchAdjustment] = []
    unresolved: list[ResearchUnresolved] = []
    analyst_notes: AnalystNotes | None = None
    citations: list[ResearchCitation] = []
    report_path: str | None = None
    report_available: bool = False
    evidence: list[EvidenceItem] = []
    generated_at: str


class CompanyDossierResponse(BaseModel):
    available: bool
    reason: str | None = None
    run_label: str | None = None
    others: int = 0
    path: str | None = None
    title: str | None = None
    html: str | None = None
    toc: list[TocEntry] = []
    claims: list[ArtifactClaim] | None = None
    size: int | None = None
    mtime: str | None = None
    generated_at: str


class DispositionRow(BaseModel):
    id: int
    kind: str | None
    status: str | None
    opened_at: str | None
    opened_by: str | None
    decided_at: str | None
    operator: str | None
    reason_code: str | None
    rationale: str | None
    intended_size: str | None
    sizing_rationale: str | None
    pre_mortem: str | None
    trigger: dict
    event_id: int | None
    journal_command: str | None


class TickerOutcomeRow(BaseModel):
    as_of_date: str | None
    decision: str | None
    conviction: int | None
    grade: str | None
    outcome_status: str | None
    close_date: str | None
    horizon_days: int | None
    entry_price: float | None
    realized_return_pct: float | None
    benchmark_return_pct: float | None
    excess_return_pct: float | None
    reached_buy_target: bool | None


class CompanyDecisionsResponse(BaseModel):
    dispositions: list[DispositionRow]
    outcomes: list[TickerOutcomeRow]
    outcomes_total: int
    outcomes_open: int
    outcomes_closed: int
    generated_at: str


class MethodScore(BaseModel):
    method: str
    n: int
    correct: int
    incorrect: int
    inconclusive: int
    hit_rate: float | None  # correct / (correct + incorrect); None if unresolved
    unadjusted_source: bool


class MethodMonthly(BaseModel):
    month: str
    method: str
    n: int
    correct: int
    incorrect: int


class MethodScoreboard(BaseModel):
    methods: list[MethodScore]
    monthly: list[MethodMonthly]


class ReturnStats(BaseModel):
    n: int
    avg_excess: float | None
    median_excess: float | None
    hit_rate: float | None
    avg_realized: float | None


class HistogramBin(BaseModel):
    low: float | None
    high: float | None
    count: int


class DecisionStats(ReturnStats):
    decision: str


class ConvictionStats(ReturnStats):
    conviction: int


class RealizedReturns(BaseModel):
    overall: ReturnStats
    histogram: list[HistogramBin]
    by_decision: list[DecisionStats]
    by_conviction: list[ConvictionStats]


class JournalEntry(BaseModel):
    id: int
    ticker: str
    kind: str | None
    status: str | None
    opened_at: str | None
    decided_at: str | None
    operator: str | None
    reason_code: str | None
    rationale: str | None
    intended_size: str | None
    sizing_rationale: str | None


class JournalBlock(BaseModel):
    entries: list[JournalEntry]
    decided_count: int
    open_count: int
    goal: int
    goal_deadline: str


class CalibrationOverall(BaseModel):
    n: int | None
    hit_rate: float | None
    excess_hit_rate: float | None
    avg_return: float | None
    avg_excess: float | None
    median_return: float | None
    target_hit_rate: float | None


class CalibrationReport(BaseModel):
    run_id: str | None
    as_of_date: str | None
    created_at: str | None
    report_path: str | None
    report_family: str | None
    headline_hit_metric: str | None
    overall: CalibrationOverall
    by_grade: dict
    by_status: dict


class OutcomesResponse(BaseModel):
    scoreboard: MethodScoreboard
    returns: RealizedReturns
    journal: JournalBlock
    calibration: list[CalibrationReport]
    generated_at: str
