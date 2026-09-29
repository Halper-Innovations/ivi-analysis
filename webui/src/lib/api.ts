/** Typed fetchers for IVI's read-only API. */

export interface MetaResponse {
  app: string;
  engine_db_path: string;
  engine_db_present: boolean;
  runs_indexed: number;
  watchlist_rows: number | null;
  generated_at: string;
}

export interface OfflineDetail {
  precondition: string;
  detail: string;
}

export class OfflineError extends Error {
  precondition: string;
  detail: string;
  constructor(offline: OfflineDetail) {
    super(`${offline.precondition}: ${offline.detail}`);
    this.precondition = offline.precondition;
    this.detail = offline.detail;
  }
}

export class HttpError extends Error {
  status: number;

  constructor(path: string, status: number) {
    super(`${path} failed: ${status}`);
    this.name = "HttpError";
    this.status = status;
  }
}

export interface WatchlistRow {
  id: number;
  ticker: string;
  status: string | null;
  presented_status: string | null;
  event_pending: string | null;
  conviction_grade: string | null;
  confidence: string | null;
  conviction_source: string | null;
  pipeline_version: string | null;
  candidate_disposition: string | null;
  decision_basis: string | null;
  selection_validation_status: string | null;
  price_trigger_eligible: boolean;
  scan_family: string | null;
  latest_price: number | null;
  latest_price_source: string | null;
  latest_price_checked_at: string | null;
  buy_price_target: number | null;
  distance_from_buy_pct: number | null;
  valuation_anchor_method: string | null;
  valuation_anchor_value: number | null;
  source_sector: string | null;
  status_reason: string | null;
  added_at: string | null;
  last_evaluated_at: string | null;
  falsifiers: string[];
  market_cap_mm: number | null;
  cap_source: string | null;
  cap_band: string | null;
  cap_band_label: string;
  adv_dollar_20d: number | null;
  adv_dollar_60d: number | null;
  adv_asof: string | null;
  capacity_class: string;
  cheapness: string;
}

export interface WatchlistResponse {
  rows: WatchlistRow[];
  total: number;
  generated_at: string;
}

export interface SearchResult {
  ticker: string;
  name: string | null;
  sector: string | null;
  covered: boolean;
  presented_status: string | null;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  generated_at: string;
}

export interface HealthCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface HealthBlock {
  state: "GREEN" | "AMBER" | "RED" | string;
  blocking: boolean;
  checks: HealthCheck[];
}

export interface DecisionCard {
  id: number;
  ticker: string;
  kind: string;
  opened_at: string | null;
  opened_by: string | null;
  trigger: Record<string, unknown>;
  pre_mortem: string | null;
  watchlist_id: number | null;
  watchlist_status: string | null;
  conviction_grade: string | null;
  confidence: string | null;
  source_sector: string | null;
  source_state: string | null;
  is_current_actionable: boolean;
  buy_price_target: number | null;
  latest_price: number | null;
  latest_price_checked_at: string | null;
  price_age_hours: number | null;
  distance_from_buy_pct: number | null;
  falsifiers: string[];
  journal_command: string;
}

export interface WaterlineItem {
  watchlist_id: number;
  ticker: string;
  presented_status: string | null;
  conviction_grade: string | null;
  confidence: string | null;
  latest_price: number | null;
  latest_price_checked_at: string | null;
  price_age_hours: number | null;
  price_suspect: boolean;
  buy_price_target: number | null;
  distance_from_buy_pct: number | null;
  valuation_anchor_method: string | null;
  valuation_anchor_value: number | null;
  source_sector: string | null;
  wake: number[];
}

export interface GateBlockedItem {
  watchlist_id: number;
  ticker: string;
  event_pending: string | null;
  conviction_grade: string | null;
  distance_from_buy_pct: number | null;
  latest_price: number | null;
  buy_price_target: number | null;
}

export interface DigestBlock {
  date: string;
  previous_date: string | null;
  path: string;
  html: string;
}

export interface WaterlineDeeper {
  count: number;
  names: { ticker: string; distance_from_buy_pct: number | null }[];
}

export interface TodayResponse {
  health: HealthBlock;
  decisions: DecisionCard[];
  waterline: WaterlineItem[];
  waterline_deeper: WaterlineDeeper;
  gate_blocked: GateBlockedItem[];
  digest: DigestBlock | null;
  generated_at: string;
}

export interface WaccAdjustment {
  code: string;
  delta: number | null;
  reason: string;
}

export interface WaccDetail {
  baseline_wacc: number | null;
  adjusted_wacc: number | null;
  adjustments: WaccAdjustment[];
}

export interface FairValue {
  kind: "band" | "line" | "not_a_price" | "none" | string;
  value: number | null;
  low: number | null;
  base: number | null;
  high: number | null;
}

export interface MethodCard {
  method: string;
  as_of_date: string;
  created_at: string | null;
  status: string | null;
  fair_value: FairValue;
  flags: string[];
  wacc: WaccDetail | null;
  quality_gate_verdict: string | null;
  confidence_class: string | null;
  gate_reason_codes: string[];
  headwinds: string[];
  supports: string[];
  extras: Record<string, unknown>;
}

export interface ApiGaugeShelf {
  method: string;
  label: string;
  value: number | null;
  low: number | null;
  base: number | null;
  high: number | null;
  emphasized: boolean;
}

export interface EvolutionPoint {
  method: string;
  as_of_date: string;
  value: number;
  archived: boolean;
  pre_hardening: boolean;
}

export interface PriceBlock {
  latest: number | null;
  checked_at: string | null;
  source: string | null;
  age_hours: number | null;
  wake: number[];
}

export interface CompanyProfile {
  watchlist_id: number;
  thesis_text: string | null;
  key_risks: string[];
  open_questions: string[];
  valuation_anchor_method: string | null;
  valuation_anchor_value: number | null;
  current_price_at_addition: number | null;
  source_run_id: string | null;
  cap_asof: string | null;
  added_at: string | null;
}

/** Whether the page's valuations are audited, and what to do when they are not. */
export interface CompanyAudit {
  state: "AUDITED" | "NOT_AUDITED" | "NO_VALUATION";
  reason: string | null;
  message: string;
  command: string | null;
}

export interface CompanyResponse {
  ticker: string;
  as_of: string | null;
  watchlist: WatchlistRow | null;
  profile: CompanyProfile | null;
  valuations: MethodCard[];
  shelves: ApiGaugeShelf[];
  evolution: EvolutionPoint[];
  /** Points left off the chart: written before a split whose factor is not recorded cleanly. */
  evolution_dropped_pre_split?: number;
  /** Points divided by a recorded split factor so the chart sits on one share basis. */
  evolution_rebased_points?: number;
  prices: PriceBlock;
  generated_at: string;
  audit?: CompanyAudit | null;
  /** Lineage-free rows (`ivi value`): reference only, never decision-eligible. */
  research_valuations?: MethodCard[];
}

export interface TtmValue {
  value: number;
  basis: "ttm" | "fy" | "quarter_end" | string;
  period_end: string;
  reason: string;
}

export interface FundamentalsFy {
  basis: "fy";
  as_of: string | null;
  units: string;
  fiscal_years: number[];
  series: Record<string, (number | null)[]>;
  derived: Record<string, (number | null)[]>;
}

export interface FundamentalsTtm {
  basis: "ttm";
  as_of: string | null;
  available: boolean;
  reason: string | null;
  values: Record<string, TtmValue>;
}

export interface NoteCitation {
  section: string;
  excerpt: string;
}

export interface AnalystNote {
  claim: string;
  direction: string | null;
  severity: string | null;
  suggested_adjustment: string | null;
  validation_status: string | null;
  citations: NoteCitation[];
}

export interface AnalystNotes {
  positives: AnalystNote[];
  risks: AnalystNote[];
  surprises: AnalystNote[];
  adjustment_triggers: AnalystNote[];
  overall_assessment: string | null;
  filing_sections_read: string[];
}

export interface ThesisBlock {
  original_dcf: number | null;
  adjusted_dcf: number | null;
  original_epv: number | null;
  adjusted_epv: number | null;
  original_graham: number | null;
  adjusted_intrinsic_mid: number | null;
  adjusted_margin_of_safety: number | null;
  adjusted_value_floored: boolean | null;
  current_price: number | null;
  average_coverage: number | null;
  high_priority_unresolved: number | null;
  hypotheses_confirmed: number | null;
  hypotheses_contradicted: number | null;
  hypotheses_partially_confirmed: number | null;
  hypotheses_inconclusive: number | null;
}

export interface ResearchAdjustment {
  hypothesis_source: string | null;
  hypothesis_claim: string | null;
  hypothesis_direction: string | null;
  hypothesis_status: string | null;
  affected_method: string | null;
  adjustment_magnitude: number | null;
  adjustment_confidence: string | null;
  calibration_detail: string | null;
}

export interface ResearchUnresolved {
  description: string | null;
  importance: string | null;
  unresolved_reason: string | null;
  hypothesis_priority: string | null;
  hypothesis_direction: string | null;
}

export interface ResearchCitation {
  citation_id: string | null;
  section: string | null;
  excerpt: string;
  relevance: string | null;
  hypothesis_source: string | null;
  source_form_type: string | null;
  source_filing_date: string | null;
  source_title: string | null;
  source_url: string | null;
}

export interface EvidenceItem {
  evidence_id: string | null;
  as_of_date: string | null;
  source_type: string | null;
  source_title: string | null;
  source_url: string | null;
  source_published_at: string | null;
  excerpt: string | null;
}

export interface CompanyResearchResponse {
  available: boolean;
  reason: string | null;
  as_of_date: string | null;
  created_at: string | null;
  status: string | null;
  conviction_class: string | null;
  conviction_score: number | null;
  gate_action: string | null;
  tension_type: string | null;
  methods_agree: boolean | null;
  consensus_strength: number | null;
  method_count: number | null;
  hypotheses_generated: number | null;
  thesis: ThesisBlock | null;
  adjustments: ResearchAdjustment[];
  unresolved: ResearchUnresolved[];
  analyst_notes: AnalystNotes | null;
  citations: ResearchCitation[];
  report_path: string | null;
  report_available: boolean;
  evidence: EvidenceItem[];
  generated_at: string;
}

export interface CompanyDossierResponse {
  available: boolean;
  reason: string | null;
  run_label: string | null;
  others: number;
  path: string | null;
  title: string | null;
  html: string | null;
  toc: TocEntry[];
  claims: ArtifactClaim[] | null;
  size: number | null;
  mtime: string | null;
  generated_at: string;
}

export interface DispositionRow {
  id: number;
  kind: string | null;
  status: string | null;
  opened_at: string | null;
  opened_by: string | null;
  decided_at: string | null;
  operator: string | null;
  reason_code: string | null;
  rationale: string | null;
  intended_size: string | null;
  sizing_rationale: string | null;
  pre_mortem: string | null;
  trigger: Record<string, unknown>;
  event_id: number | null;
  journal_command: string | null;
}

export interface TickerOutcomeRow {
  as_of_date: string | null;
  decision: string | null;
  conviction: number | null;
  grade: string | null;
  outcome_status: string | null;
  close_date: string | null;
  horizon_days: number | null;
  entry_price: number | null;
  realized_return_pct: number | null;
  benchmark_return_pct: number | null;
  excess_return_pct: number | null;
  reached_buy_target: boolean | null;
}

export interface CompanyDecisionsResponse {
  dispositions: DispositionRow[];
  outcomes: TickerOutcomeRow[];
  outcomes_total: number;
  outcomes_open: number;
  outcomes_closed: number;
  generated_at: string;
}

export interface MethodScore {
  method: string;
  n: number;
  correct: number;
  incorrect: number;
  inconclusive: number;
  hit_rate: number | null;
  unadjusted_source: boolean;
}

export interface MethodMonthly {
  month: string;
  method: string;
  n: number;
  correct: number;
  incorrect: number;
}

export interface MethodScoreboard {
  methods: MethodScore[];
  monthly: MethodMonthly[];
}

export interface ReturnStats {
  n: number;
  avg_excess: number | null;
  median_excess: number | null;
  hit_rate: number | null;
  avg_realized: number | null;
}

export interface HistogramBin {
  low: number | null;
  high: number | null;
  count: number;
}

export interface DecisionStats extends ReturnStats {
  decision: string;
}

export interface ConvictionStats extends ReturnStats {
  conviction: number;
}

export interface RealizedReturns {
  overall: ReturnStats;
  histogram: HistogramBin[];
  by_decision: DecisionStats[];
  by_conviction: ConvictionStats[];
}

export interface JournalEntry {
  id: number;
  ticker: string;
  kind: string | null;
  status: string | null;
  opened_at: string | null;
  decided_at: string | null;
  operator: string | null;
  reason_code: string | null;
  rationale: string | null;
  intended_size: string | null;
  sizing_rationale: string | null;
}

export interface JournalBlock {
  entries: JournalEntry[];
  decided_count: number;
  open_count: number;
  goal: number;
  goal_deadline: string;
}

export interface CalibrationOverall {
  n: number | null;
  hit_rate: number | null;
  excess_hit_rate: number | null;
  avg_return: number | null;
  avg_excess: number | null;
  median_return: number | null;
  target_hit_rate: number | null;
}

export interface CalibrationReport {
  run_id: string | null;
  as_of_date: string | null;
  created_at: string | null;
  report_path: string | null;
  report_family: string | null;
  headline_hit_metric: string | null;
  overall: CalibrationOverall;
  by_grade: Record<string, unknown>;
  by_status: Record<string, unknown>;
}

export interface OutcomesResponse {
  scoreboard: MethodScoreboard;
  returns: RealizedReturns;
  journal: JournalBlock;
  calibration: CalibrationReport[];
  generated_at: string;
}

export interface RunSummary {
  run_id: string;
  slug: string | null;
  path: string;
  kind: string;
  contract_version: string | null;
  pipeline_version: string | null;
  sector: string | null;
  market_cap_focus: string | null;
  scan_family: string | null;
  as_of_date: string | null;
  created_at: string | null;
  completed_at: string | null;
  status: string | null;
  execution_status: string | null;
  decision_status: string | null;
  final_verdict: string | null;
  selected_ticker: string | null;
  no_selection_reason: string | null;
  examined_count: number | null;
  disposition_counts_json: string | null;
  cost_microdollars: number | null;
  cost_usd: number | null;
  report_path: string | null;
  parse_error: string | null;
  integrity_status: string;
  decision_eligible: boolean;
  indexed_at: string;
}

export interface RunsResponse {
  runs: RunSummary[];
  total: number;
  generated_at: string;
}

export interface RunFunnelStage {
  key: string;
  label: string;
  count: number;
  tickers: string[];
}

export interface RunGate {
  rule_id: string;
  status: string;
  applicable: boolean;
  observed_value: string | null;
  threshold: string | null;
  reason_code: string | null;
  evidence_url: string | null;
  notes: string[];
}

export interface RunCandidate {
  ticker: string;
  primary_ticker: string | null;
  terminal_state: string;
  last_completed_stage: string | null;
  scope_status: string | null;
  screen_status: string | null;
  review_status: string | null;
  reason_codes: string[];
  security_type: string | null;
  is_adr: boolean | null;
  is_secondary_class: boolean | null;
  frontier_status: string | null;
  frontier_dominated_by: string | null;
  underwriting_verdict: string | null;
  underwriting_confidence: string | null;
  watchlist_eligible: boolean | null;
  gates: RunGate[];
  failed_gates: number;
}

export interface RunRankingRow {
  rank: number | null;
  ticker: string;
  audit_status: string | null;
  actionable: boolean | null;
  buy_candidate: boolean | null;
  buy_candidate_reason: string | null;
  hard_blockers: string[];
  cross_sectional_rank: number | null;
  cross_sectional_percentile: number | null;
  best_base_annualized_return: number | null;
  positioning_summary: string | null;
}

export interface RunPacket {
  ticker: string;
  blockers: string[];
  data_quality_status: string | null;
  financial_status: string | null;
  model_fit_status: string | null;
  market_cap_mm: number | null;
  market_cap_category: string | null;
  anchor_method: string | null;
  current_price: number | null;
  discount_to_anchor: number | null;
}

export interface RunSelection {
  source: string | null;
  requested_tickers: string[];
  excluded_tickers: string[];
  loaded_tickers: string[];
  selected_tickers: string[];
  admitted_tickers: string[];
  warnings: string[];
}

export interface RunDecision {
  status: string | null;
  execution_status: string | null;
  decision_status: string | null;
  final_verdict: string | null;
  selected_ticker: string | null;
  no_selection_reason: string | null;
  confidence: string | null;
  confidence_cap_reasons: string[];
  memo_present: boolean;
}

export interface RunLane {
  lane: string;
  cost_microdollars: number | null;
  cost_usd: number | null;
  max_cost_microdollars: number | null;
  max_cost_usd: number | null;
  tool_call_attempts: number | null;
  provider_call_attempts: number | null;
  input_tokens: number | null;
  cached_input_tokens: number | null;
  output_tokens: number | null;
}

export interface RunProvider {
  provider: string;
  model: string;
  calls: number;
  ok_calls: number;
  failed_calls: number;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  cost_estimate_usd: number;
}

export interface RunCosts {
  available: boolean;
  aggregate_cost_microdollars: number | null;
  aggregate_cost_usd: number | null;
  lanes: RunLane[];
  providers: RunProvider[];
}

export interface RunWatchlistLink {
  watchlist_id: number;
  ticker: string;
  status: string | null;
  conviction_grade: string | null;
}

export interface RunDetailResponse {
  summary: RunSummary;
  quarantined: boolean;
  parse_error: string | null;
  funnel: RunFunnelStage[];
  candidates: RunCandidate[];
  ranking: RunRankingRow[];
  packets: RunPacket[];
  selection: RunSelection;
  decision: RunDecision;
  costs: RunCosts;
  watchlist_rows: RunWatchlistLink[];
  report_available: boolean;
  generated_at: string;
}

export interface RunReportResponse {
  run_id: string;
  slug: string;
  path: string;
  bytes: number;
  html: string;
  integrity_status: string;
  decision_eligible: boolean;
  generated_at: string;
}

export interface SweepGroup {
  week: string;
  week_start: string;
  week_end: string;
  band: string;
  runs: number;
  sectors: string[];
  scan_families: string[];
  pipelines: string[];
  verdicts: Record<string, number>;
  cost_microdollars: number | null;
  cost_usd: number | null;
  cost_known_runs: number;
  watchlist_rows: number;
}

export interface SweepsResponse {
  groups: SweepGroup[];
  generated_at: string;
}

export interface CoverageCell {
  sector: string;
  band: string;
  tickers: number;
  last_loaded_at: string | null;
  age_days: number | null;
  run_count: number;
  pipelines: string[];
}

export interface CoverageResponse {
  sectors: string[];
  bands: string[];
  cells: CoverageCell[];
  generated_at: string;
}

export interface CoverageCellRun {
  run_id: string;
  slug: string | null;
  source: string;
  pipeline_version: string | null;
  loaded_at: string | null;
  tickers: number;
}

export interface CoverageCellDetail {
  sector: string;
  band: string;
  tickers: string[];
  runs: CoverageCellRun[];
  generated_at: string;
}

export interface EventFiling {
  accession: string;
  form_type: string | null;
  filing_date: string | null;
  role: string | null;
}

export interface EventCard {
  id: number;
  lane: "opportunity" | "queue_protection" | string;
  event_type: string;
  status: string;
  company_name: string | null;
  ticker: string | null;
  ticker_state: string | null;
  cik: string;
  detection_date: string | null;
  qualification_date: string | null;
  age_days: number | null;
  expiry_reason: string | null;
  detail: Record<string, unknown>;
  filings: EventFiling[];
  dispose_command: string | null;
}

export interface EventsColumn {
  status: string;
  total: number;
  cards: EventCard[];
}

export interface EventsLane {
  lane: string;
  total: number;
  open_total: number;
  type_counts: Record<string, number>;
  columns: EventsColumn[];
}

export interface ScanDay {
  date: string;
  status: string;
  mode: string | null;
  index_rows: number | null;
  candidate_rows: number | null;
  events_created: number | null;
  events_updated: number | null;
}

export interface ScanStrip {
  days: ScanDay[];
  gap_count: number;
  window_business_days: number;
}

export interface EventsResponse {
  lanes: EventsLane[];
  unknown_ticker_total: number;
  scan_strip: ScanStrip;
  generated_at: string;
}

export interface HeartbeatStep {
  step: string;
  status: string;
  exit_code: number | null;
  detail: string | null;
  recorded_at: string | null;
}

export interface HeartbeatDay {
  run_date: string;
  status: "COMPLETE" | "FAILED" | "STARTED" | "MISSING" | string;
  failed_steps: string[];
  steps: HeartbeatStep[];
  log_name: string | null;
}

export interface HeartbeatRow {
  heartbeat: string;
  days: HeartbeatDay[];
}

export interface HeartbeatsResponse {
  dates: string[];
  heartbeats: HeartbeatRow[];
  generated_at: string;
}

export interface BackupLog {
  log_name: string;
  date: string | null;
  size: number;
}

export interface BackupsResponse {
  ok: boolean;
  detail: string;
  ceiling_days: number;
  age_days: number | null;
  status: string | null;
  status_detail: string | null;
  date: string | null;
  duration_s: number | null;
  finished_at: string | null;
  history: BackupLog[];
  generated_at: string;
}

export interface CostWeek {
  week: string;
  scan_cost_usd: number;
  scan_runs: number;
  research_cost_usd: number;
  research_calls: number;
  total_usd: number;
}

export interface CostBucket {
  key: string;
  cost_usd: number;
  runs: number | null;
  calls: number | null;
  cached_calls: number | null;
}

export interface CostsResponse {
  weeks: CostWeek[];
  by_sector: CostBucket[];
  by_band: CostBucket[];
  by_model: CostBucket[];
  runs_with_cost: number;
  runs_without_cost: number;
  total_scan_usd: number;
  total_research_usd: number;
  generated_at: string;
}

export interface OpsHealthResponse {
  health: HealthBlock;
  generated_at: string;
}

export interface LogFile {
  name: string;
  size: number;
  mtime: string;
}

export interface LogsResponse {
  files: LogFile[];
  generated_at: string;
}

export interface LogTailResponse {
  name: string;
  lines: string[];
  truncated: boolean;
  size: number;
  mtime: string;
  generated_at: string;
}

export interface DeadLetterRow {
  id: number;
  job_id: number | null;
  job_type: string | null;
  attempts: number | null;
  error_type: string | null;
  error_message: string | null;
  moved_at: string | null;
}

export interface ResearchGapRow {
  ticker: string;
  as_of_date: string | null;
  run_id: string | null;
  recency_days_min: number | null;
  item_count_30d: number | null;
  risk_flags: string[];
  freshness_bucket: string;
  flow_bucket: string;
  created_at: string | null;
}

export interface LegacyRunRow {
  run_id: string;
  as_of_date: string | null;
  generated_at: string | null;
  status: string | null;
}

export interface ConsolesResponse {
  backlog_size: number;
  deadletters: DeadLetterRow[];
  research_gaps: ResearchGapRow[];
  legacy_runs: LegacyRunRow[];
  generated_at: string;
}

export interface ReaderItem {
  family: string;
  title: string;
  path: string;
  mtime: string;
  meta: Record<string, unknown>;
}

export interface ReaderFamily {
  family: string;
  label: string;
  total: number;
  items: ReaderItem[];
}

export interface ReaderLibraryResponse {
  families: ReaderFamily[];
  generated_at: string;
}

export interface TocEntry {
  level: number;
  text: string;
  anchor: string;
}

export interface ClaimCitation {
  source_url: string;
  snippet: string;
  section_label: string;
}

export interface ArtifactClaim {
  claim_id: string;
  label: string;
  value: number | string | boolean | null;
  unit: string | null;
  citations: ClaimCitation[];
}

export interface ReaderArtifactResponse {
  path: string;
  title: string;
  html: string;
  toc: TocEntry[];
  claims: ArtifactClaim[] | null;
  size: number;
  mtime: string;
  integrity_status: string;
  decision_eligible: boolean;
  generated_at: string;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (response.status === 503) {
    const body = (await response.json()) as { detail: OfflineDetail };
    throw new OfflineError(body.detail);
  }
  if (!response.ok) {
    throw new HttpError(path, response.status);
  }
  return (await response.json()) as T;
}

export function fetchMeta(): Promise<MetaResponse> {
  return getJson<MetaResponse>("/api/meta");
}

export function fetchToday(): Promise<TodayResponse> {
  return getJson<TodayResponse>("/api/today");
}

export function fetchWatchlist(): Promise<WatchlistResponse> {
  return getJson<WatchlistResponse>("/api/watchlist?limit=2000");
}

export function fetchSearch(q: string, limit = 20): Promise<SearchResponse> {
  return getJson<SearchResponse>(
    `/api/search?q=${encodeURIComponent(q)}&limit=${limit}`,
  );
}

export function fetchCompany(ticker: string, asOf?: string): Promise<CompanyResponse> {
  const query = asOf ? `?as_of=${encodeURIComponent(asOf)}` : "";
  return getJson<CompanyResponse>(`/api/company/${encodeURIComponent(ticker)}${query}`);
}

export function fetchFundamentals(
  ticker: string,
  basis: "fy" | "ttm",
  asOf?: string,
): Promise<FundamentalsFy | FundamentalsTtm> {
  const params = new URLSearchParams({ basis });
  if (asOf) params.set("as_of", asOf);
  return getJson<FundamentalsFy | FundamentalsTtm>(
    `/api/company/${encodeURIComponent(ticker)}/fundamentals?${params.toString()}`,
  );
}

export function fetchCompanyResearch(ticker: string): Promise<CompanyResearchResponse> {
  return getJson<CompanyResearchResponse>(
    `/api/company/${encodeURIComponent(ticker)}/research`,
  );
}

export function fetchCompanyDossier(ticker: string): Promise<CompanyDossierResponse> {
  return getJson<CompanyDossierResponse>(
    `/api/company/${encodeURIComponent(ticker)}/dossier`,
  );
}

export function fetchCompanyDecisions(ticker: string): Promise<CompanyDecisionsResponse> {
  return getJson<CompanyDecisionsResponse>(
    `/api/company/${encodeURIComponent(ticker)}/decisions`,
  );
}

export function fetchRuns(includeHistory = false): Promise<RunsResponse> {
  const history = includeHistory ? "&include_history=true" : "";
  return getJson<RunsResponse>(`/api/runs?limit=1000${history}`);
}

/** `ref` is a path slug ("root/leaf") or a bare run_id — pass through raw. */
export function fetchRunDetail(ref: string): Promise<RunDetailResponse> {
  return getJson<RunDetailResponse>(`/api/runs/${ref}`);
}

export function fetchRunReport(ref: string): Promise<RunReportResponse> {
  return getJson<RunReportResponse>(`/api/runs/${ref}/report`);
}

export function fetchSweeps(): Promise<SweepsResponse> {
  return getJson<SweepsResponse>("/api/sweeps");
}

export function fetchCoverage(): Promise<CoverageResponse> {
  return getJson<CoverageResponse>("/api/coverage");
}

export function fetchCoverageCell(
  sector: string,
  band: string,
): Promise<CoverageCellDetail> {
  return getJson<CoverageCellDetail>(
    `/api/coverage/cell?sector=${encodeURIComponent(sector)}&band=${encodeURIComponent(band)}`,
  );
}

export function fetchEvents(filters: {
  lane?: string;
  type?: string;
  q?: string;
}): Promise<EventsResponse> {
  const params = new URLSearchParams();
  if (filters.lane) params.set("lane", filters.lane);
  if (filters.type) params.set("type", filters.type);
  if (filters.q) params.set("q", filters.q);
  const suffix = params.toString();
  return getJson<EventsResponse>(`/api/events${suffix ? `?${suffix}` : ""}`);
}

export function fetchOutcomes(): Promise<OutcomesResponse> {
  return getJson<OutcomesResponse>("/api/outcomes");
}

export function fetchOpsHealth(): Promise<OpsHealthResponse> {
  return getJson<OpsHealthResponse>("/api/ops/health");
}

export function fetchHeartbeats(days = 14): Promise<HeartbeatsResponse> {
  return getJson<HeartbeatsResponse>(`/api/ops/heartbeats?days=${days}`);
}

export function fetchBackups(): Promise<BackupsResponse> {
  return getJson<BackupsResponse>("/api/ops/backups");
}

export function fetchCosts(): Promise<CostsResponse> {
  return getJson<CostsResponse>("/api/ops/costs");
}

export function fetchLogs(): Promise<LogsResponse> {
  return getJson<LogsResponse>("/api/ops/logs");
}

export function fetchLogTail(name: string, tail = 200): Promise<LogTailResponse> {
  return getJson<LogTailResponse>(
    `/api/ops/logs/${encodeURIComponent(name)}?tail=${tail}`,
  );
}

export function fetchConsoles(): Promise<ConsolesResponse> {
  return getJson<ConsolesResponse>("/api/ops/consoles");
}

export function fetchReaderLibrary(): Promise<ReaderLibraryResponse> {
  return getJson<ReaderLibraryResponse>("/api/reader/library");
}

export function fetchReaderArtifact(path: string): Promise<ReaderArtifactResponse> {
  return getJson<ReaderArtifactResponse>(
    `/api/reader/artifact?path=${encodeURIComponent(path)}`,
  );
}
