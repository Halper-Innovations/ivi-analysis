from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import AppConfig, assert_data_volume, ensure_directories, get_config
from app.logging import get_logger


logger = get_logger(__name__)


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    cik TEXT NOT NULL,
    name TEXT,
    ir_rss_url TEXT,
    homepage_url TEXT,
    allowlist_domains TEXT,
    notes TEXT,
    universe_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cik TEXT NOT NULL,
    ticker TEXT,
    accession TEXT NOT NULL,
    form_type TEXT NOT NULL,
    filing_date TEXT,
    period_end TEXT,
    primary_doc_url TEXT NOT NULL,
    local_path TEXT,
    hash TEXT,
    ingested_as_of TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(cik, accession)
);

CREATE TABLE IF NOT EXISTS extracted_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id INTEGER NOT NULL,
    fact_type TEXT NOT NULL,
    value_json TEXT NOT NULL,
    source_url TEXT NOT NULL,
    snippet TEXT,
    section_label TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(filing_id) REFERENCES filings(id)
);

CREATE TABLE IF NOT EXISTS financials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id INTEGER NOT NULL,
    statement_type TEXT NOT NULL,
    line_item TEXT NOT NULL,
    value REAL,
    units TEXT,
    period TEXT,
    source_url TEXT,
    snippet TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(filing_id) REFERENCES filings(id)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    quality_flags_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS valuations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    method TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_run_id TEXT,
    source_artifact_path TEXT,
    source_artifact_sha256 TEXT,
    financial_integrity_fingerprint TEXT,
    UNIQUE(ticker, as_of_date, method)
);

CREATE TABLE IF NOT EXISTS evidence_packets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    packet_path TEXT NOT NULL,
    packet_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    error_type TEXT,
    last_error TEXT,
    cancelled_at TEXT,
    cancel_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyst_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    output_type TEXT NOT NULL,
    output_path TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date, output_type)
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT,
    subscores_json TEXT NOT NULL,
    total_score REAL NOT NULL,
    decision TEXT NOT NULL,
    is_candidate INTEGER NOT NULL DEFAULT 0,
    is_publishable INTEGER NOT NULL DEFAULT 0,
    candidate_run_id TEXT,
    reasons_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT,
    memo_path TEXT NOT NULL,
    manifest_path TEXT,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS universe_snapshots (
    universe_id TEXT PRIMARY KEY,
    snapshot_hash TEXT NOT NULL,
    source_path TEXT NOT NULL,
    ticker_count INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS universe_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    universe_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    cik TEXT NOT NULL,
    name TEXT,
    ir_rss_url TEXT,
    homepage_url TEXT,
    allowlist_domains TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(universe_id, ticker)
);

CREATE TABLE IF NOT EXISTS price_quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    provider TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    price REAL,
    currency TEXT,
    price_basis TEXT,
    split_adjustment_factor REAL,
    split_effective_date TEXT,
    source_url TEXT,
    status TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    quote_hash TEXT NOT NULL,
    UNIQUE(ticker, provider, as_of_date)
);

CREATE TABLE IF NOT EXISTS run_manifests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    as_of_date TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dead_letter_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    job_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT,
    moved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    adapter_run_id TEXT,
    source_type TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_title TEXT,
    source_published_at TEXT,
    retrieved_at TEXT NOT NULL,
    excerpt_text TEXT NOT NULL,
    excerpt_hash TEXT NOT NULL,
    content_hash TEXT,
    dedupe_key TEXT,
    citations_json TEXT NOT NULL,
    derived_from_json TEXT NOT NULL DEFAULT '[]',
    item_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_packets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    packet_path TEXT NOT NULL,
    packet_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date, run_id)
);

CREATE TABLE IF NOT EXISTS research_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    recency_days_min INTEGER,
    item_count_30d INTEGER NOT NULL DEFAULT 0,
    has_earnings_release INTEGER NOT NULL DEFAULT 0,
    has_investor_presentation INTEGER NOT NULL DEFAULT 0,
    sentiment_flags_json TEXT NOT NULL DEFAULT '[]',
    key_topics_json TEXT NOT NULL DEFAULT '[]',
    evidence_item_ids_json TEXT NOT NULL DEFAULT '[]',
    summary_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date, run_id)
);

CREATE TABLE IF NOT EXISTS filing_coverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    run_id TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    forms_included_json TEXT NOT NULL,
    accession_numbers_json TEXT NOT NULL,
    coverage_score REAL NOT NULL DEFAULT 0,
    missing_required_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE(ticker, run_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS parsed_filings (
    accession TEXT PRIMARY KEY,
    parsed_at TEXT NOT NULL,
    content_hash TEXT
);

CREATE TABLE IF NOT EXISTS ticker_deltas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    run_id TEXT NOT NULL,
    prev_run_id TEXT,
    as_of_date TEXT NOT NULL,
    changed INTEGER NOT NULL DEFAULT 0,
    delta_path TEXT NOT NULL,
    delta_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, run_id)
);

CREATE TABLE IF NOT EXISTS delta_memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    run_id TEXT NOT NULL,
    changed INTEGER NOT NULL DEFAULT 0,
    memo_path TEXT NOT NULL,
    memo_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, run_id)
);

CREATE TABLE IF NOT EXISTS shortlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    shortlist_path TEXT NOT NULL,
    shortlist_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS synthesis_packets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    packet_path TEXT NOT NULL,
    packet_hash TEXT NOT NULL,
    packet_json TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    usage_json TEXT NOT NULL DEFAULT '{}',
    cost_estimate_usd REAL NOT NULL DEFAULT 0,
    paid_invocation_id TEXT,
    from_cache INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date, run_id)
);

CREATE TABLE IF NOT EXISTS synthesis_paid_attempts (
    attempt_id TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL,
    physical_sequence INTEGER NOT NULL CHECK(physical_sequence > 0),
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('OK', 'INCOMPLETE', 'ERROR')),
    usage_json TEXT NOT NULL,
    cost_estimate_usd REAL NOT NULL CHECK(cost_estimate_usd >= 0),
    created_at TEXT NOT NULL,
    UNIQUE(invocation_id, physical_sequence)
);

CREATE TABLE IF NOT EXISTS market_caps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    effective_as_of_date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    run_as_of_date TEXT NOT NULL,
    market_cap REAL,
    market_cap_unit TEXT,
    market_cap_status TEXT NOT NULL,
    price REAL,
    price_currency TEXT,
    price_basis TEXT,
    quote_snapshot_id TEXT,
    shares_outstanding REAL,
    shares_unit TEXT,
    shares_basis TEXT,
    split_adjustment_factor REAL,
    split_effective_date TEXT,
    provider TEXT,
    source_url TEXT,
    fetched_at TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, effective_as_of_date, run_id)
);

CREATE TABLE IF NOT EXISTS discovery_runs (
    run_id TEXT PRIMARY KEY,
    run_as_of_date TEXT NOT NULL,
    seed_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    seed_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    processed_count INTEGER NOT NULL DEFAULT 0,
    phase TEXT NOT NULL DEFAULT 'prefilter',
    tickers_targeted_json TEXT NOT NULL,
    processed_effective_dates_json TEXT NOT NULL DEFAULT '{}',
    stats_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discovery_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    run_id TEXT NOT NULL,
    discovery_score REAL NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    publication_receipt_path TEXT,
    publication_receipt_sha256 TEXT,
    UNIQUE(ticker, run_id)
);

CREATE TABLE IF NOT EXISTS discovery_phase1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    cik TEXT,
    company_name TEXT,
    eligible INTEGER NOT NULL DEFAULT 0,
    prefilter_score REAL NOT NULL DEFAULT 0,
    latest_filing_date TEXT,
    selected_accessions_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, ticker)
);

CREATE TABLE IF NOT EXISTS discovery_lifecycle (
    ticker TEXT PRIMARY KEY,
    first_seen_run_id TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_run_id TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    times_shortlisted INTEGER NOT NULL DEFAULT 0,
    last_discovery_score REAL,
    last_action TEXT
);

CREATE TABLE IF NOT EXISTS discovery_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    discovery_run_id TEXT NOT NULL,
    deep_run_id TEXT,
    run_as_of_date TEXT NOT NULL,
    entry_price REAL,
    price_source TEXT,
    forward_return_30d REAL,
    forward_return_90d REAL,
    forward_return_180d REAL,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, discovery_run_id)
);

CREATE TABLE IF NOT EXISTS ticker_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    discovery_run_id TEXT,
    deep_run_id TEXT,
    decision TEXT NOT NULL,
    conviction INTEGER NOT NULL,
    horizon_days INTEGER NOT NULL,
    thesis_tags_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT,
    outcome_status TEXT NOT NULL DEFAULT 'OPEN',
    close_date TEXT,
    realized_return_pct REAL,
    max_drawdown_pct REAL,
    entry_price REAL,
    entry_price_source TEXT,
    entry_date TEXT,
    grade TEXT,
    status TEXT,
    benchmark_symbol TEXT,
    cap_category TEXT,
    pipeline_version TEXT,
    candidate_disposition TEXT,
    decision_basis TEXT,
    selection_validation_status TEXT,
    source_sector TEXT,
    benchmark_return_pct REAL,
    excess_return_pct REAL,
    source_artifact_path TEXT,
    source_artifact_sha256 TEXT,
    source_decision_fingerprint TEXT,
    financial_integrity_fingerprint TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date, run_id)
);

CREATE TABLE IF NOT EXISTS research_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    run_id TEXT NOT NULL,
    discovery_run_id TEXT,
    as_of_date TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(ticker, run_id)
);

CREATE TABLE IF NOT EXISTS calibration_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    as_of_date TEXT,
    report_json TEXT NOT NULL,
    report_path TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id)
);

CREATE TABLE IF NOT EXISTS sector_inference (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    inferred_sector TEXT,
    score REAL NOT NULL DEFAULT 0,
    derived_from TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS rlm_runs (
    run_id TEXT PRIMARY KEY,
    sector TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    iterations INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT
);

CREATE TABLE IF NOT EXISTS rlm_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    planner_json TEXT NOT NULL,
    critic_json TEXT NOT NULL,
    progress_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, iteration)
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_filings_ticker_date ON filings(ticker, filing_date);
CREATE INDEX IF NOT EXISTS idx_financials_filing_id ON financials(filing_id);
CREATE INDEX IF NOT EXISTS idx_extracted_facts_filing_id ON extracted_facts(filing_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status_scheduled ON jobs(status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_universe_members_universe ON universe_members(universe_id);
CREATE INDEX IF NOT EXISTS idx_price_quotes_lookup ON price_quotes(ticker, provider, as_of_date);
CREATE INDEX IF NOT EXISTS idx_dead_letter_moved_at ON dead_letter_jobs(moved_at);
CREATE INDEX IF NOT EXISTS idx_evidence_items_lookup ON evidence_items(ticker, as_of_date, run_id);
CREATE INDEX IF NOT EXISTS idx_research_packets_lookup ON research_packets(ticker, as_of_date);
CREATE INDEX IF NOT EXISTS idx_research_signals_lookup ON research_signals(ticker, as_of_date, run_id);
CREATE INDEX IF NOT EXISTS idx_filing_coverage_lookup ON filing_coverage(ticker, run_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_ticker_deltas_lookup ON ticker_deltas(ticker, run_id, changed);
CREATE INDEX IF NOT EXISTS idx_synthesis_packets_run ON synthesis_packets(run_id, ticker, as_of_date);
CREATE INDEX IF NOT EXISTS idx_synthesis_packets_hash ON synthesis_packets(input_hash, prompt_hash, model, provider);
CREATE INDEX IF NOT EXISTS idx_synthesis_paid_attempts_run
    ON synthesis_paid_attempts(run_id, ticker, as_of_date);
CREATE INDEX IF NOT EXISTS idx_synthesis_paid_attempts_invocation
    ON synthesis_paid_attempts(invocation_id, physical_sequence);
CREATE TRIGGER IF NOT EXISTS synthesis_paid_attempts_no_update
BEFORE UPDATE ON synthesis_paid_attempts
BEGIN
    SELECT RAISE(ABORT, 'synthesis paid attempts are append-only');
END;
CREATE TRIGGER IF NOT EXISTS synthesis_paid_attempts_no_delete
BEFORE DELETE ON synthesis_paid_attempts
BEGIN
    SELECT RAISE(ABORT, 'synthesis paid attempts are append-only');
END;
CREATE INDEX IF NOT EXISTS idx_market_caps_lookup ON market_caps(ticker, effective_as_of_date, run_id);
CREATE INDEX IF NOT EXISTS idx_discovery_runs_date ON discovery_runs(run_as_of_date, created_at);
CREATE INDEX IF NOT EXISTS idx_discovery_candidates_run_score ON discovery_candidates(run_id, discovery_score DESC, ticker ASC);
CREATE INDEX IF NOT EXISTS idx_discovery_phase1_run_score ON discovery_phase1(run_id, prefilter_score DESC, ticker ASC);
CREATE INDEX IF NOT EXISTS idx_discovery_outcomes_run ON discovery_outcomes(discovery_run_id, ticker);
CREATE INDEX IF NOT EXISTS idx_ticker_outcomes_lookup ON ticker_outcomes(run_id, ticker, outcome_status, updated_at);
CREATE INDEX IF NOT EXISTS idx_research_cycles_lookup ON research_cycles(run_id, ticker, updated_at);
CREATE INDEX IF NOT EXISTS idx_calibration_reports_lookup ON calibration_reports(created_at, run_id);
CREATE INDEX IF NOT EXISTS idx_rlm_runs_status ON rlm_runs(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_rlm_iterations_lookup ON rlm_iterations(run_id, iteration);

CREATE TABLE IF NOT EXISTS deep_research_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    source_as_of_date TEXT NOT NULL,
    source_valuation_id INTEGER,
    source_run_id TEXT,
    source_artifact_path TEXT,
    source_artifact_sha256 TEXT,
    source_valuation_fingerprint TEXT,
    horizon_days INTEGER NOT NULL CHECK (horizon_days > 0),
    target_exit_date TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    entry_price REAL NOT NULL CHECK (entry_price > 0),
    exit_price REAL NOT NULL CHECK (exit_price > 0),
    exit_as_of_date TEXT NOT NULL,
    exit_source TEXT NOT NULL,
    price_change_pct REAL NOT NULL,
    thesis_verdict TEXT NOT NULL
        CHECK (thesis_verdict IN ('UNDERVALUED', 'OVERVALUED', 'FAIRLY_VALUED')),
    verdict_outcome TEXT NOT NULL
        CHECK (verdict_outcome IN ('CORRECT', 'INCORRECT', 'INCONCLUSIVE')),
    conviction_score INTEGER,
    conviction_class TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, source_as_of_date, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_dro_verdict
    ON deep_research_outcomes(horizon_days, verdict_outcome);
CREATE INDEX IF NOT EXISTS idx_dro_ticker_date
    ON deep_research_outcomes(ticker, source_as_of_date);

CREATE TABLE IF NOT EXISTS deep_research_method_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    outcome_id INTEGER NOT NULL
        REFERENCES deep_research_outcomes(id) ON DELETE CASCADE,
    method TEXT NOT NULL CHECK (method IN ('dcf', 'epv', 'graham')),
    predicted_value REAL NOT NULL,
    entry_price REAL NOT NULL CHECK (entry_price > 0),
    direction TEXT NOT NULL
        CHECK (direction IN ('UNDERVALUED', 'OVERVALUED', 'FAIRLY_VALUED')),
    outcome TEXT NOT NULL
        CHECK (outcome IN ('CORRECT', 'INCORRECT', 'INCONCLUSIVE')),
    unadjusted_source INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(outcome_id, method)
);

CREATE INDEX IF NOT EXISTS idx_drmo_method_outcome
    ON deep_research_method_outcomes(method, outcome);

CREATE TABLE IF NOT EXISTS catalyst_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    catalyst_type TEXT NOT NULL,
    signal_label TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    detail_json TEXT NOT NULL DEFAULT '{}',
    source_url TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(ticker, as_of_date, catalyst_type)
);

CREATE INDEX IF NOT EXISTS idx_catalyst_events_lookup
    ON catalyst_events(ticker, as_of_date, catalyst_type);

CREATE TABLE IF NOT EXISTS sec_registrants (
    cik TEXT PRIMARY KEY,
    primary_ticker TEXT NOT NULL,
    all_tickers TEXT NOT NULL DEFAULT '[]',
    name TEXT,
    exchange TEXT,
    exchange_scope TEXT NOT NULL,
    sic INTEGER,
    sic_description TEXT,
    operating_status TEXT NOT NULL,
    in_scope INTEGER NOT NULL DEFAULT 0,
    latest_operating_form_date TEXT,
    sector TEXT,
    classification_status TEXT,
    intake_status TEXT,
    intake_detail TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    removed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sec_registrants_ticker
    ON sec_registrants(primary_ticker);
CREATE INDEX IF NOT EXISTS idx_sec_registrants_scope
    ON sec_registrants(in_scope, operating_status);

-- SIC codes fetched on demand from the SEC submissions JSON for issuers that
-- have no sec_registrants row yet (a fresh install that has not run the
-- registrant intake). Read as a fallback by the shared SIC lookup.
CREATE TABLE IF NOT EXISTS sec_sic_cache (
    cik TEXT PRIMARY KEY,
    sic TEXT NOT NULL,
    sic_description TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS universe_sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    run_kind TEXT NOT NULL,
    action TEXT NOT NULL,
    cik TEXT,
    ticker TEXT,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_universe_sync_log_run
    ON universe_sync_log(run_kind, run_at);

CREATE TABLE IF NOT EXISTS us_equity_census_runs (
    run_id TEXT PRIMARY KEY,
    as_of_date TEXT NOT NULL,
    target_band TEXT NOT NULL DEFAULT 'all'
        CHECK (target_band IN (
            'all', 'large_and_mega', 'mid_cap', 'small_cap', 'micro_cap'
        )),
    status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'RUNNING', 'INTERRUPTED', 'FAILED', 'COMPLETED')),
    current_stage TEXT
        CHECK (current_stage IS NULL OR current_stage IN (
            'MEMBERSHIP',
            'IDENTITY',
            'DOMICILE_LISTING_STATUS',
            'SECURITY_TYPE',
            'PRIMARY_SECURITY_SELECTION',
            'ISSUER_DEDUPLICATION',
            'MARKET_CAP_RESOLUTION',
            'CAP_BAND_ASSIGNMENT',
            'SECTOR_CLASSIFICATION',
            'FACTS_FILINGS_AVAILABILITY',
            'COMPANY_PACKET',
            'TERMINAL_DISPOSITION'
        )),
    acceptance_status TEXT NOT NULL DEFAULT 'NOT_EVALUATED'
        CHECK (acceptance_status IN ('NOT_EVALUATED', 'INCOMPLETE', 'PASSED', 'FAILED')),
    registry_source_name TEXT,
    registry_source_url TEXT,
    registry_snapshot_path TEXT,
    registry_snapshot_sha256 TEXT,
    registry_retrieved_at TEXT,
    source_snapshot_count INTEGER NOT NULL DEFAULT 0
        CHECK (source_snapshot_count >= 0),
    source_manifest_json TEXT NOT NULL DEFAULT '{}',
    crosscheck_sources_json TEXT NOT NULL DEFAULT '[]',
    input_fingerprint TEXT,
    resume_fingerprint TEXT,
    checkpoint_path TEXT,
    discovered_security_count INTEGER NOT NULL DEFAULT 0
        CHECK (discovered_security_count >= 0),
    deduplicated_issuer_count INTEGER NOT NULL DEFAULT 0
        CHECK (deduplicated_issuer_count >= 0),
    us_domiciled_issuer_count INTEGER NOT NULL DEFAULT 0
        CHECK (us_domiciled_issuer_count >= 0),
    foreign_issuer_count INTEGER NOT NULL DEFAULT 0
        CHECK (foreign_issuer_count >= 0),
    unresolved_identity_count INTEGER NOT NULL DEFAULT 0
        CHECK (unresolved_identity_count >= 0),
    unresolved_cap_count INTEGER NOT NULL DEFAULT 0
        CHECK (unresolved_cap_count >= 0),
    unresolved_sector_count INTEGER NOT NULL DEFAULT 0
        CHECK (unresolved_sector_count >= 0),
    status_detail_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    interrupted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_us_equity_census_runs_asof
    ON us_equity_census_runs(as_of_date, target_band, status);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_runs_acceptance
    ON us_equity_census_runs(target_band, acceptance_status, as_of_date);

CREATE TABLE IF NOT EXISTS us_equity_census_issuers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    issuer_key TEXT NOT NULL,
    cik TEXT,
    legal_name TEXT,
    display_name TEXT,
    identity_status TEXT,
    identity_source_provider TEXT,
    identity_source_url TEXT,
    identity_as_of_date TEXT,
    identity_retrieved_at TEXT,
    identity_confidence TEXT,
    issuer_type TEXT,
    operating_structure TEXT,
    operating_status TEXT,
    is_operating_company INTEGER
        CHECK (is_operating_company IS NULL OR is_operating_company IN (0, 1)),
    membership_status TEXT,
    scope_status TEXT,
    scope_reason_code TEXT,
    domicile_country_code TEXT,
    domicile_jurisdiction TEXT,
    is_us_domiciled INTEGER
        CHECK (is_us_domiciled IS NULL OR is_us_domiciled IN (0, 1)),
    is_us_listed_foreign_issuer INTEGER
        CHECK (
            is_us_listed_foreign_issuer IS NULL
            OR is_us_listed_foreign_issuer IN (0, 1)
        ),
    filer_status TEXT,
    domicile_source_provider TEXT,
    domicile_source_url TEXT,
    domicile_as_of_date TEXT,
    domicile_retrieved_at TEXT,
    domicile_confidence TEXT,
    primary_security_key TEXT,
    primary_ticker TEXT,
    primary_exchange_name TEXT,
    primary_exchange_mic TEXT,
    primary_selection_status TEXT,
    primary_selection_method TEXT,
    primary_selection_source_url TEXT,
    primary_selection_as_of_date TEXT,
    primary_selection_retrieved_at TEXT,
    primary_selection_confidence TEXT,
    listed_security_count INTEGER NOT NULL DEFAULT 0
        CHECK (listed_security_count >= 0),
    market_cap_status TEXT,
    market_cap_usd REAL
        CHECK (market_cap_usd IS NULL OR market_cap_usd >= 0),
    market_cap_currency TEXT
        CHECK (market_cap_currency IS NULL OR market_cap_currency = 'USD'),
    market_cap_as_of_date TEXT,
    market_cap_method TEXT,
    market_cap_source_provider TEXT,
    market_cap_source_url TEXT,
    market_cap_retrieved_at TEXT,
    market_cap_confidence TEXT,
    cap_security_key TEXT,
    cap_price_usd REAL
        CHECK (cap_price_usd IS NULL OR cap_price_usd > 0),
    cap_price_as_of_date TEXT,
    cap_price_source_provider TEXT,
    cap_price_source_url TEXT,
    shares_outstanding REAL
        CHECK (shares_outstanding IS NULL OR shares_outstanding >= 0),
    shares_as_of_date TEXT,
    shares_source_provider TEXT,
    shares_source_url TEXT,
    shares_retrieved_at TEXT,
    shares_confidence TEXT,
    cap_ratio_adjustment REAL
        CHECK (cap_ratio_adjustment IS NULL OR cap_ratio_adjustment > 0),
    cap_ratio_source_url TEXT,
    cap_derivation_json TEXT NOT NULL DEFAULT '{}',
    cap_band TEXT
        CHECK (cap_band IS NULL OR cap_band IN (
            'large_and_mega',
            'mid_cap',
            'small_cap',
            'micro_cap',
            'UNRESOLVED',
            'NOT_APPLICABLE'
        )),
    cap_band_status TEXT,
    canonical_sector TEXT,
    source_sector_label TEXT,
    source_sector_system TEXT,
    sector_status TEXT,
    sector_mapping_method TEXT,
    sector_mapping_version TEXT,
    sector_source_url TEXT,
    sector_as_of_date TEXT,
    sector_confidence TEXT,
    sector_provenance_json TEXT NOT NULL DEFAULT '{}',
    facts_status TEXT,
    facts_as_of_date TEXT,
    facts_source_provider TEXT,
    facts_source_url TEXT,
    facts_detail_json TEXT NOT NULL DEFAULT '{}',
    filings_status TEXT,
    latest_annual_filing_date TEXT,
    latest_annual_filing_accession TEXT,
    filings_source_provider TEXT,
    filings_source_url TEXT,
    filings_detail_json TEXT NOT NULL DEFAULT '{}',
    packet_status TEXT,
    packet_as_of_date TEXT,
    packet_path TEXT,
    packet_sha256 TEXT,
    packet_detail_json TEXT NOT NULL DEFAULT '{}',
    last_completed_stage TEXT,
    next_stage TEXT,
    processing_status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (processing_status IN (
            'PENDING', 'IN_PROGRESS', 'NEEDS_DATA', 'COMPLETED', 'FAILED', 'OUT_OF_SCOPE'
        )),
    terminal_disposition TEXT,
    terminal_reason_code TEXT,
    terminal_detail_json TEXT NOT NULL DEFAULT '{}',
    terminal_at TEXT,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, issuer_key),
    UNIQUE(run_id, cik),
    UNIQUE(run_id, primary_security_key),
    FOREIGN KEY(run_id) REFERENCES us_equity_census_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id, primary_security_key)
        REFERENCES us_equity_census_securities(run_id, security_key)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY(run_id, cap_security_key)
        REFERENCES us_equity_census_securities(run_id, security_key)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_us_equity_census_issuers_cik
    ON us_equity_census_issuers(run_id, cik);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_issuers_scope
    ON us_equity_census_issuers(run_id, scope_status, processing_status);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_issuers_cap
    ON us_equity_census_issuers(run_id, cap_band, market_cap_usd);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_issuers_sector
    ON us_equity_census_issuers(run_id, canonical_sector);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_issuers_terminal
    ON us_equity_census_issuers(run_id, terminal_disposition);

CREATE TABLE IF NOT EXISTS us_equity_census_securities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    security_key TEXT NOT NULL,
    source_registry TEXT,
    source_security_id TEXT,
    ticker TEXT,
    listed_name TEXT,
    exchange_name TEXT,
    exchange_mic TEXT,
    listing_status TEXT,
    listing_status_as_of_date TEXT,
    listing_source_url TEXT,
    listing_retrieved_at TEXT,
    issuer_key TEXT,
    issuer_relationship_type TEXT,
    related_security_key TEXT,
    share_class TEXT,
    is_primary_security INTEGER
        CHECK (is_primary_security IS NULL OR is_primary_security IN (0, 1)),
    is_secondary_class INTEGER
        CHECK (is_secondary_class IS NULL OR is_secondary_class IN (0, 1)),
    is_duplicate_listing INTEGER
        CHECK (is_duplicate_listing IS NULL OR is_duplicate_listing IN (0, 1)),
    security_type TEXT,
    security_type_status TEXT,
    is_common_equity INTEGER
        CHECK (is_common_equity IS NULL OR is_common_equity IN (0, 1)),
    is_adr INTEGER
        CHECK (is_adr IS NULL OR is_adr IN (0, 1)),
    adr_ratio REAL
        CHECK (adr_ratio IS NULL OR adr_ratio > 0),
    adr_ratio_as_of_date TEXT,
    adr_ratio_source_url TEXT,
    share_class_ratio REAL
        CHECK (share_class_ratio IS NULL OR share_class_ratio > 0),
    share_class_ratio_as_of_date TEXT,
    share_class_ratio_source_url TEXT,
    price_usd REAL
        CHECK (price_usd IS NULL OR price_usd > 0),
    price_as_of_date TEXT,
    price_source_provider TEXT,
    price_source_url TEXT,
    price_retrieved_at TEXT,
    price_confidence TEXT,
    identity_status TEXT,
    identity_source TEXT,
    identity_source_url TEXT,
    identity_as_of_date TEXT,
    identity_retrieved_at TEXT,
    identity_confidence TEXT,
    last_completed_stage TEXT,
    next_stage TEXT,
    processing_status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (processing_status IN (
            'PENDING', 'IN_PROGRESS', 'NEEDS_DATA', 'COMPLETED', 'FAILED', 'OUT_OF_SCOPE'
        )),
    terminal_disposition TEXT,
    terminal_reason_code TEXT,
    terminal_detail_json TEXT NOT NULL DEFAULT '{}',
    terminal_at TEXT,
    source_snapshot_ref TEXT,
    raw_source_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, security_key),
    UNIQUE(run_id, source_registry, source_security_id),
    FOREIGN KEY(run_id) REFERENCES us_equity_census_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id, issuer_key)
        REFERENCES us_equity_census_issuers(run_id, issuer_key)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY(run_id, related_security_key)
        REFERENCES us_equity_census_securities(run_id, security_key)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_us_equity_census_securities_ticker
    ON us_equity_census_securities(run_id, ticker);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_securities_issuer
    ON us_equity_census_securities(run_id, issuer_key);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_securities_listing
    ON us_equity_census_securities(run_id, listing_status, security_type);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_securities_terminal
    ON us_equity_census_securities(run_id, terminal_disposition);

CREATE TABLE IF NOT EXISTS us_equity_census_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL
        CHECK (attempt_number > 0),
    stage TEXT NOT NULL
        CHECK (stage IN (
            'MEMBERSHIP',
            'IDENTITY',
            'DOMICILE_LISTING_STATUS',
            'SECURITY_TYPE',
            'PRIMARY_SECURITY_SELECTION',
            'ISSUER_DEDUPLICATION',
            'MARKET_CAP_RESOLUTION',
            'CAP_BAND_ASSIGNMENT',
            'SECTOR_CLASSIFICATION',
            'FACTS_FILINGS_AVAILABILITY',
            'COMPANY_PACKET',
            'TERMINAL_DISPOSITION'
        )),
    entity_kind TEXT NOT NULL
        CHECK (entity_kind IN ('RUN', 'SECURITY', 'ISSUER')),
    security_key TEXT,
    issuer_key TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (status IN (
            'PENDING', 'RUNNING', 'COMPLETED', 'NEEDS_DATA',
            'FAILED', 'INTERRUPTED', 'SKIPPED'
        )),
    resume_from_attempt_id TEXT,
    input_fingerprint TEXT,
    checkpoint_path TEXT,
    worker_id TEXT,
    source_provider TEXT,
    source_url TEXT,
    retrieved_at TEXT,
    input_json TEXT NOT NULL DEFAULT '{}',
    output_json TEXT NOT NULL DEFAULT '{}',
    error_code TEXT,
    error_message TEXT,
    detail_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, attempt_id),
    CHECK (
        (entity_kind = 'RUN' AND security_key IS NULL AND issuer_key IS NULL)
        OR (entity_kind = 'SECURITY' AND security_key IS NOT NULL AND issuer_key IS NULL)
        OR (entity_kind = 'ISSUER' AND security_key IS NULL AND issuer_key IS NOT NULL)
    ),
    FOREIGN KEY(run_id) REFERENCES us_equity_census_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id, security_key)
        REFERENCES us_equity_census_securities(run_id, security_key)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY(run_id, issuer_key)
        REFERENCES us_equity_census_issuers(run_id, issuer_key)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY(run_id, resume_from_attempt_id)
        REFERENCES us_equity_census_attempts(run_id, attempt_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_us_equity_census_attempts_status
    ON us_equity_census_attempts(run_id, stage, status);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_attempts_security
    ON us_equity_census_attempts(run_id, security_key, attempt_number);
CREATE INDEX IF NOT EXISTS idx_us_equity_census_attempts_issuer
    ON us_equity_census_attempts(run_id, issuer_key, attempt_number);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn(path: Path, cfg: AppConfig | None = None) -> sqlite3.Connection:
    cfg = cfg or get_config()
    conn = sqlite3.connect(
        path,
        timeout=max(1.0, float(cfg.sqlite_busy_timeout_ms) / 1000.0),
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute(f"PRAGMA busy_timeout={int(cfg.sqlite_busy_timeout_ms)};")
    return conn


def connect(
    db_path: str | Path | None = None,
    *,
    cfg: AppConfig | None = None,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Hardened connection for money-relevant read/write paths.

    Same settings as the get_db factory (WAL, synchronous=NORMAL, foreign
    keys, busy_timeout, Row factory) but callable with an explicit path so
    watchlist/gate/cap modules that accept a ``db_path`` parameter can route
    through it instead of raw ``sqlite3.connect``. Caller owns commit/close.
    With ``read_only=True``, require an existing file and forbid writes without
    changing journal settings or creating directories.
    """
    cfg = cfg or get_config()
    path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    if read_only:
        # Deliberately NOT mount-guarded. A read-only open creates nothing,
        # engine.db lives on the internal disk (owner decision 2026-09-09),
        # and refusing reads would turn an absent drive into a total outage
        # while blinding the checks whose job is to report that absence.
        # Do not create a missing database, migrate it, or change its journal mode.
        conn = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=max(1.0, float(cfg.sqlite_busy_timeout_ms) / 1000.0),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return conn
    # Mount guard: an absent (or swapped) data volume stops the writable
    # path here rather than letting SQLite create an empty shadow database.
    assert_data_volume(cfg)
    return _conn(path, cfg=cfg)


@contextmanager
def get_db(cfg: AppConfig | None = None) -> Iterator[sqlite3.Connection]:
    cfg = cfg or get_config()
    assert_data_volume(cfg)
    ensure_directories(cfg)
    conn = _conn(cfg.db_path, cfg=cfg)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(cfg: AppConfig | None = None, *, conn: sqlite3.Connection | None = None) -> None:
    # Local import avoids executing app.calibration.__init__ while app.db is
    # still being initialized.
    from app.calibration.recommendation_schema import ensure_recommendation_ledger_schema
    from app.holdings import ensure_holdings_schema
    from app.watchlist.schema import apply_watchlist_schema

    if conn is not None:
        conn.executescript(SCHEMA_SQL)
        ensure_recommendation_ledger_schema(conn)
        # The web read model and the read-only CLI paths query these tables
        # directly; they used to be created on first write, so a fresh
        # database answered "no such table" until something wrote a row.
        apply_watchlist_schema(conn)
        ensure_holdings_schema(conn)
        _ensure_schema_evolution(conn)
        return

    cfg = cfg or get_config()
    assert_data_volume(cfg)
    ensure_directories(cfg)
    # closing(), not the bare connection: `with sqlite3.Connection` is a
    # transaction guard that never closes. The leaked handle closed at GC's
    # whim, so the WAL checkpoint landed nondeterministically — a fresh DB's
    # on-disk bytes depended on allocation pressure elsewhere in the process.
    with closing(_conn(cfg.db_path, cfg=cfg)) as conn_local:
        conn_local.executescript(SCHEMA_SQL)
        ensure_recommendation_ledger_schema(conn_local)
        apply_watchlist_schema(conn_local)
        ensure_holdings_schema(conn_local)
        _ensure_schema_evolution(conn_local)
        conn_local.commit()


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    if _has_column(conn, table, column):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def _migrate_companyfacts_constraint(conn: sqlite3.Connection) -> None:
    """Migrate companyfacts_facts from old 3-column UNIQUE to new 4-column UNIQUE.

    SQLite can't ALTER constraints, so we rename → recreate → copy → drop.
    Only runs if the current constraint doesn't include period_type.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='companyfacts_facts'"
    ).fetchone()
    if not row:
        return
    create_sql = row["sql"] or ""
    # Already has the 4-column constraint — nothing to do
    if "period_type, line_item" in create_sql:
        return
    # Old constraint detected — migrate
    conn.execute("ALTER TABLE companyfacts_facts RENAME TO _companyfacts_old")
    conn.execute("""
        CREATE TABLE companyfacts_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL DEFAULT 'FY',
            period_end TEXT NOT NULL,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            source_url TEXT,
            fetched_at TEXT NOT NULL,
            filed_date TEXT,
            form TEXT,
            accession TEXT,
            UNIQUE(ticker, fiscal_year, period_type, line_item)
        )
    """)
    conn.execute("""
        INSERT INTO companyfacts_facts
            (
                id, ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form, accession
            )
        SELECT
            id, ticker, fiscal_year, COALESCE(period_type, 'FY'), period_end,
            line_item, value, units, source_url, fetched_at, filed_date, form,
            accession
        FROM _companyfacts_old
    """)
    conn.execute("DROP TABLE _companyfacts_old")


def _migrate_ticker_outcomes(conn: sqlite3.Connection) -> None:
    """Add additive entry/grade/benchmark columns to ticker_outcomes.

    Idempotent ALTER TABLE ADD COLUMN guarded by PRAGMA table_info so the live
    engine.db (with existing rows) upgrades in place without data loss. SQLite
    supports only single-column ADD, so each missing column is added one at a
    time. UNIQUE(ticker, as_of_date, run_id) is preserved.
    """
    for name, decl in (
        ("entry_price", "REAL"),
        ("entry_price_source", "TEXT"),
        ("entry_date", "TEXT"),
        ("grade", "TEXT"),
        ("status", "TEXT"),
        ("benchmark_symbol", "TEXT"),
        ("benchmark_return_pct", "REAL"),
        ("excess_return_pct", "REAL"),
        # The reached-buy-target hit metric needs the
        # buy target stored at snapshot time and a resolved reached flag.
        ("buy_price_target", "REAL"),
        ("reached_buy_target", "INTEGER"),
        # Per-cap benchmark + cap-cohort reporting needs the as-of cap band stored.
        ("cap_category", "TEXT"),
        # Autonomous-sector v2 literal provenance. Nullable keeps every
        # historical and non-sector outcome backward compatible.
        ("pipeline_version", "TEXT"),
        ("candidate_disposition", "TEXT"),
        ("decision_basis", "TEXT"),
        ("selection_validation_status", "TEXT"),
        ("source_sector", "TEXT"),
        # Exact decision-source and mutable outcome-row authorization. Existing
        # history remains inspectable with NULL lineage and therefore cannot be
        # promoted into current calibration until it is rebuilt from an exact
        # authorized source.
        ("source_artifact_path", "TEXT"),
        ("source_artifact_sha256", "TEXT"),
        ("source_decision_fingerprint", "TEXT"),
        ("financial_integrity_fingerprint", "TEXT"),
    ):
        _ensure_column(conn, "ticker_outcomes", name, decl)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker_outcomes_source_run "
        "ON ticker_outcomes(run_id, ticker)"
    )


def _migrate_catalyst_events(conn: sqlite3.Connection) -> None:
    """Create the catalyst_events cache table if it is missing.

    Idempotent and guarded by PRAGMA table_info (via _has_column / CREATE TABLE
    IF NOT EXISTS), mirroring the _migrate_ticker_outcomes pattern so the live
    engine.db gains the table in place without data loss. The table memoizes the
    parsed Form-4 / buyback catalyst signal per (ticker, as_of_date,
    catalyst_type) so trigger runs do not re-fetch and re-parse form4.xml, and so
    the parsed events are auditable and can feed the outcome loop.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS catalyst_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            catalyst_type TEXT NOT NULL,
            signal_label TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 0,
            detail_json TEXT NOT NULL DEFAULT '{}',
            source_url TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date, catalyst_type)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_catalyst_events_lookup "
        "ON catalyst_events(ticker, as_of_date, catalyst_type)"
    )


def _migrate_corporate_events(conn: sqlite3.Connection) -> None:
    """Events feed: idempotent corporate_event* tables (spec 2026-06-09).

    Four tables keyed by CIK (brand-new SpinCo / post-reorg CIKs have no ticker
    yet — events must never be dropped for lack of one). UNIQUE(cik,
    event_type, anchor_accession) is the idempotency key that makes re-polls
    converge; scan rows make backfills resumable per-day.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS corporate_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cik TEXT NOT NULL,
            event_type TEXT NOT NULL,
            anchor_accession TEXT NOT NULL,
            company_name TEXT NOT NULL,
            ticker TEXT,
            ticker_state TEXT NOT NULL DEFAULT 'UNKNOWN_TICKER',
            status TEXT NOT NULL DEFAULT 'DETECTED',
            detection_date TEXT NOT NULL,
            qualification_date TEXT,
            expiry_reason TEXT,
            detail_json TEXT NOT NULL DEFAULT '{}',
            source_mode TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            qualified_at TEXT,
            surfaced_at TEXT,
            decided_at TEXT,
            expired_at TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(cik, event_type, anchor_accession)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS corporate_event_filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            cik TEXT NOT NULL,
            accession TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            role TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(event_id, accession)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS corporate_event_skips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            cik TEXT NOT NULL,
            accession TEXT NOT NULL,
            form_type TEXT NOT NULL,
            detector TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(cik, accession, detector)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS corporate_event_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            index_rows INTEGER,
            candidate_rows INTEGER,
            events_created INTEGER,
            events_updated INTEGER,
            skips_recorded INTEGER,
            detail_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(scan_date)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_corporate_events_lookup "
        "ON corporate_events(cik, event_type, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_corporate_events_ticker ON corporate_events(ticker, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_corporate_event_filings_event "
        "ON corporate_event_filings(event_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_corporate_event_skips_date "
        "ON corporate_event_skips(scan_date)"
    )


def _migrate_cheapness_reports(conn: sqlite3.Connection) -> None:
    """Cheapness-explanation pass cache (app/events/cheapness.py).

    One row per (ticker, filings_fingerprint): the deterministic
    why-is-it-cheap evidence plus the cheap-LLM summary. Cached reuse also
    requires the exact canonical financial-scope fingerprint, so a changed
    quote/cap/share lineage cannot inherit an older substantive decision.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cheapness_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            cik TEXT,
            filings_fingerprint TEXT NOT NULL,
            financial_scope_fingerprint TEXT,
            financial_scope_publication_fingerprint TEXT,
            publication_source_sha256 TEXT,
            publication_row_sha256 TEXT,
            paid_attempt_id TEXT,
            as_of TEXT NOT NULL,
            flags_json TEXT NOT NULL DEFAULT '[]',
            deterministic_json TEXT NOT NULL DEFAULT '{}',
            llm_verdict TEXT,
            llm_bullets_json TEXT NOT NULL DEFAULT '[]',
            llm_model TEXT,
            llm_usage_json TEXT NOT NULL DEFAULT '{}',
            llm_cost_usd REAL NOT NULL DEFAULT 0.0,
            llm_budget_usd REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(ticker, filings_fingerprint)
        )
        """
    )
    _ensure_column(
        conn,
        "cheapness_reports",
        "financial_scope_fingerprint",
        "TEXT",
    )
    _ensure_column(
        conn,
        "cheapness_reports",
        "financial_scope_publication_fingerprint",
        "TEXT",
    )
    _ensure_column(
        conn,
        "cheapness_reports",
        "publication_source_sha256",
        "TEXT",
    )
    _ensure_column(
        conn,
        "cheapness_reports",
        "publication_row_sha256",
        "TEXT",
    )
    _ensure_column(
        conn,
        "cheapness_reports",
        "paid_attempt_id",
        "TEXT",
    )
    _ensure_column(
        conn,
        "cheapness_reports",
        "llm_usage_json",
        "TEXT NOT NULL DEFAULT '{}'",
    )
    _ensure_column(
        conn,
        "cheapness_reports",
        "llm_cost_usd",
        "REAL NOT NULL DEFAULT 0.0",
    )
    _ensure_column(
        conn,
        "cheapness_reports",
        "llm_budget_usd",
        "REAL NOT NULL DEFAULT 0.0",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cheapness_reports_ticker "
        "ON cheapness_reports(ticker, updated_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cheapness_publication_authorizations (
            authorization_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            filings_fingerprint TEXT NOT NULL,
            publication_row_sha256 TEXT NOT NULL,
            financial_scope_fingerprint TEXT NOT NULL,
            publication_source_sha256 TEXT NOT NULL,
            publication_evidence_json TEXT NOT NULL,
            authorized_at TEXT NOT NULL,
            UNIQUE (
                ticker, filings_fingerprint, publication_row_sha256,
                financial_scope_fingerprint, publication_source_sha256
            )
        )
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS cheapness_publication_authorizations_no_update
        BEFORE UPDATE ON cheapness_publication_authorizations
        BEGIN
            SELECT RAISE(ABORT, 'cheapness publication authorizations are append-only');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS cheapness_publication_authorizations_no_delete
        BEFORE DELETE ON cheapness_publication_authorizations
        BEGIN
            SELECT RAISE(ABORT, 'cheapness publication authorizations are append-only');
        END
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cheapness_publication_authorizations_lookup "
        "ON cheapness_publication_authorizations("
        "ticker, filings_fingerprint, publication_row_sha256)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cheapness_paid_attempts (
            attempt_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            cik TEXT NOT NULL,
            as_of TEXT NOT NULL,
            filings_fingerprint TEXT NOT NULL,
            financial_scope_fingerprint TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL CHECK(
                status IN ('RESERVED', 'RETURNED', 'REJECTED', 'PUBLISHED')
            ),
            outcome TEXT CHECK(outcome IN ('SUCCESS', 'ERROR')),
            estimated_cost_usd REAL NOT NULL,
            accounted_cost_usd REAL NOT NULL DEFAULT 0.0,
            result_json TEXT,
            error TEXT,
            reserved_at TEXT NOT NULL,
            completed_at TEXT,
            published_at TEXT,
            publication_row_sha256 TEXT,
            UNIQUE (
                ticker, as_of, filings_fingerprint,
                financial_scope_fingerprint, request_sha256
            )
        )
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS cheapness_paid_attempts_no_delete
        BEFORE DELETE ON cheapness_paid_attempts
        BEGIN
            SELECT RAISE(ABORT, 'cheapness paid attempts cannot be deleted');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS cheapness_paid_attempts_published_no_update
        BEFORE UPDATE ON cheapness_paid_attempts
        WHEN OLD.status = 'PUBLISHED'
        BEGIN
            SELECT RAISE(ABORT, 'published cheapness paid attempts are immutable');
        END
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cheapness_paid_attempts_lookup "
        "ON cheapness_paid_attempts(ticker, as_of, financial_scope_fingerprint)"
    )


def _migrate_measurement_and_history_tables(conn: sqlite3.Connection) -> None:
    """Backtest/production separation + append-preserving records.

    - valuations_measurement: sibling table backtest reconstruction writes
      into (via app/valuation/measurement.py scope) so measurement can never
      overwrite the production rows live decisions were made on.
    - valuations_history / ticker_outcomes_history /
      corporate_event_scans_history: pre-overwrite copies, so the platform
      can prove what it said on any past date after any re-run.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS valuations_measurement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            method TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            valuation_writer_version TEXT,
            quality_gate_verdict TEXT,
            confidence_class TEXT,
            gate_reason_codes TEXT,
            valuation_headwinds TEXT,
            valuation_supports TEXT,
            source_run_id TEXT,
            source_artifact_path TEXT,
            source_artifact_sha256 TEXT,
            financial_integrity_fingerprint TEXT,
            UNIQUE(ticker, as_of_date, method)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_valuations_measurement_lookup "
        "ON valuations_measurement(ticker, as_of_date, method)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS valuations_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            method TEXT NOT NULL,
            inputs_json TEXT,
            outputs_json TEXT,
            warnings_json TEXT,
            created_at TEXT,
            valuation_writer_version TEXT,
            quality_gate_verdict TEXT,
            confidence_class TEXT,
            gate_reason_codes TEXT,
            valuation_headwinds TEXT,
            valuation_supports TEXT,
            source_run_id TEXT,
            source_artifact_path TEXT,
            source_artifact_sha256 TEXT,
            financial_integrity_fingerprint TEXT,
            archived_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_valuations_history_lookup "
        "ON valuations_history(ticker, as_of_date, method, archived_at)"
    )
    lineage_columns = (
        ("source_run_id", "TEXT"),
        ("source_artifact_path", "TEXT"),
        ("source_artifact_sha256", "TEXT"),
        ("financial_integrity_fingerprint", "TEXT"),
    )
    for table in ("valuations", "valuations_measurement", "valuations_history"):
        for name, declaration in lineage_columns:
            _ensure_column(conn, table, name, declaration)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_valuations_source_run ON valuations(source_run_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_valuations_history_source_run "
        "ON valuations_history(source_run_id)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_outcomes_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            ticker TEXT,
            as_of_date TEXT,
            run_id TEXT,
            row_json TEXT NOT NULL,
            archived_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker_outcomes_history_lookup "
        "ON ticker_outcomes_history(ticker, as_of_date, run_id, archived_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS corporate_event_scans_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            scan_date TEXT,
            row_json TEXT NOT NULL,
            archived_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_corporate_event_scans_history_lookup "
        "ON corporate_event_scans_history(scan_date, archived_at)"
    )


def _migrate_dispositions(conn: sqlite3.Connection) -> None:
    """The disposition ledger — mandatory terminus of at-target surfacing.

    Every DEPLOY_READY presentation opens a row; the operator closes it as
    ACTED / PASSED / DEFERRED with operator, typed reason, rationale, and
    intended size. Closed rows write ticker_outcomes (run_id='journal_live')
    so the behavioral kill-criterion tally counts automatically. Rows are
    never deleted; a decided row is never re-decided (append-only ledger
    discipline, enforced in app/watchlist/dispositions.py).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dispositions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            watchlist_id INTEGER,
            kind TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            opened_at TEXT NOT NULL,
            opened_by TEXT NOT NULL,
            trigger_snapshot_json TEXT NOT NULL DEFAULT '{}',
            decided_at TEXT,
            operator TEXT,
            reason_code TEXT,
            rationale TEXT,
            intended_size TEXT,
            sizing_rationale TEXT,
            pre_mortem TEXT,
            outcome_run_id TEXT,
            event_id INTEGER
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dispositions_open ON dispositions(status, ticker, kind)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dispositions_ticker ON dispositions(ticker, opened_at)"
    )


def _migrate_heartbeat_runs(conn: sqlite3.Connection) -> None:
    """Fail-loud ops: per-step heartbeat ledger.

    One row per (heartbeat, run_date, step) attempt with the step's exit code,
    replacing done-marker log greps. The catch-up checker asks this table
    "did heartbeat H complete for date D?" and the dead-man check asserts
    step-level exit codes. Writes are best-effort from the shell (a broken
    engine.db must not also break the heartbeat that would report it).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS heartbeat_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            heartbeat TEXT NOT NULL,
            run_date TEXT NOT NULL,
            step TEXT NOT NULL,
            status TEXT NOT NULL,
            exit_code INTEGER,
            detail TEXT,
            recorded_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_heartbeat_runs_lookup "
        "ON heartbeat_runs(heartbeat, run_date, step, recorded_at)"
    )


def _migrate_sector_run_loaded_sets(conn: sqlite3.Connection) -> None:
    """Delta sweeps: per-run ticker states (app/autonomous/sweep_delta.py).

    One row per (run_id, ticker). Current coverage is granted only by an
    explicit terminal disposition reconstructed from a persisted artifact or
    written by the run itself. Legacy loader-only rows remain as audit history
    but do not suppress future review.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sector_run_loaded_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            sector TEXT NOT NULL,
            market_cap_focus TEXT NOT NULL,
            source TEXT NOT NULL,
            ticker TEXT NOT NULL,
            loaded_at TEXT NOT NULL,
            pipeline_version TEXT,
            candidate_disposition TEXT,
            coverage_campaign_id TEXT,
            coverage_evidence_json TEXT,
            coverage_evidence_sha256 TEXT,
            coverage_authority_kind TEXT,
            coverage_authority_path TEXT,
            coverage_authority_sha256 TEXT,
            coverage_source_run_id TEXT,
            coverage_complete INTEGER NOT NULL DEFAULT 1,
            UNIQUE(run_id, ticker)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sector_run_loaded_sets_band "
        "ON sector_run_loaded_sets(market_cap_focus, ticker)"
    )
    _ensure_column(conn, "sector_run_loaded_sets", "pipeline_version", "TEXT")
    _ensure_column(conn, "sector_run_loaded_sets", "candidate_disposition", "TEXT")
    _ensure_column(conn, "sector_run_loaded_sets", "coverage_campaign_id", "TEXT")
    _ensure_column(conn, "sector_run_loaded_sets", "coverage_evidence_json", "TEXT")
    _ensure_column(conn, "sector_run_loaded_sets", "coverage_evidence_sha256", "TEXT")
    _ensure_column(conn, "sector_run_loaded_sets", "coverage_authority_kind", "TEXT")
    _ensure_column(conn, "sector_run_loaded_sets", "coverage_authority_path", "TEXT")
    _ensure_column(conn, "sector_run_loaded_sets", "coverage_authority_sha256", "TEXT")
    _ensure_column(conn, "sector_run_loaded_sets", "coverage_source_run_id", "TEXT")
    _ensure_column(
        conn,
        "sector_run_loaded_sets",
        "coverage_complete",
        "INTEGER NOT NULL DEFAULT 1",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sector_run_loaded_sets_sector_coverage "
        "ON sector_run_loaded_sets(market_cap_focus, sector, coverage_complete, ticker)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sector_run_loaded_sets_campaign_coverage "
        "ON sector_run_loaded_sets(coverage_campaign_id, market_cap_focus, "
        "sector, coverage_complete, ticker)"
    )


# Every run that collected an evidence item. evidence_items holds one row per item
# and names only the latest run that collected it; this table keeps the earlier
# runs' links so re-collecting an item never empties a past run. Created on
# demand by its writer and reader, so a database made before it existed works.
EVIDENCE_ITEM_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS evidence_item_runs (
    evidence_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (evidence_id, run_id)
)
"""


def ensure_evidence_item_runs(conn: sqlite3.Connection) -> None:
    conn.execute(EVIDENCE_ITEM_RUNS_SQL)


def _evidence_duplicate_group_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS groups_count
        FROM (
            SELECT 1
            FROM evidence_items
            WHERE ticker IS NOT NULL AND dedupe_key IS NOT NULL
            GROUP BY ticker, dedupe_key
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    return int(row["groups_count"] or 0)


def dedupe_evidence_items(conn: sqlite3.Connection) -> dict[str, int]:
    groups_before = _evidence_duplicate_group_count(conn)
    rows_deleted = 0
    if groups_before > 0:
        fallback_id_expr = "id" if _has_column(conn, "evidence_items", "id") else "rowid"
        conn.execute("SAVEPOINT evidence_items_dedupe")
        try:
            before_changes = conn.total_changes
            conn.execute(
                f"""
                WITH ranked AS (
                    SELECT
                        {fallback_id_expr} AS row_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY ticker, dedupe_key
                            ORDER BY
                                CASE WHEN created_at IS NULL OR created_at = '' THEN 1 ELSE 0 END ASC,
                                created_at DESC,
                                {fallback_id_expr} DESC
                        ) AS rank_in_group
                    FROM evidence_items
                    WHERE ticker IS NOT NULL AND dedupe_key IS NOT NULL
                )
                DELETE FROM evidence_items
                WHERE rowid IN (
                    SELECT row_id
                    FROM ranked
                    WHERE rank_in_group > 1
                )
                """
            )
            rows_deleted = int(conn.total_changes - before_changes)
            conn.execute("RELEASE SAVEPOINT evidence_items_dedupe")
        except Exception:
            conn.execute("ROLLBACK TO SAVEPOINT evidence_items_dedupe")
            conn.execute("RELEASE SAVEPOINT evidence_items_dedupe")
            raise

    groups_after = _evidence_duplicate_group_count(conn)
    logger.info(
        "evidence_items_dedupe_completed",
        extra={
            "stage_name": "db_migration",
            "stage_count": rows_deleted,
            "stage_groups_before": groups_before,
            "stage_groups_after": groups_after,
        },
    )
    return {
        "duplicate_groups_before": groups_before,
        "rows_deleted": rows_deleted,
        "duplicate_groups_after": groups_after,
    }


def _ensure_schema_evolution(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "companies", "universe_id", "TEXT")
    _ensure_column(conn, "companies", "ir_rss_url", "TEXT")
    _ensure_column(conn, "companies", "homepage_url", "TEXT")
    _ensure_column(conn, "companies", "allowlist_domains", "TEXT")
    _ensure_column(conn, "companies", "notes", "TEXT")
    _ensure_column(conn, "filings", "ingested_as_of", "TEXT")
    _ensure_column(conn, "jobs", "max_attempts", "INTEGER NOT NULL DEFAULT 5")
    _ensure_column(conn, "jobs", "started_at", "TEXT")
    _ensure_column(conn, "jobs", "error_type", "TEXT")
    _ensure_column(conn, "jobs", "cancelled_at", "TEXT")
    _ensure_column(conn, "jobs", "cancel_reason", "TEXT")
    _ensure_column(conn, "memos", "manifest_path", "TEXT")
    _ensure_column(conn, "memos", "run_id", "TEXT")
    _ensure_column(conn, "scores", "run_id", "TEXT")
    _ensure_column(conn, "scores", "is_candidate", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "scores", "is_publishable", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "scores", "candidate_run_id", "TEXT")
    conn.execute(
        """
        UPDATE scores
        SET candidate_run_id = run_id
        WHERE run_id IS NOT NULL
          AND is_candidate = 1
          AND (candidate_run_id IS NULL OR candidate_run_id != run_id)
        """
    )
    _ensure_column(conn, "universe_members", "ir_rss_url", "TEXT")
    _ensure_column(conn, "universe_members", "homepage_url", "TEXT")
    _ensure_column(conn, "universe_members", "allowlist_domains", "TEXT")
    _ensure_column(conn, "universe_members", "notes", "TEXT")
    _ensure_column(conn, "evidence_items", "source_published_at", "TEXT")
    _ensure_column(conn, "evidence_items", "adapter_run_id", "TEXT")
    _ensure_column(conn, "evidence_items", "source_title", "TEXT")
    _ensure_column(conn, "evidence_items", "source_url", "TEXT")
    _ensure_column(conn, "evidence_items", "excerpt_hash", "TEXT")
    _ensure_column(conn, "evidence_items", "content_hash", "TEXT")
    _ensure_column(conn, "evidence_items", "dedupe_key", "TEXT")
    conn.execute("UPDATE evidence_items SET excerpt_hash = COALESCE(excerpt_hash, item_hash)")
    conn.execute("UPDATE evidence_items SET content_hash = COALESCE(content_hash, excerpt_hash)")
    conn.execute("UPDATE evidence_items SET dedupe_key = COALESCE(dedupe_key, excerpt_hash)")
    dedupe_evidence_items(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_companies_universe ON companies(universe_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id, ticker, as_of_date, created_at)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memos_run ON memos(run_id, ticker, as_of_date)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scores_candidate_flags ON scores(is_candidate, is_publishable, total_score)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_evidence_items_dedupe ON evidence_items(ticker, source_url, excerpt_hash)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_items_dedupe_key ON evidence_items(ticker, dedupe_key)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            recency_days_min INTEGER,
            item_count_30d INTEGER NOT NULL DEFAULT 0,
            has_earnings_release INTEGER NOT NULL DEFAULT 0,
            has_investor_presentation INTEGER NOT NULL DEFAULT 0,
            sentiment_flags_json TEXT NOT NULL DEFAULT '[]',
            key_topics_json TEXT NOT NULL DEFAULT '[]',
            evidence_item_ids_json TEXT NOT NULL DEFAULT '[]',
            summary_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date, run_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_signals_lookup ON research_signals(ticker, as_of_date, run_id)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS filing_coverage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            run_id TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            forms_included_json TEXT NOT NULL,
            accession_numbers_json TEXT NOT NULL,
            coverage_score REAL NOT NULL DEFAULT 0,
            missing_required_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            UNIQUE(ticker, run_id, as_of_date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parsed_filings (
            accession TEXT PRIMARY KEY,
            parsed_at TEXT NOT NULL,
            content_hash TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_deltas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            run_id TEXT NOT NULL,
            prev_run_id TEXT,
            as_of_date TEXT NOT NULL,
            changed INTEGER NOT NULL DEFAULT 0,
            delta_path TEXT NOT NULL,
            delta_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, run_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS delta_memos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            run_id TEXT NOT NULL,
            changed INTEGER NOT NULL DEFAULT 0,
            memo_path TEXT NOT NULL,
            memo_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, run_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shortlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            shortlist_path TEXT NOT NULL,
            shortlist_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS synthesis_packets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            packet_path TEXT NOT NULL,
            packet_hash TEXT NOT NULL,
            packet_json TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            usage_json TEXT NOT NULL DEFAULT '{}',
            cost_estimate_usd REAL NOT NULL DEFAULT 0,
            paid_invocation_id TEXT,
            from_cache INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date, run_id)
        )
        """
    )
    _ensure_column(conn, "synthesis_packets", "paid_invocation_id", "TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS synthesis_paid_attempts (
            attempt_id TEXT PRIMARY KEY,
            invocation_id TEXT NOT NULL,
            physical_sequence INTEGER NOT NULL CHECK(physical_sequence > 0),
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            schema_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('OK', 'INCOMPLETE', 'ERROR')),
            usage_json TEXT NOT NULL,
            cost_estimate_usd REAL NOT NULL CHECK(cost_estimate_usd >= 0),
            created_at TEXT NOT NULL,
            UNIQUE(invocation_id, physical_sequence)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_filing_coverage_lookup ON filing_coverage(ticker, run_id, as_of_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker_deltas_lookup ON ticker_deltas(ticker, run_id, changed)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_synthesis_packets_run ON synthesis_packets(run_id, ticker, as_of_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_synthesis_packets_hash ON synthesis_packets(input_hash, prompt_hash, model, provider)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_synthesis_packets_paid_invocation "
        "ON synthesis_packets(paid_invocation_id) "
        "WHERE paid_invocation_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_synthesis_paid_attempts_run "
        "ON synthesis_paid_attempts(run_id, ticker, as_of_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_synthesis_paid_attempts_invocation "
        "ON synthesis_paid_attempts(invocation_id, physical_sequence)"
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS synthesis_paid_attempts_no_update
        BEFORE UPDATE ON synthesis_paid_attempts
        BEGIN
            SELECT RAISE(ABORT, 'synthesis paid attempts are append-only');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS synthesis_paid_attempts_no_delete
        BEFORE DELETE ON synthesis_paid_attempts
        BEGIN
            SELECT RAISE(ABORT, 'synthesis paid attempts are append-only');
        END
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_caps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            effective_as_of_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            run_as_of_date TEXT NOT NULL,
            market_cap REAL,
            market_cap_unit TEXT,
            market_cap_status TEXT NOT NULL,
            price REAL,
            price_currency TEXT,
            price_basis TEXT,
            quote_snapshot_id TEXT,
            shares_outstanding REAL,
            shares_unit TEXT,
            shares_basis TEXT,
            split_adjustment_factor REAL,
            split_effective_date TEXT,
            provider TEXT,
            source_url TEXT,
            fetched_at TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, effective_as_of_date, run_id)
        )
        """
    )
    _ensure_column(conn, "market_caps", "market_cap_unit", "TEXT")
    _ensure_column(conn, "market_caps", "price_currency", "TEXT")
    _ensure_column(conn, "market_caps", "price_basis", "TEXT")
    _ensure_column(conn, "market_caps", "quote_snapshot_id", "TEXT")
    _ensure_column(conn, "market_caps", "shares_unit", "TEXT")
    _ensure_column(conn, "market_caps", "shares_basis", "TEXT")
    _ensure_column(conn, "market_caps", "split_adjustment_factor", "REAL")
    _ensure_column(conn, "market_caps", "split_effective_date", "TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS discovery_runs (
            run_id TEXT PRIMARY KEY,
            run_as_of_date TEXT NOT NULL,
            seed_hash TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            seed_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'RUNNING',
            processed_count INTEGER NOT NULL DEFAULT 0,
            phase TEXT NOT NULL DEFAULT 'prefilter',
            tickers_targeted_json TEXT NOT NULL,
            processed_effective_dates_json TEXT NOT NULL DEFAULT '{}',
            stats_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    _ensure_column(conn, "discovery_runs", "status", "TEXT NOT NULL DEFAULT 'RUNNING'")
    _ensure_column(conn, "discovery_runs", "processed_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "discovery_runs", "phase", "TEXT NOT NULL DEFAULT 'prefilter'")
    _ensure_column(conn, "discovery_runs", "updated_at", "TEXT")
    conn.execute("UPDATE discovery_runs SET updated_at = COALESCE(updated_at, created_at)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS discovery_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            run_id TEXT NOT NULL,
            discovery_score REAL NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            publication_receipt_path TEXT,
            publication_receipt_sha256 TEXT,
            UNIQUE(ticker, run_id)
        )
        """
    )
    _ensure_column(conn, "discovery_candidates", "publication_receipt_path", "TEXT")
    _ensure_column(conn, "discovery_candidates", "publication_receipt_sha256", "TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS discovery_phase1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            cik TEXT,
            company_name TEXT,
            eligible INTEGER NOT NULL DEFAULT 0,
            prefilter_score REAL NOT NULL DEFAULT 0,
            latest_filing_date TEXT,
            selected_accessions_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, ticker)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_market_caps_lookup ON market_caps(ticker, effective_as_of_date, run_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_discovery_runs_date ON discovery_runs(run_as_of_date, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_discovery_candidates_run_score ON discovery_candidates(run_id, discovery_score DESC, ticker ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_discovery_phase1_run_score ON discovery_phase1(run_id, prefilter_score DESC, ticker ASC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS discovery_lifecycle (
            ticker TEXT PRIMARY KEY,
            first_seen_run_id TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_run_id TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            times_shortlisted INTEGER NOT NULL DEFAULT 0,
            last_discovery_score REAL,
            last_action TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS discovery_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            discovery_run_id TEXT NOT NULL,
            deep_run_id TEXT,
            run_as_of_date TEXT NOT NULL,
            entry_price REAL,
            price_source TEXT,
            forward_return_30d REAL,
            forward_return_90d REAL,
            forward_return_180d REAL,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, discovery_run_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_discovery_outcomes_run ON discovery_outcomes(discovery_run_id, ticker)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            run_id TEXT NOT NULL,
            discovery_run_id TEXT,
            deep_run_id TEXT,
            decision TEXT NOT NULL,
            conviction INTEGER NOT NULL,
            horizon_days INTEGER NOT NULL,
            thesis_tags_json TEXT NOT NULL DEFAULT '[]',
            notes TEXT,
            outcome_status TEXT NOT NULL DEFAULT 'OPEN',
            close_date TEXT,
            realized_return_pct REAL,
            max_drawdown_pct REAL,
            entry_price REAL,
            entry_price_source TEXT,
            entry_date TEXT,
            grade TEXT,
            status TEXT,
            benchmark_symbol TEXT,
            pipeline_version TEXT,
            candidate_disposition TEXT,
            decision_basis TEXT,
            selection_validation_status TEXT,
            source_sector TEXT,
            benchmark_return_pct REAL,
            excess_return_pct REAL,
            source_artifact_path TEXT,
            source_artifact_sha256 TEXT,
            source_decision_fingerprint TEXT,
            financial_integrity_fingerprint TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date, run_id)
        )
        """
    )
    _migrate_ticker_outcomes(conn)
    _migrate_catalyst_events(conn)
    _migrate_corporate_events(conn)
    _migrate_cheapness_reports(conn)
    _migrate_sector_run_loaded_sets(conn)
    _migrate_heartbeat_runs(conn)
    _migrate_measurement_and_history_tables(conn)
    _migrate_dispositions(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            run_id TEXT NOT NULL,
            discovery_run_id TEXT,
            as_of_date TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(ticker, run_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            as_of_date TEXT,
            report_json TEXT NOT NULL,
            report_path TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sector_inference (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            inferred_sector TEXT,
            score REAL NOT NULL DEFAULT 0,
            derived_from TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rlm_runs (
            run_id TEXT PRIMARY KEY,
            sector TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            iterations INTEGER NOT NULL DEFAULT 0,
            stop_reason TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rlm_iterations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            planner_json TEXT NOT NULL,
            critic_json TEXT NOT NULL,
            progress_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, iteration)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker_outcomes_lookup ON ticker_outcomes(run_id, ticker, outcome_status, updated_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_cycles_lookup ON research_cycles(run_id, ticker, updated_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_calibration_reports_lookup ON calibration_reports(created_at, run_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sector_inference_lookup ON sector_inference(as_of_date, inferred_sector, ticker)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rlm_runs_status ON rlm_runs(status, updated_at)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rlm_iterations_lookup ON rlm_iterations(run_id, iteration)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS companyfacts_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL DEFAULT 'FY',
            period_end TEXT NOT NULL,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            source_url TEXT,
            fetched_at TEXT NOT NULL,
            UNIQUE(ticker, fiscal_year, period_type, line_item)
        )
        """
    )
    _ensure_column(conn, "companyfacts_facts", "period_type", "TEXT DEFAULT 'FY'")
    # PIT foundation: filing provenance on every fact (the normalizer has
    # these in hand; the quarterly fact chain already stores them), plus
    # the append-only vintage table so as-first-reported values survive
    # restatements. All additive.
    _ensure_column(conn, "companyfacts_facts", "filed_date", "TEXT")
    _ensure_column(conn, "companyfacts_facts", "form", "TEXT")
    _ensure_column(conn, "companyfacts_facts", "accession", "TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS companyfacts_vintages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL,
            period_end TEXT,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            filed_date TEXT NOT NULL DEFAULT '',
            form TEXT,
            accession TEXT,
            recorded_at TEXT NOT NULL,
            UNIQUE(ticker, fiscal_year, period_type, line_item, filed_date, value)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_companyfacts_vintages_lookup "
        "ON companyfacts_vintages(ticker, line_item, fiscal_year, filed_date)"
    )
    _ensure_column(conn, "companyfacts_vintages", "issuer_cik", "TEXT")
    _ensure_column(conn, "companyfacts_vintages", "source_url", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_companyfacts_vintages_issuer_lookup "
        "ON companyfacts_vintages(issuer_cik, ticker, line_item, fiscal_year, filed_date)"
    )
    _migrate_companyfacts_constraint(conn)
    # After the constraint migration, which rebuilds the table without it. The
    # concepts a DERIVED row was formed from, where one row is not one tag
    # (short_term_investments: "ShortTermInvestments", or "A + B", or
    # "CashCashEquivalentsAndShortTermInvestments - <cash tag>").
    _ensure_column(conn, "companyfacts_facts", "source_tags", "TEXT")
    # Which writer last wrote the row: "facts_writer" (the refreshing normalizer, which
    # purges the rows it no longer emits) or "companyfacts_repair" (point-in-time repairs,
    # which that purge must leave alone). NULL is a row from before the marker existed.
    _ensure_column(conn, "companyfacts_facts", "written_by", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_companyfacts_lookup ON companyfacts_facts(ticker, fiscal_year)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_companyfacts_period ON companyfacts_facts(ticker, fiscal_year, period_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_companyfacts_source_url "
        "ON companyfacts_facts(source_url, ticker)"
    )
    # Liquidity layer: share volume on daily quote rows (the input to
    # dollar-ADV). NULL for sources that carry no volume.
    _ensure_column(conn, "price_quotes", "volume", "REAL")
    _ensure_column(conn, "price_quotes", "price_basis", "TEXT")
    _ensure_column(conn, "price_quotes", "split_adjustment_factor", "REAL")
    _ensure_column(conn, "price_quotes", "split_effective_date", "TEXT")
    _ensure_column(conn, "valuations", "valuation_writer_version", "TEXT")
    _ensure_column(conn, "valuations", "quality_gate_verdict", "TEXT")
    _ensure_column(conn, "valuations", "confidence_class", "TEXT")
    _ensure_column(conn, "valuations", "gate_reason_codes", "TEXT")
    _ensure_column(conn, "valuations", "valuation_headwinds", "TEXT")
    _ensure_column(conn, "valuations", "valuation_supports", "TEXT")

    # Task 9: outcome tracking tables
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS deep_research_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source_as_of_date TEXT NOT NULL,
            source_valuation_id INTEGER,
            source_run_id TEXT,
            source_artifact_path TEXT,
            source_artifact_sha256 TEXT,
            source_valuation_fingerprint TEXT,
            horizon_days INTEGER NOT NULL CHECK (horizon_days > 0),
            target_exit_date TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            entry_price REAL NOT NULL CHECK (entry_price > 0),
            exit_price REAL NOT NULL CHECK (exit_price > 0),
            exit_as_of_date TEXT NOT NULL,
            exit_source TEXT NOT NULL,
            price_change_pct REAL NOT NULL,
            thesis_verdict TEXT NOT NULL
                CHECK (thesis_verdict IN ('UNDERVALUED', 'OVERVALUED', 'FAIRLY_VALUED')),
            verdict_outcome TEXT NOT NULL
                CHECK (verdict_outcome IN ('CORRECT', 'INCORRECT', 'INCONCLUSIVE')),
            conviction_score INTEGER,
            conviction_class TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, source_as_of_date, horizon_days)
        );

        CREATE INDEX IF NOT EXISTS idx_dro_verdict
            ON deep_research_outcomes(horizon_days, verdict_outcome);
        CREATE INDEX IF NOT EXISTS idx_dro_ticker_date
            ON deep_research_outcomes(ticker, source_as_of_date);

        CREATE TABLE IF NOT EXISTS deep_research_method_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            outcome_id INTEGER NOT NULL
                REFERENCES deep_research_outcomes(id) ON DELETE CASCADE,
            method TEXT NOT NULL CHECK (method IN ('dcf', 'epv', 'graham')),
            predicted_value REAL NOT NULL,
            entry_price REAL NOT NULL CHECK (entry_price > 0),
            direction TEXT NOT NULL
                CHECK (direction IN ('UNDERVALUED', 'OVERVALUED', 'FAIRLY_VALUED')),
            outcome TEXT NOT NULL
                CHECK (outcome IN ('CORRECT', 'INCORRECT', 'INCONCLUSIVE')),
            unadjusted_source INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(outcome_id, method)
        );

        CREATE INDEX IF NOT EXISTS idx_drmo_method_outcome
            ON deep_research_method_outcomes(method, outcome);
        """
    )
    for name, declaration in (
        ("source_run_id", "TEXT"),
        ("source_artifact_path", "TEXT"),
        ("source_artifact_sha256", "TEXT"),
        ("source_valuation_fingerprint", "TEXT"),
    ):
        _ensure_column(conn, "deep_research_outcomes", name, declaration)


def upsert_catalyst_event(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    as_of_date: str,
    catalyst_type: str,
    signal_label: str,
    score: float,
    detail: dict[str, Any] | None = None,
    source_url: str | None = None,
) -> None:
    """Persist a parsed catalyst signal to the catalyst_events cache.

    Idempotent on UNIQUE(ticker, as_of_date, catalyst_type) so a re-run for the
    same as-of date overwrites the row rather than re-fetching/re-parsing
    form4.xml each trigger run. ``detail`` is JSON-serialized into detail_json.
    """
    conn.execute(
        """
        INSERT INTO catalyst_events(
            ticker, as_of_date, catalyst_type, signal_label, score,
            detail_json, source_url, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, as_of_date, catalyst_type) DO UPDATE SET
            signal_label = excluded.signal_label,
            score = excluded.score,
            detail_json = excluded.detail_json,
            source_url = excluded.source_url,
            created_at = excluded.created_at
        """,
        (
            ticker,
            as_of_date,
            catalyst_type,
            signal_label,
            float(score),
            json.dumps(detail or {}),
            source_url,
            utc_now_iso(),
        ),
    )


def get_catalyst_event(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    as_of_date: str,
    catalyst_type: str,
) -> dict[str, Any] | None:
    """Return a cached catalyst_events row (with detail parsed) or None."""
    row = conn.execute(
        """
        SELECT ticker, as_of_date, catalyst_type, signal_label, score,
               detail_json, source_url, created_at
        FROM catalyst_events
        WHERE ticker = ? AND as_of_date = ? AND catalyst_type = ?
        """,
        (ticker, as_of_date, catalyst_type),
    ).fetchone()
    if not row:
        return None
    return {
        "ticker": row["ticker"],
        "as_of_date": row["as_of_date"],
        "catalyst_type": row["catalyst_type"],
        "signal_label": row["signal_label"],
        "score": row["score"],
        "detail": json.loads(row["detail_json"] or "{}"),
        "source_url": row["source_url"],
        "created_at": row["created_at"],
    }


def upsert_state(conn: sqlite3.Connection, key: str, value: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO state(key, value_json, updated_at)
        VALUES(?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value_json = excluded.value_json,
            updated_at = excluded.updated_at
        """,
        (key, json.dumps(value), utc_now_iso()),
    )


def get_state(conn: sqlite3.Connection, key: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT value_json FROM state WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    return json.loads(row["value_json"])
