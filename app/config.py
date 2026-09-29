from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

from app.util.credential_hygiene import InsecureEnvPermissionsError, require_private_env_file
from pydantic import BaseModel, Field

import os as _os
import sys as _sys


def _dotenv_candidates() -> list[Path]:
    """The .env files to read, first one wins.

    Always the source checkout's own ``.env``. The working directory's ``.env``
    only when ``VOE_DOTENV_CWD=1``: by default a command run inside some other
    project must not pick up that project's file (it may be world-readable, or
    crafted to switch on paid AI or redirect API traffic).
    """

    candidates = [Path(__file__).resolve().parents[1] / ".env"]
    if _os.environ.get("VOE_DOTENV_CWD", "0").strip() == "1":
        try:
            candidates.insert(0, Path.cwd() / ".env")
        except OSError:
            # The working directory was deleted under us; nothing to read there.
            print(
                "warning: VOE_DOTENV_CWD=1 but the working directory no longer exists; "
                "skipping its .env",
                file=_sys.stderr,
            )
    return list(dict.fromkeys(candidates))


def load_env_files(candidates: list[Path] | None = None) -> list[Path]:
    """Import ``VOE_*`` settings from ``.env`` files into the environment.

    Only ``VOE_*`` names are imported, so a ``.env`` cannot set ambient SDK
    variables such as ``ANTHROPIC_BASE_URL``. A file readable or writable by
    group/other is skipped with a warning on stderr, never loaded. Variables
    already in the environment are never overwritten. Returns the files read.
    """

    loaded: list[Path] = []
    for env_path in candidates if candidates is not None else _dotenv_candidates():
        try:
            if not env_path.is_file():
                continue
            require_private_env_file(env_path)
        except InsecureEnvPermissionsError as exc:
            print(f"warning: skipping {env_path}: {exc}", file=_sys.stderr)
            continue
        except OSError:
            continue
        for name, value in dotenv_values(env_path).items():
            if name.startswith("VOE_") and value is not None and name not in _os.environ:
                _os.environ[name] = value
        loaded.append(env_path)
    return loaded


# Skipped under pytest so tests control their own environment.
if "pytest" not in _sys.modules and "_pytest" not in _sys.modules:
    load_env_files()


class AppConfig(BaseModel):
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    db_path: Path = Field(default_factory=lambda: Path("data/engine.db"))
    universe_path: Path = Field(default_factory=lambda: Path("data/universe/universe.csv"))
    raw_filings_dir: Path = Field(default_factory=lambda: Path("data/raw_filings"))
    cache_dir: Path = Field(default_factory=lambda: Path("data/cache"))
    outputs_dir: Path = Field(default_factory=lambda: Path("data/outputs"))
    # Mount guard: the expected external-volume UUID and the sentinel file
    # that carries it. Unset UUID disables the check (developer checkouts).
    data_volume_uuid: str | None = Field(default=None)
    data_volume_sentinel: Path = Field(default_factory=lambda: Path("data/.ivi-volume"))
    evidence_dir: Path = Field(default_factory=lambda: Path("data/outputs/evidence_packets"))
    analyst_outputs_dir: Path = Field(default_factory=lambda: Path("data/outputs/analyst_outputs"))
    memos_dir: Path = Field(default_factory=lambda: Path("data/outputs/memos"))
    gaps_dir: Path = Field(default_factory=lambda: Path("data/outputs/gaps"))
    deltas_dir: Path = Field(default_factory=lambda: Path("data/outputs/deltas"))
    delta_memos_dir: Path = Field(default_factory=lambda: Path("data/outputs/delta_memos"))
    shortlists_dir: Path = Field(default_factory=lambda: Path("data/outputs/shortlists"))
    synthesis_dir: Path = Field(default_factory=lambda: Path("data/outputs/synthesis"))
    discovery_dir: Path = Field(default_factory=lambda: Path("data/outputs/discovery"))
    dossiers_dir: Path = Field(default_factory=lambda: Path("data/outputs/dossiers"))
    sectors_dir: Path = Field(default_factory=lambda: Path("data/outputs/sectors"))
    calibration_dir: Path = Field(default_factory=lambda: Path("data/outputs/calibration"))
    rankings_dir: Path = Field(default_factory=lambda: Path("data/outputs/rankings"))
    campaigns_dir: Path = Field(default_factory=lambda: Path("data/outputs/campaigns"))
    status_path: Path = Field(default_factory=lambda: Path("data/outputs/agent_status.json"))
    manifests_dir: Path = Field(default_factory=lambda: Path("data/outputs/manifests"))
    run_manifest_path: Path = Field(default_factory=lambda: Path("data/outputs/manifests/run_manifest.json"))
    runs_dir: Path = Field(default_factory=lambda: Path("data/outputs/runs"))
    runs_index_path: Path = Field(default_factory=lambda: Path("data/outputs/runs/index.json"))
    universe_dir: Path = Field(default_factory=lambda: Path("data/universe"))
    discovery_seed_path: Path = Field(default_factory=lambda: Path("data/universe/discovery_seed.csv"))
    sector_taxonomy_path: Path = Field(default_factory=lambda: Path("data/universe/sector_taxonomy.csv"))
    sector_overrides_path: Path = Field(default_factory=lambda: Path("data/universe/sector_overrides.csv"))
    sector_sic_config_path: Path = Field(default_factory=lambda: Path("data/universe/sector_sic_ranges.json"))

    # Required by SEC fair-access policy: identify yourself with a contact email,
    # e.g. VOE_SEC_USER_AGENT="Jane Doe jane.doe@yourdomain.com". No default ships.
    sec_user_agent: str = ""
    sec_rate_limit_per_sec: float = 10.0
    sec_max_retries: int = 4
    sec_backoff_seconds: float = 1.0
    http_timeout_seconds: float = 30.0
    # 300s: heartbeats/backfills are separate cron processes writing one
    # engine.db — a 5s timeout turned write overlaps into failed steps
    # (seen on the first live run), and a single event-detector write window
    # can span minutes. Batch jobs should WAIT for the lock, not die.
    sqlite_busy_timeout_ms: int = 300000

    allow_fred: bool = False
    fred_api_key: str | None = None

    # "auto": EODHD first when VOE_EODHD_APIKEY is set, then Yahoo Finance and Stooq
    # (both free). "disabled" turns live prices off; "yahoo", "stooq" pin one source.
    price_provider: str = "auto"
    price_fallback_days: int = 7
    price_symbol_overrides_path: Path = Field(default_factory=lambda: Path("config/price_symbol_overrides.csv"))
    shares_allow_market_cap_price_derive: bool = False
    issuer_classification_by_sic: bool = True
    quote_ttl_seconds: int = 86400
    stooq_base_url: str = "https://stooq.com/q/l/"
    stooq_history_url: str = "https://stooq.com/q/d/l/"
    stooq_apikey: str | None = None
    stooq_apikey_param: str = "apikey"
    eodhd_base_url: str = "https://eodhd.com"
    eodhd_apikey: str | None = None
    eodhd_exchange: str = "US"
    # INERT tier-2 slot in the band-filter cap chain (app/autonomous/cap_resolver.py).
    # The EODHD fundamentals endpoint 403s on the current subscription; flip this
    # only after a plan upgrade.
    cap_eodhd_fundamentals_enabled: bool = False
    # Optional dated direct-cap evidence ledger produced by provider/search
    # repair. V2 sector scans validate every row's URL/as-of/confidence and
    # issuer binding before it can resolve a cap; v1 ignores this path.
    cap_terminal_evidence_path: Path | None = None
    safe_mode: bool = True

    scheduler_poll_seconds: int = 10
    scheduler_worker_count: int = 1
    max_workers: int = 8
    max_db_write_concurrency: int = 1
    research_max_http_concurrency: int = 2
    running_job_timeout_seconds: int = 1800
    max_requests_sec_domain: int = 1200
    max_requests_data_sec_domain: int = 1200
    max_requests_www_sec_domain: int = 1200
    max_requests_stooq_domain: int = 600
    max_requests_eodhd_domain: int = 600
    max_requests_alpha_vantage_domain: int = 300
    max_requests_wikipedia_domain: int = 300
    max_requests_press_domain: int = 300

    research_dir: Path = Field(default_factory=lambda: Path("data/outputs/research"))
    research_enable_in_run_all: bool = False
    research_top_n: int = 10
    research_quality_threshold: float = 50.0
    research_enable_transcripts: bool = False
    research_transcript_provider: str = "disabled"
    research_alpha_vantage_api_key: str | None = None
    research_transcript_max_quarters: int = 4
    research_ir_press_enabled: bool = False
    research_allowlist_domains: list[str] = Field(default_factory=list)
    research_ir_max_items_per_ticker: int = 10
    research_ir_max_days_back: int = 365
    research_company_news_enabled: bool = True
    research_company_news_max_pages: int = 8
    research_external_news_enabled: bool = False
    research_external_news_provider: str = "disabled"
    research_external_news_max_items: int = 10
    research_source_reputation_path: Path = Field(default_factory=lambda: Path("config/source_reputation_history.csv"))
    research_exhibits_enabled: bool = True
    research_enable_wikipedia: bool = False
    analyst_notes_enabled: bool = False
    anthropic_api_key: str | None = None
    # Passed explicitly to the SDK so an ambient ANTHROPIC_BASE_URL cannot
    # redirect the API key to another host.
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-haiku-4-5"
    anthropic_max_output_tokens: int = 8000
    anthropic_request_timeout: int = 120
    anthropic_budget_usd: float = 20.0
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_max_output_tokens: int = 1200
    deepseek_request_timeout: float = 90.0
    deepseek_budget_usd_per_run: float = 1.0

    filing_lookback_days_10k: int = 540
    filing_lookback_days_10q: int = 210
    filing_lookback_days_8k: int = 90
    filing_lookback_days_def14a: int = 540
    filing_lookback_days_20f: int = 540

    llm_provider: str = "disabled"
    net_provider: str = "enabled"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    openai_temperature: float = 0.0
    openai_max_output_tokens: int = 1200
    openai_request_timeout: float = 45.0
    openai_budget_usd_per_run: float = 1.0
    nightly_research_cycle_enabled: bool = False
    nightly_research_cycle_top_k: int = 10
    research_cycle_max_iterations_default: int = 2
    research_cycle_gap_improvement_threshold: float = 0.5

    discovery_top_k_default: int = 25
    discovery_market_cap_min: float = 5_000_000_000.0
    discovery_market_cap_max: float = 50_000_000_000.0
    discovery_max_tickers_per_run: int = 250
    discovery_missing_cik_stop_ratio: float = 0.4
    discovery_sec_throttle_delta_limit: int = 200
    discovery_max_quarters: int = 4
    discovery_suppress_financials: bool = True
    discovery_suppress_biotech: bool = True
    discovery_suppress_prerevenue: bool = True
    discovery_suppress_rollups: bool = False
    discovery_workers: int = 4
    discovery_prefilter_cap_default: int = 300
    discovery_prefilter_keep_ratio_default: float = 0.3
    discovery_cancel_drain_seconds: float = 2.0
    sector_suppress_financials: bool = False
    sector_suppress_biotech: bool = False
    sector_suppress_prerevenue: bool = False
    sector_suppress_rollups: bool = False
    sector_taxonomy_sparse_threshold: int = 20
    # Additive v2 rollout allowlist. Scheduled sector runs remain on v1 unless
    # their normalized cap-band token appears here; manual requests may still
    # explicitly choose a pipeline version.
    autonomous_sector_v2_cap_bands: list[str] = Field(default_factory=list)
    gate_mos_min: float = 0.30
    gate_valuation_gap_min: float = 0.10
    gate_net_debt_to_cfo_max: float = 2.5
    gate_dilution_max: float = 0.02
    scout_mos_min: float = 0.30
    scout_valuation_gap_min: float = 0.10
    scout_fcf_yield_min: float = 0.03
    scout_net_debt_to_cfo_max: float = 2.5
    scout_dilution_max: float = 0.06
    scout_require_ev_yield: bool = False
    graham_discount_rate_default: float = 0.10
    scout_use_graham_dodd: bool = True
    universe_shortlist_top_n: int = 50
    universe_shortlist_max_per_sector: int = 10
    universe_depth_queue_chunk_size: int = 10
    universe_depth_queue_iterations_default: int = 2
    universe_depth_queue_top_k_default: int = 10
    universe_depth_queue_limit_dossiers_default: int = 10
    universe_tier1_revenue_min_musd: float = 100.0
    universe_continuous_interval_seconds: int = 21600
    composite_gd_mos_high: float = 0.50
    composite_gd_mos_mid: float = 0.30
    composite_gd_mos_low: float = 0.10
    composite_yield_high: float = 0.08
    composite_yield_mid: float = 0.05
    composite_yield_low: float = 0.03
    composite_quality_roic_high: float = 0.15
    composite_quality_roic_mid: float = 0.10
    composite_quality_roic_low: float = 0.05
    composite_risk_dilution_warn: float = 0.06
    composite_risk_leverage_warn: float = 2.5
    composite_risk_keyword_warn: float = 2.0
    promotion_min_appearances_high_priority: int = 2
    promotion_min_implied_return: float = 0.15
    min_sector_allocation: int = 5
    tech_category_cik_overrides: dict[str, str] = Field(
        default_factory=lambda: {
            # Oracle's cloud capex profile can read like hardware without an issuer-aware override.
            "0001341439": "ENTERPRISE_SOFTWARE",
        }
    )
    promotion_terminal_blocker_codes: list[str] = Field(
        default_factory=lambda: ["PRICE_UNKNOWN", "MISSING_GD_INPUTS", "FAIL_CONFIRMED"]
    )
    promotion_lane2_min_score: float = 60.0
    # 'soft' demotes the 12% base-return hurdle to a HURDLE-class confidence
    # cap (WATCHLIST_ONLY), 'hard' preserves the legacy hard-block behavior.
    base_return_hurdle_mode: str = "soft"

    # Adverse-events gate: any fresh 8-K by an at-target name opens an
    # unreviewed_8k queue-protection event during the daily poll.
    events_8k_freshness_enabled: bool = True
    events_8k_freshness_lookback_days: int = 21

    # Adverse-events gate: CourtListener/RECAP docket scan for recent suits
    # against actionable/at-target names (litigation_docket events).
    events_dockets_enabled: bool = False
    courtlistener_api_token: str | None = None
    events_docket_nos_codes: list[int] = Field(default_factory=lambda: [850, 370, 410])
    events_docket_lookback_days: int = 30
    max_requests_courtlistener_domain: int = 300

    # Adverse-events gate: Alpha Vantage NEWS_SENTIMENT keyword scan for
    # at-target names (adverse_news events). Free tier = 25 requests/day.
    events_adverse_news_enabled: bool = False
    events_adverse_news_max_tickers: int = 25


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    tech_category_cik_overrides_env = os.getenv("VOE_TECH_CATEGORY_CIK_OVERRIDES", "").strip()
    try:
        tech_category_cik_overrides = (
            {
                str(key).strip(): str(value).strip()
                for key, value in json.loads(tech_category_cik_overrides_env).items()
                if str(key).strip() and str(value).strip()
            }
            if tech_category_cik_overrides_env
            else {"0001341439": "ENTERPRISE_SOFTWARE"}
        )
    except Exception:
        tech_category_cik_overrides = {"0001341439": "ENTERPRISE_SOFTWARE"}

    cfg = AppConfig(
        data_dir=Path(os.getenv("VOE_DATA_DIR", "data")),
        data_volume_uuid=(os.getenv("VOE_DATA_VOLUME_UUID") or "").strip() or None,
        db_path=Path(os.getenv("VOE_DB_PATH", "data/engine.db")),
        universe_path=Path(os.getenv("VOE_UNIVERSE_PATH", "data/universe/universe.csv")),
        discovery_seed_path=Path(os.getenv("VOE_DISCOVERY_SEED_PATH", "data/universe/discovery_seed.csv")),
        sector_taxonomy_path=Path(os.getenv("VOE_SECTOR_TAXONOMY_PATH", "data/universe/sector_taxonomy.csv")),
        sector_overrides_path=Path(os.getenv("VOE_SECTOR_OVERRIDES_PATH", "data/universe/sector_overrides.csv")),
        sector_sic_config_path=Path(os.getenv("VOE_SECTOR_SIC_CONFIG_PATH", "data/universe/sector_sic_ranges.json")),
        sec_user_agent=os.getenv("VOE_SEC_USER_AGENT", ""),
        sec_rate_limit_per_sec=float(os.getenv("VOE_SEC_RPS", "5")),
        sec_max_retries=int(os.getenv("VOE_SEC_MAX_RETRIES", "4")),
        sec_backoff_seconds=float(os.getenv("VOE_SEC_BACKOFF", "1.5")),
        http_timeout_seconds=float(os.getenv("VOE_HTTP_TIMEOUT", "30")),
        sqlite_busy_timeout_ms=int(os.getenv("VOE_SQLITE_BUSY_TIMEOUT_MS", "300000")),
        allow_fred=os.getenv("VOE_ALLOW_FRED", "false").lower() == "true",
        fred_api_key=os.getenv("VOE_FRED_API_KEY") or os.getenv("FRED_API_KEY"),
        price_provider=os.getenv("VOE_PRICE_PROVIDER", "auto"),
        price_fallback_days=int(os.getenv("VOE_PRICE_FALLBACK_DAYS", "7")),
        price_symbol_overrides_path=Path(os.getenv("VOE_PRICE_SYMBOL_OVERRIDES_PATH", "config/price_symbol_overrides.csv")),
        shares_allow_market_cap_price_derive=os.getenv("VOE_SHARES_ALLOW_MKTCAP_DERIVE", "false").lower() == "true",
        issuer_classification_by_sic=os.getenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "true").lower() == "true",
        quote_ttl_seconds=int(os.getenv("VOE_QUOTE_TTL_SECONDS", "86400")),
        stooq_base_url=os.getenv("VOE_STOOQ_BASE_URL", "https://stooq.com/q/l/"),
        stooq_history_url=os.getenv("VOE_STOOQ_HISTORY_URL", "https://stooq.com/q/d/l/"),
        stooq_apikey=os.getenv("VOE_STOOQ_APIKEY") or None,
        stooq_apikey_param=os.getenv("VOE_STOOQ_APIKEY_PARAM", "apikey"),
        eodhd_base_url=os.getenv("VOE_EODHD_BASE_URL", "https://eodhd.com"),
        eodhd_apikey=os.getenv("VOE_EODHD_APIKEY") or None,
        eodhd_exchange=os.getenv("VOE_EODHD_EXCHANGE", "US"),
        cap_eodhd_fundamentals_enabled=os.getenv("VOE_CAP_EODHD_FUNDAMENTALS_ENABLED", "false").lower() == "true",
        cap_terminal_evidence_path=(
            Path(os.environ["VOE_CAP_TERMINAL_EVIDENCE_PATH"])
            if os.getenv("VOE_CAP_TERMINAL_EVIDENCE_PATH", "").strip()
            else None
        ),
        safe_mode=os.getenv("VOE_SAFE_MODE", "true").lower() == "true",
        scheduler_poll_seconds=int(os.getenv("VOE_SCHEDULER_POLL_SECONDS", "10")),
        scheduler_worker_count=int(os.getenv("VOE_MAX_WORKERS", os.getenv("VOE_WORKER_COUNT", "1"))),
        max_workers=int(os.getenv("VOE_MAX_WORKERS", os.getenv("VOE_WORKER_COUNT", "1"))),
        max_db_write_concurrency=int(os.getenv("VOE_MAX_DB_WRITE_CONCURRENCY", "1")),
        research_max_http_concurrency=int(os.getenv("VOE_RESEARCH_MAX_HTTP_CONCURRENCY", "2")),
        running_job_timeout_seconds=int(os.getenv("VOE_RUNNING_JOB_TIMEOUT_SECONDS", "1800")),
        max_requests_sec_domain=int(os.getenv("VOE_MAX_REQUESTS_SEC_DOMAIN", "1200")),
        max_requests_data_sec_domain=int(
            os.getenv("VOE_MAX_REQUESTS_DATA_SEC_DOMAIN", os.getenv("VOE_MAX_REQUESTS_SEC_DOMAIN", "1200"))
        ),
        max_requests_www_sec_domain=int(
            os.getenv("VOE_MAX_REQUESTS_WWW_SEC_DOMAIN", os.getenv("VOE_MAX_REQUESTS_SEC_DOMAIN", "1200"))
        ),
        max_requests_stooq_domain=int(os.getenv("VOE_MAX_REQUESTS_STOOQ_DOMAIN", "600")),
        max_requests_eodhd_domain=int(os.getenv("VOE_MAX_REQUESTS_EODHD_DOMAIN", "600")),
        max_requests_alpha_vantage_domain=int(os.getenv("VOE_MAX_REQUESTS_ALPHA_VANTAGE_DOMAIN", "300")),
        max_requests_wikipedia_domain=int(os.getenv("VOE_MAX_REQUESTS_WIKIPEDIA_DOMAIN", "300")),
        max_requests_press_domain=int(os.getenv("VOE_MAX_REQUESTS_PRESS_DOMAIN", "300")),
        research_enable_in_run_all=os.getenv("VOE_RESEARCH_ENABLE_IN_RUN_ALL", "false").lower() == "true",
        research_top_n=int(os.getenv("VOE_RESEARCH_TOP_N", "10")),
        research_quality_threshold=float(os.getenv("VOE_RESEARCH_QUALITY_THRESHOLD", "50")),
        research_enable_transcripts=os.getenv("VOE_RESEARCH_ENABLE_TRANSCRIPTS", "false").lower() == "true",
        research_transcript_provider=os.getenv("VOE_RESEARCH_TRANSCRIPT_PROVIDER", "disabled"),
        research_alpha_vantage_api_key=(
            os.getenv("VOE_ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
        ),
        research_transcript_max_quarters=int(os.getenv("VOE_RESEARCH_TRANSCRIPT_MAX_QUARTERS", "4")),
        research_ir_press_enabled=os.getenv("VOE_RESEARCH_IR_PRESS_ENABLED", "false").lower() == "true",
        research_allowlist_domains=[
            host.strip().lower()
            for host in os.getenv("VOE_RESEARCH_ALLOWLIST_DOMAINS", "").split(",")
            if host.strip()
        ],
        research_ir_max_items_per_ticker=int(os.getenv("VOE_RESEARCH_IR_MAX_ITEMS_PER_TICKER", "10")),
        research_ir_max_days_back=int(os.getenv("VOE_RESEARCH_IR_MAX_DAYS_BACK", "365")),
        research_company_news_enabled=os.getenv("VOE_RESEARCH_COMPANY_NEWS_ENABLED", "true").lower() == "true",
        research_company_news_max_pages=int(os.getenv("VOE_RESEARCH_COMPANY_NEWS_MAX_PAGES", "8")),
        research_external_news_enabled=os.getenv("VOE_RESEARCH_EXTERNAL_NEWS_ENABLED", "false").lower() == "true",
        research_external_news_provider=os.getenv("VOE_RESEARCH_EXTERNAL_NEWS_PROVIDER", "disabled"),
        research_external_news_max_items=int(os.getenv("VOE_RESEARCH_EXTERNAL_NEWS_MAX_ITEMS", "10")),
        research_source_reputation_path=Path(
            os.getenv("VOE_RESEARCH_SOURCE_REPUTATION_PATH", "config/source_reputation_history.csv")
        ),
        research_exhibits_enabled=os.getenv("VOE_RESEARCH_EXHIBITS_ENABLED", "true").lower() == "true",
        research_enable_wikipedia=os.getenv("VOE_RESEARCH_ENABLE_WIKIPEDIA", "false").lower() == "true",
        analyst_notes_enabled=os.getenv("VOE_ANALYST_NOTES", "disabled").strip().lower() == "enabled",
        anthropic_api_key=os.getenv("VOE_ANTHROPIC_API_KEY"),
        anthropic_base_url=os.getenv("VOE_ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
        anthropic_model=os.getenv("VOE_ANTHROPIC_MODEL", "claude-haiku-4-5"),
        anthropic_max_output_tokens=int(os.getenv("VOE_ANTHROPIC_MAX_OUTPUT_TOKENS", "8000")),
        anthropic_request_timeout=int(os.getenv("VOE_ANTHROPIC_REQUEST_TIMEOUT", "120")),
        anthropic_budget_usd=float(os.getenv("VOE_ANTHROPIC_BUDGET_USD", "20.0")),
        deepseek_api_key=os.getenv("VOE_DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("VOE_DEEPSEEK_MODEL", "deepseek-v4-pro"),
        deepseek_max_output_tokens=int(
            os.getenv("VOE_DEEPSEEK_MAX_OUTPUT_TOKENS", "1200")
        ),
        deepseek_request_timeout=float(
            os.getenv("VOE_DEEPSEEK_REQUEST_TIMEOUT", "90")
        ),
        deepseek_budget_usd_per_run=float(
            os.getenv("VOE_DEEPSEEK_BUDGET_USD_PER_RUN", "1.0")
        ),
        filing_lookback_days_10k=int(os.getenv("VOE_LOOKBACK_10K_DAYS", "540")),
        filing_lookback_days_10q=int(os.getenv("VOE_LOOKBACK_10Q_DAYS", "210")),
        filing_lookback_days_8k=int(os.getenv("VOE_LOOKBACK_8K_DAYS", "90")),
        filing_lookback_days_def14a=int(os.getenv("VOE_LOOKBACK_DEF14A_DAYS", "540")),
        filing_lookback_days_20f=int(os.getenv("VOE_LOOKBACK_20F_DAYS", "540")),
        llm_provider=os.getenv("VOE_LLM_PROVIDER", "disabled"),
        net_provider=os.getenv("VOE_NET_PROVIDER", "enabled"),
        openai_api_key=os.getenv("VOE_OPENAI_API_KEY"),
        openai_model=os.getenv("VOE_OPENAI_MODEL", "gpt-5-mini"),
        openai_temperature=float(os.getenv("VOE_OPENAI_TEMPERATURE", "0")),
        openai_max_output_tokens=int(os.getenv("VOE_OPENAI_MAX_OUTPUT_TOKENS", "1200")),
        openai_request_timeout=float(os.getenv("VOE_OPENAI_REQUEST_TIMEOUT", "45")),
        openai_budget_usd_per_run=float(os.getenv("VOE_OPENAI_BUDGET_USD_PER_RUN", "1.0")),
        nightly_research_cycle_enabled=os.getenv("VOE_NIGHTLY_RESEARCH_CYCLE_ENABLED", "false").lower() == "true",
        nightly_research_cycle_top_k=int(os.getenv("VOE_NIGHTLY_RESEARCH_CYCLE_TOP_K", "10")),
        research_cycle_max_iterations_default=int(os.getenv("VOE_RESEARCH_CYCLE_MAX_ITERATIONS", "2")),
        research_cycle_gap_improvement_threshold=float(os.getenv("VOE_RESEARCH_CYCLE_GAP_IMPROVEMENT_THRESHOLD", "0.5")),
        discovery_top_k_default=int(os.getenv("VOE_DISCOVERY_TOP_K_DEFAULT", "25")),
        discovery_market_cap_min=float(os.getenv("VOE_DISCOVERY_MCAP_MIN", "5000000000")),
        discovery_market_cap_max=float(os.getenv("VOE_DISCOVERY_MCAP_MAX", "50000000000")),
        discovery_max_tickers_per_run=int(os.getenv("VOE_DISCOVERY_MAX_TICKERS", "250")),
        discovery_missing_cik_stop_ratio=float(os.getenv("VOE_DISCOVERY_MISSING_CIK_STOP_RATIO", "0.4")),
        discovery_sec_throttle_delta_limit=int(os.getenv("VOE_DISCOVERY_SEC_THROTTLE_LIMIT", "200")),
        discovery_max_quarters=int(os.getenv("VOE_DISCOVERY_MAX_QUARTERS", "4")),
        discovery_suppress_financials=os.getenv("VOE_DISCOVERY_SUPPRESS_FINANCIALS", "true").lower() == "true",
        discovery_suppress_biotech=os.getenv("VOE_DISCOVERY_SUPPRESS_BIOTECH", "true").lower() == "true",
        discovery_suppress_prerevenue=os.getenv("VOE_DISCOVERY_SUPPRESS_PREREVENUE", "true").lower() == "true",
        discovery_suppress_rollups=os.getenv("VOE_DISCOVERY_SUPPRESS_ROLLUPS", "false").lower() == "true",
        discovery_workers=int(os.getenv("VOE_DISCOVERY_WORKERS", "4")),
        discovery_prefilter_cap_default=int(os.getenv("VOE_DISCOVERY_PREFILTER_CAP", "300")),
        discovery_prefilter_keep_ratio_default=float(os.getenv("VOE_DISCOVERY_PREFILTER_KEEP_RATIO", "0.3")),
        discovery_cancel_drain_seconds=float(os.getenv("VOE_DISCOVERY_CANCEL_DRAIN_SECONDS", "2.0")),
        sector_suppress_financials=os.getenv("VOE_SECTOR_SUPPRESS_FINANCIALS", "false").lower() == "true",
        sector_suppress_biotech=os.getenv("VOE_SECTOR_SUPPRESS_BIOTECH", "false").lower() == "true",
        sector_suppress_prerevenue=os.getenv("VOE_SECTOR_SUPPRESS_PREREVENUE", "false").lower() == "true",
        sector_suppress_rollups=os.getenv("VOE_SECTOR_SUPPRESS_ROLLUPS", "false").lower() == "true",
        sector_taxonomy_sparse_threshold=int(os.getenv("VOE_SECTOR_TAXONOMY_SPARSE_THRESHOLD", "20")),
        autonomous_sector_v2_cap_bands=list(
            dict.fromkeys(
                token.strip().lower()
                for token in os.getenv("VOE_AUTONOMOUS_SECTOR_V2_CAP_BANDS", "").split(",")
                if token.strip()
            )
        ),
        gate_mos_min=float(os.getenv("VOE_GATE_MOS_MIN", "0.30")),
        gate_valuation_gap_min=float(os.getenv("VOE_GATE_VALUATION_GAP_MIN", "0.10")),
        gate_net_debt_to_cfo_max=float(os.getenv("VOE_GATE_NET_DEBT_TO_CFO_MAX", "2.5")),
        gate_dilution_max=float(os.getenv("VOE_GATE_DILUTION_MAX", "0.02")),
        scout_mos_min=float(os.getenv("VOE_SCOUT_MOS_MIN", "0.30")),
        scout_valuation_gap_min=float(os.getenv("VOE_SCOUT_VALUATION_GAP_MIN", "0.10")),
        scout_fcf_yield_min=float(os.getenv("VOE_SCOUT_FCF_YIELD_MIN", "0.03")),
        scout_net_debt_to_cfo_max=float(os.getenv("VOE_SCOUT_NET_DEBT_TO_CFO_MAX", "2.5")),
        scout_dilution_max=float(os.getenv("VOE_SCOUT_DILUTION_MAX", "0.06")),
        scout_require_ev_yield=os.getenv("VOE_SCOUT_REQUIRE_EV_YIELD", "false").lower() == "true",
        graham_discount_rate_default=float(os.getenv("VOE_GRAHAM_DISCOUNT_RATE_DEFAULT", "0.10")),
        scout_use_graham_dodd=os.getenv("VOE_SCOUT_USE_GRAHAM_DODD", "true").lower() == "true",
        universe_shortlist_top_n=int(os.getenv("VOE_UNIVERSE_SHORTLIST_TOP_N", "50")),
        universe_shortlist_max_per_sector=int(os.getenv("VOE_UNIVERSE_SHORTLIST_MAX_PER_SECTOR", "10")),
        universe_depth_queue_chunk_size=int(os.getenv("VOE_UNIVERSE_DEPTH_QUEUE_CHUNK_SIZE", "10")),
        universe_depth_queue_iterations_default=int(os.getenv("VOE_UNIVERSE_DEPTH_QUEUE_ITERATIONS_DEFAULT", "2")),
        universe_depth_queue_top_k_default=int(os.getenv("VOE_UNIVERSE_DEPTH_QUEUE_TOP_K_DEFAULT", "10")),
        universe_depth_queue_limit_dossiers_default=int(
            os.getenv("VOE_UNIVERSE_DEPTH_QUEUE_LIMIT_DOSSIERS_DEFAULT", "10")
        ),
        universe_tier1_revenue_min_musd=float(os.getenv("VOE_UNIVERSE_TIER1_REVENUE_MIN_MUSD", "100")),
        universe_continuous_interval_seconds=int(os.getenv("VOE_UNIVERSE_CONTINUOUS_INTERVAL_SECONDS", "21600")),
        composite_gd_mos_high=float(os.getenv("VOE_COMPOSITE_GD_MOS_HIGH", "0.50")),
        composite_gd_mos_mid=float(os.getenv("VOE_COMPOSITE_GD_MOS_MID", "0.30")),
        composite_gd_mos_low=float(os.getenv("VOE_COMPOSITE_GD_MOS_LOW", "0.10")),
        composite_yield_high=float(os.getenv("VOE_COMPOSITE_YIELD_HIGH", "0.08")),
        composite_yield_mid=float(os.getenv("VOE_COMPOSITE_YIELD_MID", "0.05")),
        composite_yield_low=float(os.getenv("VOE_COMPOSITE_YIELD_LOW", "0.03")),
        composite_quality_roic_high=float(os.getenv("VOE_COMPOSITE_QUALITY_ROIC_HIGH", "0.15")),
        composite_quality_roic_mid=float(os.getenv("VOE_COMPOSITE_QUALITY_ROIC_MID", "0.10")),
        composite_quality_roic_low=float(os.getenv("VOE_COMPOSITE_QUALITY_ROIC_LOW", "0.05")),
        composite_risk_dilution_warn=float(os.getenv("VOE_COMPOSITE_RISK_DILUTION_WARN", "0.06")),
        composite_risk_leverage_warn=float(os.getenv("VOE_COMPOSITE_RISK_LEVERAGE_WARN", "2.5")),
        composite_risk_keyword_warn=float(os.getenv("VOE_COMPOSITE_RISK_KEYWORD_WARN", "2.0")),
        promotion_min_appearances_high_priority=int(os.getenv("VOE_PROMOTION_MIN_APPEARANCES_HIGH_PRIORITY", "2")),
        promotion_min_implied_return=float(os.getenv("VOE_PROMOTION_MIN_IMPLIED_RETURN", "0.15")),
        min_sector_allocation=int(os.getenv("VOE_MIN_SECTOR_ALLOCATION", "5")),
        tech_category_cik_overrides=tech_category_cik_overrides,
        promotion_terminal_blocker_codes=[
            token.strip().upper()
            for token in os.getenv("VOE_PROMOTION_TERMINAL_BLOCKER_CODES", "PRICE_UNKNOWN,MISSING_GD_INPUTS,FAIL_CONFIRMED").split(",")
            if token.strip()
        ],
        promotion_lane2_min_score=float(os.getenv("VOE_PROMOTION_LANE2_MIN_SCORE", "60.0")),
        base_return_hurdle_mode=os.getenv("BASE_RETURN_HURDLE_MODE", "soft").strip().lower(),
        events_8k_freshness_enabled=os.getenv("VOE_EVENTS_8K_FRESHNESS_ENABLED", "true").lower() == "true",
        events_8k_freshness_lookback_days=int(os.getenv("VOE_EVENTS_8K_FRESHNESS_DAYS", "21")),
        events_dockets_enabled=os.getenv("VOE_EVENTS_DOCKETS_ENABLED", "false").lower() == "true",
        courtlistener_api_token=os.getenv("VOE_COURTLISTENER_API_TOKEN") or None,
        events_docket_nos_codes=[
            int(code.strip())
            for code in os.getenv("VOE_EVENTS_DOCKET_NOS_CODES", "850,370,410").split(",")
            if code.strip().isdigit()
        ],
        events_docket_lookback_days=int(os.getenv("VOE_EVENTS_DOCKET_LOOKBACK_DAYS", "30")),
        max_requests_courtlistener_domain=int(os.getenv("VOE_MAX_REQUESTS_COURTLISTENER_DOMAIN", "300")),
        events_adverse_news_enabled=os.getenv("VOE_EVENTS_ADVERSE_NEWS_ENABLED", "false").lower() == "true",
        events_adverse_news_max_tickers=int(os.getenv("VOE_EVENTS_ADVERSE_NEWS_MAX_TICKERS", "25")),
    )

    # Fail-loud ops: anchor relative data paths to the repo root. A cron or
    # CLI invoked from the wrong cwd would otherwise create an empty shadow
    # engine.db (zero-byte app.db/facts.db/ivi.db copies have been found on
    # disk) that renders as a calm "no names at target".
    def _anchor(path: Path) -> Path:
        return path if path.is_absolute() else cfg.project_root / path

    cfg.data_dir = _anchor(cfg.data_dir)
    # Unset VOE_DB_PATH follows VOE_DATA_DIR: the database lives with the data.
    cfg.db_path = (
        _anchor(cfg.db_path) if os.getenv("VOE_DB_PATH") else cfg.data_dir / "engine.db"
    )
    cfg.price_symbol_overrides_path = _anchor(cfg.price_symbol_overrides_path)

    _sentinel_override = os.getenv("VOE_DATA_VOLUME_SENTINEL")
    cfg.data_volume_sentinel = (
        _anchor(Path(_sentinel_override)) if _sentinel_override else cfg.data_dir / ".ivi-volume"
    )

    cfg.raw_filings_dir = cfg.data_dir / "raw_filings"
    cfg.cache_dir = cfg.data_dir / "cache"
    cfg.outputs_dir = cfg.data_dir / "outputs"
    cfg.evidence_dir = cfg.outputs_dir / "evidence_packets"
    cfg.analyst_outputs_dir = cfg.outputs_dir / "analyst_outputs"
    cfg.memos_dir = cfg.outputs_dir / "memos"
    cfg.gaps_dir = cfg.outputs_dir / "gaps"
    cfg.deltas_dir = cfg.outputs_dir / "deltas"
    cfg.delta_memos_dir = cfg.outputs_dir / "delta_memos"
    cfg.shortlists_dir = cfg.outputs_dir / "shortlists"
    cfg.synthesis_dir = cfg.outputs_dir / "synthesis"
    cfg.discovery_dir = cfg.outputs_dir / "discovery"
    cfg.dossiers_dir = cfg.outputs_dir / "dossiers"
    cfg.sectors_dir = cfg.outputs_dir / "sectors"
    cfg.calibration_dir = cfg.outputs_dir / "calibration"
    cfg.rankings_dir = cfg.outputs_dir / "rankings"
    cfg.campaigns_dir = cfg.outputs_dir / "campaigns"
    cfg.status_path = cfg.outputs_dir / "agent_status.json"
    cfg.manifests_dir = cfg.outputs_dir / "manifests"
    cfg.run_manifest_path = cfg.manifests_dir / "run_manifest.json"
    cfg.runs_dir = cfg.outputs_dir / "runs"
    cfg.runs_index_path = cfg.runs_dir / "index.json"
    cfg.research_dir = cfg.outputs_dir / "research"
    cfg.universe_dir = cfg.data_dir / "universe"
    cfg.universe_path = _anchor(Path(os.getenv("VOE_UNIVERSE_PATH", str(cfg.universe_dir / "universe.csv"))))
    cfg.discovery_seed_path = _anchor(Path(os.getenv("VOE_DISCOVERY_SEED_PATH", str(cfg.universe_dir / "discovery_seed.csv"))))
    cfg.sector_taxonomy_path = _anchor(Path(os.getenv("VOE_SECTOR_TAXONOMY_PATH", str(cfg.universe_dir / "sector_taxonomy.csv"))))
    cfg.sector_overrides_path = _anchor(Path(os.getenv("VOE_SECTOR_OVERRIDES_PATH", str(cfg.universe_dir / "sector_overrides.csv"))))
    cfg.sector_sic_config_path = _anchor(Path(os.getenv("VOE_SECTOR_SIC_CONFIG_PATH", str(cfg.universe_dir / "sector_sic_ranges.json"))))

    return cfg


def canonical_market_cap_focus(value: str | None) -> str:
    """Return the canonical key used for autonomous-sector cap bands."""

    normalized = (
        str(value or "small_cap")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return {
        "micro": "micro_cap",
        "small": "small_cap",
        "smid": "smid_cap",
        "mid": "mid_cap",
        "large": "large_cap",
        "mega": "mega_cap",
    }.get(normalized, normalized)


def resolve_autonomous_sector_pipeline_version(
    market_cap_focus: str,
    requested: str | None = None,
    *,
    cfg: AppConfig | None = None,
) -> str:
    """Resolve manual override or the cap-band rollout allowlist to v1/v2."""

    if requested is not None:
        explicit = requested.strip().lower()
        if explicit not in {"v1", "v2"}:
            raise ValueError("pipeline version must be 'v1' or 'v2'")
        return explicit

    config = cfg or get_config()
    band = canonical_market_cap_focus(market_cap_focus)
    enabled_bands = {
        canonical_market_cap_focus(token)
        for token in config.autonomous_sector_v2_cap_bands
        if token.strip()
    }
    return "v2" if band in enabled_bands else "v1"


class DataVolumeError(RuntimeError):
    """The external data volume is absent, or is not the volume we expect."""


def check_data_volume(cfg: AppConfig | None = None) -> tuple[bool, str]:
    """Is the external data volume mounted, and is it the right one?

    Three questions, in order, because each can be true while the next is
    false:

    1. Does the sentinel resolve to a real file? A symlink onto an unplugged
       drive dangles, which is the plain "drive is gone" case.
    2. Is it on a *different device* than the repo? macOS remounts a drive
       whose mount point is occupied as "/Volumes/<name> 1", leaving an
       ordinary directory behind on the internal disk. If anything ever
       copies ivi-data into that directory, the sentinel reads back
       perfectly while every write lands on the internal disk -- the exact
       second-copy-of-the-world this guard exists to prevent. Only the
       device number can tell those apart, so a *symlinked* sentinel (the
       deployed layout) must resolve off the repo's own device.
    3. Does it carry the UUID we expect? This catches a different drive
       mounted at the same path.

    The check arms itself: a symlinked sentinel means this checkout is
    provisioned for a volume, so 1 and 2 apply even with no UUID configured.
    A checkout with no sentinel at all and no VOE_DATA_VOLUME_UUID is a
    developer clone and is left alone. Returns (ok, detail); the detail is
    quoted verbatim by preflight, deadman and the cron shell guard.
    """
    cfg = cfg or get_config()
    expected = (cfg.data_volume_uuid or "").strip()
    sentinel = Path(cfg.data_volume_sentinel)
    linked = sentinel.is_symlink()

    if not expected and not linked:
        return True, f"disabled (no volume sentinel at {sentinel})"

    if not sentinel.is_file():
        if linked:
            return (
                False,
                f"volume not mounted: dangling sentinel {sentinel} -> "
                f"{os.readlink(sentinel)}",
            )
        if sentinel.is_dir():
            return False, f"sentinel is a directory, not a file: {sentinel}"
        return False, f"volume not mounted: sentinel missing {sentinel}"

    try:
        sentinel_device = sentinel.stat().st_dev
        repo_device = Path(cfg.project_root).stat().st_dev
    except OSError as exc:
        return False, f"sentinel unreadable {sentinel}: {exc}"

    if linked and sentinel_device == repo_device:
        return (
            False,
            f"volume not mounted: {sentinel} resolves onto the repo's own device "
            f"({sentinel_device}) -- a leftover mount-point directory, not the drive",
        )

    try:
        found = sentinel.read_text(encoding="utf-8").strip()
    except (OSError, ValueError) as exc:
        return False, f"sentinel unreadable {sentinel}: {type(exc).__name__}: {exc}"

    if not expected:
        return True, f"{sentinel} -> {found or '(empty)'} (UUID unchecked; VOE_DATA_VOLUME_UUID unset)"
    if found.upper() != expected.upper():
        return (
            False,
            f"wrong volume at {sentinel}: found {found or '(empty)'}, expected {expected}",
        )
    return True, f"{sentinel} -> {found}"


def assert_data_volume(cfg: AppConfig | None = None) -> None:
    """check_data_volume, raising DataVolumeError with the failing detail."""
    ok, detail = check_data_volume(cfg)
    if not ok:
        raise DataVolumeError(detail)


def ensure_directories(cfg: AppConfig) -> None:
    # Never mkdir -p through an unmounted volume root: that is how the
    # second copy of the world gets written to the internal disk.
    assert_data_volume(cfg)
    for path in [
        cfg.data_dir,
        cfg.raw_filings_dir,
        cfg.cache_dir,
        cfg.outputs_dir,
        cfg.evidence_dir,
        cfg.analyst_outputs_dir,
        cfg.memos_dir,
        cfg.gaps_dir,
        cfg.deltas_dir,
        cfg.delta_memos_dir,
        cfg.shortlists_dir,
        cfg.synthesis_dir,
        cfg.discovery_dir,
        cfg.dossiers_dir,
        cfg.sectors_dir,
        cfg.calibration_dir,
        cfg.rankings_dir,
        cfg.manifests_dir,
        cfg.runs_dir,
        cfg.research_dir,
        cfg.universe_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
