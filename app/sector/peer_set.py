from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.dossier.collector import ANNUAL_FORM_TYPES_WITH_AMENDMENTS, preflight_annual_eligibility
from app.discovery.market_cap import UNKNOWN, compute_market_cap
from app.ingest.sec_client import SecClient
from app.sector.taxonomy import load_sector_taxonomy
from app.universe.sector_universe import discover_sector_universe, has_sector_definition
from app.universe.ticker_cik_map import load_ticker_cik_map
from app.util.financial_data_access import ANNUAL_CACHED_FILING_FORM_TYPES, latest_filing_row
from app.util.http import DomainBudgetExceeded
from app.valuation.price_provider import get_default_provider


PEER_MODES = {"taxonomy", "filings", "hybrid"}

CATEGORY_PRIORITY = [
    "software",
    "consumer",
    "financials",
    "healthcare",
    "energy",
    "industrials",
    "materials",
    "media",
]

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "software": (
        "software",
        "cloud",
        "saas",
        "platform",
        "semiconductor",
        "chip",
        "cybersecurity",
        "data center",
        "enterprise applications",
    ),
    "consumer": (
        "consumer",
        "retail",
        "e-commerce",
        "apparel",
        "restaurant",
        "travel",
        "lodging",
        "automotive",
        "home improvement",
    ),
    "financials": (
        "bank",
        "banking",
        "insurance",
        "insurer",
        "asset management",
        "broker",
        "lending",
        "credit",
        "payments",
        "financial services",
    ),
    "healthcare": (
        "healthcare",
        "pharma",
        "pharmaceutical",
        "biotech",
        "biotechnology",
        "therapeutics",
        "medical",
        "diagnostic",
        "hospital",
    ),
    "energy": (
        "energy",
        "oil",
        "gas",
        "petroleum",
        "renewable",
        "solar",
        "utility",
        "power generation",
    ),
    "industrials": (
        "industrial",
        "manufacturing",
        "aerospace",
        "defense",
        "machinery",
        "logistics",
        "transportation",
    ),
    "materials": (
        "materials",
        "chemicals",
        "mining",
        "steel",
        "aluminum",
        "paper",
        "lumber",
        "fertilizer",
    ),
    "media": (
        "media",
        "advertising",
        "entertainment",
        "streaming",
        "social network",
        "telecom",
        "gaming",
        "content",
    ),
}

SECTOR_ALIASES: dict[str, str] = {
    "software": "software",
    "technology": "software",
    "consumer": "consumer",
    "consumer discretionary": "consumer",
    "consumer staples": "consumer",
    "financials": "financials",
    "finance": "financials",
    "healthcare": "healthcare",
    "health care": "healthcare",
    "energy": "energy",
    "utilities": "energy",
    "industrials": "industrials",
    "industrial": "industrials",
    "materials": "materials",
    "basic materials": "materials",
    "media": "media",
    "communication services": "media",
}

SIC_RANGE_CATEGORY: list[tuple[int, int, str]] = [
    (1000, 1499, "materials"),
    (2000, 2399, "consumer"),
    (2400, 2999, "materials"),
    (2830, 2836, "healthcare"),
    (3570, 3579, "software"),
    (3600, 3699, "industrials"),
    (3800, 3851, "healthcare"),
    (4810, 4899, "media"),
    (4900, 4949, "energy"),
    (6000, 6999, "financials"),
    (7000, 7999, "consumer"),
    (8000, 8099, "healthcare"),
]

ITEM_1_BUSINESS_RE = re.compile(r"\bitem\s+1\.?\s+business\b", re.IGNORECASE)
ITEM_1A_RE = re.compile(r"\bitem\s+1a\.?\s+risk\s+factors\b", re.IGNORECASE)
SIC_HEADER_RE = re.compile(
    r"(?:standard\s+industrial\s+classification|sic)\D{0,40}(\d{4})",
    re.IGNORECASE,
)


FINANCIALS_KEYWORDS = (
    "bank",
    "banking",
    "insurance",
    "insurer",
    "broker",
    "lending",
)
BIOTECH_KEYWORDS = (
    "biotech",
    "biotechnology",
    "pharma",
    "pharmaceutical",
    "therapeutics",
    "bio",
)
ROLLUP_KEYWORDS = (
    "acquisition",
    "acquire",
    "merger",
    "rollup",
    "roll-up",
)
OTC_SUFFIXES = ("F", "Y", "Q", "W", "Z")
STAGE_ORDER_HYBRID = ["taxonomy", "sic_expand", "sic_family", "filings_inferred", "seed_fallback"]
STAGE_SOURCE_BY_KEY = {
    "taxonomy": "taxonomy",
    "sic_expand": "sic",
    "sic_family": "sic_family",
    "filings_inferred": "filings",
    "seed_fallback": "seed_fallback",
}
STAGE_LABEL_BY_SOURCE = {
    "taxonomy": "taxonomy",
    "sic": "sic_expand",
    "sic_family": "sic_family",
    "filings": "filings_inferred",
    "seed_fallback": "seed_fallback",
}


def _is_likely_otc_ticker(ticker: str) -> bool:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return False
    if "." in symbol or "/" in symbol:
        return True
    if len(symbol) == 5 and symbol[-1] in OTC_SUFFIXES:
        return True
    return False


def _peer_preflight_annual_forms(*, include_foreign: bool) -> list[str]:
    if include_foreign:
        return list(ANNUAL_FORM_TYPES_WITH_AMENDMENTS)
    return ["10-K", "10-K/A"]


def _seed_fallback_tickers(*, taxonomy: dict[str, str]) -> list[str]:
    cfg = get_config()
    out: set[str] = set()
    seed_path = cfg.discovery_seed_path
    if seed_path.exists():
        try:
            with seed_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    ticker = str(row.get("ticker") or "").strip().upper()
                    if ticker:
                        out.add(ticker)
        except Exception:
            out = set()
    out.update(str(ticker).strip().upper() for ticker in taxonomy.keys() if str(ticker).strip())
    return sorted(out)


def _sort_selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            0 if isinstance(row.get("market_cap"), (int, float)) else 1,
            -float(row["market_cap"]) if isinstance(row.get("market_cap"), (int, float)) else 0.0,
            row["ticker"],
        ),
    )


def _latest_annual_local_path(conn, ticker: str, as_of_date: str) -> str | None:
    row = latest_filing_row(
        conn,
        ticker,
        columns=("local_path",),
        form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
        as_of_date=as_of_date,
        require_local_path=True,
    )
    if not row:
        return None
    local_path = str(row["local_path"] or "").strip()
    return local_path or None


def _extract_business_text(local_path: str | None) -> str:
    if not local_path:
        return ""
    path = Path(local_path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if not text.strip():
        return ""

    start = 0
    end = min(len(text), 40_000)
    match_business = ITEM_1_BUSINESS_RE.search(text)
    if match_business:
        start = int(match_business.start())
        match_risk = ITEM_1A_RE.search(text, pos=match_business.end())
        if match_risk and int(match_risk.start()) > start:
            end = int(match_risk.start())
        else:
            end = min(len(text), start + 80_000)
    chunk = text[start:end]
    return re.sub(r"\s+", " ", chunk).strip().lower()[:20_000]


def _keyword_category_scored(text: str) -> tuple[str | None, int, dict[str, int]]:
    lowered = (text or "").lower()
    if not lowered:
        return None, 0, {}
    category_scores: dict[str, int] = {}
    best_category: str | None = None
    best_score = 0
    for category in CATEGORY_PRIORITY:
        keywords = CATEGORY_KEYWORDS.get(category, ())
        score = sum(lowered.count(keyword) for keyword in keywords)
        category_scores[category] = int(score)
        if score > best_score:
            best_score = score
            best_category = category
    return (best_category if best_score > 0 else None), int(best_score), category_scores


def _keyword_category(text: str) -> str | None:
    category, _, _ = _keyword_category_scored(text)
    return category


def _coerce_sic(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _category_from_sic(sic: int | None, sic_description: str) -> str | None:
    if sic is not None:
        for start, end, category in SIC_RANGE_CATEGORY:
            if start <= int(sic) <= end:
                return category
    return _keyword_category(sic_description)


def _sector_anchor_category(sector: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (sector or "").strip().lower())
    if normalized in SECTOR_ALIASES:
        return SECTOR_ALIASES[normalized]
    return _keyword_category(normalized)


def _ensure_sector_inference_table(conn) -> None:
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


def _ticker_cik_map(conn) -> dict[str, str]:
    rows = conn.execute("SELECT ticker, cik FROM companies ORDER BY ticker").fetchall()
    ticker_to_cik: dict[str, str] = {}
    for row in rows:
        ticker = str(row["ticker"] or "").strip().upper()
        cik = str(row["cik"] or "").strip()
        if ticker and cik:
            ticker_to_cik[ticker] = cik
    cached_map = load_ticker_cik_map(refresh_if_missing=False)
    for ticker, cik in sorted(cached_map.items()):
        ticker_norm = str(ticker or "").strip().upper()
        cik_norm = str(cik or "").strip()
        if ticker_norm and cik_norm and ticker_norm not in ticker_to_cik:
            ticker_to_cik[ticker_norm] = cik_norm
    return ticker_to_cik


def _extract_sic_from_filing_header(local_path: str | None) -> int | None:
    if not local_path:
        return None
    path = Path(local_path)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if not text:
        return None
    head = text[:50_000]
    match = SIC_HEADER_RE.search(head)
    if not match:
        return None
    return _coerce_sic(match.group(1))


def _resolve_ticker_sic(
    conn,
    *,
    ticker: str,
    cik_by_ticker: dict[str, str],
    as_of_date: str,
    sec_client: SecClient,
    sec_cache: dict[str, dict[str, Any]],
    allow_sec_network: bool = True,
) -> tuple[int | None, str]:
    ticker_norm = str(ticker or "").strip().upper()
    if not ticker_norm:
        return None, "ticker_missing"
    cik = str(cik_by_ticker.get(ticker_norm, "")).strip()
    if cik and cik.isdigit():
        if cik in sec_cache:
            metadata = sec_cache[cik]
        elif allow_sec_network:
            try:
                metadata = sec_client.submissions(cik)
            except Exception:
                metadata = {}
            sec_cache[cik] = metadata
        else:
            metadata = {}
        sic = _coerce_sic(metadata.get("sic"))
        if sic is not None:
            return sic, "sec_submissions"
    sic_from_header = _extract_sic_from_filing_header(_latest_annual_local_path(conn, ticker_norm, as_of_date))
    if sic_from_header is not None:
        return sic_from_header, "filing_header"
    return None, "unknown"


def _infer_filings_category(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    sec_client: SecClient,
    sec_cache: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    _ = (sec_client, sec_cache)  # Category inference is keyword-scored and cached in SQLite.
    ticker_norm = str(ticker or "").strip().upper()
    _ensure_sector_inference_table(conn)
    cached = conn.execute(
        """
        SELECT inferred_sector
        FROM sector_inference
        WHERE ticker = ? AND as_of_date = ?
        LIMIT 1
        """,
        (ticker_norm, as_of_date),
    ).fetchone()
    if cached:
        inferred = str(cached["inferred_sector"] or "").strip().lower()
        if inferred:
            return inferred, "keywords_cache"
        return None, "keywords_cache"

    local_path = _latest_annual_local_path(conn, ticker_norm, as_of_date)
    business_text = _extract_business_text(local_path)
    inferred, score, score_map = _keyword_category_scored(business_text)
    if inferred:
        top_keywords = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))[:3]
        derived_from = [
            f"filings.latest_annual_local_path:{local_path or 'MISSING'}",
            f"filings.item1_keywords.{inferred}:{int(score)}",
            f"filings.item1_keyword_scores:{';'.join([f'{name}:{value}' for name, value in top_keywords])}",
        ]
    else:
        derived_from = [
            f"filings.latest_annual_local_path:{local_path or 'MISSING'}",
            "filings.item1_keywords:UNKNOWN",
        ]
    conn.execute(
        """
        INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, as_of_date) DO UPDATE SET
            inferred_sector = excluded.inferred_sector,
            score = excluded.score,
            derived_from = excluded.derived_from,
            created_at = excluded.created_at
        """,
        (
            ticker_norm,
            as_of_date,
            inferred,
            float(score),
            json.dumps(derived_from),
            utc_now_iso(),
        ),
    )
    if inferred:
        return inferred, "keywords"
    return None, "unknown"


def _candidate_row(
    conn,
    *,
    ticker: str,
    source: str,
    sector_value: str | None,
    inferred_category: str | None,
    inferred_source: str | None,
    as_of_date: str,
    cap_min: float,
    cap_max: float,
    provider,
    run_id: str,
    cfg,
) -> dict[str, Any]:
    company = conn.execute(
        "SELECT name, cik FROM companies WHERE ticker = ? LIMIT 1",
        (ticker,),
    ).fetchone()
    company_name = str(company["name"] or "") if company else ""
    cik_value = str(company["cik"] or "").strip() if company else ""
    revenue = _latest_revenue(conn, ticker, as_of_date)
    shares_outstanding, shares_citation = _latest_shares(conn, ticker, as_of_date)
    market_cap = compute_market_cap(
        ticker=ticker,
        run_id=run_id,
        run_as_of_date=as_of_date,
        effective_as_of_date=as_of_date,
        shares_outstanding=shares_outstanding,
        price_provider=provider,
        cap_min=cap_min,
        cap_max=cap_max,
    )
    suppressions = _suppression_reasons(cfg, company_name=company_name, revenue=revenue)
    include = True
    excluded_reason = None
    if suppressions:
        include = False
        excluded_reason = ",".join(suppressions)
    elif isinstance(market_cap.market_cap, (int, float)) and not market_cap.market_cap_in_band:
        include = False
        excluded_reason = "MARKET_CAP_OUT_OF_BAND"

    return {
        "ticker": ticker,
        "peer_source": source,
        "stage_source": STAGE_LABEL_BY_SOURCE.get(source, source),
        "cik": cik_value or None,
        "sector": sector_value,
        "inferred_category": inferred_category,
        "inferred_category_source": inferred_source,
        "company_name": company_name or None,
        "market_cap": market_cap.market_cap,
        "market_cap_status": market_cap.market_cap_status,
        "market_cap_in_band": bool(market_cap.market_cap_in_band),
        "price": market_cap.price,
        "price_provider": market_cap.provider,
        "shares_outstanding": shares_outstanding,
        "shares_citation": shares_citation,
        "suppression_reasons": suppressions,
        "excluded_reason": excluded_reason,
        "selected": include,
    }


def _latest_revenue(conn, ticker: str, as_of_date: str) -> float | str:
    row = conn.execute(
        """
        SELECT metrics_json
        FROM fundamentals
        WHERE ticker = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if not row:
        return UNKNOWN
    try:
        metrics = json.loads(row["metrics_json"] or "{}")
    except Exception:
        metrics = {}
    value = metrics.get("revenue", UNKNOWN)
    return float(value) if isinstance(value, (int, float)) else UNKNOWN


def _latest_shares(conn, ticker: str, as_of_date: str) -> tuple[float | str, dict[str, Any] | None]:
    row = conn.execute(
        """
        SELECT ef.value_json, ef.source_url, ef.snippet, f.filing_date
        FROM extracted_facts ef
        JOIN filings f ON f.id = ef.filing_id
        WHERE f.ticker = ?
          AND ef.fact_type = 'shares_outstanding'
          AND COALESCE(f.filing_date, '1900-01-01') <= ?
        ORDER BY COALESCE(f.filing_date, '1900-01-01') DESC, ef.id DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if not row:
        return UNKNOWN, None
    try:
        payload = json.loads(row["value_json"] or "{}")
    except Exception:
        payload = {}
    value = payload.get("value")
    if isinstance(value, (int, float)):
        return float(value), {
            "source_url": row["source_url"] or "",
            "snippet": row["snippet"] or "",
            "filing_date": row["filing_date"],
        }
    return UNKNOWN, None


def _suppression_reasons(cfg, *, company_name: str, revenue: float | str) -> list[str]:
    text = (company_name or "").lower()
    reasons: list[str] = []
    if cfg.sector_suppress_prerevenue and (revenue == UNKNOWN or not isinstance(revenue, (int, float)) or revenue <= 0):
        reasons.append("SUPPRESSION_PREREVENUE")
    if cfg.sector_suppress_financials and any(keyword in text for keyword in FINANCIALS_KEYWORDS):
        reasons.append("SUPPRESSION_FINANCIALS")
    if cfg.sector_suppress_biotech and any(keyword in text for keyword in BIOTECH_KEYWORDS):
        reasons.append("SUPPRESSION_BIOTECH")
    if cfg.sector_suppress_rollups and any(keyword in text for keyword in ROLLUP_KEYWORDS):
        reasons.append("SUPPRESSION_ROLLUPS")
    return sorted(set(reasons))


def select_sector_peers(
    *,
    sector: str,
    as_of_date: str,
    years_back: int = 10,
    limit: int = 25,
    min_peers: int = 25,
    max_peers: int | None = None,
    sic_expand: bool = True,
    sic_family: bool = True,
    include_foreign: bool = True,
    include_otc: bool = False,
    min_annual_filings: int | None = None,
    max_peer_scan: int = 400,
    stop_when_min_reached: bool = False,
    exclude_tickers: list[str] | None = None,
    mktcap_min: float | None = None,
    mktcap_max: float | None = None,
    taxonomy_path=None,
    overrides_path=None,
    mode: str = "hybrid",
) -> dict[str, Any]:
    cfg = get_config()
    mode_norm = str(mode or "hybrid").strip().lower()
    if mode_norm not in PEER_MODES:
        raise ValueError(f"mode must be one of {sorted(PEER_MODES)}")

    min_peers_target = max(1, int(min_peers))
    max_peers_target = max(1, int(max_peers if max_peers is not None else limit))
    if min_peers_target > max_peers_target:
        min_peers_target = max_peers_target

    excluded_set = {str(ticker).strip().upper() for ticker in (exclude_tickers or []) if str(ticker).strip()}
    taxonomy = load_sector_taxonomy(taxonomy_path=taxonomy_path, overrides_path=overrides_path)
    target = sector.strip().lower()
    universe_all = sorted([ticker for ticker, name in taxonomy.items() if name.strip().lower() == target])
    if has_sector_definition(target, cfg=cfg):
        try:
            discovered = discover_sector_universe(target, cfg=cfg)
            dynamic_tickers = sorted(
                {
                    str(row.get("ticker") or "").strip().upper()
                    for row in discovered
                    if str(row.get("ticker") or "").strip()
                }
            )
            if dynamic_tickers:
                universe_all = sorted(set(universe_all) | set(dynamic_tickers))
        except Exception:
            pass
    universe = [ticker for ticker in universe_all if ticker not in excluded_set]
    taxonomy_sparse_threshold = max(1, int(cfg.sector_taxonomy_sparse_threshold))
    taxonomy_mapped_count = len(universe)
    fallback_steps_taken: list[str] = []
    sic_missing = 0
    filings_inferred_misses = 0
    anchor_category = _sector_anchor_category(sector)
    taxonomy_sparse_fallback_triggered = False
    if not universe and mode_norm == "taxonomy":
        return {
            "sector": sector,
            "as_of_date": as_of_date,
            "mode": mode_norm,
            "selected_tickers": [],
            "rows": [],
            "all_rows": [],
            "counts": {"mapped": 0, "selected": 0},
            "peer_selection_summary": {
                "mode": mode_norm,
                "anchor_category": anchor_category,
                "taxonomy_hits": 0,
                "sic_hits": 0,
                "sic_family_hits": 0,
                "sic_missing": 0,
                "filings_inferred_hits": 0,
                "filings_inferred_misses": 0,
                "taxonomy_mapped_count": 0,
                "taxonomy_sparse_threshold": taxonomy_sparse_threshold,
                "taxonomy_sparse_fallback_triggered": False,
                "min_peers_target": min_peers_target,
                "max_peers_target": max_peers_target,
                "final_selected_count": 0,
                "fallback_steps_taken": [],
                "missing_category": 0,
                "dropped_by_capband": 0,
                "dropped_by_suppression": 0,
                "cik_missing": 0,
                "otc_excluded": 0,
                "foreign_excluded": 0,
                "budget_skipped": 0,
                "preflight_checked": 0,
                "preflight_eligible": 0,
                "preflight_ineligible": 0,
                "seed_fallback_hits": 0,
                "stage_contribution_counts": {key: 0 for key in STAGE_ORDER_HYBRID},
                "max_peer_scan": int(max_peer_scan),
                "peer_scan_count": 0,
                "scan_exhausted": False,
                "dossierable_target_reached": False,
                "expansion_stopped_at_target": False,
            },
        }

    cap_min = float(mktcap_min if mktcap_min is not None else cfg.discovery_market_cap_min)
    cap_max = float(mktcap_max if mktcap_max is not None else cfg.discovery_market_cap_max)
    provider = get_default_provider(cfg)
    run_id = f"sector_peers_{as_of_date.replace('-', '')}"

    selected_by_source: dict[str, list[dict[str, Any]]] = {
        "taxonomy": [],
        "sic": [],
        "sic_family": [],
        "filings": [],
        "seed_fallback": [],
    }
    selected_ticker_set: set[str] = set()
    evaluated_ticker_set: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    dropped_by_capband = 0
    dropped_by_suppression = 0
    cik_missing = 0
    otc_excluded = 0
    foreign_excluded = 0
    budget_skipped = 0
    preflight_checked = 0
    preflight_eligible = 0
    preflight_ineligible = 0
    peer_scan_count = 0
    scan_exhausted = False
    expansion_stopped_at_target = False
    anchor_sics: set[int] = set()
    anchor_sic_families: set[int] = set()
    seed_universe: list[str] = []

    with get_db() as conn:
        _ensure_sector_inference_table(conn)
        cik_by_ticker = _ticker_cik_map(conn)
        if cik_by_ticker:
            seed_universe = sorted(cik_by_ticker.keys())
        else:
            seed_universe = sorted(set(universe))

        sec_client = SecClient()
        sec_cache: dict[str, dict[str, Any]] = {}
        sic_cache: dict[str, int | None] = {}
        preflight_cache: dict[str, dict[str, Any]] = {}
        inferred_by_ticker: dict[str, tuple[str | None, str]] | None = None
        preflight_forms = _peer_preflight_annual_forms(include_foreign=include_foreign)
        preflight_min_required = max(
            1,
            int(min_annual_filings if min_annual_filings is not None else (3 if int(years_back) >= 10 else 1)),
        )
        sic_network_budget = max(64, int(max_peers_target) * 8)
        sic_network_calls = 0
        peer_scan_cap = max(1, int(max_peer_scan))
        preflight_budget_exhausted = False

        def _target_reached() -> bool:
            return len(selected_ticker_set) >= min_peers_target

        def _resolve_sic_cached(ticker: str, *, allow_network: bool = True) -> int | None:
            nonlocal sic_network_calls
            ticker_norm = str(ticker).upper()
            if ticker_norm in sic_cache:
                return sic_cache[ticker_norm]
            cik = str(cik_by_ticker.get(ticker_norm, "")).strip()
            had_cached_submissions = bool(cik and cik in sec_cache)
            allow_sec_network = bool(allow_network and (had_cached_submissions or sic_network_calls < sic_network_budget))
            sic_value, _ = _resolve_ticker_sic(
                conn,
                ticker=ticker_norm,
                cik_by_ticker=cik_by_ticker,
                as_of_date=as_of_date,
                sec_client=sec_client,
                sec_cache=sec_cache,
                allow_sec_network=allow_sec_network,
            )
            if allow_sec_network and cik and not had_cached_submissions and cik in sec_cache:
                sic_network_calls += 1
            sic_cache[ticker_norm] = sic_value
            return sic_value

        def _preflight_candidate(ticker: str) -> dict[str, Any]:
            nonlocal preflight_budget_exhausted
            ticker_norm = str(ticker or "").strip().upper()
            cached = preflight_cache.get(ticker_norm)
            if cached is not None:
                return cached
            if preflight_budget_exhausted:
                payload = {
                    "ticker": ticker_norm,
                    "cik": str(cik_by_ticker.get(ticker_norm) or "") or None,
                    "query": {
                        "as_of_date": as_of_date,
                        "years_back": int(years_back),
                        "forms": list(preflight_forms),
                        "include_amendments": True,
                        "include_foreign": bool(include_foreign),
                    },
                    "cik_present": bool(str(cik_by_ticker.get(ticker_norm) or "").strip()),
                    "annual_forms_found": {},
                    "annual_filing_count_window": 0,
                    "min_annual_filings_required": int(preflight_min_required),
                    "eligible": False,
                    "skip_reason": "BUDGET_EXCEEDED",
                }
                preflight_cache[ticker_norm] = payload
                return payload
            try:
                payload = preflight_annual_eligibility(
                    ticker=ticker_norm,
                    as_of_date=as_of_date,
                    years_back=int(years_back),
                    include_amendments=True,
                    include_foreign=bool(include_foreign),
                    min_annual_filings=preflight_min_required,
                    forms_override=preflight_forms,
                    cik_hint=str(cik_by_ticker.get(ticker_norm) or ""),
                )
            except DomainBudgetExceeded:
                preflight_budget_exhausted = True
                payload = {
                    "ticker": ticker_norm,
                    "cik": str(cik_by_ticker.get(ticker_norm) or "") or None,
                    "query": {
                        "as_of_date": as_of_date,
                        "years_back": int(years_back),
                        "forms": list(preflight_forms),
                        "include_amendments": True,
                        "include_foreign": bool(include_foreign),
                    },
                    "cik_present": bool(str(cik_by_ticker.get(ticker_norm) or "").strip()),
                    "annual_forms_found": {},
                    "annual_filing_count_window": 0,
                    "min_annual_filings_required": int(preflight_min_required),
                    "eligible": False,
                    "skip_reason": "BUDGET_EXCEEDED",
                }
            except Exception as exc:  # noqa: BLE001
                if "Domain budget exceeded" in str(exc):
                    preflight_budget_exhausted = True
                    payload = {
                        "ticker": ticker_norm,
                        "cik": str(cik_by_ticker.get(ticker_norm) or "") or None,
                        "query": {
                            "as_of_date": as_of_date,
                            "years_back": int(years_back),
                            "forms": list(preflight_forms),
                            "include_amendments": True,
                            "include_foreign": bool(include_foreign),
                        },
                        "cik_present": bool(str(cik_by_ticker.get(ticker_norm) or "").strip()),
                        "annual_forms_found": {},
                        "annual_filing_count_window": 0,
                        "min_annual_filings_required": int(preflight_min_required),
                        "eligible": False,
                        "skip_reason": "BUDGET_EXCEEDED",
                    }
                else:
                    raise
            if (
                not include_foreign
                and not bool(payload.get("eligible"))
                and str(payload.get("skip_reason") or "").upper()
                in {"NO_ANNUAL_FILING_IN_WINDOW", "INSUFFICIENT_ANNUAL_FILINGS_IN_WINDOW"}
                and not preflight_budget_exhausted
            ):
                try:
                    foreign_probe = preflight_annual_eligibility(
                        ticker=ticker_norm,
                        as_of_date=as_of_date,
                        years_back=int(years_back),
                        include_amendments=True,
                        include_foreign=True,
                        min_annual_filings=1,
                        forms_override=list(ANNUAL_FORM_TYPES_WITH_AMENDMENTS),
                        cik_hint=str(cik_by_ticker.get(ticker_norm) or ""),
                    )
                except Exception:
                    foreign_probe = {}
                foreign_forms = foreign_probe.get("annual_forms_found") or {}
                foreign_count = int(foreign_forms.get("20-F", 0)) + int(foreign_forms.get("20-F/A", 0)) + int(
                    foreign_forms.get("40-F", 0)
                )
                if foreign_count > 0:
                    payload = dict(payload)
                    payload["skip_reason"] = "FOREIGN_EXCLUDED"
                    payload["foreign_annual_forms_found"] = {
                        key: int(value) for key, value in sorted(foreign_forms.items()) if int(value) > 0
                    }
            preflight_cache[ticker_norm] = payload
            return payload

        def _add_candidates(
            *,
            candidates: list[str],
            source: str,
            inferred_map: dict[str, tuple[str | None, str]] | None = None,
        ) -> tuple[int, bool]:
            nonlocal dropped_by_capband, dropped_by_suppression
            nonlocal cik_missing, otc_excluded, foreign_excluded, budget_skipped
            nonlocal preflight_checked, preflight_eligible, preflight_ineligible
            nonlocal peer_scan_count, scan_exhausted, expansion_stopped_at_target
            added = 0
            for ticker in sorted({str(value).strip().upper() for value in candidates if str(value).strip()}):
                if len(selected_ticker_set) >= max_peers_target:
                    return added, True
                if stop_when_min_reached and _target_reached():
                    expansion_stopped_at_target = True
                    return added, True
                if scan_exhausted:
                    return added, True
                if ticker in excluded_set or ticker in selected_ticker_set or ticker in evaluated_ticker_set:
                    continue
                if peer_scan_count >= peer_scan_cap:
                    scan_exhausted = True
                    return added, True

                peer_scan_count += 1
                evaluated_ticker_set.add(ticker)
                stage_label = STAGE_LABEL_BY_SOURCE.get(source, source)
                cik = str(cik_by_ticker.get(ticker, "")).strip()
                if not cik:
                    cik_missing += 1
                    all_rows.append(
                        {
                            "ticker": ticker,
                            "peer_source": source,
                            "stage_source": stage_label,
                            "cik": None,
                            "cik_present": False,
                            "annual_forms_found": {},
                            "eligible": False,
                            "sector": taxonomy.get(ticker),
                            "selected": False,
                            "excluded_reason": "MISSING_CIK",
                            "suppression_reasons": [],
                        }
                    )
                    continue
                if not include_otc and _is_likely_otc_ticker(ticker):
                    otc_excluded += 1
                    all_rows.append(
                        {
                            "ticker": ticker,
                            "peer_source": source,
                            "stage_source": stage_label,
                            "cik": cik,
                            "cik_present": True,
                            "annual_forms_found": {},
                            "eligible": False,
                            "sector": taxonomy.get(ticker),
                            "selected": False,
                            "excluded_reason": "OTC_EXCLUDED",
                            "suppression_reasons": [],
                        }
                    )
                    continue

                preflight: dict[str, Any] | None = None
                if cik.isdigit():
                    preflight_checked += 1
                    preflight = _preflight_candidate(ticker)
                    if preflight.get("eligible"):
                        preflight_eligible += 1
                    else:
                        preflight_ineligible += 1
                        skip_reason = str(preflight.get("skip_reason") or "INELIGIBLE").upper()
                        if skip_reason == "BUDGET_EXCEEDED":
                            budget_skipped += 1
                        if skip_reason == "FOREIGN_EXCLUDED":
                            foreign_excluded += 1
                        all_rows.append(
                            {
                                "ticker": ticker,
                                "peer_source": source,
                                "stage_source": stage_label,
                                "cik": cik,
                                "cik_present": True,
                                "annual_forms_found": preflight.get("annual_forms_found") or {},
                                "eligible": False,
                                "sector": taxonomy.get(ticker),
                                "selected": False,
                                "excluded_reason": f"PRECHECK_{skip_reason}",
                                "suppression_reasons": [],
                                "preflight": preflight,
                            }
                        )
                        continue

                inferred_category = None
                inferred_source = None
                if inferred_map is not None:
                    inferred_category, inferred_source = inferred_map.get(ticker, (None, "unknown"))
                row = _candidate_row(
                    conn,
                    ticker=ticker,
                    source=source,
                    sector_value=taxonomy.get(ticker),
                    inferred_category=inferred_category,
                    inferred_source=inferred_source,
                    as_of_date=as_of_date,
                    cap_min=cap_min,
                    cap_max=cap_max,
                    provider=provider,
                    run_id=run_id,
                    cfg=cfg,
                )
                row["stage_source"] = stage_label
                row["cik"] = row.get("cik") or cik
                row["cik_present"] = bool(row.get("cik"))
                row["annual_forms_found"] = (preflight or {}).get("annual_forms_found") or {}
                row["eligible"] = bool((preflight or {}).get("eligible", True))
                if preflight is not None:
                    row["preflight"] = preflight
                all_rows.append(row)
                if row["selected"]:
                    selected_ticker_set.add(ticker)
                    selected_by_source[source].append(row)
                    added += 1
                    if stop_when_min_reached and _target_reached():
                        expansion_stopped_at_target = True
                        return added, True
                elif row.get("suppression_reasons"):
                    dropped_by_suppression += 1
                elif row.get("excluded_reason") == "MARKET_CAP_OUT_OF_BAND":
                    dropped_by_capband += 1
            return added, False

        def _run_stage(stage_key: str) -> bool:
            nonlocal inferred_by_ticker, filings_inferred_misses, anchor_category
            if len(selected_ticker_set) >= max_peers_target:
                return True
            if stop_when_min_reached and _target_reached():
                return True
            if scan_exhausted:
                return True

            source = STAGE_SOURCE_BY_KEY.get(stage_key)
            if source is None:
                return False

            if stage_key == "taxonomy":
                candidates = list(universe)
                _added, halted = _add_candidates(candidates=candidates, source=source)
                return halted

            if stage_key == "sic_expand":
                fallback_steps_taken.append("sic_expand")
                if not sic_expand:
                    return False
                if not anchor_sics:
                    fallback_steps_taken.append("sic_expand_no_anchor_sic")
                    return False
                candidates = []
                for ticker in seed_universe:
                    if ticker in excluded_set or ticker in selected_ticker_set:
                        continue
                    sic_value = _resolve_sic_cached(ticker)
                    if sic_value is None:
                        continue
                    if int(sic_value) in anchor_sics:
                        candidates.append(ticker)
                _added, halted = _add_candidates(candidates=candidates, source=source)
                return halted

            if stage_key == "sic_family":
                fallback_steps_taken.append("sic_family_expand")
                if not sic_family:
                    return False
                if not anchor_sic_families:
                    fallback_steps_taken.append("sic_family_no_anchor_family")
                    return False
                candidates = []
                for ticker in seed_universe:
                    if ticker in excluded_set or ticker in selected_ticker_set:
                        continue
                    sic_value = _resolve_sic_cached(ticker)
                    if sic_value is None:
                        continue
                    if (int(sic_value) // 100) in anchor_sic_families:
                        candidates.append(ticker)
                _added, halted = _add_candidates(candidates=candidates, source=source)
                return halted

            if stage_key == "filings_inferred":
                fallback_steps_taken.append("filings_inferred")
                if inferred_by_ticker is None:
                    inferred_by_ticker = {}
                    for ticker in seed_universe:
                        if ticker in excluded_set:
                            continue
                        inferred_by_ticker[ticker] = _infer_filings_category(
                            conn,
                            ticker=ticker,
                            as_of_date=as_of_date,
                            sec_client=sec_client,
                            sec_cache=sec_cache,
                        )
                    filings_inferred_misses = len(
                        [ticker for ticker in inferred_by_ticker if not inferred_by_ticker[ticker][0]]
                    )

                if anchor_category is None and universe:
                    category_counts: dict[str, int] = {}
                    for ticker in universe:
                        category = inferred_by_ticker.get(ticker, (None, ""))[0]
                        if not category:
                            continue
                        category_counts[category] = category_counts.get(category, 0) + 1
                    if category_counts:
                        anchor_category = sorted(
                            category_counts.items(),
                            key=lambda item: (
                                -item[1],
                                CATEGORY_PRIORITY.index(item[0]) if item[0] in CATEGORY_PRIORITY else len(CATEGORY_PRIORITY),
                                item[0],
                            ),
                        )[0][0]
                if anchor_category is None and anchor_sics:
                    anchor_category = _category_from_sic(sorted(anchor_sics)[0], "")

                candidates = [
                    ticker
                    for ticker in sorted((inferred_by_ticker or {}).keys())
                    if (inferred_by_ticker or {}).get(ticker, (None, ""))[0] == anchor_category
                ]
                _added, halted = _add_candidates(candidates=candidates, source=source, inferred_map=inferred_by_ticker)
                return halted

            if stage_key == "seed_fallback":
                fallback_steps_taken.append("seed_fallback")
                candidates = [
                    ticker
                    for ticker in _seed_fallback_tickers(taxonomy=taxonomy)
                    if ticker not in excluded_set and ticker not in selected_ticker_set
                ]
                _added, halted = _add_candidates(candidates=candidates, source=source)
                return halted

            return False

        stage_plan: list[str]
        if mode_norm == "taxonomy":
            stage_plan = ["taxonomy"]
        elif mode_norm == "filings":
            stage_plan = ["filings_inferred", "seed_fallback"]
        else:
            stage_plan = list(STAGE_ORDER_HYBRID)

        needs_anchor_sic = (
            ("sic_expand" in stage_plan and sic_expand)
            or ("sic_family" in stage_plan and sic_family)
            or ("filings_inferred" in stage_plan and anchor_category is None)
        )
        if needs_anchor_sic:
            for anchor in universe:
                sic_value = _resolve_sic_cached(anchor)
                if sic_value is None:
                    sic_missing += 1
                    continue
                anchor_sics.add(int(sic_value))
                anchor_sic_families.add(int(sic_value) // 100)

        for stage_key in stage_plan:
            halted = _run_stage(stage_key)
            if halted:
                break

        taxonomy_selected_count = len(selected_by_source["taxonomy"])
        taxonomy_sparse_fallback_triggered = bool(
            mode_norm == "hybrid" and taxonomy_selected_count < taxonomy_sparse_threshold
        )

    for source, rows in list(selected_by_source.items()):
        selected_by_source[source] = _sort_selected_rows(rows)

    if mode_norm == "taxonomy":
        selection_order = ["taxonomy"]
    elif mode_norm == "filings":
        selection_order = ["filings", "seed_fallback"]
    else:
        selection_order = [STAGE_SOURCE_BY_KEY[key] for key in STAGE_ORDER_HYBRID]

    selected: list[dict[str, Any]] = []
    for source in selection_order:
        for row in selected_by_source[source]:
            if len(selected) >= max_peers_target:
                break
            selected.append(row)
        if len(selected) >= max_peers_target:
            break

    taxonomy_hits = len([row for row in selected if row.get("peer_source") == "taxonomy"])
    sic_hits = len([row for row in selected if row.get("peer_source") == "sic"])
    sic_family_hits = len([row for row in selected if row.get("peer_source") == "sic_family"])
    filings_hits = len([row for row in selected if row.get("peer_source") == "filings"])
    seed_fallback_hits = len([row for row in selected if row.get("peer_source") == "seed_fallback"])
    stage_contribution_counts = {key: 0 for key in STAGE_ORDER_HYBRID}
    for row in selected:
        stage_key = str(row.get("stage_source") or "")
        if stage_key in stage_contribution_counts:
            stage_contribution_counts[stage_key] += 1
    fallback_steps_ordered = list(dict.fromkeys(fallback_steps_taken))
    selection_summary = {
        "mode": mode_norm,
        "anchor_category": anchor_category,
        "years_back": int(years_back),
        "include_foreign": bool(include_foreign),
        "include_otc": bool(include_otc),
        "annual_forms_considered": _peer_preflight_annual_forms(include_foreign=include_foreign),
        "min_annual_filings_required": int(preflight_min_required),
        "taxonomy_hits": taxonomy_hits,
        "sic_hits": sic_hits,
        "sic_family_hits": sic_family_hits,
        "sic_missing": int(sic_missing),
        "filings_inferred_hits": filings_hits,
        "filings_inferred_misses": int(filings_inferred_misses),
        "taxonomy_mapped_count": int(taxonomy_mapped_count),
        "taxonomy_sparse_threshold": int(taxonomy_sparse_threshold),
        "taxonomy_sparse_fallback_triggered": bool(taxonomy_sparse_fallback_triggered),
        "min_peers_target": int(min_peers_target),
        "max_peers_target": int(max_peers_target),
        "final_selected_count": len(selected),
        "fallback_steps_taken": fallback_steps_ordered,
        "missing_category": int(filings_inferred_misses),
        "dropped_by_capband": int(dropped_by_capband),
        "dropped_by_suppression": int(dropped_by_suppression),
        "cik_missing": int(cik_missing),
        "otc_excluded": int(otc_excluded),
        "foreign_excluded": int(foreign_excluded),
        "budget_skipped": int(budget_skipped),
        "preflight_checked": int(preflight_checked),
        "preflight_eligible": int(preflight_eligible),
        "preflight_ineligible": int(preflight_ineligible),
        "seed_fallback_hits": int(seed_fallback_hits),
        "stage_contribution_counts": stage_contribution_counts,
        "max_peer_scan": int(max_peer_scan),
        "peer_scan_count": int(peer_scan_count),
        "scan_exhausted": bool(scan_exhausted),
        "dossierable_target_reached": bool(len(selected) >= min_peers_target),
        "expansion_stopped_at_target": bool(expansion_stopped_at_target),
        "sic_lookup_budget": int(sic_network_budget),
        "sic_lookup_used": int(sic_network_calls),
    }

    return {
        "sector": sector,
        "as_of_date": as_of_date,
        "mode": mode_norm,
        "anchor_category": anchor_category,
        "selected_tickers": [row["ticker"] for row in selected],
        "rows": selected,
        "all_rows": all_rows,
        "excluded_tickers": sorted(excluded_set),
        "counts": {
            "mapped": len(universe),
            "selected": len(selected),
            "suppressed": len([row for row in all_rows if row.get("suppression_reasons")]),
            "out_of_band": len([row for row in all_rows if row.get("excluded_reason") == "MARKET_CAP_OUT_OF_BAND"]),
            "missing_cik": int(cik_missing),
            "otc_excluded": int(otc_excluded),
            "preflight_ineligible": int(preflight_ineligible),
            "market_cap_unknown_selected": len([row for row in selected if row.get("market_cap") == UNKNOWN]),
        },
        "peer_selection_summary": selection_summary,
    }
