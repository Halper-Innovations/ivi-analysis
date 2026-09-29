"""Auto-classify tickers by sector using SIC codes from EDGAR submissions."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    ticker: str
    cik: str | None
    sic: int | None
    sector: str | None
    score: float
    derived_from: list[str]
    status: str  # "classified" | "excluded_non_operating" | "no_cik" | "no_sic" | "no_sector_match" | "fetch_error"


@dataclass
class ClassificationSummary:
    total: int
    classified: int
    excluded_non_operating: int
    no_cik: int
    no_sic: int
    no_sector_match: int
    fetch_error: int
    skipped_existing: int
    sector_counts: dict[str, int]
    unclassified_tickers: list[str]


def _extract_sic(submissions: dict) -> int | None:
    """Extract SIC code from EDGAR submissions dict."""
    for key in ("sic", "sicCode"):
        value = submissions.get(key)
        if value is None:
            continue
        token = str(value).strip()
        if token.isdigit():
            return int(token)
    return None


def _extract_company_name(submissions: dict[str, Any] | None) -> str:
    if not isinstance(submissions, dict):
        return ""
    for key in ("name", "entityName", "title"):
        value = submissions.get(key)
        if value is None:
            continue
        token = str(value).strip()
        if token:
            return token
    return ""


_NON_OPERATING_NAME_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:special purpose acquisition|blank check|acquisition "
            r"(?:corp(?:oration)?|company|holdings?|holdco|capital|partners?))\b",
            re.IGNORECASE,
        ),
        "name:spac_or_acquisition_shell",
    ),
    (re.compile(r"\bwarrants?\b", re.IGNORECASE), "name:warrant"),
    (re.compile(r"\brights?\b", re.IGNORECASE), "name:right"),
    (re.compile(r"\bunits?\b", re.IGNORECASE), "name:unit"),
    (re.compile(r"\bETF\b", re.IGNORECASE), "name:etf"),
    (re.compile(r"\bETN\b", re.IGNORECASE), "name:etn"),
    (re.compile(r"\bETP\b", re.IGNORECASE), "name:etp"),
    (re.compile(r"\bexchange[- ]traded fund\b", re.IGNORECASE), "name:etf"),
    (re.compile(r"\bexchange[- ]traded note\b", re.IGNORECASE), "name:etn"),
    (re.compile(r"\bclosed[- ]end fund\b", re.IGNORECASE), "name:closed_end_fund"),
    (re.compile(r"\bmutual fund\b", re.IGNORECASE), "name:mutual_fund"),
    (re.compile(r"\btotal return\b", re.IGNORECASE), "name:total_return_vehicle"),
    (re.compile(r"\bfund\b", re.IGNORECASE), "name:fund"),
    (re.compile(r"\btrust\b", re.IGNORECASE), "name:trust"),
    (re.compile(r"\bindex\b", re.IGNORECASE), "name:index_product"),
    (re.compile(r"\bproshares\b", re.IGNORECASE), "name:etp_sponsor"),
    (re.compile(r"\bdirexion\b", re.IGNORECASE), "name:etp_sponsor"),
    (re.compile(r"\bglobal x\b", re.IGNORECASE), "name:etp_sponsor"),
    (re.compile(r"\bishares\b", re.IGNORECASE), "name:etp_sponsor"),
    (re.compile(r"\bspdr\b", re.IGNORECASE), "name:etp_sponsor"),
    (re.compile(r"\bvaneck\b", re.IGNORECASE), "name:etp_sponsor"),
    (re.compile(r"\bgrayscale\b", re.IGNORECASE), "name:etp_sponsor"),
)


_NO_SIC_CLASSIFICATION_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\bbdc\b", re.IGNORECASE),
        "capital_markets",
        "name_pattern:bdc",
    ),
    (
        re.compile(r"\bspecialty finance\b", re.IGNORECASE),
        "capital_markets",
        "name_pattern:specialty_finance",
    ),
    (
        re.compile(r"\bsecured lending\b", re.IGNORECASE),
        "capital_markets",
        "name_pattern:secured_lending",
    ),
    (
        re.compile(r"\bgrowth finance\b", re.IGNORECASE),
        "capital_markets",
        "name_pattern:growth_finance",
    ),
    (
        re.compile(r"\b(?:finance|financial)\s+corp(?:oration)?\b", re.IGNORECASE),
        "capital_markets",
        "name_pattern:finance_corp",
    ),
    (
        re.compile(r"\binvestment\s+corp(?:oration)?\b", re.IGNORECASE),
        "capital_markets",
        "name_pattern:investment_corp",
    ),
    (
        re.compile(r"\bcapital\s+corp(?:oration)?\b", re.IGNORECASE),
        "capital_markets",
        "name_pattern:capital_corp",
    ),
    (
        re.compile(
            r"\bfinance,?\s+(?:inc\.?|corp(?:oration)?|co(?:mpany)?|ltd\.?|limited)\b",
            re.IGNORECASE,
        ),
        "capital_markets",
        "name_pattern:finance_entity",
    ),
    (
        re.compile(r"\blending\b", re.IGNORECASE),
        "capital_markets",
        "name_pattern:lending",
    ),
    (
        re.compile(r"\b(?:theatre|theater|cinema)\b", re.IGNORECASE),
        "media_entertainment",
        "name_pattern:theater_cinema",
    ),
    (
        re.compile(r"\b(?:mining|royalty)\b|\bgold\s+corp\b", re.IGNORECASE),
        "metals_mining",
        "name_pattern:mining_or_royalty",
    ),
)


_NO_SECTOR_MATCH_CLASSIFICATION_PATTERNS: tuple[
    tuple[frozenset[int], re.Pattern[str], str, str], ...
] = (
    (
        frozenset({0, 6795}),
        re.compile(r"\b(?:mining|royalty)\b|\bgold\s+corp\b", re.IGNORECASE),
        "metals_mining",
        "name_pattern:mining_or_royalty",
    ),
    (
        frozenset({6794}),
        re.compile(r"\bdigital\b|\bai\b", re.IGNORECASE),
        "internet_services",
        "name_pattern:digital_or_ai",
    ),
    (
        frozenset({2670, 2621, 2631, 2650}),
        re.compile(r"\b(?:paper|packaging|label|labels)\b", re.IGNORECASE),
        "consumer_staples",
        "name_pattern:paper_packaging",
    ),
    (
        frozenset({4700}),
        re.compile(r"\bbooking\b", re.IGNORECASE),
        "internet_services",
        "name_pattern:booking",
    ),
    (
        frozenset({4700}),
        re.compile(r"\blogistics\b", re.IGNORECASE),
        "transportation_logistics",
        "name_pattern:logistics",
    ),
    (
        frozenset({5099, 5031, 3231, 3241, 3272}),
        re.compile(
            r"\b(?:glass|cement|concrete|building|lumber|packaging)\b|core\s*&\s*main",
            re.IGNORECASE,
        ),
        "building_products",
        "name_pattern:building_products",
    ),
    (
        frozenset({5122, 5140, 5141}),
        re.compile(r"\b(?:foods?|warehouse|chef)\b", re.IGNORECASE),
        "restaurants_food_service",
        "name_pattern:food_distribution",
    ),
    (
        frozenset({7320}),
        re.compile(r"\b(?:credit|risk|union)\b", re.IGNORECASE),
        "business_services",
        "name_pattern:credit_risk",
    ),
    (
        frozenset({7500}),
        re.compile(r"\b(?:auto|collision|body)\b", re.IGNORECASE),
        "automotive",
        "name_pattern:auto_service",
    ),
    (
        frozenset({8734}),
        re.compile(r"\b(?:bio|biolabs|therapeutics?)\b", re.IGNORECASE),
        "biotech",
        "name_pattern:bio_science",
    ),
    (
        frozenset({8741}),
        re.compile(r"\bhealth\b", re.IGNORECASE),
        "healthcare_services",
        "name_pattern:health",
    ),
)


def _detect_non_operating_symbol(
    *,
    ticker: str,
    company_name: str = "",
    sic: int | None = None,
) -> str | None:
    ticker_norm = str(ticker or "").strip().upper()
    name_norm = str(company_name or "").strip()

    # Preferreds, warrants, and similar share-class suffixes.
    if re.search(r"-(?:P|PA|PB|PC|PD|PE|PF|WS|WT|W|R|U)\b", ticker_norm):
        return "ticker:special_share_class"
    # Common OTC and special-instrument suffixes are not sector-scanable here.
    if len(ticker_norm) == 5 and ticker_norm[-1:] in {"W", "Q", "Z"}:
        return f"ticker:otc_suffix_{ticker_norm[-1:].lower()}"
    # SIC 6770 is dominated by shell / blank-check / investment-company style names.
    if sic == 6770:
        return "sic:6770_non_operating"
    for pattern, reason in _NON_OPERATING_NAME_PATTERNS:
        if pattern.search(name_norm):
            return reason
    return None


def _classify_no_sic_from_name(company_name: str) -> tuple[str, str] | None:
    name_norm = str(company_name or "").strip()
    if not name_norm:
        return None
    for pattern, sector, reason in _NO_SIC_CLASSIFICATION_PATTERNS:
        if pattern.search(name_norm):
            return sector, reason
    return None


def _classify_no_sector_match_from_name(
    company_name: str,
    sic: int | None,
) -> tuple[str, str] | None:
    if sic is None:
        return None
    name_norm = str(company_name or "").strip()
    if not name_norm:
        return None
    for sic_scope, pattern, sector, reason in _NO_SECTOR_MATCH_CLASSIFICATION_PATTERNS:
        if sic in sic_scope and pattern.search(name_norm):
            return sector, reason
    return None


def build_sic_to_sector_index(
    *,
    cfg: Any | None = None,
) -> dict[int, str]:
    """Build a flat SIC-code -> sector lookup.

    When a SIC code falls in multiple sectors' ranges, the sector with
    the smallest total range span (most specific) wins.
    """
    from app.universe.sector_universe import load_sector_sic_config

    sector_config = load_sector_sic_config(cfg=cfg)

    # Compute specificity: total number of SIC codes each sector covers
    sector_spans: dict[str, int] = {}
    for sector_key, info in sector_config.items():
        total = 0
        for start, end in info["sic_ranges"]:
            total += end - start + 1
        sector_spans[sector_key] = total

    # Build flat index, preferring more specific sectors on collision
    index: dict[int, str] = {}
    for sector_key, info in sector_config.items():
        for start, end in info["sic_ranges"]:
            for sic in range(start, end + 1):
                existing = index.get(sic)
                if existing is None:
                    index[sic] = sector_key
                elif sector_spans[sector_key] < sector_spans.get(existing, 999999):
                    index[sic] = sector_key
                # else: keep existing (it's more specific or equal)

    return index


def load_scorecarded_tickers(conn: sqlite3.Connection) -> list[str]:
    """Return tickers whose newest scorecard row is decision-eligible."""
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM valuations WHERE method = 'scorecard' ORDER BY ticker"
    ).fetchall()
    return [
        str(row[0]).upper()
        for row in rows
        if latest_decision_eligible_valuation_row(
            conn,
            ticker=str(row[0]),
            method="scorecard",
        )
        is not None
    ]


def classify_ticker(
    ticker: str,
    *,
    cik_map: dict[str, str],
    sic_index: dict[int, str],
    submissions_loader: Any = None,
) -> ClassificationResult:
    """Classify a single ticker by SIC code.

    submissions_loader: callable(cik) -> dict. If None, uses
    load_company_submissions from sector_universe.
    """
    cik = cik_map.get(ticker.upper())
    non_operating_reason = _detect_non_operating_symbol(ticker=ticker)
    if non_operating_reason is not None:
        return ClassificationResult(
            ticker=ticker,
            cik=cik,
            sic=None,
            sector=None,
            score=0.0,
            derived_from=[non_operating_reason, "method:exclude_non_operating"],
            status="excluded_non_operating",
        )
    if cik is None:
        return ClassificationResult(
            ticker=ticker,
            cik=None,
            sic=None,
            sector=None,
            score=0.0,
            derived_from=["method:no_cik"],
            status="no_cik",
        )

    try:
        if submissions_loader is not None:
            submissions = submissions_loader(cik)
        else:
            from app.universe.sector_universe import load_company_submissions

            submissions = load_company_submissions(cik)
    except Exception as exc:
        logger.warning("classify %s: fetch error for CIK %s: %s", ticker, cik, exc)
        return ClassificationResult(
            ticker=ticker,
            cik=cik,
            sic=None,
            sector=None,
            score=0.0,
            derived_from=[f"cik:{cik}", f"error:{exc}"],
            status="fetch_error",
        )

    if submissions is None:
        return ClassificationResult(
            ticker=ticker,
            cik=cik,
            sic=None,
            sector=None,
            score=0.0,
            derived_from=[f"cik:{cik}", "submissions:null"],
            status="fetch_error",
        )

    sic = _extract_sic(submissions)
    company_name = _extract_company_name(submissions)
    non_operating_reason = _detect_non_operating_symbol(
        ticker=ticker,
        company_name=company_name,
        sic=sic,
    )
    if non_operating_reason is not None:
        derived_from = [f"cik:{cik}"]
        if sic is not None:
            derived_from.append(f"sic:{sic}")
        if company_name:
            derived_from.append(f"name:{company_name}")
        derived_from.extend([non_operating_reason, "method:exclude_non_operating"])
        return ClassificationResult(
            ticker=ticker,
            cik=cik,
            sic=sic,
            sector=None,
            score=0.0,
            derived_from=derived_from,
            status="excluded_non_operating",
        )
    if sic is None:
        fallback = _classify_no_sic_from_name(company_name)
        if fallback is not None:
            sector, reason = fallback
            derived_from = [f"cik:{cik}"]
            if company_name:
                derived_from.append(f"name:{company_name}")
            derived_from.extend([f"sector:{sector}", reason, "method:no_sic_name_fallback"])
            return ClassificationResult(
                ticker=ticker,
                cik=cik,
                sic=None,
                sector=sector,
                score=1.0,
                derived_from=derived_from,
                status="classified",
            )
        return ClassificationResult(
            ticker=ticker,
            cik=cik,
            sic=None,
            sector=None,
            score=0.0,
            derived_from=[
                f"cik:{cik}",
                *([f"name:{company_name}"] if company_name else []),
                "method:no_sic",
            ],
            status="no_sic",
        )

    sector = sic_index.get(sic)
    if sector is None:
        fallback = _classify_no_sector_match_from_name(company_name, sic)
        if fallback is not None:
            sector, reason = fallback
            derived_from = [f"cik:{cik}", f"sic:{sic}"]
            if company_name:
                derived_from.append(f"name:{company_name}")
            derived_from.extend([f"sector:{sector}", reason, "method:no_sector_name_fallback"])
            return ClassificationResult(
                ticker=ticker,
                cik=cik,
                sic=sic,
                sector=sector,
                score=1.0,
                derived_from=derived_from,
                status="classified",
            )
        return ClassificationResult(
            ticker=ticker,
            cik=cik,
            sic=sic,
            sector=None,
            score=0.0,
            derived_from=[f"cik:{cik}", f"sic:{sic}", "method:no_sector_match"],
            status="no_sector_match",
        )

    return ClassificationResult(
        ticker=ticker,
        cik=cik,
        sic=sic,
        sector=sector,
        score=1.0,
        derived_from=[f"cik:{cik}", f"sic:{sic}", f"sector:{sector}", "method:sic_range_match"],
        status="classified",
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_all(
    *,
    db_path: str | Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    as_of_date: str | None = None,
    tickers: list[str] | tuple[str, ...] | None = None,
    submissions_loader: Any = None,
    cfg: Any | None = None,
) -> ClassificationSummary:
    """Classify scorecarded tickers, or an explicit ticker scope, by sector.

    Parameters
    ----------
    db_path : path to engine.db. If None, uses cfg.db_path.
    dry_run : if True, skip DB writes.
    force : if True, reclassify already-classified tickers.
    as_of_date : date string (default: today).
    tickers : optional explicit ticker scope. If omitted, all scorecarded tickers are used.
    submissions_loader : injectable callable(cik) -> dict for testing.
    cfg : AppConfig override.
    """
    if cfg is None:
        from app.config import get_config

        cfg = get_config()

    if db_path is None:
        db_path = cfg.db_path

    effective_date = as_of_date or date.today().isoformat()

    # Build the SIC -> sector index
    sic_index = build_sic_to_sector_index(cfg=cfg)
    logger.info("SIC index built: %d SIC codes mapped to sectors", len(sic_index))

    # Load CIK map
    from app.universe.ticker_cik_map import load_ticker_cik_map

    cik_map = load_ticker_cik_map()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if tickers is None:
            ticker_scope = load_scorecarded_tickers(conn)
            logger.info("Found %d scorecarded tickers", len(ticker_scope))
        else:
            ticker_scope = sorted(
                {
                    str(ticker or "").upper().strip()
                    for ticker in tickers
                    if str(ticker or "").strip()
                }
            )
            logger.info("Classifying explicit ticker scope: %d tickers", len(ticker_scope))

        # Check existing classifications
        existing: set[str] = set()
        if not force:
            rows = conn.execute(
                "SELECT ticker FROM sector_inference WHERE as_of_date = ?",
                (effective_date,),
            ).fetchall()
            existing = {r["ticker"] for r in rows}

        # Classify each ticker
        results: list[ClassificationResult] = []
        skipped = 0
        for ticker in ticker_scope:
            if ticker in existing:
                skipped += 1
                continue
            result = classify_ticker(
                ticker,
                cik_map=cik_map,
                sic_index=sic_index,
                submissions_loader=submissions_loader,
            )
            results.append(result)

        # Write to DB
        if not dry_run:
            for r in results:
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
                        r.ticker,
                        effective_date,
                        r.sector,
                        r.score,
                        json.dumps(r.derived_from),
                        _utc_now_iso(),
                    ),
                )
            conn.commit()
            logger.info("Wrote %d classifications to sector_inference", len(results))

        # Build summary
        sector_counts: dict[str, int] = {}
        excluded = no_cik = no_sic = no_match = fetch_err = classified = 0
        unclassified: list[str] = []
        for r in results:
            if r.status == "classified":
                classified += 1
                sector_counts[r.sector] = sector_counts.get(r.sector, 0) + 1
            elif r.status == "excluded_non_operating":
                excluded += 1
            elif r.status == "no_cik":
                no_cik += 1
                unclassified.append(r.ticker)
            elif r.status == "no_sic":
                no_sic += 1
                unclassified.append(r.ticker)
            elif r.status == "no_sector_match":
                no_match += 1
                unclassified.append(r.ticker)
            elif r.status == "fetch_error":
                fetch_err += 1
                unclassified.append(r.ticker)

        return ClassificationSummary(
            total=len(ticker_scope),
            classified=classified,
            excluded_non_operating=excluded,
            no_cik=no_cik,
            no_sic=no_sic,
            no_sector_match=no_match,
            fetch_error=fetch_err,
            skipped_existing=skipped,
            sector_counts=dict(sorted(sector_counts.items(), key=lambda x: -x[1])),
            unclassified_tickers=sorted(unclassified),
        )
    finally:
        conn.close()
