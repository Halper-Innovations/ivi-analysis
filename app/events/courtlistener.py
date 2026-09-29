"""CourtListener/RECAP docket search for the litigation-docket gate.

A lawsuit filed against an actionable/at-target name reaches PACER (and
CourtListener's free REST API) within days — months before it surfaces in a
10-Q legal-proceedings section. The scan searches recent dockets by company
name, keeps only severity-relevant nature-of-suit codes (securities, fraud,
antitrust by default), and requires a conservative full-token name match so
generic company names don't spam the triage queue. Hits open
litigation_docket queue-protection events (anchor "CL-<docket_id>" — synthetic,
there is no EDGAR accession) that block DEPLOY_READY presentation until
`ivi events dispose`.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, timedelta

from app.config import get_config
from app.events.detectors import normalize_company_name

SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"

# Names whose normalized form is shorter than this are too generic to search
# by free-text party match without flooding the triage queue.
MIN_NORMALIZED_NAME_LEN = 4


@dataclass
class DocketHit:
    docket_id: str
    case_name: str
    court: str
    date_filed: str
    nature_of_suit: str
    nos_code: int | None
    url: str


def nos_code_from_text(suit_nature: str | None) -> int | None:
    """Nature-of-suit code from the API's free-text label ('850 Securities...')."""
    if not suit_nature:
        return None
    match = re.match(r"\s*(\d{2,4})", str(suit_nature))
    return int(match.group(1)) if match else None


def case_name_matches(company_name: str, case_name: str) -> bool:
    """Conservative match: the normalized company name must appear as a
    contiguous token sequence in the normalized case caption."""
    company = normalize_company_name(company_name)
    caption = normalize_company_name(case_name)
    if len(company) < MIN_NORMALIZED_NAME_LEN or not caption:
        return False
    company_tokens = company.split()
    caption_tokens = caption.split()
    span = len(company_tokens)
    for start in range(len(caption_tokens) - span + 1):
        if caption_tokens[start:start + span] == company_tokens:
            return True
    return False


class CourtListenerSearcher:
    """Thin search client. Filtering (NOS whitelist, name match) lives in the
    detector so it is testable against fakes."""

    def __init__(self, cfg=None, http=None) -> None:
        from app.util.http import HttpClient

        self.cfg = cfg or get_config()
        self.http = http or HttpClient(self.cfg)
        token = self.cfg.courtlistener_api_token
        if token:
            self.http.session.headers.update({"Authorization": f"Token {token}"})

    def search_recent_dockets(
        self, company_name: str, *, filed_after: str
    ) -> list[DocketHit] | None:
        params = {
            "type": "r",
            "q": f'"{company_name}"',
            "filed_after": filed_after,
            "order_by": "dateFiled desc",
        }
        try:
            payload = self.http.get_json(
                SEARCH_URL, params=params, use_cache=True, cache_ttl_seconds=6 * 3600
            )
        except Exception:
            return None
        hits: list[DocketHit] = []
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            docket_id = result.get("docket_id") or result.get("docketId") or result.get("id")
            if docket_id is None:
                continue
            suit_nature = str(result.get("suitNature") or "")
            absolute_url = str(result.get("absolute_url") or "")
            hits.append(
                DocketHit(
                    docket_id=str(docket_id),
                    case_name=str(result.get("caseName") or ""),
                    court=str(result.get("court") or result.get("court_id") or ""),
                    date_filed=str(result.get("dateFiled") or "")[:10],
                    nature_of_suit=suit_nature,
                    nos_code=nos_code_from_text(suit_nature),
                    url=f"https://www.courtlistener.com{absolute_url}" if absolute_url else "",
                )
            )
        return hits


def litigation_targets(conn: sqlite3.Connection) -> list[dict[str, str]]:
    """[{cik, ticker, company_name}] for at-target + ACTIONABLE watchlist names.

    Bounded scope keeps API volume to dozens of queries per day. CIK and
    company name both come from sec_registrants (kept current by the weekly
    universe sync); names without a registrant row are skipped.
    """
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM watchlist "
            "WHERE status != 'REMOVED' "
            "AND COALESCE(conviction_grade, '') != 'AVOID' "
            "AND (status IN ('DEPLOY_READY', 'BUY_CONFIRMED') OR conviction_grade = 'ACTIONABLE')"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    wanted = sorted({str(row["ticker"]).upper() for row in rows})
    targets: list[dict[str, str]] = []
    for ticker in wanted:
        try:
            reg = conn.execute(
                "SELECT cik, name FROM sec_registrants WHERE primary_ticker = ?",
                (ticker,),
            ).fetchone()
        except sqlite3.OperationalError:
            return []
        if reg is None or not reg["name"]:
            continue
        targets.append({
            "cik": str(reg["cik"]).zfill(10),
            "ticker": ticker,
            "company_name": str(reg["name"]),
        })
    return targets


def run_docket_scan(
    scan_date: str | None = None,
    *,
    lookback_days: int | None = None,
    tickers: list[str] | None = None,
    dry_run: bool = False,
    searcher=None,
) -> dict:
    """Entry point behind `ivi events dockets`; returns a JSON-safe summary."""
    from app.db import get_db, init_db
    from app.events.detectors import detect_litigation_dockets
    from app.events.flags import sync_event_pending_flags

    cfg = get_config()
    if not cfg.events_dockets_enabled:
        return {"status": "DISABLED", "detail": "set VOE_EVENTS_DOCKETS_ENABLED=true"}
    if searcher is None:
        if not cfg.courtlistener_api_token:
            return {"status": "NO_TOKEN", "detail": "set VOE_COURTLISTENER_API_TOKEN"}
        searcher = CourtListenerSearcher(cfg)
    effective_date = scan_date or date.today().isoformat()
    lookback = lookback_days or cfg.events_docket_lookback_days
    init_db()
    with get_db() as conn:
        targets = litigation_targets(conn)
        if tickers:
            wanted = {t.upper() for t in tickers}
            targets = [t for t in targets if t["ticker"] in wanted]
        if dry_run:
            filed_after = (
                date.fromisoformat(effective_date) - timedelta(days=lookback)
            ).isoformat()
            hits_out = []
            for target in targets:
                hits = searcher.search_recent_dockets(
                    target["company_name"], filed_after=filed_after
                )
                for hit in hits or []:
                    hits_out.append({
                        "ticker": target["ticker"],
                        "would_flag": (
                            hit.nos_code in set(cfg.events_docket_nos_codes)
                            and case_name_matches(target["company_name"], hit.case_name)
                        ),
                        **asdict(hit),
                    })
            return {
                "status": "DRY_RUN",
                "scan_date": effective_date,
                "targets": len(targets),
                "hits": hits_out,
            }
        counters = detect_litigation_dockets(
            conn, targets, scan_date=effective_date, lookback_days=lookback,
            nos_codes=set(cfg.events_docket_nos_codes), source_mode="daily",
            searcher=searcher,
        )
        sync = sync_event_pending_flags(conn)
        return {
            "status": "OK",
            "scan_date": effective_date,
            "targets": len(targets),
            "events_created": counters.events_created,
            "counts": counters.counts,
            "flag_sync": sync,
        }
