"""Deterministic structural exclusion gate (auto-quarantine, surfaced not silent).

Catches the GTEC class: names that are structurally untradeable or carry a
mechanical red flag no LLM read should be spent on — a computable but
sub-floor market cap, a sub-$1 price, a live delisting clock, positive
reported net income against negative operating cash flow, or affirmative
going-concern language. Every trigger is deterministic, point-in-time safe
(keyed on FILED dates so the backtest can apply the identical gate), and
produces an exact literal reason string ``QUARANTINE_STRUCTURAL:<code>``.

The gate fires in two places:
  1. Pre-LLM in sweep candidate loading (saves spend) — quarantined names
     never reach the analyst loop; the exclusion lands in the selection
     audit trail.
  2. At watchlist intake — a candidate that still arrives with a verdict is
     written with status QUARANTINE and the reason string, never silently
     dropped.

v1 limitations (documented in the census doc):
  - DELISTING_NOTICE reads 8-K item codes from the cached SEC submissions
    payloads (data/cache/submissions/<cik10>.json). Item-level detection IS
    cheaply available there; what v1 cannot detect is a subsequent CURE
    (regained-compliance) filing, because 8-K document text is not cached.
    A name that cured its deficiency inside the trailing window stays
    quarantined until owner review.
  - GOING_CONCERN is a text match over the latest cached 10-K/10-Q primary
    document. Hypothetical risk-factor phrasing ("could/may/might/would
    raise substantial doubt") is excluded; affirmative disclosure triggers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

from app.config import AppConfig, get_config
from app.autonomous.sector_contract import GateEvaluation, ScreenResult

logger = logging.getLogger(__name__)

QUARANTINE_STRUCTURAL_PREFIX = "QUARANTINE_STRUCTURAL"
# Fail-closed: prefix for names the gate could NOT examine (DB missing or
# erroring). Distinct from quarantine — the name is excluded because the
# platform failed, and the failure must surface rather than read as a pass.
EXCLUDED_ERROR_PREFIX = "EXCLUDED_ERROR"

STRUCTURAL_DELISTING_NOTICE = "DELISTING_NOTICE"
STRUCTURAL_PENNY_FLOOR = "PENNY_FLOOR"
STRUCTURAL_NANO_FLOOR = "NANO_FLOOR"
STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE = "EARNINGS_QUALITY_DIVERGENCE"
STRUCTURAL_GOING_CONCERN = "GOING_CONCERN"
STRUCTURAL_NON_PRIMARY_LISTING = "NON_PRIMARY_LISTING"

# Deterministic evaluation/report order.
STRUCTURAL_CODES = (
    STRUCTURAL_NON_PRIMARY_LISTING,
    STRUCTURAL_DELISTING_NOTICE,
    STRUCTURAL_PENNY_FLOOR,
    STRUCTURAL_NANO_FLOOR,
    STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE,
    STRUCTURAL_GOING_CONCERN,
)

PENNY_FLOOR_PRICE_USD = 1.00
NANO_FLOOR_MARKET_CAP_MM = 25.0
DELISTING_LOOKBACK_DAYS = 365
SubmissionsLoader = Callable[[str], dict[str, Any] | None]
FilingTextLoader = Callable[[str], str | None]
# Ordered (cik, ticker) pairs in SEC registry file order — the first ticker
# for a CIK is the primary (common-stock) listing.
TickerRegistryLoader = Callable[[], list[tuple[str, str]] | None]

_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class StructuralGateResult:
    ticker: str
    as_of_date: str
    triggered_codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)
    # Fail-closed: codes for gate checks that could not run (engine.db
    # missing/locked/unreadable). A degraded result must never be treated as
    # a clean pass — consumers exclude the name as EXCLUDED_ERROR instead of
    # letting it flow past the quarantine gate unexamined.
    degraded_codes: list[str] = field(default_factory=list)
    # Advisory flags (never quarantine): tradeability annotations
    # computed before LLM spend, e.g. MIN_ADV when the 20d dollar-ADV sits
    # below the floor. Flags mark; they do not reject.
    advisory_codes: list[str] = field(default_factory=list)
    contract_id: str | None = None
    gate_evaluations: list[GateEvaluation] = field(default_factory=list)
    screen_result: ScreenResult | None = None

    @property
    def quarantined(self) -> bool:
        return bool(self.triggered_codes)

    @property
    def excluded_error(self) -> bool:
        return bool(self.degraded_codes)

    @property
    def reason_string(self) -> str:
        return ";".join(self.reasons)

    @property
    def degraded_string(self) -> str:
        return ";".join(f"{EXCLUDED_ERROR_PREFIX}:{code}" for code in self.degraded_codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "quarantined": self.quarantined,
            "excluded_error": self.excluded_error,
            "triggered_codes": list(self.triggered_codes),
            "degraded_codes": list(self.degraded_codes),
            "advisory_codes": list(self.advisory_codes),
            "reasons": list(self.reasons),
            "details": dict(self.details),
            "contract_id": self.contract_id,
            "gate_evaluations": [item.to_dict() for item in self.gate_evaluations],
            "screen_result": self.screen_result.to_dict() if self.screen_result else None,
        }


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return None


def _load_cached_submissions(ticker: str, cfg: AppConfig) -> dict[str, Any] | None:
    try:
        from app.universe.ticker_cik_map import load_ticker_cik_map

        cik = load_ticker_cik_map(refresh_if_missing=False).get(str(ticker).upper())
    except Exception:  # noqa: BLE001 - missing map means no detection, not failure
        return None
    if not cik:
        return None
    path = cfg.cache_dir / "submissions" / f"{str(cik).strip().zfill(10)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


_REGISTRY_CACHE: dict[str, list[tuple[str, str]]] = {}


def _default_ticker_registry(cfg: AppConfig) -> list[tuple[str, str]] | None:
    """Ordered (cik, ticker) pairs from the cached SEC ticker registry.

    Prefers company_tickers_exchange.json (data rows preserve SEC order);
    falls back to company_tickers.json. Missing/unparseable caches mean no
    detection — the gate never quarantines on missing data.
    """
    for name in ("company_tickers_exchange.json", "company_tickers.json"):
        path = cfg.cache_dir / name
        key = str(path)
        if key in _REGISTRY_CACHE:
            return _REGISTRY_CACHE[key]
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pairs: list[tuple[str, str]] = []
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            fields = payload.get("fields") or []
            try:
                cik_idx, ticker_idx = fields.index("cik"), fields.index("ticker")
            except ValueError:
                continue
            for row in payload["data"]:
                pairs.append((str(row[cik_idx]), str(row[ticker_idx]).upper()))
        elif isinstance(payload, dict):
            for entry in payload.values():
                if isinstance(entry, dict) and "cik_str" in entry and "ticker" in entry:
                    pairs.append((str(entry["cik_str"]), str(entry["ticker"]).upper()))
        if pairs:
            _REGISTRY_CACHE[key] = pairs
            return pairs
    return None


def _symbol_root(symbol: str) -> str:
    """Root symbol with any '.X' / '-X' share-class suffix stripped."""
    return re.split(r"[.\-]", symbol, maxsplit=1)[0]


def _is_share_class_variant(primary: str, candidate: str) -> bool:
    """True when two symbols look like share classes of the same stock.

    Same root (BRK-A / BRK-B, LEN / LEN.B) or one normalized symbol is a
    prefix of the other (GOOG / GOOGL). Unrelated symbols on the same CIK
    (AMG vs MGRB — exchange-traded notes) are NOT variants.
    """
    norm_p = re.sub(r"[^A-Z0-9]", "", primary.upper())
    norm_c = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    if _symbol_root(primary.upper()) == _symbol_root(candidate.upper()):
        return True
    return norm_p.startswith(norm_c) or norm_c.startswith(norm_p)


def _non_primary_listing_detail(
    ticker: str,
    *,
    cfg: AppConfig,
    ticker_registry_loader: TickerRegistryLoader | None,
) -> str | None:
    """Detect secondary listings of another security's issuer.

    The SEC registry lists every trading symbol per CIK with the common
    stock first; exchange-traded notes and other non-equity listings show
    up as extra symbols (e.g. AMG's junior subordinated notes MGR/MGRB/
    MGRD/MGRE). Valuing the issuer's equity per note-ticker produces fake
    fat pitches, so any non-primary symbol that isn't a share-class
    variant of the primary is quarantined.
    """
    pairs = (
        ticker_registry_loader()
        if ticker_registry_loader is not None
        else _default_ticker_registry(cfg)
    )
    if not pairs:
        return None
    by_cik: dict[str, list[str]] = {}
    cik_of: dict[str, str] = {}
    for cik, symbol in pairs:
        by_cik.setdefault(cik, []).append(symbol)
        cik_of.setdefault(symbol, cik)
    cik = cik_of.get(ticker)
    if not cik:
        return None
    listings = by_cik[cik]
    primary = listings[0]
    if ticker == primary or _is_share_class_variant(primary, ticker):
        return None
    return f"primary={primary}"


def _delisting_notice_detail(
    ticker: str,
    *,
    as_of: date,
    cfg: AppConfig,
    submissions_loader: SubmissionsLoader | None,
) -> str | None:
    """8-K item 3.01 filed in the trailing 12 months as-of the gate date.

    Filed dates come straight from the submissions payload, so the check is
    point-in-time by construction. Returns a detail string when triggered.
    """
    loader = submissions_loader or (lambda t: _load_cached_submissions(t, cfg))
    payload = loader(ticker)
    if not isinstance(payload, dict):
        return None
    recent = payload.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        return None
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    items_list = recent.get("items") or []
    window_start = as_of - timedelta(days=DELISTING_LOOKBACK_DAYS)
    hits: list[str] = []
    for idx, form in enumerate(forms):
        if not str(form).upper().startswith("8-K"):
            continue
        filed = _parse_date(filing_dates[idx] if idx < len(filing_dates) else None)
        if filed is None or filed > as_of or filed <= window_start:
            continue
        items = str(items_list[idx] if idx < len(items_list) else "")
        if "3.01" in {token.strip() for token in items.split(",")}:
            hits.append(filed.isoformat())
    if not hits:
        return None
    hits.sort(reverse=True)
    return (
        f"8-K item 3.01 filed {hits[0]} ({len(hits)} in trailing 12m); "
        "cure_detection=not_available_v1"
    )


def _fy_visibility_date(
    conn: sqlite3.Connection,
    ticker: str,
    period_end: str,
    *,
    annual: bool,
) -> date | None:
    """Exact CompanyFacts filing date for the NI/CFO period.

    A period-end lag is not filing evidence. Missing dates therefore remain
    invisible instead of becoming investable facts on a synthetic schedule.
    """
    _ = annual
    try:
        row = conn.execute(
            """
            SELECT MAX(filed_date)
            FROM companyfacts_facts
            WHERE ticker = ?
              AND period_end = ?
              AND line_item IN ('net_income', 'cfo')
              AND filed_date IS NOT NULL
              AND TRIM(filed_date) != ''
            """,
            (str(ticker).upper(), period_end),
        ).fetchone()
    except sqlite3.Error:
        row = None
    if row is not None:
        filed = _parse_date(row[0])
        if filed is not None:
            return filed
    return None


def _is_missing_schema_error(exc: sqlite3.Error) -> bool:
    """True for missing-table/column errors — a legitimately minimal DB.

    Those degrade to "no signal" (a fresh checkout has no filings table);
    every other sqlite error (locked, I/O, corruption) propagates so the
    gate reports EXCLUDED_ERROR instead of silently passing the name.
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return "no such table" in message or "no such column" in message


def _earnings_quality_rows(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of: date,
    issuer_aware: bool,
    issuer_cik: str | None,
    aliases: Sequence[str],
) -> list[sqlite3.Row]:
    """Load NI/CFO facts with v1 or strict issuer-bound v2 identity."""

    if issuer_aware:
        from app.util.financial_data_access import issuer_companyfacts_rows

        _scope, rows = issuer_companyfacts_rows(
            conn,
            ticker,
            columns=("fiscal_year", "period_type", "period_end", "line_item", "value"),
            issuer_cik=issuer_cik,
            aliases=aliases,
            line_items=("net_income", "cfo"),
            as_of_date=as_of.isoformat(),
            value_not_null=True,
            require_filed_asof=True,
            order_by="period_end DESC, fiscal_year DESC, line_item ASC",
        )
        return rows
    from app.util.financial_data_access import companyfacts_rows

    return companyfacts_rows(
        conn,
        ticker,
        columns=(
            "fiscal_year",
            "period_type",
            "period_end",
            "line_item",
            "value",
        ),
        line_items=("net_income", "cfo"),
        as_of_date=as_of.isoformat(),
        value_not_null=True,
        require_filed_asof=True,
        order_by="period_end DESC, fiscal_year DESC, line_item ASC",
    )


def _earnings_quality_divergence_detail(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of: date,
    issuer_aware: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
) -> str | None:
    """Positive reported net income with negative operating cash flow.

    Latest visible FY first; TTM over the latest four visible quarters where
    quarterly facts allow (both line items present for all four).
    """
    try:
        rows = _earnings_quality_rows(
            conn,
            ticker,
            as_of=as_of,
            issuer_aware=issuer_aware,
            issuer_cik=issuer_cik,
            aliases=aliases,
        )
    except sqlite3.Error as exc:
        if _is_missing_schema_error(exc):
            return None
        raise

    annual: dict[tuple[int, str], dict[str, float]] = {}
    quarterly: dict[str, dict[str, float]] = {}
    for fiscal_year, period_type, period_end, line_item, value in rows:
        if str(period_type) == "FY":
            annual.setdefault((int(fiscal_year), str(period_end)), {})[str(line_item)] = float(
                value
            )
        elif str(period_type) in {"Q1", "Q2", "Q3", "Q4"}:
            quarterly.setdefault(str(period_end), {})[str(line_item)] = float(value)

    # Latest FY with both line items, visible as-of the gate date (filed date
    # from the filings table when present, conservative lag otherwise).
    for (fiscal_year, period_end), values in sorted(annual.items(), reverse=True):
        if "net_income" not in values or "cfo" not in values:
            continue
        if not issuer_aware:
            visible = _fy_visibility_date(conn, ticker, period_end, annual=True)
            if visible is None or visible > as_of:
                continue
        net_income = values["net_income"]
        cfo = values["cfo"]
        if net_income > 0 and cfo < 0:
            return f"basis=FY{fiscal_year}:net_income={net_income}:cfo={cfo}"
        break  # only the LATEST visible FY decides the annual basis

    # TTM where quarterly facts allow: the four most recent visible quarters
    # with BOTH line items, and they must be CONSECUTIVE (window <= ~1 year).
    # Quarterly CFO is sparse in companyfacts (10-Qs often file YTD only), so
    # without the consecutiveness check the "TTM" could stitch four Q1s from
    # different years into a meaningless seasonal sum.
    visible_quarters: list[tuple[str, dict[str, float]]] = []
    for period_end, values in sorted(quarterly.items(), reverse=True):
        if "net_income" not in values or "cfo" not in values:
            continue
        if not issuer_aware:
            visible = _fy_visibility_date(conn, ticker, period_end, annual=False)
            if visible is None or visible > as_of:
                continue
        visible_quarters.append((period_end, values))
        if len(visible_quarters) == 4:
            break
    if len(visible_quarters) == 4:
        newest = _parse_date(visible_quarters[0][0])
        oldest = _parse_date(visible_quarters[-1][0])
        if newest is not None and oldest is not None and (newest - oldest).days <= 300:
            ttm_ni = sum(values["net_income"] for _, values in visible_quarters)
            ttm_cfo = sum(values["cfo"] for _, values in visible_quarters)
            if ttm_ni > 0 and ttm_cfo < 0:
                window = f"{oldest.isoformat()}..{newest.isoformat()}"
                return f"basis=TTM:{window}:net_income={round(ttm_ni, 4)}:cfo={round(ttm_cfo, 4)}"
    return None


def _latest_visible_earnings_observation(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of: date,
    issuer_aware: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
) -> tuple[dict[str, Any] | None, str | None]:
    """Latest comparable NI/CFO observation and its immutable fact reference."""

    try:
        rows = _earnings_quality_rows(
            conn,
            ticker,
            as_of=as_of,
            issuer_aware=issuer_aware,
            issuer_cik=issuer_cik,
            aliases=aliases,
        )
    except sqlite3.Error as exc:
        if _is_missing_schema_error(exc):
            return None, None
        raise
    grouped: dict[tuple[str, str, int], dict[str, float]] = {}
    for fiscal_year, period_type, period_end, line_item, value in rows:
        key = (str(period_type), str(period_end), int(fiscal_year))
        grouped.setdefault(key, {})[str(line_item)] = float(value)
    issuer_key = (
        f"CIK{str(issuer_cik).strip().zfill(10)}" if issuer_aware and issuer_cik else ticker
    )

    visible_quarters: list[tuple[str, int, dict[str, float]]] = []
    for (period_type, period_end, fiscal_year), values in sorted(
        grouped.items(), key=lambda item: item[0][1], reverse=True
    ):
        if period_type not in {"Q1", "Q2", "Q3", "Q4"} or not {"net_income", "cfo"}.issubset(
            values
        ):
            continue
        if not issuer_aware:
            visible = _fy_visibility_date(conn, ticker, period_end, annual=False)
            if visible is None or visible > as_of:
                continue
        visible_quarters.append((period_end, fiscal_year, values))
        if len(visible_quarters) == 4:
            break
    if len(visible_quarters) == 4:
        newest = _parse_date(visible_quarters[0][0])
        oldest = _parse_date(visible_quarters[-1][0])
        if newest is not None and oldest is not None and (newest - oldest).days <= 300:
            observation = {
                "basis": "TTM",
                "period_start": oldest.isoformat(),
                "period_end": newest.isoformat(),
                "net_income": sum(item[2]["net_income"] for item in visible_quarters),
                "cfo": sum(item[2]["cfo"] for item in visible_quarters),
            }
            return (
                observation,
                f"companyfacts:{issuer_key}:TTM:{oldest.isoformat()}:{newest.isoformat()}",
            )

    for (period_type, period_end, fiscal_year), values in sorted(
        grouped.items(), key=lambda item: item[0][1], reverse=True
    ):
        if period_type != "FY" or not {"net_income", "cfo"}.issubset(values):
            continue
        if not issuer_aware:
            visible = _fy_visibility_date(conn, ticker, period_end, annual=True)
            if visible is None or visible > as_of:
                continue
        return (
            {
                "basis": f"FY{fiscal_year}",
                "period_end": period_end,
                "net_income": values["net_income"],
                "cfo": values["cfo"],
            },
            f"companyfacts:{issuer_key}:FY:{period_end}",
        )
    return None, None


def _strip_filing_markup(raw: str) -> str:
    text = _HTML_TAG_RE.sub(" ", raw)
    return text.replace("&nbsp;", " ").replace("&#160;", " ")


def _default_filing_text_loader(local_path: str) -> str | None:
    path = Path(local_path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _going_concern_detail(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of: date,
    filing_text_loader: FilingTextLoader | None,
    issuer_aware: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
) -> str | None:
    """Attributed blockable assertion in the latest cached annual/quarterly filing."""

    source, _raw, assertions = _going_concern_assertion_evidence(
        conn,
        ticker,
        as_of=as_of,
        filing_text_loader=filing_text_loader,
        issuer_aware=issuer_aware,
        issuer_cik=issuer_cik,
        aliases=aliases,
    )
    if source is None:
        return None
    for assertion in assertions:
        if assertion.blockable:
            excerpt = re.sub(r"\s+", " ", assertion.excerpt)[:180]
            return (
                f"{source['form_type']} filed {source['filing_date']}: "
                f"subject={assertion.subject}; mode={assertion.assertion_mode}; "
                f'section={assertion.section or "UNKNOWN"}; "{excerpt}"'
            )
    return None


def _latest_going_concern_filing_source(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of: date,
    filing_text_loader: FilingTextLoader | None,
    issuer_aware: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from app.util.financial_data_access import (
            FINANCIAL_CACHED_FILING_FORM_TYPES,
            issuer_filing_rows,
            latest_filing_row,
        )

        columns = (
            "accession",
            "form_type",
            "filing_date",
            "primary_doc_url",
            "local_path",
        )
        if issuer_aware:
            scope, rows = issuer_filing_rows(
                conn,
                ticker,
                columns=columns,
                issuer_cik=issuer_cik,
                aliases=aliases,
                form_types=FINANCIAL_CACHED_FILING_FORM_TYPES,
                as_of_date=as_of.isoformat(),
                require_local_path=True,
                limit=1,
            )
            row = rows[0] if rows else None
            resolved_issuer_cik = scope.issuer_cik
        else:
            row = latest_filing_row(
                conn,
                ticker,
                columns=columns,
                form_types=FINANCIAL_CACHED_FILING_FORM_TYPES,
                as_of_date=as_of.isoformat(),
                require_local_path=True,
            )
            resolved_issuer_cik = None
    except sqlite3.Error as exc:
        if _is_missing_schema_error(exc):
            return None, None
        raise
    if row is None:
        return None, None
    loader = filing_text_loader or _default_filing_text_loader
    raw = loader(str(row["local_path"]))
    source = {
        "accession": str(row["accession"] or ""),
        "form_type": str(row["form_type"] or ""),
        "filing_date": str(row["filing_date"] or ""),
        "primary_doc_url": str(row["primary_doc_url"] or ""),
        "local_path": str(row["local_path"] or ""),
        "issuer_cik": resolved_issuer_cik,
    }
    return source, raw


def _going_concern_assertion_evidence(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of: date,
    filing_text_loader: FilingTextLoader | None,
    issuer_aware: bool = False,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
) -> tuple[dict[str, Any] | None, str | None, list[Any]]:
    source, raw = _latest_going_concern_filing_source(
        conn,
        ticker,
        as_of=as_of,
        filing_text_loader=filing_text_loader,
        issuer_aware=issuer_aware,
        issuer_cik=issuer_cik,
        aliases=aliases,
    )
    if source is None or not raw:
        return source, raw, []
    from app.alpha.solvency_scanner import detect_going_concern_assertions

    assertions = detect_going_concern_assertions(
        _strip_filing_markup(raw),
        ticker=ticker,
        accession=source["accession"],
        form_type=source["form_type"],
        filing_date=source["filing_date"],
        issuer_cik=source["issuer_cik"],
        source_url=source["primary_doc_url"],
        content_revision=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        section=None,
        corroborating_distress=(),
    )
    return source, raw, assertions


def evaluate_going_concern_filing_text(
    ticker: str,
    filing_text: str,
    *,
    accession: str | None,
    form_type: str | None,
    filing_date: str | None,
    issuer_cik: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Evaluate one filing with the production attributed going-concern gate.

    The returned observation always binds a readable filing to its immutable
    source identity and raw-content revision, including when the detector
    finds zero assertions.  Acceptance replays use this pure entry point so
    they execute the same detector and blocking rule as live structural gates.
    """

    ticker_norm = str(ticker or "").strip().upper()
    raw = str(filing_text or "")
    accession_norm = str(accession or "").strip() or None
    form_type_norm = str(form_type or "").strip() or None
    filing_date_norm = str(filing_date or "").strip() or None
    issuer_cik_norm = str(issuer_cik or "").strip() or None
    source_url_norm = str(source_url or "").strip() or None
    content_revision = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    evidence_ref_id = f"sec-filing:{ticker_norm}:{accession_norm or 'unknown-accession'}"

    if not raw.strip():
        return {
            "status": "INCOMPLETE",
            "reason_code": "ANNUAL_OR_QUARTERLY_FILING_TEXT_UNAVAILABLE",
            "observed_value": None,
            "evidence_ref_id": evidence_ref_id,
            "evidence_url": source_url_norm,
        }

    from app.alpha.solvency_scanner import detect_going_concern_assertions

    assertions = detect_going_concern_assertions(
        _strip_filing_markup(raw),
        ticker=ticker_norm,
        accession=accession_norm,
        form_type=form_type_norm,
        filing_date=filing_date_norm,
        issuer_cik=issuer_cik_norm,
        source_url=source_url_norm,
        content_revision=content_revision,
        section=None,
        corroborating_distress=(),
    )
    blockable_assertion = next(
        (assertion for assertion in assertions if assertion.blockable),
        None,
    )
    if blockable_assertion is not None:
        observed_value = blockable_assertion.to_dict()
        status = "FAIL"
        reason_code = f"{QUARANTINE_STRUCTURAL_PREFIX}:{STRUCTURAL_GOING_CONCERN}"
    else:
        observed_value = {
            "assertion": "NO_BLOCKABLE_ATTRIBUTED_ASSERTION",
            "accession": accession_norm,
            "form_type": form_type_norm,
            "filing_date": filing_date_norm,
            "issuer_cik": issuer_cik_norm,
            "source_url": source_url_norm,
            "content_revision": content_revision,
            "assertions": [assertion.to_dict() for assertion in assertions],
        }
        status = "PASS"
        reason_code = None

    return {
        "status": status,
        "reason_code": reason_code,
        "observed_value": observed_value,
        "evidence_ref_id": evidence_ref_id,
        "evidence_url": source_url_norm,
    }


def _build_v2_structural_screen(
    *,
    contract_id: str,
    applicable_rule_ids: set[str],
    triggered: dict[str, str],
    observations: dict[str, Any],
    thresholds: dict[str, Any],
    evidence_refs: dict[str, str],
    evidence_urls: dict[str, str],
    incomplete_reasons: dict[str, str],
) -> tuple[list[GateEvaluation], ScreenResult]:
    evaluations: list[GateEvaluation] = []
    for code in STRUCTURAL_CODES:
        if code not in applicable_rule_ids:
            evaluations.append(
                GateEvaluation(
                    contract_id=contract_id,
                    rule_id=code,
                    status="NOT_APPLICABLE",
                    applicable=False,
                    observed_value=observations.get(code),
                    threshold=thresholds.get(code),
                    reason_code="SECTOR_RULE_NOT_APPLICABLE",
                )
            )
            continue
        if code in triggered:
            evaluations.append(
                GateEvaluation(
                    contract_id=contract_id,
                    rule_id=code,
                    status="FAIL",
                    applicable=True,
                    observed_value=observations.get(code, triggered[code]),
                    threshold=thresholds[code],
                    evidence_ref_id=evidence_refs.get(code),
                    evidence_url=evidence_urls.get(code),
                    reason_code=f"{QUARANTINE_STRUCTURAL_PREFIX}:{code}",
                    notes=[triggered[code]],
                )
            )
            continue
        incomplete_reason = incomplete_reasons.get(code)
        if incomplete_reason:
            evaluations.append(
                GateEvaluation(
                    contract_id=contract_id,
                    rule_id=code,
                    status="INCOMPLETE",
                    applicable=True,
                    observed_value=observations.get(code),
                    threshold=thresholds.get(code),
                    evidence_ref_id=evidence_refs.get(code),
                    evidence_url=evidence_urls.get(code),
                    reason_code=incomplete_reason,
                )
            )
            continue
        evaluations.append(
            GateEvaluation(
                contract_id=contract_id,
                rule_id=code,
                status="PASS",
                applicable=True,
                observed_value=observations[code],
                threshold=thresholds[code],
                evidence_ref_id=evidence_refs.get(code),
                evidence_url=evidence_urls.get(code),
            )
        )

    failed = [item for item in evaluations if item.status == "FAIL"]
    incomplete = [item for item in evaluations if item.status == "INCOMPLETE"]
    status = "FAIL" if failed else "INCOMPLETE" if incomplete else "PASS"
    reasons = [str(item.reason_code) for item in [*failed, *incomplete] if item.reason_code]
    screen = ScreenResult(
        contract_id=contract_id,
        status=status,
        gate_evaluations=evaluations,
        required_rule_ids=list(STRUCTURAL_CODES),
        reason_codes=reasons,
    )
    return evaluations, screen


def evaluate_structural_gate(
    ticker: str,
    *,
    as_of_date: str,
    price: float | None = None,
    market_cap_mm: float | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    submissions_loader: SubmissionsLoader | None = None,
    filing_text_loader: FilingTextLoader | None = None,
    ticker_registry_loader: TickerRegistryLoader | None = None,
    pipeline_version: str = "v1",
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    sector_contract_id: str | None = None,
    applicable_rule_ids: set[str] | None = None,
    price_evidence_ref: str | None = None,
    price_evidence_url: str | None = None,
    cap_evidence_ref: str | None = None,
    cap_evidence_url: str | None = None,
) -> StructuralGateResult:
    """Evaluate every structural trigger for one name as-of a date.

    ``price`` is the classification price from the band-filter chain and
    ``market_cap_mm`` its computable cap (None skips the respective floor —
    the gate never quarantines on missing data, only on affirmative
    evidence). Point-in-time safety: filed dates gate every fundamental and
    filing trigger, so the backtest can apply the identical function.
    """
    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    asof_norm = str(as_of_date or "").strip()
    as_of = _parse_date(asof_norm) or date.today()
    issuer_aware = str(pipeline_version or "v1").strip().lower() == "v2"

    triggered: dict[str, str] = {}
    observations: dict[str, Any] = {}
    thresholds: dict[str, Any] = {
        STRUCTURAL_NON_PRIMARY_LISTING: "PRIMARY_OR_AUTHORIZED_SHARE_CLASS",
        STRUCTURAL_DELISTING_NOTICE: f"NO_ITEM_3_01_WITHIN_{DELISTING_LOOKBACK_DAYS}_DAYS",
        STRUCTURAL_PENNY_FLOOR: {
            "operator": ">=",
            "value": PENNY_FLOOR_PRICE_USD,
            "units": "USD",
        },
        STRUCTURAL_NANO_FLOOR: {
            "operator": ">=",
            "value": NANO_FLOOR_MARKET_CAP_MM,
            "units": "USD_millions",
        },
        STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE: "NOT(net_income>0 AND cfo<0)",
        STRUCTURAL_GOING_CONCERN: "NO_CORROBORATED_ATTRIBUTED_GOING_CONCERN_ASSERTION",
    }
    evidence_refs: dict[str, str] = {}
    evidence_urls: dict[str, str] = {}
    incomplete_reasons: dict[str, str] = {}

    registry_loader = ticker_registry_loader or (lambda: _default_ticker_registry(cfg))
    registry_pairs = registry_loader()
    non_primary = _non_primary_listing_detail(
        ticker_norm,
        cfg=cfg,
        ticker_registry_loader=lambda: registry_pairs,
    )
    if non_primary:
        triggered[STRUCTURAL_NON_PRIMARY_LISTING] = non_primary
        observations[STRUCTURAL_NON_PRIMARY_LISTING] = non_primary
    elif registry_pairs:
        observations[STRUCTURAL_NON_PRIMARY_LISTING] = "PRIMARY_OR_AUTHORIZED_SHARE_CLASS"
    else:
        incomplete_reasons[STRUCTURAL_NON_PRIMARY_LISTING] = "SEC_TICKER_REGISTRY_UNAVAILABLE"
    evidence_refs[STRUCTURAL_NON_PRIMARY_LISTING] = (
        f"sec-company-ticker-registry:{ticker_norm}:{asof_norm}"
    )
    evidence_urls[STRUCTURAL_NON_PRIMARY_LISTING] = (
        "https://www.sec.gov/files/company_tickers_exchange.json"
    )

    submissions_reader = submissions_loader or (lambda t: _load_cached_submissions(t, cfg))
    submissions_payload = submissions_reader(ticker_norm)
    delisting = _delisting_notice_detail(
        ticker_norm,
        as_of=as_of,
        cfg=cfg,
        submissions_loader=lambda _ticker: submissions_payload,
    )
    if delisting:
        triggered[STRUCTURAL_DELISTING_NOTICE] = delisting
        observations[STRUCTURAL_DELISTING_NOTICE] = delisting
    elif isinstance(submissions_payload, dict):
        observations[STRUCTURAL_DELISTING_NOTICE] = "NO_ITEM_3_01_IN_LOOKBACK"
    else:
        incomplete_reasons[STRUCTURAL_DELISTING_NOTICE] = "SEC_SUBMISSIONS_UNAVAILABLE"
    evidence_refs[STRUCTURAL_DELISTING_NOTICE] = f"sec-submissions:{ticker_norm}:{asof_norm}"

    if (
        isinstance(price, (int, float))
        and float(price) > 0
        and float(price) < PENNY_FLOOR_PRICE_USD
    ):
        triggered[STRUCTURAL_PENNY_FLOOR] = f"price={float(price):.2f}"
    if isinstance(price, (int, float)) and float(price) > 0:
        observations[STRUCTURAL_PENNY_FLOOR] = float(price)
    else:
        incomplete_reasons[STRUCTURAL_PENNY_FLOOR] = "SCREEN_PRICE_UNAVAILABLE"
    evidence_refs[STRUCTURAL_PENNY_FLOOR] = price_evidence_ref or (
        f"company-packet:{ticker_norm}:cap-stage-price:{asof_norm}"
    )
    if price_evidence_url:
        evidence_urls[STRUCTURAL_PENNY_FLOOR] = price_evidence_url

    if (
        isinstance(market_cap_mm, (int, float))
        and float(market_cap_mm) > 0
        and float(market_cap_mm) < NANO_FLOOR_MARKET_CAP_MM
    ):
        triggered[STRUCTURAL_NANO_FLOOR] = f"market_cap_mm={float(market_cap_mm):.1f}"
    if isinstance(market_cap_mm, (int, float)) and float(market_cap_mm) > 0:
        observations[STRUCTURAL_NANO_FLOOR] = float(market_cap_mm)
    else:
        incomplete_reasons[STRUCTURAL_NANO_FLOOR] = "SCREEN_MARKET_CAP_UNAVAILABLE"
    evidence_refs[STRUCTURAL_NANO_FLOOR] = cap_evidence_ref or (
        f"company-packet:{ticker_norm}:market-cap:{asof_norm}"
    )
    if cap_evidence_url:
        evidence_urls[STRUCTURAL_NANO_FLOOR] = cap_evidence_url

    # Fail-closed: the two DB-backed detectors (earnings-quality
    # divergence, going-concern) must run against a readable engine DB. A
    # missing/locked/unreadable DB is recorded as a degraded code so
    # consumers exclude the name (EXCLUDED_ERROR) instead of passing it
    # through the quarantine gate unexamined.
    degraded: list[str] = []
    advisory: list[str] = []
    path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    if not path.exists():
        degraded.append("ENGINE_DB_MISSING")
        incomplete_reasons[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE] = "ENGINE_DB_MISSING"
        incomplete_reasons[STRUCTURAL_GOING_CONCERN] = "ENGINE_DB_MISSING"
    else:
        from app.db import connect as db_connect

        conn = None
        try:
            conn = db_connect(path)
        except sqlite3.Error:
            degraded.append("ENGINE_DB_CONNECT_ERROR")
            incomplete_reasons[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE] = "ENGINE_DB_CONNECT_ERROR"
            incomplete_reasons[STRUCTURAL_GOING_CONCERN] = "ENGINE_DB_CONNECT_ERROR"
        if conn is not None:
            try:
                try:
                    divergence = _earnings_quality_divergence_detail(
                        conn,
                        ticker_norm,
                        as_of=as_of,
                        issuer_aware=issuer_aware,
                        issuer_cik=issuer_cik,
                        aliases=aliases,
                    )
                    earnings_observation, earnings_ref = _latest_visible_earnings_observation(
                        conn,
                        ticker_norm,
                        as_of=as_of,
                        issuer_aware=issuer_aware,
                        issuer_cik=issuer_cik,
                        aliases=aliases,
                    )
                    if divergence:
                        triggered[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE] = divergence
                        observations[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE] = divergence
                    elif earnings_observation is not None:
                        observations[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE] = earnings_observation
                    else:
                        incomplete_reasons[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE] = (
                            "EARNINGS_QUALITY_FACTS_UNAVAILABLE"
                        )
                    if earnings_ref:
                        evidence_refs[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE] = earnings_ref
                except sqlite3.Error:
                    degraded.append("EARNINGS_QUALITY_DB_ERROR")
                    incomplete_reasons[STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE] = (
                        "EARNINGS_QUALITY_DB_ERROR"
                    )
                try:
                    filing_source, filing_raw = _latest_going_concern_filing_source(
                        conn,
                        ticker_norm,
                        as_of=as_of,
                        filing_text_loader=filing_text_loader,
                        issuer_aware=issuer_aware,
                        issuer_cik=issuer_cik,
                        aliases=aliases,
                    )
                    if filing_source is not None:
                        going_evaluation = evaluate_going_concern_filing_text(
                            ticker_norm,
                            filing_raw or "",
                            accession=filing_source["accession"],
                            form_type=filing_source["form_type"],
                            filing_date=filing_source["filing_date"],
                            issuer_cik=filing_source["issuer_cik"],
                            source_url=filing_source["primary_doc_url"],
                        )
                    else:
                        going_evaluation = None
                    if going_evaluation is not None and going_evaluation["status"] == "FAIL":
                        observed_value = going_evaluation["observed_value"]
                        excerpt = re.sub(r"\s+", " ", str(observed_value["excerpt"]))[:180]
                        going_concern = (
                            f"{observed_value['form_type']} filed "
                            f"{observed_value['filing_date']}: "
                            f"subject={observed_value['subject']}; "
                            f"mode={observed_value['assertion_mode']}; "
                            f"section={observed_value['section'] or 'UNKNOWN'}; "
                            f'"{excerpt}"'
                        )
                        triggered[STRUCTURAL_GOING_CONCERN] = going_concern
                        observations[STRUCTURAL_GOING_CONCERN] = observed_value
                    elif going_evaluation is not None and going_evaluation["status"] == "PASS":
                        observations[STRUCTURAL_GOING_CONCERN] = going_evaluation["observed_value"]
                    else:
                        incomplete_reasons[STRUCTURAL_GOING_CONCERN] = str(
                            (going_evaluation or {}).get("reason_code")
                            or "ANNUAL_OR_QUARTERLY_FILING_TEXT_UNAVAILABLE"
                        )
                    if going_evaluation is not None:
                        evidence_refs[STRUCTURAL_GOING_CONCERN] = going_evaluation[
                            "evidence_ref_id"
                        ]
                        if going_evaluation["evidence_url"]:
                            evidence_urls[STRUCTURAL_GOING_CONCERN] = filing_source[
                                "primary_doc_url"
                            ]
                except sqlite3.Error:
                    degraded.append("GOING_CONCERN_DB_ERROR")
                    incomplete_reasons[STRUCTURAL_GOING_CONCERN] = "GOING_CONCERN_DB_ERROR"
                # Advisory (flag, never reject): a computable dollar-ADV
                # below the floor marks the name untradeable-at-size before
                # any LLM spend. Missing volume history stays silent — the
                # gate never flags on absent data.
                try:
                    from app.market.adv import _dollar_adv_window, adv_dollar_floor

                    adv20 = _dollar_adv_window(conn, ticker_norm, as_of_date=asof_norm, window=20)
                    if adv20 is not None and adv20 < adv_dollar_floor():
                        advisory.append(
                            f"MIN_ADV:adv20_usd={adv20:.0f}:floor={adv_dollar_floor():.0f}"
                        )
                except sqlite3.Error:
                    pass
            finally:
                conn.close()

    codes = [code for code in STRUCTURAL_CODES if code in triggered]
    gate_evaluations: list[GateEvaluation] = []
    screen_result: ScreenResult | None = None
    contract_id: str | None = None
    if issuer_aware:
        contract_id = str(sector_contract_id or "").strip()
        if not contract_id:
            raise ValueError("v2 structural gate requires sector_contract_id")
        applicable = (
            {str(item).strip().upper() for item in applicable_rule_ids if str(item).strip()}
            if applicable_rule_ids is not None
            else set(STRUCTURAL_CODES)
        )
        gate_evaluations, screen_result = _build_v2_structural_screen(
            contract_id=contract_id,
            applicable_rule_ids=applicable,
            triggered=triggered,
            observations=observations,
            thresholds=thresholds,
            evidence_refs=evidence_refs,
            evidence_urls=evidence_urls,
            incomplete_reasons=incomplete_reasons,
        )
    return StructuralGateResult(
        ticker=ticker_norm,
        as_of_date=asof_norm,
        triggered_codes=codes,
        reasons=[f"{QUARANTINE_STRUCTURAL_PREFIX}:{code}" for code in codes],
        details={code: triggered[code] for code in codes},
        degraded_codes=degraded,
        advisory_codes=advisory,
        contract_id=contract_id,
        gate_evaluations=gate_evaluations,
        screen_result=screen_result,
    )


__all__ = [
    "EXCLUDED_ERROR_PREFIX",
    "QUARANTINE_STRUCTURAL_PREFIX",
    "STRUCTURAL_CODES",
    "STRUCTURAL_DELISTING_NOTICE",
    "STRUCTURAL_EARNINGS_QUALITY_DIVERGENCE",
    "STRUCTURAL_GOING_CONCERN",
    "STRUCTURAL_NANO_FLOOR",
    "STRUCTURAL_NON_PRIMARY_LISTING",
    "STRUCTURAL_PENNY_FLOOR",
    "NANO_FLOOR_MARKET_CAP_MM",
    "PENNY_FLOOR_PRICE_USD",
    "StructuralGateResult",
    "evaluate_going_concern_filing_text",
    "evaluate_structural_gate",
]
