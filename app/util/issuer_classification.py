from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable


logger = logging.getLogger(__name__)


ISSUER_CLASS_OPERATING = "operating"
ISSUER_CLASS_FINANCIAL = "financial"

FINANCIAL_KEYWORDS = (
    "bank",
    "banking",
    "bank holding company",
    "insurance",
    "insurer",
    "broker-dealer",
    "broker dealer",
    "asset management",
    "wealth management",
    "commercial lending",
    "deposits",
    "underwriting",
    "credit card",
    "loan portfolio",
    "consumer banking",
)

BANK_LIKE_LINE_ITEMS = {
    "deposits",
    "loans",
    "investment_securities",
    "trading_assets",
    "total_assets",
    "assets_under_management",
    "allowance_for_credit_losses",
    "provision_for_credit_losses",
    "net_charge_offs",
    "nonaccrual_loans",
    "tier_1_capital",
    "tier_1_leverage_ratio",
}


def _normalized_tokens(values: Iterable[str] | None) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        token = str(value or "").strip().lower()
        if token:
            out.add(token)
    return out


def infer_issuer_classification(
    *,
    texts: Iterable[str] | None = None,
    line_items: Iterable[str] | None = None,
) -> str:
    joined_text = " ".join(str(text or "").lower() for text in (texts or []))
    line_item_set = _normalized_tokens(line_items)

    line_item_hits = len(line_item_set & BANK_LIKE_LINE_ITEMS)
    keyword_hits = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in joined_text)

    if {"deposits", "loans"} <= line_item_set:
        return ISSUER_CLASS_FINANCIAL
    if line_item_hits >= 2:
        return ISSUER_CLASS_FINANCIAL
    if line_item_hits >= 1 and keyword_hits >= 1:
        return ISSUER_CLASS_FINANCIAL
    if keyword_hits >= 2:
        return ISSUER_CLASS_FINANCIAL
    if "bank holding company" in joined_text or "insurance company" in joined_text:
        return ISSUER_CLASS_FINANCIAL
    return ISSUER_CLASS_OPERATING


# ── classification by SEC SIC code ──────────────────────────────────────────
#
# The substring classifier above reads XBRL tag NAMES and entity text. Measured
# on the liquid universe it calls 71 of the top 200 US companies financial —
# NVIDIA, Tesla, Costco and Mastercard among them — which refuses their net-debt
# bridge and blacks out every valuation downstream of it. The SEC's own SIC code
# answers the question the bridge actually asks (is this a bank, insurer or
# broker whose deposits and float make net debt meaningless) deterministically,
# from the registrant table this repo already maintains.
#
# Financial: 6000-6499 — depositories, credit, brokers, insurance — plus the
# bank-holding and fund codes 6712/6719/6722/6726. Real estate and REITs (65xx,
# 6798) stay OPERATING: a REIT's net debt is real, priced and central.
_SIC_FINANCIAL_PREFIXES = {"60", "61", "62", "63", "64"}
_SIC_FINANCIAL_EXACT = {"6712", "6719", "6722", "6726"}


def classify_issuer_by_sic(sic: object) -> str | None:
    """Classify by SIC code, or None when no usable code was supplied."""
    token = str(sic if sic is not None else "").strip()
    if not token.isdigit():
        return None
    token = token.zfill(4)
    if token in _SIC_FINANCIAL_EXACT or token[:2] in _SIC_FINANCIAL_PREFIXES:
        return ISSUER_CLASS_FINANCIAL
    return ISSUER_CLASS_OPERATING


# Real-estate investment trusts: SEC SIC 6798. Still OPERATING for net debt, but
# earnings power and EV/EBIT capitalise operating income after real-estate
# depreciation, which for a REIT is mostly not an economic cost, so both are
# NOT_APPLICABLE for them (a deliberate policy). No FFO model stands in.
SIC_REIT = "6798"
REASON_REIT_DEPRECIATION_DISTORTS_EARNINGS = "REIT_DEPRECIATION_DISTORTS_EARNINGS"


def is_reit_sic(sic: object) -> bool:
    """True only for the SEC's real-estate-investment-trust code, 6798."""
    token = str(sic if sic is not None else "").strip()
    return token.isdigit() and token.zfill(4) == SIC_REIT


def lookup_is_reit(
    *,
    cik: object = None,
    ticker: object = None,
    conn: Any = None,
    cfg: Any = None,
) -> tuple[bool, str]:
    """Whether the registrant files under SIC 6798, and the lookup's reason code.

    A registrant with no SIC on file is not treated as a REIT (the reason says
    why): the rule is the SEC's code, never a guess from tag names.
    """
    sic, reason = registrant_sic(cik=cik, ticker=ticker, conn=conn, cfg=cfg)
    return is_reit_sic(sic), reason


def is_financial_issuer(
    *,
    texts: Iterable[str] | None = None,
    line_items: Iterable[str] | None = None,
) -> bool:
    return infer_issuer_classification(texts=texts, line_items=line_items) == ISSUER_CLASS_FINANCIAL


# ── SIC-first resolution for every call site ────────────────────────────────
#
# The net-debt bridge was the first consumer of the SIC classifier; every other
# place that asks "is this a financial issuer" (fundamentals, packets, gaps,
# discovery, tech category, the synthesis agent) used to answer with the tag-name
# substring rule alone and so misfired on the same NVIDIA/Tesla/Costco names. They
# all go through ``resolve_issuer_classification`` now: the SEC SIC code from
# ``sec_registrants`` when one is on file, the substring rule only when it is not
# (or when ``VOE_ISSUER_CLASSIFICATION_BY_SIC=false`` opts out). The second return
# value says which rule answered, so a fallback is never silent.

SOURCE_SIC = "sic"
SOURCE_SUBSTRING = "substring"


def lookup_registrant_sic(
    *,
    cik: object = None,
    ticker: object = None,
    conn: Any = None,
    cfg: Any = None,
) -> tuple[str | None, str]:
    """The SEC SIC code for a registrant, and why it is missing when it is.

    Returns ``(sic, reason)`` with reason one of OK, NO_IDENTIFIER, LOOKUP_FAILED,
    NO_REGISTRANT_ROW, NO_SIC_ON_FILE. Looks up by CIK when one is given, else by
    ticker (primary ticker first, then the registrant's alias list). Pass ``conn``
    to reuse an open connection; otherwise the configured database is opened.
    """
    cik_token = str(cik if cik is not None else "").strip()
    ticker_token = str(ticker if ticker is not None else "").strip().upper()
    if not cik_token and not ticker_token:
        return None, "NO_IDENTIFIER"

    def _query(c: Any) -> Any:
        if cik_token:
            candidates = {cik_token, cik_token.zfill(10), cik_token.lstrip("0")}
            marks = ",".join("?" for _ in candidates)
            return c.execute(
                f"SELECT sic FROM sec_registrants WHERE cik IN ({marks}) LIMIT 1",
                tuple(candidates),
            ).fetchone()
        row = c.execute(
            "SELECT sic FROM sec_registrants WHERE primary_ticker = ? LIMIT 1",
            (ticker_token,),
        ).fetchone()
        if row is not None:
            return row
        # Alias tickers live in a JSON list; narrow with LIKE, then confirm exactly.
        for candidate in c.execute(
            "SELECT sic, all_tickers FROM sec_registrants WHERE all_tickers LIKE ?",
            (f'%"{ticker_token}"%',),
        ).fetchall():
            try:
                aliases = json.loads(candidate[1] or "[]")
            except (TypeError, ValueError):
                continue
            if isinstance(aliases, list) and ticker_token in {str(a).upper() for a in aliases}:
                return candidate
        return None

    try:
        if conn is not None:
            row = _query(conn)
        else:
            from app.config import get_config
            from app.db import get_db

            with get_db(cfg or get_config()) as own_conn:
                row = _query(own_conn)
    except Exception as exc:  # noqa: BLE001 - a missing table/DB must fall back, not crash
        logger.debug("issuer_classification: SIC lookup failed (%s)", exc)
        return None, "LOOKUP_FAILED"
    value = None
    if row is not None:
        value = row["sic"] if hasattr(row, "keys") else row[0]
    sic = str(value if value is not None else "").strip()
    if not sic and cik_token:
        cached = _cached_sic(cik_token, conn=conn, cfg=cfg)
        if cached:
            return cached, "OK"
    if row is None:
        return None, "NO_REGISTRANT_ROW"
    if not sic:
        return None, "NO_SIC_ON_FILE"
    return sic, "OK"


def _cik_variants(cik_token: str) -> tuple[str, ...]:
    return tuple({cik_token, cik_token.zfill(10), cik_token.lstrip("0")})


def _cached_sic(cik_token: str, *, conn: Any = None, cfg: Any = None) -> str | None:
    """The SIC stored by ``ensure_registrant_sic`` for this CIK, if any."""

    def _query(c: Any) -> str | None:
        variants = _cik_variants(cik_token)
        marks = ",".join("?" for _ in variants)
        row = c.execute(
            f"SELECT sic FROM sec_sic_cache WHERE cik IN ({marks}) LIMIT 1", variants
        ).fetchone()
        if row is None:
            return None
        value = row["sic"] if hasattr(row, "keys") else row[0]
        token = str(value if value is not None else "").strip()
        return token or None

    try:
        if conn is not None:
            return _query(conn)
        from app.config import get_config
        from app.db import get_db

        with get_db(cfg or get_config()) as own_conn:
            return _query(own_conn)
    except Exception as exc:  # noqa: BLE001 - an older database without the table
        logger.debug("issuer_classification: SIC cache read failed (%s)", exc)
        return None


def ensure_registrant_sic(*, cik: object, cfg: Any = None) -> tuple[str | None, str]:
    """Return the SIC for a CIK, fetching it once from the SEC when none is on file.

    On a fresh install ``sec_registrants`` is empty, so the REIT rule (SIC 6798)
    and the bank/insurer routing would silently see no SIC. When the lookup finds
    none, this fetches ``https://data.sec.gov/submissions/CIK##########.json``
    (through the repo's HTTP client, so the on-disk cache and the SEC rate limit
    apply), reads its ``sic`` field and stores it in ``sec_sic_cache`` so the
    shared lookup finds it from then on.

    Returns ``(sic, reason)``. Offline (``VOE_NET_PROVIDER=disabled``), a failed
    fetch or a payload with no usable ``sic`` returns the lookup's own reason
    unchanged: an explicit gap, never a guess.
    """
    from app.config import get_config

    cfg = cfg or get_config()
    sic, reason = lookup_registrant_sic(cik=cik, cfg=cfg)
    if sic:
        return sic, reason
    fetched = _fetch_registrant_sic(str(cik if cik is not None else "").strip(), cfg=cfg)
    return (fetched, "OK") if fetched else (None, reason)


def registrant_sic(
    *,
    cik: object = None,
    ticker: object = None,
    conn: Any = None,
    cfg: Any = None,
) -> tuple[str | None, str]:
    """The SIC every classification path reads: the on-file lookup
    (``lookup_registrant_sic``), and when it has none, one fetch from the SEC
    (``ensure_registrant_sic``'s, cached in ``sec_sic_cache``).

    Only ``ivi value`` used to fetch, so on a fresh install every other path fell back
    to the tag-name rule, which calls Coca-Cola, Procter & Gamble, Costco and Realty
    Income financial and refuses their net debt. A ticker-only caller's CIK is read
    from the ``companies`` table. Offline (``VOE_NET_PROVIDER=disabled``), with no CIK,
    or on a failed fetch the lookup's own answer and reason come back unchanged.
    """
    from app.config import get_config

    cfg = cfg or get_config()
    sic, reason = lookup_registrant_sic(cik=cik, ticker=ticker, conn=conn, cfg=cfg)
    if sic:
        return sic, reason
    cik_token = str(cik if cik is not None else "").strip()
    if not cik_token:
        cik_token = _cik_for_ticker(str(ticker if ticker is not None else ""), conn=conn, cfg=cfg)
    fetched = _fetch_registrant_sic(cik_token, cfg=cfg)
    return (fetched, "OK") if fetched else (None, reason)


def _cik_for_ticker(ticker: str, *, conn: Any = None, cfg: Any = None) -> str:
    """The CIK the ``companies`` table holds for ``ticker``, or ""."""
    token = ticker.strip().upper()
    if not token:
        return ""

    def _query(c: Any) -> str:
        row = c.execute("SELECT cik FROM companies WHERE ticker = ? LIMIT 1", (token,)).fetchone()
        value = None if row is None else (row["cik"] if hasattr(row, "keys") else row[0])
        return str(value if value is not None else "").strip()

    try:
        if conn is not None:
            return _query(conn)
        from app.db import get_db

        with get_db(cfg) as own_conn:
            return _query(own_conn)
    except Exception as exc:  # noqa: BLE001 - no table: no CIK, no fetch
        logger.debug("issuer_classification: CIK lookup failed (%s)", exc)
        return ""


# One fetch per (database, CIK) per process, whatever it returned: a CIK the SEC has no
# SIC for, or a failed request, is not retried on every classification call.
_SIC_FETCHES: dict[tuple[str, str], str | None] = {}


def _fetch_registrant_sic(cik_token: str, *, cfg: Any) -> str | None:
    """Fetch the SIC for ``cik_token`` from the SEC submissions JSON and store it in
    ``sec_sic_cache``. None offline, for a non-numeric CIK, or on a failed or empty fetch."""
    if not cik_token.isdigit():
        return None
    if str(getattr(cfg, "net_provider", "enabled")).strip().lower() == "disabled":
        return None
    memo_key = (str(getattr(cfg, "db_path", "")), cik_token.zfill(10))
    if memo_key in _SIC_FETCHES:
        return _SIC_FETCHES[memo_key]
    _SIC_FETCHES[memo_key] = None
    try:
        from app.ingest.sec_client import SecClient

        payload = SecClient().submissions(cik_token)
    except Exception as exc:  # noqa: BLE001 - offline or blocked: behave as before
        logger.debug("issuer_classification: submissions fetch failed (%s)", exc)
        return None
    raw = payload.get("sic") if isinstance(payload, dict) else None
    token = str(raw if raw is not None else "").strip()
    if not token.isdigit():
        return None
    description = payload.get("sicDescription")
    try:
        from app.db import get_db

        with get_db(cfg) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sec_sic_cache (cik, sic, sic_description, fetched_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    cik_token.zfill(10),
                    token.zfill(4),
                    str(description) if description else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 - storing is best effort; the value is still right
        logger.debug("issuer_classification: SIC cache write failed (%s)", exc)
    _SIC_FETCHES[memo_key] = token.zfill(4)
    return token.zfill(4)


def resolve_issuer_classification(
    *,
    cik: object = None,
    ticker: object = None,
    texts: Iterable[str] | None = None,
    line_items: Iterable[str] | None = None,
    conn: Any = None,
    cfg: Any = None,
) -> tuple[str, str]:
    """Classify an issuer SIC-first and say which rule answered.

    Returns ``(classification, source)``: source is ``"sic"`` when the SEC SIC code
    decided; ``"substring"`` when the SIC override is switched off; and
    ``"substring:<reason>"`` when it is on but no usable SIC code was found (no
    identifier, no registrant row, no SIC on file, unparseable code) — the tag-name
    substring rule then answers, as it always did.
    """
    from app.config import get_config

    cfg = cfg or get_config()
    if getattr(cfg, "issuer_classification_by_sic", False):
        sic, reason = registrant_sic(cik=cik, ticker=ticker, conn=conn, cfg=cfg)
        sic_class = classify_issuer_by_sic(sic)
        if sic_class is not None:
            return sic_class, SOURCE_SIC
        if reason == "OK":
            reason = "SIC_UNPARSEABLE"
        source = f"{SOURCE_SUBSTRING}:{reason}"
    else:
        source = SOURCE_SUBSTRING
    return infer_issuer_classification(texts=texts, line_items=line_items), source
