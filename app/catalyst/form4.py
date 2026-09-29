"""Form-4 stub extraction + transaction-code parse for the catalyst overlay.

Reads Form-4 filing stubs from the cached SEC submissions JSON, then
fetches and parses the form4.xml transaction code. Only an open-market
purchase (transaction code ``P``) counts as an insider BUY; grants (``A``),
option exercises (``M``) and sales (``S``) are excluded so comp-grant noise does
not masquerade as conviction buying.

When the network is disabled the transaction-code confirmation cannot run, so
``enrich_events`` leaves ``transaction_code`` / ``is_purchase`` as ``None`` and
the overlay must surface a WEAK ("codes unverified") signal rather than silence.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

from app.config import get_config
from app.ingest.filings import list_form4_stubs
from app.ingest.sec_client import SecClient

# Form-4 transaction codes. Only an open-market purchase confirms an insider BUY.
OPEN_MARKET_PURCHASE_CODE = "P"


@dataclass
class Form4Event:
    filing_date: date
    accession: str
    primary_doc_url: str
    transaction_code: str | None = None
    is_purchase: bool | None = None
    reporting_owner: str | None = None


def _coerce_as_of(as_of_date: str | date) -> date:
    if isinstance(as_of_date, date):
        return as_of_date
    return date.fromisoformat(str(as_of_date))


def _resolve_cik(conn, ticker: str) -> str | None:
    """Resolve a CIK from the filings table (filings.cik is fully populated).

    companies.cik is only present on a handful of rows, so the filings table is
    the reliable source.
    """
    row = conn.execute(
        "SELECT cik FROM filings WHERE UPPER(ticker) = ? AND cik IS NOT NULL "
        "ORDER BY filing_date DESC LIMIT 1",
        (ticker.upper(),),
    ).fetchone()
    if not row:
        return None
    cik = row["cik"] if not isinstance(row, (tuple, list)) else row[0]
    return str(cik) if cik is not None else None


def read_form4_stubs(
    *,
    ticker: str,
    conn,
    as_of_date: str | date,
    lookback_days: int = 90,
) -> list[Form4Event]:
    """Return Form4Event rows for ``ticker`` filed within ``lookback_days`` of
    ``as_of_date``, sorted filing_date desc then accession desc.

    transaction_code / is_purchase are left None here (populated by the XML
    parse step). Returns an empty list when the CIK cannot be resolved.
    """
    as_of = _coerce_as_of(as_of_date)
    since = as_of - timedelta(days=lookback_days)

    cik = _resolve_cik(conn, ticker)
    if not cik:
        return []

    stubs = [stub for stub in list_form4_stubs(cik, since) if stub.filing_date <= as_of]
    return [
        Form4Event(
            filing_date=stub.filing_date,
            accession=stub.accession,
            primary_doc_url=stub.primary_doc_url,
            transaction_code=None,
            is_purchase=None,
        )
        for stub in stubs
    ]


def _local_name(tag: str) -> str:
    """Strip any XML namespace so ``{ns}transactionCode`` matches plain tags."""
    return tag.rsplit("}", 1)[-1]


def _first_text(element) -> str | None:
    """Return the inner text of an element, unwrapping a nested ``<value>``."""
    for child in element:
        if _local_name(child.tag) == "value" and child.text is not None:
            return child.text.strip()
    if element.text is not None and element.text.strip():
        return element.text.strip()
    return None


def _find_first(root, name: str):
    for element in root.iter():
        if _local_name(element.tag) == name:
            return element
    return None


def _reporting_owner_identity(root) -> str | None:
    """Return a stable reporting-owner identity (CIK preferred, else name).

    Distinct-buyer counting dedupes on this identity, so we prefer the
    ``rptOwnerCik`` (a stable numeric id) and fall back to a normalized
    ``rptOwnerName`` when no CIK is present.
    """
    owner = _find_first(root, "reportingOwnerId")
    if owner is None:
        return None
    cik_el = _find_first(owner, "rptOwnerCik")
    if cik_el is not None and cik_el.text and cik_el.text.strip():
        # Normalize zero padding so '0001111111' and '1111111' dedupe together.
        raw = cik_el.text.strip()
        return f"CIK:{int(raw)}" if raw.isdigit() else f"CIK:{raw}"
    name_el = _find_first(owner, "rptOwnerName")
    if name_el is not None and name_el.text and name_el.text.strip():
        return f"NAME:{name_el.text.strip().upper()}"
    return None


def parse_form4_xml(xml_bytes: bytes) -> dict:
    """Parse a Form-4 ``ownershipDocument`` for its insider-buy transaction.

    Returns ``{transaction_code, shares, price_per_share, is_open_market_purchase,
    reporting_owner}``. ``transaction_code`` is the first non-derivative code (for
    display); ``is_open_market_purchase`` is True if ANY transaction line is
    ``P``-coded (an open-market purchase) — grants (``A``), option exercises
    (``M``) and sales (``S``) alone are not buys. Tolerates the ``xslF345X05`` rendering wrapper and
    namespaced roots by matching on local tag names, and falls back to a regex on
    ``<transactionCode>`` when the XML will not parse.
    """
    transaction_code: str | None = None
    open_market_purchase = False
    shares: float | None = None
    price_per_share: float | None = None
    reporting_owner: str | None = None

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        root = None

    if root is not None:
        # A single Form-4 can bundle several non-derivative transactions
        # (e.g. a grant 'A' followed by an open-market purchase 'P'). Scan ALL
        # transaction codes and flag the filing as an open-market purchase if ANY
        # line is P-coded; keep the first code as the representative code. Reading
        # only the first code silently dropped a P that was not first.
        codes = [
            element.text.strip()
            for element in root.iter()
            if _local_name(element.tag) == "transactionCode"
            and element.text
            and element.text.strip()
        ]
        if codes:
            transaction_code = codes[0]
            open_market_purchase = any(c == OPEN_MARKET_PURCHASE_CODE for c in codes)
        shares_el = _find_first(root, "transactionShares")
        if shares_el is not None:
            text = _first_text(shares_el)
            if text is not None:
                try:
                    shares = float(text)
                except ValueError:
                    shares = None
        price_el = _find_first(root, "transactionPricePerShare")
        if price_el is not None:
            text = _first_text(price_el)
            if text is not None:
                try:
                    price_per_share = float(text)
                except ValueError:
                    price_per_share = None
        reporting_owner = _reporting_owner_identity(root)

    if transaction_code is None:
        # Regex fallback when the XML will not parse: scan ALL <transactionCode>.
        decoded = [
            m.decode("ascii").strip()
            for m in re.findall(
                rb"<transactionCode>\s*([A-Za-z])\s*</transactionCode>", xml_bytes
            )
        ]
        if decoded:
            transaction_code = decoded[0]
            open_market_purchase = any(c == OPEN_MARKET_PURCHASE_CODE for c in decoded)

    return {
        "transaction_code": transaction_code,
        "shares": shares,
        "price_per_share": price_per_share,
        "is_open_market_purchase": open_market_purchase,
        "reporting_owner": reporting_owner,
    }


def enrich_events(
    events: list[Form4Event],
    *,
    fetch: Callable[[str], bytes] | None = None,
) -> list[Form4Event]:
    """Populate ``transaction_code`` / ``is_purchase`` by fetching each form4.xml.

    When ``cfg.net_provider == 'disabled'`` the XML cannot be fetched, so the
    events are returned unchanged (``transaction_code`` / ``is_purchase`` stay
    ``None``) and the overlay must label any signal WEAK with a "codes
    unverified" reason rather than CONFIRMED.
    """
    cfg = get_config()
    if cfg.net_provider == "disabled":
        return events

    if fetch is None:
        fetch = SecClient().download_bytes

    for event in events:
        try:
            xml_bytes = fetch(event.primary_doc_url)
        except Exception:
            continue
        if not xml_bytes:
            continue
        parsed = parse_form4_xml(xml_bytes)
        event.transaction_code = parsed["transaction_code"]
        event.is_purchase = parsed["is_open_market_purchase"]
        event.reporting_owner = parsed["reporting_owner"]

    return events
