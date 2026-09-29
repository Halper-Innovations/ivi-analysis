"""8-K triage pack builder + memo verification for the subscription analyst pass.

The daily heartbeat's triage_8k step is split three ways so the LLM never
touches the books directly: this module (deterministic, $0) selects the open
queue-protection events currently flagging watchlist names, fetches their
filing documents from EDGAR, and writes a self-contained pack under
``data/outputs/events/triage/``; a headless Claude Code run
(subscription-metered, Read/Glob/Write tools only) turns the pack into
per-event memos; ``verify_pack`` then checks every requested memo landed.
Disposal stays with the operator — each memo carries a paste-ready
``ivi events dispose`` command, the agent never writes the DB.

Fetches bypass the HTTP cache: the cache directory lives on the external
volume and this step runs from cron, where volume writes are TCC-blocked.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from app.config import get_config
from app.events import store

# A handful of filings per morning is the expected flow; the cap bounds the
# subscription spend of the agent pass, never correctness — unmemo'd events
# re-enter tomorrow's pack.
DEFAULT_LIMIT = 8
DOC_CHAR_CAP = 25_000
THESIS_CHAR_CAP = 1_500
MAX_ACCESSIONS_PER_EVENT = 3
MAX_DOCS_PER_ACCESSION = 3
MAX_DOC_BYTES = 2_000_000
# A memo below this size cannot contain the pinned sections; verify treats it
# as not written so a truncated/aborted agent run re-packs tomorrow.
MIN_MEMO_BYTES = 200

FetchBytes = Callable[[str], bytes]
FetchJson = Callable[[str], dict[str, Any]]


def triage_root() -> Path:
    return Path(get_config().outputs_dir) / "events" / "triage"


def memo_path(event_id: int, ticker: str) -> Path:
    safe_ticker = re.sub(r"[^A-Z0-9.-]", "_", str(ticker or "UNKNOWN").upper())
    return triage_root() / "memos" / f"event_{int(event_id)}_{safe_ticker}.md"


def dispose_command(event_id: int) -> str:
    return (
        f"ivi events dispose {int(event_id)} --reason-code EVENT_REVIEWED "
        '--note "<one-line factual note>"'
    )


class _TextExtractor(HTMLParser):
    """Tag stripper for EDGAR documents; script/style content is dropped."""

    _SKIP = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        joined = " ".join(self._chunks)
        return re.sub(r"[ \t\r\f\v]+", " ", joined).strip()


def strip_html(raw: bytes) -> str:
    parser = _TextExtractor()
    parser.feed(raw.decode("utf-8", errors="replace"))
    return parser.text()


def _normalize_cik(value: Any) -> str:
    return str(value or "").strip().lstrip("0")


@dataclass(frozen=True)
class _WatchRow:
    row: dict[str, Any]

    @property
    def blocking(self) -> bool:
        # DEPLOY_READY/BUY_CONFIRMED rows are the ones the flag actually holds
        # back from at-target presentation; they triage first.
        return str(self.row.get("status") or "") in {"DEPLOY_READY", "BUY_CONFIRMED"}


def _current_watchlist_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    from app.watchlist.store import current_watchlist_cte

    rows = conn.execute(
        f"""
        WITH latest_watchlist AS ({current_watchlist_cte()})
        SELECT w.*
        FROM watchlist w
        JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
        WHERE w.status != 'REMOVED' AND w.event_pending IS NOT NULL
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _latest_price_for_ticker(
    conn: sqlite3.Connection, ticker: str
) -> tuple[float | None, str | None]:
    # Snapshots may live on a superseded row of the same ticker (dup-ticker
    # watchlists): join through ticker, never through the authoritative id.
    row = conn.execute(
        """
        SELECT p.price, p.checked_at
        FROM watchlist_price_snapshots p
        JOIN watchlist w ON w.id = p.watchlist_id
        WHERE UPPER(w.ticker) = ?
        ORDER BY p.checked_at DESC, p.id DESC
        LIMIT 1
        """,
        (ticker.upper(),),
    ).fetchone()
    if row is None or not isinstance(row["price"], (int, float)):
        return None, None
    return float(row["price"]), str(row["checked_at"] or "") or None


def _parse_falsifiers(raw: Any) -> list[str]:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
    except ValueError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()][:6]
    return []


def _watchlist_context(conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    ticker = str(row.get("ticker") or "").upper()
    latest_price, checked_at = _latest_price_for_ticker(conn, ticker)
    if latest_price is None and isinstance(row.get("current_price_at_addition"), (int, float)):
        latest_price = float(row["current_price_at_addition"])
        checked_at = "addition-time price (no snapshot)"
    buy_target = row.get("buy_price_target")
    distance_pct = None
    if latest_price is not None and isinstance(buy_target, (int, float)) and buy_target:
        distance_pct = round((latest_price - float(buy_target)) / float(buy_target) * 100.0, 1)
    thesis = str(row.get("thesis_text") or "").strip()
    if len(thesis) > THESIS_CHAR_CAP:
        thesis = thesis[:THESIS_CHAR_CAP] + " …[truncated]"
    return {
        "ticker": ticker,
        "status": row.get("status"),
        "event_pending": row.get("event_pending"),
        "conviction_grade": row.get("conviction_grade"),
        "confidence": row.get("confidence"),
        "buy_price_target": buy_target,
        "latest_price": latest_price,
        "price_checked_at": checked_at,
        "distance_from_buy_pct": distance_pct,
        "cap_band": row.get("cap_band") or "UNKNOWN_CAP",
        "source_sector": row.get("source_sector"),
        "thesis_excerpt": thesis or None,
        "falsifiers": _parse_falsifiers(row.get("falsifiers_json")),
    }


def select_events(
    conn: sqlite3.Connection, *, limit: int = DEFAULT_LIMIT
) -> tuple[list[dict[str, Any]], int]:
    """Open queue-protection events on flagged watchlist names, memo-less first.

    Selection mirrors ``sync_event_pending_flags``: CIK match primary, ticker
    match fallback, so an unresolved ticker can't hide an event from triage.
    Returns (selected, skipped_existing_memo_count); gate-blocking names
    (DEPLOY_READY/BUY_CONFIRMED) lead, then newest detections.
    """
    watch_rows = _current_watchlist_rows(conn)
    if not watch_rows:
        return [], 0
    from app.events.flags import _watchlist_ticker_ciks

    by_ticker = {str(r["ticker"]).upper(): r for r in watch_rows}
    ticker_ciks = _watchlist_ticker_ciks([str(r["ticker"]) for r in watch_rows])
    by_cik = {cik: by_ticker[ticker] for ticker, cik in ticker_ciks.items() if ticker in by_ticker}

    grouped = store.open_queue_protection_events(conn)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_ids: set[int] = set()
    for rows in grouped.values():
        for event in rows:
            event_dict = dict(event)
            event_id = int(event_dict["id"])
            if event_id in seen_ids:
                continue
            event_ticker = str(event_dict.get("ticker") or "").upper()
            watch = by_cik.get(_normalize_cik(event_dict.get("cik")))
            if watch is None and event_ticker:
                watch = by_ticker.get(event_ticker)
            if watch is None:
                continue
            seen_ids.add(event_id)
            candidates.append((event_dict, watch))

    # Newest detections first, then a stable pass so gate-blocking names lead.
    candidates.sort(
        key=lambda pair: (str(pair[0].get("detection_date") or ""), int(pair[0]["id"])),
        reverse=True,
    )
    candidates.sort(key=lambda pair: 0 if _WatchRow(pair[1]).blocking else 1)

    selected: list[dict[str, Any]] = []
    skipped = 0
    for event_dict, watch in candidates:
        target = memo_path(int(event_dict["id"]), str(watch.get("ticker") or ""))
        if target.exists():
            skipped += 1
            continue
        selected.append({"event": event_dict, "watch": watch})
        if len(selected) >= max(1, int(limit)):
            break
    return selected, skipped


def _event_accessions(conn: sqlite3.Connection, event: dict[str, Any]) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT accession, form_type, filing_date
        FROM corporate_event_filings
        WHERE event_id = ?
        ORDER BY filing_date DESC, id DESC
        """,
        (int(event["id"]),),
    ).fetchall()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    anchor = str(event.get("anchor_accession") or "").strip()
    if anchor:
        out.append({"accession": anchor, "form_type": None, "filing_date": None})
        seen.add(anchor)
    for row in rows:
        accession = str(row["accession"]).strip()
        if accession and accession not in seen:
            seen.add(accession)
            out.append(
                {
                    "accession": accession,
                    "form_type": row["form_type"],
                    "filing_date": row["filing_date"],
                }
            )
    return out[:MAX_ACCESSIONS_PER_EVENT]


def _default_fetchers() -> tuple[FetchBytes, FetchJson]:
    from app.ingest.sec_client import SecClient

    client = SecClient()

    def fetch_bytes(url: str) -> bytes:
        return client.http.get_bytes(url, use_cache=False, cache_ttl_seconds=None)

    def fetch_json(url: str) -> dict[str, Any]:
        return client.http.get_json(url, use_cache=False, cache_ttl_seconds=None)

    return fetch_bytes, fetch_json


def _doc_names_from_index(index_payload: dict[str, Any]) -> list[str]:
    items = (index_payload.get("directory") or {}).get("item") or []
    htm: list[tuple[int, str]] = []
    for item in items:
        name = str(item.get("name") or "")
        lowered = name.lower()
        if not lowered.endswith((".htm", ".html")):
            continue
        if "-index" in lowered:
            continue
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        if size > MAX_DOC_BYTES:
            continue
        # Primary document first, exhibits after: exhibits are named ex*/e99*.
        is_exhibit = lowered.startswith(("ex", "e99")) or "ex99" in lowered
        htm.append((1 if is_exhibit else 0, name))
    htm.sort()
    return [name for _, name in htm[:MAX_DOCS_PER_ACCESSION]]


def _fetch_accession_docs(
    *,
    cik: str,
    accession: str,
    docs_dir: Path,
    event_id: int,
    fetch_bytes: FetchBytes,
    fetch_json: FetchJson,
) -> tuple[list[str], str | None]:
    """Fetch, strip, and persist an accession's documents; (relpaths, error)."""
    try:
        cik_int = int(_normalize_cik(cik))
    except ValueError:
        return [], "bad_cik"
    nodash = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{nodash}/"
    written: list[str] = []
    try:
        names = _doc_names_from_index(fetch_json(base + "index.json"))
    except Exception:  # noqa: BLE001 - degrade to the full-submission text
        names = [f"{accession}.txt"]
    for name in names:
        try:
            raw = fetch_bytes(base + name)
        except Exception:  # noqa: BLE001 - a missing doc must not sink the pack
            continue
        text = strip_html(raw)[:DOC_CHAR_CAP]
        if not text:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        out_path = docs_dir / f"event_{event_id}_{nodash}_{safe_name}.txt"
        out_path.write_text(text, encoding="utf-8")
        written.append(str(out_path))
    return written, None if written else "no_documents_fetched"


def build_pack(
    *,
    limit: int = DEFAULT_LIMIT,
    now: datetime | None = None,
    fetch_bytes: FetchBytes | None = None,
    fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    """Build today's triage pack; returns {pack_path, events, skipped_existing}."""
    effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    from app.db import get_db

    with get_db() as conn:
        selected, skipped = select_events(conn, limit=limit)
        if not selected:
            return {"pack_path": None, "events": 0, "skipped_existing": skipped}
        if fetch_bytes is None or fetch_json is None:
            default_bytes, default_json = _default_fetchers()
            fetch_bytes = fetch_bytes or default_bytes
            fetch_json = fetch_json or default_json

        pack_dir = triage_root() / "packs" / effective_now.date().isoformat()
        docs_dir = pack_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        memo_dir = triage_root() / "memos"
        memo_dir.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        for item in selected:
            event = item["event"]
            watch = item["watch"]
            event_id = int(event["id"])
            ticker = str(watch.get("ticker") or event.get("ticker") or "UNKNOWN")
            filings: list[dict[str, Any]] = []
            for accession_row in _event_accessions(conn, event):
                docs, error = _fetch_accession_docs(
                    cik=str(event.get("cik") or ""),
                    accession=accession_row["accession"],
                    docs_dir=docs_dir,
                    event_id=event_id,
                    fetch_bytes=fetch_bytes,
                    fetch_json=fetch_json,
                )
                filings.append(
                    {
                        "accession": accession_row["accession"],
                        "form_type": accession_row["form_type"],
                        "filing_date": accession_row["filing_date"],
                        "docs": docs,
                        "fetch_error": error,
                    }
                )
            entries.append(
                {
                    "event_id": event_id,
                    "ticker": ticker.upper(),
                    "event_type": event.get("event_type"),
                    "flag": store.event_pending_flag(str(event.get("event_type"))),
                    "event_status": event.get("status"),
                    "detection_date": event.get("detection_date"),
                    "company_name": event.get("company_name"),
                    "watchlist": _watchlist_context(conn, watch),
                    "filings": filings,
                    "memo_path": str(memo_path(event_id, ticker)),
                    "dispose_command": dispose_command(event_id),
                }
            )

    pack = {
        "generated_at": effective_now.isoformat(),
        "instructions_version": 1,
        "memo_dir": str(memo_dir),
        "events": entries,
    }
    pack_path = pack_dir / "pack.json"
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    return {
        "pack_path": str(pack_path),
        "events": len(entries),
        "skipped_existing": skipped,
    }


def verify_pack(pack_path: str | Path) -> dict[str, Any]:
    """Check every memo the pack requested exists and is non-trivial."""
    payload = json.loads(Path(pack_path).read_text(encoding="utf-8"))
    events = payload.get("events") or []
    missing: list[dict[str, Any]] = []
    written = 0
    for entry in events:
        target = Path(str(entry.get("memo_path") or ""))
        if target.is_file() and target.stat().st_size >= MIN_MEMO_BYTES:
            written += 1
        else:
            missing.append({"event_id": entry.get("event_id"), "ticker": entry.get("ticker")})
    return {"expected": len(events), "written": written, "missing": missing}
