"""Today read model: the morning-truth deck over a read-only connection.

Four sections, each citing its book of record:

- **decisions** — OPEN ``dispositions`` rows joined to their watchlist rows,
  carrying the falsifiers and pre-mortem the UI uses as friction before it
  hands over the journal command.
- **waterline** — the queue's presentable rows nearest the buy zone on
  either side (by ``abs(distance)``), each with a 30-day price wake, plus a
  summary of the deeper names the strip cuts off.
- **health** — :func:`app.ops.data_health.compute_data_health` serialized
  verbatim (this module never reinterprets a check).
- **digest** — the latest daily digest markdown rendered server-side
  (raw HTML escaped; digests are our own artifacts but the Reader never
  trusts artifact HTML).
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    authorized_artifact_bytes,
    financial_integrity_manifest_is_usable,
)
from app.config import get_config
from app.ops.data_health import DataHealth
from app.web.readmodel.db import table_exists
from app.web.readmodel.md import render_markdown

_CONVICTION_ORDER = {"ACTIONABLE": 0, "WATCHLIST_ONLY": 1, "DATA_INCOMPLETE": 2, "AVOID": 3}

# Presented statuses that may float on the waterline strip. QUARANTINE is the
# gate's shelf, and a PRICE_DATA_SUSPECT distance is not a real distance.
_WATERLINE_STATUSES = {
    "DEPLOY_READY",
    "BUY_CONFIRMED",
    "EVENT_PENDING",
    "ACTIVE",
    "UNCERTAIN",
}

_DIGEST_PATTERN = re.compile(r"^digest_(\d{4}-\d{2}-\d{2})\.md$")


def price_age_hours(checked_at: str | None, *, now: datetime | None = None) -> float | None:
    """Hours since an ISO timestamp; None when absent or unparseable."""
    if not checked_at:
        return None
    try:
        checked = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if checked.tzinfo is None:
        checked = checked.replace(tzinfo=timezone.utc)
    effective_now = now or datetime.now(timezone.utc)
    return round((effective_now - checked).total_seconds() / 3600.0, 2)


def serialize_health(health: DataHealth) -> dict[str, Any]:
    return {
        "state": health.state,
        "blocking": bool(health.blocking),
        "checks": [
            {"name": str(c["name"]), "ok": bool(c["ok"]), "detail": str(c["detail"])}
            for c in health.checks
        ],
    }


def _ticker_latest_snapshots(
    conn: sqlite3.Connection, tickers: list[str]
) -> dict[str, tuple[float, str]]:
    """Newest price snapshot per ticker across every watchlist row of that ticker.

    Duplicate-ticker rows exist and snapshots do not always sit on the row the queue
    presents, so a per-row join shows a frozen price on the presented row while the
    live one sits on its twin. A company's price is one series (as the company page
    already treats it); this is that series' newest point.
    """
    if not tickers or not table_exists(conn, "watchlist_price_snapshots"):
        return {}
    placeholders = ",".join("?" for _ in tickers)
    rows = conn.execute(
        f"""
        SELECT w.ticker, s.price, s.checked_at
        FROM watchlist_price_snapshots s
        JOIN watchlist w ON w.id = s.watchlist_id
        WHERE w.ticker IN ({placeholders}) AND s.price IS NOT NULL
        ORDER BY w.ticker, s.checked_at ASC, s.id ASC
        """,
        tickers,
    ).fetchall()
    newest: dict[str, tuple[float, str]] = {}
    for row in rows:  # ascending, so the last write per ticker is the newest
        newest[str(row["ticker"])] = (float(row["price"]), str(row["checked_at"] or ""))
    return newest


def _watchlist_context(
    conn: sqlite3.Connection, watchlist_ids: list[int]
) -> dict[int, dict[str, Any]]:
    """Falsifiers, targets, and latest snapshots for a set of watchlist rows."""
    if not watchlist_ids:
        return {}
    placeholders = ",".join("?" for _ in watchlist_ids)
    rows = conn.execute(
        f"""
        SELECT
            w.id, w.falsifiers_json, w.confidence, w.source_sector,
            w.buy_price_target,
            snap.price AS latest_price, snap.checked_at AS latest_price_checked_at
        FROM watchlist w
        LEFT JOIN (
            SELECT watchlist_id, price, checked_at
            FROM (
                SELECT
                    s.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.watchlist_id
                        ORDER BY s.checked_at DESC, s.id DESC
                    ) AS snapshot_rank
                FROM watchlist_price_snapshots s
            )
            WHERE snapshot_rank = 1
        ) snap ON snap.watchlist_id = w.id
        WHERE w.id IN ({placeholders})
        """,
        watchlist_ids,
    ).fetchall()
    context: dict[int, dict[str, Any]] = {}
    for row in rows:
        falsifiers: list[str] = []
        try:
            parsed = json.loads(row["falsifiers_json"] or "[]")
            if isinstance(parsed, list):
                falsifiers = [str(item) for item in parsed]
        except ValueError:
            falsifiers = []
        context[int(row["id"])] = {
            "falsifiers": falsifiers,
            "confidence": row["confidence"],
            "source_sector": row["source_sector"],
            "buy_price_target": (
                float(row["buy_price_target"]) if row["buy_price_target"] is not None else None
            ),
            "latest_price": (
                float(row["latest_price"]) if row["latest_price"] is not None else None
            ),
            "latest_price_checked_at": row["latest_price_checked_at"],
        }
    return context


def open_decisions(
    conn: sqlite3.Connection, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """OPEN dispositions via the store's shared derivation, presentation-enriched.

    ``open_dispositions`` supplies the platform's own currency judgment
    (``source_state`` / ``is_current_actionable``); this layer only adds the
    falsifiers, latest price, and journal command the deck renders.
    """
    if not financial_integrity_manifest_is_usable():
        return []
    if not table_exists(conn, "watchlist"):
        return []

    from app.watchlist.dispositions import open_dispositions

    payloads = open_dispositions(conn=conn)
    ids = [
        int(p["current_watchlist_id"])
        for p in payloads
        if p.get("current_watchlist_id") is not None
    ]
    context = _watchlist_context(conn, sorted(set(ids)))
    by_ticker = _ticker_latest_snapshots(conn, sorted({str(p["ticker"]) for p in payloads}))

    decisions: list[dict[str, Any]] = []
    for payload in payloads:
        ticker = str(payload["ticker"])
        wid = payload.get("current_watchlist_id")
        ctx = dict(context.get(int(wid), {})) if wid is not None else {}
        twin = by_ticker.get(ticker)
        if twin is not None and twin[1] > str(ctx.get("latest_price_checked_at") or ""):
            ctx["latest_price"], ctx["latest_price_checked_at"] = twin
        price = ctx.get("latest_price")
        target = ctx.get("buy_price_target")
        distance = None
        if price is not None and target and target > 0:
            distance = ((price - target) / target) * 100.0
        decisions.append(
            {
                "id": int(payload["id"]),
                "ticker": ticker,
                "kind": str(payload["kind"]),
                "opened_at": payload.get("opened_at"),
                "opened_by": payload.get("opened_by"),
                "trigger": payload.get("trigger_snapshot") or {},
                "pre_mortem": payload.get("pre_mortem"),
                "watchlist_id": int(wid) if wid is not None else None,
                "watchlist_status": payload.get("current_watchlist_status"),
                "conviction_grade": payload.get("current_conviction_grade"),
                "confidence": ctx.get("confidence"),
                "source_sector": ctx.get("source_sector"),
                "source_state": payload.get("source_state"),
                "is_current_actionable": bool(payload.get("is_current_actionable")),
                "buy_price_target": target,
                "latest_price": price,
                "latest_price_checked_at": ctx.get("latest_price_checked_at"),
                "price_age_hours": price_age_hours(ctx.get("latest_price_checked_at"), now=now),
                "distance_from_buy_pct": distance,
                "falsifiers": ctx.get("falsifiers", []),
                "journal_command": (
                    f"ivi investor journal {ticker} --disposition-id {int(payload['id'])} "
                    '--action acted|passed|deferred --reason <CODE> --rationale "<why>"'
                ),
            }
        )

    decisions.sort(
        key=lambda d: (
            0 if d["is_current_actionable"] else 1,
            _CONVICTION_ORDER.get(str(d["conviction_grade"] or ""), 9),
            d["distance_from_buy_pct"] if d["distance_from_buy_pct"] is not None else 1e9,
            str(d["opened_at"] or ""),
        )
    )
    return decisions


def attach_wakes(
    conn: sqlite3.Connection,
    items: list[dict[str, Any]],
    *,
    days: int = 30,
    now: datetime | None = None,
) -> None:
    """Attach a ``wake`` (oldest → newest closes) per ticker.

    Snapshots attach to watchlist rows, but dup-ticker rows exist and the
    authoritative row is not always the snapshotted one — a company's price
    history is one series, so the wake joins through the ticker.
    """
    tickers = sorted({str(item["ticker"]) for item in items if item.get("ticker")})
    wakes: dict[str, list[float]] = {}
    if tickers:
        effective_now = now or datetime.now(timezone.utc)
        cutoff = (effective_now.replace(microsecond=0) - timedelta(days=days)).isoformat()
        placeholders = ",".join("?" for _ in tickers)
        rows = conn.execute(
            f"""
            SELECT w.ticker, s.price
            FROM watchlist_price_snapshots s
            JOIN watchlist w ON w.id = s.watchlist_id
            WHERE w.ticker IN ({placeholders}) AND s.checked_at >= ?
            ORDER BY w.ticker, s.checked_at ASC, s.id ASC
            """,
            [*tickers, cutoff],
        ).fetchall()
        for row in rows:
            wakes.setdefault(str(row["ticker"]), []).append(float(row["price"]))
    for item in items:
        item["wake"] = wakes.get(str(item.get("ticker")), [])


def _with_live_price(row: dict[str, Any], twin: tuple[float, str] | None) -> dict[str, Any]:
    """The queue row with the newest snapshot of its ticker, when that beats its own."""
    if twin is None:
        return row
    price, checked_at = twin
    if row.get("latest_price_basis") == "SNAPSHOT" and checked_at <= str(
        row.get("latest_price_checked_at") or ""
    ):
        return row
    if row.get("latest_price_basis") is None and row.get("latest_price") is not None:
        return row  # a row that does not say where its price came from is left alone
    merged = dict(row)
    merged.update(latest_price=price, latest_price_checked_at=checked_at)
    merged["latest_price_basis"] = "SNAPSHOT"
    target = merged.get("buy_price_target")
    if target and float(target) > 0:
        merged["distance_from_buy_pct"] = ((price - float(target)) / float(target)) * 100.0
    return merged


def waterline(
    conn: sqlite3.Connection,
    queue: list[dict[str, Any]],
    *,
    limit: int = 14,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Names closest to the waterline on either side, with wakes.

    Sorted by ``abs(distance_from_buy_pct)`` — the strip shows the water
    around the line (names about to submerge and names just under it), not
    the deepest names; those are summarized in the returned ``deeper``
    block: ``{"count": int, "names": [{ticker, distance_from_buy_pct}]}``
    (the three deepest, for the strip's seabed footer).
    """
    candidates = [
        row for row in queue if str(row.get("presented_status") or "") in _WATERLINE_STATUSES
    ]
    by_ticker = _ticker_latest_snapshots(conn, sorted({str(r["ticker"]) for r in candidates}))
    eligible = []
    for row in candidates:
        row = _with_live_price(row, by_ticker.get(str(row["ticker"])))
        # A price recorded when the name was added is not the price today; the strip
        # is a live-distance instrument, so such a row is left off it, not drawn.
        if row.get("latest_price_basis") == "PRICE_AT_ADDITION":
            continue
        if row.get("distance_from_buy_pct") is not None:
            eligible.append(row)
    # Stable sort: ties keep the queue's own conviction-ranked order.
    eligible.sort(key=lambda r: abs(float(r["distance_from_buy_pct"])))
    items = [
        {
            "watchlist_id": int(row["id"]),
            "ticker": row["ticker"],
            "presented_status": row["presented_status"],
            "conviction_grade": row["conviction_grade"],
            "confidence": row["confidence"],
            "latest_price": row["latest_price"],
            "latest_price_checked_at": row["latest_price_checked_at"],
            "price_age_hours": price_age_hours(row["latest_price_checked_at"], now=now),
            "price_suspect": str(row.get("status") or "") == "PRICE_DATA_SUSPECT",
            "buy_price_target": row["buy_price_target"],
            "distance_from_buy_pct": row["distance_from_buy_pct"],
            "valuation_anchor_method": row.get("valuation_anchor_method"),
            "valuation_anchor_value": row.get("valuation_anchor_value"),
            "source_sector": row["source_sector"],
        }
        for row in eligible[:limit]
    ]
    attach_wakes(conn, items, now=now)

    below = [row for row in eligible[limit:] if float(row["distance_from_buy_pct"]) < 0]
    below.sort(key=lambda r: float(r["distance_from_buy_pct"]))
    deeper = {
        "count": len(below),
        "names": [
            {
                "ticker": row["ticker"],
                "distance_from_buy_pct": row["distance_from_buy_pct"],
            }
            for row in below[:3]
        ],
    }
    return items, deeper


def gate_blocked(conn: sqlite3.Connection, queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows whose DEPLOY_READY presentation is blocked by an open event flag.

    The price is the newest snapshot of the ticker, as on the waterline. A price
    recorded when the name was added is not the price today, so a row left with
    only that shows no price and no distance rather than presenting it as live.
    """
    blocked = [row for row in queue if str(row.get("presented_status") or "") == "EVENT_PENDING"]
    by_ticker = _ticker_latest_snapshots(conn, sorted({str(r["ticker"]) for r in blocked}))
    items = []
    for row in blocked:
        row = _with_live_price(row, by_ticker.get(str(row["ticker"])))
        at_addition = row.get("latest_price_basis") == "PRICE_AT_ADDITION"
        items.append(
            {
                "watchlist_id": int(row["id"]),
                "ticker": row["ticker"],
                "event_pending": row["event_pending"],
                "conviction_grade": row["conviction_grade"],
                "distance_from_buy_pct": None if at_addition else row["distance_from_buy_pct"],
                "latest_price": None if at_addition else row["latest_price"],
                "buy_price_target": row["buy_price_target"],
            }
        )
    return items


def latest_digest() -> dict[str, Any] | None:
    """The newest digest rendered to HTML, plus the previous digest's date."""
    digests_dir = Path(get_config().outputs_dir) / "digests"
    if not digests_dir.is_dir():
        return None
    dated: list[tuple[str, Path]] = []
    for path in digests_dir.iterdir():
        match = _DIGEST_PATTERN.match(path.name)
        if match:
            dated.append((match.group(1), path))
    if not dated:
        return None
    dated.sort(key=lambda pair: pair[0])
    date, path = dated[-1]
    previous_date = dated[-2][0] if len(dated) >= 2 else None
    integrity_status, digest_bytes = authorized_artifact_bytes(path)
    if integrity_status != FINANCIAL_INTEGRITY_PASS or digest_bytes is None:
        return None
    try:
        text = digest_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return {
        "date": date,
        "previous_date": previous_date,
        "path": str(path),
        "html": render_markdown(text),
    }
