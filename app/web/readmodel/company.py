"""Company read model: masthead, valuation cards, gauge shelves, fundamentals.

Everything is keyed by ticker over the read-only connection. Valuation
numbers come from ``valuations`` (latest row per method; multiple
``as_of_date`` runs exist per ticker+method) with ``valuations_history``
unioned into the evolution series. Fundamentals come from
``companyfacts_facts`` (FY basis, values in USD millions) and from the
quarterly-fact TTM assembly — TTM is computed from discrete quarters, never
faked from FY rows.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.autonomous.artifact_financial_audit import (
    financial_integrity_manifest_is_usable,
)
from app.util.financial_data_access import issuer_companyfacts_rows
from app.valuation.anchor_policy import published_dcf_base
from app.valuation.lineage import valuation_row_is_decision_eligible
from app.watchlist.lineage import watchlist_row_is_decision_eligible
from app.web.readmodel.db import OfflineError, table_exists
from app.web.readmodel.today import price_age_hours

# The valuation-writer hardening landed 2026-07-16 (the valuation-writer hardening);
# estimate-evolution points written before it are labeled in the UI.
PRE_HARDENING_CUTOFF = "2026-07-16"

# Methods whose per-share output belongs on the Depth Gauge.
_SHELF_METHODS = ("dcf", "epv", "graham")

_SHELF_LABELS = {"dcf": "DCF", "epv": "EPV", "graham": "Graham"}

# Internal/meta rows that are not analyst-facing valuation cards.
_EXCLUDED_METHODS = {"test_method"}

# Per-method scalar outputs worth surfacing on the card, beyond fair value.
_METHOD_EXTRAS: dict[str, tuple[str, ...]] = {
    "graham": ("buy_price", "normalized_eps", "bvps"),
    "owner_earnings": ("owner_earnings_latest", "cfo_used", "normalized_capex", "sbc_used"),
    "roic": ("roic_proxy", "roic_wacc_ratio", "signal"),
    "reverse_dcf": ("feasibility", "revenue_cagr_5y_used"),
    "capital_structure": (
        "interest_coverage",
        "de_ratio",
        "net_debt_to_ebitda",
        "interest_coverage_adequacy",
    ),
    "ncav": ("signal",),
    "tangible_floor": (),
    "ev_ebit": (),
    "fcf_yield": (),
    "scorecard": ("signal", "type"),
    "epv": ("avg_operating_income",),
}

# FY line items served to the fundamentals grid (values are USD millions).
FY_LINE_ITEMS = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "cfo",
    "capex",
    "sbc",
    "shares_outstanding",
    "total_debt",
    "cash",
    "equity",
    "total_assets",
)


def _read_json_dict(raw: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_json_list(raw: str | None) -> list[Any]:
    try:
        parsed = json.loads(raw or "[]")
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def _not_a_price(value: float, outputs: dict[str, Any]) -> dict[str, Any]:
    """A zero or negative per-share value is a finding about the business, not a price.

    A negative earnings-power value used to be drawn as a line below the zero of the
    gauge, as though the stock were worth minus dollars. The card keeps the figure; the
    method's own status (already on the card) says why there is no shelf.
    """
    return {
        "kind": "not_a_price",
        "value": float(value),
    }


def _fair_value(method: str, outputs: dict[str, Any]) -> dict[str, Any]:
    """Normalize a method's fair-value output (per-share dollars)."""
    if method == "dcf" or method == "dcf_adjusted":
        low, base, high = outputs.get("low"), outputs.get("base"), outputs.get("high")
        if isinstance(base, (int, float)) and base <= 0:
            return _not_a_price(base, outputs)
        if isinstance(base, (int, float)):
            return {
                "kind": "band",
                "low": float(low) if isinstance(low, (int, float)) else None,
                "base": float(base),
                "high": float(high) if isinstance(high, (int, float)) else None,
            }
        return {"kind": "none"}
    value = outputs.get("value_per_share")
    if isinstance(value, (int, float)) and value <= 0:
        return _not_a_price(value, outputs)
    if isinstance(value, (int, float)):
        return {"kind": "line", "value": float(value)}
    return {"kind": "none"}


def _wacc(outputs: dict[str, Any]) -> dict[str, Any] | None:
    detail = outputs.get("wacc_detail")
    if not isinstance(detail, dict):
        return None
    adjustments = [
        {
            "code": str(adj.get("code")),
            "delta": adj.get("delta"),
            "reason": str(adj.get("reason") or ""),
        }
        for adj in detail.get("adjustments") or []
        if isinstance(adj, dict)
    ]
    return {
        "baseline_wacc": detail.get("baseline_wacc"),
        "adjusted_wacc": detail.get("adjusted_wacc"),
        "adjustments": adjustments,
    }


def method_card(row: sqlite3.Row) -> dict[str, Any]:
    method = str(row["method"])
    outputs = _read_json_dict(row["outputs_json"])
    extras: dict[str, Any] = {}
    for key in _METHOD_EXTRAS.get(method, ()):
        if outputs.get(key) is not None:
            extras[key] = outputs[key]
    return {
        "method": method,
        "as_of_date": row["as_of_date"],
        "created_at": row["created_at"],
        "status": outputs.get("status"),
        "fair_value": _fair_value(method, outputs),
        "flags": [str(f) for f in outputs.get("flags") or [] if isinstance(f, str)],
        "wacc": _wacc(outputs),
        "quality_gate_verdict": row["quality_gate_verdict"],
        "confidence_class": row["confidence_class"],
        "gate_reason_codes": [
            str(c) for c in _read_json_list(row["gate_reason_codes"]) if isinstance(c, str)
        ],
        "headwinds": [
            str(c) for c in _read_json_list(row["valuation_headwinds"]) if isinstance(c, str)
        ],
        "supports": [
            str(c) for c in _read_json_list(row["valuation_supports"]) if isinstance(c, str)
        ],
        "extras": extras,
    }


def _drop_rows_superseded_by_a_block(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """Method values older than a BLOCKED verdict are not current.

    The newest row is taken per METHOD, which is right for a single-method
    re-run but wrong after a block: the writer's BLOCK path persists only a
    scorecard row and returns, so the page paired today's "BLOCKED" verdict
    with the previous run's DCF, EPV and Graham cards and drew them on the
    gauge as current values. The dossier surface for the same rows prints
    "GATE BLOCKED" and no values; this is the page agreeing with it.
    """
    blocked_as_of: str | None = None
    blocked_created: str | None = None
    for row in rows:
        if str(row["method"]) != "scorecard":
            continue
        verdict = str(row["quality_gate_verdict"] or "").upper()
        signal = str(_read_json_dict(row["outputs_json"]).get("signal") or "").upper()
        if verdict == "BLOCK" or signal == "VALUATION_BLOCKED":
            blocked_as_of = str(row["as_of_date"])
            blocked_created = str(row["created_at"] or "") or None
        break
    if blocked_as_of is None:
        return list(rows)

    def _current(row: sqlite3.Row) -> bool:
        as_of = str(row["as_of_date"])
        if as_of != blocked_as_of:
            return as_of > blocked_as_of
        # Same day: the block only supersedes rows written before it. Rows with no
        # timestamp cannot be ordered, so they are kept as before.
        created = str(row["created_at"] or "")
        if str(row["method"]) == "scorecard" or not created or blocked_created is None:
            return True
        return created >= blocked_created

    return [row for row in rows if _current(row)]


def _latest_valuation_rows(
    conn: sqlite3.Connection, ticker: str, *, as_of_date: str | None = None
) -> list[sqlite3.Row]:
    """Select the newest row per method before applying authorization."""

    if as_of_date is None:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT
                    v.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY v.method
                        ORDER BY v.as_of_date DESC, v.created_at DESC, v.id DESC
                    ) AS method_rank
                FROM valuations v
                WHERE v.ticker = ?
            )
            WHERE method_rank = 1
            ORDER BY method
            """,
            (ticker,),
        ).fetchall()
        rows = _drop_rows_superseded_by_a_block(rows)
    else:
        rows = conn.execute(
            """
            WITH candidates AS (
                SELECT
                    h.id, h.ticker, h.as_of_date, h.method, h.inputs_json,
                    h.outputs_json, h.warnings_json,
                    h.created_at, h.quality_gate_verdict, h.confidence_class,
                    h.gate_reason_codes, h.valuation_headwinds,
                    h.valuation_supports, h.valuation_writer_version,
                    h.source_run_id, h.source_artifact_path,
                    h.source_artifact_sha256,
                    h.financial_integrity_fingerprint, 0 AS is_live
                FROM valuations_history h
                WHERE h.ticker = :ticker
                  AND date(COALESCE(NULLIF(h.created_at, ''), h.as_of_date))
                      <= date(:as_of_date)
                  AND date(h.archived_at) > date(:as_of_date)
                UNION ALL
                SELECT
                    v.id, v.ticker, v.as_of_date, v.method, v.inputs_json,
                    v.outputs_json, v.warnings_json,
                    v.created_at, v.quality_gate_verdict, v.confidence_class,
                    v.gate_reason_codes, v.valuation_headwinds,
                    v.valuation_supports, v.valuation_writer_version,
                    v.source_run_id, v.source_artifact_path,
                    v.source_artifact_sha256,
                    v.financial_integrity_fingerprint, 1 AS is_live
                FROM valuations v
                WHERE v.ticker = :ticker
                  AND date(COALESCE(NULLIF(v.created_at, ''), v.as_of_date))
                      <= date(:as_of_date)
            ), ranked AS (
                SELECT
                    candidates.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY method
                        ORDER BY as_of_date DESC, is_live DESC, created_at DESC, id DESC
                    ) AS method_rank
                FROM candidates
            )
            SELECT *
            FROM ranked
            WHERE method_rank = 1
            ORDER BY method
            """,
            {"ticker": ticker, "as_of_date": as_of_date},
        ).fetchall()
        # A block known by the as-of date supersedes older method rows there
        # too, exactly as it does on the live page.
        rows = _drop_rows_superseded_by_a_block(rows)
    return list(rows)


def _pricing_zone_detail(rows: list[sqlite3.Row]) -> dict[str, Any]:
    for row in rows:
        if str(row["method"]) != "scorecard":
            continue
        detail = _read_json_dict(row["outputs_json"]).get("pricing_zone_detail")
        if isinstance(detail, dict):
            return detail
    return {}


def _apply_durable_dcf(card: dict[str, Any], pricing_zone_detail: dict[str, Any]) -> None:
    """Show the DCF the writer stands behind, not the one it rejected.

    The durable (spike-corrected) base is persisted only on the scorecard, so
    the card, the gauge and the valuation-history chart all drew the raw figure
    for a name the writer had flagged as inflated and anchored elsewhere. The
    published band is dropped with it: the writer never computed durable low and
    high scenarios, and scaling the raw ones would be inventing them. The card
    becomes a line at the durable base (a band without ends drew at $0).
    """
    if card.get("method") != "dcf":
        return  # the adjusted card carries its own measured value
    fair = card.get("fair_value") or {}
    if fair.get("kind") != "band":
        return
    resolved = published_dcf_base(pricing_zone_detail, fair.get("base"))
    if resolved is None or resolved == fair.get("base"):
        return
    raw = pricing_zone_detail.get("dcf_raw_base")
    card["fair_value"] = {"kind": "line", "value": resolved}
    card.setdefault("extras", {})["dcf_raw_base"] = float(raw) if isinstance(raw, (int, float)) else None
    card["extras"]["dcf_base_basis"] = "durable_spike_corrected"


def latest_valuations(
    conn: sqlite3.Connection, ticker: str, *, as_of_date: str | None = None
) -> list[dict[str, Any]]:
    rows = _latest_valuation_rows(conn, ticker, as_of_date=as_of_date)
    pricing_zone_detail = _pricing_zone_detail(rows)
    cards = [
        method_card(row)
        for row in rows
        if str(row["method"]) not in _EXCLUDED_METHODS and valuation_row_is_decision_eligible(row)
    ]
    for card in cards:
        # Only the raw ``dcf`` row is replaced by the durable base; the adjusted
        # card carries its own measured value and keeps it.
        if card["method"] == "dcf":
            _apply_durable_dcf(card, pricing_zone_detail)
    return cards


def has_authorized_valuation(conn: sqlite3.Connection, ticker: str) -> bool:
    """Whether at least one newest-per-method valuation is exact-source PASS."""

    return any(
        valuation_row_is_decision_eligible(row)
        for row in _latest_valuation_rows(conn, ticker)
        if str(row["method"]) not in _EXCLUDED_METHODS
    )


def gauge_shelves(
    cards: list[dict[str, Any]],
    *,
    anchor_method: str | None,
    anchor_value: float | None,
) -> list[dict[str, Any]]:
    """The Depth Gauge's shelf set: DCF band + EPV/Graham lines, anchor emphasized.

    When the watchlist anchor method is not one of the shelf methods, the
    anchor value gets its own emphasized line so the number the buy target
    hangs from is always on the gauge.
    """
    normalized_anchor = str(anchor_method or "").lower()
    by_method = {card["method"]: card for card in cards}
    shelves: list[dict[str, Any]] = []
    for method in _SHELF_METHODS:
        card = by_method.get(method)
        if card is None:
            continue
        fair = card["fair_value"]
        emphasized = normalized_anchor == method
        if fair["kind"] == "band":
            shelves.append(
                {
                    "method": method,
                    "label": _SHELF_LABELS[method],
                    "low": fair["low"],
                    "base": fair["base"],
                    "high": fair["high"],
                    "emphasized": emphasized,
                }
            )
        elif fair["kind"] == "line":
            shelves.append(
                {
                    "method": method,
                    "label": _SHELF_LABELS[method],
                    "value": fair["value"],
                    "emphasized": emphasized,
                }
            )
    # An anchor that differs from its own method's card gets its own line, even
    # when the method IS a shelf method: the docstring promises the number the
    # buy target hangs from is on the gauge, and for a durable-vs-raw DCF (or an
    # anchor carried from an earlier run) the card's value is not that number.
    if anchor_value is not None and normalized_anchor in _SHELF_METHODS:
        drawn = [
            shelf.get("base") if "base" in shelf else shelf.get("value")
            for shelf in shelves
            if shelf["method"] == normalized_anchor
        ]
        if drawn and drawn[0] != anchor_value:
            for shelf in shelves:
                if shelf["method"] == normalized_anchor:
                    shelf["emphasized"] = False
            shelves.append(
                {
                    "method": normalized_anchor,
                    "label": f"{_SHELF_LABELS.get(normalized_anchor, normalized_anchor)} (anchor)",
                    "value": float(anchor_value),
                    "emphasized": True,
                }
            )
    if anchor_value is not None and normalized_anchor and normalized_anchor not in _SHELF_METHODS:
        shelves.append(
            {
                "method": normalized_anchor,
                "label": normalized_anchor,
                "value": float(anchor_value),
                "emphasized": True,
            }
        )
    return shelves


def _split_events(conn: sqlite3.Connection, ticker: str) -> dict[str, set[float]]:
    """Recorded stock splits for a ticker: effective date -> the factors quotes carry."""
    if not table_exists(conn, "price_quotes"):
        return {}
    events: dict[str, set[float]] = {}
    for row in conn.execute(
        """
        SELECT split_effective_date, split_adjustment_factor
        FROM price_quotes
        WHERE ticker = ? AND status = 'OK'
          AND split_effective_date IS NOT NULL AND split_effective_date != ''
          AND split_adjustment_factor IS NOT NULL AND split_adjustment_factor != 1
        """,
        (ticker,),
    ).fetchall():
        factor = float(row["split_adjustment_factor"])
        events.setdefault(str(row["split_effective_date"])[:10], set()).add(
            round(factor, 6) if factor > 0 else factor
        )
    return events


def _rebase_factor(
    splits: dict[str, set[float]], basis_day: str, until: str | None = None
) -> float | None:
    """Cumulative split factor between a value's write date and ``until`` (default now).

    A historical view (``until`` = its as-of date) rebases only onto the share
    basis of that day: a later split had not happened. None if unknowable.
    """
    factor = 1.0
    for effective, factors in splits.items():
        if effective <= basis_day:
            continue
        if until is not None and effective > until[:10]:
            continue
        if len(factors) != 1 or next(iter(factors)) <= 0:
            return None
        factor *= next(iter(factors))
    return factor


def valuation_evolution(
    conn: sqlite3.Connection, ticker: str, *, as_of_date: str | None = None
) -> list[dict[str, Any]]:
    return valuation_evolution_with_drops(conn, ticker, as_of_date=as_of_date)[0]


def valuation_evolution_with_drops(
    conn: sqlite3.Connection, ticker: str, *, as_of_date: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fair value over time for the shelf methods, live + archived rows.

    Deduped by (method, as_of_date) preferring the live table; each point
    carries ``pre_hardening`` (written before the 2026-07-16 valuation
    hardening) so the chart can label the regime change.

    Per-share values written before a stock split are on the old share count. They
    are divided by the recorded split factor so the whole series sits on the
    current basis (the count is returned beside the points); a point written before a
    split whose factor is missing or contradictory is dropped, and the count of
    those is returned beside the points so the page can say so.
    """
    newest = _latest_valuation_rows(conn, ticker, as_of_date=as_of_date)
    authorized_methods = {
        str(row["method"]) for row in newest if valuation_row_is_decision_eligible(row)
    }
    # The card shows the writer's durable DCF, which lives only on the scorecard's
    # zone detail; the chart resolves it the same way, from the scorecard written
    # for the same date (the newest scorecard for the newest DCF row).
    newest_detail = _pricing_zone_detail(newest)
    newest_dcf_date = next((str(r["as_of_date"]) for r in newest if str(r["method"]) == "dcf"), None)
    details_by_date: dict[str, dict[str, Any]] = {}
    for table in ("valuations_history", "valuations"):
        for srow in conn.execute(
            f"SELECT as_of_date, outputs_json FROM {table} WHERE ticker = ? AND method = 'scorecard'",
            (ticker,),
        ).fetchall():
            detail = _read_json_dict(srow["outputs_json"]).get("pricing_zone_detail")
            if isinstance(detail, dict):
                details_by_date[str(srow["as_of_date"])] = detail
    splits = _split_events(conn, ticker)
    dropped_pre_split = 0
    rebased = 0
    points: dict[tuple[str, str], dict[str, Any]] = {}
    for table, archived in (("valuations_history", True), ("valuations", False)):
        if as_of_date is None:
            rows = conn.execute(
                f"""
                SELECT *
                FROM {table}
                WHERE ticker = ? AND method IN (?, ?, ?)
                """,
                (ticker, *_SHELF_METHODS),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT *
                FROM {table}
                WHERE ticker = ? AND method IN (?, ?, ?)
                  AND date(as_of_date) <= date(?)
                """,
                (ticker, *_SHELF_METHODS, as_of_date),
            ).fetchall()
        for row in rows:
            method = str(row["method"])
            if method not in authorized_methods or not valuation_row_is_decision_eligible(row):
                continue
            outputs = _read_json_dict(row["outputs_json"])
            raw = outputs.get("base") if method == "dcf" else outputs.get("value_per_share")
            if not isinstance(raw, (int, float)):
                continue
            if method == "dcf":
                detail = details_by_date.get(str(row["as_of_date"]))
                if detail is None and str(row["as_of_date"]) == newest_dcf_date:
                    detail = newest_detail
                resolved = published_dcf_base(detail or {}, raw)
                raw = resolved if resolved is not None else raw
            if raw <= 0:
                continue  # not a price; the card says why
            stamp = str(row["created_at"] or row["as_of_date"] or "")
            point = {
                "method": method,
                "as_of_date": str(row["as_of_date"]),
                "value": float(raw),
                "archived": archived,
                "pre_hardening": stamp < PRE_HARDENING_CUTOFF,
            }
            basis_day = (str(row["created_at"] or "") or str(row["as_of_date"] or ""))[:10]
            factor = _rebase_factor(splits, basis_day, until=as_of_date)
            if factor is None:
                dropped_pre_split += 1
                continue
            if factor != 1.0:
                point["value"] = float(raw) / factor
                rebased += 1
            points[(method, str(row["as_of_date"]))] = point
    ordered = sorted(points.values(), key=lambda p: (p["method"], p["as_of_date"]))
    return ordered, {"dropped_pre_split": dropped_pre_split, "rebased": rebased}


def _price_from_valuation_inputs(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    now: datetime | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Header price for a ticker with no watchlist snapshot: the quote a valuation used.

    Valuation rows record the price they were computed against in ``inputs_json``
    (``current_price``, dated by ``price_as_of_date``). For a name that was never on
    the watchlist that is the only quote there is, so the header shows it, labelled
    with its own date and source instead of going blank.
    """
    empty = {"latest": None, "checked_at": None, "source": None, "age_hours": None, "wake": []}
    sql = "SELECT * FROM valuations WHERE ticker = ?"
    params: list[Any] = [ticker]
    if as_of_date is not None:
        sql += " AND date(as_of_date) <= date(?)"
        params.append(as_of_date)
    sql += " ORDER BY as_of_date DESC, created_at DESC, id DESC"
    for row in conn.execute(sql, params).fetchall():
        if not valuation_row_is_decision_eligible(row):
            continue  # a withheld row's inputs are withheld with it
        inputs = _read_json_dict(row["inputs_json"])
        price = inputs.get("current_price")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
            continue
        checked = str(inputs.get("price_as_of_date") or row["as_of_date"] or "") or None
        if as_of_date is not None and (checked is None or checked[:10] > as_of_date[:10]):
            continue  # a quote taken after the as-of day was not known then
        return {
            "latest": float(price),
            "checked_at": checked,
            "source": "valuation_inputs",
            "age_hours": price_age_hours(checked, now=now),
            "wake": [],
        }
    return empty


def price_history(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    now: datetime | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """The ticker's snapshot series across its watchlist rows (dup rows exist;
    the price of the company is one series)."""
    if not (table_exists(conn, "watchlist") and table_exists(conn, "watchlist_price_snapshots")):
        return {"latest": None, "checked_at": None, "source": None, "age_hours": None, "wake": []}
    if as_of_date is None:
        rows = conn.execute(
            """
            SELECT s.price, s.checked_at, s.source
            FROM watchlist_price_snapshots s
            JOIN watchlist w ON w.id = s.watchlist_id
            WHERE w.ticker = ?
            ORDER BY s.checked_at DESC, s.id DESC
            LIMIT 30
            """,
            (ticker,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT s.price, s.checked_at, s.source
            FROM watchlist_price_snapshots s
            JOIN watchlist w ON w.id = s.watchlist_id
            WHERE w.ticker = ? AND date(s.checked_at) <= date(?)
            ORDER BY s.checked_at DESC, s.id DESC
            LIMIT 30
            """,
            (ticker, as_of_date),
        ).fetchall()
    if not rows:
        return {"latest": None, "checked_at": None, "source": None, "age_hours": None, "wake": []}
    newest = rows[0]
    wake = [float(r["price"]) for r in reversed(rows)]
    return {
        "latest": float(newest["price"]),
        "checked_at": newest["checked_at"],
        "source": newest["source"],
        "age_hours": price_age_hours(newest["checked_at"], now=now),
        "wake": wake,
    }


def watchlist_profile(
    conn: sqlite3.Connection, ticker: str, *, watchlist_id: int | None = None
) -> dict[str, Any] | None:
    """Narrative fields from the ticker's authoritative watchlist row.

    ``watchlist_id`` (the queue row's id) pins the exact row the queue
    presents — dup-ticker rows exist, and MAX(id) is not the authoritative
    one. Without it, fall back to the store's own current-row selection.
    """
    if not financial_integrity_manifest_is_usable():
        return None
    if not table_exists(conn, "watchlist"):
        return None
    from app.watchlist.store import current_watchlist_cte

    if watchlist_id is not None:
        where_clause, params = "w.id = ?", (watchlist_id,)
    else:
        where_clause, params = (
            f"""w.id = (
                SELECT latest_id FROM ({current_watchlist_cte()})
                WHERE ticker = ?
            )""",
            (ticker,),
        )
    row = conn.execute(
        f"""
        SELECT w.*
        FROM watchlist w
        WHERE {where_clause}
        """,
        params,
    ).fetchone()
    if row is None or not watchlist_row_is_decision_eligible(row, ticker=ticker):
        return None
    return {
        "watchlist_id": int(row["id"]),
        "thesis_text": row["thesis_text"],
        "key_risks": [str(r) for r in _read_json_list(row["key_risks_json"])],
        "open_questions": [str(q) for q in _read_json_list(row["open_questions_json"])],
        "valuation_anchor_method": row["valuation_anchor_method"],
        "valuation_anchor_value": (
            float(row["valuation_anchor_value"])
            if row["valuation_anchor_value"] is not None
            else None
        ),
        "current_price_at_addition": (
            float(row["current_price_at_addition"])
            if row["current_price_at_addition"] is not None
            else None
        ),
        "source_run_id": row["source_run_id"],
        "cap_asof": row["cap_asof"],
        "added_at": row["added_at"],
    }


def fundamentals_fy(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    years: int = 10,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """FY series (USD millions) plus honest derivations (FCF, margins, net debt)."""
    if as_of_date is None:
        rows = conn.execute(
            f"""
            SELECT fiscal_year, line_item, value
            FROM companyfacts_facts
            WHERE ticker = ? AND period_type = 'FY'
              AND line_item IN ({",".join("?" for _ in FY_LINE_ITEMS)})
            ORDER BY fiscal_year
            """,
            (ticker, *FY_LINE_ITEMS),
        ).fetchall()
    else:
        _, rows = issuer_companyfacts_rows(
            conn,
            ticker,
            columns=("fiscal_year", "line_item", "value"),
            period_types=("FY",),
            line_items=FY_LINE_ITEMS,
            as_of_date=as_of_date,
            require_filed_asof=True,
            order_by="fiscal_year ASC, line_item ASC",
        )
    by_year: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["value"] is None:
            continue
        by_year.setdefault(int(row["fiscal_year"]), {})[str(row["line_item"])] = float(row["value"])
    fiscal_years = sorted(by_year)[-years:]

    def series(item: str) -> list[float | None]:
        return [by_year[y].get(item) for y in fiscal_years]

    def ratio(num: str, den: str) -> list[float | None]:
        out: list[float | None] = []
        for y in fiscal_years:
            n, d = by_year[y].get(num), by_year[y].get(den)
            out.append(round(n / d, 4) if n is not None and d else None)
        return out

    fcf: list[float | None] = []
    net_debt: list[float | None] = []
    for y in fiscal_years:
        cfo, capex = by_year[y].get("cfo"), by_year[y].get("capex")
        fcf.append(round(cfo - capex, 3) if cfo is not None and capex is not None else None)
        debt, cash = by_year[y].get("total_debt"), by_year[y].get("cash")
        net_debt.append(round(debt - cash, 3) if debt is not None and cash is not None else None)

    shares = series("shares_outstanding")
    dilution: list[float | None] = [None]
    for prev, curr in zip(shares, shares[1:], strict=False):
        dilution.append(
            round((curr / prev - 1.0) * 100.0, 2) if prev and curr is not None else None
        )

    return {
        "basis": "fy",
        "units": "USD_millions",
        "fiscal_years": fiscal_years,
        "series": {item: series(item) for item in FY_LINE_ITEMS},
        "derived": {
            "fcf": fcf,
            "net_debt": net_debt,
            "gross_margin": ratio("gross_profit", "revenue"),
            "operating_margin": ratio("operating_income", "revenue"),
            "net_margin": ratio("net_income", "revenue"),
            "roe": ratio("net_income", "equity"),
            "shares_dilution_pct": dilution,
        },
    }


def fundamentals_ttm(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    """Quarterly/TTM basis. Not assembled in this edition — never faked from FY."""
    del conn, ticker
    return {"basis": "ttm", "available": False, "reason": "TTM_NOT_AVAILABLE", "values": {}}


def company_exists(conn: sqlite3.Connection, ticker: str) -> bool:
    """Known to the platform: on the watchlist, valued, or with cached facts."""
    queries = [
        # Existence is routing metadata, not a decision authorization.  The
        # valuation payload itself is still selected then gated below.
        "SELECT 1 FROM valuations WHERE ticker = ? LIMIT 1",
        "SELECT 1 FROM companyfacts_facts WHERE ticker = ? LIMIT 1",
    ]
    if table_exists(conn, "watchlist"):
        queries.insert(0, "SELECT 1 FROM watchlist WHERE ticker = ? LIMIT 1")
    for query in queries:
        if conn.execute(query, (ticker,)).fetchone() is not None:
            return True
    return False


MANIFEST_UNUSABLE = "financial_integrity_manifest_unusable"

AUDIT_AUDITED = "AUDITED"
AUDIT_NOT_AUDITED = "NOT_AUDITED"
AUDIT_NO_VALUATION = "NO_VALUATION"

_EMPTY_PRICES: dict[str, Any] = {
    "latest": None,
    "checked_at": None,
    "source": None,
    "age_hours": None,
    "wake": [],
}


def research_valuations(
    conn: sqlite3.Connection, ticker: str, *, as_of_date: str | None = None
) -> list[dict[str, Any]]:
    """Newest valuation per method that no audited run ever claimed.

    These are the rows ``ivi value`` writes: research output with no
    ``source_run_id``, so they can never be decision-eligible. They are shown
    only in their own labelled block. A row that *does* name a run but fails
    verification is a claimed decision row that lost its audit; it stays
    suppressed, exactly as before.
    """

    return [
        method_card(row)
        for row in _latest_valuation_rows(conn, ticker, as_of_date=as_of_date)
        if str(row["method"]) not in _EXCLUDED_METHODS
        and not str(row["source_run_id"] or "").strip()
    ]


def _has_valuation_rows(
    conn: sqlite3.Connection, ticker: str, *, as_of_date: str | None = None
) -> bool:
    return any(
        str(row["method"]) not in _EXCLUDED_METHODS
        for row in _latest_valuation_rows(conn, ticker, as_of_date=as_of_date)
    )


def audit_status(
    ticker: str,
    *,
    manifest_usable: bool,
    decision_cards: list[dict[str, Any]],
    research_cards: list[dict[str, Any]],
    rows_withheld: bool = False,
) -> dict[str, Any]:
    """Say plainly why the page does or does not carry audited valuations.

    ``rows_withheld``: valuation rows exist but none is shown as a decision
    card or as research output (they name a run whose audit no longer holds).
    """

    if decision_cards:
        return {
            "state": AUDIT_AUDITED,
            "reason": None,
            "message": "Valuations on this page are bound to an audited run.",
            "command": None,
        }
    if research_cards:
        if manifest_usable:
            reason = "NO_AUDITED_VALUATION"
            message = (
                "Research output only. These valuations came from `ivi value`, which is not "
                "bound to an audited run, so they are shown for reference and are not "
                "decision-eligible."
            )
        else:
            reason = MANIFEST_UNUSABLE
            message = (
                "Not audited yet. This install has no financial-integrity audit, so the "
                "decision sections of this page (thesis, depth gauge, buy target) are withheld. "
                "The valuation below is research output from `ivi value`; it is not bound to "
                "an audited run and is not a decision."
            )
        return {"state": AUDIT_NOT_AUDITED, "reason": reason, "message": message, "command": None}
    if rows_withheld:
        return {
            "state": AUDIT_NOT_AUDITED,
            "reason": MANIFEST_UNUSABLE if not manifest_usable else "AUDIT_NOT_HOLDING",
            "message": (
                f"Valuation rows exist for {ticker} but are withheld: they name an audited run "
                "whose audit is missing, stale, or no longer matches, so none is shown."
            ),
            "command": None,
        }
    command = f"ivi value {ticker}"
    message = f"No valuation on file for {ticker}. Run `{command}` to compute one."
    if not manifest_usable:
        message += (
            " Decision sections of this page also need the financial-integrity audit, "
            "which this install has not run."
        )
    return {
        "state": AUDIT_NO_VALUATION,
        "reason": None if manifest_usable else MANIFEST_UNUSABLE,
        "message": message,
        "command": command,
    }


def unaudited_company_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    now: datetime | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """The company page when no audit manifest authorizes decision data.

    ``company_snapshot`` still refuses in that state; this is the honest page
    for a fresh install. Every decision-bearing field (watchlist row, thesis,
    gauge, evolution, prices) is empty, and the only numbers shown are the
    lineage-free research rows, labelled as such.
    """

    normalized = ticker.upper()
    research = research_valuations(conn, normalized, as_of_date=as_of_date)
    return {
        "ticker": normalized,
        "watchlist": None,
        "profile": None,
        "valuations": [],
        "shelves": [],
        "evolution": [],
        "prices": dict(_EMPTY_PRICES),
        "as_of": as_of_date,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "audit": audit_status(
            normalized,
            manifest_usable=False,
            decision_cards=[],
            research_cards=research,
            rows_withheld=not research
            and _has_valuation_rows(conn, normalized, as_of_date=as_of_date),
        ),
        "research_valuations": research,
    }


def company_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    queue_row: dict[str, Any] | None,
    now: datetime | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    if not financial_integrity_manifest_is_usable():
        raise OfflineError(
            MANIFEST_UNUSABLE,
            "The canonical financial-integrity audit manifest is missing, malformed, or empty.",
        )
    normalized = ticker.upper()
    queue_id = int(queue_row["id"]) if queue_row else None
    profile = watchlist_profile(conn, normalized, watchlist_id=queue_id)
    cards = latest_valuations(conn, normalized, as_of_date=as_of_date)
    anchor_method = profile["valuation_anchor_method"] if profile else None
    anchor_value = profile["valuation_anchor_value"] if profile else None
    shelves = gauge_shelves(cards, anchor_method=anchor_method, anchor_value=anchor_value)
    prices = price_history(conn, normalized, now=now, as_of_date=as_of_date)
    if prices["latest"] is None:
        prices = _price_from_valuation_inputs(conn, normalized, now=now, as_of_date=as_of_date)
    research = research_valuations(conn, normalized, as_of_date=as_of_date)
    evolution, evolution_stats = valuation_evolution_with_drops(conn, normalized, as_of_date=as_of_date)
    return {
        "ticker": normalized,
        "watchlist": queue_row,
        "profile": profile,
        "valuations": cards,
        "shelves": shelves,
        "evolution": evolution,
        "evolution_dropped_pre_split": evolution_stats["dropped_pre_split"],
        "evolution_rebased_points": evolution_stats["rebased"],
        "prices": prices,
        "as_of": as_of_date,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "audit": audit_status(
            normalized,
            manifest_usable=True,
            decision_cards=cards,
            research_cards=research,
            rows_withheld=not cards
            and not research
            and _has_valuation_rows(conn, normalized, as_of_date=as_of_date),
        ),
        "research_valuations": research,
    }
