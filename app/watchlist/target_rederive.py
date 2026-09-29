"""Re-derive watchlist anchors + buy targets through the live packet path.

Watchlist rows persist the anchor and buy target computed when the row was
written. Anchor-policy changes (decline-cap, zone suppression, anchor-sanity)
apply to NEW runs only, so existing rows can carry stale pre-policy targets —
the motivating case was a decline-class row whose growth-anchored target sat
~40% above its no-growth basis and kept ranking as the deepest "discount" on
every decision surface.

This module recomputes what production would persist TODAY for every live
row, through the exact production chain (never a local re-derivation of the
anchor rule):

    assemble_signal_packet (deterministic, filing_risk_use_llm=False)
      -> build_sector_company_financial_packet   (select_anchor incl. decline cap)
      -> watchlist.store._buy_price_target       (grade/confidence-scaled MoS)

Dispositions per row:
  * UNCHANGED    — derived target within tolerance of the stored one, same method.
  * RETARGET     — positive derived target materially differs (or fills a NULL);
                   applied when ``apply=True`` with history rows.
  * WOULD_NULL   — production would derive NO anchor/target today (gate block,
                   anomaly suppression, no positive method). NEVER applied here:
                   nulling a target silently disables the price trigger, so these
                   rows are surfaced for owner triage instead.
  * ERROR        — packet assembly raised; reported, row untouched.

Statuses REMOVED / RESOLVED are out of scope (closed rows must keep the
numbers they closed with).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.alpha.signal_assembler import assemble_signal_packet
from app.autonomous.sector_financial_packets import (
    _quality_context,
    build_sector_company_financial_packet,
)
from app.watchlist.store import _buy_price_target, _connect, _insert_history

HISTORY_SOURCE = "target_rederive"
#: Relative tolerance below which a target move is float noise, not a retarget.
RETARGET_REL_TOLERANCE = 0.005

DISPOSITION_UNCHANGED = "UNCHANGED"
DISPOSITION_RETARGET = "RETARGET"
DISPOSITION_WOULD_NULL = "WOULD_NULL"
DISPOSITION_ERROR = "ERROR"

_EXCLUDED_STATUSES = ("REMOVED", "RESOLVED")


@dataclass
class RederiveRow:
    watchlist_id: int
    ticker: str
    status: str
    conviction_grade: str | None
    confidence: str | None
    revenue_trend_class: str | None
    old_anchor_method: str | None
    old_anchor_value: float | None
    old_buy_target: float | None
    new_anchor_method: str | None
    new_anchor_value: float | None
    new_buy_target: float | None
    disposition: str
    detail: str = ""


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _materially_different(old: float | None, new: float) -> bool:
    if old is None or old <= 0:
        return True
    return abs(new - old) / old > RETARGET_REL_TOLERANCE


def _live_rows(conn, tickers: list[str] | None) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in _EXCLUDED_STATUSES)
    rows = conn.execute(
        f"""
        SELECT w.*
        FROM watchlist w
        JOIN (
            SELECT ticker, MAX(id) AS latest_id
            FROM watchlist
            GROUP BY ticker
        ) latest ON latest.latest_id = w.id
        WHERE UPPER(w.status) NOT IN ({placeholders})
        ORDER BY w.ticker
        """,
        _EXCLUDED_STATUSES,
    ).fetchall()
    out = [dict(row) for row in rows]
    if tickers:
        wanted = {t.upper() for t in tickers}
        out = [row for row in out if str(row["ticker"]).upper() in wanted]
    return out


def _derive_for_row(row: dict[str, Any], *, db_path: str | Path | None) -> RederiveRow:
    ticker = str(row["ticker"]).upper()
    base = dict(
        watchlist_id=int(row["id"]),
        ticker=ticker,
        status=str(row["status"] or ""),
        conviction_grade=row.get("conviction_grade"),
        confidence=row.get("confidence"),
        old_anchor_method=row.get("valuation_anchor_method"),
        old_anchor_value=_num(row.get("valuation_anchor_value")),
        old_buy_target=_num(row.get("buy_price_target")),
    )
    try:
        signal_packet = assemble_signal_packet(ticker, filing_risk_use_llm=False)
        fin_packet = build_sector_company_financial_packet(
            signal_packet, sector=row.get("source_sector")
        )
    except Exception as exc:  # report, never abort the sweep over one name
        return RederiveRow(
            **base,
            revenue_trend_class=None,
            new_anchor_method=None,
            new_anchor_value=None,
            new_buy_target=None,
            disposition=DISPOSITION_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        )

    valuation = fin_packet.valuation if isinstance(fin_packet.valuation, dict) else {}
    # Same fallback chain production uses (raw ctx -> scorecard quality_context),
    # so the reported trend class always matches what the decline cap saw.
    quality = _quality_context(signal_packet)
    new_method = valuation.get("anchor_method")
    new_anchor = _num(valuation.get("valuation_anchor"))
    new_target = _buy_price_target(
        fin_packet,
        conviction_grade=row.get("conviction_grade"),
        confidence=row.get("confidence"),
        db_path=db_path,
    )
    common = dict(
        **base,
        revenue_trend_class=(
            str(quality.get("revenue_trend_class")) if quality.get("revenue_trend_class") else None
        ),
        new_anchor_method=str(new_method) if new_method else None,
        new_anchor_value=new_anchor,
        new_buy_target=_num(new_target),
    )

    if new_target is None or new_target <= 0 or new_anchor is None:
        return RederiveRow(
            **common,
            disposition=DISPOSITION_WOULD_NULL,
            detail=f"gate_action={valuation.get('gate_action')}",
        )
    method_changed = (str(new_method) if new_method else None) != base["old_anchor_method"]
    if method_changed or _materially_different(base["old_buy_target"], float(new_target)):
        return RederiveRow(**common, disposition=DISPOSITION_RETARGET)
    return RederiveRow(**common, disposition=DISPOSITION_UNCHANGED)


def _apply_retarget(conn, item: RederiveRow, *, source_run_id: str | None) -> None:
    conn.execute(
        """
        UPDATE watchlist
        SET valuation_anchor_method = ?, valuation_anchor_value = ?, buy_price_target = ?
        WHERE id = ?
        """,
        (item.new_anchor_method, item.new_anchor_value, item.new_buy_target, item.watchlist_id),
    )
    for field_name, old, new in (
        ("buy_price_target", item.old_buy_target, item.new_buy_target),
        ("valuation_anchor_method", item.old_anchor_method, item.new_anchor_method),
        ("valuation_anchor_value", item.old_anchor_value, item.new_anchor_value),
    ):
        if old != new:
            _insert_history(
                conn,
                watchlist_id=item.watchlist_id,
                field_name=field_name,
                old_value=old,
                new_value=new,
                source=HISTORY_SOURCE,
                source_run_id=source_run_id,
            )


def rederive_watchlist_targets(
    *,
    db_path: str | Path | None = None,
    apply: bool = False,
    tickers: list[str] | None = None,
    source_run_id: str | None = HISTORY_SOURCE,
) -> dict[str, Any]:
    """Re-derive anchors/targets for live watchlist rows; apply RETARGETs when asked.

    Returns a report dict: ``{"applied": bool, "counts": {...}, "rows": [...]}``
    with one entry per live row. WOULD_NULL rows are never mutated (docstring).
    Statuses are trigger-owned and untouched — run the price-trigger pass after
    applying so DEPLOY_READY reflects the corrected targets.
    """
    conn = _connect(db_path)
    try:
        live = _live_rows(conn, tickers)
        results = [_derive_for_row(row, db_path=db_path) for row in live]
        if apply:
            for item in results:
                if item.disposition == DISPOSITION_RETARGET:
                    _apply_retarget(conn, item, source_run_id=source_run_id)
            conn.commit()
    finally:
        conn.close()

    counts: dict[str, int] = {}
    for item in results:
        counts[item.disposition] = counts.get(item.disposition, 0) + 1
    return {
        "applied": bool(apply),
        "counts": counts,
        "rows": [asdict(item) for item in results],
    }
