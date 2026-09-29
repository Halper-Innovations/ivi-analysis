"""Holdings registry + exit-signal feed + held-book monitoring.

IVI classic only ever said BUY — nothing represented a held position, so
delistings reached no surface, CONTRADICTED was a refresh-excluded trap,
and a position could ride through a going-concern filing or a 60% drawdown
without a tripwire. This module adds the read side:

- ``holdings`` table — **SCHEMA PROVISIONAL**: the shared contract with the
  portfolio-implementation effort, which
  should ultimately own writes. Classic owns the read-side evaluator only;
  no sizing, no order placement, ever.
- ``exit_signals`` — typed, append-only feed the daily held-book pass emits
  by joining holdings against signals the platform already computes:
  THESIS_CONTRADICTED (REVIEW_REQUIRED for held names, never the terminal
  trap), GATE_FLIP, TARGET_REPUDIATED, UNIVERSE_EXIT, SIGNAL_EXPIRED,
  DRAWDOWN_TRIPWIRE, MONITORING_FAILURE, and PRICE_AT_FAIR_VALUE
  (mechanism shipped; config-gated OFF by default —
  every existing zone threshold is buy-side; the exit threshold is a new
  policy decision).
- Held names are force-registered into the filing watch and included in
  the events queue-protection CIK scope — monitoring keys to capital at
  risk, not just watchlist presentation state.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.logging import get_logger

logger = get_logger(__name__)

EXIT_SIGNAL_TYPES = (
    "THESIS_CONTRADICTED",
    "GATE_FLIP",
    "TARGET_REPUDIATED",
    "UNIVERSE_EXIT",
    "SIGNAL_EXPIRED",
    "PRICE_AT_FAIR_VALUE",
    "DRAWDOWN_TRIPWIRE",
    "MONITORING_FAILURE",
)

# Signals that page the operator the same day (via the heartbeat alert path).
CRITICAL_SIGNAL_TYPES = {
    "GATE_FLIP",
    "UNIVERSE_EXIT",
    "DRAWDOWN_TRIPWIRE",
    "MONITORING_FAILURE",
    "THESIS_CONTRADICTED",
}

SIGNAL_EXPIRED_AGE_DAYS = 365


def _drawdown_alert_pct_default() -> float:
    raw = os.getenv("VOE_EXIT_DRAWDOWN_ALERT_PCT", "")
    try:
        return float(raw) if raw.strip() else 25.0
    except ValueError:
        return 25.0


def _fair_value_exit_enabled() -> bool:
    # Mechanism only; flipping this on (and choosing the fraction) is an
    # explicit operator policy decision.
    return os.getenv("VOE_EXIT_FAIR_VALUE_ENABLED", "").strip().lower() == "true"


def _fair_value_fraction() -> float:
    raw = os.getenv("VOE_EXIT_FAIR_VALUE_FRACTION", "")
    try:
        return float(raw) if raw.strip() else 1.0
    except ValueError:
        return 1.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_holdings_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings (
            -- Provisional schema; may change with portfolio-interface work.
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            cik TEXT,
            entry_date TEXT NOT NULL,
            entry_price REAL,
            size TEXT,
            size_pct_nav REAL,
            thesis_watchlist_id INTEGER,
            thesis_outcome_id INTEGER,
            exit_rules_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'OPEN',
            closed_at TEXT,
            close_price REAL,
            peak_price REAL,
            max_drawdown_pct REAL,
            drawdown_alert_pct REAL,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_holdings_status ON holdings(status, ticker)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exit_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            holding_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'OPEN',
            acknowledged_at TEXT,
            UNIQUE(holding_id, signal_type, as_of_date)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_exit_signals_open "
        "ON exit_signals(status, ticker, signal_type)"
    )


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    from app.db import connect
    from app.watchlist.schema import ensure_watchlist_schema

    ensure_watchlist_schema(db_path)
    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    conn = connect(path)
    ensure_holdings_schema(conn)
    return conn


def _resolve_cik(ticker: str) -> str | None:
    try:
        from app.universe.ticker_cik_map import load_ticker_cik_map

        cik = load_ticker_cik_map(refresh_if_missing=False).get(ticker.upper())
        return str(cik).zfill(10) if cik else None
    except Exception:  # noqa: BLE001
        return None


def add_holding(
    *,
    ticker: str,
    entry_date: str,
    entry_price: float | None = None,
    size: str | None = None,
    size_pct_nav: float | None = None,
    notes: str | None = None,
    drawdown_alert_pct: float | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Register a held position; thesis references auto-link to the latest
    live watchlist row and its decision-time outcome row when present."""
    ticker_norm = ticker.strip().upper()
    now = _utc_now_iso()
    conn = _connect(db_path)
    try:
        wl_row = conn.execute(
            "SELECT id FROM watchlist WHERE ticker = ? AND status != 'REMOVED' "
            "ORDER BY id DESC LIMIT 1",
            (ticker_norm,),
        ).fetchone()
        outcome_row = None
        try:
            outcome_row = conn.execute(
                "SELECT id FROM ticker_outcomes WHERE ticker = ? ORDER BY updated_at DESC LIMIT 1",
                (ticker_norm,),
            ).fetchone()
        except sqlite3.Error:
            outcome_row = None
        conn.execute(
            """
            INSERT INTO holdings(
                ticker, cik, entry_date, entry_price, size, size_pct_nav,
                thesis_watchlist_id, thesis_outcome_id, status, peak_price,
                drawdown_alert_pct, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?)
            """,
            (
                ticker_norm,
                _resolve_cik(ticker_norm),
                entry_date,
                entry_price,
                size,
                size_pct_nav,
                int(wl_row["id"]) if wl_row is not None else None,
                int(outcome_row["id"]) if outcome_row is not None else None,
                entry_price,
                drawdown_alert_pct
                if drawdown_alert_pct is not None
                else _drawdown_alert_pct_default(),
                notes,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM holdings ORDER BY id DESC LIMIT 1").fetchone()
        result = {key: row[key] for key in row.keys()}
    finally:
        conn.close()
    return result


def close_holding(
    *,
    holding_id: int,
    close_price: float | None = None,
    closed_at: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Close a holding; auto-populates max_drawdown_pct on the holding and
    (when NULL) on the linked decision-time outcome row."""
    now = closed_at or _utc_now_iso()
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM holdings WHERE id = ?", (int(holding_id),)).fetchone()
        if row is None:
            raise ValueError(f"no holding with id {holding_id}")
        if str(row["status"]) != "OPEN":
            raise ValueError(f"holding {holding_id} already {row['status']}")
        max_dd = _max_drawdown_pct(
            conn,
            ticker=str(row["ticker"]),
            start=str(row["entry_date"]),
            end=now[:10],
            entry_price=row["entry_price"],
        )
        conn.execute(
            "UPDATE holdings SET status='CLOSED', closed_at=?, close_price=?, "
            "max_drawdown_pct=?, updated_at=? WHERE id=?",
            (now, close_price, max_dd, _utc_now_iso(), int(holding_id)),
        )
        if row["thesis_outcome_id"] is not None and max_dd is not None:
            try:
                outcome = conn.execute(
                    "SELECT * FROM ticker_outcomes WHERE id = ?",
                    (int(row["thesis_outcome_id"]),),
                ).fetchone()
            except sqlite3.Error:
                outcome = None
            if (
                outcome is not None
                and outcome["max_drawdown_pct"] is None
                and str(outcome["outcome_status"] or "").upper() == "OPEN"
            ):
                from app.outcomes.lineage import (
                    outcome_row_is_decision_eligible,
                    refresh_outcome_integrity_fingerprint,
                )
                from app.outcomes.store import archive_outcome_row

                if outcome_row_is_decision_eligible(outcome):
                    archive_outcome_row(conn, int(outcome["id"]))
                    conn.execute(
                        "UPDATE ticker_outcomes SET max_drawdown_pct = ?, updated_at = ? "
                        "WHERE id = ?",
                        (max_dd, _utc_now_iso(), int(outcome["id"])),
                    )
                    if not refresh_outcome_integrity_fingerprint(
                        conn,
                        int(outcome["id"]),
                    ):
                        raise RuntimeError(
                            "outcome source authorization changed while recording drawdown"
                        )
        conn.commit()
        closed = conn.execute("SELECT * FROM holdings WHERE id = ?", (int(holding_id),)).fetchone()
        return {key: closed[key] for key in closed.keys()}
    finally:
        conn.close()


def list_holdings(
    *, status: str | None = "OPEN", db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM holdings WHERE status = ? ORDER BY ticker", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM holdings ORDER BY ticker").fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]
    finally:
        conn.close()


def open_exit_signals(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM exit_signals WHERE status = 'OPEN' ORDER BY detected_at DESC, ticker"
        ).fetchall()
        out = []
        for row in rows:
            payload = {key: row[key] for key in row.keys()}
            try:
                payload["evidence"] = json.loads(payload.get("evidence_json") or "{}")
            except (TypeError, ValueError):
                payload["evidence"] = {}
            out.append(payload)
        return out
    finally:
        conn.close()


def _emit_signal(
    conn: sqlite3.Connection,
    *,
    holding_id: int,
    ticker: str,
    signal_type: str,
    as_of_date: str,
    evidence: dict[str, Any],
) -> bool:
    """Append one exit signal; idempotent per (holding, type, day)."""
    cursor = conn.execute(
        """
        INSERT INTO exit_signals(
            holding_id, ticker, signal_type, as_of_date, detected_at, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(holding_id, signal_type, as_of_date) DO NOTHING
        """,
        (
            int(holding_id),
            ticker.upper(),
            signal_type,
            as_of_date,
            _utc_now_iso(),
            json.dumps(evidence, default=str),
        ),
    )
    return cursor.rowcount > 0


def _max_drawdown_pct(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    start: str,
    end: str,
    entry_price: float | None,
) -> float | None:
    """Max drawdown from the running peak over price_quotes in [start, end]."""
    try:
        rows = conn.execute(
            """
            SELECT as_of_date, MAX(price) AS price FROM price_quotes
            WHERE ticker = ? AND as_of_date BETWEEN ? AND ?
              AND status = 'OK' AND price IS NOT NULL AND price > 0
            GROUP BY as_of_date ORDER BY as_of_date
            """,
            (ticker.upper(), start, end),
        ).fetchall()
    except sqlite3.Error:
        return None
    prices = [float(row["price"]) for row in rows]
    if entry_price is not None and entry_price > 0:
        prices.insert(0, float(entry_price))
    if len(prices) < 2:
        return None
    peak = prices[0]
    worst = 0.0
    for price in prices[1:]:
        peak = max(peak, price)
        drawdown = (price - peak) / peak * 100.0
        worst = min(worst, drawdown)
    return round(worst, 4)


def held_cik_map(conn: sqlite3.Connection) -> dict[str, str]:
    """cik10 -> ticker for OPEN holdings (events protection scope)."""
    try:
        rows = conn.execute("SELECT ticker, cik FROM holdings WHERE status = 'OPEN'").fetchall()
    except sqlite3.Error:
        return {}
    out: dict[str, str] = {}
    for row in rows:
        ticker = str(row["ticker"]).upper()
        cik = str(row["cik"] or "").strip()
        if not cik:
            cik = _resolve_cik(ticker) or ""
        if cik:
            out[cik.zfill(10)] = ticker
    return out


def run_held_book_pass(
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    price_lookup: Any | None = None,
) -> dict[str, Any]:
    """Daily held-book pass: join OPEN holdings against every exit signal the
    platform already computes; emit typed, append-only exit_signals rows.

    Returns {"holdings", "new_signals": [...], "critical": [...]} — the CLI
    exits 3 when critical signals were emitted so the heartbeat alerts.
    """
    asof = str(as_of_date or date.today().isoformat())
    conn = _connect(db_path)
    new_signals: list[dict[str, Any]] = []
    try:
        holdings = conn.execute(
            "SELECT * FROM holdings WHERE status = 'OPEN' ORDER BY ticker"
        ).fetchall()
        if not holdings:
            return {"as_of_date": asof, "holdings": 0, "new_signals": [], "critical": []}

        for holding in holdings:
            ticker = str(holding["ticker"]).upper()
            holding_id = int(holding["id"])

            def emit(
                signal_type: str,
                evidence: dict[str, Any],
                _holding_id: int = holding_id,
                _ticker: str = ticker,
            ) -> None:
                if _emit_signal(
                    conn,
                    holding_id=_holding_id,
                    ticker=_ticker,
                    signal_type=signal_type,
                    as_of_date=asof,
                    evidence=evidence,
                ):
                    new_signals.append(
                        {"ticker": _ticker, "signal_type": signal_type, "evidence": evidence}
                    )

            # 1. UNIVERSE_EXIT: registry removal or watchlist REMOVED.
            try:
                reg = conn.execute(
                    "SELECT removed_at FROM sec_registrants "
                    "WHERE primary_ticker = ? AND removed_at IS NOT NULL",
                    (ticker,),
                ).fetchone()
            except sqlite3.Error:
                reg = None
            if reg is not None:
                emit(
                    "UNIVERSE_EXIT", {"removed_at": reg["removed_at"], "source": "sec_registrants"}
                )

            wl = conn.execute(
                "SELECT id, status, status_reason, conviction_grade, "
                "valuation_anchor_value, buy_price_target, last_evaluated_at, added_at "
                "FROM watchlist WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            if wl is not None and str(wl["status"]).upper() == "REMOVED":
                emit(
                    "UNIVERSE_EXIT",
                    {"watchlist_status": "REMOVED", "reason": wl["status_reason"]},
                )

            # 2. THESIS_CONTRADICTED — for a HELD name this is REVIEW_REQUIRED
            # plus an alert, never the refresh-excluded terminal trap.
            if wl is not None and str(wl["status"]).upper() == "CONTRADICTED":
                emit(
                    "THESIS_CONTRADICTED",
                    {
                        "action": "REVIEW_REQUIRED",
                        "reason": wl["status_reason"],
                        "note": "held name — schedule re-review, not terminal",
                    },
                )

            # 3. GATE_FLIP: structural/going-concern on current filings.
            try:
                from app.autonomous.structural_gate import evaluate_structural_gate

                gate = evaluate_structural_gate(ticker, as_of_date=asof, db_path=db_path)
                if gate.quarantined:
                    emit(
                        "GATE_FLIP",
                        {"codes": list(gate.triggered_codes), "details": dict(gate.details)},
                    )
            except Exception:  # noqa: BLE001 - a gate crash is a monitoring failure
                emit("MONITORING_FAILURE", {"stage": "structural_gate"})

            # 4. TARGET_REPUDIATED: today's rederive would NULL the target.
            try:
                from app.watchlist.target_rederive import rederive_watchlist_targets

                rederive = rederive_watchlist_targets(db_path=db_path, tickers=[ticker])
                for item_row in rederive.get("rows", []):
                    disposition = str(
                        item_row.get("disposition") or getattr(item_row, "disposition", "")
                    )
                    if disposition == "WOULD_NULL":
                        emit("TARGET_REPUDIATED", {"disposition": "WOULD_NULL"})
            except Exception:  # noqa: BLE001 - rederive is advisory here
                pass

            # 5. SIGNAL_EXPIRED: underwrite age beyond the SLA.
            basis_raw = None
            if wl is not None:
                basis_raw = wl["last_evaluated_at"] or wl["added_at"]
            if basis_raw:
                try:
                    basis_date = datetime.fromisoformat(
                        str(basis_raw).replace("Z", "+00:00")
                    ).date()
                    age_days = (date.fromisoformat(asof) - basis_date).days
                    if age_days > SIGNAL_EXPIRED_AGE_DAYS:
                        emit(
                            "SIGNAL_EXPIRED",
                            {"age_days": age_days, "sla_days": SIGNAL_EXPIRED_AGE_DAYS},
                        )
                except ValueError:
                    pass

            # 6-8. Price-driven checks share one quote.
            latest_price = None
            try:
                if price_lookup is not None:
                    snapshot = price_lookup(ticker)
                else:
                    from app.watchlist.triggers import _fetch_latest_price

                    snapshot = _fetch_latest_price(ticker)
                if snapshot is not None and isinstance(snapshot.price, (int, float)):
                    latest_price = float(snapshot.price)
            except Exception:  # noqa: BLE001
                latest_price = None
            if latest_price is None:
                # A held name that stops pricing is itself a red flag.
                emit("MONITORING_FAILURE", {"stage": "price_lookup", "detail": "no price"})
            else:
                entry_price = holding["entry_price"]
                peak = holding["peak_price"]
                new_peak = max(
                    [p for p in (peak, entry_price, latest_price) if isinstance(p, (int, float))]
                )
                conn.execute(
                    "UPDATE holdings SET peak_price = ?, updated_at = ? WHERE id = ?",
                    (new_peak, _utc_now_iso(), holding_id),
                )
                threshold = float(holding["drawdown_alert_pct"] or _drawdown_alert_pct_default())
                if isinstance(entry_price, (int, float)) and entry_price > 0:
                    dd_entry = (latest_price - entry_price) / entry_price * 100.0
                    if dd_entry <= -threshold:
                        emit(
                            "DRAWDOWN_TRIPWIRE",
                            {
                                "basis": "entry",
                                "drawdown_pct": round(dd_entry, 2),
                                "threshold_pct": threshold,
                                "entry_price": entry_price,
                                "latest_price": latest_price,
                            },
                        )
                if isinstance(new_peak, (int, float)) and new_peak > 0:
                    dd_peak = (latest_price - new_peak) / new_peak * 100.0
                    if dd_peak <= -threshold:
                        emit(
                            "DRAWDOWN_TRIPWIRE",
                            {
                                "basis": "trailing_peak",
                                "drawdown_pct": round(dd_peak, 2),
                                "threshold_pct": threshold,
                                "peak_price": new_peak,
                                "latest_price": latest_price,
                            },
                        )
                # PRICE_AT_FAIR_VALUE — mechanism only, behind an off-by-default flag.
                if (
                    _fair_value_exit_enabled()
                    and wl is not None
                    and isinstance(wl["valuation_anchor_value"], (int, float))
                    and float(wl["valuation_anchor_value"]) > 0
                    and latest_price >= _fair_value_fraction() * float(wl["valuation_anchor_value"])
                ):
                    emit(
                        "PRICE_AT_FAIR_VALUE",
                        {
                            "latest_price": latest_price,
                            "anchor_value": wl["valuation_anchor_value"],
                            "fraction": _fair_value_fraction(),
                        },
                    )
        conn.commit()
    finally:
        conn.close()

    critical = [signal for signal in new_signals if signal["signal_type"] in CRITICAL_SIGNAL_TYPES]
    return {
        "as_of_date": asof,
        "holdings": len(holdings),
        "new_signals": new_signals,
        "critical": critical,
    }


__all__ = [
    "CRITICAL_SIGNAL_TYPES",
    "EXIT_SIGNAL_TYPES",
    "add_holding",
    "close_holding",
    "ensure_holdings_schema",
    "held_cik_map",
    "list_holdings",
    "open_exit_signals",
    "run_held_book_pass",
]
