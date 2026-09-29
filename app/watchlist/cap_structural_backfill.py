"""Combined cap-chain + structural-gate backfill over existing watchlist rows.

Owner-invoked maintenance pass (Part 3 of the 2026-06-11 band-integrity
work): every non-REMOVED watchlist row gets the band-filter chain result
persisted (market_cap_mm / cap_source / cap_band / cap_asof), the structural
exclusion gate applied (status -> QUARANTINE with the literal
QUARANTINE_STRUCTURAL:<code> reason string), and a band-rule note when the
resolved cap routes the row OUT of its source sweep's band (the queue's
--band scope does the actual routing; the note makes the change auditable).

The report lists exactly which rows changed and why; ``dry_run`` previews
without writing.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

from app.autonomous.cap_resolver import (
    CapClassification,
    classify_market_cap_for_band_filter,
    default_price_lookup,
)
from app.autonomous.sector_candidates import MARKET_CAP_FOCUS_TIERS
from app.autonomous.structural_gate import evaluate_structural_gate
from app.watchlist.store import (
    _connect,
    _insert_history,
    _load_sector_artifact_for_run_id,
)

logger = logging.getLogger(__name__)

_CAP_FIELDS = ("market_cap_mm", "cap_source", "cap_band", "cap_asof")


def _latest_snapshot_price(conn, watchlist_id: int) -> float | None:
    row = conn.execute(
        """
        SELECT price FROM watchlist_price_snapshots
        WHERE watchlist_id = ?
        ORDER BY checked_at DESC, id DESC
        LIMIT 1
        """,
        (watchlist_id,),
    ).fetchone()
    if row is None or row["price"] is None:
        return None
    return float(row["price"]) if float(row["price"]) > 0 else None


def _price_amount(value: Any) -> float | None:
    raw_value = (
        value.get("price")
        if isinstance(value, dict)
        else getattr(
            value,
            "price",
            value,
        )
    )
    if (
        isinstance(raw_value, (int, float))
        and not isinstance(raw_value, bool)
        and float(raw_value) > 0
    ):
        return float(raw_value)
    return None


def _focus_for_run(
    run_id: str, cache: dict[str, str | None], *, runs_dir: str | Path | None
) -> str | None:
    if run_id in cache:
        return cache[run_id]
    artifact = _load_sector_artifact_for_run_id(run_id, runs_dir=runs_dir)
    focus = getattr(artifact, "market_cap_focus", None) if artifact is not None else None
    cache[run_id] = str(focus) if focus else None
    return cache[run_id]


def _routed_out_of_band(classification: CapClassification, focus: str | None) -> bool:
    if not focus:
        return False
    bounds = MARKET_CAP_FOCUS_TIERS.get(str(focus).strip().lower())
    if bounds is None:
        return False
    return classification.in_band(bounds[0], bounds[1]) is False


def backfill_watchlist_cap_and_structural(
    db_path: str | Path | None = None,
    *,
    as_of_date: str | None = None,
    dry_run: bool = False,
    price_lookup=None,
    runs_dir: str | Path | None = None,
    history_source: str = "cap_structural_backfill",
) -> dict[str, Any]:
    """Apply chain + band rule + structural gate to non-REMOVED rows.

    Classification price per ticker: latest price snapshot, else provider
    current price, else the addition price (stale but last known). One
    classification + one gate evaluation per ticker, applied to every
    non-REMOVED row of that ticker.
    """
    asof = str(as_of_date or "").strip() or date.today().isoformat()
    lookup = price_lookup or default_price_lookup()
    conn = _connect(db_path)
    changed_rows: list[dict[str, Any]] = []
    per_ticker: dict[str, tuple[CapClassification, Any, float | None]] = {}
    focus_cache: dict[str, str | None] = {}
    counts = {
        "rows_examined": 0,
        "rows_changed": 0,
        "tickers_quarantined_structural": 0,
        "rows_routed_out_of_band": 0,
        "cap_source": {},
    }
    try:
        rows = conn.execute(
            "SELECT * FROM watchlist WHERE status != 'REMOVED' ORDER BY ticker, id"
        ).fetchall()

        # Latest row id per ticker sources the snapshot price.
        latest_row_by_ticker: dict[str, int] = {}
        for row in rows:
            latest_row_by_ticker[str(row["ticker"]).upper()] = int(row["id"])

        for row in rows:
            counts["rows_examined"] += 1
            ticker = str(row["ticker"]).upper()
            if ticker not in per_ticker:
                price_evidence = lookup(ticker, asof)
                if price_evidence is None:
                    price_evidence = _latest_snapshot_price(
                        conn,
                        latest_row_by_ticker[ticker],
                    )
                if price_evidence is None and row["current_price_at_addition"] is not None:
                    addition = float(row["current_price_at_addition"])
                    price_evidence = addition if addition > 0 else None
                price = _price_amount(price_evidence)
                classification = classify_market_cap_for_band_filter(
                    ticker,
                    as_of_date=asof,
                    asof_price=price_evidence,
                    current_price=price_evidence,
                    db_path=db_path,
                    price_lookup=lambda _t, _a: None,
                )
                gate = evaluate_structural_gate(
                    ticker,
                    as_of_date=asof,
                    price=price,
                    market_cap_mm=classification.market_cap_mm,
                    db_path=db_path,
                )
                per_ticker[ticker] = (classification, gate, price)
                counts["cap_source"][classification.cap_source] = (
                    counts["cap_source"].get(classification.cap_source, 0) + 1
                )
                if gate.quarantined:
                    counts["tickers_quarantined_structural"] += 1
            classification, gate, price = per_ticker[ticker]

            new_values: dict[str, Any] = {
                "market_cap_mm": classification.market_cap_mm,
                "cap_source": classification.cap_source,
                "cap_band": classification.cap_band,
                "cap_asof": asof,
            }
            updates = {
                field_name: new_values[field_name]
                for field_name in _CAP_FIELDS
                if row[field_name] != new_values[field_name]
            }
            if gate.excluded_error:
                # Fail-closed: gate couldn't examine this name — count the
                # degradation and leave the structural leg untouched rather
                # than either quarantining on a platform failure or treating
                # the row as examined-and-clean.
                counts["tickers_gate_degraded"] = counts.get("tickers_gate_degraded", 0) + 1
            elif gate.quarantined and str(row["status"]) != "QUARANTINE":
                updates["status"] = "QUARANTINE"
                updates["status_reason"] = (
                    gate.reason_string
                    if not row["status_reason"]
                    else f"{gate.reason_string}|{row['status_reason']}"
                )

            focus = _focus_for_run(str(row["source_run_id"]), focus_cache, runs_dir=runs_dir)
            routed_out = _routed_out_of_band(classification, focus)

            if not updates and not routed_out:
                continue

            if updates and not dry_run:
                assignments = ", ".join(f"{field_name} = ?" for field_name in updates)
                conn.execute(
                    f"UPDATE watchlist SET {assignments} WHERE id = ?",
                    tuple(updates.values()) + (int(row["id"]),),
                )
                for field_name, new_value in updates.items():
                    _insert_history(
                        conn,
                        watchlist_id=int(row["id"]),
                        field_name=field_name,
                        old_value=row[field_name],
                        new_value=new_value,
                        source=history_source,
                        source_run_id=str(row["source_run_id"]),
                    )
            if updates:
                counts["rows_changed"] += 1
            if routed_out:
                counts["rows_routed_out_of_band"] += 1
            changed_rows.append(
                {
                    "id": int(row["id"]),
                    "ticker": ticker,
                    "source_run_id": str(row["source_run_id"]),
                    "source_market_cap_focus": focus,
                    "prior_status": str(row["status"]),
                    "new_status": str(updates.get("status", row["status"])),
                    "changes": {
                        field_name: {"old": row[field_name], "new": new_value}
                        for field_name, new_value in updates.items()
                    },
                    "structural_codes": list(gate.triggered_codes),
                    "structural_details": dict(gate.details),
                    "cap": classification.to_dict(),
                    "classification_price": price,
                    "routed_out_of_band": routed_out,
                }
            )
        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    return {
        "as_of_date": asof,
        "dry_run": dry_run,
        "counts": counts,
        "changed_rows": changed_rows,
    }


__all__ = ["backfill_watchlist_cap_and_structural"]
