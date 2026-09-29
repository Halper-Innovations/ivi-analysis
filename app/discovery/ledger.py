from __future__ import annotations

from typing import Any

from app.db import get_db, init_db, utc_now_iso
from app.discovery.runner import load_discovery_candidates
from app.valuation.price_provider import get_default_provider


def discovery_ledger_init() -> dict[str, Any]:
    init_db()
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM discovery_outcomes").fetchone()
    return {"status": "ok", "existing_rows": int(row["n"] or 0)}


def discovery_ledger_update(run_id: str) -> dict[str, Any]:
    provider = get_default_provider()
    with get_db() as conn:
        run_row = conn.execute(
            """
            SELECT run_as_of_date
            FROM discovery_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if not run_row:
            return {"status": "error", "run_id": run_id, "message": "discovery run not found", "rows_written": 0}
        run_as_of_date = str(run_row["run_as_of_date"])

    candidates = load_discovery_candidates(run_id)
    if not candidates:
        return {"status": "ok", "run_id": run_id, "rows_written": 0, "message": "no candidates"}

    prepared_rows: list[tuple[str, float | None, str]] = []
    for row in candidates:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        quote = provider.get_quote(ticker, run_as_of_date)
        entry_price = quote.price if isinstance(quote.price, (int, float)) else None
        price_source = quote.provider if quote.provider else "UNKNOWN"
        prepared_rows.append((ticker, entry_price, price_source))

    rows_written = 0
    with get_db() as conn:
        for ticker, entry_price, price_source in prepared_rows:
            conn.execute(
                """
                INSERT INTO discovery_outcomes(
                    ticker, discovery_run_id, deep_run_id, run_as_of_date,
                    entry_price, price_source, forward_return_30d, forward_return_90d, forward_return_180d, created_at
                ) VALUES(?, ?, NULL, ?, ?, ?, NULL, NULL, NULL, ?)
                ON CONFLICT(ticker, discovery_run_id) DO UPDATE SET
                    run_as_of_date=excluded.run_as_of_date,
                    entry_price=excluded.entry_price,
                    price_source=excluded.price_source
                """,
                (
                    ticker,
                    run_id,
                    run_as_of_date,
                    entry_price,
                    price_source,
                    utc_now_iso(),
                ),
            )
            rows_written += 1

    return {
        "status": "ok",
        "run_id": run_id,
        "run_as_of_date": run_as_of_date,
        "rows_written": rows_written,
        "price_provider": provider.provider_name,
        "returns_status": "placeholders_only",
    }
