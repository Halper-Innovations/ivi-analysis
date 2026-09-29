from __future__ import annotations

import json
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.market.price_provider import write_prices_for_run


def write_prices_prewarm_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    fallback_days: int = 5,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    tickers_sorted = sorted({str(symbol).strip().upper() for symbol in tickers if str(symbol).strip()})
    prices_summary = write_prices_for_run(
        tickers=tickers_sorted,
        as_of_date=as_of_date,
        run_id=run_id,
        fallback_days=max(0, int(fallback_days)),
        cfg=cfg,
    )
    ok_tickers = [str(row.get("ticker") or "").upper() for row in (prices_summary.get("rows") or []) if str(row.get("status") or "").upper() == "OK"]
    unknown_tickers = [str(row.get("ticker") or "").upper() for row in (prices_summary.get("rows") or []) if str(row.get("status") or "").upper() != "OK"]
    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "fallback_days": int(fallback_days),
        "tickers_requested": tickers_sorted,
        "tickers_ok": sorted(ok_tickers),
        "tickers_unknown": sorted(unknown_tickers),
        "ok_count": len(ok_tickers),
        "unknown_count": len(unknown_tickers),
        "reason_counts": dict(prices_summary.get("reason_counts") or {}),
        "prices_summary_path": prices_summary.get("summary_path"),
        "generated_at": utc_now_iso(),
        "derived_from": [f"sector.price_coverage.entries[{ticker}]" for ticker in tickers_sorted],
    }
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "prices_prewarm.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["prices_prewarm_path"] = str(out_path)
    return payload
