"""Read-only interim marks of classic's recorded picks, including withdrawn rows.

No resolver, ledger mutation, provider, or network is reachable from build_report.
The ledgers' contemporaneous quotes are treated as raw.
Unlabeled historical prices cannot silently become raw benchmark anchors.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from datetime import date
from pathlib import Path
from statistics import mean, median
from typing import Any

from app.calibration.decision_ledger import select_benchmark_symbol

SOURCE_LABELS = {
    "BUY": "BUY decisions",
    "BUY_AT_LIMIT": "BUY_AT_LIMIT recommendations",
    "ACTIONABLE": "ACTIONABLE grades",
}


def _day(value: Any) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        return ""


def _positive(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value > 0:
            return float(value)
    return None


def _return(entry: float, exit_: float) -> float:
    return (exit_ - entry) / entry * 100.0


def _latest(conn: sqlite3.Connection, ticker: str, on: str) -> dict[str, Any]:
    # The existing ticker/provider/date index bounds this to one ticker's history.
    # Date wins over basis: an older raw price must not replace a newer unknown one.
    row = conn.execute(
        """SELECT id, ticker, provider, as_of_date, price, currency, price_basis, fetched_at
           FROM price_quotes WHERE ticker = ? AND status = 'OK' AND as_of_date <= ?
           ORDER BY as_of_date DESC, (price_basis = 'UNADJUSTED') DESC,
                    fetched_at DESC, provider, id DESC LIMIT 1""",
        (ticker, on),
    ).fetchone()
    return dict(row) if row else {}


def _split(conn: sqlite3.Connection, ticker: str, entry: str, exit_: str) -> bool:
    if not entry or not exit_:
        return False
    return (
        conn.execute(
            """SELECT 1 FROM price_quotes WHERE ticker = ? AND as_of_date <= ? AND (
               (split_effective_date > ? AND split_effective_date <= ?)
               OR (as_of_date >= ? AND split_adjustment_factor IS NOT NULL
                   AND split_adjustment_factor != 1)) LIMIT 1""",
            (ticker, exit_, entry, exit_, entry),
        ).fetchone()
        is not None
    )


def _load_picks(conn: sqlite3.Connection, today: str) -> list[dict[str, Any]]:
    watchlist = [dict(r) for r in conn.execute("SELECT * FROM watchlist")]
    latest = {}
    for row in sorted(watchlist, key=lambda r: (r["added_at"] or "", r["id"])):
        if _day(row["added_at"]) <= today:
            latest[row["ticker"]] = row
    sources = [
        (
            "BUY",
            [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM ticker_outcomes "
                    "WHERE run_id LIKE 'autonomous_sector_%' AND decision = 'BUY'"
                )
            ],
        ),
        (
            "BUY_AT_LIMIT",
            [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM recommendation_ledger WHERE recommendation_type = 'BUY_AT_LIMIT'"
                )
            ],
        ),
        ("ACTIONABLE", [r for r in watchlist if r["conviction_grade"] == "ACTIONABLE"]),
    ]
    picks = []
    for source, rows in sources:
        for raw in rows:
            entry_at = (
                raw.get(
                    {"BUY": "entry_date", "BUY_AT_LIMIT": "staked_at", "ACTIONABLE": "added_at"}[
                        source
                    ]
                )
                or ""
            )
            entry_date = _day(entry_at)
            if entry_date and entry_date > today:
                continue
            current = latest.get(raw["ticker"], {})
            wl = raw if source == "ACTIONABLE" else current
            cap = raw.get("cap_category") if source == "BUY" else None
            cap_source = "ticker_outcomes.cap_category" if cap else "watchlist.cap_band"
            cap = cap or wl.get("cap_band") or "unknown"
            picks.append(
                {
                    "ticker": raw["ticker"],
                    "source": source,
                    "source_row_id": raw["id"],
                    "entry_at": entry_at,
                    "entry_date": entry_date,
                    "entry_price": raw.get(
                        {
                            "BUY": "entry_price",
                            "BUY_AT_LIMIT": "trigger_price",
                            "ACTIONABLE": "current_price_at_addition",
                        }[source]
                    ),
                    "entry_price_source": (
                        raw.get("entry_price_source")
                        if source == "BUY"
                        else raw.get("trigger_price_source")
                        if source == "BUY_AT_LIMIT"
                        else "watchlist.current_price_at_addition"
                    ),
                    "buy_target": raw.get(
                        "target_price" if source == "BUY_AT_LIMIT" else "buy_price_target"
                    ),
                    "benchmark_symbol": raw.get("benchmark_symbol") or select_benchmark_symbol(cap),
                    "horizon_days": raw.get("horizon_days") if source == "BUY" else None,
                    "cap_band": cap,
                    "cap_band_source": cap_source if cap != "unknown" else "unknown",
                    "status": wl.get("status"),
                    "status_reason": wl.get("status_reason"),
                    "current_watchlist_status": current.get("status"),
                    "conviction_grade": raw.get("conviction_grade") or raw.get("grade"),
                    "conviction_source": wl.get("conviction_source"),
                    "source_sector": raw.get("source_sector") or wl.get("source_sector"),
                    "event_flag": wl.get("event_pending"),
                }
            )
    # Preserve every source record, but only the latest intraday record of a dated
    # statement receives weight. Missing dates get their own key, not a false merge.
    seen = set()
    for row in sorted(picks, key=lambda r: (r["entry_at"], r["source_row_id"]), reverse=True):
        key = (row["ticker"], row["source"], row["entry_date"] or row["source_row_id"])
        row["counted"] = key not in seen
        seen.add(key)
    return sorted(
        picks,
        key=lambda r: (
            list(SOURCE_LABELS).index(r["source"]),
            r["entry_date"],
            r["ticker"],
            r["source_row_id"],
        ),
    )


def build_report(
    conn: sqlite3.Connection,
    *,
    today: date,
    refreshed_prices: dict | None = None,
    refresh_note: str = "Network requests: 0 (local prices only).",
) -> dict[str, Any]:
    """Read one database snapshot. Caller owns the read-only connection/transaction."""
    on = today.isoformat()
    rows = _load_picks(conn, on)
    quotes: dict[tuple[str, str], dict] = {}

    def quote(ticker: str, requested: str) -> dict:
        key = (ticker, requested)
        if key not in quotes:
            local = _latest(conn, ticker, requested)
            refreshed = (refreshed_prices or {}).get(key)
            if refreshed is not None:
                if refreshed.as_of_date <= requested and (
                    refreshed.as_of_date >= local.get("as_of_date", "")
                ):
                    local = {
                        "ticker": ticker,
                        "as_of_date": refreshed.as_of_date,
                        "price": refreshed.price,
                        "provider": refreshed.provider,
                        "price_basis": refreshed.price_basis,
                        "currency": "USD",
                        "fetched_at": "benchmark refresh",
                    }
            quotes[key] = local
        return quotes[key]

    for row in rows:
        ticker, entry = row["ticker"], row["entry_date"]
        latest = quote(ticker, on)
        flags = []
        row.update(
            {
                "latest_close": latest.get("price"),
                "quote_date": latest.get("as_of_date"),
                "quote_fetched_at": latest.get("fetched_at"),
                "quote_provider": latest.get("provider"),
                "price_basis": latest.get("price_basis") or "UNKNOWN",
                "days_held": (today - date.fromisoformat(entry)).days if entry else None,
                "return_pct": None,
                "benchmark_return_pct": None,
                "excess_return_pct": None,
                "distance_to_buy_target_pct": None,
            }
        )
        exit_date = row["quote_date"] or on
        entry_price, exit_price = _positive(row["entry_price"]), _positive(row["latest_close"])
        # A split may already be recorded even when its new quote is unusable.
        # Check through the report date before marking today's share entitlement
        # or comparing an older quote with the currently stored buy target.
        split = _split(conn, ticker, entry, on)
        if split:
            flags.append("SPLIT_CHECK")
        if not row["counted"]:
            flags.append("DUPLICATE_NOT_COUNTED")
        if not entry:
            flags.append("ENTRY_DATE_MISSING")
        if not entry_price:
            flags.append("ENTRY_PRICE_MISSING")
        if not exit_price:
            flags.append("PRICE_MISSING")
        if latest and row["price_basis"] != "UNADJUSTED":
            flags.append(
                "PRICE_BASIS_UNKNOWN" if row["price_basis"] == "UNKNOWN" else "PRICE_BASIS_MISMATCH"
            )
        if latest and latest.get("currency") != "USD":
            flags.append("CURRENCY_CHECK")
        if latest and exit_date < on:
            flags.append("PRICE_STALE")
        if latest and entry and exit_date < entry:
            flags.append("PRICE_BEFORE_ENTRY")
        if latest and exit_date == on:
            flags.append("CLOSE_UNCONFIRMED")
        raw_ok = row["price_basis"] == "UNADJUSTED" and latest.get("currency") == "USD"
        if entry and entry_price and exit_price and raw_ok and not split and exit_date >= entry:
            row["return_pct"] = _return(entry_price, exit_price)
        target = _positive(row["buy_target"])
        if target and exit_price and raw_ok and not split:
            row["distance_to_buy_target_pct"] = _return(exit_price, target)
        symbol = row["benchmark_symbol"]
        start = quote(symbol, entry) if entry else {}
        end = quote(symbol, exit_date)
        row.update(
            {
                "benchmark_entry_date": start.get("as_of_date"),
                "benchmark_exit_date": end.get("as_of_date"),
                "benchmark_entry_price": start.get("price"),
                "benchmark_exit_price": end.get("price"),
                "benchmark_entry_provider": start.get("provider"),
                "benchmark_exit_provider": end.get("provider"),
                "benchmark_entry_basis": start.get("price_basis") or "UNKNOWN",
                "benchmark_exit_basis": end.get("price_basis") or "UNKNOWN",
            }
        )
        if not start or not end:
            flags.append("BENCHMARK_MISSING")
        if end and end["as_of_date"] < exit_date:
            flags.append("BENCHMARK_STALE")
        if (
            start
            and entry
            and (date.fromisoformat(entry) - date.fromisoformat(start["as_of_date"])).days > 7
        ):
            flags.append("BENCHMARK_ENTRY_STALE")
        if start and start["as_of_date"] != entry:
            flags.append("BENCHMARK_ENTRY_PRIOR_DATE")
        bench_raw = all(q.get("price_basis") == "UNADJUSTED" for q in (start, end))
        if not bench_raw:
            flags.append(
                "BENCHMARK_BASIS_UNKNOWN"
                if any(not q.get("price_basis") for q in (start, end))
                else "BENCHMARK_BASIS_MISMATCH"
            )
        if start and end and any(q.get("currency") != "USD" for q in (start, end)):
            flags.append("BENCHMARK_CURRENCY_CHECK")
            bench_raw = False
        same_provider = start.get("provider") == end.get("provider")
        if start and end and not same_provider:
            flags.append("BENCHMARK_PROVIDER_MISMATCH")
        bench_split = _split(conn, symbol, start.get("as_of_date", ""), end.get("as_of_date", ""))
        if bench_split:
            flags.append("BENCHMARK_SPLIT_CHECK")
        if (
            entry
            and start
            and end
            and bench_raw
            and same_provider
            and not bench_split
            and _positive(start.get("price"))
            and _positive(end.get("price"))
            and end["as_of_date"] >= entry
        ):
            row["benchmark_return_pct"] = _return(start["price"], end["price"])
        # A stale benchmark return may be displayed with its actual date; it must
        # never masquerade as excess over the stock's later marking window.
        if (
            row["return_pct"] is not None
            and row["benchmark_return_pct"] is not None
            and end["as_of_date"] == exit_date
            and start["as_of_date"] == entry
        ):
            row["excess_return_pct"] = row["return_pct"] - row["benchmark_return_pct"]
        row["flags"] = ";".join(flags) or "OK"
    journal = [
        dict(r)
        for r in conn.execute(
            "SELECT ticker, decision, COALESCE(entry_date, as_of_date) AS entry_date "
            "FROM ticker_outcomes WHERE run_id = 'journal_live' ORDER BY as_of_date, id"
        )
    ]
    holdings = conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]
    return {
        "as_of_date": on,
        "rows": rows,
        "journal": journal,
        "holdings_count": holdings,
        "refresh_note": refresh_note,
    }


def refresh_dates(conn: sqlite3.Connection, today: date) -> set[str]:
    """All requested entry and mark dates; no provider is constructed here."""
    dates = {today.isoformat()}
    for row in _load_picks(conn, today.isoformat()):
        if row["entry_date"]:
            dates.add(row["entry_date"])
        quote = _latest(conn, row["ticker"], today.isoformat())
        if quote:
            dates.add(quote["as_of_date"])
    return dates


def summaries(report: dict) -> list[dict]:
    rows = [r for r in report["rows"] if r["counted"]]
    result = []
    for dimension, groups in (
        ("source", list(SOURCE_LABELS)),
        ("cap_band", sorted({r["cap_band"] for r in rows})),
    ):
        for group in groups:
            selected = [r for r in rows if r[dimension] == group]
            returns = [r["return_pct"] for r in selected if r["return_pct"] is not None]
            excess = [
                r["excess_return_pct"] for r in selected if r["excess_return_pct"] is not None
            ]
            result.append(
                {
                    "group_by": dimension,
                    "group": group,
                    "count": len(selected),
                    "priced_count": len(returns),
                    "mean_return_pct": mean(returns) if returns else None,
                    "median_return_pct": median(returns) if returns else None,
                    "excess_count": len(excess),
                    "positive_excess_pct": 100 * sum(x > 0 for x in excess) / len(excess)
                    if excess
                    else None,
                }
            )
    return result


def dollar_line(report: dict) -> str:
    buys = [r for r in report["rows"] if r["source"] == "BUY" and r["counted"]]
    priced = [r for r in buys if r["return_pct"] is not None]
    value = sum(1000 * (1 + r["return_pct"] / 100) for r in priced)
    if not buys:
        return "No BUY decisions are recorded; no hypothetical investment to mark."
    if len(priced) != len(buys):
        return (
            f"$1,000 in each BUY decision at entry: total value cannot be established "
            f"({len(priced)}/{len(buys)} priced; priced subset worth ${value:,.2f})."
        )
    return (
        f"$1,000 in each BUY decision at entry is worth ${value:,.2f} today "
        f"(${len(buys) * 1000:,.2f} invested; hypothetical quote marks, excluding distributions)."
    )


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def _money(value: Any) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    )


def render_markdown(report: dict) -> str:
    lines = [
        f"# IVI buy-pick marks — {report['as_of_date']}",
        "",
        dollar_line(report),
        "",
        "These are hypothetical marks of dated statements, not executed positions. "
        "BUY decisions come from autonomous sector runs; BUY_AT_LIMIT uses the recorded "
        "trigger quote (not an assumed fill); ACTIONABLE uses the watchlist addition quote. "
        "Stored BUY horizons are shown; these are interim, not resolved, returns.",
        "",
        "Basis: raw ledger quote to an explicitly UNADJUSTED USD quote; no dividends, "
        "fees, taxes or reinvestment. Latest quote means the latest status=OK row dated "
        "on or before this report. A same-day quote has CLOSE_UNCONFIRMED because this "
        "table does not certify a closing auction. Unknown or adjusted bases are not mixed. "
        "Non-identity split factors or split dates since entry require SPLIT_CHECK; those "
        "returns and target distances are unavailable until share entitlement is verified.",
        "",
        "Benchmark anchors use the latest available quote on or before entry and stock "
        "mark dates. Their actual dates, prices and bases appear below. A stale benchmark "
        "or unknown basis cannot establish same-window excess. Earlier entry anchors "
        "(including weekends) also withhold excess because the ledger quote has no "
        "confirmed matching market date. Benchmark returns require the same provider. "
        "Excess is stock return "
        "minus benchmark return in percentage points. Days are calendar days since the "
        "recorded entry; target distance is (buy target / latest quote - 1) × 100.",
        "",
        "Every source row is retained. Summaries count one ticker/source/calendar entry "
        "date, using the latest entry timestamp then row ID for duplicates. Duplicate "
        "rows have Counted=no. Sources overlap and are not one combined portfolio. "
        "Cap bands are stored outcome categories where available, otherwise current "
        "watchlist metadata. Status and targets are current stored values, not reconstructed "
        "historical versions. Missing returns stay in counts; means, medians and positive "
        "excess shares use only their displayed available denominators.",
        "",
        report["refresh_note"],
        "",
    ]
    for source, label in SOURCE_LABELS.items():
        rows = [r for r in report["rows"] if r["source"] == source]
        lines += [
            f"## {label} ({len(rows)} source rows)",
            "",
            _table(
                [
                    "Ticker / row",
                    "Entry date",
                    "Entry",
                    "Buy target",
                    "Latest quote / date",
                    "Days",
                    "Return",
                    "Benchmark / window",
                    "Benchmark return",
                    "Excess (pp)",
                    "To target",
                    "Cap",
                    "Status / current ticker status",
                    "Event",
                    "Basis",
                    "Counted",
                    "Flags",
                ],
                [
                    [
                        f"{r['ticker']} / {r['source_row_id']}",
                        r["entry_date"],
                        _money(r["entry_price"]),
                        _money(r["buy_target"]),
                        f"{_money(r['latest_close'])} / {r['quote_date'] or 'n/a'}",
                        r["days_held"],
                        _pct(r["return_pct"]),
                        f"{r['benchmark_symbol']} / {r['benchmark_entry_date'] or 'n/a'} → {r['benchmark_exit_date'] or 'n/a'}",
                        _pct(r["benchmark_return_pct"]),
                        "n/a"
                        if r["excess_return_pct"] is None
                        else f"{r['excess_return_pct']:+.2f}",
                        _pct(r["distance_to_buy_target_pct"]),
                        r["cap_band"],
                        f"{r['status'] or 'n/a'} / {r['current_watchlist_status'] or 'n/a'}",
                        r["event_flag"],
                        r["price_basis"],
                        "yes" if r["counted"] else "no",
                        r["flags"],
                    ]
                    for r in rows
                ],
            ),
            "",
        ]
    lines += [
        "## Equal-weight summaries",
        "",
        _table(
            [
                "Grouped by",
                "Group",
                "Picks",
                "Priced",
                "Mean",
                "Median",
                "Excess available",
                "Positive excess",
            ],
            [
                [
                    r["group_by"],
                    r["group"],
                    r["count"],
                    r["priced_count"],
                    _pct(r["mean_return_pct"]),
                    _pct(r["median_return_pct"]),
                    r["excess_count"],
                    _pct(r["positive_excess_pct"]),
                ]
                for r in summaries(report)
            ],
        ),
        "",
        "## Provenance and status reasons",
        "",
        _table(
            [
                "Source / ticker / row",
                "Entry timestamp / source",
                "Grade / conviction source",
                "Sector / cap source",
                "Horizon days",
                "Quote provider / fetched",
                "Benchmark entry / basis / provider",
                "Benchmark exit / basis / provider",
                "Status reason",
            ],
            [
                [
                    f"{r['source']} / {r['ticker']} / {r['source_row_id']}",
                    f"{r['entry_at']} / {r['entry_price_source']}",
                    f"{r['conviction_grade'] or 'n/a'} / {r['conviction_source'] or 'n/a'}",
                    f"{r['source_sector'] or 'n/a'} / {r['cap_band_source']}",
                    r["horizon_days"],
                    f"{r['quote_provider']} / {r['quote_fetched_at']}",
                    f"{_money(r['benchmark_entry_price'])} / {r['benchmark_entry_basis']} / {r['benchmark_entry_provider']}",
                    f"{_money(r['benchmark_exit_price'])} / {r['benchmark_exit_basis']} / {r['benchmark_exit_provider']}",
                    r["status_reason"],
                ]
                for r in report["rows"]
            ],
        ),
        "",
        "## Journaled live decisions",
        "",
        _table(
            ["Ticker", "Date", "Decision"],
            [[r["ticker"], r["entry_date"], r["decision"]] for r in report["journal"]],
        ),
        "",
        f"Recorded holdings: {report['holdings_count']}. "
        + (
            "No holdings or live BUY decisions are recorded."
            if report["holdings_count"] == 0
            and not any(r["decision"] == "BUY" for r in report["journal"])
            else "The journal and holdings above describe recorded activity."
        ),
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict, output_root: Path) -> tuple[Path, Path]:
    directory = output_root / "picks" / report["as_of_date"]
    directory.mkdir(parents=True, exist_ok=True)
    markdown = directory / "marks.md"
    csv_path = directory / "marks.csv"
    markdown.write_text(render_markdown(report), encoding="utf-8")
    # Separate machine-readable companions keep summary/journal records distinct
    # from pick rows while exporting every section of the terminal report.
    datasets = [
        (csv_path, report["rows"], ["ticker", "source", "entry_date"]),
        (directory / "summaries.csv", summaries(report), ["group_by", "group", "count"]),
        (directory / "journal.csv", report["journal"], ["ticker", "entry_date", "decision"]),
        (
            directory / "report.csv",
            [
                {
                    "as_of_date": report["as_of_date"],
                    "dollar_line": dollar_line(report),
                    "holdings_count": report["holdings_count"],
                    "refresh_note": report["refresh_note"],
                    "source_row_count": len(report["rows"]),
                    "deduplicated_count": sum(r["counted"] for r in report["rows"]),
                }
            ],
            [],
        ),
    ]
    for path, rows, empty_fields in datasets:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else empty_fields)
            writer.writeheader()
            writer.writerows(rows)
    return markdown, csv_path
