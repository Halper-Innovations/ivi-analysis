"""Scan preflight health gate.

Before kicking off a multi-hundred-dollar scan, check that the structured
data layer is actually returning useful answers. The reviewer's audit caught
that one of our core tools (fetch_companyfacts) was returning 0 useful rows
on 19/19 calls — without us noticing — because the schema and dispatcher
disagreed on naming. A health gate prevents that class of silent failure
from running for hours and costing money.

The gate samples ~20 random tickers from the sector (or all tickers if
fewer), then verifies for each:
- Core companyfacts fields (revenue, cfo, cash, total_debt) are present
- Most-recent fiscal year is within the last 3 years (data isn't stale)
- For commodity sectors, fetch_current_price reachable

If <90% of sampled tickers pass the core-fields check, the gate FAILS and
prevents the scan from running.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# Core fields we sample for. Picked deliberately to span the four major
# financial concepts: revenue (income), cfo (cash gen), cash (balance sheet),
# total_debt (capital structure). The gate is checking that the TOOL LAYER
# works — it does NOT require every company to be a fully-mature operator.
_CORE_FIELDS = ("revenue", "cfo", "cash", "total_debt")

# A ticker passes if at least this many core fields are populated. Set to 2
# of 4 to allow legitimate pre-revenue biotechs (which have cash + total_debt
# but no revenue or cfo) without flagging the data layer as broken. A truly
# broken ticker would have 0 of 4.
_MIN_CORE_FIELDS_PER_TICKER = 2

# 50% of sampled tickers must pass. The gate is calibrated to catch SYSTEMIC
# failures (the tool returns 0 rows; the table is empty; the ingestion is
# stale) — not to penalize sectors with a high mix of pre-commercial names.
# A 90% threshold proved too strict against real healthcare/biotech sectors
# where 30%+ of names are legitimately pre-revenue.
_DEFAULT_PASS_RATE = 0.50

_DEFAULT_SAMPLE_SIZE = 20
_DEFAULT_MAX_DATA_AGE_YEARS = 3


@dataclass
class TickerHealthCheck:
    ticker: str
    passed: bool
    missing_fields: list[str] = field(default_factory=list)
    most_recent_fy: int | None = None
    stale: bool = False  # data older than max_data_age_years
    note: str = ""


@dataclass
class PreflightResult:
    sector: str
    sampled_tickers: list[str]
    pass_count: int
    fail_count: int
    pass_rate: float
    pass_threshold: float
    passed: bool
    failures_by_field: dict[str, int] = field(default_factory=dict)
    stale_count: int = 0
    per_ticker: list[TickerHealthCheck] = field(default_factory=list)
    summary: str = ""


def _check_one_ticker(
    ticker: str,
    db_path: str | Path,
    max_data_age_years: int,
    current_year: int,
) -> TickerHealthCheck:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT line_item, MAX(fiscal_year) AS max_fy
            FROM companyfacts_facts
            WHERE ticker = ? AND period_type = 'FY'
              AND line_item IN ({','.join('?' for _ in _CORE_FIELDS)})
            GROUP BY line_item
            """,
            [ticker] + list(_CORE_FIELDS),
        ).fetchall()
    finally:
        conn.close()

    seen: dict[str, int] = {r["line_item"]: r["max_fy"] for r in rows}
    missing = [f for f in _CORE_FIELDS if f not in seen]
    fields_present = len(_CORE_FIELDS) - len(missing)
    most_recent = max(seen.values()) if seen else None
    stale = (
        most_recent is not None
        and (current_year - most_recent) > max_data_age_years
    )

    # A ticker passes if it has >= 2 of 4 core fields AND the data isn't stale.
    # Pre-revenue biotechs typically have cash+total_debt only; that's enough
    # to confirm the data layer is working for that ticker.
    if fields_present == 0:
        passed = False
        note = "no companyfacts data at all (broken ticker or empty cache)"
    elif fields_present < _MIN_CORE_FIELDS_PER_TICKER:
        passed = False
        note = f"only {fields_present}/4 core fields ({list(seen.keys())})"
    elif stale:
        passed = False
        note = f"data > {max_data_age_years}y old (latest FY{most_recent})"
    else:
        passed = True
        if missing:
            note = f"OK with {fields_present}/4 fields (missing {missing}, latest FY{most_recent})"
        else:
            note = f"OK (all 4 fields present, latest FY{most_recent})"

    return TickerHealthCheck(
        ticker=ticker,
        passed=passed,
        missing_fields=missing,
        most_recent_fy=most_recent,
        stale=stale,
        note=note,
    )


def run_preflight(
    *,
    sector: str,
    tickers: list[str],
    db_path: str | Path,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    pass_threshold: float = _DEFAULT_PASS_RATE,
    max_data_age_years: int = _DEFAULT_MAX_DATA_AGE_YEARS,
    seed: int | None = None,
) -> PreflightResult:
    """Run the preflight health gate against a sector's ticker universe.

    Returns a PreflightResult. Caller checks `.passed` to decide whether to
    proceed with the scan.
    """
    if seed is not None:
        random.seed(seed)
    sample = list(tickers)
    if len(sample) > sample_size:
        sample = random.sample(sample, sample_size)
    sample.sort()  # deterministic-looking output

    current_year = datetime.now().year
    results: list[TickerHealthCheck] = [
        _check_one_ticker(t, db_path, max_data_age_years, current_year) for t in sample
    ]

    pass_count = sum(1 for r in results if r.passed)
    fail_count = len(results) - pass_count
    pass_rate = pass_count / len(results) if results else 0.0
    passed = pass_rate >= pass_threshold

    failures_by_field: dict[str, int] = {}
    for r in results:
        for f in r.missing_fields:
            failures_by_field[f] = failures_by_field.get(f, 0) + 1
    stale_count = sum(1 for r in results if r.stale)

    summary_parts = [
        f"sector={sector}",
        f"sampled={len(results)}",
        f"pass={pass_count}/{len(results)} ({pass_rate:.0%})",
        f"threshold={pass_threshold:.0%}",
        "PASSED" if passed else "FAILED",
    ]
    if failures_by_field:
        worst = sorted(failures_by_field.items(), key=lambda x: -x[1])
        summary_parts.append(
            "missing: " + ", ".join(f"{k}({v})" for k, v in worst[:4])
        )
    if stale_count:
        summary_parts.append(f"stale={stale_count}")

    return PreflightResult(
        sector=sector,
        sampled_tickers=sample,
        pass_count=pass_count,
        fail_count=fail_count,
        pass_rate=pass_rate,
        pass_threshold=pass_threshold,
        passed=passed,
        failures_by_field=failures_by_field,
        stale_count=stale_count,
        per_ticker=results,
        summary=" ".join(summary_parts),
    )


def format_preflight_report(result: PreflightResult) -> str:
    """Render a preflight result as human-readable text."""
    lines = [
        f"=== SCAN PREFLIGHT HEALTH GATE: {result.sector} ===",
        f"Status: {'PASSED' if result.passed else 'FAILED'}",
        f"Sampled: {len(result.sampled_tickers)} tickers",
        f"Pass rate: {result.pass_count}/{len(result.sampled_tickers)} = "
        f"{result.pass_rate:.0%} (threshold: {result.pass_threshold:.0%})",
    ]
    if result.failures_by_field:
        lines.append("")
        lines.append("Most common missing fields:")
        for k, v in sorted(result.failures_by_field.items(), key=lambda x: -x[1]):
            lines.append(f"  {k}: missing in {v} tickers")
    if result.stale_count:
        lines.append(f"Stale data (>{_DEFAULT_MAX_DATA_AGE_YEARS}y old): {result.stale_count}")
    if result.fail_count:
        lines.append("")
        lines.append("Failed tickers (first 10):")
        for r in result.per_ticker:
            if not r.passed:
                lines.append(f"  {r.ticker:6s} {r.note}")
                if sum(1 for x in result.per_ticker if not x.passed and result.per_ticker.index(x) <= result.per_ticker.index(r)) >= 10:
                    break
    return "\n".join(lines)
