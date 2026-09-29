"""Issuer-bound, point-in-time CompanyFacts repair for sector pipeline v2.

The legacy ``facts_writer`` is deliberately ticker/global-database oriented.
The autonomous v2 repair queue needs a narrower contract: fetch one SEC
issuer, write only facts that were filed by the fixed scan date, and honor the
database path supplied by the run.  Foreign IFRS and non-USD normalization are
explicitly outside this repair and therefore return precise terminal
``NEEDS_DATA`` reasons instead of attempting an implicit conversion.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Callable

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.ingest.companyfacts import normalize_annual_facts_from_raw
from app.market.company_facts_provider import fetch_company_facts
from app.util.financial_data_access import normalize_cik


CompanyFactsFetcher = Callable[..., dict[str, Any]]

# The companyfacts_facts.written_by marker for point-in-time repair rows: the facts writer's
# refresh purge (app/ingest/facts_writer.py) deletes only rows it wrote itself.
WRITTEN_BY = "companyfacts_repair"

_TRANSIENT_FETCH_REASONS = {
    "BUDGET_EXHAUSTED",
    "EXCEPTION",
    "FETCH_5XX",
}


def _raw_namespaces(payload: dict[str, Any]) -> dict[str, Any]:
    facts = payload.get("facts")
    return facts if isinstance(facts, dict) else {}


def _currency_units(namespace: Any) -> set[str]:
    units: set[str] = set()
    if not isinstance(namespace, dict):
        return units
    for node in namespace.values():
        if not isinstance(node, dict):
            continue
        raw_units = node.get("units")
        if not isinstance(raw_units, dict):
            continue
        for unit in raw_units:
            token = str(unit or "").strip().upper()
            if len(token) == 3 and token.isalpha():
                units.add(token)
    return units


def _foreign_gap_reason(raw: dict[str, Any]) -> str | None:
    namespaces = _raw_namespaces(raw)
    ifrs_facts = namespaces.get("ifrs-full")
    us_gaap_facts = namespaces.get("us-gaap")
    if isinstance(ifrs_facts, dict) and ifrs_facts and not (
        isinstance(us_gaap_facts, dict) and us_gaap_facts
    ):
        return "IFRS_FACTS_UNSUPPORTED"
    currencies: set[str] = set()
    for namespace in namespaces.values():
        currencies.update(_currency_units(namespace))
    if currencies and "USD" not in currencies:
        return "NON_USD_FACTS_UNNORMALIZED"
    return None


def _safe_filed_date(value: Any, *, as_of_date: str) -> str | None:
    token = str(value or "").strip()[:10]
    try:
        filed = date.fromisoformat(token)
        cutoff = date.fromisoformat(str(as_of_date)[:10])
    except ValueError:
        return None
    return token if filed <= cutoff else None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _record_vintage(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    issuer_cik: str,
    fact: dict[str, Any],
    recorded_at: str,
) -> bool:
    required = {
        "ticker",
        "fiscal_year",
        "period_type",
        "period_end",
        "line_item",
        "value",
        "units",
        "filed_date",
        "form",
        "accession",
        "recorded_at",
    }
    available = _table_columns(conn, "companyfacts_vintages")
    if not required <= available:
        return False
    columns = [
        "ticker",
        "fiscal_year",
        "period_type",
        "period_end",
        "line_item",
        "value",
        "units",
        "filed_date",
        "form",
        "accession",
        "recorded_at",
    ]
    values: list[Any] = [
        ticker,
        fact["fiscal_year"],
        fact.get("period_type", "FY"),
        fact.get("period_end"),
        fact["line_item"],
        fact.get("value"),
        fact.get("units"),
        fact.get("filed_date"),
        fact.get("form"),
        fact.get("accession"),
        recorded_at,
    ]
    if "issuer_cik" in available:
        columns.append("issuer_cik")
        values.append(str(issuer_cik).zfill(10))
    if "source_url" in available:
        columns.append("source_url")
        values.append(fact.get("source_url"))
    cursor = conn.execute(
        f"INSERT OR IGNORE INTO companyfacts_vintages({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        values,
    )
    return bool(cursor.rowcount)


def _persist_fact(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    fact: dict[str, Any],
    fetched_at: str,
) -> bool:
    required = {
        "ticker",
        "fiscal_year",
        "period_type",
        "period_end",
        "line_item",
        "value",
        "units",
        "source_url",
        "fetched_at",
        "filed_date",
        "form",
        "accession",
    }
    columns = _table_columns(conn, "companyfacts_facts")
    if not required <= columns:
        return False
    # The concepts a derived row was formed from (as the facts writer stores them), and the
    # marker that keeps the facts writer's refresh purge off this point-in-time row. Both
    # only where the database has the columns.
    optional = {
        name: value
        for name, value in (
            ("source_tags", fact.get("source_tags")),
            ("written_by", WRITTEN_BY),
        )
        if name in columns
    }
    extra_columns = "".join(f", {name}" for name in optional)
    extra_marks = ", ?" * len(optional)
    extra_updates = "".join(f",\n            {name}=excluded.{name}" for name in optional)
    cursor = conn.execute(
        f"""
        INSERT INTO companyfacts_facts(
            ticker, fiscal_year, period_type, period_end, line_item, value,
            units, source_url, fetched_at, filed_date, form, accession{extra_columns}
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?{extra_marks})
        ON CONFLICT(ticker, fiscal_year, period_type, line_item) DO UPDATE SET
            period_end=excluded.period_end,
            value=excluded.value,
            units=excluded.units,
            source_url=excluded.source_url,
            fetched_at=excluded.fetched_at,
            filed_date=excluded.filed_date,
            form=excluded.form,
            accession=excluded.accession{extra_updates}
        WHERE NULLIF(TRIM(companyfacts_facts.filed_date), '') IS NOT NULL
          AND companyfacts_facts.filed_date <= excluded.filed_date
        """,
        (
            ticker,
            fact["fiscal_year"],
            fact.get("period_type", "FY"),
            fact.get("period_end"),
            fact["line_item"],
            fact.get("value"),
            fact.get("units"),
            fact.get("source_url"),
            fetched_at,
            fact.get("filed_date"),
            fact.get("form"),
            fact.get("accession"),
            *optional.values(),
        ),
    )
    return bool(cursor.rowcount)


def repair_issuer_annual_companyfacts(
    ticker: str,
    *,
    issuer_cik: str | None,
    as_of_date: str,
    db_path: str | Path,
    cfg: AppConfig | None = None,
    storage_ticker: str | None = None,
    years_back: int = 10,
    fetcher: CompanyFactsFetcher | None = None,
) -> dict[str, Any]:
    """Fetch and persist fixed-as-of US-GAAP annual facts for one issuer.

    The returned ``terminal`` flag describes source exhaustion, not business
    quality.  Transient SEC/network failures stay resumable.  A successful
    fetch with unsupported IFRS/non-USD facts is terminal ``NEEDS_DATA`` for
    this deliberately US-GAAP-only repair slice.
    """

    upper = str(storage_ticker or ticker).strip().upper()
    cik = normalize_cik(issuer_cik)
    if cik is None:
        return {
            "outcome": "NEEDS_DATA",
            "reason_code": "ISSUER_CIK_UNRESOLVED",
            "terminal": False,
            "actions": [],
        }

    resolved_cfg = cfg or get_config()
    provider = fetcher or fetch_company_facts
    response = provider(cik, cfg=resolved_cfg)
    status = str(response.get("status") or "MISSING").strip().upper()
    reason = str(response.get("reason_code") or "EXCEPTION").strip().upper()
    if status != "OK" or not isinstance(response.get("companyfacts"), dict):
        return {
            "outcome": "NEEDS_DATA",
            "reason_code": f"COMPANYFACTS_{reason}",
            "reason_detail": response.get("reason_detail"),
            "terminal": reason not in _TRANSIENT_FETCH_REASONS,
            "network_attempted": bool(response.get("network_attempted")),
            "attempts_made": int(response.get("attempts_made") or 0),
            "source_url": response.get("source_url"),
            "actions": [],
        }

    raw = response["companyfacts"]
    exposed_ciks = [
        response.get("issuer_cik"),
        response.get("cik"),
        raw.get("cik") if isinstance(raw, dict) else None,
    ]
    conflicting_ciks = {
        exposed
        for value in exposed_ciks
        if value is not None
        and (exposed := normalize_cik(value)) is not None
        and exposed != cik
    }
    if conflicting_ciks:
        return {
            "outcome": "NEEDS_DATA",
            "reason_code": "COMPANYFACTS_ISSUER_MISMATCH",
            "terminal": True,
            "requested_issuer_cik": cik,
            "observed_issuer_ciks": sorted(conflicting_ciks),
            "source_url": response.get("source_url"),
            "actions": [],
        }
    foreign_reason = _foreign_gap_reason(raw)
    if foreign_reason is not None:
        return {
            "outcome": "NEEDS_DATA",
            "reason_code": foreign_reason,
            "terminal": True,
            "source_url": response.get("source_url"),
            "source_resolution": response.get("source_resolution"),
            "actions": [],
        }

    try:
        normalized = normalize_annual_facts_from_raw(
            raw,
            cik=str(cik).zfill(10),
            years_back=max(1, int(years_back)),
            filed_as_of=as_of_date,
        )
    except Exception as exc:  # noqa: BLE001 - evidence result, not run abort
        return {
            "outcome": "NEEDS_DATA",
            "reason_code": "COMPANYFACTS_NORMALIZATION_ERROR",
            "reason_detail": f"{type(exc).__name__}: {exc}",
            "terminal": True,
            "source_url": response.get("source_url"),
            "actions": [],
        }

    visible: list[dict[str, Any]] = []
    undated = 0
    future = 0
    cutoff = date.fromisoformat(str(as_of_date)[:10])
    for raw_fact in normalized:
        fact = dict(raw_fact)
        filed_token = str(fact.get("filed_date") or "").strip()[:10]
        filed = _safe_filed_date(filed_token, as_of_date=as_of_date)
        if filed is None:
            try:
                if filed_token and date.fromisoformat(filed_token) > cutoff:
                    future += 1
                else:
                    undated += 1
            except ValueError:
                undated += 1
            continue
        fact["filed_date"] = filed
        visible.append(fact)

    if not visible:
        return {
            "outcome": "NEEDS_DATA",
            "reason_code": (
                "COMPANYFACTS_ONLY_FUTURE_FILINGS"
                if future and not undated
                else "COMPANYFACTS_NO_DATED_US_GAAP_ANNUAL_FACTS"
            ),
            "terminal": True,
            "normalized_rows": len(normalized),
            "future_rows_rejected": future,
            "undated_rows_rejected": undated,
            "source_url": response.get("source_url"),
            "actions": [],
        }

    now = utc_now_iso()
    rows_written = 0
    vintages_written = 0
    path = Path(db_path)
    conn = sqlite3.connect(str(path))
    try:
        for fact in visible:
            vintages_written += int(
                _record_vintage(
                    conn,
                    ticker=upper,
                    issuer_cik=cik,
                    fact=fact,
                    recorded_at=now,
                )
            )
            rows_written += int(
                _persist_fact(
                    conn,
                    ticker=upper,
                    fact=fact,
                    fetched_at=now,
                )
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "outcome": "FETCHED",
        "reason_code": None,
        "terminal": False,
        "issuer_cik": cik,
        "storage_ticker": upper,
        "normalized_rows": len(normalized),
        "visible_rows": len(visible),
        "future_rows_rejected": future,
        "undated_rows_rejected": undated,
        "rows_written": rows_written,
        "vintages_written": vintages_written,
        "source_url": response.get("source_url"),
        "source_resolution": response.get("source_resolution"),
        "network_attempted": bool(response.get("network_attempted")),
        "attempts_made": int(response.get("attempts_made") or 0),
        "actions": ["COMPANYFACTS_FETCHED_AND_NORMALIZED"],
    }


__all__ = ["repair_issuer_annual_companyfacts"]
