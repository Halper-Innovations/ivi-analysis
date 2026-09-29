from __future__ import annotations

import logging
import os
import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

VALID_CACHED_FILING_STATUSES = ("OK", "parsed")

_PIT_BASIS_LOGGED = False


def pit_filed_asof_enabled() -> bool:
    """PIT foundation flag (default OFF).

    When true, companyfacts as-of reads gate on filed_date <= as_of (90-day
    lag fallback for NULL-filed rows) instead of period_end alone. Requires
    a migrated DB (`ivi init-db`) and the PIT backfill for meaningful
    coverage. Turning it on is a deliberate operator decision, not a default.
    """
    return os.getenv("VOE_PIT_FILED_ASOF", "").strip().lower() == "true"


def _log_pit_basis_once() -> None:
    global _PIT_BASIS_LOGGED
    if not _PIT_BASIS_LOGGED:
        _PIT_BASIS_LOGGED = True
        logger.info(
            "companyfacts as-of basis: filed_date (explicit v2 reads require "
            "a dated filing; legacy VOE_PIT_FILED_ASOF reads retain the "
            "period_end + 90d fallback for undated rows)"
        )


ANNUAL_CACHED_FILING_FORM_TYPES = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")
QUARTERLY_CACHED_FILING_FORM_TYPES = ("10-Q", "10-Q/A")
MATERIAL_EVENT_CACHED_FILING_FORM_TYPES = ("8-K", "8-K/A")
FINANCIAL_CACHED_FILING_FORM_TYPES = (
    ANNUAL_CACHED_FILING_FORM_TYPES + QUARTERLY_CACHED_FILING_FORM_TYPES
)
ANNUAL_COMPANYFACTS_PERIOD_TYPES = ("FY",)


@dataclass(frozen=True)
class FilingIssuerScope:
    """Issuer identity used for cached filing recovery.

    ``filings`` is unique by issuer CIK and accession, but much of the older
    research path queried it by the requested security ticker alone.  Keeping
    the resolved CIK and aliases together lets an ADR, secondary class, or
    renamed security recover the issuer filing without rewriting historical
    filing rows.
    """

    requested_ticker: str
    issuer_cik: str | None
    aliases: tuple[str, ...]
    sources: tuple[str, ...]


def normalize_cik(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return None
    return digits.lstrip("0") or "0"


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    except sqlite3.DatabaseError:
        return set()


def _parse_ticker_aliases(payload: Any) -> list[str]:
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = [item.strip() for item in payload.split(",")]
    else:
        decoded = payload
    if not isinstance(decoded, (list, tuple, set)):
        return []
    return [str(item).strip().upper() for item in decoded if str(item).strip()]


def resolve_filing_issuer_scope(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
) -> FilingIssuerScope:
    """Resolve one security to an issuer CIK and its known ticker aliases.

    Resolution is local-only and intentionally additive.  It consults the
    caller-supplied identity first, then cached filing/company/universe rows,
    then the SEC registrant census.  No network lookup occurs here.
    """

    requested = str(ticker or "").strip().upper()
    known_aliases = {requested, *(str(item).strip().upper() for item in aliases)}
    known_aliases.discard("")
    resolved_cik = normalize_cik(issuer_cik)
    sources: list[str] = ["caller_cik"] if resolved_cik is not None else []

    lookup_specs = (
        ("filings", "ticker", "cik", "cached_filing"),
        ("companies", "ticker", "cik", "companies"),
        ("universe_members", "ticker", "cik", "universe_members"),
        ("sec_registrants", "primary_ticker", "cik", "sec_registrants_primary"),
    )
    if resolved_cik is None and requested:
        for table_name, ticker_column, cik_column, source in lookup_specs:
            columns = _table_columns(conn, table_name)
            if {ticker_column, cik_column} - columns:
                continue
            row = conn.execute(
                f"""
                SELECT {cik_column}
                FROM {table_name}
                WHERE UPPER({ticker_column}) = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (requested,),
            ).fetchone()
            candidate = normalize_cik(row[0]) if row else None
            if candidate is not None:
                resolved_cik = candidate
                sources.append(source)
                break

    registrant_columns = _table_columns(conn, "sec_registrants")
    if (
        resolved_cik is None
        and {
            "cik",
            "primary_ticker",
            "all_tickers",
        }
        <= registrant_columns
    ):
        for row in conn.execute("SELECT cik, primary_ticker, all_tickers FROM sec_registrants"):
            row_aliases = set(_parse_ticker_aliases(row["all_tickers"]))
            primary = str(row["primary_ticker"] or "").strip().upper()
            if primary:
                row_aliases.add(primary)
            if requested in row_aliases:
                resolved_cik = normalize_cik(row["cik"])
                known_aliases.update(row_aliases)
                sources.append("sec_registrants_alias")
                break

    if resolved_cik is not None:
        cik_int = int(resolved_cik)
        for table_name, ticker_column, cik_column, source in lookup_specs:
            columns = _table_columns(conn, table_name)
            if {ticker_column, cik_column} - columns:
                continue
            rows = conn.execute(
                f"SELECT {ticker_column} FROM {table_name} WHERE CAST({cik_column} AS INTEGER) = ?",
                (cik_int,),
            ).fetchall()
            before = len(known_aliases)
            known_aliases.update(
                str(row[0] or "").strip().upper() for row in rows if str(row[0] or "").strip()
            )
            if len(known_aliases) > before:
                sources.append(source)

        facts_columns = _table_columns(conn, "companyfacts_facts")
        if {"ticker", "source_url"} <= facts_columns:
            companyfacts_url = (
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{resolved_cik.zfill(10)}.json"
            )
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM companyfacts_facts "
                "WHERE source_url = ? AND ticker IS NOT NULL",
                (companyfacts_url,),
            ).fetchall()
            before = len(known_aliases)
            known_aliases.update(
                str(row[0] or "").strip().upper() for row in rows if str(row[0] or "").strip()
            )
            if len(known_aliases) > before:
                sources.append("companyfacts_source_identity")

        if {"cik", "primary_ticker", "all_tickers"} <= registrant_columns:
            row = conn.execute(
                """
                SELECT primary_ticker, all_tickers
                FROM sec_registrants
                WHERE CAST(cik AS INTEGER) = ?
                LIMIT 1
                """,
                (cik_int,),
            ).fetchone()
            if row:
                primary = str(row["primary_ticker"] or "").strip().upper()
                if primary:
                    known_aliases.add(primary)
                known_aliases.update(_parse_ticker_aliases(row["all_tickers"]))
                sources.append("sec_registrants_aliases")

    return FilingIssuerScope(
        requested_ticker=requested,
        issuer_cik=resolved_cik,
        aliases=tuple(sorted(known_aliases)),
        sources=tuple(dict.fromkeys(sources)),
    )


def _columns_sql(columns: Sequence[str]) -> str:
    return ", ".join(columns)


def _placeholders(values: Sequence[str]) -> str:
    return ",".join("?" for _ in values)


def _append_companyfacts_asof_filter(
    clauses: list[str],
    params: list[Any],
    *,
    as_of_date: str | None,
    require_filed_asof: bool,
) -> None:
    if not as_of_date:
        return
    if require_filed_asof:
        # Decision-bearing fixed-as-of evidence must have an actual filing
        # date. A generic
        # lag assumption cannot prove when a late or foreign filing became
        # public and would leak hindsight into historical scans.
        _log_pit_basis_once()
        clauses.append(
            "date(period_end) IS NOT NULL "
            "AND date(filed_date) IS NOT NULL "
            "AND date(period_end) <= date(filed_date) "
            "AND date(period_end) <= date(?) "
            "AND date(filed_date) <= date(?)"
        )
        params.extend([as_of_date, as_of_date])
    elif pit_filed_asof_enabled():
        # Legacy opt-in PIT mode retains its historical conservative fallback
        # for rows ingested before filed_date was captured.
        _log_pit_basis_once()
        clauses.append(
            "period_end <= ? AND ("
            "  (filed_date IS NOT NULL AND filed_date != '' AND filed_date <= ?)"
            "  OR ((filed_date IS NULL OR filed_date = '') AND period_end <= date(?, '-90 days'))"
            ")"
        )
        params.extend([as_of_date, as_of_date, as_of_date])
    else:
        clauses.append("period_end <= ?")
        params.append(as_of_date)


def filing_rows(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    columns: Sequence[str],
    form_types: Sequence[str] | None = None,
    statuses: Sequence[str] | None = VALID_CACHED_FILING_STATUSES,
    as_of_date: str | None = None,
    require_local_path: bool = False,
    limit: int | None = None,
    order_by: str = "COALESCE(filing_date, '1900-01-01') DESC, id DESC",
) -> list[sqlite3.Row]:
    params: list[str] = [ticker.upper()]
    clauses = ["ticker = ?"]
    if form_types:
        clauses.append(f"form_type IN ({_placeholders(form_types)})")
        params.extend(form_types)
    if statuses:
        clauses.append(f"status IN ({_placeholders(statuses)})")
        params.extend(statuses)
    if require_local_path:
        clauses.append("local_path IS NOT NULL")
        clauses.append("local_path != ''")
    if as_of_date:
        clauses.append("filing_date <= ?")
        params.append(as_of_date)
    sql = f"SELECT {_columns_sql(columns)} FROM filings WHERE {' AND '.join(clauses)} ORDER BY {order_by}"
    if limit is not None:
        sql = f"{sql} LIMIT {max(1, int(limit))}"
    return conn.execute(sql, params).fetchall()


def issuer_filing_rows(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    columns: Sequence[str],
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    form_types: Sequence[str] | None = None,
    statuses: Sequence[str] | None = VALID_CACHED_FILING_STATUSES,
    as_of_date: str | None = None,
    require_local_path: bool = False,
    limit: int | None = None,
    order_by: str = "COALESCE(filing_date, '1900-01-01') DESC, id DESC",
) -> tuple[FilingIssuerScope, list[sqlite3.Row]]:
    """Return filing rows bound to one issuer.

    The legacy :func:`filing_rows` helper remains ticker-exact.  This recovery
    helper is used by the v2 data plane and research filing materializer so a
    filing already cached under another class or ADR ticker is still visible.
    Once an issuer CIK is known it is authoritative and exclusive: aliases are
    discovery hints, not an alternate identity predicate.  This fails closed
    when a ticker has been reused or incorrectly associated with another CIK.
    """

    scope = resolve_filing_issuer_scope(
        conn,
        ticker,
        issuer_cik=issuer_cik,
        aliases=aliases,
    )
    params: list[Any] = []
    if scope.issuer_cik is not None:
        if "cik" not in _table_columns(conn, "filings"):
            return scope, []
        identity_clause = "CAST(cik AS INTEGER) = ?"
        params.append(int(scope.issuer_cik))
    elif scope.aliases:
        identity_clause = f"UPPER(ticker) IN ({_placeholders(scope.aliases)})"
        params.extend(scope.aliases)
    else:
        return scope, []

    clauses = [identity_clause]
    if form_types:
        normalized_forms = tuple(str(value).upper() for value in form_types)
        clauses.append(f"UPPER(form_type) IN ({_placeholders(normalized_forms)})")
        params.extend(normalized_forms)
    if statuses:
        clauses.append(f"status IN ({_placeholders(statuses)})")
        params.extend(statuses)
    if require_local_path:
        clauses.extend(("local_path IS NOT NULL", "local_path != ''"))
    if as_of_date:
        clauses.append("filing_date <= ?")
        params.append(as_of_date)
    sql = (
        f"SELECT {_columns_sql(columns)} FROM filings "
        f"WHERE {' AND '.join(clauses)} ORDER BY {order_by}"
    )
    if limit is not None:
        sql = f"{sql} LIMIT {max(1, int(limit))}"
    return scope, conn.execute(sql, params).fetchall()


def _raw_companyfacts_namespaces(payload: Any) -> dict[str, Any]:
    current = payload
    if isinstance(current, dict) and isinstance(current.get("companyfacts"), dict):
        current = current["companyfacts"]
    if isinstance(current, dict) and isinstance(current.get("facts"), dict):
        return dict(current["facts"])
    return {}


def _companyfacts_currency_units(namespaces: dict[str, Any]) -> set[str]:
    units: set[str] = set()
    for concepts in namespaces.values():
        if not isinstance(concepts, dict):
            continue
        for concept in concepts.values():
            concept_units = concept.get("units") if isinstance(concept, dict) else None
            if not isinstance(concept_units, dict):
                continue
            for unit_name in concept_units:
                normalized = str(unit_name or "").strip().upper()
                if len(normalized) == 3 and normalized.isalpha():
                    units.add(normalized)
    return units


def foreign_normalized_facts_gap_reason(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    issuer_cik: str | None,
    form_type: str | None,
    as_of_date: str | None = None,
    aliases: Sequence[str] = (),
    raw_companyfacts_path: str | Path | None = None,
    required_line_items: Sequence[str] = (),
    minimum_years: int = 1,
    require_filed_asof: bool = False,
) -> str | None:
    """Classify an unsupported foreign-filer normalized-facts gap.

    This is detection only: it does not normalize ``ifrs-full`` facts and it
    does not convert currencies.  A foreign annual filing with already
    normalized FY rows remains usable; otherwise the v2 pipeline can surface
    the precise reason as ``NEEDS_DATA``.
    """

    normalized_form = str(form_type or "").strip().upper()
    if normalized_form not in {"20-F", "20-F/A", "40-F", "40-F/A"}:
        return None

    scope = resolve_filing_issuer_scope(
        conn,
        ticker,
        issuer_cik=issuer_cik,
        aliases=aliases,
    )
    facts_columns = _table_columns(conn, "companyfacts_facts")
    required = {"ticker", "period_type", "value"}
    if required <= facts_columns:
        params: list[Any] = []
        identity_clause: str | None = None
        if scope.issuer_cik is not None and "source_url" in facts_columns:
            padded_cik = scope.issuer_cik.zfill(10)
            identity_clause = "source_url LIKE ?"
            params.append(f"%CIK{padded_cik}.json%")
        elif scope.issuer_cik is None and scope.aliases:
            identity_clause = f"UPPER(ticker) IN ({_placeholders(scope.aliases)})"
            params.extend(scope.aliases)
        if identity_clause:
            clauses = [
                identity_clause,
                "period_type = 'FY'",
                "value IS NOT NULL",
            ]
            if as_of_date and "period_end" in facts_columns:
                _append_companyfacts_asof_filter(
                    clauses,
                    params,
                    as_of_date=as_of_date,
                    require_filed_asof=require_filed_asof,
                )
            requested = tuple(
                dict.fromkeys(
                    str(item).strip() for item in required_line_items if str(item).strip()
                )
            )
            if requested and {"line_item", "fiscal_year"} <= facts_columns:
                clauses.append(f"line_item IN ({_placeholders(requested)})")
                params.extend(requested)
                normalized_rows = conn.execute(
                    f"SELECT line_item, COUNT(DISTINCT fiscal_year) AS years "
                    f"FROM companyfacts_facts WHERE {' AND '.join(clauses)} "
                    "GROUP BY line_item",
                    params,
                ).fetchall()
                coverage = {
                    str(row["line_item"]): int(row["years"] or 0) for row in normalized_rows
                }
                if all(
                    coverage.get(line_item, 0) >= max(1, int(minimum_years))
                    for line_item in requested
                ):
                    return None
            else:
                normalized = conn.execute(
                    f"SELECT 1 FROM companyfacts_facts WHERE {' AND '.join(clauses)} LIMIT 1",
                    params,
                ).fetchone()
                if normalized is not None:
                    return None

    path: Path | None
    if raw_companyfacts_path is not None:
        path = Path(raw_companyfacts_path)
    elif scope.issuer_cik is not None:
        try:
            from app.config import get_config

            path = get_config().cache_dir / "companyfacts" / f"{scope.issuer_cik.zfill(10)}.json"
        except Exception:
            path = None
    else:
        path = None

    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else None
    except (OSError, ValueError, json.JSONDecodeError):
        payload = None
    namespaces = _raw_companyfacts_namespaces(payload)
    currency_units = _companyfacts_currency_units(namespaces)
    non_usd_units = currency_units - {"USD"}
    if non_usd_units and "USD" not in currency_units:
        return "NON_USD_FACTS_UNNORMALIZED"
    if (
        isinstance(namespaces.get("ifrs-full"), dict)
        and namespaces["ifrs-full"]
        and not (isinstance(namespaces.get("us-gaap"), dict) and namespaces["us-gaap"])
    ):
        return "IFRS_FACTS_UNSUPPORTED"
    # Mixed-currency disclosures are common even in otherwise usable USD
    # CompanyFacts payloads. Without mapping each missing normalized line item
    # back to its raw concept, do not misattribute a generic coverage gap to
    # an incidental EUR/CAD disclosure.
    return "FOREIGN_NORMALIZED_FACTS_UNAVAILABLE"


def latest_filing_row(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    columns: Sequence[str],
    form_types: Sequence[str] | None = None,
    statuses: Sequence[str] | None = VALID_CACHED_FILING_STATUSES,
    as_of_date: str | None = None,
    require_local_path: bool = False,
    order_by: str = "COALESCE(filing_date, '1900-01-01') DESC, id DESC",
) -> sqlite3.Row | None:
    rows = filing_rows(
        conn,
        ticker,
        columns=columns,
        form_types=form_types,
        statuses=statuses,
        as_of_date=as_of_date,
        require_local_path=require_local_path,
        limit=1,
        order_by=order_by,
    )
    return rows[0] if rows else None


def latest_filing_ids_by_ticker(
    conn: sqlite3.Connection,
    *,
    form_types: Sequence[str] | None = FINANCIAL_CACHED_FILING_FORM_TYPES,
    statuses: Sequence[str] | None = VALID_CACHED_FILING_STATUSES,
    as_of_date: str | None = None,
) -> dict[str, int]:
    params: list[Any] = []
    clauses = ["ticker IS NOT NULL"]
    if form_types:
        clauses.append(f"form_type IN ({_placeholders(form_types)})")
        params.extend(form_types)
    if statuses:
        clauses.append(f"status IN ({_placeholders(statuses)})")
        params.extend(statuses)
    if as_of_date:
        clauses.append("filing_date IS NOT NULL AND filing_date <= ?")
        params.append(as_of_date)
    rows = conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                id,
                ticker,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker
                    ORDER BY COALESCE(filing_date, '1900-01-01') DESC, id DESC
                ) AS row_num
            FROM filings
            WHERE {" AND ".join(clauses)}
        )
        SELECT id, ticker
        FROM ranked
        WHERE row_num = 1
        """,
        params,
    ).fetchall()
    return {str(row["ticker"]): int(row["id"]) for row in rows if row["ticker"] is not None}


def companyfacts_rows(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    columns: Sequence[str],
    period_types: Sequence[str] | None = None,
    exclude_period_types: Sequence[str] | None = None,
    line_items: Sequence[str] | None = None,
    as_of_date: str | None = None,
    period_end: str | None = None,
    value_not_null: bool = False,
    require_filed_asof: bool = False,
    limit: int | None = None,
    order_by: str = "fiscal_year ASC, line_item ASC",
) -> list[sqlite3.Row]:
    if require_filed_asof:
        required_columns = {
            "period_end",
            "filed_date",
            "accession",
            "source_url",
        }
        if required_columns - _table_columns(conn, "companyfacts_facts"):
            return []
    params: list[str] = [ticker.upper()]
    clauses = ["ticker = ?"]
    if period_types:
        clauses.append(f"period_type IN ({_placeholders(period_types)})")
        params.extend(period_types)
    if exclude_period_types:
        clauses.append(f"period_type NOT IN ({_placeholders(exclude_period_types)})")
        params.extend(exclude_period_types)
    if line_items:
        clauses.append(f"line_item IN ({_placeholders(line_items)})")
        params.extend(line_items)
    _append_companyfacts_asof_filter(
        clauses,
        params,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
    )
    if require_filed_asof:
        clauses.extend(
            [
                "NULLIF(TRIM(accession), '') IS NOT NULL",
                "NULLIF(TRIM(source_url), '') IS NOT NULL",
            ]
        )
    if period_end:
        clauses.append("period_end = ?")
        params.append(period_end)
    if value_not_null:
        clauses.append("value IS NOT NULL")
    sql = f"SELECT {_columns_sql(columns)} FROM companyfacts_facts WHERE {' AND '.join(clauses)} ORDER BY {order_by}"
    if limit is not None:
        sql = f"{sql} LIMIT {max(1, int(limit))}"
    return conn.execute(sql, params).fetchall()


def _issuer_companyfacts_pit_rows_with_vintages(
    conn: sqlite3.Connection,
    *,
    scope: FilingIssuerScope,
    columns: Sequence[str],
    period_types: Sequence[str] | None,
    exclude_period_types: Sequence[str] | None,
    line_items: Sequence[str] | None,
    as_of_date: str,
    period_end: str | None,
    value_not_null: bool,
    limit: int | None,
    order_by: str,
) -> list[sqlite3.Row] | None:
    """Return the latest filing-visible fact version from live plus vintages.

    ``companyfacts_facts`` deliberately stores only one current row per
    fiscal-year/period/line-item. A later restatement can therefore replace the
    row that was visible at an older replay date. The append-only vintage table
    is the point-in-time source of truth for that case. Known-CIK reads remain
    issuer-exclusive: vintage aliases are accepted only when a CIK is absent.
    """

    live_columns = _table_columns(conn, "companyfacts_facts")
    vintage_columns = _table_columns(conn, "companyfacts_vintages")
    required = {
        "id",
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
    }
    if required - live_columns or required - vintage_columns:
        return None

    canonical_columns = {
        "id",
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
    if set(columns) - canonical_columns:
        return None

    identity_cte_sql = ""
    identity_cte_params: list[Any] = []
    live_params: list[Any] = []
    vintage_params: list[Any] = []
    if scope.issuer_cik is not None:
        cik10 = scope.issuer_cik.zfill(10)
        cik_int = int(scope.issuer_cik)
        issuer_aliases = scope.aliases or (scope.requested_ticker,)
        live_identity = f"f.ticker IN ({_placeholders(issuer_aliases)}) AND f.source_url LIKE ?"
        live_params.extend(issuer_aliases)
        live_params.append(f"%CIK{cik10}.json%")
        explicit_match_parts: list[str] = []
        explicit_match_params: list[Any] = []
        explicit_compatible_parts: list[str] = []
        explicit_compatible_params: list[Any] = []
        identity_absent_parts: list[str] = []
        if "issuer_cik" in vintage_columns:
            explicit_match_parts.append("CAST(v.issuer_cik AS INTEGER) = ?")
            explicit_match_params.append(cik_int)
            explicit_compatible_parts.append(
                "(NULLIF(TRIM(v.issuer_cik), '') IS NULL OR CAST(v.issuer_cik AS INTEGER) = ?)"
            )
            explicit_compatible_params.append(cik_int)
            identity_absent_parts.append("NULLIF(TRIM(v.issuer_cik), '') IS NULL")
        if "source_url" in vintage_columns:
            explicit_match_parts.append("v.source_url LIKE ?")
            explicit_match_params.append(f"%CIK{cik10}.json%")
            explicit_compatible_parts.append(
                "(NULLIF(TRIM(v.source_url), '') IS NULL "
                "OR UPPER(v.source_url) NOT LIKE '%CIK%.JSON%' "
                "OR v.source_url LIKE ?)"
            )
            explicit_compatible_params.append(f"%CIK{cik10}.json%")
            identity_absent_parts.append("NULLIF(TRIM(v.source_url), '') IS NULL")

        # Older vintage rows predate issuer_cik/source_url columns. Materialize
        # their authoritative aliases once inside the same SQLite statement /
        # read snapshot as the vintage scan. The previous correlated EXISTS
        # predicates rescanned the multi-million-row live facts index for every
        # vintage row (roughly one minute per issuer in the fixed replay).
        verified_alias_selects: list[str] = []
        if {"ticker", "cik"} <= _table_columns(conn, "companies"):
            verified_alias_selects.append(
                "SELECT UPPER(c.ticker) AS ticker FROM companies c "
                f"WHERE UPPER(c.ticker) IN ({_placeholders(issuer_aliases)}) "
                "AND CAST(c.cik AS INTEGER) = ?"
            )
            identity_cte_params.extend([*issuer_aliases, cik_int])
        verified_alias_selects.append(
            "SELECT DISTINCT UPPER(current_fact.ticker) AS ticker "
            "FROM companyfacts_facts current_fact "
            f"WHERE UPPER(current_fact.ticker) IN ({_placeholders(issuer_aliases)}) "
            "AND current_fact.source_url LIKE ?"
        )
        identity_cte_params.extend([*issuer_aliases, f"%CIK{cik10}.json%"])
        identity_cte_sql = (
            "verified_fallback_aliases(ticker) AS MATERIALIZED ("
            + " UNION ".join(verified_alias_selects)
            + "),"
        )

        explicit_identity = "0"
        if explicit_match_parts:
            explicit_identity = (
                f"(({' OR '.join(explicit_match_parts)}) "
                f"AND {' AND '.join(explicit_compatible_parts)})"
            )
        identity_absent = " AND ".join(identity_absent_parts) if identity_absent_parts else "1"
        fallback_identity = (
            f"(({identity_absent}) AND UPPER(v.ticker) IN "
            "(SELECT ticker FROM verified_fallback_aliases))"
        )
        vintage_identity = (
            f"(v.ticker IN ({_placeholders(issuer_aliases)}) "
            f"AND ({explicit_identity} OR {fallback_identity}))"
        )
        vintage_params = [
            *issuer_aliases,
            *explicit_match_params,
            *explicit_compatible_params,
        ]
    elif scope.aliases:
        live_identity = f"UPPER(f.ticker) IN ({_placeholders(scope.aliases)})"
        live_params.extend(scope.aliases)
        vintage_identity = f"UPPER(v.ticker) IN ({_placeholders(scope.aliases)})"
        vintage_params.extend(scope.aliases)
    else:
        return []

    vintage_source_url = "v.source_url" if "source_url" in vintage_columns else "NULL"
    vintage_fetched_at = "v.recorded_at" if "recorded_at" in vintage_columns else "v.filed_date"
    clauses: list[str] = []
    params: list[Any] = [*identity_cte_params, *live_params, *vintage_params]
    if period_types:
        clauses.append(f"period_type IN ({_placeholders(period_types)})")
        params.extend(period_types)
    if exclude_period_types:
        clauses.append(f"period_type NOT IN ({_placeholders(exclude_period_types)})")
        params.extend(exclude_period_types)
    if line_items:
        clauses.append(f"line_item IN ({_placeholders(line_items)})")
        params.extend(line_items)
    _append_companyfacts_asof_filter(
        clauses,
        params,
        as_of_date=as_of_date,
        require_filed_asof=True,
    )
    clauses.extend(
        [
            "NULLIF(TRIM(accession), '') IS NOT NULL",
            "(source_rank = 1 OR NULLIF(TRIM(source_url), '') IS NOT NULL)",
        ]
    )
    if period_end:
        clauses.append("period_end = ?")
        params.append(period_end)
    if value_not_null:
        clauses.append("value IS NOT NULL")

    sql = f"""
        WITH {identity_cte_sql} candidates AS (
            SELECT
                f.id, f.ticker, f.fiscal_year, f.period_type, f.period_end,
                f.line_item, f.value, f.units, f.source_url, f.fetched_at,
                f.filed_date, f.form, f.accession, 0 AS source_rank
            FROM companyfacts_facts f
            WHERE {live_identity}
            UNION ALL
            SELECT
                v.id, v.ticker, v.fiscal_year, v.period_type, v.period_end,
                v.line_item, v.value, v.units, {vintage_source_url} AS source_url,
                {vintage_fetched_at} AS fetched_at, v.filed_date, v.form,
                v.accession, 1 AS source_rank
            FROM companyfacts_vintages v
            WHERE {vintage_identity}
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY fiscal_year, period_type, line_item
                ORDER BY filed_date DESC, source_rank ASC, id DESC
            ) AS version_rank
            FROM candidates
            WHERE {" AND ".join(clauses)}
        )
        SELECT {_columns_sql(columns)}
        FROM ranked
        WHERE version_rank = 1
        ORDER BY {order_by}
    """
    if limit is not None:
        sql = f"{sql} LIMIT {max(1, int(limit))}"
    return conn.execute(sql, params).fetchall()


def issuer_companyfacts_rows(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    columns: Sequence[str],
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    period_types: Sequence[str] | None = None,
    exclude_period_types: Sequence[str] | None = None,
    line_items: Sequence[str] | None = None,
    as_of_date: str | None = None,
    period_end: str | None = None,
    value_not_null: bool = False,
    require_filed_asof: bool = False,
    limit: int | None = None,
    order_by: str = "fiscal_year ASC, line_item ASC",
) -> tuple[FilingIssuerScope, list[sqlite3.Row]]:
    """Read normalized facts bound to one issuer.

    V1 callers keep using ticker-exact :func:`companyfacts_rows`. V2 packet
    and checkpoint paths use this helper with ``require_filed_asof=True`` so
    the producer and its readiness check share identity and PIT semantics.
    A known CIK is matched through the SEC CompanyFacts source URL
    exclusively; ticker aliases are used only when no CIK can be resolved.
    """

    scope = resolve_filing_issuer_scope(
        conn,
        ticker,
        issuer_cik=issuer_cik,
        aliases=aliases,
    )
    facts_columns = _table_columns(conn, "companyfacts_facts")
    if require_filed_asof:
        required_columns = {
            "period_end",
            "filed_date",
            "accession",
            "source_url",
        }
        if required_columns - facts_columns:
            return scope, []
    if require_filed_asof and as_of_date:
        vintage_rows = _issuer_companyfacts_pit_rows_with_vintages(
            conn,
            scope=scope,
            columns=columns,
            period_types=period_types,
            exclude_period_types=exclude_period_types,
            line_items=line_items,
            as_of_date=as_of_date,
            period_end=period_end,
            value_not_null=value_not_null,
            limit=limit,
            order_by=order_by,
        )
        if vintage_rows is not None:
            return scope, vintage_rows
    params: list[Any] = []
    if scope.issuer_cik is not None and "source_url" in facts_columns:
        identity_clause = "source_url LIKE ?"
        params.append(f"%CIK{scope.issuer_cik.zfill(10)}.json%")
    elif scope.issuer_cik is None and scope.aliases:
        identity_clause = f"UPPER(ticker) IN ({_placeholders(scope.aliases)})"
        params.extend(scope.aliases)
    else:
        return scope, []

    clauses = [identity_clause]
    if period_types:
        clauses.append(f"period_type IN ({_placeholders(period_types)})")
        params.extend(period_types)
    if exclude_period_types:
        clauses.append(f"period_type NOT IN ({_placeholders(exclude_period_types)})")
        params.extend(exclude_period_types)
    if line_items:
        clauses.append(f"line_item IN ({_placeholders(line_items)})")
        params.extend(line_items)
    _append_companyfacts_asof_filter(
        clauses,
        params,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
    )
    if require_filed_asof:
        clauses.extend(
            [
                "NULLIF(TRIM(accession), '') IS NOT NULL",
                "NULLIF(TRIM(source_url), '') IS NOT NULL",
            ]
        )
    if period_end:
        clauses.append("period_end = ?")
        params.append(period_end)
    if value_not_null:
        clauses.append("value IS NOT NULL")
    sql = (
        f"SELECT {_columns_sql(columns)} FROM companyfacts_facts "
        f"WHERE {' AND '.join(clauses)} ORDER BY {order_by}"
    )
    if limit is not None:
        sql = f"{sql} LIMIT {max(1, int(limit))}"
    return scope, conn.execute(sql, params).fetchall()


def latest_companyfacts_period_end(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str | None = None,
    period_types: Sequence[str] | None = None,
    exclude_period_types: Sequence[str] | None = None,
    require_filed_asof: bool = False,
) -> str | None:
    rows = companyfacts_rows(
        conn,
        ticker,
        columns=("period_end",),
        period_types=period_types,
        exclude_period_types=exclude_period_types,
        as_of_date=as_of_date,
        require_filed_asof=require_filed_asof,
        limit=1,
        order_by="period_end DESC",
    )
    if not rows or not rows[0]["period_end"]:
        return None
    return str(rows[0]["period_end"])


def companyfacts_map_for_latest_period(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str | None = None,
    period_types: Sequence[str] | None = ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    require_filed_asof: bool = True,
) -> tuple[dict[str, float | None], str | None]:
    latest_period_end = latest_companyfacts_period_end(
        conn,
        ticker,
        as_of_date=as_of_date,
        period_types=period_types,
        require_filed_asof=require_filed_asof,
    )
    if latest_period_end is None:
        return {}, None
    rows = companyfacts_rows(
        conn,
        ticker,
        columns=("line_item", "value"),
        period_types=period_types,
        as_of_date=as_of_date,
        period_end=latest_period_end,
        require_filed_asof=require_filed_asof,
        order_by="line_item ASC",
    )
    return {
        str(row["line_item"]): row["value"] for row in rows if row["line_item"]
    }, latest_period_end


def latest_companyfacts_value(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    line_item: str,
    period_types: Sequence[str] | None = ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    exclude_period_types: Sequence[str] | None = None,
    as_of_date: str | None = None,
    value_not_null: bool = False,
    require_filed_asof: bool = False,
    order_by: str = "fiscal_year DESC, period_end DESC",
) -> float | None:
    rows = companyfacts_rows(
        conn,
        ticker,
        columns=("value",),
        period_types=period_types,
        exclude_period_types=exclude_period_types,
        line_items=(line_item,),
        as_of_date=as_of_date,
        value_not_null=value_not_null,
        require_filed_asof=require_filed_asof,
        order_by=order_by,
        limit=1,
    )
    if not rows or rows[0]["value"] is None:
        return None
    return float(rows[0]["value"])


def companyfacts_is_fresh(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    ttl_seconds: int,
    period_types: Sequence[str] | None = None,
    exclude_period_types: Sequence[str] | None = None,
) -> bool:
    rows = companyfacts_rows(
        conn,
        ticker,
        columns=("fetched_at",),
        period_types=period_types,
        exclude_period_types=exclude_period_types,
        order_by="fetched_at DESC",
        limit=1,
    )
    if not rows or not rows[0]["fetched_at"]:
        return False
    try:
        fetched = datetime.fromisoformat(str(rows[0]["fetched_at"]).replace("Z", "+00:00"))
        age_seconds = (datetime.now(timezone.utc) - fetched).total_seconds()
    except Exception:
        return False
    return age_seconds < max(1, int(ttl_seconds))
