"""Accepted-census-bound, provider-free v2 readiness preflight.

This module is deliberately a read-side probe, not a cache refresh and not a
sector run. It consumes the exact frozen execution ledgers, validates them
against the accepted census, and inspects only evidence already present at the
requested as-of date. Every execution name remains in the artifact as READY,
NEEDS_DATA, or INCOMPLETE; readiness never shrinks the frozen candidate set.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping

from app.autonomous.evidence_resolution import (
    PACKET_REQUIRED_FINANCIAL_FACT_HISTORY,
    PACKET_REQUIRED_INSURANCE_FACT_HISTORY,
    PACKET_REQUIRED_NORMALIZED_FACT_HISTORY,
)
from app.autonomous.sector_candidates import AcceptedCensusRunAuthority
from app.config import AppConfig, canonical_market_cap_focus, get_config
from app.util.credential_hygiene import sanitize_url_credentials
from app.util.financial_data_access import (
    ANNUAL_CACHED_FILING_FORM_TYPES,
    issuer_companyfacts_rows,
    issuer_filing_rows,
    normalize_cik,
)
from app.valuation.provenance import validate_v2_scorecard_provenance


READINESS_PREFLIGHT_SCHEMA_VERSION = "AUTONOMOUS_V2_READINESS_PREFLIGHT_V1"
READINESS_MODE = "ACCEPTED_CENSUS_QUERY_ONLY"
READINESS_READY = "READY"
READINESS_NEEDS_DATA = "NEEDS_DATA"
READINESS_INCOMPLETE = "INCOMPLETE"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _ticker_ledger_fingerprint(tickers: list[str]) -> str:
    return sha256(
        json.dumps(tickers, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def v2_execution_set_fingerprint(
    *,
    sectors: list[str],
    candidate_payloads: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return the canonical whole-run fingerprint for exact execution lists."""

    payload = [
        {
            "sector": sector,
            "execution_tickers": list(
                candidate_payloads.get(sector, {}).get("execution_tickers") or []
            ),
        }
        for sector in sectors
    ]
    return _sha256_json(payload)


@contextmanager
def _read_only_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"readiness database does not exist: {path}")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA temp_store = MEMORY")
        query_only = conn.execute("PRAGMA query_only").fetchone()
        if query_only is None or int(query_only[0]) != 1:
            raise RuntimeError("readiness connection is not query-only")
        yield conn
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        return {
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
    except sqlite3.DatabaseError:
        return set()


def _normalize_exact_tickers(values: Any, *, field: str, sector: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{sector}:{field} must be a list")
    normalized = [str(value or "").strip().upper() for value in values]
    if any(not value for value in normalized):
        raise ValueError(f"{sector}:{field} contains an empty ticker")
    if normalized != values:
        raise ValueError(f"{sector}:{field} is not canonically normalized")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{sector}:{field} contains duplicate tickers")
    return normalized


def _validate_frozen_authority(
    *,
    sectors: list[str],
    candidate_payloads: Mapping[str, Mapping[str, Any]],
    as_of_date: str,
    market_cap_focus: str,
    execution_set_fingerprint: str,
    request_fingerprint: str,
    cohort: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if canonical_market_cap_focus(market_cap_focus) != "large_and_mega":
        raise ValueError("readiness preflight requires large_and_mega")
    if not request_fingerprint:
        raise ValueError("readiness preflight requires a frozen request fingerprint")
    if len(set(sectors)) != len(sectors) or any(not str(sector).strip() for sector in sectors):
        raise ValueError("readiness sectors must be unique non-empty labels")
    if set(candidate_payloads) != set(sectors):
        raise ValueError("readiness candidate payload sectors do not match the request")

    lineage = cohort.lineage.to_dict()
    if lineage.get("as_of_date") != as_of_date:
        raise ValueError("accepted census as-of date does not match readiness request")
    if lineage.get("target_band") != "large_and_mega":
        raise ValueError("accepted census target band does not match readiness request")

    sector_bindings: dict[str, dict[str, Any]] = {}
    execution_identity: dict[str, dict[str, Any]] = {}
    all_membership: list[str] = []
    for sector in sectors:
        payload = candidate_payloads.get(sector)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{sector}: frozen candidate payload is missing")
        if str(payload.get("sector") or "") != sector:
            raise ValueError(f"{sector}: candidate sector identity drifted")
        if str(payload.get("source") or "") != "accepted_census":
            raise ValueError(f"{sector}: readiness source is not accepted_census")
        if canonical_market_cap_focus(str(payload.get("market_cap_focus") or "")) != (
            "large_and_mega"
        ):
            raise ValueError(f"{sector}: candidate market-cap focus drifted")
        if payload.get("execution_bound_frozen") is not True:
            raise ValueError(f"{sector}: execution bound is not frozen")
        if dict(payload.get("census_lineage") or {}) != lineage:
            raise ValueError(f"{sector}: accepted census lineage drifted")
        if str(payload.get("request_fingerprint") or "") != request_fingerprint:
            raise ValueError(f"{sector}: request fingerprint drifted")
        if str(payload.get("execution_set_fingerprint") or "") != (
            execution_set_fingerprint
        ):
            raise ValueError(f"{sector}: whole-run execution fingerprint drifted")
        execution_as_of = str(payload.get("execution_as_of_date") or "")[:10]
        if execution_as_of != as_of_date:
            raise ValueError(f"{sector}: execution as-of date drifted")

        membership = _normalize_exact_tickers(
            payload.get("membership_tickers"),
            field="membership_tickers",
            sector=sector,
        )
        selected = _normalize_exact_tickers(
            payload.get("selected_tickers"),
            field="selected_tickers",
            sector=sector,
        )
        execution = _normalize_exact_tickers(
            payload.get("execution_tickers"),
            field="execution_tickers",
            sector=sector,
        )
        deferred = _normalize_exact_tickers(
            payload.get("deferred_by_bound_tickers"),
            field="deferred_by_bound_tickers",
            sector=sector,
        )
        excluded = _normalize_exact_tickers(
            payload.get("excluded_tickers"),
            field="excluded_tickers",
            sector=sector,
        )
        accepted_members = list(cohort.members_for_sector(sector))
        accepted_membership = [member.ticker for member in accepted_members]
        if membership != accepted_membership or selected != accepted_membership:
            raise ValueError(f"{sector}: membership does not equal accepted census order")
        if [*execution, *deferred] != membership:
            raise ValueError(f"{sector}: execution/deferred ledgers do not partition membership")
        if excluded:
            raise ValueError(f"{sector}: accepted census readiness cannot exclude members")
        bound = payload.get("execution_bound")
        if bound is not None and (
            isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0
        ):
            raise ValueError(f"{sector}: execution bound is invalid")
        expected_execution = membership if bound is None else membership[:bound]
        if execution != expected_execution:
            raise ValueError(f"{sector}: execution ledger widened or reordered")
        if str(payload.get("membership_fingerprint") or "") != (
            _ticker_ledger_fingerprint(membership)
        ):
            raise ValueError(f"{sector}: membership fingerprint drifted")
        if str(payload.get("execution_fingerprint") or "") != (
            _ticker_ledger_fingerprint(execution)
        ):
            raise ValueError(f"{sector}: execution fingerprint drifted")

        cap_classifications = payload.get("cap_classifications")
        if not isinstance(cap_classifications, Mapping) or set(cap_classifications) != set(
            membership
        ):
            raise ValueError(f"{sector}: cap ledger does not equal accepted membership")
        members_by_ticker = {member.ticker: member for member in accepted_members}
        for ticker in membership:
            expected_cap = members_by_ticker[ticker].to_cap_classification(
                lineage=cohort.lineage
            ).to_dict()
            if dict(cap_classifications.get(ticker) or {}) != expected_cap:
                raise ValueError(f"{sector}:{ticker}: accepted cap or identity drifted")
        for ticker in execution:
            member = members_by_ticker[ticker]
            execution_identity[ticker] = {
                "ticker": ticker,
                "issuer_cik": member.cik,
                "issuer_key": member.issuer_key,
                "security_key": member.security_key,
                "source_sector_label": member.source_sector_label,
                "canonical_sector": member.canonical_sector,
            }
        sector_bindings[sector] = {
            "membership_tickers": membership,
            "execution_tickers": execution,
            "deferred_by_bound_tickers": deferred,
            "excluded_tickers": excluded,
            "membership_fingerprint": payload.get("membership_fingerprint"),
            "execution_fingerprint": payload.get("execution_fingerprint"),
            "execution_bound": bound,
        }
        all_membership.extend(membership)

    duplicates = sorted(
        ticker for ticker, count in Counter(all_membership).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "accepted census tickers appeared in multiple requested sectors: "
            + ",".join(duplicates)
        )
    recomputed_execution_set = v2_execution_set_fingerprint(
        sectors=sectors,
        candidate_payloads=candidate_payloads,
    )
    if recomputed_execution_set != execution_set_fingerprint:
        raise ValueError("whole-run execution-set fingerprint drifted")
    return sector_bindings, execution_identity


def _facts_requirement(sector: str) -> tuple[str, dict[str, int]]:
    if sector == "large_cap_financials":
        return "FINANCIAL", dict(PACKET_REQUIRED_FINANCIAL_FACT_HISTORY)
    if sector == "insurance":
        return "INSURANCE_COMMON", dict(PACKET_REQUIRED_INSURANCE_FACT_HISTORY)
    return "OPERATING", dict(PACKET_REQUIRED_NORMALIZED_FACT_HISTORY)


def _facts_readiness(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    issuer_cik: str,
    sector: str,
    as_of_date: str,
) -> dict[str, Any]:
    required_columns = {
        "ticker",
        "fiscal_year",
        "period_type",
        "period_end",
        "line_item",
        "value",
        "filed_date",
        "source_url",
    }
    if required_columns - _table_columns(conn, "companyfacts_facts"):
        _, required_history = _facts_requirement(sector)
        return {
            "status": READINESS_NEEDS_DATA,
            "reason_code": "NORMALIZED_FACTS_TABLE_UNAVAILABLE",
            "required_history_years_by_line_item": required_history,
            "missing_line_items": list(required_history),
            "insufficient_history_line_items": [],
            "annual_rows": 0,
            "annual_years": 0,
            "latest_period_end": None,
        }
    scope, rows = issuer_companyfacts_rows(
        conn,
        ticker,
        columns=("ticker", "fiscal_year", "period_end", "filed_date", "line_item"),
        issuer_cik=issuer_cik,
        aliases=(ticker,),
        period_types=("FY",),
        as_of_date=as_of_date,
        value_not_null=True,
        require_filed_asof=True,
        order_by="fiscal_year ASC, line_item ASC",
    )
    requirement_kind, required_history = _facts_requirement(sector)
    years_by_line_item: dict[str, set[int]] = {}
    latest_period_end: str | None = None
    latest_filed_date: str | None = None
    source_tickers: set[str] = set()
    for row in rows:
        line_item = str(row["line_item"] or "")
        fiscal_year = row["fiscal_year"]
        if line_item and isinstance(fiscal_year, int):
            years_by_line_item.setdefault(line_item, set()).add(fiscal_year)
        period_end = str(row["period_end"] or "")
        filed_date = str(row["filed_date"] or "")
        source_ticker = str(row["ticker"] or "").strip().upper()
        if period_end and (latest_period_end is None or period_end > latest_period_end):
            latest_period_end = period_end
        if filed_date and (latest_filed_date is None or filed_date > latest_filed_date):
            latest_filed_date = filed_date
        if source_ticker:
            source_tickers.add(source_ticker)
    missing = [name for name in required_history if name not in years_by_line_item]
    insufficient = [
        name
        for name, minimum_years in required_history.items()
        if name in years_by_line_item
        and len(years_by_line_item[name]) < minimum_years
    ]
    status = READINESS_READY if not missing and not insufficient else READINESS_NEEDS_DATA
    reason_code = (
        None
        if status == READINESS_READY
        else "NORMALIZED_FACTS_REQUIRED_LINE_ITEMS_MISSING"
        if missing
        else "NORMALIZED_FACTS_HISTORY_INSUFFICIENT"
    )
    return {
        "status": status,
        "reason_code": reason_code,
        "issuer_cik": scope.issuer_cik,
        "requirement_kind": requirement_kind,
        "required_history_years_by_line_item": required_history,
        "history_years_by_line_item": {
            name: sorted(years_by_line_item.get(name, set()))
            for name in required_history
        },
        "missing_line_items": missing,
        "insufficient_history_line_items": insufficient,
        "annual_rows": len(rows),
        "annual_years": len(
            {year for years in years_by_line_item.values() for year in years}
        ),
        "latest_period_end": latest_period_end,
        "latest_filed_date": latest_filed_date,
        "source_tickers": sorted(source_tickers),
    }


def _filing_readiness(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    issuer_cik: str,
    as_of_date: str,
) -> dict[str, Any]:
    required = {"cik", "accession", "form_type", "filing_date", "status", "local_path"}
    if required - _table_columns(conn, "filings"):
        return {
            "status": READINESS_NEEDS_DATA,
            "reason_code": "FILINGS_TABLE_UNAVAILABLE",
            "parsing_status": READINESS_NEEDS_DATA,
            "parsing_reason_code": "PARSED_FILINGS_TABLE_UNAVAILABLE",
            "annual_record_count": 0,
            "readable_file_count": 0,
            "parsed_readable_count": 0,
        }
    scope, rows = issuer_filing_rows(
        conn,
        ticker,
        columns=("accession", "form_type", "filing_date", "status", "local_path"),
        issuer_cik=issuer_cik,
        aliases=(ticker,),
        form_types=ANNUAL_CACHED_FILING_FORM_TYPES,
        statuses=None,
        as_of_date=as_of_date,
    )
    parsed_accessions: set[str] = set()
    if "accession" in _table_columns(conn, "parsed_filings"):
        parsed_accessions = {
            str(row[0])
            for row in conn.execute("SELECT accession FROM parsed_filings").fetchall()
            if row[0]
        }
    readable_accessions: list[str] = []
    complete_accessions: list[str] = []
    latest_filing_date: str | None = None
    forms: set[str] = set()
    for row in rows:
        accession = str(row["accession"] or "")
        form_type = str(row["form_type"] or "").upper()
        filing_date = str(row["filing_date"] or "")
        status = str(row["status"] or "").strip().lower()
        local_path = str(row["local_path"] or "").strip()
        readable = False
        if local_path:
            try:
                readable = Path(local_path).is_file() and Path(local_path).stat().st_size > 0
            except OSError:
                readable = False
        if readable:
            readable_accessions.append(accession)
            if accession in parsed_accessions or status in {"ok", "parsed"}:
                complete_accessions.append(accession)
        if form_type:
            forms.add(form_type)
        if filing_date and (
            latest_filing_date is None or filing_date > latest_filing_date
        ):
            latest_filing_date = filing_date
    filing_status = READINESS_READY if readable_accessions else READINESS_NEEDS_DATA
    parsing_status = READINESS_READY if complete_accessions else READINESS_NEEDS_DATA
    return {
        "status": filing_status,
        "reason_code": None if readable_accessions else "ANNUAL_FILING_CONTENT_NOT_LOCAL",
        "parsing_status": parsing_status,
        "parsing_reason_code": (
            None
            if complete_accessions
            else "ANNUAL_FILING_PARSE_NOT_AVAILABLE"
            if readable_accessions
            else "ANNUAL_FILING_CONTENT_NOT_LOCAL"
        ),
        "issuer_cik": scope.issuer_cik,
        "annual_record_count": len(rows),
        "readable_file_count": len(readable_accessions),
        "parsed_readable_count": len(complete_accessions),
        "readable_accessions": readable_accessions,
        "complete_accessions": complete_accessions,
        "forms": sorted(forms),
        "latest_filing_date": latest_filing_date,
        "local_materialization_attempted": False,
    }


def _price_readiness(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    as_of_date: str,
) -> dict[str, Any]:
    required = {"ticker", "price", "currency", "provider", "as_of_date", "status"}
    if required - _table_columns(conn, "price_quotes"):
        return {
            "status": READINESS_NEEDS_DATA,
            "reason_code": "PRICE_QUOTES_TABLE_UNAVAILABLE",
            "snapshot": None,
        }
    row = conn.execute(
        """
        SELECT price, currency, provider, as_of_date, source_url, status
        FROM price_quotes
        WHERE ticker = ? AND as_of_date <= ? AND price IS NOT NULL
        ORDER BY as_of_date DESC, id DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if row is None:
        return {
            "status": READINESS_NEEDS_DATA,
            "reason_code": "PRICE_QUOTE_NOT_FOUND_AS_OF_DATE",
            "snapshot": None,
        }
    price = row["price"]
    currency = str(row["currency"] or "").upper()
    usable = (
        not isinstance(price, bool)
        and isinstance(price, (int, float))
        and float(price) > 0.0
        and currency == "USD"
    )
    snapshot = {
        "ticker": ticker,
        "price": float(price) if isinstance(price, (int, float)) else None,
        "currency": currency or None,
        "source": row["provider"],
        "as_of_date": row["as_of_date"],
        "url": sanitize_url_credentials(row["source_url"]),
        "quote_status": row["status"],
    }
    return {
        "status": READINESS_READY if usable else READINESS_NEEDS_DATA,
        "reason_code": None if usable else "PRICE_QUOTE_NOT_USABLE",
        "snapshot": snapshot,
    }


def _valuation_readiness(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    issuer_cik: str,
    as_of_date: str,
    price: dict[str, Any],
) -> dict[str, Any]:
    if not {"ticker", "method", "as_of_date", "inputs_json", "outputs_json"} <= (
        _table_columns(conn, "valuations")
    ):
        return {
            "status": READINESS_NEEDS_DATA,
            "reason_code": "VALUATIONS_TABLE_UNAVAILABLE",
            "raw_asof": None,
            "validated_asof": None,
            "mismatch_reasons": ["V2_SCORECARD_MISSING"],
        }
    state = validate_v2_scorecard_provenance(
        conn,
        ticker,
        as_of_date=as_of_date,
        issuer_cik=issuer_cik,
        issuer_aliases=(ticker,),
        price_snapshot=price.get("snapshot"),
    )
    ready = bool(state.get("validated_asof"))
    reasons = [str(item) for item in state.get("mismatch_reasons") or []]
    return {
        "status": READINESS_READY if ready else READINESS_NEEDS_DATA,
        "reason_code": None if ready else (reasons[0] if reasons else "V2_SCORECARD_MISSING"),
        "raw_asof": state.get("raw_asof"),
        "validated_asof": state.get("validated_asof"),
        "mismatch_reasons": reasons,
    }


def _candidate_readiness(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    identity: Mapping[str, Any],
    sector: str,
    as_of_date: str,
) -> dict[str, Any]:
    issuer_cik = str(identity.get("issuer_cik") or "")
    facts = _facts_readiness(
        conn,
        ticker=ticker,
        issuer_cik=issuer_cik,
        sector=sector,
        as_of_date=as_of_date,
    )
    filings = _filing_readiness(
        conn,
        ticker=ticker,
        issuer_cik=issuer_cik,
        as_of_date=as_of_date,
    )
    price = _price_readiness(conn, ticker=ticker, as_of_date=as_of_date)
    valuation = _valuation_readiness(
        conn,
        ticker=ticker,
        issuer_cik=issuer_cik,
        as_of_date=as_of_date,
        price=price,
    )
    stages = {
        "IDENTITY": {"status": READINESS_READY, "reason_code": None},
        "CAP": {"status": READINESS_READY, "reason_code": None},
        "FACTS": facts,
        "FILING": {
            "status": filings["status"],
            "reason_code": filings["reason_code"],
        },
        "PARSING": {
            "status": filings["parsing_status"],
            "reason_code": filings["parsing_reason_code"],
        },
        "PRICE": {
            "status": price["status"],
            "reason_code": price["reason_code"],
        },
        "VALUATION": {
            "status": valuation["status"],
            "reason_code": valuation["reason_code"],
        },
    }
    missing_inputs = [
        stage for stage, state in stages.items() if state.get("status") != READINESS_READY
    ]
    reason_codes = [
        str(state.get("reason_code"))
        for state in stages.values()
        if state.get("reason_code")
    ]
    return {
        "ticker": ticker,
        "sector": sector,
        "issuer_cik": normalize_cik(issuer_cik),
        "issuer_key": identity.get("issuer_key"),
        "security_key": identity.get("security_key"),
        "readiness": READINESS_READY if not missing_inputs else READINESS_NEEDS_DATA,
        "missing_inputs": missing_inputs,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "stage_readiness": stages,
        "facts": facts,
        "filings": filings,
        "price": price,
        "valuation": valuation,
        "packet_materialized": False,
    }


def _zero_usage() -> dict[str, Any]:
    return {
        "model_calls": 0,
        "search_calls": 0,
        "network_calls": 0,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }


def run_v2_readiness_preflight(
    *,
    sectors: list[str],
    candidate_payloads: Mapping[str, Mapping[str, Any]],
    as_of_date: str,
    market_cap_focus: str,
    execution_set_fingerprint: str,
    request_fingerprint: str,
    candidate_resolution_errors: Mapping[str, str] | None = None,
    accepted_census_authority: AcceptedCensusRunAuthority | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Classify the exact frozen v2 execution set using local read-only evidence."""

    cfg = cfg or get_config()
    validation_errors = [
        f"{sector}:{message}"
        for sector, message in sorted((candidate_resolution_errors or {}).items())
    ]
    lineage: dict[str, Any] = {}
    sector_bindings: dict[str, Any] = {}
    execution_identity: dict[str, dict[str, Any]] = {}
    try:
        authority = accepted_census_authority or AcceptedCensusRunAuthority()
        cohort = authority.load(db_path=cfg.db_path, as_of_date=as_of_date)
        lineage = cohort.lineage.to_dict()
        if validation_errors:
            raise ValueError("candidate resolution failed before readiness")
        sector_bindings, execution_identity = _validate_frozen_authority(
            sectors=sectors,
            candidate_payloads=candidate_payloads,
            as_of_date=as_of_date,
            market_cap_focus=market_cap_focus,
            execution_set_fingerprint=execution_set_fingerprint,
            request_fingerprint=request_fingerprint,
            cohort=cohort,
        )
    except Exception as exc:  # noqa: BLE001 - persist a fail-closed diagnostic
        validation_errors.append(f"{type(exc).__name__}:{exc}")

    if validation_errors:
        return {
            "schema_version": READINESS_PREFLIGHT_SCHEMA_VERSION,
            "status": READINESS_INCOMPLETE,
            "readiness_status": READINESS_INCOMPLETE,
            "mode": READINESS_MODE,
            "as_of_date": as_of_date,
            "market_cap_focus": market_cap_focus,
            "sectors": list(sectors),
            "request_fingerprint": request_fingerprint,
            "execution_set_fingerprint": execution_set_fingerprint,
            "census_lineage": lineage,
            "authority_validation_status": "FAILED",
            "validation_errors": validation_errors,
            "sector_bindings": sector_bindings,
            "sector_results": {},
            "counts": {
                "sector_count": len(sectors),
                "membership_candidates": 0,
                "execution_candidates": 0,
                "deferred_by_bound": 0,
                "excluded_candidates": 0,
                "ready": 0,
                "needs_data": 0,
                "incomplete": 0,
            },
            "readiness_counts": {},
            "missing_input_counts": {},
            "reason_code_counts": {},
            "actual_usage": _zero_usage(),
            "database_access": {"mode": "ro", "query_only": True, "writes": 0},
            "cache_writes": 0,
            "packet_materialized_count": 0,
            "production_sector_scan_exercised": False,
            "watchlist_mutation_exercised": False,
            "screened_candidate_count": 0,
            "underwritten_candidate_count": 0,
            "actionable_candidate_count": 0,
        }

    sector_results: dict[str, Any] = {}
    readiness_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    with _read_only_connection(cfg.db_path) as conn:
        for sector in sectors:
            binding = sector_bindings[sector]
            candidate_rows: list[dict[str, Any]] = []
            for ticker in binding["execution_tickers"]:
                try:
                    row = _candidate_readiness(
                        conn,
                        ticker=ticker,
                        identity=execution_identity[ticker],
                        sector=sector,
                        as_of_date=as_of_date,
                    )
                except Exception as exc:  # noqa: BLE001 - one explicit incomplete row
                    row = {
                        "ticker": ticker,
                        "sector": sector,
                        "issuer_cik": normalize_cik(
                            execution_identity[ticker].get("issuer_cik")
                        ),
                        "issuer_key": execution_identity[ticker].get("issuer_key"),
                        "security_key": execution_identity[ticker].get("security_key"),
                        "readiness": READINESS_INCOMPLETE,
                        "missing_inputs": [],
                        "reason_codes": [f"READINESS_QUERY_FAILED:{type(exc).__name__}"],
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "packet_materialized": False,
                    }
                candidate_rows.append(row)
                readiness_counter[str(row["readiness"])] += 1
                missing_counter.update(str(item) for item in row.get("missing_inputs") or [])
                reason_counter.update(str(item) for item in row.get("reason_codes") or [])
            sector_readiness = Counter(str(row["readiness"]) for row in candidate_rows)
            sector_results[sector] = {
                **binding,
                "status": (
                    READINESS_INCOMPLETE
                    if sector_readiness.get(READINESS_INCOMPLETE, 0)
                    else READINESS_READY
                    if candidate_rows
                    and sector_readiness.get(READINESS_READY, 0) == len(candidate_rows)
                    else "NO_EXECUTION_REQUIRED"
                    if not candidate_rows
                    else READINESS_NEEDS_DATA
                ),
                "readiness_counts": dict(sorted(sector_readiness.items())),
                "candidate_rows": candidate_rows,
            }

    membership_count = sum(
        len(binding["membership_tickers"]) for binding in sector_bindings.values()
    )
    execution_count = sum(
        len(binding["execution_tickers"]) for binding in sector_bindings.values()
    )
    deferred_count = sum(
        len(binding["deferred_by_bound_tickers"])
        for binding in sector_bindings.values()
    )
    excluded_count = sum(
        len(binding["excluded_tickers"]) for binding in sector_bindings.values()
    )
    classified_count = sum(readiness_counter.values())
    incomplete_count = int(readiness_counter.get(READINESS_INCOMPLETE, 0))
    classification_complete = classified_count == execution_count and incomplete_count == 0
    readiness_status = (
        READINESS_INCOMPLETE
        if not classification_complete
        else "NO_EXECUTION_REQUIRED"
        if execution_count == 0
        else READINESS_READY
        if readiness_counter.get(READINESS_READY, 0) == execution_count
        else READINESS_NEEDS_DATA
    )
    return {
        "schema_version": READINESS_PREFLIGHT_SCHEMA_VERSION,
        "status": "COMPLETED" if classification_complete else READINESS_INCOMPLETE,
        "readiness_status": readiness_status,
        "mode": READINESS_MODE,
        "as_of_date": as_of_date,
        "market_cap_focus": market_cap_focus,
        "sectors": list(sectors),
        "request_fingerprint": request_fingerprint,
        "execution_set_fingerprint": execution_set_fingerprint,
        "census_lineage": lineage,
        "authority_validation_status": "PASSED",
        "authority_fingerprint": _sha256_json(
            {
                "as_of_date": as_of_date,
                "market_cap_focus": market_cap_focus,
                "sectors": sectors,
                "request_fingerprint": request_fingerprint,
                "execution_set_fingerprint": execution_set_fingerprint,
                "census_lineage": lineage,
                "sector_bindings": sector_bindings,
            }
        ),
        "validation_errors": [],
        "sector_bindings": sector_bindings,
        "sector_results": sector_results,
        "classification_complete": classification_complete,
        "counts": {
            "sector_count": len(sectors),
            "membership_candidates": membership_count,
            "execution_candidates": execution_count,
            "deferred_by_bound": deferred_count,
            "excluded_candidates": excluded_count,
            "ready": int(readiness_counter.get(READINESS_READY, 0)),
            "needs_data": int(readiness_counter.get(READINESS_NEEDS_DATA, 0)),
            "incomplete": incomplete_count,
        },
        "readiness_counts": dict(sorted(readiness_counter.items())),
        "missing_input_counts": dict(sorted(missing_counter.items())),
        "reason_code_counts": dict(sorted(reason_counter.items())),
        "actual_usage": _zero_usage(),
        "database_access": {"mode": "ro", "query_only": True, "writes": 0},
        "cache_writes": 0,
        "packet_materialized_count": 0,
        "production_sector_scan_exercised": False,
        "watchlist_mutation_exercised": False,
        "screened_candidate_count": 0,
        "underwritten_candidate_count": 0,
        "actionable_candidate_count": 0,
    }


__all__ = [
    "READINESS_INCOMPLETE",
    "READINESS_MODE",
    "READINESS_NEEDS_DATA",
    "READINESS_PREFLIGHT_SCHEMA_VERSION",
    "READINESS_READY",
    "run_v2_readiness_preflight",
    "v2_execution_set_fingerprint",
]
