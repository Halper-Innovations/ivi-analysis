"""Primary-source helpers for insurance routing and packets."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

from app.config import AppConfig, get_config
from app.db import connect, get_db
from app.util.financial_data_access import (
    ANNUAL_CACHED_FILING_FORM_TYPES,
    VALID_CACHED_FILING_STATUSES,
    filing_rows,
    issuer_filing_rows,
)
from app.valuation.lineage import latest_decision_eligible_valuation_row


def _db_context(
    *,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
):
    """Open the explicitly supplied insurance data store when provided."""

    return get_db(cfg) if db_path is None else closing(connect(db_path, cfg=cfg))


def latest_company_profile(
    ticker: str,
    *,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Return the latest local company profile fields for a ticker."""
    upper = ticker.upper()
    if issuer_cik is None and not aliases and db_path is None and cfg is None:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT ticker, cik, name, notes
                FROM companies
                WHERE ticker = ?
                LIMIT 1
                """,
                (upper,),
            ).fetchone()
        if not row:
            return {"ticker": upper}
        return {
            "ticker": str(row["ticker"] or upper).upper(),
            "cik": row["cik"],
            "name": row["name"],
            "notes": row["notes"],
        }
    with _db_context(db_path=db_path, cfg=cfg) as conn:
        cik_digits = "".join(ch for ch in str(issuer_cik or "") if ch.isdigit())
        if cik_digits:
            row = conn.execute(
                """
                SELECT ticker, cik, name, notes
                FROM companies
                WHERE CAST(cik AS INTEGER) = ?
                ORDER BY CASE WHEN UPPER(ticker) = ? THEN 0 ELSE 1 END, ticker
                LIMIT 1
                """,
                (int(cik_digits), upper),
            ).fetchone()
        else:
            normalized_aliases = tuple(
                dict.fromkeys(
                    str(item).strip().upper()
                    for item in (upper, *aliases)
                    if str(item or "").strip()
                )
            )
            placeholders = ", ".join("?" for _ in normalized_aliases)
            row = conn.execute(
                f"""
                SELECT ticker, cik, name, notes
                FROM companies
                WHERE UPPER(ticker) IN ({placeholders})
                ORDER BY CASE WHEN UPPER(ticker) = ? THEN 0 ELSE 1 END, ticker
                LIMIT 1
                """,
                (*normalized_aliases, upper),
            ).fetchone()
    if not row:
        return {"ticker": upper}
    return {
        "ticker": str(row["ticker"] or ticker.upper()).upper(),
        "cik": row["cik"],
        "name": row["name"],
        "notes": row["notes"],
    }


def _normalize_cik(cik: str | int | None) -> str:
    token = "".join(ch for ch in str(cik or "") if ch.isdigit())
    return token.zfill(10) if token else ""


def cached_submission_profile(
    cik: str | int | None,
    *,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Return cached SEC submissions identity fields for a CIK, if present."""
    cik_norm = _normalize_cik(cik)
    if not cik_norm:
        return {}
    path = (cfg or get_config()).cache_dir / "submissions" / f"{cik_norm}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    tickers = [
        str(ticker).upper() for ticker in (payload.get("tickers") or []) if str(ticker).strip()
    ]
    exchanges = [
        str(exchange) for exchange in (payload.get("exchanges") or []) if str(exchange).strip()
    ]
    return {
        "cik": cik_norm,
        "issuer_name": payload.get("name"),
        "issuer_primary_ticker": tickers[0] if tickers else None,
        "issuer_listed_tickers": tickers,
        "issuer_exchanges": exchanges,
        "source": str(path),
    }


def latest_sector(
    ticker: str,
    *,
    as_of_date: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> str | None:
    """Return the latest inferred sector for a ticker, if available."""
    upper = ticker.upper()
    if as_of_date is None and not aliases and db_path is None and cfg is None:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT inferred_sector
                FROM sector_inference
                WHERE ticker = ?
                ORDER BY as_of_date DESC, id DESC
                LIMIT 1
                """,
                (upper,),
            ).fetchone()
        if not row or row["inferred_sector"] is None:
            return None
        return str(row["inferred_sector"])
    normalized_aliases = tuple(
        dict.fromkeys(
            str(item).strip().upper() for item in (upper, *aliases) if str(item or "").strip()
        )
    )
    placeholders = ", ".join("?" for _ in normalized_aliases)
    clauses = [f"UPPER(ticker) IN ({placeholders})"]
    params: list[Any] = list(normalized_aliases)
    if as_of_date:
        clauses.append("as_of_date <= ?")
        params.append(as_of_date)
    params.append(upper)
    with _db_context(db_path=db_path, cfg=cfg) as conn:
        row = conn.execute(
            f"""
            SELECT inferred_sector
            FROM sector_inference
            WHERE {" AND ".join(clauses)}
            ORDER BY CASE WHEN UPPER(ticker) = ? THEN 0 ELSE 1 END,
                     as_of_date DESC, id DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    if not row or row["inferred_sector"] is None:
        return None
    return str(row["inferred_sector"])


def latest_scorecard(
    ticker: str,
    *,
    as_of_date: str | None = None,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    require_exact_issuer_binding: bool = False,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Return ``(as_of_date, outputs_json)`` for the latest scorecard."""
    with _db_context(db_path=db_path, cfg=cfg) as conn:
        row = latest_decision_eligible_valuation_row(
            conn,
            ticker=ticker,
            method="scorecard",
            as_of_date=as_of_date,
            expected_issuer_cik=issuer_cik,
            expected_issuer_aliases=aliases,
            require_exact_issuer_binding=require_exact_issuer_binding,
        )
    if not row:
        return None, {}
    try:
        return str(row["as_of_date"]), json.loads(row["outputs_json"] or "{}")
    except json.JSONDecodeError:
        return str(row["as_of_date"]), {}


def latest_cached_filing_text(
    ticker: str,
    *,
    as_of_date: str | None = None,
    max_chars: int = 350_000,
    issuer_cik: str | None = None,
    aliases: Sequence[str] = (),
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> tuple[str, dict[str, Any]]:
    """Read the latest cached annual filing text from the primary local cache."""
    with _db_context(db_path=db_path, cfg=cfg) as conn:
        kwargs = {
            "columns": (
                "cik",
                "form_type",
                "filing_date",
                "period_end",
                "accession",
                "local_path",
                "primary_doc_url",
            ),
            "form_types": ANNUAL_CACHED_FILING_FORM_TYPES,
            "statuses": VALID_CACHED_FILING_STATUSES,
            "as_of_date": as_of_date,
            "require_local_path": True,
            "limit": 5,
        }
        if issuer_cik is not None or aliases or db_path is not None:
            _scope, rows = issuer_filing_rows(
                conn,
                ticker,
                issuer_cik=issuer_cik,
                aliases=aliases,
                **kwargs,
            )
        else:
            rows = filing_rows(conn, ticker, **kwargs)

    for row in rows:
        raw_path = str(row["local_path"] or "")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        return text[:max_chars], {
            "cik": row["cik"],
            "form_type": row["form_type"],
            "filing_date": row["filing_date"],
            "period_end": row["period_end"],
            "accession": row["accession"],
            "source_url": row["primary_doc_url"],
            "local_path": raw_path,
        }
    return "", {}
