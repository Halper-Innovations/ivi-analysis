from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlparse

from app.autonomous.financial_integrity import (
    PRICE_BASIS_SPLIT_ADJUSTED,
    PRICE_BASIS_UNADJUSTED,
    PRICE_UNIT_USD_PER_SHARE,
    SHARES_BASIS_UNADJUSTED,
    SHARES_UNIT_MILLIONS,
    authoritative_split_proof_reference,
    stable_quote_hash,
)
from app.config import AppConfig, get_config
from app.db import get_db, utc_now_iso
from app.market.company_facts_provider import companyfacts_cache_path, normalize_cik
from app.market.shares_guard import GUARD_REFUSED, GUARDED_HIT_REASON
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.fundamentals import UNKNOWN

_SEC_COMPANYFACTS_SOURCE_PREFIX = "SEC_COMPANYFACTS"
_SEC_COMPANYFACTS_PATH_RE = re.compile(
    r"^/api/xbrl/companyfacts/CIK(?P<cik>\d{1,10})\.json$",
    re.IGNORECASE,
)
_SEC_COMPANYFACTS_HOSTS = {"data.sec.gov", "www.sec.gov", "sec.gov"}

logger = logging.getLogger(__name__)

# --- Share-count guard -------------------------------------------------------
#
# A cover-page share count filed a thousandfold off (ResMed FY2021: 145,681 for
# 145.6 million) scales every per-share number -- intrinsic value, margin of
# safety, the buy-price target -- with the slip while looking plausible. The
# guard that catches it sits with the chooser, in app/market/shares_guard.py, so
# every consumer of the CompanyFacts share pick gets the same answer. This module
# reports the guard's decision (coverage["shares_guard"], and the reason code)
# and never substitutes for a count the guard refused.
#
# It replaces the power-of-ten guard this module used to carry,
# which compared the freshest cover count with the freshest balance-sheet count
# from any filing and always preferred the balance sheet: it swapped TO a slipped
# balance-sheet count when the cover was right, and saw nothing where a filing
# carries no balance-sheet share tag (Chesapeake Utilities' 10-Q covers, Packaging
# Corp's 2026 10-K cover).
#
# Relationship to app/valuation/guards.py: validate_denominators there rejects a
# zero, negative, non-finite or astronomically large share count, and nothing
# else -- a single absolute ceiling, deliberately loose, because that function is
# reached from more than one call site and the callers do not agree on whether
# they pass absolute shares or millions. Nothing there can see a unit slip:
# 145,681 shares where 145.6 million belong is a perfectly ordinary-looking count,
# and only a cross-basis comparison catches it.


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except Exception:
        return None


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _trace_refs(trace_bucket: dict[str, Any] | None, fallback: str) -> list[str]:
    refs = [str(x) for x in ((trace_bucket or {}).get("derived_from") or []) if str(x).strip()]
    return refs or [fallback]


def _reference_without_fragment(value: Any) -> str:
    return urldefrag(str(value or "").strip())[0]


def _companyfacts_reference_cik(value: Any) -> str | None:
    parsed = urlparse(_reference_without_fragment(value))
    match = _SEC_COMPANYFACTS_PATH_RE.fullmatch(parsed.path)
    if match is None:
        return None
    normalized = normalize_cik(match.group("cik"))
    return normalized or None


def _db_companyfacts_share_matches(
    *,
    cfg: AppConfig,
    ticker: str,
    value: float,
    unit: str,
    period_end: str,
    filed_date: str,
    source_reference: str,
) -> bool:
    """Bind an artifact trace to one exact persisted CompanyFacts row."""

    parsed_reference = urlparse(_reference_without_fragment(source_reference))
    reference_match = _SEC_COMPANYFACTS_PATH_RE.fullmatch(parsed_reference.path)
    reference_cik = (
        normalize_cik(reference_match.group("cik")) if reference_match is not None else ""
    )
    if (
        parsed_reference.scheme.lower() != "https"
        or (parsed_reference.hostname or "").lower() not in _SEC_COMPANYFACTS_HOSTS
        or not reference_cik
    ):
        return False
    try:
        with get_db(cfg) as conn:
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
            }
            if not {"units", "filed_date", "source_url"}.issubset(columns):
                return False
            issuer_row = conn.execute(
                "SELECT cik FROM companies WHERE ticker = ? LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
            if issuer_row is None or normalize_cik(issuer_row["cik"]) != reference_cik:
                return False
            rows = conn.execute(
                """
                SELECT value, units, period_end, filed_date, source_url
                FROM companyfacts_facts
                WHERE ticker = ?
                  AND line_item = 'shares_outstanding'
                  AND period_end = ?
                  AND filed_date = ?
                """,
                (ticker.upper(), period_end, filed_date),
            ).fetchall()
    except Exception:
        return False

    expected_reference = _reference_without_fragment(source_reference)
    for row in rows:
        row_value = row["value"]
        if not _is_num(row_value):
            continue
        if str(row["units"] or "").strip() != unit:
            continue
        if not math.isclose(float(row_value), float(value), rel_tol=1e-9, abs_tol=1e-9):
            continue
        if _reference_without_fragment(row["source_url"]) != expected_reference:
            continue
        return True
    return False


def _cached_companyfacts_share_matches(
    *,
    cfg: AppConfig,
    ticker: str,
    value: float,
    unit: str,
    period_end: str,
    filed_date: str,
    source_reference: str,
) -> bool:
    """Bind a raw SEC trace to the exact cached CompanyFacts observation."""

    parsed = urlparse(_reference_without_fragment(source_reference))
    match = _SEC_COMPANYFACTS_PATH_RE.fullmatch(parsed.path)
    if match is None:
        return False
    reference_cik = normalize_cik(match.group("cik"))
    if not reference_cik:
        return False

    try:
        with get_db(cfg) as conn:
            issuer_row = conn.execute(
                "SELECT cik FROM companies WHERE ticker = ? LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
    except Exception:
        issuer_row = None
    if issuer_row is None or normalize_cik(issuer_row["cik"]) != reference_cik:
        return False

    payload = _safe_json(
        companyfacts_cache_path(
            reference_cik,
            cfg=cfg,
            create_parent=False,
        )
    )
    companyfacts = (
        payload.get("companyfacts") if isinstance(payload.get("companyfacts"), dict) else payload
    )
    if not isinstance(companyfacts, dict):
        return False
    payload_cik = normalize_cik(companyfacts.get("cik") or payload.get("cik"))
    if payload_cik and payload_cik != reference_cik:
        return False
    facts = companyfacts.get("facts")
    if not isinstance(facts, dict):
        return False

    share_tags = (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockOtherSharesOutstanding"),
    )
    for taxonomy, tag in share_tags:
        taxonomy_node = facts.get(taxonomy)
        tag_node = taxonomy_node.get(tag) if isinstance(taxonomy_node, dict) else None
        units = tag_node.get("units") if isinstance(tag_node, dict) else None
        rows = units.get(unit) if isinstance(units, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not _is_num(row.get("val")):
                continue
            if str(row.get("end") or "").strip()[:10] != period_end:
                continue
            if str(row.get("filed") or "").strip()[:10] != filed_date:
                continue
            if math.isclose(float(row["val"]), float(value), rel_tol=1e-9, abs_tol=1e-9):
                return True
    return False


def _artifact_share_source_is_backed(
    *,
    cfg: AppConfig,
    ticker: str,
    value: float,
    unit: str,
    period_end: str,
    filed_date: str,
    source: str,
    source_reference: str,
) -> bool:
    """Reject self-attested SEC-looking URLs that have no exact local source."""

    parsed = urlparse(_reference_without_fragment(source_reference))
    host = (parsed.hostname or "").lower()
    if (
        not source.strip().upper().startswith(_SEC_COMPANYFACTS_SOURCE_PREFIX)
        or parsed.scheme.lower() != "https"
        or host not in _SEC_COMPANYFACTS_HOSTS
    ):
        return False
    return _db_companyfacts_share_matches(
        cfg=cfg,
        ticker=ticker,
        value=value,
        unit=unit,
        period_end=period_end,
        filed_date=filed_date,
        source_reference=source_reference,
    ) or _cached_companyfacts_share_matches(
        cfg=cfg,
        ticker=ticker,
        value=value,
        unit=unit,
        period_end=period_end,
        filed_date=filed_date,
        source_reference=source_reference,
    )


def _explicit_artifact_share_lineage(
    payload: dict[str, Any],
    *,
    ticker: str,
    value: float,
    provenance: dict[str, Any],
    requested_as_of: str,
    cfg: AppConfig,
) -> dict[str, Any]:
    raw_source_value = payload.get("raw_shares_source_value")
    raw_shares_mm = payload.get("raw_shares_outstanding_mm")
    normalized_shares_mm = payload.get("shares_outstanding_mm", value)
    factor = payload.get("split_adjustment_factor")
    basis = str(payload.get("shares_basis") or "").strip().upper()
    effective_date = str(payload.get("split_effective_date") or "").strip() or None
    declared_unit = str(provenance.get("unit") or "").strip()
    period_end = str(provenance.get("period_end") or "").strip()[:10]
    filed_date = str(provenance.get("filed_date") or "").strip()[:10]
    source = str(provenance.get("source") or "").strip()
    source_reference = str(provenance.get("source_reference") or "").strip()
    provenance_value = provenance.get("value")
    if not declared_unit:
        return {
            "shares_lineage_status": "NEEDS_DATA",
            "shares_lineage_reason": "SHARES_UNIT_MISSING",
        }
    if declared_unit not in {"shares", SHARES_UNIT_MILLIONS}:
        return {
            "shares_lineage_status": "INVALID_FINANCIAL_INPUT",
            "shares_lineage_reason": "SHARES_UNIT_INVALID",
        }
    if (
        not _is_num(provenance_value)
        or not period_end
        or not filed_date
        or not source
        or not source_reference
    ):
        return {
            "shares_lineage_status": "NEEDS_DATA",
            "shares_lineage_reason": "SHARES_FILED_PROVENANCE_MISSING",
        }
    if not source.upper().startswith(_SEC_COMPANYFACTS_SOURCE_PREFIX):
        return {
            "shares_lineage_status": "INVALID_FINANCIAL_INPUT",
            "shares_lineage_reason": "SHARES_SOURCE_REFERENCE_UNTRUSTED",
        }
    source_reference_url = urlparse(source_reference)
    if (
        source_reference_url.scheme.lower() != "https"
        or (source_reference_url.hostname or "").lower() not in _SEC_COMPANYFACTS_HOSTS
    ):
        return {
            "shares_lineage_status": "INVALID_FINANCIAL_INPUT",
            "shares_lineage_reason": "SHARES_SOURCE_REFERENCE_UNTRUSTED",
        }
    period_dt = _parse_date(period_end)
    filed_dt = _parse_date(filed_date)
    requested_dt = _parse_date(requested_as_of)
    if (
        period_dt is None
        or filed_dt is None
        or requested_dt is None
        or period_dt > filed_dt
        or filed_dt > requested_dt
    ):
        return {
            "shares_lineage_status": "INVALID_FINANCIAL_INPUT",
            "shares_lineage_reason": "SHARES_FILED_PROVENANCE_CONFLICT",
        }
    if not _artifact_share_source_is_backed(
        cfg=cfg,
        ticker=ticker,
        value=float(provenance_value),
        unit=declared_unit,
        period_end=period_end,
        filed_date=filed_date,
        source=source,
        source_reference=source_reference,
    ):
        return {
            "shares_lineage_status": "INVALID_FINANCIAL_INPUT",
            "shares_lineage_reason": "SHARES_SOURCE_REFERENCE_UNVERIFIED",
        }
    if not (
        _is_num(raw_source_value)
        and float(raw_source_value) > 0
        and _is_num(raw_shares_mm)
        and float(raw_shares_mm) > 0
        and _is_num(normalized_shares_mm)
        and float(normalized_shares_mm) > 0
        and _is_num(factor)
        and float(factor) > 0
        and basis in {SHARES_BASIS_UNADJUSTED, PRICE_BASIS_SPLIT_ADJUSTED}
    ):
        return {
            "shares_lineage_status": "NEEDS_DATA",
            "shares_lineage_reason": "SHARES_SPLIT_LINEAGE_MISSING",
        }
    unit_scale = 1.0 / 1_000_000.0 if declared_unit == "shares" else 1.0
    raw_source_number = float(raw_source_value)
    raw_number = float(raw_shares_mm)
    normalized_number = float(normalized_shares_mm)
    value_number = float(value)
    provenance_value_number = float(provenance_value)
    provenance_value_mm = provenance_value_number * unit_scale
    factor_number = float(factor)
    reconciles = (
        basis == SHARES_BASIS_UNADJUSTED
        and math.isclose(raw_number, normalized_number, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(factor_number, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ) or (
        basis == PRICE_BASIS_SPLIT_ADJUSTED
        and math.isclose(
            raw_number * factor_number,
            normalized_number,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and (
            math.isclose(factor_number, 1.0, rel_tol=0.0, abs_tol=1e-12)
            or effective_date is not None
        )
    )
    if (
        not reconciles
        or not math.isclose(
            raw_source_number,
            provenance_value_number,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            raw_number,
            provenance_value_mm,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            value_number,
            normalized_number,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        return {
            "shares_lineage_status": "INVALID_FINANCIAL_INPUT",
            "shares_lineage_reason": "SHARES_SPLIT_LINEAGE_CONFLICT",
        }
    return {
        "shares_lineage_status": "PASS",
        "shares_lineage_reason": None,
        "shares_value_mm": normalized_number,
        "raw_shares_source_value": raw_source_number,
        "raw_shares_source_unit": declared_unit,
        "raw_shares_outstanding_mm": raw_number,
        "normalized_shares_outstanding_mm": normalized_number,
        "shares_unit": SHARES_UNIT_MILLIONS,
        "shares_basis": basis,
        "shares_asof_used": period_end,
        "shares_filed_date": filed_date,
        "shares_source": source,
        "shares_source_url": source_reference,
        "shares_declared_source_unit": declared_unit,
        "shares_split_adjustment_factor": factor_number,
        "shares_split_effective_date": effective_date,
    }


def _artifact_share_provenance(
    payload: dict[str, Any],
    *,
    row: dict[str, Any] | None = None,
    trace_bucket: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace_bucket = trace_bucket or {}
    _ = payload, row
    return {
        "value": trace_bucket.get("value"),
        "unit": trace_bucket.get("unit"),
        "period_end": trace_bucket.get("period_end"),
        "filed_date": trace_bucket.get("filed_date"),
        "source": trace_bucket.get("source"),
        "source_reference": trace_bucket.get("source_reference"),
    }


def _extract_shares_from_fundamentals(
    payload: dict[str, Any], *, ticker: str
) -> tuple[float | None, str | None, list[str], dict[str, Any]]:
    for key in ("shares_outstanding_latest", "latest_shares_outstanding", "shares_latest"):
        value = payload.get(key)
        if _is_num(value) and float(value) > 0:
            return (
                float(value),
                str(payload.get("as_of_date") or ""),
                [f"sector.fundamentals[{ticker}].{key}"],
                _artifact_share_provenance(
                    payload,
                    trace_bucket=(
                        payload.get("shares_trace")
                        if isinstance(payload.get("shares_trace"), dict)
                        else None
                    ),
                ),
            )

    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    row_traces = payload.get("row_traces") if isinstance(payload.get("row_traces"), dict) else {}
    rows = sorted(rows, key=lambda row: int(row.get("year", 0)))
    for row in reversed(rows):
        value = row.get("shares_outstanding")
        if not (_is_num(value) and float(value) > 0):
            continue
        year = int(row.get("year", 0))
        trace_bucket = (
            (row_traces.get(str(year)) or {}).get("shares_outstanding")
            if isinstance(row_traces.get(str(year)), dict)
            else None
        )
        refs = _trace_refs(
            trace_bucket, f"sector.fundamentals[{ticker}].rows[{year}].shares_outstanding"
        )
        return (
            float(value),
            str(payload.get("as_of_date") or ""),
            refs,
            _artifact_share_provenance(
                payload,
                row=row,
                trace_bucket=trace_bucket,
            ),
        )
    return None, None, [], {}


def _extract_shares_from_dossier(
    payload: dict[str, Any], *, run_id: str, ticker: str
) -> tuple[float | None, str | None, list[str], dict[str, Any]]:
    ts = payload.get("time_series") if isinstance(payload.get("time_series"), dict) else {}
    rows = [row for row in (ts.get("standardized_rows") or []) if isinstance(row, dict)]
    trace_map = (
        ts.get("standardized_row_traces")
        if isinstance(ts.get("standardized_row_traces"), dict)
        else {}
    )
    rows = sorted(rows, key=lambda row: int(row.get("year", 0)))
    for row in reversed(rows):
        value = row.get("shares_outstanding")
        if not (_is_num(value) and float(value) > 0):
            continue
        year = int(row.get("year", 0))
        trace_bucket = (
            (trace_map.get(str(year)) or {}).get("shares_outstanding")
            if isinstance(trace_map.get(str(year)), dict)
            else None
        )
        refs = _trace_refs(
            trace_bucket,
            f"dossiers.{run_id}.{ticker}.time_series.standardized_rows[{year}].shares_outstanding",
        )
        return (
            float(value),
            str(payload.get("as_of_date") or ""),
            refs,
            _artifact_share_provenance(
                payload,
                row=row,
                trace_bucket=trace_bucket,
            ),
        )
    return None, None, [], {}


def _iter_historical_dossiers(
    *, cfg: AppConfig, ticker: str, as_of_date: str, skip_run_id: str | None
) -> list[tuple[datetime, str, Path, dict[str, Any]]]:
    requested_dt = _parse_date(as_of_date)
    if requested_dt is None:
        return []
    candidates: list[tuple[datetime, str, Path, dict[str, Any]]] = []
    pattern = f"*/{ticker.upper()}/dossier.json"
    for path in sorted(cfg.dossiers_dir.glob(pattern)):
        run_id = path.parent.parent.name
        if skip_run_id and run_id == skip_run_id:
            continue
        payload = _safe_json(path)
        asof = str(payload.get("as_of_date") or "").strip()
        asof_dt = _parse_date(asof)
        if asof_dt is None or asof_dt > requested_dt:
            continue
        candidates.append((asof_dt, run_id, path, payload))
    return candidates


def _select_historical_candidate(
    candidates: list[tuple[datetime, str, Path, dict[str, Any]]],
) -> tuple[datetime, str, Path, dict[str, Any]] | None:
    if not candidates:
        return None
    max_date = max(row[0] for row in candidates)
    same_date = [row for row in candidates if row[0] == max_date]
    same_date.sort(key=lambda row: row[1])
    return same_date[0]


def _ordered_historical_candidates(
    candidates: list[tuple[datetime, str, Path, dict[str, Any]]],
) -> list[tuple[datetime, str, Path, dict[str, Any]]]:
    return sorted(candidates, key=lambda row: (-row[0].toordinal(), row[1]))


def _load_market_cap_snapshot(
    *,
    ticker: str,
    as_of_date: str,
    cfg: AppConfig | None = None,
) -> tuple[float | None, list[str]]:
    try:
        with get_db(cfg) as conn:
            row = conn.execute(
                """
                SELECT market_cap, effective_as_of_date, run_id
                FROM market_caps
                WHERE ticker = ?
                  AND effective_as_of_date <= ?
                  AND market_cap_status = 'OK'
                  AND market_cap IS NOT NULL
                ORDER BY effective_as_of_date DESC, COALESCE(run_id, '') ASC
                LIMIT 1
                """,
                (ticker.upper(), as_of_date),
            ).fetchone()
    except Exception:
        return None, []
    if row is None or not _is_num(row["market_cap"]):
        return None, []
    refs = [
        f"market_caps[{ticker.upper()}].effective_as_of_date={str(row['effective_as_of_date'] or '')}",
    ]
    if row["run_id"]:
        refs.append(f"market_caps[{ticker.upper()}].run_id={str(row['run_id'])}")
    return float(row["market_cap"]), refs


def _load_price_snapshot(
    *,
    ticker: str,
    as_of_date: str,
    cfg: AppConfig | None = None,
) -> tuple[float | None, list[str]]:
    try:
        with get_db(cfg) as conn:
            row = conn.execute(
                """
                SELECT price, as_of_date, provider
                FROM price_quotes
                WHERE ticker = ?
                  AND as_of_date <= ?
                  AND status = 'OK'
                  AND price IS NOT NULL
                ORDER BY as_of_date DESC, fetched_at DESC
                LIMIT 1
                """,
                (ticker.upper(), as_of_date),
            ).fetchone()
    except Exception:
        return None, []
    if row is None or not _is_num(row["price"]) or float(row["price"]) <= 0:
        return None, []
    return (
        float(row["price"]),
        [
            (
                f"price_quotes[{ticker.upper()}].as_of_date={str(row['as_of_date'] or '')}"
                f":provider={str(row['provider'] or '')}"
            )
        ],
    )


def resolve_shares_asof(
    ticker: str,
    as_of_date: str,
    run_id: str | None,
    *,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    companyfacts_cache_only: bool = False,
) -> tuple[float | None, dict[str, Any]]:
    cfg = cfg or get_config()
    if db_path is not None:
        cfg = cfg.model_copy(update={"db_path": Path(db_path)})
    ticker_norm = str(ticker or "").upper().strip()
    asof_norm = str(as_of_date or "").strip()
    coverage: dict[str, Any] = {
        "ticker": ticker_norm,
        "requested_as_of": asof_norm,
        "run_id": run_id,
        "shares_status": "UNKNOWN",
        "shares_reason_code": "NO_CURRENT_RUN_SHARES",
        "shares_reason_detail": "No shares_outstanding found in current run artifacts.",
        "shares_value": UNKNOWN,
        "shares_asof_used": None,
        "shares_filed_date": None,
        "shares_source": None,
        "shares_source_url": None,
        "shares_source_resolution": "unknown",
        "shares_lineage_status": "NEEDS_DATA",
        "shares_lineage_reason": "SHARES_SPLIT_LINEAGE_MISSING",
        "raw_shares_outstanding_mm": None,
        "normalized_shares_outstanding_mm": None,
        "shares_unit": None,
        "shares_basis": None,
        "shares_split_adjustment_factor": None,
        "shares_split_effective_date": None,
        "companyfacts_reason_code": None,
        "companyfacts_reason_detail": None,
        "shares_guard": None,
        "derived_from": [],
    }
    artifact_lineage_failure: dict[str, Any] = {}
    try:
        if not ticker_norm or not asof_norm:
            coverage["shares_reason_code"] = "EXCEPTION"
            coverage["shares_reason_detail"] = "Ticker or as_of_date is empty."
            return None, coverage

        if run_id:
            fundamentals_path = cfg.sectors_dir / run_id / f"fundamentals_{ticker_norm}.json"
            fundamentals = _safe_json(fundamentals_path)
            value, _used_asof, refs, provenance = _extract_shares_from_fundamentals(
                fundamentals, ticker=ticker_norm
            )
            if _is_num(value) and float(value) > 0:
                lineage = _explicit_artifact_share_lineage(
                    fundamentals,
                    ticker=ticker_norm,
                    value=float(value),
                    provenance=provenance,
                    requested_as_of=asof_norm,
                    cfg=cfg,
                )
                if lineage.get("shares_lineage_status") == "PASS":
                    resolved_value = float(lineage["shares_value_mm"])
                    coverage.update(
                        {
                            "shares_status": "OK",
                            "shares_reason_code": "OK",
                            "shares_reason_detail": "Resolved from current run fundamentals.",
                            "shares_value": resolved_value,
                            "shares_source_resolution": "current_run_fundamentals",
                            "derived_from": refs + [str(fundamentals_path)],
                            **lineage,
                        }
                    )
                    return resolved_value, coverage
                artifact_lineage_failure = {
                    "shares_status": lineage.get("shares_lineage_status"),
                    "shares_reason_code": lineage.get("shares_lineage_reason"),
                    "shares_reason_detail": (
                        "Current-run fundamentals shares lacked valid explicit "
                        "unit, filed-as-of provenance, or split lineage."
                    ),
                    **lineage,
                }

            dossier_path = cfg.dossiers_dir / run_id / ticker_norm / "dossier.json"
            dossier_payload = _safe_json(dossier_path)
            value, _used_asof, refs, provenance = _extract_shares_from_dossier(
                dossier_payload, run_id=run_id, ticker=ticker_norm
            )
            if _is_num(value) and float(value) > 0:
                lineage = _explicit_artifact_share_lineage(
                    dossier_payload,
                    ticker=ticker_norm,
                    value=float(value),
                    provenance=provenance,
                    requested_as_of=asof_norm,
                    cfg=cfg,
                )
                if lineage.get("shares_lineage_status") == "PASS":
                    resolved_value = float(lineage["shares_value_mm"])
                    coverage.update(
                        {
                            "shares_status": "OK",
                            "shares_reason_code": "OK",
                            "shares_reason_detail": "Resolved from current run dossier.",
                            "shares_value": resolved_value,
                            "shares_source_resolution": "current_run_dossier",
                            "derived_from": refs + [str(dossier_path)],
                            **lineage,
                        }
                    )
                    return resolved_value, coverage
                if not artifact_lineage_failure:
                    artifact_lineage_failure = {
                        "shares_status": lineage.get("shares_lineage_status"),
                        "shares_reason_code": lineage.get("shares_lineage_reason"),
                        "shares_reason_detail": (
                            "Current-run dossier shares lacked valid explicit "
                            "unit, filed-as-of provenance, or split lineage."
                        ),
                        **lineage,
                    }

        historical_candidates = _iter_historical_dossiers(
            cfg=cfg,
            ticker=ticker_norm,
            as_of_date=asof_norm,
            skip_run_id=run_id,
        )
        for _, hist_run_id, path, payload in _ordered_historical_candidates(historical_candidates):
            value, _used_asof, refs, provenance = _extract_shares_from_dossier(
                payload, run_id=hist_run_id, ticker=ticker_norm
            )
            if not (_is_num(value) and float(value) > 0):
                continue
            lineage = _explicit_artifact_share_lineage(
                payload,
                ticker=ticker_norm,
                value=float(value),
                provenance=provenance,
                requested_as_of=asof_norm,
                cfg=cfg,
            )
            if lineage.get("shares_lineage_status") == "PASS":
                resolved_value = float(lineage["shares_value_mm"])
                coverage.update(
                    {
                        "shares_status": "OK",
                        "shares_reason_code": "HISTORICAL_DOSSIER_HIT",
                        "shares_reason_detail": (
                            "Resolved shares_outstanding from historical dossier fallback."
                        ),
                        "shares_value": resolved_value,
                        "shares_source_resolution": "historical_dossier",
                        "derived_from": refs + [str(path)],
                        **lineage,
                    }
                )
                return resolved_value, coverage
            if not artifact_lineage_failure:
                artifact_lineage_failure = {
                    "shares_status": lineage.get("shares_lineage_status"),
                    "shares_reason_code": lineage.get("shares_lineage_reason"),
                    "shares_reason_detail": (
                        "Historical dossier shares lacked valid explicit "
                        "unit, filed-as-of provenance, or split lineage."
                    ),
                    **lineage,
                }

        base_reason_code = (
            "NO_HISTORICAL_DOSSIER" if not historical_candidates else "NO_CURRENT_RUN_SHARES"
        )
        base_reason_detail = (
            "No historical dossiers at or before requested as_of_date."
            if not historical_candidates
            else "Current run and selected historical dossiers did not contain shares_outstanding."
        )
        if artifact_lineage_failure:
            coverage.update(artifact_lineage_failure)
        else:
            coverage["shares_reason_code"] = base_reason_code
            coverage["shares_reason_detail"] = base_reason_detail

        facts_row = resolve_financial_facts_asof(
            ticker=ticker_norm,
            as_of_date=asof_norm,
            run_id=run_id,
            refresh=False,
            cfg=cfg,
            cache_only=companyfacts_cache_only,
        )
        companyfacts_reason_code = str(facts_row.get("fetch_reason_code") or "COMPANYFACTS_MISS")
        companyfacts_reason_detail = str(facts_row.get("fetch_reason_detail") or "")
        coverage["companyfacts_reason_code"] = companyfacts_reason_code
        coverage["companyfacts_reason_detail"] = companyfacts_reason_detail

        facts_shares = facts_row.get("shares_value")
        facts_source = str(facts_row.get("source_resolution") or "unknown")
        facts_refs = [str(ref) for ref in (facts_row.get("derived_from") or []) if str(ref).strip()]
        shares_guard = facts_row.get("shares_guard")
        shares_guard = shares_guard if isinstance(shares_guard, dict) else None
        coverage["shares_guard"] = shares_guard
        if _is_num(facts_shares) and float(facts_shares) > 0:
            resolved_shares = float(facts_shares)
            coverage.update(
                {
                    "shares_status": "OK",
                    "shares_reason_code": "COMPANYFACTS_HIT",
                    "shares_reason_detail": "Resolved shares_outstanding from SEC companyfacts.",
                    "shares_value": float(facts_shares),
                    "shares_asof_used": facts_row.get("shares_asof_used") or asof_norm,
                    "shares_filed_date": facts_row.get("shares_filed_date"),
                    "shares_source": facts_source,
                    "shares_source_url": facts_row.get("source_url"),
                    "shares_source_resolution": facts_source,
                    "derived_from": facts_refs,
                    "shares_lineage_status": "NEEDS_DATA",
                    "shares_lineage_reason": ("NO_SPLIT_EVENT_OR_NO_INTERVENING_SPLIT_PROOF"),
                    "raw_shares_source_value": facts_row.get("shares_raw_value"),
                    "raw_shares_source_unit": facts_row.get("shares_input_unit"),
                    "raw_shares_outstanding_mm": float(facts_shares),
                    "normalized_shares_outstanding_mm": None,
                    "shares_unit": SHARES_UNIT_MILLIONS,
                    "shares_basis": None,
                    "shares_split_adjustment_factor": None,
                    "shares_split_effective_date": None,
                }
            )
            if shares_guard is not None and shares_guard.get("reason_code") == GUARDED_HIT_REASON:
                # The guard refused a fresher candidate (or the chooser's pick was not a
                # count) and the value above is the candidate it accepted instead.
                coverage["shares_reason_code"] = GUARDED_HIT_REASON
                coverage["shares_reason_detail"] = (
                    "Resolved shares_outstanding from SEC companyfacts after the share-count "
                    f"guard: {shares_guard.get('detail')}"
                )
                logger.warning(
                    "shares: %s share-count guard %s -- %s",
                    ticker_norm,
                    shares_guard.get("outcome"),
                    shares_guard.get("detail"),
                )
            return resolved_shares, coverage

        if shares_guard is not None and shares_guard.get("outcome") == GUARD_REFUSED:
            # Every candidate was refused. UNKNOWN with the guard's reason -- never the
            # refused count, and never the market-cap/price derivation below, which would
            # paper over a slip the filings themselves cannot settle.
            coverage.update(
                {
                    "shares_status": "UNKNOWN",
                    "shares_reason_code": str(shares_guard.get("reason_code")),
                    "shares_reason_detail": (
                        f"{shares_guard.get('detail')} No substitute count was used."
                    ),
                    "shares_value": UNKNOWN,
                    "shares_source": facts_source,
                    "shares_source_url": facts_row.get("source_url"),
                    "shares_source_resolution": facts_source,
                    "derived_from": facts_refs,
                }
            )
            logger.warning(
                "shares: %s share-count guard refused every candidate -- %s",
                ticker_norm,
                shares_guard.get("detail"),
            )
            return None, coverage

        if str(facts_row.get("status") or "").upper() in {"OK", "PARTIAL"}:
            coverage.update(
                {
                    "shares_reason_code": "COMPANYFACTS_MISS",
                    "shares_reason_detail": (
                        "Companyfacts payload available but no shares_outstanding tag resolved."
                    ),
                    "shares_source": facts_source,
                    "shares_source_resolution": facts_source,
                    "derived_from": facts_refs,
                }
            )

        if bool(getattr(cfg, "shares_allow_market_cap_price_derive", False)):
            market_cap, market_cap_refs = _load_market_cap_snapshot(
                ticker=ticker_norm,
                as_of_date=asof_norm,
                cfg=cfg,
            )
            price, price_refs = _load_price_snapshot(
                ticker=ticker_norm,
                as_of_date=asof_norm,
                cfg=cfg,
            )
            if _is_num(market_cap) and _is_num(price) and float(price) > 0:
                derived_value = float(market_cap) / float(price)
                if derived_value > 0:
                    coverage.update(
                        {
                            "shares_status": "OK",
                            "shares_reason_code": "DERIVED_FROM_MKTCAP_PRICE",
                            "shares_reason_detail": "Derived shares_outstanding = market_cap / current_price.",
                            "shares_value": derived_value,
                            "shares_asof_used": asof_norm,
                            "shares_source": "derived_market_cap_price",
                            "shares_source_resolution": "derived_market_cap_price",
                            "derived_from": market_cap_refs
                            + price_refs
                            + ["derived:shares=market_cap/current_price"],
                        }
                    )
                    return derived_value, coverage

        return None, coverage
    except Exception as exc:  # noqa: BLE001
        coverage.update(
            {
                "shares_status": "UNKNOWN",
                "shares_reason_code": "EXCEPTION",
                "shares_reason_detail": str(exc),
                "shares_value": UNKNOWN,
            }
        )
        return None, coverage


def resolve_market_cap_from_price_asof(
    *,
    ticker: str,
    as_of_date: str,
    price: float | None,
    run_id: str | None = None,
    quote_lineage: dict[str, Any] | None = None,
    require_split_lineage: bool = False,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    companyfacts_cache_only: bool = False,
) -> tuple[float | None, dict[str, Any]]:
    ticker_norm = str(ticker or "").strip().upper()
    asof_norm = str(as_of_date or "").strip()
    coverage: dict[str, Any] = {
        "ticker": ticker_norm,
        "requested_as_of": asof_norm,
        "run_id": run_id,
        "market_cap_status": "UNKNOWN",
        "market_cap_reason_code": "INVALID_PRICE",
        "market_cap_reason_detail": "Current price is missing or invalid.",
        "market_cap_value": UNKNOWN,
        "price": price,
        "shares_outstanding": UNKNOWN,
        "shares_asof_used": None,
        "shares_filed_date": None,
        "shares_source": None,
        "shares_source_url": None,
        "shares_source_resolution": "unknown",
        "derived_from": [],
        "quote_snapshot_id": None,
        "price_basis": None,
        "raw_price": None,
        "split_adjustment_factor": None,
        "split_effective_date": None,
    }
    if not (_is_num(price) and float(price) > 0):
        return None, coverage

    shares_kwargs: dict[str, Any] = {
        "ticker": ticker_norm,
        "as_of_date": asof_norm,
        "run_id": run_id,
    }
    if db_path is not None:
        shares_kwargs["db_path"] = db_path
    if cfg is not None:
        shares_kwargs["cfg"] = cfg
    if companyfacts_cache_only:
        shares_kwargs["companyfacts_cache_only"] = True
    shares_value, shares_coverage = resolve_shares_asof(**shares_kwargs)
    shares_source_resolution = str(
        shares_coverage.get("shares_source_resolution")
        or shares_coverage.get("shares_source")
        or "unknown"
    ).strip()
    derived_from = [
        str(ref) for ref in (shares_coverage.get("derived_from") or []) if str(ref).strip()
    ]
    coverage.update(
        {
            "price": float(price),
            "shares_outstanding": shares_value if _is_num(shares_value) else UNKNOWN,
            "shares_asof_used": shares_coverage.get("shares_asof_used"),
            "shares_filed_date": shares_coverage.get("shares_filed_date"),
            "shares_source": shares_coverage.get("shares_source"),
            "shares_source_url": shares_coverage.get("shares_source_url"),
            "shares_source_resolution": shares_source_resolution,
            "shares_unit": shares_coverage.get("shares_unit"),
            "raw_shares_source_value": shares_coverage.get("raw_shares_source_value"),
            "raw_shares_source_unit": shares_coverage.get("raw_shares_source_unit"),
            "raw_shares_outstanding_mm": shares_coverage.get("raw_shares_outstanding_mm"),
            "normalized_shares_outstanding_mm": shares_coverage.get(
                "normalized_shares_outstanding_mm"
            ),
            "shares_basis": shares_coverage.get("shares_basis"),
            "shares_reason_code": shares_coverage.get("shares_reason_code"),
            "shares_guard": shares_coverage.get("shares_guard"),
            "derived_from": derived_from,
        }
    )
    if (
        shares_source_resolution == "derived_market_cap_price"
        or str(shares_coverage.get("shares_reason_code") or "").strip()
        == "DERIVED_FROM_MKTCAP_PRICE"
    ):
        coverage.update(
            {
                "market_cap_reason_code": "CIRCULAR_SHARES_SOURCE",
                "market_cap_reason_detail": "Rejected shares derived from market cap / price for market-cap computation.",
            }
        )
        return None, coverage

    if not (_is_num(shares_value) and float(shares_value) > 0):
        coverage.update(
            {
                "market_cap_reason_code": "SHARES_UNKNOWN",
                "market_cap_reason_detail": "Shares_outstanding could not be resolved from a non-circular as-of source.",
            }
        )
        return None, coverage

    shares_period_end = str(shares_coverage.get("shares_asof_used") or "").strip()[:10]
    shares_filed_date = str(shares_coverage.get("shares_filed_date") or "").strip()[:10]
    shares_source = str(shares_coverage.get("shares_source") or "").strip()
    shares_source_reference = str(shares_coverage.get("shares_source_url") or "").strip()
    shares_issuer_cik = _companyfacts_reference_cik(shares_source_reference)
    shares_unit = str(shares_coverage.get("shares_unit") or "").strip()
    raw_source_value = shares_coverage.get("raw_shares_source_value")
    raw_source_unit = str(shares_coverage.get("raw_shares_source_unit") or "").strip()
    raw_shares_mm = shares_coverage.get("raw_shares_outstanding_mm")
    if (
        shares_unit != SHARES_UNIT_MILLIONS
        or not _is_num(raw_source_value)
        or float(raw_source_value) <= 0
        or raw_source_unit not in {"shares", SHARES_UNIT_MILLIONS}
        or not _is_num(raw_shares_mm)
        or float(raw_shares_mm) <= 0
        or not shares_period_end
        or not shares_filed_date
        or not shares_source
        or not shares_source_reference
    ):
        coverage.update(
            {
                "market_cap_reason_code": "SHARES_FILED_PROVENANCE_MISSING",
                "market_cap_reason_detail": (
                    "Shares require an explicit normalized unit, period end, "
                    "filed date, source, and source reference."
                ),
            }
        )
        return None, coverage
    normalized_raw_source = (
        float(raw_source_value) / 1_000_000.0
        if raw_source_unit == "shares"
        else float(raw_source_value)
    )
    if not math.isclose(
        normalized_raw_source,
        float(raw_shares_mm),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        coverage.update(
            {
                "market_cap_reason_code": "SHARES_SOURCE_NORMALIZATION_CONFLICT",
                "market_cap_reason_detail": (
                    "Raw source shares do not reconcile to normalized shares_millions."
                ),
            }
        )
        return None, coverage
    shares_period_dt = _parse_date(shares_period_end)
    shares_filed_dt = _parse_date(shares_filed_date)
    requested_dt = _parse_date(asof_norm)
    if (
        shares_period_dt is None
        or shares_filed_dt is None
        or requested_dt is None
        or shares_period_dt > shares_filed_dt
        or shares_filed_dt > requested_dt
    ):
        coverage.update(
            {
                "market_cap_reason_code": "SHARES_FILED_PROVENANCE_CONFLICT",
                "market_cap_reason_detail": (
                    "Shares period/filed dates must be valid and filed no later "
                    "than the requested as-of date."
                ),
            }
        )
        return None, coverage

    normalized_shares = float(shares_value)
    if require_split_lineage:
        quote = dict(quote_lineage or {})
        basis = str(quote.get("price_basis") or "").strip().upper()
        raw_price = quote.get("raw_price")
        factor = quote.get("split_adjustment_factor")
        effective_date = str(quote.get("split_effective_date") or "").strip() or None
        quote_asof = str(quote.get("as_of_date") or asof_norm).strip()[:10]
        source = str(quote.get("source") or "").strip()
        currency = str(quote.get("currency") or "").strip().upper()
        unit = str(quote.get("unit") or PRICE_UNIT_USD_PER_SHARE).strip()
        raw_shares = shares_coverage.get("raw_shares_outstanding_mm")
        shares_asof = str(shares_coverage.get("shares_asof_used") or "").strip()[:10]
        if not (
            basis in {PRICE_BASIS_UNADJUSTED, PRICE_BASIS_SPLIT_ADJUSTED}
            and _is_num(raw_price)
            and float(raw_price) > 0
            and _is_num(factor)
            and float(factor) > 0
            and source
            and currency == "USD"
            and unit == PRICE_UNIT_USD_PER_SHARE
            and _is_num(raw_shares)
            and float(raw_shares) > 0
            and _parse_date(shares_asof) is not None
            and _parse_date(quote_asof) is not None
        ):
            coverage.update(
                {
                    "market_cap_reason_code": "SPLIT_LINEAGE_MISSING",
                    "market_cap_reason_detail": (
                        "Price/share split lineage is incomplete or ambiguous."
                    ),
                }
            )
            return None, coverage
        raw_price_number = float(raw_price)
        factor_number = float(factor)
        lineage_proof: dict[str, Any] | None = None
        if basis == PRICE_BASIS_UNADJUSTED:
            arithmetic_valid = (
                abs(raw_price_number - float(price)) <= 1e-9 and abs(factor_number - 1.0) <= 1e-12
            )
            no_split_proof = quote.get("no_intervening_split_proof")
            if isinstance(no_split_proof, dict):
                proof_start = str(no_split_proof.get("period_start") or "").strip()[:10]
                proof_end = str(no_split_proof.get("period_end") or "").strip()[:10]
                verified_as_of = str(no_split_proof.get("verified_as_of") or "").strip()[:10]
                proof_valid = bool(
                    str(no_split_proof.get("status") or "").strip().upper() == "PASS"
                    and str(no_split_proof.get("source") or "").strip()
                    and authoritative_split_proof_reference(
                        no_split_proof,
                        expected_ticker=ticker_norm,
                        expected_issuer_cik=shares_issuer_cik,
                        expected_as_of_date=asof_norm,
                    )
                    and _parse_date(proof_start) is not None
                    and _parse_date(proof_end) is not None
                    and _parse_date(verified_as_of) is not None
                    and proof_start <= shares_asof
                    and proof_end >= quote_asof
                    and proof_end <= verified_as_of
                    and verified_as_of <= asof_norm
                )
                lineage_proof = dict(no_split_proof)
            else:
                proof_valid = False
            valid = arithmetic_valid and proof_valid
            normalized_shares = float(raw_shares)
            normalized_shares_basis = SHARES_BASIS_UNADJUSTED
        else:
            split_event = quote.get("split_event")
            if isinstance(split_event, dict):
                event_factor = split_event.get("factor")
                event_effective_date = str(split_event.get("effective_date") or "").strip()[:10]
                event_filed_date = str(split_event.get("filed_date") or "").strip()[:10]
                proof_valid = bool(
                    _is_num(event_factor)
                    and abs(float(event_factor) - factor_number) <= 1e-12
                    and event_effective_date == effective_date
                    and str(split_event.get("source") or "").strip()
                    and authoritative_split_proof_reference(
                        split_event,
                        expected_ticker=ticker_norm,
                        expected_issuer_cik=shares_issuer_cik,
                        expected_as_of_date=asof_norm,
                    )
                    and _parse_date(event_effective_date) is not None
                    and _parse_date(event_filed_date) is not None
                    and event_filed_date <= quote_asof
                )
                lineage_proof = dict(split_event)
            else:
                proof_valid = False
            arithmetic_valid = (
                effective_date is not None
                and shares_asof
                and shares_asof < effective_date <= quote_asof
                and abs(raw_price_number / float(price) - factor_number) <= 1e-9
            )
            valid = arithmetic_valid and proof_valid
            normalized_shares = float(raw_shares) * factor_number
            normalized_shares_basis = PRICE_BASIS_SPLIT_ADJUSTED
        if not valid:
            coverage.update(
                {
                    "market_cap_reason_code": "SPLIT_LINEAGE_CONFLICT",
                    "market_cap_reason_detail": ("Price/share split lineage does not reconcile."),
                }
            )
            return None, coverage
        snapshot_id = stable_quote_hash(
            ticker=ticker_norm,
            price=float(price),
            as_of_date=quote_asof,
            currency=currency,
            source=source,
            source_url=quote.get("source_url"),
            price_basis=basis,
            raw_price=raw_price_number,
            split_adjustment_factor=factor_number,
            split_effective_date=effective_date,
        )
        declared_snapshot_id = str(quote.get("quote_snapshot_id") or "").strip()
        if declared_snapshot_id and declared_snapshot_id != snapshot_id:
            coverage.update(
                {
                    "market_cap_reason_code": "QUOTE_SNAPSHOT_ID_MISMATCH",
                    "market_cap_reason_detail": (
                        "Declared quote identity does not match canonical fields."
                    ),
                }
            )
            return None, coverage
        coverage.update(
            {
                "quote_snapshot_id": snapshot_id,
                "price_basis": basis,
                "raw_price": raw_price_number,
                "split_adjustment_factor": factor_number,
                "split_effective_date": effective_date,
                "shares_outstanding": normalized_shares,
                "raw_shares_outstanding_mm": float(raw_shares),
                "normalized_shares_outstanding_mm": normalized_shares,
                "shares_basis": normalized_shares_basis,
                "shares_unit": SHARES_UNIT_MILLIONS,
                "split_lineage_proof": lineage_proof,
            }
        )

    market_cap_value = float(price) * normalized_shares
    coverage.update(
        {
            "market_cap_status": "OK",
            "market_cap_reason_code": "OK",
            "market_cap_reason_detail": "Resolved market cap from current price × strict as-of shares.",
            "market_cap_value": market_cap_value,
            "derived_from": derived_from + ["derived:market_cap=current_price*shares_outstanding"],
        }
    )
    return market_cap_value, coverage


def write_shares_coverage_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str] | None = None,
    output_dir: Path | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    run_dir = output_dir or (cfg.sectors_dir / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        candidates = sorted(
            {
                str(path.stem).replace("fundamentals_", "").upper()
                for path in run_dir.glob("fundamentals_*.json")
                if str(path.stem).replace("fundamentals_", "").strip()
            }
        )
        if not candidates:
            dossier_root = cfg.dossiers_dir / run_id
            candidates = (
                sorted([path.name.upper() for path in dossier_root.iterdir() if path.is_dir()])
                if dossier_root.exists()
                else []
            )
    else:
        candidates = sorted(
            {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        )

    entries: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    ok_count = 0
    for ticker in candidates:
        value, row = resolve_shares_asof(
            ticker=ticker,
            as_of_date=as_of_date,
            run_id=run_id,
            cfg=cfg,
        )
        if _is_num(value) and float(value) > 0:
            ok_count += 1
        code = str(row.get("shares_reason_code") or "UNKNOWN")
        reason_counts[code] = reason_counts.get(code, 0) + 1
        entries.append(row)

    entries = sorted(entries, key=lambda row: str(row.get("ticker") or ""))
    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(entries),
        "ok_count": int(ok_count),
        "unknown_count": int(len(entries) - ok_count),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "entries": entries,
        "generated_at": utc_now_iso(),
    }
    out_path = run_dir / "shares_coverage.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["shares_coverage_path"] = str(out_path)
    return payload
