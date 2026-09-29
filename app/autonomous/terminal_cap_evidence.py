"""Validated terminal market-cap evidence ledgers for v2 sector scans.

The cap resolver deliberately does not invent a web client.  Search/provider
workers write a dated CSV or JSON evidence ledger, and this module turns that
artifact into the resolver's narrow lookup contract.  Every accepted row is a
direct issuer market cap with a source URL, as-of date, confidence, and issuer
binding when available.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from app.autonomous.cap_resolver import (
    CAP_SOURCE_KINDS,
    CAP_SOURCE_KIND_LOCAL,
    LOCAL_AUTHORITY_RECORD_PROVENANCE,
    LOCAL_AUTHORITY_SCHEMA_V1,
    SecurityIdentity,
    TerminalCapEvidence,
    TerminalCapLookup,
    derive_terminal_source_kind,
    direct_market_cap_mm_from_payload,
)


_DIRECT_ISSUER_CAP_BASIS = "DIRECT_ISSUER_MARKET_CAP"
_DIRECT_CAP_UNITS = {
    "USD_MILLIONS",
    "USD MILLIONS",
    "MUSD",
    "USD",
    "DOLLARS",
    "USD_DOLLARS",
    "USD DOLLARS",
}
_POSTMORTEM_DIRECT_CAP_STATUSES = {
    "CURRENT_BELOW_10B_COMMON_EQUITY",
    "CURRENT_LARGE_CAP_COMMON_EQUITY",
    "PRE_SCOPE_EXCLUSION_WITH_CAP_RESOLVED",
    "DUPLICATE_NONPRIMARY_BELOW_10B",
    "DUPLICATE_NONPRIMARY_LARGE_CAP",
}
_POSTMORTEM_PROOF_FIELDS = {
    "record_provenance",
    "resolution_status",
    "resolved_ticker",
    "resolved_market_cap_usd",
    "security_type",
    "listing_status",
    "is_current_common_equity",
    "resolution_confidence",
    "evidence_source",
    "evidence_url",
    "evidence_as_of",
}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _normalize_cik(value: Any) -> str | None:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    return digits.zfill(10) if digits else None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _source_kind(row: dict[str, Any], source_url: str, source_name: str) -> str:
    explicit = (
        str(row.get("source_kind") or row.get("market_cap_source_kind") or "").strip().upper()
    )
    return derive_terminal_source_kind(
        source_url=source_url,
        claimed_kind=explicit if explicit in CAP_SOURCE_KINDS else None,
        allow_local_authoritative=True,
        local_authority_schema=str(row.get("local_authority_schema") or "").strip(),
        record_provenance=str(row.get("record_provenance") or "").strip(),
    )


def _rows_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("evidence", "rows", "resolutions", "items"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return [dict(payload)]


def _proves_direct_issuer_cap(row: dict[str, Any]) -> bool:
    """Accept only an explicit direct-cap contract or the audited postmortem shape."""

    basis = str(row.get("market_cap_basis") or "").strip().upper()
    units = str(row.get("market_cap_units") or "").strip().upper()
    if basis == _DIRECT_ISSUER_CAP_BASIS and units in _DIRECT_CAP_UNITS:
        if direct_market_cap_mm_from_payload(row) is None:
            return False
        claimed_kind = (
            str(row.get("source_kind") or row.get("market_cap_source_kind") or "").strip().upper()
        )
        if claimed_kind == CAP_SOURCE_KIND_LOCAL:
            return bool(
                str(row.get("local_authority_schema") or "").strip().upper()
                == LOCAL_AUTHORITY_SCHEMA_V1
                and str(row.get("record_provenance") or "").strip().upper()
                == LOCAL_AUTHORITY_RECORD_PROVENANCE
            )
        return True

    # The fixed-as-of repair ledger predates the generic basis fields. Its
    # full schema plus a known cap-resolved terminal status is the equivalent
    # proof contract; a loose CSV containing only ticker/cap is not.
    if not _POSTMORTEM_PROOF_FIELDS <= set(row):
        return False
    status = str(row.get("resolution_status") or "").strip().upper()
    return bool(
        status in _POSTMORTEM_DIRECT_CAP_STATUSES
        and _truthy(row.get("is_current_common_equity"))
        and str(row.get("ticker") or row.get("resolved_ticker") or "").strip()
        and str(row.get("evidence_source") or "").strip()
        and str(row.get("evidence_url") or "").strip()
        and str(row.get("evidence_as_of") or "").strip()
        and str(row.get("resolution_confidence") or "").strip()
        and _positive_float(row.get("resolved_market_cap_usd")) is not None
    )


@lru_cache(maxsize=16)
def _load_rows_cached(
    path_text: str,
    modified_ns: int,
    size_bytes: int,
) -> tuple[dict[str, Any], ...]:
    del modified_ns, size_bytes
    path = Path(path_text)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(_rows_from_json(payload))


def _load_rows(path: Path) -> tuple[dict[str, Any], ...]:
    stat = path.stat()
    return _load_rows_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _row_to_evidence(row: dict[str, Any]) -> TerminalCapEvidence | None:
    if not _proves_direct_issuer_cap(row):
        return None
    ticker = str(row.get("ticker") or row.get("resolved_ticker") or "").strip().upper()
    source_url = str(row.get("source_url") or row.get("evidence_url") or "").strip()
    source_name = str(
        row.get("source_name")
        or row.get("evidence_source")
        or row.get("provider")
        or "terminal_cap_evidence_ledger"
    ).strip()
    as_of_date = str(
        row.get("as_of_date") or row.get("evidence_as_of") or row.get("effective_as_of_date") or ""
    ).strip()[:10]
    confidence = (
        str(row.get("confidence") or row.get("resolution_confidence") or "").strip().upper()
    )
    postmortem_shape = _POSTMORTEM_PROOF_FIELDS <= set(row)
    cap_mm = direct_market_cap_mm_from_payload(
        row,
        allow_implicit_usd_field=postmortem_shape,
    )
    issuer_cik = _normalize_cik(row.get("issuer_cik") or row.get("cik"))
    if (
        not ticker
        or cap_mm is None
        or not source_url
        or not as_of_date
        or not confidence
        or issuer_cik is None
    ):
        return None
    # Rows explicitly identifying a non-current/non-equity instrument can be
    # reconciled as OUT_OF_SCOPE elsewhere, but they must not supply a direct
    # issuer common-equity cap to the band classifier.
    if "is_current_common_equity" in row and not _truthy(row["is_current_common_equity"]):
        return None
    return TerminalCapEvidence(
        ticker=ticker,
        market_cap_mm=cap_mm,
        source_kind=_source_kind(row, source_url, source_name),  # type: ignore[arg-type]
        source_name=source_name,
        source_url=source_url,
        as_of_date=as_of_date,
        confidence=confidence,  # type: ignore[arg-type]
        issuer_cik=issuer_cik,
        issuer_name=str(row.get("issuer_name") or row.get("resolved_name") or "").strip() or None,
        security_name=str(row.get("security_name") or row.get("resolved_name") or "").strip()
        or None,
        local_authority_schema=(
            str(row.get("local_authority_schema") or "").strip().upper() or None
        ),
        record_provenance=(str(row.get("record_provenance") or "").strip().upper() or None),
        detail=str(row.get("detail") or row.get("notes") or "").strip()
        or "direct issuer market cap from validated terminal evidence ledger",
    )


def load_terminal_cap_evidence(path: str | Path) -> dict[str, tuple[TerminalCapEvidence, ...]]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Terminal cap evidence ledger not found: {resolved}")
    grouped: dict[str, list[TerminalCapEvidence]] = {}
    for row in _load_rows(resolved):
        evidence = _row_to_evidence(row)
        if evidence is not None:
            grouped.setdefault(evidence.ticker, []).append(evidence)
    return {ticker: tuple(rows) for ticker, rows in sorted(grouped.items())}


def terminal_cap_lookup_from_path(path: str | Path) -> TerminalCapLookup:
    """Build a deterministic resolver callback from a validated ledger."""

    ledger = load_terminal_cap_evidence(path)

    def lookup(
        ticker: str,
        as_of_date: str,
        identity: SecurityIdentity,
    ) -> Iterable[TerminalCapEvidence]:
        del as_of_date
        requested = str(ticker or "").strip().upper()
        identity_cik = _normalize_cik(identity.issuer_cik)
        if identity_cik is None:
            return ()
        return tuple(
            evidence
            for evidence in ledger.get(requested, ())
            if _normalize_cik(evidence.issuer_cik) == identity_cik
        )

    # Allows the cap resolver to consume this already-loaded, validated local
    # artifact before invoking derived shares x price tiers. Arbitrary live
    # callbacks remain late fallbacks and do not receive this marker.
    lookup._voe_prevalidated_local_ledger = True  # type: ignore[attr-defined]
    return lookup


__all__ = [
    "load_terminal_cap_evidence",
    "terminal_cap_lookup_from_path",
]
