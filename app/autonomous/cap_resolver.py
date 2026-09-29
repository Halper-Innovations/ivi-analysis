"""Band-filter market-cap classification chain (band integrity only).

This module decides which cap BAND a ticker belongs to for sweep candidate
loading and queue/brief/digest surfaces. It is deliberately NOT a valuation
input and is NOT used by the backtest strata: tier 1 delegates to the same
strict as-of resolution the valuation/backtest paths already use, and the
fallback tiers only loosen *band membership* so an unknown-cap name cannot
leak into a band it provably does not belong to (e.g. a ~$9B name reaching a
micro review queue because its strict as-of cap was uncomputable).

Chain (first tier that yields a computable cap wins; ``cap_source`` records
which tier resolved it on EVERY classification):

  0. ``terminal_local_authoritative`` — an explicitly direct issuer-cap row
     in the existing local market-cap store (legacy derived rows do not qualify).
  1. ``asof_companyfacts``  — strict as-of shares x as-of price
     (``resolve_market_cap_from_price_asof``, the pre-existing behavior).
     Known ADR/class quotes require an authoritative conversion ratio.
  2. ``terminal_sec`` / ``terminal_exchange`` — dated direct issuer-cap evidence.
  3. ``stale_shares``       — last-known companyfacts shares_outstanding
     x a provenance-carrying as-of price, subject to the same identity guard.
  4. ``eodhd_fundamentals`` — INERT slot. The EODHD fundamentals endpoint is
     NOT available on the current subscription (verified 403, 2026-06-11).
     The tier exists so an upgraded plan only needs the config flag
     ``VOE_CAP_EODHD_FUNDAMENTALS_ENABLED=true``; it is never called by
     default.
  5. ``terminal_provider`` then ``terminal_search`` — direct issuer-cap
     fallback evidence. Search requires URL, as-of date, and confidence.
  6. ``unknown``            — no computable cap. Callers must render the
     ``UNKNOWN_CAP`` label and must never present the row as in-band output.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Literal
from urllib.parse import urlparse

from app.db import connect as db_connect
from app.config import AppConfig, get_config
from app.autonomous.sector_candidates import MARKET_CAP_FOCUS_TIERS
from app.autonomous.financial_integrity import (
    MARKET_CAP_UNIT_USD_MILLIONS,
    PRICE_BASIS_SPLIT_ADJUSTED,
    PRICE_BASIS_UNADJUSTED,
    SHARES_BASIS_UNADJUSTED,
    SHARES_UNIT_MILLIONS,
    authoritative_split_proof_reference,
    stable_quote_hash,
)
from app.util.credential_hygiene import sanitize_url_credentials

logger = logging.getLogger(__name__)

CAP_SOURCE_ASOF_COMPANYFACTS = "asof_companyfacts"
CAP_SOURCE_EODHD_FUNDAMENTALS = "eodhd_fundamentals"
CAP_SOURCE_STALE_SHARES = "stale_shares"
CAP_SOURCE_TERMINAL_LOCAL = "terminal_local_authoritative"
CAP_SOURCE_TERMINAL_SEC = "terminal_sec"
CAP_SOURCE_TERMINAL_EXCHANGE = "terminal_exchange"
CAP_SOURCE_TERMINAL_PROVIDER = "terminal_provider"
CAP_SOURCE_TERMINAL_SEARCH = "terminal_search"
CAP_SOURCE_UNKNOWN = "unknown"

CAP_SOURCE_KIND_LOCAL = "LOCAL_AUTHORITATIVE"
CAP_SOURCE_KIND_SEC = "SEC"
CAP_SOURCE_KIND_EXCHANGE = "EXCHANGE"
CAP_SOURCE_KIND_PROVIDER = "PROVIDER"
CAP_SOURCE_KIND_SEARCH = "SEARCH"
CAP_SOURCE_KINDS = {
    CAP_SOURCE_KIND_LOCAL,
    CAP_SOURCE_KIND_SEC,
    CAP_SOURCE_KIND_EXCHANGE,
    CAP_SOURCE_KIND_PROVIDER,
    CAP_SOURCE_KIND_SEARCH,
}
CAP_CONFIDENCE_LEVELS = {"HIGH", "MEDIUM", "LOW"}
CAP_SOURCE_BY_KIND = {
    CAP_SOURCE_KIND_LOCAL: CAP_SOURCE_TERMINAL_LOCAL,
    CAP_SOURCE_KIND_SEC: CAP_SOURCE_TERMINAL_SEC,
    CAP_SOURCE_KIND_EXCHANGE: CAP_SOURCE_TERMINAL_EXCHANGE,
    CAP_SOURCE_KIND_PROVIDER: CAP_SOURCE_TERMINAL_PROVIDER,
    CAP_SOURCE_KIND_SEARCH: CAP_SOURCE_TERMINAL_SEARCH,
}
CAP_SOURCE_KIND_PRIORITY = {
    CAP_SOURCE_KIND_LOCAL: 0,
    CAP_SOURCE_KIND_SEC: 1,
    CAP_SOURCE_KIND_EXCHANGE: 2,
    CAP_SOURCE_KIND_PROVIDER: 3,
    CAP_SOURCE_KIND_SEARCH: 4,
}

SECURITY_ROLE_PRIMARY = "PRIMARY"
SECURITY_ROLE_SECONDARY_CLASS = "SECONDARY_CLASS"
SECURITY_ROLE_SECONDARY_SECURITY = "SECONDARY_SECURITY"
SECURITY_ROLE_ADR = "ADR"
SECURITY_ROLE_UNKNOWN = "UNKNOWN"

UNSAFE_ADR_RATIO_MISSING = "UNSAFE_ISSUER_SHARES_ADR_RATIO_MISSING"
UNSAFE_SECONDARY_CLASS_RATIO_MISSING = "UNSAFE_ISSUER_SHARES_SECONDARY_CLASS_RATIO_MISSING"
UNSAFE_SECURITY_ROLE_UNRESOLVED = "UNSAFE_ISSUER_SHARES_SECURITY_ROLE_UNRESOLVED"
UNSAFE_PRIMARY_IDENTITY_UNVERIFIED = "UNSAFE_ISSUER_SHARES_PRIMARY_IDENTITY_UNVERIFIED"

_SEC_HOST_SUFFIXES = ("sec.gov",)
_EXCHANGE_HOST_SUFFIXES = (
    "nasdaq.com",
    "nyse.com",
    "cboe.com",
    "otcmarkets.com",
    "tsx.com",
    "londonstockexchange.com",
)
_PROVIDER_HOST_SUFFIXES = (
    "bloomberg.com",
    "eodhd.com",
    "lseg.com",
    "morningstar.com",
    "refinitiv.com",
    "spglobal.com",
)
_DEPOSITARY_HOST_SUFFIXES = (
    "adrbny.com",
    "bnymellon.com",
    "citi.com",
    "db.com",
    "jpmorgan.com",
)

LOCAL_AUTHORITY_SCHEMA_V1 = "VOE_TERMINAL_CAP_EVIDENCE_V1"
LOCAL_AUTHORITY_RECORD_PROVENANCE = "PERSISTED_RUN_ARTIFACT"

_CAP_UNITS_MILLIONS = {"USD_MILLIONS", "USD MILLIONS", "MUSD"}
_CAP_UNITS_DOLLARS = {"USD", "DOLLARS", "USD_DOLLARS", "USD DOLLARS"}
_SEC_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")

TERMINAL_CAP_MAX_AGE_DAYS = 14

# Stale shares older than this cannot anchor a band CLAIM. One annual filing
# cycle + the 90-day filing lag + buffer: a cover-page share count beyond this
# window is archaeology, not evidence of today's band (e.g. a pre-IPO share
# count pricing a recent listing two bands too low).
STALE_SHARES_MAX_AGE_DAYS = 400

# Same residual-unknown token the backtest universe uses (backtest/universe.py),
# so every surface renders the one literal.
UNKNOWN_CAP_LABEL = "UNKNOWN_CAP"

PriceLookup = Callable[[str, str], Any]


@dataclass(frozen=True)
class SecurityIdentity:
    """Issuer/security relationship used to keep cap arithmetic honest.

    ``adr_ratio`` is the number of issuer ordinary shares represented by one
    quoted ADR/ADS. ``share_class_ratio`` is the equivalent issuer-common
    shares represented by one quoted secondary-class share. A known ADR or
    secondary class may use issuer-wide shares only when the applicable ratio
    and an authoritative evidence URL are both present.
    """

    ticker: str
    issuer_cik: str | None = None
    issuer_primary_ticker: str | None = None
    issuer_listed_tickers: tuple[str, ...] = ()
    security_role: str = SECURITY_ROLE_UNKNOWN
    is_secondary_class: bool | None = None
    is_adr: bool | None = None
    adr_ratio: float | None = None
    share_class_ratio: float | None = None
    ratio_source_url: str | None = None
    ratio_source_accession: str | None = None
    ratio_security_symbol: str | None = None
    identity_source: str | None = None
    identity_source_url: str | None = None
    identity_as_of_date: str | None = None
    identity_confidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issuer_listed_tickers"] = list(self.issuer_listed_tickers)
        return payload


@dataclass(frozen=True)
class TerminalCapEvidence:
    """Direct issuer market-cap evidence for the terminal fallback chain."""

    ticker: str
    market_cap_mm: float
    source_kind: Literal["LOCAL_AUTHORITATIVE", "SEC", "EXCHANGE", "PROVIDER", "SEARCH"]
    source_name: str
    source_url: str
    as_of_date: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    issuer_cik: str | None = None
    issuer_name: str | None = None
    security_name: str | None = None
    price_used: float | None = None
    price_source: str | None = None
    price_source_url: str | None = None
    price_as_of_date: str | None = None
    price_currency: str | None = None
    price_confidence: str | None = None
    price_basis: str | None = None
    raw_price: float | None = None
    split_adjustment_factor: float | None = None
    split_effective_date: str | None = None
    local_authority_schema: str | None = None
    record_provenance: str | None = None
    detail: str | None = None


TerminalCapLookup = Callable[
    [str, str, SecurityIdentity],
    TerminalCapEvidence | dict[str, Any] | Iterable[TerminalCapEvidence | dict[str, Any]] | None,
]


def band_for_market_cap(mc_millions: float | None) -> str | None:
    """Map a market cap in millions to the canonical band token.

    Thresholds come from the ONE canonical definition
    (``MARKET_CAP_FOCUS_TIERS``); lower-inclusive / upper-exclusive. Tokens
    mirror ``app.backtest.universe.cap_category_from_market_cap`` and route
    through ``select_benchmark_symbol`` (large_cap/mega_cap -> SPY, rest ->
    IWM). Returns None when the cap is missing or non-positive.
    """
    if mc_millions is None or not isinstance(mc_millions, (int, float)) or mc_millions <= 0:
        return None
    mc = float(mc_millions)
    if mc < float(MARKET_CAP_FOCUS_TIERS["micro"][1]):
        return "micro"
    if mc < float(MARKET_CAP_FOCUS_TIERS["small"][1]):
        return "small"
    if mc < float(MARKET_CAP_FOCUS_TIERS["mid"][1]):
        return "mid"
    if mc < float(MARKET_CAP_FOCUS_TIERS["large_cap"][1]):
        return "large_cap"
    return "mega_cap"


@dataclass(frozen=True)
class CapClassification:
    """One band-filter classification with full provenance."""

    ticker: str
    as_of_date: str
    market_cap_mm: float | None
    cap_source: str
    cap_band: str | None
    market_cap_unit: str | None = None
    price_used: float | None = None
    price_as_of_date: str | None = None
    price_source: str | None = None
    price_source_url: str | None = None
    price_currency: str | None = None
    price_confidence: str | None = None
    quote_snapshot_id: str | None = None
    price_basis: str | None = None
    raw_price: float | None = None
    shares_mm: float | None = None
    raw_shares_outstanding_mm: float | None = None
    raw_shares_source_value: float | None = None
    raw_shares_source_unit: str | None = None
    shares_unit: str | None = None
    shares_basis: str | None = None
    shares_source: str | None = None
    shares_source_url: str | None = None
    issuer_quote_ratio: float | None = None
    split_adjustment_factor: float | None = None
    split_effective_date: str | None = None
    split_lineage_proof: dict[str, Any] | None = None
    shares_period_end: str | None = None
    shares_filed_date: str | None = None
    cap_effective_as_of_date: str | None = None
    cap_source_kind: str | None = None
    cap_source_name: str | None = None
    cap_source_url: str | None = None
    cap_confidence: str | None = None
    scope_status: str = "IN_SCOPE"
    scope_reason: str | None = None
    issuer_cik: str | None = None
    issuer_primary_ticker: str | None = None
    issuer_listed_tickers: tuple[str, ...] = ()
    security_role: str = SECURITY_ROLE_UNKNOWN
    is_secondary_class: bool | None = None
    is_adr: bool | None = None
    adr_ratio: float | None = None
    share_class_ratio: float | None = None
    identity_source: str | None = None
    identity_source_url: str | None = None
    identity_as_of_date: str | None = None
    identity_confidence: str | None = None
    ratio_source_url: str | None = None
    ratio_source_accession: str | None = None
    ratio_security_symbol: str | None = None
    detail: str | None = None
    # Accepted-census bridge provenance.  These remain empty for the legacy
    # cap-resolution lanes and are populated only by a fully validated,
    # fixed-as-of registry-first census member.
    cap_retrieved_at: str | None = None
    cap_method: str | None = None
    cap_derivation_json: str | None = None
    issuer_key: str | None = None
    security_key: str | None = None
    census_run_id: str | None = None
    census_input_fingerprint: str | None = None
    census_semantic_output_fingerprint: str | None = None
    census_cohort_fingerprint: str | None = None

    @property
    def band_label(self) -> str:
        return self.cap_band or UNKNOWN_CAP_LABEL

    def in_band(self, cap_min: float | None, cap_max: float | None) -> bool | None:
        """True/False when the cap is computable; None when it is not.

        Bounds are lower-inclusive / upper-exclusive per the 2026-06-11 band
        redefinition. A None result means the row may never be presented as
        in-band output (it stays sweepable, labeled UNKNOWN_CAP).
        """
        if self.market_cap_mm is None:
            return None
        if cap_min is not None and self.market_cap_mm < float(cap_min):
            return False
        if cap_max is not None and self.market_cap_mm >= float(cap_max):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issuer_listed_tickers"] = list(self.issuer_listed_tickers)
        payload["band_label"] = self.band_label
        return payload


@dataclass(frozen=True)
class _ResolvedPrice:
    price: float
    as_of_date: str
    source: str
    source_url: str | None
    currency: str | None
    confidence: str | None
    raw_price: float | None = None
    price_basis: str | None = None
    split_adjustment_factor: float | None = None
    split_effective_date: str | None = None
    split_event: dict[str, Any] | None = None
    no_intervening_split_proof: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_url", sanitize_url_credentials(self.source_url))


def _price_basis_contract(
    *,
    price: float,
    raw_price: Any,
    explicit_basis: Any,
    explicit_factor: Any,
    explicit_effective_date: Any,
) -> tuple[float | None, str | None, float | None, str | None] | None:
    raw = _optional_positive_float(raw_price)
    basis = str(explicit_basis or "").strip().upper() or None
    factor = _optional_positive_float(explicit_factor)
    effective_date = _safe_iso_date(explicit_effective_date)
    if basis is None:
        # Legacy quote objects may still be useful for non-decision display,
        # but raw/adjusted magnitudes cannot establish a split basis. Leave
        # the lineage unresolved so the financial gate returns NEEDS_DATA.
        return raw, None, factor, effective_date
    if basis not in {PRICE_BASIS_UNADJUSTED, PRICE_BASIS_SPLIT_ADJUSTED}:
        return None
    if basis == PRICE_BASIS_UNADJUSTED:
        raw = float(price) if raw is None else raw
        factor = 1.0 if factor is None else factor
        if not math.isclose(raw, float(price), rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(
            factor, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            return None
        return raw, basis, factor, effective_date
    if raw is None or factor is None or effective_date is None:
        return None
    if not math.isclose(
        raw / float(price),
        factor,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return None
    return raw, basis, factor, effective_date


def _classification_quote_fields(
    ticker: str,
    price: _ResolvedPrice | None,
    *,
    shares_mm: float | None = None,
    raw_shares_mm: float | None = None,
    raw_shares_source_value: float | None = None,
    raw_shares_source_unit: str | None = None,
    shares_basis: str | None = None,
) -> dict[str, Any]:
    if price is None:
        return {
            "quote_snapshot_id": None,
            "price_basis": None,
            "raw_price": None,
            "shares_unit": SHARES_UNIT_MILLIONS if shares_mm is not None else None,
            "shares_basis": None,
            "raw_shares_outstanding_mm": (
                float(raw_shares_mm) if raw_shares_mm is not None else None
            ),
            "raw_shares_source_value": (
                float(raw_shares_source_value) if raw_shares_source_value is not None else None
            ),
            "raw_shares_source_unit": raw_shares_source_unit,
            "split_adjustment_factor": None,
            "split_effective_date": None,
            "split_lineage_proof": None,
        }
    snapshot_id = stable_quote_hash(
        ticker=ticker,
        price=price.price,
        as_of_date=price.as_of_date,
        currency=price.currency,
        source=price.source,
        source_url=price.source_url,
        price_basis=price.price_basis,
        raw_price=price.raw_price,
        split_adjustment_factor=price.split_adjustment_factor,
        split_effective_date=price.split_effective_date,
    )
    return {
        "quote_snapshot_id": snapshot_id,
        "price_basis": price.price_basis,
        "raw_price": price.raw_price,
        "shares_unit": SHARES_UNIT_MILLIONS if shares_mm is not None else None,
        "shares_basis": shares_basis,
        "raw_shares_outstanding_mm": (float(raw_shares_mm) if raw_shares_mm is not None else None),
        "raw_shares_source_value": (
            float(raw_shares_source_value) if raw_shares_source_value is not None else None
        ),
        "raw_shares_source_unit": raw_shares_source_unit,
        "split_adjustment_factor": price.split_adjustment_factor,
        "split_effective_date": price.split_effective_date,
        "split_lineage_proof": (
            dict(price.split_event)
            if isinstance(price.split_event, dict)
            else dict(price.no_intervening_split_proof)
            if isinstance(price.no_intervening_split_proof, dict)
            else None
        ),
    }


def _valid_no_intervening_split_proof(
    proof: dict[str, Any] | None,
    *,
    ticker: str,
    issuer_cik: str | None,
    shares_as_of_date: str,
    quote_as_of_date: str,
    run_as_of_date: str,
) -> bool:
    if not isinstance(proof, dict):
        return False
    period_start = _safe_iso_date(proof.get("period_start"))
    period_end = _safe_iso_date(proof.get("period_end"))
    verified_as_of = _safe_iso_date(proof.get("verified_as_of"))
    return bool(
        str(proof.get("status") or "").strip().upper() == "PASS"
        and str(proof.get("source") or "").strip()
        and authoritative_split_proof_reference(
            proof,
            expected_ticker=ticker,
            expected_issuer_cik=issuer_cik,
            expected_as_of_date=run_as_of_date,
        )
        and period_start is not None
        and period_end is not None
        and verified_as_of is not None
        and period_start <= shares_as_of_date
        and period_end >= quote_as_of_date
        and period_end <= verified_as_of
        and verified_as_of <= run_as_of_date
    )


def _valid_split_event_proof(
    proof: dict[str, Any] | None,
    *,
    ticker: str,
    issuer_cik: str | None,
    factor: float,
    effective_date: str,
    quote_as_of_date: str,
) -> bool:
    if not isinstance(proof, dict):
        return False
    proof_factor = _optional_positive_float(proof.get("factor"))
    proof_effective_date = _safe_iso_date(proof.get("effective_date"))
    filed_date = _safe_iso_date(proof.get("filed_date"))
    return bool(
        proof_factor is not None
        and math.isclose(proof_factor, factor, rel_tol=0.0, abs_tol=1e-12)
        and proof_effective_date == effective_date
        and filed_date is not None
        and filed_date <= quote_as_of_date
        and str(proof.get("source") or "").strip()
        and authoritative_split_proof_reference(
            proof,
            expected_ticker=ticker,
            expected_issuer_cik=issuer_cik,
            expected_as_of_date=quote_as_of_date,
        )
    )


def _normalized_shares_for_price(
    price: _ResolvedPrice | None,
    *,
    ticker: str,
    issuer_cik: str | None,
    raw_shares_mm: float | None,
    shares_as_of_date: str | None,
    run_as_of_date: str,
) -> tuple[float, str] | None:
    """Normalize a source share count onto the quote's declared split basis."""

    if price is None or raw_shares_mm is None or raw_shares_mm <= 0:
        return None
    if price.price_basis == PRICE_BASIS_UNADJUSTED:
        raw_price = _optional_positive_float(price.raw_price)
        factor = _optional_positive_float(price.split_adjustment_factor)
        if (
            raw_price is None
            or factor is None
            or not math.isclose(raw_price, price.price, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-12)
            or _safe_iso_date(shares_as_of_date) is None
            or not _valid_no_intervening_split_proof(
                price.no_intervening_split_proof,
                ticker=ticker,
                issuer_cik=issuer_cik,
                shares_as_of_date=str(shares_as_of_date)[:10],
                quote_as_of_date=price.as_of_date,
                run_as_of_date=run_as_of_date,
            )
        ):
            return None
        return float(raw_shares_mm), SHARES_BASIS_UNADJUSTED
    if price.price_basis != PRICE_BASIS_SPLIT_ADJUSTED:
        return None
    raw_price = _optional_positive_float(price.raw_price)
    factor = _optional_positive_float(price.split_adjustment_factor)
    effective_date = _safe_iso_date(price.split_effective_date)
    shares_date = _safe_iso_date(shares_as_of_date)
    if (
        raw_price is None
        or factor is None
        or effective_date is None
        or shares_date is None
        or shares_date >= effective_date
        or effective_date > price.as_of_date
        or not _valid_split_event_proof(
            price.split_event,
            ticker=ticker,
            issuer_cik=issuer_cik,
            factor=factor,
            effective_date=effective_date,
            quote_as_of_date=price.as_of_date,
        )
        or not math.isclose(
            raw_price / price.price,
            factor,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        return None
    return float(raw_shares_mm) * factor, PRICE_BASIS_SPLIT_ADJUSTED


def _normalize_cik(value: Any) -> str | None:
    token = "".join(ch for ch in str(value or "") if ch.isdigit())
    return token.zfill(10) if token else None


def _safe_iso_date(value: Any) -> str | None:
    token = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(token).isoformat()
    except ValueError:
        return None


def _optional_positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if numeric > 0 else None


def _host_matches(host: str, suffixes: tuple[str, ...]) -> bool:
    normalized = str(host or "").lower().split(":", 1)[0]
    return any(normalized == suffix or normalized.endswith(f".{suffix}") for suffix in suffixes)


def derive_terminal_source_kind(
    *,
    source_url: str,
    claimed_kind: str | None = None,
    allow_local_authoritative: bool = False,
    local_authority_schema: str | None = None,
    record_provenance: str | None = None,
) -> str:
    """Derive authority from source evidence; never trust a caller's label.

    ``LOCAL_AUTHORITATIVE`` is intentionally not a URL-host category. It is
    granted only to the controlled, versioned persisted-artifact contract.
    Without both markers an unrecognized URL is ordinary search evidence,
    even when a caller claims that it is local or authoritative.
    """

    host = urlparse(str(source_url or "")).netloc.lower().split(":", 1)[0]
    if _host_matches(host, _SEC_HOST_SUFFIXES):
        return CAP_SOURCE_KIND_SEC
    if _host_matches(host, _EXCHANGE_HOST_SUFFIXES):
        return CAP_SOURCE_KIND_EXCHANGE
    if _host_matches(host, _PROVIDER_HOST_SUFFIXES):
        return CAP_SOURCE_KIND_PROVIDER
    if (
        allow_local_authoritative
        and str(claimed_kind or "").strip().upper() == CAP_SOURCE_KIND_LOCAL
        and str(local_authority_schema or "").strip().upper() == LOCAL_AUTHORITY_SCHEMA_V1
        and str(record_provenance or "").strip().upper() == LOCAL_AUTHORITY_RECORD_PROVENANCE
    ):
        return CAP_SOURCE_KIND_LOCAL
    return CAP_SOURCE_KIND_SEARCH


def direct_market_cap_mm_from_payload(
    payload: dict[str, Any],
    *,
    allow_implicit_usd_field: bool = False,
) -> float | None:
    """Normalize one unambiguous direct-cap value to USD millions.

    Field names and unit labels must agree. In particular, ``market_cap_mm``
    can only carry million-dollar units; absolute dollars belong in
    ``market_cap_usd``/``resolved_market_cap_usd``. The audited postmortem
    ledger predates the generic units field, so its explicitly named USD
    field may opt into the one narrow implicit-dollars exception.
    """

    units = str(payload.get("market_cap_units") or "").strip().upper()
    mm_values = [value for value in (payload.get("market_cap_mm"),) if value not in (None, "")]
    usd_values = [
        value
        for value in (
            payload.get("resolved_market_cap_usd"),
            payload.get("market_cap_usd"),
        )
        if value not in (None, "")
    ]
    generic_values = [value for value in (payload.get("market_cap"),) if value not in (None, "")]
    populated_groups = sum(bool(values) for values in (mm_values, usd_values, generic_values))
    if populated_groups != 1:
        return None

    values = mm_values or usd_values or generic_values
    if len(values) != 1:
        return None
    if isinstance(values[0], bool):
        return None
    try:
        numeric = float(str(values[0]).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None

    if mm_values:
        return numeric if units in _CAP_UNITS_MILLIONS else None
    if usd_values:
        if units in _CAP_UNITS_DOLLARS or (allow_implicit_usd_field and not units):
            return numeric / 1_000_000.0
        return None
    if units in _CAP_UNITS_MILLIONS:
        return numeric
    if units in _CAP_UNITS_DOLLARS:
        return numeric / 1_000_000.0
    return None


def _price_date_at_or_before(value: Any, requested_as_of_date: str) -> str | None:
    resolved = _safe_iso_date(value)
    requested = _safe_iso_date(requested_as_of_date)
    if resolved is None or requested is None or resolved > requested:
        return None
    return resolved


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if token in {"true", "yes", "1"}:
        return True
    if token in {"false", "no", "0"}:
        return False
    return None


def _share_class_variant(primary: str, candidate: str) -> bool:
    primary_upper = str(primary or "").strip().upper()
    candidate_upper = str(candidate or "").strip().upper()
    if not primary_upper or not candidate_upper or primary_upper == candidate_upper:
        return False
    primary_root = re.split(r"[.\-]", primary_upper, maxsplit=1)[0]
    candidate_root = re.split(r"[.\-]", candidate_upper, maxsplit=1)[0]
    if primary_root == candidate_root:
        return True
    normalized_primary = re.sub(r"[^A-Z0-9]", "", primary_upper)
    normalized_candidate = re.sub(r"[^A-Z0-9]", "", candidate_upper)
    if (
        len(normalized_primary) == len(normalized_candidate)
        and len(normalized_primary) >= 3
        and normalized_primary[:-1] == normalized_candidate[:-1]
        and normalized_primary[-1].isalpha()
        and normalized_candidate[-1].isalpha()
    ):
        return True
    return normalized_primary.startswith(normalized_candidate) or normalized_candidate.startswith(
        normalized_primary
    )


_NUMBER_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "fifteen": 15.0,
    "twenty": 20.0,
    "fifty": 50.0,
    "one hundred": 100.0,
}
_ADR_PHRASE_RE = re.compile(
    r"\bamerican\s+depositary\s+(?:shares?|receipts?)\b",
    re.IGNORECASE,
)
_ADR_ABBREVIATION_RE = re.compile(r"\b(?:ADS|ADSs|ADR|ADRs)\b")
_ADR_RATIO_RES = (
    re.compile(
        r"(?:each|one)\s+(?:of\s+our\s+)?(?:american\s+depositary\s+(?:shares?|receipts?)|ads?|adrs?)"
        r"\s+(?:represents?|representing)\s+(?:the\s+right\s+to\s+receive\s+)?"
        r"(?P<ratio>\d+(?:\.\d+)?|one\s+hundred|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|fifty)"
        r"\s+(?:of\s+our\s+)?(?:class\s+[a-z0-9]+\s+)?(?:ordinary|common)\s+shares?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:american\s+depositary\s+(?:shares?|receipts?)|ads(?:s|'s)?|adrs?)"
        r"[^.]{0,120}?each\s+representing\s+"
        r"(?P<ratio>\d+(?:\.\d+)?|one\s+hundred|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|fifty)"
        r"\s+(?:of\s+our\s+)?(?:class\s+[a-z0-9]+\s+)?(?:ordinary|common)\s+shares?",
        re.IGNORECASE,
    ),
)


def _ratio_number(value: str) -> float | None:
    token = " ".join(str(value or "").strip().lower().split())
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    try:
        numeric = float(token)
    except ValueError:
        return None
    return numeric if numeric > 0 else None


def _normalize_filing_text(value: str) -> str:
    normalized = html.unescape(re.sub(r"<[^>]+>", " ", value))
    return " ".join(normalized.split())


def _requested_symbol_binds_adr_security(raw_text: str, ticker: str) -> bool:
    """Require the requested symbol to identify the ADR/ADS security itself.

    A foreign issuer can describe an ADR program in the same annual report as
    a separately quoted ordinary share. An issuer match therefore is not a
    security match. SEC cover pages normally bind title and trading symbol in
    one table row; the bounded plain-text fallback supports equivalent cover
    prose without granting a filing-wide symbol match.
    """

    requested = str(ticker or "").strip().upper()
    if not requested:
        return False
    symbol_re = re.compile(rf"(?<![A-Z0-9]){re.escape(requested)}(?![A-Z0-9])", re.IGNORECASE)
    adr_title_re = re.compile(
        r"(?:american\s+depositary\s+(?:shares?|receipts?)|\bADSs?\b|\bADRs?\b)",
        re.IGNORECASE,
    )
    table_rows = re.findall(
        r"<tr\b[^>]*>.*?</tr>",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for raw_row in table_rows:
        row_text = _normalize_filing_text(raw_row)
        if symbol_re.search(row_text) and adr_title_re.search(row_text):
            return True
    if table_rows:
        return False

    normalized = _normalize_filing_text(raw_text)
    for match in symbol_re.finditer(normalized):
        window = normalized[max(0, match.start() - 240) : match.end() + 240]
        if adr_title_re.search(window) and re.search(
            r"(?:trading\s+symbols?|title\s+of\s+each\s+class)",
            window,
            re.IGNORECASE,
        ):
            return True
    return False


def _trusted_sec_filing_reference(
    *,
    source_url: str,
    accession: str,
    issuer_cik: str | None,
) -> bool:
    parsed = urlparse(str(source_url or "").strip())
    host = parsed.netloc.lower().split(":", 1)[0]
    normalized_accession = str(accession or "").strip()
    normalized_cik = _normalize_cik(issuer_cik)
    if (
        parsed.scheme not in {"http", "https"}
        or not _host_matches(host, _SEC_HOST_SUFFIXES)
        or _SEC_ACCESSION_RE.fullmatch(normalized_accession) is None
        or normalized_cik is None
    ):
        return False
    compact_accession = normalized_accession.replace("-", "")
    url_text = str(source_url).lower()
    return bool(
        compact_accession in re.sub(r"[^a-z0-9]", "", url_text)
        and f"/edgar/data/{int(normalized_cik)}/" in url_text
    )


def _annual_filing_adr_identity(
    ticker: str,
    *,
    issuer_cik: str | None,
    as_of_date: str,
    db_path: str | Path | None,
) -> dict[str, Any]:
    """Read authoritative ADR identity/ratio language from a cached annual filing."""

    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    if not path.exists():
        return {}
    try:
        with db_connect(path) as conn:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(filings)").fetchall()}
            required = {
                "ticker",
                "cik",
                "accession",
                "form_type",
                "filing_date",
                "local_path",
                "primary_doc_url",
            }
            if not required <= columns:
                return {}
            status_clause = (
                "AND status IN ('OK', 'parsed', 'downloaded')" if "status" in columns else ""
            )
            if issuer_cik is not None:
                identity_clause = "printf('%010d', CAST(cik AS INTEGER)) = ?"
                identity_params: tuple[Any, ...] = (issuer_cik,)
            else:
                identity_clause = "UPPER(ticker) = ?"
                identity_params = (ticker.upper(),)
            rows = conn.execute(
                f"""
                SELECT ticker, cik, accession, form_type, filing_date, local_path,
                       primary_doc_url
                FROM filings
                WHERE {identity_clause}
                  AND form_type IN ('20-F', '20-F/A', '40-F', '40-F/A')
                  AND filing_date <= ?
                  AND local_path IS NOT NULL
                  {status_clause}
                ORDER BY filing_date DESC, id DESC
                LIMIT 12
                """,
                (*identity_params, as_of_date),
            ).fetchall()
    except sqlite3.Error:
        return {}
    if not rows:
        return {}
    newest = rows[0]
    if str(newest["form_type"] or "").upper().endswith("/A"):
        full_rows = [
            row for row in rows[1:] if not str(row["form_type"] or "").upper().endswith("/A")
        ]
        other_amendments = [row for row in rows[1:] if row not in full_rows]
        candidates = [newest, *full_rows, *other_amendments]
    else:
        candidates = list(rows)

    adr_without_ratio: dict[str, Any] | None = None
    for row in candidates:
        local_path = Path(str(row["local_path"] or ""))
        if not local_path.exists():
            continue
        try:
            raw_text = local_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        row_cik = _normalize_cik(row["cik"])
        accession = str(row["accession"] or "").strip()
        source_url = str(row["primary_doc_url"] or "").strip()
        if not _trusted_sec_filing_reference(
            source_url=source_url,
            accession=accession,
            issuer_cik=row_cik,
        ):
            continue
        # Annual filings can be large. ADR identity language is normally in
        # the cover/Item 1/market-information material; an 8 MB ceiling is
        # ample and keeps a fleet classification bounded.
        bounded_text = raw_text[:8_000_000]
        if not _requested_symbol_binds_adr_security(bounded_text, ticker):
            continue
        normalized = _normalize_filing_text(bounded_text)
        if not (_ADR_PHRASE_RE.search(normalized) or _ADR_ABBREVIATION_RE.search(normalized)):
            continue
        ratio = None
        for pattern in _ADR_RATIO_RES:
            match = pattern.search(normalized)
            if match:
                ratio = _ratio_number(match.group("ratio"))
                if ratio is not None:
                    break
        resolved = {
            "is_adr": True,
            "adr_ratio": ratio,
            "ratio_source_url": source_url if ratio is not None else None,
            "ratio_source_accession": accession if ratio is not None else None,
            "ratio_security_symbol": ticker.upper() if ratio is not None else None,
            "identity_source": "sec_cached_annual_filing",
            "identity_source_url": source_url or None,
            "identity_as_of_date": str(row["filing_date"] or "") or None,
            "identity_confidence": "HIGH",
        }
        if ratio is not None:
            return resolved
        if adr_without_ratio is None:
            adr_without_ratio = resolved
    return adr_without_ratio or {}


def _cached_foreign_annual_identity(
    ticker: str,
    *,
    issuer_cik: str | None,
    as_of_date: str,
    db_path: str | Path | None,
) -> dict[str, Any]:
    """Fail closed when a cached foreign annual filing leaves quote basis unknown.

    A 20-F/40-F establishes foreign-private-issuer status, but it does not by
    itself prove whether the requested US symbol is an ordinary share, ADR,
    or another issuer security. Until filing text or explicit identity
    evidence supplies that class/ratio, issuer shares x the security quote is
    unsafe. Direct issuer-cap evidence remains eligible downstream.
    """

    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    if not path.exists():
        return {}
    try:
        with db_connect(path) as conn:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(filings)").fetchall()}
            required = {"ticker", "cik", "form_type", "filing_date"}
            if not required <= columns:
                return {}
            url_expr = "primary_doc_url" if "primary_doc_url" in columns else "NULL"
            status_clause = ""
            parameters: list[Any] = [ticker.upper(), issuer_cik, issuer_cik, as_of_date]
            if "status" in columns:
                # Downloaded annual filings are already locally materialized
                # and therefore sufficient to establish foreign-filer status.
                # This resolver is called only by the v2 classification path;
                # v1 keeps its frozen identity behavior below.
                status_clause = " AND (status IS NULL OR status IN ('OK', 'parsed', 'downloaded'))"
            row = conn.execute(
                f"""
                SELECT ticker, cik, form_type, filing_date,
                       {url_expr} AS primary_doc_url
                FROM filings
                WHERE (ticker = ? OR (? IS NOT NULL AND printf('%010d', CAST(cik AS INTEGER)) = ?))
                  AND form_type IN ('20-F', '20-F/A', '40-F', '40-F/A')
                  AND filing_date <= ?
                  {status_clause}
                ORDER BY filing_date DESC
                LIMIT 1
                """,
                parameters,
            ).fetchone()
    except sqlite3.Error:
        return {}
    if row is None:
        return {}
    source_url = str(row["primary_doc_url"] or "").strip() or None
    return {
        "security_role": SECURITY_ROLE_SECONDARY_SECURITY,
        "is_secondary_class": False,
        "identity_source": "sec_cached_foreign_annual_filing",
        "identity_source_url": source_url,
        "identity_as_of_date": str(row["filing_date"] or "") or None,
        "identity_confidence": "HIGH",
    }


def _company_cik(
    ticker: str,
    *,
    db_path: str | Path | None,
) -> str | None:
    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    if path.exists():
        try:
            with db_connect(path) as conn:
                row = conn.execute(
                    "SELECT cik FROM companies WHERE ticker = ? LIMIT 1",
                    (ticker.upper(),),
                ).fetchone()
            if row is not None:
                resolved = _normalize_cik(row["cik"])
                if resolved:
                    return resolved
        except sqlite3.Error:
            pass
    try:
        from app.universe.ticker_cik_map import load_ticker_cik_map

        mapping = load_ticker_cik_map(refresh_if_missing=False)
    except Exception:  # noqa: BLE001 - identity resolution remains offline/best effort
        mapping = {}
    return _normalize_cik(mapping.get(ticker.upper()))


def _cached_submission_identity(
    ticker: str,
    *,
    issuer_cik: str | None,
    as_of_date: str,
    cfg: AppConfig,
) -> dict[str, Any]:
    if not issuer_cik:
        return {}
    path = cfg.cache_dir / "submissions" / f"{issuer_cik}.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            payload = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
    listed = tuple(
        dict.fromkeys(
            str(value).strip().upper()
            for value in payload.get("tickers") or []
            if str(value).strip()
        )
    )
    raw_exchanges = payload.get("exchanges") or []
    exchanges = tuple(str(value or "").strip().upper() for value in raw_exchanges)
    if not listed:
        try:
            from app.universe.ticker_cik_map import load_ticker_cik_map

            mapping = load_ticker_cik_map(refresh_if_missing=False)
        except Exception:  # noqa: BLE001
            mapping = {}
        listed = tuple(
            symbol for symbol, cik in mapping.items() if _normalize_cik(cik) == issuer_cik
        )
    requested = ticker.upper()
    recent = (
        ((payload.get("filings") or {}).get("recent") or {})
        if isinstance(payload.get("filings"), dict)
        else {}
    )
    recent_forms_raw = recent.get("form") if isinstance(recent, dict) else []
    recent_dates_raw = recent.get("filingDate") if isinstance(recent, dict) else []
    recent_forms = recent_forms_raw if isinstance(recent_forms_raw, list) else []
    recent_dates = recent_dates_raw if isinstance(recent_dates_raw, list) else []
    visible_filing_dates = [
        parsed
        for raw_date in recent_dates
        if (parsed := _safe_iso_date(raw_date)) is not None and parsed <= str(as_of_date)[:10]
    ]
    identity_as_of_date = max(visible_filing_dates) if visible_filing_dates else None
    foreign_annual_visible = any(
        str(form or "").upper() in {"20-F", "20-F/A", "40-F", "40-F/A"}
        and _safe_iso_date(recent_dates[index] if index < len(recent_dates) else None)
        and str(recent_dates[index])[:10] <= str(as_of_date)[:10]
        for index, form in enumerate(recent_forms)
    )
    # SEC's submissions ``tickers`` array is an issuer alias list, not an
    # authoritative primary-share-class ranking. Treating element zero as
    # primary made whichever class happened to appear first eligible for
    # issuer-wide shares x quote arithmetic (BRK-B was a real failure). A
    # single listed security is safe to call primary; a multi-security issuer
    # stays class/security-ambiguous until stronger evidence is supplied.
    primary_candidates = [
        symbol
        for index, symbol in enumerate(listed)
        if index < len(exchanges)
        and exchanges[index]
        and "OTC" not in exchanges[index]
        and exchanges[index] not in {"NONE", "N/A", "UNKNOWN"}
    ]
    # SEC does not promise that tickers[0] is the primary class. It does,
    # however, provide a parallel exchange list. One exchange-listed symbol
    # plus only OTC aliases (ENB is the regression case) is strong evidence
    # for the primary security. Multiple exchange-listed classes (BRK/GOOG)
    # remain intentionally ambiguous.
    exchange_bound_primary = len(listed) > 1 and len(primary_candidates) == 1
    if len(listed) == 1 and payload and not foreign_annual_visible:
        primary = listed[0]
    elif exchange_bound_primary:
        primary = primary_candidates[0]
    else:
        primary = None
    if foreign_annual_visible and len(listed) == 1:
        role = SECURITY_ROLE_SECONDARY_SECURITY
        secondary = False
    elif primary == requested:
        role = SECURITY_ROLE_PRIMARY
        secondary = False
    elif requested in listed and any(
        _share_class_variant(other, requested) for other in listed if other != requested
    ):
        role = SECURITY_ROLE_SECONDARY_CLASS
        secondary = True
    elif requested in listed and len(listed) > 1:
        role = SECURITY_ROLE_SECONDARY_SECURITY
        secondary = False
    else:
        role = SECURITY_ROLE_UNKNOWN
        secondary = None
    return {
        "issuer_primary_ticker": primary,
        "issuer_listed_tickers": listed,
        "security_role": role,
        "is_secondary_class": secondary,
        "is_adr": False if role == SECURITY_ROLE_PRIMARY else None,
        "identity_source": (
            "sec_submissions_exchange_binding"
            if exchange_bound_primary
            else "sec_submissions_foreign_security_unresolved"
            if foreign_annual_visible
            else "sec_submissions_cache"
            if payload
            else "sec_ticker_registry_cache"
        ),
        "identity_source_url": f"https://data.sec.gov/submissions/CIK{issuer_cik}.json",
        "identity_as_of_date": identity_as_of_date,
        "identity_confidence": "HIGH" if payload else "MEDIUM",
    }


def _identity_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    aliases = {
        "primary_ticker": "issuer_primary_ticker",
        "listed_tickers": "issuer_listed_tickers",
        "adr_ratio_source_url": "ratio_source_url",
    }
    for key, value in overlay.items():
        normalized_key = aliases.get(key, key)
        if value is not None and value != "":
            merged[normalized_key] = value
    return merged


def _authoritative_identity_override(
    overlay: dict[str, Any],
    *,
    as_of_date: str,
) -> bool:
    source_url = str(
        overlay.get("identity_source_url")
        or overlay.get("ratio_source_url")
        or overlay.get("adr_ratio_source_url")
        or ""
    ).strip()
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower().split(":", 1)[0]
    if not (
        _host_matches(host, _SEC_HOST_SUFFIXES) or _host_matches(host, _EXCHANGE_HOST_SUFFIXES)
    ):
        return False
    confidence = str(overlay.get("identity_confidence") or "").strip().upper()
    evidence_date = _safe_iso_date(overlay.get("identity_as_of_date"))
    target_date = _safe_iso_date(as_of_date)
    return bool(
        confidence in {"HIGH", "MEDIUM"}
        and evidence_date is not None
        and target_date is not None
        and evidence_date <= target_date
    )


def resolve_security_identity(
    ticker: str,
    *,
    as_of_date: str,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    identity_evidence: SecurityIdentity | dict[str, Any] | None = None,
) -> SecurityIdentity:
    """Resolve issuer/security identity without requiring network access."""

    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    issuer_cik = _company_cik(ticker_norm, db_path=db_path)
    base: dict[str, Any] = {
        "ticker": ticker_norm,
        "issuer_cik": issuer_cik,
    }
    submission_identity = _cached_submission_identity(
        ticker_norm,
        issuer_cik=issuer_cik,
        as_of_date=as_of_date,
        cfg=cfg,
    )
    base = _identity_overlay(base, submission_identity)
    if submission_identity.get("identity_source") != "sec_submissions_exchange_binding":
        base = _identity_overlay(
            base,
            _cached_foreign_annual_identity(
                ticker_norm,
                issuer_cik=issuer_cik,
                as_of_date=as_of_date,
                db_path=db_path,
            ),
        )
    base = _identity_overlay(
        base,
        _annual_filing_adr_identity(
            ticker_norm,
            issuer_cik=issuer_cik,
            as_of_date=as_of_date,
            db_path=db_path,
        ),
    )
    external_identity = (
        identity_evidence.to_dict()
        if isinstance(identity_evidence, SecurityIdentity)
        else identity_evidence
        if isinstance(identity_evidence, dict)
        else None
    )
    if external_identity is not None:
        inferred_safety = {
            key: base.get(key)
            for key in (
                "issuer_primary_ticker",
                "security_role",
                "is_secondary_class",
                "is_adr",
            )
        }
        inferred_role = str(inferred_safety.get("security_role") or SECURITY_ROLE_UNKNOWN).upper()
        inferred_unsafe = bool(
            inferred_role != SECURITY_ROLE_PRIMARY
            or inferred_safety.get("is_secondary_class") is True
            or inferred_safety.get("is_adr") is True
        )
        base = _identity_overlay(base, external_identity)
        merged_role = str(base.get("security_role") or SECURITY_ROLE_UNKNOWN).upper()
        merged_safe = bool(
            merged_role == SECURITY_ROLE_PRIMARY
            and _optional_bool(base.get("is_secondary_class")) is not True
            and _optional_bool(base.get("is_adr")) is not True
        )
        if (
            inferred_unsafe
            and merged_safe
            and not _authoritative_identity_override(
                external_identity,
                as_of_date=as_of_date,
            )
        ):
            base.update(inferred_safety)

    listed = tuple(
        dict.fromkeys(
            str(value).strip().upper()
            for value in base.get("issuer_listed_tickers") or []
            if str(value).strip()
        )
    )
    primary = str(base.get("issuer_primary_ticker") or "").strip().upper() or None
    is_adr = _optional_bool(base.get("is_adr"))
    secondary = _optional_bool(base.get("is_secondary_class"))
    role = str(base.get("security_role") or SECURITY_ROLE_UNKNOWN).strip().upper()
    if secondary is None and role == SECURITY_ROLE_SECONDARY_CLASS:
        secondary = True
    if is_adr is True:
        role = SECURITY_ROLE_ADR
    return SecurityIdentity(
        ticker=ticker_norm,
        issuer_cik=_normalize_cik(base.get("issuer_cik")),
        issuer_primary_ticker=primary,
        issuer_listed_tickers=listed,
        security_role=role,
        is_secondary_class=secondary,
        is_adr=is_adr,
        adr_ratio=_optional_positive_float(base.get("adr_ratio")),
        share_class_ratio=_optional_positive_float(base.get("share_class_ratio")),
        ratio_source_url=str(base.get("ratio_source_url") or "").strip() or None,
        ratio_source_accession=(str(base.get("ratio_source_accession") or "").strip() or None),
        ratio_security_symbol=(
            str(base.get("ratio_security_symbol") or "").strip().upper() or None
        ),
        identity_source=str(base.get("identity_source") or "").strip() or None,
        identity_source_url=str(base.get("identity_source_url") or "").strip() or None,
        identity_as_of_date=_safe_iso_date(base.get("identity_as_of_date")),
        identity_confidence=(str(base.get("identity_confidence") or "").strip().upper() or None),
    )


def _identity_fields(identity: SecurityIdentity) -> dict[str, Any]:
    return {
        "issuer_cik": identity.issuer_cik,
        "issuer_primary_ticker": identity.issuer_primary_ticker,
        "issuer_listed_tickers": identity.issuer_listed_tickers,
        "security_role": identity.security_role,
        "is_secondary_class": identity.is_secondary_class,
        "is_adr": identity.is_adr,
        "adr_ratio": identity.adr_ratio,
        "share_class_ratio": identity.share_class_ratio,
        "identity_source": identity.identity_source,
        "identity_source_url": identity.identity_source_url,
        "identity_as_of_date": identity.identity_as_of_date,
        "identity_confidence": identity.identity_confidence,
        "ratio_source_url": identity.ratio_source_url,
        "ratio_source_accession": identity.ratio_source_accession,
        "ratio_security_symbol": identity.ratio_security_symbol,
    }


def _authoritative_ratio_evidence(
    identity: SecurityIdentity,
    *,
    as_of_date: str,
) -> bool:
    source_url = str(identity.ratio_source_url or "").strip()
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    confidence = str(identity.identity_confidence or "").upper()
    if confidence not in {"HIGH", "MEDIUM"}:
        return False
    evidence_date = _safe_iso_date(identity.identity_as_of_date)
    target_date = _safe_iso_date(as_of_date)
    if evidence_date is None or target_date is None or evidence_date > target_date:
        return False
    host = parsed.netloc.lower().split(":", 1)[0]
    source = str(identity.identity_source or "").strip().lower()
    if str(identity.ratio_security_symbol or "").strip().upper() != identity.ticker.upper():
        return False
    if _host_matches(host, _SEC_HOST_SUFFIXES):
        return bool(
            source in {"sec_cached_annual_filing", "issuer_filing"}
            and _trusted_sec_filing_reference(
                source_url=source_url,
                accession=str(identity.ratio_source_accession or ""),
                issuer_cik=identity.issuer_cik,
            )
        )
    if source == "exchange_listing":
        return _host_matches(host, _EXCHANGE_HOST_SUFFIXES)
    if source == "depositary_agreement":
        return _host_matches(host, _DEPOSITARY_HOST_SUFFIXES)
    return False


def _issuer_share_quote_ratio(
    identity: SecurityIdentity,
    *,
    as_of_date: str,
    require_authoritative_primary: bool = False,
) -> tuple[float | None, str | None]:
    if identity.is_adr is True:
        if identity.adr_ratio is None or not _authoritative_ratio_evidence(
            identity,
            as_of_date=as_of_date,
        ):
            return None, UNSAFE_ADR_RATIO_MISSING
        return float(identity.adr_ratio), None
    if identity.is_secondary_class is True or identity.security_role in {
        SECURITY_ROLE_SECONDARY_CLASS,
    }:
        if identity.share_class_ratio is None or not _authoritative_ratio_evidence(
            identity,
            as_of_date=as_of_date,
        ):
            return None, UNSAFE_SECONDARY_CLASS_RATIO_MISSING
        return float(identity.share_class_ratio), None
    if identity.security_role in {
        SECURITY_ROLE_SECONDARY_SECURITY,
        SECURITY_ROLE_UNKNOWN,
    }:
        return None, UNSAFE_SECURITY_ROLE_UNRESOLVED
    if require_authoritative_primary:
        primary_ticker = str(identity.issuer_primary_ticker or "").strip().upper()
        if not (
            identity.security_role == SECURITY_ROLE_PRIMARY
            and primary_ticker == identity.ticker.upper()
            and identity.is_secondary_class is False
            and identity.is_adr is False
            and _authoritative_identity_override(
                identity.to_dict(),
                as_of_date=as_of_date,
            )
        ):
            return None, UNSAFE_PRIMARY_IDENTITY_UNVERIFIED
    return 1.0, None


def _price_from_value(
    value: Any,
    *,
    default_as_of_date: str,
    default_source: str,
    default_confidence: str | None,
) -> _ResolvedPrice | None:
    requested_date = _safe_iso_date(default_as_of_date)
    if requested_date is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0:
        return _ResolvedPrice(
            price=float(value),
            as_of_date=requested_date,
            source=default_source,
            source_url=None,
            currency=None,
            confidence=default_confidence,
        )
    if isinstance(value, dict):
        price = _optional_positive_float(value.get("price"))
        if price is None:
            return None
        price_date = _price_date_at_or_before(
            value.get("as_of_date") or requested_date,
            requested_date,
        )
        if price_date is None:
            return None
        split_lineage = _price_basis_contract(
            price=price,
            raw_price=value.get("raw_price"),
            explicit_basis=value.get("price_basis"),
            explicit_factor=value.get("split_adjustment_factor"),
            explicit_effective_date=value.get("split_effective_date"),
        )
        if split_lineage is None:
            return None
        raw_price, price_basis, split_factor, split_effective_date = split_lineage
        return _ResolvedPrice(
            price=price,
            as_of_date=price_date,
            source=str(value.get("source") or default_source),
            source_url=str(value.get("url") or value.get("source_url") or "").strip() or None,
            currency=str(value.get("currency") or value.get("price_currency") or "").strip().upper()
            or None,
            confidence=str(value.get("confidence") or default_confidence or "").strip().upper()
            or None,
            raw_price=raw_price,
            price_basis=price_basis,
            split_adjustment_factor=split_factor,
            split_effective_date=split_effective_date,
            split_event=(
                dict(value["split_event"]) if isinstance(value.get("split_event"), dict) else None
            ),
            no_intervening_split_proof=(
                dict(value["no_intervening_split_proof"])
                if isinstance(value.get("no_intervening_split_proof"), dict)
                else None
            ),
        )
    price = _optional_positive_float(getattr(value, "price", None))
    if price is None:
        return None
    price_date = _price_date_at_or_before(
        getattr(value, "as_of_date", None) or requested_date,
        requested_date,
    )
    if price_date is None:
        return None
    split_lineage = _price_basis_contract(
        price=price,
        raw_price=getattr(value, "raw_price", None),
        explicit_basis=getattr(value, "price_basis", None),
        explicit_factor=getattr(value, "split_adjustment_factor", None),
        explicit_effective_date=getattr(value, "split_effective_date", None),
    )
    if split_lineage is None:
        return None
    raw_price, price_basis, split_factor, split_effective_date = split_lineage
    split_event = getattr(value, "split_event", None)
    no_intervening_split_proof = getattr(
        value,
        "no_intervening_split_proof",
        None,
    )
    return _ResolvedPrice(
        price=price,
        as_of_date=price_date,
        source=str(getattr(value, "source", None) or default_source),
        source_url=str(getattr(value, "url", None) or "").strip() or None,
        currency=str(getattr(value, "currency", None) or "").strip().upper() or None,
        confidence=str(getattr(value, "confidence", None) or default_confidence or "")
        .strip()
        .upper()
        or None,
        raw_price=raw_price,
        price_basis=price_basis,
        split_adjustment_factor=split_factor,
        split_effective_date=split_effective_date,
        split_event=(dict(split_event) if isinstance(split_event, dict) else None),
        no_intervening_split_proof=(
            dict(no_intervening_split_proof)
            if isinstance(no_intervening_split_proof, dict)
            else None
        ),
    )


def _terminal_evidence_from_dict(payload: dict[str, Any]) -> TerminalCapEvidence | None:
    if str(payload.get("market_cap_basis") or "").strip().upper() != "DIRECT_ISSUER_MARKET_CAP":
        return None
    raw_cap = direct_market_cap_mm_from_payload(payload)
    if raw_cap is None:
        return None
    source_kind = (
        str(payload.get("source_kind") or payload.get("market_cap_source_kind") or "")
        .strip()
        .upper()
    )
    confidence = (
        str(payload.get("confidence") or payload.get("market_cap_confidence") or "").strip().upper()
    )
    source_url = str(payload.get("source_url") or "").strip()
    as_of_date = _safe_iso_date(payload.get("as_of_date") or payload.get("effective_as_of_date"))
    if source_kind not in CAP_SOURCE_KINDS or confidence not in CAP_CONFIDENCE_LEVELS:
        return None
    if not source_url or as_of_date is None:
        return None
    if source_kind == CAP_SOURCE_KIND_SEARCH and not source_url.lower().startswith(
        ("http://", "https://")
    ):
        return None
    return TerminalCapEvidence(
        ticker=str(payload.get("ticker") or "").strip().upper(),
        market_cap_mm=float(raw_cap),
        source_kind=source_kind,  # type: ignore[arg-type]
        source_name=str(
            payload.get("source_name") or payload.get("provider") or source_kind.lower()
        ),
        source_url=source_url,
        as_of_date=as_of_date,
        confidence=confidence,  # type: ignore[arg-type]
        issuer_cik=_normalize_cik(payload.get("issuer_cik") or payload.get("cik")),
        issuer_name=str(payload.get("issuer_name") or "").strip() or None,
        security_name=str(
            payload.get("security_name") or payload.get("resolved_name") or ""
        ).strip()
        or None,
        price_used=_optional_positive_float(payload.get("price_used", payload.get("price"))),
        price_source=str(payload.get("price_source") or "").strip() or None,
        price_source_url=str(payload.get("price_source_url") or "").strip() or None,
        price_as_of_date=_safe_iso_date(payload.get("price_as_of_date")),
        price_currency=str(payload.get("price_currency") or "").strip().upper() or None,
        price_confidence=str(payload.get("price_confidence") or "").strip().upper() or None,
        price_basis=str(payload.get("price_basis") or "").strip().upper() or None,
        raw_price=_optional_positive_float(payload.get("raw_price")),
        split_adjustment_factor=_optional_positive_float(payload.get("split_adjustment_factor")),
        split_effective_date=_safe_iso_date(payload.get("split_effective_date")),
        local_authority_schema=(
            str(payload.get("local_authority_schema") or "").strip().upper() or None
        ),
        record_provenance=(str(payload.get("record_provenance") or "").strip().upper() or None),
        detail=str(payload.get("detail") or "").strip() or None,
    )


def _valid_terminal_evidence(
    evidence: TerminalCapEvidence,
    *,
    ticker: str,
    as_of_date: str,
    identity: SecurityIdentity,
    allow_local_authoritative: bool = False,
) -> bool:
    if evidence.ticker and evidence.ticker.upper() != ticker.upper():
        return False
    if evidence.source_kind not in CAP_SOURCE_KINDS:
        return False
    if evidence.confidence not in CAP_CONFIDENCE_LEVELS or not evidence.source_url:
        return False
    parsed_url = urlparse(evidence.source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return False
    derived_kind = derive_terminal_source_kind(
        source_url=evidence.source_url,
        claimed_kind=evidence.source_kind,
        allow_local_authoritative=allow_local_authoritative,
        local_authority_schema=evidence.local_authority_schema,
        record_provenance=evidence.record_provenance,
    )
    if evidence.source_kind != derived_kind:
        return False
    if evidence.source_kind == CAP_SOURCE_KIND_SEARCH and evidence.confidence == "LOW":
        return False
    evidence_cik = _normalize_cik(evidence.issuer_cik)
    identity_cik = _normalize_cik(identity.issuer_cik)
    # A terminal cap is issuer-wide by contract. Ticker text cannot bind the
    # issuer because symbols can be reused and secondary securities share an
    # issuer. Until the v2 identity stage resolves a CIK, direct cap evidence
    # may remain visible in diagnostics but cannot settle scope.
    if identity_cik is None or evidence_cik != identity_cik:
        return False
    target = _safe_iso_date(as_of_date)
    used = _safe_iso_date(evidence.as_of_date)
    if target is None or used is None:
        return False
    target_date = date.fromisoformat(target)
    used_date = date.fromisoformat(used)
    age_days = (target_date - used_date).days
    return 0 <= age_days <= TERMINAL_CAP_MAX_AGE_DAYS and evidence.market_cap_mm > 0


def _persisted_terminal_cap_evidence(
    ticker: str,
    *,
    as_of_date: str,
    db_path: str | Path | None,
    identity: SecurityIdentity,
) -> list[TerminalCapEvidence]:
    """Read explicitly direct-cap rows from the existing market_caps store.

    Historical rows that only contain shares x price are intentionally
    ignored. A terminal row must opt in with ``market_cap_basis`` and explicit
    units/source/as-of/confidence metadata in ``payload_json``.
    """

    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    if not path.exists():
        return []
    try:
        with db_connect(path) as conn:
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(market_caps)").fetchall()
            }
            required = {
                "ticker",
                "effective_as_of_date",
                "market_cap",
                "market_cap_status",
                "provider",
                "source_url",
                "payload_json",
                "id",
            }
            if not required <= columns:
                return []
            rows = conn.execute(
                """
                SELECT ticker, effective_as_of_date, market_cap, provider,
                       source_url, payload_json
                FROM market_caps
                WHERE ticker = ?
                  AND market_cap_status = 'OK'
                  AND market_cap IS NOT NULL
                  AND effective_as_of_date <= ?
                ORDER BY effective_as_of_date DESC, id DESC
                LIMIT 25
                """,
                (ticker.upper(), as_of_date),
            ).fetchall()
    except sqlite3.Error:
        return []
    evidence: list[TerminalCapEvidence] = []
    for row in rows:
        try:
            raw = json.loads(row["payload_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        merged = {
            **raw,
            "ticker": raw.get("ticker") or row["ticker"],
            "effective_as_of_date": raw.get("effective_as_of_date") or row["effective_as_of_date"],
            "provider": raw.get("provider") or row["provider"],
            "source_url": raw.get("source_url") or row["source_url"],
        }
        if not any(
            raw.get(field) not in (None, "")
            for field in (
                "market_cap_mm",
                "market_cap_usd",
                "resolved_market_cap_usd",
                "market_cap",
            )
        ):
            merged["market_cap"] = row["market_cap"]
        parsed = _terminal_evidence_from_dict(merged)
        if parsed is not None and _valid_terminal_evidence(
            parsed,
            ticker=ticker,
            as_of_date=as_of_date,
            identity=identity,
            allow_local_authoritative=True,
        ):
            evidence.append(parsed)
    return evidence


def _lookup_terminal_cap_evidence(
    ticker: str,
    *,
    as_of_date: str,
    identity: SecurityIdentity,
    db_path: str | Path | None,
    terminal_cap_lookup: TerminalCapLookup | None,
    include_persisted: bool = True,
) -> list[TerminalCapEvidence]:
    evidence = (
        _persisted_terminal_cap_evidence(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
            identity=identity,
        )
        if include_persisted
        else []
    )
    if terminal_cap_lookup is not None:
        try:
            raw = terminal_cap_lookup(ticker, as_of_date, identity)
        except Exception:  # noqa: BLE001 - terminal lookup cannot abort the scan
            raw = None
        if isinstance(raw, (TerminalCapEvidence, dict)):
            candidates: Iterable[TerminalCapEvidence | dict[str, Any]] = [raw]
        elif raw is None or isinstance(raw, (str, bytes)):
            candidates = []
        else:
            candidates = raw
        for item in candidates:
            parsed = (
                item
                if isinstance(item, TerminalCapEvidence)
                else _terminal_evidence_from_dict(item)
            )
            if parsed is not None and _valid_terminal_evidence(
                parsed,
                ticker=ticker,
                as_of_date=as_of_date,
                identity=identity,
                allow_local_authoritative=bool(
                    getattr(
                        terminal_cap_lookup,
                        "_voe_prevalidated_local_ledger",
                        False,
                    )
                ),
            ):
                evidence.append(parsed)
    return sorted(
        evidence,
        key=lambda item: (
            CAP_SOURCE_KIND_PRIORITY[item.source_kind],
            -date.fromisoformat(item.as_of_date).toordinal(),
            item.source_name,
        ),
    )


def _terminal_classification(
    evidence: TerminalCapEvidence,
    *,
    ticker: str,
    requested_as_of_date: str,
    identity: SecurityIdentity,
    fallback_price: _ResolvedPrice | None = None,
) -> CapClassification:
    embedded_price = None
    if evidence.price_used is not None and evidence.price_as_of_date is not None:
        embedded_price = _price_from_value(
            {
                "price": evidence.price_used,
                "as_of_date": evidence.price_as_of_date,
                "source": evidence.price_source,
                "source_url": evidence.price_source_url,
                "currency": evidence.price_currency,
                "confidence": evidence.price_confidence,
                "price_basis": evidence.price_basis,
                "raw_price": evidence.raw_price,
                "split_adjustment_factor": evidence.split_adjustment_factor,
                "split_effective_date": evidence.split_effective_date,
            },
            default_as_of_date=requested_as_of_date,
            default_source="terminal_cap_evidence",
            default_confidence=evidence.price_confidence,
        )
    selected_price = embedded_price or fallback_price
    return CapClassification(
        ticker=ticker,
        as_of_date=requested_as_of_date,
        market_cap_mm=float(evidence.market_cap_mm),
        market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
        cap_source=CAP_SOURCE_BY_KIND[evidence.source_kind],
        cap_band=band_for_market_cap(float(evidence.market_cap_mm)),
        price_used=selected_price.price if selected_price else None,
        price_as_of_date=selected_price.as_of_date if selected_price else None,
        price_source=selected_price.source if selected_price else None,
        price_source_url=selected_price.source_url if selected_price else None,
        price_currency=selected_price.currency if selected_price else None,
        price_confidence=selected_price.confidence if selected_price else None,
        cap_effective_as_of_date=evidence.as_of_date,
        cap_source_kind=evidence.source_kind,
        cap_source_name=evidence.source_name,
        cap_source_url=evidence.source_url,
        cap_confidence=evidence.confidence,
        detail=evidence.detail or "direct_issuer_market_cap",
        cap_method="direct_issuer_market_cap",
        **_classification_quote_fields(ticker, selected_price),
        **_identity_fields(identity),
    )


def _strict_cap_source_metadata(
    coverage: dict[str, Any],
    *,
    companyfacts_url: str | None,
) -> tuple[str, str, str | None]:
    shares_source = str(
        coverage.get("shares_source_resolution") or coverage.get("shares_source") or "unknown"
    ).strip()
    refs = [str(item).strip() for item in coverage.get("derived_from") or [] if str(item).strip()]
    explicit_source_url = str(
        coverage.get("shares_source_url") or coverage.get("source_url") or ""
    ).strip()
    source_url = explicit_source_url or next(
        (ref for ref in refs if ref.lower().startswith(("http://", "https://"))), None
    )
    if "companyfacts" in shares_source.lower() or (
        source_url and "data.sec.gov" in source_url.lower()
    ):
        return CAP_SOURCE_KIND_SEC, f"sec_{shares_source}", source_url or companyfacts_url
    return CAP_SOURCE_KIND_LOCAL, shares_source, source_url


def _last_known_companyfacts_shares(
    ticker: str,
    *,
    as_of_date: str,
    db_path: str | Path | None = None,
    require_filed_asof: bool = False,
    identity: SecurityIdentity | None = None,
) -> tuple[
    float | None,
    str | None,
    str | None,
    str | None,
    float | None,
    str | None,
]:
    """Latest shares_outstanding (shares_millions) at or before as_of_date.

    Reads across FY and quarterly rows — the dei cover-page instant is filed
    with every 10-K/10-Q, so the newest period_end is the freshest share
    count the local cache knows about.
    """
    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    if not path.exists():
        return None, None, None, None, None, None

    def normalized_row(
        row: Any,
    ) -> tuple[
        float | None,
        str | None,
        str | None,
        str | None,
        float | None,
        str | None,
    ]:
        raw_value = row["value"]
        raw_unit = str(row["units"] or "").strip()
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            return None, None, None, None, None, None
        raw_number = float(raw_value)
        if raw_number <= 0:
            return None, None, None, None, None, None
        if raw_unit == "shares":
            normalized_mm = raw_number / 1_000_000.0
        elif raw_unit == SHARES_UNIT_MILLIONS:
            normalized_mm = raw_number
        else:
            return None, None, None, None, None, None
        return (
            normalized_mm,
            str(row["period_end"]) if row["period_end"] else None,
            str(row["source_url"]).strip() if row["source_url"] else None,
            str(row["filed_date"]) if row["filed_date"] else None,
            raw_number,
            raw_unit,
        )

    try:
        with db_connect(path) as conn:
            if require_filed_asof:
                from app.util.financial_data_access import issuer_companyfacts_rows

                _scope, issuer_rows = issuer_companyfacts_rows(
                    conn,
                    ticker,
                    columns=("value", "units", "period_end", "source_url", "filed_date"),
                    issuer_cik=identity.issuer_cik if identity is not None else None,
                    aliases=(identity.issuer_listed_tickers if identity is not None else ()),
                    line_items=("shares_outstanding",),
                    as_of_date=as_of_date,
                    value_not_null=True,
                    require_filed_asof=True,
                    order_by="period_end DESC, filed_date DESC",
                )
                valid_rows = [
                    row
                    for row in issuer_rows
                    if isinstance(row["value"], (int, float)) and float(row["value"]) > 0
                ]
                if not valid_rows:
                    return None, None, None, None, None, None
                row = valid_rows[0]
                return normalized_row(row)
            fact_columns = {
                str(item[1])
                for item in conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
            }
            filed_clause = (
                "AND filed_date IS NOT NULL AND filed_date <> '' AND filed_date <= ?"
                if require_filed_asof
                else ""
            )
            params: list[Any] = [str(ticker).upper(), str(as_of_date)]
            if require_filed_asof:
                params.append(str(as_of_date))
            rows = list(
                conn.execute(
                    f"""
                    SELECT value, units, period_end, source_url,
                           {"filed_date" if "filed_date" in fact_columns else "NULL"} AS filed_date
                    FROM companyfacts_facts
                    WHERE ticker = ?
                      AND line_item = 'shares_outstanding'
                      AND value IS NOT NULL
                      AND value > 0
                      AND period_end <= ?
                      {filed_clause}
                    """,
                    params,
                ).fetchall()
            )
    except sqlite3.Error:
        return None, None, None, None, None, None
    if not rows:
        return None, None, None, None, None, None
    row = max(
        rows,
        key=lambda item: (
            str(item["period_end"] or ""),
            str(item["filed_date"] or ""),
        ),
    )
    return normalized_row(row)


def _eodhd_fundamentals_market_cap_mm(ticker: str, cfg: AppConfig) -> tuple[float | None, str]:
    """EODHD fundamentals MarketCapitalization in millions. INERT by default.

    Only reachable when ``cfg.cap_eodhd_fundamentals_enabled`` is true AND an
    API key is configured. The endpoint 403s on the current subscription;
    this exists so a plan upgrade is a config flip, not a code change.
    """
    if not cfg.eodhd_apikey:
        return None, "EODHD_APIKEY_MISSING"
    try:
        from app.util.http import HttpClient

        url = f"{cfg.eodhd_base_url}/api/fundamentals/{str(ticker).upper()}.{cfg.eodhd_exchange}"
        payload = HttpClient(cfg).get_json(
            url,
            params={"api_token": cfg.eodhd_apikey, "fmt": "json"},
            use_cache=True,
            cache_ttl_seconds=86400,
        )
    except Exception as exc:  # noqa: BLE001 - tier must degrade, never abort
        return None, f"EODHD_FUNDAMENTALS_ERROR:{type(exc).__name__}"
    highlights = payload.get("Highlights") if isinstance(payload, dict) else None
    raw = highlights.get("MarketCapitalization") if isinstance(highlights, dict) else None
    if isinstance(raw, (int, float)) and float(raw) > 0:
        return float(raw) / 1_000_000.0, "OK"
    return None, "EODHD_FUNDAMENTALS_NO_MARKET_CAP"


def default_price_lookup(cfg: AppConfig | None = None) -> PriceLookup:
    """Provider-backed current-price lookup with one lazily built provider."""
    state: dict[str, Any] = {}

    def lookup(ticker: str, as_of_date: str) -> Any:
        if "provider" not in state:
            try:
                from app.market.price_provider import build_price_provider

                resolved_cfg = cfg or get_config()
                state["provider"] = build_price_provider(
                    cfg=resolved_cfg,
                    with_prices=True,
                    fallback_days=getattr(resolved_cfg, "price_fallback_days", None),
                )
            except Exception:  # noqa: BLE001 - band filter must not abort a sweep
                state["provider"] = None
        provider = state["provider"]
        if provider is None:
            return None
        try:
            snap = provider.get_price_asof(str(ticker).upper(), str(as_of_date))
        except Exception:  # noqa: BLE001
            return None
        if snap is not None and isinstance(snap.price, (int, float)) and snap.price > 0:
            # Preserve the provider's immutable as-of/source/URL/confidence
            # record; callers that inject the legacy float-only lookup remain
            # supported by ``_price_from_value``.
            return snap
        return None

    return lookup


def classify_market_cap_for_band_filter(
    ticker: str,
    *,
    as_of_date: str,
    asof_price: Any = None,
    asof_price_provenance: dict[str, Any] | None = None,
    current_price: Any = None,
    run_id: str | None = None,
    db_path: str | Path | None = None,
    price_lookup: PriceLookup | None = None,
    cfg: AppConfig | None = None,
    identity_evidence: SecurityIdentity | dict[str, Any] | None = None,
    terminal_cap_lookup: TerminalCapLookup | None = None,
    pipeline_version: str = "v1",
    companyfacts_cache_only: bool = False,
) -> CapClassification:
    """Run the band-filter chain for one ticker.

    ``asof_price`` feeds the strict as-of tier; when omitted it is skipped.
    ``current_price`` feeds the stale-shares tier; when omitted the
    provider-backed ``price_lookup`` resolves it, falling back to
    ``asof_price`` so a cached scorecard price can still anchor a stale-shares
    cap offline. Legacy float prices remain supported, while a PriceSnapshot
    or mapping preserves source/as-of/URL/confidence.

    Known ADR and secondary-class quotes may not be multiplied by issuer-wide
    shares unless authoritative ratio evidence is present. Direct issuer-cap
    terminal evidence remains eligible because it does not perform that unsafe
    multiplication. Search-backed terminal evidence must carry URL, as-of, and
    confidence or it is ignored.
    """
    cfg = cfg or get_config()
    if db_path is not None:
        cfg = cfg.model_copy(update={"db_path": Path(db_path)})
    ticker_norm = str(ticker or "").strip().upper()
    asof_norm = _safe_iso_date(as_of_date) or str(as_of_date or "").strip()
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    identity = resolve_security_identity(
        ticker_norm,
        as_of_date=asof_norm,
        db_path=db_path,
        cfg=cfg,
        identity_evidence=identity_evidence,
    )
    if normalized_pipeline == "v2":
        prevalidated_ledger_lookup = (
            terminal_cap_lookup
            if bool(
                getattr(
                    terminal_cap_lookup,
                    "_voe_prevalidated_local_ledger",
                    False,
                )
            )
            else None
        )
        terminal_evidence = _lookup_terminal_cap_evidence(
            ticker_norm,
            as_of_date=asof_norm,
            identity=identity,
            db_path=db_path,
            terminal_cap_lookup=prevalidated_ledger_lookup,
        )
    else:
        # V1 cannot consume terminal-cap evidence, but issuer-share arithmetic
        # must still prove that the requested quote is the issuer's
        # authoritative primary quote (or carry authoritative ADR/share-class
        # ratio evidence). Unknown identity is never silently treated as a
        # one-for-one primary listing.
        terminal_evidence = []
        prevalidated_ledger_lookup = None

    def terminal_for(*source_kinds: str) -> TerminalCapEvidence | None:
        return next(
            (item for item in terminal_evidence if item.source_kind in source_kinds),
            None,
        )

    asof_input: Any = asof_price
    if isinstance(asof_price_provenance, dict):
        asof_input = {**asof_price_provenance, "price": asof_price}
    asof_price_record = _price_from_value(
        asof_input,
        default_as_of_date=asof_norm,
        default_source="scorecard",
        default_confidence="MEDIUM",
    )

    def cap_arithmetic_currency_ok(price: _ResolvedPrice | None) -> bool:
        return bool(
            price is not None
            and (normalized_pipeline == "v1" or str(price.currency or "").strip().upper() == "USD")
        )

    def share_basis_is_safe(price: _ResolvedPrice | None) -> bool:
        # Tier 1's legacy helper only understands raw shares. Adjusted quotes
        # are routed to tier 3, where the exact share date is available and
        # both operands can be normalized before multiplication.
        return bool(price is not None and price.price_basis == PRICE_BASIS_UNADJUSTED)

    def normalized_shares(
        price: _ResolvedPrice | None,
        *,
        raw_shares_mm: float | None,
        shares_as_of_date: str | None,
    ) -> tuple[float, str] | None:
        return _normalized_shares_for_price(
            price,
            ticker=ticker_norm,
            issuer_cik=identity.issuer_cik,
            raw_shares_mm=raw_shares_mm,
            shares_as_of_date=shares_as_of_date,
            run_as_of_date=asof_norm,
        )

    def cap_currency_reason(price: _ResolvedPrice | None) -> str | None:
        if normalized_pipeline != "v2" or price is None:
            return None
        currency = str(price.currency or "").strip().upper()
        return (
            f"NON_USD_CAP_PRICE_UNSUPPORTED:{currency}"
            if currency and currency != "USD"
            else "CAP_PRICE_CURRENCY_UNRESOLVED"
            if not currency
            else None
        )

    terminal_price_resolved = False
    terminal_price_cache: _ResolvedPrice | None = None

    def terminal_price() -> _ResolvedPrice | None:
        """Resolve the immutable packet quote even when cap itself is direct.

        A direct issuer-cap row settles scope without shares arithmetic, but
        it does not make a packet price optional.  Reuse an exact-date
        scorecard quote first; production fleet classification supplies one
        shared provider lookup for the residuals.
        """

        nonlocal terminal_price_resolved, terminal_price_cache
        if terminal_price_resolved:
            return terminal_price_cache
        terminal_price_resolved = True
        terminal_price_cache = asof_price_record
        if terminal_price_cache is None and price_lookup is not None:
            terminal_price_cache = _price_from_value(
                price_lookup(ticker_norm, asof_norm),
                default_as_of_date=asof_norm,
                default_source="provider",
                default_confidence="MEDIUM",
            )
        return terminal_price_cache

    # Tier 0: a direct issuer cap from an explicitly authoritative local row.
    # Old market_caps rows derived from shares x price are not eligible.
    direct = terminal_for(CAP_SOURCE_KIND_LOCAL)
    if direct is not None:
        return _terminal_classification(
            direct,
            ticker=ticker_norm,
            requested_as_of_date=asof_norm,
            identity=identity,
            fallback_price=terminal_price(),
        )

    issuer_quote_ratio, unsafe_identity_reason = _issuer_share_quote_ratio(
        identity,
        as_of_date=asof_norm,
        require_authoritative_primary=(normalized_pipeline == "v1"),
    )
    companyfacts_url = (
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{identity.issuer_cik}.json"
        if identity.issuer_cik
        else None
    )
    strict_v2_shares: tuple[
        float | None,
        str | None,
        str | None,
        str | None,
        float | None,
        str | None,
    ] = (
        _last_known_companyfacts_shares(
            ticker_norm,
            as_of_date=asof_norm,
            db_path=db_path,
            require_filed_asof=True,
            identity=identity,
        )
        if normalized_pipeline == "v2"
        else (None, None, None, None, None, None)
    )

    # Tier 1: strict as-of companyfacts cap (pre-existing behavior), guarded
    # against ADR/class quote-basis mismatches.
    if (
        cap_arithmetic_currency_ok(asof_price_record)
        and share_basis_is_safe(asof_price_record)
        and issuer_quote_ratio is not None
    ):
        assert asof_price_record is not None
        if normalized_pipeline == "v2":
            (
                strict_shares_mm,
                strict_shares_period_end,
                strict_source_url,
                strict_shares_filed_date,
                strict_raw_shares_value,
                strict_raw_shares_unit,
            ) = strict_v2_shares
            market_cap = (
                float(asof_price_record.price) * float(strict_shares_mm)
                if strict_shares_mm is not None
                else None
            )
            coverage = {
                "shares_outstanding": strict_shares_mm,
                "shares_asof_used": strict_shares_period_end,
                "shares_filed_date": strict_shares_filed_date,
                "shares_source": "sec_companyfacts",
                "shares_source_url": strict_source_url or companyfacts_url,
                "shares_source_resolution": "sec_companyfacts_filed_asof",
                "derived_from": [ref for ref in (strict_source_url, companyfacts_url) if ref],
                "shares_unit": SHARES_UNIT_MILLIONS,
                "raw_shares_source_value": strict_raw_shares_value,
                "raw_shares_source_unit": strict_raw_shares_unit,
                "raw_shares_outstanding_mm": strict_shares_mm,
            }
        else:
            try:
                from app.valuation.shares import resolve_market_cap_from_price_asof

                market_cap, coverage = resolve_market_cap_from_price_asof(
                    ticker=ticker_norm,
                    as_of_date=asof_norm,
                    price=float(asof_price_record.price),
                    run_id=run_id,
                    quote_lineage={
                        "ticker": ticker_norm,
                        "as_of_date": asof_price_record.as_of_date,
                        "currency": asof_price_record.currency,
                        "unit": "USD_per_share",
                        "source": asof_price_record.source,
                        "source_url": asof_price_record.source_url,
                        "price_basis": asof_price_record.price_basis,
                        "raw_price": asof_price_record.raw_price,
                        "split_adjustment_factor": (asof_price_record.split_adjustment_factor),
                        "split_effective_date": (asof_price_record.split_effective_date),
                        "split_event": asof_price_record.split_event,
                        "no_intervening_split_proof": (
                            asof_price_record.no_intervening_split_proof
                        ),
                    },
                    require_split_lineage=(asof_price_record.price_basis is not None),
                    db_path=db_path,
                    cfg=cfg,
                    companyfacts_cache_only=companyfacts_cache_only,
                )
            except Exception:  # noqa: BLE001 - chain degrades tier by tier
                market_cap, coverage = None, {}
        if isinstance(market_cap, (int, float)) and float(market_cap) > 0:
            raw_coverage_shares = (
                float(
                    coverage.get(
                        "raw_shares_outstanding_mm",
                        coverage.get("shares_outstanding"),
                    )
                )
                if isinstance(
                    coverage.get(
                        "raw_shares_outstanding_mm",
                        coverage.get("shares_outstanding"),
                    ),
                    (int, float),
                )
                else None
            )
            normalized_share_lineage = normalized_shares(
                asof_price_record,
                raw_shares_mm=raw_coverage_shares,
                shares_as_of_date=(
                    str(coverage.get("shares_asof_used"))
                    if coverage.get("shares_asof_used")
                    else None
                ),
            )
            if normalized_share_lineage is None:
                market_cap = None
        if isinstance(market_cap, (int, float)) and float(market_cap) > 0:
            assert normalized_share_lineage is not None
            normalized_coverage_shares, normalized_shares_basis = normalized_share_lineage
            cap_source_kind, cap_source_name, cap_source_url = _strict_cap_source_metadata(
                coverage,
                companyfacts_url=companyfacts_url,
            )
            adjusted_cap = float(market_cap) / float(issuer_quote_ratio)
            ratio_detail = (
                f":quote_to_issuer_share_ratio={issuer_quote_ratio}"
                if issuer_quote_ratio != 1.0
                else ""
            )
            return CapClassification(
                ticker=ticker_norm,
                as_of_date=asof_norm,
                market_cap_mm=adjusted_cap,
                market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
                cap_source=CAP_SOURCE_ASOF_COMPANYFACTS,
                cap_band=band_for_market_cap(adjusted_cap),
                price_used=float(asof_price_record.price),
                price_as_of_date=asof_price_record.as_of_date,
                price_source=asof_price_record.source,
                price_source_url=asof_price_record.source_url,
                price_currency=asof_price_record.currency,
                price_confidence=asof_price_record.confidence,
                shares_mm=normalized_coverage_shares,
                shares_period_end=(
                    str(coverage.get("shares_asof_used"))
                    if coverage.get("shares_asof_used")
                    else None
                ),
                shares_filed_date=(
                    str(coverage.get("shares_filed_date"))
                    if coverage.get("shares_filed_date")
                    else None
                ),
                shares_source=str(
                    coverage.get("shares_source") or coverage.get("shares_source_resolution") or ""
                ).strip()
                or None,
                shares_source_url=(
                    str(coverage.get("shares_source_url")).strip()
                    if coverage.get("shares_source_url")
                    else cap_source_url
                ),
                cap_effective_as_of_date=asof_price_record.as_of_date,
                cap_source_kind=cap_source_kind,
                cap_source_name=cap_source_name,
                cap_source_url=cap_source_url,
                cap_confidence=("HIGH" if asof_price_record.confidence == "HIGH" else "MEDIUM"),
                cap_method="price_times_shares_divided_by_issuer_quote_ratio",
                issuer_quote_ratio=float(issuer_quote_ratio),
                detail=(
                    f"shares_source={coverage.get('shares_source_resolution') or 'unknown'}"
                    f"{ratio_detail}"
                ),
                **_classification_quote_fields(
                    ticker_norm,
                    asof_price_record,
                    shares_mm=normalized_coverage_shares,
                    raw_shares_mm=raw_coverage_shares,
                    raw_shares_source_value=coverage.get("raw_shares_source_value"),
                    raw_shares_source_unit=coverage.get("raw_shares_source_unit"),
                    shares_basis=normalized_shares_basis,
                ),
                **_identity_fields(identity),
            )

    # V1's configured fundamentals slot historically preceded stale shares.
    if normalized_pipeline == "v1" and bool(getattr(cfg, "cap_eodhd_fundamentals_enabled", False)):
        eodhd_cap, eodhd_detail = _eodhd_fundamentals_market_cap_mm(
            ticker_norm,
            cfg,
        )
        if isinstance(eodhd_cap, (int, float)) and float(eodhd_cap) > 0:
            return CapClassification(
                ticker=ticker_norm,
                as_of_date=asof_norm,
                market_cap_mm=float(eodhd_cap),
                market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
                cap_source=CAP_SOURCE_EODHD_FUNDAMENTALS,
                cap_band=band_for_market_cap(float(eodhd_cap)),
                cap_method="direct_issuer_market_cap",
                detail=eodhd_detail,
                **_identity_fields(identity),
            )

    # Tier 2: direct SEC/exchange evidence. These rows are issuer-level and
    # therefore do not need an ADR/class quote conversion ratio.
    direct = terminal_for(CAP_SOURCE_KIND_SEC, CAP_SOURCE_KIND_EXCHANGE)
    if direct is not None:
        return _terminal_classification(
            direct,
            ticker=ticker_norm,
            requested_as_of_date=asof_norm,
            identity=identity,
            fallback_price=terminal_price(),
        )

    # Tier 3: last-known companyfacts shares x as-of/current price. This stays
    # ahead of provider/search direct-cap fallbacks because SEC evidence is the
    # preferred source when its quote basis is safe.
    (
        shares_mm,
        shares_period_end,
        shares_source_url,
        shares_filed_date,
        raw_shares_source_value,
        raw_shares_source_unit,
    ) = (
        strict_v2_shares
        if normalized_pipeline == "v2"
        else _last_known_companyfacts_shares(
            ticker_norm,
            as_of_date=asof_norm,
            db_path=db_path,
            require_filed_asof=True,
            identity=identity,
        )
    )
    stale_detail: str | None = None
    if shares_mm is not None and shares_period_end is not None:
        try:
            asof_parsed = date.fromisoformat(asof_norm[:10])
            period_parsed = date.fromisoformat(str(shares_period_end)[:10])
        except ValueError:
            asof_parsed = period_parsed = None
        if (
            asof_parsed is not None
            and period_parsed is not None
            and asof_parsed - period_parsed > timedelta(days=STALE_SHARES_MAX_AGE_DAYS)
        ):
            stale_detail = f"stale_shares_too_old:{shares_period_end}"

    price_record = _price_from_value(
        current_price,
        default_as_of_date=asof_norm,
        default_source="caller_current_price",
        default_confidence=None,
    )
    currency_detail = cap_currency_reason(price_record) or cap_currency_reason(asof_price_record)
    if not cap_arithmetic_currency_ok(price_record):
        price_record = None
    price_origin = "caller"
    if (
        price_record is None
        and shares_mm is not None
        and stale_detail is None
        and issuer_quote_ratio is not None
    ):
        lookup = price_lookup or default_price_lookup(cfg)
        looked_up_price = _price_from_value(
            lookup(ticker_norm, asof_norm),
            default_as_of_date=asof_norm,
            default_source="provider",
            default_confidence="MEDIUM",
        )
        currency_detail = cap_currency_reason(looked_up_price) or currency_detail
        if cap_arithmetic_currency_ok(looked_up_price):
            price_record = looked_up_price
            price_origin = "provider"
    if price_record is None and cap_arithmetic_currency_ok(asof_price_record):
        price_record = asof_price_record
        price_origin = "asof_price_fallback"

    normalized_stale_share_lineage = normalized_shares(
        price_record,
        raw_shares_mm=shares_mm,
        shares_as_of_date=shares_period_end,
    )
    if (
        shares_mm is not None
        and stale_detail is None
        and price_record is not None
        and normalized_stale_share_lineage is not None
        and issuer_quote_ratio is not None
    ):
        normalized_shares_mm, normalized_shares_basis = normalized_stale_share_lineage
        market_cap = float(price_record.price) * normalized_shares_mm / float(issuer_quote_ratio)
        ratio_detail = (
            f":quote_to_issuer_share_ratio={issuer_quote_ratio}"
            if issuer_quote_ratio != 1.0
            else ""
        )
        return CapClassification(
            ticker=ticker_norm,
            as_of_date=asof_norm,
            market_cap_mm=market_cap,
            market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
            cap_source=CAP_SOURCE_STALE_SHARES,
            cap_band=band_for_market_cap(market_cap),
            price_used=float(price_record.price),
            price_as_of_date=price_record.as_of_date,
            price_source=price_record.source,
            price_source_url=price_record.source_url,
            price_currency=price_record.currency,
            price_confidence=price_record.confidence,
            shares_mm=normalized_shares_mm,
            shares_period_end=shares_period_end,
            shares_filed_date=shares_filed_date,
            shares_source="sec_companyfacts",
            shares_source_url=shares_source_url or companyfacts_url,
            cap_effective_as_of_date=price_record.as_of_date,
            cap_source_kind=CAP_SOURCE_KIND_SEC,
            cap_source_name="sec_companyfacts_stale_shares_derived",
            cap_source_url=shares_source_url or companyfacts_url,
            cap_confidence="MEDIUM" if price_record.confidence != "LOW" else "LOW",
            cap_method="price_times_shares_divided_by_issuer_quote_ratio",
            issuer_quote_ratio=float(issuer_quote_ratio),
            detail=(
                f"shares_period_end={shares_period_end}:price_origin={price_origin}{ratio_detail}"
            ),
            **_classification_quote_fields(
                ticker_norm,
                price_record,
                shares_mm=normalized_shares_mm,
                raw_shares_mm=float(shares_mm),
                raw_shares_source_value=raw_shares_source_value,
                raw_shares_source_unit=raw_shares_source_unit,
                shares_basis=normalized_shares_basis,
            ),
            **_identity_fields(identity),
        )

    # Tier 4: EODHD direct fundamentals — inert/config-flagged by default.
    if (
        normalized_pipeline == "v2"
        and bool(getattr(cfg, "cap_eodhd_fundamentals_enabled", False))
        and asof_norm == date.today().isoformat()
    ):
        eodhd_cap, eodhd_detail = _eodhd_fundamentals_market_cap_mm(ticker_norm, cfg)
        if isinstance(eodhd_cap, (int, float)) and float(eodhd_cap) > 0:
            direct_price = terminal_price()
            return CapClassification(
                ticker=ticker_norm,
                as_of_date=asof_norm,
                market_cap_mm=float(eodhd_cap),
                market_cap_unit=MARKET_CAP_UNIT_USD_MILLIONS,
                cap_source=CAP_SOURCE_EODHD_FUNDAMENTALS,
                cap_band=band_for_market_cap(float(eodhd_cap)),
                price_used=direct_price.price if direct_price else None,
                price_as_of_date=direct_price.as_of_date if direct_price else None,
                price_source=direct_price.source if direct_price else None,
                price_source_url=direct_price.source_url if direct_price else None,
                price_currency=direct_price.currency if direct_price else None,
                price_confidence=direct_price.confidence if direct_price else None,
                cap_effective_as_of_date=asof_norm,
                cap_source_kind=CAP_SOURCE_KIND_PROVIDER,
                cap_source_name="eodhd_fundamentals",
                cap_source_url=(
                    f"{cfg.eodhd_base_url}/api/fundamentals/{ticker_norm}.{cfg.eodhd_exchange}"
                ),
                cap_confidence="MEDIUM",
                cap_method="direct_issuer_market_cap",
                detail=eodhd_detail,
                **_classification_quote_fields(ticker_norm, direct_price),
                **_identity_fields(identity),
            )

    # Invoke an injected provider/search resolver only after every local/SEC
    # path and the configured fundamentals provider have missed. This keeps a
    # search-backed last resort from spending before cheaper evidence is
    # exhausted.
    if (
        normalized_pipeline == "v2"
        and terminal_cap_lookup is not None
        and prevalidated_ledger_lookup is None
    ):
        terminal_evidence.extend(
            _lookup_terminal_cap_evidence(
                ticker_norm,
                as_of_date=asof_norm,
                identity=identity,
                db_path=db_path,
                terminal_cap_lookup=terminal_cap_lookup,
                include_persisted=False,
            )
        )
        terminal_evidence.sort(
            key=lambda item: (
                CAP_SOURCE_KIND_PRIORITY[item.source_kind],
                -date.fromisoformat(item.as_of_date).toordinal(),
                item.source_name,
            )
        )

    # Tier 5: accept the best direct issuer cap returned by the injected
    # terminal resolver. The callback is intentionally invoked late because
    # it may be provider/search backed; nevertheless a validated ledger row
    # can itself identify an SEC or exchange source and must not be skipped
    # merely because the early persisted-only check has already run.
    direct = terminal_for(
        CAP_SOURCE_KIND_SEC,
        CAP_SOURCE_KIND_EXCHANGE,
        CAP_SOURCE_KIND_PROVIDER,
    )
    if direct is not None:
        return _terminal_classification(
            direct,
            ticker=ticker_norm,
            requested_as_of_date=asof_norm,
            identity=identity,
            fallback_price=terminal_price(),
        )

    # Tier 6 / last resort: search-backed direct issuer market cap. URL,
    # as-of, and confidence are mandatory at validation above.
    direct = terminal_for(CAP_SOURCE_KIND_SEARCH)
    if direct is not None:
        return _terminal_classification(
            direct,
            ticker=ticker_norm,
            requested_as_of_date=asof_norm,
            identity=identity,
            fallback_price=terminal_price(),
        )

    if unsafe_identity_reason:
        detail = f"{unsafe_identity_reason}:terminal_direct_cap_evidence_unavailable"
    elif stale_detail:
        detail = stale_detail
        if identity.is_adr is True or identity.is_secondary_class is True:
            detail += ":terminal_direct_cap_evidence_unavailable"
    elif shares_mm is not None and currency_detail:
        detail = currency_detail
    elif shares_mm is not None:
        detail = "shares_known_price_missing"
    else:
        detail = "no_companyfacts_shares"

    # Terminal unknown — never presentable as in-band output.
    return CapClassification(
        ticker=ticker_norm,
        as_of_date=asof_norm,
        market_cap_mm=None,
        market_cap_unit=None,
        cap_source=CAP_SOURCE_UNKNOWN,
        cap_band=None,
        price_used=price_record.price if price_record is not None else None,
        price_as_of_date=price_record.as_of_date if price_record is not None else None,
        price_source=price_record.source if price_record is not None else None,
        price_source_url=price_record.source_url if price_record is not None else None,
        price_currency=price_record.currency if price_record is not None else None,
        price_confidence=price_record.confidence if price_record is not None else None,
        shares_mm=float(shares_mm) if shares_mm is not None else None,
        shares_period_end=shares_period_end,
        shares_filed_date=shares_filed_date,
        shares_source=("sec_companyfacts" if shares_mm is not None else None),
        shares_source_url=shares_source_url or companyfacts_url if shares_mm is not None else None,
        cap_effective_as_of_date=None,
        cap_source_kind=None,
        cap_source_name=None,
        cap_source_url=None,
        cap_confidence=None,
        scope_status=("NEEDS_DATA" if unsafe_identity_reason else "IN_SCOPE"),
        scope_reason=unsafe_identity_reason,
        detail=detail,
        **_classification_quote_fields(
            ticker_norm,
            price_record,
            shares_mm=float(shares_mm) if shares_mm is not None else None,
            raw_shares_mm=float(shares_mm) if shares_mm is not None else None,
            raw_shares_source_value=raw_shares_source_value,
            raw_shares_source_unit=raw_shares_source_unit,
        ),
        **_identity_fields(identity),
    )


__all__ = [
    "CAP_SOURCE_ASOF_COMPANYFACTS",
    "CAP_SOURCE_EODHD_FUNDAMENTALS",
    "CAP_SOURCE_STALE_SHARES",
    "CAP_SOURCE_TERMINAL_EXCHANGE",
    "CAP_SOURCE_TERMINAL_LOCAL",
    "CAP_SOURCE_TERMINAL_PROVIDER",
    "CAP_SOURCE_TERMINAL_SEARCH",
    "CAP_SOURCE_TERMINAL_SEC",
    "CAP_SOURCE_UNKNOWN",
    "LOCAL_AUTHORITY_RECORD_PROVENANCE",
    "LOCAL_AUTHORITY_SCHEMA_V1",
    "SECURITY_ROLE_ADR",
    "SECURITY_ROLE_PRIMARY",
    "SECURITY_ROLE_SECONDARY_CLASS",
    "SECURITY_ROLE_SECONDARY_SECURITY",
    "SECURITY_ROLE_UNKNOWN",
    "UNKNOWN_CAP_LABEL",
    "CapClassification",
    "SecurityIdentity",
    "TerminalCapEvidence",
    "band_for_market_cap",
    "classify_market_cap_for_band_filter",
    "default_price_lookup",
    "derive_terminal_source_kind",
    "direct_market_cap_mm_from_payload",
    "resolve_security_identity",
]
