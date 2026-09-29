"""Candidate sourcing for autonomous sector financial runs."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.db import connect as db_connect
from app.config import canonical_market_cap_focus, get_config
from app.util.financial_data_access import companyfacts_rows


# THE canonical cap-band definition. Bands are lower-inclusive /
# upper-exclusive, contiguous and non-overlapping; smid_cap is the
# small+mid composite. Historical cap_category rows are NOT retro-relabeled
# — new rows and future backtest strata follow this scheme.
MARKET_CAP_FOCUS_TIERS: dict[str, tuple[float | None, float | None]] = {
    "micro_cap": (0.0, 500.0),
    "micro": (0.0, 500.0),
    "small_cap": (500.0, 2_500.0),
    "small": (500.0, 2_500.0),
    "smid_cap": (500.0, 10_000.0),
    "mid_cap": (2_500.0, 10_000.0),
    "mid": (2_500.0, 10_000.0),
    "large_cap": (10_000.0, 200_000.0),
    "mega_cap": (200_000.0, None),
    "large_and_mega": (10_000.0, None),
    "all": (None, None),
    "any": (None, None),
}

SECURITY_TYPE_NON_COMMON_EQUITY = "SECURITY_TYPE_NON_COMMON_EQUITY"

_KNOWN_NON_COMMON_EQUITY_TICKERS = {
    "ACGLN",
    "ACGLO",
    "AQNB",
    "BEPH",
    "BEPI",
    "BEPJ",
    "CMSA",
    "CMSC",
    "CMSD",
    "DTB",
    "DTG",
    "DTK",
    "DTW",
    "PPLC",
    "SOCGM",
    "SOCGP",
    "SOJC",
    "SOJD",
    "SOJE",
    "SOJF",
    "SOMN",
    "WELPM",
    "WELPP",
}
_EXPLICIT_NON_COMMON_SUFFIX_RE = re.compile(
    r"[-.](?:P[A-Z0-9]?|PR[A-Z0-9]?|WT|WS|W|RT|R|UN|U)$",
    re.IGNORECASE,
)
_DASHED_SERIES_WITHOUT_METADATA_RE = re.compile(r"^[A-Z]{1,5}-[A-Z]$", re.IGNORECASE)
_FINANCIAL_SERVICES_NON_BINDING_MODEL_BLOCKERS = {
    "SECURITY_TYPE_UNKNOWN",
    "INSURANCE_SUBTYPE_UNCLEAR",
    "NOT_INSURANCE_VALUATION_TARGET",
}


@dataclass(frozen=True)
class SecurityTypeFilterResult:
    ticker: str
    is_common_equity: bool
    reason: str | None = None
    detail: str | None = None


@dataclass
class SectorCandidateSelection:
    """Auditable ticker selection result for an autonomous sector run."""

    sector: str
    market_cap_focus: str
    selected_tickers: list[str]
    source: str
    requested_tickers: list[str] = field(default_factory=list)
    loaded_tickers: list[str] = field(default_factory=list)
    excluded_tickers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ranking_basis: str | None = None
    # Band-filter chain results keyed by ticker (cap_resolver.CapClassification
    # .to_dict()), covering examined rows INCLUDING out-of-band exclusions so
    # the audit trail shows why a name did or did not enter the run.
    cap_classifications: dict[str, Any] = field(default_factory=dict)
    structural_gate_results: dict[str, Any] = field(default_factory=dict)
    # V2 keeps the complete admitted membership distinct from the exact subset
    # authorized for paid execution. ``selected_tickers`` remains the complete
    # membership for stored-artifact compatibility.
    membership_tickers: list[str] = field(default_factory=list)
    execution_tickers: list[str] = field(default_factory=list)
    deferred_by_bound_tickers: list[str] = field(default_factory=list)
    execution_bound: int | None = None
    membership_fingerprint: str | None = None
    execution_fingerprint: str | None = None
    execution_bound_frozen: bool = False
    # The accepted-census bridge owns this shape.  The execution authority
    # treats it as an immutable canonical mapping and fingerprints it exactly.
    census_lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not (
            self.membership_tickers
            or self.execution_tickers
            or self.deferred_by_bound_tickers
            or self.execution_bound is not None
            or self.membership_fingerprint
            or self.execution_fingerprint
            or self.execution_bound_frozen
            or self.census_lineage
        ):
            for field_name in (
                "membership_tickers",
                "execution_tickers",
                "deferred_by_bound_tickers",
                "execution_bound",
                "membership_fingerprint",
                "execution_fingerprint",
                "execution_bound_frozen",
                "census_lineage",
            ):
                payload.pop(field_name, None)
        return payload


@dataclass(slots=True)
class AcceptedCensusRunAuthority:
    """Run-scoped cache for one fully validated, fixed-as-of census cohort."""

    _db_path: str | None = field(default=None, init=False, repr=False)
    _as_of_date: str | None = field(default=None, init=False, repr=False)
    _cohort: Any | None = field(default=None, init=False, repr=False)

    def load(self, *, db_path: str | Path, as_of_date: str) -> Any:
        from app.universe.accepted_census import load_accepted_census_cohort

        resolved_db_path = str(Path(db_path).resolve())
        if self._cohort is None:
            self._cohort = load_accepted_census_cohort(
                db_path=resolved_db_path,
                as_of_date=as_of_date,
            )
            self._db_path = resolved_db_path
            self._as_of_date = as_of_date
        elif self._db_path != resolved_db_path or self._as_of_date != as_of_date:
            raise ValueError("accepted census run authority cannot change database or as-of date")
        return self._cohort


def _ticker_ledger_fingerprint(tickers: list[str]) -> str:
    return sha256(json.dumps(tickers, separators=(",", ":")).encode("utf-8")).hexdigest()


def freeze_v2_execution_bound(
    selection: SectorCandidateSelection,
    max_candidates: int | None,
) -> SectorCandidateSelection:
    """Freeze one exact paid-execution subset without shrinking membership."""

    if max_candidates is not None and (
        isinstance(max_candidates, bool) or int(max_candidates) <= 0
    ):
        raise ValueError("max_candidates must be a positive integer when provided")
    membership = _normalize_tickers(selection.membership_tickers or selection.selected_tickers)
    if selection.selected_tickers and _normalize_tickers(selection.selected_tickers) != membership:
        raise ValueError("v2 selected_tickers must equal the complete membership ledger")
    bound = int(max_candidates) if max_candidates is not None else None
    execution = membership[:bound] if bound is not None else list(membership)
    deferred = membership[len(execution) :]
    if set(execution) & set(deferred) or [*execution, *deferred] != membership:
        raise ValueError("v2 execution/deferred ledgers do not partition membership")
    if set(selection.excluded_tickers) & set(membership):
        raise ValueError("v2 excluded tickers cannot also be admitted members")
    selection.selected_tickers = list(membership)
    selection.membership_tickers = list(membership)
    selection.execution_tickers = list(execution)
    selection.deferred_by_bound_tickers = list(deferred)
    selection.execution_bound = bound
    selection.membership_fingerprint = _ticker_ledger_fingerprint(membership)
    selection.execution_fingerprint = _ticker_ledger_fingerprint(execution)
    selection.execution_bound_frozen = True
    return selection


def _load_required_accepted_census(
    db_path: str,
    as_of_date: str,
) -> Any:
    """Validate one production census without retaining process-global state."""

    from app.universe.accepted_census import load_accepted_census_cohort

    return load_accepted_census_cohort(db_path=db_path, as_of_date=as_of_date)


def _accepted_census_candidate_selection(
    *,
    sector: str,
    market_cap_focus: str,
    requested_tickers: list[str],
    max_candidates: int | None,
    db_path: str | Path | None,
    as_of_date: str,
    accepted_census_authority: AcceptedCensusRunAuthority | None,
) -> SectorCandidateSelection:
    """Build one exact sector ledger from the accepted registry-first cohort."""

    from app.sector.canonical_taxonomy import ACTIVE_SECTOR_LABELS

    source_sector = str(sector or "").strip().lower()
    if source_sector not in ACTIVE_SECTOR_LABELS:
        raise ValueError(
            f"v2 large_and_mega requires an exact active census sector label: {sector}"
        )
    resolved_db_path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    cohort = (
        accepted_census_authority.load(
            db_path=resolved_db_path,
            as_of_date=as_of_date,
        )
        if accepted_census_authority is not None
        else _load_required_accepted_census(
            str(resolved_db_path.resolve()),
            as_of_date,
        )
    )
    sector_members = cohort.members_for_sector(source_sector)
    members_by_ticker = {member.ticker: member for member in sector_members}
    normalized_requested = _normalize_tickers(requested_tickers)
    outside_census = [ticker for ticker in normalized_requested if ticker not in members_by_ticker]
    if outside_census:
        raise ValueError(
            "requested v2 tickers are outside the accepted census sector cohort: "
            + ",".join(outside_census)
        )
    full_membership = [member.ticker for member in sector_members]
    if normalized_requested and set(normalized_requested) != set(full_membership):
        raise ValueError(
            "requested v2 tickers cannot narrow the complete accepted census "
            f"sector cohort; use max_candidates for a frozen bound: {source_sector}"
        )
    members = sector_members
    membership = [member.ticker for member in members]
    selection = SectorCandidateSelection(
        sector=source_sector,
        market_cap_focus=market_cap_focus,
        selected_tickers=membership,
        source="accepted_census",
        requested_tickers=normalized_requested,
        loaded_tickers=membership,
        excluded_tickers=[],
        warnings=["V2_PRE_RANK_AND_GATES_DEFERRED_UNTIL_POST_REPAIR"],
        ranking_basis="accepted_census_primary_ticker_order_pending_repaired_rank",
        cap_classifications={
            member.ticker: member.to_cap_classification(lineage=cohort.lineage).to_dict()
            for member in members
        },
        structural_gate_results={},
        census_lineage=cohort.lineage.to_dict(),
    )
    return freeze_v2_execution_bound(selection, max_candidates)


def _normalize_tickers(tickers: list[str] | tuple[str, ...] | None) -> list[str]:
    if not tickers:
        return []
    return list(
        dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    )


@lru_cache(maxsize=1)
def _cached_ticker_cik_map_for_security_filter() -> dict[str, str]:
    try:
        from app.universe.ticker_cik_map import load_ticker_cik_map

        return {
            str(ticker).strip().upper(): str(cik)
            for ticker, cik in load_ticker_cik_map(refresh_if_missing=False).items()
            if str(ticker).strip() and str(cik).strip()
        }
    except Exception:
        return {}


def _ticker_cik_for_security_filter(ticker: str) -> str | None:
    return _cached_ticker_cik_map_for_security_filter().get(str(ticker).strip().upper())


def _submission_profile_for_security_filter(cik: str | int | None) -> dict[str, Any]:
    try:
        from app.insurance.sources import cached_submission_profile

        return cached_submission_profile(cik)
    except Exception:
        return {}


def _explicit_non_common_suffix(ticker: str) -> bool:
    return bool(_EXPLICIT_NON_COMMON_SUFFIX_RE.search(str(ticker).strip().upper()))


def _dashed_series_without_metadata(ticker: str, cik: str | None) -> bool:
    if cik:
        return False
    return bool(_DASHED_SERIES_WITHOUT_METADATA_RE.search(str(ticker).strip().upper()))


def _looks_like_same_issuer_exchange_traded_debt(
    *,
    ticker: str,
    primary_ticker: str | None,
) -> bool:
    """Catch compact same-issuer baby-bond series without filtering share classes."""
    ticker_upper = str(ticker).strip().upper()
    primary_upper = str(primary_ticker or "").strip().upper()
    if not ticker_upper or not primary_upper or ticker_upper == primary_upper:
        return False
    if len(primary_upper) != 3:
        return False
    if len(ticker_upper) == 3 and ticker_upper[:2] == primary_upper[:2]:
        return True
    if len(ticker_upper) == 4 and ticker_upper.startswith(primary_upper):
        return True
    return False


def _security_filter_for_ticker(ticker: str) -> SecurityTypeFilterResult:
    """Return whether a ticker is eligible for common-equity sector analysis.

    The local SEC company_tickers cache has CIK/title, not a reliable security-type
    column, so this guard favors deterministic metadata where available and uses
    a small conservative pattern fallback for obvious preferred/debt/unit symbols.
    """
    ticker_upper = str(ticker).strip().upper()
    if not ticker_upper:
        return SecurityTypeFilterResult(
            ticker=ticker_upper, is_common_equity=False, reason=SECURITY_TYPE_NON_COMMON_EQUITY
        )

    if ticker_upper in _KNOWN_NON_COMMON_EQUITY_TICKERS:
        return SecurityTypeFilterResult(
            ticker=ticker_upper,
            is_common_equity=False,
            reason=SECURITY_TYPE_NON_COMMON_EQUITY,
            detail="known_non_common_equity_series",
        )

    if _explicit_non_common_suffix(ticker_upper):
        return SecurityTypeFilterResult(
            ticker=ticker_upper,
            is_common_equity=False,
            reason=SECURITY_TYPE_NON_COMMON_EQUITY,
            detail="ticker_suffix_non_common_equity",
        )

    cik = _ticker_cik_for_security_filter(ticker_upper)
    submission_profile = _submission_profile_for_security_filter(cik)
    primary_ticker = (
        str(submission_profile.get("issuer_primary_ticker") or "").strip().upper() or None
    )
    listed_tickers = [
        str(value).strip().upper()
        for value in (submission_profile.get("issuer_listed_tickers") or [])
        if str(value).strip()
    ]
    if listed_tickers and primary_ticker and ticker_upper in listed_tickers:
        if _looks_like_same_issuer_exchange_traded_debt(
            ticker=ticker_upper, primary_ticker=primary_ticker
        ):
            return SecurityTypeFilterResult(
                ticker=ticker_upper,
                is_common_equity=False,
                reason=SECURITY_TYPE_NON_COMMON_EQUITY,
                detail=f"non_primary_same_issuer_series:{primary_ticker}",
            )
        return SecurityTypeFilterResult(ticker=ticker_upper, is_common_equity=True)

    if _dashed_series_without_metadata(ticker_upper, cik):
        return SecurityTypeFilterResult(
            ticker=ticker_upper,
            is_common_equity=False,
            reason=SECURITY_TYPE_NON_COMMON_EQUITY,
            detail="dashed_series_without_sec_metadata",
        )

    return SecurityTypeFilterResult(ticker=ticker_upper, is_common_equity=True)


def _filter_common_equity_tickers(tickers: list[str]) -> tuple[list[str], list[str], list[str]]:
    common_tickers: list[str] = []
    excluded_tickers: list[str] = []
    details: list[str] = []
    for ticker in tickers:
        result = _security_filter_for_ticker(ticker)
        if result.is_common_equity:
            common_tickers.append(ticker)
        else:
            excluded_tickers.append(ticker)
            if result.detail:
                details.append(f"{ticker}:{result.detail}")
            else:
                details.append(ticker)
    warnings = []
    if excluded_tickers:
        warnings.append(
            f"{SECURITY_TYPE_NON_COMMON_EQUITY}:{len(excluded_tickers)}:{','.join(details)}"
        )
    return common_tickers, excluded_tickers, warnings


def _append_missing_loaded_tickers(
    ranked_tickers: list[str], loaded_tickers: list[str]
) -> tuple[list[str], list[str]]:
    ranked_set = set(ranked_tickers)
    missing = [ticker for ticker in loaded_tickers if ticker not in ranked_set]
    if not missing:
        return ranked_tickers, []
    return [*ranked_tickers, *missing], missing


def _cap_bounds(market_cap_focus: str) -> tuple[float | None, float | None, str | None]:
    key = canonical_market_cap_focus(market_cap_focus)
    if key in MARKET_CAP_FOCUS_TIERS:
        cap_min, cap_max = MARKET_CAP_FOCUS_TIERS[key]
        return cap_min, cap_max, None
    return None, None, f"UNKNOWN_MARKET_CAP_FOCUS:{market_cap_focus}"


def _v2_cap_scope(
    tickers: list[str],
    *,
    market_cap_focus: str,
    as_of_date: str,
    db_path: str | Path | None,
    terminal_cap_lookup: Any | None,
    allow_live_market_data: bool,
) -> tuple[list[str], list[str], dict[str, Any], list[str]]:
    """Classify every discovered v2 security, then apply resolved band bounds.

    Unknown cap is a repair/data state rather than an exclusion, so it remains
    eligible. Resolved out-of-band securities are retained in the returned
    classification map for terminal OUT_OF_SCOPE dispositions.
    """

    from app.sector.scan import classify_tickers_for_market_cap

    cap_min, cap_max, cap_warning = _cap_bounds(market_cap_focus)
    classifications = classify_tickers_for_market_cap(
        tickers=tickers,
        as_of_date=as_of_date,
        db_path=db_path,
        pipeline_version="v2",
        terminal_cap_lookup=terminal_cap_lookup,
        allow_live_market_data=allow_live_market_data,
    )
    eligible: list[str] = []
    out_of_band: list[str] = []
    unknown: list[str] = []
    for ticker in tickers:
        classification = classifications[ticker]
        in_band = classification.in_band(cap_min, cap_max)
        if in_band is False:
            out_of_band.append(ticker)
        else:
            eligible.append(ticker)
            if in_band is None:
                unknown.append(ticker)
    warnings = [cap_warning] if cap_warning else []
    if out_of_band:
        warnings.append(
            f"CAP_RESOLVED_OUT_OF_BAND:{len(out_of_band)}:" + ",".join(out_of_band[:20])
        )
    if unknown:
        warnings.append(f"UNKNOWN_CAP_INCLUDED:{len(unknown)}")
    return (
        eligible,
        out_of_band,
        {ticker: classification.to_dict() for ticker, classification in classifications.items()},
        warnings,
    )


def _is_financial_services_sector(sector: str | None) -> bool:
    normalized = str(sector or "").strip().lower().replace("-", "_")
    return normalized in {
        "financial_services",
        "large_cap_financials",
        "capital_markets",
        "insurance",
    } or any(token in normalized for token in ("bank", "financial", "lender", "credit"))


def _is_capital_markets_sector(sector: str | None) -> bool:
    return str(sector or "").strip().lower().replace("-", "_") == "capital_markets"


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _cached_fy_line_item_values(
    ticker: str,
    *,
    line_items: tuple[str, ...],
    as_of_date: str,
    db_path: str | Path | None = None,
) -> dict[int, dict[str, float]]:
    path = Path(db_path) if db_path is not None else Path(get_config().db_path)
    if not path.exists():
        return {}
    try:
        with db_connect(path) as conn:
            required = {
                "ticker",
                "fiscal_year",
                "period_type",
                "period_end",
                "filed_date",
                "line_item",
                "value",
            }
            if not required <= _table_columns(conn, "companyfacts_facts"):
                return {}
            rows = companyfacts_rows(
                conn,
                ticker.upper(),
                columns=(
                    "fiscal_year",
                    "line_item",
                    "value",
                    "period_end",
                    "filed_date",
                ),
                period_types=("FY",),
                line_items=line_items,
                as_of_date=as_of_date,
                require_filed_asof=True,
                value_not_null=True,
                order_by="fiscal_year DESC",
            )
    except sqlite3.Error:
        return {}
    by_year: dict[int, dict[str, float]] = {}
    for fiscal_year, line_item, value, period_end, filed_date in rows:
        if not isinstance(fiscal_year, int):
            continue
        if str(period_end or "") > str(filed_date or ""):
            continue
        if isinstance(value, (int, float)) and value > 0:
            by_year.setdefault(fiscal_year, {})[str(line_item)] = float(value)
    return by_year


def _cached_book_value_years(
    ticker: str,
    *,
    as_of_date: str,
    db_path: str | Path | None = None,
) -> int:
    by_year = _cached_fy_line_item_values(
        ticker,
        line_items=("equity", "shares_outstanding"),
        as_of_date=as_of_date,
        db_path=db_path,
    )
    return len(
        [
            fiscal_year
            for fiscal_year, values in by_year.items()
            if values.get("equity", 0.0) > 0 and values.get("shares_outstanding", 0.0) > 0
        ]
    )


def _cached_capital_markets_operating_years(
    ticker: str,
    *,
    as_of_date: str,
    db_path: str | Path | None = None,
) -> int:
    by_year = _cached_fy_line_item_values(
        ticker,
        line_items=("revenue", "net_income", "operating_income", "total_assets", "equity"),
        as_of_date=as_of_date,
        db_path=db_path,
    )
    return len(
        [
            fiscal_year
            for fiscal_year, values in by_year.items()
            if values.get("net_income", 0.0) > 0
            and (values.get("revenue", 0.0) > 0 or values.get("operating_income", 0.0) > 0)
            and (values.get("total_assets", 0.0) > 0 or values.get("equity", 0.0) > 0)
        ]
    )


def _financial_services_entry_has_sector_shape(
    entry: Any,
    *,
    sector: str | None = None,
    as_of_date: str,
    db_path: str | Path | None = None,
) -> bool:
    packet = getattr(entry, "packet", None)
    ticker = str(getattr(entry, "ticker", "") or getattr(packet, "ticker", "") or "").upper()
    if not ticker:
        return False
    if packet is None:
        if (
            _cached_book_value_years(
                ticker,
                as_of_date=as_of_date,
                db_path=db_path,
            )
            >= 3
        ):
            return True
        return (
            _is_capital_markets_sector(sector)
            and _cached_capital_markets_operating_years(
                ticker,
                as_of_date=as_of_date,
                db_path=db_path,
            )
            >= 3
        )
    model_blockers = {
        str(item) for item in (getattr(packet, "model_blockers", None) or []) if str(item)
    }
    model_status = str(getattr(packet, "model_status", "") or "").upper()
    if (
        model_status == "MODEL_BLOCKED"
        and not model_blockers <= _FINANCIAL_SERVICES_NON_BINDING_MODEL_BLOCKERS
    ):
        return False
    if model_blockers and not model_blockers <= _FINANCIAL_SERVICES_NON_BINDING_MODEL_BLOCKERS:
        return False
    if (
        _cached_book_value_years(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
        )
        >= 3
    ):
        return True
    if (
        _is_capital_markets_sector(sector)
        and _cached_capital_markets_operating_years(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
        )
        >= 3
    ):
        return True
    return False


def _entry_is_selectable(
    entry: Any,
    *,
    sector: str | None = None,
    as_of_date: str,
    db_path: str | Path | None = None,
) -> bool:
    if _is_financial_services_sector(sector) and _financial_services_entry_has_sector_shape(
        entry,
        sector=sector,
        as_of_date=as_of_date,
        db_path=db_path,
    ):
        return True

    methods_with_value = getattr(entry, "methods_with_value", None)
    if isinstance(methods_with_value, int) and methods_with_value <= 0:
        return False

    packet = getattr(entry, "packet", None)
    if packet is None:
        return True

    if str(getattr(packet, "model_status", "") or "").upper() == "MODEL_BLOCKED":
        return False
    if getattr(packet, "model_blockers", None):
        return False
    if str(getattr(packet, "gate_verdict", "") or "").upper() == "BLOCK":
        return False
    if not isinstance(getattr(packet, "current_price", None), (int, float)):
        return False

    value_fields = [
        "insurance_value",
        "dcf_value",
        "epv_value",
        "graham_value",
        "ncav_value",
    ]
    return any(isinstance(getattr(packet, field, None), (int, float)) for field in value_fields)


def _apply_structural_gate(
    tickers: list[str],
    *,
    as_of_date: str,
    cap_classifications: dict[str, Any] | None,
    db_path: str | Path | None,
    warnings: list[str],
    exclude: bool,
    include_evidence: bool = False,
) -> tuple[list[str], list[str]] | tuple[list[str], list[str], dict[str, Any]]:
    """Run the structural exclusion gate over candidate tickers pre-LLM.

    Quarantined names are removed from the run (saving LLM spend) when
    ``exclude`` is True; on the operator-controlled explicit path they are only
    surfaced as warnings. Every trigger lands in the audit trail as
    ``QUARANTINE_STRUCTURAL:<code>:<ticker>``.

    Fail-closed: a name the gate could not examine (engine DB
    missing/locked, gate crash) is EXCLUDED_ERROR — excluded from the run
    with the failure in the audit trail — never passed through as if it had
    been examined and cleared.
    """
    from app.autonomous.structural_gate import (
        EXCLUDED_ERROR_PREFIX,
        evaluate_structural_gate,
    )

    classifications = cap_classifications or {}
    kept: list[str] = []
    gated: list[str] = []
    gate_results: dict[str, Any] = {}
    for ticker in tickers:
        classification = classifications.get(ticker) or {}
        try:
            result = evaluate_structural_gate(
                ticker,
                as_of_date=as_of_date,
                price=classification.get("price_used"),
                market_cap_mm=classification.get("market_cap_mm"),
                db_path=db_path,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed, never fail open
            warnings.append(f"{EXCLUDED_ERROR_PREFIX}:GATE_EVAL_FAILED:{ticker}:{exc}")
            gate_results[ticker] = {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "quarantined": False,
                "excluded_error": True,
                "triggered_codes": [],
                "degraded_codes": ["GATE_EVAL_FAILED"],
                "reasons": [f"{EXCLUDED_ERROR_PREFIX}:GATE_EVAL_FAILED"],
                "details": {"error": str(exc)},
            }
            if exclude:
                gated.append(ticker)
            else:
                kept.append(ticker)
            continue
        if result.excluded_error:
            gate_results[ticker] = result.to_dict()
            warnings.append(f"{result.degraded_string}:{ticker}")
            if exclude:
                gated.append(ticker)
            else:
                kept.append(ticker)
            continue
        # Advisory flags (MIN_ADV) land in the audit trail but never
        # change selection — flags mark; they never reject.
        for advisory_code in result.advisory_codes:
            warnings.append(f"ADVISORY:{advisory_code}:{ticker}")
        if not result.quarantined:
            kept.append(ticker)
            continue
        gate_results[ticker] = result.to_dict()
        for reason in result.reasons:
            warnings.append(f"{reason}:{ticker}")
        if exclude:
            gated.append(ticker)
        else:
            kept.append(ticker)
    if include_evidence:
        return kept, gated, gate_results
    return kept, gated


def _pre_rank_kwargs(
    *,
    tickers: list[str],
    sector: str,
    filing_risk_use_llm: bool,
    as_of_date: str,
    db_path: str | Path | None,
) -> dict[str, Any]:
    # Candidate discovery is provider-free until canonical packet assembly
    # has produced a scope that the deterministic integrity gate can approve.
    _ = filing_risk_use_llm
    kwargs: dict[str, Any] = {
        "tickers": tickers,
        "limit": 0,
        "as_of_date": as_of_date,
        "db_path": db_path,
    }
    if _is_financial_services_sector(sector):
        kwargs["include_blocked"] = True
    return kwargs


def resolve_sector_candidate_tickers(
    *,
    sector: str,
    explicit_tickers: list[str] | tuple[str, ...] | None = None,
    candidate_pool_tickers: list[str] | tuple[str, ...] | None = None,
    market_cap_focus: str = "small_cap",
    max_candidates: int | None = None,
    db_path: str | Path | None = None,
    filing_risk_use_llm: bool = False,
    as_of_date: str | None = None,
    pipeline_version: str = "v1",
    terminal_cap_lookup: Any | None = None,
    allow_live_market_data: bool = False,
    coverage_only: bool = False,
    cap_classification_cache: dict[str, Any] | None = None,
    require_accepted_census: bool = False,
    accepted_census_authority: AcceptedCensusRunAuthority | None = None,
) -> SectorCandidateSelection:
    """Resolve candidate tickers for a sector autonomous run.

    Explicit tickers remain the most controlled path. When they are omitted,
    this resolver loads the existing sector scan membership from the local DB,
    applies the requested market-cap focus when possible, and uses the existing
    consensus pre-ranker to order the run. When max_candidates is omitted,
    every loaded ticker is retained for review.

    The structural exclusion gate (app/autonomous/structural_gate.py) fires
    here, pre-LLM: quarantined names are excluded from sweep-sourced runs
    (saving spend) and surfaced as QUARANTINE_STRUCTURAL:<code>:<ticker>
    warnings; on the operator-controlled explicit path they are warned but kept.
    ``as_of_date`` keys the gate point-in-time (default: today).
    """
    from datetime import date as _date

    gate_asof = str(as_of_date or "").strip() or _date.today().isoformat()
    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    if normalized_pipeline == "v2":
        filing_risk_use_llm = False
    normalized_explicit = _normalize_tickers(list(explicit_tickers or []))
    if max_candidates is not None and int(max_candidates) <= 0:
        raise ValueError("max_candidates must be a positive integer when provided")
    limit = int(max_candidates) if max_candidates is not None else None
    if coverage_only and limit is not None:
        raise ValueError("coverage_only candidate resolution must be unbounded")

    def finalize(selection: SectorCandidateSelection) -> SectorCandidateSelection:
        if normalized_pipeline == "v2":
            return freeze_v2_execution_bound(selection, limit)
        return selection

    census_is_authoritative = (
        normalized_pipeline == "v2"
        and canonical_market_cap_focus(market_cap_focus) == "large_and_mega"
    )
    if require_accepted_census and not census_is_authoritative:
        raise ValueError("accepted census authority is only valid for v2 large_and_mega")
    if accepted_census_authority is not None and not census_is_authoritative:
        raise ValueError("accepted census run authority is only valid for v2 large_and_mega")
    if census_is_authoritative:
        normalized_pool = _normalize_tickers(list(candidate_pool_tickers or []))
        requested = normalized_explicit or normalized_pool
        return _accepted_census_candidate_selection(
            sector=sector,
            market_cap_focus=market_cap_focus,
            requested_tickers=requested,
            max_candidates=limit,
            db_path=db_path,
            as_of_date=gate_asof,
            accepted_census_authority=accepted_census_authority,
        )

    if normalized_explicit:
        cap_classifications: dict[str, Any] = {}
        cap_excluded: list[str] = []
        security_warnings: list[str] = []
        cap_eligible = normalized_explicit
        if normalized_pipeline == "v2":
            (
                cap_eligible,
                cap_excluded,
                cap_classifications,
                cap_warnings,
            ) = _v2_cap_scope(
                normalized_explicit,
                market_cap_focus=market_cap_focus,
                as_of_date=gate_asof,
                db_path=db_path,
                terminal_cap_lookup=terminal_cap_lookup,
                allow_live_market_data=allow_live_market_data,
            )
            security_warnings.extend(cap_warnings)
        else:
            from app.sector.scan import load_sector_tickers_classified

            cap_min, cap_max, cap_warning = _cap_bounds(market_cap_focus)
            if cap_warning:
                security_warnings.append(cap_warning)
            load_kwargs: dict[str, Any] = dict(
                sector=sector,
                explicit_tickers=normalized_explicit,
                db_path=db_path,
                cap_min=cap_min,
                cap_max=cap_max,
                pipeline_version=normalized_pipeline,
                allow_live_market_data=allow_live_market_data,
                as_of_date=gate_asof,
            )
            if cap_classification_cache is not None:
                load_kwargs["cap_classification_cache"] = cap_classification_cache
            _loaded_rows, cap_classification_objs = load_sector_tickers_classified(**load_kwargs)
            cap_classifications = {
                ticker: classification.to_dict()
                for ticker, classification in cap_classification_objs.items()
            }
            resolved_out_of_band = sorted(
                ticker
                for ticker, classification in cap_classification_objs.items()
                if classification.in_band(cap_min, cap_max) is False
            )
            if resolved_out_of_band:
                security_warnings.append(
                    f"CAP_RESOLVED_OUT_OF_BAND:{len(resolved_out_of_band)}:"
                    + ",".join(resolved_out_of_band[:20])
                )
            unknown_cap_included = sorted(
                ticker
                for ticker, classification in cap_classification_objs.items()
                if classification.cap_source == "unknown"
            )
            if unknown_cap_included:
                security_warnings.append(f"UNKNOWN_CAP_INCLUDED:{len(unknown_cap_included)}")
        common_explicit, security_excluded, common_warnings = _filter_common_equity_tickers(
            cap_eligible
        )
        security_warnings.extend(common_warnings)
        if normalized_pipeline == "v2":
            # V2 discovery is an immutable scope ledger, not a review budget.
            # Ranking and evidence-backed gates run only after the resumable
            # repair sequence has produced fixed-as-of packets.
            selected = common_explicit
            structural_gate_results: dict[str, Any] = {}
            security_warnings.append("V2_PRE_RANK_AND_GATES_DEFERRED_UNTIL_POST_REPAIR")
            ranking_basis = "v2_discovery_input_order_pending_repaired_rank"
        else:
            selected = common_explicit[:limit] if limit is not None else common_explicit
            # Operator-controlled explicit path: surface structural quarantines, never exclude.
            selected, _, structural_gate_results = _apply_structural_gate(
                selected,
                as_of_date=gate_asof,
                cap_classifications=cap_classifications,
                db_path=db_path,
                warnings=security_warnings,
                exclude=False,
                include_evidence=True,
            )
            ranking_basis = "input_order"
        selected_set = set(selected)
        return finalize(
            SectorCandidateSelection(
                sector=sector,
                market_cap_focus=market_cap_focus,
                selected_tickers=selected,
                source="explicit_tickers",
                requested_tickers=normalized_explicit,
                loaded_tickers=common_explicit,
                excluded_tickers=list(
                    dict.fromkeys(
                        [
                            *security_excluded,
                            *cap_excluded,
                            *[ticker for ticker in common_explicit if ticker not in selected_set],
                        ]
                    )
                ),
                warnings=security_warnings,
                ranking_basis=ranking_basis,
                cap_classifications=cap_classifications,
                structural_gate_results=structural_gate_results,
            )
        )

    normalized_pool = _normalize_tickers(list(candidate_pool_tickers or []))
    if normalized_pool:
        requested_pool = list(normalized_pool)
        cap_classifications: dict[str, Any] = {}
        cap_excluded: list[str] = []
        warnings: list[str] = []
        if normalized_pipeline == "v2":
            (
                normalized_pool,
                cap_excluded,
                cap_classifications,
                cap_warnings,
            ) = _v2_cap_scope(
                normalized_pool,
                market_cap_focus=market_cap_focus,
                as_of_date=gate_asof,
                db_path=db_path,
                terminal_cap_lookup=terminal_cap_lookup,
                allow_live_market_data=allow_live_market_data,
            )
            warnings.extend(cap_warnings)
        normalized_pool, security_excluded, security_warnings = _filter_common_equity_tickers(
            normalized_pool
        )
        warnings.extend(security_warnings)
        if not normalized_pool:
            return finalize(
                SectorCandidateSelection(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    selected_tickers=[],
                    source="candidate_pool",
                    requested_tickers=requested_pool,
                    loaded_tickers=[],
                    excluded_tickers=list(dict.fromkeys([*security_excluded, *cap_excluded])),
                    warnings=warnings + ["NO_COMMON_EQUITY_CANDIDATES_FOUND"],
                    ranking_basis="none",
                    cap_classifications=cap_classifications,
                )
            )
        if normalized_pipeline == "v2":
            warnings.append("V2_PRE_RANK_AND_GATES_DEFERRED_UNTIL_POST_REPAIR")
            return finalize(
                SectorCandidateSelection(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    selected_tickers=normalized_pool,
                    source="candidate_pool",
                    requested_tickers=requested_pool,
                    loaded_tickers=normalized_pool,
                    excluded_tickers=list(dict.fromkeys([*security_excluded, *cap_excluded])),
                    warnings=warnings,
                    ranking_basis="v2_discovery_input_order_pending_repaired_rank",
                    cap_classifications=cap_classifications,
                    structural_gate_results={},
                )
            )
        try:
            from app.sector.scan import pre_rank_sector

            pre_rank_kwargs = _pre_rank_kwargs(
                tickers=normalized_pool,
                sector=sector,
                filing_risk_use_llm=filing_risk_use_llm,
                as_of_date=gate_asof,
                db_path=db_path,
            )
            ranked = pre_rank_sector(**pre_rank_kwargs)
            selectable_entries = [
                entry
                for entry in ranked
                if _entry_is_selectable(
                    entry,
                    sector=sector,
                    as_of_date=gate_asof,
                    db_path=db_path,
                )
            ]
            unusable_entries = [
                entry
                for entry in ranked
                if not _entry_is_selectable(
                    entry,
                    sector=sector,
                    as_of_date=gate_asof,
                    db_path=db_path,
                )
            ]
            if unusable_entries:
                if limit is None:
                    warnings.append(
                        f"UNUSABLE_CANDIDATES_INCLUDED_FOR_FULL_REVIEW:{len(unusable_entries)}"
                    )
                else:
                    warnings.append(f"FILTERED_UNUSABLE_CANDIDATES:{len(unusable_entries)}")
            if limit is None:
                ranked_tickers = _normalize_tickers([entry.ticker for entry in ranked])
                ranked_tickers, omitted_loaded = _append_missing_loaded_tickers(
                    ranked_tickers, normalized_pool
                )
                if omitted_loaded:
                    warnings.append(f"PRE_RANK_OMITTED_LOADED_CANDIDATES:{len(omitted_loaded)}")
                ranking_basis = "candidate_pool_consensus_pre_rank_all_loaded"
            elif selectable_entries:
                ranked_tickers = _normalize_tickers([entry.ticker for entry in selectable_entries])
                ranking_basis = "candidate_pool_consensus_pre_rank_selectable"
            else:
                ranked_tickers = _normalize_tickers([entry.ticker for entry in unusable_entries])
                warnings.append("NO_SELECTABLE_CANDIDATES_FOUND")
                ranking_basis = "candidate_pool_consensus_pre_rank_unusable_fallback"
        except Exception as exc:
            warnings.append(f"PRE_RANK_FAILED:{exc}")
            ranked_tickers = normalized_pool
            ranking_basis = "candidate_pool_order"
        selected = ranked_tickers[:limit] if limit is not None else ranked_tickers
        selected, _gated, structural_gate_results = _apply_structural_gate(
            selected,
            as_of_date=gate_asof,
            cap_classifications=cap_classifications,
            db_path=db_path,
            warnings=warnings,
            exclude=True,
            include_evidence=True,
        )
        return finalize(
            SectorCandidateSelection(
                sector=sector,
                market_cap_focus=market_cap_focus,
                selected_tickers=selected,
                source="candidate_pool",
                requested_tickers=requested_pool,
                loaded_tickers=normalized_pool,
                excluded_tickers=list(
                    dict.fromkeys(
                        [
                            *security_excluded,
                            *cap_excluded,
                            *[ticker for ticker in normalized_pool if ticker not in set(selected)],
                        ]
                    )
                ),
                warnings=warnings,
                ranking_basis=ranking_basis,
                cap_classifications=cap_classifications,
                structural_gate_results=structural_gate_results,
            )
        )

    warnings: list[str] = []
    cap_min, cap_max, cap_warning = _cap_bounds(market_cap_focus)
    if cap_warning:
        warnings.append(cap_warning)

    try:
        from app.sector.scan import load_sector_tickers_classified, pre_rank_sector

        load_kwargs: dict[str, Any] = dict(
            sector=sector,
            db_path=db_path,
            cap_min=cap_min,
            cap_max=cap_max,
            pipeline_version=normalized_pipeline,
            allow_live_market_data=allow_live_market_data,
        )
        if cap_classification_cache is not None:
            load_kwargs["cap_classification_cache"] = cap_classification_cache
        load_kwargs["as_of_date"] = gate_asof
        if normalized_pipeline == "v2":
            load_kwargs.update(
                terminal_cap_lookup=terminal_cap_lookup,
            )
        loaded_rows, cap_classification_objs = load_sector_tickers_classified(**load_kwargs)
        cap_classifications = {
            ticker: classification.to_dict()
            for ticker, classification in cap_classification_objs.items()
        }
        # Keep every resolved out-of-band security visible for an explicit
        # OUT_OF_SCOPE disposition, regardless of which v2 evidence lane
        # resolved its issuer cap.
        registry_out_of_scope = sorted(
            ticker
            for ticker, classification in cap_classification_objs.items()
            if getattr(classification, "scope_status", "IN_SCOPE") == "OUT_OF_SCOPE"
        )
        resolved_out_of_band = sorted(
            ticker
            for ticker, classification in cap_classification_objs.items()
            if (
                getattr(classification, "scope_status", "IN_SCOPE") == "OUT_OF_SCOPE"
                or classification.in_band(cap_min, cap_max) is False
            )
        )
        if registry_out_of_scope:
            warnings.append(
                f"REGISTRY_OUT_OF_SCOPE:{len(registry_out_of_scope)}:"
                + ",".join(registry_out_of_scope[:20])
            )
        if resolved_out_of_band:
            warnings.append(
                f"CAP_RESOLVED_OUT_OF_BAND:{len(resolved_out_of_band)}:"
                + ",".join(resolved_out_of_band[:20])
            )
        unknown_cap_included = sorted(
            ticker
            for ticker, classification in cap_classification_objs.items()
            if classification.cap_source == "unknown"
        )
        if unknown_cap_included:
            warnings.append(f"UNKNOWN_CAP_INCLUDED:{len(unknown_cap_included)}")
        loaded_tickers = _normalize_tickers([ticker for ticker, _, _ in loaded_rows])
        loaded_tickers, security_excluded, security_warnings = _filter_common_equity_tickers(
            loaded_tickers
        )
        warnings.extend(security_warnings)
        if not loaded_tickers:
            return finalize(
                SectorCandidateSelection(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    selected_tickers=[],
                    source="sector_scan_db",
                    loaded_tickers=[],
                    excluded_tickers=list(
                        dict.fromkeys([*security_excluded, *resolved_out_of_band])
                    ),
                    warnings=warnings + ["NO_SECTOR_TICKERS_FOUND"],
                    ranking_basis="none",
                    cap_classifications=cap_classifications,
                )
            )

        if normalized_pipeline == "v2":
            warnings.append("V2_PRE_RANK_AND_GATES_DEFERRED_UNTIL_POST_REPAIR")
            return finalize(
                SectorCandidateSelection(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    selected_tickers=loaded_tickers,
                    source="sector_scan_db",
                    loaded_tickers=loaded_tickers,
                    excluded_tickers=list(
                        dict.fromkeys([*security_excluded, *resolved_out_of_band])
                    ),
                    warnings=warnings,
                    ranking_basis="v2_discovery_membership_order_pending_repaired_rank",
                    cap_classifications=cap_classifications,
                    structural_gate_results={},
                )
            )

        if coverage_only:
            return finalize(
                SectorCandidateSelection(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    selected_tickers=list(loaded_tickers),
                    source="sector_scan_db",
                    loaded_tickers=loaded_tickers,
                    excluded_tickers=list(
                        dict.fromkeys(
                            [
                                *security_excluded,
                                *resolved_out_of_band,
                            ]
                        )
                    ),
                    warnings=warnings,
                    ranking_basis="coverage_only_loaded_order_gate_deferred_to_delta",
                    cap_classifications=cap_classifications,
                    structural_gate_results={},
                )
            )

        try:
            pre_rank_kwargs = _pre_rank_kwargs(
                tickers=loaded_tickers,
                sector=sector,
                filing_risk_use_llm=filing_risk_use_llm,
                as_of_date=gate_asof,
                db_path=db_path,
            )
            ranked = pre_rank_sector(**pre_rank_kwargs)
            selectable_entries = [
                entry
                for entry in ranked
                if _entry_is_selectable(
                    entry,
                    sector=sector,
                    as_of_date=gate_asof,
                    db_path=db_path,
                )
            ]
            unusable_entries = [
                entry
                for entry in ranked
                if not _entry_is_selectable(
                    entry,
                    sector=sector,
                    as_of_date=gate_asof,
                    db_path=db_path,
                )
            ]
            if unusable_entries:
                if limit is None:
                    warnings.append(
                        f"UNUSABLE_CANDIDATES_INCLUDED_FOR_FULL_REVIEW:{len(unusable_entries)}"
                    )
                else:
                    warnings.append(f"FILTERED_UNUSABLE_CANDIDATES:{len(unusable_entries)}")
            if limit is None:
                ranked_entries = ranked
                ranking_basis = "consensus_pre_rank_all_loaded"
            elif selectable_entries:
                ranked_entries = selectable_entries
                if limit is not None and len(selectable_entries) < limit:
                    warnings.append(
                        f"SELECTABLE_CANDIDATES_BELOW_LIMIT:{len(selectable_entries)}/{limit}"
                    )
                ranking_basis = "consensus_pre_rank_selectable"
            else:
                ranked_entries = unusable_entries
                warnings.append("NO_SELECTABLE_CANDIDATES_FOUND")
                ranking_basis = "consensus_pre_rank_unusable_fallback"
            ranked_tickers = _normalize_tickers([entry.ticker for entry in ranked_entries])
            if limit is None:
                ranked_tickers, omitted_loaded = _append_missing_loaded_tickers(
                    ranked_tickers, loaded_tickers
                )
                if omitted_loaded:
                    warnings.append(f"PRE_RANK_OMITTED_LOADED_CANDIDATES:{len(omitted_loaded)}")
        except Exception as exc:
            warnings.append(f"PRE_RANK_FAILED:{exc}")
            ranked_tickers = loaded_tickers
            ranking_basis = "loaded_order"

        selected = ranked_tickers[:limit] if limit is not None else ranked_tickers
        selected, _gated, structural_gate_results = _apply_structural_gate(
            selected,
            as_of_date=gate_asof,
            cap_classifications=cap_classifications,
            db_path=db_path,
            warnings=warnings,
            exclude=True,
            include_evidence=True,
        )
        excluded = list(
            dict.fromkeys(
                [
                    *security_excluded,
                    *resolved_out_of_band,
                    *_gated,
                    *[ticker for ticker in loaded_tickers if ticker not in set(selected)],
                ]
            )
        )
        return finalize(
            SectorCandidateSelection(
                sector=sector,
                market_cap_focus=market_cap_focus,
                selected_tickers=selected,
                source="sector_scan_db",
                loaded_tickers=loaded_tickers,
                excluded_tickers=excluded,
                warnings=warnings,
                ranking_basis=ranking_basis,
                cap_classifications=cap_classifications,
                structural_gate_results=structural_gate_results,
            )
        )
    except Exception as exc:
        return finalize(
            SectorCandidateSelection(
                sector=sector,
                market_cap_focus=market_cap_focus,
                selected_tickers=[],
                source="sector_scan_db",
                warnings=warnings + [f"SECTOR_CANDIDATE_SOURCE_FAILED:{exc}"],
                ranking_basis="none",
            )
        )


__all__ = [
    "MARKET_CAP_FOCUS_TIERS",
    "SECURITY_TYPE_NON_COMMON_EQUITY",
    "SectorCandidateSelection",
    "freeze_v2_execution_bound",
    "resolve_sector_candidate_tickers",
]
