from __future__ import annotations

import json
import logging
import math
import statistics
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import get_db, utc_now_iso
from app.market.price_provider import get_default_provider as get_market_price_provider
from app.util.credential_hygiene import sanitize_json_value, sanitize_url_credentials
from app.valuation.anchor_policy import published_dcf_base
from app.valuation.measurement import valuations_table
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.evidenced_zero import resolve_evidenced_zero_facts
from app.valuation.net_debt import resolve_net_debt_proxy
from app.util.issuer_classification import (
    REASON_REIT_DEPRECIATION_DISTORTS_EARNINGS,
    lookup_is_reit,
)
from app.valuation.price_provider import PriceProvider
from app.valuation.expectations_gap import (
    compute_expectations_gap,
    estimate_supportable_growth,
)
from app.valuation.rnd_capitalization import compute_rnd_adjusted_earnings
from app.valuation.reverse_dcf import implied_growth_from_price
from app.valuation.lenses import ev_ebit_anchor, fcf_yield_anchor, tangible_floor
from app.valuation.lineage import (
    latest_decision_eligible_valuation_row,
    latest_decision_eligible_valuation_rows,
    register_valuation_writer_records,
    valuation_integrity_fingerprint,
    valuation_source_record,
    valuation_source_lineage,
)
from app.valuation.share_count_stability import select_stable_shares
from app.valuation.tech_category import TRADITIONAL_OPERATING, classify_company_category
from app.util.financial_data_access import issuer_companyfacts_rows

_VERSION = "1.1.0"
_TTL_SECONDS = 24 * 3600
_TAX_RATE = 0.21
_WACC = 0.10
_TERMINAL_GROWTH = 0.02
_CAPEX_SPIKE_MULTIPLIER = 2.0
_MIN_YEARS = 3
_MAX_TECH_DIVERGENCE_RATIO = 9.99
_MAX_ADJUSTMENT_IMPLAUSIBLE_RATIO = 10.0
_FLAG_ADJUSTMENT_IMPLAUSIBLE = "ADJUSTMENT_IMPLAUSIBLE"
_FLAG_VALUATION_DIVERGENCE_CAPPED = "VALUATION_DIVERGENCE_CAPPED"

logger = logging.getLogger(__name__)


# ── data loading ──────────────────────────────────────────────────────────────


def _load_facts(
    ticker: str,
    conn: Any,
    *,
    as_of_date: str | None = None,
    issuer_cik: str | None = None,
    issuer_aliases: tuple[str, ...] = (),
    require_filed_asof: bool = True,
) -> dict[str, list[tuple[int, float]]]:
    """
    Return all companyfacts_facts rows for ticker as:
      {line_item: [(fiscal_year, value), ...] sorted descending by fiscal_year}
    Values are in USD millions (or shares millions).

    When ``as_of_date`` is given, only FY rows whose ``period_end <= as_of_date``
    and whose filing was public by that date are returned. The unsafe
    period-end-only compatibility lane is opt-in; active production callers
    retain the filed-as-of default.
    """
    if issuer_cik or issuer_aliases or require_filed_asof:
        _, rows = issuer_companyfacts_rows(
            conn,
            ticker,
            columns=("line_item", "fiscal_year", "value"),
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
            period_types=("FY",),
            as_of_date=as_of_date,
            value_not_null=True,
            require_filed_asof=require_filed_asof,
            order_by="fiscal_year DESC, line_item ASC",
        )
    else:
        # V1 compatibility: preserve the historical ticker-exact,
        # period-end-only read unless issuer/PIT semantics are explicit.
        sql = (
            "SELECT line_item, fiscal_year, value FROM companyfacts_facts "
            "WHERE ticker = ? AND value IS NOT NULL AND period_type = 'FY'"
        )
        params: list[Any] = [ticker.upper()]
        if as_of_date:
            sql += " AND period_end <= ?"
            params.append(str(as_of_date))
        sql += " ORDER BY fiscal_year DESC"
        rows = conn.execute(sql, params).fetchall()
    out: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        li = str(row["line_item"])
        out.setdefault(li, []).append((int(row["fiscal_year"]), float(row["value"])))
    return out


def _load_valuation_facts(
    ticker: str,
    conn: Any,
    *,
    as_of_date: str,
    issuer_cik: str | None,
    issuer_aliases: tuple[str, ...],
    cfg: AppConfig,
) -> tuple[dict[str, list[tuple[int, float]]], list[dict[str, Any]]]:
    facts = _load_facts(
        ticker,
        conn,
        as_of_date=as_of_date,
        issuer_cik=issuer_cik,
        issuer_aliases=issuer_aliases,
        require_filed_asof=True,
    )
    return resolve_evidenced_zero_facts(
        facts,
        ticker=ticker,
        conn=conn,
        as_of_date=as_of_date,
        issuer_cik=issuer_cik,
        issuer_aliases=issuer_aliases,
        cfg=cfg,
    )


def _facts_revision_fingerprint(
    facts: dict[str, list[tuple[int, float]]],
    evidenced_zero_facts: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        line_item: [[int(year), float(value)] for year, value in values]
        for line_item, values in sorted(facts.items())
    }
    if evidenced_zero_facts:
        payload = {
            "normalized_facts": payload,
            "evidenced_zero_facts": evidenced_zero_facts,
        }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def valuation_facts_fingerprint(
    ticker: str,
    conn: Any,
    *,
    as_of_date: str,
    issuer_cik: str | None,
    issuer_aliases: tuple[str, ...] = (),
) -> str:
    """Return the exact issuer/PIT fact revision consumed by v2 valuation."""

    facts, evidenced_zero_facts = _load_valuation_facts(
        ticker,
        conn,
        as_of_date=as_of_date,
        issuer_cik=issuer_cik,
        issuer_aliases=issuer_aliases,
        cfg=get_config(),
    )
    return _facts_revision_fingerprint(facts, evidenced_zero_facts)


def _local_v2_facts_row(
    *,
    ticker: str,
    as_of_date: str,
    issuer_cik: str | None,
    facts: dict[str, list[tuple[int, float]]],
    evidenced_zero_facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project the already-loaded PIT facts into the legacy helper contract.

    The v2 valuation path has already resolved issuer identity and read the
    immutable, filed-as-of normalized fact revision.  Calling
    ``resolve_financial_facts_asof`` here would perform a second ticker-based
    CompanyFacts acquisition and could both cross issuer boundaries and make a
    network request.  Downstream valuation helpers only need this small facts
    row contract, so derive it from the exact revision being valued.
    """

    def latest(line_item: str) -> tuple[int | None, float | None]:
        series = facts.get(line_item) or []
        if not series:
            return None, None
        year, value = series[0]
        return int(year), float(value)

    shares_year, shares = latest("shares_outstanding")
    cfo_year, cfo = latest("cfo")
    capex_year, capex = latest("capex")
    fcf = None if cfo is None or capex is None else float(cfo) - abs(float(capex))
    fingerprint = _facts_revision_fingerprint(facts, evidenced_zero_facts)

    def status(value: float | None) -> str:
        return "OK" if value is not None else "UNKNOWN"

    def reason(value: float | None) -> str:
        return "OK" if value is not None else "TAG_MISS"

    return {
        "ticker": str(ticker).upper(),
        "requested_as_of": str(as_of_date),
        "cik": str(issuer_cik or ""),
        "status": "OK" if facts else "UNKNOWN",
        "shares_status": status(shares),
        "shares_reason": reason(shares),
        "shares_value": shares,
        "shares_asof_used": str(shares_year) if shares_year is not None else None,
        "cfo_status": status(cfo),
        "cfo_reason": reason(cfo),
        "cfo_value": cfo,
        "cfo_asof_used": str(cfo_year) if cfo_year is not None else None,
        "capex_status": status(capex),
        "capex_reason": reason(capex),
        "capex_value": capex,
        "capex_asof_used": str(capex_year) if capex_year is not None else None,
        "fcf_status": status(fcf),
        "fcf_reason": reason(fcf),
        "fcf_value": fcf,
        "fcf_asof_used": (str(max(cfo_year or 0, capex_year or 0)) if fcf is not None else None),
        "source_resolution": "local_normalized_companyfacts_v2",
        "fetch_reason_code": None,
        "fetch_reason_detail": None,
        "cache_path": None,
        "source_url": None,
        "network_attempted": False,
        "derived_from": [f"companyfacts_facts:{issuer_cik or 'unknown'}:{fingerprint}"],
        "facts_fingerprint": fingerprint,
    }


_QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _load_quarterly_facts(ticker: str, conn: Any) -> dict[str, list[tuple[int, str, float]]]:
    """
    Return quarterly companyfacts rows as:
      {line_item: [(fiscal_year, period_type, value), ...] sorted desc by (fiscal_year, quarter)}
    """
    rows = conn.execute(
        "SELECT line_item, fiscal_year, period_type, value FROM companyfacts_facts "
        "WHERE ticker = ? AND value IS NOT NULL AND period_type != 'FY' ORDER BY fiscal_year DESC, period_type DESC",
        (ticker.upper(),),
    ).fetchall()
    out: dict[str, list[tuple[int, str, float]]] = {}
    for row in rows:
        li = str(row["line_item"])
        out.setdefault(li, []).append(
            (int(row["fiscal_year"]), str(row["period_type"]), float(row["value"]))
        )
    # Sort each series by (fiscal_year desc, quarter desc)
    for li in out:
        out[li].sort(key=lambda x: (x[0], _QUARTER_ORDER.get(x[1], 0)), reverse=True)
    return out


def _latest_quarterly(qfacts: dict, field: str) -> tuple[int, str, float] | None:
    """Return the most recent (fiscal_year, period_type, value) for a field, or None."""
    series = qfacts.get(field) or []
    return series[0] if series else None


def _n_years(facts: dict, field: str, n: int = 5) -> list[tuple[int, float]]:
    """Return the (fiscal_year, value) pairs of the newest n FISCAL YEARS, descending.

    The window is anchored on the newest year on file and spans n calendar-
    adjacent fiscal years; a year missing inside it stays missing (the sample
    shrinks) rather than an older year being pulled in. Taking the newest n
    ROWS (what this did before 2026-09-29) spliced, say, 2016 onto 2021-2024
    and labelled it a five-year window.
    """
    series = sorted(facts.get(field) or [], key=lambda x: x[0], reverse=True)
    if not series:
        return []
    floor_year = int(series[0][0]) - (int(n) - 1)
    return [pair for pair in series if int(pair[0]) >= floor_year][:n]


def _compute_intangible_amort_addback(
    facts: dict,
    n_years: int = 5,
):
    """Compute the intangible amortization add-back series for EPV adjustment.

    Prefers the directly-reported `intangible_amortization` field
    (from XBRL `AmortizationOfIntangibleAssets`) when available — that's the
    ground-truth value. Falls back to the (D&A − capex) heuristic capped by
    intangibles when the direct field is missing.

    Returns (IntangibleAmortSeries, addback_by_year_dict) so callers can both
    inspect the metadata and apply the add-back to an OI series.
    """
    from app.valuation.intangible_amort import compute_series

    da = {y: v for y, v in _n_years(facts, "depreciation_amortization", n=n_years)}
    cx = {y: v for y, v in _n_years(facts, "capex", n=n_years)}
    ints = {y: v for y, v in _n_years(facts, "intangible_assets", n=n_years)}
    rev = {y: v for y, v in _n_years(facts, "revenue", n=n_years)}
    ia_direct = {y: v for y, v in _n_years(facts, "intangible_amortization", n=n_years)}

    # Build per-year payload using the union of years where ANY relevant field exists
    years_present = set(da.keys()) | set(cx.keys()) | set(ints.keys()) | set(ia_direct.keys())
    years_data = [
        {
            "fiscal_year": y,
            "d_and_a": da.get(y),
            "capex": cx.get(y),
            "intangible_assets": ints.get(y),
            "intangible_amortization": ia_direct.get(y),
        }
        for y in sorted(years_present, reverse=True)[:n_years]
    ]

    series = compute_series(ticker="", years_data=years_data, revenue_by_year=rev)
    addback_by_year = {y.fiscal_year: y.addback for y in series.years}
    return series, addback_by_year


def _compute_nonrecurring_revenue(ticker: str, facts: dict, n_years: int = 5):
    """Detect non-recurring revenue spikes in the latest year."""
    from app.valuation.nonrecurring_revenue import detect

    rev_series = _n_years(facts, "revenue", n=n_years)
    return detect(ticker=ticker, revenue_series=rev_series)


def _latest_common_year(facts: dict, *fields: str) -> int | None:
    """
    Return the latest fiscal_year where ALL named fields have data.
    Returns None if any field has no data or no common year exists.
    Enforces period-alignment across multiple fields.
    """
    year_sets = []
    for field in fields:
        years = {yr for yr, _ in (facts.get(field) or [])}
        if not years:
            return None
        year_sets.append(years)
    common = year_sets[0]
    for s in year_sets[1:]:
        common = common & s
    return max(common) if common else None


def _get_value_for_year(facts: dict, field: str, year: int) -> float | None:
    """Return value for (field, year) pair, or None if absent."""
    for yr, val in facts.get(field) or []:
        if yr == year:
            return val
    return None


# The registrant's SIC could not be read, so REIT status is not known.
REASON_REIT_STATUS_UNKNOWN = "REIT_STATUS_UNKNOWN"


def _reit_not_applicable() -> dict[str, Any]:
    return {
        "status": "NOT_APPLICABLE",
        "reason_code": REASON_REIT_DEPRECIATION_DISTORTS_EARNINGS,
        "value_per_share": None,
        "flags": [REASON_REIT_DEPRECIATION_DISTORTS_EARNINGS],
    }


def _short_term_investment_tags(conn: Any, ticker: str, fiscal_year: int) -> str | None:
    """The concepts behind a short_term_investments row (provenance only)."""
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(companyfacts_facts)")}
        if "source_tags" not in columns:
            return None
        row = conn.execute(
            "SELECT source_tags FROM companyfacts_facts WHERE ticker = ? AND fiscal_year = ? "
            "AND period_type = 'FY' AND line_item = 'short_term_investments' LIMIT 1",
            (str(ticker).upper(), int(fiscal_year)),
        ).fetchone()
    except Exception:  # noqa: BLE001 - provenance must never fail a valuation
        return None
    return str(row[0]) if row is not None and row[0] else None


def _latest_balance_sheet_year(facts: dict) -> int | None:
    """The newest fiscal year with an annual balance sheet on file."""
    return max(
        (
            year
            for line_item in ("cash", "total_liabilities", "total_assets", "equity")
            for year, _value in facts.get(line_item, ())
        ),
        default=None,
    )


def _debt_year_is_stale(facts: dict, debt_year: int | None) -> bool:
    """True when debt read from ``debt_year`` says nothing about today's claims.

    The rule net debt uses (Ford: debt tagged only by segment since 2021, so
    companyfacts' last total is 2020 while its balance sheet runs to 2025): a
    year more than one fiscal year older than the latest annual balance sheet is
    refused; one year behind is accepted.
    """
    latest = _latest_balance_sheet_year(facts)
    return debt_year is not None and latest is not None and debt_year < latest - 1


def _cash_like_for_year(facts: dict, year: int) -> float | None:
    """Cash plus the current short-term investments beside it, for net debt.

    The short_term_investments row is resolved by the normalizer at the cash
    row's own balance-sheet date and already excludes anything inside the cash
    figure, so adding it never counts an amount twice. None when cash is absent.
    """
    cash = _get_value_for_year(facts, "cash", year)
    if cash is None or not _finite_numeric(cash):
        return None
    short_term = _get_value_for_year(facts, "short_term_investments", year)
    if short_term is not None and _finite_numeric(short_term) and float(short_term) > 0:
        return float(cash) + float(short_term)
    return float(cash)


def _finite_numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _normalize_price_context(
    *,
    quote: Any | None = None,
    snapshot: Any | None = None,
    diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = (
        diagnostic.get("result")
        if isinstance(diagnostic, dict) and isinstance(diagnostic.get("result"), dict)
        else {}
    )
    output_fields = (
        diagnostic.get("output_fields")
        if isinstance(diagnostic, dict) and isinstance(diagnostic.get("output_fields"), dict)
        else {}
    )
    quote_provenance = (
        quote.provenance
        if quote is not None and isinstance(getattr(quote, "provenance", None), dict)
        else {}
    )

    def _first_explicit(*keys: str) -> Any:
        for source in (snapshot, quote, quote_provenance, output_fields):
            for key in keys:
                value = (
                    source.get(key)
                    if isinstance(source, dict)
                    else getattr(source, key, None)
                    if source is not None
                    else None
                )
                if value is not None and value != "":
                    return value
        return None

    current_price = None
    if snapshot is not None and _finite_numeric(getattr(snapshot, "price", None)):
        current_price = float(snapshot.price)
    elif quote is not None and _finite_numeric(getattr(quote, "price", None)):
        current_price = float(quote.price)
    elif _finite_numeric(output_fields.get("current_price")):
        current_price = float(output_fields["current_price"])
    status = "OK" if current_price is not None else "UNKNOWN"
    reason_code = (
        "OK" if current_price is not None else str(result.get("reason_code") or "PRICE_UNKNOWN")
    )
    reason_detail = (
        ""
        if current_price is not None
        else str(result.get("reason_detail") or "Market price unavailable.")
    )
    source = None
    if snapshot is not None:
        source = getattr(snapshot, "source", None)
    elif quote is not None:
        source = getattr(quote, "provider", None)
    if not source:
        source = output_fields.get("price_source") or "unknown"
    price_basis_raw = _first_explicit("price_basis", "current_price_basis")
    price_basis = str(price_basis_raw).strip().upper() if price_basis_raw is not None else None
    raw_price = _first_explicit("raw_price", "current_raw_price")
    if not _finite_numeric(raw_price) and price_basis == "UNADJUSTED" and current_price is not None:
        # An explicitly unadjusted quote is, by definition, its own raw close.
        raw_price = current_price
    split_event = _first_explicit("split_event")
    no_intervening_split_proof = _first_explicit("no_intervening_split_proof")
    return {
        "market_price": current_price if current_price is not None else "UNKNOWN",
        "price": current_price if current_price is not None else "UNKNOWN",
        "price_status": status,
        "price_reason_code": reason_code,
        "price_reason_detail": reason_detail,
        "price_source": str(source or "unknown"),
        "current_price_source": str(source or "unknown"),
        "price_source_resolution": str(source or "unknown"),
        "price_as_of_date": _first_explicit(
            "price_asof_used",
            "current_price_as_of_date",
            "as_of_date",
        ),
        "price_fetched_at": _first_explicit("retrieved_at", "fetched_at"),
        "price_source_url": sanitize_url_credentials(
            _first_explicit(
                "url",
                "source_url",
                "price_source_url",
                "current_price_source_url",
            )
        ),
        "price_currency": _first_explicit(
            "currency",
            "price_currency",
            "current_price_currency",
        ),
        "price_confidence": _first_explicit("confidence", "price_confidence"),
        "price_basis": price_basis,
        "raw_price": float(raw_price) if _finite_numeric(raw_price) else None,
        "split_adjustment_factor": _first_explicit("split_adjustment_factor"),
        "split_effective_date": _first_explicit("split_effective_date"),
        "split_event": (
            sanitize_json_value(split_event) if isinstance(split_event, dict) else None
        ),
        "no_intervening_split_proof": (
            sanitize_json_value(no_intervening_split_proof)
            if isinstance(no_intervening_split_proof, dict)
            else None
        ),
        "quote_snapshot_id": _first_explicit("quote_snapshot_id"),
        "price_diagnostic": sanitize_json_value(diagnostic or {}),
    }


def _scorecard_price_lineage(price_context: dict[str, Any]) -> dict[str, Any]:
    """Project explicit quote lineage into the scorecard consumer contract."""

    detail: dict[str, Any] = {}
    price = price_context.get("price")
    if _finite_numeric(price) and float(price) > 0:
        detail["current_price"] = float(price)

    string_fields = {
        "current_price_as_of_date": "price_as_of_date",
        "current_price_currency": "price_currency",
        "current_price_source": "current_price_source",
        "current_price_source_url": "price_source_url",
        "current_price_basis": "price_basis",
        "split_effective_date": "split_effective_date",
        "quote_snapshot_id": "quote_snapshot_id",
    }
    for output_key, context_key in string_fields.items():
        value = price_context.get(context_key)
        normalized = str(value).strip() if value is not None else ""
        if normalized and normalized.upper() not in {"UNKNOWN", "NONE"}:
            detail[output_key] = normalized

    numeric_fields = {
        "current_raw_price": "raw_price",
        "split_adjustment_factor": "split_adjustment_factor",
    }
    for output_key, context_key in numeric_fields.items():
        value = price_context.get(context_key)
        if _finite_numeric(value):
            detail[output_key] = float(value)

    for key in ("split_event", "no_intervening_split_proof"):
        value = price_context.get(key)
        if isinstance(value, dict):
            detail[key] = sanitize_json_value(value)
    return detail


def _load_run_scoped_price_artifact(
    run_id: str,
    ticker: str,
    as_of_date: str,
    *,
    cfg: AppConfig | None = None,
) -> tuple[float | None, dict[str, Any]]:
    resolved_cfg = cfg or get_config()
    path = (
        resolved_cfg.outputs_dir
        / "prices"
        / str(run_id).strip()
        / f"{str(ticker or '').strip().upper()}.json"
    )
    if not path.exists():
        return None, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    if not isinstance(payload, dict):
        return None, {}
    if str(payload.get("status") or "").upper() != "OK":
        return None, {}
    # The artifact must say it priced THIS as-of date. One that names no date
    # used to be accepted for any date, so a price for one day could value a
    # run for another. The run's own writer always records
    # the date; an undated artifact is not evidence for a dated request.
    if not str(as_of_date or "").strip() or str(
        payload.get("requested_as_of_date") or ""
    ).strip() != str(as_of_date).strip():
        return None, {}
    snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
    diagnostic = payload.get("diagnostic") if isinstance(payload.get("diagnostic"), dict) else {}
    output_fields = (
        diagnostic.get("output_fields") if isinstance(diagnostic.get("output_fields"), dict) else {}
    )
    price = snapshot.get("price")
    if not _finite_numeric(price):
        price = output_fields.get("current_price")
    if not _finite_numeric(price) or float(price) <= 0:
        return None, {}
    context = _normalize_price_context(
        snapshot=type("Snapshot", (), snapshot)(),
        diagnostic=diagnostic,
    )
    context["price_reason_code"] = str(
        (diagnostic.get("result") or {}).get("reason_code") or "CACHE_HIT"
    )
    context["price_reason_detail"] = str(
        (diagnostic.get("result") or {}).get("reason_detail")
        or "Price resolved from outputs/prices/<run_id> cache."
    )
    context["price_source_resolution"] = "run_scoped_output"
    context["current_price_source"] = str(
        snapshot.get("source") or output_fields.get("price_source") or "run_scoped_output"
    )
    context["price_diagnostic"] = {
        "artifact_path": str(path),
        "artifact_payload_status": str(payload.get("status") or ""),
        "diagnostic": diagnostic,
    }
    return float(price), context


def _load_disk_cache_price_artifact(
    ticker: str,
    as_of_date: str,
    *,
    cfg: AppConfig | None = None,
) -> tuple[float | None, dict[str, Any]]:
    resolved_cfg = cfg or get_config()
    path = resolved_cfg.cache_dir / "prices" / f"{str(ticker or '').strip().upper()}.json"
    if not path.exists():
        return None, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None, {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        if str(row.get("requested_as_of_date") or "") != str(as_of_date):
            continue
        snapshot = row.get("snapshot") if isinstance(row.get("snapshot"), dict) else {}
        price = snapshot.get("price")
        if not isinstance(price, (int, float)):
            continue
        context = _normalize_price_context(snapshot=type("Snapshot", (), snapshot)())
        context["price_reason_code"] = "CACHE_HIT"
        context["price_reason_detail"] = "Price resolved from data/cache/prices cache."
        context["price_source_resolution"] = "disk_cache"
        context["price_diagnostic"] = {
            "artifact_path": str(path),
            "artifact_payload_status": "OK",
            "cache_entry": row,
        }
        return float(price), context
    return None, {}


def _resolve_market_price(
    ticker: str,
    as_of_date: str,
    provider: Any | None,
    *,
    run_id: str | None = None,
    price_override: float | None = None,
    cfg: AppConfig | None = None,
) -> tuple[float | None, dict[str, Any]]:
    resolved_cfg = cfg or get_config()
    if isinstance(price_override, (int, float)):
        return float(price_override), {
            "market_price": float(price_override),
            "price": float(price_override),
            "price_status": "OK",
            "price_reason_code": "PRICE_OVERRIDE",
            "price_reason_detail": "Price injected from deep-scan prewarm output.",
            "price_source": "prewarm_override",
            "current_price_source": "prewarm_override",
            "price_source_resolution": "prewarm_override",
            "price_as_of_date": as_of_date,
            "price_fetched_at": None,
            "price_source_url": None,
            "price_confidence": "HIGH",
            "price_diagnostic": {"source": "deep_scan_prewarm_override"},
        }
    if run_id:
        run_price, run_context = _load_run_scoped_price_artifact(
            run_id,
            ticker,
            as_of_date,
            cfg=resolved_cfg,
        )
        if run_price is not None:
            return run_price, run_context
    disk_cache_price, disk_cache_context = _load_disk_cache_price_artifact(
        ticker,
        as_of_date,
        cfg=resolved_cfg,
    )
    if disk_cache_price is not None:
        return disk_cache_price, disk_cache_context
    if provider is not None and hasattr(provider, "get_quote"):
        quote = provider.get_quote(ticker, as_of_date)
        price = quote.price if quote and isinstance(quote.price, (int, float)) else None
        if price is not None:
            return price, _normalize_price_context(quote=quote)
        return price, _normalize_price_context(quote=quote)

    if provider is not None and hasattr(provider, "get_price_asof"):
        snapshot = provider.get_price_asof(ticker, as_of_date)
        diagnostic = (
            provider.get_last_diagnostic(ticker, as_of_date)
            if hasattr(provider, "get_last_diagnostic")
            else None
        )
        price = snapshot.price if snapshot and isinstance(snapshot.price, (int, float)) else None
        if price is not None:
            return price, _normalize_price_context(snapshot=snapshot, diagnostic=diagnostic)
        return price, _normalize_price_context(snapshot=snapshot, diagnostic=diagnostic)

    market_provider = get_market_price_provider(resolved_cfg)
    snapshot = market_provider.get_price_asof(ticker, as_of_date)
    diagnostic = (
        market_provider.get_last_diagnostic(ticker, as_of_date)
        if hasattr(market_provider, "get_last_diagnostic")
        else None
    )
    price = snapshot.price if snapshot and isinstance(snapshot.price, (int, float)) else None
    if price is not None:
        return price, _normalize_price_context(snapshot=snapshot, diagnostic=diagnostic)
    return price, _normalize_price_context(snapshot=snapshot, diagnostic=diagnostic)


# ── owner earnings ─────────────────────────────────────────────────────────────


def _normalize_capex(series: list[tuple[int, float]], n: int = 5) -> float:
    """
    3-step spike-capping:
    1. Compute uncapped mean of up to n most-recent years.
    2. Cap any year where value > 2.0× uncapped mean (replace with uncapped mean).
    3. Return mean of capped values.
    Spike threshold is exactly 2.0× — not 1.5×, not 3×.

    Capex enters as a MAGNITUDE. Filers that report the element negated would
    otherwise turn ``cfo - norm_capex`` into an addition, and invert this cap:
    on a negative series the threshold sits below every ordinary year, so the
    ordinary years are capped down to the mean and the spike year survives.
    ``_local_v2_facts_row`` already uses ``abs(capex)`` on the same input.
    """
    vals = [abs(v) for _, v in sorted(series, key=lambda x: x[0], reverse=True)[:n]]
    if not vals:
        return 0.0
    uncapped_mean = sum(vals) / len(vals)
    threshold = _CAPEX_SPIKE_MULTIPLIER * uncapped_mean
    capped = [min(v, uncapped_mean) if v > threshold else v for v in vals]
    return sum(capped) / len(capped)


# A latest-year CFO is an outlier against its own aligned 5-year median when it
# sits more than this factor above it (a spike) or below it (a dip).
_CFO_OUTLIER_RATIO = 1.25


def _cfo_dip_is_real_decline(facts: dict, latest_year: int) -> bool | None:
    """Whether a CFO dip in ``latest_year`` is matched by the business itself.

    True when revenue OR operating income for that year sits more than
    _CFO_OUTLIER_RATIO below its own 5-year median (or operating income turned
    non-positive against a positive median), or when either one is lower in
    that year than at the start of its 5-year window (negative 5-year growth;
    2026-09-29 — a steady -10%-a-year decliner never falls 1.25x below its own
    median, and its dip was being lifted): the cash fell because the business
    did, and the dip is followed, not lifted. False when both held
    up — a one-time working-capital or tax outflow the base should not
    capitalize. None when either history is shorter than _MIN_YEARS or has no
    figure for that year: the dip cannot be told apart from a decline, so it
    is not lifted (the conservative reading).
    """
    for field in ("revenue", "operating_income"):
        series = _n_years(facts, field, n=5)
        latest = _get_value_for_year(facts, field, latest_year)
        if len(series) < _MIN_YEARS or not _finite_numeric(latest):
            return None
    for field in ("revenue", "operating_income"):
        window = sorted(
            ((int(y), float(v)) for y, v in _n_years(facts, field, n=5) if _finite_numeric(v)),
            key=lambda item: item[0],
        )
        values = [v for _, v in window]
        median = statistics.median(values)
        latest = float(_get_value_for_year(facts, field, latest_year))  # type: ignore[arg-type]
        if latest < window[0][1] and latest_year > window[0][0]:
            return True
        if median > 0 and latest * _CFO_OUTLIER_RATIO < median:
            return True
        if median > 0 and latest <= 0:
            return True
        if median <= 0:
            # No positive baseline to measure the year against.
            return None
    return False


def _compute_owner_earnings(
    facts: dict,
    *,
    maintenance_capex_ratio: float = 1.0,
    durable_revenue_series: list[tuple[int, float]] | None = None,
) -> dict[str, Any]:
    """
    Compute owner earnings (FCFF basis) for the latest year where CFO and capex
    share a fiscal year.
    SBC is a dilution-aware policy adjustment — not an accounting identity.
    Missing SBC → confidence LOWER (not silently equivalent to adjusted result).

    Basis:
      - CFO peak normalization: a transient latest-year CFO (>1.25× the median
        of the aligned series, ≥3 years) is replaced by the median — the same
        discipline _normalize_capex applies to the deduction side — unless
        revenue CAGR shows a strong secular grower (>8%). Symmetric since
        2026-09-29: a latest CFO below median / 1.25 is lifted to the median
        (CFO_DIP_NORMALIZED) when revenue and operating income held up, and
        kept when they fell too (CFO_DIP_REAL_DECLINE) or cannot be checked
        (CFO_DIP_UNVERIFIED). ``cfo_normalization`` names which happened.
      - FCFF conversion: US-GAAP CFO is post-interest, but the DCF discounts at
        WACC and bridges EV→equity via net debt, so after-tax interest expense
        is added back. Interest income is not ingested; cash-rich names get
        INTEREST_INCOME_NOT_ADJUSTED instead of a fabricated adjustment.

    maintenance_capex_ratio: fraction of normalized capex treated as maintenance
        (0.0-1.0). The DCF passes 1.0 (full capex): its scenarios project growth,
        and excluding "growth capex" while also projecting the growth it buys
        would double-count. Fractional category ratios are reserved for
        zero-growth earnings-power surfaces and gates.
    durable_revenue_series: spike-corrected revenue series (review OE-1).
        The strong-grower exemption's endpoint CAGR must not be inflated by
        the very latest-year non-recurring spike the normalization guards
        against — callers that detect a spike pass the durable series here.
    """
    flags: list[str] = []

    cfo_series = _n_years(facts, "cfo", n=5)
    if not cfo_series:
        return {"status": "OWNER_EARNINGS_INSUFFICIENT_DATA", "flags": flags, "confidence": "NONE"}

    capex_series = _n_years(facts, "capex", n=5)
    if capex_series:
        capex_years = {yr for yr, _ in capex_series}
        # Find latest year where CFO and capex are both present (period-alignment)
        aligned = [(yr, v) for yr, v in cfo_series if yr in capex_years]
        if aligned:
            latest_year, latest_cfo = aligned[0]
            cfo_basis = [v for _, v in aligned]
            latest_capex = dict(capex_series)[latest_year]
            raw_norm_capex = _normalize_capex(capex_series)
            norm_capex = raw_norm_capex * max(0.0, min(1.0, maintenance_capex_ratio))
            if maintenance_capex_ratio < 1.0:
                flags.append(f"MAINT_CAPEX_RATIO_{maintenance_capex_ratio:.0%}")
            if len(capex_series) < _MIN_YEARS:
                flags.append("CAPEX_HISTORY_SHORT")
        else:
            latest_year, latest_cfo = cfo_series[0]
            cfo_basis = [v for _, v in cfo_series]
            norm_capex = 0.0
            flags.append("CAPEX_PERIOD_MISMATCH")
    else:
        latest_year, latest_cfo = cfo_series[0]
        cfo_basis = [v for _, v in cfo_series]
        norm_capex = 0.0
        flags.append("CAPEX_UNKNOWN")

    confidence = "NORMAL"

    # CFO normalization (mirror of the gate's PEAK detection, applied to the
    # DCF's own base), symmetric since 2026-09-29: a single-year working-
    # capital swing sets the binding anchor in neither direction. A latest
    # CFO more than _CFO_OUTLIER_RATIO above the aligned 5-year median is a
    # spike and is smoothed down to it; one more than _CFO_OUTLIER_RATIO
    # below it is a dip and is lifted to it — unless revenue or operating
    # income fell by the same rule that year, which makes the dip a real
    # decline the base must follow (see _cfo_dip_is_real_decline).
    cfo_latest_raw = latest_cfo
    cfo_normalization = "NONE"
    if len(cfo_basis) >= 3:
        cfo_median = statistics.median(cfo_basis)
        exemption_rev_series = (
            durable_revenue_series
            if durable_revenue_series is not None
            else _n_years(facts, "revenue", n=5)
        )
        rev_cagr_for_cfo = _revenue_cagr(exemption_rev_series)
        strong_grower = isinstance(rev_cagr_for_cfo, (int, float)) and rev_cagr_for_cfo > 0.08
        if cfo_median > 0 and latest_cfo > _CFO_OUTLIER_RATIO * cfo_median and not strong_grower:
            latest_cfo = float(cfo_median)
            flags.append("CFO_PEAK_NORMALIZED")
            cfo_normalization = "SPIKE_SMOOTHED"
            confidence = "LOWER"
        elif cfo_median > 0 and latest_cfo > _CFO_OUTLIER_RATIO * cfo_median:
            # Strong-grower exemption: the latest CFO stands unsmoothed. Its
            # capex must be the same year's, uncapped — pairing a growth year's
            # cash inflow with a 5-year averaged and spike-capped capex deducted
            # an older, smaller investment level from this year's receipts (a
            # growth-phase issuer's owner earnings nearly double its FCF).
            if capex_series and aligned:
                norm_capex = abs(float(latest_capex)) * max(
                    0.0, min(1.0, maintenance_capex_ratio)
                )
                flags.append("CAPEX_LATEST_PAIRED_WITH_GROWER_CFO")
            cfo_normalization = "GROWER_LATEST_KEPT"
        elif cfo_median > 0 and latest_cfo * _CFO_OUTLIER_RATIO < cfo_median:
            decline = (
                None if latest_cfo <= 0 else _cfo_dip_is_real_decline(facts, latest_year)
            )
            if latest_cfo <= 0:
                # A year that burned cash is never lifted to a positive median:
                # a loss is not a working-capital timing swing to smooth away.
                flags.append("CFO_DIP_NONPOSITIVE_KEPT")
                cfo_normalization = "DIP_KEPT_NONPOSITIVE"
            elif decline is None:
                flags.append("CFO_DIP_UNVERIFIED")
                cfo_normalization = "DIP_KEPT_UNVERIFIED"
            elif decline:
                flags.append("CFO_DIP_REAL_DECLINE")
                cfo_normalization = "DIP_KEPT_REAL_DECLINE"
            else:
                latest_cfo = float(cfo_median)
                flags.append("CFO_DIP_NORMALIZED")
                cfo_normalization = "DIP_LIFTED"
                confidence = "LOWER"
        elif cfo_median <= 0 and latest_cfo > 0:
            # A loss-history name whose latest CFO is a one-year positive
            # spike keeps the spike — the median basis can't normalize it.
            # Flag instead of silently skipping (review OE-3).
            flags.append("CFO_PEAK_CHECK_NEGATIVE_MEDIAN")
    else:
        # CFO-capex aligned intersection too short for the peak check — the
        # raw single-year CFO stands unverified (review OE-3).
        flags.append("CFO_PEAK_CHECK_INSUFFICIENT_HISTORY")

    # SBC must exist for the same latest_year; policy choice — signals dilution
    sbc_val = _get_value_for_year(facts, "sbc", latest_year)
    if sbc_val is None:
        sbc_val = 0.0
        flags.append("SBC_NOT_ADJUSTED")
        confidence = "LOWER"  # LOWER, not NORMAL — callers must propagate this
    else:
        revenue = _get_value_for_year(facts, "revenue", latest_year) or 1.0
        if sbc_val / max(1.0, revenue) > 0.05:
            flags.append("SBC_BURDEN_HIGH")

    # FCFF conversion: add back after-tax interest expense, taxed at the
    # issuer's own normalized rate — the one its EPV uses — with the same 21%
    # statutory fallback when its filings do not support one.
    interest_tax_basis = _normalized_tax_rate(facts)
    if interest_tax_basis["tax_rate"] is not None:
        interest_tax_rate = float(interest_tax_basis["tax_rate"])
        interest_tax_rate_source = "ISSUER"
    else:
        interest_tax_rate = _TAX_RATE
        interest_tax_rate_source = "STATUTORY_DEFAULT"
    interest_expense = _get_value_for_year(facts, "interest_expense", latest_year)
    interest_addback = 0.0
    if isinstance(interest_expense, (int, float)) and interest_expense > 0:
        interest_addback = float(interest_expense) * (1 - interest_tax_rate)
        flags.append("FCFF_INTEREST_ADDBACK")
    elif interest_expense is None:
        # No interest fact for the year owner earnings are built on. For an
        # issuer that reports interest in other years, or carries debt that
        # year, the missing figure is an unknown addback, not a known zero: the
        # FCFF base is understated by an unknown amount, and an older year's
        # interest must not stand in for it.
        latest_debt = _get_value_for_year(facts, "total_debt", latest_year)
        reports_interest_elsewhere = any(
            isinstance(value, (int, float)) and value > 0
            for _, value in facts.get("interest_expense") or []
        )
        if reports_interest_elsewhere or (
            isinstance(latest_debt, (int, float)) and latest_debt > 0
        ):
            flags.append("FCFF_INTEREST_UNKNOWN")
            confidence = "LOWER"

    cash_latest = _get_value_for_year(facts, "cash", latest_year)
    revenue_latest = _get_value_for_year(facts, "revenue", latest_year)
    if (
        isinstance(cash_latest, (int, float))
        and isinstance(revenue_latest, (int, float))
        and revenue_latest > 0
        and cash_latest > 0.20 * revenue_latest
    ):
        flags.append("INTEREST_INCOME_NOT_ADJUSTED")

    owner_earnings = latest_cfo - norm_capex - sbc_val + interest_addback

    return {
        "status": "OK",
        "fiscal_year": latest_year,
        "owner_earnings_latest": owner_earnings,
        "cfo_used": latest_cfo,
        "cfo_latest_raw": cfo_latest_raw,
        "cfo_normalization": cfo_normalization,
        "interest_addback": interest_addback,
        "interest_addback_tax_rate": interest_tax_rate,
        "interest_addback_tax_rate_source": interest_tax_rate_source,
        "normalized_capex": norm_capex,
        "sbc_used": sbc_val,
        "flags": flags,
        "confidence": confidence,
    }


# ── four valuation methods ─────────────────────────────────────────────────────


def _revenue_cagr(revenue_series: list[tuple[int, float]], n: int = 5) -> float | None:
    """Trailing CAGR from up to n most-recent years. None if < 2 years available."""
    series = sorted(revenue_series, key=lambda x: x[0], reverse=True)[:n]
    if len(series) < 2:
        return None
    newest_val, oldest_val = series[0][1], series[-1][1]
    years = series[0][0] - series[-1][0]
    if oldest_val <= 0 or years <= 0:
        return None
    return (newest_val / oldest_val) ** (1 / years) - 1


def _average_common_ratio(
    facts: dict[str, list[tuple[int, float]]],
    numerator_field: str,
    denominator_field: str,
    *,
    years: int = 3,
    require_positive_denominator: bool = True,
    ratio_of_sums: bool = False,
) -> float | None:
    """Ratio of two fields over their newest ``years`` common fiscal years.

    Default: the mean of the per-year ratios. With ``ratio_of_sums`` the
    kept years are summed first (total numerator / total denominator), so a
    year whose denominator is near zero cannot dominate: CFO of 100 a year
    against net income of 100, 100 and 1 is 300 / 201 = 1.49x, where the mean
    of the per-year ratios is 34x.
    """
    numerator = {int(year): float(value) for year, value in facts.get(numerator_field) or []}
    denominator = {int(year): float(value) for year, value in facts.get(denominator_field) or []}
    common_years = sorted(set(numerator) & set(denominator), reverse=True)[: max(1, int(years))]
    ratios: list[float] = []
    kept: list[int] = []
    for year in common_years:
        denom = float(denominator[year])
        if require_positive_denominator and denom <= 0:
            continue
        if abs(denom) < 1e-9:
            continue
        ratios.append(float(numerator[year]) / denom)
        kept.append(year)
    if not ratios:
        return None
    if ratio_of_sums:
        total_denominator = sum(float(denominator[year]) for year in kept)
        if abs(total_denominator) < 1e-9:
            return None
        return sum(float(numerator[year]) for year in kept) / total_denominator
    return sum(ratios) / len(ratios)


def _gross_margin_series(
    facts: dict[str, list[tuple[int, float]]], *, years: int = 3
) -> list[tuple[int, float]]:
    gross_profit = {int(year): float(value) for year, value in facts.get("gross_profit") or []}
    revenue = {int(year): float(value) for year, value in facts.get("revenue") or []}
    out: list[tuple[int, float]] = []
    for year in sorted(set(gross_profit) & set(revenue), reverse=True)[: max(1, int(years))]:
        rev = revenue[year]
        if rev <= 0:
            continue
        out.append((year, gross_profit[year] / rev))
    return out


def _latest_net_debt_and_ebitda(
    facts: dict[str, list[tuple[int, float]]],
) -> tuple[float, float] | None:
    """Latest (net debt, EBITDA proxy) from one common fiscal year, or None."""
    year = _latest_common_year(facts, "total_debt", "cash", "operating_income")
    # Debt from a year well behind the balance sheet is refused, as in net debt.
    if year is None or _debt_year_is_stale(facts, year):
        return None
    debt = _get_value_for_year(facts, "total_debt", year)
    cash = _cash_like_for_year(facts, year)
    operating_income = _get_value_for_year(facts, "operating_income", year)
    if (
        not isinstance(debt, (int, float))
        or not isinstance(cash, (int, float))
        or not isinstance(operating_income, (int, float))
    ):
        return None
    # Use real D&A when available; fall back to operating income as EBITDA proxy
    da = _get_value_for_year(facts, "depreciation_amortization", year)
    if isinstance(da, (int, float)) and da > 0:
        ebitda = float(operating_income) + float(da)
    else:
        ebitda = float(operating_income)
    return float(debt) - float(cash), ebitda


def _latest_net_debt_to_ebitda_proxy(facts: dict[str, list[tuple[int, float]]]) -> float | None:
    pair = _latest_net_debt_and_ebitda(facts)
    if pair is None:
        return None
    net_debt, ebitda = pair
    if ebitda <= 0:
        return None
    return net_debt / ebitda


def _net_debt_with_no_earnings_to_service_it(
    facts: dict[str, list[tuple[int, float]]],
) -> bool:
    """True when the issuer owes net debt and its EBITDA proxy is not positive.

    The ratio helper returns None for two different situations — the inputs are
    missing, and the ratio is undefined because there are no earnings to divide
    by — and the leverage rule read both as "did not fire". A company with 1,000
    of net debt against an operating LOSS was therefore discounted a full point
    more cheaply than the same balance sheet against a profit.
    """
    pair = _latest_net_debt_and_ebitda(facts)
    if pair is None:
        return False
    net_debt, ebitda = pair
    return net_debt > 0 and ebitda <= 0


def _compute_quality_wacc(
    facts: dict[str, list[tuple[int, float]]],
    *,
    category_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = float(_WACC)
    adjusted = baseline
    adjustments: list[dict[str, Any]] = []
    rule_evaluations: list[dict[str, Any]] = []

    def _apply(code: str, delta: float, *, reason: str, metric_value: Any) -> None:
        nonlocal adjusted
        adjusted += float(delta)
        adjustments.append(
            {
                "code": code,
                "delta": float(delta),
                "reason": reason,
                "metric_value": metric_value,
            }
        )

    gross_margin_avg = _average_common_ratio(facts, "gross_profit", "revenue", years=3)
    # Ratio of sums, not the mean of per-year ratios: one near-breakeven year
    # made the mean 34x on a 1.49x business and took half a point off WACC.
    cash_conversion_avg = _average_common_ratio(
        facts, "cfo", "net_income", years=3, ratio_of_sums=True
    )
    gross_margin_rows = _gross_margin_series(facts, years=3)
    gross_margin_stable_or_growing = False
    if len(gross_margin_rows) >= 2:
        ordered = sorted(gross_margin_rows, key=lambda item: item[0])
        gross_margin_stable_or_growing = ordered[-1][1] >= ordered[0][1]

    metrics_used = (
        (category_result or {}).get("metrics_used")
        if isinstance((category_result or {}).get("metrics_used"), dict)
        else {}
    )
    rnd_intensity = metrics_used.get("avg_rnd_to_revenue_3y")
    rnd_intensity_num = float(rnd_intensity) if isinstance(rnd_intensity, (int, float)) else None
    revenue_cagr = _revenue_cagr(_n_years(facts, "revenue", n=5))
    net_debt_to_ebitda = _latest_net_debt_to_ebitda_proxy(facts)

    def _record_rule(
        code: str,
        *,
        metric_name: str,
        metric_value: Any,
        threshold: Any,
        comparison: str,
        delta: float,
        fired: bool,
        reason: str,
    ) -> None:
        rule_evaluations.append(
            {
                "code": code,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "threshold": threshold,
                "comparison": comparison,
                "delta": float(delta) if fired else 0.0,
                "candidate_delta": float(delta),
                "fired": bool(fired),
                "reason": reason,
            }
        )
        if fired:
            _apply(code, delta, reason=reason, metric_value=metric_value)

    _record_rule(
        "HIGH_GROSS_MARGIN",
        metric_name="gross_margin_avg_3y",
        metric_value=round(float(gross_margin_avg), 6)
        if isinstance(gross_margin_avg, (int, float))
        else None,
        threshold=0.70,
        comparison=">",
        delta=-0.005,
        fired=isinstance(gross_margin_avg, (int, float)) and gross_margin_avg > 0.70,
        reason="Gross margin above 70% implies more predictable cash flows.",
    )
    _record_rule(
        "STRONG_CASH_CONVERSION",
        metric_name="cash_conversion_avg_3y",
        metric_value=round(float(cash_conversion_avg), 6)
        if isinstance(cash_conversion_avg, (int, float))
        else None,
        threshold=1.5,
        comparison=">",
        delta=-0.005,
        fired=isinstance(cash_conversion_avg, (int, float)) and cash_conversion_avg > 1.5,
        reason="3-year CFO / net income (ratio of sums) above 1.5x reduces earnings quality risk.",
    )
    _record_rule(
        "RND_MOAT",
        metric_name="rnd_intensity_avg_3y",
        metric_value=round(float(rnd_intensity_num), 6)
        if isinstance(rnd_intensity_num, (int, float))
        else None,
        threshold={"avg_rnd_to_revenue_3y": 0.15, "gross_margin_stable_or_growing": True},
        comparison="rnd>0.15 and margin_trend",
        delta=-0.0025,
        fired=rnd_intensity_num is not None
        and rnd_intensity_num > 0.15
        and gross_margin_stable_or_growing,
        reason="High R&D intensity with stable/growing gross margin suggests durable reinvestment.",
    )
    _record_rule(
        "LOW_GROWTH_HEADWIND",
        metric_name="revenue_cagr_5y",
        metric_value=round(float(revenue_cagr), 6)
        if isinstance(revenue_cagr, (int, float))
        else None,
        threshold=0.05,
        comparison="<",
        delta=0.005,
        fired=isinstance(revenue_cagr, (int, float)) and revenue_cagr < 0.05,
        reason="Revenue CAGR below 5% increases terminal execution risk.",
    )
    leverage_unservicable = _net_debt_with_no_earnings_to_service_it(facts)
    _record_rule(
        "LEVERAGE_RISK",
        metric_name="net_debt_to_ebitda_proxy",
        metric_value=round(float(net_debt_to_ebitda), 6)
        if isinstance(net_debt_to_ebitda, (int, float))
        else None,
        threshold=3.0,
        comparison=">",
        delta=0.01,
        fired=(isinstance(net_debt_to_ebitda, (int, float)) and net_debt_to_ebitda > 3.0)
        or leverage_unservicable,
        reason=(
            "Net debt with no positive EBITDA to service it: the ratio is undefined, "
            "which is the most levered case rather than the least."
            if leverage_unservicable
            else "Net debt/EBITDA proxy above 3x increases leverage risk."
        ),
    )

    # Interest coverage adequacy — Graham 5x minimum standard
    ic_year = _latest_common_year(facts, "operating_income", "interest_expense")
    interest_coverage_val: float | None = None
    if ic_year:
        _oi = _get_value_for_year(facts, "operating_income", ic_year)
        _ie = _get_value_for_year(facts, "interest_expense", ic_year)
        if isinstance(_oi, (int, float)) and isinstance(_ie, (int, float)) and float(_ie) > 0:
            interest_coverage_val = float(_oi) / float(_ie)

    _record_rule(
        "INTEREST_COVERAGE_WEAK",
        metric_name="interest_coverage",
        metric_value=round(float(interest_coverage_val), 4)
        if isinstance(interest_coverage_val, (int, float))
        else None,
        # The fire condition is the BAND (IC < 1.5 escalates to CRITICAL
        # instead); the trail must describe the band, not "< 3.0"
        # (audit: wacc-ic-weak-audit-trail-band).
        threshold={"low": 1.5, "high": 3.0},
        comparison="1.5 <= ic < 3.0",
        delta=0.005,
        fired=isinstance(interest_coverage_val, (int, float))
        and 1.5 <= interest_coverage_val < 3.0,
        reason="Interest coverage in the 1.5-3.0x band indicates thin debt service capacity.",
    )
    _record_rule(
        "INTEREST_COVERAGE_CRITICAL",
        metric_name="interest_coverage",
        metric_value=round(float(interest_coverage_val), 4)
        if isinstance(interest_coverage_val, (int, float))
        else None,
        threshold=1.5,
        comparison="<",
        delta=0.01,
        fired=isinstance(interest_coverage_val, (int, float)) and interest_coverage_val < 1.5,
        reason="Interest coverage below 1.5x signals critical debt distress risk.",
    )

    # The adjustments are quarter-point steps; summing them in binary floating
    # point leaves noise in the last digits (0.1 + 0.005 = 0.10500000000000001).
    # Round to a millionth so the published rate is the rate the rules chose.
    adjusted = round(adjusted, 6)
    floor_applied = False
    cap_applied = False
    if adjusted < 0.07:
        adjusted = 0.07
        floor_applied = True
    if adjusted > 0.13:
        adjusted = 0.13
        cap_applied = True

    return {
        "baseline_wacc": baseline,
        "adjusted_wacc": adjusted,
        "adjustments": adjustments,
        "rule_evaluations": rule_evaluations,
        "floor_applied": floor_applied,
        "cap_applied": cap_applied,
        "metrics": {
            "gross_margin_avg_3y": gross_margin_avg,
            "cash_conversion_avg_3y": cash_conversion_avg,
            "rnd_intensity_avg_3y": rnd_intensity_num,
            "gross_margin_stable_or_growing": gross_margin_stable_or_growing,
            "revenue_cagr_5y": revenue_cagr,
            "net_debt_to_ebitda_proxy": net_debt_to_ebitda,
        },
    }


def _discounted_owner_earnings(
    owner_earnings: float,
    shares: float,
    net_debt: float | None,
    revenue_series: list[tuple[int, float]],
    *,
    wacc: float = _WACC,
    terminal_growth: float = _TERMINAL_GROWTH,
) -> dict[str, Any]:
    flags: list[str] = []
    if shares <= 0:
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": ["SHARES_ZERO"],
            "low": None,
            "base": None,
            "high": None,
        }
    if not isinstance(net_debt, (int, float)):
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": ["NET_DEBT_UNKNOWN"],
            "low": None,
            "base": None,
            "high": None,
        }
    cagr = _revenue_cagr(revenue_series)
    if cagr is None:
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": ["REVENUE_HISTORY_SHORT"],
            "low": None,
            "base": None,
            "high": None,
        }
    if wacc <= terminal_growth:
        # Gordon denominator undefined: failing loudly beats fabricating a
        # ~1000x terminal multiple via a clamped spread (audit:
        # dcf-terminal-guard-fabricates-value; mirrors reverse_dcf FIX 3).
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": ["INVALID_DISCOUNT_ASSUMPTIONS"],
            "low": None,
            "base": None,
            "high": None,
        }
    if owner_earnings < 0:
        flags.append("NEGATIVE_OWNER_EARNINGS")

    scenarios = {
        "low": min(max(cagr * 0.50, -0.10), 0.03),
        "base": min(max(cagr * 0.75, -0.05), 0.08),
        "high": min(max(cagr * 1.00, 0.00), 0.15),
    }

    def pv_oe(growth: float) -> float:
        oe = owner_earnings
        pv = sum(oe * ((1 + growth) ** yr) / ((1 + wacc) ** yr) for yr in range(1, 6))
        terminal = (
            oe * ((1 + growth) ** 5) * (1 + terminal_growth) / max(0.001, wacc - terminal_growth)
        )
        pv += terminal / ((1 + wacc) ** 5)
        return pv

    result = {}
    for scenario, growth in scenarios.items():
        ev = pv_oe(growth)
        result[scenario] = (ev - net_debt) / shares  # negative net_debt = net cash → higher value

    # For negative owner earnings, higher growth compounds the LOSS — the
    # scenario labels must still satisfy low <= base <= high (audit:
    # dcf-negative-oe-scenario-inversion).
    ordered = sorted(result.values())
    result = {"low": ordered[0], "base": ordered[1], "high": ordered[2]}

    return {"status": "OK", "flags": flags, **result}


def _dcf_table_notes(pricing_zone_detail: dict[str, Any], dcf: dict[str, Any]) -> list[str]:
    """Flags for the dossier's DCF table row, naming a rejected raw base."""
    notes = [str(flag) for flag in (dcf.get("flags") or [])]
    raw = pricing_zone_detail.get("dcf_raw_base")
    durable = pricing_zone_detail.get("dcf_base")
    if isinstance(raw, (int, float)) and isinstance(durable, (int, float)):
        notes.append(f"durable (spike-corrected); raw ${float(raw):,.2f}")
    return notes


# EPV directive vocabulary the pre-valuation gate emits (EPV_ADJ_* in
# pre_valuation_gate.py). Anything else is refused, never valued.
_EPV_ADJUSTMENTS = frozenset({"NONE", "USE_NORMALIZED", "BLOCK"})
# Defensible range for a supplied tax rate; the same bounds dcf_lite uses. A
# rate outside it is refused, not clamped: a clamped 140% is still a wrong
# input, and a negative rate would flip the sign of NOPAT.
_TAX_RATE_FLOOR = 0.0
_TAX_RATE_CEILING = 0.50


def _applied_tax_rate(tax_rate: float | None) -> tuple[float, str]:
    """The rate a signal taxes operating income at, and where it came from.

    The issuer's normalized rate (``_normalized_tax_rate``, the one the EPV uses)
    when it is finite and within the defensible range; otherwise the same 21%
    statutory fallback, recorded as STATUTORY_DEFAULT.
    """
    if (
        tax_rate is not None
        and _finite_numeric(tax_rate)
        and _TAX_RATE_FLOOR <= tax_rate <= _TAX_RATE_CEILING
    ):
        return float(tax_rate), "ISSUER"
    return _TAX_RATE, "STATUTORY_DEFAULT"


def _epv_refusal(status: str, flag: str, **extra: Any) -> dict[str, Any]:
    """One shape for every EPV that could not be computed."""
    return {
        "status": status,
        "flags": [flag],
        "value_per_share": None,
        "avg_operating_income": None,
        **extra,
    }


def _normalized_tax_rate(
    facts: dict[str, list[tuple[int, float]]],
    *,
    n: int = 5,
) -> dict[str, Any]:
    """The issuer's own normalized tax rate for EPV, or None with a reason.

    Ratio of sums over the most recent ``n`` fiscal years that report both
    income-tax expense and pre-tax income: total tax / total pre-tax income.
    Summing first keeps one year's settlement or one-off benefit from setting
    the rate, the same reason the EPV margin is a ratio of sums. At least
    ``_MIN_YEARS`` paired years are required, the pre-tax total must be
    positive, and the result must lie in [_TAX_RATE_FLOOR, _TAX_RATE_CEILING];
    otherwise ``tax_rate`` is None and ``_epv`` falls back to the documented
    statutory rate and says so in its flags.
    """
    tax = {int(y): float(v) for y, v in facts.get("income_tax_expense") or [] if _finite_numeric(v)}
    pretax = {int(y): float(v) for y, v in facts.get("pretax_income") or [] if _finite_numeric(v)}
    years = sorted(set(tax) & set(pretax), reverse=True)[:n]
    if len(years) < _MIN_YEARS:
        return {"tax_rate": None, "reason_code": "TAX_HISTORY_SHORT", "years": sorted(years)}
    total_pretax = sum(pretax[y] for y in years)
    if total_pretax <= 0:
        return {"tax_rate": None, "reason_code": "PRETAX_INCOME_NOT_POSITIVE", "years": sorted(years)}
    rate = sum(tax[y] for y in years) / total_pretax
    if not (_TAX_RATE_FLOOR <= rate <= _TAX_RATE_CEILING):
        return {
            "tax_rate": None,
            "reason_code": "EFFECTIVE_TAX_RATE_OUT_OF_RANGE",
            "rejected_rate": rate,
            "years": sorted(years),
        }
    return {"tax_rate": rate, "reason_code": "OK", "years": sorted(years)}


# D&A concepts that already exclude amortization of acquired intangibles.
_DEPRECIATION_ONLY_CONCEPTS = frozenset({"Depreciation"})


def _da_source_concepts(
    conn: Any,
    ticker: str,
    *,
    as_of_date: str | None,
    issuer_cik: str | None,
    issuer_aliases: tuple[str, ...],
) -> dict[int, str]:
    """The XBRL concept each FY depreciation_amortization row was read from.

    Rows ingested before the concept was persisted carry none and are absent
    from the result (the EPV then keeps its subtraction, the lower reading).
    """
    try:
        _, rows = issuer_companyfacts_rows(
            conn,
            ticker,
            columns=("fiscal_year", "source_tags"),
            issuer_cik=issuer_cik,
            aliases=issuer_aliases,
            period_types=("FY",),
            line_items=("depreciation_amortization",),
            as_of_date=as_of_date,
            value_not_null=True,
            require_filed_asof=True,
        )
    except Exception:  # noqa: BLE001 - provenance must never fail a valuation
        return {}
    return {
        int(row["fiscal_year"]): str(row["source_tags"])
        for row in rows
        if row["source_tags"]
    }


def _epv_da_and_maintenance_capex(
    facts: dict[str, list[tuple[int, float]]],
    maintenance_capex_ratio: Any,
    *,
    n: int = 5,
    da_source_concepts: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Normalized D&A and maintenance capex for the EPV's EBIT adjustment.

    D&A is the mean over the newest ``n`` fiscal years of the reported
    depreciation and amortization, less any directly reported amortization of
    acquired intangibles that year (the cash-adjusted EPV variant publishes
    that add-back on its own, so the base EPV does not take it too).
    Maintenance capex is the spike-capped capex mean (``_normalize_capex``)
    times the pre-valuation gate's growth-aware maintenance ratio; with no
    usable ratio every dollar of capex is treated as maintenance (ratio 1.0,
    the reading that credits the least). Both are magnitudes. Either history
    shorter than _MIN_YEARS returns None figures with a reason, and ``_epv``
    then makes no adjustment.

    ``da_source_concepts`` maps a fiscal year to the concept its D&A came from.
    A year read from ``Depreciation`` alone already excludes intangible
    amortization, so nothing is subtracted from it a second time (a filer
    tagging Depreciation 15,200 and AmortizationOfIntangibleAssets 4,800 has
    physical D&A of 15,200, not 10,400).
    """
    da_series = [
        (int(y), abs(float(v)))
        for y, v in sorted(facts.get("depreciation_amortization") or [], reverse=True)[:n]
        if _finite_numeric(v)
    ]
    capex_series = [
        (int(y), float(v)) for y, v in (facts.get("capex") or []) if _finite_numeric(v)
    ]
    if len(da_series) < _MIN_YEARS:
        return {"depreciation_amortization": None, "maintenance_capex": None, "reason": "DA_HISTORY_SHORT"}
    if len(capex_series) < _MIN_YEARS:
        return {"depreciation_amortization": None, "maintenance_capex": None, "reason": "CAPEX_HISTORY_SHORT"}
    intangible = {
        int(y): abs(float(v))
        for y, v in facts.get("intangible_amortization") or []
        if _finite_numeric(v)
    }
    concepts = da_source_concepts or {}
    physical_da = [
        v
        if concepts.get(y) in _DEPRECIATION_ONLY_CONCEPTS
        else max(0.0, v - intangible.get(y, 0.0))
        for y, v in da_series
    ]
    if _finite_numeric(maintenance_capex_ratio) and 0.0 <= maintenance_capex_ratio <= 1.0:
        ratio = float(maintenance_capex_ratio)
        ratio_source = "GATE_GROWTH_AWARE"
    else:
        ratio = 1.0
        ratio_source = "ALL_CAPEX_MAINTENANCE_DEFAULT"
    return {
        "depreciation_amortization": sum(physical_da) / len(physical_da),
        "maintenance_capex": _normalize_capex(capex_series, n=n) * ratio,
        "maintenance_capex_ratio": ratio,
        "maintenance_capex_ratio_source": ratio_source,
        "intangible_amortization_excluded": any(
            y in intangible and concepts.get(y) not in _DEPRECIATION_ONLY_CONCEPTS
            for y, _ in da_series
        ),
        "depreciation_only_years": sorted(
            y for y, _ in da_series if concepts.get(y) in _DEPRECIATION_ONLY_CONCEPTS
        ),
        "years": sorted(y for y, _ in da_series),
        "reason": "OK",
    }


def _epv(
    operating_income_series: list[tuple[int, float]],
    net_debt: float | None,
    shares: float,
    *,
    revenue_series: list[tuple[int, float]] | None = None,
    wacc: float = _WACC,
    epv_adjustment: str = "NONE",
    normalized_earnings: float | None = None,
    tax_rate: float | None = None,
    depreciation_amortization: float | None = None,
    maintenance_capex: float | None = None,
) -> dict[str, Any]:
    """Earnings power value: a normalized MARGIN applied to CURRENT revenue.

    Definition used (Greenwald, zero growth by construction)::

        normalized margin = sum(operating income) / sum(revenue) over the years
                            that report both (revenue > 0), at least _MIN_YEARS
        normalized EBIT   = normalized margin x CURRENT revenue (the newest
                            revenue year on file), or the gate's cyclically
                            normalized earnings when that is LOWER
        adjusted EBIT     = normalized EBIT + (D&A - maintenance capex), when
                            the caller supplies both normalized figures
        NOPAT             = adjusted EBIT x (1 - tax rate); a loss is not
                            tax-shielded
        EPV(operations)   = NOPAT / WACC
        EPV(equity)       = EPV(operations) - net debt (net cash adds)
        value per share   = EPV(equity) / shares

    D&A versus maintenance capex (2026-09-29): GAAP operating income is net
    of book D&A, but the cash cost of standing still is maintenance capex.
    When both are supplied (normalized magnitudes, finite and >= 0) the
    difference is added to EBIT and flagged EPV_DA_MAINTENANCE_CAPEX_ADJUSTED;
    when either is missing no adjustment is made and
    ``da_maintenance_capex_basis`` says NOT_ADJUSTED; a negative, boolean or
    non-finite figure is refused (INVALID_DA_MAINTENANCE_CAPEX).

    Stock-based compensation (stated 2026-09-29): EXPENSED. GAAP operating
    income already deducts it, and this method adds none of it back — SBC is
    a real cost of the labour that produces the earnings. The result carries
    ``sbc_treatment`` so the choice is read, not inferred; a caller that
    feeds a series with SBC partly capitalized (the R&D-adjusted variant)
    relabels it.

    No earnings power (2026-09-29): EPV(operations) is only defined for a
    positive NOPAT. A zero or negative NOPAT, or an equity value that net
    debt takes to zero or below, returns status EPV_NEGATIVE with
    ``value_per_share`` None — never a number a consumer could read as a
    price, and never a positive value conjured from net cash sitting beside
    a loss-making business. The computed figures stay on the result
    (``epv_operations``, ``epv_equity``, and ``negative_value_per_share``,
    the non-positive per-share reading the scorecard's anomaly checks use).

    A margin window with a hole in it (fiscal years missing between the
    oldest and newest paired year) is flagged EPV_MARGIN_WINDOW_GAP and the
    missing years are listed; it is still computed from the years on file.

    Earnings power is what the business as it exists today earns at its
    through-cycle margin — not what its average past self earned. Averaging
    operating-income LEVELS (what this function did before 2026-09-02) gives a
    company that grew revenue 100 -> 500 and one that shrank 500 -> 100 at the
    same flat margin the IDENTICAL value; the shrinking company — the
    population an earnings floor exists to protect against — came out
    overstated threefold. "Current" revenue is the newest revenue year on
    file even when that year has no operating income yet: the margin is the
    historical part, the revenue is not (2026-09-28).

    The margin is a RATIO OF SUMS, the revenue-weighted through-cycle margin.
    The unweighted mean of per-year margins was measured first and rejected:
    an early-revenue year sets a margin of its own scale (one issuer in the
    live store reported operating income of -5.7 on revenue of 0.37, a margin
    of -15.5), which then multiplies today's much larger revenue.

    Tax rate: the caller supplies the issuer's normalized rate
    (``_normalized_tax_rate``). When none is supplied the documented fallback
    is the 21% US federal statutory rate (``_TAX_RATE``), flagged
    EPV_TAX_RATE_STATUTORY_DEFAULT. A supplied rate that is not a finite
    number in [0.0, 0.50] is refused (INVALID_TAX_RATE), never clamped.

    Fails closed, each with its own reason flag and ``value_per_share`` None:
    an unrecognized gate directive (UNKNOWN_EPV_ADJUSTMENT), USE_NORMALIZED
    without a finite normalized figure (EPV_NORMALIZATION_MISSING), a
    non-positive or non-finite WACC (INVALID_DISCOUNT_ASSUMPTIONS), net debt
    that is None (NET_DEBT_UNKNOWN) or boolean/NaN/infinite
    (NET_DEBT_INVALID), share counts that are not finite and positive, no
    paired revenue history (INSUFFICIENT_MARGIN_HISTORY) and a non-positive
    current revenue (CURRENT_REVENUE_NOT_POSITIVE). Without a paired revenue
    history there is no margin to normalize, and the method says so rather
    than falling back to the levels average.

    ``avg_operating_income`` carries the base actually capitalized, which is
    what the pricing zone's earnings gap and the company page read; the plain
    mean of the levels is kept beside it as ``avg_operating_income_levels`` for
    provenance, with the tax rate, WACC and years used.
    """
    if not isinstance(epv_adjustment, str) or epv_adjustment not in _EPV_ADJUSTMENTS:
        return _epv_refusal(
            "METHOD_INSUFFICIENT_DATA",
            "UNKNOWN_EPV_ADJUSTMENT",
            epv_adjustment=str(epv_adjustment),
        )
    if epv_adjustment == "BLOCK":
        return _epv_refusal("EPV_BLOCKED_SECULAR_DECLINE", "EPV_BLOCKED_SECULAR_DECLINE")
    if len(operating_income_series) < _MIN_YEARS:
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "INSUFFICIENT_OI_HISTORY")
    if not _finite_numeric(shares):
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "SHARES_INVALID")
    if shares <= 0:
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "SHARES_ZERO")
    if net_debt is None:
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "NET_DEBT_UNKNOWN")
    if not _finite_numeric(net_debt):
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "NET_DEBT_INVALID")
    if not _finite_numeric(wacc) or wacc <= 0:
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "INVALID_DISCOUNT_ASSUMPTIONS")
    flags: list[str] = []
    if tax_rate is None:
        applied_tax_rate = _TAX_RATE
        tax_rate_source = "STATUTORY_DEFAULT"
        flags.append("EPV_TAX_RATE_STATUTORY_DEFAULT")
    elif _finite_numeric(tax_rate) and _TAX_RATE_FLOOR <= tax_rate <= _TAX_RATE_CEILING:
        applied_tax_rate = float(tax_rate)
        tax_rate_source = "ISSUER"
    else:
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "INVALID_TAX_RATE")
    if epv_adjustment == "USE_NORMALIZED" and not _finite_numeric(normalized_earnings):
        # The gate asked for the cyclical normalization and none arrived. The
        # normalization can only LOWER the base, so the un-normalized figure is
        # an upper bound, not the value the gate asked for.
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "EPV_NORMALIZATION_MISSING")

    oi_by_year = {
        int(year): float(value)
        for year, value in operating_income_series
        if _finite_numeric(value)
    }
    rev_by_year = {
        int(year): float(value) for year, value in (revenue_series or []) if _finite_numeric(value)
    }
    margin_years = sorted(
        year for year in set(oi_by_year) & set(rev_by_year) if rev_by_year[year] > 0
    )
    if len(margin_years) < _MIN_YEARS:
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "INSUFFICIENT_MARGIN_HISTORY")
    normalized_margin = sum(oi_by_year[year] for year in margin_years) / sum(
        rev_by_year[year] for year in margin_years
    )
    current_revenue_year = max(rev_by_year)
    current_revenue = rev_by_year[current_revenue_year]
    if current_revenue <= 0:
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "CURRENT_REVENUE_NOT_POSITIVE")
    da_supplied = depreciation_amortization is not None and maintenance_capex is not None
    if da_supplied and not all(
        _finite_numeric(v) and v >= 0  # type: ignore[operator]
        for v in (depreciation_amortization, maintenance_capex)
    ):
        return _epv_refusal("METHOD_INSUFFICIENT_DATA", "INVALID_DA_MAINTENANCE_CAPEX")
    margin_window_missing_years = sorted(
        set(range(margin_years[0], margin_years[-1] + 1)) - set(margin_years)
    )
    if margin_window_missing_years:
        flags.append("EPV_MARGIN_WINDOW_GAP")
    oi_levels = list(oi_by_year.values())
    avg_oi_levels = sum(oi_levels) / len(oi_levels)
    avg_oi = normalized_margin * current_revenue
    if epv_adjustment == "USE_NORMALIZED":
        # A "conservative" normalization may only LOWER the denominator —
        # never raise the anchor (audit: use-normalized-feeds-cfo-median).
        if float(normalized_earnings) < avg_oi:  # type: ignore[arg-type]
            avg_oi = float(normalized_earnings)  # type: ignore[arg-type]
            flags.append("EPV_CYCLICALLY_NORMALIZED")
        else:
            flags.append("EPV_NORMALIZATION_NOT_BINDING")
    normalized_ebit = avg_oi
    da_adjustment: float | None = None
    if da_supplied:
        da_adjustment = float(depreciation_amortization) - float(maintenance_capex)  # type: ignore[arg-type]
        avg_oi = normalized_ebit + da_adjustment
        flags.append("EPV_DA_MAINTENANCE_CAPEX_ADJUSTED")
    if avg_oi > 0:
        nopat = avg_oi * (1 - applied_tax_rate)
    else:
        # No immediate full tax shield on losses (audit:
        # negative-epv-tax-shield-on-losses) — taxing a negative average
        # shrank the loss magnitude 21%.
        nopat = avg_oi
        flags.append("EPV_NO_TAX_SHIELD_ON_LOSSES")
    epv_ev = nopat / wacc
    epv_equity = epv_ev - net_debt  # negative net_debt (net cash) increases value
    value_per_share: float | None = epv_equity / shares
    negative_value_per_share: float | None = None
    status = "OK"
    if nopat <= 0:
        # No earnings power to capitalize: not a value, whatever net cash
        # sits beside it.
        status = "EPV_NEGATIVE"
        flags.append("EPV_NO_EARNINGS_POWER")
        # The equity reading when it is itself non-positive; when net cash
        # lifts it above zero, the operations reading (the loss, per share).
        negative_value_per_share = (
            epv_equity / shares if epv_equity <= 0 else epv_ev / shares
        )
        value_per_share = None
    elif epv_equity <= 0:
        status = "EPV_NEGATIVE"
        flags.append("EPV_NET_DEBT_EXCEEDS_EARNINGS_POWER")
        negative_value_per_share = epv_equity / shares
        value_per_share = None
    return {
        "status": status,
        "value_per_share": value_per_share,
        "negative_value_per_share": negative_value_per_share,
        "avg_operating_income": avg_oi,
        "avg_operating_income_levels": avg_oi_levels,
        "normalized_ebit": normalized_ebit,
        "da_maintenance_capex_adjustment": da_adjustment,
        "da_maintenance_capex_basis": "ADJUSTED" if da_supplied else "NOT_ADJUSTED",
        "depreciation_amortization": (
            float(depreciation_amortization) if da_supplied else None  # type: ignore[arg-type]
        ),
        "maintenance_capex": float(maintenance_capex) if da_supplied else None,  # type: ignore[arg-type]
        "sbc_treatment": "EXPENSED",
        "epv_operations": epv_ev,
        "epv_equity": epv_equity,
        "normalized_margin": normalized_margin,
        "current_revenue": current_revenue,
        "current_revenue_year": current_revenue_year,
        "margin_years": margin_years,
        "margin_window_missing_years": margin_window_missing_years,
        "tax_rate": applied_tax_rate,
        "tax_rate_source": tax_rate_source,
        "wacc": float(wacc),
        "flags": flags,
    }


def _graham_formula(
    net_income_series: list[tuple[int, float]],
    equity: float | None,
    shares: float | None,
) -> dict[str, Any]:
    flags: list[str] = []
    if equity is None:
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": ["EQUITY_MISSING"],
            "value_per_share": None,
            "buy_price": None,
        }
    if shares is None:
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": ["SHARES_MISSING"],
            "value_per_share": None,
            "buy_price": None,
        }
    if equity <= 0:
        return {
            "status": "GRAHAM_NOT_APPLICABLE",
            "flags": ["NEGATIVE_EQUITY"],
            "value_per_share": None,
            "buy_price": None,
        }
    if len(net_income_series) < _MIN_YEARS:
        return {
            "status": "GRAHAM_NOT_APPLICABLE",
            "flags": ["INSUFFICIENT_NI_HISTORY"],
            "value_per_share": None,
            "buy_price": None,
        }
    if shares <= 0:
        return {
            "status": "GRAHAM_NOT_APPLICABLE",
            "flags": ["SHARES_ZERO"],
            "value_per_share": None,
            "buy_price": None,
        }
    avg_ni = sum(v for _, v in net_income_series) / len(net_income_series)
    if avg_ni <= 0:
        return {
            "status": "GRAHAM_NOT_APPLICABLE",
            "flags": ["NEGATIVE_NORMALIZED_EPS"],
            "value_per_share": None,
            "buy_price": None,
        }

    normalized_eps = avg_ni / shares
    bvps = equity / shares

    ratio = bvps / normalized_eps if normalized_eps > 0 else float("inf")
    if bvps < 1.0 or ratio > 100 or ratio < 0.01:
        flags.append("GRAHAM_LOW_CONFIDENCE")

    intrinsic = math.sqrt(22.5 * normalized_eps * bvps)
    return {
        "status": "OK",
        "value_per_share": intrinsic,
        "buy_price": intrinsic * 0.67,
        "normalized_eps": normalized_eps,
        "bvps": bvps,
        "flags": flags,
    }


def _ncav(
    cash: float | None,
    revenue: float | None,
    total_liabilities: float | None,
    total_debt: float | None,
    shares: float,
    price: float | None,
    *,
    accounts_receivable: float | None = None,
    inventory: float | None = None,
    current_assets: float | None = None,
    current_liabilities: float | None = None,
    preferred_equity: float | None = None,
) -> dict[str, Any]:
    """
    NCAV with Graham liquidation haircuts when real balance sheet data is available.

    When current_assets, accounts_receivable, and inventory are present:
        liquid_current_assets = cash×1.00 + AR×0.75 + inventory×0.50
                                + (other_current×0.25)
        NCAV = liquid_current_assets − TOTAL liabilities − preferred stock
        (current_liabilities is only a flagged understating proxy when the
        total is missing — see the liability block below)

    When real data is missing, falls back to the legacy proxy:
        receivables_est = revenue × 0.10 × 0.60
        NCAV = (cash + receivables_est) − total_liabilities
    """
    flags: list[str] = []

    if cash is None:
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": ["CASH_MISSING"],
            "value_per_share": None,
            "signal": None,
        }

    # Prefer real balance sheet data with Graham liquidation haircuts
    if current_assets is not None:
        missing_current_asset_components: list[str] = []
        if accounts_receivable is None:
            missing_current_asset_components.append("AR_MISSING")
        if inventory is None:
            missing_current_asset_components.append("INVENTORY_MISSING")
        if missing_current_asset_components:
            return {
                "status": "METHOD_INSUFFICIENT_DATA",
                "flags": missing_current_asset_components,
                "value_per_share": None,
                "signal": None,
            }
        ar = accounts_receivable
        inv = inventory
        other_current = max(0.0, current_assets - cash - ar - inv)
        liquid_current_assets = cash * 1.00 + ar * 0.75 + inv * 0.50 + other_current * 0.25
        flags.append("NCAV_REAL_BALANCE_SHEET")
    else:
        # Legacy proxy fallback
        if revenue is None:
            return {
                "status": "METHOD_INSUFFICIENT_DATA",
                "flags": ["REVENUE_MISSING"],
                "value_per_share": None,
                "signal": None,
            }
        receivables_est = revenue * 0.10 * 0.6
        liquid_current_assets = cash + receivables_est
        flags.append("NCAV_PROXY_ESTIMATED")

    # Liabilities: textbook Graham NCAV subtracts TOTAL liabilities — the
    # liquidation haircuts apply to the ASSET side only, liabilities stay
    # whole (audit: ncav-current-liabilities-only, where long-term debt
    # vanished from the floor and levered names fired NCAV_NET_NET with zero
    # real NCAV). current_liabilities is only an UNDERSTATING proxy when the
    # total is missing; total_debt is the last resort.
    if total_liabilities is not None:
        liabilities = total_liabilities
    elif current_liabilities is not None:
        liabilities = current_liabilities
        flags.append("LIABILITIES_CURRENT_ONLY_PROXY")
        if current_assets is None:
            flags.append("CURRENT_LIABILITIES_WITHOUT_CURRENT_ASSETS")
    elif total_debt is not None:
        liabilities = total_debt
        flags.append("TOTAL_LIABILITIES_PROXY")
    else:
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": flags + ["LIABILITIES_MISSING"],
            "value_per_share": None,
            "signal": None,
        }

    ncav = liquid_current_assets - liabilities
    # Preferred ranks ahead of common in liquidation (matches graham_dodd's
    # NCAV definition).
    if isinstance(preferred_equity, (int, float)) and preferred_equity > 0:
        ncav -= float(preferred_equity)
        flags.append("PREFERRED_DEDUCTED")
    if shares <= 0:
        return {
            "status": "METHOD_INSUFFICIENT_DATA",
            "flags": flags + ["SHARES_ZERO"],
            "value_per_share": None,
            "signal": None,
        }

    value_per_share = ncav / shares

    if price is not None and value_per_share > price:
        signal = "NCAV_NET_NET"
    elif value_per_share > 0:
        signal = "NCAV_PARTIAL_PROTECTION"
    else:
        signal = "NCAV_NO_ASSET_FLOOR"

    return {"status": "OK", "value_per_share": value_per_share, "signal": signal, "flags": flags}


# ── scorecard, ROIC, capital structure, reverse DCF ───────────────────────────


def _classify_epv_quality(
    revenue_cagr_5y: float | None,
    revenue_cagr_3y: float | None,
) -> tuple[str, float | None]:
    """Classify EPV input quality based on revenue CAGR.

    Returns (epv_quality, cagr_used) where cagr_used is the CAGR value
    that determined the classification (5y preferred, 3y fallback).
    """
    cagr = revenue_cagr_5y if revenue_cagr_5y is not None else revenue_cagr_3y
    if cagr is None:
        return "UNKNOWN", None
    if cagr < -0.05:
        return "DETERIORATING_BASE", cagr
    if cagr < -0.03:
        return "DECLINING_BASE", cagr
    return "STABLE", cagr


def _compute_pricing_zone(
    *,
    epv_adjusted: float | None,
    dcf_base: float | None,
    current_price: float | None,
    shares: float,
    net_debt: float | None,
    adjusted_wacc: float,
    adjusted_avg_operating_income: float | None,
    revenue_latest: float | None,
    net_debt_to_ebitda_proxy: float | None,
    revenue_cagr_5y: float | None = None,
    revenue_cagr_3y: float | None = None,
    tax_rate: float | None = None,
) -> dict[str, Any]:
    zone_tax_rate, zone_tax_rate_source = _applied_tax_rate(tax_rate)
    epv_quality, epv_quality_cagr = _classify_epv_quality(revenue_cagr_5y, revenue_cagr_3y)
    _epv_quality_fields: dict[str, Any] = {
        "epv_quality": epv_quality,
        "revenue_cagr_5y": revenue_cagr_5y,
        "revenue_cagr_3y": revenue_cagr_3y,
    }
    market_cap = (
        float(current_price) * float(shares)
        if isinstance(current_price, (int, float))
        and isinstance(shares, (int, float))
        and float(shares) > 0
        else None
    )
    debt_to_market_cap = (
        float(net_debt) / float(market_cap)
        if isinstance(net_debt, (int, float))
        and isinstance(market_cap, (int, float))
        and abs(float(market_cap)) > 1e-9
        else None
    )

    def _anomaly_sub_reason() -> str:
        if (
            isinstance(adjusted_avg_operating_income, (int, float))
            and float(adjusted_avg_operating_income) < 0
        ):
            return "NEGATIVE_EARNINGS_LEGITIMATE"
        if (isinstance(debt_to_market_cap, (int, float)) and float(debt_to_market_cap) > 1.5) or (
            isinstance(net_debt_to_ebitda_proxy, (int, float))
            and float(net_debt_to_ebitda_proxy) > 3.0
        ):
            return "DEBT_OVERWHELMS_EARNINGS"
        if (
            not isinstance(shares, (int, float))
            or float(shares) <= 0
            or not isinstance(revenue_latest, (int, float))
            or float(revenue_latest) <= 0
        ):
            return "DATA_QUALITY_ISSUE"
        return "UNDETERMINED"

    negative_fields: list[str] = []
    if isinstance(epv_adjusted, (int, float)) and float(epv_adjusted) < 0:
        negative_fields.append("EPV_adjusted")
    if isinstance(dcf_base, (int, float)) and float(dcf_base) < 0:
        negative_fields.append("DCF_base")
    if negative_fields:
        result = {
            "zone": "VALUATION_ANOMALY",
            "detail": {
                "reason": f"Negative intrinsic value detected: {', '.join(negative_fields)}.",
                "anomaly_sub_reason": _anomaly_sub_reason(),
                "negative_inputs": negative_fields,
                "current_price": float(current_price)
                if isinstance(current_price, (int, float))
                else current_price,
                "epv_adjusted": float(epv_adjusted)
                if isinstance(epv_adjusted, (int, float))
                else epv_adjusted,
                "dcf_base": float(dcf_base) if isinstance(dcf_base, (int, float)) else dcf_base,
                "adjusted_avg_operating_income": (
                    float(adjusted_avg_operating_income)
                    if isinstance(adjusted_avg_operating_income, (int, float))
                    else adjusted_avg_operating_income
                ),
                "revenue_latest": float(revenue_latest)
                if isinstance(revenue_latest, (int, float))
                else revenue_latest,
                "shares_outstanding": float(shares) if isinstance(shares, (int, float)) else shares,
                "net_debt": float(net_debt) if isinstance(net_debt, (int, float)) else net_debt,
                "market_cap": float(market_cap)
                if isinstance(market_cap, (int, float))
                else market_cap,
                "net_debt_to_market_cap": (
                    float(debt_to_market_cap)
                    if isinstance(debt_to_market_cap, (int, float))
                    else debt_to_market_cap
                ),
                "net_debt_to_ebitda_proxy": (
                    float(net_debt_to_ebitda_proxy)
                    if isinstance(net_debt_to_ebitda_proxy, (int, float))
                    else net_debt_to_ebitda_proxy
                ),
            },
        }
    elif not isinstance(net_debt, (int, float)):
        result = {
            "zone": "INSUFFICIENT_DATA",
            "detail": {
                "reason": "Net debt unavailable.",
                "current_price": float(current_price)
                if isinstance(current_price, (int, float))
                else current_price,
                "epv_adjusted": float(epv_adjusted)
                if isinstance(epv_adjusted, (int, float))
                else epv_adjusted,
                "dcf_base": float(dcf_base) if isinstance(dcf_base, (int, float)) else dcf_base,
                "net_debt": net_debt,
            },
        }
    elif not isinstance(current_price, (int, float)) or not isinstance(epv_adjusted, (int, float)):
        result = {
            "zone": "INSUFFICIENT_DATA",
            "detail": {
                "reason": "Current price or adjusted EPV unavailable.",
                "current_price": current_price,
                "epv_adjusted": epv_adjusted,
                "dcf_base": dcf_base,
            },
        }
    elif float(current_price) < float(epv_adjusted):
        result = {
            "zone": "MARGIN_OF_SAFETY",
            "detail": {
                # CONVENTION (FIX 6): TEXTBOOK margin-of-safety,
                #   mos = (intrinsic - price) / intrinsic
                # (the fraction of intrinsic value NOT paid for at the current price).
                # This differs from the upside-ratio convention (intrinsic/price - 1)
                # used in graham_dodd / intrinsic_discipline under the same name:
                # for intrinsic=150, price=100 this yields 0.333..., not 0.50.
                "margin_of_safety_vs_epv_adjusted": (float(epv_adjusted) - float(current_price))
                / abs(float(epv_adjusted)),
                "current_price": float(current_price),
                "epv_adjusted": float(epv_adjusted),
                "dcf_base": float(dcf_base) if isinstance(dcf_base, (int, float)) else None,
            },
        }
    elif isinstance(dcf_base, (int, float)) and float(current_price) <= float(dcf_base):
        required_equity_value = float(current_price) * float(shares)
        required_enterprise_value = required_equity_value + float(net_debt)
        required_nopat = required_enterprise_value * float(adjusted_wacc)
        required_avg_operating_income = required_nopat / (1 - zone_tax_rate)
        current_avg = (
            float(adjusted_avg_operating_income)
            if isinstance(adjusted_avg_operating_income, (int, float))
            else None
        )
        earnings_gap = (
            required_avg_operating_income - current_avg if current_avg is not None else None
        )
        result = {
            "zone": "GROWTH_DEPENDENT",
            "detail": {
                "current_price": float(current_price),
                "epv_adjusted": float(epv_adjusted),
                "dcf_base": float(dcf_base),
                "required_avg_operating_income": required_avg_operating_income,
                "current_adjusted_avg_operating_income": current_avg,
                "earnings_gap": earnings_gap,
                "tax_rate": zone_tax_rate,
                "tax_rate_source": zone_tax_rate_source,
                "earnings_gap_pct": (
                    earnings_gap / abs(current_avg)
                    if isinstance(earnings_gap, (int, float))
                    and isinstance(current_avg, (int, float))
                    and abs(current_avg) > 1e-9
                    else None
                ),
            },
        }
    elif isinstance(dcf_base, (int, float)):
        result = {
            "zone": "SPECULATIVE_PREMIUM",
            "detail": {
                "current_price": float(current_price),
                "epv_adjusted": float(epv_adjusted),
                "dcf_base": float(dcf_base),
                "premium_to_dcf_base": (float(current_price) - float(dcf_base))
                / abs(float(dcf_base)),
            },
        }
    else:
        result = {
            "zone": "INSUFFICIENT_DATA",
            "detail": {
                "reason": "DCF base unavailable.",
                "current_price": float(current_price),
                "epv_adjusted": float(epv_adjusted),
                "dcf_base": dcf_base,
            },
        }

    # Inject EPV quality annotation into every pricing zone detail dict
    result["detail"].update(_epv_quality_fields)

    # Warn when MOS signal sits on a declining revenue base
    if result["zone"] == "MARGIN_OF_SAFETY" and epv_quality in (
        "DECLINING_BASE",
        "DETERIORATING_BASE",
    ):
        logger.warning(
            "EPV_QUALITY [%s]: MOS signal on declining revenue base (CAGR: %.1f%%)",
            epv_quality,
            (epv_quality_cagr or 0.0) * 100,
        )

    return result


def _classify_moat_strength(
    *,
    earnings_quality: str | None = None,
    revenue_trend_class: str | None = None,
    epv_quality: str | None = None,
    allocation_grade: str | None = None,
    returns_persistence_class: str | None = None,
    capital_allocation_class: str | None = None,
) -> dict[str, Any]:
    """Classify moat strength from available quality signals.

    Uses a scoring system: each available signal contributes +2/+1/-1/-2.
    Missing and UNKNOWN signals are skipped (neither penalized nor counted).
    With no signal counted there is no evidence either way, and the class is
    MOAT_UNKNOWN: a score of 0 from nothing used to read WEAK_MOAT, which
    could label an undervalued name VALUE_TRAP_RISK on no evidence.
    """
    score = 0
    counted = 0
    detail: dict[str, Any] = {}

    def _score(name: str, value: str | None, mapping: dict[str, int]) -> None:
        nonlocal score, counted
        if value is None or str(value).upper() == "UNKNOWN":
            detail[name] = {"value": value, "contribution": 0, "counted": False}
            return
        contribution = mapping.get(value, 0)
        score += contribution
        counted += 1
        detail[name] = {"value": value, "contribution": contribution, "counted": True}

    _score(
        "earnings_quality",
        earnings_quality,
        {
            "HIGH": 2,
            "MODERATE": 1,
            "LOW": -2,
        },
    )
    _score(
        "revenue_trend_class",
        revenue_trend_class,
        {
            "GROWING": 2,
            "FLAT": 1,
            "DECLINING": -1,
            "SECULAR_DECLINE": -2,
            "VOLATILE": 0,
        },
    )
    _score(
        "epv_quality",
        epv_quality,
        {
            "STABLE": 2,
            "DECLINING_BASE": -1,
            "DETERIORATING_BASE": -2,
        },
    )
    _score(
        "allocation_grade",
        allocation_grade,
        {
            "A": 2,
            "B": 1,
            "C": 0,
            "F": -2,
        },
    )
    _score(
        "returns_persistence_class",
        returns_persistence_class,
        {
            "HIGH_RETURNS_PERSISTENCE": 2,
            "MODERATE_RETURNS_PERSISTENCE": 1,
            "LOW_RETURNS_PERSISTENCE": -1,
        },
    )
    _score(
        "capital_allocation_class",
        capital_allocation_class,
        {
            "OWNER_FRIENDLY_DISCIPLINED": 2,
            "MIXED_CAPITAL_ALLOCATION": 1,
            "OWNER_DILUTIVE_OR_DESTRUCTIVE": -2,
        },
    )

    if counted == 0:
        moat_class = "MOAT_UNKNOWN"
    elif score >= 6:
        moat_class = "STRONG_MOAT"
    elif score >= 3:
        moat_class = "MODERATE_MOAT"
    elif score >= 0:
        moat_class = "WEAK_MOAT"
    else:
        moat_class = "NO_MOAT"

    return {
        "moat_class": moat_class,
        "moat_score": score,
        "signals_counted": counted,
        "signal_detail": detail,
    }


def _compute_downside_scenario(
    *,
    owner_earnings: float | None,
    shares: float,
    net_debt: float | None,
    revenue_series: list[tuple[int, float]],
    operating_income_series: list[tuple[int, float]],
    wacc: float,
    base_case_dcf: float | None = None,
    current_price: float | None = None,
    tax_rate: float | None = None,
    is_reit: bool = False,
) -> dict[str, Any]:
    """Compute bear-case DCF and EPV with stressed assumptions.

    Bear case: revenue growth halved, terminal growth 0%, and the EPV on the
    base EPV's basis — a margin times CURRENT revenue — at the 5-year LOW
    operating margin (not the lowest operating-income level, which for a
    grower is an old, small year). The base EPV's refusals carry over (2026-09-29):
    no bear EPV for a REIT (``is_reit``; its operating income is struck after
    real-estate depreciation), and a non-positive bear NOPAT is no earnings
    power — its bear EPV is at most zero, never a positive value made of net
    cash beside a loss (BEAR_EPV_NO_EARNINGS_POWER).
    The bear EPV taxes a positive low year at ``tax_rate``, the issuer's own
    normalized rate the base EPV uses (``_normalized_tax_rate``); None falls
    back to the 21% statutory rate, as the EPV does, and the assumptions say
    which. A supplied rate outside [0, 0.50] or not finite leaves the bear
    EPV unset (BEAR_EPV_INVALID_TAX_RATE) rather than taxing at a wrong rate.
    """
    flags: list[str] = []
    if tax_rate is None:
        bear_tax_rate: float | None = _TAX_RATE
        bear_tax_rate_source = "STATUTORY_DEFAULT"
    elif _finite_numeric(tax_rate) and _TAX_RATE_FLOOR <= tax_rate <= _TAX_RATE_CEILING:
        bear_tax_rate = float(tax_rate)
        bear_tax_rate_source = "ISSUER"
    else:
        bear_tax_rate = None
        bear_tax_rate_source = "INVALID"

    if (
        not isinstance(owner_earnings, (int, float))
        or not revenue_series
        or not operating_income_series
        # A NaN share count passed ``shares <= 0`` (every comparison with NaN
        # is False) and turned the bear values into NaN.
        or not _finite_numeric(shares)
        or shares <= 0
        or not _finite_numeric(net_debt)
    ):
        return {
            "bear_case_dcf": None,
            "bear_case_epv": None,
            "bear_case_intrinsic": None,
            "downside_risk_class": "UNKNOWN",
            "flags": [],
            "assumptions": {},
        }

    # Bear-case revenue growth: historical CAGR halved
    cagr = _revenue_cagr(revenue_series)
    bear_growth = (cagr * 0.5) if isinstance(cagr, (int, float)) else 0.0
    # Floor AND cap, matching the base DCF's own "low" scenario
    # (min(max(cagr * 0.50, -0.10), 0.03)). With the floor alone a 50%-a-year
    # grower's "bear" case projected +25% for five years and came out 61% above
    # the base case and 22% above the bull case, and its downside class read
    # LIMITED — the SEVERE reading the synthesis prompt is required to discuss
    # could never be reached for exactly the names that most need it.
    bear_growth = min(max(bear_growth, -0.10), 0.03)

    # Bear-case terminal growth: 0%
    bear_terminal = 0.0

    # Bear-case DCF (same Gordon-denominator guard as the base DCF)
    def _pv_bear(oe: float, growth: float) -> float:
        pv = sum(oe * ((1 + growth) ** yr) / ((1 + wacc) ** yr) for yr in range(1, 6))
        terminal = oe * ((1 + growth) ** 5) * (1 + bear_terminal) / (wacc - bear_terminal)
        pv += terminal / ((1 + wacc) ** 5)
        return pv

    if wacc <= bear_terminal:
        bear_case_dcf = None
    else:
        bear_ev = _pv_bear(float(owner_earnings), bear_growth)
        bear_case_dcf = (bear_ev - net_debt) / shares
        # A downside case cannot be worth more than the central one. The
        # stresses above (halved growth, zero terminal growth) lower value only
        # when owner earnings are positive; on a loss they shrink the loss
        # being capitalized, and the "bear" case came out above the base.
        # Bound it by the base case the caller published.
        if (
            _finite_numeric(base_case_dcf)
            and bear_case_dcf > float(base_case_dcf)  # type: ignore[arg-type]
        ):
            bear_case_dcf = float(base_case_dcf)  # type: ignore[arg-type]
            flags.append("BEAR_CASE_BOUNDED_BY_BASE")

    # Bear-case EPV: the 5-year LOW operating margin x CURRENT revenue.
    rev_by_year = {
        int(y): float(v) for y, v in revenue_series if _finite_numeric(v)
    }
    low_margin_years = sorted(
        int(y)
        for y, v in operating_income_series
        if _finite_numeric(v) and rev_by_year.get(int(y), 0.0) > 0
    )
    current_revenue = rev_by_year[max(rev_by_year)] if rev_by_year else None
    low_margin: float | None = None
    if is_reit:
        flags.append("BEAR_EPV_NOT_APPLICABLE_REIT")
        bear_case_epv = None
    elif low_margin_years and bear_tax_rate is None:
        flags.append("BEAR_EPV_INVALID_TAX_RATE")
        bear_case_epv = None
    elif (
        low_margin_years
        and bear_tax_rate is not None
        and current_revenue is not None
        and current_revenue > 0
        and wacc > 0
    ):
        oi_by_year = {int(y): float(v) for y, v in operating_income_series if _finite_numeric(v)}
        low_margin = min(oi_by_year[y] / rev_by_year[y] for y in low_margin_years)
        low_ebit = low_margin * current_revenue
        # No immediate full tax shield on losses — the same asymmetry _epv
        # applies. Taxing a -100 low year turned it into -79 and made the bear
        # case 21% less bad on a tax benefit the company may never realise.
        nopat = low_ebit * (1 - bear_tax_rate) if low_ebit > 0 else low_ebit
        bear_epv_ev = nopat / wacc
        bear_case_epv = (bear_epv_ev - net_debt) / shares
        if nopat <= 0:
            # No earnings power: net cash beside a loss is not a bear-case
            # value (the base EPV refuses the same case, EPV_NEGATIVE).
            bear_case_epv = min(0.0, bear_case_epv)
            flags.append("BEAR_EPV_NO_EARNINGS_POWER")
    else:
        bear_case_epv = None

    # Bear-case intrinsic: min of both (most conservative)
    bear_values = [v for v in [bear_case_dcf, bear_case_epv] if isinstance(v, (int, float))]
    bear_case_intrinsic = min(bear_values) if bear_values else None

    # Downside risk classification.
    # Branches are mutually exclusive: once the UNKNOWN guard passes,
    # bear_case_intrinsic is a number and base_case_dcf > 0, so the three
    # remaining cases (<= 0, > 20% of base, in (0, 20%]) partition the line.
    # (The previous trailing `else: SEVERE` was unreachable — FIX 4.)
    if (
        bear_case_intrinsic is None
        or not isinstance(base_case_dcf, (int, float))
        or base_case_dcf <= 0
    ):
        downside_risk_class = "UNKNOWN"
    elif bear_case_intrinsic <= 0:
        downside_risk_class = "SEVERE"
    elif bear_case_intrinsic > base_case_dcf * 0.20:
        downside_risk_class = "LIMITED"
    else:
        # 0 < bear_case_intrinsic <= 20% of base_case_dcf
        downside_risk_class = "MODERATE"

    # DOWNSIDE_VULNERABLE: price between bear and base
    if (
        isinstance(current_price, (int, float))
        and isinstance(bear_case_intrinsic, (int, float))
        and isinstance(base_case_dcf, (int, float))
        and bear_case_intrinsic < current_price < base_case_dcf
    ):
        flags.append("DOWNSIDE_VULNERABLE")

    return {
        "bear_case_dcf": round(bear_case_dcf, 2)
        if isinstance(bear_case_dcf, (int, float))
        else None,
        "bear_case_epv": round(bear_case_epv, 2)
        if isinstance(bear_case_epv, (int, float))
        else None,
        "bear_case_intrinsic": round(bear_case_intrinsic, 2)
        if isinstance(bear_case_intrinsic, (int, float))
        else None,
        "downside_risk_class": downside_risk_class,
        "flags": flags,
        "assumptions": {
            "revenue_growth_used": round(bear_growth, 4),
            "terminal_growth": bear_terminal,
            "operating_income_low": min(
                (float(v) for _, v in operating_income_series if _finite_numeric(v)),
                default=None,
            ),
            "operating_margin_low": low_margin if low_margin is not None else None,
            "current_revenue": current_revenue,
            "tax_rate": bear_tax_rate,
            "tax_rate_source": bear_tax_rate_source,
        },
    }


def _margin_of_safety_scorecard(
    methods: dict,
    price: float | None,
    *,
    shares: float = 0.0,
    net_debt: float | None = None,
    wacc_detail: dict[str, Any] | None = None,
    revenue_latest: float | None = None,
    net_debt_to_ebitda_proxy: float | None = None,
    revenue_cagr_5y: float | None = None,
    revenue_cagr_3y: float | None = None,
    gate_action: str = "PROCEED",
    mos_threshold_widening: float = 0.0,
    quality_ctx: dict[str, Any] | None = None,
    dcf_base_override: float | None = None,
    tax_rate: float | None = None,
) -> dict[str, Any]:
    """
    Requires >= 2 earnings methods (DCF, EPV, Graham) with valid values.
    NCAV excluded from the >= 2 threshold — it is supplementary.
    GROWTH_DEPENDENT: price exceeds ALL earnings-method intrinsic values (stricter than
    spec's EPV+DCF-only definition — intentional; Graham discount would be ambiguous).
    BALANCE_SHEET_DRIVEN: either NCAV_NET_NET or NCAV_PARTIAL_PROTECTION per spec.
    """
    if gate_action == "BLOCK":
        return {
            "signal": "INSUFFICIENT_QUALITY",
            "stance": "AVOID",
            "discounts": {},
            "pricing_zone": "INSUFFICIENT_QUALITY",
            "pricing_zone_detail": {"gate_action": "BLOCK"},
            "gaap_signal": "INSUFFICIENT_QUALITY",
            "adjusted_signal": "INSUFFICIENT_QUALITY",
        }

    ncav_signal = (methods.get("ncav") or {}).get("signal")

    def _method_value(name: str) -> float | None:
        payload = methods.get(name) or {}
        if payload.get("status") not in ("OK", "EPV_NEGATIVE"):
            return None
        value = payload.get("base") if name.startswith("dcf") else payload.get("value_per_share")
        if value is None and payload.get("status") == "EPV_NEGATIVE":
            # EPV publishes no value without earnings power; its non-positive
            # reading still marks the method negative for the anomaly checks.
            value = payload.get("negative_value_per_share")
        return float(value) if isinstance(value, (int, float)) else None

    def _signal_for_ivs(earnings_ivs: dict[str, float]) -> tuple[str, dict[str, float]]:
        if len(earnings_ivs) < 2:
            return "INSUFFICIENT_DATA", {}
        discounts: dict[str, float] = {}
        positive_discounts: dict[str, float] = {}
        nonpositive_names = [name for name, iv in earnings_ivs.items() if iv <= 0]
        if price is not None and price > 0:
            for name, iv in earnings_ivs.items():
                if iv > 0:
                    discounts[name] = (iv - price) / iv
                    positive_discounts[name] = discounts[name]
                else:
                    discounts[name] = -1.0
        if price is None or not discounts:
            return "INSUFFICIENT_DATA", discounts
        if positive_discounts:
            discount_vals = list(positive_discounts.values())
        else:
            discount_vals = []
        deep_threshold = 0.33 + mos_threshold_widening
        mid_threshold = 0.15 + mos_threshold_widening
        deep_count = sum(1 for d in discount_vals if d > deep_threshold)
        mid_count = sum(1 for d in discount_vals if mid_threshold < d <= deep_threshold)
        all_overvalued = (
            bool(discount_vals)
            and all(d < 0 for d in discount_vals)
            and len(positive_discounts) + len(nonpositive_names) >= 2
        ) or (not positive_discounts and len(nonpositive_names) >= 2 and price > 0)
        if deep_count >= 2:
            return "DEEP_VALUE", discounts
        if deep_count + mid_count >= 2:
            return "UNDERVALUED", discounts
        if all_overvalued:
            return "OVERVALUED", discounts
        return "FAIRLY_VALUED", discounts

    def _signal_to_stance(signal: str) -> str:
        if signal in ("DEEP_VALUE", "UNDERVALUED"):
            return "INVESTIGATE"
        if signal == "OVERVALUED":
            return "AVOID"
        if signal == "FAIRLY_VALUED":
            return "WATCH"
        return "INSUFFICIENT_DATA"

    gaap_ivs: dict[str, float] = {}
    for name in ("dcf", "epv", "graham"):
        value = _method_value(name)
        if value is not None:
            gaap_ivs[name] = value

    adjusted_ivs: dict[str, float] = {}
    dcf_adjusted = _method_value("dcf_adjusted")
    epv_adjusted = _method_value("epv_adjusted")
    graham_value = _method_value("graham")
    if dcf_adjusted is not None:
        adjusted_ivs["dcf"] = dcf_adjusted
    if epv_adjusted is not None:
        adjusted_ivs["epv"] = epv_adjusted
    if graham_value is not None:
        adjusted_ivs["graham"] = graham_value

    earnings_ivs = dict(gaap_ivs)
    if dcf_adjusted is not None:
        earnings_ivs["dcf"] = max(earnings_ivs.get("dcf", dcf_adjusted), dcf_adjusted)
    if epv_adjusted is not None:
        earnings_ivs["epv"] = max(earnings_ivs.get("epv", epv_adjusted), epv_adjusted)

    legacy_signal, discounts = _signal_for_ivs(earnings_ivs)
    gaap_signal, gaap_discounts = _signal_for_ivs(gaap_ivs)
    adjusted_signal, adjusted_discounts = _signal_for_ivs(adjusted_ivs)

    tech_valuation_divergence = None
    tech_valuation_divergence_flag = None
    gaap_anchor = max(
        [value for key, value in gaap_ivs.items() if key in {"dcf", "epv"}], default=None
    )
    adjusted_anchor = max(
        [value for key, value in adjusted_ivs.items() if key in {"dcf", "epv"}], default=None
    )
    divergence_diagnostics = {
        "gaap_anchor": gaap_anchor,
        "adjusted_anchor": adjusted_anchor,
        "raw_divergence": None,
        "capped_divergence": None,
        "status": None,
    }
    if gaap_anchor not in (None, 0) and adjusted_anchor is not None:
        raw_divergence = (adjusted_anchor - gaap_anchor) / abs(gaap_anchor)
        divergence_diagnostics["raw_divergence"] = raw_divergence
        if abs(adjusted_anchor) > (_MAX_ADJUSTMENT_IMPLAUSIBLE_RATIO * abs(gaap_anchor)):
            tech_valuation_divergence_flag = _FLAG_ADJUSTMENT_IMPLAUSIBLE
            divergence_diagnostics["status"] = _FLAG_ADJUSTMENT_IMPLAUSIBLE
            logger.warning(
                "tech_adjustment_implausible",
                extra={
                    "gaap_anchor": gaap_anchor,
                    "adjusted_anchor": adjusted_anchor,
                    "raw_divergence": raw_divergence,
                },
            )
        else:
            bounded = max(
                -_MAX_TECH_DIVERGENCE_RATIO, min(_MAX_TECH_DIVERGENCE_RATIO, raw_divergence)
            )
            tech_valuation_divergence = bounded
            divergence_diagnostics["capped_divergence"] = bounded
            if bounded != raw_divergence:
                tech_valuation_divergence_flag = _FLAG_VALUATION_DIVERGENCE_CAPPED
                divergence_diagnostics["status"] = _FLAG_VALUATION_DIVERGENCE_CAPPED
            elif abs(bounded) > 0.30:
                tech_valuation_divergence_flag = "HIGH_DIVERGENCE"
                divergence_diagnostics["status"] = "HIGH_DIVERGENCE"
            else:
                divergence_diagnostics["status"] = "OK"

    # Type — BALANCE_SHEET_DRIVEN: either NCAV signal triggers per spec
    if ncav_signal in ("NCAV_NET_NET", "NCAV_PARTIAL_PROTECTION"):
        mos_type = "BALANCE_SHEET_DRIVEN"
    elif legacy_signal == "OVERVALUED":
        mos_type = "GROWTH_DEPENDENT"
    else:
        mos_type = "EARNINGS_DRIVEN"

    # dcf_base_override carries the durable (spike-corrected) DCF when a
    # non-recurring revenue spike was detected — the zone, the persisted
    # pzd["dcf_base"], live packets and the backtest anchor all inherit it
    # (audit: dcf-anchor-ignores-durable-spike-correction).
    graham_value_per_share = _method_value("graham")
    ncav_value_per_share = _method_value("ncav")
    # ``is None``, not ``or``: an adjusted EPV of exactly 0.0 is a reading, and
    # ``or`` swapped it for the unadjusted EPV.
    epv_adjusted_value = _method_value("epv_adjusted")
    pricing_zone = _compute_pricing_zone(
        epv_adjusted=(
            epv_adjusted_value if epv_adjusted_value is not None else _method_value("epv")
        ),
        dcf_base=(
            float(dcf_base_override)
            if isinstance(dcf_base_override, (int, float))
            else _method_value("dcf")
        ),
        current_price=price,
        shares=shares,
        net_debt=net_debt,
        adjusted_wacc=float((wacc_detail or {}).get("adjusted_wacc") or _WACC),
        adjusted_avg_operating_income=(
            ((methods.get("epv_adjusted") or {}).get("avg_operating_income"))
            if isinstance(
                (methods.get("epv_adjusted") or {}).get("avg_operating_income"), (int, float)
            )
            else ((methods.get("epv") or {}).get("avg_operating_income"))
        ),
        revenue_latest=revenue_latest,
        net_debt_to_ebitda_proxy=net_debt_to_ebitda_proxy,
        revenue_cagr_5y=revenue_cagr_5y,
        revenue_cagr_3y=revenue_cagr_3y,
        tax_rate=tax_rate,
    )
    _pz_detail = pricing_zone.get("detail")
    if isinstance(_pz_detail, dict):
        _pz_detail["graham_value_per_share"] = graham_value_per_share
        _pz_detail["ncav_value_per_share"] = ncav_value_per_share
    zone = str(pricing_zone.get("zone") or "INSUFFICIENT_DATA")
    signal = {
        "MARGIN_OF_SAFETY": "BUY",
        "GROWTH_DEPENDENT": "HOLD",
        "SPECULATIVE_PREMIUM": "PASS",
        "VALUATION_ANOMALY": "VALUATION_ANOMALY",
        "INSUFFICIENT_DATA": "INSUFFICIENT_DATA",
    }.get(zone, "INSUFFICIENT_DATA")
    if signal == "BUY":
        # A 0.01% margin must not render as BUY: require the deploy discount
        # (25% textbook MoS vs the no-growth value) before the zone action
        # reads as a buy (audit: deploy-inside-growth-dependent-zone).
        _zone_mos = (pricing_zone.get("detail") or {}).get("margin_of_safety_vs_epv_adjusted")
        if not (isinstance(_zone_mos, (int, float)) and float(_zone_mos) >= 0.25):
            signal = "HOLD"

    # ── Moat-valuation coupling ────────────────────────────────────────────
    moat_result: dict[str, Any] = {}
    signal_context = legacy_signal
    if isinstance(quality_ctx, dict):
        moat_result = _classify_moat_strength(
            earnings_quality=quality_ctx.get("earnings_quality"),
            revenue_trend_class=quality_ctx.get("revenue_trend_class"),
            epv_quality=quality_ctx.get("epv_quality"),
            allocation_grade=quality_ctx.get("allocation_grade"),
        )
        moat_class = moat_result.get("moat_class", "")
        if legacy_signal == "OVERVALUED" and moat_class == "STRONG_MOAT":
            signal_context = "PREMIUM_JUSTIFIED"
        elif legacy_signal in ("UNDERVALUED", "DEEP_VALUE") and moat_class in (
            "WEAK_MOAT",
            "NO_MOAT",
        ):
            signal_context = "VALUE_TRAP_RISK"
        elif zone == "MARGIN_OF_SAFETY" and moat_class in ("WEAK_MOAT", "NO_MOAT"):
            # MARGIN_OF_SAFETY is a pricing-ZONE name. _signal_for_ivs only ever
            # returns INSUFFICIENT_DATA / DEEP_VALUE / UNDERVALUED / OVERVALUED /
            # FAIRLY_VALUED, so testing legacy_signal for it made this branch
            # dead and the weak-moat caution never reached the dossier.
            signal_context = "MOS_REQUIRES_DEEP_DISCOUNT"

    return {
        "signal": signal,
        "legacy_signal": legacy_signal,
        "type": mos_type,
        "discounts": discounts,
        "track_comparison": {
            "gaap_signal": gaap_signal,
            "gaap_discounts": gaap_discounts,
            "gaap_stance": _signal_to_stance(gaap_signal),
            "adjusted_signal": adjusted_signal,
            "adjusted_discounts": adjusted_discounts,
            "adjusted_stance": _signal_to_stance(adjusted_signal),
            "stance_differs": (
                gaap_signal != "INSUFFICIENT_DATA"
                and adjusted_signal != "INSUFFICIENT_DATA"
                and _signal_to_stance(gaap_signal) != _signal_to_stance(adjusted_signal)
            ),
        },
        "pricing_zone": zone,
        "pricing_zone_detail": pricing_zone.get("detail")
        if isinstance(pricing_zone.get("detail"), dict)
        else {},
        "tech_valuation_divergence": tech_valuation_divergence,
        "tech_valuation_divergence_flag": tech_valuation_divergence_flag,
        "tech_valuation_divergence_diagnostics": divergence_diagnostics,
        "moat_strength": moat_result,
        "signal_context": signal_context,
    }


def _roic_signal(facts: dict) -> dict[str, Any]:
    oi_series = _n_years(facts, "operating_income", n=5)
    td_series = _n_years(facts, "total_debt", n=5)
    eq_series = _n_years(facts, "equity", n=5)

    if len(oi_series) < _MIN_YEARS or not td_series or not eq_series:
        return {"status": "ROIC_INSUFFICIENT_DATA"}

    avg_oi = sum(v for _, v in oi_series) / len(oi_series)
    avg_ic = sum(v for _, v in td_series) / len(td_series) + sum(v for _, v in eq_series) / len(
        eq_series
    )
    if avg_ic <= 0:
        return {"status": "ROIC_INSUFFICIENT_DATA"}

    # The hurdle is an after-tax rate, so the return must be after tax too: taxed
    # at the issuer's normalized rate the earnings-power value uses (21% fallback,
    # recorded; see _epv), and a loss carries no tax shield (the
    # pre-tax numerator graded 9.5% after tax as clearing a 10% WACC).
    roic_tax_rate, roic_tax_rate_source = _applied_tax_rate(_normalized_tax_rate(facts)["tax_rate"])
    nopat = avg_oi * (1 - roic_tax_rate) if avg_oi > 0 else avg_oi
    roic = nopat / avg_ic
    ratio = roic / _WACC

    if ratio > 1.5:
        signal = "ROIC_STRONG"
    elif ratio >= 1.0:
        signal = "ROIC_MODEST"
    elif ratio >= 0.8:
        signal = "ROIC_MARGINAL"
    else:
        signal = "ROIC_DESTROYING"

    return {
        "status": "OK",
        "roic_proxy": roic,
        "roic_wacc_ratio": ratio,
        "signal": signal,
        "tax_rate": roic_tax_rate,
        "tax_rate_source": roic_tax_rate_source,
    }


def _capital_structure_health(facts: dict) -> dict[str, Any]:
    """Each metric uses its own latest available year — no requirement to share a common year."""
    # Interest coverage: latest year where both operating_income and interest_expense exist
    ic_year = _latest_common_year(facts, "operating_income", "interest_expense")
    if ic_year:
        oi = _get_value_for_year(facts, "operating_income", ic_year)
        interest = _get_value_for_year(facts, "interest_expense", ic_year)
        interest_coverage: Any = (
            (oi / interest)
            if (interest and interest > 0 and oi is not None)
            else "INTEREST_EXPENSE_UNKNOWN"
        )
    else:
        interest_coverage = "INTEREST_EXPENSE_UNKNOWN"

    # Debt read from a year well behind the latest balance sheet is refused, the
    # same rule net debt uses: the ratio is unknown, never an old year's ratio.
    flags: list[str] = []

    def _current_debt_year(year: int | None) -> int | None:
        if year and _debt_year_is_stale(facts, year):
            if "DEBT_STALE_YEAR" not in flags:
                flags.append("DEBT_STALE_YEAR")
            return None
        return year

    # Debt/Equity: latest year where both exist
    de_year = _current_debt_year(_latest_common_year(facts, "total_debt", "equity"))
    if de_year:
        td = _get_value_for_year(facts, "total_debt", de_year)
        eq = _get_value_for_year(facts, "equity", de_year)
        de_ratio: Any = (
            float(td) / float(eq)
            if _finite_numeric(td) and _finite_numeric(eq) and float(eq) > 0
            else None
        )
    else:
        de_ratio = None

    # Cash coverage: latest year where both total_debt and cash exist. Cash
    # includes short-term investments, the same cash the net-debt/EBITDA line
    # beside it counts (the two used different cash until 2026-09-29).
    cc_year = _current_debt_year(_latest_common_year(facts, "total_debt", "cash"))
    if cc_year:
        td2 = _get_value_for_year(facts, "total_debt", cc_year)
        cash = _cash_like_for_year(facts, cc_year)
        if _finite_numeric(td2) and _finite_numeric(cash):
            cash_coverage: Any = "TOTAL_DEBT_ZERO" if float(td2) == 0 else float(cash) / float(td2)
        else:
            cash_coverage = None
    else:
        cash_coverage = None

    # Net debt to EBITDA: use real D&A when available
    nde_year = _current_debt_year(
        _latest_common_year(facts, "total_debt", "cash", "operating_income")
    )
    net_debt_to_ebitda: float | None = None
    ebitda_method = "OI_PROXY"
    if nde_year:
        nde_debt = _get_value_for_year(facts, "total_debt", nde_year)
        # Net debt counts short-term investments as cash, as the bridge does.
        nde_cash = _cash_like_for_year(facts, nde_year)
        nde_oi = _get_value_for_year(facts, "operating_income", nde_year)
        if _finite_numeric(nde_debt) and _finite_numeric(nde_cash) and _finite_numeric(nde_oi):
            nde_da = _get_value_for_year(facts, "depreciation_amortization", nde_year)
            if _finite_numeric(nde_da) and float(nde_da) > 0:
                ebitda = float(nde_oi) + float(nde_da)
                ebitda_method = "REAL_EBITDA"
            else:
                ebitda = float(nde_oi)
            if ebitda > 0:
                net_debt_to_ebitda = (float(nde_debt) - float(nde_cash)) / ebitda

    # Interest coverage adequacy classification (Graham standard: 5x minimum for industrials)
    if isinstance(interest_coverage, (int, float)):
        ic_val = float(interest_coverage)
        if ic_val >= 7.0:
            ic_adequacy = "STRONG"
        elif ic_val >= 5.0:
            ic_adequacy = "ADEQUATE"
        elif ic_val >= 3.0:
            ic_adequacy = "THIN"
        elif ic_val >= 1.5:
            ic_adequacy = "WEAK"
        else:
            ic_adequacy = "CRITICAL"
    else:
        ic_adequacy = "UNKNOWN"

    return {
        "interest_coverage": interest_coverage,
        "interest_coverage_adequacy": ic_adequacy,
        "de_ratio": de_ratio,
        "cash_coverage": cash_coverage,
        "cash_coverage_basis": "CASH_PLUS_SHORT_TERM_INVESTMENTS",
        "net_debt_to_ebitda": net_debt_to_ebitda,
        "ebitda_method": ebitda_method,
        "flags": flags,
    }


def _run_reverse_dcf(
    price: float | None,
    shares: float,
    net_debt: float | None,
    revenue: float | None,
    operating_income: float | None,
    *,
    price_context: dict[str, Any] | None = None,
    revenue_cagr_5y: float | None = None,
) -> dict[str, Any]:
    if price is None:
        return {
            "status": "REVERSE_DCF_NO_PRICE",
            "reason_code": str((price_context or {}).get("price_reason_code") or "PRICE_UNKNOWN"),
            "reason_detail": str(
                (price_context or {}).get("price_reason_detail") or "Market price unavailable."
            ),
        }
    if not isinstance(net_debt, (int, float)):
        return {"status": "REVERSE_DCF_INSUFFICIENT_DATA", "reason_code": "NET_DEBT_UNKNOWN"}
    if not revenue or operating_income is None or shares <= 0:
        return {"status": "REVERSE_DCF_INSUFFICIENT_DATA", "reason_code": "INSUFFICIENT_INPUTS"}
    margin = operating_income / revenue if revenue > 0 else None
    if margin is None:
        return {"status": "REVERSE_DCF_INSUFFICIENT_DATA", "reason_code": "INSUFFICIENT_INPUTS"}
    # Calls existing module with its own defaults: tax_rate=0.25, reinvestment_rate=0.35.
    # These differ intentionally from this module's 21% tax — do NOT change them.
    outputs, warnings = implied_growth_from_price(
        market_price=price,
        shares_outstanding=shares,
        net_debt=net_debt,
        base_revenue=revenue,
        margin=margin,
    )
    # Feasibility classification vs historical growth. Saturated solves are
    # clipped bounds, not real solves — grading them REASONABLE/IMPLAUSIBLE
    # fabricates a label (audit: saturated-bound-leaks-to-flag-ignoring-
    # consumers), so they stay UNKNOWN.
    implied_growth = (outputs or {}).get("implied_growth")
    feasibility = "UNKNOWN"
    if bool((outputs or {}).get("implied_growth_saturated")):
        implied_growth_for_feasibility = None
    else:
        implied_growth_for_feasibility = implied_growth
    implied_growth = implied_growth_for_feasibility
    if isinstance(implied_growth, (int, float)) and isinstance(revenue_cagr_5y, (int, float)):
        if revenue_cagr_5y > 0:
            if implied_growth > revenue_cagr_5y * 2.0:
                feasibility = "IMPLAUSIBLE"
            elif implied_growth > revenue_cagr_5y * 1.5:
                feasibility = "AGGRESSIVE"
            elif implied_growth > revenue_cagr_5y:
                feasibility = "OPTIMISTIC"
            else:
                feasibility = "REASONABLE"
        else:
            if implied_growth > 0.05:
                feasibility = "IMPLAUSIBLE"
            elif implied_growth > 0:
                feasibility = "OPTIMISTIC"
            else:
                feasibility = "REASONABLE"

    # Canonical expectations-gap signal. Quality_flags are threaded at the
    # packet layer; pass [] here so the gap is computed from the raw
    # supportable-growth estimate and persists into the reverse_dcf outputs_json.
    supportable_growth, _supportable_basis = estimate_supportable_growth(
        revenue_cagr_5y=revenue_cagr_5y,
        owner_earnings_cagr_5y=None,
        quality_flags=[],
    )
    expectations_gap = compute_expectations_gap(
        (outputs or {}).get("implied_growth"),
        supportable_growth,
        bool((outputs or {}).get("implied_growth_saturated", False)),
        saturated_bound=(outputs or {}).get("implied_growth_saturated_bound"),
        margin_sign=(outputs or {}).get("margin_sign"),
    )

    return {
        "status": "OK",
        "reason_code": "OK",
        "outputs": outputs,
        "warnings": warnings,
        "feasibility": feasibility,
        "revenue_cagr_5y_used": revenue_cagr_5y,
        "expectations_gap": expectations_gap,
    }


# ── recursive signal ──────────────────────────────────────────────────────────


def _prior_run_delta(
    conn: Any,
    ticker: str,
    as_of_date: str,
    method: str,
    current_value: float | None,
) -> dict[str, Any]:
    """
    Find the most-recent row with as_of_date < current and compute value delta.
    Same-day re-runs overwrite via UNIQUE constraint, so no delta is possible within a day.
    Narrative flags (MOS_WIDENED, THESIS_STRENGTHENING, etc.) deferred to Layer 3.
    """
    row = latest_decision_eligible_valuation_row(
        conn,
        ticker=ticker,
        method=method,
        before_as_of_date=as_of_date,
    )
    if not row:
        return {"status": "PRIOR_RUN_NOT_AVAILABLE"}
    try:
        prior_outputs = json.loads(row["outputs_json"] or "{}")
        prior_value = prior_outputs.get("value_per_share") or prior_outputs.get("base")
    except Exception:
        return {"status": "PRIOR_RUN_NOT_AVAILABLE"}
    if prior_value is None or prior_value == 0 or current_value is None:
        return {"status": "PRIOR_RUN_NOT_AVAILABLE"}
    change_pct = (current_value - prior_value) / abs(prior_value)
    return {
        "status": "OK",
        "prior_as_of_date": row["as_of_date"],
        "prior_value": prior_value,
        "current_value": current_value,
        "value_change_pct": change_pct,
    }


# ── TTL ───────────────────────────────────────────────────────────────────────


def _is_valuation_fresh(ticker: str, as_of_date: str, conn: Any) -> bool:
    row = conn.execute(
        f"SELECT created_at FROM {valuations_table()} "
        "WHERE ticker = ? AND as_of_date = ? AND method = 'owner_earnings' "
        "ORDER BY created_at DESC LIMIT 1",
        (ticker.upper(), as_of_date),
    ).fetchone()
    if not row:
        return False
    try:
        created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - created).total_seconds() < _TTL_SECONDS
    except Exception:
        return False


# ── main entry points ─────────────────────────────────────────────────────────


def ensure_valuation(
    ticker: str,
    as_of_date: str,
    provider: PriceProvider | None = None,
    *,
    run_id: str | None = None,
    price_override: float | None = None,
    force_refresh: bool = False,
    cfg: AppConfig | None = None,
    db_path: str | Path | None = None,
    raise_on_error: bool = False,
    issuer_cik: str | None = None,
    issuer_aliases: tuple[str, ...] = (),
    require_filed_asof: bool = True,
    price_provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Guarantee the valuations table has rows for (ticker, as_of_date).
    No-ops if data is fresh (within TTL). By default all errors are logged and
    suppressed for legacy callers. Repair pipelines may set ``raise_on_error``
    so transient failures remain visible and retryable.
    """
    try:
        records = _ensure_valuation_inner(
            ticker,
            as_of_date,
            provider,
            run_id=run_id,
            price_override=price_override,
            force_refresh=force_refresh,
            cfg=cfg,
            db_path=db_path,
            issuer_cik=issuer_cik,
            issuer_aliases=issuer_aliases,
            require_filed_asof=require_filed_asof,
            price_provenance=price_provenance,
        )
        register_valuation_writer_records(run_id, records)
        return records
    except Exception as exc:  # noqa: BLE001
        if raise_on_error:
            raise
        logger.warning("ensure_valuation failed for %s: %s", ticker, exc)
        return []


def _valuation_write_lineage(
    *,
    ticker: str,
    as_of_date: str,
    method: str,
    inputs_json: str,
    outputs_json: str,
    warnings_json: str,
    created_at: str,
    quality_gate_verdict: str | None,
    confidence_class: str | None,
    gate_reason_codes: str | None,
    valuation_headwinds: str | None,
    valuation_supports: str | None,
    source_lineage: dict[str, str | None],
) -> tuple[str | None, str | None, str | None, str]:
    """Return exact source fields plus a fingerprint for one persisted row."""

    values = {
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "method": method,
        "inputs_json": inputs_json,
        "outputs_json": outputs_json,
        "warnings_json": warnings_json,
        "created_at": created_at,
        "valuation_writer_version": _VERSION,
        "quality_gate_verdict": quality_gate_verdict,
        "confidence_class": confidence_class,
        "gate_reason_codes": gate_reason_codes,
        "valuation_headwinds": valuation_headwinds,
        "valuation_supports": valuation_supports,
        **source_lineage,
    }
    return (
        source_lineage["source_run_id"],
        source_lineage["source_artifact_path"],
        source_lineage["source_artifact_sha256"],
        valuation_integrity_fingerprint(values),
    )


def _archive_valuation_row(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    method: str,
    new_outputs_json: str,
    new_source_run_id: str | None = None,
    new_source_artifact_path: str | None = None,
    new_source_artifact_sha256: str | None = None,
    new_financial_integrity_fingerprint: str | None = None,
) -> None:
    """Copy the existing live row into valuations_history before an
    ON CONFLICT UPDATE overwrites it — skipped when the row is absent,
    identical in both output and source lineage, or the write targets the
    measurement table."""
    from app.valuation.measurement import in_measurement_scope

    if in_measurement_scope():
        return
    try:
        row = conn.execute(
            "SELECT id, inputs_json, outputs_json, warnings_json, created_at, "
            "valuation_writer_version, quality_gate_verdict, confidence_class, "
            "gate_reason_codes, valuation_headwinds, valuation_supports, "
            "source_run_id, source_artifact_path, source_artifact_sha256, "
            "financial_integrity_fingerprint "
            "FROM valuations WHERE ticker = ? AND as_of_date = ? AND method = ?",
            (ticker.upper(), as_of_date, method),
        ).fetchone()
    except Exception:  # noqa: BLE001 - archival must never block the write
        return
    if row is None:
        return
    if (
        row["outputs_json"] == new_outputs_json
        and row["source_run_id"] == new_source_run_id
        and row["source_artifact_path"] == new_source_artifact_path
        and row["source_artifact_sha256"] == new_source_artifact_sha256
        and row["financial_integrity_fingerprint"] == new_financial_integrity_fingerprint
    ):
        return
    try:
        conn.execute(
            "INSERT INTO valuations_history("
            "source_id, ticker, as_of_date, method, inputs_json, outputs_json, "
            "warnings_json, created_at, valuation_writer_version, "
            "quality_gate_verdict, confidence_class, gate_reason_codes, "
            "valuation_headwinds, valuation_supports, source_run_id, "
            "source_artifact_path, source_artifact_sha256, "
            "financial_integrity_fingerprint, archived_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                ticker.upper(),
                as_of_date,
                method,
                row["inputs_json"],
                row["outputs_json"],
                row["warnings_json"],
                row["created_at"],
                row["valuation_writer_version"],
                row["quality_gate_verdict"],
                row["confidence_class"],
                row["gate_reason_codes"],
                row["valuation_headwinds"],
                row["valuation_supports"],
                row["source_run_id"],
                row["source_artifact_path"],
                row["source_artifact_sha256"],
                row["financial_integrity_fingerprint"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    except Exception:  # noqa: BLE001 - archival must never block the write
        pass


def _ensure_valuation_inner(
    ticker: str,
    as_of_date: str,
    provider: PriceProvider | None,
    *,
    run_id: str | None,
    price_override: float | None,
    force_refresh: bool,
    cfg: AppConfig | None = None,
    db_path: str | Path | None = None,
    issuer_cik: str | None = None,
    issuer_aliases: tuple[str, ...] = (),
    require_filed_asof: bool = True,
    price_provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved_cfg = cfg or get_config()
    if db_path is not None and Path(resolved_cfg.db_path) != Path(db_path):
        resolved_cfg = resolved_cfg.model_copy(update={"db_path": Path(db_path)})
    source_lineage = valuation_source_lineage(run_id)

    with get_db(cfg=resolved_cfg) as conn:
        if (not force_refresh) and _is_valuation_fresh(ticker, as_of_date, conn):
            return []

        if require_filed_asof:
            facts, evidenced_zero_facts = _load_valuation_facts(
                ticker,
                conn,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                issuer_aliases=issuer_aliases,
                cfg=resolved_cfg,
            )
        else:
            facts = _load_facts(
                ticker,
                conn,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                issuer_aliases=issuer_aliases,
                require_filed_asof=False,
            )
            evidenced_zero_facts = []
        facts_row = (
            _local_v2_facts_row(
                ticker=ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                facts=facts,
                evidenced_zero_facts=evidenced_zero_facts,
            )
            if require_filed_asof
            else resolve_financial_facts_asof(
                ticker=ticker.upper(),
                as_of_date=as_of_date,
                run_id=run_id,
                refresh=False,
                cfg=resolved_cfg,
            )
        )
        category_result = classify_company_category(
            ticker,
            as_of_date,
            facts_row=facts_row,
            cfg=resolved_cfg,
        )
        category = str(category_result.get("category") or TRADITIONAL_OPERATING)
        quality_wacc = _compute_quality_wacc(facts, category_result=category_result)
        adjusted_wacc = float(quality_wacc.get("adjusted_wacc") or _WACC)

        # ── Pre-valuation quality gate ────────────────────────────────────────
        from app.valuation.pre_valuation_gate import compute_quality_context, BLOCK as _GATE_BLOCK

        quality_ctx = compute_quality_context(
            ticker.upper(),
            as_of_date,
            facts=facts,
            facts_row=facts_row,
            category=category,
            cfg=resolved_cfg,
            issuer_cik=issuer_cik,
            issuer_aliases=issuer_aliases,
            require_filed_asof=require_filed_asof,
            db_path=resolved_cfg.db_path if require_filed_asof else None,
        )
        gate_action = str(quality_ctx.get("gate_action") or "PROCEED")
        _tg_raw = quality_ctx.get("terminal_growth_override")
        terminal_growth = float(_tg_raw) if isinstance(_tg_raw, (int, float)) else _TERMINAL_GROWTH

        # ── Bank/financial issuer detection ─────────────────────────────────
        # Banks have deposits as liabilities and negative CFO from lending —
        # the DCF/EPV framework produces nonsensical results. Block them.
        is_financial_issuer = bool(facts.get("deposits"))
        if is_financial_issuer and gate_action != "BLOCK":
            gate_action = "BLOCK"
            quality_ctx["gate_action"] = "BLOCK"
            quality_ctx["gate_reason"] = "FINANCIAL_ISSUER_DCF_INAPPLICABLE"
            quality_ctx["gate_reason_codes"] = quality_ctx.get("gate_reason_codes", []) + [
                "FINANCIAL_ISSUER"
            ]
            quality_ctx["confidence_class"] = "INSUFFICIENT"

        # Price (may be None — intrinsic methods still run)
        price, price_context = _resolve_market_price(
            ticker,
            as_of_date,
            provider,
            run_id=run_id,
            price_override=price_override,
            cfg=resolved_cfg,
        )
        if isinstance(price_provenance, dict):
            price_context.update(
                {
                    "price_as_of_date": price_provenance.get("as_of_date"),
                    "price_currency": price_provenance.get("currency"),
                    "price_source": price_provenance.get("source"),
                    "current_price_source": price_provenance.get("source"),
                    "price_source_resolution": price_provenance.get("source_resolution")
                    or price_provenance.get("source"),
                    "price_source_url": price_provenance.get("url")
                    or price_provenance.get("source_url"),
                    "price_confidence": price_provenance.get("confidence"),
                    "price_basis": price_provenance.get("price_basis"),
                    "raw_price": price_provenance.get("raw_price"),
                    "split_adjustment_factor": price_provenance.get("split_adjustment_factor"),
                    "split_effective_date": price_provenance.get("split_effective_date"),
                    "split_event": price_provenance.get("split_event"),
                    "no_intervening_split_proof": price_provenance.get(
                        "no_intervening_split_proof"
                    ),
                    "quote_snapshot_id": price_provenance.get("quote_snapshot_id"),
                }
            )
        if require_filed_asof:
            price_context.update(
                {
                    "pipeline_version": "v2",
                    "issuer_cik": str(issuer_cik or ""),
                    "issuer_aliases": sorted(
                        {
                            str(item).strip().upper()
                            for item in (ticker, *issuer_aliases)
                            if str(item).strip()
                        }
                    ),
                    "require_filed_asof": True,
                    "facts_fingerprint": _facts_revision_fingerprint(
                        facts,
                        evidenced_zero_facts,
                    ),
                }
            )
            if evidenced_zero_facts:
                price_context["evidenced_zero_facts"] = evidenced_zero_facts
        if price is None:
            logger.warning(
                "ensure_valuation %s: NO CURRENT PRICE — margin-of-safety "
                "calculations will be incomplete. Check price_provider config.",
                ticker,
            )

        # Net Debt: latest year where BOTH total_debt and cash are present.
        # ASC 842: the DCF/EPV flow bases (CFO, operating income) are already
        # rent-burdened, so the equity bridge uses LEASE-EXCLUSIVE net debt —
        # subtracting the operating-lease liability here as well would charge
        # the lease twice (once as a perpetual expense stream, once as a debt
        # stock). Lease-INCLUSIVE net debt remains the basis for leverage and
        # solvency diagnostics only (net_debt.py proxy keeps both variants).
        net_debt_flags: list[str] = []
        nd_year = _latest_common_year(facts, "total_debt", "cash")
        net_debt: float | None = None
        # The debt/cash pair must come from a current balance sheet. When the
        # issuer's latest annual balance sheet is more than one fiscal year
        # newer than the last year debt was reported (Ford: debt tagged only by
        # segment since 2021, so companyfacts' last total is 2020), the old
        # pair says nothing about today's claims: refuse, never net it.
        net_debt_short_term_investments: dict[str, Any] | None = None
        if _debt_year_is_stale(facts, nd_year):
            net_debt_flags.extend(["NET_DEBT_STALE_YEAR", "NET_DEBT_UNKNOWN"])
        elif nd_year is not None:
            raw_debt = _get_value_for_year(facts, "total_debt", nd_year)
            # Cash plus the current short-term investments of the same balance
            # sheet.
            raw_cash = _cash_like_for_year(facts, nd_year)
            lease_liability = _get_value_for_year(
                facts,
                "operating_lease_liability",
                nd_year,
            )
            # Senior claims: preferred stock and noncontrolling interest rank
            # ahead of common in the EV->equity bridge (audit coverage gap 2).
            preferred_equity = _get_value_for_year(
                facts,
                "preferred_equity",
                nd_year,
            )
            noncontrolling_interest = _get_value_for_year(
                facts,
                "noncontrolling_interest",
                nd_year,
            )
            debt_cash_complete = _finite_numeric(raw_debt) and _finite_numeric(raw_cash)
            senior_claims_complete = (
                _finite_numeric(preferred_equity)
                and float(preferred_equity) >= 0
                and _finite_numeric(noncontrolling_interest)
                and float(noncontrolling_interest) >= 0
            )
            if not debt_cash_complete:
                net_debt_flags.extend(["NET_DEBT_INPUTS_MISSING", "NET_DEBT_UNKNOWN"])
            elif not senior_claims_complete:
                net_debt_flags.extend(["SENIOR_CLAIMS_UNKNOWN", "NET_DEBT_UNKNOWN"])
            else:
                senior_claims = float(preferred_equity) + float(noncontrolling_interest)
                net_debt = float(raw_debt) - float(raw_cash) + senior_claims
                if senior_claims > 0:
                    net_debt_flags.append("SENIOR_CLAIMS_DEDUCTED")
                short_term = _get_value_for_year(facts, "short_term_investments", nd_year)
                if short_term is not None and _finite_numeric(short_term) and short_term > 0:
                    net_debt_flags.append("SHORT_TERM_INVESTMENTS_INCLUDED")
                    net_debt_short_term_investments = {
                        "value": float(short_term),
                        "fiscal_year": int(nd_year),
                        "line_item": "short_term_investments",
                        "source_tags": _short_term_investment_tags(conn, ticker, nd_year),
                    }
            if _finite_numeric(lease_liability) and float(lease_liability) > 0:
                net_debt_flags.append("LEASE_EXCLUDED_POSTLEASE_FLOWS")
        else:
            net_debt_proxy = resolve_net_debt_proxy(
                ticker.upper(),
                as_of_date,
                run_id=run_id,
                facts_row=facts_row,
                cfg=resolved_cfg,
            )
            proxy_value = (
                net_debt_proxy.get("net_debt_proxy_lease_exclusive")
                if isinstance(net_debt_proxy, dict)
                else None
            )
            if isinstance(proxy_value, (int, float)):
                net_debt = float(proxy_value)
                net_debt_flags.append("ASOF_NET_DEBT_PROXY")
                proxy_sti = net_debt_proxy.get("short_term_investments")
                if isinstance(proxy_sti, dict):
                    net_debt_flags.append("SHORT_TERM_INVESTMENTS_INCLUDED")
                    net_debt_short_term_investments = {
                        "value": proxy_sti.get("value"),
                        "period_end": proxy_sti.get("period_end"),
                        "line_item": "short_term_investments",
                        "source_tags": proxy_sti.get("derivation"),
                    }
                # The as-of proxy extractor has no preferred/NCI tag families,
                # so this branch cannot deduct senior claims like the inline
                # nd_year branch does — the asymmetry must be visible in
                # pzd/coverage (review EVB-2).
                net_debt_flags.append("SENIOR_CLAIMS_UNKNOWN")
            else:
                net_debt_flags.append("NET_DEBT_UNKNOWN")

        # Shares: select a STABLE per-share divisor from the trailing FY series.
        # A single corrupt latest-FY share count would otherwise roughly double
        # the per-share anchor; select_stable_shares falls back to the
        # trailing-3 median and flags SHARES_LATEST_FY_OUTLIER when it does.
        # A >50% jump corroborated by an independent as-of-visible quarterly
        # cover-page count is a GENUINE capital event (reverse split / ATM /
        # buyback) and keeps the latest count instead (audit:
        # stable-shares-stale-on-capital-events).
        if issuer_cik or issuer_aliases or require_filed_asof:
            _, corroborating_rows = issuer_companyfacts_rows(
                conn,
                ticker,
                columns=("value",),
                issuer_cik=issuer_cik,
                aliases=issuer_aliases,
                exclude_period_types=("FY",),
                line_items=("shares_outstanding",),
                as_of_date=as_of_date,
                value_not_null=True,
                require_filed_asof=require_filed_asof,
                limit=1,
                order_by="period_end DESC, fiscal_year DESC",
            )
            corro_row = corroborating_rows[0] if corroborating_rows else None
        else:
            corro_sql = (
                "SELECT value FROM companyfacts_facts WHERE ticker = ? "
                "AND line_item = 'shares_outstanding' AND period_type != 'FY' "
                "AND value IS NOT NULL"
            )
            corro_params: list[Any] = [ticker.upper()]
            if as_of_date:
                corro_sql += " AND period_end <= ?"
                corro_params.append(str(as_of_date))
            corro_sql += " ORDER BY period_end DESC LIMIT 1"
            corro_row = conn.execute(corro_sql, corro_params).fetchone()
        corroborating_count = float(corro_row["value"]) if corro_row else None
        shares, shares_flag = select_stable_shares(
            _n_years(facts, "shares_outstanding", n=4),
            corroborating_count=corroborating_count,
        )

        # ── BLOCK gate: skip all valuation methods ────────────────────────────
        if gate_action == _GATE_BLOCK:
            now = utc_now_iso()
            quality_gate_verdict = gate_action
            quality_confidence_class = str(quality_ctx.get("confidence_class") or "")
            quality_gate_reason_codes = json.dumps(quality_ctx.get("gate_reason_codes") or [])
            quality_headwinds = json.dumps(quality_ctx.get("valuation_headwinds") or [])
            quality_supports = json.dumps(quality_ctx.get("valuation_supports") or [])
            blocked_price_detail = {
                "gate_action": gate_action,
                "gate_reason": quality_ctx.get("gate_reason"),
                "earnings_quality": quality_ctx.get("earnings_quality"),
                "leverage_stress": quality_ctx.get("leverage_stress"),
                "epv_quality": quality_ctx.get("epv_quality"),
                "cycle_position": quality_ctx.get("cycle_position"),
                "revenue_cagr_5y": quality_ctx.get("revenue_cagr_5y"),
                "revenue_cagr_3y": quality_ctx.get("revenue_cagr_3y"),
            }
            blocked_price_detail.update(_scorecard_price_lineage(price_context))
            blocked_scorecard = {
                "signal": "VALUATION_BLOCKED",
                "pricing_zone": "VALUATION_BLOCKED",
                "pricing_zone_detail": blocked_price_detail,
                "quality_context": quality_ctx,
            }
            _blocked_outputs_json = json.dumps(blocked_scorecard, default=str)
            _blocked_inputs_json = json.dumps(
                {**price_context, "shares": shares, "net_debt": net_debt}
            )
            _warnings_json = json.dumps([])
            _blocked_lineage = _valuation_write_lineage(
                ticker=ticker,
                as_of_date=as_of_date,
                method="scorecard",
                inputs_json=_blocked_inputs_json,
                outputs_json=_blocked_outputs_json,
                warnings_json=_warnings_json,
                created_at=now,
                quality_gate_verdict=quality_gate_verdict,
                confidence_class=quality_confidence_class,
                gate_reason_codes=quality_gate_reason_codes,
                valuation_headwinds=quality_headwinds,
                valuation_supports=quality_supports,
                source_lineage=source_lineage,
            )
            _archive_valuation_row(
                conn,
                ticker=ticker,
                as_of_date=as_of_date,
                method="scorecard",
                new_outputs_json=_blocked_outputs_json,
                new_source_run_id=_blocked_lineage[0],
                new_source_artifact_path=_blocked_lineage[1],
                new_source_artifact_sha256=_blocked_lineage[2],
                new_financial_integrity_fingerprint=_blocked_lineage[3],
            )
            conn.execute(
                f"""INSERT INTO {valuations_table()}
                   (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json,
                    created_at, valuation_writer_version,
                    quality_gate_verdict, confidence_class, gate_reason_codes,
                    valuation_headwinds, valuation_supports, source_run_id,
                    source_artifact_path, source_artifact_sha256,
                    financial_integrity_fingerprint)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, as_of_date, method) DO UPDATE SET
                       outputs_json=excluded.outputs_json,
                       inputs_json=excluded.inputs_json,
                       warnings_json=excluded.warnings_json,
                       created_at=excluded.created_at,
                       valuation_writer_version=excluded.valuation_writer_version,
                       quality_gate_verdict=excluded.quality_gate_verdict,
                       confidence_class=excluded.confidence_class,
                       gate_reason_codes=excluded.gate_reason_codes,
                       valuation_headwinds=excluded.valuation_headwinds,
                       valuation_supports=excluded.valuation_supports,
                       source_run_id=excluded.source_run_id,
                       source_artifact_path=excluded.source_artifact_path,
                       source_artifact_sha256=excluded.source_artifact_sha256,
                       financial_integrity_fingerprint=
                           excluded.financial_integrity_fingerprint""",
                (
                    ticker.upper(),
                    as_of_date,
                    "scorecard",
                    _blocked_inputs_json,
                    _blocked_outputs_json,
                    _warnings_json,
                    now,
                    _VERSION,
                    quality_gate_verdict,
                    quality_confidence_class,
                    quality_gate_reason_codes,
                    quality_headwinds,
                    quality_supports,
                    *_blocked_lineage,
                ),
            )
            conn.commit()
            logger.warning(
                "VALUATION_BLOCKED: %s — %s",
                ticker.upper(),
                quality_ctx.get("gate_reason"),
            )
            blocked_record = valuation_source_record(
                {
                    "ticker": ticker.upper(),
                    "as_of_date": as_of_date,
                    "method": "scorecard",
                    "inputs_json": _blocked_inputs_json,
                    "outputs_json": _blocked_outputs_json,
                    "warnings_json": _warnings_json,
                    "created_at": now,
                    "valuation_writer_version": _VERSION,
                    "quality_gate_verdict": quality_gate_verdict,
                    "confidence_class": quality_confidence_class,
                    "gate_reason_codes": quality_gate_reason_codes,
                    "valuation_headwinds": quality_headwinds,
                    "valuation_supports": quality_supports,
                    "source_run_id": _blocked_lineage[0],
                }
            )
            if blocked_record is None:
                raise RuntimeError("valuation writer produced a noncanonical blocked source record")
            return [blocked_record]

        # Revenue series for DCF
        rev_series = _n_years(facts, "revenue", n=5)

        # Non-recurring revenue detection: catches KROS/PTCT-style licensing
        # spikes that would otherwise inflate DCF growth assumptions. Hoisted
        # ahead of owner earnings (review OE-1) so the CFO peak-normalization
        # strong-grower exemption sees the durable series — a co-spiking
        # latest year otherwise defeats the exemption via the endpoint CAGR.
        nonrecurring_revenue = _compute_nonrecurring_revenue(ticker, facts)

        # When a spike is detected, build the "durable" revenue series (latest
        # year replaced with `durable_revenue_base`); it feeds the exemption
        # here and the durable DCF below.
        rev_series_durable: list[tuple[int, float]] | None = None
        if (
            nonrecurring_revenue.has_suspected_nonrecurring
            and isinstance(nonrecurring_revenue.durable_revenue_base, (int, float))
            and nonrecurring_revenue.durable_revenue_base > 0
            and nonrecurring_revenue.spike_year is not None
        ):
            rev_series_durable = [
                (yr, float(nonrecurring_revenue.durable_revenue_base))
                if yr == nonrecurring_revenue.spike_year
                else (yr, v)
                for yr, v in rev_series
            ]

        # Owner earnings: the DCF projects growth, so its base charges FULL
        # capex (ratio 1.0). The category maintenance ratio
        # (quality_ctx["maintenance_capex_pct"], growth-aware) is reserved for
        # zero-growth earnings-power surfaces and gates — feeding a
        # maintenance-only OE into growth scenarios would credit growth at
        # zero reinvestment cost (audit: dcf-growth-on-maintenance-capex-oe).
        oe_result = _compute_owner_earnings(
            facts,
            maintenance_capex_ratio=1.0,
            durable_revenue_series=rev_series_durable,
        )
        owner_earnings = (
            oe_result.get("owner_earnings_latest") if oe_result["status"] == "OK" else None
        )
        latest_rev = rev_series[0][1] if rev_series else None
        net_debt_to_ebitda_proxy = _latest_net_debt_to_ebitda_proxy(facts)

        # Four methods (quality-adjusted terminal growth for DCF)
        dcf_result = (
            _discounted_owner_earnings(
                owner_earnings,
                shares,
                net_debt,
                rev_series,
                wacc=adjusted_wacc,
                terminal_growth=terminal_growth,
            )
            if owner_earnings is not None and shares > 0
            else {
                "status": "METHOD_INSUFFICIENT_DATA",
                "flags": ["OWNER_EARNINGS_MISSING"],
                "low": None,
                "base": None,
                "high": None,
            }
        )

        # EPV: use epv_adjustment from quality gate, taxed at the issuer's own
        # normalized rate when its filings support one (statutory fallback,
        # flagged, otherwise).
        epv_oi_series = _n_years(facts, "operating_income", n=5)
        epv_tax_basis = _normalized_tax_rate(facts)
        # Greenwald's (D&A - maintenance capex) EBIT adjustment: the zero-
        # growth surface the gate's growth-aware maintenance ratio is for.
        epv_da_basis = _epv_da_and_maintenance_capex(
            facts,
            quality_ctx.get("maintenance_capex_pct"),
            da_source_concepts=_da_source_concepts(
                conn,
                ticker,
                as_of_date=as_of_date,
                issuer_cik=issuer_cik,
                issuer_aliases=issuer_aliases,
            ),
        )
        # The EPV's "current revenue" must be the spike-corrected one when a
        # non-recurring latest-year spike was detected: a one-off licensing
        # year multiplied by the through-cycle margin is not earnings power.
        epv_rev_series = rev_series_durable if rev_series_durable is not None else rev_series
        epv_result = _epv(
            epv_oi_series,
            net_debt,
            shares,
            revenue_series=epv_rev_series,
            wacc=adjusted_wacc,
            epv_adjustment=str(quality_ctx.get("epv_adjustment") or "NONE"),
            normalized_earnings=quality_ctx.get("normalized_earnings"),
            tax_rate=epv_tax_basis["tax_rate"],
            depreciation_amortization=epv_da_basis["depreciation_amortization"],
            maintenance_capex=epv_da_basis["maintenance_capex"],
        )
        epv_result["tax_rate_basis"] = dict(epv_tax_basis)
        epv_result["da_maintenance_capex_detail"] = dict(epv_da_basis)

        # Compute the intangible amort add-back series (used after R&D adjust
        # block for the cash-adjusted EPV). Computing the metadata here so
        # both the if-block (R&D path) and the else-block (traditional path)
        # can layer it on top of the right base.
        intangible_amort_series, addback_by_year = _compute_intangible_amort_addback(facts)
        intangible_addback_present = any(v > 0 for v in addback_by_year.values())

        # When a spike was detected (rev_series_durable built above, before
        # owner earnings), build a "durable" DCF using the corrected revenue
        # series. This is the change that actually moves the anchor — the
        # diagnostic detection by itself wasn't enough since DCF kept reading
        # the raw rev_series. We keep both the raw and durable DCFs so
        # reviewers can see how much the spike was inflating the headline.
        dcf_durable_result = None
        if rev_series_durable is not None:
            if owner_earnings is not None and shares > 0:
                dcf_durable_result = _discounted_owner_earnings(
                    owner_earnings,
                    shares,
                    net_debt,
                    rev_series_durable,
                    wacc=adjusted_wacc,
                    terminal_growth=terminal_growth,
                )
                dcf_durable_result["wacc_detail"] = dict(quality_wacc)
                dcf_durable_result["terminal_growth"] = terminal_growth
                # Tag the result so downstream consumers know it's an alternative
                dcf_durable_result.setdefault("flags", []).append(
                    "REVENUE_SERIES_DURABLE_BASE_APPLIED"
                )

        dcf_result["wacc_detail"] = dict(quality_wacc)
        dcf_result["terminal_growth"] = terminal_growth
        # Which single-year CFO correction (if any) the DCF's owner-earnings
        # base carries: SPIKE_SMOOTHED, GROWER_LATEST_KEPT, DIP_LIFTED, DIP_KEPT_REAL_DECLINE,
        # DIP_KEPT_NONPOSITIVE,
        # DIP_KEPT_UNVERIFIED or NONE.
        dcf_result["owner_earnings_cfo_normalization"] = oe_result.get(
            "cfo_normalization", "NONE"
        )
        epv_result["wacc_detail"] = dict(quality_wacc)
        # Surface the stable-shares selection flag (prepended when not None) so the per-share substitution AND
        # the single-year no-stability-check case are both auditable in
        # outputs_json.flags and the anchor-magnitude audit.
        if shares_flag is not None:
            dcf_result.setdefault("flags", []).insert(0, shares_flag)
            epv_result.setdefault("flags", []).insert(0, shares_flag)
        rnd_adjustment_payload = None
        dcf_adjusted_result = None
        epv_adjusted_result = None
        tech_adjustment_result = {
            "category_classification": category_result,
            "rnd_adjustment": None,
            "tech_valuation_divergence": None,
            "wacc_detail": dict(quality_wacc),
        }
        if category != TRADITIONAL_OPERATING:
            rnd_adjustment_payload = compute_rnd_adjusted_earnings(
                ticker,
                as_of_date,
                category=category,
                facts_row=facts_row,
                cfg=resolved_cfg,
            )
            adjustment_metadata = {
                "category": category,
                "confidence": category_result.get("confidence"),
                "amortization_life": (rnd_adjustment_payload or {}).get("amortization_life"),
                "rnd_adjustment": (rnd_adjustment_payload or {}).get("rnd_adjustment"),
                "status": (rnd_adjustment_payload or {}).get("status"),
                "flags": (rnd_adjustment_payload or {}).get("flags", []),
            }
            _rnd_unit_ok = (
                isinstance(rnd_adjustment_payload, dict)
                and str((rnd_adjustment_payload.get("guardrails") or {}).get("input_unit") or "")
                == "USD"
                and str((rnd_adjustment_payload.get("guardrails") or {}).get("output_unit") or "")
                == "USD_millions"
                and float(
                    (rnd_adjustment_payload.get("guardrails") or {}).get("input_unit_scale") or 0.0
                )
                == 1_000_000.0
            )
            if (
                isinstance(rnd_adjustment_payload, dict)
                and rnd_adjustment_payload.get("status") == "OK"
                and not _rnd_unit_ok
            ):
                # All-sub-$1M raw inputs (nano shells): the payload's series
                # stayed in raw USD and must NOT be added to the facts OI
                # series in millions (review issue 8).
                rnd_adjustment_payload = dict(rnd_adjustment_payload)
                rnd_adjustment_payload["status"] = "RND_UNIT_SCALE_AMBIGUOUS"
            adjustment_metadata["status"] = (rnd_adjustment_payload or {}).get("status")
            adjustment_metadata["flags"] = (rnd_adjustment_payload or {}).get("flags", [])
            # Persist the post-guard payload. The prior ordering retained the
            # pre-guard OK object in tech_adjustment even after the writer had
            # rejected it for adjusted valuation arithmetic.
            tech_adjustment_result["rnd_adjustment"] = rnd_adjustment_payload
            if (
                isinstance(rnd_adjustment_payload, dict)
                and rnd_adjustment_payload.get("status") == "OK"
            ):
                adjustment_by_year = {
                    int(row["year"]): float(row["rnd_adjustment"])
                    for row in (rnd_adjustment_payload.get("time_series") or [])
                    if isinstance(row, dict)
                    and isinstance(row.get("year"), int)
                    and isinstance(row.get("rnd_adjustment"), (int, float))
                    # Incomplete vintage stacks overstate the adjustment —
                    # those years keep GAAP OI (audit: rnd-vintage-boundary).
                    # Old payloads without the key are treated as complete.
                    and bool(row.get("vintage_complete", True))
                }
                latest_adjustment = adjustment_by_year.get(
                    max(adjustment_by_year.keys(), default=0)
                )
                adjusted_owner_earnings = (
                    float(owner_earnings) + float(latest_adjustment)
                    if isinstance(owner_earnings, (int, float))
                    and isinstance(latest_adjustment, (int, float))
                    else rnd_adjustment_payload.get("adjusted_owner_earnings")
                )
                gaap_oi_series = _n_years(facts, "operating_income", n=5)
                adjusted_oi_series = [
                    (int(year), float(value) + float(adjustment_by_year.get(int(year), 0.0)))
                    for year, value in gaap_oi_series
                    if isinstance(value, (int, float))
                ]
                dcf_adjusted_result = (
                    _discounted_owner_earnings(
                        float(adjusted_owner_earnings),
                        shares,
                        net_debt,
                        # Durable (spike-corrected) series when set: the
                        # sector-specific adjusted anchor wins select_anchor,
                        # so it must inherit the spike correction like the
                        # generic dcf_base does (review ANCHOR-4).
                        rev_series_durable if rev_series_durable is not None else rev_series,
                        wacc=adjusted_wacc,
                        terminal_growth=terminal_growth,
                    )
                    if isinstance(adjusted_owner_earnings, (int, float)) and shares > 0
                    else {
                        "status": "METHOD_INSUFFICIENT_DATA",
                        "flags": ["ADJUSTED_OWNER_EARNINGS_MISSING"],
                        "low": None,
                        "base": None,
                        "high": None,
                    }
                )
                if (
                    rev_series_durable is not None
                    and isinstance(dcf_adjusted_result, dict)
                    and dcf_adjusted_result.get("base") is not None
                ):
                    dcf_adjusted_result.setdefault("flags", []).append(
                        "REVENUE_SERIES_DURABLE_BASE_APPLIED"
                    )
                # The gate's cyclical normalization must reach the R&D-adjusted
                # EPV too (audit: epv-adjusted-drops-quality-normalization) —
                # compositionally, so the R&D delta is preserved:
                # normalized = gate OI-median + latest R&D adjustment.
                _gate_normalized = quality_ctx.get("normalized_earnings")
                adjusted_normalized_earnings = (
                    float(_gate_normalized) + float(latest_adjustment)
                    if isinstance(_gate_normalized, (int, float))
                    and isinstance(latest_adjustment, (int, float))
                    else None
                )
                epv_adjusted_result = _epv(
                    adjusted_oi_series,
                    net_debt,
                    shares,
                    revenue_series=epv_rev_series,
                    wacc=adjusted_wacc,
                    epv_adjustment=str(quality_ctx.get("epv_adjustment") or "NONE"),
                    normalized_earnings=adjusted_normalized_earnings,
                    tax_rate=epv_tax_basis["tax_rate"],
                    depreciation_amortization=epv_da_basis["depreciation_amortization"],
                    maintenance_capex=epv_da_basis["maintenance_capex"],
                )
                # R&D is capitalized on this variant, and the SBC inside R&D
                # with it: say so rather than inherit the base's EXPENSED.
                if epv_adjusted_result.get("sbc_treatment") is not None:
                    epv_adjusted_result["sbc_treatment"] = "EXPENSED_EXCEPT_WITHIN_CAPITALIZED_RND"
                adjustment_metadata["valuation_alignment"] = "facts_table_plus_rnd_delta"
                adjustment_metadata["latest_rnd_adjustment"] = latest_adjustment
            else:
                adjustment_status = str(
                    (rnd_adjustment_payload or {}).get("status") or "RND_ADJUSTMENT_UNAVAILABLE"
                )
                dcf_adjusted_result = {
                    "status": "METHOD_INSUFFICIENT_DATA",
                    "flags": [adjustment_status],
                    "low": None,
                    "base": None,
                    "high": None,
                }
                epv_adjusted_result = {
                    "status": "METHOD_INSUFFICIENT_DATA",
                    "flags": [adjustment_status],
                    "value_per_share": None,
                }
            dcf_adjusted_result["adjustment_metadata"] = adjustment_metadata
            epv_adjusted_result["adjustment_metadata"] = adjustment_metadata
            dcf_adjusted_result["wacc_detail"] = dict(quality_wacc)
            dcf_adjusted_result["terminal_growth"] = terminal_growth
            epv_adjusted_result["wacc_detail"] = dict(quality_wacc)

        # ── EPV cash-adjusted: layer intangible amort add-back on top of the
        # best available EPV operating-income series. We ONLY publish this
        # alternative EPV when the add-back is materially distorting the GAAP
        # number (≥5% of revenue). For companies with trivial intangible
        # amort, publishing a near-identical "cash-adjusted" number against a
        # different base would mislead readers.
        if intangible_amort_series.is_materially_distorted:
            # Use R&D-adjusted OI series if available (tech path); otherwise
            # use the GAAP series. `adjusted_oi_series` is only defined inside
            # the R&D if-block above, so guard via locals().
            base_oi_series_for_cash = locals().get("adjusted_oi_series") or epv_oi_series
            adjusted_oi_for_intangible = [
                (yr, v + addback_by_year.get(yr, 0.0)) for yr, v in base_oi_series_for_cash
            ]
            # Compositional normalization: gate OI-median (+ R&D delta when the
            # base series is the R&D-adjusted one) + average intangible
            # add-back, so USE_NORMALIZED applies on the same basis as the
            # series being averaged (audit: epv-adjusted-drops-quality-
            # normalization — the previous normalized_earnings=None silently
            # no-opped the gate's discipline on this variant).
            _cash_gate_norm = quality_ctx.get("normalized_earnings")
            if isinstance(_cash_gate_norm, (int, float)):
                _cash_norm_base = (
                    locals().get("adjusted_normalized_earnings")
                    if locals().get("adjusted_oi_series") is not None
                    and isinstance(locals().get("adjusted_normalized_earnings"), (int, float))
                    else float(_cash_gate_norm)
                )
                _avg_addback = (
                    float(intangible_amort_series.average_addback)
                    if isinstance(intangible_amort_series.average_addback, (int, float))
                    else 0.0
                )
                cash_normalized_earnings = float(_cash_norm_base) + _avg_addback
            else:
                cash_normalized_earnings = None
            # The (physical D&A - maintenance capex) adjustment the base EPV
            # takes is independent of an add-back read from the directly tagged
            # AmortizationOfIntangibleAssets (physical D&A already excludes it),
            # so it applies on the same terms. When any year's add-back came from
            # the (D&A - capex) heuristic instead, the two overlap, and the
            # adjustment is not layered on (NOT_APPLIED_HEURISTIC_ADDBACK).
            _addback_all_direct = bool(intangible_amort_series.years) and all(
                y.reason == "DIRECT_FROM_AMORTIZATION_OF_INTANGIBLE_ASSETS"
                for y in intangible_amort_series.years
                if y.addback > 0
            )
            epv_cash_adjusted_result = _epv(
                adjusted_oi_for_intangible,
                net_debt,
                shares,
                revenue_series=epv_rev_series,
                wacc=adjusted_wacc,
                epv_adjustment=str(quality_ctx.get("epv_adjustment") or "NONE"),
                normalized_earnings=cash_normalized_earnings,
                tax_rate=epv_tax_basis["tax_rate"],
                depreciation_amortization=(
                    epv_da_basis["depreciation_amortization"] if _addback_all_direct else None
                ),
                maintenance_capex=(
                    epv_da_basis["maintenance_capex"] if _addback_all_direct else None
                ),
            )
            if (
                not _addback_all_direct
                and epv_cash_adjusted_result.get("da_maintenance_capex_basis") is not None
            ):
                epv_cash_adjusted_result["da_maintenance_capex_basis"] = (
                    "NOT_APPLIED_HEURISTIC_ADDBACK"
                )
            epv_cash_adjusted_result["wacc_detail"] = dict(quality_wacc)
        else:
            # No material distortion — don't publish a redundant alternative.
            epv_cash_adjusted_result = None

        # Graham: equity and shares must come from same fiscal year
        eq_sh_year = _latest_common_year(facts, "equity", "shares_outstanding")
        graham_equity = _get_value_for_year(facts, "equity", eq_sh_year) if eq_sh_year else None
        graham_shares = (
            _get_value_for_year(facts, "shares_outstanding", eq_sh_year) if eq_sh_year else shares
        )
        graham_result = _graham_formula(
            _n_years(facts, "net_income", n=5),
            equity=graham_equity,
            shares=graham_shares if graham_shares is not None else shares,
        )

        # NCAV: cash, revenue, shares_outstanding from same fiscal year
        ncav_year = _latest_common_year(facts, "cash", "revenue", "shares_outstanding")
        if ncav_year:
            ncav_cash = _get_value_for_year(facts, "cash", ncav_year)
            ncav_revenue = _get_value_for_year(facts, "revenue", ncav_year)
            ncav_tl = _get_value_for_year(facts, "total_liabilities", ncav_year)
            ncav_td = _get_value_for_year(facts, "total_debt", ncav_year)
            ncav_shares_at_year = _get_value_for_year(
                facts,
                "shares_outstanding",
                ncav_year,
            )
            ncav_shares = ncav_shares_at_year if ncav_shares_at_year is not None else shares
            ncav_result = _ncav(
                ncav_cash,
                ncav_revenue,
                ncav_tl,
                ncav_td,
                ncav_shares,
                price,
                accounts_receivable=_get_value_for_year(facts, "accounts_receivable", ncav_year),
                inventory=_get_value_for_year(facts, "inventory", ncav_year),
                current_assets=_get_value_for_year(facts, "current_assets", ncav_year),
                current_liabilities=_get_value_for_year(facts, "current_liabilities", ncav_year),
                preferred_equity=_get_value_for_year(facts, "preferred_equity", ncav_year),
            )
        else:
            ncav_result = {
                "status": "METHOD_INSUFFICIENT_DATA",
                "flags": ["PERIOD_ALIGNMENT_FAILED"],
                "value_per_share": None,
                "signal": None,
            }

        # ── Complementary lenses (ENHANCE): deterministic, point-in-time,
        # provenance-only — never gates, never the canonical anchor.
        ev_ebit_result = ev_ebit_anchor(
            facts,
            shares=shares,
            bridge_deduction=net_debt if isinstance(net_debt, (int, float)) else None,
            category=category,
            current_price=price,
        )
        fcf_yield_result = fcf_yield_anchor(facts, shares=shares, category=category)
        tangible_floor_result = tangible_floor(
            facts,
            shares=shares,
            ncav_value_per_share=(
                float(ncav_result.get("value_per_share"))
                if isinstance(ncav_result.get("value_per_share"), (int, float))
                else None
            ),
        )

        # REITs (SEC SIC 6798): operating income is struck after real-estate
        # depreciation, which is mostly not an economic cost for them, so earnings
        # power and EV/EBIT would capitalise a distorted figure. Both are
        # NOT_APPLICABLE; no FFO model stands in. Other methods are unchanged.
        is_reit, _reit_lookup_reason = lookup_is_reit(
            cik=issuer_cik, ticker=ticker.upper(), conn=conn, cfg=resolved_cfg
        )
        if is_reit:
            epv_result = _reit_not_applicable()
            epv_adjusted_result = None
            epv_cash_adjusted_result = None
            ev_ebit_result = _reit_not_applicable()
        elif _reit_lookup_reason != "OK":
            # No SIC on file (or the lookup failed): whether this is a REIT is
            # unknown, so the earnings-power methods a REIT would invalidate say
            # so instead of passing silently as an ordinary company's.
            for _reit_sensitive in (
                epv_result,
                epv_adjusted_result,
                epv_cash_adjusted_result,
                ev_ebit_result,
            ):
                if isinstance(_reit_sensitive, dict):
                    _reit_sensitive.setdefault("flags", []).append(REASON_REIT_STATUS_UNKNOWN)
                    _reit_sensitive["reit_status_reason"] = str(_reit_lookup_reason)

        # Scorecard, ROIC, capital structure
        methods_for_scorecard = {
            "dcf": dcf_result,
            "epv": epv_result,
            "graham": graham_result,
            "ncav": ncav_result,
        }
        if dcf_adjusted_result is not None:
            methods_for_scorecard["dcf_adjusted"] = dcf_adjusted_result
        if epv_adjusted_result is not None:
            methods_for_scorecard["epv_adjusted"] = epv_adjusted_result
        rev_cagr_5y = _revenue_cagr(rev_series, n=5)
        rev_cagr_3y = _revenue_cagr(rev_series, n=3)
        durable_dcf_base_override = (
            float(dcf_durable_result["base"])
            if dcf_durable_result is not None
            and dcf_durable_result.get("status") == "OK"
            and isinstance(dcf_durable_result.get("base"), (int, float))
            else None
        )
        scorecard = _margin_of_safety_scorecard(
            methods_for_scorecard,
            price,
            shares=shares,
            net_debt=net_debt,
            wacc_detail=quality_wacc,
            revenue_latest=latest_rev,
            net_debt_to_ebitda_proxy=net_debt_to_ebitda_proxy,
            revenue_cagr_5y=rev_cagr_5y,
            revenue_cagr_3y=rev_cagr_3y,
            gate_action=gate_action,
            mos_threshold_widening=float(quality_ctx.get("mos_threshold_widening") or 0.0),
            quality_ctx=quality_ctx,
            dcf_base_override=durable_dcf_base_override,
            tax_rate=epv_tax_basis["tax_rate"],
        )
        scorecard["wacc_detail"] = dict(quality_wacc)
        # Attach quality context to pricing_zone_detail for downstream consumers
        pzd = scorecard.get("pricing_zone_detail")
        if isinstance(pzd, dict):
            pzd.update(_scorecard_price_lineage(price_context))
            pzd["gate_action"] = gate_action
            pzd["earnings_quality"] = quality_ctx.get("earnings_quality")
            pzd["cycle_position"] = quality_ctx.get("cycle_position")
            pzd["terminal_growth_used"] = terminal_growth
            # Share-count stability outcome — consumed by the backtest
            # (SHARES_LATEST_FY_OUTLIER rows are excluded from deploy_ready
            # classification) and by output surfaces.
            if shares_flag is not None:
                pzd["shares_flag"] = shares_flag
            # Lens values for anchor provenance / per-method output table.
            for _lens_key, _lens_result in (
                ("ev_ebit_value_per_share", ev_ebit_result),
                ("fcf_yield_value_per_share", fcf_yield_result),
                ("tangible_floor_per_share", tangible_floor_result),
            ):
                _lens_v = _lens_result.get("value_per_share")
                pzd[_lens_key] = float(_lens_v) if isinstance(_lens_v, (int, float)) else None
            # When the durable DCF replaced the raw base in the zone, keep the
            # raw value auditable alongside it.
            if durable_dcf_base_override is not None:
                pzd["dcf_raw_base"] = (
                    float(dcf_result["base"])
                    if isinstance(dcf_result.get("base"), (int, float))
                    else None
                )
        if shares_flag == "SHARES_LATEST_FY_OUTLIER":
            hw = quality_ctx.get("valuation_headwinds") or []
            if "SHARE_COUNT_UNSTABLE_HEADWIND" not in hw:
                hw.append("SHARE_COUNT_UNSTABLE_HEADWIND")
                quality_ctx["valuation_headwinds"] = hw
        scorecard["quality_context"] = quality_ctx
        scorecard["quality_context"]["net_debt_flags"] = net_debt_flags
        if net_debt_short_term_investments is not None:
            scorecard["quality_context"]["net_debt_short_term_investments"] = (
                net_debt_short_term_investments
            )
        if evidenced_zero_facts:
            scorecard["quality_context"]["evidenced_zero_facts"] = evidenced_zero_facts

        # ── Nonrecurring detection ─────────────────────────────────────────────
        from app.valuation.nonrecurring_filter import detect_nonrecurring_items

        nonrecurring = detect_nonrecurring_items(facts)
        scorecard["quality_context"]["nonrecurring_detection"] = nonrecurring
        if nonrecurring.get("has_nonrecurring"):
            hw = scorecard["quality_context"].get("valuation_headwinds") or []
            if "NONRECURRING_ITEMS_HEADWIND" not in hw:
                hw.append("NONRECURRING_ITEMS_HEADWIND")
                scorecard["quality_context"]["valuation_headwinds"] = hw

        # ── Intangible amort add-back (EPV cash adjustment) ───────────────────
        # Surface both the per-year detail and the cash-adjusted EPV value so
        # readers can compare GAAP-EPV (which suppresses serial-acquirer
        # earning power via purchase-price intangible amort) against the
        # cash-adjusted version that adds intangible amort back.
        scorecard["quality_context"]["intangible_amort"] = {
            "is_materially_distorted": intangible_amort_series.is_materially_distorted,
            "average_addback_m": intangible_amort_series.average_addback,
            "average_addback_pct_of_revenue": intangible_amort_series.average_addback_pct_of_revenue,
            "by_year": [
                {
                    "fiscal_year": y.fiscal_year,
                    "addback_m": y.addback,
                    "reason": y.reason,
                }
                for y in intangible_amort_series.years
            ],
        }
        if intangible_amort_series.is_materially_distorted:
            hw = scorecard["quality_context"].get("valuation_headwinds") or []
            if "EPV_INTANGIBLE_AMORT_DISTORTION" not in hw:
                hw.append("EPV_INTANGIBLE_AMORT_DISTORTION")
                scorecard["quality_context"]["valuation_headwinds"] = hw
        # Surface the cash-adjusted EPV value alongside the existing one ONLY
        # when we computed an alternative (i.e., there was material distortion).
        if isinstance(pzd, dict) and epv_cash_adjusted_result is not None:
            cash_epv_v = epv_cash_adjusted_result.get("value_per_share")
            pzd["epv_cash_adjusted"] = cash_epv_v
            if isinstance(price, (int, float)) and isinstance(cash_epv_v, (int, float)):
                cash_epv = float(cash_epv_v)
                if cash_epv > 0:
                    pzd["margin_of_safety_vs_epv_cash_adjusted"] = (cash_epv - float(price)) / abs(
                        cash_epv
                    )

        # ── Non-recurring revenue spike (KROS/PTCT-style licensing payments) ──
        scorecard["quality_context"]["nonrecurring_revenue"] = {
            "has_suspected_nonrecurring": nonrecurring_revenue.has_suspected_nonrecurring,
            "spike_year": nonrecurring_revenue.spike_year,
            "spike_revenue_m": nonrecurring_revenue.spike_revenue,
            "prior_year_revenue_m": nonrecurring_revenue.prior_year_revenue,
            "spike_ratio": nonrecurring_revenue.spike_ratio,
            "dollar_delta_m": nonrecurring_revenue.dollar_delta,
            "durable_revenue_base_m": nonrecurring_revenue.durable_revenue_base,
            "reason": nonrecurring_revenue.reason,
        }
        if nonrecurring_revenue.has_suspected_nonrecurring:
            hw = scorecard["quality_context"].get("valuation_headwinds") or []
            if "NONRECURRING_REVENUE_SPIKE" not in hw:
                hw.append("NONRECURRING_REVENUE_SPIKE")
                scorecard["quality_context"]["valuation_headwinds"] = hw
            if isinstance(pzd, dict):
                pzd["nonrecurring_revenue_spike"] = True
                pzd["durable_revenue_base_m"] = nonrecurring_revenue.durable_revenue_base

        # Publish the durable-base DCF when computed. This is the field that
        # downstream consumers (Stage 2 triage, Stage 4 deep, scan reports)
        # should prefer over `dcf_base` for spike cases.
        if dcf_durable_result is not None:
            scorecard["quality_context"]["dcf_durable"] = {
                "low": dcf_durable_result.get("low"),
                "base": dcf_durable_result.get("base"),
                "high": dcf_durable_result.get("high"),
                "status": dcf_durable_result.get("status"),
                "flags": dcf_durable_result.get("flags", []),
            }
            if isinstance(pzd, dict):
                pzd["dcf_durable_base"] = dcf_durable_result.get("base")
                # Margin of safety against the durable DCF (more honest anchor)
                if isinstance(price, (int, float)) and isinstance(
                    dcf_durable_result.get("base"), (int, float)
                ):
                    durable_dcf = float(dcf_durable_result["base"])
                    if durable_dcf > 0:
                        pzd["margin_of_safety_vs_dcf_durable"] = (durable_dcf - float(price)) / abs(
                            durable_dcf
                        )
            # Add a headwind flag when the durable DCF is materially lower
            # than the headline DCF (>=30% drop) — that's the strong signal
            # that the headline DCF was being inflated by the spike.
            raw_dcf_base = (
                dcf_result.get("base") if isinstance(dcf_result.get("base"), (int, float)) else None
            )
            durable_dcf_base = (
                dcf_durable_result.get("base")
                if isinstance(dcf_durable_result.get("base"), (int, float))
                else None
            )
            if (
                isinstance(raw_dcf_base, (int, float))
                and isinstance(durable_dcf_base, (int, float))
                and raw_dcf_base > 0
                and (raw_dcf_base - durable_dcf_base) / raw_dcf_base >= 0.30
            ):
                hw = scorecard["quality_context"].get("valuation_headwinds") or []
                if "DCF_INFLATED_BY_NONRECURRING_REVENUE" not in hw:
                    hw.append("DCF_INFLATED_BY_NONRECURRING_REVENUE")
                    scorecard["quality_context"]["valuation_headwinds"] = hw

        # ── SBC trajectory ─────────────────────────────────────────────────────
        from app.valuation.sbc_trajectory import compute_sbc_trajectory

        # The gate's filed split ratios: a corroborated split year is split-adjusted,
        # not counted as dilution (and not left as an unexplained break).
        sbc_traj = compute_sbc_trajectory(
            facts, split_rows=quality_ctx.get("share_split_ratios")
        )
        scorecard["quality_context"]["sbc_trajectory"] = sbc_traj
        _sbc_headwind_map = {
            "SBC_ACCELERATING": "SBC_ACCELERATING_HEADWIND",
            "NET_DILUTION_DESPITE_BUYBACKS": "NET_DILUTION_HEADWIND",
            "SBC_BURDEN_EXTREME": "SBC_BURDEN_EXTREME_HEADWIND",
        }
        for flag, headwind in _sbc_headwind_map.items():
            if flag in sbc_traj.get("sbc_flags", []):
                hw = scorecard["quality_context"].get("valuation_headwinds") or []
                if headwind not in hw:
                    hw.append(headwind)
                    scorecard["quality_context"]["valuation_headwinds"] = hw

        # ── Depreciation audit ─────────────────────────────────────────────────
        from app.valuation.depreciation_audit import compute_depreciation_audit

        dep_audit = compute_depreciation_audit(facts)
        scorecard["quality_context"]["depreciation_audit"] = dep_audit
        _dep_headwind_map = {
            "DEPRECIATION_RATE_DECLINING": "DEPRECIATION_RATE_DECLINING_HEADWIND",
            "CAPEX_BELOW_DEPRECIATION": "CAPEX_BELOW_DEPRECIATION_HEADWIND",
        }
        for flag, headwind in _dep_headwind_map.items():
            if flag in dep_audit.get("depreciation_flags", []):
                hw = scorecard["quality_context"].get("valuation_headwinds") or []
                if headwind not in hw:
                    hw.append(headwind)
                    scorecard["quality_context"]["valuation_headwinds"] = hw

        # ── Method tension analysis ───────────────────────────────────────────
        from app.valuation.method_tension import analyze_method_tensions

        # Keep method tension on the same DCF the zone/anchor uses: durable
        # when the spike override fired, raw otherwise (review issue 4).
        _mt_dcf = (
            durable_dcf_base_override
            if durable_dcf_base_override is not None
            else (
                dcf_result.get("base") if isinstance(dcf_result.get("base"), (int, float)) else None
            )
        )
        _mt_epv_src = epv_adjusted_result if epv_adjusted_result else epv_result
        _mt_epv = (
            float(_mt_epv_src.get("value_per_share"))
            if isinstance((_mt_epv_src or {}).get("value_per_share"), (int, float))
            else None
        )
        _mt_graham = (
            float(graham_result.get("value_per_share"))
            if isinstance(graham_result.get("value_per_share"), (int, float))
            else None
        )
        _mt_ncav = (
            float(ncav_result.get("value_per_share"))
            if isinstance(ncav_result.get("value_per_share"), (int, float))
            else None
        )
        method_tension = analyze_method_tensions(
            dcf_value=_mt_dcf,
            epv_value=_mt_epv,
            graham_value=_mt_graham,
            ncav_value=_mt_ncav,
            current_price=price,
            revenue_cagr_5y=rev_cagr_5y,
            wacc=adjusted_wacc,
            terminal_growth=terminal_growth,
        )
        scorecard["method_tension"] = method_tension

        # ── Peer-relative context ──────────────────────────────────────────────
        try:
            from app.valuation.peer_context import compute_peer_relative_metrics

            peer_ctx = compute_peer_relative_metrics(
                ticker,
                as_of_date,
                cfg=resolved_cfg,
            )
            scorecard["quality_context"]["peer_context"] = peer_ctx
            if peer_ctx.get("status") == "OK":
                pos = peer_ctx.get("relative_position")
                hw = scorecard["quality_context"].get("valuation_headwinds") or []
                sp = scorecard["quality_context"].get("valuation_supports") or []
                if pos == "LEADER" and "PEER_LEADER_SUPPORT" not in sp:
                    sp.append("PEER_LEADER_SUPPORT")
                    scorecard["quality_context"]["valuation_supports"] = sp
                elif pos == "LAGGARD" and "PEER_LAGGARD_HEADWIND" not in hw:
                    hw.append("PEER_LAGGARD_HEADWIND")
                    scorecard["quality_context"]["valuation_headwinds"] = hw
        except Exception as exc:
            logger.warning("peer_context failed for %s: %s", ticker, exc)
            scorecard["quality_context"]["peer_context"] = {"status": "ERROR", "error": str(exc)}

        # ── Downside scenario ──────────────────────────────────────────────────
        downside = _compute_downside_scenario(
            owner_earnings=owner_earnings,
            shares=shares,
            net_debt=net_debt,
            revenue_series=epv_rev_series,
            operating_income_series=_n_years(facts, "operating_income", n=5),
            wacc=adjusted_wacc,
            base_case_dcf=(
                durable_dcf_base_override
                if durable_dcf_base_override is not None
                else dcf_result.get("base")
            ),
            current_price=price,
            tax_rate=epv_tax_basis["tax_rate"],
            is_reit=is_reit,
        )
        scorecard["downside_scenario"] = downside

        tech_adjustment_result["tech_valuation_divergence"] = scorecard.get(
            "tech_valuation_divergence"
        )
        tech_adjustment_result["tech_valuation_divergence_flag"] = scorecard.get(
            "tech_valuation_divergence_flag"
        )
        tech_adjustment_result["tech_valuation_divergence_diagnostics"] = scorecard.get(
            "tech_valuation_divergence_diagnostics"
        )
        roic = _roic_signal(facts)
        cap_struct = _capital_structure_health(facts)

        # Reverse DCF
        oi_series = _n_years(facts, "operating_income", n=1)
        latest_oi = oi_series[0][1] if oi_series else None
        rev_dcf = _run_reverse_dcf(
            price,
            shares,
            net_debt,
            latest_rev,
            latest_oi,
            price_context=price_context,
            revenue_cagr_5y=quality_ctx.get("revenue_cagr_5y"),
        )

        # Write to DB with recursive delta per method
        now = utc_now_iso()
        quality_gate_verdict = gate_action
        quality_confidence_class = str(quality_ctx.get("confidence_class") or "")
        quality_gate_reason_codes = json.dumps(quality_ctx.get("gate_reason_codes") or [])
        quality_headwinds = json.dumps(quality_ctx.get("valuation_headwinds") or [])
        quality_supports = json.dumps(quality_ctx.get("valuation_supports") or [])
        method_rows = [
            ("owner_earnings", oe_result, owner_earnings),
            ("dcf", dcf_result, dcf_result.get("base")),
            ("epv", epv_result, epv_result.get("value_per_share")),
            ("graham", graham_result, graham_result.get("value_per_share")),
            ("ncav", ncav_result, ncav_result.get("value_per_share")),
            ("ev_ebit", ev_ebit_result, ev_ebit_result.get("value_per_share")),
            ("fcf_yield", fcf_yield_result, fcf_yield_result.get("value_per_share")),
            ("tangible_floor", tangible_floor_result, tangible_floor_result.get("value_per_share")),
            ("scorecard", scorecard, None),
            ("roic", roic, None),
            ("capital_structure", cap_struct, None),
            ("reverse_dcf", rev_dcf, None),
        ]
        if dcf_adjusted_result is not None:
            method_rows.append(
                ("dcf_adjusted", dcf_adjusted_result, dcf_adjusted_result.get("base"))
            )
        if epv_adjusted_result is not None:
            method_rows.append(
                ("epv_adjusted", epv_adjusted_result, epv_adjusted_result.get("value_per_share"))
            )
        if category != TRADITIONAL_OPERATING:
            method_rows.append(("tech_adjustment", tech_adjustment_result, None))

        written_source_records: list[dict[str, Any]] = []
        for method_name, result_dict, current_val in method_rows:
            delta = _prior_run_delta(conn, ticker, as_of_date, method_name, current_val)
            outputs = dict(result_dict)
            outputs["recursive_delta"] = delta

            _method_outputs_json = json.dumps(outputs, default=str)
            _method_inputs_json = json.dumps(
                {
                    **price_context,
                    "shares": shares,
                    "net_debt": (net_debt if isinstance(net_debt, (int, float)) else "UNKNOWN"),
                }
            )
            _warnings_json = json.dumps([])
            _method_lineage = _valuation_write_lineage(
                ticker=ticker,
                as_of_date=as_of_date,
                method=method_name,
                inputs_json=_method_inputs_json,
                outputs_json=_method_outputs_json,
                warnings_json=_warnings_json,
                created_at=now,
                quality_gate_verdict=quality_gate_verdict,
                confidence_class=quality_confidence_class,
                gate_reason_codes=quality_gate_reason_codes,
                valuation_headwinds=quality_headwinds,
                valuation_supports=quality_supports,
                source_lineage=source_lineage,
            )
            _archive_valuation_row(
                conn,
                ticker=ticker,
                as_of_date=as_of_date,
                method=method_name,
                new_outputs_json=_method_outputs_json,
                new_source_run_id=_method_lineage[0],
                new_source_artifact_path=_method_lineage[1],
                new_source_artifact_sha256=_method_lineage[2],
                new_financial_integrity_fingerprint=_method_lineage[3],
            )
            conn.execute(
                f"""INSERT INTO {valuations_table()}
                   (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json,
                    created_at, valuation_writer_version,
                    quality_gate_verdict, confidence_class, gate_reason_codes,
                    valuation_headwinds, valuation_supports, source_run_id,
                    source_artifact_path, source_artifact_sha256,
                    financial_integrity_fingerprint)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, as_of_date, method) DO UPDATE SET
                       outputs_json=excluded.outputs_json,
                       inputs_json=excluded.inputs_json,
                       warnings_json=excluded.warnings_json,
                       created_at=excluded.created_at,
                       valuation_writer_version=excluded.valuation_writer_version,
                       quality_gate_verdict=excluded.quality_gate_verdict,
                       confidence_class=excluded.confidence_class,
                       gate_reason_codes=excluded.gate_reason_codes,
                       valuation_headwinds=excluded.valuation_headwinds,
                       valuation_supports=excluded.valuation_supports,
                       source_run_id=excluded.source_run_id,
                       source_artifact_path=excluded.source_artifact_path,
                       source_artifact_sha256=excluded.source_artifact_sha256,
                       financial_integrity_fingerprint=
                           excluded.financial_integrity_fingerprint""",
                (
                    ticker.upper(),
                    as_of_date,
                    method_name,
                    _method_inputs_json,
                    _method_outputs_json,
                    _warnings_json,
                    now,
                    _VERSION,
                    quality_gate_verdict,
                    quality_confidence_class,
                    quality_gate_reason_codes,
                    quality_headwinds,
                    quality_supports,
                    *_method_lineage,
                ),
            )
            source_record = valuation_source_record(
                {
                    "ticker": ticker.upper(),
                    "as_of_date": as_of_date,
                    "method": method_name,
                    "inputs_json": _method_inputs_json,
                    "outputs_json": _method_outputs_json,
                    "warnings_json": _warnings_json,
                    "created_at": now,
                    "valuation_writer_version": _VERSION,
                    "quality_gate_verdict": quality_gate_verdict,
                    "confidence_class": quality_confidence_class,
                    "gate_reason_codes": quality_gate_reason_codes,
                    "valuation_headwinds": quality_headwinds,
                    "valuation_supports": quality_supports,
                    "source_run_id": _method_lineage[0],
                }
            )
            if source_record is None:
                raise RuntimeError(
                    f"valuation writer produced a noncanonical source record for {method_name}"
                )
            written_source_records.append(source_record)
        conn.commit()
        return written_source_records


# ── append valuation section to dossier.md ────────────────────────────────────


def append_valuation_section(
    ticker: str,
    as_of_date: str,
    dossier_md_path: str,
) -> None:
    """Append Valuation & Margin of Safety section to dossier.md. No-ops if data absent."""
    try:
        _append_valuation_section_inner(ticker, as_of_date, dossier_md_path)
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning(
            "append_valuation_section failed for %s: %s", ticker, exc
        )


def _append_valuation_section_inner(ticker: str, as_of_date: str, dossier_md_path: str) -> None:
    # Decision-useful block (per-method table, anchor provenance, convention
    # labels, gate/unblock hints) — shared with the analyze CLI.
    try:
        from app.valuation.valuation_render import render_valuation_decision_block

        decision_block = render_valuation_decision_block(ticker, as_of_date)
    except Exception:  # noqa: BLE001
        decision_block = ""
    with get_db() as conn:
        rows = latest_decision_eligible_valuation_rows(
            conn,
            ticker=ticker,
            as_of_date=as_of_date,
            exact_as_of_date=True,
        )
    if not rows:
        return

    data: dict[str, Any] = {}
    for row in rows:
        try:
            data[row["method"]] = json.loads(row["outputs_json"] or "{}")
        except Exception:
            data[row["method"]] = {}

    def _fmt(val: Any) -> str:
        if val is None:
            return "N/A"
        try:
            number = float(val)
        except Exception:
            return str(val)
        # Sign before the dollar: "$-6.74" read as a price.
        return f"-${abs(number):,.2f}" if number < 0 else f"${number:,.2f}"

    sc = data.get("scorecard", {})
    # The table below prints the same DCF the decision block above prints: the
    # durable (spike-corrected) base when the writer computed one. The two used
    # to disagree on the same page.
    _pzd = sc.get("pricing_zone_detail") if isinstance(sc.get("pricing_zone_detail"), dict) else {}
    epv = data.get("epv", {})
    epv_adjusted = data.get("epv_adjusted", {})
    dcf = data.get("dcf", {})
    dcf_adjusted = data.get("dcf_adjusted", {})
    gr = data.get("graham", {})
    ncav = data.get("ncav", {})
    oe = data.get("owner_earnings", {})
    roic = data.get("roic", {})
    cap = data.get("capital_structure", {})
    rev_dcf = data.get("reverse_dcf", {})
    tech_adjustment = data.get("tech_adjustment", {})
    rev_outputs = rev_dcf.get("outputs") or {}

    epv_delta = epv.get("recursive_delta") or {}

    lines = [
        "",
        "---",
        "",
        "## Valuation & Margin of Safety",
        "",
        f"**As of:** {as_of_date}  |  **Data:** SEC XBRL (companyfacts)",
        "",
    ]
    if decision_block:
        lines += [decision_block, ""]
    lines += [
        "### Owner Earnings",
        f"- CFO: {oe.get('cfo_used', 'N/A')}M",
        f"- Normalized Capex (5Y): {oe.get('normalized_capex', 'N/A')}M",
        f"- SBC: {oe.get('sbc_used', 'N/A')}M",
        f"- **Owner Earnings: {oe.get('owner_earnings_latest', 'N/A')}M**",
        f"- Confidence: {oe.get('confidence', 'N/A')}  |  Flags: {', '.join(oe.get('flags', [])) or 'none'}",
        "",
        "### Intrinsic Value Estimates",
        "| Method | Per-Share | Status | Notes |",
        "|---|---:|---|---|",
        f"| Discounted Owner Earnings (base) | {_fmt(published_dcf_base(_pzd, dcf.get('base')))} |"
        f" {dcf.get('status', 'N/A')} | {', '.join(_dcf_table_notes(_pzd, dcf))} |",
        f"| Discounted Owner Earnings (R&D adjusted) | {_fmt(dcf_adjusted.get('base'))} | {dcf_adjusted.get('status', 'N/A')} | {', '.join(dcf_adjusted.get('flags', []))} |",
        f"| Earnings Power Value | {_fmt(epv.get('value_per_share'))} | {epv.get('status', 'N/A')} | {', '.join(epv.get('flags', []))} |",
        f"| Earnings Power Value (R&D adjusted) | {_fmt(epv_adjusted.get('value_per_share'))} | {epv_adjusted.get('status', 'N/A')} | {', '.join(epv_adjusted.get('flags', []))} |",
        f"| Graham Formula | {_fmt(gr.get('value_per_share'))} | {gr.get('status', 'N/A')} | conservative floor only |",
        f"| Net Current Asset Value | {_fmt(ncav.get('value_per_share'))} | {ncav.get('signal', ncav.get('status', 'N/A'))} | {', '.join(ncav.get('flags', []))} |",
        "",
        "### Margin of Safety Scorecard",
        f"- **Signal:** {sc.get('signal', 'N/A')}",
        f"- **Type:** {sc.get('type', 'N/A')}",
    ]
    if isinstance(sc.get("tech_valuation_divergence"), float):
        lines.append(f"- **Tech Valuation Divergence:** {sc.get('tech_valuation_divergence'):+.1%}")
    category_payload = tech_adjustment.get("category_classification") or {}
    if isinstance(category_payload, dict) and category_payload.get("category"):
        lines.append(
            f"- **Tech Category:** {category_payload.get('category')} ({category_payload.get('confidence', 'N/A')})"
        )

    ig = rev_outputs.get("implied_growth")
    if rev_dcf.get("status") == "OK" and isinstance(ig, float):
        lines.append(f"- **Implied Growth (Reverse DCF):** {ig:.1%} annual (5Y)")
    else:
        lines.append(f"- **Implied Growth (Reverse DCF):** {rev_dcf.get('status', 'N/A')}")

    lines += [
        "",
        "### Capital Quality",
        f"- ROIC/WACC: {roic.get('roic_wacc_ratio', 'N/A')} ({roic.get('signal', 'N/A')})",
        f"- Debt/Equity: {cap.get('de_ratio', 'N/A')}",
        f"- Cash Coverage: {cap.get('cash_coverage', 'N/A')}",
        f"- Interest Coverage: {cap.get('interest_coverage', 'N/A')}",
        "",
        "### vs. Prior Run",
    ]
    if epv_delta.get("status") == "OK":
        chg = epv_delta.get("value_change_pct", 0)
        lines.append(
            f"- EPV: {chg:+.1%} (prior: {epv_delta.get('prior_as_of_date', 'N/A')})"
            if isinstance(chg, float)
            else "- EPV: N/A"
        )
    else:
        lines.append("- No prior run available.")
    lines.append("")

    with open(dossier_md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
