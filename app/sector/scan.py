"""Sector scan: comparative analysis across all tickers in a sector.

Three-stage pipeline:
  1. Deterministic pre-rank via consensus margin-of-safety scoring
  2. AI comparative triage (single Sonnet call, all candidates side-by-side)
  3. Deep financial review per finalist (Sonnet + filing excerpts)

Produces a ranked markdown report with deep theses on the top N picks.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app.alpha.consensus_ranker import RankedEntry, rank_by_consensus
from app.autonomous.financial_integrity import (
    FinancialIntegrityGateResult,
    FinancialIntegrityScope,
    FinancialIntegrityViolation,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
    require_unchanged_financial_integrity_scope,
)
from app.autonomous.v1_financial_context import (
    build_canonical_v1_financial_context,
)
from app.config import AppConfig
from app.llm.providers.retry_guard import LLMCostBudgetExceeded
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    attached_provider_usage_records,
    failed_provider_usage_meta,
    provider_failed_attempt_capture,
    provider_usage_budget,
    provider_usage_capture,
    provider_usage_lane,
    provider_usage_records,
    provider_usage_records_from_exception,
    provider_usage_request,
    record_provider_usage,
)
from app.market.commodity_context import CommoditySnapshot
from app.sector.scan_preflight import (
    format_preflight_report,
    run_preflight,
)
from app.util.financial_data_access import (
    ANNUAL_COMPANYFACTS_PERIOD_TYPES,
    companyfacts_rows,
    resolve_filing_issuer_scope,
)
from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ScanConfig:
    """Tuning knobs for the scan pipeline."""

    # Stage 1: deterministic pre-rank
    pre_rank_limit: int = 0  # 0 = no limit, send all tickers to triage

    # Stage 2: AI comparative triage
    triage_model: str = "claude-sonnet-4-6"
    max_deep_reviews: int = 30
    triage_max_output_tokens: int = 8192

    # Stage 3: deep financial review
    deep_model: str = "claude-sonnet-4-6"
    deep_thinking_budget: int = 4000
    deep_max_output_tokens: int = 8192

    # Pricing (cost tracking)
    input_usd_per_mtok: float = 3.00
    output_usd_per_mtok: float = 15.00


def _reject_unbound_commodity_context(
    *,
    context: str,
    run_as_of_date: str,
    commodity_block: str = "",
    commodity_snapshots: list[CommoditySnapshot] | None = None,
) -> None:
    """Fail closed when mutable commodity data reaches the paid V1 path."""

    if not commodity_block and not commodity_snapshots:
        return
    violation = FinancialIntegrityViolation(
        code="UNBOUND_COMMODITY_CONTEXT",
        field="commodity_context",
        source_values={
            "commodity_block_present": bool(commodity_block),
            "commodity_snapshot_count": len(commodity_snapshots or ()),
        },
        expected_relationship=(
            "paid V1 sector scans exclude commodity context until exact "
            "point-in-time provenance is bound to the authorized scope"
        ),
        observed_relationship="unbound commodity context was supplied",
        reason=(
            "Mutable commodity context is not authorized for paid V1 sector scans "
            "or their published artifacts."
        ),
        terminal_status="INVALID_FINANCIAL_INPUT",
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=context,
            run_as_of_date=run_as_of_date,
            status="INVALID_FINANCIAL_INPUT",
            violations=(violation,),
        )
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SectorTriageResult:
    sector: str
    candidates_count: int
    survivors: list[dict[str, Any]] = field(default_factory=list)
    surprises: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class SectorDeepResult:
    ticker: str
    triage_rank: int = 0
    triage_reasoning: str = ""
    verdict: str = ""
    confidence: str = ""
    thesis_summary: str = ""
    key_numbers: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    reasoning_trace: str = ""
    current_price: float | None = None
    buy_below_price: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class _AnthropicUsageProvider:
    """Pricing identity for direct Anthropic SDK calls."""

    model: str
    provider_name: str = "anthropic"


@dataclass(frozen=True)
class _AnthropicUsageResult:
    response: Any
    model: str

    @property
    def usage(self) -> Any:
        return getattr(self.response, "usage", None)

    @property
    def usage_input_tokens(self) -> int:
        return int(getattr(self.usage, "input_tokens", 0) or 0)

    @property
    def usage_output_tokens(self) -> int:
        return int(getattr(self.usage, "output_tokens", 0) or 0)

    @property
    def json_text(self) -> str:
        return ""


def _sector_usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "provider_usage": list(records),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in records),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in records),
        "cost_usd": round(
            sum(float(item.get("cost_estimate_usd") or 0.0) for item in records),
            6,
        ),
        "physical_calls": len(records),
    }


def _call_metered_anthropic(
    *,
    client: Any,
    request: dict[str, Any],
    schema_name: str,
    lane: str,
) -> tuple[Any, list[dict[str, Any]]]:
    """Reserve and account for one direct Anthropic Messages request."""

    provider = _AnthropicUsageProvider(model=str(request.get("model") or "unknown"))
    max_output_tokens = max(1, int(request.get("max_tokens") or 1))
    prompt = json.dumps(
        {
            "system": request.get("system"),
            "messages": request.get("messages") or [],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    schema = {
        "type": "object",
        "transport_tools": request.get("tools") or [],
        "tool_choice": request.get("tool_choice"),
    }
    failed_attempts: list[dict[str, Any]] = []
    successful_attempts: list[dict[str, Any]] = []
    with (
        provider_usage_lane(lane),
        provider_usage_request(
            provider=provider,
            prompt=prompt,
            schema=schema,
            schema_name=schema_name,
            max_output_tokens=max_output_tokens,
        ),
    ):
        try:
            with provider_failed_attempt_capture(
                provider=provider,
                prompt=prompt,
                schema_name=schema_name,
                estimated_output_tokens=max_output_tokens,
            ) as failed_attempts:
                response = client.messages.create(**request)
        except BaseException as exc:
            successful_attempts = provider_usage_records_from_exception(
                provider=provider,
                error=exc,
                prompt=prompt,
                schema_name=schema_name,
            )
            try:
                for usage_record in successful_attempts:
                    record_provider_usage(usage_record)
                if not failed_attempts and not successful_attempts:
                    failure = failed_provider_usage_meta(
                        provider=provider,
                        prompt=prompt,
                        schema_name=schema_name,
                        estimated_output_tokens=max_output_tokens,
                        error=exc,
                    )
                    failed_attempts.append(failure)
                    record_provider_usage(failure)
            except LLMCostBudgetExceeded as budget_exc:
                attach_provider_usage_to_exception(
                    budget_exc,
                    [*failed_attempts, *successful_attempts],
                )
                raise budget_exc from exc
            attach_provider_usage_to_exception(
                exc,
                [*failed_attempts, *successful_attempts],
            )
            raise

        wrapped = _AnthropicUsageResult(
            response=response,
            model=provider.model,
        )
        successful_attempts = provider_usage_records(
            provider=provider,
            result=wrapped,
            prompt=prompt,
            schema_name=schema_name,
        )
        try:
            for usage_record in successful_attempts:
                record_provider_usage(usage_record)
        except LLMCostBudgetExceeded as exc:
            attach_provider_usage_to_exception(
                exc,
                [*failed_attempts, *successful_attempts],
            )
            raise
    return response, [*failed_attempts, *successful_attempts]


# ---------------------------------------------------------------------------
# Stage 1: Load + pre-rank
# ---------------------------------------------------------------------------

CAP_TIERS = {
    "micro": (0, 300),
    "small": (300, 2_000),
    "mid": (2_000, 10_000),
    "large": (10_000, 200_000),
    "mega": (200_000, float("inf")),
}


def _scorecard_price(scorecard: dict[str, Any]) -> float | None:
    """Scorecard-as-of price (not live) for replayable strict-cap resolution."""
    import math

    pzd = scorecard.get("pricing_zone_detail") or {}
    price = pzd.get("current_price")
    if not isinstance(price, (int, float)) or math.isnan(price) or price <= 0:
        return None
    return float(price)


def _latest_scorecard_evidence(
    tickers: list[str],
    *,
    as_of_date: str,
    db_path: str | Path,
) -> dict[str, tuple[str | None, dict[str, Any]]]:
    """Load each ticker's latest scorecard at or before ``as_of_date``.

    This is deliberately optional evidence. A missing valuations table, a
    missing row, or malformed JSON leaves an empty scorecard rather than
    removing a security from v2 membership.
    """

    normalized = list(
        dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    )
    evidence = {ticker: (None, {}) for ticker in normalized}
    if not normalized or not Path(db_path).exists():
        return evidence

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        for ticker in normalized:
            issuer_scope = resolve_filing_issuer_scope(conn, ticker)
            row = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker,
                method="scorecard",
                as_of_date=as_of_date,
                expected_issuer_cik=issuer_scope.issuer_cik,
                expected_issuer_aliases=issuer_scope.aliases,
                require_exact_issuer_binding=True,
            )
            if row is None:
                continue
            try:
                payload = json.loads(row["outputs_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            evidence[ticker] = (
                str(row["as_of_date"]) if row["as_of_date"] else None,
                payload if isinstance(payload, dict) else {},
            )
    except sqlite3.OperationalError:
        return evidence
    finally:
        conn.close()
    return evidence


def classify_tickers_for_market_cap(
    *,
    tickers: list[str],
    as_of_date: str,
    db_path: str | Path | None = None,
    pipeline_version: str = "v1",
    terminal_cap_lookup: Any | None = None,
    scorecard_evidence: dict[str, tuple[str | None, dict[str, Any]]] | None = None,
    allow_live_market_data: bool = True,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Classify every supplied security through the shared PIT cap chain.

    Explicit lists, candidate pools, and DB sector membership all call this
    helper in v2. A scorecard quote is eligible only when it is dated exactly
    to the fixed run date; otherwise the price/provider or direct-cap lanes
    must establish point-in-time evidence themselves.
    """

    from app.autonomous.cap_resolver import (
        classify_market_cap_for_band_filter,
        default_price_lookup,
    )
    from app.autonomous.terminal_cap_evidence import terminal_cap_lookup_from_path
    from app.config import get_config

    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    normalized_as_of = str(as_of_date or "").strip()[:10]
    if not normalized_as_of:
        raise ValueError("as_of_date is required for market-cap classification")
    normalized_tickers = list(
        dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    )
    resolved_cfg = cfg or get_config()
    resolved_db_path = Path(db_path) if db_path is not None else Path(resolved_cfg.db_path)
    resolved_cfg = resolved_cfg.model_copy(update={"db_path": resolved_db_path})
    # Cost-preflight discovery is a read-only local/PIT freeze.  A terminal
    # callback may be provider/search backed, so only a prevalidated local
    # ledger remains eligible when live market data is disabled.
    effective_terminal_cap_lookup = (
        terminal_cap_lookup
        if allow_live_market_data
        or bool(getattr(terminal_cap_lookup, "_voe_prevalidated_local_ledger", False))
        else None
    )
    if normalized_pipeline == "v2" and effective_terminal_cap_lookup is None:
        evidence_path = resolved_cfg.cap_terminal_evidence_path
        if evidence_path is not None:
            effective_terminal_cap_lookup = terminal_cap_lookup_from_path(evidence_path)
    evidence = scorecard_evidence or _latest_scorecard_evidence(
        normalized_tickers,
        as_of_date=normalized_as_of,
        db_path=resolved_db_path,
    )
    if allow_live_market_data:
        price_lookup = default_price_lookup(resolved_cfg)
    else:
        from app.market.split_evidence import load_persisted_split_lineage_quote

        def price_lookup(ticker: str, requested_as_of: str) -> Any:
            return load_persisted_split_lineage_quote(
                ticker,
                requested_as_of,
                db_path=resolved_db_path,
                cfg=resolved_cfg,
            )
    offline_cfg = (
        resolved_cfg
        if allow_live_market_data
        else resolved_cfg.model_copy(update={"cap_eodhd_fundamentals_enabled": False})
    )
    classifications: dict[str, Any] = {}
    for ticker in normalized_tickers:
        valuation_as_of, scorecard = evidence.get(ticker, (None, {}))
        scorecard_price = _scorecard_price(scorecard)
        price_detail = scorecard.get("pricing_zone_detail") or {}
        scorecard_price_as_of = str(price_detail.get("current_price_as_of_date") or "").strip()[:10]
        scorecard_price_currency = (
            str(price_detail.get("current_price_currency") or "").strip().upper()
        )
        scorecard_price_source = str(price_detail.get("current_price_source") or "").strip()
        use_scorecard_price = bool(
            scorecard_price is not None
            and scorecard_price_as_of
            and scorecard_price_as_of <= normalized_as_of
            and scorecard_price_currency == "USD"
            and scorecard_price_source
        )
        classification_as_of = normalized_as_of
        classification_kwargs: dict[str, Any] = dict(
            ticker=ticker,
            as_of_date=classification_as_of,
            asof_price=scorecard_price if use_scorecard_price else None,
            asof_price_provenance=(
                {
                    "as_of_date": scorecard_price_as_of,
                    "source": scorecard_price_source,
                    "source_url": price_detail.get("current_price_source_url"),
                    "currency": scorecard_price_currency,
                    "price_basis": price_detail.get("current_price_basis"),
                    "raw_price": price_detail.get("current_raw_price")
                    or price_detail.get("raw_price"),
                    "split_adjustment_factor": price_detail.get("split_adjustment_factor"),
                    "split_effective_date": price_detail.get("split_effective_date"),
                    "split_event": price_detail.get("split_event"),
                    "no_intervening_split_proof": price_detail.get("no_intervening_split_proof"),
                    "confidence": "MEDIUM",
                }
                if use_scorecard_price
                else None
            ),
            db_path=resolved_db_path,
            price_lookup=price_lookup,
            terminal_cap_lookup=effective_terminal_cap_lookup,
            pipeline_version=normalized_pipeline,
            companyfacts_cache_only=not allow_live_market_data,
        )
        classification_kwargs["cfg"] = offline_cfg
        classifications[ticker] = classify_market_cap_for_band_filter(**classification_kwargs)
    return classifications


def load_sector_tickers_classified(
    *,
    sector: str,
    explicit_tickers: list[str] | tuple[str, ...] | None = None,
    db_path: str | Path | None = None,
    cap_min: float | None = None,
    cap_max: float | None = None,
    as_of_date: str | None = None,
    pipeline_version: str = "v1",
    terminal_cap_lookup: Any | None = None,
    allow_live_market_data: bool = True,
    cap_classification_cache: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    """Load sector rows plus the band-filter cap classification per ticker.

    ``explicit_tickers`` bypasses sector-membership discovery but deliberately
    reuses the same scorecard-evidence, classification, cache, and cap-band
    machinery. The returned classifications still cover every explicit name,
    including out-of-band and unknown-cap names; callers control whether the
    filtered row list is authoritative for their selection semantics.

    Cap filtering runs through the fallback chain
    (app/autonomous/cap_resolver.py): strict as-of cap first, then last-known
    companyfacts shares x current price (cap_source=stale_shares). A name
    whose cap resolves OUT of the requested band is excluded — this is what
    keeps a ~$9B name out of a micro sweep. Residual unknown-cap names stay
    sweepable (included) and must be rendered with the UNKNOWN_CAP label by
    every downstream surface. Classifications are returned for ALL examined
    rows, including the excluded ones, keyed by ticker.
    """
    from app.config import get_config

    if db_path is None:
        db_path = get_config().db_path

    normalized_pipeline = str(pipeline_version or "v1").strip().lower()
    if normalized_pipeline not in {"v1", "v2"}:
        raise ValueError("pipeline_version must be v1 or v2")
    requested_as_of = str(as_of_date or "").strip()[:10] or date.today().isoformat()
    normalized_explicit = list(
        dict.fromkeys(
            str(ticker).strip().upper() for ticker in explicit_tickers or () if str(ticker).strip()
        )
    )

    v1_query = """
            SELECT si.ticker,
                   si.as_of_date AS membership_as_of_date
            FROM sector_inference si
            WHERE si.inferred_sector = :sector
              AND si.as_of_date <= :as_of_date
              AND si.as_of_date = (
                  SELECT MAX(si2.as_of_date)
                  FROM sector_inference si2
                  WHERE si2.ticker = si.ticker
                    AND si2.inferred_sector IS NOT NULL
                    AND si2.as_of_date <= :as_of_date
              )
              {removed_clause}
            ORDER BY si.ticker
            """
    # Membership is led by sector inference, not valuations.  V1 uses the
    # latest available scorecard while v2 binds optional valuation evidence
    # point-in-time.  Both keep members with no/malformed scorecard visible;
    # otherwise candidate loading can silently define away missing data.
    v2_query = """
            SELECT si.ticker,
                   si.as_of_date AS membership_as_of_date,
                   {registry_removed_expr} AS registry_removed_at
            FROM sector_inference si
            WHERE si.inferred_sector = :sector
              AND si.as_of_date <= :as_of_date
              AND si.as_of_date = (
                  SELECT MAX(si2.as_of_date)
                  FROM sector_inference si2
                  WHERE si2.ticker = si.ticker
                    AND si2.as_of_date <= :as_of_date
              )
            ORDER BY si.ticker
            """
    # Sync-marked registry exits never re-enter sweep scope. A ticker
    # known to sec_registrants with removed_at set is out; names absent from
    # the registrant table (and minimal DBs without the table at all) stay
    # sweepable.
    removed_clause = """
              AND NOT EXISTS (
                  SELECT 1 FROM sec_registrants sr
                  WHERE sr.primary_ticker = si.ticker
                    AND sr.removed_at IS NOT NULL
              )
            """

    conn: sqlite3.Connection | None = None
    try:
        rows: list[sqlite3.Row] = []
        if not normalized_explicit:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            base_query = v2_query if normalized_pipeline == "v2" else v1_query
            query_parameters = {"sector": sector, "as_of_date": requested_as_of}
            try:
                rows = conn.execute(
                    base_query.format(
                        removed_clause=removed_clause,
                        registry_removed_expr=(
                            "(SELECT MIN(sr.removed_at) FROM sec_registrants sr "
                            "WHERE sr.primary_ticker = si.ticker)"
                            if normalized_pipeline == "v2"
                            else "NULL"
                        ),
                    ),
                    query_parameters,
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table: sec_registrants" not in str(exc):
                    raise
                rows = conn.execute(
                    base_query.format(
                        removed_clause="",
                        registry_removed_expr="NULL",
                    ),
                    query_parameters,
                ).fetchall()

        result: list[tuple[str, str, dict[str, Any]]] = []
        parsed_rows: list[tuple[str, str, dict[str, Any], str | None]] = []
        membership_tickers = normalized_explicit or [
            str(row["ticker"]).strip().upper() for row in rows if str(row["ticker"]).strip()
        ]
        scorecard_evidence = _latest_scorecard_evidence(
            membership_tickers,
            as_of_date=requested_as_of,
            db_path=db_path,
        )
        if normalized_explicit:
            for ticker in normalized_explicit:
                valuation_as_of, scorecard = scorecard_evidence.get(
                    ticker,
                    (None, {}),
                )
                stable_row_as_of = str(valuation_as_of or requested_as_of)
                parsed_rows.append((ticker, stable_row_as_of, scorecard, None))
        else:
            for row in rows:
                ticker = str(row["ticker"]).strip().upper()
                valuation_as_of, scorecard = scorecard_evidence.get(
                    ticker,
                    (None, {}),
                )
                stable_row_as_of = str(
                    valuation_as_of or row["membership_as_of_date"] or requested_as_of
                )
                removed_at = (
                    str(row["registry_removed_at"]).strip()
                    if normalized_pipeline == "v2"
                    and "registry_removed_at" in row.keys()
                    and row["registry_removed_at"]
                    else None
                )
                parsed_rows.append((ticker, stable_row_as_of, scorecard, removed_at))

        should_classify = (
            bool(normalized_explicit)
            or normalized_pipeline == "v2"
            or cap_min is not None
            or cap_max is not None
        )
        classifications: dict[str, Any] = {}
        if should_classify:
            tickers_to_classify = [ticker for ticker, _, _, _ in parsed_rows]
            if cap_classification_cache is not None:
                classifications.update(
                    {
                        ticker: cap_classification_cache[ticker]
                        for ticker in tickers_to_classify
                        if ticker in cap_classification_cache
                    }
                )
                tickers_to_classify = [
                    ticker for ticker in tickers_to_classify if ticker not in classifications
                ]
            if tickers_to_classify:
                resolved = classify_tickers_for_market_cap(
                    tickers=tickers_to_classify,
                    as_of_date=str(requested_as_of or date.today().isoformat()),
                    db_path=db_path,
                    pipeline_version=normalized_pipeline,
                    terminal_cap_lookup=terminal_cap_lookup,
                    scorecard_evidence=scorecard_evidence,
                    allow_live_market_data=allow_live_market_data,
                )
                classifications.update(resolved)
                if cap_classification_cache is not None:
                    cap_classification_cache.update(resolved)
        if normalized_pipeline == "v2":
            for ticker, _, _, removed_at in parsed_rows:
                removed_date = str(removed_at or "")[:10]
                if removed_date and removed_date <= str(requested_as_of)[:10]:
                    classifications[ticker] = replace(
                        classifications[ticker],
                        scope_status="OUT_OF_SCOPE",
                        scope_reason="SEC_REGISTRANT_REMOVED_AS_OF_SCAN",
                    )
        filtered_out = 0
        removed_out_of_scope = 0
        unknown_cap = 0
        stale_resolved_out = 0
        for ticker, row_as_of, scorecard, _removed_at in parsed_rows:
            classification = classifications.get(ticker)
            if (
                normalized_pipeline == "v2"
                and classification is not None
                and classification.scope_status == "OUT_OF_SCOPE"
            ):
                removed_out_of_scope += 1
                continue
            if cap_min is not None or cap_max is not None:
                classification = classifications[ticker]
                in_band = classification.in_band(cap_min, cap_max)
                if in_band is None:
                    unknown_cap += 1
                    # Residual unknown stays sweepable; surfaces label it UNKNOWN_CAP.
                    result.append((ticker, row_as_of, scorecard))
                    continue
                if not in_band:
                    filtered_out += 1
                    if classification.cap_source == "stale_shares":
                        stale_resolved_out += 1
                    continue

            result.append((ticker, row_as_of, scorecard))

        if filtered_out or unknown_cap or removed_out_of_scope:
            logger.info(
                "load_sector_tickers %s: %d loaded, %d filtered by cap "
                "(%d resolved out-of-band via stale_shares), %d unknown cap "
                "(included), %d registry removals out of scope",
                sector,
                len(result),
                filtered_out,
                stale_resolved_out,
                unknown_cap,
                removed_out_of_scope,
            )

        return result, classifications
    finally:
        if conn is not None:
            conn.close()


def load_sector_tickers(
    *,
    sector: str,
    db_path: str | Path | None = None,
    cap_min: float | None = None,
    cap_max: float | None = None,
    as_of_date: str | None = None,
    allow_live_market_data: bool = True,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Load (ticker, as_of_date, scorecard_dict) for all tickers in a sector.

    Back-compat wrapper over load_sector_tickers_classified; see that
    function for the band-filter chain semantics.
    """
    rows, _classifications = load_sector_tickers_classified(
        sector=sector,
        db_path=db_path,
        cap_min=cap_min,
        cap_max=cap_max,
        as_of_date=as_of_date,
        allow_live_market_data=allow_live_market_data,
    )
    return rows


def pre_rank_sector(
    *,
    tickers: list[str],
    as_of_date: str,
    db_path: str | Path | None = None,
    limit: int = 50,
    filing_risk_use_llm: bool = False,
    include_blocked: bool = False,
    cfg: AppConfig | None = None,
) -> list[RankedEntry]:
    """Deterministic consensus pre-rank.

    Uses the same provider-free, fixed-as-of canonical V1 context as the
    downstream financial-integrity scope, then ranks its exact packets.
    Returns top ``limit`` entries sorted by consensus score descending.
    """
    if filing_risk_use_llm:
        logger.warning(
            "Ignoring filing_risk_use_llm=True during deterministic pre-rank; "
            "paid filing classification requires an authorized post-assembly scope."
        )
    normalized_as_of = str(as_of_date or "").strip()[:10]
    if not normalized_as_of:
        raise ValueError("as_of_date is required for deterministic pre-rank")
    financial_context = build_canonical_v1_financial_context(
        tickers=tickers,
        as_of_date=normalized_as_of,
        db_path=db_path,
        cfg=cfg,
    )
    result = rank_by_consensus(
        financial_context.packets,
        include_blocked=include_blocked,
    )
    combined = result.ranked + result.ranked_insufficient
    combined.sort(key=lambda e: e.consensus_score, reverse=True)
    return combined[:limit] if limit > 0 else combined


# ---------------------------------------------------------------------------
# Stage 2: AI comparative triage
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM = """You are a fundamental analyst performing sector-wide comparative triage.

You receive every company in a sector with a compact deterministic scorecard summary.
Use ONLY the evidence provided in those summaries to decide which names deserve an
expensive deep review. Do not assume access to management interviews, earnings calls,
competitor data, or facts that are not shown in the packet.

YOUR JOB:
- Compare companies against each other, not against an abstract ideal.
- Find names where the current packet suggests one of three things:
  1. possible undervaluation worth verifying,
  2. an important disagreement or tension that deeper work could resolve, or
  3. a possible quality or growth story that the static model may be underrating.
- Also identify names that should NOT consume deep-review budget because the packet
  already points to an obvious PASS, obvious WATCH, or insufficient evidence.

WHAT YOU CAN RELY ON HERE:
- Valuation anchors in the scorecard: current price, DCF, EPV, Graham, method agreement,
  and tension type.
- Deterministic quality/risk signals already present in the packet: solvency status,
  filing risk, anomaly flags, growth trends, cycle position, and other summary fields.
- Relative comparison across peers in the same sector.

HOW TO THINK:
1. Prefer companies where the upside/downside question is still unresolved by the packet.
2. Treat DCF, EPV, and Graham as starting points, not automatic verdicts.
3. Penalize thin or contradictory evidence unless the uncertainty itself is the reason
   to investigate.
4. Do not pick a company only because it looks cheap.
5. Do not exclude a small company just because it is small, but only include it when
   the packet gives a concrete reason it may be mispriced or misunderstood.
6. Do not invent claims about management quality, market share, moat, or capital
   allocation history unless the summary explicitly supports them.

ADDITIONAL GUIDANCE:
- Cheap plus deteriorating cash generation is usually a value trap — do not confuse
  low price with high quality.
- Do not let size alone exclude a name when the packet shows unusual valuation/quality
  tension. A small company with strong scorecard signals deserves investigation.

OUTPUT STANDARD:
- Return only the companies that deserve deep review.
- Rank them by expected value of further investigation, not by market cap or raw cheapness.
- In each reasoning field, explain what stands out relative to sector peers and what a
  deep review needs to confirm.

Output via the sector_triage tool."""


_TRIAGE_TOOL = {
    "name": "sector_triage",
    "description": "Return the ranked list of survivors from comparative sector triage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "survivors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "rank": {"type": "integer"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["ticker", "rank", "reasoning"],
                },
            },
            "surprises": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Notable observations — companies surprisingly good or excluded.",
            },
        },
        "required": ["survivors", "surprises"],
    },
}


def _build_triage_prompt(
    sector: str,
    ranked_tickers: list[str],
    canonical_packets: dict[str, dict[str, Any]],
    survivors_target: int,
    integrity_scope_fingerprint: str,
    commodity_block: str = "",
) -> str:
    """Build the user message for sector triage.

    ``commodity_block`` is retained for legacy prompt rendering only. Paid V1
    entry points reject non-empty values until exact PIT provenance is bound.
    """
    lines = [
        f"=== SECTOR SCAN: {sector} ===",
        f"Tickers: {len(ranked_tickers)}",
        "Pre-ranked by deterministic consensus scoring (multi-method margin of safety).",
        f"financial_integrity_scope_fingerprint: {integrity_scope_fingerprint}",
        "The canonical packet JSON below is the sole authority for every financial value.",
        "",
        "Compare these companies AGAINST EACH OTHER, not in isolation.",
        "Focus on relative differences visible in the packet:",
        "valuation gaps, method disagreement, growth/quality signals,",
        "solvency or filing-risk flags, and where deeper review could change the verdict.",
        "",
        f"Pick however many deserve a deep dive — there is no fixed number.",
        f"If only 3 are genuinely interesting, pick 3. If 30 are interesting, pick 30.",
        f"Maximum allowed: {survivors_target}. But do NOT pad to reach this number.",
        "For each, explain what makes it stand out FROM ITS PEERS.",
    ]
    if commodity_block:
        lines.append(commodity_block)
    lines.extend(
        [
            "",
            "=== CANDIDATES (ranked by deterministic consensus score) ===",
            "",
        ]
    )
    for i, ticker in enumerate(ranked_tickers, 1):
        summary = json.dumps(
            canonical_packets[str(ticker).strip().upper()],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        lines.append(f"#{i}: {ticker}")
        lines.append(summary)
        lines.append("")
    return "\n".join(lines)


def run_sector_triage(
    *,
    client,
    sector: str,
    ranked_tickers: list[str],
    scorecards: dict[str, dict[str, Any]],
    config: ScanConfig,
    integrity_scope: FinancialIntegrityScope,
    commodity_block: str = "",
) -> SectorTriageResult:
    """Run comparative triage: single API call, all candidates side-by-side."""
    _reject_unbound_commodity_context(
        context=f"sector_triage:{sector}",
        run_as_of_date=integrity_scope.run_as_of_date,
        commodity_block=commodity_block,
    )
    integrity_result = require_financial_integrity_scope(integrity_scope)
    canonical_packets = {
        str(getattr(packet, "ticker", "") or "").strip().upper(): asdict(packet)
        for packet in integrity_scope.packets
    }
    missing_tickers = [
        ticker for ticker in ranked_tickers if str(ticker).strip().upper() not in canonical_packets
    ]
    if missing_tickers:
        raise ValueError(
            "triage candidates are not bound to the financial-integrity scope: "
            + ", ".join(missing_tickers)
        )
    survivors_target = min(config.max_deep_reviews, len(ranked_tickers))
    prompt = _build_triage_prompt(
        sector,
        ranked_tickers,
        canonical_packets,
        survivors_target,
        integrity_result.scope_fingerprint,
        commodity_block=commodity_block,
    )

    require_unchanged_financial_integrity_scope(
        integrity_scope,
        expected_scope_fingerprint=integrity_result.scope_fingerprint,
    )
    request = {
        "model": config.triage_model,
        "max_tokens": config.triage_max_output_tokens,
        "system": _TRIAGE_SYSTEM,
        "tools": [_TRIAGE_TOOL],
        "tool_choice": {"type": "tool", "name": "sector_triage"},
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response, usage_records = _call_metered_anthropic(
            client=client,
            request=request,
            schema_name="sector_triage_v1",
            lane=f"sector_triage:{sector}",
        )
    except Exception as exc:
        try:
            require_unchanged_financial_integrity_scope(
                integrity_scope,
                expected_scope_fingerprint=integrity_result.scope_fingerprint,
            )
        except InvalidFinancialInputError as integrity_exc:
            raise integrity_exc from exc
        raise

    usage_summary = _sector_usage_summary(usage_records)
    require_unchanged_financial_integrity_scope(
        integrity_scope,
        expected_scope_fingerprint=integrity_result.scope_fingerprint,
    )

    survivors: list[dict[str, Any]] = []
    surprises: list[str] = []
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "sector_triage"
        ):
            tool_input = block.input or {}
            survivors = list(tool_input.get("survivors", []))
            surprises = list(tool_input.get("surprises", []))
            break

    logger.info(
        "scan %s triage: %d candidates -> %d survivors ($%.4f)",
        sector,
        len(ranked_tickers),
        len(survivors),
        usage_summary["cost_usd"],
    )
    result = SectorTriageResult(
        sector=sector,
        candidates_count=len(ranked_tickers),
        survivors=survivors,
        surprises=surprises,
        input_tokens=int(usage_summary["input_tokens"]),
        output_tokens=int(usage_summary["output_tokens"]),
        cost_usd=float(usage_summary["cost_usd"]),
    )
    result._prompt = prompt  # type: ignore[attr-defined]  # audit artifact
    result._provider_usage = usage_records  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# Stage 3: Deep financial review per finalist (agentic, multi-turn)
# Uses Stage 4's deep_research_ticker loop with tools:
#   fetch_current_price, fetch_filing_section, fetch_companyfacts,
#   fetch_historical_scorecards, finalize_analysis
# ---------------------------------------------------------------------------


def _build_financials_block(
    ticker: str,
    db_path: str | Path | None = None,
    market_cap_m: float | None = None,
    current_price: float | None = None,
    as_of_date: str | None = None,
) -> str:
    """Pull key financial time series from companyfacts for the deep review prompt.

    Gives the AI actual balance sheet, income, and cash flow numbers to
    resolve solvency flags, compute margins, and validate earnings power
    instead of speculating.

    The returned block has TWO sections:
      1. KEY FINANCIALS — multi-year time series of XBRL line items
      2. DERIVED METRICS — single-year valuation/leverage/quality math
         pre-computed so the AI doesn't have to recompute (and risk arithmetic
         errors) ad hoc in its reasoning.
    """
    if db_path is None:
        from app.config import get_config

        db_path = get_config().db_path

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        line_items = [
            # Income statement
            "revenue",
            "net_income",
            "operating_income",
            "gross_profit",
            # Cash flow
            "cfo",
            "capex",
            "depreciation_amortization",
            "intangible_amortization",
            "sbc",
            "share_repurchases_amount",
            "dividends_paid_amount",
            # Balance sheet — full picture (was missing several before this fix)
            "total_assets",
            "total_liabilities",
            "equity",
            "cash",
            "total_debt",
            "current_assets",
            "current_liabilities",
            "accounts_receivable",
            "accounts_payable",
            "inventory",
            "goodwill",
            "intangible_assets",
            "investment_securities",
            "deferred_revenue",
            "operating_lease_liability",
            # Other
            "shares_outstanding",
            "interest_expense",
            "r_and_d_total",
            "restructuring_charges",
        ]
        try:
            rows = companyfacts_rows(
                conn,
                ticker,
                columns=("fiscal_year", "line_item", "value"),
                period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
                line_items=line_items,
                as_of_date=as_of_date,
                require_filed_asof=bool(as_of_date),
                order_by="line_item ASC, fiscal_year DESC",
            )
        except sqlite3.OperationalError:
            # Table doesn't exist (test environment or empty DB)
            rows = []
    finally:
        conn.close()

    if not rows:
        return "\n=== FINANCIAL DATA: NOT AVAILABLE ===\n"

    # Group by line item, show last 5 years
    from collections import defaultdict

    by_item: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by_item[r["line_item"]].append((r["fiscal_year"], r["value"]))

    lines = ["\n=== KEY FINANCIALS (from EDGAR companyfacts, millions USD) ===\n"]
    for item in line_items:
        series = by_item.get(item, [])
        if not series:
            continue
        # Latest 5 years, newest first
        series = sorted(series, key=lambda x: x[0], reverse=True)[:5]
        values = " | ".join(f"FY{y}: {v:,.0f}" for y, v in series)
        lines.append(f"{item}: {values}")

    # Build latest-year values dictionary
    latest: dict[str, float] = {}
    for item in line_items:
        series = sorted(by_item.get(item, []), key=lambda x: x[0])
        if series:
            latest[item] = series[-1][1]

    # Multi-year shares (for CAGR computation)
    shares_series = sorted(by_item.get("shares_outstanding", []), key=lambda x: x[0])

    # ── DERIVED METRICS BOX ────────────────────────────────────────────────
    # Pre-compute the math the AI would otherwise derive ad hoc — gives a
    # single source of truth and frees the AI's tokens for qualitative work.
    lines.append("")
    lines.append("=== DERIVED METRICS (pre-computed; use these directly) ===")
    lines.append("")

    rev = latest.get("revenue")
    cfo = latest.get("cfo")
    capex = abs(latest.get("capex") or 0.0) if latest.get("capex") is not None else None
    fcf = cfo - capex if (cfo is not None and capex is not None) else None
    net_income = latest.get("net_income")
    op_income = latest.get("operating_income")
    cash = latest.get("cash")
    sec = latest.get("investment_securities")
    debt = latest.get("total_debt")
    interest = latest.get("interest_expense")
    sbc = latest.get("sbc")
    equity = latest.get("equity")
    da = latest.get("depreciation_amortization")
    ia = latest.get("intangible_amortization")

    # Profitability + cash conversion
    if rev and rev > 0:
        if net_income is not None:
            lines.append(f"  net_margin: {net_income / rev:.1%}")
        if op_income is not None:
            lines.append(f"  operating_margin: {op_income / rev:.1%}")
        if cfo is not None:
            lines.append(f"  cfo_margin: {cfo / rev:.1%}")
        if fcf is not None:
            lines.append(f"  fcf: {fcf:,.0f}M  fcf_margin: {fcf / rev:.1%}")
        if latest.get("r_and_d_total"):
            lines.append(f"  r_and_d_intensity: {latest['r_and_d_total'] / rev:.1%}")
        if sbc is not None:
            lines.append(f"  sbc_pct_revenue: {sbc / rev:.1%}")
    if cfo is not None and op_income and op_income > 0:
        lines.append(f"  cfo_conversion (cfo/op_income): {cfo / op_income:.2f}x")
    if fcf is not None and net_income and net_income > 0:
        lines.append(f"  fcf_conversion (fcf/net_income): {fcf / net_income:.2f}x")

    # Cash, debt, leverage
    cash_is_finite = (
        isinstance(cash, (int, float)) and not isinstance(cash, bool) and math.isfinite(float(cash))
    )
    debt_is_finite = (
        isinstance(debt, (int, float)) and not isinstance(debt, bool) and math.isfinite(float(debt))
    )
    securities_are_finite = (
        isinstance(sec, (int, float)) and not isinstance(sec, bool) and math.isfinite(float(sec))
    )
    # Net debt requires explicit, finite cash and debt.  Reported investment
    # securities may augment cash, but an absent securities fact is not
    # silently converted to zero.
    cash_plus_sec = float(cash) if cash_is_finite else None
    if cash_plus_sec is not None and securities_are_finite:
        cash_plus_sec += float(sec)
    if cash_plus_sec is not None:
        components = []
        if cash_is_finite:
            components.append(f"cash {cash:,.0f}")
        if securities_are_finite:
            components.append(f"securities {sec:,.0f}")
        lines.append(f"  cash_plus_securities: {cash_plus_sec:,.0f}M ({' + '.join(components)})")
    net_debt = None
    if debt_is_finite and cash_plus_sec is not None:
        net_debt = float(debt) - cash_plus_sec
        lines.append(
            f"  net_debt: {net_debt:,.0f}M (debt {debt:,.0f} - cash+sec {cash_plus_sec:,.0f})"
        )
        if cfo and cfo > 0:
            lines.append(f"  net_debt_to_cfo: {net_debt / cfo:.2f}x")
    else:
        missing_components = []
        if not debt_is_finite:
            missing_components.append("total_debt")
        if not cash_is_finite:
            missing_components.append("cash")
        lines.append(
            f"  net_debt: NEEDS_DATA (explicit finite {' and '.join(missing_components)} required)"
        )
    if interest and interest > 0:
        if op_income is not None:
            lines.append(f"  interest_coverage (op_income/interest): {op_income / interest:.1f}x")
        if cfo is not None:
            lines.append(f"  cfo_to_interest: {cfo / interest:.1f}x")

    # Working capital + liquidity
    ca = latest.get("current_assets")
    cl = latest.get("current_liabilities")
    if ca is not None and cl and cl > 0:
        lines.append(f"  current_ratio: {ca / cl:.2f}x")
        if latest.get("inventory") is not None:
            quick = ca - latest["inventory"]
            lines.append(f"  quick_ratio (excl inventory): {quick / cl:.2f}x")
        wc = ca - cl
        lines.append(f"  working_capital: {wc:,.0f}M")

    # Capital structure quality
    if (
        latest.get("goodwill") is not None
        and latest.get("total_assets")
        and latest["total_assets"] > 0
    ):
        lines.append(f"  goodwill_pct_assets: {latest['goodwill'] / latest['total_assets']:.1%}")
    if (
        latest.get("intangible_assets") is not None
        and latest.get("total_assets")
        and latest["total_assets"] > 0
    ):
        lines.append(
            f"  intangibles_pct_assets: {latest['intangible_assets'] / latest['total_assets']:.1%}"
        )
    if ia is not None and rev and rev > 0:
        lines.append(f"  intangible_amortization_pct_revenue: {ia / rev:.1%}  (DIRECT from XBRL)")
    elif da is not None and capex is not None and rev and rev > 0:
        excess_da = max(0.0, da - capex)
        if excess_da > 0:
            lines.append(
                f"  excess_da_pct_revenue: {excess_da / rev:.1%}  (proxy for intangible amort)"
            )

    # Returns
    if equity and equity > 0 and net_income is not None:
        lines.append(f"  roe: {net_income / equity:.1%}")

    # Shares CAGR
    if len(shares_series) >= 4:
        # 3y CAGR — current vs 3 years ago
        try:
            recent = shares_series[-1]
            three_back = shares_series[-4]
            if three_back[1] and three_back[1] > 0 and (recent[0] - three_back[0]) > 0:
                yrs = recent[0] - three_back[0]
                cagr = (recent[1] / three_back[1]) ** (1 / yrs) - 1
                lines.append(
                    f"  shares_cagr_3y: {cagr * 100:+.1f}%/yr  ({three_back[1]:,.0f}M FY{three_back[0]} -> {recent[1]:,.0f}M FY{recent[0]})"
                )
        except Exception:
            pass

    # Market valuation multiples (only if market cap available)
    if market_cap_m and market_cap_m > 0:
        lines.append(f"  market_cap: {market_cap_m:,.0f}M")
        if net_debt is None:
            lines.append(
                "  enterprise_value (EV): NEEDS_DATA (requires explicit finite total_debt and cash)"
            )
        else:
            ev = market_cap_m + net_debt
            lines.append(f"  enterprise_value (EV): {ev:,.0f}M  (mcap + net_debt)")
            if cfo and cfo > 0:
                lines.append(f"  ev_to_cfo: {ev / cfo:.1f}x")
            if fcf and fcf > 0:
                lines.append(f"  ev_to_fcf: {ev / fcf:.1f}x")
        if fcf and fcf > 0:
            lines.append(f"  fcf_yield (fcf/mcap): {fcf / market_cap_m:.1%}")
        if net_debt is not None and net_debt <= 0:
            net_cash_pct = max(0.0, -net_debt) / market_cap_m
            lines.append(f"  net_cash_pct_market_cap: {net_cash_pct:.1%}")

    lines.append("")
    return "\n".join(lines)


# Section char limits for scan deep review — larger than discover's defaults
# because scan processes fewer tickers and can afford more context per ticker
_SCAN_SECTION_LIMITS = {
    "mda": 20000,
    "risk_factors": 15000,
    "fin_notes": 8000,
    "business": 5000,
}


def _make_scan_context_builder(
    *,
    bundle_builder: Callable[..., Any],
    sector: str,
    triage_lookup: dict[str, dict[str, Any]],
    integrity_scopes: dict[str, FinancialIntegrityScope],
    db_path: str | Path | None = None,
    commodity_block: str = "",
) -> Callable[[str], dict[str, Any]]:
    """Build a context_builder for Stage 4 that includes sector context + financials.

    ``commodity_block`` is retained for legacy prompt rendering only. The paid
    V1 deep-review entry point rejects non-empty values until exact PIT
    provenance is bound.
    """

    def context_builder(ticker: str) -> dict[str, Any]:
        normalized_ticker = str(ticker).strip().upper()
        integrity_scope = integrity_scopes.get(normalized_ticker)
        if integrity_scope is None:
            raise ValueError(f"{normalized_ticker} is not bound to a deep-review integrity scope")
        integrity_result = require_financial_integrity_scope(integrity_scope)
        matching_packets = [
            packet
            for packet in integrity_scope.packets
            if str(getattr(packet, "ticker", "") or "").strip().upper() == normalized_ticker
        ]
        if len(matching_packets) != 1:
            raise ValueError(f"{normalized_ticker} requires exactly one canonical financial packet")
        packet = matching_packets[0]
        run_as_of_date = str(integrity_scope.run_as_of_date or "").strip()[:10]
        bundle = bundle_builder(ticker, as_of_date=run_as_of_date)
        snap = bundle.valuation

        for field_name, bundle_value, canonical_value in (
            ("current_price", snap.current_price, packet.current_price),
            ("market_cap", snap.market_cap, packet.market_cap_mm),
            ("dcf_base", snap.dcf_base, packet.dcf_value),
            ("epv_adjusted", snap.epv_adjusted, packet.epv_value),
            ("graham_value", snap.graham_value, packet.graham_value),
        ):
            if (
                isinstance(bundle_value, (int, float))
                and not isinstance(bundle_value, bool)
                and isinstance(canonical_value, (int, float))
                and not isinstance(canonical_value, bool)
                and not math.isclose(
                    float(bundle_value),
                    float(canonical_value),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"{normalized_ticker} bundle {field_name} does not match "
                    "the canonical financial packet"
                )

        scorecard_block = (
            f"current_price: {packet.current_price}\n"
            f"current_price_as_of_date: {packet.current_price_as_of_date}\n"
            f"current_price_source: {packet.current_price_source}\n"
            f"quote_snapshot_id: {packet.quote_snapshot_id}\n"
            f"market_cap: {packet.market_cap_mm}\n"
            f"market_cap_effective_as_of_date: {packet.market_cap_effective_as_of_date}\n"
            f"market_cap_source: {packet.market_cap_source}\n"
            f"shares_outstanding_mm: {packet.shares_outstanding_mm}\n"
            f"shares_as_of_date: {packet.shares_as_of_date}\n"
            f"shares_source: {packet.shares_source}\n"
            f"dcf_base: {packet.dcf_value}\n"
            f"epv_adjusted: {packet.epv_value}\n"
            f"graham_value: {packet.graham_value}\n"
            f"methods_agree: {packet.methods_agree}\n"
            f"tension_type: {packet.method_tension_type}\n"
            f"gate_action: {packet.gate_verdict}\n"
            f"solvency_status: {packet.solvency_risk}\n"
            f"filing_risk_status: {packet.filing_risk_status}\n"
            f"financial_integrity_scope_fingerprint: {integrity_result.scope_fingerprint}\n"
        )

        filings_block = ""
        filing_sections: dict[str, str] = {}
        if bundle.filings:
            f0 = bundle.filings[0]
            if str(f0.filing_date or "")[:10] > run_as_of_date:
                raise ValueError(f"{normalized_ticker} bundle includes a post-as-of filing")
            filing_sections = dict(f0.section_text)
            filings_block = (
                f"\nMost recent filing: {f0.form_type} filed {f0.filing_date}, "
                f"accession {f0.accession}\n"
                f"sections_available: {list(f0.section_text.keys())}\n"
                "(Use fetch_filing_section to pull any untruncated section text.)"
            )

        warnings_block = ""
        if bundle.warnings:
            warnings_block = f"\nbundle_warnings: {bundle.warnings}"

        triage = triage_lookup.get(ticker, {})
        rank = triage.get("rank", "?")
        reasoning = triage.get("reasoning", "")
        sector_block = (
            f"\n=== SECTOR CONTEXT ===\n"
            f"Sector: {sector}\n"
            f"This company ranked #{rank} in comparative triage.\n"
            f"Triage reasoning: {reasoning}\n"
        )

        # Deep review must consume the same immutable packet that passed the
        # financial-integrity gate.  A second CompanyFacts query here used to
        # create a parallel, un-fingerprinted financial truth inside the paid
        # prompt.  Serialize the authorized packet instead; tool calls below
        # return these exact frozen bytes.
        financials_block = (
            "\n=== CANONICAL FINANCIAL PACKET (financial-integrity bound) ===\n"
            + json.dumps(
                asdict(packet),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
        )

        user_message = (
            f"=== TICKER: {ticker} ===\n"
            f"as_of_date: {run_as_of_date}\n"
            f"analysis_years: {bundle.analysis_years}{warnings_block}\n"
            f"{sector_block}"
            f"{commodity_block}"
            f"{financials_block}"
            f"\n=== VALUATION SCORECARD ===\n{scorecard_block}"
            f"{filings_block}\n"
            "\nTOOLS AVAILABLE:\n"
            "- fetch_current_price(ticker) - read the immutable quote already bound to this run; never fetch or substitute a live price\n"
            "- fetch_filing_section(ticker, section_key) - read untruncated most recent 10-K sections: "
            "mda, risk_factors, fin_notes, business\n"
            "- fetch_companyfacts(ticker, line_items) - reread the fixed canonical financial packet embedded above\n"
            "- fetch_historical_scorecards(ticker, n_years) - unavailable in this fixed run; the canonical packet is authoritative\n"
            "\nUse the triage reasoning as a hypothesis, not a mandate. Start with the "
            "most important missing fact. You do not need to call every tool. Call "
            "finalize_analysis once the verdict is stable.\n"
        )
        # current_price exposed for report rendering (Stage 4 ignores extra keys)
        return {
            "ticker": normalized_ticker,
            "as_of_date": run_as_of_date,
            "user_message": user_message,
            "current_price": packet.current_price,
            "_fixed_financials_block": financials_block,
            "_fixed_filing_sections": filing_sections,
            "_fixed_filing_meta": (
                {
                    "form_type": f0.form_type,
                    "filing_date": f0.filing_date,
                    "accession": f0.accession,
                }
                if bundle.filings
                else {}
            ),
        }

    return context_builder


def run_sector_deep_review(
    *,
    client,
    sector: str,
    triage_survivors: list[dict[str, Any]],
    bundle_builder: Callable[..., Any],
    config: ScanConfig,
    integrity_scopes: dict[str, FinancialIntegrityScope],
    run_integrity_scope: FinancialIntegrityScope | None = None,
    run_integrity_scope_fingerprint: str = "",
    db_path: str | Path | None = None,
    commodity_block: str = "",
) -> tuple[list[SectorDeepResult], dict[str, str]]:
    """Run agentic deep review using Stage 4's multi-turn tool-calling loop.

    The AI has tools to fetch prices, filing sections, companyfacts, and
    historical scorecards — it pulls what it needs to resolve its own questions.
    """
    _reject_unbound_commodity_context(
        context=f"sector_deep_review:{sector}",
        run_as_of_date=(
            run_integrity_scope.run_as_of_date
            if run_integrity_scope is not None
            else next(
                (
                    scope.run_as_of_date
                    for scope in integrity_scopes.values()
                    if scope.run_as_of_date
                ),
                "",
            )
        ),
        commodity_block=commodity_block,
    )
    from app.discover.persistence import DiscoverCostBudgetExceeded
    from app.discover.stage4 import Stage4Config, deep_research_ticker

    triage_lookup = {s["ticker"]: s for s in triage_survivors}

    scan_context_builder = _make_scan_context_builder(
        bundle_builder=bundle_builder,
        sector=sector,
        triage_lookup=triage_lookup,
        integrity_scopes=integrity_scopes,
        db_path=db_path,
        commodity_block=commodity_block,
    )

    fixed_quotes: dict[str, dict[str, Any]] = {}
    fixed_contexts: dict[str, dict[str, Any]] = {}
    for ticker, scope in integrity_scopes.items():
        packet = scope.packets[0] if scope.packets else None
        fixed_quotes[str(ticker).upper()] = {
            "ticker": str(ticker).upper(),
            "price": getattr(packet, "current_price", None),
            "currency": getattr(packet, "current_price_currency", None),
            "as_of_date": getattr(packet, "current_price_as_of_date", None),
            "source": getattr(packet, "current_price_source", None),
            "source_url": getattr(packet, "current_price_source_url", None),
            "quote_snapshot_id": getattr(packet, "quote_snapshot_id", None),
            "status": "FIXED_RUN_QUOTE",
        }

    def immutable_tool_dispatcher(tool_name: str, tool_input: dict[str, Any]) -> str:
        ticker = str(tool_input.get("ticker") or "").strip().upper()
        if tool_name == "fetch_current_price":
            return json.dumps(
                fixed_quotes.get(
                    ticker,
                    {
                        "ticker": ticker,
                        "status": "NEEDS_DATA",
                        "reason": "ticker is not bound to the authorized run scope",
                    },
                ),
                sort_keys=True,
            )
        context = fixed_contexts.get(ticker)
        if context is None:
            return json.dumps(
                {
                    "ticker": ticker,
                    "status": "NEEDS_DATA",
                    "reason": "ticker is not bound to the authorized run scope",
                },
                sort_keys=True,
            )
        if tool_name == "fetch_companyfacts":
            return str(context.get("_fixed_financials_block") or "")
        if tool_name == "fetch_historical_scorecards":
            return (
                "NEEDS_DATA: mutable historical scorecards are disabled in this "
                "fixed-as-of run; use the canonical packet in the initial prompt"
            )
        if tool_name == "fetch_filing_section":
            section_key = str(tool_input.get("section_key") or "")
            sections = context.get("_fixed_filing_sections") or {}
            if section_key not in sections:
                return f"Section {section_key!r} not present. Available: {list(sections)}"
            filing_meta = context.get("_fixed_filing_meta") or {}
            text = str(sections[section_key])
            if len(text) > 30_000:
                text = text[:30_000] + "\n\n... [truncated]"
            return (
                f"[{filing_meta.get('form_type')} filed "
                f"{filing_meta.get('filing_date')}, accession "
                f"{filing_meta.get('accession')}]\n\n{text}"
            )
        return f"ERROR: unknown tool {tool_name!r}"

    s4_config = Stage4Config(
        model=config.deep_model,
        max_output_tokens=config.deep_max_output_tokens,
        max_turns=15,
        max_cost_usd=3.00,
    )

    results: list[SectorDeepResult] = []
    deep_prompts: dict[str, str] = {}

    for survivor in triage_survivors:
        if run_integrity_scope is not None:
            require_unchanged_financial_integrity_scope(
                run_integrity_scope,
                expected_scope_fingerprint=run_integrity_scope_fingerprint,
            )
        ticker = survivor["ticker"]
        rank = survivor.get("rank", 0)
        reasoning = survivor.get("reasoning", "")

        try:
            ctx = scan_context_builder(ticker)
            fixed_contexts[str(ticker).upper()] = ctx
            deep_prompts[ticker] = ctx["user_message"]
            current_price = ctx.get("current_price")
        except InvalidFinancialInputError:
            # Bundle/context construction is part of the financial-integrity
            # boundary. A definitive integrity failure must stop the paid run
            # instead of being contained as a per-ticker context error.
            raise
        except LLMCostBudgetExceeded:
            raise
        except Exception as exc:
            logger.warning("scan deep review %s: context build failed: %s", ticker, exc)
            results.append(
                SectorDeepResult(
                    ticker=ticker,
                    triage_rank=rank,
                    triage_reasoning=reasoning,
                    error=f"context build failed: {exc}",
                )
            )
            continue

        integrity_scope = integrity_scopes.get(str(ticker).upper())
        if integrity_scope is None:
            integrity_scope = FinancialIntegrityScope(
                context=f"sector_deep_review_missing_scope:{sector}:{ticker}",
                run_as_of_date="",
                packets=(),
            )

        initial_integrity_result = require_financial_integrity_scope(integrity_scope)
        matching_packets = [
            packet
            for packet in integrity_scope.packets
            if str(getattr(packet, "ticker", "") or "").strip().upper()
            == str(ticker).strip().upper()
        ]
        if len(matching_packets) != 1:
            raise ValueError(f"{ticker} requires exactly one canonical deep-review packet")
        financial_packet = matching_packets[0]

        class _IntegrityBoundMessages:
            def __init__(
                self,
                scope: FinancialIntegrityScope,
                expected_scope_fingerprint: str,
                usage_lane: str,
            ) -> None:
                self._scope = scope
                self._expected_scope_fingerprint = expected_scope_fingerprint
                self._usage_lane = usage_lane
                self.usage_records: list[dict[str, Any]] = []

            def create(self, **kwargs: Any) -> Any:
                require_unchanged_financial_integrity_scope(
                    self._scope,
                    expected_scope_fingerprint=self._expected_scope_fingerprint,
                )
                try:
                    response, usage_records = _call_metered_anthropic(
                        client=client,
                        request=dict(kwargs),
                        schema_name="sector_deep_review_v1",
                        lane=self._usage_lane,
                    )
                    self.usage_records.extend(usage_records)
                    return response
                except LLMCostBudgetExceeded as exc:
                    self.usage_records.extend(attached_provider_usage_records(exc))
                    raise DiscoverCostBudgetExceeded(str(exc)) from exc
                except Exception as exc:
                    self.usage_records.extend(attached_provider_usage_records(exc))
                    try:
                        require_unchanged_financial_integrity_scope(
                            self._scope,
                            expected_scope_fingerprint=self._expected_scope_fingerprint,
                        )
                    except InvalidFinancialInputError as integrity_exc:
                        raise integrity_exc from exc
                    raise

        class _IntegrityBoundClient:
            def __init__(
                self,
                scope: FinancialIntegrityScope,
                expected_scope_fingerprint: str,
                usage_lane: str,
            ) -> None:
                self.messages = _IntegrityBoundMessages(
                    scope,
                    expected_scope_fingerprint,
                    usage_lane,
                )

        bound_client = _IntegrityBoundClient(
            integrity_scope,
            initial_integrity_result.scope_fingerprint,
            f"sector_deep_review:{ticker}",
        )
        s4_result = deep_research_ticker(
            client=bound_client,
            ticker=ticker,
            config=s4_config,
            context_builder=lambda requested_ticker, _ctx=ctx, _ticker=ticker: (
                _ctx
                if str(requested_ticker).strip().upper() == str(_ticker).strip().upper()
                else (_ for _ in ()).throw(
                    ValueError("requested ticker does not match frozen context")
                )
            ),
            tool_dispatcher=immutable_tool_dispatcher,
            financial_packet=financial_packet,
        )
        deep_usage = _sector_usage_summary(bound_client.messages.usage_records)
        require_unchanged_financial_integrity_scope(
            integrity_scope,
            expected_scope_fingerprint=initial_integrity_result.scope_fingerprint,
        )
        if run_integrity_scope is not None:
            require_unchanged_financial_integrity_scope(
                run_integrity_scope,
                expected_scope_fingerprint=run_integrity_scope_fingerprint,
            )

        result = SectorDeepResult(
            ticker=ticker,
            triage_rank=rank,
            triage_reasoning=reasoning,
            verdict=s4_result.verdict,
            confidence=s4_result.confidence,
            thesis_summary=s4_result.thesis,
            key_numbers=s4_result.key_findings,
            positives=[],
            risks=[],
            open_questions=s4_result.open_questions,
            reasoning_trace=s4_result.reasoning_trace,
            current_price=current_price,
            buy_below_price=s4_result.buy_below_price,
            input_tokens=int(deep_usage["input_tokens"]),
            output_tokens=int(deep_usage["output_tokens"]),
            cost_usd=float(deep_usage["cost_usd"]),
            error=s4_result.error,
        )
        # Stash Stage 4 metadata for audit artifact
        result._s4_num_turns = s4_result.num_turns  # type: ignore[attr-defined]
        result._s4_tool_call_counts = s4_result.tool_call_counts  # type: ignore[attr-defined]
        result._s4_termination_reason = s4_result.termination_reason  # type: ignore[attr-defined]
        result._s4_tool_transcript = s4_result.tool_transcript  # type: ignore[attr-defined]
        result._provider_usage = list(deep_usage["provider_usage"])  # type: ignore[attr-defined]

        results.append(result)
        logger.info(
            "scan deep review %s: %s %s ($%.4f, %d turns, %s)",
            ticker,
            result.verdict,
            result.confidence,
            result.cost_usd,
            s4_result.num_turns,
            s4_result.termination_reason,
        )

    return results, deep_prompts


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fetch_5y_snapshot(
    ticker: str,
    *,
    as_of_date: str,
    db_path: str | Path | None = None,
) -> list[dict[str, float | None]] | None:
    """Pull a 5-year snapshot table for the per-ticker tearsheet.

    Returns rows newest first with the canonical fields a Value-Line-style
    snapshot needs. Returns None when the database is empty (test runs).
    """
    if db_path is None:
        from app.config import get_config

        db_path = get_config().db_path
    line_items = (
        "revenue",
        "operating_income",
        "cfo",
        "capex",
        "cash",
        "total_debt",
        "shares_outstanding",
        "sbc",
    )
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        required = {
            "ticker",
            "fiscal_year",
            "period_type",
            "period_end",
            "filed_date",
            "line_item",
            "value",
        }
        available = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
        }
        if not required <= available:
            conn.close()
            return None
        rows = companyfacts_rows(
            conn,
            ticker,
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
            order_by="fiscal_year DESC",
        )
        conn.close()
    except sqlite3.OperationalError:
        return None
    if not rows:
        return None
    by_yr: dict[int, dict[str, float]] = {}
    for r in rows:
        if str(r["period_end"] or "") > str(r["filed_date"] or ""):
            continue
        by_yr.setdefault(r["fiscal_year"], {})[r["line_item"]] = r["value"]
    last5 = sorted(by_yr.keys(), reverse=True)[:5]
    out = []
    for yr in last5:
        d = by_yr[yr]
        rev = d.get("revenue")
        oi = d.get("operating_income")
        cfo = d.get("cfo")
        capex = d.get("capex")
        fcf = (cfo - abs(capex)) if (cfo is not None and capex is not None) else None
        op_margin = (oi / rev) if (rev and oi is not None and rev > 0) else None
        out.append(
            {
                "fy": yr,
                "revenue": rev,
                "op_margin": op_margin,
                "cfo": cfo,
                "fcf": fcf,
                "cash": d.get("cash"),
                "debt": d.get("total_debt"),
                "shares": d.get("shares_outstanding"),
                "sbc": d.get("sbc"),
            }
        )
    return out


def _render_valuation_box(
    *,
    price: float | None,
    pzd: dict[str, Any],
    buy_below: float | None,
) -> list[str]:
    """Markdown table of all valuation anchors with % vs price."""
    lines = ["**Valuation snapshot**", ""]
    lines.append("| Anchor | Value | vs price |")
    lines.append("|---|---|---|")

    def _row(label: str, val: Any, *, dollars: bool = True) -> str | None:
        if not isinstance(val, (int, float)):
            return None
        delta = ""
        if isinstance(price, (int, float)) and price > 0 and val != 0:
            pct = (val - price) / abs(val) if dollars else None
            if pct is not None:
                if val > price:
                    delta = f"+{pct:.0%} upside"
                else:
                    delta = f"{pct:.0%} (above value)"
        v_str = f"${val:.2f}" if dollars else f"{val:,.0f}"
        return f"| {label} | {v_str} | {delta or '—'} |"

    if isinstance(price, (int, float)):
        lines.append(f"| Current price | ${price:.2f} | — |")
    if isinstance(pzd.get("market_cap"), (int, float)):
        lines.append(f"| Market cap | ${pzd['market_cap']:,.0f}M | — |")
    for label, key in [
        ("DCF base", "dcf_base"),
        ("DCF (durable rev base)", "dcf_durable_base"),
        ("EPV adjusted (GAAP)", "epv_adjusted"),
        ("EPV cash-adjusted", "epv_cash_adjusted"),
        ("Graham value", "graham_value"),
    ]:
        row = _row(label, pzd.get(key))
        if row:
            lines.append(row)
    if isinstance(buy_below, (int, float)):
        lines.append(f"| **Buy below** | ${buy_below:.2f} | — |")
    return lines


def _render_5y_snapshot(snapshot: list[dict[str, Any]] | None) -> list[str]:
    """Markdown table of 5-year financial trajectory."""
    if not snapshot:
        return []
    lines = ["**5-year snapshot** (EDGAR companyfacts, USD millions)", ""]
    lines.append("| FY | Revenue | Op margin | CFO | FCF | Cash | Debt | Shares | SBC |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    def _fmt(v: Any, *, pct: bool = False, places: int = 0) -> str:
        if not isinstance(v, (int, float)):
            return "—"
        if pct:
            return f"{v:.1%}"
        return f"{v:,.{places}f}"

    for row in snapshot:
        lines.append(
            f"| {row['fy']} | {_fmt(row['revenue'])} | {_fmt(row['op_margin'], pct=True)} "
            f"| {_fmt(row['cfo'])} | {_fmt(row['fcf'])} | {_fmt(row['cash'])} "
            f"| {_fmt(row['debt'])} | {_fmt(row['shares'])} | {_fmt(row['sbc'])} |"
        )
    return lines


# Map machine-readable headwinds → a one-line human explanation. The reviewer
# noted the report read as a story instead of a tearsheet; the flags box
# brings the structured signals up to first-class display.
_HEADWIND_EXPLANATIONS: dict[str, str] = {
    "EPV_INTANGIBLE_AMORT_DISTORTION": (
        "GAAP EPV suppressed by purchase-price amortization — see cash-EPV"
    ),
    "NONRECURRING_REVENUE_SPIKE": (
        "Latest-year revenue includes a likely one-time event — see DCF (durable rev base)"
    ),
    "DCF_INFLATED_BY_NONRECURRING_REVENUE": (
        "Headline DCF anchor materially inflated by the spike (drop ≥30%)"
    ),
    "NONRECURRING_ITEMS_HEADWIND": (
        "Restructuring/impairment items in P&L — earnings quality lower than headline"
    ),
    "SBC_ACCELERATING_HEADWIND": "SBC growing faster than revenue",
    "SBC_BURDEN_EXTREME_HEADWIND": "SBC > 5% of revenue or > 25% of CFO",
    "NET_DILUTION_HEADWIND": "Share count rising despite buybacks",
    "WORKING_CAPITAL_DRAG_HEADWIND": "WC build absorbing CFO",
    "CAPEX_BELOW_DEPRECIATION_HEADWIND": "Capex below D&A — under-investing or harvesting",
    "DEPRECIATION_RATE_DECLINING_HEADWIND": "Slowing depreciation suggests aging asset base",
    "PEER_LAGGARD_HEADWIND": "Bottom-quartile vs sector on growth/margin/return",
}


def _render_flags_box(qc: dict[str, Any]) -> list[str]:
    """Markdown table of quality_context.valuation_headwinds with explanations."""
    headwinds = qc.get("valuation_headwinds") or []
    if not headwinds:
        return ["**Flags box:** none"]
    lines = ["**Flags box**", ""]
    lines.append("| Flag | Meaning |")
    lines.append("|---|---|")
    for h in headwinds:
        explanation = _HEADWIND_EXPLANATIONS.get(h, "(see deep-review thesis for context)")
        lines.append(f"| `{h}` | {explanation} |")
    return lines


def _render_ticker_tearsheet(
    *,
    rank: int,
    r: SectorDeepResult,
    scorecard: dict[str, Any] | None,
    as_of_date: str,
    db_path: str | Path | None,
) -> list[str]:
    """Render one ticker's section in the Value-Line-inspired tearsheet format.

    Ordering (per reviewer's recommendation):
      1. Header + price (already first-class)
      2. Valuation box (table)
      3. 5-year snapshot (table)
      4. Flags box (table)
      5. Thesis (compact prose)
      6. Key findings (bullet list — was previously comma-joined)
      7. Why it ranks here (triage rationale)
      8. Risks + open questions (bullet lists)
    """
    lines: list[str] = []
    lines.append(f"### #{rank}: {r.ticker} — {r.verdict} ({r.confidence})")
    lines.append("")
    if r.current_price is not None:
        lines.append(f"Price: ${r.current_price:.2f}")
        lines.append("")

    # Valuation + snapshot + flags only when scorecard data is available.
    if scorecard:
        pzd = scorecard.get("pricing_zone_detail") or {}
        qc = scorecard.get("quality_context") or {}
        lines.extend(
            _render_valuation_box(
                price=r.current_price,
                pzd=pzd,
                buy_below=r.buy_below_price,
            )
        )
        lines.append("")
        snap = _fetch_5y_snapshot(
            r.ticker,
            as_of_date=as_of_date,
            db_path=db_path,
        )
        if snap:
            lines.extend(_render_5y_snapshot(snap))
            lines.append("")
        lines.extend(_render_flags_box(qc))
        lines.append("")

    lines.append(f"**Thesis:** {r.thesis_summary}")
    lines.append("")

    # Key numbers as bullets, NOT one comma-joined paragraph.
    if r.key_numbers:
        lines.append("**Key findings:**")
        for kn in r.key_numbers:
            lines.append(f"- {kn}")
        lines.append("")

    if r.triage_reasoning:
        lines.append(f"**Why it ranks here:** {r.triage_reasoning}")
        lines.append("")
    if r.positives:
        lines.append("**Positives:**")
        for p in r.positives:
            lines.append(f"- {p}")
        lines.append("")
    if r.risks:
        lines.append("**Risks:**")
        for risk in r.risks:
            lines.append(f"- {risk}")
        lines.append("")
    if r.open_questions:
        lines.append("**Open questions:**")
        for q in r.open_questions:
            lines.append(f"- {q}")
        lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def render_scan_report(
    *,
    sector: str,
    sector_size: int,
    pre_ranked: int,
    triage_result: SectorTriageResult,
    deep_results: list[SectorDeepResult],
    top_n: int,
    total_cost: float,
    commodity_snapshots: list[CommoditySnapshot] | None = None,
    cap_filter_label: str | None = None,
    cap_min: float | None = None,
    cap_max: float | None = None,
    scorecards: dict[str, dict[str, Any]] | None = None,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Produce the final markdown scan report."""
    lines: list[str] = []

    # Title surfaces the cap filter when applied so readers immediately see
    # the scope. Example: "# Sector Scan: healthcare_pharma — small/mid cap"
    title_suffix = f" — {cap_filter_label}" if cap_filter_label else ""
    lines.append(f"# Sector Scan: {sector}{title_suffix}")
    lines.append("")

    if cap_filter_label:
        # Detail the numeric range below the title so readers know the exact
        # cutoffs the tier label maps to.
        if cap_min is not None and cap_max is not None:
            lines.append(
                f"**Cap filter:** {cap_filter_label} (${cap_min:,.0f}M – ${cap_max:,.0f}M)"
            )
        elif cap_min is not None:
            lines.append(f"**Cap filter:** {cap_filter_label} (≥ ${cap_min:,.0f}M)")
        elif cap_max is not None:
            lines.append(f"**Cap filter:** {cap_filter_label} (≤ ${cap_max:,.0f}M)")
        else:
            lines.append(f"**Cap filter:** {cap_filter_label}")

    lines.append(f"**Tickers in sector:** {sector_size}")
    lines.append("**Scan family:** normal")
    lines.append(f"**Pre-ranked:** {pre_ranked}")
    lines.append(f"**AI shortlisted:** {len(triage_result.survivors)}")
    lines.append(f"**Deep-reviewed:** {len(deep_results)}")
    lines.append(f"**Total cost:** ${total_cost:.4f}")
    lines.append("")

    if commodity_snapshots:
        lines.append("## Commodity Context")
        lines.append("")
        lines.append("| Commodity | Current | 90d avg | 180d range | Position | Read |")
        lines.append("|---|---|---|---|---|---|")
        for s in commodity_snapshots:
            rng_180 = f"{s.range_180d[0]:.2f} – {s.range_180d[1]:.2f}" if s.range_180d else "—"
            pos = (
                f"{s.position_in_180d_range * 100:.0f}%"
                if s.position_in_180d_range is not None
                else "—"
            )
            avg_90 = f"{s.avg_90d:.2f}" if s.avg_90d is not None else "—"
            lines.append(
                f"| {s.display_name} ({s.symbol}) | {s.current_price:.2f} {s.unit} "
                f"| {avg_90} | {rng_180} | {pos} | **{s.interpretation.upper()}** |"
            )
        lines.append("")
        lines.append(
            "*The AI analyst used this commodity context when evaluating each company. "
            "Read theses in light of where prices are now, not at historical averages.*"
        )
        lines.append("")

    # Sort: BUY/BUY_CANDIDATE first, then WATCH, then PASS, by triage rank within
    verdict_order = {"BUY": 0, "BUY_CANDIDATE": 0, "WATCH": 1, "PASS": 2, "": 3}
    successful = [r for r in deep_results if r.error is None]
    successful.sort(key=lambda r: (verdict_order.get(r.verdict, 3), r.triage_rank))

    top = successful[:top_n]
    lines.append(f"## Top {len(top)} Picks")
    lines.append("")

    for i, r in enumerate(top, 1):
        ticker_scorecard = (scorecards or {}).get(r.ticker.upper())
        lines.extend(
            _render_ticker_tearsheet(
                rank=i,
                r=r,
                scorecard=ticker_scorecard,
                as_of_date=str(as_of_date or ""),
                db_path=db_path,
            )
        )

    if triage_result.surprises:
        lines.append("## Triage Surprises")
        lines.append("")
        for s in triage_result.surprises:
            lines.append(f"- {s}")
        lines.append("")

    errors = [r for r in deep_results if r.error is not None]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(f"- **{r.ticker}**: {r.error}")
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- Stage 1: Consensus pre-rank (deterministic, top {pre_ranked})")
    lines.append(f"- Stage 2: AI comparative triage ({len(triage_result.survivors)} survivors)")
    lines.append(f"- Stage 3: Deep financial review ({len(deep_results)} reviews)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _save_scan_artifacts(
    output_dir: Path,
    sector: str,
    as_of_date: str,
    triage_prompt: str,
    triage_result: SectorTriageResult,
    deep_prompts: dict[str, str],
    deep_results: list[SectorDeepResult],
    financial_packets: dict[str, Any],
    financial_integrity_result: FinancialIntegrityGateResult,
    report_path: Path,
    commodity_snapshots: list[CommoditySnapshot] | None = None,
    cap_filter_label: str | None = None,
    cap_min: float | None = None,
    cap_max: float | None = None,
    paid_usage: dict[str, Any] | None = None,
) -> Path:
    """Save full prompts and responses as a JSON artifact for audit.

    Returns the artifact path. This lets the user verify exactly what
    data the AI saw and whether its judgments were grounded.
    """
    _reject_unbound_commodity_context(
        context=f"sector_scan_publication:{sector}",
        run_as_of_date=as_of_date,
        commodity_snapshots=commodity_snapshots,
    )
    if (
        not financial_integrity_result.passed
        or financial_integrity_result.run_as_of_date != as_of_date
    ):
        raise RuntimeError(
            "Refusing to publish a scan whose attestation is not PASS for the exact as-of date."
        )
    quote_snapshots: list[dict[str, Any]] = []
    financial_context: list[dict[str, Any]] = []
    for ticker, packet in sorted(financial_packets.items()):
        normalized_ticker = str(ticker).strip().upper()
        price_basis = str(getattr(packet, "price_basis", "") or "").strip().upper()
        current_price = getattr(packet, "current_price", None)
        raw_price = getattr(packet, "raw_price", None)
        if raw_price is None and price_basis == "UNADJUSTED":
            raw_price = current_price
        snapshot = {
            "ticker": normalized_ticker,
            "price": current_price,
            "currency": getattr(packet, "current_price_currency", None),
            "as_of_date": getattr(packet, "current_price_as_of_date", None),
            "source": getattr(packet, "current_price_source", None),
            "source_url": getattr(packet, "current_price_source_url", None),
            "price_unit": getattr(packet, "current_price_unit", None),
            "price_basis": price_basis,
            "raw_price": raw_price,
            "split_adjustment_factor": getattr(packet, "split_adjustment_factor", None),
            "split_effective_date": getattr(packet, "split_effective_date", None),
            "quote_snapshot_id": getattr(packet, "quote_snapshot_id", None),
        }
        quote_snapshots.append(snapshot)
        financial_context.append(
            {
                "ticker": normalized_ticker,
                "current_price": current_price,
                "quote_snapshot_id": getattr(packet, "quote_snapshot_id", None),
            }
        )
    snapshot_ids = {
        str(ticker).strip().upper(): str(snapshot_id).strip()
        for ticker, snapshot_id in financial_integrity_result.ticker_snapshot_ids.items()
    }
    derived_snapshot_ids = {
        str(item["ticker"]): str(item["quote_snapshot_id"] or "").strip()
        for item in quote_snapshots
    }
    if not derived_snapshot_ids or derived_snapshot_ids != snapshot_ids:
        raise RuntimeError(
            "Refusing to publish a scan whose exact quote snapshots differ from its attestation."
        )

    integrity = financial_integrity_result.to_dict()
    integrity["quote_snapshots"] = quote_snapshots
    artifact = {
        "scan_family": "normal",
        "sector": sector,
        "as_of_date": as_of_date,
        "artifact_paths": {"markdown": str(report_path.resolve())},
        "financial_context": {
            "scope_fingerprint": financial_integrity_result.scope_fingerprint,
            "tickers": financial_context,
        },
        "financial_integrity": integrity,
        "cap_filter": (
            {"label": cap_filter_label, "cap_min": cap_min, "cap_max": cap_max}
            if cap_filter_label or cap_min is not None or cap_max is not None
            else None
        ),
        "paid_usage": dict(paid_usage or {}),
        "triage": {
            "prompt_chars": len(triage_prompt),
            "prompt_preview": triage_prompt[:2000],
            "full_prompt": triage_prompt,
            "survivors_count": len(triage_result.survivors),
            "survivors": triage_result.survivors,
            "surprises": triage_result.surprises,
            "cost_usd": triage_result.cost_usd,
        },
        "deep_reviews": [],
    }
    for r in deep_results:
        entry: dict[str, Any] = {
            "ticker": r.ticker,
            "triage_rank": r.triage_rank,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "thesis_summary": r.thesis_summary,
            "buy_below_price": r.buy_below_price,
            "cost_usd": r.cost_usd,
            "error": r.error,
        }
        prompt_text = deep_prompts.get(r.ticker, "")
        entry["prompt_chars"] = len(prompt_text)
        entry["prompt_preview"] = prompt_text[:2000]
        entry["full_prompt"] = prompt_text
        # Flag whether the prompt contained actual filing text
        entry["had_filing_text"] = "=== MOST RECENT FILING" in prompt_text
        entry["had_no_filing"] = "NO FILING TEXT AVAILABLE" in prompt_text
        # Stage 4 agentic metadata (if available)
        entry["num_turns"] = getattr(r, "_s4_num_turns", None)
        entry["tool_call_counts"] = getattr(r, "_s4_tool_call_counts", None)
        entry["termination_reason"] = getattr(r, "_s4_termination_reason", None)
        entry["tool_transcript"] = getattr(r, "_s4_tool_transcript", None)
        artifact["deep_reviews"].append(entry)

    from app.autonomous.artifact_financial_audit import audit_payload

    violations = audit_payload(artifact)
    if violations:
        invariants = sorted(
            {
                str(item.get("invariant") or "").strip()
                for item in violations
                if str(item.get("invariant") or "").strip()
            }
        )
        raise RuntimeError(
            "Refusing to publish a financially unauthorizable scan artifact: "
            + ", ".join(invariants or ["UNKNOWN_FINANCIAL_INTEGRITY_VIOLATION"])
        )

    artifact_path = output_dir / f"{sector}_{as_of_date}_artifacts.json"
    artifact_bytes = json.dumps(artifact, indent=2, allow_nan=False).encode("utf-8")
    temporary = artifact_path.with_name(f".{artifact_path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(artifact_bytes)
    temporary.replace(artifact_path)
    return artifact_path


def _run_paid_sector_stages(
    *,
    sector: str,
    budget_usd: float,
    client: Any,
    ranked_tickers: list[str],
    scorecards: dict[str, dict[str, Any]],
    config: ScanConfig,
    integrity_scope: FinancialIntegrityScope,
    integrity_scope_fingerprint: str,
    financial_packets: dict[str, Any],
    bundle_builder: Callable[..., Any],
    db_path: str | Path | None,
) -> tuple[
    SectorTriageResult,
    list[SectorDeepResult],
    dict[str, str],
    dict[str, Any],
]:
    """Execute every paid sector path inside one exact run ceiling."""

    with (
        provider_usage_budget(budget_usd) as paid_budget,
        provider_usage_capture(f"sector_scan:{sector}") as provider_usage,
    ):
        triage_result = run_sector_triage(
            client=client,
            sector=sector,
            ranked_tickers=ranked_tickers,
            scorecards=scorecards,
            config=config,
            integrity_scope=integrity_scope,
        )
        require_unchanged_financial_integrity_scope(
            integrity_scope,
            expected_scope_fingerprint=integrity_scope_fingerprint,
        )
        deep_integrity_scopes = {
            ticker: FinancialIntegrityScope(
                context=f"sector_deep_review:{sector}:{ticker}",
                run_as_of_date=integrity_scope.run_as_of_date,
                packets=(packet,),
            )
            for ticker, packet in financial_packets.items()
        }
        deep_results, deep_prompts = run_sector_deep_review(
            client=client,
            sector=sector,
            triage_survivors=triage_result.survivors,
            bundle_builder=bundle_builder,
            config=config,
            integrity_scopes=deep_integrity_scopes,
            run_integrity_scope=integrity_scope,
            run_integrity_scope_fingerprint=integrity_scope_fingerprint,
            db_path=db_path,
        )
        usage_summary = _sector_usage_summary(provider_usage)
        usage_summary["budget_usd"] = float(budget_usd)
        usage_summary["budget_remaining_usd"] = round(paid_budget.remaining(), 6)
    return triage_result, deep_results, deep_prompts, usage_summary


def run_scan(
    *,
    sector: str,
    top_n: int = 10,
    budget_usd: float = 25.0,
    output_dir: Path | None = None,
    dry_run: bool = False,
    client=None,
    bundle_builder: Callable[..., Any] | None = None,
    config: ScanConfig | None = None,
    db_path: str | Path | None = None,
    cap_min: float | None = None,
    cap_max: float | None = None,
    cap_filter_label: str | None = None,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    """Orchestrate the full three-stage sector scan. Returns summary dict.

    cap_min / cap_max filter the sector universe by market cap (millions USD).
    cap_filter_label is a human-readable string for the report header
    (e.g. "small/mid cap" or "$300M-$10,000M"). If unset but cap_min/cap_max
    are set, a label is synthesized automatically.

    A preflight health gate runs first to catch broken structured-data layers
    BEFORE spending money on triage and deep review. The reviewer caught a
    case where one tool was failing 19/19 times silently. Set skip_preflight
    to bypass the gate (e.g., during test scans).
    """
    if config is None:
        config = ScanConfig()
    if db_path is None:
        from app.config import get_config

        db_path = get_config().db_path
    run_as_of_date = date.today().isoformat()

    # Stage 1: Load + pre-rank (honors cap filter)
    tickers_data = load_sector_tickers(
        sector=sector,
        db_path=db_path,
        cap_min=cap_min,
        cap_max=cap_max,
        as_of_date=run_as_of_date,
        allow_live_market_data=False,
    )
    if not tickers_data:
        raise ValueError(f"No tickers classified as '{sector}'")

    # Preflight: sample the universe and verify the structured-data layer is
    # actually returning useful answers. If <90% of sampled tickers have the
    # core companyfacts fields populated, abort the scan before spending money.
    if not skip_preflight and not dry_run:
        sample_tickers = [t for t, _, _ in tickers_data]
        preflight = run_preflight(
            sector=sector,
            tickers=sample_tickers,
            db_path=db_path,
        )
        logger.info("scan %s preflight: %s", sector, preflight.summary)
        if not preflight.passed:
            raise RuntimeError(
                "Preflight health gate FAILED — refusing to run scan.\n\n"
                + format_preflight_report(preflight)
                + "\n\nTo bypass (not recommended), pass skip_preflight=True."
            )

    # Synthesize a filter label for the report if caller didn't supply one
    if cap_filter_label is None and (cap_min is not None or cap_max is not None):
        lo = f"${cap_min:,.0f}M" if cap_min is not None else "any"
        hi = f"${cap_max:,.0f}M" if cap_max is not None else "any"
        cap_filter_label = f"{lo}-{hi}"

    ticker_list = [t for t, _, _ in tickers_data]
    financial_context = build_canonical_v1_financial_context(
        tickers=ticker_list,
        as_of_date=run_as_of_date,
        db_path=db_path,
        scorecard_evidence={
            ticker: (row_as_of, scorecard) for ticker, row_as_of, scorecard in tickers_data
        },
    )
    consensus = rank_by_consensus(financial_context.packets)
    ranked = consensus.ranked + consensus.ranked_insufficient
    ranked.sort(key=lambda entry: entry.consensus_score, reverse=True)
    if config.pre_rank_limit > 0:
        ranked = ranked[: config.pre_rank_limit]
    integrity_scope = financial_context.scope(context=f"sector_scan:{sector}:{run_as_of_date}")
    scorecards = {t: sc for t, _, sc in tickers_data}

    if dry_run:
        return {
            "scan_family": "normal",
            "sector": sector,
            "sector_size": len(tickers_data),
            "pre_ranked": len(ranked),
            "ranked_tickers": [(e.ticker, e.consensus_score) for e in ranked],
            "dry_run": True,
        }

    if client is None:
        raise ValueError("client is required when dry_run=False")

    ranked_tickers = [e.ticker for e in ranked]
    integrity_result = require_financial_integrity_scope(integrity_scope)

    # Stage 2/3 dependencies. The bundle builder is invoked inside the shared
    # paid context below, so nested filing-risk classification is charged to
    # the same hard run ceiling as triage and deep-review calls.
    if bundle_builder is None:
        from app.analyst.bundle_builder import (
            build_analysis_evidence_bundle_from_cached_scorecard,
        )

        scorecard_lookup = {t.upper(): (d, sc) for t, d, sc in tickers_data}

        def _fast_bundle_builder(ticker: str, as_of_date: str | None = None):
            entry = scorecard_lookup.get(ticker.upper())
            if entry is None:
                from app.analyst.bundle_builder import build_analysis_evidence_bundle

                return build_analysis_evidence_bundle(
                    ticker,
                    as_of_date=as_of_date,
                    financial_packet=financial_context.packets.get(ticker.upper()),
                )
            sc_as_of, sc = entry
            return build_analysis_evidence_bundle_from_cached_scorecard(
                ticker=ticker,
                scorecard=sc,
                scorecard_as_of_date=sc_as_of,
                as_of_date=as_of_date,
                financial_packet=financial_context.packets.get(ticker.upper()),
            )

        bundle_builder = _fast_bundle_builder

    triage_result, deep_results, deep_prompts, paid_usage = _run_paid_sector_stages(
        budget_usd=budget_usd,
        client=client,
        sector=sector,
        ranked_tickers=ranked_tickers,
        scorecards=scorecards,
        config=config,
        integrity_scope=integrity_scope,
        integrity_scope_fingerprint=integrity_result.scope_fingerprint,
        financial_packets=financial_context.packets,
        bundle_builder=bundle_builder,
        db_path=db_path,
    )
    total_cost = float(paid_usage["cost_usd"])

    # Revalidate the exact mutable packets immediately before product output.
    integrity_result = require_unchanged_financial_integrity_scope(
        integrity_scope,
        expected_scope_fingerprint=integrity_result.scope_fingerprint,
    )

    # Report
    md = render_scan_report(
        sector=sector,
        sector_size=len(tickers_data),
        pre_ranked=len(ranked),
        triage_result=triage_result,
        deep_results=deep_results,
        top_n=top_n,
        total_cost=total_cost,
        cap_filter_label=cap_filter_label,
        cap_min=cap_min,
        cap_max=cap_max,
        scorecards={t.upper(): sc for t, _, sc in tickers_data},
        as_of_date=run_as_of_date,
        db_path=db_path,
    )

    out = output_dir or Path("data/outputs/scans")
    report_path = out / f"{sector}_{run_as_of_date}.md"
    triage_prompt = getattr(triage_result, "_prompt", "")
    integrity_result = require_unchanged_financial_integrity_scope(
        integrity_scope,
        expected_scope_fingerprint=integrity_result.scope_fingerprint,
    )
    out.mkdir(parents=True, exist_ok=True)

    # Save the exact gated financial contract beside the full paid prompts.
    artifact_path = _save_scan_artifacts(
        output_dir=out,
        sector=sector,
        as_of_date=run_as_of_date,
        triage_prompt=triage_prompt,
        triage_result=triage_result,
        deep_prompts=deep_prompts,
        deep_results=deep_results,
        financial_packets=financial_context.packets,
        financial_integrity_result=integrity_result,
        report_path=report_path,
        cap_filter_label=cap_filter_label,
        cap_min=cap_min,
        cap_max=cap_max,
        paid_usage=paid_usage,
    )
    integrity_result = require_unchanged_financial_integrity_scope(
        integrity_scope,
        expected_scope_fingerprint=integrity_result.scope_fingerprint,
    )
    report_temporary = report_path.with_name(f".{report_path.name}.{uuid4().hex}.tmp")
    report_temporary.write_text(md, encoding="utf-8")
    report_temporary.replace(report_path)
    logger.info("scan %s: audit artifact saved to %s", sector, artifact_path)

    return {
        "scan_family": "normal",
        "sector": sector,
        "sector_size": len(tickers_data),
        "pre_ranked": len(ranked),
        "triage_survivors": len(triage_result.survivors),
        "deep_reviewed": len(deep_results),
        "total_cost": total_cost,
        "budget_usd": float(budget_usd),
        "budget_remaining_usd": paid_usage["budget_remaining_usd"],
        "physical_provider_calls": paid_usage["physical_calls"],
        "provider_usage": paid_usage["provider_usage"],
        "report_path": str(report_path),
        "artifact_path": str(artifact_path),
        "financial_integrity_status": integrity_result.status,
        "financial_integrity_scope_fingerprint": integrity_result.scope_fingerprint,
    }
