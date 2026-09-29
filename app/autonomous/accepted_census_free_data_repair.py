"""Bounded, accepted-census-only free-source repair for v2 readiness.

This maintenance lane is intentionally narrower than the generic cache refresh:
it accepts only an already frozen accepted-census execution ledger, uses the
accepted issuer CIK for SEC work, permits only SEC and unauthenticated Stooq
network sources, and reruns the query-only readiness evaluator after repair.
It never assembles a packet or enters sector research/runtime code.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

from app.autonomous.evidence_resolution import pre_assembly_data_gap_repair
from app.autonomous.readiness_preflight import run_v2_readiness_preflight
from app.autonomous.sector_candidates import AcceptedCensusRunAuthority
from app.config import AppConfig, canonical_market_cap_focus, get_config
from app.market.price_provider import (
    ChainedPriceProvider,
    StooqProvider,
    StooqSecondaryProvider,
)
from app.util.http import HttpClient


FREE_DATA_REPAIR_SCHEMA_VERSION = "AUTONOMOUS_V2_ACCEPTED_CENSUS_FREE_DATA_REPAIR_V1"
FREE_DATA_REPAIR_MODE = "ACCEPTED_CENSUS_FREE_SOURCES_ONLY"
FREE_DATA_REPAIR_MAX_SECTORS = 4
FREE_DATA_REPAIR_MAX_CANDIDATES_PER_SECTOR = 3
FREE_DATA_REPAIR_MAX_EXECUTION_NAMES = 12
FREE_DATA_REPAIR_ALLOWED_DOMAINS = frozenset(
    {"data.sec.gov", "www.sec.gov", "sec.gov", "stooq.com", "www.stooq.com"}
)
ANNUAL_ONLY_FILING_WINDOWS_DAYS = {
    "10-K": 800,
    "10-K/A": 800,
    "20-F": 800,
    "20-F/A": 800,
    "40-F": 800,
    "40-F/A": 800,
}


def build_free_stooq_price_provider(cfg: AppConfig | None = None) -> Any:
    """Build a price chain that cannot contain keyed or paid providers."""

    resolved = cfg or get_config()
    free_cfg = resolved.model_copy(
        update={
            "price_provider": "stooq",
            "eodhd_apikey": None,
            "stooq_apikey": None,
        }
    )
    provider_kwargs = {
        "cfg": free_cfg,
        "fallback_days": int(free_cfg.price_fallback_days),
        "max_retries": 1,
    }
    return ChainedPriceProvider(
        [
            StooqProvider(**provider_kwargs),
            StooqSecondaryProvider(**provider_kwargs),
        ]
    )


def _http_metrics(cfg: AppConfig) -> dict[str, int]:
    return HttpClient(cfg).metrics()


def _network_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> tuple[dict[str, int], int, list[str]]:
    by_domain: dict[str, int] = {}
    for key, value in after.items():
        if not str(key).startswith("domain_count:"):
            continue
        domain = str(key).split(":", 1)[1].lower()
        delta = max(0, int(value) - int(before.get(key, 0)))
        if delta:
            by_domain[domain] = delta
    unexpected = sorted(
        domain for domain in by_domain if domain not in FREE_DATA_REPAIR_ALLOWED_DOMAINS
    )
    return dict(sorted(by_domain.items())), sum(by_domain.values()), unexpected


def _readiness_rows(readiness: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    sectors = readiness.get("sector_results")
    if not isinstance(sectors, Mapping):
        return rows
    for sector_result in sectors.values():
        if not isinstance(sector_result, Mapping):
            continue
        for row in sector_result.get("candidate_rows") or []:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            if ticker:
                rows[ticker] = row
    return rows


def _readiness_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    before_rows = _readiness_rows(before)
    after_rows = _readiness_rows(after)
    tickers = list(before_rows)
    return [
        {
            "ticker": ticker,
            "before": before_rows[ticker].get("readiness"),
            "after": (after_rows.get(ticker) or {}).get("readiness"),
            "missing_inputs_before": list(
                before_rows[ticker].get("missing_inputs") or []
            ),
            "missing_inputs_after": list(
                (after_rows.get(ticker) or {}).get("missing_inputs") or []
            ),
        }
        for ticker in tickers
    ]


def _zero_paid_usage(*, free_network_calls: int) -> dict[str, Any]:
    return {
        "model_calls": 0,
        "search_calls": 0,
        "paid_provider_calls": 0,
        "network_calls": int(free_network_calls),
        "free_network_calls": int(free_network_calls),
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }


def _validate_scope_limits(
    *,
    sectors: list[str],
    candidate_payloads: Mapping[str, Mapping[str, Any]],
    market_cap_focus: str,
) -> int:
    if canonical_market_cap_focus(market_cap_focus) != "large_and_mega":
        raise ValueError("free-data-repair-only requires large_and_mega")
    if not sectors or len(sectors) > FREE_DATA_REPAIR_MAX_SECTORS:
        raise ValueError(
            "free-data-repair-only requires between 1 and "
            f"{FREE_DATA_REPAIR_MAX_SECTORS} sectors"
        )
    total = 0
    for sector in sectors:
        payload = candidate_payloads.get(sector)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{sector}: frozen candidate payload is missing")
        bound = payload.get("execution_bound")
        if (
            isinstance(bound, bool)
            or not isinstance(bound, int)
            or bound < 1
            or bound > FREE_DATA_REPAIR_MAX_CANDIDATES_PER_SECTOR
        ):
            raise ValueError(
                "free-data-repair-only requires an explicit per-sector candidate "
                f"bound between 1 and {FREE_DATA_REPAIR_MAX_CANDIDATES_PER_SECTOR}"
            )
        execution = payload.get("execution_tickers")
        if not isinstance(execution, list) or not execution:
            raise ValueError(
                f"{sector}: free-data repair requires at least one frozen execution name"
            )
        if len(execution) > bound:
            raise ValueError(f"{sector}: frozen execution ledger exceeds its bound")
        total += len(execution)
    if total > FREE_DATA_REPAIR_MAX_EXECUTION_NAMES:
        raise ValueError(
            "free-data-repair-only execution set exceeds the hard "
            f"{FREE_DATA_REPAIR_MAX_EXECUTION_NAMES}-name ceiling"
        )
    return total


def run_accepted_census_free_data_repair(
    *,
    benchmark_run_id: str,
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
    """Repair the exact frozen accepted-census execution set with free sources."""

    resolved_cfg = cfg or get_config()
    execution_count = _validate_scope_limits(
        sectors=sectors,
        candidate_payloads=candidate_payloads,
        market_cap_focus=market_cap_focus,
    )
    authority = accepted_census_authority or AcceptedCensusRunAuthority()
    before = run_v2_readiness_preflight(
        sectors=sectors,
        candidate_payloads=candidate_payloads,
        as_of_date=as_of_date,
        market_cap_focus=market_cap_focus,
        execution_set_fingerprint=execution_set_fingerprint,
        request_fingerprint=request_fingerprint,
        candidate_resolution_errors=candidate_resolution_errors,
        accepted_census_authority=authority,
        cfg=resolved_cfg,
    )
    authority_ok = (
        before.get("status") == "COMPLETED"
        and before.get("authority_validation_status") == "PASSED"
        and int((before.get("counts") or {}).get("execution_candidates") or 0)
        == execution_count
    )
    if not authority_ok:
        validation_errors = list(before.get("validation_errors") or [])
        if (
            before.get("status") == "COMPLETED"
            and before.get("authority_validation_status") == "PASSED"
            and int((before.get("counts") or {}).get("execution_candidates") or 0)
            != execution_count
        ):
            validation_errors.append(
                "READINESS_EXECUTION_COUNT_DOES_NOT_MATCH_FROZEN_REPAIR_SCOPE"
            )
        return {
            "schema_version": FREE_DATA_REPAIR_SCHEMA_VERSION,
            "status": "INCOMPLETE",
            "mode": FREE_DATA_REPAIR_MODE,
            "maintenance_only": True,
            "as_of_date": as_of_date,
            "sectors": list(sectors),
            "request_fingerprint": request_fingerprint,
            "execution_set_fingerprint": execution_set_fingerprint,
            "authority_validation_status": "FAILED",
            "validation_errors": validation_errors,
            "before_readiness": before,
            "after_readiness": before,
            "sector_repairs": {},
            "actual_usage": _zero_paid_usage(free_network_calls=0),
            "network": {
                "allowed_domains": sorted(FREE_DATA_REPAIR_ALLOWED_DOMAINS),
                "calls_by_domain": {},
                "unexpected_domains": [],
            },
            "packet_materialized_count": 0,
            "production_sector_scan_exercised": False,
            "watchlist_mutation_exercised": False,
        }

    effective_price_provider = build_free_stooq_price_provider(resolved_cfg)
    metrics_before = _http_metrics(resolved_cfg)
    run_root = (
        resolved_cfg.runs_dir
        / "autonomous_sector_benchmark"
        / benchmark_run_id
        / "free_data_repair"
    )
    sector_repairs: dict[str, Any] = {}
    for sector in sectors:
        payload = deepcopy(dict(candidate_payloads[sector]))
        payload["free_data_repair_policy"] = {
            "schema_version": FREE_DATA_REPAIR_SCHEMA_VERSION,
            "price_sources": ["stooq", "stooq_secondary"],
            "filing_forms": sorted(ANNUAL_ONLY_FILING_WINDOWS_DAYS),
            "terminal_search_allowed": False,
            "model_provider_allowed": False,
        }
        tickers = list(payload.get("execution_tickers") or [])
        checkpoint_path = run_root / f"{sector}_checkpoint.json"
        try:
            repair = pre_assembly_data_gap_repair(
                tickers=tickers,
                as_of_date=as_of_date,
                db_path=resolved_cfg.db_path,
                cfg=resolved_cfg,
                max_repairs=max(1, len(tickers)),
                pipeline_version="v2",
                checkpoint_path=checkpoint_path,
                checkpoint_scope=f"accepted_census_free_data:{sector}:{market_cap_focus}",
                candidate_context=payload,
                apply_repairs=True,
                run_id=benchmark_run_id,
                evidence_revision=FREE_DATA_REPAIR_SCHEMA_VERSION,
                candidate_context_revision=execution_set_fingerprint,
                terminal_cap_search=None,
                price_provider=effective_price_provider,
                filing_windows_days=dict(ANNUAL_ONLY_FILING_WINDOWS_DAYS),
            )
        except Exception as exc:  # noqa: BLE001 - preserve post-repair evidence
            repair = {
                "status": "INCOMPLETE",
                "execution_status": "INCOMPLETE",
                "examined": len(tickers),
                "completed_tickers": [],
                "pending_tickers": tickers,
                "checkpoint_path": str(checkpoint_path),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        sector_repairs[sector] = repair

    metrics_after = _http_metrics(resolved_cfg)
    calls_by_domain, network_calls, unexpected_domains = _network_delta(
        metrics_before, metrics_after
    )
    after = run_v2_readiness_preflight(
        sectors=sectors,
        candidate_payloads=candidate_payloads,
        as_of_date=as_of_date,
        market_cap_focus=market_cap_focus,
        execution_set_fingerprint=execution_set_fingerprint,
        request_fingerprint=request_fingerprint,
        candidate_resolution_errors=candidate_resolution_errors,
        accepted_census_authority=authority,
        cfg=resolved_cfg,
    )
    scope_preserved = (
        before.get("sector_bindings") == after.get("sector_bindings")
        and before.get("authority_fingerprint") == after.get("authority_fingerprint")
        and int((after.get("counts") or {}).get("execution_candidates") or 0)
        == execution_count
    )
    incomplete_sectors = sorted(
        sector
        for sector, repair in sector_repairs.items()
        if str(repair.get("status") or "").upper() != "COMPLETED"
    )
    changes = _readiness_changes(before, after)
    transition_counts = Counter(
        f"{row.get('before')}->{row.get('after')}" for row in changes
    )
    completed = bool(
        after.get("status") == "COMPLETED"
        and after.get("authority_validation_status") == "PASSED"
        and scope_preserved
        and not unexpected_domains
        and not incomplete_sectors
    )
    return {
        "schema_version": FREE_DATA_REPAIR_SCHEMA_VERSION,
        "status": "COMPLETED" if completed else "INCOMPLETE",
        "mode": FREE_DATA_REPAIR_MODE,
        "maintenance_only": True,
        "as_of_date": as_of_date,
        "market_cap_focus": market_cap_focus,
        "sectors": list(sectors),
        "request_fingerprint": request_fingerprint,
        "execution_set_fingerprint": execution_set_fingerprint,
        "census_lineage": dict(after.get("census_lineage") or {}),
        "authority_validation_status": (
            "PASSED" if scope_preserved else "FAILED"
        ),
        "scope_preserved": scope_preserved,
        "hard_limits": {
            "max_sectors": FREE_DATA_REPAIR_MAX_SECTORS,
            "max_candidates_per_sector": FREE_DATA_REPAIR_MAX_CANDIDATES_PER_SECTOR,
            "max_execution_names": FREE_DATA_REPAIR_MAX_EXECUTION_NAMES,
            "execution_names": execution_count,
        },
        "source_policy": {
            "accepted_cik_required": True,
            "companyfacts_source": "SEC_COMPANYFACTS",
            "filing_source": "SEC_EDGAR_ANNUAL_ONLY",
            "annual_filing_windows_days": dict(ANNUAL_ONLY_FILING_WINDOWS_DAYS),
            "price_sources": ["stooq", "stooq_secondary"],
            "paid_price_providers_allowed": [],
            "model_providers_allowed": [],
            "terminal_search_allowed": False,
        },
        "before_readiness": before,
        "after_readiness": after,
        "readiness_changes": changes,
        "readiness_transition_counts": dict(sorted(transition_counts.items())),
        "sector_repairs": sector_repairs,
        "incomplete_sectors": incomplete_sectors,
        "actual_usage": _zero_paid_usage(free_network_calls=network_calls),
        "network": {
            "allowed_domains": sorted(FREE_DATA_REPAIR_ALLOWED_DOMAINS),
            "calls_by_domain": calls_by_domain,
            "unexpected_domains": unexpected_domains,
        },
        "packet_materialized_count": 0,
        "production_sector_scan_exercised": False,
        "watchlist_mutation_exercised": False,
        "screened_candidate_count": 0,
        "underwritten_candidate_count": 0,
        "actionable_candidate_count": 0,
    }


__all__ = [
    "ANNUAL_ONLY_FILING_WINDOWS_DAYS",
    "FREE_DATA_REPAIR_ALLOWED_DOMAINS",
    "FREE_DATA_REPAIR_MAX_CANDIDATES_PER_SECTOR",
    "FREE_DATA_REPAIR_MAX_EXECUTION_NAMES",
    "FREE_DATA_REPAIR_MAX_SECTORS",
    "FREE_DATA_REPAIR_MODE",
    "FREE_DATA_REPAIR_SCHEMA_VERSION",
    "build_free_stooq_price_provider",
    "run_accepted_census_free_data_repair",
]
