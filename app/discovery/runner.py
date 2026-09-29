from __future__ import annotations

import copy
import csv
import json
import signal
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.discovery.market_cap import compute_market_cap, persist_market_cap
from app.discovery.metrics import DiscoveryMetricsResult, compute_discovery_metrics
from app.discovery.lineage import (
    publish_discovery_candidate_row,
    serialize_discovery_candidate_binding,
    serialized_discovery_candidate_binding_is_current,
)
from app.discovery.policy import DiscoveryFilingSelection, select_discovery_filings_from_submissions
from app.discovery.rubric import (
    count_critical_unknown,
    determine_discovery_stage,
    explainability_subscore,
    score_discovery_candidate,
    score_evidence_strength,
)
from app.discovery.schemas import (
    DiscoveryCandidate,
    DiscoveryReasonEvidence,
    DiscoveryRunStats,
    DiscoverySubscoreReason,
    UNKNOWN,
)
from app.discovery.seed import (
    merge_seed_with_active_universe,
    read_exclude_csv,
    read_seed_csv,
    seed_snapshot_hash,
)
from app.ingest.filings import _download_filing, _upsert_filing
from app.ingest.sec_client import SecClient
from app.logging import get_logger
from app.parse.filing_parser import parse_filing_by_id
from app.util.hashing import sha256_text
from app.valuation.price_provider import get_default_provider


logger = get_logger(__name__)


FINANCIALS_KEYWORDS = (
    "bank",
    "banking",
    "insurance",
    "insurer",
    "broker-dealer",
    "asset management",
    "commercial lending",
    "deposits",
    "underwriting",
)
BIOTECH_KEYWORDS = (
    "biotech",
    "biotechnology",
    "pharma",
    "pharmaceutical",
    "clinical trial",
    "fda",
    "drug candidate",
    "phase 1",
    "phase 2",
    "phase 3",
)
ROLLUP_KEYWORDS = (
    "acquisition",
    "acquire",
    "merger",
    "roll-up",
    "rollup",
    "purchase accounting",
)


PHASE_AUTO = "auto"
PHASE_PREFILTER = "prefilter"
PHASE_FULL = "full"
RUN_STATUS_RUNNING = "RUNNING"
RUN_STATUS_PARTIAL = "PARTIAL"
RUN_STATUS_COMPLETED = "COMPLETED"
RUN_STATUS_FAILED = "FAILED"
DISCOVERY_SCORING_VERSION = "v1.2"


class DiscoveryCancelled(Exception):
    pass


@dataclass(frozen=True)
class _CachedCandidatePayload:
    payload: dict[str, Any]
    source_binding: dict[str, Any]
    is_current_run: bool


def _load_filing_text(conn, ticker: str, accessions: list[str]) -> str:
    if not accessions:
        return ""
    placeholders = ",".join("?" for _ in accessions)
    rows = conn.execute(
        f"""
        SELECT local_path
        FROM filings
        WHERE ticker = ? AND accession IN ({placeholders})
        ORDER BY COALESCE(filing_date, '1900-01-01') DESC
        """,
        (ticker, *accessions),
    ).fetchall()
    chunks: list[str] = []
    for row in rows:
        local_path = row["local_path"]
        if not local_path:
            continue
        path = Path(local_path)
        if not path.exists():
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return " ".join(chunks).lower()


def _keyword_count(text: str, keywords: tuple[str, ...]) -> int:
    if not text:
        return 0
    return sum(text.count(word) for word in keywords)


def _suppression_reasons(
    *,
    cfg,
    ticker: str,
    company_name: str | None,
    filing_text: str,
    metrics: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    text = f"{(company_name or '').lower()} {filing_text or ''}"
    revenue = metrics.get("ttm_revenue")
    financial_hits = _keyword_count(text, FINANCIALS_KEYWORDS)
    biotech_hits = _keyword_count(text, BIOTECH_KEYWORDS)
    if cfg.discovery_suppress_prerevenue and (
        revenue == UNKNOWN or not isinstance(revenue, (int, float)) or revenue <= 0
    ):
        reasons.append("SKIPPED_SUPPRESSION:PREREVENUE")
    if cfg.discovery_suppress_financials and (
        financial_hits >= 8 or "bank holding company" in text or "insurance company" in text
    ):
        reasons.append("SKIPPED_SUPPRESSION:FINANCIALS")
    if cfg.discovery_suppress_biotech and (
        biotech_hits >= 5 or "clinical trial" in text and biotech_hits >= 3
    ):
        reasons.append("SKIPPED_SUPPRESSION:BIOTECH")
    if cfg.discovery_suppress_rollups and _keyword_count(text, ROLLUP_KEYWORDS) >= 4:
        reasons.append("SKIPPED_SUPPRESSION:ROLLUPS")
    return sorted(set(reasons))


def _claims_by_label(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for claim in claims:
        label = str(claim.get("label") or "").strip()
        if label:
            out[label] = claim
    return out


def _reason_label_candidates(reason: str) -> list[str]:
    lowered = reason.lower()
    labels: list[str] = []
    if "gross margin" in lowered:
        labels.append("gross_margin")
    if "operating" in lowered:
        labels.append("operating_margin")
    if "revenue" in lowered:
        labels.append("ttm_revenue")
    if "free cash flow" in lowered or "fcf" in lowered:
        labels.append("fcf")
    if "implied growth" in lowered:
        labels.append("implied_growth_proxy")
    return labels


def _reason_evidence(
    score_reasons: list[str],
    *,
    claims: list[dict[str, Any]],
    reason_limit: int = 8,
) -> tuple[list[str], list[dict[str, Any]], int, int]:
    key_reasons: list[str] = []
    evidence_rows: list[dict[str, Any]] = []
    claim_map = _claims_by_label(claims)
    cited_count = 0
    uncited_count = 0

    for reason in score_reasons[: max(1, reason_limit)]:
        labels = _reason_label_candidates(reason)
        citations: list[dict[str, str]] = []
        derived_from: list[str] = [f"discovery.rubric.reason::{reason.lower().replace(' ', '_')}"]
        for label in labels:
            claim = claim_map.get(label)
            if not claim:
                continue
            citations.extend(claim.get("citations") or [])
            derived_from.extend(claim.get("derived_from") or [])
        if citations:
            cited_count += 1
            key_reasons.append(reason)
        else:
            uncited_count += 1
            key_reasons.append(f"{reason} (no citation available)")
        evidence_rows.append(
            DiscoveryReasonEvidence(
                reason=key_reasons[-1],
                citations=citations[:2],
                derived_from=sorted(set(derived_from)),
            ).model_dump(mode="json")
        )
    return key_reasons, evidence_rows, cited_count, uncited_count


def _add_discovery_numeric_claims(
    *,
    base_claims: list[dict[str, Any]],
    subscores: dict[str, float],
    total_score: float,
    explainability: float,
    market_cap_result,
) -> list[dict[str, Any]]:
    claims = list(base_claims)
    for key, value in sorted(subscores.items()):
        claims.append(
            {
                "claim_id": f"subscore_{key}",
                "label": f"subscore_{key}",
                "value": float(value),
                "unit": "score",
                "citations": [],
                "derived_from": [f"discovery.rubric.{key}"],
            }
        )
    claims.append(
        {
            "claim_id": "discovery_total_score",
            "label": "discovery_total_score",
            "value": float(total_score),
            "unit": "score",
            "citations": [],
            "derived_from": ["discovery.rubric.total_score"],
        }
    )
    claims.append(
        {
            "claim_id": "discovery_explainability",
            "label": "discovery_explainability",
            "value": float(explainability),
            "unit": "score",
            "citations": [],
            "derived_from": ["discovery.rubric.explainability"],
        }
    )
    if isinstance(market_cap_result.market_cap, (int, float)):
        claims.append(
            {
                "claim_id": "market_cap",
                "label": "market_cap",
                "value": float(market_cap_result.market_cap),
                "unit": "USD",
                "citations": [],
                "derived_from": [
                    "discovery.metrics.market_cap",
                    f"market_cap.price_provider.{market_cap_result.provider}",
                ],
            }
        )
    return claims


def _update_discovery_lifecycle(
    conn,
    *,
    run_id: str,
    now: str,
    candidates: list[DiscoveryCandidate],
    shortlist_tickers: set[str],
) -> None:
    for candidate in candidates:
        row = conn.execute(
            """
            SELECT times_shortlisted
            FROM discovery_lifecycle
            WHERE ticker = ?
            """,
            (candidate.ticker,),
        ).fetchone()
        shortlisted_now = 1 if candidate.ticker in shortlist_tickers else 0
        if not row:
            conn.execute(
                """
                INSERT INTO discovery_lifecycle(
                    ticker, first_seen_run_id, first_seen_at, last_seen_run_id, last_seen_at,
                    times_shortlisted, last_discovery_score, last_action
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.ticker,
                    run_id,
                    now,
                    run_id,
                    now,
                    shortlisted_now,
                    candidate.discovery_score,
                    candidate.recommended_action,
                ),
            )
            continue
        new_times = int(row["times_shortlisted"] or 0) + shortlisted_now
        conn.execute(
            """
            UPDATE discovery_lifecycle
            SET last_seen_run_id = ?,
                last_seen_at = ?,
                times_shortlisted = ?,
                last_discovery_score = ?,
                last_action = ?
            WHERE ticker = ?
            """,
            (
                run_id,
                now,
                new_times,
                candidate.discovery_score,
                candidate.recommended_action,
                candidate.ticker,
            ),
        )


def _config_hash() -> str:
    cfg = get_config()
    return sha256_text(json.dumps(cfg.model_dump(mode="json"), sort_keys=True))


def generate_discovery_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"discovery_{ts}"


def _save_discovery_run_start(
    conn,
    *,
    run_id: str,
    run_as_of_date: str,
    seed_hash: str,
    seed_path: Path,
    tickers_targeted: list[str],
    status: str = RUN_STATUS_RUNNING,
    phase: str = PHASE_PREFILTER,
    processed_count: int = 0,
    stats: dict[str, Any] | None = None,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO discovery_runs(
            run_id, run_as_of_date, seed_hash, config_hash, seed_path,
            status, processed_count, phase,
            tickers_targeted_json, processed_effective_dates_json, stats_json, created_at, updated_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            run_as_of_date=excluded.run_as_of_date,
            seed_hash=excluded.seed_hash,
            config_hash=excluded.config_hash,
            seed_path=excluded.seed_path,
            status=excluded.status,
            processed_count=excluded.processed_count,
            phase=excluded.phase,
            tickers_targeted_json=excluded.tickers_targeted_json,
            stats_json=excluded.stats_json,
            updated_at=excluded.updated_at
        """,
        (
            run_id,
            run_as_of_date,
            seed_hash,
            _config_hash(),
            str(seed_path),
            status,
            int(processed_count),
            phase,
            json.dumps(tickers_targeted),
            json.dumps(stats or {}, sort_keys=True),
            now,
            now,
        ),
    )


def _save_discovery_run_finalize(
    conn,
    *,
    run_id: str,
    processed_effective_dates: dict[str, str],
    stats: dict[str, Any],
    status: str,
    phase: str,
    processed_count: int,
) -> None:
    conn.execute(
        """
        UPDATE discovery_runs
        SET processed_effective_dates_json = ?,
            stats_json = ?,
            status = ?,
            phase = ?,
            processed_count = ?,
            updated_at = ?
        WHERE run_id = ?
        """,
        (
            json.dumps(processed_effective_dates, sort_keys=True),
            json.dumps(stats, sort_keys=True),
            status,
            phase,
            int(processed_count),
            utc_now_iso(),
            run_id,
        ),
    )


def _save_discovery_progress(
    conn,
    *,
    run_id: str,
    phase: str,
    processed_count: int,
    stats: dict[str, Any],
    processed_effective_dates: dict[str, str],
    status: str = RUN_STATUS_RUNNING,
) -> None:
    _save_discovery_run_finalize(
        conn,
        run_id=run_id,
        processed_effective_dates=processed_effective_dates,
        stats=stats,
        status=status,
        phase=phase,
        processed_count=processed_count,
    )


def _load_run_row(conn, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT run_id, run_as_of_date, seed_hash, config_hash, seed_path, status, processed_count, phase,
               tickers_targeted_json, processed_effective_dates_json, stats_json
        FROM discovery_runs
        WHERE run_id = ?
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def _persist_candidate(
    conn,
    candidate: DiscoveryCandidate,
    *,
    source_candidate_binding: dict[str, Any] | None = None,
) -> None:
    if source_candidate_binding is not None:
        source_state = source_candidate_binding.get("candidate_state")
        if not serialized_discovery_candidate_binding_is_current(
            conn,
            source_candidate_binding,
        ) or not isinstance(source_state, dict):
            raise RuntimeError("cached discovery candidate source binding is not current")
        if (
            source_state.get("ticker") == candidate.ticker
            and source_state.get("run_id") == candidate.run_id
        ):
            raise RuntimeError("discovery candidate cannot cite itself as its cached source")
    payload = candidate.model_dump(mode="json")
    created_at = utc_now_iso()
    conn.execute(
        """
        INSERT INTO discovery_candidates(
            ticker, run_id, discovery_score, payload_json, created_at,
            publication_receipt_path, publication_receipt_sha256
        ) VALUES(?, ?, ?, ?, ?, NULL, NULL)
        ON CONFLICT(ticker, run_id) DO UPDATE SET
            discovery_score=excluded.discovery_score,
            payload_json=excluded.payload_json,
            created_at=excluded.created_at,
            publication_receipt_path=NULL,
            publication_receipt_sha256=NULL
        """,
        (
            candidate.ticker,
            candidate.run_id,
            candidate.discovery_score,
            json.dumps(payload),
            created_at,
        ),
    )
    row = conn.execute(
        """
        SELECT id
        FROM discovery_candidates
        WHERE ticker = ? AND run_id = ?
        LIMIT 1
        """,
        (candidate.ticker, candidate.run_id),
    ).fetchone()
    if row is None:
        raise RuntimeError("discovery candidate write did not produce an exact row")
    publish_discovery_candidate_row(
        conn,
        int(row["id"]),
        source_candidate_binding=source_candidate_binding,
    )


def _load_company_name(submissions: dict[str, Any]) -> str | None:
    name = submissions.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _candidate_stage_to_action(stage: str) -> tuple[str, str]:
    if stage == "ADVANCE_TO_DEEP":
        return "ADD_TO_UNIVERSE", "FULL_RESEARCH"
    if stage == "WATCHLIST_ONLY":
        return "WATCHLIST_ONLY", "NONE"
    return "SKIP", "NONE"


def _candidate_gaps(
    *,
    metrics: dict[str, Any],
    flags: list[str],
    critical_unknown_count: int,
    stage: str,
    evidence_strength_score: float,
) -> list[str]:
    gaps: list[str] = []
    if "MARKET_CAP_UNKNOWN" in flags:
        gaps.append("MARKET_CAP_UNKNOWN")
    if "FINANCIALS_INCOMPLETE" in flags:
        gaps.append("FINANCIALS_INCOMPLETE")
    if critical_unknown_count >= 2:
        gaps.append("CRITICAL_METRICS_UNKNOWN")
    if evidence_strength_score < 5:
        gaps.append("EVIDENCE_STRENGTH_LOW")
    for metric_name in [
        "ttm_revenue",
        "gross_margin",
        "operating_margin",
        "fcf",
        "shares_outstanding",
    ]:
        if metrics.get(metric_name) == UNKNOWN:
            gaps.append(f"MISSING_{metric_name.upper()}")
    if stage != "ADVANCE_TO_DEEP":
        gaps.append(f"STAGE_{stage}")
    return sorted(dict.fromkeys(gaps))[:8]


def _write_candidates_json(out_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    path = out_dir / f"discovery_candidates_{run_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_stats_json(out_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    path = out_dir / f"discovery_stats_{run_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_shortlist_json(out_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    path = out_dir / f"discovery_shortlist_{run_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_patch_csv(out_dir: Path, run_id: str, candidates: list[DiscoveryCandidate]) -> Path:
    path = out_dir / f"discovery_universe_patch_{run_id}.csv"
    fieldnames = [
        "ticker",
        "cik",
        "name",
        "homepage_url",
        "ir_rss_url",
        "allowlist_domains",
        "notes",
        "discovery_score",
        "recommended_action",
        "suggested_next_pipeline",
        "run_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            if candidate.recommended_action != "ADD_TO_UNIVERSE":
                continue
            writer.writerow(
                {
                    "ticker": candidate.ticker,
                    "cik": candidate.cik,
                    "name": candidate.company_name or "",
                    "homepage_url": "",
                    "ir_rss_url": "",
                    "allowlist_domains": "",
                    "notes": f"discovery run {run_id}",
                    "discovery_score": candidate.discovery_score,
                    "recommended_action": candidate.recommended_action,
                    "suggested_next_pipeline": candidate.suggested_next_pipeline,
                    "run_id": run_id,
                }
            )
    return path


def _write_report_md(
    out_dir: Path,
    run_id: str,
    run_as_of_date: str,
    candidates: list[DiscoveryCandidate],
    stats: dict[str, Any],
) -> Path:
    path = out_dir / f"discovery_report_{run_id}.md"
    suppressed_counts = stats.get("suppressed_counts") or {}
    stage_counts = stats.get("stage_counts") or {}
    unknown_market_cap = int(stats.get("unknown_market_cap_count", 0))
    filtered_market_cap = int(stats.get("filtered_market_cap_count", 0))
    advances = [c for c in candidates if c.stage == "ADVANCE_TO_DEEP"]
    watchlist = [c for c in candidates if c.stage == "WATCHLIST_ONLY"]

    def _fmt_mcap(value: float | str) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):,.0f}"
        return "UNKNOWN"

    def _fmt_reasons(values: list[str]) -> str:
        if not values:
            return "- none"
        top = values[:3]
        return "<br>".join(f"- {item}" for item in top)

    def _fmt_gaps(values: list[str]) -> str:
        if not values:
            return "-"
        return "; ".join(values[:4])

    lines = [
        f"# Discovery Report {run_id}",
        "",
        f"- Run as-of date: {run_as_of_date}",
        f"- Seed hash: {stats.get('seed_hash', 'UNKNOWN')}",
        "",
        "## Top Advances",
        "",
        "| Ticker | MktCap | Score | Stage | WhaleFit | Evidence | Key reasons (3 bullets) | Gaps |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    if not advances:
        lines.append("| (none) | - | - | - | - | - | - | - |")
    for candidate in advances[:25]:
        lines.append(
            f"| {candidate.ticker} | {_fmt_mcap(candidate.market_cap)} | {candidate.discovery_score:.2f} | "
            f"{candidate.stage} | {candidate.whale_fit_score:.2f} | {candidate.evidence_strength_score:.2f} | "
            f"{_fmt_reasons(candidate.key_reasons)} | {_fmt_gaps(candidate.gaps)} |"
        )

    lines.extend(
        [
            "",
            "## Top Watchlist",
            "",
            "| Ticker | MktCap | Score | Stage | WhaleFit | Evidence | Key reasons (3 bullets) | Gaps |",
            "|---|---:|---:|---|---:|---:|---|---|",
        ]
    )
    if not watchlist:
        lines.append("| (none) | - | - | - | - | - | - | - |")
    for candidate in watchlist[:25]:
        lines.append(
            f"| {candidate.ticker} | {_fmt_mcap(candidate.market_cap)} | {candidate.discovery_score:.2f} | "
            f"{candidate.stage} | {candidate.whale_fit_score:.2f} | {candidate.evidence_strength_score:.2f} | "
            f"{_fmt_reasons(candidate.key_reasons)} | {_fmt_gaps(candidate.gaps)} |"
        )

    lines.extend(
        [
            "",
            "## Coverage",
            f"- Processed count: {stats.get('tickers_processed', 0)}",
            f"- Missing CIK count: {stats.get('missing_cik_count', 0)}",
            f"- Filtered by market cap band count: {filtered_market_cap}",
            f"- Unknown market cap count: {unknown_market_cap}",
            f"- Shortlisted count: {stats.get('shortlisted_count', 0)}",
            f"- Stage counts: {json.dumps(stage_counts, sort_keys=True)}",
            f"- Suppressed counts: {json.dumps(suppressed_counts, sort_keys=True)}",
            "",
            "## Notes",
            "- Discovery uses static seed tickers plus active-universe merge, then deterministic SEC filing policy ranking.",
            "- Candidates are ranked from filings-derived metrics with market-cap target shaping and evidence floors.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _sec_throttle_delta(client: SecClient, baseline: int) -> int:
    metrics = client.http.metrics()
    current = int(metrics.get("throttled_count", 0))
    return max(0, current - baseline)


def _score_distribution(candidates: list[DiscoveryCandidate]) -> dict[str, int]:
    bins = {"80_100": 0, "60_79": 0, "40_59": 0, "0_39": 0}
    for candidate in candidates:
        score = float(candidate.discovery_score)
        if score >= 80:
            bins["80_100"] += 1
        elif score >= 60:
            bins["60_79"] += 1
        elif score >= 40:
            bins["40_59"] += 1
        else:
            bins["0_39"] += 1
    return bins


def _fetch_and_parse_minimal_filings(
    conn,
    *,
    client: SecClient,
    ticker: str,
    run_as_of_date: str,
    selection: DiscoveryFilingSelection,
) -> tuple[list[str], list[int]]:
    selected_accessions: list[str] = []
    filing_ids: list[int] = []
    for filing in selection.all_filings:
        filing_id = _upsert_filing(conn, ticker, filing, run_as_of_date)
        _download_filing(conn, client, filing_id, filing)
        selected_accessions.append(filing.accession)
        filing_ids.append(filing_id)
    return selected_accessions, filing_ids


def _effective_as_of_date(metrics: DiscoveryMetricsResult, fallback: str) -> str:
    return metrics.effective_as_of_date or fallback


def _persist_phase1_row(
    conn,
    *,
    run_id: str,
    ticker: str,
    cik: str | None,
    company_name: str | None,
    eligible: bool,
    prefilter_score: float,
    latest_filing_date: str | None,
    selected_accessions: list[str],
    status: str,
    reason: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO discovery_phase1(
            run_id, ticker, cik, company_name, eligible, prefilter_score, latest_filing_date,
            selected_accessions_json, status, reason, created_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, ticker) DO UPDATE SET
            cik=excluded.cik,
            company_name=excluded.company_name,
            eligible=excluded.eligible,
            prefilter_score=excluded.prefilter_score,
            latest_filing_date=excluded.latest_filing_date,
            selected_accessions_json=excluded.selected_accessions_json,
            status=excluded.status,
            reason=excluded.reason,
            created_at=excluded.created_at
        """,
        (
            run_id,
            ticker,
            cik,
            company_name,
            1 if eligible else 0,
            float(prefilter_score),
            latest_filing_date,
            json.dumps(sorted(set(selected_accessions))),
            status,
            reason,
            utc_now_iso(),
        ),
    )


def _prefilter_score_from_selection(
    selection: DiscoveryFilingSelection, run_as_of: date
) -> tuple[float, str | None]:
    filings = selection.all_filings
    if not filings:
        return 0.0, "no_selected_filings"
    latest = filings[0].filing_date
    age_days = max(0, (run_as_of - latest).days)
    has_annual = selection.annual is not None
    has_quarters = len(selection.quarters) > 0
    score = 0.0
    if has_annual:
        score += 40.0
    if has_quarters:
        score += 45.0
    score += max(0.0, 20.0 - min(20.0, age_days / 5.0))
    reason = None
    if not has_annual and not has_quarters:
        reason = "no_recent_10k_or_10q"
    return round(score, 2), reason


def _phase1_worker(
    *,
    ticker: str,
    run_as_of: date,
    cik_map: dict[str, str],
) -> dict[str, Any]:
    cik = cik_map.get(ticker)
    if not cik:
        return {
            "ticker": ticker,
            "status": "missing_cik",
            "eligible": False,
            "prefilter_score": 0.0,
            "selected_accessions": [],
            "selection": None,
            "latest_filing_date": None,
            "reason": "missing_cik",
            "company_name": None,
            "cik": None,
        }
    client = SecClient()
    submissions = client.submissions(cik)
    selection = select_discovery_filings_from_submissions(
        submissions,
        cik=cik,
        as_of_date=run_as_of,
        max_quarters=max(1, int(get_config().discovery_max_quarters)),
    )
    score, reason = _prefilter_score_from_selection(selection, run_as_of)
    filings = selection.all_filings
    latest_filing_date = filings[0].filing_date.isoformat() if filings else None
    return {
        "ticker": ticker,
        "status": "ok" if filings else "ineligible",
        "eligible": bool(filings),
        "prefilter_score": score,
        "selected_accessions": [f.accession for f in filings],
        "selection": selection,
        "latest_filing_date": latest_filing_date,
        "reason": reason,
        "company_name": _load_company_name(submissions),
        "cik": cik,
    }


def _load_cached_candidate_payload(
    conn,
    *,
    ticker: str,
    run_as_of_date: str,
    run_id: str | None,
    selected_accessions: list[str],
) -> _CachedCandidatePayload | None:
    def _dates_are_compatible(payload: dict[str, Any]) -> bool:
        payload_run_as_of = payload.get("run_as_of_date")
        effective_as_of = payload.get("effective_as_of_date")
        if (
            not isinstance(payload_run_as_of, str)
            or not isinstance(effective_as_of, str)
            or payload_run_as_of != run_as_of_date
        ):
            return False
        try:
            return date.fromisoformat(effective_as_of) <= date.fromisoformat(payload_run_as_of)
        except ValueError:
            return False

    def _is_payload_compatible(payload: dict[str, Any]) -> bool:
        subscores = payload.get("subscores_json") or {}
        if "stage" not in payload:
            return False
        if "whale_fit_score" not in payload or "evidence_strength_score" not in payload:
            return False
        if "whale_fit" not in subscores or "cap_band" not in subscores:
            return False
        if subscores.get("scoring_version") != DISCOVERY_SCORING_VERSION:
            return False
        return _dates_are_compatible(payload)

    def _validated_cached_row(
        row,
        *,
        is_current_run: bool,
    ) -> _CachedCandidatePayload | None:
        if row is None:
            return None
        source_binding = serialize_discovery_candidate_binding(row)
        if source_binding is None or not serialized_discovery_candidate_binding_is_current(
            conn,
            source_binding,
        ):
            return None
        payload = source_binding["candidate_state"]["payload"]
        if not isinstance(payload, dict) or not _is_payload_compatible(payload):
            return None
        prior_accessions = sorted(
            (payload.get("artifacts") or {}).get("filing_accessions_used") or []
        )
        if prior_accessions != sorted(selected_accessions):
            return None
        return _CachedCandidatePayload(
            payload=copy.deepcopy(payload),
            source_binding=copy.deepcopy(source_binding),
            is_current_run=is_current_run,
        )

    if run_id:
        row_current = conn.execute(
            """
            SELECT id, ticker, run_id, discovery_score, payload_json, created_at,
                   publication_receipt_path, publication_receipt_sha256
            FROM discovery_candidates
            WHERE ticker = ? AND run_id = ?
            LIMIT 1
            """,
            (ticker, run_id),
        ).fetchone()
        if row_current is not None:
            return _validated_cached_row(row_current, is_current_run=True)
    row = conn.execute(
        """
        SELECT dc.id, dc.ticker, dc.run_id, dc.discovery_score, dc.payload_json,
               dc.created_at, dc.publication_receipt_path, dc.publication_receipt_sha256
        FROM discovery_candidates dc
        JOIN discovery_runs dr ON dr.run_id = dc.run_id
        WHERE dc.ticker = ?
          AND dr.run_as_of_date = ?
          AND (? IS NULL OR dc.run_id <> ?)
        ORDER BY dr.created_at DESC, dc.created_at DESC, dc.id DESC
        LIMIT 1
        """,
        (ticker, run_as_of_date, run_id, run_id),
    ).fetchone()
    return _validated_cached_row(row, is_current_run=False)


def _candidate_from_cached_payload(
    payload: dict[str, Any],
    *,
    run_id: str,
    run_as_of_date: str,
    newly_surfaced: bool,
    repeat_surfaced: bool,
) -> DiscoveryCandidate:
    payload = dict(payload)
    source_run_as_of = payload.get("run_as_of_date")
    effective_as_of = payload.get("effective_as_of_date")
    if (
        not isinstance(source_run_as_of, str)
        or source_run_as_of != run_as_of_date
        or not isinstance(effective_as_of, str)
    ):
        raise ValueError("cached discovery candidate dates do not match the target run")
    try:
        effective_date = date.fromisoformat(effective_as_of)
        target_date = date.fromisoformat(run_as_of_date)
    except ValueError as exc:
        raise ValueError("cached discovery candidate dates are invalid") from exc
    if effective_date > target_date:
        raise ValueError("cached discovery candidate uses future-dated evidence")
    if "stage" not in payload:
        action = str(payload.get("recommended_action") or "").upper()
        if action == "ADD_TO_UNIVERSE":
            payload["stage"] = "ADVANCE_TO_DEEP"
        elif action == "WATCHLIST_ONLY":
            payload["stage"] = "WATCHLIST_ONLY"
        else:
            payload["stage"] = "REJECT"
    payload["run_id"] = run_id
    payload["run_as_of_date"] = run_as_of_date
    payload["newly_surfaced"] = bool(newly_surfaced)
    payload["repeat_surfaced"] = bool(repeat_surfaced)
    return DiscoveryCandidate.model_validate(payload)


def _discovery_progress_log(
    *,
    phase: str,
    processed_count: int,
    target_count: int,
    started_at: float,
    cache_skip_count: int,
) -> None:
    elapsed_min = max(0.001, (time.monotonic() - started_at) / 60.0)
    rate = processed_count / elapsed_min
    remaining = max(0, target_count - processed_count)
    eta_min = remaining / rate if rate > 0 else None
    logger.info(
        "discovery_progress_detailed",
        extra={
            "stage_name": "discovery",
            "stage_phase": phase,
            "stage_processed": processed_count,
            "stage_target": target_count,
            "stage_rate_tickers_per_min": round(rate, 2),
            "stage_eta_min": round(eta_min, 2) if eta_min is not None else None,
            "stage_cache_skips": cache_skip_count,
        },
    )


def run_discovery(
    *,
    as_of_date: str,
    limit: int | None = None,
    tickers: list[str] | None = None,
    seed_path: Path | None = None,
    exclude_path: Path | None = None,
    top_k: int | None = None,
    advance_top: int | None = None,
    run_id: str | None = None,
    phase: str = PHASE_AUTO,
    prefilter_cap: int | None = None,
    prefilter_keep_ratio: float | None = None,
    workers: int | None = None,
    mktcap_min: float | None = None,
    mktcap_max: float | None = None,
    resume: bool = False,
    _cancel_after: int | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    run_id = run_id or generate_discovery_run_id()
    top_k = int(top_k or cfg.discovery_top_k_default)
    advance_top = int(advance_top) if advance_top is not None else None
    seed_path = seed_path or cfg.discovery_seed_path
    phase = (phase or PHASE_AUTO).strip().lower()
    if phase not in {PHASE_AUTO, PHASE_PREFILTER, PHASE_FULL}:
        raise ValueError("phase must be one of: auto, prefilter, full")
    prefilter_keep_ratio = float(prefilter_keep_ratio or cfg.discovery_prefilter_keep_ratio_default)
    prefilter_keep_ratio = max(0.05, min(0.8, prefilter_keep_ratio))
    workers = max(1, int(workers or cfg.discovery_workers))
    cap_min = float(mktcap_min if mktcap_min is not None else cfg.discovery_market_cap_min)
    cap_max = float(mktcap_max if mktcap_max is not None else cfg.discovery_market_cap_max)
    run_as_of = date.fromisoformat(as_of_date)

    existing_row: dict[str, Any] | None = None
    with get_db() as conn:
        existing_row = _load_run_row(conn, run_id)
    if existing_row and resume:
        existing_seed_path = str(existing_row.get("seed_path") or "").strip()
        if existing_seed_path:
            seed_path = Path(existing_seed_path)

    seed_tickers = read_seed_csv(seed_path)
    excludes = read_exclude_csv(exclude_path)
    seed_hash = seed_snapshot_hash(
        seed_tickers, include_metadata={"seed_path": str(seed_path.resolve())}
    )

    with get_db() as conn:
        scope_all = merge_seed_with_active_universe(conn, seed_tickers, excludes)
    if tickers:
        scope_all = [t.upper() for t in tickers if t.strip()]

    max_budget = max(1, int(cfg.discovery_max_tickers_per_run))
    if limit is not None and limit > 0:
        max_budget = min(max_budget, int(limit))
    scope = sorted(scope_all)[:max_budget]

    stats: dict[str, Any] = {}
    processed_effective_dates: dict[str, str] = {}
    phase1_completed: set[str] = set()
    phase2_completed: set[str] = set()
    prefilter_tickers: list[str] = []
    if existing_row:
        if not resume and run_id:
            raise ValueError("run_id already exists; use discovery-resume --run-id to continue")
        existing_seed_hash = str(existing_row.get("seed_hash") or "").strip()
        if existing_seed_hash:
            seed_hash = existing_seed_hash
        existing_seed_path = str(existing_row.get("seed_path") or "").strip()
        if existing_seed_path:
            seed_path = Path(existing_seed_path)
        try:
            stats = json.loads(existing_row.get("stats_json") or "{}")
        except Exception:
            stats = {}
        try:
            processed_effective_dates = json.loads(
                existing_row.get("processed_effective_dates_json") or "{}"
            )
        except Exception:
            processed_effective_dates = {}
        phase1_completed = {str(t).upper() for t in stats.get("phase1_completed_tickers", [])}
        phase2_completed = {str(t).upper() for t in stats.get("phase2_completed_tickers", [])}
        prefilter_tickers = [str(t).upper() for t in stats.get("prefilter_tickers", [])]
        try:
            scope_from_row = json.loads(existing_row.get("tickers_targeted_json") or "[]")
            scope = [str(t).upper() for t in scope_from_row if str(t).strip()]
        except Exception:
            pass
    else:
        stats = {
            "phase1_count": 0,
            "phase2_count": 0,
            "tickers_skipped_cached": 0,
            "tickers_downloaded": 0,
            "tickers_parsed": 0,
            "missing_cik_count": 0,
            "errors_count": 0,
            "market_cap_known_count": 0,
            "market_cap_in_band_count": 0,
            "filtered_market_cap_count": 0,
            "unknown_market_cap_count": 0,
            "suppressed_counts": {},
            "stage_counts": {},
            "phase1_completed_tickers": [],
            "phase2_completed_tickers": [],
            "prefilter_tickers": [],
        }
        with get_db() as conn:
            _save_discovery_run_start(
                conn,
                run_id=run_id,
                run_as_of_date=as_of_date,
                seed_hash=seed_hash,
                seed_path=seed_path,
                tickers_targeted=scope,
                status=RUN_STATUS_RUNNING,
                phase=PHASE_PREFILTER,
                processed_count=0,
                stats=stats,
            )
    if existing_row and resume:
        with get_db() as conn:
            _save_discovery_run_start(
                conn,
                run_id=run_id,
                run_as_of_date=as_of_date,
                seed_hash=seed_hash,
                seed_path=seed_path,
                tickers_targeted=scope,
                status=RUN_STATUS_RUNNING,
                phase=PHASE_PREFILTER if phase in {PHASE_AUTO, PHASE_PREFILTER} else PHASE_FULL,
                processed_count=len(
                    phase2_completed if phase in {PHASE_AUTO, PHASE_FULL} else phase1_completed
                ),
                stats=stats,
            )

    out_dir = cfg.discovery_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = get_default_provider(cfg)
    from app.universe.ticker_cik_map import load_ticker_cik_map

    cik_map = load_ticker_cik_map(refresh_if_missing=True)
    candidate_map: dict[str, DiscoveryCandidate] = {}
    for prior in load_discovery_candidates(run_id):
        try:
            model = DiscoveryCandidate.model_validate(prior)
            candidate_map[model.ticker] = model
        except Exception:
            continue

    cancel_event = threading.Event()
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):  # noqa: ANN001
        _ = (signum, frame)
        cancel_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    started_at = time.monotonic()
    stopped_throttle = False
    phase1_results: dict[str, dict[str, Any]] = {}
    phase2_target_count = 0

    def _run_pool(
        items: list[str],
        worker_fn,
        *,
        phase_name: str,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not items:
            return out
        pending = list(items)
        futures: dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            while pending and len(futures) < max(1, workers) and not cancel_event.is_set():
                ticker = pending.pop(0)
                futures[executor.submit(worker_fn, ticker)] = ticker
            while futures:
                done, _ = wait(list(futures.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
                if not done:
                    if cancel_event.is_set():
                        break
                    continue
                for future in done:
                    ticker = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "ticker": ticker,
                            "status": "error",
                            "error": str(exc),
                        }
                    out.append(result)
                    if _cancel_after is not None and len(out) >= int(_cancel_after):
                        cancel_event.set()
                while pending and len(futures) < max(1, workers) and not cancel_event.is_set():
                    ticker = pending.pop(0)
                    futures[executor.submit(worker_fn, ticker)] = ticker
                if cancel_event.is_set():
                    break
            if cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                deadline = time.monotonic() + float(cfg.discovery_cancel_drain_seconds)
                while futures and time.monotonic() < deadline:
                    done, _ = wait(list(futures.keys()), timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        ticker = futures.pop(future)
                        try:
                            out.append(future.result())
                        except Exception as exc:  # noqa: BLE001
                            out.append({"ticker": ticker, "status": "error", "error": str(exc)})
        _discovery_progress_log(
            phase=phase_name,
            processed_count=len(out),
            target_count=len(items),
            started_at=started_at,
            cache_skip_count=int(stats.get("tickers_skipped_cached", 0)),
        )
        return out

    try:
        # Phase 1: submissions-only prefilter (unless --phase full)
        if phase in {PHASE_AUTO, PHASE_PREFILTER}:
            pending_phase1 = [t for t in scope if t not in phase1_completed]

            def _phase1_fn(ticker: str) -> dict[str, Any]:
                return _phase1_worker(ticker=ticker, run_as_of=run_as_of, cik_map=cik_map)

            phase1_rows = _run_pool(pending_phase1, _phase1_fn, phase_name=PHASE_PREFILTER)
            flush_rows: list[dict[str, Any]] = []
            for row in phase1_rows:
                ticker = str(row.get("ticker") or "").upper()
                if not ticker:
                    continue
                phase1_results[ticker] = row
                phase1_completed.add(ticker)
                if row.get("status") == "missing_cik":
                    stats["missing_cik_count"] = int(stats.get("missing_cik_count", 0)) + 1
                if row.get("status") == "error":
                    stats["errors_count"] = int(stats.get("errors_count", 0)) + 1
                stats["phase1_count"] = int(stats.get("phase1_count", 0)) + 1
                flush_rows.append(row)

            if flush_rows:
                with get_db() as conn:
                    for row in flush_rows:
                        _persist_phase1_row(
                            conn,
                            run_id=run_id,
                            ticker=row["ticker"],
                            cik=row.get("cik"),
                            company_name=row.get("company_name"),
                            eligible=bool(row.get("eligible")),
                            prefilter_score=float(row.get("prefilter_score", 0.0)),
                            latest_filing_date=row.get("latest_filing_date"),
                            selected_accessions=row.get("selected_accessions") or [],
                            status=row.get("status", "unknown"),
                            reason=row.get("reason"),
                        )

            stats["phase1_completed_tickers"] = sorted(phase1_completed)
            with get_db() as conn:
                _save_discovery_progress(
                    conn,
                    run_id=run_id,
                    phase=PHASE_PREFILTER,
                    processed_count=len(phase1_completed),
                    stats=stats,
                    processed_effective_dates=processed_effective_dates,
                    status=RUN_STATUS_RUNNING if not cancel_event.is_set() else RUN_STATUS_PARTIAL,
                )

            with get_db() as conn:
                pre_rows = conn.execute(
                    """
                    SELECT ticker, prefilter_score
                    FROM discovery_phase1
                    WHERE run_id = ? AND eligible = 1
                    ORDER BY prefilter_score DESC, ticker ASC
                    """,
                    (run_id,),
                ).fetchall()
            pre_ranked = [str(row["ticker"]).upper() for row in pre_rows]
            prefilter_cap_value = int(
                prefilter_cap
                if prefilter_cap is not None
                else min(len(scope), int(cfg.discovery_prefilter_cap_default))
            )
            prefilter_cap_value = max(1, min(prefilter_cap_value, len(scope)))
            ratio_keep = max(1, int(round(len(scope) * prefilter_keep_ratio)))
            min_keep = max(1, min(prefilter_cap_value, min(len(scope), max(5, int(top_k)))))
            keep_count = max(min_keep, min(prefilter_cap_value, ratio_keep))
            prefilter_tickers = pre_ranked[:keep_count]
            stats["prefilter_tickers"] = prefilter_tickers
        else:
            prefilter_tickers = list(scope)
            stats["prefilter_tickers"] = prefilter_tickers

        if phase in {PHASE_AUTO, PHASE_FULL} and not cancel_event.is_set():
            phase2_scope = (
                [t for t in prefilter_tickers if t in scope]
                if phase == PHASE_AUTO
                else [t for t in scope]
            )
            phase2_scope = [t for t in phase2_scope if t not in phase2_completed]
            phase2_scope = sorted(dict.fromkeys(phase2_scope))
            if not phase2_scope:
                phase2_scope = []
            phase2_target_count = len(phase2_scope)

            db_sem = threading.Semaphore(max(1, int(cfg.max_db_write_concurrency)))

            def _phase2_fn(ticker: str) -> dict[str, Any]:
                cik = cik_map.get(ticker)
                if not cik:
                    return {"ticker": ticker, "status": "missing_cik"}
                local_client = SecClient()
                submissions = local_client.submissions(cik)
                selection = select_discovery_filings_from_submissions(
                    submissions,
                    cik=cik,
                    as_of_date=run_as_of,
                    max_quarters=max(1, int(cfg.discovery_max_quarters)),
                )
                selected_accessions = [f.accession for f in selection.all_filings]
                if not selected_accessions:
                    return {"ticker": ticker, "status": "ineligible"}

                with db_sem:
                    with get_db() as conn:
                        lifecycle_row = conn.execute(
                            "SELECT first_seen_run_id FROM discovery_lifecycle WHERE ticker = ? LIMIT 1",
                            (ticker,),
                        ).fetchone()
                        cached_candidate = _load_cached_candidate_payload(
                            conn,
                            ticker=ticker,
                            run_as_of_date=as_of_date,
                            run_id=run_id,
                            selected_accessions=selected_accessions,
                        )
                if cached_candidate is not None:
                    if cached_candidate.is_current_run:
                        candidate = DiscoveryCandidate.model_validate(
                            copy.deepcopy(cached_candidate.payload)
                        )
                    else:
                        candidate = _candidate_from_cached_payload(
                            cached_candidate.payload,
                            run_id=run_id,
                            run_as_of_date=as_of_date,
                            newly_surfaced=lifecycle_row is None,
                            repeat_surfaced=lifecycle_row is not None,
                        )
                        with db_sem:
                            with get_db() as conn:
                                _persist_candidate(
                                    conn,
                                    candidate,
                                    source_candidate_binding=cached_candidate.source_binding,
                                )
                    return {
                        "ticker": ticker,
                        "status": "cached",
                        "candidate": candidate,
                        "downloaded": 0,
                        "parsed": 0,
                    }

                with db_sem:
                    with get_db() as conn:
                        selected_accessions_db, filing_ids = _fetch_and_parse_minimal_filings(
                            conn,
                            client=local_client,
                            ticker=ticker,
                            run_as_of_date=as_of_date,
                            selection=selection,
                        )
                parsed_count = 0
                for filing_id in filing_ids:
                    with db_sem:
                        if parse_filing_by_id(filing_id):
                            parsed_count += 1

                with db_sem:
                    with get_db() as conn:
                        metrics_result = compute_discovery_metrics(
                            conn, ticker=ticker, selected_accessions=selected_accessions_db
                        )
                        if not metrics_result:
                            return {"ticker": ticker, "status": "ineligible"}
                        effective_as_of_date = _effective_as_of_date(metrics_result, as_of_date)
                        shares = metrics_result.metrics.get("shares_outstanding", UNKNOWN)
                        market_cap_result = compute_market_cap(
                            ticker=ticker,
                            run_id=run_id,
                            run_as_of_date=as_of_date,
                            effective_as_of_date=effective_as_of_date,
                            shares_outstanding=shares,
                            price_provider=provider,
                            cap_min=cap_min,
                            cap_max=cap_max,
                        )
                        persist_market_cap(conn, market_cap_result)
                        score_result = score_discovery_candidate(
                            metrics=metrics_result.metrics,
                            market_cap=market_cap_result.market_cap,
                            market_cap_in_band=market_cap_result.market_cap_in_band,
                            price=market_cap_result.price,
                            shares_outstanding=market_cap_result.shares_outstanding,
                        )
                        filing_text = _load_filing_text(
                            conn, ticker, metrics_result.filing_accessions_used
                        )
                        suppression_reasons = _suppression_reasons(
                            cfg=cfg,
                            ticker=ticker,
                            company_name=_load_company_name(submissions),
                            filing_text=filing_text,
                            metrics=metrics_result.metrics,
                        )
                        key_reasons, reason_evidence, cited_reason_count, uncited_reason_count = (
                            _reason_evidence(
                                score_result.reasons,
                                claims=metrics_result.claims,
                            )
                        )
                        for reason in suppression_reasons:
                            label = reason.replace("SKIPPED_SUPPRESSION:", "Suppressed: ")
                            key_reasons.append(label)
                            reason_evidence.append(
                                DiscoveryReasonEvidence(
                                    reason=label,
                                    citations=[],
                                    derived_from=[
                                        f"discovery.suppression.{reason.split(':', 1)[1].lower()}"
                                    ],
                                ).model_dump(mode="json")
                            )
                        critical_unknown_count = count_critical_unknown(metrics_result.metrics)
                        explainability = explainability_subscore(
                            cited_reason_count=cited_reason_count,
                            reason_count=len(reason_evidence),
                            critical_unknown_count=critical_unknown_count,
                        )
                        evidence_strength, evidence_reason_objects = score_evidence_strength(
                            critical_unknown_count=critical_unknown_count,
                            explainability_score=explainability,
                            metrics=metrics_result.metrics,
                        )
                        total_score = round(
                            min(
                                100.0,
                                float(score_result.total_score)
                                + float(explainability)
                                + float(evidence_strength),
                            ),
                            2,
                        )
                        subscores = {
                            **score_result.subscores,
                            "explainability": explainability,
                            "evidence_strength": evidence_strength,
                        }
                        flags = sorted(
                            set(
                                metrics_result.flags
                                + market_cap_result.flags
                                + score_result.flags
                                + suppression_reasons
                            )
                        )
                        if uncited_reason_count > 0:
                            flags.append("EXPLAINABILITY_PENALTY_UNCITED")
                        flags = sorted(set(flags))
                        stage, stage_reason_objects = determine_discovery_stage(
                            total_score=total_score,
                            whale_fit_score=float(subscores.get("whale_fit", 0.0)),
                            evidence_strength_score=evidence_strength,
                            market_cap=market_cap_result.market_cap,
                            market_cap_in_band=market_cap_result.market_cap_in_band,
                            critical_unknown_count=critical_unknown_count,
                            suppressed=bool(suppression_reasons),
                        )
                        action, next_pipeline = _candidate_stage_to_action(stage)
                        gaps = _candidate_gaps(
                            metrics=metrics_result.metrics,
                            flags=flags,
                            critical_unknown_count=critical_unknown_count,
                            stage=stage,
                            evidence_strength_score=evidence_strength,
                        )
                        lifecycle_row = conn.execute(
                            "SELECT first_seen_run_id FROM discovery_lifecycle WHERE ticker = ? LIMIT 1",
                            (ticker,),
                        ).fetchone()
                        numeric_claims = _add_discovery_numeric_claims(
                            base_claims=metrics_result.claims,
                            subscores=subscores,
                            total_score=total_score,
                            explainability=explainability,
                            market_cap_result=market_cap_result,
                        )
                        evidence_row = conn.execute(
                            "SELECT packet_path FROM evidence_packets WHERE ticker = ? AND as_of_date = ? LIMIT 1",
                            (ticker, effective_as_of_date),
                        ).fetchone()
                        evidence_packet_path = evidence_row["packet_path"] if evidence_row else None
                        candidate = DiscoveryCandidate(
                            ticker=ticker,
                            cik=str(cik),
                            company_name=_load_company_name(submissions),
                            run_id=run_id,
                            run_as_of_date=as_of_date,
                            effective_as_of_date=effective_as_of_date,
                            market_cap=market_cap_result.market_cap,
                            discovery_score=total_score,
                            stage=stage,
                            whale_fit_score=float(subscores.get("whale_fit", 0.0)),
                            evidence_strength_score=evidence_strength,
                            explainability_score=explainability,
                            subscores_json={
                                **subscores,
                                "metrics": metrics_result.metrics,
                                "scoring_version": DISCOVERY_SCORING_VERSION,
                            },
                            key_reasons=key_reasons[:8],
                            gaps=gaps,
                            reason_evidence=reason_evidence[:8],
                            subscore_reasons=[
                                DiscoverySubscoreReason.model_validate(item).model_dump(mode="json")
                                for item in (
                                    score_result.reason_objects
                                    + evidence_reason_objects
                                    + stage_reason_objects
                                )[:12]
                            ],
                            numeric_claims=numeric_claims,
                            flags=flags,
                            newly_surfaced=lifecycle_row is None,
                            repeat_surfaced=lifecycle_row is not None,
                            recommended_action=action,
                            suggested_next_pipeline=next_pipeline,
                            artifacts={
                                "evidence_packet_path": evidence_packet_path,
                                "filing_accessions_used": metrics_result.filing_accessions_used,
                            },
                        )
                        _persist_candidate(conn, candidate)
                        return {
                            "ticker": ticker,
                            "status": "ok",
                            "candidate": candidate,
                            "downloaded": len(filing_ids),
                            "parsed": parsed_count,
                            "market_cap_known": isinstance(
                                market_cap_result.market_cap, (int, float)
                            ),
                            "market_cap_in_band": bool(market_cap_result.market_cap_in_band),
                            "market_cap_unknown": not isinstance(
                                market_cap_result.market_cap, (int, float)
                            ),
                            "suppression_reasons": suppression_reasons,
                            "effective_as_of_date": effective_as_of_date,
                            "stage": stage,
                        }

            phase2_rows = _run_pool(phase2_scope, _phase2_fn, phase_name=PHASE_FULL)
            for idx, row in enumerate(phase2_rows, start=1):
                ticker = str(row.get("ticker") or "").upper()
                if not ticker:
                    continue
                phase2_completed.add(ticker)
                status_row = row.get("status")
                if status_row == "cached":
                    stats["tickers_skipped_cached"] = (
                        int(stats.get("tickers_skipped_cached", 0)) + 1
                    )
                if status_row in {"ok", "cached"} and row.get("candidate") is not None:
                    candidate_obj = row["candidate"]
                    if isinstance(candidate_obj, dict):
                        candidate_obj = DiscoveryCandidate.model_validate(candidate_obj)
                    candidate_map[ticker] = candidate_obj
                    stage_counts = stats.get("stage_counts") or {}
                    stage_key = str(candidate_obj.stage or "WATCHLIST_ONLY")
                    stage_counts[stage_key] = int(stage_counts.get(stage_key, 0)) + 1
                    stats["stage_counts"] = stage_counts
                if status_row == "ok":
                    stats["tickers_downloaded"] = int(stats.get("tickers_downloaded", 0)) + int(
                        row.get("downloaded", 0)
                    )
                    stats["tickers_parsed"] = int(stats.get("tickers_parsed", 0)) + int(
                        row.get("parsed", 0)
                    )
                    if row.get("market_cap_known"):
                        stats["market_cap_known_count"] = (
                            int(stats.get("market_cap_known_count", 0)) + 1
                        )
                    if row.get("market_cap_in_band"):
                        stats["market_cap_in_band_count"] = (
                            int(stats.get("market_cap_in_band_count", 0)) + 1
                        )
                    elif row.get("market_cap_unknown"):
                        stats["unknown_market_cap_count"] = (
                            int(stats.get("unknown_market_cap_count", 0)) + 1
                        )
                    else:
                        stats["filtered_market_cap_count"] = (
                            int(stats.get("filtered_market_cap_count", 0)) + 1
                        )
                    for reason in row.get("suppression_reasons") or []:
                        counts = stats.get("suppressed_counts") or {}
                        counts[reason] = int(counts.get(reason, 0)) + 1
                        stats["suppressed_counts"] = counts
                    eff = row.get("effective_as_of_date")
                    if eff:
                        processed_effective_dates[ticker] = str(eff)
                if status_row == "error":
                    stats["errors_count"] = int(stats.get("errors_count", 0)) + 1
                stats["phase2_count"] = int(stats.get("phase2_count", 0)) + 1

                if idx % 25 == 0:
                    stats["phase2_completed_tickers"] = sorted(phase2_completed)
                    with get_db() as conn:
                        _save_discovery_progress(
                            conn,
                            run_id=run_id,
                            phase=PHASE_FULL,
                            processed_count=len(phase2_completed),
                            stats=stats,
                            processed_effective_dates=processed_effective_dates,
                            status=RUN_STATUS_RUNNING
                            if not cancel_event.is_set()
                            else RUN_STATUS_PARTIAL,
                        )
                    _discovery_progress_log(
                        phase=PHASE_FULL,
                        processed_count=len(phase2_completed),
                        target_count=len(phase2_scope),
                        started_at=started_at,
                        cache_skip_count=int(stats.get("tickers_skipped_cached", 0)),
                    )
            stats["phase2_completed_tickers"] = sorted(phase2_completed)

        candidates = sorted(candidate_map.values(), key=lambda c: (-c.discovery_score, c.ticker))
        advance_top_limit = max(1, int(advance_top if advance_top is not None else top_k))
        shortlist = [c for c in candidates if c.stage == "ADVANCE_TO_DEEP"][:advance_top_limit]
        shortlist = sorted(shortlist, key=lambda c: (-c.discovery_score, c.ticker))

        with get_db() as conn:
            _update_discovery_lifecycle(
                conn,
                run_id=run_id,
                now=utc_now_iso(),
                candidates=candidates,
                shortlist_tickers={candidate.ticker for candidate in shortlist},
            )

        status = RUN_STATUS_PARTIAL if cancel_event.is_set() else RUN_STATUS_COMPLETED
        stats_model = DiscoveryRunStats(
            run_id=run_id,
            run_as_of_date=as_of_date,
            seed_hash=seed_hash,
            config_hash=_config_hash(),
            tickers_targeted=len(scope),
            tickers_processed=len(candidates),
            missing_cik_count=int(stats.get("missing_cik_count", 0)),
            skipped_due_to_throttle=stopped_throttle,
            errors_count=int(stats.get("errors_count", 0)),
            market_cap_known_count=int(stats.get("market_cap_known_count", 0)),
            market_cap_in_band_count=int(stats.get("market_cap_in_band_count", 0)),
            filtered_market_cap_count=int(stats.get("filtered_market_cap_count", 0)),
            unknown_market_cap_count=int(stats.get("unknown_market_cap_count", 0)),
            shortlisted_count=len(shortlist),
            shadow_count=sum(1 for c in candidates if "MARKET_CAP_UNKNOWN" in c.flags),
            suppressed_counts=dict(sorted((stats.get("suppressed_counts") or {}).items())),
            stage_counts=dict(sorted((stats.get("stage_counts") or {}).items())),
            top_k=top_k,
        )
        final_stats = stats_model.model_dump(mode="json")
        final_stats.update(
            {
                "status": status,
                "phase1_count": int(stats.get("phase1_count", 0)),
                "phase2_count": int(stats.get("phase2_count", 0)),
                "tickers_skipped_cached": int(stats.get("tickers_skipped_cached", 0)),
                "tickers_downloaded": int(stats.get("tickers_downloaded", 0)),
                "tickers_parsed": int(stats.get("tickers_parsed", 0)),
                "phase1_completed_tickers": sorted(phase1_completed),
                "phase2_completed_tickers": sorted(phase2_completed),
                "prefilter_tickers": stats.get("prefilter_tickers", prefilter_tickers),
            }
        )
        final_stats["score_distribution"] = _score_distribution(candidates)

        if phase == PHASE_PREFILTER:
            final_processed_count = len(phase1_completed)
            final_target_count = len(scope)
        elif phase == PHASE_FULL:
            final_processed_count = len(phase2_completed)
            final_target_count = phase2_target_count if phase2_target_count > 0 else len(scope)
        else:
            if phase2_target_count > 0:
                final_processed_count = len(phase2_completed)
                final_target_count = phase2_target_count
            else:
                final_processed_count = len(phase1_completed)
                final_target_count = len(scope)

        with get_db() as conn:
            _save_discovery_run_finalize(
                conn,
                run_id=run_id,
                processed_effective_dates=processed_effective_dates,
                stats=final_stats,
                status=status,
                phase=PHASE_FULL if phase in {PHASE_AUTO, PHASE_FULL} else PHASE_PREFILTER,
                processed_count=final_processed_count,
            )

        candidates_payload = {
            "run_id": run_id,
            "run_as_of_date": as_of_date,
            "seed_hash": seed_hash,
            "config_hash": _config_hash(),
            "status": status,
            "stats": final_stats,
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        }
        shortlist_payload = {
            "run_id": run_id,
            "run_as_of_date": as_of_date,
            "seed_hash": seed_hash,
            "status": status,
            "top_k": top_k,
            "candidate_count": len(shortlist),
            "tickers": [candidate.ticker for candidate in shortlist],
            "candidates": [candidate.model_dump(mode="json") for candidate in shortlist],
        }

        candidates_path = _write_candidates_json(out_dir, run_id, candidates_payload)
        stats_path = _write_stats_json(out_dir, run_id, final_stats)
        shortlist_path = _write_shortlist_json(out_dir, run_id, shortlist_payload)
        patch_path = _write_patch_csv(out_dir, run_id, candidates)
        report_path = _write_report_md(out_dir, run_id, as_of_date, candidates, final_stats)

        summary = {
            "run_id": run_id,
            "status": status,
            "run_as_of_date": as_of_date,
            "seed_path": str(seed_path),
            "seed_hash": seed_hash,
            "phase": phase,
            "tickers_targeted": len(scope),
            "tickers_processed": len(candidates),
            "processed_count": final_processed_count,
            "target_count": final_target_count,
            "phase1_count": int(final_stats.get("phase1_count", 0)),
            "phase2_count": int(final_stats.get("phase2_count", 0)),
            "tickers_skipped_cached": int(final_stats.get("tickers_skipped_cached", 0)),
            "tickers_downloaded": int(final_stats.get("tickers_downloaded", 0)),
            "tickers_parsed": int(final_stats.get("tickers_parsed", 0)),
            "shortlist_count": len(shortlist),
            "shortlist_tickers": [candidate.ticker for candidate in shortlist],
            "advance_top": advance_top if advance_top is not None else top_k,
            "stage_counts": final_stats.get("stage_counts", {}),
            "discovery_candidates_path": str(candidates_path),
            "discovery_shortlist_path": str(shortlist_path),
            "discovery_universe_patch_path": str(patch_path),
            "discovery_report_path": str(report_path),
            "discovery_stats_path": str(stats_path),
        }
        return summary
    except KeyboardInterrupt:
        cancel_event.set()
        raise
    except Exception:
        if phase == PHASE_PREFILTER:
            failed_processed_count = len(phase1_completed)
        elif phase == PHASE_FULL:
            failed_processed_count = len(phase2_completed)
        else:
            failed_processed_count = (
                len(phase2_completed) if len(phase2_completed) > 0 else len(phase1_completed)
            )
        with get_db() as conn:
            _save_discovery_run_finalize(
                conn,
                run_id=run_id,
                processed_effective_dates=processed_effective_dates,
                stats=stats,
                status=RUN_STATUS_FAILED,
                phase=phase,
                processed_count=failed_processed_count,
            )
        raise
    finally:
        signal.signal(signal.SIGINT, prev_sigint)


def run_discovery_all(
    *,
    as_of_date: str,
    limit: int | None = None,
    top_k: int | None = None,
    advance_top: int | None = None,
    seed_path: Path | None = None,
    run_id: str | None = None,
    mktcap_min: float | None = None,
    mktcap_max: float | None = None,
) -> dict[str, Any]:
    return run_discovery(
        as_of_date=as_of_date,
        limit=limit,
        tickers=None,
        seed_path=seed_path,
        exclude_path=None,
        top_k=top_k,
        advance_top=advance_top,
        run_id=run_id,
        mktcap_min=mktcap_min,
        mktcap_max=mktcap_max,
    )


def run_discovery_resume(
    *,
    run_id: str,
    phase: str = PHASE_AUTO,
    prefilter_cap: int | None = None,
    prefilter_keep_ratio: float | None = None,
    workers: int | None = None,
    top_k: int | None = None,
    advance_top: int | None = None,
    mktcap_min: float | None = None,
    mktcap_max: float | None = None,
) -> dict[str, Any]:
    with get_db() as conn:
        row = _load_run_row(conn, run_id)
    if not row:
        raise ValueError(f"discovery run_id not found: {run_id}")
    as_of_date = str(row.get("run_as_of_date") or "").strip()
    if not as_of_date:
        raise ValueError(f"discovery run_id has no run_as_of_date: {run_id}")
    return run_discovery(
        as_of_date=as_of_date,
        run_id=run_id,
        resume=True,
        phase=phase,
        prefilter_cap=prefilter_cap,
        prefilter_keep_ratio=prefilter_keep_ratio,
        workers=workers,
        top_k=top_k,
        advance_top=advance_top,
        mktcap_min=mktcap_min,
        mktcap_max=mktcap_max,
    )


def load_discovery_candidates(run_id: str) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM discovery_candidates
            WHERE run_id = ?
            ORDER BY discovery_score DESC, ticker ASC
            """,
            (run_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            out.append(json.loads(row["payload_json"]))
        except Exception:
            continue
    return out


def export_discovery_run(run_id: str, out_dir: Path) -> dict[str, str]:
    cfg = get_config()
    source_files = [
        cfg.discovery_dir / f"discovery_candidates_{run_id}.json",
        cfg.discovery_dir / f"discovery_shortlist_{run_id}.json",
        cfg.discovery_dir / f"discovery_universe_patch_{run_id}.csv",
        cfg.discovery_dir / f"discovery_report_{run_id}.md",
        cfg.discovery_dir / f"discovery_stats_{run_id}.json",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for src in source_files:
        if not src.exists():
            continue
        dst = out_dir / src.name
        dst.write_bytes(src.read_bytes())
        copied[src.name] = str(dst)
    return copied
