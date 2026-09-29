from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.dossier.whale_signals import run_whale_signals_for_run
from app.dossier.peer_report import build_peer_report_from_run
from app.dossier.runner import open_dossier_run, run_dossier_for_peer_set
from app.logging import get_logger
from app.research import run_research_for_scope
from app.sector.decision_pack import build_sector_decision_pack
from app.sector.peer_set import select_sector_peers
from app.sector.synthesis import run_sector_synthesis


logger = get_logger(__name__)
PEER_QUALITY_REASON_BUCKETS = [
    "NO_CIK",
    "OTC_EXCLUDED",
    "FOREIGN_EXCLUDED",
    "NO_ANNUAL_FORMS",
    "BELOW_MIN_ANNUAL",
    "BUDGET_SKIPPED",
]


def _default_run_id() -> str:
    return f"sector_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _suppressed_rows_from_peer_payload(peer_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in peer_payload.get("all_rows") or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        reasons = sorted({str(reason) for reason in (row.get("suppression_reasons") or []) if str(reason).strip()})
        if not ticker or not reasons:
            continue
        rows.append({"ticker": ticker, "reasons": reasons})
    rows.sort(key=lambda item: item["ticker"])
    return rows


def _append_suppressed_table(*, report_path: str | None, suppressed_rows: list[dict[str, Any]]) -> None:
    if not report_path:
        return
    path = Path(report_path)
    if not path.exists():
        return
    lines = ["", "## Suppressed Tickers", "| Ticker | Suppressor Reasons |", "|---|---|"]
    if suppressed_rows:
        for row in suppressed_rows:
            lines.append(f"| {row['ticker']} | {', '.join(row['reasons'])} |")
    else:
        lines.append("| NONE | None |")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _peer_quality_bucket_for_row(row: dict[str, Any]) -> str | None:
    excluded = str(row.get("excluded_reason") or "").upper().strip()
    if excluded == "MISSING_CIK":
        return "NO_CIK"
    if excluded == "OTC_EXCLUDED":
        return "OTC_EXCLUDED"
    if not excluded.startswith("PRECHECK_"):
        return None
    skip_reason = excluded.replace("PRECHECK_", "", 1)
    if skip_reason in {"CIK_MISSING", "COMPANY_NOT_FOUND"}:
        return "NO_CIK"
    if skip_reason == "FOREIGN_EXCLUDED":
        return "FOREIGN_EXCLUDED"
    if skip_reason == "BUDGET_EXCEEDED":
        return "BUDGET_SKIPPED"
    if skip_reason in {"NO_ANNUAL_FILING_IN_WINDOW", "NO_ANNUAL_FILING_AFTER_YEAR_BUCKETING"}:
        return "NO_ANNUAL_FORMS"
    if skip_reason in {"INSUFFICIENT_ANNUAL_FILINGS_IN_WINDOW"}:
        return "BELOW_MIN_ANNUAL"
    if "ANNUAL" in skip_reason:
        return "NO_ANNUAL_FORMS"
    return None


def _build_peer_quality_payload(
    *,
    run_id: str,
    sector: str,
    as_of_date: str,
    requested_dossierable_count: int,
    peer_payload: dict[str, Any],
    sec_budget_requested: int | None,
    sec_budget_effective: dict[str, int] | None,
) -> dict[str, Any]:
    selection_summary = peer_payload.get("peer_selection_summary") or {}
    stage_counts = selection_summary.get("stage_contribution_counts") or {}
    stage_counts_normalized = {
        "taxonomy": int(stage_counts.get("taxonomy", 0)),
        "sic_expand": int(stage_counts.get("sic_expand", 0)),
        "sic_family": int(stage_counts.get("sic_family", 0)),
        "filings_inferred": int(stage_counts.get("filings_inferred", 0)),
        "seed_fallback": int(stage_counts.get("seed_fallback", 0)),
    }

    final_rows: list[dict[str, Any]] = []
    for row in peer_payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        final_rows.append(
            {
                "ticker": str(row.get("ticker") or "").upper(),
                "stage_source": str(row.get("stage_source") or row.get("peer_source") or ""),
                "peer_source": str(row.get("peer_source") or ""),
                "cik_present": bool(row.get("cik_present", bool(row.get("cik")))),
                "annual_forms_found": row.get("annual_forms_found") or {},
                "eligible": bool(row.get("eligible", True)),
            }
        )
    if not final_rows:
        for ticker in peer_payload.get("selected_tickers") or []:
            final_rows.append(
                {
                    "ticker": str(ticker).upper(),
                    "stage_source": "unknown",
                    "peer_source": "unknown",
                    "cik_present": False,
                    "annual_forms_found": {},
                    "eligible": True,
                }
            )
    final_rows.sort(key=lambda item: item["ticker"])

    bucketed: dict[str, list[dict[str, Any]]] = {name: [] for name in PEER_QUALITY_REASON_BUCKETS}
    for row in peer_payload.get("all_rows") or []:
        if not isinstance(row, dict):
            continue
        if bool(row.get("selected")):
            continue
        bucket = _peer_quality_bucket_for_row(row)
        if not bucket:
            continue
        bucketed[bucket].append(
            {
                "ticker": str(row.get("ticker") or "").upper(),
                "stage_source": str(row.get("stage_source") or row.get("peer_source") or ""),
                "excluded_reason": str(row.get("excluded_reason") or ""),
                "annual_forms_found": row.get("annual_forms_found") or {},
                "cik_present": bool(row.get("cik_present", bool(row.get("cik")))),
                "eligible": bool(row.get("eligible", False)),
            }
        )
    for key in PEER_QUALITY_REASON_BUCKETS:
        bucketed[key] = sorted(
            bucketed[key],
            key=lambda item: (item["ticker"], item["stage_source"], item["excluded_reason"]),
        )

    achieved = len(final_rows)
    requested = max(1, int(requested_dossierable_count))
    shortfall = max(0, requested - achieved)
    return {
        "run_id": run_id,
        "sector": sector,
        "as_of_date": as_of_date,
        "requested_dossierable_count": requested,
        "achieved_dossierable_count": achieved,
        "dossierable_shortfall": shortfall,
        "selection_mode": str(selection_summary.get("mode") or ""),
        "peer_scan_count": int(selection_summary.get("peer_scan_count", 0)),
        "max_peer_scan": int(selection_summary.get("max_peer_scan", 0)),
        "scan_exhausted": bool(selection_summary.get("scan_exhausted", False)),
        "final_peer_list": final_rows,
        "filtered_reason_buckets": bucketed,
        "stage_contribution_counts": stage_counts_normalized,
        "fallback_steps_taken": selection_summary.get("fallback_steps_taken") or [],
        "sec_budget": {
            "requested": int(sec_budget_requested) if sec_budget_requested is not None else None,
            "effective": sec_budget_effective or {},
        },
    }


def _render_peer_quality_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Peer Quality Report")
    lines.append("")
    lines.append(
        f"- Requested dossierable peers: `{int(payload.get('requested_dossierable_count', 0))}`"
    )
    lines.append(
        f"- Achieved dossierable peers: `{int(payload.get('achieved_dossierable_count', 0))}`"
    )
    lines.append(f"- Shortfall: `{int(payload.get('dossierable_shortfall', 0))}`")
    lines.append(f"- Peer scan: `{int(payload.get('peer_scan_count', 0))}/{int(payload.get('max_peer_scan', 0))}`")
    lines.append(f"- Scan exhausted: `{bool(payload.get('scan_exhausted', False))}`")
    lines.append("")
    lines.append("## Final Peer List")
    lines.append("")
    lines.append("| Ticker | Stage Source | CIK Present | Eligible | Annual Forms Found |")
    lines.append("|---|---|---|---|---|")
    final_rows = payload.get("final_peer_list") or []
    if not final_rows:
        lines.append("| NONE | - | - | - | - |")
    else:
        for row in final_rows:
            forms = row.get("annual_forms_found") or {}
            forms_text = ", ".join(
                [f"{str(key).upper()}:{int(value)}" for key, value in sorted(forms.items())]
            ) or "NONE"
            lines.append(
                f"| {row.get('ticker')} | {row.get('stage_source')} | {row.get('cik_present')} | "
                f"{row.get('eligible')} | {forms_text} |"
            )
    lines.append("")
    lines.append("## Stage Contributions")
    lines.append("")
    for stage, value in (payload.get("stage_contribution_counts") or {}).items():
        lines.append(f"- `{stage}`: `{int(value)}`")
    lines.append("")
    lines.append("## Filtered Reason Buckets")
    lines.append("")
    buckets = payload.get("filtered_reason_buckets") or {}
    for bucket in PEER_QUALITY_REASON_BUCKETS:
        rows = buckets.get(bucket) or []
        lines.append(f"### {bucket}")
        lines.append("")
        lines.append(f"- Count: `{len(rows)}`")
        if rows:
            lines.append("| Ticker | Stage Source | Excluded Reason |")
            lines.append("|---|---|---|")
            for row in rows:
                lines.append(
                    f"| {row.get('ticker')} | {row.get('stage_source')} | {row.get('excluded_reason')} |"
                )
        lines.append("")
    sec_budget = payload.get("sec_budget") or {}
    lines.append("## SEC Budget Settings")
    lines.append("")
    lines.append(f"- Requested: `{sec_budget.get('requested')}`")
    lines.append(f"- Effective: `{sec_budget.get('effective')}`")
    lines.append("")
    return "\n".join(lines)


def _write_peer_quality_artifacts(
    *,
    run_id: str,
    sector: str,
    as_of_date: str,
    requested_dossierable_count: int,
    peer_payload: dict[str, Any],
    sector_dir: Path,
    sec_budget_requested: int | None,
    sec_budget_effective: dict[str, int] | None,
) -> dict[str, Any]:
    report_json_path = sector_dir / "peer_quality_report.json"
    report_md_path = sector_dir / "peer_quality_report.md"
    payload = _build_peer_quality_payload(
        run_id=run_id,
        sector=sector,
        as_of_date=as_of_date,
        requested_dossierable_count=requested_dossierable_count,
        peer_payload=peer_payload,
        sec_budget_requested=sec_budget_requested,
        sec_budget_effective=sec_budget_effective,
    )
    report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md_path.write_text(_render_peer_quality_markdown(payload), encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(report_json_path),
        "md_path": str(report_md_path),
    }


def run_sector_cycle(
    *,
    sector: str,
    as_of_date: str,
    peer_limit: int = 25,
    years_back: int = 10,
    min_annual_filings: int | None = 3,
    workers: int = 4,
    with_research: bool = True,
    with_synthesis: bool = True,
    run_id: str | None = None,
    mktcap_min: float | None = None,
    mktcap_max: float | None = None,
    peer_mode: str = "hybrid",
    min_peers_dossierable: int = 25,
    max_peers: int | None = None,
    max_peer_scan: int = 400,
    sic_expand: bool = True,
    sic_family: bool = True,
    include_foreign: bool = True,
    include_otc: bool = False,
    sec_budget: int | None = None,
    limit_dossiers: int = 25,
) -> dict[str, Any]:
    run_id = run_id or _default_run_id()
    cfg = get_config()
    sector_dir = cfg.sectors_dir / run_id
    sector_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sector_dir / "sector_summary.json"
    summary: dict[str, Any] = {
        "run_id": run_id,
        "sector": sector,
        "as_of_date": as_of_date,
        "status": "RUNNING",
        "peer_limit": int(peer_limit),
        "years_back": int(years_back),
        "workers": int(workers),
        "with_research": bool(with_research),
        "with_synthesis": bool(with_synthesis),
        "peer_mode": str(peer_mode),
        "min_peers_dossierable": int(min_peers_dossierable),
        "max_peers": int(max_peers if max_peers is not None else peer_limit),
        "max_peer_scan": int(max_peer_scan),
        "sic_expand": bool(sic_expand),
        "sic_family": bool(sic_family),
        "include_foreign": bool(include_foreign),
        "include_otc": bool(include_otc),
        "min_annual_filings": int(min_annual_filings) if min_annual_filings is not None else None,
        "sec_budget_requested": int(sec_budget) if sec_budget is not None else None,
        "limit_dossiers": int(limit_dossiers),
        "artifacts": {},
    }
    _write_summary(summary_path, summary)

    peer_payload = select_sector_peers(
        sector=sector,
        as_of_date=as_of_date,
        years_back=years_back,
        limit=peer_limit,
        min_peers=min_peers_dossierable,
        max_peers=max_peers if max_peers is not None else peer_limit,
        max_peer_scan=max_peer_scan,
        stop_when_min_reached=True,
        sic_expand=sic_expand,
        sic_family=sic_family,
        include_foreign=include_foreign,
        include_otc=include_otc,
        min_annual_filings=min_annual_filings,
        mktcap_min=mktcap_min,
        mktcap_max=mktcap_max,
        mode=peer_mode,
    )
    peers = peer_payload.get("selected_tickers") or []
    (sector_dir / "sector_peers.json").write_text(json.dumps(peer_payload, indent=2), encoding="utf-8")
    peer_selection_summary = peer_payload.get("peer_selection_summary") or {}
    suppressed_rows = _suppressed_rows_from_peer_payload(peer_payload)
    (sector_dir / "peer_selection_summary.json").write_text(
        json.dumps(peer_selection_summary, indent=2),
        encoding="utf-8",
    )
    summary["peer_tickers"] = peers
    summary["dossierable_count"] = len(peers)
    summary["peer_counts"] = peer_payload.get("counts", {})
    summary["peer_selection_summary"] = peer_selection_summary
    summary["suppressed_tickers"] = suppressed_rows
    summary["artifacts"]["sector_peers"] = str(sector_dir / "sector_peers.json")
    summary["artifacts"]["peer_selection_summary"] = str(sector_dir / "peer_selection_summary.json")
    peer_quality = _write_peer_quality_artifacts(
        run_id=run_id,
        sector=sector,
        as_of_date=as_of_date,
        requested_dossierable_count=min_peers_dossierable,
        peer_payload=peer_payload,
        sector_dir=sector_dir,
        sec_budget_requested=sec_budget,
        sec_budget_effective=None,
    )
    summary["peer_quality"] = peer_quality["payload"]
    summary["artifacts"]["peer_quality_report_json"] = peer_quality["json_path"]
    summary["artifacts"]["peer_quality_report_md"] = peer_quality["md_path"]
    _write_summary(summary_path, summary)

    if not peers:
        summary["status"] = "PARTIAL"
        summary["error"] = "No peers selected for sector"
        _write_summary(summary_path, summary)
        return summary

    dossier_cap = max(1, int(limit_dossiers))
    dossier_tickers = peers[:dossier_cap]
    summary["dossier_tickers"] = dossier_tickers
    summary["dossier_ticker_count"] = len(dossier_tickers)
    _write_summary(summary_path, summary)

    dossier_summary = run_dossier_for_peer_set(
        tickers=dossier_tickers,
        as_of_date=as_of_date,
        years_back=years_back,
        run_id=run_id,
        workers=workers,
        min_annual_filings=min_annual_filings,
        sec_budget=sec_budget,
    )
    summary["dossier"] = dossier_summary
    summary["sec_budget_effective"] = (dossier_summary.get("sec_budget") or {}).get("effective")
    peer_quality = _write_peer_quality_artifacts(
        run_id=run_id,
        sector=sector,
        as_of_date=as_of_date,
        requested_dossierable_count=min_peers_dossierable,
        peer_payload=peer_payload,
        sector_dir=sector_dir,
        sec_budget_requested=sec_budget,
        sec_budget_effective=summary["sec_budget_effective"],
    )
    summary["peer_quality"] = peer_quality["payload"]
    _write_summary(summary_path, summary)

    if with_research:
        research_count = run_research_for_scope(
            as_of_date=as_of_date,
            top_n=max(1, len(dossier_tickers)),
            use_active_universe=False,
            run_id=run_id,
            tickers=dossier_tickers,
            limit=len(dossier_tickers),
        )
        summary["research_count"] = int(research_count)
        _write_summary(summary_path, summary)

    whale_summary = run_whale_signals_for_run(run_id=run_id)
    summary["whale_signals"] = {
        "summary_path": whale_summary.get("summary_path"),
        "ticker_count": len(whale_summary.get("rows") or []),
    }
    copied_whale_summary = _copy_if_exists(
        Path(whale_summary["summary_path"]),
        sector_dir / "whale_signals_summary.json",
    ) if whale_summary.get("summary_path") else None
    summary["artifacts"]["whale_signals_summary"] = copied_whale_summary
    _write_summary(summary_path, summary)

    peer_report = build_peer_report_from_run(run_id=run_id, as_of_date=as_of_date)
    summary["peer_report"] = {
        "peer_report_path": peer_report.get("peer_report_path"),
        "peer_rankings_path": peer_report.get("peer_rankings_path"),
        "peer_scoreboard_path": peer_report.get("peer_scoreboard_path"),
        "ranked_count": len(peer_report.get("rankings") or []),
    }
    _write_summary(summary_path, summary)

    copied_peer_report = _copy_if_exists(
        Path(peer_report["peer_report_path"]),
        sector_dir / "peer_report.md",
    )
    copied_peer_rankings = _copy_if_exists(
        Path(peer_report["peer_rankings_path"]),
        sector_dir / "peer_rankings.json",
    )
    copied_peer_scoreboard = _copy_if_exists(
        Path(peer_report["peer_scoreboard_path"]),
        sector_dir / "peer_scoreboard.json",
    ) if peer_report.get("peer_scoreboard_path") else None
    _append_suppressed_table(report_path=copied_peer_report, suppressed_rows=suppressed_rows)
    summary["artifacts"]["peer_report"] = copied_peer_report
    summary["artifacts"]["peer_rankings"] = copied_peer_rankings
    summary["artifacts"]["peer_scoreboard"] = copied_peer_scoreboard
    summary["artifacts"]["dossier_summary"] = str(cfg.dossiers_dir / run_id / "dossier_summary.json")

    if with_synthesis:
        synth = run_sector_synthesis(sector=sector, as_of_date=as_of_date, run_id=run_id)
        summary["sector_synthesis"] = synth
        summary["artifacts"]["sector_synthesis"] = synth.get("path")
    else:
        summary["sector_synthesis"] = None
    _write_summary(summary_path, summary)

    try:
        decision_pack = build_sector_decision_pack(
            sector=sector,
            as_of_date=as_of_date,
            run_id=run_id,
            top_n=min(10, len(dossier_tickers)),
        )
        summary["decision_pack"] = decision_pack
        summary["artifacts"]["decision_pack"] = decision_pack.get("decision_pack_path")
        summary["artifacts"]["decision_pack_md"] = decision_pack.get("decision_pack_md_path")
    except Exception as exc:  # noqa: BLE001
        summary["decision_pack"] = None
        summary["decision_pack_error"] = str(exc)

    dossier_open = open_dossier_run(run_id)
    summary["dossier_status"] = dossier_open.get("status")
    if dossier_open.get("status") == "PARTIAL":
        summary["status"] = "PARTIAL"
    else:
        summary["status"] = "DONE"
    _write_summary(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def open_sector_run(*, run_id: str) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.sectors_dir / run_id
    if not run_dir.exists():
        raise ValueError(f"sector run not found: {run_id}")
    summary_path = run_dir / "sector_summary.json"
    if not summary_path.exists():
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "UNKNOWN",
            "artifacts": {},
        }
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["run_dir"] = str(run_dir)
    payload["summary_path"] = str(summary_path)
    return payload


def sector_run_status(*, run_id: str) -> dict[str, Any]:
    payload = open_sector_run(run_id=run_id)
    run_dir = Path(payload["run_dir"])
    expected = {
        "sector_summary": run_dir / "sector_summary.json",
        "sector_peers": run_dir / "sector_peers.json",
        "peer_selection_summary": run_dir / "peer_selection_summary.json",
        "peer_quality_report_json": run_dir / "peer_quality_report.json",
        "peer_quality_report_md": run_dir / "peer_quality_report.md",
        "peer_report_md": run_dir / "peer_report.md",
        "peer_rankings_json": run_dir / "peer_rankings.json",
        "peer_scoreboard_json": run_dir / "peer_scoreboard.json",
        "price_coverage_json": run_dir / "price_coverage.json",
        "shares_coverage_json": run_dir / "shares_coverage.json",
        "fcf_coverage_json": run_dir / "fcf_coverage.json",
        "facts_coverage_json": run_dir / "facts_coverage.json",
        "valuation_coverage_json": run_dir / "valuation_coverage.json",
        "value_gates_json": run_dir / "value_gates.json",
        "value_gates_calibration_json": run_dir / "value_gates_calibration.json",
        "whale_signals_summary_json": run_dir / "whale_signals_summary.json",
        "decision_pack_json": run_dir / "decision_pack.json",
        "decision_pack_md": run_dir / "decision_pack.md",
        "sector_synthesis_json": run_dir / "sector_synthesis.json",
    }
    present = [name for name, path in expected.items() if path.exists()]
    missing = [name for name, path in expected.items() if not path.exists()]
    ticker_results = (((payload.get("dossier") or {}).get("ticker_results")) or {})
    status_counts = {"OK": 0, "FAILED": 0, "SKIPPED": 0, "SKIPPED_BUDGET": 0, "PENDING": 0}
    for row in ticker_results.values():
        status = str((row or {}).get("status") or "PENDING").upper()
        if status not in status_counts:
            status = "PENDING"
        status_counts[status] += 1
    return {
        "run_id": run_id,
        "status": payload.get("status", "UNKNOWN"),
        "sector": payload.get("sector"),
        "as_of_date": payload.get("as_of_date"),
        "run_dir": str(run_dir),
        "artifact_count_present": len(present),
        "artifact_count_missing": len(missing),
        "artifacts_present": present,
        "artifacts_missing": missing,
        "peer_count": len(payload.get("peer_tickers") or []),
        "dossierable_count": int(payload.get("dossierable_count", len(payload.get("peer_tickers") or []))),
        "dossier_ticker_count": len(payload.get("dossier_tickers") or []),
        "dossier_status_counts": status_counts,
        "peer_quality_report_path": (payload.get("artifacts") or {}).get("peer_quality_report_json"),
    }


def sector_scoreboard_open(*, run_id: str) -> dict[str, Any]:
    payload = open_sector_run(run_id=run_id)
    run_dir = Path(payload["run_dir"])
    scoreboard_path = run_dir / "peer_scoreboard.json"
    price_coverage_path = run_dir / "price_coverage.json"
    shares_coverage_path = run_dir / "shares_coverage.json"
    fcf_coverage_path = run_dir / "fcf_coverage.json"
    facts_coverage_path = run_dir / "facts_coverage.json"
    valuation_coverage_path = run_dir / "valuation_coverage.json"
    price_coverage_payload: dict[str, Any] = {}
    shares_coverage_payload: dict[str, Any] = {}
    fcf_coverage_payload: dict[str, Any] = {}
    facts_coverage_payload: dict[str, Any] = {}
    valuation_coverage_payload: dict[str, Any] = {}
    if price_coverage_path.exists():
        try:
            price_coverage_payload = json.loads(price_coverage_path.read_text(encoding="utf-8"))
        except Exception:
            price_coverage_payload = {}
    if shares_coverage_path.exists():
        try:
            shares_coverage_payload = json.loads(shares_coverage_path.read_text(encoding="utf-8"))
        except Exception:
            shares_coverage_payload = {}
    if fcf_coverage_path.exists():
        try:
            fcf_coverage_payload = json.loads(fcf_coverage_path.read_text(encoding="utf-8"))
        except Exception:
            fcf_coverage_payload = {}
    if facts_coverage_path.exists():
        try:
            facts_coverage_payload = json.loads(facts_coverage_path.read_text(encoding="utf-8"))
        except Exception:
            facts_coverage_payload = {}
    if valuation_coverage_path.exists():
        try:
            valuation_coverage_payload = json.loads(valuation_coverage_path.read_text(encoding="utf-8"))
        except Exception:
            valuation_coverage_payload = {}
    price_known_count = 0
    price_unknown_count = 0
    price_unknown_reason_counts: dict[str, int] = {}
    valuation_entries = [row for row in (valuation_coverage_payload.get("entries") or []) if isinstance(row, dict)]
    if valuation_entries:
        for row in valuation_entries:
            price_status = str(row.get("price_status") or "UNKNOWN").upper()
            price_reason = str(row.get("price_reason_code") or "UNKNOWN")
            if price_status == "OK":
                price_known_count += 1
            else:
                price_unknown_count += 1
                price_unknown_reason_counts[price_reason] = price_unknown_reason_counts.get(price_reason, 0) + 1
    else:
        price_entries = [row for row in (price_coverage_payload.get("entries") or []) if isinstance(row, dict)]
        for row in price_entries:
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            status = str((result or {}).get("status") or "UNKNOWN").upper()
            reason = str((result or {}).get("reason_code") or "UNKNOWN")
            if status == "OK":
                price_known_count += 1
            else:
                price_unknown_count += 1
                price_unknown_reason_counts[reason] = price_unknown_reason_counts.get(reason, 0) + 1
    price_unknown_reason_counts = dict(sorted(price_unknown_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    if not scoreboard_path.exists():
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "peer_scoreboard_path": str(scoreboard_path),
            "price_coverage_path": str(price_coverage_path),
            "price_coverage_reason_counts": (price_coverage_payload.get("reason_counts") if isinstance(price_coverage_payload, dict) else {}),
            "shares_coverage_path": str(shares_coverage_path),
            "shares_coverage_reason_counts": (shares_coverage_payload.get("reason_counts") if isinstance(shares_coverage_payload, dict) else {}),
            "fcf_coverage_path": str(fcf_coverage_path),
            "fcf_coverage_reason_counts": (fcf_coverage_payload.get("reason_counts") if isinstance(fcf_coverage_payload, dict) else {}),
            "facts_coverage_path": str(facts_coverage_path),
            "facts_coverage_status_counts": (facts_coverage_payload.get("status_counts") if isinstance(facts_coverage_payload, dict) else {}),
            "valuation_coverage_path": str(valuation_coverage_path),
            "valuation_coverage_reason_counts": (valuation_coverage_payload.get("reason_counts") if isinstance(valuation_coverage_payload, dict) else {}),
            "price_known_count": int(price_known_count),
            "price_unknown_count": int(price_unknown_count),
            "price_unknown_reason_counts": price_unknown_reason_counts,
            "implied_return_known_count": 0,
            "implied_return_unknown_count": 0,
            "status": "MISSING",
            "ticker_count": 0,
            "metrics": [],
        }
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    rows = scoreboard.get("rows") or []
    metrics = sorted({
        str(metric)
        for row in rows
        for metric in (row.get("metric_values") or {}).keys()
        if str(metric).strip()
    })
    implied_known = 0
    implied_unknown = 0
    for row in rows:
        value = (row.get("metric_values") or {}).get("implied_return_base", "UNKNOWN")
        if isinstance(value, (int, float)):
            implied_known += 1
        else:
            implied_unknown += 1
    valuation_unknown_reason_counts: dict[str, int] = {}
    for row in valuation_entries:
        if str(row.get("valuation_status") or "UNKNOWN").upper() == "OK":
            continue
        code = str(row.get("valuation_reason_code") or "MODEL_PRECONDITION_FAILED")
        valuation_unknown_reason_counts[code] = valuation_unknown_reason_counts.get(code, 0) + 1
    valuation_unknown_reason_counts = dict(sorted(valuation_unknown_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "peer_scoreboard_path": str(scoreboard_path),
        "price_coverage_path": str(price_coverage_path),
        "price_coverage_reason_counts": (price_coverage_payload.get("reason_counts") if isinstance(price_coverage_payload, dict) else {}),
        "shares_coverage_path": str(shares_coverage_path),
        "shares_coverage_reason_counts": (shares_coverage_payload.get("reason_counts") if isinstance(shares_coverage_payload, dict) else {}),
        "fcf_coverage_path": str(fcf_coverage_path),
        "fcf_coverage_reason_counts": (fcf_coverage_payload.get("reason_counts") if isinstance(fcf_coverage_payload, dict) else {}),
        "facts_coverage_path": str(facts_coverage_path),
        "facts_coverage_status_counts": (facts_coverage_payload.get("status_counts") if isinstance(facts_coverage_payload, dict) else {}),
        "valuation_coverage_path": str(valuation_coverage_path),
        "valuation_coverage_reason_counts": (valuation_coverage_payload.get("reason_counts") if isinstance(valuation_coverage_payload, dict) else {}),
        "valuation_unknown_reason_counts": valuation_unknown_reason_counts,
        "price_known_count": int(price_known_count),
        "price_unknown_count": int(price_unknown_count),
        "price_unknown_reason_counts": price_unknown_reason_counts,
        "implied_return_known_count": int(implied_known),
        "implied_return_unknown_count": int(implied_unknown),
        "status": "OK",
        "ticker_count": len(rows),
        "metrics": metrics,
        "top_ticker": rows[0]["ticker"] if rows else None,
    }


def sector_scoreboard_compare(*, run_id: str, metric: str) -> dict[str, Any]:
    payload = sector_scoreboard_open(run_id=run_id)
    if payload.get("status") != "OK":
        raise ValueError(f"peer scoreboard missing for run_id={run_id}")
    path = Path(payload["peer_scoreboard_path"])
    scoreboard = json.loads(path.read_text(encoding="utf-8"))
    rows = scoreboard.get("rows") or []
    metric_name = str(metric or "").strip()
    if not metric_name:
        raise ValueError("metric is required")
    out: list[dict[str, Any]] = []
    for row in rows:
        value = (row.get("metric_values") or {}).get(metric_name, "UNKNOWN")
        derived_from = ((row.get("metric_traces") or {}).get(metric_name) or {}).get("derived_from") or []
        out.append(
            {
                "ticker": row.get("ticker"),
                "value": value,
                "derived_from": derived_from,
            }
        )
    known = [row for row in out if isinstance(row.get("value"), (int, float))]
    unknown = [row for row in out if not isinstance(row.get("value"), (int, float))]
    known.sort(key=lambda row: (-float(row["value"]), str(row["ticker"])))
    unknown.sort(key=lambda row: str(row["ticker"]))
    ordered = known + unknown
    for idx, row in enumerate(ordered, start=1):
        row["rank"] = idx if isinstance(row.get("value"), (int, float)) else None
    status = "OK"
    hint = None
    if ordered and not known:
        status = "ALL_UNKNOWN"
        hint = "Missing current_price; run with --with-prices and ensure price snapshots exist"
    return {
        "run_id": run_id,
        "metric": metric_name,
        "status": status,
        "known_count": len(known),
        "unknown_count": len(unknown),
        "hint": hint,
        "rows": ordered,
    }
