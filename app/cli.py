from __future__ import annotations

import json
import secrets
import os
import shlex
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import typer

from app.config import canonical_market_cap_focus, ensure_directories, get_config
from app.db import dedupe_evidence_items, get_db, init_db, upsert_state, utc_now_iso
from app.logging import configure_logging, get_logger
from app.util.dates import parse_yyyy_mm_dd


app = typer.Typer(help="IVI Analysis — SEC-filings equity research: fundamentals, valuation and AI analyst memos.")
watchlist_app = typer.Typer(help="Inspect the persistent investment watchlist")
app.add_typer(watchlist_app, name="watchlist")
classic_scan_app = typer.Typer(
    help="Plan, run, resume, and verify classic v1 coverage campaigns (each spend is authorized explicitly)"
)
app.add_typer(classic_scan_app, name="classic-scan")
from app.cli_investor import investor_app

app.add_typer(investor_app, name="investor")
from app.cli_backtest import backtest_app

app.add_typer(backtest_app, name="backtest")
from app.cli_universe import universe_app

app.add_typer(universe_app, name="universe")
from app.cli_events import events_app

app.add_typer(events_app, name="events")


from app.cli_ops import ops_app

app.add_typer(ops_app, name="ops")
from app.cli_holdings import holdings_app

app.add_typer(holdings_app, name="holdings")
logger = get_logger(__name__)


def _campaign_provider_model(config: Any, provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "anthropic":
        return str(config.anthropic_model or "")
    if normalized == "openai":
        return str(config.openai_model or "")
    if normalized == "deepseek":
        return str(config.deepseek_model or "")
    return ""


def _format_money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _format_pct_points(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def _distance_from_buy(current_price: float | None, buy_price_target: float | None) -> float | None:
    if current_price is None or buy_price_target is None or buy_price_target <= 0:
        return None
    return (current_price - buy_price_target) / buy_price_target


def _format_watchlist_trigger_results(results: list[Any]) -> str:
    lines = [
        "| Ticker | Prior Status | Latest Price | Buy Target | New Status | Transition |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    if not results:
        lines.append("| n/a | n/a | n/a | n/a | n/a | no eligible entries |")
    for result in results:
        transition = result.warning or result.transition
        lines.append(
            f"| {result.ticker} | {result.prior_status} | {_format_money(result.latest_price)} | "
            f"{_format_money(result.buy_price_target)} | {result.new_status} | {transition} |"
        )
    return "\n".join(lines)


def _counterfactual_artifact_path(source_run_id: str) -> Path:
    return (
        get_config().runs_dir / "autonomous_sector" / source_run_id / "autonomous_sector_run.json"
    )


def _counterfactual_default_output_path() -> Path:
    timestamp = "".join(ch if ch.isdigit() else "_" for ch in utc_now_iso()).strip("_")
    return (
        get_config().outputs_dir / "counterfactuals" / f"actionable_counterfactual_{timestamp}.md"
    )


def _find_counterfactual_ranking_row(
    artifact: dict[str, Any], ticker: str
) -> dict[str, Any] | None:
    ticker_norm = ticker.upper()
    for row in artifact.get("relative_ranking") or []:
        if str(row.get("ticker") or "").upper() == ticker_norm:
            return dict(row)
    return None


def _counterfactual_signal_rows(codes: list[str]) -> list[dict[str, str]]:
    from app.autonomous.sector_runtime import classify_audit_signal

    return [
        {"code": str(code), "classification": classify_audit_signal(str(code))}
        for code in codes
        if str(code).strip()
    ]


def _evaluate_actionable_counterfactual_row(
    *,
    row: sqlite3.Row,
    hurdle_pct: float,
    allow_evidence_caps: bool,
) -> dict[str, Any]:
    ticker = str(row["ticker"]).upper()
    current_verdict = str(row["conviction_grade"] or "WATCHLIST_ONLY").upper()
    artifact_path = _counterfactual_artifact_path(str(row["source_run_id"]))
    base_return: float | None = None
    downside_return: float | None = None
    hard_blockers: list[str] = []
    confidence_caps: list[str] = []
    audit_status: str | None = None
    data_status = "OK"
    if not artifact_path.exists():
        data_status = "ARTIFACT_MISSING"
    else:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        ranking = _find_counterfactual_ranking_row(artifact, ticker)
        if ranking is None:
            data_status = "RANKING_ROW_MISSING"
        else:
            audit_status = ranking.get("audit_status")
            base_raw = ranking.get("best_base_annualized_return")
            downside_raw = ranking.get("downside_annualized_return")
            base_return = float(base_raw) if isinstance(base_raw, (int, float)) else None
            downside_return = (
                float(downside_raw) if isinstance(downside_raw, (int, float)) else None
            )
            hard_blockers = [
                str(item) for item in ranking.get("hard_blockers") or [] if str(item).strip()
            ]
            confidence_caps = [
                str(item) for item in ranking.get("confidence_caps") or [] if str(item).strip()
            ]

    hurdle = float(hurdle_pct) / 100.0
    signal_rows = _counterfactual_signal_rows([*hard_blockers, *confidence_caps])
    business_quality = [
        item["code"] for item in signal_rows if item["classification"] == "BUSINESS_QUALITY"
    ]
    evidence_quality = [
        item["code"] for item in signal_rows if item["classification"] == "EVIDENCE_QUALITY"
    ]
    hurdle_signals = [item["code"] for item in signal_rows if item["classification"] == "HURDLE"]

    reasons: list[str] = []
    if data_status != "OK":
        reasons.append(data_status)
    if base_return is None:
        reasons.append("base return missing")
    elif base_return < hurdle:
        reasons.append(f"base return {_format_pct(base_return)} below {hurdle_pct:.1f}% hurdle")
    else:
        reasons.append(f"base return {_format_pct(base_return)} clears {hurdle_pct:.1f}% hurdle")
    if business_quality:
        reasons.append(
            "business-quality signals: " + ", ".join(sorted(dict.fromkeys(business_quality)))
        )
    if evidence_quality and allow_evidence_caps:
        reasons.append(
            "evidence-quality signals ignored: "
            + ", ".join(sorted(dict.fromkeys(evidence_quality)))
        )
    elif evidence_quality:
        reasons.append(
            "evidence-quality signals still binding: "
            + ", ".join(sorted(dict.fromkeys(evidence_quality)))
        )
    if hurdle_signals and base_return is not None and base_return >= hurdle:
        reasons.append("historical hurdle signals overridden by selected hurdle")

    eligible = (
        data_status == "OK"
        and base_return is not None
        and base_return >= hurdle
        and not business_quality
        and (allow_evidence_caps or not evidence_quality)
    )
    counterfactual_verdict = "ACTIONABLE" if eligible else "WATCHLIST_ONLY"
    changed = current_verdict != counterfactual_verdict
    return {
        "ticker": ticker,
        "watchlist_id": int(row["id"]),
        "source_sector": row["source_sector"],
        "source_run_id": row["source_run_id"],
        "status": row["status"],
        "current_verdict": current_verdict,
        "counterfactual_verdict": counterfactual_verdict,
        "changed": changed,
        "audit_status": audit_status,
        "base_return": base_return,
        "downside_return": downside_return,
        "hard_blockers": hard_blockers,
        "confidence_caps": confidence_caps,
        "signal_classifications": signal_rows,
        "business_quality_signals": sorted(dict.fromkeys(business_quality)),
        "evidence_quality_signals": sorted(dict.fromkeys(evidence_quality)),
        "hurdle_signals": sorted(dict.fromkeys(hurdle_signals)),
        "what_changed": "; ".join(reasons),
        "artifact_path": str(artifact_path),
    }


def _run_actionable_counterfactual(
    *,
    hurdle_pct: float,
    allow_evidence_caps: bool,
) -> dict[str, Any]:
    from app.watchlist.schema import ensure_watchlist_schema, resolve_db_path

    db_path = resolve_db_path(None)
    ensure_watchlist_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM watchlist
            WHERE status != 'REMOVED'
            ORDER BY source_sector, ticker, id
            """
        ).fetchall()
    finally:
        conn.close()
    results = [
        _evaluate_actionable_counterfactual_row(
            row=row,
            hurdle_pct=hurdle_pct,
            allow_evidence_caps=allow_evidence_caps,
        )
        for row in rows
    ]
    shifts = [
        row
        for row in results
        if row["current_verdict"] != "ACTIONABLE" and row["counterfactual_verdict"] == "ACTIONABLE"
    ]
    hurdle = float(hurdle_pct) / 100.0
    # Resolve-then-promote = DATA_INCOMPLETE-class names: the candidate clears the
    # base-return hurdle and carries ONLY evidence-quality (data-availability)
    # gaps, with no business-quality blocker. These are not AVOID — they unlock
    # once the named evidence is fetched.
    resolve_then_promote = [
        row
        for row in results
        if row["base_return"] is not None
        and row["base_return"] >= hurdle
        and not row["business_quality_signals"]
        and row["evidence_quality_signals"]
    ]
    return {
        "hurdle_pct": float(hurdle_pct),
        "allow_evidence_caps": bool(allow_evidence_caps),
        "total_entries": len(results),
        "shift_to_actionable_count": len(shifts),
        "resolve_then_promote_count": len(resolve_then_promote),
        "results": results,
        "shifted_tickers": [row["ticker"] for row in shifts],
        "resolve_then_promote_tickers": [row["ticker"] for row in resolve_then_promote],
    }


def _format_actionable_counterfactual_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Ticker | Status | Current Verdict | Counterfactual Verdict | Base Return | Downside Return | What Changed |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['ticker']} | {row['status']} | {row['current_verdict']} | "
            f"{row['counterfactual_verdict']} | {_format_pct(row['base_return'])} | "
            f"{_format_pct(row['downside_return'])} | {row['what_changed']} |"
        )
    return "\n".join(lines)


def _render_actionable_counterfactual_markdown(summary: dict[str, Any]) -> str:
    shifted = [row for row in summary["results"] if row["counterfactual_verdict"] == "ACTIONABLE"]
    lines = [
        "# Watchlist Actionability Counterfactual",
        "",
        f"- Hurdle: {summary['hurdle_pct']:.1f}%",
        f"- Allow evidence-quality signals: {summary['allow_evidence_caps']}",
        f"- Entries analyzed: {summary['total_entries']}",
        f"- Would shift to ACTIONABLE: {summary['shift_to_actionable_count']}",
        "",
        "## Would Shift To ACTIONABLE",
        "",
        "| Ticker | Sector | Status | Base Return | Downside Return | Evidence-Quality Signals Ignored |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    if not shifted:
        lines.append("| n/a | n/a | n/a | n/a | n/a | none |")
    for row in shifted:
        lines.append(
            f"| {row['ticker']} | {row['source_sector'] or 'n/a'} | {row['status']} | "
            f"{_format_pct(row['base_return'])} | {_format_pct(row['downside_return'])} | "
            f"{', '.join(f'`{code}`' for code in row['evidence_quality_signals']) or 'none'} |"
        )
    lines.extend(
        ["", "## Full Counterfactual Table", "", _format_actionable_counterfactual_table(summary)]
    )
    return "\n".join(lines) + "\n"


def _write_actionable_counterfactual(
    summary: dict[str, Any],
    output_path: str | None = None,
) -> Path:
    path = Path(output_path) if output_path is not None else _counterfactual_default_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_actionable_counterfactual_markdown(summary), encoding="utf-8")
    return path


def _fetch_watchlist_manual_add_price(ticker: str) -> float | None:
    from app.market.price_provider import build_price_provider

    cfg = get_config()
    provider = build_price_provider(
        cfg=cfg,
        with_prices=True,
        fallback_days=getattr(cfg, "price_fallback_days", None),
    )
    snapshot = provider.get_price_asof(ticker.upper(), date.today().isoformat())
    if snapshot is None or snapshot.price <= 0:
        return None
    return float(snapshot.price)


def _parse_forms(forms: str) -> list[str]:
    return [x.strip().upper() for x in forms.split(",") if x.strip()]


def _parse_tickers_csv(tickers: str | None) -> list[str]:
    if not tickers:
        return []
    return [x.strip().upper() for x in tickers.split(",") if x.strip()]


def _validate_max_candidates_option(max_candidates: int | None) -> int | None:
    if max_candidates is not None and int(max_candidates) <= 0:
        raise typer.BadParameter("--max-candidates must be a positive integer when provided")
    return int(max_candidates) if max_candidates is not None else None


def _build_synthesis_evidence_packet(ticker: str, as_of: str) -> None:
    from app.evidence.packet_builder import build_packet_for_ticker

    build_packet_for_ticker(ticker, as_of)


def _parse_sources_csv(sources: str | None) -> set[str]:
    if not sources:
        return set()
    return {x.strip().lower() for x in sources.split(",") if x.strip()}


def _parse_price_changes_csv(values: str | None) -> dict[str, float]:
    if not values:
        return {}
    out: dict[str, float] = {}
    for item in values.split(","):
        token = item.strip()
        if not token or "=" not in token:
            continue
        ticker, raw_value = token.split("=", 1)
        ticker_norm = ticker.strip().upper()
        if not ticker_norm:
            continue
        try:
            out[ticker_norm] = float(raw_value.strip())
        except ValueError:
            continue
    return out


def _director_agenda_summary(agenda: Any) -> list[dict[str, Any]]:
    allocations = agenda.allocations if hasattr(agenda, "allocations") else []
    return [
        {
            "sector_id": str(row.sector_id),
            "priority_rank": int(row.priority_rank),
            "budget_allocation": int(row.budget_allocation),
            "recommended_depth": str(row.recommended_depth),
            "reason_narrative": str(row.reason_narrative),
        }
        for row in allocations
    ]


def _default_director_campaign_run_id(generated_at: str) -> str:
    return f"director_{str(generated_at)[:10].replace('-', '')}_{secrets.token_hex(2)}"


def _campaign_state_path_for_run(campaign_run_id: str) -> Path:
    return get_config().campaigns_dir / campaign_run_id / "campaign_state.json"


def _launch_universe_campaign_from_agenda(*, agenda_path: Path, campaign_run_id: str) -> str:
    cfg = get_config()
    python_exec = sys.executable or "python3"
    cmd = [
        python_exec,
        "-m",
        "app.cli",
        "universe-campaign-from-agenda",
        "--agenda-path",
        str(agenda_path),
        "--campaign-run-id",
        campaign_run_id,
        "--foreground",
    ]
    subprocess.Popen(
        cmd,
        cwd=str(cfg.project_root),
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return shlex.join(cmd)


def _resolve_scope(parsed_tickers: list[str], limit: int | None = None) -> list[str]:
    from app.universe.universe import get_active_universe_id, list_universe_members

    with get_db() as conn:
        if parsed_tickers:
            scope = parsed_tickers
        else:
            universe_id = get_active_universe_id(conn)
            if universe_id:
                scope = [row["ticker"] for row in list_universe_members(conn, universe_id)]
            else:
                rows = conn.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()
                scope = [row["ticker"] for row in rows]
    if limit is not None and limit > 0:
        scope = scope[:limit]
    return scope


def _run_all_impl(
    *,
    as_of: str | None,
    with_research: bool,
    with_synthesis: bool,
    with_discovery: bool,
    dossier_top: int | None,
    limit: int | None,
    tickers: str | None,
    discovery_top: int,
    discovery_seed: Path | None,
    phase: list[str] | None,
    forms: str,
    memo_mode: str,
    top: int,
) -> dict[str, object]:
    from app.agent.analyst_agent import run_analyst_agent_for_ticker, run_red_team_for_ticker
    from app.agent.queue import dead_letter_count
    from app.agent.scheduler import write_agent_status
    from app.evidence.packet_builder import build_packet_for_ticker
    from app.fundamentals.metrics import compute_fundamentals_for_ticker
    from app.ingest.filings import ingest_with_policy
    from app.ops.runs import finalize_run_outputs, generate_run_id
    from app.parse.filing_parser import parse_pending_filings
    from app.report.memo_builder import build_memo_for_ticker, build_top_memos
    from app.report.run_manifest import write_run_manifest
    from app.research.engine import run_research_agent_for_ticker
    from app.llm.synthesis_agent import run_synthesis_for_scope
    from app.score.ranker import score_and_rank, score_ticker
    from app.valuation.sanity_checks import run_valuation_for_ticker

    logger.info("stage_init_db_start", extra={"stage_name": "init-db"})
    init_db_cmd()
    memo_mode = memo_mode.strip().lower()
    if memo_mode not in {"strict", "triage"}:
        raise ValueError("memo_mode must be one of: strict, triage")
    top = max(1, int(top))

    def _mark_stage(stage: str) -> None:
        with get_db() as conn:
            upsert_state(conn, f"stage_last_success:{stage}", {"ts": utc_now_iso()})

    as_of_date = parse_yyyy_mm_dd(as_of).isoformat() if as_of else date.today().isoformat()
    parsed_tickers = _parse_tickers_csv(tickers)
    scope = _resolve_scope(parsed_tickers, limit=limit)
    discovery_summary: dict[str, object] | None = None
    dossier_summary: dict[str, object] | None = None

    if with_discovery:
        from app.discovery.runner import run_discovery

        with_research = True
        with_synthesis = True
        discovery_summary = run_discovery(
            as_of_date=as_of_date,
            limit=limit,
            tickers=parsed_tickers or None,
            seed_path=discovery_seed,
            top_k=max(1, int(discovery_top)),
        )
        scope = list(discovery_summary.get("shortlist_tickers") or [])[: max(1, int(discovery_top))]
        if not scope:
            scope = _resolve_scope(parsed_tickers, limit=limit)
        limit = len(scope)
    elif dossier_top and dossier_top > 0:
        raise ValueError("--dossier-top requires --with-discovery for run-all handoff")

    normalized_phases = [p.strip().lower() for p in (phase or []) if p.strip()]
    if not normalized_phases:
        normalized_phases = ["ingest", "valuation"]
        if with_research:
            normalized_phases.append("research")
    invalid = [p for p in normalized_phases if p not in {"ingest", "valuation", "research"}]
    if invalid:
        raise ValueError(f"Invalid phase(s): {','.join(invalid)}")

    if "research" in normalized_phases:
        with_research = True

    run_id = generate_run_id("run")
    dead_letters_before = dead_letter_count()

    if "ingest" in normalized_phases:
        ingest_with_policy(
            as_of_date=as_of_date,
            run_id=run_id,
            tickers=scope,
            limit=limit,
        )
        parse_pending_filings(limit=max(100, max(1, len(scope)) * 10), tickers=scope)
        _mark_stage("ingest")
        _mark_stage("parse")

    if "valuation" in normalized_phases:
        for ticker_symbol in scope:
            compute_fundamentals_for_ticker(ticker_symbol, as_of_date=as_of_date)
            run_valuation_for_ticker(ticker_symbol, run_as_of_date=as_of_date)
            build_packet_for_ticker(ticker_symbol)
            run_analyst_agent_for_ticker(ticker_symbol)
            run_red_team_for_ticker(ticker_symbol)
            score_ticker(ticker_symbol, run_id=run_id)
        _mark_stage("compute_fundamentals")
        _mark_stage("run_valuation")
        _mark_stage("build_evidence_packet")
        _mark_stage("run_analyst_agent")
        _mark_stage("run_red_team")
        _mark_stage("score_ticker")

    if with_research and "research" in normalized_phases:
        for ticker_symbol in scope:
            run_research_agent_for_ticker(
                ticker_symbol,
                as_of_date=as_of_date,
                run_id=run_id,
            )
            score_ticker(ticker_symbol, run_id=run_id)
        _mark_stage("run_research_agent")

    synthesis_summary: dict[str, Any] = {"built": 0, "attempted": 0}
    score_summary: dict[str, object] = {
        "score_rows_considered": 0,
        "tickers_ranked": 0,
        "candidate_count": 0,
        "publishable_count": 0,
    }
    if any(p in normalized_phases for p in ["valuation", "research"]):
        score_summary = score_and_rank(
            top_n=top,
            memo_mode=memo_mode,
            run_id=run_id,
            candidate_scope=scope,
            as_of_date=as_of_date,
            rescore=False,
        )
        if with_synthesis:
            with get_db() as conn:
                candidates = conn.execute(
                    """
                    SELECT ticker
                    FROM scores
                    WHERE run_id = ? AND is_candidate = 1 AND candidate_run_id = ?
                    ORDER BY total_score DESC, ticker ASC
                    LIMIT ?
                    """,
                    (run_id, run_id, top),
                ).fetchall()
                synthesis_scope = [row["ticker"] for row in candidates]
                if not synthesis_scope:
                    rows = conn.execute(
                        """
                        SELECT ticker
                        FROM scores
                        WHERE run_id = ?
                        ORDER BY total_score DESC, ticker ASC
                        LIMIT ?
                        """,
                        (run_id, top),
                    ).fetchall()
                    synthesis_scope = [row["ticker"] for row in rows]
            synthesis_summary = run_synthesis_for_scope(
                as_of_date=as_of_date,
                run_id=run_id,
                tickers=synthesis_scope,
                limit=top,
            )
            _mark_stage("run_synthesis_agent")
        if memo_mode == "triage":
            build_top_memos(top_n=top, memo_mode="triage", run_id=run_id)
        else:
            for ticker_symbol in scope:
                build_memo_for_ticker(ticker_symbol, memo_mode="strict", run_id=run_id)
        _mark_stage("build_memo")
        _mark_stage("build_rankings")

    report_scope = scope
    if memo_mode == "triage":
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT ticker
                FROM scores
                WHERE is_candidate = 1 AND candidate_run_id = ?
                ORDER BY total_score DESC, ticker ASC
                LIMIT ?
                """,
                (run_id, top),
            ).fetchall()
        triage_scope = [row["ticker"] for row in rows]
        if triage_scope:
            report_scope = triage_scope

    write_agent_status()
    manifest_path = write_run_manifest(as_of_date=as_of_date, run_id=run_id)
    summary = finalize_run_outputs(
        run_id=run_id,
        as_of_date=as_of_date,
        tickers_targeted=report_scope,
        with_research=with_research and "research" in normalized_phases,
        dead_letter_before=dead_letters_before,
        manifest_path=manifest_path,
    )
    summary["as_of_date"] = as_of_date
    summary["phases"] = normalized_phases
    summary["memo_mode"] = memo_mode
    summary["top"] = top
    summary["score_rows_considered"] = score_summary.get("score_rows_considered", 0)
    summary["tickers_ranked"] = score_summary.get("tickers_ranked", 0)
    summary["candidate_count"] = score_summary.get("candidate_count", 0)
    summary["publishable_count"] = score_summary.get("publishable_count", 0)
    summary["synthesis_count"] = synthesis_summary.get("built", 0)
    if discovery_summary is not None:
        summary["discovery"] = discovery_summary
        summary["discovery_run_id"] = discovery_summary.get("run_id")
        summary["discovery_shortlist_count"] = len(discovery_summary.get("shortlist_tickers") or [])
    if with_discovery and dossier_top and dossier_top > 0 and discovery_summary is not None:
        from app.dossier.runner import run_dossier_for_peer_set

        dossier_tickers = list(discovery_summary.get("shortlist_tickers") or [])[
            : max(1, int(dossier_top))
        ]
        if dossier_tickers:
            dossier_summary = run_dossier_for_peer_set(
                tickers=dossier_tickers,
                as_of_date=as_of_date,
                years_back=10,
                run_id=run_id,
                workers=min(4, max(1, len(dossier_tickers))),
            )
    if dossier_summary is not None:
        summary["dossier"] = dossier_summary
    return summary


@app.command("init-db")
def init_db_cmd() -> None:
    configure_logging()
    cfg = get_config()
    ensure_directories(cfg)
    init_db(cfg)
    typer.echo(f"Initialized DB at {cfg.db_path}")


@app.command("web")
def web_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (localhost only)."),
    port: int = typer.Option(8321, "--port", help="Port to serve the web UI on."),
    reload: bool = typer.Option(False, "--reload/--no-reload", help="Auto-reload for development."),
) -> None:
    """Serve the IVI web UI (read-only) on localhost."""
    configure_logging()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        typer.echo(
            "ivi web binds localhost only: exposure beyond 127.0.0.1 is an owner "
            "decision that is off by default (see the web UI plan §8).",
            err=True,
        )
        raise typer.Exit(code=2)
    import uvicorn

    typer.echo(f"IVI: http://{host}:{port}  (legacy console at /legacy)")
    uvicorn.run("app.web.main:app", host=host, port=port, reload=reload)


@app.command("evidence-dedupe")
def evidence_dedupe_cmd() -> None:
    configure_logging()
    try:
        with get_db() as conn:
            summary = dedupe_evidence_items(conn)
    except sqlite3.OperationalError as exc:
        if "no such table: evidence_items" not in str(exc).lower():
            raise
        init_db(get_config())
        with get_db() as conn:
            summary = dedupe_evidence_items(conn)
    typer.echo(json.dumps(summary, indent=2))


def _load_universe_impl(csv: Path, *, set_active: bool = True) -> None:
    from app.universe.ticker_cik_map import load_ticker_cik_map
    from app.universe.universe import fill_missing_ciks, load_universe_to_db, read_universe_csv

    init_db(get_config())
    rows = read_universe_csv(csv)
    rows = fill_missing_ciks(rows, load_ticker_cik_map())
    with get_db() as conn:
        universe_id, snapshot_hash, count = load_universe_to_db(
            conn, rows, csv, set_active=set_active
        )
    typer.echo(f"Loaded universe {universe_id} ({count} tickers) snapshot_hash={snapshot_hash}")


@app.command("universe-load")
def universe_load(csv: Path | None = typer.Option(None, exists=True)) -> None:
    configure_logging()
    cfg = get_config()
    csv = csv or cfg.universe_path
    _load_universe_impl(csv)


@app.command("universe-list")
def universe_list(
    limit: int = typer.Option(10), members: bool = typer.Option(False, "--members")
) -> None:
    configure_logging()
    from app.universe.universe import (
        get_active_universe_id,
        list_universe_members,
        list_universe_snapshots,
    )

    init_db(get_config())
    with get_db() as conn:
        snapshots = list_universe_snapshots(conn, limit=limit)
        active_id = get_active_universe_id(conn)

    typer.echo(json.dumps({"active_universe_id": active_id, "snapshots": snapshots}, indent=2))
    if members and active_id:
        with get_db() as conn:
            active_members = list_universe_members(conn, active_id)
        typer.echo(json.dumps({"universe_id": active_id, "members": active_members}, indent=2))


@app.command("add-universe")
def add_universe(csv: Path = typer.Option(..., exists=True)) -> None:
    configure_logging()
    _load_universe_impl(csv)


@app.command("universe-merge-load")
def universe_merge_load(
    base: Path = typer.Option(..., "--base", exists=True, help="Base universe CSV"),
    overrides: Path = typer.Option(..., "--overrides", exists=True, help="Metadata overrides CSV"),
    set_active: bool = typer.Option(
        False, "--set-active", help="Set merged universe as active snapshot"
    ),
) -> None:
    configure_logging()
    from app.universe.universe import (
        load_universe_to_db,
        merge_universe_rows,
        read_base_universe_csv,
        read_metadata_overrides_csv,
    )

    init_db(get_config())
    base_rows = read_base_universe_csv(base)
    override_map = read_metadata_overrides_csv(overrides)
    merged = merge_universe_rows(base_rows, override_map)
    source_label = Path(f"{base.resolve()}+{overrides.resolve()}")
    with get_db() as conn:
        universe_id, snapshot_hash, count = load_universe_to_db(
            conn,
            merged,
            source_label,
            set_active=set_active,
        )
    typer.echo(
        f"Merged universe {universe_id} ({count} tickers) snapshot_hash={snapshot_hash} set_active={set_active}"
    )


@app.command("universe-export")
def universe_export(
    universe_id: str = typer.Option(..., "--universe-id", help="Universe snapshot identifier"),
    out: Path = typer.Option(..., "--out", help="Output CSV path"),
) -> None:
    configure_logging()
    from app.universe.universe import export_universe_snapshot

    init_db(get_config())
    with get_db() as conn:
        path = export_universe_snapshot(conn, universe_id, out)
    typer.echo(str(path))


@app.command("universe-scout")
def universe_scout_cmd(
    as_of: str | None = typer.Option(
        None, "--as-of", help="As-of date YYYY-MM-DD (defaults to today)"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic scout run id"),
    tickers: str | None = typer.Option(None, "--tickers", help="Optional ticker CSV universe"),
    sector_run_id: str | None = typer.Option(
        None,
        "--sector-run-id",
        help="Optional prior sector/depth run id to source peer tickers",
    ),
    universe_csv: Path | None = typer.Option(
        None,
        "--universe-csv",
        exists=True,
        help="Optional universe CSV source with ticker column",
    ),
    universe_file: Path | None = typer.Option(
        None,
        "--universe-file",
        exists=True,
        help="Universe CSV source (ticker column required)",
    ),
    universe_source: str | None = typer.Option(
        None,
        "--universe-source",
        help="Universe source label (nasdaq|nyse|amex|custom)",
    ),
    universe_limit: int | None = typer.Option(
        None,
        "--universe-limit",
        help="Optional cap after normalization for large-universe smoke runs",
    ),
    top_n: int = typer.Option(50, "--top-n", help="Top N shortlist rows to emit"),
    with_prices: bool = typer.Option(
        True, "--with-prices/--no-with-prices", help="Enable provider-based price snapshots"
    ),
    batch_size: int = typer.Option(200, "--batch-size", help="Deterministic scout batch size"),
    max_batches: int | None = typer.Option(
        None, "--max-batches", help="Optional max batches to process per invocation"
    ),
    scout_sec_budget: int | None = typer.Option(
        None, "--scout-sec-budget", help="SEC request budget for this invocation"
    ),
    scout_net_budget: int | None = typer.Option(
        None, "--scout-net-budget", help="Non-SEC network budget for this invocation"
    ),
    scout_max_seconds: int | None = typer.Option(
        None, "--scout-max-seconds", help="Wall-clock budget for this invocation"
    ),
    force_restart: bool = typer.Option(
        False, "--force-restart", help="Restart scout run from scratch"
    ),
    scout_mos_min: float | None = typer.Option(
        None, "--scout-mos-min", help="Override scout MOS threshold"
    ),
    scout_valuation_gap_min: float | None = typer.Option(
        None,
        "--scout-valuation-gap-min",
        help="Override scout valuation gap minimum threshold",
    ),
    scout_fcf_yield_min: float | None = typer.Option(
        None,
        "--scout-fcf-yield-min",
        help="Override scout FCF yield minimum threshold",
    ),
    scout_net_debt_to_cfo_max: float | None = typer.Option(
        None,
        "--scout-net-debt-to-cfo-max",
        help="Override scout net-debt-to-CFO maximum threshold",
    ),
    scout_dilution_max: float | None = typer.Option(
        None,
        "--scout-dilution-max",
        help="Override scout dilution maximum threshold",
    ),
    gd_discount_rate: float | None = typer.Option(
        None,
        "--gd-discount-rate",
        help="Override Graham/Dodd EPV discount rate",
    ),
    scout_use_graham_dodd: bool | None = typer.Option(
        None,
        "--scout-use-graham-dodd/--no-scout-use-graham-dodd",
        help="Enable Graham/Dodd MOS overlay in scout gating",
    ),
    scout_require_ev_yield: bool | None = typer.Option(
        None,
        "--scout-require-ev-yield/--no-scout-require-ev-yield",
        help="Require EV-based yield denominator (no market-cap fallback)",
    ),
) -> None:
    configure_logging()
    from app.universe.scout import run_universe_scout

    as_of_date = parse_yyyy_mm_dd(as_of).isoformat() if as_of else date.today().isoformat()
    cfg = get_config()
    init_db(cfg)
    ticker_list = _parse_tickers_csv(tickers) if tickers else []
    selected_universe = universe_file or universe_csv
    if not ticker_list and not sector_run_id and selected_universe is None:
        if cfg.universe_path.exists():
            selected_universe = cfg.universe_path
        else:
            raise typer.BadParameter(
                "Provide at least one source: --tickers, --sector-run-id, --universe-file, or --universe-csv"
            )
    threshold_overrides = {
        "scout_mos_min": scout_mos_min,
        "scout_valuation_gap_min": scout_valuation_gap_min,
        "scout_fcf_yield_min": scout_fcf_yield_min,
        "scout_net_debt_to_cfo_max": scout_net_debt_to_cfo_max,
        "scout_dilution_max": scout_dilution_max,
        "gd_discount_rate": gd_discount_rate,
        "scout_use_graham_dodd": scout_use_graham_dodd,
        "scout_require_ev_yield": scout_require_ev_yield,
    }
    try:
        payload = run_universe_scout(
            run_id=run_id,
            as_of_date=as_of_date,
            top_n=max(1, int(top_n)),
            tickers=ticker_list,
            sector_run_id=sector_run_id,
            universe_csv=selected_universe,
            universe_source=universe_source,
            universe_limit=universe_limit,
            with_prices=bool(with_prices),
            threshold_overrides=threshold_overrides,
            batch_size=max(1, int(batch_size)),
            max_batches=max_batches,
            scout_sec_budget=scout_sec_budget,
            scout_net_budget=scout_net_budget,
            scout_max_seconds=scout_max_seconds,
            force_restart=bool(force_restart),
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command("universe-scout-open")
def universe_scout_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
) -> None:
    configure_logging()
    from app.universe.scout import open_universe_scout

    payload = open_universe_scout(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-scout-calibration-open")
def universe_scout_calibration_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
) -> None:
    configure_logging()
    from app.universe.scout import open_universe_scout_calibration

    payload = open_universe_scout_calibration(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-yield-coverage-open")
def universe_yield_coverage_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
) -> None:
    configure_logging()
    from app.universe.scout import open_universe_yield_coverage

    payload = open_universe_yield_coverage(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-owner-earnings-quality-open")
def universe_owner_earnings_quality_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.owner_earnings_quality import open_owner_earnings_quality

    payload = open_owner_earnings_quality(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-intangible-economics-open")
def universe_intangible_economics_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.intangible_economics import open_intangible_economics

    payload = open_intangible_economics(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-intrinsic-discipline-open")
def universe_intrinsic_discipline_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.intrinsic_discipline import open_intrinsic_discipline

    payload = open_intrinsic_discipline(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-evidence-sufficiency-open")
def universe_evidence_sufficiency_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.evidence_sufficiency import open_evidence_sufficiency

    payload = open_evidence_sufficiency(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-cyclical-normalization-open")
def universe_cyclical_normalization_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.cyclical_normalization import open_cyclical_normalization

    payload = open_cyclical_normalization(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-capital-allocation-discipline-open")
def universe_capital_allocation_discipline_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.capital_allocation_discipline import open_capital_allocation_discipline

    payload = open_capital_allocation_discipline(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-normalization-credibility-open")
def universe_normalization_credibility_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.normalization_credibility import open_normalization_credibility

    payload = open_normalization_credibility(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-impairment-classification-open")
def universe_impairment_classification_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.impairment_classification import open_impairment_classification

    payload = open_impairment_classification(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-valuation-confidence-open")
def universe_valuation_confidence_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.valuation_confidence import open_valuation_confidence

    payload = open_valuation_confidence(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-value-type-open")
def universe_value_type_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.value_type import open_value_type

    payload = open_value_type(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-valuation-integrity-open")
def universe_valuation_integrity_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.valuation_integrity import open_valuation_integrity

    payload = open_valuation_integrity(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-fundamental-regression-open")
def universe_fundamental_regression_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.fundamental_regression_analytics import open_fundamental_regression_analytics

    payload = open_fundamental_regression_analytics(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-investment-readiness-open")
def universe_investment_readiness_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.investment_readiness import open_investment_readiness

    payload = open_investment_readiness(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-reinvestment-efficiency-open")
def universe_reinvestment_efficiency_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.reinvestment_efficiency import open_reinvestment_efficiency

    payload = open_reinvestment_efficiency(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-maintenance-capex-open")
def universe_maintenance_capex_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.maintenance_capex_discipline import open_maintenance_capex_discipline

    payload = open_maintenance_capex_discipline(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-accounting-quality-open")
def universe_accounting_quality_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.accounting_quality import open_accounting_quality

    payload = open_accounting_quality(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-balance-sheet-stress-open")
def universe_balance_sheet_stress_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.balance_sheet_stress import open_balance_sheet_stress

    payload = open_balance_sheet_stress(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-returns-persistence-open")
def universe_returns_persistence_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.returns_persistence import open_returns_persistence

    payload = open_returns_persistence(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-revenue-dependence-open")
def universe_revenue_dependence_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout or sector run id"),
) -> None:
    configure_logging()
    from app.valuation.revenue_dependence import open_revenue_dependence

    payload = open_revenue_dependence(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("graham-dodd-open")
def graham_dodd_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
) -> None:
    configure_logging()
    from app.universe.scout import open_graham_dodd

    payload = open_graham_dodd(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-rankings-open")
def universe_rankings_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
) -> None:
    configure_logging()
    from app.universe.scout import open_universe_rankings

    payload = open_universe_rankings(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-depth-queue-open")
def universe_depth_queue_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
) -> None:
    configure_logging()
    from app.universe.scout import open_universe_depth_queue

    payload = open_universe_depth_queue(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-depth-queue-to-runs")
def universe_depth_queue_to_runs_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
    max_runs: int | None = typer.Option(
        None, "--max-runs", help="Optional cap on generated depth commands"
    ),
) -> None:
    configure_logging()
    from app.universe.scout import universe_depth_queue_to_runs

    payload = universe_depth_queue_to_runs(run_id=run_id, max_runs=max_runs)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-depth-batch")
def universe_depth_batch_cmd(
    universe_run_id: str = typer.Option(..., "--universe-run-id", help="Universe scout run id"),
    batch_run_id: str | None = typer.Option(None, "--batch-run-id", help="Depth batch run id"),
    max_runs: int | None = typer.Option(
        None, "--max-runs", help="Optional cap on queued depth runs"
    ),
    selection_policy: str = typer.Option(
        "QUEUE_ORDER", "--selection-policy", help="Batch selection policy"
    ),
    mode: str = typer.Option("depth", "--mode", help="RLM mode for queued runs (default: depth)"),
    iterations: int | None = typer.Option(
        None, "--iterations", help="Optional override iteration cap"
    ),
    top_k: int | None = typer.Option(None, "--top-k", help="Optional override top-k"),
    workers: int | None = typer.Option(None, "--workers", help="Optional override worker count"),
    with_prices: bool = typer.Option(
        True, "--with-prices/--no-with-prices", help="Pass-through depth price hydration"
    ),
    sec_budget: int | None = typer.Option(
        None, "--sec-budget", help="Reserved SEC budget annotation for batch run"
    ),
    llm_budget: float | None = typer.Option(
        None, "--llm-budget", help="Pass-through LLM budget (USD)"
    ),
    llm_provider: str | None = typer.Option(
        None, "--llm-provider", help="Optional VOE_LLM_PROVIDER override"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only; do not execute depth runs"),
) -> None:
    configure_logging()
    from app.universe.batch_runner import run_universe_depth_batch

    init_db(get_config())
    mode_norm = str(mode or "depth").strip().lower()
    if mode_norm != "depth":
        raise typer.BadParameter("--mode currently supports only depth")
    try:
        payload = run_universe_depth_batch(
            universe_run_id=universe_run_id,
            batch_run_id=batch_run_id,
            max_runs=max_runs,
            selection_policy=selection_policy,
            dry_run=bool(dry_run),
            mode=mode_norm,
            iterations=iterations,
            top_k=top_k,
            workers=workers,
            with_prices=bool(with_prices),
            sec_budget=sec_budget,
            llm_budget=llm_budget,
            llm_provider=llm_provider,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-depth-batch-status")
def universe_depth_batch_status_cmd(
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
) -> None:
    configure_logging()
    from app.universe.batch_runner import open_depth_batch_status

    payload = open_depth_batch_status(batch_run_id=batch_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-depth-batch-resume")
def universe_depth_batch_resume_cmd(
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
    max_runs: int | None = typer.Option(
        None, "--max-runs", help="Optional cap for this resume invocation"
    ),
    mode: str | None = typer.Option(None, "--mode", help="Optional mode override (depth)"),
    iterations: int | None = typer.Option(
        None, "--iterations", help="Optional override iteration cap"
    ),
    top_k: int | None = typer.Option(None, "--top-k", help="Optional override top-k"),
    workers: int | None = typer.Option(None, "--workers", help="Optional override worker count"),
    with_prices: bool | None = typer.Option(
        None, "--with-prices/--no-with-prices", help="Optional price hydration override"
    ),
    sec_budget: int | None = typer.Option(
        None, "--sec-budget", help="Optional SEC budget annotation override"
    ),
    llm_budget: float | None = typer.Option(
        None, "--llm-budget", help="Optional LLM budget override"
    ),
    llm_provider: str | None = typer.Option(
        None, "--llm-provider", help="Optional VOE_LLM_PROVIDER override"
    ),
) -> None:
    configure_logging()
    from app.universe.batch_runner import resume_depth_batch

    init_db(get_config())
    try:
        payload = resume_depth_batch(
            batch_run_id,
            max_runs=max_runs,
            mode=mode,
            iterations=iterations,
            top_k=top_k,
            workers=workers,
            with_prices=with_prices,
            sec_budget=sec_budget,
            llm_budget=llm_budget,
            llm_provider=llm_provider,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-depth-batch-cancel")
def universe_depth_batch_cancel_cmd(
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
    reason: str = typer.Option(..., "--reason", help="Cancellation reason"),
) -> None:
    configure_logging()
    from app.universe.batch_runner import cancel_depth_batch

    payload = cancel_depth_batch(batch_run_id, reason)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-depth-batch-rollup")
def universe_depth_batch_rollup_cmd(
    universe_run_id: str = typer.Option(..., "--universe-run-id", help="Universe scout run id"),
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
    top_n: int = typer.Option(25, "--top-n", help="Top N shortlist rows"),
    policy: str = typer.Option(
        "value_first",
        "--policy",
        help="Rollup policy (value_first, value_first_quality, value_first_intangible, value_first_discipline, value_first_confidence, value_first_typed, value_first_trustworthy, value_first_ready, value_first_reinvestment, value_first_cash_earnings, value_first_balance_sheet, value_first_durable_returns, value_first_revenue_resilience, value_first_owner_earnings_hardness, value_first_asset_support_quality, value_first_residual_equity)",
    ),
) -> None:
    configure_logging()
    from app.universe.depth_rollup import write_depth_batch_rollup

    init_db(get_config())
    try:
        payload = write_depth_batch_rollup(
            universe_run_id=universe_run_id,
            batch_run_id=batch_run_id,
            top_n=max(1, int(top_n)),
            policy=policy,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-depth-batch-rollup-open")
def universe_depth_batch_rollup_open_cmd(
    universe_run_id: str = typer.Option(..., "--universe-run-id", help="Universe scout run id"),
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
) -> None:
    configure_logging()
    from app.universe.depth_rollup import open_depth_batch_rollup

    payload = open_depth_batch_rollup(
        universe_run_id=universe_run_id, batch_run_id=batch_run_id, top_n=10
    )
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-dossier-pack")
def universe_dossier_pack_cmd(
    universe_run_id: str = typer.Option(..., "--universe-run-id", help="Universe scout run id"),
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
    top_n: int = typer.Option(25, "--top-n", help="Top N shortlist rows"),
    policy: str = typer.Option(
        "value_first",
        "--policy",
        help="Dossier pack ranking policy (value_first, value_first_quality, value_first_intangible, value_first_discipline, value_first_confidence, value_first_typed, value_first_trustworthy, value_first_ready, value_first_reinvestment, value_first_cash_earnings, value_first_balance_sheet, value_first_durable_returns, value_first_revenue_resilience, value_first_owner_earnings_hardness, value_first_asset_support_quality, value_first_residual_equity)",
    ),
) -> None:
    configure_logging()
    from app.universe.dossier_pack import write_dossier_pack

    init_db(get_config())
    try:
        payload = write_dossier_pack(
            universe_run_id=universe_run_id,
            batch_run_id=batch_run_id,
            top_n=max(1, int(top_n)),
            policy=policy,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-dossier-pack-open")
def universe_dossier_pack_open_cmd(
    universe_run_id: str = typer.Option(..., "--universe-run-id", help="Universe scout run id"),
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
) -> None:
    configure_logging()
    from app.universe.dossier_pack import open_dossier_pack

    payload = open_dossier_pack(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-memo-pack")
def universe_memo_pack_cmd(
    universe_run_id: str = typer.Option(..., "--universe-run-id", help="Universe scout run id"),
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
    top_n: int = typer.Option(10, "--top-n", help="Top N shortlist rows"),
    policy: str = typer.Option(
        "value_first",
        "--policy",
        help="Memo ranking policy (value_first, value_first_quality, value_first_intangible, value_first_discipline, value_first_confidence, value_first_typed, value_first_trustworthy, value_first_ready, value_first_reinvestment, value_first_cash_earnings, value_first_balance_sheet, value_first_durable_returns, value_first_revenue_resilience, value_first_owner_earnings_hardness, value_first_asset_support_quality, value_first_residual_equity)",
    ),
) -> None:
    configure_logging()
    from app.universe.memo_pack import write_investment_memo_pack

    init_db(get_config())
    try:
        payload = write_investment_memo_pack(
            universe_run_id=universe_run_id,
            batch_run_id=batch_run_id,
            top_n=max(1, int(top_n)),
            policy=policy,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-memo-pack-open")
def universe_memo_pack_open_cmd(
    universe_run_id: str = typer.Option(..., "--universe-run-id", help="Universe scout run id"),
    batch_run_id: str = typer.Option(..., "--batch-run-id", help="Depth batch run id"),
) -> None:
    configure_logging()
    from app.universe.memo_pack import open_investment_memo_pack

    payload = open_investment_memo_pack(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-watchlist-state-open")
def universe_watchlist_state_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe autopilot run id"),
) -> None:
    configure_logging()
    from app.universe.memo_pack import open_watchlist_state

    payload = open_watchlist_state(run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-watchlist-diff")
def universe_watchlist_diff_cmd(
    prev_run_id: str = typer.Option(..., "--prev-run-id", help="Previous universe run id"),
    run_id: str = typer.Option(..., "--run-id", help="Current universe run id"),
) -> None:
    configure_logging()
    from app.universe.memo_pack import diff_watchlist_states

    payload = diff_watchlist_states(prev_run_id=prev_run_id, run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-campaign")
def universe_campaign_cmd(
    campaign_file: Path = typer.Option(
        ..., "--campaign-file", exists=True, help="Campaign JSON spec"
    ),
    campaign_run_id: str = typer.Option(
        ..., "--campaign-run-id", help="Deterministic campaign run id"
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume existing campaign state"
    ),
    force: bool = typer.Option(False, "--force", help="Force rerun campaign items"),
    max_items: int | None = typer.Option(
        None, "--max-items", help="Optional item cap for this invocation"
    ),
    depth: str = typer.Option(
        "fundamentals", "--depth", help="Autopilot depth: fundamentals, full, or alpha-only"
    ),
) -> None:
    configure_logging()
    from app.universe.campaign import run_campaign

    init_db(get_config())
    try:
        payload = run_campaign(
            campaign_file,
            campaign_run_id=campaign_run_id,
            resume=bool(resume),
            force=bool(force),
            max_items=max_items,
            depth=depth,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") in {"FAILED", "MISSING"}:
        raise typer.Exit(code=1)


@app.command("universe-campaign-status")
def universe_campaign_status_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
) -> None:
    configure_logging()
    from app.universe.campaign import campaign_status

    payload = campaign_status(campaign_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-campaign-open")
def universe_campaign_open_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
) -> None:
    configure_logging()
    from app.universe.campaign import open_campaign

    payload = open_campaign(campaign_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-campaign-cancel")
def universe_campaign_cancel_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
    reason: str = typer.Option(..., "--reason", help="Cancellation reason"),
) -> None:
    configure_logging()
    from app.universe.campaign import cancel_campaign

    payload = cancel_campaign(campaign_run_id, reason)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-campaign-promotion-open")
def universe_campaign_promotion_open_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
) -> None:
    configure_logging()
    from app.universe.promotion import open_promotion_state

    payload = open_promotion_state(campaign_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-campaign-priority-lanes-open")
def universe_campaign_priority_lanes_open_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
) -> None:
    configure_logging()
    from app.universe.promotion import open_priority_lanes

    payload = open_priority_lanes(campaign_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-campaign-escalation-plan")
def universe_campaign_escalation_plan_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
) -> None:
    configure_logging()
    from app.universe.escalation import write_escalation_artifacts

    payload = write_escalation_artifacts(campaign_run_id)
    typer.echo(json.dumps(payload, indent=2))


@app.command("universe-campaign-escalation-open")
def universe_campaign_escalation_open_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
) -> None:
    configure_logging()
    from app.universe.escalation import open_escalation_plan

    payload = open_escalation_plan(campaign_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-campaign-escalation-run")
def universe_campaign_escalation_run_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
    max_items: int | None = typer.Option(
        None, "--max-items", help="Optional execution cap for this invocation"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Write state only; do not execute queue items"
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from existing escalation state"
    ),
    force_restart: bool = typer.Option(
        False, "--force-restart", help="Restart queue execution from the beginning"
    ),
) -> None:
    configure_logging()
    from app.universe.escalation_runner import run_escalation_queue

    init_db(get_config())
    payload = run_escalation_queue(
        campaign_run_id,
        max_items=max_items,
        dry_run=bool(dry_run),
        resume=bool(resume),
        force_restart=bool(force_restart),
    )
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") in {"FAILED", "MISSING"}:
        raise typer.Exit(code=1)


@app.command("universe-campaign-escalation-status")
def universe_campaign_escalation_status_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
) -> None:
    configure_logging()
    from app.universe.escalation_runner import open_escalation_status

    payload = open_escalation_status(campaign_run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-campaign-escalation-cancel")
def universe_campaign_escalation_cancel_cmd(
    campaign_run_id: str = typer.Option(..., "--campaign-run-id", help="Campaign run id"),
    reason: str = typer.Option(..., "--reason", help="Cancellation reason"),
) -> None:
    configure_logging()
    from app.universe.escalation_runner import cancel_escalation_queue

    payload = cancel_escalation_queue(campaign_run_id, reason)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-research-memory-open")
def universe_research_memory_open_cmd(
    ticker: str | None = typer.Option(None, "--ticker", help="Optional ticker symbol"),
) -> None:
    configure_logging()
    from app.universe.research_memory import open_research_memory

    payload = open_research_memory(ticker=ticker)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-research-memory-diff")
def universe_research_memory_diff_cmd(
    ticker: str = typer.Option(..., "--ticker", help="Ticker symbol"),
) -> None:
    configure_logging()
    from app.universe.research_memory import diff_research_memory

    payload = diff_research_memory(ticker)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-research-memory-summary")
def universe_research_memory_summary_cmd() -> None:
    configure_logging()
    from app.universe.research_memory import open_research_memory

    payload = open_research_memory()
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-research-memory-priority-open")
def universe_research_memory_priority_open_cmd() -> None:
    configure_logging()
    from app.universe.research_memory import open_research_memory_priority

    payload = open_research_memory_priority()
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-autopilot")
def universe_autopilot_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe autopilot run id"),
    as_of: str | None = typer.Option(
        None, "--as-of", help="As-of date YYYY-MM-DD (defaults to today)"
    ),
    max_runs: int | None = typer.Option(None, "--max-runs", help="Depth batch max runs"),
    top_n: int = typer.Option(25, "--top-n", help="Rollup/dossier top N"),
    policy: str = typer.Option(
        "value_first",
        "--policy",
        help="Rollup/dossier policy (value_first, value_first_quality, value_first_intangible, value_first_discipline, value_first_confidence, value_first_typed, value_first_trustworthy, value_first_ready, value_first_reinvestment, value_first_cash_earnings, value_first_balance_sheet, value_first_durable_returns, value_first_revenue_resilience, value_first_owner_earnings_hardness, value_first_asset_support_quality, value_first_residual_equity)",
    ),
    depth: str = typer.Option(
        "fundamentals", "--depth", help="Pipeline depth: fundamentals, full, or alpha-only"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write staged plan only; do not execute"),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from existing autopilot state"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force rerun refreshable stages (rollup/dossier)"
    ),
    review_intake_from_run: str | None = typer.Option(
        None,
        "--review-intake-from-run",
        help="Prior run id to seed bounded review intake from review_plan_seed.json",
    ),
    review_intake_max_names: int | None = typer.Option(
        None,
        "--review-intake-max-names",
        help="Cap on review intake tickers (default: all eligible)",
    ),
) -> None:
    configure_logging()
    from app.universe.autopilot import run_universe_autopilot

    as_of_date = parse_yyyy_mm_dd(as_of).isoformat() if as_of else date.today().isoformat()
    init_db(get_config())
    try:
        payload = run_universe_autopilot(
            universe_run_id=run_id,
            as_of_date=as_of_date,
            scout_params={},
            depth_batch_params={
                "max_runs": max_runs,
                "mode": "depth",
                "iterations": 1,
                "top_k": 5,
                "with_prices": True,
            },
            rollup_params={
                "top_n": max(1, int(top_n)),
                "policy": policy,
            },
            dossier_pack_params={
                "top_n": max(1, int(top_n)),
                "policy": policy,
            },
            memo_pack_params={
                "top_n": max(1, int(top_n)),
                "policy": policy,
            },
            resume=bool(resume),
            depth=depth,
            dry_run=bool(dry_run),
            force=bool(force),
            review_intake_from_run=review_intake_from_run or None,
            review_intake_max_names=review_intake_max_names,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") in {"FAILED", "MISSING"}:
        raise typer.Exit(code=1)


@app.command("universe-autopilot-status")
def universe_autopilot_status_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe autopilot run id"),
) -> None:
    configure_logging()
    from app.universe.autopilot import universe_autopilot_status

    payload = universe_autopilot_status(run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-autopilot-open")
def universe_autopilot_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe autopilot run id"),
) -> None:
    configure_logging()
    from app.universe.autopilot import open_universe_autopilot

    payload = open_universe_autopilot(run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-autopilot-cancel")
def universe_autopilot_cancel_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe autopilot run id"),
    reason: str = typer.Option(..., "--reason", help="Cancellation reason"),
) -> None:
    configure_logging()
    from app.universe.autopilot import cancel_universe_autopilot

    payload = cancel_universe_autopilot(run_id, reason)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


# New-sector bootstrap sequence for online cache seeding before offline depth:
# 1. python -m app.cli universe-bootstrap-sector-cache --sector industrial_tech
# 2. python -m app.cli universe-warm-filing-cache --sector industrial_tech --years 2
# 3. python -m app.cli sector-rlm-depth --sector industrial_tech --as-of 2026-03-23 --run-id industrial_tech_live_v1
#
# Repeat the same three-step flow for large_cap_value before cross-sector runs.
@app.command("universe-discover")
def universe_discover_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector key from sector_sic_ranges.json"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Force refresh of sector discovery cache"
    ),
) -> None:
    configure_logging()
    from app.universe.sector_universe import discover_sector_universe

    init_db(get_config())
    rows = discover_sector_universe(sector, refresh_cache=refresh)
    first_twenty = rows[:20]
    typer.echo(f"Discovered {len(rows)} companies for sector={sector}")
    for row in first_twenty:
        typer.echo(
            f"{str(row.get('company_name') or '').strip()} "
            f"({str(row.get('ticker') or '').upper()}, CIK {str(row.get('cik') or '')}, SIC {str(row.get('sic') or 'UNKNOWN')})"
        )


@app.command("universe-deep-scan")
def universe_deep_scan_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector key from sector_sic_ranges.json"),
    as_of: str | None = typer.Option(None, "--as-of", help="Analysis as-of date YYYY-MM-DD"),
    tier1_limit: int = typer.Option(300, "--tier1-limit", help="Max discovered names to Tier 1"),
    tier2_limit: int = typer.Option(30, "--tier2-limit", help="Max Tier 2 names to forward"),
    tier3_limit: int = typer.Option(10, "--tier3-limit", help="Max Tier 3 names to run"),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id"),
    continuous: bool = typer.Option(
        False, "--continuous", help="Continuously refresh Tier 2 prices and filing state"
    ),
    interval_seconds: int | None = typer.Option(
        None, "--interval-seconds", help="Continuous refresh interval in seconds"
    ),
) -> None:
    configure_logging()
    from app.universe.sector_universe import deep_scan_sector

    init_db(get_config())
    as_of_date = parse_yyyy_mm_dd(as_of).isoformat() if as_of else date.today().isoformat()
    summary = deep_scan_sector(
        sector=sector,
        as_of_date=as_of_date,
        tier1_limit=max(1, int(tier1_limit)),
        tier2_limit=max(1, int(tier2_limit)),
        tier3_limit=max(1, int(tier3_limit)),
        run_id=run_id,
        continuous=continuous,
        interval_seconds=interval_seconds,
    )
    typer.echo(
        f"Tier 1: {summary.get('tier1_evaluated_count', 0)} screened from {summary.get('discovered_count', 0)} discovered, "
        f"{summary.get('tier1_pass_count', 0)} passed."
    )
    typer.echo(
        f"Tier 2: {summary.get('tier2_ticker_count', 0)} valued, zones={json.dumps(summary.get('tier2_zone_distribution') or {}, sort_keys=True)}"
    )
    typer.echo(f"Tier 3: running set size {len(summary.get('tier3_tickers') or [])}")
    typer.echo(json.dumps(summary, indent=2))


@app.command("universe-bootstrap-sector-cache")
def universe_bootstrap_sector_cache_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector label"),
    tickers: str | None = typer.Option(
        None, "--tickers", help="Optional comma-separated ticker subset"
    ),
) -> None:
    configure_logging()
    from app.universe.bootstrap_sector_cache import bootstrap_sector_cache

    init_db(get_config())
    payload = bootstrap_sector_cache(
        sector=sector,
        tickers=_parse_tickers_csv(tickers) if tickers else None,
    )
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") not in {"OK", "NO_TARGETS"}:
        raise typer.Exit(code=1)


@app.command("universe-warm-filing-cache")
def universe_warm_filing_cache_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector label"),
    tickers: str | None = typer.Option(
        None, "--tickers", help="Optional comma-separated ticker subset"
    ),
    years: int = typer.Option(3, "--years", help="Most recent annual filings to cache per ticker"),
) -> None:
    configure_logging()
    from app.dossier.filing_cache import warm_filing_cache

    init_db(get_config())
    payload = warm_filing_cache(
        sector=sector,
        tickers=_parse_tickers_csv(tickers) if tickers else None,
        years=max(1, int(years)),
    )
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") not in {"OK", "NO_TARGETS"}:
        raise typer.Exit(code=1)


@app.command("net-debt-coverage-open")
def net_debt_coverage_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
) -> None:
    configure_logging()
    from app.universe.scout import open_net_debt_coverage

    payload = open_net_debt_coverage(run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-scout-status")
def universe_scout_status_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
) -> None:
    configure_logging()
    from app.universe.scout import open_universe_scout_status

    payload = open_universe_scout_status(run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-scout-resume")
def universe_scout_resume_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
    max_batches: int | None = typer.Option(
        None, "--max-batches", help="Optional max batches to process this invocation"
    ),
    scout_sec_budget: int | None = typer.Option(
        None, "--scout-sec-budget", help="SEC request budget override for resume"
    ),
    scout_net_budget: int | None = typer.Option(
        None, "--scout-net-budget", help="Non-SEC network budget override for resume"
    ),
    scout_max_seconds: int | None = typer.Option(
        None, "--scout-max-seconds", help="Wall-clock budget override for resume"
    ),
) -> None:
    configure_logging()
    from app.universe.scout import run_universe_scout_resume

    init_db(get_config())
    try:
        payload = run_universe_scout_resume(
            run_id=run_id,
            max_batches=max_batches,
            scout_sec_budget=scout_sec_budget,
            scout_net_budget=scout_net_budget,
            scout_max_seconds=scout_max_seconds,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command("universe-scout-cancel")
def universe_scout_cancel_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
    reason: str = typer.Option(..., "--reason", help="Cancellation reason"),
) -> None:
    configure_logging()
    from app.universe.scout import cancel_universe_scout_run

    payload = cancel_universe_scout_run(run_id=run_id, reason=reason)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("universe-scout-to-depth")
def universe_scout_to_depth_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Universe scout run id"),
    sector: str = typer.Option(..., "--sector", help="Sector label for depth run"),
    depth_run_id: str = typer.Option(..., "--depth-run-id", help="Depth RLM run id"),
    iterations: int = typer.Option(3, "--iterations", help="Depth iteration cap"),
    top_k: int = typer.Option(10, "--top-k", help="Depth top-k"),
    shortlist_limit: int | None = typer.Option(
        None, "--shortlist-limit", help="Optional cap on shortlist tickers passed to depth"
    ),
    with_research: bool = typer.Option(
        True, "--with-research/--no-research", help="Enable depth research actions"
    ),
    with_synthesis: bool = typer.Option(
        True, "--with-synthesis/--no-synthesis", help="Enable depth synthesis actions"
    ),
    force_restart: bool = typer.Option(
        False, "--force-restart", help="Force-restart depth run id before launch"
    ),
) -> None:
    configure_logging()
    from app.universe.scout import run_universe_scout_to_depth

    init_db(get_config())
    try:
        payload = run_universe_scout_to_depth(
            scout_run_id=run_id,
            depth_run_id=depth_run_id,
            sector=sector,
            iterations=max(1, int(iterations)),
            top_k=max(1, int(top_k)),
            shortlist_limit=shortlist_limit,
            with_research=bool(with_research),
            with_synthesis=bool(with_synthesis),
            force_restart=bool(force_restart),
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command("discovery-seed-validate")
def discovery_seed_validate(
    csv: Path = typer.Option(Path("data/universe/discovery_seed.csv"), "--csv", exists=True),
) -> None:
    configure_logging()
    from app.discovery.seed import validate_seed_csv

    init_db(get_config())
    summary = validate_seed_csv(csv)
    typer.echo(json.dumps({"csv": str(csv), **summary}, indent=2))


@app.command("discovery-run")
def discovery_run_cmd(
    as_of: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    limit: int | None = typer.Option(None, help="Optional ticker cap"),
    tickers: str | None = typer.Option(None, help="Optional explicit tickers list"),
    seed: Path | None = typer.Option(None, "--seed", help="Discovery seed CSV"),
    exclude: Path | None = typer.Option(None, "--exclude", help="Optional exclude CSV"),
    top: int = typer.Option(25, "--top", help="Top K shortlist for deep research"),
    advance_top: int | None = typer.Option(
        None, "--advance-top", help="Only export top K ADVANCE_TO_DEEP names"
    ),
    mktcap_min: float = typer.Option(
        5_000_000_000.0, "--mktcap-min", help="Market cap minimum for target band"
    ),
    mktcap_max: float = typer.Option(
        50_000_000_000.0, "--mktcap-max", help="Market cap maximum for target band"
    ),
    phase: str = typer.Option("auto", "--phase", help="Discovery phase: auto|prefilter|full"),
    prefilter_cap: int | None = typer.Option(
        None, "--prefilter-cap", help="Max phase-2 prefilter shortlist"
    ),
    prefilter_keep_ratio: float | None = typer.Option(
        None, "--prefilter-keep-ratio", help="Phase-1 keep ratio for phase-2 scope (0.05-0.8)"
    ),
    workers: int | None = typer.Option(None, "--workers", help="Discovery worker count"),
) -> None:
    configure_logging()
    from app.discovery.runner import run_discovery

    init_db(get_config())
    parse_yyyy_mm_dd(as_of)
    summary = run_discovery(
        as_of_date=as_of,
        limit=limit,
        tickers=_parse_tickers_csv(tickers),
        seed_path=seed,
        exclude_path=exclude,
        top_k=top,
        advance_top=advance_top,
        mktcap_min=mktcap_min,
        mktcap_max=mktcap_max,
        phase=phase,
        prefilter_cap=prefilter_cap,
        prefilter_keep_ratio=prefilter_keep_ratio,
        workers=workers,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("discovery-resume")
def discovery_resume_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Existing discovery run id"),
    phase: str = typer.Option(
        "auto", "--phase", help="Discovery phase to continue: auto|prefilter|full"
    ),
    prefilter_cap: int | None = typer.Option(
        None, "--prefilter-cap", help="Max phase-2 prefilter shortlist"
    ),
    prefilter_keep_ratio: float | None = typer.Option(
        None, "--prefilter-keep-ratio", help="Phase-1 keep ratio for phase-2 scope (0.05-0.8)"
    ),
    workers: int | None = typer.Option(None, "--workers", help="Discovery worker count"),
    top: int | None = typer.Option(None, "--top", help="Optional top-K shortlist override"),
    advance_top: int | None = typer.Option(
        None, "--advance-top", help="Only export top K ADVANCE_TO_DEEP names"
    ),
    mktcap_min: float = typer.Option(
        5_000_000_000.0, "--mktcap-min", help="Market cap minimum for target band"
    ),
    mktcap_max: float = typer.Option(
        50_000_000_000.0, "--mktcap-max", help="Market cap maximum for target band"
    ),
) -> None:
    configure_logging()
    from app.discovery.runner import run_discovery_resume

    init_db(get_config())
    summary = run_discovery_resume(
        run_id=run_id,
        phase=phase,
        prefilter_cap=prefilter_cap,
        prefilter_keep_ratio=prefilter_keep_ratio,
        workers=workers,
        top_k=top,
        advance_top=advance_top,
        mktcap_min=mktcap_min,
        mktcap_max=mktcap_max,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("discovery-run-all")
def discovery_run_all_cmd(
    as_of: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    limit: int | None = typer.Option(None, help="Optional ticker cap"),
    seed: Path | None = typer.Option(None, "--seed", help="Discovery seed CSV"),
    top: int = typer.Option(25, "--top", help="Top K shortlist for deep research"),
    advance_top: int | None = typer.Option(
        None, "--advance-top", help="Only export top K ADVANCE_TO_DEEP names"
    ),
    mktcap_min: float = typer.Option(
        5_000_000_000.0, "--mktcap-min", help="Market cap minimum for target band"
    ),
    mktcap_max: float = typer.Option(
        50_000_000_000.0, "--mktcap-max", help="Market cap maximum for target band"
    ),
) -> None:
    configure_logging()
    from app.discovery.runner import run_discovery_all

    init_db(get_config())
    parse_yyyy_mm_dd(as_of)
    summary = run_discovery_all(
        as_of_date=as_of,
        limit=limit,
        top_k=top,
        advance_top=advance_top,
        seed_path=seed,
        mktcap_min=mktcap_min,
        mktcap_max=mktcap_max,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("discovery-export")
def discovery_export_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Discovery run id"),
    out: Path = typer.Option(..., "--out", help="Output directory"),
) -> None:
    configure_logging()
    from app.discovery.export import export_discovery_artifacts

    init_db(get_config())
    summary = export_discovery_artifacts(run_id, out)
    typer.echo(json.dumps({"run_id": run_id, "copied": summary}, indent=2))


@app.command("discovery-apply")
def discovery_apply_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Discovery run id"),
    set_active: bool = typer.Option(
        False, "--set-active", help="Merge and set as active universe snapshot"
    ),
) -> None:
    configure_logging()
    from app.discovery.export import apply_discovery_run

    init_db(get_config())
    summary = apply_discovery_run(run_id, set_active=set_active)
    typer.echo(json.dumps(summary, indent=2))


@app.command("dossier-run")
def dossier_run_cmd(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated peer tickers"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    years_back: int = typer.Option(
        10, "--years-back", help="Years of annual filings (10-K/20-F/40-F) to analyze"
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional deterministic run id"),
    workers: int = typer.Option(2, "--workers", help="Parallel workers"),
) -> None:
    configure_logging()
    from app.dossier.runner import run_dossier_for_peer_set

    parse_yyyy_mm_dd(as_of)
    init_db(get_config())
    if workers > 1:
        typer.echo(
            "Note: dossier parsing is serialized for SQLite safety; only filing collection/download is parallel."
        )
    summary = run_dossier_for_peer_set(
        tickers=_parse_tickers_csv(tickers),
        as_of_date=as_of,
        years_back=years_back,
        run_id=run_id,
        workers=workers,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("dossier-resume")
def dossier_resume_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Dossier run id"),
    workers: int = typer.Option(2, "--workers", help="Parallel workers for stage-1 collection"),
) -> None:
    configure_logging()
    from app.dossier.runner import resume_dossier_run

    init_db(get_config())
    if workers > 1:
        typer.echo(
            "Note: dossier parsing is serialized for SQLite safety; only filing collection/download is parallel."
        )
    summary = resume_dossier_run(run_id=run_id, workers=workers)
    typer.echo(json.dumps(summary, indent=2))


@app.command("dossier-open")
def dossier_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Dossier run id"),
) -> None:
    configure_logging()
    from app.dossier.runner import open_dossier_run

    summary = open_dossier_run(run_id)
    typer.echo(json.dumps(summary, indent=2))


@app.command("dossier-compare")
def dossier_compare_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Dossier run id"),
    metric: str = typer.Option(..., "--metric", help="Metric key to compare"),
) -> None:
    configure_logging()
    from app.dossier.runner import compare_dossier_metric

    payload = compare_dossier_metric(run_id=run_id, metric=metric)
    typer.echo(json.dumps(payload, indent=2))


@app.command("filing-diff")
def filing_diff_cmd(
    ticker: str = typer.Option(..., "--ticker", help="Ticker symbol"),
    run_id: str = typer.Option(..., "--run-id", help="Dossier run id"),
    years_back: int = typer.Option(5, "--years-back", help="Number of annual filings to include"),
) -> None:
    configure_logging()
    from app.diff.engine import build_filing_diff_for_ticker, summarize_diff_report
    from app.diff.schemas import FilingDiffReport

    path = build_filing_diff_for_ticker(
        ticker=ticker.upper(),
        run_id=run_id,
        years_back=years_back,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = summarize_diff_report(FilingDiffReport.model_validate(payload))
    typer.echo(str(path))
    typer.echo(json.dumps(summary, indent=2))


@app.command("filing-diff-all")
def filing_diff_all_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Dossier run id"),
    years_back: int = typer.Option(5, "--years-back", help="Number of annual filings to include"),
    limit: int | None = typer.Option(None, "--limit", help="Optional cap on tickers"),
) -> None:
    configure_logging()
    from app.diff.engine import (
        build_filing_diff_for_ticker,
        list_dossier_run_tickers,
        summarize_diff_report,
    )
    from app.diff.schemas import FilingDiffReport

    tickers = list_dossier_run_tickers(run_id=run_id)
    if limit is not None and limit > 0:
        tickers = tickers[:limit]
    results: list[dict[str, Any]] = []
    for ticker in tickers:
        path = build_filing_diff_for_ticker(
            ticker=ticker,
            run_id=run_id,
            years_back=years_back,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = FilingDiffReport.model_validate(payload)
        results.append(
            {
                "ticker": ticker,
                "artifact_path": str(path),
                "summary": summarize_diff_report(report),
            }
        )
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "years_back": years_back,
                "tickers_processed": tickers,
                "results": results,
            },
            indent=2,
        )
    )


@app.command("dossier-whale-signals")
def dossier_whale_signals_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Dossier run id"),
) -> None:
    configure_logging()
    from app.dossier.whale_signals import run_whale_signals_for_run

    init_db(get_config())
    payload = run_whale_signals_for_run(run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))


@app.command("whale-baseline")
def whale_baseline_cmd(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated ticker list"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    years_back: int = typer.Option(
        10, "--years-back", help="Years of annual filings (10-K/20-F/40-F) to analyze"
    ),
    early_window_years: int = typer.Option(
        5, "--early-window-years", help="Analyze only earliest N years in dossier series"
    ),
    controls: str | None = typer.Option(None, "--controls", help="Optional control tickers CSV"),
    controls_count: int = typer.Option(
        20, "--controls-count", help="Auto-controls target count when --controls omitted"
    ),
    peer_mode: str = typer.Option(
        "hybrid", "--peer-mode", help="Controls peer selection mode: taxonomy|filings|hybrid"
    ),
    sector: str | None = typer.Option(
        None, "--sector", help="Optional sector for controls generation"
    ),
    min_peers: int = typer.Option(
        25, "--min-peers", help="Minimum peer target for auto-controls selection"
    ),
    max_peers: int | None = typer.Option(
        None, "--max-peers", help="Maximum peers to consider for auto-controls selection"
    ),
    sic_expand: bool = typer.Option(
        True,
        "--sic-expand/--no-sic-expand",
        help="Enable SIC exact-match expansion for auto-controls",
    ),
    sic_family: bool = typer.Option(
        True,
        "--sic-family/--no-sic-family",
        help="Enable SIC-family expansion when still short of peers",
    ),
    include_foreign: bool = typer.Option(
        True,
        "--include-foreign/--no-include-foreign",
        help="Allow foreign annual forms (20-F/40-F) for controls",
    ),
    include_otc: bool = typer.Option(
        False, "--include-otc/--no-include-otc", help="Allow OTC controls"
    ),
    min_annual_filings: int | None = typer.Option(
        None, "--min-annual-filings", help="Minimum annual filings required in lookback window"
    ),
    sec_budget: int | None = typer.Option(
        None, "--sec-budget", help="Temporary per-host SEC domain budget for this run"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic baseline run id"),
    workers: int | None = typer.Option(None, "--workers", help="Optional dossier worker override"),
) -> None:
    configure_logging()
    from app.dossier.baseline import run_whale_baseline

    parse_yyyy_mm_dd(as_of)
    init_db(get_config())
    payload = run_whale_baseline(
        tickers=_parse_tickers_csv(tickers),
        as_of_date=as_of,
        years_back=years_back,
        early_window_years=early_window_years,
        controls=_parse_tickers_csv(controls),
        controls_count=controls_count,
        peer_mode=peer_mode,
        sector=sector,
        min_peers=min_peers,
        max_peers=max_peers,
        sic_expand=sic_expand,
        sic_family=sic_family,
        include_foreign=include_foreign,
        include_otc=include_otc,
        min_annual_filings=min_annual_filings,
        sec_budget=sec_budget,
        run_id=run_id,
        workers=workers,
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("sector-classify")
def sector_classify_cmd(
    as_of: str | None = typer.Option(
        None, "--as-of", help="As-of date YYYY-MM-DD (defaults to today)"
    ),
    tickers: str | None = typer.Option(
        None, "--tickers", help="Optional comma-separated ticker scope"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show classifications without writing to DB"
    ),
    force: bool = typer.Option(False, "--force", help="Reclassify already-classified tickers"),
) -> None:
    """Auto-classify scorecarded or explicitly scoped tickers by sector using EDGAR SIC codes."""
    configure_logging()

    import json as _json
    from app.config import get_config
    from app.db import init_db
    from app.sector.classifier import classify_all

    cfg = get_config()
    init_db(cfg)

    summary = classify_all(
        dry_run=dry_run,
        force=force,
        as_of_date=as_of if as_of else None,
        tickers=_parse_tickers_csv(tickers),
        cfg=cfg,
    )

    typer.echo(
        _json.dumps(
            {
                "total": summary.total,
                "classified": summary.classified,
                "excluded_non_operating": summary.excluded_non_operating,
                "no_cik": summary.no_cik,
                "no_sic": summary.no_sic,
                "no_sector_match": summary.no_sector_match,
                "fetch_error": summary.fetch_error,
                "skipped_existing": summary.skipped_existing,
                "sector_counts": summary.sector_counts,
                "unclassified_count": len(summary.unclassified_tickers),
            },
            indent=2,
        )
    )

    if dry_run:
        typer.echo("\n(dry run — no rows written)")


@app.command("sector-list")
def sector_list_cmd() -> None:
    configure_logging()
    from app.sector.catalog import list_sector_catalogs

    typer.echo(json.dumps(list_sector_catalogs(), indent=2))


@app.command("sector-peers")
def sector_peers_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector name"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    years_back: int = typer.Option(
        10, "--years-back", help="Annual filing lookback years for precheck"
    ),
    limit: int = typer.Option(25, "--limit", help="Peer cap"),
    min_peers_dossierable: int = typer.Option(
        25,
        "--min-peers-dossierable",
        "--min-peers",
        help="Minimum dossierable peers target",
    ),
    max_peers: int | None = typer.Option(None, "--max-peers", help="Maximum peers to keep"),
    max_peer_scan: int = typer.Option(
        400, "--max-peer-scan", help="Max candidate tickers to scan during staged expansion"
    ),
    sic_expand: bool = typer.Option(
        True, "--sic-expand/--no-sic-expand", help="Enable SIC exact-match expansion"
    ),
    sic_family: bool = typer.Option(
        True, "--sic-family/--no-sic-family", help="Enable SIC-family expansion"
    ),
    include_foreign: bool = typer.Option(
        True,
        "--include-foreign/--no-include-foreign",
        help="Allow foreign annual forms (20-F/40-F)",
    ),
    include_otc: bool = typer.Option(
        False, "--include-otc/--no-include-otc", help="Include OTC tickers"
    ),
    min_annual_filings: int | None = typer.Option(
        3, "--min-annual-filings", help="Minimum annual filings required in lookback window"
    ),
    mktcap_min: float | None = typer.Option(None, "--mktcap-min", help="Market cap minimum"),
    mktcap_max: float | None = typer.Option(None, "--mktcap-max", help="Market cap maximum"),
    mode: str = typer.Option(
        "hybrid", "--mode", help="Peer selection mode: taxonomy|filings|hybrid"
    ),
) -> None:
    configure_logging()
    from app.sector.peer_set import select_sector_peers

    parse_yyyy_mm_dd(as_of)
    init_db(get_config())
    payload = select_sector_peers(
        sector=sector,
        as_of_date=as_of,
        years_back=years_back,
        limit=limit,
        min_peers=min_peers_dossierable,
        max_peers=max_peers if max_peers is not None else limit,
        max_peer_scan=max_peer_scan,
        stop_when_min_reached=True,
        sic_expand=sic_expand,
        sic_family=sic_family,
        include_foreign=include_foreign,
        include_otc=include_otc,
        min_annual_filings=min_annual_filings,
        mktcap_min=mktcap_min,
        mktcap_max=mktcap_max,
        mode=mode,
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("sector-synth-run")
def sector_synth_run_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector name"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    run_id: str = typer.Option(..., "--run-id", help="Sector/dossier run id"),
) -> None:
    configure_logging()
    from app.sector.synthesis import run_sector_synthesis

    parse_yyyy_mm_dd(as_of)
    init_db(get_config())
    summary = run_sector_synthesis(sector=sector, as_of_date=as_of, run_id=run_id)
    typer.echo(json.dumps(summary, indent=2))


@app.command("sector-cycle")
def sector_cycle_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector name"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    peer_limit: int = typer.Option(25, "--peer-limit", help="Peer cap"),
    min_peers_dossierable: int = typer.Option(
        25,
        "--min-peers-dossierable",
        "--min-peers",
        help="Minimum dossierable peers to target before dossiers",
    ),
    max_peers: int | None = typer.Option(None, "--max-peers", help="Maximum peers to retain"),
    max_peer_scan: int = typer.Option(
        400, "--max-peer-scan", help="Max candidate tickers to scan during staged expansion"
    ),
    sic_expand: bool = typer.Option(
        True, "--sic-expand/--no-sic-expand", help="Enable SIC exact-match expansion"
    ),
    sic_family: bool = typer.Option(
        True, "--sic-family/--no-sic-family", help="Enable SIC-family expansion"
    ),
    peer_mode: str = typer.Option(
        "hybrid", "--peer-mode", help="Peer selection mode: taxonomy|filings|hybrid"
    ),
    limit_dossiers: int = typer.Option(
        25, "--limit-dossiers", help="Run dossiers on top-N preliminary peers only"
    ),
    years_back: int = typer.Option(10, "--years-back", help="Annual filing years to include"),
    min_annual_filings: int | None = typer.Option(
        3, "--min-annual-filings", help="Minimum annual filings required in lookback window"
    ),
    include_foreign: bool = typer.Option(
        True,
        "--include-foreign/--no-include-foreign",
        help="Allow foreign annual forms (20-F/40-F)",
    ),
    include_otc: bool = typer.Option(
        False, "--include-otc/--no-include-otc", help="Include OTC tickers in peer set"
    ),
    sec_budget: int | None = typer.Option(
        None, "--sec-budget", help="Temporary per-host SEC domain budget for this run"
    ),
    workers: int = typer.Option(4, "--workers", help="Dossier workers"),
    with_research: bool = typer.Option(
        True, "--with-research/--no-research", help="Run research packets for peers"
    ),
    with_synthesis: bool = typer.Option(
        True, "--with-synthesis/--no-synthesis", help="Run sector synthesis"
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional deterministic run id"),
    mktcap_min: float | None = typer.Option(None, "--mktcap-min", help="Market cap minimum"),
    mktcap_max: float | None = typer.Option(None, "--mktcap-max", help="Market cap maximum"),
) -> None:
    configure_logging()
    from app.sector.cycle import run_sector_cycle

    parse_yyyy_mm_dd(as_of)
    init_db(get_config())
    summary = run_sector_cycle(
        sector=sector,
        as_of_date=as_of,
        peer_limit=peer_limit,
        years_back=years_back,
        min_annual_filings=min_annual_filings,
        workers=workers,
        with_research=with_research,
        with_synthesis=with_synthesis,
        run_id=run_id,
        mktcap_min=mktcap_min,
        mktcap_max=mktcap_max,
        peer_mode=peer_mode,
        min_peers_dossierable=min_peers_dossierable,
        max_peers=max_peers if max_peers is not None else peer_limit,
        max_peer_scan=max_peer_scan,
        sic_expand=sic_expand,
        sic_family=sic_family,
        include_foreign=include_foreign,
        include_otc=include_otc,
        sec_budget=sec_budget,
        limit_dossiers=limit_dossiers,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("sector-open")
def sector_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    from app.sector.cycle import open_sector_run

    summary = open_sector_run(run_id=run_id)
    typer.echo(json.dumps(summary, indent=2))


@app.command("sector-run-status")
def sector_run_status_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    from app.sector.cycle import sector_run_status

    payload = sector_run_status(run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))


@app.command("sector-rlm")
def sector_rlm_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector name"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic run id"),
    peer_limit: int = typer.Option(25, "--peer-limit", help="Peer cap for baseline sector-cycle"),
    min_peers_dossierable: int = typer.Option(
        15,
        "--min-peers-dossierable",
        "--min-peers",
        help="Minimum dossierable peers target for baseline cycle",
    ),
    limit_dossiers: int = typer.Option(25, "--limit-dossiers", help="Baseline dossier cap"),
    years_back: int = typer.Option(10, "--years-back", help="Years-back for dossier collection"),
    workers: int = typer.Option(4, "--workers", help="Worker count"),
    iterations: int = typer.Option(2, "--iterations", help="Maximum recursion iterations"),
    top_k: int = typer.Option(5, "--top-k", help="Top-K candidates tracked by critic"),
    with_research: bool = typer.Option(
        True, "--with-research/--no-research", help="Enable CLOSE_GAPS actions"
    ),
    with_synthesis: bool = typer.Option(
        True, "--with-synthesis/--no-synthesis", help="Enable RUN_SYNTHESIS actions"
    ),
    with_prices: bool = typer.Option(
        True, "--with-prices/--no-with-prices", help="Enable price snapshots for valuation"
    ),
    mode: str = typer.Option("auto", "--mode", help="RLM mode: auto|depth"),
    llm_provider: str | None = typer.Option(
        None,
        "--llm-provider",
        help="Override VOE_LLM_PROVIDER for this process (e.g. disabled|openai)",
    ),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="LLM budget cap for this loop run"
    ),
    gate_mos_min: float | None = typer.Option(
        None, "--gate-mos-min", help="Optional value-gate PASS MOS threshold"
    ),
    gate_valuation_gap_min: float | None = typer.Option(
        None, "--gate-valuation-gap-min", help="Optional value-gate WATCH valuation-gap threshold"
    ),
    gate_net_debt_to_cfo_max: float | None = typer.Option(
        None,
        "--gate-net-debt-to-cfo-max",
        help="Optional value-gate PASS net-debt/CFO max threshold",
    ),
    gate_dilution_max: float | None = typer.Option(
        None, "--gate-dilution-max", help="Optional value-gate PASS dilution max threshold"
    ),
    timeout_per_stage: float = typer.Option(
        30.0, "--timeout-per-stage", help="Per-stage execution timeout in seconds"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing rlm_state.json"),
    force_restart: bool = typer.Option(
        False, "--force-restart", help="Wipe this run_id state/artifacts and restart"
    ),
) -> None:
    configure_logging()
    from app.rlm.loop import run_sector_rlm_loop

    parse_yyyy_mm_dd(as_of)
    if llm_provider:
        os.environ["VOE_LLM_PROVIDER"] = llm_provider.strip().lower()
        get_config.cache_clear()
    mode_norm = mode.strip().lower()
    if mode_norm not in {"auto", "depth"}:
        raise typer.BadParameter("--mode must be auto or depth")
    init_db(get_config())
    try:
        payload = run_sector_rlm_loop(
            sector=sector,
            as_of_date=as_of,
            run_id=run_id,
            peer_limit=peer_limit,
            min_peers=min_peers_dossierable,
            limit_dossiers=limit_dossiers,
            years_back=years_back,
            workers=workers,
            iterations=iterations,
            top_k=top_k,
            with_research=with_research,
            with_synthesis=with_synthesis,
            with_prices=with_prices,
            budget_usd=budget_usd,
            resume=resume,
            mode=mode_norm,
            force_restart=force_restart,
            gate_mos_min=gate_mos_min,
            gate_valuation_gap_min=gate_valuation_gap_min,
            gate_net_debt_to_cfo_max=gate_net_debt_to_cfo_max,
            gate_dilution_max=gate_dilution_max,
            timeout_per_stage=timeout_per_stage,
        )
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command("sector-rlm-status")
def sector_rlm_status_cmd(
    run_id: str = typer.Option(..., "--run-id", help="RLM run id"),
) -> None:
    configure_logging()
    from app.rlm.loop import sector_rlm_status

    init_db(get_config())
    payload = sector_rlm_status(run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))


@app.command("sector-rlm-depth")
def sector_rlm_depth_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector name"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic run id"),
    peer_limit: int = typer.Option(25, "--peer-limit", help="Peer cap for baseline sector-cycle"),
    min_peers_dossierable: int = typer.Option(
        15,
        "--min-peers-dossierable",
        "--min-peers",
        help="Minimum dossierable peers target for baseline cycle",
    ),
    limit_dossiers: int = typer.Option(25, "--limit-dossiers", help="Baseline dossier cap"),
    years_back: int = typer.Option(10, "--years-back", help="Years-back for dossier collection"),
    workers: int = typer.Option(4, "--workers", help="Worker count"),
    iterations: int = typer.Option(2, "--iterations", help="Maximum recursion iterations"),
    top_k: int = typer.Option(5, "--top-k", help="Top-K candidates tracked by critic"),
    with_research: bool = typer.Option(
        True, "--with-research/--no-research", help="Enable CLOSE_GAPS actions"
    ),
    with_synthesis: bool = typer.Option(
        True, "--with-synthesis/--no-synthesis", help="Enable RUN_SYNTHESIS actions"
    ),
    with_prices: bool = typer.Option(
        True, "--with-prices/--no-with-prices", help="Enable price snapshots for valuation"
    ),
    llm_provider: str | None = typer.Option(
        None, "--llm-provider", help="Override VOE_LLM_PROVIDER"
    ),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="LLM budget cap for this loop run"
    ),
    gate_mos_min: float | None = typer.Option(
        None, "--gate-mos-min", help="Optional value-gate PASS MOS threshold"
    ),
    gate_valuation_gap_min: float | None = typer.Option(
        None, "--gate-valuation-gap-min", help="Optional value-gate WATCH valuation-gap threshold"
    ),
    gate_net_debt_to_cfo_max: float | None = typer.Option(
        None,
        "--gate-net-debt-to-cfo-max",
        help="Optional value-gate PASS net-debt/CFO max threshold",
    ),
    gate_dilution_max: float | None = typer.Option(
        None, "--gate-dilution-max", help="Optional value-gate PASS dilution max threshold"
    ),
    timeout_per_stage: float = typer.Option(
        30.0, "--timeout-per-stage", help="Per-stage execution timeout in seconds"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing rlm_state.json"),
    force_restart: bool = typer.Option(
        False, "--force-restart", help="Wipe this run_id state/artifacts and restart"
    ),
) -> None:
    sector_rlm_cmd(
        sector=sector,
        as_of=as_of,
        run_id=run_id,
        peer_limit=peer_limit,
        min_peers_dossierable=min_peers_dossierable,
        limit_dossiers=limit_dossiers,
        years_back=years_back,
        workers=workers,
        iterations=iterations,
        top_k=top_k,
        with_research=with_research,
        with_synthesis=with_synthesis,
        with_prices=with_prices,
        mode="depth",
        llm_provider=llm_provider,
        budget_usd=budget_usd,
        gate_mos_min=gate_mos_min,
        gate_valuation_gap_min=gate_valuation_gap_min,
        gate_net_debt_to_cfo_max=gate_net_debt_to_cfo_max,
        gate_dilution_max=gate_dilution_max,
        timeout_per_stage=timeout_per_stage,
        resume=resume,
        force_restart=force_restart,
    )


@app.command("sector-rlm-tail")
def sector_rlm_tail_cmd(
    run_id: str = typer.Option(..., "--run-id", help="RLM run id"),
    interval: float = typer.Option(2.0, "--interval", help="Polling interval seconds"),
    lines: int = typer.Option(40, "--lines", help="Log lines to tail when rlm.log exists"),
    once: bool = typer.Option(False, "--once", help="Print one snapshot and exit"),
) -> None:
    configure_logging()
    from app.rlm.loop import sector_rlm_tail

    init_db(get_config())
    sector_rlm_tail(run_id=run_id, interval=interval, lines=lines, once=once)


@app.command("sector-rlm-resume")
def sector_rlm_resume_cmd(
    run_id: str = typer.Option(..., "--run-id", help="RLM run id"),
    iterations: int | None = typer.Option(
        None, "--iterations", help="Optional override max iterations"
    ),
    top_k: int | None = typer.Option(None, "--top-k", help="Optional override top-k"),
    workers: int | None = typer.Option(None, "--workers", help="Optional override workers"),
    mode: str | None = typer.Option(None, "--mode", help="Optional mode override: auto|depth"),
    timeout_per_stage: float = typer.Option(
        30.0, "--timeout-per-stage", help="Per-stage execution timeout in seconds"
    ),
    force_restart: bool = typer.Option(
        False, "--force-restart", help="Wipe this run_id state/artifacts and restart"
    ),
) -> None:
    configure_logging()
    from app.rlm.loop import resume_sector_rlm_loop

    init_db(get_config())
    try:
        payload = resume_sector_rlm_loop(
            run_id=run_id,
            iterations=iterations,
            top_k=top_k,
            workers=workers,
            mode=mode,
            force_restart=force_restart,
            timeout_per_stage=timeout_per_stage,
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2))


@app.command("sector-rlm-cancel")
def sector_rlm_cancel_cmd(
    run_id: str = typer.Option(..., "--run-id", help="RLM run id"),
    reason: str = typer.Option("cancelled by operator", "--reason", help="Cancellation reason"),
) -> None:
    configure_logging()
    from app.rlm.loop import cancel_sector_rlm_run

    init_db(get_config())
    payload = cancel_sector_rlm_run(run_id=run_id, reason=reason)
    code = 0 if payload.get("status") != "MISSING" else 1
    typer.echo(json.dumps(payload, indent=2))
    if code != 0:
        raise typer.Exit(code=code)


@app.command("sector-rlm-open")
def sector_rlm_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="RLM run id"),
) -> None:
    configure_logging()
    from app.rlm.loop import sector_rlm_open

    init_db(get_config())
    payload = sector_rlm_open(run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))


@app.command("sector-scoreboard-open")
def sector_scoreboard_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    from app.sector.cycle import sector_scoreboard_open

    payload = sector_scoreboard_open(run_id=run_id)
    typer.echo(json.dumps(payload, indent=2))


@app.command("sector-scoreboard-compare")
def sector_scoreboard_compare_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
    metric: str = typer.Option(..., "--metric", help="Metric key from peer_scoreboard.json"),
) -> None:
    configure_logging()
    from app.sector.cycle import sector_scoreboard_compare

    payload = sector_scoreboard_compare(run_id=run_id, metric=metric)
    typer.echo(json.dumps(payload, indent=2))


@app.command("value-gates-open")
def value_gates_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    from app.valuation.value_gates import open_value_gates_for_run

    payload = open_value_gates_for_run(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("value-gates-calibration-open")
def value_gates_calibration_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    from app.valuation.value_gates import open_value_gates_calibration_for_run

    payload = open_value_gates_calibration_for_run(run_id=run_id, top_n=10)
    typer.echo(json.dumps(payload, indent=2))
    if payload.get("status") != "OK":
        raise typer.Exit(code=1)


@app.command("price-coverage-open")
def price_coverage_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    from app.market.price_provider import price_reason_suggestion

    cfg = get_config()
    path = cfg.sectors_dir / run_id / "price_coverage.json"
    if not path.exists():
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "MISSING",
                    "price_coverage_path": str(path),
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]

    def _ticker_suggestions(reason_code: str) -> list[str]:
        code = str(reason_code or "").upper()
        if code in {"SYMBOL_UNMAPPED", "SYMBOL_NOT_FOUND"}:
            return ["Add symbol override entry to config/price_symbol_overrides.csv."]
        if code == "NON_TRADING_DAY_NO_FALLBACK":
            return ["Increase fallback-days to walk back to a prior trading day."]
        if code == "OFFLINE_NO_CACHE":
            return [
                "Run companyfacts-fetch and price-fetch in an online environment once to seed caches, then rerun depth."
            ]
        if code in {"DNS_FAILURE", "TLS_FAILURE", "TIMEOUT"}:
            return ["Retry later, check connectivity, or increase fallback-days."]
        if code == "RATE_LIMIT":
            return ["Reduce workers or increase budget."]
        if code in {"PROVIDER_NO_DATA", "HTTP_4XX", "HTTP_5XX", "PARSE_ERROR"}:
            return ["Inspect provider_attempts and retry with explicit ticker prewarm."]
        if code == "BUDGET_EXHAUSTED":
            return ["Increase host/domain request budget and retry."]
        fallback = price_reason_suggestion(code)
        return [fallback or "Inspect provider_attempts and cache metadata for this ticker."]

    counts: dict[str, int] = {}
    unknown_rows: list[dict[str, Any]] = []
    for row in entries:
        result = row.get("result") or {}
        code = str(result.get("reason_code") or "UNKNOWN")
        counts[code] = counts.get(code, 0) + 1
        if str(result.get("status") or "UNKNOWN").upper() != "OK":
            row_suggestions = [
                str(item) for item in (row.get("suggestions") or []) if str(item).strip()
            ]
            if not row_suggestions:
                row_suggestions = _ticker_suggestions(code)
            unknown_rows.append(
                {
                    "ticker": str(row.get("ticker") or ""),
                    "reason_code": code,
                    "reason_detail": str(result.get("reason_detail") or ""),
                    "suggestions": row_suggestions,
                }
            )
    unknown_rows = sorted(unknown_rows, key=lambda row: row["ticker"])

    suggestions: list[str] = []
    reason_keys = set(counts.keys())
    if "SYMBOL_UNMAPPED" in reason_keys or "SYMBOL_NOT_FOUND" in reason_keys:
        suggestions.append(
            "Add/verify symbol overrides in config/price_symbol_overrides.csv and rerun price-fetch."
        )
    if "NON_TRADING_DAY_NO_FALLBACK" in reason_keys:
        suggestions.append("Increase --fallback-days in price-fetch and sector run valuation path.")
    if "CACHE_MISS" in reason_keys or "HTTP_4XX" in reason_keys or "HTTP_5XX" in reason_keys:
        suggestions.append("Run price-fetch first to prewarm cache, then rerun sector depth.")
    if "DNS_FAILURE" in reason_keys or "TLS_FAILURE" in reason_keys or "TIMEOUT" in reason_keys:
        suggestions.append("Retry later and check host connectivity or DNS/TLS environment.")
    if "RATE_LIMIT" in reason_keys:
        suggestions.append("Reduce concurrent work or increase host/domain request budget.")
    if "PARSE_ERROR" in reason_keys:
        suggestions.append(
            "Inspect provider_attempts.url and response parsing assumptions for symbol/date format."
        )
    if "OFFLINE_NO_CACHE" in reason_keys:
        suggestions.append(
            "Run companyfacts-fetch and price-fetch in an online environment once to seed caches, then rerun depth."
        )
    if "BUDGET_EXHAUSTED" in reason_keys:
        suggestions.append("Increase request budget or retry after budget reset.")
    if not suggestions and unknown_rows:
        suggestions.append(
            "Inspect provider_attempts and cache metadata per ticker in price_coverage.json."
        )

    out = {
        "run_id": run_id,
        "price_coverage_path": str(path),
        "ticker_count": len(entries),
        "reason_counts": dict(sorted(counts.items(), key=lambda kv: kv[0])),
        "unknown_tickers": unknown_rows,
        "suggestions": suggestions,
    }
    typer.echo(json.dumps(out, indent=2))


@app.command("shares-coverage-open")
def shares_coverage_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    cfg = get_config()
    path = cfg.sectors_dir / run_id / "shares_coverage.json"
    if not path.exists():
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "MISSING",
                    "shares_coverage_path": str(path),
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    counts: dict[str, int] = {}
    unknown_rows: list[dict[str, str]] = []
    for row in entries:
        code = str(row.get("shares_reason_code") or "UNKNOWN")
        counts[code] = counts.get(code, 0) + 1
        if str(row.get("shares_status") or "UNKNOWN").upper() != "OK":
            unknown_rows.append(
                {
                    "ticker": str(row.get("ticker") or ""),
                    "shares_reason_code": code,
                    "shares_reason_detail": str(row.get("shares_reason_detail") or ""),
                }
            )
    unknown_rows = sorted(unknown_rows, key=lambda row: row["ticker"])

    valuation_cov_path = cfg.sectors_dir / run_id / "valuation_coverage.json"
    valuation_cov_payload = (
        json.loads(valuation_cov_path.read_text(encoding="utf-8"))
        if valuation_cov_path.exists()
        else {}
    )
    valuation_entries = [
        row for row in (valuation_cov_payload.get("entries") or []) if isinstance(row, dict)
    ]
    price_ok_shares_unknown = sorted(
        [
            {
                "ticker": str(row.get("ticker") or ""),
                "price_reason_code": str(row.get("price_reason_code") or ""),
                "shares_reason_code": str(row.get("shares_reason_code") or ""),
            }
            for row in valuation_entries
            if str(row.get("price_status") or "UNKNOWN").upper() == "OK"
            and str(row.get("shares_status") or "UNKNOWN").upper() != "OK"
        ],
        key=lambda row: row["ticker"],
    )

    suggestions: list[str] = []
    reason_keys = set(counts.keys())
    if "NO_CURRENT_RUN_SHARES" in reason_keys:
        suggestions.append(
            "Inspect fundamentals_<TICKER>.json and dossier.json for missing shares_outstanding_latest values."
        )
    if "NO_HISTORICAL_DOSSIER" in reason_keys:
        suggestions.append(
            "Check historical dossier fallback selection (closest as_of <= requested) and dossier availability."
        )
    if "COMPANYFACTS_MISS" in reason_keys:
        suggestions.append(
            "Run companyfacts-fetch and inspect facts_coverage/companyfacts cache for missing shares tags."
        )
    if "HISTORICAL_DOSSIER_HIT" in reason_keys:
        suggestions.append(
            "Inspect selected historical dossier sources to confirm shares freshness."
        )
    if "DERIVED_FROM_MKTCAP_PRICE" in reason_keys:
        suggestions.append(
            "Review market_cap and price traces for derived shares accuracy before enabling globally."
        )
    if "EXCEPTION" in reason_keys:
        suggestions.append(
            "Inspect resolver exception details and derived_from pointers in shares_coverage.json."
        )
    if not suggestions and unknown_rows:
        suggestions.append(
            "Inspect derived_from pointers in shares_coverage.json and valuation_coverage.json for ticker-level triage."
        )

    out = {
        "run_id": run_id,
        "shares_coverage_path": str(path),
        "ticker_count": len(entries),
        "reason_counts": dict(sorted(counts.items(), key=lambda kv: kv[0])),
        "unknown_tickers": unknown_rows,
        "price_ok_but_shares_unknown": price_ok_shares_unknown,
        "suggestions": suggestions,
    }
    typer.echo(json.dumps(out, indent=2))


@app.command("fcf-coverage-open")
def fcf_coverage_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    cfg = get_config()
    path = cfg.sectors_dir / run_id / "fcf_coverage.json"
    if not path.exists():
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "MISSING",
                    "fcf_coverage_path": str(path),
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    counts: dict[str, int] = {}
    unknown_rows: list[dict[str, str]] = []
    for row in entries:
        code = str(row.get("fcf_reason_code") or "UNKNOWN")
        counts[code] = counts.get(code, 0) + 1
        if str(row.get("fcf_status") or "UNKNOWN").upper() != "OK":
            unknown_rows.append(
                {
                    "ticker": str(row.get("ticker") or ""),
                    "fcf_reason_code": code,
                    "fcf_reason_detail": str(row.get("fcf_reason_detail") or ""),
                }
            )
    unknown_rows = sorted(unknown_rows, key=lambda row: row["ticker"])

    valuation_cov_path = cfg.sectors_dir / run_id / "valuation_coverage.json"
    valuation_cov_payload = (
        json.loads(valuation_cov_path.read_text(encoding="utf-8"))
        if valuation_cov_path.exists()
        else {}
    )
    valuation_entries = [
        row for row in (valuation_cov_payload.get("entries") or []) if isinstance(row, dict)
    ]
    price_ok_fcf_unknown = sorted(
        [
            {
                "ticker": str(row.get("ticker") or ""),
                "price_reason_code": str(row.get("price_reason_code") or ""),
                "fcf_reason_code": str(row.get("fcf_reason_code") or ""),
            }
            for row in valuation_entries
            if str(row.get("price_status") or "UNKNOWN").upper() == "OK"
            and str(row.get("fcf_status") or "UNKNOWN").upper() != "OK"
        ],
        key=lambda row: row["ticker"],
    )

    suggestions: list[str] = []
    reason_keys = set(counts.keys())
    if "DOSSIER_NO_FCF" in reason_keys:
        suggestions.append(
            "Inspect dossier_debug_<TICKER>.json and dossier time_series for missing fcf."
        )
    if "MISSING_CFO" in reason_keys:
        suggestions.append(
            "Verify CFO extraction traces before attempting derived fcf = cfo - capex."
        )
    if "MISSING_CAPEX" in reason_keys:
        suggestions.append(
            "Verify capex extraction traces before attempting derived fcf = cfo - capex."
        )
    if "NO_HISTORICAL_DOSSIER" in reason_keys:
        suggestions.append(
            "Check historical dossier fallback selection for as_of windows and run ordering."
        )
    if "COMPANYFACTS_MISS" in reason_keys:
        suggestions.append(
            "Run companyfacts-fetch and inspect facts_coverage for missing CFO/capex tags."
        )
    if not suggestions and unknown_rows:
        suggestions.append(
            "Inspect derived_from pointers in fcf_coverage.json and valuation_coverage.json."
        )

    out = {
        "run_id": run_id,
        "fcf_coverage_path": str(path),
        "ticker_count": len(entries),
        "reason_counts": dict(sorted(counts.items(), key=lambda kv: kv[0])),
        "unknown_tickers": unknown_rows,
        "price_ok_but_fcf_unknown": price_ok_fcf_unknown,
        "suggestions": suggestions,
    }
    typer.echo(json.dumps(out, indent=2))


@app.command("facts-coverage-open")
def facts_coverage_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    cfg = get_config()
    path = cfg.sectors_dir / run_id / "facts_coverage.json"
    if not path.exists():
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "MISSING",
                    "facts_coverage_path": str(path),
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]

    shares_ok = len([row for row in entries if str(row.get("shares_status") or "").upper() == "OK"])
    cfo_ok = len([row for row in entries if str(row.get("cfo_status") or "").upper() == "OK"])
    capex_ok = len([row for row in entries if str(row.get("capex_status") or "").upper() == "OK"])
    fcf_ok = len([row for row in entries if str(row.get("fcf_status") or "").upper() == "OK"])

    unknown_rows = sorted(
        [
            {
                "ticker": str(row.get("ticker") or ""),
                "status": str(row.get("status") or "UNKNOWN"),
                "shares_reason": str(row.get("shares_reason") or ""),
                "cfo_reason": str(row.get("cfo_reason") or ""),
                "capex_reason": str(row.get("capex_reason") or ""),
                "fcf_reason": str(row.get("fcf_reason") or ""),
            }
            for row in entries
            if str(row.get("status") or "UNKNOWN").upper() != "OK"
        ],
        key=lambda row: row["ticker"],
    )

    reason_counts = (
        payload.get("reason_counts") if isinstance(payload.get("reason_counts"), dict) else {}
    )
    shares_reasons = (
        reason_counts.get("shares_reason")
        if isinstance(reason_counts.get("shares_reason"), dict)
        else {}
    )
    fcf_reasons = (
        reason_counts.get("fcf_reason") if isinstance(reason_counts.get("fcf_reason"), dict) else {}
    )
    cfo_reasons = (
        reason_counts.get("cfo_reason") if isinstance(reason_counts.get("cfo_reason"), dict) else {}
    )
    capex_reasons = (
        reason_counts.get("capex_reason")
        if isinstance(reason_counts.get("capex_reason"), dict)
        else {}
    )

    suggestions: list[str] = []
    if int(shares_reasons.get("CIK_MISSING", 0)) > 0:
        suggestions.append(
            "CIK missing for one or more tickers; verify companies/universe CIK mapping before hydration."
        )
    if (
        int(shares_reasons.get("OFFLINE_NO_CACHE", 0)) > 0
        or int(fcf_reasons.get("OFFLINE_NO_CACHE", 0)) > 0
    ):
        suggestions.append(
            "Run companyfacts-fetch in online-enabled mode (or seed data/cache/companyfacts) before depth hydration."
        )
    if (
        int(shares_reasons.get("TAG_MISS", 0)) > 0
        or int(cfo_reasons.get("TAG_MISS", 0)) > 0
        or int(capex_reasons.get("TAG_MISS", 0)) > 0
    ):
        suggestions.append(
            "Inspect selected companyfacts tags/end dates for missing fields and fallback tag priority coverage."
        )
    if not suggestions and unknown_rows:
        suggestions.append(
            "Inspect facts_coverage derived_from pointers and per-ticker fetch_reason_code details."
        )

    out = {
        "run_id": run_id,
        "facts_coverage_path": str(path),
        "ticker_count": len(entries),
        "shares_ok_count": int(shares_ok),
        "cfo_ok_count": int(cfo_ok),
        "capex_ok_count": int(capex_ok),
        "fcf_ok_count": int(fcf_ok),
        "facts_blocker_histogram": payload.get("facts_blocker_histogram")
        if isinstance(payload.get("facts_blocker_histogram"), dict)
        else {},
        "retryable_facts_blocker_count": int(payload.get("retryable_facts_blocker_count") or 0),
        "terminal_facts_blocker_count": int(payload.get("terminal_facts_blocker_count") or 0),
        "partial_usable_facts_count": int(payload.get("partial_usable_facts_count") or 0),
        "top_retryable_facts_blockers": [
            row
            for row in (payload.get("top_retryable_facts_blockers") or [])
            if isinstance(row, dict)
        ][:10],
        "top_terminal_facts_blockers": [
            row
            for row in (payload.get("top_terminal_facts_blockers") or [])
            if isinstance(row, dict)
        ][:10],
        "top_partial_usable_facts": [
            row for row in (payload.get("top_partial_usable_facts") or []) if isinstance(row, dict)
        ][:10],
        "recommended_next_action_counts": payload.get("recommended_next_action_counts")
        if isinstance(payload.get("recommended_next_action_counts"), dict)
        else {},
        "economic_fail_count_vs_evidence_fail_count": payload.get(
            "economic_fail_count_vs_evidence_fail_count"
        )
        if isinstance(payload.get("economic_fail_count_vs_evidence_fail_count"), dict)
        else {},
        "reason_counts": {
            "shares_reason": dict(sorted(shares_reasons.items(), key=lambda kv: kv[0])),
            "cfo_reason": dict(sorted(cfo_reasons.items(), key=lambda kv: kv[0])),
            "capex_reason": dict(sorted(capex_reasons.items(), key=lambda kv: kv[0])),
            "fcf_reason": dict(sorted(fcf_reasons.items(), key=lambda kv: kv[0])),
        },
        "unknown_tickers": unknown_rows,
        "suggestions": suggestions,
    }
    typer.echo(json.dumps(out, indent=2))


@app.command("valuation-coverage-open")
def valuation_coverage_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
) -> None:
    configure_logging()
    cfg = get_config()
    path = cfg.sectors_dir / run_id / "valuation_coverage.json"
    if not path.exists():
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "MISSING",
                    "valuation_coverage_path": str(path),
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    counts: dict[str, int] = {}
    price_unknown_reason_counts: dict[str, int] = {}
    price_ok_valuation_unknown: list[dict[str, str]] = []
    for row in entries:
        code = str(row.get("valuation_reason_code") or "OK")
        counts[code] = counts.get(code, 0) + 1
        price_status = str(row.get("price_status") or "UNKNOWN").upper()
        price_reason_code = str(row.get("price_reason_code") or "UNKNOWN")
        if price_status != "OK" or code == "PRICE_UNKNOWN":
            price_unknown_reason_counts[price_reason_code] = (
                price_unknown_reason_counts.get(price_reason_code, 0) + 1
            )
        if (
            str(row.get("price_status") or "UNKNOWN").upper() == "OK"
            and str(row.get("valuation_status") or "UNKNOWN").upper() != "OK"
        ):
            price_ok_valuation_unknown.append(
                {
                    "ticker": str(row.get("ticker") or ""),
                    "valuation_reason_code": str(
                        row.get("valuation_reason_code") or "MODEL_PRECONDITION_FAILED"
                    ),
                    "price_reason_code": str(row.get("price_reason_code") or ""),
                }
            )
    price_ok_valuation_unknown = sorted(price_ok_valuation_unknown, key=lambda row: row["ticker"])

    suggestions: list[str] = []
    reason_keys = set(counts.keys())
    if "MISSING_FCF" in reason_keys:
        suggestions.append(
            "Run HYDRATE_FINANCIAL_FACTS or inspect fcf_coverage/facts_coverage for missing CFO/capex tags."
        )
    if "MISSING_SHARES" in reason_keys:
        suggestions.append(
            "Run companyfacts-fetch and inspect shares_coverage/facts_coverage for missing shares tags."
        )
    if "MISSING_NET_DEBT" in reason_keys:
        suggestions.append("Inspect fundamentals_<T>.json for net_debt completeness.")
    if "PRICE_UNKNOWN" in reason_keys:
        suggestions.append("Run price-fetch first or verify price cache/provider coverage.")
        price_reason_keys = set(price_unknown_reason_counts.keys())
        if any(key.startswith("NON_TRADING_DAY_") for key in price_reason_keys):
            suggestions.append(
                "Run price-fetch with --fallback-days N (for example 5-7) to walk back to the nearest trading day."
            )
        if "SYMBOL_UNMAPPED" in price_reason_keys or "SYMBOL_NOT_FOUND" in price_reason_keys:
            suggestions.append(
                "Add/verify symbol overrides in config/price_symbol_overrides.csv for unresolved tickers."
            )
        if "OFFLINE_NO_CACHE" in price_reason_keys:
            suggestions.append(
                "Run companyfacts-fetch and price-fetch in an online environment once to seed caches, then rerun depth."
            )
        if {"DNS_FAILURE", "TLS_FAILURE", "TIMEOUT"} & price_reason_keys:
            suggestions.append(
                "Retry later and validate connectivity to price providers (DNS/TLS/timeout)."
            )
        if "RATE_LIMIT" in price_reason_keys:
            suggestions.append("Reduce workers or increase provider request budget.")
    if "INVALID_DENOMINATOR" in reason_keys:
        suggestions.append(
            "Check for zero/negative shares_outstanding or invalid price in valuation inputs."
        )
    if "ENGINE_EXCEPTION" in reason_keys:
        suggestions.append(
            "Inspect valuation_<T>.json warnings and stack logs for engine exceptions."
        )
    if not suggestions and price_ok_valuation_unknown:
        suggestions.append(
            "Inspect valuation_coverage derived_from pointers for failing preconditions."
        )

    out = {
        "run_id": run_id,
        "valuation_coverage_path": str(path),
        "ticker_count": len(entries),
        "reason_counts": dict(sorted(counts.items(), key=lambda kv: kv[0])),
        "price_unknown_reason_counts": dict(
            sorted(price_unknown_reason_counts.items(), key=lambda kv: kv[0])
        ),
        "price_ok_but_valuation_unknown": price_ok_valuation_unknown,
        "suggestions": suggestions,
    }
    typer.echo(json.dumps(out, indent=2))


@app.command("sector-artifact-open")
def sector_artifact_open_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
    kind: str = typer.Option(..., "--kind", help="Artifact kind: fundamentals|valuation"),
    ticker: str = typer.Option(..., "--ticker", help="Ticker symbol"),
    head: int = typer.Option(120, "--head", help="Print first N lines"),
) -> None:
    configure_logging()
    cfg = get_config()
    kind_norm = kind.strip().lower()
    if kind_norm not in {"fundamentals", "valuation"}:
        raise typer.BadParameter("--kind must be fundamentals or valuation")
    ticker_norm = ticker.strip().upper()
    if not ticker_norm:
        raise typer.BadParameter("--ticker is required")

    sector_dir = cfg.sectors_dir / run_id
    filename = f"{kind_norm}_{ticker_norm}.json"
    path = sector_dir / filename
    if not path.exists():
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "kind": kind_norm,
                    "ticker": ticker_norm,
                    "status": "MISSING",
                    "path": str(path),
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)

    payload: dict[str, object] = {
        "run_id": run_id,
        "kind": kind_norm,
        "ticker": ticker_norm,
        "status": "OK",
        "path": str(path),
    }
    lines = []
    if int(head) > 0:
        content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = content[: int(head)]
        payload["head_lines"] = len(lines)
        payload["head"] = lines
    typer.echo(json.dumps(payload, indent=2))


@app.command("discovery-ledger-init")
def discovery_ledger_init_cmd() -> None:
    configure_logging()
    from app.discovery.ledger import discovery_ledger_init

    summary = discovery_ledger_init()
    typer.echo(json.dumps(summary, indent=2))


@app.command("discovery-ledger-update")
def discovery_ledger_update_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Discovery run id"),
) -> None:
    configure_logging()
    from app.discovery.ledger import discovery_ledger_update

    init_db(get_config())
    summary = discovery_ledger_update(run_id)
    typer.echo(json.dumps(summary, indent=2))


@app.command("outcome-add")
def outcome_add_cmd(
    ticker: str = typer.Option(..., "--ticker", help="Ticker symbol"),
    run_id: str = typer.Option(..., "--run-id", help="Run identifier"),
    decision: str = typer.Option(..., "--decision", help="BUY|WATCH|SHORT|PASS"),
    conviction: int = typer.Option(..., "--conviction", help="Conviction 1-5"),
    horizon_days: int = typer.Option(..., "--horizon-days", help="Expected horizon in days"),
    notes: str = typer.Option("", "--notes", help="Optional short notes"),
    as_of: str | None = typer.Option(
        None, "--as-of", help="As-of date YYYY-MM-DD (defaults to run manifest date or today)"
    ),
    discovery_run_id: str | None = typer.Option(
        None, "--discovery-run-id", help="Optional linked discovery run"
    ),
    deep_run_id: str | None = typer.Option(None, "--deep-run-id", help="Optional linked deep run"),
    thesis_tags: str | None = typer.Option(
        None, "--thesis-tags", help="Optional comma-separated tags"
    ),
) -> None:
    configure_logging()
    from app.outcomes.store import add_outcome

    as_of_date = parse_yyyy_mm_dd(as_of).isoformat() if as_of else date.today().isoformat()
    row = add_outcome(
        ticker=ticker,
        as_of_date=as_of_date,
        run_id=run_id,
        decision=decision,
        conviction=conviction,
        horizon_days=horizon_days,
        notes=notes,
        thesis_tags=[x.strip() for x in (thesis_tags or "").split(",") if x.strip()],
        discovery_run_id=discovery_run_id,
        deep_run_id=deep_run_id,
    )
    typer.echo(json.dumps(row, indent=2))


@app.command("outcome-close")
def outcome_close_cmd(
    ticker: str = typer.Option(..., "--ticker", help="Ticker symbol"),
    run_id: str = typer.Option(..., "--run-id", help="Run identifier"),
    realized_return: float = typer.Option(..., "--realized-return", help="Realized return percent"),
    close_date: str = typer.Option(..., "--close-date", help="Close date YYYY-MM-DD"),
    max_dd: float | None = typer.Option(None, "--max-dd", help="Optional max drawdown percent"),
) -> None:
    configure_logging()
    from app.outcomes.store import close_outcome

    parse_yyyy_mm_dd(close_date)
    row = close_outcome(
        ticker=ticker,
        run_id=run_id,
        realized_return_pct=realized_return,
        close_date=close_date,
        max_drawdown_pct=max_dd,
    )
    typer.echo(json.dumps(row, indent=2))


@app.command("outcome-list")
def outcome_list_cmd(
    open_only: bool = typer.Option(False, "--open-only", help="Only show OPEN outcomes"),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run filter"),
    ticker: str | None = typer.Option(None, "--ticker", help="Optional ticker filter"),
    limit: int = typer.Option(200, "--limit", help="Row limit"),
) -> None:
    configure_logging()
    from app.outcomes.store import list_outcomes

    rows = list_outcomes(open_only=open_only, run_id=run_id, ticker=ticker, limit=limit)
    typer.echo(json.dumps({"count": len(rows), "rows": rows}, indent=2))


@app.command("recommendation-list")
def recommendation_list_cmd(
    ticker: str | None = typer.Option(None, "--ticker", help="Optional ticker filter"),
    recommendation_type: str | None = typer.Option(
        None,
        "--type",
        help="Optional BUY_AT_LIMIT, WATCH, or AVOID filter",
    ),
    vintage: str | None = typer.Option(
        None,
        "--vintage",
        help="Optional LIVE or SEED filter",
    ),
    limit: int = typer.Option(200, "--limit", min=1, max=1000, help="Row limit"),
) -> None:
    """Read the immutable recommendation ledger."""

    configure_logging()
    from app.calibration.recommendation_ledger import list_recommendations

    rows = list_recommendations(
        ticker=ticker,
        recommendation_type=recommendation_type,
        record_vintage=vintage,
        limit=limit,
    )
    typer.echo(json.dumps({"count": len(rows), "rows": rows}, indent=2))


@app.command("recommendation-show")
def recommendation_show_cmd(
    recommendation_id: str = typer.Argument(..., help="Immutable recommendation id"),
) -> None:
    """Read one recommendation by immutable id."""

    configure_logging()
    from app.calibration.recommendation_ledger import get_recommendation

    row = get_recommendation(recommendation_id)
    if row is None:
        raise typer.BadParameter(f"recommendation not found: {recommendation_id}")
    typer.echo(json.dumps(row, indent=2))


@app.command("recommendation-seed-preview")
def recommendation_seed_preview_cmd(
    limit: int = typer.Option(200, "--limit", min=1, max=1000, help="Preview row limit"),
) -> None:
    """Derive the historical seed corpus without writing any row."""

    configure_logging()
    from app.calibration.recommendation_ledger import preview_open_at_target_seed

    rows = preview_open_at_target_seed()
    type_counts: dict[str, int] = {}
    for row in rows:
        recommendation_type = str(row["recommendation_type"])
        type_counts[recommendation_type] = type_counts.get(recommendation_type, 0) + 1
    typer.echo(
        json.dumps(
            {
                "source_rows": len(rows),
                "record_vintage": "SEED",
                "type_counts": type_counts,
                "preview_count": min(len(rows), limit),
                "rows": rows[:limit],
            },
            indent=2,
        )
    )


@app.command("calibration-run")
def calibration_run_cmd(
    run_id: str | None = typer.Option(
        None, "--run-id", help="Optional target deep/discovery run id"
    ),
    last_n: int = typer.Option(
        5, "--last-n", help="Number of recent discovery runs when --run-id omitted"
    ),
    as_of: str | None = typer.Option(
        None, "--as-of", help="Resolve due perception outcomes through YYYY-MM-DD"
    ),
    prices: str | None = typer.Option(
        None, "--prices", help="Optional manual price changes: TICKER=12.5,..."
    ),
) -> None:
    configure_logging()
    init_db(get_config())
    if as_of:
        parse_yyyy_mm_dd(as_of)
        from app.calibration.weight_registry import (
            compute_calibration_weights,
            write_calibration_weights,
        )
        from app.calibration.outcome_resolver import resolve_all_due

        price_data = _parse_price_changes_csv(prices)
        resolution_summary = resolve_all_due(as_of, price_data=price_data)
        weights = compute_calibration_weights()
        weights_path = write_calibration_weights(weights)
        typer.echo(
            json.dumps(
                {
                    "as_of_date": as_of,
                    "resolution_summary": resolution_summary,
                    "weights": weights.model_dump(mode="json"),
                    "weights_path": str(weights_path),
                },
                indent=2,
            )
        )
        return

    from app.discovery.calibration import run_calibration

    summary = run_calibration(run_id=run_id, last_n=last_n)
    typer.echo(json.dumps(summary, indent=2))


@app.command("calibration-open-latest")
def calibration_open_latest_cmd() -> None:
    configure_logging()
    from app.discovery.calibration import latest_calibration_report

    init_db(get_config())
    report = latest_calibration_report()
    if not report:
        typer.echo("No calibration reports found")
        raise typer.Exit(code=1)
    typer.echo(json.dumps(report, indent=2))


@app.command("calibration-grade-report-open")
def calibration_grade_report_open_cmd() -> None:
    """Open the latest grade/status calibration report (distinct from discovery calibration)."""
    configure_logging()
    from app.calibration.calibration_report import latest_grade_status_report

    init_db(get_config())
    report = latest_grade_status_report()
    if not report:
        typer.echo("No grade/status calibration reports found")
        raise typer.Exit(code=1)
    typer.echo(json.dumps(report, indent=2))


@app.command("calibration-report")
def calibration_report_cmd(
    as_of: str = typer.Option(
        ..., "--as-of", help="Report date YYYY-MM-DD (segments CLOSED outcomes by grade + status)"
    ),
) -> None:
    """Build the grade/status calibration report from CLOSED ticker_outcomes."""
    configure_logging()
    from app.calibration.calibration_report import build_calibration_report

    parse_yyyy_mm_dd(as_of)
    init_db(get_config())
    report = build_calibration_report(as_of)
    typer.echo(json.dumps(report, indent=2))


@app.command("calibration-backfill-prices")
def calibration_backfill_prices_cmd(
    benchmark: str = typer.Option(
        "SPY", "--benchmark", help="Benchmark symbol(s), comma-separated (default SPY)"
    ),
    horizon_days: int = typer.Option(
        365, "--horizon-days", help="Forward horizon used to derive the exit anchor date"
    ),
) -> None:
    """Backfill daily price history for non-removed watchlist tickers + benchmark.

    Anchors at each OPEN outcome's entry_date and entry_date + horizon_days.
    Network-bound (owner-run): uses the configured date-aware price provider.
    """
    configure_logging()
    from datetime import datetime, timedelta

    from app.market.price_history_backfill import backfill_daily_history
    from app.outcomes.lineage import outcome_row_is_decision_eligible
    from app.watchlist.lineage import watchlist_row_is_decision_eligible
    from app.watchlist.store import current_watchlist_cte

    init_db(get_config())

    benchmark_symbols = tuple(
        sym.strip().upper() for sym in benchmark.split(",") if sym.strip()
    ) or ("SPY",)

    tickers: set[str] = set()
    anchor_dates: set[str] = set()
    with get_db() as conn:
        for row in conn.execute(
            f"""
            WITH latest_watchlist AS ({current_watchlist_cte()})
            SELECT w.*
            FROM watchlist w
            JOIN latest_watchlist ON latest_watchlist.latest_id = w.id
            WHERE w.status != 'REMOVED'
            ORDER BY w.ticker
            """
        ).fetchall():
            if watchlist_row_is_decision_eligible(row):
                tickers.add(str(row["ticker"]).upper())
        for row in conn.execute(
            """
            WITH latest_lineage AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, run_id
                           ORDER BY updated_at DESC, id DESC
                       ) AS lineage_rank
                FROM ticker_outcomes
            )
            SELECT *
            FROM latest_lineage
            WHERE lineage_rank = 1
              AND outcome_status = 'OPEN'
              AND entry_date IS NOT NULL
            """
        ).fetchall():
            if not outcome_row_is_decision_eligible(row):
                continue
            entry_date = str(row["entry_date"])
            anchor_dates.add(entry_date)
            window = int(row["horizon_days"] or horizon_days)
            exit_anchor = (
                datetime.strptime(entry_date, "%Y-%m-%d").date() + timedelta(days=window)
            ).isoformat()
            anchor_dates.add(exit_anchor)

    summary = backfill_daily_history(
        tickers=sorted(tickers),
        anchor_dates=sorted(anchor_dates),
        benchmark_symbols=benchmark_symbols,
    )
    typer.echo(
        json.dumps(
            {
                "benchmark_symbols": list(benchmark_symbols),
                "horizon_days": horizon_days,
                "ticker_count": len(tickers),
                "anchor_date_count": len(anchor_dates),
                "fetched": summary.fetched,
                "rows_written": summary.rows_written,
                "failed_tickers": summary.failed_tickers,
            },
            indent=2,
        )
    )


@app.command("calibration-register")
def calibration_register_cmd(
    run_id: str = typer.Option(
        ..., "--run-id", help="Universe run id with variant perception reports"
    ),
    ticker: str | None = typer.Option(None, "--ticker", help="Optional single ticker filter"),
) -> None:
    configure_logging()
    from app.calibration.perception_tracker import (
        load_variant_reports_for_run,
        register_perceptions_from_report,
    )

    reports = load_variant_reports_for_run(run_id=run_id, ticker=ticker)
    total_registered = 0
    for report in reports:
        total_registered += register_perceptions_from_report(report)
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "ticker": str(ticker or "").upper() or None,
                "reports_processed": len(reports),
                "registered_count": total_registered,
            },
            indent=2,
        )
    )


@app.command("calibration-resolve")
def calibration_resolve_cmd(
    as_of: str = typer.Option(
        ..., "--as-of", help="Resolve matured OPEN ticker_outcomes through YYYY-MM-DD"
    ),
) -> None:
    """Close every matured OPEN ticker_outcomes row with realized/excess returns.

    (Perception/discovery outcome resolution remains available via
    ``calibration-run --as-of``.)
    """
    configure_logging()
    from app.calibration.return_resolver import resolve_open_outcomes

    parse_yyyy_mm_dd(as_of)
    init_db(get_config())
    summary = resolve_open_outcomes(as_of)
    typer.echo(
        json.dumps(
            {
                "as_of_date": as_of,
                "eligible": summary.eligible,
                "closed": summary.closed,
                "skipped_not_matured": summary.skipped_not_matured,
                "skipped_no_price": summary.skipped_no_price,
            },
            indent=2,
        )
    )


@app.command("calibration-status")
def calibration_status_cmd() -> None:
    configure_logging()
    from app.calibration.perception_tracker import (
        list_pending_perceptions,
        list_resolvable_perceptions,
    )

    today = date.today().isoformat()
    pending = list_pending_perceptions()
    due = list_resolvable_perceptions(today)
    typer.echo(
        json.dumps(
            {
                "as_of_date": today,
                "pending_count": len(pending),
                "due_now_count": len(due),
                "future_count": max(0, len(pending) - len(due)),
                "pending": [record.model_dump(mode="json") for record in pending],
            },
            indent=2,
        )
    )


@app.command("calibration-weights")
def calibration_weights_cmd(
    save: bool = typer.Option(
        False, "--save", help="Persist computed weights to calibration_weights.json"
    ),
) -> None:
    configure_logging()
    from app.calibration.weight_registry import (
        compute_calibration_weights,
        write_calibration_weights,
    )

    weights = compute_calibration_weights()
    payload = weights.model_dump(mode="json")
    if save:
        payload["weights_path"] = str(write_calibration_weights(weights))
    typer.echo(json.dumps(payload, indent=2))


@app.command("script-print")
def script_print_cmd(
    task: str = typer.Option(..., "--task", help="Task name (supports: discovery-run)"),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
    limit: int | None = typer.Option(None, "--limit", help="Optional ticker cap"),
    tickers: str | None = typer.Option(None, "--tickers", help="Optional explicit tickers list"),
    seed: Path | None = typer.Option(None, "--seed", help="Discovery seed CSV"),
    exclude: Path | None = typer.Option(None, "--exclude", help="Exclude CSV"),
    top: int | None = typer.Option(None, "--top", help="Top-K shortlist"),
    advance_top: int | None = typer.Option(
        None, "--advance-top", help="ADVANCE_TO_DEEP shortlist cap"
    ),
    mktcap_min: float | None = typer.Option(None, "--mktcap-min", help="Market cap minimum"),
    mktcap_max: float | None = typer.Option(None, "--mktcap-max", help="Market cap maximum"),
    phase: str | None = typer.Option(None, "--phase", help="Discovery phase: auto|prefilter|full"),
    prefilter_cap: int | None = typer.Option(None, "--prefilter-cap", help="Phase-2 prefilter cap"),
    prefilter_keep_ratio: float | None = typer.Option(
        None, "--prefilter-keep-ratio", help="Phase-1 keep ratio"
    ),
    workers: int | None = typer.Option(None, "--workers", help="Worker count"),
) -> None:
    task_name = task.strip().lower()
    if task_name != "discovery-run":
        raise typer.BadParameter("script-print currently supports only --task discovery-run")
    if not as_of:
        raise typer.BadParameter("--as-of is required for --task discovery-run")
    parse_yyyy_mm_dd(as_of)

    parts = ["python", "-m", "app.cli", "discovery-run", "--as-of", as_of]
    if limit is not None:
        parts.extend(["--limit", str(limit)])
    if tickers:
        parts.extend(["--tickers", tickers])
    if seed is not None:
        parts.extend(["--seed", str(seed)])
    if exclude is not None:
        parts.extend(["--exclude", str(exclude)])
    if top is not None:
        parts.extend(["--top", str(top)])
    if advance_top is not None:
        parts.extend(["--advance-top", str(advance_top)])
    if mktcap_min is not None:
        parts.extend(["--mktcap-min", str(mktcap_min)])
    if mktcap_max is not None:
        parts.extend(["--mktcap-max", str(mktcap_max)])
    if phase is not None:
        parts.extend(["--phase", phase])
    if prefilter_cap is not None:
        parts.extend(["--prefilter-cap", str(prefilter_cap)])
    if prefilter_keep_ratio is not None:
        parts.extend(["--prefilter-keep-ratio", str(prefilter_keep_ratio)])
    if workers is not None:
        parts.extend(["--workers", str(workers)])
    typer.echo(" ".join(shlex.quote(part) for part in parts))


@app.command("ingest-sec")
def ingest_sec(
    since: str = typer.Option(..., help="YYYY-MM-DD"),
    forms: str = typer.Option("10-K,10-Q,8-K"),
    as_of: str | None = typer.Option(None, help="Run as-of date YYYY-MM-DD"),
) -> None:
    configure_logging()
    from app.ingest.filings import ingest_since

    parse_yyyy_mm_dd(since)
    if as_of:
        parse_yyyy_mm_dd(as_of)
    inserted = ingest_since(parse_yyyy_mm_dd(since), _parse_forms(forms), as_of_date=as_of)
    typer.echo(f"Ingested {inserted} filings")


@app.command("backfill")
def backfill_cmd(
    from_date: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    to: str = typer.Option(..., help="YYYY-MM-DD"),
    forms: str = typer.Option("10-K,10-Q,8-K"),
    as_of: str | None = typer.Option(None, help="Run as-of date YYYY-MM-DD"),
) -> None:
    configure_logging()
    from app.ingest.backfill import backfill_range

    if as_of:
        parse_yyyy_mm_dd(as_of)
    count = backfill_range(
        parse_yyyy_mm_dd(from_date), parse_yyyy_mm_dd(to), _parse_forms(forms), as_of_date=as_of
    )
    typer.echo(f"Backfilled {count} filings")


@app.command("parse-filings")
def parse_filings(limit: int = typer.Option(50)) -> None:
    configure_logging()
    from app.parse.filing_parser import parse_pending_filings

    count = parse_pending_filings(limit=limit)
    typer.echo(f"Parsed {count} filings")


@app.command("compute-fundamentals")
def compute_fundamentals_cmd(
    as_of: str | None = typer.Option(None, help="Run as-of date YYYY-MM-DD"),
) -> None:
    configure_logging()
    from app.fundamentals.metrics import compute_all_fundamentals

    if as_of:
        parse_yyyy_mm_dd(as_of)
    count = compute_all_fundamentals(as_of_date=as_of)
    typer.echo(f"Computed fundamentals for {count} tickers")


@app.command("run-valuation")
def run_valuation_cmd(
    as_of: str | None = typer.Option(None, help="Run as-of date YYYY-MM-DD"),
) -> None:
    configure_logging()
    from app.valuation.sanity_checks import run_all_valuations

    if as_of:
        parse_yyyy_mm_dd(as_of)
    count = run_all_valuations(as_of_date=as_of)
    typer.echo(f"Ran valuations for {count} tickers")


@app.command("fundamentals-run")
def fundamentals_run_cmd(
    tickers: str | None = typer.Option(None, "--tickers", help="Optional ticker list CSV"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    years_back: int = typer.Option(10, "--years-back", help="Years-back dossier window"),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic run id"),
    workers: int = typer.Option(4, "--workers", help="Dossier worker count"),
) -> None:
    configure_logging()
    from app.dossier.runner import run_dossier_for_peer_set
    from app.valuation.fundamentals import write_fundamentals_for_run

    parse_yyyy_mm_dd(as_of)
    ticker_list = _parse_tickers_csv(tickers)
    init_db(get_config())
    dossier_summary = None
    if ticker_list:
        dossier_summary = run_dossier_for_peer_set(
            tickers=ticker_list,
            as_of_date=as_of,
            years_back=years_back,
            run_id=run_id,
            workers=workers,
            min_annual_filings=2,
        )
    cfg = get_config()
    summary = write_fundamentals_for_run(
        run_id=run_id,
        tickers=ticker_list or None,
        years_back=years_back,
        output_dir=cfg.sectors_dir / run_id,
    )
    payload = {
        "run_id": run_id,
        "as_of_date": as_of,
        "dossier_summary": dossier_summary,
        "fundamentals_summary": summary,
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("valuation-run")
def valuation_run_cmd(
    tickers: str | None = typer.Option(None, "--tickers", help="Optional ticker list CSV"),
    as_of: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD"),
    years_back: int = typer.Option(10, "--years-back", help="Years-back dossier window"),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic run id"),
    workers: int = typer.Option(4, "--workers", help="Dossier worker count"),
) -> None:
    configure_logging()
    from app.dossier.runner import run_dossier_for_peer_set
    from app.valuation.engine import write_valuations_for_run
    from app.valuation.fundamentals import write_fundamentals_for_run

    parse_yyyy_mm_dd(as_of)
    ticker_list = _parse_tickers_csv(tickers)
    init_db(get_config())
    dossier_summary = None
    if ticker_list:
        dossier_summary = run_dossier_for_peer_set(
            tickers=ticker_list,
            as_of_date=as_of,
            years_back=years_back,
            run_id=run_id,
            workers=workers,
            min_annual_filings=2,
        )
    cfg = get_config()
    fundamentals_summary = write_fundamentals_for_run(
        run_id=run_id,
        tickers=ticker_list or None,
        years_back=years_back,
        output_dir=cfg.sectors_dir / run_id,
    )
    valuation_summary = write_valuations_for_run(
        run_id=run_id,
        tickers=ticker_list or None,
        as_of_date=as_of,
        output_dir=cfg.sectors_dir / run_id,
    )
    payload = {
        "run_id": run_id,
        "as_of_date": as_of,
        "dossier_summary": dossier_summary,
        "fundamentals_summary": fundamentals_summary,
        "valuation_summary": valuation_summary,
    }
    skipped_synthesis = (dossier_summary or {}).get("tickers_synthesis_skipped") or []
    if skipped_synthesis:
        payload["synthesis_skipped"] = {
            "tickers": skipped_synthesis,
            "reason": (
                "The LLM provider is disabled and the evidence lacks the provenance the "
                "synthesis gate requires, so the LLM summary step was skipped. The "
                "deterministic valuation is unaffected."
            ),
        }
    typer.echo(json.dumps(payload, indent=2))


@app.command("valuation-binding-backfill")
def valuation_binding_backfill_cmd(
    apply: bool = typer.Option(
        False,
        "--apply/--dry-run",
        help="Persist exact verified artifact bindings (default: dry-run)",
    ),
    artifact_root: list[Path] | None = typer.Option(
        None,
        "--artifact-root",
        help="Artifact file/root to inspect (repeatable; defaults to autonomous-sector runs)",
    ),
) -> None:
    """Bind legacy valuation rows only to exact SHA-verified run artifacts."""

    from app.valuation.binding_backfill import backfill_valuation_bindings

    payload = backfill_valuation_bindings(
        artifact_roots=artifact_root,
        apply=apply,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("price-fetch")
def price_fetch_cmd(
    tickers: str = typer.Option(..., "--tickers", help="Ticker CSV (e.g. AAPL,MSFT)"),
    as_of: str = typer.Option(..., "--as-of", help="Requested as-of date YYYY-MM-DD"),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic output run id"),
    fallback_days: int = typer.Option(
        7, "--fallback-days", help="Provider fallback lookback window in days"
    ),
) -> None:
    configure_logging()
    from app.market.price_provider import write_prices_for_run

    parse_yyyy_mm_dd(as_of)
    ticker_list = _parse_tickers_csv(tickers)
    if not ticker_list:
        raise typer.BadParameter("--tickers must include at least one symbol")
    summary = write_prices_for_run(
        tickers=ticker_list,
        as_of_date=as_of,
        run_id=run_id,
        fallback_days=max(0, int(fallback_days)),
        cfg=get_config(),
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("sector-prewarm-prices")
def sector_prewarm_prices_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Sector run id"),
    fallback_days: int = typer.Option(
        5, "--fallback-days", help="Fallback walkback window in days"
    ),
    tickers: str | None = typer.Option(None, "--tickers", help="Optional ticker CSV override"),
) -> None:
    configure_logging()
    from app.market.price_prewarm import write_prices_prewarm_for_run

    cfg = get_config()
    run_dir = cfg.sectors_dir / run_id
    if not run_dir.exists():
        typer.echo(
            json.dumps(
                {"run_id": run_id, "status": "MISSING_RUN_DIR", "run_dir": str(run_dir)}, indent=2
            )
        )
        raise typer.Exit(code=1)

    ticker_list = _parse_tickers_csv(tickers) if tickers else []
    if not ticker_list:
        scoreboard = (
            json.loads((run_dir / "peer_scoreboard.json").read_text(encoding="utf-8"))
            if (run_dir / "peer_scoreboard.json").exists()
            else {}
        )
        scoreboard_rows = [row for row in (scoreboard.get("rows") or []) if isinstance(row, dict)]
        ticker_list = [
            str(row.get("ticker") or "").upper()
            for row in scoreboard_rows
            if str(row.get("ticker") or "").strip()
        ]

    if not ticker_list:
        peers_payload = (
            json.loads((run_dir / "sector_peers.json").read_text(encoding="utf-8"))
            if (run_dir / "sector_peers.json").exists()
            else {}
        )
        ticker_list = [
            str(t).upper() for t in (peers_payload.get("selected_tickers") or []) if str(t).strip()
        ]

    if not ticker_list:
        state_payload = (
            json.loads((run_dir / "rlm_state.json").read_text(encoding="utf-8"))
            if (run_dir / "rlm_state.json").exists()
            else {}
        )
        ticker_list = [
            str(t).upper() for t in (state_payload.get("peer_set") or []) if str(t).strip()
        ]

    summary_payload = (
        json.loads((run_dir / "sector_summary.json").read_text(encoding="utf-8"))
        if (run_dir / "sector_summary.json").exists()
        else {}
    )
    state_payload = (
        json.loads((run_dir / "rlm_state.json").read_text(encoding="utf-8"))
        if (run_dir / "rlm_state.json").exists()
        else {}
    )
    as_of_date = str(
        summary_payload.get("as_of_date") or state_payload.get("as_of_date") or ""
    ).strip()
    if not as_of_date:
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "MISSING_AS_OF",
                    "message": "Could not determine as_of_date from sector_summary.json or rlm_state.json.",
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)
    parse_yyyy_mm_dd(as_of_date)

    if not ticker_list:
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "MISSING_TICKERS",
                    "message": "No tickers found in peer_scoreboard.json, sector_peers.json, or rlm_state.json.",
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1)

    payload = write_prices_prewarm_for_run(
        run_id=run_id,
        as_of_date=as_of_date,
        tickers=ticker_list,
        fallback_days=max(0, int(fallback_days)),
        cfg=cfg,
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("financial-cache-refresh")
def financial_cache_refresh_cmd(
    tickers: str | None = typer.Option(None, "--tickers", help="Optional ticker CSV scope"),
    sectors: str | None = typer.Option(None, "--sectors", help="Optional sector CSV scope"),
    all_known: bool = typer.Option(
        False, "--all-known", help="Refresh all tickers known to the local DB"
    ),
    market_cap_focus: str = typer.Option(
        "smid_cap", "--market-cap-focus", help="Market-cap focus for sector scopes"
    ),
    max_tickers: int | None = typer.Option(
        None, "--max-tickers", help="Maximum tickers in the refresh scope"
    ),
    max_tickers_per_sector: int | None = typer.Option(
        None, "--max-tickers-per-sector", help="Maximum tickers to refresh per requested sector"
    ),
    sector_selection_mode: str = typer.Option(
        "cache_order",
        "--sector-selection-mode",
        help="Sector refresh targeting mode: cache_order or autonomous_candidates",
    ),
    years: int = typer.Option(10, "--years", help="Companyfacts lookback window in years"),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
    weekly: bool = typer.Option(
        False, "--weekly", help="Label this as a weekly incremental refresh"
    ),
    force: bool = typer.Option(
        False, "--force", help="Ignore completed steps in an existing refresh manifest"
    ),
    resume_run_id: str | None = typer.Option(
        None, "--resume-run-id", help="Resume an existing financial-cache refresh run"
    ),
    with_prices: bool = typer.Option(
        True, "--with-prices/--no-with-prices", help="Prewarm price cache"
    ),
    with_filings: bool = typer.Option(
        True, "--with-filings/--no-with-filings", help="Prewarm SEC filing cache"
    ),
    fallback_days: int | None = typer.Option(
        None, "--fallback-days", help="Price fallback lookback window"
    ),
) -> None:
    """Prewarm local financial caches for autonomous sector and benchmark runs."""

    configure_logging()
    from app.ingest.financial_cache_refresh import run_financial_cache_refresh

    if as_of:
        parse_yyyy_mm_dd(as_of)
    if not _parse_tickers_csv(tickers) and not _parse_tickers_csv(sectors) and not all_known:
        raise typer.BadParameter("Provide --tickers, --sectors, or --all-known")

    summary = run_financial_cache_refresh(
        tickers=_parse_tickers_csv(tickers),
        sectors=[item.strip() for item in (sectors or "").split(",") if item.strip()],
        all_known=bool(all_known),
        market_cap_focus=market_cap_focus,
        max_tickers=max_tickers,
        max_tickers_per_sector=max_tickers_per_sector,
        sector_selection_mode=sector_selection_mode,
        years=max(1, int(years)),
        as_of_date=as_of,
        weekly=bool(weekly),
        force=bool(force),
        resume_run_id=resume_run_id,
        with_prices=bool(with_prices),
        with_filings=bool(with_filings),
        fallback_days=fallback_days,
    )
    compact = {
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "as_of_date": summary.get("as_of_date"),
        "ticker_count": summary.get("ticker_count"),
        "ticker_status_counts": summary.get("ticker_status_counts"),
        "step_status_counts": summary.get("step_status_counts"),
        "error_count": summary.get("error_count"),
        "warnings": summary.get("warnings", []),
        "max_tickers_per_sector": summary.get("scope", {}).get("max_tickers_per_sector"),
        "sector_selection_mode": summary.get("scope", {}).get("sector_selection_mode"),
        "cache_readiness_status": (summary.get("cache_readiness") or {}).get("overall_status")
        if isinstance(summary.get("cache_readiness"), dict)
        else None,
        "cache_readiness_rollups": (summary.get("cache_readiness") or {}).get("rollups")
        if isinstance(summary.get("cache_readiness"), dict)
        else {},
        "summary_path": summary.get("summary_path"),
        "manifest_path": summary.get("manifest_path"),
        "report_path": summary.get("report_path"),
    }
    typer.echo(json.dumps(compact, indent=2))


@app.command("offline-seed-prices")
def offline_seed_prices_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Deterministic output run id"),
    tickers: str = typer.Option(..., "--tickers", help="Ticker CSV (e.g. AAPL,NVDA)"),
    as_of: str = typer.Option(..., "--as-of", help="Requested as-of date YYYY-MM-DD"),
) -> None:
    configure_logging()
    from app.market.price_provider import write_prices_for_run

    parse_yyyy_mm_dd(as_of)
    ticker_list = _parse_tickers_csv(tickers)
    if not ticker_list:
        raise typer.BadParameter("--tickers must include at least one symbol")
    summary = write_prices_for_run(
        tickers=ticker_list,
        as_of_date=as_of,
        run_id=run_id,
        fallback_days=max(0, int(get_config().price_fallback_days)),
        local_only=True,
        cfg=get_config(),
    )
    payload = {
        "run_id": run_id,
        "as_of_date": as_of,
        "mode": "offline_local_seed",
        **summary,
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("shares-fetch")
def shares_fetch_cmd(
    tickers: str = typer.Option(..., "--tickers", help="Ticker CSV (e.g. AAPL,MSFT)"),
    as_of: str = typer.Option(..., "--as-of", help="Requested as-of date YYYY-MM-DD"),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic output run id"),
    cache_only: bool = typer.Option(
        False,
        "--cache-only/--no-cache-only",
        help="Only use run-scoped/disk cache; skip DB fallback",
    ),
) -> None:
    configure_logging()
    from app.market.shares_provider import write_shares_for_run

    parse_yyyy_mm_dd(as_of)
    ticker_list = _parse_tickers_csv(tickers)
    if not ticker_list:
        raise typer.BadParameter("--tickers must include at least one symbol")
    init_db(get_config())
    summary = write_shares_for_run(
        tickers=ticker_list,
        as_of_date=as_of,
        run_id=run_id,
        cache_only=bool(cache_only),
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("companyfacts-fetch")
def companyfacts_fetch_cmd(
    tickers: str = typer.Option(..., "--tickers", help="Ticker CSV (e.g. AAPL,MSFT)"),
    as_of: str = typer.Option(..., "--as-of", help="Requested as-of date YYYY-MM-DD"),
    run_id: str = typer.Option(..., "--run-id", help="Deterministic output run id"),
    sec_budget: int | None = typer.Option(
        None, "--sec-budget", help="Optional max SEC fetch attempts for this command run"
    ),
) -> None:
    configure_logging()
    from app.market.company_facts_provider import fetch_company_facts
    from app.valuation.facts import resolve_cik_for_ticker

    parse_yyyy_mm_dd(as_of)
    cfg = get_config()
    init_db(cfg)

    ticker_list = sorted({ticker for ticker in _parse_tickers_csv(tickers) if ticker})
    if not ticker_list:
        raise typer.BadParameter("--tickers must include at least one symbol")

    out_dir = cfg.outputs_dir / "companyfacts" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    remaining_budget = int(sec_budget) if sec_budget is not None else None
    reason_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    ok_count = 0

    for ticker_symbol in ticker_list:
        cik = resolve_cik_for_ticker(ticker_symbol, cfg=cfg)
        if not cik:
            fetch_row = {
                "ticker": ticker_symbol,
                "cik": None,
                "as_of_date": as_of,
                "status": "MISSING",
                "reason_code": "CIK_MISSING",
                "reason_detail": "CIK mapping unavailable for ticker.",
                "source_resolution": "unknown",
                "cache_path": None,
                "source_url": None,
                "http_status": None,
                "network_attempted": False,
            }
        else:
            budget_for_call = remaining_budget if remaining_budget is not None else None
            result = fetch_company_facts(
                cik,
                user_agent=cfg.sec_user_agent,
                sec_budget=budget_for_call,
                cfg=cfg,
            )
            fetch_row = {
                "ticker": ticker_symbol,
                "cik": cik,
                "as_of_date": as_of,
                "status": result.get("status"),
                "reason_code": result.get("reason_code"),
                "reason_detail": result.get("reason_detail"),
                "source_resolution": result.get("source_resolution"),
                "cache_path": result.get("cache_path"),
                "source_url": result.get("source_url"),
                "http_status": result.get("http_status"),
                "network_attempted": bool(result.get("network_attempted")),
            }
            if (
                remaining_budget is not None
                and bool(result.get("network_attempted"))
                and remaining_budget > 0
            ):
                remaining_budget -= 1

        code = str(fetch_row.get("reason_code") or "UNKNOWN")
        reason_counts[code] = reason_counts.get(code, 0) + 1
        if str(fetch_row.get("status") or "").upper() == "OK":
            ok_count += 1

        ticker_path = out_dir / f"{ticker_symbol}.json"
        ticker_path.write_text(json.dumps(fetch_row, indent=2), encoding="utf-8")
        rows.append(fetch_row)

    rows = sorted(rows, key=lambda row: str(row.get("ticker") or ""))
    summary = {
        "run_id": run_id,
        "as_of_date": as_of,
        "ticker_count": len(rows),
        "ok_count": int(ok_count),
        "missing_count": int(len(rows) - ok_count),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    summary_path = out_dir / "companyfacts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["companyfacts_summary_path"] = str(summary_path)
    typer.echo(json.dumps(summary, indent=2))


@app.command("price-quote")
def price_quote_cmd(
    ticker: str = typer.Option(..., help="Ticker symbol"),
    as_of: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
) -> None:
    configure_logging()
    from app.valuation.price_provider import get_default_provider

    parse_yyyy_mm_dd(as_of)
    provider = get_default_provider()
    quote = provider.get_quote(ticker.upper(), as_of)
    typer.echo(
        json.dumps(
            {
                "ticker": quote.ticker,
                "as_of_date": quote.as_of_date,
                "provider": quote.provider,
                "status": quote.status,
                "price": quote.price,
                "source_url": quote.source_url,
                "fetched_at": quote.fetched_at,
                "expires_at": quote.expires_at,
                "provenance": quote.provenance,
            },
            indent=2,
        )
    )


@app.command("build-evidence-packets")
def build_evidence_packets_cmd() -> None:
    configure_logging()
    from app.evidence.packet_builder import build_all_packets

    count = build_all_packets()
    typer.echo(f"Built {count} evidence packets")


@app.command("research-run")
def research_run(
    ticker: str = typer.Option(..., help="Ticker symbol"),
    as_of: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    run_id: str | None = typer.Option(None, help="Optional deterministic run identifier"),
    sources: str | None = typer.Option(None, help="Optional source filter list, comma-separated"),
) -> None:
    configure_logging()
    from app.research.engine import run_research_agent_for_ticker

    parse_yyyy_mm_dd(as_of)
    path = run_research_agent_for_ticker(
        ticker.upper(),
        as_of_date=as_of,
        run_id=run_id,
        source_filters=_parse_sources_csv(sources),
    )
    if not path:
        typer.echo("No research packet produced")
        raise typer.Exit(code=1)
    typer.echo(f"Research packet: {path}")


@app.command("research-run-all")
def research_run_all(
    as_of: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    top: int | None = typer.Option(None, help="Top N ranked tickers to process"),
    limit: int | None = typer.Option(None, help="Optional hard limit on number of tickers"),
    tickers: str | None = typer.Option(
        None, help="Explicit ticker list, comma-separated (e.g. AAPL,MSFT)"
    ),
    active_universe: bool = typer.Option(
        False,
        "--active-universe/--top-ranked",
        help="Use active universe membership instead of top-ranked score list",
    ),
    run_id: str | None = typer.Option(None, help="Optional deterministic run identifier"),
    sources: str | None = typer.Option(None, help="Optional source filter list, comma-separated"),
) -> None:
    configure_logging()
    from app.research.engine import run_research_for_scope

    cfg = get_config()
    parse_yyyy_mm_dd(as_of)
    count = run_research_for_scope(
        as_of_date=as_of,
        top_n=top or cfg.research_top_n,
        use_active_universe=active_universe,
        run_id=run_id,
        tickers=_parse_tickers_csv(tickers),
        limit=limit,
        source_filters=_parse_sources_csv(sources),
    )
    typer.echo(f"Built {count} research packets")


@app.command("research-close-gaps")
def research_close_gaps(
    as_of: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    run_id: str | None = typer.Option(None, help="Optional run identifier"),
    limit: int | None = typer.Option(None, help="Optional cap on number of tickers"),
    sources: str = typer.Option("news,exhibits,homepage", help="Sources list, comma-separated"),
) -> None:
    configure_logging()
    from app.research.engine import run_research_gap_closer

    parse_yyyy_mm_dd(as_of)
    summary = run_research_gap_closer(
        as_of_date=as_of,
        run_id=run_id,
        limit=limit,
        source_filters=_parse_sources_csv(sources),
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("research-cycle")
def research_cycle_cmd(
    discovery_run_id: str = typer.Option(..., "--discovery-run-id", help="Discovery run id"),
    as_of: str = typer.Option(..., "--as-of", help="Run as-of date YYYY-MM-DD"),
    max_iterations: int = typer.Option(
        2, "--max-iterations", help="Max gap-closing iterations per ticker"
    ),
    top_k: int = typer.Option(10, "--top-k", help="Top K ADVANCE_TO_DEEP tickers to process"),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="Optional synthesis budget for this cycle run"
    ),
    sources: str = typer.Option(
        "news,exhibits,homepage", "--sources", help="Allowed research source filters"
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional deterministic cycle run id"),
) -> None:
    configure_logging()
    from app.research.cycle import run_research_cycle

    parse_yyyy_mm_dd(as_of)
    init_db(get_config())
    summary = run_research_cycle(
        discovery_run_id=discovery_run_id,
        as_of_date=as_of,
        max_iterations=max_iterations,
        top_k=top_k,
        budget_usd=budget_usd,
        sources=_parse_sources_csv(sources),
        run_id=run_id,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("agent-run")
def agent_run() -> None:
    configure_logging()
    from app.agent.scheduler import run_scheduler_forever

    run_scheduler_forever()


@app.command("agent-once")
def agent_once(
    as_of: str | None = typer.Option(None, help="Run as-of date YYYY-MM-DD"),
    with_research: bool = typer.Option(
        False, "--with-research", help="Include research agent stage"
    ),
    with_synthesis: bool = typer.Option(
        False, "--with-synthesis", help="Include synthesis agent stage"
    ),
) -> None:
    configure_logging()
    from app.agent.scheduler import run_scheduler_once

    if as_of:
        parse_yyyy_mm_dd(as_of)
    run_scheduler_once(as_of_date=as_of, with_research=with_research, with_synthesis=with_synthesis)


@app.command("agent-status")
def agent_status() -> None:
    configure_logging()
    cfg = get_config()
    if not cfg.status_path.exists():
        typer.echo("No status file found")
        raise typer.Exit(code=0)
    payload = json.loads(cfg.status_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(payload, indent=2))


@app.command("agent-cancel")
def agent_cancel(
    job_type: str | None = typer.Option(None), reason: str = typer.Option("cancelled by operator")
) -> None:
    configure_logging()
    from app.agent.scheduler import cancel_scheduled_jobs

    count = cancel_scheduled_jobs(job_type=job_type, reason=reason)
    typer.echo(f"Cancelled {count} jobs")


@app.command("deadletter-list")
def deadletter_list(
    limit: int = typer.Option(50, help="Maximum rows"),
    job_type: str | None = typer.Option(None, help="Filter by job type"),
) -> None:
    configure_logging()
    from app.agent.queue import list_dead_letters

    rows = list_dead_letters(limit=limit, job_type=job_type)
    typer.echo(json.dumps({"count": len(rows), "rows": rows}, indent=2))


@app.command("deadletter-retry")
def deadletter_retry(
    deadletter_id: int = typer.Option(..., "--id", help="Dead-letter row id"),
) -> None:
    configure_logging()
    from app.agent.queue import retry_dead_letter

    job_id = retry_dead_letter(deadletter_id)
    if job_id is None:
        typer.echo("Dead-letter id not found")
        raise typer.Exit(code=1)
    typer.echo(f"Requeued as job {job_id}")


@app.command("deadletter-retry-all")
def deadletter_retry_all(
    job_type: str | None = typer.Option(None, "--job-type", help="Optional job type filter"),
    limit: int | None = typer.Option(None, help="Optional cap on number of retries"),
) -> None:
    configure_logging()
    from app.agent.queue import retry_dead_letter_all

    count = retry_dead_letter_all(job_type=job_type, limit=limit)
    typer.echo(f"Requeued {count} dead-letter jobs")


@app.command("deadletter-purge")
def deadletter_purge(older_than_days: int = typer.Option(..., "--older-than-days")) -> None:
    configure_logging()
    from app.agent.queue import purge_dead_letters

    count = purge_dead_letters(older_than_days=older_than_days)
    typer.echo(f"Purged {count} dead-letter rows")


@app.command("score")
def score_cmd() -> None:
    configure_logging()
    from app.score.ranker import score_and_rank

    summary = score_and_rank()
    typer.echo(json.dumps(summary, indent=2))


@app.command("build-report")
def build_report(
    top: int = typer.Option(5, "--top"),
    memo_mode: str = typer.Option("strict", "--memo-mode", help="Memo mode: strict | triage"),
    run_id: str | None = typer.Option(None, help="Optional run identifier for triage gaps output"),
) -> None:
    configure_logging()
    from app.report.memo_builder import build_top_memos
    from app.ops.runs import list_runs

    memo_mode = memo_mode.strip().lower()
    if memo_mode == "triage" and not run_id:
        runs = list_runs(limit=1)
        if not runs:
            typer.echo("No runs found in runs index; provide --run-id or run run-all first.")
            raise typer.Exit(code=1)
        run_id = runs[0].get("run_id")
        typer.echo(f"Using latest run_id: {run_id}")
    if memo_mode == "triage" and not run_id:
        typer.echo("run_id is required for triage mode")
        raise typer.Exit(code=1)

    count = build_top_memos(top_n=top, memo_mode=memo_mode, run_id=run_id)
    typer.echo(f"Built {count} memos")


@app.command("build-deltas")
def build_deltas_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Current run identifier"),
    prev_run_id: str | None = typer.Option(
        None, "--prev-run-id", help="Optional previous run identifier"
    ),
) -> None:
    configure_logging()
    from app.delta.engine import build_deltas

    summary = build_deltas(run_id=run_id, prev_run_id=prev_run_id)
    typer.echo(json.dumps(summary, indent=2))


@app.command("build-delta-memos")
def build_delta_memos_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Current run identifier"),
    top: int = typer.Option(20, "--top", help="Top N tickers by score"),
    only_changed: bool = typer.Option(
        False, "--only-changed", help="Only build memos for changed tickers"
    ),
) -> None:
    configure_logging()
    from app.delta.memo_builder import build_delta_memos

    summary = build_delta_memos(run_id=run_id, top_n=top, only_changed=only_changed)
    typer.echo(json.dumps(summary, indent=2))


@app.command("pattern-scan")
def pattern_scan_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Completed sector dossier run id"),
    ticker: str | None = typer.Option(
        None, "--ticker", help="Optional ticker to inspect within the scan"
    ),
    pattern: list[str] | None = typer.Option(
        None, "--pattern", help="Optional pattern_id filter; repeatable"
    ),
) -> None:
    configure_logging()
    from app.patterns.scanner import scan_peer_set, summarize_pattern_scan_for_ticker

    selected_tickers = [ticker.upper()] if ticker else None
    report = scan_peer_set(
        run_id=run_id,
        tickers=selected_tickers,
        patterns=list(pattern) if pattern else None,
    )
    if ticker:
        payload = summarize_pattern_scan_for_ticker(report, ticker.upper())
        if pattern:
            wanted = {str(item).strip() for item in pattern if str(item).strip()}
            payload["pattern_hits"] = [
                item
                for item in (payload.get("pattern_hits") or [])
                if str(item.get("pattern_id") or "") in wanted
            ]
            payload["pattern_hit_count"] = len(payload["pattern_hits"])
            payload["summary_text"] = (
                " ".join(item.get("summary_text") or "" for item in payload["pattern_hits"][:3])
                or None
            )
        typer.echo(json.dumps(payload, indent=2))
        return
    if pattern:
        wanted = {str(item).strip() for item in pattern if str(item).strip()}
        payload = report.model_dump(mode="json")
        payload["pattern_results"] = [
            item
            for item in payload.get("pattern_results", [])
            if str(item.get("pattern_id") or "") in wanted
        ]
        payload["patterns_with_signal"] = [
            item for item in payload.get("patterns_with_signal", []) if str(item) in wanted
        ]
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))


@app.command("variant-perception")
def variant_perception_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Completed dossier or sector run id"),
    ticker: str | None = typer.Option(
        None, "--ticker", help="Optional ticker to build a single variant perception report"
    ),
) -> None:
    configure_logging()
    from app.diff.engine import list_dossier_run_tickers
    from app.synthesis.variant_builder import (
        build_variant_perceptions,
        variant_perception_report_path,
    )

    cfg = get_config()

    def _as_of_for_ticker(ticker_symbol: str) -> str:
        path = cfg.dossiers_dir / run_id / ticker_symbol.upper() / "dossier.json"
        if not path.exists():
            raise typer.BadParameter(
                f"dossier.json missing for ticker={ticker_symbol.upper()} run_id={run_id}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        as_of_date = str(payload.get("as_of_date") or "").strip()
        if not as_of_date:
            raise typer.BadParameter(
                f"dossier.json missing as_of_date for ticker={ticker_symbol.upper()} run_id={run_id}"
            )
        return as_of_date

    if ticker:
        ticker_norm = ticker.upper()
        report = build_variant_perceptions(
            ticker=ticker_norm,
            as_of_date=_as_of_for_ticker(ticker_norm),
            run_id=run_id,
            cfg=cfg,
        )
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
        return

    reports_built: list[dict[str, Any]] = []
    for ticker_symbol in list_dossier_run_tickers(run_id=run_id):
        as_of_date = _as_of_for_ticker(ticker_symbol)
        report = build_variant_perceptions(
            ticker=ticker_symbol,
            as_of_date=as_of_date,
            run_id=run_id,
            cfg=cfg,
        )
        reports_built.append(
            {
                "ticker": ticker_symbol,
                "as_of_date": as_of_date,
                "perception_count": len(report.perceptions),
                "directions": [perception.direction for perception in report.perceptions],
                "highest_confidence": (
                    sorted(
                        [perception.confidence for perception in report.perceptions],
                        key=lambda value: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(str(value), -1),
                        reverse=True,
                    )[0]
                    if report.perceptions
                    else None
                ),
                "report_path": str(
                    variant_perception_report_path(ticker_symbol, as_of_date, cfg=cfg)
                ),
            }
        )

    payload = {
        "run_id": run_id,
        "ticker_count": len(reports_built),
        "tickers_with_perceptions": [
            row for row in reports_built if int(row.get("perception_count") or 0) > 0
        ],
        "reports_built": reports_built,
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("shortlist")
def shortlist_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Current run identifier"),
    top: int = typer.Option(20, "--top", help="Top N shortlisted tickers"),
    min_score: float = typer.Option(0.0, "--min-score", help="Minimum score threshold"),
    require_recent_research: str = typer.Option(
        "false",
        "--require-recent-research",
        help="Whether to require non-stale research signals: true|false",
    ),
) -> None:
    configure_logging()
    from app.delta.shortlist import build_shortlist

    summary = build_shortlist(
        run_id=run_id,
        top_n=top,
        min_score=min_score,
        require_recent_research=require_recent_research.strip().lower() == "true",
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("synth-run")
def synth_run(
    ticker: str = typer.Option(..., help="Ticker symbol"),
    as_of: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    run_id: str = typer.Option(..., help="Run identifier"),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Bypass synthesis cache and rebuild the evidence packet before a fresh run",
    ),
    strict_as_of: bool = typer.Option(
        False,
        "--strict-as-of",
        help="Require an exact as-of evidence packet; do not silently fall back",
    ),
) -> None:
    configure_logging()
    from app.llm.synthesis_agent import (
        resolve_synthesis_as_of,
        run_synthesis_for_ticker,
        run_synthesis_for_ticker_no_cache,
    )

    parse_yyyy_mm_dd(as_of)
    ticker_norm = ticker.upper()
    if no_cache:
        _build_synthesis_evidence_packet(ticker_norm, as_of)
    resolution = resolve_synthesis_as_of(ticker.upper(), as_of_date=as_of)
    if resolution is None:
        typer.echo("No evidence packet available at or before the requested as-of date")
        raise typer.Exit(code=1)
    if strict_as_of and str(resolution.get("as_of_resolution")) != "exact":
        typer.echo(
            f"No exact evidence packet for {ticker.upper()} on {as_of}; "
            f"latest available is {resolution.get('effective_as_of_date')}"
        )
        raise typer.Exit(code=1)
    runner = run_synthesis_for_ticker_no_cache if no_cache else run_synthesis_for_ticker
    path = runner(
        ticker=ticker_norm,
        as_of_date=as_of,
        run_id=run_id,
        strict_as_of=strict_as_of,
    )
    if not path:
        typer.echo("No synthesis packet produced")
        raise typer.Exit(code=1)
    typer.echo(str(path))
    typer.echo(
        f"requested_as_of_date={resolution.get('requested_as_of_date')} "
        f"effective_as_of_date={resolution.get('effective_as_of_date')} "
        f"as_of_resolution={resolution.get('as_of_resolution')}"
    )


@app.command("synth-run-all")
def synth_run_all(
    as_of: str = typer.Option(..., help="As-of date YYYY-MM-DD"),
    run_id: str = typer.Option(..., help="Run identifier"),
    limit: int | None = typer.Option(None, help="Optional cap on tickers"),
    tickers: str | None = typer.Option(None, help="Explicit ticker list, comma-separated"),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Bypass synthesis cache and rebuild evidence packets before fresh runs",
    ),
    strict_as_of: bool = typer.Option(
        False,
        "--strict-as-of",
        help="Require exact as-of evidence packets for all requested tickers",
    ),
) -> None:
    configure_logging()
    from app.llm.synthesis_agent import run_synthesis_for_scope

    parse_yyyy_mm_dd(as_of)
    ticker_scope = _parse_tickers_csv(tickers)
    if no_cache:
        refresh_scope = ticker_scope
        if not refresh_scope:
            with get_db() as conn:
                rows = conn.execute(
                    """
                    SELECT ticker
                    FROM scores
                    WHERE run_id = ?
                    ORDER BY is_candidate DESC, total_score DESC, ticker ASC
                    """,
                    (run_id,),
                ).fetchall()
                refresh_scope = [row["ticker"] for row in rows]
        if limit is not None and limit > 0:
            refresh_scope = refresh_scope[:limit]
        for ticker_symbol in refresh_scope:
            _build_synthesis_evidence_packet(str(ticker_symbol).upper(), as_of)
    summary = run_synthesis_for_scope(
        as_of_date=as_of,
        run_id=run_id,
        tickers=ticker_scope,
        limit=limit,
        allow_cache=not no_cache,
        strict_as_of=strict_as_of,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("synth-export")
def synth_export(
    ticker: str = typer.Option(..., help="Ticker symbol"),
    run_id: str = typer.Option(..., help="Run identifier"),
    out: Path = typer.Option(..., help="Output path"),
) -> None:
    configure_logging()
    from app.llm.synthesis_agent import export_synthesis_packet

    path = export_synthesis_packet(ticker=ticker.upper(), run_id=run_id, out=out)
    typer.echo(str(path))


@app.command("run-all")
def run_all(
    as_of: str | None = typer.Option(None, help="Run as-of date YYYY-MM-DD"),
    with_research: bool = typer.Option(
        False, "--with-research", help="Include research agent stage"
    ),
    with_synthesis: bool = typer.Option(
        False, "--with-synthesis", help="Include synthesis agent stage"
    ),
    with_discovery: bool = typer.Option(
        False, "--with-discovery", help="Run discovery first and scope deep pipeline to top K"
    ),
    dossier_top: int | None = typer.Option(
        None,
        "--dossier-top",
        help="When --with-discovery is enabled, run dossier on top K ADVANCE_TO_DEEP tickers",
    ),
    limit: int | None = typer.Option(None, help="Optional max tickers"),
    tickers: str | None = typer.Option(None, help="Explicit ticker list, comma-separated"),
    discovery_top: int = typer.Option(
        25, "--discovery-top", help="Top K discovery tickers to deepen"
    ),
    discovery_seed: Path | None = typer.Option(
        None, "--discovery-seed", help="Optional discovery seed CSV path"
    ),
    phase: list[str] | None = typer.Option(
        None,
        "--phase",
        help="Execution phase(s): ingest, valuation, research. Repeatable.",
    ),
    forms: str = typer.Option("10-K,10-Q,8-K", help="Form types for ingest phase"),
    memo_mode: str = typer.Option("strict", "--memo-mode", help="Memo mode: strict | triage"),
    top: int = typer.Option(10, "--top", help="Top N tickers for triage memo generation"),
) -> None:
    configure_logging()
    try:
        summary = _run_all_impl(
            as_of=as_of,
            with_research=with_research,
            with_synthesis=with_synthesis,
            with_discovery=with_discovery,
            dossier_top=dossier_top,
            limit=limit,
            tickers=tickers,
            discovery_top=discovery_top,
            discovery_seed=discovery_seed,
            phase=phase,
            forms=forms,
            memo_mode=memo_mode,
            top=top,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(summary, indent=2))


@app.command("runs-list")
def runs_list(limit: int = typer.Option(10, help="Number of recent runs to show")) -> None:
    configure_logging()
    from app.ops.runs import list_runs

    typer.echo(
        json.dumps({"count": len(list_runs(limit=limit)), "runs": list_runs(limit=limit)}, indent=2)
    )


@app.command("run-open")
def run_open(run_id: str = typer.Option(..., "--run-id", help="Run identifier")) -> None:
    configure_logging()
    from app.ops.runs import get_run_directory

    path = get_run_directory(run_id)
    if not path.exists():
        typer.echo(f"Run directory not found for run_id={run_id}")
        raise typer.Exit(code=1)
    typer.echo(str(path.resolve()))


@app.command("run-report")
def run_report(run_id: str = typer.Option(..., "--run-id", help="Run identifier")) -> None:
    configure_logging()
    from app.ops.runs import get_run_report

    report = get_run_report(run_id)
    if not report:
        typer.echo(f"Run report not found for run_id={run_id}")
        raise typer.Exit(code=1)

    entry = report.get("index_entry") or {}
    gating = report.get("gating_report") or {}
    rows = gating.get("rows") if isinstance(gating, dict) else []
    rows = rows if isinstance(rows, list) else []

    failures = []
    for row in rows:
        ticker = row.get("ticker")
        for gate in row.get("gate_statuses", []):
            if gate.get("status") == "FAIL":
                failures.append(
                    {
                        "ticker": ticker,
                        "gate": gate.get("gate"),
                        "reason_code": gate.get("reason_code"),
                        "message": gate.get("message"),
                        "fix_hint": gate.get("fix_hint"),
                    }
                )
                break

    summary = {
        "run_id": run_id,
        "run_dir": report.get("run_dir"),
        "as_of_date": entry.get("as_of_date"),
        "tickers_targeted": entry.get("tickers_targeted"),
        "tickers_processed": entry.get("tickers_processed"),
        "memos_count": entry.get("memos_count"),
        "candidates_count": entry.get("candidates_count"),
        "dead_letter_delta": entry.get("dead_letter_delta"),
        "gating_report_json": report.get("gating_report_json_path"),
        "gating_report_csv": report.get("gating_report_csv_path"),
        "first_failures": failures[:10],
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("demo-run")
def demo_run(
    as_of: str | None = typer.Option(None, help="Run as-of date YYYY-MM-DD"),
    tickers: str | None = typer.Option(None, help="Optional ticker override, comma-separated"),
    forms: str = typer.Option("10-K,10-Q,8-K", help="Form types for ingest phase"),
) -> None:
    configure_logging()
    from app.ops.runs import get_run_report
    from app.universe.universe import get_active_universe_id, list_universe_members

    as_of_date = parse_yyyy_mm_dd(as_of).isoformat() if as_of else date.today().isoformat()
    requested = _parse_tickers_csv(tickers)
    defaults = ["AAPL", "MSFT", "JPM", "V"]

    with get_db() as conn:
        universe_id = get_active_universe_id(conn)
        if universe_id:
            universe_members = {row["ticker"] for row in list_universe_members(conn, universe_id)}
        else:
            rows = conn.execute("SELECT ticker FROM companies").fetchall()
            universe_members = {row["ticker"] for row in rows}

    chosen = requested or [t for t in defaults if t in universe_members]
    if not chosen:
        typer.echo("No demo tickers available in active universe. Load universe first.")
        raise typer.Exit(code=1)

    summary = _run_all_impl(
        as_of=as_of_date,
        with_research=True,
        with_synthesis=False,
        with_discovery=False,
        dossier_top=None,
        limit=len(chosen),
        tickers=",".join(chosen),
        discovery_top=len(chosen),
        discovery_seed=None,
        phase=["ingest", "valuation", "research"],
        forms=forms,
        memo_mode="triage",
        top=len(chosen),
    )
    typer.echo(json.dumps(summary, indent=2))

    report = get_run_report(str(summary["run_id"]))
    if not report:
        raise typer.Exit(code=0)
    rows = (report.get("gating_report") or {}).get("rows", [])
    rows = rows if isinstance(rows, list) else []

    memo_rows = [row for row in rows if row.get("memo_built")]
    if memo_rows:
        typer.echo("Demo memo artifacts:")
        for row in memo_rows[:5]:
            path = (row.get("artifacts") or {}).get("memo_path")
            if path:
                typer.echo(f"- {row.get('ticker')}: {report['run_dir']}/{path}")
    else:
        typer.echo("No memo was built in demo run. Top gating failures:")
        for row in rows:
            failed = [g for g in (row.get("gate_statuses") or []) if g.get("status") == "FAIL"]
            if not failed:
                continue
            first = failed[0]
            typer.echo(
                f"- {row.get('ticker')}: {first.get('reason_code')} | {first.get('message')} | fix: {first.get('fix_hint')}"
            )


@app.command("gaps-list")
def gaps_list(
    run_id: str | None = typer.Option(None, help="Optional run identifier filter"),
) -> None:
    configure_logging()
    cfg = get_config()
    from app.ops.runs import list_runs

    target_run_id = run_id
    if not target_run_id:
        runs = list_runs(limit=1)
        if runs:
            target_run_id = runs[0].get("run_id")
    if not target_run_id:
        typer.echo(json.dumps({"run_id": None, "count": 0, "rows": []}, indent=2))
        raise typer.Exit(code=0)

    rows: list[dict[str, str]] = []
    cfg.gaps_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(cfg.gaps_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("run_id") != target_run_id:
            continue
        rows.append(
            {
                "ticker": payload.get("ticker", ""),
                "as_of_date": payload.get("as_of_date", ""),
                "path": str(path),
            }
        )
    typer.echo(json.dumps({"run_id": target_run_id, "count": len(rows), "rows": rows}, indent=2))


@app.command("autonomous-run")
def autonomous_run_cmd(
    ticker: str = typer.Argument(..., help="Ticker to analyze"),
    objective: str = typer.Option(
        "Determine whether this ticker is actionable, watchlist-only, avoid, or no-winner based on available evidence.",
        "--objective",
        help="Run objective for the autonomous analyst.",
    ),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
    max_tool_calls: int = typer.Option(
        8, "--max-tool-calls", help="Maximum alpha tool calls to execute"
    ),
    max_turns: int = typer.Option(4, "--max-turns", help="Maximum LLM turns for the v1 loop"),
    max_cost_usd: float | None = typer.Option(
        1.25, "--max-cost-usd", help="Maximum estimated LLM cost"
    ),
    years: int = typer.Option(
        5, "--years", help="Years of annual data to seed into analyst context"
    ),
    quarters: int = typer.Option(
        4, "--quarters", help="Recent quarters to seed into analyst context"
    ),
    skip_analysis_refresh: bool = typer.Option(
        False,
        "--skip-analysis-refresh",
        help="Use cached analyst context only; intended for offline/debug runs.",
    ),
) -> None:
    """Run a bounded autonomous analyst loop for a single ticker."""
    configure_logging()

    from app.autonomous.output_store import persist_autonomous_run
    from app.autonomous.run_contract import AutonomousRunBudget
    from app.autonomous.runtime import artifact_summary, run_single_candidate_autonomous_analysis

    run_budget = AutonomousRunBudget(
        max_tool_calls=max(0, int(max_tool_calls)),
        max_turns=max(1, int(max_turns)),
        max_cost_usd=max_cost_usd,
        timebox_seconds=None,
        max_candidates=1,
    )
    artifact = run_single_candidate_autonomous_analysis(
        ticker=ticker,
        objective=objective,
        as_of_date=as_of,
        budget=run_budget,
        analysis_years=max(1, int(years)),
        analysis_quarters=max(0, int(quarters)),
        skip_analysis_refresh=skip_analysis_refresh,
    )
    paths = persist_autonomous_run(artifact)
    summary = artifact_summary(artifact)
    summary["artifact_path"] = str(paths.artifact_json)
    summary["report_path"] = str(paths.report_md)
    typer.echo(json.dumps(summary, indent=2))


@classic_scan_app.command("plan")
def classic_scan_plan_cmd(
    mode: str = typer.Option(
        ...,
        "--mode",
        help="missed-only for current gaps, or full-rescan for a fresh campaign",
    ),
    pipeline_version: str = typer.Option("v1", "--pipeline-version", help="Pinned to classic v1"),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help=(
            "Structural-gate date; v1 membership remains the latest current "
            "membership, not a historical snapshot"
        ),
    ),
    backfill_artifacts: bool = typer.Option(
        False,
        "--backfill-artifacts",
        help="Idempotently recover exact historical coverage before planning",
    ),
    populate_watchlist: bool = typer.Option(
        True,
        "--watchlist/--no-watchlist",
        help="Populate the watchlist from completed child sector runs",
    ),
    resume_from_campaign_id: str | None = typer.Option(
        None,
        "--resume-from-campaign-id",
        help=(
            "For missed-only, reuse a prior frozen target and exclude only its "
            "exact terminal campaign rows"
        ),
    ),
) -> None:
    """Persist an exact provider-free campaign plan. Makes no LLM call."""
    configure_logging()
    if str(pipeline_version).strip().lower() != "v1":
        raise typer.BadParameter("classic-scan is pinned to --pipeline-version v1")
    from app.autonomous.classic_scan_driver import plan_campaign

    try:
        summary = plan_campaign(
            mode=mode,
            as_of=as_of,
            backfill_artifacts=backfill_artifacts,
            populate_watchlist=populate_watchlist,
            resume_from_campaign_id=resume_from_campaign_id,
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2))
    if summary["status"] == "BLOCKED":
        raise typer.Exit(1)


@classic_scan_app.command("run")
def classic_scan_run_cmd(
    campaign_id: str = typer.Option(..., "--campaign-id"),
    authorized_max_cost_usd: float = typer.Option(
        ...,
        "--authorized-max-cost-usd",
        help="Exact authorized LLM spend ceiling for the whole campaign (USD)",
    ),
    expect_plan_sha256: str = typer.Option(
        ..., "--expect-plan-sha256", help="Plan fingerprint shown by classic-scan plan"
    ),
    expect_provider: str = typer.Option(
        ..., "--expect-provider", help="Exact provider bound by the plan"
    ),
    expect_model: str = typer.Option(..., "--expect-model", help="Exact model bound by the plan"),
    accept_estimate_shortfall: bool = typer.Option(
        False,
        "--accept-estimate-shortfall",
        help=(
            "Owner explicitly accepts that a below-estimate hard cap may stop "
            "the campaign incomplete"
        ),
    ),
    cell_timeout_seconds: int = typer.Option(
        7200,
        "--cell-timeout-seconds",
        min=1,
        help="Hard wall-clock timeout for one sector-band child run",
    ),
) -> None:
    """Run or resume one campaign serially under an explicit aggregate cap."""
    configure_logging()
    from app.autonomous.classic_scan_driver import run_campaign

    def progress(event: dict[str, Any]) -> None:
        typer.echo(json.dumps(event, sort_keys=True), err=True)

    try:
        summary = run_campaign(
            campaign_id=campaign_id,
            authorized_max_cost_usd=authorized_max_cost_usd,
            expect_plan_sha256=expect_plan_sha256,
            expect_provider=expect_provider,
            expect_model=expect_model,
            accept_estimate_shortfall=accept_estimate_shortfall,
            cell_timeout_seconds=cell_timeout_seconds,
            progress=progress,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2))
    if summary["status"] != "COMPLETE":
        raise typer.Exit(1)


@classic_scan_app.command("status")
def classic_scan_status_cmd(
    campaign_id: str = typer.Option(..., "--campaign-id"),
) -> None:
    """Read the last atomic checkpoint without executing or spending."""
    configure_logging()
    from app.autonomous.classic_scan_driver import campaign_status

    try:
        summary = campaign_status(campaign_id=campaign_id)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2))


@classic_scan_app.command("reconcile")
def classic_scan_reconcile_cmd(
    campaign_id: str = typer.Option(..., "--campaign-id"),
    assume_reserved_spent: bool = typer.Option(
        False,
        "--assume-reserved-spent",
        help="Conservatively charge every unresolved pre-launch reservation",
    ),
    confirm_child_stopped: bool = typer.Option(
        False,
        "--confirm-child-stopped",
        help="Confirm no child process from the unresolved attempt is still running",
    ),
) -> None:
    """Resolve an ambiguous interrupted attempt without guessing its spend."""
    configure_logging()
    from app.autonomous.classic_scan_driver import reconcile_campaign

    try:
        summary = reconcile_campaign(
            campaign_id=campaign_id,
            assume_reserved_spent=assume_reserved_spent,
            confirm_child_stopped=confirm_child_stopped,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2))


@classic_scan_app.command("verify")
def classic_scan_verify_cmd(
    campaign_id: str = typer.Option(..., "--campaign-id"),
    require_complete: bool = typer.Option(
        False,
        "--require-complete",
        help="Exit nonzero unless campaign-scoped current coverage is complete",
    ),
) -> None:
    """Recompute zero-LLM closure against the campaign's frozen membership."""
    configure_logging()
    from app.autonomous.classic_scan_driver import verify_campaign

    try:
        summary = verify_campaign(campaign_id=campaign_id)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2))
    if require_complete and not summary["complete"]:
        raise typer.Exit(1)


@classic_scan_app.command("reconcile-data-gaps")
def classic_scan_reconcile_data_gaps_cmd(
    campaign_id: str = typer.Option(..., "--campaign-id"),
    expect_plan_sha256: str = typer.Option(
        ...,
        "--expect-plan-sha256",
        help="Exact frozen plan fingerprint from the stopped campaign",
    ),
    expect_state_sha256: str = typer.Option(
        ...,
        "--expect-state-sha256",
        help="Exact raw-byte SHA-256 of campaign_state.json",
    ),
    expect_target_tickers_sha256: str = typer.Option(
        ...,
        "--expect-target-tickers-sha256",
        help="Exact canonical frozen-ticker-set SHA-256",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="New mode-0600 JSON artifact path; existing files are never replaced",
    ),
) -> None:
    """Reconcile stopped zero-provider reasons without awarding coverage."""

    configure_logging()
    from app.autonomous.classic_scan_driver import reconcile_terminal_data_gaps

    try:
        summary = reconcile_terminal_data_gaps(
            campaign_id=campaign_id,
            expect_plan_sha256=expect_plan_sha256,
            expect_state_sha256=expect_state_sha256,
            expect_target_tickers_sha256=expect_target_tickers_sha256,
            artifact_path=output,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2))


@app.command("sweep-delta-report")
def sweep_delta_report_cmd(
    band: str | None = typer.Option(None, "--band", help="One market-cap focus (e.g. micro_cap)"),
    sectors: str | None = typer.Option(
        None, "--sectors", help="Optional comma-separated sector subset (default: all 34)"
    ),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
    synced_after: str | None = typer.Option(
        None,
        "--synced-after",
        help="Cross-check: registrants first seen after this timestamp must land in the delta",
    ),
    backfill_artifacts: bool = typer.Option(
        False,
        "--backfill-artifacts",
        help=(
            "Idempotently upgrade the coverage ledger from exact persisted "
            "artifact outcomes before reporting (writes the local DB; no LLM)"
        ),
    ),
    pipeline_version: str | None = typer.Option(
        None,
        "--pipeline-version",
        help="Explicit v1/v2 override; otherwise use the cap-band rollout allowlist",
    ),
    full_universe_v1: bool = typer.Option(
        False,
        "--full-universe-v1",
        help=(
            "Resolve all 34 sectors x five disjoint classic bands and reconcile "
            "coverage to current sector-tagged membership at zero LLM cost"
        ),
    ),
    require_complete: bool = typer.Option(
        False,
        "--require-complete",
        help="In full-universe mode, exit nonzero while any eligible coverage remains",
    ),
) -> None:
    """Current delta coverage for one band or the complete classic grid — $0 LLM."""
    configure_logging()

    from app.autonomous.sweep_delta import (
        backfill_loaded_sets_from_artifacts,
        band_delta_report,
        full_universe_delta_report,
    )

    payload: dict[str, Any] = {}
    if full_universe_v1:
        if band is not None:
            raise typer.BadParameter(
                "--band is not used with --full-universe-v1; the full atomic grid is fixed"
            )
        if sectors is not None:
            raise typer.BadParameter(
                "--sectors is not allowed with --full-universe-v1; all 34 are mandatory"
            )
        if pipeline_version is not None and str(pipeline_version).strip().lower() != "v1":
            raise typer.BadParameter("--full-universe-v1 requires --pipeline-version v1")
        if backfill_artifacts:
            payload["artifact_backfill"] = backfill_loaded_sets_from_artifacts()
        payload["report"] = full_universe_delta_report(
            as_of=as_of,
            synced_after=synced_after,
        )
        typer.echo(json.dumps(payload, indent=2))
        if require_complete and not payload["report"]["complete"]:
            raise typer.Exit(code=1)
        return
    if require_complete:
        raise typer.BadParameter("--require-complete requires --full-universe-v1")
    if band is None:
        raise typer.BadParameter("provide --band or use --full-universe-v1")
    if backfill_artifacts:
        payload["artifact_backfill"] = backfill_loaded_sets_from_artifacts()
    sector_list = _parse_tickers_csv(sectors) if sectors else None
    if sector_list:
        sector_list = [s.lower() for s in sector_list]
    payload["report"] = band_delta_report(
        band,
        sectors=sector_list,
        as_of=as_of,
        synced_after=synced_after,
        pipeline_version=pipeline_version,
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("autonomous-sector-run")
def autonomous_sector_run_cmd(
    sector: str = typer.Option(
        ..., "--sector", help="Sector label for the autonomous financial run"
    ),
    tickers: str | None = typer.Option(
        None,
        "--tickers",
        help="Optional comma-separated tickers. If omitted, tickers are sourced from the local sector scan cache/DB.",
    ),
    objective: str = typer.Option(
        "Determine which company has the strongest financially underwritten 5-10 year per-share return potential, or return no selection if the evidence does not clear the bar.",
        "--objective",
        help="Run objective for the autonomous sector analyst.",
    ),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
    market_cap_focus: str = typer.Option(
        "small_cap", "--market-cap-focus", help="Market-cap focus label"
    ),
    pipeline_version: str | None = typer.Option(
        None,
        "--pipeline-version",
        help="Explicit sector pipeline version (v1 or v2); omitted uses the cap-band rollout allowlist.",
    ),
    max_tool_calls: int = typer.Option(
        16, "--max-tool-calls", help="Maximum deterministic tool calls to execute"
    ),
    max_turns: int = typer.Option(
        6, "--max-turns", help="Maximum LLM turns for the autonomous loop"
    ),
    max_cost_usd: float | None = typer.Option(
        None,
        "--max-cost-usd",
        help="Optional estimated LLM cost ceiling; omitted means no per-sector dollar cap",
    ),
    max_candidates: int | None = typer.Option(
        None,
        "--max-candidates",
        help="Optional maximum tickers to review; omitted means review all loaded candidates.",
    ),
    populate_watchlist: bool = typer.Option(
        True,
        "--watchlist/--no-watchlist",
        help="Populate the persistent watchlist from actionable/watchlist finalist candidates.",
    ),
    only_unswept: bool = typer.Option(
        False,
        "--only-unswept",
        help=(
            "Delta sweep: review only names without an auditable terminal state "
            "for this sector and band; prior autonomous verdicts may carry "
            "through without a fresh LLM review. Failed per-ticker reviews stay "
            "open for retry. Exits at zero cost when the delta is empty."
        ),
    ),
    coverage_campaign_id: str | None = typer.Option(
        None,
        "--coverage-campaign-id",
        help=(
            "Fresh full-rescan coverage namespace. Valid only with v1 "
            "--only-unswept; prior campaigns do not suppress this run."
        ),
    ),
    coverage_target_file: str | None = typer.Option(
        None,
        "--coverage-target-file",
        help=(
            "Driver-generated frozen ticker target. Required with a classic "
            "coverage campaign and invalid outside campaign mode."
        ),
    ),
    carry_prior_verdicts: bool = typer.Option(
        True,
        "--carry-prior-verdicts/--no-carry-prior-verdicts",
        help=(
            "Allow an existing autonomous verdict to close an unswept name. "
            "Fresh full-rescan campaigns must disable this."
        ),
    ),
    strict_cost_cap: bool = typer.Option(
        False,
        "--strict-cost-cap",
        help="Reject even the first LLM call when its estimate exceeds --max-cost-usd.",
    ),
) -> None:
    """Run an iterative autonomous financial analyst loop over explicit sector tickers."""
    configure_logging()

    from app.autonomous.output_store import (
        persist_autonomous_sector_attempt_diagnostic,
        persist_autonomous_sector_diagnostic,
        persist_autonomous_sector_run,
    )
    from app.autonomous.candidate_review import (
        provider_usage_attestation,
        provider_usage_tail_not_in_snapshot,
    )
    from app.autonomous.run_contract import AutonomousRunBudget
    from app.autonomous.sector_candidates import resolve_sector_candidate_tickers
    from app.autonomous.sector_runtime import (
        SECTOR_LLM_RETRY_RUN_BUDGET,
        enrich_sector_artifact_memo_body,
        run_sector_autonomous_financial_analysis,
        sector_artifact_summary,
    )
    from app.config import (
        canonical_market_cap_focus,
        resolve_autonomous_sector_pipeline_version,
    )
    from app.llm.providers.retry_guard import llm_cost_budget, llm_retry_budget

    zero_provider_usage_attestation = provider_usage_attestation({})

    max_candidates = _validate_max_candidates_option(max_candidates)
    if only_unswept and max_candidates is not None:
        raise typer.BadParameter(
            "--only-unswept requires the complete loaded cell; "
            "--max-candidates would falsely mark unprocessed overflow as covered"
        )
    normalized_coverage_campaign_id = str(coverage_campaign_id or "").strip() or None
    normalized_coverage_target_file = str(coverage_target_file or "").strip() or None
    if normalized_coverage_campaign_id and (
        not only_unswept or str(pipeline_version or "v1").strip().lower() != "v1"
    ):
        raise typer.BadParameter(
            "--coverage-campaign-id requires --only-unswept --pipeline-version v1"
        )
    if not carry_prior_verdicts and not normalized_coverage_campaign_id:
        raise typer.BadParameter("--no-carry-prior-verdicts requires --coverage-campaign-id")
    if normalized_coverage_campaign_id and not normalized_coverage_target_file:
        raise typer.BadParameter(
            "--coverage-campaign-id requires the registered --coverage-target-file"
        )
    if normalized_coverage_target_file and not normalized_coverage_campaign_id:
        raise typer.BadParameter("--coverage-target-file requires --coverage-campaign-id")
    if normalized_coverage_campaign_id and carry_prior_verdicts:
        raise typer.BadParameter("classic coverage campaigns require --no-carry-prior-verdicts")
    coverage_target_tickers: set[str] | None = None
    if normalized_coverage_campaign_id:
        from app.autonomous.classic_scan_driver import validate_coverage_target_file

        coverage_target_tickers = set(
            validate_coverage_target_file(
                normalized_coverage_target_file,
                campaign_id=normalized_coverage_campaign_id,
            )
        )
    resolved_pipeline_version = resolve_autonomous_sector_pipeline_version(
        market_cap_focus,
        pipeline_version,
    )
    if resolved_pipeline_version == "v2":
        if only_unswept:
            raise typer.BadParameter(
                "--only-unswept cannot alter the frozen v2 execution authority; "
                "use autonomous-sector-benchmark resume controls instead."
            )
        from app.config import get_config

        configured_provider = str(get_config().llm_provider or "disabled").strip().lower()
        if configured_provider != "disabled":
            raise typer.BadParameter(
                "Paid v2 single-sector execution is disabled because this entry point does "
                "not own the whole-run $100 cost authorization. Use "
                "`autonomous-sector-benchmark --sectors <sector> --pipeline-version v2` "
                "for v2 canaries."
            )
    candidate_selection = None
    selection_payload: dict[str, Any] = {}
    delta_audit: dict[str, Any] | None = None
    try:
        candidate_selection = resolve_sector_candidate_tickers(
            sector=sector,
            explicit_tickers=_parse_tickers_csv(tickers),
            market_cap_focus=market_cap_focus,
            max_candidates=max_candidates,
            as_of_date=as_of,
            pipeline_version=resolved_pipeline_version,
            filing_risk_use_llm=False,
            allow_live_market_data=False,
            require_accepted_census=(
                resolved_pipeline_version == "v2"
                and canonical_market_cap_focus(market_cap_focus) == "large_and_mega"
            ),
        )
        if coverage_target_tickers is not None:
            for field_name in (
                "requested_tickers",
                "loaded_tickers",
                "selected_tickers",
                "excluded_tickers",
                "membership_tickers",
                "execution_tickers",
                "deferred_by_bound_tickers",
            ):
                values = list(getattr(candidate_selection, field_name, []) or [])
                setattr(
                    candidate_selection,
                    field_name,
                    [ticker for ticker in values if ticker in coverage_target_tickers],
                )
            candidate_selection.cap_classifications = {
                ticker: value
                for ticker, value in candidate_selection.cap_classifications.items()
                if ticker in coverage_target_tickers
            }
            candidate_selection.structural_gate_results = {
                ticker: value
                for ticker, value in candidate_selection.structural_gate_results.items()
                if ticker in coverage_target_tickers
            }
        if only_unswept:
            from app.autonomous.sweep_delta import (
                record_zero_cost_coverage,
                record_unknown_cap_cross_band_coverage,
                split_loaded_coverage,
                split_unswept,
                unknown_cap_tickers_from_selection,
            )

            candidate_source_errors = [
                str(warning)
                for warning in (candidate_selection.warnings or [])
                if str(warning).startswith("SECTOR_CANDIDATE_SOURCE_FAILED:")
            ]
            if candidate_source_errors:
                typer.echo(
                    json.dumps(
                        {
                            "scan_family": "normal",
                            "pipeline_version": resolved_pipeline_version,
                            "sector": sector,
                            "market_cap_focus": market_cap_focus,
                            "status": "INCOMPLETE_DELTA",
                            "loaded": len(candidate_selection.loaded_tickers),
                            "selected": len(candidate_selection.selected_tickers),
                            "source_errors": candidate_source_errors,
                            "coverage_unresolved": "candidate_source_failed",
                            "llm_cost": {
                                "cumulative_cost_usd": 0.0,
                                "call_count": 0,
                            },
                            "provider_usage_attestation": zero_provider_usage_attestation,
                            "provider_usage_incremental_attestation": (
                                zero_provider_usage_attestation
                            ),
                        },
                        indent=2,
                    )
                )
                raise typer.Exit(1)

            gate_error_tickers = {
                str(ticker).strip().upper()
                for ticker, result in (candidate_selection.structural_gate_results or {}).items()
                if bool((result or {}).get("excluded_error"))
            }
            with get_db() as conn:
                split = split_unswept(
                    conn,
                    band=market_cap_focus,
                    sector=sector,
                    selected_tickers=candidate_selection.selected_tickers,
                    pipeline_version=resolved_pipeline_version,
                    coverage_campaign_id=normalized_coverage_campaign_id,
                    allow_carried_verdicts=carry_prior_verdicts,
                )
                loaded_coverage = split_loaded_coverage(
                    conn,
                    band=market_cap_focus,
                    sector=sector,
                    loaded_tickers=candidate_selection.loaded_tickers,
                    pipeline_version=resolved_pipeline_version,
                    coverage_campaign_id=normalized_coverage_campaign_id,
                )
                uncovered_set = set(loaded_coverage["uncovered"])
                terminal_dispositions = {
                    str(ticker).strip().upper(): "STRUCTURAL_SCREENED"
                    for ticker, result in (
                        candidate_selection.structural_gate_results or {}
                    ).items()
                    if str(ticker).strip().upper() in uncovered_set
                    and bool((result or {}).get("quarantined"))
                    and not bool((result or {}).get("excluded_error"))
                }
                terminal_dispositions.update(
                    {
                        str(ticker).strip().upper(): "CARRIED_VERDICT"
                        for ticker in split["carried"]
                        if str(ticker).strip().upper() in uncovered_set
                    }
                )
                coverage_evidence_by_ticker = {
                    ticker: {
                        "disposition": "STRUCTURAL_SCREENED",
                        "effective_as_of": (
                            (candidate_selection.structural_gate_results or {}).get(ticker, {})
                            or {}
                        ).get("as_of_date")
                        or as_of,
                        "sector": sector,
                        "market_cap_focus": market_cap_focus,
                        "cap_classification": (candidate_selection.cap_classifications or {}).get(
                            ticker, {}
                        ),
                        "structural_gate_result": (
                            candidate_selection.structural_gate_results or {}
                        ).get(ticker, {}),
                    }
                    for ticker, disposition in terminal_dispositions.items()
                    if disposition == "STRUCTURAL_SCREENED"
                }
                coverage_evidence_by_ticker.update(
                    {
                        ticker: {
                            "disposition": "CARRIED_VERDICT",
                            "effective_as_of": (split["carried"].get(ticker, {}) or {}).get(
                                "as_of_date"
                            )
                            or as_of,
                            "sector": sector,
                            "market_cap_focus": market_cap_focus,
                            "cap_classification": (
                                candidate_selection.cap_classifications or {}
                            ).get(ticker, {}),
                            "carried_verdict": split["carried"].get(ticker, {}),
                        }
                        for ticker, disposition in terminal_dispositions.items()
                        if disposition == "CARRIED_VERDICT"
                    }
                )
                coverage_record = record_zero_cost_coverage(
                    conn,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    source=candidate_selection.source,
                    tickers=sorted(terminal_dispositions),
                    candidate_dispositions=terminal_dispositions,
                    pipeline_version=resolved_pipeline_version,
                    coverage_campaign_id=normalized_coverage_campaign_id,
                    coverage_evidence_by_ticker=coverage_evidence_by_ticker,
                )
                zero_cost_projection = record_unknown_cap_cross_band_coverage(
                    conn,
                    run_id=str(coverage_record.get("run_id") or "coverage_only"),
                    sector=sector,
                    source=candidate_selection.source,
                    primary_band=market_cap_focus,
                    unknown_cap_tickers=unknown_cap_tickers_from_selection(
                        candidate_selection.to_dict()
                    ),
                    candidate_dispositions=terminal_dispositions,
                    coverage_campaign_id=normalized_coverage_campaign_id,
                    coverage_evidence_by_ticker=coverage_evidence_by_ticker,
                )
                coverage_record["unknown_cap_cross_band_projection"] = zero_cost_projection
            unresolved_without_review = sorted(
                set(loaded_coverage["uncovered"])
                - set(coverage_record["tickers"])
                - set(split["to_review"])
            )
            delta_audit = {
                "only_unswept": True,
                "swept_excluded": split["swept"],
                "carried_verdicts": split["carried"],
                "to_review": split["to_review"],
                "coverage_uncovered": loaded_coverage["uncovered"],
                "coverage_blocked_by_gate_error": sorted(gate_error_tickers),
                "zero_cost_terminal_coverage": coverage_record,
                "unresolved_without_review": unresolved_without_review,
            }
            if not split["to_review"]:
                empty_status = "INCOMPLETE_DELTA" if unresolved_without_review else "EMPTY_DELTA"
                typer.echo(
                    json.dumps(
                        {
                            "scan_family": "normal",
                            "pipeline_version": resolved_pipeline_version,
                            "sector": sector,
                            "market_cap_focus": market_cap_focus,
                            "status": empty_status,
                            "loaded": len(loaded_coverage["loaded"]),
                            "selected": len(split["selected"]),
                            "swept_excluded": len(split["swept"]),
                            "carried_verdicts": {
                                t: v["verdict"] for t, v in split["carried"].items()
                            },
                            "to_review": 0,
                            "coverage_uncovered_before_record": len(loaded_coverage["uncovered"]),
                            "coverage_record": coverage_record,
                            "coverage_blocked_by_gate_error": sorted(gate_error_tickers),
                            "coverage_unresolved": unresolved_without_review,
                            "llm_cost": {"cumulative_cost_usd": 0.0, "call_count": 0},
                            "provider_usage_attestation": zero_provider_usage_attestation,
                            "provider_usage_incremental_attestation": (
                                zero_provider_usage_attestation
                            ),
                        },
                        indent=2,
                    )
                )
                if unresolved_without_review:
                    raise typer.Exit(1)
                return
            # The run receives the filtered list through the same plumbing the
            # --tickers flag feeds; gates and cap integrity already ran in the
            # resolution above.
            candidate_selection.selected_tickers = split["to_review"]
        if resolved_pipeline_version == "v2":
            if not candidate_selection.execution_bound_frozen:
                raise RuntimeError("v2 candidate resolution did not freeze execution authority")
            execution_tickers = list(candidate_selection.execution_tickers)
        else:
            execution_tickers = list(candidate_selection.selected_tickers)
        selection_payload = candidate_selection.to_dict()
        if normalized_coverage_campaign_id:
            from app.config import get_config

            campaign_config = get_config()
            campaign_provider = str(campaign_config.llm_provider or "disabled").strip().lower()
            campaign_model = _campaign_provider_model(
                campaign_config, campaign_provider
            )
            selection_payload["coverage_campaign_id"] = normalized_coverage_campaign_id
            selection_payload["carry_prior_verdicts"] = bool(carry_prior_verdicts)
            selection_payload["coverage_target_file"] = normalized_coverage_target_file
            selection_payload["coverage_expected_provider"] = campaign_provider
            selection_payload["coverage_expected_model"] = str(campaign_model or "")
        if delta_audit is not None:
            selection_payload["delta_audit"] = delta_audit
    except KeyboardInterrupt as exc:
        if resolved_pipeline_version == "v2":
            persist_autonomous_sector_attempt_diagnostic(
                sector=sector,
                market_cap_focus=market_cap_focus,
                objective=objective,
                as_of_date=as_of,
                pipeline_version="v2",
                candidate_selection=(
                    candidate_selection.to_dict() if candidate_selection is not None else {}
                ),
                error=exc,
                last_completed_stage=(
                    "CANDIDATE_SELECTION"
                    if candidate_selection is not None
                    else "PIPELINE_CONFIGURATION"
                ),
            )
        raise
    except Exception as exc:
        if resolved_pipeline_version != "v2":
            raise
        diagnostic_paths = persist_autonomous_sector_attempt_diagnostic(
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=objective,
            as_of_date=as_of,
            pipeline_version="v2",
            candidate_selection=(
                candidate_selection.to_dict() if candidate_selection is not None else {}
            ),
            error=exc,
            last_completed_stage=(
                "CANDIDATE_SELECTION"
                if candidate_selection is not None
                else "PIPELINE_CONFIGURATION"
            ),
        )
        typer.echo(
            json.dumps(
                {
                    "scan_family": "normal",
                    "pipeline_version": "v2",
                    "sector": sector,
                    "market_cap_focus": market_cap_focus,
                    "status": "FAILED",
                    "execution_status": "FAILED",
                    "decision_status": "INCOMPLETE",
                    "final_verdict": None,
                    "artifact_class": "diagnostic",
                    "artifact_path": str(diagnostic_paths.artifact_json),
                    "report_path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            )
        )
        raise typer.Exit(1) from exc
    run_budget = AutonomousRunBudget(
        max_tool_calls=max(0, int(max_tool_calls)),
        max_turns=max(1, int(max_turns)),
        max_cost_usd=max_cost_usd,
        timebox_seconds=None,
        max_candidates=max_candidates,
    )
    cost_context = None
    artifact = None
    try:
        with (
            llm_retry_budget(SECTOR_LLM_RETRY_RUN_BUDGET),
            llm_cost_budget(
                run_budget.max_cost_usd,
                strict_first_call=strict_cost_cap,
            ) as cost_context,
        ):
            artifact = run_sector_autonomous_financial_analysis(
                sector=sector,
                tickers=execution_tickers,
                objective=objective,
                as_of_date=as_of,
                market_cap_focus=market_cap_focus,
                budget=run_budget,
                candidate_selection=selection_payload,
                pipeline_version=resolved_pipeline_version,
            )
            artifact = enrich_sector_artifact_memo_body(artifact)
    except KeyboardInterrupt as exc:
        if resolved_pipeline_version == "v2":
            persist_autonomous_sector_attempt_diagnostic(
                sector=sector,
                market_cap_focus=market_cap_focus,
                objective=objective,
                as_of_date=as_of,
                pipeline_version=resolved_pipeline_version,
                candidate_selection=selection_payload,
                error=exc,
                last_completed_stage=(
                    "SECTOR_ANALYSIS" if artifact is not None else "CANDIDATE_SELECTION"
                ),
                artifact_snapshot=(artifact.to_dict() if artifact is not None else None),
            )
        raise
    except Exception as exc:
        resolved_pipeline = resolved_pipeline_version
        if resolved_pipeline != "v2":
            from app.llm.usage_capture import attached_provider_usage_records

            cost_summary = (
                cost_context.summary()
                if cost_context is not None and hasattr(cost_context, "summary")
                else {}
            )
            attached_usage = attached_provider_usage_records(exc)
            artifact_snapshot = (
                artifact.to_dict() if artifact is not None else None
            )
            unattested_tail = (
                provider_usage_tail_not_in_snapshot(
                    artifact_snapshot,
                    attached_usage,
                )
                if artifact_snapshot is not None
                else attached_usage
            )
            usage_payload: dict[str, Any] = {
                "provider_usage": unattested_tail
            }
            if artifact_snapshot is not None:
                usage_payload["artifact_snapshot"] = artifact_snapshot
            failure_summary = {
                "scan_family": "normal",
                "pipeline_version": "v1",
                "sector": sector,
                "market_cap_focus": market_cap_focus,
                "status": "FAILED",
                "execution_status": "FAILED",
                "decision_status": "INCOMPLETE",
                "artifact_path": None,
                "report_path": None,
                "error": f"{type(exc).__name__}: {exc}",
                "provider_usage": usage_payload["provider_usage"],
                "provider_usage_attestation": provider_usage_attestation(
                    usage_payload
                ),
                "provider_usage_incremental_attestation": (
                    provider_usage_attestation(
                        usage_payload,
                        include_reused=False,
                    )
                ),
                "llm_cost": {
                    key: value
                    for key, value in cost_summary.items()
                    if key != "events"
                },
                "coverage_accounting": {
                    "semantics": "failed_v1_attempt_remains_open",
                    "remaining_uncovered_tickers": list(execution_tickers),
                    "rerun_required": True,
                },
            }
            typer.echo(json.dumps(failure_summary, indent=2))
            raise typer.Exit(1) from exc
        diagnostic_paths = persist_autonomous_sector_attempt_diagnostic(
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=objective,
            as_of_date=as_of,
            pipeline_version=resolved_pipeline,
            candidate_selection=selection_payload,
            error=exc,
            last_completed_stage=(
                "SECTOR_ANALYSIS" if artifact is not None else "CANDIDATE_SELECTION"
            ),
            artifact_snapshot=(artifact.to_dict() if artifact is not None else None),
        )
        cost_summary = (
            cost_context.summary()
            if cost_context is not None and hasattr(cost_context, "summary")
            else {}
        )
        typer.echo(
            json.dumps(
                {
                    "scan_family": "normal",
                    "pipeline_version": "v2",
                    "sector": sector,
                    "market_cap_focus": market_cap_focus,
                    "status": "FAILED",
                    "execution_status": "FAILED",
                    "decision_status": "INCOMPLETE",
                    "final_verdict": None,
                    "artifact_class": "diagnostic",
                    "artifact_path": str(diagnostic_paths.artifact_json),
                    "report_path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "llm_cost": {
                        key: value for key, value in cost_summary.items() if key != "events"
                    },
                    "watchlist_population": {
                        "status": "skipped_failed_artifact",
                        "added_or_updated": 0,
                        "skipped": 0,
                        "entry_ids": [],
                        "skipped_reasons": {
                            "run": (
                                "POST_PROCESSING_EXCEPTION_AFTER_ARTIFACT"
                                if artifact is not None
                                else "RUNTIME_EXCEPTION_BEFORE_ARTIFACT"
                            )
                        },
                    },
                },
                indent=2,
            )
        )
        raise typer.Exit(1) from exc
    # Per-sector LLM spend for sweep summary logs; drop the per-call event list
    # to keep the printed summary compact.
    cost_summary = cost_context.summary() if hasattr(cost_context, "summary") else {}
    llm_cost = {k: v for k, v in cost_summary.items() if k != "events"}
    if artifact.status != "COMPLETED":
        summary = sector_artifact_summary(artifact)
        summary["llm_cost"] = llm_cost
        summary["provider_usage_attestation"] = provider_usage_attestation(
            artifact.to_dict()
        )
        summary["provider_usage_incremental_attestation"] = (
            provider_usage_attestation(
                artifact.to_dict(),
                include_reused=False,
            )
        )
        if str(getattr(artifact, "pipeline_version", "v1") or "v1").lower() == "v1":
            from app.autonomous.sweep_delta import (
                record_loaded_set,
                record_unknown_cap_cross_band_coverage,
                unknown_cap_tickers_from_selection,
                v1_terminal_coverage_from_artifact,
            )

            failed_projection = v1_terminal_coverage_from_artifact(
                artifact,
                fallback_candidate_selection=selection_payload,
                fallback_delta_audit=delta_audit,
            )
            loaded_set = {
                str(ticker).strip().upper()
                for ticker in candidate_selection.loaded_tickers
                if str(ticker).strip()
            }
            zero_cost_recorded = set(
                ((delta_audit or {}).get("zero_cost_terminal_coverage") or {}).get("tickers", [])
            )
            # A failed v1 artifact is not persisted, so even a successful
            # post-run candidate memo call has no durable product artifact to
            # audit.  Keep that name open for retry instead of allowing the
            # in-memory memo payload to earn completed-review coverage.
            failed_candidate_dispositions = {
                ticker: (
                    "LLM_CANDIDATE_REVIEW_UNAUDITABLE"
                    if state == "LLM_CANDIDATE_REVIEW_COMPLETED"
                    else state
                )
                for ticker, state in failed_projection["candidate_dispositions"].items()
            }
            failed_state_tickers = [
                ticker
                for ticker in sorted(failed_candidate_dispositions)
                if ticker in loaded_set and ticker not in zero_cost_recorded
            ]
            failed_terminal_states = {
                "STRUCTURAL_SCREENED",
                "CARRIED_VERDICT",
                "NEEDS_DATA_SPARSE_HISTORY",
                "NEEDS_DATA_FRAMEWORK_EVIDENCE",
            }
            failed_terminal_tickers = {
                ticker
                for ticker in failed_state_tickers
                if failed_candidate_dispositions[ticker] in failed_terminal_states
            }
            failed_incomplete_tickers = set(failed_state_tickers) - (failed_terminal_tickers)
            with get_db() as conn:
                failed_rows_recorded = record_loaded_set(
                    conn,
                    run_id=str(getattr(artifact, "run_id", "") or ""),
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    source=candidate_selection.source,
                    tickers=failed_state_tickers,
                    pipeline_version="v1",
                    candidate_dispositions={
                        ticker: failed_candidate_dispositions[ticker]
                        for ticker in failed_state_tickers
                    },
                    coverage_campaign_id=normalized_coverage_campaign_id,
                )
                failed_unknown_projection = record_unknown_cap_cross_band_coverage(
                    conn,
                    run_id=str(getattr(artifact, "run_id", "") or "failed_run"),
                    sector=sector,
                    source=candidate_selection.source,
                    primary_band=market_cap_focus,
                    unknown_cap_tickers=unknown_cap_tickers_from_selection(selection_payload),
                    candidate_dispositions={
                        ticker: failed_candidate_dispositions[ticker]
                        for ticker in failed_state_tickers
                    },
                    coverage_campaign_id=normalized_coverage_campaign_id,
                )
            coverage_uncovered = (
                {
                    str(ticker).strip().upper()
                    for ticker in delta_audit.get("coverage_uncovered") or []
                    if str(ticker).strip()
                }
                if delta_audit is not None
                else loaded_set
            )
            remaining_uncovered = sorted(
                coverage_uncovered - zero_cost_recorded - failed_terminal_tickers
            )
            summary["coverage_accounting"] = {
                "semantics": "failed_run_explicit_terminal_and_retry_states_only",
                "terminal_tickers_recorded": sorted(failed_terminal_tickers),
                "incomplete_tickers_recorded": sorted(failed_incomplete_tickers),
                "rows_recorded_or_upgraded": failed_rows_recorded,
                "remaining_uncovered_tickers": remaining_uncovered,
                "rerun_required": bool(remaining_uncovered),
            }
            if failed_unknown_projection["tickers"]:
                summary["coverage_accounting"]["unknown_cap_cross_band_projection"] = (
                    failed_unknown_projection
                )
        diagnostic_paths = (
            persist_autonomous_sector_diagnostic(artifact)
            if getattr(artifact, "pipeline_version", "v1") == "v2"
            else None
        )
        summary["artifact_path"] = (
            str(diagnostic_paths.artifact_json) if diagnostic_paths is not None else None
        )
        summary["report_path"] = None
        summary["watchlist_population"] = {
            "status": "skipped_failed_artifact",
            "added_or_updated": 0,
            "skipped": 0,
            "entry_ids": [],
            "skipped_reasons": {"run": f"STATUS_{artifact.status}"},
        }
        typer.echo(json.dumps(summary, indent=2))
        raise typer.Exit(1)
    is_v2_incomplete = (
        getattr(artifact, "pipeline_version", "v1") == "v2"
        and getattr(artifact, "decision_status", None) == "INCOMPLETE"
    )
    product_paths = None
    diagnostic_paths = None
    if is_v2_incomplete:
        diagnostic_paths = persist_autonomous_sector_diagnostic(artifact)
    else:
        product_paths = persist_autonomous_sector_run(artifact)
        from app.autonomous.artifact_financial_audit import (
            run_id_is_decision_eligible,
        )

        if not run_id_is_decision_eligible(artifact.run_id):
            raise RuntimeError(
                "Persisted autonomous run failed exact-byte financial-integrity "
                f"authorization; refusing coverage writes for {artifact.run_id}."
            )
    # Persist only auditable terminal ticker states. Raw loader membership and
    # deterministic per-candidate memo fallbacks never close coverage.
    from app.autonomous.sweep_delta import (
        record_loaded_set,
        record_unknown_cap_cross_band_coverage,
        unknown_cap_tickers_from_selection,
        v1_terminal_coverage_from_artifact,
    )

    artifact_pipeline = str(getattr(artifact, "pipeline_version", "v1") or "v1").lower()
    v1_terminal_coverage: dict[str, Any] | None = None
    with get_db() as conn:
        gate_error_tickers = {
            str(ticker).strip().upper()
            for ticker, result in (candidate_selection.structural_gate_results or {}).items()
            if bool((result or {}).get("excluded_error"))
        }
        if artifact_pipeline == "v2":
            coverage_tickers = [
                ticker
                for ticker in candidate_selection.loaded_tickers
                if str(ticker).strip().upper() not in gate_error_tickers
            ]
            coverage_dispositions = {
                disposition.ticker: disposition.terminal_state
                for disposition in getattr(artifact, "candidate_dispositions", [])
            }
        else:
            v1_terminal_coverage = v1_terminal_coverage_from_artifact(
                artifact,
                fallback_candidate_selection=selection_payload,
                fallback_delta_audit=delta_audit,
            )
            loaded_set = {
                str(ticker).strip().upper()
                for ticker in candidate_selection.loaded_tickers
                if str(ticker).strip()
            }
            zero_cost_recorded = set(
                ((delta_audit or {}).get("zero_cost_terminal_coverage") or {}).get("tickers", [])
            )
            coverage_tickers = [
                ticker
                for ticker in v1_terminal_coverage["disposition_tickers"]
                if ticker in loaded_set
                and ticker not in gate_error_tickers
                and ticker not in zero_cost_recorded
            ]
            coverage_dispositions = {
                ticker: v1_terminal_coverage["candidate_dispositions"][ticker]
                for ticker in coverage_tickers
            }
        coverage_rows_recorded = record_loaded_set(
            conn,
            run_id=str(getattr(artifact, "run_id", "") or ""),
            sector=sector,
            market_cap_focus=market_cap_focus,
            source=candidate_selection.source,
            tickers=coverage_tickers,
            pipeline_version=artifact_pipeline,
            candidate_dispositions=coverage_dispositions,
            coverage_campaign_id=(
                normalized_coverage_campaign_id if artifact_pipeline == "v1" else None
            ),
        )
        unknown_cap_projection = (
            record_unknown_cap_cross_band_coverage(
                conn,
                run_id=str(getattr(artifact, "run_id", "") or "completed_run"),
                sector=sector,
                source=candidate_selection.source,
                primary_band=market_cap_focus,
                unknown_cap_tickers=unknown_cap_tickers_from_selection(selection_payload),
                candidate_dispositions=coverage_dispositions,
                coverage_campaign_id=normalized_coverage_campaign_id,
            )
            if artifact_pipeline == "v1"
            else {"tickers": [], "bands": [], "rows_inserted_or_upgraded": 0}
        )
    summary = sector_artifact_summary(artifact)
    summary["llm_cost"] = llm_cost
    summary["provider_usage_attestation"] = provider_usage_attestation(
        artifact.to_dict()
    )
    summary["provider_usage_incremental_attestation"] = (
        provider_usage_attestation(
            artifact.to_dict(),
            include_reused=False,
        )
    )
    summary["artifact_path"] = str(
        diagnostic_paths.artifact_json
        if diagnostic_paths is not None
        else product_paths.artifact_json
    )
    summary["report_path"] = str(product_paths.report_md) if product_paths is not None else None
    summary["artifact_class"] = "diagnostic" if diagnostic_paths is not None else "product"
    if v1_terminal_coverage is not None:
        terminal_set = set(v1_terminal_coverage["terminal_tickers"])
        coverage_uncovered = set((delta_audit or {}).get("coverage_uncovered") or [])
        coverage_remaining = sorted(coverage_uncovered - terminal_set)
        summary["coverage_accounting"] = {
            "semantics": "auditable_terminal_state_not_raw_loader_presence",
            "llm_candidate_review_completed_tickers": v1_terminal_coverage[
                "llm_candidate_review_completed"
            ],
            "llm_candidate_review_failed_tickers": v1_terminal_coverage[
                "llm_candidate_review_failed"
            ],
            "structural_screened_tickers": v1_terminal_coverage["structural_screened"],
            "carried_verdict_tickers": v1_terminal_coverage["carried_verdict"],
            "needs_data_sparse_history_tickers": v1_terminal_coverage["needs_data_sparse_history"],
            "needs_data_framework_evidence_tickers": v1_terminal_coverage[
                "needs_data_framework_evidence"
            ],
            "terminal_tickers_recorded": sorted(
                set(coverage_tickers) & set(v1_terminal_coverage["terminal_tickers"])
            ),
            "incomplete_tickers_recorded": sorted(
                set(coverage_tickers) - set(v1_terminal_coverage["terminal_tickers"])
            ),
            "rows_recorded_or_upgraded": coverage_rows_recorded,
            "remaining_uncovered_tickers": coverage_remaining,
            "rerun_required": bool(coverage_remaining),
        }
        if unknown_cap_projection["tickers"]:
            summary["coverage_accounting"]["unknown_cap_cross_band_projection"] = (
                unknown_cap_projection
            )
    if delta_audit is not None:
        summary["delta_audit"] = {
            "swept_excluded": len(delta_audit["swept_excluded"]),
            "carried_verdicts": {
                t: v["verdict"] for t, v in delta_audit["carried_verdicts"].items()
            },
            "to_review": len(delta_audit["to_review"]),
            "llm_candidate_review_completed": len(
                (v1_terminal_coverage or {}).get("llm_candidate_review_completed", [])
            ),
            "llm_candidate_review_failed": len(
                (v1_terminal_coverage or {}).get("llm_candidate_review_failed", [])
            ),
            "remaining_uncovered": len(
                (summary.get("coverage_accounting") or {}).get("remaining_uncovered_tickers", [])
            ),
        }
    if populate_watchlist:
        from app.watchlist.store import populate_from_sector_artifact

        watchlist_result = populate_from_sector_artifact(artifact)
        summary["watchlist_population"] = {
            "status": "populated",
            "added_or_updated": watchlist_result.added_or_updated,
            "skipped": watchlist_result.skipped,
            "entry_ids": watchlist_result.entry_ids,
            "skipped_reasons": watchlist_result.skipped_reasons,
            "adverse_check": watchlist_result.adverse_check,
        }
    else:
        summary["watchlist_population"] = {
            "status": "skipped",
            "added_or_updated": 0,
            "skipped": 0,
            "entry_ids": [],
            "skipped_reasons": {},
        }
    typer.echo(json.dumps(summary, indent=2))


@watchlist_app.command("add")
def watchlist_add_cmd(
    ticker: str = typer.Argument(..., help="Ticker to add to the watchlist"),
    thesis: str = typer.Option(..., "--thesis", help="Manual thesis text"),
    buy_price: float = typer.Option(..., "--buy-price", help="Buy-price target"),
    conviction: str = typer.Option("WATCHLIST_ONLY", "--conviction", help="Conviction grade"),
    source_note: str | None = typer.Option(
        None, "--source-note", help="Optional source/context note"
    ),
) -> None:
    """Reject unaudited manual additions to the decision-bearing watchlist."""
    configure_logging()
    typer.echo(
        "Manual watchlist additions are disabled: current decision rows require "
        "exact authorized run/ticker lineage; no price was fetched and no entry "
        "was created.",
        err=True,
    )
    raise typer.Exit(code=1)


@watchlist_app.command("list")
def watchlist_list_cmd(
    status: str | None = typer.Option(None, "--status", help="Filter by watchlist status"),
    sector: str | None = typer.Option(None, "--sector", help="Filter by source sector"),
    scan_family: str | None = typer.Option(
        None, "--scan-family", help="Filter by scan family: normal or pearl"
    ),
) -> None:
    """List latest non-removed watchlist entries."""
    configure_logging()

    from app.watchlist.contract import is_price_trigger_eligible
    from app.watchlist.store import list_active

    try:
        entries = list_active(status=status, sector=sector, scan_family=scan_family)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if not entries:
        typer.echo("No watchlist entries found.")
        return

    lines = [
        "| Ticker | Status | Conviction | Confidence | Conviction Source | Scan Family | Valuation Anchor | Buy Price | Current Price | Distance From Buy | Source Sector | Added |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        price_eligible = is_price_trigger_eligible(entry)
        presented_status = entry.status
        if not price_eligible and entry.status in {"DEPLOY_READY", "BUY_CONFIRMED"}:
            presented_status = (
                "UNCERTAIN"
                if str(entry.conviction_grade or "").upper() == "DATA_INCOMPLETE"
                else "ACTIVE"
            )
        anchor = (
            f"{entry.valuation_anchor_method or 'anchor'} {_format_money(entry.valuation_anchor_value)}"
            if entry.valuation_anchor_value is not None
            else "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    entry.ticker,
                    presented_status,
                    entry.conviction_grade or "n/a",
                    entry.confidence or "n/a",
                    entry.conviction_source or "n/a",
                    entry.scan_family,
                    anchor,
                    _format_money(entry.buy_price_target if price_eligible else None),
                    _format_money(entry.current_price_at_addition),
                    _format_pct(
                        _distance_from_buy(
                            entry.current_price_at_addition,
                            entry.buy_price_target if price_eligible else None,
                        )
                    ),
                    entry.source_sector or "n/a",
                    entry.added_at,
                ]
            )
            + " |"
        )
    typer.echo("\n".join(lines))


@watchlist_app.command("show")
def watchlist_show_cmd(ticker: str = typer.Argument(..., help="Ticker to inspect")) -> None:
    """Show full watchlist entry detail plus abbreviated history."""
    configure_logging()

    from app.watchlist.contract import is_price_trigger_eligible
    from app.watchlist.store import get_history, get_latest

    entry = get_latest(ticker)
    if entry is None:
        raise typer.BadParameter(f"No watchlist entry found for {ticker.upper()}")

    price_eligible = is_price_trigger_eligible(entry)
    presented_status = entry.status
    if not price_eligible and entry.status in {"DEPLOY_READY", "BUY_CONFIRMED"}:
        presented_status = (
            "UNCERTAIN"
            if str(entry.conviction_grade or "").upper() == "DATA_INCOMPLETE"
            else "ACTIVE"
        )
    elif entry.event_pending and entry.status in {"DEPLOY_READY", "BUY_CONFIRMED"}:
        presented_status = f"EVENT_PENDING (stored: {entry.status})"
    lines = [
        f"# {entry.ticker} Watchlist Entry",
        "",
        f"- Status: {presented_status}",
        f"- Events pending: {entry.event_pending or 'none'}",
        f"- Conviction: {entry.conviction_grade or 'n/a'}",
        f"- Confidence: {entry.confidence or 'n/a'}",
        f"- Conviction source: {entry.conviction_source or 'n/a'}",
        f"- Scan family: {entry.scan_family}",
        f"- Valuation anchor: {entry.valuation_anchor_method or 'n/a'} {_format_money(entry.valuation_anchor_value)}",
        f"- Investable: {'yes' if price_eligible else 'no — research only'}",
    ]
    if price_eligible:
        lines.extend(
            [
                f"- Buy price target: {_format_money(entry.buy_price_target)}",
                f"- Current price at addition: {_format_money(entry.current_price_at_addition)}",
                f"- Distance from buy: {_format_pct(_distance_from_buy(entry.current_price_at_addition, entry.buy_price_target))}",
            ]
        )
    else:
        lines.append(
            f"- Current price at addition: {_format_money(entry.current_price_at_addition)}"
        )
    lines.extend(
        [
            f"- Source sector: {entry.source_sector or 'n/a'}",
            f"- Source run: {entry.source_run_id}",
            f"- Added: {entry.added_at}",
            f"- Last evaluated: {entry.last_evaluated_at or 'n/a'}",
            f"- Status reason: {entry.status_reason or 'n/a'}",
            "",
            "## Thesis",
            entry.thesis_text or "No thesis captured.",
            "",
            "## Key Risks",
        ]
    )
    lines.extend([f"- {item}" for item in entry.key_risks] or ["- n/a"])
    lines.extend(["", "## Falsifiers"])
    lines.extend([f"- {item}" for item in entry.falsifiers] or ["- n/a"])
    lines.extend(["", "## Open Questions"])
    lines.extend([f"- {item}" for item in entry.open_questions] or ["- n/a"])

    from app.db import get_db as _get_db
    from app.events.cheapness import latest_cheapness_by_ticker, render_cheapness_block

    with _get_db() as _conn:
        cheapness_report = latest_cheapness_by_ticker(_conn, [entry.ticker]).get(entry.ticker)
    lines.extend(["", "## Known Reasons It May Be Cheap"])
    lines.extend(render_cheapness_block(cheapness_report))
    lines.extend(
        [
            "",
            "## History",
            "| Changed At | Field | Old | New | Source |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in get_history(entry.ticker)[-10:]:
        lines.append(
            f"| {row['changed_at']} | {row['field_name']} | {row['old_value'] or ''} | "
            f"{row['new_value'] or ''} | {row['source']} |"
        )
    typer.echo("\n".join(lines))


@watchlist_app.command("remove")
def watchlist_remove_cmd(
    ticker: str = typer.Argument(..., help="Ticker to soft-delete"),
    reason: str = typer.Option(..., "--reason", help="Removal reason"),
) -> None:
    """Soft-delete a ticker from the active watchlist."""
    configure_logging()

    from app.watchlist.store import remove

    remove(ticker, reason)
    typer.echo(f"Removed {ticker.upper()} from the active watchlist.")


@watchlist_app.command("backfill-adv")
def watchlist_backfill_adv_cmd(
    days: int = typer.Option(90, "--days", help="Calendar days of daily history to seed"),
    ticker: list[str] = typer.Option(
        None,
        "--ticker",
        help="Restrict to specific tickers (repeatable); default = all live watchlist names",
    ),
) -> None:
    """Seed daily close+volume history and persist dollar-ADV/capacity.

    One full-history download per symbol (provider cache serves every anchor
    date), rows land in price_quotes idempotently, then adv_dollar_20d/60d +
    capacity_class are persisted on the live watchlist rows.
    """
    configure_logging()

    from app.market.adv import backfill_watchlist_price_history

    summary = backfill_watchlist_price_history(
        days=days, tickers=[t.upper() for t in ticker] if ticker else None
    )
    known = sum(1 for adv in summary["adv"].values() if adv.get("adv_dollar_20d") is not None)
    typer.echo(
        json.dumps(
            {
                "tickers": summary["tickers"],
                "anchor_days": summary["anchor_days"],
                "rows_written": summary["rows_written"],
                "failed_tickers": summary["failed_tickers"],
                "adv_resolved": known,
                "adv_unknown": summary["tickers"] - known,
            },
            indent=2,
        )
    )


@watchlist_app.command("stats")
def watchlist_stats_cmd(
    all_rows: bool = typer.Option(
        False,
        "--all-rows",
        help="Show historical/source-run row totals instead of latest unique tickers",
    ),
) -> None:
    """Show watchlist counts by status and source sector."""
    configure_logging()

    from app.watchlist.store import stats

    payload = stats(all_rows=all_rows)
    mode_label = "all rows" if payload.get("mode") == "all_rows" else "current unique tickers"
    lines = [
        "# Watchlist Stats",
        "",
        f"- Mode: {mode_label}",
        f"- Total entries: {payload['total_entries']}",
        f"- Unique tickers: {payload['unique_tickers']}",
        f"- All non-removed rows: {payload['all_rows_total']}",
        f"- Duplicate non-removed rows: {payload['duplicate_rows']}",
        f"- Oldest entry: {payload['oldest_entry'] or 'n/a'}",
        f"- Newest entry: {payload['newest_entry'] or 'n/a'}",
        "",
        "## By Status",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in payload["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## By Sector", "| Sector | Count |", "| --- | ---: |"])
    for sector, count in payload["sector_counts"].items():
        lines.append(f"| {sector} | {count} |")
    typer.echo("\n".join(lines))


@watchlist_app.command("queue")
def watchlist_queue_cmd(
    limit: int = typer.Option(25, "--limit", help="Maximum rows to show"),
    sector: str | None = typer.Option(None, "--sector", help="Filter by source sector"),
    scan_family: str | None = typer.Option(
        None, "--scan-family", help="Filter by scan family: normal or pearl"
    ),
    include_price_suspect: bool = typer.Option(
        False,
        "--include-price-suspect",
        help="Include rows currently flagged with suspect price data",
    ),
    band: str | None = typer.Option(
        None,
        "--band",
        help=(
            "Cap-band scope (e.g. micro_cap, small_cap, mid_cap, smid_cap). "
            "Band integrity: only rows with a computable in-band market cap "
            "qualify; UNKNOWN_CAP rows never present as in-band output."
        ),
    ),
    compact: bool = typer.Option(
        False,
        "--compact",
        help=(
            "Compact signal board: Signal/Ticker/Conviction/Confidence/Price/"
            "Buy Target/Distance/Status, sorted with live buy signals on top."
        ),
    ),
) -> None:
    """Show the current watchlist review queue ranked by decision priority."""
    configure_logging()

    from app.watchlist.store import watchlist_queue

    if limit <= 0:
        raise typer.BadParameter("--limit must be positive")

    try:
        rows = watchlist_queue(
            limit=limit,
            sector=sector,
            scan_family=scan_family,
            include_price_suspect=include_price_suspect,
            band=band,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if not rows:
        typer.echo("No watchlist queue entries found.")
        return

    if compact:
        from app.watchlist.digest import render_compact_signal_table

        typer.echo("\n".join(render_compact_signal_table(rows)))
        return

    lines = [
        "| Ticker | Status | Conviction | Confidence | Conviction Source | Scan Family | Latest/Add Price | Buy Target | Distance From Buy | Band | Events | Why Cheap | Source Sector | Reason |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        price_eligible = bool(row.get("price_trigger_eligible", True))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ticker"]),
                    str(row.get("presented_status") or row["status"] or "n/a"),
                    str(row["conviction_grade"] or "n/a"),
                    str(row["confidence"] or "n/a"),
                    str(row["conviction_source"] or "n/a"),
                    str(row.get("scan_family") or "normal"),
                    _format_money(row["latest_price"]),
                    _format_money(row["buy_price_target"] if price_eligible else None),
                    _format_pct_points(row["distance_from_buy_pct"] if price_eligible else None),
                    str(row.get("cap_band_label") or "UNKNOWN_CAP"),
                    str(row.get("event_pending") or "—"),
                    str(row.get("cheapness") or "n/a"),
                    str(row["source_sector"] or "n/a"),
                    str(row["status_reason"] or "n/a").replace("\n", " "),
                ]
            )
            + " |"
        )
    typer.echo("\n".join(lines))


@watchlist_app.command("backfill-cap-structural")
def watchlist_backfill_cap_structural_cmd(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without writing"),
    as_of: str | None = typer.Option(
        None, "--as-of", help="Classification as-of date YYYY-MM-DD (default today)"
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Report JSON path (default data/outputs/diagnostics/cap_structural_backfill_<asof>.json)",
    ),
) -> None:
    """Backfill cap chain + structural gate over non-removed watchlist rows."""
    configure_logging()

    from app.config import get_config
    from app.watchlist.cap_structural_backfill import backfill_watchlist_cap_and_structural

    report = backfill_watchlist_cap_and_structural(as_of_date=as_of, dry_run=dry_run)
    out_path = (
        Path(output)
        if output
        else (
            get_config().outputs_dir
            / "diagnostics"
            / f"cap_structural_backfill_{report['as_of_date']}{'_dryrun' if dry_run else ''}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "as_of_date": report["as_of_date"],
        "dry_run": report["dry_run"],
        "counts": report["counts"],
        "changed_tickers": sorted(
            {row["ticker"] for row in report["changed_rows"] if row["changes"]}
        ),
        "quarantined": sorted(
            {
                row["ticker"]
                for row in report["changed_rows"]
                if row["new_status"] == "QUARANTINE" and row["prior_status"] != "QUARANTINE"
            }
        ),
        "routed_out_of_band": sorted(
            {row["ticker"] for row in report["changed_rows"] if row["routed_out_of_band"]}
        ),
        "report_path": str(out_path),
    }
    typer.echo(json.dumps(summary, indent=2))


@watchlist_app.command("rederive-targets")
def watchlist_rederive_targets_cmd(
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Write RETARGET rows (default: dry-run report only)",
    ),
    tickers: str | None = typer.Option(
        None, "--tickers", help="Optional comma-separated ticker subset"
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Report JSON path (default data/outputs/diagnostics/target_rederive_<date>[_dryrun].json)",
    ),
) -> None:
    """Re-derive anchors + buy targets through the live packet path (decline cap et al.).

    Dry-run by default. WOULD_NULL rows (production would derive no target
    today) are never written — triage them instead of letting a NULL silently
    disable the price trigger. Statuses are trigger-owned: run the trigger
    pass after applying so DEPLOY_READY reflects corrected targets.
    """
    configure_logging()

    from datetime import datetime, timezone

    from app.config import get_config
    from app.watchlist.target_rederive import rederive_watchlist_targets

    report = rederive_watchlist_targets(
        apply=apply, tickers=_parse_tickers_csv(tickers) if tickers else None
    )
    date_token = datetime.now(timezone.utc).date().isoformat()
    out_path = (
        Path(output)
        if output
        else (
            get_config().outputs_dir
            / "diagnostics"
            / f"target_rederive_{date_token}{'' if apply else '_dryrun'}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Target Re-Derivation ({'APPLIED' if apply else 'dry-run'})",
        f"counts: {json.dumps(report['counts'])}",
        f"report: {out_path}",
    ]
    interesting = [
        row for row in report["rows"] if row["disposition"] in {"RETARGET", "WOULD_NULL", "ERROR"}
    ]
    if interesting:
        lines.append("")
        lines.append(
            "| Ticker | Grade | Trend | Old target | New target | Old->New method | Disposition |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
        for row in interesting:
            old_t = f"${row['old_buy_target']:.2f}" if row["old_buy_target"] is not None else "—"
            new_t = f"${row['new_buy_target']:.2f}" if row["new_buy_target"] is not None else "—"
            lines.append(
                f"| {row['ticker']} | {row['conviction_grade'] or 'n/a'} | "
                f"{row['revenue_trend_class'] or 'n/a'} | {old_t} | {new_t} | "
                f"{row['old_anchor_method'] or '—'}->{row['new_anchor_method'] or '—'} | "
                f"{row['disposition']}{(' ' + row['detail']) if row['detail'] else ''} |"
            )
    if apply and report["counts"].get("RETARGET"):
        lines.append("")
        lines.append(
            "NOTE: statuses are trigger-owned — run the watchlist trigger pass "
            "so DEPLOY_READY reflects the corrected targets."
        )
    typer.echo("\n".join(lines))


@app.command("cap-census-resolve")
def cap_census_resolve_cmd(
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD (default today)"),
    max_tickers: int | None = typer.Option(
        None, "--max-tickers", help="Optional cap for a quick sample run"
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Report JSON path (default data/outputs/diagnostics/cap_census_resolution_<asof>.json)",
    ),
) -> None:
    """Run the unknown-cap census through the stale-shares fallback tier."""
    configure_logging()

    from app.config import get_config
    from app.autonomous.cap_census import census_cap_resolution

    report = census_cap_resolution(as_of_date=as_of, max_tickers=max_tickers)
    out_path = (
        Path(output)
        if output
        else (
            get_config().outputs_dir
            / "diagnostics"
            / f"cap_census_resolution_{report['as_of_date']}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps({"counts": report["counts"], "report_path": str(out_path)}, indent=2))


@watchlist_app.command("refresh")
def watchlist_refresh_cmd(
    ticker: str | None = typer.Argument(
        None, help="Optional ticker to refresh; omitted refreshes active entries"
    ),
    since: str | None = typer.Option(
        None, "--since", help="Override evidence cutoff date YYYY-MM-DD"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Detect evidence without LLM calls or DB updates"
    ),
    max_cost_usd: float | None = typer.Option(
        None, "--max-cost-usd", help="Maximum cumulative LLM spend for this refresh"
    ),
) -> None:
    """Re-evaluate watchlist theses against new filings/events."""
    configure_logging()

    from app.watchlist.reevaluation import refresh_watchlist

    summary = refresh_watchlist(
        ticker=ticker,
        since=since,
        dry_run=dry_run,
        max_cost_usd=max_cost_usd,
    )
    lines = [
        "| Ticker | Prior Status | New Status | Evaluation | Cost |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for result in summary.results:
        lines.append(
            f"| {result.ticker} | {result.prior_status} | {result.new_status} | "
            f"{result.evaluation} | ${result.cost_estimate_usd:.2f} |"
        )
    if not summary.results:
        lines.append("| n/a | n/a | n/a | NO_ENTRIES | $0.00 |")
    lines.extend(
        [
            "",
            f"Total estimated LLM cost: ${summary.total_cost_estimate_usd:.2f}",
            f"Refresh run: {summary.run_id}",
        ]
    )
    typer.echo("\n".join(lines))


@watchlist_app.command("check-triggers")
def watchlist_check_triggers_cmd(
    ticker: str | None = typer.Argument(
        None, help="Optional ticker to check; omitted checks trigger-eligible entries"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report trigger transitions without DB updates"
    ),
    force: bool = typer.Option(
        False, "--force", help="Allow explicit ticker checks for CONTRADICTED/RESOLVED entries"
    ),
) -> None:
    """Check current prices against watchlist buy-price targets."""
    configure_logging()

    from app.watchlist.triggers import check_watchlist_triggers

    results = check_watchlist_triggers(ticker=ticker, dry_run=dry_run, force=force)
    lines = [_format_watchlist_trigger_results(results)]
    if dry_run:
        lines.extend(["", "(dry run — no watchlist rows or price snapshots updated)"])
    typer.echo("\n".join(lines))


@watchlist_app.command("price-refresh")
def watchlist_price_refresh_cmd(
    as_of: str | None = typer.Option(
        None, "--as-of", help="Quote as-of date YYYY-MM-DD (today only; default: today)"
    ),
) -> None:
    """Refresh factual quotes for canonical current rows without decision actions."""

    configure_logging()
    from app.watchlist.market_data import (
        PriceRefreshMigrationRequired,
        refresh_current_watchlist_prices,
    )

    today = date.today().isoformat()
    if as_of is not None and as_of.strip()[:10] != today:
        # Backdated closes from the free chain can be retroactively
        # split-adjusted; stamping them UNADJUSTED/factor-1.0 would lie about
        # the basis. The refresh is a current-fact writer, so it is
        # current-date only.
        typer.echo(
            json.dumps(
                {
                    "status": "FAILED",
                    "error": f"--as-of must be today ({today}); backdated refreshes "
                    "cannot certify an UNADJUSTED price basis",
                },
                indent=2,
            )
        )
        raise typer.Exit(code=2)
    try:
        summary = refresh_current_watchlist_prices(as_of_date=as_of)
    except (PriceRefreshMigrationRequired, ValueError) as exc:
        typer.echo(json.dumps({"status": "FAILED", "error": str(exc)}, indent=2))
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(summary.to_dict(), indent=2))
    if summary.exit_code:
        raise typer.Exit(code=summary.exit_code)


@watchlist_app.command("digest")
def watchlist_digest_cmd(
    days_back: int = typer.Option(1, "--days-back", help="Number of trailing days to include"),
    output: str | None = typer.Option(
        None, "--output", help="Optional digest Markdown output path"
    ),
    check_prices_first: bool = typer.Option(
        False, "--check-prices-first", help="Run trigger checks before rendering"
    ),
) -> None:
    """Render the daily watchlist digest."""
    configure_logging()

    from app.watchlist.digest import write_digest

    result = write_digest(
        days_back=max(0, int(days_back)),
        output_path=output,
        check_prices_first=check_prices_first,
    )
    typer.echo(result.markdown, nl=False)


@watchlist_app.command("actionable-counterfactual")
def watchlist_actionable_counterfactual_cmd(
    hurdle_pct: float = typer.Option(
        12.0, "--hurdle-pct", help="Base-case return hurdle percentage"
    ),
    allow_evidence_caps: bool = typer.Option(
        False,
        "--allow-evidence-caps",
        help="Allow evidence-quality audit signals to stop blocking ACTIONABLE in the counterfactual",
    ),
    output: str | None = typer.Option(None, "--output", help="Optional Markdown output path"),
) -> None:
    """Render a read-only ACTIONABLE counterfactual for active watchlist entries."""
    configure_logging()

    if hurdle_pct <= 0:
        raise typer.BadParameter("--hurdle-pct must be positive")
    summary = _run_actionable_counterfactual(
        hurdle_pct=hurdle_pct,
        allow_evidence_caps=allow_evidence_caps,
    )
    output_path = _write_actionable_counterfactual(summary, output_path=output)
    lines = [
        _format_actionable_counterfactual_table(summary),
        "",
        f"Would shift to ACTIONABLE: {summary['shift_to_actionable_count']}",
        f"Resolve-then-promote (DATA_INCOMPLETE): {summary['resolve_then_promote_count']}",
        f"Counterfactual written: {output_path}",
    ]
    typer.echo("\n".join(lines))


@watchlist_app.command("relight-plan")
def watchlist_relight_plan_cmd(
    tickers: str | None = typer.Option(
        None, "--tickers", help="Explicit comma-separated tickers to re-review"
    ),
    from_blocked: bool = typer.Option(
        False,
        "--from-blocked",
        help=(
            "Add every current at-target / open-disposition row the provenance "
            "gate blocks, read from the database."
        ),
    ),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
    cell_order: str | None = typer.Option(
        None,
        "--cell-order",
        help="Comma-separated exact permutation of all planned sector:band cell IDs",
    ),
    no_watchlist: bool = typer.Option(
        False, "--no-watchlist", help="Do not populate the watchlist from the re-review"
    ),
) -> None:
    """Plan a scoped provenance re-review. Performs no LLM call and spends nothing."""
    configure_logging()

    from app.watchlist.relight import (
        RelightError,
        parse_cell_order,
        parse_ticker_list,
        plan_relight,
    )

    try:
        plan = plan_relight(
            tickers=parse_ticker_list(tickers),
            from_blocked=from_blocked,
            as_of=as_of,
            populate_watchlist=not no_watchlist,
            cell_order=parse_cell_order(cell_order),
        )
    except RelightError as exc:
        typer.echo(f"Relight plan refused: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    estimate = plan["cost_estimate"]
    lines = [
        f"Relight planned: {plan['relight_id']}",
        f"  tickers routed : {len(plan['routed_tickers'])}",
        f"  cells          : {len(plan['cells'])}",
        f"  provider/model : {plan['provider_binding']['provider']}/"
        f"{plan['provider_binding']['model']}",
        f"  estimated cost : ${estimate['estimated_cost_usd']:.2f} "
        f"(${estimate['estimated_cost_per_review_usd']:.4f}/review + "
        f"${estimate['estimated_base_cost_per_cell_usd']:.4f}/cell, "
        f"{estimate['safety_multiplier']}x reserve)",
        f"  method         : {estimate['method']} "
        f"({estimate['matching_historical_runs']} matching runs)",
    ]
    if plan["unroutable"]:
        lines.append(f"  UNROUTABLE     : {len(plan['unroutable'])}")
        for item in plan["unroutable"][:20]:
            lines.append(f"    - {item['ticker']}: {','.join(item['reasons'])}")
    if not plan["preflight"]["ready"]:
        lines.append("  PREFLIGHT BLOCKED — execution will refuse to spend:")
        for blocker in plan["preflight"]["blockers"]:
            lines.append(f"    - {blocker}")
    lines.append("")
    lines.append("Cells:")
    for cell in plan["cells"]:
        per_cell = estimate["per_cell_estimated_cost_usd"].get(cell["cell_id"], 0.0)
        lines.append(
            f"  {cell['cell_id']:<44s} {len(cell['tickers']):3d} tickers  "
            f"${per_cell:6.3f}  {','.join(cell['tickers'])}"
        )
    lines.append("")
    lines.append(
        "Nothing has been spent. Authorize with: "
        f"ivi watchlist relight-run --relight-id {plan['relight_id']} --budget-usd <USD>"
    )
    typer.echo("\n".join(lines))


@watchlist_app.command("relight-run")
def watchlist_relight_run_cmd(
    relight_id: str = typer.Option(..., "--relight-id", help="Planned relight campaign id"),
    budget_usd: float | None = typer.Option(
        None,
        "--budget-usd",
        help=(
            "REQUIRED hard ceiling in USD for this relight. An absent budget "
            "authorizes nothing and the command refuses to run."
        ),
    ),
    accept_estimate_shortfall: bool = typer.Option(
        False,
        "--accept-estimate-shortfall",
        help="Permit a ceiling below the plan estimate, knowingly running a partial relight.",
    ),
    cell_reservation_floor_usd: float = typer.Option(
        0.50,
        "--cell-reservation-floor-usd",
        help=(
            "Minimum strict child cost cap in USD. A cell does not start when "
            "less than this amount remains under the total campaign ceiling."
        ),
    ),
) -> None:
    """Execute a planned relight under a hard budget ceiling. Resumable."""
    configure_logging()

    from app.watchlist.relight import RelightError, execute_relight

    try:
        state = execute_relight(
            relight_id=relight_id,
            budget_usd=budget_usd,
            accept_estimate_shortfall=accept_estimate_shortfall,
            cell_reservation_floor_usd=cell_reservation_floor_usd,
            progress=lambda message: typer.echo(f"  {message}"),
        )
    except RelightError as exc:
        typer.echo(f"Relight refused: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    eligibility = state.get("eligibility") or {}
    typer.echo("")
    typer.echo(f"Relight {state['status']}: {state['relight_id']}")
    typer.echo(
        f"  spent ${float(state.get('cumulative_cost_usd') or 0.0):.4f} "
        f"of ${float(state.get('authorized_max_cost_usd') or 0.0):.2f} authorized"
    )
    typer.echo(f"  decision-eligible runs : {len(eligibility.get('decision_eligible_run_ids', []))}")
    typer.echo(f"  still ineligible runs  : {len(eligibility.get('ineligible_run_ids', []))}")
    if state["status"] != "COMPLETED":
        raise typer.Exit(code=1)


@watchlist_app.command("relight-bootstrap-valuations")
def watchlist_relight_bootstrap_valuations_cmd(
    relight_id: str = typer.Option(
        ...,
        "--relight-id",
        help="Existing relight campaign whose routed READY population will be bootstrapped",
    ),
) -> None:
    """Publish deterministic valuation rows for a relight cohort at zero spend."""
    configure_logging()

    from app.valuation.bootstrap import bootstrap_relight_valuations
    from app.watchlist.relight import RelightError

    try:
        summary = bootstrap_relight_valuations(
            relight_id=relight_id,
            progress=lambda message: typer.echo(f"  {message}"),
        )
    except RelightError as exc:
        typer.echo(f"Valuation bootstrap refused: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo("")
    typer.echo(f"Valuation bootstrap: {summary['relight_id']}")
    typer.echo(
        f"  split evidence : READY {summary['split_ready']} / "
        f"UNKNOWN {summary['split_unknown']}"
    )
    typer.echo(
        f"  valuation rows : PUBLISHED {summary['PUBLISHED']} / "
        f"ALREADY_AUTHORIZED {summary['ALREADY_AUTHORIZED']} / "
        f"NEEDS_DATA {summary['NEEDS_DATA']} / FAILED {summary['FAILED']}"
    )
    typer.echo(
        f"  provider calls : {summary['provider_calls']} "
        f"(LLM cost ${float(summary['llm_cost_usd']):.2f})"
    )
    for row in summary["results"]:
        if row["status"] in {"PUBLISHED", "ALREADY_AUTHORIZED"}:
            continue
        typer.echo(
            f"    - {row['ticker']}: {row['status']} "
            f"{row.get('reason') or ''}".rstrip()
        )
    if int(summary["FAILED"]) > 0:
        raise typer.Exit(code=1)


@watchlist_app.command("relight-status")
def watchlist_relight_status_cmd(
    relight_id: str = typer.Option(..., "--relight-id", help="Planned relight campaign id"),
) -> None:
    """Read a relight checkpoint without mutating it."""
    configure_logging()

    from app.watchlist.relight import RelightError, relight_status

    try:
        typer.echo(json.dumps(relight_status(relight_id=relight_id), indent=2))
    except RelightError as exc:
        typer.echo(f"Relight status unavailable: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@watchlist_app.command("daily")
def watchlist_daily_cmd(
    digest_output: str | None = typer.Option(
        None, "--digest-output", help="Optional digest Markdown output path"
    ),
    no_prices: bool = typer.Option(False, "--no-prices", help="Skip buy-price trigger checks"),
    no_digest: bool = typer.Option(False, "--no-digest", help="Skip digest rendering"),
) -> None:
    """Run the cron-friendly daily watchlist routine."""
    configure_logging()

    if no_prices and no_digest:
        typer.echo("Nothing to do: --no-prices and --no-digest cannot both be set.", err=True)
        raise typer.Exit(code=1)

    if not no_prices:
        from app.watchlist.triggers import check_watchlist_triggers

        trigger_results = check_watchlist_triggers()
        typer.echo("## Price Trigger Check")
        typer.echo("")
        typer.echo(_format_watchlist_trigger_results(trigger_results))

        # Every presentable at-target name opens a disposition record —
        # the mandatory terminus of the surfacing. Idempotent per day.
        from app.watchlist.dispositions import sync_at_target_dispositions

        disposition_sync = sync_at_target_dispositions()
        if disposition_sync["opened"]:
            typer.echo("")
            typer.echo(
                f"Opened {disposition_sync['opened']} at-target disposition(s) — "
                "close with `ivi investor journal <TICKER> ...`"
            )
        if no_digest:
            return
        typer.echo("")

    if not no_digest:
        from app.watchlist.digest import write_digest

        result = write_digest(output_path=digest_output, check_prices_first=False)
        typer.echo(f"Digest written: {result.path}")
        typer.echo("")
        typer.echo(result.markdown, nl=False)


@app.command("autonomous-sector-benchmark")
def autonomous_sector_benchmark_cmd(
    sectors: str = typer.Option(
        "enterprise_software,diversified_industrials,semiconductors,medical_devices",
        "--sectors",
        help="Comma-separated sector labels to benchmark",
    ),
    objective: str = typer.Option(
        "Determine which company has the strongest financially underwritten 5-10 year per-share return potential, or return no selection if the evidence does not clear the bar.",
        "--objective",
        help="Run objective used for each autonomous sector analyst run.",
    ),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
    market_cap_focus: str = typer.Option(
        "smid_cap", "--market-cap-focus", help="Market-cap focus label"
    ),
    pipeline_version: str | None = typer.Option(
        None,
        "--pipeline-version",
        help="Explicit sector pipeline version (v1 or v2); omitted uses the cap-band rollout allowlist.",
    ),
    max_tool_calls: int = typer.Option(
        16, "--max-tool-calls", help="Maximum deterministic tool calls per sector"
    ),
    max_turns: int = typer.Option(6, "--max-turns", help="Maximum LLM turns per sector"),
    max_cost_usd: float | None = typer.Option(
        None,
        "--max-cost-usd",
        help="Optional estimated LLM cost ceiling per sector; omitted means uncapped",
    ),
    max_candidates: int | None = typer.Option(
        None,
        "--max-candidates",
        help=(
            "Optional per-sector v2 execution bound; complete membership is preserved and "
            "overflow receives DEFERRED_BY_BOUND. Omitted means execute all loaded candidates."
        ),
    ),
    prewarm_cache: bool = typer.Option(
        False,
        "--prewarm-cache/--no-prewarm-cache",
        help=(
            "Prewarm local financial caches before v1 sectors; v2 rejects prewarming "
            "until refresh consumes its frozen execution authority"
        ),
    ),
    cache_refresh_run_id: str | None = typer.Option(
        None, "--cache-refresh-run-id", help="Resume/use a financial cache refresh run id"
    ),
    cache_years: int = typer.Option(
        10, "--cache-years", help="Companyfacts lookback years for benchmark cache prewarm"
    ),
    force_cache_refresh: bool = typer.Option(
        False, "--force-cache-refresh", help="Ignore completed cache refresh manifest steps"
    ),
    cache_max_tickers_per_sector: int | None = typer.Option(
        None,
        "--cache-max-tickers-per-sector",
        help="Maximum tickers to prewarm per benchmark sector",
    ),
    provider_preflight: bool = typer.Option(
        True,
        "--provider-preflight/--no-provider-preflight",
        help="Run a cheap LLM provider check before benchmark sector execution",
    ),
    cost_preflight_only: bool = typer.Option(
        False,
        "--cost-preflight-only",
        help=(
            "Freeze v2 candidates, persist the authoritative whole-run estimate, "
            "and exit before every provider or sector call."
        ),
    ),
    readiness_preflight_only: bool = typer.Option(
        False,
        "--readiness-preflight-only",
        help=(
            "Freeze accepted-census v2 candidates, inspect exact execution names "
            "with query-only local evidence, and exit without repair, providers, "
            "packet materialization, or sector execution."
        ),
    ),
    free_data_repair_only: bool = typer.Option(
        False,
        "--free-data-repair-only",
        help=(
            "Repair at most four accepted-census v2 sectors and three exact "
            "execution names per sector using SEC annual data and free Stooq "
            "prices, then rerun query-only readiness without a sector scan."
        ),
    ),
    diagnostic_reprice_model: str | None = typer.Option(
        None,
        "--diagnostic-reprice-model",
        help=(
            "Attach a price-only alternative-model comparison to a v2 cost preflight. "
            "Requires --cost-preflight-only and never changes execution authority."
        ),
    ),
    terminal_cap_search_max_attempts: int | None = typer.Option(
        None,
        "--terminal-cap-search-max-attempts",
        min=0,
        help="Maximum unknown-cap terminal searches to reserve; omitted reserves all frozen unknowns.",
    ),
    continue_on_provider_unavailable: bool = typer.Option(
        False,
        "--continue-on-provider-unavailable",
        help="Continue remaining sectors after a benchmark provider failure",
    ),
    resume_benchmark_run_id: str | None = typer.Option(
        None,
        "--resume-benchmark-run-id",
        help="Reuse successful sector results from a prior benchmark run",
    ),
    rerun_completed_sectors: bool = typer.Option(
        False, "--rerun-completed-sectors", help="Rerun completed sectors when resuming a benchmark"
    ),
    terminal_cap_search_preflight: str | None = typer.Option(
        None,
        "--terminal-cap-search-preflight",
        help=(
            "Persisted authorized whole-run cost-preflight JSON. Supplying a generic flag is "
            "not sufficient to enable paid terminal market-cap search."
        ),
    ),
    terminal_cap_search_ledger: str | None = typer.Option(
        None,
        "--terminal-cap-search-ledger",
        help="Attempt/evidence ledger path (defaults beside the preflight artifact).",
    ),
) -> None:
    """Run autonomous sector analysis across multiple sectors and aggregate results."""
    configure_logging()

    from app.autonomous.run_contract import AutonomousRunBudget
    from app.autonomous.sector_benchmark import (
        benchmark_compact_summary,
        persist_autonomous_sector_benchmark,
        run_autonomous_sector_benchmark,
    )

    max_candidates = _validate_max_candidates_option(max_candidates)
    if cost_preflight_only and str(pipeline_version or "").strip().lower() != "v2":
        raise typer.BadParameter("--cost-preflight-only requires explicit --pipeline-version v2")
    if readiness_preflight_only and str(pipeline_version or "").strip().lower() != "v2":
        raise typer.BadParameter(
            "--readiness-preflight-only requires explicit --pipeline-version v2"
        )
    if readiness_preflight_only and not as_of:
        raise typer.BadParameter("--readiness-preflight-only requires --as-of")
    if readiness_preflight_only and canonical_market_cap_focus(market_cap_focus) != (
        "large_and_mega"
    ):
        raise typer.BadParameter(
            "--readiness-preflight-only requires --market-cap-focus large_and_mega"
        )
    if readiness_preflight_only and cost_preflight_only:
        raise typer.BadParameter(
            "--readiness-preflight-only cannot be combined with --cost-preflight-only"
        )
    if readiness_preflight_only and prewarm_cache:
        raise typer.BadParameter(
            "--readiness-preflight-only cannot be combined with --prewarm-cache"
        )
    if readiness_preflight_only and resume_benchmark_run_id:
        raise typer.BadParameter("--readiness-preflight-only cannot reuse benchmark sector results")
    if readiness_preflight_only and rerun_completed_sectors:
        raise typer.BadParameter("--readiness-preflight-only cannot rerun benchmark sector results")
    if readiness_preflight_only and terminal_cap_search_preflight:
        raise typer.BadParameter(
            "--readiness-preflight-only cannot receive terminal search authority"
        )
    if readiness_preflight_only and terminal_cap_search_max_attempts is not None:
        raise typer.BadParameter(
            "--readiness-preflight-only cannot reserve terminal search attempts"
        )
    if readiness_preflight_only and terminal_cap_search_ledger:
        raise typer.BadParameter(
            "--readiness-preflight-only cannot receive a terminal search ledger"
        )
    if free_data_repair_only and str(pipeline_version or "").strip().lower() != "v2":
        raise typer.BadParameter("--free-data-repair-only requires explicit --pipeline-version v2")
    if free_data_repair_only and not as_of:
        raise typer.BadParameter("--free-data-repair-only requires --as-of")
    if free_data_repair_only and canonical_market_cap_focus(market_cap_focus) != ("large_and_mega"):
        raise typer.BadParameter(
            "--free-data-repair-only requires --market-cap-focus large_and_mega"
        )
    if free_data_repair_only and (cost_preflight_only or readiness_preflight_only):
        raise typer.BadParameter(
            "--free-data-repair-only cannot be combined with another preflight-only mode"
        )
    if free_data_repair_only and (
        max_candidates is None or max_candidates < 1 or max_candidates > 3
    ):
        raise typer.BadParameter(
            "--free-data-repair-only requires --max-candidates between 1 and 3"
        )
    if free_data_repair_only and prewarm_cache:
        raise typer.BadParameter("--free-data-repair-only cannot be combined with --prewarm-cache")
    if free_data_repair_only and resume_benchmark_run_id:
        raise typer.BadParameter("--free-data-repair-only cannot reuse benchmark sector results")
    if free_data_repair_only and rerun_completed_sectors:
        raise typer.BadParameter("--free-data-repair-only cannot rerun benchmark sector results")
    if free_data_repair_only and terminal_cap_search_preflight:
        raise typer.BadParameter("--free-data-repair-only cannot receive terminal search authority")
    if free_data_repair_only and terminal_cap_search_max_attempts is not None:
        raise typer.BadParameter("--free-data-repair-only cannot reserve terminal search attempts")
    if free_data_repair_only and terminal_cap_search_ledger:
        raise typer.BadParameter("--free-data-repair-only cannot receive a terminal search ledger")
    normalized_reprice_model = str(diagnostic_reprice_model or "").strip().lower() or None
    if normalized_reprice_model is not None and not cost_preflight_only:
        raise typer.BadParameter("--diagnostic-reprice-model requires --cost-preflight-only")
    if normalized_reprice_model is not None:
        from app.autonomous.all_sector_cost_preflight import (
            DIAGNOSTIC_REPRICE_MODELS,
        )

        if normalized_reprice_model not in DIAGNOSTIC_REPRICE_MODELS:
            raise typer.BadParameter(
                "--diagnostic-reprice-model must be one of " + ", ".join(DIAGNOSTIC_REPRICE_MODELS)
            )
    run_budget = AutonomousRunBudget(
        max_tool_calls=max(0, int(max_tool_calls)),
        max_turns=max(1, int(max_turns)),
        max_cost_usd=max_cost_usd,
        timebox_seconds=None,
        max_candidates=max_candidates,
    )
    sector_values = [item.strip() for item in sectors.split(",") if item.strip()]
    terminal_cap_search = None
    if terminal_cap_search_preflight:
        from app.autonomous.terminal_cap_search import (
            build_authorized_terminal_cap_search,
            whole_run_preflight_request_fingerprint,
        )
        from app.config import get_config
        from app.llm.providers.openai_provider import OpenAIProvider

        if str(pipeline_version or "").strip().lower() != "v2":
            raise typer.BadParameter(
                "--terminal-cap-search-preflight requires explicit --pipeline-version v2"
            )

        preflight_path = Path(terminal_cap_search_preflight)
        if not preflight_path.is_file():
            raise typer.BadParameter(
                "--terminal-cap-search-preflight must name a persisted JSON artifact"
            )
        try:
            preflight_payload = json.loads(preflight_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(
                "--terminal-cap-search-preflight is not readable JSON"
            ) from exc
        if not isinstance(preflight_payload, dict):
            raise typer.BadParameter("--terminal-cap-search-preflight must contain a JSON object")
        ledger_path = (
            Path(terminal_cap_search_ledger)
            if terminal_cap_search_ledger
            else preflight_path.with_name(f"{preflight_path.stem}.terminal_cap_search_ledger.json")
        )
        try:
            expected_request_fingerprint = whole_run_preflight_request_fingerprint(
                sectors=sector_values,
                objective=objective,
                as_of_date=as_of or date.today().isoformat(),
                market_cap_focus=market_cap_focus,
                pipeline_version="v2",
                budget=run_budget.to_dict(),
                max_candidates=max_candidates,
            )
            terminal_cap_search = build_authorized_terminal_cap_search(
                preflight=preflight_payload,
                provider=OpenAIProvider(get_config()),
                ledger_path=ledger_path,
                preflight_artifact_path=preflight_path,
                expected_request_fingerprint=expected_request_fingerprint,
            )
        except (ValueError, RuntimeError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    artifact = run_autonomous_sector_benchmark(
        sectors=sector_values,
        objective=objective,
        as_of_date=as_of,
        market_cap_focus=market_cap_focus,
        pipeline_version=pipeline_version,
        budget=run_budget,
        max_candidates=max_candidates,
        prewarm_cache=bool(prewarm_cache),
        cache_refresh_run_id=cache_refresh_run_id,
        cache_years=max(1, int(cache_years)),
        force_cache_refresh=bool(force_cache_refresh),
        cache_max_tickers_per_sector=cache_max_tickers_per_sector,
        provider_preflight=bool(provider_preflight),
        continue_on_provider_unavailable=bool(continue_on_provider_unavailable),
        resume_benchmark_run_id=resume_benchmark_run_id,
        rerun_completed_sectors=bool(rerun_completed_sectors),
        terminal_cap_search=terminal_cap_search,
        terminal_cap_search_max_attempts=terminal_cap_search_max_attempts,
        terminal_cap_search_ledger_path=terminal_cap_search_ledger,
        cost_preflight_only=bool(cost_preflight_only),
        diagnostic_reprice_model=normalized_reprice_model,
        readiness_preflight_only=bool(readiness_preflight_only),
        free_data_repair_only=bool(free_data_repair_only),
    )
    paths = persist_autonomous_sector_benchmark(artifact)
    summary = benchmark_compact_summary(artifact)
    summary["benchmark_path"] = str(paths.summary_json)
    summary["report_path"] = str(paths.report_md)
    cost_preflight_path = getattr(paths, "cost_preflight_json", None)
    summary["cost_preflight_path"] = (
        str(cost_preflight_path) if cost_preflight_path is not None else None
    )
    readiness_preflight_path = getattr(paths, "readiness_preflight_json", None)
    summary["readiness_preflight_path"] = (
        str(readiness_preflight_path) if readiness_preflight_path is not None else None
    )
    free_data_repair_path = getattr(paths, "free_data_repair_json", None)
    summary["free_data_repair_path"] = (
        str(free_data_repair_path) if free_data_repair_path is not None else None
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("alpha-scan")
def alpha_scan_cmd(
    sector: str = typer.Option(..., "--sector", help="Sector name (e.g., enterprise_software)"),
    skip_quarterly: bool = typer.Option(
        False, "--skip-quarterly", help="Skip quarterly data ingestion"
    ),
    years_back: int = typer.Option(10, "--years-back", help="Years of history to fetch"),
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Use quantitative ranking only (no LLM)"
    ),
) -> None:
    """Run recursive fundamental analysis to find the most undervalued company in a sector."""
    configure_logging()

    from app.alpha.publication import prepare_alpha_publication, publish_alpha_report
    from app.alpha.sector_comparator import run_sector_comparison
    from app.autonomous.financial_integrity import (
        InvalidFinancialInputError,
        require_financial_integrity_scope,
        require_unchanged_financial_integrity_scope,
    )
    from app.autonomous.v1_financial_context import (
        build_canonical_v1_financial_context,
    )
    from app.ingest.facts_writer import ensure_all_facts
    from app.valuation.peer_context import _load_sector_tickers

    # Auto-detect: if no alpha-capable LLM is available, use deterministic mode
    if not deterministic:
        try:
            from app.alpha.llm_runtime import alpha_llm_available

            if not alpha_llm_available():
                typer.echo("LLM provider disabled — using deterministic mode")
                deterministic = True
        except Exception:
            deterministic = True

    tickers = _load_sector_tickers(sector)
    if not tickers:
        typer.echo(json.dumps({"error": f"No tickers found for sector: {sector}"}))
        raise typer.Exit(1)
    typer.echo(f"Alpha scan: {sector} ({len(tickers)} tickers)")

    if not skip_quarterly:
        typer.echo("Ingesting annual + quarterly data...")
        for t in tickers:
            try:
                ensure_all_facts(t, years_back=years_back)
            except Exception as exc:
                logger.warning("alpha-scan: ingestion failed for %s: %s", t, exc)

    run_as_of_date = date.today().isoformat()
    typer.echo("Assembling signal packets...")
    financial_context = build_canonical_v1_financial_context(
        tickers=tickers,
        as_of_date=run_as_of_date,
        db_path=get_config().db_path,
    )
    packets = financial_context.packets
    integrity_scope = financial_context.scope(context=f"alpha_scan:{sector}")
    try:
        integrity_result = require_financial_integrity_scope(integrity_scope)
    except InvalidFinancialInputError as exc:
        typer.echo(
            json.dumps(
                {
                    "sector": sector,
                    "financial_integrity_status": exc.status,
                    "financial_integrity": exc.result.to_dict(),
                    "report_path": None,
                    "report_md_path": None,
                    "financial_integrity_authorization_path": None,
                },
                indent=2,
            )
        )
        raise typer.Exit(1) from exc

    # Classify attrition: why non-viable tickers were dropped
    viable_tickers = []
    attrition = {"no_scorecard": [], "no_price": [], "no_valuation": [], "blocked": []}
    for t, p in packets.items():
        has_valuation = any(
            value is not None
            for value in (p.insurance_value, p.dcf_value, p.epv_value, p.graham_value, p.ncav_value)
        )
        if p.gate_verdict == "BLOCK":
            attrition["blocked"].append(t)
        elif p.current_price is None:
            attrition["no_price"].append(t)
        elif p.gate_verdict is None and not has_valuation:
            attrition["no_scorecard"].append(t)
        elif not has_valuation:
            attrition["no_valuation"].append(t)
        else:
            viable_tickers.append(t)
    # Tickers not in packets at all (ingestion failed completely)
    missing = [t for t in tickers if t not in packets]
    viable = len(viable_tickers)

    typer.echo(f"  {viable}/{len(tickers)} viable candidates")
    dropped = len(tickers) - viable
    if dropped > 0:
        parts = []
        if missing:
            parts.append(f"{len(missing)} no data")
        if attrition["no_scorecard"]:
            parts.append(f"{len(attrition['no_scorecard'])} no scorecard")
        if attrition["no_price"]:
            parts.append(f"{len(attrition['no_price'])} no price")
        if attrition["no_valuation"]:
            parts.append(f"{len(attrition['no_valuation'])} no valuation")
        if attrition["blocked"]:
            parts.append(f"{len(attrition['blocked'])} blocked")
        typer.echo(f"  {dropped} dropped: {', '.join(parts)}")

    typer.echo(f"Running {'deterministic' if deterministic else 'recursive LLM'} comparison...")
    report = run_sector_comparison(
        sector,
        packets,
        deterministic=deterministic,
        integrity_scope=integrity_scope,
    )

    if report.selection_basis == "llm_decision":
        if report.pre_investigation_winner and report.pre_investigation_winner != report.winner:
            typer.echo(
                f"\nLLM decision changed the leader: {report.pre_investigation_winner} -> {report.winner}"
            )
    elif report.selection_basis == "llm_no_winner" and report.pre_investigation_winner:
        typer.echo(
            f"\nNo final LLM winner selected. "
            f"Consensus leader {report.pre_investigation_winner} did not clear the final screen."
        )

    if report.candidate_investigations:
        typer.echo(
            f"Investigated {len(report.candidate_investigations)} shortlisted candidates "
            f"selected by the LLM."
        )

    # Revalidate the exact mutable packets immediately before product publication.
    try:
        integrity_result = require_unchanged_financial_integrity_scope(
            integrity_scope,
            expected_scope_fingerprint=integrity_result.scope_fingerprint,
        )
    except InvalidFinancialInputError as exc:
        typer.echo(
            json.dumps(
                {
                    "sector": sector,
                    "financial_integrity_status": exc.status,
                    "financial_integrity": exc.result.to_dict(),
                    "report_path": None,
                    "report_md_path": None,
                    "financial_integrity_authorization_path": None,
                },
                indent=2,
            )
        )
        raise typer.Exit(1) from exc

    # Freeze every supplemental PIT row used by Markdown, then reacquire and
    # compare it under the publication lock before exact bytes are authorized.
    cfg = get_config()
    out_dir = cfg.data_dir / "outputs" / "alpha"
    try:
        publication_context = prepare_alpha_publication(
            report=report,
            packets=packets,
            run_as_of_date=run_as_of_date,
            decision_scope=integrity_scope,
        )
        publication_paths = publish_alpha_report(
            context=publication_context,
            decision_scope=integrity_scope,
            output_dir=out_dir,
        )
    except InvalidFinancialInputError as exc:
        typer.echo(
            json.dumps(
                {
                    "sector": sector,
                    "financial_integrity_status": exc.status,
                    "financial_integrity": exc.result.to_dict(),
                    "report_path": None,
                    "report_md_path": None,
                    "financial_integrity_authorization_path": None,
                },
                indent=2,
            )
        )
        raise typer.Exit(1) from exc
    report_path = publication_paths.report_json
    md_path = publication_paths.report_markdown

    # Summary output
    attrition_summary = {k: len(v) for k, v in attrition.items() if v}
    if missing:
        attrition_summary["no_data"] = len(missing)

    winner_preview = (
        next(
            (p for p in report.investigation_previews if p.get("ticker") == report.winner),
            None,
        )
        if report.winner
        else None
    )
    runner_preview = (
        next(
            (p for p in report.investigation_previews if p.get("ticker") == report.runner_up),
            None,
        )
        if report.runner_up
        else None
    )

    summary = {
        "sector": report.sector,
        "financial_integrity_status": integrity_result.status,
        "financial_integrity_scope_fingerprint": integrity_result.scope_fingerprint,
        "financial_integrity_publication_scope_fingerprint": (
            publication_context.bound_scope.expected_scope_fingerprint
        ),
        "financial_integrity_authorization_path": str(publication_paths.authorization),
        "total_candidates": report.total_candidates,
        "viable_candidates": viable,
        "attrition": attrition_summary,
        "selection_basis": report.selection_basis,
        "pre_investigation_winner": report.pre_investigation_winner,
        "winner": report.winner,
        "winner_conviction": report.winner_conviction
        or (winner_preview.get("conviction_class") if winner_preview else None),
        "winner_conviction_score": winner_preview.get("conviction_score")
        if winner_preview
        else None,
        "winner_adjusted_mos": winner_preview.get("adjusted_mos") if winner_preview else None,
        "winner_hard_blockers": winner_preview.get("hard_blockers", 0) if winner_preview else 0,
        "winner_report_path": winner_preview.get("report_path") if winner_preview else None,
        "runner_up": report.runner_up,
        "runner_up_conviction": runner_preview.get("conviction_class") if runner_preview else None,
        "runner_up_conviction_score": runner_preview.get("conviction_score")
        if runner_preview
        else None,
        "key_risk": report.key_risk,
        "falsification_trigger": report.falsification_trigger,
        "time_horizon": report.time_horizon,
        "elimination_rounds": len(report.rounds),
        "report_path": str(report_path),
        "report_md_path": str(md_path),
        "investigation_previews": report.investigation_previews,
        "prior_ranking": report.prior_ranking,
        "investigation_plan": report.investigation_plan,
        "candidate_investigations": report.candidate_investigations,
        "hard_blocked_candidates": report.hard_blocked_candidates,
        "llm_decision_trace": report.llm_decision_trace,
        "tool_budget_summary": report.tool_budget_summary,
    }
    typer.echo(json.dumps(summary, indent=2))


def _echo_no_scorecard(upper: str, report: Any) -> None:
    """The no-data state of `analyze` / `deep-research`: say so, print no verdict."""
    # Nothing to analyze. Say so instead of rendering an analyst verdict: a
    # "STAY_AWAY / INSUFFICIENT" header reads as advice when it is missing
    # data, and the analyst artifacts it would persist carry the same false
    # verdict into the reader.
    typer.echo(f"No valuation yet for {upper}: nothing to analyze.")
    typer.echo(
        f"For a valuation from free SEC data, run: ivi value {upper}\n"
        "`ivi analyze` reads only valuations bound to an audited run artifact, and "
        "`ivi value` output is research-only, so analyze will not use it."
    )
    typer.echo(f"Status: NO_SCORECARD (as of {report.as_of_date})")
    why = [str(item) for item in (getattr(report, "warnings", None) or [])]
    if why:
        typer.echo("Warnings:")
        for item in why:
            typer.echo(f"  - {item}")


def _validated_as_of(value: str | None) -> str | None:
    """``--as-of`` as an ISO date, or a usage error (exit 2) naming the option."""

    if not value:
        return None
    try:
        return parse_yyyy_mm_dd(value.strip()).isoformat()
    except ValueError:
        raise typer.BadParameter(
            f"{value!r} is not a date; use YYYY-MM-DD, e.g. 2024-06-30.",
            param_hint="--as-of",
        ) from None


@app.command("analyze")
def analyze_cmd(
    ticker: str = typer.Argument(help="Stock ticker symbol"),
    years: int = typer.Option(5, "--years", "-y", help="Years of annual data to analyze"),
    quarters: int = typer.Option(
        0, "--quarters", "-q", help="Recent quarters to include (0 = annual only)"
    ),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
) -> None:
    """Run full fundamental analysis on a ticker (valuation, evidence, conviction)."""
    as_of_date = _validated_as_of(as_of)
    configure_logging()
    upper = ticker.upper()
    from app.research.deep_research import run_deep_research

    report = run_deep_research(upper, as_of_date=as_of_date, years=years, quarters=quarters)

    if str(report.status) == "NO_SCORECARD":
        _echo_no_scorecard(upper, report)
        raise typer.Exit(code=1)

    try:
        from app.analyst.materializer import materialize_analysis_outputs_from_research
        from app.analyst.report_renderer import render_summary

        materialized = materialize_analysis_outputs_from_research(
            report,
            years=years,
            quarters=quarters,
        )

        typer.echo(render_summary(materialized.report))
        try:
            from app.valuation.valuation_render import render_valuation_decision_block

            decision_block = render_valuation_decision_block(upper, report.as_of_date)
            if decision_block:
                typer.echo("")
                typer.echo(decision_block)
        except Exception:
            logger.exception("analyze: valuation decision block failed for %s", upper)
        typer.echo("")
        typer.echo(f"Analyst bundle: {materialized.paths.bundle_json}")
        typer.echo(f"Analyst report JSON: {materialized.paths.report_json}")
        typer.echo(f"Analyst report Markdown: {materialized.paths.report_markdown}")
    except Exception:
        logger.exception("analyze: analyst runtime render failed for %s", upper)
        try:
            from app.research.report_renderer import build_view, render_summary

            view = build_view(report)
            typer.echo(render_summary(view))
        except Exception:
            logger.exception("analyze: research fallback render failed for %s", upper)
            typer.echo(f"Status: {report.status}")
            if report.conviction_class:
                typer.echo(f"Conviction: {report.conviction_class} ({report.conviction_score}/100)")
        if report.artifact_path:
            typer.echo(f"Research JSON: {report.artifact_path}")
        if report.report_path:
            typer.echo(f"Research Markdown: {report.report_path}")

    # OK is success; NO_FILING / NO_HYPOTHESES etc. are failures.
    if str(report.status) != "OK":
        raise typer.Exit(code=1)


@app.command("value")
def value_cmd(
    ticker: str = typer.Argument(help="Stock ticker symbol"),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD (default: today)"),
    years: int = typer.Option(10, "--years", "-y", help="Years of annual facts to ingest"),
    as_json: bool = typer.Option(False, "--json", help="Print the summary as JSON"),
) -> None:
    """Deterministic valuation from free data: SEC facts + a free quote, no LLM or paid key.

    Exit code 0: values and a price; 2: values but no price; 1: failed.
    """
    as_of_date = _validated_as_of(as_of)
    configure_logging()
    from app.valuation.quick_value import render_value_summary, run_value

    try:
        summary = run_value(ticker, as_of_date=as_of_date, years_back=years)
        rendered = json.dumps(summary, indent=2) if as_json else render_value_summary(summary)
    except Exception as exc:  # noqa: BLE001 - one line for the user, no traceback
        from app.util.credential_hygiene import redact_credential_text

        logger.debug("value: unexpected failure for %s", ticker.upper(), exc_info=True)
        detail = " ".join(str(redact_credential_text(str(exc)) or "").split())
        typer.echo(f"ivi value {ticker.upper()} failed: {type(exc).__name__}: {detail}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(rendered)
    if summary["status"] == "OK":
        return
    raise typer.Exit(code=2 if summary["status"] == "PARTIAL_NO_PRICE" else 1)


@app.command("expand-universe")
def expand_universe_cmd(
    source: str = typer.Option(
        "seed",
        "--source",
        help="Ticker source: 'seed' (2500), 'russell3000' (staged IWV holdings), or 'sec' (full SEC registry ~10,000+)",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Max tickers to process (default: all)"),
    as_of: str | None = typer.Option(
        None, "--as-of", help="As-of date YYYY-MM-DD (default: today)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be processed without running"
    ),
) -> None:
    """Expand the universe by ingesting and scoring tickers.

    For each ticker that doesn't have a scorecard: fetches companyfacts
    from EDGAR, runs the full valuation pipeline, and produces a scorecard.
    Resumable — skips already-scored tickers.

    Use --source russell3000 for the staged Russell 3000 list, or --source sec
    to expand from the full SEC registry (~10,000 tickers).
    """
    configure_logging()

    import json as _json
    from app.config import get_config
    from app.db import init_db
    from app.universe.expand import expand_universe

    cfg = get_config()
    init_db(cfg)

    summary = expand_universe(
        source=source,
        limit=limit,
        as_of_date=as_of if as_of else None,
        dry_run=dry_run,
    )

    typer.echo(
        _json.dumps(
            {
                "seed_total": summary.seed_total,
                "already_scored": summary.already_scored,
                "attempted": summary.attempted,
                "succeeded": summary.succeeded,
                "failed": summary.failed,
                "elapsed_seconds": round(summary.elapsed_seconds, 1),
                "failed_count": len(summary.failed_tickers),
            },
            indent=2,
        )
    )

    if dry_run:
        typer.echo(
            f"\n{summary.seed_total - summary.already_scored} tickers would be processed (dry run)"
        )
    else:
        typer.echo(
            f"\nDone. {summary.succeeded} new scorecards. Run `ivi sector-classify` to classify them."
        )


@app.command("backfill-prices")
def backfill_prices_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be updated without writing"
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-fetch prices even for scorecards that have one"
    ),
    as_of: str | None = typer.Option(None, "--as-of", help="Price as-of date (default: today)"),
) -> None:
    """Fetch and patch missing prices in cached scorecards.

    Uses the configured price provider (Yahoo -> Stooq) to fill in
    current_price for scorecards that were built with prices disabled.
    """
    configure_logging()

    import json as _json
    from app.config import get_config
    from app.db import init_db
    from app.market.backfill_prices import backfill_prices

    cfg = get_config()
    init_db(cfg)

    summary = backfill_prices(
        dry_run=dry_run,
        force=force,
        as_of_date=as_of if as_of else None,
    )

    typer.echo(
        _json.dumps(
            {
                "total_scorecards": summary.total_scorecards,
                "missing_price": summary.missing_price,
                "fetched": summary.fetched,
                "updated": summary.updated,
                "failed": summary.failed,
                "skipped_existing": summary.skipped_existing,
                "failed_count": len(summary.failed_tickers),
            },
            indent=2,
        )
    )

    if dry_run:
        typer.echo("\n(dry run — no rows updated)")


@app.command("scan")
def scan_cmd(
    sector: str = typer.Argument(..., help="Sector key (e.g. semiconductors, enterprise_software)"),
    top: int = typer.Option(10, "--top", help="Number of top picks in the report"),
    cap_tier: str | None = typer.Option(
        None,
        "--cap-tier",
        help="Filter by market cap tier: micro,small,mid,large,mega (comma-separated)",
    ),
    cap_min: float | None = typer.Option(None, "--cap-min", help="Min market cap in millions USD"),
    cap_max: float | None = typer.Option(None, "--cap-max", help="Max market cap in millions USD"),
    budget_usd: float = typer.Option(25.0, "--budget-usd", help="Hard paid-provider cost cap"),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Override output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show pre-rank without API calls"),
) -> None:
    """Comparative sector scan: pre-rank, AI triage, deep review.

    Loads all classified tickers in SECTOR, ranks them deterministically,
    then uses Sonnet to compare candidates side-by-side and deep-review
    the finalists. Produces a ranked report with the top N picks.

    Market cap filtering: --cap-tier small,mid filters to $300M-$10B.
    Or use --cap-min/--cap-max for custom ranges (millions USD).
    """
    configure_logging()

    from pathlib import Path as _Path
    from app.config import get_config
    from app.sector.scan import load_sector_tickers, pre_rank_sector, ScanConfig, CAP_TIERS

    cfg = get_config()
    db_path = _Path(cfg.db_path)
    scan_cfg = ScanConfig()

    # Resolve cap filtering
    effective_cap_min = cap_min
    effective_cap_max = cap_max
    cap_filter_label: str | None = None
    if cap_tier and not cap_min and not cap_max:
        tiers = [t.strip().lower() for t in cap_tier.split(",")]
        mins = []
        maxs = []
        for t in tiers:
            if t not in CAP_TIERS:
                typer.echo(f"Unknown cap tier '{t}'. Available: {', '.join(CAP_TIERS.keys())}")
                raise typer.Exit(code=1)
            lo, hi = CAP_TIERS[t]
            mins.append(lo)
            maxs.append(hi)
        effective_cap_min = min(mins)
        effective_cap_max = max(maxs)
        # "small,mid" → "small/mid cap" for a natural report title
        cap_filter_label = f"{'/'.join(tiers)} cap"
        typer.echo(
            f"Cap filter: {cap_tier} (${effective_cap_min:,.0f}M - ${effective_cap_max:,.0f}M)"
        )
    elif cap_min is not None or cap_max is not None:
        lo_s = f"${cap_min:,.0f}M" if cap_min is not None else "any"
        hi_s = f"${cap_max:,.0f}M" if cap_max is not None else "any"
        cap_filter_label = f"{lo_s}–{hi_s}"

    # Stage 1: Load and pre-rank
    run_as_of_date = date.today().isoformat()
    tickers_data = load_sector_tickers(
        sector=sector,
        db_path=db_path,
        cap_min=effective_cap_min,
        cap_max=effective_cap_max,
        as_of_date=run_as_of_date,
        allow_live_market_data=False,
    )
    if not tickers_data:
        typer.echo(f"No tickers classified as '{sector}'. Run `ivi sector-classify` first.")
        from app.sector.catalog import list_scannable_sectors

        rows = list_scannable_sectors(db_path=db_path)
        if rows:
            typer.echo("Available sectors:")
            for r in rows:
                typer.echo(f"  {r['sector']}: {r['count']} tickers")
        raise typer.Exit(code=1)

    typer.echo(f"Sector: {sector} ({len(tickers_data)} tickers)")
    typer.echo("Scan family: normal")

    ticker_list = [t for t, _, _ in tickers_data]
    ranked = pre_rank_sector(
        tickers=ticker_list,
        as_of_date=run_as_of_date,
        db_path=db_path,
        limit=scan_cfg.pre_rank_limit,
    )
    typer.echo(f"Pre-ranked: {len(ranked)} (top by consensus margin of safety)")

    if dry_run:
        typer.echo("\nPre-rank results:")
        for i, entry in enumerate(ranked[:20], 1):
            typer.echo(f"  #{i}: {entry.ticker} (score: {entry.consensus_score:.2f})")
        if len(ranked) > 20:
            typer.echo(f"  ... and {len(ranked) - 20} more")
        survivors = min(scan_cfg.max_deep_reviews, len(ranked))
        typer.echo(f"\nEstimated cost: ~${0.30 + survivors * 0.80:.2f}")
        typer.echo("(dry run — no API calls)")
        return

    from app.llm.providers import get_anthropic_provider
    from app.sector.scan import run_scan

    provider = get_anthropic_provider()
    if provider is None or not provider.enabled():
        typer.echo("VOE_ANTHROPIC_API_KEY not set. Configure via .env or environment.")
        raise typer.Exit(code=1)

    import anthropic as _anthropic

    client = _anthropic.Anthropic(
        api_key=cfg.anthropic_api_key,
        base_url=cfg.anthropic_base_url,
        timeout=float(cfg.anthropic_request_timeout),
        max_retries=0,
    )

    from pathlib import Path as _Path

    summary = run_scan(
        sector=sector,
        top_n=top,
        budget_usd=budget_usd,
        output_dir=_Path(output_dir) if output_dir else None,
        client=client,
        config=scan_cfg,
        db_path=db_path,
        cap_min=effective_cap_min,
        cap_max=effective_cap_max,
        cap_filter_label=cap_filter_label,
    )

    typer.echo(f"\nScan complete. Total cost: ${summary['total_cost']:.4f}")
    typer.echo(f"Scan family: {summary.get('scan_family', 'normal')}")
    typer.echo(f"Report: {summary['report_path']}")


@app.command("discover")
def discover_cmd(
    limit: int | None = typer.Option(
        None, "--limit", help="Max tickers to process from the universe"
    ),
    budget_usd: float = typer.Option(
        25.0, "--budget-usd", help="Soft budget cap for cost tracking"
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Override output directory"),
) -> None:
    """Run the four-stage discover funnel against cached scorecards.

    Stage 1: deterministic scorecards (already cached)
    Stage 2: Haiku 4.5 triage classifier
    Stage 3: Sonnet 4.6 mid-depth researcher (on KEEPs)
    Stage 4: Sonnet 4.6 deep loop with tool use (on WATCH/BUY_CANDIDATE)

    Writes a markdown report to data/outputs/discover/{sweep_id}.md.
    """
    configure_logging()

    from pathlib import Path as _Path
    from anthropic import Anthropic
    from app.config import get_config
    from app.discover.persistence import get_sweep
    from app.discover.report import render_sweep_report
    from app.discover.sweep import load_authorized_discover_universe, run_sweep

    cfg = get_config()
    session_db = _Path(cfg.db_path).parent / "discover_sessions.db"

    # Select the newest row first, then authorize that exact stored source.
    # An ineligible newest row is terminal; do not resurrect older scorecards.
    universe, financial_packets = load_authorized_discover_universe(cfg.db_path)

    if not universe:
        typer.echo("No cached scorecards found. Run ensure_valuation first.")
        raise typer.Exit(code=1)

    typer.echo(f"Loaded {len(universe)} tickers from {cfg.db_path}")
    if limit is not None:
        typer.echo(f"Applying --limit {limit}")

    api_key = cfg.anthropic_api_key
    if not api_key:
        typer.echo("VOE_ANTHROPIC_API_KEY not set. Configure via .env or environment.")
        raise typer.Exit(code=1)
    client = Anthropic(api_key=api_key, base_url=cfg.anthropic_base_url, max_retries=0)

    sweep_id = run_sweep(
        db_path=session_db,
        universe=universe,
        limit=limit,
        budget_usd=budget_usd,
        client=client,
        financial_packets=financial_packets,
    )

    sweep = get_sweep(session_db, sweep_id)
    typer.echo(f"\nSweep {sweep_id} complete.")
    typer.echo(f"Total cost: ${sweep['total_cost_usd']:.4f}")

    md = render_sweep_report(session_db, sweep_id)
    out_dir = _Path(output_dir) if output_dir else _Path("data/outputs/discover")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{sweep_id}.md"
    report_path.write_text(md, encoding="utf-8")
    typer.echo(f"Report written to {report_path}")


@app.command("discover-resume")
def discover_resume_cmd(
    sweep_id: str = typer.Argument(..., help="Sweep ID to resume (e.g. 20260412-001359)"),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="Additional budget cap for this resume run"
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Override output directory"),
) -> None:
    """Resume an interrupted discover sweep from where it stopped.

    Skips tickers that already have results in the session DB.
    Produces the same report as a fresh sweep.
    """
    configure_logging()

    from pathlib import Path as _Path
    from anthropic import Anthropic
    from app.config import get_config
    from app.discover.persistence import get_sweep
    from app.discover.report import render_sweep_report
    from app.discover.sweep import resume_sweep

    cfg = get_config()
    session_db = _Path(cfg.db_path).parent / "discover_sessions.db"
    engine_db = _Path(cfg.db_path)

    api_key = cfg.anthropic_api_key
    if not api_key:
        typer.echo("VOE_ANTHROPIC_API_KEY not set. Configure via .env or environment.")
        raise typer.Exit(code=1)
    client = Anthropic(api_key=api_key, base_url=cfg.anthropic_base_url, max_retries=0)

    pre_cost = (get_sweep(session_db, sweep_id) or {}).get("total_cost_usd", 0.0)

    try:
        resume_sweep(
            db_path=session_db,
            engine_db_path=engine_db,
            sweep_id=sweep_id,
            client=client,
            budget_override=budget_usd,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    sweep = get_sweep(session_db, sweep_id)
    new_cost = sweep["total_cost_usd"] - pre_cost
    if new_cost < 0.0001:
        typer.echo(f"\nSweep {sweep_id}: nothing to resume — all tickers already processed.")
    else:
        typer.echo(f"\nSweep {sweep_id} resume complete (${new_cost:.4f} additional cost).")
    typer.echo(f"Total cost: ${sweep['total_cost_usd']:.4f}")

    md = render_sweep_report(session_db, sweep_id)
    out_dir = _Path(output_dir) if output_dir else _Path("data/outputs/discover")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{sweep_id}.md"
    report_path.write_text(md, encoding="utf-8")
    typer.echo(f"Report written to {report_path}")


@app.command("deep-research")
def deep_research_cmd(
    ticker: str = typer.Option(..., "--ticker", help="Stock ticker symbol"),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
) -> None:
    """Run deep research investigation for a single ticker (alias: use 'analyze' instead)."""
    as_of_date = _validated_as_of(as_of)
    configure_logging()
    from app.research.deep_research import run_deep_research

    report = run_deep_research(ticker.upper(), as_of_date=as_of_date)

    if str(report.status) == "NO_SCORECARD":
        # Same no-data answer as `ivi analyze`: no verdict header, no persisted artifacts.
        _echo_no_scorecard(ticker.upper(), report)
        raise typer.Exit(code=1)

    try:
        from app.research.report_renderer import build_view, render_summary

        view = build_view(report)
        typer.echo(render_summary(view))
    except Exception:
        # Fallback to basic output if renderer fails
        logger.exception("deep-research: renderer failed for %s", ticker.upper())
        typer.echo(f"Status: {report.status}")
        if report.thesis:
            typer.echo(f"Thesis status: {report.thesis.status}")
        typer.echo(f"Hypotheses generated: {report.hypotheses_generated}")
        typer.echo(
            f"Adjustments: {report.total_adjustments} ({report.fact_calibrated_count} fact-calibrated, {report.heuristic_count} heuristic)"
        )
        typer.echo(
            f"Unresolved: {len(report.researchable_items)} researchable, {len(report.not_researchable_items)} not researchable"
        )
        if report.conviction_class:
            typer.echo(f"Conviction: {report.conviction_class} ({report.conviction_score}/100)")

    # OK is success; NO_SCORECARD / NO_FILING / NO_HYPOTHESES etc. are failures.
    if str(report.status) != "OK":
        raise typer.Exit(code=1)


@app.command("calibrate-research")
def calibrate_research_cmd(
    horizon_days: int = typer.Option(90, "--horizon-days", help="Fixed holding period in days"),
    scan_date: str | None = typer.Option(
        None, "--scan-date", help="Eligibility cutoff YYYY-MM-DD (default: today)"
    ),
) -> None:
    """Evaluate deep-research theses against subsequent price reality."""
    configure_logging()
    init_db(get_config())
    from app.calibration.outcome_tracker import scan_research_outcomes

    effective_date = scan_date if scan_date else date.today().isoformat()
    result = scan_research_outcomes(effective_date, horizon_days=horizon_days)

    typer.echo(
        json.dumps(
            {
                "scan_date": result.scan_date,
                "horizon_days": result.horizon_days,
                "total_eligible": result.total_eligible,
                "evaluated": result.evaluated,
                "skipped_no_price": result.skipped_no_price,
                "outcomes": [
                    {
                        "ticker": o.ticker,
                        "source_as_of_date": o.source_as_of_date,
                        "target_exit_date": o.target_exit_date,
                        "exit_as_of_date": o.exit_as_of_date,
                        "thesis_verdict": o.thesis_verdict,
                        "verdict_outcome": o.verdict_outcome,
                        "price_change_pct": round(o.price_change_pct, 2),
                        "conviction_class": o.conviction_class,
                        "methods_evaluated": o.methods_evaluated,
                    }
                    for o in result.outcomes
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
