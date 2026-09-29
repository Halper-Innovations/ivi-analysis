from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.analyst.output_store import latest_analysis_report
from app.autonomous.financial_integrity import (
    InvalidFinancialInputError,
    require_financial_integrity_scope,
)
from app.autonomous.v1_financial_context import (
    bind_v1_financial_scope,
    build_canonical_v1_financial_context,
    financial_input_scenario,
)
from app.config import get_config
from app.db import get_db, utc_now_iso
from app.llm.providers import get_llm_provider
from app.llm.providers.retry_guard import llm_physical_attempt_guard
from app.llm.schemas import (
    SectorSynthesisPacket,
    sector_synthesis_schema_for_prompt,
    validate_sector_synthesis_packet,
)
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    provider_failed_attempt_capture,
    provider_usage_records,
    provider_usage_records_from_exception,
    provider_usage_request,
    record_provider_usage,
)
from app.logging import get_logger
from app.util.hashing import sha256_text
from app.valuation.lineage import latest_decision_eligible_valuation_rows


logger = get_logger(__name__)


def _load_peer_rankings(*, run_id: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.dossiers_dir / run_id / "peer_rankings.json"
    if not path.exists():
        raise ValueError(f"peer_rankings.json not found for run_id={run_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid peer rankings payload")
    return payload


def _load_dossiers(*, run_id: str, tickers: list[str], max_count: int = 8) -> list[dict[str, Any]]:
    cfg = get_config()
    out: list[dict[str, Any]] = []
    for ticker in tickers[:max_count]:
        path = cfg.dossiers_dir / run_id / ticker / "dossier.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _load_legacy_research_packets(
    *, tickers: list[str], run_id: str, as_of_date: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with get_db() as conn:
        for ticker in tickers:
            row = conn.execute(
                """
                SELECT packet_path
                FROM research_packets
                WHERE ticker = ? AND run_id = ? AND as_of_date <= ?
                ORDER BY as_of_date DESC, created_at DESC
                LIMIT 1
                """,
                (ticker, run_id, as_of_date),
            ).fetchone()
            if not row:
                row = conn.execute(
                    """
                    SELECT packet_path
                    FROM research_packets
                    WHERE ticker = ? AND as_of_date <= ?
                    ORDER BY as_of_date DESC, created_at DESC
                    LIMIT 1
                    """,
                    (ticker, as_of_date),
                ).fetchone()
            if not row:
                continue
            path = Path(row["packet_path"])
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                out[ticker] = payload
    return out


def _extract_string_list(items: list[Any], key: str) -> list[str]:
    values: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get(key) or "").strip()
        else:
            text = str(item).strip()
        if text:
            values.append(text)
    return values


def _normalize_analysis_report_payload(ticker: str, as_of_date: str) -> dict[str, Any] | None:
    report = latest_analysis_report(ticker, as_of_date=as_of_date)
    if report is None:
        return None
    return {
        "verdict": report.verdict,
        "confidence_label": report.confidence_label,
        "confidence_score": report.confidence_score,
        "thesis_summary": report.thesis_summary,
        "positives": [finding.claim for finding in report.positives[:4]],
        "risks": [finding.claim for finding in report.risks[:4]],
        "recent_event_impacts": [finding.claim for finding in report.recent_event_impacts[:4]],
        "open_questions": [question.question for question in report.open_questions[:4]],
        "falsifiers": [item.description for item in report.falsifiers[:3]],
        "warnings": report.warnings[:6],
    }


def _normalize_legacy_research_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "",
        "confidence_label": "",
        "confidence_score": None,
        "thesis_summary": "",
        "positives": _extract_string_list((packet.get("findings") or [])[:4], "summary"),
        "risks": _extract_string_list((packet.get("risks") or [])[:4], "summary"),
        "recent_event_impacts": _extract_string_list(
            (packet.get("catalysts") or [])[:4], "summary"
        ),
        "open_questions": _extract_string_list((packet.get("evidence_gaps") or [])[:4], "question"),
        "falsifiers": _extract_string_list(
            (packet.get("disconfirming_evidence") or [])[:3], "summary"
        ),
        "warnings": ["legacy_research_packet_fallback"],
    }


def _load_ticker_analysis(
    *, tickers: list[str], run_id: str, as_of_date: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    legacy_packets = _load_legacy_research_packets(
        tickers=tickers, run_id=run_id, as_of_date=as_of_date
    )
    for ticker in tickers:
        normalized = _normalize_analysis_report_payload(ticker, as_of_date)
        if normalized is not None:
            out[ticker] = normalized
            continue
        packet = legacy_packets.get(ticker)
        if packet is not None:
            out[ticker] = _normalize_legacy_research_packet(packet)
    return out


def _load_valuation_snapshots(*, tickers: list[str], as_of_date: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with get_db() as conn:
        for ticker in tickers:
            rows = latest_decision_eligible_valuation_rows(
                conn,
                ticker=ticker,
                as_of_date=as_of_date,
                exact_as_of_date=True,
            )
            if not rows:
                continue
            out[ticker] = {
                row["method"]: {
                    "outputs": json.loads(row["outputs_json"]),
                    "warnings": json.loads(row["warnings_json"]),
                }
                for row in rows
            }
    return out


def _load_synthesis_packets(
    *, tickers: list[str], run_id: str, as_of_date: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with get_db() as conn:
        for ticker in tickers:
            row = conn.execute(
                """
                SELECT packet_path
                FROM synthesis_packets
                WHERE ticker = ? AND run_id = ? AND as_of_date <= ?
                ORDER BY as_of_date DESC, created_at DESC
                LIMIT 1
                """,
                (ticker, run_id, as_of_date),
            ).fetchone()
            if not row:
                continue
            path = Path(row["packet_path"])
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                out[ticker] = payload
    return out


def _compact_dossier(dossier: dict[str, Any]) -> dict[str, Any]:
    time_series = dossier.get("time_series") or {}
    return {
        "ticker": dossier.get("ticker"),
        "as_of_date": dossier.get("as_of_date"),
        "standardized_rows": (time_series.get("standardized_rows") or [])[-10:],
        "derived_signals": time_series.get("derived_signals") or [],
    }


def _build_inputs(
    *,
    sector: str,
    as_of_date: str,
    run_id: str,
    peer_rankings: dict[str, Any],
    ticker_analysis: dict[str, dict[str, Any]],
    canonical_financial_packets: dict[str, dict[str, Any]],
    financial_integrity_binding: dict[str, Any],
) -> dict[str, Any]:
    top_tickers = (peer_rankings.get("future_whale_rank") or [])[:10]
    return {
        "sector": sector,
        "as_of_date": as_of_date,
        "run_id": run_id,
        "peer_tickers": peer_rankings.get("tickers") or [],
        "peer_rankings": {
            "future_whale_rank": peer_rankings.get("future_whale_rank") or [],
            "quality_rank": peer_rankings.get("quality_rank") or [],
            "valuation_rank": peer_rankings.get("valuation_rank") or [],
            "risk_rank": peer_rankings.get("risk_rank") or [],
            "whale_signature_rank": peer_rankings.get("whale_signature_rank") or [],
        },
        "canonical_financial_packets": {
            ticker: canonical_financial_packets[ticker]
            for ticker in top_tickers
            if ticker in canonical_financial_packets
        },
        "financial_integrity_binding": financial_integrity_binding,
        "ticker_analysis": ticker_analysis,
    }


def _build_prompt(inputs: dict[str, Any]) -> str:
    return (
        "You are the Sector Synthesis Agent.\n"
        "Output ONLY JSON matching the schema.\n"
        "No fabricated facts. Use only provided inputs.\n"
        "canonical_financial_packets is the sole authority for every price, share count, market cap, valuation, ratio, and other numeric financial claim.\n"
        "Do not infer financial values from ranking labels or ticker_analysis narrative text.\n"
        "Numeric claims must include citations or derived_from in claims[].\n"
        "Narratives should emphasize moat indicators, operating leverage, capital allocation, runway, and valuation interpretation.\n"
        "Use ticker_analysis as the analyst-facing summary for each company.\n"
        "When ticker_analysis.warnings includes legacy_research_packet_fallback, treat that ticker's qualitative layer as lower confidence.\n"
        "Fill the Layer 3 fields directly: sector_summary, valuation_interpretation, risk_frame, catalyst_frame, evidence_gaps, recommended_next_actions, confidence_notes.\n"
        "Provide a top pick, runner-ups, avoid list, falsifiers, and what-to-read-next EDGAR follow-ups.\n\n"
        f"INPUT_PAYLOAD:\n{json.dumps(inputs, sort_keys=True)}"
    )


def _fallback_packet(
    *,
    sector: str,
    as_of_date: str,
    run_id: str,
    peer_tickers: list[str],
    model: str,
    prompt_hash: str,
    input_hash: str,
) -> dict[str, Any]:
    top_pick = peer_tickers[0] if peer_tickers else "UNKNOWN"
    runner = peer_tickers[1:3] if len(peer_tickers) > 1 else []
    avoid = peer_tickers[-2:] if len(peer_tickers) > 2 else []
    narratives = []
    for ticker in peer_tickers:
        narratives.append(
            {
                "ticker": ticker,
                "moat_indicators": ["UNKNOWN"],
                "operating_leverage": "UNKNOWN (LLM provider disabled)",
                "capital_allocation": "UNKNOWN (LLM provider disabled)",
                "reinvestment_runway": "UNKNOWN (LLM provider disabled)",
                "citations": [],
                "derived_from": [f"peer_rankings.{ticker}"],
            }
        )
    return {
        "sector": sector,
        "as_of_date": as_of_date,
        "run_id": run_id,
        "sector_summary": "Fallback sector synthesis based on deterministic peer rankings only.",
        "valuation_interpretation": "Qualitative sector valuation framing is unavailable because the LLM provider is disabled.",
        "risk_frame": "Interpretive confidence is limited because sector synthesis ran in fallback mode.",
        "catalyst_frame": "Read the latest MD&A, segment notes, and cash flow disclosures for the highest-ranked peers.",
        "evidence_gaps": ["LLM provider disabled for sector synthesis"],
        "recommended_next_actions": [
            "Enable model provider and rerun sector synthesis for richer Layer 3 comparison."
        ],
        "confidence_notes": "Fallback output only. Deterministic rankings remain authoritative.",
        "peer_tickers": peer_tickers,
        "top_pick": top_pick,
        "runner_ups": runner,
        "avoid_list": avoid,
        "whale_checklist": [
            "Sustained revenue compounding (derived from long-horizon annual filing time-series).",
            "Operating leverage trend positive over multi-year horizon.",
            "FCF margin trajectory improving with controlled dilution.",
        ],
        "per_ticker_narrative": narratives,
        "falsifiers": ["Enable provider and compare with model-generated synthesis."],
        "what_to_read_next": [
            "Latest annual and quarterly MD&A, Segment Notes, and Cash Flow notes for top pick."
        ],
        "claims": [
            {
                "id": "sector_synth_provider_disabled",
                "text": "Sector synthesis generated in deterministic fallback mode.",
                "type": "non_numeric",
                "citations": [],
                "derived_from": ["config.llm_provider"],
            }
        ],
        "llm_meta": {
            "model": model,
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "cost_estimate_usd": 0.0,
            "created_at": utc_now_iso(),
        },
    }


def run_sector_synthesis(
    *,
    sector: str,
    as_of_date: str,
    run_id: str,
) -> dict[str, Any]:
    peer_rankings = _load_peer_rankings(run_id=run_id)
    peer_tickers = [str(t).upper() for t in (peer_rankings.get("tickers") or [])]
    financial_context = build_canonical_v1_financial_context(
        tickers=peer_tickers,
        as_of_date=as_of_date,
        db_path=get_config().db_path,
    )
    packet_integrity_scope = financial_context.scope(context=f"sector_synthesis:{sector}:{run_id}")
    initial_integrity_result = require_financial_integrity_scope(packet_integrity_scope)
    canonical_financial_packets = {
        ticker: asdict(packet) for ticker, packet in financial_context.packets.items()
    }
    financial_integrity_binding = {
        "status": initial_integrity_result.status,
        "scope_fingerprint": initial_integrity_result.scope_fingerprint,
        "run_as_of_date": initial_integrity_result.run_as_of_date,
        "ticker_snapshot_ids": initial_integrity_result.ticker_snapshot_ids,
    }
    ticker_analysis = _load_ticker_analysis(
        tickers=peer_tickers, run_id=run_id, as_of_date=as_of_date
    )
    provider = get_llm_provider()
    model = (
        get_config().openai_model
        if getattr(provider, "provider_name", "") == "openai"
        else "disabled"
    )

    inputs = _build_inputs(
        sector=sector,
        as_of_date=as_of_date,
        run_id=run_id,
        peer_rankings=peer_rankings,
        ticker_analysis=ticker_analysis,
        canonical_financial_packets=canonical_financial_packets,
        financial_integrity_binding=financial_integrity_binding,
    )
    schema = sector_synthesis_schema_for_prompt()
    schema_name = "sector_synthesis_packet_v1"
    max_output_tokens = (
        int(get_config().openai_max_output_tokens)
        if getattr(provider, "provider_name", "") == "openai"
        else int(get_config().anthropic_max_output_tokens)
    )
    provider_request = {
        "schema": schema,
        "schema_name": schema_name,
        "provider": getattr(provider, "provider_name", ""),
        "model": model,
        "max_output_tokens": max_output_tokens,
    }

    def current_financial_scenarios() -> tuple[dict[str, Any], ...]:
        financial_inputs = {
            key: value for key, value in inputs.items() if key != "financial_integrity_binding"
        }
        return tuple(
            financial_input_scenario(
                financial_context.packets[ticker],
                financial_inputs={
                    "sector_synthesis_inputs": financial_inputs,
                    "provider_request": provider_request,
                },
            )
            for ticker in peer_tickers
            if ticker in financial_context.packets
        )

    bound_financial_scope = bind_v1_financial_scope(
        context=f"sector_synthesis:{sector}:{run_id}",
        run_as_of_date=as_of_date,
        packets=tuple(
            financial_context.packets[ticker]
            for ticker in peer_tickers
            if ticker in financial_context.packets
        ),
        scenarios=current_financial_scenarios(),
    )
    bound_integrity_result = bound_financial_scope.require(scenarios=current_financial_scenarios())
    financial_integrity_binding = {
        "status": bound_integrity_result.status,
        "scope_fingerprint": bound_integrity_result.scope_fingerprint,
        "run_as_of_date": bound_integrity_result.run_as_of_date,
        "ticker_snapshot_ids": bound_integrity_result.ticker_snapshot_ids,
    }
    inputs["financial_integrity_binding"] = financial_integrity_binding
    prompt = _build_prompt(inputs)
    prompt_hash = sha256_text(prompt)
    input_hash = sha256_text(json.dumps(inputs, sort_keys=True))
    failed_attempts: list[dict[str, Any]] = []
    successful_attempts: list[dict[str, Any]] = []

    if not provider.enabled():
        bound_financial_scope.require(scenarios=current_financial_scenarios())
        payload = _fallback_packet(
            sector=sector,
            as_of_date=as_of_date,
            run_id=run_id,
            peer_tickers=peer_tickers,
            model=model,
            prompt_hash=prompt_hash,
            input_hash=input_hash,
        )
        packet: SectorSynthesisPacket = validate_sector_synthesis_packet(payload)
        usage = {"input_tokens": 0, "output_tokens": 0}
        cost = 0.0
    else:

        def require_exact_scope(_attempt: dict[str, Any]) -> None:
            bound_financial_scope.require(scenarios=current_financial_scenarios())

        try:
            require_exact_scope({})
            with provider_usage_request(
                provider=provider,
                prompt=prompt,
                schema=schema,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            ) as request_kwargs:
                try:
                    with (
                        provider_failed_attempt_capture(
                            provider=provider,
                            prompt=prompt,
                            schema_name=schema_name,
                            estimated_output_tokens=max_output_tokens,
                        ) as failed_attempts,
                        llm_physical_attempt_guard(require_exact_scope),
                    ):
                        result = provider.synthesize_json(
                            prompt=prompt,
                            schema=schema,
                            schema_name=schema_name,
                            **request_kwargs,
                        )
                except BaseException as exc:
                    successful_attempts = provider_usage_records_from_exception(
                        provider=provider,
                        error=exc,
                        prompt=prompt,
                        schema_name=schema_name,
                    )
                    for usage_record in successful_attempts:
                        record_provider_usage(usage_record)
                    attach_provider_usage_to_exception(
                        exc, [*failed_attempts, *successful_attempts]
                    )
                    try:
                        require_exact_scope({})
                    except InvalidFinancialInputError as integrity_exc:
                        attach_provider_usage_to_exception(
                            integrity_exc,
                            [*failed_attempts, *successful_attempts],
                        )
                        raise integrity_exc from exc
                    raise
                successful_attempts = provider_usage_records(
                    provider=provider,
                    result=result,
                    prompt=prompt,
                    schema_name=schema_name,
                )
                for usage_record in successful_attempts:
                    record_provider_usage(usage_record)

            raw_payload = json.loads(result.json_text)
            if not isinstance(raw_payload, dict):
                raise RuntimeError("sector synthesis output was not a JSON object")
            raw_payload["sector"] = sector
            raw_payload["as_of_date"] = as_of_date
            raw_payload["run_id"] = run_id
            raw_payload.setdefault("llm_meta", {})
            cost = round(
                sum(
                    float(record.get("cost_estimate_usd") or 0.0)
                    for record in [*failed_attempts, *successful_attempts]
                ),
                6,
            )
            raw_payload["llm_meta"]["model"] = result.model
            raw_payload["llm_meta"]["prompt_hash"] = prompt_hash
            raw_payload["llm_meta"]["input_hash"] = input_hash
            raw_payload["llm_meta"]["cost_estimate_usd"] = cost
            raw_payload["llm_meta"]["created_at"] = utc_now_iso()
            packet = validate_sector_synthesis_packet(raw_payload)
            usage = {
                "input_tokens": result.usage_input_tokens,
                "cached_input_tokens": getattr(result, "usage_cached_input_tokens", None),
                "output_tokens": result.usage_output_tokens,
            }
        except BaseException as exc:
            attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
            raise

    cfg = get_config()
    try:
        # The exact mutable packet inputs must still be valid at publication.
        integrity_result = bound_financial_scope.require(scenarios=current_financial_scenarios())
        out_dir = cfg.sectors_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sector_synthesis.json"
        payload = packet.model_dump(mode="json")
        payload["financial_integrity"] = {
            **financial_integrity_binding,
            "input_hash": input_hash,
            "prompt_hash": prompt_hash,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except BaseException as exc:
        if provider.enabled():
            attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
        raise
    meta = {
        "run_id": run_id,
        "sector": sector,
        "as_of_date": as_of_date,
        "path": str(out_path),
        "prompt_hash": prompt_hash,
        "input_hash": input_hash,
        "usage": usage,
        "provider_usage": [*failed_attempts, *successful_attempts],
        "cost_estimate_usd": cost,
        "financial_integrity_status": integrity_result.status,
        "financial_integrity_scope_fingerprint": integrity_result.scope_fingerprint,
    }
    logger.info(
        "sector_synthesis_completed",
        extra={"stage_name": "sector_synthesis", "stage_run_id": run_id},
    )
    return meta
