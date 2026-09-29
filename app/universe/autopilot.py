from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.db import utc_now_iso
from app.logging import get_logger


STAGE_SCOUT = "SCOUT"
STAGE_DEPTH_QUEUE = "DEPTH_QUEUE"
STAGE_DEPTH_BATCH = "DEPTH_BATCH"
STAGE_ROLLUP = "ROLLUP"
STAGE_FILING_DIFF = "FILING_DIFF"
STAGE_PATTERN_SCAN = "PATTERN_SCAN"
STAGE_VARIANT_SYNTHESIS = "VARIANT_SYNTHESIS"
STAGE_DOSSIER_PACK = "DOSSIER_PACK"
STAGE_MEMO_PACK = "MEMO_PACK"
_STAGE_ORDER = [
    STAGE_SCOUT,
    STAGE_DEPTH_QUEUE,
    STAGE_DEPTH_BATCH,
    STAGE_ROLLUP,
    STAGE_FILING_DIFF,
    STAGE_PATTERN_SCAN,
    STAGE_VARIANT_SYNTHESIS,
    STAGE_DOSSIER_PACK,
    STAGE_MEMO_PACK,
]

STAGE_NOT_STARTED = "NOT_STARTED"
STAGE_RUNNING = "RUNNING"
STAGE_DONE = "DONE"
STAGE_FAILED = "FAILED"
STAGE_CANCELLED = "CANCELLED"
STAGE_PARTIAL = "PARTIAL"

AUTOPILOT_PLANNED = "PLANNED"
AUTOPILOT_RUNNING = "RUNNING"
AUTOPILOT_DONE = "DONE"
AUTOPILOT_FAILED = "FAILED"
AUTOPILOT_CANCELLED = "CANCELLED"
AUTOPILOT_PARTIAL = "PARTIAL"

STOP_COMPLETED = "COMPLETED"
STOP_MAX_STAGES_REACHED = "MAX_STAGES_REACHED"
STOP_CANCEL_REQUESTED = "CANCEL_REQUESTED"
STOP_STAGE_FAILED = "STAGE_FAILED"
STOP_DRY_RUN = "DRY_RUN"

_DEPTH_MODES = {"fundamentals", "full", "alpha-only"}

logger = get_logger(__name__)


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _autopilot_dir(universe_run_id: str) -> Path:
    cfg = get_config()
    return cfg.outputs_dir / "universe" / universe_run_id / "autopilot"


def _autopilot_paths(universe_run_id: str) -> dict[str, Path]:
    root = _autopilot_dir(universe_run_id)
    return {
        "root": root,
        "state_path": root / "autopilot_state.json",
        "summary_path": root / "autopilot_summary.json",
    }


def _batch_run_id(universe_run_id: str) -> str:
    return f"{universe_run_id}_depth_batch"


def _stage_expected_paths(universe_run_id: str, batch_run_id: str) -> dict[str, dict[str, str]]:
    cfg = get_config()
    universe_root = cfg.outputs_dir / "universe" / universe_run_id
    sectors_root = cfg.sectors_dir / universe_run_id
    batch_root = universe_root / "depth_batches" / batch_run_id
    filing_diff_root = universe_root / "filing_diffs"
    pattern_root = universe_root / "pattern_scan"
    variant_root = universe_root / "variant_perceptions"
    dossier_root = batch_root / "dossier_pack"
    memo_root = batch_root / "memo_pack"
    autopilot_root = universe_root / "autopilot"
    return {
        STAGE_SCOUT: {
            "universe_summary_path": str(sectors_root / "universe_summary.json"),
            "universe_scoreboard_path": str(sectors_root / "universe_scoreboard.json"),
            "universe_shortlist_path": str(sectors_root / "universe_shortlist.json"),
            "universe_coverage_path": str(sectors_root / "universe_coverage.json"),
            "universe_scout_calibration_path": str(sectors_root / "universe_scout_calibration.json"),
            "depth_queue_path": str(universe_root / "depth_queue.json"),
        },
        STAGE_DEPTH_QUEUE: {
            "depth_queue_path": str(universe_root / "depth_queue.json"),
        },
        STAGE_DEPTH_BATCH: {
            "batch_state_path": str(batch_root / "batch_state.json"),
            "batch_summary_path": str(batch_root / "batch_summary.json"),
            "batch_log_path": str(batch_root / "batch_log.jsonl"),
        },
        STAGE_ROLLUP: {
            "global_shortlist_json_path": str(batch_root / "global_shortlist.json"),
            "global_shortlist_md_path": str(batch_root / "global_shortlist.md"),
            "global_rollup_json_path": str(batch_root / "global_rollup.json"),
        },
        STAGE_FILING_DIFF: {
            "filing_diff_summary_path": str(filing_diff_root / "filing_diff_summary.json"),
            "filing_diff_dir_path": str(filing_diff_root),
        },
        STAGE_PATTERN_SCAN: {
            "pattern_scan_report_path": str(pattern_root / "pattern_scan_report.json"),
            "pattern_scan_summary_path": str(pattern_root / "pattern_scan_summary.json"),
        },
        STAGE_VARIANT_SYNTHESIS: {
            "variant_synthesis_summary_path": str(variant_root / "variant_synthesis_summary.json"),
            "variant_perceptions_dir_path": str(variant_root),
        },
        STAGE_DOSSIER_PACK: {
            "dossier_pack_dir": str(dossier_root),
            "manifest_path": str(dossier_root / "dossier_pack_manifest.json"),
            "watchlist_csv_path": str(dossier_root / "watchlist.csv"),
            "watchlist_json_path": str(dossier_root / "watchlist.json"),
        },
        STAGE_MEMO_PACK: {
            "memo_pack_dir": str(memo_root),
            "manifest_path": str(memo_root / "memo_pack_manifest.json"),
            "watchlist_state_path": str(autopilot_root / "watchlist_state.json"),
        },
    }


def _init_stage_state(expected_paths: dict[str, str]) -> dict[str, Any]:
    return {
        "status": STAGE_NOT_STARTED,
        "started_at": None,
        "updated_at": utc_now_iso(),
        "artifact_paths": dict(expected_paths),
        "result": {},
        "error": None,
    }


def _default_state(
    *,
    universe_run_id: str,
    as_of_date: str,
    batch_run_id: str,
    depth: str,
    scout_params: dict[str, Any],
    depth_batch_params: dict[str, Any],
    rollup_params: dict[str, Any],
    filing_diff_params: dict[str, Any],
    pattern_scan_params: dict[str, Any],
    variant_synthesis_params: dict[str, Any],
    dossier_pack_params: dict[str, Any],
    memo_pack_params: dict[str, Any],
) -> dict[str, Any]:
    expected_paths = _stage_expected_paths(universe_run_id, batch_run_id)
    return {
        "universe_run_id": universe_run_id,
        "as_of_date": as_of_date,
        "batch_run_id": batch_run_id,
        "depth": depth,
        "status": AUTOPILOT_RUNNING,
        "stop_reason_code": "",
        "stop_summary": "",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "pid": os.getpid(),
        "stages": {
            stage: _init_stage_state(expected_paths[stage])
            for stage in _STAGE_ORDER
        },
        "params": {
            "scout_params": scout_params,
            "depth_batch_params": depth_batch_params,
            "rollup_params": rollup_params,
            "filing_diff_params": filing_diff_params,
            "pattern_scan_params": pattern_scan_params,
            "variant_synthesis_params": variant_synthesis_params,
            "dossier_pack_params": dossier_pack_params,
            "memo_pack_params": memo_pack_params,
        },
    }


def _state_is_cancelled(state: dict[str, Any]) -> bool:
    return str(state.get("status") or "").upper() == AUTOPILOT_CANCELLED


def _artifact_exists(path_value: str | None) -> bool:
    if not path_value:
        return False
    return Path(str(path_value)).exists()


def _stage_artifacts_exist(stage_name: str, stage_state: dict[str, Any]) -> bool:
    artifacts = stage_state.get("artifact_paths") if isinstance(stage_state.get("artifact_paths"), dict) else {}
    if stage_name == STAGE_SCOUT:
        return _artifact_exists(str(artifacts.get("universe_summary_path") or "")) and _artifact_exists(
            str(artifacts.get("depth_queue_path") or "")
        )
    if stage_name == STAGE_DEPTH_QUEUE:
        return _artifact_exists(str(artifacts.get("depth_queue_path") or ""))
    if stage_name == STAGE_DEPTH_BATCH:
        return _artifact_exists(str(artifacts.get("batch_state_path") or "")) and _artifact_exists(
            str(artifacts.get("batch_summary_path") or "")
        )
    if stage_name == STAGE_ROLLUP:
        return _artifact_exists(str(artifacts.get("global_shortlist_json_path") or "")) and _artifact_exists(
            str(artifacts.get("global_rollup_json_path") or "")
        )
    if stage_name == STAGE_FILING_DIFF:
        return _artifact_exists(str(artifacts.get("filing_diff_summary_path") or "")) and _artifact_exists(
            str(artifacts.get("filing_diff_dir_path") or "")
        )
    if stage_name == STAGE_PATTERN_SCAN:
        return _artifact_exists(str(artifacts.get("pattern_scan_report_path") or "")) and _artifact_exists(
            str(artifacts.get("pattern_scan_summary_path") or "")
        )
    if stage_name == STAGE_VARIANT_SYNTHESIS:
        return _artifact_exists(str(artifacts.get("variant_synthesis_summary_path") or "")) and _artifact_exists(
            str(artifacts.get("variant_perceptions_dir_path") or "")
        )
    if stage_name == STAGE_DOSSIER_PACK:
        return _artifact_exists(str(artifacts.get("manifest_path") or "")) and _artifact_exists(
            str(artifacts.get("watchlist_csv_path") or "")
        )
    if stage_name == STAGE_MEMO_PACK:
        return _artifact_exists(str(artifacts.get("manifest_path") or "")) and _artifact_exists(
            str(artifacts.get("watchlist_state_path") or "")
        )
    return False


def _normalize_depth(value: Any) -> str:
    depth = str(value or "fundamentals").strip().lower()
    if depth not in _DEPTH_MODES:
        raise ValueError(f"Unsupported depth={value}; expected one of: alpha-only, full, fundamentals")
    return depth


def _stage_enabled_for_depth(stage_name: str, depth: str) -> bool:
    depth_norm = _normalize_depth(depth)
    if depth_norm == "full":
        return True
    if depth_norm == "alpha-only":
        return stage_name not in {STAGE_DOSSIER_PACK, STAGE_MEMO_PACK}
    return stage_name not in {STAGE_FILING_DIFF, STAGE_PATTERN_SCAN, STAGE_VARIANT_SYNTHESIS}


def _stage_result_is_skipped(stage_state: dict[str, Any]) -> bool:
    result = stage_state.get("result") if isinstance(stage_state.get("result"), dict) else {}
    return bool(result.get("skipped_by_depth")) or str(result.get("status") or "").upper() == "SKIPPED"


def _batch_state_payload(*, universe_run_id: str, batch_run_id: str) -> dict[str, Any]:
    batch_state_path = Path(
        str(_stage_expected_paths(universe_run_id, batch_run_id)[STAGE_DEPTH_BATCH].get("batch_state_path") or "")
    )
    return _safe_json(batch_state_path)


def _completed_depth_runs(batch_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (batch_state.get("completed_runs") or []) if isinstance(row, dict)]
    return [
        row
        for row in rows
        if str(row.get("status") or "").upper() in {"DONE", "COMPLETED"}
    ]


def _completed_depth_tickers(batch_state: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in _completed_depth_runs(batch_state):
        for ticker in row.get("tickers") or []:
            token = str(ticker or "").strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def _ticker_run_id_map(batch_state: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in _completed_depth_runs(batch_state):
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        for ticker in row.get("tickers") or []:
            token = str(ticker or "").strip().upper()
            if token and token not in out:
                out[token] = run_id
    return out


def _merge_artifact_paths(stage_state: dict[str, Any]) -> None:
    result = stage_state.get("result") if isinstance(stage_state.get("result"), dict) else {}
    artifacts = stage_state.get("artifact_paths") if isinstance(stage_state.get("artifact_paths"), dict) else {}
    artifacts.update(
        {
            key: str(value)
            for key, value in result.items()
            if isinstance(key, str)
            and key.endswith("_path")
            and isinstance(value, str)
            and value.strip()
        }
    )
    stage_state["artifact_paths"] = artifacts


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def _load_rollup_rows(*, universe_run_id: str, batch_run_id: str) -> list[dict[str, Any]]:
    shortlist_path = Path(
        str(_stage_expected_paths(universe_run_id, batch_run_id)[STAGE_ROLLUP].get("global_shortlist_json_path") or "")
    )
    payload = _safe_json(shortlist_path)
    return [row for row in (payload.get("rows") or []) if isinstance(row, dict)]


def _intangible_payloads_by_ticker(*, universe_run_id: str, batch_run_id: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _load_rollup_rows(universe_run_id=universe_run_id, batch_run_id=batch_run_id):
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker:
            out[ticker] = row
    return out


def _depth_batch_progress(batch_state_path: str | None) -> dict[str, Any]:
    if not batch_state_path:
        return {}
    payload = _safe_json(Path(str(batch_state_path)))
    if not payload:
        return {}
    return {
        "cursor_next_idx": int(payload.get("cursor_next_idx") or 0),
        "planned_count": len([row for row in (payload.get("planned_runs") or []) if isinstance(row, dict)]),
        "completed_count": len([row for row in (payload.get("completed_runs") or []) if isinstance(row, dict)]),
        "run_status": str(payload.get("status") or ""),
        "stop_reason_code": str(payload.get("stop_reason_code") or ""),
    }


def _depth_batch_complete(batch_state_path: str | None) -> bool:
    progress = _depth_batch_progress(batch_state_path)
    return str(progress.get("run_status") or "").upper() == "DONE"


def _should_rerun_rollup(
    stage_state: dict[str, Any],
    *,
    force: bool,
    input_paths: list[Path] | None = None,
) -> bool:
    artifacts = stage_state.get("artifact_paths") if isinstance(stage_state.get("artifact_paths"), dict) else {}
    output_paths = [
        Path(str(artifacts.get("global_shortlist_json_path") or "")),
        Path(str(artifacts.get("global_shortlist_md_path") or "")),
        Path(str(artifacts.get("global_rollup_json_path") or "")),
    ]
    in_paths = [path for path in (input_paths or []) if isinstance(path, Path)]
    if force:
        return True
    return _newer_than(in_paths, output_paths)


def _should_rerun_dossier(
    stage_state: dict[str, Any],
    *,
    force: bool,
    input_paths: list[Path] | None = None,
) -> bool:
    artifacts = stage_state.get("artifact_paths") if isinstance(stage_state.get("artifact_paths"), dict) else {}
    output_paths = [
        Path(str(artifacts.get("manifest_path") or "")),
        Path(str(artifacts.get("watchlist_csv_path") or "")),
        Path(str(artifacts.get("watchlist_json_path") or "")),
    ]
    in_paths = [path for path in (input_paths or []) if isinstance(path, Path)]
    if force:
        return True
    return _newer_than(in_paths, output_paths)


def _should_rerun_memo(
    stage_state: dict[str, Any],
    *,
    force: bool,
    input_paths: list[Path] | None = None,
) -> bool:
    artifacts = stage_state.get("artifact_paths") if isinstance(stage_state.get("artifact_paths"), dict) else {}
    output_paths = [
        Path(str(artifacts.get("manifest_path") or "")),
        Path(str(artifacts.get("watchlist_state_path") or "")),
    ]
    in_paths = [path for path in (input_paths or []) if isinstance(path, Path)]
    if force:
        return True
    return _newer_than(in_paths, output_paths)


def _newer_than(inputs: list[Path], outputs: list[Path]) -> bool:
    existing_inputs = [path for path in inputs if path.exists()]
    existing_outputs = [path for path in outputs if path.exists()]
    if not existing_outputs:
        return True
    if not existing_inputs:
        return False
    newest_input = max(path.stat().st_mtime for path in existing_inputs)
    oldest_output = min(path.stat().st_mtime for path in existing_outputs)
    return newest_input > oldest_output


def _write_state(universe_run_id: str, state: dict[str, Any]) -> None:
    paths = _autopilot_paths(universe_run_id)
    state["updated_at"] = utc_now_iso()
    state["pid"] = os.getpid()
    _json_write(paths["state_path"], state)
    _json_write(paths["summary_path"], _build_summary(state))


def _run_scout(*, universe_run_id: str, as_of_date: str, scout_params: dict[str, Any]) -> dict[str, Any]:
    from app.universe.scout import run_universe_scout

    cfg = get_config()
    universe_csv = scout_params.get("universe_csv")
    if isinstance(universe_csv, str) and universe_csv.strip():
        universe_csv = Path(universe_csv)
    elif universe_csv is None and cfg.universe_path.exists():
        universe_csv = cfg.universe_path
    return run_universe_scout(
        run_id=universe_run_id,
        as_of_date=as_of_date,
        top_n=max(1, int(scout_params.get("top_n", 50))),
        tickers=scout_params.get("tickers"),
        sector_run_id=scout_params.get("sector_run_id"),
        universe_csv=universe_csv if isinstance(universe_csv, Path) else None,
        universe_source=scout_params.get("universe_source"),
        universe_limit=scout_params.get("universe_limit"),
        with_prices=bool(scout_params.get("with_prices", True)),
        threshold_overrides=scout_params.get("threshold_overrides")
        if isinstance(scout_params.get("threshold_overrides"), dict)
        else None,
        batch_size=max(1, int(scout_params.get("batch_size", 200))),
        max_batches=scout_params.get("max_batches"),
        scout_sec_budget=scout_params.get("scout_sec_budget"),
        scout_net_budget=scout_params.get("scout_net_budget"),
        scout_max_seconds=scout_params.get("scout_max_seconds"),
        force_restart=bool(scout_params.get("force_restart", False)),
    )


def _run_depth_batch(
    *,
    universe_run_id: str,
    batch_run_id: str,
    depth_batch_params: dict[str, Any],
) -> dict[str, Any]:
    from app.universe.batch_runner import run_universe_depth_batch

    return run_universe_depth_batch(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        max_runs=depth_batch_params.get("max_runs"),
        selection_policy=str(depth_batch_params.get("selection_policy") or "QUEUE_ORDER"),
        dry_run=bool(depth_batch_params.get("dry_run", False)),
        mode=str(depth_batch_params.get("mode") or "depth"),
        iterations=depth_batch_params.get("iterations"),
        top_k=depth_batch_params.get("top_k"),
        workers=depth_batch_params.get("workers"),
        with_prices=bool(depth_batch_params.get("with_prices", True)),
        sec_budget=depth_batch_params.get("sec_budget"),
        llm_budget=depth_batch_params.get("llm_budget"),
        llm_provider=depth_batch_params.get("llm_provider"),
    )


def _resume_depth_batch(batch_run_id: str, *, depth_batch_params: dict[str, Any]) -> dict[str, Any]:
    from app.universe.batch_runner import resume_depth_batch

    return resume_depth_batch(
        batch_run_id,
        max_runs=depth_batch_params.get("max_runs"),
        mode=depth_batch_params.get("mode"),
        iterations=depth_batch_params.get("iterations"),
        top_k=depth_batch_params.get("top_k"),
        workers=depth_batch_params.get("workers"),
        with_prices=depth_batch_params.get("with_prices")
        if isinstance(depth_batch_params.get("with_prices"), bool)
        else None,
        sec_budget=depth_batch_params.get("sec_budget"),
        llm_budget=depth_batch_params.get("llm_budget"),
        llm_provider=depth_batch_params.get("llm_provider"),
    )


def _run_rollup(
    *,
    universe_run_id: str,
    batch_run_id: str,
    rollup_params: dict[str, Any],
) -> dict[str, Any]:
    from app.universe.depth_rollup import write_depth_batch_rollup

    return write_depth_batch_rollup(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        top_n=max(1, int(rollup_params.get("top_n", 25))),
        policy=str(rollup_params.get("policy") or "value_first"),
    )


def _run_filing_diff(
    *,
    universe_run_id: str,
    batch_run_id: str,
    as_of_date: str,
    filing_diff_params: dict[str, Any],
) -> dict[str, Any]:
    from app.diff.engine import build_filing_diff_report

    cfg = get_config()
    batch_state = _batch_state_payload(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    batch_status = str(batch_state.get("status") or "").upper()
    if batch_status in {STAGE_NOT_STARTED, STAGE_FAILED, ""}:
        return {
            "status": "SKIPPED",
            "reason": "DEPTH_BATCH_NOT_READY",
            "error": f"DEPTH_BATCH status={batch_status or STAGE_NOT_STARTED}",
            "tickers_attempted": 0,
            "tickers_completed": 0,
            "tickers_failed": 0,
            "tickers_skipped": 0,
            "failed_tickers": [],
            "report_paths": {},
        }

    ticker_to_run_id = _ticker_run_id_map(batch_state)
    tickers = list(ticker_to_run_id.keys())
    max_tickers = filing_diff_params.get("max_tickers")
    if _is_num(max_tickers) and int(max_tickers) > 0:
        tickers = tickers[: int(max_tickers)]
    years_back = max(2, int(filing_diff_params.get("years_back", 5)))

    expected = _stage_expected_paths(universe_run_id, batch_run_id)[STAGE_FILING_DIFF]
    diff_dir = Path(str(expected.get("filing_diff_dir_path") or ""))
    diff_dir.mkdir(parents=True, exist_ok=True)
    global_diff_dir = cfg.outputs_dir / "diffs"
    global_diff_dir.mkdir(parents=True, exist_ok=True)

    report_paths: dict[str, str] = {}
    failed_tickers: list[dict[str, str]] = []
    tickers_completed = 0
    for ticker in tickers:
        source_run_id = ticker_to_run_id.get(ticker)
        if not source_run_id:
            failed_tickers.append({"ticker": ticker, "error": "MISSING_SOURCE_RUN_ID"})
            logger.warning("filing_diff_missing_source_run_id", extra={"ticker": ticker, "universe_run_id": universe_run_id})
            continue
        try:
            report = build_filing_diff_report(ticker=ticker, run_id=source_run_id, years_back=years_back)
            local_path = diff_dir / f"{ticker}.json"
            global_path = global_diff_dir / f"{ticker}_{universe_run_id}_diff.json"
            _json_write(local_path, report.model_dump(mode="json"))
            _json_write(global_path, report.model_dump(mode="json"))
            report_paths[ticker] = str(local_path)
            tickers_completed += 1
        except InvalidFinancialInputError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "filing_diff_ticker_failed",
                extra={"ticker": ticker, "universe_run_id": universe_run_id, "error": str(exc)},
            )
            failed_tickers.append({"ticker": ticker, "error": str(exc)})

    summary = {
        "status": "OK",
        "as_of_date": as_of_date,
        "tickers_attempted": len(tickers),
        "tickers_completed": tickers_completed,
        "tickers_failed": len(failed_tickers),
        "tickers_skipped": 0,
        "failed_tickers": failed_tickers,
        "report_paths": report_paths,
        "filing_diff_summary_path": str(expected.get("filing_diff_summary_path") or ""),
        "filing_diff_dir_path": str(diff_dir),
    }
    _json_write(Path(summary["filing_diff_summary_path"]), summary)
    return summary


def _materialize_pattern_scan_dossiers(
    *,
    universe_run_id: str,
    batch_state: dict[str, Any],
) -> list[str]:
    cfg = get_config()
    ticker_to_run_id = _ticker_run_id_map(batch_state)
    materialized: list[str] = []
    for ticker, source_run_id in ticker_to_run_id.items():
        source = cfg.dossiers_dir / source_run_id / ticker / "dossier.json"
        if not source.exists():
            continue
        dest = cfg.dossiers_dir / universe_run_id / ticker / "dossier.json"
        if not dest.exists() or source.stat().st_mtime > dest.stat().st_mtime:
            _copy_file(source, dest)
        materialized.append(ticker)
    return materialized


def _run_pattern_scan(
    *,
    universe_run_id: str,
    batch_run_id: str,
    as_of_date: str,
    pattern_scan_params: dict[str, Any],
) -> dict[str, Any]:
    from app.patterns.scanner import count_patterns_with_hits, pattern_scan_report_path, scan_peer_set

    batch_state = _batch_state_payload(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    batch_status = str(batch_state.get("status") or "").upper()
    if batch_status == STAGE_PARTIAL or batch_status == "PARTIAL":
        return {
            "status": "SKIPPED",
            "reason": "DEPTH_BATCH_INCOMPLETE",
            "peer_set_size": 0,
            "patterns_tested": 0,
            "patterns_with_signal": 0,
            "total_hits": 0,
        }
    if batch_status != STAGE_DONE:
        return {
            "status": "SKIPPED",
            "reason": "DEPTH_BATCH_NOT_READY",
            "peer_set_size": 0,
            "patterns_tested": 0,
            "patterns_with_signal": 0,
            "total_hits": 0,
        }

    tickers = _materialize_pattern_scan_dossiers(universe_run_id=universe_run_id, batch_state=batch_state)
    if not tickers:
        return {
            "status": "SKIPPED",
            "reason": "NO_DOSSIERS",
            "peer_set_size": 0,
            "patterns_tested": 0,
            "patterns_with_signal": 0,
            "total_hits": 0,
        }
    report = scan_peer_set(
        run_id=universe_run_id,
        tickers=tickers,
        patterns=pattern_scan_params.get("patterns"),
        cfg=get_config(),
    )
    expected = _stage_expected_paths(universe_run_id, batch_run_id)[STAGE_PATTERN_SCAN]
    local_report_path = Path(str(expected.get("pattern_scan_report_path") or ""))
    local_summary_path = Path(str(expected.get("pattern_scan_summary_path") or ""))
    global_report_path = pattern_scan_report_path(universe_run_id, cfg=get_config())
    _json_write(local_report_path, report.model_dump(mode="json"))
    _json_write(global_report_path, report.model_dump(mode="json"))
    summary = {
        "status": "OK",
        "as_of_date": as_of_date,
        "peer_set_size": int(report.peer_set_size),
        "patterns_tested": len(report.pattern_results),
        "patterns_with_signal": count_patterns_with_hits(report),
        "total_hits": sum(int(result.hit_count) for result in report.pattern_results),
        "report_path": str(local_report_path),
        "pattern_scan_report_path": str(local_report_path),
        "pattern_scan_summary_path": str(local_summary_path),
    }
    _json_write(local_summary_path, summary)
    return summary


def _load_pattern_report_for_variant(*, universe_run_id: str, batch_run_id: str) -> dict[str, Any]:
    expected = _stage_expected_paths(universe_run_id, batch_run_id)[STAGE_PATTERN_SCAN]
    return _safe_json(Path(str(expected.get("pattern_scan_report_path") or "")))


def _load_filing_diff_reports(*, universe_run_id: str, batch_run_id: str) -> dict[str, dict[str, Any]]:
    expected = _stage_expected_paths(universe_run_id, batch_run_id)[STAGE_FILING_DIFF]
    summary = _safe_json(Path(str(expected.get("filing_diff_summary_path") or "")))
    report_paths = summary.get("report_paths") if isinstance(summary.get("report_paths"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for ticker, path_value in report_paths.items():
        path = Path(str(path_value or ""))
        payload = _safe_json(path)
        if payload:
            out[str(ticker).strip().upper()] = payload
    return out


def _run_variant_synthesis(
    *,
    universe_run_id: str,
    batch_run_id: str,
    as_of_date: str,
    variant_synthesis_params: dict[str, Any],
) -> dict[str, Any]:
    from app.patterns.schemas import PatternScanReport
    from app.patterns.scanner import summarize_pattern_scan_for_ticker
    from app.synthesis.variant_builder import build_variant_perceptions, variant_perception_report_path

    expected_paths = _stage_expected_paths(universe_run_id, batch_run_id)
    filing_stage_summary = _safe_json(Path(str(expected_paths[STAGE_FILING_DIFF].get("filing_diff_summary_path") or "")))
    pattern_stage_summary = _safe_json(Path(str(expected_paths[STAGE_PATTERN_SCAN].get("pattern_scan_summary_path") or "")))
    filing_ready = bool(filing_stage_summary) and str(filing_stage_summary.get("status") or "").upper() != "SKIPPED"
    pattern_ready = bool(pattern_stage_summary) and str(pattern_stage_summary.get("status") or "").upper() != "SKIPPED"
    if not filing_ready and not pattern_ready:
        return {
            "status": "SKIPPED",
            "reason": "NO_L4_SIGNALS_AVAILABLE",
            "tickers_attempted": 0,
            "tickers_with_perceptions": 0,
            "total_perceptions": 0,
            "high_confidence_count": 0,
            "medium_confidence_count": 0,
            "perception_paths": {},
        }

    batch_state = _batch_state_payload(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    tickers = _completed_depth_tickers(batch_state)
    diff_reports = _load_filing_diff_reports(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    pattern_payload = _load_pattern_report_for_variant(universe_run_id=universe_run_id, batch_run_id=batch_run_id)
    pattern_report = PatternScanReport.model_validate(pattern_payload) if pattern_payload else None
    intangible_by_ticker = _intangible_payloads_by_ticker(universe_run_id=universe_run_id, batch_run_id=batch_run_id)

    local_dir = Path(str(expected_paths[STAGE_VARIANT_SYNTHESIS].get("variant_perceptions_dir_path") or ""))
    local_dir.mkdir(parents=True, exist_ok=True)

    perception_paths: dict[str, str] = {}
    tickers_attempted = 0
    tickers_with_perceptions = 0
    total_perceptions = 0
    high_confidence_count = 0
    medium_confidence_count = 0
    for ticker in tickers:
        available_sources = {"VALUATION"}
        diff_payload = diff_reports.get(ticker)
        if isinstance(diff_payload, dict) and diff_payload:
            available_sources.add("FILING_DIFF")
        pattern_hit_summary = (
            summarize_pattern_scan_for_ticker(pattern_report, ticker)
            if isinstance(pattern_report, PatternScanReport)
            else {}
        )
        if int(pattern_hit_summary.get("pattern_hit_count") or 0) > 0:
            available_sources.add("PATTERN")
        intangible_payload = intangible_by_ticker.get(ticker)
        if isinstance(intangible_payload, dict) and intangible_payload:
            available_sources.add("INTANGIBLE_ECONOMICS")
        if len(available_sources) < 2:
            continue
        tickers_attempted += 1
        try:
            report = build_variant_perceptions(
                ticker=ticker,
                as_of_date=as_of_date,
                run_id=universe_run_id,
                cfg=get_config(),
                diff_report=diff_payload,
                pattern_report=pattern_report.model_dump(mode="json") if isinstance(pattern_report, PatternScanReport) else None,
                intangible_payload=intangible_payload,
                persist=True,
            )
            global_path = variant_perception_report_path(ticker, as_of_date, cfg=get_config())
            local_path = local_dir / global_path.name
            if global_path.exists():
                _copy_file(global_path, local_path)
                perception_paths[ticker] = str(local_path)
            if report.perceptions:
                tickers_with_perceptions += 1
                total_perceptions += len(report.perceptions)
                for perception in report.perceptions:
                    if perception.confidence == "HIGH":
                        high_confidence_count += 1
                    elif perception.confidence == "MEDIUM":
                        medium_confidence_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "variant_synthesis_ticker_failed",
                extra={"ticker": ticker, "universe_run_id": universe_run_id, "error": str(exc)},
            )

    summary = {
        "status": "OK",
        "as_of_date": as_of_date,
        "tickers_attempted": tickers_attempted,
        "tickers_with_perceptions": tickers_with_perceptions,
        "total_perceptions": total_perceptions,
        "high_confidence_count": high_confidence_count,
        "medium_confidence_count": medium_confidence_count,
        "perception_paths": perception_paths,
        "variant_synthesis_summary_path": str(expected_paths[STAGE_VARIANT_SYNTHESIS].get("variant_synthesis_summary_path") or ""),
        "variant_perceptions_dir_path": str(local_dir),
    }
    _json_write(Path(summary["variant_synthesis_summary_path"]), summary)
    return summary


def _run_dossier_pack(
    *,
    universe_run_id: str,
    batch_run_id: str,
    dossier_pack_params: dict[str, Any],
) -> dict[str, Any]:
    from app.universe.dossier_pack import write_dossier_pack

    return write_dossier_pack(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        top_n=max(1, int(dossier_pack_params.get("top_n", 25))),
        policy=str(dossier_pack_params.get("policy") or "value_first"),
    )


def _run_memo_pack(
    *,
    universe_run_id: str,
    batch_run_id: str,
    memo_pack_params: dict[str, Any],
) -> dict[str, Any]:
    from app.universe.memo_pack import write_investment_memo_pack

    return write_investment_memo_pack(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        top_n=max(1, int(memo_pack_params.get("top_n", 25))),
        policy=str(memo_pack_params.get("policy") or "value_first"),
    )


def _build_summary(state: dict[str, Any]) -> dict[str, Any]:
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    scout_stage = stages.get(STAGE_SCOUT) if isinstance(stages.get(STAGE_SCOUT), dict) else {}
    depth_stage = stages.get(STAGE_DEPTH_BATCH) if isinstance(stages.get(STAGE_DEPTH_BATCH), dict) else {}
    rollup_stage = stages.get(STAGE_ROLLUP) if isinstance(stages.get(STAGE_ROLLUP), dict) else {}
    filing_diff_stage = stages.get(STAGE_FILING_DIFF) if isinstance(stages.get(STAGE_FILING_DIFF), dict) else {}
    pattern_stage = stages.get(STAGE_PATTERN_SCAN) if isinstance(stages.get(STAGE_PATTERN_SCAN), dict) else {}
    variant_stage = stages.get(STAGE_VARIANT_SYNTHESIS) if isinstance(stages.get(STAGE_VARIANT_SYNTHESIS), dict) else {}
    dossier_stage = stages.get(STAGE_DOSSIER_PACK) if isinstance(stages.get(STAGE_DOSSIER_PACK), dict) else {}
    memo_stage = stages.get(STAGE_MEMO_PACK) if isinstance(stages.get(STAGE_MEMO_PACK), dict) else {}

    scout_result = scout_stage.get("result") if isinstance(scout_stage.get("result"), dict) else {}
    depth_result = depth_stage.get("result") if isinstance(depth_stage.get("result"), dict) else {}
    rollup_artifacts = rollup_stage.get("artifact_paths") if isinstance(rollup_stage.get("artifact_paths"), dict) else {}
    dossier_artifacts = dossier_stage.get("artifact_paths") if isinstance(dossier_stage.get("artifact_paths"), dict) else {}
    memo_artifacts = memo_stage.get("artifact_paths") if isinstance(memo_stage.get("artifact_paths"), dict) else {}

    global_shortlist_path = Path(str(rollup_artifacts.get("global_shortlist_json_path") or ""))
    shortlist_payload = _safe_json(global_shortlist_path) if global_shortlist_path.exists() else {}
    shortlist_rows = [row for row in (shortlist_payload.get("rows") or []) if isinstance(row, dict)]
    top10 = [
        {
            "ticker": str(row.get("ticker") or ""),
            "implied_return_base": row.get("implied_return_base", "UNKNOWN"),
            "mos_epv": row.get("mos_epv", "UNKNOWN"),
            "yield_metric_used": str(row.get("yield_metric_used") or "UNKNOWN"),
        }
        for row in shortlist_rows[:10]
    ]

    batch_summary_path = Path(str(depth_stage.get("artifact_paths", {}).get("batch_summary_path") or ""))
    batch_summary_payload = _safe_json(batch_summary_path) if batch_summary_path.exists() else {}

    scout_summary_path = Path(str((scout_stage.get("artifact_paths") or {}).get("universe_summary_path") or ""))
    scout_summary_payload = _safe_json(scout_summary_path) if scout_summary_path.exists() else {}
    scout_counts = scout_result.get("counts") if isinstance(scout_result.get("counts"), dict) else {}
    if not scout_counts:
        scout_counts = (
            scout_summary_payload.get("counts")
            if isinstance(scout_summary_payload.get("counts"), dict)
            else {}
        )

    return {
        "universe_run_id": str(state.get("universe_run_id") or ""),
        "as_of_date": str(state.get("as_of_date") or ""),
        "depth": str(state.get("depth") or "fundamentals"),
        "status": str(state.get("status") or AUTOPILOT_RUNNING),
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "financial_integrity": (
            state.get("financial_integrity")
            if isinstance(state.get("financial_integrity"), dict)
            else {}
        ),
        "updated_at": utc_now_iso(),
        "scout_counts": scout_counts,
        "scout_hydration_status": str(
            scout_result.get("hydration_status")
            or scout_summary_payload.get("hydration_status")
            or "UNKNOWN"
        ),
        "primary_scout_blocker": str(
            scout_result.get("primary_scout_blocker")
            or scout_summary_payload.get("primary_scout_blocker")
            or ""
        ),
        "scout_last_progress_phase": str(
            scout_result.get("last_progress_phase")
            or scout_summary_payload.get("last_progress_phase")
            or "UNKNOWN"
        ),
        "scout_stalled_reason_code": str(
            scout_result.get("stalled_reason_code")
            or scout_summary_payload.get("stalled_reason_code")
            or ""
        ),
        "scout_hydration_progress": scout_summary_payload.get("hydration_progress")
        if isinstance(scout_summary_payload.get("hydration_progress"), dict)
        else {},
        "facts_blocker_histogram": scout_summary_payload.get("facts_blockers", {}).get("facts_blocker_histogram")
        if isinstance(scout_summary_payload.get("facts_blockers"), dict)
        and isinstance(scout_summary_payload.get("facts_blockers", {}).get("facts_blocker_histogram"), dict)
        else {},
        "retryable_facts_blocker_count": int(scout_summary_payload.get("facts_blockers", {}).get("retryable_facts_blocker_count") or 0)
        if isinstance(scout_summary_payload.get("facts_blockers"), dict)
        else 0,
        "terminal_facts_blocker_count": int(scout_summary_payload.get("facts_blockers", {}).get("terminal_facts_blocker_count") or 0)
        if isinstance(scout_summary_payload.get("facts_blockers"), dict)
        else 0,
        "partial_usable_facts_count": int(scout_summary_payload.get("facts_blockers", {}).get("partial_usable_facts_count") or 0)
        if isinstance(scout_summary_payload.get("facts_blockers"), dict)
        else 0,
        "top_retryable_facts_blockers": scout_summary_payload.get("facts_blockers", {}).get("top_retryable_facts_blockers")
        if isinstance(scout_summary_payload.get("facts_blockers"), dict)
        and isinstance(scout_summary_payload.get("facts_blockers", {}).get("top_retryable_facts_blockers"), list)
        else [],
        "top_terminal_facts_blockers": scout_summary_payload.get("facts_blockers", {}).get("top_terminal_facts_blockers")
        if isinstance(scout_summary_payload.get("facts_blockers"), dict)
        and isinstance(scout_summary_payload.get("facts_blockers", {}).get("top_terminal_facts_blockers"), list)
        else [],
        "economic_fail_count_vs_evidence_fail_count": scout_summary_payload.get("facts_blockers", {}).get("economic_fail_count_vs_evidence_fail_count")
        if isinstance(scout_summary_payload.get("facts_blockers"), dict)
        and isinstance(scout_summary_payload.get("facts_blockers", {}).get("economic_fail_count_vs_evidence_fail_count"), dict)
        else {},
        "depth_batch_completion": {
            "run_status": str(depth_result.get("run_status") or ""),
            "planned_count": int(depth_result.get("planned_count") or 0),
            "completed_count": int(depth_result.get("completed_count") or 0),
            "run_count_total": int(batch_summary_payload.get("total_planned") or 0),
            "run_count_done": int(batch_summary_payload.get("done_count") or 0),
            "run_count_failed": int(batch_summary_payload.get("failed_count") or 0),
            "run_count_cancelled": int(batch_summary_payload.get("cancelled_count") or 0),
        },
        "global_shortlist_top_10": top10,
        "watchlist_paths": {
            "watchlist_csv_path": str(dossier_artifacts.get("watchlist_csv_path") or ""),
            "watchlist_json_path": str(dossier_artifacts.get("watchlist_json_path") or ""),
            "watchlist_state_path": str(memo_artifacts.get("watchlist_state_path") or ""),
        },
        "dossier_pack_manifest_path": str(dossier_artifacts.get("manifest_path") or ""),
        "memo_pack_manifest_path": str(memo_artifacts.get("manifest_path") or ""),
        "l4_stage_summary": {
            "filing_diff": {
                "status": str(filing_diff_stage.get("status") or STAGE_NOT_STARTED),
                "result": filing_diff_stage.get("result") if isinstance(filing_diff_stage.get("result"), dict) else {},
            },
            "pattern_scan": {
                "status": str(pattern_stage.get("status") or STAGE_NOT_STARTED),
                "result": pattern_stage.get("result") if isinstance(pattern_stage.get("result"), dict) else {},
            },
            "variant_synthesis": {
                "status": str(variant_stage.get("status") or STAGE_NOT_STARTED),
                "result": variant_stage.get("result") if isinstance(variant_stage.get("result"), dict) else {},
            },
        },
        "artifact_paths": {
            stage: (
                stage_payload.get("artifact_paths")
                if isinstance(stage_payload, dict)
                and isinstance(stage_payload.get("artifact_paths"), dict)
                else {}
            )
            for stage, stage_payload in stages.items()
            if isinstance(stage_payload, dict)
        },
    }


def run_universe_autopilot(
    universe_run_id: str,
    as_of_date: str | None = None,
    scout_params: dict[str, Any] | None = None,
    depth_batch_params: dict[str, Any] | None = None,
    rollup_params: dict[str, Any] | None = None,
    filing_diff_params: dict[str, Any] | None = None,
    pattern_scan_params: dict[str, Any] | None = None,
    variant_synthesis_params: dict[str, Any] | None = None,
    dossier_pack_params: dict[str, Any] | None = None,
    memo_pack_params: dict[str, Any] | None = None,
    max_stages: int | None = None,
    resume: bool = True,
    depth: str = "fundamentals",
    review_intake_from_run: str | None = None,
    review_intake_max_names: int | None = None,
    director_replay_seeds_from_run: str | None = None,
    director_replay_max_names: int | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    as_of = str(as_of_date or date.today().isoformat())
    scout_params = dict(scout_params or {})
    depth_batch_params = dict(depth_batch_params or {})
    rollup_params = dict(rollup_params or {})
    filing_diff_params = dict(filing_diff_params or {})
    pattern_scan_params = dict(pattern_scan_params or {})
    variant_synthesis_params = dict(variant_synthesis_params or {})
    dossier_pack_params = dict(dossier_pack_params or {})
    memo_pack_params = dict(memo_pack_params or {})
    requested_depth = _normalize_depth(depth)
    if bool(force):
        scout_params["force_restart"] = True
    elif not bool(resume) and "force_restart" not in scout_params:
        scout_params["force_restart"] = True

    batch_run_id = _batch_run_id(universe_run_id)
    expected_paths = _stage_expected_paths(universe_run_id, batch_run_id)
    paths = _autopilot_paths(universe_run_id)
    state_existing = _safe_json(paths["state_path"]) if resume else {}
    if state_existing:
        state = state_existing
        state["as_of_date"] = str(state.get("as_of_date") or as_of)
        state["batch_run_id"] = str(state.get("batch_run_id") or batch_run_id)
        effective_depth = _normalize_depth(state.get("depth") or requested_depth)
        state["depth"] = effective_depth
        state["params"] = {
            "scout_params": scout_params,
            "depth_batch_params": depth_batch_params,
            "rollup_params": rollup_params,
            "filing_diff_params": filing_diff_params,
            "pattern_scan_params": pattern_scan_params,
            "variant_synthesis_params": variant_synthesis_params,
            "dossier_pack_params": dossier_pack_params,
            "memo_pack_params": memo_pack_params,
        }
        stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
        for stage in _STAGE_ORDER:
            if stage not in stages or not isinstance(stages.get(stage), dict):
                stages[stage] = _init_stage_state(expected_paths[stage])
            else:
                stage_payload = stages[stage]
                artifacts = stage_payload.get("artifact_paths") if isinstance(stage_payload.get("artifact_paths"), dict) else {}
                merged = dict(expected_paths[stage])
                merged.update({str(k): str(v) for k, v in artifacts.items() if str(k).strip()})
                stage_payload["artifact_paths"] = merged
        state["stages"] = stages
    else:
        effective_depth = requested_depth
        state = _default_state(
            universe_run_id=universe_run_id,
            as_of_date=as_of,
            batch_run_id=batch_run_id,
            depth=effective_depth,
            scout_params=scout_params,
            depth_batch_params=depth_batch_params,
            rollup_params=rollup_params,
            filing_diff_params=filing_diff_params,
            pattern_scan_params=pattern_scan_params,
            variant_synthesis_params=variant_synthesis_params,
            dossier_pack_params=dossier_pack_params,
            memo_pack_params=memo_pack_params,
        )

    if _state_is_cancelled(state):
        _write_state(universe_run_id, state)
        return {
            "status": "CANCELLED",
            "universe_run_id": universe_run_id,
            "stop_reason_code": str(state.get("stop_reason_code") or STOP_CANCEL_REQUESTED),
            "stop_summary": str(state.get("stop_summary") or ""),
            "autopilot_state_path": str(paths["state_path"]),
            "autopilot_summary_path": str(paths["summary_path"]),
        }

    if bool(dry_run):
        state["status"] = AUTOPILOT_PLANNED
        state["stop_reason_code"] = STOP_DRY_RUN
        state["stop_summary"] = "Dry-run planned only."
        _write_state(universe_run_id, state)
        return {
            "status": AUTOPILOT_PLANNED,
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "as_of_date": as_of,
            "depth": effective_depth,
            "plan": [
                {"stage": stage, "expected_artifacts": expected_paths[stage]}
                for stage in _STAGE_ORDER
            ],
            "autopilot_state_path": str(paths["state_path"]),
            "autopilot_summary_path": str(paths["summary_path"]),
        }

    state["status"] = AUTOPILOT_RUNNING
    state["stop_reason_code"] = ""
    state["stop_summary"] = ""

    stages_executed = 0
    for stage_name in _STAGE_ORDER:
        if _is_num(max_stages) and int(max_stages) > 0 and stages_executed >= int(max_stages):
            state["status"] = AUTOPILOT_PARTIAL
            state["stop_reason_code"] = STOP_MAX_STAGES_REACHED
            state["stop_summary"] = "Reached max_stages before completion."
            break
        if _state_is_cancelled(state):
            state["status"] = AUTOPILOT_CANCELLED
            state["stop_reason_code"] = STOP_CANCEL_REQUESTED
            if not str(state.get("stop_summary") or "").strip():
                state["stop_summary"] = "Autopilot cancelled by operator."
            break

        stage_state = state["stages"][stage_name]
        if not _stage_enabled_for_depth(stage_name, effective_depth):
            stage_state["status"] = STAGE_DONE
            stage_state["result"] = {
                "status": "OK",
                "skipped_by_depth": True,
                "depth": effective_depth,
            }
            stage_state["error"] = None
            stage_state["updated_at"] = utc_now_iso()
            stages_executed += 1
            _write_state(universe_run_id, state)
            continue
        if resume and not force and stage_name in {STAGE_SCOUT, STAGE_DEPTH_QUEUE} and _stage_artifacts_exist(stage_name, stage_state):
            stage_state["status"] = STAGE_DONE
            stage_state["updated_at"] = utc_now_iso()
            _write_state(universe_run_id, state)
            continue
        if resume and not force and stage_name == STAGE_DEPTH_BATCH:
            batch_state_path = str(stage_state.get("artifact_paths", {}).get("batch_state_path") or "")
            if _depth_batch_complete(batch_state_path):
                stage_state["status"] = STAGE_DONE
                stage_state["progress"] = _depth_batch_progress(batch_state_path)
                stage_state["updated_at"] = utc_now_iso()
                _write_state(universe_run_id, state)
                continue

        stage_state["status"] = STAGE_RUNNING
        stage_state["started_at"] = utc_now_iso()
        stage_state["updated_at"] = utc_now_iso()
        stage_state["error"] = None
        _write_state(universe_run_id, state)

        try:
            if stage_name == STAGE_SCOUT:
                result = _run_scout(universe_run_id=universe_run_id, as_of_date=as_of, scout_params=scout_params)
                stage_state["result"] = result if isinstance(result, dict) else {}
                if not _stage_artifacts_exist(stage_name, stage_state):
                    _merge_artifact_paths(stage_state)
                scout_run_status = str(stage_state["result"].get("run_status") or stage_state["result"].get("status") or "").upper()
                if scout_run_status in {STAGE_CANCELLED, AUTOPILOT_CANCELLED}:
                    stage_state["status"] = STAGE_CANCELLED
                    state["status"] = AUTOPILOT_CANCELLED
                    state["stop_reason_code"] = STOP_CANCEL_REQUESTED
                    state["stop_summary"] = str(
                        stage_state["result"].get("stop_summary") or "Scout stage cancelled."
                    )
                    _write_state(universe_run_id, state)
                    break
                if scout_run_status in {STAGE_FAILED, AUTOPILOT_FAILED}:
                    stage_state["status"] = STAGE_FAILED
                    state["status"] = AUTOPILOT_FAILED
                    state["stop_reason_code"] = STOP_STAGE_FAILED
                    state["stop_summary"] = str(
                        stage_state["result"].get("stop_summary") or "Scout stage failed."
                    )
                    _write_state(universe_run_id, state)
                    return {
                        "status": "FAILED",
                        "universe_run_id": universe_run_id,
                        "batch_run_id": batch_run_id,
                        "failed_stage": stage_name,
                        "error": state["stop_summary"],
                        "autopilot_state_path": str(paths["state_path"]),
                        "autopilot_summary_path": str(paths["summary_path"]),
                    }
                if scout_run_status == STAGE_PARTIAL:
                    stage_state["status"] = STAGE_PARTIAL
                    state["status"] = AUTOPILOT_PARTIAL
                    state["stop_reason_code"] = STOP_STAGE_FAILED
                    state["stop_summary"] = str(
                        stage_state["result"].get("stop_summary") or "Scout stage completed PARTIAL."
                    )
                    _write_state(universe_run_id, state)
                    break
                stage_state["status"] = STAGE_DONE
            elif stage_name == STAGE_DEPTH_QUEUE:
                if not _stage_artifacts_exist(stage_name, stage_state):
                    raise ValueError("depth_queue.json missing after scout stage")
                stage_state["result"] = {
                    "status": "OK",
                    "depth_queue_path": stage_state["artifact_paths"].get("depth_queue_path"),
                }
                stage_state["status"] = STAGE_DONE
            elif stage_name == STAGE_DEPTH_BATCH:
                batch_state_path = Path(str(stage_state["artifact_paths"].get("batch_state_path") or ""))
                result: dict[str, Any]
                if resume and batch_state_path.exists() and not force:
                    existing_batch_state = _safe_json(batch_state_path)
                    existing_status = str(existing_batch_state.get("status") or "").upper()
                    if existing_status == "DONE":
                        result = {
                            "status": "OK",
                            "run_status": "DONE",
                            "batch_state_path": str(batch_state_path),
                            "batch_summary_path": stage_state["artifact_paths"].get("batch_summary_path"),
                            "batch_log_path": stage_state["artifact_paths"].get("batch_log_path"),
                        }
                    elif existing_status == "CANCELLED":
                        stage_state["status"] = STAGE_CANCELLED
                        stage_state["result"] = {
                            "status": "CANCELLED",
                            "run_status": "CANCELLED",
                            "batch_state_path": str(batch_state_path),
                        }
                        state["status"] = AUTOPILOT_CANCELLED
                        state["stop_reason_code"] = STOP_CANCEL_REQUESTED
                        state["stop_summary"] = "Depth batch run is CANCELLED."
                        _write_state(universe_run_id, state)
                        break
                    else:
                        result = _resume_depth_batch(batch_run_id, depth_batch_params=depth_batch_params)
                else:
                    result = _run_depth_batch(
                        universe_run_id=universe_run_id,
                        batch_run_id=batch_run_id,
                        depth_batch_params=depth_batch_params,
                    )
                stage_state["result"] = result if isinstance(result, dict) else {}
                run_status = str(stage_state["result"].get("run_status") or "").upper()
                stage_state["progress"] = _depth_batch_progress(str(batch_state_path))
                stage_state["status"] = STAGE_PARTIAL if run_status == "PARTIAL" else STAGE_DONE
            elif stage_name == STAGE_ROLLUP:
                batch_artifacts = (
                    state.get("stages", {}).get(STAGE_DEPTH_BATCH, {}).get("artifact_paths")
                    if isinstance(state.get("stages"), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_DEPTH_BATCH), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_DEPTH_BATCH, {}).get("artifact_paths"), dict)
                    else {}
                )
                rerun = _should_rerun_rollup(
                    stage_state,
                    force=bool(force),
                    input_paths=[
                        Path(str(batch_artifacts.get("batch_state_path") or "")),
                        Path(str(batch_artifacts.get("batch_summary_path") or "")),
                    ],
                )
                if rerun:
                    result = _run_rollup(
                        universe_run_id=universe_run_id,
                        batch_run_id=batch_run_id,
                        rollup_params=rollup_params,
                    )
                    stage_state["result"] = result if isinstance(result, dict) else {}
                    _merge_artifact_paths(stage_state)
                else:
                    stage_state["result"] = {"status": "OK", "reused_existing": True}
                stage_state["status"] = STAGE_DONE
            elif stage_name == STAGE_FILING_DIFF:
                result = _run_filing_diff(
                    universe_run_id=universe_run_id,
                    batch_run_id=batch_run_id,
                    as_of_date=as_of,
                    filing_diff_params=filing_diff_params,
                )
                stage_state["result"] = result if isinstance(result, dict) else {}
                _merge_artifact_paths(stage_state)
                stage_state["status"] = STAGE_DONE
            elif stage_name == STAGE_PATTERN_SCAN:
                result = _run_pattern_scan(
                    universe_run_id=universe_run_id,
                    batch_run_id=batch_run_id,
                    as_of_date=as_of,
                    pattern_scan_params=pattern_scan_params,
                )
                stage_state["result"] = result if isinstance(result, dict) else {}
                _merge_artifact_paths(stage_state)
                stage_state["status"] = STAGE_DONE
            elif stage_name == STAGE_VARIANT_SYNTHESIS:
                result = _run_variant_synthesis(
                    universe_run_id=universe_run_id,
                    batch_run_id=batch_run_id,
                    as_of_date=as_of,
                    variant_synthesis_params=variant_synthesis_params,
                )
                stage_state["result"] = result if isinstance(result, dict) else {}
                _merge_artifact_paths(stage_state)
                stage_state["status"] = STAGE_DONE
            elif stage_name == STAGE_DOSSIER_PACK:
                rollup_artifacts = (
                    state.get("stages", {}).get(STAGE_ROLLUP, {}).get("artifact_paths")
                    if isinstance(state.get("stages"), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_ROLLUP), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_ROLLUP, {}).get("artifact_paths"), dict)
                    else {}
                )
                filing_diff_artifacts = (
                    state.get("stages", {}).get(STAGE_FILING_DIFF, {}).get("artifact_paths")
                    if isinstance(state.get("stages"), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_FILING_DIFF), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_FILING_DIFF, {}).get("artifact_paths"), dict)
                    else {}
                )
                pattern_artifacts = (
                    state.get("stages", {}).get(STAGE_PATTERN_SCAN, {}).get("artifact_paths")
                    if isinstance(state.get("stages"), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_PATTERN_SCAN), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_PATTERN_SCAN, {}).get("artifact_paths"), dict)
                    else {}
                )
                variant_artifacts = (
                    state.get("stages", {}).get(STAGE_VARIANT_SYNTHESIS, {}).get("artifact_paths")
                    if isinstance(state.get("stages"), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_VARIANT_SYNTHESIS), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_VARIANT_SYNTHESIS, {}).get("artifact_paths"), dict)
                    else {}
                )
                rerun = _should_rerun_dossier(
                    stage_state,
                    force=bool(force),
                    input_paths=[
                        Path(str(rollup_artifacts.get("global_shortlist_json_path") or "")),
                        Path(str(rollup_artifacts.get("global_rollup_json_path") or "")),
                        Path(str(filing_diff_artifacts.get("filing_diff_summary_path") or "")),
                        Path(str(pattern_artifacts.get("pattern_scan_summary_path") or "")),
                        Path(str(variant_artifacts.get("variant_synthesis_summary_path") or "")),
                    ],
                )
                if rerun:
                    result = _run_dossier_pack(
                        universe_run_id=universe_run_id,
                        batch_run_id=batch_run_id,
                        dossier_pack_params=dossier_pack_params,
                    )
                    stage_state["result"] = result if isinstance(result, dict) else {}
                    _merge_artifact_paths(stage_state)
                    if isinstance(stage_state["result"].get("dossier_pack_dir"), str):
                        stage_state["artifact_paths"]["dossier_pack_dir"] = str(stage_state["result"]["dossier_pack_dir"])
                    if isinstance(stage_state["result"].get("manifest_path"), str):
                        stage_state["artifact_paths"]["manifest_path"] = str(stage_state["result"]["manifest_path"])
                else:
                    stage_state["result"] = {"status": "OK", "reused_existing": True}
                stage_state["status"] = STAGE_DONE
            elif stage_name == STAGE_MEMO_PACK:
                dossier_artifacts = (
                    state.get("stages", {}).get(STAGE_DOSSIER_PACK, {}).get("artifact_paths")
                    if isinstance(state.get("stages"), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_DOSSIER_PACK), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_DOSSIER_PACK, {}).get("artifact_paths"), dict)
                    else {}
                )
                rollup_artifacts = (
                    state.get("stages", {}).get(STAGE_ROLLUP, {}).get("artifact_paths")
                    if isinstance(state.get("stages"), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_ROLLUP), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_ROLLUP, {}).get("artifact_paths"), dict)
                    else {}
                )
                variant_artifacts = (
                    state.get("stages", {}).get(STAGE_VARIANT_SYNTHESIS, {}).get("artifact_paths")
                    if isinstance(state.get("stages"), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_VARIANT_SYNTHESIS), dict)
                    and isinstance(state.get("stages", {}).get(STAGE_VARIANT_SYNTHESIS, {}).get("artifact_paths"), dict)
                    else {}
                )
                rerun = _should_rerun_memo(
                    stage_state,
                    force=bool(force),
                    input_paths=[
                        Path(str(rollup_artifacts.get("global_shortlist_json_path") or "")),
                        Path(str(dossier_artifacts.get("manifest_path") or "")),
                        Path(str(variant_artifacts.get("variant_synthesis_summary_path") or "")),
                    ],
                )
                if rerun:
                    result = _run_memo_pack(
                        universe_run_id=universe_run_id,
                        batch_run_id=batch_run_id,
                        memo_pack_params=memo_pack_params,
                    )
                    stage_state["result"] = result if isinstance(result, dict) else {}
                    _merge_artifact_paths(stage_state)
                else:
                    stage_state["result"] = {"status": "OK", "reused_existing": True}
                stage_state["status"] = STAGE_DONE
            else:
                raise ValueError(f"Unsupported stage={stage_name}")
        except InvalidFinancialInputError as exc:
            stage_state["status"] = STAGE_FAILED
            stage_state["error"] = str(exc)
            stage_state["result"] = {
                "status": exc.status,
                "financial_integrity": exc.result.to_dict(),
            }
            stage_state["updated_at"] = utc_now_iso()
            state["status"] = AUTOPILOT_FAILED
            state["stop_reason_code"] = exc.status
            state["stop_summary"] = f"{stage_name} failed financial-integrity validation."
            state["financial_integrity"] = exc.result.to_dict()
            _write_state(universe_run_id, state)
            raise
        except Exception as exc:  # noqa: BLE001
            stage_state["status"] = STAGE_FAILED
            stage_state["error"] = str(exc)
            state["status"] = AUTOPILOT_FAILED
            state["stop_reason_code"] = STOP_STAGE_FAILED
            state["stop_summary"] = f"{stage_name} failed: {exc}"
            stage_state["updated_at"] = utc_now_iso()
            _write_state(universe_run_id, state)
            return {
                "status": "FAILED",
                "universe_run_id": universe_run_id,
                "batch_run_id": batch_run_id,
                "failed_stage": stage_name,
                "error": str(exc),
                "autopilot_state_path": str(paths["state_path"]),
                "autopilot_summary_path": str(paths["summary_path"]),
            }

        stages_executed += 1
        stage_state["updated_at"] = utc_now_iso()
        _write_state(universe_run_id, state)

    if state.get("status") == AUTOPILOT_RUNNING:
        all_done = all(
            str((state["stages"][stage].get("status") or "")).upper() in {STAGE_DONE, STAGE_PARTIAL}
            for stage in _STAGE_ORDER
        )
        if all_done:
            state["status"] = AUTOPILOT_DONE
            state["stop_reason_code"] = STOP_COMPLETED
            state["stop_summary"] = "Autopilot completed."
        else:
            state["status"] = AUTOPILOT_PARTIAL
            if not str(state.get("stop_reason_code") or "").strip():
                state["stop_reason_code"] = STOP_MAX_STAGES_REACHED
                state["stop_summary"] = "Autopilot stopped before all stages completed."
        _write_state(universe_run_id, state)

    return {
        "status": str(state.get("status") or AUTOPILOT_RUNNING),
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "as_of_date": str(state.get("as_of_date") or as_of),
        "depth": str(state.get("depth") or effective_depth),
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "stages": {
            stage: str((state["stages"].get(stage) or {}).get("status") or STAGE_NOT_STARTED)
            for stage in _STAGE_ORDER
        },
        "autopilot_state_path": str(paths["state_path"]),
        "autopilot_summary_path": str(paths["summary_path"]),
    }


def universe_autopilot_status(universe_run_id: str) -> dict[str, Any]:
    paths = _autopilot_paths(universe_run_id)
    state = _safe_json(paths["state_path"])
    if not state:
        return {
            "status": "MISSING",
            "universe_run_id": universe_run_id,
            "autopilot_state_path": str(paths["state_path"]),
            "autopilot_summary_path": str(paths["summary_path"]),
        }
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    depth_stage = stages.get(STAGE_DEPTH_BATCH) if isinstance(stages.get(STAGE_DEPTH_BATCH), dict) else {}
    depth_batch_state_path = str((depth_stage.get("artifact_paths") or {}).get("batch_state_path") or "")
    depth_progress = _depth_batch_progress(depth_batch_state_path)
    return {
        "status": "OK",
        "universe_run_id": universe_run_id,
        "run_status": str(state.get("status") or AUTOPILOT_RUNNING),
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "as_of_date": str(state.get("as_of_date") or ""),
        "batch_run_id": str(state.get("batch_run_id") or ""),
        "depth": str(state.get("depth") or "fundamentals"),
        "stages": {
            stage: {
                "status": str((stages.get(stage) or {}).get("status") or STAGE_NOT_STARTED),
                "artifact_paths": (
                    (stages.get(stage) or {}).get("artifact_paths")
                    if isinstance((stages.get(stage) or {}).get("artifact_paths"), dict)
                    else {}
                ),
                "error": (stages.get(stage) or {}).get("error"),
            }
            for stage in _STAGE_ORDER
        },
        "depth_batch_progress": depth_progress,
        "autopilot_state_path": str(paths["state_path"]),
        "autopilot_summary_path": str(paths["summary_path"]),
    }


def open_universe_autopilot(universe_run_id: str) -> dict[str, Any]:
    status_payload = universe_autopilot_status(universe_run_id)
    if status_payload.get("status") != "OK":
        return status_payload
    paths = _autopilot_paths(universe_run_id)
    summary = _safe_json(paths["summary_path"])
    return {
        "status": "OK",
        "universe_run_id": universe_run_id,
        "run_status": status_payload.get("run_status"),
        "stop_reason_code": status_payload.get("stop_reason_code"),
        "stop_summary": status_payload.get("stop_summary"),
        "as_of_date": status_payload.get("as_of_date"),
        "batch_run_id": status_payload.get("batch_run_id"),
        "stages": status_payload.get("stages"),
        "summary": summary,
        "autopilot_state_path": str(paths["state_path"]),
        "autopilot_summary_path": str(paths["summary_path"]),
    }


def cancel_universe_autopilot(universe_run_id: str, reason: str) -> dict[str, Any]:
    paths = _autopilot_paths(universe_run_id)
    state = _safe_json(paths["state_path"])
    if not state:
        return {
            "status": "MISSING",
            "universe_run_id": universe_run_id,
            "autopilot_state_path": str(paths["state_path"]),
        }
    state["status"] = AUTOPILOT_CANCELLED
    state["stop_reason_code"] = STOP_CANCEL_REQUESTED
    state["stop_summary"] = str(reason or "Cancelled by operator.")
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    for stage in _STAGE_ORDER:
        stage_state = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        if str(stage_state.get("status") or "").upper() == STAGE_RUNNING:
            stage_state["status"] = STAGE_CANCELLED
            stage_state["updated_at"] = utc_now_iso()
            stages[stage] = stage_state
    state["stages"] = stages
    _write_state(universe_run_id, state)
    return {
        "status": "OK",
        "universe_run_id": universe_run_id,
        "run_status": AUTOPILOT_CANCELLED,
        "stop_reason_code": STOP_CANCEL_REQUESTED,
        "stop_summary": str(state.get("stop_summary") or ""),
        "autopilot_state_path": str(paths["state_path"]),
        "autopilot_summary_path": str(paths["summary_path"]),
    }
