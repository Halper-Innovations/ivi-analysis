from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso
from app.universe.depth_rollup import write_depth_batch_rollup


UNKNOWN = "UNKNOWN"
_WATCHLIST_COLUMNS = [
    "ticker",
    "rank",
    "value_gate_status",
    "implied_return_base",
    "mos_epv",
    "mos_netnet",
    "yield_metric_used",
    "yield_denominator_used",
    "owner_earnings_yield_ev_3y",
    "fcf_yield_ev_3y",
    "price_status",
    "valuation_status",
    "primary_blocker",
]


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _render_value(value: Any) -> str:
    return str(value) if _is_num(value) else UNKNOWN


def _packet_markdown(packet: dict[str, Any]) -> str:
    ticker = str(packet.get("ticker") or "")
    lines: list[str] = []
    lines.append(f"# Candidate Packet: {ticker}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Rank Global: `{packet.get('rank_global')}`")
    lines.append(f"- Value Gate Status: `{packet.get('value_gate_status')}`")
    lines.append(f"- Gate Reasons: `{', '.join(packet.get('value_gate_reasons') or [])}`")
    lines.append("")
    lines.append("## Price / Valuation")
    price = packet.get("price") if isinstance(packet.get("price"), dict) else {}
    valuation = packet.get("valuation") if isinstance(packet.get("valuation"), dict) else {}
    lines.append(f"- current_price: `{_render_value(price.get('current_price'))}`")
    lines.append(f"- price_asof_used: `{price.get('price_asof_used', UNKNOWN)}`")
    lines.append(f"- price_source_resolution: `{price.get('price_source_resolution', UNKNOWN)}`")
    lines.append(f"- price_reason_code: `{price.get('price_reason_code', UNKNOWN)}`")
    lines.append(f"- intrinsic_per_share_base: `{_render_value(valuation.get('intrinsic_per_share_base'))}`")
    lines.append(f"- implied_return_base: `{_render_value(valuation.get('implied_return_base'))}`")
    lines.append(f"- valuation_reason_code: `{valuation.get('valuation_reason_code', UNKNOWN)}`")
    lines.append("")
    lines.append("## Graham / Dodd")
    gd = packet.get("graham_dodd") if isinstance(packet.get("graham_dodd"), dict) else {}
    lines.append(f"- epv_per_share: `{_render_value(gd.get('epv_per_share'))}`")
    lines.append(f"- mos_epv: `{_render_value(gd.get('mos_epv'))}`")
    lines.append(f"- netnet_per_share: `{_render_value(gd.get('netnet_per_share'))}`")
    lines.append(f"- mos_netnet: `{_render_value(gd.get('mos_netnet'))}`")
    lines.append(f"- gd_primary_reason_code: `{gd.get('gd_primary_reason_code', UNKNOWN)}`")
    lines.append("")
    lines.append("## Yield")
    yields = packet.get("yields") if isinstance(packet.get("yields"), dict) else {}
    lines.append(f"- yield_metric_used: `{yields.get('yield_metric_used', UNKNOWN)}`")
    lines.append(f"- yield_denominator_used: `{yields.get('yield_denominator_used', UNKNOWN)}`")
    lines.append(f"- owner_earnings_yield_ev_3y: `{_render_value(yields.get('owner_earnings_yield_ev_3y'))}`")
    lines.append(f"- fcf_yield_ev_3y: `{_render_value(yields.get('fcf_yield_ev_3y'))}`")
    lines.append(f"- yield_reason_code: `{yields.get('yield_reason_code', UNKNOWN)}`")
    lines.append("")
    lines.append("## Leverage / Quality / Risk")
    leverage = packet.get("leverage") if isinstance(packet.get("leverage"), dict) else {}
    quality_risk = packet.get("quality_risk") if isinstance(packet.get("quality_risk"), dict) else {}
    lines.append(f"- net_debt_proxy: `{_render_value(leverage.get('net_debt_proxy'))}`")
    lines.append(f"- ev_status: `{leverage.get('ev_status', UNKNOWN)}`")
    lines.append(f"- net_debt_reason_code: `{leverage.get('net_debt_reason_code', UNKNOWN)}`")
    lines.append(f"- quality_score: `{_render_value(quality_risk.get('quality_score'))}`")
    lines.append(f"- risk_penalty: `{_render_value(quality_risk.get('risk_penalty'))}`")
    lines.append(f"- risk_blockers: `{', '.join(quality_risk.get('risk_blocker_categories') or [])}`")
    lines.append("")
    lines.append("## Coverage")
    coverage = packet.get("coverage") if isinstance(packet.get("coverage"), dict) else {}
    for key in ["price_status", "shares_status", "fcf_status", "facts_status", "valuation_status"]:
        lines.append(f"- {key}: `{coverage.get(key, UNKNOWN)}`")
    lines.append("")
    lines.append("## Blockers / Unknowns")
    blockers = packet.get("blockers_unknowns") if isinstance(packet.get("blockers_unknowns"), list) else []
    if blockers:
        for item in blockers:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "")
            reason = str(item.get("reason_code") or UNKNOWN)
            value = item.get("value", UNKNOWN)
            lines.append(f"- `{field}`: value=`{_render_value(value)}` reason=`{reason}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Derived From")
    derived_from = packet.get("derived_from") if isinstance(packet.get("derived_from"), list) else []
    for item in derived_from:
        lines.append(f"- `{str(item)}`")
    return "\n".join(lines).rstrip() + "\n"


def _dossier_pack_dir(universe_run_id: str, batch_run_id: str, out_dir: Path | None = None) -> Path:
    if out_dir is not None:
        return out_dir
    cfg = get_config()
    return cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "dossier_pack"


def load_global_shortlist(universe_run_id: str, batch_run_id: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_shortlist.json"
    payload = _safe_json(path)
    if not payload:
        raise ValueError(f"Missing global_shortlist.json for universe_run_id={universe_run_id} batch_run_id={batch_run_id}")
    return {
        "global_shortlist_path": str(path),
        "payload": payload,
    }


def build_candidate_packet(row: dict[str, Any], rank_global: int) -> dict[str, Any]:
    value_gate_status = str(row.get("value_gate_status") or UNKNOWN)
    value_gate_reasons = [str(reason) for reason in (row.get("value_gate_reasons") or []) if str(reason).strip()]

    implied_return_base = row.get("implied_return_base", UNKNOWN)
    intrinsic_per_share_base = row.get("intrinsic_per_share_base", UNKNOWN)
    mos_epv = row.get("mos_epv", UNKNOWN)
    mos_netnet = row.get("mos_netnet", UNKNOWN)
    owner_yield_ev = row.get("owner_earnings_yield_ev_3y", UNKNOWN)
    fcf_yield_ev = row.get("fcf_yield_ev_3y", UNKNOWN)

    price_reason_code = str(row.get("price_reason_code") or UNKNOWN)
    valuation_reason_code = str(row.get("valuation_reason_code") or UNKNOWN)
    shares_reason_code = str(row.get("shares_reason_code") or UNKNOWN)
    fcf_reason_code = str(row.get("fcf_reason_code") or UNKNOWN)
    facts_reason_code = str(row.get("facts_reason_code") or UNKNOWN)

    blockers_unknowns: list[dict[str, Any]] = []
    unknown_specs = [
        ("implied_return_base", implied_return_base, valuation_reason_code),
        ("intrinsic_per_share_base", intrinsic_per_share_base, valuation_reason_code),
        ("mos_epv", mos_epv, str(row.get("gd_primary_reason_code") or UNKNOWN)),
        ("mos_netnet", mos_netnet, str(row.get("gd_primary_reason_code") or UNKNOWN)),
        ("owner_earnings_yield_ev_3y", owner_yield_ev, str(row.get("yield_reason_code") or UNKNOWN)),
        ("fcf_yield_ev_3y", fcf_yield_ev, str(row.get("yield_reason_code") or UNKNOWN)),
    ]
    for field, value, reason in unknown_specs:
        if _is_num(value):
            continue
        blockers_unknowns.append({"field": field, "value": value, "reason_code": reason})

    status_specs = [
        ("price_status", str(row.get("price_status") or UNKNOWN), price_reason_code),
        ("valuation_status", str(row.get("valuation_status") or UNKNOWN), valuation_reason_code),
        ("shares_status", str(row.get("shares_status") or UNKNOWN), shares_reason_code),
        ("fcf_status", str(row.get("fcf_status") or UNKNOWN), fcf_reason_code),
        ("facts_status", str(row.get("facts_status") or UNKNOWN), facts_reason_code),
    ]
    for field, status_value, reason in status_specs:
        if status_value.upper() == "OK":
            continue
        blockers_unknowns.append({"field": field, "value": status_value, "reason_code": reason})

    derived_from = [str(token) for token in (row.get("derived_from") or []) if str(token).strip()]
    return {
        "ticker": str(row.get("ticker") or ""),
        "rank_global": int(rank_global),
        "source_depth_runs": [
            {
                "run_id": str(item.get("run_id") or ""),
                "sector": str(item.get("sector") or ""),
                "as_of_date": str(item.get("as_of_date") or ""),
            }
            for item in (row.get("source_depth_runs") or [])
            if isinstance(item, dict)
        ],
        "value_gate_status": value_gate_status,
        "value_gate_reasons": value_gate_reasons,
        "price": {
            "current_price": row.get("current_price", UNKNOWN),
            "price_asof_used": str(row.get("price_asof_used") or UNKNOWN),
            "price_source_resolution": str(row.get("price_source_resolution") or UNKNOWN),
            "price_reason_code": price_reason_code,
        },
        "valuation": {
            "intrinsic_per_share_base": intrinsic_per_share_base if _is_num(intrinsic_per_share_base) else UNKNOWN,
            "implied_return_base": implied_return_base if _is_num(implied_return_base) else UNKNOWN,
            "valuation_reason_code": valuation_reason_code,
            "derived_from": [str(token) for token in (row.get("intrinsic_per_share_base_derived_from") or []) if str(token).strip()],
        },
        "graham_dodd": {
            "epv_per_share": row.get("epv_per_share", UNKNOWN),
            "mos_epv": mos_epv if _is_num(mos_epv) else UNKNOWN,
            "netnet_per_share": row.get("netnet_per_share", UNKNOWN),
            "mos_netnet": mos_netnet if _is_num(mos_netnet) else UNKNOWN,
            "gd_primary_reason_code": str(row.get("gd_primary_reason_code") or UNKNOWN),
            "derived_from": [str(token) for token in (row.get("mos_epv_derived_from") or []) if str(token).strip()]
            + [str(token) for token in (row.get("mos_netnet_derived_from") or []) if str(token).strip()],
        },
        "yields": {
            "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
            "yield_denominator_used": str(row.get("yield_denominator_used") or UNKNOWN),
            "owner_earnings_yield_ev_3y": owner_yield_ev if _is_num(owner_yield_ev) else UNKNOWN,
            "fcf_yield_ev_3y": fcf_yield_ev if _is_num(fcf_yield_ev) else UNKNOWN,
            "yield_reason_code": str(row.get("yield_reason_code") or UNKNOWN),
            "derived_from": [str(token) for token in (row.get("owner_earnings_yield_ev_3y_derived_from") or []) if str(token).strip()]
            + [str(token) for token in (row.get("fcf_yield_ev_3y_derived_from") or []) if str(token).strip()],
        },
        "leverage": {
            "net_debt_proxy": row.get("net_debt_proxy", UNKNOWN),
            "ev_status": str(row.get("ev_status") or UNKNOWN),
            "net_debt_reason_code": str(row.get("net_debt_reason_code") or UNKNOWN),
        },
        "quality_risk": {
            "quality_score": row.get("quality_score", UNKNOWN),
            "risk_penalty": row.get("risk_penalty", UNKNOWN),
            "risk_blocker_categories": [str(row.get("primary_blocker") or "NONE")] if str(row.get("primary_blocker") or "NONE") != "NONE" else [],
        },
        "coverage": {
            "price_status": str(row.get("price_status") or UNKNOWN),
            "shares_status": str(row.get("shares_status") or UNKNOWN),
            "fcf_status": str(row.get("fcf_status") or UNKNOWN),
            "facts_status": str(row.get("facts_status") or UNKNOWN),
            "valuation_status": str(row.get("valuation_status") or UNKNOWN),
        },
        "primary_blocker": str(row.get("primary_blocker") or "NONE"),
        "primary_blocker_reason_code": str(row.get("primary_blocker_reason_code") or str(row.get("primary_blocker") or "NONE")),
        "notes": str(row.get("notes") or ""),
        "blockers_unknowns": blockers_unknowns,
        "derived_from": derived_from,
    }


def write_watchlist_exports(
    universe_run_id: str,
    batch_run_id: str,
    rows: list[dict[str, Any]],
    out_dir: Path | None = None,
) -> dict[str, str]:
    pack_dir = _dossier_pack_dir(universe_run_id, batch_run_id, out_dir=out_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pack_dir / "watchlist.csv"
    json_path = pack_dir / "watchlist.json"

    csv_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        csv_rows.append(
            {
                "ticker": str(row.get("ticker") or ""),
                "rank": int(idx),
                "value_gate_status": str(row.get("value_gate_status") or UNKNOWN),
                "implied_return_base": row.get("implied_return_base", UNKNOWN),
                "mos_epv": row.get("mos_epv", UNKNOWN),
                "mos_netnet": row.get("mos_netnet", UNKNOWN),
                "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
                "yield_denominator_used": str(row.get("yield_denominator_used") or UNKNOWN),
                "owner_earnings_yield_ev_3y": row.get("owner_earnings_yield_ev_3y", UNKNOWN),
                "fcf_yield_ev_3y": row.get("fcf_yield_ev_3y", UNKNOWN),
                "price_status": str(row.get("price_status") or UNKNOWN),
                "valuation_status": str(row.get("valuation_status") or UNKNOWN),
                "primary_blocker": str(row.get("primary_blocker") or "NONE"),
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_WATCHLIST_COLUMNS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    _json_write(
        json_path,
        {
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "columns": list(_WATCHLIST_COLUMNS),
            "rows": csv_rows,
            "generated_at": utc_now_iso(),
        },
    )
    return {
        "watchlist_csv_path": str(csv_path),
        "watchlist_json_path": str(json_path),
    }


def write_dossier_pack(
    universe_run_id: str,
    batch_run_id: str,
    top_n: int,
    policy: str,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    write_depth_batch_rollup(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        top_n=max(1, int(top_n)),
        policy=policy,
    )
    shortlist_payload = load_global_shortlist(universe_run_id, batch_run_id)
    shortlist = shortlist_payload["payload"]
    rows = [row for row in (shortlist.get("rows") or []) if isinstance(row, dict)]
    rows = rows[: max(1, int(top_n))]

    pack_dir = _dossier_pack_dir(universe_run_id, batch_run_id, out_dir=out_dir)
    candidates_dir = pack_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    packet_paths: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        packet = build_candidate_packet(row, rank_global=idx)
        ticker = str(packet.get("ticker") or f"TICKER_{idx}")
        packet_dir = candidates_dir / ticker
        packet_dir.mkdir(parents=True, exist_ok=True)
        packet_json_path = packet_dir / "packet.json"
        packet_md_path = packet_dir / "packet.md"
        _json_write(packet_json_path, packet)
        packet_md_path.write_text(_packet_markdown(packet), encoding="utf-8")
        packet_paths.append(
            {
                "ticker": ticker,
                "rank_global": int(idx),
                "packet_json_path": str(packet_json_path),
                "packet_md_path": str(packet_md_path),
            }
        )
        packets.append(packet)

    watch_paths = write_watchlist_exports(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        rows=rows,
        out_dir=pack_dir,
    )

    unknown_counts = {
        "unknown_price_or_status": 0,
        "unknown_valuation": 0,
        "unknown_shares": 0,
        "unknown_fcf": 0,
        "unknown_facts": 0,
    }
    for packet in packets:
        coverage = packet.get("coverage") if isinstance(packet.get("coverage"), dict) else {}
        valuation = packet.get("valuation") if isinstance(packet.get("valuation"), dict) else {}
        price = packet.get("price") if isinstance(packet.get("price"), dict) else {}
        if str(coverage.get("price_status") or UNKNOWN).upper() != "OK" or not _is_num(price.get("current_price")):
            unknown_counts["unknown_price_or_status"] += 1
        if str(coverage.get("valuation_status") or UNKNOWN).upper() != "OK" or not _is_num(valuation.get("implied_return_base")):
            unknown_counts["unknown_valuation"] += 1
        if str(coverage.get("shares_status") or UNKNOWN).upper() != "OK":
            unknown_counts["unknown_shares"] += 1
        if str(coverage.get("fcf_status") or UNKNOWN).upper() != "OK":
            unknown_counts["unknown_fcf"] += 1
        if str(coverage.get("facts_status") or UNKNOWN).upper() != "OK":
            unknown_counts["unknown_facts"] += 1

    manifest_path = pack_dir / "dossier_pack_manifest.json"
    manifest_payload = {
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "policy": str(policy),
        "top_n": int(top_n),
        "candidate_count": len(rows),
        "global_shortlist_path": str(shortlist_payload.get("global_shortlist_path") or ""),
        "watchlist_csv_path": watch_paths["watchlist_csv_path"],
        "watchlist_json_path": watch_paths["watchlist_json_path"],
        "packet_paths": packet_paths,
        "unknown_counts": unknown_counts,
        "generated_at": utc_now_iso(),
    }
    _json_write(manifest_path, manifest_payload)

    return {
        "status": "OK",
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "dossier_pack_dir": str(pack_dir),
        "manifest_path": str(manifest_path),
        "watchlist_csv_path": watch_paths["watchlist_csv_path"],
        "watchlist_json_path": watch_paths["watchlist_json_path"],
        "candidate_count": len(rows),
        "unknown_counts": unknown_counts,
        "top_rows": [
            {
                "ticker": str(row.get("ticker") or ""),
                "rank": idx,
                "implied_return_base": row.get("implied_return_base", UNKNOWN),
                "mos_epv": row.get("mos_epv", UNKNOWN),
                "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
            }
            for idx, row in enumerate(rows[:10], start=1)
        ],
    }


def open_dossier_pack(universe_run_id: str, batch_run_id: str) -> dict[str, Any]:
    pack_dir = _dossier_pack_dir(universe_run_id, batch_run_id)
    manifest_path = pack_dir / "dossier_pack_manifest.json"
    manifest = _safe_json(manifest_path)
    if not manifest:
        return {
            "status": "MISSING",
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "dossier_pack_dir": str(pack_dir),
            "manifest_path": str(manifest_path),
        }
    watchlist_json_path = Path(str(manifest.get("watchlist_json_path") or ""))
    watchlist_payload = _safe_json(watchlist_json_path) if str(watchlist_json_path).strip() else {}
    watch_rows = [row for row in (watchlist_payload.get("rows") or []) if isinstance(row, dict)]
    top_rows = watch_rows[:10]
    return {
        "status": "OK",
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "dossier_pack_dir": str(pack_dir),
        "manifest_path": str(manifest_path),
        "watchlist_csv_path": str(manifest.get("watchlist_csv_path") or ""),
        "watchlist_json_path": str(manifest.get("watchlist_json_path") or ""),
        "top_10": top_rows,
        "unknown_counts": manifest.get("unknown_counts") if isinstance(manifest.get("unknown_counts"), dict) else {},
        "candidate_count": int(manifest.get("candidate_count") or len(watch_rows)),
    }
