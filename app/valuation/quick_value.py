"""One-ticker deterministic valuation from free data (``ivi value``).

Composes existing pipeline steps and adds no valuation logic of its own:
SEC companyfacts fetch -> normalized facts -> the valuation writer (with the
issuer CIK, so evidenced-zero proofs can run) -> a live quote from the
configured free price chain. No LLM and no paid key is needed.

The rows this writes are research output, not decision-eligible valuations:
they are not bound to an audited run artifact, and the decision surfaces
(``analyze``, the web read model) keep refusing them. The summary says so.
"""

from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

from app.config import AppConfig, get_config
from app.db import get_db, init_db
from app.valuation.anchor_policy import published_dcf_base

# Per-share methods shown in the table, in display order.
VALUE_METHODS: tuple[tuple[str, str], ...] = (
    ("dcf", "DCF (base)"),
    ("epv", "Earnings power (EPV)"),
    ("graham", "Graham number"),
    ("fcf_yield", "FCF yield"),
    ("ev_ebit", "EV/EBIT"),
    ("ncav", "Net current asset value"),
    ("tangible_floor", "Tangible floor"),
)


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _latest_rows(conn: Any, ticker: str, as_of_date: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT method, outputs_json, quality_gate_verdict
        FROM valuations
        WHERE ticker = ? AND as_of_date = ?
        ORDER BY created_at DESC, id DESC
        """,
        (ticker, as_of_date),
    ).fetchall()
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        method = str(row["method"])
        if method in latest:
            continue
        try:
            outputs = json.loads(row["outputs_json"] or "{}")
        except json.JSONDecodeError:
            outputs = {}
        latest[method] = {
            "outputs": outputs if isinstance(outputs, dict) else {},
            "gate": row["quality_gate_verdict"],
        }
    return latest


def _method_row(
    method: str,
    label: str,
    outputs: dict[str, Any] | None,
    price: float | None,
    scorecard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outputs = outputs or {}
    flags = [str(flag) for flag in (outputs.get("flags") or [])]
    if method == "dcf":
        # The DCF the platform stands behind: after a non-recurring revenue
        # spike that is the durable (spike-corrected) base in the scorecard,
        # not the raw ``dcf`` row. The scorecard, dossier and web page all
        # resolve it through anchor_policy.published_dcf_base; this row did not.
        scorecard = scorecard or {}
        value = published_dcf_base(scorecard.get("pricing_zone_detail"), outputs.get("base"))
        low, high = _num(outputs.get("low")), _num(outputs.get("high"))
        durable = (scorecard.get("quality_context") or {}).get("dcf_durable") or {}
        if value is not None and value != _num(outputs.get("base")):
            low, high = _num(durable.get("low")), _num(durable.get("high"))
            flags.append("DCF_DURABLE_BASE_PUBLISHED")
    else:
        value = _num(outputs.get("value_per_share"))
        low = high = None
    mos = None
    if value is not None and value > 0 and price is not None:
        mos = (value - price) / value
    status = str(outputs.get("status") or ("OK" if value is not None else "NOT_WRITTEN"))
    return {
        "method": method,
        "label": label,
        "status": status,
        "value_per_share": value,
        "low": low,
        "high": high,
        "margin_of_safety": mos,
        "flags": flags,
    }


def run_value(
    ticker: str,
    *,
    as_of_date: str | None = None,
    years_back: int = 10,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Fetch, compute and summarize one ticker. Never raises for data gaps."""

    from app.ingest.facts_writer import ensure_all_facts
    from app.market.company_facts_provider import fetch_company_facts
    from app.valuation.facts import resolve_cik_for_ticker
    from app.valuation.price_provider import get_default_provider
    from app.valuation.valuation_writer import ensure_valuation

    cfg = cfg or get_config()
    upper = str(ticker or "").strip().upper()
    as_of = as_of_date or date.today().isoformat()
    summary: dict[str, Any] = {
        "ticker": upper,
        "as_of_date": as_of,
        "status": "FAILED",
        "decision_eligible": False,
        "cik": None,
        "facts": None,
        "price": None,
        "scorecard": None,
        "methods": [],
    }
    init_db(cfg)

    cik = resolve_cik_for_ticker(upper, cfg=cfg)
    if not cik:
        from app.universe.ticker_cik_map import cached_mapping_path
        from app.util.http import network_disabled

        if network_disabled(cfg) and not cached_mapping_path().exists():
            summary["reason"] = "OFFLINE_NOT_CACHED"
            summary["detail"] = (
                f"Offline (VOE_NET_PROVIDER=disabled) and the SEC ticker list is not cached, "
                f"so {upper} cannot be looked up."
            )
            return summary
        summary["reason"] = "CIK_NOT_FOUND"
        summary["detail"] = f"No SEC CIK is known for {upper} (SEC company_tickers map)."
        return summary
    summary["cik"] = cik

    # A fresh install has no registrant table yet; without the SIC the REIT rule and
    # the bank/insurer routing cannot fire. Fetch it once (a no-op when on file or offline).
    from app.util.issuer_classification import ensure_registrant_sic

    ensure_registrant_sic(cik=cik, cfg=cfg)

    facts = fetch_company_facts(cik, cfg=cfg)
    summary["facts"] = {
        key: facts.get(key)
        for key in ("status", "reason_code", "reason_detail", "source_url", "size_bytes", "retrieved_at")
    }
    if str(facts.get("status")) != "OK":
        summary["reason"] = f"COMPANYFACTS_{facts.get('reason_code')}"
        summary["detail"] = str(facts.get("reason_detail") or "")
        return summary
    # Normalizes the payload just cached above; no second download. The
    # ingest window counts back from today, so widen it to reach years_back
    # before an older as-of date.
    try:
        as_of_year = date.fromisoformat(as_of[:10]).year
    except ValueError:
        as_of_year = date.today().year
    ensure_all_facts(upper, years_back=years_back + max(0, date.today().year - as_of_year))

    # Every method needs at least one annual report. A registrant with none —
    # a new holding company that succeeded an older registrant (the SEC ticker
    # map moves the ticker to the new CIK, whose XBRL history starts at its
    # first 10-Q), or a recent listing — is refused with that reason instead
    # of a table of unrelated method gaps. The predecessor's history is not
    # borrowed: SEC data does not link the two CIKs.
    with get_db(cfg=cfg) as conn:
        annual_rows = conn.execute(
            "SELECT COUNT(*) FROM companyfacts_facts WHERE ticker = ? AND period_type = 'FY' "
            "AND filed_date IS NOT NULL AND filed_date != '' AND filed_date <= ?",
            (upper, as_of),
        ).fetchone()[0]
    if not annual_rows:
        summary["reason"] = "NO_ANNUAL_REPORT"
        summary["detail"] = (
            f"SEC companyfacts for CIK {cik} holds no annual report (10-K, 20-F or 40-F) filed by "
            f"{as_of}. A newly formed registrant, such as a holding company that replaced an "
            "older registrant in a reorganization, starts a new filing history under its new "
            "CIK, and SEC data does not link it to the predecessor's; re-run once it has filed "
            "an annual report."
        )
        return summary

    provider = get_default_provider(cfg)
    ensure_valuation(
        upper,
        as_of,
        provider,
        force_refresh=True,
        cfg=cfg,
        raise_on_error=True,
        issuer_cik=cik,
        require_filed_asof=True,
    )

    with get_db(cfg=cfg) as conn:
        latest = _latest_rows(conn, upper, as_of)
    scorecard = (latest.get("scorecard") or {}).get("outputs") or {}
    pzd = scorecard.get("pricing_zone_detail") or {}
    price = _num(pzd.get("current_price"))
    if price is not None and price <= 0:
        price = None
    summary["price"] = {
        "value": price,
        "source": pzd.get("current_price_source"),
        "quote_date": pzd.get("current_price_as_of_date"),
        "basis": pzd.get("current_price_basis"),
        "status": "OK" if price is not None else "UNAVAILABLE",
        "reason": None,
    }
    if price is None:
        # Ask the same provider why (served from its cache when it has one).
        quote = provider.get_quote(upper, as_of)
        summary["price"]["source"] = quote.provider
        summary["price"]["reason"] = quote.provenance.get("reason") or quote.status
    summary["scorecard"] = {
        "signal": scorecard.get("signal"),
        "pricing_zone": scorecard.get("pricing_zone"),
        "reason": pzd.get("reason") or pzd.get("gate_reason"),
        "net_debt": pzd.get("net_debt"),
        "gate": (latest.get("scorecard") or {}).get("gate"),
    }
    summary["methods"] = [
        _method_row(method, label, (latest.get(method) or {}).get("outputs"), price, scorecard)
        for method, label in VALUE_METHODS
    ]
    valued = [row for row in summary["methods"] if row["value_per_share"] is not None]
    if scorecard.get("signal") == "VALUATION_BLOCKED":
        summary["reason"] = "VALUATION_BLOCKED"
        summary["detail"] = str(pzd.get("gate_reason") or "pre-valuation quality gate")
        return summary
    if not latest or not valued:
        summary["reason"] = "NO_METHOD_VALUE"
        summary["detail"] = "The valuation writer produced no per-share value for this issuer."
        return summary
    summary["status"] = "OK" if price is not None else "PARTIAL_NO_PRICE"
    return summary


def _money(value: float | None) -> str:
    return f"{value:,.2f}" if value is not None else "n/a"


def render_value_summary(summary: dict[str, Any]) -> str:
    cik = summary.get("cik") or "unknown"
    lines = [f"{summary['ticker']}  as of {summary['as_of_date']}  (CIK {cik})"]
    if summary.get("reason") and not summary.get("methods"):
        lines.append(f"FAILED: {summary['reason']} - {summary.get('detail') or ''}".rstrip(" -"))
        return "\n".join(lines)
    facts = summary.get("facts") or {}
    lines.append(
        f"Fundamentals: SEC companyfacts ({facts.get('reason_code')}, "
        f"{(facts.get('size_bytes') or 0) / 1e6:.1f} MB)"
    )
    price = summary.get("price") or {}
    if price.get("status") == "OK":
        lines.append(
            f"Price: {_money(price.get('value'))} ({price.get('source')}, close of "
            f"{price.get('quote_date')}, basis {price.get('basis')})"
        )
    else:
        lines.append(
            f"Price: UNAVAILABLE ({price.get('reason') or 'no quote'}); margins of safety not computed"
        )
    lines.append("")
    if summary.get("reason") == "VALUATION_BLOCKED":
        lines.append(
            f"Valuation blocked by the pre-valuation quality gate: {summary.get('detail')}. "
            "No method values were written."
        )
        return "\n".join(lines)
    lines.append(f"{'Method':<26}{'Value/share':>12}{'Margin of safety':>18}  Status")
    for row in summary.get("methods") or []:
        value = _money(row["value_per_share"])
        mos = f"{row['margin_of_safety']:+.0%}" if row["margin_of_safety"] is not None else "n/a"
        status = row["status"]
        if row["value_per_share"] is None and row["flags"]:
            status = f"{status} ({', '.join(row['flags'][:2])})"
        lines.append(f"{row['label']:<26}{value:>12}{mos:>18}  {status}")
        if row["low"] is not None and row["high"] is not None:
            span = f"{_money(row['low'])} - {_money(row['high'])}"
            lines.append(f"{'  range (low - high)':<26}{span:>30}")
    scorecard = summary.get("scorecard") or {}
    lines.append("")
    lines.append(
        f"Scorecard: {scorecard.get('signal')} / zone {scorecard.get('pricing_zone')}"
        + (f" ({scorecard['reason']})" if scorecard.get("reason") else "")
    )
    lines.append(
        "Margin of safety = (value - price) / value; negative means the price is above that estimate."
    )
    lines.append(
        "Research output only: these rows are not bound to an audited run artifact, so decision "
        "surfaces (ivi analyze, the web UI) will not treat them as decision-eligible."
    )
    return "\n".join(lines)
