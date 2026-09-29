"""Markdown rendering for autonomous sector financial run artifacts."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.autonomous.financial_integrity import MARKET_CAP_UNIT_USD_MILLIONS
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
)
from app.watchlist.margin_of_safety import compute_buy_target
from app.watchlist.volatility import realized_volatility
from app.util.financial_data_access import companyfacts_rows


BASE_RETURN_HURDLE = 0.12
MEMO_BODY_DEGRADED_STATE = "SECTOR_MEMO_BODY_LLM_UNAVAILABLE"


def _na(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "N/A"
    return str(value) if str(value) else "N/A"


def _fmt_money(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"${float(value):,.2f}"
    return "N/A"


def _fmt_market_cap(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "N/A"
    amount_mm = float(value)
    if abs(amount_mm) >= 1_000:
        return f"${amount_mm / 1_000:,.1f}B"
    return f"${amount_mm:,.1f}M"


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100:.1f}%"
    return "N/A"


def _fmt_num(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):,.2f}"
    return "N/A"


def _fmt_multiple(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):,.1f}x"
    return "N/A"


def _fmt_days(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):,.1f} days"
    return "N/A"


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _join(items: list[Any] | None) -> str:
    return ", ".join(str(item) for item in items or []) if items else "N/A"


def _line_break(text: Any) -> str:
    return str(text or "").replace("\n", "<br>")


def _cell(text: Any) -> str:
    return _line_break(text).replace("|", "\\|")


def _db_path() -> Path | None:
    try:
        from app.config import get_config

        return Path(get_config().db_path)
    except Exception:
        return None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _fetch_company_identity(ticker: str) -> dict[str, Any]:
    path = _db_path()
    if path is None or not path.exists():
        return {}
    try:
        with sqlite3.connect(str(path)) as conn:
            conn.row_factory = sqlite3.Row
            if not {"ticker", "name", "cik", "homepage_url", "created_at"} <= _table_columns(
                conn, "universe_members"
            ):
                return {}
            row = conn.execute(
                """
                SELECT cik, name, homepage_url
                FROM universe_members
                WHERE ticker = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (ticker.upper(),),
            ).fetchone()
            return dict(row) if row else {}
    except sqlite3.Error:
        return {}


def _fetch_market_cap(ticker: str) -> float | None:
    path = _db_path()
    if path is None or not path.exists():
        return None
    try:
        with sqlite3.connect(str(path)) as conn:
            conn.row_factory = sqlite3.Row
            if not {
                "ticker",
                "market_cap",
                "market_cap_status",
                "effective_as_of_date",
            } <= _table_columns(conn, "market_caps"):
                return None
            row = conn.execute(
                """
                SELECT market_cap
                FROM market_caps
                WHERE ticker = ?
                  AND market_cap_status = 'OK'
                  AND market_cap IS NOT NULL
                ORDER BY effective_as_of_date DESC
                LIMIT 1
                """,
                (ticker.upper(),),
            ).fetchone()
            value = row["market_cap"] if row else None
            return float(value) if isinstance(value, (int, float)) else None
    except sqlite3.Error:
        return None


def _latest_shares_outstanding(ticker: str) -> float | None:
    path = _db_path()
    if path is None or not path.exists():
        return None
    try:
        with sqlite3.connect(str(path)) as conn:
            conn.row_factory = sqlite3.Row
            required = {"ticker", "fiscal_year", "period_type", "line_item", "value"}
            if not required <= _table_columns(conn, "companyfacts_facts"):
                return None
            row = conn.execute(
                """
                SELECT value
                FROM companyfacts_facts
                WHERE ticker = ?
                  AND period_type = 'FY'
                  AND line_item = 'shares_outstanding'
                  AND value IS NOT NULL
                ORDER BY fiscal_year DESC
                LIMIT 1
                """,
                (ticker.upper(),),
            ).fetchone()
            value = row["value"] if row else None
            return float(value) if isinstance(value, (int, float)) else None
    except sqlite3.Error:
        return None


def _fetch_5y_snapshot(
    ticker: str,
    *,
    as_of_date: str,
) -> list[dict[str, Any]]:
    path = _db_path()
    if path is None or not path.exists():
        return []
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
        with sqlite3.connect(str(path)) as conn:
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
            if not required <= _table_columns(conn, "companyfacts_facts"):
                return []
            rows = companyfacts_rows(
                conn,
                ticker.upper(),
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
    except sqlite3.Error:
        return []
    by_year: dict[int, dict[str, float]] = {}
    for row in rows:
        year = row["fiscal_year"]
        if not isinstance(year, int):
            continue
        if str(row["period_end"] or "") > str(row["filed_date"] or ""):
            continue
        by_year.setdefault(year, {})[str(row["line_item"])] = row["value"]
    snapshot: list[dict[str, Any]] = []
    for year in sorted(by_year.keys(), reverse=True)[:5]:
        values = by_year[year]
        revenue = values.get("revenue")
        operating_income = values.get("operating_income")
        cfo = values.get("cfo")
        capex = values.get("capex")
        fcf = (
            (cfo - abs(capex))
            if isinstance(cfo, (int, float)) and isinstance(capex, (int, float))
            else None
        )
        op_margin = (
            operating_income / revenue
            if isinstance(revenue, (int, float))
            and revenue > 0
            and isinstance(operating_income, (int, float))
            else None
        )
        snapshot.append(
            {
                "fy": year,
                "revenue": revenue,
                "op_margin": op_margin,
                "cfo": cfo,
                "fcf": fcf,
                "cash": values.get("cash"),
                "debt": values.get("total_debt"),
                "shares": values.get("shares_outstanding"),
                "sbc": values.get("sbc"),
            }
        )
    return snapshot


def _alternate_audit_results(results: list[dict[str, Any]] | None) -> str:
    formatted: list[str] = []
    for item in results or []:
        formatted.append(
            f"{_na(item.get('ticker'))}: {_na(item.get('audit_status'))} "
            f"(source={_na(item.get('source'))}; base={_fmt_pct(item.get('best_base_annualized_return'))}; "
            f"blockers={_join(item.get('hard_blockers') or [])}; caps={_join(item.get('confidence_caps') or [])})"
        )
    return "; ".join(formatted) if formatted else "N/A"


def _maturity_schedule(rows: list[dict[str, Any]] | None) -> str:
    formatted: list[str] = []
    for row in rows or []:
        label = _na(
            row.get("maturity_date")
            or (row.get("year") if row.get("year") is not None else row.get("period"))
        )
        amount = row.get("amount")
        unit = _na(row.get("unit"))
        formatted.append(
            f"{label}: {float(amount):g} {unit}" if isinstance(amount, (int, float)) else label
        )
    return "; ".join(formatted) if formatted else "N/A"


def _covenant_terms(rows: list[dict[str, Any]] | None) -> str:
    formatted: list[str] = []
    for row in rows or []:
        metric = _na(row.get("metric"))
        condition = _na(row.get("condition"))
        threshold = row.get("threshold")
        unit = _na(row.get("unit"))
        formatted.append(
            f"{metric}: {condition} {float(threshold):g} {unit}"
            if isinstance(threshold, (int, float))
            else f"{metric}: {condition}"
        )
    return "; ".join(formatted) if formatted else "N/A"


def _guardrail_outcome_applies(artifact: AutonomousSectorFinancialRunArtifact) -> bool:
    verdict = str(artifact.final_verdict or "").upper()
    if verdict not in {"NO_SELECTION", "NO_WINNER"}:
        return False
    decision = artifact.final_decision
    return bool(
        artifact.degraded_states
        or (decision and decision.selection_blockers)
        or artifact.no_selection_reason
    )


def _selection_audit_applies(artifact: AutonomousSectorFinancialRunArtifact) -> bool:
    return isinstance(artifact.selection_audit, dict) and bool(artifact.selection_audit)


def _scenario_summary_rows(scenarios: list[SectorExpectedReturnScenario]) -> list[str]:
    by_key: dict[tuple[str, int], dict[str, SectorExpectedReturnScenario]] = defaultdict(dict)
    for scenario in scenarios:
        by_key[(scenario.ticker, scenario.horizon_years)][scenario.scenario_name.lower()] = scenario

    rows = [
        "| Ticker | Horizon | Downside Return | Base Return | Upside Return | Downside Value | Base Value | Upside Value |",
        "|--------|---------|-----------------|-------------|---------------|----------------|------------|--------------|",
    ]
    for (ticker, horizon), cases in sorted(by_key.items()):
        downside = cases.get("downside")
        base = cases.get("base")
        upside = cases.get("upside")
        rows.append(
            "| "
            f"{ticker} | {horizon}Y | "
            f"{_fmt_pct(downside.annualized_return if downside else None)} | "
            f"{_fmt_pct(base.annualized_return if base else None)} | "
            f"{_fmt_pct(upside.annualized_return if upside else None)} | "
            f"{_fmt_money(downside.estimated_future_value_per_share if downside else None)} | "
            f"{_fmt_money(base.estimated_future_value_per_share if base else None)} | "
            f"{_fmt_money(upside.estimated_future_value_per_share if upside else None)} |"
        )
    return rows


def _packet_by_ticker(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> dict[str, SectorCompanyFinancialPacket]:
    return {packet.ticker.upper(): packet for packet in artifact.company_packets}


def _ranking_by_ticker(artifact: AutonomousSectorFinancialRunArtifact) -> dict[str, dict[str, Any]]:
    return {str(item.get("ticker") or "").upper(): item for item in artifact.relative_ranking}


def _candidate_tickers(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    limit: int | None = None,
) -> list[str]:
    tickers: list[str] = []
    selection = artifact.candidate_selection or {}
    for ticker in selection.get("selected_tickers") or []:
        token = str(ticker or "").upper()
        if token and token not in tickers:
            tickers.append(token)
    for item in artifact.relative_ranking:
        token = str(item.get("ticker") or "").upper()
        if token and token not in tickers:
            tickers.append(token)
    for packet in artifact.company_packets:
        token = packet.ticker.upper()
        if token not in tickers:
            tickers.append(token)
    return tickers[:limit] if limit is not None else tickers


def _company_name(packet: SectorCompanyFinancialPacket | None, ticker: str) -> str:
    if packet:
        for bucket in (packet.valuation, packet.business_quality, packet.accounting_quality):
            for key in ("company_name", "name"):
                value = bucket.get(key) if isinstance(bucket, dict) else None
                if value:
                    return str(value)
    identity = _fetch_company_identity(ticker)
    return str(identity.get("name") or ticker)


def _market_cap(packet: SectorCompanyFinancialPacket | None, ticker: str) -> float | None:
    _ = ticker
    if (
        packet is not None
        and packet.market_cap_unit == MARKET_CAP_UNIT_USD_MILLIONS
        and isinstance(packet.market_cap_mm, (int, float))
        and not isinstance(packet.market_cap_mm, bool)
    ):
        return float(packet.market_cap_mm)
    return None


def _valuation_anchor(packet: SectorCompanyFinancialPacket | None) -> float | None:
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    value = valuation.get("valuation_anchor")
    # Guard value > 0 to match store._valuation_anchor — a non-positive
    # anchor must yield None (n/a) so the memo never renders a bogus negative/zero
    # buy-price target for a name the store would have dropped.
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _anchor_method(packet: SectorCompanyFinancialPacket | None) -> str:
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    return _na(valuation.get("anchor_method"))


def _intrinsic_bound(packet: SectorCompanyFinancialPacket | None, key: str) -> float | None:
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    value = valuation.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _buy_price_target(
    packet: SectorCompanyFinancialPacket | None,
    *,
    conviction_grade: str | None = None,
    confidence: str | None = None,
    db_path: str | Path | None = None,
) -> float | None:
    # Single source of truth — delegate to the same per-name discount the
    # watchlist store persists, so the memo BUY-price line matches the DB value.
    # Explicit LLM-supplied targets still win. Conviction grade may be absent at
    # render time; when None the neutral WATCHLIST_ONLY base is used.
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    for key in ("buy_below_price", "buy_price_target", "buy_below"):
        value = valuation.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            return float(value)
    anchor = _valuation_anchor(packet)
    # Consult live realized volatility exactly as the store does (it returns
    # None in production today, but threading it keeps the memo BUY-price equal to
    # the persisted DB value once the volatility weight is raised above 0).
    rv = realized_volatility(packet.ticker, db_path=db_path) if packet is not None else None
    return compute_buy_target(
        anchor,
        conviction_grade=conviction_grade,
        confidence=confidence,
        intrinsic_low=_intrinsic_bound(packet, "intrinsic_range_low"),
        intrinsic_high=_intrinsic_bound(packet, "intrinsic_range_high"),
        realized_volatility=rv,
    )


def _buy_target_grade_confidence(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
) -> tuple[str | None, str | None]:
    # Resolve the SAME (conviction_grade, confidence) the watchlist
    # store persists for this ticker, so the memo BUY-price line visibly matches
    # the stored DB buy_price_target. Delegating to the store's canonical
    # _candidate_verdict/_candidate_confidence (not the memo's display-only
    # _conviction_grade, whose vocabulary differs) guarantees identical inputs
    # to compute_buy_target -> identical output. No circular import: store.py
    # does not import sector_report.
    from app.watchlist import store

    ranking = _ranking_by_ticker(artifact).get(ticker.upper())
    grade = store._candidate_verdict(artifact, ticker, ranking)
    confidence = store._candidate_confidence(artifact, ticker, ranking)
    return grade, confidence


def _distance_from_buy(
    packet: SectorCompanyFinancialPacket | None,
    *,
    conviction_grade: str | None = None,
    confidence: str | None = None,
) -> float | None:
    if packet is None or packet.current_price is None:
        return None
    buy_price = _buy_price_target(packet, conviction_grade=conviction_grade, confidence=confidence)
    if buy_price is None or buy_price <= 0:
        return None
    return (float(packet.current_price) - buy_price) / buy_price


def _returns_on_capital(packet: SectorCompanyFinancialPacket | None) -> dict[str, Any]:
    return (
        packet.returns_on_capital if packet and isinstance(packet.returns_on_capital, dict) else {}
    )


def _business_quality(packet: SectorCompanyFinancialPacket | None) -> dict[str, Any]:
    return packet.business_quality if packet and isinstance(packet.business_quality, dict) else {}


def _cash_conversion(packet: SectorCompanyFinancialPacket | None) -> dict[str, Any]:
    return packet.cash_conversion if packet and isinstance(packet.cash_conversion, dict) else {}


def _capital_allocation(packet: SectorCompanyFinancialPacket | None) -> dict[str, Any]:
    return (
        packet.capital_allocation if packet and isinstance(packet.capital_allocation, dict) else {}
    )


def _roic(packet: SectorCompanyFinancialPacket | None) -> float | None:
    value = _returns_on_capital(packet).get("roic")
    return float(value) if isinstance(value, (int, float)) else None


def _roic_wacc_spread(packet: SectorCompanyFinancialPacket | None) -> float | None:
    value = _returns_on_capital(packet).get("roic_wacc_spread")
    return float(value) if isinstance(value, (int, float)) else None


def _fcf_yield(packet: SectorCompanyFinancialPacket | None) -> float | None:
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    value = valuation.get("fcf_yield")
    return float(value) if isinstance(value, (int, float)) else None


def _peer_relative_metric(
    packet: SectorCompanyFinancialPacket | None, metric_key: str
) -> dict[str, Any]:
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    peer = valuation.get("peer_relative_valuation")
    if not isinstance(peer, dict):
        return {}
    metrics = peer.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    item = metrics.get(metric_key)
    return item if isinstance(item, dict) else {}


def _render_roic_trajectory_line(packet: SectorCompanyFinancialPacket | None) -> str:
    trajectory = _returns_on_capital(packet).get("roic_trajectory_5y")
    if not isinstance(trajectory, list) or not trajectory:
        reasons = _returns_on_capital(packet).get("roic_not_computable_reasons")
        return (
            f"5-year ROIC trajectory: N/A ({_join(reasons if isinstance(reasons, list) else [])})."
        )
    values: list[str] = []
    missing_reasons: list[str] = []
    for row in trajectory:
        if not isinstance(row, dict):
            continue
        year = _na(row.get("fiscal_year"))
        roic_value = row.get("roic")
        if isinstance(roic_value, (int, float)):
            values.append(f"{year}: {_fmt_pct(roic_value)}")
        else:
            values.append(f"{year}: N/A")
            missing_reasons.extend(str(item) for item in (row.get("not_computable_reasons") or []))
    suffix = (
        f" Missing reasons: {_join_humanized(list(dict.fromkeys(missing_reasons)))}."
        if missing_reasons
        else ""
    )
    return f"5-year ROIC trajectory: {'; '.join(values) if values else 'N/A'}.{suffix}"


def _render_pct_trajectory(
    trajectory: Any,
    *,
    value_key: str,
    reasons_key: str,
    empty_reasons: Any = None,
) -> str:
    if not isinstance(trajectory, list) or not trajectory:
        return f"N/A ({_join_humanized(empty_reasons if isinstance(empty_reasons, list) else [])})"
    values: list[str] = []
    missing_reasons: list[str] = []
    for row in trajectory:
        if not isinstance(row, dict):
            continue
        year = _na(row.get("fiscal_year"))
        value = row.get(value_key)
        if isinstance(value, (int, float)):
            values.append(f"{year}: {_fmt_pct(value)}")
        else:
            values.append(f"{year}: N/A")
            missing_reasons.extend(str(item) for item in (row.get(reasons_key) or []))
    suffix = (
        f" Missing reasons: {_join_humanized(list(dict.fromkeys(missing_reasons)))}."
        if missing_reasons
        else ""
    )
    return f"{'; '.join(values) if values else 'N/A'}.{suffix}"


def _render_days_trajectory(
    trajectory: Any,
    *,
    value_key: str,
    reasons_key: str,
    empty_reasons: Any = None,
) -> str:
    if not isinstance(trajectory, list) or not trajectory:
        return f"N/A ({_join_humanized(empty_reasons if isinstance(empty_reasons, list) else [])})"
    values: list[str] = []
    missing_reasons: list[str] = []
    for row in trajectory:
        if not isinstance(row, dict):
            continue
        year = _na(row.get("fiscal_year"))
        value = row.get(value_key)
        if isinstance(value, (int, float)):
            values.append(f"{year}: {_fmt_days(value)}")
        else:
            values.append(f"{year}: N/A")
            missing_reasons.extend(str(item) for item in (row.get(reasons_key) or []))
    suffix = (
        f" Missing reasons: {_join_humanized(list(dict.fromkeys(missing_reasons)))}."
        if missing_reasons
        else ""
    )
    return f"{'; '.join(values) if values else 'N/A'}.{suffix}"


def _render_gross_margin_line(packet: SectorCompanyFinancialPacket | None) -> str:
    quality = _business_quality(packet)
    return (
        f"Gross margin: latest {_fmt_pct(quality.get('gross_margin'))}; "
        "5-year trajectory "
        f"{_render_pct_trajectory(quality.get('gross_margin_trajectory_5y'), value_key='gross_margin', reasons_key='not_computable_reasons', empty_reasons=quality.get('gross_margin_not_computable_reasons'))}"
    )


def _render_cash_conversion_cycle_line(packet: SectorCompanyFinancialPacket | None) -> str:
    conversion = _cash_conversion(packet)
    return (
        f"Cash conversion cycle: latest {_fmt_days(conversion.get('cash_conversion_cycle'))}; "
        f"DSO {_fmt_days(conversion.get('days_sales_outstanding'))}; "
        f"DIO {_fmt_days(conversion.get('days_inventory_outstanding'))}; "
        f"DPO {_fmt_days(conversion.get('days_payable_outstanding'))}; "
        "5-year trajectory "
        f"{_render_days_trajectory(conversion.get('cash_conversion_cycle_trajectory_5y'), value_key='cash_conversion_cycle', reasons_key='not_computable_reasons', empty_reasons=conversion.get('cash_conversion_cycle_not_computable_reasons'))}"
    )


def _cash_conversion_missing_reasons(
    conversion: dict[str, Any],
    *,
    default: str,
    relevant: set[str] | None = None,
) -> list[str]:
    reasons = conversion.get("cash_conversion_cycle_not_computable_reasons")
    values = (
        [str(item) for item in reasons if str(item).strip()] if isinstance(reasons, list) else []
    )
    if relevant:
        values = [item for item in values if item in relevant]
    return list(dict.fromkeys(values or [default]))


def _render_free_cash_flow_margin_line(packet: SectorCompanyFinancialPacket | None) -> str:
    conversion = _cash_conversion(packet)
    margin = _num(conversion.get("fcf_margin"))
    if margin is None:
        reasons = conversion.get("fcf_margin_not_computable_reasons")
        if not isinstance(reasons, list) or not reasons:
            reasons = ["FCF_MARGIN_NOT_COMPUTABLE"]
        return f"Free cash flow margin: N/A ({_join_humanized(reasons)})."
    return f"Free cash flow margin: {_fmt_pct(margin)}."


def _render_normalized_operating_margin_line(packet: SectorCompanyFinancialPacket | None) -> str:
    quality = _business_quality(packet)
    normalized = _num(quality.get("normalized_operating_margin"))
    latest = _num(quality.get("latest_operating_margin"))
    if normalized is None:
        reasons = quality.get("normalized_operating_margin_not_computable_reasons")
        if not isinstance(reasons, list) or not reasons:
            reasons = ["NORMALIZED_MARGIN_INSUFFICIENT_HISTORY"]
        return f"Normalized operating margin (5Y avg): N/A ({_join_humanized(reasons)})."
    return f"Normalized operating margin (5Y avg): {_fmt_pct(normalized)} vs latest {_fmt_pct(latest)}."


def _render_inventory_days_line(packet: SectorCompanyFinancialPacket | None) -> str:
    conversion = _cash_conversion(packet)
    days = _num(conversion.get("days_inventory_outstanding"))
    if days is None:
        reasons = _cash_conversion_missing_reasons(
            conversion,
            default="INVENTORY_DAYS_NOT_COMPUTABLE",
            relevant={
                "CCC_NOT_COMPUTABLE",
                "INVENTORY_MISSING",
                "COGS_MISSING",
                "COGS_NONPOSITIVE",
            },
        )
        return f"Inventory days: N/A ({_join_humanized(reasons)})."
    return f"Inventory days: {_fmt_days(days)}."


def _render_inventory_turnover_line(packet: SectorCompanyFinancialPacket | None) -> str:
    conversion = _cash_conversion(packet)
    days = _num(conversion.get("days_inventory_outstanding"))
    if days is None or days <= 0:
        reasons = _cash_conversion_missing_reasons(
            conversion,
            default="INVENTORY_TURNOVER_NOT_COMPUTABLE",
            relevant={
                "CCC_NOT_COMPUTABLE",
                "INVENTORY_MISSING",
                "COGS_MISSING",
                "COGS_NONPOSITIVE",
            },
        )
        return f"Inventory turnover: N/A ({_join_humanized(reasons)})."
    return f"Inventory turnover: {365.0 / days:.1f}x."


def _render_share_count_cagr_line(packet: SectorCompanyFinancialPacket | None) -> str:
    allocation = _capital_allocation(packet)
    cagr = allocation.get("share_count_cagr")
    if not isinstance(cagr, (int, float)) or isinstance(cagr, bool):
        return f"Share count CAGR: N/A ({_join_humanized(allocation.get('share_count_cagr_not_computable_reasons') if isinstance(allocation.get('share_count_cagr_not_computable_reasons'), list) else [])})."
    direction = str(allocation.get("share_count_cagr_direction") or "").replace("_", " ").lower()
    oldest = allocation.get("share_count_oldest")
    latest = allocation.get("share_count_latest")
    span = (
        f"; shares {_fmt_num(oldest)} to {_fmt_num(latest)}"
        if isinstance(oldest, (int, float)) and isinstance(latest, (int, float))
        else ""
    )
    return f"Share count CAGR: {_fmt_pct(cagr)} 5Y ({direction or 'direction unavailable'}{span})."


def _render_score_value(
    quality: dict[str, Any],
    *,
    label: str,
    value_key: str,
    interpretation_key: str,
    missing_key: str,
    status_key: str,
    formatter: str,
) -> str:
    value = quality.get(value_key)
    interpretation = str(quality.get(interpretation_key) or "interpretation unavailable")
    if (
        quality.get(status_key) == "OK"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        if formatter == "integer_nine":
            return f"{label}={int(value)}/9 ({interpretation})"
        return f"{label}={float(value):.2f} ({interpretation})"
    missing = quality.get(missing_key)
    reason = _join(missing if isinstance(missing, list) else [])
    return f"{label} not computable ({reason})"


def _render_quality_scores_line(packet: SectorCompanyFinancialPacket | None) -> str:
    quality = _business_quality(packet)
    return (
        "Quality scores: "
        + "; ".join(
            [
                _render_score_value(
                    quality,
                    label="Piotroski F",
                    value_key="piotroski_f_score",
                    interpretation_key="piotroski_interpretation",
                    missing_key="piotroski_missing_inputs",
                    status_key="piotroski_status",
                    formatter="integer_nine",
                ),
                _render_score_value(
                    quality,
                    label="Beneish M",
                    value_key="beneish_m_score",
                    interpretation_key="beneish_interpretation",
                    missing_key="beneish_missing_inputs",
                    status_key="beneish_status",
                    formatter="decimal",
                ),
                _render_score_value(
                    quality,
                    label="Altman Z''",
                    value_key="altman_z_score",
                    interpretation_key="altman_interpretation",
                    missing_key="altman_missing_inputs",
                    status_key="altman_status",
                    formatter="decimal",
                ),
            ]
        )
        + "."
    )


def _render_peer_valuation_summary(packet: SectorCompanyFinancialPacket | None) -> str:
    ev_ebitda = _peer_relative_metric(packet, "ev_ebitda")
    fcf_yield = _peer_relative_metric(packet, "fcf_yield")
    parts: list[str] = []
    if ev_ebitda:
        parts.append(
            "EV/EBITDA "
            f"{_fmt_multiple(ev_ebitda.get('stock_value'))} versus sector median "
            f"{_fmt_multiple(ev_ebitda.get('sector_median'))} "
            f"(Q1 {_fmt_multiple(ev_ebitda.get('sector_q1'))}, Q3 {_fmt_multiple(ev_ebitda.get('sector_q3'))}, "
            f"percentile {_fmt_pct((ev_ebitda.get('percentile_rank') / 100.0) if isinstance(ev_ebitda.get('percentile_rank'), (int, float)) else None)})."
        )
    if fcf_yield:
        parts.append(
            "FCF yield "
            f"{_fmt_pct(fcf_yield.get('stock_value'))} versus sector median "
            f"{_fmt_pct(fcf_yield.get('sector_median'))} "
            f"(Q1 {_fmt_pct(fcf_yield.get('sector_q1'))}, Q3 {_fmt_pct(fcf_yield.get('sector_q3'))}, "
            f"percentile {_fmt_pct((fcf_yield.get('percentile_rank') / 100.0) if isinstance(fcf_yield.get('percentile_rank'), (int, float)) else None)})."
        )
    return (
        " ".join(parts)
        if parts
        else "Peer-relative valuation context was unavailable for EV/EBITDA and FCF yield."
    )


_HISTORICAL_MULTIPLE_LABELS = {
    "ev_to_ebitda": "EV/EBITDA",
    "pe": "P/E",
    "price_to_book": "P/B",
    "fcf_yield": "FCF yield",
}


def _historical_multiples(packet: SectorCompanyFinancialPacket | None) -> dict[str, Any]:
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    multiples = valuation.get("historical_multiples")
    return multiples if isinstance(multiples, dict) else {}


def _fmt_historical_multiple(metric: str, value: Any) -> str:
    if metric == "fcf_yield":
        return _fmt_pct(value)
    return _fmt_multiple(value)


def _render_historical_multiple_fragment(metric: str, payload: dict[str, Any]) -> str | None:
    status = str(payload.get("status") or "").upper()
    years = int(payload.get("years_of_history") or 0)
    percentile = _num(payload.get("current_percentile"))
    current = payload.get("current_value")
    if status not in {"OK", "INSUFFICIENT_HISTORY"} or years <= 0 or percentile is None:
        return None
    label = _HISTORICAL_MULTIPLE_LABELS.get(metric, metric)
    range_label = "10-year" if years >= 10 else f"available {years}-year"
    history_note = "" if status == "OK" else " (full 10y history not available)"
    cheap_note = " (high yield = cheap)" if metric == "fcf_yield" else ""
    return (
        f"{label} at the {percentile:.1f}th percentile of its {range_label} range "
        f"(current {_fmt_historical_multiple(metric, current)} vs range "
        f"{_fmt_historical_multiple(metric, payload.get('range_min'))} – "
        f"{_fmt_historical_multiple(metric, payload.get('range_max'))}, median "
        f"{_fmt_historical_multiple(metric, payload.get('range_median'))})"
        f"{history_note}{cheap_note}"
    )


def _render_historical_multiple_bands(packet: SectorCompanyFinancialPacket | None) -> str:
    multiples = _historical_multiples(packet)
    ordered = ("ev_to_ebitda", "pe", "price_to_book", "fcf_yield")
    fragments: list[str] = []
    for metric in ordered:
        payload = multiples.get(metric)
        if isinstance(payload, dict):
            fragment = _render_historical_multiple_fragment(metric, payload)
            if fragment:
                fragments.append(fragment)
    if fragments:
        return "Historical multiples: " + "; ".join(fragments) + "."
    reasons: list[str] = []
    for payload in multiples.values():
        if isinstance(payload, dict):
            reasons.extend(str(reason) for reason in payload.get("not_computable_reasons") or [])
    reason_text = (
        ", ".join(list(dict.fromkeys(reasons))) if reasons else "HISTORICAL_MULTIPLE_DATA_MISSING"
    )
    return f"Historical multiple bands not computable: {reason_text}."


def _humanize_code(value: Any) -> str:
    code = str(value or "").strip()
    mapping = {
        "NO_FILING": "filing access is missing",
        "NO_READABLE_ANNUAL_FILING": "readable annual filing evidence is missing",
        "FILING_RISK_NO_FILING": "filing-risk evidence is unavailable",
        "BASE_RETURN_BELOW_12PCT_HURDLE": "base-case return remains below the 12% hurdle",
        "MISSING_VALUATION": "valuation evidence is missing",
        "MISSING_COMPANY_SPECIFIC_EVIDENCE": "company-specific evidence is incomplete",
        "CURRENT_EVENTS_UNAVAILABLE": "current-event evidence is unavailable",
        "CAPITAL_LOSS_EVIDENCE_DEGRADED": "capital-loss underwriting evidence is degraded",
        "FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE": "framework-required evidence is incomplete",
        "NEAR_TERM_MATURITY_UNRESOLVED": "near-term debt maturity evidence is unresolved",
        "ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK": "liquidity or solvency risk is elevated",
        "DOWNSIDE_ASYMMETRY_UNRESOLVED": "downside asymmetry remains unresolved",
        "FOLLOW_UP_CAPITAL_STRUCTURE_ACTIVE_DISTRESS": "follow-up evidence still shows active capital-structure distress",
        "FOLLOW_UP_NO_ASSURANCE_FINANCING": "follow-up evidence still contains no-assurance financing risk",
        "MISSING_KPI_QUALITY_EVIDENCE": "KPI quality evidence is missing",
        "FINANCIAL_ANOMALIES_PRESENT": "financial anomalies remain present",
        "QUARTERLY_REVENUE_TREND_UNKNOWN": "quarterly revenue trend is unknown",
        "MISSING_DOWNSIDE_RETURN_CASE": "downside return case is missing",
        "STRUCTURALLY_WEAK_ECONOMICS": "economic quality appears structurally weak",
        "MISSING_BASE_RETURN_CASE": "base-return case is missing",
        "GROSS_MARGIN_NOT_COMPUTABLE": "gross margin is not computable",
        "GROSS_PROFIT_OR_COGS_MISSING": "gross profit or COGS is missing",
        "CCC_NOT_COMPUTABLE": "cash conversion cycle is not computable",
        "COGS_MISSING": "COGS is missing",
        "COGS_NONPOSITIVE": "COGS is nonpositive",
        "ACCOUNTS_RECEIVABLE_MISSING": "accounts receivable is missing",
        "INVENTORY_MISSING": "inventory is missing",
        "ACCOUNTS_PAYABLE_MISSING": "accounts payable is missing",
        "REVENUE_MISSING": "revenue is missing",
        "REVENUE_NONPOSITIVE": "revenue is nonpositive",
        "SHARE_COUNT_CAGR_NOT_COMPUTABLE": "share-count CAGR is not computable",
        "SHARE_COUNT_HISTORY_INSUFFICIENT": "share-count history is insufficient",
        "FCF_MARGIN_NOT_COMPUTABLE": "free cash flow margin is not computable",
        "INVENTORY_DAYS_NOT_COMPUTABLE": "inventory days are not computable",
        "INVENTORY_TURNOVER_NOT_COMPUTABLE": "inventory turnover is not computable",
        "NORMALIZED_MARGIN_INSUFFICIENT_HISTORY": "normalized margin history is insufficient",
        "OPERATING_MARGIN_NOT_COMPUTABLE": "operating margin is not computable",
        "OPERATING_INCOME_MISSING": "operating income is missing",
        "COMPANYFACTS_UNAVAILABLE": "companyfacts data is unavailable",
    }
    if code in mapping:
        return mapping[code]
    return code.replace("_", " ").lower() if code else "unresolved evidence"


def _join_humanized(items: list[Any] | None) -> str:
    values = [_humanize_code(item) for item in items or [] if str(item or "").strip()]
    return ", ".join(values) if values else "no binding flags"


def _humanize_memo_text(text: Any) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    return re.sub(
        r"\b[A-Z0-9]+(?:_[A-Z0-9]+)+\b", lambda match: _humanize_code(match.group(0)), raw
    )


def _best_base_return(artifact: AutonomousSectorFinancialRunArtifact, ticker: str) -> float | None:
    ranking = _ranking_by_ticker(artifact).get(ticker.upper()) or {}
    value = _num(ranking.get("best_base_annualized_return"))
    if value is not None:
        return value
    scenario = _ticker_scenarios(artifact, ticker).get("base")
    return _num(scenario.annualized_return if scenario else None)


def _best_downside_return(
    artifact: AutonomousSectorFinancialRunArtifact, ticker: str
) -> float | None:
    ranking = _ranking_by_ticker(artifact).get(ticker.upper()) or {}
    value = _num(ranking.get("downside_annualized_return"))
    if value is not None:
        return value
    scenario = _ticker_scenarios(artifact, ticker).get("downside")
    return _num(scenario.annualized_return if scenario else None)


def _latest_margin_delta(
    ticker: str,
    *,
    as_of_date: str,
) -> tuple[float | None, float | None, float | None, Any, Any]:
    rows = _fetch_5y_snapshot(ticker, as_of_date=as_of_date)
    with_margin = [row for row in rows if _num(row.get("op_margin")) is not None]
    if len(with_margin) < 2:
        latest = with_margin[0] if with_margin else {}
        return _num(latest.get("op_margin")), None, None, latest.get("fy"), None
    latest = with_margin[0]
    oldest = with_margin[-1]
    latest_margin = _num(latest.get("op_margin"))
    oldest_margin = _num(oldest.get("op_margin"))
    delta = (
        latest_margin - oldest_margin
        if latest_margin is not None and oldest_margin is not None
        else None
    )
    return latest_margin, oldest_margin, delta, latest.get("fy"), oldest.get("fy")


def _candidate_metric_rows(artifact: AutonomousSectorFinancialRunArtifact) -> list[dict[str, Any]]:
    packets = _packet_by_ticker(artifact)
    rankings = _ranking_by_ticker(artifact)
    rows: list[dict[str, Any]] = []
    for order, ticker in enumerate(_candidate_tickers(artifact), start=1):
        packet = packets.get(ticker)
        ranking = rankings.get(ticker.upper()) or {}
        quality = (
            packet.business_quality if packet and isinstance(packet.business_quality, dict) else {}
        )
        valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
        anchor = _valuation_anchor(packet)
        current_price = _num(packet.current_price if packet else None)
        discount_to_anchor = (
            (anchor - current_price) / anchor
            if anchor is not None and anchor > 0 and current_price is not None
            else _num(valuation.get("discount_to_anchor"))
        )
        rows.append(
            {
                "ticker": ticker,
                "company": _company_name(packet, ticker),
                "candidate_order": order,
                "deterministic_rank": ranking.get("rank"),
                "base_return": _best_base_return(artifact, ticker),
                "downside_return": _best_downside_return(artifact, ticker),
                "revenue_cagr_5y": _num(quality.get("revenue_cagr_5y")),
                "quarterly_revenue_trend": quality.get("quarterly_revenue_trend"),
                "roic": _roic(packet),
                "roic_wacc_spread": _roic_wacc_spread(packet),
                "fcf_yield": _fcf_yield(packet),
                "current_price": current_price,
                "anchor": anchor,
                "anchor_method": _anchor_method(packet),
                "discount_to_anchor": discount_to_anchor,
                "audit_status": ranking.get("audit_status")
                or _ticker_audit_status(artifact, ticker),
                "conviction": _conviction_grade(artifact, ticker),
                "blockers": list(packet.blockers if packet else [])
                + list(ranking.get("hard_blockers") or []),
                "confidence_caps": list(packet.confidence_caps if packet else [])
                + list(ranking.get("confidence_caps") or []),
                "valuation": valuation,
                "packet": packet,
            }
        )
    return rows


def _row_with_max(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if _num(row.get(key)) is not None]
    return max(candidates, key=lambda row: float(row[key])) if candidates else None


def _row_with_min(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if _num(row.get(key)) is not None]
    return min(candidates, key=lambda row: float(row[key])) if candidates else None


def _metric_computed_for_packet(
    packet: SectorCompanyFinancialPacket | None,
    metric: str,
    scenarios: list[SectorExpectedReturnScenario],
) -> bool:
    if packet is None:
        return False
    quality = _business_quality(packet)
    conversion = _cash_conversion(packet)
    allocation = _capital_allocation(packet)
    if metric in {"revenue_cagr_5y", "organic_revenue_cagr", "organic_revenue_growth"}:
        return _num(quality.get("revenue_cagr_5y")) is not None
    if metric == "gross_margin":
        return _num(quality.get("gross_margin")) is not None
    if metric == "cash_conversion_cycle":
        return _num(conversion.get("cash_conversion_cycle")) is not None
    if metric == "share_count_cagr":
        return (
            _num(allocation.get("share_count_cagr")) is not None
            or _num(allocation.get("dilution_rate_shares_cagr")) is not None
        )
    if metric == "free_cash_flow_margin":
        return _num(conversion.get("fcf_margin")) is not None
    if metric == "inventory_days":
        return _num(conversion.get("days_inventory_outstanding")) is not None
    if metric == "inventory_turnover":
        days = _num(conversion.get("days_inventory_outstanding"))
        return days is not None and days > 0
    if metric == "normalized_operating_margin":
        return _num(quality.get("normalized_operating_margin")) is not None or any(
            _num(scenario.normalized_operating_margin) is not None
            for scenario in scenarios
            if scenario.ticker.upper() == packet.ticker.upper()
        )
    return False


def _framework_metric_usage(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> tuple[list[str], list[str]]:
    declared = list(artifact.framework.selected_metrics if artifact.framework else [])
    packets = _packet_by_ticker(artifact)
    scenarios = list(artifact.expected_return_scenarios)
    computed: list[str] = []
    not_implemented: list[str] = []
    for metric in declared:
        if any(
            _metric_computed_for_packet(packet, metric, scenarios) for packet in packets.values()
        ):
            computed.append(metric)
        else:
            not_implemented.append(metric)
    return computed, not_implemented


def _memo_section(artifact: AutonomousSectorFinancialRunArtifact, key: str) -> dict[str, Any]:
    memo_body = artifact.memo_body if isinstance(artifact.memo_body, dict) else {}
    section = memo_body.get(key)
    return section if isinstance(section, dict) else {}


def _memo_candidate_payload(
    artifact: AutonomousSectorFinancialRunArtifact, ticker: str
) -> dict[str, Any]:
    memo_body = artifact.memo_body if isinstance(artifact.memo_body, dict) else {}
    candidates = memo_body.get("candidates")
    if isinstance(candidates, dict):
        item = candidates.get(ticker.upper()) or candidates.get(ticker)
        return item if isinstance(item, dict) else {}
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict) and str(item.get("ticker") or "").upper() == ticker.upper():
                return item
    return {}


def _section_degraded_line(
    section: dict[str, Any], default_state: str = MEMO_BODY_DEGRADED_STATE
) -> str | None:
    source = str(section.get("source") or "").lower()
    status = str(section.get("status") or "").upper()
    if source == "deterministic_fallback" or "DEGRADED" in status:
        state = str(section.get("degraded_state") or default_state)
        return f"**DEGRADED_STATE:** {state}; deterministic fallback uses packet values."
    return None


def _cohort_comparison_fallback(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    rows = _candidate_metric_rows(artifact)
    if not rows:
        return ["No displayed candidates had packet data available for cohort comparison."]
    best_return = _row_with_max(rows, "base_return")
    best_growth = _row_with_max(rows, "revenue_cagr_5y")
    best_roic = _row_with_max(rows, "roic")
    best_fcf_yield = _row_with_max(rows, "fcf_yield")
    cheapest_anchor = _row_with_max(rows, "discount_to_anchor")
    clearing = [
        row for row in rows if (_num(row.get("base_return")) or -999.0) >= BASE_RETURN_HURDLE
    ]
    blocked = [row for row in rows if str(row.get("audit_status") or "").upper() == "BLOCKED"]

    quality_parts = []
    if best_growth:
        quality_parts.append(
            f"{best_growth['ticker']} has the strongest 5Y revenue CAGR at {_fmt_pct(best_growth.get('revenue_cagr_5y'))}"
        )
    if best_roic:
        quality_parts.append(
            f"{best_roic['ticker']} has the highest latest ROIC at {_fmt_pct(best_roic.get('roic'))}"
        )
    if not quality_parts:
        quality_parts.append("packet quality metrics are incomplete across the cohort")

    valuation_parts = []
    if best_fcf_yield:
        valuation_parts.append(
            f"{best_fcf_yield['ticker']} shows the highest FCF yield at {_fmt_pct(best_fcf_yield.get('fcf_yield'))}"
        )
    if cheapest_anchor:
        valuation_parts.append(
            f"{cheapest_anchor['ticker']} screens furthest below its {_na(cheapest_anchor.get('anchor_method'))} anchor at {_fmt_pct(cheapest_anchor.get('discount_to_anchor'))}"
        )
    if best_return:
        valuation_parts.append(
            f"{best_return['ticker']} leads base-case return at {_fmt_pct(best_return.get('base_return'))}"
        )

    return [
        (
            f"Among the {len(rows)} displayed candidates, "
            f"{'; '.join(quality_parts)}. "
            f"{len(clearing)} candidate(s) clear the 12.0% base-return hurdle before audit overlays, "
            f"while {len(blocked)} are blocked by deterministic guardrails."
        ),
        (
            f"Valuation asymmetry is concentrated rather than broad: "
            f"{'; '.join(valuation_parts) if valuation_parts else 'packet valuation metrics are incomplete across the cohort'}. "
            "The memo should therefore treat the cohort as a watchlist-ranking exercise first and an actionable-selection exercise only after the appendix guardrails clear."
        ),
    ]


def _triage_surprises_fallback(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    rows = _candidate_metric_rows(artifact)
    surprises: list[str] = []
    for row in rows:
        rank = row.get("deterministic_rank")
        if isinstance(rank, int) and rank != row["candidate_order"]:
            surprises.append(
                f"{row['ticker']} appears #{row['candidate_order']} in the displayed finalist order but #{rank} in deterministic base-return rank. "
                f"That matters because its base-case return is {_fmt_pct(row.get('base_return'))}, so the memo needs an explicit qualitative reason for the ordering gap."
            )
            break
    top_return = _row_with_max(rows, "base_return")
    if top_return and str(top_return.get("audit_status") or "").upper() != "PASS":
        reasons = _join_humanized(
            top_return.get("blockers") or top_return.get("confidence_caps") or []
        )
        surprises.append(
            f"{top_return['ticker']} has the strongest base-case return at {_fmt_pct(top_return.get('base_return'))} but does not pass audit. "
            f"That matters because {reasons} prevents the highest-return screen from becoming the clean memo answer."
        )
    for row in rows:
        values = [
            value
            for value in (
                _num(
                    row["valuation"].get("dcf_value")
                    if isinstance(row.get("valuation"), dict)
                    else None
                ),
                _num(
                    row["valuation"].get("epv_value")
                    if isinstance(row.get("valuation"), dict)
                    else None
                ),
                _num(
                    row["valuation"].get("graham_value")
                    if isinstance(row.get("valuation"), dict)
                    else None
                ),
            )
            if value is not None and value > 0
        ]
        if len(values) >= 2:
            high = max(values)
            low = min(values)
            spread = (high - low) / high if high else None
            if spread is not None and spread >= 0.50:
                surprises.append(
                    f"{row['ticker']} has wide valuation-method disagreement, with method values spanning {_fmt_money(low)} to {_fmt_money(high)}. "
                    f"That matters because the {_fmt_pct(spread)} spread makes single-anchor upside less reliable."
                )
                break
    for row in rows:
        if (
            isinstance(row.get("deterministic_rank"), int)
            and row["deterministic_rank"] > 3
            and (_num(row.get("discount_to_anchor")) or -999.0) >= 0.25
        ):
            surprises.append(
                f"{row['ticker']} trades {_fmt_pct(row.get('discount_to_anchor'))} below its valuation anchor but sits outside the top three deterministic ranks. "
                "That matters because apparent cheapness is being offset by return, quality, or evidence constraints."
            )
            break
    return surprises or [
        "No mechanical triage surprises surfaced: deterministic pre-rank, return ranking, and guardrail outcome were aligned across the displayed candidates."
    ]


def _render_cohort_comparison(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    section = _memo_section(artifact, "cohort_comparison")
    paragraphs = [
        str(item).strip() for item in section.get("paragraphs") or [] if str(item).strip()
    ]
    if not paragraphs:
        paragraphs = _cohort_comparison_fallback(artifact)
        section = {
            **section,
            "source": "deterministic_fallback",
            "degraded_state": section.get("degraded_state") or MEMO_BODY_DEGRADED_STATE,
        }
    lines = ["## Cohort Comparison", ""]
    degraded = _section_degraded_line(section)
    if degraded:
        lines.append(degraded)
        lines.append("")
    for paragraph in paragraphs[:3]:
        lines.append(_line_break(_humanize_memo_text(paragraph)))
        lines.append("")
    return lines


def _render_triage_surprises(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    section = _memo_section(artifact, "triage_surprises")
    items = [str(item).strip() for item in section.get("items") or [] if str(item).strip()]
    if not items:
        items = _triage_surprises_fallback(artifact)
        section = {
            **section,
            "source": "deterministic_fallback",
            "degraded_state": section.get("degraded_state") or MEMO_BODY_DEGRADED_STATE,
        }
    lines = ["## Triage Surprises", ""]
    degraded = _section_degraded_line(section)
    if degraded:
        lines.append(degraded)
        lines.append("")
    for item in items[:8]:
        lines.append(f"- {_line_break(_humanize_memo_text(item))}")
    lines.append("")
    return lines


def _deterministic_candidate_thesis(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    packet: SectorCompanyFinancialPacket | None,
    ranking: dict[str, Any],
) -> str:
    quality = (
        packet.business_quality if packet and isinstance(packet.business_quality, dict) else {}
    )
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    blockers = list(packet.blockers if packet else []) + list(ranking.get("hard_blockers") or [])
    caps = list(packet.confidence_caps if packet else []) + list(
        ranking.get("confidence_caps") or []
    )
    base_return = _best_base_return(artifact, ticker)
    downside_return = _best_downside_return(artifact, ticker)
    anchor = _valuation_anchor(packet)
    current_price = _num(packet.current_price if packet else None)
    tension_flags = list(blockers or caps)
    if base_return is not None and base_return < BASE_RETURN_HURDLE:
        tension_flags.append("BASE_RETURN_BELOW_12PCT_HURDLE")
    tension = _join_humanized(tension_flags)
    return (
        f"{ticker} is presented as {_company_name(packet, ticker)}, a candidate with {_na(packet.financial_status if packet else None).lower()} packet status and {_na(quality.get('moat_classification')).lower()} quality markers. "
        f"The central tension is {_fmt_pct(quality.get('revenue_cagr_5y'))} 5Y revenue growth and {_fmt_pct(_fcf_yield(packet))} FCF yield against {tension}. "
        f"The base case projects {_fmt_pct(base_return)} annualized versus a downside case of {_fmt_pct(downside_return)}. "
        f"Valuation anchors on {_na(valuation.get('anchor_method'))} at {_fmt_money(anchor)} versus a current price of {_fmt_money(current_price)}. "
        f"The current memo stance is {_conviction_grade(artifact, ticker).lower()} because the packet still has {_join_humanized(blockers or caps)}."
    )


def _deterministic_candidate_risks(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    packet: SectorCompanyFinancialPacket | None,
    ranking: dict[str, Any],
) -> list[str]:
    quality = (
        packet.business_quality if packet and isinstance(packet.business_quality, dict) else {}
    )
    balance_sheet = (
        packet.balance_sheet if packet and isinstance(packet.balance_sheet, dict) else {}
    )
    risks: list[str] = []
    revenue_cagr = _num(quality.get("revenue_cagr_5y"))
    if revenue_cagr is not None:
        direction = "declined" if revenue_cagr < 0 else "grew"
        risks.append(
            f"Revenue trajectory: 5Y revenue {direction} at {_fmt_pct(revenue_cagr)} CAGR; durability needs confirmation against the cycle."
        )
    latest_margin, oldest_margin, margin_delta, latest_year, oldest_year = _latest_margin_delta(
        ticker,
        as_of_date=artifact.as_of_date,
    )
    if margin_delta is not None:
        verb = "compressed" if margin_delta < 0 else "expanded"
        risks.append(
            f"Operating margin {verb}: {_fmt_pct(oldest_margin)} ({_na(oldest_year)}) to {_fmt_pct(latest_margin)} ({_na(latest_year)}), a {_fmt_pct(abs(margin_delta))} move."
        )
    base_return = _best_base_return(artifact, ticker)
    if base_return is not None and base_return < BASE_RETURN_HURDLE:
        risks.append(
            f"Return hurdle: base-case annualized return is {_fmt_pct(base_return)}, below the 12.0% product hurdle."
        )
    solvency = balance_sheet.get("solvency_risk")
    cash_runway = _num(balance_sheet.get("cash_runway_quarters"))
    if solvency and str(solvency).upper() not in {"LOW", "N/A", "NONE"}:
        risks.append(
            f"Balance sheet: solvency risk is {_na(solvency)} and cash runway is {_na(cash_runway)} quarters."
        )
    elif cash_runway is not None and cash_runway < 8:
        risks.append(
            f"Balance sheet: cash runway is {_fmt_num(cash_runway)} quarters, leaving limited room for cyclical pressure."
        )
    flags = (
        list(packet.blockers if packet else [])
        + list(ranking.get("hard_blockers") or [])
        + list(packet.confidence_caps if packet else [])
        + list(ranking.get("confidence_caps") or [])
    )
    for flag in flags:
        sentence = f"Evidence flag: {_humanize_code(flag)} still needs resolution before conviction can rise."
        if sentence not in risks:
            risks.append(sentence)
        if len(risks) >= 6:
            break
    if len(risks) < 3:
        risks.append(
            f"Valuation evidence: {_anchor_method(packet)} anchor is {_fmt_money(_valuation_anchor(packet))} versus current price {_fmt_money(packet.current_price if packet else None)}."
        )
    if len(risks) < 3:
        risks.append("Company-specific qualitative evidence remains thin relative to the memo bar.")
    return risks[:6]


def _deterministic_candidate_falsifiers(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    packet: SectorCompanyFinancialPacket | None,
    ranking: dict[str, Any],
) -> list[str]:
    latest_margin, oldest_margin, margin_delta, latest_year, oldest_year = _latest_margin_delta(
        ticker,
        as_of_date=artifact.as_of_date,
    )
    items = [
        "Fresh valuation work leaves the base-case annualized return below the 12.0% hurdle.",
        "Updated filing evidence fails to resolve the company-specific evidence gaps flagged in the appendix.",
    ]
    if margin_delta is not None:
        items.append(
            f"Operating margin fails to recover from {_fmt_pct(latest_margin)} ({_na(latest_year)}) toward the {_fmt_pct(oldest_margin)} ({_na(oldest_year)}) baseline."
            if margin_delta < 0
            else f"Operating margin gives back the {_fmt_pct(abs(margin_delta))} improvement seen over the 5-year packet window."
        )
    if _fcf_yield(packet) is not None:
        items.append(
            f"FCF yield falls from {_fmt_pct(_fcf_yield(packet))} without a corresponding improvement in growth or balance-sheet quality."
        )
    blockers = list(packet.blockers if packet else []) + list(ranking.get("hard_blockers") or [])
    if blockers:
        items.append(
            f"The appendix blockers remain unresolved, especially {_join_humanized(blockers[:2])}."
        )
    return list(dict.fromkeys(items))[:4]


def _deterministic_candidate_open_questions(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    packet: SectorCompanyFinancialPacket | None,
    ranking: dict[str, Any],
) -> list[str]:
    quality = (
        packet.business_quality if packet and isinstance(packet.business_quality, dict) else {}
    )
    questions = [
        f"What specific demand, backlog, or pricing evidence explains the {_fmt_pct(quality.get('revenue_cagr_5y'))} 5Y revenue trajectory?",
        "Is the latest operating-margin profile cyclical, mix-driven, or structurally impaired?",
        f"What evidence would resolve {_join_humanized((packet.blockers if packet else []) + list(ranking.get('hard_blockers') or []))}?",
        f"How should the {_anchor_method(packet)} valuation anchor be adjusted if normalized margins or working-capital intensity change?",
        f"Does peer valuation support the current {_fmt_pct(_fcf_yield(packet))} FCF yield, or is the screen masking balance-sheet or cyclicality risk?",
    ]
    return questions[:5]


def _ticker_audit_status(artifact: AutonomousSectorFinancialRunArtifact, ticker: str) -> str:
    ranking = _ranking_by_ticker(artifact).get(ticker.upper()) or {}
    if ranking.get("audit_status"):
        return str(ranking["audit_status"])
    audit = artifact.selection_audit or {}
    if str(audit.get("selected_ticker") or "").upper() == ticker.upper():
        return str(audit.get("status") or "UNKNOWN")
    return "NOT_AUDITED"


def _conviction_grade(artifact: AutonomousSectorFinancialRunArtifact, ticker: str) -> str:
    status = _ticker_audit_status(artifact, ticker).upper()
    if (
        ticker.upper() == str(artifact.selected_ticker or "").upper()
        and str(artifact.final_verdict).upper() == "SELECTED"
    ):
        return artifact.confidence or "MODERATE"
    if status == "PASS":
        ranking = _ranking_by_ticker(artifact).get(ticker.upper()) or {}
        if ranking.get("buy_candidate") is True:
            return "ACTIONABLE"
        return "MODERATE"
    if status == "DATA_INCOMPLETE":
        return "DATA_INCOMPLETE"
    if status == "WATCHLIST_ONLY":
        return "WATCHLIST"
    if status == "BLOCKED":
        return "AVOID"
    return "UNRATED"


def _ticker_scenarios(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
) -> dict[str, SectorExpectedReturnScenario]:
    scenarios: dict[str, SectorExpectedReturnScenario] = {}
    for scenario in artifact.expected_return_scenarios:
        if scenario.ticker.upper() == ticker.upper():
            scenarios[scenario.scenario_name.lower()] = scenario
    return scenarios


def _render_financial_snapshot(
    ticker: str,
    *,
    as_of_date: str,
) -> list[str]:
    rows = _fetch_5y_snapshot(ticker, as_of_date=as_of_date)
    lines = ["**5-Year Financial Table** (FY, cached SEC companyfacts; USD where reported)", ""]
    lines.append("| FY | Revenue | Op Margin | CFO | FCF | Cash | Debt | Shares | SBC |")
    lines.append("|----|---------|-----------|-----|-----|------|------|--------|-----|")
    if not rows:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{_na(row.get('fy'))} | "
            f"{_fmt_num(row.get('revenue'))} | "
            f"{_fmt_pct(row.get('op_margin'))} | "
            f"{_fmt_num(row.get('cfo'))} | "
            f"{_fmt_num(row.get('fcf'))} | "
            f"{_fmt_num(row.get('cash'))} | "
            f"{_fmt_num(row.get('debt'))} | "
            f"{_fmt_num(row.get('shares'))} | "
            f"{_fmt_num(row.get('sbc'))} |"
        )
    return lines


def _render_candidate_table(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    packets = _packet_by_ticker(artifact)
    lines = [
        "| Ticker | Company | Market Cap | Conviction Grade | ROIC | ROIC−WACC Spread | FCF Yield | Valuation Anchor | Buy-Price Target | Current Price | Distance From Buy |",
        "|--------|---------|------------|------------------|------|------------------|-----------|------------------|------------------|---------------|-------------------|",
    ]
    for ticker in _candidate_tickers(artifact):
        packet = packets.get(ticker)
        anchor = _valuation_anchor(packet)
        # Pass the store-canonical grade/confidence so the memo
        # buy-price + distance columns match the persisted DB buy_price_target.
        buy_grade, buy_confidence = _buy_target_grade_confidence(artifact, ticker)
        lines.append(
            "| "
            f"{ticker} | "
            f"{_cell(_company_name(packet, ticker))} | "
            f"{_fmt_market_cap(_market_cap(packet, ticker))} | "
            f"{_conviction_grade(artifact, ticker)} | "
            f"{_fmt_pct(_roic(packet))} | "
            f"{_fmt_pct(_roic_wacc_spread(packet))} | "
            f"{_fmt_pct(_fcf_yield(packet))} | "
            f"{_anchor_method(packet)} {_fmt_money(anchor)} | "
            f"{_fmt_money(_buy_price_target(packet, conviction_grade=buy_grade, confidence=buy_confidence))} | "
            f"{_fmt_money(packet.current_price if packet else None)} | "
            f"{_fmt_pct(_distance_from_buy(packet, conviction_grade=buy_grade, confidence=buy_confidence))} |"
        )
    return lines


def _render_candidate_section(
    artifact: AutonomousSectorFinancialRunArtifact,
    ticker: str,
    packet: SectorCompanyFinancialPacket | None,
) -> list[str]:
    company = _company_name(packet, ticker)
    ranking = _ranking_by_ticker(artifact).get(ticker.upper()) or {}
    scenarios = _ticker_scenarios(artifact, ticker)
    base = scenarios.get("base")
    downside = scenarios.get("downside")
    upside = scenarios.get("upside")
    valuation = packet.valuation if packet and isinstance(packet.valuation, dict) else {}
    quality = (
        packet.business_quality if packet and isinstance(packet.business_quality, dict) else {}
    )
    balance_sheet = (
        packet.balance_sheet if packet and isinstance(packet.balance_sheet, dict) else {}
    )
    caps = list(packet.confidence_caps if packet else []) + list(
        ranking.get("confidence_caps") or []
    )
    blockers = list(packet.blockers if packet else []) + list(ranking.get("hard_blockers") or [])
    is_selected = ticker.upper() == str(artifact.selected_ticker or "").upper()
    decision = artifact.final_decision
    memo_payload = _memo_candidate_payload(artifact, ticker)
    memo_source = str(memo_payload.get("source") or "").lower()
    memo_degraded = _section_degraded_line(
        memo_payload,
        default_state=str(memo_payload.get("degraded_state") or MEMO_BODY_DEGRADED_STATE),
    )

    lines = [f"### {ticker} — {_cell(company)}", ""]
    lines.extend(
        _render_financial_snapshot(
            ticker,
            as_of_date=artifact.as_of_date,
        )
    )
    lines.append("")
    lines.append("**Business Quality**")
    lines.append("")
    lines.append(
        f"- Financial status: {_humanize_memo_text(_na(packet.financial_status if packet else None))}; "
        f"model fit: {_humanize_memo_text(_na(packet.model_fit_status if packet else None))}."
    )
    lines.append(
        f"- Moat / quality: {_humanize_memo_text(_na(quality.get('moat_classification')))}; "
        f"moat score: {_na(quality.get('moat_score'))}; "
        f"earnings quality: {_humanize_memo_text(_na(quality.get('earnings_quality')))}."
    )
    lines.append(
        f"- Revenue trend: 5Y CAGR {_fmt_pct(quality.get('revenue_cagr_5y'))}; "
        f"quarterly trend {_humanize_memo_text(_na(quality.get('quarterly_revenue_trend')))}."
    )
    lines.append(f"- {_render_normalized_operating_margin_line(packet)}")
    lines.append(f"- {_render_gross_margin_line(packet)}")
    lines.append(f"- {_render_cash_conversion_cycle_line(packet)}")
    lines.append(f"- {_render_free_cash_flow_margin_line(packet)}")
    lines.append(f"- {_render_inventory_days_line(packet)}")
    lines.append(f"- {_render_inventory_turnover_line(packet)}")
    lines.append(f"- {_render_share_count_cagr_line(packet)}")
    lines.append(f"- {_render_quality_scores_line(packet)}")
    lines.append(
        f"- Balance sheet: solvency risk {_humanize_memo_text(_na(balance_sheet.get('solvency_risk')))}; "
        f"cash runway {_na(balance_sheet.get('cash_runway_quarters'))} quarters."
    )
    lines.append(
        f"- Returns on capital: latest ROIC {_fmt_pct(_roic(packet))}; ROIC−WACC spread {_fmt_pct(_roic_wacc_spread(packet))}; incremental 3Y ROIC {_fmt_pct(_returns_on_capital(packet).get('incremental_roic_3y'))}."
    )
    lines.append(f"- {_render_roic_trajectory_line(packet)}")
    lines.append("")
    lines.append("**Valuation And Expected-Return Cases**")
    lines.append("")
    lines.append("| Case | Annualized Return | Future Value / Share | Key Sensitivities |")
    lines.append("|------|-------------------|----------------------|-------------------|")
    for label, scenario in (("Downside", downside), ("Base", base), ("Upside", upside)):
        lines.append(
            "| "
            f"{label} | "
            f"{_fmt_pct(scenario.annualized_return if scenario else None)} | "
            f"{_fmt_money(scenario.estimated_future_value_per_share if scenario else None)} | "
            f"{_cell(_join(scenario.key_sensitivities if scenario else []))} |"
        )
    lines.append("")
    # Pass the store-canonical grade/confidence so the memo
    # buy-price target visibly matches the persisted DB buy_price_target.
    buy_grade, buy_confidence = _buy_target_grade_confidence(artifact, ticker)
    lines.append(
        f"Valuation anchor: {_anchor_method(packet)} {_fmt_money(valuation.get('valuation_anchor'))}; "
        f"buy-price target: {_fmt_money(_buy_price_target(packet, conviction_grade=buy_grade, confidence=buy_confidence))}; current price: {_fmt_money(packet.current_price if packet else None)}."
    )
    lines.append("")
    lines.append("Peer-relative valuation: " + _render_peer_valuation_summary(packet))
    lines.append("")
    lines.append(_render_historical_multiple_bands(packet))
    lines.append("")
    lines.append("**Key Risks**")
    lines.append("")
    risks = [str(item).strip() for item in memo_payload.get("key_risks") or [] if str(item).strip()]
    if not risks:
        risks = _deterministic_candidate_risks(artifact, ticker, packet, ranking)
        memo_source = memo_source or "deterministic_fallback"
    for risk in [item for item in risks if item]:
        lines.append(f"- {_line_break(_humanize_memo_text(risk))}")
    if not [item for item in risks if item]:
        lines.append("- No binding risk flags were surfaced in the artifact.")
    lines.append("")
    lines.append("**Thesis**")
    lines.append("")
    thesis = str(memo_payload.get("thesis") or "").strip()
    if not thesis:
        thesis = _deterministic_candidate_thesis(artifact, ticker, packet, ranking)
        memo_source = memo_source or "deterministic_fallback"
    if memo_source != "llm" and not memo_degraded:
        memo_degraded = f"**DEGRADED_STATE:** {MEMO_BODY_DEGRADED_STATE}; deterministic fallback uses packet values."
    if memo_degraded and memo_source != "llm":
        lines.append(memo_degraded)
        lines.append("")
    lines.append(_line_break(_humanize_memo_text(thesis)))
    lines.append("")
    lines.append("**Falsifiers**")
    lines.append("")
    falsifiers = [
        str(item).strip() for item in memo_payload.get("falsifiers") or [] if str(item).strip()
    ]
    if not falsifiers and is_selected and decision:
        falsifiers = list(decision.falsifiers)
    if not falsifiers:
        falsifiers = _deterministic_candidate_falsifiers(artifact, ticker, packet, ranking)
    if falsifiers:
        for falsifier in falsifiers:
            lines.append(f"- {_line_break(_humanize_memo_text(falsifier))}")
    lines.append("")
    lines.append("**Open Questions**")
    lines.append("")
    open_questions = [
        str(item).strip() for item in memo_payload.get("open_questions") or [] if str(item).strip()
    ]
    if not open_questions:
        open_questions = _deterministic_candidate_open_questions(artifact, ticker, packet, ranking)
    for question in open_questions:
        lines.append(f"- {_line_break(_humanize_memo_text(question))}")
    lines.append("")
    lines.append("**Conviction Grade With Reasoning**")
    lines.append("")
    lines.append(
        f"{_conviction_grade(artifact, ticker)} — audit status {_ticker_audit_status(artifact, ticker)}; "
        f"base return {_fmt_pct(base.annualized_return if base else ranking.get('best_base_annualized_return'))}; "
        f"blockers {_join_humanized(blockers)}; caps {_join_humanized(caps)}."
    )
    lines.append("")
    return lines


def _render_decision_section(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    decision = artifact.final_decision or getattr(artifact, "provisional_final_decision", None)
    is_v2_incomplete = (
        getattr(artifact, "pipeline_version", "v1") == "v2"
        and getattr(artifact, "decision_status", None) == "INCOMPLETE"
    )
    lines = ["## Selection / Watchlist / No-Selection Decision", ""]
    if is_v2_incomplete:
        lines.append("**Binding result:** Decision Incomplete.")
        lines.append("")
        lines.append(
            "The execution is preserved, but no final investment verdict is published because "
            "one or more admitted candidates still require data, underwriting, or selected-company validation."
        )
    else:
        lines.append(f"**Binding result:** {_humanize_memo_text(artifact.final_verdict).title()}.")
    lines.append("")
    if is_v2_incomplete:
        lines.append(
            "The sector execution is complete enough to preserve its research state, but the investment "
            "decision is **incomplete**. No selection or no-selection claim is published until every live "
            "frontier candidate has resolved data, underwriting, and validation state."
        )
    elif artifact.final_verdict == "SELECTED":
        lines.append(
            f"The sector memo selects **{_na(artifact.selected_ticker)}** at **{_na(artifact.confidence)}** confidence."
        )
    elif artifact.final_verdict == "WATCHLIST":
        lines.append("The audited result is **WATCHLIST**, not an actionable sector selection.")
        lines.append("")
        lines.append(
            f"The run keeps **{_na(artifact.selected_ticker)}** on watchlist rather than marking it actionable."
        )
    else:
        lines.append(
            _line_break(
                _humanize_memo_text(
                    artifact.no_selection_reason
                    or (decision.no_selection_reason if decision else None)
                    or "No company cleared the product gates."
                )
            )
        )
    if decision:
        lines.append("")
        label_prefix = "Provisional " if is_v2_incomplete else ""
        lines.append(
            f"{label_prefix}thesis (non-binding): "
            f"{_line_break(_humanize_memo_text(decision.thesis))}"
            if is_v2_incomplete
            else f"Thesis: {_line_break(_humanize_memo_text(decision.thesis))}"
        )
        lines.append("")
        lines.append(
            f"{label_prefix}key risk (non-binding): "
            f"{_line_break(_humanize_memo_text(decision.key_risk))}"
            if is_v2_incomplete
            else f"Key risk: {_line_break(_humanize_memo_text(decision.key_risk))}"
        )
        lines.append("")
        lines.append(
            f"{label_prefix}downside case (non-binding): "
            f"{_line_break(_humanize_memo_text(decision.downside_case))}"
            if is_v2_incomplete
            else f"Downside case: {_line_break(_humanize_memo_text(decision.downside_case))}"
        )
    lines.append("")
    return lines


def _render_v2_disposition_section(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> list[str]:
    if getattr(artifact, "pipeline_version", "v1") != "v2":
        return []
    dispositions = list(getattr(artifact, "candidate_dispositions", []) or [])
    counts: dict[str, int] = {}
    for item in dispositions:
        counts[item.terminal_state] = counts.get(item.terminal_state, 0) + 1
    lines = ["## Candidate Disposition Funnel", ""]
    lines.append(
        f"Admitted securities: **{len(getattr(artifact, 'admitted_tickers', []) or [])}**. "
        + " ".join(f"{key}: **{counts[key]}**." for key in sorted(counts))
    )
    lines.extend(
        [
            "",
            "| Ticker | Scope | Screen | Review | Terminal State | Underwriting Verdict | Watchlist Eligible | Reasons |",
            "|--------|-------|--------|--------|----------------|-----------------------|--------------------|---------|",
        ]
    )
    for item in dispositions:
        lines.append(
            f"| {item.ticker} | {item.scope_status} | {item.screen_status} | "
            f"{item.review_status} | {item.terminal_state} | {_na(item.underwriting_verdict)} | "
            f"{'YES' if item.watchlist_eligible else 'NO'} | {_join(item.reason_codes)} |"
        )
    lines.append("")
    return lines


def _render_audit_appendix(artifact: AutonomousSectorFinancialRunArtifact) -> list[str]:
    lines: list[str] = ["## Audit Appendix", ""]
    lines.append(
        "This appendix preserves runtime instrumentation, evidence references, degraded states, and guardrail math."
    )
    lines.append("")
    lines.append("### Run Metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Run ID | {artifact.run_id} |")
    lines.append(f"| Status | {artifact.status} |")
    lines.append(f"| Pipeline Version | {getattr(artifact, 'pipeline_version', 'v1')} |")
    lines.append(f"| Execution Status | {_na(getattr(artifact, 'execution_status', None))} |")
    lines.append(f"| Decision Status | {_na(getattr(artifact, 'decision_status', None))} |")
    lines.append(f"| Final Verdict | {artifact.final_verdict} |")
    lines.append(f"| Selected Ticker | {_na(artifact.selected_ticker)} |")
    lines.append(f"| Confidence | {_na(artifact.confidence)} |")
    lines.append(f"| Market Cap Focus | {artifact.market_cap_focus} |")
    lines.append(f"| As Of Date | {artifact.as_of_date} |")
    lines.append(f"| Created At | {artifact.created_at} |")
    lines.append(f"| Completed At | {_na(artifact.completed_at)} |")
    validation = getattr(artifact, "selection_validation", None)
    lines.append(f"| Selection Validation | {_na(validation.status if validation else None)} |")
    lines.append("")

    if artifact.degraded_states:
        lines.append("### Degraded States")
        lines.append("")
        for state in artifact.degraded_states:
            lines.append(f"- {state}")
        lines.append("")

    selection = artifact.candidate_selection or {}
    lines.append("### Candidate Selection")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Source | {_na(selection.get('source'))} |")
    lines.append(f"| Ranking Basis | {_na(selection.get('ranking_basis'))} |")
    lines.append(f"| Selected Tickers | {_join(selection.get('selected_tickers') or [])} |")
    lines.append(f"| Loaded Tickers | {_join(selection.get('loaded_tickers') or [])} |")
    lines.append(f"| Excluded Tickers | {_join(selection.get('excluded_tickers') or [])} |")
    lines.append(f"| Warnings | {_join(selection.get('warnings') or [])} |")
    lines.append("")

    if artifact.framework:
        framework = artifact.framework
        lines.append("### Financial Framework")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Economic Model | {_line_break(framework.economic_model)} |")
        lines.append(f"| Horizon Years | {_join(framework.horizon_years)} |")
        lines.append(f"| Value Drivers | {_join(framework.selected_value_drivers)} |")
        lines.append(f"| Selected Metrics | {_join(framework.selected_metrics)} |")
        lines.append(f"| Valid Valuation Methods | {_join(framework.valid_valuation_methods)} |")
        lines.append(f"| Required Evidence | {_join(framework.required_evidence)} |")
        lines.append("")

    if _selection_audit_applies(artifact):
        audit = artifact.selection_audit
        lines.append("### Selection Audit")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for label, key, formatter in (
            ("Audit Status", "status", _na),
            ("Actionable", "actionable", _na),
            ("Selected / Focus Ticker", "selected_ticker", _na),
            ("Confidence Ceiling", "confidence_ceiling", _na),
            ("Base Return Hurdle", "base_return_hurdle", _fmt_pct),
            ("Selected Return Cushion Hurdle", "selected_return_cushion_hurdle", _fmt_pct),
            ("Best Base Annualized Return", "best_base_annualized_return", _fmt_pct),
            ("Base Return Margin Over Hurdle", "base_return_margin_over_hurdle", _fmt_pct),
            ("Best Base Horizon", "best_base_horizon_years", _na),
            ("Return Cushion Status", "return_cushion_status", _na),
            ("Downside Annualized Return", "downside_annualized_return", _fmt_pct),
            ("Downside Evidence Status", "downside_evidence_status", _na),
            ("Capital Loss Underwriting Status", "capital_loss_underwriting_status", _na),
            ("Capital Loss Impairment Class", "capital_loss_impairment_class", _na),
            ("Capital Loss Underwriting Caution", "capital_loss_underwriting_caution", _na),
            ("Expected-Return Evidence Count", "expected_return_evidence_count", _na),
            ("Company-Specific Evidence Count", "company_specific_evidence_count", _na),
            (
                "Framework Required Evidence Coverage",
                "framework_required_evidence_coverage_ratio",
                _fmt_pct,
            ),
            ("Capital Structure Resolution Status", "capital_structure_resolution_status", _na),
            (
                "Capital Structure Resolution Summary",
                "capital_structure_resolution_summary",
                _line_break,
            ),
            ("Refinancing Timeline Status", "refinancing_timeline_status", _na),
            ("Debt Due Within 12 Months", "debt_due_within_12mo", _na),
            ("Going Concern Language", "going_concern_language", _na),
            ("No-Assurance Financing", "no_assurance_financing", _na),
            ("Cash Runway Quarters", "cash_runway_quarters", _na),
        ):
            lines.append(f"| {label} | {formatter(audit.get(key))} |")
        lines.append(
            f"| Capital Loss Reason Codes | {_join(audit.get('capital_loss_reason_codes') or [])} |"
        )
        lines.append(
            f"| Selected Company Evidence Tools | {_join(audit.get('selected_company_evidence_tools') or [])} |"
        )
        lines.append(
            f"| Selected Evidence Pillars | {_join(audit.get('selected_evidence_pillars') or [])} |"
        )
        lines.append(
            f"| Selected Risk Evidence Tools | {_join(audit.get('selected_risk_evidence_tools') or [])} |"
        )
        lines.append(
            f"| Framework Required Evidence | {_join(audit.get('framework_required_evidence') or [])} |"
        )
        lines.append(
            f"| Framework Required Evidence Covered | {_join(audit.get('framework_required_evidence_covered') or [])} |"
        )
        lines.append(
            f"| Framework Required Evidence Missing | {_join(audit.get('framework_required_evidence_missing') or [])} |"
        )
        lines.append(
            f"| Maturity Schedule | {_maturity_schedule(audit.get('maturity_schedule') or [])} |"
        )
        lines.append(f"| Covenant Terms | {_covenant_terms(audit.get('covenant_terms') or [])} |")
        lines.append(f"| Audit Gap Repair Attempted | {_na(artifact.audit_gap_repair_attempted)} |")
        lines.append(f"| Audit Gap Repair Status | {_na(artifact.audit_gap_repair_status)} |")
        lines.append(f"| Audit Gap Repair Notes | {_join(artifact.audit_gap_repair_notes)} |")
        lines.append(
            f"| Watchlist Resolution Attempted | {_na(artifact.watchlist_resolution_attempted)} |"
        )
        lines.append(
            f"| Watchlist Resolution Status | {_na(artifact.watchlist_resolution_status)} |"
        )
        lines.append(
            f"| Watchlist Resolution Notes | {_join(artifact.watchlist_resolution_notes)} |"
        )
        lines.append(
            f"| No-Selection Finalist Audit Attempted | {_na(artifact.no_selection_finalist_audit_attempted)} |"
        )
        lines.append(
            f"| No-Selection Finalist Audit Status | {_na(artifact.no_selection_finalist_audit_status)} |"
        )
        lines.append(
            f"| No-Selection Finalist Audit Focus | {_na(artifact.no_selection_finalist_audit_focus_ticker)} |"
        )
        lines.append(
            f"| No-Selection Finalist Audit Notes | {_join(artifact.no_selection_finalist_audit_notes)} |"
        )
        lines.append(
            f"| No-Selection Finalist Resolution Attempted | {_na(artifact.no_selection_finalist_resolution_attempted)} |"
        )
        lines.append(
            f"| No-Selection Finalist Resolution Status | {_na(artifact.no_selection_finalist_resolution_status)} |"
        )
        lines.append(
            f"| No-Selection Finalist Resolution Notes | {_join(artifact.no_selection_finalist_resolution_notes)} |"
        )
        lines.append(
            f"| Alternate Finalist Audit Attempted | {_na(artifact.alternate_finalist_audit_attempted)} |"
        )
        lines.append(
            f"| Alternate Finalist Audit Status | {_na(artifact.alternate_finalist_audit_status)} |"
        )
        lines.append(
            f"| Alternate Finalist Audit Notes | {_join(artifact.alternate_finalist_audit_notes)} |"
        )
        lines.append(
            f"| Alternate Finalist Audit Results | {_alternate_audit_results(artifact.alternate_finalist_audit_results)} |"
        )
        lines.append(f"| Company Autonomy Attempted | {_na(artifact.company_autonomy_attempted)} |")
        lines.append(f"| Company Autonomy Status | {_na(artifact.company_autonomy_status)} |")
        lines.append(f"| Company Autonomy Notes | {_join(artifact.company_autonomy_notes)} |")
        lines.append(f"| Hard Blockers | {_join(audit.get('hard_blockers') or [])} |")
        lines.append(f"| Confidence Caps | {_join(audit.get('confidence_caps') or [])} |")
        lines.append(f"| Notes | {_join(audit.get('notes') or [])} |")
        lines.append("")

    if _guardrail_outcome_applies(artifact):
        decision = artifact.final_decision
        blockers = decision.selection_blockers if decision else []
        lines.append("### Guardrail Outcome")
        lines.append("")
        lines.append("The binding artifact result is **NO_SELECTION**.")
        lines.append("")
        lines.append(f"- Selected ticker: {_na(artifact.selected_ticker)}")
        lines.append(f"- Confidence: {_na(artifact.confidence)}")
        lines.append(
            f"- No-selection reason: {_line_break(artifact.no_selection_reason or (decision.no_selection_reason if decision else None) or 'N/A')}"
        )
        if blockers:
            lines.append(f"- Selection blockers: {_join(blockers)}")
        lines.append("")

    if artifact.company_autonomy_attempted or artifact.company_autonomy_runs:
        lines.append("### Company Autonomy")
        lines.append("")
        lines.append(
            "| Ticker | Status | Child Verdict | Confidence | Tool Calls | Evidence | Degraded States |"
        )
        lines.append(
            "|--------|--------|---------------|------------|------------|----------|-----------------|"
        )
        for run in artifact.company_autonomy_runs:
            lines.append(
                "| "
                f"{_na(run.get('ticker'))} | {_na(run.get('status'))} | {_na(run.get('final_verdict'))} | "
                f"{_na(run.get('confidence'))} | {_na(run.get('tool_calls'))} | "
                f"{_na(run.get('evidence_references'))} | {_join(run.get('degraded_states') or [])} |"
            )
        if not artifact.company_autonomy_runs:
            lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        lines.append("")

    if artifact.relative_ranking:
        lines.append("### Relative Ranking")
        lines.append("")
        lines.append(
            "| Rank | Ticker | Base Return | Downside Return | Audit | Actionable | Child Verdict | Caps | Blockers | Positioning |"
        )
        lines.append(
            "|------|--------|-------------|-----------------|-------|------------|---------------|------|----------|-------------|"
        )
        for item in artifact.relative_ranking:
            lines.append(
                "| "
                f"{_na(item.get('rank'))} | {_na(item.get('ticker'))} | "
                f"{_fmt_pct(item.get('best_base_annualized_return'))} | "
                f"{_fmt_pct(item.get('downside_annualized_return'))} | "
                f"{_na(item.get('audit_status'))} | {_na(item.get('actionable'))} | "
                f"{_na(item.get('company_autonomy_verdict'))} | "
                f"{_cell(_join(item.get('confidence_caps') or []))} | "
                f"{_cell(_join(item.get('hard_blockers') or []))} | "
                f"{_cell(item.get('positioning_summary'))} |"
            )
        lines.append("")

    if artifact.company_packets:
        lines.append("### Company Financial Packets")
        lines.append("")
        lines.append(
            "| Ticker | Financial Status | Model Fit | Data Quality | Price | Anchor Method | Anchor | Discount | Blockers | Confidence Caps |"
        )
        lines.append(
            "|--------|------------------|-----------|--------------|-------|---------------|--------|----------|----------|-----------------|"
        )
        for packet in artifact.company_packets:
            valuation = packet.valuation or {}
            lines.append(
                "| "
                f"{packet.ticker} | {packet.financial_status} | {packet.model_fit_status} | "
                f"{packet.data_quality_status} | {_fmt_money(packet.current_price)} | "
                f"{_na(valuation.get('anchor_method'))} | {_fmt_money(valuation.get('valuation_anchor'))} | "
                f"{_fmt_pct(valuation.get('discount_to_anchor'))} | {_join(packet.blockers)} | "
                f"{_join(packet.confidence_caps)} |"
            )
        lines.append("")

    if artifact.expected_return_scenarios:
        lines.append("### Expected Return Scenarios")
        lines.append("")
        lines.extend(_scenario_summary_rows(artifact.expected_return_scenarios))
        lines.append("")

    if artifact.research_questions:
        lines.append("### Research Questions")
        lines.append("")
        lines.append(
            "| ID | Status | Priority | Pillar | Target Tickers | Planned Tools | Question |"
        )
        lines.append(
            "|----|--------|----------|--------|----------------|---------------|----------|"
        )
        for question in artifact.research_questions:
            lines.append(
                "| "
                f"{question.question_id} | {question.status} | {question.priority} | {question.financial_pillar} | "
                f"{_join(question.target_tickers)} | {_join(question.planned_tools)} | {_line_break(question.question)} |"
            )
        lines.append("")

    if artifact.tool_calls:
        lines.append("### Tool Calls")
        lines.append("")
        lines.append("| ID | Question | Tool | Status | Evidence | Error |")
        lines.append("|----|----------|------|--------|----------|-------|")
        for call in artifact.tool_calls:
            lines.append(
                "| "
                f"{call.call_id} | {_na(call.question_id)} | {call.tool_name} | {call.status} | "
                f"{_join(call.evidence_ref_ids)} | {_na(call.error)} |"
            )
        lines.append("")

    if artifact.evidence:
        lines.append("### Evidence References")
        lines.append("")
        lines.append("| ID | Source | Ticker | Confidence | Summary |")
        lines.append("|----|--------|--------|------------|---------|")
        for evidence in artifact.evidence:
            lines.append(
                "| "
                f"{evidence.evidence_id} | {evidence.source_label} | {_na(evidence.ticker)} | "
                f"{_na(evidence.confidence)} | {_line_break(evidence.summary)} |"
            )
        lines.append("")

    if artifact.belief_updates:
        lines.append("### Belief Updates")
        lines.append("")
        lines.append(
            "| ID | Question | Ticker | Direction | Confidence | Summary | Remaining Uncertainty |"
        )
        lines.append(
            "|----|----------|--------|-----------|------------|---------|-----------------------|"
        )
        for update in artifact.belief_updates:
            lines.append(
                "| "
                f"{update.update_id} | {_na(update.question_id)} | {_na(update.ticker)} | {update.direction} | "
                f"{update.confidence_after} | {_line_break(update.summary)} | {_join(update.remaining_uncertainty)} |"
            )
        lines.append("")

    if artifact.audit_notes:
        lines.append("### AI Working Notes And Runtime Audit")
        lines.append("")
        if _guardrail_outcome_applies(artifact):
            lines.append("These notes are pre/post-processing audit trail, not the final decision.")
            lines.append("")
        for note in artifact.audit_notes:
            lines.append(f"- {_line_break(note)}")
        lines.append("")

    return lines


def render_autonomous_sector_report(artifact: AutonomousSectorFinancialRunArtifact) -> str:
    """Render an autonomous sector run artifact as an investor-readable memo."""

    packets = _packet_by_ticker(artifact)
    candidate_tickers = _candidate_tickers(artifact)
    selected = artifact.selected_ticker or (
        artifact.final_decision.selected_ticker if artifact.final_decision else None
    )
    selection = artifact.candidate_selection or {}
    loaded = selection.get("loaded_tickers") or []
    excluded = selection.get("excluded_tickers") or []

    lines: list[str] = []
    lines.append(f"# {artifact.sector} Sector Research Memo")
    lines.append("")
    lines.append(f"**As of:** {artifact.as_of_date}  ")
    lines.append(f"**Market-cap focus:** {artifact.market_cap_focus}  ")
    lines.append(f"**Run ID:** {artifact.run_id}")
    lines.append("")
    lines.append("## Executive Conclusion")
    lines.append("")
    is_v2_incomplete = (
        getattr(artifact, "pipeline_version", "v1") == "v2"
        and getattr(artifact, "decision_status", None) == "INCOMPLETE"
    )
    if is_v2_incomplete:
        lines.append(
            "**Decision Incomplete.** This run preserves the broad research queue, but publishes "
            "neither a selection nor a no-selection conclusion while admitted candidates still "
            "need data, underwriting, or selected-company validation."
        )
    elif artifact.final_verdict == "SELECTED":
        lines.append(
            f"The sector run selects **{_na(selected)}** as the strongest current candidate, with "
            f"**{_na(artifact.confidence)}** confidence. "
            f"{_line_break(artifact.final_decision.thesis if artifact.final_decision else 'The selected company cleared the runtime guardrails.')} "
            f"The main risk is {_line_break(artifact.final_decision.key_risk if artifact.final_decision else 'the underwritten case degrading after fresh evidence')}."
        )
    elif artifact.final_verdict == "WATCHLIST":
        lines.append(
            f"The sector run does not mark a company actionable today; **{_na(selected)}** remains a watchlist candidate. "
            "The evidence supports continued monitoring, but at least one audit cap prevents a clean selection. "
            f"The gating issue is {_join_humanized((artifact.selection_audit or {}).get('confidence_caps') or [])}."
        )
    else:
        lines.append(
            "No company cleared the selection guardrails in this run. "
            f"{_line_break(_humanize_memo_text(artifact.no_selection_reason or (artifact.final_decision.no_selection_reason if artifact.final_decision else 'The run stopped because evidence gaps or blockers remained.')))} "
            "The correct product action is to keep the sector on the research queue rather than force a pick."
        )
    lines.append("")
    lines.append("## Candidate Coverage")
    lines.append("")
    admitted_count = len(getattr(artifact, "admitted_tickers", []) or [])
    loader_surfaced_count = len(loaded) or admitted_count or len(artifact.company_packets)
    selected_count = len(selection.get("selected_tickers") or [])
    memo_body = artifact.memo_body if isinstance(artifact.memo_body, dict) else {}
    memo_candidates = (
        memo_body.get("candidates") if isinstance(memo_body.get("candidates"), dict) else {}
    )
    if str(getattr(artifact, "pipeline_version", "v1") or "v1").lower() == "v1":
        from app.autonomous.sweep_delta import v1_terminal_coverage_from_artifact

        review_projection = v1_terminal_coverage_from_artifact(artifact)
        candidate_review_completed = len(review_projection["llm_candidate_review_completed"])
        candidate_review_failed = len(review_projection["llm_candidate_review_failed"])
    else:
        candidate_review_completed = sum(
            1
            for payload in memo_candidates.values()
            if isinstance(payload, dict)
            and str(payload.get("source") or "").lower() == "llm"
            and str(payload.get("status") or "").upper() == "OK"
        )
        candidate_review_failed = max(0, len(artifact.company_packets) - candidate_review_completed)
    prompt_context = (
        artifact.final_decision_prompt_context
        if isinstance(artifact.final_decision_prompt_context, dict)
        else {}
    )
    sector_context_count = len(prompt_context.get("prompt_scoped_tickers") or [])
    repair = (
        selection.get("data_gap_repair")
        if isinstance(selection.get("data_gap_repair"), dict)
        else {}
    )
    lines.append(
        f"The loader surfaced {loader_surfaced_count} candidates; data-gap repair examined "
        f"{int(repair.get('examined') or 0)}; the runtime retained "
        f"{selected_count or admitted_count or len(artifact.company_packets)} after candidate "
        f"filters; and built {len(artifact.company_packets)} financial packets. The per-ticker "
        f"LLM review completed for {candidate_review_completed} and failed or fell back for "
        f"{candidate_review_failed}. The shared sector-decision context included "
        f"{sector_context_count} ticker(s) (capped at 25 in v1). These are separate pipeline "
        "stages, not interchangeable 'reviewed' counts. "
        f"The candidate source was {_na(selection.get('source'))}, ranked by {_na(selection.get('ranking_basis'))}."
    )
    if excluded:
        lines.append(f"Excluded tickers: {_join(excluded)}.")
    if selection.get("warnings"):
        lines.append(
            f"Universe warnings: {_humanize_memo_text(_join(selection.get('warnings') or []))}."
        )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    framework = artifact.framework
    if framework:
        computed_metrics, declared_not_implemented = _framework_metric_usage(artifact)
        metric_sentence = f"Used metrics: {_join(computed_metrics)}."
        if declared_not_implemented:
            metric_sentence += (
                f" Declared but not yet implemented: {_join(declared_not_implemented)}."
            )
        lines.append(
            f"The run applied a sector-specific framework for {framework.economic_model} "
            f"{metric_sentence} It required "
            f"{_join(framework.required_evidence)} before treating a candidate as actionable. "
            f"It executed {len([call for call in artifact.tool_calls if call.status == 'OK'])} successful tool call(s), "
            f"kept guardrail/audit logic binding, and moved instrumentation to the appendix."
        )
    else:
        lines.append(
            f"The run compared sector candidates using deterministic packets, expected-return scenarios, "
            f"LLM-guided questions, and binding runtime guardrails. It executed "
            f"{len([call for call in artifact.tool_calls if call.status == 'OK'])} successful tool call(s)."
        )
    lines.append("")
    lines.append("## Top Candidates")
    lines.append("")
    lines.extend(_render_candidate_table(artifact))
    lines.append("")
    lines.extend(_render_cohort_comparison(artifact))
    lines.extend(_render_triage_surprises(artifact))
    lines.append("## Per-Candidate Sections")
    lines.append("")
    for ticker in candidate_tickers:
        lines.extend(_render_candidate_section(artifact, ticker, packets.get(ticker)))
    lines.extend(_render_v2_disposition_section(artifact))
    lines.extend(_render_decision_section(artifact))
    lines.extend(_render_audit_appendix(artifact))
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_autonomous_sector_report"]
