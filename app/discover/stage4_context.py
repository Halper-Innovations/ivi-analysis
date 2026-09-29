"""Real context builder and tool dispatcher for Stage 4 production runs.

Split from stage4.py so that stage4.py's tests can inject fakes without
pulling in the DB / bundle builder / companyfacts transitively.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.valuation.lineage import latest_decision_eligible_valuation_row


MAX_SECTION_CHARS = 30000
MAX_FACTS_ROWS = 40


def _packet_mapping(packet: Any | None) -> dict[str, Any]:
    if isinstance(packet, dict):
        return dict(packet)
    return dict(vars(packet)) if packet is not None else {}


def _canonical_scorecard_block(bundle, packet: Any | None) -> str:
    snap = bundle.valuation
    values = _packet_mapping(packet)
    return (
        f"current_price: {values.get('current_price', snap.current_price)}\n"
        f"market_cap: {values.get('market_cap_mm', snap.market_cap)}\n"
        f"dcf_base: {values.get('dcf_value', snap.dcf_base)}\n"
        f"epv_adjusted: {values.get('epv_value', snap.epv_adjusted)}\n"
        f"graham_value: {values.get('graham_value', snap.graham_value)}\n"
        f"methods_agree: {values.get('methods_agree', snap.methods_agree)}\n"
        f"tension_type: {values.get('method_tension_type', snap.tension_type)}\n"
        f"gate_action: {values.get('gate_verdict', snap.gate_action)}\n"
        f"solvency_status: {values.get('solvency_risk', snap.solvency_status)}\n"
        f"filing_risk_status: "
        f"{values.get('filing_risk_status', snap.filing_risk_status)}\n"
    )


def default_context_builder(ticker: str) -> dict[str, Any]:
    """Build the initial Stage 4 user message from the live evidence bundle."""
    from app.analyst.bundle_builder import build_analysis_evidence_bundle

    bundle = build_analysis_evidence_bundle(ticker, as_of_date=None)

    scorecard_block = _canonical_scorecard_block(bundle, None)

    filings_block = ""
    if bundle.filings:
        f0 = bundle.filings[0]
        filings_block = (
            f"\nMost recent filing: {f0.form_type} filed {f0.filing_date}, "
            f"accession {f0.accession}\n"
            f"sections_available: {list(f0.section_text.keys())}\n"
            "(Use fetch_filing_section to pull any untruncated section text.)"
        )

    warnings_block = ""
    if bundle.warnings:
        warnings_block = f"\nbundle_warnings: {bundle.warnings}"

    user_message = (
        f"=== TICKER: {ticker} ===\n"
        f"as_of_date: {bundle.as_of_date}\n"
        f"analysis_years: {bundle.analysis_years}{warnings_block}\n"
        f"\n=== VALUATION SCORECARD ===\n{scorecard_block}"
        f"{filings_block}\n"
        "\nUse the prompt first, then fetch only the missing evidence that matters. "
        "You do not need to call every tool. Call finalize_analysis when further "
        "evidence is unlikely to change the verdict."
    )
    return {"ticker": ticker, "user_message": user_message}


def _fetch_current_price(ticker: str) -> str:
    """Fetch live price via the configured market price provider."""
    from app.market.price_provider import get_default_provider
    from datetime import date

    provider = get_default_provider()
    snapshot = provider.get_price_asof(ticker, date.today().isoformat())
    if snapshot is None or snapshot.price is None:
        return f"ERROR: no price available for {ticker}"
    import math

    if math.isnan(snapshot.price):
        return f"ERROR: price returned NaN for {ticker}"
    return (
        f"ticker: {ticker}\n"
        f"price: ${snapshot.price:.2f}\n"
        f"as_of_date: {getattr(snapshot, 'as_of_date', 'unknown')}\n"
        f"source: {getattr(snapshot, 'source', 'yahoo')}\n"
    )


def default_tool_dispatcher(name: str, tool_input: dict[str, Any]) -> str:
    """Dispatch Stage 4 tool calls against the live DB / bundle builder."""
    try:
        if name == "fetch_filing_section":
            return _fetch_filing_section(tool_input["ticker"], tool_input["section_key"])
        if name == "fetch_historical_scorecards":
            return _fetch_historical_scorecards(tool_input["ticker"], int(tool_input["n_years"]))
        if name == "fetch_companyfacts":
            return _fetch_companyfacts(tool_input["ticker"], list(tool_input["line_items"]))
        if name == "fetch_current_price":
            return _fetch_current_price(tool_input["ticker"])
        return f"ERROR: unknown tool {name!r}"
    except Exception as exc:
        return f"ERROR: tool dispatch raised {type(exc).__name__}: {exc}"


def make_stage4_helpers(
    bundle_builder,
    stage3_lookup=None,
    financial_packets: dict[str, Any] | None = None,
):
    """Return (context_builder, tool_dispatcher) that route bundle construction
    through the injected ``bundle_builder`` callable instead of the slow
    ``build_analysis_evidence_bundle`` path.

    Used by ``ivi discover`` to plumb the cached-scorecard fast path into
    both the initial Stage 4 prompt (context_builder) and the
    fetch_filing_section tool (which otherwise rebuilds the full bundle
    every time it's called).

    The non-bundle tools (fetch_historical_scorecards, fetch_companyfacts)
    query engine.db directly and don't need a bundle builder — they are
    reused from the module-level helpers unchanged.

    Parameters
    ----------
    bundle_builder:
        Callable with signature ``(ticker, as_of_date=None) -> AnalysisEvidenceBundle``.
        In sweep contexts this is typically a closure over a
        ``{ticker: (scorecard_as_of_date, scorecard)}`` lookup that calls
        ``build_analysis_evidence_bundle_from_cached_scorecard`` for known
        tickers and falls back to ``build_analysis_evidence_bundle`` for
        unknown ones.
    stage3_lookup:
        Optional dict mapping ``{ticker: stage3_result_row}`` where each
        row has ``verdict``, ``thesis_summary``, ``open_questions_json``.
        When provided, Stage 4 context includes Stage 3's findings so the
        deep loop can resolve prior open questions instead of starting blind.

    Returns
    -------
    tuple[Callable[[str], dict], Callable[[str, dict], str]]
        ``(context_builder, tool_dispatcher)``
    """

    def context_builder(ticker: str) -> dict[str, Any]:
        bundle = bundle_builder(ticker, as_of_date=None)
        packet = (financial_packets or {}).get(ticker.upper())
        scorecard_block = _canonical_scorecard_block(bundle, packet)

        filings_block = ""
        if bundle.filings:
            f0 = bundle.filings[0]
            filings_block = (
                f"\nMost recent filing: {f0.form_type} filed {f0.filing_date}, "
                f"accession {f0.accession}\n"
                f"sections_available: {list(f0.section_text.keys())}\n"
                "(Use fetch_filing_section to pull any untruncated section text.)"
            )

        warnings_block = ""
        if bundle.warnings:
            warnings_block = f"\nbundle_warnings: {bundle.warnings}"

        # Stage 3 handoff context: include prior verdict, thesis, and open questions
        stage3_block = ""
        if stage3_lookup is not None:
            s3 = stage3_lookup.get(ticker.upper()) or stage3_lookup.get(ticker)
            if s3:
                import json as _json

                oq_raw = s3.get("open_questions_json", "[]")
                try:
                    open_qs = _json.loads(oq_raw) if isinstance(oq_raw, str) else oq_raw
                except _json.JSONDecodeError:
                    open_qs = []
                stage3_block = (
                    f"\n=== STAGE 3 HANDOFF ===\n"
                    f"Prior verdict: {s3.get('verdict', 'UNKNOWN')}\n"
                    f"Prior confidence: {s3.get('confidence', 'UNKNOWN')}\n"
                    f"Prior thesis: {s3.get('thesis_summary', '')}\n"
                )
                if open_qs:
                    stage3_block += "Open questions to resolve:\n"
                    for q in open_qs:
                        stage3_block += f"  - {q}\n"
                stage3_block += (
                    "\nYour job: use the tools to resolve the questions that matter most. "
                    "If the prior thesis is wrong, say so and explain why.\n"
                )

        user_message = (
            f"=== TICKER: {ticker} ===\n"
            f"as_of_date: {bundle.as_of_date}\n"
            f"analysis_years: {bundle.analysis_years}{warnings_block}\n"
            f"{stage3_block}"
            f"\n=== VALUATION SCORECARD ===\n{scorecard_block}"
            f"{filings_block}\n"
            "\nUse the prompt first, then fetch only the missing evidence that matters. "
            "You do not need to call every tool. Call finalize_analysis when further "
            "evidence is unlikely to change the verdict."
        )
        return {
            "ticker": ticker,
            "as_of_date": bundle.as_of_date,
            "user_message": user_message,
        }

    def _fast_fetch_filing_section(ticker: str, section_key: str) -> str:
        """fetch_filing_section using the injected bundle_builder."""
        bundle = bundle_builder(ticker, as_of_date=None)
        if not bundle.filings:
            return "ERROR: no cached filings for this ticker"
        filing = bundle.filings[0]
        if section_key not in filing.section_text:
            return (
                f"Section '{section_key}' not present. Available: "
                f"{list(filing.section_text.keys())}"
            )
        text = filing.section_text[section_key]
        if len(text) > MAX_SECTION_CHARS:
            text = (
                text[:MAX_SECTION_CHARS]
                + f"\n\n... [truncated {len(text) - MAX_SECTION_CHARS} chars]"
            )
        return (
            f"[{filing.form_type} filed {filing.filing_date}, "
            f"accession {filing.accession}]\n\n{text}"
        )

    def tool_dispatcher(name: str, tool_input: dict[str, Any]) -> str:
        try:
            if name == "fetch_filing_section":
                return _fast_fetch_filing_section(tool_input["ticker"], tool_input["section_key"])
            if name == "fetch_historical_scorecards":
                return (
                    "ERROR: historical scorecards are not authorized in the "
                    "integrity-bound discover lane"
                )
            if name == "fetch_companyfacts":
                packet = _packet_mapping(
                    (financial_packets or {}).get(str(tool_input["ticker"]).upper())
                )
                return _fetch_companyfacts(
                    tool_input["ticker"],
                    list(tool_input["line_items"]),
                    as_of_date=str(packet.get("current_price_as_of_date") or "")[:10] or None,
                )
            if name == "fetch_current_price":
                packet = _packet_mapping(
                    (financial_packets or {}).get(str(tool_input["ticker"]).upper())
                )
                if not packet:
                    return "ERROR: no authorized canonical price packet"
                return (
                    f"ticker: {str(tool_input['ticker']).upper()}\n"
                    f"price: ${float(packet['current_price']):.2f}\n"
                    f"as_of_date: {packet.get('current_price_as_of_date')}\n"
                    f"source: {packet.get('current_price_source')}\n"
                    f"quote_snapshot_id: {packet.get('quote_snapshot_id')}\n"
                )
            return f"ERROR: unknown tool {name!r}"
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            return f"ERROR: tool dispatch raised {type(exc).__name__}: {exc}"

    return context_builder, tool_dispatcher


def _fetch_filing_section(ticker: str, section_key: str) -> str:
    from app.analyst.bundle_builder import build_analysis_evidence_bundle

    bundle = build_analysis_evidence_bundle(ticker, as_of_date=None)
    if not bundle.filings:
        return "ERROR: no cached filings for this ticker"
    filing = bundle.filings[0]
    if section_key not in filing.section_text:
        return f"Section '{section_key}' not present. Available: {list(filing.section_text.keys())}"
    text = filing.section_text[section_key]
    if len(text) > MAX_SECTION_CHARS:
        text = (
            text[:MAX_SECTION_CHARS] + f"\n\n... [truncated {len(text) - MAX_SECTION_CHARS} chars]"
        )
    return (
        f"[{filing.form_type} filed {filing.filing_date}, accession {filing.accession}]\n\n{text}"
    )


def _fetch_historical_scorecards(ticker: str, n_years: int) -> str:
    cfg = get_config()
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    try:
        limit = max(1, min(n_years, 10))
        rows: list[Any] = []
        before_as_of_date: str | None = None
        while len(rows) < limit:
            row = latest_decision_eligible_valuation_row(
                conn,
                ticker=ticker.upper(),
                method="scorecard",
                before_as_of_date=before_as_of_date,
            )
            if row is None:
                break
            rows.append(row)
            row_as_of_date = str(row["as_of_date"] or "").strip()
            if not row_as_of_date:
                break
            before_as_of_date = row_as_of_date
    finally:
        conn.close()

    if not rows:
        return f"No scorecards found for {ticker}"
    out: list[str] = []
    for r in rows:
        try:
            sc = json.loads(r["outputs_json"] or "{}")
            pzd = sc.get("pricing_zone_detail") or {}
            out.append(
                f"{r['as_of_date']}: zone={sc.get('pricing_zone', '?')} "
                f"price={pzd.get('current_price')} dcf={pzd.get('dcf_base')} "
                f"epv={pzd.get('epv_adjusted')} gate={pzd.get('gate_action')} "
                f"cycle={pzd.get('cycle_position')}"
            )
        except json.JSONDecodeError:
            out.append(f"{r['as_of_date']}: <unparseable>")
    return "\n".join(out)


# Translate raw XBRL element names → normalized line_item names that are
# actually stored in companyfacts_facts. The database stores normalized names
# only; older versions of the tool advertised raw XBRL names which produced
# 0% hit rate. We accept both for backwards compatibility but the tool
# description now points the model at normalized names.
_XBRL_TO_NORMALIZED: dict[str, str] = {
    # Income statement
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "SalesRevenueGoodsNet": "revenue",
    "NetIncomeLoss": "net_income",
    "ProfitLoss": "net_income",
    "OperatingIncomeLoss": "operating_income",
    "GrossProfit": "gross_profit",
    "InterestExpense": "interest_expense",
    "ResearchAndDevelopmentExpense": "r_and_d_total",
    "ShareBasedCompensation": "sbc",
    "AllocatedShareBasedCompensationExpense": "sbc",
    "RestructuringCharges": "restructuring_charges",
    "DepreciationAndAmortization": "depreciation_amortization",
    "DepreciationAmortizationAndAccretionNet": "depreciation_amortization",
    "DepreciationDepletionAndAmortization": "depreciation_amortization",
    "Depreciation": "depreciation",
    "AmortizationOfIntangibleAssets": "depreciation_amortization",
    # Cash flow
    "NetCashProvidedByUsedInOperatingActivities": "cfo",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": "cfo",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "PaymentsToAcquireProductiveAssets": "capex",
    "PaymentsForRepurchaseOfCommonStock": "share_repurchases_amount",
    "PaymentsOfDividends": "dividends_paid_amount",
    "PaymentsOfDividendsCommonStock": "dividends_paid_amount",
    # Balance sheet
    "Cash": "cash",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "cash",
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
    "StockholdersEquity": "equity",
    "AssetsCurrent": "current_assets",
    "LiabilitiesCurrent": "current_liabilities",
    "AccountsReceivableNetCurrent": "accounts_receivable",
    "AccountsPayableCurrent": "accounts_payable",
    "InventoryNet": "inventory",
    "Goodwill": "goodwill",
    "IntangibleAssetsNetExcludingGoodwill": "intangible_assets",
    "FiniteLivedIntangibleAssetsNet": "intangible_assets",
    "MarketableSecurities": "investment_securities",
    "AvailableForSaleSecurities": "investment_securities",
    "ContractWithCustomerLiabilityCurrent": "deferred_revenue",
    "DeferredRevenue": "deferred_revenue",
    "PropertyPlantAndEquipmentGross": "gross_ppe",
    "OperatingLeaseLiability": "operating_lease_liability",
    # Debt — most common XBRL → normalized total_debt
    "LongTermDebt": "total_debt",
    "LongTermDebtNoncurrent": "total_debt",
    "DebtCurrent": "total_debt",
    "ShortTermBorrowings": "total_debt",
    "Debt": "total_debt",
    # Shares
    "EntityCommonStockSharesOutstanding": "shares_outstanding",
    "CommonStockSharesOutstanding": "shares_outstanding",
}


def _normalize_line_items(line_items: list[str]) -> tuple[list[str], dict[str, str]]:
    """Map any raw XBRL names to normalized; return (normalized list, translations applied)."""
    normalized: list[str] = []
    translations: dict[str, str] = {}
    for raw in line_items:
        s = (raw or "").strip()
        if not s:
            continue
        if s in _XBRL_TO_NORMALIZED:
            n = _XBRL_TO_NORMALIZED[s]
            translations[s] = n
            if n not in normalized:
                normalized.append(n)
        else:
            if s not in normalized:
                normalized.append(s)
    return normalized, translations


# All normalized line_items present in the database. Used for the on-miss
# suggestion list so the AI sees what's actually queryable.
_KNOWN_NORMALIZED_ITEMS = sorted(
    {
        "revenue",
        "net_income",
        "operating_income",
        "gross_profit",
        "cfo",
        "capex",
        "depreciation_amortization",
        "depreciation",
        "sbc",
        "share_repurchases_amount",
        "dividends_paid_amount",
        "cash",
        "total_debt",
        "total_assets",
        "total_liabilities",
        "equity",
        "current_assets",
        "current_liabilities",
        "accounts_receivable",
        "accounts_payable",
        "inventory",
        "goodwill",
        "intangible_assets",
        "investment_securities",
        "deferred_revenue",
        "gross_ppe",
        "operating_lease_liability",
        "shares_outstanding",
        "interest_expense",
        "r_and_d_total",
        "restructuring_charges",
    }
)


def _fetch_companyfacts(
    ticker: str,
    line_items: list[str],
    *,
    as_of_date: str | None = None,
) -> str:
    # Translate raw XBRL names to normalized, but pass through unknown strings
    # so a user that already supplied normalized names hits the DB directly.
    normalized_items, translations = _normalize_line_items(line_items)
    if not normalized_items:
        return "ERROR: line_items is empty"

    cfg = get_config()
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    try:
        from app.util.financial_data_access import companyfacts_rows

        rows = companyfacts_rows(
            conn,
            ticker.upper(),
            columns=(
                "line_item",
                "fiscal_year",
                "period_type",
                "value",
                "units",
                "filed_date",
            ),
            period_types=("FY",),
            line_items=normalized_items,
            as_of_date=as_of_date,
            value_not_null=True,
            require_filed_asof=as_of_date is not None,
            order_by="line_item, fiscal_year DESC",
        )
    except sqlite3.OperationalError as exc:
        conn.close()
        return f"ERROR: companyfacts query failed: {exc}"
    finally:
        conn.close()

    if not rows:
        # Be specific: tell the AI exactly what's available so the next call lands.
        unknown = [it for it in normalized_items if it not in _KNOWN_NORMALIZED_ITEMS]
        unknown_msg = f"\nUnknown (not in cache) line_items: {unknown}." if unknown else ""
        return (
            f"No data for {ticker} with line_items {normalized_items}.{unknown_msg}\n"
            f"Use NORMALIZED names from this list (case-sensitive):\n  "
            + ", ".join(_KNOWN_NORMALIZED_ITEMS)
        )

    by_item: dict[str, list[tuple[int, float, str, str | None]]] = {}
    for r in rows:
        by_item.setdefault(r["line_item"], []).append(
            (
                r["fiscal_year"],
                r["value"],
                str(r["units"] or "UNKNOWN"),
                r["filed_date"],
            )
        )

    out: list[str] = []
    if translations:
        out.append(
            "(Translated raw XBRL → normalized: "
            + ", ".join(f"{k}→{v}" for k, v in translations.items())
            + ")"
        )
    for item, series in by_item.items():
        series.sort(key=lambda x: x[0], reverse=True)
        series = series[:MAX_FACTS_ROWS]
        formatted = ", ".join(
            f"{year}: {val:,.0f} {units} (filed {filed or 'UNKNOWN'})"
            for year, val, units, filed in series
        )
        out.append(f"{item}: {formatted}")
    return "\n".join(out)
