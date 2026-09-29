from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.config import get_config
from app.market.company_facts_extract import (
    DEBT_COMPLETENESS_TAGS,
    SHORT_TERM_INVESTMENT_TAGS,
    TOTAL_DEBT_FALLBACK_TAG_PRIORITY,
    resolve_complete_total_debt,
    short_term_investments_addition,
    short_term_investments_derivation,
)
from app.market.shares_guard import check_share_count
from app.util.http import HttpClient

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

TAG_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "income_continuing": [
        "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
        "IncomeLossFromContinuingOperations",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurringObligation",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquirePremisesAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
        "CashAndCashEquivalents",
        "CashEquivalentsAtCarryingValue",
        "CashAndDueFromBanks",
        # Restricted-cash-INCLUSIVE tag ranks LAST, matching the as-of
        # extractor (audit: cash-tag-order-restricted-cash-divergence) —
        # restricted cash is not available cash for net-debt purposes.
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    # What counts beside "cash" in net debt: CURRENT short-term investments at the
    # cash row's own balance-sheet date, net of any already inside the cash figure
    # (resolved from the chosen cash row by _resolve_short_term_investment_rows).
    "short_term_investments": list(SHORT_TERM_INVESTMENT_TAGS),
    # One fallback list with the as-of extractor (app.market.company_facts_extract), which
    # documents it. DebtCurrent comes last: it is read so the chain can sum it with a
    # noncurrent line, and alone it is only the current debt (completed or refused below).
    # DebtInstrumentCarryingAmount is deliberately absent: an instrument-level concept
    # (one note, one facility) is never an issuer's complete debt.
    "total_debt": [
        *(tag for _taxonomy, tag in TOTAL_DEBT_FALLBACK_TAG_PRIORITY),
        "DebtCurrent",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    # The consolidated total on its own, so a consumer can tell whether "equity"
    # above is the parent-only figure (it is whenever the two differ).
    "equity_including_nci": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarningsUnappropriated",
        "AccumulatedDeficit",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockOtherSharesOutstanding",
    ],
    "r_and_d_total": [
        "ResearchAndDevelopmentExpense",
    ],
    "deferred_revenue": [
        "ContractWithCustomerLiability",
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenue",
        "DeferredRevenueCurrent",
    ],
    "depreciation": [
        "Depreciation",
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
    ],
    "share_repurchases_amount": [
        "PaymentsForRepurchaseOfCommonStock",
    ],
    "dividends_paid_amount": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ],
    "deposits": [
        "Deposits",
        "DepositsLiabilities",
    ],
    "loans": [
        "LoansAndLeasesReceivableNetReportedAmount",
        "LoansReceivableNetReportedAmount",
        "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
        "PrincipalAmountOutstandingOfLoansHeldInPortfolio",
        "LoansAndLeasesReceivableNetOfDeferredIncome",
        "LoansReceivableNet",
        "LoansReceivableHeldForInvestmentNetOfAllowance",
        "LoansAndLeasesHeldForInvestmentNetOfAllowance",
        "FinancingReceivableRecordedInvestment",
        "ReceivablesNetCurrentAndNoncurrent",
    ],
    "investment_securities": [
        "AvailableForSaleSecuritiesDebtSecurities",
        "AvailableForSaleDebtSecurities",
        "DebtSecuritiesAvailableForSaleAmortizedCost",
    ],
    "total_assets": [
        "Assets",
    ],
    "assets_under_management": [
        "AssetsUnderManagement",
    ],
    "allowance_for_credit_losses": [
        "AllowanceForCreditLosses",
        "AllowanceForLoanAndLeaseLosses",
        "AllowanceForCreditLossesOnFinancingReceivables",
        "AllowanceForCreditLossesOnLoansAndLeases",
        "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
        "FinancingReceivableAllowanceForCreditLosses",
        "LoansAndLeasesReceivableAllowance",
    ],
    "provision_for_credit_losses": [
        "ProvisionForLoanLeaseAndOtherLosses",
        "ProvisionForLoanAndLeaseLosses",
        "ProvisionForLoanLossesExpensed",
        "AllowanceForLoanAndLeaseLossesProvisionForLossNet",
    ],
    "net_charge_offs": [
        "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery",
        "FinancingReceivableAllowanceForCreditLossWriteoffAfterRecovery",
    ],
    "nonaccrual_loans": [
        "FinancingReceivableRecordedInvestmentNonaccrualStatus",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",
    ],
    "sbc": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        "SellingGeneralAndAdministrativeExpenseExcludingDepreciationDepletionAndAmortization",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    # ── EV->equity bridge senior claims (audit coverage gap 2) ────────────────
    "preferred_equity": [
        "PreferredStockValue",
        "PreferredStockValueOutstanding",
        "TemporaryEquityCarryingAmountAttributableToParent",
    ],
    "noncontrolling_interest": [
        "MinorityInterest",
        "RedeemableNoncontrollingInterestEquityCarryingAmount",
    ],
    # ── Wave 2: Balance sheet integrity items ─────────────────────────────────
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
        "ReceivablesNetCurrent",
    ],
    "inventory": [
        "InventoryNet",
        "InventoryFinishedGoods",
        "Inventories",
    ],
    "accounts_payable": [
        "AccountsPayableCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "gross_ppe": [
        "PropertyPlantAndEquipmentGross",
        "PropertyPlantAndEquipmentNet",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    # Intangible amortization is captured SEPARATELY from depreciation_amortization
    # because for serial acquirers (COLL, JAZZ, ANIP) it dominates total D&A by
    # 50-100x. Combining it into depreciation_amortization (which uses
    # first-tag-wins semantics) would either lose the depreciation signal or
    # lose the amortization signal depending on which tag fires first.
    # Surfacing both as separate fields lets the AI and EPV cash-adjustment
    # module reason about each component correctly.
    "intangible_amortization": [
        "AmortizationOfIntangibleAssets",
        "AmortizationOfAcquiredIntangibleAssets",
    ],
    "goodwill": [
        "Goodwill",
    ],
    "intangible_assets": [
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
    ],
    "operating_lease_liability": [
        "OperatingLeaseLiability",
        "OperatingLeaseLiabilityNoncurrent",
        "OperatingLeaseLiabilityCurrent",
    ],
    "restructuring_charges": [
        "RestructuringCharges",
        "RestructuringAndRelatedCostIncurredCost",
    ],
    # ── Quality-metric inputs (ROIC effective-tax-rate + gross margin) ─────
    # income_tax_expense / pretax_income feed _effective_tax_rate -> ROIC NOPAT.
    # cost_of_revenue feeds the revenue-minus-cost gross-margin fallback. These
    # us-gaap tags are present in the raw cache (IncomeTaxExpenseBenefit ~91%,
    # the pretax tag ~88%) but were never ingested, so ROIC was 0% computable.
    "income_tax_expense": [
        "IncomeTaxExpenseBenefit",
        "IncomeTaxExpenseBenefitContinuingOperations",
        "CurrentIncomeTaxExpenseBenefit",
    ],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
}

PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "revenue": (0.01, 10_000_000.0),
    "gross_profit": (-500_000.0, 10_000_000.0),
    "operating_income": (-500_000.0, 5_000_000.0),
    "net_income": (-500_000.0, 5_000_000.0),
    "income_continuing": (-500_000.0, 5_000_000.0),
    "cfo": (-500_000.0, 5_000_000.0),
    "capex": (-500_000.0, 2_000_000.0),
    "cash": (0.0, 5_000_000.0),
    "total_debt": (0.0, 5_000_000.0),
    "short_term_investments": (0.0, 5_000_000.0),
    "equity": (-2_000_000.0, 10_000_000.0),
    "equity_including_nci": (-2_000_000.0, 10_000_000.0),
    "retained_earnings": (-5_000_000.0, 10_000_000.0),
    # Floor 0.01 (10,000 shares): genuine sub-1M counts (post heavy
    # reverse splits) must convert via /1e6, not be rejected or "rescued"
    # as thousands; counts below 10k shares are treated as data corruption.
    "shares_outstanding": (0.01, 1_000_000.0),
    "r_and_d_total": (0.0, 1_000_000.0),
    "deferred_revenue": (0.0, 5_000_000.0),
    "depreciation": (0.0, 1_000_000.0),
    "share_repurchases_amount": (0.0, 2_000_000.0),
    "dividends_paid_amount": (0.0, 2_000_000.0),
    "deposits": (0.0, 10_000_000.0),
    "loans": (0.0, 10_000_000.0),
    "investment_securities": (0.0, 10_000_000.0),
    "total_assets": (0.0, 15_000_000.0),
    "assets_under_management": (0.0, 50_000_000.0),
    "allowance_for_credit_losses": (0.0, 1_000_000.0),
    "provision_for_credit_losses": (-1_000_000.0, 1_000_000.0),
    "net_charge_offs": (-1_000_000.0, 1_000_000.0),
    "nonaccrual_loans": (0.0, 10_000_000.0),
    "interest_expense": (0.0, 1_000_000.0),
    "sbc": (0.0, 1_000_000.0),
    "sga": (0.0, 5_000_000.0),
    "total_liabilities": (0.0, 5_000_000.0),  # $5 trillion — covers largest banks (JPMorgan ~$3.7T)
    "preferred_equity": (0.0, 1_000_000.0),
    "noncontrolling_interest": (-100_000.0, 1_000_000.0),
    # ── Wave 2: Balance sheet integrity items ─────────────────────────────────
    "accounts_receivable": (0.0, 500_000.0),
    "inventory": (0.0, 200_000.0),
    "accounts_payable": (0.0, 300_000.0),
    "current_assets": (0.0, 1_000_000.0),
    "current_liabilities": (0.0, 1_000_000.0),
    "gross_ppe": (0.0, 1_000_000.0),
    "depreciation_amortization": (0.0, 200_000.0),
    "intangible_amortization": (0.0, 200_000.0),
    "goodwill": (0.0, 500_000.0),
    "intangible_assets": (0.0, 500_000.0),
    "operating_lease_liability": (0.0, 200_000.0),
    "restructuring_charges": (0.0, 50_000.0),
    # ── Signed bounds — tax can be a benefit (negative), pretax a loss ─────
    "income_tax_expense": (-2_000_000.0, 2_000_000.0),
    "pretax_income": (-2_000_000.0, 5_000_000.0),
    "cost_of_revenue": (0.0, 10_000_000.0),
}

_MIN_ANNUAL_DAYS = 350
_MAX_ANNUAL_DAYS = 380
_ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
_QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
_QUARTERLY_FPS = {"Q1", "Q2", "Q3", "Q4"}
_MIN_QUARTERLY_DAYS = 80
_MAX_QUARTERLY_DAYS = 100
_SHARES_TAG_PRIORITY: list[tuple[str, str]] = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockOtherSharesOutstanding"),
]
_DIRECT_TOTAL_DEBT_TAG_PRIORITY: list[str] = [
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtAndCapitalLeaseObligations",
]
_TOTAL_DEBT_COMPONENT_TAGS = {
    "current": "DebtCurrent",
    "noncurrent": "LongTermDebtNoncurrent",
}
_TOTAL_DEBT_FALLBACK_TAG_PRIORITY: list[str] = [
    tag for tag in TAG_MAP["total_debt"] if tag not in _DIRECT_TOTAL_DEBT_TAG_PRIORITY
]
_LEASE_DIRECT_TAG_PRIORITY: list[str] = ["OperatingLeaseLiability"]
_LEASE_COMPONENT_TAGS = {
    "current": "OperatingLeaseLiabilityCurrent",
    "noncurrent": "OperatingLeaseLiabilityNoncurrent",
}


def _fetch_raw(cik: str) -> dict[str, Any]:
    cfg = get_config()
    http = HttpClient(cfg)
    url = COMPANYFACTS_URL.format(cik=cik)
    return http.get_json(url, use_cache=True, cache_ttl_seconds=24 * 3600)


def _to_millions(value: float, line_item: str) -> float | None:
    converted = value / 1_000_000.0
    lo, hi = PLAUSIBILITY.get(line_item, (None, None))
    if lo is not None and not (lo <= converted <= hi):
        # No thousands "rescue" for shares: the three ingested outstanding-
        # shares tags are raw counts in SEC XBRL, and the rescue misread any
        # genuine sub-1M count (post-reverse-split nanocaps) as thousands —
        # 1000x too many shares (audit: share-scale-1000x-sub-million-counts).
        return None
    return converted


def _period_days(start: str, end: str) -> int:
    try:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        return (e - s).days
    except Exception:
        return 0


def _fiscal_year(end_date: str) -> int:
    return int(end_date[:4])


def _tag_priority(line_item: str) -> list[tuple[str, str]]:
    if line_item == "shares_outstanding":
        return list(_SHARES_TAG_PRIORITY)
    return [("us-gaap", tag) for tag in TAG_MAP.get(line_item, [])]


def _normalization_cutoff_year(*, years_back: int, filed_as_of: str | None) -> int:
    if filed_as_of:
        try:
            return date.fromisoformat(str(filed_as_of)[:10]).year - years_back
        except ValueError:
            pass
    return datetime.now(timezone.utc).year - years_back


def _fact_visible_as_filed(fact: dict[str, Any], filed_as_of: str | None) -> bool:
    if filed_as_of is None:
        return True
    filed = str(fact.get("filed") or "").strip()[:10]
    try:
        return bool(filed) and date.fromisoformat(filed) <= date.fromisoformat(
            str(filed_as_of)[:10]
        )
    except ValueError:
        return False


def _iter_normalized_tag_facts(
    facts_root: dict[str, Any],
    *,
    taxonomy: str,
    tag: str,
    line_item: str,
    annual: bool,
    cutoff_year: int,
    filed_as_of: str | None = None,
):
    taxonomy_node = (facts_root.get(taxonomy) or {}) if isinstance(facts_root, dict) else {}
    tag_data = taxonomy_node.get(tag) if isinstance(taxonomy_node, dict) else None
    if not tag_data:
        return
    units_dict = tag_data.get("units") or {}
    unit_key = "shares" if line_item == "shares_outstanding" else "USD"
    facts = units_dict.get(unit_key) or []

    for fact in facts:
        # V2 fixed-as-of reads fail closed on undated facts and future
        # restatements. Legacy callers omit filed_as_of and preserve the
        # historical normalizer behavior exactly.
        if not _fact_visible_as_filed(fact, filed_as_of):
            continue
        form = fact.get("form") or ""
        end = fact.get("end") or ""
        if not end:
            continue
        start = fact.get("start")
        if annual:
            if form not in _ANNUAL_FORMS:
                continue
            if start:
                days = _period_days(start, end)
                if not (_MIN_ANNUAL_DAYS <= days <= _MAX_ANNUAL_DAYS):
                    continue
            period_type = "FY"
            if (
                taxonomy == "dei"
                and line_item == "shares_outstanding"
                and fact.get("fy") is not None
            ):
                # dei cover-page instants are dated at the FILING date,
                # months after FY end — labeling by end-date year shifted a
                # Dec-FYE company's share count to FY+1 (audit:
                # dei-shares-fiscal-year-label-shift). The filing's fy
                # metadata is the fiscal year actually covered.
                try:
                    fiscal_year = int(fact["fy"])
                except (TypeError, ValueError):
                    fiscal_year = _fiscal_year(end)
            else:
                fiscal_year = _fiscal_year(end)
        else:
            if form not in _QUARTERLY_FORMS:
                continue
            fp = fact.get("fp") or ""
            if fp not in _QUARTERLY_FPS:
                continue
            if start:
                days = _period_days(start, end)
                if not (_MIN_QUARTERLY_DAYS <= days <= _MAX_QUARTERLY_DAYS):
                    continue
            raw_fy = fact.get("fy")
            fiscal_year = int(raw_fy) if raw_fy is not None else _fiscal_year(end)
            period_type = fp
        if fiscal_year < cutoff_year:
            continue
        raw_val = fact.get("val")
        if raw_val is None:
            continue
        normalized = _to_millions(float(raw_val), line_item)
        if normalized is None:
            continue
        yield {
            "taxonomy": taxonomy,
            "tag": tag,
            "fiscal_year": fiscal_year,
            "period_start": str(start or ""),
            "period_end": end,
            "period_type": period_type,
            "value": normalized,
            "raw_value": float(raw_val),
            "form": form,
            "filed": fact.get("filed") or "",
            "accession": fact.get("accn") or "",
        }


def _track_share_alternate(
    alternates: dict[Any, dict[str, dict[str, Any]]], key: Any, candidate: dict[str, Any]
) -> None:
    """Per period and share tag, the candidate the first-tag-wins loop would keep for that
    tag (first yielded, replaced only by a same-tag restatement)."""
    by_tag = alternates.setdefault(key, {})
    existing = by_tag.get(str(candidate["tag"]))
    if existing is None or _prefer_restatement(existing, candidate):
        by_tag[str(candidate["tag"])] = candidate


def _guard_share_choices(
    raw: dict[str, Any],
    chosen: dict[Any, dict[str, Any]],
    order: list[Any],
    alternates: dict[Any, dict[str, dict[str, Any]]],
) -> None:
    """Apply the share-count guard to each period's share pick, in place.

    The pick is the first share tag in _SHARES_TAG_PRIORITY with a candidate for the period
    (usually the dei cover count). It is judged by app.market.shares_guard.check_share_count
    as of its own filing: same-filing references, then the tag's history. A refused pick
    falls to the next tag's candidate for the SAME period, judged the same way; when none
    survives the period has no share row (UNKNOWN), never an unchecked substitute. A pick
    the guard accepts is kept as it was, so companies the guard does not touch are
    unchanged.
    """
    for key in list(order):
        if key[0] != "shares_outstanding":
            continue
        accepted: dict[str, Any] | None = None
        for taxonomy, tag in _SHARES_TAG_PRIORITY:
            candidate = alternates.get(key, {}).get(tag)
            if candidate is None:
                continue
            judged = check_share_count(
                raw,
                taxonomy=taxonomy,
                tag=tag,
                value=float(candidate.get("raw_value", float(candidate["value"]) * 1e6)),
                end=str(candidate["period_end"]),
                filed=str(candidate.get("filed") or ""),
                accn=str(candidate.get("accession") or ""),
            )
            if judged["decision"] == "accept":
                accepted = candidate
                break
        if accepted is None:
            del chosen[key]
            order.remove(key)
        else:
            chosen[key] = accepted


# The fiscal-year share count of last resort: an issuer whose outstanding count
# exists only per share class (Ford's Class A and Class B cover-page counts) has
# no undimensioned outstanding-count fact in companyfacts at all, since the SEC
# API drops dimensioned facts. Its year's diluted weighted-average count is the
# closest reported figure: every class, and more shares than basic when there
# is dilution.
_WEIGHTED_SHARES_FALLBACK_TAG = ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")


def _weighted_share_fallback(
    raw: dict[str, Any],
    facts_root: dict[str, Any],
    chosen: dict[Any, dict[str, Any]],
    order: list[Any],
    *,
    reported: set[Any],
    cutoff_year: int,
    filed_as_of: str | None,
) -> None:
    """Fill each fiscal year's share count from the diluted weighted average, in place.

    Decided per fiscal year: only a year for which NO outstanding-count tag (dei cover
    count, balance-sheet count) reported anything -- a key absent from ``reported`` -- is
    filled, so it never overrides or stands in for a count the share guard refused. It used
    to be all-or-nothing for the window, and one old cover count left Ford's FY2011-2019
    without any share count. Each fallback count is judged by the same guard as of its own
    filing; a refused one leaves that year UNKNOWN.
    """
    taxonomy, tag = _WEIGHTED_SHARES_FALLBACK_TAG
    picks: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in _iter_normalized_tag_facts(
        facts_root,
        taxonomy=taxonomy,
        tag=tag,
        line_item="shares_outstanding",
        annual=True,
        cutoff_year=cutoff_year,
        filed_as_of=filed_as_of,
    ):
        key = ("shares_outstanding", int(candidate["fiscal_year"]))
        picks[key] = _keep_restatement(picks.get(key), candidate)
    for key in sorted(picks):
        if key in chosen or key in reported:
            continue
        candidate = picks[key]
        judged = check_share_count(
            raw,
            taxonomy=taxonomy,
            tag=tag,
            value=float(candidate["raw_value"]),
            end=str(candidate["period_end"]),
            filed=str(candidate.get("filed") or ""),
            accn=str(candidate.get("accession") or ""),
        )
        if judged["decision"] == "accept":
            chosen[key] = candidate
            order.append(key)


def _source(candidate: dict[str, Any]) -> dict[str, Any]:
    """Provenance of a single-fact row: the XBRL concept it was read from."""
    return {
        "taxonomy": str(candidate.get("taxonomy") or "us-gaap"),
        "tag": str(candidate.get("tag") or ""),
        "period_start": str(candidate.get("period_start") or ""),
    }


def _summed_source(
    components: list[dict[str, Any]], *, line_item: str | None = None
) -> dict[str, Any]:
    """Provenance of a row summed from several facts: no single tag; the components
    (each with its own tag, value in USD millions, and filing), in the line item's tag
    priority order when it has one."""
    if len(components) == 1:
        return _source(components[0])
    order = {tag: index for index, (_tax, tag) in enumerate(_tag_priority(line_item or ""))}
    ordered = sorted(components, key=lambda c: order.get(str(c.get("tag") or ""), len(order)))
    return {
        "taxonomy": None,
        "tag": None,
        "period_start": "",
        "components": [
            {
                "taxonomy": str(c.get("taxonomy") or "us-gaap"),
                "tag": str(c.get("tag") or ""),
                "value": float(c["value"]),
                "filed_date": str(c.get("filed") or ""),
                "form": str(c.get("form") or ""),
                "accession": str(c.get("accession") or ""),
            }
            for c in ordered
        ],
    }


def _resolve_component_summed_rows(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int,
    annual: bool,
    line_item: str,
    direct_tag_priority: list[str],
    component_tags: dict[str, str],
    fallback_tag_priority: list[str],
    sum_partial_components: bool = False,
    filed_as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve a line item where filers may tag a direct total OR only its
    Current/Noncurrent components: per (fiscal_year, period_type, period_end),
    prefer the direct total tag, else sum the components, else fall back.

    ``sum_partial_components=True`` accepts a single present component as the
    total (used for operating leases, where a noncurrent-only filer is common
    and under-reporting beats dropping the fact entirely).
    """
    facts_root = raw.get("facts") or {}
    cutoff_year = _normalization_cutoff_year(
        years_back=years_back,
        filed_as_of=filed_as_of,
    )
    source_url = COMPANYFACTS_URL.format(cik=cik)
    grouped: dict[tuple[int, str], dict[str, dict[str, dict[str, Any]]]] = {}

    for taxonomy, tag in _tag_priority(line_item):
        for candidate in _iter_normalized_tag_facts(
            facts_root,
            taxonomy=taxonomy,
            tag=tag,
            line_item=line_item,
            annual=annual,
            cutoff_year=cutoff_year,
            filed_as_of=filed_as_of,
        ):
            bucket_key = (int(candidate["fiscal_year"]), str(candidate["period_type"]))
            period_key = str(candidate["period_end"])
            by_tag = grouped.setdefault(bucket_key, {}).setdefault(period_key, {})
            by_tag[tag] = _keep_restatement(by_tag.get(tag), candidate)

    results: list[dict[str, Any]] = []
    for (fiscal_year, period_type), by_period_end in sorted(grouped.items()):
        period_end = max(by_period_end)
        tag_candidates = by_period_end[period_end]
        selected: dict[str, Any] | None = None

        for tag in direct_tag_priority:
            candidate = tag_candidates.get(tag)
            if candidate is not None:
                selected = candidate
                break

        if selected is None:
            current = tag_candidates.get(component_tags["current"])
            noncurrent = tag_candidates.get(component_tags["noncurrent"])
            components = [c for c in (current, noncurrent) if c is not None]
            if len(components) == 2 or (sum_partial_components and components):
                latest_component = max(
                    components,
                    key=lambda item: (
                        str(item.get("filed") or ""),
                        bool(_is_amended_form(str(item.get("form") or ""))),
                    ),
                )
                selected = {
                    "fiscal_year": fiscal_year,
                    "period_end": period_end,
                    "period_type": period_type,
                    "value": sum(float(c["value"]) for c in components),
                    # A sum is not knowable until its latest component is
                    # filed; retain that effective filing provenance.
                    "filed": str(latest_component.get("filed") or ""),
                    "form": str(latest_component.get("form") or ""),
                    "accession": str(latest_component.get("accession") or ""),
                    "_source": _summed_source(components, line_item=line_item),
                }

        if selected is None:
            for tag in fallback_tag_priority:
                candidate = tag_candidates.get(tag)
                if candidate is not None:
                    selected = candidate
                    break

        if selected is None:
            continue
        results.append(
            {
                "line_item": line_item,
                "fiscal_year": int(selected["fiscal_year"]),
                "period_end": str(selected["period_end"]),
                "period_type": str(selected["period_type"]),
                "value": float(selected["value"]),
                "units": "USD_millions",
                "source_url": source_url,
                "filed_date": str(selected.get("filed") or ""),
                "form": str(selected.get("form") or ""),
                "accession": str(selected.get("accession") or ""),
                **(selected.get("_source") or _source(selected)),
            }
        )
    return results


def _resolve_total_debt_rows(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int,
    annual: bool,
    filed_as_of: str | None = None,
) -> list[dict[str, Any]]:
    rows = _resolve_component_summed_rows(
        raw,
        cik=cik,
        years_back=years_back,
        annual=annual,
        line_item="total_debt",
        direct_tag_priority=_DIRECT_TOTAL_DEBT_TAG_PRIORITY,
        component_tags=_TOTAL_DEBT_COMPONENT_TAGS,
        fallback_tag_priority=_TOTAL_DEBT_FALLBACK_TAG_PRIORITY,
        filed_as_of=filed_as_of,
    )
    # A period the chain resolved counts as covered even when its total is then
    # refused as incomplete: the gap tier below must not answer it with a narrower
    # instrument-family sum.
    # The exception is DebtCurrent alone: the current debt line and nothing else, so the
    # period is left open for the gap tier, which may complete it with a noncurrent family.
    covered = {
        (int(row["fiscal_year"]), str(row["period_type"]))
        for row in rows
        if row.get("components") or str(row.get("tag") or "") != "DebtCurrent"
    }
    facts_by_end = _debt_completeness_lines(
        raw, years_back=years_back, annual=annual, filed_as_of=filed_as_of
    )
    rows = [
        completed
        for row in rows
        if (completed := _complete_total_debt_row(row, facts_by_end.get(str(row["period_end"]), {})))
        is not None
    ]
    rows.extend(
        _resolve_total_debt_gap_rows(
            raw,
            cik=cik,
            years_back=years_back,
            annual=annual,
            filed_as_of=filed_as_of,
            covered=covered,
        )
    )
    return sorted(rows, key=lambda row: (int(row["fiscal_year"]), str(row["period_type"])))


def _complete_total_debt_row(
    row: dict[str, Any], facts: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """The chain's total-debt row made complete, or None (UNKNOWN): see
    app.market.company_facts_extract.resolve_complete_total_debt. ``facts`` are the
    balance sheet's own debt lines at the row's period end."""
    components = row.get("components") or []
    chain_tags = [str(c["tag"]) for c in components] if components else [str(row.get("tag") or "")]
    resolved = resolve_complete_total_debt(
        float(row["value"]),
        tags=chain_tags,
        lines={tag: float(fact["value"]) for tag, fact in facts.items()},
    )
    if resolved is None:
        return None
    total, tags = resolved
    if set(tags) == set(chain_tags):
        return row
    parts = [facts[tag] for tag in tags]
    latest = max(
        parts,
        key=lambda item: (
            str(item.get("filed") or ""),
            bool(_is_amended_form(str(item.get("form") or ""))),
        ),
    )
    return {
        **row,
        "value": float(total),
        "filed_date": str(latest.get("filed") or ""),
        "form": str(latest.get("form") or ""),
        "accession": str(latest.get("accession") or ""),
        **_summed_source(parts, line_item="total_debt"),
    }


def _debt_completeness_lines(
    raw: dict[str, Any],
    *,
    years_back: int,
    annual: bool,
    filed_as_of: str | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """period_end -> {tag: fact} for DEBT_COMPLETENESS_TAGS, restatement-resolved the same
    way as the chain (values in USD millions): the balance sheet's own debt lines, used to
    refuse a total they contradict and to complete a narrow one."""
    facts_root = raw.get("facts") or {}
    cutoff_year = _normalization_cutoff_year(years_back=years_back, filed_as_of=filed_as_of)
    by_end: dict[str, dict[str, dict[str, Any]]] = {}
    for tag in DEBT_COMPLETENESS_TAGS:
        for candidate in _iter_normalized_tag_facts(
            facts_root,
            taxonomy="us-gaap",
            tag=tag,
            line_item="total_debt",
            annual=annual,
            cutoff_year=cutoff_year,
            filed_as_of=filed_as_of,
        ):
            by_tag = by_end.setdefault(str(candidate["period_end"]), {})
            by_tag[tag] = _keep_restatement(by_tag.get(tag), candidate)
    return by_end


# ── total_debt gap tier ───────────────────────────────────────────────────────
# About a third of cached filers with a current balance sheet resolved no
# total_debt: they tag their borrowings only under instrument-level concepts
# (revolver, notes, loans, secured/unsecured, short-term borrowings) that the
# chain above never reads. These families are consulted ONLY for a
# (fiscal_year, period_type) the chain above leaves empty, so no existing row
# changes, and only from ONE filing — the latest-filed one reporting any of
# them for that balance-sheet date — so current and noncurrent parts can never
# mix vintages. Each family is (total, current, noncurrent): its total is used
# when present, else its current + noncurrent; a total is never added to its
# own parts. Two families are summed only where they cannot overlap — one
# purely current plus one purely noncurrent, or secured plus unsecured; any
# other mix stays missing, because filers routinely tag the same instrument
# under two families. Finance leases stay out, as in the chain above. Bank and
# insurer funding balances disqualify the filing: for those issuers "total
# debt" is not the EV bridge's concept. A filing whose unread balances are all
# zero emits nothing — evidencing a zero is a separate policy.
_TOTAL_DEBT_GAP_FAMILIES: dict[str, tuple[str | None, str | None, str | None]] = {
    # Main-chain members of a family (LongTermDebt, LongTermDebtNoncurrent,
    # LineOfCredit, NotesPayable, ...) are absent from any gap period by
    # construction, so only the members the chain does not read are listed.
    "long_term_debt": (None, "LongTermDebtCurrent", None),
    "long_term_debt_and_lease": (None, "LongTermDebtAndCapitalLeaseObligationsCurrent", None),
    "line_of_credit": (None, "LinesOfCreditCurrent", "LongTermLineOfCredit"),
    "notes_payable": (None, "NotesPayableCurrent", "LongTermNotesPayable"),
    "notes_payable_to_bank": (None, "NotesPayableToBankCurrent", "NotesPayableToBankNoncurrent"),
    "loans_payable": ("LoansPayable", "LoansPayableCurrent", "LongTermLoansPayable"),
    "loans_payable_to_bank": ("LoansPayableToBank", "LoansPayableToBankCurrent", None),
    "notes_and_loans_payable": (
        "NotesAndLoansPayable",
        "NotesAndLoansPayableCurrent",
        "LongTermNotesAndLoans",
    ),
    "other_notes_payable": (
        "OtherNotesPayable",
        "OtherNotesPayableCurrent",
        "OtherLongTermNotesPayable",
    ),
    "other_loans_payable": ("OtherLoansPayable", "OtherLoansPayableCurrent", None),
    "other_long_term_debt": (
        "OtherLongTermDebt",
        "OtherLongTermDebtCurrent",
        "OtherLongTermDebtNoncurrent",
    ),
    "convertible_notes": (None, "ConvertibleNotesPayableCurrent", None),
    "convertible_debt": (None, "ConvertibleDebtCurrent", None),
    "senior_notes": (None, "SeniorNotesCurrent", "SeniorLongTermNotes"),
    "secured_debt": ("SecuredDebt", "SecuredDebtCurrent", "SecuredLongTermDebt"),
    "unsecured_debt": ("UnsecuredDebt", "UnsecuredDebtCurrent", "UnsecuredLongTermDebt"),
    "subordinated_debt": ("SubordinatedDebt", None, "SubordinatedLongTermDebt"),
    "junior_subordinated_notes": (
        "JuniorSubordinatedNotes",
        None,
        "JuniorSubordinatedLongTermNotes",
    ),
    # Short-term borrowings are current by definition.
    "short_term_borrowings": (None, "ShortTermBorrowings", None),
    # The whole current debt line: reached only by a period whose chain answer was
    # DebtCurrent alone (the chain refuses that and leaves the period open), so it is summed
    # with a purely noncurrent family when the filing has one, and otherwise stands alone
    # like any other purely current family here.
    "debt_current": (None, "DebtCurrent", None),
    "short_term_bank_loans": (None, "ShortTermBankLoansAndNotesPayable", None),
    "short_term_nonbank_loans": (None, "ShortTermNonBankLoansAndNotesPayable", None),
    "other_short_term_borrowings": (None, "OtherShortTermBorrowings", None),
    "commercial_paper": ("CommercialPaper", None, None),
}
_TOTAL_DEBT_GAP_FAMILY_OF: dict[str, str] = {
    tag: family
    for family, members in _TOTAL_DEBT_GAP_FAMILIES.items()
    for tag in members
    if tag is not None
}
_TOTAL_DEBT_GAP_DISQUALIFIERS: tuple[str, ...] = (
    *TAG_MAP["deposits"],
    "AdvancesFromFederalHomeLoanBanks",
    "FederalHomeLoanBankAdvances",
    "OtherBorrowings",
    "FederalFundsPurchased",
    "SecuritiesSoldUnderAgreementsToRepurchase",
    "FederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase",
    "JuniorSubordinatedDebentureOwedToUnconsolidatedSubsidiaryTrust",
    "SurplusNotes",
    "SecuredBorrowingsGrossIncludingNotSubjectToMasterNettingArrangement",
    "WarehouseAgreementBorrowings",
)


def _gap_family_value(family: str, tag_facts: dict[str, dict[str, Any]]) -> float:
    total, current, noncurrent = _TOTAL_DEBT_GAP_FAMILIES[family]
    if total is not None and total in tag_facts:
        return float(tag_facts[total]["value"])
    return sum(
        float(tag_facts[tag]["value"])
        for tag in (current, noncurrent)
        if tag is not None and tag in tag_facts
    )


def _gap_family_tags(family: str, tag_facts: dict[str, dict[str, Any]]) -> list[str]:
    """The tags `_gap_family_value` reads for ``family``."""
    total, current, noncurrent = _TOTAL_DEBT_GAP_FAMILIES[family]
    if total is not None and total in tag_facts:
        return [total]
    return [tag for tag in (current, noncurrent) if tag is not None and tag in tag_facts]


def _gap_families_are_disjoint(families: set[str], tag_facts: dict[str, dict[str, Any]]) -> bool:
    if len(families) == 1:
        return True
    if len(families) != 2:
        return False
    if families == {"secured_debt", "unsecured_debt"}:
        return True
    sides: set[frozenset[str]] = set()
    for family in families:
        total, current, noncurrent = _TOTAL_DEBT_GAP_FAMILIES[family]
        sides.add(
            frozenset(
                side
                for side, tag in (
                    ("total", total),
                    ("current", current),
                    ("noncurrent", noncurrent),
                )
                if tag is not None and tag in tag_facts
            )
        )
    return sides == {frozenset({"current"}), frozenset({"noncurrent"})}


def _resolve_total_debt_gap_rows(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int,
    annual: bool,
    filed_as_of: str | None,
    covered: set[tuple[int, str]],
) -> list[dict[str, Any]]:
    facts_root = raw.get("facts") or {}
    cutoff_year = _normalization_cutoff_year(years_back=years_back, filed_as_of=filed_as_of)
    source_url = COMPANYFACTS_URL.format(cik=cik)
    # (fiscal_year, period_type) -> period_end -> filing -> tag -> fact
    grouped: dict[
        tuple[int, str], dict[str, dict[tuple[str, str, str], dict[str, dict[str, Any]]]]
    ] = {}
    for tag in (*_TOTAL_DEBT_GAP_FAMILY_OF, *_TOTAL_DEBT_GAP_DISQUALIFIERS):
        for candidate in _iter_normalized_tag_facts(
            facts_root,
            taxonomy="us-gaap",
            tag=tag,
            line_item="total_debt",
            annual=annual,
            cutoff_year=cutoff_year,
            filed_as_of=filed_as_of,
        ):
            bucket_key = (int(candidate["fiscal_year"]), str(candidate["period_type"]))
            if bucket_key in covered:
                continue
            filing = (
                str(candidate.get("filed") or ""),
                str(candidate.get("form") or ""),
                str(candidate.get("accession") or ""),
            )
            by_tag = (
                grouped.setdefault(bucket_key, {})
                .setdefault(str(candidate["period_end"]), {})
                .setdefault(filing, {})
            )
            by_tag[tag] = _keep_restatement(by_tag.get(tag), candidate)

    results: list[dict[str, Any]] = []
    for (fiscal_year, period_type), by_period_end in sorted(grouped.items()):
        period_end = max(by_period_end)
        latest: dict[str, Any] | None = None
        for filed, form, accession in by_period_end[period_end]:
            probe = {"filed": filed, "form": form, "accession": accession, "value": 0.0}
            latest = _keep_restatement(latest, probe)
        if latest is None:
            continue
        filing = (latest["filed"], latest["form"], latest["accession"])
        tag_facts = by_period_end[period_end][filing]
        nonzero = [tag for tag, fact in tag_facts.items() if float(fact["value"]) != 0.0]
        if any(tag in _TOTAL_DEBT_GAP_DISQUALIFIERS for tag in nonzero):
            continue
        families = {_TOTAL_DEBT_GAP_FAMILY_OF[tag] for tag in nonzero}
        if not families or not _gap_families_are_disjoint(families, tag_facts):
            continue
        value = sum(_gap_family_value(family, tag_facts) for family in sorted(families))
        if value == 0.0:
            continue
        results.append(
            {
                "line_item": "total_debt",
                "fiscal_year": fiscal_year,
                "period_end": period_end,
                "period_type": period_type,
                "value": value,
                "units": "USD_millions",
                "source_url": source_url,
                "filed_date": latest["filed"],
                "form": latest["form"],
                "accession": latest["accession"],
                **_summed_source(
                    [
                        tag_facts[tag]
                        for family in sorted(families)
                        for tag in _gap_family_tags(family, tag_facts)
                    ]
                ),
            }
        )
    return results


def _resolve_short_term_investment_rows(
    facts_root: dict[str, Any],
    *,
    cash_rows: dict[Any, dict[str, Any]],
    annual: bool,
    cutoff_year: int,
    filed_as_of: str | None,
    source_url: str,
) -> list[dict[str, Any]]:
    """One short_term_investments row beside each chosen cash row, same period.

    The amount is what net debt adds to that cash: the current short-term
    investments reported at the cash row's own balance-sheet date (never another
    date's), by app.market.company_facts_extract.short_term_investments_addition —
    nothing when the cash row is already the combined cash-and-short-term line.
    """
    by_end: dict[str, dict[str, dict[str, Any]]] = {}
    for tag in SHORT_TERM_INVESTMENT_TAGS:
        for candidate in _iter_normalized_tag_facts(
            facts_root,
            taxonomy="us-gaap",
            tag=tag,
            line_item="short_term_investments",
            annual=annual,
            cutoff_year=cutoff_year,
            filed_as_of=filed_as_of,
        ):
            per_tag = by_end.setdefault(str(candidate["period_end"]), {})
            per_tag[tag] = _keep_restatement(per_tag.get(tag), candidate)

    results: list[dict[str, Any]] = []
    for cash in cash_rows.values():
        period_end = str(cash["period_end"])
        facts = by_end.get(period_end) or {}
        added = short_term_investments_addition(
            {tag: float(fact["value"]) for tag, fact in facts.items()},
            cash_tag=str(cash.get("tag") or ""),
            cash_value=float(cash["value"]),
        )
        if added is None:
            continue
        amount, tags = added
        parts = [facts[tag] for tag in tags]
        latest = max(
            parts,
            key=lambda item: (str(item.get("filed") or ""), _is_amended_form(str(item.get("form") or ""))),
        )
        results.append(
            {
                "line_item": "short_term_investments",
                "fiscal_year": int(cash["fiscal_year"]),
                "period_end": period_end,
                "period_type": str(cash["period_type"]),
                "value": amount,
                "units": "USD_millions",
                "source_url": source_url,
                "filed_date": str(latest.get("filed") or ""),
                "form": str(latest.get("form") or ""),
                "accession": str(latest.get("accession") or ""),
                **_summed_source(parts, line_item="short_term_investments"),
                "source_tags": short_term_investments_derivation(
                    tags, cash_tag=str(cash.get("tag") or "")
                ),
            }
        )
    return results


def _resolve_operating_lease_rows(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int,
    annual: bool,
    filed_as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Per period_end prefer the direct OperatingLeaseLiability total, else sum
    Current + Noncurrent components (audit: lease-current-portion-cross-surface-
    divergence — first-tag-wins dropped the current portion for filers that tag
    only the split components)."""
    return _resolve_component_summed_rows(
        raw,
        cik=cik,
        years_back=years_back,
        annual=annual,
        line_item="operating_lease_liability",
        direct_tag_priority=_LEASE_DIRECT_TAG_PRIORITY,
        component_tags=_LEASE_COMPONENT_TAGS,
        fallback_tag_priority=[],
        sum_partial_components=True,
        filed_as_of=filed_as_of,
    )


# Senior-claim component groups (review EVB-3): within a group the first tag
# with a NONZERO value wins (PreferredStockValue = 0 routinely coexists with
# redeemable preferred in temporary equity — the 0 must not mask it); ACROSS
# groups the values are disjoint claims and are SUMMED.
_PREFERRED_COMPONENT_GROUPS: list[list[str]] = [
    ["PreferredStockValue", "PreferredStockValueOutstanding"],
    ["TemporaryEquityCarryingAmountAttributableToParent"],
]
_NCI_COMPONENT_GROUPS: list[list[str]] = [
    ["MinorityInterest"],
    ["RedeemableNoncontrollingInterestEquityCarryingAmount"],
]


def _resolve_grouped_sum_rows(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int,
    annual: bool,
    line_item: str,
    component_tag_groups: list[list[str]],
    filed_as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve a line item whose tags are DISJOINT components: per
    (fiscal_year, period_type) at the latest period_end, each group
    contributes its first nonzero-valued tag (falling back to a present zero)
    and the groups are summed. Emits a row whenever any component tag is
    present — an explicit 0 is affirmative absence, not missing data."""
    facts_root = raw.get("facts") or {}
    cutoff_year = _normalization_cutoff_year(
        years_back=years_back,
        filed_as_of=filed_as_of,
    )
    source_url = COMPANYFACTS_URL.format(cik=cik)
    grouped: dict[tuple[int, str], dict[str, dict[str, dict[str, Any]]]] = {}

    for taxonomy, tag in _tag_priority(line_item):
        for candidate in _iter_normalized_tag_facts(
            facts_root,
            taxonomy=taxonomy,
            tag=tag,
            line_item=line_item,
            annual=annual,
            cutoff_year=cutoff_year,
            filed_as_of=filed_as_of,
        ):
            bucket_key = (int(candidate["fiscal_year"]), str(candidate["period_type"]))
            period_key = str(candidate["period_end"])
            by_tag = grouped.setdefault(bucket_key, {}).setdefault(period_key, {})
            by_tag[tag] = _keep_restatement(by_tag.get(tag), candidate)

    results: list[dict[str, Any]] = []
    for (fiscal_year, period_type), by_period_end in sorted(grouped.items()):
        period_end = max(by_period_end)
        tag_candidates = by_period_end[period_end]
        total = 0.0
        any_present = False
        contributors: list[dict[str, Any]] = []
        for group in component_tag_groups:
            group_value: float | None = None
            group_candidate: dict[str, Any] | None = None
            for tag in group:
                candidate = tag_candidates.get(tag)
                if candidate is None:
                    continue
                any_present = True
                value = float(candidate["value"])
                if group_value is None:
                    group_value = value
                    group_candidate = candidate
                if value != 0.0:
                    group_value = value
                    group_candidate = candidate
                    break
            total += group_value or 0.0
            if group_candidate is not None:
                contributors.append(group_candidate)
        if not any_present:
            continue
        latest_component = max(
            contributors,
            key=lambda item: (
                str(item.get("filed") or ""),
                bool(_is_amended_form(str(item.get("form") or ""))),
            ),
        )
        results.append(
            {
                "line_item": line_item,
                "fiscal_year": fiscal_year,
                "period_end": period_end,
                "period_type": period_type,
                "value": total,
                "units": "USD_millions",
                "source_url": source_url,
                "filed_date": str(latest_component.get("filed") or ""),
                "form": str(latest_component.get("form") or ""),
                "accession": str(latest_component.get("accession") or ""),
                **_summed_source(contributors, line_item=line_item),
            }
        )
    return results


def _resolve_preferred_equity_rows(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int,
    annual: bool,
    filed_as_of: str | None = None,
) -> list[dict[str, Any]]:
    return _resolve_grouped_sum_rows(
        raw,
        cik=cik,
        years_back=years_back,
        annual=annual,
        line_item="preferred_equity",
        component_tag_groups=_PREFERRED_COMPONENT_GROUPS,
        filed_as_of=filed_as_of,
    )


def _resolve_noncontrolling_interest_rows(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int,
    annual: bool,
    filed_as_of: str | None = None,
) -> list[dict[str, Any]]:
    return _resolve_grouped_sum_rows(
        raw,
        cik=cik,
        years_back=years_back,
        annual=annual,
        line_item="noncontrolling_interest",
        component_tag_groups=_NCI_COMPONENT_GROUPS,
        filed_as_of=filed_as_of,
    )


def _is_amended_form(form: str) -> bool:
    return str(form or "").upper().endswith("/A")


def _prefer_restatement(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Return True if `candidate` should replace `existing` within a dedup bucket.

    SEC companyfacts facts arrive filed-ascending, so `existing` is the earliest
    yielded fact (original filing / highest-priority tag). We replace it only when
    the candidate represents a genuine restatement: a strictly later `filed` date,
    or an amended form (10-K/A, 10-Q/A) on the same `filed` date.
    """
    existing_filed = str(existing.get("filed") or "")
    candidate_filed = str(candidate.get("filed") or "")
    if candidate_filed > existing_filed:
        return True
    if candidate_filed == existing_filed:
        return _is_amended_form(candidate.get("form", "")) and not _is_amended_form(
            existing.get("form", "")
        )
    return False


def _keep_restatement(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    """Same-tag, same-period restatement choice that SEC array order cannot decide.

    Applies ``_prefer_restatement`` (later ``filed`` wins; on the same ``filed``
    date an amended form beats an original) in BOTH directions, so the result
    no longer depends on which fact the array yields first — companyfacts
    arrays are not reliably filed-ascending (a live payload lists a 2017
    comparative before the 2016 original it restates). Only an exact tie on
    filed date and amendment status falls to a deterministic accession/value
    key; that key has no economic meaning, it just makes the pick stable.
    """
    if existing is None:
        return candidate
    if _prefer_restatement(existing, candidate):
        return candidate
    if _prefer_restatement(candidate, existing):
        return existing

    def _tie_key(fact: dict[str, Any]) -> tuple[str, float]:
        return (str(fact.get("accession") or ""), float(fact.get("value") or 0.0))

    return candidate if _tie_key(candidate) > _tie_key(existing) else existing


def normalize_annual_facts_from_raw(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int = 10,
    filed_as_of: str | None = None,
) -> list[dict[str, Any]]:
    facts_root = raw.get("facts") or {}
    cutoff_year = _normalization_cutoff_year(
        years_back=years_back,
        filed_as_of=filed_as_of,
    )
    source_url = COMPANYFACTS_URL.format(cik=cik)
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    order: list[tuple[str, int]] = []
    share_alternates: dict[Any, dict[str, dict[str, Any]]] = {}

    for line_item in TAG_MAP:
        if line_item in (
            "total_debt",
            "short_term_investments",
            "operating_lease_liability",
            # Senior claims resolve via grouped component summation —
            # first-tag-wins lets a zero-valued permanent tag mask the
            # redeemable component (review EVB-3).
            "preferred_equity",
            "noncontrolling_interest",
        ):
            continue
        for taxonomy, tag in _tag_priority(line_item):
            for candidate in _iter_normalized_tag_facts(
                facts_root,
                taxonomy=taxonomy,
                tag=tag,
                line_item=line_item,
                annual=True,
                cutoff_year=cutoff_year,
                filed_as_of=filed_as_of,
            ):
                fy = int(candidate["fiscal_year"])
                key = (line_item, fy)
                if line_item == "shares_outstanding":
                    _track_share_alternate(share_alternates, key, candidate)
                existing = chosen.get(key)
                if existing is None:
                    chosen[key] = candidate
                    order.append(key)
                elif (
                    # Restatement replacement is SAME-TAG only: a later-filed
                    # comparative from a lower-priority tag must not clobber a
                    # higher-priority tag's fact (audit: restricted-cash /
                    # lease-component cross-tag clobbering).
                    candidate.get("tag") == existing.get("tag")
                    and _prefer_restatement(existing, candidate)
                ):
                    chosen[key] = candidate
    _guard_share_choices(raw, chosen, order, share_alternates)
    _weighted_share_fallback(
        raw,
        facts_root,
        chosen,
        order,
        reported=set(share_alternates),
        cutoff_year=cutoff_year,
        filed_as_of=filed_as_of,
    )

    results: list[dict[str, Any]] = []
    for key in order:
        line_item, fy = key
        candidate = chosen[key]
        results.append(
            {
                "line_item": line_item,
                "fiscal_year": fy,
                "period_end": str(candidate["period_end"]),
                "value": float(candidate["value"]),
                "units": "USD_millions" if line_item != "shares_outstanding" else "shares_millions",
                "source_url": source_url,
                "period_type": "FY",
                "filed_date": str(candidate.get("filed") or ""),
                "form": str(candidate.get("form") or ""),
                "accession": str(candidate.get("accession") or ""),
                **_source(candidate),
                # D&A is first-tag-wins over concepts that do and do not include
                # intangible amortization; persist which one answered so the EPV
                # does not take amortization out of a Depreciation-only figure.
                **(
                    {"source_tags": str(candidate.get("tag") or "")}
                    if line_item == "depreciation_amortization"
                    else {}
                ),
            }
        )
    results.extend(
        _resolve_short_term_investment_rows(
            facts_root,
            cash_rows={key: chosen[key] for key in order if key[0] == "cash"},
            annual=True,
            cutoff_year=cutoff_year,
            filed_as_of=filed_as_of,
            source_url=source_url,
        )
    )
    results.extend(
        _resolve_total_debt_rows(
            raw,
            cik=cik,
            years_back=years_back,
            annual=True,
            filed_as_of=filed_as_of,
        )
    )
    results.extend(
        _resolve_operating_lease_rows(
            raw,
            cik=cik,
            years_back=years_back,
            annual=True,
            filed_as_of=filed_as_of,
        )
    )
    results.extend(
        _resolve_preferred_equity_rows(
            raw,
            cik=cik,
            years_back=years_back,
            annual=True,
            filed_as_of=filed_as_of,
        )
    )
    results.extend(
        _resolve_noncontrolling_interest_rows(
            raw,
            cik=cik,
            years_back=years_back,
            annual=True,
            filed_as_of=filed_as_of,
        )
    )
    return results


# ── quarterly period labels ───────────────────────────────────────────────────
# SEC stamps every fact with the fiscal year/period (``fy``/``fp``) of the FILING
# that carries it, so a Q1-2026 10-Q reports its Q1-2025 comparative as "fy 2026,
# Q1" and its prior year-end balance sheet as "fy 2026, Q1" too. The quarterly
# normalizer labels each fact by the period it MEASURES instead: a fiscal calendar
# maps each periodic filing's own period end to that filing's (fy, fp), and every
# 10-Q fact takes the label of its own end date. dei cover facts describe the
# filing itself and keep the filing's stamps.
_PERIODIC_FORMS = _ANNUAL_FORMS | _QUARTERLY_FORMS


def _as_date(text: Any) -> date | None:
    try:
        return date.fromisoformat(str(text or "")[:10])
    except ValueError:
        return None


def fiscal_calendar(
    raw: dict[str, Any], *, filed_as_of: str | None = None
) -> dict[date, tuple[int, str]]:
    """Map each periodic filing's own period-end date to its (fiscal year, fiscal period).

    A filing's own period end is the most frequent ``end`` among its us-gaap facts (its
    current-period columns outnumber any single comparative date). With ``filed_as_of``
    only filings visible on that date are used.
    """
    ends: dict[str, dict[str, int]] = {}
    labels: dict[str, dict[tuple[int, str], int]] = {}
    for node in ((raw.get("facts") or {}).get("us-gaap") or {}).values():
        if not isinstance(node, dict):
            continue
        for facts in (node.get("units") or {}).values():
            for fact in facts or []:
                if not isinstance(fact, dict):
                    continue
                form = str(fact.get("form") or "")
                accn = str(fact.get("accn") or "")
                if form not in _PERIODIC_FORMS or not accn or not fact.get("end"):
                    continue
                if not _fact_visible_as_filed(fact, filed_as_of):
                    continue
                by_end = ends.setdefault(accn, {})
                by_end[str(fact["end"])] = by_end.get(str(fact["end"]), 0) + 1
                fy, fp = fact.get("fy"), fact.get("fp")
                if fy is not None and fp:
                    try:
                        label = (int(fy), str(fp))
                    except (TypeError, ValueError):
                        continue
                    by_label = labels.setdefault(accn, {})
                    by_label[label] = by_label.get(label, 0) + 1
    calendar: dict[date, tuple[int, str]] = {}
    for accn, counter in ends.items():
        if not labels.get(accn):
            continue
        period_end = _as_date(max(counter.items(), key=lambda item: (item[1], item[0]))[0])
        if period_end is None:
            continue
        label = max(labels[accn].items(), key=lambda item: (item[1], item[0]))[0]
        calendar.setdefault(period_end, label)
    return calendar


def _lookup_label(
    calendar: dict[date, tuple[int, str]], end: date, sorted_ends: list[date]
) -> tuple[int, str] | None:
    exact = calendar.get(end)
    if exact:
        return exact
    near = [d for d in sorted_ends if abs((d - end).days) <= 3]
    if near:
        return calendar[min(near, key=lambda d: abs((d - end).days))]
    # A comparative for a period older than the company's first XBRL filing: the
    # same quarter one year later is in the calendar.
    shifted = end + timedelta(days=364)
    near = [d for d in sorted_ends if abs((d - shifted).days) <= 7]
    if near:
        fy, fp = calendar[min(near, key=lambda d: abs((d - shifted).days))]
        return fy - 1, fp
    return None


def _label_quarterly_periods(raw: dict[str, Any], *, filed_as_of: str | None) -> dict[str, Any]:
    """Copy of ``raw`` where every us-gaap 10-Q fact carries its own period's ``fy``/``fp``.

    Facts whose period cannot be placed get ``fp: None`` (the quarterly normalizer then
    ignores them); a prior year-end balance carried in a 10-Q is labelled with its
    annual period and so never lands in a quarter. The input is never mutated.
    """
    calendar = fiscal_calendar(raw, filed_as_of=filed_as_of)
    sorted_ends = sorted(calendar)
    facts_root = dict(raw.get("facts") or {})
    us_gaap = facts_root.get("us-gaap")
    if not isinstance(us_gaap, dict):
        return raw
    labelled: dict[str, Any] = {}
    for concept, node in us_gaap.items():
        if not isinstance(node, dict):
            labelled[concept] = node
            continue
        units_out: dict[str, list[Any]] = {}
        for unit, facts in (node.get("units") or {}).items():
            out: list[Any] = []
            for fact in facts or []:
                if not isinstance(fact, dict) or str(fact.get("form") or "") not in _QUARTERLY_FORMS:
                    out.append(fact)
                    continue
                end = _as_date(fact.get("end"))
                label = _lookup_label(calendar, end, sorted_ends) if end else None
                if label is None:
                    out.append({**fact, "fp": None})
                else:
                    out.append({**fact, "fy": label[0], "fp": label[1]})
            units_out[unit] = out
        labelled[concept] = {**node, "units": units_out}
    facts_root["us-gaap"] = labelled
    return {**raw, "facts": facts_root}


def normalize_quarterly_facts_from_raw(
    raw: dict[str, Any],
    *,
    cik: str,
    years_back: int = 10,
    filed_as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Quarterly (10-Q) facts, each labelled with the fiscal year and quarter it measures.

    ``filed_as_of`` gives a point-in-time view: only facts filed on or before that date
    are read (undated facts never are), and the fiscal calendar is built from those
    filings alone. Without it the latest-filed value of every period is used.
    """
    raw = _label_quarterly_periods(raw, filed_as_of=filed_as_of)
    facts_root = raw.get("facts") or {}
    cutoff_year = _normalization_cutoff_year(years_back=years_back, filed_as_of=filed_as_of)
    source_url = COMPANYFACTS_URL.format(cik=cik)
    chosen: dict[tuple[str, int, str], dict[str, Any]] = {}
    order: list[tuple[str, int, str]] = []
    share_alternates: dict[Any, dict[str, dict[str, Any]]] = {}

    for line_item in TAG_MAP:
        if line_item in (
            "total_debt",
            "short_term_investments",
            "operating_lease_liability",
            # Senior claims resolve via grouped component summation —
            # first-tag-wins lets a zero-valued permanent tag mask the
            # redeemable component (review EVB-3).
            "preferred_equity",
            "noncontrolling_interest",
        ):
            continue
        for taxonomy, tag in _tag_priority(line_item):
            for candidate in _iter_normalized_tag_facts(
                facts_root,
                taxonomy=taxonomy,
                tag=tag,
                line_item=line_item,
                annual=False,
                cutoff_year=cutoff_year,
                filed_as_of=filed_as_of,
            ):
                fy = int(candidate["fiscal_year"])
                fp = str(candidate["period_type"])
                key = (line_item, fy, fp)
                if line_item == "shares_outstanding":
                    _track_share_alternate(share_alternates, key, candidate)
                existing = chosen.get(key)
                if existing is None:
                    chosen[key] = candidate
                    order.append(key)
                elif (
                    # Restatement replacement is SAME-TAG only: a later-filed
                    # comparative from a lower-priority tag must not clobber a
                    # higher-priority tag's fact (audit: restricted-cash /
                    # lease-component cross-tag clobbering).
                    candidate.get("tag") == existing.get("tag")
                    and _prefer_restatement(existing, candidate)
                ):
                    chosen[key] = candidate
    _guard_share_choices(raw, chosen, order, share_alternates)

    results: list[dict[str, Any]] = []
    for key in order:
        line_item, fy, fp = key
        candidate = chosen[key]
        results.append(
            {
                "line_item": line_item,
                "fiscal_year": fy,
                "period_end": str(candidate["period_end"]),
                "period_type": fp,
                "value": float(candidate["value"]),
                "units": "USD_millions" if line_item != "shares_outstanding" else "shares_millions",
                "source_url": source_url,
                "filed_date": str(candidate.get("filed") or ""),
                "form": str(candidate.get("form") or ""),
                "accession": str(candidate.get("accession") or ""),
                **_source(candidate),
                # D&A is first-tag-wins over concepts that do and do not include
                # intangible amortization; persist which one answered so the EPV
                # does not take amortization out of a Depreciation-only figure.
                **(
                    {"source_tags": str(candidate.get("tag") or "")}
                    if line_item == "depreciation_amortization"
                    else {}
                ),
            }
        )
    results.extend(
        _resolve_short_term_investment_rows(
            facts_root,
            cash_rows={key: chosen[key] for key in order if key[0] == "cash"},
            annual=False,
            cutoff_year=cutoff_year,
            filed_as_of=filed_as_of,
            source_url=source_url,
        )
    )
    for resolver in (
        _resolve_total_debt_rows,
        _resolve_operating_lease_rows,
        _resolve_preferred_equity_rows,
        _resolve_noncontrolling_interest_rows,
    ):
        results.extend(
            resolver(
                raw, cik=cik, years_back=years_back, annual=False, filed_as_of=filed_as_of
            )
        )
    return results


def fetch_quarterly_facts(cik: str, years_back: int = 10) -> list[dict[str, Any]]:
    raw = _fetch_raw(cik)
    return normalize_quarterly_facts_from_raw(raw, cik=cik, years_back=years_back)


def fetch_annual_facts(cik: str, years_back: int = 10) -> list[dict[str, Any]]:
    raw = _fetch_raw(cik)
    return normalize_annual_facts_from_raw(raw, cik=cik, years_back=years_back)
