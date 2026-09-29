from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Any, Iterable, Sequence

from app.config import AppConfig
from app.ingest.companyfacts import TAG_MAP, _TOTAL_DEBT_GAP_FAMILIES
from app.market.company_facts_provider import companyfacts_cache_path, normalize_cik
from app.util.financial_data_access import issuer_companyfacts_rows

DEBT_INSTRUMENT_CONCEPTS: tuple[str, ...] = (
    "Debt",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "LongTermDebtAndFinanceLeaseObligations",
    "LongTermDebtAndFinanceLeaseObligationsCurrent",
    "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "DebtAndCapitalLeaseObligations",
    "DebtCurrent",
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtInstrumentCarryingAmount",
    "Borrowings",
    "ShortTermBorrowings",
    "ShortTermDebt",
    "CommercialPaper",
    "NotesPayable",
    "NotesPayableCurrent",
    "NotesAndLoansPayableLongtermPortion",
    "SeniorNotes",
    "SeniorNotesNoncurrent",
    "ConvertibleLongTermNotesPayable",
    "ConvertibleNotesPayable",
    "ConvertibleDebtNoncurrent",
    "ConvertibleDebt",
    "ConvertibleSubordinatedDebtNoncurrent",
    "LongTermLoansFromBank",
    "LineOfCredit",
    "FinanceLeaseLiability",
    "FinanceLeaseLiabilityCurrent",
    "FinanceLeaseLiabilityNoncurrent",
    "CapitalLeaseObligations",
    "CapitalLeaseObligationsCurrent",
    "CapitalLeaseObligationsNoncurrent",
)

def _debt_absence_concepts() -> tuple[str, ...]:
    """Every concept whose presence means the filing reports debt.

    The tag list above, then the facts spine's own total-debt chain and
    its gap families (revolvers current and long-term, loans, secured and
    unsecured debt, subordinated notes, ...) and bank funding balances. A
    filer that reports debt only under one of those families is not debt-free.
    Reused by import so the two lists cannot drift apart.
    """
    extra = [
        *TAG_MAP["total_debt"],
        *(
            tag
            for members in _TOTAL_DEBT_GAP_FAMILIES.values()
            for tag in members
            if tag is not None
        ),
        *TAG_MAP["deposits"],
    ]
    ordered = list(DEBT_INSTRUMENT_CONCEPTS)
    for concept in extra:
        if concept not in ordered:
            ordered.append(concept)
    return tuple(ordered)


DEBT_ABSENCE_CONCEPTS: tuple[str, ...] = _debt_absence_concepts()

# Balance-sheet liability lines that are not borrowings. Regulation S-X 5-02
# requires borrowings to be presented on lines of their own, so a liabilities
# total the filing builds entirely from these lines holds no material debt.
# Interest payable is deliberately absent: it is not debt, but it says there is
# some, and the test should then fail rather than pass.
NON_DEBT_LIABILITY_CONCEPTS: tuple[str, ...] = (
    "AccountsPayableCurrent",
    "AccountsPayableNoncurrent",
    "AccountsPayableTradeCurrent",
    "AccountsPayableRelatedPartiesCurrent",
    "AccruedLiabilitiesCurrent",
    "AccruedLiabilitiesNoncurrent",
    "AccountsPayableAndAccruedLiabilitiesCurrent",
    "AccountsPayableAndAccruedLiabilitiesNoncurrent",
    "EmployeeRelatedLiabilitiesCurrent",
    "EmployeeRelatedLiabilitiesNoncurrent",
    "AccruedSalariesCurrent",
    "AccruedEmployeeBenefitsCurrent",
    "TaxesPayableCurrent",
    "AccruedIncomeTaxesCurrent",
    "AccruedIncomeTaxesNoncurrent",
    "SalesAndExciseTaxPayableCurrent",
    "DeferredIncomeTaxLiabilitiesNet",
    "DeferredTaxLiabilitiesNoncurrent",
    "LiabilityForUncertainTaxPositionsNoncurrent",
    "ContractWithCustomerLiabilityCurrent",
    "ContractWithCustomerLiabilityNoncurrent",
    "DeferredRevenueCurrent",
    "DeferredRevenueNoncurrent",
    "OperatingLeaseLiabilityCurrent",
    "OperatingLeaseLiabilityNoncurrent",
    "DividendsPayableCurrent",
    "DerivativeLiabilitiesCurrent",
    "DerivativeLiabilitiesNoncurrent",
    "DeferredCompensationLiabilityCurrent",
    "DeferredCompensationLiabilityClassifiedNoncurrent",
    "DefinedBenefitPensionPlanLiabilitiesNoncurrent",
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
    "AssetRetirementObligationsNoncurrent",
    "BusinessCombinationContingentConsiderationLiabilityCurrent",
    "BusinessCombinationContingentConsiderationLiabilityNoncurrent",
    "OtherAccruedLiabilitiesCurrent",
    "OtherAccruedLiabilitiesNoncurrent",
    "OtherLiabilitiesCurrent",
    "OtherLiabilitiesNoncurrent",
)
# How closely the named lines must add up to the reported total. A balance
# sheet foots exactly at its own rounding; this allows only that rounding.
_LIABILITIES_FOOTING_TOLERANCE = 0.001

# A positive senior claim this many fiscal years or more before the net-debt
# year is stale: the filings since have presented equity without it. Only a
# claim reported inside this window still blocks the evidenced zero.
SENIOR_CLAIM_STALE_AFTER_YEARS = 3

SENIOR_CLAIM_CONCEPTS: dict[str, tuple[str, ...]] = {
    "preferred_equity": (
        "PreferredStockValue",
        "PreferredStockValueOutstanding",
        "TemporaryEquityCarryingAmountAttributableToParent",
    ),
    "noncontrolling_interest": (
        "MinorityInterest",
        "RedeemableNoncontrollingInterestEquityCarryingAmount",
    ),
}

_EQUITY_AGGREGATE_CONCEPTS: tuple[str, ...] = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
_LIABILITIES_AGGREGATE_CONCEPTS: tuple[str, ...] = ("Liabilities",)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _years(facts: dict[str, list[tuple[int, float]]], line_item: str) -> set[int]:
    return {int(year) for year, _ in facts.get(line_item, ())}


def _has_year(
    facts: dict[str, list[tuple[int, float]]],
    line_item: str,
    fiscal_year: int,
) -> bool:
    return fiscal_year in _years(facts, line_item)


def _latest_common_year(
    facts: dict[str, list[tuple[int, float]]],
    *line_items: str,
) -> int | None:
    common: set[int] | None = None
    for line_item in line_items:
        years = _years(facts, line_item)
        if not years:
            return None
        common = years if common is None else common & years
    return max(common) if common else None


def _materialized_companyfacts(
    *,
    issuer_cik: str,
    cfg: AppConfig,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    normalized_cik = normalize_cik(issuer_cik)
    if not normalized_cik:
        return None
    path = companyfacts_cache_path(
        normalized_cik,
        cfg=cfg,
        create_parent=False,
    )
    expected_root = (cfg.cache_dir / "companyfacts").resolve()
    try:
        resolved_path = path.resolve(strict=True)
        if not resolved_path.is_relative_to(expected_root):
            return None
        raw_bytes = resolved_path.read_bytes()
        wrapper = json.loads(raw_bytes)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(wrapper, dict):
        return None
    companyfacts = (
        wrapper.get("companyfacts") if isinstance(wrapper.get("companyfacts"), dict) else wrapper
    )
    if not isinstance(companyfacts, dict) or not isinstance(companyfacts.get("facts"), dict):
        return None
    materialized_cik = normalize_cik(wrapper.get("cik") or companyfacts.get("cik"))
    if materialized_cik != normalized_cik:
        return None
    return companyfacts, {
        "path": str(resolved_path),
        "sha256": sha256(raw_bytes).hexdigest(),
        "retrieved_at": wrapper.get("retrieved_at"),
        "source_url": wrapper.get("source_url")
        or f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized_cik}.json",
        "issuer_cik": normalized_cik,
    }


def _entry_fiscal_year(entry: dict[str, Any]) -> int | None:
    raw_year = entry.get("fy")
    try:
        return int(raw_year)
    except (TypeError, ValueError):
        period_end = str(entry.get("end") or "")
        if len(period_end) >= 4 and period_end[:4].isdigit():
            return int(period_end[:4])
    return None


def _entry_is_visible(entry: dict[str, Any], *, as_of_date: str) -> bool:
    period_end = str(entry.get("end") or "")
    filed_date = str(entry.get("filed") or "")
    if not period_end or not filed_date:
        return False
    return period_end <= filed_date <= as_of_date and period_end <= as_of_date


def _raw_concept_records(
    companyfacts: dict[str, Any],
    *,
    concepts: Sequence[str],
    fiscal_year: int,
    as_of_date: str,
    visible_only: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    facts_root = companyfacts.get("facts")
    if not isinstance(facts_root, dict):
        return records
    concept_set = set(concepts)
    for taxonomy, taxonomy_facts in sorted(facts_root.items()):
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept, payload in sorted(taxonomy_facts.items()):
            if concept not in concept_set or not isinstance(payload, dict):
                continue
            units = payload.get("units")
            if not isinstance(units, dict):
                continue
            for unit, entries in sorted(units.items()):
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if (
                        not isinstance(entry, dict)
                        or _entry_fiscal_year(entry) != fiscal_year
                        or (visible_only and not _entry_is_visible(entry, as_of_date=as_of_date))
                    ):
                        continue
                    records.append(
                        {
                            "taxonomy": str(taxonomy),
                            "concept": str(concept),
                            "unit": str(unit),
                            "fiscal_year": fiscal_year,
                            "period_end": str(entry.get("end") or ""),
                            "filed_date": str(entry.get("filed") or ""),
                            "form": str(entry.get("form") or ""),
                            "accession": str(entry.get("accn") or ""),
                            "value": entry.get("val"),
                        }
                    )
    return records


def _raw_claim_records(
    companyfacts: dict[str, Any],
    *,
    concepts: Sequence[str],
    as_of_date: str,
) -> list[dict[str, Any]]:
    """Every record of ``concepts``, any year and any visibility, marked visible or not."""
    records: list[dict[str, Any]] = []
    facts_root = companyfacts.get("facts")
    if not isinstance(facts_root, dict):
        return records
    concept_set = set(concepts)
    for taxonomy_facts in facts_root.values():
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept, payload in taxonomy_facts.items():
            if concept not in concept_set or not isinstance(payload, dict):
                continue
            units = payload.get("units")
            if not isinstance(units, dict):
                continue
            for unit, entries in units.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    records.append(
                        {
                            "concept": str(concept),
                            "unit": str(unit),
                            "fiscal_year": _entry_fiscal_year(entry),
                            "period_end": str(entry.get("end") or ""),
                            "filed_date": str(entry.get("filed") or ""),
                            "form": str(entry.get("form") or ""),
                            "accession": str(entry.get("accn") or ""),
                            "value": entry.get("val"),
                            "visible": _entry_is_visible(entry, as_of_date=as_of_date),
                        }
                    )
    return records


def _raw_recent_positive_claim_blocks(
    companyfacts: dict[str, Any],
    *,
    concepts: Sequence[str],
    net_debt_year: int,
    not_before_fiscal_year: int,
    balance_sheet_end: str,
    as_of_date: str,
) -> bool:
    """True when a recent positive senior claim is not shown to be gone.

    A positive claim reported in a filing of the stale window (through the
    net-debt year itself) blocks unless a later annual balance sheet (an
    as-of-visible 10-K equity aggregate dated after the claim and before the
    net-debt balance sheet) reported no positive claim at its own date.
    """
    # Only filings public at the as-of date: a claim first reported later says nothing
    # about what was knowable then.
    claim_records = [
        record
        for record in _raw_claim_records(companyfacts, concepts=concepts, as_of_date=as_of_date)
        if record["visible"]
    ]
    positives = [
        record
        for record in claim_records
        if record["fiscal_year"] is not None
        and not_before_fiscal_year <= record["fiscal_year"] <= net_debt_year
        and _finite_number(record["value"])
        and float(record["value"]) > 0
    ]
    if not positives:
        return False
    positive_dates = {
        record["period_end"]
        for record in claim_records
        if not _finite_number(record["value"]) or float(record["value"]) != 0.0
    }
    annual_dates = sorted(
        {
            record["period_end"]
            for record in _raw_claim_records(
                companyfacts, concepts=_EQUITY_AGGREGATE_CONCEPTS, as_of_date=as_of_date
            )
            if record["visible"]
            and record["form"].startswith("10-K")
            and record["period_end"] < balance_sheet_end
            and record["period_end"] not in positive_dates
        }
    )
    return any(
        not any(record["period_end"] < annual_date for annual_date in annual_dates)
        for record in positives
    )


def _normalized_basis_row(
    conn: Any,
    ticker: str,
    *,
    issuer_cik: str,
    issuer_aliases: Sequence[str],
    as_of_date: str,
    line_item: str,
    fiscal_year: int,
) -> dict[str, Any] | None:
    _, rows = issuer_companyfacts_rows(
        conn,
        ticker,
        columns=(
            "line_item",
            "fiscal_year",
            "period_end",
            "value",
            "units",
            "source_url",
            "fetched_at",
            "filed_date",
            "form",
            "accession",
        ),
        issuer_cik=issuer_cik,
        aliases=issuer_aliases,
        period_types=("FY",),
        line_items=(line_item,),
        as_of_date=as_of_date,
        value_not_null=True,
        require_filed_asof=True,
        order_by="fiscal_year DESC, period_end DESC",
    )
    for row in rows:
        if int(row["fiscal_year"]) == fiscal_year and _finite_number(row["value"]):
            return {
                "line_item": str(row["line_item"]),
                "fiscal_year": int(row["fiscal_year"]),
                "period_end": str(row["period_end"]),
                "value": float(row["value"]),
                "units": str(row["units"] or ""),
                "source_url": str(row["source_url"] or ""),
                "fetched_at": str(row["fetched_at"] or ""),
                "filed_date": str(row["filed_date"] or ""),
                "form": str(row["form"] or ""),
                "accession": str(row["accession"] or ""),
            }
    return None


def _normalized_row_exists(
    conn: Any,
    ticker: str,
    *,
    issuer_cik: str,
    issuer_aliases: Sequence[str],
    as_of_date: str,
    line_item: str,
    fiscal_year: int,
) -> bool:
    _, rows = issuer_companyfacts_rows(
        conn,
        ticker,
        columns=("fiscal_year",),
        issuer_cik=issuer_cik,
        aliases=issuer_aliases,
        period_types=("FY",),
        line_items=(line_item,),
        as_of_date=as_of_date,
        value_not_null=False,
        require_filed_asof=True,
        order_by="fiscal_year DESC",
    )
    return any(int(row["fiscal_year"]) == fiscal_year for row in rows)


def _append_zero(
    facts: dict[str, list[tuple[int, float]]],
    *,
    line_item: str,
    fiscal_year: int,
) -> None:
    values = list(facts.get(line_item) or ())
    values.append((fiscal_year, 0.0))
    values.sort(key=lambda item: item[0], reverse=True)
    facts[line_item] = values


def _normalized_prior_positive_exists(
    facts: dict[str, list[tuple[int, float]]],
    *,
    line_item: str,
    before_fiscal_year: int,
    not_before_fiscal_year: int | None = None,
) -> bool:
    return any(
        int(year) < before_fiscal_year
        and (not_before_fiscal_year is None or int(year) >= not_before_fiscal_year)
        and _finite_number(value)
        and float(value) > 0
        for year, value in facts.get(line_item, ())
    )


def _usd_instant_records(
    companyfacts: dict[str, Any],
    *,
    concepts: Sequence[str],
    accession: str,
    period_end: str,
) -> dict[str, float] | None:
    """concept -> value for the USD balances one filing reports at one date.

    None when a concept carries two different values there (nothing says which
    one the balance sheet used).
    """
    facts_root = companyfacts.get("facts")
    if not isinstance(facts_root, dict):
        return {}
    concept_set = set(concepts)
    out: dict[str, float] = {}
    for taxonomy_facts in facts_root.values():
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept, payload in taxonomy_facts.items():
            if concept not in concept_set or not isinstance(payload, dict):
                continue
            units = payload.get("units")
            entries = units.get("USD") if isinstance(units, dict) else None
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if (
                    not isinstance(entry, dict)
                    or str(entry.get("accn") or "") != accession
                    or str(entry.get("end") or "") != period_end
                    or entry.get("start")
                    or not _finite_number(entry.get("val"))
                ):
                    continue
                value = float(entry["val"])
                if concept in out and out[concept] != value:
                    return None
                out[concept] = value
    return out


def complete_liabilities_components(
    companyfacts: dict[str, Any],
    *,
    liabilities_record: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """The named non-debt lines that account for a reported liabilities total.

    Returns the lines (concept and value) when the same filing, at the same
    balance-sheet date, reports non-debt liability lines that add up to the
    total; None otherwise. A total that the named lines do not reach may hold
    debt under a line this test cannot read, so it proves nothing.
    """
    accession = str(liabilities_record.get("accession") or "")
    period_end = str(liabilities_record.get("period_end") or "")
    total = liabilities_record.get("value")
    if not accession or not period_end or not _finite_number(total) or float(total) < 0:
        return None
    components = _usd_instant_records(
        companyfacts,
        concepts=NON_DEBT_LIABILITY_CONCEPTS,
        accession=accession,
        period_end=period_end,
    )
    if components is None:
        return None
    total_value = float(total)
    footed = sum(components.values())
    if abs(footed - total_value) > _LIABILITIES_FOOTING_TOLERANCE * max(abs(total_value), 1.0):
        return None
    return [
        {"concept": concept, "value": components[concept]}
        for concept in NON_DEBT_LIABILITY_CONCEPTS
        if concept in components
    ]


def debt_concept_reported(
    companyfacts: dict[str, Any],
    *,
    accession: str,
    period_end: str,
    as_of_date: str,
) -> bool:
    """Whether any debt concept appears in the filing, or at the date in any visible filing."""
    facts_root = companyfacts.get("facts")
    if not isinstance(facts_root, dict):
        return False
    concept_set = set(DEBT_ABSENCE_CONCEPTS)
    for taxonomy_facts in facts_root.values():
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept, payload in taxonomy_facts.items():
            if concept not in concept_set or not isinstance(payload, dict):
                continue
            units = payload.get("units")
            if not isinstance(units, dict):
                continue
            for entries in units.values():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if str(entry.get("accn") or "") == accession:
                        return True
                    if str(entry.get("end") or "") == period_end and _entry_is_visible(
                        entry, as_of_date=as_of_date
                    ):
                        return True
    return False


def raw_debt_zero_evidence(
    companyfacts: dict[str, Any],
    *,
    period_end: str,
    as_of_date: str,
) -> dict[str, Any] | None:
    """Evidence, from the raw filing alone, that the balance sheet at ``period_end`` has no debt.

    The debt-only evidenced-zero policy for the as-of path: the latest filing
    visible at ``as_of_date`` that reports total liabilities at ``period_end``
    must report no debt concept at all, no visible filing may report one at
    that date, and the filing's named non-debt lines must add up to the total.
    Cash is never evidenced this way.
    """
    candidates = [
        record
        for record in _raw_concept_records_any_year(
            companyfacts, concepts=_LIABILITIES_AGGREGATE_CONCEPTS, as_of_date=as_of_date
        )
        if record["period_end"] == period_end and record["unit"] == "USD"
    ]
    if not candidates:
        return None
    basis = max(
        candidates,
        key=lambda record: (str(record["filed_date"]), str(record["accession"])),
    )
    if debt_concept_reported(
        companyfacts,
        accession=str(basis["accession"]),
        period_end=period_end,
        as_of_date=as_of_date,
    ):
        return None
    components = complete_liabilities_components(companyfacts, liabilities_record=basis)
    if components is None:
        return None
    return {
        "derivation": "EVIDENCED_ZERO_DEBT_COMPLETE_LIABILITIES",
        "basis_raw_record": basis,
        "liabilities_components": components,
    }


def _raw_concept_records_any_year(
    companyfacts: dict[str, Any],
    *,
    concepts: Sequence[str],
    as_of_date: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    facts_root = companyfacts.get("facts")
    if not isinstance(facts_root, dict):
        return records
    concept_set = set(concepts)
    for taxonomy, taxonomy_facts in sorted(facts_root.items()):
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept, payload in sorted(taxonomy_facts.items()):
            if concept not in concept_set or not isinstance(payload, dict):
                continue
            units = payload.get("units")
            if not isinstance(units, dict):
                continue
            for unit, entries in sorted(units.items()):
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict) or not _entry_is_visible(
                        entry, as_of_date=as_of_date
                    ):
                        continue
                    if not _finite_number(entry.get("val")):
                        continue
                    records.append(
                        {
                            "taxonomy": str(taxonomy),
                            "concept": str(concept),
                            "unit": str(unit),
                            "period_end": str(entry.get("end") or ""),
                            "filed_date": str(entry.get("filed") or ""),
                            "form": str(entry.get("form") or ""),
                            "accession": str(entry.get("accn") or ""),
                            "value": float(entry["val"]),
                        }
                    )
    return records


def _proof(
    *,
    line_item: str,
    fiscal_year: int,
    proof_type: str,
    basis_row: dict[str, Any],
    basis_raw_records: Iterable[dict[str, Any]],
    absent_concepts: Sequence[str],
    materialized: dict[str, Any],
    liabilities_components: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    proof = {
        "line_item": line_item,
        "fiscal_year": fiscal_year,
        "value": 0.0,
        "derivation": proof_type,
        "basis_row": basis_row,
        "basis_raw_records": list(basis_raw_records),
        "absent_concepts": list(absent_concepts),
        "materialized_companyfacts": materialized,
    }
    if liabilities_components is not None:
        proof["liabilities_components"] = list(liabilities_components)
    return proof


def _basis_raw_evidence(
    records: Sequence[dict[str, Any]],
    *,
    basis_row: dict[str, Any],
) -> list[dict[str, Any]]:
    accession = str(basis_row.get("accession") or "")
    period_end = str(basis_row.get("period_end") or "")
    exact = [
        record
        for record in records
        if str(record.get("accession") or "") == accession
        and str(record.get("period_end") or "") == period_end
    ]
    same_period = [
        record for record in records
        if period_end and str(record.get("period_end") or "") == period_end
    ]
    candidates = (exact if period_end else []) or same_period
    if not candidates:
        return []
    return [
        max(
            candidates,
            key=lambda record: (
                str(record.get("filed_date") or ""),
                str(record.get("accession") or ""),
                str(record.get("concept") or ""),
            ),
        )
    ]


def resolve_evidenced_zero_facts(
    facts: dict[str, list[tuple[int, float]]],
    *,
    ticker: str,
    conn: Any,
    as_of_date: str,
    issuer_cik: str | None,
    issuer_aliases: Sequence[str] = (),
    cfg: AppConfig,
) -> tuple[dict[str, list[tuple[int, float]]], list[dict[str, Any]]]:
    """Mint narrowly qualified zero-valued net-debt operands.

    The returned facts remain an in-memory valuation revision. No normalized
    source row is inserted or altered; the proof is bound into the valuation
    facts fingerprint and method inputs by the caller.
    """

    resolved_facts = {
        line_item: [(int(year), float(value)) for year, value in values]
        for line_item, values in facts.items()
    }
    normalized_cik = normalize_cik(issuer_cik)
    materialized_result = (
        _materialized_companyfacts(issuer_cik=normalized_cik, cfg=cfg) if normalized_cik else None
    )
    if materialized_result is None:
        return resolved_facts, []
    companyfacts, materialized = materialized_result
    proofs: list[dict[str, Any]] = []

    existing_net_debt_year = _latest_common_year(resolved_facts, "total_debt", "cash")
    debt_candidate_years = sorted(
        (_years(resolved_facts, "cash") & _years(resolved_facts, "total_liabilities"))
        - _years(resolved_facts, "total_debt"),
        reverse=True,
    )
    if debt_candidate_years and (
        existing_net_debt_year is None or debt_candidate_years[0] > existing_net_debt_year
    ):
        debt_year = debt_candidate_years[0]
        liabilities_row = _normalized_basis_row(
            conn,
            ticker,
            issuer_cik=normalized_cik,
            issuer_aliases=issuer_aliases,
            as_of_date=as_of_date,
            line_item="total_liabilities",
            fiscal_year=debt_year,
        )
        liabilities_raw = _raw_concept_records(
            companyfacts,
            concepts=_LIABILITIES_AGGREGATE_CONCEPTS,
            fiscal_year=debt_year,
            as_of_date=as_of_date,
        )
        debt_records = _raw_concept_records(
            companyfacts,
            concepts=DEBT_ABSENCE_CONCEPTS,
            fiscal_year=debt_year,
            as_of_date=as_of_date,
        )
        liabilities_basis_raw = (
            _basis_raw_evidence(liabilities_raw, basis_row=liabilities_row)
            if liabilities_row is not None else []
        )
        # The complete-liabilities test (debt-only policy): the named
        # non-debt lines of the same filing must add up to the total. A total
        # with no debt tag but an unexplained remainder proves nothing.
        liabilities_components = (
            complete_liabilities_components(
                companyfacts, liabilities_record=liabilities_basis_raw[0]
            )
            if liabilities_basis_raw and not debt_records
            else None
        )
        if (
            liabilities_row is not None
            and liabilities_basis_raw
            and not debt_records
            and liabilities_components is not None
        ):
            _append_zero(
                resolved_facts,
                line_item="total_debt",
                fiscal_year=debt_year,
            )
            proofs.append(
                _proof(
                    line_item="total_debt",
                    fiscal_year=debt_year,
                    proof_type="EVIDENCED_ZERO_DEBT_INSTRUMENT_ABSENCE",
                    basis_row=liabilities_row,
                    basis_raw_records=liabilities_basis_raw,
                    absent_concepts=DEBT_ABSENCE_CONCEPTS,
                    materialized=materialized,
                    liabilities_components=liabilities_components,
                )
            )

    net_debt_year = _latest_common_year(resolved_facts, "total_debt", "cash")
    if net_debt_year is None:
        return resolved_facts, proofs
    equity_row = _normalized_basis_row(
        conn,
        ticker,
        issuer_cik=normalized_cik,
        issuer_aliases=issuer_aliases,
        as_of_date=as_of_date,
        line_item="equity",
        fiscal_year=net_debt_year,
    )
    equity_raw = _raw_concept_records(
        companyfacts,
        concepts=_EQUITY_AGGREGATE_CONCEPTS,
        fiscal_year=net_debt_year,
        as_of_date=as_of_date,
    )
    equity_basis_raw = (
        _basis_raw_evidence(equity_raw, basis_row=equity_row)
        if equity_row is not None else []
    )
    if equity_row is None or not equity_basis_raw:
        return resolved_facts, proofs

    for line_item, concepts in SENIOR_CLAIM_CONCEPTS.items():
        if _has_year(resolved_facts, line_item, net_debt_year):
            continue
        if _normalized_row_exists(
            conn,
            ticker,
            issuer_cik=normalized_cik,
            issuer_aliases=issuer_aliases,
            as_of_date=as_of_date,
            line_item=line_item,
            fiscal_year=net_debt_year,
        ):
            continue
        # The claim as reported AT the balance-sheet date, by any filing public
        # at the as-of date (a later 10-Q's comparative column included). A positive or unreadable
        # value there that the normalized facts do not carry refuses; a
        # reported zero is direct evidence. Records for OTHER dates in the
        # same fiscal year (interim quarters, prior-year comparatives in this
        # year's filings) say nothing about this date by themselves: they are
        # weighed by the recent-positive rule below instead of blocking
        # outright (the Realty Income / LeMaitre shapes).
        balance_sheet_end = str(equity_row.get("period_end") or "")
        at_date = [
            record
            for record in _raw_claim_records(companyfacts, concepts=concepts, as_of_date=as_of_date)
            if record["period_end"] == balance_sheet_end and record["visible"]
        ]
        if at_date:
            visible_zeros = [
                record
                for record in at_date
                if record["visible"]
                and _finite_number(record["value"])
                and float(record["value"]) == 0.0
            ]
            if not visible_zeros or any(
                not _finite_number(record["value"]) or float(record["value"]) != 0.0
                for record in at_date
            ):
                continue
            _append_zero(resolved_facts, line_item=line_item, fiscal_year=net_debt_year)
            proofs.append(
                _proof(
                    line_item=line_item,
                    fiscal_year=net_debt_year,
                    proof_type="EVIDENCED_ZERO_SENIOR_CLAIM_REPORTED",
                    basis_row=equity_row,
                    basis_raw_records=[*equity_basis_raw, *visible_zeros],
                    absent_concepts=(),
                    materialized=materialized,
                )
            )
            continue
        # Only a RECENT positive claim blocks. One from years before the
        # balance-sheet date (a 2012 minority interest against a 2025 balance
        # sheet) is stale: every filing since has presented equity without it,
        # so it is treated as absent for this date rather than as blocking.
        # A recent one also stops blocking once a LATER annual balance sheet
        # presented equity with the claim gone (a minority interest that
        # existed for one quarter before a spin-off; preferred stock redeemed
        # the same year it was issued).
        stale_before = net_debt_year - SENIOR_CLAIM_STALE_AFTER_YEARS + 1
        if _normalized_prior_positive_exists(
            resolved_facts,
            line_item=line_item,
            before_fiscal_year=net_debt_year,
            not_before_fiscal_year=stale_before,
        ) or _raw_recent_positive_claim_blocks(
            companyfacts,
            concepts=concepts,
            net_debt_year=net_debt_year,
            not_before_fiscal_year=stale_before,
            balance_sheet_end=balance_sheet_end,
            as_of_date=as_of_date,
        ):
            continue
        _append_zero(
            resolved_facts,
            line_item=line_item,
            fiscal_year=net_debt_year,
        )
        proofs.append(
            _proof(
                line_item=line_item,
                fiscal_year=net_debt_year,
                proof_type="EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE",
                basis_row=equity_row,
                basis_raw_records=equity_basis_raw,
                absent_concepts=concepts,
                materialized=materialized,
            )
        )
    return resolved_facts, proofs
