from __future__ import annotations

from datetime import datetime
from typing import Any

from app.market.shares_guard import guard_shares_fact


SHARES_TAG_PRIORITY: list[tuple[str, str]] = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockOtherSharesOutstanding"),
]

CFO_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]

CAPEX_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ("us-gaap", "CapitalExpendituresIncurringObligation"),
    ("us-gaap", "PaymentsForCapitalImprovements"),
    ("us-gaap", "PaymentsToAcquirePremisesAndEquipment"),
    ("us-gaap", "CapitalExpendituresIncurredButNotYetPaid"),
]

FCF_DIRECT_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "FreeCashFlow"),
]

CASH_EQUIVALENTS_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
    ("us-gaap", "CashCashEquivalentsAndShortTermInvestments"),
    ("us-gaap", "Cash"),
    ("us-gaap", "CashAndCashEquivalents"),
    ("us-gaap", "CashEquivalentsAtCarryingValue"),
    ("us-gaap", "CashAndDueFromBanks"),
    ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    (
        "us-gaap",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
    ),
]

# One fallback list for both total-debt paths: this as-of extractor and the ingest
# normalizer (app/ingest/companyfacts.py TAG_MAP["total_debt"] is built from it). The
# as-of path used to stop at SeniorNotes/Debt and missed the convertible, bank-loan and
# credit-line tags the normalizer read, so a filer whose only debt is a convertible note
# (LeMaitre, ConvertibleDebtNoncurrent) had total debt on one path and UNKNOWN on the other.
TOTAL_DEBT_FALLBACK_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
    ("us-gaap", "LongTermDebt"),
    ("us-gaap", "LongTermDebtNoncurrent"),
    ("us-gaap", "DebtAndCapitalLeaseObligations"),
    ("us-gaap", "NotesPayable"),
    ("us-gaap", "SeniorNotes"),
    ("us-gaap", "SeniorNotesNoncurrent"),
    ("us-gaap", "DebtLongtermAndShorttermCombinedAmount"),
    ("us-gaap", "Debt"),
    # Convertible-debt instruments -- pharma/biotech often carry their entire capital
    # stack as convertible notes; the legacy LongTermDebt tag is often empty for them.
    ("us-gaap", "ConvertibleLongTermNotesPayable"),
    ("us-gaap", "ConvertibleNotesPayable"),
    ("us-gaap", "ConvertibleDebtNoncurrent"),
    ("us-gaap", "ConvertibleDebt"),
    ("us-gaap", "ConvertibleSubordinatedDebtNoncurrent"),
    # Term loans + credit facilities
    ("us-gaap", "LongTermLoansFromBank"),
    ("us-gaap", "LineOfCredit"),
    ("us-gaap", "NotesAndLoansPayableLongtermPortion"),
]
# DebtInstrumentCarryingAmount is deliberately absent: it is an instrument-level
# concept (one note, one facility), and undimensioned it is whatever instrument
# table the filer happened to total -- PayPal's excludes its commercial paper and
# its current term debt. It never establishes an issuer's complete debt.
# LongTermDebtAndCapitalLeaseObligations is not a direct total either: it is the
# NONCURRENT long-term debt and lease line (Coca-Cola: 42,119 + its current
# 1,822 = its IncludingCurrentMaturities 43,941), so it stays a fallback.
TOTAL_DEBT_DIRECT_PREFERRED_TAGS = {
    "DebtAndCapitalLeaseObligations",
    "DebtLongtermAndShorttermCombinedAmount",
    "Debt",
}
# A total whose own definition covers every borrowing, current and noncurrent. Any
# other candidate (long-term debt, a noncurrent line, one instrument family) is
# narrow by definition and is completed from the balance sheet's own lines below.
TOTAL_DEBT_COMPLETE_SCOPE_TAGS = frozenset(TOTAL_DEBT_DIRECT_PREFERRED_TAGS)
TOTAL_DEBT_DIRECT_TAG_PRIORITY: list[tuple[str, str]] = [
    item for item in TOTAL_DEBT_FALLBACK_TAG_PRIORITY if item[1] in TOTAL_DEBT_DIRECT_PREFERRED_TAGS
]
TOTAL_DEBT_COMPONENT_FALLBACK_TAG_PRIORITY: list[tuple[str, str]] = [
    item
    for item in TOTAL_DEBT_FALLBACK_TAG_PRIORITY
    if item[1] not in TOTAL_DEBT_DIRECT_PREFERRED_TAGS
]

DEBT_CURRENT_TAG: tuple[str, str] = ("us-gaap", "DebtCurrent")
LONG_TERM_DEBT_NONCURRENT_TAG: tuple[str, str] = ("us-gaap", "LongTermDebtNoncurrent")

# Completeness evidence for a total-debt candidate. A balance sheet that shows
# noncurrent long-term debt of N and a current debt line of C carries at least
# N + C of debt, whatever the candidate's tag claims; a candidate below that floor
# is a narrow amount (long-term debt without its current lines, or noncurrent debt
# alone) and cannot be the issuer's complete debt. The current lines are summed
# with max(), never added to each other: filers put the same borrowing under more
# than one of them (one tags its whole current debt line as LongTermDebtCurrent
# while also tagging its short-term borrowings), so only the largest is a floor.
# Principal due next year (a maturity-schedule figure, often tagged for another
# window in a 10-Q) counts only where the balance sheet carries no current portion
# of its own. The noncurrent debt-and-lease line stands in for noncurrent debt only when
# LongTermDebtNoncurrent is not tagged, so a lease-exclusive candidate is not refused over
# its leases when the lease-free line is there. The floor only ever REFUSES a total --
# it is never itself returned as one.
# The noncurrent line is LongTermDebtNoncurrent, else the noncurrent debt-and-lease line
# (LongTermDebtAndCapitalLeaseObligations is noncurrent-only: Coca-Cola, ExxonMobil).
# DebtCurrent -- every current borrowing on one line -- is a current line like the rest:
# ExxonMobil's June 2026 10-Q shows 32,229 noncurrent and 10,139 current, so 32,229 alone
# is not its debt.
DEBT_COMPLETENESS_NONCURRENT_TAG = "LongTermDebtNoncurrent"
DEBT_COMPLETENESS_NONCURRENT_LEASE_TAG = "LongTermDebtAndCapitalLeaseObligations"
DEBT_PRINCIPAL_DUE_NEXT_YEAR_TAG = "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"
DEBT_COMPLETENESS_CURRENT_TAGS: tuple[str, ...] = (
    "LongTermDebtCurrent",
    DEBT_PRINCIPAL_DUE_NEXT_YEAR_TAG,
    "ShortTermBorrowings",
    "CommercialPaper",
    "DebtCurrent",
)
# Balance-sheet (carrying) current portions of long-term debt.
DEBT_CARRYING_CURRENT_TAGS: tuple[str, ...] = (
    "LongTermDebtCurrent",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "DebtCurrent",
)
# Assembling a complete total. Long-term debt INCLUDING its current maturities, in a
# shape whose parts are disjoint by definition, plus short-term borrowings (which
# exclude current maturities by definition): ShortTermBorrowings when tagged, else
# commercial paper plus other short-term borrowings (disjoint: "other" is what is not
# separately tagged); a balance sheet with no short-term line has none. A noncurrent
# line stands for the whole of long-term debt only when the balance sheet shows no
# current portion and no principal due next year at all. The shape of the
# candidate's own family is tried first (a lease-inclusive candidate's lease-inclusive
# shapes next), so a company with nothing missing keeps its value and tag.
DEBT_LONG_TERM_SHAPES: tuple[tuple[str, ...], ...] = (
    ("LongTermDebt",),
    ("LongTermDebtNoncurrent", "LongTermDebtCurrent"),
    ("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",),
    ("LongTermDebtAndCapitalLeaseObligations", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    ("LongTermDebtNoncurrent",),
    ("LongTermDebtAndCapitalLeaseObligations",),
)
# Noncurrent debt plus DebtCurrent, the balance sheet's whole current debt line (current
# maturities AND short-term borrowings), is the complete total: nothing short-term is added
# on top of these shapes.
DEBT_CURRENT_TOTAL_TAG = "DebtCurrent"
DEBT_NONCURRENT_PLUS_CURRENT_SHAPES: tuple[tuple[str, ...], ...] = (
    ("LongTermDebtNoncurrent", DEBT_CURRENT_TOTAL_TAG),
    ("LongTermDebtAndCapitalLeaseObligations", DEBT_CURRENT_TOTAL_TAG),
)
_DEBT_NONCURRENT_ONLY_TAGS = frozenset(
    {"LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"}
)
# Every line that shows the balance sheet carries current debt.
_DEBT_ANY_CURRENT_TAGS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *DEBT_COMPLETENESS_CURRENT_TAGS,
            "LongTermDebtAndCapitalLeaseObligationsCurrent",
            "OtherShortTermBorrowings",
        )
    )
)
DEBT_LEASE_INCLUSIVE_TAGS = frozenset(
    {
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    }
)
DEBT_SHORT_TERM_TOTAL_TAG = "ShortTermBorrowings"
DEBT_SHORT_TERM_PART_TAGS: tuple[str, ...] = ("CommercialPaper", "OtherShortTermBorrowings")
DEBT_COMPLETENESS_TAGS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            DEBT_COMPLETENESS_NONCURRENT_TAG,
            *DEBT_COMPLETENESS_CURRENT_TAGS,
            *DEBT_CARRYING_CURRENT_TAGS,
            *(tag for shape in DEBT_LONG_TERM_SHAPES for tag in shape),
            *(tag for shape in DEBT_NONCURRENT_PLUS_CURRENT_SHAPES for tag in shape),
            DEBT_SHORT_TERM_TOTAL_TAG,
            *DEBT_SHORT_TERM_PART_TAGS,
        )
    )
)
# Carrying amounts sit a little under the principal the maturity schedule reports
# (issuance costs, discounts); a floor must beat the candidate by more than that.
DEBT_COMPLETENESS_TOLERANCE = 0.02

# Short-term investments count as cash in net debt:
# Microsoft held 55.9 billion of them beside 20.9 billion of cash at June 2026, and
# netting only the cash overstated its net debt by the whole 55.9 billion. Only
# CURRENT marketable securities / short-term investments count, never long-term
# investments, and only from the cash's own balance-sheet date. One family is read,
# in this order, so no amount is counted twice: a single total tag, else the
# current marketable-securities total, else the available-for-sale and
# held-to-maturity current lines (disjoint by definition, so summed).
SHORT_TERM_INVESTMENT_TAG_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("ShortTermInvestments",),
    ("MarketableSecuritiesCurrent",),
    ("AvailableForSaleSecuritiesDebtSecuritiesCurrent", "HeldToMaturitySecuritiesCurrent"),
)
# The combined line already holds cash AND short-term investments: as the cash
# figure nothing is added to it; beside a cash-only figure, the difference is the
# short-term investments when no part is tagged.
CASH_AND_SHORT_TERM_INVESTMENTS_TAG = "CashCashEquivalentsAndShortTermInvestments"
SHORT_TERM_INVESTMENT_TAGS: tuple[str, ...] = (
    *(tag for family in SHORT_TERM_INVESTMENT_TAG_FAMILIES for tag in family),
    CASH_AND_SHORT_TERM_INVESTMENTS_TAG,
)


def short_term_investments_derivation(tags: list[str], *, cash_tag: str) -> str:
    """How the added amount was formed, in tag names ("A + B", or "combined - cash")."""
    if list(tags) == [CASH_AND_SHORT_TERM_INVESTMENTS_TAG]:
        return f"{CASH_AND_SHORT_TERM_INVESTMENTS_TAG} - {cash_tag}"
    return " + ".join(tags)


def short_term_investments_addition(
    lines: dict[str, float], *, cash_tag: str, cash_value: float
) -> tuple[float, list[str]] | None:
    """What to add to cash for net debt, from one balance sheet's lines.

    ``lines`` maps SHORT_TERM_INVESTMENT_TAGS reported at the cash's own
    balance-sheet date to their values, in the units of ``cash_value``. Returns
    ``(amount, tags used)``, or None when nothing is to be added: the cash figure
    is already the combined line, no short-term line is reported, or what is
    reported is not positive.
    """
    if cash_tag == CASH_AND_SHORT_TERM_INVESTMENTS_TAG:
        return None
    for family in SHORT_TERM_INVESTMENT_TAG_FAMILIES:
        present = [tag for tag in family if tag in lines]
        if present:
            amount = sum(float(lines[tag]) for tag in present)
            return (amount, present) if amount > 0 else None
    combined = lines.get(CASH_AND_SHORT_TERM_INVESTMENTS_TAG)
    if combined is not None and float(combined) - float(cash_value) > 0:
        return float(combined) - float(cash_value), [CASH_AND_SHORT_TERM_INVESTMENTS_TAG]
    return None


def debt_completeness_floor(lines: dict[str, float]) -> float | None:
    """The least debt a balance sheet's own lines admit (same units as ``lines``).

    ``lines`` maps DEBT_COMPLETENESS_TAGS present at one balance-sheet date to their
    values. None when no evidence line is present.
    """
    carries_current = any(tag in lines for tag in DEBT_CARRYING_CURRENT_TAGS)
    noncurrent = lines.get(
        DEBT_COMPLETENESS_NONCURRENT_TAG, lines.get(DEBT_COMPLETENESS_NONCURRENT_LEASE_TAG)
    )
    current = [
        float(lines[tag])
        for tag in DEBT_COMPLETENESS_CURRENT_TAGS
        if tag in lines and not (tag == DEBT_PRINCIPAL_DUE_NEXT_YEAR_TAG and carries_current)
    ]
    if noncurrent is None and not current:
        return None
    return float(noncurrent or 0.0) + max([0.0, *current])


def debt_total_is_incomplete(total: float, lines: dict[str, float]) -> bool:
    """True when the same balance sheet's own debt lines show more debt than ``total``."""
    floor = debt_completeness_floor(lines)
    if floor is None:
        return False
    return floor > float(total) * (1.0 + DEBT_COMPLETENESS_TOLERANCE)


def _short_term_add_is_confirmed(
    lines: dict[str, float], shape: tuple[str, ...], short_term: list[str]
) -> bool | None:
    """Whether short-term borrowings may be added on top of a shape that reads
    LongTermDebtCurrent. Filers put the same borrowing under both: one tags its whole
    current debt line (current maturities AND short-term borrowings) as
    LongTermDebtCurrent while also tagging the short-term borrowings, and adding the two
    counts those borrowings twice (900 + 150 + 150 = 1,200 where the debt is 1,050).

    Returns True when the addition is disjoint (the current portion is smaller than the
    short-term borrowings, so it cannot contain them; or the balance sheet's own combined
    lines reconcile to the sum), False when a combined line shows the current portion
    already holds them, and None when nothing tells the two apart -- the shape is then not
    used. Policy (2026-09-29, conservative): an ambiguous overlap is UNKNOWN, never the
    larger or the smaller guess."""
    if "LongTermDebtCurrent" not in shape or not short_term:
        return True
    current = float(lines["LongTermDebtCurrent"])
    added = sum(float(lines[tag]) for tag in short_term)
    if current < added:
        return True

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= max(abs(a), abs(b)) * DEBT_COMPLETENESS_TOLERANCE

    combined_current = lines.get(DEBT_CURRENT_TOTAL_TAG)
    if combined_current is not None:
        if close(float(combined_current), current + added):
            return True
        if close(float(combined_current), current):
            return False
        return None
    long_term = lines.get("LongTermDebt")
    noncurrent = lines.get("LongTermDebtNoncurrent")
    if long_term is not None and noncurrent is not None:
        # Long-term debt including current maturities reconciles to noncurrent plus the
        # current portion: that portion is current maturities only.
        if close(float(long_term), float(noncurrent) + current):
            return True
        if close(float(long_term), float(noncurrent) + current - added):
            return False
    return None


def _current_lines_overlap_is_ambiguous(lines: dict[str, float]) -> bool:
    """True when the balance sheet reports a current portion and short-term borrowings and
    nothing tells whether the one already holds the other (_short_term_add_is_confirmed)."""
    if "LongTermDebtCurrent" not in lines:
        return False
    return (
        _short_term_add_is_confirmed(lines, ("LongTermDebtCurrent",), _short_term_lines(lines))
        is None
    )


def _short_term_lines(lines: dict[str, float]) -> list[str]:
    """The short-term borrowing tags added on top of long-term debt: ShortTermBorrowings
    when tagged, else commercial paper plus other short-term borrowings; zeros left out."""
    if DEBT_SHORT_TERM_TOTAL_TAG in lines:
        short_term = [DEBT_SHORT_TERM_TOTAL_TAG]
    else:
        short_term = [tag for tag in DEBT_SHORT_TERM_PART_TAGS if tag in lines]
    return [tag for tag in short_term if float(lines[tag]) != 0.0]


def assemble_total_debt(
    lines: dict[str, float], *, prefer: str | None = None, at_least: float | None = None
) -> tuple[float, list[str]] | None:
    """Long-term debt including current maturities plus short-term borrowings, from one
    balance sheet's ``lines`` (DEBT_COMPLETENESS_TAGS -> value): (total, tags summed), or
    None. The first long-term shape present (``prefer``'s own shape first) whose total is
    not below ``at_least`` is used. Zero-valued short-term lines add nothing and are left
    out of the tags. Noncurrent debt plus DebtCurrent (the whole current debt line) is a
    complete shape of its own, with nothing short-term added."""
    no_current_portion = not any(
        tag in lines for tag in (*DEBT_CARRYING_CURRENT_TAGS, DEBT_PRINCIPAL_DUE_NEXT_YEAR_TAG)
    )
    shapes = sorted(
        (*DEBT_LONG_TERM_SHAPES, *DEBT_NONCURRENT_PLUS_CURRENT_SHAPES),
        key=lambda shape: (
            prefer not in shape,
            not (prefer in DEBT_LEASE_INCLUSIVE_TAGS and set(shape) <= DEBT_LEASE_INCLUSIVE_TAGS),
        ),
    )
    short_term = _short_term_lines(lines)
    for shape in shapes:
        if not all(tag in lines for tag in shape):
            continue
        if len(shape) == 1 and shape[0] in _DEBT_NONCURRENT_ONLY_TAGS and not no_current_portion:
            continue
        if DEBT_CURRENT_TOTAL_TAG in shape:
            tags = list(shape)
        else:
            confirmed = _short_term_add_is_confirmed(lines, shape, short_term)
            if confirmed is None:
                continue
            tags = [*shape, *short_term] if confirmed else list(shape)
        total = sum(float(lines[tag]) for tag in tags)
        if at_least is None or total >= at_least:
            return total, tags
    return None


def debt_long_term_shapes_conflict(lines: dict[str, float]) -> bool:
    """True when the balance sheet's own measures of long-term debt including current
    maturities do not reconcile: two lease-exclusive (or two lease-inclusive) shapes more
    than DEBT_COMPLETENESS_TOLERANCE apart, or a lease-inclusive total below a
    lease-exclusive one (finance leases are never negative). BOSC's long-term debt (1.12m)
    against its noncurrent plus current lines (1.747m); Vivid Seats' debt-and-lease lines
    (29.8) against its long-term debt (264.9)."""
    totals: dict[bool, list[float]] = {False: [], True: []}
    for shape in DEBT_LONG_TERM_SHAPES:
        if len(shape) == 1 and shape[0] in _DEBT_NONCURRENT_ONLY_TAGS:
            continue
        if all(tag in lines for tag in shape):
            totals[set(shape) <= DEBT_LEASE_INCLUSIVE_TAGS].append(
                sum(float(lines[tag]) for tag in shape)
            )
    for values in totals.values():
        if values and max(values) > min(values) * (1.0 + DEBT_COMPLETENESS_TOLERANCE):
            return True
    if totals[False] and totals[True]:
        return max(totals[False]) > min(totals[True]) * (1.0 + DEBT_COMPLETENESS_TOLERANCE)
    return False


def resolve_complete_total_debt(
    value: float, *, tags: list[str], lines: dict[str, float]
) -> tuple[float, list[str]] | None:
    """The complete total debt for one balance sheet, or None (UNKNOWN).

    ``value`` is the chain's candidate and ``tags`` the tag(s) it was read or summed from.
    A candidate whose scope is not complete by definition is replaced by the total the
    balance sheet's own lines assemble (never one below the candidate: parts that add up
    to less do not replace it). Every answer must clear the completeness floor; a total
    below it -- even one tagged as complete -- is replaced by the assembled total when
    that clears the floor, else refused. Returns (total, tags) in the units of ``lines``.
    """
    prefer = tags[0] if len(tags) == 1 else None
    # DebtCurrent is complete only summed with a noncurrent line; alone it is the current
    # debt and nothing else (a 10-K with DebtCurrent 100 and 900 of senior notes under a
    # tag the chain does not read is not 100 of debt).
    complete = any(tag in TOTAL_DEBT_COMPLETE_SCOPE_TAGS for tag in tags) or (
        DEBT_CURRENT_TOTAL_TAG in tags and any(tag in _DEBT_NONCURRENT_ONLY_TAGS for tag in tags)
    )
    # A candidate that is one side of the balance sheet only: noncurrent debt while a
    # current debt line is reported, or the current debt line alone.
    one_sided = bool(tags) and (
        (
            set(tags) <= _DEBT_NONCURRENT_ONLY_TAGS
            and any(float(lines.get(tag) or 0.0) != 0.0 for tag in _DEBT_ANY_CURRENT_TAGS)
        )
        or set(tags) == {DEBT_CURRENT_TOTAL_TAG}
    )
    result: tuple[float, list[str]] = (float(value), list(tags))
    conflict = debt_long_term_shapes_conflict(lines)
    if not complete and conflict:
        # A narrow candidate is one of the balance sheet's long-term measures, and they
        # contradict each other: none of them is established.
        return None
    if not complete:
        assembled = assemble_total_debt(
            lines, prefer=prefer, at_least=float(value) * (1.0 - DEBT_COMPLETENESS_TOLERANCE)
        )
        if assembled is not None:
            result = assembled
        elif one_sided or _current_lines_overlap_is_ambiguous(lines):
            # Nothing completes it: UNKNOWN, never the one side -- nor a narrow candidate
            # whose short-term borrowings may or may not sit inside its current portion.
            return None
    if not debt_total_is_incomplete(result[0], lines):
        return result
    if conflict:
        return None
    assembled = assemble_total_debt(lines, prefer=prefer, at_least=result[0])
    if assembled is not None and not debt_total_is_incomplete(assembled[0], lines):
        return assembled
    return None

# Operating lease liabilities (ASC 842). Prefer the direct total tag; otherwise
# sum the current + noncurrent components when they share a period end-date.
OPERATING_LEASE_DIRECT_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "OperatingLeaseLiability"),
]
OPERATING_LEASE_CURRENT_TAG: tuple[str, str] = ("us-gaap", "OperatingLeaseLiabilityCurrent")
OPERATING_LEASE_NONCURRENT_TAG: tuple[str, str] = ("us-gaap", "OperatingLeaseLiabilityNoncurrent")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except Exception:
        return None


def _facts_node(companyfacts: dict[str, Any], taxonomy: str, tag: str) -> dict[str, Any]:
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    taxonomy_node = facts.get(taxonomy) if isinstance(facts, dict) else {}
    if not isinstance(taxonomy_node, dict):
        return {}
    tag_node = taxonomy_node.get(tag)
    return tag_node if isinstance(tag_node, dict) else {}


def _unit_preference_key(
    unit: str, *, expected_prefixes: tuple[str, ...], expected_exact: tuple[str, ...]
) -> tuple[int, str]:
    unit_norm = str(unit or "").strip().lower()
    if unit_norm in {u.lower() for u in expected_exact}:
        return (0, unit_norm)
    if any(unit_norm.startswith(prefix.lower()) for prefix in expected_prefixes):
        return (1, unit_norm)
    return (2, unit_norm)


def _best_fact_for_tag(
    *,
    companyfacts: dict[str, Any],
    taxonomy: str,
    tag: str,
    as_of_date: str,
    expected_unit_exact: tuple[str, ...],
    expected_unit_prefixes: tuple[str, ...],
) -> dict[str, Any] | None:
    tag_node = _facts_node(companyfacts, taxonomy, tag)
    units = tag_node.get("units") if isinstance(tag_node.get("units"), dict) else {}
    if not isinstance(units, dict):
        return None

    asof_dt = _parse_date(as_of_date)
    if asof_dt is None:
        return None

    expected_units = {unit.strip().lower() for unit in expected_unit_exact}
    candidates: list[tuple[datetime, datetime, str, str, float, str, str]] = []
    # (end_date, filed_date, form, frame, value, unit, ref)
    for unit_key in sorted(
        units.keys(),
        key=lambda u: _unit_preference_key(
            u, expected_prefixes=expected_unit_prefixes, expected_exact=expected_unit_exact
        ),
    ):
        if str(unit_key).strip().lower() not in expected_units:
            continue
        rows = units.get(unit_key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("val")
            if not _is_num(value):
                continue
            end_date = _parse_date(str(row.get("end") or ""))
            if end_date is None or end_date > asof_dt:
                continue
            filed_date = _parse_date(str(row.get("filed") or ""))
            if filed_date is None or filed_date > asof_dt:
                continue
            form = str(row.get("form") or "")
            frame = str(row.get("frame") or "")
            filed_raw = str(row.get("filed") or "")
            accn_raw = str(row.get("accn") or "")
            filed_fragment = f",filed={filed_raw}" if filed_raw else ""
            accn_fragment = f",accn={accn_raw}" if accn_raw else ""
            ref = (
                f"companyfacts.{taxonomy}.{tag}[end_date={end_date.date().isoformat()},"
                f"unit={unit_key}{filed_fragment}{accn_fragment}]"
            )
            candidates.append((end_date, filed_date, form, frame, float(value), str(unit_key), ref))

    if not candidates:
        return None

    # Stable deterministic tie-break:
    # latest end_date, then latest filed_date, then lexical form/frame.
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[5]), reverse=True)
    chosen = candidates[0]
    return {
        "value": float(chosen[4]),
        "unit": chosen[5],
        "fact_end_date": chosen[0].date().isoformat(),
        "filed_date": chosen[1].date().isoformat(),
        "taxonomy": taxonomy,
        "tag": tag,
        "derived_from": [chosen[6]],
    }


def _merge_refs(*refs: list[str]) -> list[str]:
    out: list[str] = []
    for group in refs:
        for ref in group:
            token = str(ref).strip()
            if token and token not in out:
                out.append(token)
    return out


def _metric_resolution_state(fact: dict[str, Any] | None, *, derived: bool = False) -> str:
    if isinstance(fact, dict) and _is_num(fact.get("value")):
        return "DERIVED" if derived else "RESOLVED"
    return "UNAVAILABLE"


def _fact_recency_key(
    fact: dict[str, Any], *, priority_index: int = 0, prefer_direct: bool = False
) -> tuple[datetime, datetime, int, int]:
    end_dt = _parse_date(str(fact.get("fact_end_date") or "")) or datetime.min
    filed_dt = _parse_date(str(fact.get("filed_date") or "")) or datetime.min
    direct_tiebreak = 1 if prefer_direct else 0
    return (end_dt, filed_dt, direct_tiebreak, -priority_index)


def _choose_freshest_fact(
    candidates: list[tuple[dict[str, Any], int, bool]],
) -> dict[str, Any] | None:
    valid = [
        (fact, priority_index, prefer_direct)
        for fact, priority_index, prefer_direct in candidates
        if isinstance(fact, dict) and _is_num(fact.get("value"))
    ]
    if not valid:
        return None
    chosen, _, _ = max(
        valid,
        key=lambda item: _fact_recency_key(
            item[0],
            priority_index=item[1],
            prefer_direct=item[2],
        ),
    )
    return chosen


def _extract_from_priority(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...],
    expected_unit_prefixes: tuple[str, ...],
) -> dict[str, Any] | None:
    candidates: list[tuple[dict[str, Any], int, bool]] = []
    for index, (taxonomy, tag) in enumerate(priority):
        fact = _best_fact_for_tag(
            companyfacts=companyfacts,
            taxonomy=taxonomy,
            tag=tag,
            as_of_date=as_of_date,
            expected_unit_exact=expected_unit_exact,
            expected_unit_prefixes=expected_unit_prefixes,
        )
        if isinstance(fact, dict):
            candidates.append((fact, index, False))
    return _choose_freshest_fact(candidates)


def extract_cash_equivalents_asof(
    companyfacts: dict[str, Any], as_of_date: str
) -> dict[str, Any] | None:
    fact = _extract_from_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=CASH_EQUIVALENTS_TAG_PRIORITY,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    if isinstance(fact, dict):
        fact["resolution"] = _metric_resolution_state(fact)
    return fact


def extract_short_term_investments_asof(
    companyfacts: dict[str, Any], as_of_date: str, *, cash_fact: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The short-term investments to count beside ``cash_fact`` in net debt.

    Read only at the cash's own balance-sheet date and only from filings public by
    ``as_of_date``; the rule is ``short_term_investments_addition``. None when the
    cash is unknown or nothing is to be added.
    """
    if not isinstance(cash_fact, dict) or not _is_num(cash_fact.get("value")):
        return None
    end_date = str(cash_fact.get("fact_end_date") or "")
    if str(cash_fact.get("unit") or "").strip().upper() != "USD" or not end_date:
        return None
    lines, facts = _usd_lines_at(
        companyfacts, SHORT_TERM_INVESTMENT_TAGS, as_of_date=as_of_date, end_date=end_date
    )
    added = short_term_investments_addition(
        lines, cash_tag=str(cash_fact.get("tag") or ""), cash_value=float(cash_fact["value"])
    )
    if added is None:
        return None
    amount, tags = added
    return {
        "value": amount,
        "unit": "USD",
        "fact_end_date": end_date,
        "filed_date": max(str(facts[tag]["filed_date"]) for tag in tags),
        "taxonomy": "us-gaap" if len(tags) == 1 else "derived",
        "tag": tags[0] if len(tags) == 1 else None,
        "tags": list(tags),
        "derivation": short_term_investments_derivation(
            tags, cash_tag=str(cash_fact.get("tag") or "")
        ),
        "resolution": "RESOLVED",
        "derived_from": _merge_refs(*(facts[tag]["derived_from"] for tag in tags)),
    }


def extract_total_debt_asof(companyfacts: dict[str, Any], as_of_date: str) -> dict[str, Any] | None:
    debt_current = _best_fact_for_tag(
        companyfacts=companyfacts,
        taxonomy=DEBT_CURRENT_TAG[0],
        tag=DEBT_CURRENT_TAG[1],
        as_of_date=as_of_date,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    if debt_current is None:
        # A disclosed current LT-debt component must not be dropped. Prefer
        # inclusive DebtCurrent whenever present to avoid counting it twice.
        current_long_term = _best_fact_for_tag(
            companyfacts=companyfacts,
            taxonomy="us-gaap",
            tag="LongTermDebtCurrent",
            as_of_date=as_of_date,
            expected_unit_exact=("USD",),
            expected_unit_prefixes=("usd",),
        )
        if isinstance(current_long_term, dict):
            same_period_aggregate = False
            for taxonomy, tag in [("us-gaap", "LongTermDebt"), *TOTAL_DEBT_DIRECT_TAG_PRIORITY]:
                aggregate = _best_fact_for_tag(
                    companyfacts=companyfacts,
                    taxonomy=taxonomy,
                    tag=tag,
                    as_of_date=as_of_date,
                    expected_unit_exact=("USD",),
                    expected_unit_prefixes=("usd",),
                )
                if (
                    isinstance(aggregate, dict)
                    and aggregate.get("fact_end_date") == current_long_term.get("fact_end_date")
                ):
                    same_period_aggregate = True
                    break
            # A later-filed narrow component must not displace an existing
            # same-period aggregate. Conflicting tag scopes require separate
            # reconciliation; this fallback does not sum other short-term debt.
            if not same_period_aggregate:
                debt_current = current_long_term
    debt_noncurrent = _best_fact_for_tag(
        companyfacts=companyfacts,
        taxonomy=LONG_TERM_DEBT_NONCURRENT_TAG[0],
        tag=LONG_TERM_DEBT_NONCURRENT_TAG[1],
        as_of_date=as_of_date,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    derived_fact: dict[str, Any] | None = None
    if (
        isinstance(debt_current, dict)
        and isinstance(debt_noncurrent, dict)
        and _is_num(debt_current.get("value"))
        and _is_num(debt_noncurrent.get("value"))
        and str(debt_current.get("fact_end_date") or "")
        and str(debt_current.get("fact_end_date") or "")
        == str(debt_noncurrent.get("fact_end_date") or "")
    ):
        end_date = str(debt_current.get("fact_end_date") or "") or None
        filed_date = (
            max(
                str(debt_current.get("filed_date") or ""),
                str(debt_noncurrent.get("filed_date") or ""),
            )
            or None
        )
        derived_fact = {
            "value": float(debt_current["value"]) + float(debt_noncurrent["value"]),
            "unit": str(debt_current.get("unit") or debt_noncurrent.get("unit") or "USD"),
            "fact_end_date": end_date,
            "filed_date": filed_date,
            "taxonomy": "derived",
            "tag": f"{debt_current['tag']}_plus_LongTermDebtNoncurrent",
            "derived_from": _merge_refs(
                list(debt_current.get("derived_from") or []),
                list(debt_noncurrent.get("derived_from") or []),
            ),
            "computation": "SUM_COMPONENTS",
            "resolution": "DERIVED",
            "components": {
                "debt_current": debt_current,
                "long_term_debt_noncurrent": debt_noncurrent,
            },
        }
    direct_fact = _extract_from_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=TOTAL_DEBT_DIRECT_TAG_PRIORITY,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    fallback_fact = _extract_from_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=TOTAL_DEBT_COMPONENT_FALLBACK_TAG_PRIORITY,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    if isinstance(direct_fact, dict):
        direct_fact["resolution"] = _metric_resolution_state(direct_fact)
    if isinstance(fallback_fact, dict):
        fallback_fact["resolution"] = _metric_resolution_state(fallback_fact)
    chosen = _choose_freshest_fact(
        ([(direct_fact, 0, True)] if isinstance(direct_fact, dict) else [])
        + ([(derived_fact, 1, False)] if isinstance(derived_fact, dict) else [])
        + ([(fallback_fact, 2, False)] if isinstance(fallback_fact, dict) else [])
    )
    if isinstance(chosen, dict) and _is_num(chosen.get("value")):
        end_date = str(chosen.get("fact_end_date") or "")
        lines, facts_at_end = _debt_lines_at(
            companyfacts, as_of_date=as_of_date, end_date=end_date
        )
        chosen_tags = (
            [str(component["tag"]) for component in chosen["components"].values()]
            if isinstance(chosen.get("components"), dict)
            else [str(chosen.get("tag") or "")]
        )
        resolved = resolve_complete_total_debt(
            float(chosen["value"]), tags=chosen_tags, lines=lines
        )
        if resolved is None:
            # Completeness cannot be established from the raw tags: the same balance
            # sheet shows more debt than the candidate admits, and its own parts do not
            # assemble a total that does. UNKNOWN, never the narrow amount.
            return None
        total, tags = resolved
        if set(tags) != set(chosen_tags):
            parts = {tag: facts_at_end[tag] for tag in tags}
            chosen = {
                "value": float(total),
                "unit": "USD",
                "fact_end_date": end_date,
                "filed_date": max(str(part.get("filed_date") or "") for part in parts.values())
                or None,
                "taxonomy": "derived",
                "tag": "_plus_".join(tags),
                "derived_from": _merge_refs(
                    *[list(part.get("derived_from") or []) for part in parts.values()]
                ),
                "computation": "SUM_COMPONENTS",
                "resolution": "DERIVED",
                "components": parts,
            }
            return chosen
    if chosen is derived_fact and isinstance(chosen, dict):
        chosen["resolution"] = "DERIVED"
    return chosen


def _debt_lines_at(
    companyfacts: dict[str, Any], *, as_of_date: str, end_date: str
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """DEBT_COMPLETENESS_TAGS reported for the balance-sheet date ``end_date``: per tag the
    latest value filed on or before ``as_of_date`` (USD only), as (tag -> value, tag ->
    fact in the chooser's shape)."""
    return _usd_lines_at(
        companyfacts, DEBT_COMPLETENESS_TAGS, as_of_date=as_of_date, end_date=end_date
    )


def _usd_lines_at(
    companyfacts: dict[str, Any],
    tags: tuple[str, ...],
    *,
    as_of_date: str,
    end_date: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """``tags`` reported for the balance-sheet date ``end_date``: per tag the latest value
    filed on or before ``as_of_date`` (USD only), as (tag -> value, tag -> fact in the
    chooser's shape)."""
    asof_dt = _parse_date(as_of_date)
    end_dt = _parse_date(end_date)
    if asof_dt is None or end_dt is None:
        return {}, {}
    lines: dict[str, float] = {}
    facts: dict[str, dict[str, Any]] = {}
    for tag in tags:
        units = _facts_node(companyfacts, "us-gaap", tag).get("units")
        rows = units.get("USD") if isinstance(units, dict) else None
        best: tuple[datetime, dict[str, Any]] | None = None
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not _is_num(row.get("val")):
                continue
            if _parse_date(str(row.get("end") or "")) != end_dt:
                continue
            filed_dt = _parse_date(str(row.get("filed") or ""))
            if filed_dt is None or filed_dt > asof_dt:
                continue
            if best is None or filed_dt >= best[0]:
                best = (filed_dt, row)
        if best is not None:
            filed_dt, row = best
            accn = str(row.get("accn") or "")
            lines[tag] = float(row["val"])
            facts[tag] = {
                "value": float(row["val"]),
                "unit": "USD",
                "fact_end_date": end_dt.date().isoformat(),
                "filed_date": filed_dt.date().isoformat(),
                "taxonomy": "us-gaap",
                "tag": tag,
                "derived_from": [
                    f"companyfacts.us-gaap.{tag}[end_date={end_dt.date().isoformat()},unit=USD,"
                    f"filed={filed_dt.date().isoformat()}" + (f",accn={accn}" if accn else "") + "]"
                ],
            }
    return lines, facts


def extract_operating_lease_liability_asof(
    companyfacts: dict[str, Any], as_of_date: str
) -> dict[str, Any] | None:
    """Extract the operating lease liability (ASC 842) as of ``as_of_date``.

    FIX 1: net-debt parity. The inline valuation_writer path treats operating lease
    liabilities as debt-like (LEASE_ADJUSTED); the as-of net-debt proxy must use the
    same definition. Prefer the direct ``OperatingLeaseLiability`` total; otherwise
    sum ``OperatingLeaseLiabilityCurrent`` + ``OperatingLeaseLiabilityNoncurrent``
    when both report the same period end-date. Returns None when unavailable.
    """
    lease_current = _best_fact_for_tag(
        companyfacts=companyfacts,
        taxonomy=OPERATING_LEASE_CURRENT_TAG[0],
        tag=OPERATING_LEASE_CURRENT_TAG[1],
        as_of_date=as_of_date,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    lease_noncurrent = _best_fact_for_tag(
        companyfacts=companyfacts,
        taxonomy=OPERATING_LEASE_NONCURRENT_TAG[0],
        tag=OPERATING_LEASE_NONCURRENT_TAG[1],
        as_of_date=as_of_date,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    derived_fact: dict[str, Any] | None = None
    if (
        isinstance(lease_current, dict)
        and isinstance(lease_noncurrent, dict)
        and _is_num(lease_current.get("value"))
        and _is_num(lease_noncurrent.get("value"))
        and str(lease_current.get("fact_end_date") or "")
        and str(lease_current.get("fact_end_date") or "")
        == str(lease_noncurrent.get("fact_end_date") or "")
    ):
        end_date = str(lease_current.get("fact_end_date") or "") or None
        filed_date = (
            max(
                str(lease_current.get("filed_date") or ""),
                str(lease_noncurrent.get("filed_date") or ""),
            )
            or None
        )
        derived_fact = {
            "value": float(lease_current["value"]) + float(lease_noncurrent["value"]),
            "unit": str(lease_current.get("unit") or lease_noncurrent.get("unit") or "USD"),
            "fact_end_date": end_date,
            "filed_date": filed_date,
            "taxonomy": "derived",
            "tag": "OperatingLeaseLiabilityCurrent_plus_Noncurrent",
            "derived_from": _merge_refs(
                list(lease_current.get("derived_from") or []),
                list(lease_noncurrent.get("derived_from") or []),
            ),
            "computation": "SUM_COMPONENTS",
            "resolution": "DERIVED",
            "components": {
                "operating_lease_liability_current": lease_current,
                "operating_lease_liability_noncurrent": lease_noncurrent,
            },
        }
    direct_fact = _extract_from_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=OPERATING_LEASE_DIRECT_TAG_PRIORITY,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    if isinstance(direct_fact, dict):
        direct_fact["resolution"] = _metric_resolution_state(direct_fact)
    chosen = _choose_freshest_fact(
        ([(direct_fact, 0, True)] if isinstance(direct_fact, dict) else [])
        + ([(derived_fact, 1, False)] if isinstance(derived_fact, dict) else [])
    )
    if chosen is derived_fact and isinstance(chosen, dict):
        chosen["resolution"] = "DERIVED"
    return chosen


def extract_shares_outstanding_asof(
    companyfacts: dict[str, Any], as_of_date: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The as-of share count every per-share consumer should use: (fact or None, guard record).

    The chooser's own pick over SHARES_TAG_PRIORITY (freshest period end, then latest filed
    date, then tag priority -- unchanged), checked by the share-count guard in
    app/market/shares_guard.py against the same filing's other share bases and the
    company's own history. Untouched companies get the chooser's fact object verbatim; a
    refused count comes back as None with the reason in the record, never as a substitute.
    """
    unguarded = _extract_from_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=SHARES_TAG_PRIORITY,
        expected_unit_exact=("shares",),
        expected_unit_prefixes=("shares",),
    )
    return guard_shares_fact(
        companyfacts, as_of_date, unguarded=unguarded, priority=SHARES_TAG_PRIORITY
    )


def extract_company_facts_asof(companyfacts: dict[str, Any], as_of_date: str) -> dict[str, Any]:
    shares, shares_guard = extract_shares_outstanding_asof(companyfacts, as_of_date)
    cfo = _extract_from_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=CFO_TAG_PRIORITY,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    capex = _extract_from_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=CAPEX_TAG_PRIORITY,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    fcf_direct = _extract_from_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=FCF_DIRECT_TAG_PRIORITY,
        expected_unit_exact=("USD",),
        expected_unit_prefixes=("usd",),
    )
    cash_equivalents = extract_cash_equivalents_asof(companyfacts, as_of_date)
    total_debt = extract_total_debt_asof(companyfacts, as_of_date)

    fcf_derived: dict[str, Any] | None = None
    fcf: dict[str, Any] | None = None
    if isinstance(fcf_direct, dict):
        fcf_direct = {
            **fcf_direct,
            "computation": "DIRECT",
            "resolution": "RESOLVED",
            "reason_code": "COMPANYFACTS_FCF_DIRECT_HIT",
        }
    if (
        isinstance(cfo, dict)
        and isinstance(capex, dict)
        and _is_num(cfo.get("value"))
        and _is_num(capex.get("value"))
        and str(cfo.get("fact_end_date") or "")
        and str(cfo.get("fact_end_date") or "") == str(capex.get("fact_end_date") or "")
    ):
        # Deduct spending magnitude; retain the reported signs in the source
        # facts and bridge context, including a negative operating cash flow.
        fcf_value = float(cfo["value"]) - abs(float(capex["value"]))
        fact_end = str(cfo.get("fact_end_date") or "") or None
        filed_date = (
            max(str(cfo.get("filed_date") or ""), str(capex.get("filed_date") or "")) or None
        )
        fcf_derived = {
            "value": float(fcf_value),
            "unit": str(cfo.get("unit") or capex.get("unit") or "USD"),
            "fact_end_date": fact_end or None,
            "filed_date": filed_date,
            "taxonomy": "derived",
            "tag": "fcf_cfo_minus_capex",
            "derived_from": _merge_refs(
                list(cfo.get("derived_from") or []),
                list(capex.get("derived_from") or []),
            ),
            "computation": "CFO_MINUS_CAPEX",
            "resolution": "DERIVED",
            "reason_code": "COMPANYFACTS_CFO_CAPEX_HIT",
            "bridge_context": {
                "formula": "FCF = CFO - abs(CapEx)",
                "cfo_value": float(cfo["value"]),
                "cfo_tag": str(cfo.get("tag") or ""),
                "capex_value": float(capex["value"]),
                "capex_tag": str(capex.get("tag") or ""),
                "period_end": fact_end or None,
            },
        }
    fcf = _choose_freshest_fact(
        ([(fcf_direct, 0, True)] if isinstance(fcf_direct, dict) else [])
        + ([(fcf_derived, 1, False)] if isinstance(fcf_derived, dict) else [])
    )

    if isinstance(shares, dict):
        shares["resolution"] = _metric_resolution_state(shares)
    if isinstance(cfo, dict):
        cfo["resolution"] = _metric_resolution_state(cfo)
    if isinstance(capex, dict):
        capex["resolution"] = _metric_resolution_state(capex)

    return {
        "shares_outstanding_asof": shares,
        "shares_outstanding_guard": shares_guard,
        "cfo_asof": cfo,
        "capex_asof": capex,
        "fcf_asof": fcf,
        "cash_equivalents_asof": cash_equivalents,
        "total_debt_asof": total_debt,
    }
