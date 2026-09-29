"""The share-count guard: a composition over the unguarded share chooser.

Ported from a census-tested guard in a sibling tool (whose resolver was itself
ported from this repository's own chooser), adapted to this
repository's names, units and staleness bound. Not a replacement for the chooser: the
chooser in `app/market/company_facts_extract.py` (freshest period end, then latest filed
date, then tag priority) is untouched and its sort order stays as it is. This module only
decides which of that chooser's candidates to trust. The resolved count always comes from
the chooser's own candidates -- the three share tags in `SHARES_TAG_PRIORITY`, in the
chooser's own order.

Why. Cover pages are occasionally filed a thousandfold off, in EITHER direction, and every
per-share number scales with the slip: ResMed's FY2021 cover read 145,681 for 145.6
million (on both its 10-K and its 10-K/A); Chesapeake Utilities' 10-Q covers from late
2023 to late 2025 read billions for millions while its 10-K covers were right; Packaging
Corp's 10-K cover of 2026-02-26 read 89.2 billion for 89.2 million. The slips are in the
filings themselves, not in the chooser's sort: a filed-first order was measured to fix
none of the three. The guard, in the order applied to each candidate:

  0. NaN, infinity, booleans and non-positive values are not share counts.
  1. STALE: a count whose period end is more than SHARES_STALE_DAYS before the as-of is
     not the current count. Next candidate, or UNKNOWN.
  2. FALL-THROUGH AGE: a candidate reached because a fresher one was rejected is never
     accepted from an earlier fiscal year than the rejected one. UNKNOWN instead.
  3. REFERENCES, same filing (same accession number): the balance-sheet count
     (CommonStockSharesOutstanding), the diluted weighted-average count, and the count
     implied by net income / earnings per share for one period of that filing (see
     below). References only, never the resolved count. A reference within
     SHARES_CORROBORATION_X of the candidate corroborates; one SHARES_MAX_MOVE or more
     away contradicts. The references vote: more contradictions than corroborations
     rejects, more corroborations than contradictions accepts regardless of history, and a
     tie (or nothing decisive) falls to the history test with a note. The balance-sheet
     and diluted counts are share tags from the same statements, filed under one scale
     convention (Aemetis, Aehr and, in 2016, Garmin state both in thousands beside a
     correct cover), so on a question of scale they are one class of evidence: for a
     cover-page candidate they cast one vote between them (none when they point opposite
     ways); for a statement-tag candidate a same-statement corroboration does not vote
     (it shares the candidate's convention), while a contradiction does. Against a
     cover-page count the income reference (a period average, measured before the
     cover's date) confirms a contradiction the statements make but never refuses the
     cover alone: after a large issuance between the two dates it sits far below a
     correct cover. Where it corroborates a cover it accepts it, whatever the statements
     or the cover's history say: computed from two USD figures, it cannot share a share-
     tag scale slip, and a period average within 2x cannot sit beside a thousandfold one.
  4. HISTORY: the count must not move more than SHARES_MAX_MOVE against the median of the
     company's last SHARES_PRIOR_WINDOW filings on the same tag. With no prior filing and
     no deciding reference the count is accepted unchecked, and the record says so.

The income reference. A filing's diluted weighted-average count is itself sometimes filed
on another scale -- McDonald's and Ashland file it in millions (713.5 "shares"), Nutanix,
Cytokinetics and Suburban Propane in thousands, Landmark Bancorp a thousandfold too large --
and alone it then contradicted, and so refused, a correct cover count. Net income divided by
diluted earnings per share (NetIncomeLoss, else ProfitLoss; the two facts for the same
duration in the same filing) is an independent third measure of the same weighted-average
count, computed from two USD figures rather than read from a share tag, so a share-tag
scale error cannot reach it. Diluted EPS first; only when no usable diluted pair exists,
basic (EarningsPerShareBasicAndDiluted, then EarningsPerShareBasic), a basic
weighted-average count. A pair is unusable when |EPS| < INCOME_EPS_MIN_ABS (rounding to
the cent would dominate) or when the two signs disagree. The freshest period end is used,
then the longest duration. Against a cover-page count it is asymmetric (see step 3):
where it agrees with the cover it accepts it over a mis-scaled diluted count; against a
slipped cover it joins the statements and the slip is refused as before (ResMed,
Chesapeake Utilities, Packaging Corp); it never refuses a cover alone.

Pre-conversion references. Before an IPO the balance sheet carries a handful of common
shares beside the convertible preferred in temporary equity (Kailera's first 10-Q: 29,953
common, 78.8 million preferred), and its weighted averages and per-share loss are built on
that handful; the cover page, dated after the offering, counts the converted shares
(129.6 million). When the same filing reports more temporary-equity shares
(TemporaryEquitySharesOutstanding) than common shares at its balance-sheet date, the
references measured before the candidate's own date do not describe the same capital
structure: they are recorded as "pre_conversion" and do not vote. The candidate then
goes to history (none, for a first filing: accepted unchecked, with the note).

The median of a four-filing window rather than the single prior filing, because the slips
come in runs (ResMed's 10-K and 10-K/A; Chesapeake's consecutive 10-Qs): against the single
prior filing a second slip passes at a ratio of 1.0.

SHARES_MAX_MOVE = 100: the slips are a thousandfold; the largest real forward split on a
reference roster is a 50-for-1. There is no absolute "reasonable share
count" band in any unit -- an absolute ceiling in the wrong unit refused every listed
company once (see app/valuation/guards.py).

Adapted, and why:
  - Units: the chooser returns a raw count in `shares`; this module works in raw counts and
    returns a fact in the chooser's own shape, so `app/valuation/facts.py` still divides by
    a million in one place.
  - Staleness: 400 days, the bound this repository already uses for a share count that can
    no longer anchor a claim (`app/autonomous/cap_resolver.py::STALE_SHARES_MAX_AGE_DAYS`,
    pinned equal by a test rather than imported, so the market layer keeps no dependency on
    the autonomous layer).
  - Replaced: the original guard's third reference, a release-level point-in-time market
    share count. Nothing equivalent reaches the chooser here; the one market-derived count
    this repository has (market cap / price, off by default) is circular for a market-cap
    computation. The income reference above takes its place. Two bases slipped alike in the
    same filing (cover AND balance sheet x1,000) still pass unless the diluted
    weighted-average count or the income reference in that filing outvotes them.
  - Recording: every decision is in the returned record; a refused candidate is an explicit
    reason code (SHARES_STALE, SHARES_CONTRADICTED, SHARES_DISCONTINUITY,
    SHARES_FALLTHROUGH_TOO_OLD, SHARES_NOT_A_COUNT), never a silent substitute.

When no candidate is rejected the result is the unguarded chooser's fact object itself --
same value, same dates, same references -- so the guard is invisible on every company it
does not touch.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Any

SHARES_MAX_MOVE = 100.0  # the multiple at which a reference contradicts / history refuses a count
SHARES_CORROBORATION_X = 2.0  # a same-filing reference within this multiple corroborates
SHARES_PRIOR_WINDOW = 4  # history: the median of this many prior filings on the same tag
# A count whose period end is older than this against the as-of is not the current count.
# Equal to app/autonomous/cap_resolver.py::STALE_SHARES_MAX_AGE_DAYS (one filing cycle, the
# filing lag, a buffer); tests/test_shares_guard.py pins the two equal.
SHARES_STALE_DAYS = 400
FALLTHROUGH_MAX_AGE_DAYS = 366  # a fall-through candidate must be from the rejected one's fiscal year
# How references that disagree with EACH OTHER are settled: "majority" (the references vote;
# a tie falls to history with the note -- the original guard's default) or
# "contradiction_wins" (any contradiction rejects; found by census to reject twice-corroborated
# counts wherever the diluted weighted-average count is filed in thousands or millions).
REFERENCES_DECIDE = "majority"

BALANCE_SHEET_TAG = ("us-gaap", "CommonStockSharesOutstanding")
DILUTED_WAD_TAG = ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")
TEMPORARY_EQUITY_SHARES_TAG = ("us-gaap", "TemporaryEquitySharesOutstanding")
# The income reference: net income / EPS for the same duration of the same filing.
NET_INCOME_TAGS: tuple[str, ...] = ("NetIncomeLoss", "ProfitLoss")
EPS_TAGS: tuple[tuple[str, str], ...] = (
    ("net_income_per_diluted_eps", "EarningsPerShareDiluted"),
    ("net_income_per_basic_eps", "EarningsPerShareBasicAndDiluted"),
    ("net_income_per_basic_eps", "EarningsPerShareBasic"),
)
INCOME_EPS_MIN_ABS = 0.05  # below this, rounding EPS to the cent dominates the quotient
# Share tags from the financial statements, filed under one scale convention.
_STATEMENT_BASES = frozenset({"balance_sheet", "diluted_weighted_average"})
_STATEMENT_TAGS = frozenset(
    {BALANCE_SHEET_TAG[1], "CommonStockOtherSharesOutstanding", DILUTED_WAD_TAG[1]}
)

# Outcomes recorded in the guard record.
GUARD_PASS = "PASS"  # the unguarded pick, verbatim
GUARD_FELL_THROUGH = "FELL_THROUGH"  # a fresher candidate was refused; a later one accepted
GUARD_TYPE_EXCLUDED = "TYPE_EXCLUDED"  # the unguarded pick was not a share count
GUARD_REFUSED = "REFUSED"  # no candidate survived: UNKNOWN
GUARD_NO_CANDIDATE = "NO_CANDIDATE"  # no share tag at all: the unguarded chooser also found nothing
GUARD_NOT_EVALUATED = "NOT_EVALUATED"  # no payload or no parseable as-of

GUARDED_HIT_REASON = "COMPANYFACTS_HIT_GUARDED"


def _is_share_count(value: Any) -> bool:
    """A share count is a finite, positive number and not a boolean."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:  # the common, zero-padded case, without strptime's cost
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _tag_units(companyfacts: dict[str, Any], taxonomy: str, tag: str) -> dict[str, Any]:
    facts = companyfacts.get("facts") if isinstance(companyfacts.get("facts"), dict) else {}
    taxonomy_node = facts.get(taxonomy) if isinstance(facts, dict) else None
    tag_node = taxonomy_node.get(tag) if isinstance(taxonomy_node, dict) else None
    units = tag_node.get("units") if isinstance(tag_node, dict) else None
    return units if isinstance(units, dict) else {}


def _rows(
    companyfacts: dict[str, Any], taxonomy: str, tag: str, asof_dt: datetime
) -> list[dict[str, Any]]:
    """Every usable row of one share tag visible at the as-of: unit `shares`, a real count,
    period end and filed date both on or before the as-of. The reference string is built
    exactly as the chooser builds it, so a fall-through fact reads like any other."""
    out: list[dict[str, Any]] = []
    for unit_key, rows in _tag_units(companyfacts, taxonomy, tag).items():
        if str(unit_key).strip().lower() != "shares" or not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not _is_share_count(row.get("val")):
                continue
            end_dt = _parse(str(row.get("end") or ""))
            filed_dt = _parse(str(row.get("filed") or ""))
            if end_dt is None or filed_dt is None or end_dt > asof_dt or filed_dt > asof_dt:
                continue
            filed_raw = str(row.get("filed") or "")
            accn = str(row.get("accn") or "")
            filed_fragment = f",filed={filed_raw}" if filed_raw else ""
            accn_fragment = f",accn={accn}" if accn else ""
            out.append(
                {
                    "value": float(row["val"]),
                    "end": end_dt.date().isoformat(),
                    "start": str(row.get("start") or ""),
                    "filed": filed_dt.date().isoformat(),
                    "form": str(row.get("form") or ""),
                    "frame": str(row.get("frame") or ""),
                    "accn": accn,
                    "taxonomy": taxonomy,
                    "tag": tag,
                    "unit": str(unit_key),
                    "ref": (
                        f"companyfacts.{taxonomy}.{tag}[end_date={end_dt.date().isoformat()},"
                        f"unit={unit_key}{filed_fragment}{accn_fragment}]"
                    ),
                }
            )
    return out


def _latest_per_filing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per distinct filing date -- the freshest count that filing carried -- oldest
    filing first."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = best.get(row["filed"])
        if current is None or (row["end"], row["form"]) > (current["end"], current["form"]):
            best[row["filed"]] = row
    return [best[filed] for filed in sorted(best)]


def _same_filing_reference(rows: list[dict[str, Any]], accn: str) -> dict[str, Any] | None:
    """The freshest value of one basis carried by the filing `accn`."""
    if not accn:
        return None
    same = [row for row in rows if row["accn"] == accn]
    if not same:
        return None
    return max(same, key=lambda row: (row["end"], row["start"], row["form"]))


def _verdict(
    candidate_value: float, reference_value: float, max_move: float, corroboration: float
) -> tuple[str, float]:
    ratio = candidate_value / reference_value if reference_value > 0 else float("inf")
    if ratio >= max_move or ratio <= 1.0 / max_move:
        return "contradicts", ratio
    if 1.0 / corroboration <= ratio <= corroboration:
        return "corroborates", ratio
    return "disagrees", ratio


def _duration_rows(
    companyfacts: dict[str, Any], tag: str, accn: str, asof_dt: datetime, *, per_share: bool
) -> dict[tuple[str, str, str], tuple[float, str]]:
    """(currency, start, end) -> (value, ref) for one us-gaap duration tag in the filing
    ``accn``. ``per_share`` reads ``<currency>/shares`` units (EPS); otherwise plain currency
    units (income). Rows filed or ending after the as-of are not visible."""
    out: dict[tuple[str, str, str], tuple[float, str]] = {}
    for unit_key, rows in _tag_units(companyfacts, "us-gaap", tag).items():
        unit = str(unit_key).strip()
        if per_share:
            if not unit.lower().endswith("/shares"):
                continue
            currency = unit[: -len("/shares")].upper()
        else:
            if "/" in unit:
                continue
            currency = unit.upper()
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or str(row.get("accn") or "") != accn:
                continue
            value = row.get("val")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            start, end = str(row.get("start") or ""), str(row.get("end") or "")
            end_dt, filed_dt = _parse(end), _parse(str(row.get("filed") or ""))
            if not start or end_dt is None or filed_dt is None:
                continue
            if end_dt > asof_dt or filed_dt > asof_dt:
                continue
            ref = f"companyfacts.us-gaap.{tag}[start={start},end_date={end},unit={unit_key},accn={accn}]"
            out[(currency, start, end)] = (float(value), ref)
    return out


def _income_reference(
    companyfacts: dict[str, Any], accn: str, asof_dt: datetime
) -> dict[str, Any] | None:
    """The weighted-average count implied by net income / EPS in the filing ``accn``: the
    same duration, the same currency. Diluted EPS first, basic only when no usable diluted
    pair exists; NetIncomeLoss first, then ProfitLoss. Freshest period end, then the
    longest duration. None when no usable pair exists."""
    if not accn:
        return None
    for basis, eps_tag in EPS_TAGS:
        eps_rows = _duration_rows(companyfacts, eps_tag, accn, asof_dt, per_share=True)
        if not eps_rows:
            continue
        for income_tag in NET_INCOME_TAGS:
            income_rows = _duration_rows(companyfacts, income_tag, accn, asof_dt, per_share=False)
            pairs = []
            for key in income_rows.keys() & eps_rows.keys():
                income, income_ref = income_rows[key]
                eps, eps_ref = eps_rows[key]
                if abs(eps) < INCOME_EPS_MIN_ABS or income == 0 or (income > 0) != (eps > 0):
                    continue
                start_dt, end_dt = _parse(key[1]), _parse(key[2])
                days = (end_dt - start_dt).days if start_dt and end_dt else 0
                pairs.append(((key[2], days), key, income, eps, income_ref, eps_ref))
            if pairs:
                _order, key, income, eps, income_ref, eps_ref = max(pairs, key=lambda item: item[0])
                return {
                    "basis": basis,
                    "value": income / eps,
                    "ref": f"{income_ref} / {eps_ref}",
                    "end": key[2],
                    "net_income": income,
                    "eps": eps,
                }
    return None


def _pre_conversion_date(
    companyfacts: dict[str, Any], accn: str, asof_dt: datetime
) -> str | None:
    """The balance-sheet date at which the filing ``accn`` reports more temporary-equity
    (convertible preferred) shares than common shares, or None. References measured before
    a later-dated candidate in such a filing describe the pre-conversion structure."""
    if not accn:
        return None
    temp_rows = [r for r in _rows(companyfacts, *TEMPORARY_EQUITY_SHARES_TAG, asof_dt) if r["accn"] == accn]
    if not temp_rows:
        return None
    common = {
        r["end"]: r["value"]
        for r in _rows(companyfacts, *BALANCE_SHEET_TAG, asof_dt)
        if r["accn"] == accn
    }
    latest = max(temp_rows, key=lambda r: r["end"])
    common_count = common.get(latest["end"])
    if common_count is None or latest["value"] <= common_count:
        return None
    return str(latest["end"])


def _judge(
    companyfacts: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    asof_dt: datetime,
    balance_rows: list[dict[str, Any]],
    wad_rows: list[dict[str, Any]],
    max_move: float,
    prior_window: int,
    corroboration: float,
    references_decide: str,
) -> dict[str, Any]:
    """Steps 3 and 4 for one candidate: the same-filing references, then history.

    Returns {"decision": "accept"|"reject", ...}: an accept carries ``by`` ("reference",
    "history" or "unchecked"), a reject its ``reason`` (SHARES_CONTRADICTED or
    SHARES_DISCONTINUITY); both carry the judged ``references`` and the vote's notes.
    """
    references: list[dict[str, Any]] = []
    if candidate["tag"] != BALANCE_SHEET_TAG[1]:
        balance = _same_filing_reference(balance_rows, candidate["accn"])
        if balance is not None:
            references.append(
                {
                    "basis": "balance_sheet",
                    "value": balance["value"],
                    "ref": balance["ref"],
                    "end": balance["end"],
                }
            )
    wad = _same_filing_reference(wad_rows, candidate["accn"])
    if wad is not None:
        references.append(
            {
                "basis": "diluted_weighted_average",
                "value": wad["value"],
                "ref": wad["ref"],
                "end": wad["end"],
            }
        )
    income = _income_reference(companyfacts, candidate["accn"], asof_dt)
    if income is not None and income["value"] > 0:
        references.append(income)
    conversion_end = (
        _pre_conversion_date(companyfacts, candidate["accn"], asof_dt) if references else None
    )
    for reference in references:
        reference["verdict"], reference["ratio"] = _verdict(
            candidate["value"], reference["value"], max_move, corroboration
        )
        if conversion_end is not None and reference["end"] < candidate["end"]:
            reference["verdict"] = "pre_conversion"
    voting = [r for r in references if r["verdict"] != "pre_conversion"]
    notes = {
        "references": references,
        "conflicted": False,
        "bases_disagreed": False,
        "pre_conversion": len(voting) < len(references),
    }
    statement_votes = [r["verdict"] for r in voting if r["basis"] in _STATEMENT_BASES]
    votes = [r["verdict"] for r in voting if r["basis"] not in _STATEMENT_BASES]
    income_corroborates_cover = False
    if candidate["tag"] in _STATEMENT_TAGS:
        # A statement-tag candidate: a same-statement corroboration shares its scale
        # convention and proves nothing about scale; a same-statement contradiction does.
        votes += [verdict for verdict in statement_votes if verdict == "contradicts"]
    else:
        # The balance-sheet and diluted counts are share tags from the same statements,
        # filed under one scale convention: on a question of scale they are one class of
        # evidence and cast one vote -- none when they point opposite ways.
        decisive = {v for v in statement_votes if v in ("contradicts", "corroborates")}
        statement_vote = decisive.pop() if len(decisive) == 1 else None
        # The income reference is a weighted average over a period that ends before the
        # cover's date; after a large issuance in between (Bollinger, Empery Digital) it
        # sits far below a correct cover. It may confirm a contradiction the statements
        # make, or tie one against the cover, but it does not refuse a cover on its own.
        votes = [
            verdict
            for verdict in votes
            if not (verdict == "contradicts" and statement_vote != "contradicts")
        ]
        if statement_vote is not None:
            votes.append(statement_vote)
        # ...but where it agrees with the cover, it settles the scale: a period average
        # within 2x of the cover cannot sit beside a thousandfold slip, and it cannot share
        # a slip with the cover, being computed from two USD figures. So it accepts the
        # cover even against the statements or a run of slipped covers in its history
        # (Garmin's 2016 10-Qs read billions; its 10-K cover is right).
        income_corroborates_cover = any(
            r["verdict"] == "corroborates" for r in voting if r["basis"] not in _STATEMENT_BASES
        )
    if references_decide == "majority":
        notes["conflicted"] = bool(
            {"contradicts", "corroborates"} <= {r["verdict"] for r in voting}
        )
    n_contra = sum(1 for verdict in votes if verdict == "contradicts")
    n_corro = sum(1 for verdict in votes if verdict == "corroborates")
    if references_decide == "majority":
        rejects = n_contra > n_corro and not income_corroborates_cover
        accepts = n_corro > n_contra or income_corroborates_cover
    else:
        n_contra = sum(1 for r in voting if r["verdict"] == "contradicts")
        n_corro = sum(1 for r in voting if r["verdict"] == "corroborates")
        rejects = n_contra > 0
        accepts = n_corro > 0 and not rejects
    if rejects:
        return {"decision": "reject", "reason": "SHARES_CONTRADICTED", **notes}
    if accepts:
        return {"decision": "accept", "by": "reference", **notes}
    # 4. history on the same tag
    notes["bases_disagreed"] = bool(voting)
    filings = _latest_per_filing(rows)
    priors = [f["value"] for f in filings if f["filed"] < candidate["filed"]][-prior_window:]
    if not priors:
        return {"decision": "accept", "by": "unchecked", **notes}
    reference_value = float(statistics.median(priors))
    ratio = candidate["value"] / reference_value
    history = {"reference": reference_value, "ratio": ratio, "priors": len(priors)}
    if (1.0 / max_move) <= ratio <= max_move:
        return {"decision": "accept", "by": "history", **history, **notes}
    return {"decision": "reject", "reason": "SHARES_DISCONTINUITY", **history, **notes}


def check_share_count(
    companyfacts: dict[str, Any],
    *,
    taxonomy: str,
    tag: str,
    value: float,
    end: str,
    filed: str,
    accn: str,
    max_move: float = SHARES_MAX_MOVE,
    prior_window: int = SHARES_PRIOR_WINDOW,
    corroboration: float = SHARES_CORROBORATION_X,
    references_decide: str = REFERENCES_DECIDE,
) -> dict[str, Any]:
    """The guard's judgement of ONE given share count (raw shares), as of its own filing.

    For callers that choose a count by their own rule -- the ingest normalizer's per
    fiscal year pick -- rather than the as-of chooser's. Steps 0, 3 and 4 apply (a count
    is judged as of the day it was filed, so staleness and fall-through do not). Returns
    {"decision": "accept"|"reject", "reason" | "by", "references", ...}. A count with no
    parseable filing date cannot be placed in history and is accepted as not evaluated.
    """
    if not _is_share_count(value):
        return {"decision": "reject", "reason": "SHARES_NOT_A_COUNT", "references": []}
    asof_dt = _parse(str(filed or "")[:10])
    end_dt = _parse(str(end or "")[:10])
    if asof_dt is None or end_dt is None:
        return {"decision": "accept", "by": "not_evaluated", "references": []}
    candidate = {
        "value": float(value),
        "end": end_dt.date().isoformat(),
        "filed": asof_dt.date().isoformat(),
        "accn": str(accn or ""),
        "tag": tag,
        "taxonomy": taxonomy,
    }
    return _judge(
        companyfacts,
        candidate,
        _rows(companyfacts, taxonomy, tag, asof_dt),
        asof_dt=asof_dt,
        balance_rows=_rows(companyfacts, *BALANCE_SHEET_TAG, asof_dt),
        wad_rows=_rows(companyfacts, *DILUTED_WAD_TAG, asof_dt),
        max_move=max_move,
        prior_window=prior_window,
        corroboration=corroboration,
        references_decide=references_decide,
    )


def _fact_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """A fact in the chooser's own shape (see `_best_fact_for_tag`)."""
    return {
        "value": float(row["value"]),
        "unit": row["unit"],
        "fact_end_date": row["end"],
        "filed_date": row["filed"],
        "taxonomy": row["taxonomy"],
        "tag": row["tag"],
        "derived_from": [row["ref"]],
    }


def guard_shares_fact(
    companyfacts: dict[str, Any],
    as_of_date: str,
    *,
    unguarded: dict[str, Any] | None,
    priority: list[tuple[str, str]],
    max_move: float = SHARES_MAX_MOVE,
    prior_window: int = SHARES_PRIOR_WINDOW,
    corroboration: float = SHARES_CORROBORATION_X,
    stale_days: int = SHARES_STALE_DAYS,
    references_decide: str = REFERENCES_DECIDE,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Check the unguarded chooser's share pick; return (fact or None, guard record).

    `unguarded` is the chooser's own pick over `priority` for the same payload and as-of.
    When nothing is refused and the guard accepts that same observation, `unguarded` is
    returned as-is. Otherwise the fact is rebuilt in the chooser's shape from the accepted
    candidate, or None when no candidate survived -- the record then carries the reason.
    """
    guard: dict[str, Any] = {
        "outcome": GUARD_NOT_EVALUATED,
        "reason_code": None,
        "detail": None,
        "max_move": max_move,
        "prior_window": prior_window,
        "corroboration": corroboration,
        "stale_days": stale_days,
        "references_decide": references_decide,
        "rejected": [],
        "accepted": None,
        "stopped": None,
        "unchecked": False,
        "bases_disagreed": False,
        "references_conflicted": False,
        "references_pre_conversion": False,
    }
    asof_dt = _parse(str(as_of_date or "").strip())
    if not isinstance(companyfacts, dict) or asof_dt is None:
        guard["detail"] = "No CompanyFacts payload or no parseable as-of date; not evaluated."
        return unguarded, guard

    candidates: list[tuple[tuple[str, str, int], dict[str, Any], list[dict[str, Any]]]] = []
    for index, (taxonomy, tag) in enumerate(priority):
        rows = _rows(companyfacts, taxonomy, tag, asof_dt)
        if not rows:
            continue
        # The chooser's own per-tag order: latest end, latest filed, then form/frame/unit.
        rows.sort(
            key=lambda r: (r["end"], r["filed"], r["form"], r["frame"], r["unit"]), reverse=True
        )
        candidates.append(((rows[0]["end"], rows[0]["filed"], -index), rows[0], rows))
    # The chooser's cross-tag order: latest end, latest filed, then tag priority.
    candidates.sort(key=lambda item: item[0], reverse=True)

    if not candidates:
        if isinstance(unguarded, dict):
            guard["outcome"] = GUARD_REFUSED
            guard["reason_code"] = "SHARES_NOT_A_COUNT"
            guard["detail"] = (
                "The only share-tag values at the as-of are not share counts "
                "(non-positive, non-finite or boolean)."
            )
            unguarded_refs = list(unguarded.get("derived_from") or [])
            guard["rejected"].append(
                {
                    "ref": unguarded_refs[0] if unguarded_refs else None,
                    "value": unguarded.get("value"),
                    "reason": "SHARES_NOT_A_COUNT",
                }
            )
            return None, guard
        guard["outcome"] = GUARD_NO_CANDIDATE
        guard["detail"] = "No share-count tag at the as-of."
        return None, guard

    balance_rows = _rows(companyfacts, *BALANCE_SHEET_TAG, asof_dt)
    wad_rows = _rows(companyfacts, *DILUTED_WAD_TAG, asof_dt)

    accepted: dict[str, Any] | None = None
    first_rejected_end: str | None = None
    for _key, candidate, rows in candidates:
        end_dt = _parse(candidate["end"])
        assert end_dt is not None  # _rows only keeps parseable dates
        # 1. stale
        age_days = (asof_dt - end_dt).days
        if age_days > stale_days:
            guard["rejected"].append(
                {
                    "ref": candidate["ref"],
                    "value": candidate["value"],
                    "reason": "SHARES_STALE",
                    "age_days": age_days,
                }
            )
            first_rejected_end = first_rejected_end or candidate["end"]
            continue
        # 2. fall-through age
        if first_rejected_end is not None:
            rejected_end_dt = _parse(first_rejected_end)
            assert rejected_end_dt is not None
            if end_dt < rejected_end_dt - timedelta(days=FALLTHROUGH_MAX_AGE_DAYS):
                guard["stopped"] = {
                    "ref": candidate["ref"],
                    "value": candidate["value"],
                    "reason": "SHARES_FALLTHROUGH_TOO_OLD",
                    "rejected_end": first_rejected_end,
                }
                break
        # 3. references in the same filing, then 4. history on the same tag
        judged = _judge(
            companyfacts,
            candidate,
            rows,
            asof_dt=asof_dt,
            balance_rows=balance_rows,
            wad_rows=wad_rows,
            max_move=max_move,
            prior_window=prior_window,
            corroboration=corroboration,
            references_decide=references_decide,
        )
        references = judged["references"]
        if judged["conflicted"]:
            guard["references_conflicted"] = True
        if judged["pre_conversion"]:
            guard["references_pre_conversion"] = True
        if judged["bases_disagreed"]:
            guard["bases_disagreed"] = True
        if judged["decision"] == "reject" and judged["reason"] == "SHARES_CONTRADICTED":
            guard["rejected"].append(
                {
                    "ref": candidate["ref"],
                    "value": candidate["value"],
                    "reason": "SHARES_CONTRADICTED",
                    "references": references,
                }
            )
            first_rejected_end = first_rejected_end or candidate["end"]
            continue
        if judged["decision"] == "accept":
            accepted = candidate
            guard["accepted"] = {
                "ref": candidate["ref"],
                "value": candidate["value"],
                "by": judged["by"],
                "references": references,
            }
            if judged["by"] == "unchecked":
                guard["unchecked"] = True
                guard["accepted"]["note"] = (
                    "No prior filing on this tag and no same-filing reference decided; "
                    "accepted unchecked."
                )
            elif judged["by"] == "history":
                for key in ("reference", "ratio", "priors"):
                    guard["accepted"][key] = judged[key]
            break
        guard["rejected"].append(
            {
                "ref": candidate["ref"],
                "value": candidate["value"],
                "reason": "SHARES_DISCONTINUITY",
                "reference": judged["reference"],
                "ratio": judged["ratio"],
                "priors": judged["priors"],
                "references": references,
            }
        )
        first_rejected_end = first_rejected_end or candidate["end"]

    if accepted is None:
        reasons = [r["reason"] for r in guard["rejected"]]
        if guard["stopped"]:
            reasons.append(guard["stopped"]["reason"])
        guard["outcome"] = GUARD_REFUSED
        guard["reason_code"] = reasons[0] if reasons else "SHARES_NOT_A_COUNT"
        guard["detail"] = (
            "No share-count candidate survived the guard (" + ", ".join(sorted(set(reasons))) + ")."
        )
        return None, guard

    same_as_unguarded = (
        isinstance(unguarded, dict)
        and unguarded.get("filed_date") == accepted["filed"]
        and unguarded.get("fact_end_date") == accepted["end"]
        and unguarded.get("tag") == accepted["tag"]
        and _is_share_count(unguarded.get("value"))
        and abs(float(unguarded["value"]) - accepted["value"]) <= 0.5
    )
    if same_as_unguarded and not guard["rejected"]:
        guard["outcome"] = GUARD_PASS
        return unguarded, guard

    guard["reason_code"] = GUARDED_HIT_REASON
    if guard["rejected"]:
        guard["outcome"] = GUARD_FELL_THROUGH
        guard["detail"] = (
            f"The guard refused {len(guard['rejected'])} fresher share-count candidate(s) ("
            + ", ".join(sorted({r["reason"] for r in guard["rejected"]}))
            + f") and accepted {accepted['ref']}."
        )
    else:
        guard["outcome"] = GUARD_TYPE_EXCLUDED
        guard["detail"] = (
            "The unguarded pick was not a share count (non-positive, non-finite or boolean); "
            f"accepted {accepted['ref']}."
        )
    return _fact_from_row(accepted), guard
