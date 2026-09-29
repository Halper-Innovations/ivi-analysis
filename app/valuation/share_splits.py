"""Share-count breaks and the filed splits that explain them.

A year-over-year share-count move outside [2/3, 3/2], or within 3% of a clean
split factor, is a BREAK: something other than ordinary issuance or buybacks
may explain it. Until 2026-09-29 every break was read as a split and dropped,
so a 50% equity raise (100 -> 150, exactly a "3-for-2" ratio) vanished from
the dilution figure and a heavy diluter read as flat. A break is now a split
only when a FILED split corroborates it: a stock-split conversion ratio in the
issuer's companyfacts, dated within a year of the break, whose ratio matches
the observed move. A corroborated split is split-adjusted (earlier counts are
multiplied by the filed ratio, so the whole window is measured on today's
basis); an uncorroborated break leaves the rate UNKNOWN
(``SHARE_COUNT_BREAK_UNCORROBORATED``), never "low dilution".
"""

from __future__ import annotations

import math
from typing import Any

REASON_SHARE_COUNT_BREAK_UNCORROBORATED = "SHARE_COUNT_BREAK_UNCORROBORATED"

_SHARE_COUNT_JUMP_BOUNDS = (2.0 / 3.0, 1.5)
# Clean split factors (and their reverses): 3-for-2, 2, 3, 4, 5, 10, 20 for 1.
# A year-over-year count ratio within _SPLIT_FACTOR_TOLERANCE of one of these is
# a break even when it sits inside the [2/3, 3/2] band the jump bounds use
# (a 3-for-2 split is a ratio of exactly 1.5).
_CLEAN_SPLIT_FACTORS = (1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
_SPLIT_FACTOR_TOLERANCE = 0.03

# The us-gaap concepts that carry a filed split ratio (current and deprecated).
SPLIT_RATIO_CONCEPTS = (
    "StockholdersEquityNoteStockSplitConversionRatio1",
    "StockholdersEquityNoteStockSplitConversionRatio",
)
# A filed ratio corroborates a break when the observed year-over-year move is
# within 10% of it: the count also moves by the year's ordinary issuance and
# buybacks, so a 2-for-1 year with a 3% buyback is 1.94, not 2.00.
_SPLIT_MATCH_TOLERANCE = 0.10
# The filed ratio's period end may fall in the fiscal year before or after the
# year whose count shows the move (fiscal years that are not calendar years, and
# a split effective after year end but before the count was taken).
_SPLIT_YEAR_SLACK = 1


def _is_num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_share_count_break(older: float, newer: float) -> bool:
    """A year-over-year move a split, a reverse split, a large raise or a mis-scaled count explains."""
    ratio = newer / older
    low, high = _SHARE_COUNT_JUMP_BOUNDS
    if not (low <= ratio <= high):
        return True
    for factor in _CLEAN_SPLIT_FACTORS:
        for clean in (factor, 1.0 / factor):
            if abs(ratio / clean - 1.0) <= _SPLIT_FACTOR_TOLERANCE:
                return True
    return False


def split_ratio_rows_from_companyfacts(
    companyfacts: dict[str, Any] | None,
    as_of_date: str | None = None,
) -> list[dict[str, Any]]:
    """Filed split ratios as ``[{"year", "value", "derived_from"}]``.

    ``year`` is the calendar year of the fact's period end. Facts filed or
    ending after ``as_of_date`` are ignored (point in time).
    """
    if not isinstance(companyfacts, dict):
        return []
    facts = companyfacts.get("facts")
    usgaap = facts.get("us-gaap") if isinstance(facts, dict) else None
    if not isinstance(usgaap, dict):
        return []
    cutoff = str(as_of_date or "")[:10]
    seen: set[tuple[int, float]] = set()
    rows: list[dict[str, Any]] = []
    for concept in SPLIT_RATIO_CONCEPTS:
        node = usgaap.get(concept)
        units = node.get("units") if isinstance(node, dict) else None
        if not isinstance(units, dict):
            continue
        for unit, entries in units.items():
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict) or not _is_num(entry.get("val")):
                    continue
                ratio = float(entry["val"])
                end = str(entry.get("end") or "")[:10]
                filed = str(entry.get("filed") or "")[:10]
                if ratio <= 0.0 or ratio == 1.0 or len(end) < 4 or not end[:4].isdigit():
                    continue
                if cutoff and (end > cutoff or (filed and filed > cutoff)):
                    continue
                key = (int(end[:4]), round(ratio, 9))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "year": int(end[:4]),
                        "value": ratio,
                        "derived_from": [
                            f"companyfacts:us-gaap:{concept}:{unit}:{end}:{entry.get('accn') or ''}"
                        ],
                    }
                )
    return sorted(rows, key=lambda row: (row["year"], row["value"]))


def corroborating_split_ratio(
    older: float,
    newer: float,
    year: int,
    split_rows: list[dict[str, Any]] | None,
) -> float | None:
    """The filed split factor (oriented to the observed move) that explains ``older -> newer``, or None."""
    if older <= 0 or newer <= 0:
        return None
    observed = newer / older
    for row in split_rows or []:
        if not isinstance(row, dict) or not _is_num(row.get("value")):
            continue
        if abs(int(row.get("year") or 0) - int(year)) > _SPLIT_YEAR_SLACK:
            continue
        filed = float(row["value"])
        if filed <= 0.0 or filed == 1.0:
            continue
        # Filers tag a 1-for-10 reverse split as 10 or as 0.1; the observed
        # direction settles which way it went.
        for factor in (filed, 1.0 / filed):
            if abs(observed / factor - 1.0) <= _SPLIT_MATCH_TOLERANCE:
                return factor
    return None


def split_adjust_share_series(
    series: list[tuple[int, float]],
    split_rows: list[dict[str, Any]] | None,
) -> tuple[list[tuple[int, float]], list[int], list[int]]:
    """Split-adjust a positive, year-ordered share series to its newest basis.

    Returns ``(adjusted_series, split_years, uncorroborated_break_years)``. When
    ``uncorroborated_break_years`` is non-empty the adjusted series still spans
    those breaks unadjusted, and the caller must treat the rate as UNKNOWN.
    """
    ordered = sorted(((int(y), float(v)) for y, v in series), key=lambda item: item[0])
    adjusted = [value for _year, value in ordered]
    split_years: list[int] = []
    uncorroborated: list[int] = []
    for index in range(1, len(ordered)):
        older = ordered[index - 1][1]
        year, newer = ordered[index]
        if not is_share_count_break(older, newer):
            continue
        factor = corroborating_split_ratio(older, newer, year, split_rows)
        if factor is None:
            uncorroborated.append(year)
            continue
        split_years.append(year)
        for earlier in range(index):
            adjusted[earlier] *= factor
    return (
        [(year, adjusted[i]) for i, (year, _value) in enumerate(ordered)],
        split_years,
        uncorroborated,
    )
