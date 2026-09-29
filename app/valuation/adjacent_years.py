"""Calendar-adjacent fiscal-year runs.

A "recent window" taken as the newest N ROWS silently reaches back over a gap:
2016 spliced onto 2021-2024 reads as a four-year history. The valuation writer
(``_n_years``) and owner-earnings quality anchor on the newest year and bound
the window by calendar years; this helper is the same rule for the modules that
need a contiguous run: keep only the run of consecutive fiscal years that ends
at the latest year on file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def trailing_adjacent_run(
    items: Iterable[T], year_of: Callable[[T], Any] = lambda item: item[0]  # type: ignore[index]
) -> list[T]:
    """The items whose fiscal years form the consecutive run ending at the latest year.

    Returned oldest first. One item per year is expected; if a year repeats the
    repeats are all kept with that year. Items without a usable year are ignored.
    """
    dated: list[tuple[int, T]] = []
    for item in items:
        try:
            year = int(year_of(item))
        except (TypeError, ValueError):
            continue
        dated.append((year, item))
    if not dated:
        return []
    dated.sort(key=lambda pair: pair[0])
    run: list[tuple[int, T]] = [dated[-1]]
    for year, item in reversed(dated[:-1]):
        if year == run[0][0]:
            run.insert(0, (year, item))
        elif year == run[0][0] - 1:
            run.insert(0, (year, item))
        else:
            break
    return [item for _year, item in run]
