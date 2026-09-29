"""Point-in-time guards for the edge backtest.

Tier 1 (period_end <= as_of) lives in companyfacts_rows / _load_facts and is
correct for the live path too. Tier 2 here is the backtest-only filing-lag: a
fiscal year is only "filed" (and thus knowable) once period_end + lag has passed,
so a backtest reconstruction as-of T uses an EARLIER cutoff than T.
"""
from __future__ import annotations

from datetime import datetime, timedelta

DEFAULT_FILING_LAG_DAYS = 90


def effective_asof_cutoff(as_of_date: str, *, filing_lag_days: int = DEFAULT_FILING_LAG_DAYS) -> str:
    """The latest period_end visible at ``as_of_date`` under the filing-lag rule.

    A FY is visible only if ``period_end + filing_lag_days <= as_of_date``, i.e.
    ``period_end <= as_of_date - filing_lag_days``. Returns that cutoff date.
    """
    d = datetime.strptime(str(as_of_date), "%Y-%m-%d").date()
    return (d - timedelta(days=int(filing_lag_days))).isoformat()


def assert_no_future_rows(rows, *, as_of_date: str, key: str = "period_end") -> None:
    """Raise AssertionError if any row's ``key`` date is after ``as_of_date``.

    The look-ahead guard used in tests to pin that no future-dated fundamental
    leaked into an as-of-T computation.
    """
    cutoff = str(as_of_date)
    for row in rows:
        value = row[key]
        if value is not None and str(value) > cutoff:
            raise AssertionError(f"look-ahead: {key}={value} is after as_of={cutoff}")
