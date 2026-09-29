from __future__ import annotations

import pytest

from app.backtest.asof import effective_asof_cutoff, assert_no_future_rows


def test_effective_asof_cutoff_subtracts_filing_lag():
    # cutoff is the latest period_end allowed to be visible at as_of_date.
    assert effective_asof_cutoff("2024-07-01", filing_lag_days=90) == "2024-04-02"


def test_assert_no_future_rows_raises_on_leak():
    rows = [{"period_end": "2024-12-31"}, {"period_end": "2025-06-30"}]
    with pytest.raises(AssertionError):
        assert_no_future_rows(rows, as_of_date="2025-01-01", key="period_end")


def test_assert_no_future_rows_passes_when_clean():
    rows = [{"period_end": "2024-12-31"}, {"period_end": "2024-09-30"}]
    assert assert_no_future_rows(rows, as_of_date="2025-01-01", key="period_end") is None
