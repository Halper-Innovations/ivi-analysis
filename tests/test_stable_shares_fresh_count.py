"""A lone fiscal-year share count must not be accepted when a fresh count contradicts it.

Observed on a live store. CrowdStrike's only
fiscal-year ``shares_outstanding`` row that carries a filing date is FY2019's
47.421 million (filed 2020-03-23); the later fiscal-year rows have no filing date
and are invisible on the point-in-time path. ``select_stable_shares`` receives that
single point together with a corroborating cover-page count of 1,023.93 million
(filed 2026-08-27) and returns the seven-year-old 47.421 million with the flag
SHARES_SINGLE_YEAR_NO_STABILITY_CHECK (share_count_stability.py:55-56): the
corroborating count is never consulted on the one-year branch. The valuation stored
for CrowdStrike on 2026-09-07 divides by 47.421 and prints a DCF of $192.36 a share;
on the company's own count the same arithmetic gives about $36. Thirty-eight stored
rows across eight names since 2026-08-01 carry a single-year count that a newer
count contradicts by more than 20%. When the one filed count and an independent count
disagree by more than the module's own tolerance, the honest answer is no count.
"""

from __future__ import annotations

from app.valuation.share_count_stability import select_stable_shares


def test_a_single_year_contradicted_by_the_corroborating_count_is_refused():
    shares, flag = select_stable_shares([(2019, 47.421)], corroborating_count=1023.934842)
    assert shares == 0.0
    assert flag == "SHARES_SINGLE_YEAR_CONTRADICTED"


def test_a_single_year_the_corroborating_count_agrees_with_is_kept():
    shares, flag = select_stable_shares([(2025, 133.947444)], corroborating_count=126.759738)
    assert shares == 133.947444
    assert flag == "SHARES_SINGLE_YEAR_NO_STABILITY_CHECK"


def test_a_single_year_with_no_corroboration_is_unchanged():
    shares, flag = select_stable_shares([(2025, 7.028934)], corroborating_count=None)
    assert shares == 7.028934
    assert flag == "SHARES_SINGLE_YEAR_NO_STABILITY_CHECK"
