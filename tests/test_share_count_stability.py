"""Unit tests for app.valuation.share_count_stability.select_stable_shares.

Literal BTM/BKNG/CALM fixtures. Asserts exact literal expected
values, never recomputed from the logic under test.
"""

from app.valuation.share_count_stability import select_stable_shares


def test_btm_latest_fy_outlier_uses_trailing_median():
    # latest 7.147534 deviates 56% below trailing-3 median 16.384636,
    # exceeding the 0.5 band so the median is used.
    shares, flag = select_stable_shares(
        [(2025, 7.147534), (2024, 16.384636), (2023, 16.675529)]
    )
    assert shares == 16.384636
    assert flag == "SHARES_LATEST_FY_OUTLIER"


def test_bkng_latest_within_band_keeps_latest():
    # latest within 50% of trailing median 32.815201, keep latest.
    shares, flag = select_stable_shares(
        [(2026, 31.673346), (2025, 32.815201), (2024, 34.171027)]
    )
    assert shares == 31.673346
    assert flag is None


def test_calm_stable_keeps_latest():
    # CALM is not a shares bug; latest stable, keep it.
    shares, flag = select_stable_shares(
        [(2025, 48.497477), (2024, 48.9), (2023, 49.1)]
    )
    assert shares == 48.497477
    assert flag is None


def test_single_year_cannot_validate():
    shares, flag = select_stable_shares([(2025, 7.147534)])
    assert shares == 7.147534
    assert flag == "SHARES_SINGLE_YEAR_NO_STABILITY_CHECK"


def test_empty_series_missing():
    shares, flag = select_stable_shares([])
    assert shares == 0.0
    assert flag == "SHARES_MISSING"


# ── The corroborating count against a stable series ────────────
# Policy call (conservative, matching the single-year branch): a stable fiscal-year
# series that the newer independent count contradicts by more than 20% is refused
# (shares 0.0) -- after a split or a large issuance, which count should divide is not
# knowable from the two numbers, and dividing by the stale one silently was the defect.


def test_scale_slip_multiple_is_the_share_guards_contradiction_multiple():
    from app.market.shares_guard import SHARES_MAX_MOVE
    from app.valuation.share_count_stability import SCALE_SLIP_MULTIPLE

    assert SCALE_SLIP_MULTIPLE == SHARES_MAX_MOVE == 100.0


def test_stable_series_contradicted_by_a_newer_count_is_refused_and_named():
    # CECO Environmental: stored rows ~35.7M, the newer cover-page count 58.57M.
    shares, flag = select_stable_shares(
        [(2025, 35.666), (2024, 35.0), (2023, 34.5), (2022, 34.0)],
        corroborating_count=58.57,
    )
    assert shares == 0.0
    assert flag == "SHARES_SERIES_CONTRADICTED"


def test_stable_series_agreeing_with_the_newer_count_is_unchanged():
    shares, flag = select_stable_shares(
        [(2025, 35.666), (2024, 35.0), (2023, 34.5)], corroborating_count=36.1
    )
    assert shares == 35.666
    assert flag is None


def test_a_scale_slipped_corroborating_count_cannot_refuse_a_stable_series():
    # The newer count filed a thousandfold off (a cover-page slip in the stored
    # quarterly rows) is the slip, not the stable series.
    shares, flag = select_stable_shares(
        [(2025, 23.1), (2024, 22.9), (2023, 22.4)], corroborating_count=23_544.479
    )
    assert shares == 23.1
    assert flag == "SHARES_CORROBORATING_COUNT_SCALE_SLIP"


def test_a_thousandfold_jump_corroborated_by_a_count_slipped_alike_is_not_a_capital_event():
    # The latest FY row and the newer count carry the same x1,000 slip; before, the
    # agreement "corroborated" it as a capital event and the slip divided.
    shares, flag = select_stable_shares(
        [(2025, 89_213.394), (2024, 89.6), (2023, 89.9)], corroborating_count=89_098.647
    )
    assert shares == 89.9
    assert flag == "SHARES_LATEST_FY_OUTLIER"
