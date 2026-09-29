from __future__ import annotations

from datetime import datetime, timezone

from app.research.research_quality import _age_in_days

ANCHOR = datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_item_published_later_on_the_as_of_day_is_age_zero():
    """The as-of date is a calendar day; its midnight anchor must not exclude an item
    published at noon that day (it used to, giving the neutral freshness score)."""
    assert _age_in_days("2026-03-01T12:00:00Z", ANCHOR) == 0.0


def test_next_day_item_is_still_excluded():
    assert _age_in_days("2026-03-02T00:00:00Z", ANCHOR) is None
    assert _age_in_days("2026-03-05T09:00:00Z", ANCHOR) is None


def test_earlier_item_keeps_its_fractional_age():
    assert _age_in_days("2026-02-28T00:00:00Z", ANCHOR) == 1.0
