"""Measurement scope.

Backtest reconstruction re-runs the valuation writer with historical
prices and force_refresh — inside this scope every valuations read/write
routes to the sibling ``valuations_measurement`` table, so measurement
work can never overwrite the production rows live decisions were made on
(the platform previously hand-rolled a 117k-row manual snapshot to work
around exactly that).

Deliberately NOT keyed on price_override: the live sweep prewarm
legitimately injects prices. The scope is an explicit declaration by the
caller that this run is measurement, not research.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_MEASUREMENT_SCOPE: ContextVar[bool] = ContextVar(
    "valuation_measurement_scope", default=False
)

VALUATIONS_TABLE = "valuations"
VALUATIONS_MEASUREMENT_TABLE = "valuations_measurement"


def in_measurement_scope() -> bool:
    return bool(_MEASUREMENT_SCOPE.get())


def valuations_table() -> str:
    """Table name every valuations read/write must interpolate."""
    return VALUATIONS_MEASUREMENT_TABLE if _MEASUREMENT_SCOPE.get() else VALUATIONS_TABLE


@contextmanager
def measurement_scope() -> Iterator[None]:
    token = _MEASUREMENT_SCOPE.set(True)
    try:
        yield
    finally:
        _MEASUREMENT_SCOPE.reset(token)


__all__ = [
    "VALUATIONS_MEASUREMENT_TABLE",
    "VALUATIONS_TABLE",
    "in_measurement_scope",
    "measurement_scope",
    "valuations_table",
]
