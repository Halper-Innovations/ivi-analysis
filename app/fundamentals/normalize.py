from __future__ import annotations

UNKNOWN = "UNKNOWN"


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def maybe_unknown(value: float | int | None) -> float | int | str:
    return UNKNOWN if value is None else value
