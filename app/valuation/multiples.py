from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, TypeGuard

from app.fundamentals.normalize import UNKNOWN

# Authored fallback bands, already ordered (low, base, high). Used only when a
# ticker has no usable multiple history of its own.
_DEFAULT_EV_FCF_BAND = (8.0, 12.0, 16.0)
_DEFAULT_EV_EBIT_BAND = (7.0, 10.0, 14.0)

# Two methods whose base enterprise values differ by more than this factor are
# disagreeing about the business, not bracketing it. We still blend them, but say so.
_METHOD_DISAGREEMENT_RATIO = 2.0

_UNKNOWN_TRIPLE: dict[str, Any] = {"low": UNKNOWN, "base": UNKNOWN, "high": UNKNOWN}


@dataclass
class ValuationRange:
    low: float | str
    base: float | str
    high: float | str


def _is_finite_number(value: Any) -> TypeGuard[float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _refusal(warnings: list[str]) -> tuple[dict[str, Any], list[str]]:
    """The shape returned when no method is computable (mirrors dcf_lite)."""
    return (
        {
            "ev_range": dict(_UNKNOWN_TRIPLE),
            "equity_value_range": dict(_UNKNOWN_TRIPLE),
            "per_share_range": dict(_UNKNOWN_TRIPLE),
            "confidence": "LOW",
        },
        warnings,
    )


def _multiple_band(
    history: list[float],
    default_band: tuple[float, float, float],
    multiple_label: str,
    warnings: list[str],
) -> tuple[tuple[float, float, float], bool]:
    """The (low, base, high) multiple band for one method, plus whether history was used.

    A caller's history is a *sample* of observed multiples in no guaranteed order --
    sanity_checks passes a chronological series -- so the band comes from the sample's
    order statistics, never from list position: the 25th percentile, the median and the
    75th percentile (linear interpolation between closest ranks). The interquartile band
    is used rather than min/max so one distressed or bubble year cannot set the low or
    the high on its own. Non-positive or non-finite multiples are not valuation
    multiples at all (a loss year prints them) and are dropped before the statistics.
    A single surviving observation is not a band -- it would report low == base == high,
    false precision from one year -- so fewer than two leaves us on the default band.
    """
    usable = [float(m) for m in history if _is_finite_number(m) and float(m) > 0.0]
    dropped = len(history) - len(usable)
    if dropped:
        warnings.append(f"Dropped {dropped} non-positive {multiple_label} multiple(s) from history")
    if len(usable) < 2:
        warnings.append(f"Using wide default {multiple_label} band due to insufficient history")
        return default_band, False
    low, base, high = statistics.quantiles(usable, n=4, method="inclusive")
    return (low, base, high), True


def _method_ev_band(
    driver: float | None,
    history: list[float],
    default_band: tuple[float, float, float],
    metric_label: str,
    multiple_label: str,
    warnings: list[str],
) -> tuple[tuple[float, float, float], bool] | None:
    """Enterprise-value band for one method, or None when the method is not computable.

    A non-positive driver makes the method meaningless: a negative FCF times a positive
    multiple is a negative enterprise value whose ordering inverts (the *highest*
    multiple gives the *lowest* EV), and a zero driver values the business at zero
    regardless of the band. Refuse the method rather than report either.
    """
    if not _is_finite_number(driver):
        return None
    driver_value = float(driver)
    if driver_value <= 0.0:
        warnings.append(
            f"{metric_label} is non-positive ({driver_value:,.2f}); "
            f"{multiple_label} multiples valuation is not meaningful"
        )
        return None
    band, used_history = _multiple_band(history, default_band, multiple_label, warnings)
    low_m, base_m, high_m = band
    return (driver_value * low_m, driver_value * base_m, driver_value * high_m), used_history


def compute_intrinsic_range_from_multiples(
    fcf: float | None,
    ebit_proxy: float | None,
    historical_ev_fcf: list[float],
    historical_ev_ebit: list[float],
    net_debt: float | None,
    shares_outstanding: float | None,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    if fcf is None and ebit_proxy is None:
        outputs, _ = _refusal([])
        return outputs, ["Insufficient FCF/EBIT inputs for multiples valuation"]

    fcf_result = _method_ev_band(
        fcf, historical_ev_fcf, _DEFAULT_EV_FCF_BAND, "FCF", "EV/FCF", warnings
    )
    ebit_result = _method_ev_band(
        ebit_proxy, historical_ev_ebit, _DEFAULT_EV_EBIT_BAND, "EBIT", "EV/EBIT", warnings
    )

    results = [r for r in (fcf_result, ebit_result) if r is not None]
    if not results:
        warnings.append("No computable multiples method; intrinsic range UNKNOWN")
        return _refusal(warnings)

    used_history = any(used for _, used in results)
    if len(results) == 1:
        low_ev, base_ev, high_ev = results[0][0]
    else:
        # Combination rule: an equal-weight blend of the two methods, statistic by
        # statistic -- low with low, base with base, high with high. Pooling every
        # point into one sorted list (the previous behaviour) let the FCF method set
        # the low and the EBIT method the high, so the reported "range" was the
        # disagreement between two models rather than the uncertainty inside either.
        (fcf_low, fcf_base, fcf_high), _ = results[0]
        (ebit_low, ebit_base, ebit_high), _ = results[1]
        low_ev = (fcf_low + ebit_low) / 2.0
        base_ev = (fcf_base + ebit_base) / 2.0
        high_ev = (fcf_high + ebit_high) / 2.0
        lower_base, higher_base = sorted((fcf_base, ebit_base))
        if lower_base > 0.0 and higher_base / lower_base > _METHOD_DISAGREEMENT_RATIO:
            warnings.append(
                f"FCF and EBIT multiples disagree by {higher_base / lower_base:.1f}x "
                f"(base EV {fcf_base:,.0f} vs {ebit_base:,.0f}); blended band is uncertain"
            )

    ev_range = {"low": low_ev, "base": base_ev, "high": high_ev}

    # Net debt is what turns an enterprise value into an equity value. Without it the
    # enterprise value is NOT an equity proxy -- using it as one overstates equity by the
    # whole of net debt for any levered company -- so refuse, as dcf_lite does.
    net_debt_value = float(net_debt) if _is_finite_number(net_debt) else None
    if net_debt_value is None:
        equity_value_range: dict[str, Any] = dict(_UNKNOWN_TRIPLE)
        warnings.append("Net debt UNKNOWN for equity-value multiples")
    else:
        equity_value_range = {
            "low": low_ev - net_debt_value,
            "base": base_ev - net_debt_value,
            "high": high_ev - net_debt_value,
        }

    shares = float(shares_outstanding) if _is_finite_number(shares_outstanding) else None
    if shares is None or shares <= 0.0:
        warnings.append("Shares outstanding UNKNOWN; per-share range unavailable")
        per_share: dict[str, Any] = dict(_UNKNOWN_TRIPLE)
    elif net_debt_value is None:
        per_share = dict(_UNKNOWN_TRIPLE)
    else:
        per_share = {
            "low": equity_value_range["low"] / shares,
            "base": equity_value_range["base"] / shares,
            "high": equity_value_range["high"] / shares,
        }

    outputs = {
        "ev_range": ev_range,
        "equity_value_range": equity_value_range,
        "per_share_range": per_share,
        "confidence": "MEDIUM" if used_history and net_debt_value is not None else "LOW",
    }
    return outputs, warnings
