"""Canonical decision-header model and derivation helpers.

This module is pure: no DB access, no config import, no I/O. Every
decision-first surface (analyst memo, sector memo pack, daily digest, investor
CLI) renders from :func:`render_decision_block_markdown` so the action verb,
buy-price target, and reconciliation note are derived in exactly one place.

Verdict source: the price-trigger STATUS drives the
action verb. ``DEPLOY_READY`` with a non-AVOID grade is "Review at target";
``AVOID`` grade is "Pass"; everything else "Wait for $<target>". The conviction
GRADE rides alongside as sizing/trust and never gates the action. A
``PRICE_DATA_SUSPECT`` or ``QUARANTINE`` status is never trustworthy enough to
render the at-target action.

Language policy: the deploy flag is an
attention router, not a buy signal — no rendered surface may phrase it as a
buy directive or claim an established edge for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Price-trigger statuses that may surface the at-target review action. A status
# that is not in this set can never produce an at-target line (suspect and
# quarantined anchors are not trustworthy).
_BUY_ELIGIBLE_STATUSES = frozenset({"DEPLOY_READY", "BUY_CONFIRMED"})

# The single action verb rendered when the price trigger fires. Compared by
# identity at every consumer site — never inline the literal elsewhere.
ACTION_REVIEW_AT_TARGET = "Review at target"

# Fixed platform time-horizon string (no per-name horizon is stored).
DEFAULT_TIME_HORIZON = "3-5 yr"


@dataclass
class DecisionBlock:
    """The decision-first header shared across every rendering surface."""

    action: str
    current_price: float | None = None
    buy_price_target: float | None = None
    # Signed fraction; negative means the current price is already below target.
    pct_to_target: float | None = None
    base_case_expected_return: float | None = None
    conviction_grade: str | None = None
    confidence: str | None = None
    price_trigger_status: str | None = None
    time_horizon: str | None = None
    what_would_change_my_mind: list[str] = field(default_factory=list)
    verdict_reconciliation_note: str | None = None


def _format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.2f}"


def action_from_status_and_grade(
    status: str | None,
    grade: str | None,
    buy_price_target: float | None = None,
    *,
    price_trigger_eligible: bool = True,
) -> str:
    """Derive the action verb from the price-trigger STATUS and conviction GRADE.

    STATUS drives the verb; GRADE only vetoes via AVOID.

    - ``AVOID`` grade -> "Pass" (a quality reject, regardless of price).
    - ``DEPLOY_READY`` status (and not AVOID) -> ``ACTION_REVIEW_AT_TARGET``.
    - anything else -> "Wait for $<buy_price_target>".

    ``PRICE_DATA_SUSPECT`` and ``QUARANTINE`` are not in the buy-eligible set so
    they fall through to "Wait" and never render the at-target action.
    """

    if grade == "AVOID":
        return "Pass"
    if price_trigger_eligible and status in _BUY_ELIGIBLE_STATUSES:
        return ACTION_REVIEW_AT_TARGET
    return f"Wait for {_format_money(buy_price_target)}"


def pct_to_target(
    current_price: float | None, buy_price_target: float | None
) -> float | None:
    """Signed fraction (current - target) / target.

    Negative means the price is already below (through) the target. Returns
    ``None`` when the target is missing or non-positive, or when the current
    price is missing.
    """

    if current_price is None or buy_price_target is None:
        return None
    if buy_price_target <= 0:
        return None
    return (current_price - buy_price_target) / buy_price_target


def buy_now_imperative_line(
    ticker: str,
    current_price: float | None,
    buy_price_target: float | None,
    base_case_expected_return: float | None,
    conviction_grade: str | None,
) -> str:
    """Render the at-target attention line for a DEPLOY_READY non-AVOID row.

    The deploy flag routes attention; it is not a buy signal, and the line says
    so explicitly because it can render standalone (memo pack, investor CLI).

    Exact format with a calibrated return:
        ``AT TARGET CRTO: $18.33 <= target $32.19 (+34.0%/yr, ACTIONABLE) — review trigger, not a buy signal``

    When no calibrated return is available, OMIT the per-year
    token rather than fabricate it:
        ``AT TARGET CRTO: $18.33 <= target $32.19 (ACTIONABLE) — review trigger, not a buy signal``
    """

    paren_parts: list[str] = []
    if base_case_expected_return is not None:
        paren_parts.append(f"{base_case_expected_return:+.1%}/yr")
    if conviction_grade is not None:
        paren_parts.append(conviction_grade)
    suffix = f" ({', '.join(paren_parts)})" if paren_parts else ""
    return (
        f"AT TARGET {ticker}: {_format_money(current_price)} <= "
        f"target {_format_money(buy_price_target)}{suffix}"
        " — review trigger, not a buy signal"
    )


def render_decision_block_markdown(block: DecisionBlock) -> str:
    """Render the canonical fixed-order markdown decision block.

    The first line is always ``## Decision`` and the second is always
    ``- ACTION: <action>`` so downstream callers can assert exact positions.
    """

    lines: list[str] = ["## Decision", f"- ACTION: {block.action}"]

    if block.buy_price_target is not None:
        lines.append(f"- BUY PRICE TARGET: {_format_money(block.buy_price_target)}")
    if block.current_price is not None:
        lines.append(f"- CURRENT PRICE: {_format_money(block.current_price)}")
    if block.pct_to_target is not None:
        lines.append(f"- % TO TARGET: {block.pct_to_target:+.1%}")

    conviction_bits: list[str] = []
    if block.conviction_grade is not None:
        conviction_bits.append(block.conviction_grade)
    if block.confidence is not None:
        conviction_bits.append(f"confidence {block.confidence}")
    if conviction_bits:
        lines.append(f"- CONVICTION: {' / '.join(conviction_bits)}")

    if block.time_horizon is not None:
        lines.append(f"- TIME HORIZON: {block.time_horizon}")

    for item in block.what_would_change_my_mind:
        lines.append(f"- WHAT WOULD CHANGE MY MIND: {item}")

    if block.verdict_reconciliation_note is not None:
        lines.append(f"- NOTE: {block.verdict_reconciliation_note}")

    return "\n".join(lines)
