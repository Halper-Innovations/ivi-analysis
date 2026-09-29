"""Deterministic financial-input integrity contracts for autonomous research.

The helpers in this module are deliberately provider-free.  They validate the
exact packet/scenario objects that are about to cross a reasoning boundary and
make unit, quote, split-basis, and formula lineage machine-checkable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field as dataclass_field, is_dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlparse


FINANCIAL_INTEGRITY_PASS = "PASS"
INVALID_FINANCIAL_INPUT = "INVALID_FINANCIAL_INPUT"
NEEDS_DATA = "NEEDS_DATA"
MARKET_CAP_UNIT_USD_MILLIONS = "USD_millions"
PRICE_UNIT_USD_PER_SHARE = "USD_per_share"
SHARES_UNIT_MILLIONS = "shares_millions"
PRICE_BASIS_UNADJUSTED = "UNADJUSTED"
PRICE_BASIS_SPLIT_ADJUSTED = "SPLIT_ADJUSTED"
SHARES_BASIS_UNADJUSTED = PRICE_BASIS_UNADJUSTED
# Backward-compatible import name. Issuer-reported is provenance, not a split
# basis; all new packets serialize the literal UNADJUSTED basis.
SHARES_BASIS_ISSUER_REPORTED = SHARES_BASIS_UNADJUSTED

_VALID_PRICE_BASES = frozenset({PRICE_BASIS_UNADJUSTED, PRICE_BASIS_SPLIT_ADJUSTED})
_VALID_SHARES_BASES = frozenset({SHARES_BASIS_UNADJUSTED, PRICE_BASIS_SPLIT_ADJUSTED})
_IDENTITY_SOURCE_HOST_SUFFIXES = (
    "sec.gov",
    "nasdaq.com",
    "nyse.com",
    "cboe.com",
    "otcmarkets.com",
    "tsx.com",
    "londonstockexchange.com",
)
_RATIO_SOURCE_HOST_SUFFIXES = (
    *_IDENTITY_SOURCE_HOST_SUFFIXES,
    "adrbny.com",
    "bnymellon.com",
    "citi.com",
    "db.com",
    "jpmorgan.com",
)
_SEC_SOURCE_HOST_SUFFIXES = ("sec.gov",)
_EXCHANGE_SOURCE_HOST_SUFFIXES = tuple(
    suffix for suffix in _IDENTITY_SOURCE_HOST_SUFFIXES if suffix not in _SEC_SOURCE_HOST_SUFFIXES
)
_DEPOSITARY_SOURCE_HOST_SUFFIXES = tuple(
    suffix for suffix in _RATIO_SOURCE_HOST_SUFFIXES if suffix not in _IDENTITY_SOURCE_HOST_SUFFIXES
)
_SPLIT_SOURCE_HOST_SUFFIXES = (
    *_IDENTITY_SOURCE_HOST_SUFFIXES,
    "bloomberg.com",
    "eodhd.com",
    "lseg.com",
    "morningstar.com",
    "refinitiv.com",
    "spglobal.com",
    "stooq.com",
)
_SEC_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SPLIT_PROOF_CACHE_SCHEMA_VERSION = "ivi.corporate_action_split_proof.v1"
SPLIT_PROOF_CACHE_RECORD_TYPE = "split_lineage_proof"
SPLIT_PROOF_RAW_CACHE_SCHEMA_VERSION = "ivi.corporate_action_raw.v1"
SPLIT_PROOF_RAW_CACHE_RECORD_TYPE = "provider_split_response"
SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT = "NO_INTERVENING_SPLIT"
SPLIT_PROOF_KIND_SPLIT_EVENT = "SPLIT_EVENT"
_INVESTMENT_METRIC_ALIASES = {
    "fcf_yield": "fcf_yield",
    "enterprise_value": "enterprise_value",
    "enterprise_value_mm": "enterprise_value",
    "ev_ebitda": "ev_to_ebitda",
    "ev_to_ebitda": "ev_to_ebitda",
    "pe": "pe",
    "p_e": "pe",
    "pe_ratio": "pe",
    "price_to_earnings": "pe",
    "pb": "price_to_book",
    "p_b": "price_to_book",
    "pb_ratio": "price_to_book",
    "price_to_book": "price_to_book",
}
_KNOWN_TRACE_METRICS = frozenset(
    {
        "market_cap_mm",
        "fcf_yield",
        "enterprise_value",
        "ev_to_ebitda",
        "pe",
        "price_to_book",
        "annualized_return",
    }
)
_CANONICAL_TRACE_FORMULAS = {
    "market_cap_mm": frozenset({"current_price * shares_outstanding_mm / issuer_quote_ratio"}),
    "fcf_yield": frozenset(
        {
            "(cfo_usd_millions - abs(capex_usd_millions)) / market_cap_usd_millions",
            "free_cash_flow_usd_millions / market_cap_usd_millions",
            # Deterministic smoke artifacts emitted before the canonical long-form
            # unit names were introduced remain explicitly recognized.
            "free_cash_flow_usd_mm / market_cap_usd_mm",
        }
    ),
    "enterprise_value": frozenset(
        {
            "market_cap_usd_millions + debt_usd_millions - cash_usd_millions",
            "market_cap_usd_mm + total_debt_usd_mm - cash_usd_mm",
        }
    ),
    "ev_to_ebitda": frozenset(
        {
            "enterprise_value_usd_millions / ebitda_usd_millions",
            "enterprise_value_usd_mm / ebitda_usd_mm",
        }
    ),
    "pe": frozenset(
        {
            "market_cap_usd_millions / net_income_usd_millions",
            "current_price / earnings_per_share",
            "market_cap_usd_mm / net_income_usd_mm",
        }
    ),
    "price_to_book": frozenset(
        {
            "market_cap_usd_millions / equity_usd_millions",
            "current_price / book_value_per_share",
            "market_cap_usd_mm / equity_usd_mm",
        }
    ),
    "annualized_return": frozenset(
        {
            "round((estimated_future_value_per_share / current_price) ** (1 / horizon_years) - 1, 6)",
            "(future_value_per_share / current_price) ** (1 / years) - 1",
        }
    ),
}
_CANONICAL_TRACE_OUTPUT_UNITS = {
    "market_cap_mm": frozenset({MARKET_CAP_UNIT_USD_MILLIONS}),
    "fcf_yield": frozenset({"ratio"}),
    "enterprise_value": frozenset({MARKET_CAP_UNIT_USD_MILLIONS}),
    "ev_to_ebitda": frozenset({"ratio", "multiple"}),
    "pe": frozenset({"ratio", "multiple"}),
    "price_to_book": frozenset({"ratio", "multiple"}),
    "annualized_return": frozenset({"annualized_ratio"}),
}
_CANONICAL_TRACE_INPUT_UNITS = {
    "current_price": PRICE_UNIT_USD_PER_SHARE,
    "price_usd_per_share": PRICE_UNIT_USD_PER_SHARE,
    "estimated_future_value_per_share": PRICE_UNIT_USD_PER_SHARE,
    "future_value_per_share": PRICE_UNIT_USD_PER_SHARE,
    "earnings_per_share": PRICE_UNIT_USD_PER_SHARE,
    "eps": PRICE_UNIT_USD_PER_SHARE,
    "book_value_per_share": PRICE_UNIT_USD_PER_SHARE,
    "shares_outstanding_mm": SHARES_UNIT_MILLIONS,
    "shares_millions": SHARES_UNIT_MILLIONS,
    "issuer_quote_ratio": "ratio",
    "horizon_years": "years",
    "years": "years",
    "market_cap_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "market_cap_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "market_cap_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "free_cash_flow_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "free_cash_flow_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "free_cash_flow_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "cfo_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "cfo_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "cfo_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "capex_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "capex_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "capex_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "debt_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "debt_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "debt_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "cash_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "cash_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "cash_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "enterprise_value_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "enterprise_value_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "enterprise_value_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "ebitda_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "ebitda_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "ebitda_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "net_income_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "net_income_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "net_income_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "equity_usd_millions": MARKET_CAP_UNIT_USD_MILLIONS,
    "equity_usd_mm": MARKET_CAP_UNIT_USD_MILLIONS,
    "equity_mm": MARKET_CAP_UNIT_USD_MILLIONS,
}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    payload = getattr(value, "__dict__", None)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _finite_number(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0):
        return None
    return numeric


def _canonical_number(value: Any) -> str | None:
    numeric = _finite_number(value)
    if numeric is None:
        return None
    try:
        decimal = Decimal(str(numeric)).normalize()
    except InvalidOperation:
        return None
    return format(decimal, "f")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_quote_hash(
    snapshot: Mapping[str, Any] | None = None,
    **fields: Any,
) -> str:
    """Return a stable SHA-256 identity for one immutable quote snapshot.

    The function accepts either a mapping or keyword fields.  Only quote
    identity fields participate, so callers cannot accidentally make the ID
    depend on surrounding packet order or mutable presentation text.
    """

    raw = {**dict(snapshot or {}), **fields}
    price = raw.get(
        "price",
        raw.get("value", raw.get("current_price", raw.get("price_used"))),
    )
    price_basis = (
        str(raw.get("price_basis") or raw.get("current_price_basis") or "").strip().upper()
    )
    raw_price = raw.get("raw_price")
    split_adjustment_factor = raw.get("split_adjustment_factor")
    if price_basis == PRICE_BASIS_UNADJUSTED:
        raw_price = price if raw_price is None else raw_price
        split_adjustment_factor = (
            1.0 if split_adjustment_factor is None else split_adjustment_factor
        )
    payload = {
        "ticker": str(raw.get("ticker") or "").strip().upper(),
        "as_of_date": str(
            raw.get("as_of_date")
            or raw.get("price_as_of_date")
            or raw.get("current_price_as_of_date")
            or ""
        ).strip()[:10],
        "price": _canonical_number(price),
        "currency": str(
            raw.get("currency")
            or raw.get("price_currency")
            or raw.get("current_price_currency")
            or ""
        )
        .strip()
        .upper(),
        "source": str(
            raw.get("source") or raw.get("price_source") or raw.get("current_price_source") or ""
        ).strip(),
        "source_url": str(
            raw.get("source_url")
            or raw.get("url")
            or raw.get("price_source_url")
            or raw.get("current_price_source_url")
            or ""
        ).strip(),
        "price_basis": price_basis,
        "raw_price": _canonical_number(raw_price),
        "split_adjustment_factor": _canonical_number(split_adjustment_factor),
        "split_effective_date": str(raw.get("split_effective_date") or "").strip()[:10],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


stable_quote_snapshot_id = stable_quote_hash


def metric_values_reconcile(
    expected: Any,
    actual: Any,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    expected_number = _finite_number(expected)
    actual_number = _finite_number(actual)
    return bool(
        expected_number is not None
        and actual_number is not None
        and math.isclose(expected_number, actual_number, rel_tol=rel_tol, abs_tol=abs_tol)
    )


def canonical_metric_trace(
    *,
    metric: str,
    formula: str,
    inputs: Mapping[str, Any],
    output: Any,
    output_unit: str,
    recomputed_output: Any | None = None,
    quote_snapshot_id: str | None = None,
    input_provenance: Mapping[str, Any] | None = None,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> dict[str, Any]:
    """Build one canonical, serializable formula-provenance record."""

    expected = output if recomputed_output is None else recomputed_output
    trace = {
        "metric": str(metric),
        "formula": str(formula),
        "inputs": json.loads(_canonical_json(dict(inputs))),
        "output": output,
        "output_unit": str(output_unit),
        "recomputed_output": expected,
        "reconciles": metric_values_reconcile(
            output,
            expected,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ),
        "quote_snapshot_id": quote_snapshot_id,
    }
    if input_provenance is not None:
        trace["input_provenance"] = json.loads(_canonical_json(dict(input_provenance)))
    return trace


build_metric_trace = canonical_metric_trace


def reconcile_metric_trace(
    trace: Mapping[str, Any],
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    return bool(
        trace.get("reconciles") is True
        and metric_values_reconcile(
            trace.get("output"),
            trace.get("recomputed_output"),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    )


metric_trace_reconciles = reconcile_metric_trace


def _investment_metric_occurrences(
    value: Any,
    *,
    path: str = "",
) -> list[tuple[str, str, Any]]:
    occurrences: list[tuple[str, str, Any]] = []
    if not isinstance(value, Mapping):
        return occurrences
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key in {"metric_trace", "metric_traces"} or key.startswith("raw_"):
            continue
        current_path = f"{path}.{key}" if path else key
        canonical_name = _INVESTMENT_METRIC_ALIASES.get(key.lower())
        if (
            canonical_name is not None
            and raw_value is not None
            and not isinstance(raw_value, (Mapping, list, tuple))
        ):
            occurrences.append((canonical_name, current_path, raw_value))
        if isinstance(raw_value, Mapping):
            occurrences.extend(_investment_metric_occurrences(raw_value, path=current_path))
    return occurrences


def _trace_for_metric(
    traces: Mapping[str, Any],
    *,
    canonical_name: str,
    metric_path: str,
) -> Any:
    for candidate in (
        metric_path,
        canonical_name,
        metric_path.rsplit(".", 1)[-1],
    ):
        if candidate in traces:
            return traces[candidate]
    return None


def _trace_input_number(inputs: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in inputs:
            return _finite_number(inputs.get(name))
    return None


def _canonical_trace_metric_name(metric: Any) -> str:
    normalized = str(metric or "").strip().lower()
    if normalized == "market_cap_mm":
        return normalized
    if normalized == "annualized_return":
        return normalized
    return _INVESTMENT_METRIC_ALIASES.get(normalized, normalized)


def _trace_contract_state(
    trace: Mapping[str, Any],
    *,
    expected_metric: Any,
) -> tuple[str, str, bool, str, bool, bool]:
    """Return deterministic contract state for one known metric trace.

    Formula text is intentionally validated against an exact allowlist.  The
    independent recomputation below remains the arithmetic authority; this
    check prevents arbitrary or misleading prose from being accepted as
    formula provenance merely because the trace copied its own output.
    """

    canonical_expected = _canonical_trace_metric_name(expected_metric)
    declared_raw = str(trace.get("metric") or "").strip()
    canonical_declared = _canonical_trace_metric_name(declared_raw)
    metric = (
        canonical_expected if canonical_expected in _KNOWN_TRACE_METRICS else canonical_declared
    )
    declared_matches = bool(declared_raw) and canonical_declared == metric
    formula = str(trace.get("formula") or "").strip()
    formula_valid = formula in _CANONICAL_TRACE_FORMULAS.get(metric, frozenset())
    output_unit_raw = trace.get("output_unit")
    output_unit = str(output_unit_raw or "").strip()
    output_unit_valid = output_unit in _CANONICAL_TRACE_OUTPUT_UNITS.get(metric, frozenset())
    return (
        metric,
        formula,
        formula_valid,
        output_unit,
        output_unit_valid,
        declared_matches,
    )


def _independent_trace_recompute(trace: Mapping[str, Any]) -> tuple[bool, float | None]:
    """Recompute supported finance formulas without trusting trace assertions."""

    metric = _canonical_trace_metric_name(trace.get("metric"))
    if metric not in _KNOWN_TRACE_METRICS:
        return True, None
    inputs = trace.get("inputs")
    if not isinstance(inputs, Mapping):
        return False, None
    if metric == "market_cap_mm":
        price = _trace_input_number(inputs, "current_price", "price_usd_per_share")
        shares = _trace_input_number(inputs, "shares_outstanding_mm", "shares_millions")
        ratio = _trace_input_number(inputs, "issuer_quote_ratio")
        ratio = 1.0 if ratio is None and "issuer_quote_ratio" not in inputs else ratio
        if price is None or shares is None or ratio is None or ratio <= 0:
            return False, None
        return True, price * shares / ratio
    if metric == "fcf_yield":
        market_cap = _trace_input_number(inputs, "market_cap_usd_millions", "market_cap_mm")
        free_cash_flow = _trace_input_number(
            inputs, "free_cash_flow_usd_millions", "free_cash_flow_mm"
        )
        if free_cash_flow is None:
            cfo = _trace_input_number(inputs, "cfo_usd_millions", "cfo_mm")
            capex = _trace_input_number(inputs, "capex_usd_millions", "capex_mm")
            if cfo is not None and capex is not None:
                free_cash_flow = cfo - abs(capex)
        if market_cap is None or market_cap <= 0 or free_cash_flow is None:
            return False, None
        return True, free_cash_flow / market_cap
    if metric == "enterprise_value":
        market_cap = _trace_input_number(inputs, "market_cap_usd_millions", "market_cap_mm")
        debt = _trace_input_number(inputs, "debt_usd_millions", "debt_mm")
        cash = _trace_input_number(inputs, "cash_usd_millions", "cash_mm")
        if market_cap is None or debt is None or cash is None:
            return False, None
        return True, market_cap + debt - cash
    if metric == "ev_to_ebitda":
        enterprise_value = _trace_input_number(
            inputs, "enterprise_value_usd_millions", "enterprise_value_mm"
        )
        ebitda = _trace_input_number(inputs, "ebitda_usd_millions", "ebitda_mm")
        if enterprise_value is None or ebitda is None or math.isclose(ebitda, 0.0):
            return False, None
        return True, enterprise_value / ebitda
    if metric == "pe":
        market_cap = _trace_input_number(inputs, "market_cap_usd_millions", "market_cap_mm")
        net_income = _trace_input_number(inputs, "net_income_usd_millions", "net_income_mm")
        if market_cap is not None and net_income is not None and not math.isclose(net_income, 0.0):
            return True, market_cap / net_income
        price = _trace_input_number(inputs, "current_price", "price_usd_per_share")
        eps = _trace_input_number(inputs, "earnings_per_share", "eps")
        if price is None or eps is None or math.isclose(eps, 0.0):
            return False, None
        return True, price / eps
    if metric == "price_to_book":
        market_cap = _trace_input_number(inputs, "market_cap_usd_millions", "market_cap_mm")
        equity = _trace_input_number(inputs, "equity_usd_millions", "equity_mm")
        if market_cap is not None and equity is not None and not math.isclose(equity, 0.0):
            return True, market_cap / equity
        price = _trace_input_number(inputs, "current_price", "price_usd_per_share")
        book_per_share = _trace_input_number(inputs, "book_value_per_share")
        if price is None or book_per_share is None or math.isclose(book_per_share, 0.0):
            return False, None
        return True, price / book_per_share
    future_value = _trace_input_number(inputs, "estimated_future_value_per_share")
    current_price = _trace_input_number(inputs, "current_price")
    horizon = _trace_input_number(inputs, "horizon_years")
    if (
        future_value is None
        or future_value <= 0
        or current_price is None
        or current_price <= 0
        or horizon is None
        or horizon <= 0
    ):
        return False, None
    return True, round((future_value / current_price) ** (1.0 / horizon) - 1.0, 6)


@dataclass(frozen=True)
class FinancialIntegrityViolation:
    code: str
    ticker: str | None = None
    field: str | None = None
    source_values: dict[str, Any] = dataclass_field(default_factory=dict)
    expected_relationship: Any = None
    observed_relationship: Any = None
    reason: str = ""
    terminal_status: Literal["INVALID_FINANCIAL_INPUT", "NEEDS_DATA"] = INVALID_FINANCIAL_INPUT

    @property
    def message(self) -> str:
        """Compatibility alias for older callers."""

        return self.reason

    @property
    def observed(self) -> Any:
        """Compatibility alias for older callers."""

        return self.observed_relationship

    @property
    def expected(self) -> Any:
        """Compatibility alias for older callers."""

        return self.expected_relationship

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "message": self.message,
                "observed": self.observed,
                "expected": self.expected,
            }
        )
        return payload


@dataclass(frozen=True)
class FinancialIntegrityScope:
    context: str
    run_as_of_date: str
    packets: tuple[Any, ...] = ()
    scenarios: tuple[Any, ...] = ()


@dataclass(frozen=True)
class FinancialIntegrityGateResult:
    context: str
    run_as_of_date: str
    status: Literal["PASS", "INVALID_FINANCIAL_INPUT", "NEEDS_DATA"]
    violations: tuple[FinancialIntegrityViolation, ...] = ()
    scope_fingerprint: str = ""
    packet_count: int = 0
    scenario_count: int = 0
    ticker_snapshot_ids: dict[str, str] = dataclass_field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == FINANCIAL_INTEGRITY_PASS and not self.violations

    @property
    def is_valid(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = [item.to_dict() for item in self.violations]
        payload["passed"] = self.passed
        return payload


class InvalidFinancialInputError(RuntimeError):
    """Raised before reasoning when deterministic financial inputs are invalid."""

    def __init__(self, result: FinancialIntegrityGateResult):
        self.result = result
        self.violations = result.violations
        self.status = result.status
        codes = ", ".join(item.code for item in result.violations[:8]) or "unknown"
        super().__init__(f"{result.status}: {result.context}: {codes}")


def _violation(
    violations: list[FinancialIntegrityViolation],
    code: str,
    message: str,
    *,
    ticker: str | None = None,
    field_name: str | None = None,
    observed: Any = None,
    expected: Any = None,
    source_values: Mapping[str, Any] | None = None,
    expected_relationship: Any = None,
    observed_relationship: Any = None,
    reason: str | None = None,
    terminal_status: Literal["INVALID_FINANCIAL_INPUT", "NEEDS_DATA"] = (INVALID_FINANCIAL_INPUT),
) -> None:
    normalized_sources = dict(source_values or {})
    if not normalized_sources and field_name is not None:
        normalized_sources[field_name] = observed
    violations.append(
        FinancialIntegrityViolation(
            code=code,
            ticker=ticker,
            field=field_name,
            source_values=normalized_sources,
            expected_relationship=(
                expected if expected_relationship is None else expected_relationship
            ),
            observed_relationship=(
                observed if observed_relationship is None else observed_relationship
            ),
            reason=reason or message,
            terminal_status=terminal_status,
        )
    )


def _valid_iso_at_or_before(value: Any, cutoff: str) -> bool:
    try:
        observed_date = date.fromisoformat(str(value or "")[:10])
        cutoff_date = date.fromisoformat(str(cutoff or "")[:10])
    except ValueError:
        return False
    return observed_date <= cutoff_date


def _valid_authoritative_reference(
    value: Any,
    *,
    host_suffixes: tuple[str, ...],
) -> bool:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower().split(":", 1)[0]
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in host_suffixes)


def _strict_canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_split_proof_cache_root() -> Path:
    from app.config import get_config

    return (Path(get_config().cache_dir) / "corporate_actions" / "v1").resolve()


def _configured_cache_root() -> Path:
    from app.config import get_config

    return Path(get_config().cache_dir).resolve()


def _split_source_provider(source_reference: Any) -> str | None:
    parsed = urlparse(str(source_reference or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return None
    host = parsed.netloc.lower().split(":", 1)[0]
    if host == "sec.gov" or host.endswith(".sec.gov"):
        return "sec"
    if host == "eodhd.com" or host.endswith(".eodhd.com"):
        return "eodhd"
    return None


def _split_proof_kind(record: Mapping[str, Any]) -> str | None:
    explicit = str(record.get("proof_kind") or "").strip().upper()
    if explicit in {
        SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT,
        SPLIT_PROOF_KIND_SPLIT_EVENT,
    }:
        return explicit
    if all(record.get(key) is not None for key in ("factor", "effective_date", "filed_date")):
        return SPLIT_PROOF_KIND_SPLIT_EVENT
    if str(record.get("status") or "").strip().upper() == "PASS" and all(
        record.get(key) is not None for key in ("period_start", "period_end", "verified_as_of")
    ):
        return SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT
    return None


def split_proof_raw_materialization_envelope(
    record: Mapping[str, Any],
    raw_payload: Any,
) -> dict[str, Any]:
    """Return the exact provider-response envelope a normalized proof cites."""

    source_provider = str(record.get("source_provider") or "").strip().lower()
    issuer_cik = _normalize_cik(record.get("issuer_cik"))
    provider_symbol = str(record.get("provider_symbol") or "").strip().upper()
    source_reference = str(record.get("source_reference") or "").strip()
    retrieved_at = str(record.get("retrieved_at") or "").strip()
    if (
        source_provider not in {"sec", "eodhd"}
        or issuer_cik is None
        or not provider_symbol
        or not source_reference
        or not retrieved_at
    ):
        raise ValueError("split-proof raw material requires provider, CIK, symbol, URL, and date")
    payload_sha256 = hashlib.sha256(_strict_canonical_json_bytes(raw_payload)).hexdigest()
    return {
        "schema_version": SPLIT_PROOF_RAW_CACHE_SCHEMA_VERSION,
        "record_type": SPLIT_PROOF_RAW_CACHE_RECORD_TYPE,
        "source_provider": source_provider,
        "issuer_cik": issuer_cik,
        "provider_symbol": provider_symbol,
        "source_reference": source_reference,
        "retrieved_at": retrieved_at,
        "payload_sha256": payload_sha256,
        "payload": raw_payload,
    }


def canonical_split_proof_raw_cache_path(
    record: Mapping[str, Any],
    raw_payload: Any,
) -> Path:
    """Return the configured path for exact raw provider split-response bytes."""

    envelope = split_proof_raw_materialization_envelope(record, raw_payload)
    return (
        canonical_split_proof_cache_root()
        / "raw"
        / str(envelope["source_provider"])
        / str(envelope["issuer_cik"])
        / str(envelope["provider_symbol"])
        / f"{envelope['payload_sha256']}.json"
    ).resolve()


def split_proof_materialization_envelope(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical cache envelope for one exact split-proof record."""

    normalized_record = {str(key): value for key, value in record.items()}
    ticker = str(normalized_record.get("ticker") or "").strip().upper()
    source_provider = str(normalized_record.get("source_provider") or "").strip().lower()
    issuer_cik = _normalize_cik(normalized_record.get("issuer_cik"))
    provider_symbol = str(normalized_record.get("provider_symbol") or "").strip().upper()
    proof_kind = _split_proof_kind(normalized_record)
    retrieved_at = str(normalized_record.get("retrieved_at") or "").strip()
    raw_relative_path = str(normalized_record.get("raw_relative_path") or "").strip()
    raw_sha256 = str(normalized_record.get("raw_sha256") or "").strip().lower()
    raw_payload_sha256 = str(normalized_record.get("raw_payload_sha256") or "").strip().lower()
    if (
        not ticker
        or source_provider not in {"sec", "eodhd"}
        or issuer_cik is None
        or not provider_symbol
        or proof_kind is None
        or not retrieved_at
        or not raw_relative_path
        or _SHA256_RE.fullmatch(raw_sha256) is None
        or _SHA256_RE.fullmatch(raw_payload_sha256) is None
    ):
        raise ValueError("split-proof record lacks authoritative raw-response binding")
    normalized_record["source_provider"] = source_provider
    normalized_record["issuer_cik"] = issuer_cik
    normalized_record["provider_symbol"] = provider_symbol
    normalized_record["proof_kind"] = proof_kind
    record_bytes = _strict_canonical_json_bytes(normalized_record)
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    return {
        "schema_version": SPLIT_PROOF_CACHE_SCHEMA_VERSION,
        "record_type": SPLIT_PROOF_CACHE_RECORD_TYPE,
        "ticker": ticker,
        "record_sha256": record_sha256,
        "record": normalized_record,
    }


def canonical_split_proof_cache_path(record: Mapping[str, Any]) -> Path:
    """Return the configured, identity-bound cache path for one proof record."""

    envelope = split_proof_materialization_envelope(record)
    return (
        canonical_split_proof_cache_root()
        / "proofs"
        / str(envelope["record"]["source_provider"])
        / str(envelope["record"]["issuer_cik"])
        / str(envelope["ticker"])
        / str(envelope["record"]["proof_kind"])
        / f"{envelope['record_sha256']}.json"
    ).resolve()


def _strict_iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return None


def _split_ratio(value: Any) -> float | None:
    number = _finite_number(value, positive=True)
    if number is not None:
        return number
    token = str(value or "").strip()
    if "/" not in token:
        return None
    numerator, denominator = token.split("/", 1)
    top = _finite_number(numerator, positive=True)
    bottom = _finite_number(denominator, positive=True)
    if top is None or bottom is None:
        return None
    return top / bottom


def _eodhd_split_rows(raw_payload: Any) -> list[tuple[date, float]] | None:
    if not isinstance(raw_payload, list):
        return None
    rows: list[tuple[date, float]] = []
    for item in raw_payload:
        if not isinstance(item, Mapping):
            return None
        effective_date = _strict_iso_date(item.get("date") or item.get("effective_date"))
        factor = _split_ratio(item.get("split") or item.get("factor"))
        if effective_date is None or factor is None:
            return None
        rows.append((effective_date, factor))
    return rows


def _sec_source_identity(
    source_reference: str,
) -> tuple[str, str] | None:
    parsed = urlparse(source_reference)
    match = re.fullmatch(
        r"/Archives/edgar/data/(\d+)/(\d{18})/[^/]+",
        parsed.path,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    issuer_cik = _normalize_cik(match.group(1))
    compact_accession = match.group(2)
    if issuer_cik is None:
        return None
    accession = f"{compact_accession[:10]}-{compact_accession[10:12]}-{compact_accession[12:]}"
    if _SEC_ACCESSION_RE.fullmatch(accession) is None:
        return None
    return issuer_cik, accession


def _raw_split_material_supports_record(record: Mapping[str, Any]) -> bool:
    raw_relative_path = str(record.get("raw_relative_path") or "").strip()
    raw_sha256 = str(record.get("raw_sha256") or "").strip().lower()
    raw_payload_sha256 = str(record.get("raw_payload_sha256") or "").strip().lower()
    relative = Path(raw_relative_path)
    if (
        not raw_relative_path
        or relative.is_absolute()
        or ".." in relative.parts
        or _SHA256_RE.fullmatch(raw_sha256) is None
        or _SHA256_RE.fullmatch(raw_payload_sha256) is None
    ):
        return False
    lexical_path = canonical_split_proof_cache_root() / relative
    try:
        resolved = lexical_path.resolve(strict=True)
    except OSError:
        return False
    if str(lexical_path) != str(resolved) or not resolved.is_file():
        return False
    try:
        raw_bytes = resolved.read_bytes()
    except OSError:
        return False
    if hashlib.sha256(raw_bytes).hexdigest() != raw_sha256:
        return False
    try:
        raw_envelope = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw_envelope, Mapping):
        return False
    raw_payload = raw_envelope.get("payload")
    try:
        expected_raw_envelope = split_proof_raw_materialization_envelope(record, raw_payload)
        expected_raw_path = canonical_split_proof_raw_cache_path(record, raw_payload)
    except (TypeError, ValueError, OSError):
        return False
    if (
        resolved != expected_raw_path
        or raw_relative_path
        != expected_raw_path.relative_to(canonical_split_proof_cache_root()).as_posix()
        or dict(raw_envelope) != expected_raw_envelope
        or str(expected_raw_envelope["payload_sha256"]) != raw_payload_sha256
    ):
        return False

    source_provider = str(record.get("source_provider") or "").strip().lower()
    source_reference = str(record.get("source_reference") or "").strip()
    provider_symbol = str(record.get("provider_symbol") or "").strip().upper()
    ticker = str(record.get("ticker") or "").strip().upper()
    issuer_cik = _normalize_cik(record.get("issuer_cik"))
    proof_kind = _split_proof_kind(record)
    retrieved_at = _strict_iso_date(record.get("retrieved_at"))
    if (
        issuer_cik is None
        or not provider_symbol
        or not ticker
        or proof_kind is None
        or retrieved_at is None
        or source_provider != _split_source_provider(source_reference)
    ):
        return False

    if source_provider == "eodhd":
        parsed = urlparse(source_reference)
        expected_path = f"/api/splits/{provider_symbol}".lower()
        if (
            provider_symbol != ticker
            or parsed.path.rstrip("/").lower() != expected_path
            or parsed.query
        ):
            return False
        rows = _eodhd_split_rows(raw_payload)
        if rows is None:
            return False
        if proof_kind == SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT:
            period_start = _strict_iso_date(record.get("period_start"))
            period_end = _strict_iso_date(record.get("period_end"))
            verified_as_of = _strict_iso_date(record.get("verified_as_of"))
            return bool(
                period_start is not None
                and period_end is not None
                and verified_as_of is not None
                and period_start <= period_end <= verified_as_of
                and retrieved_at == verified_as_of
                and not any(period_start <= event_date <= period_end for event_date, _ in rows)
            )
        effective_date = _strict_iso_date(record.get("effective_date"))
        factor = _finite_number(record.get("factor"), positive=True)
        filed_date = _strict_iso_date(record.get("filed_date"))
        return bool(
            effective_date is not None
            and factor is not None
            and filed_date is not None
            and filed_date <= retrieved_at
            and any(
                event_date == effective_date and metric_values_reconcile(event_factor, factor)
                for event_date, event_factor in rows
            )
        )

    if proof_kind != SPLIT_PROOF_KIND_SPLIT_EVENT or provider_symbol != ticker:
        return False
    sec_identity = _sec_source_identity(source_reference)
    if sec_identity is None:
        return False
    source_cik, source_accession = sec_identity
    accession = str(record.get("accession") or "").strip()
    if (
        source_cik != issuer_cik
        or accession != source_accession
        or not isinstance(raw_payload, Mapping)
        or _normalize_cik(raw_payload.get("issuer_cik")) != issuer_cik
        or str(raw_payload.get("accession") or "").strip() != accession
        or str(raw_payload.get("source_reference") or "").strip() != source_reference
    ):
        return False
    document_relative_path = str(raw_payload.get("document_relative_path") or "").strip()
    document_sha256 = str(raw_payload.get("document_sha256") or "").strip().lower()
    expected_document_path = (
        _configured_cache_root() / "filings" / issuer_cik / accession / "primary_document.html"
    )
    relative_document = Path(document_relative_path)
    if (
        not document_relative_path
        or relative_document.is_absolute()
        or ".." in relative_document.parts
        or _SHA256_RE.fullmatch(document_sha256) is None
    ):
        return False
    lexical_document_path = _configured_cache_root() / relative_document
    try:
        resolved_document_path = lexical_document_path.resolve(strict=True)
    except OSError:
        return False
    if (
        str(lexical_document_path) != str(resolved_document_path)
        or resolved_document_path != expected_document_path.resolve()
        or not resolved_document_path.is_file()
    ):
        return False
    try:
        document_bytes = resolved_document_path.read_bytes()
        document_text = document_bytes.decode("utf-8").strip().lower()
    except (OSError, UnicodeDecodeError):
        return False
    if hashlib.sha256(document_bytes).hexdigest() != document_sha256:
        return False
    effective_date = str(record.get("effective_date") or "").strip()[:10]
    factor = _finite_number(record.get("factor"), positive=True)
    filed_date = _strict_iso_date(record.get("filed_date"))
    if not document_text or not effective_date or factor is None or filed_date is None:
        return False
    factor_token = f"{factor:g}-for-1"
    return bool(
        factor_token in document_text
        and effective_date in document_text
        and filed_date <= retrieved_at
    )


def authoritative_split_proof_reference(
    proof: Mapping[str, Any],
    *,
    expected_ticker: str,
    expected_issuer_cik: str | None = None,
    expected_as_of_date: str | None = None,
) -> bool:
    """Bind normalized proof to exact provider material in the configured cache."""

    source_reference = str(proof.get("source_reference") or "").strip()
    parsed = urlparse(source_reference)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return False
    host = parsed.netloc.lower().split(":", 1)[0]
    if not any(
        host == suffix or host.endswith(f".{suffix}") for suffix in _SPLIT_SOURCE_HOST_SUFFIXES
    ):
        return False

    ticker = str(proof.get("ticker") or "").strip().upper()
    if not ticker or ticker != str(expected_ticker or "").strip().upper():
        return False
    proof_issuer_cik = _normalize_cik(proof.get("issuer_cik"))
    normalized_expected_cik = _normalize_cik(expected_issuer_cik)
    if proof_issuer_cik is None or (
        normalized_expected_cik is not None and proof_issuer_cik != normalized_expected_cik
    ):
        return False
    expected_as_of = _strict_iso_date(expected_as_of_date)
    retrieved_at = _strict_iso_date(proof.get("retrieved_at"))
    if expected_as_of_date is not None and (
        expected_as_of is None or retrieved_at is None or retrieved_at > expected_as_of
    ):
        return False

    raw_materialized_path = str(proof.get("materialized_path") or "").strip()
    expected_sha256 = str(proof.get("materialized_sha256") or "").strip().lower()
    if not raw_materialized_path or _SHA256_RE.fullmatch(expected_sha256) is None:
        return False
    candidate = Path(raw_materialized_path).expanduser()
    if not candidate.is_absolute():
        return False
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return False
    if str(candidate) != str(resolved) or not resolved.is_file():
        return False

    try:
        materialized_bytes = resolved.read_bytes()
    except OSError:
        return False
    if hashlib.sha256(materialized_bytes).hexdigest() != expected_sha256:
        return False
    try:
        materialized_payload = json.loads(materialized_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    expected_record = {
        str(key): value
        for key, value in proof.items()
        if str(key) not in {"materialized_path", "materialized_sha256"}
    }
    try:
        expected_envelope = split_proof_materialization_envelope(expected_record)
        expected_path = canonical_split_proof_cache_path(expected_record)
    except (TypeError, ValueError, OSError):
        return False
    return bool(
        resolved == expected_path
        and materialized_payload == expected_envelope
        and _raw_split_material_supports_record(expected_record)
    )


def _normalize_cik(value: Any) -> str | None:
    token = "".join(character for character in str(value or "") if character.isdigit())
    return token.zfill(10) if token else None


def _packet_issuer_cik(packet: Mapping[str, Any]) -> str | None:
    explicit = _normalize_cik(packet.get("issuer_cik"))
    if explicit is not None:
        return explicit
    for field_name in (
        "identity_source_url",
        "shares_source_url",
        "ratio_source_url",
    ):
        value = str(packet.get(field_name) or "")
        match = re.search(r"CIK(\d{1,10})", value, flags=re.IGNORECASE)
        if match is None:
            match = re.search(r"/edgar/data/(\d{1,10})/", value, flags=re.IGNORECASE)
        if match is not None:
            normalized = _normalize_cik(match.group(1))
            if normalized is not None:
                return normalized
    return None


def _valid_security_ratio_reference(
    packet: Mapping[str, Any],
    *,
    ticker: str,
) -> bool:
    source_url = str(packet.get("ratio_source_url") or "").strip()
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower().split(":", 1)[0]
    source = str(packet.get("identity_source") or "").strip().lower()
    if str(packet.get("ratio_security_symbol") or "").strip().upper() != ticker:
        return False
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in _SEC_SOURCE_HOST_SUFFIXES):
        accession = str(packet.get("ratio_source_accession") or "").strip()
        issuer_cik = _normalize_cik(packet.get("issuer_cik"))
        if (
            source not in {"sec_cached_annual_filing", "issuer_filing"}
            or _SEC_ACCESSION_RE.fullmatch(accession) is None
            or issuer_cik is None
        ):
            return False
        compact_accession = accession.replace("-", "")
        compact_url = re.sub(r"[^a-z0-9]", "", source_url.lower())
        return bool(
            compact_accession in compact_url
            and f"/edgar/data/{int(issuer_cik)}/" in source_url.lower()
        )
    if source == "exchange_listing":
        return any(
            host == suffix or host.endswith(f".{suffix}")
            for suffix in _EXCHANGE_SOURCE_HOST_SUFFIXES
        )
    if source == "depositary_agreement":
        return any(
            host == suffix or host.endswith(f".{suffix}")
            for suffix in _DEPOSITARY_SOURCE_HOST_SUFFIXES
        )
    return False


def _validate_trace_input_provenance(
    trace: Mapping[str, Any],
    *,
    run_as_of_date: str,
    violations: list[FinancialIntegrityViolation],
    ticker: str | None,
    field_prefix: str,
    code_prefix: str,
) -> None:
    """Require one exact, point-in-time provenance record per trace input."""

    inputs = trace.get("inputs")
    if not isinstance(inputs, Mapping):
        return
    provenance = trace.get("input_provenance")
    if not isinstance(provenance, Mapping):
        _violation(
            violations,
            f"{code_prefix}_INPUT_PROVENANCE_MISSING",
            "Known metric trace requires provenance for every canonical input.",
            ticker=ticker,
            field_name=f"{field_prefix}.input_provenance",
            source_values={"inputs": dict(inputs), "input_provenance": provenance},
            expected_relationship="one complete provenance record per trace input",
            observed_relationship="input_provenance missing",
            terminal_status=NEEDS_DATA,
        )
        return

    def validate_components(
        record: Mapping[str, Any],
        *,
        input_field: str,
    ) -> None:
        components = record.get("components")
        if components is None:
            return
        if not isinstance(components, Mapping) or not components:
            _violation(
                violations,
                f"{code_prefix}_INPUT_PROVENANCE_MISSING",
                "Derived trace provenance requires complete component records.",
                ticker=ticker,
                field_name=f"{input_field}.components",
                source_values={"components": components},
                expected_relationship="one complete provenance record per derived component",
                observed_relationship="components missing or empty",
                terminal_status=NEEDS_DATA,
            )
            return

        for component_name, component in components.items():
            component_field = f"{input_field}.components.{component_name}"
            if not isinstance(component, Mapping):
                _violation(
                    violations,
                    f"{code_prefix}_INPUT_PROVENANCE_MISSING",
                    "Derived trace provenance requires complete component records.",
                    ticker=ticker,
                    field_name=component_field,
                    source_values={
                        "component_name": component_name,
                        "component_provenance": component,
                    },
                    expected_relationship=(
                        "record has value, unit, source, period_end, filed_date, "
                        "and source_reference"
                    ),
                    observed_relationship="component provenance record missing",
                    terminal_status=NEEDS_DATA,
                )
                continue

            required_fields = (
                "value",
                "unit",
                "source",
                "period_end",
                "filed_date",
                "source_reference",
            )
            missing_fields = [
                name
                for name in required_fields
                if name not in component
                or component.get(name) is None
                or (name != "value" and not str(component.get(name)).strip())
            ]
            if missing_fields:
                _violation(
                    violations,
                    f"{code_prefix}_INPUT_PROVENANCE_MISSING",
                    "Derived trace component provenance is incomplete.",
                    ticker=ticker,
                    field_name=component_field,
                    source_values={
                        "component_name": component_name,
                        "component_provenance": dict(component),
                        "missing_fields": missing_fields,
                    },
                    expected_relationship=required_fields,
                    observed_relationship={"missing_fields": missing_fields},
                    terminal_status=NEEDS_DATA,
                )
                continue

            if _finite_number(component.get("value")) is None:
                _violation(
                    violations,
                    f"{code_prefix}_INPUT_PROVENANCE_VALUE_INVALID",
                    "Derived trace component value must be finite.",
                    ticker=ticker,
                    field_name=f"{component_field}.value",
                    observed=component.get("value"),
                    expected="finite component value",
                )

            expected_unit = _CANONICAL_TRACE_INPUT_UNITS.get(str(component_name))
            observed_unit = str(component.get("unit") or "").strip()
            if expected_unit is not None and observed_unit != expected_unit:
                _violation(
                    violations,
                    f"{code_prefix}_INPUT_PROVENANCE_UNIT_INVALID",
                    "Derived trace component must carry the exact canonical unit.",
                    ticker=ticker,
                    field_name=f"{component_field}.unit",
                    source_values={
                        "component_name": component_name,
                        "unit": observed_unit,
                    },
                    expected_relationship=expected_unit,
                    observed_relationship=observed_unit,
                )

            component_period_end = str(component.get("period_end") or "").strip()[:10]
            component_filed_date = str(component.get("filed_date") or "").strip()[:10]
            if not _valid_iso_at_or_before(
                component_period_end, run_as_of_date
            ) or not _valid_iso_at_or_before(component_filed_date, run_as_of_date):
                _violation(
                    violations,
                    f"{code_prefix}_INPUT_PROVENANCE_ASOF_INVALID",
                    "Derived trace component must be visible on or before the run as-of date.",
                    ticker=ticker,
                    field_name=component_field,
                    source_values={
                        "period_end": component_period_end,
                        "filed_date": component_filed_date,
                        "run_as_of_date": run_as_of_date,
                    },
                    expected_relationship=(
                        f"period_end <= {run_as_of_date} and filed_date <= {run_as_of_date}"
                    ),
                    observed_relationship={
                        "period_end": component_period_end,
                        "filed_date": component_filed_date,
                    },
                )
                continue
            if date.fromisoformat(component_period_end) > date.fromisoformat(component_filed_date):
                _violation(
                    violations,
                    f"{code_prefix}_INPUT_PROVENANCE_CHRONOLOGY_INVALID",
                    "Derived trace component filing availability cannot precede its period end.",
                    ticker=ticker,
                    field_name=component_field,
                    source_values={
                        "period_end": component_period_end,
                        "filed_date": component_filed_date,
                    },
                    expected_relationship="period_end <= filed_date",
                    observed_relationship=(f"{component_period_end} > {component_filed_date}"),
                )

            validate_components(component, input_field=component_field)

    for input_name, input_value in inputs.items():
        record = provenance.get(input_name)
        input_field = f"{field_prefix}.input_provenance.{input_name}"
        if not isinstance(record, Mapping):
            _violation(
                violations,
                f"{code_prefix}_INPUT_PROVENANCE_MISSING",
                "Known metric trace requires provenance for every canonical input.",
                ticker=ticker,
                field_name=input_field,
                source_values={
                    "input_name": input_name,
                    "input_value": input_value,
                    "input_provenance": record,
                },
                expected_relationship=(
                    "record has value, unit, source, period_end, filed_date, and source_reference"
                ),
                observed_relationship="provenance record missing",
                terminal_status=NEEDS_DATA,
            )
            continue

        required_fields = (
            "value",
            "unit",
            "source",
            "period_end",
            "filed_date",
            "source_reference",
        )
        missing_fields = [
            name
            for name in required_fields
            if name not in record
            or record.get(name) is None
            or (name != "value" and not str(record.get(name)).strip())
        ]
        if missing_fields:
            _violation(
                violations,
                f"{code_prefix}_INPUT_PROVENANCE_MISSING",
                "Trace input provenance is incomplete.",
                ticker=ticker,
                field_name=input_field,
                source_values={
                    "input_name": input_name,
                    "input_value": input_value,
                    "input_provenance": dict(record),
                    "missing_fields": missing_fields,
                },
                expected_relationship=required_fields,
                observed_relationship={"missing_fields": missing_fields},
                terminal_status=NEEDS_DATA,
            )
            continue

        if not metric_values_reconcile(input_value, record.get("value")):
            _violation(
                violations,
                f"{code_prefix}_INPUT_PROVENANCE_VALUE_MISMATCH",
                "Trace input value must exactly reconcile to its provenance value.",
                ticker=ticker,
                field_name=f"{input_field}.value",
                source_values={
                    "trace_input_value": input_value,
                    "provenance_value": record.get("value"),
                },
                expected_relationship="trace input == provenance value",
                observed_relationship=(f"{input_value} != {record.get('value')}"),
            )

        expected_unit = _CANONICAL_TRACE_INPUT_UNITS.get(str(input_name))
        observed_unit = str(record.get("unit") or "").strip()
        if expected_unit is None or observed_unit != expected_unit:
            _violation(
                violations,
                f"{code_prefix}_INPUT_PROVENANCE_UNIT_INVALID",
                "Trace input provenance must carry the exact canonical unit.",
                ticker=ticker,
                field_name=f"{input_field}.unit",
                source_values={
                    "input_name": input_name,
                    "unit": observed_unit,
                },
                expected_relationship=expected_unit or "recognized canonical input",
                observed_relationship=observed_unit,
            )

        if str(input_name) == "shares_outstanding_mm":
            raw_source_value = record.get("raw_source_value")
            raw_source_unit = str(record.get("raw_source_unit") or "").strip()
            normalized_value = record.get("normalized_value")
            normalized_unit = str(record.get("normalized_unit") or "").strip()
            split_factor = record.get("split_adjustment_factor")
            if (
                _finite_number(raw_source_value, positive=True) is None
                or raw_source_unit not in {"shares", SHARES_UNIT_MILLIONS}
                or _finite_number(normalized_value, positive=True) is None
                or normalized_unit != SHARES_UNIT_MILLIONS
                or _finite_number(split_factor, positive=True) is None
            ):
                _violation(
                    violations,
                    f"{code_prefix}_SHARES_NORMALIZATION_PROVENANCE_MISSING",
                    "Shares trace must preserve raw source value/unit and normalized value/unit.",
                    ticker=ticker,
                    field_name=input_field,
                    source_values={
                        "raw_source_value": raw_source_value,
                        "raw_source_unit": raw_source_unit,
                        "normalized_value": normalized_value,
                        "normalized_unit": normalized_unit,
                        "split_adjustment_factor": split_factor,
                    },
                    expected_relationship=(
                        "raw source shares plus explicit unit normalize to shares_millions"
                    ),
                    observed_relationship="shares normalization provenance incomplete",
                    terminal_status=NEEDS_DATA,
                )
            else:
                raw_source_mm = (
                    float(raw_source_value) / 1_000_000.0
                    if raw_source_unit == "shares"
                    else float(raw_source_value)
                )
                expected_normalized = raw_source_mm * float(split_factor)
                if not metric_values_reconcile(
                    expected_normalized,
                    normalized_value,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ) or not metric_values_reconcile(
                    input_value,
                    normalized_value,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    _violation(
                        violations,
                        f"{code_prefix}_SHARES_NORMALIZATION_CONFLICT",
                        "Shares trace raw and normalized values do not reconcile.",
                        ticker=ticker,
                        field_name=input_field,
                        source_values={
                            "trace_input_value": input_value,
                            "raw_source_value": raw_source_value,
                            "raw_source_unit": raw_source_unit,
                            "normalized_value": normalized_value,
                            "normalized_unit": normalized_unit,
                            "split_adjustment_factor": split_factor,
                        },
                        expected_relationship=(
                            "normalized_value == raw source value converted to "
                            "shares_millions * split_adjustment_factor"
                        ),
                        observed_relationship=(f"{normalized_value} != {expected_normalized}"),
                    )

        period_end = str(record.get("period_end") or "").strip()[:10]
        filed_date = str(record.get("filed_date") or "").strip()[:10]
        if not _valid_iso_at_or_before(period_end, run_as_of_date) or not _valid_iso_at_or_before(
            filed_date, run_as_of_date
        ):
            _violation(
                violations,
                f"{code_prefix}_INPUT_PROVENANCE_ASOF_INVALID",
                "Trace input provenance must be visible on or before the run as-of date.",
                ticker=ticker,
                field_name=input_field,
                source_values={
                    "period_end": period_end,
                    "filed_date": filed_date,
                    "run_as_of_date": run_as_of_date,
                },
                expected_relationship=(
                    f"period_end <= {run_as_of_date} and filed_date <= {run_as_of_date}"
                ),
                observed_relationship={
                    "period_end": period_end,
                    "filed_date": filed_date,
                },
            )
            continue
        if date.fromisoformat(period_end) > date.fromisoformat(filed_date):
            _violation(
                violations,
                f"{code_prefix}_INPUT_PROVENANCE_CHRONOLOGY_INVALID",
                "Trace input filing availability cannot precede its period end.",
                ticker=ticker,
                field_name=input_field,
                source_values={
                    "period_end": period_end,
                    "filed_date": filed_date,
                },
                expected_relationship="period_end <= filed_date",
                observed_relationship=f"{period_end} > {filed_date}",
            )
        validate_components(record, input_field=input_field)


def _packet_quote_payload(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ticker": packet.get("ticker"),
        "price": packet.get("current_price"),
        "as_of_date": packet.get("current_price_as_of_date"),
        "currency": packet.get("current_price_currency"),
        "source": packet.get("current_price_source"),
        "source_url": packet.get("current_price_source_url"),
        "price_basis": packet.get("price_basis"),
        "raw_price": packet.get("raw_price"),
        "split_adjustment_factor": packet.get("split_adjustment_factor"),
        "split_effective_date": packet.get("split_effective_date"),
    }


def _validate_packet(
    packet: Mapping[str, Any],
    *,
    run_as_of_date: str,
    violations: list[FinancialIntegrityViolation],
    snapshot_ids: dict[str, str],
) -> None:
    ticker = str(packet.get("ticker") or "").strip().upper()
    if not ticker:
        _violation(
            violations,
            "TICKER_MISSING",
            "Packet ticker identity is required.",
            field_name="ticker",
            observed=packet.get("ticker"),
            terminal_status=NEEDS_DATA,
        )
        return

    valuation = packet.get("valuation")
    canonical_metric_status = (
        str(valuation.get("canonical_metric_status") or "").strip().upper()
        if isinstance(valuation, Mapping)
        else ""
    )
    if canonical_metric_status in {NEEDS_DATA, INVALID_FINANCIAL_INPUT}:
        _violation(
            violations,
            "CANONICAL_METRIC_INPUTS_UNAVAILABLE",
            "Canonical valuation inputs must be complete before a decision boundary.",
            ticker=ticker,
            field_name="valuation.canonical_metric_status",
            source_values={
                "status": canonical_metric_status,
                "reason": valuation.get("canonical_metric_reason"),
            },
            expected_relationship="canonical_metric_status == OK",
            observed_relationship=canonical_metric_status,
            terminal_status=(
                NEEDS_DATA if canonical_metric_status == NEEDS_DATA else INVALID_FINANCIAL_INPUT
            ),
        )

    market_cap = packet.get("market_cap_mm")
    if market_cap is None:
        _violation(
            violations,
            "MARKET_CAP_MISSING",
            "Market cap is required.",
            ticker=ticker,
            field_name="market_cap_mm",
            observed=market_cap,
            expected="finite positive USD millions",
            terminal_status=NEEDS_DATA,
        )
    elif _finite_number(market_cap, positive=True) is None:
        _violation(
            violations,
            "MARKET_CAP_INVALID",
            "Market cap must be finite and positive.",
            ticker=ticker,
            field_name="market_cap_mm",
            observed=market_cap,
        )
    market_cap_unit = packet.get("market_cap_unit")
    if market_cap_unit is None or not str(market_cap_unit).strip():
        _violation(
            violations,
            "MARKET_CAP_UNIT_MISSING",
            "Market cap unit provenance is required.",
            ticker=ticker,
            field_name="market_cap_unit",
            observed=market_cap_unit,
            expected=MARKET_CAP_UNIT_USD_MILLIONS,
            terminal_status=NEEDS_DATA,
        )
    elif market_cap_unit != MARKET_CAP_UNIT_USD_MILLIONS:
        _violation(
            violations,
            "MARKET_CAP_UNIT_INVALID",
            "Market cap requires the literal USD_millions unit.",
            ticker=ticker,
            field_name="market_cap_unit",
            observed=packet.get("market_cap_unit"),
            expected=MARKET_CAP_UNIT_USD_MILLIONS,
        )
    market_cap_source = str(packet.get("market_cap_source") or "").strip()
    if not market_cap_source:
        _violation(
            violations,
            "MARKET_CAP_SOURCE_MISSING",
            "Market-cap source provenance is required.",
            ticker=ticker,
            field_name="market_cap_source",
            observed=packet.get("market_cap_source"),
            expected="non-empty market-cap source",
            terminal_status=NEEDS_DATA,
        )
    market_cap_method = str(packet.get("market_cap_method") or "").strip()
    if not market_cap_method:
        _violation(
            violations,
            "MARKET_CAP_METHOD_MISSING",
            "Market-cap derivation method is required.",
            ticker=ticker,
            field_name="market_cap_method",
            observed=packet.get("market_cap_method"),
            expected="explicit direct or derived market-cap method",
            terminal_status=NEEDS_DATA,
        )
    market_cap_as_of = packet.get("market_cap_effective_as_of_date")
    if not str(market_cap_as_of or "").strip():
        _violation(
            violations,
            "MARKET_CAP_ASOF_MISSING",
            "Market-cap effective as-of date is required.",
            ticker=ticker,
            field_name="market_cap_effective_as_of_date",
            observed=market_cap_as_of,
            expected=f"valid ISO date <= {run_as_of_date}",
            terminal_status=NEEDS_DATA,
        )
    elif not _valid_iso_at_or_before(market_cap_as_of, run_as_of_date):
        _violation(
            violations,
            "MARKET_CAP_ASOF_INVALID",
            "Market-cap effective as-of must not be future dated.",
            ticker=ticker,
            field_name="market_cap_effective_as_of_date",
            observed=market_cap_as_of,
            expected=f"<= {run_as_of_date}",
        )

    current_price = packet.get("current_price")
    if current_price is None:
        _violation(
            violations,
            "QUOTE_PRICE_MISSING",
            "Current quote is required.",
            ticker=ticker,
            field_name="current_price",
            observed=current_price,
            expected="finite positive USD per share",
            terminal_status=NEEDS_DATA,
        )
    elif _finite_number(current_price, positive=True) is None:
        _violation(
            violations,
            "QUOTE_PRICE_INVALID",
            "Current quote must be finite and positive.",
            ticker=ticker,
            field_name="current_price",
            observed=current_price,
        )
    current_price_currency = str(packet.get("current_price_currency") or "").upper()
    if not current_price_currency:
        _violation(
            violations,
            "QUOTE_CURRENCY_MISSING",
            "Current quote currency provenance is required.",
            ticker=ticker,
            field_name="current_price_currency",
            observed=packet.get("current_price_currency"),
            expected="USD",
            terminal_status=NEEDS_DATA,
        )
    elif current_price_currency != "USD":
        _violation(
            violations,
            "QUOTE_CURRENCY_INVALID",
            "Current quote must be explicitly USD.",
            ticker=ticker,
            field_name="current_price_currency",
            observed=packet.get("current_price_currency"),
            expected="USD",
        )
    current_price_unit = packet.get("current_price_unit")
    if current_price_unit is None or not str(current_price_unit).strip():
        _violation(
            violations,
            "QUOTE_UNIT_MISSING",
            "Current quote unit provenance is required.",
            ticker=ticker,
            field_name="current_price_unit",
            observed=current_price_unit,
            expected=PRICE_UNIT_USD_PER_SHARE,
            terminal_status=NEEDS_DATA,
        )
    elif current_price_unit != PRICE_UNIT_USD_PER_SHARE:
        _violation(
            violations,
            "QUOTE_UNIT_INVALID",
            "Current quote requires the literal USD_per_share unit.",
            ticker=ticker,
            field_name="current_price_unit",
            observed=packet.get("current_price_unit"),
            expected=PRICE_UNIT_USD_PER_SHARE,
        )
    current_price_as_of_date = packet.get("current_price_as_of_date")
    if not str(current_price_as_of_date or "").strip():
        _violation(
            violations,
            "QUOTE_ASOF_MISSING",
            "Quote as-of provenance is required.",
            ticker=ticker,
            field_name="current_price_as_of_date",
            observed=current_price_as_of_date,
            expected=f"valid ISO date <= {run_as_of_date}",
            terminal_status=NEEDS_DATA,
        )
    elif not _valid_iso_at_or_before(current_price_as_of_date, run_as_of_date):
        _violation(
            violations,
            "QUOTE_ASOF_INVALID",
            "Quote as-of must be a valid date at or before the run date.",
            ticker=ticker,
            field_name="current_price_as_of_date",
            observed=packet.get("current_price_as_of_date"),
            expected=f"<= {run_as_of_date}",
        )
    current_price_source = str(packet.get("current_price_source") or "").strip()
    if not current_price_source:
        _violation(
            violations,
            "QUOTE_SOURCE_MISSING",
            "Quote source provenance is required.",
            ticker=ticker,
            field_name="current_price_source",
            observed=packet.get("current_price_source"),
            expected="non-empty quote source",
            terminal_status=NEEDS_DATA,
        )
    price_basis = str(packet.get("price_basis") or "").upper()
    if not price_basis:
        _violation(
            violations,
            "QUOTE_BASIS_MISSING",
            "Quote split-basis provenance is required.",
            ticker=ticker,
            field_name="price_basis",
            observed=packet.get("price_basis"),
            expected=sorted(_VALID_PRICE_BASES),
            terminal_status=NEEDS_DATA,
        )
    elif price_basis not in _VALID_PRICE_BASES:
        _violation(
            violations,
            "QUOTE_BASIS_INVALID",
            "Quote split basis must be explicit.",
            ticker=ticker,
            field_name="price_basis",
            observed=packet.get("price_basis"),
            expected=sorted(_VALID_PRICE_BASES),
        )
    raw_price = packet.get("raw_price")
    if raw_price is None:
        _violation(
            violations,
            "RAW_QUOTE_PRICE_MISSING",
            "Raw quote value is required to prove the declared split basis.",
            ticker=ticker,
            field_name="raw_price",
            observed=raw_price,
            expected="finite positive raw quote",
            terminal_status=NEEDS_DATA,
        )
    elif _finite_number(raw_price, positive=True) is None:
        _violation(
            violations,
            "RAW_QUOTE_PRICE_INVALID",
            "Raw quote value must be finite and positive.",
            ticker=ticker,
            field_name="raw_price",
            observed=raw_price,
            expected="finite positive raw quote",
        )

    shares = packet.get("shares_outstanding_mm")
    if shares is None:
        _violation(
            violations,
            "SHARES_MISSING",
            "Shares outstanding are required to prove the cap/share quote relationship.",
            ticker=ticker,
            field_name="shares_outstanding_mm",
            observed=shares,
            expected="finite positive shares in millions",
            terminal_status=NEEDS_DATA,
        )
    elif _finite_number(shares, positive=True) is None:
        _violation(
            violations,
            "SHARES_INVALID",
            "Shares outstanding must be finite and positive.",
            ticker=ticker,
            field_name="shares_outstanding_mm",
            observed=shares,
        )
    shares_unit = packet.get("shares_unit")
    if shares_unit is None or not str(shares_unit).strip():
        _violation(
            violations,
            "SHARES_UNIT_MISSING",
            "Shares unit provenance is required.",
            ticker=ticker,
            field_name="shares_unit",
            observed=shares_unit,
            expected=SHARES_UNIT_MILLIONS,
            terminal_status=NEEDS_DATA,
        )
    elif shares_unit != SHARES_UNIT_MILLIONS:
        _violation(
            violations,
            "SHARES_UNIT_INVALID",
            "Shares outstanding requires the literal shares_millions unit.",
            ticker=ticker,
            field_name="shares_unit",
            observed=packet.get("shares_unit"),
            expected=SHARES_UNIT_MILLIONS,
        )
    shares_basis = str(packet.get("shares_basis") or "").upper()
    if not shares_basis:
        _violation(
            violations,
            "SHARES_BASIS_MISSING",
            "Shares split-basis provenance is required.",
            ticker=ticker,
            field_name="shares_basis",
            observed=packet.get("shares_basis"),
            expected=sorted(_VALID_SHARES_BASES),
            terminal_status=NEEDS_DATA,
        )
    elif shares_basis not in _VALID_SHARES_BASES:
        _violation(
            violations,
            "SHARES_BASIS_INVALID",
            "Shares split basis must be explicit.",
            ticker=ticker,
            field_name="shares_basis",
            observed=packet.get("shares_basis"),
            expected=sorted(_VALID_SHARES_BASES),
        )
    raw_shares = packet.get("raw_shares_outstanding_mm")
    raw_shares_number = _finite_number(raw_shares, positive=True)
    if shares_basis == PRICE_BASIS_SPLIT_ADJUSTED and raw_shares is None:
        _violation(
            violations,
            "RAW_SHARES_MISSING",
            "Split-adjusted shares require the source share count before normalization.",
            ticker=ticker,
            field_name="raw_shares_outstanding_mm",
            observed=raw_shares,
            expected="finite positive source shares in millions",
            terminal_status=NEEDS_DATA,
        )
    elif raw_shares is not None and raw_shares_number is None:
        _violation(
            violations,
            "RAW_SHARES_INVALID",
            "Raw shares must be finite and positive when supplied.",
            ticker=ticker,
            field_name="raw_shares_outstanding_mm",
            observed=raw_shares,
            expected="finite positive source shares in millions",
        )
    raw_shares_source_value = packet.get("raw_shares_source_value")
    raw_shares_source_number = _finite_number(raw_shares_source_value, positive=True)
    raw_shares_source_unit = str(packet.get("raw_shares_source_unit") or "").strip()
    if raw_shares_source_number is None or raw_shares_source_unit not in {
        "shares",
        SHARES_UNIT_MILLIONS,
    }:
        _violation(
            violations,
            "RAW_SHARES_SOURCE_PROVENANCE_MISSING",
            "Shares must preserve the exact raw source value and its literal unit.",
            ticker=ticker,
            field_name="raw_shares_source_value/raw_shares_source_unit",
            source_values={
                "raw_shares_source_value": raw_shares_source_value,
                "raw_shares_source_unit": packet.get("raw_shares_source_unit"),
            },
            expected_relationship="positive raw value with unit shares or shares_millions",
            observed_relationship=(
                f"value={raw_shares_source_value}; unit={raw_shares_source_unit or 'MISSING'}"
            ),
            terminal_status=NEEDS_DATA,
        )
    elif raw_shares_number is not None:
        normalized_raw_source = (
            raw_shares_source_number / 1_000_000.0
            if raw_shares_source_unit == "shares"
            else raw_shares_source_number
        )
        if not math.isclose(
            normalized_raw_source,
            raw_shares_number,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            _violation(
                violations,
                "RAW_SHARES_SOURCE_NORMALIZATION_CONFLICT",
                "Raw source shares do not reconcile to raw shares in millions.",
                ticker=ticker,
                field_name="raw_shares_source_value/raw_shares_outstanding_mm",
                source_values={
                    "raw_shares_source_value": raw_shares_source_number,
                    "raw_shares_source_unit": raw_shares_source_unit,
                    "raw_shares_outstanding_mm": raw_shares_number,
                },
                expected_relationship=(
                    "raw source value converted by explicit unit == raw_shares_outstanding_mm"
                ),
                observed_relationship=f"{normalized_raw_source} != {raw_shares_number}",
            )
    shares_as_of = packet.get("shares_as_of_date")
    if shares is not None:
        if not str(shares_as_of or "").strip():
            _violation(
                violations,
                "SHARES_ASOF_MISSING",
                "Shares outstanding as-of provenance is required.",
                ticker=ticker,
                field_name="shares_as_of_date",
                observed=shares_as_of,
                expected=f"valid ISO date <= {run_as_of_date}",
                terminal_status=NEEDS_DATA,
            )
        elif not _valid_iso_at_or_before(shares_as_of, run_as_of_date):
            _violation(
                violations,
                "SHARES_ASOF_INVALID",
                "Shares outstanding as-of must not be future dated.",
                ticker=ticker,
                field_name="shares_as_of_date",
                observed=shares_as_of,
                expected=f"<= {run_as_of_date}",
            )
        shares_source = str(packet.get("shares_source") or "").strip()
        if not shares_source:
            _violation(
                violations,
                "SHARES_SOURCE_MISSING",
                "Shares outstanding source provenance is required.",
                ticker=ticker,
                field_name="shares_source",
                observed=packet.get("shares_source"),
                expected="non-empty shares source",
                terminal_status=NEEDS_DATA,
            )
        shares_source_reference = str(packet.get("shares_source_url") or "").strip()
        if not shares_source_reference:
            _violation(
                violations,
                "SHARES_SOURCE_REFERENCE_MISSING",
                "Shares outstanding requires an exact source reference.",
                ticker=ticker,
                field_name="shares_source_url",
                observed=packet.get("shares_source_url"),
                expected="non-empty shares source reference",
                terminal_status=NEEDS_DATA,
            )
        shares_filed_date = packet.get("shares_filed_date")
        if not str(shares_filed_date or "").strip():
            _violation(
                violations,
                "SHARES_FILED_DATE_MISSING",
                "Shares outstanding requires the filing date visible at the run boundary.",
                ticker=ticker,
                field_name="shares_filed_date",
                observed=shares_filed_date,
                expected=f"valid ISO date <= {run_as_of_date}",
                terminal_status=NEEDS_DATA,
            )
        elif not _valid_iso_at_or_before(shares_filed_date, run_as_of_date):
            _violation(
                violations,
                "SHARES_FILED_DATE_INVALID",
                "Shares filing date must not be future dated.",
                ticker=ticker,
                field_name="shares_filed_date",
                observed=shares_filed_date,
                expected=f"<= {run_as_of_date}",
            )
        elif str(shares_as_of or "").strip() and not _valid_iso_at_or_before(
            shares_as_of,
            str(shares_filed_date)[:10],
        ):
            _violation(
                violations,
                "SHARES_PERIOD_AFTER_FILED_DATE",
                "Shares period end cannot be later than its filing date.",
                ticker=ticker,
                field_name="shares_as_of_date/shares_filed_date",
                source_values={
                    "shares_as_of_date": shares_as_of,
                    "shares_filed_date": shares_filed_date,
                },
                expected_relationship="shares_as_of_date <= shares_filed_date",
                observed_relationship=f"{shares_as_of} > {shares_filed_date}",
            )

    split_factor = packet.get("split_adjustment_factor")
    if split_factor is None:
        _violation(
            violations,
            "SPLIT_FACTOR_MISSING",
            "Split adjustment-factor provenance is required.",
            ticker=ticker,
            field_name="split_adjustment_factor",
            observed=split_factor,
            expected="finite positive factor",
            terminal_status=NEEDS_DATA,
        )
    elif _finite_number(split_factor, positive=True) is None:
        _violation(
            violations,
            "SPLIT_FACTOR_INVALID",
            "Split adjustment factor must be finite and positive.",
            ticker=ticker,
            field_name="split_adjustment_factor",
            observed=split_factor,
        )
    split_factor_number = _finite_number(split_factor, positive=True)
    split_declared = bool(
        price_basis == PRICE_BASIS_SPLIT_ADJUSTED
        or shares_basis == PRICE_BASIS_SPLIT_ADJUSTED
        or (
            split_factor_number is not None
            and not math.isclose(split_factor_number, 1.0, rel_tol=0.0, abs_tol=1e-12)
        )
    )
    split_effective_date = packet.get("split_effective_date")
    if split_declared and not str(split_effective_date or "").strip():
        _violation(
            violations,
            "SPLIT_EFFECTIVE_DATE_MISSING",
            "A split-adjusted basis or non-unit split factor requires an effective date.",
            ticker=ticker,
            field_name="split_effective_date",
            expected=f"valid ISO date <= {run_as_of_date}",
            terminal_status=NEEDS_DATA,
        )
    elif str(split_effective_date or "").strip() and not _valid_iso_at_or_before(
        split_effective_date, run_as_of_date
    ):
        _violation(
            violations,
            "SPLIT_EFFECTIVE_DATE_INVALID",
            "Split effective date must be a valid non-future date.",
            ticker=ticker,
            field_name="split_effective_date",
            observed=split_effective_date,
            expected=f"valid ISO date <= {run_as_of_date}",
        )

    split_lineage_proof = packet.get("split_lineage_proof")
    if price_basis == PRICE_BASIS_SPLIT_ADJUSTED and shares_basis == PRICE_BASIS_SPLIT_ADJUSTED:
        shares_as_of_text = str(shares_as_of or "").strip()[:10]
        split_effective_text = str(split_effective_date or "").strip()[:10]
        quote_as_of_text = str(current_price_as_of_date or "").strip()[:10]
        try:
            shares_as_of_value = date.fromisoformat(shares_as_of_text)
            split_effective_value = date.fromisoformat(split_effective_text)
            quote_as_of_value = date.fromisoformat(quote_as_of_text)
        except ValueError:
            shares_as_of_value = split_effective_value = quote_as_of_value = None
        if (
            shares_as_of_value is not None
            and split_effective_value is not None
            and quote_as_of_value is not None
            and not (shares_as_of_value < split_effective_value <= quote_as_of_value)
        ):
            _violation(
                violations,
                "SPLIT_CHRONOLOGY_INVALID",
                (
                    "Split-adjusted shares require a source share count from "
                    "before the split and a quote from on or after the split."
                ),
                ticker=ticker,
                field_name="split_effective_date",
                source_values={
                    "shares_as_of_date": shares_as_of_text,
                    "split_effective_date": split_effective_text,
                    "current_price_as_of_date": quote_as_of_text,
                },
                expected_relationship=(
                    "shares_as_of_date < split_effective_date <= current_price_as_of_date"
                ),
                observed_relationship=(
                    f"{shares_as_of_text} < {split_effective_text} <= {quote_as_of_text}"
                ),
            )
        if not isinstance(split_lineage_proof, Mapping):
            _violation(
                violations,
                "SPLIT_EVENT_PROOF_MISSING",
                "Split-adjusted operands require an exact split-event source.",
                ticker=ticker,
                field_name="split_lineage_proof",
                source_values={
                    "split_adjustment_factor": split_factor,
                    "split_effective_date": split_effective_date,
                },
                expected_relationship=(
                    "proof has factor, effective_date, filed_date, source, and source_reference"
                ),
                observed_relationship=split_lineage_proof,
                terminal_status=NEEDS_DATA,
            )
        else:
            proof_factor = _finite_number(split_lineage_proof.get("factor"), positive=True)
            proof_effective_date = str(split_lineage_proof.get("effective_date") or "").strip()[:10]
            proof_filed_date = str(split_lineage_proof.get("filed_date") or "").strip()[:10]
            proof_source = str(split_lineage_proof.get("source") or "").strip()
            proof_reference = str(split_lineage_proof.get("source_reference") or "").strip()
            proof_reference_authoritative = authoritative_split_proof_reference(
                split_lineage_proof,
                expected_ticker=ticker or "",
                expected_issuer_cik=_packet_issuer_cik(packet),
                expected_as_of_date=run_as_of_date,
            )
            proof_complete = bool(
                proof_factor is not None
                and proof_effective_date
                and proof_filed_date
                and proof_source
                and proof_reference
                and proof_reference_authoritative
            )
            proof_matches = bool(
                proof_complete
                and split_factor_number is not None
                and metric_values_reconcile(proof_factor, split_factor_number)
                and proof_effective_date == str(split_effective_date or "")[:10]
                and _valid_iso_at_or_before(proof_filed_date, run_as_of_date)
            )
            if not proof_complete:
                _violation(
                    violations,
                    "SPLIT_EVENT_PROOF_MISSING",
                    "Split-event provenance is incomplete.",
                    ticker=ticker,
                    field_name="split_lineage_proof",
                    source_values={"split_lineage_proof": split_lineage_proof},
                    expected_relationship=(
                        "factor, effective_date, filed_date, source, and "
                        "authoritative source_reference are complete"
                    ),
                    observed_relationship=split_lineage_proof,
                    terminal_status=NEEDS_DATA,
                )
            elif not proof_matches:
                _violation(
                    violations,
                    "SPLIT_EVENT_PROOF_INVALID",
                    "Split-event proof must match the normalized operands and be visible as-of.",
                    ticker=ticker,
                    field_name="split_lineage_proof",
                    source_values={
                        "split_lineage_proof": split_lineage_proof,
                        "split_adjustment_factor": split_factor_number,
                        "split_effective_date": split_effective_date,
                        "run_as_of_date": run_as_of_date,
                    },
                    expected_relationship="proof matches factor/effective date and is filed as-of",
                    observed_relationship=split_lineage_proof,
                )
    elif price_basis == PRICE_BASIS_UNADJUSTED and shares_basis == SHARES_BASIS_UNADJUSTED:
        if not isinstance(split_lineage_proof, Mapping):
            _violation(
                violations,
                "NO_INTERVENING_SPLIT_PROOF_MISSING",
                "Dated unadjusted shares require proof that no split intervened before the quote.",
                ticker=ticker,
                field_name="split_lineage_proof",
                source_values={
                    "shares_as_of_date": shares_as_of,
                    "current_price_as_of_date": current_price_as_of_date,
                },
                expected_relationship="complete no-intervening-split proof",
                observed_relationship=split_lineage_proof,
                terminal_status=NEEDS_DATA,
            )
        else:
            proof_start = str(split_lineage_proof.get("period_start") or "").strip()[:10]
            proof_end = str(split_lineage_proof.get("period_end") or "").strip()[:10]
            verified_as_of = str(split_lineage_proof.get("verified_as_of") or "").strip()[:10]
            proof_valid = bool(
                str(split_lineage_proof.get("status") or "").strip().upper() == "PASS"
                and str(split_lineage_proof.get("source") or "").strip()
                and authoritative_split_proof_reference(
                    split_lineage_proof,
                    expected_ticker=ticker or "",
                    expected_issuer_cik=_packet_issuer_cik(packet),
                    expected_as_of_date=run_as_of_date,
                )
                and _valid_iso_at_or_before(proof_start, run_as_of_date)
                and _valid_iso_at_or_before(proof_end, run_as_of_date)
                and _valid_iso_at_or_before(verified_as_of, run_as_of_date)
                and proof_start <= str(shares_as_of)[:10]
                and proof_end >= str(current_price_as_of_date)[:10]
                and proof_end <= verified_as_of
            )
            if not proof_valid:
                _violation(
                    violations,
                    "NO_INTERVENING_SPLIT_PROOF_INVALID",
                    "No-intervening-split proof does not cover the share-to-quote interval.",
                    ticker=ticker,
                    field_name="split_lineage_proof",
                    source_values={
                        "split_lineage_proof": split_lineage_proof,
                        "shares_as_of_date": shares_as_of,
                        "current_price_as_of_date": current_price_as_of_date,
                    },
                    expected_relationship="proof covers shares date through quote date",
                    observed_relationship=split_lineage_proof,
                )

    if (price_basis == PRICE_BASIS_SPLIT_ADJUSTED) != (shares_basis == PRICE_BASIS_SPLIT_ADJUSTED):
        _violation(
            violations,
            "SPLIT_BASIS_MISMATCH",
            "A split-adjusted price cannot be multiplied by issuer-reported/unadjusted shares.",
            ticker=ticker,
            field_name="price_basis/shares_basis",
            source_values={
                "price_basis": packet.get("price_basis"),
                "shares_basis": packet.get("shares_basis"),
                "split_adjustment_factor": split_factor,
                "split_effective_date": packet.get("split_effective_date"),
            },
            expected_relationship="price and shares use the same declared split basis",
            observed_relationship=(
                f"price_basis={price_basis}; shares_basis={shares_basis or 'MISSING'}"
            ),
        )
    price_number_for_basis = _finite_number(current_price, positive=True)
    raw_price_number = _finite_number(raw_price, positive=True)
    if (
        price_basis == PRICE_BASIS_UNADJUSTED
        and price_number_for_basis is not None
        and raw_price_number is not None
        and not metric_values_reconcile(price_number_for_basis, raw_price_number)
    ):
        _violation(
            violations,
            "UNADJUSTED_QUOTE_RAW_PRICE_MISMATCH",
            "An unadjusted quote must equal its raw quote value.",
            ticker=ticker,
            field_name="raw_price",
            source_values={
                "current_price": price_number_for_basis,
                "raw_price": raw_price_number,
            },
            expected_relationship="current_price == raw_price for UNADJUSTED basis",
            observed_relationship=(f"{price_number_for_basis} != {raw_price_number}"),
        )
    if (
        price_basis == PRICE_BASIS_SPLIT_ADJUSTED
        and price_number_for_basis is not None
        and raw_price_number is not None
        and split_factor_number is not None
        and not metric_values_reconcile(
            raw_price_number / price_number_for_basis, split_factor_number
        )
    ):
        _violation(
            violations,
            "SPLIT_FACTOR_RECONCILIATION_FAILED",
            "Split factor must reconcile raw price to split-adjusted price.",
            ticker=ticker,
            field_name="split_adjustment_factor",
            source_values={
                "current_price": price_number_for_basis,
                "raw_price": raw_price_number,
                "split_adjustment_factor": split_factor_number,
            },
            expected_relationship="raw_price / current_price == split_adjustment_factor",
            observed_relationship=(
                f"{raw_price_number / price_number_for_basis} != {split_factor_number}"
            ),
        )
    shares_number_for_basis = _finite_number(shares, positive=True)
    if (
        shares_basis == SHARES_BASIS_UNADJUSTED
        and shares_number_for_basis is not None
        and raw_shares_number is not None
        and not metric_values_reconcile(shares_number_for_basis, raw_shares_number)
    ):
        _violation(
            violations,
            "UNADJUSTED_SHARES_RAW_VALUE_MISMATCH",
            "Unadjusted shares must equal their source share value.",
            ticker=ticker,
            field_name="raw_shares_outstanding_mm",
            source_values={
                "shares_outstanding_mm": shares_number_for_basis,
                "raw_shares_outstanding_mm": raw_shares_number,
            },
            expected_relationship=(
                "shares_outstanding_mm == raw_shares_outstanding_mm for UNADJUSTED basis"
            ),
            observed_relationship=(f"{shares_number_for_basis} != {raw_shares_number}"),
        )
    if (
        shares_basis == PRICE_BASIS_SPLIT_ADJUSTED
        and shares_number_for_basis is not None
        and raw_shares_number is not None
        and split_factor_number is not None
        and not metric_values_reconcile(
            shares_number_for_basis / raw_shares_number,
            split_factor_number,
        )
    ):
        _violation(
            violations,
            "SHARES_SPLIT_FACTOR_RECONCILIATION_FAILED",
            "Split factor must reconcile source shares to normalized shares.",
            ticker=ticker,
            field_name="raw_shares_outstanding_mm",
            source_values={
                "shares_outstanding_mm": shares_number_for_basis,
                "raw_shares_outstanding_mm": raw_shares_number,
                "split_adjustment_factor": split_factor_number,
            },
            expected_relationship=(
                "shares_outstanding_mm / raw_shares_outstanding_mm == split_adjustment_factor"
            ),
            observed_relationship=(
                f"{shares_number_for_basis / raw_shares_number} != {split_factor_number}"
            ),
        )

    quote_snapshot_id = str(packet.get("quote_snapshot_id") or "").strip().lower()
    expected_snapshot_id = stable_quote_hash(_packet_quote_payload(packet))
    quote_payload_complete = bool(
        _finite_number(current_price, positive=True) is not None
        and current_price_currency
        and current_price_unit == PRICE_UNIT_USD_PER_SHARE
        and str(current_price_as_of_date or "").strip()
        and current_price_source
        and price_basis in _VALID_PRICE_BASES
        and _finite_number(split_factor, positive=True) is not None
    )
    if not quote_snapshot_id:
        _violation(
            violations,
            "QUOTE_SNAPSHOT_ID_MISSING",
            "Canonical quote snapshot identity is required.",
            ticker=ticker,
            field_name="quote_snapshot_id",
            observed=None,
            expected=expected_snapshot_id,
            terminal_status=NEEDS_DATA,
        )
    elif quote_payload_complete and quote_snapshot_id != expected_snapshot_id:
        _violation(
            violations,
            "QUOTE_SNAPSHOT_ID_MISMATCH",
            "Quote snapshot ID does not match canonical quote fields.",
            ticker=ticker,
            field_name="quote_snapshot_id",
            observed=quote_snapshot_id or None,
            expected=expected_snapshot_id,
        )
    elif quote_payload_complete:
        snapshot_ids[ticker] = quote_snapshot_id

    cap_stage_snapshot_id = str(packet.get("cap_stage_quote_snapshot_id") or "").strip().lower()
    cap_stage_fields = {
        "price": packet.get("cap_stage_price"),
        "as_of_date": packet.get("cap_stage_price_as_of_date"),
        "currency": packet.get("cap_stage_price_currency"),
        "source": packet.get("cap_stage_price_source"),
        "source_url": packet.get("cap_stage_price_source_url"),
        "quote_snapshot_id": cap_stage_snapshot_id or None,
    }
    current_quote_fields = {
        "price": packet.get("current_price"),
        "as_of_date": packet.get("current_price_as_of_date"),
        "currency": packet.get("current_price_currency"),
        "source": packet.get("current_price_source"),
        "source_url": packet.get("current_price_source_url"),
        "quote_snapshot_id": quote_snapshot_id or None,
    }
    if cap_stage_fields["price"] is None or not cap_stage_snapshot_id:
        _violation(
            violations,
            "CAP_STAGE_QUOTE_PROVENANCE_MISSING",
            "Cap resolution must bind the same immutable quote used downstream.",
            ticker=ticker,
            field_name="cap_stage_quote_snapshot_id",
            source_values=cap_stage_fields,
            expected_relationship="cap stage carries a complete immutable quote snapshot",
            observed_relationship=cap_stage_fields,
            terminal_status=NEEDS_DATA,
        )
    elif quote_payload_complete:
        price_matches = metric_values_reconcile(
            cap_stage_fields["price"], current_quote_fields["price"]
        )
        metadata_matches = all(
            str(cap_stage_fields[key] or "").strip() == str(current_quote_fields[key] or "").strip()
            for key in ("as_of_date", "currency", "source", "source_url", "quote_snapshot_id")
        )
        if not price_matches or not metadata_matches:
            _violation(
                violations,
                "QUOTE_SNAPSHOT_MISMATCH",
                "Cap-stage and downstream quote snapshots differ.",
                ticker=ticker,
                field_name="cap_stage_quote/current_quote",
                source_values={
                    "cap_stage_quote": cap_stage_fields,
                    "current_quote": current_quote_fields,
                },
                expected_relationship="cap_stage_quote == current_quote",
                observed_relationship=(
                    f"price_match={price_matches}; metadata_match={metadata_matches}"
                ),
            )

    market_cap_number = _finite_number(market_cap, positive=True)
    price_number = _finite_number(current_price, positive=True)
    shares_number = _finite_number(shares, positive=True)
    cap_method = str(packet.get("market_cap_method") or "").strip()
    issuer_share_arithmetic = cap_method == "price_times_shares_divided_by_issuer_quote_ratio"
    quote_ratio = _finite_number(packet.get("issuer_quote_ratio"), positive=True)
    if issuer_share_arithmetic:
        security_role = str(packet.get("security_role") or "").strip().upper()
        issuer_primary_ticker = str(packet.get("issuer_primary_ticker") or "").strip().upper()
        identity_source = str(packet.get("identity_source") or "").strip()
        identity_source_url = str(packet.get("identity_source_url") or "").strip()
        identity_as_of_date = str(packet.get("identity_as_of_date") or "").strip()[:10]
        identity_confidence = str(packet.get("identity_confidence") or "").strip().upper()
        identity_complete = bool(
            security_role
            and issuer_primary_ticker
            and isinstance(packet.get("is_secondary_class"), bool)
            and isinstance(packet.get("is_adr"), bool)
            and identity_source
            and identity_source.lower() != "v1_legacy_unverified"
            and _valid_authoritative_reference(
                identity_source_url,
                host_suffixes=_IDENTITY_SOURCE_HOST_SUFFIXES,
            )
            and _valid_iso_at_or_before(identity_as_of_date, run_as_of_date)
            and identity_confidence in {"HIGH", "MEDIUM"}
            and quote_ratio is not None
        )
        if not identity_complete:
            _violation(
                violations,
                "CAP_SECURITY_IDENTITY_UNVERIFIED",
                "Issuer-share market-cap arithmetic requires dated authoritative quote identity.",
                ticker=ticker,
                field_name="security_role",
                source_values={
                    "issuer_primary_ticker": issuer_primary_ticker or None,
                    "security_role": security_role or None,
                    "is_secondary_class": packet.get("is_secondary_class"),
                    "is_adr": packet.get("is_adr"),
                    "issuer_quote_ratio": packet.get("issuer_quote_ratio"),
                    "identity_source": identity_source or None,
                    "identity_source_url": identity_source_url or None,
                    "identity_as_of_date": identity_as_of_date or None,
                    "identity_confidence": identity_confidence or None,
                },
                expected_relationship=(
                    "dated authoritative PRIMARY, ADR, or secondary-class quote identity"
                ),
                observed_relationship="identity proof is incomplete or unverified",
                terminal_status=NEEDS_DATA,
            )
        elif security_role == "PRIMARY":
            primary_conflict = bool(
                issuer_primary_ticker != ticker
                or packet.get("is_secondary_class") is not False
                or packet.get("is_adr") is not False
                or quote_ratio is None
                or not metric_values_reconcile(quote_ratio, 1.0)
            )
            if primary_conflict:
                _violation(
                    violations,
                    "CAP_SECURITY_IDENTITY_CONFLICT",
                    "Primary-quote identity conflicts with the cap arithmetic operands.",
                    ticker=ticker,
                    field_name="issuer_primary_ticker",
                    source_values={
                        "ticker": ticker,
                        "issuer_primary_ticker": issuer_primary_ticker,
                        "security_role": security_role,
                        "is_secondary_class": packet.get("is_secondary_class"),
                        "is_adr": packet.get("is_adr"),
                        "issuer_quote_ratio": quote_ratio,
                    },
                    expected_relationship=(
                        "PRIMARY ticker == issuer_primary_ticker with ratio 1 and no ADR/class flags"
                    ),
                    observed_relationship="primary identity fields conflict",
                )
        elif security_role == "ADR":
            adr_ratio = _finite_number(packet.get("adr_ratio"), positive=True)
            ratio_evidence_complete = bool(
                packet.get("is_adr") is True
                and packet.get("is_secondary_class") is False
                and adr_ratio is not None
                and quote_ratio is not None
                and metric_values_reconcile(adr_ratio, quote_ratio)
                and _valid_security_ratio_reference(packet, ticker=ticker)
            )
            if not ratio_evidence_complete:
                _violation(
                    violations,
                    "CAP_SECURITY_RATIO_UNVERIFIED",
                    "ADR issuer-share arithmetic requires authoritative security-bound ratio evidence.",
                    ticker=ticker,
                    field_name="adr_ratio",
                    source_values={
                        "adr_ratio": packet.get("adr_ratio"),
                        "issuer_quote_ratio": packet.get("issuer_quote_ratio"),
                        "ratio_source_url": packet.get("ratio_source_url"),
                        "ratio_source_accession": packet.get("ratio_source_accession"),
                        "ratio_security_symbol": packet.get("ratio_security_symbol"),
                    },
                    expected_relationship="ADR ratio proof bound to the requested quote",
                    observed_relationship="ADR ratio proof is incomplete or inconsistent",
                    terminal_status=NEEDS_DATA,
                )
        elif security_role == "SECONDARY_CLASS":
            share_class_ratio = _finite_number(packet.get("share_class_ratio"), positive=True)
            ratio_evidence_complete = bool(
                packet.get("is_secondary_class") is True
                and packet.get("is_adr") is False
                and share_class_ratio is not None
                and quote_ratio is not None
                and metric_values_reconcile(share_class_ratio, quote_ratio)
                and _valid_security_ratio_reference(packet, ticker=ticker)
            )
            if not ratio_evidence_complete:
                _violation(
                    violations,
                    "CAP_SECURITY_RATIO_UNVERIFIED",
                    "Secondary-class issuer-share arithmetic requires authoritative ratio evidence.",
                    ticker=ticker,
                    field_name="share_class_ratio",
                    source_values={
                        "share_class_ratio": packet.get("share_class_ratio"),
                        "issuer_quote_ratio": packet.get("issuer_quote_ratio"),
                        "ratio_source_url": packet.get("ratio_source_url"),
                        "ratio_source_accession": packet.get("ratio_source_accession"),
                        "ratio_security_symbol": packet.get("ratio_security_symbol"),
                    },
                    expected_relationship="share-class ratio proof bound to the requested quote",
                    observed_relationship="share-class ratio proof is incomplete or inconsistent",
                    terminal_status=NEEDS_DATA,
                )
        else:
            _violation(
                violations,
                "CAP_SECURITY_IDENTITY_UNVERIFIED",
                "Issuer-wide shares cannot be multiplied by an unresolved security quote.",
                ticker=ticker,
                field_name="security_role",
                observed=security_role,
                expected="PRIMARY, ADR, or SECONDARY_CLASS",
                terminal_status=NEEDS_DATA,
            )
    if market_cap_number is not None and price_number is not None and shares_number is not None:
        if quote_ratio is None and not issuer_share_arithmetic and packet.get("is_adr") is True:
            quote_ratio = _finite_number(packet.get("adr_ratio"), positive=True)
        if (
            quote_ratio is None
            and not issuer_share_arithmetic
            and packet.get("is_secondary_class") is True
        ):
            quote_ratio = _finite_number(packet.get("share_class_ratio"), positive=True)
        if quote_ratio is None and not issuer_share_arithmetic:
            quote_ratio = 1.0
        expected_cap = (
            price_number * shares_number / quote_ratio if quote_ratio is not None else None
        )
        if expected_cap is not None and not metric_values_reconcile(
            market_cap_number,
            expected_cap,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            _violation(
                violations,
                "MARKET_CAP_RECONCILIATION_FAILED",
                "market_cap_mm must reconcile to current_price * shares_outstanding_mm / issuer_quote_ratio.",
                ticker=ticker,
                field_name="market_cap_mm",
                source_values={
                    "market_cap_mm": market_cap_number,
                    "current_price": price_number,
                    "shares_outstanding_mm": shares_number,
                    "issuer_quote_ratio": quote_ratio,
                },
                expected_relationship=(
                    "market_cap_mm == current_price * shares_outstanding_mm / issuer_quote_ratio"
                ),
                observed_relationship=(
                    f"market_cap_mm={market_cap_number}; recomputed={expected_cap}"
                ),
            )

    traces = packet.get("metric_traces")
    if not isinstance(traces, Mapping):
        traces = {}
    for canonical_name, metric_path, metric_value in _investment_metric_occurrences(packet):
        if _finite_number(metric_value) is None:
            _violation(
                violations,
                "DERIVED_METRIC_NON_FINITE",
                "Investment-relevant derived metrics must be finite.",
                ticker=ticker,
                field_name=metric_path,
                observed=metric_value,
                expected="finite derived value",
            )
        raw_trace = _trace_for_metric(
            traces,
            canonical_name=canonical_name,
            metric_path=metric_path,
        )
        if raw_trace is None:
            _violation(
                violations,
                "METRIC_TRACE_MISSING",
                f"Metric {metric_path!s} requires exact inputs and formula provenance.",
                ticker=ticker,
                field_name=f"metric_traces.{canonical_name}",
                source_values={metric_path: metric_value},
                expected_relationship=(
                    f"metric_traces contains a reconciling trace for {metric_path}"
                ),
                observed_relationship="trace missing",
                terminal_status=NEEDS_DATA,
            )
        elif not isinstance(raw_trace, Mapping) or not reconcile_metric_trace(raw_trace):
            _violation(
                violations,
                "METRIC_TRACE_RECONCILIATION_FAILED",
                f"Metric trace {metric_path!s} does not reconcile.",
                ticker=ticker,
                field_name=f"metric_traces.{canonical_name}",
                source_values={
                    "metric_value": metric_value,
                    "metric_trace": raw_trace,
                },
                expected_relationship="trace.output == trace.recomputed_output",
                observed_relationship=(
                    raw_trace.get("reconciles")
                    if isinstance(raw_trace, Mapping)
                    else "trace is not a mapping"
                ),
            )
        else:
            if not metric_values_reconcile(metric_value, raw_trace.get("output")):
                _violation(
                    violations,
                    "DERIVED_METRIC_TRACE_OUTPUT_MISMATCH",
                    "Stored derived metric differs from its formula trace output.",
                    ticker=ticker,
                    field_name=metric_path,
                    source_values={
                        "stored_metric_value": metric_value,
                        "trace_output": raw_trace.get("output"),
                    },
                    expected_relationship="stored metric value == trace.output",
                    observed_relationship=(f"{metric_value} != {raw_trace.get('output')}"),
                )
            if (
                raw_trace.get("quote_snapshot_id")
                and raw_trace.get("quote_snapshot_id") != quote_snapshot_id
            ):
                _violation(
                    violations,
                    "METRIC_QUOTE_SNAPSHOT_MISMATCH",
                    "Derived metric trace uses a different quote snapshot.",
                    ticker=ticker,
                    field_name=f"metric_traces.{canonical_name}.quote_snapshot_id",
                    source_values={
                        "packet_quote_snapshot_id": quote_snapshot_id,
                        "metric_quote_snapshot_id": raw_trace.get("quote_snapshot_id"),
                    },
                    expected_relationship="metric quote snapshot == packet quote snapshot",
                    observed_relationship=(
                        f"{raw_trace.get('quote_snapshot_id')} != {quote_snapshot_id}"
                    ),
                )

    for metric_name, raw_trace in traces.items():
        if not isinstance(raw_trace, Mapping) or not reconcile_metric_trace(raw_trace):
            field_name = f"metric_traces.{metric_name}"
            if any(item.field == field_name for item in violations):
                continue
            _violation(
                violations,
                "METRIC_TRACE_RECONCILIATION_FAILED",
                f"Metric trace {metric_name!s} does not reconcile.",
                ticker=ticker,
                field_name=field_name,
                source_values={"metric_trace": raw_trace},
                expected_relationship="trace.output == trace.recomputed_output",
                observed_relationship=(
                    raw_trace.get("reconciles")
                    if isinstance(raw_trace, Mapping)
                    else "trace is not a mapping"
                ),
            )
            continue
        trace_metric = _canonical_trace_metric_name(raw_trace.get("metric") or metric_name)
        if trace_metric not in _KNOWN_TRACE_METRICS:
            continue
        (
            trace_metric,
            formula,
            formula_valid,
            output_unit,
            output_unit_valid,
            declared_metric_matches,
        ) = _trace_contract_state(raw_trace, expected_metric=metric_name)
        if not declared_metric_matches:
            _violation(
                violations,
                "METRIC_TRACE_METRIC_MISMATCH",
                "Metric trace must declare the metric carried by its mapping key.",
                ticker=ticker,
                field_name=f"metric_traces.{metric_name}.metric",
                source_values={
                    "trace_key": metric_name,
                    "declared_metric": raw_trace.get("metric"),
                },
                expected_relationship=f"trace.metric canonicalizes to {trace_metric}",
                observed_relationship=raw_trace.get("metric"),
                terminal_status=(
                    NEEDS_DATA
                    if not str(raw_trace.get("metric") or "").strip()
                    else INVALID_FINANCIAL_INPUT
                ),
            )
        if not formula_valid:
            _violation(
                violations,
                ("METRIC_TRACE_FORMULA_MISSING" if not formula else "METRIC_TRACE_FORMULA_INVALID"),
                "Known metric trace requires an exact recognized formula.",
                ticker=ticker,
                field_name=f"metric_traces.{metric_name}.formula",
                source_values={"formula": formula, "metric": trace_metric},
                expected_relationship=sorted(
                    _CANONICAL_TRACE_FORMULAS.get(trace_metric, frozenset())
                ),
                observed_relationship=formula or None,
                terminal_status=NEEDS_DATA if not formula else INVALID_FINANCIAL_INPUT,
            )
        if not output_unit_valid:
            _violation(
                violations,
                (
                    "METRIC_TRACE_OUTPUT_UNIT_MISSING"
                    if not output_unit
                    else "METRIC_TRACE_OUTPUT_UNIT_INVALID"
                ),
                "Known metric trace requires its canonical output unit.",
                ticker=ticker,
                field_name=f"metric_traces.{metric_name}.output_unit",
                source_values={"output_unit": output_unit, "metric": trace_metric},
                expected_relationship=sorted(
                    _CANONICAL_TRACE_OUTPUT_UNITS.get(trace_metric, frozenset())
                ),
                observed_relationship=output_unit or None,
                terminal_status=(NEEDS_DATA if not output_unit else INVALID_FINANCIAL_INPUT),
            )
        _validate_trace_input_provenance(
            raw_trace,
            run_as_of_date=run_as_of_date,
            violations=violations,
            ticker=ticker,
            field_prefix=f"metric_traces.{metric_name}",
            code_prefix="METRIC_TRACE",
        )
        inputs_complete, independently_recomputed = _independent_trace_recompute(raw_trace)
        if not inputs_complete:
            _violation(
                violations,
                "METRIC_TRACE_INPUTS_MISSING",
                "Known investment metric trace lacks the canonical inputs needed for independent recomputation.",
                ticker=ticker,
                field_name=f"metric_traces.{metric_name}.inputs",
                source_values={"metric_trace": raw_trace},
                expected_relationship=(
                    f"canonical inputs are present for {trace_metric} recomputation"
                ),
                observed_relationship="required canonical trace inputs missing",
                terminal_status=NEEDS_DATA,
            )
        elif not metric_values_reconcile(
            raw_trace.get("output"), independently_recomputed, rel_tol=1e-9, abs_tol=1e-12
        ):
            _violation(
                violations,
                "METRIC_TRACE_INDEPENDENT_RECOMPUTATION_FAILED",
                "Known investment metric trace does not reconcile to an independent formula recomputation.",
                ticker=ticker,
                field_name=f"metric_traces.{metric_name}",
                source_values={
                    "metric_trace": raw_trace,
                    "independently_recomputed": independently_recomputed,
                },
                expected_relationship="trace.output == independent formula result",
                observed_relationship=(f"{raw_trace.get('output')} != {independently_recomputed}"),
            )


def _validate_scenario(
    scenario: Mapping[str, Any],
    *,
    run_as_of_date: str,
    packet_snapshot_ids: Mapping[str, str],
    packets_by_ticker: Mapping[str, Mapping[str, Any]],
    violations: list[FinancialIntegrityViolation],
) -> None:
    ticker = str(scenario.get("ticker") or "").strip().upper()
    snapshot_id = str(scenario.get("quote_snapshot_id") or "").strip().lower()
    expected_snapshot_id = packet_snapshot_ids.get(ticker)
    if not ticker or expected_snapshot_id is None or snapshot_id != expected_snapshot_id:
        missing_binding = not ticker or expected_snapshot_id is None or not snapshot_id
        _violation(
            violations,
            "SCENARIO_QUOTE_SNAPSHOT_MISMATCH",
            "Scenario must bind the exact packet quote snapshot.",
            ticker=ticker or None,
            field_name="quote_snapshot_id",
            observed=snapshot_id or None,
            expected=expected_snapshot_id,
            terminal_status=NEEDS_DATA if missing_binding else INVALID_FINANCIAL_INPUT,
        )
    packet = packets_by_ticker.get(ticker, {})
    scenario_price = scenario.get("current_price")
    packet_price = packet.get("current_price")
    if (
        _finite_number(scenario_price, positive=True) is not None
        and _finite_number(packet_price, positive=True) is not None
        and not metric_values_reconcile(scenario_price, packet_price, rel_tol=1e-6, abs_tol=1e-6)
    ):
        _violation(
            violations,
            "SCENARIO_QUOTE_VALUE_MISMATCH",
            "Scenario must use the exact packet quote value.",
            ticker=ticker or None,
            field_name="current_price",
            source_values={
                "scenario_current_price": scenario_price,
                "packet_current_price": packet_price,
            },
            expected_relationship="scenario current_price == packet current_price",
            observed_relationship=f"{scenario_price} != {packet_price}",
        )
    scenario_price_unit = scenario.get("current_price_unit")
    if not str(scenario_price_unit or "").strip():
        _violation(
            violations,
            "SCENARIO_PRICE_UNIT_MISSING",
            "Scenario current-price unit provenance is required.",
            ticker=ticker or None,
            field_name="current_price_unit",
            observed=scenario_price_unit,
            expected=PRICE_UNIT_USD_PER_SHARE,
            terminal_status=NEEDS_DATA,
        )
    elif scenario_price_unit != PRICE_UNIT_USD_PER_SHARE:
        _violation(
            violations,
            "SCENARIO_PRICE_UNIT_INVALID",
            "Scenario current price requires USD_per_share.",
            ticker=ticker or None,
            field_name="current_price_unit",
            observed=scenario.get("current_price_unit"),
            expected=PRICE_UNIT_USD_PER_SHARE,
        )
    packet_price_unit = packet.get("current_price_unit")
    if scenario_price_unit and packet_price_unit and scenario_price_unit != packet_price_unit:
        _violation(
            violations,
            "SCENARIO_PRICE_UNIT_MISMATCH",
            "Scenario price unit must equal the packet price unit.",
            ticker=ticker or None,
            field_name="current_price_unit",
            source_values={
                "scenario_price_unit": scenario_price_unit,
                "packet_price_unit": packet_price_unit,
            },
            expected_relationship="scenario price unit == packet price unit",
            observed_relationship=f"{scenario_price_unit} != {packet_price_unit}",
        )
    scenario_price_basis = str(scenario.get("price_basis") or "").upper()
    if not scenario_price_basis:
        _violation(
            violations,
            "SCENARIO_PRICE_BASIS_MISSING",
            "Scenario price-basis provenance is required.",
            ticker=ticker or None,
            field_name="price_basis",
            observed=scenario.get("price_basis"),
            expected=sorted(_VALID_PRICE_BASES),
            terminal_status=NEEDS_DATA,
        )
    elif scenario_price_basis not in _VALID_PRICE_BASES:
        _violation(
            violations,
            "SCENARIO_PRICE_BASIS_INVALID",
            "Scenario price basis must be explicit.",
            ticker=ticker or None,
            field_name="price_basis",
            observed=scenario.get("price_basis"),
        )
    packet_price_basis = str(packet.get("price_basis") or "").upper()
    if (
        scenario_price_basis in _VALID_PRICE_BASES
        and packet_price_basis in _VALID_PRICE_BASES
        and scenario_price_basis != packet_price_basis
    ):
        _violation(
            violations,
            "SCENARIO_PRICE_BASIS_MISMATCH",
            "Scenario price basis must equal the packet price basis.",
            ticker=ticker or None,
            field_name="price_basis",
            source_values={
                "scenario_price_basis": scenario_price_basis,
                "packet_price_basis": packet_price_basis,
            },
            expected_relationship="scenario price basis == packet price basis",
            observed_relationship=(f"{scenario_price_basis} != {packet_price_basis}"),
        )
    annualized_return = scenario.get("annualized_return")
    if annualized_return is not None:
        if _finite_number(annualized_return) is None:
            _violation(
                violations,
                "SCENARIO_RETURN_NON_FINITE",
                "Scenario annualized return must be finite.",
                ticker=ticker or None,
                field_name="annualized_return",
                observed=annualized_return,
                expected="finite annualized return",
            )
        trace = scenario.get("metric_trace")
        if not isinstance(trace, Mapping) or not trace:
            _violation(
                violations,
                "SCENARIO_METRIC_TRACE_MISSING",
                "Scenario annualized return requires exact formula provenance.",
                ticker=ticker or None,
                field_name="metric_trace",
                source_values={"annualized_return": annualized_return},
                expected_relationship="scenario metric trace reconciles annualized return",
                observed_relationship="trace missing",
                terminal_status=NEEDS_DATA,
            )
        elif not reconcile_metric_trace(trace):
            _violation(
                violations,
                "SCENARIO_METRIC_TRACE_RECONCILIATION_FAILED",
                "Scenario annualized-return trace does not reconcile.",
                ticker=ticker or None,
                field_name="metric_trace",
                source_values={"metric_trace": trace},
                expected_relationship="trace.output == trace.recomputed_output",
                observed_relationship=trace.get("reconciles"),
            )
        else:
            (
                _trace_metric,
                formula,
                formula_valid,
                output_unit,
                output_unit_valid,
                declared_metric_matches,
            ) = _trace_contract_state(trace, expected_metric="annualized_return")
            if not declared_metric_matches:
                _violation(
                    violations,
                    "SCENARIO_METRIC_TRACE_METRIC_MISMATCH",
                    "Scenario trace must declare annualized_return.",
                    ticker=ticker or None,
                    field_name="metric_trace.metric",
                    source_values={"declared_metric": trace.get("metric")},
                    expected_relationship="trace.metric == annualized_return",
                    observed_relationship=trace.get("metric"),
                    terminal_status=(
                        NEEDS_DATA
                        if not str(trace.get("metric") or "").strip()
                        else INVALID_FINANCIAL_INPUT
                    ),
                )
            if not formula_valid:
                _violation(
                    violations,
                    (
                        "SCENARIO_METRIC_TRACE_FORMULA_MISSING"
                        if not formula
                        else "SCENARIO_METRIC_TRACE_FORMULA_INVALID"
                    ),
                    "Scenario trace requires an exact recognized annualized-return formula.",
                    ticker=ticker or None,
                    field_name="metric_trace.formula",
                    source_values={"formula": formula},
                    expected_relationship=sorted(_CANONICAL_TRACE_FORMULAS["annualized_return"]),
                    observed_relationship=formula or None,
                    terminal_status=(NEEDS_DATA if not formula else INVALID_FINANCIAL_INPUT),
                )
            if not output_unit_valid:
                _violation(
                    violations,
                    (
                        "SCENARIO_METRIC_TRACE_OUTPUT_UNIT_MISSING"
                        if not output_unit
                        else "SCENARIO_METRIC_TRACE_OUTPUT_UNIT_INVALID"
                    ),
                    "Scenario trace requires the annualized_ratio output unit.",
                    ticker=ticker or None,
                    field_name="metric_trace.output_unit",
                    source_values={"output_unit": output_unit},
                    expected_relationship="annualized_ratio",
                    observed_relationship=output_unit or None,
                    terminal_status=(NEEDS_DATA if not output_unit else INVALID_FINANCIAL_INPUT),
                )
            _validate_trace_input_provenance(
                trace,
                run_as_of_date=run_as_of_date,
                violations=violations,
                ticker=ticker or None,
                field_prefix="metric_trace",
                code_prefix="SCENARIO_METRIC_TRACE",
            )
            if not metric_values_reconcile(annualized_return, trace.get("output")):
                _violation(
                    violations,
                    "SCENARIO_RETURN_TRACE_OUTPUT_MISMATCH",
                    "Stored scenario return differs from its trace output.",
                    ticker=ticker or None,
                    field_name="annualized_return",
                    source_values={
                        "annualized_return": annualized_return,
                        "trace_output": trace.get("output"),
                    },
                    expected_relationship="annualized_return == trace.output",
                    observed_relationship=(f"{annualized_return} != {trace.get('output')}"),
                )
            inputs_complete, independently_recomputed = _independent_trace_recompute(trace)
            if not inputs_complete:
                _violation(
                    violations,
                    "SCENARIO_METRIC_TRACE_INPUTS_MISSING",
                    "Scenario trace lacks inputs for independent return recomputation.",
                    ticker=ticker or None,
                    field_name="metric_trace.inputs",
                    source_values={"metric_trace": trace},
                    expected_relationship="scenario trace carries future value, price, and horizon",
                    observed_relationship="required canonical trace inputs missing",
                    terminal_status=NEEDS_DATA,
                )
            elif not metric_values_reconcile(trace.get("output"), independently_recomputed):
                _violation(
                    violations,
                    "SCENARIO_METRIC_TRACE_INDEPENDENT_RECOMPUTATION_FAILED",
                    "Scenario trace does not reconcile to the independently recomputed return.",
                    ticker=ticker or None,
                    field_name="metric_trace",
                    source_values={
                        "metric_trace": trace,
                        "independently_recomputed": independently_recomputed,
                    },
                    expected_relationship="trace.output == independent annualized return",
                    observed_relationship=(f"{trace.get('output')} != {independently_recomputed}"),
                )
            if trace.get("quote_snapshot_id") != snapshot_id:
                _violation(
                    violations,
                    "SCENARIO_METRIC_QUOTE_SNAPSHOT_MISMATCH",
                    "Scenario formula trace must bind the scenario quote snapshot.",
                    ticker=ticker or None,
                    field_name="metric_trace.quote_snapshot_id",
                    source_values={
                        "scenario_quote_snapshot_id": snapshot_id,
                        "metric_quote_snapshot_id": trace.get("quote_snapshot_id"),
                    },
                    expected_relationship="metric quote snapshot == scenario quote snapshot",
                    observed_relationship=(f"{trace.get('quote_snapshot_id')} != {snapshot_id}"),
                )


def validate_financial_integrity_scope(
    scope: FinancialIntegrityScope,
) -> FinancialIntegrityGateResult:
    violations: list[FinancialIntegrityViolation] = []
    snapshot_ids: dict[str, str] = {}
    packet_payloads = [_mapping(packet) for packet in scope.packets]
    scenario_payloads = [_mapping(scenario) for scenario in scope.scenarios]
    seen_packet_snapshots: dict[str, str] = {}
    for packet in packet_payloads:
        ticker = str(packet.get("ticker") or "").strip().upper()
        snapshot_id = str(packet.get("quote_snapshot_id") or "").strip().lower()
        prior_snapshot_id = seen_packet_snapshots.get(ticker)
        if ticker and snapshot_id and prior_snapshot_id and snapshot_id != prior_snapshot_id:
            _violation(
                violations,
                "DUPLICATE_TICKER_QUOTE_SNAPSHOT_CONFLICT",
                "Duplicate ticker packets in one scope cannot carry different quote snapshots.",
                ticker=ticker,
                field_name="quote_snapshot_id",
                source_values={
                    "first_quote_snapshot_id": prior_snapshot_id,
                    "duplicate_quote_snapshot_id": snapshot_id,
                },
                expected_relationship="one quote snapshot identity per ticker per scope",
                observed_relationship=f"{prior_snapshot_id} != {snapshot_id}",
            )
        elif ticker and snapshot_id:
            seen_packet_snapshots.setdefault(ticker, snapshot_id)
    packets_by_ticker = {
        str(packet.get("ticker") or "").strip().upper(): packet
        for packet in packet_payloads
        if str(packet.get("ticker") or "").strip()
    }

    try:
        date.fromisoformat(str(scope.run_as_of_date or "")[:10])
    except ValueError:
        _violation(
            violations,
            "RUN_ASOF_INVALID",
            "Financial integrity scope requires a valid ISO run date.",
            field_name="run_as_of_date",
            observed=scope.run_as_of_date,
            terminal_status=(
                NEEDS_DATA
                if not str(scope.run_as_of_date or "").strip()
                else INVALID_FINANCIAL_INPUT
            ),
        )
    if not packet_payloads:
        _violation(
            violations,
            "PACKETS_MISSING",
            "Financial integrity scope requires at least one packet.",
            field_name="packets",
            terminal_status=NEEDS_DATA,
        )

    for packet in packet_payloads:
        _validate_packet(
            packet,
            run_as_of_date=scope.run_as_of_date,
            violations=violations,
            snapshot_ids=snapshot_ids,
        )
    for scenario in scenario_payloads:
        _validate_scenario(
            scenario,
            run_as_of_date=scope.run_as_of_date,
            packet_snapshot_ids=snapshot_ids,
            packets_by_ticker=packets_by_ticker,
            violations=violations,
        )

    fingerprint_payload = {
        "context": scope.context,
        "run_as_of_date": scope.run_as_of_date,
        "packets": packet_payloads,
        "scenarios": scenario_payloads,
    }
    if not violations:
        status = FINANCIAL_INTEGRITY_PASS
    elif any(item.terminal_status == INVALID_FINANCIAL_INPUT for item in violations):
        status = INVALID_FINANCIAL_INPUT
    else:
        status = NEEDS_DATA
    return FinancialIntegrityGateResult(
        context=scope.context,
        run_as_of_date=scope.run_as_of_date,
        status=status,
        violations=tuple(violations),
        scope_fingerprint=hashlib.sha256(
            _canonical_json(fingerprint_payload).encode("utf-8")
        ).hexdigest(),
        packet_count=len(packet_payloads),
        scenario_count=len(scenario_payloads),
        ticker_snapshot_ids=dict(sorted(snapshot_ids.items())),
    )


def require_financial_integrity_scope(
    scope: FinancialIntegrityScope,
) -> FinancialIntegrityGateResult:
    result = validate_financial_integrity_scope(scope)
    if not result.passed:
        raise InvalidFinancialInputError(result)
    return result


def require_unchanged_financial_integrity_scope(
    scope: FinancialIntegrityScope,
    *,
    expected_scope_fingerprint: str,
) -> FinancialIntegrityGateResult:
    """Revalidate a previously authorized scope and reject any input drift."""

    result = require_financial_integrity_scope(scope)
    expected = str(expected_scope_fingerprint or "").strip()
    if expected and result.scope_fingerprint == expected:
        return result
    violation = FinancialIntegrityViolation(
        code="BOUND_FINANCIAL_INPUT_MUTATED",
        field="financial_integrity_scope",
        source_values={
            "expected_scope_fingerprint": expected or None,
            "observed_scope_fingerprint": result.scope_fingerprint,
        },
        expected_relationship="financial input scope fingerprint remains unchanged",
        observed_relationship=(f"{result.scope_fingerprint} != {expected or 'MISSING'}"),
        reason=("Financial inputs changed after the exact provider scope was authorized."),
        terminal_status=INVALID_FINANCIAL_INPUT,
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=result.context,
            run_as_of_date=result.run_as_of_date,
            status=INVALID_FINANCIAL_INPUT,
            violations=(violation,),
            scope_fingerprint=result.scope_fingerprint,
            packet_count=result.packet_count,
            scenario_count=result.scenario_count,
            ticker_snapshot_ids=result.ticker_snapshot_ids,
        )
    )


__all__ = [
    "FINANCIAL_INTEGRITY_PASS",
    "INVALID_FINANCIAL_INPUT",
    "NEEDS_DATA",
    "MARKET_CAP_UNIT_USD_MILLIONS",
    "PRICE_BASIS_SPLIT_ADJUSTED",
    "PRICE_BASIS_UNADJUSTED",
    "PRICE_UNIT_USD_PER_SHARE",
    "SHARES_BASIS_ISSUER_REPORTED",
    "SHARES_BASIS_UNADJUSTED",
    "SHARES_UNIT_MILLIONS",
    "SPLIT_PROOF_CACHE_RECORD_TYPE",
    "SPLIT_PROOF_CACHE_SCHEMA_VERSION",
    "SPLIT_PROOF_KIND_NO_INTERVENING_SPLIT",
    "SPLIT_PROOF_KIND_SPLIT_EVENT",
    "SPLIT_PROOF_RAW_CACHE_RECORD_TYPE",
    "SPLIT_PROOF_RAW_CACHE_SCHEMA_VERSION",
    "FinancialIntegrityGateResult",
    "FinancialIntegrityScope",
    "FinancialIntegrityViolation",
    "InvalidFinancialInputError",
    "authoritative_split_proof_reference",
    "build_metric_trace",
    "canonical_split_proof_cache_path",
    "canonical_split_proof_cache_root",
    "canonical_split_proof_raw_cache_path",
    "canonical_metric_trace",
    "metric_trace_reconciles",
    "metric_values_reconcile",
    "reconcile_metric_trace",
    "require_financial_integrity_scope",
    "require_unchanged_financial_integrity_scope",
    "stable_quote_hash",
    "stable_quote_snapshot_id",
    "split_proof_materialization_envelope",
    "split_proof_raw_materialization_envelope",
    "validate_financial_integrity_scope",
]
