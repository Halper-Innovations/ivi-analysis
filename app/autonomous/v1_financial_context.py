"""Provider-free construction of canonical V1 financial-integrity scopes.

Legacy callers historically assembled signal packets before resolving the
quote/cap context that the deterministic integrity gate requires.  This module
keeps the repair sequence in one place: classify from cached point-in-time
evidence with live market data disabled, bind that exact classification into
signal assembly, and expose the resulting immutable scope.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.alpha.schemas import TickerSignalPacket
from app.autonomous.financial_integrity import (
    INVALID_FINANCIAL_INPUT,
    FinancialIntegrityGateResult,
    FinancialIntegrityScope,
    FinancialIntegrityViolation,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
)
from app.config import AppConfig, get_config


def _scope_value(value: Any) -> Any:
    """Return a detached value composed only of canonical JSON primitives."""

    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "financial-integrity scope mappings require string keys; "
                    f"received {type(key).__name__}"
                )
            normalized[key] = _scope_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_scope_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "financial-integrity scope values must be JSON primitives, mappings, "
        f"sequences, or dataclasses; received {type(value).__name__}"
    )


def _first_nonfinite_path(value: Any, *, path: str = "") -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return path or "<root>"
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _first_nonfinite_path(
                item,
                path=f"{path}.{key}" if path else str(key),
            )
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _first_nonfinite_path(
                item,
                path=f"{path}[{index}]" if path else f"[{index}]",
            )
            if found is not None:
                return found
    return None


def _raise_scope_binding_error(
    *,
    context: str,
    run_as_of_date: str,
    code: str,
    reason: str,
    source_values: Mapping[str, Any],
    expected_relationship: Any,
    observed_relationship: Any,
    scope_fingerprint: str = "",
    packet_count: int = 0,
    scenario_count: int = 0,
) -> None:
    violation = FinancialIntegrityViolation(
        code=code,
        field="financial_integrity_scope",
        source_values=dict(source_values),
        expected_relationship=expected_relationship,
        observed_relationship=observed_relationship,
        reason=reason,
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=context,
            run_as_of_date=run_as_of_date,
            status=INVALID_FINANCIAL_INPUT,
            violations=(violation,),
            scope_fingerprint=scope_fingerprint,
            packet_count=packet_count,
            scenario_count=scenario_count,
        )
    )


def financial_input_scenario(
    packet: TickerSignalPacket | Mapping[str, Any],
    *,
    financial_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact prompt financial inputs to one canonical quote packet."""

    packet_payload = _scope_value(packet)
    if not isinstance(packet_payload, Mapping):
        packet_payload = {}
    return {
        "ticker": str(packet_payload.get("ticker") or "").strip().upper(),
        "quote_snapshot_id": packet_payload.get("quote_snapshot_id"),
        "current_price": packet_payload.get("current_price"),
        "current_price_unit": packet_payload.get("current_price_unit"),
        "price_basis": packet_payload.get("price_basis"),
        "financial_inputs": _scope_value(financial_inputs),
    }


@dataclass(frozen=True)
class BoundV1FinancialScope:
    """A canonical scope whose exact paid-boundary inputs cannot drift."""

    context: str
    run_as_of_date: str
    packets: tuple[Any, ...]
    scenarios: tuple[Any, ...]
    expected_scope_fingerprint: str

    def require(
        self,
        *,
        scenarios: Sequence[Any] | None = None,
    ) -> FinancialIntegrityGateResult:
        try:
            current_scenarios = tuple(
                _scope_value(item) for item in (self.scenarios if scenarios is None else scenarios)
            )
            nonfinite_path = _first_nonfinite_path(current_scenarios)
            if nonfinite_path is not None:
                _raise_scope_binding_error(
                    context=self.context,
                    run_as_of_date=self.run_as_of_date,
                    code="BOUND_FINANCIAL_INPUT_NON_FINITE",
                    reason="Paid-boundary financial inputs must be finite.",
                    source_values={"nonfinite_path": nonfinite_path},
                    expected_relationship="all bound financial inputs are finite",
                    observed_relationship=f"non-finite value at {nonfinite_path}",
                    scope_fingerprint=self.expected_scope_fingerprint,
                    packet_count=len(self.packets),
                    scenario_count=len(current_scenarios),
                )
            json.dumps(
                current_scenarios,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            _raise_scope_binding_error(
                context=self.context,
                run_as_of_date=self.run_as_of_date,
                code="BOUND_FINANCIAL_INPUT_NOT_CANONICAL_JSON",
                reason=("Paid-boundary financial inputs must be strictly canonical JSON."),
                source_values={"error": str(exc)},
                expected_relationship=(
                    "only JSON primitives, string-keyed mappings, sequences, and dataclasses"
                ),
                observed_relationship=type(exc).__name__,
                scope_fingerprint=self.expected_scope_fingerprint,
                packet_count=len(self.packets),
                scenario_count=len(self.scenarios),
            )
        scope = FinancialIntegrityScope(
            context=self.context,
            run_as_of_date=self.run_as_of_date,
            packets=self.packets,
            scenarios=current_scenarios,
        )
        result = require_financial_integrity_scope(scope)
        if result.scope_fingerprint != self.expected_scope_fingerprint:
            _raise_scope_binding_error(
                context=self.context,
                run_as_of_date=self.run_as_of_date,
                code="BOUND_FINANCIAL_INPUT_MUTATED",
                reason=(
                    "Financial inputs changed after the canonical provider scope was authorized."
                ),
                source_values={
                    "expected_scope_fingerprint": self.expected_scope_fingerprint,
                    "observed_scope_fingerprint": result.scope_fingerprint,
                },
                expected_relationship=("provider-bound input fingerprint remains unchanged"),
                observed_relationship=(
                    f"{result.scope_fingerprint} != {self.expected_scope_fingerprint}"
                ),
                scope_fingerprint=result.scope_fingerprint,
                packet_count=result.packet_count,
                scenario_count=result.scenario_count,
            )
        return result


def bind_v1_financial_scope(
    *,
    context: str,
    run_as_of_date: str,
    packets: Sequence[Any],
    scenarios: Sequence[Any],
) -> BoundV1FinancialScope:
    """Validate and freeze the exact inputs intended for paid reasoning."""

    try:
        frozen_packets = tuple(_scope_value(item) for item in packets)
        frozen_scenarios = tuple(_scope_value(item) for item in scenarios)
        nonfinite_path = _first_nonfinite_path(
            {"packets": frozen_packets, "scenarios": frozen_scenarios}
        )
        if nonfinite_path is not None:
            _raise_scope_binding_error(
                context=context,
                run_as_of_date=run_as_of_date,
                code="BOUND_FINANCIAL_INPUT_NON_FINITE",
                reason="Paid-boundary financial inputs must be finite.",
                source_values={"nonfinite_path": nonfinite_path},
                expected_relationship="all bound financial inputs are finite",
                observed_relationship=f"non-finite value at {nonfinite_path}",
                packet_count=len(frozen_packets),
                scenario_count=len(frozen_scenarios),
            )
        # The serializer deliberately has no ``default`` hook. Unsupported
        # objects, non-string mapping keys, and NaN/Infinity are authorization
        # failures instead of self-attested string fingerprints.
        json.dumps(
            {
                "packets": frozen_packets,
                "scenarios": frozen_scenarios,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        _raise_scope_binding_error(
            context=context,
            run_as_of_date=run_as_of_date,
            code="BOUND_FINANCIAL_INPUT_NOT_CANONICAL_JSON",
            reason="Paid-boundary financial inputs must be strictly canonical JSON.",
            source_values={"error": str(exc)},
            expected_relationship=(
                "only JSON primitives, string-keyed mappings, sequences, and dataclasses"
            ),
            observed_relationship=type(exc).__name__,
            packet_count=len(packets),
            scenario_count=len(scenarios),
        )
    scope = FinancialIntegrityScope(
        context=context,
        run_as_of_date=run_as_of_date,
        packets=frozen_packets,
        scenarios=frozen_scenarios,
    )
    result = require_financial_integrity_scope(scope)
    return BoundV1FinancialScope(
        context=context,
        run_as_of_date=run_as_of_date,
        packets=frozen_packets,
        scenarios=frozen_scenarios,
        expected_scope_fingerprint=result.scope_fingerprint,
    )


@dataclass(frozen=True)
class CanonicalV1FinancialContext:
    as_of_date: str
    packets: dict[str, TickerSignalPacket]
    current_prices: dict[str, float | None]
    issuer_contexts: dict[str, dict[str, Any]]

    def scope(self, *, context: str) -> FinancialIntegrityScope:
        return FinancialIntegrityScope(
            context=context,
            run_as_of_date=self.as_of_date,
            packets=tuple(self.packets.values()),
        )


def build_canonical_v1_financial_context(
    *,
    tickers: list[str] | tuple[str, ...],
    as_of_date: str,
    db_path: str | Path | None = None,
    scorecard_evidence: dict[str, tuple[str | None, dict[str, Any]]] | None = None,
    cfg: AppConfig | None = None,
) -> CanonicalV1FinancialContext:
    """Freeze cached V1 cap/quote evidence and assemble packets without spend."""

    from app.alpha.signal_assembler import assemble_sector_packets
    from app.sector.scan import classify_tickers_for_market_cap

    normalized_tickers = list(
        dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    )
    normalized_as_of = str(as_of_date or "").strip()[:10]
    if not normalized_as_of:
        raise ValueError("as_of_date is required for canonical V1 financial context")
    resolved_cfg = cfg or get_config()
    resolved_db_path = Path(db_path) if db_path is not None else Path(resolved_cfg.db_path)
    resolved_cfg = resolved_cfg.model_copy(update={"db_path": resolved_db_path})
    classifications = classify_tickers_for_market_cap(
        tickers=normalized_tickers,
        as_of_date=normalized_as_of,
        db_path=resolved_db_path,
        pipeline_version="v1",
        scorecard_evidence=scorecard_evidence,
        allow_live_market_data=False,
        cfg=resolved_cfg,
    )
    issuer_contexts = {
        ticker: classification.to_dict() for ticker, classification in classifications.items()
    }
    current_prices = {
        ticker: context.get("price_used")
        if isinstance(context.get("price_used"), (int, float))
        and not isinstance(context.get("price_used"), bool)
        else None
        for ticker, context in issuer_contexts.items()
    }
    packets = assemble_sector_packets(
        normalized_tickers,
        filing_risk_use_llm=False,
        as_of_date=normalized_as_of,
        pipeline_version="v1",
        current_prices=current_prices,
        issuer_contexts=issuer_contexts,
        db_path=resolved_db_path,
        cfg=resolved_cfg,
        allowed_filing_roots=(
            resolved_cfg.raw_filings_dir,
            resolved_cfg.cache_dir,
        ),
    )
    return CanonicalV1FinancialContext(
        as_of_date=normalized_as_of,
        packets=packets,
        current_prices=current_prices,
        issuer_contexts=issuer_contexts,
    )


__all__ = [
    "BoundV1FinancialScope",
    "CanonicalV1FinancialContext",
    "bind_v1_financial_scope",
    "build_canonical_v1_financial_context",
    "financial_input_scenario",
]
