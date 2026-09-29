from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


DEFAULT_LLM_CALL_TIMEOUT_SECONDS = 90.0
DEFAULT_LLM_CALL_DEADLINE_SECONDS = 180.0
DEFAULT_LLM_MAX_RETRIES_PER_CALL = 2
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = (1.0, 4.0)
DEFAULT_LLM_RETRY_RUN_BUDGET = 15


class LLMCallTimeoutError(TimeoutError):
    """Raised when a single LLM call attempt exceeds its wall-clock timeout."""


class LLMMaxRetriesExceeded(RuntimeError):
    """Raised when one LLM call exhausts its retry allowance."""


class LLMRetryBudgetExceeded(RuntimeError):
    """Raised when a sector run exhausts its cumulative retry budget."""


class LLMCostBudgetExceeded(RuntimeError):
    """Raised when a sector run would exceed its cumulative LLM cost budget."""


@dataclass
class LLMRetryEvent:
    provider: str
    schema_name: str
    attempt: int
    retry_number: int
    reason: str


@dataclass
class LLMRetryContext:
    max_retries: int = DEFAULT_LLM_RETRY_RUN_BUDGET
    retry_count: int = 0
    events: list[LLMRetryEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_retry(
        self,
        *,
        provider: str,
        schema_name: str,
        attempt: int,
        reason: str,
    ) -> None:
        with self._lock:
            self.retry_count += 1
            retry_number = self.retry_count
            event = LLMRetryEvent(
                provider=provider,
                schema_name=schema_name,
                attempt=attempt,
                retry_number=retry_number,
                reason=reason[:240],
            )
            self.events.append(event)
            print(
                "LLM retry "
                f"{retry_number}/{self.max_retries}: provider={provider} "
                f"schema={schema_name} attempt={attempt} reason={event.reason}",
                flush=True,
            )
            if retry_number > self.max_retries:
                raise LLMRetryBudgetExceeded(
                    "LLM retry budget exceeded "
                    f"({retry_number}>{self.max_retries}) while calling {provider}:{schema_name}"
                )

    def summary(self) -> dict[str, Any]:
        return {
            "retry_count": self.retry_count,
            "retry_budget": self.max_retries,
            "events": [
                {
                    "provider": event.provider,
                    "schema_name": event.schema_name,
                    "attempt": event.attempt,
                    "retry_number": event.retry_number,
                    "reason": event.reason,
                }
                for event in self.events
            ],
        }


@dataclass(frozen=True)
class LLMCostReservation:
    provider: str
    schema_name: str
    call_number: int
    estimated_cost_usd: float


@dataclass
class LLMCostEvent:
    provider: str
    schema_name: str
    call_number: int
    estimated_cost_usd: float
    actual_cost_usd: float
    cumulative_cost_usd: float


@dataclass
class LLMCostContext:
    max_cost_usd: float | None
    strict_first_call: bool = False
    call_count: int = 0
    cumulative_cost_usd: float = 0.0
    reserved_cost_usd: float = 0.0
    events: list[LLMCostEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve_call(
        self,
        *,
        provider: str,
        schema_name: str,
        estimated_cost_usd: float,
    ) -> LLMCostReservation | None:
        estimate = max(0.0, float(estimated_cost_usd or 0.0))
        with self._lock:
            call_number = self.call_count + 1
            projected = self.cumulative_cost_usd + self.reserved_cost_usd + estimate
            if (
                self.max_cost_usd is not None
                and (self.strict_first_call or call_number > 1)
                and projected > float(self.max_cost_usd)
            ):
                raise LLMCostBudgetExceeded(
                    "LLM cost budget exceeded "
                    f"(projected ${projected:.4f} > budget ${float(self.max_cost_usd):.4f}) "
                    f"before calling {provider}:{schema_name}"
                )
            self.call_count = call_number
            self.reserved_cost_usd += estimate
            return LLMCostReservation(
                provider=provider,
                schema_name=schema_name,
                call_number=call_number,
                estimated_cost_usd=estimate,
            )

    def complete_call(
        self,
        reservation: LLMCostReservation | None,
        *,
        actual_cost_usd: float,
    ) -> None:
        if reservation is None:
            return
        actual = max(0.0, float(actual_cost_usd or 0.0))
        with self._lock:
            self.reserved_cost_usd = max(
                0.0, self.reserved_cost_usd - reservation.estimated_cost_usd
            )
            self.cumulative_cost_usd += actual
            self.events.append(
                LLMCostEvent(
                    provider=reservation.provider,
                    schema_name=reservation.schema_name,
                    call_number=reservation.call_number,
                    estimated_cost_usd=reservation.estimated_cost_usd,
                    actual_cost_usd=actual,
                    cumulative_cost_usd=self.cumulative_cost_usd,
                )
            )

    def cancel_call(self, reservation: LLMCostReservation | None) -> None:
        if reservation is None:
            return
        with self._lock:
            self.reserved_cost_usd = max(
                0.0, self.reserved_cost_usd - reservation.estimated_cost_usd
            )

    def summary(self) -> dict[str, Any]:
        return {
            "max_cost_usd": self.max_cost_usd,
            "call_count": self.call_count,
            "cumulative_cost_usd": round(self.cumulative_cost_usd, 6),
            "reserved_cost_usd": round(self.reserved_cost_usd, 6),
            "events": [
                {
                    "provider": event.provider,
                    "schema_name": event.schema_name,
                    "call_number": event.call_number,
                    "estimated_cost_usd": event.estimated_cost_usd,
                    "actual_cost_usd": event.actual_cost_usd,
                    "cumulative_cost_usd": event.cumulative_cost_usd,
                }
                for event in self.events
            ],
        }


_context_local = threading.local()
_global_context_lock = threading.Lock()
_global_context: LLMRetryContext | None = None
_cost_context_local = threading.local()
_global_cost_context_lock = threading.Lock()
_global_cost_context: LLMCostContext | None = None
_attempt_observer_local = threading.local()
_physical_attempt_guard_local = threading.local()


@contextmanager
def llm_attempt_observer(
    observer: Callable[[dict[str, Any]], None] | None,
) -> Iterator[None]:
    """Observe each failed physical provider attempt without changing retry policy.

    Provider implementations can own their retry guard, so the observer is
    thread-local rather than threaded through every provider signature.  The
    callback runs synchronously after the physical attempt fails and before a
    retry (if any) is scheduled.
    """

    previous = getattr(_attempt_observer_local, "observer", None)
    _attempt_observer_local.observer = observer
    try:
        yield
    finally:
        if previous is None:
            if hasattr(_attempt_observer_local, "observer"):
                delattr(_attempt_observer_local, "observer")
        else:
            _attempt_observer_local.observer = previous


@contextmanager
def llm_physical_attempt_guard(
    guard: Callable[[dict[str, Any]], None] | None,
) -> Iterator[None]:
    """Require authorization immediately before every physical LLM attempt.

    The retry guard can own transport retries inside a provider, so a check
    performed only by the outer caller is insufficient. The callback runs
    synchronously before each attempt and may raise to suppress that attempt.
    """

    previous = getattr(_physical_attempt_guard_local, "guard", None)
    _physical_attempt_guard_local.guard = guard
    try:
        yield
    finally:
        if previous is None:
            if hasattr(_physical_attempt_guard_local, "guard"):
                delattr(_physical_attempt_guard_local, "guard")
        else:
            _physical_attempt_guard_local.guard = previous


def require_llm_physical_attempt_authorization(
    *,
    provider: str,
    schema_name: str,
    attempt: int,
) -> None:
    guard = getattr(_physical_attempt_guard_local, "guard", None)
    if guard is None:
        return
    guard(
        {
            "provider": provider,
            "schema_name": schema_name,
            "attempt": attempt,
        }
    )


def _notify_failed_attempt(
    *,
    provider: str,
    schema_name: str,
    attempt: int,
    exc: BaseException,
    retryable: bool,
    will_retry: bool,
) -> None:
    observer = getattr(_attempt_observer_local, "observer", None)
    if observer is None:
        return
    observer(
        {
            "provider": provider,
            "schema_name": schema_name,
            "attempt": attempt,
            "status": "ERROR",
            "retryable": retryable,
            "will_retry": will_retry,
            "error": exc,
        }
    )


def current_retry_context() -> LLMRetryContext | None:
    context = getattr(_context_local, "context", None)
    if isinstance(context, LLMRetryContext):
        return context
    with _global_context_lock:
        return _global_context


def current_cost_context() -> LLMCostContext | None:
    context = getattr(_cost_context_local, "context", None)
    if isinstance(context, LLMCostContext):
        return context
    with _global_cost_context_lock:
        return _global_cost_context


@contextmanager
def llm_retry_budget(max_retries: int = DEFAULT_LLM_RETRY_RUN_BUDGET) -> Iterator[LLMRetryContext]:
    context = LLMRetryContext(max_retries=max(0, int(max_retries)))
    previous_local = getattr(_context_local, "context", None)
    global _global_context
    with _global_context_lock:
        previous_global = _global_context
        _global_context = context
    _context_local.context = context
    try:
        yield context
    finally:
        if previous_local is None:
            if hasattr(_context_local, "context"):
                delattr(_context_local, "context")
        else:
            _context_local.context = previous_local
        with _global_context_lock:
            _global_context = previous_global


@contextmanager
def llm_cost_budget(
    max_cost_usd: float | None,
    *,
    strict_first_call: bool = False,
) -> Iterator[LLMCostContext]:
    context = LLMCostContext(
        max_cost_usd=max_cost_usd,
        strict_first_call=bool(strict_first_call),
    )
    previous_local = getattr(_cost_context_local, "context", None)
    global _global_cost_context
    with _global_cost_context_lock:
        previous_global = _global_cost_context
        _global_cost_context = context
    _cost_context_local.context = context
    try:
        yield context
    finally:
        if previous_local is None:
            if hasattr(_cost_context_local, "context"):
                delattr(_cost_context_local, "context")
        else:
            _cost_context_local.context = previous_local
        with _global_cost_context_lock:
            _global_cost_context = previous_global


def _exception_summary(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return " ".join(text.split())[:240]


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (LLMCallTimeoutError, LLMRetryBudgetExceeded, LLMMaxRetriesExceeded)):
        return False
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    text = f"{exc.__class__.__name__} {exc}".lower()
    if "insufficient_quota" in text or "insufficient quota" in text:
        return False
    return any(
        marker in text
        for marker in (
            "status=429",
            "status 429",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "connection reset",
            "server error",
            "status=500",
            "status=502",
            "status=503",
            "status=504",
        )
    )


def _run_with_timeout(call: Callable[[], Any], timeout_seconds: float) -> Any:
    """Wait for a definitive transport result.

    Python threads cannot cancel an in-flight HTTP request.  The former daemon
    wrapper returned ``LLMCallTimeoutError`` while the paid request continued
    in the background, allowing retries, fallback publication, and budget
    reconciliation to race a still-live provider call.  Provider transports
    own their native request timeouts; this layer must not abandon them.

    ``timeout_seconds`` remains part of the compatibility surface and is used
    by callers when configuring the transport itself.
    """

    _ = timeout_seconds
    return call()


def call_with_llm_retry_guard(
    *,
    provider_name: str,
    schema_name: str | None,
    call: Callable[[], Any],
    timeout_seconds: float | None = None,
    max_retries: int = DEFAULT_LLM_MAX_RETRIES_PER_CALL,
    backoff_seconds: tuple[float, ...] = DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
    call_deadline_seconds: float = DEFAULT_LLM_CALL_DEADLINE_SECONDS,
    sleep_fn: Callable[[float], None] | None = None,
) -> Any:
    provider = str(provider_name or "unknown").lower()
    schema_label = str(schema_name or "structured_output")
    timeout = float(timeout_seconds or DEFAULT_LLM_CALL_TIMEOUT_SECONDS)
    deadline = time.monotonic() + max(0.001, float(call_deadline_seconds))
    retries_allowed = max(0, int(max_retries))
    cost_context = current_cost_context()
    if cost_context is not None and cost_context.strict_first_call:
        # A strict whole-run authorization is a physical-call ceiling, not a
        # logical-call ceiling.  A transparent provider retry can spend again
        # before the caller has a chance to reconcile the first attempt, so a
        # strict run must stop after the first physical attempt.
        retries_allowed = 0
    sleeper = sleep_fn or time.sleep
    last_retryable_exc: BaseException | None = None

    for attempt_index in range(retries_allowed + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LLMCallTimeoutError(
                f"LLM call deadline exceeded after {call_deadline_seconds:.2f}s for {provider}:{schema_label}"
            )
        require_llm_physical_attempt_authorization(
            provider=provider,
            schema_name=schema_label,
            attempt=attempt_index + 1,
        )
        try:
            return _run_with_timeout(call, min(timeout, remaining))
        except BaseException as exc:  # noqa: BLE001
            retryable = _is_retryable_exception(exc)
            will_retry = (
                retryable
                and not isinstance(exc, (LLMCallTimeoutError, LLMRetryBudgetExceeded))
                and attempt_index < retries_allowed
            )
            _notify_failed_attempt(
                provider=provider,
                schema_name=schema_label,
                attempt=attempt_index + 1,
                exc=exc,
                retryable=retryable,
                will_retry=will_retry,
            )
            if isinstance(exc, (LLMCallTimeoutError, LLMRetryBudgetExceeded)):
                raise
            if not retryable:
                raise
            last_retryable_exc = exc
            if attempt_index >= retries_allowed:
                raise LLMMaxRetriesExceeded(
                    f"{provider}:{schema_label} exceeded max retries ({retries_allowed}) "
                    f"after {attempt_index + 1} attempts: {_exception_summary(exc)}"
                ) from exc
            context = current_retry_context()
            if context is not None:
                context.record_retry(
                    provider=provider,
                    schema_name=schema_label,
                    attempt=attempt_index + 1,
                    reason=_exception_summary(exc),
                )
            sleep_for = (
                backoff_seconds[min(attempt_index, len(backoff_seconds) - 1)]
                if backoff_seconds
                else 0.0
            )
            remaining_after_sleep = deadline - time.monotonic()
            if sleep_for > 0 and remaining_after_sleep > 0:
                sleeper(min(float(sleep_for), max(0.0, remaining_after_sleep)))

    raise LLMMaxRetriesExceeded(
        f"{provider}:{schema_label} exceeded max retries ({retries_allowed}): "
        f"{_exception_summary(last_retryable_exc) if last_retryable_exc else 'unknown retryable failure'}"
    )
