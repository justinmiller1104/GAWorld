"""Canary budget controller for MCF cost-capped LLM runs.

Enforces hard limits (calls / tokens / cost / wall-clock / error rate) for a
bounded canary run so a real provider can never run away. The canary's
measured per-call latency supersedes every assumed latency in the audit docs.

Pure and stdlib-only; no global state; never imported by production paths.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any


class CanaryExceeded(Exception):
    """Raised when the canary budget is exhausted; carries the stop reason."""

    def __init__(self, reason: str, usage: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.usage = usage


@dataclass(frozen=True)
class CanaryLimits:
    """Hard caps for a canary run. A verified pricing table is mandatory."""

    max_calls: int = 400
    max_tokens: int = 250_000
    max_cost_usd: float = 1.00
    max_elapsed_seconds: float = 1800
    max_error_rate: float = 0.05
    min_error_sample: int = 20
    pricing: dict[str, dict[str, float]] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.pricing or not isinstance(self.pricing, dict):
            raise ValueError("pricing table is required (model -> {'input': $/1M, 'output': $/1M})")
        for model, rates in self.pricing.items():
            if not isinstance(rates, dict) or "input" not in rates or "output" not in rates:
                raise ValueError(f"pricing for {model!r} must define both 'input' and 'output' rates")

    def token_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        rates = self.pricing[model]
        return (tokens_in * rates["input"] + tokens_out * rates["output"]) / 1_000_000


@dataclass
class Decision:
    """Result of a pre-call budget check."""

    allowed: bool
    reason: str = ""


@dataclass
class CanaryBudget:
    """Mutable usage ledger for one canary run; instances are independent."""

    limits: CanaryLimits

    def __post_init__(self) -> None:
        self._tokens = 0
        self._calls = 0
        self._errors = 0
        self._cost = 0.0
        self._in_flight: dict[str, int] = {}
        self._frozen: dict[str, Any] | None = None
        self._started = time.monotonic()

    def freeze_usage(self) -> None:
        """Pin the current usage as final (post-kill manifest stability)."""
        self._frozen = self._snapshot()

    # -- pre-call -----------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        rate = self._errors / self._calls if self._calls else 0.0
        return {
            "calls": self._calls,
            "tokens": self._tokens,
            "errors": self._errors,
            "error_rate": rate,
            "elapsed": time.monotonic() - self._started,
            "cost_usd": self._cost,
        }

    def check(self, call_number: int, est_tokens: int, est_cost: float) -> Decision:
        usage = self._snapshot()
        if self._calls >= self.limits.max_calls:
            # The ledger has already consumed the call budget: the run is
            # over, not merely discouraged — surface it as an exception.
            self._frozen = usage
            raise CanaryExceeded(
                f"call budget exhausted: {self._calls} calls >= max_calls={self.limits.max_calls}", usage
            )
        if call_number > self.limits.max_calls:
            return Decision(
                False, f"call budget exceeded: call {call_number} > max_calls={self.limits.max_calls}"
            )
        # In-flight reservations count toward the token ceiling: recorded
        # usage alone understates exposure while calls are outstanding.
        projected = self._tokens + sum(self._in_flight.values()) + est_tokens
        if projected > self.limits.max_tokens:
            return Decision(
                False, f"token budget exceeded: projected {projected} > max_tokens={self.limits.max_tokens}"
            )
        if self._cost + est_cost > self.limits.max_cost_usd:
            return Decision(
                False,
                f"cost budget exceeded: ${self._cost + est_cost:.6f} > max_cost_usd={self.limits.max_cost_usd}",
            )
        return Decision(True)

    def reserve(self, call_id: str, est_tokens: int) -> None:
        self._in_flight[call_id] = est_tokens

    def release(self, call_id: str) -> None:
        self._in_flight.pop(call_id, None)

    # -- post-call ----------------------------------------------------------

    def _raise_if_exceeded(self) -> None:
        usage = self._snapshot()
        if self._calls > self.limits.max_calls:
            self._frozen = usage
            raise CanaryExceeded(
                f"call budget exceeded: {self._calls} > max_calls={self.limits.max_calls}", usage
            )
        if self._tokens > self.limits.max_tokens:
            self._frozen = usage
            raise CanaryExceeded(
                f"token budget exceeded: {self._tokens} > max_tokens={self.limits.max_tokens}", usage
            )
        if self._cost > self.limits.max_cost_usd:
            self._frozen = usage
            raise CanaryExceeded(
                f"cost budget exceeded: ${self._cost:.6f} > max_cost_usd={self.limits.max_cost_usd}", usage
            )
        if usage["elapsed"] > self.limits.max_elapsed_seconds:
            self._frozen = usage
            raise CanaryExceeded(
                f"elapsed budget exceeded: {usage['elapsed']:.1f}s > max_elapsed_seconds={self.limits.max_elapsed_seconds}",
                usage,
            )
        if self._calls >= self.limits.min_error_sample:
            rate = self._errors / self._calls
            # Strictly greater: exactly-at-threshold is not a kill.
            if rate > self.limits.max_error_rate:
                self._frozen = usage
                raise CanaryExceeded(
                    f"error rate exceeded: {rate:.3f} > max_error_rate={self.limits.max_error_rate} at n={self._calls}",
                    usage,
                )

    def record(self, call_id: str, ok: bool, tokens_in: int, tokens_out: int, latency_s: float) -> None:
        self._calls += 1
        self._tokens += tokens_in + tokens_out
        model = call_id if call_id in self.limits.pricing else _default_model(self.limits)
        self._cost += self.limits.token_cost(model, tokens_in, tokens_out)
        if not ok:
            self._errors += 1
        self._in_flight.pop(call_id, None)
        self._raise_if_exceeded()

    def usage_snapshot(self) -> dict[str, Any]:
        if self._frozen is not None:
            return dict(self._frozen)
        return self._snapshot()


def wrap_call_llm(base, budget: CanaryBudget, *, dry_run: bool = False):
    """Wrap ``base`` (a ``call_llm``-compatible callable) so every dispatch is
    budgeted: pre-call check → reserve → base → record. A failed provider
    dispatch still consumes a slot; if that trips an error ceiling, the
    budget's kill (``CanaryExceeded``) surfaces instead of the provider error.
    Dry-run mode issues ZERO base calls.

    When ``base`` declares no token counts, a conservative 400-token/call
    estimate is used for the pre-call projection — an unestimable call size
    of exactly zero would let outstanding in-flight reservations combine with
    real traffic past the token ceiling (the projection must stay strictly
    optimistic-free). Actual recorded usage always comes from the result.
    """
    default_est = 400
    est_in = int(getattr(base, "tokens_in", default_est) or default_est)
    est_out = int(getattr(base, "tokens_out", default_est) or default_est)
    model = _default_model(budget.limits)
    counter = itertools.count(1)

    def _result_tokens(result) -> tuple[int, int]:
        source = result if isinstance(result, dict) else getattr(result, "__dict__", {})
        return int(source.get("tokens_in", 0) or 0), int(source.get("tokens_out", 0) or 0)

    def wrapped(prompt, task=None, agent_id=None, **kwargs):
        call_id = f"{task or 'call'}-{next(counter)}"
        est = est_in + est_out
        est_cost = budget.limits.token_cost(model, est_in, est_out)
        usage = budget.usage_snapshot()
        decision = budget.check(call_number=usage["calls"] + 1, est_tokens=est, est_cost=est_cost)
        if not decision.allowed:
            raise CanaryExceeded(decision.reason, budget.usage_snapshot())
        budget.reserve(call_id, est_tokens=est)
        try:
            if dry_run:
                budget.record(call_id, ok=True, tokens_in=est_in, tokens_out=est_out, latency_s=0.0)
                return f"[canary-dry-run] {prompt[:80]}"
            result = base(prompt, task=task, agent_id=agent_id, **kwargs)
        except CanaryExceeded:
            budget.release(call_id)
            raise
        except Exception:
            budget.release(call_id)
            # The failed dispatch is itself budgeted; a ceiling it trips
            # (e.g. the error-rate kill) takes precedence over the provider
            # error so the caller sees the termination reason.
            budget.record(call_id, ok=False, tokens_in=0, tokens_out=0, latency_s=0.0)
            raise
        tokens_in, tokens_out = _result_tokens(result)
        budget.record(call_id, ok=True, tokens_in=tokens_in, tokens_out=tokens_out, latency_s=0.0)
        return result

    return wrapped


def _default_model(limits: CanaryLimits) -> str:
    return next(iter(limits.pricing))


def run_canary(
    config: dict[str, Any],
    provider: Any,
    limits: CanaryLimits,
    dry_run: bool = True,
    authorize: bool = False,
) -> dict[str, Any]:
    """Run a bounded canary against ``provider``.

    Dry run issues ZERO provider calls but keeps full synthetic accounting
    from the provider's declared token counts. Live mode requires an
    explicit ``authorize=True`` gate.
    """
    if not dry_run and not authorize:
        raise PermissionError("live canary run requires authorize=True")

    model = config.get("model") or _default_model(limits)
    budget = CanaryBudget(limits)
    calls: int = config["calls"]
    reason: str | None = None

    if dry_run:
        est_in = getattr(provider, "tokens_in", 0)
        est_out = getattr(provider, "tokens_out", 0)
        for i in range(calls):
            try:
                # Pre-check so the refused call is never recorded.
                decision = budget.check(call_number=i + 1, est_tokens=est_in + est_out, est_cost=0.0)
                if not decision.allowed:
                    reason = decision.reason
                    break
                budget.record(f"dry-{i}", ok=True, tokens_in=est_in, tokens_out=est_out, latency_s=0.0)
            except CanaryExceeded as exc:
                reason = exc.reason
                break
    else:
        for i in range(calls):
            call_id = f"live-{i}"
            try:
                decision = budget.check(call_number=i + 1, est_tokens=0, est_cost=0.0)
            except CanaryExceeded as exc:
                reason = exc.reason
                break
            if not decision.allowed:
                reason = decision.reason
                break
            budget.reserve(call_id, est_tokens=0)
            try:
                result = provider.call(prompt=str(config.get("prompt", "")))
                budget.release(call_id)
                budget.record(
                    call_id,
                    ok=True,
                    tokens_in=result.get("tokens_in", 0),
                    tokens_out=result.get("tokens_out", 0),
                    latency_s=0.0,
                )
            except CanaryExceeded as exc:
                budget.release(call_id)
                reason = exc.reason
                break
            except Exception as exc:  # provider failure counts as a canary error
                budget.release(call_id)
                budget.record(call_id, ok=False, tokens_in=0, tokens_out=0, latency_s=0.0)
                reason = f"provider error: {exc}"
                break

    return {
        "dry_run": dry_run,
        "model": model,
        "reason": reason,
        "usage": budget.usage_snapshot(),
    }
