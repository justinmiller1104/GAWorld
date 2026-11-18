"""Phase 1.2 RED — canary integration-path tests (inference boundary coverage).

Pins the smallest safe opt-in integration between the MCF canary budget
controller (``benchmark/mcf/canary.py``) and the REAL simulation inference
path. Today the canary is an isolated controller: nothing in
``gaworld.llm.providers.call_llm`` (or any other inference entry point)
routes through it, so a real run has zero budget protection.

The contract under test:

* ``benchmark.mcf.canary.wrap_call_llm(base_call_llm, budget)`` returns a
  callable with the same signature that routes EVERY call through
  ``budget.check`` → base → ``budget.record`` — including retries (a retry
  after a provider error is a new call against the budget).
* Default-off: without the wrapper, ``gaworld.llm.providers.call_llm``
  behaves exactly as before — the budget is never consulted.
* Enabling refuses without a verified pricing table.
* Budget exhaustion mid-run raises ``CanaryExceeded`` carrying reason +
  usage, and the usage snapshot survives the exception.
* Provider errors count toward the error-rate kill.
* Dry-run mode issues ZERO real calls.
* No global mutable state leaks between wrapped and unwrapped runs.

Zero real inference: every base callable is a mock.
"""

from __future__ import annotations

import inspect

import pytest

from benchmark.mcf.canary import (
    CanaryBudget,
    CanaryExceeded,
    CanaryLimits,
    wrap_call_llm,
)

PRICING = {"gpt-mock": {"input": 1.0, "output": 1.0}}


def make_limits(**over):
    return CanaryLimits(pricing=dict(PRICING), **over)


class FakeResult(dict):
    pass


def make_base(ok=True, tokens=(300, 200), result=None, error=None):
    """A mock stand-in for ``gaworld.llm.providers.call_llm``."""
    calls: list[dict] = []

    def base(prompt, task=None, agent_id=None, provider=None, system=None,
             temperature=None, allow_fallback=True, images=None, max_tokens=None):
        calls.append({"prompt": prompt, "task": task, "agent_id": agent_id})
        if error is not None:
            raise error
        return result if result is not None else FakeResult(
            text="ok", tokens_in=tokens[0], tokens_out=tokens[1]
        )

    base.calls = calls
    base.ok = ok
    return base


# ---------------------------------------------------------------------------
# 1. The wrapper exists and routes every call through check → base → record
# ---------------------------------------------------------------------------

class TestWrapCallLLMExists:
    def test_wrap_call_llm_is_exported(self):
        assert callable(wrap_call_llm)

    def test_signature_takes_base_and_budget(self):
        params = inspect.signature(wrap_call_llm).parameters
        names = set(params)
        assert any("base" in n for n in names), f"no base-callable param in {names}"
        assert any("budget" in n or "limit" in n for n in names), (
            f"no budget/limits param in {names}"
        )

    def test_every_call_routes_through_budget(self):
        base = make_base()
        budget = CanaryBudget(make_limits())
        wrapped = wrap_call_llm(base, budget)
        wrapped("hello", task="perception", agent_id=7)
        assert len(base.calls) == 1
        usage = budget.usage_snapshot()
        assert usage["calls"] == 1, "call not recorded against the budget"
        assert usage["tokens"] == 500, "tokens not recorded against the budget"

    def test_wrapped_call_preserves_result_and_args(self):
        base = make_base(result=FakeResult(text="answer", tokens_in=10, tokens_out=5))
        budget = CanaryBudget(make_limits())
        wrapped = wrap_call_llm(base, budget)
        out = wrapped("hi", task="action", agent_id=3)
        assert out["text"] == "answer"
        assert base.calls[0]["task"] == "action"
        assert base.calls[0]["agent_id"] == 3


# ---------------------------------------------------------------------------
# 2. Retries count as additional calls against the budget
# ---------------------------------------------------------------------------

class TestRetriesCountTowardBudget:
    def test_retry_after_provider_error_is_a_new_budgeted_call(self):
        flaky = make_base(error=RuntimeError("transient"))
        budget = CanaryBudget(make_limits())
        wrapped = wrap_call_llm(flaky, budget)
        with pytest.raises(RuntimeError):
            wrapped("a")  # first call fails
        # the sim's retry loop calls again through the wrapped path
        with pytest.raises(RuntimeError):
            wrapped("a")
        usage = budget.usage_snapshot()
        assert usage["calls"] == 2, "retries must each count as budgeted calls"
        assert usage["errors"] == 2

    def test_retry_sequence_recorded_then_success(self):
        results = [RuntimeError("boom"), None]

        def base(prompt, **kwargs):
            r = results.pop(0)
            if r is not None:
                raise r
            return FakeResult(text="ok", tokens_in=100, tokens_out=50)

        base.calls = []
        budget = CanaryBudget(make_limits())
        wrapped = wrap_call_llm(base, budget)
        with pytest.raises(RuntimeError):
            wrapped("a")
        wrapped("a")
        usage = budget.usage_snapshot()
        assert usage["calls"] == 2
        assert usage["errors"] == 1
        assert usage["tokens"] == 150  # only the successful call carried tokens


# ---------------------------------------------------------------------------
# 3. Default-off: unwrapped path is untouched
# ---------------------------------------------------------------------------

class TestDefaultOff:
    def test_call_llm_works_without_canary_and_does_not_import_it(self):
        import gaworld.llm.providers as providers
        import inspect

        # Module-graph assertion: the production module must not import the
        # canary (or any benchmark module) — checked at source level, since
        # sys.modules absence is not observable once any test in the session
        # has imported the canary deliberately.
        source = inspect.getsource(providers)
        assert "benchmark" not in source, (
            "production providers module pulled in the canary at import time"
        )
        # And the plain public entry point still exists with its signature.
        sig = inspect.signature(providers.call_llm)
        assert "prompt" in sig.parameters

    def test_unwrapped_base_is_never_consulting_a_budget(self):
        base = make_base()
        wrapped = base  # default-off: no wrapper in the path at all
        wrapped("prompt", task="day_routine")
        assert len(base.calls) == 1  # plain passthrough, no budget machinery


# ---------------------------------------------------------------------------
# 4. Missing pricing → refuses to enable
# ---------------------------------------------------------------------------

class TestRefusesWithoutPricing:
    def test_limits_without_pricing_raise(self):
        with pytest.raises(ValueError):
            CanaryLimits(pricing={})

    def test_enabling_with_incomplete_pricing_refuses(self):
        with pytest.raises(ValueError):
            CanaryLimits(pricing={"gpt-mock": {"input": 1.0}})  # no 'output'

    def test_wrap_refuses_budget_without_pricing(self):
        with pytest.raises(ValueError):
            CanaryBudget(CanaryLimits(pricing=None))


# ---------------------------------------------------------------------------
# 5. Budget exhaustion mid-run, including during retries / in-flight
# ---------------------------------------------------------------------------

class TestBudgetExhaustionMidRun:
    def test_call_budget_exhaustion_raises_canary_exceeded(self):
        base = make_base(tokens=(100, 100))
        budget = CanaryBudget(make_limits(max_calls=3))
        wrapped = wrap_call_llm(base, budget)
        for _ in range(3):
            wrapped("p")
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p")
        assert exc.value.reason
        assert exc.value.usage["calls"] >= 3

    def test_exhaustion_during_retry_stops_cleanly(self):
        # every call fails; max_calls small so the kill lands mid retry loop
        flaky = make_base(error=RuntimeError("transient"))
        budget = CanaryBudget(make_limits(max_calls=2))
        wrapped = wrap_call_llm(flaky, budget)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                wrapped("p")
        with pytest.raises(CanaryExceeded):
            wrapped("p")  # the third attempt (a retry) must hit the wall

    def test_in_flight_reservation_counts_toward_token_ceiling(self):
        base = make_base(tokens=(0, 0))
        budget = CanaryBudget(make_limits(max_tokens=600))
        budget.reserve("outstanding-call", est_tokens=500)
        wrapped = wrap_call_llm(base, budget)
        # projected 500 (in-flight) + 400 (this call) > 600 → refused
        with pytest.raises((CanaryExceeded, PermissionError)):
            wrapped("p")
        usage = budget.usage_snapshot()
        assert usage["calls"] == 0, "refused call must not be recorded"


# ---------------------------------------------------------------------------
# 6. Provider errors count toward the error-rate kill
# ---------------------------------------------------------------------------

class TestErrorRateKill:
    def test_provider_errors_trigger_error_rate_kill(self):
        flaky = make_base(error=RuntimeError("provider down"))
        limits = make_limits(max_calls=10_000, min_error_sample=5, max_error_rate=0.5)
        budget = CanaryBudget(limits)
        wrapped = wrap_call_llm(flaky, budget)
        killed = False
        for _ in range(10):
            try:
                wrapped("p")
            except CanaryExceeded as exc:
                assert "error rate" in exc.reason
                killed = True
                break
            except RuntimeError:
                continue
        assert killed, "error-rate kill never fired despite 10 consecutive provider errors"


# ---------------------------------------------------------------------------
# 7. Usage snapshot preserved on termination
# ---------------------------------------------------------------------------

class TestUsagePreservedOnTermination:
    def test_usage_survives_canary_exceeded(self):
        base = make_base(tokens=(300, 200))
        budget = CanaryBudget(make_limits(max_calls=2))
        wrapped = wrap_call_llm(base, budget)
        wrapped("p")
        wrapped("p")
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p")
        usage = exc.value.usage
        assert usage["calls"] == 2
        assert usage["tokens"] == 1000
        assert "cost_usd" in usage
        assert budget.usage_snapshot() == usage

    def test_snapshot_is_a_copy_not_live_internal_state(self):
        budget = CanaryBudget(make_limits())
        snap = budget.usage_snapshot()
        base = make_base()
        wrapped = wrap_call_llm(base, budget)
        wrapped("p")
        assert snap["calls"] == 0, "snapshot mutated after the fact — not a copy"


# ---------------------------------------------------------------------------
# 8. Dry-run: zero real calls through the wrapped path
# ---------------------------------------------------------------------------

class TestDryRunZeroCalls:
    def test_dry_run_wrapped_path_issues_no_base_calls(self):
        base = make_base()
        budget = CanaryBudget(make_limits())
        wrapped = wrap_call_llm(base, budget, dry_run=True)
        for _ in range(5):
            wrapped("p")
        assert len(base.calls) == 0, "dry run must issue ZERO real provider calls"

    def test_dry_run_still_tracks_budget(self):
        base = make_base()
        budget = CanaryBudget(make_limits(max_calls=2))
        wrapped = wrap_call_llm(base, budget, dry_run=True)
        stopped = None
        for _ in range(5):
            try:
                wrapped("p")
            except CanaryExceeded as exc:
                stopped = exc.reason
                break
        assert stopped, "dry run never enforced the call budget"
        assert budget.usage_snapshot()["calls"] >= 2


# ---------------------------------------------------------------------------
# 9. No global mutable state leak between wrapped / unwrapped runs
# ---------------------------------------------------------------------------

class TestNoGlobalStateLeak:
    def test_two_budgets_are_independent(self):
        b1 = CanaryBudget(make_limits())
        b2 = CanaryBudget(make_limits())
        base = make_base()
        wrap_call_llm(base, b1)("p")
        assert b1.usage_snapshot()["calls"] == 1
        assert b2.usage_snapshot()["calls"] == 0, "second budget saw the first run's calls"

    def test_module_has_no_mutable_run_state(self):
        import benchmark.mcf.canary as canary_mod

        for name, val in vars(canary_mod).items():
            if name.startswith("_"):
                continue
            assert not isinstance(val, (dict, list)), (
                f"module-level mutable global {name!r} risks leaking run state"
            )
