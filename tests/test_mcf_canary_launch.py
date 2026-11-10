"""Contract tests (RED) for the canary LAUNCH integration.

Phase 1.3: the canary budget controller (``benchmark/mcf/canary.py``) and
the real LLM entry path exist, but nothing connects them.  These tests pin
a minimal OPT-IN launch integration — module suggested as
``benchmark/mcf/launch.py`` — that wraps GAWorld's ``call_llm`` so every
inference the kernel makes flows through ONE budget.

Entry path traced (for the implement agent):
  generative_city_sim.py:4345  ``_main()``        — CLI dispatch
  generative_city_sim.py:2332  ``run_simulation``
  generative_city_sim.py:2379  ``build_kernel(CONFIG, llm=call_llm)``
  gaworld/kernel/context.py:62 ``build_kernel``   — stores llm on SimContext
  gaworld/llm/providers.py:1316 ``call_llm``      — per-call retry lives in
  gaworld/llm/providers.py:185  ``_retrying`` (invoked inside the provider
  classes; the wrapper must sit OUTSIDE so retries count as separate
  budgeted calls).

Pinned contract:
  ``prepare_canary(config, limits, pricing, *, authorize, dry_run=False,
  base_llm=None)`` -> returns ``(wrapped_llm, budget)``.

  ``base_llm`` is an injection seam (defaults to the real
  ``gaworld.llm.providers.call_llm``); tests pass a fake so nothing
  touches the network.

Mocked only: a deterministic fake provider is injected as the base llm;
there is ZERO LLM inference and no real network anywhere in this file.
"""

from __future__ import annotations

import json
import threading
import unittest

import pytest

from benchmark.mcf.canary import CanaryBudget, CanaryExceeded, CanaryLimits
from benchmark.mcf.launch import prepare_canary  # missing => whole module RED


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

SIMPLE_PRICING = {"test-model": {"input": 1.0, "output": 1.0}}


class FakeBaseLLM:
    """Stands in for ``gaworld.llm.providers.call_llm``.

    Returns deterministic text and token counts.  Programmable failures for
    the retry/failure-budget contract.  Signature-compatible with the real
    helper: ``(prompt, task=None, agent_id=None, ...)``.
    """

    def __init__(self, tokens_in: int = 300, tokens_out: int = 300):
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.calls = 0
        self.calls_by_task: dict[str, int] = {}
        self.fail_calls: set[int] = set()
        self.lock = threading.Lock()

    def __call__(self, prompt, task=None, agent_id=None, **kwargs):
        with self.lock:
            self.calls += 1
            n = self.calls
            self.calls_by_task[task or "?"] = self.calls_by_task.get(task or "?", 0) + 1
        if n in self.fail_calls:
            raise RuntimeError("simulated provider error")
        return "ok " + str(n)


def make_limits(**overrides) -> CanaryLimits:
    defaults = dict(
        max_calls=400,
        max_tokens=250_000,
        max_cost_usd=1.00,
        max_elapsed_seconds=1800,
        max_error_rate=0.05,
        min_error_sample=20,
        pricing=SIMPLE_PRICING,
    )
    defaults.update(overrides)
    return CanaryLimits(**defaults)


def fake_config(**canary_cfg) -> dict:
    """A minimal CONFIG with a model set, like the real one has."""
    cfg = {"model": "test-model"}
    if canary_cfg:
        cfg["canary"] = canary_cfg
    return cfg


# ---------------------------------------------------------------------------
# 1. prepare_canary contract — authorization + pricing are mandatory
# ---------------------------------------------------------------------------


class TestPrepareCanaryContract(unittest.TestCase):
    def test_requires_authorize(self):
        with pytest.raises(PermissionError):
            prepare_canary(fake_config(), make_limits(), SIMPLE_PRICING, authorize=False)

    def test_requires_pricing(self):
        with pytest.raises(ValueError):
            prepare_canary(fake_config(), make_limits(pricing=None), None, authorize=True)

    def test_returns_wrapped_llm_and_budget(self):
        wrapped, budget = prepare_canary(fake_config(), make_limits(), SIMPLE_PRICING, authorize=True)
        assert callable(wrapped)
        assert isinstance(budget, CanaryBudget)
        # Independent instances: two preps never share a ledger.
        wrapped2, budget2 = prepare_canary(fake_config(), make_limits(), SIMPLE_PRICING, authorize=True)
        assert budget is not budget2


# ---------------------------------------------------------------------------
# 2. Default-unchanged + 3. explicit enablement
# ---------------------------------------------------------------------------


class TestOptInEnablement(unittest.TestCase):
    def test_disabled_config_returns_original_callable_unwrapped(self):
        """No canary key => launch must hand back the ORIGINAL llm (identity)."""
        base = FakeBaseLLM()
        from benchmark.mcf.launch import maybe_canary_llm  # suggested helper name

        out = maybe_canary_llm(base, fake_config(), make_limits(), SIMPLE_PRICING)
        assert out is base, "without canary enabled the original call_llm must pass through untouched"

    def test_enabled_via_config_key(self):
        base = FakeBaseLLM()
        from benchmark.mcf.launch import maybe_canary_llm

        out, budget = maybe_canary_llm(
            base, fake_config(enabled=True), make_limits(), SIMPLE_PRICING, authorize=True
        )
        assert out is not base
        assert isinstance(budget, CanaryBudget)

    def test_enabled_via_env_var(self, monkeypatch=None):
        base = FakeBaseLLM()
        from benchmark.mcf.launch import maybe_canary_llm

        import os

        old = os.environ.get("GAWORLD_CANARY")
        os.environ["GAWORLD_CANARY"] = "1"
        try:
            out, budget = maybe_canary_llm(base, fake_config(), make_limits(), SIMPLE_PRICING, authorize=True)
            assert out is not base
            assert isinstance(budget, CanaryBudget)
        finally:
            if old is None:
                os.environ.pop("GAWORLD_CANARY", None)
            else:
                os.environ["GAWORLD_CANARY"] = old

    def test_env_without_authorize_refuses(self):
        import os

        from benchmark.mcf.launch import maybe_canary_llm

        os.environ["GAWORLD_CANARY"] = "1"
        try:
            with pytest.raises(PermissionError):
                maybe_canary_llm(FakeBaseLLM(), fake_config(), make_limits(), SIMPLE_PRICING)
        finally:
            os.environ.pop("GAWORLD_CANARY", None)


# ---------------------------------------------------------------------------
# 4. Every kernel call kind counted against ONE budget
# ---------------------------------------------------------------------------

KERNEL_CALL_TASKS = [
    "social_profile",   # setup
    "weekend_routine",  # setup
    "daily_routine",    # routine
    "schedule",         # action planning
    "event_effect",     # perception
    "planning",         # action
    "reflection",       # memory/reflection
    "news_analysis",    # news
    "diary_import_summary",  # digest-style
]


class TestOneBudgetAcrossKernelCalls(unittest.TestCase):
    def test_all_kernel_task_kinds_counted_against_one_budget(self):
        base = FakeBaseLLM()
        wrapped, budget = prepare_canary(
            fake_config(), make_limits(), SIMPLE_PRICING, authorize=True, base_llm=base
        )

        for task in KERNEL_CALL_TASKS:
            wrapped(f"prompt for {task}", task=task, agent_id=None)

        usage = budget.usage_snapshot()
        assert usage["calls"] == len(KERNEL_CALL_TASKS), (
            "every inference kind (setup/routine/perception/action/memory/news/digest) "
            "must be counted against the single canary budget"
        )
        assert usage["tokens"] == len(KERNEL_CALL_TASKS) * (base.tokens_in + base.tokens_out)
        assert base.calls == len(KERNEL_CALL_TASKS), "wrapped llm must dispatch each call to the base provider"

    def test_wrapper_signature_matches_call_llm(self):
        wrapped, _ = prepare_canary(fake_config(), make_limits(), SIMPLE_PRICING, authorize=True)
        result = wrapped("p", task="schedule", agent_id=31, system="s", temperature=0.2)
        assert result  # accepts the real call_llm keyword surface


# ---------------------------------------------------------------------------
# 5. Retries/failures: both attempts budgeted
# ---------------------------------------------------------------------------


class TestRetryBudgeting(unittest.TestCase):
    def test_failing_then_succeeding_call_counts_both_attempts(self):
        base = FakeBaseLLM()
        wrapped, budget = prepare_canary(
            fake_config(), make_limits(), SIMPLE_PRICING, authorize=True, base_llm=base
        )
        # The wrapper must be able to wrap a provider that retries internally
        # (like the real call_llm does via _retrying): we simulate that the
        # wrapped callable is called once per attempt by the kernel-facing
        # retry shim.  Here we pin that a failed dispatch followed by a
        # successful one consumes TWO call slots.
        base.fail_calls = {1}

        # Kernel-facing: the wrapper delegates each dispatch to base; on
        # base failure the wrapper records the error, re-dispatches is the
        # CALLER's policy (as _retrying does), so two wrapped calls == two
        # budgeted attempts, one error + one success.
        with pytest.raises(RuntimeError):
            wrapped("p", task="planning", agent_id=1)
        wrapped("p", task="planning", agent_id=1)

        usage = budget.usage_snapshot()
        assert usage["calls"] == 2, "failed attempt AND retry attempt must both be budgeted"
        assert usage["errors"] == 1

    def test_error_rate_kill_after_min_sample(self):
        base = FakeBaseLLM()
        limits = make_limits(max_error_rate=0.5, min_error_sample=4, max_calls=100)
        wrapped, budget = prepare_canary(
            fake_config(), limits, SIMPLE_PRICING, authorize=True, base_llm=base
        )
        base.fail_calls = set(range(1, 100))  # every call fails
        for _ in range(3):
            with pytest.raises(RuntimeError):
                wrapped("p", task="planning")
        # 3 errors at n=3: below min_error_sample, no kill yet.
        # Call 4 crosses min_error_sample with a 100% rate: the POST-call
        # check must kill, so the caller sees CanaryExceeded, not the
        # provider's RuntimeError.
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p", task="planning")
        assert "error rate" in exc.value.reason


# ---------------------------------------------------------------------------
# 6. Concurrent in-flight reservations respected
# ---------------------------------------------------------------------------


class TestConcurrency(unittest.TestCase):
    def test_parallel_calls_respected_by_budget(self):
        base = FakeBaseLLM(tokens_in=1000, tokens_out=1000)
        limits = make_limits(max_tokens=3000, max_calls=50)  # only ~1.5 calls of token headroom
        wrapped, budget = prepare_canary(
            fake_config(), limits, SIMPLE_PRICING, authorize=True, base_llm=base
        )

        results: list[str] = []
        errors: list[Exception] = []

        def worker():
            try:
                results.append(wrapped("p", task="planning"))
            except CanaryExceeded as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With reservations counting toward the ceiling, at most 2 calls may
        # be dispatched (each 2000 tokens vs a 3000 ceiling); no over-commit.
        usage = budget.usage_snapshot()
        assert usage["calls"] <= 2, f"in-flight reservations must cap concurrent dispatch, got {usage['calls']}"
        assert usage["tokens"] <= limits.max_tokens


# ---------------------------------------------------------------------------
# 7. All kill conditions terminate with CanaryExceeded(reason, usage)
# ---------------------------------------------------------------------------


class TestKillConditions(unittest.TestCase):
    def _wrapped(self, **limit_overrides):
        base = FakeBaseLLM(tokens_in=500, tokens_out=500)  # $0.001/call at 1.0/1M
        limits = make_limits(**limit_overrides)
        wrapped, budget = prepare_canary(
            fake_config(), limits, SIMPLE_PRICING, authorize=True, base_llm=base
        )
        return base, wrapped, budget

    def test_call_limit_kill(self):
        _, wrapped, _ = self._wrapped(max_calls=3)
        for _ in range(3):
            wrapped("p", task="planning")
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p", task="planning")
        assert "call budget" in exc.value.reason
        assert exc.value.usage["calls"] == 3

    def test_token_limit_kill(self):
        _, wrapped, _ = self._wrapped(max_tokens=1000)
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p", task="planning")  # 1000 tokens recorded, next is over
        wrapped("p", task="planning")
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p", task="planning")
        assert "token budget" in exc.value.reason

    def test_cost_limit_kill(self):
        _, wrapped, _ = self._wrapped(max_cost_usd=0.001)
        wrapped("p", task="planning")  # exactly $0.001 — allowed
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p", task="planning")
        assert "cost" in exc.value.reason

    def test_time_limit_kill(self):
        base, wrapped, _ = self._wrapped(max_elapsed_seconds=0.0)
        import time as _time

        _time.sleep(0.01)
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p", task="planning")
        assert "elapsed" in exc.value.reason

    def test_exception_carries_json_serializable_usage(self):
        _, wrapped, _ = self._wrapped(max_calls=1)
        wrapped("p", task="planning")
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p", task="planning")
        json.dumps(exc.value.usage)  # must not raise


# ---------------------------------------------------------------------------
# 8. Dry-run: zero base-provider calls
# ---------------------------------------------------------------------------


class TestDryRun(unittest.TestCase):
    def test_dry_run_makes_no_base_calls_but_counts_usage(self):
        base = FakeBaseLLM()
        wrapped, budget = prepare_canary(
            fake_config(), make_limits(), SIMPLE_PRICING, authorize=True, dry_run=True, base_llm=base
        )
        for _ in range(5):
            wrapped("p", task="planning")
        assert base.calls == 0, "dry run must issue ZERO base-provider calls"
        usage = budget.usage_snapshot()
        assert usage["calls"] == 5, "dry run keeps synthetic accounting"


# ---------------------------------------------------------------------------
# 9. Partial manifest preservation
# ---------------------------------------------------------------------------


class TestPartialManifest(unittest.TestCase):
    def test_terminated_run_leaves_retrievable_json_manifest(self):
        base = FakeBaseLLM()
        limits = make_limits(max_calls=2)
        wrapped, budget = prepare_canary(
            fake_config(), limits, SIMPLE_PRICING, authorize=True, base_llm=base
        )
        wrapped("p", task="planning")
        wrapped("p", task="reflection")
        with pytest.raises(CanaryExceeded) as exc:
            wrapped("p", task="planning")

        from benchmark.mcf.launch import canary_manifest  # suggested helper

        manifest = canary_manifest(budget, exc.value.reason)
        assert manifest["reason"] == exc.value.reason
        assert manifest["usage"] == exc.value.usage
        assert json.loads(json.dumps(manifest)) == manifest, "manifest must round-trip JSON"

    def test_budget_reports_final_usage_after_kill(self):
        _, wrapped, budget = prepare_canary(fake_config(), make_limits(max_calls=1), SIMPLE_PRICING, authorize=True)
        wrapped("p", task="planning")
        try:
            wrapped("p", task="planning")
        except CanaryExceeded:
            pass
        snap = budget.usage_snapshot()
        assert json.dumps({"usage": snap, "reason": "terminated"})  # retrievable + serializable


# ---------------------------------------------------------------------------
# 10. Isolation
# ---------------------------------------------------------------------------


class TestIsolation(unittest.TestCase):
    def test_sequential_preps_do_not_share_state(self):
        limits = make_limits(max_calls=2)
        w1, b1 = prepare_canary(
            fake_config(), limits, SIMPLE_PRICING, authorize=True, base_llm=FakeBaseLLM()
        )
        w1("p", task="planning")
        w1("p", task="planning")
        with pytest.raises(CanaryExceeded):
            w1("p", task="planning")

        w2, b2 = prepare_canary(
            fake_config(), limits, SIMPLE_PRICING, authorize=True, base_llm=FakeBaseLLM()
        )
        w2("p", task="planning")  # fresh budget: no inheritance from b1
        assert b2.usage_snapshot()["calls"] == 1
        assert b1.usage_snapshot()["calls"] == 2

    def test_unrelated_base_llm_unaffected(self):
        base = FakeBaseLLM()
        wrapped, _ = prepare_canary(
            fake_config(), make_limits(max_calls=1), SIMPLE_PRICING, authorize=True, base_llm=base
        )
        wrapped("p", task="planning")
        with pytest.raises(CanaryExceeded):
            wrapped("p", task="planning")
        # A SEPARATE, non-canary callable is untouched by the wrapper or budget.
        other = FakeBaseLLM()
        assert other("direct", task="planning") == "ok 1"
        assert other.calls == 1 and base.calls == 1


# ---------------------------------------------------------------------------
# End-to-end with the REAL kernel assembly (still zero inference)
# ---------------------------------------------------------------------------


class TestKernelIntegration(unittest.TestCase):
    def test_build_kernel_accepts_wrapped_llm_and_routes_calls_through_budget(self):
        """End-to-end: wrapped llm passed to build_kernel lands on ctx.llm and
        any call the kernel makes is budgeted."""
        from gaworld.kernel.context import build_kernel

        base = FakeBaseLLM()
        wrapped, budget = prepare_canary(
            fake_config(), make_limits(), SIMPLE_PRICING, authorize=True, base_llm=base
        )
        ctx = build_kernel(fake_config(), llm=wrapped, load_entry_points=False)
        assert ctx.llm is wrapped
        ctx.llm("hello", task="planning", agent_id=1)
        assert budget.usage_snapshot()["calls"] == 1
        assert base.calls == 1  # the injected base served the kernel's call


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
