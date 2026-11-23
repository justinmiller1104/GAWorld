"""Contract tests (RED) for the canary budget controller.

Pins the behaviour of a future ``benchmark/mcf/canary.py`` before it exists:
limits, pre-call budget enforcement with in-flight reservation, usage
recording + snapshots, kill conditions with termination reason, dry-run
mode with ZERO inference, and clean-stop semantics.

The module under test must not be created by this phase — every test here
fails on the missing import (that is the RED confirmation).

Mocked only: a deterministic FakeProvider; no LLM inference anywhere.
"""

from __future__ import annotations

import time
import unittest

from benchmark.mcf.canary import (
    CanaryBudget,
    CanaryExceeded,
    CanaryLimits,
    run_canary,
)

# ---------------------------------------------------------------------------
# Deterministic fake provider: never touches the network or a real LLM.
# ---------------------------------------------------------------------------


class FakeProvider:
    """Returns deterministic token counts; error rate is programmable."""

    def __init__(self, tokens_in: int = 300, tokens_out: int = 300, error_every: int = 0):
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.error_every = error_every  # 0 = never fail; n = every n-th call fails
        self.calls = 0

    def call(self, prompt: str = "") -> dict:
        self.calls += 1
        if self.error_every and self.calls % self.error_every == 0:
            raise RuntimeError("simulated provider error")
        return {"tokens_in": self.tokens_in, "tokens_out": self.tokens_out}


#: 1.0 $ per 1M tokens, both directions — keeps cost arithmetic hand-computable.
SIMPLE_PRICING = {"test-model": {"input": 1.0, "output": 1.0}}


def make_limits(**overrides) -> CanaryLimits:
    defaults = {
        "max_calls": 400,
        "max_tokens": 250_000,
        "max_cost_usd": 1.00,
        "max_elapsed_seconds": 1800,
        "max_error_rate": 0.05,
        "min_error_sample": 20,
        "pricing": SIMPLE_PRICING,
    }
    defaults.update(overrides)
    return CanaryLimits(**defaults)


# ---------------------------------------------------------------------------
# 1. CanaryLimits
# ---------------------------------------------------------------------------


class TestCanaryLimits(unittest.TestCase):
    def test_default_limits(self):
        limits = make_limits()
        self.assertEqual(limits.max_calls, 400)
        self.assertEqual(limits.max_tokens, 250_000)
        self.assertEqual(limits.max_cost_usd, 1.00)
        self.assertEqual(limits.max_elapsed_seconds, 1800)
        self.assertEqual(limits.max_error_rate, 0.05)
        self.assertEqual(limits.min_error_sample, 20)

    def test_pricing_is_required(self):
        # Refuses to run without a verified pricing table.
        with self.assertRaises((ValueError, TypeError)):
            make_limits(pricing=None)

    def test_pricing_requires_input_and_output_rates(self):
        with self.assertRaises((ValueError, KeyError)):
            make_limits(pricing={"test-model": {"input": 1.0}})  # output missing


# ---------------------------------------------------------------------------
# 2/6. Pre-call budget enforcement with in-flight reservation
# ---------------------------------------------------------------------------


class TestBudgetCheck(unittest.TestCase):
    def test_check_ok_within_budget(self):
        budget = CanaryBudget(make_limits())
        decision = budget.check(call_number=1, est_tokens=600, est_cost=0.0006)
        self.assertTrue(decision.allowed)

    def test_check_kills_on_call_count(self):
        budget = CanaryBudget(make_limits(max_calls=2))
        budget.check(1, 600, 0.0006)
        decision = budget.check(3, 600, 0.0006)  # call_number beyond max_calls
        self.assertFalse(decision.allowed)
        self.assertIn("call", decision.reason.lower())

    def test_check_kills_on_token_overshoot(self):
        budget = CanaryBudget(make_limits(max_tokens=1000))
        decision = budget.check(call_number=1, est_tokens=1001, est_cost=0.0)
        self.assertFalse(decision.allowed)
        self.assertIn("token", decision.reason.lower())

    def test_check_kills_on_cost(self):
        budget = CanaryBudget(make_limits(max_cost_usd=0.01))
        decision = budget.check(call_number=1, est_tokens=0, est_cost=0.02)
        self.assertFalse(decision.allowed)
        self.assertIn("cost", decision.reason.lower())

    def test_in_flight_reservation_counts_toward_token_budget(self):
        limits = make_limits(max_tokens=1200)
        budget = CanaryBudget(limits)
        budget.reserve(call_id="a", est_tokens=800)  # call in flight
        # 800 in flight + 500 est > 1200 → must refuse even though recorded
        # usage is still zero.
        decision = budget.check(call_number=2, est_tokens=500, est_cost=0.0)
        self.assertFalse(decision.allowed)
        self.assertIn("token", decision.reason.lower())

    def test_release_after_completion_frees_reservation(self):
        budget = CanaryBudget(make_limits(max_tokens=1200))
        budget.reserve(call_id="a", est_tokens=800)
        budget.release(call_id="a")
        decision = budget.check(call_number=2, est_tokens=500, est_cost=0.0)
        self.assertTrue(decision.allowed)


# ---------------------------------------------------------------------------
# 3/4. record + usage_snapshot + kill conditions
# ---------------------------------------------------------------------------


class TestRecordAndSnapshot(unittest.TestCase):
    def test_snapshot_accumulates_usage(self):
        budget = CanaryBudget(make_limits())
        budget.record("c1", ok=True, tokens_in=300, tokens_out=300, latency_s=0.1)
        budget.record("c2", ok=False, tokens_in=0, tokens_out=0, latency_s=0.1)
        snap = budget.usage_snapshot()
        self.assertEqual(snap["calls"], 2)
        self.assertEqual(snap["tokens"], 600)
        self.assertEqual(snap["errors"], 1)
        self.assertAlmostEqual(snap["error_rate"], 0.5)
        self.assertGreaterEqual(snap["elapsed"], 0.0)

    def test_cost_computed_from_pricing_table(self):
        # 300 in + 300 out at 1.0 $/1M each way = 0.0006 $ exactly.
        budget = CanaryBudget(make_limits())
        budget.record("c1", ok=True, tokens_in=300, tokens_out=300, latency_s=0.1)
        snap = budget.usage_snapshot()
        self.assertAlmostEqual(snap["cost_usd"], 0.0006, places=9)

    def test_cost_limit_kills_via_record(self):
        budget = CanaryBudget(make_limits(max_cost_usd=0.001))
        with self.assertRaises(CanaryExceeded) as ctx:
            budget.record("c1", ok=True, tokens_in=1000, tokens_out=1000, latency_s=0.1)
        self.assertIn("cost", str(ctx.exception.reason).lower())
        # Termination record carries accurate usage at stop.
        self.assertEqual(ctx.exception.usage["tokens"], 2000)

    def test_error_rate_not_evaluated_below_min_sample(self):
        # 1 error out of 10 calls (10% ≥ 5%) but n < 20 → must NOT kill.
        budget = CanaryBudget(make_limits())
        for i in range(10):
            budget.record(f"c{i}", ok=(i != 0), tokens_in=10, tokens_out=10, latency_s=0.01)
        snap = budget.usage_snapshot()
        self.assertEqual(snap["calls"], 10)

    def test_error_rate_kill_at_sample_20_with_2_errors(self):
        # 2 errors @ 20 calls = 10% ≥ 5% → kill, raising CanaryExceeded.
        budget = CanaryBudget(make_limits())
        with self.assertRaises(CanaryExceeded) as ctx:
            for i in range(20):
                budget.record(f"c{i}", ok=(i not in (0, 1)), tokens_in=10, tokens_out=10, latency_s=0.01)
        self.assertIn("error", str(ctx.exception.reason).lower())

    def test_error_rate_boundary_exactly_at_threshold(self):
        # 1 error @ 20 calls = exactly 5%. Pin the comparison: kill only when
        # rate is strictly greater than max_error_rate → no kill here.
        budget = CanaryBudget(make_limits())
        for i in range(20):
            budget.record(f"c{i}", ok=(i != 0), tokens_in=10, tokens_out=10, latency_s=0.01)
        snap = budget.usage_snapshot()
        self.assertEqual(snap["calls"], 20)  # still alive

    def test_call_count_kill_via_record(self):
        budget = CanaryBudget(make_limits(max_calls=3))
        with self.assertRaises(CanaryExceeded) as ctx:
            for i in range(4):
                budget.record(f"c{i}", ok=True, tokens_in=10, tokens_out=10, latency_s=0.01)
        self.assertIn("call", str(ctx.exception.reason).lower())

    def test_elapsed_kill_via_record(self):
        budget = CanaryBudget(make_limits(max_elapsed_seconds=0.0))
        time.sleep(0.01)
        with self.assertRaises(CanaryExceeded) as ctx:
            budget.record("c1", ok=True, tokens_in=10, tokens_out=10, latency_s=0.01)
        self.assertIn("elapsed", str(ctx.exception.reason).lower())

    def test_exception_carries_reason_and_usage(self):
        budget = CanaryBudget(make_limits(max_calls=1))
        budget.record("c1", ok=True, tokens_in=100, tokens_out=100, latency_s=0.01)
        with self.assertRaises(CanaryExceeded) as ctx:
            budget.check(call_number=2, est_tokens=10, est_cost=0.0)
        self.assertTrue(ctx.exception.reason)
        self.assertEqual(ctx.exception.usage["calls"], 1)


# ---------------------------------------------------------------------------
# 5. run_canary — dry run issues ZERO inference
# ---------------------------------------------------------------------------


class TestRunCanaryDryRun(unittest.TestCase):
    def test_dry_run_issues_no_provider_calls(self):
        provider = FakeProvider()
        report = run_canary(
            config={"model": "test-model", "calls": 5},
            provider=provider,
            limits=make_limits(),
            dry_run=True,
        )
        self.assertEqual(provider.calls, 0)  # ZERO inference
        self.assertEqual(report["usage"]["calls"], 5)  # full accounting anyway

    def test_dry_run_uses_synthetic_records(self):
        report = run_canary(
            config={"model": "test-model", "calls": 3},
            provider=FakeProvider(tokens_in=300, tokens_out=300),
            limits=make_limits(),
            dry_run=True,
        )
        # 3 × 600 tokens = 1800 tokens at 1.0 $/1M = 0.0018 $
        self.assertEqual(report["usage"]["tokens"], 1800)
        self.assertAlmostEqual(report["usage"]["cost_usd"], 0.0018, places=9)

    def test_live_mode_requires_explicit_authorize(self):
        provider = FakeProvider()
        with self.assertRaises((ValueError, PermissionError, CanaryExceeded)):
            run_canary(
                config={"model": "test-model", "calls": 1},
                provider=provider,
                limits=make_limits(),
                dry_run=False,
            )  # authorize not passed
        self.assertEqual(provider.calls, 0)

    def test_live_mode_with_mock_provider_and_authorize_runs(self):
        # NEVER against a real provider — a mock only.
        provider = FakeProvider(tokens_in=100, tokens_out=100)
        report = run_canary(
            config={"model": "test-model", "calls": 2},
            provider=provider,
            limits=make_limits(),
            dry_run=False,
            authorize=True,
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(report["usage"]["tokens"], 400)

    def test_run_canary_stops_at_budget_and_reports_reason(self):
        provider = FakeProvider(tokens_in=600, tokens_out=600)
        report = run_canary(
            config={"model": "test-model", "calls": 1000},  # asks for far too many
            provider=provider,
            limits=make_limits(max_calls=4),
            dry_run=True,
        )
        self.assertEqual(report["usage"]["calls"], 4)
        self.assertTrue(report["reason"])
        self.assertIn("call", report["reason"].lower())


# ---------------------------------------------------------------------------
# 7. Isolation: no global state
# ---------------------------------------------------------------------------


class TestIsolation(unittest.TestCase):
    def test_budgets_are_independent(self):
        a = CanaryBudget(make_limits(max_calls=1))
        b = CanaryBudget(make_limits(max_calls=1))
        a.record("c1", ok=True, tokens_in=10, tokens_out=10, latency_s=0.01)
        snap_b = b.usage_snapshot()
        self.assertEqual(snap_b["calls"], 0)  # no shared global counters


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
