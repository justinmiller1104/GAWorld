"""Measured vs estimated token accounting for the MCF canary wrapper.

Fake responses only — no network. Three contract cases per the canary
manifest: observed usage is recorded as measured, a missing usage object
falls back to the conservative estimate and is MARKED estimated, and a
malformed/incomplete usage report is never presented as observed.
"""

from __future__ import annotations

import unittest

import gaworld.llm.providers as providers
from benchmark.mcf.canary import CanaryBudget, CanaryLimits
from benchmark.mcf.launch import _CanaryLLM, _measured_usage, canary_manifest


def make_limits(max_tokens: int = 100_000) -> CanaryLimits:
    return CanaryLimits(
        max_calls=50,
        max_tokens=max_tokens,
        pricing={"qwen/qwen3-4b-instruct-2507": {"input": 0.0, "output": 0.0}},
    )


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class TestProviderUsageStash(unittest.TestCase):
    """The OpenAI provider must stash the response's `usage` object (and only
    a well-formed one) for the budget wrapper to pop."""

    def setUp(self):
        self.provider = providers.OpenAIProvider("https://example.invalid/v1", "m", api_key="k")
        self._original_post = providers.requests.post

    def tearDown(self):
        providers.requests.post = self._original_post
        providers.pop_last_token_usage()

    def _payload(self, usage):
        data = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        if usage is not None:
            data["usage"] = usage
        return data

    def test_usage_is_popped_once(self):
        usage = {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165}
        providers.requests.post = lambda *a, **kw: _FakeResponse(self._payload(usage))
        self.provider.call("hi")
        self.assertEqual(providers.pop_last_token_usage(), usage)
        self.assertIsNone(providers.pop_last_token_usage(), "pop must clear the stash")

    def test_missing_usage_yields_none(self):
        providers.requests.post = lambda *a, **kw: _FakeResponse(self._payload(None))
        self.provider.call("hi")
        self.assertIsNone(providers.pop_last_token_usage())

    def test_thread_local_isolation(self):
        import threading

        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        providers.requests.post = lambda *a, **kw: _FakeResponse(self._payload(usage))
        self.provider.call("hi")
        seen = {}
        done = threading.Event()

        def other():
            seen["usage"] = providers.pop_last_token_usage()
            done.set()

        t = threading.Thread(target=other)
        t.start()
        t.join(5)
        self.assertTrue(done.is_set())
        self.assertIsNone(seen["usage"], "another thread must not see this call's usage")
        self.assertEqual(providers.pop_last_token_usage(), usage)


class TestMeasuredUsageParsing(unittest.TestCase):
    def test_valid_pair(self):
        self.assertEqual(
            _measured_usage({"prompt_tokens": 120, "completion_tokens": 45}),
            (120, 45),
        )

    def test_missing_side_falls_back(self):
        self.assertIsNone(_measured_usage({"prompt_tokens": 120}))
        self.assertIsNone(_measured_usage({}))
        self.assertIsNone(_measured_usage(None))
        self.assertIsNone(_measured_usage("120/45"))

    def test_malformed_never_observed(self):
        self.assertIsNone(_measured_usage({"prompt_tokens": -3, "completion_tokens": 5}))
        self.assertIsNone(_measured_usage({"prompt_tokens": "120", "completion_tokens": 45}))
        self.assertIsNone(_measured_usage({"prompt_tokens": True, "completion_tokens": 45}))
        self.assertIsNone(_measured_usage({"prompt_tokens": 1.5, "completion_tokens": 2}))


class _FakeBase:
    """Mimics gaworld.llm.providers.call_llm: returns text, optionally
    leaving a usage object in the thread-local stash."""

    def __init__(self, usage=None, reply="ok"):
        self.usage = usage
        self.reply = reply
        self.calls = 0

    def __call__(self, prompt, task=None, agent_id=None, **kwargs):
        self.calls += 1
        if self.usage is not None:
            providers._LAST_TOKEN_USAGE.openai = self.usage
        return self.reply


class TestCanaryWrapperAccounting(unittest.TestCase):
    """The wrapper records OBSERVED counts when available, marks the rest
    estimated, and keeps pre-call reservations estimate-based."""

    def _wrapper(self, usage, est_in=400, est_out=400, max_tokens=100_000):
        providers.pop_last_token_usage()
        from benchmark.mcf.canary import CanaryBudget

        budget = CanaryBudget(make_limits(max_tokens=max_tokens))
        wrapped = _CanaryLLM(
            _FakeBase(usage=usage),
            budget,
            model="qwen/qwen3-4b-instruct-2507",
            dry_run=False,
            est_tokens_in=est_in,
            est_tokens_out=est_out,
        )
        return wrapped, budget

    def test_measured_usage_recorded_as_observed(self):
        wrapped, budget = self._wrapper({"prompt_tokens": 120, "completion_tokens": 45})
        wrapped("hi", task="t")
        self.assertEqual(budget.usage_snapshot()["tokens"], 165)
        acc = wrapped.token_accounting
        self.assertEqual(acc["measured_calls"], 1)
        self.assertEqual(acc["tokens_measured"], 165)
        self.assertEqual(acc["estimated_calls"], 0)

    def test_missing_usage_falls_back_marked_estimated(self):
        wrapped, budget = self._wrapper(None)
        wrapped("hi", task="t")
        self.assertEqual(budget.usage_snapshot()["tokens"], 800)
        acc = wrapped.token_accounting
        self.assertEqual(acc["estimated_calls"], 1)
        self.assertEqual(acc["tokens_estimated"], 800)
        self.assertEqual(acc["measured_calls"], 0)

    def test_malformed_usage_never_counts_as_measured(self):
        wrapped, budget = self._wrapper({"prompt_tokens": "big", "completion_tokens": 5})
        wrapped("hi", task="t")
        self.assertEqual(budget.usage_snapshot()["tokens"], 800)
        self.assertEqual(wrapped.token_accounting["estimated_calls"], 1)

    def test_pre_call_reservation_stays_estimate_based(self):
        # max_tokens=819: the estimate-based projection (800 + 20 recorded)
        # refuses the second call, while a projection on the measured count
        # (20 + 20) would sail through — enforcement must stay conservative.
        wrapped, _budget = self._wrapper({"prompt_tokens": 10, "completion_tokens": 10}, max_tokens=819)
        wrapped("hi", task="t")
        with self.assertRaises(Exception):
            wrapped("hi again", task="t")

    def test_manifest_reports_accounting(self):
        wrapped, budget = self._wrapper({"prompt_tokens": 30, "completion_tokens": 20})
        budget.latencies = wrapped.latencies
        budget.token_accounting = wrapped.token_accounting
        wrapped("hi", task="t")
        manifest = canary_manifest(budget, "completed")
        self.assertEqual(manifest["token_accounting"]["measured_calls"], 1)
        self.assertEqual(manifest["token_accounting"]["tokens_measured"], 50)

    def test_manifest_without_accounting_attribute(self):
        budget = CanaryBudget(make_limits())
        manifest = canary_manifest(budget, "completed")
        self.assertNotIn("token_accounting", manifest)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
