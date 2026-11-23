"""M3 RED tests — live max_error_rate enforcement during multi-leg runs.

docs/MCF_GAWORLD.md §5 Stage 1 caps include max_error_rate (default 0.05),
but ``orchestrator.run_legs`` currently only enforces max_calls /
max_tokens / max_elapsed_seconds. These tests specify enforcement driven by
the harness/canary error accounting: a harness result whose usage snapshot
carries per-call error counts (what ``canary.wrap_call_llm`` records via
``CanaryBudget.usage_snapshot()``: ``calls`` + ``errors``) must interrupt the
affected leg with ``stop_reason`` naming ``max_error_rate`` while every other
leg completes untouched, with evidence preserved.

Enforcement contract (mirrors canary.CanaryLimits semantics):
- error_rate = errors / calls, enforced only once calls >= min_error_sample
  (default 20, overridable via limits["min_error_sample"]);
- rate strictly greater than max_error_rate interrupts; exactly equal does not;
- the interrupted leg keeps its manifest + usage evidence; other legs are
  unaffected;
- a usage dict without error counts is never treated as an error-rate
  violation (no fabricated error data).

All tests use stub harnesses — zero LLM calls, zero inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.mcf.orchestrator import DEFAULT_LEG_LIMITS, run_legs


def _usage(calls: int, errors: int, tokens_in: int = 10, tokens_out: int = 5) -> dict[str, Any]:
    """A usage snapshot shaped like canary.CanaryBudget.usage_snapshot() —
    what the canary-wrapped harness reports per leg."""
    return {"calls": calls, "errors": errors, "tokens_in": tokens_in, "tokens_out": tokens_out}


class ErrorRateHarness:
    """Returns per-leg canned results keyed by invocation order."""

    def __init__(self, results: list[dict[str, Any]]):
        self.results = list(results)
        self.seen: list[Path] = []

    def __call__(self, config: dict, run_dir: Path, *, seed: int | None = None) -> dict:
        self.seen.append(Path(run_dir))
        return dict(self.results.pop(0))


# ---------------------------------------------------------------------------
# default cap present (already specified in M2 — kept as an anchor)
# ---------------------------------------------------------------------------


class TestDefaultCap:
    def test_default_limits_include_max_error_rate(self):
        assert DEFAULT_LEG_LIMITS["max_error_rate"] == 0.05


# ---------------------------------------------------------------------------
# enforcement
# ---------------------------------------------------------------------------


class TestErrorRateEnforcement:
    def test_leg_over_error_rate_is_interrupted_with_named_reason(self, tmp_path):
        # 10 failures / 100 calls = 10% > 5%, sample large enough
        harness = ErrorRateHarness([{"ok": True, "usage": _usage(100, errors=10)}])
        table = run_legs([({"a.b": 1}, 7)], harness=harness, run_root=tmp_path)
        row = table["legs"][0]
        assert row["status"] == "interrupted"
        assert "max_error_rate" in row["stop_reason"], (
            f"stop_reason must name the violated cap, got: {row['stop_reason']!r}"
        )

    def test_leg_at_or_under_error_rate_completes(self, tmp_path):
        # 5 failures / 100 calls = 5% — exactly at the cap, not over
        harness = ErrorRateHarness([{"ok": True, "usage": _usage(100, errors=5)}])
        row = run_legs([({"a.b": 1}, 7)], harness=harness, run_root=tmp_path)["legs"][0]
        assert row["status"] == "completed"

    def test_boundary_rate_just_over_cap_interrupts(self, tmp_path):
        # 2 / 40 = 5.0% at cap; 3 / 40 = 7.5% over
        harness = ErrorRateHarness(
            [{"ok": True, "usage": _usage(40, errors=2)}, {"ok": True, "usage": _usage(40, errors=3)}]
        )
        rows = run_legs([({"a": 1}, 1), ({"a": 2}, 2)], harness=harness, run_root=tmp_path)["legs"]
        assert rows[0]["status"] == "completed"
        assert rows[1]["status"] == "interrupted"

    def test_error_counts_surface_in_row_usage(self, tmp_path):
        harness = ErrorRateHarness([{"ok": True, "usage": _usage(100, errors=10)}])
        row = run_legs([({"a.b": 1}, 7)], harness=harness, run_root=tmp_path)["legs"][0]
        assert row["usage"].get("errors") == 10, (
            "the interrupted leg's evidence must include the error count that triggered enforcement"
        )

    def test_bad_leg_does_not_affect_other_legs(self, tmp_path):
        harness = ErrorRateHarness(
            [
                {"ok": True, "usage": _usage(100, errors=50)},  # 50% — way over
                {"ok": True, "usage": _usage(100, errors=0)},
            ]
        )
        rows = run_legs([({"a": 1}, 1), ({"a": 2}, 2)], harness=harness, run_root=tmp_path)["legs"]
        assert rows[0]["status"] == "interrupted"
        assert rows[1]["status"] == "completed"

    def test_evidence_preserved_on_interrupted_leg(self, tmp_path):
        harness = ErrorRateHarness([{"ok": True, "usage": _usage(100, errors=10)}])
        row = run_legs([({"a.b": 1}, 7)], harness=harness, run_root=tmp_path)["legs"][0]
        manifest = json.loads((Path(row["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["outcome"] == "interrupted"
        assert "max_error_rate" in manifest["stop_reason"]

    def test_min_error_sample_guard_small_samples_not_enforced(self, tmp_path):
        # 1 failure / 10 calls = 10% but below the default min sample of 20
        harness = ErrorRateHarness([{"ok": True, "usage": _usage(10, errors=1)}])
        row = run_legs([({"a.b": 1}, 7)], harness=harness, run_root=tmp_path)["legs"][0]
        assert row["status"] == "completed"

    def test_min_error_sample_overridable_via_limits(self, tmp_path):
        harness = ErrorRateHarness([{"ok": True, "usage": _usage(4, errors=1)}])  # 25%
        row = run_legs(
            [({"a.b": 1}, 7)],
            harness=harness,
            run_root=tmp_path,
            limits={"min_error_sample": 4},
        )["legs"][0]
        assert row["status"] == "interrupted"

    def test_no_error_counts_means_no_error_rate_enforcement(self, tmp_path):
        # estimated-only usage (no canary snapshot): never a fabricated violation
        harness = ErrorRateHarness([{"ok": True, "usage": {"calls": 100, "tokens_in": 10, "tokens_out": 5}}])
        row = run_legs([({"a.b": 1}, 7)], harness=harness, run_root=tmp_path)["legs"][0]
        assert row["status"] == "completed"

    def test_configurable_stricter_cap_enforced(self, tmp_path):
        # 1 / 20 = 5% — over a tightened cap of 0.01, sample met
        harness = ErrorRateHarness([{"ok": True, "usage": _usage(20, errors=1)}])
        row = run_legs(
            [({"a.b": 1}, 7)],
            harness=harness,
            run_root=tmp_path,
            limits={"max_error_rate": 0.01},
        )["legs"][0]
        assert row["status"] == "interrupted"
        assert "max_error_rate" in row["stop_reason"]
