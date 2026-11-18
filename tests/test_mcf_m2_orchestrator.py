"""M2 RED tests — multi-leg orchestrator (config × seed loop, per-leg caps).

Spec (MCF_COMPLETION_PLAN.md M2, docs/MCF_GAWORLD.md §5 Stage 1):
- expand configs × matched seeds into legs;
- each leg runs through the existing isolated-run path (experiment.py
  run_experiment/_execute_arm shape, launch.py create_run/finalize_run,
  canary-style kill limits) with an INJECTED stub harness (offline);
- per-leg caps as configurable defaults from doc §5 (max_calls ≈ 21000,
  max_tokens ≈ 11M, max_elapsed_seconds ≈ 80000, error rate 5%);
- a violation terminates ONLY the affected leg and preserves its manifest
  + evidence with a stop reason;
- measured-vs-estimated token honesty (launch.py _measured_usage semantics);
- missing scoring data -> explicit unscored marker, never fabricated numbers;
- one machine-readable JSON results table; identical offline inputs ->
  identical table content (modulo run ids/dirs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.mcf.orchestrator import (  # noqa: F401
    DEFAULT_LEG_LIMITS,
    expand_legs,
    run_legs,
)
from benchmark.mcf.time_use import ScoreResult, score_against_targets


# ---------------------------------------------------------------------------
# stub harness (offline — no inference anywhere in this file)
# ---------------------------------------------------------------------------

class StubHarness:
    """Records (config, seed, run_dir) per leg and writes deterministic episode files.

    Orchestrator contract: ``harness(config, run_dir, *, seed=...)`` — the seed
    travels as a keyword, the config patch stays clean (no reserved keys)."""

    def __init__(self, calls: int = 5, tokens_in: int = 100, tokens_out: int = 50, fail_on_seed: Any = None):
        self.seen: list[tuple[dict, int, Path]] = []
        self.calls = calls
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.fail_on_seed = fail_on_seed

    def __call__(self, config: dict, run_dir: Path, *, seed: int | None = None) -> dict:
        self.seen.append((config, seed, Path(run_dir)))
        (Path(run_dir) / "agent_1.jsonl").write_text(
            json.dumps([{"agent_id": 1, "day": 2, "minutes": {"sleep": 480}}]),
            encoding="utf-8",
        )
        usage: dict[str, Any] = {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }
        if self.fail_on_seed is not None and seed == self.fail_on_seed:
            return {"ok": False, "usage": usage}
        return {"ok": True, "usage": usage}


class SpyScorer:
    """Injectable scorer standing in for time_use.score_directory."""

    def __init__(self, result: ScoreResult | None):
        self.calls: list[dict[str, Any]] = []
        self.result = result

    def __call__(self, directory, employed_agent_ids, tick_minutes=10):
        self.calls.append({"directory": str(directory), "employed_agent_ids": list(employed_agent_ids), "tick_minutes": tick_minutes})
        return self.result


def _scored_result() -> ScoreResult:
    return score_against_targets(
        {1: {2: {"sleep": 480.0, "paid_work": 240.0, "leisure_social": 150.0}}},
        employed_agents={1},
    )


# ---------------------------------------------------------------------------
# leg expansion
# ---------------------------------------------------------------------------

class TestExpandLegs:
    def test_two_configs_two_seeds_make_four_legs(self):
        configs = [{"personality.style_fit_amplitude": 0.1}, {"personality.style_fit_amplitude": 0.4}]
        legs = expand_legs(configs, [101, 202])
        assert len(legs) == 4

    def test_every_combination_present_exactly_once(self):
        configs = [{"a.b": 1}, {"a.b": 2}]
        legs = expand_legs(configs, [7, 8])
        seen = [(json.dumps(patch, sort_keys=True), seed) for patch, seed in legs]
        assert len(set(seen)) == 4
        for patch in configs:
            for seed in (7, 8):
                assert (json.dumps(patch, sort_keys=True), seed) in seen

    def test_configs_are_not_mutated_by_expansion(self):
        config = {"personality.style_fit_amplitude": 0.1}
        expand_legs([config], [1, 2])
        assert config == {"personality.style_fit_amplitude": 0.1}


# ---------------------------------------------------------------------------
# default limits — doc §5 Stage 1 caps, configurable
# ---------------------------------------------------------------------------

class TestLimits:
    def test_stage1_default_caps(self):
        assert DEFAULT_LEG_LIMITS["max_calls"] == 21000
        assert DEFAULT_LEG_LIMITS["max_tokens"] == 11_000_000
        assert DEFAULT_LEG_LIMITS["max_elapsed_seconds"] == 80000
        assert DEFAULT_LEG_LIMITS["max_error_rate"] == 0.05

    def test_limits_are_configurable_per_call(self, tmp_path: Path):
        harness = StubHarness(calls=5)
        results = run_legs(
            [({"a.b": 1}, 1)],
            harness=harness,
            run_root=tmp_path,
            limits={"max_calls": 3, "max_tokens": 11_000_000, "max_elapsed_seconds": 80000, "max_error_rate": 0.05},
        )
        assert results["legs"][0]["status"] != "completed"


# ---------------------------------------------------------------------------
# leg isolation + manifests
# ---------------------------------------------------------------------------

class TestLegIsolation:
    def test_each_leg_gets_its_own_run_dir_and_manifest(self, tmp_path: Path):
        harness = StubHarness()
        results = run_legs(
            [({"a.b": 1}, 1), ({"a.b": 2}, 2)],
            harness=harness,
            run_root=tmp_path,
        )
        dirs = [Path(leg["run_dir"]) for leg in results["legs"]]
        assert len(set(dirs)) == 2, "two legs must never share a run directory"
        for leg in results["legs"]:
            manifest = json.loads((Path(leg["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
            assert manifest["outcome"] == "completed"
            assert manifest["config"] == leg["config"]
            assert manifest["seed"] == leg["seed"]

    def test_harness_receives_each_legs_config_and_run_dir(self, tmp_path: Path):
        harness = StubHarness()
        run_legs([({"a.b": 1}, 1), ({"a.b": 2}, 2)], harness=harness, run_root=tmp_path)
        seeds = [seed for _, seed, _ in harness.seen]
        assert sorted(str(s) for s in seeds) == ["1", "2"], "each leg's seed must reach the harness"
        configs = [json.dumps(config, sort_keys=True) for config, _, _ in harness.seen]
        assert configs.count(json.dumps({"a.b": 1}, sort_keys=True)) == 1
        assert configs.count(json.dumps({"a.b": 2}, sort_keys=True)) == 1


# ---------------------------------------------------------------------------
# budget violations
# ---------------------------------------------------------------------------

class TestBudgetViolation:
    def test_over_budget_leg_stops_with_reason_others_complete(self, tmp_path: Path):
        # seed 999's harness burns far more calls than the (configured) cap:
        # only that leg must die; the others keep running to completion.
        class MixedHarness(StubHarness):
            def __call__(self, config, run_dir, *, seed=None):
                result = super().__call__(config, run_dir, seed=seed)
                if seed == 999:
                    result["usage"] = {"calls": 999999, "tokens_in": 1, "tokens_out": 1}
                return result

        results = run_legs(
            [({"a.b": 1}, 1), ({"a.b": 1}, 999)],
            harness=MixedHarness(),
            run_root=tmp_path,
            limits={"max_calls": 21000, "max_tokens": 11_000_000, "max_elapsed_seconds": 80000, "max_error_rate": 0.05},
            scorer=SpyScorer(_scored_result()),
        )
        by_seed = {leg["seed"]: leg for leg in results["legs"]}
        assert by_seed[1]["status"] == "completed"
        assert by_seed[999]["status"] in ("interrupted", "failed")
        assert by_seed[999]["stop_reason"], "a killed leg must record WHY it was stopped"
        assert "max_calls" in by_seed[999]["stop_reason"]

    def test_killed_leg_preserves_manifest_and_evidence(self, tmp_path: Path):
        harness = StubHarness(fail_on_seed=999)
        results = run_legs(
            [({"a.b": 1}, 999)],
            harness=harness,
            run_root=tmp_path,
        )
        leg = results["legs"][0]
        manifest = json.loads((Path(leg["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
        # evidence is on disk and not overwritten by a fabricated "completed"
        assert manifest["outcome"] != "completed"
        assert manifest["config"] == {"a.b": 1}
        assert manifest["seed"] == 999
        assert (Path(leg["run_dir"]) / "agent_1.jsonl").exists(), "partial evidence must survive"


# ---------------------------------------------------------------------------
# token honesty (measured vs estimated)
# ---------------------------------------------------------------------------

class TestTokenHonesty:
    def test_measured_usage_is_reported_as_measured(self, tmp_path: Path):
        class MeasuredHarness(StubHarness):
            def __call__(self, config, run_dir, *, seed=None):
                result = super().__call__(config, run_dir, seed=seed)
                result["measured_usage"] = {"prompt_tokens": 100, "completion_tokens": 50}
                return result

        results = run_legs([({"a.b": 1}, 1)], harness=MeasuredHarness(), run_root=tmp_path)
        leg = results["legs"][0]
        assert leg["usage_source"] == "measured"
        assert leg["usage"]["tokens_in"] == 100
        assert leg["usage"]["tokens_out"] == 50

    def test_missing_usage_falls_back_to_estimated_marker(self, tmp_path: Path):
        class NoUsageHarness(StubHarness):
            def __call__(self, config, run_dir, *, seed=None):
                return {"ok": True}

        results = run_legs(
            [({"a.b": 1}, 1)], harness=NoUsageHarness(), run_root=tmp_path, scorer=SpyScorer(_scored_result())
        )
        leg = results["legs"][0]
        assert leg["usage_source"] == "estimated"
        assert leg["status"] == "completed"

    def test_malformed_usage_is_never_presented_as_measured(self, tmp_path: Path):
        # mirrors launch.py:_measured_usage — a partial usage object is
        # estimated, not observed
        class BrokenUsageHarness(StubHarness):
            def __call__(self, config, run_dir, *, seed=None):
                result = super().__call__(config, run_dir, seed=seed)
                result["measured_usage"] = {"prompt_tokens": 100}  # completion missing
                return result

        results = run_legs(
            [({"a.b": 1}, 1)], harness=BrokenUsageHarness(), run_root=tmp_path, scorer=SpyScorer(_scored_result())
        )
        assert results["legs"][0]["usage_source"] == "estimated"


# ---------------------------------------------------------------------------
# scoring collection
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_result_is_collected_when_scoring_data_exists(self, tmp_path: Path):
        scorer = SpyScorer(_scored_result())
        results = run_legs([({"a.b": 1}, 1)], harness=StubHarness(), run_root=tmp_path, scorer=scorer)
        leg = results["legs"][0]
        assert leg["status"] == "completed"
        assert isinstance(leg["score"], dict)
        assert "overall_error" in leg["score"]
        assert "coverage" in leg["score"]
        assert len(scorer.calls) == 1 and scorer.calls[0]["directory"] == leg["run_dir"]

    def test_missing_scoring_data_is_explicitly_unscored(self, tmp_path: Path):
        # scorer finds no usable agent-days (or returns None / raises) ->
        # the leg is explicitly marked unscored, never given fabricated numbers
        results = run_legs(
            [({"a.b": 1}, 1)], harness=StubHarness(), run_root=tmp_path, scorer=SpyScorer(None)
        )
        leg = results["legs"][0]
        assert leg["status"] == "unscored"
        assert leg["score"] is None, "a leg with no scoring data must never carry fabricated numbers"
        assert "overall_error" not in json.dumps(leg)

    def test_scorer_error_is_unscored_not_failed(self, tmp_path: Path):
        def boom(directory, employed_agent_ids, tick_minutes=10):
            raise RuntimeError("no episode files")

        results = run_legs([({"a.b": 1}, 1)], harness=StubHarness(), run_root=tmp_path, scorer=boom)
        assert results["legs"][0]["status"] == "unscored"
        assert results["legs"][0]["score"] is None


# ---------------------------------------------------------------------------
# results table
# ---------------------------------------------------------------------------

class TestResultsTable:
    def test_table_is_json_serializable_with_required_fields(self, tmp_path: Path):
        results = run_legs([({"a.b": 1}, 1)], harness=StubHarness(), run_root=tmp_path)
        blob = json.dumps(results)
        leg = json.loads(blob)["legs"][0]
        for field in ("config_id", "seed", "model", "status", "run_dir", "usage"):
            assert field in leg, f"results table row missing {field!r}"

    def test_model_recorded_from_real_config_routing_default(self, tmp_path: Path):
        from gaworld.settings import CONFIG

        results = run_legs([({"a.b": 1}, 1)], harness=StubHarness(), run_root=tmp_path)
        assert results["legs"][0]["model"] == CONFIG["llm"]["routing"]["default"]

    def test_identical_offline_inputs_identical_table(self, tmp_path: Path):
        def strip(run: dict) -> str:
            legs = [{k: v for k, v in leg.items() if k not in ("run_dir",)} for leg in run["legs"]]
            legs.sort(key=lambda leg: (leg["config_id"], str(leg["seed"])))
            return json.dumps(legs, sort_keys=True)

        harness_a = StubHarness()
        harness_b = StubHarness()
        run_a = run_legs([({"a.b": 1}, 1), ({"a.b": 2}, 2)], harness=harness_a, run_root=tmp_path / "a")
        run_b = run_legs([({"a.b": 1}, 1), ({"a.b": 2}, 2)], harness=harness_b, run_root=tmp_path / "b")
        assert strip(run_a) == strip(run_b)
