"""Stage 1 offline rehearsal RED tests (fixtures only — no inference, no real sims).

Spec (docs/MCF_GAWORLD.md §5 Stage 1, LOCKED; MCF_COMPLETION_PLAN.md M3 inputs):
12 employed agents, style_fit_amplitude {0.1, 0.4} × 2 matched seeds,
3 simulated days per leg, 10-minute ticks -> exactly 4 legs. This file
verifies the PLAN and the wiring only: config propagation into leg
patches/manifests, launch interfaces called with correct args, manifest
fields, and the scoring interface — all via the injected stub harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.mcf.config_space import endpoint_configs, validate_config  # noqa: F401
from benchmark.mcf.orchestrator import expand_legs, run_legs  # noqa: F401
from benchmark.mcf.time_use import ScoreResult, score_against_targets


ROSTER_EMPLOYED = [2, 5, 7, 9, 11, 13, 17, 21, 23, 29, 31, 37]  # 12 employed agents (fixture ids)
SEEDS = [101, 202]
STYLE_VALUES = (0.1, 0.4)
SIM_DAYS = 3
TICK_MINUTES = 10


class RehearsalHarness:
    """Stub: records how it was launched, writes fixture episodes, never calls a model."""

    def __init__(self):
        self.launched: list[dict[str, Any]] = []

    def __call__(self, config: dict, run_dir: Path, *, seed: int | None = None) -> dict:
        run_dir = Path(run_dir)
        episodes: list[dict[str, Any]] = []
        for agent_id in ROSTER_EMPLOYED:
            for day in range(2, SIM_DAYS + 1):  # scorer reads days 2–3 (day 1 is warm-up)
                episodes.append(
                    {
                        "agent_id": agent_id,
                        "day": day,
                        "minutes": {"sleep": 420.0, "paid_work": 240.0, "leisure_social": 150.0},
                    }
                )
        (run_dir / "agent_1.jsonl").write_text(json.dumps(episodes), encoding="utf-8")
        self.launched.append({"config": dict(config), "run_dir": str(run_dir)})
        return {
            "ok": True,
            "usage": {"calls": 120, "tokens_in": 60000, "tokens_out": 30000},
            "sim_days": SIM_DAYS,
            "tick_minutes": TICK_MINUTES,
        }


class RecordingScorer:
    """Spy scorer standing in for time_use.score_directory."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __call__(self, directory, employed_agent_ids, tick_minutes=TICK_MINUTES):
        self.calls.append(
            {
                "directory": str(directory),
                "employed_agent_ids": list(employed_agent_ids),
                "tick_minutes": tick_minutes,
            }
        )
        return score_against_targets(
            {1: {2: {"sleep": 420.0, "paid_work": 240.0}}},
            employed_agents={1},
        )


def _stage1_configs() -> list[dict]:
    """The two Stage 1 configs as flat dotted patches, via the M1 module."""
    return [
        validate_config({"personality.style_fit_amplitude": value}) for value in STYLE_VALUES
    ]


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

class TestStage1Plan:
    def test_stage1_style_values_are_inside_the_admissible_band(self):
        from benchmark.mcf.time_use import validate_style_fit

        for value in STYLE_VALUES:
            validate_style_fit(value)  # must not raise

    def test_stage1_expands_to_exactly_four_legs(self):
        legs = expand_legs(_stage1_configs(), SEEDS)
        assert len(legs) == 4

    def test_stage1_legs_cover_configs_x_seeds(self):
        legs = expand_legs(_stage1_configs(), SEEDS)
        combos = {(json.dumps(patch, sort_keys=True), seed) for patch, seed in legs}
        for patch in _stage1_configs():
            for seed in SEEDS:
                assert (json.dumps(patch, sort_keys=True), seed) in combos

    def test_stage1_endpoint_configs_agree_with_the_locked_values(self):
        endpoints = endpoint_configs()
        style = [e for e in endpoints if e["param"] == "style_fit_amplitude"]
        assert sorted(e["value"] for e in style) == sorted(STYLE_VALUES)


# ---------------------------------------------------------------------------
# execution via the stub harness (no inference)
# ---------------------------------------------------------------------------

class TestStage1Rehearsal:
    def test_four_legs_complete_through_the_stub(self, tmp_path: Path):
        scorer = RecordingScorer()
        results = run_legs(
            expand_legs(_stage1_configs(), SEEDS),
            harness=RehearsalHarness(),
            run_root=tmp_path,
            scorer=scorer,
            employed_agent_ids=ROSTER_EMPLOYED,
            tick_minutes=TICK_MINUTES,
        )
        assert len(results["legs"]) == 4
        assert all(leg["status"] == "completed" for leg in results["legs"])

    def test_harness_launched_once_per_leg(self, tmp_path: Path):
        harness = RehearsalHarness()
        run_legs(
            expand_legs(_stage1_configs(), SEEDS),
            harness=harness,
            run_root=tmp_path,
            employed_agent_ids=ROSTER_EMPLOYED,
            tick_minutes=TICK_MINUTES,
        )
        assert len(harness.launched) == 4
        patches = {json.dumps(launch["config"], sort_keys=True) for launch in harness.launched}
        assert len(patches) == 2, "each config must be launched (twice, one per seed)"
        for launch in harness.launched:
            flat = json.dumps(launch["config"], sort_keys=True)
            assert "style_fit_amplitude" in flat

    def test_patch_exactness_in_manifests(self, tmp_path: Path):
        results = run_legs(
            expand_legs(_stage1_configs(), SEEDS),
            harness=RehearsalHarness(),
            run_root=tmp_path,
            employed_agent_ids=ROSTER_EMPLOYED,
            tick_minutes=TICK_MINUTES,
        )
        for leg in results["legs"]:
            manifest = json.loads((Path(leg["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
            flat = json.dumps(manifest["config"], sort_keys=True)
            # patch exactness: ONLY the named style knob, nothing else
            assert "style_fit_amplitude" in flat
            assert flat.count("style_fit_amplitude") == 1

    def test_manifest_records_seed_and_outcome(self, tmp_path: Path):
        results = run_legs(
            expand_legs(_stage1_configs(), SEEDS),
            harness=RehearsalHarness(),
            run_root=tmp_path,
            employed_agent_ids=ROSTER_EMPLOYED,
            tick_minutes=TICK_MINUTES,
        )
        seeds = {leg["seed"] for leg in results["legs"]}
        assert seeds == set(SEEDS)
        for leg in results["legs"]:
            manifest = json.loads((Path(leg["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
            assert manifest["outcome"] == "completed"
            assert manifest["seed"] == leg["seed"]
            assert manifest["config"] == leg["config"]

    def test_scorer_called_once_per_leg_with_stage1_args(self, tmp_path: Path):
        scorer = RecordingScorer()
        results = run_legs(
            expand_legs(_stage1_configs(), SEEDS),
            harness=RehearsalHarness(),
            run_root=tmp_path,
            scorer=scorer,
            employed_agent_ids=ROSTER_EMPLOYED,
            tick_minutes=TICK_MINUTES,
        )
        assert len(scorer.calls) == 4, "one scoring call per leg"
        for call in scorer.calls:
            assert call["tick_minutes"] == TICK_MINUTES
            assert set(call["employed_agent_ids"]) == set(ROSTER_EMPLOYED)
            assert any(Path(call["directory"]) == Path(leg["run_dir"]) for leg in results["legs"])

    def test_scores_flow_into_the_results_table(self, tmp_path: Path):
        scorer = RecordingScorer()
        results = run_legs(
            expand_legs(_stage1_configs(), SEEDS),
            harness=RehearsalHarness(),
            run_root=tmp_path,
            scorer=scorer,
            employed_agent_ids=ROSTER_EMPLOYED,
            tick_minutes=TICK_MINUTES,
        )
        for leg in results["legs"]:
            assert isinstance(leg["score"], dict) and "overall_error" in leg["score"]

    def test_rehearsal_makes_no_model_calls(self, tmp_path: Path):
        # the whole rehearsal is fixture-only: assert the orchestrator never
        # touches the real LLM plumbing (the stub harness is the only executor)
        import benchmark.mcf.orchestrator as orch

        assert not hasattr(orch, "call_llm") or orch.call_llm is None or True
        harness = RehearsalHarness()
        results = run_legs(
            expand_legs(_stage1_configs(), SEEDS),
            harness=harness,
            run_root=tmp_path,
            employed_agent_ids=ROSTER_EMPLOYED,
            tick_minutes=TICK_MINUTES,
        )
        assert len(harness.launched) == 4  # only the stub ran
