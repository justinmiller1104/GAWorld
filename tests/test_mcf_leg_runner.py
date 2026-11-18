"""RED tests — the real per-leg MCF runner (task 3753 sub-task A).

``gaworld/apps/mcf_api.py:_build_harness`` needs a run_legs-compatible
per-leg harness that executes one (config, seed) leg in an isolated run dir
under canary budget control and feeds ``legs.json``. These tests pin the
contract offline: stub legs (zero inference), budget-kill honesty, failure
isolation, manifest determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.mcf.config_space import config_id
from benchmark.mcf.orchestrator import run_legs


def _flatten(patch, prefix=""):
    flat = {}
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


LOW = {"personality": {"style_fit_amplitude": 0.10}}
HIGH = {"personality": {"style_fit_amplitude": 0.40}}


@pytest.fixture()
def _clean_llm_state():
    import gaworld.llm.providers as providers

    original = providers.call_llm
    yield
    providers.call_llm = original


# ----------------------------------------------------------------- shape


def test_run_leg_harness_factory_exists():
    from benchmark.mcf.experiment import run_leg
    from benchmark.mcf.leg_runner import make_leg_harness

    assert callable(make_leg_harness)
    assert callable(run_leg)


def test_default_run_leg_is_the_real_vehicle():
    """The module-level run_leg is bound to the real simulator execution, not
    the offline stub — the stub is opt-in per harness."""
    import inspect

    from benchmark.mcf import leg_runner

    src = inspect.getsource(leg_runner)
    assert "run_simulation" in src  # the real entry module is the vehicle


# ------------------------------------------------------- stub leg behavior


def test_stub_leg_isolated_run_dir_and_manifest(tmp_path, _clean_llm_state):
    from benchmark.mcf.leg_runner import make_leg_harness

    harness = make_leg_harness(stub=True)
    leg_dir = tmp_path / "leg_000"
    leg_dir.mkdir()
    result = harness(LOW, leg_dir, seed=7)
    assert result["ok"] is True
    run_dir = Path(result["run_dir"])
    assert run_dir.parent == leg_dir and run_dir != leg_dir  # isolated inside
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outcome"] == "completed"
    assert manifest["run_id"] == run_dir.name
    assert result["usage"]["calls"] >= 1


def test_stub_leg_scores_through_the_real_scorer(tmp_path, _clean_llm_state):
    """The stub writes real episode shapes so time_use.score_directory scores
    the leg — run_legs rows carry a score dict, not None."""
    from benchmark.mcf import leg_runner
    from benchmark.mcf.leg_runner import make_leg_harness
    from benchmark.mcf.time_use import score_directory

    harness = make_leg_harness(stub=True)
    result = harness(LOW, tmp_path, seed=7)
    score = score_directory(result["run_dir"], leg_runner.STUB_EMPLOYED_AGENTS)
    assert score.per_category["paid_work"]["n_agent_days"] > 0


def test_stub_leg_deterministic_and_patch_sensitive(tmp_path, _clean_llm_state):
    from benchmark.mcf.leg_runner import make_leg_harness
    from benchmark.mcf.time_use import score_directory

    harness = make_leg_harness(stub=True)
    low1 = score_directory(harness(LOW, tmp_path / "a", seed=7)["run_dir"], ())
    low2 = score_directory(harness(LOW, tmp_path / "b", seed=7)["run_dir"], ())
    high = score_directory(harness(HIGH, tmp_path / "c", seed=7)["run_dir"], ())
    assert low1.overall_error == low2.overall_error  # deterministic per (config, seed)
    assert low1.overall_error != high.overall_error  # the patch moves the leg


def test_run_legs_rows_feed_legs_json_contract(tmp_path, _clean_llm_state):
    """One full run_legs pass over stub legs: rows carry config_id, status,
    usage, score — exactly what mcf_api._compute_results reads."""
    from benchmark.mcf.leg_runner import make_leg_harness
    from benchmark.mcf.time_use import score_directory

    harness = make_leg_harness(stub=True)
    table = run_legs(
        [(LOW, 7), (HIGH, 7)],
        harness=harness,
        run_root=tmp_path / "runs",
        scorer=score_directory,
    )
    rows = table["legs"]
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "completed"
        assert row["config_id"] == config_id(_flatten(row["config"]))
        assert row["score"] is not None
        assert row["score"]["per_category"]
    assert rows[0]["score"]["overall_error"] != rows[1]["score"]["overall_error"]


# ------------------------------------------------------------ budget kill


class _ChattyExecute:
    """Stub execution that hammers the (wrapped) call_llm — drives the canary
    over its caps without any inference."""

    def __init__(self, calls: int):
        self.calls = calls

    def __call__(self, config, run_dir, *, seed=0):
        import gaworld.llm.providers as providers

        for _ in range(self.calls):
            providers.call_llm("prompt", task="leg")
        return {"ok": True, "usage": {"calls": self.calls, "tokens_in": 0, "tokens_out": 0}}


def test_budget_kill_is_honest_and_isolated(tmp_path, _clean_llm_state):
    """A leg running past its caps is killed with a budget stop reason; the
    other legs still complete."""
    from benchmark.mcf.leg_runner import make_leg_harness
    from benchmark.mcf.time_use import score_directory

    limits = {"max_calls": 5, "max_tokens": 10_000_000, "max_elapsed_seconds": 3600}
    killed = make_leg_harness(stub=True, limits=limits, execute=_ChattyExecute(50))
    good = make_leg_harness(stub=True, limits=limits)

    def harness(config, run_dir, *, seed=0):
        return (killed if config == LOW else good)(config, run_dir, seed=seed)

    table = run_legs(
        [(LOW, 7), (HIGH, 7)],
        harness=harness,
        run_root=tmp_path / "runs",
        scorer=score_directory,
    )
    by_patch = {json.dumps(_flatten(r["config"]), sort_keys=True): r for r in table["legs"]}
    low_row = by_patch[json.dumps(_flatten(LOW), sort_keys=True)]
    high_row = by_patch[json.dumps(_flatten(HIGH), sort_keys=True)]
    assert low_row["status"] != "completed"
    assert "budget" in low_row["stop_reason"]
    assert high_row["status"] == "completed"
    # the killed leg's manifest records the interruption, usage from the
    # canary ledger, never a fabricated completion
    killed_manifest = json.loads((Path(low_row["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert killed_manifest["outcome"] == "interrupted"


def test_budget_kill_without_scorer_leaves_leg_unscored_not_failed(tmp_path, _clean_llm_state):
    from benchmark.mcf.leg_runner import make_leg_harness

    killed = make_leg_harness(stub=True, limits={"max_calls": 3}, execute=_ChattyExecute(30))
    result = killed(LOW, tmp_path, seed=7)
    assert result["ok"] is False
    assert "budget" in result["error"]


# -------------------------------------------------------- failure isolation


def test_one_crashing_leg_does_not_kill_others(tmp_path, _clean_llm_state):
    from benchmark.mcf.leg_runner import make_leg_harness
    from benchmark.mcf.time_use import score_directory

    class Boom:
        def __call__(self, config, run_dir, *, seed=0):
            raise RuntimeError("leg exploded")

    good = make_leg_harness(stub=True)
    boom = make_leg_harness(stub=True, execute=Boom())

    def harness(config, run_dir, *, seed=0):
        return (boom if config == LOW else good)(config, run_dir, seed=seed)

    table = run_legs(
        [(LOW, 7), (HIGH, 7)],
        harness=harness,
        run_root=tmp_path / "runs",
        scorer=score_directory,
    )
    by_patch = {json.dumps(_flatten(r["config"]), sort_keys=True): r for r in table["legs"]}
    assert by_patch[json.dumps(_flatten(LOW), sort_keys=True)]["status"] == "failed"
    assert by_patch[json.dumps(_flatten(HIGH), sort_keys=True)]["status"] == "completed"


# --------------------------------------------------------- manifest honesty


def test_manifest_records_model_info_and_seed(tmp_path, _clean_llm_state):
    from benchmark.mcf.leg_runner import make_leg_harness

    harness = make_leg_harness(stub=True, model_info={"provider": "stub", "model": "m"})
    result = harness(LOW, tmp_path, seed=11)
    manifest = json.loads(Path(result["run_dir"], "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 11
    assert manifest["model"] == {"provider": "stub", "model": "m"}


def test_repeated_calls_restart_per_run(tmp_path, _clean_llm_state):
    from benchmark.mcf.leg_runner import make_leg_harness

    harness = make_leg_harness(stub=True)
    r1 = harness(LOW, tmp_path, seed=7)
    r2 = harness(LOW, tmp_path, seed=7)
    assert r1["run_id"] != r2["run_id"]
