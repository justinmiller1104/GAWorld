"""M3 RED tests — real-harness adapter bridging orchestrator.run_legs to the
real execution path (experiment.py / launch.py / canary.py).

Known gap (MCF_COMPLETION_PLAN.md M3): ``orchestrator.run_legs`` calls its
harness as ``harness(config, run_dir, *, seed=...)``, while the real vehicle
(``experiment.run_experiment``/``_execute_arm``) takes ``harness(config,
run_dir)`` with no seed kwarg, and expects the run dir to be an isolated
``launch.create_run`` directory with canary budget controls installed.

Spec (docs/MCF_GAWORLD.md §5, MCF_COMPLETION_PLAN.md M3) — a pure adapter in
``benchmark/mcf/real_harness.py``:
- ``make_real_harness(*, seed, limits=None, execute=None)`` returns a callable
  with the run_legs signature ``(config, run_dir)`` (also tolerant of the
  ``seed=`` keyword run_legs passes);
- creates an isolated run dir via ``launch.create_run`` (restart-per-run: no
  checkpoint resume, every call a fresh run_id);
- installs canary budget controls (``launch.install_canary`` +
  ``canary.wrap_call_llm``) with the leg limits threaded into the canary
  config, and restores the unwrapped ``call_llm`` afterwards;
- threads the seed through the ONLY wired seeding mechanism —
  ``CONFIG["random_seed"]`` (generative_city_sim._resolve_run_seed /
  gaworld parallel runner) — never an invented one;
- passes the config patch into the arm execution untouched;
- returns a run_legs-compatible result dict (ok / usage / measured_usage
  passthrough for measured-vs-estimated accounting);
- an inner-execution crash preserves the partial (in_progress) manifest and
  reports ``ok: False``.

All tests use an INJECTED stub inner execution — zero LLM calls, zero
inference anywhere in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.mcf.real_harness import make_real_harness

# ---------------------------------------------------------------------------
# stub inner execution (offline — no inference)
# ---------------------------------------------------------------------------


class SpyExecute:
    """Records how the adapter invoked the real arm-execution path."""

    def __init__(
        self,
        result: dict[str, Any] | None = None,
        raise_exc: Exception | None = None,
        baseline_llm: Any = None,
    ):
        self.baseline_llm = baseline_llm
        self.calls: list[dict[str, Any]] = []
        self.result = (
            result
            if result is not None
            else {"ok": True, "usage": {"calls": 1, "tokens_in": 10, "tokens_out": 5}}
        )
        self.raise_exc = raise_exc

    def __call__(self, config: dict, run_dir: Path, *, seed: int) -> dict:
        import gaworld.llm.providers as providers
        import gaworld.settings as settings

        self.calls.append(
            {
                "config": json.loads(json.dumps(config)),
                "run_dir": Path(run_dir),
                "seed": seed,
                # state observable DURING the run (canary installed? seed applied?)
                "call_llm_is_wrapped": providers.call_llm is not self.baseline_llm,
                "config_canary": json.loads(json.dumps((config or {}).get("canary", {}))),
                "random_seed_during_run": settings.CONFIG.get("random_seed"),
            }
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return dict(self.result)


@pytest.fixture()
def _clean_canary_state():
    """Snapshot call_llm and CONFIG.random_seed around every test."""
    import gaworld.llm.providers as providers
    import gaworld.settings as settings

    original_llm = providers.call_llm
    original_seed = settings.CONFIG.get("random_seed")
    yield
    providers.call_llm = original_llm
    if original_seed is None:
        settings.CONFIG.pop("random_seed", None)
    else:
        settings.CONFIG["random_seed"] = original_seed


# ---------------------------------------------------------------------------
# adapter factory shape and signature
# ---------------------------------------------------------------------------


class TestAdapterFactory:
    def test_make_real_harness_exists_and_is_callable_factory(self):
        assert callable(make_real_harness)

    def test_produced_harness_takes_config_and_run_dir_positionally(self, tmp_path, _clean_canary_state):
        harness = make_real_harness(seed=7, execute=SpyExecute())
        result = harness({"a.b": 1}, tmp_path / "leg_000")  # NO seed kwarg — experiment.Harness contract
        assert isinstance(result, dict)

    def test_produced_harness_tolerates_run_legs_seed_keyword(self, tmp_path, _clean_canary_state):
        harness = make_real_harness(seed=7, execute=SpyExecute())
        result = harness({"a.b": 1}, tmp_path / "leg_000", seed=7)  # run_legs passes seed=
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# isolated run dir via launch.create_run; restart-per-run
# ---------------------------------------------------------------------------


class TestIsolatedRunDir:
    def test_inner_execution_receives_fresh_isolated_subdir_with_manifest(
        self, tmp_path, _clean_canary_state
    ):
        spy = SpyExecute()
        harness = make_real_harness(seed=7, execute=spy)
        leg_dir = tmp_path / "leg_000"
        leg_dir.mkdir(parents=True)
        harness({"a.b": 1}, leg_dir)
        run_dir = spy.calls[0]["run_dir"]
        assert run_dir.parent == leg_dir  # isolated INSIDE the leg dir, not the leg dir itself
        assert run_dir.name != "" and run_dir != leg_dir
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest.get("run_id") == run_dir.name

    def test_two_calls_restart_per_run_distinct_run_ids(self, tmp_path, _clean_canary_state):
        spy = SpyExecute()
        harness = make_real_harness(seed=7, execute=spy)
        r1 = harness({"a.b": 1}, tmp_path / "leg_000")
        r2 = harness({"a.b": 1}, tmp_path / "leg_000")  # same leg dir, second attempt
        ids = {spy.calls[0]["run_dir"].name, spy.calls[1]["run_dir"].name}
        assert len(ids) == 2, "restart-per-run: every call must be a fresh run_id"
        assert r1.get("run_id") != r2.get("run_id")

    def test_no_checkpoint_resume_flags_anywhere(self, tmp_path, _clean_canary_state):
        spy = SpyExecute()
        harness = make_real_harness(seed=7, execute=spy)
        harness({"a.b": 1}, tmp_path / "leg_000")
        run_dir = spy.calls[0]["run_dir"]
        leftover = [
            p.name for p in run_dir.rglob("*") if "checkpoint" in p.name.lower() or "resume" in p.name.lower()
        ]
        assert leftover == []
        config_passed = spy.calls[0]["config"]
        assert not any("checkpoint" in str(k).lower() or "resume" in str(k).lower() for k in config_passed)


# ---------------------------------------------------------------------------
# seed threading through the wired CONFIG["random_seed"] path
# ---------------------------------------------------------------------------


class TestSeedThreading:
    def test_seed_reaches_inner_execution(self, tmp_path, _clean_canary_state):
        spy = SpyExecute()
        harness = make_real_harness(seed=4242, execute=spy)
        harness({"a.b": 1}, tmp_path / "leg_000")
        assert spy.calls[0]["seed"] == 4242

    def test_seed_applied_via_config_random_seed_during_run(self, tmp_path, _clean_canary_state):
        import gaworld.settings as settings

        settings.CONFIG.pop("random_seed", None)
        spy = SpyExecute()
        harness = make_real_harness(seed=4242, execute=spy)
        harness({"a.b": 1}, tmp_path / "leg_000")
        assert spy.calls[0]["random_seed_during_run"] == 4242, (
            "the adapter must apply the seed through the wired CONFIG['random_seed'] path"
        )

    def test_seed_recorded_in_run_manifest(self, tmp_path, _clean_canary_state):
        spy = SpyExecute()
        harness = make_real_harness(seed=4242, execute=spy)
        harness({"a.b": 1}, tmp_path / "leg_000")
        manifest = json.loads((spy.calls[0]["run_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest.get("seed") == 4242


# ---------------------------------------------------------------------------
# canary budget controls installed + restored
# ---------------------------------------------------------------------------


class TestCanaryInstallation:
    def test_call_llm_wrapped_during_execution(self, tmp_path, _clean_canary_state):
        import gaworld.llm.providers as providers

        spy = SpyExecute(baseline_llm=providers.call_llm)
        harness = make_real_harness(seed=7, execute=spy)
        harness({"a.b": 1}, tmp_path / "leg_000")
        assert spy.calls[0]["call_llm_is_wrapped"] is True, (
            "canary budget controls (install_canary + wrap_call_llm) must be active during arm execution"
        )

    def test_call_llm_restored_after_execution(self, tmp_path, _clean_canary_state):
        import gaworld.llm.providers as providers

        original = providers.call_llm
        harness = make_real_harness(seed=7, execute=SpyExecute())
        harness({"a.b": 1}, tmp_path / "leg_000")
        assert providers.call_llm is original, "adapter must restore the unwrapped call_llm afterwards"

    def test_leg_limits_threaded_into_canary_config(self, tmp_path, _clean_canary_state):
        limits = {
            "max_calls": 1234,
            "max_tokens": 5_000_000,
            "max_elapsed_seconds": 999,
            "max_error_rate": 0.02,
        }
        spy = SpyExecute()
        harness = make_real_harness(seed=7, limits=limits, execute=spy)
        harness({"a.b": 1}, tmp_path / "leg_000")
        canary_cfg = spy.calls[0]["config_canary"]
        assert canary_cfg.get("enabled") is True
        assert canary_cfg.get("max_calls") == 1234
        assert canary_cfg.get("max_tokens") == 5_000_000
        assert canary_cfg.get("max_elapsed_seconds") == 999
        assert canary_cfg.get("max_error_rate") == 0.02

    def test_config_patch_not_polluted_for_the_arm(self, tmp_path, _clean_canary_state):
        """canary enablement rides on the config the adapter builds for
        install_canary, but the arm's own config patch stays clean."""
        limits = {"max_calls": 1234}
        spy = SpyExecute()
        harness = make_real_harness(seed=7, limits=limits, execute=spy)
        harness({"a.b": 1}, tmp_path / "leg_000")
        assert spy.calls[0]["config"] == {"a.b": 1}, (
            "the config patch handed to arm execution must stay clean (no canary/seed keys injected)"
        )


# ---------------------------------------------------------------------------
# result compatibility with run_legs (status/usage/measured accounting)
# ---------------------------------------------------------------------------


class TestResultContract:
    def test_ok_and_usage_passthrough(self, tmp_path, _clean_canary_state):
        spy = SpyExecute(result={"ok": True, "usage": {"calls": 3, "tokens_in": 30, "tokens_out": 15}})
        harness = make_real_harness(seed=7, execute=spy)
        result = harness({"a.b": 1}, tmp_path / "leg_000")
        assert result["ok"] is True
        assert result["usage"]["calls"] == 3
        assert result["usage"]["tokens_in"] == 30
        assert result["usage"]["tokens_out"] == 15

    def test_measured_usage_passthrough_for_token_honesty(self, tmp_path, _clean_canary_state):
        measured = {"prompt_tokens": 111, "completion_tokens": 22}  # _measured_usage shape
        spy = SpyExecute(
            result={
                "ok": True,
                "usage": {"calls": 1, "tokens_in": 0, "tokens_out": 0},
                "measured_usage": measured,
            }
        )
        harness = make_real_harness(seed=7, execute=spy)
        result = harness({"a.b": 1}, tmp_path / "leg_000")
        assert result["measured_usage"] == measured, (
            "measured-vs-estimated accounting requires the measured usage to reach run_legs untouched"
        )

    def test_run_id_and_run_dir_reported(self, tmp_path, _clean_canary_state):
        spy = SpyExecute()
        harness = make_real_harness(seed=7, execute=spy)
        result = harness({"a.b": 1}, tmp_path / "leg_000")
        assert result["run_id"] and result["run_dir"]
        assert Path(result["run_dir"]).is_dir()

    def test_inner_crash_reports_failure_and_preserves_partial_manifest(self, tmp_path, _clean_canary_state):
        spy = SpyExecute(raise_exc=RuntimeError("arm exploded"))
        harness = make_real_harness(seed=7, execute=spy)
        result = harness({"a.b": 1}, tmp_path / "leg_000")
        assert result.get("ok") is False
        assert "RuntimeError" in str(result.get("error") or result.get("stop_reason") or "")
        manifest = json.loads((spy.calls[0]["run_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["outcome"] == "in_progress", "a crash must leave the partial manifest as evidence"

    def test_inner_failure_outcome_finalized_in_manifest(self, tmp_path, _clean_canary_state):
        spy = SpyExecute(result={"ok": False, "error": "arm reported failure", "usage": {}})
        harness = make_real_harness(seed=7, execute=spy)
        result = harness({"a.b": 1}, tmp_path / "leg_000")
        assert result.get("ok") is False
        manifest = json.loads((spy.calls[0]["run_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["outcome"] == "incomplete"
