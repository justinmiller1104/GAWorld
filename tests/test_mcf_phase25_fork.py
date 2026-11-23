"""Phase 2.5 RED tests — checkpoint fork launcher (TASK-3590).

``fork_arms_from_checkpoint`` does not exist yet (neither in
``benchmark/mcf/experiment.py`` nor anywhere else): these tests pin the
contract BEFORE it is written, per the Phase 2.5 investigation Part 1
(checkpoint-fork + pre-trigger LLM replay = the smallest robust
shared-history mechanism).

Contracts pinned:

  1. Both arms restore the SAME checkpoint dir (the fork source), each
     into its own separate run directory — run isolation preserved.
  2. News cache frozen: the checkpoint's news cache file is copied into
     both run dirs (or background news disabled) so wall-clock headlines
     cannot differ between arms.
  3. Replay installed in the non-baseline arm(s) until the intervention
     trigger tick; the baseline arm samples live (no replay).
  4. Both arms' manifests record fork provenance: the checkpoint dir and
     the replay window (trigger).

Deterministic stub harnesses; no LLM, no network, no real simulation.
"""

import json
from pathlib import Path

from benchmark.mcf.experiment import fork_arms_from_checkpoint

TRIGGER = {"day": 2, "time": "10:00"}


def _fake_checkpoint(tmp_path: Path) -> Path:
    """A stand-in day-boundary checkpoint dir with a news cache to freeze."""
    ckpt = tmp_path / "checkpoints" / "day_0001"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "manifest.json").write_text(json.dumps({"day": 1}), encoding="utf-8")
    (ckpt / "news_cache.json").write_text(json.dumps({"headlines": ["frozen"]}), encoding="utf-8")
    return ckpt


def _recording_harness(log: list, tag: str):
    def harness(config, run_dir):
        log.append({"tag": tag, "config": dict(config), "run_dir": str(run_dir)})
        (Path(run_dir) / "agent_1.jsonl").write_text("", encoding="utf-8")
        return {"ok": True}

    return harness


def _fork(tmp_path: Path, baseline_cfg=None, intervention_cfg=None):
    log = []
    ckpt = _fake_checkpoint(tmp_path)
    arms = {
        "baseline": {
            "config": baseline_cfg or {},
            "harness": _recording_harness(log, "baseline"),
            "trigger": dict(TRIGGER),
        },
        "intervention": {
            "config": intervention_cfg or {},
            "harness": _recording_harness(log, "intervention"),
            "trigger": dict(TRIGGER),
        },
    }
    result = fork_arms_from_checkpoint(ckpt, arms, run_root=tmp_path / "fork")
    return ckpt, result, log


# ---------------------------------------------------------------------------
# Contract 1 — same checkpoint source, separate run dirs
# ---------------------------------------------------------------------------


def test_both_arms_restore_the_same_checkpoint(tmp_path):
    """Each arm's harness-visible config must point at the SAME checkpoint
    dir (the shared pre-intervention history source)."""
    _, _result, log = _fork(tmp_path)
    by_tag = {rec["tag"]: rec for rec in log}
    # both arms name the same checkpoint dir as their restore source
    assert by_tag["baseline"]["config"].get("resume_from_checkpoint") == by_tag["intervention"]["config"].get(
        "resume_from_checkpoint"
    )
    assert by_tag["baseline"]["config"].get("resume_from_checkpoint"), (
        "restore source missing from baseline config"
    )


def test_arms_get_separate_run_dirs(tmp_path):
    """Run isolation: two distinct run directories under the fork root."""
    _, result, log = _fork(tmp_path)
    run_dirs = {rec["run_dir"] for rec in log}
    assert len(run_dirs) == 2, "each arm must run in its own directory"
    assert all(Path(d).is_dir() for d in run_dirs)
    assert result["baseline"]["run_dir"] != result["intervention"]["run_dir"]


# ---------------------------------------------------------------------------
# Contract 2 — news cache frozen
# ---------------------------------------------------------------------------


def test_news_cache_copied_into_both_run_dirs(tmp_path):
    """The checkpoint's frozen news cache is copied into each arm's run dir
    so both arms' prompts see identical background headlines."""
    _, result, _ = _fork(tmp_path)
    for arm in ("baseline", "intervention"):
        run_dir = Path(result[arm]["run_dir"])
        assert (run_dir / "news_cache.json").exists(), f"{arm}: news cache not frozen into the run dir"


# ---------------------------------------------------------------------------
# Contract 3 — replay in non-baseline arms until the trigger
# ---------------------------------------------------------------------------


def test_replay_installed_in_intervention_arm_only(tmp_path):
    """The intervention arm runs under the LLM replay wrapper until the
    trigger; the baseline arm samples live."""
    _, result, _ = _fork(tmp_path)
    assert result["intervention"].get("replay"), "intervention arm must record an installed replay wrapper"
    assert not result["baseline"].get("replay"), "baseline arm must sample live (no replay)"
    window = result["intervention"]["replay"]
    assert window["stop"]["day"] == TRIGGER["day"] and window["stop"]["time"] == TRIGGER["time"]


# ---------------------------------------------------------------------------
# Contract 4 — fork provenance in both arms' manifests
# ---------------------------------------------------------------------------


def test_manifests_record_fork_provenance(tmp_path):
    """Each arm's run manifest carries: the fork checkpoint dir + the replay
    window (trigger) — auditable provenance for the shared history."""
    ckpt, result, _ = _fork(tmp_path)
    for arm in ("baseline", "intervention"):
        manifest = json.loads(
            (Path(result[arm]["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8")
        )
        fork = manifest.get("fork")
        assert fork, f"{arm}: manifest missing fork provenance"
        assert fork["checkpoint"] == str(ckpt)
        assert fork["replay_window"] == {"start": "checkpoint", "stop": dict(TRIGGER)}
