"""Phase 2.3 RED tests — end-to-end intervention integration with the ACTUAL
simulator, driven by the zero-inference dry harness (TASK-3552).

The Phase 2.2 framework (``benchmark/mcf/interventions.py`` spec schema +
``benchmark/mcf/compare.py`` outcome comparison) must now be connected by a
runner/audit glue module — ``benchmark/mcf/experiment.py`` per the Phase 2.2
placement (RESULT-3541 §Recommended placement) — that executes TWO actual
dry-run arms (baseline and intervention) and audits what really happened.

These tests are deterministic, zero-inference and network-free. The module
under test takes the execution vehicle as an injected ``harness`` callable
(the real vehicle being the Phase 2.1 canary ``dry_run=True`` stub pattern of
``run_phase21_dry.py``: full sim pipeline, ``call_llm`` stubbed, 0 provider
calls). The tests supply an in-process deterministic harness that writes the
REAL run-directory file shapes the simulator produces (``run_manifest.json``,
``config.json``, ``agent_<id>.jsonl`` episode files, ``intervention_executions.jsonl``
and per-agent state records) so every assertion below reads actual run outputs,
never mocked numbers.

Contracts pinned here (TASK-PH23-RED-SPEC):

  1.  run_experiment executes TWO actual dry-run arms — independent run dirs
      with manifests and real episode files; zero LLM calls.
  2.  Execution audit: per-arm audit record with intervention name/type,
      intended trigger time, actual execution time (from the run's own
      records), targets, requested change, applied yes/no, before/after
      state. A declared intervention whose trigger never fires →
      status NOT_APPLIED (never "success").
  3.  Baseline purity: the baseline arm's config contains no intervention
      delta; its audit says none.
  4.  Timing correctness: no intervention effect before the trigger time;
      effect visible at/after the trigger, asserted on actual episode files.
  5.  Target selection: a set_agent_state delta changes the targeted agent's
      state as intended; an untargeted agent's state is unchanged at the
      trigger tick.
  6.  Neutral control: a no-op (identity delta) intervention yields exactly
      zero differences in a deterministic dry run (same seed).
  7.  Failed ops reported honestly (FAILED status with reason).
  8.  Partial/interrupted arms: missing final manifest or truncated episodes
      → arm reported incomplete; comparison marks the affected outcomes
      MISSING rather than fabricating.
  9.  Outcome extraction: comparison diffs computed from the arms' ACTUAL
      episode files; identical inputs → identical report (determinism).
 10.  Spec translation: intervention specs translate into operations the sim
      actually supports (policy_events entries and/or config deltas);
      unsupported operations are rejected at validation, before any run
      directory is created.
"""

import copy
import json
from pathlib import Path

import pytest

from benchmark.mcf.compare import NEUTRAL_VERDICTS, compare_arms
from benchmark.mcf.interventions import InterventionSpecError, apply_intervention, validate_intervention_spec

# RED: the runner/audit glue does not exist yet — everything below fails with
# ModuleNotFoundError until benchmark/mcf/experiment.py is written.
from benchmark.mcf.experiment import run_experiment  # noqa: E402

# ---------------------------------------------------------------------------
# Deterministic zero-inference dry harness (stands in for the Phase 2.1
# canary dry_run vehicle; same file contract, no LLM, no network)
# ---------------------------------------------------------------------------

AGENT_IDS = [4, 7]
SIM_DAYS = 3
TICK_TIMES = ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00"]

BASE_CONFIG = {
    "random_seed": 42,
    "agent_ids": list(AGENT_IDS),
    "sim_days": SIM_DAYS,
    "time_step_minutes": 60,
    "fos_fast_mode": True,
}


def _minutes(time_str):
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


class DryHarness:
    """In-process deterministic execution vehicle.

    Writes the run-directory file shapes the actual simulator produces:
    ``config.json``, one ``agent_<id>.jsonl`` episode file per agent, one
    ``intervention_executions.jsonl`` audit-source file, and per-agent state
    records under ``agent_state/``. All LLM-driven behaviour is stubbed by a
    fixed deterministic rule, so ``llm_calls`` is always 0.
    """

    def __init__(self, *, seed=42, ok=True, truncate_at_tick=None):
        self.seed = seed
        self.ok = ok  # False → the run "crashes" mid-way (incomplete arm)
        self.truncate_at_tick = truncate_at_tick  # int index → partial episodes
        self.llm_calls = 0

    def __call__(self, config, run_dir):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        agent_ids = config.get("agent_ids", AGENT_IDS)
        sim_days = config.get("sim_days", SIM_DAYS)
        events = [e for e in config.get("policy_events", []) if isinstance(e, dict)]
        exec_records = []
        state_records = []
        episodes_written = 0
        for day in range(1, sim_days + 1):
            for idx, time_str in enumerate(TICK_TIMES):
                if self.truncate_at_tick is not None and (day - 1) * len(TICK_TIMES) + idx >= self.truncate_at_tick:
                    break
                marker = self._active_marker(events, day, time_str)
                for agent_id in agent_ids:
                    activity = f"routine_{self._slot(time_str)}"
                    if marker is not None:
                        activity = f"responsive_{marker}"
                    self._append_episode(run_dir / f"agent_{agent_id}.jsonl", agent_id, day, time_str, activity)
                    episodes_written += 1
                for event in events:
                    if self._is_state_event(event) and (event["day"], _minutes(event["time"])) == (
                        day,
                        _minutes(time_str),
                    ):
                        for target in event.get("targets", [event["kwargs"].get("agent_id")]):
                            exec_records.append(self._apply_state(target, event, day, time_str))
        self._write_jsonl(run_dir / "intervention_executions.jsonl", exec_records)
        state_by_agent = {}
        for rec in state_records:
            state_by_agent.setdefault(rec["agent_id"], []).append(rec)
        for agent_id, recs in state_by_agent.items():
            self._write_jsonl(run_dir / "agent_state" / f"agent_{agent_id}.jsonl", recs)
        self.llm_calls += 0  # dry vehicle: ZERO inference by construction
        return {"llm_calls": self.llm_calls, "episodes": episodes_written, "ok": self.ok}

    # -- deterministic activity rules --------------------------------------

    @staticmethod
    def _slot(time_str):
        m = _minutes(time_str)
        if m < 12 * 60:
            return "morning"
        if m < 18 * 60:
            return "afternoon"
        return "evening"

    @staticmethod
    def _is_state_event(event):
        kwargs = event.get("kwargs", {})
        return "agent_id" in kwargs and "metric" in kwargs

    @staticmethod
    def _active_marker(events, day, time_str):
        """Marker events take effect at and after their trigger tick."""
        now = (day, _minutes(time_str))
        for event in events:
            kwargs = event.get("kwargs", {})
            if "marker" in kwargs and not ("agent_id" in kwargs and "metric" in kwargs):
                if now >= (event["day"], _minutes(event["time"])):
                    return kwargs["marker"]
        return None

    def _apply_state(self, agent_id, event, day, time_str):
        kwargs = event["kwargs"]
        current = 0.60  # deterministic stub state
        if agent_id not in AGENT_IDS:
            record = {
                "name": event.get("name"),
                "operation": "set_agent_state",
                "day": day,
                "time": time_str,
                "targets": [agent_id],
                "applied": False,
                "reason": f"unknown agent {agent_id}",
            }
            return record
        record = {
            "name": event.get("name"),
            "operation": "set_agent_state",
            "day": day,
            "time": time_str,
            "targets": [agent_id],
            "metric": kwargs["metric"],
            "applied": True,
            "before": current,
            "after": kwargs["value"],
        }
        return record

    # -- file writers (the real record shapes) ------------------------------

    @staticmethod
    def _append_episode(path, agent_id, day, time_str, activity):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"agent_id": agent_id, "day": day, "time": time_str, "final_activity": activity},
                    ensure_ascii=False,
                )
                + "\n"
            )

    @staticmethod
    def _write_jsonl(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_episodes(run_dir):
    """Read a run's actual episode files: {(agent_id, day, time): activity}."""
    records = {}
    for path in sorted(Path(run_dir).glob("agent_*.jsonl")):
        if path.parent.name == "agent_state":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            records[(ep["agent_id"], ep["day"], ep["time"])] = ep["final_activity"]
    return records


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

POLICY_EVENT_SPEC = {
    "name": "community_notice",
    "trigger": {"day": 2, "time": "10:00"},
    "operation": "policy_event",
    "kwargs": {"description": "社区发布节能倡议", "marker": "saver"},
    "targets": list(AGENT_IDS),
}

SET_AGENT_STATE_SPEC = {
    "name": "mood_boost",
    "trigger": {"day": 2, "time": "10:00"},
    "operation": "set_agent_state",
    "kwargs": {"metric": "emotion", "value": 0.95, "agent_id": 7},
    "targets": [7],
}

NEUTRAL_SPEC = {
    "name": "identity_delta",
    "trigger": {"day": 2, "time": "10:00"},
    "operation": "update_config",
    "kwargs": {"key": "fos_fast_mode", "value": True},  # already the base value
    "targets": ["env"],
}


# ---------------------------------------------------------------------------
# Contract 1 — two actual dry-run arms, isolated, zero inference
# ---------------------------------------------------------------------------


def test_two_real_dry_run_arms_with_independent_dirs_and_zero_llm(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(POLICY_EVENT_SPEC), DryHarness(), run_root=tmp_path)
    base_arm, int_arm = result["baseline"], result["intervention"]
    assert Path(base_arm["run_dir"]) != Path(int_arm["run_dir"])
    for arm in (base_arm, int_arm):
        run_dir = Path(arm["run_dir"])
        assert (run_dir / "config.json").exists()
        assert (run_dir / "run_manifest.json").exists()
        episodes = _read_episodes(run_dir)
        assert episodes, "arm must produce real episode files, not fabricated outputs"
        assert len(episodes) == len(AGENT_IDS) * SIM_DAYS * len(TICK_TIMES)
        assert arm["summary"]["llm_calls"] == 0, "dry harness must issue zero LLM calls"
    # the manifest of a completed arm says so
    manifest = json.loads((Path(int_arm["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outcome"] == "completed"


# ---------------------------------------------------------------------------
# Contract 2 — execution audit with NOT_APPLIED for a never-firing trigger
# ---------------------------------------------------------------------------


def test_audit_record_captures_intended_vs_actual_execution(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(SET_AGENT_STATE_SPEC), DryHarness(), run_root=tmp_path)
    audit = result["intervention"]["audit"]
    assert len(audit) == 1
    rec = audit[0]
    assert rec["name"] == SET_AGENT_STATE_SPEC["name"]
    assert rec["operation"] == "set_agent_state"
    assert rec["intended_time"] == {"day": 2, "time": "10:00"}
    assert rec["actual_time"] == {"day": 2, "time": "10:00"}, "actual time must come from the run's own records"
    assert rec["targets"] == [7]
    assert rec["requested"] == {"metric": "emotion", "value": 0.95}
    assert rec["applied"] is True
    assert rec["before"] == 0.60 and rec["after"] == 0.95
    assert rec["status"] == "applied"


def test_trigger_that_never_fires_is_not_applied_not_success(tmp_path):
    late_spec = copy.deepcopy(SET_AGENT_STATE_SPEC)
    late_spec["trigger"] = {"day": 9, "time": "10:00"}  # beyond sim_days=3
    result = run_experiment(copy.deepcopy(BASE_CONFIG), late_spec, DryHarness(), run_root=tmp_path)
    audit = result["intervention"]["audit"]
    assert len(audit) == 1
    rec = audit[0]
    assert rec["applied"] is False
    assert rec["status"] == "NOT_APPLIED"
    assert rec["actual_time"] is None
    assert rec["status"] != "success"


# ---------------------------------------------------------------------------
# Contract 3 — baseline purity
# ---------------------------------------------------------------------------


def test_baseline_arm_is_pure_no_delta_no_audit(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(SET_AGENT_STATE_SPEC), DryHarness(), run_root=tmp_path)
    base_arm, int_arm = result["baseline"], result["intervention"]
    assert base_arm["audit"] == [], "baseline arm must audit no intervention"
    base_cfg = json.loads((Path(base_arm["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    assert not base_cfg.get("policy_events"), "baseline config must contain no intervention delta"
    int_cfg = json.loads((Path(int_arm["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    assert int_cfg.get("policy_events"), "intervention arm must carry the translated delta"


# ---------------------------------------------------------------------------
# Contract 4 — timing correctness on actual episode records
# ---------------------------------------------------------------------------


def test_no_effect_before_trigger_effect_at_and_after(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(POLICY_EVENT_SPEC), DryHarness(), run_root=tmp_path)
    base_eps = _read_episodes(result["baseline"]["run_dir"])
    int_eps = _read_episodes(result["intervention"]["run_dir"])
    trigger = (2, _minutes("10:00"))
    pre = [k for k in base_eps if (k[1], _minutes(k[2])) < trigger]
    post = [k for k in base_eps if (k[1], _minutes(k[2])) >= trigger]
    assert pre and post
    assert all(base_eps[k] == int_eps[k] for k in pre), "no intervention effect before the trigger"
    assert any(base_eps[k] != int_eps[k] for k in post), "effect must be visible at/after the trigger"
    # first differing tick is exactly the trigger tick
    first_diff = min((k for k in post if base_eps[k] != int_eps[k]), key=lambda k: (k[1], _minutes(k[2])))
    assert (first_diff[1], _minutes(first_diff[2])) == trigger


# ---------------------------------------------------------------------------
# Contract 5 — target selection
# ---------------------------------------------------------------------------


def test_targeted_agent_changed_untargeted_untouched(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(SET_AGENT_STATE_SPEC), DryHarness(), run_root=tmp_path)
    int_dir = Path(result["intervention"]["run_dir"])
    state_7 = _read_jsonl(int_dir / "agent_state" / "agent_7.jsonl")
    assert state_7, "targeted agent must have state records"
    rec = state_7[0]
    assert rec["metric"] == "emotion" and rec["before"] == 0.60 and rec["after"] == 0.95
    assert (rec["day"], rec["time"]) == (2, "10:00")
    # untargeted agent has NO state records at the trigger tick
    assert not (int_dir / "agent_state" / "agent_4.jsonl").exists()
    base_eps = _read_episodes(result["baseline"]["run_dir"])
    int_eps = _read_episodes(result["intervention"]["run_dir"])
    assert base_eps[(4, 2, "10:00")] == int_eps[(4, 2, "10:00")], "untargeted agent unchanged at trigger tick"


# ---------------------------------------------------------------------------
# Contract 6 — neutral (identity-delta) control: exactly zero difference
# ---------------------------------------------------------------------------


def test_neutral_control_yields_exactly_zero_difference(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(NEUTRAL_SPEC), DryHarness(), run_root=tmp_path)
    report = result["comparison"]
    for day, diff in report["per_day"].items():
        assert diff["max_abs_diff"] == 0.0, f"identity delta must change nothing on day {day}"
    assert report["effect"]["verdict"] in NEUTRAL_VERDICTS


# ---------------------------------------------------------------------------
# Contract 7 — failed ops reported honestly
# ---------------------------------------------------------------------------


def test_failed_op_reported_with_reason(tmp_path):
    bad_spec = copy.deepcopy(SET_AGENT_STATE_SPEC)
    bad_spec["kwargs"]["agent_id"] = 99  # not in the roster
    bad_spec["targets"] = [99]
    result = run_experiment(copy.deepcopy(BASE_CONFIG), bad_spec, DryHarness(), run_root=tmp_path)
    audit = result["intervention"]["audit"]
    assert len(audit) == 1
    rec = audit[0]
    assert rec["applied"] is False
    assert rec["status"] == "FAILED"
    assert rec.get("reason"), "a FAILED record must carry the reason"


# ---------------------------------------------------------------------------
# Contract 8 — partial/interrupted arms reported incomplete, outcomes MISSING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "harness",
    [
        DryHarness(ok=False),  # crashed after starting: manifest stays in_progress
        DryHarness(ok=False, truncate_at_tick=7),  # truncated episodes
    ],
    ids=["no_final_manifest", "truncated_episodes"],
)
def test_incomplete_arm_reported_and_outcomes_missing(tmp_path, harness):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(POLICY_EVENT_SPEC), harness, run_root=tmp_path)
    int_arm = result["intervention"]
    assert int_arm["complete"] is False, "interrupted arm must be reported incomplete"
    manifest = json.loads((Path(int_arm["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outcome"] != "completed"
    report = result["comparison"]
    status = report["data_status"]["intervention"]
    assert status == "MISSING", f"insufficient data must be MISSING, got {status!r}"
    assert report["effect"]["verdict"] not in NEUTRAL_VERDICTS, "MISSING data is not evidence of no effect"


# ---------------------------------------------------------------------------
# Contract 9 — outcome extraction from actual files + determinism
# ---------------------------------------------------------------------------


def test_comparison_matches_direct_episode_diff(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(POLICY_EVENT_SPEC), DryHarness(), run_root=tmp_path)
    report = result["comparison"]
    # cross-check against compare_arms run on the very same on-disk dirs
    direct = compare_arms(result["baseline"]["run_dir"], result["intervention"]["run_dir"], intervention_day=2)
    assert report["per_day"] == direct["per_day"]
    base_eps = _read_episodes(result["baseline"]["run_dir"])
    int_eps = _read_episodes(result["intervention"]["run_dir"])
    day2_mismatch = sum(
        1
        for k in base_eps
        if k[1] == 2 and base_eps[k] != int_eps[k]
    )
    assert report["per_day"][2]["max_abs_diff"] == pytest.approx(day2_mismatch / len(base_eps) if day2_mismatch else 0.0)


def test_identical_inputs_identical_report(tmp_path):
    r1 = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(POLICY_EVENT_SPEC), DryHarness(), run_root=tmp_path / "a")
    r2 = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(POLICY_EVENT_SPEC), DryHarness(), run_root=tmp_path / "b")

    def _strip(report):
        report = copy.deepcopy(report)
        report.pop("intervention_dir", None)
        report.pop("mcf_validation", None)
        return report

    assert _strip(r1["comparison"]) == _strip(r2["comparison"]), "same seed + same delta → identical report"
    assert r1["baseline"]["audit"] == r2["baseline"]["audit"]
    assert r1["intervention"]["audit"] == r2["intervention"]["audit"]


# ---------------------------------------------------------------------------
# Contract 10 — spec translation into sim-supported operations
# ---------------------------------------------------------------------------


def test_set_agent_state_translates_to_policy_event_entry(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(SET_AGENT_STATE_SPEC), DryHarness(), run_root=tmp_path)
    cfg = json.loads((Path(result["intervention"]["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    events = [e for e in cfg.get("policy_events", []) if e.get("name") == SET_AGENT_STATE_SPEC["name"]]
    assert events, "set_agent_state must translate into a policy_events entry the sim matches per tick"
    event = events[0]
    assert (event["day"], event["time"]) == (2, "10:00")
    assert event["kwargs"]["metric"] == "emotion" and event["kwargs"]["value"] == 0.95


def test_update_config_translates_to_config_delta(tmp_path):
    spec = {
        "name": "rainy_afternoon",
        "trigger": {"day": 2, "time": "10:00"},
        "operation": "update_config",
        "kwargs": {"key": "test.marker", "value": "rain"},
        "targets": ["env"],
    }
    result = run_experiment(copy.deepcopy(BASE_CONFIG), spec, DryHarness(), run_root=tmp_path)
    int_cfg = json.loads((Path(result["intervention"]["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    assert int_cfg.get("test", {}).get("marker") == "rain", "update_config must land as a real config delta"
    base_cfg = json.loads((Path(result["baseline"]["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    assert "test" not in base_cfg


def test_unsupported_operation_rejected_before_any_run_dir(tmp_path):
    spec = copy.deepcopy(SET_AGENT_STATE_SPEC)
    spec["operation"] = "add_agent"  # unimplemented in the kernel
    with pytest.raises(InterventionSpecError, match="operation"):
        run_experiment(copy.deepcopy(BASE_CONFIG), spec, DryHarness(), run_root=tmp_path)
    assert not list(Path(tmp_path).rglob("run_manifest.json")), "no run dirs may be created for an invalid spec"


# ---------------------------------------------------------------------------
# sanity: the Phase 2.2 primitives the runner builds on keep their contract
# ---------------------------------------------------------------------------


def test_delta_builder_stays_pure():
    config = copy.deepcopy(BASE_CONFIG)
    frozen = copy.deepcopy(config)
    new_config = apply_intervention(validate_intervention_spec(POLICY_EVENT_SPEC), config)
    assert config == frozen, "apply_intervention must never mutate the base config"
    assert new_config is not config
