"""RED regression tests for the verified pre-merge blockers (repair pass).

Blockers covered (from the repair-premerge RED spec):

* H1 - the Phase-2.0 realism engine must not change DEFAULT-run behaviour:
  the gated changes (extra 00:00 head slot, Phase-1.9 head-slot conversion,
  {"kind":"span"} episode rows, actions token budget override) must only
  fire when behavior.mcf_realism is explicitly enabled (GAWorld
  comparability rule).
* H5 - the interrupt-cooldown registry keys on agent["agent_id"], but
  pipeline agents are shaped {"id": ...} (gaworld/sim/_stages.py), so the
  cooldown never activates for real agents.
* H4 - mcf_api experiment paths join exp_id unvalidated, so a crafted id
  (../../evil) reads/writes outside output/mcf.
* M1 - compare_arms with intervention_time classifies the pre/post window
  by time-of-day only, ignoring the record's day, so pre-trigger
  divergence after the trigger clock-time leaks into the post window
  (docstring contract: a record is post when its day is after
  intervention_day OR it is on the trigger day at/after the time).
* M2 - an UNRELIABLE arm (below MIN_COMPARISON_DAYS/POPULATION) falls
  through to no_effect; insufficient data is not evidence of no effect
  (NEUTRAL_VERDICTS invariant, compare.py:33).
* M5 - the launch path's check-then-save window is not atomic: two
  concurrent launches of the same experiment can both be accepted.

All tests are offline and deterministic: no LLM, no network, no simulation.
"""

from __future__ import annotations

import json
import os

from typing import ClassVar

import pytest

from gaworld.apps import mcf_api
from gaworld.behavior.dynamic import (
    _ACTIVE_INTERRUPTS,
    _apply_interrupt_cooldown,
    InterruptCandidate,
)
from gaworld.memory import experience
from gaworld.settings import CONFIG
from gaworld.sim import _action
from gaworld.sim._schedule import ensure_sleep_in_schedule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """The gate must be read from its default state unless a test opts in."""
    monkeypatch.delenv("GAWORLD_MCF_REALISM", raising=False)


# The Phase-1.8 fixture shape (no 00:00 slot at all) - no head slot exists,
# so ensure_sleep_in_schedule derives only the profile tail slot.
DAY_SCHEDULE_NO_HEAD: list[tuple[str, str]] = [
    ("08:00", "吃早饭"),
    ("09:00", "通勤"),
    ("10:00", "上午工作"),
    ("12:00", "午饭"),
    ("14:00", "下午工作"),
    ("18:30", "下班"),
    ("21:00", "个人时间"),
]

# The shape that defeated the Phase-1.8 guard: a NON-sleep 00:00 head slot
# plus unscheduled night slots before the first morning slot.
NIGHT_HEAD_SCHEDULE: list[tuple[str, str]] = [
    ("00:00", "个人时间"),
    ("00:10", "技能提升"),
    ("00:20", "聊邻居停车"),
    ("08:00", "吃早饭"),
    ("09:00", "通勤"),
]


# ---------------------------------------------------------------------------
# H1a - the extra ("00:00", "睡前") head slot must be gated (default run:
# old behaviour appends only the (candidate, "睡前") tail slot)
# ---------------------------------------------------------------------------


def test_h1a_extra_head_slot_gated(monkeypatch):
    # Default run: only the profile-derived tail slot may be appended - no
    # synthetic ("00:00", "睡前") head slot.
    out = ensure_sleep_in_schedule({"job": "职员"}, list(DAY_SCHEDULE_NO_HEAD))
    heads = [(t, a) for t, a in out if t == "00:00"]
    assert heads == [], (
        "default run must not append the extra 00:00 head slot; got %r" % (heads,)
    )
    # The old tail behaviour is preserved: a 睡前 slot still exists.
    assert any(a == "睡前" for _, a in out)
    # mcf_realism ON must still add head coverage.
    monkeypatch.setenv("GAWORLD_MCF_REALISM", "1")
    out_on = ensure_sleep_in_schedule({"job": "职员"}, list(DAY_SCHEDULE_NO_HEAD))
    assert ("00:00", "睡前") in out_on, "mcf_realism ON must still add head coverage"


# ---------------------------------------------------------------------------
# H1b - Phase-1.9 head-slot conversion must be gated (default run: a
# non-sleep 00:00 head slot is NOT rewritten to 睡觉 and pre-07:00 slots
# are NOT dropped)
# ---------------------------------------------------------------------------


def test_h1b_head_slot_conversion_gated(monkeypatch):
    out = ensure_sleep_in_schedule({"job": "职员"}, list(NIGHT_HEAD_SCHEDULE))
    assert ("00:00", "个人时间") in out, (
        "default run must leave a non-sleep 00:00 head slot untouched"
    )
    assert ("00:10", "技能提升") in out and ("00:20", "聊邻居停车") in out, (
        "default run must not drop pre-07:00 slots"
    )
    assert not any(a == "睡觉" for _, a in out), (
        "default run must not rewrite the head slot to 睡觉"
    )
    # mcf_realism ON: the conversion still fires.
    monkeypatch.setenv("GAWORLD_MCF_REALISM", "1")
    out_on = ensure_sleep_in_schedule({"job": "职员"}, list(NIGHT_HEAD_SCHEDULE))
    assert out_on[0] == ("00:00", "睡觉"), "gated conversion must rewrite the head slot"
    assert ("00:10", "技能提升") not in out_on, "gated conversion drops unscheduled night slots"


# ---------------------------------------------------------------------------
# H1c - {"kind":"span"} episode rows must be gated (default run writes none)
# ---------------------------------------------------------------------------


def _span_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ep = json.loads(line)
                if ep.get("kind") == "span":
                    rows.append(ep)
    return rows


def _exercise_spans(tmp_path):
    cfg = {"memory_dir": str(tmp_path / "memory")}
    experience._reset_span_state()
    try:
        experience.record_activity_span("alice", 1, "08:00", "工作", cfg=cfg)
        experience.record_activity_span("alice", 1, "12:00", "午饭", cfg=cfg)
        experience.close_agent_span("alice", 1, "23:00", cfg=cfg)
    finally:
        experience._reset_span_state()
    return os.path.join(str(tmp_path / "memory"), "agent_alice_episodes.jsonl")


def test_h1c_activity_spans_gated(tmp_path, monkeypatch):
    path = _exercise_spans(tmp_path)
    assert _span_rows(path) == [], (
        "default run must not write {'kind':'span'} episode rows"
    )
    # mcf_realism ON: explicit span rows are still written.
    monkeypatch.setenv("GAWORLD_MCF_REALISM", "1")
    path_on = _exercise_spans(tmp_path)
    rows = _span_rows(path_on)
    assert rows, "mcf_realism ON must still write explicit span rows"
    assert rows[0]["activity"] == "工作" and rows[0]["end"] == "12:00"


# ---------------------------------------------------------------------------
# H1d - the actions task must not override the provider budget when
# action_space.max_tokens is at its default (None / omitted)
# ---------------------------------------------------------------------------


def test_h1d_actions_task_default_budget_not_overridden(monkeypatch):
    captured: list[dict] = []

    def fake_call_llm(prompt, task=None, agent_id=None, **kwargs):
        captured.append({"task": task, **kwargs})
        return "{}"

    monkeypatch.setattr(
        _action, "_parse_action_space",
        lambda text, activities: {a: ["动作"] for a in activities},
    )
    monkeypatch.setattr(_action._llm_providers, "call_llm", fake_call_llm)
    monkeypatch.setitem(CONFIG, "action_space", {})

    _action._llm_generate_actions_batch({"id": 1, "name": "张三", "job": "职员"}, ["吃饭"])

    assert captured, "the actions task must still call the LLM"
    for call in captured:
        mt = call.get("max_tokens", "__absent__")
        assert mt in (None, "__absent__"), (
            "at the default action_space.max_tokens the provider budget must "
            "not be overridden (got max_tokens=%r)" % (mt,)
        )
        assert call.get("task") == "actions"

    # An explicit non-default action_space.max_tokens must still be sent.
    captured.clear()
    monkeypatch.setitem(CONFIG, "action_space", {"max_tokens": 999})
    _action._llm_generate_actions_batch({"id": 1, "name": "张三", "job": "职员"}, ["吃饭"])
    assert captured and all(c.get("max_tokens") == 999 for c in captured), (
        "an explicit non-default action_space.max_tokens must still be sent"
    )


# ---------------------------------------------------------------------------
# H5 - interrupt cooldown must key on the pipeline agent shape ({"id": ...})
# ---------------------------------------------------------------------------


def _candidate() -> InterruptCandidate:
    return InterruptCandidate(
        source="environment",
        kind="weather_change",
        activity="躲雨",
        reason="暴雨",
        priority=0.8,
        duration_minutes=30,
    )


def test_h5_cooldown_suppresses_for_pipeline_shaped_agent():
    _ACTIVE_INTERRUPTS.clear()
    try:
        # A recorded interrupt for this agent - registered under the agent's
        # id, as a fixed pipeline path would produce.
        _ACTIVE_INTERRUPTS["alice"] = {
            "environment:weather_change": {
                "source": "environment",
                "kind": "weather_change",
                "priority": 0.8,
                "duration": 30,
                "until": 9 * 60 + 30,
            }
        }
        agent = {"id": "alice", "name": "张三"}  # pipeline shape, no "agent_id"
        kept = _apply_interrupt_cooldown(agent, "09:00", [_candidate()])
        assert kept == [], (
            "the cooldown must activate for an agent dict shaped like the "
            "pipeline agents ({'id': ...}) - got the candidate back unsuppressed"
        )
        # An expired window must not suppress.
        _ACTIVE_INTERRUPTS["alice"] = {
            "environment:weather_change": {
                "source": "environment",
                "kind": "weather_change",
                "priority": 0.8,
                "duration": 30,
                "until": 8 * 60 + 30,  # expired by 09:00
            }
        }
        kept = _apply_interrupt_cooldown(agent, "09:00", [_candidate()])
        assert len(kept) == 1, "an expired cooldown window must not suppress"
    finally:
        _ACTIVE_INTERRUPTS.clear()


# ---------------------------------------------------------------------------
# M1 - day-aware within-day window classification in compare_arms
# ---------------------------------------------------------------------------


def _write_run(root: str, records: list[dict]) -> str:
    run_dir = os.path.join(root, "run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "agent_1.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return run_dir


def _rec(day: int, time: str, activity: str) -> dict:
    return {
        "agent_id": 1,
        "day": day,
        "time": time,
        "final_activity": activity,
        "scheduled_activity": activity,
    }


def test_m1_pre_trigger_day_divergence_after_trigger_clock_time_is_pre(tmp_path):
    from benchmark.mcf.compare import compare_arms

    # Arms diverge ONLY on day 1 (two days BEFORE intervention_day=3), at
    # 23:00 - a clock time AFTER the trigger time 08:00. The docstring
    # contract: a record is post only when its day is after intervention_day
    # or it is on the trigger day at/after the trigger time. Day 1 at 23:00
    # is therefore PRE and must not leak into the post window.
    baseline = [
        _rec(1, "23:00", "工作"),
        _rec(3, "07:50", "吃早饭"),
        _rec(3, "08:00", "通勤"),
        _rec(5, "23:00", "工作"),
    ]
    intervention = [
        _rec(1, "23:00", "睡觉"),
        _rec(3, "07:50", "吃早饭"),
        _rec(3, "08:00", "通勤"),
        _rec(5, "23:00", "工作"),
    ]
    b_dir = _write_run(str(tmp_path / "b"), baseline)
    i_dir = _write_run(str(tmp_path / "i"), intervention)

    report = compare_arms(b_dir, i_dir, intervention_day=3, intervention_time="08:00")
    windows = report["windows"]
    assert windows["pre"]["max_abs_diff"] > 0, (
        "divergence on a pre-trigger day must land in the pre window "
        "(got pre=%r, post=%r)" % (windows["pre"], windows["post"])
    )
    assert windows["post"]["max_abs_diff"] == 0, (
        "post window must stay clean; got %r" % (windows["post"],)
    )

    # Regression guard: the trigger tick itself (intervention day, at/after
    # the trigger time) belongs to the post window.
    baseline = [_rec(3, "07:50", "吃早饭"), _rec(3, "08:00", "通勤")]
    intervention = [_rec(3, "07:50", "吃早饭"), _rec(3, "08:00", "躲雨")]
    b_dir = _write_run(str(tmp_path / "b2"), baseline)
    i_dir = _write_run(str(tmp_path / "i2"), intervention)
    report = compare_arms(b_dir, i_dir, intervention_day=3, intervention_time="08:00")
    assert report["windows"]["post"]["max_abs_diff"] > 0, (
        "the trigger tick itself belongs to the post window"
    )


def test_m1_post_trigger_day_divergence_before_trigger_clock_time_is_post(tmp_path):
    from benchmark.mcf.compare import compare_arms

    # Arms diverge ONLY on day 5 (AFTER the trigger day), at 07:00 - a clock
    # time BEFORE the trigger time 08:00. By day-awareness this is POST;
    # classifying by clock time alone would (wrongly) call it pre.
    baseline = [
        _rec(3, "07:00", "工作"),
        _rec(3, "08:00", "通勤"),
        _rec(5, "07:00", "睡觉"),
    ]
    intervention = [
        _rec(3, "07:00", "工作"),
        _rec(3, "08:00", "通勤"),
        _rec(5, "07:00", "个人时间"),
    ]
    b_dir = _write_run(str(tmp_path / "b"), baseline)
    i_dir = _write_run(str(tmp_path / "i"), intervention)

    report = compare_arms(b_dir, i_dir, intervention_day=3, intervention_time="08:00")
    windows = report["windows"]
    assert windows["post"]["max_abs_diff"] > 0, (
        "divergence on a day after the trigger day is post regardless of clock time"
    )
    assert windows["pre"]["max_abs_diff"] == 0, "the pre window must stay clean"


# ---------------------------------------------------------------------------
# M2 - an UNRELIABLE arm must never yield no_effect
# ---------------------------------------------------------------------------


def test_m2_unreliable_arm_never_maps_to_no_effect():
    from benchmark.mcf.compare import _effect_verdict

    windows = {
        "pre": {"max_abs_diff": 0.0, "episode_disagreement_rate": 0.0},
        "post": {"max_abs_diff": 0.0, "episode_disagreement_rate": 0.0},
    }
    verdict = _effect_verdict({"baseline": "ok", "intervention": "UNRELIABLE"}, windows)
    assert verdict["verdict"] != "no_effect", (
        "an UNRELIABLE arm (below MIN_COMPARISON_DAYS/POPULATION) is insufficient "
        "data, not evidence of no effect"
    )
    assert verdict == {"verdict": "insufficient_data", "beneficial": False}
    # Regression guard: MISSING still maps to insufficient_data.
    assert _effect_verdict(
        {"baseline": "MISSING", "intervention": "ok"}, windows
    )["verdict"] == "insufficient_data"


# ---------------------------------------------------------------------------
# H4 - exp_id traversal must be rejected in mcf_api
# ---------------------------------------------------------------------------


@pytest.fixture
def mcf_env(tmp_path, monkeypatch):
    monkeypatch.setattr(mcf_api, "_mcf_root", lambda: str(tmp_path / "mcf"))
    monkeypatch.setattr(mcf_api, "_start_job", lambda exp_id: None)
    os.makedirs(str(tmp_path / "mcf"), exist_ok=True)
    return tmp_path


def _plant_exp(tmp_path, rel_target: str, exp_id: str) -> str:
    """Plant an experiment.json OUTSIDE the mcf root, reachable via exp_id."""
    target_dir = os.path.normpath(os.path.join(str(tmp_path / "mcf"), rel_target))
    os.makedirs(target_dir, exist_ok=True)
    exp = {
        "id": exp_id,
        "status": "draft",
        "estimated_calls": 10,
        "progress": {"legs_done": 0, "legs_total": 4, "failures": []},
        "architectures": [{"id": "m0", "label": "M0", "patch": {}}],
        "n_configs": 2,
        "n_seeds": 2,
    }
    path = os.path.join(target_dir, "experiment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exp, f)
    return target_dir


@pytest.mark.parametrize("exp_id", ["..", "../../evil", "a/b"])
def test_h4_load_exp_rejects_traversal_ids(mcf_env, tmp_path, exp_id):
    _plant_exp(tmp_path, exp_id, exp_id)
    assert mcf_api._load_exp(exp_id) is None, (
        "exp_id %r must never resolve to a file outside output/mcf" % (exp_id,)
    )


def test_h4_get_rejects_traversal_id(mcf_env, tmp_path):
    _plant_exp(tmp_path, "..", "..")
    body, status = mcf_api.handle_get("/api/mcf/experiment/..", {})
    assert 400 <= status < 500, (
        "a traversal exp_id must be rejected with a 4xx; got %s: %r" % (status, body)
    )
    assert body.get("id") != "..", (
        "a traversal id must never resolve to a file outside output/mcf"
    )


def test_h4_launch_rejects_traversal_id(mcf_env, tmp_path):
    _plant_exp(tmp_path, "../../evil", "../../evil")
    body, status = mcf_api.launch_experiment("../../evil", {"confirm_cost": True})
    assert 400 <= status < 500, (
        "a traversal exp_id must never launch (would write outside output/mcf); "
        "got %s: %r" % (status, body)
    )


def test_h4_nested_id_rejected(mcf_env):
    body, status = mcf_api.handle_get("/api/mcf/experiment/a/b", {})
    assert status == 400, (
        "an exp_id containing '/' must be rejected; got %s: %r" % (status, body)
    )


# ---------------------------------------------------------------------------
# M5 - atomic duplicate-launch prevention
# ---------------------------------------------------------------------------


def _create_exp(**extra) -> str:
    spec = {
        "name": "smoke",
        "architectures": [
            {"id": "m0", "label": "M0 baseline", "patch": {}},
            {"id": "m1", "label": "M1 realism", "patch": {"personality.channels.rules": True}},
        ],
        "parameters": ["style_fit_amplitude"],
        "n_configs": 2,
        "sampler": "endpoints",
        "n_seeds": 2,
        "seeds": [7, 11],
        "budget": {"max_calls": 100, "max_tokens": 1000, "max_elapsed_seconds": 60},
    }
    body, status = mcf_api.handle_post("/api/mcf/experiment", {**spec, **extra})
    assert status == 201, body
    return body["id"]


def test_m5_double_launch_is_atomic(mcf_env, monkeypatch):
    exp_id = _create_exp()

    # Deterministic interleave in the check-then-save window of launch #1:
    # while it is inside _save_exp, a second launch of the same experiment
    # runs to completion against the still-on-disk draft.
    real_save = mcf_api._save_exp
    second: dict = {}
    entered = {"n": 0}

    def interleaved_save(exp):
        if entered["n"] == 0 and exp.get("id") == exp_id:
            entered["n"] += 1
            second["result"] = mcf_api.launch_experiment(exp_id, {"confirm_cost": True})
        real_save(exp)

    monkeypatch.setattr(mcf_api, "_save_exp", interleaved_save)

    first_body, first_status = mcf_api.handle_post(
        f"/api/mcf/experiment/{exp_id}/launch", {"confirm_cost": True}
    )
    second_body, second_status = second.get("result", ({}, None))

    statuses = sorted([first_status, second_status])
    assert statuses == [202, 409], (
        "two concurrent launches of the same experiment must yield exactly one "
        "202 and one 409; got %s and %s (%r)" % (first_status, second_status, second_body)
    )


# ---------------------------------------------------------------------------
# M4 - guarded int parsing in _validate_spec (bad JSON types yield 400, not 500)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_configs", "many"),
        ("n_seeds", "lots"),
        ("seeds", [7, "eleven"]),
        ("budget", {"max_calls": "heaps"}),
    ],
)
def test_m4_bad_int_types_yield_400(mcf_env, field, value):
    spec = {
        "name": "smoke",
        "architectures": [
            {"id": "m0", "label": "M0 baseline", "patch": {}},
            {"id": "m1", "label": "M1 realism", "patch": {}},
        ],
        "parameters": [],
        field: value,
    }
    body, status = mcf_api.handle_post("/api/mcf/experiment", spec)
    assert status == 400, (
        f"a non-integer {field} must be rejected with 400, not blow up with 500; "
        f"got {status}: {body!r}"
    )
    assert field.replace("budget.", "") in str(body.get("error", "")) or "integer" in str(
        body.get("error", "")
    )


# ---------------------------------------------------------------------------
# M6 - policy_events fail-fast at start + type-safe day matching
# ---------------------------------------------------------------------------


def test_m6_validate_rejects_unsupported_operation_before_the_run():
    from gaworld.kernel.policy_events import validate_policy_events

    with pytest.raises(ValueError) as excinfo:
        validate_policy_events([{"day": 3, "time": "08:00", "operation": "detonate"}])
    assert "detonate" in str(excinfo.value)
    # A valid list passes; description-only legacy entries are fine too.
    validate_policy_events([{"day": 3, "time": "08:00"}])
    validate_policy_events([{"day": 3, "time": "08:00", "operation": "set_agent_state"}])


def test_m6_day_matching_is_type_coerced():
    from gaworld.kernel.policy_events import execute_policy_events

    fired_calls: list[str] = []

    class _Controller:
        def intervene(self, operation, ctx, **kw):
            fired_calls.append(kw.get("agent_id"))
            return "ok"

    class _Ctx:
        agents_by_id: ClassVar[dict] = {7: {"id": 7, "state": {}}}
        controller = _Controller()
        recorder = None

    event = {"name": "e", "day": "3", "time": "08:00", "operation": "set_agent_state",
             "targets": [7], "kwargs": {"mood": "calm"}}

    # A string "3" in the spec and int 3 from the tick loop are the same day.
    assert execute_policy_events(_Ctx(), [event], 3, "08:00") == 1
    assert fired_calls == [7]

    # Non-matching days / times still do not fire (regression guard).
    assert execute_policy_events(_Ctx(), [event], 9, "08:00") == 0
    assert execute_policy_events(_Ctx(), [event], 3, "09:00") == 0
    assert fired_calls == [7]
