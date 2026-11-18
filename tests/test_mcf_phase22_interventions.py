"""Phase 2.2 RED tests — intervention spec schema (TASK-3542, contracts 2 & 9).

The intervention-testing framework does not exist yet: these tests pin the
declarative intervention-spec contract that ``benchmark/mcf/interventions.py``
must satisfy BEFORE it is written. Deterministic pure-Python tests — no LLM,
no network, no simulation.

Contracts pinned here:

  2. Delta-only change: applying a validated intervention to a base config
     produces a new config that differs from the base in EXACTLY the
     intervention's intended keys — nothing else.
  9. Spec validation: malformed intervention specs are rejected with clear
     errors — unknown operation, missing trigger, bad targets — while every
     valid shape in the operation vocabulary (set_agent_state / update_config /
     policy_event, matching the kernel interventions and the policy_events
     matcher) validates cleanly.

Natural home of the implementation: ``benchmark/mcf/interventions.py``
(kernel untouched — RESULT-3541 §Recommended placement).
"""

import copy

import pytest

from benchmark.mcf.interventions import (
    InterventionSpecError,
    apply_intervention,
    validate_intervention_spec,
)

# ---------------------------------------------------------------------------
# Valid specs, one per kernel operation (gaworld/kernel/interventions.py:62-65)
# ---------------------------------------------------------------------------

UPDATE_CONFIG_SPEC = {
    "name": "rainy_afternoon",
    "trigger": {"day": 4, "time": "14:00"},
    "operation": "update_config",
    "kwargs": {"key": "weather.rain_probability", "value": 0.9},
    "targets": ["env"],
}

POLICY_EVENT_SPEC = {
    "name": "community_notice",
    "trigger": {"day": 5, "time": "09:00"},
    "operation": "policy_event",
    "kwargs": {"description": "社区发布节能倡议"},
    "targets": [1, 2, 3],
}

SET_AGENT_STATE_SPEC = {
    "name": "mood_boost",
    "trigger": {"day": 3, "time": "08:00"},
    "operation": "set_agent_state",
    "kwargs": {"metric": "emotion", "value": 0.95},
    "targets": [7],
}

ALL_VALID_SPECS = [UPDATE_CONFIG_SPEC, POLICY_EVENT_SPEC, SET_AGENT_STATE_SPEC]

BASE_CONFIG = {
    "random_seed": 42,
    "weather": {"rain_probability": 0.1, "season": "梅雨"},
    "policy_events": [],
    "fos_fast_mode": True,
}


# ---------------------------------------------------------------------------
# Contract 9 — spec validation: valid shapes pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_VALID_SPECS, ids=lambda s: s["name"])
def test_valid_specs_validate_cleanly(spec):
    normalized = validate_intervention_spec(copy.deepcopy(spec))
    assert normalized["name"] == spec["name"]
    assert normalized["operation"] == spec["operation"]
    assert normalized["trigger"]["day"] == spec["trigger"]["day"]
    assert normalized["trigger"]["time"] == spec["trigger"]["time"]


def test_unknown_operation_rejected():
    spec = copy.deepcopy(UPDATE_CONFIG_SPEC)
    spec["operation"] = "add_agent"  # explicitly unimplemented in the kernel
    with pytest.raises(InterventionSpecError, match="operation"):
        validate_intervention_spec(spec)


def test_missing_trigger_rejected():
    spec = copy.deepcopy(UPDATE_CONFIG_SPEC)
    del spec["trigger"]
    with pytest.raises(InterventionSpecError, match="trigger"):
        validate_intervention_spec(spec)


@pytest.mark.parametrize("trigger", [{}, {"day": 4}, {"time": "14:00"}])
def test_partial_trigger_rejected(trigger):
    spec = copy.deepcopy(UPDATE_CONFIG_SPEC)
    spec["trigger"] = trigger
    with pytest.raises(InterventionSpecError):
        validate_intervention_spec(spec)


def test_bad_time_format_rejected():
    spec = copy.deepcopy(UPDATE_CONFIG_SPEC)
    spec["trigger"]["time"] = "2:30pm"  # must match the HH:MM policy matcher
    with pytest.raises(InterventionSpecError, match="time"):
        validate_intervention_spec(spec)


@pytest.mark.parametrize(
    "targets",
    [[], "agent_7", [1, "two"], None],
    ids=["empty", "non_int_string", "mixed", "none"],
)
def test_bad_targets_rejected(targets):
    spec = copy.deepcopy(POLICY_EVENT_SPEC)
    spec["targets"] = targets
    with pytest.raises(InterventionSpecError, match="target"):
        validate_intervention_spec(spec)


def test_missing_name_rejected():
    spec = copy.deepcopy(UPDATE_CONFIG_SPEC)
    del spec["name"]
    with pytest.raises(InterventionSpecError, match="name"):
        validate_intervention_spec(spec)


def test_update_config_without_key_rejected():
    spec = copy.deepcopy(UPDATE_CONFIG_SPEC)
    del spec["kwargs"]["key"]
    with pytest.raises(InterventionSpecError):
        validate_intervention_spec(spec)


# ---------------------------------------------------------------------------
# Contract 2 — delta-only change
# ---------------------------------------------------------------------------


def _deep_diff(old, new, path=""):
    """All leaf paths whose values differ between two nested dicts."""
    diffs = set()
    keys = set(old) | set(new)
    for key in keys:
        here = f"{path}.{key}" if path else str(key)
        if key not in old:
            diffs.add(here)
        elif key not in new:
            diffs.add(here + " (removed)")
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            diffs |= _deep_diff(old[key], new[key], here)
        elif old[key] != new[key]:
            diffs.add(here)
    return diffs


def test_update_config_changes_exactly_the_named_path():
    base = copy.deepcopy(BASE_CONFIG)
    applied = apply_intervention(UPDATE_CONFIG_SPEC, base)
    assert _deep_diff(base, applied) == {"weather.rain_probability"}
    assert applied["weather"]["rain_probability"] == 0.9


def test_policy_event_appends_exactly_one_event():
    base = copy.deepcopy(BASE_CONFIG)
    applied = apply_intervention(POLICY_EVENT_SPEC, base)
    diffs = _deep_diff(base, applied)
    assert diffs == {"policy_events"} or diffs == set()  # list replaced or extended
    events = applied["policy_events"]
    assert len(events) == 1
    assert events[0]["day"] == 5 and events[0]["time"] == "09:00"
    assert events[0]["description"] == "社区发布节能倡议"


def test_apply_does_not_mutate_the_base_config():
    base = copy.deepcopy(BASE_CONFIG)
    snapshot = copy.deepcopy(base)
    apply_intervention(UPDATE_CONFIG_SPEC, base)
    assert base == snapshot


def test_non_intervention_fields_untouched():
    base = copy.deepcopy(BASE_CONFIG)
    applied = apply_intervention(POLICY_EVENT_SPEC, base)
    assert applied["random_seed"] == base["random_seed"]
    assert applied["fos_fast_mode"] == base["fos_fast_mode"]
    assert applied["weather"] == base["weather"]
