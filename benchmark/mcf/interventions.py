"""Declarative intervention-spec schema for scenario arms (Phase 2.2).

A scenario arm is described by one JSON-able spec dict:

    {
        "name": "rainy_afternoon",
        "trigger": {"day": 4, "time": "14:00"},     # policy_events matcher convention
        "operation": "update_config",               # or set_agent_state / policy_event
        "kwargs": {"key": "weather.rain_probability", "value": 0.9},
        "targets": ["env"],                         # agent ids, or an env component
    }

The operation vocabulary is exactly the kernel's standard interventions
(``gaworld/kernel/interventions.py``) plus the declarative ``policy_event``
entry matched per tick in the main loop — no kernel changes.

``apply_intervention`` builds pure config/policy_events deltas: it never
mutates the base config, so scenario arms differ from the baseline ONLY by
their delta (fresh-start equivalence: same seed, same roster, config-delta-only).
"""

import copy
import re

__all__ = ["InterventionSpecError", "apply_intervention", "validate_intervention_spec"]

# Kernel standard interventions (set_agent_state / update_config) plus the
# declarative policy_events entry; add_agent is NOT here — it is unimplemented
# in the kernel and needs a profile/social/memory bootstrap path.
OPERATIONS = ("set_agent_state", "update_config", "policy_event")

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class InterventionSpecError(ValueError):
    """A malformed intervention spec (validation error with a clear message)."""


def validate_intervention_spec(spec):
    """Validate one intervention spec and return a normalized deep copy.

    Raises :class:`InterventionSpecError` with a message naming the offending
    field for: missing name, missing/partial trigger, non-HH:MM trigger time,
    unknown operation, bad targets, non-dict kwargs, update_config without a
    ``key``, or a non-positive duration.
    """
    if not isinstance(spec, dict):
        raise InterventionSpecError("spec must be a dict")
    normalized = copy.deepcopy(spec)

    name = normalized.get("name")
    if not isinstance(name, str) or not name.strip():
        raise InterventionSpecError("spec is missing a non-empty 'name'")

    trigger = normalized.get("trigger")
    if not isinstance(trigger, dict):
        raise InterventionSpecError("spec is missing a 'trigger' dict with {'day': int, 'time': 'HH:MM'}")
    day = trigger.get("day")
    if not isinstance(day, int) or isinstance(day, bool) or day < 1:
        raise InterventionSpecError(f"trigger.day must be a positive int, got {day!r}")
    time_str = trigger.get("time")
    if not isinstance(time_str, str) or not _TIME_RE.match(time_str):
        raise InterventionSpecError(
            f"trigger.time must match the policy_events HH:MM matcher, got {time_str!r}"
        )

    operation = normalized.get("operation")
    if operation not in OPERATIONS:
        raise InterventionSpecError(
            f"unknown operation {operation!r}; supported: {', '.join(OPERATIONS)} "
            "(matches the kernel's registered standard interventions)"
        )

    targets = normalized.get("targets")
    if not isinstance(targets, list) or not targets:
        raise InterventionSpecError(
            "targets must be a non-empty list of agent ids (ints) or, for "
            "update_config, an env component name"
        )
    for target in targets:
        if isinstance(target, bool):
            raise InterventionSpecError(f"bad target {target!r}: expected an agent id (int)")
        if isinstance(target, int):
            continue
        if isinstance(target, str) and operation == "update_config":
            continue  # env component target, e.g. "env"
        raise InterventionSpecError(
            f"bad target {target!r}: agent ids must be ints"
            + ("; string targets are only valid for update_config" if isinstance(target, str) else "")
        )

    kwargs = normalized.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise InterventionSpecError(f"kwargs must be a dict, got {type(kwargs).__name__}")
    if operation == "update_config" and "key" not in kwargs:
        raise InterventionSpecError("update_config kwargs must contain a 'key' (dotted config path)")

    duration = normalized.get("duration")
    if duration is not None and (not isinstance(duration, int) or isinstance(duration, bool) or duration < 1):
        raise InterventionSpecError(f"duration must be a positive int (days), got {duration!r}")

    return normalized


def apply_intervention(spec, config):
    """Return a NEW config with the intervention's delta applied.

    Pure delta builder: the input ``config`` is never mutated. ``update_config``
    sets one dotted-path value (creating intermediate dicts); ``policy_event``
    appends exactly one event to ``config["policy_events"]``; ``set_agent_state``
    is applied by the kernel at run time against target agents, so it yields an
    unchanged config copy here (the delta lives in the spec, not the config).
    """
    new_config = copy.deepcopy(config)
    operation = spec.get("operation")
    trigger = spec.get("trigger", {})
    kwargs = spec.get("kwargs", {})
    if operation == "update_config":
        _set_dotted(new_config, kwargs["key"], kwargs.get("value"))
    elif operation == "policy_event":
        event = {"day": trigger["day"], "time": trigger["time"]}
        event.update(kwargs)
        events = new_config.setdefault("policy_events", [])
        if not isinstance(events, list):
            raise InterventionSpecError("config['policy_events'] must be a list to append to")
        events.append(event)
    return new_config


def _set_dotted(config, dotted_key, value):
    """Set ``dotted_key`` (e.g. ``weather.rain_probability``) into nested dicts."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value
