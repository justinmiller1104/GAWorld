"""Declarative policy_events executor (Phase 2.3, §8.1 of docs/MCF_GAWORLD.md).

``policy_events`` entries carrying an ``operation`` translate into an actual
supported simulation operation at the matched tick, via the existing audited
kernel path (``Controller.intervene``). Entries WITHOUT an ``operation`` keep
the legacy text-feed behaviour (``gaworld/policy/plugin.py`` untouched).
"""

from __future__ import annotations

_SUPPORTED = {"set_agent_state"}


def execute_policy_events(ctx, events, day, time_str) -> int:
    """Fire the entries whose day/time match this tick; return how many fired.

    Only the matched entries are executed; anything else is left untouched.
    """
    fired = 0
    for event in events or []:
        if event.get("day") != day or str(event.get("time", "")) != str(time_str):
            continue
        operation = event.get("operation")
        if not operation:
            # Legacy description-only entry: text feed only, no state change.
            continue
        if operation not in _SUPPORTED:
            raise ValueError(
                f"policy_events operation {operation!r} is not supported; "
                f"supported: {sorted(_SUPPORTED)}"
            )
        for agent_id in event.get("targets", []):
            for field, value in (event.get("kwargs") or {}).items():
                agent = ctx.agents_by_id.get(agent_id)
                state = agent.get("state", {}) if agent else {}
                before = state.get(field)
                result = ctx.controller.intervene(
                    operation, ctx, agent_id=agent_id, key=field, value=value
                )
                after = ctx.agents_by_id.get(agent_id, {}).get("state", {}).get(field)
                if ctx.recorder is not None:
                    ctx.recorder.record(
                        "policy_events.execution",
                        {
                            "name": event.get("name", ""),
                            "operation": operation,
                            "agent_id": agent_id,
                            "field": field,
                            "before": before,
                            "after": after,
                            "result": result,
                        },
                    )
        fired += 1
    return fired
