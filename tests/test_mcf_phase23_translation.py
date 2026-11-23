"""Phase 2.3 RED test — runner translation must carry operation/targets.

RESULT-3561 bug: ``benchmark/mcf/experiment.py::_translated_config`` delivers a
``set_agent_state`` spec as a ``policy_events`` entry containing only
``{day, time, name, kwargs}`` — it omits ``operation`` and ``targets``. The
executor (``gaworld/kernel/policy_events.py::execute_policy_events``) sees no
``operation`` and silently treats the entry as legacy text-only: nothing fires.

This test pins the contract: the translated intervention-arm config's
``policy_events`` entry for a ``set_agent_state`` spec must carry
``operation``, ``targets`` and ``kwargs`` so the kernel executor fires
without external fixups. Deterministic, zero-inference, network-free.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from benchmark.mcf.experiment import run_experiment

# A minimal, valid set_agent_state intervention spec (same schema as the
# Phase 2.3 experiment tests).
SPEC = {
    "name": "mood_boost",
    "trigger": {"day": 2, "time": "10:00"},
    "operation": "set_agent_state",
    "kwargs": {"metric": "emotion", "value": 0.95, "agent_id": 7},
    "targets": [7],
}

BASE_CONFIG = {"fos_fast_mode": True, "sim_days": 3}


class DryHarness:
    """A deterministic zero-inference harness that just records the config
    it was handed and writes the minimal run shapes."""

    def __call__(self, config: dict, run_dir) -> dict:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(json.dumps({"outcome": "completed"}), encoding="utf-8")
        return {"ok": True, "run_dir": str(run_dir), "summary": {"llm_calls": 0}}


def test_set_agent_state_policy_event_entry_carries_operation_and_targets(tmp_path):
    result = run_experiment(copy.deepcopy(BASE_CONFIG), copy.deepcopy(SPEC), DryHarness(), run_root=tmp_path)
    cfg = json.loads((Path(result["intervention"]["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    events = [e for e in cfg.get("policy_events", []) if e.get("name") == SPEC["name"]]
    assert events, "set_agent_state must translate into a policy_events entry the sim matches per tick"
    event = events[0]
    # the fields the kernel executor needs to actually fire
    assert event.get("operation") == "set_agent_state", (
        "policy_events entry must carry operation — without it the executor "
        "silently treats it as legacy text-only and nothing fires"
    )
    assert event.get("targets") == [7], (
        "policy_events entry must carry targets — without it the executor "
        "has no agent to apply the state change to"
    )
    assert event.get("kwargs") == {"metric": "emotion", "value": 0.95, "agent_id": 7}
    assert (event["day"], event["time"]) == (2, "10:00")
