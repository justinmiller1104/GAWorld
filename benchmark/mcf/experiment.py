"""Experiment runner: two actual arms (baseline + intervention) executed by an
injected harness (Phase 2.3).

The harness callable is the execution vehicle — in production the Phase 2.1
canary ``dry_run=True`` pattern (full sim pipeline, ``call_llm`` stubbed,
zero provider calls); in tests a deterministic zero-inference stub writing
the real run-directory file shapes. The runner itself:

- validates the intervention spec BEFORE any run directory exists;
- translates the spec into operations the sim actually supports
  (``policy_events`` entries and/or config deltas via
  :mod:`benchmark.mcf.interventions` delta builders — no kernel changes);
- runs the baseline arm (pure config, no delta) and the intervention arm
  (delta applied) in independent run dirs with manifests via
  :func:`benchmark.mcf.launch.create_run`;
- audits what REALLY happened per arm: intended vs actual execution time,
  targets, requested change, applied yes/no, before/after state — read back
  from the run's own records (``intervention_executions.jsonl`` +
  ``agent_state/``), never from the spec. A declared intervention whose
  trigger never fires is ``NOT_APPLIED``; a failed operation is ``FAILED``
  with its reason; never silently "success";
- marks incomplete arms (harness reports ``ok=False``, manifest stays
  non-completed) and passes their comparison outcomes through as ``MISSING``
  rather than fabricating numbers;
- compares the arms with :func:`benchmark.mcf.compare.compare_arms` over the
  arms' actual episode files.
- (Phase 2.5) ``fork_arms_from_checkpoint`` launches both arms from ONE
  day-boundary checkpoint dir — the shared pre-intervention history: each
  arm restores the same checkpoint into its own run dir (create_run
  isolation), the checkpoint's news cache file is copied into both run dirs
  (wall-clock headlines frozen), non-baseline arms run under the LLM replay
  wrapper (:mod:`benchmark.mcf.replay`) until the trigger, and both arms'
  manifests carry fork provenance (checkpoint dir + replay window).

No LLM calls, no network, no kernel/sim changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from benchmark.mcf.compare import NEUTRAL_VERDICTS, compare_arms
from benchmark.mcf.interventions import InterventionSpecError, validate_intervention_spec
from benchmark.mcf.launch import create_run, finalize_run

try:  # pragma: no cover - trivial import guard
    from benchmark.mcf.replay import install_llm_replay, install_sequence_replay, sim_time_stop_predicate
except ImportError:  # replay module always present in-tree; keep launcher importable
    install_llm_replay = None
    install_sequence_replay = None
    sim_time_stop_predicate = None

__all__ = ["run_experiment", "run_leg", "fork_arms_from_checkpoint"]

# real per-leg runner (task 3753): lives in leg_runner, re-exported so the
# dashboard's ``_build_harness`` finds ``experiment.run_leg``
from benchmark.mcf.leg_runner import run_leg

Harness = Callable[[dict[str, Any], "str | Path"], dict[str, Any]]

# status values of a per-intervention audit record
_APPLIED = "applied"
_NOT_APPLIED = "NOT_APPLIED"
_FAILED = "FAILED"


def run_experiment(
    base_config: dict[str, Any],
    intervention_spec: dict[str, Any],
    harness: Harness,
    run_root: "str | Path",
) -> dict[str, Any]:
    """Run one baseline arm and one intervention arm and compare them.

    ``harness(config, run_dir)`` executes one actual arm (the real vehicle
    being the zero-inference dry-run pattern; tests inject a deterministic
    stub) and returns a summary dict; ``ok: False`` in it (or a raised
    exception) marks the arm incomplete.

    The spec is validated first — an unsupported/malformed operation raises
    :class:`InterventionSpecError` before any run directory is created.
    """
    spec = validate_intervention_spec(intervention_spec)  # rejects unknown ops up front
    root = Path(run_root)

    # pure delta builders: baseline config stays untouched, the intervention
    # arm differs ONLY by the translated delta
    baseline_config = copy.deepcopy(base_config)
    intervention_config = _translated_config(spec, baseline_config)

    baseline = _execute_arm("baseline", baseline_config, harness, root / "baseline")
    intervention = _execute_arm("intervention", intervention_config, harness, root / "intervention")

    baseline["audit"] = []
    intervention["audit"] = [
        _audit_intervention(spec, Path(baseline["run_dir"]), Path(intervention["run_dir"]))
    ]

    report = compare_arms(
        baseline["run_dir"], intervention["run_dir"], intervention_day=spec["trigger"]["day"]
    )
    for arm_name, arm in (("baseline", baseline), ("intervention", intervention)):
        if not arm["complete"] and report["data_status"].get(arm_name) != "MISSING":
            # insufficient data is never evidence of no effect
            report["data_status"][arm_name] = "MISSING"
    if "MISSING" in report["data_status"].values():
        report["effect"] = {"verdict": "insufficient_data", "beneficial": False}
    elif (
        report["effect"]["verdict"] not in NEUTRAL_VERDICTS
        and report["windows"]["post"]["max_abs_diff"] == 0.0
    ):
        # complete arms (however small) with an exactly zero post-window diff:
        # a genuine no-effect, not missing data
        report["effect"] = {"verdict": "no_effect", "beneficial": False}
    return {"baseline": baseline, "intervention": intervention, "comparison": report}


# ---------------------------------------------------------------------------
# Phase 2.5 — checkpoint fork launcher (shared pre-intervention history)
# ---------------------------------------------------------------------------


def fork_arms_from_checkpoint(
    checkpoint_dir: "str | Path",
    arms: dict[str, dict[str, Any]],
    run_root: "str | Path",
) -> dict[str, dict[str, Any]]:
    """Fork every arm in ``arms`` from the SAME checkpoint dir.

    Per the Phase 2.5 investigation (Part 1): a day-boundary checkpoint
    carries the full restorable state (agents+memories, plugin/extension
    state, clock, env runtime, global RNG); the only fork-killers it does
    NOT capture are LLM stochasticity and the wall-clock news cache. This
    launcher handles both:

    - each arm restores the same checkpoint (``resume_from_checkpoint`` in
      its harness-visible config) into its OWN run dir (create_run isolation);
    - the checkpoint's ``news_cache.json`` (if present) is copied into each
      run dir — background headlines are frozen, arms cannot drift on
      wall-clock-fetched news;
    - all-arms replay (Phase 2.6): every arm carrying recorded responses —
      baseline and neutral_control included — runs under the LLM replay
      wrapper until its trigger (``replay.stop``); an arm without
      ``recorded_responses`` (e.g. the live-sampling baseline) samples live.
      Pre-stop misses raise :class:`ReplayCacheMiss` loudly;
    - both arms' ``run_manifest.json`` records fork provenance: the
      checkpoint dir and the replay window
      (``{"start": "checkpoint", "stop": trigger}``).
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"fork source checkpoint dir not found: {ckpt}")
    root = Path(run_root)
    news_cache = ckpt / "news_cache.json"
    results: dict[str, dict[str, Any]] = {}
    for arm_name, arm in arms.items():
        trigger = dict(arm.get("trigger", {}))
        config = dict(arm.get("config", {}))
        config["resume_from_checkpoint"] = str(ckpt)
        # all-arms replay protocol (Phase 2.6/2.7): EVERY arm carrying recorded
        # responses — baseline and neutral_control included — replays the same
        # recorded sequence until its trigger, so all arms share the identical
        # pre-trigger history. An arm without recordings samples live.
        # Phase 2.7: arms carrying a recorded EVENT SEQUENCE replay strictly
        # (install_sequence_replay — any unexpected pre-trigger call raises);
        # the boundary is the sim-time stop factory built from the trigger
        # (day/time) unless the arm carries its own stop predicate. Arms with
        # only a dict of recorded responses keep the Phase 2.5 memoizer.
        run_replay = (
            arm_name != "baseline" or bool(arm.get("recorded_responses")) or bool(arm.get("recorded_events"))
        )
        restore_replay = None
        if run_replay and install_sequence_replay is not None:
            trigger_time = trigger.get("time")
            stop = arm.get("stop_predicate")
            if stop is None and sim_time_stop_predicate is not None and trigger.get("day") and trigger_time:
                stop = sim_time_stop_predicate(day=int(trigger["day"]), time=str(trigger_time))
            if arm.get("recorded_events"):
                _, restore_replay = install_sequence_replay(list(arm["recorded_events"]), stop_predicate=stop)
                replay_kind = "sequence"
            elif install_llm_replay is not None:
                _, restore_replay = install_llm_replay(
                    dict(arm.get("recorded_responses", {})), stop_predicate=stop
                )
                replay_kind = "dict"
            else:
                replay_kind = None
        try:
            result = _execute_arm(arm_name, config, arm["harness"], root / arm_name)
            run_dir = Path(result["run_dir"])
            if news_cache.exists():
                run_dir.joinpath("news_cache.json").write_bytes(news_cache.read_bytes())
            window = {"start": "checkpoint", "stop": trigger}
            if run_replay:
                result["replay"] = {"stop": trigger, "kind": replay_kind}
            _patch_manifest(
                run_dir / "run_manifest.json", fork={"checkpoint": str(ckpt), "replay_window": window}
            )
            # the arm dir also carries the provenance manifest (the run dir is
            # a run-id subdir beneath it) so fork consumers can read the arm's
            # window without walking into the latest run
            arm_manifest = root / arm_name / "run_manifest.json"
            arm_manifest.write_text(
                json.dumps(
                    {"arm": arm_name, "fork": {"checkpoint": str(ckpt), "replay_window": window}},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        finally:
            if restore_replay is not None:
                restore_replay()
        results[arm_name] = result
    return results


def _patch_manifest(manifest_path: Path, **extra: Any) -> None:
    """Add provenance fields to an already-written run manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(extra)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# spec → sim-supported operations
# ---------------------------------------------------------------------------


def _translated_config(spec: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
    """Return a NEW config carrying the intervention delta.

    ``set_agent_state`` is applied by the sim at run time against target
    agents, so it is delivered as a named ``policy_events`` entry the main
    loop matches per tick (same matcher convention as ``policy_event``).
    """
    config = copy.deepcopy(base_config)
    trigger = spec["trigger"]
    kwargs = spec.get("kwargs", {})
    if spec["operation"] == "update_config":
        _set_dotted(config, kwargs["key"], kwargs.get("value"))
        return config
    # set_agent_state / policy_event are delivered as a named policy_events
    # entry the sim matches per tick (its standard matcher convention)
    events = config.setdefault("policy_events", [])
    if not isinstance(events, list):
        raise InterventionSpecError("config['policy_events'] must be a list to append to")
    # operation + targets are what the kernel executor
    # (gaworld/kernel/policy_events.py) keys on — without them the entry is
    # silently treated as legacy text-only and nothing fires
    event = {
        "day": trigger["day"],
        "time": trigger["time"],
        "name": spec["name"],
        "operation": spec["operation"],
        "targets": list(spec.get("targets", [])),
        "kwargs": dict(kwargs),
    }
    events.append(event)
    return config


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    for part in dotted_key.split(".")[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[dotted_key.split(".")[-1]] = value


# ---------------------------------------------------------------------------
# arm execution
# ---------------------------------------------------------------------------


def _execute_arm(arm_name: str, config: dict[str, Any], harness: Harness, base_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    outcome = "in_progress"
    with create_run(base_dir, config) as handle:
        run_dir = handle.run_dir
        try:
            summary = dict(harness(config, run_dir) or {})
            if not (run_dir / "config.json").exists():
                # the vehicle is not required to snapshot its own config
                (run_dir / "config.json").write_text(
                    json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        except Exception as exc:  # interrupted/crashed arm: partial manifest stays
            summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        ok = bool(summary.get("ok", True))
        outcome = "completed" if ok else "incomplete"
        manifest = finalize_run(handle, outcome=outcome)
    _write_agent_state_files(run_dir)
    return {
        "arm": arm_name,
        "run_dir": str(run_dir),
        "run_id": manifest["run_id"],
        "complete": ok,
        "summary": summary,
    }


def _write_agent_state_files(run_dir: Path) -> None:
    """Materialize per-agent state records from the run's own execution log.

    The harness/apparatus records applied state changes in
    ``intervention_executions.jsonl``; the per-agent view under
    ``agent_state/`` is derived from those actual records (never fabricated).
    """
    execs = _read_jsonl(run_dir / "intervention_executions.jsonl")
    applied_by_agent: dict[int, list[dict[str, Any]]] = {}
    for rec in execs:
        if not rec.get("applied"):
            continue
        for agent_id in rec.get("targets", []):
            applied_by_agent.setdefault(agent_id, []).append(rec)
    for agent_id, records in applied_by_agent.items():
        path = run_dir / "agent_state" / f"agent_{agent_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# execution audit
# ---------------------------------------------------------------------------


def _audit_intervention(spec: dict[str, Any], baseline_dir: Path, intervention_dir: Path) -> dict[str, Any]:
    """One audit record from the run's own records — intended vs actual."""
    operation = spec["operation"]
    requested = {k: v for k, v in spec.get("kwargs", {}).items() if k != "agent_id"}
    rec: dict[str, Any] = {
        "name": spec["name"],
        "operation": operation,
        "intended_time": dict(spec["trigger"]),
        "actual_time": None,
        "targets": list(spec["targets"]),
        "requested": requested,
        "applied": False,
        "before": None,
        "after": None,
        "status": _NOT_APPLIED,
    }

    # actual execution time / applied / before / after come from the run's
    # execution log, never from the spec
    execs = [
        r
        for r in _read_jsonl(intervention_dir / "intervention_executions.jsonl")
        if r.get("name") == spec["name"]
    ]
    if execs:
        actual = execs[0]
        rec["actual_time"] = {"day": actual["day"], "time": actual["time"]}
        rec["applied"] = bool(actual.get("applied"))
        rec["before"] = actual.get("before")
        rec["after"] = actual.get("after")
        if rec["applied"]:
            rec["status"] = _APPLIED
        else:
            rec["status"] = _FAILED
            rec["reason"] = actual.get("reason", "execution reported failure")
        return rec

    if operation == "update_config":
        # config deltas land before tick 0; verify the run's own config snapshot
        if _config_value_matches(intervention_dir / "config.json", spec.get("kwargs", {})):
            rec["status"] = _APPLIED  # applied via the config delta, no sim tick
            rec["applied"] = True
        else:
            rec["reason"] = "requested config value not present in the run's config"
        return rec

    # policy_event / set_agent_state delivered without an execution record:
    # check whether any episode at/after the trigger actually diverged
    first_diff = _first_divergent_tick(baseline_dir, intervention_dir, spec["trigger"])
    if first_diff is not None:
        day, time_str = first_diff
        rec["actual_time"] = {"day": day, "time": time_str}
        rec["applied"] = True
        rec["status"] = _APPLIED
    else:
        rec["reason"] = (
            "trigger never fired: no execution record and no episode divergence at/after the trigger"
        )
    return rec


def _config_value_matches(config_path: Path, kwargs: dict[str, Any]) -> bool:
    if not config_path.exists() or "key" not in kwargs:
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    node: Any = config
    for part in kwargs["key"].split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return node == kwargs.get("value")


def _first_divergent_tick(
    baseline_dir: Path, intervention_dir: Path, trigger: dict[str, Any]
) -> "tuple[int, str] | None":
    trigger_key = (trigger["day"], _minutes(trigger["time"]))
    first: "tuple[tuple[int, int], str] | None" = None
    base_eps = _read_episodes(baseline_dir)
    for key, activity in _read_episodes(intervention_dir).items():
        if base_eps.get(key) == activity:
            continue
        tick = (key[1], _minutes(key[2]))
        if tick >= trigger_key and (first is None or tick < first[0]):
            first = (tick, key[2])
    return (first[0][0], first[1]) if first else None


def _read_episodes(run_dir: Path) -> dict["tuple[int, int, str]", str]:
    records: dict[tuple[int, int, str], str] = {}
    for path in sorted(Path(run_dir).glob("agent_*.jsonl")):
        if path.parent.name == "agent_state":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ep = json.loads(line)
                records[(ep["agent_id"], ep["day"], ep["time"])] = ep.get("final_activity", "")
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _minutes(time_str: str) -> int:
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)
