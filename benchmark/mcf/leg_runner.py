"""Real per-leg MCF runner (task 3753): one (config, seed) leg executed in an
isolated run dir under unified canary budget control.

This is the execution vehicle ``gaworld/apps/mcf_api.py:_build_harness`` hands
to ``orchestrator.run_legs``. Per leg:

- an isolated run dir via ``launch.create_run`` (restart-per-run, partial
  manifest kept as evidence when the leg dies);
- the in-execution canary budget (``canary.CanaryLimits`` through
  ``launch.prepare_canary``) rebound on ``gaworld.llm.providers.call_llm`` —
  caps are enforced BEFORE each dispatch, and a leg that runs past them dies
  with ``CanaryExceeded`` instead of burning its whole budget;
- the actual simulator executed through the existing entry module
  (``generative_city_sim.run_simulation``) with the config patch applied to
  the live ``CONFIG`` for the duration — the same launch path the canary CLI
  uses; no new execution or resume mechanism is invented;
- on completion the run manifest is finalized with the leg's seed and model
  info; usage is reported from the canary ledger (measured token accounting
  flows through the wrapper), never fabricated.

The offline stub (``stub=True`` or ``GAWORLD_OFFLINE=1``) writes real
episode-span shapes deterministically derived from the config patch so
``time_use.score_directory`` scores the leg through the real scorer with zero
inference; the canary runs in dry-run mode, so even a runaway stub leg is
budget-killed honestly.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmark.mcf.canary import CanaryExceeded, CanaryLimits
from benchmark.mcf.launch import create_run, finalize_run, prepare_canary
from benchmark.mcf.orchestrator import DEFAULT_LEG_LIMITS

__all__ = ["make_leg_harness", "run_leg", "sim_leg_execute", "stub_leg_execute"]

Execute = Callable[..., dict[str, Any]]

# the CanaryLimits caps a leg may carry (orchestrator's min_error_sample stays
# an orchestrator-side post-check; the canary also accepts it, so pass through)
_LIMIT_KEYS = (
    "max_calls",
    "max_tokens",
    "max_cost_usd",
    "max_elapsed_seconds",
    "max_error_rate",
    "min_error_sample",
)

_ZERO_PRICING = {"mcf-leg": {"input": 0.0, "output": 0.0}}

#: The stub cohort's employed agents (participants basis): paid_work is
#: scored only over employed agent-days, so the scorer must know who is
#: employed or every paid_work error reads as ``None`` (mcf_api passes this
#: to ``run_legs`` when the stub harness is active).
STUB_EMPLOYED_AGENTS = (1, 2)


def _flatten(patch: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _offline() -> bool:
    return os.environ.get("GAWORLD_OFFLINE") == "1"


# ---------------------------------------------------------------------------
# execution vehicles
# ---------------------------------------------------------------------------


def _span(agent_id: int, day: int, start: str, end: str, activity: str) -> dict[str, Any]:
    return {
        "kind": "span",
        "agent_id": agent_id,
        "day": day,
        "start": start,
        "end": end,
        "activity": activity,
        "location": "",
        "purpose": "",
    }


def _clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def stub_leg_execute(
    config: dict[str, Any], run_dir: Path, *, seed: int = 0, sim_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Deterministic offline leg: episode-span files whose time use is a pure
    function of the flattened config patch, scored by the real scorer.

    A higher ``style_fit_amplitude`` pushes paid-work minutes toward the
    survey target (lower paid_work_error); other categories are fixed. The
    canary dry-run wrapper's per-call synthetic accounting is the leg's
    honest usage record.
    """
    flat = _flatten(config)
    style = float(flat.get("personality.style_fit_amplitude", 0.10))
    # one canary-ledgered dispatch: the leg's (dry-run) inference record —
    # the stub burns budget through the same wrapper a live leg would
    import gaworld.llm.providers as providers

    providers.call_llm("mcf-stub-leg", task="leg")
    n_agents, n_days = 4, 8
    # paid-work span grows with the patch: 08:00 + 120·style hours of 工作
    paid_minutes = int(120 + 480 * style)
    work_start = 8 * 60
    work_end = work_start + paid_minutes
    for agent_id in range(1, n_agents + 1):
        rows = []
        for day in range(1, n_days + 1):
            rows.append(_span(agent_id, day, "00:00", "08:00", "睡觉"))
            rows.append(_span(agent_id, day, _clock(work_start), _clock(work_end), "工作"))
            rows.append(_span(agent_id, day, _clock(work_end), "19:00", "娱乐"))
            rows.append(_span(agent_id, day, "19:00", "20:00", "社交"))
            rows.append(_span(agent_id, day, "20:00", "21:00", "吃饭"))
            rows.append(_span(agent_id, day, "21:00", _clock(24 * 60), "睡觉"))
        path = Path(run_dir) / f"agent_{agent_id}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "ok": True,
        "usage": {},
        "agents": n_agents,
        "days": n_days,
        "episodes_dir": str(run_dir),
    }


def sim_leg_execute(
    config: dict[str, Any], run_dir: Path, *, seed: int = 0, sim_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The REAL execution vehicle: apply the nested config patch to the live
    CONFIG, seed it through the only wired mechanism
    (``CONFIG["random_seed"]``), snapshot the config into the run dir, and run
    the actual simulator entry module — the same launch path
    ``launch.run_canary_simulation`` uses.

    ``sim_context`` (population integration): ``city`` / ``csv_path`` /
    ``map_path`` overrides + ``agent_ids`` + ``sim_days`` from the resolved
    experiment population. These are the exact CONFIG keys the simulator
    consumes (``AGENT_IDS`` / ``SIM_DAYS`` / ``CSV_PATH`` are module-level
    bindings derived from CONFIG at import time), so after the entry module is
    imported the corresponding module globals are rebound for the duration of
    the leg — the same rebinding pattern the call_llm wrapper already uses —
    and restored afterwards."""
    import gaworld.llm.providers as providers
    from gaworld.settings import CONFIG

    original = {key: json.loads(json.dumps(CONFIG.get(key))) for key in config}
    sim_original: dict[str, Any] = {}
    if sim_context:
        for key in ("city", "csv_path", "map_path", "real_map_path", "md_path", "agent_ids", "sim_days"):
            if key in sim_context:
                sim_original[key] = json.loads(json.dumps(CONFIG.get(key)))
                CONFIG[key] = json.loads(json.dumps(sim_context[key]))
    original_seed = CONFIG.get("random_seed")
    try:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(CONFIG.get(key), dict):
                CONFIG[key].update(json.loads(json.dumps(value)))
            else:
                CONFIG[key] = json.loads(json.dumps(value))
        CONFIG["random_seed"] = int(seed)
        # the sim emits episode files under CONFIG["memory_dir"] (default
        # output/memory); point it at this leg's run dir so _surface_episodes
        # can hand the real spans to the scorer — without it a real leg always
        # scored as empty
        original_memory_dir = CONFIG.get("memory_dir")
        CONFIG["memory_dir"] = str(run_dir)
        (Path(run_dir) / "config.json").write_text(
            json.dumps(json.loads(json.dumps(CONFIG)), ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
        # from-import of call_llm captures the wrapped callable
        # the entry module binds call_llm at its FIRST import, so leg 2+ would
        # keep calling leg 1's exhausted canary wrapper (instant budget kill);
        # rebind to this leg's current (wrapped) callable and restore after.
        # gaworld.cognition.realism has the same top-level from-import and is
        # on the per-tick cognition path — rebind it too
        import gaworld.cognition.realism as _realism
        import generative_city_sim  # imported after the canary install so its

        original_sim_llm = generative_city_sim.__dict__.get("call_llm")
        original_realism_llm = _realism.__dict__.get("call_llm")
        generative_city_sim.call_llm = providers.call_llm
        _realism.call_llm = providers.call_llm
        # population integration: the module bound AGENT_IDS / SIM_DAYS /
        # CSV_PATH / MAP_PATH from CONFIG at import time; rebind them from this
        # leg's sim context so the engine simulates exactly the selected city,
        # residents and duration (restored after the leg, like call_llm).
        sim_globals_original: dict[str, Any] = {}
        if sim_context:
            for gname, ckey in (
                ("_BASE_AGENT_IDS", "agent_ids"),
                ("AGENT_IDS", "agent_ids"),
                ("SIM_DAYS", "sim_days"),
                ("CSV_PATH", "csv_path"),
                ("MAP_PATH", "map_path"),
                ("REAL_MAP_PATH", "real_map_path"),
            ):
                if gname in generative_city_sim.__dict__:
                    sim_globals_original[gname] = generative_city_sim.__dict__[gname]
                    value = sim_context.get(ckey)
                    if ckey == "agent_ids":
                        value = [int(i) for i in (value or [])]
                    elif ckey == "sim_days":
                        value = int(value)
                    generative_city_sim.__dict__[gname] = value
        try:
            if sim_context:
                run_manifest_event = (
                    generative_city_sim.__dict__.get("AGENT_IDS"),
                    generative_city_sim.__dict__.get("SIM_DAYS"),
                )
                (Path(run_dir) / "population_applied.json").write_text(
                    json.dumps(
                        {
                            "agent_ids": list(sim_context.get("agent_ids") or []),
                            "n_agents": len(sim_context.get("agent_ids") or []),
                            "sim_days": sim_context.get("sim_days"),
                            "city": sim_context.get("city"),
                            "csv_path": sim_context.get("csv_path"),
                            "provenance": sim_context.get("population_provenance") or {},
                            "module_bindings_seen": {
                                "AGENT_IDS": list(run_manifest_event[0] or []),
                                "SIM_DAYS": run_manifest_event[1],
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            generative_city_sim.run_simulation()
        finally:
            if original_sim_llm is not None:
                generative_city_sim.call_llm = original_sim_llm
            if original_realism_llm is not None:
                _realism.call_llm = original_realism_llm
            for gname, value in sim_globals_original.items():
                generative_city_sim.__dict__[gname] = value
    finally:
        for key, value in original.items():
            if value is None:
                CONFIG.pop(key, None)
            else:
                CONFIG[key] = value
        if original_memory_dir is None:
            CONFIG.pop("memory_dir", None)
        else:
            CONFIG["memory_dir"] = original_memory_dir
        if original_seed is None:
            CONFIG.pop("random_seed", None)
        else:
            CONFIG["random_seed"] = original_seed
        for key, value in sim_original.items():
            if value is None:
                CONFIG.pop(key, None)
            else:
                CONFIG[key] = value
    return {
        "ok": True,
        "usage": {},
        "episodes_dir": str(run_dir),
        "agents": len((sim_context or {}).get("agent_ids") or []),
        "days": (sim_context or {}).get("sim_days"),
    }


# ---------------------------------------------------------------------------
# harness factory
# ---------------------------------------------------------------------------


def make_leg_harness(
    *,
    limits: dict[str, Any] | None = None,
    stub: bool | None = None,
    model_info: dict[str, Any] | None = None,
    execute: Execute | None = None,
    sim_context: dict[str, Any] | None = None,
) -> Execute:
    """Build a ``run_legs``-compatible harness ``(config, run_dir, *, seed)``.

    ``stub=None`` (auto): the offline stub runs when ``GAWORLD_OFFLINE=1``,
    otherwise the real simulator executes. ``execute=`` overrides the vehicle
    (offline tests inject deterministic stubs).
    """
    leg_limits = {**DEFAULT_LEG_LIMITS, **(limits or {})}
    if stub is None:
        stub = _offline()
    vehicle = execute or (stub_leg_execute if stub else sim_leg_execute)
    # dry-run wrapper when the leg is a stub: the canary enforces its caps on
    # synthetic accounting with zero provider dispatches, so an offline test
    # can never fire inference even through an injected chatty stub
    wrapper_dry_run = stub
    pricing = dict((model_info or {}).get("pricing") or _ZERO_PRICING)
    # the wrapper bills against pricing[model]; a named model without a rates
    # entry would KeyError on its first call — bill it under the table's key
    model = next(iter(pricing))
    if (model_info or {}).get("model") in pricing:
        model = model_info["model"]
    canary_limits = CanaryLimits(
        pricing=pricing, **{k: leg_limits[k] for k in _LIMIT_KEYS if k in leg_limits}
    )

    vehicle_kwargs: dict[str, Any] = {}
    if sim_context is not None:
        vehicle_kwargs["sim_context"] = sim_context

    def harness(config: dict[str, Any], run_dir: Path, *, seed: int = 0) -> dict[str, Any]:
        create_cfg = {"seed": seed, "config_patch": _flatten(config), "model": dict(model_info or {})}
        if sim_context is not None:
            create_cfg["population"] = dict(sim_context.get("population_provenance") or {}) | {
                "n_agents": len(sim_context.get("agent_ids") or []),
                "sim_days": sim_context.get("sim_days"),
            }
        run_ctx = create_run(
            run_dir,
            config=create_cfg,
        )
        import gaworld.llm.providers as providers

        with run_ctx as handle:
            wrapped, budget = prepare_canary(
                {"model": model, "canary": {}},
                canary_limits,
                pricing,
                authorize=True,
                dry_run=wrapper_dry_run,
                base_llm=providers.call_llm,
            )
            original_llm = providers.call_llm
            providers.call_llm = wrapped
            outcome = "completed"
            error_msg = ""
            result: dict[str, Any] = {}
            try:
                result = dict(vehicle(config, handle.run_dir, seed=seed, **vehicle_kwargs) or {})
            except CanaryExceeded as exc:
                # budget kill: honest stop reason, usage from the frozen ledger
                outcome = "interrupted"
                error_msg = f"budget: {exc.reason}"
            except Exception as exc:
                outcome = "failed"
                error_msg = f"{type(exc).__name__}: {exc}"
            finally:
                providers.call_llm = original_llm
            if outcome == "completed":
                manifest = finalize_run(handle, outcome="completed")
                manifest["seed"] = seed
                manifest["model"] = dict(model_info or {})
                handle.manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            else:
                # interrupted (budget kill) / failed: finalized with the real
                # outcome — the run never claims a completion it did not reach
                manifest = finalize_run(handle, outcome=outcome)
            snapshot = budget.usage_snapshot()
            if outcome == "completed":
                _surface_episodes(handle.run_dir, Path(run_dir))
        if outcome != "completed":
            return {
                "ok": False,
                "error": error_msg,
                "status": outcome,
                "usage": {"calls": snapshot["calls"], "tokens_in": snapshot["tokens"], "tokens_out": 0},
                "run_id": handle.run_id,
                "run_dir": str(handle.run_dir),
            }
        inner = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        usage = {
            "calls": snapshot["calls"] or int(inner.get("calls") or 0),
            "tokens_in": int(inner.get("tokens_in") or 0),
            "tokens_out": int(inner.get("tokens_out") or 0),
        }
        # the canary ledger tracks total tokens; when the vehicle did not
        # split them, report the ledger total as tokens_in (never dropped)
        if usage["tokens_in"] + usage["tokens_out"] == 0 and snapshot["tokens"]:
            usage["tokens_in"] = snapshot["tokens"]
        if snapshot.get("errors"):
            usage["errors"] = int(snapshot["errors"])
        return {
            "ok": True,
            "usage": usage,
            "run_id": handle.run_id,
            "run_dir": str(handle.run_dir),
            "measured_usage": (
                {"prompt_tokens": usage["tokens_in"], "completion_tokens": usage["tokens_out"]}
                if usage["tokens_in"] + usage["tokens_out"]
                else None
            ),
            "episodes_dir": result.get("episodes_dir"),
        }

    return harness


def _surface_episodes(run_subdir: Path, leg_dir: Path) -> None:
    """Copy the leg's episode files up to the orchestrator's per-leg scoring
    dir: ``run_legs`` scores the leg dir, while create_run isolation keeps the
    vehicle's outputs one level down. Same data, no re-scoring ambiguity."""
    import shutil

    for path in sorted(Path(run_subdir).glob("agent_*.jsonl")):
        if path.parent.name == "agent_state":
            continue
        shutil.copy2(path, Path(leg_dir) / path.name)


def run_leg(config: dict[str, Any], run_dir: Path, *, seed: int = 0) -> dict[str, Any]:
    """The real per-leg runner bound to Stage-1 default caps and the real
    simulator vehicle (the ``experiment.run_leg`` contract _build_harness
    looks for)."""
    return make_leg_harness()(config, run_dir, seed=seed)
