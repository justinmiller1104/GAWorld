"""M3 — real-harness adapter: bridge ``orchestrator.run_legs`` to the real
execution vehicle (``experiment._execute_arm`` shape via ``launch`` +
``canary``).

``run_legs`` calls its harness as ``harness(config, run_dir, *, seed=...)``,
while the real path wants ``(config, run_dir)`` with the leg run inside an
isolated ``launch.create_run`` directory under installed canary budget
controls and the seed applied through the ONLY wired mechanism,
``CONFIG["random_seed"]``. This adapter is that bridge — no new seeding or
resume mechanism is invented: restart-per-run (fresh ``create_run`` per
call, no checkpoint flags), and the arm's config patch stays clean.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmark.mcf.launch import create_run, finalize_run, install_canary
from gaworld.settings import CONFIG

Execute = Callable[..., dict[str, Any]]

__all__ = ["make_real_harness"]


class _ArmConfig(dict):
    """The arm's config patch as a dict, with the leg's canary limits visible
    through ``.get("canary")`` without the key being injected into the patch
    itself (the patch stays exactly what the caller passed; the canary
    config lives only in what the adapter hands ``install_canary``)."""

    def __init__(self, patch: dict[str, Any], canary_cfg: dict[str, Any]):
        super().__init__(patch)
        self._canary_cfg = canary_cfg

    def get(self, key, default=None):  # type: ignore[override]
        if key == "canary":
            return self._canary_cfg
        return super().get(key, default)


def _stamp_seed(handle: Any, seed: int) -> None:
    """Record the leg seed at the manifest's top level (create_run's manifest
    is rewritten by finalize_run, so stamp after each write)."""
    manifest = json.loads(handle.manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = seed
    handle.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _default_execute(config: dict[str, Any], run_dir: Path, *, seed: int) -> dict[str, Any]:
    """Placeholder for the real arm-execution vehicle: Stage 1 live
    execution is authorized in a separate task; inject ``execute=`` for
    offline/stub use (as the tests do)."""
    raise NotImplementedError("no inner arm execution injected — pass execute= to make_real_harness")


def make_real_harness(
    *, seed: int, limits: dict[str, Any] | None = None, execute: Execute | None = None
) -> Execute:
    """Build a run_legs-compatible harness for one (config, seed) family.

    Returns a callable ``(config, run_dir)`` (also tolerant of the ``seed=``
    keyword run_legs passes). Each call: fresh isolated run dir via
    ``launch.create_run`` (restart-per-run), ``CONFIG["random_seed"]`` set
    for the duration, canary budget controls installed with the leg limits
    threaded in, ``providers.call_llm`` restored afterwards.
    """
    leg_limits = dict(limits or {})
    arm_execute = execute or _default_execute

    def harness(config: dict[str, Any], run_dir: Path, *, seed: int = seed) -> dict[str, Any]:
        run_ctx = create_run(run_dir, config={"seed": seed, "config_patch": config})
        original_seed = CONFIG.get("random_seed")
        with run_ctx as handle:
            _stamp_seed(handle, seed)
            canary_cfg = {**leg_limits, "enabled": True}
            if not canary_cfg.get("pricing"):
                # a leg-scoped canary bills nothing by itself; the zero-cost
                # table satisfies the mandatory pricing contract without
                # fabricating rates for a billed run (real pricing rides in
                # with limits when Stage 1 authorizes live execution)
                canary_cfg["pricing"] = {CONFIG["llm"]["routing"]["default"]: {"input": 0.0, "output": 0.0}}
            budget, restore = install_canary({**config, "canary": canary_cfg}, authorize=True)
            try:
                CONFIG["random_seed"] = seed
                result = dict(arm_execute(_ArmConfig(config, canary_cfg), handle.run_dir, seed=seed) or {})
            except Exception as exc:
                # crash: leave the partial (in_progress) manifest as evidence,
                # skip finalize, report failure to run_legs
                restore()
                CONFIG["random_seed"] = original_seed
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "usage": {},
                    "run_id": handle.run_id,
                    "run_dir": str(handle.run_dir),
                }
            finally:
                restore()
                CONFIG["random_seed"] = original_seed

        outcome = "completed" if result.get("ok", True) else "incomplete"
        finalize_run(handle, outcome=outcome)
        _stamp_seed(handle, seed)

        budget_snapshot = budget.usage_snapshot() if budget is not None else {}
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        merged = {
            "calls": int(usage.get("calls") or budget_snapshot.get("calls") or 0),
            "tokens_in": int(usage.get("tokens_in") or 0),
            "tokens_out": int(usage.get("tokens_out") or 0),
        }
        # canary error accounting flows to run_legs for max_error_rate
        # enforcement; with no canary there is no error count and none is
        # fabricated
        if "errors" in usage or budget_snapshot.get("errors") is not None:
            merged["errors"] = int(usage.get("errors") or budget_snapshot.get("errors") or 0)
        result["usage"] = merged
        result["run_id"] = handle.run_id
        result["run_dir"] = str(handle.run_dir)
        return result

    return harness
