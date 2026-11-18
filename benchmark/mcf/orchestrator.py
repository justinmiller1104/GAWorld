"""M2 — thin multi-leg orchestrator: config × seed legs through the isolated-run path.

Each leg runs in its own run directory via an injected harness (offline stub in
tests; the real path is experiment.run_experiment/_execute_arm with
launch.create_run/finalize_run + canary limits). Per-leg caps follow
docs/MCF_GAWORLD.md §5 Stage 1. A violation terminates ONLY the affected leg
and preserves its manifest + evidence. Restart-per-run: no checkpoint resume.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmark.mcf.config_space import config_id
from benchmark.mcf.launch import _measured_usage
from gaworld.settings import CONFIG

DEFAULT_LEG_LIMITS: dict[str, float] = {
    "max_calls": 21000,
    "max_tokens": 11_000_000,
    "max_elapsed_seconds": 80000,
    "max_error_rate": 0.05,
    "min_error_sample": 20,
}

Harness = Callable[..., dict[str, Any]]
Scorer = Callable[..., Any]


def expand_legs(configs: list[dict[str, Any]], seeds: list[int]) -> list[tuple[dict[str, Any], int]]:
    """Full cross of configs × matched seeds; input configs are never mutated."""
    return [(copy.deepcopy(config), seed) for config in configs for seed in seeds]


def _flatten(patch: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def run_legs(
    legs: list[tuple[dict[str, Any], int]],
    *,
    harness: Harness,
    run_root: Path,
    limits: dict[str, Any] | None = None,
    scorer: Scorer | None = None,
    employed_agent_ids: tuple[int, ...] = (),
    tick_minutes: int = 10,
) -> dict[str, Any]:
    """Execute every (config, seed) leg; return one machine-readable table.

    Unknown scoring data yields an explicit unscored marker — numbers are never
    fabricated.
    """
    caps = {**DEFAULT_LEG_LIMITS, **(limits or {})}
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    model = CONFIG["llm"]["routing"]["default"]
    rows: list[dict[str, Any]] = []

    for index, (config, seed) in enumerate(legs):
        cid = config_id(_flatten(config))
        run_dir = run_root / f"leg_{index:03d}_{cid}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        try:
            result = harness(copy.deepcopy(config), run_dir, seed=seed) or {}
            harness_error: str | None = None
        except Exception as exc:  # a leg crash must not kill the other legs
            result = {}
            harness_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started

        status = "completed"
        stop_reason = ""

        if harness_error is not None:
            status = "failed"
            stop_reason = harness_error
        elif not result.get("ok", True):
            status = "failed"
            stop_reason = str(result.get("error") or "harness reported failure")

        # token honesty: measured only when BOTH counts are present and valid
        measured = _measured_usage(result.get("measured_usage"))
        raw_usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        if measured is not None:
            usage = {
                "calls": int(raw_usage.get("calls") or 0),
                "tokens_in": measured[0],
                "tokens_out": measured[1],
            }
            if raw_usage.get("errors") is not None:
                usage["errors"] = int(raw_usage["errors"])
            usage_source = "measured"
        else:
            usage = {
                "calls": int(raw_usage.get("calls") or 0),
                "tokens_in": int(raw_usage.get("tokens_in") or 0),
                "tokens_out": int(raw_usage.get("tokens_out") or 0),
            }
            if raw_usage.get("errors") is not None:
                usage["errors"] = int(raw_usage["errors"])
            usage_source = "estimated"

        if status == "completed":
            violations = []
            if usage["calls"] > caps["max_calls"]:
                violations.append(f"max_calls {usage['calls']} > {caps['max_calls']}")
            if usage["tokens_in"] + usage["tokens_out"] > caps["max_tokens"]:
                violations.append(
                    f"max_tokens {usage['tokens_in'] + usage['tokens_out']} > {caps['max_tokens']}"
                )
            if elapsed > caps["max_elapsed_seconds"]:
                violations.append(f"max_elapsed_seconds {elapsed:.0f} > {caps['max_elapsed_seconds']}")
            # live error-rate kill: driven by the canary's per-call error
            # accounting (usage_snapshot carries `errors`); estimated-only
            # usage has no error counts and is never treated as a violation
            errors = raw_usage.get("errors")
            if errors is not None and usage["calls"] >= caps["min_error_sample"]:
                error_rate = errors / usage["calls"]
                if error_rate > caps["max_error_rate"]:
                    violations.append(
                        f"max_error_rate {errors}/{usage['calls']} = {error_rate:.3f}"
                        f" > {caps['max_error_rate']}"
                    )
            if violations:
                status = "interrupted"
                stop_reason = "; ".join(violations)

        score: dict[str, Any] | None = None
        if status == "completed" and scorer is not None:
            try:
                result_score = scorer(run_dir, employed_agent_ids, tick_minutes=tick_minutes)
            except Exception:
                result_score = None
            if result_score is None:
                status = "unscored"
            else:
                score = (
                    dict(vars(result_score))
                    if not hasattr(result_score, "__dataclass_fields__")
                    else {field: getattr(result_score, field) for field in result_score.__dataclass_fields__}
                )

        outcome = status if status in ("completed", "failed", "unscored", "interrupted") else "failed"
        manifest = {
            "outcome": outcome,
            "config": config,
            "seed": seed,
            "config_id": cid,
            "launch_mode": "restart-per-run",
            "stop_reason": stop_reason,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        rows.append(
            {
                "config_id": cid,
                "config": config,
                "seed": seed,
                "model": model,
                "status": status,
                "stop_reason": stop_reason,
                "usage": usage,
                "usage_source": usage_source,
                "run_dir": str(run_dir),
                "score": score,
                "elapsed_seconds": round(elapsed, 3),
            }
        )

    return {"legs": rows}
