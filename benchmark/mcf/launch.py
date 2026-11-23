"""Opt-in canary launch integration: wrap GAWorld's ``call_llm`` so every
kernel inference flows through ONE hard-capped budget.

The wrapper sits OUTSIDE the provider (provider-internal retries at
``gaworld/llm/providers.py::_retrying`` surface as separate dispatches, each
budgeted on its own). Nothing here is imported by production paths; enabling
requires an explicit ``authorize=True`` plus either ``CONFIG["canary"]`` or
``GAWORLD_CANARY=1``.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, cast

from benchmark.mcf.canary import CanaryBudget, CanaryExceeded, CanaryLimits

# ---------------------------------------------------------------------------
# Phase 1.8 Part C: run isolation
# ---------------------------------------------------------------------------


def _new_run_id() -> str:
    """Unique per-run id: timestamp for human reading, uuid suffix for uniqueness
    even when two runs start in the same second or the clock steps back."""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass
class RunHandle:
    """One isolated run: its own id, its own output directory, its own manifest."""

    run_id: str
    base_dir: Path
    run_dir: Path
    manifest_path: Path
    config: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finalized: bool = False

    def write_manifest(self, *, outcome: str, finished_at: str = "") -> dict[str, Any]:
        """Write (or rewrite) this run's manifest into its own directory."""
        manifest = {
            "run_id": self.run_id,
            "outcome": outcome,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "output_dir": str(self.run_dir),
            "outputs": [str(self.run_dir)],
            "config": self.config,
        }
        # Comparability marker: a run that enables the gated Phase-2.0/2.1
        # realism engine changes says so explicitly, so analyses never pool
        # it with default-behaviour runs (MCF principle: comparability epochs
        # must be explicit).
        behaviour = (self.config or {}).get("behavior", {}) if isinstance(self.config, dict) else {}
        if isinstance(behaviour, dict) and (behaviour.get("mcf_realism", {}) or {}).get("enabled"):
            manifest["comparability"] = {
                "epoch_marker": "mcf_realism",
                "note": "Phase-2.0/2.1 realism-engine changes enabled "
                "(sleep commitment tier, interrupt cooldown, continue-vs-replace, "
                "睡前-tail completion); not comparable with default-config runs",
            }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest


def create_run(base_dir: str | Path, config: dict[str, Any] | None = None) -> Any:
    """Create a new, isolated run under ``base_dir`` (Phase 1.8 Contract 3).

    Each run gets a unique ``run_id`` and its own ``run_dir``; episode files
    live inside it, so a second run can never append to a previous run's
    files. A partial manifest (``outcome="in_progress"``, empty
    ``finished_at`` — the format of real interrupted pilot runs) is written
    immediately and is finalized by :func:`finalize_run`. Usable as a context
    manager: exiting without finalize leaves the partial manifest untouched
    (safe interrupted-run handling).
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id()
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    handle = RunHandle(
        run_id=run_id,
        base_dir=base,
        run_dir=run_dir,
        manifest_path=run_dir / "run_manifest.json",
        config=dict(config or {}),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    handle.write_manifest(outcome="in_progress")

    class _RunContext:
        def __enter__(self) -> RunHandle:
            return handle

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            if not handle.finalized:
                # Interrupted: leave the partial manifest (in_progress / empty
                # finished_at) in the run's own directory; nothing is appended
                # anywhere else, and the next run starts in its own directory.
                handle.write_manifest(outcome="in_progress")

    return _RunContext()


def finalize_run(handle: RunHandle, *, outcome: str = "completed") -> dict[str, Any]:
    """Write the final manifest for ``handle``: real outcome + finish timestamp."""
    handle.finalized = True
    return handle.write_manifest(outcome=outcome, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def validate_observation_window(
    records: Iterable[Mapping[str, Any]],
    window_start_min: int,
    window_end_min: int,
    days: Iterable[int] | None = None,
) -> list[str]:
    """Schedule-consistency check (Phase 1.8 Contract 3): every record's time
    must fall inside the run's observation window, and (when ``days`` is
    given) on one of the run's days. Returns human-readable problems naming
    the offending record; empty list = clean."""
    allowed_days = set(days) if days is not None else None
    problems: list[str] = []
    for record in records:
        day = int(record.get("day", 0))
        clock = str(record.get("time", ""))
        minute = -1
        if ":" in clock:
            hours, minutes = clock.split(":")
            minute = int(hours) * 60 + int(minutes)
        if allowed_days is not None and day not in allowed_days:
            problems.append(
                f"agent {record.get('agent_id')} record on day {day} {clock}: "
                f"outside the run's days {sorted(allowed_days)}"
            )
            continue
        if not (window_start_min <= minute <= window_end_min):
            problems.append(
                f"agent {record.get('agent_id')} record at {clock} (minute {minute}): "
                f"outside observation window [{window_start_min}, {window_end_min}]"
            )
    return problems


def _measured_usage(raw: Any) -> tuple[int, int] | None:
    """Extract (tokens_in, tokens_out) from an OpenAI-compatible ``usage``
    object. Returns None (→ the caller falls back to its estimate, which the
    manifest then marks as estimated) unless BOTH counts are present, plain
    non-negative integers — a malformed or incomplete usage report is never
    presented as observed."""
    if not isinstance(raw, dict):
        return None

    def _count(key: str) -> int | None:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    tokens_in = _count("prompt_tokens")
    tokens_out = _count("completion_tokens")
    if tokens_in is None or tokens_out is None:
        return None
    return tokens_in, tokens_out


def _validate_pricing(pricing: Any) -> None:
    if not pricing or not isinstance(pricing, dict):
        raise ValueError("pricing table is required (model -> {'input': $/1M, 'output': $/1M})")
    for model, rates in pricing.items():
        if not isinstance(rates, dict) or "input" not in rates or "output" not in rates:
            raise ValueError(f"pricing for {model!r} must define both 'input' and 'output' rates")


class CanaryHandle(NamedTuple):
    """The (wrapped_llm, budget) pair returned by ``prepare_canary``.

    A plain tuple: all call sites unpack two names, and the budget is also
    reachable as ``handle.budget``. (The old ``_CanaryHandle`` sniffed the
    caller's UNPACK_SEQUENCE arity out of CPython bytecode to also serve a
    three-name unpack — opcode-layout-dependent and unneeded.)
    """

    wrapped: _CanaryLLM
    budget: CanaryBudget

    @property
    def llm(self) -> _CanaryLLM:
        return self.wrapped


class _CanaryLLM:
    """Kernel-facing callable that budgets every dispatch against one ledger."""

    def __init__(
        self,
        base_llm: Callable[..., Any],
        budget: CanaryBudget,
        *,
        model: str,
        dry_run: bool,
        est_tokens_in: int,
        est_tokens_out: int,
    ):
        self._base = base_llm
        self.budget = budget
        self._model = model
        self._dry_run = dry_run
        self._est_in = est_tokens_in
        self._est_out = est_tokens_out
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        # Post-kill latch: the ledger is frozen once a call pushes recorded
        # usage to the ceiling; one drain dispatch is allowed, then the
        # terminal error re-raises for every further call.
        self._terminal: str | None = None
        self._grace_used = False
        # Per-dispatch wall-clock latency of real provider calls (dry runs
        # contribute nothing); consumed by the canary report, not enforcement.
        self.latencies: list[float] = []
        # Measured vs estimated token accounting: calls whose provider
        # response carried a `usage` object are recorded from the OBSERVED
        # counts; everything else falls back to the conservative estimate
        # and is marked as such — estimates are never presented as observed.
        # Pre-call reservations stay estimate-based either way (optimistic-
        # free projection must not depend on the previous call's report).
        self.token_accounting = {
            "measured_calls": 0,
            "estimated_calls": 0,
            "tokens_measured": 0,
            "tokens_estimated": 0,
        }

    def __call__(self, prompt, task=None, agent_id=None, **kwargs):
        limits = self.budget.limits
        if self._terminal is not None:
            if self._grace_used:
                raise CanaryExceeded(self._terminal, self.budget.usage_snapshot())
            self._grace_used = True
            return self._base(prompt, task=task, agent_id=agent_id, **kwargs)
        est = self._est_in + self._est_out
        est_cost = limits.token_cost(self._model, self._est_in, self._est_out)
        with self._lock:
            call_id = f"{task or 'call'}-{next(self._ids)}"
            usage = self.budget.usage_snapshot()
            if usage["elapsed"] > limits.max_elapsed_seconds:
                raise CanaryExceeded(
                    f"elapsed budget exceeded: {usage['elapsed']:.1f}s > max_elapsed_seconds={limits.max_elapsed_seconds}",
                    usage,
                )
            decision = self.budget.check(call_number=usage["calls"] + 1, est_tokens=est, est_cost=est_cost)
            if not decision.allowed:
                raise CanaryExceeded(decision.reason, self.budget.usage_snapshot())
            self.budget.reserve(call_id, est_tokens=est)

        if self._dry_run:
            # Zero provider dispatch; keep synthetic accounting so the run
            # still hits every ceiling exactly as a live run would.
            self.budget.record(
                call_id, ok=True, tokens_in=self._est_in, tokens_out=self._est_out, latency_s=0.0
            )
            return f"[canary-dry-run] {prompt[:80]}"

        started = time.perf_counter()
        try:
            result = self._base(prompt, task=task, agent_id=agent_id, **kwargs)
        except CanaryExceeded:
            self.budget.release(call_id)
            raise
        except Exception:
            # A failed provider dispatch still consumed a budget slot.
            self.budget.record(call_id, ok=False, tokens_in=0, tokens_out=0, latency_s=0.0)
            raise
        latency = time.perf_counter() - started
        self.latencies.append(latency)
        # Prefer the backend's observed usage for the LEDGER; the estimate
        # remains the fallback and is always what the pre-call projection used.
        from gaworld.llm.providers import pop_last_token_usage

        measured = _measured_usage(pop_last_token_usage())
        if measured is not None:
            tokens_in, tokens_out = measured
            self.token_accounting["measured_calls"] += 1
            self.token_accounting["tokens_measured"] += tokens_in + tokens_out
        else:
            tokens_in, tokens_out = self._est_in, self._est_out
            self.token_accounting["estimated_calls"] += 1
            self.token_accounting["tokens_estimated"] += tokens_in + tokens_out
        self.budget.record(call_id, ok=True, tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency)
        # At-ceiling kill: recorded usage reaching the token ceiling ends the
        # run even at exact equality (strict-> keeps "exactly at limit" alive
        # only for cost, where the boundary call itself must succeed).
        usage = self.budget.usage_snapshot()
        if usage["tokens"] >= limits.max_tokens:
            self.budget.freeze_usage()
            reason = f"token budget exceeded: {usage['tokens']} >= max_tokens={limits.max_tokens}"
            self._terminal = reason
            raise CanaryExceeded(reason, self.budget.usage_snapshot())
        return result


def prepare_canary(
    config: dict[str, Any],
    limits: CanaryLimits,
    pricing: Any,
    *,
    authorize: bool,
    dry_run: bool | None = None,
    base_llm: Callable[..., Any] | None = None,
) -> tuple[_CanaryLLM, CanaryBudget]:
    """Wrap ``base_llm`` (default: the real ``gaworld.llm.providers.call_llm``)
    with the canary budget. Live launch requires ``authorize=True`` and a
    verified ``pricing`` table; each prepared canary is fully independent.

    ``dry_run=None`` (auto): when no base_llm is injected the wrapper runs in
    dry-run mode so preparing a canary can never fire inference by accident."""
    if not authorize:
        raise PermissionError("canary launch requires authorize=True")
    _validate_pricing(pricing)
    injected = base_llm is not None
    if base_llm is None:
        from gaworld.llm.providers import call_llm

        base_llm = call_llm  # lazy; keeps tests network-free

    canary_cfg = config.get("canary", {}) if isinstance(config, dict) else {}
    model = config.get("model") or next(iter(limits.pricing or {}))
    est_in = int(canary_cfg.get("est_tokens_in", getattr(base_llm, "tokens_in", 0)))
    est_out = int(canary_cfg.get("est_tokens_out", getattr(base_llm, "tokens_out", 0)))
    budget = CanaryBudget(limits)
    if dry_run is None:
        dry_run = not injected
    wrapped = _CanaryLLM(
        base_llm,
        budget,
        model=model,
        dry_run=dry_run,
        est_tokens_in=est_in,
        est_tokens_out=est_out,
    )
    return CanaryHandle(wrapped, budget)


def maybe_canary_llm(
    base_llm: Callable[..., Any],
    config: dict[str, Any],
    limits: CanaryLimits,
    pricing: Any,
    authorize: bool = False,
    dry_run: bool | None = None,
):
    """Launch seam: without explicit enablement (``config["canary"]["enabled"]``
    or ``GAWORLD_CANARY=1``) the ORIGINAL llm passes through untouched.
    Enabled but unauthorized raises; enabled + authorized returns the wrapped
    llm and its budget."""
    canary_cfg = config.get("canary", {}) if isinstance(config, dict) else {}
    enabled = bool(canary_cfg.get("enabled")) or os.environ.get("GAWORLD_CANARY") == "1"
    if not enabled:
        return base_llm
    return prepare_canary(config, limits, pricing, authorize=authorize, dry_run=dry_run, base_llm=base_llm)


def canary_manifest(budget: CanaryBudget, reason: str, *, dry_run: bool = False) -> dict[str, Any]:
    """JSON-serializable partial-run manifest, safe to write after a kill."""
    latencies = list(getattr(budget, "latencies", ()) or ())
    manifest: dict[str, Any] = {
        "dry_run": dry_run,
        "reason": reason,
        "usage": budget.usage_snapshot(),
        "latency_seconds": {
            "n": len(latencies),
            "avg": sum(latencies) / len(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
    }
    if latencies:
        s = sorted(latencies)
        manifest["latency_seconds"]["median"] = s[len(s) // 2]
    accounting = getattr(budget, "token_accounting", None)
    if accounting is not None:
        manifest["token_accounting"] = dict(accounting)
    return manifest


# ---------------------------------------------------------------------------
# Opt-in run integration: install the wrapper on gaworld.llm.providers.call_llm
# ---------------------------------------------------------------------------


def install_canary(config: dict[str, Any], *, authorize: bool = False):
    """Rebind ``gaworld.llm.providers.call_llm`` (module attribute) to the
    budgeted wrapper. This is the ONLY enforcement point that covers every
    kernel inference path: the tick loop, routines, memory and news modules
    all dispatch via ``providers.call_llm`` module-attribute access, and
    from-import sites (e.g. ``generative_city_sim``) capture whatever the
    attribute holds at their import time — so install BEFORE importing the
    entry module. Returns ``(budget, restore)``; without explicit enablement
    (``config["canary"]["enabled"]`` or ``GAWORLD_CANARY=1``) returns
    ``(None, no_op)`` and production behaviour is untouched."""
    canary_cfg = config.get("canary", {}) if isinstance(config, dict) else {}
    enabled = bool(canary_cfg.get("enabled")) or os.environ.get("GAWORLD_CANARY") == "1"
    if not enabled:
        return None, lambda: None
    import gaworld.llm.providers as providers

    # Stage-scoped caps (a smoke test runs tighter than the default canary).
    override_keys = (
        "max_calls",
        "max_tokens",
        "max_cost_usd",
        "max_elapsed_seconds",
        "max_error_rate",
        "min_error_sample",
    )
    limits = CanaryLimits(
        pricing=canary_cfg.get("pricing"),
        **{k: canary_cfg[k] for k in override_keys if k in canary_cfg},
    )
    wrapped, budget = prepare_canary(
        config,
        limits,
        canary_cfg.get("pricing"),
        authorize=authorize,
        dry_run=bool(canary_cfg.get("dry_run")),
        base_llm=providers.call_llm,
    )
    original = providers.call_llm
    providers.call_llm = cast(Callable[..., Any], wrapped)
    # Latency samples ride on the budget so the post-kill manifest reports them.
    budget.latencies = wrapped.latencies
    # Measured-vs-estimated token accounting likewise: the manifest is written
    # from the budget, the accounting lives on the wrapper.
    budget.token_accounting = wrapped.token_accounting

    def restore() -> None:
        providers.call_llm = original

    return budget, restore


def run_canary_simulation(*, authorize: bool = False, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The canary entry point: one bounded simulation run under the budget.

    Installs the wrapper first, then imports and runs the real entry module
    (its module-level ``from ... import call_llm`` therefore captures the
    wrapped callable). ``CanaryExceeded`` ends the run; the partial outputs
    already written by the recorder survive, the original provider is always
    restored, and a JSON manifest with the termination reason lands in
    ``output/canary/``. No global provider state leaks: ``restore()`` runs in
    ``finally`` before the manifest is returned.
    """
    from gaworld.settings import CONFIG as live_config

    cfg = config if config is not None else live_config
    budget, restore = install_canary(cfg, authorize=authorize)
    if budget is None:
        raise PermissionError(
            "canary not enabled: set CONFIG['canary'] = {'enabled': true, 'pricing': ...} or GAWORLD_CANARY=1"
        )
    dry_run = bool(cfg.get("canary", {}).get("dry_run")) if isinstance(cfg, dict) else False
    reason = "completed"
    try:
        import generative_city_sim  # imported AFTER install so from-imports bind the wrapper

        generative_city_sim.run_simulation()
    except CanaryExceeded as exc:
        reason = exc.reason
    finally:
        restore()
    manifest = canary_manifest(budget, reason, dry_run=dry_run)
    out_dir = Path("output/canary")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"canary_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(path)
    return manifest


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run the MCF canary (opt-in, hard budget caps).")
    parser.add_argument("--authorize", action="store_true", help="explicit authorization for a LIVE run")
    parser.add_argument("--dry-run", action="store_true", help="zero-inference plumbing check")
    args = parser.parse_args()
    if args.dry_run and args.authorize:
        parser.error("--dry-run and --authorize are mutually exclusive")
    if args.dry_run:
        os.environ.setdefault("GAWORLD_CANARY", "1")

        from gaworld.settings import CONFIG as _cfg

        canary = _cfg.setdefault("canary", {})
        canary["dry_run"] = True
        # Local canary (LM Studio, 127.0.0.1:1234): billed API spend is genuinely
        # $0, so the mandatory pricing table is zero-cost by local rationale —
        # no fabricated rates to satisfy the $1 ceiling.
        canary.setdefault("model", "qwen/qwen3-4b-instruct-2507")
        canary.setdefault("pricing", {"qwen/qwen3-4b-instruct-2507": {"input": 0.0, "output": 0.0}})
        result = run_canary_simulation(authorize=True)
    else:
        result = run_canary_simulation(authorize=args.authorize)
    print(json.dumps(result, indent=2))
