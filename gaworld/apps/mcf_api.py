"""HTTP surface for the MCF research workflow (``benchmark/mcf``).

One experiment = a set of behaviour architectures (M0 baseline, M1 realism, …)
swept over the admissible parameter bands of ``benchmark.mcf.config_space``,
each (architecture, config, seed) leg scored against the observed time-use
targets. Experiments persist as JSON under ``output/mcf/<id>/``; results are
computed by the real ``benchmark.mcf.frontier`` code from the leg table the
orchestrator writes — the API never fabricates numbers.

The benchmark modules are imported lazily inside the handlers: a parallel
worker is still extending them (``benchmark/mcf/sampler.py``, k-dimensional
vectors, the per-leg harness), so a missing capability answers a clear 501
("not yet available in this build") instead of crashing.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any

from gaworld.accounts import ownership
from gaworld.apps import world_paths

#: Measured on the canary14 run (~280 calls per agent-day leg, docs §5).
_ESTIMATED_CALLS_PER_LEG = 280
#: Above this estimate a launch needs an explicit ``confirm_cost: true``.
_COST_CONFIRM_THRESHOLD = 2000
_MAX_CONFIGS = 64
_MAX_SEEDS = 32

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _mcf_root() -> str:
    return os.path.join(world_paths.REPO_ROOT, "output", "mcf")


def _exp_dir(root: str, exp_id: str) -> str:
    return os.path.join(root, exp_id)


def _exp_path(root: str, exp_id: str) -> str:
    return os.path.join(_exp_dir(root, exp_id), "experiment.json")


def _load_exp(exp_id: str) -> dict[str, Any] | None:
    path = _exp_path(_mcf_root(), exp_id)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_exp(exp: dict[str, Any]) -> None:
    path = _exp_path(_mcf_root(), exp["id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------- bands


def bands() -> dict[str, Any]:
    """Admissible parameter ranges with their dotted config paths and sources."""
    from benchmark.mcf import config_space

    out = []
    for name, band in config_space.BANDS.items():
        out.append(
            {
                "name": name,
                "path": band["path"],
                "low": band["low"],
                "high": band["high"],
                "source": band.get("source", ""),
            }
        )
    return {"bands": out}


class _Unavailable(Exception):
    """A capability the parallel benchmark worker has not landed yet."""


# ---------------------------------------------------------------- creation


def _bands_by_name() -> dict[str, dict[str, Any]]:
    from benchmark.mcf import config_space

    return config_space.BANDS


def _validate_spec(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Check the create payload; return (spec, error-or-None)."""
    name = str(payload.get("name") or "").strip()
    if not name:
        return {}, "name is required"

    archs = payload.get("architectures") or []
    if not isinstance(archs, list) or len(archs) < 2:
        return {}, "at least two architectures are required (M0 baseline and a treatment)"
    clean_archs = []
    for arch in archs:
        arch_id = str(arch.get("id") or "").strip()
        if not arch_id:
            return {}, "every architecture needs an id"
        patch = dict(arch.get("patch") or {})
        try:
            from benchmark.mcf.config_space import validate_arch_patch

            validate_arch_patch(patch)
        except ValueError as exc:
            return {}, str(exc)
        clean_archs.append(
            {
                "id": arch_id,
                "label": str(arch.get("label") or arch_id),
                "patch": patch,
            }
        )

    all_bands = _bands_by_name()
    params = payload.get("parameters")
    if params is None:
        params = []
    if not isinstance(params, list):
        return {}, "parameters must be a list of band names"
    unknown = [p for p in params if p not in all_bands]
    if unknown:
        return {}, f"unknown configuration parameters: {', '.join(map(str, unknown))}"

    sampler = str(payload.get("sampler") or "endpoints")
    if sampler not in ("endpoints", "sobol"):
        return {}, "sampler must be 'endpoints' or 'sobol'"
    harness = str(payload.get("harness") or "real")
    if harness not in ("real", "stub"):
        return {}, "harness must be 'real' or 'stub'"
    if sampler == "sobol":
        try:
            import benchmark.mcf.sampler  # noqa: F401
        except ImportError as exc:
            raise _Unavailable(f"sobol sampling not yet available in this build ({exc})") from None

    objectives = list(payload.get("objectives") or ["paid_work_error", "leisure_social_error"])

    n_configs = max(1, min(int(payload.get("n_configs") or 3), _MAX_CONFIGS))
    seeds = [int(s) for s in (payload.get("seeds") or [])][:_MAX_SEEDS]
    n_seeds = int(payload.get("n_seeds") or 0) or len(seeds)
    if n_seeds < 1:
        n_seeds = 1
    defaults = [7 + 4 * i for i in range(n_seeds)]
    if not seeds:
        seeds = defaults
    while len(seeds) < n_seeds:  # extend deterministically, never duplicate
        for candidate in defaults:
            if candidate not in seeds:
                seeds.append(candidate)
                if len(seeds) >= n_seeds:
                    break

    reference = str(payload.get("reference_architecture") or clean_archs[0]["id"])
    if reference not in {a["id"] for a in clean_archs}:
        return {}, "reference_architecture must be one of the architectures"

    budget = dict(payload.get("budget") or {})
    employed_agent_ids = [int(i) for i in (payload.get("employed_agent_ids") or [])]
    spec = {
        "name": name,
        "note": str(payload.get("note") or ""),
        "architectures": clean_archs,
        "reference_architecture": reference,
        "parameters": list(params),
        "objectives": objectives,
        "sampler": sampler,
        "harness": harness,
        "employed_agent_ids": employed_agent_ids,
        "n_configs": n_configs,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "budget": budget,
        "model": dict(payload.get("model") or {}),
    }
    return spec, None


def _sample_configs(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Config patches (flat dotted → value) swept over the chosen bands."""

    all_bands = _bands_by_name()
    chosen = [all_bands[name] for name in spec["parameters"]]
    baseline = {band["path"]: band["low"] for band in chosen}
    configs = [dict(baseline)]
    for band in chosen:
        step = dict(baseline)
        step[band["path"]] = band["high"]
        configs.append(step)
    if spec["sampler"] == "sobol":
        import benchmark.mcf.sampler as sampler_mod

        points = sampler_mod.sample(spec["n_configs"], chosen)
        configs = [
            {band["path"]: value for band, value in zip(chosen, point, strict=True)} for point in points
        ]
    return configs[: spec["n_configs"]]


def _legs_total(spec: dict[str, Any]) -> int:
    return len(spec["architectures"]) * spec["n_configs"] * spec["n_seeds"]


def _estimate_calls(spec: dict[str, Any]) -> int:
    return _legs_total(spec) * _ESTIMATED_CALLS_PER_LEG


def create_experiment(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    try:
        spec, error = _validate_spec(payload)
    except _Unavailable as exc:
        return {"error": str(exc)}, 501
    if error:
        return {"error": error}, 400
    exp_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    exp = {
        "id": exp_id,
        "created_at": time.time(),
        "status": "draft",
        "progress": {"legs_done": 0, "legs_total": _legs_total(spec), "failures": []},
        "estimated_calls": _estimate_calls(spec),
        "last_error": "",
        **spec,
    }
    _save_exp(exp)
    return exp, 201


# ---------------------------------------------------------------- launch


def _start_job(exp_id: str) -> None:
    ownership.spawn(_run_job, exp_id, name=f"mcf-{exp_id}")


def launch_experiment(exp_id: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    exp = _load_exp(exp_id)
    if exp is None:
        return {"error": "unknown experiment"}, 404
    if exp["status"] == "running":
        return {"error": "experiment is already running"}, 409
    estimated = exp["estimated_calls"]
    if estimated > _COST_CONFIRM_THRESHOLD and not payload.get("confirm_cost"):
        return (
            {
                "error": (
                    f"estimated {estimated} model calls exceeds the "
                    f"{_COST_CONFIRM_THRESHOLD}-call guard; retry with confirm_cost=true"
                ),
                "estimated_calls": estimated,
            },
            400,
        )
    exp["status"] = "running"
    exp["progress"] = {"legs_done": 0, "legs_total": _legs_total(exp), "failures": []}
    exp["last_error"] = ""
    _save_exp(exp)
    job_id = uuid.uuid4().hex[:8]
    with _LOCK:
        _JOBS[job_id] = {"id": job_id, "experiment": exp_id, "status": "running", "started_at": time.time()}
    _start_job(exp_id)
    return {"job_id": job_id, "experiment_id": exp_id, "estimated_calls": estimated}, 202


def _is_stub(exp: dict[str, Any]) -> bool:
    return exp.get("harness") == "stub" or os.environ.get("GAWORLD_OFFLINE") == "1"


def _build_harness(exp: dict[str, Any]):
    """The real per-leg runner (``benchmark.mcf.leg_runner``): isolated run
    dir, in-execution canary budget, the actual simulator vehicle. The stub
    execution (explicit ``harness: "stub"`` experiment option, or a forced
    ``GAWORLD_OFFLINE=1``) scores deterministic legs through the real scorer
    with zero inference — tests and dry-runs can never fire live calls."""
    from benchmark.mcf.leg_runner import make_leg_harness

    limits = {k: v for k, v in (exp.get("budget") or {}).items() if v is not None}
    return make_leg_harness(
        limits=limits or None, stub=_is_stub(exp), model_info=dict(exp.get("model") or {})
    )


def _build_scorer(exp: dict[str, Any]):
    from benchmark.mcf import time_use

    scorer = getattr(time_use, "score_directory", None)
    if scorer is None:
        return None
    return scorer


def _run_job(exp_id: str) -> None:
    exp = _load_exp(exp_id)
    if exp is None:
        return
    try:
        harness = _build_harness(exp)
    except Exception as exc:
        exp["status"] = "failed"
        exp["last_error"] = str(exc)
        _save_exp(exp)
        return
    try:
        from benchmark.mcf.config_space import apply_patch, validate_arch_patch, validate_config
        from benchmark.mcf.orchestrator import run_legs

        scorer = _build_scorer(exp)
        limits = {k: v for k, v in (exp.get("budget") or {}).items() if v is not None}
        rows: list[dict[str, Any]] = []
        root = _mcf_root()
        for arch in exp["architectures"]:
            # The arch patch is mechanism-level (Θ band paths are rejected at
            # spec level) so it validates separately from the sampled Θ patch.
            base = validate_arch_patch({str(path): value for path, value in arch.get("patch").items()})
            leg_pairs = []
            for cfg in _sample_configs(exp):
                nested = validate_config(cfg)
                apply_patch(nested, base)
                for seed in exp["seeds"]:
                    leg_pairs.append((nested, seed))
            run_root = os.path.join(_exp_dir(root, exp_id), arch["id"])
            # stub legs declare their built-in cohort; real legs take the
            # cohort declared in the spec (participants basis): without it
            # paid_work scores as error=None and the frontier assembler
            # rightly refuses the vector
            from benchmark.mcf.leg_runner import STUB_EMPLOYED_AGENTS

            if _is_stub(exp):
                employed = STUB_EMPLOYED_AGENTS
            else:
                employed = tuple(int(i) for i in (exp.get("employed_agent_ids") or []))
            table = run_legs(
                leg_pairs,
                harness=harness,
                run_root=run_root,
                limits=limits or None,
                scorer=scorer,
                employed_agent_ids=employed,
            )
            for row in table.get("legs", []):
                row["arch"] = arch["id"]
                row["config_patch_flat"] = _flatten_patch(row.get("config") or {})
                rows.append(row)
            exp["progress"]["legs_done"] = len(rows)
            failures = [r for r in rows if r["status"] not in ("completed",)]
            exp["progress"]["failures"] = [
                {
                    "arch": r["arch"],
                    "config_id": r["config_id"],
                    "seed": r["seed"],
                    "status": r["status"],
                    "stop_reason": r["stop_reason"],
                }
                for r in failures
            ]
            _save_exp(exp)
            legs_path = os.path.join(_exp_dir(root, exp_id), "legs.json")
            tmp = legs_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            os.replace(tmp, legs_path)
        exp["status"] = "completed" if not exp["progress"]["failures"] else "completed_with_failures"
        _save_exp(exp)
        # persist the reopenable experiment manifest (plan step 12) — the GUI
        # results endpoint reads it back via benchmark.mcf.manifest.load_manifest
        try:
            from benchmark.mcf.manifest import build_manifest, save_manifest

            try:
                results = _compute_results(exp)
            except Exception:
                results = None
            save_manifest(
                os.path.join(_exp_dir(_mcf_root(), exp_id), "manifest.json"),
                build_manifest(exp, rows, results=results),
            )
        except Exception:
            pass
    except Exception as exc:
        exp["status"] = "failed"
        exp["last_error"] = f"{type(exc).__name__}: {exc}"
        _save_exp(exp)


def _flatten_patch(patch: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_patch(value, path))
        else:
            flat[path] = value
    return flat


# ---------------------------------------------------------------- results


def _load_legs(exp_id: str) -> list[dict[str, Any]]:
    path = os.path.join(_exp_dir(_mcf_root(), exp_id), "legs.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _supported_objectives(exp: dict[str, Any]) -> bool:
    return list(exp.get("objectives") or ["paid_work_error", "leisure_social_error"]) == [
        "paid_work_error",
        "leisure_social_error",
    ]


def _compute_results(exp: dict[str, Any]) -> dict[str, Any]:
    from benchmark.mcf import frontier

    if not _supported_objectives(exp):
        raise _Unavailable(
            "k-dimensional objectives not yet available in this build "
            "(frontier axes are paid_work_error / leisure_social_error)"
        )

    legs = _load_legs(exp["id"])
    vectors: list[dict[str, Any]] = []
    for row in legs:
        score = row.get("score")
        # interrupted-but-scored legs carry real (partial) agent-days and
        # must aggregate — otherwise real runs show an empty GUI. Only
        # unscored legs stay out (they are already in the interruptions
        # list below).
        if row.get("status") not in ("completed", "interrupted") or not isinstance(score, dict):
            continue
        per_category = score.get("per_category")
        if not isinstance(per_category, dict) or not per_category:
            continue
        label = f"{row['arch']}/{row['config_id']}"
        leg_status = row.get("status")
        score_result = SimpleNamespace(
            per_category=score.get("per_category") or {},
            frontier_valid=bool(score.get("frontier_valid")),
            fixture=bool(score.get("fixture")),
        )
        try:
            vector = frontier.assemble_error_vector(
                score_result,
                config=label,
                seed=int(row["seed"]),
                fixture=bool(score.get("fixture")),
            )
            vector["leg_status"] = leg_status
            vectors.append(vector)
        except KeyError:
            # a real leg can leave a frontier dimension unscored (paid_work
            # over zero employed agent-days scores as error=None). One
            # unscored leg must not void the results for the others — mark
            # it excluded with an explicit reason (same choice as the
            # None/NaN frontier sanitization) and keep the NaN axis out of
            # the points via average_across_seeds.
            vectors.append(
                {
                    "config": label,
                    "seed": int(row["seed"]),
                    "frontier_valid": False,
                    "fixture": bool(score.get("fixture")),
                    "leg_status": leg_status,
                    "excluded": True,
                    "reason": (
                        "unscored frontier dimensions: paid_work_error is None "
                        "(zero employed agent-days scored) — leg excluded, not fatal"
                    ),
                }
            )

    expected_seeds = sorted(int(s) for s in exp["seeds"])
    arch_ids = [a["id"] for a in exp["architectures"]]
    per_arch: dict[str, dict[str, Any]] = {}
    for arch in arch_ids:
        own = [v for v in vectors if v["config"].startswith(arch + "/")]
        per_seed: dict[str, dict[int, dict]] = {}
        for vector in own:
            per_seed.setdefault(vector["config"], {})[vector["seed"]] = vector
        configs = [
            frontier.average_across_seeds(
                config,
                seeds,
                expected_seeds=expected_seeds,
                axes=list(frontier._DEFAULT_AXES),
            )
            for config, seeds in sorted(per_seed.items())
        ]
        points = [
            (c["paid_work_error"], c["leisure_social_error"]) for c in configs if c["complete"] and c["seeds"]
        ]
        front = frontier.pareto_frontier(points)
        per_arch[arch] = {"arch": arch, "configs": configs, "frontier": [list(p) for p in front]}

    reference = exp.get("reference_architecture") or arch_ids[0]
    for arch_id in arch_ids:
        gain = (
            frontier.arch_compare(per_arch[arch_id]["frontier"], per_arch[reference]["frontier"])
            if arch_id != reference
            else []
        )
        lost = (
            frontier.arch_compare(per_arch[reference]["frontier"], per_arch[arch_id]["frontier"])
            if arch_id != reference
            else []
        )
        per_arch[arch_id]["gain"] = [list(p) for p in gain]
        per_arch[arch_id]["lost"] = [list(p) for p in lost]

    # diagnostics: noise domination per axis, tuning limits, gate honesty
    diagnostics: dict[str, Any] = {"noise_domination": {}, "tuning_limit_configs": []}
    for axis, spread_key in (
        ("paid_work_error", "paid_work_spread"),
        ("leisure_social_error", "leisure_social_spread"),
    ):
        spreads = [c[spread_key] for a in per_arch.values() for c in a["configs"] if c["complete"]]
        values = [c[axis] for a in per_arch.values() for c in a["configs"] if c["complete"]]
        if spreads and values:
            diagnostics["noise_domination"][axis] = frontier.noise_domination(
                max(spreads), max(values) - min(values)
            )
    # bound-pressing: a config pinned at a band endpoint whose tuning did not
    # beat the reference architecture at the same config point
    all_bands = _bands_by_name()
    ref_configs = {c["config"]: c for c in per_arch[reference]["configs"]}
    tuning_limits: list[dict[str, Any]] = []
    for arch_id in arch_ids:
        if arch_id == reference:
            continue
        for cfg in per_arch[arch_id]["configs"]:
            flat = next(
                (
                    r.get("config_patch_flat") or {}
                    for r in legs
                    if r.get("arch") == arch_id and r.get("config_id") == cfg["config"]
                ),
                {},
            )
            ref_cfg = ref_configs.get(cfg["config"])
            for path, value in flat.items():
                band = next((b for b in all_bands.values() if b["path"] == path), None)
                if band is None:
                    continue
                improved = bool(
                    ref_cfg and cfg["complete"] and cfg["paid_work_error"] < ref_cfg["paid_work_error"]
                )
                if frontier.tuning_limit(float(value), (band["low"], band["high"]), improved):
                    tuning_limits.append(
                        {"arch": arch_id, "config": cfg["config"], "param": path, "value": value}
                    )
    diagnostics["tuning_limit_configs"] = tuning_limits

    valid = [v for v in vectors if not v.get("excluded")]
    frontier_valid = bool(valid) and not any(v.get("excluded") for v in vectors)
    diagnostics["frontier_valid"] = frontier_valid
    diagnostics["excluded"] = [
        {"config": v["config"], "seed": v["seed"], "reason": v.get("reason", "")}
        for v in vectors
        if v.get("excluded")
    ]
    diagnostics["incomplete"] = [
        {"arch": a["arch"], "config": c["config"], "missing_seeds": c["missing_seeds"]}
        for a in per_arch.values()
        for c in a["configs"]
        if not c["complete"]
    ]
    # interruptions list: interrupted legs stay flagged here even when they
    # scored (partial agent-days), so the GUI can still mark them partial
    diagnostics["failed_legs"] = [
        {
            "arch": r.get("arch"),
            "config_id": r.get("config_id"),
            "seed": r.get("seed"),
            "status": r.get("status"),
            "stop_reason": r.get("stop_reason"),
            "scored": bool(isinstance(r.get("score"), dict) and r.get("score", {}).get("per_category")),
        }
        for r in legs
        if r.get("status") not in ("completed",)
    ]
    diagnostics["fixture"] = bool(vectors) and all(v.get("fixture") for v in vectors)

    return {
        "vectors": vectors,
        "architectures": [per_arch[a] for a in arch_ids],
        "reference_architecture": reference,
        "diagnostics": diagnostics,
    }


def results(exp_id: str) -> tuple[dict[str, Any], int]:
    exp = _load_exp(exp_id)
    if exp is None:
        return {"error": "unknown experiment"}, 404
    try:
        computed = _compute_results(exp)
    except _Unavailable as exc:
        return {"error": str(exc), "status": exp["status"], "progress": exp["progress"]}, 501
    body = {
        "id": exp_id,
        "status": exp["status"],
        "progress": exp["progress"],
        "results": computed if computed["vectors"] else None,
    }
    if computed["vectors"] is None or not computed["vectors"]:
        body["message"] = "no scored legs yet"
    return body, 200


# ---------------------------------------------------------------- report / export


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def report(exp_id: str) -> tuple[dict[str, Any], int]:
    exp = _load_exp(exp_id)
    if exp is None:
        return {"error": "unknown experiment"}, 404
    try:
        computed = _compute_results(exp)
    except _Unavailable as exc:
        return {"error": str(exc)}, 501
    lines = [
        f"# MCF 实验：{exp['name']}",
        "",
        f"- experiment id: `{exp_id}`",
        f"- 参照架构（M0）：`{exp.get('reference_architecture')}`",
        f"- 架构：{', '.join(a['id'] + ' (' + a['label'] + ')' for a in exp['architectures'])}",
        f"- 参数带：{', '.join(exp['parameters']) or '（无）'}",
        f"- 种子：{', '.join(map(str, exp['seeds']))}",
        f"- 模型：{exp.get('model') or '按配置路由'}",
        "",
    ]
    if not computed["vectors"]:
        lines.append("_还没有完成的评分腿——先启动实验并等待腿部完成。_")
    else:
        diagnostics = computed["diagnostics"]
        lines += [
            "## 逐配置结果（跨种子均值）",
            "",
            "| 架构 | 配置 | paid_work_error | leisure_social_error | 种子 | 完整 |",
            "|---|---|---|---|---|---|",
        ]
        for arch in computed["architectures"]:
            for cfg in arch["configs"]:
                lines.append(
                    f"| {arch['arch']} | {cfg['config']} | {_fmt(cfg['paid_work_error'])} | "
                    f"{_fmt(cfg['leisure_social_error'])} | {', '.join(map(str, cfg['seeds']))} | "
                    f"{'是' if cfg['complete'] else '否（缺 ' + ', '.join(map(str, cfg['missing_seeds'])) + '）'} |"
                )
        lines += ["", "## 帕累托前沿与 A(M1|M0)", ""]
        for arch in computed["architectures"]:
            lines.append(f"- `{arch['arch']}` 前沿点：{arch['frontier'] or '（无有效点）'}")
            tag = "参照架构" if arch["arch"] == computed["reference_architecture"] else "相对参照"
            lines.append(f"  - {tag}：新增可达 {arch['gain'] or '无'}；新丢失 {arch['lost'] or '无'}")
        lines += [
            "",
            "## 诊断",
            "",
            f"- frontier_valid：**{diagnostics['frontier_valid']}**"
            + ("" if diagnostics["frontier_valid"] else "（有腿被门槛排除，比较不作数）"),
            f"- 噪声支配：{diagnostics.get('noise_domination') or '（样本不足）'}",
            f"- fixture 数据：{diagnostics['fixture']}",
        ]
        if diagnostics["excluded"]:
            lines.append("- 被排除的腿：")
            lines += [f"  - {e['config']}/seed {e['seed']}：{e['reason']}" for e in diagnostics["excluded"]]
        if diagnostics["incomplete"]:
            lines += [
                f"  - {i['arch']}/{i['config']} 缺种子 {', '.join(map(str, i['missing_seeds']))}"
                for i in diagnostics["incomplete"]
            ]
        if diagnostics["failed_legs"]:
            lines.append("- 失败/中断的腿：")
            lines += [
                f"  - {f['arch']}/{f['config_id']}/seed {f['seed']}：{f['status']} — {f['stop_reason']}"
                for f in diagnostics["failed_legs"]
            ]
    lines.append("")
    return {"experiment_id": exp_id, "markdown": "\n".join(lines)}, 200


def export(exp_id: str) -> tuple[dict[str, Any], int]:
    exp = _load_exp(exp_id)
    if exp is None:
        return {"error": "unknown experiment"}, 404
    try:
        computed = _compute_results(exp)
    except _Unavailable:
        computed = {"error": "results not yet available in this build"}
    return {"experiment": exp, "legs": _load_legs(exp_id), "results": computed}, 200


def experiments() -> dict[str, Any]:
    root = _mcf_root()
    out = []
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root), reverse=True):
            exp = _load_exp(entry)
            if exp is None:
                continue
            out.append(
                {
                    "id": exp["id"],
                    "name": exp["name"],
                    "status": exp["status"],
                    "created_at": exp["created_at"],
                    "progress": exp["progress"],
                    "estimated_calls": exp.get("estimated_calls"),
                }
            )
    return {"experiments": out}


def experiment(exp_id: str) -> tuple[dict[str, Any], int]:
    exp = _load_exp(exp_id)
    if exp is None:
        return {"error": "unknown experiment"}, 404
    return exp, 200


# ---------------------------------------------------------------- entry points


def handle_get(path: str, query: dict) -> tuple[dict[str, Any], int]:
    route = path.rstrip("/")
    if route == "/api/mcf/bands":
        try:
            return bands(), 200
        except ImportError as exc:
            return {
                "error": f"{exc}",
            }, 501
    if route == "/api/mcf/experiments":
        return experiments(), 200
    if route.startswith("/api/mcf/experiment/"):
        parts = route[len("/api/mcf/experiment/") :].split("/")
        exp_id = parts[0]
        action = parts[1] if len(parts) > 1 else ""
        if not action:
            return experiment(exp_id)
        if action == "results":
            return results(exp_id)
        if action == "report":
            return report(exp_id)
        if action == "export":
            return export(exp_id)
        return {"error": "Unknown endpoint"}, 404
    return {"error": "Unknown endpoint"}, 404


def handle_post(path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    route = path.rstrip("/")
    if route == "/api/mcf/experiment":
        return create_experiment(payload)
    if route.startswith("/api/mcf/experiment/"):
        parts = route[len("/api/mcf/experiment/") :].split("/")
        if len(parts) == 2 and parts[1] == "launch":
            return launch_experiment(parts[0], payload)
        return {"error": "Unknown endpoint"}, 404
    return {"error": "Unknown endpoint"}, 404
