"""M5 offline frontier module — time-use error frontier across configs.

Assembles 2-D error vectors (paid work, leisure/social) from ScoreResults,
averages them across seeds, extracts the Pareto (componentwise-minimization)
frontier, and computes A(M1|M0) — the error points newly attainable under one
architecture but not another (docs/MCF_GAWORLD.md §4). Reports are
deterministic: identical inputs yield byte-identical JSON and Markdown (§8).

Validity gates from time_use (MIN_COMPARISON_DAYS / MIN_COMPARISON_POPULATION,
``ScoreResult.frontier_valid``) are enforced upstream of the frontier: gated-out
legs are excluded *with a reason*, never silently averaged in.

Everything here runs offline against synthetic fixtures — reports produced with
``fixture=True`` are labeled as such and must never be read as empirical results.
"""

from __future__ import annotations

import json
import math

from benchmark.mcf.time_use import ScoreResult

__all__ = [
    "assemble_error_vector",
    "average_across_seeds",
    "pareto_frontier",
    "arch_compare",
    "tuning_limit",
    "noise_domination",
    "render_report",
    "register_axis",
    "axes_from_names",
    "paired_arch_stats",
    "flat_dimensions",
    "bound_pressing",
]

# Axes registry (plan step 9): objective name → extractor over the
# ``per_category`` dict of a ScoreResult. The time-use 2-D default is kept;
# ≥3-objective studies register or pass their own axes explicitly.
AXES: dict[str, object] = {
    "paid_work_error": lambda pc: float(pc["paid_work"]["error"]),
    "leisure_social_error": lambda pc: (
        float(pc["leisure"]["error"]) + float(pc["socializing"]["error"])
    ) / 2.0,
}
_DEFAULT_AXES = tuple(AXES)


def register_axis(name: str, extractor) -> None:
    """Add an objective to the global registry (last-writer-wins)."""
    AXES[name] = extractor


def axes_from_names(names) -> dict:
    """Ordered axis mapping for the named objectives; unknown names raise."""
    unknown = [n for n in names if n not in AXES]
    if unknown:
        raise KeyError(f"unregistered frontier axes: {unknown}")
    return {n: AXES[n] for n in names}

_DIMENSIONS = ("paid_work_error", "leisure_social_error")


def assemble_error_vector(
    score_result: ScoreResult, *, config: str, seed: int, fixture: bool, axes: dict | None = None
) -> dict:
    """Extract the frontier dimensions from one ScoreResult.

    ``axes`` maps objective name → extractor over ``per_category``; it
    defaults to the registered time-use 2-D axes. A gate-failed leg
    (``frontier_valid=False``) is excluded with an explicit reason; a missing
    dimension raises.
    """
    axes = axes if axes is not None else {n: AXES[n] for n in _DEFAULT_AXES}
    per_category = score_result.per_category

    values = {}
    missing = []
    for name, extractor in axes.items():
        try:
            values[name] = float(extractor(per_category))
        except (KeyError, TypeError):
            missing.append(name)
    if missing:
        raise KeyError(f"missing frontier dimensions for {config}/seed {seed}: {missing}")

    vector = {
        "config": config,
        "seed": seed,
        **values,
        "frontier_valid": bool(score_result.frontier_valid),
        "fixture": bool(fixture),
    }
    if not score_result.frontier_valid:
        vector["excluded"] = True
        vector["reason"] = (
            "frontier_valid=False: comparison below the "
            f"MIN_COMPARISON_DAYS/MIN_COMPARISON_POPULATION gates ({config}/seed {seed})"
        )
    else:
        vector["excluded"] = False
    return vector


def _axis_names(vectors: dict) -> list[str]:
    """Axis keys (= ``*_error`` values) present in the vectors, registry order first."""
    sample = next(iter(vectors.values()), {})
    registry_order = [n for n in AXES if n in sample]
    extra = sorted(k for k in sample if k.endswith("_error") and k not in registry_order)
    return registry_order + extra


def average_across_seeds(
    config: str, per_seed_vectors: dict, expected_seeds: list | None = None, axes: list | None = None
) -> dict:
    """Mean vector (+ half-range spread) over matched seeds for one config.

    A seed named in ``expected_seeds`` but absent makes the config incomplete —
    reported loudly, never silently treated as a full set. ``axes`` selects the
    objective keys to aggregate; it defaults to whatever ``*_error`` keys the
    vectors carry (registry order first), so k-dimensional vectors work.
    """
    seeds = sorted(per_seed_vectors)
    missing = sorted(s for s in (expected_seeds or []) if s not in per_seed_vectors)
    complete = not missing
    names = axes if axes is not None else _axis_names(per_seed_vectors)

    out = {"config": config}
    for axis in names:
        values = [float(v[axis]) for v in per_seed_vectors.values() if axis in v]
        out[axis] = sum(values) / len(values) if values else float("nan")
        spread_key = axis.replace("_error", "") + "_spread"
        out[spread_key] = (max(values) - min(values)) / 2.0 if len(values) > 1 else 0.0
    out.update(
        {
            "seeds": seeds,
            "complete": complete,
            "missing_seeds": missing,
        }
    )
    if not complete:
        out["reason"] = f"{config} is incomplete: missing seeds {missing}"
        out["warning"] = out["reason"]
    return out


def _dominates(a: tuple, b: tuple) -> bool:
    """True if ``a`` is strictly better than ``b`` on every axis.

    Weak domination (strict on *all* components): points tied on one axis
    neither dominate nor exclude each other, so ties both survive on the front.
    """
    return all(x < y for x, y in zip(a, b))


def pareto_frontier(points) -> list[tuple]:
    """Nondominated set under componentwise minimization, deterministic order."""
    unique = sorted({tuple(p) for p in points})
    front = [
        p for p in unique if not any(q != p and _dominates(q, p) for q in unique)
    ]
    return sorted(front)


def arch_compare(frontier_m1, frontier_m0) -> list:
    """A(M1|M0): frontier points attainable under M1 but dominated under M0.

    An explicit empty list means the architectures are equally capable or M0
    dominates — absence of gain is a reported result, not a missing one.
    """
    # §4 / paper Eq. 2: a point is excluded when an M0 frontier point WEAKLY
    # dominates it (better-or-equal on every axis, at least one strict) —
    # strict-on-all-axes (_dominates) would let ties on one axis through.
    def _weakly_dominates(q: tuple, p: tuple) -> bool:
        return q != p and all(x <= y for x, y in zip(q, p))

    m0 = [tuple(p) for p in frontier_m0]
    gain = [
        tuple(p)
        for p in frontier_m1
        if tuple(p) not in m0 and not any(_weakly_dominates(q, tuple(p)) for q in m0)
    ]
    return sorted(gain)


def paired_arch_stats(
    arch: str,
    arch_vectors: dict,
    reference: str,
    reference_vectors: dict,
    axes: list | None = None,
) -> dict:
    """Matched-config paired statistics of ``arch`` against ``reference``.

    Inputs map seed → assembled error vector at one matched config. Per axis
    (never collapsed): per-seed paired differences (arch − reference),
    improvement/worsening/tie counts, standardized mean difference dz
    (mean(d)/sd(d), ddof=1; 0.0 when sd is 0), and an exact two-sided sign
    test over the non-tied differences. Seeds present in only one architecture
    are reported under ``unmatched`` and excluded from the pairing.
    """
    import math

    axes = axes if axes is not None else [n for n in AXES if n in next(iter(arch_vectors.values()), {})]
    matched = sorted(set(arch_vectors) & set(reference_vectors))
    unmatched = {
        arch: sorted(set(arch_vectors) - set(reference_vectors)),
        reference: sorted(set(reference_vectors) - set(arch_vectors)),
    }
    out: dict = {"arch": arch, "reference": reference, "matched_seeds": matched, "axes": {}, "unmatched": unmatched, "complete": not any(unmatched.values())}
    for axis in axes:
        diffs = [
            float(arch_vectors[s][axis]) - float(reference_vectors[s][axis])
            for s in matched
            if axis in arch_vectors[s] and axis in reference_vectors[s]
        ]
        improvements = sum(d < 0 for d in diffs)
        worsenings = sum(d > 0 for d in diffs)
        ties = sum(d == 0 for d in diffs)
        n_eff = len(diffs) - ties
        mean = sum(diffs) / len(diffs) if diffs else 0.0
        var = (
            sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
            if len(diffs) > 1
            else 0.0
        )
        sd = math.sqrt(var)
        # dz is oriented toward improvement: (reference − arch)/sd, so a
        # positive dz means the architecture reduced the error vs the reference
        dz = (-mean / sd) if sd > 0 else 0.0
        # exact two-sided sign test over the non-tied differences
        if n_eff > 0:
            k = min(improvements, worsenings)
            tail = sum(math.comb(n_eff, i) for i in range(0, k + 1)) / 2**n_eff
            p = min(1.0, 2 * tail)
        else:
            p = 1.0
        verdict = "no-difference" if n_eff == 0 or improvements == worsenings else (
            "improved" if improvements > worsenings else "worsened"
        )
        out["axes"][axis] = {
            "n": len(diffs),
            "matched_seeds": matched,
            "diffs": diffs,
            "improvements": improvements,
            "worsenings": worsenings,
            "ties": ties,
            "mean_diff": mean,
            "dz": dz,
            "sign_test_p": p,
            "verdict": verdict,
        }
    return out


def tuning_limit(config_value: float, band: tuple, improved: bool) -> bool:
    """True iff the config sits at a band endpoint and tuning did not improve it."""
    lo, hi = band
    at_edge = math.isclose(config_value, lo) or math.isclose(config_value, hi)
    return at_edge and not improved


def noise_domination(seed_spread: float, config_range: float) -> bool:
    """True iff between-seed spread >= between-config range (inclusive boundary).

    Then the between-config effect is not distinguishable from seed noise.
    """
    return seed_spread >= config_range


# ---------------------------------------------------------------------------
# Deterministic report (JSON + Markdown)
# ---------------------------------------------------------------------------


def _ascii_scatter(points) -> str:
    """A crude fixed-grid ASCII scatter of (x, y) points — no plotting libs."""
    if not points:
        return "(no frontier points to plot — all legs excluded or incomplete)"
    width, height = 40, 12
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or 1.0

    def col(p):
        return round((p[0] - x_lo) / x_span * (width - 1)) if width > 1 else 0

    def row(p):
        return round((p[1] - y_lo) / y_span * (height - 1)) if height > 1 else 0

    grid = [[" "] * width for _ in range(height)]
    for p in points:
        grid[height - 1 - row(p)][col(p)] = "*"
    top = f"+{'-' * width}+  x = paid_work_error [{x_lo:g}..{x_hi:g}], y = leisure_social_error [{y_lo:g}..{y_hi:g}]"
    body = ["|" + "".join(r) + "|  y-axis: leisure_social_error (lower is better)" for r in grid]
    bottom = "+" + "-" * width + "+  x-axis: paid_work_error (lower is better)"
    return "\n".join([top, *body, bottom])


def flat_dimensions(rows, axes: list | None = None) -> list[dict]:
    """Per-axis insensitive-parameter flags over the sampled configs.

    ``rows`` are assembled per-seed rows (``config`` + ``score_result``, as
    taken by ``render_report``). For each axis: the largest within-config seed
    spread vs the between-config range of the seed-averaged values — an axis
    is flat when seed noise covers the config range (``noise_domination``),
    i.e. tuning cannot move it.
    """
    per_config: dict[str, dict[int, dict]] = {}
    for r in rows:
        vector = (
            assemble_error_vector(
                r["score_result"], config=str(r["config"]), seed=int(r["seed"]), fixture=bool(r.get("fixture", True))
            )
            if "score_result" in r
            else r
        )
        if not vector.get("excluded"):
            per_config.setdefault(str(r["config"]), {})[int(r["seed"])] = vector
    names = []
    if axes is not None:
        names = axes
    elif per_config:
        first_config_vectors = next(iter(per_config.values()))
        first_vector = next(iter(first_config_vectors.values()), {})
        names = _axis_names({"v": first_vector})
    out = []
    for axis in names:
        spreads = []
        avgs = []
        for seeds in per_config.values():
            values = [float(v[axis]) for v in seeds.values() if axis in v]
            if values:
                spreads.append(max(values) - min(values))
                avgs.append(sum(values) / len(values))
        max_spread = max(spreads) if spreads else 0.0
        config_range = (max(avgs) - min(avgs)) if len(avgs) > 1 else 0.0
        out.append(
            {
                "axis": axis,
                "max_seed_spread": max_spread,
                "config_range": config_range,
                "flat": noise_domination(max_spread, config_range),
            }
        )
    return out


def bound_pressing(patch: dict, bands_by_path: dict) -> list[dict]:
    """Band-edge hits in a flat config patch ({dotted.path: value}).

    Unknown paths are ignored (architecture-level patches legitimately carry
    non-band keys). Returns one entry per parameter sitting on an edge.
    """
    hits = []
    for path, value in patch.items():
        band = bands_by_path.get(path)
        if band is None:
            continue
        if math.isclose(float(value), band["low"]):
            hits.append({"param": path, "value": float(value), "edge": "low"})
        elif math.isclose(float(value), band["high"]):
            hits.append({"param": path, "value": float(value), "edge": "high"})
    return sorted(hits, key=lambda h: h["param"])


def render_report(rows, fixture: bool = True, expected_seeds: list | None = None) -> dict:
    """Deterministic JSON + Markdown report over assembled rows.

    Each row: ``{"config", "seed", "score_result", ["n_agents"], ["n_days"]}``.
    Byte-identical outputs for identical inputs (doc §8); ``fixture=True``
    labels the data synthetic in both outputs.
    """
    vectors = [
        assemble_error_vector(
            row["score_result"],
            config=str(row["config"]),
            seed=int(row["seed"]),
            fixture=fixture,
        )
        for row in rows
    ]
    # aggregate over whatever axes the vectors carry (2-D time-use default)
    axis_names = _axis_names({"v": vectors[0]}) if vectors else list(_DEFAULT_AXES)
    seeds_seen = sorted({int(r["seed"]) for r in rows})
    want_seeds = sorted(expected_seeds) if expected_seeds is not None else seeds_seen

    per_config: dict[str, dict] = {}
    for vector in sorted(vectors, key=lambda v: (v["config"], v["seed"])):
        per_config.setdefault(vector["config"], {})[vector["seed"]] = vector

    configs_out: dict[str, dict] = {}
    for config in sorted(per_config):
        legs = {
            seed: v for seed, v in per_config[config].items() if not v.get("excluded")
        }
        avg = average_across_seeds(config, legs, expected_seeds=want_seeds, axes=axis_names)
        if not legs and expected_seeds is None:
            avg["complete"] = True  # single gated-out leg: incompleteness is not the issue
        configs_out[config] = avg

    valid_vectors = [v for v in vectors if not v.get("excluded")]
    frontier_valid = bool(valid_vectors) and not any(v.get("excluded") for v in vectors)
    gate_reason = (
        "all legs frontier-valid"
        if frontier_valid
        else (
            "excluded legs: "
            + "; ".join(v["reason"] for v in vectors if v.get("excluded"))
            if any(v.get("excluded") for v in vectors)
            else "no valid legs"
        )
    )

    points = [
        (avg["paid_work_error"], avg["leisure_social_error"])
        for avg in configs_out.values()
        if not avg["paid_work_error"] != avg["paid_work_error"]  # drop NaN means
    ]
    front = pareto_frontier(points)
    sorted_configs = sorted(configs_out, key=lambda c: configs_out[c]["paid_work_error"])
    rest = [c for c in sorted_configs if c not in front]
    m0_points = [
        (configs_out[c]["paid_work_error"], configs_out[c]["leisure_social_error"])
        for c in sorted_configs
        if (configs_out[c]["paid_work_error"], configs_out[c]["leisure_social_error"]) not in front
    ] or points
    arch_gain = arch_compare(front, pareto_frontier(m0_points)) if front else []

    spreads_pw = [a["paid_work_spread"] for a in configs_out.values()]
    ranges_pw = (
        max(a["paid_work_error"] for a in configs_out.values())
        - min(a["paid_work_error"] for a in configs_out.values())
        if len(configs_out) > 1
        else 0.0
    )
    noisy = bool(spreads_pw) and noise_domination(max(spreads_pw), ranges_pw)
    diagnostics = {
        "noise_domination": {
            "flagged": noisy,
            "max_seed_spread": max(spreads_pw) if spreads_pw else None,
            "config_range": ranges_pw,
            "note": (
                "between-seed spread >= between-config range: effect indistinguishable "
                "from seed noise" if noisy else "config differences exceed seed spread"
            ),
        },
        "tuning_limit": [],
    }

    # per-axis insensitive-parameter flags + band-edge pressing (step 11)
    flat_axes = flat_dimensions(rows)
    bands_by_path = {}
    try:
        from benchmark.mcf import config_space as _cs

        bands_by_path = {b["path"]: b for b in _cs.BANDS.values()}
    except ImportError:
        pass
    patches_by_config: dict[str, dict] = {}
    for r in rows:
        patch = r.get("patch")
        if patch and str(r["config"]) not in patches_by_config:
            patches_by_config[str(r["config"])] = patch
    front_points_set = {tuple(p) for p in front}
    bound_hits = []
    for config, avg in configs_out.items():
        point = (avg["paid_work_error"], avg["leisure_social_error"])
        if point not in front_points_set:
            continue
        for hit in bound_pressing(patches_by_config.get(config, {}), bands_by_path):
            bound_hits.append({"config": config, **hit})
    diagnostics["flat_dimensions"] = flat_axes
    diagnostics["bound_pressing"] = bound_hits

    insensitive = [d["axis"] for d in flat_axes if d["flat"]]
    empirical_limitations = {
        "coverage_gates": {
            "frontier_valid": frontier_valid,
            "reason": gate_reason,
            "MIN_COMPARISON_DAYS": 7,
            "MIN_COMPARISON_POPULATION": 20,
        },
        "incomplete_configs": sorted(c for c in configs_out if not configs_out[c]["complete"]),
        "insensitive_params": insensitive,
        "fixture_label": "SYNTHETIC FIXTURE" if fixture else "empirical",
        "note": (
            "Frontier comparisons are only as good as leg coverage: gated-out legs "
            "and incomplete configs shrink the evidence; insensitive parameters "
            "cannot be tuned within their bands; band-edge configs may only look "
            "nondominated because the sampler cannot go further."
        ),
    }

    incomplete = [c for c, a in configs_out.items() if not a["complete"]]
    doc = {
        "configs": configs_out,
        "seeds": want_seeds,
        "frontier": [list(p) for p in front],
        "arch_gain": [list(p) for p in arch_gain],
        "diagnostics": diagnostics,
        "gates": {
            "frontier_valid": frontier_valid,
            "reason": gate_reason,
            "MIN_COMPARISON_DAYS": 7,
            "MIN_COMPARISON_POPULATION": 20,
        },
        "fixture": bool(fixture),
        "incomplete_configs": incomplete,
        "empirical_limitations": empirical_limitations,
    }
    json_text = json.dumps(doc, indent=2, sort_keys=True) + "\n"

    lines = [
        "# Time-use error frontier report",
        "",
        "**Data label: SYNTHETIC FIXTURE** — these numbers are synthetic test"
        if fixture
        else "**Data label: empirical** (not fixture-labeled)",
        "",
        f"Gates verdict: frontier_valid: {frontier_valid} — {gate_reason}",
        "",
    ]
    if incomplete:
        lines.append(f"INCOMPLETE configs (missing legs, loud): {', '.join(sorted(incomplete))}")
        lines.append("")
    lines.append("## Per-config seed-averaged error vectors")
    lines.append("")
    lines.append("| config | seeds | paid_work_error | leisure_social_error | pw_spread | complete |")
    lines.append("|---|---|---|---|---|---|")
    for config in sorted(configs_out):
        a = configs_out[config]
        lines.append(
            f"| {config} | {a['seeds']} | {a['paid_work_error']:.4f} | "
            f"{a['leisure_social_error']:.4f} | {a['paid_work_spread']:.4f} | {a['complete']} |"
        )
    lines += ["", "## Pareto frontier (componentwise minimization)", ""]
    lines += [f"- {p}" for p in front] or ["- (empty)"]
    lines += ["", "## A(M1|M0) newly-attainable points", ""]
    lines += [f"- {p}" for p in arch_gain] or ["- (empty set: no newly-attainable points)"]
    lines += ["", "## ASCII frontier visualization", "", "```", _ascii_scatter(front), "```", ""]
    lines += [
        "## Diagnostics",
        "",
        f"- noise_domination flagged: {noisy}",
        f"- incomplete configs: {incomplete or 'none'}",
        f"- insensitive parameters (flat axes): {insensitive or 'none'}",
        f"- bound-pressing frontier configs: {bound_hits or 'none'}",
        "",
        "## Empirical limitations",
        "",
        f"- coverage gates: frontier_valid: {frontier_valid} — {gate_reason}",
        f"- MIN_COMPARISON_DAYS=7, MIN_COMPARISON_POPULATION=20",
        f"- incomplete configs (missing legs, loud): {incomplete or 'none'}",
        f"- insensitive parameters (seed spread covers config range, tuning cannot "
        f"move them): {insensitive or 'none'}",
        f"- data label: {empirical_limitations['fixture_label']}",
        "",
    ]
    md_text = "\n".join(lines)
    return {"json": json_text, "markdown": md_text}
