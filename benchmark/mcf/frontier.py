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
]

_DIMENSIONS = ("paid_work_error", "leisure_social_error")


def assemble_error_vector(
    score_result: ScoreResult, *, config: str, seed: int, fixture: bool
) -> dict:
    """Extract the two validated frontier dimensions from one ScoreResult.

    ``leisure_social_error`` is the mean of the leisure and socializing
    category errors. A gate-failed leg (``frontier_valid=False``) is excluded
    with an explicit reason; a missing dimension raises.
    """
    per_category = score_result.per_category
    paid_work = per_category.get("paid_work")
    leisure = per_category.get("leisure")
    socializing = per_category.get("socializing")
    if paid_work is None or leisure is None or socializing is None:
        missing = [
            name
            for name, spec in (
                ("paid_work", paid_work),
                ("leisure", leisure),
                ("socializing", socializing),
            )
            if spec is None
        ]
        raise KeyError(f"missing frontier dimensions for {config}/seed {seed}: {missing}")

    vector = {
        "config": config,
        "seed": seed,
        "paid_work_error": float(paid_work["error"]),
        "leisure_social_error": (float(leisure["error"]) + float(socializing["error"])) / 2.0,
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


def average_across_seeds(
    config: str, per_seed_vectors: dict, expected_seeds: list | None = None
) -> dict:
    """Mean vector (+ half-range spread) over matched seeds for one config.

    A seed named in ``expected_seeds`` but absent makes the config incomplete —
    reported loudly, never silently treated as a full set.
    """
    seeds = sorted(per_seed_vectors)
    missing = sorted(s for s in (expected_seeds or []) if s not in per_seed_vectors)
    complete = not missing

    def mean(axis: str) -> float:
        values = [float(v[axis]) for v in per_seed_vectors.values()]
        return sum(values) / len(values) if values else float("nan")

    def spread(axis: str) -> float:
        values = [float(v[axis]) for v in per_seed_vectors.values()]
        return (max(values) - min(values)) / 2.0 if len(values) > 1 else 0.0

    out = {
        "config": config,
        "paid_work_error": mean("paid_work_error"),
        "leisure_social_error": mean("leisure_social_error"),
        "paid_work_spread": spread("paid_work_error"),
        "leisure_social_spread": spread("leisure_social_error"),
        "seeds": seeds,
        "complete": complete,
        "missing_seeds": missing,
    }
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
        avg = average_across_seeds(config, legs, expected_seeds=want_seeds)
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
        "",
    ]
    md_text = "\n".join(lines)
    return {"json": json_text, "markdown": md_text}
