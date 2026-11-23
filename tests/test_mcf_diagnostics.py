"""Step 11 — flat-dimension & bound-pressing diagnostics + limitations block.

Per axis, between-seed spread is compared with the between-config range
across sampled configs: an axis whose seed noise covers its config range is
"flat" (insensitive parameter) and must be flagged. A nondominated config
whose sampled parameter sits at a band edge is bound-pressing. The
deterministic report carries a diagnostics + empirical-limitations block
(coverage gates, frontier_valid, incomplete legs, insensitive params).
"""

from __future__ import annotations

import json

from benchmark.mcf import frontier


def make_sr(paid, leisure, social):
    return type(
        "ScoreResult",
        (),
        {
            "per_category": {
                "paid_work": {"error": paid},
                "leisure": {"error": leisure},
                "socializing": {"error": social},
            },
            "frontier_valid": True,
            "fixture": True,
        },
    )()


def row(config, seed, paid, leisure, social, patch=None):
    r = {"config": config, "seed": seed, "score_result": make_sr(paid, leisure, social)}
    if patch is not None:
        r["patch"] = patch
    return r


# ---------------------------------------------------------------- flat axes


def test_flat_dimension_flagged_when_seed_spread_covers_config_range():
    # leisure axis barely moves between configs, seeds vary more than configs
    rows = [
        row("a", 7, 0.30, 0.10, 0.10),
        row("a", 11, 0.30, 0.14, 0.14),
        row("b", 7, 0.20, 0.11, 0.11),
        row("b", 11, 0.20, 0.15, 0.15),
    ]
    out = frontier.flat_dimensions(rows)
    by_axis = {d["axis"]: d for d in out}
    assert by_axis["leisure_social_error"]["flat"] is True
    assert by_axis["paid_work_error"]["flat"] is False
    assert (
        by_axis["leisure_social_error"]["config_range"] < by_axis["leisure_social_error"]["max_seed_spread"]
    )


def test_no_flat_dimension_when_configs_separate():
    rows = [
        row("a", 7, 0.30, 0.10, 0.10),
        row("a", 11, 0.30, 0.12, 0.12),
        row("b", 7, 0.10, 0.30, 0.30),
        row("b", 11, 0.10, 0.32, 0.32),
    ]
    out = frontier.flat_dimensions(rows)
    assert all(not d["flat"] for d in out)


# ----------------------------------------------------------- bound pressing


def test_bound_pressing_flags_band_edges():
    bands = {"personality.style_fit_amplitude": {"low": 0.45, "high": 0.55}}
    assert frontier.bound_pressing({"personality.style_fit_amplitude": 0.45}, bands) == [
        {"param": "personality.style_fit_amplitude", "value": 0.45, "edge": "low"}
    ]
    assert frontier.bound_pressing({"personality.style_fit_amplitude": 0.55}, bands) == [
        {"param": "personality.style_fit_amplitude", "value": 0.55, "edge": "high"}
    ]
    assert frontier.bound_pressing({"personality.style_fit_amplitude": 0.50}, bands) == []
    assert frontier.bound_pressing({"some.unknown.path": 0.45}, bands) == []


def test_report_reports_bound_pressing_for_frontier_configs():
    # the real style-fit band is [0.1, 0.4]; 0.4 presses the high edge
    patch_a = {"personality.style_fit_amplitude": 0.4}  # band edge, config a is on the frontier
    patch_b = {"personality.style_fit_amplitude": 0.25}
    rows = [
        row("a", 7, 0.10, 0.10, 0.10, patch=patch_a),
        row("a", 11, 0.10, 0.12, 0.12, patch=patch_a),
        row("b", 7, 0.30, 0.30, 0.30, patch=patch_b),
        row("b", 11, 0.30, 0.32, 0.32, patch=patch_b),
    ]
    report = frontier.render_report(rows, fixture=True)
    doc = json.loads(report["json"])
    pressing = doc["diagnostics"]["bound_pressing"]
    assert pressing and pressing[0]["config"] == "a"
    assert pressing[0]["param"] == "personality.style_fit_amplitude"
    assert pressing[0]["edge"] == "high"


# ------------------------------------------------------- limitations block


def test_report_carries_empirical_limitations_block():
    rows = [
        row("a", 7, 0.30, 0.10, 0.10),
        row("a", 11, 0.30, 0.14, 0.14),
    ]
    report = frontier.render_report(rows, fixture=True, expected_seeds=[7, 11, 13])
    doc = json.loads(report["json"])
    limits = doc["empirical_limitations"]
    assert limits["coverage_gates"]["frontier_valid"] is True
    assert limits["incomplete_configs"] == ["a"]
    assert limits["insensitive_params"]  # from flat_dimensions
    assert limits["fixture_label"] == "SYNTHETIC FIXTURE"
    assert "Empirical limitations" in report["markdown"]
    assert "insensitive" in report["markdown"]
