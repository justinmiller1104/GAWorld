"""Step 12 — one experiment manifest JSON, reopenable by the GUI.

A manifest binds everything one experiment produced: identity, architecture
definitions (M0/M1 config deltas), model/provider info, sampler method, the
full config list with config_ids, seeds, per-leg references + usage +
outcomes (failures and incomplete legs included, loudly), error vectors, the
nondominated set, A(M1|M0) results, diagnostics, and verdicts.
``load_manifest`` returns the full structure for the GUI results endpoint.
"""

from __future__ import annotations

import json

import pytest

from benchmark.mcf import manifest

EXP = {
    "id": "20260601-000000-abc123",
    "created_at": 1780000000.0,
    "name": "smoke",
    "note": "fixture",
    "architectures": [
        {"id": "m0", "label": "M0 baseline", "patch": {}},
        {"id": "m1", "label": "M1 realism", "patch": {"personality.channels.rules": True}},
    ],
    "reference_architecture": "m0",
    "sampler": "endpoints",
    "n_configs": 2,
    "n_seeds": 2,
    "seeds": [7, 11],
    "parameters": ["style_fit_amplitude"],
    "objectives": ["paid_work_error", "leisure_social_error"],
    "model": {"provider": "omlx_qwen", "temperature": 0.3},
}


def leg(arch, cid, seed, status="completed", stop_reason="", usage=None):
    return {
        "arch": arch,
        "config_id": cid,
        "config_patch_flat": {"personality.style_fit_amplitude": 0.5},
        "seed": seed,
        "status": status,
        "stop_reason": stop_reason,
        "usage": usage or {"calls": 10, "tokens_in": 5, "tokens_out": 5},
        "usage_source": "measured",
        "run_path": f"output/mcf/{EXP['id']}/{arch}/{cid}/seed{seed}",
    }


RESULTS = {
    "vectors": [{"config": "m0/cfglow", "seed": 7, "paid_work_error": 0.2, "leisure_social_error": 0.1}],
    "architectures": [
        {"arch": "m0", "configs": [], "frontier": [[0.2, 0.1]], "gain": [], "lost": []},
        {"arch": "m1", "configs": [], "frontier": [[0.1, 0.06]], "gain": [[0.1, 0.06]], "lost": []},
    ],
    "reference_architecture": "m0",
    "diagnostics": {"frontier_valid": True},
}


def test_manifest_binds_everything():
    m = manifest.build_manifest(EXP, [leg("m0", "cfglow", 7)], results=RESULTS)
    assert m["schema_version"] == 1
    assert m["experiment"]["id"] == EXP["id"] and m["experiment"]["name"] == "smoke"
    assert m["model"] == EXP["model"]
    assert m["sampler"] == "endpoints"
    assert m["seeds"] == [7, 11]
    assert [a["id"] for a in m["architectures"]] == ["m0", "m1"]
    assert m["architectures"][1]["patch"] == {"personality.channels.rules": True}
    assert m["legs"][0]["config_id"] == "cfglow"
    assert m["legs"][0]["usage"]["calls"] == 10
    assert m["legs"][0]["run_path"].endswith("/seed7")
    assert m["nondominated"]["m0"] == [[0.2, 0.1]]
    assert m["arch_compare"]["m1"]["gain"] == [[0.1, 0.06]]
    assert m["diagnostics"] == RESULTS["diagnostics"]


def test_manifest_configs_carry_config_ids():
    from benchmark.mcf.config_space import config_id

    legs = [
        leg("m0", config_id({"personality.style_fit_amplitude": 0.1}), 7),
        leg("m0", config_id({"personality.style_fit_amplitude": 0.4}), 7),
    ]
    legs[0]["config_patch_flat"] = {"personality.style_fit_amplitude": 0.1}
    legs[1]["config_patch_flat"] = {"personality.style_fit_amplitude": 0.4}
    m = manifest.build_manifest(EXP, legs, results=RESULTS)
    ids = {c["config_id"] for c in m["configs"]}
    assert len(ids) == 2
    # config ids are reproducible over the flat patch (orchestrator contract)
    assert all(c["config_id"] == config_id(c["patch"]) for c in m["configs"])


def test_manifest_verdicts_per_architecture():
    m = manifest.build_manifest(EXP, [leg("m0", "cfglow", 7)], results=RESULTS)
    assert m["verdicts"]["m1"]["gain"] == [[0.1, 0.06]]
    assert m["verdicts"]["m1"]["verdict"] == "newly-attainable-points"
    assert m["verdicts"]["m0"]["verdict"] == "reference"


def test_manifest_incomplete_leg_honesty():
    legs = [
        leg("m0", "cfglow", 7),
        leg("m0", "cfglow", 11, status="failed", stop_reason="budget exhausted"),
    ]
    m = manifest.build_manifest(EXP, legs, results=RESULTS)
    assert m["complete"] is False
    bad = [entry for entry in m["legs"] if entry["status"] != "completed"]
    assert len(bad) == 1
    assert bad[0]["stop_reason"] == "budget exhausted"
    assert m["failures"] == [
        {"arch": "m0", "config_id": "cfglow", "seed": 11, "status": "failed", "stop_reason": "budget exhausted"}
    ]


def test_manifest_without_results_is_honest():
    m = manifest.build_manifest(EXP, [leg("m0", "cfglow", 7)], results=None)
    assert m["results"] is None
    assert "no scored results" in m["verdicts"]["m1"]["verdict"]


def test_save_load_roundtrip(tmp_path):
    m = manifest.build_manifest(EXP, [leg("m0", "cfglow", 7)], results=RESULTS)
    path = manifest.save_manifest(str(tmp_path / "experiment.json"), m)
    loaded = manifest.load_manifest(path)
    assert loaded == m
    assert manifest.load_manifest(str(tmp_path / "missing.json")) is None


def test_manifest_build_is_deterministic():
    legs = [leg("m0", "cfglow", 7), leg("m1", "cfghigh", 11)]
    a = json.dumps(manifest.build_manifest(EXP, legs, results=RESULTS), sort_keys=True)
    b = json.dumps(manifest.build_manifest(EXP, legs, results=RESULTS), sort_keys=True)
    assert a == b
