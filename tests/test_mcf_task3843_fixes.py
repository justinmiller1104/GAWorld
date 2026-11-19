"""Task 3843 — RED tests for two demonstrated defects.

1. ``mcf_api._compute_results`` must aggregate interrupted-but-scored legs
   (kept flagged in the interruptions list, per-leg status preserved).
2. The experiment manifest must carry reproducibility fields:
   ``employed_agent_ids``, ``tick_minutes``, ``code_commit``,
   ``targets_provenance`` (per-leg usage records already exist).
"""

from __future__ import annotations

import json
import os

from benchmark.mcf import manifest
from gaworld.apps import mcf_api

SPEC = {
    "name": "smoke",
    "architectures": [
        {"id": "m0", "label": "M0 baseline", "patch": {}},
        {"id": "m1", "label": "M1 realism", "patch": {"personality.channels.rules": True}},
    ],
    "parameters": ["style_fit_amplitude"],
    "n_configs": 2,
    "sampler": "endpoints",
    "n_seeds": 2,
    "seeds": [7, 11],
    "budget": {"max_calls": 100, "max_tokens": 1000, "max_elapsed_seconds": 60},
}


def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(mcf_api, "_mcf_root", lambda: str(tmp_path / "mcf"))
    monkeypatch.setattr(mcf_api, "_start_job", lambda exp_id: None)
    return tmp_path


def _score(paid, leisure, social):
    return {
        "per_category": {
            "paid_work": {"error": paid},
            "leisure": {"error": leisure},
            "socializing": {"error": social},
        },
        "frontier_valid": True,
        "fixture": True,
    }


def _write_legs(root, exp_id, rows):
    path = os.path.join(mcf_api._exp_dir(str(root / "mcf"), exp_id), "legs.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def _leg(arch, cid, seed, status, score=None, stop_reason=""):
    return {
        "arch": arch,
        "config_id": cid,
        "config_patch_flat": {"personality.style_fit_amplitude": 0.5},
        "seed": seed,
        "status": status,
        "stop_reason": stop_reason,
        "usage": {"calls": 10, "tokens_in": 5, "tokens_out": 5},
        "usage_source": "measured",
        "run_path": f"output/mcf/exp/{arch}/{cid}/seed{seed}",
        "score": score,
    }


# ------------------------------------------------------- defect 1


def test_interrupted_scored_legs_aggregate(tmp_path, monkeypatch):
    root = _env(tmp_path, monkeypatch)
    body, status = mcf_api.handle_post("/api/mcf/experiment", dict(SPEC))
    assert status == 201
    exp_id = body["id"]

    legs = [
        _leg("m0", "cfglow", 7, "interrupted", score=_score(0.2, 0.1, 0.05), stop_reason="budget"),
        _leg("m0", "cfglow", 11, "completed", score=_score(0.22, 0.12, 0.06)),
        _leg("m0", "cfghigh", 7, "interrupted", score=None, stop_reason="killed"),
        _leg("m1", "cfglow", 7, "completed", score=_score(0.15, 0.14, 0.04)),
    ]
    _write_legs(root, exp_id, legs)

    results = mcf_api._compute_results({"id": exp_id, **SPEC})

    # interrupted-but-scored legs produce vectors with their status preserved
    vectors = {(v["config"], v["seed"]): v for v in results["vectors"]}
    assert ("m0/cfglow", 7) in vectors, "interrupted-but-scored leg must aggregate"
    assert vectors[("m0/cfglow", 7)]["leg_status"] == "interrupted"
    assert vectors[("m0/cfglow", 11)]["leg_status"] == "completed"
    # the unscored interrupted leg stays excluded
    assert ("m0/cfghigh", 7) not in vectors

    # the interruptions list still flags the interrupted legs (both of them)
    failed = {(f["arch"], f["config_id"], f["seed"]) for f in results["diagnostics"]["failed_legs"]}
    assert ("m0", "cfglow", 7) in failed
    assert ("m0", "cfghigh", 7) in failed


def test_interrupted_scored_legs_reach_gui_results(tmp_path, monkeypatch):
    root = _env(tmp_path, monkeypatch)
    body, status = mcf_api.handle_post("/api/mcf/experiment", dict(SPEC))
    assert status == 201
    exp_id = body["id"]
    _write_legs(
        root,
        exp_id,
        [
            _leg("m0", "cfglow", 7, "interrupted", score=_score(0.2, 0.1, 0.05)),
            _leg("m1", "cfglow", 7, "interrupted", score=_score(0.15, 0.14, 0.04)),
        ],
    )
    payload, code = mcf_api.results(exp_id)
    assert code == 200
    assert payload["results"] is not None, "real (interrupted+scored) legs must not show an empty GUI"


# ------------------------------------------------------- defect 2


EXP = {
    "id": "20260601-000000-abc123",
    "created_at": 1780000000.0,
    "name": "smoke",
    "note": "fixture",
    "architectures": [
        {"id": "m0", "label": "M0 baseline", "patch": {}},
        {"id": "m1", "label": "M1 realism", "patch": {}},
    ],
    "reference_architecture": "m0",
    "sampler": "endpoints",
    "n_configs": 1,
    "n_seeds": 1,
    "seeds": [7],
    "parameters": [],
    "objectives": ["paid_work_error", "leisure_social_error"],
    "model": {},
    "employed_agent_ids": [3, 9],
    "tick_minutes": 10,
}


def test_manifest_round_trip_reproducibility_fields(tmp_path):
    m = manifest.build_manifest(EXP, [_leg("m0", "cfglow", 7, "completed", score=_score(0.2, 0.1, 0.05))])
    path = str(tmp_path / "manifest.json")
    manifest.save_manifest(path, m)
    loaded = manifest.load_manifest(path)

    assert loaded["employed_agent_ids"] == [3, 9]
    assert loaded["tick_minutes"] == 10
    # best-effort git commit: a real hash here, a clear placeholder offline
    assert isinstance(loaded["code_commit"], str) and loaded["code_commit"]
    provenance = loaded["targets_provenance"]
    assert "TARGETS_2024" in json.dumps(provenance)
    assert loaded["legs"][0]["usage"]["calls"] == 10


def test_manifest_defaults_when_spec_omits_fields():
    m = manifest.build_manifest(
        {k: v for k, v in EXP.items() if k not in ("employed_agent_ids", "tick_minutes")}, []
    )
    assert m["employed_agent_ids"] == []
    assert m["tick_minutes"] == 10
    assert "TARGETS_2024" in json.dumps(m["targets_provenance"])
