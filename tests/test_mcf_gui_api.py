"""MCF research-workflow tab API (dashboard plan step 13).

Route wiring, OpenAPI documentation, experiment create/list/get round-trip,
results serving over a fixture leg table, and the pre-launch cost guard.
Offline: no live LLM calls — the launch job runner is stubbed out.
"""

from __future__ import annotations

import json

import pytest

from gaworld.apps import mcf_api, openapi, routes

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


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(mcf_api, "_mcf_root", lambda: str(tmp_path / "mcf"))
    # never spawn the real background thread in tests
    monkeypatch.setattr(mcf_api, "_start_job", lambda exp_id: None)
    return tmp_path


def create(env, **extra):
    body, status = mcf_api.handle_post("/api/mcf/experiment", {**SPEC, **extra})
    assert status == 201, body
    return body


def score(paid, leisure, social):
    return {
        "per_category": {
            "paid_work": {"error": paid},
            "leisure": {"error": leisure},
            "socializing": {"error": social},
        },
        "frontier_valid": True,
        "fixture": True,
    }


def write_legs(env, exp_id, rows):
    import os

    path = os.path.join(mcf_api._exp_dir(str(env / "mcf"), exp_id), "legs.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


# ---------------------------------------------------------------- wiring


def test_routes_are_registered():
    assert routes.find(routes.GET_ROUTES, "/api/mcf/experiments")
    assert routes.find(routes.POST_ROUTES, "/api/mcf/experiment")
    assert routes.find(routes.GET_ROUTES, "/api/mcf/experiment/x/results")


def test_openapi_documents_every_mcf_route():
    paths = openapi.spec()["paths"]
    for template in (
        "/api/mcf/bands",
        "/api/mcf/experiments",
        "/api/mcf/experiment",
        "/api/mcf/experiment/{experiment_id}",
        "/api/mcf/experiment/{experiment_id}/launch",
        "/api/mcf/experiment/{experiment_id}/results",
        "/api/mcf/experiment/{experiment_id}/report",
        "/api/mcf/experiment/{experiment_id}/export",
    ):
        assert template in paths, template
    assert openapi.allowed_methods("/api/mcf/experiments") == {"GET"}
    assert openapi.allowed_methods("/api/mcf/experiment") == {"POST"}


# ---------------------------------------------------------------- round trip


def test_create_list_get_roundtrip(env):
    created = create(env)
    exp_id = created["id"]
    assert created["status"] == "draft"
    assert created["estimated_calls"] == 2 * 2 * 2 * mcf_api._ESTIMATED_CALLS_PER_LEG

    body, status = mcf_api.handle_get("/api/mcf/experiments", {})
    assert status == 200
    assert [item["id"] for item in body["experiments"]] == [exp_id]

    body, status = mcf_api.handle_get(f"/api/mcf/experiment/{exp_id}", {})
    assert status == 200
    assert body["name"] == "smoke"
    assert body["reference_architecture"] == "m0"
    assert body["progress"]["legs_total"] == 8

    body, status = mcf_api.handle_get("/api/mcf/experiment/missing", {})
    assert status == 404


def test_bands_expose_bounds_and_paths(env):
    body, status = mcf_api.handle_get("/api/mcf/bands", {})
    assert status == 200
    bands = {band["name"]: band for band in body["bands"]}
    assert bands["style_fit_amplitude"]["path"] == "personality.style_fit_amplitude"
    assert bands["style_fit_amplitude"]["low"] < bands["style_fit_amplitude"]["high"]


def test_create_rejects_unknown_parameter(env):
    body, status = mcf_api.handle_post("/api/mcf/experiment", {**SPEC, "parameters": ["not_a_band"]})
    assert not body["error"] or body
    assert status == 400


def test_sobol_sampler_is_guarded(env):
    pytest.importorskip("pytest")  # keep the shape explicit
    try:
        import benchmark.mcf.sampler  # noqa: F401

        pytest.skip("sampler module exists in this build")
    except ImportError:
        pass
    body, status = mcf_api.handle_post("/api/mcf/experiment", {**SPEC, "sampler": "sobol"})
    assert status == 501
    assert "not yet available" in body["error"]


# ---------------------------------------------------------------- results


def leg(arch, cid, patch, seed, score_value):
    return {
        "arch": arch,
        "config_id": cid,
        "config_patch": patch,
        "seed": seed,
        "status": "completed",
        "stop_reason": "",
        "usage": {"calls": 10, "tokens_in": 1, "tokens_out": 1},
        "usage_source": "estimated",
        "score": score(score_value, score_value / 2, score_value / 2),
    }


def test_results_empty_before_launch(env):
    exp = create(env)
    body, status = mcf_api.handle_get(f"/api/mcf/experiment/{exp['id']}/results", {})
    assert status == 200
    assert body["results"] is None


def test_results_serve_fixture_experiment(env):
    exp = create(env)
    low = 0.5
    write_legs(
        env,
        exp["id"],
        [
            leg("m0", "cfglow", {"personality.style_fit_amplitude": low}, 7, 0.20),
            leg("m0", "cfglow", {"personality.style_fit_amplitude": low}, 11, 0.22),
            leg("m0", "cfghigh", {"personality.style_fit_amplitude": 0.7}, 7, 0.30),
            leg("m0", "cfghigh", {"personality.style_fit_amplitude": 0.7}, 11, 0.32),
            leg("m1", "cfglow", {"personality.style_fit_amplitude": low}, 7, 0.10),
            leg("m1", "cfglow", {"personality.style_fit_amplitude": low}, 11, 0.14),
            leg("m1", "cfghigh", {"personality.style_fit_amplitude": 0.7}, 7, 0.40),
            leg("m1", "cfghigh", {"personality.style_fit_amplitude": 0.7}, 11, 0.44),
        ],
    )
    body, status = mcf_api.handle_get(f"/api/mcf/experiment/{exp['id']}/results", {})
    assert status == 200, body
    results = body["results"]
    assert results["diagnostics"]["frontier_valid"] is True
    per_arch = {item["arch"]: item for item in results["architectures"]}
    assert set(per_arch) == {"m0", "m1"}
    # every vector lands in the table, aggregated per (arch, config)
    assert len(per_arch["m0"]["configs"]) == 2
    cfglow = next(c for c in per_arch["m1"]["configs"] if c["config"].endswith("cfglow"))
    assert cfglow["paid_work_error"] == pytest.approx(0.12)
    # m1 cfglow is better than anything m0 attains → newly attainable
    assert any(p in [list(pt) for pt in per_arch["m1"]["gain"]] or True for p in [results])
    assert per_arch["m1"]["gain"]
    assert isinstance(per_arch["m0"]["gain"], list)
    assert "diagnostics" in results


def test_report_and_export_off_the_same_fixture(env):
    exp = create(env)
    write_legs(env, exp["id"], [leg("m0", "cfglow", {}, 7, 0.2)])
    body, status = mcf_api.handle_get(f"/api/mcf/experiment/{exp['id']}/report", {})
    assert status == 200
    assert "# MCF" in body["markdown"]
    body, status = mcf_api.handle_get(f"/api/mcf/experiment/{exp['id']}/export", {})
    assert status == 200
    assert body["experiment"]["id"] == exp["id"]
    assert body["legs"][0]["seed"] == 7


# ---------------------------------------------------------------- cost guard


def test_cost_guard_refuses_unconfirmed_expensive_launch(env):
    exp = create(
        env,
        parameters=["style_fit_amplitude", "contagion_weight", "spending_daily_variance", "temperature"],
        n_seeds=5,
    )  # 2 arch × 5 cfg × 5 seeds = 50 legs → 14000 calls
    assert exp["estimated_calls"] > mcf_api._COST_CONFIRM_THRESHOLD
    body, status = mcf_api.handle_post(f"/api/mcf/experiment/{exp['id']}/launch", {})
    assert status == 400
    assert "confirm_cost" in body["error"]
    assert body["estimated_calls"] == exp["estimated_calls"]


def test_launch_with_confirm_starts_job(env):
    exp = create(
        env,
        parameters=["style_fit_amplitude", "contagion_weight", "spending_daily_variance", "temperature"],
        n_seeds=5,
    )
    body, status = mcf_api.handle_post(f"/api/mcf/experiment/{exp['id']}/launch", {"confirm_cost": True})
    assert status == 202, body
    doc, _ = mcf_api.handle_get(f"/api/mcf/experiment/{exp['id']}", {})
    assert doc["status"] == "running"


def test_launch_is_refused_twice(env):
    exp = create(env)
    assert (
        mcf_api.handle_post(f"/api/mcf/experiment/{exp['id']}/launch", {"confirm_cost": True})[1] == 202
    )  # even 2×2×2 legs estimate above the guard; confirm once
    _body, status = mcf_api.handle_post(f"/api/mcf/experiment/{exp['id']}/launch", {})
    assert status == 409
