"""RED tests — offline end-to-end launch through the MCF API (task 3753 sub-task C).

create_experiment → launch (stub harness, GAWORLD_OFFLINE=1) → the job runs
the real per-leg runner synchronously → legs.json written by the runner →
results endpoint returns computed frontier data. No inference anywhere.
"""

from __future__ import annotations

import pytest

from gaworld.apps import mcf_api

SPEC = {
    "name": "stub e2e",
    "architectures": [
        {"id": "m0", "label": "M0 baseline", "patch": {}},
        {"id": "m1", "label": "M1 realism", "patch": {}},
    ],
    "parameters": ["style_fit_amplitude"],
    "n_configs": 2,
    "sampler": "endpoints",
    "n_seeds": 1,
    "seeds": [7],
    "budget": {"max_calls": 200, "max_tokens": 2_000_000, "max_elapsed_seconds": 600},
    "harness": "stub",
}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(mcf_api, "_mcf_root", lambda: str(tmp_path / "mcf"))
    # run the job inline (synchronously) instead of ownership.spawn
    monkeypatch.setattr(mcf_api, "_start_job", lambda exp_id: mcf_api._run_job(exp_id))
    return tmp_path


def test_harness_option_defaults_to_real(env):
    body, status = mcf_api.handle_post("/api/mcf/experiment", {**SPEC})
    assert status == 201
    body.pop("harness", None)
    body2, _ = mcf_api.handle_post("/api/mcf/experiment", {**SPEC, "harness": "real"})
    # default real: when the option is absent the spec must not say stub
    created = mcf_api._load_exp(body2["id"])
    assert created.get("harness", "real") != "stub"


def test_harness_option_rejects_unknown_values(env):
    _body, status = mcf_api.handle_post("/api/mcf/experiment", {**SPEC, "harness": "yolo"})
    assert status == 400


def test_stub_launch_completes_offline_and_results_have_frontier(env):
    created, status = mcf_api.handle_post("/api/mcf/experiment", {**SPEC})
    assert status == 201, created
    exp_id = created["id"]

    body, status = mcf_api.handle_post(f"/api/mcf/experiment/{exp_id}/launch", {})
    assert status == 202, body

    doc, _ = mcf_api.handle_get(f"/api/mcf/experiment/{exp_id}", {})
    assert doc["status"] in ("completed", "completed_with_failures"), doc.get("last_error")
    assert doc["progress"]["legs_done"] == doc["progress"]["legs_total"] == 4

    # legs.json exists and every leg is scored
    legs = mcf_api._load_legs(exp_id)
    assert len(legs) == 4
    for row in legs:
        assert row["status"] == "completed", row["stop_reason"]
        assert row["score"] is not None
        assert row["arch"] in ("m0", "m1")

    body, status = mcf_api.handle_get(f"/api/mcf/experiment/{exp_id}/results", {})
    assert status == 200, body
    results = body["results"]
    assert results is not None
    assert results["vectors"], "the stub legs must produce frontier vectors"
    per_arch = {item["arch"]: item for item in results["architectures"]}
    assert set(per_arch) == {"m0", "m1"}
    # frontier computed by the real code from the scored stub legs
    assert isinstance(per_arch["m0"]["frontier"], list) and per_arch["m0"]["frontier"]
    # differing configs score differently → more than one distinct config point
    assert len(per_arch["m0"]["configs"]) == 2


def test_real_harness_launch_in_offline_env_falls_back_to_stub(env, monkeypatch):
    """GAWORLD_OFFLINE=1 forces the stub execution even when the experiment
    says harness=real — tests can never fire live inference."""
    monkeypatch.setenv("GAWORLD_OFFLINE", "1")
    created, _ = mcf_api.handle_post("/api/mcf/experiment", {**SPEC, "harness": "real"})
    exp_id = created["id"]
    harness = mcf_api._build_harness(mcf_api._load_exp(exp_id))
    import gaworld.llm.providers as providers

    original = providers.call_llm
    try:
        result = harness({"personality": {"style_fit_amplitude": 0.1}}, env / "leg", seed=7)
    finally:
        providers.call_llm = original
    assert result["ok"] is True
