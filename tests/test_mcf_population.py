"""Runner-integration tests: city / resident selection / duration must reach
the actual simulation engine, and persisted experiments must round-trip.

LLM boundary: the real vehicle code path (``sim_leg_execute``) is exercised
with ``generative_city_sim.run_simulation`` replaced by a capturing double —
no inference is possible, while everything around it (CONFIG patching, module
global rebinding, restore) is the real production code.
"""

import json

from benchmark.mcf.population import (
    city_roster,
    config_overrides_for,
    random_sample,
    resolve_population,
)
from gaworld.apps import mcf_api

CITY_A = "伦敦"
CITY_B = "wuzhen"  # 乌镇镇 — a bundle with a real 63-resident roster


def _ids(city):
    return [r["id"] for r in city_roster(city)]


# 1. two different cities produce different selected populations


def test_two_cities_produce_different_populations():
    a = resolve_population({"city": CITY_A, "method": "random", "sample_size": 10, "sample_seed": 1})
    b = resolve_population({"city": CITY_B, "method": "random", "sample_size": 10, "sample_seed": 1})
    assert a["city"] == CITY_A and b["city"] == CITY_B
    assert a["agent_ids"] != b["agent_ids"]


# 2. manual selection survives pagination (selection across pages resolves fully)


def test_manual_selection_across_pages():
    from gaworld.apps import city_api

    page1 = city_api.agents(CITY_A, limit=5, offset=0)
    page2 = city_api.agents(CITY_A, limit=5, offset=5)
    assert page1["agents"] and page2["agents"]
    picked = [a["id"] for a in page1["agents"][:2]] + [a["id"] for a in page2["agents"][:2]]
    resolved = resolve_population({"city": CITY_A, "method": "manual", "agent_ids": picked})
    assert sorted(resolved["agent_ids"]) == sorted(picked)


# 3. identical seeds reproduce identical samples


def test_random_sample_reproducible():
    ids = _ids(CITY_A)
    s1 = random_sample(ids, 15, 42)
    s2 = random_sample(ids, 15, 42)
    assert s1 == s2
    assert s1 != random_sample(ids, 15, 43)


# 4. entire-population selection does not silently truncate


def test_all_population_no_truncation():
    all_ids = _ids(CITY_A)
    resolved = resolve_population({"city": CITY_A, "method": "all"})
    assert resolved["agent_ids"] == sorted(all_ids)
    assert resolved["n_agents"] == len(all_ids)


# 5. selected residents reach actual simulation initialization (real vehicle,
#    LLM boundary replaced by a capturing double)


def test_selected_residents_and_city_reach_the_engine(monkeypatch):
    import generative_city_sim as gcs

    captured = {}

    def fake_run_simulation():
        captured["AGENT_IDS"] = list(gcs.AGENT_IDS)
        captured["SIM_DAYS"] = gcs.SIM_DAYS
        captured["CSV_PATH"] = str(gcs.CSV_PATH)
        captured["CONFIG_CITY"] = gcs.CONFIG.get("city")
        captured["CONFIG_AGENTS"] = list(gcs.CONFIG.get("agent_ids") or [])
        captured["CONFIG_DAYS"] = gcs.CONFIG.get("sim_days")

    monkeypatch.setattr(gcs, "run_simulation", fake_run_simulation)
    overrides = config_overrides_for(CITY_B)
    ids = _ids(CITY_B)[:6]
    sim_context = dict(overrides)
    sim_context["agent_ids"] = ids
    sim_context["sim_days"] = 12
    sim_context["population_provenance"] = {"city": CITY_B, "method": "manual", "n_agents": len(ids)}
    from pathlib import Path

    from benchmark.mcf.leg_runner import sim_leg_execute

    sim_leg_execute({"personality.style_fit_amplitude": 0.3}, Path("/tmp"), seed=7, sim_context=sim_context)
    assert captured["AGENT_IDS"] == ids
    assert captured["SIM_DAYS"] == 12
    assert captured["CONFIG_AGENTS"] == ids
    assert captured["CONFIG_DAYS"] == 12
    assert captured["CONFIG_CITY"] == CITY_B
    assert captured["CSV_PATH"].endswith("agents.csv")


def test_engine_state_restored_after_leg(monkeypatch):
    import generative_city_sim as gcs

    monkeypatch.setattr(gcs, "run_simulation", lambda: None)
    before_ids = list(gcs.AGENT_IDS)
    before_days = gcs.SIM_DAYS
    before_csv = str(gcs.CSV_PATH)
    from pathlib import Path

    from benchmark.mcf.leg_runner import sim_leg_execute

    sim_context = dict(config_overrides_for(CITY_A))
    sim_context["agent_ids"] = _ids(CITY_A)[:3]
    sim_context["sim_days"] = 3
    sim_leg_execute({}, Path("/tmp"), seed=1, sim_context=sim_context)
    assert list(gcs.AGENT_IDS) == before_ids
    assert before_days == gcs.SIM_DAYS
    assert str(gcs.CSV_PATH) == before_csv


# 5b. harness-level propagation: the orchestrator-facing harness hands the
# sim context to the vehicle (LLM boundary: execute double)


def test_harness_passes_sim_context_to_vehicle(tmp_path):
    from benchmark.mcf.leg_runner import make_leg_harness

    seen = {}

    def vehicle(config, run_dir, *, seed=0, sim_context=None):
        seen["context"] = sim_context
        return {"ok": True, "usage": {}, "episodes_dir": str(run_dir)}

    harness = make_leg_harness(
        execute=vehicle,
        sim_context={
            "city": CITY_A,
            "csv_path": "data/cities/伦敦/agents.csv",
            "agent_ids": [1, 2, 3],
            "sim_days": 9,
            "population_provenance": {"city": CITY_A, "method": "manual", "n_agents": 3},
        },
    )
    out = harness({"personality.style_fit_amplitude": 0.2}, tmp_path / "leg", seed=5)
    assert out["ok"] is True
    assert seen["context"]["agent_ids"] == [1, 2, 3]
    assert seen["context"]["sim_days"] == 9


# 6. duration changes the requested simulation horizon


def test_duration_changes_horizon(monkeypatch):
    import generative_city_sim as gcs

    seen = []
    monkeypatch.setattr(gcs, "run_simulation", lambda: seen.append(gcs.SIM_DAYS))
    from pathlib import Path

    from benchmark.mcf.leg_runner import sim_leg_execute

    base = dict(config_overrides_for(CITY_A))
    base["agent_ids"] = _ids(CITY_A)[:2]
    for days in (3, 11):
        ctx = dict(base)
        ctx["sim_days"] = days
        sim_leg_execute({}, Path("/tmp"), seed=0, sim_context=ctx)
    assert seen == [3, 11]


# 7. derived employment IDs reach the orchestrator/scorer path


def test_derived_employed_ids_reach_run_legs(monkeypatch, tmp_path):
    exp_payload = {
        "name": "pop-integration",
        "population": {"city": CITY_A, "method": "manual", "agent_ids": _ids(CITY_A)[:8]},
        "sim_days": 9,
        "architectures": [
            {"id": "m0", "label": "on", "patch": {}},
            {"id": "m1", "label": "off", "patch": {"personality.channels.rules": False}},
        ],
        "parameters": ["style_fit_amplitude"],
        "sampler": "endpoints",
        "n_configs": 1,
        "seeds": [7],
    }
    monkeypatch.setattr(mcf_api, "_mcf_root", lambda: str(tmp_path / "mcf"))
    created, code = mcf_api.create_experiment(exp_payload)
    assert code == 201
    derived = created["population"]["employed_agent_ids"]
    roster = {r["id"]: r for r in city_roster(CITY_A)}
    expected = sorted(i for i in created["population"]["agent_ids"] if roster[i]["employment"] == "employed")
    assert derived == expected and derived

    captured = {}
    fake_exp = dict(created)

    import benchmark.mcf.orchestrator as orch

    def fake_run_legs(legs, *, harness, run_root, **kwargs):
        captured["employed"] = kwargs.get("employed_agent_ids")
        return {"legs": []}

    monkeypatch.setattr(orch, "run_legs", fake_run_legs)
    monkeypatch.setattr(mcf_api, "_build_harness", lambda exp: lambda c, d, seed=0: {})
    monkeypatch.setattr(mcf_api, "_save_exp", lambda exp: None)
    monkeypatch.setattr(mcf_api, "_load_exp", lambda _id: fake_exp)
    mcf_api._run_job(created["id"])
    assert captured["employed"] == tuple(derived)


# 8. saved experiments reopen without losing configuration


def test_population_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mcf_api, "_mcf_root", lambda: str(tmp_path))
    payload = {
        "name": "roundtrip",
        "population": {"city": CITY_B, "method": "random", "sample_size": 4, "sample_seed": 11},
        "sim_days": 10,
        "architectures": [
            {"id": "m0", "label": "on", "patch": {}},
            {"id": "m1", "label": "off", "patch": {"personality.channels.rules": False}},
        ],
        "parameters": ["style_fit_amplitude"],
        "sampler": "endpoints",
        "n_configs": 1,
        "seeds": [7],
    }
    created, code = mcf_api.create_experiment(payload)
    assert code == 201
    reopened = mcf_api._load_exp(created["id"])
    assert reopened["population"]["city"] == CITY_B
    assert reopened["population"]["method"] == "random"
    assert reopened["population"]["sample_seed"] == 11
    assert reopened["population"]["agent_ids"] == created["population"]["agent_ids"]
    assert reopened["sim_days"] == 10
    assert reopened["employed_agent_ids"] == created["employed_agent_ids"]


# 9. invalid inputs rejected at the backend


def test_invalid_inputs_rejected():
    good_archs = [
        {"id": "m0", "label": "a", "patch": {}},
        {"id": "m1", "label": "b", "patch": {"personality.channels.rules": False}},
    ]
    base = {
        "architectures": good_archs,
        "parameters": ["style_fit_amplitude"],
        "sampler": "endpoints",
        "n_configs": 1,
        "seeds": [7],
    }
    cases = [
        {"population": {"city": "narnia", "method": "all"}},
        {"population": {"city": CITY_A, "method": "manual", "agent_ids": [99999999]}},
        {"population": {"city": CITY_A, "method": "random", "sample_size": 10**6, "sample_seed": 1}},
        {"population": {"city": CITY_A, "method": "bogus"}},
        {"population": {"city": CITY_A, "method": "all"}, "sim_days": 0},
        {"population": {"city": CITY_A, "method": "all"}, "sim_days": 2.5},
    ]
    for extra in cases:
        payload = dict(base)
        payload.update(extra)
        created, code = mcf_api.create_experiment(payload)
        assert code == 400, (extra, created)


# 10. historical MCF results remain unchanged / old manifests readable


def test_historical_experiments_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(mcf_api, "_mcf_root", lambda: str(tmp_path / "mcf"))
    with open("output/mcf/phase5-final/experiment.json", encoding="utf-8") as fh:
        exp = json.load(fh)
    assert "population" not in exp and "sim_days" not in exp
    created, code = mcf_api.create_experiment(
        {
            "name": "legacy-shape",
            "architectures": [
                {"id": "m0", "label": "a", "patch": {}},
                {"id": "m1", "label": "b", "patch": {"personality.channels.rules": False}},
            ],
            "parameters": ["style_fit_amplitude"],
            "sampler": "endpoints",
            "n_configs": 1,
            "seeds": [7],
        }
    )
    assert code == 201
    assert "population" not in created and "sim_days" not in created
    harness = mcf_api._build_harness(created)
    assert harness is not None


def test_sample_preview_endpoint_contract():
    preview, code = mcf_api.handle_get(
        "/api/mcf/sample-preview",
        {"city": CITY_A, "method": "random", "sample_size": "5", "sample_seed": "42"},
    )
    assert code == 200
    assert len(preview["agent_ids"]) == 5
    assert preview["n_employed"] <= 5
    again, _ = mcf_api.handle_get(
        "/api/mcf/sample-preview",
        {"city": CITY_A, "method": "random", "sample_size": "5", "sample_seed": "42"},
    )
    assert again["agent_ids"] == preview["agent_ids"]
    _err, code = mcf_api.handle_get("/api/mcf/sample-preview", {"city": "narnia", "method": "all"})
    assert code == 400
