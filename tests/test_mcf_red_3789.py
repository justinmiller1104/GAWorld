"""TDD RED tests for the 5 MCF defects from RESULT-3787 (task 3789).

Each test pins behaviour that does NOT exist on HEAD @ 67199cd, so each must
fail for the right reason (missing feature), never a syntax error:

1. Arch/Θ disjointness — an architecture patch naming a Θ band parameter
   (e.g. contagion_weight) is rejected; mechanism toggles are accepted
   (benchmark/mcf/config_space.py).
2. Spec-level validation of the same rule (gaworld/apps/mcf_api.py
   ``_validate_spec``).
3. ``employed_agent_ids`` accepted for real-harness experiments and threaded
   into ``run_legs`` (paid_work scoreable for real legs); stub unchanged.
4. Span/snapshot double counting — when spans exist for an agent-day, the
   merged classified+unknown totals must not exceed 1440 min (currently
   snapshot reconstruction re-attributes span-covered intervals).
5. Keyword vocabulary — 兼职/接单 classify as paid_work.

All fixtures are synthetic dicts / JSONL in tmp_path. No LLM, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.mcf.config_space import validate_config
from benchmark.mcf.time_use import classify_activity, score_directory
from gaworld.apps import mcf_api

THETA_PATH = "personality.emotion_baseline.contagion_weight"
RESERVED_MESSAGE_PARTS = ("Θ parameter", "architecture")


# ---------------------------------------------------------------------------
# 1. Arch/Θ disjointness (config_space)
# ---------------------------------------------------------------------------


def test_validate_arch_patch_rejects_band_parameter():
    """An arch patch containing a Θ band path must be rejected with the
    documented error (feature: ``validate_arch_patch``)."""
    from benchmark.mcf import config_space

    validator = getattr(config_space, "validate_arch_patch", None)
    assert validator is not None, (
        "config_space.validate_arch_patch is missing — arch/Θ disjointness "
        "is unenforceable (RED: feature not implemented)"
    )
    with pytest.raises(ValueError) as excinfo:
        validator({THETA_PATH: 0.12})
    message = str(excinfo.value)
    for part in RESERVED_MESSAGE_PARTS:
        assert part in message, f"error {message!r} must mention {part!r}"


def test_validate_arch_patch_accepts_mechanism_toggle():
    """A mechanism-level patch (boolean channel toggle) passes validation."""
    from benchmark.mcf import config_space

    validator = getattr(config_space, "validate_arch_patch", None)
    assert validator is not None, "config_space.validate_arch_patch is missing (RED)"
    nested = validator({"personality.channels.rules": True})
    assert isinstance(nested, dict)
    assert nested, "mechanism toggle must produce a non-empty nested patch"


def test_theta_path_remains_a_valid_config_parameter():
    """The Θ side is untouched: contagion_weight is still a valid sampled
    config parameter (the two namespaces must stay disjoint, not both die)."""
    nested = validate_config({THETA_PATH: 0.06})
    assert nested


# ---------------------------------------------------------------------------
# 2. Spec-level validation (mcf_api._validate_spec)
# ---------------------------------------------------------------------------

SPEC = {
    "name": "smoke",
    "architectures": [
        {"id": "m0", "label": "M0 baseline", "patch": {}},
        {"id": "m1", "label": "M1 mechanism", "patch": {"personality.channels.rules": True}},
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
    monkeypatch.setattr(mcf_api, "_start_job", lambda exp_id: None)
    return tmp_path


def test_spec_rejects_theta_parameter_in_arch_patch(env):
    body, status = mcf_api.handle_post(
        "/api/mcf/experiment",
        {
            **SPEC,
            "architectures": [
                {"id": "m0", "label": "M0", "patch": {}},
                {"id": "m1", "label": "M1 band-high", "patch": {THETA_PATH: 0.12}},
            ],
        },
    )
    assert status == 400, f"Θ parameter in an arch patch must be rejected, got {status}: {body}"
    message = str(body)
    for part in RESERVED_MESSAGE_PARTS:
        assert part in message, f"error {message!r} must mention {part!r}"


def test_spec_accepts_mechanism_toggle_in_arch_patch(env):
    body, status = mcf_api.handle_post("/api/mcf/experiment", SPEC)
    assert status == 201, body
    exp_id = body["id"]
    stored, _ = mcf_api.handle_get(f"/api/mcf/experiment/{exp_id}", {})
    archs = stored["experiment"]["architectures"] if "experiment" in stored else stored["architectures"]
    assert archs[1]["patch"] == {"personality.channels.rules": True}


# ---------------------------------------------------------------------------
# 3. employed_agent_ids plumbing for real experiments
# ---------------------------------------------------------------------------


class _RecordingHarness:
    def __call__(self, *args, **kwargs):  # pragma: no cover - replaced
        raise AssertionError("harness must not run in this test")


def test_run_job_threads_employed_agent_ids_for_real_experiments(env, monkeypatch):
    """A real (non-stub) experiment spec declares ``employed_agent_ids`` and
    ``_run_job`` passes them into ``run_legs`` so paid_work is scoreable."""
    body, status = mcf_api.handle_post(
        "/api/mcf/experiment", {**SPEC, "harness": "real", "employed_agent_ids": [3, 4]}
    )
    assert status == 201, body
    exp_id = body["id"]
    stored, _ = mcf_api.handle_get(f"/api/mcf/experiment/{exp_id}", {})
    exp = stored.get("experiment", stored)
    assert exp.get("employed_agent_ids") == [3, 4], (
        "spec must persist employed_agent_ids for real experiments (RED: plumbing missing)"
    )

    recorded: dict[str, object] = {}

    def fake_run_legs(leg_pairs, *, harness, run_root, limits=None, scorer=None, **kwargs):
        recorded["employed_agent_ids"] = kwargs.get("employed_agent_ids")
        return {"legs": []}

    import benchmark.mcf.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "run_legs", fake_run_legs)
    monkeypatch.setattr(mcf_api, "_build_harness", lambda exp: _RecordingHarness())
    monkeypatch.setattr(mcf_api, "_build_scorer", lambda exp: None)
    monkeypatch.setattr(mcf_api, "_is_stub", lambda exp: False)

    mcf_api._run_job(exp_id)
    assert recorded.get("employed_agent_ids") == (3, 4), (
        f"_run_job must thread the declared cohort into run_legs, got "
        f"{recorded.get('employed_agent_ids')!r} (RED: real legs always get ())"
    )


def test_stub_employed_cohort_unchanged(env, monkeypatch):
    """Stub legs keep their built-in cohort — no regression on the stub path."""
    from benchmark.mcf.leg_runner import STUB_EMPLOYED_AGENTS

    body, status = mcf_api.handle_post("/api/mcf/experiment", {**SPEC, "harness": "stub"})
    assert status == 201, body
    exp_id = body["id"]

    recorded: dict[str, object] = {}

    def fake_run_legs(leg_pairs, *, harness, run_root, limits=None, scorer=None, **kwargs):
        recorded["employed_agent_ids"] = kwargs.get("employed_agent_ids")
        return {"legs": []}

    import benchmark.mcf.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "run_legs", fake_run_legs)
    monkeypatch.setattr(mcf_api, "_build_harness", lambda exp: _RecordingHarness())
    monkeypatch.setattr(mcf_api, "_build_scorer", lambda exp: None)
    monkeypatch.setattr(mcf_api, "_is_stub", lambda exp: True)

    mcf_api._run_job(exp_id)
    assert recorded.get("employed_agent_ids") == STUB_EMPLOYED_AGENTS


# ---------------------------------------------------------------------------
# 4. Span/snapshot double counting (time_use.score_directory)
# ---------------------------------------------------------------------------


def _write_agent_jsonl(path: Path) -> None:
    """One agent-day where spans tile 00:00–20:00 and two snapshots sit at
    07:00 (inside the sleep span) and 21:00 (outside every span).

    Correct accounting: span minutes (480 sleep + 720 paid_work) + snapshot
    attribution confined to the unspanned 20:00–24:00 tail ⇒ total ≤ 1440.
    The current code reconstructs the full 07:00→21:00 interval between the
    two surviving snapshots, re-attributing span-covered minutes ⇒ > 1440.
    """
    rows: list[dict[str, object]] = [
        {
            "kind": "span",
            "agent_id": 5,
            "day": 1,
            "start": "00:00",
            "end": "08:00",
            "activity": "睡觉",
            "purpose": "",
        },
        {
            "kind": "span",
            "agent_id": 5,
            "day": 1,
            "start": "08:00",
            "end": "20:00",
            "activity": "上班",
            "purpose": "",
        },
        {"agent_id": 5, "day": 1, "time": "07:00", "final_activity": "睡觉"},
        {"agent_id": 5, "day": 1, "time": "21:00", "final_activity": "休闲"},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_spans_plus_snapshots_do_not_double_count(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_agent_jsonl(run_dir / "agent_5.jsonl")

    result = score_directory(run_dir, employed_agent_ids=(5,), tick_minutes=10)
    day = result.per_agent_day[5][1]
    total = sum(day.values())
    assert total <= 1440, (
        f"agent-day totals must not exceed 1440 min when spans exist; got {total} "
        f"({dict(day)}) — span/snapshot double counting (RED: defect 4 unfixed)"
    )
    assert day["paid_work"] == 720, f"span minutes must be preserved exactly, got {day['paid_work']}"


# ---------------------------------------------------------------------------
# 5. Keyword vocabulary: 兼职 / 接单 are paid_work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["兼职", "接单"])
def test_part_time_labels_classify_as_paid_work(label):
    category, confidence = classify_activity(label)
    assert category == "paid_work", (
        f"{label!r} must classify as paid_work, got {category!r} "
        f"(RED: {label!r} missing from the paid_work vocabulary)"
    )
    assert confidence == "keyword"
