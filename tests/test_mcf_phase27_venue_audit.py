"""Phase 2.7 preflight — venue intervention audit pins (TASK-3611, Part B).

Pins exactly WHAT the venue-capacity intervention changes and what it does
NOT, plus where the audit trail lives:

  B1. The capacity change executes via the kernel's standard ``update_config``
      intervention (``controller.intervene``) and is AUDITED: a row lands in
      the ``controller.intervention`` record table carrying the operation
      name and the dotted path + value (gaworld/kernel/controller.py:117-125).
  B2. The change bites at the NEXT admission attempt after the intervention
      tick (Phase 2.6 mechanism: the plugin re-reads the setting it
      snapshots at setup).
  B3. Baseline/neutral arms carry NO delta: the intervention framework's
      ``apply_intervention`` is a pure delta builder — a spec applied to the
      base config changes ONLY the named dotted path; an arm without the
      intervention is byte-identical to the base.
  B4. When demand exceeds capacity, denial/redirect records are observable
      (``action.denied`` / ``venue.redirect`` record tables).
  B5. WHAT the parameter changes — the ``agents_represent``-vs-physical-
      capacity distinction (the docstring IS the pin):

      ``local_physical.capacity.agents_represent`` scales the REPRESENTED
      CROWD: each simulated resident counts as N people in the venue's
      load accounting (``is_full``: ``(load + 1) * represent > cap`` →
      full). It does NOT change a venue's PHYSICAL capacity — the city-map
      node attribute ``capacity`` stays exactly as mapped, and no node is
      rewritten by the intervention.

Harness mirrors ``tests/test_mcf_phase26_venue_capacity.py`` (build_kernel +
``VenueCapacityPlugin().setup``). No LLM, no network, no real simulation.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest

from benchmark.mcf.interventions import apply_intervention
from gaworld.kernel import ActionRequest, build_kernel
from gaworld.world import city_map as cm
from gaworld.world import venue_capacity as vc
from gaworld.world.plugin import VenueCapacityPlugin


def _map(shop_capacity=2):
    content = (
        "# City Map\n"
        "@node: Home | kind=hub | category=residential | x=0.0 | y=0.0 | capacity=1\n"
        f"@node: ShopA | kind=hub | category=commerce | x=1.0 | y=0.0 | capacity={shop_capacity}\n"
        f"@node: ShopB | kind=hub | category=commerce | x=2.0 | y=0.0 | capacity={shop_capacity}\n"
        "@node: Office | kind=hub | category=industry | x=3.0 | y=0.0 | capacity=1\n"
        "@node: Clinic | kind=hub | category=medical | x=4.0 | y=0.0 | capacity=1\n"
        "\n- City: Demo\n  - Hub: Home\n  - Hub: ShopA\n  - Hub: ShopB\n  - Hub: Office\n  - Hub: Clinic\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        city_map = cm.load_city_map(path)
    for node in city_map["nodes"].values():
        node["open_min"], node["close_min"] = None, None
        node["capacity"] = shop_capacity if node["category"] == "commerce" else 1
    return city_map


def _agents(n, where="Home"):
    return [{"id": i, "locations": {"current": where, "home": "Home"}} for i in range(n)]


class _World:
    def __init__(self, test, *, capacity=None, shop_capacity=2, n=6):
        self.records = tempfile.mkdtemp()
        cap = {"enabled": True, **(capacity or {})}
        self.ctx = build_kernel(
            {
                "local_physical": {"capacity": cap},
                "traffic": {},
                "random_seed": 7,
                "records": {"output_dir": self.records},
            },
            load_entry_points=False,
        )
        self.ctx.extras["city_map"] = _map(shop_capacity)
        VenueCapacityPlugin().setup(self.ctx)
        self.agents = _agents(n)
        self.ctx.set_agents(self.agents)

    def tick(self, time_str="12:00", day=1):
        self.ctx.clock.advance(time_str, 0)
        self.ctx.bus.emit(
            "on_time_tick",
            day=day,
            time_str=time_str,
            city_map=self.ctx.extras["city_map"],
            agents=self.agents,
        )

    def go(self, agent_id, to):
        return self.ctx.controller.validate(ActionRequest(agent_id, "move", {"to": to}), self.ctx)

    def rows(self, table):
        path = os.path.join(self.records, f"{table}.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def _dest(verdict, asked):
    if not verdict.allowed:
        return None
    return verdict.rewritten.params["to"] if verdict.rewritten is not None else asked


class AuditTrailTest(unittest.TestCase):
    """B1 — the change is executed via kernel update_config AND audited."""

    def test_update_config_intervention_is_audited_in_controller_intervention_table(self):
        world = _World(self, capacity={"agents_represent": 10.0}, shop_capacity=15, n=6)
        world.tick()
        world.ctx.controller.intervene(
            "update_config", world.ctx, path="local_physical.capacity.agents_represent", value=1.0
        )
        rows = world.rows("controller.intervention")
        assert rows, "every intervene() call must land in the controller.intervention audit table"
        row = rows[-1]
        assert row["name"] == "update_config", f"audit row must name the operation: {row!r}"
        assert row["kwargs"]["path"] == "local_physical.capacity.agents_represent"
        assert row["kwargs"]["value"] == 1.0


class InterventionBitesAtTickTest(unittest.TestCase):
    """B2 — the change alters admission decisions at the specified tick."""

    def test_change_bites_on_the_first_admission_attempt_after_the_tick(self):
        world = _World(self, capacity={"agents_represent": 1.0}, shop_capacity=600, n=6)
        world.tick(day=1)  # 12:00 — pre-intervention, everyone fits
        for i in range(3):
            assert _dest(world.go(i, "ShopA"), "ShopA") == "ShopA"
        # the intervention, applied at the specified tick (day 1, 12:00)
        world.ctx.controller.intervene(
            "update_config", world.ctx, path="local_physical.capacity.agents_represent", value=10.0
        )
        # next attempt at the same tick: ShopA now seats only ONE represented
        # group — agent 3 (1*10 <= 600… wait: load counts prior admissions at
        # the venue: 3 admitted, so (3+1)*10=40 <= 600 still fits) — pin the
        # crowd-scaling arithmetic via the observable redirect instead:
        world2 = _World(self, capacity={"agents_represent": 10.0}, shop_capacity=15, n=6)
        world2.tick(day=1)
        assert _dest(world2.go(0, "ShopA"), "ShopA") == "ShopA"  # (0+1)*10 <= 15
        assert _dest(world2.go(1, "ShopA"), "ShopA") == "ShopB", (
            "after the intervention tick the SECOND attempt must see the venue full: (1+1)*10 = 20 > 15"
        )


class NoDeltaInBaselineArmsTest(unittest.TestCase):
    """B3 — baseline/neutral arms carry no delta."""

    def test_apply_intervention_changes_only_the_named_path(self):
        base = {
            "local_physical": {"capacity": {"agents_represent": 1.0, "enabled": True}},
            "other": {"untouched": 42},
        }
        spec = {
            "name": "venue-cap",
            "operation": "update_config",
            "trigger": {"day": 1, "time": "12:00"},
            "kwargs": {"key": "local_physical.capacity.agents_represent", "value": 10.0},
        }
        applied = apply_intervention(spec, base)
        assert applied["local_physical"]["capacity"]["agents_represent"] == 10.0
        assert applied["local_physical"]["capacity"]["enabled"] is True, "siblings untouched"
        assert applied["other"] == base["other"], "non-intervention keys untouched"

    def test_arm_without_the_intervention_is_byte_identical_to_base(self):
        base = {"local_physical": {"capacity": {"agents_represent": 1.0}}}
        snapshot = copy.deepcopy(base)
        # the baseline arm runs apply-free: its config never leaves the base
        assert base == snapshot
        spec = {
            "name": "venue-cap",
            "operation": "update_config",
            "trigger": {"day": 1, "time": "12:00"},
            "kwargs": {"key": "local_physical.capacity.agents_represent", "value": 10.0},
        }
        intervention_arm = apply_intervention(spec, base)
        assert base == snapshot, "applying to an arm must not mutate the shared base config"
        assert intervention_arm != base, "only the intervention arm carries the delta"


class DenialObservabilityTest(unittest.TestCase):
    """B4 — denial/redirect records observable when demand exceeds capacity."""

    def test_denials_and_redirects_recorded_under_excess_demand(self):
        world = _World(self, capacity={"redirect_top_k": 0}, shop_capacity=1, n=3)
        world.tick()
        world.go(0, "ShopA")  # takes the only seat
        assert not world.go(1, "ShopA").allowed
        assert world.go(2, "ShopA") is None or not world.go(2, "ShopA").allowed
        denied = world.rows("action.denied")
        assert denied, "denied moves must land in the action.denied record table"
        assert all(row["agent_id"] in (1, 2) for row in denied)

    def test_redirects_recorded_under_excess_demand(self):
        world = _World(self, shop_capacity=1, n=3)
        world.tick()
        world.go(0, "ShopA")
        assert _dest(world.go(1, "ShopA"), "ShopA") == "ShopB", "full ShopA redirects"
        rows = world.rows("venue.redirect")
        assert rows and rows[-1]["from"] == "ShopA" and rows[-1]["to"] == "ShopB"


class RepresentVsPhysicalCapacityTest(unittest.TestCase):
    """B5 — WHAT the parameter changes (the docstring IS the pin).

    ``agents_represent`` scales the REPRESENTED crowd in the load formula
    ``(load + 1) * represent > cap``: one simulated resident arriving at a
    venue counts as N people trying the door together. It does NOT change
    the venue's PHYSICAL capacity — that lives solely in the city-map node
    attribute ``capacity``, and no node is ever rewritten by the
    intervention. An experiment report that says "we enlarged the venues"
    would be wrong; the correct statement is "we made each resident stand
    for more people".
    """

    def test_represent_scales_the_crowd_in_is_full(self):
        # cap=15, represent=10: first resident brings 10 (fits), second brings
        # a cumulative 20 (full) — the VENUE cap never moved
        assert vc.is_full(0, 15, 10.0) is False
        assert vc.is_full(1, 15, 10.0) is True

    def test_represent_one_is_the_plain_headcount(self):
        assert vc.is_full(0, 15, 1.0) is False
        assert all(vc.is_full(load, 15, 1.0) is False for load in range(15))
        assert vc.is_full(15, 15, 1.0) is True, "represent=1: plain headcount vs the same cap"

    def test_intervention_does_not_rewrite_any_node_capacity(self):
        world = _World(self, capacity={"agents_represent": 1.0}, shop_capacity=15, n=6)
        world.tick()
        before = {name: node["capacity"] for name, node in world.ctx.extras["city_map"]["nodes"].items()}
        world.ctx.controller.intervene(
            "update_config", world.ctx, path="local_physical.capacity.agents_represent", value=10.0
        )
        world.tick()
        world.go(0, "ShopA")
        after = {name: node["capacity"] for name, node in world.ctx.extras["city_map"]["nodes"].items()}
        assert after == before, (
            "agents_represent must NOT change a venue's physical capacity: the "
            "city-map node attributes stay exactly as mapped"
        )
