"""Phase 2.6 RED tests — venue-capacity runtime contracts (TASK-3597).

Part A of the Phase 2.6 spec: pin how ``VenueCapacityPlugin``
(``gaworld/world/plugin.py`` + ``gaworld/world/venue_capacity.py``) behaves
under a RUNTIME capacity change, how denials are recorded, and that an
uncongested config produces zero denials.

Investigation findings being pinned (see RESULT-3597 for full citations):

  * ``setup()`` snapshots ``local_physical.capacity`` ONCE
    (plugin.py:513-541: ``enabled`` :515, ``categories`` :516,
    ``agents_represent`` :518-523, ``redirect_top_k`` :524). Capacities
    themselves come from city-map NODE attributes, read live per admission
    attempt (``node.get("capacity")`` at plugin.py:643 and :676; scaled by
    ``is_full`` in venue_capacity.py:24-30).
  * ``update_config`` (gaworld/kernel/interventions.py:40-47) writes the live
    CONFIG dict — but the plugin never re-reads it, so config-side knobs
    (``agents_represent``, ``categories``) do NOT reach admission decisions
    in an already-running sim today (the docstring caveat at
    interventions.py:9-11). These tests pin that they MUST (the minimal
    seam: the plugin re-reading the settings it snapshots at setup).
  * Denials are observable today through ``controller.validate``:
    ``action.denied`` records (gaworld/kernel/controller.py:95-101) and
    ``venue.redirect`` records (plugin.py:671-678).

Harness mirrors ``tests/test_venue_capacity.py`` (build_kernel +
``VenueCapacityPlugin().setup``). No LLM, no network, no real simulation.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

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
    # Keep the shops open all day and pin capacities (the loader scales hubs
    # by 1.5 — same normalization as tests/test_venue_capacity.py).
    for node in city_map["nodes"].values():
        node["open_min"], node["close_min"] = None, None
        node["capacity"] = shop_capacity if node["category"] == "commerce" else 1
    return city_map


def _agents(n, where="Home"):
    return [{"id": i, "locations": {"current": where, "home": "Home"}} for i in range(n)]


class _World:
    def __init__(self, test, *, capacity=None, traffic=None, shop_capacity=2, n=6, enabled=True):
        self.records = tempfile.mkdtemp()
        cap = {"enabled": enabled, **(capacity or {})}
        self.ctx = build_kernel(
            {
                "local_physical": {"capacity": cap},
                "traffic": traffic or {},
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


class RepresentAccountingTest(unittest.TestCase):
    """agents_represent in capacity accounting (venue_capacity.py:24-30,
    plugin.py:518-523): one simulated resident stands for N people."""

    def test_represent_scales_the_crowd_in_is_full(self):
        assert vc.is_full(0, 15, 10.0) is False, "first visitor: 1*10=10 <= 15 → room"
        assert vc.is_full(1, 15, 10.0) is True, "second visitor: 2*10=20 > 15 → full"

    def test_represent_one_fills_small_venues_slowly(self):
        assert vc.is_full(0, 2, 1.0) is False
        assert vc.is_full(1, 2, 1.0) is False
        assert vc.is_full(2, 2, 1.0) is True


class RuntimeCapacityChangeTest(unittest.TestCase):
    """Contract 1: a runtime capacity change reaches admission decisions at
    the NEXT admission attempt, without a re-run."""

    def test_node_attribute_capacity_change_bites_on_next_attempt(self):
        """The live-read path (node.get("capacity"), plugin.py:643): raising
        the node's capacity mid-run un-denies the next move. This is the
        minimal SUPPORTED mechanism — the city map is the capacity source."""
        world = _World(self, shop_capacity=1, n=4)
        world.tick()
        assert _dest(world.go(0, "ShopA"), "ShopA") == "ShopA"
        assert _dest(world.go(1, "ShopA"), "ShopA") == "ShopB", "ShopA full at 1 → redirect"
        assert _dest(world.go(2, "ShopA"), "ShopA") is None, "both shops full → denied"
        # runtime capacity change on the node itself
        world.ctx.extras["city_map"]["nodes"]["ShopA"]["capacity"] = 10
        assert _dest(world.go(2, "ShopA"), "ShopA") == "ShopA", (
            "raising the node's capacity must change the NEXT admission attempt"
        )

    def test_update_config_reaches_admission_decisions_mid_run(self):
        """The kernel's standard ``update_config`` intervention
        (gaworld/kernel/interventions.py:40-47) writes the live CONFIG dict;
        the plugin snapshot at setup (plugin.py:513-541) never re-reads it —
        so ``agents_represent`` (and the other capacity-side knobs) cannot be
        changed mid-run today. Pinned: after the intervention, the NEXT
        admission attempt must use the updated setting."""
        world = _World(self, capacity={"agents_represent": 10.0}, shop_capacity=15, n=6)
        world.tick()
        # represent=10: ShopA admits one (10 <= 15), then is full (20 > 15);
        # the redirect target ShopB takes its own one — agent 1 lands there
        assert _dest(world.go(0, "ShopA"), "ShopA") == "ShopA"
        assert _dest(world.go(1, "ShopA"), "ShopA") == "ShopB", "represent=10 fills ShopA after one visitor"
        world.ctx.controller.intervene(
            "update_config", world.ctx, path="local_physical.capacity.agents_represent", value=1.0
        )
        assert _dest(world.go(1, "ShopA"), "ShopA") == "ShopA", (
            "update_config on local_physical.capacity.agents_represent must reach "
            "admission decisions in an already-running sim (next attempt)"
        )


class DenialObservabilityTest(unittest.TestCase):
    """Contract 2: admission denials are recorded/observable."""

    def test_denied_moves_are_recorded_as_action_denied(self):
        world = _World(self, capacity={"redirect_top_k": 0}, shop_capacity=1, n=3)
        world.tick()
        world.go(0, "ShopA")  # takes the only seat
        denied = world.go(1, "ShopA")
        assert not denied.allowed
        rows = world.rows("action.denied")
        assert rows, "a denied move must land in the action.denied record table"
        row = rows[-1]
        assert row["agent_id"] == 1
        assert row["action"] == "move"
        assert "已经满了" in row["reason"], f"deny reason must explain the full venue: {row['reason']!r}"

    def test_redirects_are_recorded_as_venue_redirect(self):
        world = _World(self, shop_capacity=1, n=3)
        world.tick()
        world.go(0, "ShopA")
        verdict = world.go(1, "ShopA")
        assert _dest(verdict, "ShopA") == "ShopB", "full ShopA must redirect to the nearest open shop"
        rows = world.rows("venue.redirect")
        assert rows and rows[-1]["from"] == "ShopA" and rows[-1]["to"] == "ShopB"
        assert rows[-1]["agent_id"] == 1
        assert rows[-1]["capacity"] == 1, "redirect record carries the venue's capacity"


class UncongestedTest(unittest.TestCase):
    """Contract 3: an uncongested config produces ZERO denials."""

    def test_big_capacity_zero_denials_and_zero_redirects(self):
        world = _World(self, shop_capacity=600, n=6)
        world.tick()
        for i in range(6):
            verdict = world.go(i, "ShopA")
            assert verdict.allowed, f"agent {i} denied in an uncongested city"
            assert _dest(verdict, "ShopA") == "ShopA", "uncongested admissions never redirect"
        assert world.rows("action.denied") == []
        assert world.rows("venue.redirect") == []

    def test_represent_1_never_fills_600_cap_with_six_agents(self):
        assert all(vc.is_full(load, 600, 1.0) is False for load in range(6))
