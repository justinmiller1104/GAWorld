"""Phase 2.8 RED tests — venue.admission attempt records (TASK-3675).

Spec (phase28-tdd.md §1): every capped-venue ``move`` request must emit
exactly one ``venue.admission`` row with an explicit ``outcome`` of
``admitted`` / ``redirected`` / ``denied`` — including the success path,
which today returns ``None`` silently (gaworld/world/plugin.py:672-673).
Existing outcome tables ``venue.redirect`` (plugin.py:684-690) and
``action.denied`` (gaworld/kernel/controller.py:94-101) must stay untouched.

Semantics pinned here (per the admissions investigation, RESULT-phase28-
admissions §5 option (a)): records are PER REQUEST — a retried intent
produces one row per attempt; a single accepted request never duplicates.

Harness mirrors ``tests/test_mcf_phase26_venue_capacity.py`` (build_kernel +
``VenueCapacityPlugin().setup`` on a minimal fake city map). No LLM, no
network, no real simulation.

RED expectation: ``venue.admission`` does not exist yet — every test below
fails because zero rows are written, not because of any syntax error.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from gaworld.kernel import ActionRequest, build_kernel
from gaworld.world import city_map as cm
from gaworld.world.plugin import VenueCapacityPlugin


def _map(shop_capacity=1):
    """Three commerce venues so redirects have somewhere to land."""
    content = (
        "# City Map\n"
        "@node: Home | kind=hub | category=residential | x=0.0 | y=0.0 | capacity=1\n"
        f"@node: ShopA | kind=hub | category=commerce | x=1.0 | y=0.0 | capacity={shop_capacity}\n"
        f"@node: ShopB | kind=hub | category=commerce | x=2.0 | y=0.0 | capacity={shop_capacity}\n"
        f"@node: ShopC | kind=hub | category=commerce | x=3.0 | y=0.0 | capacity={shop_capacity}\n"
        "\n- City: Demo\n  - Hub: Home\n  - Hub: ShopA\n  - Hub: ShopB\n  - Hub: ShopC\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        city_map = cm.load_city_map(path)
    for node in city_map["nodes"].values():
        node["open_min"], node["close_min"] = None, None  # always open
        node["capacity"] = shop_capacity if node["category"] == "commerce" else 1
    return city_map


def _agents(n, where="Home"):
    return [{"id": i, "locations": {"current": where, "home": "Home"}} for i in range(n)]


class _World:
    """Minimal kernel with VenueCapacityPlugin, mirroring the Phase 2.6 harness."""

    def __init__(self, *, shop_capacity=1, n=8, agents_represent=1.0):
        self.records = tempfile.mkdtemp()
        self.ctx = build_kernel(
            {
                "local_physical": {"capacity": {"enabled": True, "agents_represent": agents_represent}},
                "random_seed": 7,
                "records": {"output_dir": self.records},
            },
            load_entry_points=False,
        )
        self.ctx.extras["city_map"] = _map(shop_capacity)
        VenueCapacityPlugin().setup(self.ctx)
        self.agents = _agents(n)
        self.ctx.set_agents(self.agents)
        self.ctx.clock.advance("12:00", 0)
        self.ctx.bus.emit(
            "on_time_tick",
            day=3,
            time_str="12:00",
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


def _admitted_to(verdict, asked):
    if not verdict.allowed:
        return None
    return verdict.rewritten.params["to"] if verdict.rewritten is not None else asked


class SuccessfulAdmissionRecorded(unittest.TestCase):
    """Spec 1a: a successful admission writes exactly one outcome="admitted" row."""

    def test_admitted_move_writes_one_row(self):
        w = _World(shop_capacity=2)
        v = w.go(0, "ShopA")
        assert v.allowed, "capacity 2, nobody inside: first move must be admitted"
        rows = w.rows("venue.admission")
        assert len(rows) == 1, (
            f"expected exactly one venue.admission row for one admitted request, got {len(rows)} "
            "(the success path at plugin.py:672-673 currently records nothing)"
        )
        assert rows[0]["outcome"] == "admitted", rows[0]

    def test_no_duplicate_row_for_a_single_request(self):
        w = _World(shop_capacity=2)
        w.go(0, "ShopA")
        rows = w.rows("venue.admission")
        assert len(rows) == 1, f"one request → one row, got {len(rows)}"


class DeniedAdmissionRecorded(unittest.TestCase):
    """Spec 1b: a denied admission writes exactly one outcome="denied" row."""

    @staticmethod
    def _fill_all_shops(w):
        # capacity 1 each: agent 0 admitted at ShopA; 1,2 redirected to B, C;
        # afterwards every commerce venue is full.
        assert _admitted_to(w.go(0, "ShopA"), "ShopA") == "ShopA"
        assert _admitted_to(w.go(1, "ShopA"), "ShopA") == "ShopB"
        assert _admitted_to(w.go(2, "ShopA"), "ShopA") == "ShopC"

    def test_denied_move_writes_one_row(self):
        w = _World(shop_capacity=1)
        self._fill_all_shops(w)
        v = w.go(3, "ShopA")
        assert not v.allowed, "every commerce venue full: move must be denied"
        rows = w.rows("venue.admission")
        assert len(rows) == 4, f"four attempts → four rows, got {len(rows)}"
        assert rows[-1]["outcome"] == "denied", rows[-1]


class RedirectedAdmissionRecorded(unittest.TestCase):
    """Spec 1c: a redirect writes one outcome="redirected" row carrying the
    alternative destination plus its capacity and occupancy."""

    def test_redirect_writes_row_with_destination_and_capacity(self):
        w = _World(shop_capacity=1)
        assert _admitted_to(w.go(0, "ShopA"), "ShopA") == "ShopA"  # ShopA full
        v = w.go(1, "ShopA")
        assert v.allowed and v.rewritten is not None, "full ShopA must redirect, not deny"
        assert v.rewritten.params["to"] == "ShopB", v.rewritten.params
        rows = w.rows("venue.admission")
        assert len(rows) == 2, f"two attempts → two rows, got {len(rows)}"
        row = rows[-1]
        assert row["outcome"] == "redirected", row
        assert row["to"] == "ShopB", f"destination must be the alternative venue, got {row}"
        assert "capacity" in row and "occupancy" in row, row


class RetrySemantics(unittest.TestCase):
    """Spec 1d: one row PER ATTEMPT (per-request semantics, documented);
    a retried intent after a denial is a new row, never deduped away."""

    def test_retry_after_denial_records_each_attempt(self):
        w = _World(shop_capacity=1)
        DeniedAdmissionRecorded._fill_all_shops(w)
        v1 = w.go(3, "ShopA")
        assert not v1.allowed, "all venues full: first attempt must be denied"
        v2 = w.go(3, "ShopA")  # same agent, same venue, retried
        assert not v2.allowed
        denied = [r for r in w.rows("venue.admission") if r.get("outcome") == "denied"]
        assert len(denied) == 2, (
            f"per-request semantics: each retry is its own attempt row, got {len(denied)} denied rows"
        )


class RequiredFields(unittest.TestCase):
    """Spec 1e: every row carries agent_id, destination venue id, capacity,
    current occupancy, agents_represent, and the recorder's _day/_time stamps."""

    REQUIRED = ("agent_id", "to", "capacity", "occupancy", "agents_represent")

    def test_every_row_has_required_fields(self):
        w = _World(shop_capacity=1, agents_represent=1.0)
        w.go(0, "ShopA")  # admitted
        w.go(1, "ShopA")  # redirected to ShopB
        w.go(2, "ShopA")  # redirected or denied (B full by agent 1's redirect)
        w.go(3, "ShopA")  # another attempt
        rows = w.rows("venue.admission")
        assert rows, "attempts were made; venue.admission must exist"
        for row in rows:
            for field in self.REQUIRED:
                assert field in row, f"venue.admission row missing {field!r}: {row}"
            assert "_day" in row and "_time" in row, f"recorder stamps missing: {row}"

    def test_agents_represent_reflects_config(self):
        w = _World(shop_capacity=3, agents_represent=5.0)
        w.go(0, "ShopA")
        rows = w.rows("venue.admission")
        assert rows and rows[0]["agents_represent"] == 5.0, rows


class LegacyTablesUntouched(unittest.TestCase):
    """Spec 2 (admission half): the new attempt table must not replace the
    existing outcome tables — venue.redirect and action.denied keep firing."""

    def test_redirect_still_writes_venue_redirect(self):
        w = _World(shop_capacity=1)
        w.go(0, "ShopA")
        w.go(1, "ShopA")
        redirects = w.rows("venue.redirect")
        assert len(redirects) == 1, f"venue.redirect must be untouched, got {redirects}"

    def test_denial_still_writes_action_denied(self):
        w = _World(shop_capacity=1)
        DeniedAdmissionRecorded._fill_all_shops(w)
        w.go(3, "ShopA")  # every venue full → denial
        denials = w.rows("action.denied")
        assert len(denials) == 1, f"action.denied must be untouched, got {denials}"


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
