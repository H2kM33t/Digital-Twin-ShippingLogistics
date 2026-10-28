"""
Storm rerouting tests for the TwinRoute-M graph engine.

Run from the project root:   python -m pytest tests -q
"""

import math

import pytest

from routing.hazard import BLOCK_SEVERITY, CORE_FRACTION, Storm
from routing.planner import SearchAlgorithm
from routing.voyage import VoyageConfig, VoyageSimulator

ORIGIN = (19.0760, 72.8777)          # Mumbai
DEST = (25.276987, 55.296249)        # Dubai
SPEED_KMH = 16 * 1.852


def _km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


@pytest.fixture()
def wide_generator():
    """Graph wide enough to route around a mid-voyage storm."""
    sim = VoyageSimulator(ORIGIN, DEST, [Storm(22.3, 64.0, 150, 0.9)])
    return sim.generator


def _route_nodes(gen, algorithm=SearchAlgorithm.ASTAR):
    return gen.replan_from(gen.start_node, algorithm).node_ids


def test_astar_matches_dijkstra_without_storm(wide_generator):
    gen = wide_generator
    gen.clear_storms()
    a = gen.replan_from(gen.start_node, SearchAlgorithm.ASTAR)
    d = gen.replan_from(gen.start_node, SearchAlgorithm.DIJKSTRA)
    assert a.node_ids == d.node_ids
    assert a.total_cost == pytest.approx(d.total_cost)


def test_storm_changes_route_and_avoids_core(wide_generator):
    gen = wide_generator
    gen.clear_storms()
    before = _route_nodes(gen)

    storm = Storm(22.3, 64.0, radius_km=150, severity=0.9)
    gen.set_storms([storm])
    snap = gen.apply_storms(0.0, None, SPEED_KMH)
    assert snap.blocked_edges > 0

    after = gen.replan_from(gen.start_node)
    assert after.node_ids != before

    # no waypoint inside the blocked core
    cx, cy = gen.frame.to_xy(storm.lat, storm.lon)
    for lat, lon in after.coordinates:
        x, y = gen.frame.to_xy(lat, lon)
        assert math.hypot(x - cx, y - cy) > CORE_FRACTION * storm.radius_km

    # the detour costs more than the clear-weather route
    assert after.total_distance > gen.generate_route().total_distance - 1e-6


def test_astar_matches_dijkstra_with_storm(wide_generator):
    gen = wide_generator
    gen.set_storms([Storm(22.3, 64.0, 150, 0.9)])
    gen.apply_storms(0.0, None, SPEED_KMH)
    a = gen.replan_from(gen.start_node, SearchAlgorithm.ASTAR)
    d = gen.replan_from(gen.start_node, SearchAlgorithm.DIJKSTRA)
    assert a.node_ids == d.node_ids
    assert a.total_cost == pytest.approx(d.total_cost)


def test_clearing_storm_restores_original_route(wide_generator):
    gen = wide_generator
    gen.clear_storms()
    before = _route_nodes(gen)

    gen.set_storms([Storm(22.3, 64.0, 150, 0.9)])
    gen.apply_storms(0.0, None, SPEED_KMH)
    assert _route_nodes(gen) != before

    gen.clear_storms()
    assert gen.graph.number_of_blocked_edges == 0
    assert _route_nodes(gen) == before


def test_low_severity_storm_penalises_but_does_not_block(wide_generator):
    gen = wide_generator
    gen.set_storms([Storm(22.3, 64.0, 150, BLOCK_SEVERITY - 0.2)])
    snap = gen.apply_storms(0.0, None, SPEED_KMH)
    assert snap.blocked_edges == 0
    assert snap.edge_risk                      # but edges carry risk cost


def test_undetected_storm_is_ignored_until_detected(wide_generator):
    gen = wide_generator
    gen.set_storms([Storm(22.3, 64.0, 150, 0.9, detect_h=10)])
    assert gen.apply_storms(5.0, None, SPEED_KMH).edge_risk == {}
    assert gen.apply_storms(10.0, None, SPEED_KMH).edge_risk


def test_ship_deviates_in_flight_and_arrives():
    storm = Storm(19.0, 63.0, 160, 0.9, heading_deg=0, speed_kmh=12, detect_h=18)
    res = VoyageSimulator(ORIGIN, DEST, [storm]).run()

    statuses = [e["status"] for e in res["events"]]
    assert "DEVIATING" in statuses
    assert res["summary"]["arrived"]
    assert res["summary"]["extra_km"] > 0


def test_ship_never_sails_into_storm_core():
    storm = Storm(19.0, 63.0, 160, 0.9, heading_deg=0, speed_kmh=12, detect_h=18)
    res = VoyageSimulator(ORIGIN, DEST, [storm]).run()

    for f in res["frames"]:
        for s in f["storms"]:
            if s["severity"] >= BLOCK_SEVERITY:
                d = _km(tuple(f["ship"]), (s["lat"], s["lon"]))
                assert d > CORE_FRACTION * s["radius_km"] * 0.95


def test_ship_halts_then_resumes_when_storm_clears():
    # Storm too big to go around (corridor is capped) but moving off the route.
    storm = Storm(22.5, 64.0, 1100, 0.95, heading_deg=270, speed_kmh=15, detect_h=5)
    res = VoyageSimulator(ORIGIN, DEST, [storm]).run()

    statuses = [f["status"] for f in res["frames"]]
    assert "HALTED" in statuses
    first_halt = statuses.index("HALTED")
    assert any(s in ("CRUISING", "DEVIATING", "REJOINING") for s in statuses[first_halt:])
    assert res["summary"]["arrived"]
    assert res["summary"]["halted_hours"] > 0
    assert "FORCED" not in statuses          # it resumed on its own


def test_status_does_not_flap_on_a_detour():
    storm = Storm(22.5, 64.0, 1100, 0.95, heading_deg=270, speed_kmh=15, detect_h=5)
    res = VoyageSimulator(ORIGIN, DEST, [storm]).run()
    assert len(res["events"]) <= 8


def test_halt_limit_forces_minimum_risk_transit():
    # Stationary storm that blocks every lane: ship holds, then is forced.
    storm = Storm(22.5, 64.0, 2500, 0.95, heading_deg=0, speed_kmh=0, detect_h=5)
    cfg = VoyageConfig(max_halt_h=10)
    res = VoyageSimulator(ORIGIN, DEST, [storm], cfg).run()

    statuses = [e["status"] for e in res["events"]]
    assert "HALTED" in statuses and "FORCED" in statuses
    assert res["summary"]["arrived"]
    assert res["summary"]["halted_hours"] == pytest.approx(10, abs=1.5)


def test_no_storm_follows_baseline():
    res = VoyageSimulator(ORIGIN, DEST, []).run()
    assert res["summary"]["arrived"]
    assert abs(res["summary"]["extra_km"]) < 5
    assert all(f["status"] in ("CRUISING", "ARRIVED") for f in res["frames"])
