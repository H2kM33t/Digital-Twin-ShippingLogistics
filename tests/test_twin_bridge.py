"""Tests for the Digital Twin pipeline running on top of the storm voyage."""

from routing.hazard import Storm
from routing.voyage import VoyageSimulator
from simulator import simulate_route
from models import Route, Waypoint
from twin_bridge import TwinObserver, storm_proximity

ORIGIN = (19.0760, 72.8777)
DEST = (25.276987, 55.296249)
STORM = Storm(19.0, 63.0, 160, 0.9, heading_deg=0, speed_kmh=12, detect_h=18)


def _run(storms=(STORM,)):
    return VoyageSimulator(ORIGIN, DEST, list(storms)).run(
        observer=TwinObserver(ORIGIN, DEST))


def test_every_frame_carries_twin_data():
    res = _run()
    for f in res["frames"][:-1]:
        assert set(f["twin"]["subscores"]) == {"sensor", "forecast", "model", "decision"}
        assert 0.0 <= f["twin"]["confidence"] <= 1.0
        assert f["twin"]["band"] in ("NOMINAL", "ADVISORY", "CRITICAL")
        assert len(f["routes_performance"]) == len(f["candidates"]) == 3
        assert f["twin_pick"] in range(3)


def test_ignoring_the_storm_is_scored_riskier_than_the_detour():
    res = _run()
    after = [f for f in res["frames"] if 19 <= f["t"] <= 30]
    assert after
    for f in after:
        plan, direct, safest = f["routes_performance"]
        assert direct["risk"] > plan["risk"]
        assert direct["risk"] >= safest["risk"]


def test_twin_never_recommends_the_storm_ignoring_route_while_storm_blocks_it():
    res = _run()
    for f in res["frames"]:
        if f["status"] == "DEVIATING" and f["t"] > 18 and f["routes_performance"][1]["risk"] >= 0.99:
            assert f["twin_pick"] != 1


def test_environment_rises_near_storm_and_relaxes_after():
    res = _run()
    wind = [f["twin"]["wind_kt"] for f in res["frames"][:-1]]
    assert max(wind) > 20          # storm periphery felt
    assert wind[0] < 15 and wind[-1] < 15


def test_storm_proximity_scales_with_distance():
    s = [{"lat": 22.0, "lon": 64.0, "radius_km": 100, "severity": 1.0}]
    assert storm_proximity((22.0, 64.0), s) > storm_proximity((23.0, 64.0), s) > 0
    assert storm_proximity((40.0, 64.0), s) == 0


def test_simulator_exposure_override_matches_old_behaviour_by_default():
    obs = TwinObserver(ORIGIN, DEST)
    ctx = {"t": 0, "pos": ORIGIN, "status": "CRUISING", "ship_risk": 0.0,
           "storms": [], "speed_kts": 16,
           "candidates": [{"name": "x", "kind": "plan", "peak_risk": 0.0,
                           "coords": [ORIGIN, DEST]}]}
    twin_out = obs(ctx)
    assert twin_out["routes_performance"][0]["risk"] < 0.5

    # exposure override raises risk; None keeps the legacy single-point path
    route = Route(waypoints=[Waypoint(*ORIGIN), Waypoint(*DEST)])
    twin = _make_twin(obs)
    base = simulate_route(route, twin)["risk"]
    hot = simulate_route(route, twin, storm_exposure=1.0, storm_severity=1.0)["risk"]
    assert hot > base


def _make_twin(obs):
    from twin_bridge import _twin_from_state
    return _twin_from_state(obs.sim.step(dt=1.0), ORIGIN, DEST, ORIGIN, 16)
