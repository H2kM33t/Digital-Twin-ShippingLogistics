"""
Tests for the TADIF package (tadif/): the operators on their own, then the
closed loop that steers the ship.

Run from the project root:   python -m pytest tests -q
"""

import math

import numpy as np
import pytest

from routing.hazard import BLOCK_SEVERITY, CORE_FRACTION, Storm
from tadif.decision import promethee_ii, topsis
from tadif.estimation import VesselStateEstimator
from tadif.learning import KnowledgeBase, OnlineLearner
from tadif.objectives import AdaptiveWeightBuilder
from tadif.optimization import NSGA3, constrained_fronts, das_dennis
from tadif.orchestrator import TadifConfig, TadifVoyage
from tadif.risk import weighted_cvar
from tadif.triggers import Trigger, TriggerEngine

ORIGIN = (19.0760, 72.8777)
DEST = (25.276987, 55.296249)
IN_FLIGHT = Storm(19.0, 63.0, 160, 0.9, heading_deg=0, speed_kmh=12, detect_h=18)
BIG_MOVING = Storm(22.5, 64.0, 1100, 0.95, heading_deg=270, speed_kmh=15, detect_h=5)
BLOCKER = Storm(22.5, 64.0, 2500, 0.95, heading_deg=0, speed_kmh=0, detect_h=5)


def _km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _never_in_core(res):
    for f in res["frames"]:
        for s in f["storms"]:
            if s["severity"] >= BLOCK_SEVERITY:
                d = _km(tuple(f["ship"]), (s["lat"], s["lon"]))
                assert d > CORE_FRACTION * s["radius_km"] * 0.95, f["t"]


# ======================================================================
# operators
# ======================================================================

def test_cvar_is_mean_of_worst_tail():
    v = np.arange(10, dtype=float)[None]          # 0..9, equal weights
    w = np.full(10, 0.1)
    assert weighted_cvar(v, w, 0.8)[0] == pytest.approx(8.5)     # mean of 8, 9
    assert weighted_cvar(v, w, 0.0)[0] == pytest.approx(4.5)     # plain mean
    # a heavier scenario counts more
    w2 = np.r_[np.full(9, 0.05), 0.55]
    assert weighted_cvar(v, w2, 0.5)[0] == pytest.approx(9.0)


def test_das_dennis_and_fronts():
    ref = das_dennis(4, 5)
    assert len(ref) == math.comb(8, 3)
    assert np.allclose(ref.sum(1), 1.0)
    F = np.array([[1, 1], [2, 2], [0, 3], [3, 0], [1, 1]], float)
    CV = np.array([0, 0, 0, 0, 1.0])
    fronts = constrained_fronts(F, CV)
    assert set(fronts[0]) == {0, 2, 3}
    assert fronts[-1].tolist() == [4]                 # infeasible last


def test_nsga3_finds_a_spread_pareto_front():
    # genes = 6 "lanes" + 2 discrete genes; minimise (x, 1 - x) along lane 0
    def evaluate(G):
        x = np.clip(G[:, 0] / 10.0, 0, 1)
        F = np.column_stack([x, 1 - x, 0 * x, 0 * x])
        return F, np.zeros(len(G)), np.round(G, 2)

    seeds = np.zeros((4, 8))
    opt = NSGA3(n_obj=4, partitions=3, n_gen=20, seed=1)
    G, F, CV = opt.run(evaluate, seeds, n_lanes=10, levels=(2, 2))
    assert F[:, 0].max() - F[:, 0].min() > 0.3        # population spread out


def test_weights_stay_on_simplex_and_robustness_rises_when_twin_is_unsure():
    b = AdaptiveWeightBuilder("cargo_standard")
    sure = b.build_weights(0.0, 0.0, 1.0)["weights"]
    unsure = b.build_weights(0.0, 0.0, 0.3)["weights"]
    stormy = b.build_weights(0.0, 1.0, 1.0)["weights"]
    assert sum(sure.values()) == pytest.approx(1.0)
    assert sure == pytest.approx({"fuel": .30, "eta_var": .15, "emissions": .15,
                                  "safety": .20, "reliability": .15, "robustness": .05}, abs=1e-3)
    assert unsure["robustness"] > sure["robustness"]
    assert stormy["safety"] > sure["safety"]


def test_topsis_and_promethee_prefer_a_dominating_option():
    X = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.5, 3.0, 0.5]])
    w = np.array([0.4, 0.3, 0.3])
    assert int(np.argmax(topsis(X, w))) == 0
    assert int(np.argmax(promethee_ii(X, w))) == 0


def test_ukf_tracks_and_flags_a_gps_fault():
    v = (20.0, 10.0)
    good = VesselStateEstimator((0.0, 0.0), v, gps_std_km=0.5, seed=1)
    bad = VesselStateEstimator((0.0, 0.0), v, gps_std_km=0.5, gps_fault_h=5, seed=1)
    for k in range(1, 15):
        truth = (v[0] * k, v[1] * k)
        good.step(truth, k, 1.0, v)
        bad.step(truth, k, 1.0, v)
    assert good.snapshot((v[0] * 14, v[1] * 14))["error_km"] < 1.5
    assert good.confidence() > 0.8
    assert bad.confidence() < 0.3


def test_learner_converges_to_true_fuel_factor(tmp_path):
    kb = KnowledgeBase(str(tmp_path / "kb.json"))
    lr = OnlineLearner(kb)
    rng = np.random.default_rng(0)
    for _ in range(30):
        model = 1.6
        lr.observe(model * 1.12 * (1 + rng.normal(0, 0.03)), model)
    assert lr.theta == pytest.approx(1.12, abs=0.02)
    assert lr.sd < 0.02


def test_trigger_arbitration_takes_highest_severity_and_earliest_stage():
    trigs = [Trigger("weather_update", "medium", "scenario", ""),
             Trigger("plan_core_crossing", "critical", "sim", ""),
             Trigger("sensor_drift", "medium", "state", "")]
    arb = TriggerEngine.arbitrate(trigs)
    assert arb["severity"] == "critical" and arb["lead"] == "plan_core_crossing"
    assert arb["reentry"] == "state"


# ======================================================================
# closed loop
# ======================================================================

@pytest.fixture(scope="module")
def calm():
    return TadifVoyage(ORIGIN, DEST, []).run()


@pytest.fixture(scope="module")
def in_flight():
    return TadifVoyage(ORIGIN, DEST, [IN_FLIGHT]).run()


@pytest.fixture(scope="module")
def big_moving():
    return TadifVoyage(ORIGIN, DEST, [BIG_MOVING]).run()


def test_calm_voyage_sails_the_storm_free_route_on_time(calm):
    s = calm["summary"]
    assert s["arrived"] and s["on_time"]
    assert abs(s["extra_km"]) < 5
    assert all(f["status"] in ("CRUISING", "ARRIVED") for f in calm["frames"])


def test_replanning_is_event_driven_not_hourly(calm):
    s = calm["summary"]
    assert s["decision_cycles"] < s["total_hours"] / 3
    names = {x["name"] for x in calm["trigger_log"]}
    assert {"departure", "weather_update"} <= names


def test_twin_learns_the_fouled_hull_in_flight(calm):
    s = calm["summary"]
    assert s["theta_fuel"] == pytest.approx(s["theta_true"], abs=0.03)
    assert s["fuel_predicted_t"] == pytest.approx(s["fuel_used_t"], rel=0.03)


def test_storm_detection_triggers_a_twin_decided_detour(in_flight):
    s = in_flight["summary"]
    assert s["arrived"] and s["extra_km"] > 0
    assert any(x["name"] == "storm_detected" and x["t"] == 18 for x in in_flight["trigger_log"])
    dev = [e for e in in_flight["events"] if e["status"] == "DEVIATING"]
    assert dev and dev[0]["t"] == 18
    _never_in_core(in_flight)


def test_decision_output_follows_the_api_schema(in_flight):
    d = next(d for d in in_flight["decisions"] if d["t"] == 18)
    sr = d["selected_route"]
    assert set(sr["objective_vector"]) >= {"fuel_t", "eta_std_h", "co2_t", "safety_index",
                                           "reliability", "robustness_cvar"}
    ex = d["explanation"]
    assert set(ex) >= {"score", "runner_up_gap", "dominant_criteria", "weight_sensitivity",
                       "twin_confidence_at_decision"}
    assert d["n_rejected"] > 0                       # the old route was filtered out
    assert sr["objective_vector"]["robustness_cvar"] <= d["risk_thresholds"]["cvar_max"]
    lo, hi = sr["cvar_ci95"]
    assert lo <= hi


def test_big_storm_makes_the_ship_wait_then_arrive_safely(big_moving):
    statuses = [f["status"] for f in big_moving["frames"]]
    assert "HALTED" in statuses
    first = statuses.index("HALTED")
    assert any(s in ("CRUISING", "DEVIATING", "REJOINING") for s in statuses[first:])
    assert big_moving["summary"]["arrived"]
    assert "FORCED" not in statuses
    _never_in_core(big_moving)


def test_hold_limit_forces_minimum_risk_transit():
    res = TadifVoyage(ORIGIN, DEST, [BLOCKER], TadifConfig(max_halt_h=10)).run()
    statuses = [e["status"] for e in res["events"]]
    assert "HALTED" in statuses and "FORCED" in statuses
    assert res["summary"]["arrived"]
    assert res["summary"]["halted_hours"] == pytest.approx(10, abs=1.5)


def test_gps_fault_lowers_confidence_and_fires_sensor_drift():
    res = TadifVoyage(ORIGIN, DEST, [], TadifConfig(gps_fault_h=20)).run()
    assert any(x["name"] == "sensor_drift" and x["t"] > 20 for x in res["trigger_log"])
    after = [f["twin"]["subscores"]["sensor"] for f in res["frames"] if 24 <= f["t"] <= 30]
    assert max(after) < 0.5
    assert res["summary"]["arrived"]


def test_knowledge_base_carries_learning_to_the_next_voyage(tmp_path):
    path = str(tmp_path / "kb.json")
    first = TadifVoyage(ORIGIN, DEST, [], TadifConfig(kb_path=path)).run()
    second = TadifVoyage(ORIGIN, DEST, [], TadifConfig(kb_path=path)).run()
    assert first["summary"]["theta_prior"] == pytest.approx(1.0)
    assert second["summary"]["theta_prior"] == pytest.approx(1.12, abs=0.03)

    def fuel_triggers(r):
        return sum(x["name"] == "fuel_prediction_error" for x in r["trigger_log"])
    assert fuel_triggers(second) < fuel_triggers(first)
    assert len(KnowledgeBase(path).data["voyages"]) == 2


def test_read_only_run_does_not_write_the_knowledge_base(tmp_path):
    path = str(tmp_path / "kb.json")
    TadifVoyage(ORIGIN, DEST, [], TadifConfig(kb_path=path, kb_write=False)).run()
    assert KnowledgeBase(path).data["voyages"] == []
