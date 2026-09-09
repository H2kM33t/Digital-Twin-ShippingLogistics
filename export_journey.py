"""
Export a voyage journey to JSON for the twinroute_storm_demo.html front end
to consume.

This runs the real pipeline (digital_twin_generator -> route_generator ->
simulator -> objective_builder -> optimizer -> twin_health) forward over a
voyage, exactly like adaptive_demo.run_storm_journey(), but instead of
printing to console it records one JSON record per checkpoint:

  {
    "frac":              0.0-1.0 fraction of the voyage (maps directly onto
                         the HTML's animation timeline),
    "cycle_index":       int,
    "wind_kt", "wave_m", "visibility_km": current environment readings,
    "storm_active":      bool,
    "storm_severity":    0-1 (0 when no storm),
    "storm_point":       [lat, lon] once the storm has formed, else null,
    "routes":            [[[lat,lon], ...], ...] -- the ACTUAL candidate
                         route waypoints returned by generate_candidate_routes()
                         at this checkpoint, one polyline per candidate route,
    "selected_route":    1-based index into "routes",
    "route_changed":     true the first time the recommendation flips,
    "fuel_t", "time_h", "risk": performance of the selected route,
    "routes_performance": [{"fuel","time","risk"}, ...] -- simulator.py's
                         output for EVERY candidate route this checkpoint,
                         same order as "routes", so the UI can show a real
                         side-by-side comparison instead of only the winner,
    "pareto_indices":    1-based indices of Pareto-optimal candidates,
    "twin_confidence":   0-1 scalar from twin_health.py,
    "twin_band":         "NOMINAL" | "ADVISORY" | "CRITICAL",
    "twin_subscores":    {"sensor", "forecast", "model", "decision"}
  }

Output: journey.json, written next to this script (same folder as the
HTML file, so a plain `fetch('journey.json')` from the page finds it).
"""

import json

from models import (
    NavigationState, PropulsionState, EnergyState, VesselState,
    WeatherState, OceanState, EnvironmentState,
    MissionState, DigitalTwin
)
from route_generator import generate_candidate_routes, generate_longhaul_routes, SECURITY_RISK_ZONE
from simulator import simulate_all_routes
from optimizer import select_best_route
from objective_builder import build_adaptive_weights
from twin_health import compute_twin_health
from digital_twin_generator import DigitalTwinSimulator, VesselConfig, MissionConfig, SimulationConfig


def twin_state_to_digital_twin(state, origin_coords, destination_coords, max_speed=18.0, fuel_budget=300.0):
    v = state["vessel"]
    e = state["environment"]

    nav = NavigationState(lat=v["latitude"], lon=v["longitude"], speed=v["speed_knots"], heading=v["heading_deg"])
    prop = PropulsionState(power_kw=5000, rpm=v["rpm"], efficiency=0.85)
    energy = EnergyState(fuel_remaining=v["fuel_remaining_tonnes"])
    vessel = VesselState(navigation=nav, propulsion=prop, energy=energy)

    weather = WeatherState(
        wind_speed=e["wind_speed_knots"],
        wind_dir=e["wind_direction_deg"],
        air_temp=e["air_temperature_c"],
        visibility=e["visibility_km"],
    )
    ocean = OceanState(
        current_speed=e["ocean_current_speed_knots"],
        current_dir=e["ocean_current_direction_deg"],
        wave_height=e["wave_height_m"],
        wave_period=e["wave_period_s"],
    )
    env = EnvironmentState(weather=weather, ocean=ocean)
    mission = MissionState(origin=origin_coords, destination=destination_coords,
                            max_speed=max_speed, fuel_budget=fuel_budget)
    return DigitalTwin(vessel=vessel, environment=env, mission=mission)


def export_journey(origin="Mumbai", destination="Dubai",
                    total_steps=60, storm_start_step=18, checkpoints=50,
                    out_path="journey.json"):
    sim = DigitalTwinSimulator(
        vessel=VesselConfig(),
        mission=MissionConfig(origin=origin, destination=destination),
        simulation=SimulationConfig(scenario="normal", seed=42),
    )

    storm_point = None
    twin_history = []
    last_best_index = None
    records = []

    checkpoint_every = max(1, total_steps // checkpoints)

    for step in range(1, total_steps + 1):
        if step >= storm_start_step:
            sim.sim_cfg.scenario = "extreme_weather"
            sim.sim_cfg.anomaly_severity = min(1.0, (step - storm_start_step) / 12)
        else:
            sim.sim_cfg.scenario = "normal"

        state = sim.step(dt=1.0)

        if step % checkpoint_every != 0:
            continue

        twin = twin_state_to_digital_twin(
            state,
            origin_coords=state["mission"]["origin_coordinates"],
            destination_coords=state["mission"]["destination_coordinates"],
        )

        storm_active = sim.sim_cfg.scenario == "extreme_weather"
        storm_severity = sim.sim_cfg.anomaly_severity if storm_active else 0.0

        if storm_active and storm_point is None:
            lat1, lon1 = twin.mission.origin
            lat2, lon2 = twin.mission.destination
            storm_point = ((lat1 + lat2) / 2, (lon1 + lon2) / 2)

        routes = generate_candidate_routes(
            twin.mission.origin, twin.mission.destination,
            storm_point=storm_point, storm_severity=storm_severity
        )
        performance = simulate_all_routes(routes, twin, storm_point=storm_point, storm_severity=storm_severity)
        weights = build_adaptive_weights(twin)
        best_index, report = select_best_route(performance, weights=weights)
        health = compute_twin_health(twin, performance, report, history=twin_history[-3:])
        twin_history.append(twin)

        route_changed = last_best_index is not None and best_index != last_best_index
        last_best_index = best_index

        perf = performance[best_index]
        records.append({
            "frac": round(step / total_steps, 4),
            "cycle_index": step,
            "wind_kt": round(twin.environment.weather.wind_speed, 1),
            "wave_m": round(twin.environment.ocean.wave_height, 1),
            "visibility_km": round(twin.environment.weather.visibility, 1),
            "storm_active": storm_active,
            "storm_severity": round(storm_severity, 3),
            "storm_point": [round(storm_point[0], 3), round(storm_point[1], 3)] if storm_point else None,
            "routes": [
                [[round(wp.lat, 4), round(wp.lon, 4)] for wp in route.waypoints]
                for route in routes
            ],
            "selected_route": best_index + 1,
            "route_changed": route_changed,
            "fuel_t": perf["fuel"],
            "time_h": perf["time"],
            "risk": perf["risk"],
            "routes_performance": [
                {"fuel": p["fuel"], "time": p["time"], "risk": p["risk"]}
                for p in performance
            ],
            "pareto_indices": [i + 1 for i in report["pareto_indices"]],
            "twin_confidence": health.confidence,
            "twin_band": health.band,
            "twin_subscores": health.subscores,
        })

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} checkpoints to {out_path}")
    return records


# ---------------------------------------------------------------------
# Suez vs Cape of Good Hope: real-pipeline export for
# twinroute_ecdis_cape_vs_suez_demo.html
# ---------------------------------------------------------------------
# Same checkpoint-loop shape as export_journey() above, but with two
# differences that match how this scenario actually works:
#
#   1. Candidates come from generate_longhaul_routes() -- two FIXED
#      real-world geometries (Suez, Cape), not deformed straight lines.
#      Which one wins is entirely a simulator/optimizer decision, never
#      a route-shape decision.
#   2. Risk comes from route_security_exposure() (a region a route does
#      or doesn't pass through) rather than route_storm_exposure() (a
#      point + falloff radius). severity ramps up the same way
#      anomaly_severity does in export_journey(), which is what gives
#      the eventual recommendation flip a natural, non-scripted trigger
#      point instead of a hand-picked "phase".
#
# Environment/vessel state here is a plausible calm-to-moderate baseline
# (this scenario isn't about weather) rather than digital_twin_generator's
# full physics stepper, since the story is a security advisory, not a
# storm -- but it's built from the same DigitalTwin dataclasses so it
# flows through simulate_all_routes / objective_builder / optimizer /
# twin_health exactly like the storm export does.

def _longhaul_twin_state(step, total_steps, fuel_budget):
    """A calm-to-moderate synthetic environment/vessel snapshot for one
    checkpoint of the Suez/Cape voyage. Mild variation only, since this
    scenario's risk driver is the security corridor, not weather."""
    progress = step / total_steps

    nav = NavigationState(lat=0.0, lon=0.0, speed=20.0, heading=90.0)
    prop = PropulsionState(power_kw=18000, rpm=90, efficiency=0.85)
    # Fuel burns down steadily over the voyage, but gently: fuel_budget
    # is set well above simulator.py's worst-case route consumption
    # (~5000t for the longer Cape route), and the depletion floor is
    # kept high enough that fuel_pressure never grows large enough to
    # out-vote a saturated security-risk signal late in the voyage --
    # a voyage that's nowhere near running out of fuel shouldn't cause
    # the recommendation to flip back once a real advisory is active.
    fuel_remaining = fuel_budget * max(0.5, 1 - 0.35 * progress)
    energy = EnergyState(fuel_remaining=fuel_remaining)
    vessel = VesselState(navigation=nav, propulsion=prop, energy=energy)

    weather = WeatherState(wind_speed=12 + 4 * progress, wind_dir=200, visibility=14.0)
    ocean = OceanState(current_speed=1.2, current_dir=180, wave_height=1.5 + 0.5 * progress)
    env = EnvironmentState(weather=weather, ocean=ocean)

    mission = MissionState(origin=(51.9, 4.5), destination=(1.3, 103.8),
                            max_speed=22.0, fuel_budget=fuel_budget, progress=progress)
    return DigitalTwin(vessel=vessel, environment=env, mission=mission)


def export_longhaul_journey(total_steps=60, security_risk_start_step=14,
                             security_risk_ramp_steps=10, checkpoints=50,
                             fuel_budget=6500.0, out_path="journey_suez_cape.json"):
    """
    Run the real pipeline (route_generator -> simulator -> objective_builder
    -> optimizer -> twin_health) forward over the Rotterdam -> Singapore
    voyage, choosing between the fixed Suez and Cape route geometries at
    every checkpoint as a security-risk advisory in the Red Sea corridor
    ramps up. Writes journey_suez_cape.json in the same record shape as
    export_journey(), so twinroute_ecdis_cape_vs_suez_demo.html can
    fetch() it exactly like the storm demo fetches journey.json.
    """
    routes = generate_longhaul_routes()  # [Suez, Cape]

    twin_history = []
    last_best_index = None
    records = []
    checkpoint_every = max(1, total_steps // checkpoints)

    for step in range(1, total_steps + 1):
        if step >= security_risk_start_step:
            severity = min(1.0, (step - security_risk_start_step) / security_risk_ramp_steps)
        else:
            severity = 0.0

        if step % checkpoint_every != 0:
            continue

        twin = _longhaul_twin_state(step, total_steps, fuel_budget)

        performance = simulate_all_routes(
            routes, twin,
            security_risk_zone=SECURITY_RISK_ZONE, security_risk_severity=severity
        )
        # beta_risk raised above objective_builder's 0.6 default for this
        # scenario: a security/geopolitical advisory is categorically
        # different from routine weather drift (it doesn't fluctuate
        # checkpoint-to-checkpoint the way wind/waves do), so once it's
        # active it should dominate decisively rather than leave a
        # narrow TOPSIS margin that minor weather noise could flip back.
        weights = build_adaptive_weights(twin, extra_risk_pressure=severity, beta_risk=1.5)
        best_index, report = select_best_route(performance, weights=weights)
        health = compute_twin_health(twin, performance, report, history=twin_history[-3:])
        twin_history.append(twin)

        route_changed = last_best_index is not None and best_index != last_best_index
        last_best_index = best_index

        perf = performance[best_index]
        records.append({
            "frac": round(step / total_steps, 4),
            "cycle_index": step,
            "security_risk_active": severity > 0,
            "security_risk_severity": round(severity, 3),
            "routes": [
                [[round(wp.lat, 4), round(wp.lon, 4)] for wp in route.waypoints]
                for route in routes
            ],
            "route_names": ["Suez", "Cape of Good Hope"],
            "selected_route": best_index + 1,
            "route_changed": route_changed,
            "fuel_t": perf["fuel"],
            "time_h": perf["time"],
            "risk": perf["risk"],
            "pareto_indices": [i + 1 for i in report["pareto_indices"]],
            "twin_confidence": health.confidence,
            "twin_band": health.band,
            "twin_subscores": health.subscores,
        })

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} checkpoints to {out_path}")
    return records


if __name__ == "__main__":
    export_journey()
    export_longhaul_journey()
