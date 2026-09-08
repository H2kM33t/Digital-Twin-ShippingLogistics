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
from route_generator import generate_candidate_routes
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
