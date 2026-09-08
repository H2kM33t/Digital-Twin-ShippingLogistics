from models import (
    NavigationState, PropulsionState, EnergyState, VesselState,
    WeatherState, OceanState, EnvironmentState,
    MissionState, DigitalTwin
)
from route_generator import generate_candidate_routes
from simulator import simulate_all_routes
from optimizer import select_best_route
from objective_builder import build_adaptive_weights, compute_pressures, explain_weights, BASE_WEIGHTS
from visualizer import plot_routes
from digital_twin_generator import DigitalTwinSimulator, VesselConfig, MissionConfig, SimulationConfig
from twin_health import compute_twin_health


def generate_stepped_state(origin, destination, scenario="normal", steps=10, seed=42, anomaly_severity=0.9):
    """Run the simulator forward so scenario effects (e.g. storms) actually apply."""
    sim = DigitalTwinSimulator(
        vessel=VesselConfig(),
        mission=MissionConfig(origin=origin, destination=destination),
        simulation=SimulationConfig(scenario=scenario, seed=seed, anomaly_severity=anomaly_severity),
    )
    state = None
    for _ in range(steps):
        state = sim.step(dt=1.0)
    return state


def twin_state_to_digital_twin(state, origin_coords, destination_coords, max_speed=18.0, fuel_budget=300.0):
    """Convert the generator's raw state dict into a models.py DigitalTwin object."""
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


def print_report(label, performance, best_index, report, adaptive_weights, pressures, health=None):
    print(f"\n{'='*50}")
    print(label)
    print('='*50)
    for i, perf in enumerate(performance):
        print(f"Route {i+1}: Fuel={perf['fuel']}t  Time={perf['time']}h  Risk={perf['risk']}")
    print(explain_weights(BASE_WEIGHTS, adaptive_weights, pressures))
    print(f"\nPareto-optimal routes: {[i+1 for i in report['pareto_indices']]}")
    print(f">>> Recommended Route: Route {best_index + 1}")
    if health is not None:
        print(f"\nTwin Confidence: {health.confidence}  [{health.band}]")
        print(f"  sub-scores: {health.subscores}")
        if health.band != "NOMINAL":
            print(f"  >>> {health.escalation}")


def run_before_after_comparison():
    origin = "Mumbai"
    destination = "Dubai"

    # ---- BEFORE: normal conditions ----
    state_normal = generate_stepped_state(origin, destination, scenario="normal", steps=10)
    twin_normal = twin_state_to_digital_twin(
        state_normal,
        state_normal["mission"]["origin_coordinates"],
        state_normal["mission"]["destination_coordinates"],
    )

    routes_before = generate_candidate_routes(twin_normal.mission.origin, twin_normal.mission.destination)

    performance_before = simulate_all_routes(routes_before, twin_normal)
    weights_before = build_adaptive_weights(twin_normal)
    pressures_before = compute_pressures(twin_normal)
    best_before, report_before = select_best_route(performance_before, weights=weights_before)
    health_before = compute_twin_health(twin_normal, performance_before, report_before)
    print_report("BEFORE: Normal Conditions", performance_before, best_before, report_before, weights_before, pressures_before, health_before)

    # ---- AFTER: storm conditions ----
    state_storm = generate_stepped_state(origin, destination, scenario="extreme_weather", steps=10, anomaly_severity=0.9)
    twin_storm = twin_state_to_digital_twin(
        state_storm,
        state_storm["mission"]["origin_coordinates"],
        state_storm["mission"]["destination_coordinates"],
    )

    # Storm route: generated dynamically, curving around the storm's location
    lat1, lon1 = twin_storm.mission.origin
    lat2, lon2 = twin_storm.mission.destination
    storm_point = ((lat1 + lat2) / 2, (lon1 + lon2) / 2)

    routes_after = generate_candidate_routes(
        twin_storm.mission.origin,
        twin_storm.mission.destination,
        storm_point=storm_point,
        storm_severity=0.9
    )

    performance_after = simulate_all_routes(routes_after, twin_storm, storm_point=storm_point, storm_severity=0.9)
    weights_after = build_adaptive_weights(twin_storm)
    pressures_after = compute_pressures(twin_storm)
    best_after, report_after = select_best_route(performance_after, weights=weights_after)
    health_after = compute_twin_health(twin_storm, performance_after, report_after, history=[twin_normal])
    print_report("AFTER: Storm Detected (extreme_weather)", performance_after, best_after, report_after, weights_after, pressures_after, health_after)

    # ---- Compare ----
    print(f"\n{'='*50}")
    if best_before != best_after:
        print(f"ROUTE CHANGED: Route {best_before+1} -> Route {best_after+1}")
        print("The Digital Twin adapted its recommendation due to changing conditions.")
    else:
        print(f"Route unchanged: Route {best_before+1} remains optimal even with the storm.")
    print('='*50)

    # ---- Visualize ----
    print("\nShowing BEFORE plot...")
    plot_routes(
        routes_before,
        vessel_position=(twin_normal.vessel.navigation.lat, twin_normal.vessel.navigation.lon),
        origin=twin_normal.mission.origin,
        destination=twin_normal.mission.destination,
        recommended_index=best_before
    )

    print("Showing AFTER (storm) plot...")
    plot_routes(
        routes_after,
        vessel_position=(twin_storm.vessel.navigation.lat, twin_storm.vessel.navigation.lon),
        origin=twin_storm.mission.origin,
        destination=twin_storm.mission.destination,
        recommended_index=best_after
    )


def run_storm_journey(origin="Mumbai", destination="Dubai", total_steps=40, storm_start_step=15, checkpoint_every=5):
    """
    Simulates a voyage over time. A storm gradually builds starting at
    storm_start_step, centered at a fixed geographic point. At each
    checkpoint, routes are regenerated dynamically to curve around the
    storm (scaled by current severity), then re-evaluated - so both the
    ROUTE SHAPES and the recommendation can change over time.
    """
    print(f"\n{'#'*60}")
    print("STORM JOURNEY SIMULATION")
    print("(Routes dynamically curve around the storm's location as it intensifies)")
    print(f"{'#'*60}")

    sim = DigitalTwinSimulator(
        vessel=VesselConfig(),
        mission=MissionConfig(origin=origin, destination=destination),
        simulation=SimulationConfig(scenario="normal", seed=42),
    )

    storm_point = None
    last_best_index = None
    last_heading = None
    twin_history = []

    for step in range(1, total_steps + 1):
        if step >= storm_start_step:
            sim.sim_cfg.scenario = "extreme_weather"
            sim.sim_cfg.anomaly_severity = min(1.0, (step - storm_start_step) / 10)
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

        # Fix the storm's location once, the first time it becomes active
        if storm_active and storm_point is None:
            lat1, lon1 = twin.mission.origin
            lat2, lon2 = twin.mission.destination
            storm_point = ((lat1 + lat2) / 2, (lon1 + lon2) / 2)

        # Regenerate routes every checkpoint so they dynamically reflect
        # the current storm severity/location instead of staying fixed
        routes = generate_candidate_routes(
            twin.mission.origin,
            twin.mission.destination,
            storm_point=storm_point,
            storm_severity=storm_severity
        )

        performance = simulate_all_routes(routes, twin, storm_point=storm_point, storm_severity=storm_severity)
        weights = build_adaptive_weights(twin)
        best_index, report = select_best_route(performance, weights=weights)
        health = compute_twin_health(twin, performance, report, history=twin_history[-3:])
        twin_history.append(twin)

        current_heading = twin.vessel.navigation.heading

        route_changed = (last_best_index is not None and best_index != last_best_index)
        heading_changed = (last_heading is not None and abs(current_heading - last_heading) > 2.0)

        status = "STORM" if storm_active else "calm"
        print(f"\n--- Step {step:3d} [{status}] ---")
        print(f"  Wind={twin.environment.weather.wind_speed:.1f}kt  "
              f"Wave={twin.environment.ocean.wave_height:.1f}m  "
              f"Visibility={twin.environment.weather.visibility:.1f}km")
        print(f"  Heading: {current_heading:.1f} deg")
        print(f"  Recommended: Route {best_index + 1}")
        print(f"  Twin Confidence: {health.confidence} [{health.band}]"
              + ("" if health.band == "NOMINAL" else f"  >>> {health.escalation}"))

        if route_changed:
            print(f"  >>> ROUTE UPDATED: Route {last_best_index+1} -> Route {best_index+1}")
        if heading_changed:
            print(f"  >>> SHIP TURNING: heading shifted {last_heading:.1f} -> {current_heading:.1f} deg")
        if not route_changed and not heading_changed and last_best_index is not None:
            print(f"  Holding course and route.")

        last_best_index = best_index
        last_heading = current_heading

    print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    run_before_after_comparison()
    run_storm_journey()