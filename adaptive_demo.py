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


def generate_stepped_state(origin, destination, scenario, steps=10, seed=42, anomaly_severity=0.9):
    """
    Run the DigitalTwinSimulator forward a few steps so scenario-specific
    effects (like extreme_weather) actually get applied. Calling .state()
    directly does NOT apply scenario effects - it always reports as "normal".
    """
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
    """Convert a state dict from digital_twin_generator.py into a DigitalTwin object."""
    v = state["vessel"]
    e = state["environment"]

    nav = NavigationState(
        lat=v["latitude"], lon=v["longitude"],
        speed=v["speed_knots"], heading=v["heading_deg"],
    )
    prop = PropulsionState(power_kw=5000, rpm=v["rpm"], efficiency=0.85)
    energy = EnergyState(fuel_remaining=v["fuel_remaining_tonnes"])
    vessel = VesselState(navigation=nav, propulsion=prop, energy=energy)

    weather = WeatherState(wind_speed=e["wind_speed_knots"], wind_dir=e["wind_direction_deg"])
    ocean = OceanState(
        current_speed=e["ocean_current_speed_knots"],
        current_dir=e["ocean_current_direction_deg"],
        wave_height=e["wave_height_m"],
    )
    env = EnvironmentState(weather=weather, ocean=ocean)

    mission = MissionState(
        origin=origin_coords, destination=destination_coords,
        max_speed=max_speed, fuel_budget=fuel_budget,
    )

    return DigitalTwin(vessel=vessel, environment=env, mission=mission)


def print_report(label, performance, best_index, report, weight_explanation=None):
    print(f"\n{'='*50}")
    print(label)
    print('='*50)
    for i, perf in enumerate(performance):
        print(f"Route {i+1}: Fuel={perf['fuel']}t  Time={perf['time']}h  Risk={perf['risk']}")
    if weight_explanation:
        print(f"\n{weight_explanation}")
    print(f"\nPareto-optimal routes: {[i+1 for i in report['pareto_indices']]}")
    print(f">>> Recommended Route: Route {best_index + 1}")


if __name__ == "__main__":
    origin = "Mumbai"
    destination = "Dubai"

    # ---- BEFORE: normal conditions ----
    state_normal = generate_stepped_state(origin, destination, scenario="normal", steps=10)
    twin_normal = twin_state_to_digital_twin(
        state_normal,
        state_normal["mission"]["origin_coordinates"],
        state_normal["mission"]["destination_coordinates"],
    )

    routes = generate_candidate_routes(twin_normal.mission.origin, twin_normal.mission.destination)

    performance_before = simulate_all_routes(routes, twin_normal)
    weights_before = build_adaptive_weights(twin_normal)
    pressures_before = compute_pressures(twin_normal)
    explanation_before = explain_weights(BASE_WEIGHTS, weights_before, pressures_before)
    best_before, report_before = select_best_route(performance_before, weights=weights_before)
    print_report("BEFORE: Normal Conditions", performance_before, best_before, report_before, explanation_before)

    # ---- AFTER: storm conditions ----
    state_storm = generate_stepped_state(origin, destination, scenario="extreme_weather", steps=10, anomaly_severity=0.9)
    twin_storm = twin_state_to_digital_twin(
        state_storm,
        state_storm["mission"]["origin_coordinates"],
        state_storm["mission"]["destination_coordinates"],
    )

    performance_after = simulate_all_routes(routes, twin_storm)
    weights_after = build_adaptive_weights(twin_storm)
    pressures_after = compute_pressures(twin_storm)
    explanation_after = explain_weights(BASE_WEIGHTS, weights_after, pressures_after)
    best_after, report_after = select_best_route(performance_after, weights=weights_after)
    print_report("AFTER: Storm Detected (extreme_weather)", performance_after, best_after, report_after, explanation_after)

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
        routes,
        vessel_position=(twin_normal.vessel.navigation.lat, twin_normal.vessel.navigation.lon),
        origin=twin_normal.mission.origin,
        destination=twin_normal.mission.destination,
        recommended_index=best_before
    )

    print("Showing AFTER (storm) plot...")
    plot_routes(
        routes,
        vessel_position=(twin_storm.vessel.navigation.lat, twin_storm.vessel.navigation.lon),
        origin=twin_storm.mission.origin,
        destination=twin_storm.mission.destination,
        recommended_index=best_after
    )