from models import (
    NavigationState, PropulsionState, EnergyState, VesselState,
    WeatherState, OceanState, EnvironmentState,
    MissionState, DigitalTwin
)
from route_generator import generate_candidate_routes
from visualizer import plot_routes
from simulator import simulate_all_routes
from optimizer import select_best_route
from objective_builder import build_adaptive_weights, compute_pressures, explain_weights, BASE_WEIGHTS
from digital_twin_generator import DigitalTwinSimulator, VesselConfig, MissionConfig, SimulationConfig


def generate_stepped_state(origin, destination, scenario="normal", steps=10, seed=42):
    """Run the realistic simulator forward a few steps to get a proper state."""
    sim = DigitalTwinSimulator(
        vessel=VesselConfig(),
        mission=MissionConfig(origin=origin, destination=destination),
        simulation=SimulationConfig(scenario=scenario, seed=seed),
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

    weather = WeatherState(wind_speed=e["wind_speed_knots"], wind_dir=e["wind_direction_deg"])
    ocean = OceanState(
        current_speed=e["ocean_current_speed_knots"],
        current_dir=e["ocean_current_direction_deg"],
        wave_height=e["wave_height_m"],
    )
    env = EnvironmentState(weather=weather, ocean=ocean)

    mission = MissionState(origin=origin_coords, destination=destination_coords,
                            max_speed=max_speed, fuel_budget=fuel_budget)

    return DigitalTwin(vessel=vessel, environment=env, mission=mission)


def build_sample_twin():
    """Now powered by the realistic Digital Twin generator instead of hand-typed numbers."""
    state = generate_stepped_state(origin="Mumbai", destination="Dubai", scenario="normal", steps=10)
    return twin_state_to_digital_twin(
        state,
        origin_coords=state["mission"]["origin_coordinates"],
        destination_coords=state["mission"]["destination_coordinates"],
    )


if __name__ == "__main__":
    twin = build_sample_twin()
    print(twin)

    routes = generate_candidate_routes(twin.mission.origin, twin.mission.destination)

    for i, route in enumerate(routes):
        print(f"\nRoute {i+1}:")
        for wp in route.waypoints:
            print(f"  lat={wp.lat:.2f}, lon={wp.lon:.2f}")

    performance = simulate_all_routes(routes, twin)

    for i, perf in enumerate(performance):
        print(f"\nRoute {i+1} Performance:")
        print(f"  Distance: {perf['distance']} km")
        print(f"  Fuel:     {perf['fuel']} tonnes")
        print(f"  Time:     {perf['time']} hours")
        print(f"  Risk:     {perf['risk']}")

    adaptive_weights = build_adaptive_weights(twin)
    pressures = compute_pressures(twin)
    print(f"\n{explain_weights(BASE_WEIGHTS, adaptive_weights, pressures)}")

    best_index, report = select_best_route(performance, weights=adaptive_weights)

    print(f"\nPareto-optimal routes: {[i+1 for i in report['pareto_indices']]}")
    print("TOPSIS scores (higher = better):")
    for idx, score in report['topsis_scores'].items():
        print(f"  Route {idx+1}: {score:.3f}")

    print(f"\n>>> Recommended Route: Route {best_index + 1}")

    plot_routes(
        routes,
        vessel_position=(twin.vessel.navigation.lat, twin.vessel.navigation.lon),
        origin=twin.mission.origin,
        destination=twin.mission.destination,
        recommended_index=best_index
    )