from models import (
    NavigationState, PropulsionState, EnergyState, VesselState,
    WeatherState, OceanState, EnvironmentState,
    MissionState, DigitalTwin
)

from route_generator import generate_candidate_routes
from visualizer import plot_routes


def build_sample_twin():
    nav = NavigationState(lat=10.0, lon=75.0, speed=12.0, heading=90.0)
    prop = PropulsionState(power_kw=5000, rpm=100)
    energy = EnergyState(fuel_remaining=200.0)
    vessel = VesselState(navigation=nav, propulsion=prop, energy=energy)

    weather = WeatherState(wind_speed=8.0, wind_dir=45.0)
    ocean = OceanState(current_speed=1.2, current_dir=90.0, wave_height=2.0)
    env = EnvironmentState(weather=weather, ocean=ocean)

    mission = MissionState(origin=(10.0, 75.0), destination=(15.0, 80.0),
                            max_speed=18.0, fuel_budget=300.0)

    return DigitalTwin(vessel=vessel, environment=env, mission=mission)


if __name__ == "__main__":
    twin = build_sample_twin()
    print(twin)

    routes = generate_candidate_routes(twin.mission.origin, twin.mission.destination)

    for i, route in enumerate(routes):
        print(f"\nRoute {i+1}:")
        for wp in route.waypoints:
            print(f"  lat={wp.lat:.2f}, lon={wp.lon:.2f}")

    plot_routes(
        routes,
        vessel_position=(twin.vessel.navigation.lat, twin.vessel.navigation.lon),
        origin=twin.mission.origin,
        destination=twin.mission.destination,
        recommended_index=0
    )