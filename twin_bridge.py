"""
twin_bridge.py

Connects the graph-search voyage engine (routing/voyage.py) to the original
Digital Twin pipeline, once per voyage frame:

    DigitalTwinSimulator   wind / waves / visibility, pushed up by nearby storms
    simulator.py           fuel, time and risk for each candidate route
    objective_builder.py   adaptive weights from the current conditions
    optimizer.py           Pareto set + TOPSIS ranking of the candidates
    twin_health.py         confidence in the twin's own recommendation

The ship's route and halt / deviate decisions come from the graph search; the
twin scores the candidate routes against the same storms and reports what
it would recommend and how much it trusts that recommendation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from digital_twin_generator import (
    DigitalTwinSimulator, MissionConfig, SimulationConfig, VesselConfig,
)
from models import (
    DigitalTwin, EnergyState, EnvironmentState, MissionState, NavigationState,
    OceanState, PropulsionState, Route, VesselState, Waypoint, WeatherState,
)
from objective_builder import build_adaptive_weights
from optimizer import select_best_route
from simulator import simulate_route
from twin_health import compute_twin_health

Coordinate = Tuple[float, float]

# How far from a storm centre (in storm radii) the ship starts to feel it.
FEEL_RADIUS = 1.6
CALM_WIND_KT = 12.0
CALM_VISIBILITY_KM = 15.0


def _km(a: Coordinate, b: Coordinate) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def storm_proximity(pos: Coordinate, storms: List[dict]) -> float:
    """0..1: how strongly the ship feels the storms around it right now."""

    feel = 0.0
    for s in storms:
        d = _km(pos, (s["lat"], s["lon"]))
        k = max(0.0, 1.0 - d / (FEEL_RADIUS * s["radius_km"]))
        feel = max(feel, s["severity"] * k)
    return min(1.0, feel)


def _twin_from_state(state: dict, origin: Coordinate, destination: Coordinate,
                     pos: Coordinate, speed_kts: float) -> DigitalTwin:
    v, e = state["vessel"], state["environment"]

    nav = NavigationState(lat=pos[0], lon=pos[1], speed=speed_kts,
                          heading=v["heading_deg"])
    prop = PropulsionState(power_kw=5000, rpm=v["rpm"], efficiency=0.85)
    energy = EnergyState(fuel_remaining=v["fuel_remaining_tonnes"])

    weather = WeatherState(
        wind_speed=e["wind_speed_knots"], wind_dir=e["wind_direction_deg"],
        air_temp=e["air_temperature_c"], visibility=e["visibility_km"],
    )
    ocean = OceanState(
        current_speed=e["ocean_current_speed_knots"],
        current_dir=e["ocean_current_direction_deg"],
        wave_height=e["wave_height_m"], wave_period=e["wave_period_s"],
    )
    return DigitalTwin(
        vessel=VesselState(navigation=nav, propulsion=prop, energy=energy),
        environment=EnvironmentState(weather=weather, ocean=ocean),
        mission=MissionState(origin=origin, destination=destination,
                             max_speed=18.0, fuel_budget=300.0),
    )


class TwinObserver:
    """Callable passed to VoyageSimulator.run(observer=...)."""

    def __init__(self, origin: Coordinate, destination: Coordinate) -> None:
        self.origin = origin
        self.destination = destination
        self.sim = DigitalTwinSimulator(
            vessel=VesselConfig(),
            mission=MissionConfig(origin="Mumbai", destination="Dubai",
                                  origin_coords=origin,
                                  destination_coords=destination),
            simulation=SimulationConfig(scenario="normal", seed=42),
        )
        self.history: List[DigitalTwin] = []

    def environment_step(self, pos: Coordinate, storms: List[dict],
                         speed_kts: float) -> Tuple[DigitalTwin, float]:
        """Advance the twin's environment one hour at `pos`; returns the
        DigitalTwin snapshot and how strongly the storms are felt (0..1)."""

        sim = self.sim
        prox = storm_proximity(pos, storms)

        # Storms push the twin's environment up; it relaxes once they are gone.
        if prox > 0.05:
            sim.sim_cfg.scenario = "extreme_weather"
            sim.sim_cfg.anomaly_severity = prox
        else:
            sim.sim_cfg.scenario = "normal"
            sim.wind_speed += 0.25 * (CALM_WIND_KT - sim.wind_speed)
            sim.visibility_km += 0.25 * (CALM_VISIBILITY_KM - sim.visibility_km)

        state = sim.step(dt=1.0)
        twin = _twin_from_state(state, self.origin, self.destination, pos, speed_kts)
        return twin, prox

    def __call__(self, ctx: dict) -> Dict:
        twin, prox = self.environment_step(ctx["pos"], ctx["storms"], ctx["speed_kts"])

        routes = [
            Route(waypoints=[Waypoint(lat=la, lon=lo) for la, lo in c["coords"]])
            for c in ctx["candidates"]
        ]
        performance = [
            simulate_route(r, twin, storm_exposure=c["peak_risk"], storm_severity=1.0)
            for r, c in zip(routes, ctx["candidates"])
        ]

        weights = build_adaptive_weights(twin)
        best, report = select_best_route(performance, weights=weights)
        health = compute_twin_health(twin, performance, report,
                                     history=self.history[-3:])
        self.history.append(twin)

        return {
            "twin": {
                "wind_kt": round(twin.environment.weather.wind_speed, 1),
                "wave_m": round(twin.environment.ocean.wave_height, 1),
                "visibility_km": round(twin.environment.weather.visibility, 1),
                "storm_feel": round(prox, 3),
                "confidence": round(health.confidence, 3),
                "band": health.band,
                "subscores": {k: round(v, 3) for k, v in health.subscores.items()},
            },
            "routes_performance": [
                {"name": c["name"], **p}
                for c, p in zip(ctx["candidates"], performance)
            ],
            "twin_pick": int(best),
            "pareto": [int(i) for i in report["pareto_indices"]],
        }
