
"""
Digital Twin Vessel Data Generator
-----------------------------------
Generates:
1) A synchronized Digital Twin state snapshot.
2) Time-series simulation data.

Designed around:
- Vessel State
- Environment State
- Mission State

The generator is synthetic but internally consistent:
position evolves from speed/heading, fuel changes with propulsion/load,
engine temperature follows load, mission progress follows distance, and
environmental conditions evolve smoothly.

Anomalies can be injected deliberately for extreme-case demonstrations.

Dependencies:
    pip install numpy pandas
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List, Any
import math
import random

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VesselConfig:
    vessel_type: str = "cargo_ship"
    mass_tonnes: float = 12000.0
    max_speed_knots: float = 18.0
    cruise_speed_knots: float = 13.0
    max_rpm: float = 180.0
    fuel_capacity_tonnes: float = 800.0
    initial_fuel_tonnes: float = 700.0
    nominal_engine_temp_c: float = 78.0
    nominal_load_percent: float = 70.0


@dataclass
class MissionConfig:
    origin: Any = "Mumbai"
    destination: Any = "Dubai"
    # Optional explicit coordinates: (latitude, longitude)
    origin_coords: Optional[Tuple[float, float]] = None
    destination_coords: Optional[Tuple[float, float]] = None


@dataclass
class SimulationConfig:
    duration_s: float = 3600.0
    dt_s: float = 1.0
    seed: Optional[int] = 42
    scenario: str = "normal"
    anomaly_probability: float = 0.02
    anomaly_severity: float = 0.7


# A small built-in location database so the demonstration can use names.
LOCATIONS = {
    "Mumbai": (19.0760, 72.8777),
    "Dubai": (25.2048, 55.2708),
    "Goa": (15.4909, 73.8278),
    "Kochi": (9.9312, 76.2673),
    "Colombo": (6.9271, 79.8612),
    "Singapore": (1.3521, 103.8198),
    "Chennai": (13.0827, 80.2707),
    "Visakhapatnam": (17.6868, 83.2185),
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle_deg: float) -> float:
    return angle_deg % 360.0


def angle_difference(target_deg: float, current_deg: float) -> float:
    """Signed shortest angular difference in [-180, 180]."""
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def destination_point(
    lat_deg: float,
    lon_deg: float,
    bearing_deg: float,
    distance_km: float,
) -> Tuple[float, float]:
    """Move a point along a great-circle bearing."""
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    bearing = math.radians(bearing_deg)
    angular_distance = distance_km / EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), ((math.degrees(lon2) + 540) % 360) - 180


def haversine_km(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> float:
    lat1, lat2 = math.radians(lat1_deg), math.radians(lat2_deg)
    dlat = math.radians(lat2_deg - lat1_deg)
    dlon = math.radians(lon2_deg - lon1_deg)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def initial_bearing(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> float:
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    dlon = math.radians(lon2_deg - lon1_deg)

    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )
    return normalize_angle(math.degrees(math.atan2(x, y)))


def resolve_location(
    value: Any,
    explicit_coords: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    if explicit_coords is not None:
        return float(explicit_coords[0]), float(explicit_coords[1])

    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])

    if isinstance(value, str) and value in LOCATIONS:
        return LOCATIONS[value]

    raise ValueError(
        f"Unknown location {value!r}. Use a built-in name {list(LOCATIONS)} "
        "or provide (latitude, longitude)."
    )


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class DigitalTwinSimulator:
    """
    Digital Twin simulator for a vessel.

    Supported scenarios:
        normal
        engine_overheating
        rpm_drop
        fuel_leak
        propulsion_degradation
        sensor_gps_anomaly
        extreme_weather
        strong_current
        reduced_visibility
        random_anomalies
    """

    VALID_SCENARIOS = {
        "normal",
        "engine_overheating",
        "rpm_drop",
        "fuel_leak",
        "propulsion_degradation",
        "sensor_gps_anomaly",
        "extreme_weather",
        "strong_current",
        "reduced_visibility",
        "random_anomalies",
    }

    def __init__(
        self,
        vessel: Optional[VesselConfig] = None,
        mission: Optional[MissionConfig] = None,
        simulation: Optional[SimulationConfig] = None,
    ):
        self.vessel_cfg = vessel or VesselConfig()
        self.mission_cfg = mission or MissionConfig()
        self.sim_cfg = simulation or SimulationConfig()

        if self.sim_cfg.scenario not in self.VALID_SCENARIOS:
            raise ValueError(
                f"Invalid scenario {self.sim_cfg.scenario!r}. "
                f"Choose from {sorted(self.VALID_SCENARIOS)}."
            )

        self.rng = random.Random(self.sim_cfg.seed)
        self.np_rng = np.random.default_rng(self.sim_cfg.seed)

        self.origin = resolve_location(
            self.mission_cfg.origin,
            self.mission_cfg.origin_coords,
        )
        self.destination = resolve_location(
            self.mission_cfg.destination,
            self.mission_cfg.destination_coords,
        )

        self.total_route_km = haversine_km(
            *self.origin,
            *self.destination,
        )
        self.initial_bearing = initial_bearing(
            *self.origin,
            *self.destination,
        )

        # Dynamic state
        self.t = 0.0
        self.lat, self.lon = self.origin
        self.speed_knots = self.vessel_cfg.cruise_speed_knots
        self.heading_deg = self.initial_bearing
        self.rpm = 0.90 * self.vessel_cfg.max_rpm
        self.fuel_tonnes = self.vessel_cfg.initial_fuel_tonnes
        self.engine_temp_c = self.vessel_cfg.nominal_engine_temp_c
        self.load_percent = self.vessel_cfg.nominal_load_percent
        self.health_percent = 100.0

        # Slowly varying environmental state
        self.wind_speed = 12.0
        self.wind_direction = normalize_angle(self.initial_bearing + 70)
        self.air_temp = 28.0
        self.wave_height = 1.4
        self.wave_period = 7.5
        self.current_speed = 0.7
        self.current_direction = normalize_angle(self.initial_bearing + 110)
        self.visibility_km = 15.0

        self.active_anomalies: List[str] = []

    # -----------------------------------------------------------------------
    # Environment model
    # -----------------------------------------------------------------------

    def _update_environment(self, dt: float):
        # Smooth random walk around realistic baseline conditions.
        self.wind_speed += self.np_rng.normal(0, 0.025 * math.sqrt(max(dt, 0.1)))
        self.wind_speed = clamp(self.wind_speed, 3.0, 30.0)

        self.wind_direction = normalize_angle(
            self.wind_direction
            + self.np_rng.normal(0, 0.18 * math.sqrt(max(dt, 0.1)))
        )

        self.air_temp += self.np_rng.normal(
            0, 0.008 * math.sqrt(max(dt, 0.1))
        )
        self.air_temp = clamp(self.air_temp, 18.0, 36.0)

        target_wave = 0.05 * self.wind_speed + 0.7
        self.wave_height += 0.03 * (target_wave - self.wave_height)
        self.wave_height += self.np_rng.normal(0, 0.008)
        self.wave_height = clamp(self.wave_height, 0.2, 6.0)

        self.wave_period = clamp(
            5.5 + 1.2 * math.sqrt(self.wave_height)
            + self.np_rng.normal(0, 0.05),
            4.0,
            14.0,
        )

        self.current_speed += self.np_rng.normal(0, 0.01)
        self.current_speed = clamp(self.current_speed, 0.0, 2.5)

        self.current_direction = normalize_angle(
            self.current_direction + self.np_rng.normal(0, 0.3)
        )

        self.visibility_km += self.np_rng.normal(0, 0.02)
        self.visibility_km = clamp(self.visibility_km, 1.0, 25.0)

        scenario = self.sim_cfg.scenario

        if scenario == "extreme_weather":
            sev = self.sim_cfg.anomaly_severity
            self.wind_speed = max(self.wind_speed, 20 + 10 * sev)
            self.wave_height = max(self.wave_height, 3 + 3 * sev)
            self.visibility_km = min(self.visibility_km, 8 - 5 * sev)

        elif scenario == "strong_current":
            sev = self.sim_cfg.anomaly_severity
            self.current_speed = max(
                self.current_speed,
                1.8 + 1.0 * sev,
            )

        elif scenario == "reduced_visibility":
            sev = self.sim_cfg.anomaly_severity
            self.visibility_km = min(
                self.visibility_km,
                max(0.5, 4.0 - 3.0 * sev),
            )

    # -----------------------------------------------------------------------
    # Vessel model
    # -----------------------------------------------------------------------

    def _update_vessel(self, dt: float):
        scenario = self.sim_cfg.scenario
        sev = self.sim_cfg.anomaly_severity

        # Desired speed is reduced in poor conditions.
        weather_penalty = (
            0.04 * max(self.wind_speed - 15, 0)
            + 0.7 * max(self.wave_height - 2, 0)
        )
        visibility_penalty = 0.8 if self.visibility_km < 3 else 0.0

        desired_speed = (
            self.vessel_cfg.cruise_speed_knots
            - weather_penalty
            - visibility_penalty
        )

        # Anomaly effects.
        if scenario == "propulsion_degradation":
            desired_speed *= 1.0 - 0.40 * sev

        if scenario == "rpm_drop":
            self.rpm *= 1.0 - 0.65 * sev

        else:
            desired_rpm = (
                desired_speed / self.vessel_cfg.max_speed_knots
            ) * self.vessel_cfg.max_rpm
            self.rpm += 0.04 * (desired_rpm - self.rpm) * dt

        self.rpm = clamp(self.rpm, 0.0, self.vessel_cfg.max_rpm)

        # Speed responds gradually to available propulsion.
        propulsion_fraction = self.rpm / self.vessel_cfg.max_rpm
        target_speed = (
            propulsion_fraction * self.vessel_cfg.max_speed_knots
        )

        if scenario == "propulsion_degradation":
            target_speed *= 1.0 - 0.35 * sev

        self.speed_knots += 0.08 * (target_speed - self.speed_knots) * dt
        self.speed_knots = clamp(
            self.speed_knots + self.np_rng.normal(0, 0.01),
            0.0,
            self.vessel_cfg.max_speed_knots,
        )

        # Heading follows the direct route with small realistic steering noise.
        target_heading = initial_bearing(
            self.lat,
            self.lon,
            self.destination[0],
            self.destination[1],
        )
        heading_error = angle_difference(target_heading, self.heading_deg)
        self.heading_deg = normalize_angle(
            self.heading_deg + clamp(heading_error, -2.0, 2.0) * min(dt, 1.0)
        )

        # Engine load.
        self.load_percent = clamp(
            100.0
            * propulsion_fraction
            + 0.8 * self.wave_height
            + self.np_rng.normal(0, 0.5),
            10.0,
            110.0,
        )

        # Engine temperature follows load.
        target_temp = (
            self.vessel_cfg.nominal_engine_temp_c
            + 0.18 * (self.load_percent - 65)
        )

        if scenario == "engine_overheating":
            target_temp += 35.0 * sev

        self.engine_temp_c += (
            0.015 * (target_temp - self.engine_temp_c) * dt
        )
        self.engine_temp_c += self.np_rng.normal(0, 0.04)

        # Fuel consumption: approximately proportional to RPM/load.
        base_burn_tph = (
            1.8
            + 0.00045 * self.vessel_cfg.mass_tonnes
            + 0.035 * self.load_percent
        )

        if scenario == "fuel_leak":
            base_burn_tph *= 1.0 + 2.5 * sev

        if scenario == "propulsion_degradation":
            base_burn_tph *= 1.0 + 0.5 * sev

        fuel_used = base_burn_tph * dt / 3600.0
        self.fuel_tonnes = max(0.0, self.fuel_tonnes - fuel_used)

        # Health reflects thermal and propulsion stress.
        thermal_stress = max(self.engine_temp_c - 90.0, 0.0) * 0.08
        load_stress = max(self.load_percent - 90.0, 0.0) * 0.03
        health_loss = (thermal_stress + load_stress) * dt / 3600.0

        if scenario == "engine_overheating":
            health_loss += 0.04 * sev * dt

        if scenario == "propulsion_degradation":
            health_loss += 0.015 * sev * dt

        self.health_percent = clamp(
            self.health_percent - health_loss,
            0.0,
            100.0,
        )

        # Position update. 1 knot = 1 nautical mile/hour.
        # Add current vector to vessel ground velocity.
        vessel_distance_km = self.speed_knots * 1.852 * dt / 3600.0

        # Current contribution.
        current_distance_km = self.current_speed * dt / 3600.0
        current_heading = math.radians(self.current_direction)

        vessel_heading = math.radians(self.heading_deg)

        north_km = (
            vessel_distance_km * math.cos(vessel_heading)
            + current_distance_km * math.cos(current_heading)
        )
        east_km = (
            vessel_distance_km * math.sin(vessel_heading)
            + current_distance_km * math.sin(current_heading)
        )

        new_lat = self.lat + north_km / 111.0
        longitude_scale = max(math.cos(math.radians(self.lat)), 0.1)
        new_lon = self.lon + east_km / (111.0 * longitude_scale)

        self.lat, self.lon = new_lat, new_lon

        # GPS sensor anomaly affects reported position, not actual position.
        reported_lat, reported_lon = self.lat, self.lon

        if scenario == "sensor_gps_anomaly":
            offset = 0.02 * sev
            reported_lat += offset
            reported_lon -= offset

        return reported_lat, reported_lon

    # -----------------------------------------------------------------------
    # Mission state
    # -----------------------------------------------------------------------

    def _mission_state(self):
        remaining_km = haversine_km(
            self.lat,
            self.lon,
            self.destination[0],
            self.destination[1],
        )

        travelled_km = max(
            0.0,
            self.total_route_km - remaining_km,
        )

        progress = clamp(
            100.0 * travelled_km / max(self.total_route_km, 0.001),
            0.0,
            100.0,
        )

        ground_speed_kmh = max(self.speed_knots * 1.852, 0.1)
        eta_hours = remaining_km / ground_speed_kmh

        route_status = "ON_ROUTE"
        if self.visibility_km < 3:
            route_status = "CAUTION"
        if self.wave_height > 4:
            route_status = "SEVERE_WEATHER"
        if self.health_percent < 60:
            route_status = "MACHINERY_RISK"

        return {
            "origin": self.mission_cfg.origin,
            "destination": self.mission_cfg.destination,
            "origin_coordinates": self.origin,
            "destination_coordinates": self.destination,
            "route_bearing_deg": round(self.initial_bearing, 3),
            "distance_travelled_km": round(travelled_km, 3),
            "distance_remaining_km": round(remaining_km, 3),
            "progress_percent": round(progress, 3),
            "eta_hours": round(eta_hours, 3),
            "route_status": route_status,
        }

    # -----------------------------------------------------------------------
    # Alerts / anomalies
    # -----------------------------------------------------------------------

    def _alerts(self):
        alerts = []

        if self.engine_temp_c >= 100:
            alerts.append("ENGINE_OVERHEATING")
        elif self.engine_temp_c >= 90:
            alerts.append("HIGH_ENGINE_TEMPERATURE")

        if self.fuel_tonnes <= 0.10 * self.vessel_cfg.fuel_capacity_tonnes:
            alerts.append("LOW_FUEL")

        if self.wave_height >= 4.0:
            alerts.append("HIGH_WAVE_HEIGHT")

        if self.wind_speed >= 25:
            alerts.append("HIGH_WIND")

        if self.visibility_km <= 3:
            alerts.append("LOW_VISIBILITY")

        if self.health_percent <= 60:
            alerts.append("MACHINERY_HEALTH_DEGRADED")

        scenario = self.sim_cfg.scenario
        if scenario != "normal":
            alerts.append(f"SCENARIO_{scenario.upper()}")

        return alerts

    # -----------------------------------------------------------------------
    # State generation
    # -----------------------------------------------------------------------

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Advance the simulation and return one synchronized state.

        In ``random_anomalies`` mode, an anomaly is sampled at each step
        according to ``anomaly_probability``. The selected anomaly is applied
        only for that simulation step, while the underlying vessel state
        continues evolving.
        """
        dt = float(dt if dt is not None else self.sim_cfg.dt_s)

        original_scenario = self.sim_cfg.scenario
        active_scenario = original_scenario

        if original_scenario == "random_anomalies":
            anomaly_choices = [
                "normal",
                "engine_overheating",
                "rpm_drop",
                "fuel_leak",
                "propulsion_degradation",
                "sensor_gps_anomaly",
                "extreme_weather",
                "strong_current",
                "reduced_visibility",
            ]

            if self.rng.random() < self.sim_cfg.anomaly_probability:
                active_scenario = self.rng.choice(anomaly_choices)
            else:
                active_scenario = "normal"

            self.active_anomalies = (
                [] if active_scenario == "normal" else [active_scenario]
            )

        self.sim_cfg.scenario = active_scenario

        self.t += dt
        self._update_environment(dt)
        reported_lat, reported_lon = self._update_vessel(dt)

        alerts = self._alerts()

        # Restore the user-selected mode after applying the temporary
        # random anomaly.
        self.sim_cfg.scenario = original_scenario

        if original_scenario == "random_anomalies":
            if self.active_anomalies:
                alerts.append(
                    f"RANDOM_ANOMALY_{self.active_anomalies[0].upper()}"
                )

        return {
            "timestamp_s": round(self.t, 3),
            "vessel": {
                "latitude": round(reported_lat, 6),
                "longitude": round(reported_lon, 6),
                "actual_latitude": round(self.lat, 6),
                "actual_longitude": round(self.lon, 6),
                "speed_knots": round(self.speed_knots, 3),
                "heading_deg": round(self.heading_deg, 3),
                "rpm": round(self.rpm, 3),
                "fuel_remaining_tonnes": round(self.fuel_tonnes, 3),
                "fuel_remaining_percent": round(
                    100 * self.fuel_tonnes
                    / self.vessel_cfg.fuel_capacity_tonnes,
                    3,
                ),
                "fuel_consumption_estimate_tph": round(
                    max(
                        0,
                        (self.vessel_cfg.initial_fuel_tonnes
                         - self.fuel_tonnes)
                        / max(self.t / 3600.0, 1e-6),
                    ),
                    3,
                ),
                "engine_temperature_c": round(self.engine_temp_c, 3),
                "load_percent": round(self.load_percent, 3),
                "propulsion_status": (
                    "DEGRADED"
                    if self.health_percent < 75
                    else "NORMAL"
                ),
                "machinery_health_percent": round(self.health_percent, 3),
            },
            "environment": {
                "wind_speed_knots": round(self.wind_speed, 3),
                "wind_direction_deg": round(self.wind_direction, 3),
                "air_temperature_c": round(self.air_temp, 3),
                "wave_height_m": round(self.wave_height, 3),
                "wave_period_s": round(self.wave_period, 3),
                "ocean_current_speed_knots": round(
                    self.current_speed, 3
                ),
                "ocean_current_direction_deg": round(
                    self.current_direction, 3
                ),
                "visibility_km": round(self.visibility_km, 3),
            },
            "mission": self._mission_state(),
            "alerts": alerts,
        }

    def state(self) -> Dict[str, Any]:
        """Generate one complete Digital Twin state without advancing time."""
        original_scenario = self.sim_cfg.scenario
        self.sim_cfg.scenario = "normal"

        state = {
            "timestamp_s": round(self.t, 3),
            "vessel": {
                "latitude": round(self.lat, 6),
                "longitude": round(self.lon, 6),
                "actual_latitude": round(self.lat, 6),
                "actual_longitude": round(self.lon, 6),
                "speed_knots": round(self.speed_knots, 3),
                "heading_deg": round(self.heading_deg, 3),
                "rpm": round(self.rpm, 3),
                "fuel_remaining_tonnes": round(self.fuel_tonnes, 3),
                "fuel_remaining_percent": round(
                    100 * self.fuel_tonnes
                    / self.vessel_cfg.fuel_capacity_tonnes,
                    3,
                ),
                "fuel_consumption_estimate_tph": 0.0,
                "engine_temperature_c": round(self.engine_temp_c, 3),
                "load_percent": round(self.load_percent, 3),
                "propulsion_status": "NORMAL",
                "machinery_health_percent": round(self.health_percent, 3),
            },
            "environment": {
                "wind_speed_knots": round(self.wind_speed, 3),
                "wind_direction_deg": round(self.wind_direction, 3),
                "air_temperature_c": round(self.air_temp, 3),
                "wave_height_m": round(self.wave_height, 3),
                "wave_period_s": round(self.wave_period, 3),
                "ocean_current_speed_knots": round(self.current_speed, 3),
                "ocean_current_direction_deg": round(
                    self.current_direction, 3
                ),
                "visibility_km": round(self.visibility_km, 3),
            },
            "mission": self._mission_state(),
            "alerts": [],
        }

        self.sim_cfg.scenario = original_scenario
        return state

    def run(self) -> pd.DataFrame:
        """Generate synchronized time-series data."""
        if self.sim_cfg.dt_s <= 0:
            raise ValueError("dt_s must be > 0")

        n_steps = int(math.floor(
            self.sim_cfg.duration_s / self.sim_cfg.dt_s
        ))

        rows = []
        for _ in range(n_steps + 1):
            state = self.step(self.sim_cfg.dt_s)
            rows.append(flatten_state(state))

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Flatten nested state for DataFrame output
# ---------------------------------------------------------------------------

def flatten_state(state: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "timestamp_s": state["timestamp_s"],
    }

    for group in ("vessel", "environment", "mission"):
        for key, value in state[group].items():
            if isinstance(value, (tuple, list)):
                row[f"{group}_{key}"] = str(value)
            else:
                row[f"{group}_{key}"] = value

    row["alerts"] = "|".join(state["alerts"])
    return row


# ---------------------------------------------------------------------------
# Simple command-style API
# ---------------------------------------------------------------------------

def generate_twin(
    mode: str = "state",
    *,
    duration: float = 3600.0,
    dt: float = 1.0,
    origin: Any = "Mumbai",
    destination: Any = "Dubai",
    scenario: str = "normal",
    anomaly_probability: float = 0.02,
    anomaly_severity: float = 0.7,
    seed: Optional[int] = 42,
    vessel_type: str = "cargo_ship",
) -> Any:
    """
    Main user-facing function.

    mode:
        "state"      -> one Digital Twin snapshot
        "timeseries" -> pandas DataFrame

    Examples:
        state = generate_twin("state")

        data = generate_twin(
            "timeseries",
            duration=3600,
            dt=1,
            scenario="engine_overheating",
        )
    """

    if mode.lower() not in {"state", "timeseries"}:
        raise ValueError("mode must be 'state' or 'timeseries'")

    vessel = VesselConfig(vessel_type=vessel_type)

    mission = MissionConfig(
        origin=origin,
        destination=destination,
    )

    simulation = SimulationConfig(
        duration_s=duration,
        dt_s=dt,
        seed=seed,
        scenario=scenario,
        anomaly_probability=anomaly_probability,
        anomaly_severity=clamp(anomaly_severity, 0.0, 1.0),
    )

    sim = DigitalTwinSimulator(
        vessel=vessel,
        mission=mission,
        simulation=simulation,
    )

    if mode.lower() == "state":
        return sim.state()

    return sim.run()


# ---------------------------------------------------------------------------
# Optional route-cost evaluation for future route optimization
# ---------------------------------------------------------------------------

def estimate_route_cost(
    distance_km: float,
    average_speed_knots: float,
    fuel_consumption_tph: float,
    weather_penalty: float = 0.0,
    risk_penalty: float = 0.0,
) -> Dict[str, float]:
    """
    Lightweight route-cost function.

    This is intentionally separate from the data generator so that a
    route-optimization algorithm can later replace it.
    """
    travel_hours = distance_km / max(average_speed_knots * 1.852, 0.01)
    fuel_used = fuel_consumption_tph * travel_hours

    cost = (
        travel_hours
        + 0.01 * fuel_used
        + weather_penalty
        + risk_penalty
    )

    return {
        "travel_hours": travel_hours,
        "estimated_fuel_tonnes": fuel_used,
        "route_cost": cost,
    }


if __name__ == "__main__":
    # Demonstration: A - one state
    state = generate_twin(
        mode="state",
        origin="Mumbai",
        destination="Dubai",
        scenario="normal",
    )

    print("\n--- DIGITAL TWIN STATE ---")
    print(state["vessel"])
    print(state["environment"])
    print(state["mission"])
    print("Alerts:", state["alerts"])

    # Demonstration: B - time series
    data = generate_twin(
        mode="timeseries",
        duration=600,
        dt=1,
        origin="Mumbai",
        destination="Dubai",
        scenario="engine_overheating",
        anomaly_severity=0.8,
        seed=42,
    )

    print("\n--- TIME SERIES ---")
    print(data.head())
    print("\nRows:", len(data))
    print("Columns:", len(data.columns))

    # Save for use in MATLAB/Excel/dashboard/ML pipeline.
    data.to_csv("digital_twin_timeseries.csv", index=False)
