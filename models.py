from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class NavigationState:
    lat: float
    lon: float
    speed: float
    heading: float

@dataclass
class PropulsionState:
    power_kw: float
    rpm: float
    efficiency: float = 0.85

@dataclass
class EnergyState:
    fuel_remaining: float

@dataclass
class VesselState:
    navigation: NavigationState
    propulsion: PropulsionState
    energy: EnergyState


@dataclass
class WeatherState:
    wind_speed: float
    wind_dir: float

@dataclass
class OceanState:
    current_speed: float
    current_dir: float
    wave_height: float

@dataclass
class EnvironmentState:
    weather: WeatherState
    ocean: OceanState


@dataclass
class MissionState:
    origin: tuple
    destination: tuple
    max_speed: float
    fuel_budget: float
    progress: float = 0.0


@dataclass
class Waypoint:
    lat: float
    lon: float
    eta: datetime = None
    speed: float = None
    heading: float = None

@dataclass
class Route:
    waypoints: List[Waypoint] = field(default_factory=list)

    def add_waypoint(self, wp: Waypoint):
        self.waypoints.append(wp)


@dataclass
class DigitalTwin:
    vessel: VesselState
    environment: EnvironmentState
    mission: MissionState