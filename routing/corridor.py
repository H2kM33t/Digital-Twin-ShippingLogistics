"""
corridor.py

Navigation corridor representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Coordinate = Tuple[float, float]


@dataclass(slots=True)
class NavigationCorridor:
    """
    Multi-lane navigation corridor.
    """

    centerline: List[Coordinate]

    lanes: List[List[Coordinate]]

    longitudinal_resolution: int

    lateral_resolution: int

    corridor_width_km: float

    @property
    def number_of_nodes(self) -> int:

        return (
            self.longitudinal_resolution *
            self.lateral_resolution
        )