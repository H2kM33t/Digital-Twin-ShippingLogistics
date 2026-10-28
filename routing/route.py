"""
route.py

Route representation for the TwinRoute-M routing engine.

A Route represents a complete navigable voyage between an origin
and a destination. Unlike SearchResult, which is tied to graph
search algorithms, Route is the domain-level object exchanged
between the routing engine, simulator, Digital Twin, optimizer,
and visualisation modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .planner import SearchAlgorithm, SearchStatistics

Coordinate = Tuple[float, float]


# ==============================================================
# Route
# ==============================================================

@dataclass(slots=True)
class Route:
    """
    Complete navigation route.

    Parameters
    ----------
    node_ids
        Ordered graph node IDs.

    coordinates
        Ordered geographic coordinates.

    total_distance
        Voyage distance (km).

    total_cost
        Overall routing cost.

    algorithm
        Search algorithm used.

    statistics
        Search statistics.
    """

    node_ids: List[int] = field(default_factory=list)

    coordinates: List[Coordinate] = field(default_factory=list)

    total_distance: float = 0.0

    total_cost: float = 0.0

    algorithm: SearchAlgorithm = SearchAlgorithm.ASTAR

    statistics: SearchStatistics = field(
        default_factory=SearchStatistics
    )

    # ---------------------------------------------------------

    @property
    def number_of_waypoints(self) -> int:
        """
        Number of route waypoints.
        """

        return len(self.coordinates)

    # ---------------------------------------------------------

    @property
    def origin(self) -> Coordinate:
        """
        First coordinate.
        """

        return self.coordinates[0]

    # ---------------------------------------------------------

    @property
    def destination(self) -> Coordinate:
        """
        Final coordinate.
        """

        return self.coordinates[-1]

    # ---------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a compact summary.
        """

        return {
            "algorithm": self.algorithm.value,
            "waypoints": self.number_of_waypoints,
            "distance_km": self.total_distance,
            "cost": self.total_cost,
        }