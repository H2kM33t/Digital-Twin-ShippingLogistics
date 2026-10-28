"""
edge.py

Graph edge definitions for the TwinRoute-M routing engine.

The routing graph stores only static topology and geometric properties.
Mission-dependent quantities such as travel time and fuel consumption
are computed dynamically by the Digital Twin during simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ======================================================================
# Edge Cost
# ======================================================================

@dataclass(slots=True)
class EdgeCost:
    """
    Static and dynamic cost components associated with traversing an edge.

    Parameters
    ----------
    distance : float
        Great-circle distance between the source and target nodes (km).

    bearing : float
        Initial bearing from the source node to the target node (degrees).

    weather_cost : float, optional
        Weather penalty applied to this edge.

    wave_cost : float, optional
        Wave-induced penalty.

    current_cost : float, optional
        Ocean-current penalty.

    risk_cost : float, optional
        Hazard or security penalty.

    lane_penalty : float, optional
        Additional traversal penalty used to discourage unnecessary
        lane changes or excessive deviation from the preferred corridor.
    """

    distance: float

    bearing: float

    weather_cost: float = 0.0

    wave_cost: float = 0.0

    current_cost: float = 0.0

    risk_cost: float = 0.0

    lane_penalty: float = 0.0

    @property
    def total_cost(self) -> float:
        """
        Combined routing cost.

        Notes
        -----
        This cost is intended for graph search only.

        The Digital Twin may overwrite or reweight individual
        components during dynamic route recomputation.
        """

        return (
            self.distance
            + self.weather_cost
            + self.wave_cost
            + self.current_cost
            + self.risk_cost
            + self.lane_penalty
        )

    def reset_dynamic_costs(self) -> None:
        """
        Reset all dynamic environmental costs.

        Static geometric quantities such as distance and bearing
        remain unchanged.
        """

        self.weather_cost = 0.0
        self.wave_cost = 0.0
        self.current_cost = 0.0
        self.risk_cost = 0.0


# ======================================================================
# Graph Edge
# ======================================================================

@dataclass(slots=True)
class GraphEdge:
    """
    Directed edge connecting two graph nodes.

    Parameters
    ----------
    source : int
        Source node ID.

    target : int
        Target node ID.

    cost : EdgeCost
        Edge traversal cost model.
    """

    source: int

    target: int

    cost: EdgeCost = field(repr=False)

    @property
    def distance(self) -> float:
        """Return the edge distance (km)."""

        return self.cost.distance

    @property
    def bearing(self) -> float:
        """Return the edge bearing (degrees)."""

        return self.cost.bearing