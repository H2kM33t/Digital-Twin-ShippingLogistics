from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Set


class NodeType(Enum):
    """Classification of graph nodes."""

    OPEN_OCEAN = "open_ocean"
    PORT = "port"
    WAYPOINT = "waypoint"
    INTERSECTION = "intersection"
    RESTRICTED = "restricted"


@dataclass(slots=True)
class GraphNode:
    """
    Navigation graph node.
    """

    id: int

    latitude: float

    longitude: float

    # Normalized voyage progress (0 → origin, 1 → destination)
    progress: float

    # Local corridor coordinates
    x: float
    y: float

    node_type: NodeType = NodeType.OPEN_OCEAN

    neighbors: Set[int] = field(default_factory=set)

    # ---------------------------------------------------------

    @property
    def coordinate(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)

    # ---------------------------------------------------------

    @property
    def lane(self) -> int:
        """
        Corridor lane index.
        """
        return int(self.y)

    # ---------------------------------------------------------

    def add_neighbor(self, node_id: int) -> None:
        self.neighbors.add(node_id)

    # ---------------------------------------------------------

    def remove_neighbor(self, node_id: int) -> None:
        self.neighbors.discard(node_id)

    # ---------------------------------------------------------

    def __hash__(self) -> int:
        return hash(self.id)

    # ---------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GraphNode("
            f"id={self.id}, "
            f"lat={self.latitude:.4f}, "
            f"lon={self.longitude:.4f}, "
            f"x={self.x:.0f}, "
            f"y={self.y:.0f}, "
            f"progress={self.progress:.2f})"
        )