"""
route_generator.py

High-level route generation interface for the TwinRoute-M routing engine.

This module coordinates the complete routing workflow while hiding all
graph implementation details from the rest of the project.

Workflow
--------
Origin/Destination
        │
        ▼
GraphBuilder
        │
        ▼
NavigationGraph
        │
        ▼
GraphPlanner
        │
        ▼
Route
"""

from __future__ import annotations

from typing import Optional, Tuple

from .corridor import NavigationCorridor
from .graph import NavigationGraph
from .graph_builder import GraphBuilder
from .hazard import (
    GraphGeometry,
    HazardField,
    HazardSnapshot,
    LocalFrame,
    Storm,
)
from .planner import (
    GraphPlanner,
    SearchAlgorithm,
    SearchResult,
)
from .route import Route

Coordinate = Tuple[float, float]


# ============================================================
# RouteGenerator
# ============================================================

class RouteGenerator:
    """
    High-level routing interface.

    This class owns the navigation graph and planner, allowing
    multiple routing algorithms to operate on the same graph
    without rebuilding it.
    """

    def __init__(
        self,
        builder: GraphBuilder | None = None,
    ) -> None:
        """
        Parameters
        ----------
        builder
            Optional GraphBuilder instance.
        """

        self.builder = builder or GraphBuilder()

        self.corridor: Optional[NavigationCorridor] = None

        self.graph: Optional[NavigationGraph] = None

        self.planner: Optional[GraphPlanner] = None

        # Storm support (set up by build_graph / set_storms)
        self.frame: Optional[LocalFrame] = None
        self.geometry: Optional[GraphGeometry] = None
        self.hazards: Optional[HazardField] = None

    # ---------------------------------------------------------
    # Graph Construction
    # ---------------------------------------------------------

    def build_graph(
        self,
        origin: Coordinate,
        destination: Coordinate,
    ) -> None:
        """
        Build the navigation graph for a voyage.

        This method constructs the navigation corridor and graph,
        then initializes the graph planner. The resulting graph
        may be reused for multiple routing algorithms.

        Parameters
        ----------
        origin
            Voyage origin (latitude, longitude).

        destination
            Voyage destination (latitude, longitude).
        """

        # Build corridor and graph
        self.corridor, self.graph = self.builder.build(
            origin,
            destination,
        )

        # Create planner
        self.planner = GraphPlanner(
            self.graph,
        )

        # Local km frame + numpy view of the graph for storm maths
        self.frame = LocalFrame(
            (origin[0] + destination[0]) / 2.0,
            (origin[1] + destination[1]) / 2.0,
        )
        self.geometry = GraphGeometry(self.graph, self.frame)
        self.hazards = None

    # ---------------------------------------------------------
    # Storms and Replanning
    # ---------------------------------------------------------

    def set_storms(self, storms: list[Storm]) -> None:
        """Replace the active storms (an empty list clears them)."""

        if self.frame is None:
            raise RuntimeError(
                "Graph has not been built. Call build_graph() first."
            )

        self.hazards = HazardField(storms, self.frame) if storms else None

    def clear_storms(self) -> None:
        """Remove all storms and restore every edge to its static cost."""

        self.hazards = None
        self.apply_storms(0.0, None, 1.0)

    def apply_storms(
        self,
        t: float,
        ship_position: Optional[Coordinate],
        speed_kmh: float,
        from_column: int = 0,
        protected_nodes: frozenset = frozenset(),
        risk_weight: float = 8.0,
        block: bool = True,
    ) -> HazardSnapshot:
        """
        Update graph edge costs for the storms as seen at time `t` by a
        ship at `ship_position`. With no storms this resets the graph.
        """

        assert self.graph is not None and self.geometry is not None

        field = self.hazards or HazardField([], self.frame)

        ship_xy = (0.0, 0.0)
        if ship_position is not None:
            x, y = self.frame.to_xy(*ship_position)
            ship_xy = (float(x), float(y))

        return field.apply(
            self.graph,
            self.geometry,
            t,
            ship_xy,
            speed_kmh,
            from_column=from_column,
            protected_nodes=protected_nodes,
            risk_weight=risk_weight,
            block=block,
        )

    def replan_from(
        self,
        start_node: int,
        algorithm: SearchAlgorithm = SearchAlgorithm.ASTAR,
    ) -> Optional[Route]:
        """
        Search from `start_node` to the goal on the graph as it is now
        (with whatever storm costs / blocked edges are applied).
        Returns None if no path exists.
        """

        if self.planner is None:
            raise RuntimeError("Graph has not been built. Call build_graph() first.")

        goal = self._select_goal_node()

        try:
            if algorithm == SearchAlgorithm.DIJKSTRA:
                result = self.planner.dijkstra(start_node, goal)
            else:
                result = self.planner.astar(start_node, goal)
        except ValueError:
            return None

        return self._search_result_to_route(result)

    @property
    def start_node(self) -> int:
        """Voyage start node ID."""

        return self._select_start_node()

    @property
    def goal_node(self) -> int:
        """Voyage goal node ID."""

        return self._select_goal_node()

    # ---------------------------------------------------------
    # Route Generation
    # ---------------------------------------------------------

    def generate_route(
        self,
        algorithm: SearchAlgorithm = SearchAlgorithm.ASTAR,
    ) -> Route:
        """
        Generate a navigation route using the selected search algorithm.

        Parameters
        ----------
        algorithm
            Routing algorithm.

        Returns
        -------
        Route
            Generated navigation route.
        """

        if self.graph is None:
            raise RuntimeError(
                "Graph has not been built. Call build_graph() first."
            )

        if self.corridor is None:
            raise RuntimeError(
                "Corridor has not been built. Call build_graph() first."
            )

        if self.planner is None:
            raise RuntimeError(
                "Planner has not been initialized."
            )

        start = self._select_start_node()

        goal = self._select_goal_node()

        if algorithm == SearchAlgorithm.DIJKSTRA:

            result = self.planner.dijkstra(
                start,
                goal,
            )

        elif algorithm == SearchAlgorithm.ASTAR:

            result = self.planner.astar(
                start,
                goal,
            )

        else:

            raise ValueError(
                f"Unsupported algorithm: {algorithm}"
            )

        return self._search_result_to_route(
            result,
        )
    # ---------------------------------------------------------
    # Node Selection
    # ---------------------------------------------------------

    def _select_start_node(
        self,
    ) -> int:
        """
        Return the graph node used as the voyage origin.

        The current implementation selects the centre lane of the
        first longitudinal column.

        Returns
        -------
        int
            Start node ID.
        """

        assert self.graph is not None
        assert self.corridor is not None

        start_column = 0

        centre_lane = (
            self.corridor.lateral_resolution // 2
        )

        return self.graph.node_lookup[
            (
                start_column,
                centre_lane,
            )
        ]
        
    # ---------------------------------------------------------

    def _select_goal_node(
        self,
    ) -> int:
        """
        Return the graph node used as the voyage destination.

        The current implementation selects the centre lane of the
        final longitudinal column.

        Returns
        -------
        int
            Goal node ID.
        """

        assert self.graph is not None
        assert self.corridor is not None

        goal_column = (
            self.corridor.longitudinal_resolution - 1
        )

        centre_lane = (
            self.corridor.lateral_resolution // 2
        )

        return self.graph.node_lookup[
            (
                goal_column,
                centre_lane,
            )
        ]
        
    # ---------------------------------------------------------
    # Route Conversion
    # ---------------------------------------------------------

    def _node_path_to_coordinates(
        self,
        node_ids: list[int],
    ) -> list[Coordinate]:
        """
        Convert a graph node path into geographic coordinates.

        Parameters
        ----------
        node_ids
            Ordered graph node IDs.

        Returns
        -------
        list[Coordinate]
            Ordered geographic coordinates.
        """

        assert self.graph is not None

        coordinates: list[Coordinate] = []

        for node_id in node_ids:

            node = self.graph.get_node(node_id)

            coordinates.append(
                (
                    node.latitude,
                    node.longitude,
                )
            )

        return coordinates
    
    # ---------------------------------------------------------

    def _search_result_to_route(
        self,
        result: SearchResult,
    ) -> Route:
        """
        Convert a SearchResult into a Route object.

        Parameters
        ----------
        result
            Search result returned by GraphPlanner.

        Returns
        -------
        Route
            High-level route object.
        """

        coordinates = self._node_path_to_coordinates(
            result.path,
        )

        return Route(
            node_ids=result.path,
            coordinates=coordinates,
            total_distance=result.statistics.total_distance,
            total_cost=result.statistics.total_cost,
            algorithm=result.algorithm,
            statistics=result.statistics,
        )
        
    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @staticmethod
    def route_summary(
        route: Route,
    ) -> dict:
        """
        Return a compact summary of a route.

        Parameters
        ----------
        route
            Route object.

        Returns
        -------
        dict
            Summary statistics.
        """

        return {
            "algorithm": route.algorithm.value,
            "waypoints": route.number_of_waypoints,
            "distance_km": route.total_distance,
            "total_cost": route.total_cost,
            "expanded_nodes": route.statistics.expanded_nodes,
            "explored_edges": route.statistics.explored_edges,
            "execution_time": route.statistics.execution_time,
        }
        
