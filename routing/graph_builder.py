"""
graph_builder.py

Graph construction utilities for the TwinRoute-M routing engine.

This module builds a synthetic maritime navigation graph between an
origin and a destination. The graph forms the foundation for
graph-based route planning algorithms (Dijkstra, A*, K-shortest
paths, etc.).

Responsibilities
----------------
1. Generate the centerline: the great circle, or a sea-lane polyline
   from routing/ocean.py resampled to the longitudinal resolution.
2. Expand the centerline into a multi-lane navigation corridor.
3. Convert the corridor into graph nodes (land nodes marked RESTRICTED
   when the land mask is on).
4. Connect neighbouring nodes with directed, forward-only edges that
   stay in the water.
5. Validate graph integrity and connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

from .corridor import NavigationCorridor
from .edge import EdgeCost, GraphEdge
from .graph import NavigationGraph
from .node import GraphNode, NodeType


# ==============================================================
# Constants
# ==============================================================

EARTH_RADIUS_KM = 6371.0


# ==============================================================
# Coordinate Alias
# ==============================================================

Coordinate = Tuple[float, float]


# ==============================================================
# GraphBuilderConfig
# ==============================================================

@dataclass(slots=True)
class GraphBuilderConfig:
    """
    Configuration controlling synthetic graph generation.

    Parameters
    ----------
    longitudinal_resolution
        Number of samples taken along the voyage (centerline points).
    lateral_resolution
        Number of navigation lanes generated across the corridor.
        Must be an odd number greater than or equal to 3 so that a
        well-defined center lane exists.
    corridor_width_km
        Total width of the synthetic navigation corridor, in
        kilometres, measured from the outermost port lane to the
        outermost starboard lane.
    """

    longitudinal_resolution: int = 40
    lateral_resolution: int = 7
    corridor_width_km: float = 120.0
    # Optional sea-lane polyline (see routing/ocean.py). When given, the
    # corridor follows it instead of the great circle.
    centerline_path: Optional[List[Coordinate]] = None
    # Drop nodes and edges on land (1 km global mask). The centre lane is
    # the sea route itself and always stays navigable (canals, port
    # approaches narrower than the mask resolution).
    land_mask: bool = False


# ==============================================================
# GraphBuilder
# ==============================================================

class GraphBuilder:
    """
    Constructs the synthetic navigation graph used by the routing
    engine.

    Workflow
    --------
    build()
        ├── generate_centerline()
        ├── generate_navigation_corridor()
        ├── _generate_graph_nodes()
        ├── _connect_graph()
        ├── validate_graph()
        ├── validate_connectivity()
        └── return (NavigationCorridor, NavigationGraph)
    """

    # ----------------------------------------------------------

    def __init__(self, config: GraphBuilderConfig | None = None) -> None:
        """
        Parameters
        ----------
        config
            Optional graph builder configuration. If omitted, a
            default :class:`GraphBuilderConfig` is used.
        """

        self.config = config or GraphBuilderConfig()

    # ----------------------------------------------------------

    def build(
        self,
        origin: Coordinate,
        destination: Coordinate,
    ) -> Tuple[NavigationCorridor, NavigationGraph]:
        """
        Build a complete synthetic navigation graph.

        Parameters
        ----------
        origin
            Voyage origin coordinate (latitude, longitude).
        destination
            Voyage destination coordinate (latitude, longitude).

        Returns
        -------
        tuple[NavigationCorridor, NavigationGraph]
            The generated navigation corridor and the fully
            connected, validated navigation graph.
        """

        if self.config.centerline_path:
            from .ocean import resample
            centerline = resample(self.config.centerline_path,
                                  self.config.longitudinal_resolution)
        else:
            centerline = self.generate_centerline(origin, destination)

        corridor = self.generate_navigation_corridor(centerline)

        graph = self._generate_graph_nodes(corridor)

        self._water = self._water_mask(corridor) if self.config.land_mask else None
        if self._water is not None:
            for (x, y), node_id in graph.node_lookup.items():
                if not self._water[x, y]:
                    graph.get_node(node_id).node_type = NodeType.RESTRICTED

        self._connect_graph(graph, corridor)

        self.validate_graph(graph)
        self.validate_connectivity(graph)

        return corridor, graph

    # ============================================================
    # Geographic Utilities
    # ============================================================

    @staticmethod
    def haversine_distance(p1: Coordinate, p2: Coordinate) -> float:
        """
        Compute the great-circle distance between two coordinates.

        Parameters
        ----------
        p1
            (latitude, longitude) of the first point.
        p2
            (latitude, longitude) of the second point.

        Returns
        -------
        float
            Distance in kilometres.
        """

        lat1, lon1 = map(radians, p1)
        lat2, lon2 = map(radians, p2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            sin(dlat / 2.0) ** 2
            + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
        )

        c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))

        return EARTH_RADIUS_KM * c

    # ----------------------------------------------------------

    @staticmethod
    def _angular_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Compute the great-circle angular distance, in radians,
        between two points given in radians.

        Parameters
        ----------
        lat1, lon1
            Latitude and longitude of the first point, in radians.
        lat2, lon2
            Latitude and longitude of the second point, in radians.

        Returns
        -------
        float
            Angular distance in radians.
        """

        a = (
            sin((lat2 - lat1) / 2.0) ** 2
            + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2.0) ** 2
        )

        return 2.0 * atan2(sqrt(a), sqrt(1.0 - a))

    # ----------------------------------------------------------

    @staticmethod
    def _compute_bearing(p1: Coordinate, p2: Coordinate) -> float:
        """
        Compute the initial great-circle bearing from p1 to p2.

        Parameters
        ----------
        p1
            (latitude, longitude) of the starting point.
        p2
            (latitude, longitude) of the target point.

        Returns
        -------
        float
            Bearing in degrees, normalised to the range [0, 360).
        """

        lat1, lon1 = map(radians, p1)
        lat2, lon2 = map(radians, p2)

        dlon = lon2 - lon1

        x = sin(dlon) * cos(lat2)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

        bearing = degrees(atan2(x, y))

        return (bearing + 360.0) % 360.0

    # ----------------------------------------------------------

    @staticmethod
    def _destination_point(
        latitude: float,
        longitude: float,
        bearing_deg: float,
        distance_km: float,
    ) -> Coordinate:
        """
        Compute a destination point given a start coordinate, an
        initial bearing, and a travel distance.

        Parameters
        ----------
        latitude
            Start latitude in degrees.
        longitude
            Start longitude in degrees.
        bearing_deg
            Initial bearing in degrees.
        distance_km
            Distance travelled, in kilometres. May be negative to
            indicate travel in the opposite (reciprocal) direction.

        Returns
        -------
        Coordinate
            (latitude, longitude) of the destination point, in
            degrees.
        """

        lat1 = radians(latitude)
        lon1 = radians(longitude)
        bearing = radians(bearing_deg)

        angular_distance = distance_km / EARTH_RADIUS_KM

        lat2 = asin(
            sin(lat1) * cos(angular_distance)
            + cos(lat1) * sin(angular_distance) * cos(bearing)
        )

        lon2 = lon1 + atan2(
            sin(bearing) * sin(angular_distance) * cos(lat1),
            cos(angular_distance) - sin(lat1) * sin(lat2),
        )

        return (degrees(lat2), degrees(lon2))

    # ============================================================
    # Centerline Generation
    # ============================================================

    def generate_centerline(
        self,
        origin: Coordinate,
        destination: Coordinate,
    ) -> List[Coordinate]:
        """
        Generate a great-circle centerline between origin and
        destination using spherical linear interpolation (slerp).

        Parameters
        ----------
        origin
            Starting coordinate (latitude, longitude).
        destination
            Destination coordinate (latitude, longitude).

        Returns
        -------
        list[Coordinate]
            Ordered list of equally spaced centerline coordinates,
            from origin to destination inclusive.
        """

        lat1, lon1 = map(radians, origin)
        lat2, lon2 = map(radians, destination)

        angular_distance = self._angular_distance(lat1, lon1, lat2, lon2)

        n = self.config.longitudinal_resolution

        if angular_distance == 0.0:
            return [origin for _ in range(n)]

        points: List[Coordinate] = []

        for i in range(n):

            t = i / (n - 1)

            a = sin((1.0 - t) * angular_distance) / sin(angular_distance)
            b = sin(t * angular_distance) / sin(angular_distance)

            x = a * cos(lat1) * cos(lon1) + b * cos(lat2) * cos(lon2)
            y = a * cos(lat1) * sin(lon1) + b * cos(lat2) * sin(lon2)
            z = a * sin(lat1) + b * sin(lat2)

            lat = atan2(z, sqrt(x * x + y * y))
            lon = atan2(y, x)

            points.append((degrees(lat), degrees(lon)))

        return points

    # ============================================================
    # Navigation Corridor Generation
    # ============================================================

    def generate_navigation_corridor(
        self,
        centerline: List[Coordinate],
    ) -> NavigationCorridor:
        """
        Generate a synthetic multi-lane navigation corridor by
        offsetting the centerline perpendicular to the local sailing
        direction at each sample point.

        Parameters
        ----------
        centerline
            Great-circle centerline coordinates.

        Returns
        -------
        NavigationCorridor
            The generated corridor, containing one list of
            coordinates per lane.
        """

        total_lanes = self.config.lateral_resolution
        half = (total_lanes - 1) / 2.0

        lane_spacing = (
            self.config.corridor_width_km / (total_lanes - 1)
            if total_lanes > 1
            else 0.0
        )

        bearings = self._compute_local_bearings(centerline)

        lanes: List[List[Coordinate]] = []

        for lane in range(total_lanes):

            offset_km = (lane - half) * lane_spacing

            lane_points: List[Coordinate] = []

            for i, point in enumerate(centerline):

                perpendicular_bearing = bearings[i] + 90.0

                latitude, longitude = self._destination_point(
                    point[0],
                    point[1],
                    perpendicular_bearing,
                    offset_km,
                )

                lane_points.append((latitude, longitude))

            lanes.append(lane_points)

        return NavigationCorridor(
            centerline=centerline,
            lanes=lanes,
            longitudinal_resolution=self.config.longitudinal_resolution,
            lateral_resolution=self.config.lateral_resolution,
            corridor_width_km=self.config.corridor_width_km,
        )

    # ----------------------------------------------------------

    def _compute_local_bearings(
        self,
        centerline: List[Coordinate],
    ) -> List[float]:
        """
        Compute the local sailing bearing at every centerline sample.

        Interior points use the average of the incoming and outgoing
        bearings so that lane offsets remain smooth around turns.

        Parameters
        ----------
        centerline
            Great-circle centerline coordinates.

        Returns
        -------
        list[float]
            Local bearing, in degrees, for every centerline point.
        """

        n = len(centerline)
        if n < 2:
            return [0.0] * n

        # bearing of every leg, as unit vectors so 350 and 10 average to 0
        legs = [radians(self._compute_bearing(a, b))
                for a, b in zip(centerline[:-1], centerline[1:])]
        ux = np.array([sin(b) for b in legs])
        uy = np.array([cos(b) for b in legs])

        # each point: mean of the legs around it, widened on a sea route so
        # lanes do not fold over each other at sharp turns
        window = 3 if self.config.centerline_path else 1
        bearings: List[float] = []
        for i in range(n):
            lo, hi = max(0, i - window), min(n - 1, i + window)
            sx, sy = ux[lo:hi].sum(), uy[lo:hi].sum()
            bearings.append((degrees(atan2(sx, sy)) + 360.0) % 360.0)

        return bearings

    # ----------------------------------------------------------

    def _water_mask(self, corridor: NavigationCorridor) -> np.ndarray:
        """(columns, lanes) bool: node is at sea. Centre lane always is."""

        from .ocean import is_water

        lat = np.array(corridor.lanes)[:, :, 0].T       # (x, y)
        lon = np.array(corridor.lanes)[:, :, 1].T
        water = is_water(lat, lon)
        water[:, corridor.lateral_resolution // 2] = True
        return water

    def _edge_in_water(self, corridor: NavigationCorridor) -> np.ndarray:
        """(columns - 1, lanes, 3) bool for moves to lane y-1, y, y+1."""

        from .ocean import is_water

        P = np.array(corridor.lanes).transpose(1, 0, 2)  # (x, y, 2)
        nx, ny = P.shape[:2]
        centre = ny // 2
        ok = np.zeros((nx - 1, ny, 3), bool)
        for m, dy in enumerate((-1, 0, 1)):
            ys = np.arange(ny)
            yt = ys + dy
            valid = (yt >= 0) & (yt < ny)
            yt = np.clip(yt, 0, ny - 1)
            a, b = P[:-1][:, ys], P[1:][:, yt]           # (nx-1, ny, 2)
            fine = valid[None].repeat(nx - 1, 0)
            for f in (0.25, 0.5, 0.75):
                q = a + f * (b - a)
                fine &= is_water(q[..., 0], q[..., 1])
            fine &= self._water[:-1][:, ys] & self._water[1:][:, yt]
            if dy == 0:
                fine[:, centre] = True                  # the sea route itself
            ok[:, :, m] = fine & valid[None]
        return ok

    # ============================================================
    # Graph Construction
    # ============================================================

    def _generate_graph_nodes(
        self,
        corridor: NavigationCorridor,
    ) -> NavigationGraph:
        """
        Convert every corridor point into a :class:`GraphNode` and
        populate a fresh :class:`NavigationGraph`.

        Parameters
        ----------
        corridor
            The navigation corridor to convert into graph nodes.

        Returns
        -------
        NavigationGraph
            A graph populated with nodes and an empty edge set.
        """

        graph = NavigationGraph()

        longitudinal_steps = corridor.longitudinal_resolution
        lateral_steps = corridor.lateral_resolution

        for x in range(longitudinal_steps):

            progress = (
                x / (longitudinal_steps - 1)
                if longitudinal_steps > 1
                else 0.0
            )

            for y in range(lateral_steps):

                latitude, longitude = corridor.lanes[y][x]

                node_id = self._make_node_id(
                    x,
                    y,
                    corridor.lateral_resolution,
                )

                node = GraphNode(
                    id=node_id,
                    latitude=latitude,
                    longitude=longitude,
                    progress=progress,
                    x=x,
                    y=y,
                )

                graph.add_node(node)
                graph.node_lookup[(x, y)] = node.id

        return graph

    # ----------------------------------------------------------

    @staticmethod
    def _make_node_id(
        x: int,
        y: int,
        lateral_resolution: int,
    ) -> int:
        """
        Generate a deterministic integer node ID.
        """

        return x * lateral_resolution + y

    # ----------------------------------------------------------

    def _connect_graph(
        self,
        graph: NavigationGraph,
        corridor: NavigationCorridor,
    ) -> None:
        """
        Connect graph nodes using forward-only navigation semantics.

        Every node connects to up to three successors at the next
        longitudinal step: forward, forward-left, and forward-right.
        Nodes at the final longitudinal step (destination nodes) have
        no outgoing edges.

        Parameters
        ----------
        graph
            The graph whose nodes have already been generated.
        corridor
            The navigation corridor supplying grid dimensions.
        """

        longitudinal_steps = corridor.longitudinal_resolution
        lateral_steps = corridor.lateral_resolution
        in_water = (self._edge_in_water(corridor)
                    if getattr(self, "_water", None) is not None else None)

        for x in range(longitudinal_steps - 1):

            next_x = x + 1

            for y in range(lateral_steps):

                source_id = graph.node_lookup[(x, y)]

                for target_y in (y, y - 1, y + 1):

                    if target_y < 0 or target_y >= lateral_steps:
                        continue

                    if in_water is not None and not in_water[x, y, target_y - y + 1]:
                        continue

                    target_id = graph.node_lookup[(next_x, target_y)]

                    edge = self._create_edge(
                        graph,
                        source_id,
                        target_id,
                        y,
                        target_y,
                    )

                    graph.add_edge(edge)

    # ----------------------------------------------------------

    def _create_edge(
        self,
        graph: NavigationGraph,
        source_id: str,
        target_id: str,
        source_lane: int,
        target_lane: int,
    ) -> GraphEdge:
        """
        Create a directed edge between two nodes, complete with an
        :class:`EdgeCost`.

        Parameters
        ----------
        graph
            The graph containing both endpoint nodes.
        source_id
            Identifier of the source node.
        target_id
            Identifier of the target node.
        source_lane
            Lane index of the source node.
        target_lane
            Lane index of the target node.

        Returns
        -------
        GraphEdge
            The newly constructed edge.
        """

        source_node = graph.get_node(source_id)
        target_node = graph.get_node(target_id)

        source_coord: Coordinate = (source_node.latitude, source_node.longitude)
        target_coord: Coordinate = (target_node.latitude, target_node.longitude)

        distance = self.haversine_distance(source_coord, target_coord)
        bearing = self._compute_bearing(source_coord, target_coord)

        lateral_steps = self.config.lateral_resolution
        center_lane = (lateral_steps - 1) / 2.0

        deviation_penalty = abs(target_lane - center_lane) * 0.1
        lane_change_penalty = 0.5 if target_lane != source_lane else 0.0

        lane_penalty = deviation_penalty + lane_change_penalty

        cost = EdgeCost(
            distance=distance,
            bearing=bearing,
            lane_penalty=lane_penalty,
        )

        return GraphEdge(
            source=source_id,
            target=target_id,
            cost=cost,
        )

    # ============================================================
    # Validation
    # ============================================================

    def validate_graph(self, graph: NavigationGraph) -> None:
        """
        Validate that the graph contains both nodes and edges.

        Parameters
        ----------
        graph
            The graph to validate.

        Raises
        ------
        ValueError
            If the graph has no nodes or no edges.
        """

        if not graph.nodes:
            raise ValueError("Graph validation failed: graph contains no nodes.")

        if not graph.edges:
            raise ValueError("Graph validation failed: graph contains no edges.")

    # ----------------------------------------------------------

    def validate_connectivity(self, graph: NavigationGraph) -> None:
        """
        Validate that every non-destination node has at least one
        outgoing edge.

        Parameters
        ----------
        graph
            The graph to validate.

        Raises
        ------
        ValueError
            If a non-destination node has no outgoing edges.
        """

        max_x = max(node.x for node in graph.nodes.values())

        outgoing_counts: Dict[str, int] = {
            node_id: 0 for node_id in graph.nodes
        }

        for edge in graph.edges.values():
            outgoing_counts[edge.source] += 1

        if self.config.land_mask:
            # Coastal water nodes may be dead ends (land ahead); planners
            # simply never use them. What must hold is that the centre
            # lane - the sea route - runs unbroken to the destination.
            centre = self.config.lateral_resolution // 2
            for x in range(int(max_x)):
                if not graph.has_edge(graph.node_lookup[(x, centre)],
                                      graph.node_lookup[(x + 1, centre)]):
                    raise ValueError(f"Sea route broken at column {x}.")
            return

        for node_id, node in graph.nodes.items():

            is_destination = node.x == max_x

            if not is_destination and outgoing_counts[node_id] == 0:
                raise ValueError(
                    f"Connectivity validation failed: node '{node_id}' "
                    "has no outgoing edges."
                )

    # ----------------------------------------------------------

    def graph_summary(self, graph: NavigationGraph) -> Dict[str, int]:
        """
        Produce a summary of the graph's size.

        Parameters
        ----------
        graph
            The graph to summarise.

        Returns
        -------
        dict[str, int]
            Dictionary containing the total number of nodes and
            edges in the graph.
        """

        return {
            "nodes": graph.number_of_nodes,
            "edges": graph.number_of_edges,
        }