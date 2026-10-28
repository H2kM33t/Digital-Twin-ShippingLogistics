"""
graph.py

Navigation graph implementation for TwinRoute-M.

The graph stores

- navigation nodes
- graph edges
- adjacency information

and serves as the computational backbone of the routing engine.
"""

from __future__ import annotations

from typing import Dict

from .node import GraphNode
from .edge import GraphEdge, EdgeCost


class NavigationGraph:
    """
    Directed navigation graph.

    The graph stores only the navigation topology.
    Dynamic routing information (weather, currents, risk, etc.)
    is contained inside each edge's EdgeCost object.
    """

    def __init__(self) -> None:
        """Initialize an empty navigation graph."""

        # Node storage
        self.nodes: Dict[int, GraphNode] = {}

        # Directed edges
        self.edges: Dict[tuple[int, int], GraphEdge] = {}

        # Adjacency list
        self.adjacency: Dict[int, set[int]] = {}

        # Internal lookup:
        # (column_index, lane_index) -> node_id
        #
        # Used by GraphBuilder for efficient edge generation.
        self.node_lookup: Dict[tuple[int, int], int] = {}

        # Temporarily blocked edges (source -> {targets}). Blocked edges
        # stay in `edges` / `adjacency` but are hidden from `neighbors()`,
        # so graph search cannot use them until `clear_blocked()` is called.
        self._blocked_out: Dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # Node Operations
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> None:
        """
        Add a graph node.

        Parameters
        ----------
        node : GraphNode
            Node to add.

        Raises
        ------
        ValueError
            If the node already exists.
        """

        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists.")

        self.nodes[node.id] = node
        self.adjacency[node.id] = set()

    def get_node(self, node_id: int) -> GraphNode:
        """Return the requested node."""

        return self.nodes[node_id]

    def has_node(self, node_id: int) -> bool:
        """Return True if the node exists."""

        return node_id in self.nodes

    def remove_node(self, node_id: int) -> None:
        """
        Remove a node and all associated edges.
        """

        if node_id not in self.nodes:
            return

        self.nodes.pop(node_id)
        self.adjacency.pop(node_id, None)

        # Remove attached edges
        remove_edges = [
            edge
            for edge in self.edges
            if node_id in edge
        ]

        for edge in remove_edges:
            self.edges.pop(edge)

        # Remove adjacency references
        for neighbours in self.adjacency.values():
            neighbours.discard(node_id)

        # Remove lookup entries
        remove_lookup = [
            key
            for key, value in self.node_lookup.items()
            if value == node_id
        ]

        for key in remove_lookup:
            self.node_lookup.pop(key)

    # ------------------------------------------------------------------
    # Edge Operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> None:
        """
        Add a directed edge.
        """

        if edge.source not in self.nodes:
            raise KeyError(edge.source)

        if edge.target not in self.nodes:
            raise KeyError(edge.target)

        self.edges[(edge.source, edge.target)] = edge
        self.adjacency[edge.source].add(edge.target)

    def get_edge(self, source: int, target: int) -> GraphEdge:
        """Return the requested edge."""

        return self.edges[(source, target)]

    def has_edge(self, source: int, target: int) -> bool:
        """Return True if an edge exists."""

        return (source, target) in self.edges

    def update_edge_cost(
        self,
        source: int,
        target: int,
        cost: EdgeCost,
    ) -> None:
        """
        Update the traversal cost of an edge.
        """

        self.edges[(source, target)].cost = cost

    def remove_edge(
        self,
        source: int,
        target: int,
    ) -> None:
        """
        Remove an edge.
        """

        self.edges.pop((source, target), None)

        if source in self.adjacency:
            self.adjacency[source].discard(target)

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def neighbors(self, node_id: int) -> set[int]:
        """
        Return neighbouring node IDs (temporarily blocked edges excluded).
        """

        blocked = self._blocked_out.get(node_id)

        if blocked:
            return self.adjacency[node_id] - blocked

        return self.adjacency[node_id]

    def block_edge(self, source: int, target: int) -> None:
        """Hide an edge from graph search until `clear_blocked()`."""

        self._blocked_out.setdefault(source, set()).add(target)

    def clear_blocked(self) -> None:
        """Restore every temporarily blocked edge."""

        self._blocked_out.clear()

    @property
    def number_of_blocked_edges(self) -> int:
        """Number of currently blocked edges."""

        return sum(len(v) for v in self._blocked_out.values())

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all graph contents."""

        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self.node_lookup.clear()
        self._blocked_out.clear()

    def graph_summary(self) -> dict:
        """
        Return summary statistics.

        Returns
        -------
        dict
            Graph statistics.
        """

        return {
            "nodes": self.number_of_nodes,
            "edges": self.number_of_edges,
        }

    # ------------------------------------------------------------------
    # Python Special Methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of nodes."""

        return len(self.nodes)

    def __contains__(self, node_id: int) -> bool:
        """Support: `node_id in graph`."""

        return node_id in self.nodes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def number_of_nodes(self) -> int:
        """Number of graph nodes."""

        return len(self.nodes)

    @property
    def number_of_edges(self) -> int:
        """Number of graph edges."""

        return len(self.edges)