"""
Phi_graph helper: a numpy view of the routing lattice.

The navigation graph built by routing/graph_builder.py is a regular lattice:
column x runs along the voyage, lane y across the corridor, and every edge
goes from column x to column x + 1 while moving at most one lane. So any
route from a node to the goal is just "one lane per remaining column", which
lets the optimizer and the simulator work on whole populations of routes as
integer arrays instead of node lists.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from routing.route_generator import RouteGenerator


class Lattice:

    def __init__(self, gen: RouteGenerator) -> None:
        graph = gen.graph
        self.ncols = gen.corridor.longitudinal_resolution
        self.nlanes = gen.corridor.lateral_resolution
        self.goal_lane = int(graph.get_node(gen.goal_node).y)

        lat = np.zeros((self.ncols, self.nlanes))
        lon = np.zeros((self.ncols, self.nlanes))
        for x in range(self.ncols):
            for y in range(self.nlanes):
                n = graph.get_node(graph.node_lookup[(x, y)])
                lat[x, y], lon[x, y] = n.latitude, n.longitude

        self.lat, self.lon = lat, lon
        x, y = gen.frame.to_xy(lat, lon)
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)

    # ------------------------------------------------------------------

    def node_id(self, col: int, lane: int) -> int:
        return col * self.nlanes + lane

    def lanes_of(self, nodes: Sequence[int]) -> np.ndarray:
        return np.array([n % self.nlanes for n in nodes], dtype=int)

    def nodes_of(self, col0: int, lanes: Sequence[int]) -> List[int]:
        return [self.node_id(col0 + j, int(l)) for j, l in enumerate(lanes)]

    def coords(self, col0: int, lanes: Sequence[int]) -> List[tuple]:
        return [(float(self.lat[col0 + j, l]), float(self.lon[col0 + j, l]))
                for j, l in enumerate(lanes)]

    # ------------------------------------------------------------------

    def repair(self, desired: np.ndarray, col0: int, lane0: int) -> np.ndarray:
        """
        Turn continuous desired lanes (P, M) into feasible lane paths.

        Column 0 is fixed to `lane0` (the node the ship is heading to). Each
        later column moves at most one lane, stays inside the corridor and
        stays close enough to the goal lane to still reach it by the last
        column. That is exactly the graph's edge set, so every repaired path
        is a valid route.
        """

        desired = np.atleast_2d(desired)
        P, M = desired.shape
        out = np.empty((P, M), dtype=int)
        out[:, 0] = lane0
        last = self.ncols - 1

        for j in range(1, M):
            r = last - (col0 + j)                     # columns left after this one
            lo = np.maximum.reduce([out[:, j - 1] - 1,
                                    np.full(P, self.goal_lane - r), np.zeros(P, int)])
            hi = np.minimum.reduce([out[:, j - 1] + 1,
                                    np.full(P, self.goal_lane + r),
                                    np.full(P, self.nlanes - 1)])
            out[:, j] = np.clip(np.rint(desired[:, j]).astype(int), lo, hi)

        return out

    def path_xy(self, col0: int, lanes: np.ndarray):
        """(P, M) lanes -> (P, M) x and y arrays in the km frame."""

        cols = col0 + np.arange(lanes.shape[1])
        return self.x[cols, lanes], self.y[cols, lanes]
