"""
Phi_graph helper: a numpy view of the routing lattice.

The navigation graph built by routing/graph_builder.py is a regular lattice
along the sea route: column x runs along the voyage, lane y across the
corridor, and every edge goes from column x to column x + 1 while moving at
most one lane. Edges onto land, or crossing land, do not exist. So any
route from a node to the goal is "one lane per remaining column" using only
existing moves, which lets the optimizer and the simulator work on whole
populations of routes as integer arrays instead of node lists.

    move[x, y, m]     edge (x, y) -> (x+1, y+m-1) exists      m = 0, 1, 2
    reach[x, y]       the goal can still be reached from (x, y)
    repair()          continuous desired lanes -> feasible sea routes
    best_path()       dynamic programming over the lattice (it is a DAG),
                      used for the seed routes instead of A* (~100x faster)
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from routing.hazard import midpoint
from routing.ocean import to_ecef
from routing.route_generator import RouteGenerator

MOVES = np.array([-1, 0, 1])


class Lattice:

    def __init__(self, gen: RouteGenerator) -> None:
        graph = gen.graph
        self.ncols = nc = gen.corridor.longitudinal_resolution
        self.nlanes = nl = gen.corridor.lateral_resolution
        self.goal_lane = int(graph.get_node(gen.goal_node).y)
        self.start_lane = int(graph.get_node(gen.start_node).y)

        lanes = np.array(gen.corridor.lanes)              # (nl, nc, 2)
        self.lat = lanes[:, :, 0].T.copy()               # (nc, nl)
        self.lon = lanes[:, :, 1].T.copy()
        self.p = to_ecef(self.lat, self.lon)              # (nc, nl, 3)

        move = np.zeros((nc - 1, nl, 3), bool)
        for x in range(nc - 1):
            for y in range(nl):
                src = graph.node_lookup[(x, y)]
                for m, dy in enumerate(MOVES):
                    ty = y + dy
                    if 0 <= ty < nl and graph.has_edge(src, graph.node_lookup[(x + 1, ty)]):
                        move[x, y, m] = True
        self.move = move

        reach = np.zeros((nc, nl), bool)
        reach[-1, self.goal_lane] = True
        for x in range(nc - 2, -1, -1):
            for m, dy in enumerate(MOVES):
                ty = np.clip(np.arange(nl) + dy, 0, nl - 1)
                reach[x] |= move[x, :, m] & reach[x + 1, ty]
        self.reach = reach

        # edge geometry: length and midpoint of every move
        self.edge_km = np.full((nc - 1, nl, 3), np.inf)
        self.edge_mid = np.zeros((nc - 1, nl, 3, 3))
        for m, dy in enumerate(MOVES):
            ty = np.clip(np.arange(nl) + dy, 0, nl - 1)
            a, b = self.p[:-1], self.p[1:, ty]
            d = np.linalg.norm(a - b, axis=-1)
            self.edge_km[:, :, m] = np.where(move[:, :, m], d, np.inf)
            self.edge_mid[:, :, m] = midpoint(a, b)

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

    def path_xyz(self, col0: int, lanes: np.ndarray) -> np.ndarray:
        """(P, M) lanes -> (P, M, 3) ECEF points."""
        cols = col0 + np.arange(lanes.shape[1])
        return self.p[cols, lanes]

    # ------------------------------------------------------------------

    def repair(self, desired: np.ndarray, col0: int, lane0: int) -> np.ndarray:
        """
        Turn continuous desired lanes (P, M) into feasible sea routes.

        Column 0 is fixed to `lane0` (the node the ship is heading to). At
        every later column the path takes, among the existing moves that
        still lead to the goal, the one closest to the desired lane. Since
        the current node can reach the goal, such a move always exists.
        """

        desired = np.atleast_2d(desired)
        P, M = desired.shape
        out = np.empty((P, M), dtype=int)
        out[:, 0] = lane0
        rows = np.arange(P)

        for j in range(1, M):
            x = col0 + j - 1
            prev = out[:, j - 1]
            cand = prev[:, None] + MOVES[None]                       # (P, 3)
            ok = self.move[x, prev] & self.reach[x + 1, np.minimum(np.maximum(cand, 0), self.nlanes - 1)]
            cost = np.where(ok, np.abs(cand - desired[:, j:j + 1]), np.inf)
            # no feasible move can only happen from an unreachable start:
            # fall back to staying in lane (keeps the array well formed)
            pick = np.argmin(cost, axis=1)
            out[:, j] = np.where(np.isfinite(cost[rows, pick]), cand[rows, pick], prev)

        return out

    def best_path(self, col0: int, lane0: int, extra: np.ndarray | None = None,
                  lane_penalty_km: float = 2.0) -> np.ndarray:
        """
        Cheapest route from (col0, lane0) to the goal by dynamic programming.

        Edge cost = length + lane_penalty_km per lane change + `extra`
        (same shape as edge_km[col0:], e.g. storm risk; inf = blocked).
        Returns the lane per column, or None if the goal is unreachable.
        """

        nl = self.nlanes
        cost = self.edge_km[col0:] + lane_penalty_km * (MOVES != 0)[None, None]
        if extra is not None:
            cost = cost + extra
        C = cost.shape[0]
        togo = np.full((C + 1, nl), np.inf)
        togo[C, self.goal_lane] = 0.0
        choice = np.zeros((C, nl), int)
        target = np.arange(nl)[:, None] + MOVES[None]                   # (nl, 3)
        inside = (target >= 0) & (target < nl)
        target = np.where(inside, target, 0)
        rows = np.arange(nl)
        for c in range(C - 1, -1, -1):
            tot = cost[c] + np.where(inside, togo[c + 1][target], np.inf)
            choice[c] = tot.argmin(1)
            togo[c] = tot[rows, choice[c]]
        if not np.isfinite(togo[0, lane0]):
            return None
        lanes = [lane0]
        for c in range(C):
            lanes.append(lanes[-1] + int(MOVES[choice[c, lanes[-1]]]))
        return np.array(lanes)
