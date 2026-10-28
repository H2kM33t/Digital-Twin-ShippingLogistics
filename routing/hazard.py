"""
hazard.py

Moving-storm model and graph hazard field for the TwinRoute-M routing engine.

A storm is a circular hazard that travels along a straight track and whose
severity ramps up, peaks and decays over its lifetime. `HazardField.apply()`
turns the storms (as they will be when the ship arrives) into

    * extra `EdgeCost.risk_cost` on every edge that touches a storm, and
    * temporarily blocked edges through the storm core,

so the unchanged Dijkstra / A* planner reroutes around the storm.

All storm geometry is done in a local flat-earth km frame centred on the
voyage (equirectangular projection). That is accurate to a few percent over
voyages of a few thousand km, which is plenty for route planning and keeps
the per-edge maths vectorized with numpy.

Every penalty is >= 0, so the haversine A* heuristic stays admissible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .graph import NavigationGraph

Coordinate = Tuple[float, float]

EARTH_RADIUS_KM = 6371.0

# A storm blocks edges only when its current severity is at least
# BLOCK_SEVERITY, and only inside CORE_FRACTION of its radius.
BLOCK_SEVERITY = 0.7
CORE_FRACTION = 0.5


# ======================================================================
# Local flat-earth frame
# ======================================================================

class LocalFrame:
    """Equirectangular (lat, lon) <-> (x, y) km frame around a reference."""

    def __init__(self, lat0: float, lon0: float) -> None:
        self.lat0 = lat0
        self.lon0 = lon0
        self._kx = EARTH_RADIUS_KM * math.cos(math.radians(lat0)) * math.pi / 180.0
        self._ky = EARTH_RADIUS_KM * math.pi / 180.0

    def to_xy(self, lat, lon):
        return (
            (np.asarray(lon) - self.lon0) * self._kx,
            (np.asarray(lat) - self.lat0) * self._ky,
        )

    def to_latlon(self, x, y) -> Tuple[float, float]:
        return (self.lat0 + y / self._ky, self.lon0 + x / self._kx)


# ======================================================================
# Storm
# ======================================================================

@dataclass
class Storm:
    """
    Circular moving storm.

    Parameters
    ----------
    lat, lon        : storm centre at t = 0 (hours since departure).
    radius_km       : outer radius of the hazard.
    severity        : peak severity, 0..1.
    heading_deg     : direction of travel (0 = north, 90 = east).
    speed_kmh       : travel speed (0 = stationary).
    detect_h        : hour at which the twin first learns about the storm.
                      Before that the planner ignores it (the storm is
                      still physically there); 0 = known from departure.
    lifetime_h      : if set, severity ramps up over the first 25 %, holds,
                      and decays over the last 30 %; zero outside the
                      lifetime. None = constant severity forever.
    """

    lat: float
    lon: float
    radius_km: float = 150.0
    severity: float = 0.9
    heading_deg: float = 0.0
    speed_kmh: float = 0.0
    lifetime_h: Optional[float] = None
    detect_h: float = 0.0
    name: str = "storm"

    def severity_at(self, t):
        """Severity at time t (scalar or numpy array)."""

        t = np.asarray(t, dtype=float)

        if self.lifetime_h is None or self.lifetime_h <= 0:
            return np.full_like(t, self.severity)

        x = t / self.lifetime_h
        envelope = np.where(
            x < 0.25, x / 0.25,
            np.where(x > 0.7, (1.0 - x) / 0.3, 1.0),
        )
        envelope = np.where((x < 0) | (x > 1), 0.0, envelope)

        return self.severity * np.clip(envelope, 0.0, 1.0)


# ======================================================================
# Graph geometry cache (numpy arrays, built once per graph)
# ======================================================================

class GraphGeometry:
    """Numpy view of a NavigationGraph in the local km frame."""

    def __init__(self, graph: NavigationGraph, frame: LocalFrame) -> None:
        self.frame = frame

        self.node_ids = list(graph.nodes.keys())
        index = {nid: i for i, nid in enumerate(self.node_ids)}

        lats = np.array([graph.nodes[n].latitude for n in self.node_ids])
        lons = np.array([graph.nodes[n].longitude for n in self.node_ids])
        self.nx, self.ny = frame.to_xy(lats, lons)
        self.ncol = np.array([int(graph.nodes[n].x) for n in self.node_ids])

        self.edge_keys: List[Tuple[int, int]] = list(graph.edges.keys())
        self.edge_objs = [graph.edges[k] for k in self.edge_keys]
        self.es = np.array([index[s] for s, _ in self.edge_keys])
        self.et = np.array([index[t] for _, t in self.edge_keys])
        self.edist = np.array([e.cost.distance for e in self.edge_objs])
        self.ecol = self.ncol[self.es]

        self._risky: List[int] = []   # edge indices with non-zero risk_cost


# ======================================================================
# Hazard field
# ======================================================================

@dataclass
class HazardSnapshot:
    """Result of applying the storms to the graph at one instant."""

    edge_risk: Dict[Tuple[int, int], float]
    blocked_edges: int


class HazardField:
    """A set of storms plus the maths to project them onto a graph."""

    def __init__(self, storms: List[Storm], frame: LocalFrame) -> None:
        self.storms = list(storms)
        self.frame = frame

        self._x0 = []
        self._y0 = []
        self._vx = []
        self._vy = []

        for s in self.storms:
            x0, y0 = frame.to_xy(s.lat, s.lon)
            h = math.radians(s.heading_deg)
            self._x0.append(float(x0))
            self._y0.append(float(y0))
            self._vx.append(s.speed_kmh * math.sin(h))
            self._vy.append(s.speed_kmh * math.cos(h))

    # ------------------------------------------------------------------

    def storm_state(self, t: float) -> List[dict]:
        """Where every storm is, and how severe, at time t."""

        out = []
        for i, s in enumerate(self.storms):
            lat, lon = self.frame.to_latlon(
                self._x0[i] + self._vx[i] * t,
                self._y0[i] + self._vy[i] * t,
            )
            out.append({
                "name": s.name,
                "lat": lat,
                "lon": lon,
                "radius_km": s.radius_km,
                "severity": float(s.severity_at(t)),
                "detected": t >= s.detect_h,
            })
        return out

    def track(self, i: int, t_end: float, step_h: float = 1.0) -> List[Coordinate]:
        """Centre positions of storm i from t = 0 to t_end."""

        n = max(1, int(math.ceil(t_end / step_h)))
        return [
            self.frame.to_latlon(
                self._x0[i] + self._vx[i] * k * step_h,
                self._y0[i] + self._vy[i] * k * step_h,
            )
            for k in range(n + 1)
        ]

    def point_risk(self, x: float, y: float, t: float) -> float:
        """Combined risk (0..1) at a point of the local frame at time t."""

        keep = 1.0
        for i, s in enumerate(self.storms):
            sev = float(s.severity_at(t))
            if sev <= 0 or t < s.detect_h:
                continue
            d = math.hypot(x - (self._x0[i] + self._vx[i] * t),
                           y - (self._y0[i] + self._vy[i] * t))
            keep *= 1.0 - sev * min(max(1.0 - d / s.radius_km, 0.0), 1.0)
        return 1.0 - keep

    # ------------------------------------------------------------------

    def apply(
        self,
        graph: NavigationGraph,
        geo: GraphGeometry,
        t_now: float,
        ship_xy: Tuple[float, float],
        speed_kmh: float,
        from_column: int = 0,
        protected_nodes: FrozenSet[int] = frozenset(),
        risk_weight: float = 8.0,
        block: bool = True,
    ) -> HazardSnapshot:
        """
        Write storm risk into the graph for a ship at `ship_xy` at `t_now`.

        Look-ahead: the storm is evaluated at the time the ship is
        expected to reach each edge (straight-line distance / speed), so
        the planner avoids where the storm WILL be, not only where it is.
        Edges behind `from_column` are ignored. Edges leaving a node in
        `protected_nodes` are never blocked (the ship must be able to leave
        the node it is heading to).
        """

        graph.clear_blocked()

        for k in geo._risky:
            geo.edge_objs[k].cost.risk_cost = 0.0
        geo._risky = []

        if not self.storms or all(t_now < s.detect_h for s in self.storms):
            return HazardSnapshot({}, 0)

        idx = np.nonzero(geo.ecol >= from_column)[0]
        sx, sy = geo.nx[geo.es[idx]], geo.ny[geo.es[idx]]
        tx, ty = geo.nx[geo.et[idx]], geo.ny[geo.et[idx]]
        mx, my = (sx + tx) / 2.0, (sy + ty) / 2.0

        d_ship = np.hypot(mx - ship_xy[0], my - ship_xy[1])
        t_arr = t_now + d_ship / max(speed_kmh, 1e-6)

        keep = np.ones(len(idx))
        in_core = np.zeros(len(idx), dtype=bool)

        for i, s in enumerate(self.storms):
            if t_now < s.detect_h:
                continue                      # twin does not know it yet
            sev = s.severity_at(t_arr)
            cx = self._x0[i] + self._vx[i] * t_arr
            cy = self._y0[i] + self._vy[i] * t_arr

            worst = np.zeros(len(idx))
            for px, py in ((sx, sy), (mx, my), (tx, ty)):
                d = np.hypot(px - cx, py - cy)
                worst = np.maximum(worst, sev * np.clip(1.0 - d / s.radius_km, 0.0, 1.0))
                in_core |= (d < CORE_FRACTION * s.radius_km) & (sev >= BLOCK_SEVERITY)

            keep *= 1.0 - worst

        risk = 1.0 - keep
        hit = np.nonzero(risk > 1e-6)[0]

        edge_risk: Dict[Tuple[int, int], float] = {}
        blocked = 0

        for h in hit:
            k = int(idx[h])
            edge = geo.edge_objs[k]
            r = float(risk[h])
            edge.cost.risk_cost = risk_weight * r * geo.edist[k]
            geo._risky.append(k)
            edge_risk[geo.edge_keys[k]] = r

            if block and in_core[h] and edge.source not in protected_nodes:
                graph.block_edge(edge.source, edge.target)
                blocked += 1

        return HazardSnapshot(edge_risk, blocked)


# ======================================================================
# Corridor sizing
# ======================================================================

def required_corridor_width_km(
    frame: LocalFrame,
    origin: Coordinate,
    destination: Coordinate,
    storms: List[Storm],
    base_width_km: float,
    max_width_km: float,
    horizon_h: float,
) -> float:
    """
    Corridor width needed so that every lane can still pass around the
    storms near the voyage (storm radius plus margin on the far side).
    """

    ox, oy = frame.to_xy(*origin)
    dx, dy = frame.to_xy(*destination)
    ux, uy = float(dx - ox), float(dy - oy)
    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length

    half = base_width_km / 2.0

    for s in storms:
        x0, y0 = frame.to_xy(s.lat, s.lon)
        h = math.radians(s.heading_deg)

        for t in np.linspace(0.0, horizon_h, 25):
            x = float(x0) + s.speed_kmh * math.sin(h) * t - float(ox)
            y = float(y0) + s.speed_kmh * math.cos(h) * t - float(oy)
            along = x * ux + y * uy
            cross = abs(-x * uy + y * ux)

            if -s.radius_km <= along <= length + s.radius_km:
                half = max(half, cross + 1.3 * s.radius_km)

    return min(2.0 * half, max_width_km)
