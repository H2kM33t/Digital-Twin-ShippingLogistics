"""
hazard.py

Moving-storm model and graph hazard field for the TwinRoute-M routing engine.

A storm is a circular hazard that travels along a great circle and whose
severity ramps up, peaks and decays over its lifetime. `HazardField.apply()`
turns the storms (as they will be when the ship arrives) into

    * extra `EdgeCost.risk_cost` on every edge that touches a storm, and
    * temporarily blocked edges through the storm core,

so the unchanged Dijkstra / A* planner reroutes around the storm.

Geometry is done on the sphere in Earth-centred (ECEF) km coordinates, so
it stays correct on ocean-crossing voyages: a storm moving with velocity v
(tangent at its start c0) is at c(t) = R * (c0 + v t) / |c0 + v t|, and
distances are chord lengths (within 0.5 % of the arc up to 2,000 km, far
beyond any storm radius). The maths is vectorized with numpy.

Every penalty is >= 0, so the haversine A* heuristic stays admissible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .graph import NavigationGraph
from .ocean import EARTH_RADIUS_KM, from_ecef, to_ecef

Coordinate = Tuple[float, float]

# A storm blocks edges only when its current severity is at least
# BLOCK_SEVERITY, and only inside CORE_FRACTION of its radius.
BLOCK_SEVERITY = 0.7
CORE_FRACTION = 0.5


# ======================================================================
# Local flat-earth frame (still used by the state estimator)
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
# Sphere helpers
# ======================================================================

def tangent_velocity(lat, lon, heading_deg, speed_kmh) -> np.ndarray:
    """ECEF velocity (km/h) of a body moving on `heading_deg` at `lat, lon`."""

    la, lo, h = map(np.radians, (np.asarray(lat, float), np.asarray(lon, float),
                                 np.asarray(heading_deg, float)))
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    s = np.asarray(speed_kmh, float)[..., None]
    return s * (np.sin(h)[..., None] * east + np.cos(h)[..., None] * north)


def move_on_sphere(c0: np.ndarray, v: np.ndarray, dt) -> np.ndarray:
    """Position after dt hours (see module docstring)."""

    p = c0 + v * np.asarray(dt, float)[..., None]
    return EARTH_RADIUS_KM * p / np.linalg.norm(p, axis=-1, keepdims=True)


def chord_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = (np.asarray(a) + np.asarray(b)) / 2.0
    return EARTH_RADIUS_KM * m / np.linalg.norm(m, axis=-1, keepdims=True)


def latlon_near(p: np.ndarray, lon_ref: float) -> Coordinate:
    """ECEF -> (lat, lon) with the longitude unwrapped next to lon_ref."""

    lat, lon = from_ecef(p)
    lon = float(lon)
    lon += 360.0 * round((lon_ref - lon) / 360.0)
    return float(lat), lon


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

    @property
    def c0(self) -> np.ndarray:
        return to_ecef(self.lat, self.lon)

    @property
    def velocity(self) -> np.ndarray:
        return tangent_velocity(self.lat, self.lon, self.heading_deg, self.speed_kmh)

    def centre(self, t) -> np.ndarray:
        return move_on_sphere(self.c0, self.velocity, t)


# ======================================================================
# Graph geometry cache (numpy arrays, built once per graph)
# ======================================================================

class GraphGeometry:
    """Numpy view of a NavigationGraph in ECEF km."""

    def __init__(self, graph: NavigationGraph, frame: Optional[LocalFrame] = None) -> None:
        self.node_ids = list(graph.nodes.keys())
        index = {nid: i for i, nid in enumerate(self.node_ids)}

        lats = np.array([graph.nodes[n].latitude for n in self.node_ids])
        lons = np.array([graph.nodes[n].longitude for n in self.node_ids])
        self.p = to_ecef(lats, lons)
        self.ncol = np.array([int(graph.nodes[n].x) for n in self.node_ids])

        self.edge_keys: List[Tuple[int, int]] = list(graph.edges.keys())
        self.edge_objs = [graph.edges[k] for k in self.edge_keys]
        self.es = np.array([index[s] for s, _ in self.edge_keys], dtype=int)
        self.et = np.array([index[t] for _, t in self.edge_keys], dtype=int)
        self.edist = np.array([e.cost.distance for e in self.edge_objs])
        self.ecol = self.ncol[self.es] if len(self.es) else np.zeros(0, int)

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

    def __init__(self, storms: List[Storm], frame: Optional[LocalFrame] = None) -> None:
        self.storms = list(storms)
        self._c0 = np.array([s.c0 for s in self.storms]).reshape(-1, 3)
        self._v = np.array([s.velocity for s in self.storms]).reshape(-1, 3)

    # ------------------------------------------------------------------

    def centre_ll(self, i: int, t: float) -> Coordinate:
        return latlon_near(move_on_sphere(self._c0[i], self._v[i], t), self.storms[i].lon)

    def storm_state(self, t: float) -> List[dict]:
        """Where every storm is, and how severe, at time t."""

        out = []
        for i, s in enumerate(self.storms):
            lat, lon = self.centre_ll(i, t)
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
        return [self.centre_ll(i, k * step_h) for k in range(n + 1)]

    def point_risk(self, lat: float, lon: float, t: float, detected_only: bool = True) -> float:
        """Combined risk (0..1) at a point at time t."""

        p = to_ecef(lat, lon)
        keep = 1.0
        for i, s in enumerate(self.storms):
            sev = float(s.severity_at(t))
            if sev <= 0 or (detected_only and t < s.detect_h):
                continue
            d = float(chord_km(p, move_on_sphere(self._c0[i], self._v[i], t)))
            keep *= 1.0 - sev * min(max(1.0 - d / s.radius_km, 0.0), 1.0)
        return 1.0 - keep

    # ------------------------------------------------------------------

    def apply(
        self,
        graph: NavigationGraph,
        geo: GraphGeometry,
        t_now: float,
        ship_ll: Optional[Coordinate],
        speed_kmh: float,
        from_column: int = 0,
        protected_nodes: FrozenSet[int] = frozenset(),
        risk_weight: float = 8.0,
        block: bool = True,
    ) -> HazardSnapshot:
        """
        Write storm risk into the graph for a ship at `ship_ll` at `t_now`.

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
        ps, pt = geo.p[geo.es[idx]], geo.p[geo.et[idx]]
        pm = midpoint(ps, pt)

        if ship_ll is None:
            d_ship = np.zeros(len(idx))
        else:
            d_ship = chord_km(pm, to_ecef(*ship_ll))
        t_arr = t_now + d_ship / max(speed_kmh, 1e-6)

        keep = np.ones(len(idx))
        in_core = np.zeros(len(idx), dtype=bool)

        for i, s in enumerate(self.storms):
            if t_now < s.detect_h:
                continue                      # twin does not know it yet
            sev = s.severity_at(t_arr)
            c = move_on_sphere(self._c0[i], self._v[i], t_arr)

            worst = np.zeros(len(idx))
            for q in (ps, pm, pt):
                d = chord_km(q, c)
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
    centerline: Sequence[Coordinate],
    storms: List[Storm],
    base_width_km: float,
    max_width_km: float,
    horizon_h: float,
) -> float:
    """
    Corridor width needed so that lanes can still pass around the storms
    that come near the route (distance to the route plus radius and
    margin on the far side).
    """

    half = base_width_km / 2.0
    if not storms:
        return min(base_width_km, max_width_km)

    route = to_ecef(np.array([p[0] for p in centerline]), np.array([p[1] for p in centerline]))
    reach = max_width_km / 2.0

    for s in storms:
        ts = np.linspace(0.0, horizon_h, 25)
        c = move_on_sphere(s.c0, s.velocity, ts)                    # (25, 3)
        d = np.min(np.linalg.norm(c[:, None, :] - route[None], axis=-1), axis=1)
        near = d[d < reach + s.radius_km]
        if len(near):
            half = max(half, float(near.min()) + 1.3 * s.radius_km)

    return min(2.0 * half, max_width_km)
