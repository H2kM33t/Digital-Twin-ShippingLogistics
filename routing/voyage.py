"""
voyage.py

Simulates a ship sailing the navigation graph while storms move across it.

At every time step the twin

    1. projects the storms (as they will be when the ship gets there) onto
       the graph edge costs and blocks the storm cores,
    2. runs A* from the node the ship is heading to,
    3. decides what the ship does:

       CRUISING   keep following the current plan
       DEVIATING  the plan became risky or blocked -> sail the new A* path
       REJOINING  the plan was safe, but a clearly cheaper path opened up
       HALTED     no path keeps risk below the halt threshold -> hold
                  position (re-checked every step; resumes when one opens)
       FORCED     halted longer than `max_halt_h` -> sail the minimum-risk
                  path anyway, ignoring storm-core blocking
       ARRIVED    reached the destination

Every decision is recorded with the reason, and the full frame sequence is
returned so a front end can replay it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .graph_builder import GraphBuilder, GraphBuilderConfig
from .hazard import (
    HazardField,
    Storm,
    LocalFrame,
    required_corridor_width_km,
)
from .route_generator import RouteGenerator

Coordinate = Tuple[float, float]

KNOT_KMH = 1.852


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class VoyageConfig:
    speed_kts: float = 16.0
    dt_h: float = 1.0
    max_hours: float = 400.0

    # graph / corridor
    longitudinal_resolution: int = 50
    lane_spacing_km: float = 40.0
    base_corridor_km: float = 240.0
    max_corridor_km: float = 1600.0

    # decision thresholds
    risk_weight: float = 8.0      # extra cost = weight * risk * distance
    continue_risk: float = 0.15   # plan with peak risk below this is "safe"
    halt_risk: float = 0.55       # best path riskier than this -> hold
    max_halt_h: float = 36.0      # then sail the min-risk path anyway
    rejoin_gain: float = 0.02     # switch back if >= 2 % cheaper


# ======================================================================
# Helpers
# ======================================================================

def _haversine_km(a: Coordinate, b: Coordinate) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _round_pt(p: Coordinate) -> list:
    return [round(p[0], 4), round(p[1], 4)]


# ======================================================================
# VoyageSimulator
# ======================================================================

class VoyageSimulator:
    """Run one voyage against a set of (moving) storms."""

    def __init__(
        self,
        origin: Coordinate,
        destination: Coordinate,
        storms: Optional[List[Storm]] = None,
        config: Optional[VoyageConfig] = None,
    ) -> None:

        self.cfg = config or VoyageConfig()
        self.origin = origin
        self.destination = destination
        self.storms = list(storms or [])

        self.speed_kmh = self.cfg.speed_kts * KNOT_KMH

        self._baseline_nodes: List[int] = []
        self.generator = self._build_generator()

    # ------------------------------------------------------------------

    def _build_generator(self) -> RouteGenerator:
        """Build the graph, widening the corridor to fit nearby storms."""

        c = self.cfg

        frame = LocalFrame(
            (self.origin[0] + self.destination[0]) / 2.0,
            (self.origin[1] + self.destination[1]) / 2.0,
        )

        straight_km = _haversine_km(self.origin, self.destination)
        horizon_h = 2.0 * straight_km / self.speed_kmh

        self.corridor_km = required_corridor_width_km(
            frame, self.origin, self.destination, self.storms,
            c.base_corridor_km, c.max_corridor_km, horizon_h,
        )

        lanes = int(round(self.corridor_km / c.lane_spacing_km)) + 1
        lanes = max(7, lanes if lanes % 2 == 1 else lanes + 1)

        builder = GraphBuilder(GraphBuilderConfig(
            longitudinal_resolution=c.longitudinal_resolution,
            lateral_resolution=lanes,
            corridor_width_km=self.corridor_km,
        ))

        gen = RouteGenerator(builder)
        gen.build_graph(self.origin, self.destination)
        gen.set_storms(self.storms)

        return gen

    # ------------------------------------------------------------------

    def _column(self, node_id: int) -> int:
        return int(self.generator.graph.get_node(node_id).x)

    def _coord(self, node_id: int) -> Coordinate:
        n = self.generator.graph.get_node(node_id)
        return (n.latitude, n.longitude)

    def _path_risk(self, nodes: List[int], edge_risk: dict) -> float:
        """Peak edge risk along a node path."""

        peak = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            peak = max(peak, edge_risk.get((a, b), 0.0))
        return peak

    def _path_cost(self, nodes: List[int]) -> float:
        planner = self.generator.planner
        return planner.compute_path_cost(nodes) if len(nodes) > 1 else 0.0

    def _path_blocked(self, nodes: List[int]) -> bool:
        graph = self.generator.graph
        return any(
            b not in graph.neighbors(a)
            for a, b in zip(nodes[:-1], nodes[1:])
        )

    # ------------------------------------------------------------------

    def _candidates(self, nxt, nxt_col, pos, t, plan, snap) -> List[dict]:
        """
        Three candidate routes from the ship's position to the destination,
        all scored against the SAME storm field (`snap.edge_risk`):

            plan    the route the ship is actually following
            direct  shortest route if the storms were ignored
            safest  A* with a much heavier storm penalty and no blocking
        """

        gen = self.generator

        def pack(name, kind, nodes):
            coords = [pos] + [self._coord(n) for n in nodes]
            km = sum(_haversine_km(a, b) for a, b in zip(coords[:-1], coords[1:]))
            return {
                "name": name, "kind": kind, "nodes": nodes, "coords": coords,
                "peak_risk": round(self._path_risk(nodes, snap.edge_risk), 3),
                "length_km": round(km, 1),
            }

        out = [pack("Ship plan", "plan", list(plan))]

        saved = gen.hazards
        gen.hazards = None
        gen.apply_storms(t, pos, self.speed_kmh, from_column=nxt_col)
        direct = gen.replan_from(nxt)
        gen.hazards = saved

        gen.apply_storms(
            t, pos, self.speed_kmh, from_column=nxt_col,
            protected_nodes=frozenset({nxt}),
            risk_weight=self.cfg.risk_weight * 5.0, block=False,
        )
        safest = gen.replan_from(nxt)

        if direct:
            out.append(pack("Direct (ignores storms)", "direct", list(direct.node_ids)))
        if safest:
            out.append(pack("Safest (max storm avoidance)", "safest", list(safest.node_ids)))

        return out

    def baseline_route(self) -> List[Coordinate]:
        """Storm-free A* route (what the ship would sail on a clear day)."""

        gen = self.generator
        saved = gen.hazards
        gen.hazards = None
        gen.apply_storms(0.0, None, self.speed_kmh)   # reset to static costs

        route = gen.replan_from(gen.start_node)

        gen.hazards = saved
        self._baseline_nodes = list(route.node_ids) if route else []
        return list(route.coordinates) if route else []

    # ------------------------------------------------------------------

    def run(self, observer: Optional[Callable[[dict], dict]] = None) -> dict:
        """
        Sail the voyage. `observer(ctx) -> dict`, if given, is called once
        per frame with the ship / storm / candidate-route context and may
        return extra fields to merge into that frame (the Digital Twin
        pipeline plugs in here).
        """
        gen = self.generator
        cfg = self.cfg

        baseline = self.baseline_route()

        field = gen.hazards or HazardField([], gen.frame)

        # Ship state -----------------------------------------------------
        pos: Coordinate = self._coord(gen.start_node)
        plan: List[int] = []            # nodes still ahead, plan[0] = next node
        t = 0.0
        halted_h = 0.0
        forced = False
        deviated = False                # sailing a hazard detour
        sailed_km = 0.0

        frames: List[dict] = []
        events: List[dict] = []
        last_status = None

        # initial plan
        gen.apply_storms(0.0, pos, self.speed_kmh,
                         protected_nodes=frozenset({gen.start_node}),
                         risk_weight=cfg.risk_weight)
        first = gen.replan_from(gen.start_node)
        if first is None:   # fully blocked at departure: plan the min-risk path
            gen.apply_storms(0.0, pos, self.speed_kmh,
                             protected_nodes=frozenset({gen.start_node}),
                             risk_weight=cfg.risk_weight, block=False)
            first = gen.replan_from(gen.start_node)
        plan = list(first.node_ids)

        steps = int(cfg.max_hours / cfg.dt_h)

        for _ in range(steps):

            nxt = plan[0]
            nxt_col = self._column(nxt)

            # 1. storms -> graph ------------------------------------------
            snap = gen.apply_storms(
                t, pos, self.speed_kmh,
                from_column=nxt_col,
                protected_nodes=frozenset({nxt}),
                risk_weight=cfg.risk_weight,
                block=not forced,
            )

            # 2. best path from the node ahead ---------------------------
            best = gen.replan_from(nxt)
            best_nodes = list(best.node_ids) if best else None

            ship_xy = gen.frame.to_xy(*pos)
            ship_risk = field.point_risk(float(ship_xy[0]), float(ship_xy[1]), t)

            best_risk = (
                self._path_risk(best_nodes, snap.edge_risk) if best_nodes else 1.0
            )

            # 3. decide -------------------------------------------------------
            status = "CRUISING"
            reason = "Path clear of storms."
            new_plan = plan

            old_blocked = self._path_blocked(plan)
            old_risk = self._path_risk(plan, snap.edge_risk)
            old_ok = (not old_blocked) and old_risk < cfg.continue_risk

            # Hold only if sailing on is worse than staying where we are.
            must_hold = (not forced) and (
                best_nodes is None
                or (best_risk >= cfg.halt_risk and ship_risk < best_risk)
            )

            if must_hold and halted_h >= cfg.max_halt_h:
                forced = True
                must_hold = False
                gen.apply_storms(
                    t, pos, self.speed_kmh, from_column=nxt_col,
                    protected_nodes=frozenset({nxt}),
                    risk_weight=cfg.risk_weight, block=False,
                )
                best = gen.replan_from(nxt)
                best_nodes = list(best.node_ids) if best else None
                best_risk = (
                    self._path_risk(best_nodes, snap.edge_risk) if best_nodes else 1.0
                )

            if must_hold:
                status = "HALTED"
                why = ("every route to the destination crosses a storm core"
                       if best_nodes is None
                       else f"best path peak risk {best_risk:.2f} >= {cfg.halt_risk:.2f}")
                reason = f"Holding position: {why}."
                halted_h += cfg.dt_h
            else:
                if best_nodes is None:
                    best_nodes = plan
                    reason = "No alternative path; keeping plan."
                elif forced:
                    status = "FORCED"
                    reason = (f"Halt limit ({cfg.max_halt_h:.0f} h) reached; "
                              f"sailing minimum-risk path (peak risk {best_risk:.2f}).")
                    new_plan = best_nodes
                elif old_ok:
                    gain = (self._path_cost(plan) - self._path_cost(best_nodes))
                    if (best_nodes != plan
                            and gain > cfg.rejoin_gain * max(self._path_cost(plan), 1.0)):
                        status = "REJOINING"
                        reason = f"Storm cleared; cheaper path found (saves {gain:.0f} cost units)."
                        new_plan = best_nodes
                elif best_nodes == plan:
                    if deviated and not old_ok:
                        status = "DEVIATING"
                        reason = (f"Still on detour; best available path "
                                  f"(peak risk {best_risk:.2f}).")
                    else:
                        reason = (f"Only viable path; keeping plan "
                                  f"(peak risk {best_risk:.2f}).")
                else:
                    status = "DEVIATING"
                    cause = ("blocked by storm core" if old_blocked
                             else f"peak risk {old_risk:.2f} on current plan")
                    reason = (f"Plan {cause}; new path peak risk {best_risk:.2f}.")
                    new_plan = best_nodes

                if status == "DEVIATING":
                    deviated = True
                elif status in ("CRUISING", "REJOINING") and old_ok:
                    deviated = False

                if halted_h > 0 and not forced:
                    reason = f"Resuming after {halted_h:.0f} h hold. " + reason
                    halted_h = 0.0

            if (t == 0.0 and status == "CRUISING"
                    and plan != self._baseline_nodes):
                status = "DEVIATING"
                reason = "Departure plan detours around the forecast storm track."

            plan = new_plan

            # candidate routes + optional observer -------------------------
            cands = self._candidates(nxt, nxt_col, pos, t, plan, snap)

            # record frame -----------------------------------------------
            path_coords = [pos] + [self._coord(n) for n in plan]
            frame = {
                "t": round(t, 2),
                "ship": _round_pt(pos),
                "status": status,
                "reason": reason,
                "ship_risk": round(ship_risk, 3),
                "plan_risk": round(best_risk if best_nodes else old_risk, 3),
                "sailed_km": round(sailed_km, 1),
                "storms": [
                    {**s, "lat": round(s["lat"], 4), "lon": round(s["lon"], 4),
                     "severity": round(s["severity"], 3)}
                    for s in field.storm_state(t)
                ],
            }
            frame["path"] = [_round_pt(c) for c in path_coords]

            if observer is not None:
                frame.update(observer({
                    "t": t, "pos": pos, "status": status,
                    "ship_risk": ship_risk, "storms": field.storm_state(t),
                    "candidates": cands, "speed_kts": cfg.speed_kts,
                }))

            frame["candidates"] = [
                {"name": c["name"], "kind": c["kind"],
                 "peak_risk": c["peak_risk"], "length_km": c["length_km"],
                 "path": None if c["kind"] == "plan"
                         else [[round(a, 3), round(b, 3)] for a, b in c["coords"]]}
                for c in cands
            ]
            frames.append(frame)

            if status != last_status:
                events.append({
                    "t": round(t, 2), "status": status,
                    "reason": reason, "frame": len(frames) - 1,
                })
                last_status = status

            # 4. move -------------------------------------------------------
            if status == "HALTED":
                t += cfg.dt_h
                continue

            remaining = self.speed_kmh * cfg.dt_h
            while remaining > 1e-9 and plan:
                target = self._coord(plan[0])
                d = _haversine_km(pos, target)
                if d <= remaining:
                    pos = target
                    sailed_km += d
                    remaining -= d
                    plan = plan[1:]
                else:
                    f = remaining / d
                    pos = (pos[0] + f * (target[0] - pos[0]),
                           pos[1] + f * (target[1] - pos[1]))
                    sailed_km += remaining
                    remaining = 0.0

            t += cfg.dt_h

            if not plan:
                frames.append({
                    "t": round(t, 2), "ship": _round_pt(pos), "status": "ARRIVED",
                    "reason": "Destination reached.", "ship_risk": 0.0,
                    "plan_risk": 0.0, "sailed_km": round(sailed_km, 1),
                    "storms": [
                        {**s, "lat": round(s["lat"], 4), "lon": round(s["lon"], 4),
                         "severity": round(s["severity"], 3)}
                        for s in field.storm_state(t)
                    ],
                })
                events.append({"t": round(t, 2), "status": "ARRIVED",
                               "reason": "Destination reached.",
                               "frame": len(frames) - 1})
                break

        # summary ---------------------------------------------------------
        baseline_km = sum(
            _haversine_km(a, b) for a, b in zip(baseline[:-1], baseline[1:])
        )
        halted_frames = sum(1 for f in frames if f["status"] == "HALTED")

        return {
            "baseline": [_round_pt(c) for c in baseline],
            "frames": frames,
            "events": events,
            "storm_tracks": [
                [_round_pt(p) for p in field.track(i, t + cfg.dt_h, cfg.dt_h)]
                for i in range(len(self.storms))
            ],
            "summary": {
                "arrived": frames[-1]["status"] == "ARRIVED",
                "total_hours": round(t, 1),
                "sailed_km": round(sailed_km, 1),
                "baseline_km": round(baseline_km, 1),
                "extra_km": round(sailed_km - baseline_km, 1),
                "halted_hours": round(halted_frames * cfg.dt_h, 1),
                "corridor_km": round(self.corridor_km, 0),
                "graph": self.generator.graph.graph_summary(),
            },
        }
