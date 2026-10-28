"""
TADIF orchestrator: the closed loop (spec Chapters 13-14).

TadifVoyage sails the same lattice and storms as routing.voyage, but the
twin is now in charge of the ship. Every hour:

    1. Phi_state   UKF filters a noisy GPS fix (the twin plans from it)
    2.             environment step (wind / waves from twin_bridge), true
                   fuel burn observed -> Phi_learn updates theta_fuel
    3.             the current plan is re-scored on the scenario ensemble
    4. Phi_health  Twin Confidence Index from the four live sub-scores
    5. triggers    detectors decide whether to re-plan, and from which stage

and only if a trigger fired, one decision cycle:

    Phi_scenario -> Phi_weight -> Phi_opt (NSGA-III) -> Phi_sim on the full
    ensemble -> Phi_risk (CVaR filter) -> Phi_dec (TOPSIS + PROMETHEE II,
    sensitivity, explanation) -> sail / hold / forced transit

The selected route and speed are what the ship actually sails next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from routing.hazard import HazardField, Storm
from routing.voyage import VoyageConfig, VoyageSimulator, _haversine_km, _round_pt
from twin_bridge import TwinObserver

from .decision import DecisionRanker, criteria_matrix, explanation_text
from .estimation import VesselStateEstimator
from .health import TwinHealthMonitor, decision_confidence, forecast_confidence
from .lattice import Lattice
from .learning import KnowledgeBase, OnlineLearner
from .objectives import AdaptiveWeightBuilder, deadline_pressure, weather_severity
from .optimization import NSGA3, constrained_fronts
from .risk import RiskAssessor, RiskThresholds, bootstrap_ci, weighted_cvar
from .scenarios import ScenarioGenerator
from .simulation import SPEED_LOSS, FuelModel, storm_hazard
from .simulation import VoyageSimulator as BatchSimulator
from .triggers import STAGES, TriggerEngine

Coordinate = Tuple[float, float]
KNOT_KMH = 1.852

# Table 9.1 alpha per mission type
PROFILE_ALPHA = {"cargo_standard": 0.9, "fuel_saving": 0.9, "deadline_critical": 0.9,
                 "emergency": 0.95, "heavy_weather": 0.95}


@dataclass
class TadifConfig(VoyageConfig):
    profile: str = "cargo_standard"
    n_scenarios: int = 48           # N_base (Eq. 9.1)
    n_scenarios_max: int = 128      # N_max
    opt_scenarios: int = 12         # reduced ensemble used inside NSGA-III
    nsga_generations: int = 25
    speed_levels: tuple = (0.75, 0.85, 0.95, 1.0)   # x service speed
    deadline_slack: float = 0.15    # deadline = storm-free time at service speed * (1 + slack)
    deadline_h: Optional[float] = None
    hold_levels_h: tuple = (0, 2, 4, 8, 12, 24)    # wait-then-sail options
    hold_horizon_h: float = 6.0
    switch_margin: float = 0.03     # hysteresis: keep the plan unless clearly beaten
    theta_true: float = 1.12        # real hull/fuel factor (unknown to the twin)
    gps_std_km: float = 0.5
    gps_fault_h: Optional[float] = None
    kb_path: Optional[str] = None
    kb_write: bool = True           # store the finished voyage in the knowledge base
    seed: int = 0


class TadifVoyage(VoyageSimulator):

    def __init__(self, origin: Coordinate, destination: Coordinate,
                 storms: Optional[List[Storm]] = None,
                 config: Optional[TadifConfig] = None) -> None:
        super().__init__(origin, destination, storms, config or TadifConfig())

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _physical_risk(self, xy, t: float) -> float:
        """Hazard the ship really meets (detected or not)."""

        keep = 1.0
        f = self._field
        for i, s in enumerate(self.storms):
            sev = float(s.severity_at(t))
            if sev <= 0:
                continue
            d = math.hypot(xy[0] - (f._x0[i] + f._vx[i] * t), xy[1] - (f._y0[i] + f._vy[i] * t))
            keep *= 1.0 - sev * min(max(1.0 - d / s.radius_km, 0.0), 1.0)
        return 1.0 - keep

    def _astar_lanes(self, t, pos, nxt, col0, mode: str):
        gen, cfg = self.generator, self.cfg
        saved = gen.hazards
        if mode == "direct":
            gen.hazards = None
            gen.apply_storms(t, pos, self.speed_kmh, from_column=col0)
        else:
            gen.apply_storms(t, pos, self.speed_kmh, from_column=col0,
                             protected_nodes=frozenset({nxt}),
                             risk_weight=cfg.risk_weight * (5.0 if mode == "safest" else 1.0),
                             block=(mode == "storm"))
        route = gen.replan_from(nxt)
        gen.hazards = saved
        return None if route is None else self._lat.lanes_of(route.node_ids)

    # ------------------------------------------------------------------
    # one decision cycle
    # ------------------------------------------------------------------

    def _cycle(self, k: int, t: float, est_xy, est_ll, nxt: int, plan: List[int],
               speed: float, ens_full, ens_opt, theta: float, wx: float,
               weights: dict, conf: float, halted_h: float, forced: bool, arb: dict) -> dict:

        cfg, lat, sim = self.cfg, self._lat, self._sim
        col0, lane0 = self._column(nxt), nxt % lat.nlanes
        M = lat.ncols - col0
        L = len(self._levels)
        holds = np.array(cfg.hold_levels_h, float)
        H = len(holds)
        may_hold = not forced and halted_h < cfg.max_halt_h

        # seeds: incumbent, three A* variants, two corridor sweeps
        routes = {}
        if plan and plan[0] == nxt and len(plan) == M:
            routes["incumbent"] = lat.lanes_of(plan)
        for mode in ("storm", "direct", "safest"):
            lanes = self._astar_lanes(t, est_ll, nxt, col0, mode)
            if lanes is not None and len(lanes) == M:
                routes[mode] = lanes
        for name, lane in (("sweep_port", 0), ("sweep_starboard", lat.nlanes - 1)):
            routes[name] = lat.repair(np.full((1, M), float(lane)), col0, lane0)[0]
        direct = routes.get("direct")

        sg = lambda i: i / max(L - 1, 1)
        hg = lambda j: j / max(H - 1, 1)
        seeds = [np.r_[lanes.astype(float), sg(i), 0.0]
                 for lanes in routes.values() for i in range(L)]
        if may_hold:
            for name in ("incumbent", "direct", "storm"):
                if name in routes:
                    seeds += [np.r_[routes[name].astype(float), sg(L - 1), hg(j)]
                              for j in range(1, H)]
        seeds = np.array(seeds)

        def decode(G):
            lanes = lat.repair(G[:, :M], col0, lane0)
            idx = np.rint(np.clip(G[:, M], 0, 1) * (L - 1)).astype(int)
            hidx = np.rint(np.clip(G[:, M + 1], 0, 1) * (H - 1)).astype(int)
            if not may_hold:
                hidx[:] = 0
            return lanes, idx, hidx

        def evaluate(G):
            lanes, idx, hidx = decode(G)
            r = sim.simulate(col0, lanes, self._levels[idx], est_xy, t, ens_opt,
                             theta, wx, self._deadline, holds[hidx])
            F = np.column_stack([r["fuel_t"], r["lateness_h"], r["safety"], r["robustness"]])
            return F, r["nominal_core"].astype(float), np.column_stack([lanes, idx, hidx])

        opt = NSGA3(n_obj=4, partitions=5, n_gen=cfg.nsga_generations, seed=cfg.seed + k)
        G, _, _ = opt.run(evaluate, seeds, lat.nlanes, (L, H))

        # candidate set = final population + every seed, on the FULL ensemble
        lanes_all, idx_all, hidx_all = decode(np.vstack([G, seeds]))
        keys = np.column_stack([lanes_all, idx_all, hidx_all])
        _, uniq = np.unique(keys, axis=0, return_index=True)
        uniq = np.sort(uniq)
        lanes_c, idx_c, hold_c = lanes_all[uniq], idx_all[uniq], holds[hidx_all[uniq]]
        speeds_c = self._levels[idx_c]
        res = sim.simulate(col0, lanes_c, speeds_c, est_xy, t, ens_full, theta, wx,
                           self._deadline, hold_c)
        rk = self._risk.assess(res, ens_full.weights)
        n_eval = opt.n_evals + len(lanes_c)

        def same(i, lanes, spd=None):
            return (np.array_equal(lanes_c[i], lanes) and hold_c[i] == 0
                    and (spd is None or abs(speeds_c[i] - spd) < 1e-6))

        inc = next((i for i in range(len(lanes_c))
                    if "incumbent" in routes and same(i, routes["incumbent"], speed)), None)
        accepted = np.nonzero(rk["accepted"])[0]
        X_all = criteria_matrix(res, rk, np.arange(len(lanes_c)))

        action, ranking, chosen = "sail", None, None
        ranked_idx = np.array([], int)
        if len(accepted):
            fronts = constrained_fronts(X_all[accepted], np.zeros(len(accepted)))
            ranked_idx = accepted[fronts[0]]
            if inc is not None and rk["accepted"][inc] and inc not in ranked_idx:
                ranked_idx = np.r_[ranked_idx, inc]
            ranking = self._ranker.rank(X_all[ranked_idx], weights["weights"])
            chosen = int(ranked_idx[ranking["winner"]])
            if inc is not None and inc in ranked_idx and chosen != inc:
                pos = int(np.nonzero(ranked_idx == inc)[0][0])
                if ranking["scores"][ranking["winner"]] - ranking["scores"][pos] < cfg.switch_margin:
                    chosen = inc
        else:
            hold_cvar = float(weighted_cvar(sim.hold_peak(est_xy, t, cfg.hold_horizon_h, ens_full)[None],
                                            ens_full.weights, self._risk.thr.alpha)[0])
            sailing = np.nonzero(hold_c == 0)[0]
            order = sailing[np.lexsort((rk["cvar"][sailing], res["nominal_core"][sailing]))]
            best = int(order[0])
            if not may_hold:
                action, chosen = "forced", int(sailing[np.argmin(rk["cvar"][sailing])])
            elif hold_cvar < rk["cvar"][best]:
                action = "hold"
            else:
                action, chosen = "least_risk", best

        if action == "sail" and hold_c[chosen] > 0:
            action = "planned_hold"

        # ---------------- decision record (API schema, Section 15.4) ----------
        def cand_view(i, kind):
            coords = [est_ll] + lat.coords(col0, lanes_c[i])
            return {"id": f"c{k}-{i}", "kind": kind,
                    "path": [[round(a, 3), round(b, 3)] for a, b in coords],
                    "speed_kts": round(float(speeds_c[i]), 1),
                    "hold_h": float(hold_c[i]),
                    "fuel_t": round(float(res["fuel_t"][i]), 1),
                    "time_h": round(float(res["time_h"][i]), 1),
                    "km": round(float(res["distance_km"][i]), 0),
                    "cvar": round(float(rk["cvar"][i]), 3),
                    "p_fail": round(float(rk["p_fail"][i]), 3),
                    "p_late": round(float(res["p_late"][i]), 3),
                    "accepted": bool(rk["accepted"][i])}

        shown, views = set(), []

        def show(i, kind):
            if i is not None and i not in shown and len(views) < 7:
                shown.add(i)
                views.append(cand_view(i, kind))

        show(chosen, "selected")
        if ranking is not None and len(ranked_idx) > 1:
            show(int(ranked_idx[ranking["order"][1]]), "runner_up")
        if len(accepted):
            show(int(accepted[np.argmin(res["fuel_t"][accepted])]), "min_fuel")
            show(int(accepted[np.argmin(res["time_h"][accepted])]), "fastest")
        show(int(np.argmin(rk["cvar"])), "min_risk")
        if direct is not None:
            show(next((i for i in range(len(lanes_c)) if same(i, direct, speed)), None), "direct")
        for rej in sorted(rk["rejections"], key=lambda r: -r["margin"])[:2]:
            show(rej["index"], "rejected")

        decision = {
            "cycle_index": k, "t": round(t, 2), "action": action,
            "trigger": arb, "weights": weights,
            "risk_thresholds": rk["thresholds"],
            "n_candidates": int(len(lanes_c)), "n_accepted": int(len(accepted)),
            "n_pareto_ranked": int(len(ranked_idx)), "n_evaluations": int(n_eval),
            "n_scenarios": int(ens_full.n), "n_opt_scenarios": int(ens_opt.n),
            "candidates": views,
            "rejected_by_risk": [{"route_id": f"c{k}-{r['index']}",
                                  "violated_criterion": r["violated_criterion"],
                                  "margin": r["margin"]} for r in rk["rejections"][:20]],
            "n_rejected": len(rk["rejections"]),
        }

        if chosen is not None:
            i = chosen
            coords = lat.coords(col0, lanes_c[i])
            seg = [0.0] + [_haversine_km(a, b) for a, b in zip([est_ll] + coords[:-1], coords)]
            eta, v = t + hold_c[i], speeds_c[i] * KNOT_KMH
            waypoints = []
            for (la, lo), d in zip(coords, seg[1:]):
                eta += d / v
                waypoints.append({"lat": round(la, 4), "lon": round(lo, 4), "eta_h": round(eta, 2)})
            lo_ci, hi_ci = bootstrap_ci(res["peak_by_scenario"][i], ens_full.weights,
                                        self._risk.thr.alpha, seed=k)
            decision["selected_route"] = {
                "route_id": f"c{k}-{i}", "waypoints": waypoints[::3] + waypoints[-1:],
                "speed_profile_knots": [round(float(speeds_c[i]), 1)],
                "hold_h": float(hold_c[i]),
                "objective_vector": {
                    "fuel_t": round(float(res["fuel_t"][i]), 2),
                    "eta_std_h": round(float(res["eta_std_h"][i]), 2),
                    "co2_t": round(float(res["co2_t"][i]), 1),
                    "safety_index": round(1.0 - float(res["safety"][i]), 3),
                    "reliability": round(1.0 - float(res["p_late"][i]), 3),
                    "robustness_cvar": round(float(rk["cvar"][i]), 3),
                    "time_h": round(float(res["time_h"][i]), 1),
                    "distance_km": round(float(res["distance_km"][i]), 0),
                },
                "cvar_ci95": [round(lo_ci, 3), round(hi_ci, 3)],
                "p_fail": round(float(rk["p_fail"][i]), 3),
            }
            decision["plan_nodes"] = lat.nodes_of(col0, lanes_c[i])
            decision["speed_kts"] = float(speeds_c[i])
            decision["hold_h"] = float(hold_c[i])
            decision["cvar"] = float(rk["cvar"][i])
            decision["detour"] = direct is None or not np.array_equal(lanes_c[i], direct)
            decision["incumbent_ok"] = inc is not None and bool(rk["accepted"][inc])
            decision["kept_incumbent"] = inc is not None and chosen == inc
            inc_rej = next((r for r in rk["rejections"] if r["index"] == inc), None)
            decision["incumbent_rejection"] = inc_rej

        if ranking is not None:
            pos = int(np.nonzero(ranked_idx == chosen)[0][0])
            gap = (ranking["scores"][pos] - max(np.delete(ranking["scores"], pos))
                   if len(ranked_idx) > 1 else 1.0)
            decision["explanation"] = {
                "score": round(float(ranking["scores"][pos]), 4),
                "runner_up_gap": round(float(max(gap, 0.0)), 4),
                "dominant_criteria": ranking["dominant"],
                "weight_sensitivity": ranking["weight_sensitivity"],
                "methods_agree": ranking["methods_agree"],
                "twin_confidence_at_decision": round(conf, 3),
                "text": explanation_text("Route selected", ranking["dominant"], max(gap, 0.0),
                                         len(rk["rejections"]), len(ranked_idx)),
            }
            decision["decision_confidence"] = decision_confidence(
                ranking["weight_sensitivity"], max(gap, 0.0))
        else:
            decision["explanation"] = {
                "text": {"hold": "No route passes the risk filter; holding position is safer.",
                         "forced": f"Hold limit ({cfg.max_halt_h:.0f} h) reached; sailing the minimum-CVaR route.",
                         "least_risk": "No route passes the risk filter and holding is riskier; sailing the minimum-CVaR route."}[action],
                "twin_confidence_at_decision": round(conf, 3)}
            decision["decision_confidence"] = 0.2
        return decision

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        cfg: TadifConfig = self.cfg
        gen = self.generator
        frame_ll = gen.frame

        baseline = self.baseline_route()
        baseline_km = sum(_haversine_km(a, b) for a, b in zip(baseline[:-1], baseline[1:]))
        self._field = gen.hazards or HazardField([], frame_ll)
        self._lat = lat = Lattice(gen)
        self._levels = np.array(cfg.speed_levels) * cfg.speed_kts
        alpha = PROFILE_ALPHA.get(cfg.profile, 0.9)
        self._sim = BatchSimulator(lat, FuelModel(), alpha)
        self._risk = RiskAssessor(RiskThresholds(alpha=alpha, cvar_max=cfg.halt_risk))
        self._ranker = DecisionRanker()
        self._deadline = (cfg.deadline_h if cfg.deadline_h else
                          baseline_km / self.speed_kmh * (1.0 + cfg.deadline_slack))

        weights_builder = AdaptiveWeightBuilder(cfg.profile)
        scen = ScenarioGenerator(frame_ll, n=cfg.n_scenarios, seed=cfg.seed + 7)
        health = TwinHealthMonitor()
        triggers = TriggerEngine()
        learner = OnlineLearner(KnowledgeBase(cfg.kb_path))
        env = TwinObserver(self.origin, self.destination)
        rng = np.random.default_rng(cfg.seed + 3)

        pos = self._coord(gen.start_node)
        xy = tuple(float(v) for v in frame_ll.to_xy(*pos))
        dest_xy = frame_ll.to_xy(*self.destination)
        brg = math.atan2(float(dest_xy[0]) - xy[0], float(dest_xy[1]) - xy[1])
        est = VesselStateEstimator(xy, (self.speed_kmh * math.sin(brg), self.speed_kmh * math.cos(brg)),
                                   gps_std_km=cfg.gps_std_km, gps_fault_h=cfg.gps_fault_h,
                                   seed=cfg.seed + 11)

        plan: List[int] = [gen.start_node]
        speed = cfg.speed_kts
        t, sailed_km, halted_h = 0.0, 0.0, 0.0
        forced = detour = False
        ens_full = ens_opt = None
        decided_cvar, dec_conf, last_dec = 0.0, 1.0, None
        fuel_err, fuel_used, fuel_pred = 0.0, 0.0, 0.0
        frames, events, decisions = [], [], []
        last_status = None
        status, reason = "CRUISING", ""
        first = True
        k = 0
        u_cmd = None
        hold_left = 0.0                 # hours left of the current planned wait

        for _ in range(int(cfg.max_hours / cfg.dt_h)):
            nxt = plan[0]

            # 1. state estimation ------------------------------------------
            snap = est.step(xy, t, 0.0 if first else cfg.dt_h, u_cmd)
            est_xy = est.position
            est_ll = frame_ll.to_latlon(*est_xy)

            # 2. environment -------------------------------------------------
            storms_now = self._field.storm_state(t)
            twin, feel = env.environment_step(pos, storms_now, speed)
            wx_sev = weather_severity(twin.environment.weather.wind_speed,
                                      twin.environment.ocean.wave_height,
                                      twin.environment.weather.visibility)
            wx = 1.0 + 0.3 * wx_sev            # weather resistance on top of storm hazard

            # 3. forecast + plan check ---------------------------------------
            known = [s.name for s in self.storms if t >= s.detect_h]
            if ens_full is None:
                ens_full = scen.sample_forecast(self.storms, t)
                ens_opt = scen.cluster_ensemble(ens_full, cfg.opt_scenarios)
            waiting = status == "HALTED"
            plan_eval = self._sim.simulate(self._column(nxt), lat.lanes_of(plan)[None], [speed],
                                           est_xy, t, ens_full, learner.theta, wx, self._deadline,
                                           hold_left if waiting else 0.0)
            plan_cvar = float(weighted_cvar(plan_eval["peak_by_scenario"], ens_full.weights,
                                            self._risk.thr.alpha)[0])
            plan_core = bool(plan_eval["nominal_core"][0])

            # 4. health ------------------------------------------------------
            h = health.tick(t, est.confidence(),
                            forecast_confidence(ens_full.spread_km, t - ens_full.issued_h),
                            learner.confidence(), dec_conf)

            # 5. triggers ----------------------------------------------------
            fired = triggers.detect(t, first=first, storms_known=known, plan_cvar=plan_cvar,
                                    decided_cvar=decided_cvar,
                                    # a forced transit crosses the core by decision
                                    plan_core=plan_core and not forced,
                                    sensor_conf=h["subscores"]["sensor"], fuel_err=fuel_err,
                                    halted=waiting and hold_left <= 0,
                                    health_trigger=h["trigger"])
            fuel_err = 0.0
            arb = TriggerEngine.arbitrate(fired)
            cycle = arb is not None and self._column(nxt) < lat.ncols - 1

            if arb and "sensor_drift" in arb["names"]:
                est.inflate_noise()
            if cycle:
                if STAGES.index(arb["reentry"]) <= STAGES.index("scenario"):
                    n = health.scenario_count(cfg.n_scenarios, cfg.n_scenarios_max, h["confidence"])
                    ens_full = scen.sample_forecast(self.storms, t, n)
                    ens_opt = scen.cluster_ensemble(ens_full, cfg.opt_scenarios)
                self._risk.set_conservatism(health.conservatism())
                weights = weights_builder.build_weights(
                    deadline_pressure(t, self._deadline,
                                      self._remaining_km(plan, pos) / self.speed_kmh),
                    wx_sev, h["confidence"])

                d = self._cycle(k, t, est_xy, est_ll, nxt, plan, speed, ens_full, ens_opt,
                                learner.theta, wx, weights, h["confidence"], halted_h, forced, arb)
                k += 1
                decisions.append(d)
                last_dec = len(decisions) - 1
                dec_conf = float(d["decision_confidence"])
                status, reason, plan, speed, forced, detour = self._apply(
                    d, plan, speed, forced, detour, halted_h, first)
                if status != "HALTED":
                    halted_h = 0.0
                # planned wait: sleep until it expires (other triggers still
                # wake the cycle); fallback hold: re-check every hour
                hold_left = {"planned_hold": d.get("hold_h", 0.0), "hold": cfg.dt_h}.get(
                    d["action"], 0.0)
                decided_cvar = d.get("cvar", plan_cvar)
            elif status != "HALTED":
                status = "FORCED" if forced else ("DEVIATING" if detour else "CRUISING")
                reason = "Following plan; no trigger fired." if not fired else "Final approach."

            # 6. record ------------------------------------------------------
            ship_risk = self._physical_risk(xy, t)
            fr = {
                "t": round(t, 2), "ship": _round_pt(pos), "ship_est": _round_pt(est_ll),
                "status": status, "reason": reason,
                "ship_risk": round(ship_risk, 3), "plan_risk": round(plan_cvar, 3),
                "sailed_km": round(sailed_km, 1), "speed_kts": round(speed, 1),
                "storms": [{**s, "lat": round(s["lat"], 4), "lon": round(s["lon"], 4),
                            "severity": round(s["severity"], 3)} for s in storms_now],
                "path": [_round_pt(pos)] + [_round_pt(self._coord(n)) for n in plan],
                "triggers": [tr.name for tr in fired], "cycle": cycle,
                "decision_id": last_dec,
                "twin": {
                    "confidence": h["confidence"], "band": h["band"],
                    "subscores": h["subscores"], "escalation": h["escalation"],
                    "wind_kt": round(twin.environment.weather.wind_speed, 1),
                    "wave_m": round(twin.environment.ocean.wave_height, 1),
                    "visibility_km": round(twin.environment.weather.visibility, 1),
                    "storm_feel": round(feel, 3),
                    "theta_fuel": round(learner.mu, 3), "theta_sd": round(learner.sd, 3),
                    "fuel_used_t": round(fuel_used, 1), "estimation": snap,
                    "scenarios": ens_full.n, "forecast_spread_km": round(ens_full.spread_km, 0),
                },
            }
            frames.append(fr)
            if status != last_status:
                events.append({"t": round(t, 2), "status": status, "reason": reason,
                               "frame": len(frames) - 1})
                last_status = status
            first = False

            # 7. move (truth) + fuel observation -----------------------------
            if status == "HALTED":
                model_t = float(self._sim.fuel.rate_tph(4.0, ship_risk)) * cfg.dt_h
                halted_h += cfg.dt_h
                hold_left = max(0.0, hold_left - cfg.dt_h)
                u_cmd = (0.0, 0.0)
            else:
                # what the twin expects (ordered speed, its own forecast hazard)
                # -> control input of the UKF; then what really happens
                exp_hz = float(storm_hazard(np.array([est_xy[0]]), np.array([est_xy[1]]),
                                            np.array([t]), ens_full)[0][0, 0])
                exp_pos, _, _ = self._advance(pos, plan, speed * KNOT_KMH *
                                              (1.0 - SPEED_LOSS * exp_hz) * cfg.dt_h)
                ex, ey = frame_ll.to_xy(*exp_pos)
                u_cmd = ((float(ex) - xy[0]) / cfg.dt_h, (float(ey) - xy[1]) / cfg.dt_h)
                pos, plan, km = self._advance(pos, plan, speed * KNOT_KMH *
                                              (1.0 - SPEED_LOSS * ship_risk) * cfg.dt_h)
                sailed_km += km
                model_t = float(self._sim.fuel.rate_tph(speed, ship_risk, wx)) * cfg.dt_h
            realized = model_t * cfg.theta_true * (1.0 + rng.normal(0.0, 0.03))
            fuel_used += realized
            fuel_pred += model_t * learner.theta
            learner.observe(realized, model_t)
            fuel_err = learner.trend_error()
            xy = tuple(float(v) for v in frame_ll.to_xy(*pos))
            t += cfg.dt_h

            if not plan:
                frames.append({
                    "t": round(t, 2), "ship": _round_pt(pos), "ship_est": _round_pt(pos),
                    "status": "ARRIVED", "reason": "Destination reached.", "ship_risk": 0.0,
                    "plan_risk": 0.0, "sailed_km": round(sailed_km, 1), "speed_kts": 0.0,
                    "storms": [{**s, "lat": round(s["lat"], 4), "lon": round(s["lon"], 4),
                                "severity": round(s["severity"], 3)}
                               for s in self._field.storm_state(t)],
                    "triggers": [], "cycle": False, "decision_id": last_dec,
                    "twin": frames[-1]["twin"],
                })
                events.append({"t": round(t, 2), "status": "ARRIVED",
                               "reason": "Destination reached.", "frame": len(frames) - 1})
                break

        halted_hours = sum(cfg.dt_h for f in frames if f["status"] == "HALTED")
        summary = {
            "arrived": frames[-1]["status"] == "ARRIVED",
            "total_hours": round(t, 1), "sailed_km": round(sailed_km, 1),
            "baseline_km": round(baseline_km, 1),
            "extra_km": round(sailed_km - baseline_km, 1),
            "halted_hours": round(halted_hours, 1),
            "corridor_km": round(self.corridor_km, 0),
            "graph": gen.graph.graph_summary(),
            "deadline_h": round(self._deadline, 1),
            "on_time": frames[-1]["status"] == "ARRIVED" and t <= self._deadline,
            "fuel_used_t": round(fuel_used, 1),
            "fuel_predicted_t": round(fuel_pred, 1),
            "theta_fuel": round(learner.mu, 3), "theta_true": cfg.theta_true,
            "decision_cycles": len(decisions),
            "evaluations": int(sum(d["n_evaluations"] for d in decisions)),
            "profile": cfg.profile,
        }
        summary["theta_prior"] = round(learner.kb.prior[0], 3)
        if summary["arrived"] and cfg.kb_write:
            learner.finish_voyage(summary)

        return {
            "baseline": [_round_pt(c) for c in baseline],
            "frames": frames, "events": events, "decisions": decisions,
            "trigger_log": triggers.log,
            "storm_tracks": [[_round_pt(p) for p in self._field.track(i, t + cfg.dt_h, cfg.dt_h)]
                             for i in range(len(self.storms))],
            "summary": summary,
        }

    # ------------------------------------------------------------------

    def _advance(self, pos: Coordinate, plan: List[int], km: float):
        """Sail `km` along the plan. Returns (pos, remaining plan, km sailed);
        the plan is empty once the last node is reached."""

        plan, sailed = list(plan), 0.0
        while km > 1e-9 and plan:
            target = self._coord(plan[0])
            dkm = _haversine_km(pos, target)
            if dkm <= km:
                pos, sailed, km = target, sailed + dkm, km - dkm
                plan = plan[1:]
            else:
                f = km / dkm
                pos = (pos[0] + f * (target[0] - pos[0]), pos[1] + f * (target[1] - pos[1]))
                sailed += km
                km = 0.0
        return pos, plan, sailed

    def _remaining_km(self, plan: List[int], pos: Coordinate) -> float:
        pts = [pos] + [self._coord(n) for n in plan]
        return sum(_haversine_km(a, b) for a, b in zip(pts[:-1], pts[1:]))

    def _apply(self, d: dict, plan, speed, forced, detour, halted_h, first):
        """Turn a decision into the ship's status, reason, plan and speed."""

        lead = d["trigger"]["lead"].replace("_", " ")
        act = d["action"]
        if act == "hold":
            return ("HALTED", f"Holding position ({lead}): " + d["explanation"]["text"],
                    plan, speed, forced, detour)

        new_plan, new_speed = d["plan_nodes"], d["speed_kts"]
        if act == "planned_hold":
            return ("HALTED", f"Waiting {d['hold_h']:.0f} h for the storm to pass, then "
                    f"sailing at {new_speed:.1f} kn ({lead}). " + d["explanation"]["text"],
                    new_plan, new_speed, forced, d["detour"])
        resumed = f"Resuming after {halted_h:.0f} h hold. " if halted_h > 0 else ""
        spd = (f" Speed {speed:.1f} -> {new_speed:.1f} kn." if abs(new_speed - speed) > 1e-6 else "")
        detour = d["detour"]

        if act == "forced":
            return ("FORCED", resumed + d["explanation"]["text"] + spd,
                    new_plan, new_speed, True, detour)
        if act == "least_risk":
            return ("DEVIATING", resumed + d["explanation"]["text"] + spd,
                    new_plan, new_speed, forced, detour)

        text = d["explanation"]["text"]
        if first:
            status = "DEVIATING" if detour else "CRUISING"
            reason = ("Departure plan detours around the forecast storm track. " if detour
                      else "Departure plan. ") + text
        elif new_plan == plan:
            status = "DEVIATING" if detour else "CRUISING"
            reason = resumed + f"Plan confirmed after {lead}." + spd
        elif not d["incumbent_ok"]:
            rej = d.get("incumbent_rejection") or {}
            why = rej.get("violated_criterion", "risk").replace("_", " ")
            status = "DEVIATING"
            reason = resumed + f"Plan failed the risk filter ({why}) after {lead}; re-routed. " + text + spd
        else:
            status = "REJOINING" if not detour else "DEVIATING"
            reason = resumed + f"Better route found after {lead}. " + text + spd
        return status, reason, new_plan, new_speed, False, detour
