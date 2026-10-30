"""
Phi_sim: batch voyage simulation (spec Section 5.4).

Every candidate route (a lane path plus a speed and a waiting time) is
sailed against every storm scenario at once with numpy broadcasting over

    candidates (P) x scenarios (S) x route segments (M) x storms (K).

Per segment the storms are evaluated where they will be when the ship gets
there (start, middle and end of the segment), giving a 0..1 hazard. The
hazard slows the ship (involuntary speed loss) and raises fuel burn (added
resistance in waves), so storms show up in fuel, ETA and ETA spread, not only
in the risk numbers. The fuel model is the propeller law, burn ~ speed^3,
scaled by the twin's learned hull/fuel factor theta_fuel.

Geometry is ECEF km on the sphere. Distance from a route point p to a storm
centre c(t) = R (c_ref + v lead) / sqrt(R^2 + |v|^2 lead^2) only needs dot
products (|p - c|^2 = 2R^2 - 2 p.c), so the whole tensor is built from two
small matrix products instead of trigonometry per element.

Limitation (documented, not hidden): the arrival time used to place the
storms is the no-slowdown time; one fixed-point pass, not an RK4 integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from routing.hazard import midpoint
from routing.ocean import EARTH_RADIUS_KM

from .lattice import Lattice
from .risk import weighted_cvar
from .scenarios import ScenarioSet

KNOT_KMH = 1.852
CO2_PER_TONNE_FUEL = 3.114        # IMO factor for HFO
BLOCK_SEVERITY = 0.7               # same rule as routing/hazard.py
CORE_FRACTION = 0.5
SPEED_LOSS = 0.35                  # fraction of speed lost at hazard 1.0
ADDED_RESISTANCE = 0.8             # extra fuel fraction at hazard 1.0
HOLD_SPEED_KTS = 4.0               # station keeping / slow heave-to while holding
R2 = EARTH_RADIUS_KM ** 2


@dataclass
class FuelModel:
    ref_speed_kts: float = 16.0
    ref_burn_tph: float = 1.6

    def rate_tph(self, speed_kts, hazard=0.0, wx=1.0, theta=1.0):
        return (self.ref_burn_tph * (np.asarray(speed_kts) / self.ref_speed_kts) ** 3
                * (1.0 + ADDED_RESISTANCE * np.asarray(hazard)) * wx * theta)


def _sev(ens: ScenarioSet, tt):
    life = ens.lifetime[:, None, :]
    x = tt / life
    env = np.where(x < 0.25, x / 0.25, np.where(x > 0.7, (1.0 - x) / 0.3, 1.0))
    env = np.where((x < 0) | (x > 1), 0.0, env)
    env = np.where(np.isinf(life), 1.0, env)
    return np.clip(ens.peak_sev[:, None, :] * ens.sev_scale[:, None, :]
                   * np.clip(env, 0, 1), 0.0, 1.0)


def storm_hazard(p, t, ens: ScenarioSet):
    """
    Hazard of points p (..., M, 3) at times t (..., M) under every scenario.
    Returns (risk, core) with shape (..., S, M).
    """

    p = np.asarray(p, float)
    t = np.asarray(t, float)
    lead_shape = np.broadcast_shapes(p.shape[:-1], t.shape)
    if ens.n_storms == 0:
        shape = lead_shape[:-1] + (ens.n, lead_shape[-1])
        return np.zeros(shape), np.zeros(shape, bool)

    pc = np.einsum("...mi,ski->...smk", p, ens.c_ref)
    pv = np.einsum("...mi,ski->...smk", p, ens.v)
    tt = t[..., None, :, None]
    lead = tt - ens.t_ref[:, None, :]
    vv = (ens.v ** 2).sum(-1)[:, None, :]
    dot = EARTH_RADIUS_KM * (pc + lead * pv) / np.sqrt(R2 + lead ** 2 * vv)
    d = np.sqrt(np.maximum(2.0 * R2 - 2.0 * dot, 0.0))

    R = ens.radius[:, None, :]
    sev = _sev(ens, tt)
    r_k = sev * np.clip(1.0 - d / R, 0.0, 1.0)
    core = ((d < CORE_FRACTION * R) & (sev >= BLOCK_SEVERITY)).any(-1)
    risk = 1.0 - np.prod(1.0 - r_k, axis=-1)
    return risk, core


class VoyageSimulator:
    """Batch evaluator: candidates x scenarios -> objective arrays."""

    def __init__(self, lattice: Lattice, fuel: FuelModel | None = None,
                 alpha: float = 0.9, horizon_h: float = 400.0) -> None:
        self.lat = lattice
        self.fuel = fuel or FuelModel()
        self.alpha = alpha
        self.horizon_h = horizon_h

    def relevant_columns(self, ens: ScenarioSet) -> np.ndarray:
        """
        (ncols,) bool: lattice columns that any scenario storm can reach at
        any time in the voyage horizon (storm radius + corridor half width +
        6 h of storm motion, checked every 6 h). Hazard elsewhere is exactly
        zero, so simulate() skips those segments. Cached on the ensemble.
        """

        cached = getattr(ens, "_relevant", None)
        if cached is not None:
            return cached
        lat = self.lat
        if ens.n_storms == 0:
            mask = np.zeros(lat.ncols, bool)
        else:
            step = 6.0
            ts = ens.t_obs + np.arange(0.0, self.horizon_h + step, step)
            centres = ens.centres(ts[:, None, None]).reshape(-1, 3)     # (T*S*K, 3)
            speed = np.linalg.norm(ens.v, axis=-1)                       # (S, K)
            half = 0.5 * np.max(np.linalg.norm(lat.p[:, 0] - lat.p[:, -1], axis=-1))
            reach = np.broadcast_to(ens.radius + step * speed + half,
                                    (len(ts),) + speed.shape).reshape(-1)
            col = lat.p[:, lat.nlanes // 2]                              # (C, 3)
            mask = np.zeros(lat.ncols, bool)
            for c0 in range(0, lat.ncols, 32):                           # chunked: bounded memory
                dot = col[c0:c0 + 32] @ centres.T
                d = np.sqrt(np.maximum(2.0 * R2 - 2.0 * dot, 0.0))
                mask[c0:c0 + 32] = np.any(d < reach, axis=1)
            # a column next to a relevant one shares its edges
            mask = mask | np.r_[mask[1:], False] | np.r_[False, mask[:-1]]
        ens._relevant = mask
        return mask

    def simulate(self, col0: int, lanes: np.ndarray, speed_kts: np.ndarray,
                 pos_p, t_now: float, ens: ScenarioSet, theta: float = 1.0,
                 wx: float = 1.0, deadline_h: float = np.inf, hold_h=0.0) -> dict:
        """pos_p: ship position (ECEF km). hold_h: hours each candidate waits
        at the ship's position before sailing."""

        lanes = np.atleast_2d(lanes)
        P = lanes.shape[0]
        speed_kts = np.broadcast_to(np.asarray(speed_kts, float), (P,))
        hold = np.broadcast_to(np.asarray(hold_h, float), (P,))
        pos_p = np.asarray(pos_p, float)
        pts = np.concatenate([np.broadcast_to(pos_p, (P, 1, 3)),
                              self.lat.path_xyz(col0, lanes)], axis=1)  # (P, M+1, 3)
        a, b = pts[:, :-1], pts[:, 1:]
        d = np.linalg.norm(b - a, axis=-1)                                # (P, M)
        v = (speed_kts * KNOT_KMH)[:, None]

        # waiting phase: hazard at the holding position over [t, t + hold]
        th = t_now + hold[:, None] * np.linspace(0.0, 1.0, 5)[None]      # (P, 5)
        h_risk, h_core = storm_hazard(np.broadcast_to(pos_p, th.shape + (3,)), th, ens)
        h_core = h_core & (hold > 0)[:, None, None]    # sailing on is not "holding in a core"

        t_end = t_now + hold[:, None] + np.cumsum(d, axis=1) / v
        t_start = t_end - d / v

        # only segments a storm can reach (relevant_columns) are evaluated
        M = d.shape[1]
        ends = col0 + np.arange(M)
        rel = self.relevant_columns(ens)
        seg = np.nonzero(rel[np.maximum(ends - 1, 0)] | rel[ends])[0]
        risk = np.zeros((P, ens.n, M))
        core = np.zeros((P, ens.n, M), bool)
        if len(seg):
            for q, tt in ((a, t_start), (midpoint(a, b), (t_start + t_end) / 2), (b, t_end)):
                r, c = storm_hazard(q[:, seg], tt[:, seg], ens)          # (P, S, m)
                risk[:, :, seg] = np.maximum(risk[:, :, seg], r)
                core[:, :, seg] |= c

        seg_h = d[:, None, :] / (v[:, None, :] * (1.0 - SPEED_LOSS * risk))
        eta = t_now + hold[:, None] + seg_h.sum(-1)                       # (P, S)
        rate = self.fuel.rate_tph(speed_kts[:, None, None], risk, wx, theta)
        fuel = (rate * seg_h).sum(-1)                                     # (P, S)
        fuel = fuel + self.fuel.rate_tph(HOLD_SPEED_KTS, h_risk.mean(-1), wx, theta) * hold[:, None]
        peak = np.maximum(risk.max(-1) if risk.shape[-1] else np.zeros((P, ens.n)),
                          h_risk.max(-1))
        core = np.concatenate([core, h_core], axis=-1)
        dist = d.sum(1)

        w = ens.weights
        eta_mean = eta @ w
        return {
            "distance_km": dist,
            "fuel_t": fuel @ w,
            "co2_t": CO2_PER_TONNE_FUEL * (fuel @ w),
            "time_h": eta_mean - t_now,
            "eta_h": eta_mean,
            "eta_std_h": np.sqrt(np.maximum(((eta - eta_mean[:, None]) ** 2) @ w, 0.0)),
            # ETA deviation from the schedule: RMS of hours past the deadline.
            # Unlike the plain spread it does not reward waiting for a storm
            # to clear while the ship is already late.
            "eta_dev_h": np.sqrt((np.maximum(eta - deadline_h, 0.0) ** 2) @ w),
            "safety": peak @ w,                       # expected peak hazard
            "robustness": weighted_cvar(peak, w, self.alpha),
            "p_core": core.any(-1).astype(float) @ w,
            "p_late": (eta > deadline_h).astype(float) @ w,
            "lateness_h": np.maximum(eta - deadline_h, 0.0) @ w,
            "nominal_core": core[:, 0, :].any(-1) if ens.n else np.zeros(P, bool),
            "peak_by_scenario": peak,
            "core_by_scenario": core.any(-1),
            "speed_kts": speed_kts.copy(),
            "hold_h": hold.copy(),
        }

    def hold_peak(self, pos_p, t_now: float, hours: float, ens: ScenarioSet) -> np.ndarray:
        """Per-scenario peak hazard of holding position for the next `hours`."""

        ts = t_now + np.linspace(0.0, hours, 7)
        risk, _ = storm_hazard(np.broadcast_to(np.asarray(pos_p, float), ts.shape + (3,)),
                               ts, ens)                                   # (S, T)
        return risk.max(-1)

    def edge_hazard(self, col0: int, t_now: float, speed_kts: float, ens: ScenarioSet,
                    scenario: int = 0):
        """
        Hazard and core flag on every lattice move from col0 on, for one
        scenario, at the time the ship would get there (distance along the
        route / speed). Feeds Lattice.best_path for the seed routes.
        """

        mids = self.lat.edge_mid[col0:]                                  # (C, L, 3, 3)
        C, L = mids.shape[:2]
        seg = self.lat.edge_km[col0:, self.lat.nlanes // 2, 1]          # sea-route legs
        along = np.cumsum(seg) - seg / 2.0                               # to each leg's middle
        t = t_now + along / (speed_kts * KNOT_KMH)                       # (C,)
        one = ens.subset(np.array([scenario]), np.array([1.0]))
        risk = np.zeros((C, L, 3))
        core = np.zeros((C, L, 3), bool)
        cols = np.nonzero(self.relevant_columns(ens)[col0:col0 + C])[0]
        if len(cols):
            pts = mids[cols].reshape(len(cols), L * 3, 3)
            tt = np.broadcast_to(t[cols, None], (len(cols), L * 3))
            r, c = storm_hazard(pts, tt, one)                            # (c, 1, L*3)
            risk[cols] = r[:, 0].reshape(len(cols), L, 3)
            core[cols] = c[:, 0].reshape(len(cols), L, 3)
        return risk, core
