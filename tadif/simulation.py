"""
Phi_sim: batch voyage simulation (spec Section 5.4).

Every candidate route (a lane path plus a speed) is sailed against every
storm scenario at once with numpy broadcasting over

    candidates (P) x scenarios (S) x route segments (M) x storms (K).

Per segment the storms are evaluated where they will be when the ship gets
there (start, middle and end of the segment), giving a 0..1 hazard. The
hazard slows the ship (involuntary speed loss) and raises fuel burn (added
resistance in waves), so storms show up in fuel, ETA and ETA spread, not only
in the risk numbers. The fuel model is the propeller law, burn ~ speed^3,
scaled by the twin's learned hull/fuel factor theta_fuel.

Limitation (documented, not hidden): the arrival time used to place the
storms is the no-slowdown time; one fixed-point pass, not an RK4 integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


@dataclass
class FuelModel:
    ref_speed_kts: float = 16.0
    ref_burn_tph: float = 1.6

    def rate_tph(self, speed_kts, hazard=0.0, wx=1.0, theta=1.0):
        return (self.ref_burn_tph * (np.asarray(speed_kts) / self.ref_speed_kts) ** 3
                * (1.0 + ADDED_RESISTANCE * np.asarray(hazard)) * wx * theta)


def storm_hazard(px, py, t, ens: ScenarioSet):
    """
    Hazard of points (..., M) at times (..., M) under every scenario.
    Returns (risk, core) with shape (..., S, M); broadcasting adds the S axis.
    """

    if ens.n_storms == 0:
        shape = np.broadcast_shapes(np.shape(px), np.shape(t))
        shape = shape[:-1] + (ens.n, shape[-1])
        return np.zeros(shape), np.zeros(shape, bool)

    px = np.asarray(px)[..., None, :, None]          # (..., 1, M, 1)
    py = np.asarray(py)[..., None, :, None]
    tt = np.asarray(t)[..., None, :, None]
    lead = tt - ens.t_obs
    cx = ens.x0[:, None, :] + ens.vx[:, None, :] * lead   # (..., S, M, K)
    cy = ens.y0[:, None, :] + ens.vy[:, None, :] * lead
    R = ens.radius[:, None, :]
    sev = _sev(ens, tt)
    d = np.hypot(px - cx, py - cy)
    r_k = sev * np.clip(1.0 - d / R, 0.0, 1.0)
    core = ((d < CORE_FRACTION * R) & (sev >= BLOCK_SEVERITY)).any(-1)
    risk = 1.0 - np.prod(1.0 - r_k, axis=-1)
    return risk, core


def _sev(ens: ScenarioSet, tt):
    life = ens.lifetime[:, None, :]
    x = tt / life
    env = np.where(x < 0.25, x / 0.25, np.where(x > 0.7, (1.0 - x) / 0.3, 1.0))
    env = np.where((x < 0) | (x > 1), 0.0, env)
    env = np.where(np.isinf(life), 1.0, env)
    return np.clip(ens.peak_sev[:, None, :] * ens.sev_scale[:, None, :]
                   * np.clip(env, 0, 1), 0.0, 1.0)


class VoyageSimulator:
    """Batch evaluator: candidates x scenarios -> objective arrays."""

    def __init__(self, lattice: Lattice, fuel: FuelModel | None = None,
                 alpha: float = 0.9) -> None:
        self.lat = lattice
        self.fuel = fuel or FuelModel()
        self.alpha = alpha

    def simulate(self, col0: int, lanes: np.ndarray, speed_kts: np.ndarray,
                 pos_xy, t_now: float, ens: ScenarioSet, theta: float = 1.0,
                 wx: float = 1.0, deadline_h: float = np.inf, hold_h=0.0) -> dict:
        """hold_h: hours each candidate waits at pos_xy before sailing."""

        lanes = np.atleast_2d(lanes)
        P = lanes.shape[0]
        speed_kts = np.broadcast_to(np.asarray(speed_kts, float), (P,))
        hold = np.broadcast_to(np.asarray(hold_h, float), (P,))
        nx, ny = self.lat.path_xy(col0, lanes)
        xs = np.concatenate([np.full((P, 1), pos_xy[0]), nx], axis=1)
        ys = np.concatenate([np.full((P, 1), pos_xy[1]), ny], axis=1)
        d = np.hypot(np.diff(xs, axis=1), np.diff(ys, axis=1))         # (P, M)
        v = (speed_kts * KNOT_KMH)[:, None]

        # waiting phase: hazard at the holding position over [t, t + hold]
        th = t_now + hold[:, None] * np.linspace(0.0, 1.0, 5)[None]    # (P, 5)
        h_risk, h_core = storm_hazard(np.full(th.shape, pos_xy[0]),
                                      np.full(th.shape, pos_xy[1]), th, ens)
        h_core = h_core & (hold > 0)[:, None, None]    # sailing on is not "holding in a core"

        t_end = t_now + hold[:, None] + np.cumsum(d, axis=1) / v
        t_start = t_end - d / v
        mx, my = (xs[:, :-1] + xs[:, 1:]) / 2, (ys[:, :-1] + ys[:, 1:]) / 2

        risk = core = None
        for px, py, tt in ((xs[:, :-1], ys[:, :-1], t_start), (mx, my, (t_start + t_end) / 2),
                           (xs[:, 1:], ys[:, 1:], t_end)):
            r, c = storm_hazard(px, py, tt, ens)                       # (P, S, M)
            risk = r if risk is None else np.maximum(risk, r)
            core = c if core is None else core | c

        seg_h = d[:, None, :] / (v[:, None, :] * (1.0 - SPEED_LOSS * risk))
        eta = t_now + hold[:, None] + seg_h.sum(-1)                     # (P, S)
        rate = self.fuel.rate_tph(speed_kts[:, None, None], risk, wx, theta)
        fuel = (rate * seg_h).sum(-1)                                   # (P, S)
        fuel = fuel + self.fuel.rate_tph(HOLD_SPEED_KTS, h_risk.mean(-1), wx, theta) * hold[:, None]
        peak = np.maximum(risk.max(-1) if risk.shape[-1] else np.zeros((P, ens.n)),
                          h_risk.max(-1))
        core = np.concatenate([core, h_core], axis=-1)
        dist = d.sum(1)

        w = ens.weights
        eta_mean = eta @ w
        out = {
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
        return out

    def hold_peak(self, pos_xy, t_now: float, hours: float, ens: ScenarioSet) -> np.ndarray:
        """Per-scenario peak hazard of holding position for the next `hours`."""

        ts = t_now + np.linspace(0.0, hours, 7)
        px = np.full(ts.shape, pos_xy[0])
        py = np.full(ts.shape, pos_xy[1])
        risk, _ = storm_hazard(px, py, ts, ens)                         # (S, T)
        return risk.max(-1)
