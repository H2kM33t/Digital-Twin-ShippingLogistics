"""
Phi_scenario: storm-track scenario ensemble (spec Section 5.3, Chapter 9).

The twin knows where a detected storm is now, but not where it will go. Each
scenario is one plausible future: the storm's heading, speed, radius and
severity are perturbed, so the position error grows with lead time the way a
real track forecast's cone does. Scenario 0 is always the unperturbed
forecast (the nominal track).

Tracks live on the sphere (ECEF km): a scenario storm is at
R * (c_ref + v (t - t_ref)) / |...|, the same great-circle motion model as
routing/hazard.py, so the nominal scenario is exactly the true storm.

For the optimizer the raw ensemble is reduced to a few weighted
representative scenarios with k-means on the storms' +24 h positions
(cluster_ensemble), which keeps NSGA-III cheap; the risk assessor uses the
full ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from routing.hazard import Storm, move_on_sphere
from routing.ocean import EARTH_RADIUS_KM

SPREAD_LEAD_H = 24.0


@dataclass
class ScenarioSet:
    """Storm parameters per scenario: (S, n_storms) arrays, (S, n_storms, 3) vectors."""

    c_ref: np.ndarray       # storm centre at t_ref (ECEF km)
    v: np.ndarray           # velocity, tangent at c_ref (km/h)
    t_ref: np.ndarray
    radius: np.ndarray      # km
    sev_scale: np.ndarray   # multiplies the storm's own severity envelope
    peak_sev: np.ndarray    # storm's nominal peak severity (same in every row)
    lifetime: np.ndarray    # h, inf = forever
    t_obs: float
    weights: np.ndarray     # (S,), sums to 1
    names: List[str]
    spread_km: float        # mean +24 h position spread (forecast confidence input)
    issued_h: float

    @property
    def n(self) -> int:
        return len(self.weights)

    @property
    def n_storms(self) -> int:
        return self.radius.shape[1]

    def centres(self, t: float) -> np.ndarray:
        return move_on_sphere(self.c_ref, self.v, t - self.t_ref)

    def subset(self, idx: np.ndarray, weights: np.ndarray) -> "ScenarioSet":
        pick = lambda a: a[idx]
        return ScenarioSet(pick(self.c_ref), pick(self.v), pick(self.t_ref),
                           pick(self.radius), pick(self.sev_scale), pick(self.peak_sev),
                           pick(self.lifetime), self.t_obs, weights / weights.sum(),
                           self.names, self.spread_km, self.issued_h)


def _tangent_basis(c: np.ndarray):
    """Unit normal, east and north vectors at points c (..., 3)."""

    n = c / np.linalg.norm(c, axis=-1, keepdims=True)
    east = np.cross(np.array([0.0, 0.0, 1.0]), n)
    east /= np.maximum(np.linalg.norm(east, axis=-1, keepdims=True), 1e-12)
    north = np.cross(n, east)
    return n, east, north


class ScenarioGenerator:

    def __init__(self, frame=None, n: int = 48, seed: int = 7,
                 sigma_heading_deg: float = 14.0, sigma_speed: float = 0.30,
                 sigma_radius: float = 0.12, sigma_severity: float = 0.08,
                 sigma_pos_km: float = 15.0) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.sig = dict(h=sigma_heading_deg, v=sigma_speed, r=sigma_radius,
                        s=sigma_severity, p=sigma_pos_km)
        self.spread_scale = 1.0     # calibration hook (see learning.py)

    def known_storms(self, storms: List[Storm], t: float) -> List[Storm]:
        return [s for s in storms if t >= s.detect_h]

    def sample_forecast(self, storms: List[Storm], t: float,
                        n: Optional[int] = None) -> ScenarioSet:
        """Ensemble of n scenarios for the storms known at time t."""

        n = n or self.n
        known = self.known_storms(storms, t)
        k = len(known)
        sc = self.spread_scale
        shape = (n, k)

        def noise(scale):
            z = self.rng.standard_normal(shape) * scale * sc
            z[0] = 0.0                                   # scenario 0 = nominal
            return z

        c_now = np.array([s.centre(t) for s in known]).reshape(k, 3)
        # true velocity now (tangent at c_now) from the track itself
        v_now = np.array([(s.centre(t + 0.5) - s.centre(t - 0.5)) for s in known]).reshape(k, 3)
        nrm, east, north = _tangent_basis(c_now)
        v_now -= (v_now * nrm).sum(-1, keepdims=True) * nrm
        speed = np.linalg.norm(v_now, axis=-1)
        u = np.where(speed[:, None] > 1e-9, v_now / np.maximum(speed, 1e-9)[:, None], north)
        w = np.cross(nrm, u)                             # 90 deg left of travel

        dh = np.radians(noise(self.sig["h"]))           # heading error
        sp = speed[None] * np.exp(noise(self.sig["v"])) + np.abs(noise(2.0))   # stationary storms drift
        c_ref = (c_now[None]
                 + noise(self.sig["p"])[..., None] * east[None]
                 + noise(self.sig["p"])[..., None] * north[None])
        c_ref = EARTH_RADIUS_KM * c_ref / np.linalg.norm(c_ref, axis=-1, keepdims=True)
        v = sp[..., None] * (np.cos(dh)[..., None] * u[None] + np.sin(dh)[..., None] * w[None])
        nr = c_ref / EARTH_RADIUS_KM
        v -= (v * nr).sum(-1, keepdims=True) * nr        # keep it tangent
        t_ref = np.full(shape, float(t))

        # scenario 0 is the true track, exactly
        if k:
            c_ref[0] = np.array([s.c0 for s in known])
            v[0] = np.array([s.velocity for s in known])
            t_ref[0] = 0.0

        radius = np.array([s.radius_km for s in known], float) * np.exp(noise(self.sig["r"]))
        sev_scale = np.clip(1.0 + noise(self.sig["s"]), 0.6, 1.3)
        peak = np.broadcast_to(np.array([s.severity for s in known], float), shape).copy()
        life = np.array([s.lifetime_h if s.lifetime_h else np.inf for s in known], float)
        life = np.broadcast_to(life, shape).copy()

        ens = ScenarioSet(c_ref, v, t_ref, radius, sev_scale, peak, life, float(t),
                          np.full(n, 1.0 / n), [s.name for s in known], 0.0, float(t))
        if k:
            at = ens.centres(t + SPREAD_LEAD_H)
            ens.spread_km = float(np.mean(np.linalg.norm(at - at[0], axis=-1)))
        return ens

    def cluster_ensemble(self, ens: ScenarioSet, k: int = 12, iters: int = 15) -> ScenarioSet:
        """k-means reduction to k weighted representative scenarios (medoids)."""

        if ens.n_storms == 0 or ens.n <= k:
            return ens

        at = ens.centres(ens.t_obs + SPREAD_LEAD_H).reshape(ens.n, -1)
        feats = np.concatenate([at, ens.radius, 100.0 * ens.sev_scale], axis=1)
        rng = np.random.default_rng(0)
        centers = feats[np.r_[0, rng.choice(np.arange(1, ens.n), k - 1, replace=False)]]
        for _ in range(iters):
            d = ((feats[:, None, :] - centers[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            for c in range(k):
                if np.any(lab == c):
                    centers[c] = feats[lab == c].mean(0)

        d = ((feats[:, None, :] - centers[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        idx, w = [0], [0.0]                     # keep the nominal scenario first
        for c in range(k):
            members = np.nonzero(lab == c)[0]
            if not len(members):
                continue
            mass = ens.weights[members].sum()
            if 0 in members:
                w[0] += mass
                continue
            medoid = members[d[members, c].argmin()]
            idx.append(int(medoid))
            w.append(mass)
        return ens.subset(np.array(idx), np.array(w))
