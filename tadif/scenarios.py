"""
Phi_scenario: storm-track scenario ensemble (spec Section 5.3, Chapter 9).

The twin knows where a detected storm is now, but not where it will go. Each
scenario is one plausible future: the storm's heading, speed, radius and
severity are perturbed, so the position error grows with lead time the way a
real track forecast's cone does. Scenario 0 is always the unperturbed
forecast (the nominal track).

For the optimizer the raw ensemble is reduced to a few weighted
representative scenarios with k-means on the storms' +24 h positions
(cluster_ensemble), which keeps NSGA-III cheap; the risk assessor uses the
full ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from routing.hazard import LocalFrame, Storm

SPREAD_LEAD_H = 24.0


@dataclass
class ScenarioSet:
    """Storm parameters per scenario, shape (S, n_storms) each."""

    x0: np.ndarray          # storm centre at t_obs (km frame)
    y0: np.ndarray
    vx: np.ndarray          # km/h
    vy: np.ndarray
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
        return self.x0.shape[1]

    def subset(self, idx: np.ndarray, weights: np.ndarray) -> "ScenarioSet":
        pick = lambda a: a[idx]
        return ScenarioSet(pick(self.x0), pick(self.y0), pick(self.vx), pick(self.vy),
                           pick(self.radius), pick(self.sev_scale), pick(self.peak_sev),
                           pick(self.lifetime), self.t_obs, weights / weights.sum(),
                           self.names, self.spread_km, self.issued_h)


class ScenarioGenerator:

    def __init__(self, frame: LocalFrame, n: int = 48, seed: int = 7,
                 sigma_heading_deg: float = 14.0, sigma_speed: float = 0.30,
                 sigma_radius: float = 0.12, sigma_severity: float = 0.08,
                 sigma_pos_km: float = 15.0) -> None:
        self.frame = frame
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.sig = dict(h=sigma_heading_deg, v=sigma_speed, r=sigma_radius,
                        s=sigma_severity, p=sigma_pos_km)
        self.spread_scale = 1.0     # learned calibration (see learning.py)

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

        heading = np.array([s.heading_deg for s in known], float)
        speed = np.array([s.speed_kmh for s in known], float)
        x_start, y_start = self.frame.to_xy(np.array([s.lat for s in known], float),
                                            np.array([s.lon for s in known], float))
        x_start, y_start = np.asarray(x_start, float), np.asarray(y_start, float)

        # where the storm is now (observed): its true track up to t
        h_rad = np.radians(heading)
        x_now = x_start + speed * np.sin(h_rad) * t
        y_now = y_start + speed * np.cos(h_rad) * t

        def noise(scale):
            z = self.rng.standard_normal(shape) * scale * sc
            z[0] = 0.0                                   # scenario 0 = nominal
            return z

        hd = np.radians(heading + noise(self.sig["h"]))
        # stationary storms still drift a little in the perturbed scenarios
        sp = np.maximum(speed * np.exp(noise(self.sig["v"])) + np.abs(noise(2.0)), 0.0)
        sp[0] = speed
        x0 = x_now + noise(self.sig["p"])
        y0 = y_now + noise(self.sig["p"])

        radius = np.array([s.radius_km for s in known], float) * np.exp(noise(self.sig["r"]))
        sev_scale = np.clip(1.0 + noise(self.sig["s"]), 0.6, 1.3)
        peak = np.broadcast_to(np.array([s.severity for s in known], float), shape).copy()
        life = np.array([s.lifetime_h if s.lifetime_h else np.inf for s in known], float)
        life = np.broadcast_to(life, shape).copy()

        vx, vy = sp * np.sin(hd), sp * np.cos(hd)

        lead = SPREAD_LEAD_H
        if k:
            ex = x0 + vx * lead - (x0[0] + vx[0] * lead)
            ey = y0 + vy * lead - (y0[0] + vy[0] * lead)
            spread = float(np.mean(np.hypot(ex, ey)))
        else:
            spread = 0.0

        return ScenarioSet(x0, y0, vx, vy, radius, sev_scale, peak, life, float(t),
                           np.full(n, 1.0 / n), [s.name for s in known], spread, float(t))

    def cluster_ensemble(self, ens: ScenarioSet, k: int = 12, iters: int = 15) -> ScenarioSet:
        """k-means reduction to k weighted representative scenarios (medoids)."""

        if ens.n_storms == 0 or ens.n <= k:
            return ens

        lead = SPREAD_LEAD_H
        feats = np.concatenate([ens.x0 + ens.vx * lead, ens.y0 + ens.vy * lead,
                                ens.radius, 100.0 * ens.sev_scale], axis=1)
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
