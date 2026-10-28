"""
Phi_risk: Monte Carlo risk assessment (spec Section 5.7, Chapter 9).

The loss of a candidate in one scenario is the peak storm hazard it meets.
Over the (weighted) scenario ensemble we compute

    CVaR_alpha   mean of the worst (1 - alpha) tail, Rockafellar-Uryasev form
                 evaluated by sorting (Eq. 9.4)
    p_fail       probability of entering a storm core (Eq. 9.2) + its std error
    bootstrap    percentile CI of CVaR (Section 9.1.5)

and reject candidates that break the acceptance criterion (Eq. 9.5):
CVaR <= cvar_max, p_fail <= p_fail_max, and no core crossing on the nominal
forecast track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def weighted_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    """CVaR_alpha of each row of `values` (P, S) under scenario weights (S,)."""

    values = np.atleast_2d(values)
    if values.shape[1] == 0:
        return np.zeros(values.shape[0])
    tail = max(1.0 - alpha, 1e-9)
    order = np.argsort(-values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    w = weights[order]
    before = np.cumsum(w, axis=1) - w
    take = np.clip(tail - before, 0.0, w)
    return (take * v).sum(1) / tail


def bootstrap_ci(values: np.ndarray, weights: np.ndarray, alpha: float,
                 B: int = 300, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.choice(n, size=(B, n), p=weights)
    boot = weighted_cvar(values[idx], np.full(n, 1.0 / n), alpha)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


@dataclass
class RiskThresholds:
    alpha: float = 0.9
    cvar_max: float = 0.55
    p_fail_max: float = 0.05

    def tightened(self, factor: float) -> "RiskThresholds":
        """Conservatism factor applied in the CRITICAL health band."""
        return RiskThresholds(self.alpha, self.cvar_max * factor, self.p_fail_max * factor)


class RiskAssessor:

    def __init__(self, thresholds: RiskThresholds) -> None:
        self.base = thresholds
        self.thr = thresholds

    def set_conservatism(self, factor: float) -> None:
        self.thr = self.base.tightened(factor) if factor < 1.0 else self.base

    def assess(self, sim: dict, weights: np.ndarray) -> dict:
        """Full-ensemble metrics + acceptance mask for every candidate."""

        thr = self.thr
        peak = sim["peak_by_scenario"]
        cvar = weighted_cvar(peak, weights, thr.alpha)
        p_fail = np.clip(sim["core_by_scenario"].astype(float) @ weights, 0.0, 1.0)
        n_eff = 1.0 / np.sum(weights ** 2)
        se = np.sqrt(p_fail * (1 - p_fail) / n_eff)

        ok_cvar = cvar <= thr.cvar_max
        ok_fail = p_fail <= thr.p_fail_max
        ok_core = ~sim["nominal_core"]
        accepted = ok_cvar & ok_fail & ok_core

        rejections: List[dict] = []
        for i in np.nonzero(~accepted)[0]:
            if not ok_core[i]:
                crit, margin = "nominal_core_crossing", 1.0
            elif not ok_cvar[i]:
                crit, margin = "cvar", float(cvar[i] - thr.cvar_max)
            else:
                crit, margin = "p_fail", float(p_fail[i] - thr.p_fail_max)
            rejections.append({"index": int(i), "violated_criterion": crit,
                               "margin": round(margin, 3)})

        return {"cvar": cvar, "p_fail": p_fail, "p_fail_se": se,
                "accepted": accepted, "rejections": rejections,
                "thresholds": {"alpha": thr.alpha, "cvar_max": round(thr.cvar_max, 3),
                               "p_fail_max": round(thr.p_fail_max, 3)}}
