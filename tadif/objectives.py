"""
Phi_weight: adaptive objective builder (spec Section 5.5, Eq. 5.21).

    lambda = softmax( log lambda0(type)
                      + k_dl  * sigmoid(deadline pressure) * a_dl
                      + k_wx  * weather severity          * a_safety
                      + k_rob * (1 - twin confidence)     * a_robust )

over the six criteria {fuel, ETA-variance, emissions, safety, reliability,
robustness}. Working in log space keeps every weight positive and lets
several pressures blend smoothly instead of switching profiles.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

CRITERIA = ["fuel", "eta_var", "emissions", "safety", "reliability", "robustness"]

# Table 5.1: base weight vectors per mission archetype
PROFILES: Dict[str, list] = {
    "cargo_standard":    [0.30, 0.15, 0.15, 0.20, 0.15, 0.05],
    "fuel_saving":       [0.50, 0.05, 0.20, 0.15, 0.05, 0.05],
    "deadline_critical": [0.10, 0.45, 0.05, 0.20, 0.15, 0.05],
    "emergency":         [0.05, 0.35, 0.05, 0.35, 0.15, 0.05],
    "heavy_weather":     [0.10, 0.10, 0.05, 0.45, 0.15, 0.15],
}

# direction vectors of Eq. 5.21
A_DEADLINE = np.array([0, 1.0, 0, 0, 1.0, 0])
A_SAFETY = np.array([0, 0, 0, 1.0, 0, 0.5])
A_ROBUST = np.array([0, 0, 0, 0, 0, 1.0])
K_DL, K_WX, K_ROB = 1.5, 1.2, 2.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def deadline_pressure(t_now: float, deadline_h: float, nominal_remaining_h: float) -> float:
    """sigma(1 - t_remain / t_nominal), centred so pressure ramps up as the
    slack falls below ~15 %."""

    if not math.isfinite(deadline_h) or nominal_remaining_h <= 0:
        return 0.0
    dl = 1.0 - (deadline_h - t_now) / nominal_remaining_h
    return _sigmoid(8.0 * (dl + 0.15))


def weather_severity(wind_kt: float, wave_m: float, visibility_km: float) -> float:
    vis = min(max((10.0 - visibility_km) / 10.0, 0.0), 1.0)
    return min(max(0.35 * wind_kt / 50 + 0.35 * wave_m / 10 + 0.30 * vis, 0.0), 1.0)


class AdaptiveWeightBuilder:

    def __init__(self, profile: str = "cargo_standard") -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown mission profile {profile!r}")
        self.profile = profile

    def build_weights(self, dl_pressure: float, wx_severity: float,
                      confidence: float) -> dict:
        base = np.array(PROFILES[self.profile])
        logits = (np.log(base) + K_DL * dl_pressure * A_DEADLINE
                  + K_WX * wx_severity * A_SAFETY
                  + K_ROB * (1.0 - confidence) * A_ROBUST)
        w = np.exp(logits - logits.max())
        w /= w.sum()
        return {
            "profile": self.profile,
            "weights": {c: round(float(v), 4) for c, v in zip(CRITERIA, w)},
            "base": dict(zip(CRITERIA, PROFILES[self.profile])),
            "pressures": {"deadline": round(dl_pressure, 3),
                          "weather": round(wx_severity, 3),
                          "low_confidence": round(1.0 - confidence, 3)},
        }
