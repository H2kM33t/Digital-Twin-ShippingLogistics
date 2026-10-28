"""
Phi_health: Digital Twin Health Monitor (spec Section 5.10, Chapter 12).

All four sub-scores now come from real signals:

    sensor    UKF innovation statistics            (Eq. 12.1, estimation.py)
    forecast  storm-ensemble spread at +24 h       (Eq. 12.2, scenarios.py)
    model     realized-vs-predicted fuel error     (Eq. 12.3, learning.py)
    decision  sensitivity margin + runner-up gap   (Eq. 12.4, decision.py)

Aggregation weights and bands are those of Eq. 12.5 / Table 12.1 (reused
from twin_health.py). Band changes follow Algorithm 14.12 and come back as
triggers; the escalation actions (scenario count, risk conservatism) are
read by the orchestrator.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from twin_health import BASELINE_WEIGHTS, C_CRIT, C_WARN, classify_band, escalation_for_band

SPREAD_REF_KM = 200.0     # "climatological" track-error reference at +24 h
GAP_REF = 0.15


def forecast_confidence(spread_km: float, age_h: float) -> float:
    c = math.exp(-spread_km / SPREAD_REF_KM)
    return c * math.exp(-max(age_h - 6.0, 0.0) / 24.0)     # stale forecast decays


def decision_confidence(sensitivity: dict, gap: float) -> float:
    if not sensitivity:
        return 1.0
    margin = float(np.mean(list(sensitivity.values())))
    return 0.5 * (margin + min(1.0, gap / GAP_REF))


class TwinHealthMonitor:

    def __init__(self) -> None:
        self.band = "NOMINAL"
        self.series = []

    def tick(self, t: float, sensor: float, forecast: float, model: float,
             decision: float) -> dict:
        w = BASELINE_WEIGHTS
        c = (w["sensor"] * sensor + w["forecast"] * forecast
             + w["model"] * model + w["decision"] * decision)
        c = min(max(c, 0.0), 1.0)
        new_band = classify_band(c)

        trigger: Optional[dict] = None
        if new_band == "CRITICAL" and self.band != "CRITICAL":
            trigger = {"severity": "critical", "reentry": "state",
                       "detail": f"Twin confidence {c:.2f} < {C_CRIT}"}
        elif new_band == "ADVISORY" and self.band == "NOMINAL":
            trigger = {"severity": "high", "reentry": "scenario",
                       "detail": f"Twin confidence {c:.2f} < {C_WARN}"}
        self.band = new_band

        rec = {"confidence": round(c, 3), "band": new_band,
               "subscores": {"sensor": round(sensor, 3), "forecast": round(forecast, 3),
                             "model": round(model, 3), "decision": round(decision, 3)},
               "escalation": escalation_for_band(new_band), "trigger": trigger}
        self.series.append((t, rec["confidence"]))
        return rec

    def scenario_count(self, base: int, n_max: int, c: float) -> int:
        """Eq. 9.1: N_s = N_base + (1 - c) * N_max, capped."""
        if self.band == "CRITICAL":
            return n_max
        return int(min(n_max, base + (1.0 - c) * n_max))

    def conservatism(self) -> float:
        return 0.85 if self.band == "CRITICAL" else 1.0
