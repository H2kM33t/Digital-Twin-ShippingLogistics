"""
Phi_learn: continual learning (spec Section 5.9, Chapter 11).

Implemented:

  * Bayesian update of the hidden fuel/hull factor theta_fuel (Section 11.5).
    The real ship burns theta_true times what the clean-hull model predicts
    (fouling); the twin starts from a prior and, after every hour, observes
    realized vs. modelled burn. Normal-normal conjugate update:

        1/s2_post = 1/s2_prior + 1/s2_obs
        mu_post   = s2_post * (mu_prior/s2_prior + y/s2_obs)

  * A rolling realized-vs-predicted fuel error, used by the model sub-score
    (Eq. 12.3) and by the fuel-prediction-error trigger (Table 7.1).

  * A knowledge base (Section 11.3) persisted as JSON: every finished voyage
    stores its outcome and the theta posterior, and the next voyage starts
    from that posterior instead of the generic prior. "Every completed voyage
    refines the twin."

Not implemented from the spec: PINN fine-tuning and preference learning.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Optional

import numpy as np

PRIOR_MU, PRIOR_SD = 1.0, 0.15
OBS_SD = 0.05            # per-hour fuel-meter noise (relative)
FUEL_ERR_REF = 0.05      # Eq. 12.3 reference error


class KnowledgeBase:

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.data = {"theta_fuel": {"mu": PRIOR_MU, "sd": PRIOR_SD}, "voyages": []}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.data.update(json.load(fh))
            except (OSError, ValueError):
                pass

    @property
    def prior(self):
        th = self.data["theta_fuel"]
        # widen a little between voyages: fouling keeps changing in port
        return th["mu"], float(np.hypot(th["sd"], 0.02))

    def record(self, voyage: dict, mu: float, sd: float) -> None:
        self.data["theta_fuel"] = {"mu": round(mu, 4), "sd": round(sd, 4)}
        self.data["voyages"] = (self.data["voyages"] + [voyage])[-50:]
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=1)

    def reset(self) -> None:
        self.data = {"theta_fuel": {"mu": PRIOR_MU, "sd": PRIOR_SD}, "voyages": []}
        if self.path and os.path.exists(self.path):
            os.remove(self.path)


class OnlineLearner:

    def __init__(self, kb: KnowledgeBase, window: int = 6) -> None:
        self.kb = kb
        self.mu, self.sd = kb.prior
        self.sd0 = self.sd
        self.errors = deque(maxlen=window)
        self.realized_t = 0.0
        self.predicted_t = 0.0
        self.updates = 0

    @property
    def theta(self) -> float:
        return self.mu

    def observe(self, realized_t: float, model_t: float) -> float:
        """One leg: realized burn vs clean-hull model burn. Returns the
        relative error of the twin's prediction (model * theta) before the
        update."""

        if model_t <= 1e-9:
            return 0.0
        predicted = model_t * self.mu
        err = (realized_t - predicted) / predicted
        self.errors.append(err)
        self.realized_t += realized_t
        self.predicted_t += predicted

        y = realized_t / model_t
        prec = 1.0 / self.sd ** 2 + 1.0 / OBS_SD ** 2
        self.mu = (self.mu / self.sd ** 2 + y / OBS_SD ** 2) / prec
        self.sd = float(np.sqrt(1.0 / prec))
        self.updates += 1
        return err

    def trend_error(self, n: int = 3) -> float:
        """Signed mean error of the last n legs (trigger input: a single
        noisy hour should not re-plan the voyage)."""
        last = list(self.errors)[-n:]
        return float(np.mean(last)) if last else 0.0

    def recent_error(self) -> float:
        return float(np.mean(np.abs(self.errors))) if self.errors else 0.0

    def confidence(self) -> float:
        """Model sub-score (Eq. 12.3, fuel term only; no PINN residual)."""

        c_err = np.exp(-self.recent_error() / FUEL_ERR_REF)
        c_unc = np.exp(-max(self.sd / PRIOR_SD - 0.2, 0.0))
        return float(0.7 * c_err + 0.3 * c_unc)

    def finish_voyage(self, summary: dict) -> None:
        self.kb.record({
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hours": summary.get("total_hours"),
            "sailed_km": summary.get("sailed_km"),
            "fuel_realized_t": round(self.realized_t, 2),
            "fuel_predicted_t": round(self.predicted_t, 2),
            "theta_fuel": round(self.mu, 4),
        }, self.mu, self.sd)
