"""
Phi_dec: decision intelligence (spec Section 5.8, Chapter 10).

Ranks the risk-filtered Pareto set on the six criteria with Adaptive TOPSIS
(primary) and PROMETHEE II (cross-check), runs a weight-sensitivity sweep,
and builds the explanation object of the decision-output schema (Section
15.4): score, runner-up gap, dominant criteria, risk rejections, weight
sensitivity and twin confidence at decision time.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .objectives import CRITERIA

LABELS = {"fuel": "fuel", "eta_var": "ETA deviation", "emissions": "CO2",
          "safety": "storm exposure", "reliability": "schedule (hours late)",
          "robustness": "worst-case (CVaR) risk"}


def criteria_matrix(sim: dict, risk: dict, idx) -> np.ndarray:
    """
    Rows = candidates, columns = CRITERIA, all lower-is-better.

      eta_var      RMS hours past the deadline over the scenarios (ETA
                   variance measured about the schedule, not the mean)
      reliability  expected hours late (not just P(late): once a deadline
                   is lost, P(late) = 1 for everything and waiting looks free)
    """

    idx = np.asarray(idx)
    X = np.column_stack([
        sim["fuel_t"][idx], sim["eta_dev_h"][idx], sim["co2_t"][idx],
        sim["safety"][idx], sim["lateness_h"][idx], risk["cvar"][idx],
    ])
    # Round away float noise: a 1e-9 h "ETA spread" must not make a longer
    # route look non-dominated or swing TOPSIS's vector normalization.
    return np.round(X, 3)


def topsis(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Closeness to the ideal (same steps as optimizer.topsis_rank, which is
    fixed to three criteria)."""

    norms = np.sqrt((X ** 2).sum(0))
    norms[norms == 0] = 1.0
    V = X / norms * w
    best, worst = V.min(0), V.max(0)
    d_best = np.sqrt(((V - best) ** 2).sum(1))
    d_worst = np.sqrt(((V - worst) ** 2).sum(1))
    return d_worst / (d_best + d_worst + 1e-12)


def promethee_ii(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Net outranking flow with a linear (V-shape) preference function."""

    n = len(X)
    if n < 2:
        return np.zeros(n)
    rng = X.max(0) - X.min(0)
    rng[rng <= 1e-12] = 1.0
    d = (X[None, :, :] - X[:, None, :]) / rng          # d[i, j] = advantage of i over j
    pi = (np.clip(d, 0.0, 1.0) * w).sum(-1)
    return (pi.sum(1) - pi.sum(0)) / (n - 1)


def sensitivity_sweep(X: np.ndarray, w: np.ndarray, method=topsis) -> Dict[str, float]:
    """
    For every criterion: the smallest relative change of its weight (up or
    down, other weights renormalized) that changes the winner, capped at 1.0
    (= stable even if that weight is halved or doubled).
    """

    base = int(np.argmax(method(X, w)))
    out = {}
    factors = [1.1, 1.25, 1.5, 1.75, 2.0]
    for c, name in enumerate(CRITERIA):
        margin = 1.0
        for f in factors:
            flipped = False
            for g in (f, 1.0 / f):
                w2 = w.copy()
                w2[c] *= g
                w2 /= w2.sum()
                if int(np.argmax(method(X, w2))) != base:
                    flipped = True
                    break
            if flipped:
                margin = f - 1.0
                break
        out[name] = round(margin, 3)
    return out


def dominant_criteria(X: np.ndarray, w: np.ndarray, winner: int, k: int = 2) -> List[str]:
    if len(X) < 2:
        return [CRITERIA[int(i)] for i in np.argsort(-w)[:k]]
    rng = X.max(0) - X.min(0)
    varies = rng > 1e-12
    rng[~varies] = 1.0
    others = np.delete(X, winner, axis=0).mean(0)
    adv = np.where(varies, w * (others - X[winner]) / rng, -np.inf)
    order = [int(i) for i in np.argsort(-adv) if np.isfinite(adv[i])]
    return [CRITERIA[i] for i in order[:k]] or [CRITERIA[int(np.argmax(w))]]


class DecisionRanker:

    def __init__(self, method: str = "topsis") -> None:
        self.method = method

    def rank(self, X: np.ndarray, weights: dict) -> dict:
        w = np.array([weights[c] for c in CRITERIA])
        t_scores = topsis(X, w)
        p_flows = promethee_ii(X, w)
        primary = t_scores if self.method == "topsis" else p_flows
        order = np.argsort(-primary)
        winner = int(order[0])
        gap = float(primary[order[0]] - primary[order[1]]) if len(X) > 1 else 1.0
        sens = sensitivity_sweep(X, w, topsis if self.method == "topsis" else promethee_ii)
        return {
            "scores": primary, "topsis": t_scores, "promethee": p_flows,
            "order": order, "winner": winner, "runner_up_gap": gap,
            "methods_agree": int(np.argmax(t_scores)) == int(np.argmax(p_flows)),
            "weight_sensitivity": sens,
            "dominant": dominant_criteria(X, w, winner),
        }


def explanation_text(choice: str, dominant: List[str], gap: float,
                     n_rejected: int, n_ranked: int) -> str:
    why = " and ".join(LABELS[d] for d in dominant)
    txt = f"{choice}: best balance among {n_ranked} risk-accepted routes, mainly on {why}"
    if n_rejected:
        txt += f"; {n_rejected} route(s) rejected by the risk filter"
    if gap < 0.03:
        txt += " (close call: runner-up within 0.03)"
    return txt + "."
