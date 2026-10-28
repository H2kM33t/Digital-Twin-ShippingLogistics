"""
Phi_opt: NSGA-III global optimizer (spec Section 5.6, Chapter 8).

Decision variables (Section 8.1), for the rest of the voyage from the node
the ship is heading to:

    genes[0 : M]   desired lane at every remaining lattice column (continuous;
                   repaired into a feasible lane path by Lattice.repair)
    genes[M]       speed gene in [0, 1], mapped onto discrete speed levels
    genes[M + 1]   hold gene in [0, 1], mapped onto discrete hold times: wait
                   where you are for h hours, then sail (waiting is a
                   decision the optimizer weighs against detouring)

Objectives (all minimized): expected fuel, expected lateness against the
deadline, expected peak hazard (safety) and CVaR of peak hazard
(robustness). Constraint: no storm-core crossing on the nominal forecast
track (constraint violation = 1 if crossed), handled with constrained
domination (Section 8.4).

Operators are route-aware rather than generic SBX: one-point crossover keeps
the head of one route and the tail of another, and the main mutation adds a
smooth Gaussian "bump" of lanes, i.e. a detour of random size and position.

Survival is standard NSGA-III (Deb & Jain 2014): non-dominated sorting,
normalization by ideal point and hyperplane intercepts, association with
Das-Dennis reference directions, and niche-preserving selection.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Tuple

import numpy as np


# ======================================================================
# Reference directions and sorting
# ======================================================================

def das_dennis(n_obj: int, partitions: int) -> np.ndarray:
    """Uniform reference directions on the unit simplex."""

    dirs = []
    for c in combinations(range(partitions + n_obj - 1), n_obj - 1):
        prev, row = -1, []
        for b in c:
            row.append(b - prev - 1)
            prev = b
        row.append(partitions + n_obj - 2 - prev)
        dirs.append(row)
    return np.array(dirs, float) / partitions


def constrained_fronts(F: np.ndarray, CV: np.ndarray) -> list:
    """Fronts under constrained domination: feasible by Pareto rank, then
    infeasible ones in order of increasing violation."""

    feas = np.nonzero(CV <= 0)[0]
    fronts = []
    if len(feas):
        Ff = F[feas]
        le = (Ff[:, None, :] <= Ff[None, :, :]).all(-1)
        lt = (Ff[:, None, :] < Ff[None, :, :]).any(-1)
        dom = le & lt                                   # dom[i, j]: i dominates j
        count = dom.sum(0)
        current = np.nonzero(count == 0)[0]
        while len(current):
            fronts.append(feas[current])
            count = count - dom[current].sum(0)
            count[current] = -1
            current = np.nonzero(count == 0)[0]
    infeas = np.nonzero(CV > 0)[0]
    for v in np.unique(CV[infeas]):
        fronts.append(infeas[CV[infeas] == v])
    return fronts


def _normalize(F: np.ndarray) -> np.ndarray:
    ideal = F.min(0)
    Fp = F - ideal
    M = F.shape[1]
    ext = []
    for i in range(M):
        w = np.full(M, 1e-6)
        w[i] = 1.0
        ext.append(np.argmin((Fp / w).max(1)))
    E = Fp[ext]
    try:
        b = np.linalg.solve(E, np.ones(M))
        with np.errstate(divide="ignore"):
            intercepts = 1.0 / b
        if np.any(~np.isfinite(intercepts)) or np.any(intercepts <= 1e-9):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = Fp.max(0)
    intercepts = np.where(intercepts <= 1e-9, 1.0, intercepts)
    return Fp / intercepts


# ======================================================================
# NSGA-III
# ======================================================================

class NSGA3:

    def __init__(self, n_obj: int = 4, partitions: int = 5, n_gen: int = 25,
                 seed: int = 0) -> None:
        self.ref = das_dennis(n_obj, partitions)
        self.pop_size = len(self.ref) + (len(self.ref) % 2)
        self.n_gen = n_gen
        self.rng = np.random.default_rng(seed)
        self.n_evals = 0

    # ---------------- variation ----------------

    def _bump(self, G: np.ndarray, n_lanes: int, M: int) -> np.ndarray:
        P = len(G)
        if M < 2 or P == 0:
            return G
        centre = self.rng.integers(1, M, P)[:, None]
        width = self.rng.uniform(2.0, max(3.0, M / 3), P)[:, None]
        amp = self.rng.normal(0.0, n_lanes / 4, P)[:, None]
        cols = np.arange(M)[None]
        G = G.copy()
        G[:, :M] += amp * np.exp(-0.5 * ((cols - centre) / width) ** 2)
        return G

    def _offspring(self, G: np.ndarray, CV: np.ndarray, n_lanes: int, levels: tuple):
        P, D = G.shape
        M = D - len(levels)
        a, b = self.rng.integers(0, P, (2, P))
        parents = np.where((CV[a] <= CV[b])[:, None], G[a], G[b])
        c, d = self.rng.integers(0, P, (2, P))
        mates = np.where((CV[c] <= CV[d])[:, None], G[c], G[d])

        kids = parents.copy()
        if M > 2:
            cut = self.rng.integers(1, M, P)
            take = (np.arange(M)[None] >= cut[:, None]) & (self.rng.random(P) < 0.9)[:, None]
            kids[:, :M] = np.where(take, mates[:, :M], parents[:, :M])
        mut = self.rng.random(P) < 0.7
        kids[mut] = self._bump(kids[mut], n_lanes, M)

        for g, n_levels in enumerate(levels):
            col = M + g
            mix = self.rng.random(P) < 0.5
            kids[mix, col] = mates[mix, col]
            smut = self.rng.random(P) < 0.3
            step = 1.0 / max(n_levels - 1, 1)
            kids[smut, col] = np.clip(
                kids[smut, col] + self.rng.choice([-step, step], smut.sum()), 0, 1)
        return kids

    # ---------------- survival ----------------

    def _survive(self, F, CV, n):
        fronts = constrained_fronts(F, CV)
        chosen, last = [], None
        for fr in fronts:
            if len(chosen) + len(fr) <= n:
                chosen.extend(fr.tolist())
            else:
                last = fr
                break
        if last is None or len(chosen) == n:
            return np.array(chosen[:n])
        if CV[last[0]] > 0:                       # infeasible front: by violation
            return np.array(chosen + last[: n - len(chosen)].tolist())

        pool = np.array(chosen + last.tolist())
        feas_pool = pool[CV[pool] <= 0]
        Nf = _normalize(F[feas_pool])
        W = self.ref / np.linalg.norm(self.ref, axis=1, keepdims=True)
        proj = Nf @ W.T
        dist = np.sqrt(np.maximum((Nf ** 2).sum(1)[:, None] - proj ** 2, 0.0))
        assoc = dist.argmin(1)
        dmin = dist.min(1)
        pos = {int(p): k for k, p in enumerate(feas_pool)}

        niche = np.zeros(len(self.ref), int)
        for p in chosen:
            if p in pos:
                niche[assoc[pos[p]]] += 1

        remaining = list(last)
        active = np.ones(len(self.ref), bool)
        while len(chosen) < n and remaining:
            cand_niche = np.where(active, niche, np.iinfo(int).max)
            j_min = cand_niche.min()
            js = np.nonzero(cand_niche == j_min)[0]
            j = int(self.rng.choice(js))
            members = [p for p in remaining if assoc[pos[p]] == j]
            if not members:
                active[j] = False
                if not active.any():
                    chosen.extend(remaining[: n - len(chosen)])
                    break
                continue
            pick = (min(members, key=lambda p: dmin[pos[p]]) if niche[j] == 0
                    else members[int(self.rng.integers(len(members)))])
            chosen.append(pick)
            remaining.remove(pick)
            niche[j] += 1
        return np.array(chosen)

    # ---------------- main loop ----------------

    def run(self, evaluate: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
            seeds: np.ndarray, n_lanes: int, levels: tuple):
        """
        evaluate(G) -> (F, CV, key) where key is a hashable-row array used to
        drop duplicate routes. `levels` gives the number of discrete levels of
        each trailing gene (speed, hold). Returns the final population (G, F, CV).
        """

        n = self.pop_size
        M = seeds.shape[1] - len(levels)
        G = seeds.copy()
        while len(G) < n:
            extra = seeds[self.rng.integers(0, len(seeds), n - len(G))]
            G = np.vstack([G, self._bump(extra, n_lanes, M)])
        G = G[:n]
        F, CV, K = evaluate(G)
        self.n_evals += len(G)

        for _ in range(self.n_gen):
            Q = self._offspring(G, CV, n_lanes, levels)
            FQ, CVQ, KQ = evaluate(Q)
            self.n_evals += len(Q)
            G2, F2, CV2, K2 = (np.vstack([G, Q]), np.vstack([F, FQ]),
                               np.concatenate([CV, CVQ]), np.vstack([K, KQ]))
            _, uniq = np.unique(K2, axis=0, return_index=True)
            uniq = np.sort(uniq)
            G2, F2, CV2, K2 = G2[uniq], F2[uniq], CV2[uniq], K2[uniq]
            keep = self._survive(F2, CV2, min(n, len(G2)))
            G, F, CV, K = G2[keep], F2[keep], CV2[keep], K2[keep]

        return G, F, CV
