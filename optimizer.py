import numpy as np


def normalize(values, value):
    """Scale a value to 0-1 range relative to a list of values."""
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return 0.0
    return (value - min_v) / (max_v - min_v)


# ---------------------------------------------------------------------
# Step 1: Pareto Dominance Filtering (spec Ch 8.5)
# ---------------------------------------------------------------------

def dominates(perf_a, perf_b):
    """
    Route A dominates Route B if A is at least as good as B on every
    metric (fuel, time, risk) and strictly better on at least one.
    Lower is better for all three metrics here.
    """
    keys = ["fuel", "time", "risk"]
    at_least_as_good = all(perf_a[k] <= perf_b[k] for k in keys)
    strictly_better = any(perf_a[k] < perf_b[k] for k in keys)
    return at_least_as_good and strictly_better


def pareto_front(performance):
    """
    Return the indices of routes that are NOT dominated by any other
    route — i.e. the Pareto-optimal set (spec Ch 8.5).
    """
    non_dominated = []
    for i, p in enumerate(performance):
        dominated = False
        for j, q in enumerate(performance):
            if i != j and dominates(q, p):
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)
    return non_dominated


# ---------------------------------------------------------------------
# Step 2: TOPSIS Ranking (spec Ch 10.1)
# ---------------------------------------------------------------------

def topsis_rank(performance, indices, weights=None):
    """
    Rank a subset of routes (given by 'indices') using TOPSIS:
    Technique for Order of Preference by Similarity to Ideal Solution.

    Steps:
    1. Build a decision matrix (routes x criteria)
    2. Normalize it
    3. Apply weights
    4. Find the ideal best and ideal worst solution
    5. Compute each route's distance to both
    6. Score = closeness to ideal best (0 to 1, higher is better)
    """
    if weights is None:
        weights = {"fuel": 0.4, "time": 0.3, "risk": 0.3}

    criteria = ["fuel", "time", "risk"]  # all "lower is better"
    w = np.array([weights[c] for c in criteria])

    # Build raw decision matrix for just the Pareto-optimal routes
    matrix = np.array([[performance[i][c] for c in criteria] for i in indices], dtype=float)

    # Vector normalization (standard TOPSIS step)
    norms = np.sqrt((matrix ** 2).sum(axis=0))
    norms[norms == 0] = 1  # avoid divide-by-zero
    norm_matrix = matrix / norms

    # Apply weights
    weighted_matrix = norm_matrix * w

    # Ideal best (lowest, since all criteria are "lower is better")
    # and ideal worst (highest)
    ideal_best = weighted_matrix.min(axis=0)
    ideal_worst = weighted_matrix.max(axis=0)

    # Distance from each route to ideal best / worst
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    # Closeness score: higher = better (closer to ideal, farther from worst)
    closeness = dist_worst / (dist_best + dist_worst + 1e-9)

    return closeness


# ---------------------------------------------------------------------
# Step 3: Combined Decision Support Operator (spec Ch 5.8 / Ch 10)
# ---------------------------------------------------------------------

def select_best_route(performance, weights=None):
    """
    Full decision pipeline:
    1. Filter to the Pareto-optimal set (non-dominated routes)
    2. Rank that set using TOPSIS
    3. Return the overall best route index + a full report

    Returns: (best_index, report)
    report is a dict with 'pareto_indices', 'topsis_scores', 'all_scores'
    """
    pareto_indices = pareto_front(performance)
    topsis_scores = topsis_rank(performance, pareto_indices, weights)

    # Map TOPSIS scores back to full route index list
    best_local_idx = int(np.argmax(topsis_scores))  # higher closeness = better
    best_index = pareto_indices[best_local_idx]

    # Build a full score list (routes not in Pareto front get score 0)
    all_scores = [0.0] * len(performance)
    for local_idx, route_idx in enumerate(pareto_indices):
        all_scores[route_idx] = float(topsis_scores[local_idx])

    report = {
        "pareto_indices": pareto_indices,
        "topsis_scores": {pareto_indices[i]: float(topsis_scores[i]) for i in range(len(pareto_indices))},
        "all_scores": all_scores,
    }

    return best_index, report