def normalize(values, value):
    """Scale a value to 0-1 range relative to a list of values."""
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def select_best_route(performance, weights=None):
    """
    Select the best route index based on a weighted score of
    fuel, time, and risk (lower score = better).

    performance: list of dicts, each with 'fuel', 'time', 'risk'
    weights: dict with keys 'fuel', 'time', 'risk' (defaults below)

    Returns: (best_index, list_of_scores)
    """
    if weights is None:
        weights = {"fuel": 0.4, "time": 0.3, "risk": 0.3}

    fuels = [p["fuel"] for p in performance]
    times = [p["time"] for p in performance]
    risks = [p["risk"] for p in performance]

    scores = []
    for p in performance:
        norm_fuel = normalize(fuels, p["fuel"])
        norm_time = normalize(times, p["time"])
        norm_risk = normalize(risks, p["risk"])

        score = (
            weights["fuel"] * norm_fuel
            + weights["time"] * norm_time
            + weights["risk"] * norm_risk
        )
        scores.append(score)

    best_index = scores.index(min(scores))
    return best_index, scores