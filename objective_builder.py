"""
Adaptive Objective Builder (Phi_weight)
----------------------------------------
Spec reference: TwinRoute-M / TADIF, Chapter 5.5 "Operator 5 - Adaptive
Objective Builder".

Maps the current Digital Twin state onto a normalized weight vector
{fuel, time, risk} used by the optimizer to score candidate routes,
replacing hand-fixed weights with weights that respond to the real
mission situation (low fuel, rough weather, poor visibility).
"""

from dataclasses import dataclass


BASE_WEIGHTS = {"fuel": 0.4, "time": 0.3, "risk": 0.3}


@dataclass
class WeightPressures:
    fuel_pressure: float      # 0 (full tank) -> 1 (critically low)
    weather_severity: float   # 0 (calm) -> 1 (severe wind/waves/low visibility)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_pressures(twin) -> WeightPressures:
    """Derive continuous 0-1 pressure signals from the current twin state."""
    fuel_remaining = twin.vessel.energy.fuel_remaining
    fuel_budget = twin.mission.fuel_budget

    if fuel_budget > 0:
        fuel_pressure = _clamp01(1 - (fuel_remaining / fuel_budget))
    else:
        fuel_pressure = 0.0

    wind_speed = twin.environment.weather.wind_speed
    wave_height = twin.environment.ocean.wave_height
    visibility = twin.environment.weather.visibility
    visibility_severity = _clamp01((10 - visibility) / 10)

    weather_severity = _clamp01(
        0.35 * (wind_speed / 50) +
        0.35 * (wave_height / 10) +
        0.30 * visibility_severity
    )

    return WeightPressures(fuel_pressure=fuel_pressure, weather_severity=weather_severity)


def build_adaptive_weights(twin, base_weights: dict = None, beta_fuel: float = 0.5, beta_risk: float = 0.6) -> dict:
    """
    Compute a mission-conditional {fuel, time, risk} weight vector.

    - Fuel weight rises as fuel_pressure rises (low fuel -> prioritize economy)
    - Risk weight rises as weather_severity rises (rough seas/poor visibility -> prioritize safety)
    - Time absorbs the shift so it's the objective de-prioritized under pressure
    - Result is renormalized to sum to 1
    """
    if base_weights is None:
        base_weights = BASE_WEIGHTS

    pressures = compute_pressures(twin)

    fuel_w = base_weights["fuel"] + beta_fuel * pressures.fuel_pressure
    risk_w = base_weights["risk"] + beta_risk * pressures.weather_severity
    time_w = max(0.05, base_weights["time"] - beta_fuel * pressures.fuel_pressure * 0.5
                 - beta_risk * pressures.weather_severity * 0.5)

    total = fuel_w + time_w + risk_w
    weights = {
        "fuel": fuel_w / total,
        "time": time_w / total,
        "risk": risk_w / total,
    }
    return weights


def explain_weights(base_weights: dict, adaptive_weights: dict, pressures: WeightPressures) -> str:
    """Human-readable explanation of why weights shifted."""
    parts = []
    if pressures.fuel_pressure > 0.15:
        parts.append(f"fuel pressure {pressures.fuel_pressure:.2f} (low fuel relative to budget)")
    if pressures.weather_severity > 0.15:
        parts.append(f"weather severity {pressures.weather_severity:.2f} (rough conditions)")

    if not parts:
        return "Weights unchanged from baseline (no significant fuel or weather pressure)."

    delta_str = ", ".join(
        f"{k}: {base_weights[k]:.2f} -> {adaptive_weights[k]:.2f}"
        for k in ("fuel", "time", "risk")
    )
    return f"Weights adapted due to {' and '.join(parts)}. {delta_str}"