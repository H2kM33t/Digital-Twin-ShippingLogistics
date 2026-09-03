"""
Adaptive Objective Builder (Phi_weight)
----------------------------------------
Spec reference: TwinRoute-M / TADIF, Chapter 5.5 "Operator 5 - Adaptive
Objective Builder".

Purpose (from the spec, simplified for this project's scope):
    Map the current Digital Twin state onto a normalized weight vector
    {fuel, time, risk} that the optimizer (optimizer.py -> topsis_rank)
    uses to score candidate routes -- replacing hand-fixed weights with
    weights that respond to the mission's real situation.

This is a pure function: same twin state in -> same weights out, no
side effects, no hidden state. That mirrors the spec's "Possible
Implementation" note (5.5.6), which calls for a stateless, side-effect
free implementation.

Simplifications vs. the full spec:
    - The full TADIF builds weights over 6 objectives (fuel, ETA-variance,
      emissions, structural safety, reliability, robustness) using a
      softmax over log-weights, blended with a Twin Confidence Index.
      This project only optimizes {fuel, time, risk} (see optimizer.py),
      so we adapt only those three.
    - Instead of softmax-of-log-weights, we use an additive nudge
      followed by renormalization. This is simpler to read and keeps
      the "graceful blending" property the spec cares about (no sudden
      jumps when a threshold is crossed) as long as the pressure terms
      themselves are continuous -- which they are here.
"""

from dataclasses import dataclass


# Default weights for a "standard cargo" voyage with no unusual pressure.
# Same numbers optimizer.py used to hardcode.
BASE_WEIGHTS = {"fuel": 0.4, "time": 0.3, "risk": 0.3}


@dataclass
class WeightPressures:
    """Intermediate pressure signals, exposed mainly so callers can explain
    *why* the weights shifted (see explain_weights below)."""
    fuel_pressure: float   # 0 (tank full relative to budget) -> 1 (critically low)
    weather_severity: float  # 0 (calm) -> 1 (severe wind + waves)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_pressures(twin) -> WeightPressures:
    """
    Derive continuous 0-1 pressure signals from the current twin state.

    fuel_pressure: how close the vessel is to running out of its fuel
    budget. 0 = plenty of fuel left, 1 = at or past budget.

    weather_severity: how rough current conditions are, blending wind
    speed and wave height into a single 0-1 severity score.
    """
    fuel_remaining = twin.vessel.energy.fuel_remaining
    fuel_budget = twin.mission.fuel_budget

    if fuel_budget > 0:
        fuel_pressure = _clamp01(1 - (fuel_remaining / fuel_budget))
    else:
        fuel_pressure = 0.0

    wind_speed = twin.environment.weather.wind_speed
    wave_height = twin.environment.ocean.wave_height

    # Same normalization scales simulator.py already uses for wind/wave
    # risk (wind/50, wave/10), so "severity" lines up with what simulate_route
    # actually treats as risky.
    weather_severity = _clamp01(0.5 * (wind_speed / 50) + 0.5 * (wave_height / 10))

    return WeightPressures(fuel_pressure=fuel_pressure, weather_severity=weather_severity)


def build_adaptive_weights(
    twin,
    base_weights: dict = None,
    beta_fuel: float = 0.5,
    beta_risk: float = 0.6,
) -> dict:
    """
    Compute a mission-conditional {fuel, time, risk} weight vector for
    optimizer.select_best_route / topsis_rank.

    Algorithm (simplified Algorithm 5.5):
      1. Start from base_weights (a "standard cargo" prior).
      2. Push weight onto 'fuel' as fuel_pressure rises (low fuel -> the
         twin starts prioritizing fuel economy over speed/comfort).
      3. Push weight onto 'risk' as weather_severity rises (rough seas ->
         the twin becomes more conservative about picking a risky route).
      4. Take weight for the nudges from 'time' first, so time isn't
         double-penalized on top of the two pressure terms.
      5. Renormalize so the vector still sums to 1 (required by TOPSIS).

    beta_fuel / beta_risk control how aggressively each pressure signal
    reshapes the weights -- these correspond to the spec's beta_dl /
    beta_wx modulation gains (5.5.2), just linear here instead of
    logistic/log-space.
    """
    if base_weights is None:
        base_weights = BASE_WEIGHTS

    pressures = compute_pressures(twin)

    fuel_w = base_weights["fuel"] + beta_fuel * pressures.fuel_pressure
    risk_w = base_weights["risk"] + beta_risk * pressures.weather_severity
    # Time absorbs the shift so low-fuel / stormy situations don't just
    # inflate everything -- it's the objective de-prioritized when the
    # twin gets nervous about fuel or safety.
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
    """Human-readable one-liner describing why the weights shifted, for
    printing alongside the route recommendation."""
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
