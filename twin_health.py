"""
Digital Twin Health Monitor (Phi_health)
-----------------------------------------
Spec reference: TwinRoute-M / TADIF, Chapter 5.10 + Chapter 12.

Full spec version combines 4 sub-scores from a Kalman filter, a PINN,
a Monte Carlo scenario ensemble, and TOPSIS sensitivity analysis. This
mini-project doesn't have the first three yet, so csensor/cfcst/cmodel
are simplified proxies built from data we DO have (bounds checks and
step-to-step deltas), clearly marked below. cdec is computed properly
since we already have real TOPSIS scores from optimizer.py.

Aggregation formula, baseline weights, and escalation bands are taken
directly from the spec (Eq. 12.5, Table 12.1) so the proxies can be
swapped for real sub-score computations later without touching the
rest of the pipeline.
"""

from dataclasses import dataclass
from typing import Optional


# Spec baseline weights (Eq. 12.5) and escalation bands (Table 12.1)
BASELINE_WEIGHTS = {"sensor": 0.30, "forecast": 0.25, "model": 0.30, "decision": 0.15}
C_WARN = 0.65
C_CRIT = 0.40


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class TwinHealth:
    confidence: float
    subscores: dict
    band: str
    escalation: str


# ---------------------------------------------------------------------
# Sub-score 1: Sensor confidence (proxy for csensor, spec Eq. 12.1)
# ---------------------------------------------------------------------
# Real version: rolling UKF innovation statistics. Proxy here: plausibility
# bounds check on the current vessel state (fuel, speed, rpm). A reading
# outside sane physical bounds is treated as a "surprising innovation".

def compute_sensor_confidence(twin) -> float:
    nav = twin.vessel.navigation
    energy = twin.vessel.energy
    mission = twin.mission

    penalties = 0.0
    if nav.speed < 0 or nav.speed > mission.max_speed * 1.15:
        penalties += 0.4
    if energy.fuel_remaining < 0:
        penalties += 0.4
    if energy.fuel_remaining > mission.fuel_budget * 1.2:
        penalties += 0.2  # implausibly high vs stated budget

    return _clamp01(1.0 - penalties)


# ---------------------------------------------------------------------
# Sub-score 2: Forecast confidence (proxy for cfcst, spec Eq. 12.2)
# ---------------------------------------------------------------------
# Real version: ensemble spread from Phi_scenario, normalized by a
# climatological reference. Proxy here: step-to-step volatility of
# wind/wave/visibility, if a short history is supplied. Falls back to
# a severity-based estimate (calmer seas = more predictable) when no
# history is available yet.

def compute_forecast_confidence(twin, history: Optional[list] = None) -> float:
    weather = twin.environment.weather
    ocean = twin.environment.ocean

    if history:
        prev = history[-1]
        d_wind = abs(weather.wind_speed - prev.environment.weather.wind_speed)
        d_wave = abs(ocean.wave_height - prev.environment.ocean.wave_height)
        d_vis = abs(weather.visibility - prev.environment.weather.visibility)
        volatility = 0.5 * (d_wind / 10) + 0.35 * (d_wave / 2) + 0.15 * (d_vis / 5)
        return _clamp01(1.0 - volatility)

    # No history yet: fall back to a static severity estimate.
    severity = _clamp01(0.5 * (weather.wind_speed / 50) + 0.5 * (ocean.wave_height / 10))
    return _clamp01(1.0 - 0.6 * severity)


# ---------------------------------------------------------------------
# Sub-score 3: Model confidence (proxy for cmodel, spec Eq. 12.3)
# ---------------------------------------------------------------------
# Real version: PINN physics residual + realized-vs-predicted fuel
# error tracked by Phi_learn (neither exists yet). Proxy here: how
# often simulator.py's risk formula is saturating at its 1.0 clip
# across candidate routes -- clipping means the model can no longer
# distinguish between routes, which is itself a loss of model fidelity.

def compute_model_confidence(performance: list) -> float:
    if not performance:
        return 1.0
    clipped = sum(1 for p in performance if p["risk"] >= 1.0)
    return _clamp01(1.0 - (clipped / len(performance)))


# ---------------------------------------------------------------------
# Sub-score 4: Decision confidence (real, spec Eq. 12.4)
# ---------------------------------------------------------------------
# This one we can compute properly: it's the gap between the winning
# TOPSIS score and its runner-up among the Pareto-optimal candidates.
# A wide gap = confident pick; a narrow gap = borderline/fragile pick.

def compute_decision_confidence(report: dict) -> float:
    scores = sorted(report["topsis_scores"].values(), reverse=True)
    if len(scores) < 2:
        return 1.0  # only one Pareto-optimal candidate: nothing to be uncertain against
    best, runner_up = scores[0], scores[1]
    gap = best - runner_up
    return _clamp01(gap / 0.5)  # normalize: a 0.5 closeness gap is treated as maximally confident


# ---------------------------------------------------------------------
# Aggregation + escalation (spec Eq. 12.5, Table 12.1)
# ---------------------------------------------------------------------

def classify_band(c: float) -> str:
    if c >= C_WARN:
        return "NOMINAL"
    elif c >= C_CRIT:
        return "ADVISORY"
    else:
        return "CRITICAL"


def escalation_for_band(band: str) -> str:
    return {
        "NOMINAL": "No escalation; standard trigger-driven re-planning only.",
        "ADVISORY": "Reduced-confidence banner attached to decision explanation; advisory logged.",
        "CRITICAL": "Force full re-estimation cycle; widen risk thresholds; operator acknowledgment required.",
    }[band]


def compute_twin_health(twin, performance: list, report: dict,
                         history: Optional[list] = None,
                         weights: dict = None) -> TwinHealth:
    """
    Full Phi_health evaluation for one cycle.

    twin:        current DigitalTwin snapshot
    performance: simulate_all_routes() output
    report:      optimizer.select_best_route() report (needs 'topsis_scores')
    history:     optional list of previous DigitalTwin snapshots (most recent last)
    """
    if weights is None:
        weights = BASELINE_WEIGHTS

    c_sensor = compute_sensor_confidence(twin)
    c_forecast = compute_forecast_confidence(twin, history)
    c_model = compute_model_confidence(performance)
    c_decision = compute_decision_confidence(report)

    c = (weights["sensor"] * c_sensor
         + weights["forecast"] * c_forecast
         + weights["model"] * c_model
         + weights["decision"] * c_decision)
    c = _clamp01(c)

    band = classify_band(c)
    return TwinHealth(
        confidence=round(c, 3),
        subscores={
            "sensor": round(c_sensor, 3),
            "forecast": round(c_forecast, 3),
            "model": round(c_model, 3),
            "decision": round(c_decision, 3),
        },
        band=band,
        escalation=escalation_for_band(band),
    )
