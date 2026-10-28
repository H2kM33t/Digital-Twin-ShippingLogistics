"""
Adaptive trigger engine (spec Chapter 7).

The decision cycle no longer runs every hour. Detectors look at the twin
state each step and raise triggers; only when one fires does the
orchestrator re-enter the cognitive cycle, at the earliest stage any of the
fired triggers needs (Section 7.3: severity dominates, re-entry points are
unioned, nothing is discarded: every trigger is logged).

Catalogue implemented (Table 7.1):

    departure                 critical  state     first plan
    storm_detected            high      scenario  a storm became known
    weather_update            medium    scenario  new forecast cycle (every 6 h)
    storm_field_delta         high      sim       plan CVaR moved by > delta
    plan_core_crossing        critical  sim       plan now crosses a storm core
    sensor_drift              medium    state     GPS innovations too large
    fuel_prediction_error     medium    sim       |realized - predicted| fuel > eps
    low_twin_confidence       high/crit scenario/state  health band dropped
    hold_review               low       sim       while halted, re-check hourly

Not simulated: mission update, port congestion, communication loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

STAGES = ["state", "graph", "scenario", "sim", "weight", "opt", "risk", "dec"]
SEVERITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class Trigger:
    name: str
    severity: str
    reentry: str
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "severity": self.severity,
                "reentry": self.reentry, "detail": self.detail}


@dataclass
class TriggerConfig:
    forecast_cycle_h: float = 6.0
    cvar_delta: float = 0.08
    fuel_error: float = 0.08
    sensor_conf: float = 0.6
    sensor_cycles: int = 3


@dataclass
class TriggerEngine:
    cfg: TriggerConfig = field(default_factory=TriggerConfig)
    log: List[dict] = field(default_factory=list)
    _known: set = field(default_factory=set)
    _last_forecast: float = 0.0
    _low_sensor: int = 0

    def detect(self, t: float, *, first: bool, storms_known: List[str],
               plan_cvar: float, decided_cvar: float, plan_core: bool,
               sensor_conf: float, fuel_err: float, halted: bool,
               health_trigger: dict | None) -> List[Trigger]:
        c = self.cfg
        out: List[Trigger] = []

        if first:
            out.append(Trigger("departure", "critical", "state", "Initial voyage plan."))
            self._last_forecast = t

        new = [s for s in storms_known if s not in self._known]
        if new:
            out.append(Trigger("storm_detected", "high", "scenario",
                               f"New storm(s) in forecast: {', '.join(new)}."))
            self._last_forecast = t
        self._known.update(storms_known)

        if not first and t - self._last_forecast >= c.forecast_cycle_h - 1e-9:
            out.append(Trigger("weather_update", "medium", "scenario",
                               f"Forecast cycle ({c.forecast_cycle_h:.0f} h) received."))
            self._last_forecast = t

        if not first and plan_core:
            out.append(Trigger("plan_core_crossing", "critical", "sim",
                               "Current plan crosses a storm core on the forecast track."))
        elif not first and abs(plan_cvar - decided_cvar) > c.cvar_delta:
            out.append(Trigger("storm_field_delta", "high", "sim",
                               f"Plan CVaR {decided_cvar:.2f} -> {plan_cvar:.2f}."))

        self._low_sensor = self._low_sensor + 1 if sensor_conf < c.sensor_conf else 0
        if self._low_sensor >= c.sensor_cycles:
            out.append(Trigger("sensor_drift", "medium", "state",
                               f"GPS confidence {sensor_conf:.2f} for {self._low_sensor} h."))
            self._low_sensor = 0

        if abs(fuel_err) > c.fuel_error:
            out.append(Trigger("fuel_prediction_error", "medium", "sim",
                               f"Fuel burn {fuel_err:+.0%} vs prediction."))

        if health_trigger:
            out.append(Trigger("low_twin_confidence", health_trigger["severity"],
                               health_trigger["reentry"], health_trigger["detail"]))

        if halted and not out:
            out.append(Trigger("hold_review", "low", "sim", "Re-checking routes while holding."))

        for trig in out:
            self.log.append({"t": round(t, 2), **trig.as_dict()})
        return out

    @staticmethod
    def arbitrate(triggers: List[Trigger]) -> dict | None:
        """Section 7.3: highest severity leads, earliest re-entry wins."""

        if not triggers:
            return None
        lead = max(triggers, key=lambda tr: SEVERITY[tr.severity])
        reentry = min((tr.reentry for tr in triggers), key=STAGES.index)
        return {"severity": lead.severity, "reentry": reentry,
                "names": [tr.name for tr in triggers], "lead": lead.name}
