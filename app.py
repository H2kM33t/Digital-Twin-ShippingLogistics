"""
app.py

Local web app for the storm-rerouting demo.

    python app.py            ->  http://127.0.0.1:5000

The page (static/storm_map.html) lets you place, drag and tune storms on a
map. Every change is POSTed to /api/simulate, which runs one of two engines:

  * "tadif" (default): the closed TADIF loop in tadif/. A UKF estimates the
    ship's position, a trigger engine decides when to re-plan, NSGA-III
    searches routes, speeds and waiting times against a storm scenario
    ensemble, a CVaR filter rejects unsafe ones, TOPSIS / PROMETHEE II pick
    one with an explanation, and that decision is what the ship sails. The
    learning engine refines the fuel model and, on request, stores the
    voyage in the knowledge base (data/knowledge_base.json).
  * "graph": the original engine (routing/voyage.py): A* replanning with
    the Digital Twin pipeline (twin_bridge.py) scoring three routes beside it.
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, request, send_from_directory

from routing.hazard import Storm
from routing.voyage import VoyageConfig, VoyageSimulator
from tadif.learning import KnowledgeBase
from tadif.objectives import PROFILES
from tadif.orchestrator import TadifConfig, TadifVoyage
from twin_bridge import TwinObserver

app = Flask(__name__, static_folder="static")

ORIGIN = (19.0760, 72.8777)            # Mumbai
DESTINATION = (25.276987, 55.296249)   # Dubai

MAX_STORMS = 8
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base.json")


def _num(d: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(d.get(key, default))
    except (TypeError, ValueError):
        v = default
    return min(max(v, lo), hi)


def parse_storm(d: dict, i: int) -> Storm:
    life = d.get("lifetime_h")
    return Storm(
        lat=_num(d, "lat", 22.0, -85, 85),
        lon=_num(d, "lon", 64.0, -180, 180),
        radius_km=_num(d, "radius_km", 150, 20, 2000),
        severity=_num(d, "severity", 0.9, 0.05, 1.0),
        heading_deg=_num(d, "heading_deg", 0, 0, 360),
        speed_kmh=_num(d, "speed_kmh", 0, 0, 120),
        lifetime_h=None if life in (None, "", 0) else _num(d, "lifetime_h", 48, 1, 1000),
        detect_h=_num(d, "detect_h", 0, 0, 1000),
        name=str(d.get("name") or f"Storm {i + 1}")[:40],
    )


def _kb_view() -> dict:
    kb = KnowledgeBase(KB_PATH)
    return {"theta_fuel": kb.data["theta_fuel"], "voyages": kb.data["voyages"][-10:],
            "n_voyages": len(kb.data["voyages"])}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "storm_map.html")


@app.get("/api/config")
def config():
    return jsonify({
        "origin": ORIGIN,
        "destination": DESTINATION,
        "profiles": list(PROFILES),
        "default_storms": [{
            "name": "Storm 1", "lat": 19.0, "lon": 63.0, "radius_km": 160,
            "severity": 0.9, "heading_deg": 0, "speed_kmh": 12,
            "lifetime_h": None, "detect_h": 18,
        }],
    })


@app.post("/api/simulate")
def simulate():
    body = request.get_json(silent=True) or {}

    raw = body.get("storms", [])
    if not isinstance(raw, list) or len(raw) > MAX_STORMS:
        return jsonify({"error": f"storms must be a list of at most {MAX_STORMS}"}), 400

    storms = [parse_storm(s, i) for i, s in enumerate(raw) if isinstance(s, dict)]
    speed = _num(body, "speed_kts", 16, 4, 40)
    halt_risk = _num(body, "halt_risk", 0.55, 0.1, 1.0)

    if body.get("engine") == "graph":
        cfg = VoyageConfig(speed_kts=speed, halt_risk=halt_risk)
        result = VoyageSimulator(ORIGIN, DESTINATION, storms, cfg).run(
            observer=TwinObserver(ORIGIN, DESTINATION))
        result["engine"] = "graph"
    else:
        profile = body.get("profile", "cargo_standard")
        if profile not in PROFILES:
            return jsonify({"error": f"unknown profile {profile!r}"}), 400
        gps_fault = _num(body, "gps_fault_h", 0, 0, 400)
        learn = bool(body.get("learn"))
        cfg = TadifConfig(
            speed_kts=speed, halt_risk=halt_risk, profile=profile,
            deadline_slack=_num(body, "deadline_slack", 0.15, 0.0, 2.0),
            gps_fault_h=gps_fault or None,
            # every run starts from the stored knowledge; only "learn" writes back
            kb_path=KB_PATH, kb_write=learn,
        )
        result = TadifVoyage(ORIGIN, DESTINATION, storms, cfg).run()
        result["engine"] = "tadif"
        result["knowledge"] = _kb_view()

    result["origin"] = ORIGIN
    result["destination"] = DESTINATION
    return jsonify(result)


@app.get("/api/knowledge")
def knowledge():
    return jsonify(_kb_view())


@app.post("/api/knowledge/reset")
def knowledge_reset():
    KnowledgeBase(KB_PATH).reset()
    return jsonify(_kb_view())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
