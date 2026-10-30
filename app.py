"""
app.py

Local web app for the storm-rerouting demo.

    python app.py            ->  http://127.0.0.1:5000

Type a start and an end (a port name such as "Mumbai" or "Rotterdam", or
"lat, lon"). The voyage follows the real sea route (routing/ocean.py:
shipping lanes through Suez, Malacca, Panama...), and every detour the
twin considers stays in the water (1 km land mask). Place, drag and tune
storms on the map; every change is POSTed to /api/simulate, which runs one
of two engines:

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

import math
import os

from flask import Flask, jsonify, request, send_from_directory

from routing.hazard import Storm
from routing.ocean import haversine_km, resample, resolve_place, sea_route, search_ports
from routing.voyage import VoyageConfig, VoyageSimulator
from tadif.learning import KnowledgeBase
from tadif.objectives import PROFILES
from tadif.orchestrator import TadifConfig, TadifVoyage
from twin_bridge import TwinObserver

app = Flask(__name__, static_folder="static")

DEFAULT_ORIGIN = "Mumbai, India"
DEFAULT_DESTINATION = "Dubai, United Arab Emirates"

MAX_STORMS = 8
MIN_ROUTE_KM = 150
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
        lon=_num(d, "lon", 64.0, -540, 540),
        radius_km=_num(d, "radius_km", 150, 20, 2000),
        severity=_num(d, "severity", 0.9, 0.05, 1.0),
        heading_deg=_num(d, "heading_deg", 0, 0, 360),
        speed_kmh=_num(d, "speed_kmh", 0, 0, 120),
        lifetime_h=None if life in (None, "", 0) else _num(d, "lifetime_h", 48, 1, 1000),
        detect_h=_num(d, "detect_h", 0, 0, 2000),
        name=str(d.get("name") or f"Storm {i + 1}")[:40],
    )


class RouteError(ValueError):
    pass


def resolve_route(origin_text: str, destination_text: str):
    o = resolve_place(origin_text or DEFAULT_ORIGIN)
    d = resolve_place(destination_text or DEFAULT_DESTINATION)
    if o is None:
        raise RouteError(f"Could not find a port or coordinate for start: {origin_text!r}")
    if d is None:
        raise RouteError(f"Could not find a port or coordinate for end: {destination_text!r}")
    if haversine_km((o["lat"], o["lon"]), (d["lat"], d["lon"])) < MIN_ROUTE_KM:
        raise RouteError(f"Start and end are less than {MIN_ROUTE_KM} km apart.")
    return o, d


def demo_storm(path, speed_kts: float = 16.0) -> dict:
    """A storm that crosses the route ~45 % of the way along, detected in
    flight so the twin has to re-plan."""

    pts = resample(path, 41)
    i = 18
    (la1, lo1), (la2, lo2) = pts[i], pts[i + 1]
    brg = math.degrees(math.atan2(math.sin(math.radians(lo2 - lo1)) * math.cos(math.radians(la2)),
                                  math.cos(math.radians(la1)) * math.sin(math.radians(la2))
                                  - math.sin(math.radians(la1)) * math.cos(math.radians(la2))
                                  * math.cos(math.radians(lo2 - lo1))))
    side = (brg - 90.0) % 360.0                      # start to the port side
    off = 260.0                                      # km away from the lane
    lat = la1 + off * math.cos(math.radians(side)) / 111.2
    lon = lo1 + off * math.sin(math.radians(side)) / (111.2 * max(math.cos(math.radians(la1)), 0.2))
    total = sum(haversine_km(a, b) for a, b in zip(path[:-1], path[1:]))
    t_reach = 0.45 * total / (speed_kts * 1.852)
    radius = min(max(total * 0.06, 120.0), 400.0)
    # storms move from t = 0: time the drift so it reaches the lane about
    # when the ship does, whatever the voyage length
    drift = round(min(12.0, max(0.5, off / max(t_reach, 1.0))), 1)
    return {"name": "Storm 1", "lat": round(lat, 3), "lon": round(lon, 3),
            "radius_km": round(radius), "severity": 0.9,
            "heading_deg": round((side + 180.0) % 360.0), "speed_kmh": drift,
            "lifetime_h": None, "detect_h": round(max(0.0, t_reach - 30.0))}


def _kb_view() -> dict:
    kb = KnowledgeBase(KB_PATH)
    return {"theta_fuel": kb.data["theta_fuel"], "voyages": kb.data["voyages"][-10:],
            "n_voyages": len(kb.data["voyages"])}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "storm_map.html")


@app.get("/api/config")
def config():
    return jsonify({"origin": DEFAULT_ORIGIN, "destination": DEFAULT_DESTINATION,
                    "profiles": list(PROFILES)})


@app.get("/api/places")
def places():
    return jsonify(search_ports(request.args.get("q", ""), 12))


@app.get("/api/route")
def route():
    """Resolve the two places and return the sea route (fast preview)."""

    try:
        o, d = resolve_route(request.args.get("origin", ""), request.args.get("destination", ""))
        sea = sea_route((o["lat"], o["lon"]), (d["lat"], d["lon"]))
    except RouteError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"origin": o, "destination": d, "path": sea["path"],
                    "km": round(sea["length_km"], 1), "passages": sea["passages"],
                    "demo_storm": demo_storm(sea["path"])})


@app.post("/api/simulate")
def simulate():
    body = request.get_json(silent=True) or {}

    raw = body.get("storms", [])
    if not isinstance(raw, list) or len(raw) > MAX_STORMS:
        return jsonify({"error": f"storms must be a list of at most {MAX_STORMS}"}), 400

    try:
        o, d = resolve_route(body.get("origin", ""), body.get("destination", ""))
    except RouteError as e:
        return jsonify({"error": str(e)}), 400
    origin, destination = (o["lat"], o["lon"]), (d["lat"], d["lon"])

    storms = [parse_storm(s, i) for i, s in enumerate(raw) if isinstance(s, dict)]
    speed = _num(body, "speed_kts", 16, 4, 40)
    halt_risk = _num(body, "halt_risk", 0.55, 0.1, 1.0)

    try:
        if body.get("engine") == "graph":
            cfg = VoyageConfig(speed_kts=speed, halt_risk=halt_risk)
            voyage = VoyageSimulator(origin, destination, storms, cfg)
            result = voyage.run(observer=TwinObserver(origin, destination))
            result["engine"] = "graph"
        else:
            profile = body.get("profile", "cargo_standard")
            if profile not in PROFILES:
                return jsonify({"error": f"unknown profile {profile!r}"}), 400
            gps_fault = _num(body, "gps_fault_h", 0, 0, 2000)
            cfg = TadifConfig(
                speed_kts=speed, halt_risk=halt_risk, profile=profile,
                deadline_slack=_num(body, "deadline_slack", 0.15, 0.0, 2.0),
                gps_fault_h=gps_fault or None,
                # every run starts from the stored knowledge; only "learn" writes back
                kb_path=KB_PATH, kb_write=bool(body.get("learn")),
            )
            voyage = TadifVoyage(origin, destination, storms, cfg)
            result = voyage.run()
            result["engine"] = "tadif"
            result["knowledge"] = _kb_view()
    except ValueError as e:                      # e.g. no sea route through the corridor
        return jsonify({"error": f"Could not build a sea route: {e}"}), 400

    result["origin"] = origin
    result["destination"] = destination
    result["origin_label"], result["destination_label"] = o["label"], d["label"]
    result["sea_route"] = {"path": [[round(a, 4), round(b, 4)] for a, b in voyage.centerline],
                           "km": round(voyage.route_km, 1), "passages": voyage.passages}
    return jsonify(result)


@app.get("/api/knowledge")
def knowledge():
    return jsonify(_kb_view())


@app.post("/api/knowledge/reset")
def knowledge_reset():
    KnowledgeBase(KB_PATH).reset()
    return jsonify(_kb_view())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
