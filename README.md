# Digital Twin - Shipping Logistics (TwinRoute-M Mini Project)

A simplified, working implementation of a maritime Digital Twin decision-support system for adaptive voyage route planning — inspired by the TwinRoute-M system design blueprint and the TADIF (TwinRoute Adaptive Decision Intelligence Framework) technical specification.

This project implements a scoped subset of a much larger proposed research framework, focusing on the core decision-making loop: synchronized vessel/environment state, candidate route generation, multi-objective route evaluation, Pareto-optimal decision ranking, and adaptive re-routing in response to changing conditions.

---

## Concept

Traditional ship routing picks a single route before departure and never updates it. This project demonstrates the alternative: a Digital Twin that generates multiple candidate routes, evaluates them under realistic simulated conditions, ranks them using multi-criteria decision analysis, and re-evaluates its recommendation when conditions change (e.g., a storm).

Core cycle:
Observe -> Synchronize -> Predict -> Simulate -> Optimize -> Adapt

---

## Project Structure

Digital Twin in Shipping Logistics/
├── models.py                  - Data model: Vessel, Environment, Mission states
├── route_generator.py         - Generates candidate voyage routes
├── simulator.py                - Estimates fuel/time/risk per route (with per-route condition variation)
├── optimizer.py                 - Pareto dominance filtering + TOPSIS multi-criteria ranking
├── visualizer.py                 - Plots routes on a lat/lon map using matplotlib
├── digital_twin_generator.py       - Physics-based synthetic vessel/environment simulator with scenario support (storms, engine faults, etc.)
├── mains.py                          - Entry point: single-scenario pipeline demo
├── adaptive_demo.py                    - Before/after storm comparison demo showing adaptive re-routing
├── requirements.txt
├── .gitignore
└── README.md

### File Descriptions

**models.py** — Defines core data structures using Python dataclasses: VesselState, EnvironmentState, MissionState, Waypoint, Route, and the combined DigitalTwin object.

**route_generator.py** — Generates 3 candidate routes between origin and destination: a direct route and two detour routes (north/south), curving dynamically away from a storm point when one is active. Also provides `generate_longhaul_routes()`: two FIXED, real-world route geometries between Rotterdam and Singapore — Suez Canal vs. Cape of Good Hope — threaded through actual navigable pinch points (Gibraltar, Sicilian Channel, Suez, Bab-el-Mandeb, Cape Point, Agulhas). Unlike the storm case, neither geometry deforms; which one wins is purely a simulator/optimizer decision.

**simulator.py** — Calculates distance, fuel consumption, ETA, and risk per route, with per-route multipliers so different routes reflect meaningfully different conditions. Risk has two independent, route-shape-aware sources: `route_storm_exposure()` (how close a route's waypoints pass to a storm center, with radius falloff) and `route_security_exposure()` (what fraction of a route's waypoints fall inside a fixed geographic risk corridor, e.g. the Red Sea / Bab-el-Mandeb box used for the Suez-vs-Cape scenario) — so a route only takes a risk penalty it actually earns by its geometry, not a flat per-route multiplier.

**optimizer.py** — Two-stage decision process: (1) Pareto dominance filtering removes routes strictly worse than another route on every metric; (2) TOPSIS ranking scores the remaining routes by closeness to an ideal solution. Based on Chapters 8.5 and 10.1 of the TADIF specification.

**visualizer.py** — Plots all candidate routes, origin, destination, vessel position, and the recommended route (highlighted) on a 2D map.

**digital_twin_generator.py** — A standalone, more realistic vessel simulator supporting scenario injection (extreme_weather, engine_overheating, fuel_leak, rpm_drop, sensor_gps_anomaly, and more), used to drive realistic environmental conditions into the main pipeline.

**mains.py** — Runs the core pipeline once under fixed sample conditions: generate routes -> simulate -> optimize -> visualize.

**adaptive_demo.py** — The centerpiece adaptive demo. Generates a realistic "normal" voyage state, evaluates all routes, then generates a "storm" state (extreme_weather scenario) and re-evaluates. Prints a before/after comparison and reports whether the recommended route changed, then visualizes both scenarios. Also includes `run_storm_journey()`, which steps a voyage forward continuously through a building storm.

**objective_builder.py** — Adaptive Objective Builder (Phi_weight, TADIF Ch 5.5). Derives fuel/time/risk optimizer weights from the current mission state (low fuel, rough weather, poor visibility) instead of using fixed weights, with `explain_weights()` for a human-readable explanation of why weights shifted. `build_adaptive_weights()` also accepts an `extra_risk_pressure` signal from outside weather (e.g. a security/geopolitical advisory severity), combined with weather-driven pressure via `max()` so both express the same "how much should safety dominate right now" signal without double-counting.

**twin_health.py** — Digital Twin health monitor (Phi_health, TADIF Ch 5.10 + 12). Combines sensor/forecast/model/decision confidence sub-scores into one 0-1 confidence value and a NOMINAL/ADVISORY/CRITICAL escalation band. The decision sub-score is computed properly from real TOPSIS scores; the other three are documented proxies standing in for spec components (Kalman filter, PINN, Monte Carlo ensemble) this mini project doesn't implement.

**export_journey.py** — Two export functions, both running the real pipeline forward through a checkpoint loop and writing one JSON record per checkpoint:
- `export_journey()` — digital_twin_generator → route_generator → simulator → objective_builder → optimizer → twin_health, forward over a simulated voyage through a building storm. Writes `journey.json` (candidate route waypoints, storm location/severity, per-route `routes_performance` fuel/time/risk for every candidate, and twin_health output).
- `export_longhaul_journey()` — route_generator's `generate_longhaul_routes()` (fixed Suez/Cape geometries) → simulator → objective_builder → optimizer → twin_health, forward over the Rotterdam → Singapore voyage while a security-risk advisory in the Red Sea corridor ramps up from 0 to 1 over a configurable window. Writes `journey_suez_cape.json`, with the same record shape plus `route_names`, `security_risk_active`, and `security_risk_severity`. The checkpoint where `route_changed` first flips true is the actual, non-scripted moment the optimizer switches its recommendation from Suez to Cape.

Both run when the script is executed directly (`python export_journey.py`).

**twinroute_storm_demo.html** — Browser playback of `journey.json`, entirely data-driven (no scripted narrative anywhere in this one — every number on screen comes from a checkpoint record). Draws the real candidate routes and storm location for each checkpoint, animates the ship along the recommended route, tags Pareto-optimal candidates in the route legend, and shows a live per-route fuel/time/risk comparison table built from `routes_performance` — simulator.py's actual output for every candidate, not just the one the optimizer picked. Also shows the live confidence gauge and event log as the recommendation changes, with a scrubber, play/pause, auto-restart on reaching the end, and space/arrow-key playback controls. Needs a local HTTP server (`python -m http.server`) since `fetch()` is blocked on a plain `file://` page.
- The candidate-route avoidance bulge (`generate_avoidance_route()`) is real but geographically small (~150–300km) relative to the full voyage span, so the map zooms into the storm's neighborhood the moment it forms — instead of staying zoomed out to the whole route — so the deviation between candidate routes is actually visible on screen; it zooms back out once the storm clears. Playback is 45s end-to-end (was 30s) so the storm ramp-up and reroute aren't over before you can see them.

**twinroute_ecdis_cape_vs_suez_demo.html** — Browser playback of `journey_suez_cape.json`, styled as an ECDIS-style navigation console. The Suez and Cape route geometries drawn on the map are the same fixed waypoint lists `route_generator.py` uses to build the candidates; the on-screen replan moment, caption phases, confidence gauges, and event log are all timed off the real checkpoint where the loaded journey data shows `route_changed` (falling back to a hand-picked fraction only if the JSON fails to load, e.g. opened without a local server). Also overlays genuinely live sea-state data from the free Open-Meteo API along the route. Needs a local HTTP server, same as the storm demo.

---

## Requirements

- Python 3.10 or newer
- pip

### Dependencies
- matplotlib
- numpy
- pandas

---

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/H2kM33t/Digital-Twin-ShippingLogistics.git
cd Digital-Twin-ShippingLogistics

### 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate      (Windows)
source venv/bin/activate   (macOS/Linux)

### 3. Install dependencies
pip install -r requirements.txt

---

## Running the Project

### Basic pipeline (single scenario)
python mains.py

### Adaptive re-routing demo (recommended - the main feature)
python adaptive_demo.py

### Storm replan demo in the browser (real algorithm output, not scripted)
python export_journey.py
python -m http.server 8000
# then open http://localhost:8000/twinroute_storm_demo.html

### Suez vs. Cape of Good Hope ECDIS demo (real algorithm output, not scripted)
python export_journey.py
# (writes both journey.json and journey_suez_cape.json)
python -m http.server 8000
# then open http://localhost:8000/twinroute_ecdis_cape_vs_suez_demo.html

### Example output (adaptive_demo.py)

BEFORE: Normal Conditions
Route 1: Fuel=360.23t  Time=72.53h  Risk=0.257
Route 2: Fuel=396.73t  Time=72.62h  Risk=0.412
Route 3: Fuel=381.23t  Time=73.11h  Risk=0.154
Pareto-optimal routes: [1, 3]
>>> Recommended Route: Route 3

AFTER: Storm Detected (extreme_weather)
Route 1: Fuel=458.94t  Time=74.54h  Risk=0.86
Route 2: Fuel=505.45t  Time=74.63h  Risk=1.0
Route 3: Fuel=485.7t  Time=75.13h  Risk=0.516
Pareto-optimal routes: [1, 3]
>>> Recommended Route: Route 3

The system correctly identifies Route 3 as the safest option in both scenarios,
with risk scores roughly tripling across all routes once storm conditions hit -
demonstrating the twin's sensitivity to real environmental change.

---

## Troubleshooting

**ModuleNotFoundError (pandas/numpy/matplotlib):**
pip install pandas numpy matplotlib

**No plot window appears:**
Add this near the top of visualizer.py:
import matplotlib
matplotlib.use('TkAgg')

**ImportError: cannot import name 'X' from 'models':**
Ensure models.py contains full class definitions and sits in the same folder as the script being run.

---

## Roadmap

[x] Digital Twin data model
[x] Candidate route generation
[x] Route visualization
[x] Simulation engine (fuel/time/risk estimation)
[x] Multi-objective optimization (Pareto filtering + TOPSIS)
[x] Realistic physics-based scenario simulator (digital_twin_generator.py)
[x] Adaptive re-routing demo (storm before/after comparison)
[x] Twin health / confidence scoring (twin_health.py)
[x] Journey export + browser storm-replan demo (export_journey.py, twinroute_storm_demo.html)
[x] Suez vs. Cape of Good Hope security-risk scenario, real algorithm output (generate_longhaul_routes(), route_security_exposure(), export_longhaul_journey(), twinroute_ecdis_cape_vs_suez_demo.html)
[ ] Real Monte Carlo / CVaR risk quantification
[ ] Continual learning / online model adaptation
[ ] Graph-based environment representation
[ ] Simple API or dashboard wrapper

---

## Background

This project implements a scoped subset of two reference documents:
1. A system design blueprint defining the overall architecture (6 subsystems: Mission Manager, Data Acquisition, State Synchronization, Digital Twin Core, Scenario Simulation, Decision Support)
2. TADIF (TwinRoute Adaptive Decision Intelligence Framework) - a thesis-grade technical specification defining 10 computational operators for a full production-scale adaptive Digital Twin

This implementation covers the core decision-making loop (route generation, simulation, Pareto/TOPSIS-based optimization, and adaptive re-evaluation) as a demonstrative mini project. Advanced components described in the reference specification - such as online learning, Monte Carlo risk estimation, and graph-based environment modeling - were intentionally scoped out as beyond the mini-project scope.

## License

(Add a license if desired, e.g. MIT.)
