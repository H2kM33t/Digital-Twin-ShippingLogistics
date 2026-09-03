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

**route_generator.py** — Generates 3 candidate routes between origin and destination: a direct route and two detour routes (north/south).

**simulator.py** — Calculates distance, fuel consumption, ETA, and risk per route, with per-route multipliers so different routes reflect meaningfully different conditions.

**optimizer.py** — Two-stage decision process: (1) Pareto dominance filtering removes routes strictly worse than another route on every metric; (2) TOPSIS ranking scores the remaining routes by closeness to an ideal solution. Based on Chapters 8.5 and 10.1 of the TADIF specification.

**visualizer.py** — Plots all candidate routes, origin, destination, vessel position, and the recommended route (highlighted) on a 2D map.

**digital_twin_generator.py** — A standalone, more realistic vessel simulator supporting scenario injection (extreme_weather, engine_overheating, fuel_leak, rpm_drop, sensor_gps_anomaly, and more), used to drive realistic environmental conditions into the main pipeline.

**mains.py** — Runs the core pipeline once under fixed sample conditions: generate routes -> simulate -> optimize -> visualize.

**adaptive_demo.py** — The centerpiece adaptive demo. Generates a realistic "normal" voyage state, evaluates all routes, then generates a "storm" state (extreme_weather scenario) and re-evaluates. Prints a before/after comparison and reports whether the recommended route changed, then visualizes both scenarios.

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
