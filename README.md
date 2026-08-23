# Digital Twin - Shipping Logistics (TwinRoute-M Mini Project)

A simplified, working implementation of a maritime Digital Twin decision-support system for adaptive voyage route planning — inspired by the TwinRoute-M system design blueprint.

This is a scaled-down educational/demo version of a much larger proposed architecture. It focuses on the core concept: maintaining a synchronized digital representation of a vessel and its environment, generating candidate routes, and visualizing them — without the production-scale infrastructure (no database, backend server, or live data feeds).

---

## Concept

Traditional ship routing picks a single route before departure and never updates it. This project demonstrates the alternative: a Digital Twin — a live virtual model of the vessel and its environment — that can generate multiple candidate routes and visually compare them, forming the foundation for adaptive, continuously-updated voyage planning.

Core cycle (from the original blueprint):
Observe -> Synchronize -> Predict -> Simulate -> Optimize -> Adapt

---

## Project Structure

Digital Twin in Shipping Logistics/
├── models.py           - Data model: Vessel, Environment, Mission states
├── route_generator.py  - Generates candidate voyage routes
├── visualizer.py        - Plots routes on a lat/lon map using matplotlib
├── mains.py              - Entry point, runs the full pipeline
├── requirements.txt       - Python dependencies
├── .gitignore
└── README.md

### File Descriptions

models.py
Defines the core data structures using Python dataclasses: VesselState, EnvironmentState, MissionState, Waypoint, Route, and the combined DigitalTwin object. No logic, just structure.

route_generator.py
Generates 3 candidate routes between origin and destination: one straight line and two "detour" routes (north/south bulge) to simulate route alternatives.

visualizer.py
Uses matplotlib to plot all candidate routes, origin, destination, current vessel position, and (eventually) the recommended route on a 2D lat/lon map.

mains.py
Builds a sample DigitalTwin, generates routes, prints route data to console, and displays the visualization.

---

## Requirements

- Python 3.10 or newer
- pip (comes with Python)

### Dependencies
- matplotlib - for route visualization

---

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/H2kM33t/Digital-Twin-ShippingLogistics.git
cd Digital-Twin-ShippingLogistics

### 2. (Optional but recommended) Create a virtual environment
python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

If you don't have a requirements.txt yet, create one with:
matplotlib

Or install directly:
pip install matplotlib

---

## Running the Project

python mains.py

### Expected output

1. Console: The full DigitalTwin object printed, followed by each candidate route's waypoints (lat/lon).

2. Popup window: A matplotlib chart showing:
   - Origin (black square)
   - Destination (black star)
   - Current vessel position (cyan triangle)
   - Dashed colored lines for each candidate route
   - Red solid line for the currently "recommended" route (placeholder logic for now)

---

## Troubleshooting

No plot window appears:
- Confirm matplotlib is installed: pip show matplotlib
- Confirm plot_routes(...) is actually called at the bottom of mains.py
- Try forcing a backend at the top of visualizer.py:
  import matplotlib
  matplotlib.use('TkAgg')
- As a fallback, save the plot to a file instead of showing a window:
  plt.savefig("route_plot.png")

ImportError: cannot import name 'X' from 'models':
- Make sure models.py is saved with the full class definitions and sits in the same folder as mains.py.

Indentation errors:
- Python is whitespace-sensitive, make sure code inside if __name__ == "__main__": and for loops is indented consistently (4 spaces recommended).

---

## Roadmap

[x] Digital Twin data model (models.py)
[x] Candidate route generation (route_generator.py)
[x] Route visualization (visualizer.py)
[ ] Simulation engine - predict fuel consumption, ETA, and risk per route
[ ] Multi-objective scoring/optimization - rank routes by fuel/time/risk trade-offs
[ ] Decision support logic - select and highlight the recommended route dynamically
[ ] Adaptive re-routing demo - simulate changing weather mid-voyage and show route re-evaluation
[ ] (Stretch) Simple web dashboard using FastAPI + Plotly

---

## Background

This project is a simplified, single-file-per-module implementation based on a full system design blueprint (TwinRoute-M) covering vision, architecture, mathematical formulation, and software implementation guidelines for a production-scale maritime Digital Twin platform. This repository implements only the core conceptual pipeline for educational and demonstration purposes.

## License

(Add a license here if you want, e.g., MIT, or leave this section out if unsure.)