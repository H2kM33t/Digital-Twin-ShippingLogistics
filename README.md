# TwinRoute-M: a closed-loop Digital Twin for storm routing (TADIF)

Type any start and end port (or "lat, lon"). The ship follows the real sea
route (shipping lanes through Suez, Malacca, Panama, Hormuz...), and every
detour the twin considers stays in the water. Place and drag moving storms on
the map. The Digital Twin estimates where the
ship is, decides **when** it needs to re-plan, searches routes, speeds and
waiting times against an ensemble of possible storm tracks, rejects anything
too risky (CVaR), ranks the rest with an explanation, and **steers the
ship** with the result. It also learns the ship's real fuel burn in flight
and, if you save a voyage, remembers it for the next one.

The design follows the *TwinRoute-M / TADIF Technical Specification*; the
table below says how far each part goes.

## Run

    pip install -r requirements.txt
    python app.py
    # open http://127.0.0.1:5000   (internet needed for map tiles)
    # another port:  PORT=5050 python app.py

## Use the map

- **Route**: type a start and end, as a port name ("Shanghai", "Jebel Ali",
  "Rotterdam, Netherlands"; suggestions come from about 3,900 ports) or
  "lat, lon", then **Plan sea route** or Enter. ⇄ swaps them. The panel shows
  the sea distance and the straits or canals passed. Each new route gets a
  demo storm timed to cross the lane about when the ship gets there
  (**Reset demo** brings it back).

- **＋ Add storm**, then click the sea. Drag the numbered marker to move it.
- Storm sliders: radius, severity, heading, storm speed, lifetime, and
  **Detected at (h)**, the hour the twin first learns of the storm. Set it to
  about 18 to watch the ship re-plan in flight; 0 plans the detour at departure.
- Ship & mission: engine (TADIF or the legacy A* engine), **mission profile**
  (spec Table 5.1 weights), service speed, **risk limit** (max CVaR),
  **deadline slack**, and **GPS fault at (h)** to inject a drifting GPS.
- **Play / scrub** to replay. The status box shows the triggers that fired
  each hour; **Latest decision** shows why the twin chose what it did
  (trigger, search size, risk limits, objective vector, CVaR confidence
  interval, weights, sensitivity); the candidate table and dashed map lines
  show the alternatives, including the ones the risk filter rejected.
- The yellow ring is the twin's *estimated* position (UKF); the ship icon is
  the true one.
- **Complete voyage → update twin** saves the voyage to
  `data/knowledge_base.json`; the next voyage starts from what was learned.
  **Reset memory** forgets it.

## How it fits together

    start/end ─► routing/ocean.py   port lookup, searoute sea lane, 1 km land mask
    storms ─► routing/            sea-lane lattice (land nodes/edges removed),
                 │                hazard field on the sphere, A* (legacy engine)
    tadif/orchestrator.py  TadifVoyage: every hour
       estimation.py   Φ_state     UKF on noisy GPS (+ commanded velocity), NIS statistics
       twin_bridge.py              wind / waves / visibility near storms
       learning.py     Φ_learn     Bayesian update of the hull/fuel factor θ, knowledge base
       health.py       Φ_health    Twin Confidence Index from 4 live sub-scores, bands
       triggers.py                 event-driven re-planning + arbitration (Ch. 7)
     and, only when a trigger fires, one decision cycle:
       scenarios.py    Φ_scenario  48–128 perturbed storm tracks, k-means reduction
       objectives.py   Φ_weight    softmax weight modulation (Eq. 5.21)
       optimization.py Φ_opt       NSGA-III over lanes × speed × waiting time
       simulation.py   Φ_sim       batch sim: candidates × scenarios × segments × storms
       risk.py         Φ_risk      CVaR_α, P(core), bootstrap CI, acceptance (Eq. 9.5)
       decision.py     Φ_dec       TOPSIS + PROMETHEE II, sensitivity, explanation
    app.py + static/storm_map.html   Flask API + Leaflet front end

The selected route, speed and waiting time are what the ship sails. If no
candidate passes the risk filter, the ship holds position when that is safer
than any route. After `max_halt_h` (36 h) it sails the minimum-CVaR route
anyway (FORCED). Waiting is also something NSGA-III can choose on its own
(the hold gene), so the ship can let a storm pass instead of detouring
around it.

Decisions follow the decision-output schema of spec Section 15.4
(`selected_route.objective_vector`, `explanation.score / runner_up_gap /
dominant_criteria / weight_sensitivity / twin_confidence_at_decision`,
`rejected_by_risk`, `twin_health`).

## Spec coverage

| Operator | Implemented | Simplified or missing |
|---|---|---|
| Φ_state | UKF on [x, y, vx, vy] with GPS noise, commanded-velocity input, NIS-based sensor confidence, R inflation on drift | no PINN process model; hull degradation is estimated by Φ_learn instead |
| Φ_graph | lattice along the real sea lane, land nodes and land-crossing edges removed (1 km mask), time-dependent storm hazard evaluated at arrival time, storm geometry on the sphere | no real forecast data, bathymetry or under-keel clearance |
| Φ_scenario | perturbed heading/speed/radius/severity/position, error growing with lead time; k-means reduction with weights | not driven by real NWP ensembles; spread is configured, not learned |
| Φ_sim | vectorized batch sim with speed loss and added resistance, cubic fuel law × θ | one fixed-point pass, no RK4 or hull stress |
| Φ_weight | six criteria, Table 5.1 profiles, Eq. 5.21 softmax with deadline, weather and confidence pressures | no operator override |
| Φ_opt | NSGA-III (Das–Dennis references, normalization, niching), route-aware crossover and mutation, constrained domination | 4 objectives inside the search, all 6 in the ranking |
| Φ_risk | weighted CVaR (Rockafellar–Uryasev by sorting), P(core) and standard error, bootstrap 95 % CI, Eq. 9.5 filter, conservatism in the CRITICAL band | no adaptive stopping rule; one loss channel (no hull stress) |
| Φ_dec | TOPSIS, PROMETHEE II, weight-sensitivity sweep, dominant criteria, explanation text, hysteresis against flapping | |
| Φ_learn | conjugate Bayesian θ update, rolling fuel error, persistent knowledge base | no PINN fine-tuning or preference learning |
| Φ_health | Eq. 12.1–12.5 with live inputs, Table 12.1 bands, band-change triggers, Eq. 9.1 scenario scaling | |
| Triggers | departure, storm detected, 6-hourly forecast, storm-field delta, core crossing, sensor drift, fuel error, low confidence, hold review | mission update, port congestion and comms loss are not simulated |

Two deliberate deviations from the spec text: "ETA variance" is measured as
RMS hours past the deadline, and "reliability" as expected hours late. With
the literal definitions, waiting for a storm to clear always looked cheaper,
even when the ship was already late.

## Tests

    python -m pytest tests -q

`tests/test_tadif.py` covers each operator and the closed loop: calm voyage,
in-flight detour, waiting out a big storm, forced transit, GPS fault, the
decision schema, and learning across voyages. `tests/test_ocean.py` covers
sea routing: Suez routing, typed places, every planned route staying in the
water, storm motion on the sphere, and the API. The original engine's tests
(`test_storm_reroute.py`, `test_twin_bridge.py`) still pass.

## Performance

- Seed routes come from dynamic programming over the lattice (a DAG), not
  repeated A* searches.
- Storm distances use dot products on Earth-centred coordinates, with no
  trigonometry per element.
- Once per forecast, the twin works out which route columns any scenario storm
  can reach; hazard maths is skipped elsewhere (it is exactly zero there).
- If no seed route meets any storm, NSGA-III is skipped (spec Alg. 14.3
  "no-op" fast path).
- NSGA-III warm-starts from the previous cycle's population and stops when
  its first front stalls.
- Frames reference shared plan paths instead of repeating them.

Typical times: about 2 s for a calm regional voyage, 2–3 s for an in-flight
storm, 7 s for a large storm that forces waiting, and about 12 s for
Shanghai → Rotterdam (19,600 km).

## Limits

- Sea lanes come from searoute's maritime network and the land mask is about
  1 km; there is no bathymetry, so shallow water is not avoided. The centre
  lane (the sea lane itself) is always allowed, so narrow canals and port
  approaches finer than the mask still work.
- Storms move along great circles at constant speed and do not weaken over
  land.
- The true hull fuel factor (1.12) is a fixed demo value, unknown to the twin.
