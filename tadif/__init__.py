"""
TADIF: the TwinRoute Adaptive Decision Intelligence Framework.

One module per computational operator of the specification (Chapter 15):

    estimation.py    Phi_state     unscented Kalman filter over GPS fixes
    lattice.py       Phi_graph     vectorized view of the routing lattice
    scenarios.py     Phi_scenario  storm-track ensemble + scenario reduction
    simulation.py    Phi_sim       batch voyage simulation (candidates x scenarios)
    objectives.py    Phi_weight    mission profile + softmax weight modulation
    optimization.py  Phi_opt       NSGA-III over lane paths and speed
    risk.py          Phi_risk      Monte Carlo CVaR, failure probability, bootstrap CI
    decision.py      Phi_dec       TOPSIS / PROMETHEE II, sensitivity, explanation
    learning.py      Phi_learn     Bayesian fuel-model update + knowledge base
    health.py        Phi_health    Twin Confidence Index with escalation
    triggers.py      adaptive trigger engine (Chapter 7)
    orchestrator.py  TadifVoyage: the closed loop that actually steers the ship
"""
