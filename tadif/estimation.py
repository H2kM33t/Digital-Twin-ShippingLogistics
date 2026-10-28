"""
Phi_state: state estimation with an unscented Kalman filter (spec Section 5.1).

State x = [x, y, vx, vy] in the voyage's km frame (constant-velocity process
model with white-acceleration noise); measurement z = noisy GPS fix [x, y].
The twin plans from the filtered position, not the raw fix.

Every update yields the normalized innovation squared (NIS). For a healthy
sensor its rolling mean is ~2 (two measurement dimensions); a drifting or
faulty GPS pushes it up. That gives the sensor sub-score of the Twin
Confidence Index (Eq. 12.1) and the sensor-drift trigger, whose response is
to inflate the measurement noise R (Table 7.1).

Not implemented from the spec: the PINN process model and the hidden
hydrodynamic parameters inside the filter state. Hull/fuel degradation is
estimated separately by the learning engine (learning.py).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class UnscentedKalmanFilter:

    def __init__(self, x0, P0, Q_accel: float, R_std: float,
                 alpha: float = 0.5, beta: float = 2.0, kappa: float = 0.0) -> None:
        self.x = np.asarray(x0, float)
        self.P = np.asarray(P0, float)
        self.q = Q_accel
        self.R = np.eye(2) * R_std ** 2
        n = len(self.x)
        self.lam = alpha ** 2 * (n + kappa) - n
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha ** 2 + beta)
        self.n = n

    def _sigma(self):
        S = np.linalg.cholesky((self.n + self.lam) * (self.P + 1e-9 * np.eye(self.n)))
        return np.vstack([self.x, self.x + S.T, self.x - S.T])

    @staticmethod
    def _f(X, dt, u):
        X = X.copy()
        if u is not None:
            # the twin knows its own helm/engine orders: u is the mean
            # velocity it expects over the step (plan geometry, ordered
            # speed, forecast speed loss); the process noise covers the rest
            X[:, :2] += np.asarray(u) * dt
            X[:, 2:] = u
        else:
            X[:, 0] += X[:, 2] * dt
            X[:, 1] += X[:, 3] * dt
        return X

    def predict(self, dt: float, u=None) -> None:
        X = self._f(self._sigma(), dt, u)
        self.x = self.Wm @ X
        D = X - self.x
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q ** 2
        self.P = (self.Wc[:, None] * D).T @ D + Q

    def update(self, z) -> float:
        X = self._sigma()
        Z = X[:, :2]
        z_hat = self.Wm @ Z
        dZ, dX = Z - z_hat, X - self.x
        S = (self.Wc[:, None] * dZ).T @ dZ + self.R
        C = (self.Wc[:, None] * dX).T @ dZ
        K = C @ np.linalg.inv(S)
        nu = np.asarray(z, float) - z_hat
        self.x = self.x + K @ nu
        self.P = self.P - K @ S @ K.T
        self.P = (self.P + self.P.T) / 2
        return float(nu @ np.linalg.solve(S, nu))


class VesselStateEstimator:
    """GPS sensor model + UKF + rolling innovation statistics."""

    def __init__(self, pos_xy, vel_xy, gps_std_km: float = 0.5,
                 gps_fault_h: float | None = None, fault_drift_kmh: float = 3.0,
                 seed: int = 11, window: int = 6) -> None:
        self.ukf = UnscentedKalmanFilter(
            [pos_xy[0], pos_xy[1], vel_xy[0], vel_xy[1]],
            np.diag([1.0, 1.0, 4.0, 4.0]), Q_accel=0.8, R_std=gps_std_km)
        self.gps_std = gps_std_km
        self.fault_h = gps_fault_h
        self.drift = fault_drift_kmh
        self.rng = np.random.default_rng(seed)
        self.nis = deque(maxlen=window)
        self.r_inflation = 1.0

    def measure(self, true_xy, t: float):
        """Simulated GPS fix; after `gps_fault_h` a growing bias appears."""

        z = np.asarray(true_xy, float) + self.rng.normal(0, self.gps_std, 2)
        if self.fault_h is not None and t >= self.fault_h:
            z += self.drift * (t - self.fault_h) * np.array([0.8, -0.6])
        return z

    def step(self, true_xy, t: float, dt: float, u=None) -> dict:
        """u: commanded velocity (km/h, x/y) over the last step, if known."""
        if dt > 0:
            self.ukf.predict(dt, u)
        nis = self.ukf.update(self.measure(true_xy, t))
        self.nis.append(nis)
        return self.snapshot(true_xy)

    def inflate_noise(self, factor: float = 4.0) -> None:
        """Sensor-drift response: trust the GPS less."""

        self.r_inflation *= factor
        self.ukf.R *= factor
        self.nis.clear()

    @property
    def position(self):
        return (float(self.ukf.x[0]), float(self.ukf.x[1]))

    def confidence(self) -> float:
        """Eq. 12.1: exp(-(mean NIS / expected NIS - 1)+)."""

        if not self.nis:
            return 1.0
        ratio = float(np.mean(self.nis)) / 2.0
        return float(np.exp(-max(ratio - 1.0, 0.0) / 2.0))

    def snapshot(self, true_xy) -> dict:
        err = float(np.hypot(self.ukf.x[0] - true_xy[0], self.ukf.x[1] - true_xy[1]))
        return {"pos_std_km": round(float(np.sqrt(self.ukf.P[0, 0] + self.ukf.P[1, 1])), 3),
                "error_km": round(err, 3),
                "nis": round(float(np.mean(self.nis)) if self.nis else 0.0, 2),
                "sog_kn": round(float(np.hypot(*self.ukf.x[2:])) / 1.852, 2),
                "r_inflation": self.r_inflation}
