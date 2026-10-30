"""
ocean.py

Real-world sea geometry for the routing engine.

    sea_route()      shipping-lane route between two points (searoute's
                     maritime network: straits and canals such as Suez,
                     Malacca, Panama), used as the corridor centerline
    is_water()       1 km global land mask (global-land-mask)
    resample()       evenly spaced points along a polyline
    resolve_place()  "Rotterdam", "Jebel Ali, UAE" or "19.07, 72.88" -> (lat, lon)
    search_ports()   name lookup in searoute's port list (~3,900 ports)

Longitudes of a route are unwrapped (no jump at +/-180) so distances,
offsets and map drawing stay continuous across the antimeridian.
"""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

Coordinate = Tuple[float, float]
EARTH_RADIUS_KM = 6371.0


# ======================================================================
# Geometry
# ======================================================================

def haversine_km(a: Coordinate, b: Coordinate) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def to_ecef(lat, lon) -> np.ndarray:
    """(lat, lon) degrees -> (..., 3) points on a sphere of radius R km."""

    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return EARTH_RADIUS_KM * np.stack([np.cos(la) * np.cos(lo),
                                       np.cos(la) * np.sin(lo),
                                       np.sin(la)], axis=-1)


def from_ecef(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, float)
    lat = np.degrees(np.arctan2(p[..., 2], np.hypot(p[..., 0], p[..., 1])))
    lon = np.degrees(np.arctan2(p[..., 1], p[..., 0]))
    return lat, lon


def unwrap_lons(points: List[Coordinate]) -> List[Coordinate]:
    out, prev = [], None
    for lat, lon in points:
        if prev is not None:
            while lon - prev > 180:
                lon -= 360
            while lon - prev < -180:
                lon += 360
        out.append((lat, lon))
        prev = lon
    return out


def _slerp(a: Coordinate, b: Coordinate, f: float) -> Coordinate:
    pa, pb = to_ecef(*a), to_ecef(*b)
    omega = math.acos(max(-1.0, min(1.0, float(pa @ pb) / EARTH_RADIUS_KM ** 2)))
    if omega < 1e-9:
        return a
    p = (math.sin((1 - f) * omega) * pa + math.sin(f * omega) * pb) / math.sin(omega)
    lat, lon = from_ecef(p)
    lon = float(lon)
    while lon - a[1] > 180:
        lon -= 360
    while lon - a[1] < -180:
        lon += 360
    return (float(lat), lon)


def polyline_km(points: List[Coordinate]) -> float:
    return sum(haversine_km(a, b) for a, b in zip(points[:-1], points[1:]))


def resample(points: List[Coordinate], n: int) -> List[Coordinate]:
    """n points evenly spaced by distance along the polyline (great-circle legs)."""

    seg = [haversine_km(a, b) for a, b in zip(points[:-1], points[1:])]
    total = sum(seg)
    if total <= 0 or n < 2:
        return [points[0]] * max(n, 1)
    out, i, acc = [], 0, 0.0
    for k in range(n):
        target = total * k / (n - 1)
        while i < len(seg) - 1 and acc + seg[i] < target:
            acc += seg[i]
            i += 1
        f = 0.0 if seg[i] <= 0 else min(max((target - acc) / seg[i], 0.0), 1.0)
        out.append(_slerp(points[i], points[i + 1], f))
    out[0], out[-1] = points[0], points[-1]
    return out


# ======================================================================
# Land mask
# ======================================================================

def is_water(lat, lon) -> np.ndarray:
    """Vectorized: True where the 1 km global land mask says ocean."""

    from global_land_mask import globe

    lat = np.clip(np.asarray(lat, float), -89.999, 89.999)
    lon = (np.asarray(lon, float) + 180.0) % 360.0 - 180.0
    return np.asarray(globe.is_ocean(lat, lon))


# ======================================================================
# Sea route
# ======================================================================

@lru_cache(maxsize=64)
def _sea_route_cached(origin: Coordinate, destination: Coordinate):
    import searoute

    feat = searoute.searoute([origin[1], origin[0]], [destination[1], destination[0]],
                             append_orig_dest=True, return_passages=True)
    coords = [(float(lat), float(lon)) for lon, lat in feat["geometry"]["coordinates"]]
    pts = [coords[0]]
    for p in coords[1:]:
        if haversine_km(pts[-1], p) > 0.5:          # drop duplicates
            pts.append(p)
    if len(pts) < 2:
        pts = [origin, destination]
    pts = unwrap_lons(pts)
    passages = feat["properties"].get("traversed_passages") or []
    return tuple(pts), tuple(passages)


def sea_route(origin: Coordinate, destination: Coordinate) -> dict:
    """Shipping-lane polyline from origin to destination."""

    pts, passages = _sea_route_cached((round(origin[0], 5), round(origin[1], 5)),
                                      (round(destination[0], 5), round(destination[1], 5)))
    pts = list(pts)
    return {"path": pts, "length_km": polyline_km(pts), "passages": list(passages)}


# ======================================================================
# Places
# ======================================================================

@lru_cache(maxsize=1)
def _ports() -> list:
    import searoute

    path = os.path.join(os.path.dirname(searoute.__file__), "data", "ports.geojson")
    with open(path, encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    out = []
    for f in feats:
        p = f["properties"]
        if not p.get("name"):
            continue
        lon, lat = f["geometry"]["coordinates"]
        country = (p.get("cty") or "").replace("_", " ").title()
        out.append({"name": p["name"], "country": country,
                    "code": p.get("port", ""), "lat": float(lat), "lon": float(lon)})
    return out


def _label(p: dict) -> str:
    return f"{p['name']}, {p['country']}" if p["country"] else p["name"]


def search_ports(query: str, limit: int = 12) -> list:
    q = query.strip().lower()
    if not q:
        return []
    exact, starts, contains = [], [], []
    for p in _ports():
        label = _label(p).lower()
        if p["name"].lower() == q or p["code"].lower() == q:
            exact.append(p)
        elif label.startswith(q):
            starts.append(p)
        elif q in label:
            contains.append(p)
    hits = exact + sorted(starts, key=lambda p: len(p["name"])) + contains
    return [{"label": _label(p), "lat": p["lat"], "lon": p["lon"]} for p in hits[:limit]]


_LATLON = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$")


def resolve_place(text: str) -> Optional[dict]:
    """Port name ("Rotterdam", "Jebel Ali, United Arab Emirates") or "lat, lon"."""

    if not text:
        return None
    m = _LATLON.match(text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if -85 <= lat <= 85 and -180 <= lon <= 180:
            return {"label": f"{lat:.4f}, {lon:.4f}", "lat": lat, "lon": lon}
        return None
    q = text.strip().lower()
    exact = [p for p in _ports() if _label(p).lower() == q or p["name"].lower() == q]
    if exact:
        p = exact[0]
        return {"label": _label(p), "lat": p["lat"], "lon": p["lon"]}
    hits = search_ports(text, 1)
    return hits[0] if hits else None
