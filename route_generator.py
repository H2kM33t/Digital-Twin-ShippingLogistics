from models import Waypoint, Route
import math


def generate_straight_route(origin, destination, num_points=5):
    lat1, lon1 = origin
    lat2, lon2 = destination
    route = Route()
    for i in range(num_points + 1):
        t = i / num_points
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        route.add_waypoint(Waypoint(lat=lat, lon=lon))
    return route


def generate_avoidance_route(origin, destination, avoid_point, avoid_strength=1.0, num_points=5):
    """
    Generate a route that bulges AWAY from a specific danger point
    (e.g. storm center), instead of a fixed pre-set direction.
    avoid_strength: 0 = no detour, higher = bigger detour.
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    avoid_lat, avoid_lon = avoid_point

    route = Route()
    for i in range(num_points + 1):
        t = i / num_points
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)

        # Never displace the endpoints. Without this, a route generated to
        # avoid a point that sits at (or very near) the origin or
        # destination -- e.g. the no-storm fallback routes below, which
        # pass the origin/destination themselves in as the "danger"
        # point -- pushes waypoint 0 or waypoint num_points away from the
        # actual origin/destination via the dist_to_danger<=0.001 fallback,
        # so the route visibly doesn't start/end where it's supposed to.
        if i == 0 or i == num_points:
            route.add_waypoint(Waypoint(lat=lat, lon=lon))
            continue

        # how close this point on the direct path is to the danger zone
        dist_to_danger = math.hypot(lat - avoid_lat, lon - avoid_lon)
        influence = max(0, 3.0 - dist_to_danger)  # only nearby points get pushed

        # push away from the danger point, scaled by strength and how close it is
        if dist_to_danger > 0.001:
            push_lat = (lat - avoid_lat) / dist_to_danger
            push_lon = (lon - avoid_lon) / dist_to_danger
        else:
            push_lat, push_lon = 1.0, 0.0

        lat += push_lat * influence * avoid_strength
        lon += push_lon * influence * avoid_strength

        route.add_waypoint(Waypoint(lat=lat, lon=lon))
    return route


# ---------------------------------------------------------------------
# Long-haul, real-geometry routes (Suez vs Cape of Good Hope)
# ---------------------------------------------------------------------
# Unlike generate_candidate_routes() below -- which deforms a straight
# line between two arbitrary points -- these are two FIXED, real-world
# route geometries between Rotterdam and Singapore, threaded through
# actual navigable pinch points (Gibraltar, Sicilian Channel, Suez,
# Bab-el-Mandeb / Cape Point, Agulhas). The choice here isn't "which
# shape avoids a hazard" (route_generator's job for the storm case) --
# both geometries are fixed in advance; what changes is which one the
# optimizer prefers as security_risk severity in the Suez corridor
# rises. That risk differentiation happens in simulator.py.

SHARED_DEPARTURE = [(51.9, 4.5), (49.0, -2.0), (43.3, -9.5)]

SUEZ_WAYPOINTS = SHARED_DEPARTURE + [
    (36.0, -5.6), (37.2, 1.0), (37.8, 9.0), (37.0, 11.8),
    (34.2, 24.0), (32.0, 30.0), (31.257, 32.301), (29.973, 32.550),
    (26.0, 35.2), (20.2, 38.9), (15.0, 41.2), (12.66, 43.42),
    (12.0, 50.0), (8.0, 65.0), (5.5, 80.3), (2.85, 100.9), (1.3, 103.8),
]

CAPE_WAYPOINTS = SHARED_DEPARTURE + [
    (36.0, -9.5), (27.0, -19.0), (14.5, -18.5), (4.0, -5.0),
    (-15.0, 8.0), (-30.0, 15.5), (-34.5, 18.2), (-35.5, 22.0),
    (-34.0, 27.5), (-30.0, 32.0), (-27.0, 47.0), (-15.0, 60.0),
    (2.85, 100.9), (1.3, 103.8),
]

# Bounding box for the elevated-risk corridor: Red Sea / Gulf of Aden /
# Bab-el-Mandeb (lat_min, lon_min, lat_max, lon_max). Matches RISK_ZONE
# in twinroute_ecdis_cape_vs_suez_demo.html.
SECURITY_RISK_ZONE = (10.0, 32.0, 30.0, 44.0)


def _route_from_coords(coords):
    route = Route()
    for lat, lon in coords:
        route.add_waypoint(Waypoint(lat=lat, lon=lon))
    return route


def generate_longhaul_routes():
    """
    Return the two fixed long-haul candidates: [Suez route, Cape route].
    Index 0 = Suez (shorter, passes through the security risk zone).
    Index 1 = Cape of Good Hope (longer, avoids it entirely).
    """
    return [_route_from_coords(SUEZ_WAYPOINTS), _route_from_coords(CAPE_WAYPOINTS)]


def generate_candidate_routes(origin, destination, storm_point=None, storm_severity=0.0):
    """
    Generate candidate routes. If storm_point is given, routes will
    dynamically curve away from it, scaled by storm_severity.
    """
    routes = [generate_straight_route(origin, destination)]

    if storm_point is not None and storm_severity > 0:
        # Two dynamic avoidance routes with different avoidance strength
        routes.append(generate_avoidance_route(origin, destination, storm_point, avoid_strength=storm_severity * 2.0))
        routes.append(generate_avoidance_route(origin, destination, storm_point, avoid_strength=storm_severity * 4.0))
    else:
        # No storm known yet -> fall back to old fixed detours
        lat1, lon1 = origin
        lat2, lon2 = destination
        routes.append(generate_avoidance_route(origin, destination, (lat1, lon1), avoid_strength=0.5))
        routes.append(generate_avoidance_route(origin, destination, (lat2, lon2), avoid_strength=0.5))

    return routes