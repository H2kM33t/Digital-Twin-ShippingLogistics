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