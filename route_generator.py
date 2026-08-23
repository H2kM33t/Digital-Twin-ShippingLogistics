from models import Waypoint, Route

def generate_straight_route(origin, destination, num_points=5):
    """Generate a simple straight-line route with interpolated waypoints."""
    lat1, lon1 = origin
    lat2, lon2 = destination

    route = Route()
    for i in range(num_points + 1):
        t = i / num_points
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        route.add_waypoint(Waypoint(lat=lat, lon=lon))
    return route


def generate_detour_route(origin, destination, offset, num_points=5):
    """Generate a route that bulges away from the straight line by 'offset' degrees."""
    lat1, lon1 = origin
    lat2, lon2 = destination

    route = Route()
    for i in range(num_points + 1):
        t = i / num_points
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)

        # Add a bulge in the middle of the route (peaks at t=0.5)
        bulge = offset * (1 - abs(2 * t - 1))
        lat += bulge

        route.add_waypoint(Waypoint(lat=lat, lon=lon))
    return route


def generate_candidate_routes(origin, destination):
    """Generate a small set of alternative candidate routes."""
    routes = [
        generate_straight_route(origin, destination),
        generate_detour_route(origin, destination, offset=1.0),   # north detour
        generate_detour_route(origin, destination, offset=-1.0),  # south detour
    ]
    return routes