import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def simulate_route(route, twin):
    """
    Estimate fuel, time, and risk for a given route based on the
    current Digital Twin's environment state.
    Returns a dict: {fuel, time, risk, distance}
    """
    total_distance = 0.0
    for i in range(len(route.waypoints) - 1):
        wp1 = route.waypoints[i]
        wp2 = route.waypoints[i + 1]
        total_distance += haversine_distance(wp1.lat, wp1.lon, wp2.lat, wp2.lon)

    speed = twin.vessel.navigation.speed  # knots
    speed_kmh = speed * 1.852  # convert knots to km/h

    # Base fuel rate per km (simplified assumption)
    base_fuel_rate = 0.15  # tonnes per km

    # Environmental penalties
    wind_penalty = 1 + (twin.environment.weather.wind_speed / 50)
    wave_risk = twin.environment.ocean.wave_height / 10  # normalized 0-1 roughly

    fuel = total_distance * base_fuel_rate * wind_penalty
    time_hours = total_distance / speed_kmh if speed_kmh > 0 else float('inf')
    risk = min(wave_risk + (twin.environment.weather.wind_speed / 100), 1.0)

    return {
        "distance": round(total_distance, 2),
        "fuel": round(fuel, 2),
        "time": round(time_hours, 2),
        "risk": round(risk, 3)
    }


def simulate_all_routes(routes, twin):
    """Run simulate_route on every candidate route, return list of performance dicts."""
    return [simulate_route(r, twin) for r in routes]