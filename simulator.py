import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def simulate_route(route, twin, risk_multiplier=1.0, fuel_multiplier=1.0):
    """
    Estimate fuel, time, and risk for a given route.
    risk_multiplier / fuel_multiplier let different routes experience
    different effective conditions.
    """
    total_distance = 0.0
    for i in range(len(route.waypoints) - 1):
        wp1 = route.waypoints[i]
        wp2 = route.waypoints[i + 1]
        total_distance += haversine_distance(wp1.lat, wp1.lon, wp2.lat, wp2.lon)

    speed = twin.vessel.navigation.speed
    speed_kmh = speed * 1.852

    base_fuel_rate = 0.15
    wind_penalty = 1 + (twin.environment.weather.wind_speed / 50)
    wave_risk = twin.environment.ocean.wave_height / 10

    fuel = total_distance * base_fuel_rate * wind_penalty * fuel_multiplier
    time_hours = total_distance / speed_kmh if speed_kmh > 0 else float('inf')
    risk = min(
        (wave_risk + (twin.environment.weather.wind_speed / 100)) * risk_multiplier,
        1.0
    )

    return {
        "distance": round(total_distance, 2),
        "fuel": round(fuel, 2),
        "time": round(time_hours, 2),
        "risk": round(risk, 3)
    }


def simulate_all_routes(routes, twin):
    """
    Run simulate_route on every candidate route, applying different
    risk/fuel multipliers per route (straight, north detour, south detour).
    """
    multipliers = [
        {"risk": 1.0, "fuel": 1.0},   # Route 1: straight — baseline
        {"risk": 1.6, "fuel": 1.1},   # Route 2: north detour — rougher
        {"risk": 0.6, "fuel": 1.05},  # Route 3: south detour — calmer
    ]

    results = []
    for i, route in enumerate(routes):
        m = multipliers[i % len(multipliers)]
        results.append(simulate_route(route, twin, risk_multiplier=m["risk"], fuel_multiplier=m["fuel"]))
    return results