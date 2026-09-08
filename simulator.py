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


def route_storm_exposure(route, storm_point, storm_radius_deg=3.0):
    """
    How much of this route actually passes close to the storm center,
    in degrees (lat/lon units, consistent with route_generator's own
    avoid_strength math -- this is a mini-project, not a geodesic model).

    Returns a 0-1 exposure score: 1.0 = a waypoint sits right on the
    storm center, 0.0 = every waypoint is storm_radius_deg or farther away.
    This is what makes a detour route actually *earn* a lower risk score,
    instead of getting a flat multiplier regardless of its shape.
    """
    if storm_point is None:
        return 0.0
    storm_lat, storm_lon = storm_point
    closest = min(
        math.hypot(wp.lat - storm_lat, wp.lon - storm_lon)
        for wp in route.waypoints
    )
    return max(0.0, 1.0 - min(closest / storm_radius_deg, 1.0))


def simulate_route(route, twin, risk_multiplier=1.0, fuel_multiplier=1.0,
                    storm_point=None, storm_severity=0.0):
    """
    Estimate fuel, time, and risk for a given route.
    Considers wind, wave height, wave period, visibility, AND (if a storm
    is active) how close this specific route's path actually passes to the
    storm center -- so a route that genuinely detours around the storm is
    rewarded with real lower risk, not just a flat per-route multiplier.
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

    # Low visibility increases risk (fog, storm, heavy rain)
    visibility_risk = max(0, (10 - twin.environment.weather.visibility) / 10) * 0.3

    # Short wave period = choppier, more violent sea state = extra risk
    wave_period_risk = max(0, (8 - twin.environment.ocean.wave_period) / 8) * 0.15

    # Route-specific storm exposure: routes that actually pass near the
    # storm center take a real risk penalty scaled by how severe it is;
    # routes that genuinely detour around it don't.
    exposure = route_storm_exposure(route, storm_point)
    storm_exposure_risk = exposure * storm_severity * 0.9

    fuel = total_distance * base_fuel_rate * wind_penalty * fuel_multiplier
    time_hours = total_distance / speed_kmh if speed_kmh > 0 else float('inf')

    # Cap the weather-only component below 1.0 so it leaves headroom for
    # storm_exposure_risk to actually differentiate routes even in a severe
    # storm. Without this cap, once ambient wind/wave risk alone saturates
    # every route at 1.0, a route that genuinely detours around the storm
    # loses its advantage purely to clipping, not because it's equally risky.
    weather_risk = min(
        (wave_risk
         + (twin.environment.weather.wind_speed / 100)
         + visibility_risk
         + wave_period_risk) * risk_multiplier,
        0.75
    )
    risk = min(weather_risk + storm_exposure_risk, 1.0)

    return {
        "distance": round(total_distance, 2),
        "fuel": round(fuel, 2),
        "time": round(time_hours, 2),
        "risk": round(risk, 3)
    }


def simulate_all_routes(routes, twin, storm_point=None, storm_severity=0.0):
    multipliers = [
        {"risk": 1.0, "fuel": 1.0},
        {"risk": 1.2, "fuel": 1.15},
        {"risk": 0.9, "fuel": 1.3},
    ]
    results = []
    for i, route in enumerate(routes):
        m = multipliers[i % len(multipliers)]
        results.append(simulate_route(
            route, twin, risk_multiplier=m["risk"], fuel_multiplier=m["fuel"],
            storm_point=storm_point, storm_severity=storm_severity
        ))
    return results