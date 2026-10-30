"""Tests for real sea routing: shipping lanes, land mask, typed places, API."""

import numpy as np
import pytest

from routing.hazard import Storm, latlon_near
from routing.ocean import haversine_km, is_water, resample, resolve_place, sea_route
from tadif.orchestrator import TadifVoyage

MUMBAI = (19.0760, 72.8777)
DUBAI = (25.276987, 55.296249)
ROTTERDAM = (51.9244, 4.4777)


def test_sea_route_goes_through_suez_and_is_longer_than_the_great_circle():
    r = sea_route(MUMBAI, ROTTERDAM)
    assert "suez" in r["passages"]
    assert r["length_km"] > 1.3 * haversine_km(MUMBAI, ROTTERDAM)
    # the lane itself is at sea away from the two ports
    mid = resample(r["path"], 60)[5:-5]
    wet = is_water(np.array([p[0] for p in mid]), np.array([p[1] for p in mid]))
    assert wet.mean() > 0.9


def test_places_resolve_by_name_and_coordinates():
    assert resolve_place("Rotterdam")["label"].startswith("Rotterdam")
    assert resolve_place("Dubai, United Arab Emirates") is not None
    p = resolve_place("19.07, 72.88")
    assert (p["lat"], p["lon"]) == (19.07, 72.88)
    assert resolve_place("nowhere-port-xyz") is None


def test_storm_moves_on_the_sphere():
    s = Storm(10.0, 60.0, 100, 0.9, heading_deg=90, speed_kmh=100)
    lat, lon = latlon_near(s.centre(10.0), 60.0)
    assert haversine_km((10.0, 60.0), (lat, lon)) == pytest.approx(1000, rel=0.02)
    assert lon > 60.0


@pytest.fixture(scope="module")
def storm_voyage():
    v = TadifVoyage(MUMBAI, DUBAI, [Storm(19.0, 63.0, 160, 0.9, heading_deg=0,
                                          speed_kmh=12, detect_h=18)])
    return v, v.run()


def test_every_planned_route_stays_in_the_water(storm_voyage):
    v, res = storm_voyage
    lane = v.centerline
    lane_pts = np.array(resample(lane, 400))
    for path in res["paths"]:
        pts = np.array(path)
        wet = is_water(pts[:, 0], pts[:, 1])
        for p in pts[~wet]:
            # only the sea route itself may touch the mask (port approaches)
            d = min(haversine_km(tuple(p), tuple(q)) for q in lane_pts)
            assert d < 15, f"route node on land at {p}"


def test_ship_sails_the_sea_route_distance(storm_voyage):
    v, res = storm_voyage
    s = res["summary"]
    assert s["arrived"]
    assert s["sea_route_km"] > haversine_km(MUMBAI, DUBAI)        # not through land
    assert s["sailed_km"] >= s["baseline_km"] - 1


def test_api_plans_typed_route_and_simulates():
    from app import app
    c = app.test_client()
    r = c.get("/api/route?origin=Mumbai&destination=Jebel%20Ali").get_json()
    assert r["km"] > 1500 and r["demo_storm"]["radius_km"] > 0
    assert c.get("/api/places?q=rotter").get_json()[0]["label"].startswith("Rotterdam")
    bad = c.get("/api/route?origin=nowhere-port-xyz&destination=Dubai")
    assert bad.status_code == 400
    sim = c.post("/api/simulate", json={"origin": "Mumbai", "destination": "Jebel Ali",
                                        "storms": [r["demo_storm"]]}).get_json()
    assert sim["summary"]["arrived"] and sim["paths"]
    assert sim["origin_label"].startswith("Mumbai")
