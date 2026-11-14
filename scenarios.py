"""Hard-coded, seeded demo scenarios (synthetic but realistic: real city coordinates, weekday windows)."""
import random

import pandas as pd

DEPOTS = {
    "Atlanta, GA DC": {"name": "Atlanta, GA DC", "lat": 33.749, "lon": -84.388},
    "Columbus, OH DC": {"name": "Columbus, OH DC", "lat": 39.961, "lon": -82.999},
}

SOUTHEAST = [
    ("Macon GA", 32.841, -83.632), ("Savannah GA", 32.081, -81.091), ("Augusta GA", 33.474, -82.010),
    ("Columbus GA", 32.461, -84.988), ("Athens GA", 33.951, -83.357), ("Albany GA", 31.578, -84.156),
    ("Valdosta GA", 30.833, -83.280), ("Rome GA", 34.257, -85.165), ("Dalton GA", 34.770, -84.970),
    ("Gainesville GA", 34.298, -83.824), ("Birmingham AL", 33.519, -86.810), ("Montgomery AL", 32.367, -86.300),
    ("Huntsville AL", 34.730, -86.586), ("Chattanooga TN", 35.046, -85.309), ("Knoxville TN", 35.961, -83.921),
    ("Nashville TN", 36.163, -86.782), ("Greenville SC", 34.853, -82.394), ("Columbia SC", 34.001, -81.035),
    ("Charlotte NC", 35.227, -80.843), ("Jacksonville FL", 30.332, -81.656), ("Tallahassee FL", 30.438, -84.281),
    ("Spartanburg SC", 34.950, -81.932), ("Anniston AL", 33.660, -85.831), ("LaGrange GA", 33.039, -85.031),
    ("Miami FL", 25.762, -80.192), ("Memphis TN", 35.149, -90.049), ("New Orleans LA", 29.951, -90.072),
]

MIDWEST = [
    ("Dayton OH", 39.759, -84.192), ("Cincinnati OH", 39.103, -84.512), ("Cleveland OH", 41.499, -81.694),
    ("Akron OH", 41.081, -81.519), ("Toledo OH", 41.654, -83.537), ("Canton OH", 40.799, -81.378),
    ("Youngstown OH", 41.100, -80.650), ("Mansfield OH", 40.758, -82.515), ("Zanesville OH", 39.940, -82.013),
    ("Lima OH", 40.743, -84.105), ("Springfield OH", 39.924, -83.809), ("Athens OH", 39.329, -82.101),
    ("Indianapolis IN", 39.768, -86.158), ("Fort Wayne IN", 41.079, -85.139), ("Richmond IN", 39.829, -84.890),
    ("Muncie IN", 40.193, -85.386), ("Louisville KY", 38.253, -85.759), ("Lexington KY", 38.040, -84.504),
    ("Pittsburgh PA", 40.441, -79.996), ("Erie PA", 42.129, -80.085), ("Detroit MI", 42.331, -83.046),
    ("Ann Arbor MI", 42.281, -83.743), ("Charleston WV", 38.350, -81.633), ("Huntington WV", 38.419, -82.445),
    ("Wheeling WV", 40.064, -80.721), ("Chicago IL", 41.878, -87.630), ("Minneapolis MN", 44.978, -93.265),
    ("Dallas TX", 32.777, -96.797), ("Boston MA", 42.360, -71.059),
]


def _make(cities, n, seed, hazmat_share=0.0, tight_share=0.25):
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        name, lat, lon = cities[k % len(cities)] if k < len(cities) else rng.choice(cities)
        lat += rng.uniform(-0.05, 0.05)
        lon += rng.uniform(-0.05, 0.05)
        weight = rng.choice([400, 800, 1200, 2000, 3500, 5000, 7500, 9000])
        volume = round(weight / rng.uniform(9, 14))
        day = rng.randint(0, 4)  # Mon..Fri, no weekend delivery
        if rng.random() < tight_share:
            start = rng.choice([8, 10, 13])
            e, l = 24 * day + start, 24 * day + start + 4                    # 4-hour appointment
        else:
            d2 = min(4, day + rng.randint(0, 1))
            e, l = 24 * day + 8, 24 * d2 + 17                                 # 1-2 business-day window
        rows.append({
            "shipment_id": f"S{k+1:03d}", "customer": name,
            "lat": round(lat, 4), "lon": round(lon, 4), "weight_lb": weight, "volume_cuft": volume,
            "earliest_h": e, "latest_h": l,
            "cargo_class": "HAZMAT" if rng.random() < hazmat_share else "GENERAL",
        })
    return pd.DataFrame(rows)


def toy_instance():
    """5-shipment test case used in the wiki's hand-worked example (TC-F01)."""
    return pd.DataFrame([
        {"shipment_id": "T1", "customer": "Dayton OH", "lat": 39.759, "lon": -84.192, "weight_lb": 6000,
         "volume_cuft": 500, "earliest_h": 8, "latest_h": 17, "cargo_class": "GENERAL"},
        {"shipment_id": "T2", "customer": "Cincinnati OH", "lat": 39.103, "lon": -84.512, "weight_lb": 8000,
         "volume_cuft": 600, "earliest_h": 8, "latest_h": 17, "cargo_class": "GENERAL"},
        {"shipment_id": "T3", "customer": "Indianapolis IN", "lat": 39.768, "lon": -86.158, "weight_lb": 7000,
         "volume_cuft": 550, "earliest_h": 8, "latest_h": 41, "cargo_class": "GENERAL"},
        {"shipment_id": "T4", "customer": "Cleveland OH", "lat": 41.499, "lon": -81.694, "weight_lb": 500,
         "volume_cuft": 40, "earliest_h": 8, "latest_h": 17, "cargo_class": "GENERAL"},
        {"shipment_id": "T5", "customer": "Denver CO", "lat": 39.739, "lon": -104.990, "weight_lb": 3000,
         "volume_cuft": 250, "earliest_h": 8, "latest_h": 113, "cargo_class": "GENERAL"},
    ])


SCENARIOS = {
    "S1 · Southeast weekly lane plan (Atlanta, 40 shipments)":
        ("Atlanta, GA DC", lambda: _make(SOUTHEAST, 40, seed=11)),
    "S2 · Midwest bid project (Columbus, 60 shipments, incl. HAZMAT & far-away)":
        ("Columbus, OH DC", lambda: _make(MIDWEST, 60, seed=23, hazmat_share=0.08)),
    "S3 · Toy test case TC-F01 (Columbus, 5 shipments)":
        ("Columbus, OH DC", toy_instance),
}
