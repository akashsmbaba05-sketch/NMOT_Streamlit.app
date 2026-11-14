"""
NMOT-Lite engine: Vehicle Routing Problem with Time Regulations and Common Carriers (VRPTRCC)
Prototype re-implementation inspired by Dang, Singh & Allen (2021), INFORMS J. Applied Analytics 51(3):179-199.

Decision: for every shipment, deliver with the DHL dedicated fleet (and on which route, in which order)
or "buy" delivery from a common carrier at its rated cost (lambda_i).

Pure Python + pandas (no solver needed) so it runs on Streamlit Community Cloud.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict

import pandas as pd

# ----------------------------------------------------------------------------------------------
# Parameters (defaults = "standard inputs" reported in the paper, Table 3)
# ----------------------------------------------------------------------------------------------
@dataclass
class Params:
    fixed_cost_per_day: float = 200.0     # f   : $ per truck-day
    cost_per_mile: float = 2.0            # c   : $ per mile
    cost_per_stop: float = 30.0           # p   : $ per unloading stop
    cap_weight_lb: float = 45000.0        # q   : truck weight capacity
    cap_volume_cuft: float = 3000.0       #       truck cube capacity
    cap_utilisation: float = 0.90         #       usable share of capacity
    speed_mph: float = 55.0               # g   : average speed
    min_unload_h: float = 0.5             # mu  : minimum unloading time
    unload_rate_cuft_h: float = 300.0     #       unloading rate
    max_stop_gap_mi: float = 120.0        # d_max between consecutive customer stops
    max_route_mi: float = 1200.0          # max route length
    max_duty_h: float = 14.0              # delta: HOS on-duty window
    layover_h: float = 10.0               # b    : mandatory rest
    max_layovers: int = 3                 #       per round trip
    circuity: float = 1.18                #       road miles / straight-line miles
    # ACO (paper Table 3 Panel B defaults where given)
    n_ants: int = 8
    n_iterations: int = 25
    q0: float = 0.9                       # exploitation probability
    beta: float = 0.9                     # visibility exponent
    rho: float = 0.1                      # evaporation
    candidate_list: int = 12              # nearest-neighbour candidate set
    seed: int = 7


# ----------------------------------------------------------------------------------------------
# Geometry & carrier rating
# ----------------------------------------------------------------------------------------------
def haversine_mi(lat1, lon1, lat2, lon2) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def ltl_rate(weight_lb: float, miles: float) -> float:
    """Illustrative common-carrier (LTL) tariff: $/cwt rising with distance, $175 minimum charge.
    The paper uses DHL's contracted tariffs (not published); this proxy keeps the same economics:
    small/far shipments are cheap to outsource, dense heavy clusters are cheaper on the fleet."""
    return round(max(175.0, (weight_lb / 100.0) * (9.0 + 0.045 * miles)), 2)


# ----------------------------------------------------------------------------------------------
# Instance
# ----------------------------------------------------------------------------------------------
class Instance:
    REQUIRED = ["shipment_id", "customer", "lat", "lon", "weight_lb", "volume_cuft",
                "earliest_h", "latest_h"]

    def __init__(self, depot: dict, shipments: pd.DataFrame, params: Params):
        self.p = params
        self.depot = depot
        df = shipments.copy().reset_index(drop=True)
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if "cargo_class" not in df.columns:
            df["cargo_class"] = "GENERAL"
        df["cargo_class"] = df["cargo_class"].fillna("GENERAL").astype(str).str.upper()
        df["depot_miles"] = [
            round(haversine_mi(depot["lat"], depot["lon"], r.lat, r.lon) * params.circuity, 1)
            for r in df.itertuples()
        ]
        if "carrier_cost" not in df.columns or df["carrier_cost"].isna().any():
            rated = [ltl_rate(w, m) for w, m in zip(df.weight_lb, df.depot_miles)]
            if "carrier_cost" in df.columns:
                df["carrier_cost"] = df["carrier_cost"].fillna(pd.Series(rated))
            else:
                df["carrier_cost"] = rated
        self.df = df
        self.n = len(df)
        # node 0 = depot, nodes 1..n = shipments (row i -> node i+1)
        lats = [depot["lat"]] + df.lat.tolist()
        lons = [depot["lon"]] + df.lon.tolist()
        self.D = [[0.0] * (self.n + 1) for _ in range(self.n + 1)]
        for i in range(self.n + 1):
            for j in range(i + 1, self.n + 1):
                d = haversine_mi(lats[i], lons[i], lats[j], lons[j]) * params.circuity
                self.D[i][j] = self.D[j][i] = d
        self.w = [0.0] + df.weight_lb.tolist()
        self.v = [0.0] + df.volume_cuft.tolist()
        self.e = [0.0] + df.earliest_h.tolist()
        self.a = [1e9] + df.latest_h.tolist()
        self.lam = [0.0] + df.carrier_cost.tolist()
        self.cls = ["DEPOT"] + df.cargo_class.tolist()
        self.ids = ["DEPOT"] + df.shipment_id.astype(str).tolist()
        self.unload = [0.0] + [max(params.min_unload_h, v / params.unload_rate_cuft_h) for v in self.v[1:]]
        # candidate lists (nearest neighbours) – the paper's "candidate set" speed-up
        self.cand = {}
        for i in range(self.n + 1):
            order = sorted((j for j in range(1, self.n + 1) if j != i), key=lambda j: self.D[i][j])
            self.cand[i] = order[: params.candidate_list] if i else order


# ----------------------------------------------------------------------------------------------
# Route simulation – the feasibility oracle (time windows + HOS + capacity + distance)
# ----------------------------------------------------------------------------------------------
@dataclass
class RouteEval:
    feasible: bool
    reason: str = ""
    miles: float = 0.0
    duration_h: float = 0.0
    days: int = 0
    layovers: int = 0
    cost: float = 0.0
    depart_h: float = 0.0
    timeline: list = field(default_factory=list)  # (activity, start_h, end_h, node)


def simulate(inst: Instance, route: list[int], want_timeline: bool = False) -> RouteEval:
    p = inst.p
    if not route:
        return RouteEval(True)
    # capacity
    if sum(inst.w[i] for i in route) > p.cap_weight_lb * p.cap_utilisation:
        return RouteEval(False, "weight capacity")
    if sum(inst.v[i] for i in route) > p.cap_volume_cuft * p.cap_utilisation:
        return RouteEval(False, "volume capacity")
    classes = {inst.cls[i] for i in route}
    if "HAZMAT" in classes and len(classes) > 1:
        return RouteEval(False, "cargo-class mixing")
    # distance rules
    seq = [0] + route + [0]
    miles = sum(inst.D[seq[k]][seq[k + 1]] for k in range(len(seq) - 1))
    if miles > p.max_route_mi:
        return RouteEval(False, "max route distance")
    for k in range(1, len(route)):
        if inst.D[route[k - 1]][route[k]] > p.max_stop_gap_mi:
            return RouteEval(False, "max distance between stops")

    # time simulation with Hours-of-Service
    tl = []
    first = route[0]
    t = max(0.0, inst.e[first] - inst.D[0][first] / p.speed_mph)
    depart = t
    duty, lay = 0.0, 0

    def drive(t, duty, lay, frm, to):
        remaining = inst.D[frm][to] / p.speed_mph
        while remaining > 1e-9:
            avail = p.max_duty_h - duty
            if avail <= 1e-9:
                if want_timeline:
                    tl.append(("Layover", t, t + p.layover_h, frm))
                t += p.layover_h
                duty, lay = 0.0, lay + 1
                continue
            step = min(avail, remaining)
            if want_timeline:
                tl.append(("Drive", t, t + step, to))
            t += step
            duty += step
            remaining -= step
        return t, duty, lay

    prev = 0
    for j in route:
        t, duty, lay = drive(t, duty, lay, prev, j)
        if t < inst.e[j]:
            wait = inst.e[j] - t
            if wait >= p.layover_h:          # overnight wait = rest
                if want_timeline:
                    tl.append(("Layover", t, inst.e[j], j))
                duty, lay = 0.0, lay + 1
            else:
                if want_timeline:
                    tl.append(("Wait", t, inst.e[j], j))
                duty += wait
            t = inst.e[j]
        if duty + inst.unload[j] > p.max_duty_h:  # must rest before unloading
            if want_timeline:
                tl.append(("Layover", t, t + p.layover_h, j))
            t += p.layover_h
            duty, lay = 0.0, lay + 1
        if t > inst.a[j] + 1e-9:
            return RouteEval(False, f"time window missed at {inst.ids[j]}")
        if want_timeline:
            tl.append(("Unload", t, t + inst.unload[j], j))
        t += inst.unload[j]
        duty += inst.unload[j]
        prev = j
    t, duty, lay = drive(t, duty, lay, prev, 0)
    if lay > p.max_layovers:
        return RouteEval(False, "too many layovers")
    dur = t - depart
    days = max(1, math.ceil(dur / 24 - 1e-9))
    cost = p.fixed_cost_per_day * days + p.cost_per_mile * miles + p.cost_per_stop * len(route)
    return RouteEval(True, "", round(miles, 1), round(dur, 2), days, lay, round(cost, 2), round(depart, 2), tl)


def route_cost(inst, route, cache):
    key = tuple(route)
    if key not in cache:
        r = simulate(inst, route)
        cache[key] = r.cost if r.feasible else math.inf
    return cache[key]


# ----------------------------------------------------------------------------------------------
# Local search: intra-route 2-opt (stand-in for the paper's DP local search)
# ----------------------------------------------------------------------------------------------
def two_opt(inst, route, cache):
    best, best_c = route[:], route_cost(inst, route, cache)
    improved = True
    while improved and len(best) > 2:
        improved = False
        for i in range(len(best) - 1):
            for k in range(i + 1, len(best)):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                c = route_cost(inst, cand, cache)
                if c < best_c - 1e-6:
                    best, best_c, improved = cand, c, True
    return best


# ----------------------------------------------------------------------------------------------
# Make-or-buy: Node Removal (NR) + profitable reinsertion
# ----------------------------------------------------------------------------------------------
def node_removal(inst, routes, outsourced, cache, reasons):
    routes = [r[:] for r in routes if r]
    outsourced = set(outsourced)
    for _ in range(4 * inst.n + 10):
        best_gain, best_move = 1e-6, None
        # 1) remove a node whose marginal fleet cost exceeds its carrier rate
        for ri, r in enumerate(routes):
            base = route_cost(inst, r, cache)
            for i in r:
                rest = [x for x in r if x != i]
                saving = base - (route_cost(inst, rest, cache) if rest else 0.0)
                gain = saving - inst.lam[i]
                if gain > best_gain:
                    best_gain, best_move = gain, ("remove", ri, i)
        # 2) reinsert an outsourced node where cheaper than the carrier
        for i in outsourced:
            for ri, r in enumerate(routes):
                base = route_cost(inst, r, cache)
                for pos in range(len(r) + 1):
                    c = route_cost(inst, r[:pos] + [i] + r[pos:], cache)
                    gain = inst.lam[i] - (c - base)
                    if gain > best_gain:
                        best_gain, best_move = gain, ("insert", ri, i, pos)
        if best_move is None:
            break
        if best_move[0] == "remove":
            _, ri, i = best_move
            routes[ri] = [x for x in routes[ri] if x != i]
            outsourced.add(i)
            reasons[i] = "Marginal fleet cost > carrier rate (node removal)"
        else:
            _, ri, i, pos = best_move
            routes[ri] = routes[ri][:pos] + [i] + routes[ri][pos:]
            outsourced.discard(i)
            reasons.pop(i, None)
        routes = [r for r in routes if r]
    # whole-route check: outsource entire route if carrier is cheaper
    keep = []
    for r in routes:
        if route_cost(inst, r, cache) > sum(inst.lam[i] for i in r):
            outsourced.update(r)
            for i in r:
                reasons[i] = "Whole route cheaper by carrier"
        else:
            keep.append(r)
    return keep, outsourced


# ----------------------------------------------------------------------------------------------
# Solution container
# ----------------------------------------------------------------------------------------------
@dataclass
class Solution:
    name: str
    routes: list
    outsourced: set
    reasons: dict
    fleet_cost: float
    carrier_cost: float

    @property
    def total(self):
        return round(self.fleet_cost + self.carrier_cost, 2)


def make_solution(inst, name, routes, outsourced, reasons):
    fleet = sum(simulate(inst, r).cost for r in routes)
    carrier = sum(inst.lam[i] for i in outsourced)
    return Solution(name, routes, set(outsourced), dict(reasons), round(fleet, 2), round(carrier, 2))


def infeasible_singletons(inst):
    forced, reasons = set(), {}
    for i in range(1, inst.n + 1):
        r = simulate(inst, [i])
        if not r.feasible:
            forced.add(i)
            reasons[i] = f"Forced to carrier – fleet infeasible ({r.reason})"
    return forced, reasons


# ----------------------------------------------------------------------------------------------
# Plan A: all common carrier
# ----------------------------------------------------------------------------------------------
def solve_all_carrier(inst):
    nodes = set(range(1, inst.n + 1))
    return make_solution(inst, "All common carrier", [], nodes, {i: "Policy: outsource all" for i in nodes})


# ----------------------------------------------------------------------------------------------
# Plan B: manual-style baseline (nearest neighbour, fleet-first, route-level check only)
# ----------------------------------------------------------------------------------------------
def solve_baseline(inst):
    cache = {}
    forced, reasons = infeasible_singletons(inst)
    unvisited = set(range(1, inst.n + 1)) - forced
    routes = []
    while unvisited:
        # start with earliest-deadline shipment (how planners typically sort)
        cur_route = [min(unvisited, key=lambda j: (inst.a[j], inst.D[0][j]))]
        unvisited.discard(cur_route[0])
        while True:
            last = cur_route[-1]
            opts = sorted(unvisited, key=lambda j: inst.D[last][j])
            nxt = next((j for j in opts if route_cost(inst, cur_route + [j], cache) < math.inf), None)
            if nxt is None:
                break
            cur_route.append(nxt)
            unvisited.discard(nxt)
        routes.append(cur_route)
    out = set(forced)
    keep = []
    for r in routes:
        if route_cost(inst, r, cache) > sum(inst.lam[i] for i in r):
            out.update(r)
            for i in r:
                reasons[i] = "Whole route cheaper by carrier"
        else:
            keep.append(r)
    return make_solution(inst, "Manual baseline (nearest-neighbour)", keep, out, reasons)


# ----------------------------------------------------------------------------------------------
# Plan C: NMOT – Ant Colony System + local search + node removal (NR-ACSLS style)
# ----------------------------------------------------------------------------------------------
def solve_nmot(inst, progress=None):
    p = inst.p
    rng = random.Random(p.seed)
    cache = {}
    forced, base_reasons = infeasible_singletons(inst)
    nodes = [i for i in range(1, inst.n + 1) if i not in forced]

    seed_sol = solve_baseline(inst)
    tau0 = 1.0 / max(1.0, inst.n * max(seed_sol.total, 1.0))
    tau = [[tau0] * (inst.n + 1) for _ in range(inst.n + 1)]
    eta = [[(1.0 / inst.D[i][j]) if inst.D[i][j] > 0 else 1.0 for j in range(inst.n + 1)]
           for i in range(inst.n + 1)]

    best = None
    history = []

    def attractiveness(i, j):
        return tau[i][j] * (eta[i][j] ** p.beta)

    for it in range(p.n_iterations):
        iter_best = None
        for _ in range(p.n_ants):
            unvisited = set(nodes)
            routes = []
            while unvisited:
                route, cur = [], 0
                while True:
                    pool = [j for j in inst.cand[cur] if j in unvisited] if cur else list(unvisited)
                    feas = [j for j in pool if route_cost(inst, route + [j], cache) < math.inf]
                    if not feas and cur:  # widen candidate list once before closing the route
                        feas = [j for j in unvisited if j not in pool
                                and route_cost(inst, route + [j], cache) < math.inf][:5]
                    if not feas:
                        break
                    if rng.random() < p.q0:
                        j = max(feas, key=lambda x: attractiveness(cur, x))
                    else:
                        wts = [attractiveness(cur, x) for x in feas]
                        j = rng.choices(feas, weights=wts, k=1)[0]
                    tau[cur][j] = (1 - p.rho) * tau[cur][j] + p.rho * tau0   # local update
                    route.append(j)
                    unvisited.discard(j)
                    cur = j
                if not route:   # should not happen (singletons are feasible) – safety
                    break
                routes.append(route)
            routes = [two_opt(inst, r, cache) for r in routes]
            reasons = dict(base_reasons)
            routes, out = node_removal(inst, routes, forced, cache, reasons)
            sol = make_solution(inst, "NMOT (ACO + node removal)", routes, out, reasons)
            if iter_best is None or sol.total < iter_best.total:
                iter_best = sol
        if best is None or iter_best.total < best.total:
            best = iter_best
        # global pheromone update on best-so-far
        for r in best.routes:
            seq = [0] + r + [0]
            for a, b in zip(seq, seq[1:]):
                tau[a][b] = (1 - p.rho) * tau[a][b] + p.rho / best.total
        history.append({"iteration": it + 1, "iteration_best": iter_best.total, "best_so_far": best.total})
        if progress:
            progress((it + 1) / p.n_iterations)
    best.history = history
    return best


# ----------------------------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------------------------
def shipment_table(inst, sol: Solution) -> pd.DataFrame:
    rows = []
    where = {}
    for k, r in enumerate(sol.routes, 1):
        for pos, i in enumerate(r, 1):
            where[i] = (f"R{k:02d}", pos)
    for i in range(1, inst.n + 1):
        rec = inst.df.iloc[i - 1]
        if i in where:
            mode, rid, pos, why = "DHL fleet", where[i][0], where[i][1], "Cheaper on consolidated route"
        else:
            mode, rid, pos, why = "Common carrier", "-", None, sol.reasons.get(i, "Carrier cheaper")
        rows.append({"shipment_id": rec.shipment_id, "customer": rec.customer, "weight_lb": rec.weight_lb,
                     "depot_miles": rec.depot_miles, "carrier_cost": rec.carrier_cost, "mode": mode,
                     "route": rid, "stop_seq": pos, "reason": why})
    return pd.DataFrame(rows)


def route_table(inst, sol: Solution) -> pd.DataFrame:
    rows = []
    for k, r in enumerate(sol.routes, 1):
        ev = simulate(inst, r)
        rows.append({"route": f"R{k:02d}", "stops": len(r), "sequence": " → ".join(inst.ids[i] for i in r),
                     "weight_lb": sum(inst.w[i] for i in r),
                     "util_weight_%": round(100 * sum(inst.w[i] for i in r) / inst.p.cap_weight_lb, 1),
                     "miles": ev.miles, "duration_h": ev.duration_h, "days": ev.days,
                     "layovers": ev.layovers, "fleet_cost": ev.cost,
                     "carrier_alt_cost": round(sum(inst.lam[i] for i in r), 2)})
    return pd.DataFrame(rows)


def timeline_table(inst, sol: Solution) -> pd.DataFrame:
    rows = []
    for k, r in enumerate(sol.routes, 1):
        ev = simulate(inst, r, want_timeline=True)
        for act, s, e, node in ev.timeline:
            rows.append({"route": f"R{k:02d}", "activity": act, "start_h": round(s, 2), "end_h": round(e, 2),
                         "at": inst.ids[node]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Automated validation – mirrors the test plan in the wiki
# ----------------------------------------------------------------------------------------------
def validate(inst, sol: Solution) -> pd.DataFrame:
    p = inst.p
    res = []
    served = [i for r in sol.routes for i in r]
    allnodes = set(range(1, inst.n + 1))
    dup = len(served) != len(set(served))
    both = set(served) & sol.outsourced
    missing = allnodes - set(served) - sol.outsourced
    res.append(("TC-V01", "Every shipment assigned to exactly one mode (A.4)",
                not dup and not both and not missing,
                f"duplicates={dup}, in both={len(both)}, unassigned={len(missing)}"))
    evs = [simulate(inst, r) for r in sol.routes]
    bad = [f"R{k+1:02d}:{e.reason}" for k, e in enumerate(evs) if not e.feasible]
    res.append(("TC-V02", "All fleet routes feasible on re-simulation", not bad, ", ".join(bad) or "all feasible"))
    over_w = [k + 1 for k, r in enumerate(sol.routes)
              if sum(inst.w[i] for i in r) > p.cap_weight_lb * p.cap_utilisation]
    res.append(("TC-V03", "Capacity respected (A.13–A.14)", not over_w, f"violations={over_w}"))
    lay = max([e.layovers for e in evs], default=0)
    res.append(("TC-V04", f"HOS: ≤{p.max_layovers} layovers per route (A.9–A.12)", lay <= p.max_layovers,
                f"max layovers={lay}"))
    mx = max([e.miles for e in evs], default=0)
    res.append(("TC-V05", f"Route length ≤ {p.max_route_mi:.0f} mi", mx <= p.max_route_mi, f"max={mx:.0f} mi"))
    recomputed = round(sum(e.cost for e in evs) + sum(inst.lam[i] for i in sol.outsourced), 2)
    res.append(("TC-V06", "Objective reconciles (A.1)", abs(recomputed - sol.total) < 0.05,
                f"reported={sol.total:,.2f}, recomputed={recomputed:,.2f}"))
    # make-or-buy rationality: no fleet route costs more than its carrier alternative
    irr = [k + 1 for k, r in enumerate(sol.routes) if evs[k].cost > sum(inst.lam[i] for i in r) + 1e-6]
    res.append(("TC-V07", "No route dearer than outsourcing it", not irr, f"violations={irr}"))
    return pd.DataFrame(res, columns=["test_id", "check", "passed", "detail"])


# ----------------------------------------------------------------------------------------------
# Unit / edge-case test suite (TC-F*, TC-E*) – same cases as the wiki test plan
# ----------------------------------------------------------------------------------------------
def run_unit_tests() -> pd.DataFrame:
    from scenarios import toy_instance, DEPOTS
    dep = DEPOTS["Columbus, OH DC"]
    out = []

    def row(tid, name, expected, actual, ok):
        out.append({"test_id": tid, "scenario": name, "expected": expected, "actual": actual, "passed": bool(ok)})

    # TC-F01 toy instance – hand-verified expected values
    inst = Instance(dep, toy_instance(), Params())
    sol = solve_nmot(inst)
    fleet_ids = sorted(inst.ids[i] for r in sol.routes for i in r)
    row("TC-F01", "Toy 5-shipment make-or-buy", "Fleet {T1,T2,T3}; carrier {T4,T5}; total ≈ $3,486",
        f"Fleet {set(fleet_ids)}; total ${sol.total:,.2f}",
        fleet_ids == ["T1", "T2", "T3"] and abs(sol.total - 3486.45) < 1)
    row("TC-F02", "NMOT ≤ baseline ≤ all-carrier", "monotone cost ordering",
        f"{sol.total:,.0f} ≤ {solve_baseline(inst).total:,.0f} ≤ {solve_all_carrier(inst).total:,.0f}",
        sol.total <= solve_baseline(inst).total <= solve_all_carrier(inst).total)

    def one(**kw):
        base = {"shipment_id": "X1", "customer": "Test", "lat": 39.103, "lon": -84.512, "weight_lb": 2000,
                "volume_cuft": 150, "earliest_h": 8, "latest_h": 17, "cargo_class": "GENERAL"}
        base.update(kw)
        return base

    # TC-E01 overweight
    inst = Instance(dep, pd.DataFrame([one(weight_lb=42000, volume_cuft=2000)]), Params())
    s = solve_nmot(inst)
    row("TC-E01", "Shipment above 90% truck capacity (42,000 lb)", "Forced to carrier",
        s.reasons.get(1, "on fleet"), 1 in s.outsourced and "capacity" in s.reasons.get(1, ""))
    # TC-E02 HOS layover insertion (Nashville from Columbus)
    inst = Instance(dep, pd.DataFrame([one(customer="Nashville TN", lat=36.163, lon=-86.782,
                                           weight_lb=30000, volume_cuft=600, latest_h=60)]), Params())
    ev = simulate(inst, [1])
    row("TC-E02", "Round trip > 14 h duty (Nashville)", "Feasible with 1 layover, 2 truck-days",
        f"feasible={ev.feasible}, layovers={ev.layovers}, days={ev.days}, {ev.duration_h} h",
        ev.feasible and ev.layovers == 1 and ev.days == 2)
    # TC-E03 hazmat separation
    inst = Instance(dep, pd.DataFrame([one(shipment_id="H1", cargo_class="HAZMAT"),
                                       one(shipment_id="G1", lat=39.11, lon=-84.50)]), Params())
    ev = simulate(inst, [1, 2])
    row("TC-E03", "HAZMAT + GENERAL on same trailer", "Route rejected (cargo-class mixing)",
        ev.reason or "accepted", (not ev.feasible) and ev.reason == "cargo-class mixing")
    # TC-E04 impossible time window
    inst = Instance(dep, pd.DataFrame([one(customer="Pittsburgh PA", lat=40.441, lon=-79.996,
                                           earliest_h=0, latest_h=2)]), Params())
    ev = simulate(inst, [1])
    row("TC-E04", "Window closes before truck can arrive (Pittsburgh by 02:00 Mon)",
        "Time window missed → carrier", ev.reason or "feasible",
        (not ev.feasible) and ev.reason.startswith("time window"))
    # TC-E05 stop gap rule
    inst = Instance(dep, pd.DataFrame([one(shipment_id="A", lat=39.103, lon=-84.512),
                                       one(shipment_id="B", customer="Cleveland", lat=41.499, lon=-81.694)]),
                    Params())
    ev = simulate(inst, [1, 2])
    row("TC-E05", "Consecutive stops 250+ mi apart", "Rejected (max 120 mi between stops)",
        ev.reason or "accepted", ev.reason == "max distance between stops")
    return pd.DataFrame(out)
