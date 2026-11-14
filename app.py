"""
NMOT-Lite · Network Mode Optimization Tool (prototype decision-support system)
Based on Dang, Singh & Allen (2021) "Network Mode Optimization for the DHL Supply Chain",
INFORMS Journal on Applied Analytics 51(3):179-199. https://doi.org/10.1287/inte.2020.1046
Run:  streamlit run app.py
"""
from dataclasses import asdict
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nmot_engine import (Instance, Params, route_table, run_unit_tests, shipment_table, simulate,
                         solve_all_carrier, solve_baseline, solve_nmot, timeline_table, validate)
from scenarios import DEPOTS, SCENARIOS

st.set_page_config(page_title="NMOT-Lite | DHL Network Mode Optimization", page_icon="🚚", layout="wide")

DHL_RED, DHL_YELLOW, GREY = "#D40511", "#FFCC00", "#8A8A8A"
WEEK0 = datetime(2026, 11, 16)  # Monday 00:00 of the planning week
PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Safe


def h2dt(h):
    return WEEK0 + timedelta(hours=float(h))


def fmt_h(h):
    d = h2dt(h)
    return d.strftime("%a %H:%M")


# ------------------------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------------------------
st.markdown(
    f"<div style='background:{DHL_YELLOW};padding:14px 20px;border-radius:8px'>"
    f"<span style='font-size:28px;font-weight:800;color:{DHL_RED}'>NMOT-Lite</span>"
    f"<span style='font-size:18px;color:#222'> · Network Mode Optimization Tool — make-or-buy routing "
    f"for DHL Supply Chain</span></div>", unsafe_allow_html=True)
st.caption("Prototype DSS for the VRP with Time Regulations & Common Carriers (VRPTRCC). "
           "Paper: Dang, Singh & Allen (2021), INFORMS J. Applied Analytics 51(3) · doi:10.1287/inte.2020.1046")

# ------------------------------------------------------------------------------------------------
# Sidebar – data & parameters
# ------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Input source", ["Demo scenario", "Upload CSV"], horizontal=True)
    if source == "Demo scenario":
        scen = st.selectbox("Scenario", list(SCENARIOS))
        depot_key, loader = SCENARIOS[scen]
        depot = DEPOTS[depot_key]
        ship_df = loader()
    else:
        up = st.file_uploader("Shipments CSV", type="csv")
        dname = st.selectbox("Depot", list(DEPOTS) + ["Custom"])
        if dname == "Custom":
            depot = {"name": "Custom depot",
                     "lat": st.number_input("Depot lat", value=39.96), "lon": st.number_input("Depot lon", value=-83.0)}
        else:
            depot = DEPOTS[dname]
        ship_df = pd.read_csv(up) if up else SCENARIOS[list(SCENARIOS)[0]][1]()
        if not up:
            st.info("No file yet – showing S1 data. Download the template below.")
        scen = "Uploaded data"
    st.download_button("⬇ CSV template", SCENARIOS[list(SCENARIOS)[2]][1]().to_csv(index=False),
                       "nmot_template.csv", "text/csv")

    st.header("2 · Parameters")
    with st.expander("Dedicated-fleet costs", expanded=False):
        f = st.number_input("Fixed cost per truck-day ($)", 0.0, 2000.0, 200.0, 10.0)
        cpm = st.number_input("Cost per mile ($)", 0.0, 10.0, 2.0, 0.1)
        cps = st.number_input("Cost per stop ($)", 0.0, 500.0, 30.0, 5.0)
    with st.expander("Truck & regulations", expanded=False):
        capw = st.number_input("Capacity (lb)", 1000.0, 80000.0, 45000.0, 1000.0)
        capv = st.number_input("Capacity (cu ft)", 100.0, 6000.0, 3000.0, 100.0)
        util = st.slider("Usable capacity", 0.5, 1.0, 0.9, 0.05)
        duty = st.number_input("HOS duty window (h)", 8.0, 16.0, 14.0, 0.5)
        lay = st.number_input("Layover length (h)", 6.0, 12.0, 10.0, 0.5)
        maxlay = st.number_input("Max layovers / trip", 0, 5, 3)
        gap = st.number_input("Max miles between stops", 20.0, 500.0, 120.0, 10.0)
        maxr = st.number_input("Max route miles", 100.0, 3000.0, 1200.0, 50.0)
    with st.expander("ACO search", expanded=False):
        ants = st.slider("Ants per iteration", 2, 30, 8)
        iters = st.slider("Iterations", 5, 100, 25)
        q0 = st.slider("q0 (exploitation)", 0.0, 1.0, 0.9, 0.05)
        beta = st.slider("β (visibility)", 0.1, 5.0, 0.9, 0.1)
        rho = st.slider("ρ (evaporation)", 0.01, 0.5, 0.1, 0.01)
        seed = st.number_input("Random seed", 0, 9999, 7)
    params = Params(fixed_cost_per_day=f, cost_per_mile=cpm, cost_per_stop=cps, cap_weight_lb=capw,
                    cap_volume_cuft=capv, cap_utilisation=util, max_duty_h=duty, layover_h=lay,
                    max_layovers=int(maxlay), max_stop_gap_mi=gap, max_route_mi=maxr, n_ants=ants,
                    n_iterations=iters, q0=q0, beta=beta, rho=rho, seed=int(seed))
    run = st.button("🚀 Run NMOT optimisation", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------------------------
# Build instance
# ------------------------------------------------------------------------------------------------
try:
    inst = Instance(depot, ship_df, params)
except Exception as ex:  # data validation errors shown to the user
    st.error(f"Input data error: {ex}")
    st.stop()

run_key = (scen, depot["name"], tuple(sorted(asdict(params).items())), pd.util.hash_pandas_object(ship_df).sum())
if run or "res" not in st.session_state or st.session_state.get("key") != run_key:
    bar = st.progress(0.0, text="Ant colony searching…")
    res = {"carrier": solve_all_carrier(inst), "baseline": solve_baseline(inst),
           "nmot": solve_nmot(inst, progress=lambda x: bar.progress(x, text=f"Ant colony searching… {x:.0%}"))}
    bar.empty()
    st.session_state.res, st.session_state.key = res, run_key
res = st.session_state.res
A, B, C = res["carrier"], res["baseline"], res["nmot"]

tab0, tab1, tab2, tab3, tab4 = st.tabs(["① Problem", "② Input data", "③ Optimised plan",
                                         "④ Route drill-down", "⑤ Test cases"])

# ------------------------------------------------------------------------------------------------
# ① Problem
# ------------------------------------------------------------------------------------------------
with tab0:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Business problem")
        st.markdown("""
DHL Supply Chain North America moves **20M+ packages a year**. For each customer project, transport analysts must decide,
**for every shipment**, whether to:

* **Make** – deliver it on a DHL *dedicated* truck (multi-stop route, pay fixed + mileage + stop costs), or
* **Buy** – hand it to a *common carrier* (LTL/parcel) at its tariff-rated price **λᵢ**.

Routes must respect **customer time windows**, **truck capacity**, **max route / stop-gap distance**, cargo-class
separation and **US Hours-of-Service** (14 h duty → 10 h layover, ≤ 3 layovers per trip).

The manual Excel process took **~120 analyst-hours per project**; the paper's NMOT cut it to ~40 h and reported
**15–20 % cost savings** vs. the previous process (> $5M / yr).
""")
    with c2:
        st.subheader("How NMOT-Lite solves it")
        st.markdown("""
1. **Pre-check** each shipment alone → infeasible ones are *forced* to carrier.
2. **Ant Colony System** builds many route plans (pheromone τ × visibility η^β, q₀ exploitation).
3. **2-opt local search** tidies each route.
4. **Node Removal (NR)**: drop any stop whose *marginal* fleet cost > λᵢ; re-insert carrier
   shipments where the fleet is cheaper; outsource whole routes that cost more than their carrier total.
5. **Pheromone update** on the best plan → repeat.
""")
    st.info(f"Active scenario: **{scen}** · Depot: **{depot['name']}** · {inst.n} shipments")

# ------------------------------------------------------------------------------------------------
# ② Input data
# ------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("Shipment demand (input)")
    show = inst.df.copy()
    show["window"] = [f"{fmt_h(e)} → {fmt_h(l)}" for e, l in zip(show.earliest_h, show.latest_h)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Shipments", inst.n)
    k2.metric("Total weight", f"{show.weight_lb.sum():,.0f} lb")
    k3.metric("Avg distance from depot", f"{show.depot_miles.mean():,.0f} mi")
    k4.metric("Carrier cost if all outsourced", f"${show.carrier_cost.sum():,.0f}")
    st.dataframe(show[["shipment_id", "customer", "cargo_class", "weight_lb", "volume_cuft", "window",
                       "depot_miles", "carrier_cost"]], use_container_width=True, hide_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=show.lat, lon=show.lon, text=show.shipment_id + " · " + show.customer,
                                mode="markers", marker=dict(size=show.weight_lb ** 0.5 / 6 + 4, color=DHL_RED,
                                                            opacity=0.7), name="Shipments"))
    fig.add_trace(go.Scattergeo(lat=[depot["lat"]], lon=[depot["lon"]], text=[depot["name"]], mode="markers",
                                marker=dict(size=18, symbol="star", color=DHL_YELLOW,
                                            line=dict(width=1, color="black")), name="Depot"))
    fig.update_layout(geo=dict(scope="usa", showland=True, landcolor="#f4f4f4"), height=460,
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Marker size ∝ √weight. Carrier cost λᵢ = illustrative LTL tariff (min $175; $/cwt rising with miles) "
               "unless supplied in the CSV column `carrier_cost`.")

# ------------------------------------------------------------------------------------------------
# ③ Optimised plan
# ------------------------------------------------------------------------------------------------
with tab2:
    sav_b = (B.total - C.total) / B.total * 100 if B.total else 0
    sav_a = (A.total - C.total) / A.total * 100 if A.total else 0
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("NMOT total cost", f"${C.total:,.0f}", f"-{sav_b:.1f}% vs manual", delta_color="inverse")
    k2.metric("vs all-carrier", f"${A.total:,.0f}", f"-{sav_a:.1f}%", delta_color="inverse")
    k3.metric("Dedicated routes", len(C.routes), f"{len(C.routes) - len(B.routes):+d} vs manual",
              delta_color="off")
    k4.metric("Shipments on fleet", inst.n - len(C.outsourced))
    k5.metric("Outsourced to carrier", len(C.outsourced))

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Network map — fleet routes (colour) vs. carrier lanes (grey dotted)**")
        fig = go.Figure()
        for k, r in enumerate(C.routes):
            seq = [0] + r + [0]
            fig.add_trace(go.Scattergeo(
                lat=[depot["lat"] if i == 0 else inst.df.lat[i - 1] for i in seq],
                lon=[depot["lon"] if i == 0 else inst.df.lon[i - 1] for i in seq],
                mode="lines+markers", line=dict(width=2.5, color=PALETTE[k % len(PALETTE)]),
                marker=dict(size=7), name=f"R{k+1:02d}",
                text=["Depot" if i == 0 else f"{inst.ids[i]} {inst.df.customer[i-1]}" for i in seq]))
        for i in sorted(C.outsourced):
            fig.add_trace(go.Scattergeo(lat=[depot["lat"], inst.df.lat[i - 1]], lon=[depot["lon"], inst.df.lon[i - 1]],
                                        mode="lines", line=dict(width=1, color=GREY, dash="dot"),
                                        showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scattergeo(lat=[inst.df.lat[i - 1] for i in sorted(C.outsourced)],
                                    lon=[inst.df.lon[i - 1] for i in sorted(C.outsourced)],
                                    text=[f"{inst.ids[i]} · carrier ${inst.lam[i]:,.0f}" for i in sorted(C.outsourced)],
                                    mode="markers", marker=dict(size=9, symbol="x", color=GREY), name="Carrier"))
        fig.add_trace(go.Scattergeo(lat=[depot["lat"]], lon=[depot["lon"]], mode="markers", name="Depot",
                                    marker=dict(size=18, symbol="star", color=DHL_YELLOW,
                                                line=dict(width=1, color="black"))))
        fig.update_layout(geo=dict(scope="usa", showland=True, landcolor="#f4f4f4", fitbounds="locations"),
                          height=520, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Plan comparison (total network cost)**")
        comp = pd.DataFrame([{"plan": s.name.split(" (")[0], "Dedicated fleet": s.fleet_cost,
                              "Common carrier": s.carrier_cost} for s in (A, B, C)])
        fig = px.bar(comp, x="plan", y=["Dedicated fleet", "Common carrier"], text_auto=",.0f",
                     color_discrete_map={"Dedicated fleet": DHL_RED, "Common carrier": GREY})
        fig.update_layout(height=300, yaxis_title="$", xaxis_title="", legend_title="", margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**ACO convergence**")
        hist = pd.DataFrame(getattr(C, "history", []))
        if not hist.empty:
            fig = px.line(hist, x="iteration", y=["iteration_best", "best_so_far"],
                          color_discrete_sequence=[GREY, DHL_RED])
            fig.add_hline(y=B.total, line_dash="dash", annotation_text="manual baseline")
            fig.update_layout(height=230, yaxis_title="$", legend_title="", margin=dict(t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Route summary**")
    rt = route_table(inst, C)
    st.dataframe(rt, use_container_width=True, hide_index=True)
    st.markdown("**Shipment-level mode decisions** (export for the operations team)")
    stbl = shipment_table(inst, C)
    st.dataframe(stbl, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download plan (CSV)", stbl.to_csv(index=False), "nmot_plan.csv", "text/csv")

# ------------------------------------------------------------------------------------------------
# ④ Route drill-down
# ------------------------------------------------------------------------------------------------
with tab3:
    if not C.routes:
        st.warning("All shipments outsourced – no fleet routes to show.")
    else:
        tl = timeline_table(inst, C)
        tl["Start"], tl["Finish"] = tl.start_h.map(h2dt), tl.end_h.map(h2dt)
        st.markdown("**Driver schedule (Gantt) – drive / unload / wait / HOS layover**")
        fig = px.timeline(tl, x_start="Start", x_end="Finish", y="route", color="activity", hover_data=["at"],
                          color_discrete_map={"Drive": DHL_RED, "Unload": DHL_YELLOW, "Wait": "#9ecae1",
                                              "Layover": "#333333"})
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=max(250, 38 * len(C.routes)), margin=dict(t=10, b=0), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

        pick = st.selectbox("Inspect route", [f"R{k+1:02d}" for k in range(len(C.routes))])
        r = C.routes[int(pick[1:]) - 1]
        ev = simulate(inst, r, want_timeline=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Fleet cost", f"${ev.cost:,.0f}")
        m2.metric("Carrier alternative", f"${sum(inst.lam[i] for i in r):,.0f}")
        m3.metric("Miles", f"{ev.miles:,.0f}")
        m4.metric("Truck-days", ev.days)
        m5.metric("Layovers", ev.layovers)
        stops = []
        for pos, i in enumerate(r, 1):
            arr = next(s for a, s, e, n in ev.timeline if a == "Unload" and n == i)
            stops.append({"seq": pos, "shipment": inst.ids[i], "customer": inst.df.customer[i - 1],
                          "window": f"{fmt_h(inst.e[i])} → {fmt_h(inst.a[i])}", "service_start": fmt_h(arr),
                          "slack_h": round(inst.a[i] - arr, 1), "weight_lb": inst.w[i],
                          "carrier_rate": inst.lam[i]})
        st.dataframe(pd.DataFrame(stops), use_container_width=True, hide_index=True)
        p = params
        st.caption(f"Cost = ${p.fixed_cost_per_day:.0f} × {ev.days} day(s) + ${p.cost_per_mile} × {ev.miles:,.1f} mi "
                   f"+ ${p.cost_per_stop:.0f} × {len(r)} stops = ${ev.cost:,.2f}")

# ------------------------------------------------------------------------------------------------
# ⑤ Test cases
# ------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("A. Solution validation on the current plan")
    v = validate(inst, C)
    st.dataframe(v.style.map(lambda x: "background-color:#d4edda" if bool(x) else "background-color:#f8d7da",
                             subset=["passed"]),
                 use_container_width=True, hide_index=True)
    st.subheader("B. Functional & edge-case test suite")
    if st.button("▶ Run test suite"):
        t = run_unit_tests()
        st.session_state.tests = t
    if "tests" in st.session_state:
        t = st.session_state.tests
        if t.passed.all():
            st.success(f"{t.passed.sum()} / {len(t)} tests passed")
        else:
            st.error(f"{(~t.passed).sum()} test(s) failed")
        st.dataframe(t, use_container_width=True, hide_index=True)
