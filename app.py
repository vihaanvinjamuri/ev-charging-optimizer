"""
Interactive web app for the EV charging optimization framework.

Run with:
    streamlit run app.py

Opens a local website (default http://localhost:8501) where every input
from the paper's methodology can be changed live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evcharge import battery, pricing
from evcharge.scenarios import run_all_strategies, summary_table
from evcharge.multi_vehicle import solve_multi_vehicle
from evcharge.sensitivity import sweep_degradation_weight, sweep_charger_power

st.set_page_config(page_title="EV Charging Optimization", layout="wide")

TOU_COLORS = {"Off-Peak": "#d9f2e6", "Shoulder": "#fff3cf", "Peak": "#fbdada"}

st.sidebar.title("⚡ Model Parameters")

st.sidebar.subheader("Battery")
capacity_kwh = st.sidebar.number_input("Battery capacity (kWh)", 5.0, 200.0, 60.0, 1.0)
soc_init = st.sidebar.slider("Initial SoC (%)", 0, 100, 30)
soc_target = st.sidebar.slider("Target SoC (%)", 0, 100, 90)
efficiency = st.sidebar.slider("Charging efficiency \u03b7", 0.70, 1.00, 0.95, 0.01)

st.sidebar.subheader("Charger & Schedule")
p_max = st.sidebar.number_input("Max charger power P_max (kW)", 1.0, 350.0, 7.0, 0.5)
arrival_hour = st.sidebar.slider("Arrival time (24h clock)", 0.0, 23.75, 18.0, 0.25)
window_hours = st.sidebar.slider("Charging window length (h)", 0.5, 24.0, 10.0, 0.25)
dt_minutes = st.sidebar.selectbox("Interval \u0394t (minutes)", [5, 10, 15, 30, 60], index=2)
dt_hours = dt_minutes / 60.0
n_intervals = max(1, int(round(window_hours / dt_hours)))

st.sidebar.subheader("Multi-Objective Weighting")
degradation_weight = st.sidebar.slider("Degradation weight \u03bb", 0.0, 2.0, 0.20, 0.01)

st.sidebar.subheader("Time-of-Use Tariff (\u20b9/kWh)")
tariff_df_default = pd.DataFrame(
    [{"Period": lbl, "Start (h)": s, "End (h)": e, "Price (\u20b9/kWh)": p}
     for s, e, p, lbl in pricing.DEFAULT_TOU_SCHEDULE]
)
tariff_df = st.sidebar.data_editor(tariff_df_default, hide_index=True, num_rows="fixed", key="tariff_editor")
schedule = [(row["Start (h)"], row["End (h)"], row["Price (\u20b9/kWh)"], row["Period"])
            for _, row in tariff_df.iterrows()]
try:
    pricing.validate_schedule(schedule)
except ValueError as e:
    st.sidebar.error(f"Tariff schedule invalid: {e}")
    schedule = pricing.DEFAULT_TOU_SCHEDULE

price_vector = pricing.build_price_vector(arrival_hour, n_intervals, dt_hours, schedule)
time_axis = arrival_hour + np.arange(n_intervals) * dt_hours
time_axis_soc = arrival_hour + np.arange(n_intervals + 1) * dt_hours

min_intervals = battery.min_intervals_required(soc_init, soc_target, capacity_kwh, efficiency, p_max, dt_hours)
if min_intervals > n_intervals:
    st.sidebar.warning(
        f"\u26a0\ufe0f Reaching {soc_target}% SoC from {soc_init}% needs at least "
        f"{min_intervals * dt_hours:.2f} h of charging at P_max, but the window is only "
        f"{window_hours:.2f} h. Increase the window or P_max."
    )


def tou_background_shapes():
    shapes = []
    for t in time_axis:
        lbl = pricing.label_at_hour(t, schedule)
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=t, x1=t + dt_hours, y0=0, y1=1,
                            fillcolor=TOU_COLORS.get(lbl, "#eeeeee"), opacity=0.5, line_width=0, layer="below"))
    return shapes


st.title("\U0001F50B EV Charging Optimization \u2014 Interactive Dashboard")
st.caption("Companion tool for the EV charging optimization paper. Adjust parameters in the sidebar; every chart updates live.")

tab_about, tab_single, tab_multi, tab_sens = st.tabs(
    ["About the Model", "Single Vehicle", "Multi-Vehicle (Shared Grid)", "Sensitivity Analysis"]
)

with tab_about:
    st.subheader("What This Model Actually Does")
    st.markdown(
        """
Charging an EV isn't just "plug in and wait." Every 15 minutes, the optimizer has to
decide **how much power to draw right now**, and it makes that decision by weighing two
things against each other: how much electricity costs at that moment, and how much
stress that charging rate puts on the battery. This dashboard lets you change every
input that decision depends on and watch the result update live.
"""
    )

    st.markdown("### What is State of Charge (SoC)?")
    st.markdown(
        """
SoC is just how full the battery is, expressed as a percentage: 0% is empty, 100% is
full. If a 60 kWh battery is at 30% SoC, it's holding 18 kWh. As the charger delivers
power, SoC climbs. Not all of that power actually makes it into the battery, though —
some is lost as heat in the charging electronics, which is what the **efficiency**
slider (η) in the sidebar controls. At η = 0.95, only 95% of what the charger draws
from the wall actually ends up stored.

**The math** (Eq. 4.3): each 15-minute step, the battery gains
$SOC_{t+1} = SOC_t + \\dfrac{\\eta P_t \\Delta t}{C_{bat}} \\times 100$ percent, where
$P_t$ is the charging power you (or the optimizer) chose for that step and $C_{bat}$
is the battery's total capacity.
"""
    )

    st.markdown("### How is electricity cost calculated?")
    st.markdown(
        """
Simple: whatever power was drawn during an interval, multiplied by that interval's
price, multiplied by how long the interval lasted. Add that up across the whole
session and you get total cost. Because the sidebar's tariff table lets prices vary by
time of day, *when* you charge (not just how much) directly changes the bill — this is
the whole reason shifting charging to off-peak hours saves money.

**The math** (Eq. 4.5): $C_{total} = \\sum_t c_t P_t \\Delta t$
"""
    )

    st.markdown("### What is the degradation index, and where does it come from?")
    st.markdown(
        """
This is the part most worth reading carefully, because it's the model's biggest
simplification, and it's important to know what it does and doesn't claim.

Real battery aging is governed by messy electrochemistry — heat, high currents, and
time spent near full charge all contribute, and modeling it precisely would need
laboratory data specific to the exact battery chemistry involved, which isn't
realistically available for a project like this. Instead, this model uses a
**stress proxy**: it assumes that pushing more power through the battery at any given
moment increases stress, and it penalizes that stress *quadratically* rather than
linearly, meaning doubling the charging power more than doubles the penalty.

**Why square it specifically?** Two reasons. First, it matches the general pattern in
battery research that aggressive, high-current charging is disproportionately harder
on a battery than gentle charging — a mild penalty wouldn't discourage spiky charging
schedules enough. Second, and just as important for making the optimization work at
all: a squared term keeps the whole problem *convex*, which is what lets the solver
guarantee it found the actual best schedule rather than just a decent one.

**What this number does *not* mean:** it is not a prediction of how much capacity the
battery will actually lose, and it can't be converted into "this schedule adds 3 months
to battery life." It's a *relative* score — a lower number means a smoother, gentler
charging profile than a higher number, for the same total energy delivered. Comparing
two strategies' index values tells you which one is gentler; it doesn't tell you by how
much in real-world terms.

**The math** (Section 4.3.5): $D_{total} = \\sum_t P_t^2$
"""
    )

    st.markdown("### How do cost and degradation get combined?")
    st.markdown(
        """
The optimizer doesn't pick one or the other — it minimizes a weighted combination of
both, where you control the weighting with **λ (lambda)** in the sidebar. At λ = 0, it
only cares about cost (the "Cost-Only" strategy). As λ increases, it starts trading
some cost for a smoother, lower-stress profile. There's no single "correct" λ — it
depends on how much you personally value saving money versus being gentle on the
battery. The Sensitivity Analysis tab lets you see exactly how that trade-off plays out
as λ changes.

**The math**: $\\min_P J = \\sum_t c_t P_t \\Delta t + \\lambda \\sum_t P_t^2$
"""
    )

    st.markdown("### How does the multi-vehicle model work?")
    st.markdown(
        """
When several vehicles share one power connection (think: an apartment garage or office
lot), they're not just optimizing independently — they're competing for the same
limited pool of power at every moment. The model adds one more rule on top of
everything above: at any given time step, the *total* power drawn by every vehicle
combined can't exceed the shared connection's capacity.

If there's enough power for everyone to fully charge, the model just solves each
vehicle's schedule as usual. But if total demand exceeds what the connection can
supply (an "oversubscribed" grid), someone has to get less than they asked for. Left to
its own devices, the cost-minimizing solver tends to fully satisfy some vehicles and
leave one badly short, rather than spreading the shortfall around evenly — that's a
genuine finding from this project, not a bug, and it's why a fairness-aware version of
the model exists as an extension.

**The math** (Section 4.3.7.6): $\\sum_{i=1}^{M} P_{i,t} \\le G_{max}$ for every vehicle
$i$ at every time step $t$.
"""
    )

    st.info("If a strategy shows infeasible, the charging window is too short (or P_max too low) "
            "to reach the target SoC \u2014 widen the window, raise P_max, or lower the target SoC.")

with tab_single:
    st.subheader("Strategy Comparison")
    st.markdown(
        "This tab runs the same charging session three different ways and lines up the "
        "results side by side: **Immediate Charging** (plug in, go full power right away, "
        "ignore everything else), **Cost-Only** (schedule purely to minimize the electricity "
        "bill), and **Multi-Objective** (also try to keep the power profile smooth, "
        "controlled by the λ slider in the sidebar). The table below shows each strategy's "
        "final numbers; the two charts show *how* each one gets there over time. In the power "
        "chart, a flatter, lower line means a gentler charging profile — sharp spikes are what "
        "the degradation term is trying to avoid."
    )
    results = run_all_strategies(price_vector, dt_hours, capacity_kwh, efficiency,
                                  soc_init, soc_target, p_max, degradation_weight)
    rows = summary_table(results)
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    any_infeasible = any(not np.isfinite(r.cost) for r in results.values())

    if any_infeasible:
        st.error("One or more strategies were infeasible with the current settings. Try a longer window, higher P_max, or lower target SoC.")

    colors = {"baseline": "#888888", "cost_only": "#1f77b4", "multi_objective": "#d62728"}
    fig_power = go.Figure()
    for key, r in results.items():
        if np.all(np.isfinite(r.power)):
            fig_power.add_trace(go.Scatter(x=time_axis, y=r.power, mode="lines", name=r.label,
                                            line=dict(color=colors[key], shape="hv", width=2.5)))
    fig_power.update_layout(title="Charging Power Profile", xaxis_title="Time of day (h)",
                             yaxis_title="Charging power (kW)", shapes=tou_background_shapes(),
                             legend=dict(orientation="h", y=-0.2), height=420)
    st.plotly_chart(fig_power, width='stretch')

    fig_soc = go.Figure()
    for key, r in results.items():
        if np.all(np.isfinite(r.soc)):
            fig_soc.add_trace(go.Scatter(x=time_axis_soc, y=r.soc, mode="lines", name=r.label,
                                          line=dict(color=colors[key], width=2.5)))
    fig_soc.add_hline(y=soc_target, line_dash="dot", line_color="black", annotation_text="Target SoC")
    fig_soc.update_layout(title="Battery State of Charge", xaxis_title="Time of day (h)",
                           yaxis_title="SoC (%)", yaxis_range=[0, 100], shapes=tou_background_shapes(),
                           legend=dict(orientation="h", y=-0.2), height=420)
    st.plotly_chart(fig_soc, width='stretch')
    st.caption("Background shading marks off-peak (green), shoulder (yellow), and peak (red) tariff periods.")

with tab_multi:
    st.subheader("Coordinated Multi-Vehicle Charging")
    st.markdown(
        "Several vehicles here share one power connection with a fixed total capacity "
        "(**G_max**, the slider below). Set each vehicle's battery and target below, then "
        "check the total: if everyone's combined demand fits within G_max, every vehicle "
        "reaches its target. If it doesn't, the shortfall gets concentrated onto whichever "
        "vehicle the optimizer decides is cheapest to shortchange, not split evenly. The "
        "stacked chart shows every vehicle's share of the shared connection over time; if "
        "the colored area ever touches the dashed **Grid Capacity** line, the connection is "
        "running at its limit."
    )
    n_vehicles = st.number_input("Number of vehicles", 1, 8, 3, 1)
    grid_cap_kw = st.slider("Shared grid capacity G_max (kW)", 1.0, 100.0, 14.0, 0.5)

    vehicles = []
    cols = st.columns(int(n_vehicles))
    for i in range(int(n_vehicles)):
        with cols[i]:
            st.markdown(f"**EV {i + 1}**")
            cap_i = st.number_input(f"Capacity (kWh)##{i}", 5.0, 200.0, 60.0, 1.0, key=f"cap_{i}")
            init_i = st.slider(f"Initial SoC (%)##{i}", 0, 100, 20 + 5 * i, key=f"init_{i}")
            target_i = st.slider(f"Target SoC (%)##{i}", 0, 100, 90, key=f"target_{i}")
            pmax_i = st.number_input(f"P_max (kW)##{i}", 1.0, 50.0, 7.0, 0.5, key=f"pmax_{i}")
            lam_i = st.slider(f"\u03bb (degradation)##{i}", 0.0, 2.0, degradation_weight, 0.01, key=f"lam_{i}")
            vehicles.append({"name": f"EV {i + 1}", "capacity_kwh": cap_i, "efficiency": efficiency,
                              "soc_init": init_i, "soc_target": target_i, "p_max": pmax_i,
                              "degradation_weight": lam_i})

    mv_result = solve_multi_vehicle(vehicles, price_vector, dt_hours, grid_cap_kw)
    if mv_result.status not in ("optimal", "optimal_inaccurate"):
        st.error(f"Multi-vehicle problem status: {mv_result.status}. Try raising the grid cap or the charging window.")
    else:
        mv_rows = [{"Vehicle": r.label, "Cost (\u20b9)": round(r.cost, 2), "Degradation Index": round(r.degradation, 1),
                     "Final SoC (%)": round(float(r.soc[-1]), 1)} for r in mv_result.vehicles]
        mv_rows.append({"Vehicle": "TOTAL", "Cost (\u20b9)": round(mv_result.total_cost, 2),
                         "Degradation Index": round(mv_result.total_degradation, 1), "Final SoC (%)": "\u2014"})
        st.dataframe(pd.DataFrame(mv_rows), width='stretch', hide_index=True)
        st.metric("Peak grid utilization", f"{mv_result.meta['peak_utilization_pct']:.1f}%")

        fig_mv = go.Figure()
        for r in mv_result.vehicles:
            fig_mv.add_trace(go.Scatter(x=time_axis, y=r.power, name=r.label, mode="lines",
                                         line=dict(shape="hv", width=2), stackgroup="one"))
        fig_mv.add_trace(go.Scatter(x=time_axis, y=np.full(n_intervals, grid_cap_kw), name="Grid Capacity",
                                     mode="lines", line=dict(color="black", dash="dash")))
        fig_mv.update_layout(title="Stacked Charging Power vs. Shared Grid Capacity",
                              xaxis_title="Time of day (h)", yaxis_title="Charging power (kW)",
                              shapes=tou_background_shapes(), legend=dict(orientation="h", y=-0.25), height=460)
        st.plotly_chart(fig_mv, width='stretch')

with tab_sens:
    st.subheader("Sensitivity Analysis")
    st.markdown(
        "This tab sweeps one parameter across a range of values and re-solves the "
        "single-vehicle problem at every point, so you can see the trend rather than just "
        "one snapshot. Sweeping **λ** shows how much you have to pay (in cost) to buy a "
        "given amount of battery-stress reduction — usually a steep drop at first, then "
        "diminishing returns. Sweeping **P_max** shows how a stronger or weaker charger "
        "changes the achievable trade-off. The second chart re-plots the same data as cost "
        "versus degradation directly, which is often the easiest way to see the trade-off's "
        "shape at a glance."
    )
    sweep_choice = st.radio("Parameter to sweep", ["Degradation weight (\u03bb)", "Max charger power (P_max)"], horizontal=True)

    if sweep_choice.startswith("Degradation"):
        lam_min, lam_max = st.slider("\u03bb range", 0.0, 5.0, (0.0, 1.0), 0.05)
        n_points = st.slider("Number of points", 3, 30, 12)
        weights = np.linspace(lam_min, lam_max, n_points)
        rows = sweep_degradation_weight(price_vector, dt_hours, capacity_kwh, efficiency,
                                         soc_init, soc_target, p_max, weights)
        df = pd.DataFrame(rows)
        x_col = "lambda"
    else:
        p_min, p_max_sweep = st.slider("P_max range (kW)", 1.0, 50.0, (3.0, 22.0), 0.5)
        n_points = st.slider("Number of points", 3, 30, 12)
        p_values = np.linspace(p_min, p_max_sweep, n_points)
        rows = sweep_charger_power(price_vector, dt_hours, capacity_kwh, efficiency,
                                    soc_init, soc_target, p_values, degradation_weight=degradation_weight)
        df = pd.DataFrame(rows)
        x_col = "p_max"

    st.dataframe(df, width='stretch', hide_index=True)
    feasible = df[df["status"].isin(["optimal", "optimal_inaccurate"])]
    if feasible.empty:
        st.warning("No feasible points in this sweep range with the current battery/charger settings.")
    else:
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=feasible[x_col], y=feasible["cost"], name="Cost (\u20b9)", mode="lines+markers", yaxis="y1"))
        fig_sens.add_trace(go.Scatter(x=feasible[x_col], y=feasible["degradation"], name="Degradation Index", mode="lines+markers", yaxis="y2"))
        fig_sens.update_layout(title=f"Cost & Degradation vs. {x_col}", xaxis_title=x_col,
                                yaxis=dict(title="Cost (\u20b9)"), yaxis2=dict(title="Degradation Index", overlaying="y", side="right"),
                                legend=dict(orientation="h", y=-0.2), height=420)
        st.plotly_chart(fig_sens, width='stretch')

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Scatter(x=feasible["cost"], y=feasible["degradation"], mode="markers+lines",
                                         text=[f"{x_col}={v:.3g}" for v in feasible[x_col]], hoverinfo="text+x+y"))
        fig_pareto.update_layout(title="Cost vs. Degradation Trade-off", xaxis_title="Cost (\u20b9)",
                                  yaxis_title="Degradation Index", height=420)
        st.plotly_chart(fig_pareto, width='stretch')
