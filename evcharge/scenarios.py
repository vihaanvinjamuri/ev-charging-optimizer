"""
Runs the three single-vehicle strategies compared in Chapter 5:
baseline immediate charging, cost-only, and multi-objective.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from .optimizer import ChargingResult, solve_immediate_charging, solve_single_vehicle


def run_all_strategies(price_vector, dt_hours: float, capacity_kwh: float, efficiency: float,
                        soc_init: float, soc_target: float, p_max: float,
                        degradation_weight: float) -> Dict[str, ChargingResult]:
    baseline = solve_immediate_charging(price_vector, dt_hours, capacity_kwh, efficiency,
                                         soc_init, soc_target, p_max, label="Immediate Charging")
    cost_only = solve_single_vehicle(price_vector, dt_hours, capacity_kwh, efficiency,
                                      soc_init, soc_target, p_max, degradation_weight=0.0,
                                      label="Cost-Only Optimization")
    multi_objective = solve_single_vehicle(price_vector, dt_hours, capacity_kwh, efficiency,
                                            soc_init, soc_target, p_max, degradation_weight=degradation_weight,
                                            label=f"Multi-Objective (λ={degradation_weight:g})")
    return {"baseline": baseline, "cost_only": cost_only, "multi_objective": multi_objective}


def summary_table(results: Dict[str, ChargingResult]):
    """Build a list-of-dicts summary suitable for a pandas DataFrame / st.table.

    Numeric columns stay numeric (NaN when infeasible) so downstream
    rendering (e.g. Streamlit's Arrow-based dataframe display) doesn't choke
    on a column mixing floats and the string "infeasible".
    """
    rows = []
    for key, r in results.items():
        feasible = np.isfinite(r.cost)
        rows.append({
            "Strategy": r.label or key,
            "Cost (₹)": round(r.cost, 2) if feasible else np.nan,
            "Degradation Index": round(r.degradation, 1) if feasible else np.nan,
            "Final SoC (%)": round(float(r.soc[-1]), 1) if np.all(np.isfinite(r.soc)) else np.nan,
            "Completion Time (h)": round(r.completion_hours, 2) if np.isfinite(r.completion_hours) else np.nan,
            "Status": r.status,
        })
    return rows
