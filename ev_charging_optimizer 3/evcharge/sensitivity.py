"""
Sensitivity analysis (Section 4.5.5).
"""
from __future__ import annotations
from typing import Iterable, List, Dict, Any
from .optimizer import solve_single_vehicle


def sweep_degradation_weight(price_vector, dt_hours: float, capacity_kwh: float, efficiency: float,
                              soc_init: float, soc_target: float, p_max: float,
                              weights: Iterable[float]) -> List[Dict[str, Any]]:
    rows = []
    for lam in weights:
        r = solve_single_vehicle(price_vector, dt_hours, capacity_kwh, efficiency,
                                  soc_init, soc_target, p_max, degradation_weight=float(lam))
        rows.append({"lambda": float(lam), "cost": r.cost, "degradation": r.degradation,
                      "completion_hours": r.completion_hours, "status": r.status})
    return rows


def sweep_charger_power(price_vector, dt_hours: float, capacity_kwh: float, efficiency: float,
                         soc_init: float, soc_target: float,
                         p_max_values: Iterable[float], degradation_weight: float = 0.0) -> List[Dict[str, Any]]:
    rows = []
    for pmax in p_max_values:
        r = solve_single_vehicle(price_vector, dt_hours, capacity_kwh, efficiency,
                                  soc_init, soc_target, float(pmax), degradation_weight=degradation_weight)
        rows.append({"p_max": float(pmax), "cost": r.cost, "degradation": r.degradation,
                      "completion_hours": r.completion_hours, "status": r.status})
    return rows
