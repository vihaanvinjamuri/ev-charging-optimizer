"""
Single-vehicle charging optimizer (Section 4.3.6-4.3.7, Sections 4.5.1-4.5.3).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "cvxpy is required for optimization. Install project dependencies with:\n"
        "    pip install -r requirements.txt"
    ) from exc


@dataclass
class ChargingResult:
    power: np.ndarray
    soc: np.ndarray
    cost: float
    degradation: float
    status: str
    solve_time: Optional[float] = None
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def completion_hours(self) -> float:
        return float(self.meta.get("completion_hours", np.nan))


def solve_single_vehicle(
    price_vector, dt_hours: float, capacity_kwh: float, efficiency: float,
    soc_init: float, soc_target: float, p_max: float,
    degradation_weight: float = 0.0, solver: Optional[str] = None, label: str = "",
) -> ChargingResult:
    price_vector = np.asarray(price_vector, dtype=float)
    n = len(price_vector)
    if n == 0:
        raise ValueError("price_vector must contain at least one interval.")

    P = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)

    constraints = [soc[0] == soc_init]
    for t in range(n):
        constraints.append(soc[t + 1] == soc[t] + (P[t] * efficiency * dt_hours / capacity_kwh) * 100.0)
    constraints += [P <= p_max, soc >= 0, soc <= 100, soc[n] >= soc_target]

    cost_term = cp.sum(cp.multiply(price_vector, P)) * dt_hours
    degradation_term = cp.sum_squares(P)
    objective = cp.Minimize(cost_term + degradation_weight * degradation_term)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    if P.value is None:
        return ChargingResult(
            power=np.full(n, np.nan), soc=np.full(n + 1, np.nan),
            cost=float("nan"), degradation=float("nan"),
            status=problem.status, solve_time=None, label=label,
        )

    power = np.clip(P.value, 0, None)
    soc_values = np.clip(soc.value, 0, 100)
    completion_idx = np.argmax(soc_values >= soc_target) if np.any(soc_values >= soc_target) else n
    completion_hours = completion_idx * dt_hours
    solve_time = problem.solver_stats.solve_time if problem.solver_stats is not None else None

    return ChargingResult(
        power=power, soc=soc_values,
        cost=float(np.sum(price_vector * power) * dt_hours),
        degradation=float(np.sum(power ** 2)),
        status=problem.status, solve_time=solve_time, label=label,
        meta={"completion_hours": completion_hours, "degradation_weight": degradation_weight},
    )


def solve_immediate_charging(
    price_vector, dt_hours: float, capacity_kwh: float, efficiency: float,
    soc_init: float, soc_target: float, p_max: float, label: str = "Immediate Charging",
) -> ChargingResult:
    price_vector = np.asarray(price_vector, dtype=float)
    n = len(price_vector)
    power = np.zeros(n)
    soc = np.zeros(n + 1)
    soc[0] = soc_init
    gain_per_kw = efficiency * dt_hours / capacity_kwh * 100.0
    completion_hours = n * dt_hours

    for t in range(n):
        remaining_pct = soc_target - soc[t]
        if remaining_pct <= 0:
            power[t] = 0.0
        else:
            full_gain = p_max * gain_per_kw
            power[t] = p_max if full_gain <= remaining_pct else min(p_max, remaining_pct / gain_per_kw)
        soc[t + 1] = soc[t] + power[t] * gain_per_kw
        if soc[t + 1] >= soc_target and soc[t] < soc_target:
            completion_hours = (t + 1) * dt_hours

    soc = np.clip(soc, 0, 100)
    return ChargingResult(
        power=power, soc=soc,
        cost=float(np.sum(price_vector * power) * dt_hours),
        degradation=float(np.sum(power ** 2)),
        status="simulated (non-optimized baseline)", solve_time=0.0, label=label,
        meta={"completion_hours": completion_hours, "degradation_weight": None},
    )
