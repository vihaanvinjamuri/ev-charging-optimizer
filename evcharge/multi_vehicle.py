"""
Coordinated multi-vehicle charging under a shared grid capacity constraint
(Section 4.3.7.6, Section 4.5.4).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, TypedDict
import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "cvxpy is required for optimization. Install project dependencies with:\n"
        "    pip install -r requirements.txt"
    ) from exc

from .optimizer import ChargingResult


class VehicleSpec(TypedDict, total=False):
    name: str
    capacity_kwh: float
    efficiency: float
    soc_init: float
    soc_target: float
    p_max: float
    degradation_weight: float


@dataclass
class MultiVehicleResult:
    vehicles: List[ChargingResult]
    total_power: np.ndarray
    total_cost: float
    total_degradation: float
    status: str
    grid_cap_kw: float
    meta: dict = field(default_factory=dict)


def solve_multi_vehicle(
    vehicles: List[VehicleSpec], price_vector, dt_hours: float, grid_cap_kw: float,
    solver: Optional[str] = None,
) -> MultiVehicleResult:
    price_vector = np.asarray(price_vector, dtype=float)
    n = len(price_vector)
    m = len(vehicles)
    if m == 0:
        raise ValueError("At least one vehicle is required.")

    P = [cp.Variable(n, nonneg=True) for _ in range(m)]
    soc = [cp.Variable(n + 1) for _ in range(m)]
    shortfall = [cp.Variable(nonneg=True) for _ in range(m)]  # soft final-SoC slack
    constraints = []
    objective_terms = []
    SHORTFALL_PENALTY = 1e4  # large weight: only accept shortfall when truly infeasible to avoid it

    for i, v in enumerate(vehicles):
        eta = v["efficiency"]
        cap = v["capacity_kwh"]
        lam = v.get("degradation_weight", 0.0)
        constraints.append(soc[i][0] == v["soc_init"])
        for t in range(n):
            constraints.append(soc[i][t + 1] == soc[i][t] + (P[i][t] * eta * dt_hours / cap) * 100.0)
        constraints += [P[i] <= v["p_max"], soc[i] >= 0, soc[i] <= 100]
        # Soft final-SoC target (Section 4.3.7.3): under an oversubscribed grid,
        # a hard constraint here would make the whole problem infeasible by
        # construction, so shortfall is allowed but heavily penalized instead.
        constraints.append(soc[i][n] >= v["soc_target"] - shortfall[i])
        cost_term = cp.sum(cp.multiply(price_vector, P[i])) * dt_hours
        deg_term = lam * cp.sum_squares(P[i])
        objective_terms.append(cost_term + deg_term + SHORTFALL_PENALTY * shortfall[i])

    total_power_expr = sum(P)
    constraints.append(total_power_expr <= grid_cap_kw)

    problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
    problem.solve(solver=solver)

    if any(p.value is None for p in P):
        results = [
            ChargingResult(power=np.full(n, np.nan), soc=np.full(n + 1, np.nan),
                            cost=float("nan"), degradation=float("nan"),
                            status=problem.status, label=vehicles[i].get("name", f"EV {i+1}"))
            for i in range(m)
        ]
        return MultiVehicleResult(vehicles=results, total_power=np.full(n, np.nan),
                                   total_cost=float("nan"), total_degradation=float("nan"),
                                   status=problem.status, grid_cap_kw=grid_cap_kw)

    results = []
    for i, v in enumerate(vehicles):
        power = np.clip(P[i].value, 0, None)
        soc_values = np.clip(soc[i].value, 0, 100)
        completion_idx = np.argmax(soc_values >= v["soc_target"]) if np.any(soc_values >= v["soc_target"]) else n
        results.append(ChargingResult(
            power=power, soc=soc_values,
            cost=float(np.sum(price_vector * power) * dt_hours),
            degradation=float(np.sum(power ** 2)),
            status=problem.status, label=v.get("name", f"EV {i + 1}"),
            meta={"completion_hours": completion_idx * dt_hours, "degradation_weight": v.get("degradation_weight", 0.0)},
        ))

    total_power = np.sum([r.power for r in results], axis=0)
    return MultiVehicleResult(
        vehicles=results, total_power=total_power,
        total_cost=float(sum(r.cost for r in results)),
        total_degradation=float(sum(r.degradation for r in results)),
        status=problem.status, grid_cap_kw=grid_cap_kw,
        meta={"peak_utilization_pct": float(np.max(total_power) / grid_cap_kw * 100.0) if grid_cap_kw > 0 else float("nan")},
    )
