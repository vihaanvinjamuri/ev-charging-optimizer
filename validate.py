"""
Validation script. Run after installing requirements:

    python validate.py

Cross-checks the CVXPY solver against results derivable independently of
CVXPY: SoC-recursion replay, direct cost/degradation recomputation, a
hand-built greedy closed-form solution for the cost-only case, and (if
SciPy is installed) an independent SciPy SLSQP solve of the multi-objective
QP.
"""
from __future__ import annotations
import numpy as np

from evcharge import battery, pricing, degradation
from evcharge.optimizer import solve_single_vehicle, solve_immediate_charging

TOL = 1e-4


def check_soc_and_cost_consistency(result, price_vector, dt_hours, capacity_kwh, efficiency, label):
    print(f"\n[{label}] SoC / cost / degradation self-consistency")
    replayed_soc = battery.soc_trajectory(result.power, result.soc[0], dt_hours, capacity_kwh, efficiency)
    soc_err = np.max(np.abs(replayed_soc - result.soc))
    print(f"  max |replayed_soc - solver_soc| = {soc_err:.3e}")
    assert soc_err < 1e-3, "SoC recursion mismatch!"

    manual_cost = pricing.charging_cost(price_vector, result.power, dt_hours)
    cost_err = abs(manual_cost - result.cost)
    print(f"  |manual_cost - solver_cost|     = {cost_err:.3e}  (cost={result.cost:.3f})")
    assert cost_err < TOL

    manual_degr = degradation.total_degradation(result.power)
    degr_err = abs(manual_degr - result.degradation)
    print(f"  |manual_degr - solver_degr|     = {degr_err:.3e}  (degr={result.degradation:.3f})")
    assert degr_err < TOL
    print("  PASS")


def greedy_cost_only(price_vector, dt_hours, capacity_kwh, efficiency, soc_init, soc_target, p_max):
    n = len(price_vector)
    energy_needed_kwh = battery.energy_required_kwh(soc_init, soc_target, capacity_kwh)
    order = np.argsort(price_vector)
    power = np.zeros(n)
    remaining_kwh = energy_needed_kwh
    for idx in order:
        if remaining_kwh <= 0:
            break
        max_energy_this_slot = p_max * efficiency * dt_hours
        take = min(max_energy_this_slot, remaining_kwh)
        power[idx] = take / (efficiency * dt_hours)
        remaining_kwh -= take
    cost = pricing.charging_cost(price_vector, power, dt_hours)
    return power, cost, remaining_kwh


def main():
    dt_hours = 0.25
    capacity_kwh = 60.0
    efficiency = 0.95
    soc_init = 30.0
    soc_target = 90.0
    p_max = 7.0
    arrival_hour = 22.0
    window_hours = 8.0
    n_intervals = int(round(window_hours / dt_hours))

    price_vector = pricing.build_price_vector(arrival_hour, n_intervals, dt_hours)

    print("=" * 70)
    print("1) Baseline immediate charging")
    baseline = solve_immediate_charging(price_vector, dt_hours, capacity_kwh, efficiency, soc_init, soc_target, p_max)
    check_soc_and_cost_consistency(baseline, price_vector, dt_hours, capacity_kwh, efficiency, "baseline")
    print(f"  cost=Rs.{baseline.cost:.2f}  degradation={baseline.degradation:.1f}  final SoC={baseline.soc[-1]:.1f}%")

    print("\n" + "=" * 70)
    print("2) Cost-only optimization (lambda=0) vs. hand-built greedy solution")
    cost_only = solve_single_vehicle(price_vector, dt_hours, capacity_kwh, efficiency, soc_init, soc_target,
                                      p_max, degradation_weight=0.0, label="Cost-Only")
    check_soc_and_cost_consistency(cost_only, price_vector, dt_hours, capacity_kwh, efficiency, "cost-only")

    greedy_power, greedy_cost, leftover = greedy_cost_only(price_vector, dt_hours, capacity_kwh, efficiency,
                                                             soc_init, soc_target, p_max)
    print(f"  CVXPY cost    = Rs.{cost_only.cost:.4f}")
    print(f"  Greedy cost   = Rs.{greedy_cost:.4f}  (unmet energy: {leftover:.4f} kWh)")
    cost_gap = abs(cost_only.cost - greedy_cost)
    print(f"  |gap|         = {cost_gap:.4f}")
    assert cost_gap < 0.5, "CVXPY cost-only solution should match the greedy optimum closely!"
    print("  PASS (within numerical/tariff-quantization tolerance)")

    print("\n" + "=" * 70)
    print("3) Multi-objective optimization (lambda=0.2)")
    multi = solve_single_vehicle(price_vector, dt_hours, capacity_kwh, efficiency, soc_init, soc_target,
                                  p_max, degradation_weight=0.2, label="Multi-Objective")
    check_soc_and_cost_consistency(multi, price_vector, dt_hours, capacity_kwh, efficiency, "multi-objective")
    print(f"  cost=Rs.{multi.cost:.2f}  degradation={multi.degradation:.1f}  (baseline degradation was {baseline.degradation:.1f})")
    assert multi.degradation <= cost_only.degradation + TOL
    assert multi.cost >= cost_only.cost - TOL
    print("  PASS: degradation decreased and cost increased relative to cost-only, as expected")

    try:
        from scipy.optimize import minimize
        k = efficiency * dt_hours / capacity_kwh * 100.0

        def objective(P):
            return np.sum(price_vector * P) * dt_hours + 0.2 * np.sum(P ** 2)

        cons = [{"type": "ineq", "fun": lambda P: soc_init + k * np.sum(P) - soc_target}]
        bounds = [(0, p_max)] * n_intervals
        x0 = np.full(n_intervals, (soc_target - soc_init) / (k * n_intervals))
        res = minimize(objective, x0, bounds=bounds, constraints=cons, method="SLSQP",
                        options={"maxiter": 500, "ftol": 1e-10})
        scipy_obj = objective(np.clip(res.x, 0, None))
        cvxpy_obj = multi.cost + 0.2 * multi.degradation
        print("\n" + "=" * 70)
        print("4) Independent SciPy SLSQP cross-check of the multi-objective QP")
        print(f"  CVXPY objective = {cvxpy_obj:.4f}")
        print(f"  SciPy objective = {scipy_obj:.4f}")
        gap = abs(cvxpy_obj - scipy_obj)
        print(f"  |gap|           = {gap:.4f}")
        assert gap < 1.0
        print("  PASS")
    except ImportError:
        print("\n(scipy not installed -- skipping optional independent cross-check)")

    print("\n" + "=" * 70)
    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
