"""
stress_test.py
----------------
Tests whether the "free flexibility" region identified in Section 5.5
(a small degradation weighting that reduces battery stress with zero
cost increase) survives changes in charging window length, required
energy, and battery/charger scale. Produces the numbers reported in
Table 5.3 (Section 5.5.1).

    python stress_test.py
"""
from evcharge import pricing
from evcharge.optimizer import solve_single_vehicle


def check_free_region(label, capacity_kwh, soc_init, soc_target, p_max, arrival_hour, window_hours, dt_hours=0.25):
    n_intervals = int(round(window_hours / dt_hours))
    price_vector = pricing.build_price_vector(arrival_hour, n_intervals, dt_hours)
    weights = [0.0, 0.02, 0.05, 0.1, 0.2]
    results = []
    for lam in weights:
        r = solve_single_vehicle(price_vector, dt_hours, capacity_kwh, 0.95, soc_init, soc_target, p_max, degradation_weight=lam)
        results.append((lam, r.cost, r.degradation, r.status))
    base_cost = results[0][1]
    print(f'{label}:')
    for lam, cost, degr, status in results:
        cost_delta = cost - base_cost
        flag = '<- still free' if abs(cost_delta) < 0.01 and lam > 0 else ''
        print(f'  lambda={lam:.2f}  cost={cost:8.2f} (+{cost_delta:6.2f})  degr={degr:8.1f}  {status}  {flag}')
    print()


if __name__ == "__main__":
    check_free_region('BASELINE (10h window)', 60, 30, 90, 7, 18.0, 10.0)
    check_free_region('Near-minimum feasible window (6h)', 60, 30, 90, 7, 18.0, 6.0)
    check_free_region('Tight window (7h)', 60, 30, 90, 7, 18.0, 7.0)
    check_free_region('Long window (16h)', 60, 30, 90, 7, 18.0, 16.0)
    check_free_region('Larger energy need (10->95%)', 60, 10, 95, 7, 18.0, 10.0)
    check_free_region('Smaller energy need (60->75%)', 60, 60, 75, 7, 18.0, 10.0)
    check_free_region('Smaller battery and charger (30kWh/3.5kW)', 30, 30, 90, 3.5, 18.0, 10.0)
