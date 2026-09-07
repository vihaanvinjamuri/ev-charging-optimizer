"""
fairness_constraint.py
------------------------
Reproduces the proportional-fairness result of Section 5.4.1: a two-stage
solve where every vehicle receives the same fraction of its originally
requested energy, rather than the cost-minimizing allocation of Section
5.4 that concentrates the entire shortfall on one vehicle.

    python fairness_constraint.py

Expected output: common fraction ~92.7%, final SoCs ~84.9% / 81.0% / 84.5%.
"""
import cvxpy as cp
import numpy as np
from evcharge import pricing

DT_HOURS = 0.25
EFFICIENCY = 0.95
P_MAX = 7.0
ARRIVAL_HOUR = 18.0
N_INTERVALS = 40
GRID_CAP_KW = 12.0

price_vector = pricing.build_price_vector(ARRIVAL_HOUR, N_INTERVALS, DT_HOURS)

vehicles = [
    {"name": "Vehicle 1", "cap": 60, "init": 20, "target": 90},
    {"name": "Vehicle 2", "cap": 45, "init": 30, "target": 85},
    {"name": "Vehicle 3", "cap": 75, "init": 15, "target": 90},
]
M = len(vehicles)
LAM = 0.2


def k_of(v):
    return EFFICIENCY * DT_HOURS / v["cap"] * 100.0


def main():
    print("Stage 1: find the maximum common fraction f achievable given the grid constraint")
    f = cp.Variable(nonneg=True)
    P = [cp.Variable(N_INTERVALS, nonneg=True) for _ in range(M)]
    cons = [P[i] <= P_MAX for i in range(M)]
    cons += [sum(P) <= GRID_CAP_KW]
    for i, v in enumerate(vehicles):
        k = k_of(v)
        fs = v["init"] + k * cp.sum(P[i])
        needed = v["target"] - v["init"]
        cons += [fs <= 100]
        cons += [fs == v["init"] + f * needed]
    cons += [f <= 1]

    prob1 = cp.Problem(cp.Maximize(f), cons)
    prob1.solve()
    f_max = f.value
    print(f"  Maximum common fraction f = {f_max:.4f}  ({f_max*100:.1f}%)")

    print("\nStage 2: minimize cost + degradation subject to every vehicle receiving exactly f_max")
    P2 = [cp.Variable(N_INTERVALS, nonneg=True) for _ in range(M)]
    cons2 = [P2[i] <= P_MAX for i in range(M)]
    cons2 += [sum(P2) <= GRID_CAP_KW]
    for i, v in enumerate(vehicles):
        k = k_of(v)
        fs = v["init"] + k * cp.sum(P2[i])
        needed = v["target"] - v["init"]
        cons2 += [fs <= 100]
        cons2 += [fs >= v["init"] + f_max * needed - 1e-4]

    obj2 = sum(cp.sum(cp.multiply(price_vector, P2[i])) * DT_HOURS + LAM * cp.sum_squares(P2[i]) for i in range(M))
    prob2 = cp.Problem(cp.Minimize(obj2), cons2)
    prob2.solve()

    print(f"\n{'Vehicle':<12s} {'Target%':>8s} {'Final%':>8s} {'Delivered/Needed':>17s} {'Cost(Rs)':>10s} {'Degr':>10s}")
    for i, v in enumerate(vehicles):
        k = k_of(v)
        p_val = np.clip(P2[i].value, 0, None)
        fs = v["init"] + k * np.sum(p_val)
        delivered_frac = (fs - v["init"]) / (v["target"] - v["init"])
        cost = float(np.sum(price_vector * p_val) * DT_HOURS)
        degr = float(np.sum(p_val ** 2))
        print(f"{v['name']:<12s} {v['target']:8.1f} {fs:8.1f} {delivered_frac*100:16.1f}% {cost:10.2f} {degr:10.1f}")


if __name__ == "__main__":
    main()
