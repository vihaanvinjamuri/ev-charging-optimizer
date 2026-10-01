"""
reproduce_results.py
---------------------
Regenerates every headline number reported in the paper from a single,
clean run of the tested evcharge codebase. Run this end-to-end with no
arguments; every table below should match the corresponding table in the
paper exactly.

    python reproduce_results.py

If any number here doesn't match the paper, that's a real problem worth
flagging, not something to quietly fix in the paper instead.
"""
from evcharge import pricing
from evcharge.optimizer import solve_immediate_charging, solve_single_vehicle
from evcharge.multi_vehicle import solve_multi_vehicle
from evcharge.sensitivity import sweep_degradation_weight

DT_HOURS = 0.25
CAPACITY_KWH = 60.0
EFFICIENCY = 0.95
SOC_INIT = 30.0
SOC_TARGET = 90.0
P_MAX = 7.0
ARRIVAL_HOUR = 18.0
N_INTERVALS = 40

ILLUSTRATIVE_SCHEDULE = None

TGERC_SCHEDULE = [
    (0.0, 6.0, 6.00, "Overnight (base)"),
    (6.0, 10.0, 7.50, "Peak"),
    (10.0, 18.0, 5.50, "Off-Peak Rebate"),
    (18.0, 22.0, 7.50, "Peak"),
    (22.0, 24.0, 6.00, "Overnight (base)"),
]


def section_5_1_to_5_3():
    print("=" * 78)
    print("SECTIONS 5.1-5.3: Baseline / Cost-Only / Multi-Objective (illustrative tariff)")
    print("=" * 78)
    price_vector = pricing.build_price_vector(ARRIVAL_HOUR, N_INTERVALS, DT_HOURS, ILLUSTRATIVE_SCHEDULE)

    baseline = solve_immediate_charging(price_vector, DT_HOURS, CAPACITY_KWH, EFFICIENCY,
                                         SOC_INIT, SOC_TARGET, P_MAX)
    cost_only = solve_single_vehicle(price_vector, DT_HOURS, CAPACITY_KWH, EFFICIENCY,
                                      SOC_INIT, SOC_TARGET, P_MAX, degradation_weight=0.0)
    multi_obj = solve_single_vehicle(price_vector, DT_HOURS, CAPACITY_KWH, EFFICIENCY,
                                      SOC_INIT, SOC_TARGET, P_MAX, degradation_weight=0.2)

    print(f"{'Strategy':<20s} {'Cost (Rs)':>12s} {'Degradation':>14s}   Paper says")
    print(f"{'Baseline':<20s} {baseline.cost:12.2f} {baseline.degradation:14.1f}   279.87 / 1050.0")
    print(f"{'Cost-only':<20s} {cost_only.cost:12.2f} {cost_only.degradation:14.1f}   185.37 / 935.6")
    print(f"{'Multi-obj (l=0.2)':<20s} {multi_obj.cost:12.2f} {multi_obj.degradation:14.1f}   215.85 / 617.4")
    return price_vector


def section_5_3_1(illustrative_price_vector):
    print()
    print("=" * 78)
    print("SECTION 5.3.1: Real TGERC tariff comparison")
    print("=" * 78)
    price_vector_real = pricing.build_price_vector(ARRIVAL_HOUR, N_INTERVALS, DT_HOURS, TGERC_SCHEDULE)

    baseline = solve_immediate_charging(price_vector_real, DT_HOURS, CAPACITY_KWH, EFFICIENCY,
                                         SOC_INIT, SOC_TARGET, P_MAX)
    cost_only = solve_single_vehicle(price_vector_real, DT_HOURS, CAPACITY_KWH, EFFICIENCY,
                                      SOC_INIT, SOC_TARGET, P_MAX, degradation_weight=0.0)
    multi_obj = solve_single_vehicle(price_vector_real, DT_HOURS, CAPACITY_KWH, EFFICIENCY,
                                      SOC_INIT, SOC_TARGET, P_MAX, degradation_weight=0.2)

    print(f"{'Strategy':<20s} {'Cost (Rs)':>12s} {'Degradation':>14s}   Paper says")
    print(f"{'Baseline':<20s} {baseline.cost:12.2f} {baseline.degradation:14.1f}   269.37 / 1050.0")
    print(f"{'Cost-only':<20s} {cost_only.cost:12.2f} {cost_only.degradation:14.1f}   227.38 / 972.7")
    print(f"{'Multi-obj (l=0.2)':<20s} {multi_obj.cost:12.2f} {multi_obj.degradation:14.1f}   246.73 / 582.8")


def section_5_4(illustrative_price_vector):
    print()
    print("=" * 78)
    print("SECTION 5.4: Coordinated multi-vehicle charging (cost-minimizing, shortfall unconstrained)")
    print("=" * 78)
    vehicles = [
        {"name": "Vehicle 1", "capacity_kwh": 60, "efficiency": 0.95, "soc_init": 20, "soc_target": 90,
         "p_max": 7, "degradation_weight": 0.2},
        {"name": "Vehicle 2", "capacity_kwh": 45, "efficiency": 0.95, "soc_init": 30, "soc_target": 85,
         "p_max": 7, "degradation_weight": 0.2},
        {"name": "Vehicle 3", "capacity_kwh": 75, "efficiency": 0.95, "soc_init": 15, "soc_target": 90,
         "p_max": 7, "degradation_weight": 0.2},
    ]
    result = solve_multi_vehicle(vehicles, illustrative_price_vector, DT_HOURS, grid_cap_kw=12.0)
    print(f"Status: {result.status}")
    expected = ["271.65 / 781.8 / 90.0%", "160.61 / 271.5 / 85.0%", "305.74 / 989.5 / 78.0%"]
    for r, exp in zip(result.vehicles, expected):
        print(f"{r.label:<12s} cost={r.cost:8.2f}  degr={r.degradation:8.1f}  "
              f"final_soc={r.soc[-1]:5.1f}%   Paper says {exp}")


def section_5_4_1(illustrative_price_vector):
    print()
    print("=" * 78)
    print("SECTION 5.4.1: Proportional-fairness alternative")
    print("=" * 78)
    print("(Two-stage max-min fairness solve -- see paper Section 5.4.1 methodology)")
    print("Expected: common fraction 92.7%, final SoCs 84.9% / 81.0% / 84.5%")
    print("(Not re-derived in this script; see fairness_constraint.py for the standalone solve)")


def section_5_5():
    print()
    print("=" * 78)
    print("SECTION 5.5: Sensitivity sweep (Table 5.2)")
    print("=" * 78)
    price_vector = pricing.build_price_vector(ARRIVAL_HOUR, N_INTERVALS, DT_HOURS, ILLUSTRATIVE_SCHEDULE)
    weights = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    rows = sweep_degradation_weight(price_vector, DT_HOURS, CAPACITY_KWH, EFFICIENCY,
                                     SOC_INIT, SOC_TARGET, P_MAX, weights)
    print(f"{'lambda':>8s} {'cost':>10s} {'degradation':>14s}")
    for row in rows:
        print(f"{row['lambda']:8.3g} {row['cost']:10.2f} {row['degradation']:14.1f}")


if __name__ == "__main__":
    pv = section_5_1_to_5_3()
    section_5_3_1(pv)
    section_5_4(pv)
    section_5_4_1(pv)
    section_5_5()
    print()
    print("=" * 78)
    print("Done. Compare every number above against the paper's Chapter 5 tables.")
    print("=" * 78)
