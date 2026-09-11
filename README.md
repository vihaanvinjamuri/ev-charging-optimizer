# EV Charging Optimization — Computational Framework

**Try it live:** https://vihaan-ev-charging-optimizer.streamlit.app — no install needed, every parameter is adjustable in your browser.

Companion codebase for the EV charging optimization paper. Every module
implements one part of the methodology (Chapter 4) against the paper's own
equation numbers. `app.py` is an interactive Streamlit dashboard — this is
the file that opens as a website in your browser.

**This build was tested end-to-end** (not just eyeballed): the optimizer was
run live with real CVXPY solves, the Streamlit app was launched and driven
programmatically through every tab and an infeasible-input edge case, and
`validate.py` was run for real, including an independent SciPy cross-check
that matched CVXPY's result to 4 decimal places.

## 1. Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the interactive web app

```bash
streamlit run app.py
```

This opens a local website (usually `http://localhost:8501`) in your
browser. **Important:** run it with `streamlit run app.py`, not
`python app.py` — the latter will not open a website; it just triggers a
"missing ScriptRunContext" warning and does nothing useful.

Everything in the sidebar is editable — battery capacity, initial/target
SoC, efficiency, charger power, arrival time, charging window, interval
length, degradation weight λ, and the TOU tariff table — and three tabs
update live:

- **Single Vehicle** — Immediate Charging (baseline) vs. Cost-Only vs.
  Multi-Objective, side by side.
- **Multi-Vehicle (Shared Grid)** — 1–8 vehicles sharing one grid capacity
  limit.
- **Sensitivity Analysis** — sweep λ or `P_max` and see the cost/degradation
  trade-off curve.

## 3. Run the validation script

```bash
python validate.py
```

Cross-checks the CVXPY solver against independently-derived results and
prints `ALL VALIDATION CHECKS PASSED` on success.

## 4. Reproduce every headline number in the paper

```bash
python reproduce_results.py
```

Regenerates the single-vehicle, real-tariff, multi-vehicle, and sensitivity
results from Chapter 5 in one run, printing each alongside the value
reported in the paper for direct comparison.

```bash
python fairness_constraint.py
```

Reproduces the Section 5.4.1 proportional-fairness result specifically
(the two-stage max-min solve).

## 5. Use the modules directly (no UI)

```python
from evcharge import pricing
from evcharge.optimizer import solve_single_vehicle

price_vector = pricing.build_price_vector(arrival_hour=22.0, n_intervals=32, dt_hours=0.25)
result = solve_single_vehicle(
    price_vector, dt_hours=0.25, capacity_kwh=60.0, efficiency=0.95,
    soc_init=30.0, soc_target=90.0, p_max=7.0, degradation_weight=0.2,
)
print(result.cost, result.degradation, result.status)
```

## 6. Folder structure

```
ev_charging_optimizer/
├── app.py                   # Streamlit interactive dashboard (run this)
├── validate.py               # Cross-checks of the solver
├── reproduce_results.py      # Regenerates every headline number in the paper
├── fairness_constraint.py    # Reproduces the Section 5.4.1 fairness result
├── stress_test.py             # Reproduces the Section 5.5.1 robustness results
├── requirements.txt
├── README.md
└── evcharge/
    ├── __init__.py
    ├── battery.py          # SoC model                 -> Sec. 4.3.3, Eq. 4.1-4.3
    ├── pricing.py          # TOU electricity pricing    -> Sec. 4.3.4, Eq. 4.4-4.5
    ├── degradation.py      # Degradation proxy          -> Sec. 4.3.5
    ├── optimizer.py        # Single-vehicle CVXPY QP +
    │                       #   immediate-charging baseline -> Sec. 4.3.6-4.3.7, 4.5.1-4.5.3
    ├── multi_vehicle.py    # Shared-grid multi-EV QP    -> Sec. 4.3.7.6, 4.5.4
    ├── sensitivity.py      # Parameter sweeps           -> Sec. 4.5.5
    └── scenarios.py        # Runs all 3 single-vehicle strategies + summary table
```


## 7. If something still doesn't open a website

1. Confirm you typed `streamlit run app.py`, not `python app.py`.
2. Confirm `streamlit` is on your PATH: `streamlit --version`. If that
   fails, try `python -m streamlit run app.py` instead.
3. It should print a `Local URL: http://localhost:8501` line — open that
   link if it doesn't auto-open.
