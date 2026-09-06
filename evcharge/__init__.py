"""
evcharge
========
Computational framework for multi-objective EV charging optimization
(electricity cost vs. battery degradation, under grid constraints).

    battery.py        -> Section 4.3.3 (SoC model, Eq. 4.1-4.3)
    pricing.py         -> Section 4.3.4 (TOU electricity pricing, Eq. 4.4-4.5)
    degradation.py     -> Section 4.3.5 (degradation proxy)
    optimizer.py       -> Section 4.3.6-4.3.7 (single-vehicle multi-objective QP)
    multi_vehicle.py   -> Section 4.3.7.6 (shared grid capacity, multi-EV)
    sensitivity.py     -> Section 4.5.5 (parameter sweeps)
    scenarios.py       -> runs the 3 single-vehicle strategies together
"""
from . import battery, pricing, degradation

try:
    from . import optimizer, multi_vehicle, sensitivity, scenarios
except ImportError:  # pragma: no cover - cvxpy not yet installed
    optimizer = None       # type: ignore
    multi_vehicle = None   # type: ignore
    sensitivity = None     # type: ignore
    scenarios = None       # type: ignore

__all__ = ["battery", "pricing", "degradation", "optimizer", "multi_vehicle", "sensitivity", "scenarios"]
__version__ = "1.0.1"
