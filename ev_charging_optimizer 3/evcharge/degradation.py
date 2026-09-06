"""
Battery degradation proxy (Section 4.3.5): D_t = P_t^2, D_total = sum(D_t).
"""
from __future__ import annotations
import numpy as np


def degradation_contribution(power_kw) -> np.ndarray:
    return np.square(np.asarray(power_kw, dtype=float))


def total_degradation(power_profile_kw) -> float:
    return float(np.sum(degradation_contribution(power_profile_kw)))
