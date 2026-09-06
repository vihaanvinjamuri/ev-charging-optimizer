"""
Battery State-of-Charge (SoC) model (Section 4.3.3, Eq. 4.1-4.3).
"""
from __future__ import annotations
import numpy as np


def energy_to_soc(energy_kwh, capacity_kwh: float):
    return np.asarray(energy_kwh) / capacity_kwh * 100.0


def soc_to_energy(soc_pct, capacity_kwh: float):
    return np.asarray(soc_pct) / 100.0 * capacity_kwh


def soc_update(soc_t, power_kw, dt_hours: float, capacity_kwh: float, efficiency: float):
    """Eq. 4.3 -- one discrete SoC step."""
    return soc_t + (power_kw * efficiency * dt_hours / capacity_kwh) * 100.0


def soc_trajectory(power_profile_kw, soc_init: float, dt_hours: float,
                    capacity_kwh: float, efficiency: float) -> np.ndarray:
    power_profile_kw = np.asarray(power_profile_kw, dtype=float)
    n = len(power_profile_kw)
    soc = np.empty(n + 1, dtype=float)
    soc[0] = soc_init
    for t in range(n):
        soc[t + 1] = soc_update(soc[t], power_profile_kw[t], dt_hours, capacity_kwh, efficiency)
    return soc


def energy_required_kwh(soc_init: float, soc_target: float, capacity_kwh: float) -> float:
    return max(0.0, (soc_target - soc_init) / 100.0 * capacity_kwh)


def min_intervals_required(soc_init: float, soc_target: float, capacity_kwh: float,
                            efficiency: float, p_max: float, dt_hours: float) -> int:
    if soc_target <= soc_init:
        return 0
    gain_per_interval = (p_max * efficiency * dt_hours / capacity_kwh) * 100.0
    if gain_per_interval <= 0:
        return int(1e9)
    return int(np.ceil((soc_target - soc_init) / gain_per_interval))
