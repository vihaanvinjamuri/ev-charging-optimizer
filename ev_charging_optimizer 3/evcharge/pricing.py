"""
Time-of-Use (TOU) electricity pricing model (Section 4.3.4, Eq. 4.4-4.5).
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

DEFAULT_TOU_SCHEDULE: List[Tuple[float, float, float, str]] = [
    (0.0, 6.0, 4.5, "Off-Peak"),
    (6.0, 16.0, 6.0, "Shoulder"),
    (16.0, 21.0, 8.5, "Peak"),
    (21.0, 24.0, 6.0, "Shoulder"),
]


def price_at_hour(hour: float, schedule=None) -> float:
    schedule = schedule or DEFAULT_TOU_SCHEDULE
    h = hour % 24.0
    for start, end, price, _label in schedule:
        if start <= h < end:
            return price
    return schedule[-1][2]


def label_at_hour(hour: float, schedule=None) -> str:
    schedule = schedule or DEFAULT_TOU_SCHEDULE
    h = hour % 24.0
    for start, end, _price, label in schedule:
        if start <= h < end:
            return label
    return schedule[-1][3]


def build_price_vector(arrival_hour: float, n_intervals: int, dt_hours: float, schedule=None) -> np.ndarray:
    times = arrival_hour + np.arange(n_intervals) * dt_hours
    return np.array([price_at_hour(t, schedule) for t in times])


def build_label_vector(arrival_hour: float, n_intervals: int, dt_hours: float, schedule=None) -> np.ndarray:
    times = arrival_hour + np.arange(n_intervals) * dt_hours
    return np.array([label_at_hour(t, schedule) for t in times])


def charging_cost(price_vector, power_profile_kw, dt_hours: float) -> float:
    price_vector = np.asarray(price_vector, dtype=float)
    power_profile_kw = np.asarray(power_profile_kw, dtype=float)
    return float(np.sum(price_vector * power_profile_kw) * dt_hours)


def validate_schedule(schedule) -> None:
    hours = sorted(schedule, key=lambda row: row[0])
    if hours[0][0] != 0.0 or hours[-1][1] != 24.0:
        raise ValueError("TOU schedule must start at hour 0 and end at hour 24.")
    for (s1, e1, *_), (s2, e2, *_) in zip(hours, hours[1:]):
        if e1 != s2:
            raise ValueError(f"TOU schedule has a gap/overlap between {e1}h and {s2}h.")
