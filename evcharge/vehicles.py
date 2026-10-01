"""
Preset vehicles popular in India (battery capacity and AC charger rating).

Values are manufacturer or major-portal published figures, checked in
October 2026. Where a model comes in several battery sizes, one common
variant is used and noted. Users can still edit any value in the app.

Fields: category, variant, battery_kwh, ac_kw (rated AC charger power at
the wall, in kW), source, optional note.
"""
from __future__ import annotations

VEHICLE_PRESETS: dict[str, dict] = {
    # ---- Cars (ranked by 2026 sales, top 10) ----
    "MG Windsor EV": dict(category="Car", variant="Pro, 52.9 kWh", battery_kwh=52.9, ac_kw=7.4,
                          source="https://www.zigwheels.com/mg-motor-cars/windsor-ev/"),
    "Mahindra XEV 9S": dict(category="Car", variant="79 kWh pack, 7.2 kW charger option", battery_kwh=79.0, ac_kw=7.2,
                            source="https://www.cardekho.com/mahindra/xev-9s/specs"),
    "Tata Nexon EV": dict(category="Car", variant="45 kWh", battery_kwh=45.0, ac_kw=7.2,
                          source="https://www.cardekho.com/tata/nexon-ev/specs"),
    "Tata Punch EV": dict(category="Car", variant="40 kWh (Long Range)", battery_kwh=40.0, ac_kw=7.2,
                          source="https://www.cardekho.com/tata/punch-ev/specs"),
    "Mahindra XEV 9e": dict(category="Car", variant="79 kWh pack, 7.2 kW charger option", battery_kwh=79.0, ac_kw=7.2,
                            source="https://www.cardekho.com/mahindra/xev-9e/specs"),
    "Tata Harrier EV": dict(category="Car", variant="75 kWh", battery_kwh=75.0, ac_kw=7.2,
                            source="https://www.cardekho.com/tata/harrier-ev/specs"),
    "Maruti e Vitara": dict(category="Car", variant="61 kWh (Zeta/Alpha)", battery_kwh=61.0, ac_kw=7.4,
                            source="https://www.autocarindia.com/cars/maruti-suzuki/e-vitara/specifications"),
    "Tata Curvv EV": dict(category="Car", variant="55 kWh", battery_kwh=55.0, ac_kw=7.2,
                          source="https://www.cardekho.com/tata/curvv-ev/specs"),
    "Mahindra BE 6": dict(category="Car", variant="79 kWh pack, 11.2 kW charger option", battery_kwh=79.0, ac_kw=11.2,
                          source="https://www.cardekho.com/mahindra/be-6/specs"),
    "Tata Tiago EV": dict(category="Car", variant="24 kWh", battery_kwh=24.0, ac_kw=7.2,
                          source="https://www.cardekho.com/tata/tiago-ev/specs"),
    # ---- Electric scooters (top 5) ----
    "Bajaj Chetak": dict(category="Scooter", variant="3503, 3.5 kWh, 950 W charger", battery_kwh=3.5, ac_kw=0.95,
                         note="950 W charger rating is from ZigWheels; consistent with the official 0-80% in about 3 h 25 min.",
                         source="https://www.zigwheels.com/bajaj-bikes/chetak/specifications/"),
    "TVS iQube": dict(category="Scooter", variant="3.5 kWh, 650 W charger", battery_kwh=3.5, ac_kw=0.65,
                      source="https://www.tvsmotor.com/electric-scooters/tvs-iqube/faq"),
    "Hero Vida VX2 Plus": dict(category="Scooter", variant="Plus 4.4 kWh, 1 kW charger", battery_kwh=4.4, ac_kw=1.0,
                          source="https://www.rushlane.com/hero-vida-vx2-plus-4-4-kwh-launched-187-kms-range-90-kmph-top-speed-12550810.html"),
    "Ather Rizta": dict(category="Scooter", variant="2.9 kWh, 350 W portable charger", battery_kwh=2.9, ac_kw=0.35,
                        note="350 W rating is from BikeWale; consistent with the 0-100% in about 8.5 h (AutocarIndia).",
                        source="https://www.autocarindia.com/auto-features/ather-rizta-faqs-on-battery-range-features-and-price-438114"),
    "TVS Orbiter": dict(category="Scooter", variant="3.1 kWh, 650 W charger", battery_kwh=3.1, ac_kw=0.65,
                        source="https://www.tvsmotor.com/electric-scooters/tvs-orbiter/faq"),
}


def preset_names() -> list[str]:
    return list(VEHICLE_PRESETS)


def label(name: str) -> str:
    p = VEHICLE_PRESETS[name]
    return f"{name} ({p['battery_kwh']:g} kWh, {p['ac_kw']:g} kW)"
