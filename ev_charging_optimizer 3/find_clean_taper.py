"""
find_clean_taper.py
--------------------
Pulls several EV charging sessions from ACN-Data (Caltech's Adaptive
Charging Network dataset) for two sites -- JPL and a Silicon Valley
office -- and scores each one by how "smooth" its charging current
curve is over time.

A low noise_score means the current mostly moves in one direction
(a good candidate for showing a clean CC -> CV taper, like textbook
lithium-ion charging). A high noise_score means the current keeps
reversing direction (like the oscillating, infrastructure-throttled
session already documented in Section 5.6 of the paper).

BEFORE RUNNING:
1. pip install acnportal
2. Paste your ACN-Data API token below (get one free at
   https://ev.caltech.edu/register if you don't have one).
3. Run:  python find_clean_taper.py
"""

from acnportal.acndata import DataClient
from datetime import datetime
import numpy as np

TOKEN = "kcOV82xez-VHsa0272gZipaGeY4LLuxyDoaOZxr2pKo"  # <-- put your API token here

SITES_TO_CHECK = ["jpl", "office_001"]  # NOTE: office site id has an underscore
SESSIONS_PER_SITE = 5
MIN_ENERGY_KWH = 10
MIN_DATA_POINTS = 50
MAX_SESSIONS_TO_SCAN = 20  # give up on a site after checking this many, in case of repeated server errors


def noise_score(current):
    """
    Lower score = smoother, more monotonic decline (good taper candidate).
    Higher score = noisier, more direction reversals (like the
    infrastructure-throttled session already in the paper).
    """
    current = np.array(current)
    diffs = np.diff(current)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    return sign_changes / len(current)


def main():
    client = DataClient(api_token=TOKEN)
    results = []

    for site in SITES_TO_CHECK:
        print(f"\n{'=' * 70}")
        print(f"SITE: {site}")
        print(f"{'=' * 70}")
        try:
            sessions = client.get_sessions_by_time(
                site=site,
                start=datetime(2019, 1, 1),
                end=datetime(2020, 1, 1),
                min_energy=MIN_ENERGY_KWH,
                timeseries=True,
            )
        except Exception as e:
            print(f"  Could not query site '{site}': {e}")
            continue

        count = 0
        scanned = 0
        session_iter = iter(sessions)
        while count < SESSIONS_PER_SITE and scanned < MAX_SESSIONS_TO_SCAN:
            try:
                s = next(session_iter)
            except StopIteration:
                break
            except Exception as e:
                # ACN-Data's /ts endpoint is documented as occasionally slow/flaky;
                # skip this one session and keep going rather than aborting the whole site.
                print(f"  (skipped one session after a server error: {e})")
                scanned += 1
                continue

            scanned += 1
            cc = s.get("chargingCurrent")
            if not cc or "current" not in cc:
                continue
            current = cc["current"]
            if len(current) < MIN_DATA_POINTS:
                continue

            score = noise_score(current)
            count += 1
            results.append({
                "site": site,
                "sessionID": s["sessionID"],
                "kWhDelivered": s["kWhDelivered"],
                "points": len(current),
                "noise_score": score,
            })
            print(f"  {s['sessionID'][:42]:44s} kWh={s['kWhDelivered']:6.1f}  "
                  f"points={len(current):5d}  noise_score={score:.3f}  (lower = smoother)")

        if count == 0:
            print(f"  No usable sessions found for site '{site}' after scanning {scanned}.")

    if not results:
        print("\nNo usable sessions found. Try widening the date range or lowering MIN_ENERGY_KWH.")
        return

    results.sort(key=lambda r: r["noise_score"])
    print(f"\n{'=' * 70}")
    print("BEST (smoothest) CANDIDATES, most likely to show a clean taper:")
    print(f"{'=' * 70}")
    for r in results[:3]:
        print(f"  site={r['site']:10s} sessionID={r['sessionID']}")
        print(f"    kWh={r['kWhDelivered']:.1f}  points={r['points']}  noise_score={r['noise_score']:.3f}")

    print("\nNext step: send this printed output back, and we'll pick the top")
    print("candidate to actually plot and inspect visually.")


if __name__ == "__main__":
    main()
