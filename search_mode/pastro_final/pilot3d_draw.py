#!/usr/bin/env python
"""STEP 3d draw: stratified random NON-EVENT segments for the epsilon
segment-representativeness control (specification 2026-08-12).

Position-independence is closed (offset-quartile audit); 3d tests whether
epsilon measured on the event-hosting injection segments is representative
of the run at large. Per run: 25 non-event segments, stratified across the
run's calendar span; the O4b sample MUST include recovered-301 members
(6 of 25; set derived 2026-08-12 from the 785-era survivors/trigger union,
search_mode/o4b_recovered301.json - exact 785/301 split).

Exclusions from the universe (confirmed 2026-08-12): segments hosting
ANY time from the 2/day exclusion catalog - the UNION of the GWTC releases'
candidate lists at the FAR<2/day threshold PLUS our own MADGRAV detections
(search_mode/exclusion_2perday.json, built by build_exclusion_2perday.py;
the draw REFUSES to run on a non-final catalog, i.e. one missing the O3
GWTC-2.1/GWTC-3 subthreshold tarball tables) - with +-8 s margin at segment
edges; plus the confident-catalog CSVs (GPS parsed from event names,
redundant safety) and the existing injection host segments
(<run>_events_inj.json). Universe requires both strain files on scratch and
dur >= 1024 s.

Outputs:
  search_mode/pilot3d_<run>_events.json   {name: t0 - 1e6}  (anchor far away
                                          -> inject.py's avoidance is a no-op)
  search_mode/pilot3d_<run>_segs.json     {name: {"coincident_lock": [t0]}}
  pastro_final/pilot3d_manifest.json      full draw record (strata, r301 flags)

Run: madgrav-venv python pilot3d_draw.py   (idempotent; seeded rng 20260812)
"""
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
SM = f"{MG}/search_mode"
HERE = f"{MG}/search_mode/pastro_final"
RUNS = ["o3a", "o3b", "o4a", "o4b"]
K = 25
K_R301 = 6           # O4b: quota from the recovered-301 set
MIN_DUR = 1024.0
EDGE_S = 8.0
GPS_UNIX = 315964800
LEAP = 18
RNG = np.random.default_rng(20260812)


def name_to_gps(name):
    """GWyymmdd_hhmmss -> GPS (+-1 s; segment-level exclusion only)."""
    try:
        d = datetime.strptime(name.strip()[2:], "%y%m%d_%H%M%S")
        return d.replace(tzinfo=timezone.utc).timestamp() - GPS_UNIX + LEAP
    except ValueError:
        return None


def event_gps(run):
    path = f"{MG}/figures/catalog_o3o4/cat_{run}_v2.csv"
    out, skipped = [], []
    for row in csv.DictReader(open(path)):
        g = name_to_gps(row["name"])
        (out if g is not None else skipped).append(g or row["name"])
    if skipped:
        print(f"  [{run}] WARNING: {len(skipped)} unparsable event names "
              f"(excluded by name impossible): {skipped[:3]}", flush=True)
    return np.array(out)


def stratified_pick(cands, k, rng):
    """cands sorted by t0; k strata by index, one uniform pick per stratum."""
    if len(cands) <= k:
        return list(range(len(cands)))
    edges = np.linspace(0, len(cands), k + 1).astype(int)
    return [int(rng.integers(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])
            if hi > lo]


def main():
    excl = json.load(open(f"{SM}/exclusion_2perday.json"))
    assert excl.get("final"), \
        "exclusion_2perday.json is NOT final (O3 tarball tables missing) - " \
        "rerun build_exclusion_2perday.py without --no-tarballs first"
    excl_gps = np.asarray(excl["gps"], float)
    manifest = {"seed": 20260812, "K": K, "runs": {},
                "exclusion_provenance": excl["provenance"],
                "exclusion_n": excl["n"]}
    r301 = set(json.load(open(f"{SM}/o4b_recovered301.json"))["recovered301"])
    for run in RUNS:
        segs = json.load(open(f"{SC}/{run}_full_coincident.json"))["segments"]
        evg = np.concatenate([event_gps(run), excl_gps])
        inj_hosts = set(json.load(open(f"{SM}/{run}_events_inj.json")))
        uni = []
        for t0, t1, dur, name in segs:
            if dur < MIN_DUR or name in inj_hosts:
                continue
            if not (os.path.exists(f"{SC}/strain_{run}_full/{name}_H1.npz")
                    and os.path.exists(f"{SC}/strain_{run}_full/{name}_L1.npz")):
                continue
            if len(evg) and np.any((evg >= t0 - EDGE_S) & (evg <= t1 + EDGE_S)):
                continue
            uni.append((float(t0), float(dur), name))
        uni.sort()
        if run == "o4b":
            u301 = [u for u in uni if u[2] in r301]
            urest = [u for u in uni if u[2] not in r301]
            picks = ([urest[i] for i in stratified_pick(urest, K - K_R301, RNG)]
                     + [u301[i] for i in stratified_pick(u301, K_R301, RNG)])
        else:
            picks = [uni[i] for i in stratified_pick(uni, K, RNG)]
        picks.sort()
        ev_json = {name: t0 - 1e6 for t0, dur, name in picks}
        seg_json = {name: {"coincident_lock": [t0]} for t0, dur, name in picks}
        json.dump(ev_json, open(f"{SM}/pilot3d_{run}_events.json", "w"), indent=1)
        json.dump(seg_json, open(f"{SM}/pilot3d_{run}_segs.json", "w"), indent=1)
        months = sorted({datetime.fromtimestamp(GPS_UNIX + t0, tz=timezone.utc)
                         .strftime("%Y_%m") for t0, _, _ in picks})
        manifest["runs"][run] = dict(
            universe=len(uni), picked=len(picks), months=months,
            r301_members=[n for _, _, n in picks if n in r301],
            segments=[dict(name=n, t0=t0, dur=d,
                           r301=(n in r301)) for t0, d, n in picks])
        print(f"[{run}] universe {len(uni)} (of {len(segs)}) -> picked "
              f"{len(picks)} across {len(months)} months"
              + (f", r301: {len(manifest['runs'][run]['r301_members'])}"
                 if run == "o4b" else ""), flush=True)
    json.dump(manifest, open(f"{HERE}/pilot3d_manifest.json", "w"), indent=1)
    print(f"[draw] -> {HERE}/pilot3d_manifest.json", flush=True)


if __name__ == "__main__":
    main()
