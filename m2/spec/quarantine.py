"""Quarantine for the specialist retrain (PREREG_SPECIALISTS_20260923 §3): no training tile, of either class, within
+-64 s (same detector) of
  - any catalogue event (GWOSC GWTC-* event API files, both detectors),
  - any O3/O4 catalogue name in figures/catalog_o3o4/hl_snr_o3o4.json (both detectors),
  - any MADGRAV1 adopted detection (madgrav_far_final_x1.csv) and ANY MADGRAV2 detection file (detections*.json, all runs, all tags),
  - any counted MADGRAV1 FAR-family leg (bg_families_all.csv, per detector) -- same anchors as m2/arm/assemble.py.
Extra anchor files (e.g. the MADGRAV2 FAR families for the negative drawer) can be passed to build(extra_csv=[...]).
Also: fold_of(day) and calib_of(day), identical to m2/arm/assemble.py."""
import glob, json, csv, re, hashlib, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)

M1 = MADGRAV_SCRATCH + "/MADGRAV1"; M2OUT = MADGRAV_SCRATCH + "/madgrav2"; QUAR_S = 64.0
def fold_of(day): return int(hashlib.sha1(str(int(day)).encode()).hexdigest(), 16) % 2
def calib_of(day): return int(hashlib.sha1(("c" + str(int(day))).encode()).hexdigest(), 16) % 10 == 0
def build(extra_csv=(), verbose=True):
    from gwpy.time import to_gps
    q = {"H1": [], "L1": []}; src = {}
    def both(g, tag):
        for d in q: q[d].append(float(g))
        src[tag] = src.get(tag, 0) + 1
    for fn in glob.glob(MADGRAV_EXTDATA + "/gwosc_eventapi/GWTC-*.json"):
        for v in json.load(open(fn)).get("events", {}).values(): both(v["GPS"], "gwtc")
    for n in json.load(open(f"{M1}/figures/catalog_o3o4/hl_snr_o3o4.json")):
        m = re.match(r"GW(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$", n)
        if m: both(to_gps(f"20{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}"), "o3o4names")
    for r in csv.DictReader(open(f"{M1}/figures/catalog_o3o4/madgrav_far_final_x1.csv")): both(r["gps"], "madgrav1_dets")
    for fn in glob.glob(f"{M2OUT}/search_out_*/detections*.json"):
        try:
            for d_ in json.load(open(fn)): both(d_["gps"], "madgrav2_dets")
        except Exception: pass
    for fn in ["$MADGRAV_SCRATCH/madgrav_qt_build/bg_top50/bg_families_all.csv", *extra_csv]:
        for r in csv.DictReader(open(fn)):
            q["H1"].append(float(r["gps_H"])); q["L1"].append(float(r["gps_L"])); src[fn.split("/")[-1]] = src.get(fn.split("/")[-1], 0) + 1
    Q = {d: np.unique(np.array(v)) for d, v in q.items()}
    if verbose: print(f"[quarantine] anchors H1 {len(Q['H1'])} / L1 {len(Q['L1'])}; sources {src}; +-{QUAR_S} s", flush=True)
    return Q
def quarantined(Q, det, gps):
    a = Q[det]; gps = np.asarray(gps, float); i = np.searchsorted(a, gps); lo = np.clip(i - 1, 0, len(a) - 1); hi = np.clip(i, 0, len(a) - 1)
    return (np.abs(a[lo] - gps) <= QUAR_S) | (np.abs(a[hi] - gps) <= QUAR_S)
