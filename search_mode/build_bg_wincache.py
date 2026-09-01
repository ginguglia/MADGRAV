#!/usr/bin/env python
"""Index-aligned WINDOW cache for the successor statistic (amendment 2026-08-18).

bg_cache_<run>.npz (build_bg_cache.py) holds one row per record of survivors_bg.json["survivors"]
(same order) but not the window coordinates. The successor's narrowed self-exclusion (own H1 window
+/-4 s, own L1 window +/-4 s) and the background consistency veto (asd_consistency_bg.py) need them.
This streams the (1.5-3.9 GB, pretty-printed) survivors_bg.json line by line -- no full JSON load --
and writes details/successor_statistic/bg_cache_<run>_win.npz with, per row i (== bg_cache row i):
    iH, gps_H, sigma_H, cen_H, iL, gps_L, sigma_L, cen_L, lag_s, loglr_chk, hm_chk, lm_chk
plus H1_seg_ix / L1_seg_ix (indices into bg_cache seg_names) as a cross-check.
Run: nice -n 10 python build_bg_wincache.py <run>   (o3a|o3b|o4a|o4b|o4ars)
"""
import os, sys, re, time
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


RUNS = {
    "o3a": MADGRAV_SCRATCH + "/search_out_o3a_far_f40",
    "o3b": MADGRAV_SCRATCH + "/search_out_o3b_far_f40",
    "o4a": MADGRAV_SCRATCH + "/search_out_o4a_far",
    "o4b": MADGRAV_SCRATCH + "/search_out_o4b_far",
    "o4ars": MADGRAV_SCRATCH + "/search_out_o4ars_far",
}
DET = MADGRAV_ROOT + "/details/successor_statistic"
KEYS = ("H1_seg", "iH", "gps_H", "sigma_H", "cen_H", "L1_seg", "iL", "gps_L", "sigma_L", "cen_L", "lag_s", "loglr", "cnn_hm", "cnn_lm")
RX = re.compile(r'^\s*"([A-Za-z0-9_]+)":\s*(.*?),?\s*$')

def main(run):
    d = RUNS[run]; t0 = time.time()
    z = np.load(f"{d}/bg_cache_{run}.npz", allow_pickle=False)
    n = len(z["hseg"]); seg_ix = {str(nm): i for i, nm in enumerate(z["seg_names"])}
    cols = {k: [] for k in KEYS}
    cur = {}; depth_rec = False; nrec = 0
    with open(f"{d}/survivors_bg.json") as fh:
        in_surv = False
        for line in fh:
            if not in_surv:
                if line.startswith('  "survivors"'): in_surv = True
                continue
            s = line.strip()
            if s.startswith("]"): break
            if s.startswith("{"):
                cur = {}; continue
            if s.startswith("}"):
                for k in KEYS: cols[k].append(cur.get(k))
                nrec += 1
                if nrec % 500000 == 0: print(f"[{run}] {nrec} records ({time.time()-t0:.0f}s)", flush=True)
                continue
            m = RX.match(line)
            if m: cur[m.group(1)] = m.group(2)
    assert nrec == n, (nrec, n)
    def f(k, dt=np.float64):
        v = cols[k]
        if dt is np.float64:
            return np.array([np.nan if (x is None or x == "null") else float(x) for x in v], np.float64)
        return np.array([int(x) for x in v], np.int64)
    out = dict(iH=f("iH", np.int64), gps_H=f("gps_H"), sigma_H=f("sigma_H"), cen_H=f("cen_H"),
               iL=f("iL", np.int64), gps_L=f("gps_L"), sigma_L=f("sigma_L"), cen_L=f("cen_L"), lag_s=f("lag_s"),
               loglr_chk=f("loglr"), hm_chk=f("cnn_hm"), lm_chk=f("cnn_lm"))
    out["H1_seg_ix"] = np.array([seg_ix[x.strip('"')] for x in cols["H1_seg"]], np.int64)
    out["L1_seg_ix"] = np.array([seg_ix[x.strip('"')] for x in cols["L1_seg"]], np.int64)
    # cross-checks against bg_cache row order
    assert np.array_equal(out["H1_seg_ix"], z["hseg"].astype(np.int64)), "hseg mismatch"
    assert np.array_equal(out["L1_seg_ix"], z["lseg"].astype(np.int64)), "lseg mismatch"
    ll = z["loglr"]; lc = out["loglr_chk"]
    fin = np.isfinite(ll)
    assert np.array_equal(fin, np.isfinite(lc)) and np.allclose(ll[fin], lc[fin]), "loglr mismatch"
    assert np.allclose(z["cnn_hm"], out["hm_chk"]) and np.allclose(z["cnn_lm"], out["lm_chk"]), "cnn mismatch"
    net = (out["sigma_H"] + out["sigma_L"]) / np.sqrt(2.0)
    assert np.allclose(net, z["net"]), "net mismatch"
    fam = z["lseg"].astype(np.int64) * 10_000_000 + (out["gps_L"] // 4.0).astype(np.int64)
    assert np.array_equal(fam, z["fam"].astype(np.int64)), "fam mismatch"
    dst = f"{DET}/bg_cache_{run}_win.npz"
    np.savez_compressed(dst, **out, n=np.array(n), source=np.array(f"{d}/survivors_bg.json"))
    print(f"[{run}] {n} rows, all cross-checks (hseg,lseg,loglr,cnn,net,fam) PASS -> {dst} ({os.path.getsize(dst)/1e6:.0f} MB, {time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    for r in sys.argv[1:]: main(r)
