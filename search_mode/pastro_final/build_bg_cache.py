#!/usr/bin/env python
"""Compact per-run background cache for the FINAL p_astro + VT analysis.

Converts survivors_bg.json (1.5-3.8 GB JSON) into a small npz with exactly the
columns the per-arm FAR counting needs, plus the seg->fold map rebuilt from
blindscan.json triggers (validated against sorted-GPS parity so segments that
never produced a foreground trigger still get a fold).

Run (login node, sequential): python build_bg_cache.py <run>   # o3a|o3b|o4a|o4b
Output: <outdir>/bg_cache_<run>.npz
"""
import os, sys, json
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


RUNS = {
    "o3a": MADGRAV_SCRATCH + "/search_out_o3a_far_f40",
    "o3b": MADGRAV_SCRATCH + "/search_out_o3b_far_f40",
    "o4a": MADGRAV_SCRATCH + "/search_out_o4a_far",
    "o4b": MADGRAV_SCRATCH + "/search_out_o4b_far",
    "o4ars": MADGRAV_SCRATCH + "/search_out_o4ars_far",
}
MERGE_S = 4.0   # driver_search_multi.MERGE_S (L1 family bin, seconds)

def seg_gps(name):  # o3a_1239069668 -> 1239069668
    return int(name.rsplit("_", 1)[1])

def main(run):
    out_dir = RUNS[run]
    b = json.load(open(f"{out_dir}/blindscan.json"))
    far_live = {int(k): float(v) for k, v in b["far_live_yr"].items()}

    # seg -> fold from foreground triggers (authoritative where present)
    fold_trig = {}
    for t in b["triggers"]:
        fold_trig.setdefault(t["seg"], t["fold"])
    del b

    print(f"[{run}] loading survivors_bg.json ...", flush=True)
    S = json.load(open(f"{out_dir}/survivors_bg.json"))
    surv = S["survivors"]
    print(f"[{run}] {len(surv)} scored bg pairs (glitch_thresh={S['glitch_thresh']})", flush=True)

    # full segment set = triggers ∪ survivors; fold = sorted-GPS parity, validated on triggers
    segs = set(fold_trig)
    for r in surv:
        segs.add(r["H1_seg"]); segs.add(r["L1_seg"])
    ordered = sorted(segs, key=seg_gps)
    fold_par = {nm: i % 2 for i, nm in enumerate(ordered)}
    mism = [nm for nm in fold_trig if fold_par[nm] != fold_trig[nm]]
    if mism:
        # parity model failed (segment set incomplete vs the run's avail list) ->
        # fall back to trigger folds only and refuse pairs on unknown segments.
        print(f"[{run}] WARNING: sorted-GPS parity mismatches trigger folds on "
              f"{len(mism)}/{len(fold_trig)} segs -> using trigger folds only", flush=True)
        fold = dict(fold_trig)
    else:
        fold = fold_par
        print(f"[{run}] fold = sorted-GPS parity, validated on {len(fold_trig)} trigger segs "
              f"({len(segs)} segs total)", flush=True)

    seg_ix = {nm: i for i, nm in enumerate(ordered)}
    n = len(surv)
    hseg = np.empty(n, np.int32); lseg = np.empty(n, np.int32)
    fld  = np.empty(n, np.int8);  fam  = np.empty(n, np.int64)
    ll   = np.full(n, -np.inf);   net  = np.empty(n)
    hm   = np.empty(n);           lm   = np.empty(n)
    bad_fold = 0
    for i, r in enumerate(surv):
        hs, ls = r["H1_seg"], r["L1_seg"]
        hseg[i] = seg_ix[hs]; lseg[i] = seg_ix[ls]
        fh = fold.get(hs, -1); fl = fold.get(ls, -1)
        if fh < 0: fh = fl
        if fh != fold.get(ls, fh):
            bad_fold += 1
        fld[i] = fh
        fam[i] = seg_ix[ls] * 10_000_000 + int(r["gps_L"] // MERGE_S)
        if r["loglr"] is not None:
            ll[i] = r["loglr"]
        net[i] = (r["sigma_H"] + r["sigma_L"]) / np.sqrt(2.0)
        hm[i] = r["cnn_hm"]; lm[i] = r["cnn_lm"]
    if bad_fold:
        print(f"[{run}] WARNING: {bad_fold} pairs with inconsistent H/L folds", flush=True)

    dst = f"{out_dir}/bg_cache_{run}.npz"
    np.savez_compressed(
        dst, hseg=hseg, lseg=lseg, fold=fld, fam=fam, loglr=ll, net=net,
        cnn_hm=hm, cnn_lm=lm,
        seg_names=np.array(ordered), seg_fold=np.array([fold[nm] for nm in ordered], np.int8),
        far_live=np.array([far_live[0], far_live[1]]),
        glitch_thresh=np.array(S["glitch_thresh"]),
    )
    print(f"[{run}] -> {dst} ({os.path.getsize(dst)/1e6:.0f} MB)", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
