"""Glitch-arm scores for every cached time-slide background pair (2026-09-09).

For each background pair (segH, iH) x (segL, iL) in bg_cache_<run>.npz / bg_cache_<run>_win.npz, read the
5-seed deploy-arm ensemble logit g that STAGE A already cached per 1-s window per detector
(streams_<run>_full/<seg>_{H1,L1}_meta.npz["g"], the same quantity that enters lnLambda through
driver_search_multi.gate(g, sigma)). Nothing is recomputed: this is a lookup, asserted against the cached
window GPS times. Output: details/successor_statistic/bg_g_<run>.npz with gH, gL (per pair, bg_cache order),
n, and the source directory.  g_net = (gH + gL)/sqrt(2) is formed downstream in successor_stat.Background.
Usage: build_bg_g.py [O3a O3b O4a O4b]
"""
import os, sys, time
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

sys.path.insert(0, MADGRAV_ROOT + "/search_mode")
import successor_stat as S
SC = MADGRAV_SCRATCH
runs = sys.argv[1:] or S.MAIN_RUNS
for run in runs:
    ta = time.time()
    bg = S.Background(run, verbose=False)
    w = np.load(f"{S.DET}/bg_cache_{run.lower()}_win.npz", allow_pickle=False)
    src = f"{SC}/streams_{run.lower()}_full"
    gH = np.full(bg.n, np.nan); gL = np.full(bg.n, np.nan)
    for det, seg, ii, gps, out in (("H1", bg.hseg, w["iH"].astype(int), w["gps_H"], gH),
                                   ("L1", bg.lseg, w["iL"].astype(int), w["gps_L"], gL)):
        for s in np.unique(seg):
            m = np.load(f"{src}/{bg.seg_names[s]}_{det}_meta.npz"); k = seg == s
            assert np.allclose(m["gps"][ii[k]], gps[k]), (run, det, bg.seg_names[s])
            out[k] = m["g"][ii[k]]
    assert np.isfinite(gH).all() and np.isfinite(gL).all()
    np.savez(f"{S.DET}/bg_g_{run.lower()}.npz", gH=gH, gL=gL, n=bg.n, source=src, built=time.strftime("%F %T"))
    gn = (gH + gL) / np.sqrt(2)
    print(f"[bg_g] {run}: {bg.n} pairs, g_net median {np.median(gn):.2f} p1 {np.percentile(gn,1):.2f} p99 {np.percentile(gn,99):.2f}  ({(time.time()-ta)/60:.1f} m)", flush=True)
