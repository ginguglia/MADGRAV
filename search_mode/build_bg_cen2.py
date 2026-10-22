"""MADGRAV2: centroid v2 for every cached time-slide background pair (lookup of streams meta "centroid2", 2026-09-17).

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import successor_stat as S
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = MADGRAV_SCRATCH
runs = sys.argv[1:] or S.MAIN_RUNS
for run in runs:
    ta = time.time()
    bg = S.Background(run, verbose=False)
    w = np.load(f"{S.DET}/bg_cache_{run.lower()}_win.npz", allow_pickle=False)
    src = f"{SC}/streams_{run.lower()}_full"
    cH2 = np.full(bg.n, np.nan); cL2 = np.full(bg.n, np.nan)
    for det, seg, ii, gps, out in (("H1", bg.hseg, w["iH"].astype(int), w["gps_H"], cH2),
                                   ("L1", bg.lseg, w["iL"].astype(int), w["gps_L"], cL2)):
        for s in np.unique(seg):
            m = np.load(f"{src}/{bg.seg_names[s]}_{det}_meta.npz"); k = seg == s
            assert np.allclose(m["gps"][ii[k]], gps[k]), (run, det, bg.seg_names[s])
            out[k] = m["centroid2"][ii[k]]
    np.savez(f"{S.DET}/bg_cen2_{run.lower()}.npz", cH2=cH2, cL2=cL2, n=bg.n, source=src, built=time.strftime("%F %T"))
    print(f"[bg_cen2] {run}: {bg.n} pairs, centroid2 median H {np.nanmedian(cH2):.0f} L {np.nanmedian(cL2):.0f}, pairs failing 220: {np.mean((cH2>=220)|(cL2>=220)):.3f}  ({(time.time()-ta)/60:.1f} m)", flush=True)
