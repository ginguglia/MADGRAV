"""Local-ASD consistency veto for BACKGROUND (time-slid) pairs -- successor-statistic amendment (2026-08-18, Sec. 2).

Extends asd_consistency.py (zero-lag only: one segment, one window index for both detectors) to a
time-slid pair (segH, idxH) x (segL, idxL): the H1 local +/-64 s median-Welch ASD is taken from segH around
idxH, the L1 one from segL around idxL; net-sigma is recomputed per detector under its own local ASD; the
HM/LM arms are scored with the pipeline's cnn_hm_lm(segH, idxH, segL, idxL) under the swapped ASDs. The
rule (applied downstream in successor_stat.py) is UNCHANGED from the foreground veto:
    lnLambda-channel family vetoed  iff max(hm_loc, lm_loc) < GLITCH_THRESH
    sigma_net-channel family vetoed iff max(hm_loc, lm_loc) < GLITCH_THRESH  or  net_loc < NETSIG_FLOOR (4.0)
Only the recomputed quantities are produced here (background-only; reads NO foreground quantity).

Input : a pair list npz (rows = bg_cache indices + iH/iL/segment names), optionally sharded k:N.
Output: <out>/bg_veto_<run>_shard<k>.npz  with idx, hm_loc, lm_loc, net_loc, t_pair_s (per pair seconds).
Run inside the run's env (SM_STRAIN, SM_BGJSON, SM_PREP, MADGRAV_ROOT, BLIND_DEV):
    python asd_consistency_bg.py <run> <pairlist.npz> <outdir> [k:N]
"""
import os, sys, time, json
import numpy as np
from gwpy.timeseries import TimeSeries
import driver_blindscan as B
import improved_pipeline as ip
import asd_consistency as AV     # reuse _build_qt / constants (FS, WIN, ASD_HALF)

FS = AV.FS; WIN = AV.WIN; ASD_HALF = AV.ASD_HALF


def _local_asd(seg, idx, det):
    gc = idx + WIN / 2.0
    r = B._strain(seg, det)
    j0 = max(0, int((gc - ASD_HALF) * FS)); j1 = min(len(r), int((gc + ASD_HALF) * FS))
    return TimeSeries(r[j0:j1].astype(np.float64), sample_rate=FS).asd(fftlength=4, overlap=2, method="median")


def _sigma_det(pipe, seg, idx, det):
    mu, sd = (pipe.norm["muH"], pipe.norm["sdH"]) if det == "H1" else (pipe.norm["muL"], pipe.norm["sdL"])
    wh = pipe._whiten(B._win(seg, det, idx)[None, :], det)
    return float((pipe._recon(AV._build_qt(pipe, wh)).reshape(-1)[0] - mu) / sd)


def recompute_local_pair(segH, idxH, segL, idxL):
    """net-sigma + (hm, lm) of a background pair under per-detector local median-Welch ASDs."""
    pipe = B.cpipe()
    sav = {d: pipe.asd[d] for d in ("H1", "L1")}
    aH = _local_asd(segH, idxH, "H1"); aL = _local_asd(segL, idxL, "L1")
    try:
        pipe.asd["H1"], pipe.asd["L1"] = aH, aL
        net = (_sigma_det(pipe, segH, idxH, "H1") + _sigma_det(pipe, segL, idxL, "L1")) / np.sqrt(2.0)
        hm, lm = B.cnn_hm_lm(segH, idxH, segL, idxL)
    finally:
        pipe.asd["H1"], pipe.asd["L1"] = sav["H1"], sav["L1"]
    return dict(net_loc=float(net), hm_loc=float(hm), lm_loc=float(lm))


def main():
    run, plist, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    k, N = (int(x) for x in sys.argv[4].split(":")) if len(sys.argv) > 4 else (0, 1)
    z = np.load(plist, allow_pickle=False)
    idx = z["idx"].astype(np.int64); iH = z["iH"].astype(np.int64); iL = z["iL"].astype(np.int64)
    segH = np.array([str(s) for s in z["segH"]]); segL = np.array([str(s) for s in z["segL"]])
    # contiguous shard of the (hseg, lseg)-sorted list -> strain-cache locality
    o = np.lexsort((segL, segH)); idx, iH, iL, segH, segL = idx[o], iH[o], iL[o], segH[o], segL[o]
    n = len(idx); lo, hi = (n * k) // N, (n * (k + 1)) // N
    os.makedirs(outdir, exist_ok=True)
    dst = f"{outdir}/bg_veto_{run}_shard{k}.npz"
    if os.path.exists(dst):
        print(f"[veto-bg] {dst} exists -- skipping (resumable)", flush=True); return
    print(f"[veto-bg] {run} shard {k}/{N}: pairs {lo}..{hi} of {n}; strain={B.M.STRAIN} prep={B.DS.O4A} dev={B.DEV} "
          f"ASD_HALF={ASD_HALF}s glitch_thresh={B.GLITCH_THRESH}", flush=True)
    hm = np.full(hi - lo, np.nan); lm = np.full(hi - lo, np.nan); nl = np.full(hi - lo, np.nan); tp = np.full(hi - lo, np.nan)
    t0 = time.time(); ck = f"{outdir}/.bg_veto_{run}_shard{k}.ckpt.npz"; start = 0
    if os.path.exists(ck):
        c = np.load(ck); start = int(c["done"]); hm[:start] = c["hm"][:start]; lm[:start] = c["lm"][:start]; nl[:start] = c["net"][:start]; tp[:start] = c["t"][:start]
        print(f"[veto-bg] resuming at {start}", flush=True)
    for j in range(start, hi - lo):
        i = lo + j; t1 = time.time()
        try:
            r = recompute_local_pair(segH[i], int(iH[i]), segL[i], int(iL[i]))
            hm[j], lm[j], nl[j] = r["hm_loc"], r["lm_loc"], r["net_loc"]
        except Exception as e:
            print(f"[veto-bg] pair {int(idx[i])} FAILED: {e!r}", flush=True)
        tp[j] = time.time() - t1
        if (j + 1) % 25 == 0 or j == hi - lo - 1:
            el = time.time() - t0
            print(f"[veto-bg] {j+1}/{hi-lo} done, {el:.0f}s, {el/(j+1-start):.2f} s/pair (last {tp[j]:.2f}s) "
                  f"H={segH[i]} L={segL[i]} net_loc={nl[j]:.2f} hm={hm[j]:.3f} lm={lm[j]:.3f}", flush=True)
            np.savez(ck, done=np.array(j + 1), hm=hm, lm=lm, net=nl, t=tp)
    np.savez(dst, idx=idx[lo:hi], iH=iH[lo:hi], iL=iL[lo:hi], segH=segH[lo:hi], segL=segL[lo:hi],
             hm_loc=hm, lm_loc=lm, net_loc=nl, t_pair_s=tp, run=np.array(run), asd_half=np.array(ASD_HALF),
             wall_s=np.array(time.time() - t0))
    if os.path.exists(ck): os.remove(ck)
    print(f"[veto-bg] DONE shard {k}/{N}: {hi-lo} pairs in {time.time()-t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
