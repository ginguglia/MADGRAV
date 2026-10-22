"""_off5 copy of m2/rank/build_bg_feats.py (2026-10-02). ONE change: the livetime is read from the _off5 merge's blindscan.json
(untagged -2.05 merge) instead of blindscan_lm.json. Output dir from R_OUT (madgrav2/s3_off5/rank), never the production rank dir.
Features of the WHOLE background that reaches the specialists (search_out_<run>_m2_gx_off5/survivors_bg.json,
admission = louder-leg g2 >= -2.05, centroid2 < 220 Hz both legs, sigma_net >= 3), 12 raw + 8 LR-form, coherence and gx from the
series cache. Env: _off5 campaign env (MADGRAV2-gx), SM_RUN, R_OUT, R_CHUNK (default 40000)."""
import os, sys, json, time, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]; sys.path.insert(0, f"{MADGRAV_ROOT}/search_mode"); os.chdir(MADGRAV_ROOT)
os.environ.setdefault("SM_DEV", "cpu"); os.environ.setdefault("SM_ARM_V2", "1")
import driver_search_multi as M
RUN = os.environ["SM_RUN"]; OUTD = os.environ["R_OUT"]; CH = int(os.environ.get("R_CHUNK", "40000")); assert M.GX.ON
assert "madgrav2_spec" not in OUTD, "refusing to write into the production rank dir"
ARM_MIN, CEN2, NET_MIN = -2.05, 220.0, 3.0; t0 = time.time()
names = [M.segname(a, nm) for (a, b, d, nm) in M.SEG["segments"]]; name2i = {n: i for i, n in enumerate(names)}
d = json.load(open(f"{M.OUT}/survivors_bg.json")); S = d["survivors"]; n = len(S); print(f"[bg] {RUN}: {n} scored survivors ({(time.time()-t0)/60:.1f} min)", flush=True)
col = lambda k, dt=float: np.array([r[k] for r in S], dtype=dt)
segH = col("H1_seg", object); segL = col("L1_seg", object); iH = col("iH", int); iL = col("iL", int); sH = col("sigma_H"); sL = col("sigma_L")
loglr = np.array([r["loglr"] if r.get("loglr") is not None else np.nan for r in S]); cnn_hm = col("cnn_hm"); cnn_lm = col("cnn_lm"); gps_H = col("gps_H"); gps_L = col("gps_L"); del S, d
meta = {}
def m_(seg, det):
    if (seg, det) not in meta: z = np.load(f"{M.S_BG}/{seg}_{det}_meta.npz"); meta[(seg, det)] = {k: np.asarray(z[k], float) for k in ("centroid", "centroid2", "g", "g2")}
    return meta[(seg, det)]
def gather(segs, idx, det, key):
    out = np.empty(len(idx)); order = np.argsort(segs, kind="stable"); s_sorted = segs[order]; bounds = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        o = order[a:b]; out[o] = m_(s_sorted[a], det)[key][idx[o]]
    return out
cH2 = gather(segH, iH, "H1", "centroid2"); cL2 = gather(segL, iL, "L1", "centroid2"); gH2 = gather(segH, iH, "H1", "g2"); gL2 = gather(segL, iL, "L1", "g2")
cH1 = gather(segH, iH, "H1", "centroid"); cL1 = gather(segL, iL, "L1", "centroid"); gH1 = gather(segH, iH, "H1", "g"); gL1 = gather(segL, iL, "L1", "g")
net = (sH + sL) / np.sqrt(2); lead = np.where(sH >= sL, gH2, gL2); cen_ok = ~((np.isfinite(cH2) & (cH2 >= CEN2)) | (np.isfinite(cL2) & (cL2 >= CEN2)))
reach = (lead >= ARM_MIN) & cen_ok & (net >= NET_MIN); R = np.where(reach)[0]
print(f"[bg] {RUN}: reach specialists {len(R)} (S0 pass {int((np.maximum(cnn_hm, cnn_lm)[R] > 0.5).sum())}); unique H1 legs {len(set(zip(segH[R], iH[R])))}, L1 legs {len(set(zip(segL[R], iL[R])))} ({(time.time()-t0)/60:.1f} min)", flush=True)
coh = np.full(len(R), np.nan); gx = np.full(len(R), np.nan)
# chunk by H1 segment so each series file is read once per chunk group
order = R[np.argsort(segH[R], kind="stable")]
for c0 in range(0, len(order), CH):
    idx = order[c0:c0 + CH]; hs, ls = {}, {}
    for s_, w in zip(segH[idx], iH[idx]): hs.setdefault(s_, set()).add(int(w))
    for s_, w in zip(segL[idx], iL[idx]): ls.setdefault(s_, set()).add(int(w))
    SH = {s_: M.series_for(s_, "H1", sorted(w)) for s_, w in hs.items()}; SL = {s_: M.series_for(s_, "L1", sorted(w)) for s_, w in ls.items()}
    A = np.array([SH[s_][int(w)] for s_, w in zip(segH[idx], iH[idx])]); B = np.array([SL[s_][int(w)] for s_, w in zip(segL[idx], iL[idx])])
    pos = np.searchsorted(R, idx); coh[pos] = M.coh_vec(A, B); gx[pos] = M.gx_vec(A, B); del A, B, SH, SL
    if (c0 // CH) % 5 == 0: print(f"[bg] {RUN}: {c0 + len(idx)}/{len(R)} pairs ({(time.time()-t0)/60:.1f} min)", flush=True)
F = dict(sH=sH[R], sL=sL[R], coh=coh, gx=gx, cH2=cH2[R], cL2=cL2[R], gH2=gH2[R], gL2=gL2[R], cH1=cH1[R], cL1=cL1[R], gH1=gH1[R], gL1=gL1[R])
F["feats_lr"] = M.feats(F["sH"], F["sL"], F["coh"], F["cH2"], F["cL2"], F["gH2"], F["gL2"], gx=F["gx"])
F.update(fold=np.array([name2i[s_] % 2 for s_ in segH[R]]), loglr=loglr[R], cnn_hm=cnn_hm[R], cnn_lm=cnn_lm[R], gps_H=gps_H[R], gps_L=gps_L[R], H1_seg=segH[R].astype(str), L1_seg=segL[R].astype(str), iH=iH[R], iL=iL[R])
lm = json.load(open(f"{M.OUT}/blindscan.json")); live = {int(k): float(v) for k, v in lm["far_live_yr"].items()}
os.makedirs(OUTD, exist_ok=True); np.savez(f"{OUTD}/bgfeats_{RUN}.npz", run=RUN, live0=live.get(0, np.nan), live1=live.get(1, np.nan), **{f"bg_{k}": v for k, v in F.items()})
print(f"[bg] {RUN}: DONE {len(R)} rows; coh median {np.nanmedian(coh):.3f}, gx median {np.nanmedian(gx):.2f}; livetime yr {live} ({(time.time()-t0)/60:.1f} min)", flush=True)
