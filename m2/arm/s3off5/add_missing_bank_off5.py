"""_off5 copy of m2/rank/add_missing_bank.py (2026-10-02). ONE change: the bgfeats file lives in R_RANK_DIR (madgrav2/s3_off5/rank),
never in the production rank dir. Completes the FAR-bank feature file with the rows the _off5 merge's shards hold but
survivors_bg.json does not (bank rows that no candidate ever needed CNN-scored). Arm rule max:-2.05 on the louder leg's g2.
Env: _off5 campaign env (SM_SHARD_DIR, SM_NSHARD, SM_STREAMS...), SM_RUN, R_RANK_DIR."""
import os, sys, time, glob, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]; sys.path.insert(0, f"{MADGRAV_ROOT}/search_mode"); os.chdir(MADGRAV_ROOT)
os.environ.setdefault("SM_DEV", "cpu"); os.environ.setdefault("SM_ARM_V2", "1")
import driver_search_multi as M
RUN = os.environ["SM_RUN"]; D = os.environ["R_RANK_DIR"]; SHD = os.environ["SM_SHARD_DIR"]; NSH = int(os.environ["SM_NSHARD"]); THR = -2.05; t0 = time.time()
assert "madgrav2_spec" not in D and "_off5" in SHD, (D, SHD)
segs = [M.load_seg(a, nm) for (a, b, d, nm) in M.SEG["segments"]]; names = [s["name"] for s in segs]
rows = []
for k in range(NSH):
    p = np.load(f"{SHD}/shard_{k}.npz")
    for key in ("bg0", "bg1"):
        for r in p[key]: rows.append((float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])))
print(f"[missing] {RUN}: {len(rows)} shard bg rows ({(time.time()-t0)/60:.1f} min)", flush=True)
keep = []
for ll, ai, iH, bi, iL in rows:
    A, Bs = segs[ai], segs[bi]; g = A["gH"][iH] if A["sH"][iH] >= Bs["sL"][iL] else Bs["gL"][iL]
    if g >= THR: keep.append((ll, ai, iH, bi, iL))
B = np.load(f"{D}/bgfeats_{RUN}.npz", allow_pickle=True); have = set(zip(B["bg_H1_seg"].astype(str), B["bg_iH"].astype(int), B["bg_L1_seg"].astype(str), B["bg_iL"].astype(int)))
miss = [r for r in keep if (names[r[1]], r[2], names[r[3]], r[4]) not in have]
print(f"[missing] {RUN}: after arm rule {len(keep)} rows; in bgfeats {len(keep) - len(miss)}; MISSING {len(miss)}", flush=True)
if not miss: sys.exit(0)
ai_ = np.array([r[1] for r in miss]); iH_ = np.array([r[2] for r in miss]); bi_ = np.array([r[3] for r in miss]); iL_ = np.array([r[4] for r in miss])
meta = {}
def m_(seg, det):
    if (seg, det) not in meta: z = np.load(f"{M.S_BG}/{seg}_{det}_meta.npz"); meta[(seg, det)] = {k: np.asarray(z[k], float) for k in ("gps", "sigma", "centroid", "centroid2", "g", "g2")}
    return meta[(seg, det)]
gv = lambda key, si, w, det: np.array([m_(names[s], det)[key][ww] for s, ww in zip(si, w)])
hs, ls = {}, {}
for s_, w in zip(ai_, iH_): hs.setdefault(names[s_], set()).add(int(w))
for s_, w in zip(bi_, iL_): ls.setdefault(names[s_], set()).add(int(w))
SH = {s_: M.series_for(s_, "H1", sorted(w)) for s_, w in hs.items()}; SL = {s_: M.series_for(s_, "L1", sorted(w)) for s_, w in ls.items()}
A_ = np.array([SH[names[s_]][int(w)] for s_, w in zip(ai_, iH_)]); B_ = np.array([SL[names[s_]][int(w)] for s_, w in zip(bi_, iL_)]); coh = M.coh_vec(A_, B_); gx = M.gx_vec(A_, B_); del A_, B_
F = dict(sH=gv("sigma", ai_, iH_, "H1"), sL=gv("sigma", bi_, iL_, "L1"), coh=coh, gx=gx, cH2=gv("centroid2", ai_, iH_, "H1"), cL2=gv("centroid2", bi_, iL_, "L1"), gH2=gv("g2", ai_, iH_, "H1"), gL2=gv("g2", bi_, iL_, "L1"),
         cH1=gv("centroid", ai_, iH_, "H1"), cL1=gv("centroid", bi_, iL_, "L1"), gH1=gv("g", ai_, iH_, "H1"), gL1=gv("g", bi_, iL_, "L1"))
F["feats_lr"] = M.feats(F["sH"], F["sL"], F["coh"], F["cH2"], F["cL2"], F["gH2"], F["gL2"], gx=F["gx"])
F.update(fold=np.array([s_ % 2 for s_ in ai_]), loglr=np.array([r[0] for r in miss]), cnn_hm=np.full(len(miss), np.nan), cnn_lm=np.full(len(miss), np.nan), gps_H=gv("gps", ai_, iH_, "H1"), gps_L=gv("gps", bi_, iL_, "L1"),
         H1_seg=np.array([names[s_] for s_ in ai_]), L1_seg=np.array([names[s_] for s_ in bi_]), iH=iH_, iL=iL_)
out = {k: B[k] for k in B.files}
for k, v in F.items(): out[f"bg_{k}"] = np.concatenate([B[f"bg_{k}"], v]) if B[f"bg_{k}"].ndim == 1 else np.vstack([B[f"bg_{k}"], v])
out["bg_added_missing"] = np.array(len(miss)); np.savez(f"{D}/bgfeats_{RUN}.npz", **out)
print(f"[missing] {RUN}: appended {len(miss)} rows (coh median {np.nanmedian(coh):.3f}); bgfeats now {len(out['bg_loglr'])} rows ({(time.time()-t0)/60:.1f} min)", flush=True)
