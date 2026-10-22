"""Ranker pilot, step 1b: features of the 2,996 counted FAR-family pairs (madgrav2_spec/protected_sel.npz, source == 'sideband') for
run SM_RUN, in the same form as build_feats.py (12 raw + 8 LR-form), with coherence and gx from the series cache exactly as the
LR refit computes them. Fold = campaign fold of the H1 segment (segment-index parity). Output: rank/feats_side_<run>.npz."""
import os, sys, time, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]; sys.path.insert(0, f"{MADGRAV_ROOT}/search_mode"); os.chdir(MADGRAV_ROOT)
os.environ.setdefault("SM_DEV", "cpu"); os.environ.setdefault("SM_ARM_V2", "1")
import driver_search_multi as M
RUN = os.environ["SM_RUN"]; OUTD = os.environ.get("R_OUT", MADGRAV_SCRATCH + "/madgrav2_spec/rank"); assert M.GX.ON
S = np.load(MADGRAV_SCRATCH + "/madgrav2_spec/protected_sel.npz", allow_pickle=True); sel = np.where((S["run"] == RUN) & (S["source"] == "sideband"))[0]
SEG = M.SEG["segments"]; names = [M.segname(a, nm) for (a, b, d, nm) in SEG]; name2i = {n: i for i, n in enumerate(names)}
t0 = time.time(); print(f"[side] {RUN}: {len(sel)} sideband pairs", flush=True)
meta = {}
def m_(seg, det):
    if (seg, det) not in meta: z = np.load(f"{M.S_BG}/{seg}_{det}_meta.npz"); meta[(seg, det)] = {k: np.asarray(z[k], float) for k in ("sigma", "centroid", "centroid2", "g", "g2")}
    return meta[(seg, det)]
segH = S["H1_seg"][sel]; segL = S["L1_seg"][sel]; iH = S["iH"][sel].astype(int); iL = S["iL"][sel].astype(int)
legsH, legsL = {}, {}
for s_, i_ in zip(segH, iH): legsH.setdefault(s_, set()).add(int(i_))
for s_, i_ in zip(segL, iL): legsL.setdefault(s_, set()).add(int(i_))
SH = {s_: M.series_for(s_, "H1", sorted(w)) for s_, w in legsH.items()}; SL = {s_: M.series_for(s_, "L1", sorted(w)) for s_, w in legsL.items()}
A = np.array([SH[s_][int(i_)] for s_, i_ in zip(segH, iH)]); B = np.array([SL[s_][int(i_)] for s_, i_ in zip(segL, iL)]); coh = M.coh_vec(A, B); gx = M.gx_vec(A, B); del A, B
F = dict(sH=np.array([m_(s_, "H1")["sigma"][i_] for s_, i_ in zip(segH, iH)]), sL=np.array([m_(s_, "L1")["sigma"][i_] for s_, i_ in zip(segL, iL)]), coh=coh, gx=gx,
         cH2=np.array([m_(s_, "H1")["centroid2"][i_] for s_, i_ in zip(segH, iH)]), cL2=np.array([m_(s_, "L1")["centroid2"][i_] for s_, i_ in zip(segL, iL)]),
         gH2=np.array([m_(s_, "H1")["g2"][i_] for s_, i_ in zip(segH, iH)]), gL2=np.array([m_(s_, "L1")["g2"][i_] for s_, i_ in zip(segL, iL)]),
         cH1=np.array([m_(s_, "H1")["centroid"][i_] for s_, i_ in zip(segH, iH)]), cL1=np.array([m_(s_, "L1")["centroid"][i_] for s_, i_ in zip(segL, iL)]),
         gH1=np.array([m_(s_, "H1")["g"][i_] for s_, i_ in zip(segH, iH)]), gL1=np.array([m_(s_, "L1")["g"][i_] for s_, i_ in zip(segL, iL)]))
F["feats_lr"] = M.feats(F["sH"], F["sL"], F["coh"], F["cH2"], F["cL2"], F["gH2"], F["gL2"], gx=F["gx"])
F["fold"] = np.array([name2i[s_] % 2 for s_ in segH]); F["loglr"] = S["loglr"][sel].astype(float); F["gps_H"] = S["gps_H"][sel]; F["gps_L"] = S["gps_L"][sel]
F["cnn_hm"] = S["cnn_hm"][sel].astype(float); F["cnn_lm"] = S["cnn_lm"][sel].astype(float); F["top50"] = S["top50"][sel]
assert np.allclose(F["sH"], S["sigma_H"][sel].astype(float), atol=1e-6), "sigma mismatch vs selection"
np.savez(f"{OUTD}/feats_side_{RUN}.npz", run=RUN, **{f"side_{k}": v for k, v in F.items()})
print(f"[side] {RUN}: coh median {np.median(coh):.3f}, gx median {np.median(gx):.2f}, folds {np.bincount(F['fold'], minlength=2).tolist()} ({(time.time()-t0)/60:.1f} min) -> feats_side_{RUN}.npz", flush=True)
