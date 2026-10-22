"""Ranker pilot, stage 1 = FEATURES (CPU). Same noise-pair sampler, injection files and fold convention as
search_mode/refit_lr_m2.py (MADGRAV2-gx), but every raw feature is kept instead of the 8-column LR form, and the zero-lag
triggers of the campaign merge (blindscan_lm.json) are added with the same features, so one file per run holds:
  noise      : time-slid pairs at training lags (4k+2 s; the FAR grid is 4k s), gate-passing, net >= NETMIN, no catalogue
               window; fold = H1 segment parity (as refit_lr_m2); R_NCAP per fold (default 60000)
  inj        : injection rows of inj_<run>_m2 (gate-passing, net >= NETMIN) + mtot, net_snr, is_um
  trig       : zero-lag triggers with the campaign lnL, S0 CNN scores and catalogue match
Raw features (both legs): sigma, centroid v1/v2, arm g v1/g2, coherence (raw + guarded), gx (raw); plus the exact 8-column
LR form (M.feats) so the campaign lnL can be reproduced as the baseline.
Env: campaign env of the run (MADGRAV_ROOT=MADGRAV2-gx, SM_RUN, SM_STREAMS, SM_OUT, SM_GX=1, ...), R_NCAP, R_OUT."""
import os, sys, json, glob, time, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]; sys.path.insert(0, f"{MADGRAV_ROOT}/search_mode"); os.chdir(MADGRAV_ROOT)
os.environ.setdefault("SM_DEV", "cpu"); os.environ.setdefault("SM_ARM_V2", "1")
import driver_search_multi as M
SC = os.environ.get("SCRATCH", MADGRAV_SCRATCH); RUN = os.environ["SM_RUN"]
INJ_GLOB = os.environ.get("SM_LR_INJ_GLOB", f"{SC}/madgrav2/inj_{RUN}_m2/*/*_inj.npz"); OUTD = os.environ.get("R_OUT", MADGRAV_SCRATCH + "/madgrav2_spec/rank"); os.makedirs(OUTD, exist_ok=True)
NETMIN = float(os.environ.get("SM_LR_NETMIN", "3.0")); NCAP = int(os.environ.get("R_NCAP", "60000")); TRIG = f"{M.OUT}/blindscan_lm.json"
assert M.GX.ON, "SM_GX=1 expected (campaign env)"
rng = np.random.default_rng(20260923); t0 = time.time()
SEG = M.SEG["segments"]; avail = [(a, nm) for (a, b, d, nm) in SEG if os.path.exists(f"{M.S_BG}/{M.segname(a,nm)}_H1_meta.npz") and os.path.exists(f"{M.S_BG}/{M.segname(a,nm)}_L1_meta.npz")]
assert len(avail) == len(SEG), "streams incomplete"
segs = [M.load_seg(a, nm) for a, nm in avail]; fold = [i % 2 for i in range(len(segs))]; name2i = {s["name"]: i for i, s in enumerate(segs)}
EX = []                                                   # v1 centroid / v1 arm per segment (not in load_seg)
for s in segs:
    mH = np.load(f"{M.S_BG}/{s['name']}_H1_meta.npz"); mL = np.load(f"{M.S_BG}/{s['name']}_L1_meta.npz"); k = s["n"]
    EX.append(dict(cH1=mH["centroid"][:k], gH1=mH["g"][:k], cL1=mL["centroid"][:k], gL1=mL["g"][:k]))
EVG = np.array(sorted(M.EVENTS.values()), float)
def is_ev(si, idx):
    if not len(EVG): return np.zeros(len(idx), bool)
    g = segs[si]["gps"][idx]; j = np.searchsorted(EVG, g); lo = np.clip(j - 1, 0, len(EVG) - 1); hi = np.clip(j, 0, len(EVG) - 1)
    return (np.abs(EVG[lo] - g) <= 4.0) | (np.abs(EVG[hi] - g) <= 4.0)
print(f"[rank] {RUN}: {len(segs)} segments, NCAP {NCAP}/fold, triggers {TRIG}", flush=True)
# ---- noise pairs at training lags (refit_lr_m2 sampler) ----
npr = {0: [], 1: []}; tries = 0
while (len(npr[0]) < NCAP or len(npr[1]) < NCAP) and tries < 800000:
    tries += 1; ai, bi = rng.integers(len(segs), size=2)
    if ai == bi: continue
    g = fold[ai]
    if len(npr[g]) >= NCAP: continue
    A, B = segs[ai], segs[bi]; k = int(rng.integers(-60, 61)); off = 4 * k + 2
    iH, iL = M.pair_idx(A["n"], B["n"], off)
    if not len(iH): continue
    net = (A["sH"][iH] + B["sL"][iL]) / np.sqrt(2.0)
    m = np.where((net >= NETMIN) & M.cen_gate(A["cH"][iH], B["cL"][iL]) & M.arm_gate(A["gH"][iH], B["gL"][iL], A["sH"][iH], B["sL"][iL]) & ~(is_ev(ai, iH) | is_ev(bi, iL)))[0]
    for j in m[:200]:
        npr[g].append((ai, int(iH[j]), bi, int(iL[j]), off))
        if len(npr[g]) >= NCAP: break
print(f"[rank] noise pairs: fold0 {len(npr[0])} fold1 {len(npr[1])} ({tries} pairings, {(time.time()-t0)/60:.1f} min)", flush=True)
out = {}
for g in (0, 1):
    P = npr[g]; ai_ = np.array([p[0] for p in P]); iH_ = np.array([p[1] for p in P]); bi_ = np.array([p[2] for p in P]); iL_ = np.array([p[3] for p in P])
    nH, nL = {}, {}
    for a_, h_, b_, l_, _ in P: nH.setdefault(a_, set()).add(h_); nL.setdefault(b_, set()).add(l_)
    SH = {a_: M.series_for(segs[a_]["name"], "H1", sorted(w)) for a_, w in nH.items()}; SL = {b_: M.series_for(segs[b_]["name"], "L1", sorted(w)) for b_, w in nL.items()}
    _A = np.array([SH[a_][h_] for a_, h_, b_, l_, _ in P]); _B = np.array([SL[b_][l_] for a_, h_, b_, l_, _ in P]); co = M.coh_vec(_A, _B); gxn = M.gx_vec(_A, _B); del _A, _B
    gv = lambda key, si, w: np.array([segs[s][key][ww] for s, ww in zip(si, w)]); ev = lambda key, si, w: np.array([EX[s][key][ww] for s, ww in zip(si, w)])
    F = dict(sH=gv("sH", ai_, iH_), sL=gv("sL", bi_, iL_), coh=co, gx=gxn, cH2=gv("cH", ai_, iH_), cL2=gv("cL", bi_, iL_), gH2=gv("gH", ai_, iH_), gL2=gv("gL", bi_, iL_),
             cH1=ev("cH1", ai_, iH_), cL1=ev("cL1", bi_, iL_), gH1=ev("gH1", ai_, iH_), gL1=ev("gL1", bi_, iL_), lag=np.array([p[4] for p in P], float))
    F["feats_lr"] = M.feats(F["sH"], F["sL"], F["coh"], F["cH2"], F["cL2"], F["gH2"], F["gL2"], gx=F["gx"])
    for k_, v_ in F.items(): out[f"noise{g}_{k_}"] = v_
    print(f"[rank] fold {g}: noise {len(P)} rows, coh median {np.median(co):.3f}, gx median {np.median(gxn):.2f} ({(time.time()-t0)/60:.1f} min)", flush=True)
# ---- injections ----
I = {k: [] for k in ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1", "mtot", "net_snr", "is_um", "fold")}
for f in sorted(glob.glob(INJ_GLOB)):
    z = np.load(f); nm = os.path.basename(f).replace("_inj.npz", ""); si = name2i.get(nm); g = fold[si] if si is not None else (abs(hash(nm)) % 2)
    k = M.cen_gate(z["cenH2"], z["cenL2"]) & M.arm_gate(z["gH2"], z["gL2"], z["sigH"], z["sigL"]) & (z["net"] >= NETMIN)
    for a, b in (("sH", "sigH"), ("sL", "sigL"), ("coh", "coh"), ("gx", "gx"), ("cH2", "cenH2"), ("cL2", "cenL2"), ("gH2", "gH2"), ("gL2", "gL2"), ("cH1", "cenH"), ("cL1", "cenL"), ("gH1", "gH"), ("gL1", "gL"), ("mtot", "mtot"), ("net_snr", "net_snr"), ("is_um", "is_um")):
        I[a].append(np.asarray(z[b])[k])
    I["fold"].append(np.full(int(k.sum()), g))
I = {k: np.concatenate(v) for k, v in I.items()}; I["feats_lr"] = M.feats(I["sH"], I["sL"], I["coh"], I["cH2"], I["cL2"], I["gH2"], I["gL2"], gx=I["gx"])
for k_, v_ in I.items(): out[f"inj_{k_}"] = v_
print(f"[rank] injections: {len(I['sH'])} rows (fold0 {int((I['fold']==0).sum())}, fold1 {int((I['fold']==1).sum())})", flush=True)
# ---- zero-lag triggers of the campaign merge ----
T = json.load(open(TRIG))["triggers"]; TR = {k: [] for k in ("seg", "gps", "fold", "sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1", "loglr", "net", "cnn_hm", "cnn_lm", "known", "is_glitch")}
for t in T:
    si = name2i.get(t["seg"]); i = int(t["idx"])
    if si is None or i >= segs[si]["n"]: continue
    s = segs[si]; e = EX[si]
    for k_, v_ in (("seg", t["seg"]), ("gps", float(t["gps"])), ("fold", fold[si]), ("sH", s["sH"][i]), ("sL", s["sL"][i]), ("coh", t.get("coh") if t.get("coh") is not None else np.nan), ("gx", t.get("gx") if t.get("gx") is not None else np.nan),
                   ("cH2", s["cH"][i]), ("cL2", s["cL"][i]), ("gH2", s["gH"][i]), ("gL2", s["gL"][i]), ("cH1", e["cH1"][i]), ("cL1", e["cL1"][i]), ("gH1", e["gH1"][i]), ("gL1", e["gL1"][i]),
                   ("loglr", t.get("loglr") if t.get("loglr") is not None else np.nan), ("net", t.get("net") or np.nan), ("cnn_hm", t.get("cnn_hm") if t.get("cnn_hm") is not None else np.nan),
                   ("cnn_lm", t.get("cnn_lm") if t.get("cnn_lm") is not None else np.nan), ("known", t.get("matches_known") or ""), ("is_glitch", bool(t.get("is_glitch", False)))): TR[k_].append(v_)
TR = {k: np.array(v) for k, v in TR.items()}; TR["feats_lr"] = M.feats(TR["sH"], TR["sL"], TR["coh"], TR["cH2"], TR["cL2"], TR["gH2"], TR["gL2"], gx=TR["gx"])
for k_, v_ in TR.items(): out[f"trig_{k_}"] = v_
print(f"[rank] triggers: {len(TR['gps'])} (catalogue-matched {int((TR['known']!='').sum())})", flush=True)
np.savez(f"{OUTD}/feats_{RUN}.npz", run=RUN, netmin=NETMIN, **out)
json.dump(dict(run=RUN, streams=M.S_BG, out=M.OUT, inj_glob=INJ_GLOB, ncap=NCAP, netmin=NETMIN, n_noise={g: len(npr[g]) for g in (0, 1)}, n_inj=int(len(I["sH"])), n_trig=int(len(TR["gps"])),
               lr_model=os.environ.get("SM_LR_MODEL", ""), built=time.strftime("%F %T")), open(f"{OUTD}/feats_{RUN}.json", "w"), indent=1)
print(f"[rank] saved {OUTD}/feats_{RUN}.npz ({(time.time()-t0)/60:.1f} min)", flush=True)
