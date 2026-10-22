"""MADGRAV2 likelihood-ratio refit for one run (pilot: O4b), same 7-feature form as the frozen model
(sH, sL, coh, cH2, cL2, gate(g2H), gate(g2L)) so driver_blindscan consumes it unchanged via SM_LR_MODEL.
Noise pairs: time-slid window pairs from the MADGRAV2 streams at TRAINING lags = offsets of 4k+2 s (the FAR prune uses the
4 s offset grid -> disjoint by construction), same-segment pairs excluded, windows within +-4 s of a catalog event
excluded, morphology gate v2 applied (only gate-passing pairs are ranked), net >= SM_LR_NETMIN (3.0). Coherence is the
real one from the whitened series cache. Signal rows: this run's injection files (inject.py with SM_ARM_V2=1), gate-passing.
Folds as in driver_blindscan: segment index parity over the segments with streams; injections take the fold of their
host segment. Output: SM_LR_OUT npz with mu{0,1}, sd{0,1}, be{0,1}, Xs, sf, floor (+ a .json provenance)."""
import os, sys, json, glob, time, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]; sys.path.insert(0, f"{MADGRAV_ROOT}/search_mode"); os.chdir(MADGRAV_ROOT)
os.environ.setdefault("SM_DEV", "cpu"); os.environ.setdefault("SM_ARM_V2", "1")
import driver_search_multi as M
SC = os.environ.get("SCRATCH", MADGRAV_SCRATCH); RUN = os.environ["SM_RUN"]
INJ_GLOB = os.environ.get("SM_LR_INJ_GLOB", f"{SC}/madgrav2/inj_{RUN}/*/*_inj.npz"); OUTP = os.environ.get("SM_LR_OUT", f"{SC}/madgrav2/lr_{RUN}_m2.npz")
NETMIN = float(os.environ.get("SM_LR_NETMIN", "3.0")); NCAP = int(os.environ.get("SM_LR_NCAP", "30000")); FLOOR = float(os.environ.get("SM_CAND_FLOOR", "3.0"))
rng = np.random.default_rng(20260917); t0 = time.time()
SEG = M.SEG["segments"]; avail = [(a, nm) for (a, b, d, nm) in SEG if os.path.exists(f"{M.S_BG}/{M.segname(a,nm)}_H1_meta.npz") and os.path.exists(f"{M.S_BG}/{M.segname(a,nm)}_L1_meta.npz")]
print(f"[lr] {RUN}: {len(avail)}/{len(SEG)} segments with streams; features cH/cL={M.CEN_KEY} gH/gL={M.ARM_KEY}", flush=True)
assert len(avail) == len(SEG), "streams incomplete — fold parity would not match the blind scan"
segs = [M.load_seg(a, nm) for a, nm in avail]; fold = [i % 2 for i in range(len(segs))]; name2i = {s["name"]: i for i, s in enumerate(segs)}
EVG = np.array(sorted(M.EVENTS.values()), float)
def is_ev(si, idx):
    if not len(EVG): return np.zeros(len(idx), bool)
    g = segs[si]["gps"][idx]; j = np.searchsorted(EVG, g); lo = np.clip(j - 1, 0, len(EVG) - 1); hi = np.clip(j, 0, len(EVG) - 1)
    return (np.abs(EVG[lo] - g) <= 4.0) | (np.abs(EVG[hi] - g) <= 4.0)
# ---- noise pairs at training lags ----
npr = {0: [], 1: []}; tries = 0
while (len(npr[0]) < NCAP or len(npr[1]) < NCAP) and tries < 400000:
    tries += 1; ai, bi = rng.integers(len(segs), size=2)
    if ai == bi: continue
    g = fold[ai]
    if len(npr[g]) >= NCAP: continue
    A, B = segs[ai], segs[bi]; k = int(rng.integers(-60, 61)); off = 4 * k + 2                        # training lag grid: 4k+2 s (FAR uses 4k)
    iH, iL = M.pair_idx(A["n"], B["n"], off)
    if not len(iH): continue
    net = (A["sH"][iH] + B["sL"][iL]) / np.sqrt(2.0)
    m = np.where((net >= NETMIN) & M.cen_gate(A["cH"][iH], B["cL"][iL]) & M.arm_gate(A["gH"][iH], B["gL"][iL], A["sH"][iH], B["sL"][iL]) & ~(is_ev(ai, iH) | is_ev(bi, iL)))[0]
    for j in m[: max(1, 200 - 0)]:
        npr[g].append((ai, int(iH[j]), bi, int(iL[j])))
        if len(npr[g]) >= NCAP: break
print(f"[lr] noise pairs: fold0 {len(npr[0])} fold1 {len(npr[1])} ({tries} pairings, {(time.time()-t0)/60:.1f} min)", flush=True)
Xn = {}
for g in (0, 1):
    P = npr[g]; ai_ = np.array([p[0] for p in P]); iH_ = np.array([p[1] for p in P]); bi_ = np.array([p[2] for p in P]); iL_ = np.array([p[3] for p in P])
    nH, nL = {}, {}
    for a_, h_, b_, l_ in P: nH.setdefault(a_, set()).add(h_); nL.setdefault(b_, set()).add(l_)
    SH = {a_: M.series_for(segs[a_]["name"], "H1", sorted(w)) for a_, w in nH.items()}; SL = {b_: M.series_for(segs[b_]["name"], "L1", sorted(w)) for b_, w in nL.items()}
    _A = np.array([SH[a_][h_] for a_, h_, b_, l_ in P]); _B = np.array([SL[b_][l_] for a_, h_, b_, l_ in P]); co = M.coh_vec(_A, _B); gxn = M.gx_vec(_A, _B)   # MADGRAV2.1: + guided excess
    gv = lambda key, si, w: np.array([segs[s][key][ww] for s, ww in zip(si, w)])
    Xn[g] = M.feats(gv("sH", ai_, iH_), gv("sL", bi_, iL_), co, gv("cH", ai_, iH_), gv("cL", bi_, iL_), gv("gH", ai_, iH_), gv("gL", bi_, iL_), gx=gxn)
    print(f"[lr] fold {g}: noise features {Xn[g].shape}, coh median {np.median(co):.3f}" + (f", gx median {np.median(gxn):.2f} p99 {np.percentile(gxn,99):.2f} p99.9 {np.percentile(gxn,99.9):.2f}" if gxn is not None else "") + f" ({(time.time()-t0)/60:.1f} min)", flush=True)
# ---- injections ----
Xs, sf = [], []
for f in sorted(glob.glob(INJ_GLOB)):
    z = np.load(f); nm = os.path.basename(f).replace("_inj.npz", ""); si = name2i.get(nm)
    g = fold[si] if si is not None else (abs(hash(nm)) % 2)
    cH, cL = z["cenH2"], z["cenL2"]; gH, gL = z["gH2"], z["gL2"]; k = M.cen_gate(cH, cL) & M.arm_gate(gH, gL, z["sigH"], z["sigL"]) & (z["net"] >= NETMIN)
    X = M.feats(z["sigH"][k], z["sigL"][k], z["coh"][k], cH[k], cL[k], gH[k], gL[k], gx=(z["gx"][k] if M.GX.ON else None)); Xs.append(X); sf.append(np.full(len(X), g))
Xs = np.vstack(Xs); sf = np.concatenate(sf); print(f"[lr] injections: {len(Xs)} gate-passing triggering rows (fold0 {int((sf==0).sum())}, fold1 {int((sf==1).sum())})", flush=True)
out = {"Xs": Xs, "sf": sf, "floor": np.array(FLOOR)}
for g in (0, 1):
    mu, sd, be = M.fit_lr(Xn[g], Xs[sf == g]); out[f"mu{g}"], out[f"sd{g}"], out[f"be{g}"] = mu, sd, be
    print(f"[lr] fold {g}: beta (bias, sH, sL, coh, cH2, cL2, gate g2H, gate g2L{', gx' if M.GX.ON else ''}) = {np.round(be, 3).tolist()}", flush=True)
    ln = M.loglr(mu, sd, be, Xn[g]); ls = M.loglr(mu, sd, be, Xs[sf == g]); print(f"        noise loglr median {np.median(ln):.2f} p99 {np.percentile(ln,99):.2f} max {ln.max():.2f} | inj median {np.median(ls):.2f} frac>=floor {np.mean(ls>=FLOOR):.3f}", flush=True)
np.savez(OUTP, **out); json.dump(dict(run=RUN, streams=M.S_BG, cen_key=M.CEN_KEY, arm_key=M.ARM_KEY, netmin=NETMIN, ncap=NCAP, floor=FLOOR, n_noise={g: int(len(Xn[g])) for g in (0, 1)}, n_inj=int(len(Xs)), inj_glob=INJ_GLOB, gx=(dict(on=True, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in M.GX.PARAMS.items()}) if M.GX.ON else dict(on=False)), coh_guard_hz=M.COH_GUARD_HZ, built=time.strftime("%F %T")), open(OUTP.replace(".npz", ".json"), "w"), indent=1)
print(f"[lr] saved {OUTP} ({(time.time()-t0)/60:.1f} min)", flush=True)
