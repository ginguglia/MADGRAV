"""Ranker pilot, step 1d: sideband read-out WITHOUT leg leakage. Replays build_feats.py's noise sampler (seed 20260923) to recover the
leg identity of every training-noise pair, drops from the training noise of fold g every pair whose H1 or L1 leg lies within +-64 s
(same segment) of ANY FAR-family leg (both detectors, all folds), retrains HGB LR8 / HGB RAW12 on the stripped noise + fold-g
injections, and scores the fold-(1-g) injections and FAR-family pairs. Same metrics as eval_sideband.py; the campaign lnL row is the
stored model (trained on the full noise -- it cannot memorise legs). Output: rank/sideband_clean_<run>.txt (+ .json)."""
import os, sys, json, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]; sys.path.insert(0, f"{MADGRAV_ROOT}/search_mode"); os.chdir(MADGRAV_ROOT)
os.environ.setdefault("SM_DEV", "cpu"); os.environ.setdefault("SM_ARM_V2", "1")
import driver_search_multi as M
from sklearn.ensemble import HistGradientBoostingClassifier
RUN = os.environ["SM_RUN"]; NETMIN = float(os.environ.get("SM_LR_NETMIN", "3.0")); NCAP = int(os.environ.get("R_NCAP", "60000")); D = MADGRAV_SCRATCH + "/madgrav2_spec/rank"; SC = MADGRAV_SCRATCH + "/madgrav2"
RAW = ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1")
rng = np.random.default_rng(20260923)
SEG = M.SEG["segments"]; segs = [M.load_seg(a, nm) for (a, b, d, nm) in SEG]; fold = [i % 2 for i in range(len(segs))]
EVG = np.array(sorted(M.EVENTS.values()), float)
def is_ev(si, idx):
    if not len(EVG): return np.zeros(len(idx), bool)
    g = segs[si]["gps"][idx]; j = np.searchsorted(EVG, g); lo = np.clip(j - 1, 0, len(EVG) - 1); hi = np.clip(j, 0, len(EVG) - 1)
    return (np.abs(EVG[lo] - g) <= 4.0) | (np.abs(EVG[hi] - g) <= 4.0)
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
Z = np.load(f"{D}/feats_{RUN}.npz", allow_pickle=True); W = np.load(f"{D}/feats_side_{RUN}.npz", allow_pickle=True); lr = np.load(f"{SC}/lr_{RUN}_m2_gx.npz")
assert all(np.allclose(np.array([p[4] for p in npr[g]]), Z[f"noise{g}_lag"]) for g in (0, 1)), "sampler replay mismatch"
S = np.load(MADGRAV_SCRATCH + "/madgrav2_spec/protected_sel.npz", allow_pickle=True); sel = np.where((S["run"] == RUN) & (S["source"] == "sideband"))[0]
anch = {"H1": {}, "L1": {}}
for i in sel: anch["H1"].setdefault(S["H1_seg"][i], []).append(int(S["iH"][i])); anch["L1"].setdefault(S["L1_seg"][i], []).append(int(S["iL"][i]))
anch = {d: {s_: np.sort(np.array(v)) for s_, v in a.items()} for d, a in anch.items()}
def near(d, s_, w):
    a = anch[d].get(s_)
    if a is None: return False
    j = np.searchsorted(a, w); return (j < len(a) and a[j] - w <= 64) or (j > 0 and w - a[j - 1] <= 64)
keep = {g: np.array([not (near("H1", segs[p[0]]["name"], p[1]) or near("L1", segs[p[2]]["name"], p[3])) for p in npr[g]]) for g in (0, 1)}
def loglr(g, F): return lr[f"be{g}"][0] + ((np.asarray(F, float) - lr[f"mu{g}"]) / lr[f"sd{g}"]) @ lr[f"be{g}"][1:]
def raw(A, p, idx=None): X = np.column_stack([A[f"{p}_{k}"] for k in RAW]); return X if idx is None else X[idx]
def hgb(): return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, class_weight="balanced", random_state=0)
res = {m: dict(inj=[], side=[]) for m in ("lnL (campaign)", "HGB LR8 clean", "HGB RAW12 clean")}
for g in (0, 1):
    h = 1 - g; inj_tr = Z["inj_fold"] == g; inj_ho = Z["inj_fold"] == h; sh = W["side_fold"] == h; kp = keep[g]
    res["lnL (campaign)"]["inj"].append(loglr(g, Z["inj_feats_lr"][inj_ho])); res["lnL (campaign)"]["side"].append(loglr(g, W["side_feats_lr"][sh]))
    for name, use_lr in (("HGB LR8 clean", True), ("HGB RAW12 clean", False)):
        Xn = Z[f"noise{g}_feats_lr"][kp] if use_lr else raw(Z, f"noise{g}", np.where(kp)[0]); Xi = Z["inj_feats_lr"][inj_tr] if use_lr else raw(Z, "inj", inj_tr)
        mdl = hgb().fit(np.vstack([Xn, Xi]), np.concatenate([np.zeros(len(Xn)), np.ones(len(Xi))]))
        res[name]["inj"].append(mdl.predict_proba(Z["inj_feats_lr"][inj_ho] if use_lr else raw(Z, "inj", inj_ho))[:, 1]); res[name]["side"].append(mdl.predict_proba(W["side_feats_lr"][sh] if use_lr else raw(W, "side", sh))[:, 1])
out = [f"== {RUN} CLEAN: training noise stripped of legs within +-64 s of any FAR-family leg: kept fold0 {int(keep[0].sum())}/{len(keep[0])}, fold1 {int(keep[1].sum())}/{len(keep[1])}; sideband {len(W['side_fold'])} pairs",
       f"  {'model':<16} | inj eff above loudest side pair / 10th loudest | side pairs above inj p10 / p50"]
J = {}
for name, r in res.items():
    e1, e10, a10, a50 = [], [], [], []
    for g in (0, 1):
        inj, side = r["inj"][g], r["side"][g]
        if not len(side): continue
        e1.append((inj > side.max()).mean()); e10.append((inj > np.sort(side)[-min(10, len(side))]).mean()); a10.append((side >= np.percentile(inj, 10)).mean()); a50.append((side >= np.percentile(inj, 50)).mean())
    J[name] = dict(e1=float(np.mean(e1)), e10=float(np.mean(e10)), a10=float(np.mean(a10)), a50=float(np.mean(a50)))
    out.append(f"  {name:<16} | {100*np.mean(e1):6.1f}% / {100*np.mean(e10):6.1f}%                       | {100*np.mean(a10):5.1f}% / {100*np.mean(a50):5.1f}%")
txt = "\n".join(out); print(txt, flush=True); open(f"{D}/sideband_clean_{RUN}.txt", "w").write(txt + "\n"); json.dump(J, open(f"{D}/sideband_clean_{RUN}.json", "w"), indent=1)
