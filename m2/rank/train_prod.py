"""PREREG_RANKER_20260924 §2-§3.2: production ranker for one run. Replays the build_feats.py noise sampler (seed 20260923) to recover
leg identities, removes training-noise pairs with a leg within +-64 s (same segment/detector) of (a) a counted FAR-family leg or
(b) any zero-lag trigger window of the campaign merge, trains HGB RAW12 with seeds 0-4 per fold, scores the FAR bank
(bgfeats_<run>.npz, finite lnL >= 3) and the zero-lag triggers (feats_<run>.npz) with the OTHER fold's 5 models, and writes
  stat_<run>.npz          : stat = lnL if lnL < 3 else 3 + 20*pbar   (the hook file for driver_blindscan_merge_hook.py)
  stat_identity_<run>.npz : stat = lnL  (reproduction check, §3.4)
Env: campaign env (MADGRAV2-gx), SM_RUN, R_NCAP (60000)."""
import os, sys, json, time, numpy as np, joblib
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]; sys.path.insert(0, f"{MADGRAV_ROOT}/search_mode"); os.chdir(MADGRAV_ROOT)
os.environ.setdefault("SM_DEV", "cpu"); os.environ.setdefault("SM_ARM_V2", "1")
import driver_search_multi as M
from sklearn.ensemble import HistGradientBoostingClassifier
RUN = os.environ["SM_RUN"]; NETMIN = float(os.environ.get("SM_LR_NETMIN", "3.0")); NCAP = int(os.environ.get("R_NCAP", "60000")); D = MADGRAV_SCRATCH + "/madgrav2_spec/rank"
RAW = ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1"); SEEDS = (0, 1, 2, 3, 4); FLOOR = 3.0; SCALE = 20.0; QUAR = 64
def hgb(seed): return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, class_weight="balanced", random_state=seed)
def raw(A, p, idx=None): X = np.column_stack([A[f"{p}_{k}"] for k in RAW]); return X if idx is None else X[idx]
t0 = time.time(); rng = np.random.default_rng(20260923)
SEG = M.SEG["segments"]; segs = [M.load_seg(a, nm) for (a, b, d, nm) in SEG]; fold = [i % 2 for i in range(len(segs))]; name2i = {s["name"]: i for i, s in enumerate(segs)}
EVG = np.array(sorted(M.EVENTS.values()), float)
def is_ev(si, idx):
    if not len(EVG): return np.zeros(len(idx), bool)
    g = segs[si]["gps"][idx]; j = np.searchsorted(EVG, g); lo = np.clip(j - 1, 0, len(EVG) - 1); hi = np.clip(j, 0, len(EVG) - 1)
    return (np.abs(EVG[lo] - g) <= 4.0) | (np.abs(EVG[hi] - g) <= 4.0)
# ---- replay the sampler (identical to build_feats.py) ----
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
Z = np.load(f"{D}/feats_{RUN}.npz", allow_pickle=True); assert all(np.allclose(np.array([p[4] for p in npr[g]]), Z[f"noise{g}_lag"]) for g in (0, 1)), "sampler replay mismatch"
# ---- quarantine anchors: FAR-family legs (both detectors) + zero-lag trigger windows (both detectors) ----
S = np.load(MADGRAV_SCRATCH + "/madgrav2_spec/protected_sel.npz", allow_pickle=True); sel = np.where((S["run"] == RUN) & (S["source"] == "sideband"))[0]
anch = {"H1": {}, "L1": {}}
for i in sel: anch["H1"].setdefault(S["H1_seg"][i], []).append(int(S["iH"][i])); anch["L1"].setdefault(S["L1_seg"][i], []).append(int(S["iL"][i]))
T = json.load(open(f"{M.OUT}/blindscan_lm.json"))["triggers"]; ntw = 0
for t in T:
    si = name2i.get(t["seg"])
    if si is None: continue
    j = int(np.argmin(np.abs(segs[si]["gps"] - t["gps"]))); anch["H1"].setdefault(t["seg"], []).append(j); anch["L1"].setdefault(t["seg"], []).append(j); ntw += 1
anch = {d: {s_: np.sort(np.array(v)) for s_, v in a.items()} for d, a in anch.items()}
def near(d, s_, w):
    a = anch[d].get(s_)
    if a is None: return False
    j = np.searchsorted(a, w); return (j < len(a) and a[j] - w <= QUAR) or (j > 0 and w - a[j - 1] <= QUAR)
keep = {g: np.array([not (near("H1", segs[p[0]]["name"], p[1]) or near("L1", segs[p[2]]["name"], p[3])) for p in npr[g]]) for g in (0, 1)}
print(f"[prod] {RUN}: anchors FAR-family pairs {len(sel)}, trigger windows {ntw}; training noise kept fold0 {int(keep[0].sum())}/{NCAP}, fold1 {int(keep[1].sum())}/{NCAP} ({(time.time()-t0)/60:.1f} min)", flush=True)
# ---- train 2 folds x 5 seeds ----
models = {}; rep = []
for g in (0, 1):
    kp = np.where(keep[g])[0]; tr = Z["inj_fold"] == g; h = 1 - g
    X = np.vstack([raw(Z, f"noise{g}", kp), raw(Z, "inj", tr)]); y = np.concatenate([np.zeros(len(kp)), np.ones(int(tr.sum()))])
    models[g] = [hgb(s).fit(X, y) for s in SEEDS]
    Xh_n = raw(Z, f"noise{h}"); Xh_i = raw(Z, "inj", Z["inj_fold"] == h)
    pn = np.mean([m.predict_proba(Xh_n)[:, 1] for m in models[g]], 0); pi = np.mean([m.predict_proba(Xh_i)[:, 1] for m in models[g]], 0)
    e = {x: float((pi >= np.quantile(pn, 1 - x)).mean()) for x in (1e-2, 1e-3, 1e-4)}; rep.append(dict(fold=g, n_noise=int(len(kp)), n_inj=int(tr.sum()), eff_heldout=e))
    print(f"[prod] {RUN} fold {g}: trained 5 seeds on {len(kp)} noise + {int(tr.sum())} inj; held-out fold {h} eff @ noise 1e-2/1e-3/1e-4 = {100*e[1e-2]:.1f}/{100*e[1e-3]:.1f}/{100*e[1e-4]:.1f}% ({(time.time()-t0)/60:.1f} min)", flush=True)
joblib.dump({g: models[g] for g in models}, f"{D}/models_{RUN}.joblib")
# ---- score the bank and the triggers with the OTHER fold's models ----
B = np.load(f"{D}/bgfeats_{RUN}.npz", allow_pickle=True); ok = np.isfinite(B["bg_loglr"]) & (B["bg_loglr"] >= FLOOR) & np.isfinite(B["bg_coh"]) & np.isfinite(B["bg_gx"])
Xb = raw({f"bg_{k}": B[f"bg_{k}"][ok] for k in RAW}, "bg"); bf = B["bg_fold"][ok]; bl = B["bg_loglr"][ok]; pb = np.full(len(bl), np.nan)
for h in (0, 1):
    m = bf == h; pb[m] = np.mean([mdl.predict_proba(Xb[m])[:, 1] for mdl in models[1 - h]], 0)
tok = np.isfinite(Z["trig_coh"]) & np.isfinite(Z["trig_gx"]) & np.isfinite(Z["trig_loglr"]); Xt = raw(Z, "trig", tok); tf = Z["trig_fold"][tok]; tl = Z["trig_loglr"][tok]; pt = np.full(len(tl), np.nan)
for h in (0, 1):
    m = tf == h; pt[m] = np.mean([mdl.predict_proba(Xt[m])[:, 1] for mdl in models[1 - h]], 0)
n_hi_missing = int((np.isfinite(Z["trig_loglr"]) & (Z["trig_loglr"] >= FLOOR) & ~tok).sum()); assert n_hi_missing == 0, f"{n_hi_missing} triggers with lnL>=3 lack features"
stat_b = np.where(bl < FLOOR, bl, FLOOR + SCALE * pb); stat_t = np.where(tl < FLOOR, tl, FLOOR + SCALE * pt)
common = dict(bank_H1_seg=B["bg_H1_seg"][ok], bank_iH=B["bg_iH"][ok], bank_L1_seg=B["bg_L1_seg"][ok], bank_iL=B["bg_iL"][ok], bank_loglr=bl, bank_p=pb, trig_seg=Z["trig_seg"][tok], trig_gps=Z["trig_gps"][tok], trig_loglr=tl, trig_p=pt)
np.savez(f"{D}/stat_{RUN}.npz", bank_stat=stat_b, trig_stat=stat_t, **common); np.savez(f"{D}/stat_identity_{RUN}.npz", bank_stat=bl, trig_stat=tl, **common)
json.dump(dict(run=RUN, prereg="PREREG_RANKER_20260924", seeds=list(SEEDS), floor=FLOOR, scale=SCALE, quarantine_s=QUAR, n_anchor_pairs=int(len(sel)), n_trigger_windows=ntw, folds=rep, n_bank=int(len(bl)), n_trig=int(len(tl)), built=time.strftime("%F %T")), open(f"{D}/stat_{RUN}.json", "w"), indent=1)
print(f"[prod] {RUN}: DONE bank {len(bl)} rows, triggers {len(tl)} (lnL>=3: {int((tl>=FLOOR).sum())}); stat_{RUN}.npz + stat_identity_{RUN}.npz ({(time.time()-t0)/60:.1f} min)", flush=True)
