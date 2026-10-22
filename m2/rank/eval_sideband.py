"""Ranker pilot, step 1c: how do the counted FAR-family pairs (the background that sets the detections' FARs) rank under the
cross-fitted rankers, relative to injections? Per run: train HGB (same recipe as train_rank.py) on fold g, score fold 1-g injections
and the fold-(1-g) sideband pairs. Metrics per model:
  - injection efficiency above the LOUDEST sideband pair and above the 10th loudest (FAR-relevant: what a detection must beat);
  - fraction of sideband pairs scoring above the injection 10th / 50th percentile (how signal-like the FAR-setting pairs look);
  - sanity: campaign lnL recomputed vs the stored survivors lnL.
Output: rank/sideband_readout.txt"""
import os, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/madgrav2_spec/rank"; SC = MADGRAV_SCRATCH + "/madgrav2"; RAW = ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1")
def loglr(lr, g, F): return lr[f"be{g}"][0] + ((np.asarray(F, float) - lr[f"mu{g}"]) / lr[f"sd{g}"]) @ lr[f"be{g}"][1:]
def raw(Z, p, idx=None): X = np.column_stack([Z[f"{p}_{k}"] for k in RAW]); return X if idx is None else X[idx]
def hgb(): return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, class_weight="balanced", random_state=0)
out = []; pooled = {}
for run in ("o3a", "o3b", "o4a", "o4b"):
    if not os.path.exists(f"{D}/feats_side_{run}.npz"): continue
    Z = np.load(f"{D}/feats_{run}.npz", allow_pickle=True); W = np.load(f"{D}/feats_side_{run}.npz", allow_pickle=True); lr = np.load(f"{SC}/lr_{run}_m2_gx.npz")
    res = {m: dict(inj=[], side=[]) for m in ("lnL (campaign)", "HGB LR8", "HGB RAW12")}
    chk = []
    for g in (0, 1):
        h = 1 - g; inj_tr = Z["inj_fold"] == g; inj_ho = Z["inj_fold"] == h; sh = W["side_fold"] == h
        # lnL: fold-h rows scored with the other fold's beta (campaign convention, verified in train_rank.py)
        res["lnL (campaign)"]["inj"].append(loglr(lr, g, Z["inj_feats_lr"][inj_ho])); s_ = loglr(lr, g, W["side_feats_lr"][sh]); res["lnL (campaign)"]["side"].append(s_)
        ok = np.isfinite(W["side_loglr"][sh]); chk.append(np.abs(s_[ok] - W["side_loglr"][sh][ok]))
        for name, use_lr in (("HGB LR8", True), ("HGB RAW12", False)):
            Xtr = np.vstack([Z[f"noise{g}_feats_lr"], Z["inj_feats_lr"][inj_tr]]) if use_lr else np.vstack([raw(Z, f"noise{g}"), raw(Z, "inj", inj_tr)])
            y = np.concatenate([np.zeros(len(Z[f"noise{g}_sH"])), np.ones(int(inj_tr.sum()))]); mdl = hgb().fit(Xtr, y)
            res[name]["inj"].append(mdl.predict_proba(Z["inj_feats_lr"][inj_ho] if use_lr else raw(Z, "inj", inj_ho))[:, 1])
            res[name]["side"].append(mdl.predict_proba(W["side_feats_lr"][sh] if use_lr else raw(Z if False else W, "side", sh))[:, 1])
    chk = np.concatenate(chk); out.append(f"== {run}: {len(W['side_fold'])} FAR-family pairs (folds {np.bincount(W['side_fold'], minlength=2).tolist()}); lnL recomputed vs stored: median |diff| {np.median(chk):.3f}, p95 {np.percentile(chk, 95):.3f}")
    out.append(f"  {'model':<16} | inj eff above loudest side pair / 10th loudest (per fold, mean) | side pairs above inj p10 / p50")
    for name, r in res.items():
        e1, e10, a10, a50 = [], [], [], []
        for g in (0, 1):
            inj, side = r["inj"][g], r["side"][g]
            if not len(side): continue
            e1.append((inj > side.max()).mean()); e10.append((inj > np.sort(side)[-min(10, len(side))]).mean())
            a10.append((side >= np.percentile(inj, 10)).mean()); a50.append((side >= np.percentile(inj, 50)).mean())
        pooled.setdefault(name, []).append((np.mean(e1), np.mean(e10), np.mean(a10), np.mean(a50)))
        out.append(f"  {name:<16} | {100*np.mean(e1):6.1f}% / {100*np.mean(e10):6.1f}%                                   | {100*np.mean(a10):5.1f}% / {100*np.mean(a50):5.1f}%")
out.append("== mean over runs"); [out.append(f"  {n:<16} | eff above loudest {100*np.mean([v[0] for v in vals]):5.1f}% / 10th {100*np.mean([v[1] for v in vals]):5.1f}% | side above inj p10 {100*np.mean([v[2] for v in vals]):5.1f}% / p50 {100*np.mean([v[3] for v in vals]):5.1f}%") for n, vals in pooled.items()]
txt = "\n".join(out); print(txt); open(f"{D}/sideband_readout.txt", "w").write(txt + "\n")
