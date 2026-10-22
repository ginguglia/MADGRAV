"""Ranker pilot, stage 2 = cross-fitted TRAIN + READ-OUT (CPU, sklearn). Per run and per fold g: train on fold g (noise + injections),
score fold 1-g (noise, injections, zero-lag triggers). Models:
  lnL (campaign)  : the campaign likelihood ratio, recomputed from lr_<run>_m2_gx.npz with the cross-fit convention verified against
                    the merge's own trigger lnL (printed); the baseline every other row must beat.
  logit LR8       : balanced logistic regression on the same 8 LR-form features (sanity: should ~ lnL).
  HGB LR8         : gradient-boosted trees (sklearn HistGradientBoosting) on the same 8 features -> learner effect only.
  HGB RAW12       : gradient-boosted trees on the 12 raw features (sigma, centroid v1+v2, arm g v1+g2, raw coh, gx) -> + feature effect.
  MLP RAW12       : small neural net (64-64) on the 12 raw features.
Held-out metrics (mean over the two folds): AUC, injection efficiency at noise pass 1e-2 / 1e-3 / 1e-4, efficiency by Mtot at 1e-3.
Zero-lag: S0-gate-passing catalogue triggers whose score exceeds ALL held-out noise of their fold / all but 1e-3 of it; same for
non-catalogue triggers (candidates). NOTE: the CNN scores are not available for noise/injections here, so the CNN is applied as
the production gate to the triggers only, identically for every model. Output: madgrav2_spec/rank/rank_readout.txt, rank_roc.png."""
import os, glob, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/madgrav2_spec/rank"; SC = MADGRAV_SCRATCH + "/madgrav2"; RUNS = [r for r in ("o3a", "o3b", "o4a", "o4b") if os.path.exists(f"{D}/feats_{r}.npz")]
RAW = ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1"); MB = [(10, 30), (30, 100), (100, 300), (300, 1e9)]
C = {"lnL (campaign)": "#6b6a63", "logit LR8": "#eda100", "HGB LR8": "#eb6834", "HGB RAW12": "#2a78d6", "MLP RAW12": "#1baf7a"}
LS = {"lnL (campaign)": "-", "logit LR8": "--", "HGB LR8": "-", "HGB RAW12": "-", "MLP RAW12": "--"}
def loglr(lr, g, F): return lr[f"be{g}"][0] + ((np.asarray(F, float) - lr[f"mu{g}"]) / lr[f"sd{g}"]) @ lr[f"be{g}"][1:]
def raw(Z, pfx, idx=None):
    X = np.column_stack([Z[f"{pfx}_{k}"] for k in RAW]); return X if idx is None else X[idx]
def make(name):
    if name == "logit LR8": return make_pipeline(StandardScaler(), LogisticRegression(C=10.0, class_weight="balanced", max_iter=2000))
    if name.startswith("HGB"): return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, class_weight="balanced", random_state=0)
    if name == "MLP RAW12": return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0))
def roc(sp, sn):
    th = np.unique(np.concatenate([sp, sn]))[::-1]; th = th[np.linspace(0, len(th) - 1, min(len(th), 3000)).astype(int)]
    return np.concatenate([[0], 1 - np.searchsorted(np.sort(sn), th, side="left") / len(sn), [1]]), np.concatenate([[0], 1 - np.searchsorted(np.sort(sp), th, side="left") / len(sp), [1]])
def auc(f, e): return float(np.trapezoid(e, f))
def eff_at(sp, sn, fpr): return float((sp >= np.quantile(sn, 1 - fpr)).mean())
out = []; curves = {}; summary = {}
for run in RUNS:
    Z = np.load(f"{D}/feats_{run}.npz", allow_pickle=True); lr = np.load(f"{SC}/lr_{run}_m2_gx.npz")
    # cross-fit convention of the campaign lnL: which fold's beta reproduces the merge's trigger lnL?
    tf = Z["trig_fold"]; tl = Z["trig_loglr"]; ok = np.isfinite(tl)
    same = np.nanmedian(np.abs(np.concatenate([loglr(lr, g, Z["trig_feats_lr"][ok & (tf == g)]) for g in (0, 1)]) - np.concatenate([tl[ok & (tf == g)] for g in (0, 1)])))
    other = np.nanmedian(np.abs(np.concatenate([loglr(lr, 1 - g, Z["trig_feats_lr"][ok & (tf == g)]) for g in (0, 1)]) - np.concatenate([tl[ok & (tf == g)] for g in (0, 1)])))
    XF = (lambda g: 1 - g) if other < same else (lambda g: g)   # model index used to score a fold-g row
    out.append(f"== {run}: noise {len(Z['noise0_sH'])}+{len(Z['noise1_sH'])}, injections {len(Z['inj_sH'])}, triggers {len(Z['trig_gps'])}; campaign lnL reproduced with "
               f"{'the OTHER fold' if other < same else 'the SAME fold'}'s beta (median |diff| {min(same, other):.3f} vs {max(same, other):.3f})")
    S = {}  # held-out scores per model: dict model -> (pos list, neg list, trig array)
    for name in C:
        P, N, TS = [], [], np.full(len(Z["trig_gps"]), np.nan)
        for g in (0, 1):                                         # train on fold g, score fold 1-g
            h = 1 - g; inj_tr = Z["inj_fold"] == g; inj_ho = Z["inj_fold"] == h; tr_ho = Z["trig_fold"] == h
            if name == "lnL (campaign)":
                f = lambda F_lr, F_raw: loglr(lr, XF(h), F_lr)
            else:
                Xtr = np.vstack([Z[f"noise{g}_feats_lr"], Z["inj_feats_lr"][inj_tr]]) if "LR8" in name else np.vstack([raw(Z, f"noise{g}"), raw(Z, "inj", inj_tr)])
                y = np.concatenate([np.zeros(len(Z[f"noise{g}_sH"])), np.ones(int(inj_tr.sum()))]); mdl = make(name).fit(Xtr, y)
                f = (lambda F_lr, F_raw: mdl.predict_proba(F_lr)[:, 1]) if "LR8" in name else (lambda F_lr, F_raw: mdl.predict_proba(F_raw)[:, 1])
            P.append(f(Z["inj_feats_lr"][inj_ho], raw(Z, "inj", inj_ho))); N.append(f(Z[f"noise{h}_feats_lr"], raw(Z, f"noise{h}")))
            if tr_ho.any(): TS[tr_ho] = f(Z["trig_feats_lr"][tr_ho], raw(Z, "trig", tr_ho))
        S[name] = (P, N, TS)
    out.append(f"  {'model':<16} {'AUC':>6} | eff @ noise 1e-2  1e-3  1e-4 | eff@1e-3 by Mtot 10-30 30-100 100-300 >=300 | catalogue trig (S0-pass) > all noise / > 99.9% | non-cat trig > all noise")
    gate = np.nan_to_num(np.maximum(Z["trig_cnn_hm"], Z["trig_cnn_lm"]), nan=0) > 0.5; known = Z["trig_known"] != ""
    for name, (P, N, TS) in S.items():
        A = []; E = {x: [] for x in (1e-2, 1e-3, 1e-4)}; EM = {b: [] for b in MB}
        for g in (0, 1):
            h = 1 - g; f_, e_ = roc(P[g], N[g]); A.append(auc(f_, e_)); curves[(run, name, g)] = (f_, e_)
            for x in E: E[x].append(eff_at(P[g], N[g], x))
            mt = Z["inj_mtot"][Z["inj_fold"] == h]; q = np.quantile(N[g], 1 - 1e-3)
            for b in MB: EM[b].append(float((P[g][(mt >= b[0]) & (mt < b[1])] >= q).mean()) if ((mt >= b[0]) & (mt < b[1])).any() else np.nan)
        # zero-lag: trigger of fold h vs held-out noise of fold h (scored by the same model)
        above_all = np.zeros(len(TS), bool); above_999 = np.zeros(len(TS), bool)
        for g in (0, 1):
            h = 1 - g; m = Z["trig_fold"] == h; above_all[m] = TS[m] > N[g].max(); above_999[m] = TS[m] >= np.quantile(N[g], 1 - 1e-3)
        summary[(run, name)] = dict(auc=float(np.mean(A)), eff=[float(np.mean(E[x])) for x in E], cat_all=int((above_all & gate & known).sum()), cat_999=int((above_999 & gate & known).sum()), noncat_all=int((above_all & gate & ~known).sum()))
        out.append(f"  {name:<16} {np.mean(A):6.4f} | {100*np.mean(E[1e-2]):16.1f} {100*np.mean(E[1e-3]):5.1f} {100*np.mean(E[1e-4]):5.1f} | "
                   + " ".join(f"{100*np.nanmean(EM[b]):6.1f}" for b in MB) + f"  | {int((above_all & gate & known).sum()):3d} / {int((above_999 & gate & known).sum()):3d} of {int((gate & known).sum())} | {int((above_all & gate & ~known).sum()):3d} of {int((gate & ~known).sum())}")
    # which catalogue triggers change status between lnL and HGB RAW12
    a0 = S["lnL (campaign)"][2]; a1 = S["HGB RAW12"][2]; ch = []
    for g in (0, 1):
        h = 1 - g; m = np.where((Z["trig_fold"] == h) & gate & known)[0]; n0 = S["lnL (campaign)"][1][g]; n1 = S["HGB RAW12"][1][g]
        for i in m:
            p0 = float((n0 >= a0[i]).mean()); p1 = float((n1 >= a1[i]).mean())
            if (p0 <= 1e-3) != (p1 <= 1e-3): ch.append(f"{Z['trig_known'][i]} (noise exceedance lnL {p0:.1e} -> HGB {p1:.1e})")
    if ch: out.append("  status changes at 1e-3 (lnL -> HGB RAW12): " + "; ".join(ch))
txt = "\n".join(out); print(txt); open(f"{D}/rank_readout.txt", "w").write(txt + "\n"); json.dump({f"{r}|{m}": v for (r, m), v in summary.items()}, open(f"{D}/rank_summary.json", "w"), indent=1)
fig, ax = plt.subplots(1, len(RUNS), figsize=(4.6 * len(RUNS), 4.8), squeeze=False)
for j, run in enumerate(RUNS):
    a = ax[0, j]
    for name in C:
        f_, e_ = curves[(run, name, 0)]; m = f_ > 0; a.plot(f_[m], 100 * e_[m], color=C[name], ls=LS[name], lw=2, label=name if j == 0 else None)
    a.set_xscale("log"); a.set_xlim(1e-5, 1); a.set_ylim(40, 100.5); a.set_title(f"{run}: injections vs time-slide noise (held-out fold 1)", fontsize=9, color="#1a1a19")
    a.set_xlabel("noise pass fraction", color="#6b6a63"); a.set_ylabel("injection efficiency (%)" if j == 0 else "", color="#6b6a63"); a.grid(True, color="#e6e5df", lw=0.8, which="both"); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.tick_params(colors="#6b6a63", labelsize=8)
ax[0, 0].legend(frameon=False, fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(f"{D}/rank_roc.png", dpi=130, facecolor="white"); print(f"wrote {D}/rank_roc.png")
