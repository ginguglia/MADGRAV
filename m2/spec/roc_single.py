"""ROC comparison of the specialist gates on the single-split TEST set (lockbox sealed): S0 (June), S1 = max(hm, lm), S2, and
hm / lm alone. Signal = test injection pairs. Two background definitions:
  (A) production-weighted time-slide background (test negatives; per-run strata campaign S0-pass / S0-fail weighted by the
      as-run S0 pass fraction, runs by their reach-the-specialists counts = PREREG A4 addendum); FPR = weighted pass fraction;
  (B) the 2,996 counted FAR-family pairs (hardest background, quarantined from training; S0 passes all by construction).
Reports AUC, partial AUC for FPR <= 1% (A) / <= 20% (B) normalised to [0, 1], efficiency at fixed FPR, FPR at fixed
efficiency, and purity S/(S+B) at a stated signal:background prior. Writes roc_single_s0.png / .txt."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/madgrav2_spec"; MD = f"{D}/models_single"
t = {c: np.load(f"{MD}/spec_{c}_single_s0_test.npz") for c in ("hm", "lm", "s2")}
def cat(c):
    a, b = (dict(np.load(f"{D}/train/{c}_f{f}_meta.npz", allow_pickle=True)) for f in (0, 1)); return {k: np.concatenate([a[k], b[k]]) for k in a}
mp = {k: v[t["s2"]["g_pos"]] for k, v in cat("pos").items()}; mn = {k: v[t["s2"]["g_neg"]] for k, v in cat("neg").items()}
REACH = {"o3a": (12775, 602203), "o3b": (4899, 284755), "o4a": (1828, 360925), "o4b": (1742, 142740)}
w = np.zeros(len(mn["run"]))                         # per-negative weight -> weighted FPR = sum(w * pass)
for r, (npass, nreach) in REACH.items():
    W = nreach / sum(v[1] for v in REACH.values()); fp = npass / nreach
    a = (mn["run"] == r) & (mn["source"] == "campaign_pass"); b = (mn["run"] == r) & (mn["source"] == "campaign_fail")
    w[a] = W * fp / a.sum(); w[b] = W * (1 - fp) / b.sum()
R = list(csv.DictReader(open(f"{MD}/protected_scores_s0.csv"))); sb = [r for r in R if r["source"] == "sideband"]
G = {"S0 (June)": (np.maximum(mp["s0_hm"], mp["s0_lm"]), np.maximum(mn["s0_hm"], mn["s0_lm"]), np.array([max(float(r["s0_hm"]), float(r["s0_lm"])) for r in sb])),
     "S1 max(hm,lm)": (np.maximum(t["hm"]["p_pos"], t["lm"]["p_pos"]), np.maximum(t["hm"]["p_neg"], t["lm"]["p_neg"]), np.array([max(float(r["hm"]), float(r["lm"])) for r in sb])),
     "S2 pair": (t["s2"]["p_pos"], t["s2"]["p_neg"], np.array([float(r["s2"]) for r in sb])),
     "hm alone": (t["hm"]["p_pos"], t["hm"]["p_neg"], np.array([float(r["hm"]) for r in sb])),
     "lm alone": (t["lm"]["p_pos"], t["lm"]["p_neg"], np.array([float(r["lm"]) for r in sb]))}
def roc(sp, sn, wn=None):
    wn = np.full(len(sn), 1 / len(sn)) if wn is None else wn
    th = np.unique(np.concatenate([sp, sn]))[::-1]; th = th[np.linspace(0, len(th) - 1, min(len(th), 4000)).astype(int)]
    ps = np.sort(sp); o = np.argsort(sn); sns = sn[o]; cw = np.concatenate([[0], np.cumsum(wn[o])])
    tpr = 1 - np.searchsorted(ps, th, side="left") / len(ps); fpr = cw[-1] - cw[np.searchsorted(sns, th, side="left")]
    return np.concatenate([[0], fpr, [1]]), np.concatenate([[0], tpr, [1]])
def auc(f, e, fmax=1.0):
    m = f <= fmax; ff = np.append(f[m], fmax); ee = np.append(e[m], np.interp(fmax, f, e)); return float(np.trapezoid(ee, ff) / fmax)
def eff_at(f, e, x): return float(np.interp(x, f, e))
def fpr_at(f, e, x): i = np.searchsorted(e, x); return float(f[min(i, len(f) - 1)])
C = {"S0 (June)": "#6b6a63", "S1 max(hm,lm)": "#2a78d6", "S2 pair": "#eb6834", "hm alone": "#1baf7a", "lm alone": "#eda100"}
LS = {"S0 (June)": "-", "S1 max(hm,lm)": "-", "S2 pair": "-", "hm alone": "--", "lm alone": "--"}
out = [f"ROC on the single-split TEST set (seed 0): {len(mp['run'])} injection pairs; (A) {len(mn['run'])} time-slide negatives, production-weighted; (B) {len(sb)} counted FAR-family pairs",
       "", "(A) production-weighted background"]
hdr = f"  {'gate':<15} {'AUC':>6} {'pAUC<1%':>8} | eff @ FPR 0.1% 0.5% 1% 2% | FPR @ eff 85% 90% 95% | purity @ eff90 (prior S:B 1:1000)"
out.append(hdr); curves = {}
for nm, (sp, sn, sside) in G.items():
    f, e = roc(sp, sn, w); curves[nm] = (f, e)
    f90 = fpr_at(f, e, 0.90); pur = 0.90 / (0.90 + 1000 * f90)
    out.append(f"  {nm:<15} {auc(f, e):6.4f} {auc(f, e, 0.01):8.4f} | " + " ".join(f"{100*eff_at(f, e, x):5.1f}" for x in (0.001, 0.005, 0.01, 0.02))
               + " | " + " ".join(f"{100*fpr_at(f, e, x):6.3f}" for x in (0.85, 0.90, 0.95)) + f" | {100*pur:5.1f}%")
out.append(f"  S0 production point: eff {100*(G['S0 (June)'][0] >= 0.5).mean():.1f}% at FPR {100*float((w * (G['S0 (June)'][1] >= 0.5)).sum()):.3f}%")
out += ["", "(B) counted FAR families (S0 passes 100% by construction -> its curve is degenerate)", f"  {'gate':<15} {'AUC':>6} {'pAUC<20%':>9} | FPR @ eff 85% 90% 95%"]
curvesB = {}
for nm, (sp, sn, sside) in G.items():
    f, e = roc(sp, sside); curvesB[nm] = (f, e)
    out.append(f"  {nm:<15} {auc(f, e):6.4f} {auc(f, e, 0.2):9.4f} | " + " ".join(f"{100*fpr_at(f, e, x):6.1f}" for x in (0.85, 0.90, 0.95)))
txt = "\n".join(out); print(txt); open(f"{MD}/roc_single_s0.txt", "w").write(txt + "\n")
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for a, cv, title, xl in ((ax[0], curves, "(A) production-weighted time-slide background", (1e-5, 1)), (ax[1], curvesB, "(B) counted FAR-family pairs (hard background)", (1e-3, 1))):
    for nm, (f, e) in cv.items():
        m = f > 0; a.plot(f[m], 100 * e[m], color=C[nm], ls=LS[nm], lw=2, label=nm)
    a.set_xscale("log"); a.set_xlim(*xl); a.set_ylim(50, 100.5); a.set_title(title, fontsize=10, color="#1a1a19")
    a.set_xlabel("background pass fraction (FPR)", color="#6b6a63"); a.set_ylabel("signal efficiency (%)", color="#6b6a63")
    a.grid(True, color="#e6e5df", lw=0.8, which="both"); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color("#e6e5df")
    a.tick_params(colors="#6b6a63", labelsize=9)
s0e = 100 * (G["S0 (June)"][0] >= 0.5).mean(); s0f = float((w * (G["S0 (June)"][1] >= 0.5)).sum())
ax[0].plot([s0f], [s0e], "o", ms=9, color="#6b6a63", mec="white", mew=2); ax[0].annotate("S0 @ 0.5 (production)", (s0f, s0e), xytext=(8, -14), textcoords="offset points", fontsize=9, color="#1a1a19")
ax[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(f"{MD}/roc_single_s0.png", dpi=130, facecolor="white"); print(f"wrote {MD}/roc_single_s0.png")
