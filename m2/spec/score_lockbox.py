"""PREREG A5 §3: open the lockbox ONCE. Scores the lockbox pairs (split_manifest role 'lock') with the seed-0 single-split nets and
reports S0 / S1 = max(hm,lm) / S2 / S3 = hm alone: production-weighted ROC (AUC, pAUC FPR<1%), eff at fixed FPR, FPR at fixed eff,
and the two operating points whose thresholds come from the TEST set only: (i) 90% test efficiency, (ii) equal weighted background
to S0 @ 0.5 on the test set. Per run and per negative source. Output: models_single/lockbox_readout_s0.{txt,npz}."""
import numpy as np, torch, torch.nn as nn
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/madgrav2_spec"; MD = f"{D}/models_single"; dev = "cuda:0" if torch.cuda.is_available() else "cpu"
class Net2(nn.Module):
    def __init__(s):
        super().__init__(); ch = [2, 16, 32, 64, 128]
        s.b = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
        s.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        s.register_buffer("temperature", torch.ones(1))
    def forward(s, x):
        for bl in s.b: x = bl(x)
        return s.h(x).squeeze(-1)
BANDS = {"hm": (20.0, 300.0), "lm": (50.0, 500.0), "s2": (20.0, 500.0)}; fax = 20.0 + 0.5 * np.arange(961)
man = np.load(f"{MD}/split_manifest.npz"); X = {c: [np.load(f"{D}/train/{c}_f{f}.npy", mmap_mode="r") for f in (0, 1)] for c in ("pos", "neg")}
def cat(c):
    a, b = (dict(np.load(f"{D}/train/{c}_f{f}_meta.npz", allow_pickle=True)) for f in (0, 1)); return {k: np.concatenate([a[k], b[k]]) for k in a}
MC = {c: cat(c) for c in ("pos", "neg")}; N0 = {c: len(X[c][0]) for c in X}
REACH = {"o3a": (12775, 602203), "o3b": (4899, 284755), "o4a": (1828, 360925), "o4b": (1742, 142740)}
def weights(mn):
    w = np.zeros(len(mn["run"]))
    for r, (npass, nreach) in REACH.items():
        W = nreach / sum(v[1] for v in REACH.values()); fp = npass / nreach
        a = (mn["run"] == r) & (mn["source"] == "campaign_pass"); b = (mn["run"] == r) & (mn["source"] == "campaign_fail")
        if a.any() and b.any(): w[a] = W * fp / a.sum(); w[b] = W * (1 - fp) / b.sum()
    return w / w.sum()
def score_rows(c, g):
    models = {}
    for cfg in BANDS:
        m = Net2().to(dev); m.load_state_dict(torch.load(f"{MD}/spec_{cfg}_single_s0.pt", map_location=dev)); m.eval(); models[cfg] = m
    out = {cfg: [] for cfg in BANDS}
    for i in range(0, len(g), 1024):
        gg = g[i:i + 1024]; blk = np.concatenate([X[c][0][gg[gg < N0[c]]], X[c][1][gg[gg >= N0[c]] - N0[c]]])   # g sorted -> order preserved
        for cfg, m in models.items():
            r = np.where((fax >= BANDS[cfg][0]) & (fax <= BANDS[cfg][1]))[0]
            with torch.no_grad():
                x = torch.from_numpy(blk[:, :, r[0]:r[-1] + 1].astype(np.float32)).to(dev); mn = x.amin((2, 3), keepdim=True); mx = x.amax((2, 3), keepdim=True)
                out[cfg].append(torch.sigmoid(m((x - mn) / (mx - mn + 1e-9)) / m.temperature).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
gl = {c: np.sort(np.where(man[f"role_{c}"] == "lock")[0]) for c in ("pos", "neg")}
L = {c: score_rows(c, gl[c]) for c in ("pos", "neg")}; mpL = {k: v[gl["pos"]] for k, v in MC["pos"].items()}; mnL = {k: v[gl["neg"]] for k, v in MC["neg"].items()}
np.savez(f"{MD}/lockbox_scores_s0.npz", g_pos=gl["pos"], g_neg=gl["neg"], **{f"{c}_{k}": v for c in L for k, v in L[c].items()})
# ---- thresholds from the TEST set only ----
T = {cfg: np.load(f"{MD}/spec_{cfg}_single_s0_test.npz") for cfg in BANDS}; mpT = {k: v[T["s2"]["g_pos"]] for k, v in MC["pos"].items()}; mnT = {k: v[T["s2"]["g_neg"]] for k, v in MC["neg"].items()}
def gates(Lp, Ln, mp, mn):
    return {"S0 (June)": (np.maximum(mp["s0_hm"], mp["s0_lm"]), np.maximum(mn["s0_hm"], mn["s0_lm"])), "S1 max(hm,lm)": (np.maximum(Lp["hm"], Lp["lm"]), np.maximum(Ln["hm"], Ln["lm"])),
            "S2 pair": (Lp["s2"], Ln["s2"]), "S3 hm alone": (Lp["hm"], Ln["hm"])}
GT = gates({k: T[k]["p_pos"] for k in BANDS}, {k: T[k]["p_neg"] for k in BANDS}, mpT, mnT); wT = weights(mnT)
ref = float((wT * (GT["S0 (June)"][1] >= 0.5)).sum()); TH = {"S0 (June)": {"eff90": float(np.percentile(GT["S0 (June)"][0], 10)), "eqbg": 0.5}}
for nm, (sp, sn) in GT.items():
    if nm == "S0 (June)": continue
    grid = np.unique(np.percentile(sp, np.linspace(0, 30, 1201))); TH[nm] = {"eff90": float(np.percentile(sp, 10)), "eqbg": float(min(grid, key=lambda x: abs((wT * (sn >= x)).sum() - ref)))}
# ---- lockbox metrics ----
GL = gates(L["pos"], L["neg"], mpL, mnL); wL = weights(mnL)
def roc(sp, sn, wn):
    th = np.unique(np.concatenate([sp, sn]))[::-1]; th = th[np.linspace(0, len(th) - 1, min(len(th), 4000)).astype(int)]
    ps = np.sort(sp); o = np.argsort(sn); cw = np.concatenate([[0], np.cumsum(wn[o])])
    return np.concatenate([[0], cw[-1] - cw[np.searchsorted(sn[o], th, side="left")], [1]]), np.concatenate([[0], 1 - np.searchsorted(ps, th, side="left") / len(ps), [1]])
def auc(f, e, fmax=1.0):
    m = f <= fmax; return float(np.trapezoid(np.append(e[m], np.interp(fmax, f, e)), np.append(f[m], fmax)) / fmax)
def fpr_at(f, e, x): return float(f[min(np.searchsorted(e, x), len(f) - 1)])
RUNS = ("o3a", "o3b", "o4a", "o4b"); SRCS = ("campaign_pass", "repair", "campaign_fail")
out = [f"LOCKBOX read-out (opened once, PREREG A5; seed-0 single-split nets): {len(gl['pos'])} injection pairs, {len(gl['neg'])} time-slide pairs ("
       + ", ".join(f"{s} {int((mnL['source']==s).sum())}" for s in SRCS) + "); background production-weighted",
       f"  {'gate':<15} {'AUC':>6} {'pAUC<1%':>8} | eff @ FPR 0.1% 0.5% 1% 2% | FPR @ eff 85% 90% 95%"]
for nm, (sp, sn) in GL.items():
    f, e = roc(sp, sn, wL)
    out.append(f"  {nm:<15} {auc(f, e):6.4f} {auc(f, e, 0.01):8.4f} | " + " ".join(f"{100*float(np.interp(x, f, e)):5.1f}" for x in (0.001, 0.005, 0.01, 0.02))
               + " | " + " ".join(f"{100*fpr_at(f, e, x):6.3f}" for x in (0.85, 0.90, 0.95)))
for op, lab in (("eff90", "(i) 90% TEST efficiency"), ("eqbg", "(ii) equal weighted bg to S0 @ 0.5 on TEST")):
    out += ["", f"Operating point {lab} (thresholds fixed from the test set)"]
    for nm, (sp, sn) in GL.items():
        t = TH[nm][op]; ok_p = sp >= t; ok_n = sn >= t
        out.append(f"  {nm:<15} thr {t:.4f} | lockbox eff {100*ok_p.mean():5.2f}% | weighted bg {100*float((wL*ok_n).sum()):6.3f}% | "
                   + " ".join(f"{s}:{100*ok_n[mnL['source']==s].mean():.2f}%" for s in SRCS)
                   + " | eff by run " + " ".join(f"{r}:{100*ok_p[mpL['run']==r].mean():.1f}" for r in RUNS)
                   + " | eff Mtot>=300 " + f"{100*ok_p[mpL['mtot']>=300].mean():.1f}")
out.append(f"\n  S0 @ 0.5 on the lockbox: eff {100*(GL['S0 (June)'][0]>=0.5).mean():.2f}%, weighted bg {100*float((wL*(GL['S0 (June)'][1]>=0.5)).sum()):.3f}%")
txt = "\n".join(out); print(txt); open(f"{MD}/lockbox_readout_s0.txt", "w").write(txt + "\n")
