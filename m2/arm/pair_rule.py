"""How should the arm enter MADGRAV2? Compare pair-level rules on the same cross-fit arm ensemble (arm_def, 20 models):
  both   : both legs P(sig) >= t (per-leg veto, current scorer)
  ramp   : production-style: score = sum_i clip(sigma_i/3,0,1) * logit(P_i) over the two legs (faint legs count less)
  mean   : mean logit over the two legs
  max    : louder-leg logit (leg with larger sigma)
Threshold for each rule = 1st percentile of the rule's score over TRIGGERING injection pairs (sigma_net>=4, m2/positives,
scored cross-fit by day). Report: injection loss (=1% by construction), 532 families removed, watchlist (48 events, all legs
scored with the fold that never saw their day), and where GW230707 lands. CPU; 6000 injection pairs subsample."""
import os, sys, glob, json, re, hashlib, collections, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_model.py")).read())
from gwpy.time import to_gps
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

torch.set_num_threads(8); M = MADGRAV_SCRATCH + "/madgrav2/arm/models"; CEN = MADGRAV_SCRATCH + "/madgrav2/centroid"; P = os.environ.get("PR_POS", MADGRAV_SCRATCH + "/madgrav2/positives")   # 2026-09-28: PR_POS = positives dir of the arm under test
TAG = os.environ.get("ARM_TAG", "_def"); NSUB = int(os.environ.get("PR_NSUB", "6000")); rng = np.random.default_rng(3)
def fold_of(day): return int(hashlib.sha1(str(int(day)).encode()).hexdigest(), 16) % 2
models = {0: [], 1: []}
for js in sorted(glob.glob(f"{M}/arm{TAG}_f*_s*.json")):
    S = json.load(open(js)); m = Arm(); m.load_state_dict(torch.load(js.replace(".json", ".pt"), map_location="cpu")); m.eval(); models[S["fold"]].append(m)
print("models:", {f: len(v) for f, v in models.items()}, flush=True)
def logit_sig(fold_of_leg, tiles):
    x = torch.from_numpy(np.asarray(tiles, np.float32)[:, None]); out = []
    with torch.no_grad():
        for m in models[1 - fold_of_leg]:
            p = torch.cat([m.psig(x[i:i + 256]) for i in range(0, len(x), 256)]).numpy().astype(np.float64); out.append(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)))
    return np.mean(out, 0)
def rules(lH, lL, sH, sL):
    wH, wL = np.clip(sH / 3.0, 0, 1), np.clip(sL / 3.0, 0, 1)
    return {"both": np.minimum(lH, lL), "ramp": wH * lH + wL * lL, "mean": 0.5 * (lH + lL), "max": np.where(sH >= sL, lH, lL)}
# --- injections: triggering pairs, cross-fit by the H1/L1 window day ---
fs = sorted(f for f in glob.glob(f"{P}/pos_*_shard*.npz") if "smoke" not in f); qh, ql, sh, sl, dh, dl = [], [], [], [], [], []
per = max(1, NSUB // len(fs))
for f in fs:
    z = np.load(f); trig = np.where(z["pass_net"] == 1)[0]; pick = np.sort(rng.choice(trig, size=min(per, len(trig)), replace=False))
    qh.append(np.asarray(z["qt_H"])[pick]); ql.append(np.asarray(z["qt_L"])[pick]); sh.append(z["sigma_H"][pick]); sl.append(z["sigma_L"][pick]); dh.append(z["day_H"][pick]); dl.append(z["day_L"][pick])
qh, ql, sh, sl, dh, dl = map(np.concatenate, (qh, ql, sh, sl, dh, dl)); n = len(sh); print(f"injection pairs: {n}", flush=True)
lH = np.empty(n); lL = np.empty(n)
for f in (0, 1):
    mh = np.array([fold_of(d) for d in dh]) == f; ml = np.array([fold_of(d) for d in dl]) == f
    if mh.any(): lH[mh] = logit_sig(f, qh[mh])
    if ml.any(): lL[ml] = logit_sig(f, ql[ml])
R_inj = rules(lH, lL, sh, sl); thr = {k: float(np.percentile(v, 1.0)) for k, v in R_inj.items()}
print("thresholds (p1 of triggering injection pairs):", {k: round(v, 3) for k, v in thr.items()}, flush=True)
# --- families ---
ft = np.load(f"{CEN}/fam532_tiles.npz", allow_pickle=True); fm = np.load(f"{CEN}/feat_fam532.npz", allow_pickle=True)
day = (fm["gps"] // 86400).astype(int); folds = np.array([fold_of(d) for d in day]); L = np.empty(len(day))
for f in (0, 1):
    m = folds == f; L[m] = logit_sig(f, ft["qt"][m])
fam = collections.OrderedDict()
for i, k in enumerate(fm["fam"]): fam.setdefault(k, {"run": fm["run"][i], "out": str(fm["outranks"][i])})[fm["det"][i]] = (L[i], float(fm["sigma"][i]))
keys = [k for k, v in fam.items() if "H1" in v and "L1" in v]
FR = rules(np.array([fam[k]["H1"][0] for k in keys]), np.array([fam[k]["L1"][0] for k in keys]), np.array([fam[k]["H1"][1] for k in keys]), np.array([fam[k]["L1"][1] for k in keys]))
# --- catalog events (48 adopted incl GW190521 via det tiles; use cat_tiles for all 138) ---
ct = np.load(f"{CEN}/cat_tiles.npz", allow_pickle=True); cf = np.load(f"{CEN}/feat_cat.npz", allow_pickle=True)
def day_of_name(nm): m = re.match(r"GW(\d{2})(\d{2})(\d{2})", nm); return int(float(to_gps(f"20{m[1]}-{m[2]}-{m[3]}")) // 86400)
cday = np.array([day_of_name(nm) for nm in cf["name"]]); cfold = np.array([fold_of(d) for d in cday]); CL = np.empty(len(cday))
for f in (0, 1):
    m = cfold == f; CL[m] = logit_sig(f, ct["qt"][m])
ev = collections.OrderedDict()
for i, nm in enumerate(cf["name"]): ev.setdefault(nm, {"ad": bool(cf["adopted"][i])})[cf["det"][i]] = (CL[i], float(cf["snr_det"][i]))   # snr as sigma proxy for the ramp weight
ek = [k for k, v in ev.items() if "H1" in v and "L1" in v]
ER = rules(np.array([ev[k]["H1"][0] for k in ek]), np.array([ev[k]["L1"][0] for k in ek]), np.array([ev[k]["H1"][1] for k in ek]) * 0.6, np.array([ev[k]["L1"][1] for k in ek]) * 0.6)  # sigma ~ 0.6 snr (rough)
ad = np.array([ev[k]["ad"] for k in ek])
print(f"\n| rule | thr | inj loss | families removed /532 | adopted pass /{ad.sum()} | missed pass /{(~ad).sum()} | failing adopted |"); print("|---|---|---|---|---|---|---|")
for k in ("both", "ramp", "mean", "max"):
    t = thr[k]; rem = int((FR[k] < t).sum()); ap = int((ER[k][ad] >= t).sum()); mp = int((ER[k][~ad] >= t).sum())
    bad = [ek[i] for i in np.where(ad & (ER[k] < t))[0]]
    print(f"| {k} | {t:.2f} | {100*np.mean(R_inj[k] < t):.1f}% | {rem} | {ap} | {mp} | {', '.join(bad) if bad else '-'} |")
i = ek.index("GW230707_124047") if "GW230707_124047" in ek else None
if i is not None: print(f"\nGW230707_124047: logit H1 {ER['both'][i]:.2f} (min leg), ramp {ER['ramp'][i]:.2f}, mean {ER['mean'][i]:.2f}, max {ER['max'][i]:.2f}; thresholds {json.dumps({k: round(v,2) for k,v in thr.items()})}")
# margin of the weakest adopted event per rule
for k in ("both", "ramp", "mean", "max"):
    o = np.argsort(ER[k][ad])[:3]; print(f"{k}: weakest adopted " + ", ".join(f"{np.array(ek)[ad][j]} {ER[k][ad][j]:.2f}" for j in o) + f" (thr {thr[k]:.2f})")
