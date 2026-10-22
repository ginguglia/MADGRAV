"""MADGRAV2.1 gx validation, step 2 (CPU, seconds): given two LR models fitted on the same run (with gx: SM_GX=1 refit; without: SM_GX=0 refit),
re-rank the banked background survivors + zero-lag triggers (gx_rows_from_shards.py output) and the run's injections under both, and compare
injection efficiency at matched loudest-background thresholds (per-pair rows, after the arm rule). Prints the gate table.
Env: ROWS (npz), INJ_GLOB, LR_GX, LR_NOGX, ARM_RULE_THR (-2.05), NETMIN (3), COH_GUARD_HZ/FLOOR (as the campaign: 50/0.115), GX_CAP (15)."""
import os, glob, json, numpy as np
R = np.load(os.environ["ROWS"]); LRG = np.load(os.environ["LR_GX"]); LRN = np.load(os.environ["LR_NOGX"])
INJ = sorted(glob.glob(os.environ["INJ_GLOB"])); THR = float(os.environ.get("ARM_RULE_THR", "-2.05")); NETMIN = float(os.environ.get("NETMIN", "3.0"))
GHZ = float(os.environ.get("COH_GUARD_HZ", "50")); GFL = float(os.environ.get("COH_GUARD_FLOOR", "0.115")); CAP = float(os.environ.get("GX_CAP", "15"))
def gate(g, s): return np.clip(g, -6, 6) * np.clip(np.asarray(s) / 3.0, 0, 1)
def guard(coh, cH, cL): m = (cH < GHZ) & (cL < GHZ); return np.where(m, np.minimum(coh, GFL), coh)
def feats(sH, sL, coh, cH, cL, gH, gL, gx, with_gx):
    cols = [sH, sL, guard(coh, cH, cL), cH, cL, gate(gH, sH), gate(gL, sL)] + ([np.minimum(gx, CAP)] if with_gx else [])
    return np.column_stack(cols)
def loglr(LR, fold_model, F):                        # fold_model = model index used to SCORE (1 - fold of the data)
    out = np.empty(len(F))
    for g in (0, 1):
        m = fold_model == g; mu, sd, be = LR[f"mu{g}"], LR[f"sd{g}"], LR[f"be{g}"]; out[m] = be[0] + ((F[m] - mu) / sd) @ be[1:]
    return out
# ---- background + triggers ----
kind = R["kind"]; fold = R["fold"]; sH, sL, cH, cL, gH, gL, coh, gx = (R[k] for k in ("sH", "sL", "cH", "cL", "gH", "gL", "coh", "gx"))
arm_ok = np.where(sH >= sL, gH, gL) >= THR
Fg = feats(sH, sL, coh, cH, cL, gH, gL, gx, True); Fn = feats(sH, sL, coh, cH, cL, gH, gL, gx, False)
llg = loglr(LRG, 1 - fold, Fg); lln = loglr(LRN, 1 - fold, Fn)
bg = (kind == 0) & arm_ok; zl = kind == 1
print(f"rows: {int((kind==0).sum())} bg ({int(bg.sum())} pass arm rule max:{THR}), {int(zl.sum())} zero-lag triggers; gx params {R['gx_params']}")
print(f"bg gx: median {np.median(gx[kind==0]):.2f} p99 {np.percentile(gx[kind==0],99):.2f} p99.9 {np.percentile(gx[kind==0],99.9):.2f} max {gx[kind==0].max():.1f}; frac >= CAP {np.mean(gx[kind==0]>=CAP):.4f}")
for lab, LR in (("gx", LRG), ("no-gx", LRN)):
    for g in (0, 1):
        be = LR[f"be{g}"]; mu = LR[f"mu{g}"]; sd = LR[f"sd{g}"]
        extra = f" beta_gx {be[8]:+.3f} -> max gain vs noise-median gx {be[8]*(CAP-mu[7])/sd[7]:.2f} loglr" if len(be) == 9 else ""
        print(f"  LR[{lab}] fold {g}: beta_coh {be[3]:+.3f}{extra}")
def tail(ll, m, n=(1, 3, 10, 30, 100, 300, 1000)):
    s = np.sort(ll[m])[::-1]; return {k: (s[k - 1] if k <= len(s) else -np.inf) for k in n}
tg = tail(llg, bg); tn = tail(lln, bg)
print("\nloudest background rows (post arm rule), loglr under each model:")
print("   n      no-gx      gx"); [print(f"{k:4d}   {tn[k]:7.2f}  {tg[k]:7.2f}") for k in tg]
# ---- injections ----
X = {}; 
for f in INJ:
    z = np.load(f); k = (z["net"] >= NETMIN)
    for key in ("sigH", "sigL", "coh", "cenH2", "cenL2", "gH2", "gL2", "gx", "net_snr", "mtot", "net"): X.setdefault(key, []).append(z[key][k])
X = {k: np.concatenate(v) for k, v in X.items()}
from importlib import import_module
import sys; sys.path.insert(0, os.path.join(os.environ["MADGRAV_ROOT"], "search_mode")); C2 = import_module("centroid_v2")
gk = C2.cen2_pass(X["cenH2"], X["cenL2"]) & (np.where(X["sigH"] >= X["sigL"], X["gH2"], X["gL2"]) >= THR)
for k in X: X[k] = X[k][gk]
rng = np.random.default_rng(1); fi = rng.integers(0, 2, len(X["net"]))                      # injections have no fold here: random model, both models see both
Fig = feats(X["sigH"], X["sigL"], X["coh"], X["cenH2"], X["cenL2"], X["gH2"], X["gL2"], X["gx"], True); Fin = feats(X["sigH"], X["sigL"], X["coh"], X["cenH2"], X["cenL2"], X["gH2"], X["gL2"], X["gx"], False)
lig = loglr(LRG, fi, Fig); lin = loglr(LRN, fi, Fin)
print(f"\ninjections: {len(lig)} gate-passing triggering rows (net>={NETMIN}); gx median {np.median(X['gx']):.2f}; SNR grid {sorted(set(X['net_snr'].round(1)))}")
print("injection efficiency above the n-th loudest background row (all injections | net SNR 8 | 10 | 12 | 15 | mtot<30 | 30-100 | >100):")
strata = [("all", np.ones(len(lig), bool)), ("snr8", X["net_snr"] == 8), ("snr10", X["net_snr"] == 10), ("snr12", X["net_snr"] == 12), ("snr15", X["net_snr"] == 15),
          ("mt<30", X["mtot"] < 30), ("mt30-100", (X["mtot"] >= 30) & (X["mtot"] < 100)), ("mt>100", X["mtot"] >= 100)]
print("   n   model  " + "  ".join(f"{s[0]:>8s}" for s in strata))
for k in tg:
    for lab, thr, ll in (("no-gx", tn[k], lin), ("gx", tg[k], lig)):
        print(f"{k:4d}   {lab:6s} " + "  ".join(f"{np.mean(ll[m] > thr):8.3f}" for _, m in strata))
print(f"floor-3 pass: no-gx {np.mean(lin>=3):.3f}  gx {np.mean(lig>=3):.3f}")
# ---- zero-lag triggers: rank vs background under each model ----
if zl.any():
    order = np.argsort(-llg[zl]); gps = R["gpsH"][zl]; segn = R["ai"][zl]
    print("\nzero-lag triggers (pilot list): loglr and number of louder post-rule bg rows (same fold), old prune loglr | no-gx | gx")
    print(f"{'gps':>12s} {'fold':>4s} {'old':>7s} {'no-gx':>7s} {'n>':>7s} {'gx':>7s} {'n>':>7s}  {'gxval':>6s}")
    for j in order[:40]:
        i = np.where(zl)[0][j]; f_ = fold[i]; mb = bg & (fold == f_)
        print(f"{gps[j]:12.1f} {f_:4d} {R['loglr_old'][i]:7.2f} {lln[i]:7.2f} {int((lln[mb]>lln[i]).sum()):7d} {llg[i]:7.2f} {int((llg[mb]>llg[i]).sum()):7d}  {gx[i]:6.2f}")
