"""Pilot read-out (PREREG_SPECIALISTS_20260923 §6, offline part): S0 vs S1 vs S2 on the SAME held-out fold pairs.
Gates: S0 = max(June HM, June LM) (stored s0_hm/s0_lm, same crops) | S1 = max(p_hm, p_lm) | S2 = p_s2.
Threshold = t99 of each gate on the held-out positives (1% signal loss); S0 also at its production 0.5.
Reports per training-set size: bg pass (all / per run / per source: campaign_pass = as-run S0-passing pairs, repair = A3
re-pairs, campaign_fail), reduction vs S0 at t99, signal pass per run and per Mtot bin (§4c needs >= 99% everywhere).
Amendment A4: rule 1 = matched efficiency (new gate at S0 @ 0.5's signal efficiency): bg >= 30% lower, no run worse;
rule 2 = at its own t99 the new gate passes no more bg than S0 @ 0.5 (aggregate and per run); rule 3 = bg-vs-eff table.
Not covered here (tiles not rendered): watchlist / 67 catalogue detections / counted FAR families / top-50 FAR pairs.
SPEC_SPLIT=single: read the single-split test set instead (train_single.py outputs in models_single; lockbox untouched).
Env: SPEC_DATA, SPEC_OUT (models dir), SPEC_FOLD (training fold, default 0), SPEC_SEED, SPEC_SPLIT."""
import os, json, glob, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = os.environ.get("SPEC_DATA", MADGRAV_SCRATCH + "/madgrav2_spec/train"); MD = os.environ.get("SPEC_OUT", MADGRAV_SCRATCH + "/madgrav2_spec/models")
FOLD = int(os.environ.get("SPEC_FOLD", "0")); SEED = int(os.environ.get("SPEC_SEED", "0")); HO = 1 - FOLD
SINGLE = os.environ.get("SPEC_SPLIT") == "single"; TAGS = ("nfull",) if SINGLE else ("n25k", "n50k", "n100k", "nfull")
if SINGLE:
    MD = os.environ.get("SPEC_OUT", MADGRAV_SCRATCH + "/madgrav2_spec/models_single"); gz = [np.load(f) for f in sorted(glob.glob(f"{MD}/spec_*_single_s{SEED}_test.npz"))]
    assert gz and all(np.array_equal(z["g_pos"], gz[0]["g_pos"]) and np.array_equal(z["g_neg"], gz[0]["g_neg"]) for z in gz), "test sets differ between configs"
    def cat(c): m0, m1 = (dict(np.load(f"{D}/{c}_f{f}_meta.npz", allow_pickle=True)) for f in (0, 1)); return {k: np.concatenate([m0[k], m1[k]]) for k in m0}
    mp = {k: v[gz[0]["g_pos"]] for k, v in cat("pos").items()}; mn = {k: v[gz[0]["g_neg"]] for k, v in cat("neg").items()}
else:
    mp = dict(np.load(f"{D}/pos_f{HO}_meta.npz", allow_pickle=True)); mn = dict(np.load(f"{D}/neg_f{HO}_meta.npz", allow_pickle=True))
MB = [(10, 30), (30, 100), (100, 300), (300, 800.1)]; RUNS = sorted(set(mn["run"]) | set(mp["run"])); SRCS = ("campaign_pass", "repair", "campaign_fail")
def heldout(cfg, tag):
    f = f"{MD}/spec_{cfg}_single_s{SEED}_test.npz" if SINGLE else f"{MD}/spec_{cfg}_f{FOLD}_s{SEED}_{tag}_heldout.npz"
    return np.load(f) if os.path.exists(f) else None
def report(name, sp, sn, t, ref=None):
    L = [f"  {name:<18} t={t:.4f} | bg pass {100*(sn>=t).mean():6.3f}%" + (f" ({100*(1-(sn>=t).mean()/ref):+.0f}% vs S0)" if ref else "")
         + " | " + " ".join(f"{s}:{100*(sn[mn['source']==s]>=t).mean():.2f}%" for s in SRCS)]
    L.append("  " + " " * 18 + " bg by run   " + " ".join(f"{r}:{100*(sn[mn['run']==r]>=t).mean():.2f}%" for r in RUNS))
    L.append("  " + " " * 18 + " sig by run  " + " ".join(f"{r}:{100*(sp[mp['run']==r]>=t).mean():.2f}%" for r in RUNS)
             + " | by Mtot " + " ".join(f"{a}-{int(b)}:{100*(sp[(mp['mtot']>=a)&(mp['mtot']<b)]>=t).mean():.2f}%" for a, b in MB))
    return (sn >= t).mean(), L
out = [(f"Specialist SINGLE-SPLIT read-out (lockbox sealed), seed {SEED}, TEST set:" if SINGLE else f"Specialist pilot read-out: train fold {FOLD}, seed {SEED}, held-out fold {HO}:") + f" pos {len(mp['run'])}, neg {len(mn['run'])} ("
       + ", ".join(f"{s} {int((mn['source']==s).sum())}" for s in SRCS) + ")"]
s0p = np.maximum(mp["s0_hm"], mp["s0_lm"]); s0n = np.maximum(mn["s0_hm"], mn["s0_lm"]); t0 = float(np.percentile(s0p, 1.0))
ref, L = report("S0 @ t99", s0p, s0n, t0); out += L; _, L = report("S0 @ 0.5 (prod)", s0p, s0n, 0.5); out += L
REACH = {"o3a": (12775, 602203), "o3b": (4899, 284755), "o4a": (1828, 360925), "o4b": (1742, 142740)}   # A4 addendum: as-run (S0 pass, reach specialists), negsel_1760649
def bgrun(sn, t):                                   # production-weighted bg pass per run (strata: campaign S0-pass / S0-fail)
    r_ = {}
    for r, (npass, nreach) in REACH.items():
        mp_ = (mn["run"] == r) & (mn["source"] == "campaign_pass"); mf_ = (mn["run"] == r) & (mn["source"] == "campaign_fail")
        if mp_.any() and mf_.any(): fp = npass / nreach; r_[r] = fp * float((sn[mp_] >= t).mean()) + (1 - fp) * float((sn[mf_] >= t).mean())
    return r_
def wbg(sn, t):
    br = bgrun(sn, t); w = {r: REACH[r][1] for r in br}; return sum(w[r] * br[r] for r in br) / sum(w.values()), br
eff0 = float((s0p >= 0.5).mean()); ref05 = wbg(s0n, 0.5)[0]
out.append(f"  (S0 @ 0.5 keeps {100*eff0:.2f}% of held-out positives; matched-eff rows use that efficiency)")
summary = {"S0": dict(t99=t0, bg_pass=float(ref), eff_at_05=eff0, bg_pass_at_05=ref05)}
S0RUN = bgrun(s0n, 0.5); EFFS = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99); CURVES = {}
def effrow(nm, sp, sn):
    CURVES[nm] = dict(eff=np.linspace(0.5, 0.999, 200)); CURVES[nm]["bg"] = np.array([wbg(sn, np.percentile(sp, 100 * (1 - e)))[0] for e in CURVES[nm]["eff"]])
    return f"  {nm:<18} " + " ".join(f"{100*wbg(sn, np.percentile(sp, 100*(1-e)))[0]:7.3f}" for e in EFFS)
def matched(nm, sp, sn):
    t = float(np.percentile(sp, 100 * (1 - eff0))); b, br = wbg(sn, t); t99 = float(np.percentile(sp, 1.0)); b99, br99 = wbg(sn, t99)
    r1 = (b <= 0.7 * ref05) and all(br[r] <= S0RUN[r] for r in br); r2 = (b99 <= ref05) and all(br99[r] <= S0RUN[r] for r in br99)
    out.append(f"  {nm:<18} t={t:.4f} | weighted bg {100*b:6.3f}% ({100*(1-b/ref05):+.0f}% vs S0 @ 0.5 = {100*ref05:.3f}%) | as drawn " + " ".join(f"{s_}:{100*(sn[mn['source']==s_]>=t).mean():.2f}%" for s_ in SRCS))
    out.append("  " + " " * 18 + " weighted by run " + " ".join(f"{r}:{100*br[r]:.3f}% (S0 {100*S0RUN[r]:.3f}%)" for r in br))
    out.append(f"  {'':<18} A4 rule1 (>=30% lower, no run worse) {'PASS' if r1 else 'FAIL'} | rule2 (t99 weighted bg {100*b99:.3f}% <= S0@0.5 {100*ref05:.3f}%, per run) {'PASS' if r2 else 'FAIL'}"
               + ("" if r2 else " -> per run t99: " + " ".join(f"{r}:{100*br99[r]:.3f}%" for r in br99)))
    out.append(f"  {'':<18} hard class at matched eff: repair pass {100*(sn[mn['source']=='repair']>=t).mean():.2f}% (S0 @ 0.5 {100*(s0n[mn['source']=='repair']>=0.5).mean():.2f}%)")
    return dict(bg=b, bg_run=br, rule1=bool(r1), rule2=bool(r2), bg_t99=b99)
for tag in TAGS:
    h, l, s2 = heldout("hm", tag), heldout("lm", tag), heldout("s2", tag); out.append(f"-- training size {tag}")
    if h is not None and l is not None:
        sp, sn = np.maximum(h["p_pos"], l["p_pos"]), np.maximum(h["p_neg"], l["p_neg"]); t = float(np.percentile(sp, 1.0))
        bp, L = report("S1 max(hm,lm)", sp, sn, t, ref); out += L; summary[f"S1_{tag}"] = dict(t99=t, bg_pass=float(bp), A4=matched("S1 matched-eff", sp, sn))
        for nm, z in (("  hm alone", h), ("  lm alone", l)):
            t_ = float(np.percentile(z["p_pos"], 1.0)); out.append(f"  {nm:<18} t={t_:.4f} | bg pass {100*(z['p_neg']>=t_).mean():6.3f}%")
    if s2 is not None:
        t = float(np.percentile(s2["p_pos"], 1.0)); bp, L = report("S2 pair 20-500", s2["p_pos"], s2["p_neg"], t, ref); out += L; summary[f"S2_{tag}"] = dict(t99=t, bg_pass=float(bp), A4=matched("S2 matched-eff", s2["p_pos"], s2["p_neg"]))
out.append("-- A4 rule 3: held-out WEIGHTED bg pass (%) at signal efficiency " + " ".join(f"{e:7.2f}" for e in EFFS))
out.append(effrow("S0", s0p, s0n))
for tag in TAGS:
    h, l, s2 = heldout("hm", tag), heldout("lm", tag), heldout("s2", tag)
    if h is not None and l is not None: out.append(effrow(f"S1 {tag}", np.maximum(h["p_pos"], l["p_pos"]), np.maximum(h["p_neg"], l["p_neg"])))
    if s2 is not None: out.append(effrow(f"S2 {tag}", s2["p_pos"], s2["p_neg"]))
SFX = f"single_s{SEED}" if SINGLE else f"f{FOLD}_s{SEED}"
np.savez(f"{MD}/pilot_curves_{SFX}.npz", **{f"{k}__{kk}": vv for k, v in CURVES.items() for kk, vv in v.items()})
txt = "\n".join(out); print(txt)
open(f"{MD}/pilot_readout_{SFX}.txt", "w").write(txt + "\n"); json.dump(summary, open(f"{MD}/pilot_readout_{SFX}.json", "w"), indent=1)
