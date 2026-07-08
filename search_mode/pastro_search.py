#!/usr/bin/env python
"""Search-mode p_astro (FGMC) for the BLIND-search detections — FIRST CUT.

p_s : recovered injections (SM_INJ/*_inj.npz) scored to the BLIND loglr via the frozen LR model
      (feats -> loglr, byte-identical to driver_blindscan:187).
p_n : time-slide background loglr from the run.
events: detections.json loglr. Poisson-mixture MLE (Ls,Ln) -> per-detection p_astro.

SMOKE-CUT SIMPLIFICATIONS (flagged; to harden for production per the review panel):
  * single-fold injection scoring (fold-0 frozen model) instead of per-fold + fold bookkeeping;
  * p_s POOLED over the mass population (blind detections carry no mtot) -- population reweighting TODO;
  * p_n from survivors_bg.json (CNN-gated/floor-truncated) instead of the FULL time-slide background;
  * no f_HM systematic band, no segment bootstrap interval, no PP-calibration, no detectability floor x0.
Output p_astro is therefore PROVISIONAL (mechanism validation), not the calibrated headline.

Run: MADGRAV_ROOT=.. SM_OUT=<run dir> SM_INJ=<inj dir> SM_LR_MODEL=<frozen npz> python pastro_search.py
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

OUT = os.environ.get("SM_OUT", os.path.join(os.environ.get("SCRATCH", "."), "search_out"))
INJ = os.environ.get("SM_INJ", os.path.join(os.environ.get("SCRATCH", "."), "inj_out"))
LRM = os.environ["SM_LR_MODEL"]
GCLIP = 6.0

def gate(g, s): return np.clip(g, -GCLIP, GCLIP) * np.clip(np.asarray(s, float) / 3.0, 0, 1)
def feats(sH, sL, coh, cH, cL, gH, gL):
    return np.column_stack([sH, sL, coh, cH, cL, gate(gH, sH), gate(gL, sL)])
def loglr(mu, sd, beta, F): return beta[0] + ((np.asarray(F, float) - mu) / sd) @ beta[1:]

m = np.load(LRM)
mu, sd, be = m["mu0"], m["sd0"], m["be0"]            # SMOKE-CUT: fold-0 model for all
floor = float(m["floor"]) if "floor" in m.files else 4.5

# ---- p_s : injections -> blind loglr, gated to the FGMC domain (loglr >= floor) ----
xs = []
for f in sorted(glob.glob(f"{INJ}/*_inj.npz")):
    z = np.load(f)
    F = feats(z["sigH"], z["sigL"], z["coh"], z["cenH"], z["cenL"], z["gH"], z["gL"])
    xs.append(loglr(mu, sd, be, F))
xs = np.concatenate(xs)
xs_g = xs[np.isfinite(xs) & (xs >= floor)]
if len(xs_g) < 10:
    print(f"[pastro] too few injections above floor ({len(xs_g)}) -> abort"); sys.exit(0)
ps_kde = gaussian_kde(xs_g)

# ---- p_n : background loglr (SMOKE-CUT: survivors_bg, truncated) ----
_surv = json.load(open(f"{OUT}/survivors_bg.json"))["survivors"]
sbg = np.array([s["loglr"] for s in _surv if isinstance(s.get("loglr"), (int, float))], dtype=float)
sbg_g = sbg[np.isfinite(sbg) & (sbg >= floor)]
pn_kde = gaussian_kde(sbg_g) if len(sbg_g) >= 10 else (lambda x: np.full(np.shape(x), 1.0 / max(1, len(sbg_g))))

# ---- events + FGMC Poisson-mixture fit ----
dets = json.load(open(f"{OUT}/detections.json"))
xe = np.array([d.get("loglr", np.nan) for d in dets])
psv = np.maximum(ps_kde(xe), 1e-300)
pnv = np.maximum(pn_kde(xe) if callable(pn_kde) else pn_kde(xe), 1e-300)
psa, pna = psv, pnv
def nll(th):
    Ls, Ln = np.exp(th)
    return (Ls + Ln) - np.sum(np.log(Ls * psa + Ln * pna))
r = minimize(nll, np.log([max(1.0, 0.3 * len(dets)), max(1.0, 0.7 * len(dets))]), method="Nelder-Mead")
Ls, Ln = np.exp(r.x)
pastro = Ls * psv / (Ls * psv + Ln * pnv)

for d, pa, x in zip(dets, pastro, xe):
    d["p_astro"] = float(pa) if np.isfinite(x) else None
    d["p_astro_provisional"] = True
json.dump(dets, open(f"{OUT}/detections_pastro.json", "w"), indent=2)
print(f"[pastro] {len(dets)} dets | p_s inj={len(xs_g)} p_n bg={len(sbg_g)} floor={floor:.2f} "
      f"| Ls={Ls:.2f} Ln={Ln:.2f}")
for d in dets:
    print(f"    {str(d.get('matches_known') or d.get('seg')):22} loglr={d.get('loglr',0):6.2f} "
          f"best_far={d.get('best_far')}  p_astro(prov)={d.get('p_astro')}")
print(f"[pastro] -> {OUT}/detections_pastro.json  (PROVISIONAL)")
