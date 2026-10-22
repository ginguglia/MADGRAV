#!/usr/bin/env python
"""p_astro for the MADGRAV 2.2 detections (2026-10-06): the paper's self-calibrated Poisson-mixture (FGMC) fit on the
best_far axis, ported 1:1 from MADGRAV/search_mode/pastro_final/pastro_final.py with the 2.2 inputs.

Per run r in {o3a,o3b,o4a,o4b}:
  candidates  : search_out_<r>_m2_gx_off5/detections<VARIANT>.json, axis x_i = best_far (FAR < 1/yr rows; no K factor in 2.2)
  noise       : p_n uniform on [0,1]/yr over FAR_EDGES; L_n PINNED to 1/yr x T_fg, T_fg = scanned coincident time x 0.5 (ANALYZED_FRAC,
                the FAR-denominator convention of the caches; identical to the paper's T_fg since the segments are the same)
  signal      : p_s = histogram of best_far of the DETECTED injections of the 2.2 campaign (vt22/inj_scored_<r>_m22.npz, each
                injection scored with its own CNN pair, so no pair multiset), weights w0 (rho^-4 d rho, Euclidean, as in the paper's
                adopted product) x Mtot reweighting to the published-catalogue Mtot histogram (MASS_EDGES, merged_plot_v2.csv);
                det rule = best_far < 1 (SM_DET_RULE=far, the adopted paper rule; 'and' adds best_ul90 < 1)
  fit         : L_s by Nelder-Mead MLE with L_n pinned; p_astro_i = L_s p_s(x_i) / (L_s p_s(x_i) + L_n p_n(x_i)) (binned densities)
  sensitivity : free (L_s, L_n) fit -> p_astro_freeLn
  uncertainty : 200 bootstrap refits with Dirichlet injection weights (seed 20260811) -> 5-95 % (p_astro_lo/hi)
Out: m2/pastro22/pastro22<VARIANT>.{csv,json}. Env: SM_DET_VARIANT (default _s3rank), SM_DET_RULE (far|and).
"""
import csv, json, os, sys
import numpy as np
from scipy.optimize import minimize

import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = MADGRAV_SCRATCH + "/madgrav2"
SEGD = MADGRAV_SCRATCH
CATF = MADGRAV_SCRATCH + "/MADGRAV1/figures/catalog_o3o4/merged_plot_v2.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["o3a", "o3b", "o4a", "o4b"]
VARIANT = os.environ.get("SM_DET_VARIANT", "_s3rank")
DET_RULE = os.environ.get("SM_DET_RULE", "far")
DET_FAR = 1.0
MASS_EDGES = np.array([0., 50., 80., 120., 180., 1e9])
FAR_EDGES = np.array([0., 0.005, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
B = 200


def fit_fgmc(xs, ps, pn, widths, Ln_fixed=None):
    bi = np.clip(np.digitize(xs, FAR_EDGES) - 1, 0, len(ps) - 1)
    psv, pnv = ps[bi] / widths[bi], pn[bi] / widths[bi]
    if Ln_fixed is None:
        def nll(th):
            Ls, Ln = np.exp(th)
            return (Ls + Ln) - np.sum(np.log(Ls * psv + Ln * pnv))
        r = minimize(nll, np.log([max(0.5, 0.8 * len(xs)), max(0.5, 0.2 * len(xs))]), method="Nelder-Mead")
        Ls, Ln = np.exp(r.x)
    else:
        Ln = float(Ln_fixed)
        def nll(th):
            Ls = np.exp(th[0])
            return (Ls + Ln) - np.sum(np.log(Ls * psv + Ln * pnv))
        r = minimize(nll, [np.log(max(0.5, len(xs)))], method="Nelder-Mead")
        Ls = float(np.exp(r.x[0]))
    return Ls, Ln, Ls * psv / (Ls * psv + Ln * pnv)


def main():
    cat_mtot = np.array([float(r["total_mass_source"]) for r in csv.DictReader(open(CATF)) if r["total_mass_source"].strip()])
    f_cat = np.histogram(cat_mtot, MASS_EDGES)[0] / len(cat_mtot)
    widths = np.diff(FAR_EDGES); pn = widths / 1.0
    results = []
    for run in RUNS:
        dets = json.load(open(f"{SC}/search_out_{run}_m2_gx_off5/detections{VARIANT}.json"))
        dets = [d for d in dets if d["best_far"] < DET_FAR]
        inj = np.load(f"{SC}/vt22/inj_scored_{run}_m22.npz", allow_pickle=True)
        far = inj["best_far"].astype(float); det = np.isfinite(far) & (far < DET_FAR)
        if DET_RULE == "and":
            det &= np.isfinite(inj["best_ul90"]) & (inj["best_ul90"] < DET_FAR)
        w0 = inj["w0"].astype(float); mb = np.clip(np.digitize(inj["mtot"], MASS_EDGES) - 1, 0, len(MASS_EDGES) - 2)

        def ps_of(w):
            W = w[det]; MTb = mb[det]
            f_inj = np.bincount(MTb, weights=W, minlength=len(MASS_EDGES) - 1); f_inj /= max(f_inj.sum(), 1e-300)
            wm = np.where(f_inj > 0, f_cat / np.maximum(f_inj, 1e-12), 0.0)
            ps, _ = np.histogram(far[det], FAR_EDGES, weights=W * wm[MTb])
            ps = ps / max(ps.sum(), 1e-300); ps = np.maximum(ps, 1e-6)
            return ps / ps.sum()

        tabs = np.load(f"{SC}/vt22/far_tables_{run}.npz", allow_pickle=True)
        segj = {s[3]: s[2] for s in json.load(open(f"{SEGD}/{run}_full_coincident.json"))["segments"]}
        T_fg = sum(segj.get(str(n), 0.0) for n in tabs["seg_name"]) * 0.5 / 3.1557e7
        Ln_null = DET_FAR * T_fg
        xs = np.array([d["best_far"] for d in dets], float)
        ps = ps_of(w0)
        Ls, Ln, pa = fit_fgmc(xs, ps, pn, widths, Ln_fixed=Ln_null)
        Ls_f, Ln_f, pa_f = fit_fgmc(xs, ps, pn, widths)
        eff = float((det * w0).sum() / w0.sum())
        rng = np.random.default_rng(20260811); pa_boot = np.empty((B, len(xs)))
        for b in range(B):
            wb = w0 * rng.dirichlet(np.ones(len(w0))) * len(w0)
            pa_boot[b] = fit_fgmc(xs, ps_of(wb), pn, widths, Ln_fixed=Ln_null)[2]
        lo, hi = np.percentile(pa_boot, [5, 95], axis=0)
        print(f"[{run}] n={len(xs)} eff={eff:.3f} T_fg={T_fg:.4f}yr Ln={Ln:.3f} Ls={Ls:.2f} (free fit: Ls={Ls_f:.2f} Ln={Ln_f:.2f}) "
              f"ps={np.array2string(ps, precision=3)}", flush=True)
        for i, d in enumerate(dets):
            results.append(dict(run=run.upper()[:2] + run[2], name=d.get("matches_known") or "", seg=d["seg"], gps=d["gps"], channel=d["channel"],
                                loglr=d["loglr"], net=d["net"], best_far=float(xs[i]), best_ul90=d["best_ul90"], p_astro=float(pa[i]),
                                p_astro_lo=float(lo[i]), p_astro_hi=float(hi[i]), p_astro_freeLn=float(pa_f[i]),
                                Ls=float(Ls), Ln=float(Ln), Ls_free=float(Ls_f), Ln_free=float(Ln_f), T_fg_yr=float(T_fg)))
    tag = VARIANT + ("" if DET_RULE == "far" else "_and")
    json.dump(results, open(f"{HERE}/pastro22{tag}.json", "w"), indent=1)
    with open(f"{HERE}/pastro22{tag}.csv", "w") as fh:
        w = csv.writer(fh); w.writerow(["run", "name", "gps", "channel", "best_far", "p_astro", "p_astro_lo", "p_astro_hi"])
        for r in results:
            w.writerow([r["run"], r["name"], f"{r['gps']:.1f}", r["channel"], f"{r['best_far']:.4g}", f"{r['p_astro']:.4f}", f"{r['p_astro_lo']:.4f}", f"{r['p_astro_hi']:.4f}"])
    pa = np.array([r["p_astro"] for r in results])
    print(f"[done] {len(results)} rows -> {HERE}/pastro22{tag}.csv/.json; p_astro min={pa.min():.4f} median={np.median(pa):.4f} "
          f"n(p<0.5)={(pa < 0.5).sum()} n(p<0.9)={(pa < 0.9).sum()}; non-catalogue rows: " +
          ", ".join(f"{r['run']} {r['gps']:.0f} far={r['best_far']:.3g} p={r['p_astro']:.3f}" for r in results if not r["name"]))


if __name__ == "__main__":
    main()
