#!/usr/bin/env python
"""successor_inj.py -- Sec. 7.2: injections re-scored with the SUCCESSOR counting + FGMC p_astro (2026-08-18).

Mirrors pastro_final.py (same injection files, SNR-grid weights, mass reweighting to the catalog, FAR bins, pinned
Lambda_n = DET_FAR * T_fg, bootstrap) with the counting swapped to successor_stat: per injection (cross-fit fold model)
    N_lr  = # surviving lnLambda-channel families of the fold with r_lr > lnLambda_inj   (if lnLambda_inj >= 4.0)
    N_net = # surviving sigma_net-channel families with r_net >= sigma_net_inj              (if sigma_net_inj >= 4.0)
    FAR   = N_eff(run, fold) * min_c N_c / T_f ;  UL90 = N_eff * chi2(.9,2(N+1))/2 / T_f ;  det = trig & FAR<1 & UL90<1
The injection's own time is not stored (random placement in inject.py), so the narrowed self-exclusion (own +/-4 s
windows) cannot be applied per injection; it is omitted (a <1e-3 relative effect on N; the injections' as-run
scoring applied a whole-segment exclusion instead, which is now abolished).  The successor rank does not depend on
the candidate's arm scores, so no CNN-pair averaging is needed (npair = 1); the FGMC uses the successor detections
(detections_successor.json, best_far = successor FAR).
Outputs (this dir's pastro_final/): inj_scored_<run>_successor.npz (same fields), pastro_final_successor.{json,csv}.
"""
import os, sys, json, glob, csv, hashlib
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import successor_stat as S
from successor_neff import neff_of
sys.path.insert(0, f"{S.MG}/search_mode/pastro_final")
import pastro_final as PF                # constants + feats/loglr_of/fit_fgmc (unchanged)

DET = os.environ.get("SUCC_DET", S.DET); OUTDIR = os.environ.get("SUCC_OUT_DIR")
HERE = OUTDIR or PF.HERE


def main(runs):
    S.assert_spec(); fzp = f"{DET}/neff_freeze.json"; fz = json.load(open(fzp)); fmd5 = hashlib.md5(open(fzp, "rb").read()).hexdigest()
    cat_mtot = np.array([float(r["total_mass_source"]) for r in csv.DictReader(open(PF.CATF)) if r["total_mass_source"].strip()])
    f_cat = np.histogram(cat_mtot, PF.MASS_EDGES)[0] / len(cat_mtot)
    mids = (PF.SNR_GRID[1:] + PF.SNR_GRID[:-1]) / 2; dr = np.diff(np.concatenate([[PF.SNR_GRID[0]], mids, [PF.SNR_GRID[-1]]]))
    w_snr_grid = PF.SNR_GRID ** -4.0 * dr; w_snr_grid /= w_snr_grid.sum(); w_snr_of = {float(s): w for s, w in zip(PF.SNR_GRID, w_snr_grid)}
    pn = np.diff(PF.FAR_EDGES) / (PF.FAR_EDGES[-1] - PF.FAR_EDGES[0]); widths = np.diff(PF.FAR_EDGES)
    results = []; summary = {}
    ev2seg = {d["matches_known"]: d["seg"] for d in json.load(open(f"{PF.RUNS['O3a']['out']}/detections.json")) if d.get("matches_known")}
    for run in runs:
        vp = f"{S.DET}/bg_veto_{run.lower()}.npz"; bg = S.Background(run, vp if os.path.exists(vp) else None, verbose=False)
        dets = json.load(open(f"{OUTDIR}/detections_successor_{run.lower()}.json" if OUTDIR else f"{S.RUNS[run]}/detections_successor.json"))["detections"]
        files = []
        for d in PF.RUNS[run]["inj"]:
            fl = sorted(glob.glob(f"{d}/*_inj.npz")); assert fl, f"{run}: no injections in {d}"; files += fl
        far_all = []; ul_all = []; det_all = []; inj_mtot = []; inj_snr = []; inj_w0 = []; inj_ev = []
        for f in files:
            ev = os.path.basename(f)[:-8]
            if ev in bg.seg_ix: seg = ev
            elif ev in ev2seg and ev2seg[ev] in bg.seg_ix: seg = ev2seg[ev]
            elif ev in PF.EV_GPS:
                by_start = sorted(bg.seg_names, key=lambda n: int(n.rsplit("_", 1)[1])); starts = np.array([int(n.rsplit("_", 1)[1]) for n in by_start])
                seg = by_start[int(np.searchsorted(starts, PF.EV_GPS[ev], side="right") - 1)]
            else: raise SystemExit(f"{run}: cannot resolve {ev}")
            z = np.load(f); F = PF.feats(z); net = z["net"].astype(float); mtot = z["mtot"].astype(float); snr = z["net_snr"].astype(float)
            w0 = np.array([w_snr_of[s] for s in snr]); g = int(bg.seg_fold[bg.seg_ix[seg]]); T = bg.F[g]["T"]
            x = PF.loglr_of(F, 1 - g)                                       # cross-fit as before
            Fd = bg.F[g]
            N_lr = np.searchsorted(-Fd["lr_sorted"], -x, "left").astype(float); N_net = np.searchsorted(-Fd["net_sorted"], -net, "right").astype(float)
            lr_ok = x >= PF.FLOOR; nt_ok = net >= PF.NETSIG_FLOOR; trig = net > PF.NET_CUT
            N_lr[~lr_ok] = np.inf; N_net[~nt_ok] = np.inf
            Nmin = np.minimum(N_lr, N_net); ne = neff_of(fz["runs"], run, g)
            nef = np.array([ne(v) for v in Nmin / T]) if callable(ne) else ne
            far = np.where(np.isfinite(Nmin), nef * Nmin / T, np.nan)
            ul = np.where(np.isfinite(Nmin), nef * chi2.ppf(0.9, 2 * (np.where(np.isfinite(Nmin), Nmin, 0) + 1)) / 2 / T, np.nan)
            det = trig & np.isfinite(far) & (far < PF.DET_FAR) & (ul < PF.DET_FAR)
            far_all.append(far); ul_all.append(ul); det_all.append(det); inj_mtot.append(mtot); inj_snr.append(snr); inj_w0.append(w0); inj_ev.append(np.full(len(x), f"{ev}:f{g}"))
        far_all = np.concatenate(far_all); det_all = np.concatenate(det_all); inj_mtot = np.concatenate(inj_mtot); inj_snr = np.concatenate(inj_snr)
        inj_w0 = np.concatenate(inj_w0); inj_ev = np.concatenate(inj_ev)
        mbins = np.clip(np.digitize(inj_mtot, PF.MASS_EDGES) - 1, 0, len(PF.MASS_EDGES) - 2)
        def ps_of(w_inj):
            D = det_all; W = w_inj[D]; MTb = mbins[D]
            f_inj = np.bincount(MTb, weights=W, minlength=len(PF.MASS_EDGES) - 1); f_inj = f_inj / max(f_inj.sum(), 1e-300)
            wm = np.where(f_inj > 0, f_cat / np.maximum(f_inj, 1e-12), 0.0)
            ps, _ = np.histogram(far_all[D], PF.FAR_EDGES, weights=W * wm[MTb]); ps = ps / max(ps.sum(), 1e-300); ps = np.maximum(ps, 1e-6)
            return ps / ps.sum()
        segj = {s[3]: s[2] for s in json.load(open(f"{PF.SC}/{run.lower()}_full_coincident.json"))["segments"]}
        T_fg = sum(segj.get(n, 0.0) for n in bg.seg_names) * 0.5 / 3.1557e7; Ln_null = PF.DET_FAR * T_fg
        xs = np.array([d["far"] for d in dets]); ps = ps_of(inj_w0)
        if len(xs):
            Ls, Ln, pa = PF.fit_fgmc(xs, ps, pn, widths, Ln_fixed=Ln_null); Ls_f, Ln_f, pa_f = PF.fit_fgmc(xs, ps, pn, widths)
            rng = np.random.default_rng(20260811); B = 200; pa_boot = np.empty((B, len(xs)))
            for b in range(B):
                wb = inj_w0 * rng.dirichlet(np.ones(len(inj_w0))) * len(inj_w0)
                pa_boot[b] = PF.fit_fgmc(xs, ps_of(wb), pn, widths, Ln_fixed=Ln_null)[2]
            lo, hi = np.percentile(pa_boot, [5, 95], axis=0)
        else:
            Ls = Ln = Ls_f = Ln_f = float("nan"); pa = pa_f = lo = hi = np.zeros(0)
        eff = float((det_all * inj_w0).sum() / inj_w0.sum())
        print(f"[{run}] successor: eff={eff:.3f} T_fg={T_fg:.4f}yr Ln={Ln:.3f} Ls={Ls:.2f} (free Ls={Ls_f:.2f} Ln={Ln_f:.2f}) n_det={len(xs)} ps={np.array2string(ps, precision=3)}", flush=True)
        for i, d in enumerate(dets):
            results.append(dict(run=run, seg=d["seg"], gps=d["gps"], name=d.get("matches_known", ""), loglr=d["loglr"], net=d["net"], far=float(xs[i]), channel=d["channel"],
                                p_astro=float(pa[i]), p_astro_lo=float(lo[i]), p_astro_hi=float(hi[i]), p_astro_freeLn=float(pa_f[i]), Ls=float(Ls), Ln=float(Ln), Ls_free=float(Ls_f), Ln_free=float(Ln_f), T_fg_yr=float(T_fg)))
        np.savez(f"{HERE}/inj_scored_{run.lower()}_successor.npz", far_mean=far_all, det_frac=det_all.astype(float), mtot=inj_mtot, net_snr=inj_snr, w0=inj_w0, ev=inj_ev, npair=1,
                 neff_freeze_md5=np.array(fmd5), module_md5=np.array(S.module_md5()))
        summary[run] = dict(eff=eff, T_fg=T_fg, Ls=Ls, Ln=Ln, n_det=len(xs))
    json.dump(dict(module_md5=S.module_md5(), neff_freeze_md5=fmd5, summary=summary, detections=results), open(f"{HERE}/pastro_final_successor.json", "w"), indent=1)
    with open(f"{HERE}/pastro_final_successor.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["run", "name", "far", "p_astro", "p_astro_lo", "p_astro_hi"])
        for r in results: w.writerow([r["run"], r["name"] or f"{r['seg']}@{r['gps']:.0f}", r["far"], f"{r['p_astro']:.4f}", f"{r['p_astro_lo']:.4f}", f"{r['p_astro_hi']:.4f}"])
    print(f"-> {HERE}/pastro_final_successor.json/.csv; inj_scored_<run>_successor.npz")


if __name__ == "__main__":
    main(sys.argv[1:] or S.MAIN_RUNS)
