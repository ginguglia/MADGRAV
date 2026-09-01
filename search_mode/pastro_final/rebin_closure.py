#!/usr/bin/env python
"""Rebin-geometry closure dump (referee-response appendix for the two-panel
arithmetic; machinery for Gate 5). Verifies the comoving relabeled numerator
against its OWN decomposition, per run — distinct from check_fig_consistency.py
(which verifies the figures agree with each other).

Identities checked (from relabel_inj_<run>.npz + inj_scored_<run>.npz, no
recomputation of horizons):
  C1 det-frame identity: VT_det(b) == eps(b) * <Vc_trunc>_w0(b) * T  where
     eps(b) = sum(w0*kept*det)/sum(w0) over det bin b and <Vc>_w0 is the
     w0*kept*det-weighted mean truncated ball  ->  by construction exact;
     ALSO the stored vt_comoving_gpc3yr(b) is reproduced.
  C2 src-frame conservation: sum_k VT_src(k) + sink == sum_b VT_det(b).
  C3 src-frame identity: VT_src(k) == eps_eff(k) * Veff(k) * T with the
     volume-weighted effective efficiency eps_eff(k) = sum(share*det)/sum(share)
     over injections landing in src bin k (share = w0*kept*Vc/Wb) - reported
     per bin (this is the 'efficiency after the rebin' a referee will ask for).
  C4 horizon-volume ratios: coverage_comoving(b) = <Vc(D_rel/c)>/<Vc(D_rel)>
     vs the Euclidean expectation <c^-3> (low-z limit) - ratio must -> 1 as
     z_trunc -> 0 and stay <= 1 otherwise; the stored coverage_comoving is
     reproduced.
  C5 stored vt_comoving_srcframe_gpc3yr reproduced from the per-injection
     products (rebuild identity < 1e-9 rel).
Out: rebin_closure.json + rebin_closure.txt (tables); exit 1 on any failure.
"""
import json, sys, numpy as np
sys.path.insert(0, MADGRAV_ROOT + "/search_mode/pastro_final")
HERE = MADGRAV_ROOT + "/search_mode/pastro_final"
E = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.]); NB = len(E) - 1
from vt_relabel_comoving import comoving_machinery
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

vmax, z_of_dl = comoving_machinery()
rel = json.load(open(f"{HERE}/vt_relabel_comoving.json"))
out, lines, fail = {}, [], []
for run in ("O3a", "O3b", "O4a", "O4b"):
    R = rel["runs"][run]; T = R["T_obs_yr"]
    ri = np.load(f"{HERE}/relabel_inj_{run.lower()}.npz"); zi = np.load(f"{HERE}/inj_scored_{run.lower()}.npz")
    w0, det = zi["w0"], zi["det_frac"]; drel, c, kept, zsrc = ri["drel"], ri["c"], ri["kept"].astype(float), ri["z"]
    db, sb = ri["det_bin"], ri["src_bin"]
    vc_tr = vmax(drel / c); vc_rel = vmax(drel)
    r = dict(T_obs_yr=T, bins=[f"{E[b]:.0f}-{E[b+1]:.0f}" for b in range(NB)])
    # C1 / stored det-frame reproduction
    eps, Vmean, VTdet, cov, cinv3 = [], [], [], [], []
    for b in range(NB):
        s = db == b
        W = w0[s].sum()
        e = float((w0[s] * kept[s] * det[s]).sum() / W)
        wv = w0[s] * kept[s] * det[s]
        V = float((wv * vc_tr[s]).sum() / wv.sum()) if wv.sum() > 0 else 0.0
        VTdet.append(e * V * T * 1e-9); eps.append(e); Vmean.append(V)
        cov.append(float((w0[s] * vc_tr[s]).sum() / (w0[s] * vc_rel[s]).sum()))
        cinv3.append(float((w0[s] * c[s] ** -3).sum() / W))
    stored_det = np.array([np.nan if v is None else v for v in R["vt_comoving_gpc3yr"]], float)
    c1 = np.nanmax(np.abs(np.array(VTdet) / stored_det - 1))
    stored_cov = np.array([np.nan if v is None else v for v in R["coverage_comoving"]], float)
    c4 = np.nanmax(np.abs(np.array(cov) / stored_cov - 1))
    # C5 / C2 / C3 src-frame rebuild
    VTsrc = np.zeros(NB); sink = 0.0; share_sum = np.zeros(NB); share_det = np.zeros(NB); Veff_num = np.zeros(NB)
    for b in range(NB):
        s = np.where(db == b)[0]; Wb = w0[s].sum()
        share = w0[s] * kept[s] * vc_tr[s] / Wb            # volume share (before det)
        contrib = share * det[s] * T * 1e-9
        for i, k in enumerate(sb[s]):
            if k < 0: sink += contrib[i]
            else:
                VTsrc[k] += contrib[i]; share_sum[k] += share[i]; share_det[k] += share[i] * det[s][i]
    stored_src = np.array(R["vt_comoving_srcframe_gpc3yr"], float)
    c5 = np.max(np.abs(VTsrc - stored_src) / np.maximum(stored_src, 1e-12))
    c2 = abs((VTsrc.sum() + sink) - np.nansum(VTdet)) / np.nansum(VTdet)
    eps_eff = np.where(share_sum > 0, share_det / np.maximum(share_sum, 1e-300), np.nan)   # volume-weighted eff per src bin
    Veff = np.where(share_sum > 0, share_sum, np.nan)   # sum of w0-normalized truncated volume shares landing in k (Mpc^3, cohort-normalized)
    c3 = np.nanmax(np.abs(eps_eff * Veff * T * 1e-9 / np.where(stored_src > 0, stored_src, np.nan) - 1))
    ztr = np.array([np.nan if v is None else v for v in R["z_trunc_median"]], float)
    r.update(eps_det=eps, Vmean_trunc_Mpc3=Vmean, VT_det_rebuilt=VTdet, VT_det_stored=stored_det.tolist(),
             coverage_rebuilt=cov, coverage_stored=stored_cov.tolist(), c_inv3_expectation=cinv3, z_trunc_median=ztr.tolist(),
             VT_src_rebuilt=VTsrc.tolist(), VT_src_stored=stored_src.tolist(), sink_below20=float(sink),
             eps_eff_src=eps_eff.tolist(), Veff_src_share=Veff.tolist(),
             checks=dict(C1_detframe_identity_maxrel=float(c1), C2_conservation_rel=float(c2), C3_srcframe_identity_maxrel=float(c3),
                         C4_coverage_maxrel=float(c4), C4_cov_le_1=bool(np.all(np.array(cov) <= 1 + 1e-9)),
                         C4_cov_over_cinv3=[float(a / b) for a, b in zip(cov, cinv3)], C5_src_rebuild_maxrel=float(c5)))
    ok = c1 < 1e-9 and c2 < 1e-9 and c3 < 1e-9 and c4 < 1e-9 and c5 < 1e-9 and r["checks"]["C4_cov_le_1"]
    if not ok: fail.append(run)
    out[run] = r
    lines.append(f"== {run}  T={T:.4f} yr   C1 {c1:.1e}  C2 {c2:.1e}  C3 {c3:.1e}  C4 {c4:.1e}  C5 {c5:.1e}  {'PASS' if ok else 'FAIL'}")
    lines.append(f"  {'bin':>8} {'eps_det':>7} {'<Vc>Gpc3':>9} {'VT_det':>7} {'cov':>6} {'<c^-3>':>7} {'cov/c3':>6} {'z_tr':>5} | {'VT_src':>7} {'eps_eff':>7}")
    for b in range(NB):
        lines.append(f"  {r['bins'][b]:>8} {eps[b]:7.3f} {Vmean[b]*1e-9:9.3f} {VTdet[b]:7.3f} {cov[b]:6.3f} {cinv3[b]:7.3f} {cov[b]/cinv3[b]:6.3f} {ztr[b]:5.3f} | {VTsrc[b]:7.3f} {eps_eff[b]:7.3f}")
    lines.append(f"  sink(<20 Msun) = {sink:.4f} Gpc3yr ; sum src+sink = {VTsrc.sum()+sink:.4f} ; sum det = {np.nansum(VTdet):.4f}")
json.dump(out, open(f"{HERE}/rebin_closure.json", "w"), indent=1)
open(f"{HERE}/rebin_closure.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
if fail: print("CLOSURE FAIL:", fail); sys.exit(1)
print("[rebin_closure] ALL PASS")
