#!/usr/bin/env python
"""Z-CONSISTENT per-pipeline VT(source-frame Mtot) on OUR injected
population (approved 2026-08-12; amends the accepted target derivation
by explicit order - restraint standard logged).

WHY: our injected population is fixed-waveform - DETECTOR-frame masses drawn
from the bank law, independent of placement z. Its source-frame description
is therefore z-DEPENDENT: p(m1s, m2s | z) = p_bank((1+z)m1s, (1+z)m2s) *
(1+z)^2 (mass-measure Jacobian). The accepted target derivation read the
bank law as a z-independent source-frame density; after the numerator's
source-frame rebin the two sides described different populations (the
flagged mid-band residual). This script reweights the release injections to
exactly the z-dependent law.

ESTIMATOR (reduces to the accepted one as (1+z)->0-shift):
  per release injection j (recorded SOURCE masses m_j, redshift z_j):
    det-frame masses  M_j = (1+z_j) m_j
    det-bin  b_j = bin(Mtot_det,j)     [bank-frame bins; P(B) = accepted
                                        mass-only quadrature, z-independent]
    src-bin  k_j = bin(Mtot_src,j)     [the reported axis]
    w_j = p_bank(M1_j, M2_j) (1+z_j)^2 * bar(z_j) / p_draw_j
  VT_zc[k] = T * sum_b [ sum_{j in (b,k), det} w_j ] / (N * PB[b])
  - each det-frame cohort b is normalized by ITS OWN P(B) and its weight is
  scattered into source bins, mirroring the numerator's rebin exactly
  (same functional; same population; same cohort normalization).

Gates: identical to the accepted run (analytic-law KS, P(B) quadrature vs
bank samples, f_um, cosmology dL, weights-vs-month, z-law) - imported.
N_eff >= 300 enforced AFTER reweighting per SOURCE bin (widen rule).

Run: madgrav-venv python vt_pipelines_target_zc.py
Out: vt_pipelines_target_zc.json (schema of vt_pipelines_target.json)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import json
import sys

import h5py
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

import numpy as np

MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
sys.path.insert(0, HERE)

from vt_pipelines_gwtc import (COSMO_GWTC3, FAR_THR, MASS_EDGES, NEFF_MIN, YR,
                               ImbhLaw, bbh_mass_pdf, in_segments, load_o3,
                               neff, our_segments, verify, z_machinery,
                               zpdf_from_grid)
from vt_pipelines_gwtc5 import FILE as G5_FILE
from vt_pipelines_gwtc5 import MONTHS, RUN_RANGE
from vt_pipelines_target import (BANKS, F_UM_DESIGN, TargetLaw, bank_masses,
                                 bins_table, um_fraction, verify_target_law)

NB = len(MASS_EDGES) - 1


def scatter_vt(w_t, det, ins, mt_src, mt_det, pb_t, T_norm, N_tot):
    """Double-binned z-consistent estimator + per-src-bin N_eff."""
    binsD = np.digitize(mt_det, MASS_EDGES) - 1
    binsS = np.digitize(mt_src, MASS_EDGES) - 1
    vt = np.zeros(NB)
    wk = [[] for _ in range(NB)]
    sel_all = det & ins & (w_t > 0)
    for b in range(NB):
        if pb_t[b] <= 0:
            continue
        sel_b = sel_all & (binsD == b)
        if not sel_b.sum():
            continue
        for k in range(NB):
            sel = sel_b & (binsS == k)
            if not sel.sum():
                continue
            s = w_t[sel]
            vt[k] += T_norm * s.sum() / (N_tot * pb_t[b])
            wk[k].append(s / pb_t[b])   # pb-scaled weights for N_eff
    ne = [neff(np.concatenate(w)) if w else 0.0 for w in wk]
    return vt, ne


def main():
    banks = {tag: bank_masses(bdir) for tag, bdir in BANKS.items()}
    print(f"[banks] sig n={len(banks['sig'][0])}, um n={len(banks['um'][0])}")
    verify_target_law(banks)
    law = TargetLaw(F_UM_DESIGN)
    pb_t = law.PB()
    res = {"far_thr_per_yr": FAR_THR, "mass_edges": MASS_EDGES.tolist(),
           "neff_min": NEFF_MIN,
           "target": "Z-CONSISTENT: p(m_src|z) = p_bank((1+z) m_src) (1+z)^2 "
                     "(detector-frame bank law, fixed-waveform population); "
                     "src-frame bins; det-frame cohort P(B) normalization; "
                     "z prior dVc/dz/(1+z); spins at release law; f_um 0.5",
           "runs": {}, "validation": {}}
    for run in ("O3a", "O3b", "O4a", "O4b"):
        f_um, n_inj = um_fraction(run)
        res["validation"][f"{run}_target"] = dict(
            f_um_realized=f_um, f_um_law=F_UM_DESIGN, PB_target=pb_t.tolist())

    # ======== GWTC-3: O3a / O3b ========
    zg, _, bar, _ = z_machinery(COSMO_GWTC3)
    for run in ("O3a", "O3b"):
        data = load_o3(run)
        bbh, imbh = data["bbhpop"], data["imbhpop"]
        pm = bbh_mass_pdf(bbh["mass1_source"], bbh["mass2_source"],
                          a1=float(bbh["attrs"]["pow_mass1"]),
                          a2=float(bbh["attrs"]["pow_mass2"]))
        dev = np.abs(np.log(pm) -
                     np.log(bbh["mass1_source_mass2_source_sampling_pdf"]))
        verify(f"{run} bbh mass law", float(dev.max()))
        zmax = float(bbh["attrs"]["max_redshift"])
        zs_b, pz_b = zpdf_from_grid(zg, bar * (1 + zg) ** float(bbh["attrs"]["pow_z"]), zmax)
        ilaw = ImbhLaw(imbh["mass1_source"], imbh["mass2_source"],
                       np.log(imbh["mass1_source_mass2_source_sampling_pdf"]))
        verify(f"{run} imbh mass law (fit a={ilaw.a:.3f} b={ilaw.b:.3f})",
               ilaw.maxdev, tol=5e-3)
        zmax_i = float(imbh["redshift"].max())
        zs_i, pz_i = zpdf_from_grid(zg, bar, zmax_i)

        m1 = np.concatenate([bbh["mass1_source"], imbh["mass1_source"]])
        m2 = np.concatenate([bbh["mass2_source"], imbh["mass2_source"]])
        z = np.concatenate([bbh["redshift"], imbh["redshift"]])
        gps = np.concatenate([bbh["gps_time"], imbh["gps_time"]])
        Nb, Ni = bbh["N_gen"], imbh["N_gen"]
        p_b = bbh_mass_pdf(m1, m2) * np.interp(z, zs_b, pz_b) * (z <= zmax)
        p_i = ilaw.pdf(m1, m2) * np.interp(z, zs_i, pz_i) * (z <= zmax_i)
        p_mix = (Nb * p_b + Ni * p_i) / (Nb + Ni)
        assert np.all(p_mix > 0), "zero mixture density at recorded points"
        # Z-CONSISTENT law: bank law at det-frame masses, (1+z)^2 Jacobian
        opz = 1.0 + z
        w_t = law.pdf(m1 * opz, m2 * opz) * opz ** 2 * np.interp(z, zg, bar) / p_mix
        mt_src = m1 + m2
        mt_det = mt_src * opz
        cover = float((w_t[mt_src < 400] > 0).mean())
        print(f"[{run}] zc-target support coverage (src Mtot<400): {cover:.1%}")

        pipelines = {"cWB": "ifar_cwb", "GstLAL": "ifar_gstlal",
                     "MBTA": "ifar_mbta", "PyCBC-BBH": "ifar_pycbc_bbh",
                     "PyCBC-broad": "ifar_pycbc_hyperbank"}
        ifars = {p: np.concatenate([bbh[c], imbh[c]]) for p, c in pipelines.items()}
        iv, T_ours = our_segments(run)
        ins = in_segments(gps, iv)
        run_out = {"T_ours_yr": T_ours, "f_um": F_UM_DESIGN, "pipelines": {}}
        for p in pipelines:
            det = ifars[p] > 1.0 / FAR_THR
            vt, ne = scatter_vt(w_t, det, ins, mt_src, mt_det, pb_t,
                                T_ours, Nb + Ni)
            run_out["pipelines"][p] = bins_table(vt, ne)
        res["runs"][run] = run_out

    # ======== GWTC-5: O4a / O4b ========
    f5 = h5py.File(G5_FILE, "r")
    at = dict(f5.attrs.items())
    ev = f5["events"]
    c5 = {k: ev[k][:] for k in
          ("mass1_source", "mass2_source", "z", "luminosity_distance",
           "lnpdraw_mass1_source", "lnpdraw_mass2_source_GIVEN_mass1_source",
           "lnpdraw_z", "time_geocenter", "weights",
           "cwb-bbh_far", "gstlal_far", "mbta_far", "pycbc_far")}
    f5.close()
    m1, m2, z5 = c5["mass1_source"], c5["mass2_source"], c5["z"]
    N_gen, T_tot = int(at["total_generated"]), float(at["total_analysis_time"])

    zg5 = np.geomspace(1e-6, float(z5.max()) * 1.001, 6000)
    import astropy.units as u
    dlg = COSMO_GWTC3.luminosity_distance(zg5).to(u.Mpc).value
    sub = np.random.default_rng(1).choice(len(z5), 200_000, replace=False)
    verify("O4ab cosmology dL(z) [max rel]",
           float(np.abs(np.interp(z5[sub], zg5, dlg) /
                        c5["luminosity_distance"][sub] - 1).max()), tol=1e-3)
    mi = np.digitize(c5["time_geocenter"], [m[0] for m in MONTHS] + [MONTHS[-1][1]]) - 1
    w_pred = np.array([(hi - lo) / n / (T_tot / N_gen) for lo, hi, n in MONTHS])
    verify("O4ab weights vs month table [max rel]",
           float(np.abs(c5["weights"] / w_pred[mi] - 1).max()), tol=1e-3)
    dvdz5 = (COSMO_GWTC3.differential_comoving_volume(zg5)
             .to(u.Gpc ** 3 / u.sr).value * 4 * np.pi)
    bar5 = dvdz5 / (1 + zg5)
    A = np.column_stack([np.ones_like(z5), np.log1p(z5)])
    (a0, k0), *_ = np.linalg.lstsq(A, c5["lnpdraw_z"] -
                                   np.interp(z5, zg5, np.log(dvdz5)), rcond=None)
    verify("O4ab z law k(1+z) ~ 0 [|k|]", abs(float(k0)), tol=1e-2)

    lnp_draw = (c5["lnpdraw_mass1_source"] +
                c5["lnpdraw_mass2_source_GIVEN_mass1_source"] + c5["lnpdraw_z"])
    opz5 = 1.0 + z5
    w_t5 = (c5["weights"] * law.pdf(m1 * opz5, m2 * opz5) * opz5 ** 2 *
            np.interp(z5, zg5, bar5) * np.exp(-lnp_draw))
    mt_src5 = m1 + m2
    mt_det5 = mt_src5 * opz5
    pipelines5 = {"cWB": "cwb-bbh_far", "GstLAL": "gstlal_far",
                  "MBTA": "mbta_far", "PyCBC": "pycbc_far"}
    gps5 = c5["time_geocenter"]
    for run in ("O4a", "O4b"):
        cover = float((w_t5[mt_src5 < 400] > 0).mean())
        print(f"[{run}] zc-target support coverage (src Mtot<400): {cover:.1%}")
        iv, T_ours = our_segments(run)
        lo_r, hi_r = RUN_RANGE[run]
        ivc = np.column_stack([np.clip(iv[:, 0], lo_r, None),
                               np.clip(iv[:, 1], None, hi_r)])
        ivc = ivc[ivc[:, 1] > ivc[:, 0]]
        tau_clip = float((ivc[:, 1] - ivc[:, 0]).sum()) / YR
        scale = T_ours / tau_clip
        if scale > 1.10:
            raise SystemExit(f"{run}: clip scale {scale} > 1.10")
        ins = in_segments(gps5, ivc)
        print(f"[S] {run}: {int(ins.sum())}/{len(gps5)} in segments; "
              f"exposure scale = {scale:.4f}")
        run_out = {"T_ours_yr": T_ours, "tau_inrange_yr": tau_clip,
                   "exposure_scale": scale, "f_um": F_UM_DESIGN,
                   "n_inj_in_segments": int(ins.sum()), "pipelines": {}}
        for p, cname in pipelines5.items():
            det = c5[cname] < FAR_THR
            vt, ne = scatter_vt(w_t5, det, ins, mt_src5, mt_det5, pb_t,
                                scale * T_tot / YR, N_gen)
            run_out["pipelines"][p] = bins_table(vt, ne)
        res["runs"][run] = run_out

    # ---- vs the accepted (z-independent) target tables ----
    ref = json.load(open(f"{HERE}/vt_pipelines_target.json"))
    print("\n=== zc/accepted VT ratio (bins with both N_eff>=300) ===")
    for run in res["runs"]:
        for p in res["runs"][run]["pipelines"]:
            a = res["runs"][run]["pipelines"][p]
            b = ref["runs"][run]["pipelines"][p]
            rr = [f"{av/bv:4.2f}" if (an >= NEFF_MIN and bn >= NEFF_MIN
                                      and bv and bv > 0) else "  - "
                  for av, an, bv, bn in zip(a["vt_gpc3yr"], a["neff"],
                                            b["vt_gpc3yr"], b["neff"])]
            print(f"  {run} {p:>12}: " + " ".join(rr))

    json.dump(res, open(f"{HERE}/vt_pipelines_target_zc.json", "w"), indent=1)
    print(f"[done] -> {HERE}/vt_pipelines_target_zc.json")


if __name__ == "__main__":
    main()
