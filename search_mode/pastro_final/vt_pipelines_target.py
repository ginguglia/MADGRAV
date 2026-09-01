#!/usr/bin/env python
"""Per-pipeline VT(Mtot) bin-conditioned on OUR injected population
(target-drift fix, specification 2026-08-12; ANALYTIC target law per specification:
extracted sampling law, verified against bank samples - NO fitted densities).

TARGET POPULATION (analytic, KS-verified at startup - abort on failure):
  stellar bank (o1_o3_signal_bank_projected_2s_x10, dir-name params
  m1 10-120 qmax 6; m2 floor 10 identified from Mtot_min=20):
      m1 ~ U(10,120);  m2|m1 ~ U(max(m1/6,10), m1)
      p_sig(m1,m2) = (1/110) / (m1 - max(m1/6,10))
  ultramassive bank (Mtot 150-400, qmax 4):
      Mtot ~ U(150,400) independent of q ~ U(1,4)
      p_um(m1,m2) = (1/750) (m1+m2)/m2^2       [Jacobian d(Mt,q)/d(m1,m2)]
  mixture: f_um = 0.5 by design (inject.py UM_FRAC mass-stratified draw,
  uniform over bank entries -> injected law == bank generation law);
  realized fraction gated binomially per run.
  z prior: dVc/dz (1+z)^-1 (uniform-comoving + dilation) - the population our
  rho^-4 weighting reweights to; SNR-uniform placement is a scheme, not a
  population. Spins (stated, not corrected): pipeline-side efficiency
  marginalized at the RELEASE spin law; our stellar arm is non-spinning Pv2,
  UM arm XPHM precessing.
  P_t(B): deterministic quadrature of the analytic law; MC-vs-bank-samples
  binomial gate.

ESTIMATOR: identical skeleton/gates to the accepted step-2 scripts;
N_eff >= 300 enforced AFTER reweighting (widen rule).
Run: madgrav-venv python vt_pipelines_target.py
Output: vt_pipelines_target.json + printed target/release-mixture ratios.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import csv
import glob
import json
import sys

import h5py
import numpy as np
from scipy.stats import kstest
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
sys.path.insert(0, HERE)

from vt_pipelines_gwtc import (COSMO_GWTC3, FAR_THR, MASS_EDGES, NEFF_MIN, YR,
                               ImbhLaw, bbh_mass_pdf, in_segments, load_o3,
                               neff, our_segments, verify, widen_bins,
                               z_machinery, zpdf_from_grid)
from vt_pipelines_gwtc5 import FILE as G5_FILE
from vt_pipelines_gwtc5 import MONTHS, RUN_RANGE

BANKS = {"sig": f"{MG}/data/o1_o3_signal_bank_projected_2s_x10",
         "um": f"{MG}/data/ultramassive_bank"}
F_UM_DESIGN = 0.5
D5 = MADGRAV_EXTDATA + "/gwtc5_sensitivity"


# ---------- target mass law (ANALYTIC) ----------
def bank_masses(bdir):
    m1, m2 = [], []
    for path in sorted(glob.glob(os.path.join(bdir, "**", "signals_*.csv"),
                                 recursive=True)):
        with open(path, newline="") as fin:
            for row in csv.DictReader(fin):
                m1.append(float(row["mass1"])); m2.append(float(row["mass2"]))
    return np.array(m1), np.array(m2)


def p_sig(m1, m2):
    lo = np.maximum(m1 / 6.0, 10.0)
    sup = (m1 >= 10.0) & (m1 <= 120.0) & (m2 >= lo) & (m2 <= m1) & (m1 > lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (1.0 / 110.0) / (m1 - lo)
    return np.where(sup, p, 0.0)


def p_um(m1, m2):
    mt, q = m1 + m2, m1 / m2
    sup = (mt >= 150.0) & (mt <= 400.0) & (q >= 1.0) & (q <= 4.0)
    return np.where(sup, (1.0 / 750.0) * mt / m2 ** 2, 0.0)


class TargetLaw:
    """Analytic per-run bank-mixture mass law."""

    def __init__(self, f_um):
        self.f_um = f_um

    def pdf(self, m1, m2):
        return (1.0 - self.f_um) * p_sig(m1, m2) + self.f_um * p_um(m1, m2)

    def PB(self):
        # stellar: 1D quadrature over m1 of the in-bin m2 fraction
        m1g = np.linspace(10.0, 120.0, 200_001)[1:]
        lo = np.maximum(m1g / 6.0, 10.0)
        pb_sig = []
        for k in range(len(MASS_EDGES) - 1):
            a, b = MASS_EDGES[k], MASS_EDGES[k + 1]
            l = np.maximum(lo, a - m1g)
            u_ = np.minimum(m1g, b - m1g)
            frac = np.clip(u_ - l, 0.0, None) / np.maximum(m1g - lo, 1e-12)
            pb_sig.append(float(np.trapezoid(frac, m1g) / 110.0))
        # um: q integrates out; Mtot uniform on [150,400]
        pb_um = [max(0.0, (min(MASS_EDGES[k + 1], 400.0) -
                           max(MASS_EDGES[k], 150.0))) / 250.0
                 for k in range(len(MASS_EDGES) - 1)]
        return ((1.0 - self.f_um) * np.array(pb_sig) +
                self.f_um * np.array(pb_um))


def verify_target_law(banks):
    """KS-verify the frozen analytic laws against the bank samples (gates)."""
    m1, m2 = banks["sig"]
    verify("sig m1 ~ U(10,120) [1-KSp]",
           1.0 - kstest(m1, "uniform", args=(10, 110)).pvalue, tol=1.0 - 1e-3)
    lo = np.maximum(m1 / 6.0, 10.0)
    verify("sig m2|m1 ~ U(max(m1/6,10),m1) [1-KSp]",
           1.0 - kstest((m2 - lo) / (m1 - lo), "uniform").pvalue, tol=1.0 - 1e-3)
    m1u, m2u = banks["um"]
    mt, q = m1u + m2u, m1u / m2u
    verify("um Mtot ~ U(150,400) [1-KSp]",
           1.0 - kstest(mt, "uniform", args=(150, 250)).pvalue, tol=1.0 - 1e-3)
    verify("um q ~ U(1,4) [1-KSp]",
           1.0 - kstest(q, "uniform", args=(1, 3)).pvalue, tol=1.0 - 1e-3)
    verify("um corr(Mtot,q) ~ 0 [|r|]",
           abs(float(np.corrcoef(mt, q)[0, 1])), tol=0.05)
    # P_t(B) quadrature vs empirical bank bin fractions (binomial gate, 5 sig)
    law = TargetLaw(F_UM_DESIGN)
    pb = law.PB()
    pe = np.zeros(len(MASS_EDGES) - 1)
    var = np.zeros(len(MASS_EDGES) - 1)
    for tag, w in (("sig", 1 - F_UM_DESIGN), ("um", F_UM_DESIGN)):
        a, b = banks[tag]
        mtt = a + b
        n = len(mtt)
        pk = np.array([((mtt >= MASS_EDGES[k]) & (mtt < MASS_EDGES[k + 1])).sum() / n
                       for k in range(len(MASS_EDGES) - 1)])
        pe += w * pk
        var += w ** 2 * pk * (1 - pk) / n      # per-component binomial variance
    sig = np.sqrt(var)
    zmax = float(np.abs((pe - pb) / np.maximum(sig, 1e-9)).max())
    print(f"        (P_t(B) quad vs bank samples: max |z| = {zmax:.2f}; "
          f"PB = {np.array2string(pb, precision=5)})")
    verify("P_t(B) quadrature vs bank samples [max|z|/5]", zmax / 5.0, tol=1.0)


def um_fraction(run):
    import vt_search as vs
    um = vs.load_is_um(run)
    f, n = float(um.mean()), len(um)
    zdev = abs(f - F_UM_DESIGN) / np.sqrt(F_UM_DESIGN * 0.5 / n)
    verify(f"{run} realized f_um vs design 0.5 [|z|/5]", zdev / 5.0, tol=1.0)
    return f, n


def bins_table(per_bin_vt, per_bin_neff):
    groups = widen_bins(MASS_EDGES, per_bin_neff)
    return dict(vt_gpc3yr=[float(x) for x in per_bin_vt],
                neff=[float(x) for x in per_bin_neff], widened_groups=groups)


def main():
    banks = {tag: bank_masses(bdir) for tag, bdir in BANKS.items()}
    print(f"[banks] sig n={len(banks['sig'][0])}, um n={len(banks['um'][0])}")
    verify_target_law(banks)
    res = {"far_thr_per_yr": FAR_THR, "mass_edges": MASS_EDGES.tolist(),
           "neff_min": NEFF_MIN,
           "target": "our injected bank mixture, ANALYTIC law (see module "
                     "docstring); z prior dVc/dz/(1+z); spins marginalized "
                     "at release law; f_um = design 0.5",
           "runs": {}, "validation": {}}
    laws = {}
    for run in ("O3a", "O3b", "O4a", "O4b"):
        f_um, n_inj = um_fraction(run)
        law = TargetLaw(F_UM_DESIGN)      # design law; realized f_um gated above
        pb = law.PB()
        print(f"[target {run}] realized f_um = {f_um:.4f} ({n_inj} records), "
              f"law uses design 0.5")
        res["validation"][f"{run}_target"] = dict(
            f_um_realized=f_um, f_um_law=F_UM_DESIGN, PB_target=pb.tolist())
        laws[run] = (law, pb)

    ref3 = json.load(open(f"{HERE}/vt_pipelines_gwtc.json"))
    ref5 = json.load(open(f"{HERE}/vt_pipelines_gwtc5.json"))

    # ======== GWTC-3: O3a / O3b (draw machinery identical to accepted run) ==
    zg, _, bar, _ = z_machinery(COSMO_GWTC3)
    for run in ("O3a", "O3b"):
        law_t, pb_t = laws[run]
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
        w_t = law_t.pdf(m1, m2) * np.interp(z, zg, bar) / p_mix
        mt = m1 + m2
        cover = float((w_t[mt < 400] > 0).mean())
        print(f"[{run}] target-support coverage of recorded injections "
              f"(Mtot<400): {cover:.1%}")

        pipelines = {"cWB": "ifar_cwb", "GstLAL": "ifar_gstlal",
                     "MBTA": "ifar_mbta", "PyCBC-BBH": "ifar_pycbc_bbh",
                     "PyCBC-broad": "ifar_pycbc_hyperbank"}
        ifars = {p: np.concatenate([bbh[c], imbh[c]]) for p, c in pipelines.items()}
        iv, T_ours = our_segments(run)
        ins = in_segments(gps, iv)
        bins = np.digitize(mt, MASS_EDGES) - 1
        run_out = {"T_ours_yr": T_ours, "f_um": laws[run][0].f_um, "pipelines": {}}
        for p in pipelines:
            det = (ifars[p] > 1.0 / FAR_THR) & ins
            vt_b, ne_b = [], []
            for k in range(len(MASS_EDGES) - 1):
                sel = det & (bins == k)
                w = w_t[sel]
                vt_b.append(T_ours * w.sum() / ((Nb + Ni) * pb_t[k])
                            if pb_t[k] > 0 else np.nan)
                ne_b.append(neff(w))
            run_out["pipelines"][p] = bins_table(vt_b, ne_b)
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
    mt5 = m1 + m2
    N_gen, T_tot = int(at["total_generated"]), float(at["total_analysis_time"])

    # same gates as the accepted 12:18 run (G1 dL, G2 weights, G3 z law)
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
    pipelines5 = {"cWB": "cwb-bbh_far", "GstLAL": "gstlal_far",
                  "MBTA": "mbta_far", "PyCBC": "pycbc_far"}
    gps5 = c5["time_geocenter"]
    bins5 = np.digitize(mt5, MASS_EDGES) - 1
    for run in ("O4a", "O4b"):
        law_t, pb_t = laws[run]
        w_t = (c5["weights"] * law_t.pdf(m1, m2) *
               np.interp(z5, zg5, bar5) * np.exp(-lnp_draw))
        cover = float((w_t[mt5 < 400] > 0).mean())
        print(f"[{run}] target-support coverage of recorded injections "
              f"(Mtot<400): {cover:.1%}")
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
                   "exposure_scale": scale, "f_um": law_t.f_um,
                   "n_inj_in_segments": int(ins.sum()), "pipelines": {}}
        for p, cname in pipelines5.items():
            det = (c5[cname] < FAR_THR) & ins
            vt_b, ne_b = [], []
            for k in range(len(MASS_EDGES) - 1):
                sel = det & (bins5 == k)
                w = w_t[sel]
                vt_b.append(scale * T_tot / YR * w.sum() / (N_gen * pb_t[k])
                            if pb_t[k] > 0 else np.nan)
                ne_b.append(neff(w))
            run_out["pipelines"][p] = bins_table(vt_b, ne_b)
        res["runs"][run] = run_out

    # ---- comparison vs the accepted release-mixture tables ----
    print("\n=== target/release-mixture VT ratio (bins with both N_eff>=300) ===")
    for run, ref in (("O3a", ref3), ("O3b", ref3), ("O4a", ref5), ("O4b", ref5)):
        for p in res["runs"][run]["pipelines"]:
            if p not in ref["runs"][run]["pipelines"]:
                continue
            a = res["runs"][run]["pipelines"][p]
            b = ref["runs"][run]["pipelines"][p]
            rr = [f"{av/bv:4.2f}" if (an >= NEFF_MIN and bn >= NEFF_MIN and bv > 0)
                  else "  - "
                  for av, an, bv, bn in zip(a["vt_gpc3yr"], a["neff"],
                                            b["vt_gpc3yr"], b["neff"])]
            print(f"  {run} {p:>12}: " + " ".join(rr))

    json.dump(res, open(f"{HERE}/vt_pipelines_target.json", "w"), indent=1)
    print(f"[done] -> {HERE}/vt_pipelines_target.json")


if __name__ == "__main__":
    main()
