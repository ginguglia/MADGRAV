import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)

#!/usr/bin/env python
"""Per-pipeline VT(Mtot) from the GWTC-5.0 O4a+O4b injection release, on
MADGRAV's bins. Companion to vt_pipelines_gwtc.py (GWTC-3 / O3); separate
module because the release schema is structurally different (checked
2026-08-12 against gwtc-5_o4ab_sensitivity-estimates.md):

  * ONE file, ONE population (samples-rpo4ab, zenodo 19500064) - no bbh/imbh
    pair, so the O3 mixture logic does NOT apply. Target mass density factors
    cancel exactly (target = draw mass marginal conditioned on the bin); only
    P(B) requires the mass law.
  * significances are far_<search> in 1/yr (default inf), not ifar_*.
  * draw densities are per-event lnpdraw_* columns, not global attrs.
  * per-month `weights` column (Essick 2021 mixture weights over months):
    w_m = (tau_m/N_m)/(T_total/N_total), full month windows. Restricting the
    release estimator  vt = T_total * sum_det[w * r]/N_total  to a time
    subset S then yields tau_S-weighted exposure AUTOMATICALLY (per month:
    E[sum_{i in S} w r] = (N/T) tau_S <eps r>), i.e. wall-clock T of our HL
    segments enters through the weights - do NOT multiply by T_ours again.
  * no surveyed_VT attr -> V1 analog = reproduce the release's own
    luminosity_distance(z) column from the identified cosmology, plus exact
    reconstruction of the weights column and T_total from the month table.
  * clipped file: only possibly-detected injections recorded (2.96M of 8.8M
    accepted / 870.5M generated) -> P(B) must NOT use recorded events;
    numerator is complete by release design.

Identified draw laws (recon 2026-08-12, all re-verified with abort gates
at every run):
  cosmology  FlatLambdaCDM(H0=67.9, Om0=0.3065)   [dL max rel dev 5.5e-5]
  p(z)      ~ dVc/dz (NO (1+z)^-1), z <= ~3       [fitted k = 0.0000]
  p(m2|m1)  = (b+1) m2^b / (m1^(b+1)-mmin^(b+1)), b ~= 1.0006, mmin ~= 1
  p(m1)     NOT a simple power law (global fit resid ~2) -> reconstructed
            nonparametrically: grid-interpolated from the file's own exact
            lnpdraw_mass1_source values, verified on a held-out half and
            required to integrate to 1 (coverage gate).

Estimator (per run R in {O4a,O4b}, pipeline p, Mtot bin B, segs = MADGRAV
HL segments of R):
    VT_B^p = (T_total/N_gen) * sum_{i in B, det_p, segs} w_i r_i / P(B)
    r_i    = [dVc/dz (1+z)^-1](z_i) / exp(lnpdraw_z_i)          [Gpc^3]
    det_p  = far_p < FAR_THR
  (mass draw densities cancel between target and draw; P(B) from the
  reconstructed mass law; tau_segs exposure carried by w_i, see above.)

N_eff rule (specification): per bin N_eff = (sum w)^2 / sum w^2 over detected
weights; bins with N_eff < 300 are WIDENED (merged rightward), not plotted.

Output: vt_pipelines_gwtc5.json (same structure as vt_pipelines_gwtc.json).
Run: madgrav-venv python vt_pipelines_gwtc5.py [--validate-only]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys
import json
import h5py
import numpy as np

D = MADGRAV_EXTDATA + "/gwtc5_sensitivity"
MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
HERE = f"{MG}/search_mode/pastro_final"
FILE = f"{D}/samples-rpo4ab-1366933504-55469568-clipped.hdf"
YR = 3.1557e7
FAR_THR = 1.0                                   # 1/yr
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])
NEFF_MIN = 300

# official run ranges (release md); our segments lie inside these
RUN_RANGE = {"O4a": (1368975618, 1389456018), "O4b": (1396969218, 1422118818)}

# month table from the release md: (gps_lo, gps_hi, N_generated)
MONTHS = [
    (1366933504, 1369612288, 60499005), (1369612288, 1372205056, 53692138),
    (1372205056, 1374882304, 53318599), (1374882304, 1377561088, 49285598),
    (1377561088, 1380153856, 44134007), (1380153856, 1382831104, 43618343),
    (1382831104, 1385423872, 41870421), (1385423872, 1388102656, 43738263),
    (1388102656, 1390779904, 43001343),
    (1393286656, 1395963904, 40061791), (1395963904, 1398556672, 38060841),
    (1398556672, 1401235456, 40800410), (1401235456, 1403826688, 39871350),
    (1403826688, 1406505472, 40679560), (1406505472, 1409184256, 42671640),
    (1409184256, 1411775488, 38018959), (1411775488, 1414454272, 39558043),
    (1414454272, 1417045504, 38677337), (1417045504, 1419724288, 39624771),
    (1419724288, 1422403072, 39272453),
]

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

COSMO = FlatLambdaCDM(H0=67.9, Om0=0.3065)

from vt_pipelines_gwtc import verify, neff, widen_bins, our_segments, in_segments


def main(validate_only=False):
    f = h5py.File(FILE, "r")
    at = dict(f.attrs.items())
    ev = f["events"]
    cols = {}
    for k in ("mass1_source", "mass2_source", "z", "luminosity_distance",
              "lnpdraw_mass1_source", "lnpdraw_mass2_source_GIVEN_mass1_source",
              "lnpdraw_z", "time_geocenter", "weights",
              "cwb-bbh_far", "gstlal_far", "mbta_far", "pycbc_far"):
        cols[k] = ev[k][:]
    f.close()
    m1, m2, z = cols["mass1_source"], cols["mass2_source"], cols["z"]
    mt = m1 + m2
    N_gen = int(at["total_generated"])
    T_tot = float(at["total_analysis_time"])

    res = {"far_thr_per_yr": FAR_THR, "mass_edges": MASS_EDGES.tolist(),
           "neff_min": NEFF_MIN, "runs": {}, "validation": {}}

    # ---- G0: month table consistency: T_total == sum of month windows ----
    t_md = sum(hi - lo for lo, hi, _ in MONTHS)
    verify("O4ab T_total == sum(month windows) [rel]",
           abs(t_md - T_tot) / T_tot, tol=1e-9)
    n_md = sum(n for _, _, n in MONTHS)
    verify("O4ab N_gen == sum(month N) [rel]", abs(n_md - N_gen) / N_gen, tol=1e-9)

    # ---- G1: cosmology via the release's own dL(z) column ----
    zg = np.geomspace(1e-6, float(z.max()) * 1.001, 6000)
    dlg = COSMO.luminosity_distance(zg).to(u.Mpc).value
    sub = np.random.default_rng(1).choice(len(z), 200_000, replace=False)
    devdl = np.abs(np.interp(z[sub], zg, dlg) / cols["luminosity_distance"][sub] - 1)
    verify("O4ab cosmology dL(z) FlatLCDM(67.9,0.3065) [max rel]",
           float(devdl.max()), tol=1e-3)

    # ---- G2: weights column == md month table reconstruction ----
    mi = np.digitize(cols["time_geocenter"], [m[0] for m in MONTHS] + [MONTHS[-1][1]]) - 1
    ok = (mi >= 0) & (mi < len(MONTHS))
    verify("O4ab all events inside month windows [frac outside]",
           float((~ok).mean()), tol=1e-12)
    w_pred = np.array([(hi - lo) / n / (T_tot / N_gen) for lo, hi, n in MONTHS])
    devw = np.abs(cols["weights"] / w_pred[mi] - 1)
    verify("O4ab weights column vs month table [max rel]", float(devw.max()), tol=1e-3)

    # ---- G3: z draw law ~ dVc/dz (no dilation), fit norm + k, gate k~0 ----
    dvdz = COSMO.differential_comoving_volume(zg).to(u.Gpc**3 / u.sr).value * 4 * np.pi
    bar = dvdz / (1 + zg)                       # dVc/dz (1+z)^-1  [Gpc^3]
    lndv = np.interp(z, zg, np.log(dvdz))
    A = np.column_stack([np.ones_like(z), np.log1p(z)])
    (a0, k0), *_ = np.linalg.lstsq(A, cols["lnpdraw_z"] - lndv, rcond=None)
    rz = cols["lnpdraw_z"] - lndv - a0 - k0 * np.log1p(z)
    zm = z >= 0.05
    print(f"        (z law: norm a={a0:.4f} -> Vc-like {np.exp(-a0):.1f} Gpc^3, "
          f"k={k0:.5f}; full-range max resid {float(np.abs(rz).max()):.3f}; "
          f"z<0.05 carries {int((~zm).sum())} inj)")
    verify("O4ab z law k(1+z) ~ 0 [|k|]", abs(float(k0)), tol=1e-2)
    verify("O4ab z law resid (z>=0.05) [max |ln|]",
           float(np.abs(rz[zm]).max()), tol=1e-2)

    # ---- G4: m2|m1 conditional power law with closed-form normalization ----
    A2 = np.column_stack([np.ones_like(m1), np.log(m2), np.log(m1)])
    (c2, b2, g2), *_ = np.linalg.lstsq(A2, cols["lnpdraw_mass2_source_GIVEN_mass1_source"],
                                       rcond=None)
    mmin = float(m2.min())
    pred2 = b2 * np.log(m2) + np.log(b2 + 1) - np.log(m1**(b2 + 1) - mmin**(b2 + 1))
    dev2 = np.abs(cols["lnpdraw_mass2_source_GIVEN_mass1_source"] - pred2)
    print(f"        (m2|m1: b={b2:.4f}, mmin={mmin:.4f}, median resid "
          f"{float(np.median(dev2)):.2e})")
    verify("O4ab m2|m1 law [max |ln|]", float(dev2.max()), tol=2e-2)
    verify("O4ab m2|m1 law [median |ln|]", float(np.median(dev2)), tol=1e-3)

    # ---- G5: m1 marginal reconstructed on a grid from the file's own values,
    #      verified on a held-out half + must integrate to 1 (coverage) ----
    order = np.argsort(m1)
    m1s, lp1s = m1[order], cols["lnpdraw_mass1_source"][order]
    half = np.zeros(len(m1s), bool); half[::2] = True
    grid_m, grid_lp = m1s[half], lp1s[half]
    lp_hat = np.interp(m1s[~half], grid_m, grid_lp)
    dev1 = np.abs(lp_hat - lp1s[~half])
    print(f"        (m1 marginal grid: {half.sum()} nodes, held-out median "
          f"resid {float(np.median(dev1)):.2e})")
    verify("O4ab m1 marginal grid, held-out [max |ln|]", float(dev1.max()), tol=2e-2)
    verify("O4ab m1 marginal grid, held-out [median |ln|]",
           float(np.median(dev1)), tol=1e-3)
    mg = np.geomspace(grid_m[0], grid_m[-1], 40_000)
    p1g = np.exp(np.interp(mg, grid_m, grid_lp))
    norm1 = float(np.trapezoid(p1g, mg))
    verify("O4ab m1 marginal integrates to 1 [|int-1|]", abs(norm1 - 1.0), tol=5e-3)

    # ---- P(B): integrate p1(m1) * F(m1,B) with F the closed-form conditional
    #      mass integral; MC cross-check with inverse-CDF sampling ----
    def F_bin(m1v, lo, hi):
        l = np.maximum(mmin, lo - m1v)
        u_ = np.minimum(m1v, hi - m1v)
        good = u_ > l
        out = np.zeros_like(m1v)
        out[good] = ((u_[good]**(b2 + 1) - l[good]**(b2 + 1))
                     / (m1v[good]**(b2 + 1) - mmin**(b2 + 1)))
        return out

    PB = np.array([float(np.trapezoid(p1g * F_bin(mg, MASS_EDGES[k], MASS_EDGES[k + 1]), mg))
                   for k in range(len(MASS_EDGES) - 1)]) / norm1
    # MC cross-check, gated in units of the MC binomial error (small bins
    # have P(B) down to ~1e-4 -> percent-level MC noise is expected)
    rng = np.random.default_rng(20260812)
    NMC = 16_000_000
    cdf1 = np.concatenate([[0.0], np.cumsum(0.5 * (p1g[1:] + p1g[:-1]) * np.diff(mg))])
    cdf1 /= cdf1[-1]
    m1mc = np.interp(rng.random(NMC), cdf1, mg)
    u2 = rng.random(NMC)
    m2mc = (u2 * (m1mc**(b2 + 1) - mmin**(b2 + 1)) + mmin**(b2 + 1)) ** (1 / (b2 + 1))
    mtmc = m1mc + m2mc
    PB_mc = np.array([((mtmc >= MASS_EDGES[k]) & (mtmc < MASS_EDGES[k + 1])).mean()
                      for k in range(len(MASS_EDGES) - 1)])
    sig = np.sqrt(PB * (1 - PB) / NMC)
    zsc = float(np.abs((PB_mc - PB) / sig).max())
    print(f"        (P(B) quad vs MC: max rel dev {float(np.abs(PB_mc/PB-1).max()):.2%}, "
          f"max |z| = {zsc:.2f} MC-sigma; P(B) = {np.array2string(PB, precision=5)})")
    verify("O4ab P(B) quadrature vs MC [max |z|/5sigma]", zsc / 5.0, tol=1.0)
    res["validation"]["O4ab_gates"] = dict(
        dL_maxrel=float(devdl.max()), weights_maxrel=float(devw.max()),
        z_k=float(k0), m2_b=float(b2), m1_norm=norm1, PB=PB.tolist())

    # ---- importance ratio (mass factors cancel; z target = bar) ----
    r = np.exp(np.interp(z, zg, np.log(bar)) - cols["lnpdraw_z"])   # Gpc^3
    wr = cols["weights"] * r
    pipelines = {"cWB": "cwb-bbh_far", "GstLAL": "gstlal_far",
                 "MBTA": "mbta_far", "PyCBC": "pycbc_far"}

    # ---- V2 analog: full-set per-pipeline bin-conditioned VT under release
    #      conventions (full T via weights, no HL cut), FAR<1/yr and IFAR>=100.
    #      No published O4 per-pipeline VT table to anchor against; recorded
    #      for the O3-magnitude cross-check (O4 >= O3 expected at 100-200). ----
    Pc = {(lo, hi): float(((mtmc >= lo) & (mtmc < hi)).mean())
          for lo, hi in ((100, 200), (200, 400))}
    v2 = {}
    for thr_name, thr in (("far1", 1.0), ("ifar100", 0.01)):
        for p, c in pipelines.items():
            det = cols[c] < thr
            for lo, hi in ((100, 200), (200, 400)):
                sel = det & (mt >= lo) & (mt < hi)
                vt = T_tot / YR * wr[sel].sum() / (N_gen * Pc[(lo, hi)])
                v2[f"{p}_Mtot{lo}-{hi}_{thr_name}"] = vt
    res["validation"]["O4ab_fullset_VT"] = v2
    print("[V2] O4ab full-set bin-conditioned VT [Gpc^3 yr], release conventions: " +
          ", ".join(f"{k}={v:.2f}" for k, v in v2.items() if "far1" in k))
    print("     (IFAR>=100: " +
          ", ".join(f"{k}={v:.2f}" for k, v in v2.items() if "ifar100" in k) + ")")

    if validate_only:
        json.dump(res, open(f"{HERE}/vt_pipelines_gwtc5.json", "w"), indent=1)
        print(f"[done validate-only] -> {HERE}/vt_pipelines_gwtc5.json")
        return

    # ---- S: MADGRAV HL-segment restriction, per run; exposure enters via
    #      the month weights (see module docstring) ----
    gps = cols["time_geocenter"]
    for run in ("O4a", "O4b"):
        iv, T_ours = our_segments(run)
        lo_r, hi_r = RUN_RANGE[run]
        # injections were explicitly removed outside the official run range
        # (release md) -> clip our segments to it for the injection sum and
        # scale back to full wall-clock exposure under stationarity.
        # Measured 2026-08-12: O4a 0% clipped; O4b 3.60 d of 114.14 d (3.15%,
        # all pre-official-start early-Apr-2024 segments).
        ivc = np.column_stack([np.clip(iv[:, 0], lo_r, None),
                               np.clip(iv[:, 1], None, hi_r)])
        ivc = ivc[ivc[:, 1] > ivc[:, 0]]
        tau_clip = float((ivc[:, 1] - ivc[:, 0]).sum()) / YR
        scale = T_ours / tau_clip
        if scale > 1.10:
            raise SystemExit(f"{run}: >10% of segment time outside official "
                             f"injection range - stationarity scaling not defensible")
        ins = in_segments(gps, ivc)
        print(f"[S] {run}: {int(ins.sum())}/{len(gps)} recorded injections in our "
              f"HL segments (clipped to official range); tau_segs full = "
              f"{T_ours:.4f} yr, in-range = {tau_clip:.4f} yr, "
              f"exposure scale = {scale:.4f}")
        run_out = {"T_ours_yr": T_ours, "tau_inrange_yr": tau_clip,
                   "exposure_scale": scale,
                   "n_inj_in_segments": int(ins.sum()), "pipelines": {}}
        bins = np.digitize(mt, MASS_EDGES) - 1
        for p, c in pipelines.items():
            det = (cols[c] < FAR_THR) & ins
            per_bin_vt, per_bin_neff = [], []
            for k in range(len(MASS_EDGES) - 1):
                sel = det & (bins == k)
                w = wr[sel]
                vt = (scale * T_tot / YR * w.sum() / (N_gen * PB[k])
                      if PB[k] > 0 else np.nan)
                per_bin_vt.append(vt); per_bin_neff.append(neff(w))
            groups = widen_bins(MASS_EDGES, per_bin_neff)
            run_out["pipelines"][p] = dict(vt_gpc3yr=per_bin_vt, neff=per_bin_neff,
                                           widened_groups=groups)
        res["runs"][run] = run_out

    json.dump(res, open(f"{HERE}/vt_pipelines_gwtc5.json", "w"), indent=1)
    print(f"[done] -> {HERE}/vt_pipelines_gwtc5.json")


if __name__ == "__main__":
    main(validate_only="--validate-only" in sys.argv)
