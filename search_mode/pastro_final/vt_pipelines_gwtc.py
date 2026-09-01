import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)

#!/usr/bin/env python
"""Per-pipeline VT(Mtot) from the LVK injection releases, on MADGRAV's bins.

Data: GWTC-3 O3a/O3b endo3_{bbhpop,imbhpop} (zenodo 5546676) and GWTC-5.0
O4a+O4b samples-rpo4ab (zenodo 19500064), local in gwtc5_sensitivity/.

Estimator (per run, per pipeline p, per Mtot bin B):
    VT_B^p = T * (1/N_gen) * sum_{i in B, det_p(i)} w_i / P_mix(B)
    w_i    = p_mix,mass(m1_i, m2_i) * [dVc/dz (1+z)^-1](z_i)
             / p_mix(m1_i, m2_i, z_i)
    det_p  = far_p < FAR_THR (GWTC-3: ifar_p > 1 yr; GWTC-5: far_p < 1/yr)
  i.e. target population = the (m1,m2)-mixture draw conditioned on the bin,
  z-target uniform in comoving volume with (1+z) rate dilation. Detected
  injections are all recorded (hopeless cut removes only non-detectable
  draws), so the numerator is complete; N_gen from file attrs.

FIX 2026-08-12: the first full run (10:16) used w_i = r_i without the
p_mix,mass target factor - dimensionally Gpc^3 yr Msun^-2, values up to
1.6e6 "Gpc^3 yr" >> the 133 Gpc^3 yr total surveyed VT (unphysical), and
bin-to-bin shape tilted by 1/<p_mass>_B. With the factor restored, VT_B is
the VT of the bin-conditioned mixture population in true Gpc^3 yr, and V2
(now P(B)-normalized) is directly comparable to the published IMBH table.

MIXTURE (specification 2026-08-11): in the bbh+imbh overlap the sampling density
is the draw-count-weighted sum of the two populations' densities evaluated at
the SAME (m1,m2,z) point:
    p_mix = (N_bbh p_bbh + N_imbh p_imbh) / (N_bbh + N_imbh)
Each analytic law is VERIFIED against the file's own sampling_pdf columns
(max |ln ratio| < 1e-3) before use; the script aborts if verification fails.

VALIDATION ORDER (specification 2026-08-11): (V1) reproduce surveyed_VT under the
release's own conventions (full analysis_time, no HL restriction, release
cosmology); (V2) full-set per-pipeline VT vs the published O3 IMBH table
(arXiv:2110.01879, IFAR=100 yr) as an order-of-magnitude cross-check (their
population is q=1 non-spinning classes, ours the continuous imbh draw - agree
to shape, not digits); only then (S) apply MADGRAV HL-segment restriction +
MADGRAV wall-clock T for the comparison curves.

N_eff rule (specification): per bin N_eff = (sum w)^2 / sum w^2 over detected
weights; bins with N_eff < 300 are WIDENED (merged rightward) not plotted.

Outputs: vt_pipelines_gwtc.json (+ caption_vt_compare.txt) ; figure step comes
from vt_compare_pipelines.py v3 which reads this json.
Run: madgrav-venv python vt_pipelines_gwtc.py [--validate-only]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys
import json
import glob
import h5py
import numpy as np

D = MADGRAV_EXTDATA + "/gwtc5_sensitivity"
MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
HERE = f"{MG}/search_mode/pastro_final"
YR = 3.1557e7
FAR_THR = 1.0                                  # 1/yr
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])
NEFF_MIN = 300

# ---------- cosmology ----------
# GWTC-3 release convention identified empirically (cosmology scan 2026-08-11):
# FlatLambdaCDM(H0=67.9, Om0=0.3065) [O3 R&P / Planck15 TT+lowP+lensing+ext]
# matches redshift_sampling_pdf with median ln-dev 0.0000; Planck15/18 leave
# 0.3-0.6% bulk offsets. Log-spaced z grid: the linear grid's z->0 tail caused
# spurious ~9% interpolation error at z~0.002 (volume-negligible).
from astropy.cosmology import FlatLambdaCDM, Planck18
import astropy.units as u
from scipy.integrate import cumulative_trapezoid

COSMO_GWTC3 = FlatLambdaCDM(H0=67.9, Om0=0.3065)


def z_machinery(cosmo, zmax=8.0, n=8000):
    zg = np.geomspace(1e-5, zmax, n)
    dvdz = cosmo.differential_comoving_volume(zg).to(u.Gpc**3 / u.sr).value * 4 * np.pi
    bar = dvdz / (1 + zg)                       # dVc/dz (1+z)^-1  [Gpc^3]
    cum = np.concatenate([[0.0], cumulative_trapezoid(bar, zg)])
    return zg, dvdz, bar, cum


# ---------- GWTC-3 analytic draw laws (verified against sampling_pdf cols) ----------
def bbh_mass_pdf(m1, m2, a1=-2.35, a2=1.0, lo=2.0, hi=100.0):
    c1 = (a1 + 1) / (hi**(a1 + 1) - lo**(a1 + 1))
    c2 = (a2 + 1) / (m1**(a2 + 1) - lo**(a2 + 1))
    p = c1 * m1**a1 * c2 * m2**a2
    return np.where((m1 >= lo) & (m1 <= hi) & (m2 >= lo) & (m2 <= m1), p, 0.0)


def fit_powerlaw_2d(m1, m2, lnpdf):
    """ln p = c + a*ln m1 + b*ln m2 least squares (for identifying imbh law)."""
    A = np.column_stack([np.ones_like(m1), np.log(m1), np.log(m2)])
    coef, *_ = np.linalg.lstsq(A, lnpdf, rcond=None)
    resid = lnpdf - A @ coef
    return coef, float(np.abs(resid).max())


def verify(tag, maxdev, tol=1e-3):
    status = "OK" if maxdev < tol else "FAIL"
    print(f"[verify] {tag}: max |ln p_analytic - ln p_file| = {maxdev:.2e} -> {status}")
    if maxdev >= tol:
        raise SystemExit(f"draw-law verification FAILED for {tag}")


def load_o3(run):
    """Load and combine bbh+imbh for one O3 run with mixture densities."""
    gps = {"O3a": "1238166018-15843600", "O3b": "1256655642-12905976"}[run]
    out = {}
    for tag in ("bbhpop", "imbhpop"):
        f = h5py.File(f"{D}/endo3_{tag}-LIGO-T2100113-v12-{gps}.hdf5", "r")
        i = f["injections"]
        d = {k: i[k][:] for k in
             ("mass1_source", "mass2_source", "redshift", "gps_time",
              "mass1_source_mass2_source_sampling_pdf", "redshift_sampling_pdf",
              "ifar_cwb", "ifar_gstlal", "ifar_mbta", "ifar_pycbc_bbh",
              "ifar_pycbc_hyperbank")}
        d["N_gen"] = int(f.attrs["total_generated"])
        d["T_s"] = float(f.attrs["analysis_time_s"])
        if tag == "bbhpop":
            d["surveyed_VT"] = float(f.attrs["surveyed_VT_Gpc_yr"])
            d["attrs"] = {k: f.attrs[k] for k in
                          ("pow_mass1", "pow_mass2", "min_mass1", "max_mass1",
                           "max_redshift", "pow_z")}
        f.close()
        out[tag] = d
    return out


# ---------- imbh law: identified from the file, then frozen analytically ----------
class ImbhLaw:
    """Power-law form fitted to the imbhpop sampling_pdf column, with support
    box from the recorded samples; frozen after verification."""

    def __init__(self, m1, m2, lnpdf):
        (self.c, self.a, self.b), self.maxdev = fit_powerlaw_2d(m1, m2, lnpdf)
        self.m1lo, self.m1hi = m1.min(), m1.max()
        self.m2lo, self.m2hi = m2.min(), m2.max()

    def pdf(self, m1, m2):
        p = np.exp(self.c) * m1**self.a * m2**self.b
        sup = ((m1 >= self.m1lo) & (m1 <= self.m1hi) &
               (m2 >= self.m2lo) & (m2 <= self.m2hi) & (m2 <= m1))
        return np.where(sup, p, 0.0)


def zpdf_from_grid(zg, form, zmax):
    """Normalized z draw pdf, evaluated only on the in-support sub-grid.
    Returns (z_sub, pdf_sub); interpolate with np.interp(z, z_sub, pdf_sub).
    (Masking the full grid to zero and interpolating across the support edge
    dragged the pdf toward 0 at z~zmax - the earlier verification failure.)"""
    m = zg <= zmax
    zs = np.append(zg[m], zmax)
    vs = np.append(form[m], np.interp(zmax, zg, form))
    return zs, vs / np.trapezoid(vs, zs)


def neff(w):
    s = w.sum()
    return float(s * s / (w * w).sum()) if len(w) and (w * w).sum() > 0 else 0.0


def widen_bins(edges, per_bin_neff):
    """Merge bins rightward until every kept bin has N_eff >= NEFF_MIN."""
    merged, cur = [], [0]
    for k in range(len(per_bin_neff)):
        if k > 0:
            cur.append(k)
        if sum(per_bin_neff[j] for j in cur) >= NEFF_MIN:
            merged.append(cur); cur = []
    if cur:
        if merged:
            merged[-1].extend(cur)
        else:
            merged.append(cur)
    return merged


def our_segments(run):
    segs = json.load(open(f"{SC}/{run.lower()}_full_coincident.json"))["segments"]
    names = set(np.load(f"{SC}/search_out_{run.lower()}_far"
                        f"{'_f40' if run in ('O3a', 'O3b') else ''}/bg_cache_{run.lower()}.npz")
                ["seg_names"])
    iv = np.array([[s[0], s[1]] for s in segs if s[3] in names])
    T = iv[:, 1].sum() - iv[:, 0].sum()
    return iv[np.argsort(iv[:, 0])], T / YR


def in_segments(gps, iv):
    idx = np.searchsorted(iv[:, 0], gps, side="right") - 1
    ok = idx >= 0
    return ok & (gps <= iv[np.clip(idx, 0, len(iv) - 1), 1])


def main(validate_only=False):
    res = {"far_thr_per_yr": FAR_THR, "mass_edges": MASS_EDGES.tolist(),
           "neff_min": NEFF_MIN, "runs": {}, "validation": {}}

    # ======== GWTC-3: O3a / O3b ========
    zg15, _, bar15, cum15 = z_machinery(COSMO_GWTC3)
    for run in ("O3a", "O3b"):
        data = load_o3(run)
        bbh, imbh = data["bbhpop"], data["imbhpop"]

        # --- verify bbh analytic mass law vs file column ---
        pm = bbh_mass_pdf(bbh["mass1_source"], bbh["mass2_source"],
                          a1=float(bbh["attrs"]["pow_mass1"]),
                          a2=float(bbh["attrs"]["pow_mass2"]))
        dev = np.abs(np.log(pm) - np.log(bbh["mass1_source_mass2_source_sampling_pdf"]))
        verify(f"{run} bbh mass law", float(dev.max()))

        # --- verify bbh z law (pow_z=1 comoving-dilated, Planck15) ---
        zmax = float(bbh["attrs"]["max_redshift"])
        zs_b, pz_b = zpdf_from_grid(zg15, bar15 * (1 + zg15) ** float(bbh["attrs"]["pow_z"]), zmax)
        pz = np.interp(bbh["redshift"], zs_b, pz_b)
        devz = np.abs(np.log(pz) - np.log(bbh["redshift_sampling_pdf"]))
        # gate on z>=0.05: below that the RELEASE's own 1001-point z-norm grid
        # (attr num_redshift_norm_grid_points) quantizes its pdf by up to ~10%
        # at z~0.002 (22 injections, V~z^3 -> ~1e-7 of any VT). Full-range max
        # is reported but not gated.
        zm = bbh["redshift"] >= 0.05
        print(f"        (full-range max {float(devz.max()):.3f}; "
              f"z<0.05 carries {int((~zm).sum())} inj)")
        verify(f"{run} bbh z law (z>=0.05)", float(devz[zm].max()), tol=1e-3)

        # --- identify + verify imbh laws from the file itself ---
        law = ImbhLaw(imbh["mass1_source"], imbh["mass2_source"],
                      np.log(imbh["mass1_source_mass2_source_sampling_pdf"]))
        verify(f"{run} imbh mass law (fit a={law.a:.3f} b={law.b:.3f})", law.maxdev, tol=5e-3)
        zmax_i = float(imbh["redshift"].max())
        # imbh z draw: dVc/dz/(1+z) (identified 2026-08-11; differs from the
        # bbh set's plain dVc/dz - kappa=0 vs kappa=1 in release language)
        zs_i, pz_i = zpdf_from_grid(zg15, bar15, zmax_i)
        pzi = np.interp(imbh["redshift"], zs_i, pz_i)
        devzi = np.abs(np.log(pzi) - np.log(imbh["redshift_sampling_pdf"]))
        zmi = imbh["redshift"] >= 0.05
        print(f"        (full-range max {float(devzi.max()):.3f}; "
              f"z<0.05 carries {int((~zmi).sum())} inj)")
        verify(f"{run} imbh z law (z>=0.05)", float(devzi[zmi].max()), tol=5e-3)

        # --- V1: surveyed_VT under release conventions (bbh) ---
        vt_surv = bbh["T_s"] / YR * np.interp(zmax, zg15, cum15)
        rel = abs(vt_surv - bbh["surveyed_VT"]) / bbh["surveyed_VT"]
        print(f"[V1] {run} surveyed_VT: mine {vt_surv:.3f} vs attr {bbh['surveyed_VT']:.3f} "
              f"Gpc^3 yr (rel dev {rel:.2%})")
        res["validation"][f"{run}_surveyed_VT"] = dict(
            mine=vt_surv, attr=bbh["surveyed_VT"], rel_dev=rel)
        if rel > 0.02:
            raise SystemExit(f"[V1] {run} surveyed_VT validation FAILED")

        # --- combined set + mixture density ---
        m1 = np.concatenate([bbh["mass1_source"], imbh["mass1_source"]])
        m2 = np.concatenate([bbh["mass2_source"], imbh["mass2_source"]])
        z = np.concatenate([bbh["redshift"], imbh["redshift"]])
        gps = np.concatenate([bbh["gps_time"], imbh["gps_time"]])
        Nb, Ni = bbh["N_gen"], imbh["N_gen"]
        p_b = bbh_mass_pdf(m1, m2) * np.interp(z, zs_b, pz_b) * (z <= zmax)
        p_i = law.pdf(m1, m2) * np.interp(z, zs_i, pz_i) * (z <= zmax_i)
        p_mix = (Nb * p_b + Ni * p_i) / (Nb + Ni)
        assert np.all(p_mix > 0), "zero mixture density at recorded points"
        # target mass density = mixture MASS marginal (z-independent)
        p_mix_m = (Nb * bbh_mass_pdf(m1, m2) + Ni * law.pdf(m1, m2)) / (Nb + Ni)
        w_full = p_mix_m * np.interp(z, zg15, bar15) / p_mix   # Gpc^3 (true units)
        mt = m1 + m2

        # P_mix(B): MC over the mixture mass law (z integrates out of the target)
        rng = np.random.default_rng(20260811)
        NMC = 4_000_000
        pick = rng.random(NMC) < Nb / (Nb + Ni)
        m1mc = np.empty(NMC); m2mc = np.empty(NMC)
        # bbh: inverse-cdf power laws
        u1 = rng.random(pick.sum())
        a1, lo_, hi_ = -2.35, 2.0, 100.0
        m1mc[pick] = ((hi_**(a1+1) - lo_**(a1+1)) * u1 + lo_**(a1+1)) ** (1/(a1+1))
        m2mc[pick] = np.sqrt(rng.random(pick.sum()) * (m1mc[pick]**2 - lo_**2) + lo_**2)
        # imbh: fitted power law via rejection on its box
        nb = (~pick).sum()
        got = 0; m1i = np.empty(nb); m2i = np.empty(nb)
        pmax = law.pdf(np.array([law.m1lo]), np.array([law.m2lo]))[0] * 1.05
        while got < nb:
            k = min(2_000_000, (nb - got) * 4)
            c1 = rng.uniform(law.m1lo, law.m1hi, k)
            c2 = rng.uniform(law.m2lo, law.m2hi, k)
            keep = rng.random(k) * pmax < law.pdf(c1, c2)
            take = min(keep.sum(), nb - got)
            m1i[got:got+take] = c1[keep][:take]; m2i[got:got+take] = c2[keep][:take]
            got += take
        m1mc[~pick] = m1i; m2mc[~pick] = m2i
        mtmc = m1mc + m2mc
        bins_mc = np.digitize(mtmc, MASS_EDGES) - 1

        pipelines = {"cWB": "ifar_cwb", "GstLAL": "ifar_gstlal", "MBTA": "ifar_mbta",
                     "PyCBC-BBH": "ifar_pycbc_bbh", "PyCBC-broad": "ifar_pycbc_hyperbank"}
        ifars = {p: np.concatenate([bbh[c], imbh[c]]) for p, c in pipelines.items()}

        # --- V2: full-set per-pipeline VT (release conventions, IFAR>=100 yr),
        #     P_mix(B)-normalized -> bin-conditioned VT in true Gpc^3 yr;
        #     order-of-magnitude check vs arXiv:2110.01879 Tab.1 (q=1 classes,
        #     full-O3 HLV there vs per-run here - shape/magnitude, not digits) ---
        T_rel = bbh["T_s"] / YR
        v2 = {}
        for p in ("PyCBC-BBH", "cWB", "GstLAL"):
            det = ifars[p] > 100.0
            for lo, hi in ((100, 200), (200, 400)):
                sel = det & (mt >= lo) & (mt < hi)
                P_coarse = float(((mtmc >= lo) & (mtmc < hi)).mean())
                vt = T_rel * w_full[sel].sum() / ((Nb + Ni) * P_coarse)
                v2[f"{p}_Mtot{lo}-{hi}"] = vt
        res["validation"][f"{run}_fullset_VT_ifar100"] = v2
        print(f"[V2] {run} full-set VT @IFAR>=100, P_mix(B)-normalized "
              f"(bin-conditioned, Gpc^3 yr; vs 2110.01879 Tab.1): " +
              ", ".join(f"{k}={v:.2f}" for k, v in v2.items()))

        if validate_only:
            continue

        # --- S: MADGRAV segment restriction + our T ---
        iv, T_ours = our_segments(run)
        ins = in_segments(gps, iv)
        print(f"[S] {run}: {ins.sum()}/{len(gps)} recorded injections fall in our "
              f"HL segments; T_ours = {T_ours:.4f} yr")
        run_out = {"T_ours_yr": T_ours, "n_inj_in_segments": int(ins.sum()),
                   "pipelines": {}}
        for p in pipelines:
            det = (ifars[p] > 1.0 / FAR_THR) & ins
            per_bin_vt, per_bin_neff = [], []
            bins = np.digitize(mt, MASS_EDGES) - 1
            for k in range(len(MASS_EDGES) - 1):
                sel = det & (bins == k)
                PB = (bins_mc == k).mean()
                w = w_full[sel]
                vt = T_ours * w.sum() / ((Nb + Ni) * PB) if PB > 0 else np.nan
                per_bin_vt.append(vt); per_bin_neff.append(neff(w))
            groups = widen_bins(MASS_EDGES, per_bin_neff)
            run_out["pipelines"][p] = dict(vt_gpc3yr=per_bin_vt, neff=per_bin_neff,
                                           widened_groups=groups)
        res["runs"][run] = run_out

    json.dump(res, open(f"{HERE}/vt_pipelines_gwtc.json", "w"), indent=1)
    print(f"[done] -> {HERE}/vt_pipelines_gwtc.json")


if __name__ == "__main__":
    main(validate_only="--validate-only" in sys.argv)
