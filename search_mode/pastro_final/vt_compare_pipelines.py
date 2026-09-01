#!/usr/bin/env python
"""MADGRAV blind-search VT vs other pipelines, O3+O4 — COMOVING comparison figure.

Motivation: the paper's vt_search.py figure is an explicitly-labelled EUCLIDEAN
in-band proxy. Published benchmark VTs are COMOVING, so a cross-pipeline overlay
must convert our horizon volumes to comoving V_max = int_0^z(Dh) dV_c/(1+z)
(FlatLambdaCDM 67.9/0.3065, the GWTC-3 release convention) before any curve
shares an axis with them. Efficiencies, weights,
T_obs, and the per-injection population pairing are IDENTICAL to vt_search.py;
only the per-source volume changes.

Benchmarks (hardcoded, sources in comments):
  * LVK O3 IMBH search VT(Mtot), PyCBC/cWB/GstLAL, IFAR=100 yr, q=1 non-spinning
    (arXiv:2105.15120; VT table arXiv:2110.01879 Tab. 1). Full-O3 HLV livetime.
  * O3 BBH-search sensitive-volume band, 6 pipelines (cWB/GstLAL/MBTA/PyCBC-BBH/
    PyCBC-Broad/Aframe) at FAR=1/yr, equal-mass line, read off arXiv:2403.18661
    (same digitisation as lr_cascade/vt_comoving_overlay.py). Volumes (Gpc^3)
    are converted to VT by multiplying with MADGRAV's ANALYZED O3 time — an
    equal-livetime framing that is apples-to-apples with the MADGRAV O3 curve
    (NOT with the O3+O4 total, which accumulates more time; labelled).
O4 anchors are O3-only in THIS figure and the MADGRAV O4 contribution keeps
its RELATIVE caveat (reference-ASD low-frequency bias, 9-30x at 20-60 Hz vs
O3a). NOTE: per-pipeline O4a/O4b VT(Mtot) is DERIVABLE at matched FAR from the
GWTC-5.0 sensitivity-injection release (Zenodo 10.5281/zenodo.19500064,
2026-04-10; cwb-bbh/gstlal/mbta/pycbc *_far columns + lnpdraw reweighting +
total_analysis_time), local copy $MADGRAV_EXTDATA/gwtc5_sensitivity/;
same for O3 from the GWTC-3 release (zenodo 5546676, DIFFERENT schema).

AMENDMENT 2026-08-15 (logged; supersedes the 2026-08-11 figure, archived in
campaign/archive/vt_compare_pre_relabel/): the MADGRAV numerator is now read
from vt_relabel_comoving.json (vt_comoving_srcframe_gpc3yr) - the SAME
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)

numerator as the four-epoch figure of record - instead of being recomputed
here. This carries the three 2026-08-12 corrections: (a) float64 masks (the
float32 underflow guard in horizons_comoving() culled 11.5% of the sig bank,
biasing low-mass VT high); (b) O4 horizons release-RELABELED (as-run
outcomes, labels corrected to release run-median PSDs; truncated balls) so
the O4 contribution is no longer "relative"; (c) SOURCE-frame Mtot axis,
matching the comparators. The legacy in-script recompute is kept ONLY as
--legacy-asrun (float32-guard, detector-frame, as-run O4) for provenance.

Run: nice -n 10 madgrav-venv python vt_compare_pipelines.py [--plot-only|--legacy-asrun]
  default      -> numerator from vt_relabel_comoving.json (fast, no bank I/O)
  --plot-only  re-renders the figure from the saved json
  --legacy-asrun  the pre-2026-08-15 recompute path (writes *_legacy.json only)
Outputs: vt_compare_pipelines.json + figures/vt_compare/vt_compare_o3o4.{pdf,png}
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys
import json
import numpy as np

MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
HERE = f"{MG}/search_mode/pastro_final"
FIGDIR = f"{MG}/figures/vt_compare"
# SM_VT_SUF selects the criterion supplying the MADGRAV numerator (same switch as
# fig_vt_frames.py / fig_fourepoch_ratio.py). "" = 2026-08-19 successor build (historical);
# "_x1cnnadoptveto" = adopted criterion frozen 2026-08-31. Outputs carry the suffix.
SUF = os.environ.get("SM_VT_SUF", "")
YR = 3.1557e7

# --- benchmark data ---
# LVK O3 IMBH search (arXiv:2105.15120 / 2110.01879 Tab.1): <VT> Gpc^3 yr,
# IFAR=100 yr, q=1 non-spinning, full-O3 HLV. Continues to Mtot 600/800 off-axis.
LVK_M = [120, 150, 200, 400]
LVK = {"PyCBC": [11.5, 12.0, 14.8, 4.8],
       "cWB": [8.9, 10.0, 12.8, 4.6],
       "GstLAL": [8.2, 7.4, 10.3, 4.3]}
# O3 BBH-search band, 6 pipelines, sensitive volume Gpc^3 at FAR=1/yr,
# equal-mass (arXiv:2403.18661; digitisation from lr_cascade/vt_comoving_overlay.py).
MTOT_B = (2 * np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])).tolist()
PB = {"cWB": [0.2, 0.7, 1.3, 2.6, 4.1, 6.0, 8.3, 10.1, 11.8, 13.3, 14.5, 15.2, 15.8],
      "GstLAL": [0.5, 1.6, 3.0, 5.0, 7.1, 10.0, 12.5, 14.8, 16.6, 18.4, 19.8, 20.6, 21.0],
      "MBTA": [0.5, 1.4, 2.3, 3.5, 4.9, 6.7, 9.0, 11.1, 13.0, 14.8, 16.0, 16.4, 16.4],
      "PyCBC-BBH": [0.5, 1.5, 2.5, 4.0, 7.0, 9.5, 12.0, 14.1, 15.6, 16.8, 17.7, 18.0, 18.1],
      "PyCBC-Broad": [0.5, 1.3, 2.2, 4.5, 5.5, 7.1, 10.1, 10.1, 11.1, 12.4, 13.4, 13.7, 13.9],
      "Aframe": [0.6, 1.7, 3.1, 5.2, 7.4, 10.4, 13.6, 16.5, 18.9, 20.6, 22.0, 23.0, 23.5]}


def compute():
    import vt_search as vs                  # reuses ip, banks, edges, is_um logic
    # COSMOLOGY HARMONIZED (specification 2026-08-11): FlatLambdaCDM(67.9, 0.3065)
    # = the GWTC-3 release convention identified in vt_pipelines_gwtc.py, so
    # MADGRAV curves and release-derived pipeline curves share cosmology AND
    # the dVc/dz/(1+z) dilation convention on figure v3. (Was Planck18; ~1%.)
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=67.9, Om0=0.3065)

    # comoving machinery: dL(z) grid + V_max(<z) with (1+z) time-dilation
    zg = np.linspace(1e-4, 8.0, 6000)
    dl_g = cosmo.luminosity_distance(zg).to(u.Mpc).value
    dvdz = cosmo.differential_comoving_volume(zg).to(u.Mpc**3 / u.sr).value * 4 * np.pi
    integ = 0.5 * (dvdz[1:] / (1 + zg[1:]) + dvdz[:-1] / (1 + zg[:-1])) * np.diff(zg)
    vmaxz = np.concatenate([[0.0], np.cumsum(integ)])

    def vmax_comoving(dl_mpc):
        dl = np.asarray(dl_mpc, float)
        if np.any(dl > dl_g[-1]):
            raise ValueError(f"horizon dL beyond z={zg[-1]} grid: max {dl.max():.0f} Mpc")
        return np.interp(np.interp(dl, dl_g, zg), zg, vmaxz)

    def horizons_comoving(run):
        """Per-bin mean COMOVING horizon volume, split by bank population —
        same structure/selection as vt_search.horizons, volume law swapped."""
        ip = vs.ip
        asd = {d: ip.load_detector_asd_o1(f"{MG}/data/{run.lower()}_search_prep", d)
               for d in ("H1", "L1")}
        v_m = {}
        for tag, bdir in (("sig", vs.BANKS[0]), ("um", vs.BANKS[1])):
            b = ip.load_o1_signal_bank(bdir)
            mt_all, dh_all = [], []
            for i in range(len(b["H1"])):
                dref, mt = float(b["distance_mpc"][i]), float(b["total_mass"][i])
                if not (np.isfinite(dref) and np.isfinite(mt)):
                    continue
                wH = np.asarray(b["H1"][i], np.float32); wL = np.asarray(b["L1"][i], np.float32)
                if (wH ** 2).sum() <= 0 or (wL ** 2).sum() <= 0:
                    continue
                rho = float(np.hypot(ip.compute_optimal_snr(wH, asd["H1"]),
                                     ip.compute_optimal_snr(wL, asd["L1"])))
                mt_all.append(mt); dh_all.append(dref * rho / vs.RHO_TH)
            mt_all = np.array(mt_all)
            v_all = vmax_comoving(dh_all)
            bins = np.digitize(mt_all, vs.MASS_EDGES) - 1
            vm = np.full(len(vs.MIDS), np.nan)
            for k in range(len(vs.MIDS)):
                sel = bins == k
                if sel.sum():
                    vm[k] = v_all[sel].mean()
            v_m[tag] = vm
        return v_m

    T_obs = {}
    for run in vs.RUNS:
        segj = json.load(open(f"{SC}/{run.lower()}_full_coincident.json"))
        names = set(np.load(f"{SC}/search_out_{run.lower()}_far"
                            f"{'_f40' if run in ('O3a', 'O3b') else ''}/bg_cache_{run.lower()}.npz")
                    ["seg_names"])
        # WALL-CLOCK exposure (decision 2026-08-11) — matches vt_search.py;
        # ANALYZED_FRAC=0.5 remains a FAR-denominator convention only.
        T_obs[run] = sum(s[2] for s in segj["segments"] if s[3] in names) / YR

    eu = json.load(open(f"{HERE}/vt_search.json"))     # Euclidean run for cross-check
    out = {"mass_edges": vs.MASS_EDGES.tolist(), "runs": {}}
    for run in vs.RUNS:
        z = np.load(f"{HERE}/inj_scored_{run.lower()}.npz")
        w, d, mt = z["w0"], z["det_frac"], z["mtot"]
        um = vs.load_is_um(run)
        assert len(um) == len(mt), f"{run}: is_um order mismatch"
        bins = np.digitize(mt, vs.MASS_EDGES) - 1
        v_m = horizons_comoving(run)
        vt = np.full(len(vs.MIDS), np.nan)
        for k in range(len(vs.MIDS)):
            sel = bins == k
            if not sel.sum():
                continue
            vi = np.where(um[sel], v_m["um"][k], v_m["sig"][k])
            if np.any(np.isnan(vi)):
                vi = np.where(np.isnan(vi), np.nanmean(vi), vi)
            vt[k] = float((d[sel] * w[sel] * vi).sum() / w[sel].sum()) * T_obs[run] * 1e-9
        out["runs"][run] = dict(T_obs_yr=T_obs[run], vt_gpc3yr=vt.tolist())
        with np.printoptions(precision=3):
            print(f"[{run}] comoving VT[Gpc3yr]={vt}")
            print(f"        euclid/comoving ratio ="
                  f" {np.array(eu['runs'][run]['vt_gpc3yr']) / vt}")

    vt_o3 = np.nansum([out["runs"][r]["vt_gpc3yr"] for r in ("O3a", "O3b")], axis=0)
    vt_tot = np.nansum([out["runs"][r]["vt_gpc3yr"] for r in vs.RUNS], axis=0)
    allnan3 = np.all([np.isnan(out["runs"][r]["vt_gpc3yr"]) for r in ("O3a", "O3b")], axis=0)
    allnanT = np.all([np.isnan(out["runs"][r]["vt_gpc3yr"]) for r in vs.RUNS], axis=0)
    vt_o3 = np.where(allnan3, np.nan, vt_o3); vt_tot = np.where(allnanT, np.nan, vt_tot)
    T_o3 = T_obs["O3a"] + T_obs["O3b"]

    out.update(vt_o3_gpc3yr=vt_o3.tolist(), vt_total_gpc3yr=vt_tot.tolist(),
               T_o3_analyzed_yr=T_o3,
               T_total_analyzed_yr=float(sum(T_obs.values())),
               lvk_imbh=dict(mtot=LVK_M, **LVK),
               bbh_band_mtot=MTOT_B, bbh_band_v_gpc3=PB,
               caveats=["MADGRAV comoving (FlatLCDM 67.9/0.3065 = GWTC-3 release convention, (1+z) dilation); efficiency/weights as vt_search.py",
                        "MADGRAV population = injected 50/50 stellar/UM mixed-q; LVK IMBH = q=1 non-spinning",
                        "FAR mismatch: MADGRAV blind FAR<1/yr vs LVK IMBH IFAR=100 yr (100x stricter)",
                        "BBH band = sensitive volume x MADGRAV analyzed O3 time -> compare with the O3 curve",
                        "O3+O4 total: O4 horizon RELATIVE (reference-ASD low-frequency bias, 9-30x at 20-60 Hz)",
                        "per-pipeline O4 VT derivable from GWTC-5.0 injection release (Zenodo 10.5281/zenodo.19500064); not yet derived here",
                        "MADGRAV T_obs = analyzed zero-lag convention (ANALYZED_FRAC=0.5); wall-clock exposure is 2x larger"])
    return out


def from_relabel():
    """MADGRAV numerator = corrected comoving relabeled VT (source-frame bins),
    identical to fig_fourepoch_ratio.py's numerator; benchmarks unchanged."""
    rel = json.load(open(f"{HERE}/vt_relabel_comoving{SUF}.json"))
    RUNS = ("O3a", "O3b", "O4a", "O4b")
    edges = np.array(rel["mass_edges"], float)
    out = {"mass_edges": edges.tolist(), "runs": {}, "numerator_source":
           f"vt_relabel_comoving{SUF}.json:vt_comoving_srcframe_gpc3yr (amendment 2026-08-15)",
           "cosmology": rel["cosmology"]}
    T_obs = {}
    for run in RUNS:
        R = rel["runs"][run]
        vt = np.array([np.nan if v is None else v for v in R["vt_comoving_srcframe_gpc3yr"]], float)
        T_obs[run] = float(R["T_obs_yr"])
        out["runs"][run] = dict(T_obs_yr=T_obs[run], vt_gpc3yr=vt.tolist(),
                                vt_detframe_gpc3yr=R["vt_comoving_gpc3yr"],
                                coverage_comoving=R["coverage_comoving"],
                                c_median=R["c_median"])
    # ---- N_eff >= 300 support rule on SOURCE-frame bins (2026-08-15) ----
    # The bank's detector-frame ceiling is Mtot=400: a source-frame bin
    # 330-400 at the O3 truncated horizons (z~0.4-0.5) needs M_det 460-600,
    # absent from the bank, so the rebin drains that bin to its low-z sliver
    # (kinematic draining, not a sensitivity cliff). Same rule as
    # fig_fourepoch_ratio.py: mask any point whose SUPPORT N_eff
    # = (sum w)^2/sum w^2 (w = cohort-normalized w0*kept, pre-detection) < 300;
    # a summed curve is masked wherever ANY component run is masked.
    NEFF_MIN = 300.0
    out["neff_srcframe"] = {}
    for run in RUNS:
        ri = np.load(f"{HERE}/relabel_inj_{run.lower()}.npz")
        zi = np.load(f"{HERE}/inj_scored_{run.lower()}.npz")
        w0, det = zi["w0"], zi["det_frac"]
        db, sb, kept = ri["det_bin"], ri["src_bin"], ri["kept"]
        W = np.zeros_like(w0)
        for b in range(len(edges) - 1):
            sel = db == b
            if sel.sum():
                W[sel] = w0[sel] / w0[sel].sum()
        # SUPPORT N_eff (population weights before detection) is the masking
        # quantity - the analogue of the pipeline-side reweighted-injection
        # N_eff in fig_fourepoch_ratio.py; the detection-weighted
        # (contribution) N_eff is stored alongside for transparency only.
        def _neff(c):
            r = []
            for k in range(len(edges) - 1):
                ck = c[sb == k]
                r.append(float(ck.sum() ** 2 / (ck ** 2).sum()) if ck.sum() > 0 else 0.0)
            return r
        neff = _neff(W * kept)
        out["neff_srcframe"][run] = dict(support=neff, contribution=_neff(W * kept * det),
                                         n_inj=[int((sb == k).sum()) for k in range(len(edges) - 1)])
        vt = np.array(out["runs"][run]["vt_gpc3yr"], float)
        masked = np.array(neff) < NEFF_MIN
        out["runs"][run]["vt_gpc3yr_unmasked"] = vt.tolist()
        out["runs"][run]["neff_masked_bins"] = [int(k) for k in np.where(masked)[0]]
        vt[masked] = np.nan
        out["runs"][run]["vt_gpc3yr"] = vt.tolist()
        print(f"[{run}] src-frame N_eff = {np.round(neff, 0).astype(int).tolist()} "
              f"-> masked bins {out['runs'][run]['neff_masked_bins']}")

    def nsum(runs):
        arr = np.array([out["runs"][r]["vt_gpc3yr"] for r in runs], float)
        s = np.nansum(arr, axis=0)
        # any component masked (NaN while unmasked value exists) -> point masked
        anymask = np.any([np.isnan(out["runs"][r]["vt_gpc3yr"]) &
                          ~np.isnan(np.array(out["runs"][r]["vt_gpc3yr_unmasked"], float))
                          for r in runs], axis=0)
        return np.where(np.all(np.isnan(arr), axis=0) | anymask, np.nan, s)
    vt_o3, vt_tot = nsum(("O3a", "O3b")), nsum(RUNS)
    T_o3 = T_obs["O3a"] + T_obs["O3b"]
    out.update(vt_o3_gpc3yr=vt_o3.tolist(), vt_total_gpc3yr=vt_tot.tolist(),
               T_o3_analyzed_yr=T_o3, T_total_analyzed_yr=float(sum(T_obs.values())),
               lvk_imbh=dict(mtot=LVK_M, **LVK),
               bbh_band_mtot=MTOT_B, bbh_band_v_gpc3=PB,
               caveats=["MADGRAV comoving relabeled VT (FlatLCDM 67.9/0.3065, V_max=int dVc/(1+z)); float64 masks; SOURCE-frame Mtot rebin (below-20 sink tabulated in vt_relabel_comoving.json)",
                        "O4: as-run found/missed outcomes with labels/horizons corrected to release run-median PSDs (relabel, not rescan); truncated as-run balls -> O4 curve is an as-run ABSOLUTE comoving VT, no longer 'relative'",
                        "MADGRAV population = injected 50/50 stellar/UM mixed-q; LVK IMBH = q=1 non-spinning",
                        "FAR mismatch: MADGRAV blind FAR<1/yr (AND UL90<1/yr) vs LVK IMBH IFAR=100 yr (100x stricter)",
                        "BBH band = sensitive volume x MADGRAV wall-clock analyzed O3 time -> compare with the O3 curve",
                        "T = wall-clock analyzed coincident HL livetime (registered 2026-08-11); ANALYZED_FRAC=0.5 is a FAR-denominator convention only",
                        "per-pipeline O4 VT at matched FAR: see figures/vt_fourepoch (GWTC-5.0 injection release)",
                        "N_eff>=300 rule on source-frame bins (support N_eff, pre-detection population weights); bank M_det ceiling 400 -> src bin 330-400 kinematically drained at O3 horizons (O3a N_eff 125): masked on O3 and total curves; 260-330 is a lower bound near the bank edge"])
    json.dump(out, open(f"{HERE}/vt_compare_pipelines{SUF}.json", "w"), indent=1)
    for run in RUNS:
        with np.printoptions(precision=3):
            print(f"[{run}] relabeled comoving VT[Gpc3yr] (src-frame) = {np.array(out['runs'][run]['vt_gpc3yr'])}")
    return out


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                         "font.family": "DejaVu Sans"})
    edges = np.array(out["mass_edges"])
    mids = 0.5 * (edges[1:] + edges[:-1])
    vt_o3 = np.array(out["vt_o3_gpc3yr"], float)
    vt_tot = np.array(out["vt_total_gpc3yr"], float)
    T_o3 = out["T_o3_analyzed_yr"]
    mtot_b = np.array(out["bbh_band_mtot"], float)
    pb = np.array(list(out["bbh_band_v_gpc3"].values()), float)
    bmin, bmax = pb.min(0) * T_o3, pb.max(0) * T_o3
    lvk_m = np.array(out["lvk_imbh"]["mtot"], float)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(mtot_b, bmin, bmax, color="0.55", alpha=0.25, lw=0,
                    label=r"O3 BBH searches $\times T_{\rm O3}^{\rm analyzed}$"
                          "\n(6 pipelines, FAR=1/yr, equal-mass;\ncompare with the O3 curve)")
    STY = {"PyCBC": dict(c="#D55E00", m="^"), "cWB": dict(c="#009E73", m="s"),
           "GstLAL": dict(c="#CC79A7", m="D")}
    for name in ("PyCBC", "cWB", "GstLAL"):
        st = STY[name]
        ax.plot(lvk_m, out["lvk_imbh"][name], ls="--", lw=1.4, color=st["c"],
                marker=st["m"], ms=6,
                label=f"LVK O3 IMBH search: {name}\n(IFAR=100 yr, q=1)" if name == "PyCBC"
                      else f"LVK O3 IMBH search: {name}")
    ax.plot(mids, vt_o3, color="#0072B2", lw=2.4, marker="o", ms=5.5,
            label="MADGRAV O3 (O3a in-sample + O3b blind, FAR<1/yr)")
    ax.plot(mids, vt_tot, color="0.25", lw=1.6, marker="", ls="-",
            label=r"MADGRAV O3+O4 total (O4 relabeled)")
    ax.set_xlim(20, 400)
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm tot}$ (source frame) [$M_\odot$]")
    ax.set_ylabel(r"$\langle VT\rangle$ [Gpc$^3$ yr, comoving]")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(frameon=False, fontsize=7.8, loc="lower right", ncol=1)
    ax.set_title("Sensitive volume–time vs total mass — MADGRAV vs O3 pipelines",
                 fontsize=10.5)
    # Footnote suppressed by default (2026-08-15, decision: the caption lives
    # in the paper); pass --footnote to restore it for standalone use.
    if "--footnote" in sys.argv:
      fig.text(0.5, -0.06,
             r"MADGRAV: comoving relabeled $\langle VT\rangle$, source-frame $M_{\rm tot}$, mixed-$q$ 50/50 stellar/UM population,"
             r" wall-clock analyzed HL time (O3 %.2f yr; O3+O4 %.2f yr — band scaled to the O3 time only);"
             "\n$^\\dagger$O4: as-run outcomes, injection labels/horizons corrected to release run-median PSDs"
             " (relabel, not rescan). LVK IMBH: full-O3 HLV, IFAR=100 yr (100$\\times$ stricter than"
             " MADGRAV's 1/yr) — compare shape, not height. Per-pipeline O4 VT: see the four-epoch ratio figure."
             "\nMADGRAV points with injection-support $N_{\\rm eff}<300$ are not plotted (330–400: source-frame bin drained by the bank's"
             r" $M_{\rm det}\leq 400\,M_\odot$ ceiling at the O3 horizons, $z\sim0.4$–0.5); 260–330 is a lower bound near the bank edge."
             % (T_o3, out["T_total_analyzed_yr"]),
             ha="center", fontsize=7, color="dimgrey")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/vt_compare_o3o4{SUF}.{ext}", dpi=200, bbox_inches="tight")
    print(f"[vt_compare] O3 comoving peak = {np.nanmax(vt_o3):.1f} Gpc3yr @ "
          f"Mtot~{mids[np.nanargmax(vt_o3)]:.0f}; total peak = {np.nanmax(vt_tot):.1f}")
    print(f"[vt_compare] -> {FIGDIR}/vt_compare_o3o4{SUF}.pdf/.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        out = json.load(open(f"{HERE}/vt_compare_pipelines{SUF}.json"))
    elif "--legacy-asrun" in sys.argv:
        out = compute()          # pre-2026-08-15 path; provenance only
        json.dump(out, open(f"{HERE}/vt_compare_pipelines_legacy.json", "w"), indent=1)
        print("[vt_compare] --legacy-asrun: json written to *_legacy.json; figure NOT regenerated")
        sys.exit(0)
    else:
        out = from_relabel()
    make_figure(out)
