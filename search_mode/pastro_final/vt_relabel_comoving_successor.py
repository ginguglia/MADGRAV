#!/usr/bin/env python
# SUCCESSOR-STATISTIC COPY (amendment 2026-08-18): identical to the accepted script except the *_successor input/output names.
"""COMOVING relabeled VT - the volumetric-equivalent numerator for any
cross-pipeline axis (audit fix, design decision 2026-08-12).

WHY: the withheld four-epoch figure put the relabel's EUCLIDEAN proxy
volumes over the pipelines' inherently COMOVING <VT> (GWTC injection
estimator lives in dVc/(1+z)); the registered 2026-08-11 rule says the
Euclidean numbers must never share an axis with published VTs (up to ~10x
apart at high mass). This script recomputes the relabeled VT with the
volume law swapped to comoving, matching vt_compare_pipelines' harmonized
convention EXACTLY: FlatLambdaCDM(H0=67.9, Om0=0.3065) - the GWTC-3
release cosmology - and V_max(<z) = int dVc/(1+z) (time dilation).

Construction (labels identical to vt_relabel_release.py; volumes swapped):
  per bank entry: D_L,rel = d_ref * rho_release / RHO_TH  (luminosity-
    distance horizon at physical network SNR 5 under release run-median
    PSDs; float64 masks - the float32 underflow guard is NOT reproduced);
  per injection (exact mtot->entry template pairing): c = rho_rel/rho_ref,
    kept = nominal*c >= RHO_TH, truncated as-run ball D_L <= D_rel/c
    (physical rho >= 5c - what the as-run search could reach);
  VT_comoving(bin) = T * sum_kept[w0 * det * Vc(D_rel/c)] / sum_all[w0]
  with Vc = V_max comoving; coverage_comoving = Vc(D_rel/c)/Vc(D_rel)
  (w0-weighted bin mean) - the comoving analogue of c^-3.

Gates: G1 vectorized rho == compute_optimal_snr (sampled); G2 raw order ==
inj_scored; G3 100% template match; G5 Vc(D) -> (4pi/3)D^3 in the
Euclidean limit (<1% at 100 Mpc).

Run: madgrav-venv python vt_relabel_comoving.py (env SM_RUNS to restrict)
Out: vt_relabel_comoving_successor.json + per-run tables on stdout.
"""
import glob
import json
import os
import sys

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
HERE = f"{MG}/search_mode/pastro_final"
sys.path.insert(0, f"{MG}/improved")
import improved_pipeline as ip
from gwpy.frequencyseries import FrequencySeries

RUNS = os.environ.get("SM_RUNS", "O3a,O3b,O4a,O4b").split(",")
RHO_TH = 5.0
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])
MIDS = 0.5 * (MASS_EDGES[1:] + MASS_EDGES[:-1])
YR = 3.1557e7
FS = 4096
BANKS = {"sig": f"{MG}/data/o1_o3_signal_bank_projected_2s_x10",
         "um": f"{MG}/data/ultramassive_bank"}
SM = f"{MG}/search_mode"
INJ_DIRS = {"O3a": [f"{SC}/inj_out_o3a_56", f"{SM}/inj_out_o3a_lowsnr"],
            "O3b": [f"{SM}/inj_out_o3b", f"{SM}/inj_out_o3b_lowsnr"],
            "O4a": [f"{SM}/inj_out_o4a", f"{SM}/inj_out_o4a_lowsnr"],
            "O4b": [f"{SM}/inj_out_o4b", f"{SM}/inj_out_o4b_lowsnr"]}


def load_asd(prep_dir, det, release):
    tag = "reference_psd_release" if release else "reference_psd"
    z = np.load(f"{prep_dir}/{tag}_{det}.npz")
    psd = z["psd"].astype(np.float64)
    pos = psd[np.isfinite(psd) & (psd > 0.0)]
    floor = float(np.median(pos) * 1e-10)
    return z["freq"].astype(np.float64), np.sqrt(np.maximum(psd, floor))


def comoving_machinery():
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=67.9, Om0=0.3065)   # vt_compare harmonized conv.
    # 10x denser than vt_compare's 6000-pt grid: same convention, needed so
    # the trapezoid integral is accurate down to the ~100 Mpc Euclidean
    # limit that gate G5 checks (vt_compare only evaluates Gpc horizons).
    zg = np.linspace(1e-4, 8.0, 60000)
    dl_g = cosmo.luminosity_distance(zg).to(u.Mpc).value
    dvdz = cosmo.differential_comoving_volume(zg).to(u.Mpc**3 / u.sr).value * 4 * np.pi
    integ = 0.5 * (dvdz[1:] / (1 + zg[1:]) + dvdz[:-1] / (1 + zg[:-1])) * np.diff(zg)
    vmaxz = np.concatenate([[0.0], np.cumsum(integ)])

    def vmax(dl_mpc):
        dl = np.asarray(dl_mpc, float)
        assert not np.any(dl > dl_g[-1]), f"dL beyond z=8 grid: {np.max(dl):.0f}"
        return np.interp(np.interp(dl, dl_g, zg), zg, vmaxz)
    # G5: low-z limit. Vc(D_L) -> (4pi/3) D_L^3 * (1+z)^-4 (D_C = D_L/(1+z)
    # cubed, plus the 1/(1+z) time dilation) - NOT the bare Euclidean value:
    # at 100 Mpc (z~0.023) the correct ratio is ~0.92.
    for dl in (10.0, 100.0):
        z = np.interp(dl, dl_g, zg)
        ref = (4 / 3) * np.pi * dl ** 3 * (1 + z) ** -4
        assert abs(vmax(dl) / ref - 1) < 0.02, \
            f"G5 FAIL at {dl} Mpc: {vmax(dl) / ref:.4f}"
    print("[comoving] G5 pass (low-z (1+z)^-4 limit at 10/100 Mpc)", flush=True)

    def z_of_dl(dl_mpc):
        return np.interp(np.asarray(dl_mpc, float), dl_g, zg)
    return vmax, z_of_dl


def main():
    vmax, z_of_dl = comoving_machinery()
    banks = {}
    for tag, bdir in BANKS.items():
        b = ip.load_o1_signal_bank(bdir)
        mt = np.asarray(b["total_mass"], float)
        dref = np.asarray(b["distance_mpc"], float)
        L = len(np.asarray(b["H1"][0]))
        freqs = np.fft.rfftfreq(L, d=1.0 / FS)
        df = float(freqs[1] - freqs[0])
        hf2, energy = {}, {}
        for det in ("H1", "L1"):
            n = len(b[det])
            H = np.empty((n, len(freqs)))
            E = np.empty(n)
            B = 4096
            for i0 in range(0, n, B):
                blk = np.stack([np.asarray(w, np.float64)
                                for w in b[det][i0:i0 + B]])
                H[i0:i0 + B] = np.abs(np.fft.rfft(blk, axis=1) * (1.0 / FS)) ** 2
                E[i0:i0 + B] = (blk ** 2).sum(1)
            hf2[det] = H
            energy[det] = E
        ok = (np.isfinite(mt) & np.isfinite(dref)
              & (energy["H1"] > 0) & (energy["L1"] > 0))
        groups = {}
        for i in np.flatnonzero(ok):
            groups.setdefault(float(mt[i]), []).append(i)
        banks[tag] = dict(b=b, mt=mt, dref=dref, ok=ok, groups=groups,
                          hf2=hf2, freqs=freqs, df=df)
        print(f"[bank {tag}] N={len(mt)} ok={int(ok.sum())}", flush=True)

    def rho_net(tag, asdH, asdL):
        bk = banks[tag]
        acc = np.zeros(len(bk["mt"]))
        for det, (f_, a_) in (("H1", asdH), ("L1", asdL)):
            psd = np.interp(bk["freqs"], f_, a_ ** 2, left=np.inf, right=np.inf)
            valid = np.isfinite(psd) & (psd > 0.0) & (bk["freqs"] >= 20.0)
            inv = np.zeros_like(psd)
            inv[valid] = 1.0 / psd[valid]
            acc += 4.0 * (bk["hf2"][det] @ inv) * bk["df"]
        return np.sqrt(np.maximum(acc, 0.0))

    out = {"mass_edges": MASS_EDGES.tolist(), "rho_th": RHO_TH,
           "cosmology": "FlatLambdaCDM(H0=67.9, Om0=0.3065), Vmax=int dVc/(1+z)",
           "runs": {}}
    for run in RUNS:
        prep = f"{MG}/data/{run.lower()}_search_prep"
        asd = {key: (load_asd(prep, "H1", rel), load_asd(prep, "L1", rel))
               for key, rel in (("ref", False), ("rel", True))}
        rho = {key: {tag: rho_net(tag, *asd[key]) for tag in banks}
               for key in asd}
        rng = np.random.default_rng(1)
        for key in ("ref", "rel"):
            (fH, aH), (fL, aL) = asd[key]
            fsH = FrequencySeries(aH, f0=float(fH[0]), df=float(fH[1] - fH[0]))
            fsL = FrequencySeries(aL, f0=float(fL[0]), df=float(fL[1] - fL[0]))
            for tag in banks:
                b = banks[tag]["b"]
                for i in rng.integers(0, len(banks[tag]["mt"]), 5):
                    refv = np.hypot(
                        ip.compute_optimal_snr(np.asarray(b["H1"][i], np.float64), fsH),
                        ip.compute_optimal_snr(np.asarray(b["L1"][i], np.float64), fsL))
                    assert abs(refv - rho[key][tag][i]) <= 1e-6 * max(refv, 1e-30), \
                        f"G1 FAIL {run}/{key}/{tag} entry {i}"
        print(f"[{run}] G1 pass", flush=True)

        c_ent, drel_ent = {}, {}
        for tag in banks:
            ok = banks[tag]["ok"]
            with np.errstate(divide="ignore", invalid="ignore"):
                c = rho["rel"][tag] / rho["ref"][tag]
            c[~ok] = np.nan
            c_ent[tag] = c
            d = banks[tag]["dref"] * rho["rel"][tag] / RHO_TH
            d[~ok] = np.nan
            drel_ent[tag] = d

        z = np.load(f"{HERE}/inj_scored_{run.lower()}_successor.npz")
        parts = {k: [] for k in ("mtot", "net_snr", "is_um")}
        for d_ in INJ_DIRS[run]:
            for f_ in sorted(glob.glob(f"{d_}/*_inj.npz")):
                zz = np.load(f_)
                for k in parts:
                    parts[k].append(np.asarray(zz[k]))
        raw = {k: np.concatenate(v) for k, v in parts.items()}
        assert np.array_equal(raw["mtot"].astype(float), z["mtot"]) and \
            np.array_equal(raw["net_snr"].astype(float), z["net_snr"]), \
            f"G2 FAIL {run}"
        um = raw["is_um"].astype(bool)
        print(f"[{run}] G2 pass ({len(um)} injections)", flush=True)

        c_inj = np.empty(len(um))
        drel_inj = np.empty(len(um))
        unmatched = 0
        for i, (m, u) in enumerate(zip(z["mtot"], um)):
            tag = "um" if u else "sig"
            g = banks[tag]["groups"].get(float(m))
            if not g:
                unmatched += 1
                c_inj[i] = drel_inj[i] = np.nan
                continue
            c_inj[i] = np.nanmean(c_ent[tag][g])
            drel_inj[i] = np.nanmean(drel_ent[tag][g])
        assert unmatched == 0 and np.all(np.isfinite(c_inj)), f"G3 FAIL {run}"
        print(f"[{run}] G3 pass", flush=True)

        w0 = z["w0"]
        det = z["det_frac"]
        kept = z["net_snr"] * c_inj >= RHO_TH
        bins_i = np.digitize(z["mtot"], MASS_EDGES) - 1
        vc_trunc = vmax(drel_inj / c_inj)     # as-run-reachable ball, comoving
        vc_rel = vmax(drel_inj)               # release-accessible ball

        segj = json.load(open(f"{SC}/{run.lower()}_full_coincident.json"))
        names = set(np.load(
            f"{SC}/search_out_{run.lower()}_far"
            f"{'_f40' if run in ('O3a', 'O3b') else ''}/bg_cache_{run.lower()}.npz")["seg_names"])
        T = sum(s[2] for s in segj["segments"] if s[3] in names) / YR

        # ---- SOURCE-FRAME REBIN (clearing condition, 2026-08-12) ----
        # Each injection has a definite implied luminosity distance from the
        # relabel layer, d_i = D_rel * RHO_TH / rho_phys (rho_phys = s*c), a
        # definite z_i, and hence a source-frame mass M_det/(1+z_i). Its
        # contribution - the w0 share of its truncated comoving ball,
        # normalized within its DETECTOR-frame bin cohort - is reassigned to
        # the source-frame bin. Volume moves strictly down-mass; the total is
        # conserved; shells falling below Mtot=20 land in a reported sink.
        with np.errstate(divide="ignore", invalid="ignore"):
            d_i = drel_inj * RHO_TH / (z["net_snr"] * c_inj)
        z_i = z_of_dl(d_i)
        msrc = z["mtot"] / (1.0 + z_i)
        src_bins = np.digitize(msrc, MASS_EDGES) - 1
        z_tr = z_of_dl(drel_inj / c_inj)      # z at the truncated horizon
        vt_src = np.zeros(len(MIDS))
        vt_sink_below = 0.0
        for b in range(len(MIDS)):
            sel_b = bins_i == b
            if not sel_b.sum():
                continue
            Wb = w0[sel_b].sum()
            contrib = (w0[sel_b] * kept[sel_b] * det[sel_b]
                       * vc_trunc[sel_b]) / Wb * T * 1e-9
            for k, v in zip(src_bins[sel_b], contrib):
                if k < 0:
                    vt_sink_below += v
                else:
                    vt_src[k] += v
        np.savez(f"{HERE}/relabel_inj_{run.lower()}_successor.npz",
                 drel=drel_inj, c=c_inj, kept=kept, z=z_i, msrc=msrc,
                 det_bin=bins_i, src_bin=src_bins)

        R = dict(T_obs_yr=T, vt_comoving_gpc3yr=[],
                 vt_comoving_srcframe_gpc3yr=vt_src.tolist(),
                 vt_srcframe_below20_sink_gpc3yr=float(vt_sink_below),
                 z_trunc_median=[], coverage_comoving=[],
                 c_median=[], eff_covered=[], n_inj=[], kept_frac=[])
        rows = []
        for k in range(len(MIDS)):
            sel = bins_i == k
            R["n_inj"].append(int(sel.sum()))
            if not sel.sum():
                for key in ("vt_comoving_gpc3yr", "coverage_comoving",
                            "c_median", "eff_covered", "kept_frac",
                            "z_trunc_median"):
                    R[key].append(None)
                continue
            w = w0[sel]; d = det[sel]; kp = kept[sel]
            vt = float((w * kp * d * vc_trunc[sel]).sum() / w.sum() * T * 1e-9)
            cov = float((w * vc_trunc[sel]).sum() / (w * vc_rel[sel]).sum())
            cm = float(np.median(c_inj[sel]))
            eff = float((d * w).sum() / w.sum())
            ztm = float(np.median(z_tr[sel]))
            R["vt_comoving_gpc3yr"].append(vt)
            R["coverage_comoving"].append(cov)
            R["c_median"].append(cm)
            R["eff_covered"].append(eff)
            R["kept_frac"].append(float(kp.mean()))
            R["z_trunc_median"].append(ztm)
            rows.append(f"  {MASS_EDGES[k]:.0f}-{MASS_EDGES[k+1]:.0f}: "
                        f"VTc={vt:.4g} -> src {vt_src[k]:.4g} Gpc3yr  "
                        f"z_tr={ztm:.3f}  cov_c={cov:.3g}  "
                        f"c_med={cm:.3f}  eff={eff:.3f}")
        out["runs"][run] = R
        print(f"[{run}] T={T:.4f} yr  COMOVING relabeled VT:", flush=True)
        for r_ in rows:
            print(r_, flush=True)

    # ---- before/after ratio report for Mtot >= 200 (clearing condition) ----
    try:
        tgt = json.load(open(f"{HERE}/vt_pipelines_target.json"))
    except FileNotFoundError:
        tgt = None
    if tgt:
        rep = {}
        print("\n[ratio report] MADGRAV/pipeline at Mtot>=200, "
              "det-frame (before) -> src-frame (after), N_eff>=300 only:",
              flush=True)
        for run in out["runs"]:
            if run not in tgt.get("runs", {}):
                continue
            for k, lab in ((7, "200-260"), (8, "260-330"), (9, "330-400")):
                before = out["runs"][run]["vt_comoving_gpc3yr"][k]
                after = out["runs"][run]["vt_comoving_srcframe_gpc3yr"][k]
                for p, pd in tgt["runs"][run]["pipelines"].items():
                    neff = (pd.get("neff") or [None] * 10)[k]
                    theirs = pd["vt_gpc3yr"][k]
                    if neff is None or neff < 300 or not theirs or theirs <= 0 \
                            or before is None:
                        continue
                    rep.setdefault(run, {}).setdefault(lab, {})[p] = dict(
                        before=before / theirs, after=after / theirs)
                    print(f"  {run} {lab} vs {p:12s}: "
                          f"{before / theirs:6.2f} -> {after / theirs:6.2f}",
                          flush=True)
        out["ratio_report_200plus"] = rep

    json.dump(out, open(f"{HERE}/vt_relabel_comoving_successor.json", "w"), indent=1)
    print(f"[done] -> {HERE}/vt_relabel_comoving_successor.json", flush=True)


if __name__ == "__main__":
    main()
