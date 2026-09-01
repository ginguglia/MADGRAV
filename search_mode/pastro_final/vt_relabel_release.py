import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

#!/usr/bin/env python
"""STEP 4: relabel-not-rescan VT correction under release run-median PSDs.

Decision (2026-08-12): the biased as-run reference ASDs entered the
detection path, but background AND injections were whitened identically, so
found/missed outcomes and FARs STAND. What changes is the physical meaning of
each injection's SNR/distance label. This script:

  1. computes, per bank entry (both banks), the network optimal SNR against
     (a) the as-run reference ASD and (b) the release run-median PSD of each
     run -> per-entry ratio c = rho_release / rho_asrun;
  2. maps every injection to its bank entry by exact float total-mass match
     (inject.py stored mtot = float(MT[k]); sig bank 102159/102400 unique
     mtot, dup<=3, UM 2000/2000 - duplicate groups share intrinsic params so
     c is averaged over the <=3 matches);
  3. relabels: rho_phys = net_snr * c (physical SNR under release PSD);
  4. rebuilds VT with outcomes unchanged. Estimator (per bin, per injected
     population, T = wall-clock exposure):

       VT = T * sum_kept[ w0 * c^-3 * det * V_rel(pop,bin) ] / sum_all[ w0 ]

     kept = {rho_phys >= RHO_TH}; V_rel = release-PSD horizon volume at
     physical network SNR RHO_TH. The c^-3 factor is the Jacobian of the
     uniform-nominal -> Euclidean-population importance weight under the
     relabel; with the mask the estimator is exact for c >= 1 (uncovered
     faint shell [RHO_TH, RHO_TH*c) counted UNDETECTED - conservative, the
     as-run search never probed it) and consistent for c < 1 (sub-threshold
     relabels fall outside the population). Since c^-3 * V_rel = V_asrun per
     entry, the corrected VT reduces to the as-run VT when c is bin-constant:
     the relabel puts the SAME search on an absolute (release-PSD) footing,
     it does not manufacture volume.

  Per-bin diagnostics: label-shift stats (median c, p16/p84 - THE O3
  deliverable: if the O3 refs were fine, c ~= 1 shows it), coverage c^-3
  (fraction of the release-accessible volume the as-run search probed; its
  inverse bounds the rescan upside), eff on the covered ball vs eff on the
  full release ball, VT shift vs vt_search.json.

Gates (abort on failure):
  G1 vectorized rho == ip.compute_optimal_snr on 20 random entries (<1e-6);
  G2 injection raw-concat order == inj_scored order (mtot AND net_snr exact);
  G3 mtot->entry match rate = 100%;
  G4 as-run bin-mean horizon volumes reproduce vt_search.json UNDER ITS OWN
     float32 ENERGY GUARD (<0.1%) - end-to-end validation vs the stored file.

FLOAT32-UNDERFLOW FINDING (2026-08-12, this campaign): vt_search.horizons()
and vt_compare_pipelines cast waveforms to float32 before their zero-power
guard; quiet/distant templates (strain ~1e-23, squares ~1e-46 < float32 min
subnormal) underflow to 0 and are silently dropped from the volume means -
sig bank 11776/102400 (34% in bin 20-40 down to 2% at 160-200; UM ~1%).
inject.py computes s0 in float64, so the injected population INCLUDES those
entries: the stored bin-mean volumes average a louder subset than the
population they are paired with -> low-mass as-run VT biased HIGH (bin-0
O4a +43%); bins above Mtot~160 (incl. the UM headline range) <~2%. This
script uses float64 masks for the actual estimator and reports the
decomposition per bin: vt_old (stored, biased) -> vt_asrun_fixed (float64
volumes, old labels) -> vt (release relabel).

Variant policy (design decision 2026-08-12): release-ABSOLUTE is the quotable
variant everywhere; mid-band-matched is copied into the JSON as a robustness
cross-check only (not_for_figures=true) - it rescales invalid refs and
inherits their shape.

Run: madgrav-venv python vt_relabel_release.py   (env SM_RUNS to restrict)
Output: vt_relabel_release.json (+ per-run tables on stdout for the mail)
"""
import glob
import json
import os
import sys

import numpy as np

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
    """Replicates ip.load_detector_asd_o1 (incl. flooring), selectable file."""
    tag = "reference_psd_release" if release else "reference_psd"
    z = np.load(f"{prep_dir}/{tag}_{det}.npz")
    psd = z["psd"].astype(np.float64)
    pos = psd[np.isfinite(psd) & (psd > 0.0)]
    floor = float(np.median(pos) * 1e-10)
    return z["freq"].astype(np.float64), np.sqrt(np.maximum(psd, floor))


def main():
    vt_old = json.load(open(f"{HERE}/vt_search.json"))

    # ---- banks: load once; precompute |hf|^2 once per (bank, det) ----
    banks = {}
    for tag, bdir in BANKS.items():
        b = ip.load_o1_signal_bank(bdir)
        mt = np.asarray(b["total_mass"], float)
        dref = np.asarray(b["distance_mpc"], float)
        L = len(np.asarray(b["H1"][0]))
        freqs = np.fft.rfftfreq(L, d=1.0 / FS)
        df = float(freqs[1] - freqs[0])
        hf2, energy, energy32 = {}, {}, {}
        for det in ("H1", "L1"):
            n = len(b[det])
            H = np.empty((n, len(freqs)))
            E = np.empty(n)
            E32 = np.empty(n, np.float32)
            B = 4096
            for i0 in range(0, n, B):
                blk = np.stack([np.asarray(w, np.float64)
                                for w in b[det][i0:i0 + B]])
                H[i0:i0 + B] = np.abs(np.fft.rfft(blk, axis=1) * (1.0 / FS)) ** 2
                E[i0:i0 + B] = (blk ** 2).sum(1)
                E32[i0:i0 + B] = (blk.astype(np.float32) ** 2).sum(1)
            hf2[det] = H
            energy[det] = E
            energy32[det] = E32
        ok = (np.isfinite(mt) & np.isfinite(dref)
              & (energy["H1"] > 0) & (energy["L1"] > 0))
        # vt_search/vt_compare's historical guard: float32 squares underflow
        keep32 = ok & (energy32["H1"] > 0) & (energy32["L1"] > 0)
        groups = {}
        for i in np.flatnonzero(ok):
            groups.setdefault(float(mt[i]), []).append(i)
        banks[tag] = dict(b=b, mt=mt, dref=dref, ok=ok, keep32=keep32,
                          groups=groups, bins=np.digitize(mt, MASS_EDGES) - 1,
                          hf2=hf2, freqs=freqs, df=df)
        print(f"[bank {tag}] N={len(mt)} ok={int(ok.sum())} "
              f"float32-guard drops={int((ok & ~keep32).sum())} "
              f"groups={len(groups)}", flush=True)

    def rho_net(tag, asdH, asdL):
        """Per-entry network optimal SNR from cached |hf|^2; exact
        compute_optimal_snr math (psd interp left/right inf, f>=20, *4*df)."""
        bk = banks[tag]
        acc = np.zeros(len(bk["mt"]))
        for det, (f_, a_) in (("H1", asdH), ("L1", asdL)):
            psd = np.interp(bk["freqs"], f_, a_ ** 2,
                            left=np.inf, right=np.inf)
            valid = np.isfinite(psd) & (psd > 0.0) & (bk["freqs"] >= 20.0)
            inv = np.zeros_like(psd)
            inv[valid] = 1.0 / psd[valid]
            acc += 4.0 * (bk["hf2"][det] @ inv) * bk["df"]
        return np.sqrt(np.maximum(acc, 0.0))

    out = {"mass_edges": MASS_EDGES.tolist(), "rho_th": RHO_TH,
           "variant_policy": "release-absolute quotable; mid-band-matched = "
                             "robustness only (not_for_figures)",
           "runs": {}}
    for run in RUNS:
        prep = f"{MG}/data/{run.lower()}_search_prep"
        asd = {key: (load_asd(prep, "H1", rel), load_asd(prep, "L1", rel))
               for key, rel in (("ref", False), ("rel", True))}
        rho = {key: {tag: rho_net(tag, *asd[key]) for tag in banks}
               for key in asd}

        # G1: exact agreement with the scalar reference implementation
        rng = np.random.default_rng(1)
        for key in ("ref", "rel"):
            (fH, aH), (fL, aL) = asd[key]
            fsH = FrequencySeries(aH, f0=float(fH[0]), df=float(fH[1] - fH[0]))
            fsL = FrequencySeries(aL, f0=float(fL[0]), df=float(fL[1] - fL[0]))
            for tag in banks:
                b = banks[tag]["b"]
                for i in rng.integers(0, len(banks[tag]["mt"]), 10):
                    refv = np.hypot(
                        ip.compute_optimal_snr(np.asarray(b["H1"][i], np.float64), fsH),
                        ip.compute_optimal_snr(np.asarray(b["L1"][i], np.float64), fsL))
                    assert abs(refv - rho[key][tag][i]) <= 1e-6 * max(refv, 1e-30), \
                        f"G1 FAIL {run}/{key}/{tag} entry {i}: {refv} vs {rho[key][tag][i]}"
        print(f"[{run}] G1 pass (vectorized rho == compute_optimal_snr)", flush=True)

        c_ent, v_rel, v_run = {}, {}, {}
        for tag in banks:
            ok = banks[tag]["ok"]
            with np.errstate(divide="ignore", invalid="ignore"):
                c = rho["rel"][tag] / rho["ref"][tag]
            c[~ok] = np.nan
            c_ent[tag] = c
            dh_rel = banks[tag]["dref"] * rho["rel"][tag] / RHO_TH
            dh_run = banks[tag]["dref"] * rho["ref"][tag] / RHO_TH
            v_rel[tag] = np.where(ok, (4 / 3) * np.pi * dh_rel ** 3, np.nan)
            v_run[tag] = np.where(ok, (4 / 3) * np.pi * dh_run ** 3, np.nan)
        # persist per-entry c so downstream consumers (step-3d report) can
        # relabel NEW injections with the exact same mtot->entry->c mapping
        # without reloading waveform banks (2026-08-12, step-3d wiring)
        np.savez(f"{HERE}/relabel_c_{run.lower()}.npz",
                 mt_sig=banks["sig"]["mt"], c_sig=c_ent["sig"],
                 ok_sig=banks["sig"]["ok"],
                 mt_um=banks["um"]["mt"], c_um=c_ent["um"],
                 ok_um=banks["um"]["ok"])

        # G4: reproduce vt_search.json under ITS float32 guard (validation
        # of this pipeline end-to-end against the stored file); the actual
        # estimator below uses the float64 ok mask (underflow finding above).
        drop_frac = {}
        for tag, jkey in (("sig", "v_sig_mpc3"), ("um", "v_um_mpc3")):
            stored = np.array(vt_old["runs"][run][jkey], float)
            bins = banks[tag]["bins"]
            df_ = []
            for k in range(len(MIDS)):
                sel32 = (bins == k) & banks[tag]["keep32"]
                sel = (bins == k) & banks[tag]["ok"]
                df_.append(float((banks[tag]["ok"] & ~banks[tag]["keep32"]
                                  & (bins == k)).sum() / max(sel.sum(), 1)))
                if not sel32.sum():
                    continue
                mine = v_run[tag][sel32].mean()
                assert abs(mine - stored[k]) <= 1e-3 * abs(stored[k]), \
                    f"G4 FAIL {run} {tag} bin {k}: {mine} vs {stored[k]}"
            drop_frac[tag] = df_
        print(f"[{run}] G4 pass (float32-guard subset reproduces "
              f"vt_search.json)", flush=True)

        # ---- injections ----
        z = np.load(f"{HERE}/inj_scored_{run.lower()}.npz")
        parts = {k: [] for k in ("mtot", "net_snr", "is_um")}
        for d_ in INJ_DIRS[run]:
            for f_ in sorted(glob.glob(f"{d_}/*_inj.npz")):
                zz = np.load(f_)
                for k in parts:
                    parts[k].append(np.asarray(zz[k]))
        raw = {k: np.concatenate(v) for k, v in parts.items()}
        assert np.array_equal(raw["mtot"].astype(float), z["mtot"]), \
            f"G2 FAIL {run}: raw mtot order != inj_scored"
        assert np.array_equal(raw["net_snr"].astype(float), z["net_snr"]), \
            f"G2 FAIL {run}: raw net_snr order != inj_scored"
        um = raw["is_um"].astype(bool)
        print(f"[{run}] G2 pass ({len(um)} injections, order verified)", flush=True)

        c_inj = np.empty(len(um))
        spread_max, unmatched = 0.0, 0
        for i, (m, u) in enumerate(zip(z["mtot"], um)):
            g = banks["um" if u else "sig"]["groups"].get(float(m))
            if not g:
                unmatched += 1
                c_inj[i] = np.nan
                continue
            cs = c_ent["um" if u else "sig"][g]
            c_inj[i] = np.nanmean(cs)
            if len(cs) > 1 and np.all(np.isfinite(cs)):
                spread_max = max(spread_max, float(cs.max() / cs.min() - 1))
        assert unmatched == 0, f"G3 FAIL {run}: {unmatched} unmatched mtot"
        assert np.all(np.isfinite(c_inj)), f"G3 FAIL {run}: NaN c after match"
        print(f"[{run}] G3 pass (100% mtot->entry; max dup-group c spread "
              f"{spread_max * 100:.3f}%)", flush=True)

        w0 = z["w0"]
        det = z["det_frac"]
        rho_phys = z["net_snr"] * c_inj
        kept = rho_phys >= RHO_TH
        bins_i = np.digitize(z["mtot"], MASS_EDGES) - 1

        segj = json.load(open(f"{SC}/{run.lower()}_full_coincident.json"))
        names = set(np.load(
            f"{SC}/search_out_{run.lower()}_far"
            f"{'_f40' if run in ('O3a', 'O3b') else ''}/bg_cache_{run.lower()}.npz")["seg_names"])
        T = sum(s[2] for s in segj["segments"] if s[3] in names) / YR

        vrel_bin, vrun_bin = {}, {}
        for tag in banks:
            bb = banks[tag]["bins"]
            ok = banks[tag]["ok"]
            vrel_bin[tag] = np.array(
                [v_rel[tag][(bb == k) & ok].mean()
                 if ((bb == k) & ok).sum() else np.nan
                 for k in range(len(MIDS))])
            vrun_bin[tag] = np.array(
                [v_run[tag][(bb == k) & ok].mean()
                 if ((bb == k) & ok).sum() else np.nan
                 for k in range(len(MIDS))])

        R = dict(T_obs_yr=T, c_median=[], c_p16=[], c_p84=[], coverage=[],
                 eff_covered=[], eff_phys=[], vt_gpc3yr=[],
                 vt_asrun_fixed_gpc3yr=[], vt_old_gpc3yr=[],
                 v_rel_sig_mpc3=vrel_bin["sig"].tolist(),
                 v_rel_um_mpc3=vrel_bin["um"].tolist(),
                 v_asrun_fixed_sig_mpc3=vrun_bin["sig"].tolist(),
                 v_asrun_fixed_um_mpc3=vrun_bin["um"].tolist(),
                 float32_drop_frac_sig=drop_frac["sig"],
                 float32_drop_frac_um=drop_frac["um"],
                 n_inj=[], kept_frac=[])
        vt_old_arr = np.array(vt_old["runs"][run]["vt_gpc3yr"], float)
        rows = []
        for k in range(len(MIDS)):
            sel = bins_i == k
            R["n_inj"].append(int(sel.sum()))
            old = vt_old_arr[k]
            R["vt_old_gpc3yr"].append(None if np.isnan(old) else float(old))
            if not sel.sum():
                for key in ("c_median", "c_p16", "c_p84", "coverage",
                            "eff_covered", "eff_phys", "vt_gpc3yr",
                            "vt_asrun_fixed_gpc3yr", "kept_frac"):
                    R[key].append(None)
                continue
            c = c_inj[sel]; w = w0[sel]; d = det[sel]; kp = kept[sel]
            vi = np.where(um[sel], vrel_bin["um"][k], vrel_bin["sig"][k])
            vi_run = np.where(um[sel], vrun_bin["um"][k], vrun_bin["sig"][k])
            if np.any(np.isnan(vi)):
                vi = np.where(np.isnan(vi), np.nanmean(vi), vi)
            if np.any(np.isnan(vi_run)):
                vi_run = np.where(np.isnan(vi_run), np.nanmean(vi_run), vi_run)
            cm = float(np.median(c))
            eff_cov = float((d * w).sum() / w.sum())
            eff_ph = float((w * kp * c ** -3.0 * d).sum() / w.sum())
            vt = float((w * kp * c ** -3.0 * d * vi).sum() / w.sum() * T * 1e-9)
            vt_fix = float((w * d * vi_run).sum() / w.sum() * T * 1e-9)
            R["c_median"].append(cm)
            R["c_p16"].append(float(np.percentile(c, 16)))
            R["c_p84"].append(float(np.percentile(c, 84)))
            R["coverage"].append(float(np.median(c ** -3.0)))
            R["eff_covered"].append(eff_cov)
            R["eff_phys"].append(eff_ph)
            R["vt_gpc3yr"].append(vt)
            R["vt_asrun_fixed_gpc3yr"].append(vt_fix)
            R["kept_frac"].append(float(kp.mean()))
            shift = (vt / old - 1) * 100 if np.isfinite(old) and old > 0 else float("nan")
            rows.append(f"  {MASS_EDGES[k]:.0f}-{MASS_EDGES[k+1]:.0f}: "
                        f"c_med={cm:.3f} [{(cm - 1) * 100:+.1f}%] "
                        f"cov={cm ** -3.0:.3g} eff_cov={eff_cov:.3f} "
                        f"eff_phys={eff_ph:.3g} | VT old {old:.4g} -> "
                        f"f32fix {vt_fix:.4g} -> relabel {vt:.4g} "
                        f"({shift:+.1f}% vs old)")
        out["runs"][run] = R
        print(f"[{run}] T={T:.4f} yr  label-shift & VT table:", flush=True)
        for r_ in rows:
            print(r_, flush=True)

    try:
        s3 = json.load(open(f"{HERE}/step3_horizons_release.json"))
        out["robustness_midband_matched"] = {
            "not_for_figures": True,
            "reason": "rescales invalid refs and inherits their shape; kept "
                      "as robustness cross-check only (design decision "
                      "2026-08-12)",
            "data": s3}
    except FileNotFoundError:
        pass
    json.dump(out, open(f"{HERE}/vt_relabel_release.json", "w"), indent=1)
    print(f"[done] -> {HERE}/vt_relabel_release.json", flush=True)


if __name__ == "__main__":
    main()
