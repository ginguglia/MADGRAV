#!/usr/bin/env python
"""Blind-search sensitive volume-time (VT) vs total mass — PROXY figure.

Combines, per run:
  eff(m)  : volume-averaged detection efficiency of the blind search at the
            paper's detection criterion (trials-corrected best_far < 1/yr AND
            UL90 < 1/yr), from the injection campaigns pushed through the
            frozen per-arm FAR counting (inj_scored_<run>.npz, built by
            pastro_final.py; det_frac = mean over the 47 empirical cnn pairs).
            Injections are importance-weighted rho^-4 (Euclidean volume) on
            the SNR grid [8,25].
  V8(m)   : per-source Euclidean horizon volume at network SNR RHO_TH (=grid floor 5) from the
            projected banks (stellar + ultramassive) against the run's
            reference ASD: D = d_ref * rho_net(d_ref) / RHO_TH, V = 4pi/3 D^3.
  VT(m)   = sum_i w_i det_i V_pop(i)(m) / sum_i w_i * T_obs -- each injection paired
            with ITS OWN bank population horizon (stellar vs UM differ in
            approximant/f_min).  T_obs = WALL-CLOCK analyzed coincident
            livetime (exposure; the ANALYZED_FRAC=0.5 trials convention
            applies to FAR denominators only, decision 2026-08-11).

PROXY caveats (printed + stored in the json):
  * Euclidean volumes, no cosmology; in-band statement only.
  * signals below network SNR RHO_TH (5) counted as undetected (conservative).
  * O4a reference ASD is event-clustered/low-frequency-biased -> relative,
    not absolute, for O4a.
  * O3b borrows the O3a injection set scored against the O3b background.

Run: MADGRAV venv python vt_search.py
Outputs: vt_search.json + figures/vt_search/vt_vs_mass_search.{pdf,png}
"""
import os, sys, json
import numpy as np

MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
HERE = f"{MG}/search_mode/pastro_final"
FIGDIR = f"{MG}/figures/vt_search"
sys.path.insert(0, f"{MG}/improved")
import improved_pipeline as ip
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


RUNS = ["O3a", "O3b", "O4a", "O4b"]
RHO_TH = 5.0                       # injection-grid support floor (5-7 extension) = detectability threshold
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])
MIDS = 0.5 * (MASS_EDGES[1:] + MASS_EDGES[:-1])
YR = 3.1557e7
STYLE = {"O3a": dict(c="#0072B2", m="o"), "O3b": dict(c="#009E73", m="s"),
         "O4a": dict(c="#E69F00", m="D"), "O4b": dict(c="#D55E00", m="^")}

BANKS = [f"{MG}/data/o1_o3_signal_bank_projected_2s_x10", f"{MG}/data/ultramassive_bank"]


def horizons(run):
    """Per-mass-bin mean Euclidean horizon volume (Mpc^3, net SNR RHO_TH)
    against the run's reference ASD, SPLIT BY BANK POPULATION so efficiency
    and volume are paired on the same sub-population (the stellar and UM banks
    have different approximants/f_min and very different horizons at the same
    Mtot; pooling them mismatches the injected mix)."""
    asd = {d: ip.load_detector_asd_o1(f"{MG}/data/{run.lower()}_search_prep", d)
           for d in ("H1", "L1")}
    v_m = {}
    for tag, bdir in (("sig", BANKS[0]), ("um", BANKS[1])):
        b = ip.load_o1_signal_bank(bdir)
        mt_all, v_all = [], []
        for i in range(len(b["H1"])):
            dref, mt = float(b["distance_mpc"][i]), float(b["total_mass"][i])
            if not (np.isfinite(dref) and np.isfinite(mt)):
                continue
            wH = np.asarray(b["H1"][i], np.float32); wL = np.asarray(b["L1"][i], np.float32)
            if (wH ** 2).sum() <= 0 or (wL ** 2).sum() <= 0:
                continue
            rho = float(np.hypot(ip.compute_optimal_snr(wH, asd["H1"]),
                                 ip.compute_optimal_snr(wL, asd["L1"])))
            Dh = dref * rho / RHO_TH
            mt_all.append(mt); v_all.append((4 / 3) * np.pi * Dh ** 3)
        mt_all = np.array(mt_all); v_all = np.array(v_all)
        bins = np.digitize(mt_all, MASS_EDGES) - 1
        vm = np.full(len(MIDS), np.nan)
        for k in range(len(MIDS)):
            sel = bins == k
            if sel.sum():
                vm[k] = v_all[sel].mean()
        v_m[tag] = vm
    return v_m


def load_is_um(run):
    """Rebuild the per-injection is_um flag in inj_scored order (pastro_final
    concatenates the raw npz files sorted per directory, one fold each)."""
    import glob
    SM = f"{MG}/search_mode"
    dirs = {"O3a": [f"{SC}/inj_out_o3a_56", f"{SM}/inj_out_o3a_lowsnr"],
            "O3b": [f"{SM}/inj_out_o3b", f"{SM}/inj_out_o3b_lowsnr"],
            "O4a": [f"{SM}/inj_out_o4a", f"{SM}/inj_out_o4a_lowsnr"],
            "O4b": [f"{SM}/inj_out_o4b", f"{SM}/inj_out_o4b_lowsnr"]}[run]
    parts = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*_inj.npz")):
            parts.append(np.load(f)["is_um"].astype(bool))
    return np.concatenate(parts)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    T_obs = {}
    for run in RUNS:
        segj = json.load(open(f"{SC}/{run.lower()}_full_coincident.json"))
        names = set(np.load(f"{SC}/search_out_{run.lower()}_far"
                            f"{'_f40' if run in ('O3a', 'O3b') else ''}/bg_cache_{run.lower()}.npz")
                    ["seg_names"])
        # WALL-CLOCK exposure (decision 2026-08-11): a signal at any time in a
        # segment is detectable (1-s stride, overlapping 2-s crops; O4 eff~1 is
        # position-independent), so VT uses the full coincident livetime. The
        # ANALYZED_FRAC=0.5 trials convention stays in FAR denominators only.
        T_obs[run] = sum(s[2] for s in segj["segments"] if s[3] in names) / YR

    out = {"mass_edges": MASS_EDGES.tolist(), "runs": {}}
    for run in RUNS:
        z = np.load(f"{HERE}/inj_scored_{run.lower()}.npz")
        w = z["w0"]; d = z["det_frac"]; mt = z["mtot"]
        um = load_is_um(run)
        assert len(um) == len(mt), f"{run}: is_um order mismatch ({len(um)} vs {len(mt)})"
        bins = np.digitize(mt, MASS_EDGES) - 1
        v_m = horizons(run)
        eff = np.full(len(MIDS), np.nan)
        vt = np.full(len(MIDS), np.nan)
        n_inj = np.zeros(len(MIDS), int)
        for k in range(len(MIDS)):
            sel = bins == k
            n_inj[k] = sel.sum()
            if not sel.sum():
                continue
            eff[k] = float((d[sel] * w[sel]).sum() / w[sel].sum())
            # per-injection pairing: each injection's detection probability
            # multiplies ITS OWN population's horizon volume in this bin
            vi = np.where(um[sel], v_m["um"][k], v_m["sig"][k])
            if np.any(np.isnan(vi)):        # population without bank support in bin
                vi = np.where(np.isnan(vi), np.nanmean(vi), vi)
            vt[k] = float((d[sel] * w[sel] * vi).sum() / w[sel].sum()) \
                * T_obs[run] * 1e-9          # Mpc^3 yr -> Gpc^3 yr
        out["runs"][run] = dict(T_obs_yr=T_obs[run], eff=eff.tolist(),
                                v_sig_mpc3=v_m["sig"].tolist(), v_um_mpc3=v_m["um"].tolist(),
                                um_frac=[float(um[bins == k].mean()) if (bins == k).sum() else None
                                         for k in range(len(MIDS))],
                                vt_gpc3yr=vt.tolist(), n_inj=n_inj.tolist())
        with np.printoptions(precision=3, suppress=False):
            print(f"[{run}] T_obs={T_obs[run]:.4f} yr  eff={eff}  VT[Gpc3yr]={vt}")
    tot = np.nansum([out["runs"][r]["vt_gpc3yr"] for r in RUNS], axis=0)
    # bins with no injection support in ANY run stay NaN, not 0
    allnan = np.all([np.isnan(out["runs"][r]["vt_gpc3yr"]) for r in RUNS], axis=0)
    tot = np.where(allnan, np.nan, tot)
    out["total_vt_gpc3yr"] = tot.tolist()
    out["caveats"] = ["Euclidean, in-band proxy (no cosmology)",
                      "SNR<5 counted undetected (conservative)",
                      "O4a/O4b reference ASD low-frequency-biased -> relative for O4",
                      "population mix = injected 50/50 stellar/UM draw (explicit prior)"]
    json.dump(out, open(f"{HERE}/vt_search.json", "w"), indent=1)

    # ---- figure: two single-axis panels (efficiency | VT) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                         "font.family": "DejaVu Sans"})
    fig, (axE, axV) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for run in RUNS:
        st = STYLE[run]
        eff = np.array(out["runs"][run]["eff"], float)
        vt = np.array(out["runs"][run]["vt_gpc3yr"], float)
        axE.plot(MIDS, eff, color=st["c"], marker=st["m"], ms=4.5, lw=1.6,
                 label=run, clip_on=False)
        axV.plot(MIDS, vt, color=st["c"], marker=st["m"], ms=4.5, lw=1.6, label=run)
    axV.plot(MIDS, tot, color="0.25", lw=2.2, marker="", label="O3+O4 total")
    axE.set_xlabel(r"$M_{\rm tot}$ [$M_\odot$]"); axE.set_ylabel("volume-averaged efficiency")
    axE.set_ylim(0, 1.02); axE.legend(frameon=False, fontsize=9)
    axV.set_xlabel(r"$M_{\rm tot}$ [$M_\odot$]")
    axV.set_ylabel(r"$\langle VT\rangle$ [Gpc$^3$ yr]")
    axV.set_yscale("log"); axV.legend(frameon=False, fontsize=9)
    for ax in (axE, axV):
        ax.grid(alpha=0.25, lw=0.5); ax.set_xlim(MASS_EDGES[0], MASS_EDGES[-1])
    axE.set_title("Blind-search efficiency (FAR$<1$/yr)", fontsize=10)
    axV.set_title("Proxy sensitive volume-time", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/vt_vs_mass_search.{ext}", dpi=200, bbox_inches="tight")
    print(f"[vt] -> {HERE}/vt_search.json ; {FIGDIR}/vt_vs_mass_search.pdf/.png")

if __name__ == "__main__":
    main()
