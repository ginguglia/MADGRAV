#!/usr/bin/env python
"""MADGRAV 2.2 sensitive volume-time figures (2026-10-06).

Two figures, four panels (O3a, O3b, O4a, O4b) each, source-frame total mass on x:
  vt22_abs   : absolute comoving <VT> per bin -- MADGRAV 2.2 (13 fine bins, 10-400 Msun), the adopted MADGRAV1
               pipeline (10 bins, 20-400) and the LVK pipelines at matched FAR < 1/yr (release-derived,
               reweighted to the injected population, N_eff >= 300), all on the same axis.
  vt22_ratio : <VT>_2.2 / <VT>_x on the MADGRAV1 bins, x axis cut at 250 Msun (above it the MADGRAV1 VT is ~0 and the ratio meaningless) (2.2 bins 20-30 and 30-40 summed), x = MADGRAV1 and each
               LVK pipeline; the 2.2 counterpart of MADGRAV/search_mode/pastro_final/fig_fourepoch_ratio.py.
Inputs (read-only): vt22/vt_relabel_comoving_m22.json (2.2 numerator, this campaign), MADGRAV1 relabel and
vt_pipelines_target_zc.json from the frozen MADGRAV tree. Outputs: figures_m2/vt22_{abs,ratio}.{pdf,png} + .json.
Caveat carried in the caption: the pipeline side is reweighted to the MADGRAV1 injected population (main bank law);
the 2.2 field adds the low-mass and ultramassive banks, so within-bin populations differ slightly below 40 Msun.
"""
import json, os
import numpy as np

import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M1 = MADGRAV_SCRATCH + "/MADGRAV1/search_mode/pastro_final"
REL22 = os.environ.get("VT22_JSON", MADGRAV_SCRATCH + "/madgrav2/vt22/vt_relabel_comoving_m22.json")
REL1 = f"{M1}/vt_relabel_comoving_x1cnnadopt48fveto.json"
TGT = f"{M1}/vt_pipelines_target_zc.json"
FIGDIR = os.environ.get("MG_FIGDIR", MADGRAV_SCRATCH + "/madgrav2/figures_m2")
RUNS = ["O3a", "O3b", "O4a", "O4b"]
PIPES = ["cWB", "GstLAL", "MBTA", "PyCBC-BBH", "PyCBC-broad", "PyCBC"]
LABEL = {"PyCBC-BBH": "PyCBC-BBH (O3)", "PyCBC-broad": "PyCBC-broad (O3)", "PyCBC": "PyCBC (O4)"}
PSTYLE = {"cWB": dict(c="#0072B2", m="o"), "GstLAL": dict(c="#009E73", m="s"), "MBTA": dict(c="#E69F00", m="D"),
          "PyCBC-BBH": dict(c="#D55E00", m="^"), "PyCBC-broad": dict(c="#CC79A7", m="v"), "PyCBC": dict(c="#56B4E9", m="P")}
NEFF_MIN = 300


def arr(v):
    return np.array([x if x is not None else np.nan for x in v], float)


def rebin(vals, edges_from, edges_to):
    """Sum fine bins into coarse bins; every coarse edge must be a fine edge."""
    out = np.full(len(edges_to) - 1, np.nan)
    for i, (lo, hi) in enumerate(zip(edges_to[:-1], edges_to[1:])):
        sel = (edges_from[:-1] >= lo - 1e-9) & (edges_from[1:] <= hi + 1e-9)
        if sel.any():
            out[i] = np.nansum(vals[sel])
    return out


def main():
    r22, r1, tgt = (json.load(open(f)) for f in (REL22, REL1, TGT))
    e22, e1 = np.array(r22["mass_edges"], float), np.array(r1["mass_edges"], float)
    assert np.allclose(e1, np.array(tgt["mass_edges"], float))
    mid22, mid1 = 0.5 * (e22[1:] + e22[:-1]), 0.5 * (e1[1:] + e1[:-1])
    os.makedirs(FIGDIR, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    out = {"mass_edges_22": e22.tolist(), "mass_edges_m1": e1.tolist(), "runs": {}}

    # ---------- figure 1: absolute <VT> ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True, sharey=True)
    for ax, run in zip(axes.ravel(), RUNS):
        v22 = arr(r22["runs"][run]["vt_comoving_srcframe_gpc3yr"]); v1 = arr(r1["runs"][run]["vt_comoving_srcframe_gpc3yr"])
        out["runs"][run] = {"T_obs_yr": r22["runs"][run]["T_obs_yr"], "vt22": v22.tolist(), "vt_m1": v1.tolist(), "pipelines": {}}
        ax.stairs(v22, e22, color="black", lw=2.0, label="MADGRAV 2.2", baseline=None, zorder=5)
        ax.stairs(v1, e1, color="0.45", lw=1.4, ls="--", label="MADGRAV (paper, adopted)", baseline=None, zorder=4)
        for p in PIPES:
            pd = tgt["runs"][run]["pipelines"].get(p)
            if pd is None: continue
            th = arr(pd["vt_gpc3yr"]); ne = arr(pd.get("neff") or [None] * len(th))
            th = np.where(ne >= NEFF_MIN, th, np.nan); th = np.where(th > 0, th, np.nan)
            out["runs"][run]["pipelines"][p] = th.tolist()
            st = PSTYLE[p]; ax.plot(mid1, th, color=st["c"], marker=st["m"], ms=4.5, lw=0, mew=0.8, mfc="white", label=LABEL.get(p, p), zorder=3)
        ax.set_yscale("log"); ax.set_xscale("log"); ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.set_title(f"{run}  (T = {r22['runs'][run]['T_obs_yr']:.3f} yr)", fontsize=10)
        ax.set_xlim(10, 400); ax.set_ylim(2e-3, 30)
        ax.set_xticks([10, 20, 40, 100, 200, 400]); ax.set_xticklabels(["10", "20", "40", "100", "200", "400"])
    for ax in axes[1]: ax.set_xlabel(r"$M_{\rm tot}$ (source frame) [$M_\odot$]")
    for ax in axes[:, 0]: ax.set_ylabel(r"$\langle VT\rangle$ [Gpc$^3$ yr]")
    hh, ll = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ll: hh.append(h); ll.append(l)
    fig.legend(hh, ll, frameon=False, fontsize=9, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    if os.environ.get("SM_FIG_NOTITLE") != "1":
        fig.suptitle(r"Sensitive volume-time at FAR$<1\,{\rm yr}^{-1}$: MADGRAV 2.2 vs the paper pipeline and the LVK pipelines"
                     "\n" r"(comoving, source-frame $M_{\rm tot}$, release PSDs, pipeline points need $N_{\rm eff}\geq300$)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"): fig.savefig(f"{FIGDIR}/vt22_abs.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 2: ratios on the MADGRAV1 bins ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True, sharey=True)
    for ax, run in zip(axes.ravel(), RUNS):
        v22c = rebin(arr(r22["runs"][run]["vt_comoving_srcframe_gpc3yr"]), e22, e1); v1 = arr(r1["runs"][run]["vt_comoving_srcframe_gpc3yr"])
        ax.axhline(1.0, color="0.4", lw=0.9, ls="--", zorder=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rm1 = np.where(v1 > 0, v22c / v1, np.nan)
        out["runs"][run]["vt22_on_m1_bins"] = v22c.tolist(); out["runs"][run]["ratio_vs_m1"] = rm1.tolist(); out["runs"][run]["ratio_vs_pipeline"] = {}
        ax.plot(mid1, rm1, color="black", marker="*", ms=7, lw=1.6, label="vs MADGRAV (paper)", clip_on=True, zorder=5)
        for p in PIPES:
            pd = tgt["runs"][run]["pipelines"].get(p)
            if pd is None: continue
            th = arr(pd["vt_gpc3yr"]); ne = arr(pd.get("neff") or [None] * len(th)); th = np.where(ne >= NEFF_MIN, th, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where((th > 0) & np.isfinite(v22c), v22c / th, np.nan)
            out["runs"][run]["ratio_vs_pipeline"][p] = ratio.tolist()
            st = PSTYLE[p]; ax.plot(mid1, ratio, color=st["c"], marker=st["m"], ms=4, lw=1.4, label="vs " + LABEL.get(p, p), clip_on=True)
        ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.5, which="both"); ax.set_xlim(e1[0], 250.0); ax.set_title(run, fontsize=10)
    for ax in axes[1]: ax.set_xlabel(r"$M_{\rm tot}$ (source frame) [$M_\odot$]")
    for ax in axes[:, 0]: ax.set_ylabel(r"$\langle VT\rangle_{\rm MADGRAV\,2.2}/\langle VT\rangle_{x}$")
    hh, ll = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ll: hh.append(h); ll.append(l)
    fig.legend(hh, ll, frameon=False, fontsize=9, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    if os.environ.get("SM_FIG_NOTITLE") != "1":
        fig.suptitle(r"MADGRAV 2.2 $\langle VT\rangle$ relative to the paper pipeline and to the LVK pipelines at matched FAR$<1\,{\rm yr}^{-1}$"
                     "\n" r"(our population, comoving, release PSDs, $N_{\rm eff}\geq300$; 2.2 bins 20-30 and 30-40 summed to 20-40)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"): fig.savefig(f"{FIGDIR}/vt22_ratio.{ext}", dpi=200, bbox_inches="tight")
    json.dump(out, open(f"{FIGDIR}/vt22_figures.json", "w"), indent=1)

    # pooled numbers for the mail / notebook
    print(f"{'run':4s} {'2.2 total':>9s} {'M1 total':>9s} {'ratio':>6s} | 2.2/cWB per bin (M1 bins)")
    for run in RUNS:
        t22 = np.nansum(out["runs"][run]["vt22"]); t1 = np.nansum(out["runs"][run]["vt_m1"])
        rc = out["runs"][run]["ratio_vs_pipeline"].get("cWB", [])
        print(f"{run:4s} {t22:9.3f} {t1:9.3f} {t22/t1:6.2f} | " + " ".join(f"{x:.2f}" if x == x else "  - " for x in rc))
    print(f"[fig] -> {FIGDIR}/vt22_abs.{{pdf,png}}, vt22_ratio.{{pdf,png}}, vt22_figures.json")


if __name__ == "__main__":
    main()
